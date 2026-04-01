"""
MeMoSORT Memory-Augmented Motion Prediction (PLAN_06)

Memory-augmented Kalman filter + motion-adaptive IoU matching.
SAM2 answers "which pixels belong to this person";
MeMoSORT answers "where this person is likely to be next frame."

Together they reduce track loss when SAM2 struggles (very small/distant masks)
while SAM2 still handles heavy overlap better than box-only association.

Env:
  SWAY_MEMOSORT_MEMORY_LENGTH       – frames of position memory (default 30)
  SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA  – expansion factor for fast-moving objects (default 0.50)
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


@dataclass
class PredictedBox:
    """Output of MemoryKalmanFilter.predict()."""
    bbox_xyxy: np.ndarray  # (4,) x1,y1,x2,y2
    velocity: np.ndarray   # (2,) vx,vy center velocity
    confidence: float = 1.0


class MemoryKalmanFilter:
    """Kalman filter with memory-augmented divergence correction.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    Memory: rolling buffer of observed center positions for divergence detection.
    """

    def __init__(self, bbox_xyxy: np.ndarray, memory_length: int = 30):
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        w = bbox_xyxy[2] - bbox_xyxy[0]
        h = bbox_xyxy[3] - bbox_xyxy[1]

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # State transition
        self.F = np.eye(8, dtype=np.float64)
        self.F[:4, 4:] = np.eye(4)

        # Process noise (adapted during divergence)
        self._Q_base = np.diag([10, 10, 10, 10, 25, 25, 10, 10]).astype(np.float64)
        self.Q = self._Q_base.copy()

        # Measurement matrix (observe position + size)
        self.H = np.eye(4, 8, dtype=np.float64)

        # Measurement noise
        self.R = np.diag([10, 10, 10, 10]).astype(np.float64)

        # Covariance
        self.P = np.eye(8, dtype=np.float64) * 100.0

        # Memory buffer
        self.position_memory: deque = deque(maxlen=memory_length)
        self.position_memory.append(np.array([cx, cy]))

    def predict(self) -> PredictedBox:
        """Predict next state with memory-augmented divergence correction."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        predicted_center = self.x[:2].copy()

        # Memory-augmented correction
        if len(self.position_memory) >= 3:
            k = min(10, len(self.position_memory))
            recent_positions = np.array(list(self.position_memory))[-k:]
            memory_mean = recent_positions.mean(axis=0)
            memory_std = recent_positions.std(axis=0) + 1e-6

            err = predicted_center - memory_mean
            divergence = np.abs(err) / memory_std

            if np.any(divergence > 2.0):
                scale = 1.0 + np.linalg.norm(divergence) * 0.5
                self.Q = self._Q_base * min(scale, 10.0)
            else:
                self.Q = self._Q_base.copy()

        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
        velocity = self.x[4:6].astype(np.float32)

        return PredictedBox(bbox_xyxy=bbox, velocity=velocity)

    def update(self, bbox_xyxy: np.ndarray) -> None:
        """Update state with observed measurement."""
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        w = bbox_xyxy[2] - bbox_xyxy[0]
        h = bbox_xyxy[3] - bbox_xyxy[1]
        z = np.array([cx, cy, w, h], dtype=np.float64)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.position_memory.append(np.array([cx, cy]))


def adaptive_iou(
    box_a: np.ndarray,
    box_b: np.ndarray,
    velocity_a: np.ndarray,
    velocity_b: np.ndarray,
    alpha: float = 0.5,
) -> float:
    """Motion-adaptive IoU: expands boxes proportionally to movement speed.

    Dynamically expands the matching space when objects are moving fast
    and contracts it when stationary.
    """
    standard = _box_iou_single(box_a, box_b)

    speed = max(np.linalg.norm(velocity_a), np.linalg.norm(velocity_b))
    expansion = alpha * speed

    expanded_a = box_a.copy()
    expanded_b = box_b.copy()
    expanded_a[:2] -= expansion
    expanded_a[2:] += expansion
    expanded_b[:2] -= expansion
    expanded_b[2:] += expansion

    expanded = _box_iou_single(expanded_a, expanded_b)
    return max(standard, expanded)


def _box_iou_single(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class MeMoSORT:
    """Memory-augmented motion tracker with adaptive IoU matching."""

    def __init__(
        self,
        memory_length: Optional[int] = None,
        adaptive_iou_alpha: Optional[float] = None,
    ):
        self.memory_length = memory_length or _env_int("SWAY_MEMOSORT_MEMORY_LENGTH", 30)
        self.adaptive_iou_alpha = adaptive_iou_alpha or _env_float("SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA", 0.50)
        self._filters: Dict[int, MemoryKalmanFilter] = {}

    def init_track(self, track_id: int, bbox_xyxy: np.ndarray) -> None:
        """Initialize a new Kalman filter for a track."""
        self._filters[track_id] = MemoryKalmanFilter(bbox_xyxy, self.memory_length)

    def predict(self, track_id: int) -> Optional[PredictedBox]:
        """Predict next position for a track."""
        f = self._filters.get(track_id)
        if f is None:
            return None
        return f.predict()

    def update(self, track_id: int, bbox_xyxy: np.ndarray) -> None:
        """Update a track with observed bbox."""
        f = self._filters.get(track_id)
        if f is not None:
            f.update(bbox_xyxy)

    def predict_all(self) -> Dict[int, PredictedBox]:
        """Predict for all active tracks."""
        return {tid: self.predict(tid) for tid in self._filters if self.predict(tid) is not None}

    def match(
        self,
        predictions: Dict[int, PredictedBox],
        detection_boxes: List[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian matching with motion-adaptive IoU cost.

        Returns:
            (matches, unmatched_track_ids, unmatched_det_indices)
        """
        if not predictions or not detection_boxes:
            unmatched_tracks = list(predictions.keys()) if predictions else []
            unmatched_dets = list(range(len(detection_boxes))) if detection_boxes else []
            return [], unmatched_tracks, unmatched_dets

        track_ids = list(predictions.keys())
        n_tracks = len(track_ids)
        n_dets = len(detection_boxes)

        cost = np.zeros((n_tracks, n_dets), dtype=np.float64)
        for i, tid in enumerate(track_ids):
            pred = predictions[tid]
            for j, det_box in enumerate(detection_boxes):
                iou = adaptive_iou(
                    pred.bbox_xyxy, det_box,
                    pred.velocity, np.zeros(2, dtype=np.float32),
                    self.adaptive_iou_alpha,
                )
                cost[i, j] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 0.7:  # reject poor matches
                matches.append((track_ids[r], c))
                matched_tracks.add(track_ids[r])
                matched_dets.add(c)

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets

    def remove_track(self, track_id: int) -> None:
        self._filters.pop(track_id, None)

    def has_track(self, track_id: int) -> bool:
        return track_id in self._filters
