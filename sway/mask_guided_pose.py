"""
Visibility-Masked Pose Estimation with Per-Keypoint Confidence (PLAN_17)

Upgrades pose estimation to:
  (a) Use SAM2 masks to suppress keypoints from occluding people
  (b) Per-keypoint confidence classifier (HIGH/MEDIUM/LOW/NOT_VISIBLE)

The mask gate (LOCKED) prevents chimeric skeletons: if a keypoint has high
heatmap score but falls OUTSIDE the SAM2 mask, it's downgraded to LOW.

Env:
  SWAY_POSE_MODEL                      – vitpose_large | vitpose_huge (default vitpose_large)
  SWAY_POSE_MASK_GUIDED                – 0|1 (default 1)
  SWAY_POSE_KEYPOINT_SET               – coco17 | wholebody133 (default coco17)
  SWAY_POSE_SMART_PAD                  – 0|1 (default 1, LOCKED) — smart bbox expansion before crops
  SWAY_POSE_VISIBILITY_THRESHOLD       – score below this marks joint NOT_VISIBLE (default 0.30)
  SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH  – heatmap peak for HIGH (default 0.70)
  SWAY_CONFIDENCE_HEATMAP_THRESH_MED   – heatmap peak for MEDIUM (default 0.40)
  SWAY_CONFIDENCE_TEMPORAL_WINDOW      – frames for temporal consistency (default 5)
  SWAY_CONFIDENCE_MASK_GATE            – LOCKED at 1
"""

from __future__ import annotations

import logging
import os
from collections import deque
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KeypointConfidence(IntEnum):
    NOT_VISIBLE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


class MaskGuidedPoseEstimator:
    """Pose estimation with SAM2 mask guidance and per-keypoint confidence."""

    def __init__(
        self,
        pose_estimator=None,
        heatmap_thresh_high: Optional[float] = None,
        heatmap_thresh_med: Optional[float] = None,
        temporal_window: Optional[int] = None,
    ):
        self._estimator = pose_estimator
        self.heatmap_thresh_high = heatmap_thresh_high or _env_float(
            "SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH", 0.70
        )
        self.heatmap_thresh_med = heatmap_thresh_med or _env_float(
            "SWAY_CONFIDENCE_HEATMAP_THRESH_MED", 0.40
        )
        self.temporal_window = temporal_window or _env_int(
            "SWAY_CONFIDENCE_TEMPORAL_WINDOW", 5
        )
        self.mask_gate = _env_bool("SWAY_CONFIDENCE_MASK_GATE", True)
        self.mask_guided = _env_bool("SWAY_POSE_MASK_GUIDED", True)

        # Per-track keypoint position history for temporal consistency
        self._kp_history: Dict[int, deque] = {}

    def estimate(
        self,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None,
        track_id: int = 0,
        bbox_xyxy: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose estimation with mask guidance and confidence classification.

        Args:
            crop: BGR person crop.
            mask: binary mask in crop dimensions.
            track_id: for temporal consistency tracking.
            bbox_xyxy: optional bbox for context.

        Returns:
            (keypoints, confidence_levels) where:
              keypoints: (17, 3) [x, y, heatmap_score]
              confidence_levels: (17,) KeypointConfidence values
        """
        # Apply mask guidance
        if self.mask_guided and mask is not None and mask.shape[:2] == crop.shape[:2]:
            masked_crop = crop.copy()
            masked_crop[~mask] = [124, 116, 104]  # ImageNet mean BGR
        else:
            masked_crop = crop
            mask = None

        # Run pose estimation
        keypoints = self._run_pose(masked_crop, bbox_xyxy)

        # Classify per-keypoint confidence
        confidence_levels = self._classify_confidence(
            keypoints, mask, track_id
        )

        return keypoints, confidence_levels

    def _run_pose(
        self, crop: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Run the underlying pose estimator."""
        if self._estimator is not None:
            try:
                result = self._estimator.estimate(crop, bbox=bbox_xyxy)
                if isinstance(result, np.ndarray):
                    return result
                if hasattr(result, "keypoints"):
                    return result.keypoints
            except Exception as exc:
                logger.debug("Pose estimation failed: %s", exc)

        # Fallback: return zeros (no pose detected)
        return np.zeros((17, 3), dtype=np.float32)

    def _classify_confidence(
        self,
        keypoints: np.ndarray,
        mask: Optional[np.ndarray],
        track_id: int,
    ) -> np.ndarray:
        """Classify each keypoint as HIGH, MEDIUM, LOW, or NOT_VISIBLE."""
        n_kp = keypoints.shape[0]
        levels = np.full(n_kp, KeypointConfidence.NOT_VISIBLE, dtype=np.int32)

        # Update temporal history
        if track_id not in self._kp_history:
            self._kp_history[track_id] = deque(maxlen=self.temporal_window)
        self._kp_history[track_id].append(keypoints[:, :2].copy())

        for k in range(n_kp):
            heatmap_peak = keypoints[k, 2]

            if heatmap_peak < 0.10:
                levels[k] = KeypointConfidence.NOT_VISIBLE
                continue

            # Check if keypoint is inside the mask
            mask_inside = True
            if mask is not None:
                x, y = int(keypoints[k, 0]), int(keypoints[k, 1])
                h, w = mask.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    mask_inside = bool(mask[y, x])
                else:
                    mask_inside = False

            # Temporal consistency
            temporally_consistent = self._check_temporal_consistency(
                track_id, k, keypoints[k, :2]
            )

            # Classification
            if heatmap_peak >= self.heatmap_thresh_high and mask_inside and temporally_consistent:
                levels[k] = KeypointConfidence.HIGH
            elif heatmap_peak >= self.heatmap_thresh_med and (mask_inside or temporally_consistent):
                levels[k] = KeypointConfidence.MEDIUM
            elif heatmap_peak >= 0.10:
                levels[k] = KeypointConfidence.LOW
            else:
                levels[k] = KeypointConfidence.NOT_VISIBLE

            # Mask gate (LOCKED): downgrade if outside mask
            if self.mask_gate and mask is not None and not mask_inside:
                if levels[k] > KeypointConfidence.LOW:
                    levels[k] = KeypointConfidence.LOW

        return levels

    def _check_temporal_consistency(
        self, track_id: int, kp_idx: int, position: np.ndarray
    ) -> bool:
        """Check if a keypoint position is consistent with recent history."""
        history = self._kp_history.get(track_id)
        if not history or len(history) < 2:
            return True

        positions = np.array([h[kp_idx] for h in history])
        mean_pos = positions.mean(axis=0)
        std_pos = positions.std(axis=0) + 1e-6

        dist = np.abs(position - mean_pos)
        return bool(np.all(dist < 2.0 * std_pos))

    def clear_history(self, track_id: int) -> None:
        self._kp_history.pop(track_id, None)


def pose_keypoint_set() -> str:
    """Keypoint format: 'coco17' or 'wholebody133'."""
    return os.environ.get("SWAY_POSE_KEYPOINT_SET", "coco17").strip() or "coco17"


def pose_smart_pad_enabled() -> bool:
    """Smart bbox expansion before pose crops (LOCKED ON)."""
    return _env_bool("SWAY_POSE_SMART_PAD", True)


def pose_visibility_threshold() -> float:
    """Keypoint confidence below this → NOT_VISIBLE."""
    return _env_float("SWAY_POSE_VISIBILITY_THRESHOLD", 0.30)


def should_run_pose_for_state(state) -> bool:
    """Wrapper for track_state.should_run_pose for import convenience."""
    from sway.track_state import should_run_pose
    return should_run_pose(state)
