"""
Pose-Gated EMA Gallery Manager (PLAN_14)

Only updates a dancer's identity embeddings when the current frame is trustworthy.
  - Isolated dancer + extended pose → high alpha (aggressive update)
  - Crouched in cluster + overlapping people → alpha = 0 (frozen)

This 2-parameter mechanism prevents gallery pollution, the primary cause
of cascading identity errors.

Env:
  SWAY_REID_EMA_ALPHA_HIGH         – update rate when isolated + good pose (default 0.15)
  SWAY_REID_EMA_ALPHA_LOW          – LOCKED at 0.00 (never update in clusters)
  SWAY_REID_EMA_ISOLATION_DIST     – min bbox-height fractions to be "isolated" (default 1.5)
  SWAY_REID_EMA_POSE_QUALITY_THRESH – min mean keypoint confidence (default 0.60)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from sway.enrollment import DancerGallery
from sway.track_state import TrackState

logger = logging.getLogger(__name__)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


class PoseGatedEMA:
    """Gallery update manager with pose-quality and isolation gating."""

    def __init__(
        self,
        alpha_high: Optional[float] = None,
        alpha_low: Optional[float] = None,
        isolation_dist: Optional[float] = None,
        pose_quality_thresh: Optional[float] = None,
    ):
        self.alpha_high = alpha_high if alpha_high is not None else _env_float(
            "SWAY_REID_EMA_ALPHA_HIGH", 0.15
        )
        self.alpha_low = alpha_low if alpha_low is not None else _env_float(
            "SWAY_REID_EMA_ALPHA_LOW", 0.00
        )
        self.isolation_dist = isolation_dist if isolation_dist is not None else _env_float(
            "SWAY_REID_EMA_ISOLATION_DIST", 1.5
        )
        self.pose_quality_thresh = pose_quality_thresh if pose_quality_thresh is not None else _env_float(
            "SWAY_REID_EMA_POSE_QUALITY_THRESH", 0.60
        )

    def compute_alpha(
        self,
        dancer_bbox: np.ndarray,
        all_bboxes: List[np.ndarray],
        keypoint_confidences: np.ndarray,
        track_state: TrackState,
    ) -> float:
        """Compute EMA update rate based on isolation and pose quality.

        Args:
            dancer_bbox: (4,) xyxy of the target dancer.
            all_bboxes: list of (4,) xyxy for ALL dancers in frame.
            keypoint_confidences: (K,) confidence values for each keypoint.
            track_state: current TrackState.

        Returns:
            Alpha value [0.0, alpha_high].
        """
        if track_state in (TrackState.DORMANT, TrackState.LOST):
            return 0.0

        if track_state == TrackState.PARTIAL:
            return self.alpha_high * 0.3

        # Isolation score: min distance to other dancers / own bbox height
        isolation = self._compute_isolation(dancer_bbox, all_bboxes)

        # Pose quality: mean confidence of visible keypoints
        visible = keypoint_confidences[keypoint_confidences > 0.1]
        pose_quality = float(visible.mean()) if len(visible) > 0 else 0.0

        is_isolated = isolation > self.isolation_dist
        is_good_pose = pose_quality >= self.pose_quality_thresh

        if is_isolated and is_good_pose:
            return self.alpha_high
        else:
            return self.alpha_low

    def _compute_isolation(
        self, bbox: np.ndarray, all_bboxes: List[np.ndarray]
    ) -> float:
        """Min center-to-center distance / bbox height."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        h = max(bbox[3] - bbox[1], 1.0)

        min_dist = float("inf")
        for other in all_bboxes:
            ox = (other[0] + other[2]) / 2
            oy = (other[1] + other[3]) / 2
            dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
            if dist > 1.0:  # skip self (distance ≈ 0)
                min_dist = min(min_dist, dist)

        return min_dist / h if min_dist < float("inf") else float("inf")

    def update_gallery(
        self,
        gallery: DancerGallery,
        new_global: Optional[np.ndarray],
        new_parts: Optional[Dict[str, np.ndarray]],
        new_color: Optional[Dict[str, np.ndarray]],
        new_face: Optional[np.ndarray],
        alpha: float,
    ) -> None:
        """EMA-update gallery embeddings at the computed alpha rate.

        Gait embedding is NOT updated via EMA — it's a one-time computation.
        """
        if alpha <= 0.0:
            return

        # Global embedding
        if new_global is not None and gallery.global_embedding is not None:
            gallery.global_embedding = (
                (1 - alpha) * gallery.global_embedding + alpha * new_global
            )
            gallery.global_embedding /= np.linalg.norm(gallery.global_embedding) + 1e-8

        elif new_global is not None:
            gallery.global_embedding = new_global

        # Part embeddings
        if new_parts and gallery.part_embeddings:
            for key in new_parts:
                if key in gallery.part_embeddings:
                    gallery.part_embeddings[key] = (
                        (1 - alpha) * gallery.part_embeddings[key] + alpha * new_parts[key]
                    )
                    gallery.part_embeddings[key] /= (
                        np.linalg.norm(gallery.part_embeddings[key]) + 1e-8
                    )
                else:
                    gallery.part_embeddings[key] = new_parts[key]
        elif new_parts:
            gallery.part_embeddings = new_parts

        # Color histograms
        if new_color and gallery.color_histograms:
            for key in new_color:
                if key in gallery.color_histograms:
                    gallery.color_histograms[key] = (
                        (1 - alpha) * gallery.color_histograms[key] + alpha * new_color[key]
                    )
                else:
                    gallery.color_histograms[key] = new_color[key]
        elif new_color:
            gallery.color_histograms = new_color

        # Face: only update if new embedding is higher quality
        if new_face is not None:
            if gallery.face_embedding is None:
                gallery.face_embedding = new_face
            else:
                gallery.face_embedding = (
                    (1 - alpha) * gallery.face_embedding + alpha * new_face
                )
                gallery.face_embedding /= np.linalg.norm(gallery.face_embedding) + 1e-8
