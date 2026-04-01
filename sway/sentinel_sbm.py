"""
Sentinel Survival Boosting Mechanism (PLAN_21 — Module 2)

EXPERIMENT ADD-ON — default OFF.

When a track's detection confidence drops below threshold but the track
has a strong historical record, grants a grace period where weak detections
(below normal threshold) are accepted to prevent premature track loss.

Env:
  SWAY_SENTINEL_SBM               – 0|1 (default 0)
  SWAY_SENTINEL_GRACE_MULTIPLIER  – multiplier on max_age for grace (default 3.0)
  SWAY_SENTINEL_WEAK_DET_CONF     – minimum confidence during grace (default 0.08)
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


@dataclass
class SurvivalRecord:
    """Per-track survival state."""
    track_id: int
    survival_score: float = 0.0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=100))
    in_grace: bool = False
    grace_frames_remaining: int = 0
    grace_start_frame: int = -1


class SurvivalBoostingMechanism:
    """Sentinel: prevent premature track loss for historically strong tracks."""

    def __init__(
        self,
        grace_multiplier: Optional[float] = None,
        weak_det_conf: Optional[float] = None,
        max_age: int = 200,
    ):
        self.grace_multiplier = grace_multiplier or _env_float(
            "SWAY_SENTINEL_GRACE_MULTIPLIER", 3.0
        )
        self.weak_det_conf = weak_det_conf or _env_float(
            "SWAY_SENTINEL_WEAK_DET_CONF", 0.08
        )
        self.max_age = max_age

        # Cap multiplier
        self.grace_multiplier = min(self.grace_multiplier, 5.0)

        self._records: Dict[int, SurvivalRecord] = {}

    def update_confidence(self, track_id: int, confidence: float) -> None:
        """Record a detection confidence for a track."""
        if track_id not in self._records:
            self._records[track_id] = SurvivalRecord(track_id=track_id)

        rec = self._records[track_id]
        rec.confidence_history.append(confidence)

        if len(rec.confidence_history) >= 5:
            rec.survival_score = float(np.mean(list(rec.confidence_history)))

    def should_grant_grace(
        self, track_id: int, current_conf: float, normal_conf_threshold: float
    ) -> bool:
        """Check if a track should be granted survival grace.

        Returns True if the track has a strong history but current detection
        is below threshold.
        """
        rec = self._records.get(track_id)
        if rec is None:
            return False

        if rec.in_grace:
            return True

        if current_conf < normal_conf_threshold and rec.survival_score > normal_conf_threshold * 1.5:
            return True

        return False

    def enter_grace(self, track_id: int, frame_idx: int) -> None:
        """Start grace period for a track."""
        rec = self._records.get(track_id)
        if rec is None:
            return

        grace_frames = int(self.max_age * self.grace_multiplier)
        rec.in_grace = True
        rec.grace_frames_remaining = grace_frames
        rec.grace_start_frame = frame_idx

        logger.info(
            "Sentinel: track %d entered grace period (%d frames, survival=%.3f)",
            track_id, grace_frames, rec.survival_score,
        )

    def tick_grace(self, track_id: int) -> bool:
        """Decrement grace counter. Returns True if still in grace, False if expired."""
        rec = self._records.get(track_id)
        if rec is None or not rec.in_grace:
            return False

        rec.grace_frames_remaining -= 1
        if rec.grace_frames_remaining <= 0:
            rec.in_grace = False
            logger.info("Sentinel: track %d grace expired", track_id)
            return False

        return True

    def exit_grace(self, track_id: int) -> None:
        """End grace period (track re-found)."""
        rec = self._records.get(track_id)
        if rec is not None:
            rec.in_grace = False
            rec.grace_frames_remaining = 0
            logger.info("Sentinel: track %d recovered during grace", track_id)

    def is_in_grace(self, track_id: int) -> bool:
        rec = self._records.get(track_id)
        return rec.in_grace if rec else False

    def get_weak_det_threshold(self) -> float:
        return self.weak_det_conf

    def remove_track(self, track_id: int) -> None:
        self._records.pop(track_id, None)


def is_sentinel_enabled() -> bool:
    return _env_bool("SWAY_SENTINEL_SBM", False)


# Alias expected by main.py / integration tests
SentinelSBM = SurvivalBoostingMechanism
