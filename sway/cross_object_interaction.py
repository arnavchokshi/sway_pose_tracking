"""
Cross-Object Interaction Module (PLAN_05)

Collision-detection and memory-quarantine system inspired by SAM2MOT.
When two tracked masks overlap significantly, this module detects which
track is being occluded by analyzing logit score variance, then quarantines
the contaminated memory entries from SAM2's memory bank.

This directly addresses the core failure mode: ID swaps caused by the tracker
absorbing the wrong person's pixels during crossovers.

Env:
  SWAY_COI_MASK_IOU_THRESH         – pairwise mask IoU above which → collision (default 0.25)
  SWAY_COI_LOGIT_VARIANCE_WINDOW   – frames of logit history for variance (default 10)
  SWAY_COI_QUARANTINE_MODE         – "delete" or "freeze" (default "delete")
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


@dataclass
class QuarantineAction:
    """A directive to quarantine a track's memory during a collision."""
    track_id: int
    mode: str              # "delete" or "freeze"
    start_frame: int
    end_frame: Optional[int] = None  # filled when collision resolves
    partner_track_id: int = -1


@dataclass
class CollisionEvent:
    """Tracks an active collision between two tracks."""
    track_a: int
    track_b: int
    start_frame: int
    quarantined_track: Optional[int] = None
    resolved: bool = False
    end_frame: Optional[int] = None


class CrossObjectInteraction:
    """Detects mask-level collisions and quarantines corrupted memory."""

    def __init__(
        self,
        mask_iou_thresh: Optional[float] = None,
        logit_variance_window: Optional[int] = None,
        quarantine_mode: Optional[str] = None,
    ):
        self.mask_iou_thresh = mask_iou_thresh or _env_float("SWAY_COI_MASK_IOU_THRESH", 0.25)
        self.logit_variance_window = logit_variance_window or _env_int("SWAY_COI_LOGIT_VARIANCE_WINDOW", 10)
        self.quarantine_mode = quarantine_mode or _env_str("SWAY_COI_QUARANTINE_MODE", "delete")

        self.logit_history: Dict[int, deque] = {}
        self._active_collisions: List[CollisionEvent] = []
        self._quarantined_tracks: Set[int] = set()

    def update_logits(self, track_id: int, logit_score: float) -> None:
        """Append a logit score for a track (called every frame)."""
        if track_id not in self.logit_history:
            self.logit_history[track_id] = deque(maxlen=max(self.logit_variance_window * 2, 30))
        self.logit_history[track_id].append(logit_score)

    def check_collisions(
        self,
        masks: Dict[int, np.ndarray],
        logits: Dict[int, float],
        frame_idx: int,
    ) -> List[QuarantineAction]:
        """Check for mask-level collisions and determine which tracks to quarantine.

        Args:
            masks: {track_id: binary_mask} for all active tracks.
            logits: {track_id: logit_score} for all active tracks.
            frame_idx: current frame index.

        Returns:
            List of QuarantineAction directives.
        """
        # Update logit histories
        for tid, score in logits.items():
            self.update_logits(tid, score)

        actions: List[QuarantineAction] = []
        track_ids = list(masks.keys())
        n = len(track_ids)

        if n < 2:
            self._check_collision_exits(masks, frame_idx)
            return actions

        # Compute pairwise Mask IoU
        for i in range(n):
            for j in range(i + 1, n):
                tid_a = track_ids[i]
                tid_b = track_ids[j]
                mask_a = masks[tid_a]
                mask_b = masks[tid_b]

                iou = self._compute_mask_iou(mask_a, mask_b)

                if iou > self.mask_iou_thresh:
                    action = self._handle_collision(tid_a, tid_b, frame_idx)
                    if action is not None:
                        actions.append(action)

        # Check for collision exits (hysteresis)
        self._check_collision_exits(masks, frame_idx)

        return actions

    def _compute_mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        if mask_a.shape != mask_b.shape:
            # Resize to match if needed
            h = max(mask_a.shape[0], mask_b.shape[0])
            w = max(mask_a.shape[1], mask_b.shape[1])
            a = np.zeros((h, w), dtype=bool)
            b = np.zeros((h, w), dtype=bool)
            a[:mask_a.shape[0], :mask_a.shape[1]] = mask_a
            b[:mask_b.shape[0], :mask_b.shape[1]] = mask_b
            mask_a, mask_b = a, b

        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return float(intersection) / float(union) if union > 0 else 0.0

    def _handle_collision(
        self, tid_a: int, tid_b: int, frame_idx: int
    ) -> Optional[QuarantineAction]:
        """Handle a detected collision between two tracks."""
        # Check if collision already active
        for event in self._active_collisions:
            if event.resolved:
                continue
            pair = {event.track_a, event.track_b}
            if pair == {tid_a, tid_b}:
                return None  # Already being handled

        # New collision — determine which track is occluded
        var_a = self._logit_variance(tid_a)
        var_b = self._logit_variance(tid_b)

        # Higher variance = more disrupted = the one being occluded
        if var_a >= var_b:
            occluded = tid_a
            occluder = tid_b
        else:
            occluded = tid_b
            occluder = tid_a

        event = CollisionEvent(
            track_a=tid_a,
            track_b=tid_b,
            start_frame=frame_idx,
            quarantined_track=occluded,
        )
        self._active_collisions.append(event)
        self._quarantined_tracks.add(occluded)

        logger.info(
            "COI: Collision detected tracks %d↔%d at frame %d. "
            "Quarantining track %d (mode=%s, var_a=%.4f, var_b=%.4f)",
            tid_a, tid_b, frame_idx, occluded, self.quarantine_mode, var_a, var_b,
        )

        return QuarantineAction(
            track_id=occluded,
            mode=self.quarantine_mode,
            start_frame=frame_idx,
            partner_track_id=occluder,
        )

    def _logit_variance(self, track_id: int) -> float:
        """Compute logit score variance over the recent window."""
        history = self.logit_history.get(track_id)
        if not history or len(history) < 2:
            return 0.0
        window = list(history)[-self.logit_variance_window:]
        return float(np.var(window))

    def _check_collision_exits(self, masks: Dict[int, np.ndarray], frame_idx: int) -> None:
        """Detect when collisions end (hysteresis: exit at 50% of entry threshold)."""
        exit_threshold = self.mask_iou_thresh * 0.5

        for event in self._active_collisions:
            if event.resolved:
                continue

            mask_a = masks.get(event.track_a)
            mask_b = masks.get(event.track_b)

            if mask_a is None or mask_b is None:
                event.resolved = True
                event.end_frame = frame_idx
                if event.quarantined_track in self._quarantined_tracks:
                    self._quarantined_tracks.discard(event.quarantined_track)
                logger.info(
                    "COI: Collision %d↔%d ended at frame %d (track disappeared)",
                    event.track_a, event.track_b, frame_idx,
                )
                continue

            iou = self._compute_mask_iou(mask_a, mask_b)
            if iou < exit_threshold:
                event.resolved = True
                event.end_frame = frame_idx
                if event.quarantined_track in self._quarantined_tracks:
                    self._quarantined_tracks.discard(event.quarantined_track)
                logger.info(
                    "COI: Collision %d↔%d ended at frame %d (IoU=%.3f < exit=%.3f)",
                    event.track_a, event.track_b, frame_idx, iou, exit_threshold,
                )

    def apply_quarantine(self, sam2_tracker, action: QuarantineAction) -> None:
        """Apply a quarantine action to the SAM2 tracker's memory bank."""
        if action.mode == "delete":
            sam2_tracker.remove_memory_entries(
                action.track_id, action.start_frame,
                action.end_frame or action.start_frame,
            )
        elif action.mode == "freeze":
            sam2_tracker.freeze_memory(action.track_id)

    def release_quarantine(self, sam2_tracker, track_id: int) -> None:
        """Resume normal memory updates for a previously quarantined track."""
        sam2_tracker.unfreeze_memory(track_id)

    def is_quarantined(self, track_id: int) -> bool:
        return track_id in self._quarantined_tracks

    def get_active_collisions(self) -> List[CollisionEvent]:
        return [e for e in self._active_collisions if not e.resolved]

    def get_collision_history(self) -> List[CollisionEvent]:
        return list(self._active_collisions)

    def cleanup_resolved(self, max_history: int = 100) -> None:
        """Prune resolved collision events to bound memory."""
        if len(self._active_collisions) > max_history:
            self._active_collisions = [
                e for e in self._active_collisions if not e.resolved
            ] + self._active_collisions[-max_history:]
