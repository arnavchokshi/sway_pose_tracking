"""
Occlusion-Aware Track State Machine (PLAN_01)

Four-state lifecycle: ACTIVE → PARTIAL → DORMANT → LOST.
Every downstream component (pose estimation, re-ID, critique) reads track state
to decide whether to process, update galleries, or report "no data."

State transitions are driven by SAM2 mask area fraction and visible joint count.
When SAM2 is not the primary tracker, bounding-box area can substitute for mask area.

Config via environment variables (SWAY_STATE_* prefix):
  SWAY_STATE_PARTIAL_MASK_FRAC   – min mask area fraction for ACTIVE  (default 0.30)
  SWAY_STATE_PARTIAL_MIN_JOINTS  – min visible joints for ACTIVE      (default 5)
  SWAY_STATE_DORMANT_MASK_FRAC   – threshold below which → DORMANT    (default 0.05)
  SWAY_STATE_DORMANT_MAX_FRAMES  – frames in DORMANT before LOST      (default 300)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple


class TrackState(IntEnum):
    """Track visibility states, ordered by decreasing visibility."""
    ACTIVE = 3
    PARTIAL = 2
    DORMANT = 1
    LOST = 0


# ---------------------------------------------------------------------------
# Environment-driven thresholds
# ---------------------------------------------------------------------------

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


def _partial_mask_frac() -> float:
    # Future docs use SWAY_TRACK_PARTIAL_MASK_FRAC; current code uses SWAY_STATE_PARTIAL_MASK_FRAC
    v = os.environ.get("SWAY_TRACK_PARTIAL_MASK_FRAC", "").strip()
    if not v:
        v = os.environ.get("SWAY_STATE_PARTIAL_MASK_FRAC", "").strip()
    try:
        return float(v) if v else 0.30
    except ValueError:
        return 0.30


def _partial_min_joints() -> int:
    return _env_int("SWAY_STATE_PARTIAL_MIN_JOINTS", 5)


def _dormant_mask_frac() -> float:
    # Future docs use SWAY_TRACK_DORMANT_MASK_FRAC; current code uses SWAY_STATE_DORMANT_MASK_FRAC
    v = os.environ.get("SWAY_TRACK_DORMANT_MASK_FRAC", "").strip()
    if not v:
        v = os.environ.get("SWAY_STATE_DORMANT_MASK_FRAC", "").strip()
    try:
        return float(v) if v else 0.05
    except ValueError:
        return 0.05


def _dormant_max_frames() -> int:
    # Accept both explicit state lifetime and global tracker max-age alias.
    v = os.environ.get("SWAY_TRACK_MAX_AGE", "").strip()
    if v:
        try:
            return int(v)
        except ValueError:
            pass
    return _env_int("SWAY_STATE_DORMANT_MAX_FRAMES", 300)


# ---------------------------------------------------------------------------
# TrackLifecycle dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackLifecycle:
    """Per-track state machine with transition history."""

    track_id: int
    state: TrackState = TrackState.ACTIVE
    reference_mask_area: float = 0.0
    current_mask_area: float = 0.0
    frames_in_dormant: int = 0
    visible_joint_ids: List[int] = field(default_factory=list)
    last_active_frame: int = 0
    state_history: List[Tuple[int, TrackState]] = field(default_factory=list)

    # Internal: running max over first 30 frames to auto-compute reference area
    _area_samples: List[float] = field(default_factory=list, repr=False)
    _reference_locked: bool = field(default=False, repr=False)

    def record_area_sample(self, area: float, frame_idx: int) -> None:
        """Accumulate mask/bbox area samples during the first 30 frames
        to auto-compute reference_mask_area when enrollment is unavailable."""
        if self._reference_locked:
            return
        self._area_samples.append(area)
        if len(self._area_samples) >= 30 or frame_idx >= 30:
            self.reference_mask_area = max(self._area_samples) if self._area_samples else area
            self._reference_locked = True

    def set_reference_area(self, area: float) -> None:
        """Explicitly set reference area (from enrollment gallery)."""
        self.reference_mask_area = area
        self._reference_locked = True


# ---------------------------------------------------------------------------
# Pure transition logic
# ---------------------------------------------------------------------------

def update_state(
    lifecycle: TrackLifecycle,
    mask_area: Optional[float],
    num_visible_joints: int,
    frame_idx: int,
) -> TrackState:
    """Compute the next state for a track based on mask area and joint visibility.

    This is a pure function (no I/O). Side-effects on *lifecycle* are limited to
    counter updates and history appends.

    Args:
        lifecycle: the track's mutable lifecycle object.
        mask_area: pixel area of the SAM2 mask (or bbox area when masks unavailable).
                   None means the mask/detection is missing entirely.
        num_visible_joints: number of pose keypoints with confidence above threshold.
        frame_idx: current frame index (0-based).

    Returns:
        The new TrackState.
    """
    partial_frac = _partial_mask_frac()
    partial_joints = _partial_min_joints()
    dormant_frac = _dormant_mask_frac()
    dormant_max = _dormant_max_frames()

    prev_state = lifecycle.state

    # Auto-collect reference area when not yet locked
    if mask_area is not None:
        lifecycle.record_area_sample(mask_area, frame_idx)
        lifecycle.current_mask_area = mask_area

    ref = lifecycle.reference_mask_area
    if ref <= 0:
        ref = mask_area if (mask_area is not None and mask_area > 0) else 1.0

    mask_frac = (mask_area / ref) if (mask_area is not None and ref > 0) else 0.0

    # --- Transition logic ---
    if mask_frac >= partial_frac and num_visible_joints >= partial_joints:
        new_state = TrackState.ACTIVE
    elif mask_frac >= dormant_frac and num_visible_joints >= 1:
        new_state = TrackState.PARTIAL
    elif mask_frac < dormant_frac or mask_area is None:
        new_state = TrackState.DORMANT
    else:
        new_state = TrackState.DORMANT

    # Dormant counter management
    if new_state == TrackState.DORMANT:
        lifecycle.frames_in_dormant += 1
        if lifecycle.frames_in_dormant > dormant_max:
            new_state = TrackState.LOST
    else:
        lifecycle.frames_in_dormant = 0

    # Update last_active_frame
    if new_state in (TrackState.ACTIVE, TrackState.PARTIAL):
        lifecycle.last_active_frame = frame_idx

    # Record transition
    if new_state != prev_state or not lifecycle.state_history:
        lifecycle.state_history.append((frame_idx, new_state))

    lifecycle.state = new_state
    return new_state


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------

def should_run_pose(state: TrackState) -> bool:
    """True for ACTIVE and PARTIAL — pose estimation produces usable data."""
    return state in (TrackState.ACTIVE, TrackState.PARTIAL)


def should_update_gallery(state: TrackState) -> bool:
    """True for ACTIVE only.

    PARTIAL updates are handled by pose-gated EMA (PLAN_14)
    with restricted signals — not the general gallery updater.
    """
    return state == TrackState.ACTIVE


def should_generate_critique(state: TrackState) -> bool:
    """True for ACTIVE and PARTIAL — only these have reliable pose data."""
    return state in (TrackState.ACTIVE, TrackState.PARTIAL)


# ---------------------------------------------------------------------------
# Serialization helpers (for data.json output)
# ---------------------------------------------------------------------------

def state_to_str(state: TrackState) -> str:
    return state.name  # "ACTIVE", "PARTIAL", "DORMANT", "LOST"


def state_from_str(s: str) -> TrackState:
    return TrackState[s.upper()]


def lifecycle_to_dict(lc: TrackLifecycle) -> dict:
    """Serialize lifecycle for JSON output (data.json per-track field)."""
    return {
        "track_id": lc.track_id,
        "state": state_to_str(lc.state),
        "reference_mask_area": lc.reference_mask_area,
        "current_mask_area": lc.current_mask_area,
        "frames_in_dormant": lc.frames_in_dormant,
        "visible_joint_ids": lc.visible_joint_ids,
        "last_active_frame": lc.last_active_frame,
        "state_history": [(f, state_to_str(s)) for f, s in lc.state_history],
    }
