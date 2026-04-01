"""
UMOT Historical Backtracking Module (PLAN_21 — Module 3)

EXPERIMENT ADD-ON — default OFF.

Stores full trajectory + embeddings for every track ever seen.
When a new detection appears that doesn't match any active track,
queries the trajectory bank for historical matches to reactivate
long-dormant tracks.

Env:
  SWAY_UMOT_BACKTRACK       – 0|1 (default 0)
  SWAY_UMOT_HISTORY_LENGTH  – max frames per track in bank (default 500)
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


@dataclass
class TrajectoryEntry:
    """Single frame observation in a track's history."""
    frame: int
    x: float
    y: float
    embedding: Optional[np.ndarray] = None


@dataclass
class HistoricalTrack:
    """A complete track trajectory stored in the bank."""
    track_id: int
    dancer_id: int = -1
    entries: List[TrajectoryEntry] = field(default_factory=list)
    is_active: bool = True
    last_frame: int = 0


class HistoricalTrajectoryBank:
    """Stores and queries historical trajectories for re-identification."""

    def __init__(
        self,
        history_length: Optional[int] = None,
        match_threshold: float = 0.6,
    ):
        self.history_length = history_length or _env_int("SWAY_UMOT_HISTORY_LENGTH", 500)
        self.match_threshold = match_threshold
        self._bank: Dict[int, HistoricalTrack] = {}

    def record(
        self,
        track_id: int,
        frame: int,
        x: float,
        y: float,
        embedding: Optional[np.ndarray] = None,
        dancer_id: int = -1,
    ) -> None:
        """Record an observation for a track."""
        if track_id not in self._bank:
            self._bank[track_id] = HistoricalTrack(
                track_id=track_id, dancer_id=dancer_id
            )

        ht = self._bank[track_id]
        ht.entries.append(TrajectoryEntry(
            frame=frame, x=x, y=y, embedding=embedding
        ))
        ht.last_frame = frame
        ht.is_active = True

        # Prune old entries
        if len(ht.entries) > self.history_length:
            ht.entries = ht.entries[-self.history_length:]

    def mark_lost(self, track_id: int) -> None:
        """Mark a track as no longer active."""
        if track_id in self._bank:
            self._bank[track_id].is_active = False

    def query(
        self,
        embedding: np.ndarray,
        position: Tuple[float, float],
        current_frame: int,
        max_age_multiplier: int = 5,
        max_age: int = 200,
    ) -> Optional[int]:
        """Find a matching historical track for a new detection.

        Args:
            embedding: feature embedding of the new detection.
            position: (x, y) of the new detection.
            current_frame: current frame index.
            max_age_multiplier: only search tracks lost within N×max_age frames.
            max_age: base max_age.

        Returns:
            track_id of the best match, or None.
        """
        max_frame_gap = max_age * max_age_multiplier
        best_track = None
        best_sim = -1.0

        for tid, ht in self._bank.items():
            if ht.is_active:
                continue

            frame_gap = current_frame - ht.last_frame
            if frame_gap > max_frame_gap or frame_gap <= 0:
                continue

            # Compute similarity using most recent embedding
            recent_embs = [
                e.embedding for e in reversed(ht.entries)
                if e.embedding is not None
            ][:5]

            if not recent_embs:
                continue

            sims = [float(np.dot(embedding, e)) for e in recent_embs]
            avg_sim = np.mean(sims)

            if avg_sim > best_sim and avg_sim > self.match_threshold:
                best_sim = avg_sim
                best_track = tid

        if best_track is not None:
            logger.info(
                "UMOT: reactivating track %d (sim=%.3f, gap=%d frames)",
                best_track, best_sim, current_frame - self._bank[best_track].last_frame,
            )
            self._bank[best_track].is_active = True

        return best_track

    def prune(self, current_frame: int, max_age: int = 200, multiplier: int = 5) -> int:
        """Remove tracks older than max_age × multiplier. Returns count removed."""
        cutoff = current_frame - max_age * multiplier
        to_remove = [
            tid for tid, ht in self._bank.items()
            if not ht.is_active and ht.last_frame < cutoff
        ]
        for tid in to_remove:
            del self._bank[tid]
        return len(to_remove)

    @property
    def size(self) -> int:
        return len(self._bank)

    @property
    def inactive_count(self) -> int:
        return sum(1 for ht in self._bank.values() if not ht.is_active)


def is_umot_enabled() -> bool:
    return _env_bool("SWAY_UMOT_BACKTRACK", False)


# Alias expected by main.py / integration tests
UMOTBacktracker = HistoricalTrajectoryBank
