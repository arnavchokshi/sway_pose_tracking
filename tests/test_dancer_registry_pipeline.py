"""Unit tests for optional Dancer Registry Phase 1–3 pass."""

from __future__ import annotations

import numpy as np

from sway.dancer_registry_pipeline import (
    _bhattacharyya,
    _build_frame_map,
    _swap_track_interval,
)
from sway.track_observation import TrackObservation


def test_bhattacharyya_identical() -> None:
    p = np.ones(8, dtype=np.float64) / 8.0
    assert abs(_bhattacharyya(p, p) - 1.0) < 1e-6


def test_swap_track_interval() -> None:
    raw = {
        1: [
            TrackObservation(0, (0.0, 0.0, 10.0, 20.0), 0.9),
            TrackObservation(1, (1.0, 1.0, 11.0, 21.0), 0.9),
            TrackObservation(2, (2.0, 2.0, 12.0, 22.0), 0.9),
        ],
        2: [
            TrackObservation(0, (100.0, 0.0, 110.0, 20.0), 0.8),
            TrackObservation(1, (101.0, 1.0, 111.0, 21.0), 0.8),
        ],
    }
    _swap_track_interval(raw, 1, 2, 0, 1)
    by_f = _build_frame_map(raw)
    assert 1 in by_f[0] and 2 in by_f[0]
    # Swapped boxes at frames 0–1
    assert abs(by_f[0][1][0] - 100.0) < 0.1
    assert abs(by_f[0][2][0] - 0.0) < 0.1
    assert abs(by_f[1][1][0] - 101.0) < 0.1
    assert abs(by_f[1][2][0] - 1.0) < 0.1
    # Frame 2 unchanged for tid 1
    assert abs(by_f[2][1][0] - 2.0) < 0.1
