"""Tests for Sway Handshake Phase 1–3 helpers."""

from __future__ import annotations

import numpy as np

from sway.handshake_tracking import _bhattacharyya, _profile_score, phase13_handshake_enabled


def test_phase13_handshake_env(monkeypatch) -> None:
    monkeypatch.delenv("SWAY_PHASE13_MODE", raising=False)
    assert phase13_handshake_enabled() is False
    monkeypatch.setenv("SWAY_PHASE13_MODE", "sway_handshake")
    assert phase13_handshake_enabled() is True


def test_profile_score_with_prof() -> None:
    p = np.ones(32, dtype=np.float64) / 32.0
    s = _profile_score(p, 0.4, (p.copy(), 0.4))
    assert s > 0.5
