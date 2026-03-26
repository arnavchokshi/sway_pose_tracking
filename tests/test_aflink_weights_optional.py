"""Neural AFLink needs ``models/AFLink_epoch20.pth`` (``python -m tools.prefetch_models``)."""

from __future__ import annotations

import pytest

from sway.global_track_link import neural_global_stitch, resolve_aflink_weights
from sway.track_observation import TrackObservation


def _aflink_weights() -> bool:
    return resolve_aflink_weights().is_file()


@pytest.mark.skipif(not _aflink_weights(), reason="AFLink_epoch20.pth missing; run: python -m tools.prefetch_models")
def test_aflink_checkpoint_loads() -> None:
    from sway.aflink import AFLink

    p = resolve_aflink_weights()
    linker = AFLink(str(p), thrT=(0, 30), thrS=75, thrP=0.05)
    assert linker is not None


@pytest.mark.skipif(not _aflink_weights(), reason="AFLink_epoch20.pth missing")
def test_neural_global_stitch_runs_without_error() -> None:
    p = resolve_aflink_weights()
    box = (100.0, 100.0, 140.0, 200.0)
    raw = {
        1: [TrackObservation(0, box, 0.9), TrackObservation(1, box, 0.9), TrackObservation(2, box, 0.9)],
        2: [TrackObservation(10, box, 0.9), TrackObservation(11, box, 0.9), TrackObservation(12, box, 0.9)],
    }
    out = neural_global_stitch(raw, total_frames=30, path_AFLink=str(p))
    assert isinstance(out, dict)
    for tid, obs in out.items():
        assert isinstance(tid, int)
        assert all(isinstance(o, TrackObservation) for o in obs)
