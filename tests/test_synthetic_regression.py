"""
Synthetic regression contracts (plan E2 stand-in until full MOT fixtures exist).

A — crossover: Pass 1.5 IoU+centroid merge is OKS-gated.
B — mirror: smart-mirror pruner is available for edge+velocity+lower-body rules.
C — static silhouette: short / low-kinetic tracks are dropped by pre-pose pruning.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_A_crossover_pass15_oks_gate_in_source():
    p = ROOT / "sway" / "crossover.py"
    text = p.read_text()
    assert "OKS_VETO_PASS15" in text
    assert "merge_iou_centroid" in text


def test_B_smart_mirror_pruner_callable():
    from sway.track_pruning import prune_smart_mirrors

    assert callable(prune_smart_mirrors)


def test_C_short_track_pruned():
    from sway.track_pruning import prune_tracks

    # One frame only in 1000-frame video -> below duration threshold
    raw = {99: [(0, (0.0, 0.0, 10.0, 20.0), 0.9)]}
    kept = prune_tracks(raw, total_frames=1000, min_duration_ratio=0.20)
    assert 99 not in kept
