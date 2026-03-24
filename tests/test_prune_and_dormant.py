import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from sway.dormant_tracks import apply_dormant_merges  # noqa: E402
from sway.track_pruning import compute_confirmed_human_set  # noqa: E402


def test_confirmed_human_set():
    scores = np.ones(17, dtype=np.float32) * 0.9
    # Bbox center in-frame (not mirror edge) so Tier A spatial sanity passes with frame_width set
    box = (500.0, 100.0, 600.0, 400.0)
    fd0 = {
        "frame_idx": 0,
        "track_ids": [1],
        "boxes": [box],
        "poses": {1: {"scores": scores}},
    }
    fd1 = {
        "frame_idx": 50,
        "track_ids": [1],
        "boxes": [box],
        "poses": {1: {"scores": scores}},
    }
    conf = compute_confirmed_human_set([fd0, fd1], total_frames=100, frame_width=1920)
    assert 1 in conf


def test_dormant_merge_gap():
    box = (100.0, 100.0, 150.0, 200.0)
    # A ends at 10, B starts at 100 -> gap 89f (> track_buffer 90? 89<=90 fails)
    # Need gap > 90: end 10, start 102 -> gap 102-10-1 = 91
    raw = {
        1: [(8, box, 0.9), (10, box, 0.9)],
        2: [(102, box, 0.9)],
    }
    out = apply_dormant_merges(dict(raw), total_frames=200, track_buffer=90, max_gap=150)
    assert 2 not in out
    assert 1 in out
    assert len(out[1]) == 3
