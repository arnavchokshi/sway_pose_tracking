from types import SimpleNamespace

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sway.noougat_graph_stitch import NOOUGATGraphStitcher, make_tracklet_node
from sway.reid_fusion import ReIDQuery


def _part(v):
    return SimpleNamespace(global_emb=np.asarray(v, dtype=np.float32))


def test_noougat_resolves_dark_zone_with_global_assignment():
    stitcher = NOOUGATGraphStitcher()

    # Entry nodes (frozen identities before coalescence).
    q1 = ReIDQuery(track_id=1, part_embeddings=_part([1.0, 0.0, 0.0]), spatial_position=(100.0, 100.0))
    q2 = ReIDQuery(track_id=2, part_embeddings=_part([0.0, 1.0, 0.0]), spatial_position=(300.0, 100.0))
    n1 = make_tracklet_node(1, 10, np.array([80, 80, 120, 120], dtype=np.float32), q1, prev_center_xy=(98.0, 100.0))
    n2 = make_tracklet_node(2, 10, np.array([280, 80, 320, 120], dtype=np.float32), q2, prev_center_xy=(302.0, 100.0))

    stitcher.start_dark_zone(track_ids=[1, 2], entry_frame=10, entry_nodes={1: n1, 2: n2})

    # Exit IDs are swapped by tracker, but appearance should recover identity continuity.
    q11 = ReIDQuery(track_id=11, part_embeddings=_part([1.0, 0.0, 0.0]), spatial_position=(110.0, 102.0))
    q22 = ReIDQuery(track_id=22, part_embeddings=_part([0.0, 1.0, 0.0]), spatial_position=(290.0, 98.0))
    e11 = make_tracklet_node(11, 20, np.array([92, 82, 128, 122], dtype=np.float32), q11, prev_center_xy=(108.0, 101.0))
    e22 = make_tracklet_node(22, 20, np.array([272, 78, 312, 118], dtype=np.float32), q22, prev_center_xy=(292.0, 99.0))

    result = stitcher.resolve_dark_zone(track_ids=[1, 2], exit_frame=20, exit_nodes={11: e11, 22: e22})

    assert result is not None
    assert sorted(result.assignments) == [(1, 11), (2, 22)]
    assert result.cost_matrix.shape == (2, 2)
