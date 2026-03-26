"""Integration and edge-case tests for GNN post-stitch track refinement."""

from __future__ import annotations

import os
from typing import Dict, Iterator, List

import pytest
import torch

from sway.experimental_hooks import maybe_gnn_refine_raw_tracks
from sway.gnn_track_refine import RelationalTrackGNN, _hard_forbid_merge, gnn_refine_raw_tracks
from sway.track_observation import TrackObservation


_GNN_ENV_KEYS = (
    "SWAY_GNN_MERGE_THRESH",
    "SWAY_GNN_HIDDEN",
    "SWAY_GNN_HEADS",
    "SWAY_GNN_LAYERS",
    "SWAY_GNN_DROPOUT",
    "SWAY_GNN_MAX_GAP",
    "SWAY_GNN_PRIOR_SCALE",
    "SWAY_GNN_WEIGHTS",
    "SWAY_GNN_DEVICE",
    "SWAY_GNN_SEED",
    "SWAY_GNN_TRACK_REFINE",
)


@pytest.fixture
def isolate_gnn_env() -> Iterator[None]:
    saved: Dict[str, str | None] = {k: os.environ.get(k) for k in _GNN_ENV_KEYS}
    for k in _GNN_ENV_KEYS:
        os.environ.pop(k, None)
    yield
    for k in _GNN_ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def _cpu_gnn_defaults() -> None:
    os.environ["SWAY_GNN_DEVICE"] = "cpu"
    os.environ["SWAY_GNN_SEED"] = "0"


def test_gnn_refine_forward_module_shapes(isolate_gnn_env) -> None:
    """Sanity check: RelationalTrackGNN runs on a tiny dense graph."""
    _cpu_gnn_defaults()
    n, node_in, edge_in = 4, 16, 6
    x = torch.randn(n, node_in)
    ef = torch.randn(n, n, edge_in)
    adj = torch.eye(n)
    adj[0, 1] = adj[1, 0] = adj[1, 2] = adj[2, 1] = 1.0
    m = RelationalTrackGNN(
        node_in=node_in,
        edge_in=edge_in,
        hidden=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )
    m.eval()
    with torch.no_grad():
        logits = m(x, ef, adj)
    assert logits.shape == (n, n)


def test_hard_forbid_merge_rules() -> None:
    assert _hard_forbid_merge({"n_overlap": 10.0, "max_iou": 0.05, "mean_dist_h": 0.5}) is True
    assert _hard_forbid_merge({"n_overlap": 2.0, "max_iou": 0.5, "mean_dist_h": 0.1}) is False


def test_merge_three_duplicate_tracks(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    box = (5.0, 5.0, 55.0, 205.0)
    raw = {
        10: [(0, box, 0.9), (1, box, 0.9)],
        20: [(0, box, 0.88), (1, box, 0.88)],
        30: [(0, box, 0.87), (1, box, 0.87)],
    }
    gnn_refine_raw_tracks(raw, total_frames=20, ystride=1)
    assert len(raw) == 1
    assert len(next(iter(raw.values()))) == 2


def test_high_merge_threshold_prevents_merge(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    os.environ["SWAY_GNN_MERGE_THRESH"] = "1.01"
    box = (10.0, 20.0, 50.0, 120.0)
    raw = {
        1: [(0, box, 0.9), (1, box, 0.9)],
        2: [(0, box, 0.85), (1, box, 0.85)],
    }
    gnn_refine_raw_tracks(raw, total_frames=15, ystride=1)
    assert len(raw) == 2


def test_preserves_empty_track_when_merging_others(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    box = (1.0, 1.0, 40.0, 100.0)
    raw = {
        0: [],
        1: [(0, box, 0.9), (1, box, 0.9)],
        2: [(0, box, 0.86), (1, box, 0.86)],
    }
    gnn_refine_raw_tracks(raw, total_frames=10, ystride=1)
    assert 0 in raw and raw[0] == []
    assert len(raw) == 2


def test_two_people_overlapping_time_not_merged(isolate_gnn_env) -> None:
    """Side-by-side boxes on same frames → hard forbid or low same-person score → 2 tracks."""
    _cpu_gnn_defaults()
    left = (10.0, 100.0, 50.0, 200.0)
    right = (400.0, 100.0, 440.0, 200.0)
    frames = list(range(12))
    raw = {
        1: [(f, left, 0.9) for f in frames],
        2: [(f, right, 0.9) for f in frames],
    }
    gnn_refine_raw_tracks(raw, total_frames=30, ystride=1)
    assert len(raw) == 2
    assert len(raw[1]) == 12
    assert len(raw[2]) == 12


def test_track_observation_merge_keeps_dataclass(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    box = (20.0, 30.0, 80.0, 180.0)
    raw = {
        1: [
            TrackObservation(0, box, 0.91, is_sam_refined=True),
            TrackObservation(1, box, 0.90, is_sam_refined=False),
        ],
        2: [
            TrackObservation(0, box, 0.89, is_sam_refined=False),
            TrackObservation(1, box, 0.88, is_sam_refined=True),
        ],
    }
    gnn_refine_raw_tracks(raw, total_frames=12, ystride=2)
    assert len(raw) == 1
    merged = next(iter(raw.values()))
    assert len(merged) == 2
    for e in merged:
        assert isinstance(e, TrackObservation)


def test_small_architecture_env_runs(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    os.environ["SWAY_GNN_HIDDEN"] = "32"
    os.environ["SWAY_GNN_HEADS"] = "4"
    os.environ["SWAY_GNN_LAYERS"] = "2"
    box = (0.0, 0.0, 30.0, 90.0)
    raw = {1: [(0, box, 0.9)], 2: [(0, box, 0.85)]}
    gnn_refine_raw_tracks(raw, total_frames=5, ystride=1)
    assert len(raw) == 1


def test_state_dict_roundtrip_via_weights_env(isolate_gnn_env, tmp_path) -> None:
    _cpu_gnn_defaults()
    pt = tmp_path / "gnn_dummy.pt"
    m = RelationalTrackGNN(
        node_in=16,
        edge_in=6,
        hidden=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )
    torch.save(m.state_dict(), pt)
    os.environ["SWAY_GNN_WEIGHTS"] = str(pt)
    os.environ["SWAY_GNN_HIDDEN"] = "32"
    os.environ["SWAY_GNN_HEADS"] = "4"
    os.environ["SWAY_GNN_LAYERS"] = "1"
    box = (10.0, 10.0, 60.0, 160.0)
    raw = {5: [(0, box, 0.9)], 7: [(0, box, 0.8)]}
    gnn_refine_raw_tracks(raw, total_frames=8, ystride=1)
    assert len(raw) == 1


def test_maybe_gnn_hook_disabled_short_circuits(isolate_gnn_env) -> None:
    """No SWAY_GNN_TRACK_REFINE → never import path that builds the graph (lightweight)."""
    os.environ.pop("SWAY_GNN_TRACK_REFINE", None)
    raw: Dict[int, List] = {1: [(0, (0, 0, 1, 1), 0.5)]}
    out = maybe_gnn_refine_raw_tracks(raw, 5, 1)
    assert out is raw
    assert len(raw) == 1


def test_maybe_gnn_hook_enabled_matches_direct_call(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    os.environ["SWAY_GNN_TRACK_REFINE"] = "1"
    box = (2.0, 2.0, 42.0, 102.0)
    raw_hook = {1: [(0, box, 0.9), (1, box, 0.9)], 2: [(0, box, 0.85), (1, box, 0.85)]}
    raw_direct = {1: [(0, box, 0.9), (1, box, 0.9)], 2: [(0, box, 0.85), (1, box, 0.85)]}
    maybe_gnn_refine_raw_tracks(raw_hook, 20, 1)
    gnn_refine_raw_tracks(raw_direct, 20, 1)
    assert set(raw_hook.keys()) == set(raw_direct.keys())
    assert len(raw_hook) == 1
    assert len(raw_direct) == 1
    assert len(raw_hook[next(iter(raw_hook.keys()))]) == 2


def test_single_nonempty_track_noop(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    raw = {99: [(0, (0, 0, 10, 10), 0.7)]}
    gnn_refine_raw_tracks(raw, total_frames=100, ystride=1)
    assert raw == {99: [(0, (0, 0, 10, 10), 0.7)]}


def test_empty_raw_tracks_dict(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    raw: Dict[int, List] = {}
    assert gnn_refine_raw_tracks(raw, 10, 1) == raw


def test_only_empty_tracks_early_exit(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    raw = {1: [], 2: []}
    gnn_refine_raw_tracks(raw, 10, 1)
    assert raw == {1: [], 2: []}


def test_build_track_graph_tensors_matches_inference(isolate_gnn_env) -> None:
    from sway.gnn_track_refine import build_track_graph_tensors

    _cpu_gnn_defaults()
    box = (1.0, 1.0, 41.0, 131.0)
    raw = {1: [(0, box, 0.9), (1, box, 0.9)], 2: [(0, box, 0.8), (1, box, 0.8)]}
    g = build_track_graph_tensors(
        raw,
        total_frames=50,
        max_gap=120.0,
        prior_scale=1.0,
        device=torch.device("cpu"),
    )
    assert g is not None
    assert g.x0.shape == (2, 16)
    assert g.edge_feat.shape[0] == 2
    assert g.adj[0, 1] > 0


def test_edge_bce_loss_no_edges_still_has_grad() -> None:
    """Regression: empty edge set must not return a detached zero (breaks backward)."""
    from tools.train_gnn_track_refine import edge_bce_loss

    n = 3
    learn = torch.randn(n, n, requires_grad=True)
    prior = torch.zeros(n, n)
    adj = torch.eye(n)
    node_ids = [1, 2, 3]
    tid_to_person = {1: 0, 2: 1, 3: 2}
    loss = edge_bce_loss(learn, prior, adj, node_ids, tid_to_person)
    loss.backward()
    assert learn.grad is not None


def test_quick_train_writes_checkpoint(isolate_gnn_env, tmp_path) -> None:
    from tools.train_gnn_track_refine import run_training

    out = tmp_path / "gnn.pt"
    run_training(
        out_path=out,
        steps=5,
        lr=1e-3,
        device=torch.device("cpu"),
        hidden=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        max_gap=120.0,
        prior_scale=1.0,
        total_frames=180,
        seed=1,
        log_every=0,
    )
    assert out.is_file()
    ckpt = torch.load(out, map_location="cpu")
    assert "state_dict" in ckpt
    assert "meta" in ckpt
