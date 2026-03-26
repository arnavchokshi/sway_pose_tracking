"""In-pipeline experimental hooks (GNN / HMR sidecar)."""

import json
import os
from pathlib import Path

from sway.experimental_hooks import (
    gnn_track_refine_enabled,
    maybe_gnn_refine_raw_tracks,
    write_hmr_mesh_sidecar_json,
)


def test_gnn_disabled_by_default():
    os.environ.pop("SWAY_GNN_TRACK_REFINE", None)
    assert gnn_track_refine_enabled() is False
    assert maybe_gnn_refine_raw_tracks({1: []}, 10, 1) == {1: []}


def test_gnn_enabled_single_empty_track():
    os.environ["SWAY_GNN_TRACK_REFINE"] = "1"
    try:
        raw = {1: []}
        out = maybe_gnn_refine_raw_tracks(raw, 3, 1)
        assert out is raw
        assert 1 in raw and raw[1] == []
    finally:
        os.environ.pop("SWAY_GNN_TRACK_REFINE", None)


def test_gnn_merges_high_iou_duplicate_ids():
    """Two IDs covering the same boxes on the same frames → one component after GNN."""
    os.environ["SWAY_GNN_TRACK_REFINE"] = "1"
    os.environ["SWAY_GNN_DEVICE"] = "cpu"
    try:
        box = (10.0, 20.0, 50.0, 120.0)
        raw = {
            1: [(0, box, 0.9), (1, box, 0.9), (2, box, 0.9)],
            2: [(0, box, 0.85), (1, box, 0.85), (2, box, 0.85)],
        }
        out = maybe_gnn_refine_raw_tracks(raw, total_frames=10, ystride=1)
        assert len(out) == 1
        tid = next(iter(out))
        assert len(out[tid]) == 3
    finally:
        os.environ.pop("SWAY_GNN_TRACK_REFINE", None)
        os.environ.pop("SWAY_GNN_DEVICE", None)


def test_gnn_skips_far_disjoint_tracks():
    os.environ["SWAY_GNN_TRACK_REFINE"] = "1"
    os.environ["SWAY_GNN_DEVICE"] = "cpu"
    os.environ["SWAY_GNN_MAX_GAP"] = "5"
    try:
        a = (10.0, 20.0, 40.0, 100.0)
        b = (200.0, 20.0, 240.0, 100.0)
        raw = {
            1: [(0, a, 0.9), (1, a, 0.9)],
            2: [(100, b, 0.9), (101, b, 0.9)],
        }
        out = maybe_gnn_refine_raw_tracks(raw, total_frames=200, ystride=1)
        assert len(out) == 2
    finally:
        os.environ.pop("SWAY_GNN_TRACK_REFINE", None)
        os.environ.pop("SWAY_GNN_DEVICE", None)
        os.environ.pop("SWAY_GNN_MAX_GAP", None)


def test_hmr_sidecar_writes(tmp_path):
    os.environ["SWAY_HMR_MESH_SIDECAR"] = "1"
    try:
        write_hmr_mesh_sidecar_json(tmp_path)
        p = tmp_path / "hmr_mesh_sidecar.json"
        assert p.is_file()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data.get("schema") == "sway.hmr_mesh_sidecar.v1"
    finally:
        os.environ.pop("SWAY_HMR_MESH_SIDECAR", None)
