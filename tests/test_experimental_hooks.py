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


def test_gnn_enabled_identity():
    os.environ["SWAY_GNN_TRACK_REFINE"] = "1"
    try:
        raw = {1: []}
        out = maybe_gnn_refine_raw_tracks(raw, 3, 1)
        assert out is raw
    finally:
        os.environ.pop("SWAY_GNN_TRACK_REFINE", None)


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
