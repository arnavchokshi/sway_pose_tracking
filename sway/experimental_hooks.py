"""
Default-off experimental hooks called from ``main.py`` (in-pipeline, not Lab post-subprocess).

GNN refine is a full graph pass (see ``sway.gnn_track_refine``); HMR remains a sidecar
placeholder. Flags still change run behavior (logs, manifest diagnostics, optional JSON).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def gnn_track_refine_enabled() -> bool:
    return os.environ.get("SWAY_GNN_TRACK_REFINE", "").strip().lower() in ("1", "true", "yes")


def maybe_gnn_refine_raw_tracks(
    raw_tracks: Dict[int, List[Any]],
    total_frames: int,
    ystride: int,
) -> Dict[int, List[Any]]:
    """
    Optional post-stitch graph refine: edge-conditioned multi-head GAT + link logits
    (see ``sway.gnn_track_refine``). Mutates ``raw_tracks`` in place when enabled.
    """
    if not gnn_track_refine_enabled():
        return raw_tracks
    from sway.gnn_track_refine import gnn_refine_raw_tracks

    return gnn_refine_raw_tracks(raw_tracks, int(total_frames), int(ystride))


def hmr_mesh_sidecar_enabled() -> bool:
    return os.environ.get("SWAY_HMR_MESH_SIDECAR", "").strip().lower() in ("1", "true", "yes")


def write_hmr_mesh_sidecar_json(output_dir: Path) -> None:
    """Placeholder sidecar until HMR / mesh export is implemented."""
    if not hmr_mesh_sidecar_enabled():
        return
    p = output_dir / "hmr_mesh_sidecar.json"
    payload = {
        "schema": "sway.hmr_mesh_sidecar.v1",
        "status": "placeholder",
        "note": (
            "HMR / 4DHumans mesh is not bundled. This file is written when SWAY_HMR_MESH_SIDECAR=1 "
            "so downstream tools have a stable path; replace with real vertices/faces when integrated."
        ),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  HMR mesh sidecar (placeholder): {p}", flush=True)
