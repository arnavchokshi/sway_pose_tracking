#!/usr/bin/env python3
"""
Queue Lab runs that exercise ideas from ``docs/FUTURE_MODULES_IDENTITY_AND_POSE_CROPS.md``.

Uses ``POST /api/runs/batch_path`` with a shared ``batch_id`` per video pair so the Lab can open
``/?batch=<id>``. Also prints ``/?runs=`` including prior run IDs so everything appears in one session.

Usage:
    python -m tools.queue_future_module_matrix [--api-url http://127.0.0.1:8765] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import uuid
from typing import Any, Dict, List, Tuple

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

VIDEOS = [
    "/Users/arnavchokshi/Desktop/BigTest.mov",
    "/Users/arnavchokshi/Desktop/IMG_2946.MP4",
]

# Prior preset-test run IDs (empty batch_id on server) — keep in session alongside new matrix.
LEGACY_RUN_IDS = [
    "d81bd849-a2a5-4ccd-bce3-a642fd8539ab",
    "53af2010-957b-468e-9f2f-530331552c3b",
    "a6020786-2b71-4747-a1d6-5b4f127c9088",
    "4fd4200f-b12a-41f0-a394-8034eee0813f",
    "e8122e28-3509-46b0-acfc-3b9e8d6b86ec",
    "17fd1941-6f1e-47e1-bd21-413e7358e0cd",
    "ae3fedf0-76fe-4ce8-bb9e-834fc8bd7bb9",
    "7a8fed64-888c-4002-b583-b8ec223e9107",
    "8d09d992-416e-4225-8498-3751d50d76cd",
    "7993a239-8738-4794-8edd-9790f0494a1f",
    "dcae484b-3d3a-4e8e-aaa1-c4cf3fadd3a5",
    "92fc3270-712e-4859-bdaf-1742133e1434",
    "33c6439e-9e48-4b75-a2f3-93ed958d5ee7",
    "446dde76-f45f-4019-942d-c5ac02892c7f",
]

BASE: Dict[str, Any] = {
    "sway_phase13_mode": "standard",
    "sway_yolo_weights": "yolo26l_dancetrack",
    "tracker_technology": "deep_ocsort",
    "sway_hybrid_sam_iou_trigger": 0.42,
    "sway_boxmot_max_age": 150,
    "pose_model": "ViTPose-Base",
    "sway_pose_3d_lift": True,
    "pose_visibility_threshold": 0.30,
    "prune_threshold": 0.65,
    "sync_score_min": 0.10,
    "smoother_beta": 0.70,
}

# (label_suffix, field_overrides) — maps doc bundles + registry to concrete fields.
FUTURE_MATRIX: List[Tuple[str, Dict[str, Any]]] = [
    ("FM crop EMA light", {"sway_pose_crop_smooth_alpha": 0.22}),
    ("FM crop EMA + foot", {"sway_pose_crop_smooth_alpha": 0.28, "sway_pose_crop_foot_bias_frac": 0.08}),
    ("FM crop EMA + head + anti-jitter", {"sway_pose_crop_smooth_alpha": 0.2, "sway_pose_crop_head_bias_frac": 0.12, "sway_pose_crop_anti_jitter_px": 28.0}),
    ("FM crop full stack", {"sway_pose_crop_smooth_alpha": 0.22, "sway_pose_crop_foot_bias_frac": 0.06, "sway_pose_crop_head_bias_frac": 0.10, "sway_pose_crop_anti_jitter_px": 32.0}),
    ("FM GNN track refine", {"sway_gnn_track_refine": True}),
    ("FM dancer registry", {"sway_phase13_mode": "dancer_registry"}),
    ("FM sway handshake", {"sway_phase13_mode": "sway_handshake"}),
    ("FM Deep OC-SORT + OSNet", {"tracker_technology": "deep_ocsort_osnet"}),
    ("FM bidirectional track pass", {"sway_bidirectional_track_pass": True}),
    ("FM temporal pose refine", {"temporal_pose_refine": True, "temporal_pose_radius": 3}),
    ("FM dense crowd stack", {"sway_yolo_weights": "yolo26x", "sway_pretrack_nms_iou": 0.60, "sway_yolo_conf": 0.18, "pose_model": "ViTPose-Large"}),
    ("FM hybrid weak cues", {"sway_hybrid_sam_weak_cues": True}),
    ("FM crop + GNN", {"sway_pose_crop_smooth_alpha": 0.25, "sway_pose_crop_foot_bias_frac": 0.05, "sway_gnn_track_refine": True}),
    ("FM crop + registry", {"sway_pose_crop_smooth_alpha": 0.2, "sway_phase13_mode": "dancer_registry"}),
]


def _runs_payload(video_stem: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for suffix, extra in FUTURE_MATRIX:
        fields = {**BASE, **extra}
        out.append({"recipe_name": f"{video_stem} — FutureDoc — {suffix}", "fields": fields})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://127.0.0.1:8765")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api = args.api_url.rstrip("/")

    batch_id = str(uuid.uuid4())
    all_new_ids: List[str] = []

    for vp in VIDEOS:
        stem = vp.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        body = {
            "video_path": vp,
            "batch_id": batch_id,
            "source_label": stem,
            "runs": _runs_payload(stem),
        }
        if args.dry_run:
            print(f"[dry-run] POST batch_path {stem} n={len(body['runs'])} batch_id={batch_id}")
            continue
        r = requests.post(f"{api}/api/runs/batch_path", json=body, timeout=600)
        r.raise_for_status()
        j = r.json()
        all_new_ids.extend(j.get("run_ids") or [])
        print(f"Queued {len(j.get('run_ids', []))} runs for {stem}")

    combined = LEGACY_RUN_IDS + all_new_ids
    runs_q = ",".join(combined)
    print()
    print("--- Lab URLs (Vite dev) ---")
    print(f"http://localhost:5173/?batch={batch_id}")
    print("(new matrix only — auto-syncs all runs with this batch_id)")
    print()
    print("http://localhost:5173/?runs=" + runs_q)
    print("(legacy 14 + new matrix — single session)")
    print()
    print("Bundled UI + API:")
    print(f"http://localhost:8765/?batch={batch_id}")


if __name__ == "__main__":
    main()
