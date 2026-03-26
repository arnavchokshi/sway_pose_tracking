#!/usr/bin/env python3
"""
Queue preset-based test runs for two videos via the Pipeline Lab API.

Usage:
    python -m tools.queue_preset_tests [--api-url http://127.0.0.1:8765]

Queues a selection of preset combinations on each video to compare.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Tuple

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

VIDEOS = [
    "/Users/arnavchokshi/Desktop/BigTest.mov",
    "/Users/arnavchokshi/Desktop/IMG_2946.MP4",
]

# Each entry: (recipe_name, fields_dict)
# Matches the presets defined in configPresets.ts
PRESET_COMBOS: List[Tuple[str, Dict[str, Any]]] = [
    # ---- Fast scan (quick preview) ----
    (
        "FastScan + FastSkeleton + Balanced + Minimal",
        {
            "sway_phase13_mode": "standard",
            "sway_yolo_weights": "yolo26s",
            "tracker_technology": "bytetrack",
            "sway_yolo_detection_stride": 4,
            "sway_global_aflink_mode": "force_heuristic",
            "sway_boxmot_max_age": 90,
            "pose_model": "ViTPose-Base",
            "sway_pose_3d_lift": False,
            "sway_vitpose_use_fast": True,
            "pose_visibility_threshold": 0.35,
            "save_phase_previews": False,
            "prune_threshold": 0.65,
            "sync_score_min": 0.10,
            "smoother_beta": 0.70,
            "montage": False,
        },
    ),
    # ---- Standard baseline ----
    (
        "Standard + Balanced + Balanced + Standard",
        {
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
        },
    ),
    # ---- Dense Crowd + High Fidelity ----
    (
        "DenseCrowd + HighFidelity + Balanced + Standard",
        {
            "sway_phase13_mode": "standard",
            "sway_yolo_weights": "yolo26x",
            "tracker_technology": "deep_ocsort",
            "sway_pretrack_nms_iou": 0.60,
            "sway_yolo_conf": 0.18,
            "sway_hybrid_sam_iou_trigger": 0.30,
            "sway_boxmot_max_age": 200,
            "sway_boxmot_match_thresh": 0.25,
            "pose_model": "ViTPose-Large",
            "sway_pose_3d_lift": True,
            "pose_visibility_threshold": 0.25,
            "prune_threshold": 0.65,
            "sync_score_min": 0.10,
            "smoother_beta": 0.70,
        },
    ),
    # ---- Open Floor + Competition Grade + Sharp Hip-Hop ----
    (
        "OpenFloor + Competition + SharpHipHop + Standard",
        {
            "sway_phase13_mode": "standard",
            "sway_yolo_weights": "yolo26l_dancetrack",
            "tracker_technology": "deep_ocsort",
            "sway_yolo_conf": 0.30,
            "sway_pretrack_nms_iou": 0.80,
            "sway_hybrid_sam_iou_trigger": 0.50,
            "sway_boxmot_max_age": 120,
            "sway_boxmot_match_thresh": 0.35,
            "pose_model": "ViTPose-Large",
            "sway_pose_3d_lift": True,
            "temporal_pose_refine": True,
            "pose_visibility_threshold": 0.22,
            "dedup_min_pair_oks": 0.75,
            "dedup_antipartner_min_iou": 0.15,
            "prune_threshold": 0.60,
            "sync_score_min": 0.12,
            "smoother_beta": 0.55,
        },
    ),
    # ---- Open Floor + Competition + Sharp Hip-Hop (recovery bias) — see docs/PIPELINE_FINDINGS_AND_BEST_CONFIGS.md ----
    (
        "OpenFloor + Competition + SharpHipHop + Standard (recovery bias)",
        {
            "sway_phase13_mode": "standard",
            "sway_yolo_weights": "yolo26l_dancetrack_crowdhuman",
            "tracker_technology": "deep_ocsort",
            "sway_yolo_conf": 0.30,
            "sway_pretrack_nms_iou": 0.80,
            "sway_hybrid_sam_iou_trigger": 0.42,
            "sway_boxmot_max_age": 165,
            "sway_boxmot_match_thresh": 0.29,
            "pose_model": "ViTPose-Large",
            "sway_pose_3d_lift": True,
            "temporal_pose_refine": True,
            "pose_visibility_threshold": 0.22,
            "dedup_min_pair_oks": 0.75,
            "dedup_antipartner_min_iou": 0.15,
            "prune_threshold": 0.60,
            "sync_score_min": 0.12,
            "smoother_beta": 0.55,
        },
    ),
    # ---- Standard + Balanced + Fluid Ballet ----
    (
        "Standard + Balanced + FluidBallet + Standard",
        {
            "sway_phase13_mode": "standard",
            "sway_yolo_weights": "yolo26l_dancetrack",
            "tracker_technology": "deep_ocsort",
            "sway_hybrid_sam_iou_trigger": 0.42,
            "sway_boxmot_max_age": 150,
            "pose_model": "ViTPose-Base",
            "sway_pose_3d_lift": True,
            "temporal_pose_refine": True,
            "pose_visibility_threshold": 0.30,
            "prune_threshold": 0.65,
            "sync_score_min": 0.08,
            "pruning_w_low_sync": 0.5,
            "pruning_w_smart_mirror": 0.7,
            "pruning_w_low_conf": 0.4,
            "smoother_beta": 0.85,
        },
    ),
    # ---- Standard + Balanced + Mirror Studio ----
    (
        "Standard + Balanced + MirrorStudio + Standard",
        {
            "sway_phase13_mode": "standard",
            "sway_yolo_weights": "yolo26l_dancetrack",
            "tracker_technology": "deep_ocsort",
            "sway_hybrid_sam_iou_trigger": 0.42,
            "sway_boxmot_max_age": 150,
            "pose_model": "ViTPose-Base",
            "sway_pose_3d_lift": True,
            "pose_visibility_threshold": 0.30,
            "prune_threshold": 0.60,
            "sync_score_min": 0.10,
            "pruning_w_low_sync": 0.75,
            "pruning_w_smart_mirror": 1.0,
            "pruning_w_low_conf": 0.5,
            "smoother_beta": 0.70,
        },
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Queue preset test runs")
    parser.add_argument("--api-url", default="http://127.0.0.1:8765", help="Pipeline Lab API base URL")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be queued without submitting")
    parser.add_argument("--fast-only", action="store_true", help="Only queue fast presets for a quick test")
    args = parser.parse_args()

    api = args.api_url.rstrip("/")

    # Verify API is reachable
    try:
        r = requests.get(f"{api}/api/schema", timeout=5)
        r.raise_for_status()
        print(f"API reachable at {api}")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {api}: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify videos exist
    for v in VIDEOS:
        import os
        if not os.path.isfile(v):
            print(f"WARNING: Video not found: {v}", file=sys.stderr)

    combos = PRESET_COMBOS
    if args.fast_only:
        combos = [combos[0]]  # Just the fast scan

    queued = []
    for video_path in VIDEOS:
        video_name = video_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        for recipe_name, fields in combos:
            full_name = f"{video_name} — {recipe_name}"
            if args.dry_run:
                print(f"  [DRY RUN] Would queue: {full_name}")
                continue

            payload = {
                "video_path": video_path,
                "recipe_name": full_name,
                "fields": fields,
            }
            try:
                r = requests.post(f"{api}/api/runs", json=payload, timeout=10)
                r.raise_for_status()
                data = r.json()
                run_id = data.get("run_id", "?")
                print(f"  Queued: {full_name} -> run_id={run_id[:8]}...")
                queued.append(run_id)
            except Exception as e:
                print(f"  ERROR queuing {full_name}: {e}", file=sys.stderr)

        if not args.dry_run:
            time.sleep(0.3)

    if not args.dry_run:
        print(f"\nTotal queued: {len(queued)} runs")
        print(f"Monitor at {api} or http://127.0.0.1:5175")


if __name__ == "__main__":
    main()
