#!/usr/bin/env python3
"""
Queue the Tier 1–3 A/B run set described in product docs via Pipeline Lab API (CLI-only usage).

  cd sway_pose_mvp
  python3 -m tools.queue_bigtest_ab_runs --video /path/to/BigTest.mov --verify-start

Requires uvicorn on PIPELINE_LAB_URL (default http://localhost:8765). Uses POST /api/runs/batch_path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout_s: float = 600.0) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        raw = json.dumps(payload).encode("utf-8")
        data = raw
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code} {url}: {detail[:2000]}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Request failed {url}: {e}") from e
    if not body.strip():
        return None
    return json.loads(body)


def _build_runs() -> List[Dict[str, Any]]:
    """23 recipe rows: field overrides merge on top of Lab schema defaults in the server."""
    return [
        # Tier 1
        {"recipe_name": "T01_pose_stride1_baseline", "fields": {"pose_stride": 1}},
        {"recipe_name": "T01_pose_stride2_gap_GSI", "fields": {"pose_stride": 2, "sway_pose_gap_interp_mode": "gsi"}},
        {"recipe_name": "T02_yolo_dancetrack", "fields": {"sway_yolo_weights": "yolo26l_dancetrack"}},
        {"recipe_name": "T02_yolo26x", "fields": {"sway_yolo_weights": "yolo26x"}},
        {"recipe_name": "T03_pose_ViTPose_Large", "fields": {"pose_model": "ViTPose-Large"}},
        {"recipe_name": "T03_pose_ViTPose_Base", "fields": {"pose_model": "ViTPose-Base"}},
        {"recipe_name": "T04_tracker_deep_ocsort_motion", "fields": {"tracker_technology": "deep_ocsort"}},
        {
            "recipe_name": "T04_tracker_deep_ocsort_osnet_x0_25",
            "fields": {
                "tracker_technology": "deep_ocsort_osnet",
                "sway_boxmot_reid_model": "osnet_x0_25",
            },
        },
        # Tier 2
        {"recipe_name": "T05_pretrack_nms_0.40", "fields": {"sway_pretrack_nms_iou": 0.4}},
        {"recipe_name": "T05_pretrack_nms_0.50", "fields": {"sway_pretrack_nms_iou": 0.5}},
        {"recipe_name": "T05_pretrack_nms_0.65", "fields": {"sway_pretrack_nms_iou": 0.65}},
        {"recipe_name": "T06_hybrid_sam_iou_0.30", "fields": {"sway_hybrid_sam_iou_trigger": 0.30}},
        {"recipe_name": "T06_hybrid_sam_iou_0.42", "fields": {"sway_hybrid_sam_iou_trigger": 0.42}},
        {"recipe_name": "T06_hybrid_sam_iou_0.55", "fields": {"sway_hybrid_sam_iou_trigger": 0.55}},
        {"recipe_name": "T07_dedup_antipartner_0.12", "fields": {"dedup_antipartner_min_iou": 0.12}},
        {"recipe_name": "T07_dedup_antipartner_0.25", "fields": {"dedup_antipartner_min_iou": 0.25}},
        # Tier 3
        {"recipe_name": "T08_temporal_pose_refine_on", "fields": {"temporal_pose_refine": True}},
        {"recipe_name": "T08_temporal_pose_refine_off", "fields": {"temporal_pose_refine": False}},
        {"recipe_name": "T09_stitch_max_gap_60", "fields": {"sway_stitch_max_frame_gap": 60}},
        {"recipe_name": "T09_stitch_max_gap_120", "fields": {"sway_stitch_max_frame_gap": 120}},
        {"recipe_name": "T10_prune_threshold_default", "fields": {"prune_threshold": 0.65}},
        {"recipe_name": "T10_prune_threshold_lenient", "fields": {"prune_threshold": 0.55}},
        {
            "recipe_name": "T10_prune_lower_mirror_sync_weights",
            "fields": {"pruning_w_smart_mirror": 0.4, "pruning_w_low_sync": 0.4},
        },
    ]


def _verify_batch(
    base: str,
    batch_id: str,
    run_ids: List[str],
    timeout_s: float = 90.0,
    poll_s: float = 2.0,
) -> Tuple[bool, str]:
    want = set(run_ids)
    deadline = time.time() + timeout_s
    saw_running = False
    last_msg = ""
    while time.time() < deadline:
        rows = _http_json("GET", f"{base}/api/runs", None, timeout_s=60.0)
        if not isinstance(rows, list):
            time.sleep(poll_s)
            continue
        by_id = {str(r.get("run_id")): r for r in rows if isinstance(r, dict)}
        in_batch = [by_id[rid] for rid in run_ids if rid in by_id]
        if len(in_batch) != len(run_ids):
            last_msg = f"listed {len(in_batch)}/{len(run_ids)} runs (API list lag?)"
            time.sleep(poll_s)
            continue
        bad_batch = [r for r in in_batch if str(r.get("batch_id") or "") != batch_id]
        if bad_batch:
            last_msg = "batch_id mismatch on some rows"
            time.sleep(poll_s)
            continue
        statuses = [str(r.get("status") or "") for r in in_batch]
        queued_n = sum(1 for s in statuses if s == "queued")
        running = [r for r in in_batch if r.get("status") == "running"]
        if running:
            alive = [r for r in running if r.get("subprocess_alive") is True]
            if alive:
                saw_running = True
                last_msg = (
                    f"RUNNING_OK: {len(alive)} subprocess(es) alive (e.g. {alive[0].get('run_id')[:8]}…); "
                    f"{queued_n} still queued"
                )
                return True, last_msg
            last_msg = f"status=running but subprocess_alive not true yet ({len(running)} rows)"
        else:
            last_msg = f"statuses sample: {statuses[:5]}… queued={queued_n}"
        time.sleep(poll_s)
    return saw_running, last_msg or "timeout"


def main() -> None:
    ap = argparse.ArgumentParser(description="Queue BigTest A/B batch on Pipeline Lab (CLI)")
    ap.add_argument("--video", type=str, required=True, help="Absolute path to video on the Lab server machine")
    ap.add_argument("--lab-url", type=str, default="http://localhost:8765", help="Lab base URL (no trailing slash)")
    ap.add_argument("--dry-run", action="store_true", help="Print JSON payload only")
    ap.add_argument(
        "--verify-start",
        action="store_true",
        help="Poll /api/runs until a run in this batch is running with subprocess_alive=true",
    )
    args = ap.parse_args()
    video = Path(args.video).expanduser()
    if not args.dry_run and not video.is_file():
        raise SystemExit(f"video not found: {video}")

    runs = _build_runs()
    base = args.lab_url.rstrip("/")
    body: Dict[str, Any] = {
        "video_path": str(video.resolve()),
        "runs": runs,
        "source_label": video.stem,
    }

    health = _http_json("GET", f"{base}/api/health", None, timeout_s=10.0)
    if not isinstance(health, dict) or health.get("ok") != "true":
        raise SystemExit(f"Lab health failed at {base}/api/health: {health!r}")

    if args.dry_run:
        print(json.dumps({"endpoint": f"{base}/api/runs/batch_path", "body": body}, indent=2))
        print(f"\n(run count: {len(runs)})", file=sys.stderr)
        return

    print(f"POST {base}/api/runs/batch_path — {len(runs)} runs, video={video}", flush=True)
    resp = _http_json("POST", f"{base}/api/runs/batch_path", body, timeout_s=600.0)
    if not isinstance(resp, dict):
        raise SystemExit(f"Unexpected response: {resp!r}")
    print(json.dumps(resp, indent=2), flush=True)

    batch_id = str(resp.get("batch_id") or "")
    run_ids = resp.get("run_ids")
    if not batch_id or not isinstance(run_ids, list) or len(run_ids) != len(runs):
        raise SystemExit("batch response missing batch_id or run_ids count mismatch")

    # Immediate list check (queued)
    rows = _http_json("GET", f"{base}/api/runs", None, timeout_s=60.0)
    if isinstance(rows, list):
        ours = [r for r in rows if isinstance(r, dict) and r.get("run_id") in set(run_ids)]
        print(f"\nListed {len(ours)}/{len(run_ids)} runs from GET /api/runs", flush=True)
        for r in sorted(ours, key=lambda x: str(x.get("recipe_name") or ""))[:5]:
            print(
                f"  {str(r.get('run_id'))[:8]}… {r.get('recipe_name')} status={r.get('status')} batch={r.get('batch_id') == batch_id}",
                flush=True,
            )
        if len(ours) < len(run_ids):
            print("  (remaining rows appear as the server finishes merging state)", flush=True)

    if args.verify_start:
        print("\nPolling until a run is RUNNING with subprocess_alive=true …", flush=True)
        ok, msg = _verify_batch(base, batch_id, [str(x) for x in run_ids])
        print(msg, flush=True)
        if not ok:
            raise SystemExit(
                "VERIFY_FAILED: no subprocess marked alive within timeout. "
                "Check Lab logs, PIPELINE_LAB_MAX_PARALLEL, and pipeline_lab/runs/<id>/stdout.log"
            )
        # Optional: prove stdout is being written for the first running id
        first = str(run_ids[0])
        log_path = REPO_ROOT / "pipeline_lab" / "runs" / first / "stdout.log"
        if log_path.is_file():
            print(f"\nstdout.log exists for first queued id {first[:8]}… size={log_path.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
