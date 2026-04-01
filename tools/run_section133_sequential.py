#!/usr/bin/env python3
"""
Queue §13.3 detector A/B recipes one at a time on the Pipeline Lab, stop each run at
``after_phase_2`` (detection + tracking, no post-track stitch / pose), then pull preview MP4s.

Designed to run on the same machine as uvicorn (e.g. Lambda) so ``--video`` is a local path.

Examples:
  cd sway_pose_mvp
  python -m tools.run_section133_sequential \\
    --lab-url http://127.0.0.1:8765 \\
    --video data/ground_truth/bigtest/BigTest.mov \\
    --download-dir ~/section133_previews \\
    --open-each

  # From your Mac (Lab + video on Lambda): SSH in and run the same command there, or use
  # SSH port-forward and ensure the video path exists on the Lab host.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sway.pipeline_matrix_presets import PIPELINE_MATRIX_RECIPES  # noqa: E402

SECTION_133_IDS: List[str] = [
    "f10_det_yolo_only",
    "f11_det_codetr_only",
    "f12_det_codino_only",
    "f13_det_rtdetr_only",
    "f14_det_hybrid_codino",
    "f15_det_hybrid_rtdetr",
    "f16a_det_hybrid_overlap_lo",
    "f16_det_hybrid_sweep",
    "f16c_det_hybrid_overlap_hi",
    "f17_det_yolo_crowdhuman",
]


def _recipes_by_id() -> Dict[str, Dict[str, Any]]:
    return {str(r["id"]): r for r in PIPELINE_MATRIX_RECIPES}


def _http_request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 120.0,
    binary: bool = False,
) -> Any:
    data = None
    headers: Dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code} {url}: {detail[:1200]}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Request failed {url}: {e}") from e
    if binary:
        return body
    if not body.strip():
        return None
    return json.loads(body.decode("utf-8"))


def _wait_run_terminal(base: str, run_id: str, poll_s: float) -> Dict[str, Any]:
    terminal = frozenset({"done", "error", "cancelled"})
    while True:
        row = _http_request("GET", f"{base}/api/runs/{run_id}", None, timeout_s=60.0)
        if isinstance(row, dict) and str(row.get("status") or "") in terminal:
            return row
        time.sleep(max(0.5, poll_s))


def main() -> None:
    ap = argparse.ArgumentParser(description="§13.3 detector matrix: sequential Lab runs, after_phase_2")
    ap.add_argument("--lab-url", type=str, default="http://127.0.0.1:8765")
    ap.add_argument("--video", type=str, required=True, help="Path on the Lab server machine")
    ap.add_argument(
        "--download-dir",
        type=str,
        default="",
        help="If set, save phase-1 (and tracks) MP4s here after each run",
    )
    ap.add_argument(
        "--open-each",
        action="store_true",
        help="After each successful download on macOS, `open` the phase-1 MP4",
    )
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument(
        "--recipe-ids",
        type=str,
        default="",
        help="Comma-separated subset of §13.3 ids (default: full §13.3 list)",
    )
    args = ap.parse_args()

    base = args.lab_url.rstrip("/")
    health = _http_request("GET", f"{base}/api/health", None, timeout_s=15.0)
    if not isinstance(health, dict) or health.get("ok") != "true":
        raise SystemExit(f"Lab health check failed: {health!r}")

    by_id = _recipes_by_id()
    if args.recipe_ids.strip():
        want = [x.strip() for x in args.recipe_ids.split(",") if x.strip()]
    else:
        want = list(SECTION_133_IDS)
    missing = [x for x in want if x not in by_id]
    if missing:
        raise SystemExit(f"Unknown recipe id(s): {missing}")

    dl_root = Path(args.download_dir).expanduser() if args.download_dir.strip() else None
    if dl_root is not None:
        dl_root.mkdir(parents=True, exist_ok=True)

    video_path = str(Path(args.video).expanduser())
    stem = Path(video_path).stem

    for rid in want:
        r = by_id[rid]
        fields = dict(r.get("fields") or {})
        body = {
            "video_path": video_path,
            "runs": [
                {
                    "recipe_name": r["recipe_name"],
                    "fields": fields,
                    "checkpoint": {"stop_after_boundary": "after_phase_2"},
                }
            ],
            "source_label": f"{stem}_{rid}",
        }
        print(f"\n=== Queue {rid} ({r['recipe_name']}) …", flush=True)
        resp = _http_request("POST", f"{base}/api/runs/batch_path", body, timeout_s=7200.0)
        if not isinstance(resp, dict):
            raise SystemExit(f"Unexpected batch response: {resp!r}")
        run_ids = resp.get("run_ids") or []
        if len(run_ids) != 1:
            raise SystemExit(f"Expected 1 run_id, got {run_ids!r}")
        run_id = str(run_ids[0])
        print(f"    run_id={run_id} — waiting…", flush=True)
        row = _wait_run_terminal(base, run_id, args.poll_interval)
        status = str(row.get("status") or "")
        print(f"    status={status}", flush=True)
        if status != "done":
            err = row.get("error")
            print(f"    error={err!r}", flush=True)
            continue

        rel_phase1 = "phase_previews/00_phase1_detections.mp4"
        rel_tracks = "phase_previews/01_tracks_post_stitch.mp4"
        if dl_root is not None:
            for rel, suffix in ((rel_phase1, "phase1"), (rel_tracks, "tracks_pre_stitch")):
                url = f"{base}/api/runs/{run_id}/file/{rel}"
                try:
                    raw = _http_request("GET", url, None, timeout_s=600.0, binary=True)
                except SystemExit:
                    print(f"    (skip missing {rel})", flush=True)
                    continue
                if not isinstance(raw, bytes) or len(raw) < 100:
                    print(f"    (skip empty {rel})", flush=True)
                    continue
                out = dl_root / f"{rid}_{suffix}.mp4"
                out.write_bytes(raw)
                print(f"    wrote {out} ({len(raw)} bytes)", flush=True)
                if args.open_each and sys.platform == "darwin" and suffix == "phase1":
                    subprocess.run(["open", str(out)], check=False)

        print(f"=== Done {rid}", flush=True)


if __name__ == "__main__":
    main()
