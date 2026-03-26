#!/usr/bin/env python3
"""
Queue the curated pipeline matrix against a server-side video via the Pipeline Lab API.

Requires the Lab API (uvicorn) running; the video path must exist on that machine.

  cd sway_pose_mvp
  uvicorn pipeline_lab.server.app:app --host localhost --port 8765

  python -m tools.pipeline_matrix_runs --video /path/to/clip.mp4
  python -m tools.pipeline_matrix_runs --video /path/to/clip.mp4 --lab-url http://localhost:8765
  python -m tools.pipeline_matrix_runs --video clip.mp4 --only baseline,tracker_deep_ocsort_osnet,sam_off
  python -m tools.pipeline_matrix_runs --dry-run

Optional: ``pip install httpx`` for a friendlier HTTP client; otherwise stdlib ``urllib`` is used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sway.pipeline_matrix_presets import PIPELINE_MATRIX_RECIPES  # noqa: E402


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout_s: float = 120.0) -> Any:
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
        raise SystemExit(f"HTTP {e.code} {url}: {detail[:800]}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Request failed {url}: {e}") from e
    if not body.strip():
        return None
    return json.loads(body)


def _recipes_subset(only: Optional[str]) -> List[Dict[str, Any]]:
    if not only or not only.strip():
        return list(PIPELINE_MATRIX_RECIPES)
    want: Set[str] = {x.strip() for x in only.split(",") if x.strip()}
    out = [r for r in PIPELINE_MATRIX_RECIPES if str(r.get("id")) in want]
    if len(out) != len(want):
        known = {str(r.get("id")) for r in PIPELINE_MATRIX_RECIPES}
        bad = want - known
        if bad:
            raise SystemExit(f"Unknown recipe id(s): {sorted(bad)}. Known: {sorted(known)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Queue pipeline matrix batch via Lab API")
    ap.add_argument(
        "--video",
        type=str,
        default="",
        help="Absolute or relative path on the API server machine (required unless --dry-run)",
    )
    ap.add_argument(
        "--lab-url",
        type=str,
        default="http://localhost:8765",
        help="Pipeline Lab base URL (no trailing slash)",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated recipe ids (default: all). Example: baseline,tracker_deep_ocsort_osnet,sam_off",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print payload only; do not POST")
    ap.add_argument(
        "--wait",
        action="store_true",
        help="Poll /api/runs until all run_ids in this batch are terminal (done/error/cancelled)",
    )
    ap.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polls with --wait")
    args = ap.parse_args()

    recipes = _recipes_subset(args.only or None)
    runs_payload = [{"recipe_name": r["recipe_name"], "fields": dict(r.get("fields") or {})} for r in recipes]

    base = args.lab_url.rstrip("/")
    health = _http_json("GET", f"{base}/api/health", None, timeout_s=10.0)
    if not isinstance(health, dict) or health.get("ok") != "true":
        raise SystemExit(f"Lab health check failed at {base}/api/health: {health!r}")

    body = {
        "video_path": str(Path(args.video).expanduser()) if args.video else "",
        "runs": runs_payload,
        "source_label": Path(args.video).stem if args.video else "",
    }

    if args.dry_run:
        print(json.dumps({"endpoint": f"{base}/api/runs/batch_path", "body": body}, indent=2))
        return

    if not args.video:
        raise SystemExit("--video is required (or use --dry-run)")

    print(f"POST {base}/api/runs/batch_path ({len(runs_payload)} runs)…")
    resp = _http_json("POST", f"{base}/api/runs/batch_path", body, timeout_s=600.0)
    if not isinstance(resp, dict):
        raise SystemExit(f"Unexpected response: {resp!r}")
    print(json.dumps(resp, indent=2))

    run_ids = resp.get("run_ids")
    if args.wait and isinstance(run_ids, list) and run_ids:
        terminal = {"done", "error", "cancelled"}
        pending = {str(x) for x in run_ids}
        print(f"Waiting for {len(pending)} runs (--wait)…")
        while pending:
            time.sleep(max(0.5, float(args.poll_interval)))
            rows = _http_json("GET", f"{base}/api/runs", None, timeout_s=60.0)
            if not isinstance(rows, list):
                continue
            by_id = {str(r.get("run_id")): r for r in rows if isinstance(r, dict)}
            for rid in list(pending):
                row = by_id.get(rid)
                if row and str(row.get("status") or "") in terminal:
                    print(f"  {rid[:8]}… → {row.get('status')}")
                    pending.discard(rid)
            if pending:
                print(f"  … {len(pending)} still running/queued")


if __name__ == "__main__":
    main()
