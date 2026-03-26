#!/usr/bin/env python3
"""
Re-queue **failed** Pipeline Lab runs for a given batch_id without touching successful jobs.

Reads each run's ``pipeline_lab/runs/<uuid>/request.json`` (same host as uvicorn), builds one
``POST /api/runs/batch_path`` with the original ``source_path``, ``checkpoint``, fields, and
recipe names. Successful runs are never modified unless you pass ``--delete-failed`` (only
deletes runs you are about to replace — status error/cancelled, after the spec is captured).

Typical use after a tree stage fails (e.g. final stage fixed in ``main.py``)::

  cd sway_pose_mvp
  python3 -m tools.retry_failed_batch_runs \\
    --batch-id c6ae59e0-a413-4a61-bf4b-22b04d64b7d1

Dry-run::

  python3 -m tools.retry_failed_batch_runs --batch-id ... --dry-run

Then open the Lab with ``/?batch=<batch_id>`` (same id) to watch the new jobs.
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
RUNS_ROOT = REPO_ROOT / "pipeline_lab" / "runs"


def _http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 600.0,
) -> Any:
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


def _wait_terminal(
    base: str,
    run_ids: List[str],
    *,
    poll_s: float,
) -> Tuple[List[str], List[str]]:
    pending = set(run_ids)
    ok: List[str] = []
    bad: List[str] = []
    while pending:
        rows = _http_json("GET", f"{base}/api/runs", None, timeout_s=120.0)
        if not isinstance(rows, list):
            time.sleep(poll_s)
            continue
        by_id = {str(r.get("run_id")): r for r in rows if isinstance(r, dict) and r.get("run_id")}
        for rid in list(pending):
            row = by_id.get(rid)
            if not row:
                continue
            st = str(row.get("status") or "")
            if st in ("done", "error", "cancelled"):
                pending.discard(rid)
                if st == "done":
                    ok.append(rid)
                else:
                    bad.append(rid)
        if pending:
            print(f"  … {len(pending)} run(s) still queued/running", flush=True)
            time.sleep(poll_s)
    return ok, bad


def _load_request(run_id: str) -> Dict[str, Any]:
    p = RUNS_ROOT / run_id / "request.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing request.json for {run_id}: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _disk_incomplete_ids(
    batch_id: str,
    recipe_prefix: str,
    *,
    require_params_yaml: bool,
) -> List[str]:
    """Runs with this batch_id + checkpoint + recipe, but no run_manifest.json (never finished).

    By default ``require_params_yaml=True``: only dirs where the worker already wrote
    ``params.yaml`` (main.py was started at least once). That excludes fresh
    ``batch_path`` jobs still waiting in the Lab queue — re-batching those would duplicate work.
    """
    out: List[str] = []
    if not RUNS_ROOT.is_dir():
        return out
    for p in RUNS_ROOT.iterdir():
        if not p.is_dir() or p.name.startswith("_"):
            continue
        req_path = p / "request.json"
        if not req_path.is_file():
            continue
        try:
            meta = json.loads(req_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(meta.get("batch_id") or "").strip() != batch_id:
            continue
        name = str(meta.get("recipe_name") or "")
        if recipe_prefix and not name.startswith(recipe_prefix):
            continue
        ck = meta.get("checkpoint")
        if not isinstance(ck, dict) or not ck:
            continue
        if (p / "run_manifest.json").is_file():
            continue
        if require_params_yaml and not (p / "params.yaml").is_file():
            continue
        out.append(p.name)
    return sorted(out)


def _filter_not_still_queued(base: str, run_ids: List[str]) -> Tuple[List[str], int, int]:
    """Drop runs that are *actually* waiting in the Lab queue or have a live main.py Popen.

    Do **not** drop ``status=queued`` when ``params.yaml`` exists: that is a stale row after a
    crash or uvicorn restart (worker already ran main.py once; safe to re-batch).

    Returns ``(keep, n_skip_live, n_skip_queued_no_params)``.
    """
    keep: List[str] = []
    n_live = 0
    n_q_np = 0
    for rid in run_ids:
        try:
            row = _http_json("GET", f"{base}/api/runs/{rid}", None, timeout_s=60.0)
        except SystemExit:
            keep.append(rid)
            continue
        if not isinstance(row, dict):
            keep.append(rid)
            continue
        st = str(row.get("status") or "")
        if st == "running" and row.get("subprocess_alive") is True:
            n_live += 1
            continue
        if st == "queued":
            params = RUNS_ROOT / rid / "params.yaml"
            if not params.is_file():
                n_q_np += 1
                continue
        keep.append(rid)
    return keep, n_live, n_q_np


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-queue failed Lab runs for a batch (reads runs/*/request.json on disk)",
    )
    ap.add_argument(
        "--batch-id",
        required=True,
        help="Only runs whose request.json batch_id matches (same ?batch= filter)",
    )
    ap.add_argument(
        "--lab-url",
        default="http://localhost:8765",
        help="Lab API base URL",
    )
    ap.add_argument(
        "--ui-origin",
        default=None,
        help="Origin for printed Lab links (defaults to --lab-url). Set for Vite dev, e.g. http://localhost:5173",
    )
    ap.add_argument(
        "--recipe-prefix",
        default="tree_final",
        help="Only retry runs whose recipe_name starts with this (default: final tree stage). "
        "Use empty string: --recipe-prefix '' to match any recipe in the batch.",
    )
    ap.add_argument(
        "--any-recipe",
        action="store_true",
        help="Ignore recipe name (same as --recipe-prefix '')",
    )
    ap.add_argument(
        "--include-cancelled",
        action="store_true",
        help="Also retry runs in status cancelled (default: error only)",
    )
    ap.add_argument(
        "--disk-incomplete",
        action="store_true",
        help="Find runs by scanning pipeline_lab/runs/*/request.json: same batch_id + recipe "
        "prefix + checkpoint, but no run_manifest.json. By default only dirs that already have "
        "params.yaml (pipeline was dequeued once). Use after a uvicorn restart when failed "
        "jobs no longer show status=error. API filter skips a live main.py Popen.",
    )
    ap.add_argument(
        "--include-never-started",
        action="store_true",
        help="With --disk-incomplete: also match dirs with no params.yaml (still in Lab queue). "
        "Risky: can duplicate jobs unless the worker queue is empty; prefer letting the worker "
        "drain or POST /api/runs/rehydrate_from_disk.",
    )
    ap.add_argument(
        "--no-api-filter",
        action="store_true",
        help="With --disk-incomplete: do not call GET /api/runs/<id> (retry every incomplete "
        "folder; stop uvicorn first if you need zero duplicates).",
    )
    ap.add_argument(
        "--list-batch",
        action="store_true",
        help="Print all runs (API) with this batch_id and exit (debug)",
    )
    ap.add_argument(
        "--delete-failed",
        action="store_true",
        help="After reading each spec, DELETE that failed run dir via API (frees clutter; "
        "parent checkpoints are untouched)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print plan only; no POST/DELETE")
    ap.add_argument(
        "--wait",
        action="store_true",
        help="After enqueue, poll until all new runs finish (done/error/cancelled)",
    )
    ap.add_argument("--poll", type=float, default=10.0, help="Seconds between polls with --wait")
    args = ap.parse_args()
    recipe_prefix = "" if args.any_recipe else str(args.recipe_prefix)

    base = args.lab_url.rstrip("/")
    ui = (args.ui_origin or args.lab_url).rstrip("/")
    batch_id = args.batch_id.strip()
    want_status = {"error"}
    if args.include_cancelled:
        want_status.add("cancelled")

    rows = _http_json("GET", f"{base}/api/runs", None, timeout_s=120.0)
    if not isinstance(rows, list):
        raise SystemExit("GET /api/runs did not return a list")

    if args.list_batch:
        hits = [
            r
            for r in rows
            if isinstance(r, dict) and str(r.get("batch_id") or "").strip() == batch_id
        ]
        print(f"batch_id={batch_id!r}: {len(hits)} run(s) in API list")
        for r in sorted(hits, key=lambda x: (str(x.get("recipe_name")), str(x.get("run_id")))):
            print(
                f"  {str(r.get('run_id'))[:8]}…  status={r.get('status')!r}  "
                f"recipe={r.get('recipe_name')!r}"
            )
        return

    candidates: List[str] = []
    if args.disk_incomplete:
        require_params = not args.include_never_started
        candidates = _disk_incomplete_ids(
            batch_id, recipe_prefix, require_params_yaml=require_params
        )
        if require_params:
            wider = _disk_incomplete_ids(
                batch_id, recipe_prefix, require_params_yaml=False
            )
            n_no_params = sum(
                1
                for rid in wider
                if not (RUNS_ROOT / rid / "params.yaml").is_file()
            )
            if n_no_params:
                print(
                    f"  (note: {n_no_params} incomplete run dir(s) have no params.yaml — still "
                    "in the Lab queue; not selected for re-batch. Use --include-never-started "
                    "only if those jobs are stuck.)",
                    flush=True,
                )
        if not args.no_api_filter:
            before = len(candidates)
            candidates, n_sk_live, n_sk_q = _filter_not_still_queued(base, candidates)
            dropped = before - len(candidates)
            if dropped:
                parts = []
                if n_sk_live:
                    parts.append(f"{n_sk_live} still running (live main.py)")
                if n_sk_q:
                    parts.append(f"{n_sk_q} API-queued without params.yaml on disk")
                print(f"  (skipped {dropped}: " + "; ".join(parts) + ")", flush=True)
    else:
        for r in rows:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("run_id") or "").strip()
            if not rid:
                continue
            if str(r.get("batch_id") or "").strip() != batch_id:
                continue
            st = str(r.get("status") or "")
            if st not in want_status:
                continue
            name = str(r.get("recipe_name") or "")
            if recipe_prefix and not name.startswith(recipe_prefix):
                continue
            candidates.append(rid)

    if not candidates:
        print(f"No matching failed runs (batch_id={batch_id!r}, prefix={recipe_prefix!r}).")
        print(
            "Tip: after restarting uvicorn, failed jobs may not show status=error. Try:\n"
            f"  python3 -m tools.retry_failed_batch_runs --batch-id {batch_id!r} --disk-incomplete\n"
            "If tree_final jobs are only queued (no params.yaml), let the worker drain or call "
            "POST /api/runs/rehydrate_from_disk — do not re-batch unless stuck.\n"
            "Or list what the API thinks is in this batch:\n"
            f"  python3 -m tools.retry_failed_batch_runs --batch-id {batch_id!r} --list-batch",
        )
        return

    candidates.sort()
    specs: List[Dict[str, Any]] = []
    rids_to_replace: List[str] = []
    video_path: Optional[str] = None
    source_label = ""

    for rid in candidates:
        try:
            meta = _load_request(rid)
        except FileNotFoundError as e:
            print(f"  skip {rid[:8]}… — {e}", flush=True)
            continue
        mb = str(meta.get("batch_id") or "").strip()
        if mb and mb != batch_id:
            print(f"  skip {rid[:8]}… — request batch_id mismatch", flush=True)
            continue
        ck = meta.get("checkpoint")
        if not isinstance(ck, dict) or not ck:
            print(f"  skip {rid[:8]}… — no checkpoint in request.json (not a resume job)", flush=True)
            continue
        vp = str(meta.get("source_path") or "").strip()
        if not vp:
            print(f"  skip {rid[:8]}… — no source_path in request.json", flush=True)
            continue
        if video_path is None:
            video_path = vp
        elif vp != video_path:
            raise SystemExit(
                f"Runs use different source_path — cannot batch in one request:\n  {video_path}\n  {vp}"
            )
        if not source_label:
            sl = str(meta.get("video_stem") or "").strip()
            if sl:
                source_label = sl

        specs.append(
            {
                "recipe_name": str(meta.get("recipe_name") or "retry"),
                "fields": dict(meta.get("fields") or {}) if isinstance(meta.get("fields"), dict) else {},
                "checkpoint": dict(ck),
            }
        )
        rids_to_replace.append(rid)

    if not specs:
        raise SystemExit("No runnable specs after reading request.json (all skipped).")

    if not Path(video_path or "").is_file():
        raise SystemExit(f"video_path does not exist on this machine: {video_path!r}")

    print(f"Retry {len(specs)} run(s) for batch {batch_id!r} (video {video_path!r})", flush=True)
    if args.dry_run:
        for i, sp in enumerate(specs[:5]):
            print(f"  [{i+1}] {sp['recipe_name']!r} resume={sp['checkpoint'].get('resume_from', '')!r}")
        if len(specs) > 5:
            print(f"  … and {len(specs) - 5} more")
        print("Dry run — no POST.")
        return

    if args.delete_failed:
        for rid in rids_to_replace:
            try:
                req = urllib.request.Request(f"{base}/api/runs/{rid}", method="DELETE")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    code = resp.status
                if code not in (204, 404):
                    print(f"  DELETE {rid[:8]}… HTTP {code}", flush=True)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")[:300]
                print(f"  DELETE {rid[:8]}… HTTP {e.code} {body}", flush=True)

    body: Dict[str, Any] = {
        "video_path": str(Path(video_path).resolve()),
        "runs": specs,
        "batch_id": batch_id,
    }
    if source_label:
        body["source_label"] = source_label

    resp = _http_json("POST", f"{base}/api/runs/batch_path", body, timeout_s=900.0)
    if not isinstance(resp, dict):
        raise SystemExit(f"Unexpected batch_path response: {resp!r}")
    new_ids = resp.get("run_ids")
    if not isinstance(new_ids, list):
        raise SystemExit("batch_path missing run_ids")
    print(f"Queued {len(new_ids)} run(s). batch_id={batch_id!r}", flush=True)
    print(f"  Lab home: {ui}/", flush=True)
    print(f"  Optional deep link: {ui}/?batch={batch_id}", flush=True)

    if args.wait:
        print("Waiting for new runs to finish…", flush=True)
        ok, bad = _wait_terminal(base, [str(x) for x in new_ids], poll_s=args.poll)
        print(f"Done: {len(ok)} succeeded, {len(bad)} failed.", flush=True)


if __name__ == "__main__":
    main()
