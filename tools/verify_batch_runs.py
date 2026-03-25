#!/usr/bin/env python3
"""
Wait for all runs in a Pipeline Lab batch to finish, verify success on disk, rerun failures.

  python3 -m tools.verify_batch_runs --batch-id <uuid> --lab-url http://127.0.0.1:8765

Exit 0 only if every run ends as done with run_manifest.json (after optional reruns).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "pipeline_lab" / "runs"
TERMINAL = frozenset({"done", "error", "cancelled"})


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout_s: float = 300.0) -> Any:
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


def _get_runs(base: str) -> List[Dict[str, Any]]:
    out = _http_json("GET", f"{base}/api/runs", None, timeout_s=120.0)
    if not isinstance(out, list):
        raise SystemExit(f"Unexpected /api/runs: {type(out)}")
    return out


def _manifest_ok(run_id: str) -> bool:
    return (RUNS_ROOT / run_id / "run_manifest.json").is_file()


def _wait_batch_terminal(base: str, batch_id: str, expected: int, poll_s: float, max_wait_s: float) -> List[Dict[str, Any]]:
    t0 = time.time()
    last_log = 0.0
    while True:
        rows = [r for r in _get_runs(base) if r.get("batch_id") == batch_id]
        if len(rows) != expected:
            print(f"WARNING: batch has {len(rows)} runs, expected {expected}", flush=True)
        non_term = [r for r in rows if r.get("status") not in TERMINAL]
        if not non_term:
            return rows
        now = time.time()
        if now - last_log >= 40.0:
            from collections import Counter

            c = Counter(r.get("status") for r in rows)
            print(f"poll … non_terminal={len(non_term)} {dict(c)} elapsed_s={now - t0:.0f}", flush=True)
            last_log = now
        if now - t0 > max_wait_s:
            print("TIMEOUT. Non-terminal:", flush=True)
            for r in sorted(non_term, key=lambda x: x.get("recipe_name", "")):
                print(f"  {r.get('status')} {r.get('recipe_name')} {r.get('error')}", flush=True)
            raise SystemExit(2)
        time.sleep(poll_s)


def _wait_run_ids_terminal(base: str, run_ids: Set[str], poll_s: float, max_wait_s: float) -> None:
    t0 = time.time()
    last_log = 0.0
    while True:
        by_id = {r["run_id"]: r for r in _get_runs(base) if r.get("run_id") in run_ids}
        missing = run_ids - set(by_id.keys())
        non_term = [by_id[rid] for rid in run_ids if rid in by_id and by_id[rid].get("status") not in TERMINAL]
        if not non_term and not missing:
            return
        now = time.time()
        if now - last_log >= 40.0:
            print(
                f"rerun poll … non_terminal={len(non_term)} missing_from_api={len(missing)} elapsed_s={now - t0:.0f}",
                flush=True,
            )
            last_log = now
        if now - t0 > max_wait_s:
            raise SystemExit("TIMEOUT waiting for rerun jobs")
        time.sleep(poll_s)


def _rerun(base: str, run_id: str) -> str:
    req = urllib.request.Request(
        f"{base}/api/runs/{run_id}/rerun",
        method="POST",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code} rerun: {detail[:2000]}") from e
    resp = json.loads(body) if body.strip() else {}
    if not isinstance(resp, dict) or not resp.get("run_id"):
        raise SystemExit(f"Bad rerun response: {resp!r}")
    return str(resp["run_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--lab-url", default="http://127.0.0.1:8765")
    ap.add_argument("--expected", type=int, default=23)
    ap.add_argument("--poll-s", type=float, default=20.0)
    ap.add_argument("--max-wait-s", type=float, default=4 * 3600.0)
    ap.add_argument("--no-rerun", action="store_true", help="Do not POST /rerun for error rows")
    args = ap.parse_args()
    base = args.lab_url.rstrip("/")

    h = _http_json("GET", f"{base}/api/health", None, timeout_s=15.0)
    if not isinstance(h, dict) or h.get("ok") != "true":
        raise SystemExit(f"Lab not healthy: {h!r}")

    print(f"Waiting for batch {args.batch_id} ({args.expected} runs) …", flush=True)
    rows = _wait_batch_terminal(base, args.batch_id, args.expected, args.poll_s, args.max_wait_s)

    done = [r for r in rows if r.get("status") == "done"]
    err = [r for r in rows if r.get("status") == "error"]
    oth = [r for r in rows if r.get("status") not in ("done", "error")]

    print(f"Terminal: done={len(done)} error={len(err)} other={len(oth)}", flush=True)
    for r in sorted(err, key=lambda x: x.get("recipe_name", "")):
        print(f"  ERROR {r.get('recipe_name')}: {r.get('error')}", flush=True)
    for r in sorted(oth, key=lambda x: x.get("recipe_name", "")):
        print(f"  {r.get('status')} {r.get('recipe_name')}", flush=True)

    bad_manifest: List[str] = []
    for r in done:
        rid = r["run_id"]
        if not _manifest_ok(rid):
            bad_manifest.append(rid)
            print(f"  WARN done but no run_manifest.json: {rid} {r.get('recipe_name')}", flush=True)

    if bad_manifest:
        raise SystemExit(f"{len(bad_manifest)} runs marked done but missing run_manifest.json")

    if err and args.no_rerun:
        raise SystemExit(1)

    if err:
        print(f"Rerunning {len(err)} failed jobs …", flush=True)
        new_ids: Set[str] = set()
        for r in err:
            rid = r["run_id"]
            print(f"  POST rerun {rid} ({r.get('recipe_name')})", flush=True)
            new_ids.add(_rerun(base, rid))

        _wait_run_ids_terminal(base, new_ids, args.poll_s, args.max_wait_s)

        still_bad: List[str] = []
        for nid in sorted(new_ids):
            row = next((x for x in _get_runs(base) if x.get("run_id") == nid), None)
            st = row.get("status") if row else "missing"
            if st != "done" or not _manifest_ok(nid):
                still_bad.append(f"{nid} status={st} manifest={_manifest_ok(nid)}")
                print(f"  FAIL after rerun: {nid} status={st}", flush=True)

        if still_bad:
            raise SystemExit(f"Rerun failures: {still_bad}")

    print("OK: all batch runs completed successfully (done + run_manifest.json).", flush=True)


if __name__ == "__main__":
    main()
