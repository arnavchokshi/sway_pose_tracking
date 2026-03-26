#!/usr/bin/env python3
"""
CLI for Pipeline Lab experiments / checkpoint tree API.

  cd sway_pose_mvp
  uvicorn pipeline_lab.server.app:app --host localhost --port 8765

  python -m tools.pipeline_experiments list
  python -m tools.pipeline_experiments create --label "my branch study"
  python -m tools.pipeline_experiments get <experiment_id>
  python -m tools.pipeline_experiments append-node <experiment_id> --run-id <uuid> --boundary after_phase_3 --label "variant A"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout_s: float = 60.0) -> Any:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline Lab experiments API client")
    ap.add_argument("--lab-url", type=str, default="http://localhost:8765", help="Lab API base URL")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List experiments")

    p_create = sub.add_parser("create", help="Create an experiment graph")
    p_create.add_argument("--label", type=str, default="")

    p_get = sub.add_parser("get", help="Fetch graph.json for an experiment")
    p_get.add_argument("experiment_id")

    p_node = sub.add_parser("append-node", help="Append a node (and optional edge) to graph.json")
    p_node.add_argument("experiment_id")
    p_node.add_argument("--run-id", type=str, required=True)
    p_node.add_argument("--parent-run-id", type=str, default="")
    p_node.add_argument("--boundary", type=str, default="")
    p_node.add_argument("--label", type=str, default="")

    args = ap.parse_args()
    base = args.lab_url.rstrip("/")

    if args.cmd == "list":
        print(json.dumps(_http_json("GET", f"{base}/api/experiments"), indent=2))
    elif args.cmd == "create":
        print(
            json.dumps(
                _http_json("POST", f"{base}/api/experiments", {"label": args.label}),
                indent=2,
            )
        )
    elif args.cmd == "get":
        print(json.dumps(_http_json("GET", f"{base}/api/experiments/{args.experiment_id}"), indent=2))
    elif args.cmd == "append-node":
        pl: Dict[str, Any] = {
            "run_id": args.run_id,
            "boundary": args.boundary,
            "label": args.label,
        }
        if args.parent_run_id:
            pl["parent_run_id"] = args.parent_run_id
        print(
            json.dumps(
                _http_json("POST", f"{base}/api/experiments/{args.experiment_id}/nodes", pl),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
