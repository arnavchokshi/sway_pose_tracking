#!/usr/bin/env python3
"""
Validate recent sweep trials against canary acceptance gates.

Usage examples:
  python -m tools.validate_sweep_canary --status-json output/sweeps/optuna/sweep_status.json
  python -m tools.validate_sweep_canary --status-json /tmp/sweep_status.json --api-log /tmp/pipeline_api.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_status(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SystemExit(f"status json not found: {path}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid status json: {e}") from e


def _recent_complete_trials(status: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    trials = [t for t in (status.get("trials") or []) if t.get("state") == "COMPLETE"]
    trials.sort(key=lambda t: int(t.get("number", -1)))
    return trials[-count:]


def _api_error_count(api_log: Path) -> int:
    txt = api_log.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r'POST /api/optuna-sweep/pull-lambda HTTP/1\.1" (500|502)\b', txt))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate recent sweep canary health gates")
    parser.add_argument("--status-json", type=Path, required=True, help="Path to sweep_status*.json")
    parser.add_argument(
        "--last-n",
        type=int,
        default=3,
        help="Validate the most recent N COMPLETE trials",
    )
    parser.add_argument(
        "--max-trial-duration-s",
        type=float,
        default=1200.0,
        help="Maximum allowed trial duration for canary trials",
    )
    parser.add_argument(
        "--api-log",
        type=Path,
        default=None,
        help="Optional Pipeline Lab API log file; when set, must contain zero 500/502 pull-lambda responses",
    )
    args = parser.parse_args()

    status = _load_status(args.status_json)
    recent = _recent_complete_trials(status, args.last_n)
    if len(recent) < args.last_n:
        raise SystemExit(f"need {args.last_n} complete trials, found {len(recent)}")

    failures: List[str] = []
    required_flags = (
        "engine_mismatch",
        "unknown_runtime_hit",
        "timeout_hit",
        "floor_score_hit",
    )
    for t in recent:
        n = int(t.get("number", -1))
        ua = dict(t.get("user_attrs") or {})
        for k in required_flags:
            if k not in ua:
                failures.append(f"trial {n}: missing user_attr {k}")
        if int(ua.get("engine_mismatch", 0) or 0) != 0:
            failures.append(f"trial {n}: engine_mismatch != 0")
        if int(ua.get("unknown_runtime_hit", 0) or 0) != 0:
            failures.append(f"trial {n}: unknown_runtime_hit != 0")
        if int(ua.get("timeout_hit", 0) or 0) != 0:
            failures.append(f"trial {n}: timeout_hit != 0")
        if int(ua.get("floor_score_hit", 0) or 0) != 0:
            failures.append(f"trial {n}: floor_score_hit != 0")

        dur = t.get("duration_s")
        if isinstance(dur, (int, float)) and float(dur) > args.max_trial_duration_s:
            failures.append(f"trial {n}: duration_s={dur} exceeds {args.max_trial_duration_s}")

    if args.api_log is not None:
        try:
            err_count = _api_error_count(args.api_log)
        except FileNotFoundError as e:
            raise SystemExit(f"api log not found: {args.api_log}") from e
        if err_count != 0:
            failures.append(f"api log has {err_count} pull-lambda 500/502 events")

    if failures:
        print("Canary validation FAILED:")
        for row in failures:
            print(f"- {row}")
        raise SystemExit(1)

    print(
        "Canary validation PASSED "
        f"(trials={len(recent)}, max_duration_s={args.max_trial_duration_s}, "
        f"api_log={'yes' if args.api_log else 'no'})"
    )


if __name__ == "__main__":
    main()

