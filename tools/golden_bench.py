#!/usr/bin/env python3
"""
Optional golden-style check bundle (latency is whatever your machine does for pytest).

This does **not** run from ``main.py``. Invoke explicitly, e.g.:

  python -m tools.golden_bench
  python -m tools.golden_bench --config benchmarks/golden_bench.example.yaml
  python -m tools.golden_bench -- -k pose

See ``docs/PIPELINE_IMPROVEMENTS_ROADMAP.md`` (golden-set benchmarks).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TESTS: List[str] = [
    "tests/test_hybrid_sam_roi.py",
    "tests/test_infer_batch_env.py",
    "tests/test_bidirectional_track_merge.py",
    "tests/test_yolo_runtime_env.py",
    "tests/test_track_stats_export.py",
    "tests/test_golden_bench_config.py",
    "tests/test_validate_pipeline_e2e_helpers.py",
]


def _tests_from_config(config_path: Path) -> List[str]:
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    tests = cfg.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"{config_path}: expected non-empty 'tests' list")
    return [str(x) for x in tests]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a curated pytest subset (opt-in; not part of the video pipeline).",
    )
    parser.add_argument(
        "--list-defaults",
        action="store_true",
        help="Print the default test paths and exit.",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="After pytest, print one JSON line with exit code and wall seconds.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file with 'tests:' list (paths relative to repo root).",
    )
    args, passthrough = parser.parse_known_args()

    if args.list_defaults:
        print(json.dumps({"default_tests": DEFAULT_TESTS}, indent=2))
        return 0

    tests = list(DEFAULT_TESTS)
    if args.config:
        p = Path(args.config).expanduser()
        if not p.is_file():
            print(f"Config not found: {p}", file=sys.stderr)
            return 2
        try:
            tests = _tests_from_config(p.resolve())
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2

    cmd = [sys.executable, "-m", "pytest", *tests, *passthrough]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.perf_counter() - t0
    if args.json_summary:
        print(
            json.dumps(
                {
                    "exit_code": r.returncode,
                    "wall_seconds": round(elapsed, 3),
                    "tests": tests,
                    "config": args.config,
                }
            ),
            flush=True,
        )
    return int(r.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
