#!/usr/bin/env python3
"""
Refresh ``sweep_status.json`` from Optuna storage (no running sweep required).

  python -m tools.export_optuna_study_status
  python -m tools.export_optuna_study_status --study-name sway_phase13_v1 --output /tmp/status.json

Use from your laptop after ``scp``, or on the instance via cron, while a sweep runs in another process.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Optuna study to JSON snapshot")
    parser.add_argument("--study-name", type=str, default="sway_phase13_v1")
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optuna storage URL (default: sqlite under output/sweeps/optuna/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: output/sweeps/optuna/sweep_status.json)",
    )
    args = parser.parse_args()

    import optuna

    from sway.optuna_live_status import write_live_sweep_status

    opt_dir = REPO_ROOT / "output" / "sweeps" / "optuna"
    storage = args.storage or f"sqlite:///{(opt_dir / 'sweep.db').resolve()}"
    out = args.output or (opt_dir / "sweep_status.json")

    study = optuna.load_study(study_name=args.study_name, storage=storage)
    write_live_sweep_status(
        study,
        out,
        extra={"source": "export_optuna_study_status", "storage": storage},
    )
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
