#!/usr/bin/env python3
"""
Optional Sapiens qualitative spot-check — separate from the main pipeline.

For **in-pipeline** native Sapiens (COCO-17), set ``SWAY_SAPIENS_TORCHSCRIPT`` to a lite
``*_coco_*_torchscript.pt2`` and use ``--pose-model sapiens`` (see ``sway/sapiens_estimator.py``).
This script remains an independent opt-in stub.

Install and experiment in a separate venv per vendor docs; keep this repo’s default
pipeline unchanged. When Meta’s Sapiens (or your fork) is importable, extend this
script to load frames and run inference.

Usage (today: prints guidance only unless you pass --yes-i-know):

  python -m tools.sapiens_qualitative_smoke --video path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Optional Sapiens qualitative hook (no default pipeline integration).",
    )
    ap.add_argument("--video", type=str, default="", help="Video path (for future inference).")
    ap.add_argument(
        "--yes-i-know",
        action="store_true",
        help="Acknowledge this is a stub; still only prints next steps unless you add code.",
    )
    args = ap.parse_args()

    if os.environ.get("SWAY_SAPIENS_SMOKE", "").strip().lower() not in ("1", "true", "yes"):
        print(
            "Sapiens smoke is opt-in. Set SWAY_SAPIENS_SMOKE=1 for this script to proceed, "
            "or read docs/PIPELINE_IMPROVEMENTS_ROADMAP.md.",
            file=sys.stderr,
        )
        return 2

    if not args.yes_i_know:
        print(
            "Stub only: add your Sapiens install + inference here. "
            "Re-run with --yes-i-know after reading vendor license and weights terms.",
            file=sys.stderr,
        )
        return 2

    v = Path(args.video) if args.video else None
    if v is not None and not v.is_file():
        print(f"Video not found: {v}", file=sys.stderr)
        return 1

    print(
        "SWAY_SAPIENS_SMOKE=1 and --yes-i-know set. "
        "Hook your checkpoint and dataloader here; pipeline main.py remains unchanged.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
