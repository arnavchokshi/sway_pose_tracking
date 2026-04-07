#!/usr/bin/env python3
"""
Convert pipeline ``data.json`` to MOTChallenge-style prediction lines (file on disk).

Uses ``sway.mot_format.data_json_to_mot_lines`` (``frames[].tracks`` / ``box``).

Usage (from ``sway_pose_mvp/``):

  python -m tools.convert_data_json_to_mot output/run/data.json output/run/pred.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sway.mot_format import data_json_to_mot_lines, write_mot_file


def main() -> None:
    parser = argparse.ArgumentParser(description="data.json → MOT prediction .txt")
    parser.add_argument("data_json", type=Path, help="Path to data.json")
    parser.add_argument("output_txt", type=Path, help="Output path, e.g. pred.txt")
    args = parser.parse_args()
    if not args.data_json.is_file():
        print(f"Error: not found: {args.data_json}", file=sys.stderr)
        sys.exit(2)
    data = json.loads(args.data_json.read_text())
    lines = data_json_to_mot_lines(data)
    write_mot_file(lines, args.output_txt)
    print(f"Wrote {len(lines)} MOT lines to {args.output_txt}")


if __name__ == "__main__":
    main()
