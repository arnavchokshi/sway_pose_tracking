#!/usr/bin/env python3
"""
TrackEval CLI: MOT-format GT vs pipeline data.json (IDF1, IDSW, HOTA, …).

Ground-truth YAML may include:

  trackeval:
    gt_mot_file: benchmarks/my_seq_gt.txt   # MOTChallenge rows, 1-based frames, class=1
    sequence_name: my_seq
    im_width: 1920
    im_height: 1080

Or pass paths explicitly:

  python benchmark_trackeval.py --gt-mot benchmarks/gt.txt --json output/data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="TrackEval: MOT GT vs data.json")
    parser.add_argument("--ground-truth", type=Path, help="YAML with trackeval.gt_mot_file")
    parser.add_argument("--gt-mot", type=Path, help="Direct path to MOT gt.txt")
    parser.add_argument("--json", type=Path, required=True, help="Pipeline data.json")
    parser.add_argument("--sequence-name", type=str, default="swayseq")
    parser.add_argument("--im-width", type=int, default=1920)
    parser.add_argument("--im-height", type=int, default=1080)
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPT_DIR))
    from sway.mot_format import load_mot_lines_from_file
    from sway.mot_format import data_json_to_mot_lines
    from sway.trackeval_runner import run_trackeval_single_sequence

    gt_path = args.gt_mot
    seq = args.sequence_name
    w, h = args.im_width, args.im_height
    if args.ground_truth:
        with open(args.ground_truth) as f:
            gt_yaml = yaml.safe_load(f) or {}
        te = gt_yaml.get("trackeval") or {}
        if not gt_path:
            g = te.get("gt_mot_file")
            gt_path = Path(g) if g else None
        seq = te.get("sequence_name", seq)
        w = int(te.get("im_width", w))
        h = int(te.get("im_height", h))

    if not gt_path or not gt_path.is_file():
        print("Error: provide --gt-mot or --ground-truth with trackeval.gt_mot_file", file=sys.stderr)
        sys.exit(2)

    with open(args.json) as f:
        data = json.load(f)

    gt_lines = load_mot_lines_from_file(gt_path)
    pred_lines = data_json_to_mot_lines(data)
    if not gt_lines:
        print("Error: empty GT MOT file", file=sys.stderr)
        sys.exit(2)

    try:
        metrics = run_trackeval_single_sequence(gt_lines, pred_lines, seq, w, h)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(3)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
