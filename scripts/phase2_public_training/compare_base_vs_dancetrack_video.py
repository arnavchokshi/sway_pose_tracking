#!/usr/bin/env python3
"""
Side-by-side video: DanceTrack fine-tuned YOLO26l vs COCO-pretrained yolo26l.pt.

Requires:
  - models/yolo26l_dancetrack.pt (download from Lambda — see download_lambda_weights.sh)
  - ultralytics, opencv-python

Example:
  cd sway_pose_mvp
  python scripts/phase2_public_training/compare_base_vs_dancetrack_video.py \\
    path/to/clip.mp4 -o /tmp/compare.mp4 --parallel-infer
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "yolo_side_by_side_compare.py"
DEFAULT_FINETUNED = REPO_ROOT / "models" / "yolo26l_dancetrack.pt"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="H-stack DanceTrack fine-tuned YOLO26l (left) vs yolo26l.pt COCO (right)"
    )
    ap.add_argument("video", type=Path, help="Input video")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output .mp4")
    ap.add_argument(
        "--finetuned",
        type=Path,
        default=DEFAULT_FINETUNED,
        help=f"Fine-tuned weights (default: {DEFAULT_FINETUNED})",
    )
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("yolo26l.pt"),
        help="COCO baseline (default: yolo26l.pt — Ultralytics downloads if missing)",
    )
    ap.add_argument("--conf", type=float, default=0.22)
    ap.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference size (training used 960; use 640 for speed)",
    )
    ap.add_argument("--tw", type=int, default=1280, help="Panel width")
    ap.add_argument("--th", type=int, default=720, help="Panel height")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument(
        "--parallel-infer",
        action="store_true",
        help="Run both detectors in parallel threads each frame",
    )
    args = ap.parse_args()

    if not COMPARE_SCRIPT.is_file():
        print(f"Missing: {COMPARE_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    if not args.finetuned.is_file():
        print(
            f"Fine-tuned weights not found: {args.finetuned}\n"
            "  Run:  bash scripts/phase2_public_training/download_lambda_weights.sh",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.video.is_file():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        str(args.video),
        "-o",
        str(args.output),
        "--left-model",
        str(args.finetuned),
        "--right-model",
        str(args.base),
        "--left-title",
        "YOLO26l · DanceTrack FT",
        "--right-title",
        "YOLO26l · COCO base",
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--tw",
        str(args.tw),
        "--th",
        str(args.th),
    ]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.parallel_infer:
        cmd.append("--parallel-infer")

    print("Running:\n ", " ".join(cmd))
    sys.exit(subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode)


if __name__ == "__main__":
    main()
