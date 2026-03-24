#!/usr/bin/env python3
"""
Trim a few seconds of video and run sway.tracker.run_tracking with hybrid SAM enabled.
Usage:
  python scripts/smoke_hybrid_boxmot_sam_track.py [input.mp4] [--frames 90]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent


def trim_video(src: Path, dst: Path, max_frames: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wri = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not wri.isOpened():
        cap.release()
        raise RuntimeError("VideoWriter failed")
    n = 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        wri.write(fr)
        n += 1
    cap.release()
    wri.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", type=Path, default=None)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--iou-trigger", type=float, default=0.4, help="Higher = fewer SAM calls (faster smoke)")
    args = ap.parse_args()

    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    vid = args.video
    if vid is None:
        cand = Path("/Users/arnavchokshi/Desktop/IMG_2946.MP4")
        vid = cand if cand.is_file() else None
    if vid is None or not vid.is_file():
        print("Provide input.mp4 or place IMG_2946.MP4 on Desktop.", file=sys.stderr)
        sys.exit(1)

    os.environ["SWAY_HYBRID_SAM_OVERLAP"] = "1"
    os.environ["SWAY_HYBRID_SAM_IOU_TRIGGER"] = str(args.iou_trigger)
    os.environ["SWAY_YOLO_DETECTION_STRIDE"] = "1"
    # Shorter chunks OK for smoke
    os.environ["SWAY_CHUNK_SIZE"] = "150"

    clip = REPO / "output" / "_hybrid_smoke_clip.mp4"
    n = trim_video(vid, clip, args.frames)
    print(f"Wrote {clip} ({n} frames)")

    from sway.tracker import run_tracking

    raw, total_frames, out_fps, _, native_fps, fw, fh = run_tracking(str(clip))
    n_tid = len(raw)
    n_obs = sum(len(v) for v in raw.values())
    print(
        f"OK: tracks={n_tid} observations={n_obs} total_frames={total_frames} "
        f"native_fps={native_fps:.2f} size={fw}x{fh}"
    )


if __name__ == "__main__":
    main()
