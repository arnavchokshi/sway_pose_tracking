#!/usr/bin/env python3
"""
Side-by-side two pose MP4s (1280x720 each) with burned-in labels — no ffmpeg drawtext required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def letterbox(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def draw_corner_label(img: np.ndarray, title: str, subtitle: str) -> None:
    """Dark bar top-left + white text."""
    h, w = img.shape[:2]
    bar_w = min(w - 16, 780)
    cv2.rectangle(img, (8, 8), (8 + bar_w, 102), (25, 25, 25), -1)
    cv2.putText(
        img,
        title,
        (20, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        1.05,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        subtitle,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="H-stack two videos with labels")
    ap.add_argument("left_video", type=Path, help="Left panel (e.g. BoT-SORT)")
    ap.add_argument("right_video", type=Path, help="Right panel (e.g. BoxMOT)")
    ap.add_argument("output", type=Path)
    ap.add_argument("--tw", type=int, default=1280)
    ap.add_argument("--th", type=int, default=720)
    args = ap.parse_args()

    cap_l = cv2.VideoCapture(str(args.left_video))
    cap_r = cv2.VideoCapture(str(args.right_video))
    if not cap_l.isOpened() or not cap_r.isOpened():
        print("Error: could not open inputs", file=sys.stderr)
        sys.exit(1)

    fps = cap_l.get(cv2.CAP_PROP_FPS) or cap_r.get(cv2.CAP_PROP_FPS) or 30.0
    out_w, out_h = args.tw * 2, args.th
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print("Error: could not open VideoWriter", file=sys.stderr)
        sys.exit(1)

    left_title = "BoT-SORT"
    left_sub = "SWAY_USE_BOXMOT=0"
    right_title = "BoxMOT"
    right_sub = "Default (Deep OC-SORT)"

    while True:
        ok_l, fr_l = cap_l.read()
        ok_r, fr_r = cap_r.read()
        if not ok_l or not ok_r:
            break
        p_l = letterbox(fr_l, args.tw, args.th)
        p_r = letterbox(fr_r, args.tw, args.th)
        draw_corner_label(p_l, left_title, left_sub)
        draw_corner_label(p_r, right_title, right_sub)
        combo = np.hstack([p_l, p_r])
        writer.write(combo)

    cap_l.release()
    cap_r.release()
    writer.release()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
