#!/usr/bin/env python3
"""
Isolate ViTPose only (no detection/tracking/pipeline).

Usage (from repo root sway_pose_mvp/):
  python -u tools/vitpose_isolate_smoke.py
  python -u tools/vitpose_isolate_smoke.py --video /path/to/clip.mp4 --boxes 10
  SWAY_VITPOSE_DEBUG=1 python -u tools/vitpose_isolate_smoke.py --masked  # verbose ViTPose logs

Exit 0 if all forwards complete; prints timings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Repo root: sway_pose_mvp/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _frame_synthetic(h: int, w: int) -> "object":
    import numpy as np

    return np.zeros((h, w, 3), dtype=np.uint8) + 40


def _frame_from_video(path: str) -> "object":
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise SystemExit("cannot read first frame")
    return bgr[:, :, ::-1].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="ViTPose isolate smoke test")
    ap.add_argument("--video", type=str, default=None, help="Use first frame of this video (else synthetic)")
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--boxes", type=int, default=10, help="Number of fake person boxes")
    ap.add_argument("--device", type=str, default=None, help="cpu | mps | cuda (default: same as main.get_device)")
    ap.add_argument("--masked", action="store_true", help="Use hybrid SAM path: half plain, half fake masks")
    ap.add_argument("--model", type=str, default="usyd-community/vitpose-plus-base")
    args = ap.parse_args()

    import numpy as np
    import torch

    if args.video:
        print(f"[isolate] loading frame from {args.video}", flush=True)
        t0 = time.perf_counter()
        frame_rgb = _frame_from_video(args.video)
        print(f"[isolate] frame shape={frame_rgb.shape} read in {(time.perf_counter()-t0)*1000:.1f}ms", flush=True)
    else:
        frame_rgb = _frame_synthetic(args.h, args.w)
        print(f"[isolate] synthetic frame {frame_rgb.shape}", flush=True)

    h, w = frame_rgb.shape[:2]
    n = max(1, args.boxes)
    # Spread non-overlapping boxes across frame (rough grid)
    cols = max(1, int(n**0.5))
    bw, bh = max(40, w // (cols + 2)), max(80, h // (cols + 2))
    boxes = []
    track_ids = list(range(1, n + 1))
    for i in range(n):
        r, c = i // cols, i % cols
        x1 = 20 + c * (bw + 10)
        y1 = 20 + r * (bh + 10)
        x2 = min(w - 2, x1 + bw)
        y2 = min(h - 2, y1 + bh)
        boxes.append((float(x1), float(y1), float(x2), float(y2)))
    paddings = [0.15] * n

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[isolate] device={device} torch={torch.__version__}", flush=True)

    from sway.pose_estimator import PoseEstimator, vitpose_debug_enabled, vitpose_effective_max_per_forward

    chunk = vitpose_effective_max_per_forward(device)
    print(f"[isolate] effective vitpose chunk cap={chunk} SWAY_VITPOSE_DEBUG={vitpose_debug_enabled()}", flush=True)

    t_load = time.perf_counter()
    est = PoseEstimator(device=device, model_name=args.model)
    print(f"[isolate] PoseEstimator constructed in {(time.perf_counter()-t_load):.2f}s", flush=True)

    seg = None
    if args.masked:
        seg = []
        for i in range(n):
            if i % 2 == 0:
                seg.append(None)
            else:
                x1, y1, x2, y2 = map(int, boxes[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mh, mw = max(1, y2 - y1), max(1, x2 - x1)
                m = np.ones((mh, mw), dtype=bool)
                seg.append(m)
        n_mask = sum(1 for x in seg if x is not None)
        print(f"[isolate] hybrid masks: plain={n - n_mask} masked={n_mask}", flush=True)

    print("[isolate] calling estimate_poses (this is where pipeline stalls if ViTPose/MPS is the issue)…", flush=True)
    t_inf = time.perf_counter()
    out = est.estimate_poses(frame_rgb, boxes, track_ids, paddings, segmentation_masks=seg)
    dt = time.perf_counter() - t_inf
    print(f"[isolate] estimate_poses OK in {dt:.2f}s n_results={len(out)}", flush=True)
    if len(out) != n:
        print(f"[isolate] WARNING expected {n} poses got {len(out)}", flush=True)
    for tid in sorted(out.keys())[:3]:
        k = out[tid]["keypoints"]
        print(f"[isolate]   tid={tid} keypoints shape={k.shape}", flush=True)
    print("[isolate] done.", flush=True)


if __name__ == "__main__":
    main()
