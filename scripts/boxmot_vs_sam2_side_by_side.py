#!/usr/bin/env python3
"""
Side-by-side video: YOLO detections + BoxMOT DeepOCSORT (tracks) vs same YOLO dets + SAM2 masks.

YOLO runs once per frame (shared). Timings are accumulated with perf_counter and written to JSON.

Requires: ultralytics, boxmot, opencv, torch (see repo requirements.txt).
SAM: default Ultralytics SAM2.1 base weights (sam2.1_b.pt).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM, YOLO

# Repo imports when run from sway_pose_mvp/
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sway.boxmot_compat import apply_boxmot_kf_unfreeze_guard
from sway.hybrid_sam_refiner import resolve_hybrid_sam_weights
from sway.tracker import (
    diou_nms_indices,
    load_tracking_runtime,
    resolve_yolo_model_path,
    _resolve_boxmot_reid_weights,
)


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
    h, w = img.shape[:2]
    bar_w = min(w - 16, 780)
    cv2.rectangle(img, (8, 8), (8 + bar_w, 102), (25, 25, 25), -1)
    cv2.putText(
        img, title, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        img, subtitle, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1, cv2.LINE_AA
    )


def _hsv_bgr(i: int) -> tuple[int, int, int]:
    h = (i * 47) % 180
    hsv = np.uint8([[[h, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_boxmot_panel(
    frame_bgr: np.ndarray,
    tracker_out: np.ndarray | None,
    yconf: float,
) -> np.ndarray:
    out = frame_bgr.copy()
    if tracker_out is None or len(tracker_out) == 0:
        return out
    for row in np.atleast_2d(tracker_out):
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        tid = int(row[4])
        if tid < 0:
            continue
        cf = float(row[5]) if len(row) > 5 else yconf
        c = _hsv_bgr(tid)
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), c, 2, cv2.LINE_AA)
        lab = f"id{tid} {cf:.2f}"
        cv2.putText(
            out, lab, (x1i, max(18, y1i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA
        )
    return out


def draw_sam_panel(
    frame_bgr: np.ndarray,
    masks_tensor,
) -> np.ndarray:
    out = frame_bgr.astype(np.float32)
    if masks_tensor is None or masks_tensor.shape[0] == 0:
        return frame_bgr.copy()
    mh, mw = frame_bgr.shape[:2]
    for i in range(masks_tensor.shape[0]):
        m = masks_tensor[i].detach().cpu().numpy()
        if m.shape != (mh, mw):
            m = cv2.resize(m.astype(np.float32), (mw, mh), interpolation=cv2.INTER_NEAREST)
        mb = m > 0.5
        col = np.array(_hsv_bgr(i + 3), dtype=np.float32)
        out[mb] = out[mb] * 0.45 + col * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Side-by-side BoxMOT DeepOCSORT vs SAM2 (Ultralytics), shared YOLO26l per frame"
    )
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output .mp4")
    ap.add_argument(
        "--timing-json",
        type=Path,
        default=None,
        help="Write timing breakdown JSON (default: next to output with _timing.json)",
    )
    ap.add_argument("--yolo", type=str, default="yolo26l.pt", help="YOLO weights (or set SWAY_YOLO_WEIGHTS)")
    ap.add_argument("--sam", type=str, default="sam2.1_b.pt", help="Ultralytics SAM2.1 checkpoint")
    ap.add_argument("--tw", type=int, default=1280, help="Panel width after letterbox")
    ap.add_argument("--th", type=int, default=720, help="Panel height after letterbox")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = full video")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    import os

    # Full overlays every frame (stride>1 would leave SAM/boxmot panels stale).
    os.environ["SWAY_YOLO_DETECTION_STRIDE"] = "1"

    if args.yolo and not os.environ.get("SWAY_YOLO_WEIGHTS"):
        p = Path(args.yolo).expanduser()
        if p.is_file():
            os.environ["SWAY_YOLO_WEIGHTS"] = str(p.resolve())
        else:
            os.environ["SWAY_YOLO_WEIGHTS"] = args.yolo

    tr = load_tracking_runtime()
    yconf = float(tr["yolo_conf"])
    ystride = int(tr["yolo_stride"])
    base_detect = int(tr["detect_size"])
    model_path = resolve_yolo_model_path()

    timing_path = args.timing_json
    if timing_path is None:
        timing_path = args.output.with_name(args.output.stem + "_timing.json")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    apply_boxmot_kf_unfreeze_guard()
    from boxmot import DeepOcSort

    print(f"Loading YOLO: {model_path}")
    yolo = YOLO(model_path)
    sam_w = resolve_hybrid_sam_weights(args.sam)
    print(f"Loading SAM: {sam_w}")
    sam = SAM(sam_w)

    reid_w = _resolve_boxmot_reid_weights()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = DeepOcSort(
        reid_weights=reid_w,
        device=dev,
        half=bool(dev.type == "cuda"),
        det_thresh=yconf,
        max_age=150,
        min_hits=2,
        iou_threshold=0.3,
        embedding_off=True,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Error: could not open video", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w, out_h = args.tw * 2, args.th
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        print("Error: could not open VideoWriter", file=sys.stderr)
        sys.exit(1)

    # Seconds accumulated (perf_counter)
    sec_yolo = 0.0
    sec_boxmot = 0.0
    sec_sam = 0.0
    sec_resize = 0.0
    sec_compose = 0.0

    n = 0
    frames_written = 0
    n_yolo_calls = 0
    wall0 = time.perf_counter()

    current_detect_size = base_detect

    while True:
        if args.max_frames and frames_written >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        fi = n
        n += 1

        h_fr, w_fr = frame.shape[:2]

        tracker_out = None
        masks_tensor = None

        if fi % ystride == 0:
            t0 = time.perf_counter()
            frame_low = cv2.resize(frame, (current_detect_size, current_detect_size))
            frame_low_rgb = frame_low[:, :, ::-1]
            scale_x = w_fr / current_detect_size
            scale_y = h_fr / current_detect_size
            sec_resize += time.perf_counter() - t0

            t0 = time.perf_counter()
            res = yolo.predict(frame_low_rgb, classes=[0], conf=yconf, verbose=False)
            sec_yolo += time.perf_counter() - t0
            n_yolo_calls += 1

            r0 = res[0] if isinstance(res, list) else res
            if r0.boxes is None or len(r0.boxes) == 0:
                dets = np.empty((0, 6), dtype=np.float32)
                xyxy_full = np.empty((0, 4), dtype=np.float32)
            else:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                conf = r0.boxes.conf.cpu().numpy()
                xyxy[:, 0] *= scale_x
                xyxy[:, 1] *= scale_y
                xyxy[:, 2] *= scale_x
                xyxy[:, 3] *= scale_y
                keep = diou_nms_indices(xyxy, conf, iou_threshold=0.7)
                xyxy = xyxy[keep]
                conf = conf[keep]
                xyxy_full = xyxy.copy()
                cls0 = np.zeros((len(xyxy), 1), dtype=np.float32)
                dets = np.hstack([xyxy, conf.reshape(-1, 1), cls0]).astype(np.float32)

            t0 = time.perf_counter()
            tracker_out = tracker.update(dets, frame)
            sec_boxmot += time.perf_counter() - t0

            t0 = time.perf_counter()
            if len(xyxy_full) > 0:
                rs = sam.predict(frame, bboxes=xyxy_full, verbose=False)[0]
                if rs.masks is not None and rs.masks.data is not None:
                    masks_tensor = rs.masks.data
            sec_sam += time.perf_counter() - t0
        else:
            # Stride > 1: skip YOLO/SAM this frame; tracker unchanged (matches sway tracker behavior)
            pass

        t0 = time.perf_counter()
        left = draw_boxmot_panel(frame, tracker_out, yconf)
        right = draw_sam_panel(frame, masks_tensor) if masks_tensor is not None else frame.copy()
        p_l = letterbox(left, args.tw, args.th)
        p_r = letterbox(right, args.tw, args.th)
        draw_corner_label(p_l, "YOLO + BoxMOT DeepOCSORT", Path(model_path).name)
        draw_corner_label(p_r, "YOLO + SAM2.1", args.sam)
        writer.write(np.hstack([p_l, p_r]))
        frames_written += 1
        sec_compose += time.perf_counter() - t0

        if frames_written % 30 == 0:
            print(f"  frames: {frames_written}", flush=True)

    cap.release()
    writer.release()
    wall = time.perf_counter() - wall0

    ny = max(n_yolo_calls, 1)
    summary = {
        "video": str(args.video.resolve()),
        "output_mp4": str(args.output.resolve()),
        "frames_written": frames_written,
        "yolo_inference_calls": n_yolo_calls,
        "yolo_stride": ystride,
        "wall_clock_s": round(wall, 4),
        "seconds_total": {
            "yolo_predict": round(sec_yolo, 4),
            "boxmot_tracker_update": round(sec_boxmot, 4),
            "sam_predict": round(sec_sam, 4),
            "resize_for_yolo": round(sec_resize, 4),
            "letterbox_draw_write": round(sec_compose, 4),
        },
        "ms_per_frame_avg": {
            "yolo_predict_on_yolo_frames": round(1000 * sec_yolo / ny, 3),
            "boxmot_update_on_yolo_frames": round(1000 * sec_boxmot / ny, 3),
            "sam_predict_on_yolo_frames": round(1000 * sec_sam / ny, 3),
            "left_branch_estimate_yolo_plus_boxmot": round(1000 * (sec_yolo + sec_boxmot) / ny, 3),
            "right_branch_estimate_yolo_plus_sam": round(1000 * (sec_yolo + sec_sam) / ny, 3),
        },
        "note": "YOLO runs once per ystride frame; BoxMOT and SAM run on those same frames. "
        "Per-frame averages use yolo_inference_calls as denominator. "
        "Wall clock includes decode, all panels, and VideoWriter.",
    }

    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {args.output} ({frames_written} frames)")
    print(f"Timing JSON: {timing_path}")
    print(json.dumps(summary["seconds_total"], indent=2))
    print(json.dumps(summary["ms_per_frame_avg"], indent=2))
    print(f"Wall clock: {wall:.2f}s")


if __name__ == "__main__":
    main()
