#!/usr/bin/env python3
"""
Side-by-side detection on one video: two Ultralytics YOLO models (default: yolo26l vs yolo26x).
Same pattern as label_side_by_side_compare.py: letterboxed panels + corner labels, then hstack.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors as yolo_class_color


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


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def annotate_detections_readable(
    img_bgr: np.ndarray,
    result,
    *,
    font_scale: float = 0.52,
    thickness: int = 1,
    pad: int = 4,
    gap: int = 5,
) -> np.ndarray:
    """
    Draw boxes + labels without Ultralytics' default top-edge banners (they collide in crowds).
    Tries several anchor points per box and skips positions that overlap prior labels.
    """
    out = img_bgr.copy()
    ih, iw = out.shape[:2]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out

    names = result.names
    font = cv2.FONT_HERSHEY_SIMPLEX

    dets: list[dict] = []
    for b in boxes:
        xyxy = b.xyxy[0].detach().cpu().numpy()
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        dets.append(
            {
                "xyxy": xyxy,
                "text": f"{names[cls_id]} {conf:.2f}",
                "color": yolo_class_color(cls_id, bgr=True),
            }
        )

    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw - 1, x2), min(ih - 1, y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), d["color"], 2, cv2.LINE_AA)

    dets.sort(key=lambda d: (d["xyxy"][1], d["xyxy"][0]))
    placed: list[tuple[int, int, int, int]] = []

    def label_bg_rect(px: int, py_bl: int, tw: int, th_text: int, baseline: int) -> tuple[int, int, int, int]:
        return (
            px - pad,
            py_bl - th_text - pad,
            px + tw + pad,
            py_bl + baseline + pad,
        )

    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw - 1, x2), min(ih - 1, y2)
        text = d["text"]
        bgr = d["color"]
        (tw, th_text), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        candidates: list[tuple[int, int]] = []
        # Inside — top band (compact; stays on the person)
        if y1 + pad + th_text + baseline < y2 - pad:
            candidates.append((x1 + pad, y1 + pad + th_text))
        # Inside — bottom band
        py_bl = y2 - pad
        if py_bl - th_text > y1 + pad and py_bl + baseline < ih - 1:
            candidates.append((x1 + pad, py_bl))
        # Below box
        py_bl = y2 + gap + th_text
        if py_bl + baseline + pad < ih:
            candidates.append((x1 + pad, py_bl))
        # Above box (baseline so descenders sit just under y1 - gap)
        py_bl = y1 - gap - baseline
        if py_bl - th_text - pad > 0:
            candidates.append((x1 + pad, py_bl))
        # Inside — top-right
        px_r = x2 - tw - pad
        if px_r > x1 + pad and y1 + pad + th_text + baseline < y2 - pad:
            candidates.append((px_r, y1 + pad + th_text))

        seen: set[tuple[int, int]] = set()
        uniq: list[tuple[int, int]] = []
        for px, py in candidates:
            key = (px, py)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((px, py))
        candidates = uniq

        chosen: tuple[int, int, tuple[int, int, int, int]] | None = None
        for px, py_bl in candidates:
            rx1, ry1, rx2, ry2 = label_bg_rect(px, py_bl, tw, th_text, baseline)
            if rx1 < 0 or ry1 < 0 or rx2 >= iw or ry2 >= ih:
                continue
            if any(_rects_overlap((rx1, ry1, rx2, ry2), p) for p in placed):
                continue
            chosen = (px, py_bl, (rx1, ry1, rx2, ry2))
            break

        if chosen is None:
            px, py_bl = x1 + pad, y1 + pad + th_text
            rx1, ry1, rx2, ry2 = label_bg_rect(px, py_bl, tw, th_text, baseline)
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(iw - 1, rx2), min(ih - 1, ry2)
            chosen = (px, py_bl, (rx1, ry1, rx2, ry2))

        px, py_bl, bg = chosen
        rx1, ry1, rx2, ry2 = bg
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (32, 32, 32), -1, cv2.LINE_AA)
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), bgr, 1, cv2.LINE_AA)
        cv2.putText(
            out,
            text,
            (px, py_bl),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        placed.append((rx1, ry1, rx2, ry2))

    return out


def remux_mp4_to_h264(path: Path) -> bool:
    """
    OpenCV's default mp4v (MPEG-4 Part 2) often won't open in QuickTime / some players.
    Re-encode to H.264 when ffmpeg is available.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or path.suffix.lower() != ".mp4" or not path.is_file():
        return False
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        path.unlink()
        shutil.move(str(tmp_path), path)
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"Note: H.264 remux skipped ({e}); install ffmpeg for QuickTime-friendly MP4.", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="H-stack two YOLO detections on the same video")
    ap.add_argument("video", type=Path, help="Input video path")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output .mp4 path")
    ap.add_argument("--left-model", default="yolo26l.pt", help="Ultralytics weights (left panel)")
    ap.add_argument("--right-model", default="yolo26x.pt", help="Ultralytics weights (right panel)")
    ap.add_argument("--left-title", default="YOLO26l", help="Left panel title")
    ap.add_argument("--right-title", default="YOLO26x", help="Right panel title")
    ap.add_argument("--conf", type=float, default=0.22, help="Detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size (short side)")
    ap.add_argument("--tw", type=int, default=1280, help="Panel width after letterbox")
    ap.add_argument("--th", type=int, default=720, help="Panel height after letterbox")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = full video)",
    )
    ap.add_argument(
        "--legacy-plot",
        action="store_true",
        help="Use Ultralytics default plot() (overlapping top labels in crowds)",
    )
    ap.add_argument(
        "--parallel-infer",
        action="store_true",
        help="Run both model.predict() calls in parallel (threads; GPU may still serialize)",
    )
    ap.add_argument(
        "--skip-h264-remux",
        action="store_true",
        help="Keep OpenCV mp4v output (may not play in QuickTime; default remuxes via ffmpeg if installed)",
    )
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading models (may download weights once)...")
    model_l = YOLO(args.left_model)
    model_r = YOLO(args.right_model)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Error: could not open video", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w, out_h = args.tw * 2, args.th
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print("Error: could not open VideoWriter", file=sys.stderr)
        sys.exit(1)

    left_sub = str(args.left_model)
    right_sub = str(args.right_model)
    n = 0
    infer_left_s = 0.0
    infer_right_s = 0.0
    infer_parallel_s = 0.0
    t_wall0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.parallel_infer:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as ex:
                fl = ex.submit(
                    model_l.predict,
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )
                fr = ex.submit(
                    model_r.predict,
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )
                rl = fl.result()[0]
                rr = fr.result()[0]
            infer_parallel_s += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            rl = model_l.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            infer_left_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            rr = model_r.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            infer_right_s += time.perf_counter() - t0
        if args.legacy_plot:
            vis_l, vis_r = rl.plot(), rr.plot()
        else:
            vis_l = annotate_detections_readable(frame, rl)
            vis_r = annotate_detections_readable(frame, rr)
        p_l = letterbox(vis_l, args.tw, args.th)
        p_r = letterbox(vis_r, args.tw, args.th)
        draw_corner_label(p_l, args.left_title, left_sub)
        draw_corner_label(p_r, args.right_title, right_sub)
        writer.write(np.hstack([p_l, p_r]))
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        if n % 30 == 0:
            print(f"  frames: {n}")

    cap.release()
    writer.release()
    wall_s = time.perf_counter() - t_wall0
    if n > 0 and not args.skip_h264_remux and remux_mp4_to_h264(args.output):
        print("Remuxed to H.264 (QuickTime / broader player support).")
    print(f"Wrote {args.output} ({n} frames)")
    if n > 0:
        if args.parallel_infer:
            print(
                f"Timing (parallel infer — wall for both models per batch of frames):\n"
                f"  Both models: {infer_parallel_s:7.2f}s total  ({1000 * infer_parallel_s / n:5.1f} ms/frame)\n"
                f"  Wall (read + infer + plot/letterbox/write): {wall_s:7.2f}s"
            )
        else:
            print(
                f"Timing (inference only, perf_counter around each predict):\n"
                f"  {args.left_title}:  {infer_left_s:7.2f}s total  ({1000 * infer_left_s / n:5.1f} ms/frame)\n"
                f"  {args.right_title}: {infer_right_s:7.2f}s total  ({1000 * infer_right_s / n:5.1f} ms/frame)\n"
                f"  Wall (read + both infers + plot/letterbox/write): {wall_s:7.2f}s"
            )


if __name__ == "__main__":
    main()
