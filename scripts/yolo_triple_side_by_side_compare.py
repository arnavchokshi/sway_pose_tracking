#!/usr/bin/env python3
"""
Triple side-by-side YOLO detection on one video (e.g. base vs DanceTrack vs DanceTrack+CrowdHuman).
YOLO predict only — no tracking, no pose. Reuses plotting style from yolo_side_by_side_compare.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
        if y1 + pad + th_text + baseline < y2 - pad:
            candidates.append((x1 + pad, y1 + pad + th_text))
        py_bl = y2 - pad
        if py_bl - th_text > y1 + pad and py_bl + baseline < ih - 1:
            candidates.append((x1 + pad, py_bl))
        py_bl = y2 + gap + th_text
        if py_bl + baseline + pad < ih:
            candidates.append((x1 + pad, py_bl))
        py_bl = y1 - gap - baseline
        if py_bl - th_text - pad > 0:
            candidates.append((x1 + pad, py_bl))
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


def _count_dets(result) -> int:
    boxes = result.boxes
    if boxes is None:
        return 0
    return len(boxes)


def main() -> None:
    ap = argparse.ArgumentParser(description="H-stack three YOLO detections on the same video")
    ap.add_argument("video", type=Path, help="Input video path")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output .mp4 path")
    ap.add_argument(
        "--models",
        nargs=3,
        metavar=("M1", "M2", "M3"),
        required=True,
        help="Three Ultralytics weight paths or hub ids",
    )
    ap.add_argument(
        "--titles",
        nargs=3,
        metavar=("T1", "T2", "T3"),
        default=None,
        help="Panel titles (default: derived from basenames)",
    )
    ap.add_argument("--conf", type=float, default=0.22, help="Detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=960, help="Inference size (short side; matches sway crowd-friendly size)")
    ap.add_argument("--tw", type=int, default=640, help="Panel width after letterbox (3*tw total width)")
    ap.add_argument("--th", type=int, default=360, help="Panel height after letterbox")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full video)")
    ap.add_argument("--legacy-plot", action="store_true", help="Use Ultralytics plot()")
    ap.add_argument("--parallel-infer", action="store_true", help="Run three predict() calls in parallel (threads)")
    ap.add_argument("--skip-h264-remux", action="store_true")
    ap.add_argument("--summary-json", type=Path, default=None, help="Write per-frame stats + aggregates here")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    for i, m in enumerate(args.models):
        p = Path(m)
        if not p.is_file() and not str(m).endswith(".pt"):
            print(f"Warning: model {i + 1} may be hub id (not found as file): {m}", file=sys.stderr)

    titles = list(args.titles) if args.titles else [Path(m).stem for m in args.models]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading 3 models...")
    models = [YOLO(m) for m in args.models]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Error: could not open video", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w, out_h = args.tw * 3, args.th
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print("Error: could not open VideoWriter", file=sys.stderr)
        sys.exit(1)

    subs = [str(m) for m in args.models]
    n = 0
    infer_parallel_s = 0.0
    infer_seq_s = [0.0, 0.0, 0.0]
    counts_per_frame: list[list[int]] = []
    t_wall0 = time.perf_counter()

    def predict_one(mi: int, frame: np.ndarray):
        t0 = time.perf_counter()
        r = models[mi].predict(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        infer_seq_s[mi] += time.perf_counter() - t0
        return r

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.parallel_infer:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(predict_one, 0, frame)
                f1 = ex.submit(predict_one, 1, frame)
                f2 = ex.submit(predict_one, 2, frame)
                r0, r1, r2 = f0.result(), f1.result(), f2.result()
            infer_parallel_s += time.perf_counter() - t0
        else:
            r0 = predict_one(0, frame)
            r1 = predict_one(1, frame)
            r2 = predict_one(2, frame)

        c0, c1, c2 = _count_dets(r0), _count_dets(r1), _count_dets(r2)
        counts_per_frame.append([c0, c1, c2])

        if args.legacy_plot:
            vis0, vis1, vis2 = r0.plot(), r1.plot(), r2.plot()
        else:
            vis0 = annotate_detections_readable(frame, r0)
            vis1 = annotate_detections_readable(frame, r1)
            vis2 = annotate_detections_readable(frame, r2)

        panels = []
        for vis, title, sub in zip((vis0, vis1, vis2), titles, subs, strict=True):
            p = letterbox(vis, args.tw, args.th)
            draw_corner_label(p, title, sub)
            panels.append(p)
        writer.write(np.hstack(panels))
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        if n % 30 == 0:
            print(f"  frames: {n}")

    cap.release()
    writer.release()
    wall_s = time.perf_counter() - t_wall0

    summary: dict[str, Any] = {
        "video": str(args.video.resolve()),
        "output_video": str(args.output.resolve()),
        "frames": n,
        "fps": fps,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "models": [
            {"path": subs[i], "title": titles[i]} for i in range(3)
        ],
        "wall_seconds": wall_s,
    }

    if n > 0:
        arr = np.array(counts_per_frame, dtype=np.int32)
        per_model: list[dict[str, Any]] = []
        for i in range(3):
            col = arr[:, i]
            per_model.append(
                {
                    "title": titles[i],
                    "weights": subs[i],
                    "total_detections_sum": int(col.sum()),
                    "mean_detections_per_frame": float(col.mean()),
                    "std_detections_per_frame": float(col.std()) if n > 1 else 0.0,
                    "max_detections_single_frame": int(col.max()),
                    "min_detections_single_frame": int(col.min()),
                    "frames_with_zero": int(np.sum(col == 0)),
                    "infer_seconds_sequential_slice": infer_seq_s[i],
                }
            )
        summary["per_model"] = per_model
        if args.parallel_infer:
            summary["infer_seconds_parallel_total"] = infer_parallel_s
            summary["mean_ms_parallel_per_frame"] = 1000.0 * infer_parallel_s / n

    if n > 0 and not args.skip_h264_remux and remux_mp4_to_h264(args.output):
        print("Remuxed to H.264 (QuickTime / broader player support).")

    print(f"Wrote {args.output} ({n} frames)")

    if args.summary_json:
        out_j = {"summary": summary, "detections_per_frame": counts_per_frame}
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(out_j, indent=2))
        print(f"Wrote {args.summary_json}")

    if n > 0 and summary.get("per_model"):
        print("\n--- Summary (detection counts, person class) ---")
        for pm in summary["per_model"]:
            print(
                f"  {pm['title']}: mean {pm['mean_detections_per_frame']:.2f} det/frame, "
                f"max {pm['max_detections_single_frame']}, total {pm['total_detections_sum']} over {n} frames"
            )


if __name__ == "__main__":
    main()
