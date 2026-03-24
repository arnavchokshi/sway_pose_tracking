#!/usr/bin/env python3
"""Per-frame YOLO stats on a video for two models → JSONL + summary (detection-only)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def frame_stats(result) -> dict:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {"n_detections": 0, "confs": [], "mean_conf": None}
    confs = [float(b.conf[0]) for b in boxes]
    return {
        "n_detections": len(confs),
        "confs": confs,
        "mean_conf": float(statistics.mean(confs)) if confs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--model-a", type=Path, required=True)
    ap.add_argument("--model-b", type=Path, required=True)
    ap.add_argument("--label-a", default="model_a")
    ap.add_argument("--label-b", default="model_b")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.22)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    def resolve_weights(p: Path) -> str:
        if p.is_file():
            return str(p.resolve())
        s = str(p)
        if s.endswith(".pt") and "/" not in s and "\\" not in s:
            return s
        print(f"Weights not found: {p}", file=sys.stderr)
        sys.exit(1)

    wa = resolve_weights(args.model_a)
    wb = resolve_weights(args.model_b)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path_a = args.out_dir / f"{args.label_a}.jsonl"
    path_b = args.out_dir / f"{args.label_b}.jsonl"
    path_summary = args.out_dir / "summary.json"

    print("Loading models...")
    ma = YOLO(wa)
    mb = YOLO(wb)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Could not open video", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    rows_a: list[dict] = []
    rows_b: list[dict] = []
    t0 = time.perf_counter()
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ra = ma.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        rb = mb.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        sa = frame_stats(ra)
        sb = frame_stats(rb)
        line_a = {"frame": fi, **sa}
        line_b = {"frame": fi, **sb}
        rows_a.append(line_a)
        rows_b.append(line_b)
        fi += 1
        if fi % 60 == 0:
            print(f"  frames: {fi}")

    cap.release()
    elapsed = time.perf_counter() - t0

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    write_jsonl(path_a, rows_a)
    write_jsonl(path_b, rows_b)

    def aggregate(rows: list[dict]) -> dict:
        ns = [r["n_detections"] for r in rows]
        all_confs = [c for r in rows for c in r["confs"]]
        return {
            "frames": len(rows),
            "total_detections": int(sum(ns)),
            "mean_detections_per_frame": float(statistics.mean(ns)) if ns else 0.0,
            "max_detections_in_frame": int(max(ns)) if ns else 0,
            "mean_conf_over_all_dets": float(statistics.mean(all_confs)) if all_confs else None,
        }

    diff_frames = [
        {
            "frame": r["frame"],
            "n_a": r["n_detections"],
            "n_b": s["n_detections"],
            "delta": r["n_detections"] - s["n_detections"],
        }
        for r, s in zip(rows_a, rows_b)
        if r["n_detections"] != s["n_detections"]
    ]

    summary = {
        "video": str(args.video.resolve()),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "fps_reported": fps,
        "wall_seconds": round(elapsed, 3),
        args.label_a: {
            "weights": wa,
            **aggregate(rows_a),
            "jsonl": str(path_a.resolve()),
        },
        args.label_b: {
            "weights": wb,
            **aggregate(rows_b),
            "jsonl": str(path_b.resolve()),
        },
        "frames_where_count_differs": len(diff_frames),
        "frame_diffs_sample": diff_frames[:40],
        "frame_diffs_tail": diff_frames[-20:] if len(diff_frames) > 60 else [],
    }
    path_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {path_a}")
    print(f"Wrote {path_b}")
    print(f"Wrote {path_summary}")


if __name__ == "__main__":
    main()
