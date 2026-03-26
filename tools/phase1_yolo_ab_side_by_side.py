#!/usr/bin/env python3
"""
Build a side-by-side Phase-1 (YOLO-only) preview from two ``phase1_yolo.npz`` checkpoints
and the same source video, with per-frame and aggregate confidence / count metrics.

Run from ``sway_pose_mvp``:
  python -m tools.phase1_yolo_ab_side_by_side --video path.mp4 \\
    --npz-a out_a/checkpoints/after_phase_1/phase1_yolo.npz \\
    --npz-b out_b/checkpoints/after_phase_1/phase1_yolo.npz \\
    --label-a "dancetrack" --label-b "dancetrack+crowdhuman" \\
    --out compare.mp4 --metrics-json compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import _repo_path  # noqa: F401, E402

from sway.checkpoint_io import load_phase1_yolo_npz_file  # noqa: E402

Pair = Tuple[Tuple[float, float, float, float], float]


def _metrics(pairs: List[Pair]) -> Dict[str, float]:
    if not pairs:
        return {"n": 0.0, "mean_c": 0.0, "max_c": 0.0, "min_c": 0.0}
    cs = [float(p[1]) for p in pairs]
    return {
        "n": float(len(cs)),
        "mean_c": float(sum(cs) / len(cs)),
        "max_c": float(max(cs)),
        "min_c": float(min(cs)),
    }


def _draw_half(
    frame_bgr: np.ndarray,
    pairs: List[Pair],
    box_bgr: Tuple[int, int, int],
) -> np.ndarray:
    out = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_col = (255, 255, 255)
    for (x1, y1, x2, y2), cf in pairs:
        xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (xi1, yi1), (xi2, yi2), box_bgr, 2)
        cv2.putText(
            out,
            f"{float(cf):.2f}",
            (xi1, max(18, yi1 - 6)),
            font,
            0.5,
            text_col,
            1,
            cv2.LINE_AA,
        )
    return out


def _rolling_update(
    dq_n: Deque[float],
    dq_mc: Deque[float],
    n: float,
    mean_c: float,
    win: int,
) -> Tuple[float, float]:
    dq_n.append(n)
    dq_mc.append(mean_c if n > 0 else 0.0)
    while len(dq_n) > win:
        dq_n.popleft()
        dq_mc.popleft()
    if not dq_n:
        return 0.0, 0.0
    rn = sum(dq_n) / len(dq_n)
    # rolling mean conf: average mean_c over frames that had ≥1 det in window
    active = [m for m, nn in zip(dq_mc, dq_n) if nn > 0]
    rmc = sum(active) / len(active) if active else 0.0
    return rn, rmc


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-1 YOLO A/B side-by-side + metrics")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--npz-a", required=True, type=Path)
    ap.add_argument("--npz-b", required=True, type=Path)
    ap.add_argument("--label-a", default="Model A")
    ap.add_argument("--label-b", default="Model B")
    ap.add_argument("--out", required=True, type=Path, help="Output MP4 path")
    ap.add_argument("--metrics-json", required=True, type=Path)
    ap.add_argument("--rolling-window", type=int, default=30, help="Frames for rolling avg")
    args = ap.parse_args()

    vid = args.video.expanduser().resolve()
    npz_a = args.npz_a.expanduser().resolve()
    npz_b = args.npz_b.expanduser().resolve()
    if not vid.is_file():
        raise SystemExit(f"Video not found: {vid}")
    if not npz_a.is_file() or not npz_b.is_file():
        raise SystemExit(f"Missing npz: {npz_a} or {npz_b}")

    map_a, meta_a = load_phase1_yolo_npz_file(npz_a)
    map_b, meta_b = load_phase1_yolo_npz_file(npz_b)
    tf = int(meta_a["total_frames"])
    if int(meta_b["total_frames"]) != tf:
        raise SystemExit("total_frames mismatch between npz A and B")
    nfps = float(meta_a["native_fps"])
    if abs(float(meta_b["native_fps"]) - nfps) > 1e-3:
        print("Warning: native_fps differs between npz; using A", flush=True)

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {vid}")

    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ncap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_read = min(tf, ncap) if ncap > 0 else tf

    # BGR: left cyan-orange, right green-magenta contrast
    col_a = (255, 180, 0)
    col_b = (60, 220, 60)
    banner_h = 120
    combo_w = w0 * 2 + 4
    combo_h = h0 + banner_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), fourcc, nfps, (combo_w, combo_h))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter failed for {args.out}")

    win = max(1, int(args.rolling_window))
    dq_na: Deque[float] = deque()
    dq_mca: Deque[float] = deque()
    dq_nb: Deque[float] = deque()
    dq_mcb: Deque[float] = deque()

    per_frame: List[Dict[str, Any]] = []
    sum_conf_a = 0.0
    sum_conf_b = 0.0
    cnt_a = 0
    cnt_b = 0
    frames_any_a = 0
    frames_any_b = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    for fi in range(frames_to_read):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        pa = map_a.get(fi, [])
        pb = map_b.get(fi, [])
        ma = _metrics(pa)
        mb = _metrics(pb)

        if ma["n"] > 0:
            frames_any_a += 1
            cnt_a += int(ma["n"])
            sum_conf_a += ma["mean_c"] * ma["n"]
        if mb["n"] > 0:
            frames_any_b += 1
            cnt_b += int(mb["n"])
            sum_conf_b += mb["mean_c"] * mb["n"]

        ra_n, ra_c = _rolling_update(dq_na, dq_mca, ma["n"], ma["mean_c"], win)
        rb_n, rb_c = _rolling_update(dq_nb, dq_mcb, mb["n"], mb["mean_c"], win)

        left = _draw_half(frame, pa, col_a)
        right = _draw_half(frame, pb, col_b)
        gap = np.zeros((h0, 4, 3), dtype=np.uint8)
        gap[:] = (40, 40, 40)
        body = np.hstack([left, gap, right])

        banner = np.zeros((banner_h, combo_w, 3), dtype=np.uint8)
        banner[:] = (28, 28, 28)

        def _line(y: int, text: str, color: Tuple[int, int, int] = (240, 240, 240)) -> None:
            cv2.putText(banner, text, (12, y), font, 0.5, color, 1, cv2.LINE_AA)

        _line(22, f"{args.label_a}  |  n={int(ma['n'])}  mean={ma['mean_c']:.3f}  max={ma['max_c']:.3f}", col_a)
        _line(
            44,
            f"  rolling[{win}f]  n_avg={ra_n:.2f}  mean_c_avg={ra_c:.3f}  (frames w/ dets: {frames_any_a}/{fi + 1})",
            (200, 200, 200),
        )
        _line(66, f"{args.label_b}  |  n={int(mb['n'])}  mean={mb['mean_c']:.3f}  max={mb['max_c']:.3f}", col_b)
        _line(
            88,
            f"  rolling[{win}f]  n_avg={rb_n:.2f}  mean_c_avg={rb_c:.3f}  (frames w/ dets: {frames_any_b}/{fi + 1})",
            (200, 200, 200),
        )
        diff_n = int(mb["n"] - ma["n"])
        diff_m = mb["mean_c"] - ma["mean_c"] if ma["n"] and mb["n"] else 0.0
        _line(110, f"Δ (B-A): n={diff_n:+d}   mean_c(when both>0)={diff_m:+.4f}", (180, 220, 255))

        combo = np.vstack([banner, body])
        writer.write(combo)

        per_frame.append(
            {
                "frame": fi,
                "a": {k: float(ma[k]) for k in ("n", "mean_c", "max_c", "min_c")},
                "b": {k: float(mb[k]) for k in ("n", "mean_c", "max_c", "min_c")},
                "rolling_a": {"n_avg": ra_n, "mean_c_avg": ra_c},
                "rolling_b": {"n_avg": rb_n, "mean_c_avg": rb_c},
            }
        )

    cap.release()
    writer.release()

    n_frames = len(per_frame)
    agg = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "frames_rendered": n_frames,
        "native_fps": nfps,
        "a": {
            "total_boxes": cnt_a,
            "mean_conf_over_all_boxes": (sum_conf_a / cnt_a) if cnt_a else 0.0,
            "frames_with_any_detection": frames_any_a,
            "mean_boxes_per_frame_any": (cnt_a / frames_any_a) if frames_any_a else 0.0,
        },
        "b": {
            "total_boxes": cnt_b,
            "mean_conf_over_all_boxes": (sum_conf_b / cnt_b) if cnt_b else 0.0,
            "frames_with_any_detection": frames_any_b,
            "mean_boxes_per_frame_any": (cnt_b / frames_any_b) if frames_any_b else 0.0,
        },
    }
    agg["interpretation"] = (
        "Higher mean_conf_over_all_boxes usually means sharper/more certain person boxes; "
        "higher total_boxes can mean more recall (or more FPs). Use side-by-side video to judge."
    )

    out_doc = {
        "video": str(vid),
        "npz_a": str(npz_a),
        "npz_b": str(npz_b),
        "aggregate": agg,
        "per_frame": per_frame,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)
    print(f"Wrote {args.metrics_json}", flush=True)


if __name__ == "__main__":
    main()
