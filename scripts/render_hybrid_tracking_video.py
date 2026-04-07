#!/usr/bin/env python3
"""
Run hybrid tracking and write one full-frame annotated video.

Visual modes (see --visual-mode):
  overlap  — DEFAULT, optimized: SAM masks only when max pairwise box IoU ≥ hybrid
             trigger (same rule as tracker). Other frames = boxes only. Cuts ~80%+ of
             render-time SAM vs masking every frame.
  all      — SAM mask on every frame with detections (slow; matches old behavior).
  boxes    — No SAM at render time (fastest preview).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _draw_track_labels(
    vis: np.ndarray,
    boxes: list,
    track_ids: list[int],
) -> None:
    for box, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.putText(
            vis,
            f"ID:{tid}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _sam_predict_masks(sam, frame_bgr: np.ndarray, xyxy: np.ndarray, sam_imgsz: int):
    import torch

    kwargs: dict = {"verbose": False}
    if sam_imgsz > 0:
        kwargs["imgsz"] = sam_imgsz
    with torch.inference_mode():
        return sam.predict(frame_bgr, bboxes=xyxy, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path, help="Input video")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output .mp4")
    ap.add_argument(
        "--visual-mode",
        choices=("overlap", "all", "boxes"),
        default="overlap",
        help="overlap=SAM masks only on high-IoU frames (fast, matches hybrid gate); all=every frame; boxes=no SAM",
    )
    ap.add_argument(
        "--boxes-only",
        action="store_true",
        help="Alias for --visual-mode boxes",
    )
    ap.add_argument(
        "--subtitle",
        default="",
        help="Override second title line (empty = auto from visual mode)",
    )
    ap.add_argument(
        "--sam-weights",
        default="",
        help="Ultralytics SAM checkpoint (default: SWAY_HYBRID_SAM_WEIGHTS or sam2.1_b.pt)",
    )
    ap.add_argument(
        "--sam-imgsz",
        type=int,
        default=0,
        help="If >0, pass imgsz= to SAM predict (smaller = faster, slightly softer masks). 0 = model default.",
    )
    ap.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Write timing / SAM call counts (default: <output>_render_metrics.json)",
    )
    args = ap.parse_args()

    visual = "boxes" if args.boxes_only else args.visual_mode

    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    os.environ.setdefault("SWAY_HYBRID_SAM_OVERLAP", "1")
    os.environ.setdefault("SWAY_YOLO_DETECTION_STRIDE", "1")
    y26 = REPO / "models" / "yolo26l.pt"
    if y26.is_file():
        os.environ.setdefault("SWAY_YOLO_WEIGHTS", str(y26))

    sam_raw = args.sam_weights.strip() or os.environ.get("SWAY_HYBRID_SAM_WEIGHTS", "sam2.1_b.pt").strip()

    import sway.tracker as sway_tracker

    from sway.hybrid_sam_refiner import (
        blend_sam_masks_for_tracks,
        load_hybrid_sam_config,
        resolve_hybrid_sam_weights,
        track_color_bgr,
    )

    sam_w = resolve_hybrid_sam_weights(sam_raw)
    from sway.track_pruning import raw_tracks_to_per_frame
    from sway.visualizer import draw_boxes_only

    hybrid_cfg = load_hybrid_sam_config()

    if not args.video.is_file():
        print(f"Missing video: {args.video}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    print("Running hybrid tracking…", flush=True)
    raw, total_frames, _, _, native_fps, fw, fh = sway_tracker.run_tracking(str(args.video))
    t_track = time.perf_counter() - t0
    ids = set(raw.keys())
    per = raw_tracks_to_per_frame(raw, total_frames, ids)
    n_frames_any_sam = sum(1 for row in per if any(row.get("is_sam_refined") or []))

    sam = None
    if visual != "boxes":
        from ultralytics import SAM

        print(f"Loading SAM for mask overlay: {sam_w}", flush=True)
        sam = SAM(sam_w)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or native_fps or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (fw, fh))
    if not writer.isOpened():
        print("VideoWriter failed", file=sys.stderr)
        sys.exit(1)

    if args.subtitle:
        subtitle = args.subtitle
    elif visual == "overlap":
        subtitle = (
            f"SAM masks on {n_frames_any_sam} frames (any TrackObservation.is_sam_refined); "
            f"sam_imgsz={args.sam_imgsz or 'default'}"
        )
    elif visual == "all":
        subtitle = "SAM2 mask every frame (slow path)"
    else:
        subtitle = "Boxes only (no SAM at encode)"

    if visual == "overlap":
        title = "Sway — hybrid (SAM pixels on overlap frames)"
    elif visual == "all":
        title = "Sway — hybrid (SAM every frame)"
    else:
        title = "Sway — hybrid tracking (boxes only)"

    sam_calls = 0
    mask_frames = 0
    box_only_frames = 0
    t_enc0 = time.perf_counter()

    fi = 0
    while fi < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        d = per[fi] if fi < len(per) else {"boxes": [], "track_ids": [], "is_sam_refined": []}
        boxes = d["boxes"]
        tids = d["track_ids"]

        sam_this_frame = any(d.get("is_sam_refined") or [])
        run_sam = sam is not None and len(boxes) > 0 and (
            visual == "all" or (visual == "overlap" and sam_this_frame)
        )

        if run_sam:
            xyxy = np.array(
                [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in boxes],
                dtype=np.float32,
            )
            try:
                res = _sam_predict_masks(sam, frame, xyxy, args.sam_imgsz)
                r0 = res[0] if isinstance(res, list) else res
            except TypeError:
                res = _sam_predict_masks(sam, frame, xyxy, 0)
                r0 = res[0] if isinstance(res, list) else res
            except Exception as e:
                print(f"  SAM predict failed frame {fi}: {e}", flush=True)
                vis = draw_boxes_only(frame, boxes, tids)
                box_only_frames += 1
            else:
                sam_calls += 1
                if r0.masks is not None and r0.masks.data is not None:
                    masks = r0.masks.data.detach().cpu().numpy()
                    vis = blend_sam_masks_for_tracks(frame, masks, tids)
                    for box, tid in zip(boxes, tids):
                        x1, y1, x2, y2 = map(int, box)
                        c = track_color_bgr(tid)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)
                    _draw_track_labels(vis, boxes, tids)
                    mask_frames += 1
                else:
                    vis = draw_boxes_only(frame, boxes, tids)
                    box_only_frames += 1
        else:
            vis = draw_boxes_only(frame, boxes, tids)
            if len(boxes) > 0:
                box_only_frames += 1

        bar_h = 86
        cv2.rectangle(vis, (8, 8), (min(fw - 8, 1020), bar_h), (25, 25, 25), -1)
        cv2.putText(
            vis, title, (16, 38), cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            vis,
            subtitle[:100],
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        writer.write(vis)
        fi += 1
        if fi % 60 == 0:
            print(f"  wrote frame {fi}/{total_frames}", flush=True)

    cap.release()
    writer.release()
    t_encode = time.perf_counter() - t_enc0
    t_wall = time.perf_counter() - t0

    metrics = {
        "video": str(args.video.resolve()),
        "output_mp4": str(args.output.resolve()),
        "visual_mode": visual,
        "hybrid_iou_trigger": hybrid_cfg["iou_trigger"],
        "hybrid_min_dets": hybrid_cfg["min_dets"],
        "sam_imgsz_arg": args.sam_imgsz,
        "frames_encoded": fi,
        "seconds_tracking": round(t_track, 3),
        "seconds_encode_write": round(t_encode, 3),
        "seconds_wall_total": round(t_wall, 3),
        "sam_predict_calls_render": sam_calls,
        "frames_with_mask_overlay": mask_frames,
        "frames_boxes_only_visual": box_only_frames,
        "frames_any_sam_refined_observation": n_frames_any_sam,
        "optimization_note": (
            "visual_mode=overlap runs SAM only on frames where at least one observation has "
            "is_sam_refined=True (from raw_tracks / raw_tracks_to_per_frame), not on every frame."
        ),
    }
    mj = args.metrics_json
    if mj is None:
        mj = args.output.with_name(args.output.stem + "_render_metrics.json")
    mj.parent.mkdir(parents=True, exist_ok=True)
    with open(mj, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {args.output}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Metrics: {mj}", flush=True)


if __name__ == "__main__":
    main()
