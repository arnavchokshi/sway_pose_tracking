"""
Re-encode phase3_vitpose.mp4 from source video + phase3_vitpose_overlay.json (no ViTPose model).

Example:
  python -m tools.render_phase3_vitpose_mp4 \\
    --video /path/to/newTest.mov \\
    --overlay output/newTest/phase3_vitpose_overlay.json \\
    --out output/newTest/phase3_vitpose.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import _repo_path  # noqa: F401
import cv2
import numpy as np

from sway.phase3_vitpose_viz import (
    apply_frame_confidence_grade,
    draw_vitpose_confidence_overlay,
    frame_mean_keypoint_conf,
    PHASE3_SKEL_BONES,
)


def _draw_text(frame, text, pos, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_phase_banner(frame, text, color=(50, 50, 50)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), color, -1)
    _draw_text(frame, text, (10, 22), (255, 255, 255), 0.6, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Render ViTPose phase MP4 from overlay JSON + video.")
    p.add_argument("--video", type=Path, default=None, help="Source video (default: metadata in overlay JSON)")
    p.add_argument("--overlay", type=Path, required=True, help="phase3_vitpose_overlay.json from a pipeline run")
    p.add_argument("--out", type=Path, default=None, help="Output mp4 (default: overlay dir / phase3_vitpose.mp4)")
    args = p.parse_args()

    overlay_path = args.overlay.expanduser().resolve()
    if not overlay_path.is_file():
        print(f"Overlay JSON not found: {overlay_path}", file=sys.stderr)
        sys.exit(1)

    with open(overlay_path) as f:
        data: Dict[str, Any] = json.load(f)

    meta = data.get("metadata") or {}
    video_path = args.video
    if video_path is None:
        vp = meta.get("video_path")
        if not vp:
            print("No --video and no metadata.video_path in overlay JSON.", file=sys.stderr)
            sys.exit(1)
        video_path = Path(str(vp)).expanduser()
    else:
        video_path = video_path.expanduser().resolve()

    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        out_path = overlay_path.parent / "phase3_vitpose.mp4"
    else:
        out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    fps = float(meta.get("fps") or 30.0)
    pv = meta.get("phase3_viz") or {}
    min_point = float(pv.get("min_point_conf", 0.03))
    min_bone = float(pv.get("min_bone_conf", 0.05))
    adapt_q = float(pv.get("adapt_quantile", 0.70))

    frames_json: List[Dict[str, Any]] = list(data.get("frames") or [])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    if width <= 0 or height <= 0:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"VideoWriter failed: {out_path}", file=sys.stderr)
        sys.exit(1)

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        fr_entry = frames_json[fi] if fi < len(frames_json) else None
        dets = (fr_entry or {}).get("detections") or []

        vk_for_grade: List[Optional[np.ndarray]] = []
        for det in dets:
            raw = det.get("vitpose_keypoints")
            if raw is not None and len(raw) > 0:
                vk_for_grade.append(np.asarray(raw, dtype=np.float32))
            else:
                vk_for_grade.append(None)

        out_vit = frame.copy()
        apply_frame_confidence_grade(out_vit, frame_mean_keypoint_conf(vk_for_grade))
        _draw_phase_banner(
            out_vit,
            f"Phase 3: ViTPose | conf-graded | Frame {fi} | dets={len(dets)}",
        )

        for det in dets:
            box = det.get("bbox")
            raw_kp = det.get("vitpose_keypoints")
            if not box or len(box) < 4 or raw_kp is None:
                continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            v_kp = np.asarray(raw_kp, dtype=np.float32)
            draw_vitpose_confidence_overlay(
                out_vit,
                v_kp,
                bbox_xyxy=(x1, y1, x2, y2),
                min_point_conf=min_point,
                min_bone_conf=min_bone,
                adapt_quantile=adapt_q,
                bones=PHASE3_SKEL_BONES,
            )

        writer.write(out_vit)
        fi += 1

    cap.release()
    writer.release()
    print(f"Wrote {fi} frames to {out_path}")


if __name__ == "__main__":
    main()
