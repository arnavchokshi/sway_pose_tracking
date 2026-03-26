"""Live recompute helpers for Pipeline Lab (NMS IoU, etc.) without re-running YOLO."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from sway.tracker import classical_nms_indices


def phase1_pre_classical_to_per_frame_boxes(
    pre_by_frame: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    iou: float,
    frame_start: int,
    frame_end: int,
) -> List[Dict[str, Any]]:
    """
    Apply classical pre-track NMS with a custom IoU threshold to stored pre-NMS detections.

    ``pre_by_frame`` matches checkpoint ``phase1_yolo.npz`` ``counts_pc`` / ``boxes_pc`` (after DIoU when used).
    """
    out: List[Dict[str, Any]] = []
    iou = float(np.clip(iou, 0.05, 0.99))
    for fi in range(frame_start, frame_end + 1):
        pairs = pre_by_frame.get(fi, [])
        if not pairs:
            out.append({"frame_idx": fi, "boxes": [], "confs": []})
            continue
        xyxy = np.array([[p[0][0], p[0][1], p[0][2], p[0][3]] for p in pairs], dtype=np.float32)
        conf = np.array([p[1] for p in pairs], dtype=np.float32)
        keep = classical_nms_indices(xyxy, conf, iou_thresh=iou)
        xyxy_k = xyxy[keep]
        conf_k = conf[keep]
        boxes = [[float(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in xyxy_k]
        out.append({"frame_idx": fi, "boxes": boxes, "confs": [float(c) for c in conf_k]})
    return out
