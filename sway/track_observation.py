"""
Per-observation tracking record for handoff to pruning and pose.

Supports legacy 3-tuple (frame_idx, bbox_xyxy, conf) via coerce_observation().
New observations may carry SAM instance masks for mask-gated ViTPose crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

LegacyTrackEntry = Tuple[int, Tuple[float, float, float, float], float]


@dataclass
class TrackObservation:
    frame_idx: int
    bbox: Tuple[float, float, float, float]
    conf: float
    is_sam_refined: bool = False
    """True when hybrid SAM refined this detection before BoxMOT saw it."""
    segmentation_mask: Optional[np.ndarray] = None
    """
    Boolean mask aligned to bbox: shape (int(y2-y1), int(x2-x1)) for integer
    clip of bbox. True = dancer pixels from SAM; False = background/occluder.
    """

    def __iter__(self):
        yield self.frame_idx
        yield self.bbox
        yield self.conf

    def __getitem__(self, index: int):
        if index == 0:
            return self.frame_idx
        if index == 1:
            return self.bbox
        if index == 2:
            return self.conf
        raise IndexError(index)


def coerce_observation(entry: Union[LegacyTrackEntry, TrackObservation]) -> TrackObservation:
    if isinstance(entry, TrackObservation):
        return entry
    fi, box, conf = entry
    bb = tuple(float(x) for x in box[:4])
    return TrackObservation(int(fi), bb, float(conf), False, None)


def iou_xyxy_np(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for two xyxy arrays shape (4,)."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    b1 = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    u = a1 + b1 - inter
    return float(inter / u) if u > 0 else 0.0


def assign_sam_masks_to_tracker_output(
    dets: np.ndarray,
    out: np.ndarray,
    per_det_masks: List[Optional[np.ndarray]],
    *,
    iou_thresh: float = 0.25,
) -> List[Tuple[bool, Optional[np.ndarray]]]:
    """
    Map per-detection SAM masks (aligned with rows of ``dets``) to BoxMOT output
    rows using greedy IoU matching. BoxMOT track IDs are unrelated to SAM; this
    step re-associates masks to whichever output row best overlaps each det row.
    Returns one (is_sam_refined, mask_or_none) per row of ``out``.
    """
    if out is None or len(out) == 0:
        return []
    out = np.atleast_2d(out)
    if len(dets) == 0:
        return [(False, None)] * len(out)

    n_d = len(dets)
    pads = list(per_det_masks) if per_det_masks is not None else []
    while len(pads) < n_d:
        pads.append(None)

    used_d = set()
    result: List[Tuple[bool, Optional[np.ndarray]]] = []
    for i in range(len(out)):
        ob = out[i, :4].astype(np.float32)
        best_j = -1
        best_iou = 0.0
        for j in range(n_d):
            if j in used_d:
                continue
            iou = iou_xyxy_np(ob, dets[j, :4])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            used_d.add(best_j)
            m = pads[best_j]
            result.append((m is not None, m))
        else:
            result.append((False, None))
    return result


def resize_mask_to_bbox(
    mask: Optional[np.ndarray],
    box_xyxy: Tuple[float, float, float, float],
) -> Optional[np.ndarray]:
    """Resize boolean/float mask to (H,W) of integer bbox (tracker output)."""
    if mask is None:
        return None
    x1, y1, x2, y2 = box_xyxy
    tw = max(1, int(round(x2)) - int(round(x1)))
    th = max(1, int(round(y2)) - int(round(y1)))
    import cv2

    m = mask.astype(np.float32)
    if m.shape[0] == th and m.shape[1] == tw:
        return m > 0.5
    out = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
    return out > 0.5
