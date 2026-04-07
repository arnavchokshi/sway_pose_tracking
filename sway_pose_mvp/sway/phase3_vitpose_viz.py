"""ViTPose phase-3 overlay: confidence-driven skeleton colors and optional frame grading."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# COCO-17 edges used by the pipeline (same as run_pipeline_v23_bigtest phase3_skel_bones).
PHASE3_SKEL_BONES: List[Tuple[int, int]] = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]


def phase3_adaptive_thresh(
    scores: np.ndarray,
    default_thr: float,
    floor_thr: float,
    *,
    adapt_quantile: float = 0.70,
) -> float:
    if scores is None or scores.size == 0:
        return float(default_thr)
    valid = scores[np.isfinite(scores)]
    valid = valid[valid > 0.0]
    if valid.size == 0:
        return float(max(floor_thr, min(default_thr, 0.0)))
    qv = float(np.quantile(valid, adapt_quantile))
    return float(max(floor_thr, min(default_thr, qv)))


def vitpose_confidence_to_bgr(conf: float) -> Tuple[int, int, int]:
    """Map joint/bone confidence to BGR: low=red, high=green (OpenCV HSV hue)."""
    c = float(np.clip(conf, 0.0, 1.0))
    h = int(round(60.0 * c))
    hsv = np.uint8([[[h, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def frame_mean_keypoint_conf(keypoints_list: Sequence[Optional[np.ndarray]]) -> float:
    """Mean confidence over all strictly-positive keypoint scores in the frame."""
    vals: List[float] = []
    for kp in keypoints_list:
        if kp is None or not isinstance(kp, np.ndarray) or kp.size == 0 or kp.shape[1] < 3:
            continue
        sc = kp[:, 2]
        sc = sc[np.isfinite(sc) & (sc > 0.0)]
        if sc.size:
            vals.append(float(np.mean(sc)))
    return float(np.mean(vals)) if vals else 0.0


def apply_frame_confidence_grade(frame: np.ndarray, mean_conf: float) -> None:
    """In-place: darken/desaturate the base image when mean_conf is low (in [0, 1])."""
    m = float(np.clip(mean_conf, 0.0, 1.0))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    gain_v = 0.42 + 0.58 * m
    gain_s = 0.45 + 0.55 * m
    s = np.clip(s * gain_s, 0, 255)
    v = np.clip(v * gain_v, 0, 255)
    graded = cv2.merge([h, s, v]).astype(np.uint8)
    frame[:] = cv2.cvtColor(graded, cv2.COLOR_HSV2BGR)


def draw_vitpose_confidence_overlay(
    frame: np.ndarray,
    v_kp: np.ndarray,
    *,
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    min_point_conf: float = 0.03,
    min_bone_conf: float = 0.05,
    adapt_quantile: float = 0.70,
    bones: Sequence[Tuple[int, int]] = PHASE3_SKEL_BONES,
    line_thickness: int = 2,
    point_radius: int = 3,
) -> None:
    """Draw COCO-17 ViTPose skeleton on frame; colors encode per-joint / per-bone confidence."""
    if v_kp is None or not isinstance(v_kp, np.ndarray) or v_kp.size == 0:
        return
    _v_scores = v_kp[:, 2] if v_kp.shape[1] >= 3 else np.zeros((v_kp.shape[0],), dtype=np.float32)
    v_bone_thr = phase3_adaptive_thresh(_v_scores, default_thr=0.25, floor_thr=min_bone_conf, adapt_quantile=adapt_quantile)
    v_point_thr = phase3_adaptive_thresh(_v_scores, default_thr=0.20, floor_thr=min_point_conf, adapt_quantile=adapt_quantile)

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        person_m = float(np.mean(_v_scores[np.isfinite(_v_scores) & (_v_scores > 0.0)])) if np.any(_v_scores > 0) else 0.0
        bc = vitpose_confidence_to_bgr(person_m)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)

    for j1, j2 in bones:
        if j1 < len(v_kp) and j2 < len(v_kp) and v_kp[j1, 2] > v_bone_thr and v_kp[j2, 2] > v_bone_thr:
            c = min(float(v_kp[j1, 2]), float(v_kp[j2, 2]))
            col = vitpose_confidence_to_bgr(c)
            p1 = (int(v_kp[j1, 0]), int(v_kp[j1, 1]))
            p2 = (int(v_kp[j2, 0]), int(v_kp[j2, 1]))
            cv2.line(frame, p1, p2, col, line_thickness, cv2.LINE_AA)

    v_drawn = 0
    for j in range(min(17, len(v_kp))):
        if v_kp[j, 2] > v_point_thr:
            col = vitpose_confidence_to_bgr(float(v_kp[j, 2]))
            cv2.circle(frame, (int(v_kp[j, 0]), int(v_kp[j, 1])), point_radius, col, -1)
            v_drawn += 1
    if v_drawn == 0 and len(v_kp) > 0:
        order = np.argsort(-v_kp[:, 2])
        for j in order[: min(8, len(order))]:
            if v_kp[j, 2] <= 0.0:
                continue
            col = vitpose_confidence_to_bgr(float(v_kp[j, 2]))
            cv2.circle(frame, (int(v_kp[j, 0]), int(v_kp[j, 1])), point_radius, col, -1)
