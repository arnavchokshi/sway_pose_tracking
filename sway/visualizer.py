"""
Serialization & Visualization Module

Exports smoothed keypoint data to JSON and renders an MP4 video with
bounding boxes, track IDs, and skeleton overlays. Supports Phase 3 & 4:
joint angle deviation heatmap (Red/Yellow/Green per bone) and JSON export
with angles and deviations.

Export-time blend between processed_fps samples and native_fps (default **linear**).
Optional: ``SWAY_VIS_TEMPORAL_INTERP_MODE=gsi`` and ``SWAY_VIS_GSI_LENGTHSCALE`` /
``SWAY_GSI_LENGTHSCALE`` for smoother overlay motion in MP4s (does not change JSON samples).
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from .interp_utils import blend_scalar
from .pose_estimator import COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES

# Colors (BGR for OpenCV)
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)
KEYPOINT_COLOR = (0, 0, 255)  # Red
KEYPOINT_THRESHOLD = 0.3

# V3.0 Dual-Signal Heatmap colors
COLOR_IN_SYNC = (0, 255, 0)        # Green: Shape within tolerance AND Timing ≤ 2 frames
COLOR_OFF_BEAT = (255, 128, 0)     # Blue (BGR): Shape good, Timing > 2 frames
COLOR_SLIGHTLY_OFF = (0, 255, 255) # Yellow: Minor Shape Error
COLOR_MAJOR_ERROR = (0, 0, 255)   # Red: Major Shape Error
COLOR_OCCLUDED = (150, 150, 150)  # Gray: missing/occluded/ripple

# V3.0 Per-joint shape thresholds (degrees)
SHAPE_GREEN_SPINE = 10.0   # Spine/Hips (shoulders): Green < 10°
SHAPE_RED_SPINE = 20.0     # Spine/Hips: Red > 20°
SHAPE_GREEN_LIMB = 20.0    # Elbows/Knees: Green < 20°
SHAPE_RED_LIMB = 35.0      # Elbows/Knees: Red > 35°

# Timing threshold for Off-Beat (frames)
TIMING_OFF_BEAT_THRESHOLD = 2

# Spine joints (use stricter thresholds)
SPINE_JOINTS = {"left_shoulder", "right_shoulder"}

# Bone (a, b) -> (diff_key, shape_key, timing_key) for heatmap
EDGE_TO_DEVIATION_KEY = {
    (5, 7): "left_shoulder_diff",
    (7, 9): "left_elbow_diff",
    (6, 8): "right_shoulder_diff",
    (8, 10): "right_elbow_diff",
    (11, 13): "left_knee_diff",
    (13, 15): "left_knee_diff",
    (12, 14): "right_knee_diff",
    (14, 16): "right_knee_diff",
}


def _deviation_to_color(
    dev: Optional[float],
    shape_err: Optional[float] = None,
    timing_err: Optional[float] = None,
    joint_base: Optional[str] = None,
) -> tuple:
    """
    V3.0 Dual-signal with per-joint thresholds:
    - Spine/Hips (shoulders): Green < 10°, Red > 20°
    - Elbows/Knees: Green < 20°, Red > 35°
    """
    if dev is None or (isinstance(dev, float) and (math.isnan(dev) or math.isinf(dev))):
        return COLOR_OCCLUDED
    shape = shape_err if shape_err is not None and not (isinstance(shape_err, float) and math.isnan(shape_err)) else dev
    timing = timing_err if timing_err is not None and not (isinstance(timing_err, float) and math.isnan(timing_err)) else 0.0
    if math.isnan(shape) or math.isinf(shape):
        return COLOR_OCCLUDED
    green_thresh = SHAPE_GREEN_SPINE if joint_base and joint_base in SPINE_JOINTS else SHAPE_GREEN_LIMB
    red_thresh = SHAPE_RED_SPINE if joint_base and joint_base in SPINE_JOINTS else SHAPE_RED_LIMB
    yellow_hi = red_thresh
    if shape < green_thresh:
        if abs(timing) <= TIMING_OFF_BEAT_THRESHOLD:
            return COLOR_IN_SYNC
        return COLOR_OFF_BEAT
    if shape < red_thresh:
        return COLOR_SLIGHTLY_OFF
    return COLOR_MAJOR_ERROR


def _mean_pose_confidence(tid: int, poses: Optional[Dict[int, Dict]]) -> Optional[float]:
    """Mean keypoint confidence for a track (scores array or keypoints[:, 2])."""
    if not poses:
        return None
    key = int(tid)
    if key not in poses:
        return None
    data = poses[key]
    sc = data.get("scores")
    if sc is not None:
        arr = np.asarray(sc, dtype=np.float64)
        if arr.size > 0:
            return float(np.mean(arr))
    kp = data.get("keypoints")
    if kp is not None:
        arr = np.asarray(kp, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] > 0:
            return float(np.mean(arr[:, 2]))
    return None


def draw_boxes_only(
    frame: np.ndarray,
    boxes: List[tuple],
    track_ids: List[int],
    poses: Optional[Dict[int, Dict]] = None,
) -> np.ndarray:
    """Draw only bounding boxes and track ID labels (no skeletons).

    When ``poses`` is provided, each label includes mean keypoint confidence, e.g. ``ID:3 0.82``.
    """
    out = frame.copy()
    for box, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"ID:{tid}"
        mc = _mean_pose_confidence(int(tid), poses)
        if mc is not None:
            label = f"{label} {mc:.2f}"
        cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, TEXT_COLOR, 1, cv2.LINE_AA)
    return out


def _draw_dashed_rect_bgr(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash: int = 14,
    gap: int = 8,
) -> None:
    """Dashed axis-aligned rectangle (BGR)."""
    if x2 <= x1 or y2 <= y1:
        return
    step = dash + gap

    def hline(y: int, xa: int, xb: int) -> None:
        x = xa
        while x < xb:
            xe = min(x + dash, xb)
            cv2.line(img, (x, y), (xe, y), color, thickness, cv2.LINE_AA)
            x += step

    def vline(x: int, ya: int, yb: int) -> None:
        y = ya
        while y < yb:
            ye = min(y + dash, yb)
            cv2.line(img, (x, y), (x, ye), color, thickness, cv2.LINE_AA)
            y += step

    hline(y1, x1, x2)
    hline(y2, x1, x2)
    vline(x1, y1, y2)
    vline(x2, y1, y2)


def draw_tracks_post_stitch_preview(
    frame: np.ndarray,
    boxes: List[tuple],
    track_ids: List[int],
    sam2_input_roi_xyxy: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Phase 1–3 preview: boxes/IDs plus optional dashed rectangle for the image region fed to SAM2
    when hybrid overlap refinement ran (ROI crop or full frame).
    """
    out = draw_boxes_only(frame, boxes, track_ids)
    if sam2_input_roi_xyxy is None or len(sam2_input_roi_xyxy) < 4:
        return out
    h, w = out.shape[:2]
    x1 = int(round(float(sam2_input_roi_xyxy[0])))
    y1 = int(round(float(sam2_input_roi_xyxy[1])))
    x2 = int(round(float(sam2_input_roi_xyxy[2])))
    y2 = int(round(float(sam2_input_roi_xyxy[3])))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return out
    # Orange-cyan (BGR): visible on stage footage
    col = (0, 165, 255)
    _draw_dashed_rect_bgr(out, x1, y1, x2, y2, col, thickness=2)
    label = "SAM2 input"
    cv2.putText(
        out,
        label,
        (x1, max(12, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        col,
        2,
        cv2.LINE_AA,
    )
    return out


# Distinct from green track boxes in other phase previews (BGR).
PHASE1_PREVIEW_BOX_BGR = (0, 220, 255)


def draw_phase1_detection_preview(frame: np.ndarray, fd: Dict) -> np.ndarray:
    """Draw every Phase-1 person box with detector confidence (no track IDs)."""
    boxes = fd.get("phase1_boxes") or []
    confs = fd.get("phase1_confs") or []
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, cf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), PHASE1_PREVIEW_BOX_BGR, 2)
        cv2.putText(
            out,
            f"{float(cf):.2f}",
            (x1, max(18, y1 - 6)),
            font,
            0.55,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_frame_with_boxes(
    frame: np.ndarray,
    boxes: List[tuple],
    track_ids: List[int],
    poses: Dict[int, Dict],
    deviations: Optional[Dict[int, Dict[str, float]]] = None,
    shape_errors: Optional[Dict[int, Dict[str, float]]] = None,
    timing_errors: Optional[Dict[int, Dict[str, float]]] = None,
) -> np.ndarray:
    """Draw bounding boxes, track ID labels, AND skeleton overlays."""
    out = draw_boxes_only(frame, boxes, track_ids)
    for tid, data in poses.items():
        keypoints = data["keypoints"]
        scores = data.get("scores", np.ones(17))
        if keypoints.shape[0] < 17:
            continue
        track_deviations = (deviations or {}).get(tid, {})
        track_shape = (shape_errors or {}).get(tid, {})
        track_timing = (timing_errors or {}).get(tid, {})

        for (a, b) in COCO_SKELETON_EDGES:
            if a >= keypoints.shape[0] or b >= keypoints.shape[0]:
                continue
            sa = float(scores[a]) if hasattr(scores, "__len__") else float(scores)
            sb = float(scores[b]) if hasattr(scores, "__len__") else float(scores)
            if sa < KEYPOINT_THRESHOLD or sb < KEYPOINT_THRESHOLD:
                continue
            x1, y1 = int(keypoints[a, 0]), int(keypoints[a, 1])
            x2, y2 = int(keypoints[b, 0]), int(keypoints[b, 1])
            edge = (a, b)
            if edge in EDGE_TO_DEVIATION_KEY:
                dev_key = EDGE_TO_DEVIATION_KEY[edge]
                dev = track_deviations.get(dev_key)
                base = dev_key.replace("_diff", "")
                shape_key = f"{base}_shape"
                timing_key = f"{base}_timing"
                se = track_shape.get(shape_key)
                te = track_timing.get(timing_key)
                color = _deviation_to_color(dev, se, te, joint_base=base)
            else:
                color = COLOR_OCCLUDED
            cv2.line(out, (x1, y1), (x2, y2), color, 2)

        for j in range(keypoints.shape[0]):
            sc = float(scores[j]) if hasattr(scores, "__len__") else float(scores)
            if sc < KEYPOINT_THRESHOLD:
                continue
            x, y = int(keypoints[j, 0]), int(keypoints[j, 1])
            cv2.circle(out, (x, y), 4, KEYPOINT_COLOR, -1)
    return out


def draw_frame(
    frame: np.ndarray,
    boxes: List[tuple],
    track_ids: List[int],
    poses: Dict[int, Dict],
    deviations: Optional[Dict[int, Dict[str, float]]] = None,
    shape_errors: Optional[Dict[int, Dict[str, float]]] = None,
    timing_errors: Optional[Dict[int, Dict[str, float]]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes, track ID labels, and smoothed skeletons on a frame.
    V2.3 Dual-signal heatmap: Green/Blue/Yellow/Red based on shape + timing error.

    Args:
        frame: BGR image.
        boxes: List of (x1, y1, x2, y2).
        track_ids: List of track IDs.
        poses: {track_id: {"keypoints": (17,3), "scores": (17,)}}.
        deviations: {track_id: {"left_elbow_diff": 12.5, ...}}.
        shape_errors: {track_id: {"left_elbow_shape": 8.2, ...}} (optional).
        timing_errors: {track_id: {"left_elbow_timing": 3.1, ...}} (optional).
    """
    out = frame.copy()

    # Draw green bounding boxes + track ID labels (for review / feedback)
    out = draw_boxes_only(out, boxes, track_ids)

    # Draw skeletons for each tracked person
    for tid, data in poses.items():
        keypoints = data["keypoints"]
        scores = data.get("scores", np.ones(17))
        if keypoints.shape[0] < 17:
            continue
        track_deviations = (deviations or {}).get(tid, {})
        track_shape = (shape_errors or {}).get(tid, {})
        track_timing = (timing_errors or {}).get(tid, {})

        # Draw skeleton edges with V2.3 dual-signal heatmap
        for (a, b) in COCO_SKELETON_EDGES:
            if a >= keypoints.shape[0] or b >= keypoints.shape[0]:
                continue
            sa = float(scores[a]) if hasattr(scores, "__len__") else float(scores)
            sb = float(scores[b]) if hasattr(scores, "__len__") else float(scores)
            if sa < KEYPOINT_THRESHOLD or sb < KEYPOINT_THRESHOLD:
                continue
            x1, y1 = int(keypoints[a, 0]), int(keypoints[a, 1])
            x2, y2 = int(keypoints[b, 0]), int(keypoints[b, 1])
            edge = (a, b)
            if edge in EDGE_TO_DEVIATION_KEY:
                dev_key = EDGE_TO_DEVIATION_KEY[edge]
                dev = track_deviations.get(dev_key)
                base = dev_key.replace("_diff", "")
                shape_key = f"{base}_shape"
                timing_key = f"{base}_timing"
                se = track_shape.get(shape_key)
                te = track_timing.get(timing_key)
                color = _deviation_to_color(dev, se, te, joint_base=base)
            else:
                color = COLOR_OCCLUDED
            cv2.line(out, (x1, y1), (x2, y2), color, 2)

        # Draw keypoints
        for j in range(keypoints.shape[0]):
            sc = float(scores[j]) if hasattr(scores, "__len__") else float(scores)
            if sc < KEYPOINT_THRESHOLD:
                continue
            x, y = int(keypoints[j, 0]), int(keypoints[j, 1])
            cv2.circle(out, (x, y), 4, KEYPOINT_COLOR, -1)

    return out


def _track_id_bgr(tid: int) -> Tuple[int, int, int]:
    """Stable saturated BGR for per-instance overlays (segmentation-style preview)."""
    hue = (abs(int(tid)) * 47) % 180
    hsv = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_segmentation_style(
    frame: np.ndarray,
    boxes: List[tuple],
    track_ids: List[int],
    *,
    is_sam_refined: Optional[List[bool]] = None,
    segmentation_masks: Optional[List[Optional[np.ndarray]]] = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Colored pixels only for detections that were hybrid-SAM-refined (is_sam_refined).
    Uses per-instance SAM mask when present; otherwise a bbox-region fill for that instance.
    Non-SAM tracks: boxes and ID labels only (no colored fill).
    """
    h, w = frame.shape[:2]
    flags = is_sam_refined if isinstance(is_sam_refined, list) else []
    masks = segmentation_masks if isinstance(segmentation_masks, list) else []
    out = frame.astype(np.float32)

    for i, (box, tid) in enumerate(zip(boxes, track_ids)):
        sam = bool(flags[i]) if i < len(flags) else False
        if not sam:
            continue
        color = np.array(_track_id_bgr(int(tid)), dtype=np.float32)
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        bh, bw = y2 - y1, x2 - x1
        m = masks[i] if i < len(masks) else None
        if m is not None:
            mm = np.asarray(m)
            if mm.dtype == bool:
                mb = mm
            else:
                mb = mm > 0.5
            if mb.shape[0] != bh or mb.shape[1] != bw:
                mb = cv2.resize(mb.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST).astype(bool)
            if mb.shape[:2] != roi.shape[:2]:
                mb = cv2.resize(mb.astype(np.uint8), (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            roi[mb] = roi[mb] * (1.0 - alpha) + color * alpha
        else:
            layer = np.zeros_like(roi)
            layer[:, :] = color
            roi[:] = roi * (1.0 - alpha) + layer * alpha

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    return draw_boxes_only(out_u8, boxes, track_ids)


def draw_skeleton_only(
    frame: np.ndarray,
    _boxes: List[tuple],
    _track_ids: List[int],
    poses: Dict[int, Dict],
    deviations: Optional[Dict[int, Dict[str, float]]] = None,
    shape_errors: Optional[Dict[int, Dict[str, float]]] = None,
    timing_errors: Optional[Dict[int, Dict[str, float]]] = None,
) -> np.ndarray:
    """Heatmap skeletons + keypoints only (no bounding boxes or ID labels)."""
    out = frame.copy()
    for tid, data in poses.items():
        keypoints = data["keypoints"]
        scores = data.get("scores", np.ones(17))
        if keypoints.shape[0] < 17:
            continue
        track_deviations = (deviations or {}).get(tid, {})
        track_shape = (shape_errors or {}).get(tid, {})
        track_timing = (timing_errors or {}).get(tid, {})

        for (a, b) in COCO_SKELETON_EDGES:
            if a >= keypoints.shape[0] or b >= keypoints.shape[0]:
                continue
            sa = float(scores[a]) if hasattr(scores, "__len__") else float(scores)
            sb = float(scores[b]) if hasattr(scores, "__len__") else float(scores)
            if sa < KEYPOINT_THRESHOLD or sb < KEYPOINT_THRESHOLD:
                continue
            x1, y1 = int(keypoints[a, 0]), int(keypoints[a, 1])
            x2, y2 = int(keypoints[b, 0]), int(keypoints[b, 1])
            edge = (a, b)
            if edge in EDGE_TO_DEVIATION_KEY:
                dev_key = EDGE_TO_DEVIATION_KEY[edge]
                dev = track_deviations.get(dev_key)
                base = dev_key.replace("_diff", "")
                shape_key = f"{base}_shape"
                timing_key = f"{base}_timing"
                se = track_shape.get(shape_key)
                te = track_timing.get(timing_key)
                color = _deviation_to_color(dev, se, te, joint_base=base)
            else:
                color = COLOR_OCCLUDED
            cv2.line(out, (x1, y1), (x2, y2), color, 2)

        for j in range(keypoints.shape[0]):
            sc = float(scores[j]) if hasattr(scores, "__len__") else float(scores)
            if sc < KEYPOINT_THRESHOLD:
                continue
            x, y = int(keypoints[j, 0]), int(keypoints[j, 1])
            cv2.circle(out, (x, y), 4, KEYPOINT_COLOR, -1)

    return out


def _bgr_depth_shade(base: Tuple[int, int, int], z_edge: float, z_min: float, z_max: float) -> Tuple[int, int, int]:
    """Scale BGR by depth so nearer joints (higher z in lifted coords) read slightly brighter."""
    if not (math.isfinite(z_edge) and math.isfinite(z_min) and math.isfinite(z_max)):
        return base
    span = z_max - z_min
    if span < 1e-9:
        t = 0.5
    else:
        t = (z_edge - z_min) / span
    t = max(0.0, min(1.0, t))
    # Emphasize depth: low z dimmer, high z brighter (typical after per-track normalize)
    bright = 0.58 + 0.42 * t
    return tuple(max(0, min(255, int(c * bright))) for c in base)


def draw_3d_registered_video_frame(
    frame: np.ndarray,
    interp: Dict[str, Any],
) -> np.ndarray:
    """
    Video-registered 3D preview: joints stay at ViTPose (x, y) pixels — same layout as the
    real dancers. MotionAGFormer ``z`` in ``keypoints_3d`` only modulates limb brightness
    (depth cue). Same heatmap colors as ``draw_skeleton_only`` when scoring data is present.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    poses = interp.get("poses") or {}
    deviations = interp.get("deviations") or {}
    shape_errors = interp.get("shape_errors") or {}
    timing_errors = interp.get("timing_errors") or {}

    any_k3 = False
    for _tid, data in poses.items():
        k3 = data.get("keypoints_3d")
        if k3 is None:
            continue
        k3a = np.asarray(k3)
        if k3a.ndim >= 2 and k3a.shape[0] >= 17:
            any_k3 = True
            break

    if not any_k3:
        cv2.putText(
            out,
            "No keypoints_3d — enable 3D lift (MotionAGFormer + weights)",
            (16, max(40, h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    for tid, data in poses.items():
        keypoints = data["keypoints"]
        scores = data.get("scores", np.ones(17))
        if keypoints.shape[0] < 17:
            continue
        k3 = data.get("keypoints_3d")
        zs: Optional[np.ndarray] = None
        z_min, z_max = 0.0, 1.0
        if k3 is not None:
            k3a = np.asarray(k3, dtype=np.float64)
            if k3a.shape[0] >= 17 and k3a.shape[1] >= 3:
                zs = k3a[:17, 2]
                zz = zs[np.isfinite(zs)]
                if zz.size > 0:
                    z_min = float(np.min(zz))
                    z_max = float(np.max(zz))
                    if z_max <= z_min:
                        z_max = z_min + 1e-6

        track_deviations = deviations.get(tid, {})
        track_shape = shape_errors.get(tid, {})
        track_timing = timing_errors.get(tid, {})

        for (a, b) in COCO_SKELETON_EDGES:
            if a >= keypoints.shape[0] or b >= keypoints.shape[0]:
                continue
            sa = float(scores[a]) if hasattr(scores, "__len__") else float(scores)
            sb = float(scores[b]) if hasattr(scores, "__len__") else float(scores)
            if sa < KEYPOINT_THRESHOLD or sb < KEYPOINT_THRESHOLD:
                continue
            x1, y1 = int(keypoints[a, 0]), int(keypoints[a, 1])
            x2, y2 = int(keypoints[b, 0]), int(keypoints[b, 1])
            edge = (a, b)
            if edge in EDGE_TO_DEVIATION_KEY:
                dev_key = EDGE_TO_DEVIATION_KEY[edge]
                dev = track_deviations.get(dev_key)
                base = dev_key.replace("_diff", "")
                shape_key = f"{base}_shape"
                timing_key = f"{base}_timing"
                se = track_shape.get(shape_key)
                te = track_timing.get(timing_key)
                base_color = _deviation_to_color(dev, se, te, joint_base=base)
            else:
                base_color = COLOR_OCCLUDED
            if zs is not None:
                z_edge = 0.5 * (float(zs[a]) + float(zs[b]))
                color = _bgr_depth_shade(base_color, z_edge, z_min, z_max)
            else:
                color = base_color
            cv2.line(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        for j in range(keypoints.shape[0]):
            sc = float(scores[j]) if hasattr(scores, "__len__") else float(scores)
            if sc < KEYPOINT_THRESHOLD:
                continue
            x, y = int(keypoints[j, 0]), int(keypoints[j, 1])
            if zs is not None:
                zj = float(zs[j])
                color = _bgr_depth_shade(KEYPOINT_COLOR, zj, z_min, z_max)
            else:
                color = KEYPOINT_COLOR
            cv2.circle(out, (x, y), 4, color, -1, cv2.LINE_AA)

    cv2.putText(
        out,
        "3D lift: skeleton on video (x,y = pose; shade = depth z)",
        (8, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return out


def _serialize_angle(val) -> Optional[float]:
    """Convert angle to JSON-safe float. None/NaN -> null."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return float(val)


def _poses_to_serializable(
    poses: Dict[int, Dict],
    boxes: List,
    track_ids: List,
    track_angles: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    consensus_angles: Optional[Dict[str, float]] = None,
    deviations: Optional[Dict[int, Dict[str, float]]] = None,
    shape_errors: Optional[Dict[int, Dict[str, float]]] = None,
    timing_errors: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, Dict]:
    """Convert poses dict to JSON-serializable format with angles, deviations, shape & timing errors."""
    out = {}
    tid_to_box = {tid: box for tid, box in zip(track_ids, boxes)} if track_ids else {}
    for tid, data in poses.items():
        kpts = data["keypoints"]
        box = tid_to_box.get(tid, [0, 0, 0, 0])
        entry = {
            "keypoints": [
                [float(kpts[j, 0]), float(kpts[j, 1]), float(kpts[j, 2]) if kpts.shape[1] > 2 else 1.0]
                for j in range(kpts.shape[0])
            ],
            "box": [float(b) for b in box],
        }
        if track_angles and tid in track_angles:
            entry["joint_angles"] = {
                k: _serialize_angle(v) for k, v in track_angles[tid].items()
            }
        if deviations and tid in deviations:
            entry["deviations"] = {
                k: _serialize_angle(v) for k, v in deviations[tid].items()
            }
        if shape_errors and tid in shape_errors:
            entry["shape_errors"] = {
                k: _serialize_angle(v) for k, v in shape_errors[tid].items()
            }
        if timing_errors and tid in timing_errors:
            entry["timing_errors"] = {
                k: _serialize_angle(v) for k, v in timing_errors[tid].items()
            }
        k3 = data.get("keypoints_3d")
        if k3 is not None:
            entry["keypoints_3d"] = k3
        out[str(tid)] = entry
    return out


def build_pruned_overlay_for_review(
    all_frame_data_pre: List[Dict[str, Any]],
    phase7_prune_ids: Set[int],
    raw_tracks: Dict[int, Any],
    prune_log_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Per processed frame, list pruned tracks with bbox + rule for the human review UI overlay.

    Post-pose pruned IDs: boxes come from all_frame_data_pre. Pre-pose pruned: boxes from
    raw_tracks at matching frame_idx (tracks never entered all_frame_data_pre).
    """
    ids_in_pre: Set[int] = set()
    for fd in all_frame_data_pre:
        ids_in_pre.update(fd.get("track_ids") or [])

    prune_reason: Dict[int, str] = {}
    for e in prune_log_entries:
        tid = e.get("track_id")
        if tid is not None:
            prune_reason[int(tid)] = str(e.get("rule", ""))

    pre_pose_only: Set[int] = set(prune_reason.keys()) - ids_in_pre

    # Fast lookup: pruned track -> list of (frame_idx, box)
    pre_pose_by_tid: Dict[int, List[Tuple[int, tuple]]] = {}
    for tid in pre_pose_only:
        entries = raw_tracks.get(tid) or []
        pre_pose_by_tid[tid] = [(int(e[0]), tuple(e[1])) for e in entries]

    out: List[Dict[str, Any]] = []
    for fd in all_frame_data_pre:
        fidx = int(fd["frame_idx"])
        tids = fd.get("track_ids") or []
        boxes = fd.get("boxes") or []
        tid_to_box = {int(t): b for t, b in zip(tids, boxes)}

        pruned: List[Dict[str, Any]] = []
        for tid in phase7_prune_ids:
            tid = int(tid)
            if tid in tid_to_box:
                b = tid_to_box[tid]
                pruned.append({
                    "track_id": tid,
                    "box": [float(x) for x in b],
                    "rule": prune_reason.get(tid, "post_pose"),
                })

        for tid in pre_pose_only:
            tid = int(tid)
            for frame_idx, box in pre_pose_by_tid.get(tid, []):
                if frame_idx == fidx:
                    pruned.append({
                        "track_id": tid,
                        "box": [float(x) for x in box],
                        "rule": prune_reason.get(tid, "pre_pose"),
                    })
                    break

        out.append({"pruned": pruned})
    return out


def snapshot_tid_box_map(fd: Dict[str, Any]) -> Dict[int, Tuple[float, float, float, float]]:
    """Map track_id → box for one frame (used to diff dedup / sanitize drops for review UI)."""
    tids = fd.get("track_ids") or []
    boxes = fd.get("boxes") or []
    out: Dict[int, Tuple[float, float, float, float]] = {}
    for i, tid in enumerate(tids):
        if i < len(boxes):
            b = boxes[i]
            out[int(tid)] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return out


def build_dropped_pose_overlay(
    snap_pre_dedup: List[Dict[int, Tuple[float, float, float, float]]],
    snap_post_dedup_pre_sanitize: List[Dict[int, Tuple[float, float, float, float]]],
    frames_after_sanitize: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Per processed frame: boxes that had a pose before collision dedup / sanitize but were removed.

    - collision_dedup: lower-confidence duplicate of another ID on the same person (merged).
    - bbox_sanitize: pose stripped because head keypoints were inconsistent with the bbox.
    """
    result: List[Dict[str, Any]] = []
    for i, fd in enumerate(frames_after_sanitize):
        pre = snap_pre_dedup[i]
        mid = snap_post_dedup_pre_sanitize[i]
        post_ids = set(int(t) for t in (fd.get("track_ids") or []))
        dropped: List[Dict[str, Any]] = []
        for tid, box in pre.items():
            if tid not in mid:
                dropped.append({
                    "track_id": int(tid),
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "rule": "collision_dedup",
                })
        for tid, box in mid.items():
            if tid not in post_ids:
                dropped.append({
                    "track_id": int(tid),
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "rule": "bbox_sanitize",
                })
        result.append({"dropped": dropped})
    return result


def _frame_to_json_tracks(
    fd: Dict,
    track_angles: Optional[Dict] = None,
    consensus_angles: Optional[Dict] = None,
    deviations: Optional[Dict] = None,
    shape_errors: Optional[Dict] = None,
    timing_errors: Optional[Dict] = None,
) -> Dict:
    """Build JSON-serializable frame entry with optional scoring data."""
    frame_entry = {
        "frame_idx": fd["frame_idx"],
        "tracks": _poses_to_serializable(
            fd["poses"],
            fd["boxes"],
            fd["track_ids"],
            track_angles=track_angles or fd.get("track_angles"),
            consensus_angles=consensus_angles or fd.get("consensus_angles"),
            deviations=deviations or fd.get("deviations"),
            shape_errors=shape_errors or fd.get("shape_errors"),
            timing_errors=timing_errors or fd.get("timing_errors"),
        ),
    }
    if fd.get("consensus_angles") is not None:
        frame_entry["consensus_angles"] = {
            k: _serialize_angle(v) for k, v in fd["consensus_angles"].items()
        }
    return frame_entry


def read_export_temporal_interp_config() -> Tuple[str, float]:
    """
    Export / phase-preview temporal blend between processed_fps samples (default linear).
    Env: SWAY_VIS_TEMPORAL_INTERP_MODE=linear|gsi, SWAY_VIS_GSI_LENGTHSCALE (else SWAY_GSI_LENGTHSCALE).
    """
    mode = os.environ.get("SWAY_VIS_TEMPORAL_INTERP_MODE", "linear").strip().lower() or "linear"
    v = os.environ.get("SWAY_VIS_GSI_LENGTHSCALE", "").strip()
    if v:
        gsl = float(v)
    else:
        g = os.environ.get("SWAY_GSI_LENGTHSCALE", "").strip()
        gsl = float(g) if g else 0.35
    return mode, gsl


def _pick_phase1_dets_for_interp(fd_lo: Dict, fd_hi: Dict, t: float) -> Tuple[List, List]:
    """Nearest-neighbor blend for unordered detection lists (no stable per-det ID)."""
    bl = fd_lo.get("phase1_boxes") or []
    cl = fd_lo.get("phase1_confs") or []
    bh = fd_hi.get("phase1_boxes") or []
    ch = fd_hi.get("phase1_confs") or []
    if not bl and not bh:
        return [], []
    if not bl:
        return list(bh), list(ch)
    if not bh:
        return list(bl), list(cl)
    if t < 0.5:
        return list(bl), list(cl)
    return list(bh), list(ch)


def _interpolate_frame_data(
    fd_lo: Dict,
    fd_hi: Dict,
    t: float,
    *,
    temporal_mode: str = "linear",
    gsi_lengthscale: float = 0.35,
) -> Dict:
    """Interpolate boxes, poses, deviations between two frame-data dicts. t in [0,1]."""
    all_tids = set(fd_lo.get("track_ids", [])) | set(fd_hi.get("track_ids", []))
    boxes_lo = {tid: b for tid, b in zip(fd_lo.get("track_ids", []), fd_lo.get("boxes", []))}
    boxes_hi = {tid: b for tid, b in zip(fd_hi.get("track_ids", []), fd_hi.get("boxes", []))}
    poses_lo = fd_lo.get("poses", {})
    poses_hi = fd_hi.get("poses", {})
    devs_lo = fd_lo.get("deviations", {})
    devs_hi = fd_hi.get("deviations", {})
    shape_lo = fd_lo.get("shape_errors", {})
    shape_hi = fd_hi.get("shape_errors", {})
    timing_lo = fd_lo.get("timing_errors", {})
    timing_hi = fd_hi.get("timing_errors", {})

    boxes_out = []
    track_ids_out = []
    poses_out = {}
    deviations_out = {}
    shape_out = {}
    timing_out = {}

    for tid in sorted(all_tids):
        if tid in boxes_lo and tid in boxes_hi:
            b_lo, b_hi = boxes_lo[tid], boxes_hi[tid]
            box = tuple(
                blend_scalar(t, float(b_lo[i]), float(b_hi[i]), mode=temporal_mode, gsi_l=gsi_lengthscale)
                for i in range(4)
            )
        elif tid in boxes_lo:
            box = boxes_lo[tid]
        elif tid in boxes_hi:
            box = boxes_hi[tid]
        else:
            continue

        boxes_out.append(box)
        track_ids_out.append(tid)

        if tid in poses_lo and tid in poses_hi:
            kpt_lo = poses_lo[tid]["keypoints"]
            kpt_hi = poses_hi[tid]["keypoints"]
            sc_lo = np.asarray(poses_lo[tid].get("scores", np.ones(17)))
            sc_hi = np.asarray(poses_hi[tid].get("scores", np.ones(17)))
            if sc_lo.ndim == 0:
                sc_lo = np.full(17, float(sc_lo))
            if sc_hi.ndim == 0:
                sc_hi = np.full(17, float(sc_hi))
            kpt_lo_f = np.asarray(kpt_lo, dtype=np.float64)
            kpt_hi_f = np.asarray(kpt_hi, dtype=np.float64)
            if temporal_mode == "gsi":
                kpt_interp = np.empty_like(kpt_lo_f, dtype=np.float64)
                for ii in range(kpt_interp.size):
                    kpt_interp.flat[ii] = blend_scalar(
                        t,
                        float(kpt_lo_f.flat[ii]),
                        float(kpt_hi_f.flat[ii]),
                        mode="gsi",
                        gsi_l=gsi_lengthscale,
                    )
                kpt_interp = kpt_interp.astype(np.float32)
            else:
                kpt_interp = (kpt_lo_f + t * (kpt_hi_f - kpt_lo_f)).astype(np.float32)
            sc_interp = np.maximum(sc_lo, sc_hi)  # keep higher confidence (same for linear / gsi export)
            poses_out[tid] = {"keypoints": kpt_interp, "scores": sc_interp}
            k3_lo = poses_lo[tid].get("keypoints_3d")
            k3_hi = poses_hi[tid].get("keypoints_3d")
            if k3_lo is not None and k3_hi is not None:
                a = np.asarray(k3_lo, dtype=np.float64)
                b = np.asarray(k3_hi, dtype=np.float64)
                if a.shape == b.shape:
                    if temporal_mode == "gsi":
                        out3 = np.empty_like(a)
                        for ii in range(out3.size):
                            out3.flat[ii] = blend_scalar(
                                t, float(a.flat[ii]), float(b.flat[ii]), mode="gsi", gsi_l=gsi_lengthscale
                            )
                        poses_out[tid]["keypoints_3d"] = out3.astype(np.float32).tolist()
                    else:
                        poses_out[tid]["keypoints_3d"] = (a + t * (b - a)).astype(np.float32).tolist()
            elif k3_lo is not None:
                poses_out[tid]["keypoints_3d"] = k3_lo
            elif k3_hi is not None:
                poses_out[tid]["keypoints_3d"] = k3_hi
            lift_lo = poses_lo[tid].get("lift_xyz")
            lift_hi = poses_hi[tid].get("lift_xyz")
            if lift_lo is not None and lift_hi is not None:
                L0 = np.asarray(lift_lo, dtype=np.float64)
                L1 = np.asarray(lift_hi, dtype=np.float64)
                if L0.shape == L1.shape:
                    if temporal_mode == "gsi":
                        out_l = np.empty_like(L0)
                        for ii in range(out_l.size):
                            out_l.flat[ii] = blend_scalar(
                                t, float(L0.flat[ii]), float(L1.flat[ii]), mode="gsi", gsi_l=gsi_lengthscale
                            )
                        poses_out[tid]["lift_xyz"] = out_l.astype(np.float32)
                    else:
                        poses_out[tid]["lift_xyz"] = (L0 + t * (L1 - L0)).astype(np.float32)
            elif lift_lo is not None:
                poses_out[tid]["lift_xyz"] = np.asarray(lift_lo, dtype=np.float32).copy()
            elif lift_hi is not None:
                poses_out[tid]["lift_xyz"] = np.asarray(lift_hi, dtype=np.float32).copy()
        elif tid in poses_lo:
            poses_out[tid] = poses_lo[tid]
        elif tid in poses_hi:
            poses_out[tid] = poses_hi[tid]

        if tid in devs_lo and tid in devs_hi:
            d_lo, d_hi = devs_lo[tid], devs_hi[tid]
            all_keys = set(d_lo.keys()) | set(d_hi.keys())
            deviations_out[tid] = {}
            for k in all_keys:
                v_lo = d_lo.get(k)
                v_hi = d_hi.get(k)
                if v_lo is not None and v_hi is not None:
                    if not (math.isnan(v_lo) or math.isinf(v_lo) or math.isnan(v_hi) or math.isinf(v_hi)):
                        deviations_out[tid][k] = blend_scalar(
                            t, float(v_lo), float(v_hi), mode=temporal_mode, gsi_l=gsi_lengthscale
                        )
                    else:
                        deviations_out[tid][k] = v_hi if t >= 0.5 else v_lo
                elif v_lo is not None:
                    deviations_out[tid][k] = v_lo
                elif v_hi is not None:
                    deviations_out[tid][k] = v_hi
        elif tid in devs_lo:
            deviations_out[tid] = devs_lo[tid]
        elif tid in devs_hi:
            deviations_out[tid] = devs_hi[tid]

        # Shape/timing: use first available (constant over clip)
        if tid in shape_lo:
            shape_out[tid] = shape_lo[tid]
        elif tid in shape_hi:
            shape_out[tid] = shape_hi[tid]
        if tid in timing_lo:
            timing_out[tid] = timing_lo[tid]
        elif tid in timing_hi:
            timing_out[tid] = timing_hi[tid]

    def _sam_by_tid(fd: Dict) -> Dict[int, Tuple[bool, Optional[np.ndarray]]]:
        tids = fd.get("track_ids", [])
        fl = fd.get("is_sam_refined")
        ml = fd.get("segmentation_masks")
        if not isinstance(fl, list):
            fl = []
        if not isinstance(ml, list):
            ml = []
        m: Dict[int, Tuple[bool, Optional[np.ndarray]]] = {}
        for ii, tid in enumerate(tids):
            fb = bool(fl[ii]) if ii < len(fl) else False
            mk = ml[ii] if ii < len(ml) else None
            m[int(tid)] = (fb, mk)
        return m

    sam_lo = _sam_by_tid(fd_lo)
    sam_hi = _sam_by_tid(fd_hi)
    pick_hi = t >= 0.5
    is_sam_out: List[bool] = []
    seg_masks_out: List[Optional[np.ndarray]] = []
    for tid in track_ids_out:
        primary = sam_hi if pick_hi else sam_lo
        fallback = sam_lo if pick_hi else sam_hi
        if tid in primary:
            fb, mk = primary[tid]
        elif tid in fallback:
            fb, mk = fallback[tid]
        else:
            fb, mk = False, None
        is_sam_out.append(fb)
        seg_masks_out.append(mk)

    p1b, p1c = _pick_phase1_dets_for_interp(fd_lo, fd_hi, t)

    return {
        "boxes": boxes_out,
        "track_ids": track_ids_out,
        "poses": poses_out,
        "deviations": deviations_out,
        "shape_errors": shape_out,
        "timing_errors": timing_out,
        "is_sam_refined": is_sam_out,
        "segmentation_masks": seg_masks_out,
        "phase1_boxes": p1b,
        "phase1_confs": p1c,
    }


def _source_has_audio(source_video: Path) -> bool:
    probe = shutil.which("ffprobe")
    if not probe:
        return True
    try:
        r = subprocess.run(
            [probe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             str(source_video)],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in r.stdout
    except Exception:
        return True


_mux_warned_ffmpeg = False


def _mux_audio(source_video: Path, output_video: Path) -> None:
    """
    Re-encode OpenCV's video to H.264 + yuv420p and mux AAC from the source.

    OpenCV writes MPEG-4 Part 2 (``mp4v``); many browsers show a **black** picture
    while the clock runs. Re-encoding fixes HTML5 playback. Replaces
    ``output_video`` in-place when ffmpeg succeeds.
    """
    global _mux_warned_ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if not _mux_warned_ffmpeg:
            print("  WARNING: ffmpeg not found; exported MP4 may not play in web browsers (brew install ffmpeg).")
            _mux_warned_ffmpeg = True
        return

    has_audio = _source_has_audio(source_video)
    tmp = Path(tempfile.mktemp(suffix=".mp4", dir=str(output_video.parent)))
    try:
        # H.264 + yuv420p: works in Safari, Chrome, Firefox for <video>
        vargs = [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "22",
            "-preset", "fast",
            "-movflags", "+faststart",
        ]
        if has_audio:
            cmd = [
                ffmpeg, "-y",
                "-i", str(output_video),
                "-i", str(source_video),
                *vargs,
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                str(tmp),
            ]
        else:
            cmd = [
                ffmpeg, "-y", "-i", str(output_video),
                *vargs,
                "-an",
                str(tmp),
            ]
        r = subprocess.run(cmd, capture_output=True, timeout=None)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(output_video)
        else:
            err = (r.stderr or b"").decode("utf-8", errors="replace")[:400]
            print(f"  WARNING: ffmpeg finalize failed (code {r.returncode}); browser may show black video. {err}")
    except Exception as e:
        print(f"  WARNING: ffmpeg finalize error: {e}")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _compute_track_summaries(
    all_frame_data: List[Dict],
    joint_names: List[str],
) -> Dict[str, Dict]:
    """
    Aggregate per-track stats across all frames for the captain dashboard.
    Returns {track_id: {mean_deviation, mean_shape_error, max_timing_error,
    per_joint breakdown, frame_count, sync_score}}.
    """
    track_stats: Dict[str, Dict[str, list]] = {}

    for fd in all_frame_data:
        devs = fd.get("deviations", {})
        shapes = fd.get("shape_errors", {})
        timings = fd.get("timing_errors", {})
        track_ids = fd.get("track_ids", [])

        for tid in track_ids:
            sid = str(tid)
            if sid not in track_stats:
                track_stats[sid] = {
                    "dev_vals": [], "shape_vals": [], "timing_vals": [],
                    "per_joint_dev": {j: [] for j in joint_names},
                    "per_joint_shape": {j: [] for j in joint_names},
                    "per_joint_timing": {j: [] for j in joint_names},
                    "frame_count": 0,
                }
            track_stats[sid]["frame_count"] += 1

            if tid in devs:
                for j in joint_names:
                    v = devs[tid].get(f"{j}_diff")
                    if v is not None and isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v)):
                        track_stats[sid]["dev_vals"].append(v)
                        track_stats[sid]["per_joint_dev"][j].append(v)

            if tid in shapes:
                for j in joint_names:
                    v = shapes[tid].get(f"{j}_shape")
                    if v is not None and isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v)):
                        track_stats[sid]["shape_vals"].append(v)
                        track_stats[sid]["per_joint_shape"][j].append(v)

            if tid in timings:
                for j in joint_names:
                    v = timings[tid].get(f"{j}_timing")
                    if v is not None and isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v)):
                        track_stats[sid]["timing_vals"].append(v)
                        track_stats[sid]["per_joint_timing"][j].append(v)

    summaries = {}
    for sid, stats in track_stats.items():
        dev_arr = stats["dev_vals"]
        shape_arr = stats["shape_vals"]
        timing_arr = stats["timing_vals"]

        mean_dev = float(np.mean(dev_arr)) if dev_arr else None
        mean_shape = float(np.mean(shape_arr)) if shape_arr else None
        max_timing = float(max(abs(t) for t in timing_arr)) if timing_arr else None
        mean_timing = float(np.mean([abs(t) for t in timing_arr])) if timing_arr else None

        per_joint = {}
        for j in joint_names:
            jd = stats["per_joint_dev"][j]
            js = stats["per_joint_shape"][j]
            jt = stats["per_joint_timing"][j]
            per_joint[j] = {
                "mean_deviation": _serialize_angle(float(np.mean(jd))) if jd else None,
                "mean_shape_error": _serialize_angle(float(np.mean(js))) if js else None,
                "mean_timing_error": _serialize_angle(float(np.mean([abs(t) for t in jt]))) if jt else None,
            }

        worst_joints = sorted(
            [(j, per_joint[j]["mean_shape_error"]) for j in joint_names if per_joint[j]["mean_shape_error"] is not None],
            key=lambda x: x[1],
            reverse=True,
        )

        sync_score = max(0.0, 100.0 - (mean_dev or 0.0) * 2) if mean_dev is not None else None

        summaries[sid] = {
            "frame_count": stats["frame_count"],
            "mean_deviation": _serialize_angle(mean_dev) if mean_dev is not None else None,
            "mean_shape_error": _serialize_angle(mean_shape) if mean_shape is not None else None,
            "max_timing_error": _serialize_angle(max_timing) if max_timing is not None else None,
            "mean_timing_error": _serialize_angle(mean_timing) if mean_timing is not None else None,
            "sync_score": round(sync_score, 1) if sync_score is not None else None,
            "worst_joints": [j for j, _ in worst_joints[:3]],
            "per_joint": per_joint,
        }

    return summaries


def render_and_export(
    video_path: Path,
    all_frame_data: List[Dict],
    processed_fps: float,
    native_fps: float,
    output_dir: Path,
    *,
    pruned_overlay: Optional[List[Dict[str, Any]]] = None,
    prune_entries: Optional[List[Dict[str, Any]]] = None,
    dropped_pose_overlay: Optional[List[Dict[str, Any]]] = None,
    pose_3d: Optional[Dict[str, Any]] = None,
    lab_export_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    lift_depth_series: Optional[List[Tuple[int, np.ndarray]]] = None,
) -> Dict[str, str]:
    """
    Write JSON keypoint data and rendered MP4 videos.

    JSON and pose data use processed_fps (15 FPS). Output videos are at native_fps
    with overlays interpolated from the 15 FPS data.

    Writes the primary ``{stem}_poses.mp4`` plus review variants: track IDs only,
    skeleton-only heatmap, and SAM-style colored pixels only on hybrid-SAM-refined detections.

    Returns:
        Map of logical keys to output filenames (under ``output_dir``): full, track_ids,
        skeleton, segmentation_style, and ``3d`` (``*_3d.mp4``) when ``pose_3d`` is present:
        same video frame with skeleton drawn at ViTPose (x,y); depth from lift modulates color.
        Empty dict if there are no frames to render.
    """
    output_dir = Path(output_dir)
    base_name = video_path.stem
    json_path = output_dir / "data.json"
    video_out_path = output_dir / f"{base_name}_poses.mp4"
    # Write to .part first, then os.replace so interrupted runs never leave a false-complete *_poses.mp4
    # (batch --skip-existing only looks for the final name).
    video_part_path = output_dir / f"{base_name}_poses.part.mp4"

    joint_names = [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_knee", "right_knee",
    ]

    cap_meta = cv2.VideoCapture(str(video_path))
    vw = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    vh = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap_meta.release()

    try:
        from sway.pose_lift_3d import (
            export_3d_for_viewer,
            lift_savgol_enabled,
            refresh_keypoints_3d_from_lift,
            smooth_lift_xyz_for_export,
            video_camera_from_pose_3d_camera,
        )

        if lift_savgol_enabled():
            n_tr = smooth_lift_xyz_for_export(all_frame_data)
            if n_tr > 0:
                _vc = video_camera_from_pose_3d_camera(
                    pose_3d.get("camera") if isinstance(pose_3d, dict) else None
                )
                refresh_keypoints_3d_from_lift(
                    all_frame_data, vw, vh, lift_depth_series, video_camera=_vc
                )
                if pose_3d is not None and pose_3d.get("tracks"):
                    tids_3d = sorted(int(k) for k in pose_3d["tracks"].keys())
                    pose_3d = export_3d_for_viewer(
                        all_frame_data,
                        tids_3d,
                        len(all_frame_data),
                        float(processed_fps),
                        vw,
                        vh,
                        video_camera=_vc,
                    )
                print(f"  [3D Lift] Savitzky-Golay smoothed lift_xyz for {n_tr} track(s) before export.")
    except Exception as ex:
        print(f"  [3D Lift] Export-time lift smoothing skipped: {ex}", flush=True)

    # Build JSON structure with joint angles and deviations per frame (at processed_fps)
    metadata: Dict[str, Any] = {
        "video_path": str(video_path),
        "fps": processed_fps,
        "native_fps": native_fps,
        "num_frames": len(all_frame_data),
        "keypoint_names": COCO_KEYPOINT_NAMES,
        "joint_angle_names": joint_names,
    }
    if prune_entries is not None:
        metadata["prune_entries"] = prune_entries
    if pruned_overlay is not None or dropped_pose_overlay is not None:
        metadata["review_overlay_legend"] = {
            "green": "Baked into video — final kept dancers",
            "red": "Pruned by track rules (prune_log)",
            "amber": "collision_dedup — duplicate pose dropped (same person, other ID kept)",
            "violet": "bbox_sanitize — pose dropped (head/limbs vs bbox)",
        }
    frames_json = []
    for fd in all_frame_data:
        frames_json.append(_frame_to_json_tracks(fd))

    track_summaries = _compute_track_summaries(all_frame_data, joint_names)

    payload: Dict[str, Any] = {
        "metadata": metadata,
        "track_summaries": track_summaries,
        "frames": frames_json,
    }
    if pruned_overlay is not None:
        payload["pruned_overlay"] = pruned_overlay
    if dropped_pose_overlay is not None:
        payload["dropped_pose_overlay"] = dropped_pose_overlay
    if pose_3d is not None:
        payload["pose_3d"] = pose_3d

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {json_path}")

    # Write video at native FPS with interpolated overlays
    if len(all_frame_data) == 0:
        print("  No frames to render.")
        return {}

    # V3.0: Get dimensions from video (frame may be None in streaming mode)
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap.release()

    track_ids_path = output_dir / f"{base_name}_track_ids.mp4"
    track_ids_part = output_dir / f"{base_name}_track_ids.part.mp4"
    skeleton_path = output_dir / f"{base_name}_skeleton.mp4"
    skeleton_part = output_dir / f"{base_name}_skeleton.part.mp4"
    seg_path = output_dir / f"{base_name}_sam_style.mp4"
    seg_part = output_dir / f"{base_name}_sam_style.part.mp4"
    d3_path = output_dir / f"{base_name}_3d.mp4"
    d3_part = output_dir / f"{base_name}_3d.part.mp4"

    variant_rows: List[Tuple[str, Path, Path, Any]] = [
        (
            "full",
            video_out_path,
            video_part_path,
            lambda fr, itp, _fi=0: draw_frame(
                fr,
                itp["boxes"],
                itp["track_ids"],
                itp["poses"],
                deviations=itp.get("deviations"),
                shape_errors=itp.get("shape_errors"),
                timing_errors=itp.get("timing_errors"),
            ),
        ),
        (
            "track_ids",
            track_ids_path,
            track_ids_part,
            lambda fr, itp, _fi=0: draw_boxes_only(
                fr, itp["boxes"], itp["track_ids"], poses=itp.get("poses")
            ),
        ),
        (
            "skeleton",
            skeleton_path,
            skeleton_part,
            lambda fr, itp, _fi=0: draw_skeleton_only(
                fr,
                itp["boxes"],
                itp["track_ids"],
                itp["poses"],
                deviations=itp.get("deviations"),
                shape_errors=itp.get("shape_errors"),
                timing_errors=itp.get("timing_errors"),
            ),
        ),
        (
            "segmentation_style",
            seg_path,
            seg_part,
            lambda fr, itp, _fi=0: draw_segmentation_style(
                fr,
                itp["boxes"],
                itp["track_ids"],
                is_sam_refined=itp.get("is_sam_refined"),
                segmentation_masks=itp.get("segmentation_masks"),
            ),
        ),
    ]
    if pose_3d is not None:
        variant_rows.append(
            (
                "3d",
                d3_path,
                d3_part,
                lambda fr, itp, _fi=0: draw_3d_registered_video_frame(fr, itp),
            ),
        )

    for _, _, part_p, _ in variant_rows:
        if part_p.exists():
            part_p.unlink()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = [
        cv2.VideoWriter(str(part_p), fourcc, native_fps, (w, h)) for _, _, part_p, _ in variant_rows
    ]
    num_processed = len(all_frame_data)
    vis_t_mode, vis_gsi_l = read_export_temporal_interp_config()

    cap = cv2.VideoCapture(str(video_path))
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_native_frames < 1:
        total_native_frames = max(num_processed * 2, 1)
    orig_idx = 0
    export_last_pct = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        alpha = orig_idx * processed_fps / native_fps if native_fps > 0 else orig_idx
        idx_lo = max(0, min(int(alpha), num_processed - 1))
        idx_hi = min(idx_lo + 1, num_processed - 1)
        t = max(0.0, min(1.0, alpha - idx_lo)) if idx_hi > idx_lo else 0.0

        fd_lo = all_frame_data[idx_lo]
        fd_hi = all_frame_data[idx_hi]
        interp = _interpolate_frame_data(
            fd_lo, fd_hi, t, temporal_mode=vis_t_mode, gsi_lengthscale=vis_gsi_l
        )

        for wr, (_, _, _, draw_fn) in zip(writers, variant_rows):
            wr.write(draw_fn(frame, interp, orig_idx))
        orig_idx += 1
        if lab_export_progress is not None and total_native_frames > 0:
            pct = int(100 * orig_idx / total_native_frames)
            pct = min(100, max(0, pct))
            if (
                orig_idx == 1
                or orig_idx % 45 == 0
                or pct >= export_last_pct + 5
                or orig_idx >= total_native_frames
            ):
                lab_export_progress(
                    {
                        "step": "encode_mp4_variants",
                        "frame": int(orig_idx),
                        "total_frames": int(total_native_frames),
                        "pct": int(pct),
                    }
                )
                export_last_pct = pct
    cap.release()
    for wr in writers:
        wr.release()

    rel_map: Dict[str, str] = {}
    for key, final_p, part_p, _ in variant_rows:
        _mux_audio(video_path, part_p)
        os.replace(part_p, final_p)
        rel_map[key] = final_p.name
        print(f"  Wrote {final_p} @ {native_fps:.1f} FPS")

    return rel_map


# ── Montage rendering ─────────────────────────────────────────────────


def _reencode_mp4_for_html5_video(path: Path) -> None:
    """
    OpenCV mp4v (MPEG-4 Part 2) often plays as a black / empty canvas in Safari
    and other browsers' <video>. Re-encode to H.264 yuv420p + faststart when ffmpeg exists.
    """
    ff = shutil.which("ffmpeg")
    if not ff:
        return
    tmp = path.with_suffix(".html5.tmp.mp4")
    cmd_chains: list[list[str]] = []
    if sys.platform == "darwin":
        cmd_chains.append(
            [
                ff,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c:v",
                "h264_videotoolbox",
                "-b:v",
                "8M",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ]
        )
    cmd_chains.append(
        [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp),
        ]
    )
    for cmd in cmd_chains:
        try:
            if tmp.exists():
                tmp.unlink()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 64:
                tmp.replace(path)
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def _make_title_card(
    text: str,
    width: int,
    height: int,
    fps: float,
    duration: float = 1.5,
) -> List[np.ndarray]:
    """Generate title card frames: dark background with centered white text."""
    bg = np.full((height, width, 3), (26, 26, 26), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(width, height) / 500.0
    thickness = max(2, int(scale * 2))

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    tx = (width - tw) // 2
    ty = (height + th) // 2

    cv2.putText(bg, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Subtle subtitle line
    sub = "SWAY POSE PIPELINE"
    sub_scale = scale * 0.35
    sub_thick = max(1, int(sub_scale * 2))
    (sw, sh), _ = cv2.getTextSize(sub, font, sub_scale, sub_thick)
    cv2.putText(bg, sub, ((width - sw) // 2, ty + th + 20), font, sub_scale,
                (120, 120, 120), sub_thick, cv2.LINE_AA)

    num_frames = max(1, int(fps * duration))
    return [bg] * num_frames


def render_phase_clip(
    video_path: Path,
    frame_data: List[Dict],
    phase_label: str,
    draw_fn,
    native_fps: float,
    processed_fps: float,
    output_path: Path,
    clip_duration: float = 9.0,
    start_frame: int = 0,
    caption: Optional[str] = None,
    *,
    full_length: bool = False,
    show_title_card: bool = False,
) -> Path:
    """
    Render a clip for one pipeline phase (optionally full source length).

    Args:
        video_path: Source video.
        frame_data: Per-frame overlay data (boxes, track_ids, poses, etc.).
        phase_label: Label for logs / optional title card.
        draw_fn: Callable(frame, fd) -> annotated_frame.
        native_fps: Source video FPS.
        processed_fps: FPS of frame_data.
        output_path: Where to write this clip.
        clip_duration: Seconds of video (ignored when full_length=True).
        start_frame: First source frame (ignored when full_length=True; uses 0).
        caption: Text shown at top of each overlay frame.
        full_length: If True, encode from frame 0 through end of source (or len(frame_data)).
        show_title_card: If True, prepend a short title slate (legacy montage-style).
    """
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    num_fd = len(frame_data)

    if full_length:
        start_frame = 0
        if total_source_frames > 0:
            end_frame = total_source_frames
        else:
            end_frame = max(num_fd, 1)
        end_frame = max(end_frame, 1)
    else:
        clip_frames = max(1, int(native_fps * clip_duration))
        if total_source_frames > 0:
            start_frame = min(start_frame, max(0, total_source_frames - clip_frames))
            end_frame = min(total_source_frames, start_frame + clip_frames)
        else:
            start_frame = min(start_frame, max(0, num_fd - clip_frames))
            end_frame = min(max(num_fd, 1), start_frame + clip_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, native_fps, (w, h))
    if not writer.isOpened():
        print(f"    Warning: VideoWriter failed for {output_path.name} (mp4v); phase clip may be empty.")

    vis_t_mode, vis_gsi_l = read_export_temporal_interp_config()

    if show_title_card:
        for card in _make_title_card(phase_label, w, h, native_fps):
            writer.write(card)

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    num_processed = len(frame_data)

    for orig_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        alpha = orig_idx * processed_fps / native_fps if native_fps > 0 else orig_idx
        idx_lo = max(0, min(int(alpha), num_processed - 1))
        idx_hi = min(idx_lo + 1, num_processed - 1)
        t = max(0.0, min(1.0, alpha - idx_lo)) if idx_hi > idx_lo else 0.0

        fd_lo = frame_data[idx_lo]
        fd_hi = frame_data[idx_hi]
        interp = dict(
            _interpolate_frame_data(
                fd_lo, fd_hi, t, temporal_mode=vis_t_mode, gsi_lengthscale=vis_gsi_l
            )
        )
        interp["frame_idx"] = int(orig_idx)

        annotated = draw_fn(frame, interp)
        if caption:
            cv2.rectangle(annotated, (0, 0), (w, 50), (26, 26, 26), -1)
            cv2.rectangle(annotated, (0, 48), (w, 52), (80, 80, 80), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = min(w, h) / 700.0
            thickness = max(1, int(scale))
            cv2.putText(annotated, caption, (16, 36), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        writer.write(annotated)

    cap.release()
    writer.release()
    _reencode_mp4_for_html5_video(output_path)
    print(f"    Phase clip: {output_path.name} ({end_frame - start_frame} frames @ {native_fps:.1f} fps)")
    return output_path


def stitch_montage(clip_paths: List[Path], output_path: Path, native_fps: float) -> None:
    """Concatenate phase clips into a single montage video using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("  ffmpeg not found — falling back to OpenCV concat")
        _stitch_opencv(clip_paths, output_path, native_fps)
        return

    concat_file = output_path.parent / "_montage_concat.txt"
    with open(concat_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    try:
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_file), "-c", "copy", str(output_path)],
            capture_output=True, timeout=120,
        )
        print(f"  Montage written: {output_path}")
    except Exception as e:
        print(f"  ffmpeg concat failed ({e}), falling back to OpenCV")
        _stitch_opencv(clip_paths, output_path, native_fps)
    finally:
        if concat_file.exists():
            concat_file.unlink()


def _stitch_opencv(clip_paths: List[Path], output_path: Path, fps: float) -> None:
    """Fallback: concatenate clips using OpenCV if ffmpeg is unavailable."""
    writer = None
    for clip in clip_paths:
        cap = cv2.VideoCapture(str(clip))
        if writer is None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
    if writer:
        writer.release()
    print(f"  Montage written: {output_path}")
