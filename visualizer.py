"""
Serialization & Visualization Module

Exports smoothed keypoint data to JSON and renders an MP4 video with
bounding boxes, track IDs, and skeleton overlays. Supports Phase 3 & 4:
joint angle deviation heatmap (Red/Yellow/Green per bone) and JSON export
with angles and deviations.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from pose_estimator import COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES

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

    # Draw boxes and labels — only for tracks that have valid pose data
    for i, (box, tid) in enumerate(zip(boxes, track_ids)):
        if tid not in poses:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"ID:{tid}"
        cv2.putText(
            out,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

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
) -> Dict[str, Dict]:
    """Convert poses dict to JSON-serializable format with angles and deviations."""
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
        out[str(tid)] = entry
    return out


def _frame_to_json_tracks(
    fd: Dict,
    track_angles: Optional[Dict] = None,
    consensus_angles: Optional[Dict] = None,
    deviations: Optional[Dict] = None,
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
        ),
    }
    if fd.get("consensus_angles") is not None:
        frame_entry["consensus_angles"] = {
            k: _serialize_angle(v) for k, v in fd["consensus_angles"].items()
        }
    return frame_entry


def _interpolate_frame_data(
    fd_lo: Dict,
    fd_hi: Dict,
    t: float,
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
            box = tuple(b_lo[i] + t * (b_hi[i] - b_lo[i]) for i in range(4))
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
            kpt_interp = kpt_lo + t * (kpt_hi - kpt_lo)
            sc_interp = np.maximum(sc_lo, sc_hi)  # keep higher confidence
            poses_out[tid] = {"keypoints": kpt_interp, "scores": sc_interp}
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
                        deviations_out[tid][k] = v_lo + t * (v_hi - v_lo)
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

    return {
        "boxes": boxes_out,
        "track_ids": track_ids_out,
        "poses": poses_out,
        "deviations": deviations_out,
        "shape_errors": shape_out,
        "timing_errors": timing_out,
    }


def render_and_export(
    video_path: Path,
    all_frame_data: List[Dict],
    processed_fps: float,
    native_fps: float,
    output_dir: Path,
) -> None:
    """
    Write JSON keypoint data and rendered MP4 video.

    JSON and pose data use processed_fps (15 FPS). Output video is at native_fps
    with overlays interpolated from the 15 FPS data.

    Args:
        video_path: Input video path.
        all_frame_data: List of {"frame_idx", "frame", "boxes", "track_ids", "poses"} at processed_fps.
        processed_fps: FPS of pose data (15).
        native_fps: Original video FPS for output.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    base_name = video_path.stem
    json_path = output_dir / "data.json"
    video_out_path = output_dir / f"{base_name}_poses.mp4"

    # Build JSON structure with joint angles and deviations per frame (at processed_fps)
    metadata = {
        "video_path": str(video_path),
        "fps": processed_fps,
        "native_fps": native_fps,
        "num_frames": len(all_frame_data),
        "keypoint_names": COCO_KEYPOINT_NAMES,
        "joint_angle_names": [
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_knee", "right_knee"
        ],
    }
    frames_json = []
    for fd in all_frame_data:
        frames_json.append(_frame_to_json_tracks(fd))

    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "frames": frames_json}, f, indent=2)
    print(f"  Wrote {json_path}")

    # Write video at native FPS with interpolated overlays
    if len(all_frame_data) == 0:
        print("  No frames to render.")
        return

    # V3.0: Get dimensions from video (frame may be None in streaming mode)
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out_path), fourcc, native_fps, (w, h))
    num_processed = len(all_frame_data)

    cap = cv2.VideoCapture(str(video_path))
    orig_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Map original frame index to processed frame position (V3.0: 1:1 at native FPS)
        alpha = orig_idx * processed_fps / native_fps if native_fps > 0 else orig_idx
        idx_lo = max(0, min(int(alpha), num_processed - 1))
        idx_hi = min(idx_lo + 1, num_processed - 1)
        t = max(0.0, min(1.0, alpha - idx_lo)) if idx_hi > idx_lo else 0.0

        fd_lo = all_frame_data[idx_lo]
        fd_hi = all_frame_data[idx_hi]
        interp = _interpolate_frame_data(fd_lo, fd_hi, t)

        annotated = draw_frame(
            frame,
            interp["boxes"],
            interp["track_ids"],
            interp["poses"],
            deviations=interp.get("deviations"),
            shape_errors=interp.get("shape_errors"),
            timing_errors=interp.get("timing_errors"),
        )
        writer.write(annotated)
        orig_idx += 1
    cap.release()
    writer.release()
    print(f"  Wrote {video_out_path} @ {native_fps:.1f} FPS")
