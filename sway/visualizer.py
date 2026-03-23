"""
Serialization & Visualization Module

Exports smoothed keypoint data to JSON and renders an MP4 video with
bounding boxes, track IDs, and skeleton overlays. Supports Phase 3 & 4:
joint angle deviation heatmap (Red/Yellow/Green per bone) and JSON export
with angles and deviations.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

import cv2
import numpy as np

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


def draw_boxes_only(
    frame: np.ndarray,
    boxes: List[tuple],
    track_ids: List[int],
) -> np.ndarray:
    """Draw only bounding boxes and track ID labels (no skeletons)."""
    out = frame.copy()
    for box, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"ID:{tid}"
        cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, TEXT_COLOR, 1, cv2.LINE_AA)
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
        pruned_overlay: Optional per-frame list of pruned boxes for review UI (same length as frames).
        prune_entries: Optional copy of prune diagnostics (track_id, rule, …) for review UI.
        dropped_pose_overlay: Optional per-frame boxes removed by dedup/sanitize (review UI).
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

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
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

    if video_part_path.exists():
        video_part_path.unlink()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_part_path), fourcc, native_fps, (w, h))
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

    # Mux audio from source video into output (OpenCV drops audio)
    _mux_audio(video_path, video_part_path)

    os.replace(video_part_path, video_out_path)

    print(f"  Wrote {video_out_path} @ {native_fps:.1f} FPS")


# ── Montage rendering ─────────────────────────────────────────────────


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
) -> Path:
    """
    Render a short clip for one pipeline phase: title card + overlay segment.

    Args:
        video_path: Source video.
        frame_data: Per-frame overlay data (boxes, track_ids, poses, etc.).
        phase_label: Title text, e.g. "Stage 1: Detection".
        draw_fn: Callable(frame, fd) -> annotated_frame.
        native_fps: Source video FPS.
        processed_fps: FPS of frame_data.
        output_path: Where to write this clip.
        clip_duration: Seconds of video to include.
        start_frame: Source video frame to start the clip from.
        caption: Text shown at top of each clip frame (what this stage shows).
    """
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    clip_frames = int(native_fps * clip_duration)
    start_frame = min(start_frame, max(0, total_source_frames - clip_frames))
    end_frame = min(total_source_frames, start_frame + clip_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, native_fps, (w, h))

    # Write title card
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
        interp = _interpolate_frame_data(fd_lo, fd_hi, t)

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
    print(f"    Montage clip: {output_path.name}")
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
