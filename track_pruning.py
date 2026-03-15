"""
Phase 3 & 6: Track Pruning — V3.1 Resolution-Aware + Smart Mirror + Completeness

V3.1 Pruning:
- Duration: Must appear in > 20% of *possible* frames (first_frame to end), floor 30 frames (V3.2).
  Does not penalize late entrants.
- Normalized Kinetic: Bbox center movement std > 5% of median dancer bbox height
  (resolution-agnostic across 720p, 1080p, 4K)
- Min Bbox Height: Track median bbox height >= 35% of global median (V3.1: far-side dancers smaller)
- Smart Mirror (Phase 6): Prune ONLY IF edge + inverted velocity + low lower-body conf
- Completeness Audit (V3.2, after Phase 4): Lifetime Peak Lower-Body — max(knee,ankle)<0.25 AND mean_shoulder>0.40
  (seated corner observers, floorwork/skirts; runs before Phase 5 to save OKS computation)
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Any

# COCO lower-body keypoint indices: knees (13, 14), ankles (15, 16) — used by Smart Mirror
LOWER_BODY_INDICES = (13, 14, 15, 16)
# Phase 6 Completeness Audit (V3.2): shoulders (5, 6), knees (13, 14), ankles (15, 16)
SHOULDER_INDICES = (5, 6)
LOWER_BODY_PEAK_INDICES = (13, 14, 15, 16)  # knees + ankles for Lifetime Peak Lower-Body Audit

# V3.3: Kinetic threshold = this fraction of median dancer bbox height (0.03 keeps side dancers)
KINETIC_STD_FRAC = 0.03

# V3.0 Smart mirror: outer 10% of frame
EDGE_MARGIN_FRAC = 0.10
# Min fraction of frames a track must be "on edge" to be considered
EDGE_PRESENCE_FRAC = 0.3


def prune_tracks(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    total_frames: int,
    min_duration_ratio: float = 0.20,
    kinetic_std_frac: float = KINETIC_STD_FRAC,
) -> Set[int]:
    """
    V3.0: Filter tracks to keep only active, moving dancers.

    Rule 1 (Duration): Remove tracks appearing in < 20% of their *possible* lifespan (first_frame
        to end), with hard floor of 30 frames (V3.2). Does not penalize late entrants.
    Rule 2 (Normalized Kinetic): Remove tracks where bbox center movement std
        < kinetic_std_frac * median_dancer_bbox_height (resolution-agnostic).
    No min_bbox_height (V3.3) — was pruning side/far dancers.

    Args:
        raw_tracks: Dict[track_id, List[(frame_idx, (x1,y1,x2,y2), conf), ...]]
        total_frames: Total number of video frames.
        min_duration_ratio: Minimum fraction of frames (default 0.20 = 20%).
        kinetic_std_frac: Kinetic threshold as fraction of median bbox height (default 0.03).

    Returns:
        Set of track_ids that pass both rules.
    """
    if total_frames <= 0:
        return set()

    # V3.1: Duration uses possible_frames (not penalizing late entrants). V3.2: floor 30 frames.
    DURATION_FLOOR_FRAMES = 30
    min_duration_frac = min_duration_ratio  # 0.20

    # First pass: duration filter + collect bbox heights for global median
    duration_surviving: Set[int] = set()
    bbox_heights: List[float] = []

    for track_id, entries in raw_tracks.items():
        track_length = len(entries)
        sorted_entries = sorted(entries, key=lambda e: e[0])
        track_first_frame = sorted_entries[0][0]
        possible_frames = total_frames - track_first_frame
        if possible_frames <= 0:
            continue
        # Must exist for > 20% of possible lifespan; floor 60 frames when possible_frames allows
        min_for_ratio = int(possible_frames * min_duration_frac) + 1  # strictly > 20%
        floor = min(DURATION_FLOOR_FRAMES, possible_frames)  # don't require 60 if < 60 possible
        min_frames = max(floor, min_for_ratio)
        if track_length < min_frames:
            continue
        duration_surviving.add(track_id)
        for _, box, _ in entries:
            x1, y1, x2, y2 = box
            h = y2 - y1
            if h > 0:
                bbox_heights.append(h)

    if not bbox_heights:
        return set()

    median_bbox_height = float(np.median(bbox_heights))
    min_movement_std = kinetic_std_frac * median_bbox_height

    surviving: Set[int] = set()
    for track_id in duration_surviving:
        entries = raw_tracks[track_id]
        centers = []
        for _, box, _ in entries:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])

        centers = np.array(centers)
        std_x = np.std(centers[:, 0])
        std_y = np.std(centers[:, 1])
        movement_std = np.sqrt(std_x ** 2 + std_y ** 2)

        if movement_std < min_movement_std:
            continue
        surviving.add(track_id)

    return surviving


def prune_geometric_mirrors(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    frame_width: int,
    frame_height: int,
    edge_margin_frac: float = EDGE_MARGIN_FRAC,
    edge_presence_frac: float = EDGE_PRESENCE_FRAC,
) -> Set[int]:
    """
    V3.0: Pre-pose geometric mirror pruning (edge + inverted velocity only).
    Saves ViTPose inference on obvious mirrors. Smart mirror (Phase 6) adds
    lower-body conf check after pose.
    """
    if frame_width <= 0 or len(surviving_ids) < 2:
        return set()

    margin_px = frame_width * edge_margin_frac
    left_edge = margin_px
    right_edge = frame_width - margin_px

    track_data: Dict[int, List[Tuple[int, float, float]]] = {}
    for track_id in surviving_ids:
        if track_id not in raw_tracks:
            continue
        entries = raw_tracks[track_id]
        if len(entries) < 2:
            continue
        centers = []
        for frame_idx, box, _ in entries:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            centers.append((frame_idx, cx))
        vxs = []
        for i in range(1, len(centers)):
            f0, cx0 = centers[i - 1]
            f1, cx1 = centers[i]
            df = f1 - f0
            if df > 0:
                vx = (cx1 - cx0) / df
                vxs.append((f1, cx1, vx))
        if not vxs:
            continue
        track_data[track_id] = vxs

    if not track_data:
        return set()

    group_vxs = [np.mean([v for _, _, v in vxs]) for vxs in track_data.values()]
    group_mean_vx = np.median(group_vxs)

    to_prune: Set[int] = set()
    for track_id, vxs in track_data.items():
        mean_vx = np.mean([v for _, _, v in vxs])
        if abs(group_mean_vx) < 1.0:
            continue
        if (mean_vx * group_mean_vx) >= 0:
            continue
        on_edge = sum(1 for _, cx, _ in vxs if cx <= left_edge or cx >= right_edge)
        if on_edge / len(vxs) >= edge_presence_frac:
            to_prune.add(track_id)

    return to_prune


# V3.2: Corner detection for completeness audit — outer fraction on each axis
CORNER_MARGIN_X_FRAC = 0.15   # outer 15% left/right
CORNER_MARGIN_Y_FRAC = 0.20   # outer 20% top/bottom


def _track_in_corner(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    track_id: int,
    frame_width: int,
    frame_height: int,
) -> bool:
    """True if track's bbox centers are predominantly in a frame corner (partial-person zone).
    Corner = outer 15% horizontally AND outer 20% vertically (e.g. seated person in corner)."""
    if track_id not in raw_tracks or frame_width <= 0 or frame_height <= 0:
        return False
    left = frame_width * CORNER_MARGIN_X_FRAC
    right = frame_width * (1.0 - CORNER_MARGIN_X_FRAC)
    top = frame_height * CORNER_MARGIN_Y_FRAC
    bottom = frame_height * (1.0 - CORNER_MARGIN_Y_FRAC)

    corner_count = 0
    for _, box, _ in raw_tracks[track_id]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        in_corner = (cx <= left or cx >= right) and (cy <= top or cy >= bottom)
        if in_corner:
            corner_count += 1
    total = len(raw_tracks[track_id])
    return total > 0 and (corner_count / total) >= 0.5


def prune_completeness_audit(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    raw_poses_by_frame: List[Dict[int, Dict]],
    frame_width: int = 1920,
    frame_height: int = 1080,
    max_lower_peak_thresh: float = 0.25,
    mean_shoulder_thresh: float = 0.40,
    corner_max_lower_thresh: float = 0.35,
    corner_mean_shoulder_thresh: float = 0.35,
) -> Set[int]:
    """
    V3.2 Phase 6: Lifetime Peak Lower-Body Audit — prune seated corner observers (head/shoulders only).

    For tracks IN A CORNER (outer 12% x and outer 15% y): use relaxed thresholds so we catch
    partial persons whose ankles/knees ViTPose may hallucinate with low conf.
    For tracks NOT in corner: use strict thresholds to avoid pruning real dancers with skirts.

    Prune IF:
      - Corner track: max_lower_peak < 0.35 AND mean_shoulder_conf > 0.35
      - Non-corner: max_lower_peak < 0.25 AND mean_shoulder_conf > 0.40 (unchanged)
    """
    to_prune: Set[int] = set()
    for track_id in surviving_ids:
        if track_id not in raw_tracks:
            continue
        lower_peak_confs: List[float] = []
        shoulder_confs: List[float] = []
        for frame_idx, poses in enumerate(raw_poses_by_frame):
            if frame_idx >= len(raw_poses_by_frame) or track_id not in poses:
                continue
            data = poses[track_id]
            scores = data.get("scores")
            if scores is None and "keypoints" in data and data["keypoints"].shape[1] > 2:
                scores = data["keypoints"][:, 2]
            if scores is None or len(scores) <= 16:
                continue
            scores = np.asarray(scores)
            for i in LOWER_BODY_PEAK_INDICES:
                if i < len(scores):
                    lower_peak_confs.append(float(scores[i]))
            for i in SHOULDER_INDICES:
                if i < len(scores):
                    shoulder_confs.append(float(scores[i]))
        if not lower_peak_confs or not shoulder_confs:
            continue
        max_lower_peak = float(np.max(lower_peak_confs))
        mean_shoulder_conf = float(np.mean(shoulder_confs))

        in_corner = _track_in_corner(raw_tracks, track_id, frame_width, frame_height)
        if in_corner:
            thresh_low = corner_max_lower_thresh
            thresh_shoulder = corner_mean_shoulder_thresh
        else:
            thresh_low = max_lower_peak_thresh
            thresh_shoulder = mean_shoulder_thresh

        if max_lower_peak < thresh_low and mean_shoulder_conf > thresh_shoulder:
            to_prune.add(track_id)
    return to_prune


def prune_smart_mirrors(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    raw_poses_by_frame: List[Dict[int, Dict]],
    frame_width: int,
    edge_margin_frac: float = EDGE_MARGIN_FRAC,
    edge_presence_frac: float = EDGE_PRESENCE_FRAC,
    min_lower_body_conf: float = 0.3,
) -> Set[int]:
    """
    V3.0 Phase 6: Smart Mirror Pruning — prune ONLY IF all three hold:
    1. Track in outer 10% of frame (left/right edge)
    2. x-velocity inverted relative to group
    3. Lower-body keypoints (ankles, knees) avg confidence < 0.3
    """
    if frame_width <= 0 or len(surviving_ids) < 2:
        return set()

    margin_px = frame_width * edge_margin_frac
    left_edge = margin_px
    right_edge = frame_width - margin_px

    track_data: Dict[int, Tuple[List[Tuple[int, float, float]], List[Tuple[int, float]]]] = {}
    for track_id in surviving_ids:
        if track_id not in raw_tracks:
            continue
        entries = raw_tracks[track_id]
        if len(entries) < 2:
            continue
        centers = []
        for frame_idx, box, _ in entries:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            centers.append((frame_idx, cx))
        vxs = []
        for i in range(1, len(centers)):
            f0, cx0 = centers[i - 1]
            f1, cx1 = centers[i]
            df = f1 - f0
            if df > 0:
                vx = (cx1 - cx0) / df
                vxs.append((f1, cx1, vx))
        if not vxs:
            continue

        lower_body_confs = []
        for frame_idx, poses in enumerate(raw_poses_by_frame):
            if frame_idx >= len(raw_poses_by_frame) or track_id not in poses:
                continue
            data = poses[track_id]
            scores = data.get("scores")
            if scores is None and "keypoints" in data and data["keypoints"].shape[1] > 2:
                scores = data["keypoints"][:, 2]
            if scores is None or len(scores) <= 16:
                lower_body_confs.append((frame_idx, 0.0))
                continue
            scores = np.asarray(scores)
            avg_conf = np.mean([float(scores[i]) for i in LOWER_BODY_INDICES])
            lower_body_confs.append((frame_idx, avg_conf))

        track_data[track_id] = (vxs, lower_body_confs)

    if not track_data:
        return set()

    group_vxs = [np.mean([v for _, _, v in vxs]) for vxs, _ in track_data.values()]
    group_mean_vx = np.median(group_vxs)

    to_prune: Set[int] = set()
    for track_id, (vxs, lb_confs) in track_data.items():
        mean_vx = np.mean([v for _, _, v in vxs])
        if abs(group_mean_vx) < 1.0:
            continue
        if (mean_vx * group_mean_vx) >= 0:
            continue
        on_edge = sum(1 for _, cx, _ in vxs if cx <= left_edge or cx >= right_edge)
        if on_edge / len(vxs) < edge_presence_frac:
            continue
        avg_lower_conf = np.mean([c for _, c in lb_confs]) if lb_confs else 0.0
        if avg_lower_conf >= min_lower_body_conf:
            continue
        to_prune.add(track_id)

    return to_prune


def prune_mirror_tracks(
    raw_poses_by_frame: List[Dict[int, Dict]],
    min_lower_body_conf: float = 0.3,
    max_low_conf_ratio: float = 0.5,
) -> Set[int]:
    """
    Legacy pose-based mirror pruning (lower-body only).
    V3.0 prefers prune_smart_mirrors which adds edge + velocity.
    """
    track_frames: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for frame_idx, poses in enumerate(raw_poses_by_frame):
        for track_id, data in poses.items():
            scores = data.get("scores")
            if scores is None:
                if "keypoints" in data and data["keypoints"].shape[1] > 2:
                    scores = data["keypoints"][:, 2]
                else:
                    continue
            scores = np.asarray(scores)
            if track_id not in track_frames:
                track_frames[track_id] = []
            track_frames[track_id].append((frame_idx, scores))

    to_prune: Set[int] = set()
    for track_id, frame_scores_list in track_frames.items():
        if len(frame_scores_list) == 0:
            continue
        low_conf_count = 0
        for _, scores in frame_scores_list:
            if len(scores) <= 16:
                low_conf_count += 1
                continue
            lower_conf = np.mean([float(scores[i]) for i in LOWER_BODY_INDICES])
            if lower_conf < min_lower_body_conf:
                low_conf_count += 1
        ratio = low_conf_count / len(frame_scores_list)
        if ratio > max_low_conf_ratio:
            to_prune.add(track_id)
    return to_prune


def raw_tracks_to_per_frame(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    total_frames: int,
    surviving_ids: Set[int],
) -> List[Dict[str, Any]]:
    """
    Convert raw tracks (filtered by surviving_ids) to per-frame format.
    """
    frame_boxes: Dict[int, List[Tuple[Tuple, int, float]]] = {
        i: [] for i in range(total_frames)
    }
    for track_id in surviving_ids:
        if track_id not in raw_tracks:
            continue
        for frame_idx, box, conf in raw_tracks[track_id]:
            frame_boxes[frame_idx].append((box, track_id, conf))

    results = []
    for frame_idx in range(total_frames):
        entries = frame_boxes[frame_idx]
        entries.sort(key=lambda x: x[1])
        boxes = [e[0] for e in entries]
        track_ids = [e[1] for e in entries]
        confs = [e[2] for e in entries]
        results.append({
            "frame_idx": frame_idx,
            "boxes": boxes,
            "track_ids": track_ids,
            "confs": confs,
        })
    return results
