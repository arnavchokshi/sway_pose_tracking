"""
Pre-pose + post-pose track pruning — V3.5 Non-Person Object Detection

Pre-pose rules run in main Phase 4; post-pose Tier A/B/C runs in main Phase 8:
Tier C hard skeleton auto-reject, Tier A confirmed-human
whitelist (torso + span + mirror-edge spatial sanity), Tier B weighted vote across six signals.

V3.5 Pruning (additions):
- Bbox Aspect Ratio: Pre-pose prune tracks with median width/height > 1.2 (non-person objects)
- Mean Keypoint Confidence: Post-pose prune tracks with mean ViTPose confidence < 0.45
- Keypoint Jitter: Post-pose prune tracks with frame-to-frame jitter/bbox_height > 0.10

V3.4 Pruning (retained):
- Spatial Outlier: Prune tracks > 2σ from group centroid (with min-spread floor + late-entrant safeguard)
- Traversal Detector: Prune walkers (high straightness + low acceleration variance)
- Bbox Size Outlier: Prune tracks with median bbox height outside 40–200% of group median
- Sync Score: Post-pose prune tracks with near-zero correlation to group truth angles

V3.1 Pruning (retained):
- Duration: Must appear in > 20% of *possible* frames (first_frame to end), floor 30 frames (V3.2).
  Does not penalize late entrants.
- Normalized Kinetic: Bbox center movement std > 3% of median dancer bbox height
  (resolution-agnostic across 720p, 1080p, 4K)
- Smart Mirror (Phase 6): Prune ONLY IF edge + inverted velocity + low lower-body conf
- Completeness Audit (V3.2, after Phase 4): Lifetime Peak Lower-Body — max(knee,ankle)<0.25 AND mean_shoulder>0.40
  (seated corner observers, floorwork/skirts; runs before Phase 5 to save OKS computation)
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional

from .track_observation import coerce_observation

# COCO lower-body keypoint indices: knees (13, 14), ankles (15, 16) — used by Smart Mirror
LOWER_BODY_INDICES = (13, 14, 15, 16)
# Phase 6 Completeness Audit (V3.2): shoulders (5, 6), knees (13, 14), ankles (15, 16)
SHOULDER_INDICES = (5, 6)
LOWER_BODY_PEAK_INDICES = (13, 14, 15, 16)  # knees + ankles for Lifetime Peak Lower-Body Audit

# V3.3: Kinetic threshold = this fraction of median dancer bbox height (0.02 keeps side dancers)
KINETIC_STD_FRAC = 0.02

# V3.0 Smart mirror: outer 15% of frame
EDGE_MARGIN_FRAC = 0.15
# Min fraction of frames a track must be "on edge" to be considered
EDGE_PRESENCE_FRAC = 0.3

# Phase 8 — Tier B: weighted vote across legacy Phase-7 rules (tune via params / sweep_config.yaml)
PRUNE_THRESHOLD = 0.65
PRUNING_WEIGHTS = {
    "prune_low_sync_tracks": 0.7,
    "prune_smart_mirrors": 0.9,
    "prune_completeness_audit": 0.6,
    "prune_head_only_tracks": 0.8,
    "prune_low_confidence_tracks": 0.5,
    "prune_jittery_tracks": 0.5,
}

# Optional stage polygon definition: List of (x, y) normalized coordinates [0.0, 1.0]
# e.g., [(0.1, 0.2), (0.9, 0.2), (1.0, 0.9), (0.0, 0.9)]
STAGE_POLYGON_NORMALIZED = None

def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point in polygon."""
    n = len(polygon)
    inside = False
    if n == 0: return False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def prune_by_stage_polygon(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    frame_width: int,
    frame_height: int,
    polygon_normalized: List[Tuple[float, float]] = STAGE_POLYGON_NORMALIZED,
) -> Set[int]:
    """
    Remove tracks where the median bottom-center (feet) of the bbox falls outside the explicit stage polygon.
    """
    if not polygon_normalized or frame_width <= 0 or frame_height <= 0 or not surviving_ids:
        return set()
    
    # Convert normalized polygon to pixel coordinates
    polygon_px = [(px * frame_width, py * frame_height) for px, py in polygon_normalized]
    
    to_prune = set()
    for track_id in surviving_ids:
        if track_id not in raw_tracks:
            continue
        entries = raw_tracks[track_id]
        if not entries:
            continue
        # Check median bottom-center position
        feet_x = [(box[0] + box[2]) / 2 for _, box, _ in entries]
        feet_y = [box[3] for _, box, _ in entries]
            
        med_x = float(np.median(feet_x))
        med_y = float(np.median(feet_y))
        
        if not point_in_polygon(med_x, med_y, polygon_px):
            to_prune.add(track_id)
            
    return to_prune


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


# V3.4: Spatial outlier — minimum std floor as fraction of frame dimension
SPATIAL_MIN_SPREAD_FRAC = 0.05
SPATIAL_OUTLIER_STD_FACTOR = 2.0


def _is_approaching_group(
    entries: List[Tuple[int, Tuple, float]],
    group_centroid: np.ndarray,
) -> bool:
    """
    True if track's net movement is toward the group centroid (late entrant).
    Compares distance-to-centroid of first quarter vs last quarter of the track.
    """
    if len(entries) < 4:
        return False
    sorted_entries = sorted(entries, key=lambda e: e[0])
    q = max(1, len(sorted_entries) // 4)
    early = sorted_entries[:q]
    late = sorted_entries[-q:]
    early_center = np.median([((b[0]+b[2])/2, (b[1]+b[3])/2) for _, b, _ in early], axis=0)
    late_center = np.median([((b[0]+b[2])/2, (b[1]+b[3])/2) for _, b, _ in late], axis=0)
    dist_early = np.linalg.norm(early_center - group_centroid)
    dist_late = np.linalg.norm(late_center - group_centroid)
    return dist_late < dist_early * 0.7


def prune_spatial_outliers(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    frame_width: int,
    frame_height: int,
    outlier_std_factor: float = SPATIAL_OUTLIER_STD_FACTOR,
) -> Set[int]:
    """
    V3.4: Prune tracks whose median position is far from the dance group centroid.
    Uses a minimum spread floor to prevent false pruning during tight formations.
    Late entrants approaching the group are protected.
    """
    if len(surviving_ids) < 3 or frame_width <= 0 or frame_height <= 0:
        return set()

    track_centers: Dict[int, np.ndarray] = {}
    for tid in surviving_ids:
        if tid not in raw_tracks or not raw_tracks[tid]:
            continue
        entries = raw_tracks[tid]
        centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for _, b, _ in entries]
        track_centers[tid] = np.median(centers, axis=0)

    if len(track_centers) < 3:
        return set()

    all_centers = np.array(list(track_centers.values()))
    group_centroid = np.median(all_centers, axis=0)
    group_std = np.std(all_centers, axis=0)
    min_spread = np.array([frame_width * SPATIAL_MIN_SPREAD_FRAC,
                           frame_height * SPATIAL_MIN_SPREAD_FRAC])
    group_std = np.maximum(group_std, min_spread)

    # V3.6: Foreground dancers (lower y = closer to camera) get relaxed threshold
    foreground_y_frac = 0.35
    foreground_y_thresh = frame_height * (1.0 - foreground_y_frac)

    to_prune: Set[int] = set()
    for tid, center in track_centers.items():
        normalized_dist = np.abs(center - group_centroid) / group_std
        eff_factor = outlier_std_factor
        if center[1] >= foreground_y_thresh:
            eff_factor = outlier_std_factor * 1.25
        if np.max(normalized_dist) > eff_factor:
            if not _is_approaching_group(raw_tracks[tid], group_centroid):
                to_prune.add(tid)

    return to_prune


SHORT_TRACK_MIN_FRAC = 0.15

# V3.6: Audience region — bottom-right corner often has seated/partial audience
AUDIENCE_REGION_X_MIN_FRAC = 0.75  # Right 25% of frame
AUDIENCE_REGION_Y_MIN_FRAC = 0.70  # Bottom 30% of frame
AUDIENCE_REGION_WINDOW_FRAMES = 120  # Check first ~4s at 30fps

# V3.6: Late-entrant short-span — ghosts that appear late and last briefly
LATE_ENTRANT_START_FRAC = 0.35  # Track starts after 35% of video
LATE_ENTRANT_MAX_SPAN_FRAC = 0.17  # Span (last-first) < 17% pruned. Track 61 (75 frames) kept as late entrant.
# V3.8: Spatial-aware — outer 5-10% of frame = edge entry; don't prune (fast dancer entering from side)
EDGE_ENTRANT_MARGIN_FRAC = 0.08  # Outer 8% of frame border


def _is_edge_entrant(
    first_box: Tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    edge_margin_frac: float = EDGE_ENTRANT_MARGIN_FRAC,
) -> bool:
    """True if bbox center is in outer edge margin (left/right/top/bottom border)."""
    if frame_width <= 0 or frame_height <= 0:
        return False
    cx = (first_box[0] + first_box[2]) / 2
    cy = (first_box[1] + first_box[3]) / 2
    margin_x = frame_width * edge_margin_frac
    margin_y = frame_height * edge_margin_frac
    return (
        cx <= margin_x or cx >= frame_width - margin_x
        or cy <= margin_y or cy >= frame_height - margin_y
    )


def prune_audience_region(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    frame_width: int,
    frame_height: int,
    x_min_frac: float = AUDIENCE_REGION_X_MIN_FRAC,
    y_min_frac: float = AUDIENCE_REGION_Y_MIN_FRAC,
    window_frames: int = AUDIENCE_REGION_WINDOW_FRAMES,
) -> Set[int]:
    """
    V3.6: Prune tracks whose median position (in early frames) falls in the audience zone.
    Bottom-right corner often has seated audience (head/shoulders only). Catches
    tracks like ID 58 that should have been pruned.
    """
    if frame_width <= 0 or frame_height <= 0 or not surviving_ids:
        return set()
    x_thresh = frame_width * x_min_frac
    y_thresh = frame_height * y_min_frac
    to_prune: Set[int] = set()
    for tid in surviving_ids:
        if tid not in raw_tracks or not raw_tracks[tid]:
            continue
        entries = [e for e in raw_tracks[tid] if e[0] < window_frames]
        if not entries:
            continue
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b, _ in entries]
        med_cx = float(np.median([c[0] for c in centers]))
        med_cy = float(np.median([c[1] for c in centers]))
        if med_cx >= x_thresh and med_cy >= y_thresh:
            to_prune.add(tid)
    return to_prune


def prune_late_entrant_short_span(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    total_frames: int,
    start_frac: float = LATE_ENTRANT_START_FRAC,
    max_span_frac: float = LATE_ENTRANT_MAX_SPAN_FRAC,
    frame_width: int = 0,
    frame_height: int = 0,
    edge_margin_frac: float = EDGE_ENTRANT_MARGIN_FRAC,
) -> Set[int]:
    """
    V3.6: Prune tracks that start late in the video AND have a short contiguous span.
    Catches ghosts, reflection fragments, and brief passersby that slip past duration filter
    (which uses possible_frames from first_frame, not total video).

    V3.8: Spatial-aware initialization — tracks that originate in the outer 5-10% of the
    frame (edge entry) are NOT pruned. Fast-moving dancers entering from the side would
    otherwise be dropped by min_hits / short-span logic.
    """
    if total_frames <= 0:
        return set()
    start_thresh = int(total_frames * start_frac)
    max_span = int(total_frames * max_span_frac)
    to_prune: Set[int] = set()
    for tid in surviving_ids:
        if tid not in raw_tracks or not raw_tracks[tid]:
            continue
        entries = raw_tracks[tid]
        sorted_entries = sorted(entries, key=lambda e: e[0])
        first_f = sorted_entries[0][0]
        last_f = sorted_entries[-1][0]
        span = last_f - first_f + 1
        if first_f >= start_thresh and span < max_span:
            # V3.8: Protect edge entrants — dancer entering from side of frame
            if frame_width > 0 and frame_height > 0:
                first_box = sorted_entries[0][1]
                if _is_edge_entrant(first_box, frame_width, frame_height, edge_margin_frac):
                    continue
            to_prune.add(tid)
    return to_prune


# Edge-entrant short-track: floor for possible_frames denominator
EDGE_ENTRANT_MIN_FLOOR_FRAMES = 30


def prune_short_tracks(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    total_frames: int,
    min_frac: float = SHORT_TRACK_MIN_FRAC,
    frame_width: int = 0,
    frame_height: int = 0,
    edge_margin_frac: float = EDGE_ENTRANT_MARGIN_FRAC,
) -> Set[int]:
    """
    Prune tracks present in less than min_frac of total video frames.
    Unlike the duration filter in prune_tracks (which uses possible lifespan),
    this uses total video length as the denominator so walkers and passersby
    who appear briefly get pruned regardless of when they entered.

    V3.9: Edge-entrant exemption — tracks spawning at frame edge (outer 8%)
    use possible_frames (total - first_frame) instead of total_frames, with
    net displacement toward center and truncated-track safeguard (last 5%).
    """
    if total_frames <= 0:
        return set()

    min_frames_global = int(total_frames * min_frac)
    truncate_keep_thresh = int(total_frames * 0.95)  # Last 5%: truncated, keep
    to_prune: Set[int] = set()
    margin_x = frame_width * edge_margin_frac if frame_width > 0 else 0

    for tid in surviving_ids:
        if tid not in raw_tracks:
            continue
        entries = raw_tracks[tid]
        sorted_entries = sorted(entries, key=lambda e: e[0])
        first_f = sorted_entries[0][0]
        last_f = sorted_entries[-1][0]
        n_frames = len(entries)

        # Already passes global rule — no need for edge exemption
        if n_frames >= min_frames_global:
            continue

        # Edge-entrant exemption (only when would otherwise be pruned)
        if frame_width > 0 and frame_height > 0:
            first_box = sorted_entries[0][1]
            if _is_edge_entrant(first_box, frame_width, frame_height, edge_margin_frac):
                # Truncated-track safeguard: track ends in last 5% of video
                if last_f >= truncate_keep_thresh:
                    continue  # Keep truncated late entrant
                possible_frames = total_frames - first_f
                if possible_frames <= 0:
                    to_prune.add(tid)
                    continue
                min_frames = max(
                    EDGE_ENTRANT_MIN_FLOOR_FRAMES,
                    int(possible_frames * min_frac),
                )
                if n_frames >= min_frames:
                    # Net displacement: moved toward center?
                    n_check = min(15, n_frames)
                    last_idx = n_check - 1
                    box_first = sorted_entries[0][1]
                    box_last = sorted_entries[last_idx][1]
                    cx_first = (box_first[0] + box_first[2]) / 2
                    cx_last = (box_last[0] + box_last[2]) / 2
                    net_x = cx_last - cx_first
                    # Left edge: inward = moved right (net_x > 0). Right edge: inward = moved left (net_x < 0)
                    on_left = cx_first <= margin_x
                    on_right = cx_first >= frame_width - margin_x
                    inward_ok = (on_left and net_x > 0) or (on_right and net_x < 0)
                    if inward_ok:
                        continue  # Keep edge entrant
                # Failed velocity or possible_frames: prune
                to_prune.add(tid)
                continue

        # Standard rule: min_frac of total frames
        if n_frames < min_frames_global:
            to_prune.add(tid)

    return to_prune


# V3.4: Bbox size outlier band
BBOX_SIZE_MIN_FRAC = 0.40
BBOX_SIZE_MAX_FRAC = 2.00


def prune_bbox_size_outliers(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    min_frac: float = BBOX_SIZE_MIN_FRAC,
    max_frac: float = BBOX_SIZE_MAX_FRAC,
    frame_height: int = 0,
) -> Set[int]:
    """
    V3.4: Prune tracks whose median bbox height is outside 40–200% of the group median.
    V3.6: Foreground dancers (lower in frame) get relaxed max (250%) to avoid pruning front row.
    """
    if len(surviving_ids) < 3:
        return set()

    track_median_heights: Dict[int, float] = {}
    for tid in surviving_ids:
        if tid not in raw_tracks or not raw_tracks[tid]:
            continue
        heights = [b[3] - b[1] for _, b, _ in raw_tracks[tid] if b[3] - b[1] > 0]
        if heights:
            track_median_heights[tid] = float(np.median(heights))

    if len(track_median_heights) < 3:
        return set()

    group_median_h = float(np.median(list(track_median_heights.values())))
    if group_median_h <= 0:
        return set()

    foreground_tracks: Set[int] = set()
    if frame_height > 0:
        for tid in surviving_ids:
            if tid not in raw_tracks or not raw_tracks[tid]:
                continue
            centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for _, b, _ in raw_tracks[tid]]
            med_cy = float(np.median([c[1] for c in centers]))
            if med_cy >= frame_height * 0.65:
                foreground_tracks.add(tid)

    to_prune: Set[int] = set()
    for tid, med_h in track_median_heights.items():
        ratio = med_h / group_median_h
        eff_max = max_frac * 1.25 if tid in foreground_tracks else max_frac
        if ratio < min_frac or ratio > eff_max:
            to_prune.add(tid)

    return to_prune


# V3.5: Bbox aspect ratio — person bboxes are taller than wide
ASPECT_RATIO_MAX = 2.5


def prune_bad_aspect_ratio(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    max_aspect: float = ASPECT_RATIO_MAX,
) -> Set[int]:
    """
    V3.5: Prune tracks whose median bbox is wider than tall (non-person objects
    like bags, tables, equipment). Person bounding boxes always have height > width.
    Only applies when 3+ tracks survive.
    """
    if len(surviving_ids) < 3:
        return set()

    to_prune: Set[int] = set()
    for tid in surviving_ids:
        if tid not in raw_tracks or not raw_tracks[tid]:
            continue
        ratios = []
        for _, box, _ in raw_tracks[tid]:
            w = box[2] - box[0]
            h = box[3] - box[1]
            if h > 0:
                ratios.append(w / h)
        if ratios and float(np.median(ratios)) > max_aspect:
            to_prune.add(tid)

    return to_prune


# V3.5: Post-pose mean keypoint confidence — non-persons get uniformly low scores
MEAN_CONFIDENCE_MIN = 0.45


def prune_low_confidence_tracks(
    surviving_ids: Set[int],
    raw_poses_by_frame: List[Dict[int, Dict]],
    min_mean_conf: float = MEAN_CONFIDENCE_MIN,
    confirmed_humans: Optional[Set[int]] = None,
    phase7_prune_log: Optional[list] = None,
    raw_tracks: Optional[Dict[int, List[Tuple[int, Tuple, float]]]] = None,
    frame_width: int = 0,
    frame_height: int = 0,
    total_frames: int = 0,
) -> Set[int]:
    """
    V3.5 Post-pose: Prune tracks where keypoint confidence is consistently low.
    ViTPose on non-person objects produces uniformly low confidence (~0.35) vs
    real dancers (0.81+).

    Uses the 75th percentile of per-frame mean confidence instead of a raw
    overall mean.  This prevents false-pruning of dancers who enter from the
    frame edge: early frames (partially visible) drag down the mean, but once
    fully visible the per-frame confidence rises well above threshold.
    Non-person objects remain uniformly low across all frames.

    Only applies when 3+ tracks survive.
    """
    if len(surviving_ids) < 3:
        return set()

    to_prune: Set[int] = set()
    for tid in surviving_ids:
        if _phase7_should_skip_confirmed(
            tid,
            confirmed_humans,
            phase7_prune_log,
            raw_tracks,
            "prune_low_confidence_tracks",
            frame_width,
            frame_height,
            total_frames,
        ):
            continue
        frame_means: List[float] = []
        for poses in raw_poses_by_frame:
            if tid not in poses:
                continue
            data = poses[tid]
            scores = data.get("scores")
            if scores is None and "keypoints" in data:
                kp = data["keypoints"]
                if hasattr(kp, 'shape') and kp.shape[1] > 2:
                    scores = kp[:, 2]
                elif isinstance(kp, list) and kp and len(kp[0]) > 2:
                    scores = [k[2] for k in kp]
            if scores is not None:
                frame_means.append(float(np.mean([float(s) for s in scores])))
        if frame_means and float(np.percentile(frame_means, 75)) < min_mean_conf:
            to_prune.add(tid)

    return to_prune


# V3.5: Post-pose keypoint jitter — non-persons have wildly unstable keypoints
JITTER_RATIO_MAX = 0.10


def prune_jittery_tracks(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    raw_poses_by_frame: List[Dict[int, Dict]],
    max_jitter: float = JITTER_RATIO_MAX,
    confirmed_humans: Optional[Set[int]] = None,
    phase7_prune_log: Optional[list] = None,
    frame_width: int = 0,
    frame_height: int = 0,
    total_frames: int = 0,
) -> Set[int]:
    """
    V3.5 Post-pose: Prune tracks with excessive frame-to-frame keypoint jitter
    normalized by bbox height. Non-person objects produce jitter/h ~0.20 vs
    real dancers at 0.01-0.03.
    Only applies when 3+ tracks survive.
    """
    if len(surviving_ids) < 3:
        return set()

    to_prune: Set[int] = set()
    for tid in surviving_ids:
        if _phase7_should_skip_confirmed(
            tid,
            confirmed_humans,
            phase7_prune_log,
            raw_tracks,
            "prune_jittery_tracks",
            frame_width,
            frame_height,
            total_frames,
        ):
            continue
        if tid not in raw_tracks or not raw_tracks[tid]:
            continue
        heights = [b[3] - b[1] for _, b, _ in raw_tracks[tid] if b[3] - b[1] > 0]
        if not heights:
            continue
        med_h = float(np.median(heights))
        if med_h <= 0:
            continue

        kpts_by_frame: Dict[int, np.ndarray] = {}
        for fidx, poses in enumerate(raw_poses_by_frame):
            if tid not in poses:
                continue
            data = poses[tid]
            kp = data.get("keypoints")
            if kp is None:
                continue
            kp = np.asarray(kp)
            if kp.ndim == 2 and kp.shape[0] >= 17:
                kpts_by_frame[fidx] = kp[:17, :2] if kp.shape[1] >= 2 else kp[:17]

        sorted_fidxs = sorted(kpts_by_frame.keys())
        displacements: List[float] = []
        for i in range(1, len(sorted_fidxs)):
            f_prev, f_curr = sorted_fidxs[i - 1], sorted_fidxs[i]
            if f_curr - f_prev > 2:
                continue
            diff = kpts_by_frame[f_curr] - kpts_by_frame[f_prev]
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            displacements.extend(dists.tolist())

        if displacements and np.mean(displacements) / med_h > max_jitter:
            to_prune.add(tid)

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
    max_lower_peak_thresh: float = 0.35,
    mean_shoulder_thresh: float = 0.35,
    corner_max_lower_thresh: float = 0.45,
    corner_mean_shoulder_thresh: float = 0.25,
    confirmed_humans: Optional[Set[int]] = None,
    phase7_prune_log: Optional[list] = None,
    total_frames: int = 0,
) -> Set[int]:
    """
    V3.2 Phase 6: Lifetime Peak Lower-Body Audit — prune seated corner observers (head/shoulders only).

    For tracks IN A CORNER (outer 12% x and outer 15% y): use relaxed thresholds so we catch
    partial persons whose ankles/knees ViTPose may hallucinate with low conf.
    For tracks NOT in corner: use strict thresholds to avoid pruning real dancers with skirts.

    Prune IF:
      - Corner track: max_lower_peak < 0.45 AND mean_shoulder_conf > 0.25
      - Non-corner: max_lower_peak < 0.35 AND mean_shoulder_conf > 0.35
    """
    to_prune: Set[int] = set()
    for track_id in surviving_ids:
        if _phase7_should_skip_confirmed(
            track_id,
            confirmed_humans,
            phase7_prune_log,
            raw_tracks,
            "prune_completeness_audit",
            frame_width,
            frame_height,
            total_frames,
        ):
            continue
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
        # Use 95th percentile to ignore 1-2 frame hallucination spikes
        robust_lower_peak = float(np.percentile(lower_peak_confs, 95))
        mean_shoulder_conf = float(np.mean(shoulder_confs))

        in_corner = _track_in_corner(raw_tracks, track_id, frame_width, frame_height)
        if in_corner:
            thresh_low = corner_max_lower_thresh
            thresh_shoulder = corner_mean_shoulder_thresh
        else:
            thresh_low = max_lower_peak_thresh
            thresh_shoulder = mean_shoulder_thresh

        if robust_lower_peak < thresh_low and mean_shoulder_conf > thresh_shoulder:
            to_prune.add(track_id)
    return to_prune


# V3.9: Head-only prune — audience heads in back (small bbox, low lower-body, peripheral)
HEAD_ONLY_LOWER_CONF_THRESH = 0.1
HEAD_ONLY_LOWER_CONF_FRAC = 0.9
HEAD_ONLY_BBOX_FRAC_OF_GROUP = 0.4
HEAD_ONLY_SPATIAL_OUTLIER_FACTOR = 2.0


def prune_head_only_tracks(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    raw_poses_by_frame: List[Dict[int, Dict]],
    frame_width: int = 1920,
    frame_height: int = 1080,
    lower_conf_thresh: float = HEAD_ONLY_LOWER_CONF_THRESH,
    lower_conf_frac: float = HEAD_ONLY_LOWER_CONF_FRAC,
    bbox_frac_of_group: float = HEAD_ONLY_BBOX_FRAC_OF_GROUP,
    confirmed_humans: Optional[Set[int]] = None,
    phase7_prune_log: Optional[list] = None,
    total_frames: int = 0,
) -> Set[int]:
    """
    V3.9: Prune head-only audience tracks (small bbox, low lower-body conf, spatially peripheral).
    Catches people in the back whose heads pop into frame. Does NOT prune distant full-body
    dancers (spatially central) — only peripheral (corner or far from group).
    """
    if len(surviving_ids) < 3 or frame_width <= 0 or frame_height <= 0:
        return set()

    # Group median bbox height
    track_heights: Dict[int, float] = {}
    track_centers: Dict[int, np.ndarray] = {}
    for tid in surviving_ids:
        if tid not in raw_tracks or not raw_tracks[tid]:
            continue
        heights = [b[3] - b[1] for _, b, _ in raw_tracks[tid] if b[3] - b[1] > 0]
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b, _ in raw_tracks[tid]]
        if heights and centers:
            track_heights[tid] = float(np.median(heights))
            track_centers[tid] = np.median(centers, axis=0)

    if len(track_heights) < 3:
        return set()

    group_median_h = float(np.median(list(track_heights.values())))
    all_centers = np.array(list(track_centers.values()))
    group_centroid = np.median(all_centers, axis=0)
    group_std = np.std(all_centers, axis=0)
    min_spread = np.array([frame_width * SPATIAL_MIN_SPREAD_FRAC, frame_height * SPATIAL_MIN_SPREAD_FRAC])
    group_std = np.maximum(group_std, min_spread)

    to_prune: Set[int] = set()
    for tid in surviving_ids:
        if _phase7_should_skip_confirmed(
            tid,
            confirmed_humans,
            phase7_prune_log,
            raw_tracks,
            "prune_head_only_tracks",
            frame_width,
            frame_height,
            total_frames,
        ):
            continue
        if tid not in raw_tracks or tid not in track_heights or tid not in track_centers:
            continue
        med_h = track_heights[tid]
        if med_h >= bbox_frac_of_group * group_median_h:
            continue  # (b) Not small enough
        if med_h <= 0:
            continue

        # (a) Lower-body conf < thresh for >= frac of frames
        frame_count = 0
        low_lower_count = 0
        for frame_idx, poses in enumerate(raw_poses_by_frame):
            if frame_idx >= len(raw_poses_by_frame) or tid not in poses:
                continue
            data = poses[tid]
            scores = data.get("scores")
            if scores is None and "keypoints" in data and data["keypoints"].shape[1] > 2:
                scores = data["keypoints"][:, 2]
            if scores is None or len(scores) <= 16:
                continue
            scores = np.asarray(scores)
            avg_lower = np.mean([float(scores[i]) for i in LOWER_BODY_INDICES if i < len(scores)])
            frame_count += 1
            if avg_lower < lower_conf_thresh:
                low_lower_count += 1
        if frame_count == 0 or (low_lower_count / frame_count) < lower_conf_frac:
            continue  # (a) Failed

        # (c) Spatially peripheral: in corner OR far from group
        in_corner = _track_in_corner(raw_tracks, tid, frame_width, frame_height)
        center = track_centers[tid]
        normalized_dist = np.abs(center - group_centroid) / group_std
        is_outlier = np.max(normalized_dist) > HEAD_ONLY_SPATIAL_OUTLIER_FACTOR
        if not in_corner and not is_outlier:
            continue  # Central — likely distant full-body dancer, not audience head

        to_prune.add(tid)
    return to_prune


def prune_smart_mirrors(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    surviving_ids: Set[int],
    raw_poses_by_frame: List[Dict[int, Dict]],
    frame_width: int,
    edge_margin_frac: float = EDGE_MARGIN_FRAC,
    edge_presence_frac: float = EDGE_PRESENCE_FRAC,
    min_lower_body_conf: float = 0.3,
    confirmed_humans: Optional[Set[int]] = None,
    phase7_prune_log: Optional[list] = None,
    frame_height: int = 1080,
    total_frames: int = 0,
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
        if _phase7_should_skip_confirmed(
            track_id,
            confirmed_humans,
            phase7_prune_log,
            raw_tracks,
            "prune_smart_mirrors",
            frame_width,
            frame_height,
            total_frames,
        ):
            continue
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


def _track_mirror_edge_frame_fraction(
    all_frame_data_pre: List[Dict[str, Any]],
    tid: int,
    frame_width: int,
    edge_margin_frac: float = EDGE_MARGIN_FRAC,
) -> Optional[float]:
    """
    Among frames where `tid` has a bbox in all_frame_data_pre, fraction of frames whose bbox
    center lies in the mirror-prone left/right edge strips (same geometry as smart mirror).
    Returns None if there are no such frames.
    """
    if frame_width <= 0:
        return None
    margin_px = frame_width * edge_margin_frac
    left_edge = margin_px
    right_edge = frame_width - margin_px
    on_edge = 0
    total = 0
    for fd in all_frame_data_pre:
        tids = fd.get("track_ids") or []
        boxes = fd.get("boxes") or []
        if tid not in tids:
            continue
        idx = tids.index(tid)
        if idx >= len(boxes):
            continue
        box = boxes[idx]
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        total += 1
        if cx <= left_edge or cx >= right_edge:
            on_edge += 1
    if total == 0:
        return None
    return on_edge / total


def compute_confirmed_human_set(
    all_frame_data_pre: List[Dict[str, Any]],
    total_frames: int,
    min_torso_conf: float = 0.5,
    min_frame_frac: float = 0.40,
    min_span_frac: float = 0.10,
    frame_width: int = 0,
    edge_margin_frac: float = EDGE_MARGIN_FRAC,
    max_edge_presence_frac: float = 0.20,
) -> Set[int]:
    """
    Tier A (protected from Tier B voting): strong mean torso conf on enough pose frames,
    temporal span >= min_span_frac of video (short FP tracks fail), and spatial sanity —
    not in mirror-prone edge zone on >80% of bbox frames (i.e. edge presence <= 20%).
    When frame_width <= 0, spatial sanity is skipped (tests / callers without geometry).
    """
    TORSO_KPT_INDICES = (5, 6, 11, 12)
    frame_counts: Dict[int, int] = {}
    confident_counts: Dict[int, int] = {}
    first_frame: Dict[int, int] = {}
    last_frame: Dict[int, int] = {}

    for fd in all_frame_data_pre:
        fidx = int(fd.get("frame_idx", 0))
        for tid, pose_data in (fd.get("poses") or {}).items():
            tid = int(tid)
            scores = pose_data.get("scores")
            if scores is None:
                continue
            scores = np.asarray(scores)
            frame_counts[tid] = frame_counts.get(tid, 0) + 1
            first_frame.setdefault(tid, fidx)
            last_frame[tid] = fidx
            torso = [float(scores[i]) for i in TORSO_KPT_INDICES if i < len(scores)]
            if torso and (sum(torso) / len(torso)) >= min_torso_conf:
                confident_counts[tid] = confident_counts.get(tid, 0) + 1

    confirmed: Set[int] = set()
    tf = max(total_frames, 1)
    for tid, total in frame_counts.items():
        span = (last_frame.get(tid, 0) - first_frame.get(tid, 0)) / tf
        conf_frac = confident_counts.get(tid, 0) / max(total, 1)
        if conf_frac < min_frame_frac or span < min_span_frac:
            continue
        if frame_width > 0:
            edge_frac = _track_mirror_edge_frame_fraction(
                all_frame_data_pre, tid, frame_width, edge_margin_frac=edge_margin_frac
            )
            # Require bbox-backed spatial check; edge in >20% of bbox frames => fail
            if edge_frac is None or edge_frac > max_edge_presence_frac:
                continue
        confirmed.add(tid)
    return confirmed


def _phase7_should_skip_confirmed(
    tid: int,
    confirmed_humans: Optional[Set[int]],
    phase7_prune_log: Optional[list],
    raw_tracks: Optional[Dict[int, List[Tuple[int, Tuple, float]]]],
    rule_name: str,
    frame_width: int,
    frame_height: int,
    total_frames: int,
) -> bool:
    if not confirmed_humans or tid not in confirmed_humans:
        return False
    if phase7_prune_log is not None and raw_tracks is not None:
        phase7_prune_log.append(
            get_pruned_track_info(
                raw_tracks,
                tid,
                rule_name,
                frame_width,
                frame_height,
                total_frames,
                decision="exempted",
                exempt_reason="confirmed_human",
            )
        )
    return True


# Tier C: hard auto-reject — no confident skeleton (replaces geometric FP rules for bad poses)
ULTRA_LOW_SKELETON_MEAN = 0.15
ULTRA_LOW_SKELETON_FRAME_FRAC = 0.80


def _extract_pose_scores_list(pose: Dict) -> List[float]:
    scores = pose.get("scores")
    if scores is None and "keypoints" in pose:
        kp = pose["keypoints"]
        if hasattr(kp, "shape") and kp.ndim == 2 and kp.shape[1] > 2:
            scores = kp[:, 2].tolist()
        elif isinstance(kp, list) and kp and len(kp[0]) > 2:
            scores = [float(k[2]) for k in kp]
    if scores is None:
        return []
    return [float(s) for s in scores]


def is_auto_reject(
    tid: int,
    raw_poses_by_frame: List[Dict[int, Dict]],
    threshold: float = ULTRA_LOW_SKELETON_MEAN,
    min_frac: float = ULTRA_LOW_SKELETON_FRAME_FRAC,
) -> bool:
    """
    Tier C: True if the track should be auto-rejected — no ViTPose skeleton confidence.
    Mean keypoint confidence < threshold on >= min_frac of pose frames, or no pose at all.
    """
    low_conf_frames = 0
    total_pose_frames = 0
    for poses in raw_poses_by_frame:
        pose = poses.get(tid)
        if pose is None:
            continue
        total_pose_frames += 1
        scores = _extract_pose_scores_list(pose)
        if not scores:
            low_conf_frames += 1
            continue
        if (sum(scores) / len(scores)) < threshold:
            low_conf_frames += 1
    if total_pose_frames == 0:
        return True
    return (low_conf_frames / total_pose_frames) >= min_frac


def prune_ultra_low_skeleton_tracks(
    surviving_ids: Set[int],
    raw_poses_by_frame: List[Dict[int, Dict]],
    mean_thresh: float = ULTRA_LOW_SKELETON_MEAN,
    min_low_frac: float = ULTRA_LOW_SKELETON_FRAME_FRAC,
    confirmed_humans: Optional[Set[int]] = None,
    phase7_prune_log: Optional[list] = None,
    raw_tracks: Optional[Dict[int, List[Tuple[int, Tuple, float]]]] = None,
    frame_width: int = 0,
    frame_height: int = 0,
    total_frames: int = 0,
) -> Set[int]:
    """
    Tier C: auto-reject from skeleton confidence only. No Tier-A exemption — if there is no
    confident skeleton, drop the track. Legacy kwargs are ignored.
    """
    _ = (
        confirmed_humans,
        phase7_prune_log,
        raw_tracks,
        frame_width,
        frame_height,
        total_frames,
    )
    return {tid for tid in surviving_ids if is_auto_reject(tid, raw_poses_by_frame, mean_thresh, min_low_frac)}


# V3.4: Sync score pruning — minimum correlation to keep a track
SYNC_SCORE_MIN = 0.10


def prune_low_sync_tracks(
    all_frame_data: List[Dict[str, Any]],
    surviving_ids: Set[int],
    min_sync_score: float = SYNC_SCORE_MIN,
    raw_tracks: Optional[Dict[int, List[Tuple[int, Tuple, float]]]] = None,
    total_frames: int = 0,
    confirmed_humans: Optional[Set[int]] = None,
    phase7_prune_log: Optional[list] = None,
    frame_width: int = 0,
    frame_height: int = 0,
) -> Set[int]:
    """
    V3.4 Post-pose pruning: remove tracks with near-zero angular correlation to the group truth.
    Non-dancers (standing, sitting, walking) have ~0 correlation across all 6 joint angles.
    Uses max-across-joints so a soloist syncing on even 1 joint is protected.
    V3.7: Exempt late-entrant short tracks — they have few frames so correlation is unreliable.
    """
    from .kinematics import compute_joint_angles_vectorized, JOINT_NAMES
    from .scoring import _build_keypoints_array, _compute_group_truth, CONFIDENCE_THRESHOLD

    if len(surviving_ids) < 3 or len(all_frame_data) < 30:
        return set()

    kpts_arr, tid_to_idx, track_ids_per_frame = _build_keypoints_array(all_frame_data)
    num_frames, num_tracks, _, _ = kpts_arr.shape
    if num_tracks < 3:
        return set()

    angles = compute_joint_angles_vectorized(kpts_arr, confidence_threshold=CONFIDENCE_THRESHOLD)
    group_truth = _compute_group_truth(angles)

    to_prune: Set[int] = set()
    for tid in surviving_ids:
        if _phase7_should_skip_confirmed(
            tid,
            confirmed_humans,
            phase7_prune_log,
            raw_tracks,
            "prune_low_sync_tracks",
            frame_width,
            frame_height,
            total_frames,
        ):
            continue
        if tid not in tid_to_idx:
            continue
        # Late-entrant safeguard: tracks starting after 25% of video with <100 frames often
        # have unreliable sync (sparse pose data); may be real late entrant.
        if raw_tracks and total_frames > 0 and tid in raw_tracks:
            entries = raw_tracks[tid]
            first_f = min(e[0] for e in entries)
            n_frames = len(entries)
            if first_f >= total_frames * 0.25 and n_frames < 100:
                continue  # Don't sync-prune; could be late entrant
        t_idx = tid_to_idx[tid]
        max_corr = 0.0
        for j in range(6):
            track_series = angles[:, t_idx, j]
            group_series = group_truth[:, j]
            valid = ~np.isnan(track_series) & ~np.isnan(group_series)
            if np.sum(valid) < 15:
                continue
            t_vals = track_series[valid]
            g_vals = group_series[valid]
            t_std = np.std(t_vals)
            g_std = np.std(g_vals)
            if t_std < 1e-6 or g_std < 1e-6:
                continue
            corr = float(np.corrcoef(t_vals, g_vals)[0, 1])
            if np.isnan(corr):
                continue
            max_corr = max(max_corr, abs(corr))

        if max_corr < min_sync_score:
            to_prune.add(tid)

    return to_prune


def compute_phase7_voting_prune_set(
    all_frame_data_pre: List[Dict[str, Any]],
    candidate_ids: Set[int],
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    raw_poses_by_frame: List[Dict[int, Dict]],
    frame_width: int,
    frame_height: int,
    total_frames: int,
    confirmed_humans: Set[int],
    pruning_weights: Dict[str, float],
    prune_threshold: float,
    min_sync_score: float = SYNC_SCORE_MIN,
    edge_margin_frac: float = EDGE_MARGIN_FRAC,
    edge_presence_frac: float = EDGE_PRESENCE_FRAC,
    min_lower_body_conf: float = 0.3,
    min_mean_conf: float = MEAN_CONFIDENCE_MIN,
    max_jitter: float = JITTER_RATIO_MAX,
    phase7_prune_log: Optional[list] = None,
) -> Tuple[Set[int], Dict[int, Dict[str, Any]]]:
    """
    Tier B: run each Phase-7 rule without confirmed-human skips, map hit/miss to 0/1 scores,
    prune when sum(weight_i * score_i) > prune_threshold. Tier-A tracks are never pruned here.
    """
    _no_skip: Dict[str, Any] = {"confirmed_humans": None, "phase7_prune_log": None}

    sync_ids = prune_low_sync_tracks(
        all_frame_data_pre,
        candidate_ids,
        min_sync_score=min_sync_score,
        raw_tracks=raw_tracks,
        total_frames=total_frames,
        frame_width=frame_width,
        frame_height=frame_height,
        **_no_skip,
    )
    mirror_ids = prune_smart_mirrors(
        raw_tracks,
        candidate_ids,
        raw_poses_by_frame,
        frame_width,
        edge_margin_frac=edge_margin_frac,
        edge_presence_frac=edge_presence_frac,
        min_lower_body_conf=min_lower_body_conf,
        frame_height=frame_height,
        total_frames=total_frames,
        **_no_skip,
    )
    completeness_ids = prune_completeness_audit(
        raw_tracks,
        candidate_ids,
        raw_poses_by_frame,
        frame_width,
        frame_height,
        total_frames=total_frames,
        **_no_skip,
    )
    head_ids = prune_head_only_tracks(
        raw_tracks,
        candidate_ids,
        raw_poses_by_frame,
        frame_width,
        frame_height,
        total_frames=total_frames,
        **_no_skip,
    )
    low_conf_ids = prune_low_confidence_tracks(
        candidate_ids,
        raw_poses_by_frame,
        min_mean_conf=min_mean_conf,
        raw_tracks=raw_tracks,
        frame_width=frame_width,
        frame_height=frame_height,
        total_frames=total_frames,
        **_no_skip,
    )
    jitter_ids = prune_jittery_tracks(
        raw_tracks,
        candidate_ids,
        raw_poses_by_frame,
        max_jitter=max_jitter,
        frame_width=frame_width,
        frame_height=frame_height,
        total_frames=total_frames,
        **_no_skip,
    )

    rule_sets = {
        "prune_low_sync_tracks": sync_ids,
        "prune_smart_mirrors": mirror_ids,
        "prune_completeness_audit": completeness_ids,
        "prune_head_only_tracks": head_ids,
        "prune_low_confidence_tracks": low_conf_ids,
        "prune_jittery_tracks": jitter_ids,
    }

    voted: Set[int] = set()
    telemetry: Dict[int, Dict[str, Any]] = {}
    for tid in candidate_ids:
        if tid in confirmed_humans:
            continue
        scores_hit = {name: (1.0 if tid in pset else 0.0) for name, pset in rule_sets.items()}
        wsum = sum(pruning_weights.get(name, 0.0) * v for name, v in scores_hit.items())
        telemetry[tid] = {
            "rule_hits": scores_hit,
            "weighted_sum": round(wsum, 4),
            "prune_threshold": prune_threshold,
        }
        if wsum > prune_threshold:
            voted.add(tid)

    if phase7_prune_log is not None:
        for tid in sorted(voted):
            info = get_pruned_track_info(
                raw_tracks,
                tid,
                "phase7_voting",
                frame_width,
                frame_height,
                total_frames=total_frames,
            )
            info["tier_b_vote"] = telemetry.get(tid, {})
            phase7_prune_log.append(info)

    return voted, telemetry


def get_pruned_track_info(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    tid: int,
    rule_name: str,
    frame_width: int = 0,
    frame_height: int = 0,
    total_frames: int = 0,
    decision: str = "pruned",
    threshold: Optional[float] = None,
    actual_value: Optional[float] = None,
    exempt_reason: Optional[str] = None,
) -> dict:
    """Return structured info for one pruned track (for JSON logging)."""
    entries = raw_tracks.get(tid, [])
    if not entries:
        out = {"track_id": tid, "rule": rule_name, "n_frames": 0, "decision": decision}
        if threshold is not None:
            out["threshold"] = threshold
        if actual_value is not None:
            out["actual_value"] = actual_value
        if exempt_reason:
            out["reason"] = exempt_reason
        return out
    centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for _, b, _ in entries]
    med_cx = float(np.median([c[0] for c in centers]))
    med_cy = float(np.median([c[1] for c in centers]))
    heights = [b[3] - b[1] for _, b, _ in entries if b[3] - b[1] > 0]
    widths = [b[2] - b[0] for _, b, _ in entries if b[2] - b[0] > 0]
    med_h = float(np.median(heights)) if heights else 0.0
    med_w = float(np.median(widths)) if widths else 0.0
    first_f = min(e[0] for e in entries)
    last_f = max(e[0] for e in entries)
    span = max(0, last_f - first_f)
    info = {
        "track_id": tid,
        "rule": rule_name,
        "n_frames": len(entries),
        "frame_range": [first_f, last_f],
        "frame_span": [first_f, last_f],
        "median_center": [round(med_cx, 1), round(med_cy, 1)],
        "median_h": round(med_h, 1),
        "median_w": round(med_w, 1),
        "aspect_ratio": round(med_w / med_h, 2) if med_h > 0 else None,
        "decision": decision,
    }
    if med_w > 0 and med_h > 0:
        info["bbox_xyxy_median"] = [
            round(med_cx - med_w / 2, 1),
            round(med_cy - med_h / 2, 1),
            round(med_cx + med_w / 2, 1),
            round(med_cy + med_h / 2, 1),
        ]
    if total_frames > 0:
        info["frames_in_video_pct"] = round(span / total_frames, 4)
    if threshold is not None:
        info["threshold"] = threshold
    if actual_value is not None:
        info["actual_value"] = actual_value
    if exempt_reason:
        info["reason"] = exempt_reason
    if frame_width > 0 and frame_height > 0:
        info["pos_pct"] = [round(100 * med_cx / frame_width, 1), round(100 * med_cy / frame_height, 1)]
    return info


def log_pruned_tracks(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    pruned_ids: Set[int],
    rule_name: str,
    frame_width: int = 0,
    frame_height: int = 0,
    prune_log: Optional[list] = None,
    total_frames: int = 0,
) -> None:
    """Print per-track diagnostics for every pruned track so root-cause is visible in logs."""
    if not pruned_ids:
        return
    for tid in sorted(pruned_ids):
        entries = raw_tracks.get(tid, [])
        n_frames = len(entries)
        if prune_log is not None:
            prune_log.append(
                get_pruned_track_info(
                    raw_tracks,
                    tid,
                    rule_name,
                    frame_width,
                    frame_height,
                    total_frames=total_frames,
                    decision="pruned",
                )
            )
        if not entries:
            print(f"    [PRUNED by {rule_name}] track {tid}: no entries")
            continue
        centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for _, b, _ in entries]
        med_cx = float(np.median([c[0] for c in centers]))
        med_cy = float(np.median([c[1] for c in centers]))
        heights = [b[3] - b[1] for _, b, _ in entries if b[3] - b[1] > 0]
        med_h = float(np.median(heights)) if heights else 0.0
        first_f = min(e[0] for e in entries)
        last_f = max(e[0] for e in entries)
        pos_label = ""
        if frame_width > 0 and frame_height > 0:
            x_pct = med_cx / frame_width * 100
            y_pct = med_cy / frame_height * 100
            pos_label = f" pos=({x_pct:.0f}%x, {y_pct:.0f}%y)"
        print(f"    [PRUNED by {rule_name}] track {tid}: {n_frames} frames "
              f"(f{first_f}-{last_f}), median_center=({med_cx:.0f},{med_cy:.0f}){pos_label}, "
              f"median_h={med_h:.0f}")


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
        for entry in raw_tracks[track_id]:
            obs = coerce_observation(entry)
            frame_boxes[obs.frame_idx].append(
                (obs.bbox, track_id, obs.conf, obs.is_sam_refined, obs.segmentation_mask)
            )

    results = []
    for frame_idx in range(total_frames):
        entries = frame_boxes[frame_idx]
        entries.sort(key=lambda x: x[1])
        boxes = [e[0] for e in entries]
        track_ids = [e[1] for e in entries]
        confs = [e[2] for e in entries]
        is_sam_refined = [e[3] for e in entries]
        segmentation_masks = [e[4] for e in entries]
        results.append({
            "frame_idx": frame_idx,
            "boxes": boxes,
            "track_ids": track_ids,
            "confs": confs,
            "is_sam_refined": is_sam_refined,
            "segmentation_masks": segmentation_masks,
        })
    return results
