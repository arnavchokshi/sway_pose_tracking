"""
Phase 2: Crossover Management — OKS-Based Dense Overlap Handling (V3.4)

Handles dancer overlaps using Object Keypoint Similarity (OKS):
- Crossover Entry: When IoU > 0.6 between two track boxes, switch to OKS matching
- OKS Occlusion Fallback: When total keypoint confidence < 0.3, suspend matching
  and project box forward using Constant Velocity Model (CVM)
- Crossover Exit: When IoU < 0.4 for 3 consecutive frames, return to standard tracking

V3.4 additions:
- Visibility Scoring: Per-track visibility [0,1] using containment + foot-position depth
- Keypoint Collision Dedup: Detect duplicate pose overlays via median keypoint distance
- Hybrid CVM: CVM for first 5 frames of occlusion, then freeze + confidence decay

Phase 5.5 (Collision Handling):
- Collision State: IoU > 0.6 for > 3 consecutive frames → in_collision=True, suspend ID swaps
- Collision Exit: IoU < 0.3 → run Temporal OKS Audit (lookback 5 + lookahead 5 frames)
- If swapped path (Pre-A→Post-B, Pre-B→Post-A) has higher cumulative OKS, force ID swap

V3.6 (Acceleration Audit):
- Detect implausible box jumps (center displacement > 0.5*bbox_height in 1 frame)
- When tracker wrongly assigns ID to another performer (e.g. missed pose for 1 frame),
  the box "jumps" to wrong person. Use OKS to find which pose matches the track's history
  and correct the ID assignment.

Runs AFTER pose estimation (requires keypoints). All math vectorized with NumPy.
"""

from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np

# V3.8: Optional appearance cost for crossover (red vs blue during occlusion)
try:
    from .reid_embedder import cosine_similarity
    _REID_AVAILABLE = True
except ImportError:
    _REID_AVAILABLE = False

# COCO 17 keypoint sigmas (per-keypoint std dev, scale-normalized)
# Order: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
COCO_OKS_SIGMAS = np.array(
    [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
     0.107, 0.107, 0.087, 0.087, 0.089, 0.089],
    dtype=np.float64
)

# Crossover thresholds
IOU_CROSSOVER_ENTRY = 0.6
IOU_CROSSOVER_EXIT = 0.4
CROSSOVER_EXIT_FRAMES = 3
OKS_OCCLUSION_THRESHOLD = 0.3  # Total keypoint confidence below this -> CVM ghost

# V3.0: Occlusion re-ID — align with track_buffer=90
REID_MAX_FRAME_GAP = 90  # Max frames between dead track end and newborn start (3s @ 30 FPS)
REID_MIN_OKS = 0.35

# V3.4: Visibility scoring — 0.85 allows pose estimation for partially visible
# dancers in dense formations (ViTPose handles partial visibility well)
VISIBILITY_CONTAINMENT_THRESH = 0.85
VISIBILITY_MIN_SCORE = 0.3

# V3.4: Hybrid CVM constants
CVM_MAX_FRAMES = 5
CVM_VELOCITY_DECAY = 0.9
CONF_DECAY_FACTOR = 0.85
BLEND_FRAMES = 3
# V3.5: CVM displacement cap — fraction of bbox height per frame
CVM_MAX_DISPLACEMENT_FRAC = 0.3
# V3.5: CVM overlap freeze — don't project into another live track
# 0.85: allows CVM box to pass through overlaps during crossovers while
# still preventing full collision with unrelated tracks
CVM_OVERLAP_FREEZE_IOU = 0.85

# V3.4: Keypoint collision dedup thresholds
# V3.5: Relaxed from 0.2/0.3 to catch nearby phantoms (e.g. 67px apart at bbox_h=165px)
COLLISION_KPT_DIST_FRAC = 0.35
COLLISION_CENTER_DIST_FRAC = 0.5

# Phase 5.5: Collision State (Locked Identity) — suspend ID decisions during overlap
COLLISION_ENTRY_IOU = 0.6       # IoU > this for N consecutive frames → enter collision
COLLISION_ENTRY_CONSECUTIVE = 3 # Frames at IoU > entry to trigger collision state
COLLISION_EXIT_IOU = 0.3        # IoU < this → collision ended (temporal OKS audit runs)
COLLISION_EXIT_CONSECUTIVE = 1  # Frames at IoU < exit to confirm collision end
TEMPORAL_OKS_LOOKBACK = 5       # Last N "clean" pre-collision frames per ID
TEMPORAL_OKS_LOOKAHEAD = 5     # First N "clean" post-collision frames per ID
TEMPORAL_OKS_SWAP_MARGIN = 0.05  # Swapped path must exceed standard by this to force swap

# V3.6: Acceleration audit — detect ID switches from implausible jumps
ACCEL_JUMP_THRESH_FRAC = 0.50   # Displacement > this * bbox_height = implausible
ACCEL_OKS_SWAP_MARGIN = 0.08    # Correct swap when OKS(other) exceeds OKS(self) by this
ACCEL_LOOKBACK_FRAMES = 2       # Use pose from t-2 if t-1 is missing

# V3.8: Appearance weight in crossover — when red≠blue, appearance overrides OKS
APPEARANCE_WEIGHT = 0.4         # Add sim*weight to score; red↔blue = low sim → reject swap


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    """Compute box areas from (N, 4) xyxy format. Vectorized."""
    w = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    h = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return w * h


def _compute_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU matrix. boxes: (N, 4) xyxy.
    Returns (N, N) symmetric matrix. Vectorized.
    """
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0))
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = _box_areas(boxes)
    # Broadcast for all pairs
    xx1 = np.maximum(x1[:, np.newaxis], x1[np.newaxis, :])
    yy1 = np.maximum(y1[:, np.newaxis], y1[np.newaxis, :])
    xx2 = np.minimum(x2[:, np.newaxis], x2[np.newaxis, :])
    yy2 = np.minimum(y2[:, np.newaxis], y2[np.newaxis, :])
    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    union = areas[:, np.newaxis] + areas[np.newaxis, :] - inter
    union = np.maximum(union, 1e-6)
    iou = inter / union
    np.fill_diagonal(iou, 0.0)  # No self-IoU
    return iou


def _compute_bbox_iou(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """Compute IoU of two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _compute_oks(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    area: float,
    sigmas: np.ndarray = COCO_OKS_SIGMAS,
) -> float:
    """
    Compute OKS between two keypoint sets. kpts: (17, 3) with (x, y, vis/score).
    area: object area (bbox w*h) for scale. Uses visible keypoints only.
    """
    if kpts_a.shape[0] < 17 or kpts_b.shape[0] < 17:
        return 0.0
    s = np.sqrt(max(area, 1.0))
    kpa = kpts_a[:17, :2].astype(np.float64)
    kpb = kpts_b[:17, :2].astype(np.float64)
    # Visibility: use score if > 0, else 0
    va = kpts_a[:17, 2] if kpts_a.shape[1] > 2 else np.ones(17)
    vb = kpts_b[:17, 2] if kpts_b.shape[1] > 2 else np.ones(17)
    visible = (va > 0.01) & (vb > 0.01)
    if np.sum(visible) == 0:
        return 0.0
    d2 = np.sum((kpa - kpb) ** 2, axis=1)
    # KS_i = exp(-d^2 / (2 * s^2 * sigma^2))
    sigma2 = sigmas ** 2
    ks = np.exp(-d2 / (2.0 * s * s * sigma2 + 1e-9))
    ks[~visible] = 0.0
    return float(np.sum(ks) / np.sum(visible))


def _compute_total_keypoint_confidence(kpts: np.ndarray) -> float:
    """Mean keypoint confidence. kpts: (17, 3) with last col = score."""
    if kpts.shape[0] < 17 or kpts.shape[1] < 3:
        return 0.0
    return float(np.mean(kpts[:17, 2]))


def _bbox_from_xyxy(box: Tuple[float, float, float, float]) -> np.ndarray:
    return np.array(box, dtype=np.float64)


def _compute_velocity(
    box_prev: Tuple[float, float, float, float],
    box_curr: Tuple[float, float, float, float],
) -> np.ndarray:
    """Center velocity (vx, vy) from prev to curr frame."""
    cx_prev = (box_prev[0] + box_prev[2]) / 2
    cy_prev = (box_prev[1] + box_prev[3]) / 2
    cx_curr = (box_curr[0] + box_curr[2]) / 2
    cy_curr = (box_curr[1] + box_curr[3]) / 2
    return np.array([cx_curr - cx_prev, cy_curr - cy_prev], dtype=np.float64)


def _compute_containment(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Fraction of box_a's area that is contained within box_b. [0, 1].
    1.0 means box_a is fully inside box_b.
    """
    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area_a = max((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]), 1e-6)
    return float(inter_area / area_a)


def compute_visibility_scores(
    boxes: List[Tuple[float, float, float, float]],
    track_ids: List[int],
) -> Dict[int, float]:
    """
    V3.4: Per-track visibility score [0, 1]. 0 = fully occluded behind another dancer.
    Uses foot Y-position (ymax) for depth ordering: lower feet = closer to camera = in front.
    A track is considered occluded when its box is largely contained within a box that
    has lower feet (is in front).
    """
    n = len(boxes)
    if n == 0:
        return {}
    scores = np.ones(n, dtype=np.float64)
    boxes_arr = np.array(boxes, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            containment = _compute_containment(boxes_arr[i], boxes_arr[j])
            ymax_i = boxes_arr[i][3]
            ymax_j = boxes_arr[j][3]
            i_is_behind = ymax_i < ymax_j
            if containment > VISIBILITY_CONTAINMENT_THRESH and i_is_behind:
                scores[i] = min(scores[i], 1.0 - containment)
    return {tid: float(scores[idx]) for idx, tid in enumerate(track_ids)}


# Keypoint-bbox consistency: suppress pose when keypoints are far outside bbox (tracker ID switch)
KPT_BBOX_MAX_OFFSET_FRAC = 0.6  # Max distance from bbox center as fraction of bbox diagonal


def sanitize_pose_bbox_consistency(
    frame_data: Dict[str, Any],
    max_offset_frac: float = KPT_BBOX_MAX_OFFSET_FRAC,
    phase6_log: Optional[list] = None,
) -> int:
    """
    Remove poses where head keypoints are largely outside their bbox (e.g. tracker ID switch),
    and zero out individual limb keypoints that fall far outside the bbox.
    Returns number of poses removed.
    """
    poses = frame_data.get("poses", {})
    track_ids = frame_data.get("track_ids", [])
    boxes = frame_data.get("boxes", [])
    to_remove = set()
    for idx, tid in enumerate(track_ids):
        if tid not in poses or idx >= len(boxes):
            continue
        kpts = poses[tid]["keypoints"]
        if kpts.shape[0] < 17 or kpts.shape[1] < 2:
            continue
        box = boxes[idx]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        diag = np.sqrt((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2)
        if diag < 1:
            continue
        
        vis = kpts[:17, 2] if kpts.shape[1] > 2 else np.ones(17)
        invalid_head = False
        
        for i in range(17):
            if vis[i] < 0.1:
                continue
            dx = kpts[i, 0] - cx
            dy = kpts[i, 1] - cy
            offset = np.sqrt(dx * dx + dy * dy)
            if offset > max_offset_frac * diag:
                if i < 5:
                    to_remove.add(tid)
                    invalid_head = True
                    break
                elif kpts.shape[1] > 2:
                    kpts[i, 2] = 0.0
                    
        if invalid_head:
            continue
            
    fj = int(frame_data.get("frame_idx", -1))
    for tid in to_remove:
        if phase6_log is not None:
            phase6_log.append(
                {
                    "rule": "sanitize_pose_bbox_consistency",
                    "track_id": int(tid),
                    "frame_idx": fj,
                    "decision": "removed_pose",
                }
            )
        if tid in poses:
            del poses[tid]
    if to_remove:
        new_boxes = []
        new_track_ids = []
        for box, t in zip(boxes, track_ids):
            if t not in to_remove:
                new_boxes.append(box)
                new_track_ids.append(t)
        frame_data["boxes"] = new_boxes
        frame_data["track_ids"] = new_track_ids
    return len(to_remove)


def deduplicate_collocated_poses(
    frame_data: Dict[str, Any],
    kpt_dist_frac: float = COLLISION_KPT_DIST_FRAC,
    center_dist_frac: float = COLLISION_CENTER_DIST_FRAC,
    protected_tids: Optional[Set[int]] = None,
    track_frame_count: Optional[Dict[int, int]] = None,
    phase6_log: Optional[list] = None,
) -> None:
    """
    V3.4: Remove duplicate pose overlays on the same physical person.
    Uses median keypoint distance (robust to hallucinated outlier keypoints) and
    requires bbox centers to also be close (protects legitimate partner work).
    Suppresses the lower-confidence track. V3.7: protected_tids are never suppressed
    (e.g. late entrants — real people, not duplicates). Modifies frame_data in-place.
    """
    protected_tids = protected_tids or set()
    poses = frame_data.get("poses", {})
    track_ids = frame_data.get("track_ids", [])
    boxes = frame_data.get("boxes", [])
    if len(track_ids) < 2:
        return

    to_suppress = set()
    for i, tid_a in enumerate(track_ids):
        if tid_a in to_suppress or tid_a not in poses:
            continue
        for j in range(i + 1, len(track_ids)):
            tid_b = track_ids[j]
            if tid_b in to_suppress or tid_b not in poses:
                continue
            kpts_a = poses[tid_a]["keypoints"]
            kpts_b = poses[tid_b]["keypoints"]
            if kpts_a.shape[0] < 17 or kpts_b.shape[0] < 17:
                continue

            per_kpt_dist = np.linalg.norm(kpts_a[:17, :2] - kpts_b[:17, :2], axis=1)
            median_dist = float(np.median(per_kpt_dist))
            bbox_h = max(boxes[i][3] - boxes[i][1], boxes[j][3] - boxes[j][1], 1.0)
            box_a = tuple(boxes[i])
            box_b = tuple(boxes[j])
            iou = _compute_bbox_iou(box_a, box_b)

            # V3.9: Depth overlap — when one box largely contains the other, these are two people
            # at different depths (front/back), not duplicates. Skip dedup to avoid suppressing
            # the front person (e.g. green dancer in front, fully visible).
            cont_ab = _compute_containment(np.array(box_a), np.array(box_b))
            cont_ba = _compute_containment(np.array(box_b), np.array(box_a))
            if cont_ab > 0.7 or cont_ba > 0.7:
                continue

            if median_dist > kpt_dist_frac * bbox_h:
                continue

            center_a = np.array([(boxes[i][0]+boxes[i][2])/2, (boxes[i][1]+boxes[i][3])/2])
            center_b = np.array([(boxes[j][0]+boxes[j][2])/2, (boxes[j][1]+boxes[j][3])/2])
            center_dist = float(np.linalg.norm(center_a - center_b))
            # When keypoints are very close, allow dedup even if bbox centers differ (tracker ID switch)
            kpt_tight = median_dist < 0.2 * bbox_h
            if not kpt_tight and center_dist > center_dist_frac * bbox_h:
                continue

            # V3.7: When poses are clearly the SAME person, allow dedup even for protected tracks.
            # Strict: IoU>0.5 AND keypoint proximity = true duplicate (e.g. 61 duplicate of 9).
            # Loose: keypoints nearly identical alone — can be false positive (late entrant near
            # main dancer, similar pose). When protected vs long track, only dedup on strict.
            clearly_same_strict = iou > 0.5 and median_dist < 0.25 * bbox_h
            clearly_same_loose = median_dist < 0.12 * bbox_h
            clearly_same_person = clearly_same_strict or clearly_same_loose
            if (tid_a in protected_tids or tid_b in protected_tids) and not clearly_same_person:
                continue
            # Extra: protected (late entrant) vs long (main dancer) — only dedup if strict,
            # else we may wrongly suppress the late entrant when they stand near a dancer.
            if track_frame_count and clearly_same_loose and not clearly_same_strict:
                count_a = track_frame_count.get(tid_a, 0)
                count_b = track_frame_count.get(tid_b, 0)
                if (tid_a in protected_tids and count_b >= count_a * 3) or (
                    tid_b in protected_tids and count_a >= count_b * 3
                ):
                    continue
            conf_a = float(np.mean(kpts_a[:17, 2]))
            conf_b = float(np.mean(kpts_b[:17, 2]))
            # When both protected and same person: prefer keeping the longer track (late entrant to end)
            if (
                clearly_same_person
                and (tid_a in protected_tids or tid_b in protected_tids)
                and track_frame_count
            ):
                count_a = track_frame_count.get(tid_a, 0)
                count_b = track_frame_count.get(tid_b, 0)
                if count_a != count_b:
                    suppressed = tid_b if count_a >= count_b else tid_a
                else:
                    suppressed = tid_b if conf_a >= conf_b else tid_a
                to_suppress.add(suppressed)
            else:
                if track_frame_count:
                    count_a = track_frame_count.get(tid_a, 0)
                    count_b = track_frame_count.get(tid_b, 0)
                    if count_a != count_b:
                        suppressed = tid_b if count_a >= count_b else tid_a
                    else:
                        # Tie (including both 0): fall back to confidence
                        suppressed = tid_b if conf_a >= conf_b else tid_a
                else:
                    suppressed = tid_b if conf_a >= conf_b else tid_a
                to_suppress.add(suppressed)

    fj = int(frame_data.get("frame_idx", -1))
    for tid in to_suppress:
        if phase6_log is not None:
            phase6_log.append(
                {
                    "rule": "deduplicate_collocated_poses",
                    "track_id": int(tid),
                    "frame_idx": fj,
                    "decision": "suppressed_duplicate",
                }
            )
        if tid in poses:
            del poses[tid]

    # Also strip suppressed tracks from boxes and track_ids so phantom boxes
    # don't persist downstream (CVM, rendering, scoring).
    if to_suppress:
        new_boxes = []
        new_track_ids = []
        for box, tid in zip(boxes, track_ids):
            if tid not in to_suppress:
                new_boxes.append(box)
                new_track_ids.append(tid)
        frame_data["boxes"] = new_boxes
        frame_data["track_ids"] = new_track_ids


# Late entrant window: tracks starting in this frame range are likely new people, not fragments
REID_LATE_ENTRANT_START_FRAME = 180   # ~6s @ 30fps
REID_LATE_ENTRANT_END_FRAME = 450     # ~15s
# When OKS exceeds this, allow merge even for late entrants (catches tracker fragments)
REID_LATE_ENTRANT_OVERRIDE_OKS = 0.5


def apply_occlusion_reid(
    all_frame_data: List[Dict[str, Any]],
    max_frame_gap: int = REID_MAX_FRAME_GAP,
    min_oks: float = REID_MIN_OKS,
    debug: bool = False,
    total_frames: int = 0,
) -> List[Dict[str, Any]]:
    """
    Re-link tracks that fragmented from occlusion using pose similarity (OKS).
    When track A disappears and track B appears within max_frame_gap frames,
    if OKS(A_last_pose, B_first_pose) >= min_oks, merge B into A (keep A's ID).

    Runs before crossover refinement. Modifies all_frame_data in-place.
    """
    if not all_frame_data or len(all_frame_data) < 2:
        return all_frame_data

    # Build track_id -> (first_frame, last_frame, first_pose, last_pose, first_box, last_box)
    # Use frame_idx from fd when available (sparse frames), else enumerate index
    track_info: Dict[int, Dict] = {}
    for f_idx, fd in enumerate(all_frame_data):
        frame_idx = fd.get("frame_idx", f_idx)
        track_ids = fd.get("track_ids", [])
        poses = fd.get("poses", {})
        boxes = fd.get("boxes", [])
        for idx, tid in enumerate(track_ids):
            if tid not in poses or idx >= len(boxes):
                continue
            kpts = poses[tid].get("keypoints")
            if kpts is None or kpts.shape[0] < 17:
                continue
            # Ensure (17,3) with x,y,score for OKS (some poses have scores in separate key)
            if kpts.shape[1] < 3:
                sc = np.asarray(poses[tid].get("scores", np.ones(17))).flatten()
                kpts = np.column_stack([kpts[:, :2], sc[:17]])
            if tid not in track_info:
                track_info[tid] = {
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "first_pose": kpts.copy(),
                    "last_pose": kpts.copy(),
                    "first_box": tuple(boxes[idx]),
                    "last_box": tuple(boxes[idx]),
                }
            else:
                track_info[tid]["last_frame"] = frame_idx
                track_info[tid]["last_pose"] = kpts.copy()
                track_info[tid]["last_box"] = tuple(boxes[idx])

    total_frames = len(all_frame_data)
    last_frame_idx = max(fd.get("frame_idx", i) for i, fd in enumerate(all_frame_data))
    dead_ids = [tid for tid, info in track_info.items() if info["last_frame"] < last_frame_idx]
    newborn_ids = [tid for tid, info in track_info.items() if info["first_frame"] > 0]

    if debug:
        import sys
        print("  [Re-ID debug] dead:", sorted(dead_ids), "newborn:", sorted(newborn_ids), file=sys.stderr)
        for tid in track_info:
            info = track_info[tid]
            print(f"    track {tid}: frames {info['first_frame']}-{info['last_frame']}", file=sys.stderr)

    merge_map: Dict[int, int] = {}  # new_id -> keep_id (replace new_id with keep_id)
    for tid_a in dead_ids:
        if tid_a not in track_info or tid_a in merge_map.values():
            continue
        info_a = track_info[tid_a]
        frame_a_last = info_a["last_frame"]
        last_pose_a = info_a["last_pose"]
        last_box_a = info_a["last_box"]
        area_a = (last_box_a[2] - last_box_a[0]) * (last_box_a[3] - last_box_a[1])
        if area_a < 1:
            area_a = 1.0

        for tid_b in newborn_ids:
            if tid_b == tid_a or tid_b in merge_map or tid_b in merge_map.values():
                continue
            info_b = track_info[tid_b]
            frame_b_first = info_b["first_frame"]
            gap = frame_b_first - frame_a_last
            if gap <= 0 or gap > max_frame_gap:
                continue
            first_pose_b = info_b["first_pose"]
            last_box_b = info_b["first_box"]
            area_b = (last_box_b[2] - last_box_b[0]) * (last_box_b[3] - last_box_b[1])
            # Use max area: dead track's last box may be tiny (occluded); newborn's is likely full
            area = max(area_a, area_b, 1.0)
            oks = _compute_oks(last_pose_a, first_pose_b, area)
            if debug and gap <= 60:  # Log close pairs
                import sys
                print(f"    [Re-ID] dead {tid_a} (f{frame_a_last}) vs newborn {tid_b} (f{frame_b_first}) gap={gap} oks={oks:.3f} need>={min_oks}", file=sys.stderr)
            if oks >= min_oks:
                merge_map[tid_b] = tid_a
                break  # One merge per dead track

    # --- Pass 1.5: Overlapping fragments (same person, two IDs during handoff) ---
    # e.g. track 8 (2-299) and 115 (148-428): B starts during A's run. Compare A at boundary with B at start.
    for tid_a in list(track_info.keys()):
        if tid_a in merge_map or tid_a in merge_map.values():
            continue
        for tid_b in list(track_info.keys()):
            if tid_b <= tid_a or tid_b in merge_map or tid_b in merge_map.values():
                continue
            info_a, info_b = track_info[tid_a], track_info[tid_b]
            start_a, end_a = info_a["first_frame"], info_a["last_frame"]
            start_b, end_b = info_b["first_frame"], info_b["last_frame"]
            # B starts while A is still active (overlap), and A started first
            if not (start_a < start_b <= end_a):
                continue
            # Compare A's pose/box at frame (boundary-1) with B's at boundary — temporal adjacency
            boundary = start_b
            pose_a = None
            box_a_at_boundary = None
            pose_b = info_b["first_pose"]
            area = (info_b["first_box"][2] - info_b["first_box"][0]) * (info_b["first_box"][3] - info_b["first_box"][1])
            for f_idx, fd in enumerate(all_frame_data):
                frame_idx = fd.get("frame_idx", f_idx)
                if frame_idx == boundary - 1:
                    poses = fd.get("poses", {})
                    boxes, tids = fd.get("boxes", []), fd.get("track_ids", [])
                    if tid_a in poses:
                        kpa = poses[tid_a].get("keypoints")
                        if kpa is not None and kpa.shape[0] >= 17:
                            for idx, t in enumerate(tids):
                                if t == tid_a and idx < len(boxes):
                                    area = max(area, (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1]))
                                    pose_a = kpa
                                    box_a_at_boundary = tuple(boxes[idx])
                                    break
                    break
            if pose_a is None:
                pose_a = info_a["last_pose"]
                box_a_at_boundary = info_a["last_box"]
                area = max(area, (info_a["last_box"][2] - info_a["last_box"][0]) * (info_a["last_box"][3] - info_a["last_box"][1]))
            if pose_b is None or pose_b.shape[0] < 17:
                continue
            oks = _compute_oks(pose_a, pose_b, max(area, 1.0))
            box_a = box_a_at_boundary or info_a["last_box"]
            box_b = info_b["first_box"]
            iou = _compute_bbox_iou(box_a, box_b)
            cx_a = (box_a[0] + box_a[2]) / 2
            cy_a = (box_a[1] + box_a[3]) / 2
            cx_b = (box_b[0] + box_b[2]) / 2
            cy_b = (box_b[1] + box_b[3]) / 2
            dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
            diag = np.sqrt((box_b[2] - box_b[0]) ** 2 + (box_b[3] - box_b[1]) ** 2)
            bbox_near = dist < diag * 0.5 if diag > 1 else dist < 50
            # Merge if OKS passes, OR IoU/centroid fallback when OKS fails
            OKS_VETO_PASS15 = 0.25
            merge_oks = oks >= min_oks
            merge_proximity = bbox_near and oks >= min_oks * 0.5
            merge_iou_centroid = (
                iou >= 0.15 and dist < diag * 1.0 and oks >= OKS_VETO_PASS15
            )
            # Skip late entrants unless OKS is high (tracker fragment, not new person)
            if REID_LATE_ENTRANT_START_FRAME <= start_b <= REID_LATE_ENTRANT_END_FRAME:
                if oks < REID_LATE_ENTRANT_OVERRIDE_OKS:
                    continue
                # V3.7: Short track in late-entrant window = likely new person entering, not fragment.
                # Never merge when B starts after 25% of video and has <100 frames.
                if total_frames > 0:
                    len_b = end_b - start_b + 1
                    if start_b >= total_frames * 0.25 and len_b < 100:
                        continue
            if debug:
                import sys
                va = pose_a[:17, 2] if pose_a.shape[1] > 2 else np.zeros(17)
                vb = pose_b[:17, 2] if pose_b.shape[1] > 2 else np.zeros(17)
                n_vis = np.sum((va > 0.01) & (vb > 0.01))
                print(f"    [Re-ID Pass1.5] {tid_a}(f{boundary-1}) vs {tid_b}(f{boundary}) oks={oks:.3f} iou={iou:.3f} dist={dist:.0f} n_vis={n_vis}", file=sys.stderr)
                if n_vis < 5:
                    print(f"      pose_a conf: min={float(np.min(va)):.3f} max={float(np.max(va)):.3f} mean={float(np.mean(va)):.3f}", file=sys.stderr)
                    print(f"      pose_b conf: min={float(np.min(vb)):.3f} max={float(np.max(vb)):.3f} mean={float(np.mean(vb)):.3f}", file=sys.stderr)
            if merge_oks or merge_proximity or merge_iou_centroid:
                merge_map[tid_b] = tid_a
                if debug:
                    import sys
                    print(f"    [Re-ID Pass1.5] merged {tid_b}->{tid_a} oks={oks:.3f}", file=sys.stderr)
                break

    # --- Pass 1.6: Last-resort fragment merge — when OKS fails, use IoU over overlap zone ---
    if not merge_map:
        best_pair = None
        best_iou = 0.0
        REID_PASS16_MIN_IOU = 0.05  # Lowered from 0.08 to catch handoff fragments (same person, 2 IDs)
        for tid_a in list(track_info.keys()):
            for tid_b in list(track_info.keys()):
                if tid_b <= tid_a:
                    continue
                info_a, info_b = track_info[tid_a], track_info[tid_b]
                start_a, end_a = info_a["first_frame"], info_a["last_frame"]
                start_b, end_b = info_b["first_frame"], info_b["last_frame"]
                if not (start_a < start_b <= end_a):
                    continue
                overlap_start, overlap_end = start_b, min(end_a, end_b)
                if overlap_end - overlap_start < 10:
                    continue
                # Don't merge when B is a likely late entrant (Pass 1.6 uses IoU, no OKS override)
                if REID_LATE_ENTRANT_START_FRAME <= start_b <= REID_LATE_ENTRANT_END_FRAME:
                    continue
                iou_sum, iou_n = 0.0, 0
                oks_sum, oks_n = 0.0, 0
                OKS_VETO_PASS16 = 0.15
                for f_idx, fd in enumerate(all_frame_data):
                    frame_idx = fd.get("frame_idx", f_idx)
                    if frame_idx < overlap_start or frame_idx > overlap_end:
                        continue
                    boxes, tids = fd.get("boxes", []), fd.get("track_ids", [])
                    if tid_a not in tids or tid_b not in tids:
                        continue
                    idx_a, idx_b = tids.index(tid_a), tids.index(tid_b)
                    if idx_a >= len(boxes) or idx_b >= len(boxes):
                        continue
                    iou_sum += _compute_bbox_iou(tuple(boxes[idx_a]), tuple(boxes[idx_b]))
                    iou_n += 1
                    poses = fd.get("poses", {})
                    pa = poses.get(tid_a, {}).get("keypoints")
                    pb = poses.get(tid_b, {}).get("keypoints")
                    if pa is not None and pb is not None:
                        pa = np.asarray(pa)
                        pb = np.asarray(pb)
                        if pa.shape[0] >= 17 and pb.shape[0] >= 17:
                            ba, bb = tuple(boxes[idx_a]), tuple(boxes[idx_b])
                            area = max(
                                (ba[2] - ba[0]) * (ba[3] - ba[1]),
                                (bb[2] - bb[0]) * (bb[3] - bb[1]),
                                1.0,
                            )
                            oks_sum += float(_compute_oks(pa, pb, area))
                            oks_n += 1
                if iou_n >= 5:
                    mean_iou = iou_sum / iou_n
                    mean_oks = (oks_sum / oks_n) if oks_n > 0 else 0.0
                    oks_ok = (oks_n >= 3 and mean_oks >= OKS_VETO_PASS16) or (oks_n < 3)
                    if debug:
                        import sys
                        print(
                            f"    [Re-ID Pass1.6] {tid_a}-{tid_b} overlap={overlap_start}-{overlap_end} n={iou_n} mean_iou={mean_iou:.3f} mean_oks={mean_oks:.3f} oks_n={oks_n}",
                            file=sys.stderr,
                        )
                    if (
                        oks_ok
                        and mean_iou > best_iou
                        and mean_iou >= REID_PASS16_MIN_IOU
                    ):
                        best_iou = mean_iou
                        best_pair = (tid_a, tid_b)
        if best_pair:
            tid_a, tid_b = best_pair
            merge_map[tid_b] = tid_a
            if debug:
                import sys
                print(f"    [Re-ID Pass1.6] last-resort merged {tid_b}->{tid_a} mean_iou={best_iou:.3f}", file=sys.stderr)

    # --- Pass 1.7: Temporal continuation — B starts during A, continues past A's end ---
    # Tracker may assign different boxes when both IDs present (IoU=0), but B temporally continues A
    if not merge_map:
        best_pair = None
        best_continuation = 0  # frames B extends past A
        for tid_a in list(track_info.keys()):
            for tid_b in list(track_info.keys()):
                if tid_b <= tid_a:
                    continue
                info_a, info_b = track_info[tid_a], track_info[tid_b]
                start_a, end_a = info_a["first_frame"], info_a["last_frame"]
                start_b, end_b = info_b["first_frame"], info_b["last_frame"]
                # B starts during A and continues past A's end (B is the "new" ID that took over)
                if not (start_a < start_b <= end_a and end_b > end_a):
                    continue
                if REID_LATE_ENTRANT_START_FRAME <= start_b <= REID_LATE_ENTRANT_END_FRAME:
                    continue
                continuation = end_b - end_a
                if continuation > best_continuation:
                    best_continuation = continuation
                    best_pair = (tid_a, tid_b)
        if best_pair:
            tid_a, tid_b = best_pair
            merge_map[tid_b] = tid_a
            if debug:
                import sys
                print(f"    [Re-ID Pass1.7] temporal continuation merged {tid_b}->{tid_a} (extends {best_continuation}f past A)", file=sys.stderr)

    # --- Pass 2: Complementary track merge (non-overlapping alternating IDs) ---
    # Catches cases like track 4 (frames 0-150, 260-385) and track 17 (152-258)
    # where neither is cleanly "dead" before the other is "born".
    track_frame_sets: Dict[int, set] = {}
    for f_idx, fd in enumerate(all_frame_data):
        for tid in fd.get("track_ids", []):
            if tid not in track_frame_sets:
                track_frame_sets[tid] = set()
            track_frame_sets[tid].add(f_idx)

    all_tids = list(track_info.keys())
    for i in range(len(all_tids)):
        tid_a = all_tids[i]
        if tid_a not in track_info or tid_a in merge_map or tid_a in merge_map.values():
            continue
        for j in range(i + 1, len(all_tids)):
            tid_b = all_tids[j]
            if tid_b not in track_info or tid_b in merge_map or tid_b in merge_map.values():
                continue
            if tid_a == tid_b:
                continue

            frames_a = track_frame_sets.get(tid_a, set())
            frames_b = track_frame_sets.get(tid_b, set())
            if frames_a & frames_b:
                continue

            # Check OKS at transition boundaries
            info_a = track_info[tid_a]
            info_b = track_info[tid_b]

            # Try both orderings: A ends then B starts, and B ends then A starts
            oks_scores = []
            if info_a["last_frame"] < info_b["first_frame"]:
                gap = info_b["first_frame"] - info_a["last_frame"]
                if gap <= max_frame_gap:
                    area = (info_a["last_box"][2] - info_a["last_box"][0]) * (info_a["last_box"][3] - info_a["last_box"][1])
                    oks_scores.append(_compute_oks(info_a["last_pose"], info_b["first_pose"], max(area, 1.0)))
            if info_b["last_frame"] < info_a["first_frame"]:
                gap = info_a["first_frame"] - info_b["last_frame"]
                if gap <= max_frame_gap:
                    area = (info_b["last_box"][2] - info_b["last_box"][0]) * (info_b["last_box"][3] - info_b["last_box"][1])
                    oks_scores.append(_compute_oks(info_b["last_pose"], info_a["first_pose"], max(area, 1.0)))

            # For alternating tracks, check boundaries in temporal order
            if not oks_scores:
                all_frames_sorted = sorted(frames_a | frames_b)
                prev_owner = None
                for fidx in all_frames_sorted:
                    owner = tid_a if fidx in frames_a else tid_b
                    if prev_owner is not None and owner != prev_owner:
                        # Transition boundary — get poses from adjacent frames
                        prev_fd = all_frame_data[fidx - 1] if fidx > 0 else None
                        curr_fd = all_frame_data[fidx]
                        if prev_fd is not None:
                            prev_tid = prev_owner
                            curr_tid = owner
                            prev_poses = prev_fd.get("poses", {})
                            curr_poses = curr_fd.get("poses", {})
                            if prev_tid in prev_poses and curr_tid in curr_poses:
                                pk = prev_poses[prev_tid].get("keypoints")
                                ck = curr_poses[curr_tid].get("keypoints")
                                if pk is not None and ck is not None and pk.shape[0] >= 17 and ck.shape[0] >= 17:
                                    prev_boxes = prev_fd.get("boxes", [])
                                    prev_tids = prev_fd.get("track_ids", [])
                                    if prev_tid in prev_tids:
                                        pidx = prev_tids.index(prev_tid)
                                        if pidx < len(prev_boxes):
                                            area = (prev_boxes[pidx][2] - prev_boxes[pidx][0]) * (prev_boxes[pidx][3] - prev_boxes[pidx][1])
                                            oks_scores.append(_compute_oks(pk, ck, max(area, 1.0)))
                    prev_owner = owner

            if oks_scores and np.mean(oks_scores) >= min_oks:
                keep = tid_a if len(frames_a) >= len(frames_b) else tid_b
                kill = tid_b if keep == tid_a else tid_a
                merge_map[kill] = keep

    if not merge_map:
        return all_frame_data

    for fd in all_frame_data:
        track_ids = fd.get("track_ids", [])
        if not track_ids:
            continue
            
        boxes = fd.get("boxes", [])
        poses = fd.get("poses", {})
        
        unique_tids = {}  # mapped_tid -> (best_idx, score)
        
        for idx, tid in enumerate(track_ids):
            mapped_tid = merge_map.get(tid, tid)
            
            # Score this detection to keep the best one if A and B overlap
            score = 0.0
            if tid in poses:
                # Use keypoint confidence as primary score
                kpts = poses[tid]["keypoints"]
                score += sum(k[2] for k in kpts)
            # Add area as secondary score
            area = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
            score += area * 1e-6
            
            if mapped_tid not in unique_tids or score > unique_tids[mapped_tid][1]:
                unique_tids[mapped_tid] = (idx, score, tid)
                
        # Reconstruct frame data without duplicates
        new_track_ids = []
        new_boxes = []
        new_poses = {}
        for mapped_tid, (original_idx, _, original_tid) in unique_tids.items():
            new_track_ids.append(mapped_tid)
            new_boxes.append(boxes[original_idx])
            if original_tid in poses:
                new_poses[mapped_tid] = poses[original_tid]
                
        fd["track_ids"] = new_track_ids
        fd["boxes"] = new_boxes
        fd["poses"] = new_poses

    return all_frame_data


def _run_temporal_oks_audit(
    all_frame_data: List[Dict[str, Any]],
    tid_a: int,
    tid_b: int,
    collision_start_fidx: int,
    collision_end_fidx: int,
    occlusion_thresh: float,
    lookback: int = TEMPORAL_OKS_LOOKBACK,
    lookahead: int = TEMPORAL_OKS_LOOKAHEAD,
) -> Optional[bool]:
    """
    Phase 5.5: Temporal OKS Audit on collision exit.
    Returns True if IDs should be swapped (swapped path has higher OKS), False otherwise.
    Returns None if audit cannot be performed (insufficient clean frames).
    """
    # Collect last N clean pre-collision frames (before collision_start_fidx)
    pre_a_poses: List[np.ndarray] = []
    pre_a_areas: List[float] = []
    pre_b_poses: List[np.ndarray] = []
    pre_b_areas: List[float] = []

    for f_idx in range(collision_start_fidx - 1, -1, -1):
        if len(pre_a_poses) >= lookback and len(pre_b_poses) >= lookback:
            break
        if f_idx >= len(all_frame_data):
            continue
        fd = all_frame_data[f_idx]
        poses = fd.get("poses", {})
        boxes = fd.get("boxes", [])
        tids = fd.get("track_ids", [])
        for tid, need_list in [(tid_a, "pre_a"), (tid_b, "pre_b")]:
            if tid not in tids or tid not in poses:
                continue
            idx = tids.index(tid)
            kpts = poses[tid]["keypoints"]
            if kpts.shape[0] < 17 or _compute_total_keypoint_confidence(kpts) < occlusion_thresh:
                continue
            area = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
            if need_list == "pre_a" and len(pre_a_poses) < lookback:
                pre_a_poses.append(kpts.copy())
                pre_a_areas.append(area)
            elif need_list == "pre_b" and len(pre_b_poses) < lookback:
                pre_b_poses.append(kpts.copy())
                pre_b_areas.append(area)

    # Collect first N clean post-collision frames
    post_a_poses: List[np.ndarray] = []
    post_a_areas: List[float] = []
    post_b_poses: List[np.ndarray] = []
    post_b_areas: List[float] = []

    for f_idx in range(collision_end_fidx, len(all_frame_data)):
        if len(post_a_poses) >= lookahead and len(post_b_poses) >= lookahead:
            break
        fd = all_frame_data[f_idx]
        poses = fd.get("poses", {})
        boxes = fd.get("boxes", [])
        tids = fd.get("track_ids", [])
        if tid_a not in tids or tid_b not in tids:
            continue
        for tid, need_list in [(tid_a, "post_a"), (tid_b, "post_b")]:
            if tid not in poses:
                continue
            idx = tids.index(tid)
            kpts = poses[tid]["keypoints"]
            if kpts.shape[0] < 17 or _compute_total_keypoint_confidence(kpts) < occlusion_thresh:
                continue
            area = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
            if need_list == "post_a" and len(post_a_poses) < lookahead:
                post_a_poses.append(kpts.copy())
                post_a_areas.append(area)
            elif need_list == "post_b" and len(post_b_poses) < lookahead:
                post_b_poses.append(kpts.copy())
                post_b_areas.append(area)

    if not pre_a_poses or not pre_b_poses or not post_a_poses or not post_b_poses:
        return None

    def _avg_oks(pose_list_a: List[np.ndarray], pose_list_b: List[np.ndarray],
                 area_list_a: List[float], area_list_b: List[float]) -> float:
        total, count = 0.0, 0
        for pa, aa in zip(pose_list_a, area_list_a):
            for pb, ab in zip(pose_list_b, area_list_b):
                area = max(aa, ab, 1.0)
                total += _compute_oks(pa, pb, area)
                count += 1
        return total / count if count > 0 else 0.0

    standard_a = _avg_oks(pre_a_poses, post_a_poses, pre_a_areas, post_a_areas)
    standard_b = _avg_oks(pre_b_poses, post_b_poses, pre_b_areas, post_b_areas)
    standard_total = standard_a + standard_b

    swapped_ab = _avg_oks(pre_a_poses, post_b_poses, pre_a_areas, post_b_areas)
    swapped_ba = _avg_oks(pre_b_poses, post_a_poses, pre_b_areas, post_a_areas)
    swapped_total = swapped_ab + swapped_ba

    return swapped_total > standard_total + TEMPORAL_OKS_SWAP_MARGIN


def _apply_id_swap_from_frame(
    all_frame_data: List[Dict[str, Any]],
    tid_a: int,
    tid_b: int,
    from_fidx: int,
) -> None:
    """Swap track_ids tid_a <-> tid_b (and associated boxes/poses) in all frames from from_fidx to end."""
    for f_idx in range(from_fidx, len(all_frame_data)):
        fd = all_frame_data[f_idx]
        track_ids = fd.get("track_ids", [])
        poses = fd.get("poses", {})
        boxes = fd.get("boxes", [])
        if tid_a not in track_ids or tid_b not in track_ids:
            continue
        idx_a, idx_b = track_ids.index(tid_a), track_ids.index(tid_b)
        new_track_ids = list(track_ids)
        new_boxes = list(boxes)
        new_poses = dict(poses)
        new_track_ids[idx_a], new_track_ids[idx_b] = tid_b, tid_a
        new_boxes[idx_a], new_boxes[idx_b] = boxes[idx_b], boxes[idx_a]
        new_poses[tid_a], new_poses[tid_b] = poses[tid_b], poses[tid_a]
        fd["track_ids"] = new_track_ids
        fd["boxes"] = new_boxes
        fd["poses"] = new_poses


def apply_acceleration_audit(
    all_frame_data: List[Dict[str, Any]],
    jump_thresh_frac: float = ACCEL_JUMP_THRESH_FRAC,
    oks_margin: float = ACCEL_OKS_SWAP_MARGIN,
    occlusion_thresh: float = OKS_OCCLUSION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    V3.6: Detect and correct ID switches from implausible box jumps.
    When a performer is missed for one frame, the tracker may assign this ID's box
    to another performer. The box "jumps" an impossible distance in 1 frame.
    Uses OKS to find which pose in the frame belongs to this track's history.
    Modifies all_frame_data in-place.
    """
    if not all_frame_data or len(all_frame_data) < 2:
        return all_frame_data

    corrections = 0
    for f_idx in range(1, len(all_frame_data)):
        fd = all_frame_data[f_idx]
        prev_fd = all_frame_data[f_idx - 1]
        boxes = fd.get("boxes", [])
        track_ids = fd.get("track_ids", [])
        poses = fd.get("poses", {})
        prev_boxes = prev_fd.get("boxes", [])
        prev_tids = prev_fd.get("track_ids", [])
        prev_poses = prev_fd.get("poses", {})

        if len(track_ids) < 2:
            continue

        for idx_a, tid_a in enumerate(track_ids):
            if tid_a not in poses or idx_a >= len(boxes):
                continue
            box_a = boxes[idx_a]
            h_a = max(box_a[3] - box_a[1], 1.0)
            cx_a = (box_a[0] + box_a[2]) / 2
            cy_a = (box_a[1] + box_a[3]) / 2

            tid_a_prev_idx = prev_tids.index(tid_a) if tid_a in prev_tids else -1
            if tid_a_prev_idx < 0 or tid_a_prev_idx >= len(prev_boxes):
                continue
            prev_box_a = prev_boxes[tid_a_prev_idx]
            prev_cx = (prev_box_a[0] + prev_box_a[2]) / 2
            prev_cy = (prev_box_a[1] + prev_box_a[3]) / 2
            displacement = np.sqrt((cx_a - prev_cx) ** 2 + (cy_a - prev_cy) ** 2)

            if displacement <= jump_thresh_frac * h_a:
                continue

            pose_prev_a = prev_poses.get(tid_a)
            lookback = 1
            if pose_prev_a is None or _compute_total_keypoint_confidence(
                pose_prev_a.get("keypoints", np.zeros((17, 3)))
            ) < occlusion_thresh:
                for lb in range(2, min(f_idx, ACCEL_LOOKBACK_FRAMES + 2)):
                    pf = all_frame_data[f_idx - lb]
                    if tid_a in pf.get("track_ids", []) and tid_a in pf.get("poses", {}):
                        pose_prev_a = pf["poses"][tid_a]
                        lookback = lb
                        break
            if pose_prev_a is None:
                continue

            kpt_prev = pose_prev_a.get("keypoints")
            if kpt_prev is None or kpt_prev.shape[0] < 17:
                continue
            if kpt_prev.shape[1] < 3:
                sc = np.asarray(pose_prev_a.get("scores", np.ones(17))).flatten()
                kpt_prev = np.column_stack([kpt_prev[:, :2], sc[:17]])
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            oks_self = _compute_oks(poses[tid_a]["keypoints"], kpt_prev, max(area_a, 1.0))

            best_other = None
            best_oks = oks_self

            expected_cx = prev_cx
            expected_cy = prev_cy
            if lookback == 1 and tid_a in prev_tids:
                tid_a_prev2_idx = -1
                for lb in range(2, min(f_idx + 1, 4)):
                    pf = all_frame_data[f_idx - lb]
                    ptids = pf.get("track_ids", [])
                    if tid_a in ptids:
                        pboxes = pf.get("boxes", [])
                        pidx = ptids.index(tid_a)
                        if pidx < len(pboxes):
                            pb = pboxes[pidx]
                            vx = (prev_cx - (pb[0] + pb[2]) / 2) / lb
                            vy = (prev_cy - (pb[1] + pb[3]) / 2) / lb
                            expected_cx = prev_cx + vx
                            expected_cy = prev_cy + vy
                        break

            max_near_radius = h_a * 1.2

            for idx_b, tid_b in enumerate(track_ids):
                if idx_b == idx_a or tid_b not in poses:
                    continue
                box_b = boxes[idx_b]
                cx_b = (box_b[0] + box_b[2]) / 2
                cy_b = (box_b[1] + box_b[3]) / 2
                dist_to_expected = np.sqrt((cx_b - expected_cx) ** 2 + (cy_b - expected_cy) ** 2)
                if dist_to_expected > max_near_radius:
                    continue
                area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                area = max(area_a, area_b, 1.0)
                oks_other = _compute_oks(poses[tid_b]["keypoints"], kpt_prev, area)
                if oks_other > best_oks + oks_margin:
                    best_oks = oks_other
                    best_other = (idx_b, tid_b)

            if best_other is not None:
                idx_b, tid_b = best_other
                new_track_ids = list(track_ids)
                new_boxes = list(boxes)
                new_poses = dict(poses)
                new_track_ids[idx_a], new_track_ids[idx_b] = tid_b, tid_a
                new_boxes[idx_a], new_boxes[idx_b] = boxes[idx_b], boxes[idx_a]
                new_poses[tid_a] = poses[tid_b]
                new_poses[tid_b] = poses[tid_a]
                fd["track_ids"] = new_track_ids
                fd["boxes"] = new_boxes
                fd["poses"] = new_poses
                corrections += 1
                break

    return all_frame_data


def apply_crossover_refinement(
    all_frame_data: List[Dict[str, Any]],
    iou_entry: float = IOU_CROSSOVER_ENTRY,
    iou_exit: float = IOU_CROSSOVER_EXIT,
    exit_frames: int = CROSSOVER_EXIT_FRAMES,
    occlusion_thresh: float = OKS_OCCLUSION_THRESHOLD,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> List[Dict[str, Any]]:
    """
    Refine track assignments during dense dancer overlaps using OKS.

    Modifies all_frame_data in-place: corrects track_ids and boxes when:
    - IoU > iou_entry: use OKS to match poses to tracks
    - Total keypoint conf < occlusion_thresh: CVM ghost the box
    - IoU < iou_exit for exit_frames: return to standard

    Args:
        all_frame_data: List of {frame_idx, boxes, track_ids, poses, ...}
        iou_entry: IoU threshold to enter OKS matching
        iou_exit: IoU threshold to exit crossover
        exit_frames: Consecutive low-IoU frames to exit
        occlusion_thresh: Keypoint conf below which to use CVM

    Returns:
        Modified all_frame_data (in-place).
    """
    if not all_frame_data:
        return all_frame_data

    # Track state: last pose, last box, velocity, crossover pair state
    track_last_pose: Dict[int, np.ndarray] = {}
    track_last_box: Dict[int, Tuple[float, float, float, float]] = {}
    track_last_embedding: Dict[int, np.ndarray] = {}  # V3.8: Appearance for Re-ID
    track_velocity: Dict[int, np.ndarray] = {}
    track_kpt_velocity: Dict[int, np.ndarray] = {}  # shape (17, 2)
    track_occlusion_frames: Dict[int, int] = {}  # V3.4: frames spent occluded
    track_frozen_pose: Dict[int, np.ndarray] = {}  # V3.4: pose frozen after CVM phase
    track_re_emerging: Dict[int, int] = {}  # V3.4: blend-back frame counter
    pair_overlap_count: Dict[Tuple[int, int], int] = {}
    pair_in_crossover: Dict[Tuple[int, int], bool] = {}
    # Phase 5.5: Collision state — suspend ID swaps during overlap
    pair_collision_entry_count: Dict[Tuple[int, int], int] = {}
    pair_collision_exit_count: Dict[Tuple[int, int], int] = {}
    pair_in_collision: Dict[Tuple[int, int], bool] = {}
    pair_collision_start_fidx: Dict[Tuple[int, int], int] = {}  # first frame of collision (for audit)

    for f_idx, fd in enumerate(all_frame_data):
        boxes = fd.get("boxes", [])
        track_ids = fd.get("track_ids", [])
        poses = fd.get("poses", {})

        if len(boxes) == 0:
            continue

        boxes_arr = np.array(boxes, dtype=np.float64)
        n = len(boxes)
        tids = list(track_ids)

        # Pairwise IoU
        iou_mat = _compute_iou_matrix(boxes_arr)

        # Identify pairs in crossover (IoU > entry)
        crossover_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if iou_mat[i, j] >= iou_entry:
                    pair = (min(tids[i], tids[j]), max(tids[i], tids[j]))
                    crossover_pairs.append((i, j, pair))

        # Update exit counters for pairs not overlapping this frame
        for (ta, tb), count in list(pair_overlap_count.items()):
            key = (ta, tb)
            idx_a = next((k for k, tid in enumerate(tids) if tid == ta), None)
            idx_b = next((k for k, tid in enumerate(tids) if tid == tb), None)
            if idx_a is not None and idx_b is not None:
                iou_val = iou_mat[idx_a, idx_b]
                if iou_val < iou_exit:
                    pair_overlap_count[(ta, tb)] = count + 1
                    if count + 1 >= exit_frames:
                        pair_in_crossover[key] = False
                    # Phase 5.5: Collision exit — IoU < COLLISION_EXIT_IOU
                    pair_collision_entry_count[key] = 0
                    if iou_val < COLLISION_EXIT_IOU:
                        exit_cnt = pair_collision_exit_count.get(key, 0) + 1
                        pair_collision_exit_count[key] = exit_cnt
                        pair_collision_entry_count[key] = 0
                        if exit_cnt >= COLLISION_EXIT_CONSECUTIVE and pair_in_collision.get(key, False):
                            pair_in_collision[key] = False
                            pair_collision_entry_count[key] = 0
                            # Run Temporal OKS Audit; if swap wins, apply globally
                            start_fidx = pair_collision_start_fidx.pop(key, max(0, f_idx - 30))
                            audit = _run_temporal_oks_audit(
                                all_frame_data, ta, tb, start_fidx, f_idx, occlusion_thresh,
                                lookback=TEMPORAL_OKS_LOOKBACK, lookahead=TEMPORAL_OKS_LOOKAHEAD,
                            )
                            if audit is True:
                                _apply_id_swap_from_frame(all_frame_data, ta, tb, f_idx)
                                # Refresh local refs after swap
                                tids = all_frame_data[f_idx].get("track_ids", tids)
                                poses = all_frame_data[f_idx].get("poses", poses)
                                boxes = all_frame_data[f_idx].get("boxes", boxes)
                else:
                    pair_overlap_count[(ta, tb)] = 0
                    pair_collision_exit_count[key] = 0
                    if iou_val < COLLISION_ENTRY_IOU:
                        pair_collision_entry_count[key] = 0
            else:
                pair_overlap_count[(ta, tb)] = count + 1
                if count + 1 >= exit_frames:
                    pair_in_crossover[key] = False

        # For each crossover pair, enter crossover mode
        for (i, j, pair) in crossover_pairs:
            pair_overlap_count[pair] = 0
            pair_in_crossover[pair] = True
            pair_collision_exit_count[pair] = 0
            iou_val = iou_mat[i, j]
            if iou_val >= COLLISION_ENTRY_IOU:
                entry_cnt = pair_collision_entry_count.get(pair, 0) + 1
                pair_collision_entry_count[pair] = entry_cnt
                if entry_cnt >= COLLISION_ENTRY_CONSECUTIVE:
                    pair_in_collision[pair] = True
                    if pair not in pair_collision_start_fidx:
                        pair_collision_start_fidx[pair] = f_idx

        # V3.4 Hybrid CVM: ghost boxes with low keypoint confidence
        new_boxes = list(boxes)
        for idx, tid in enumerate(tids):
            if tid not in poses:
                continue
            kpts = poses[tid]["keypoints"]
            total_conf = _compute_total_keypoint_confidence(kpts)

            if total_conf < occlusion_thresh and tid in track_last_box and tid in track_velocity:
                occ_frames = track_occlusion_frames.get(tid, 0) + 1
                track_occlusion_frames[tid] = occ_frames

                # CVM: project box forward with displacement cap
                last_box = track_last_box[tid]
                v = track_velocity[tid]
                cx = (last_box[0] + last_box[2]) / 2
                cy = (last_box[1] + last_box[3]) / 2
                w = last_box[2] - last_box[0]
                h = last_box[3] - last_box[1]

                # V3.5: Cap displacement at CVM_MAX_DISPLACEMENT_FRAC * bbox_height
                max_disp = CVM_MAX_DISPLACEMENT_FRAC * max(h, 1.0)
                disp = np.sqrt(v[0] ** 2 + v[1] ** 2)
                if disp > max_disp and disp > 0:
                    scale = max_disp / disp
                    v = np.array([v[0] * scale, v[1] * scale])

                new_cx = cx + v[0]
                new_cy = cy + v[1]

                # V3.5: Clamp to frame bounds
                new_cx = max(w / 2, min(frame_width - w / 2, new_cx))
                new_cy = max(h / 2, min(frame_height - h / 2, new_cy))

                new_box = (
                    new_cx - w / 2, new_cy - h / 2,
                    new_cx + w / 2, new_cy + h / 2,
                )

                # V3.5: Freeze if projected box overlaps another live track
                freeze = False
                new_box_arr = np.array([[new_box[0], new_box[1], new_box[2], new_box[3]]])
                for other_idx, other_tid in enumerate(tids):
                    if other_idx == idx:
                        continue
                    other_box = new_boxes[other_idx]
                    other_arr = np.array([[other_box[0], other_box[1], other_box[2], other_box[3]]])
                    both = np.vstack([new_box_arr, other_arr])
                    iou_val = _compute_iou_matrix(both)[0, 1]
                    if iou_val > CVM_OVERLAP_FREEZE_IOU:
                        freeze = True
                        break

                if not freeze:
                    new_boxes[idx] = new_box
                # else: keep original box position (frozen)

                if tid in track_last_pose:
                    if occ_frames <= CVM_MAX_FRAMES:
                        # Phase 1: CVM extrapolation (frames 1-5) — preserve momentum
                        if tid in track_kpt_velocity:
                            v_kpt = track_kpt_velocity[tid]
                            new_pose = track_last_pose[tid].copy()
                            track_kpt_velocity[tid] = v_kpt * CVM_VELOCITY_DECAY
                            new_pose[:17, :2] += track_kpt_velocity[tid]
                            poses[tid]["keypoints"] = new_pose
                            track_frozen_pose[tid] = new_pose.copy()
                        else:
                            poses[tid]["keypoints"] = track_last_pose[tid].copy()
                            track_frozen_pose[tid] = track_last_pose[tid].copy()
                    else:
                        # Phase 2: Freeze pose + decay confidence (frames 6+)
                        frozen = track_frozen_pose.get(tid, track_last_pose[tid]).copy()
                        frozen[:17, 2] *= CONF_DECAY_FACTOR
                        track_frozen_pose[tid] = frozen
                        poses[tid]["keypoints"] = frozen
                        poses[tid]["scores"] = frozen[:17, 2].copy()
            else:
                # Track is visible — handle re-emergence blending
                prev_occ = track_occlusion_frames.get(tid, 0)
                if prev_occ > 0 and tid in track_frozen_pose:
                    blend_count = track_re_emerging.get(tid, 0) + 1
                    track_re_emerging[tid] = blend_count
                    if blend_count <= BLEND_FRAMES:
                        t = blend_count / BLEND_FRAMES
                        frozen = track_frozen_pose[tid]
                        live = kpts.copy()
                        blended = (1.0 - t) * frozen[:17] + t * live[:17]
                        blended[:, 2] = live[:17, 2]
                        poses[tid]["keypoints"] = blended
                    else:
                        track_re_emerging.pop(tid, None)
                        track_frozen_pose.pop(tid, None)
                else:
                    track_re_emerging.pop(tid, None)
                    track_frozen_pose.pop(tid, None)
                track_occlusion_frames[tid] = 0

        fd["boxes"] = new_boxes
        boxes = new_boxes

        # OKS matching for crossover pairs (skip swap when in collision — Phase 5.5)
        # V3.8: Add appearance cost — reject swap when red≠blue (cosine sim low)
        embeddings = fd.get("embeddings", {})
        swap_map = {}  # idx -> new_tid (if we need to swap)
        for (i, j, pair) in crossover_pairs:
            if not pair_in_crossover.get(pair, True):
                continue
            if pair_in_collision.get(pair, False):
                continue  # Phase 5.5: Locked state — don't commit to ID assignment
            tid_i, tid_j = tids[i], tids[j]
            if tid_i not in poses or tid_j not in poses:
                continue
            kpts_i = poses[tid_i]["keypoints"]
            kpts_j = poses[tid_j]["keypoints"]
            conf_i = _compute_total_keypoint_confidence(kpts_i)
            conf_j = _compute_total_keypoint_confidence(kpts_j)
            if conf_i < occlusion_thresh or conf_j < occlusion_thresh:
                continue
            area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            area = max(area_i, area_j, 1.0)
            last_pose_i = track_last_pose.get(tid_i)
            last_pose_j = track_last_pose.get(tid_j)
            if last_pose_i is None or last_pose_j is None:
                continue
            oks_i_to_i = _compute_oks(kpts_i, last_pose_i, area)
            oks_i_to_j = _compute_oks(kpts_i, last_pose_j, area)
            oks_j_to_i = _compute_oks(kpts_j, last_pose_i, area)
            oks_j_to_j = _compute_oks(kpts_j, last_pose_j, area)
            score_no_swap = oks_i_to_i + oks_j_to_j
            score_swap = oks_i_to_j + oks_j_to_i

            # V3.8: Appearance cost — if embeddings exist, favor assignment that matches appearance
            if _REID_AVAILABLE and embeddings:
                emb_i = embeddings.get(tid_i)
                emb_j = embeddings.get(tid_j)
                last_emb_i = track_last_embedding.get(tid_i)
                last_emb_j = track_last_embedding.get(tid_j)
                if emb_i is not None and emb_j is not None and last_emb_i is not None and last_emb_j is not None:
                    app_no_swap = (cosine_similarity(emb_i, last_emb_i) + cosine_similarity(emb_j, last_emb_j)) * APPEARANCE_WEIGHT
                    app_swap = (cosine_similarity(emb_i, last_emb_j) + cosine_similarity(emb_j, last_emb_i)) * APPEARANCE_WEIGHT
                    score_no_swap += app_no_swap
                    score_swap += app_swap

            if score_swap > score_no_swap:
                swap_map[i] = tid_j
                swap_map[j] = tid_i

        # Apply swaps: reassign track_ids and poses
        if swap_map:
            new_track_ids = list(track_ids)
            new_poses = {tid: data for tid, data in poses.items()}

            for idx, old_tid in enumerate(track_ids):
                if idx in swap_map:
                    new_tid = swap_map[idx]
                    new_track_ids[idx] = new_tid
                    if old_tid in poses:
                        new_poses[new_tid] = poses[old_tid]
                        if old_tid in new_poses and old_tid != new_tid:
                            del new_poses[old_tid]

            fd["track_ids"] = new_track_ids
            fd["poses"] = new_poses
            tids = new_track_ids
            poses = new_poses
            boxes = fd["boxes"]

        # Update track state for next frame (use resolved tids after swap)
        for idx, tid in enumerate(tids):
            if tid not in poses:
                continue
            kpts = poses[tid]["keypoints"]
            if kpts.shape[0] >= 17 and _compute_total_keypoint_confidence(kpts) >= occlusion_thresh:
                track_last_pose[tid] = kpts.copy()
            track_last_box[tid] = boxes[idx]
            if tid in embeddings:
                track_last_embedding[tid] = embeddings[tid].copy()
            if f_idx > 0:
                for pf in range(f_idx - 1, -1, -1):
                    pfd = all_frame_data[pf]
                    ptids = pfd.get("track_ids", [])
                    pboxes = pfd.get("boxes", [])
                    if tid in ptids:
                        pidx = ptids.index(tid)
                        if pidx < len(pboxes):
                            track_velocity[tid] = _compute_velocity(
                                tuple(pboxes[pidx]), tuple(boxes[idx])
                            )
                            if tid in track_last_pose:
                                prev_poses = pfd.get("poses", {})
                                if tid in prev_poses:
                                    pkpts = prev_poses[tid]["keypoints"]
                                    if pkpts.shape[0] >= 17:
                                        track_kpt_velocity[tid] = kpts[:17, :2] - pkpts[:17, :2]
                            break

    return all_frame_data
