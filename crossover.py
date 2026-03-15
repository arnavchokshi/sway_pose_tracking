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

Runs AFTER pose estimation (requires keypoints). All math vectorized with NumPy.
"""

from typing import Dict, List, Tuple, Any, Optional

import numpy as np

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

# V3.4: Visibility scoring
VISIBILITY_CONTAINMENT_THRESH = 0.7
VISIBILITY_MIN_SCORE = 0.3

# V3.4: Hybrid CVM constants
CVM_MAX_FRAMES = 5
CVM_VELOCITY_DECAY = 0.9
CONF_DECAY_FACTOR = 0.85
BLEND_FRAMES = 3
# V3.5: CVM displacement cap — fraction of bbox height per frame
CVM_MAX_DISPLACEMENT_FRAC = 0.3
# V3.5: CVM overlap freeze — don't project into another live track
CVM_OVERLAP_FREEZE_IOU = 0.5

# V3.4: Keypoint collision dedup thresholds
# V3.5: Relaxed from 0.2/0.3 to catch nearby phantoms (e.g. 67px apart at bbox_h=165px)
COLLISION_KPT_DIST_FRAC = 0.35
COLLISION_CENTER_DIST_FRAC = 0.5


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


def deduplicate_collocated_poses(
    frame_data: Dict[str, Any],
    kpt_dist_frac: float = COLLISION_KPT_DIST_FRAC,
    center_dist_frac: float = COLLISION_CENTER_DIST_FRAC,
) -> None:
    """
    V3.4: Remove duplicate pose overlays on the same physical person.
    Uses median keypoint distance (robust to hallucinated outlier keypoints) and
    requires bbox centers to also be close (protects legitimate partner work).
    Suppresses the lower-confidence track. Modifies frame_data in-place.
    """
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

            if median_dist > kpt_dist_frac * bbox_h:
                continue

            center_a = np.array([(boxes[i][0]+boxes[i][2])/2, (boxes[i][1]+boxes[i][3])/2])
            center_b = np.array([(boxes[j][0]+boxes[j][2])/2, (boxes[j][1]+boxes[j][3])/2])
            center_dist = float(np.linalg.norm(center_a - center_b))
            if center_dist > center_dist_frac * bbox_h:
                continue

            conf_a = float(np.mean(kpts_a[:17, 2]))
            conf_b = float(np.mean(kpts_b[:17, 2]))
            to_suppress.add(tid_b if conf_a >= conf_b else tid_a)

    for tid in to_suppress:
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


def apply_occlusion_reid(
    all_frame_data: List[Dict[str, Any]],
    max_frame_gap: int = REID_MAX_FRAME_GAP,
    min_oks: float = REID_MIN_OKS,
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
    track_info: Dict[int, Dict] = {}
    for f_idx, fd in enumerate(all_frame_data):
        track_ids = fd.get("track_ids", [])
        poses = fd.get("poses", {})
        boxes = fd.get("boxes", [])
        for idx, tid in enumerate(track_ids):
            if tid not in poses or idx >= len(boxes):
                continue
            kpts = poses[tid].get("keypoints")
            if kpts is None or kpts.shape[0] < 17:
                continue
            if tid not in track_info:
                track_info[tid] = {
                    "first_frame": f_idx,
                    "last_frame": f_idx,
                    "first_pose": kpts.copy(),
                    "last_pose": kpts.copy(),
                    "first_box": tuple(boxes[idx]),
                    "last_box": tuple(boxes[idx]),
                }
            else:
                track_info[tid]["last_frame"] = f_idx
                track_info[tid]["last_pose"] = kpts.copy()
                track_info[tid]["last_box"] = tuple(boxes[idx])

    total_frames = len(all_frame_data)
    dead_ids = [tid for tid, info in track_info.items() if info["last_frame"] < total_frames - 1]
    newborn_ids = [tid for tid, info in track_info.items() if info["first_frame"] > 0]

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
            oks = _compute_oks(last_pose_a, first_pose_b, area_a)
            if oks >= min_oks:
                merge_map[tid_b] = tid_a
                break  # One merge per dead track

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
    track_velocity: Dict[int, np.ndarray] = {}
    track_kpt_velocity: Dict[int, np.ndarray] = {}  # shape (17, 2)
    track_occlusion_frames: Dict[int, int] = {}  # V3.4: frames spent occluded
    track_frozen_pose: Dict[int, np.ndarray] = {}  # V3.4: pose frozen after CVM phase
    track_re_emerging: Dict[int, int] = {}  # V3.4: blend-back frame counter
    pair_overlap_count: Dict[Tuple[int, int], int] = {}
    pair_in_crossover: Dict[Tuple[int, int], bool] = {}

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
                else:
                    pair_overlap_count[(ta, tb)] = 0
            else:
                pair_overlap_count[(ta, tb)] = count + 1
                if count + 1 >= exit_frames:
                    pair_in_crossover[key] = False

        # For each crossover pair, enter crossover mode
        for (i, j, pair) in crossover_pairs:
            pair_overlap_count[pair] = 0
            pair_in_crossover[pair] = True

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

        # OKS matching for crossover pairs
        swap_map = {}  # idx -> new_tid (if we need to swap)
        for (i, j, pair) in crossover_pairs:
            if not pair_in_crossover.get(pair, True):
                continue
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
