"""
Phase 2: Crossover Management — OKS-Based Dense Overlap Handling

Handles dancer overlaps using Object Keypoint Similarity (OKS):
- Crossover Entry: When IoU > 0.6 between two track boxes, switch to OKS matching
- OKS Occlusion Fallback: When total keypoint confidence < 0.3, suspend matching
  and project box forward using Constant Velocity Model (CVM)
- Crossover Exit: When IoU < 0.4 for 3 consecutive frames, return to standard tracking

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
REID_MIN_OKS = 0.5  # Min OKS between last pose of dead and first pose of newborn to merge


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

    if not merge_map:
        return all_frame_data

    for fd in all_frame_data:
        track_ids = fd.get("track_ids", [])
        poses = fd.get("poses", {})
        new_track_ids = [merge_map.get(tid, tid) for tid in track_ids]
        new_poses = {merge_map.get(tid, tid): poses[tid] for tid in track_ids if tid in poses}
        fd["track_ids"] = new_track_ids
        fd["poses"] = new_poses

    return all_frame_data


def apply_crossover_refinement(
    all_frame_data: List[Dict[str, Any]],
    iou_entry: float = IOU_CROSSOVER_ENTRY,
    iou_exit: float = IOU_CROSSOVER_EXIT,
    exit_frames: int = CROSSOVER_EXIT_FRAMES,
    occlusion_thresh: float = OKS_OCCLUSION_THRESHOLD,
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

        # Process occlusion: ghost boxes with low keypoint confidence
        new_boxes = list(boxes)
        for idx, tid in enumerate(tids):
            if tid not in poses:
                continue
            kpts = poses[tid]["keypoints"]
            total_conf = _compute_total_keypoint_confidence(kpts)
            if total_conf < occlusion_thresh and tid in track_last_box and tid in track_velocity:
                # CVM: project box forward
                last_box = track_last_box[tid]
                v = track_velocity[tid]
                cx = (last_box[0] + last_box[2]) / 2
                cy = (last_box[1] + last_box[3]) / 2
                w = last_box[2] - last_box[0]
                h = last_box[3] - last_box[1]
                new_cx, new_cy = cx + v[0], cy + v[1]
                new_box = (
                    new_cx - w / 2, new_cy - h / 2,
                    new_cx + w / 2, new_cy + h / 2,
                )
                new_boxes[idx] = new_box
                # Don't update track state from occluded pose

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

        # Apply swaps: reassign track_ids and poses (box i's pose goes to new_tid)
        if swap_map:
            new_track_ids = list(track_ids)
            new_poses = {tid: data for tid, data in poses.items()}
            for idx, new_tid in swap_map.items():
                old_tid = tids[idx]
                new_track_ids[idx] = new_tid
                new_poses[new_tid] = poses[old_tid]
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
                            break

    return all_frame_data
