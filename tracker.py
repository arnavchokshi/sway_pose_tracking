"""
Detection & Tracking Module — YOLO11l + BoT-SORT (V3.4)

V3.0: Streaming 300-frame chunks, native FPS, YOLO11l conf=0.25.
V3.3: YOLO runs on every frame for better dancer detection.
V3.4: Relative stitch radius (0.5x bbox height) + velocity-consistency check.

Double-Layer Tracking: Base tracker + OKS crossover refinement (see crossover.py)
handles dense overlaps when IoU > 0.6.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator

import cv2
import numpy as np
from ultralytics import YOLO

# V3.0: Streaming chunk size (10 seconds at 30 FPS)
CHUNK_SIZE = 300

# Detection resolution for YOLO
DETECT_SIZE = 640

# V3.0: YOLO confidence — lower to catch more dancers (pruning removes false positives)
YOLO_CONF = 0.25

# YOLO detection stride: 1 = every frame, 2 = even frames only (2x speed, minimal accuracy loss)
# Odd frames filled via linear interpolation before downstream phases.
YOLO_DETECTION_STRIDE = 1

# Stitching params: occlusion drop-out recovery (V3.0: 180 frames = 6s @ 30 FPS, or 3s @ 15fps YOLO)
STITCH_MAX_FRAME_GAP = 60
# V3.4: Relative stitch radius — fraction of track's last bbox height
STITCH_RADIUS_BBOX_FRAC = 0.5
# Fallback absolute radius when bbox height unavailable
STITCH_MAX_PIXEL_RADIUS = 120.0
STITCH_PREDICTED_RADIUS_FRAC = 0.75
# Short-gap threshold: gaps this short use generous matching (no velocity check)
SHORT_GAP_FRAMES = 20


def _box_center(box: Tuple) -> Tuple[float, float]:
    """(x1,y1,x2,y2) -> (cx, cy)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two boxes (x1, y1, x2, y2)."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _fill_stride_gaps(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    stride: int,
) -> None:
    """
    Fill missing frame entries when YOLO runs only every Nth frame.
    Inserts linearly interpolated boxes for skipped frames. Modifies raw_tracks in place.
    """
    if stride <= 1:
        return
    for tid, entries in list(raw_tracks.items()):
        sorted_entries = sorted(entries, key=lambda e: e[0])
        new_entries = []
        for i, (f, box, conf) in enumerate(sorted_entries):
            new_entries.append((f, box, conf))
            if i + 1 < len(sorted_entries):
                next_f, next_box, _ = sorted_entries[i + 1]
                for gap_f in range(f + 1, next_f):
                    t = (gap_f - f) / (next_f - f)
                    interp_box = _interpolate_box(box, next_box, t)
                    new_entries.append((gap_f, interp_box, 0.5))
        raw_tracks[tid] = sorted(new_entries, key=lambda e: e[0])


def _interpolate_box_sequence(
    box_last: Tuple,
    box_first: Tuple,
    frame_last: int,
    frame_first: int,
) -> List[Tuple[int, Tuple, float]]:
    """Linearly interpolate boxes between frame_last and frame_first (exclusive)."""
    if frame_first <= frame_last + 1:
        return []
    entries = []
    conf = 0.5  # Placeholder for interpolated frames
    for f in range(frame_last + 1, frame_first):
        t = (f - frame_last) / (frame_first - frame_last)
        box = _interpolate_box(box_last, box_first, t)
        entries.append((f, box, conf))
    return entries


def _estimate_track_velocity(entries: List[Tuple[int, Tuple, float]], n_tail: int = 5) -> Tuple[float, float]:
    """
    Estimate velocity (vx, vy) in pixels per frame from last n_tail entries.
    Returns (0, 0) if insufficient data.
    """
    if len(entries) < 2 or n_tail < 2:
        return (0.0, 0.0)
    sorted_entries = sorted(entries, key=lambda e: e[0])
    tail = sorted_entries[-n_tail:]
    centers = [_box_center(e[1]) for e in tail]
    frames = [e[0] for e in tail]
    total_df = frames[-1] - frames[0]
    if total_df <= 0:
        return (0.0, 0.0)
    vx = (centers[-1][0] - centers[0][0]) / total_df
    vy = (centers[-1][1] - centers[0][1]) / total_df
    return (vx, vy)


def _bbox_height(box: Tuple) -> float:
    """(x1,y1,x2,y2) -> height."""
    return box[3] - box[1]


def stitch_fragmented_tracks(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    total_frames: int,
    max_frame_gap: int = STITCH_MAX_FRAME_GAP,
    radius_bbox_frac: float = STITCH_RADIUS_BBOX_FRAC,
    predicted_radius_frac: float = STITCH_PREDICTED_RADIUS_FRAC,
    fallback_radius: float = STITCH_MAX_PIXEL_RADIUS,
) -> Dict[int, List[Tuple[int, Tuple, float]]]:
    """
    V3.4: Stitch tracks that fragmented due to occlusion. Uses relative stitch
    radius (fraction of bbox height) instead of fixed pixel radius. Adds
    velocity-consistency check to prevent merging unrelated tracks.

    Modifies raw_tracks in place and returns it.
    """
    if total_frames <= 0 or not raw_tracks:
        return raw_tracks

    track_info: Dict[int, Dict] = {}
    for tid, entries in raw_tracks.items():
        if not entries:
            continue
        sorted_entries = sorted(entries, key=lambda e: e[0])
        first_f = sorted_entries[0][0]
        last_f = sorted_entries[-1][0]
        track_info[tid] = {
            "first_frame": first_f,
            "last_frame": last_f,
            "first_box": sorted_entries[0][1],
            "last_box": sorted_entries[-1][1],
            "entries": sorted_entries,
        }

    changed = True
    while changed:
        changed = False
        dead_ids = [
            tid for tid, info in track_info.items()
            if info["last_frame"] < total_frames - 1
        ]

        for tid_a in dead_ids:
            if tid_a not in track_info:
                continue
            info_a = track_info[tid_a]
            frame_a_last = info_a["last_frame"]
            cx_a, cy_a = _box_center(info_a["last_box"])
            h_a = _bbox_height(info_a["last_box"])

            # V3.4: Radius scales with bbox height
            radius_last = max(radius_bbox_frac * h_a, fallback_radius * 0.5) if h_a > 0 else fallback_radius
            vx, vy = _estimate_track_velocity(info_a["entries"])
            has_velocity = (vx != 0 or vy != 0)
            radius_pred = max(predicted_radius_frac * h_a, fallback_radius * 0.75) if (has_velocity and h_a > 0) else radius_last

            best_b = None
            best_dist = max(radius_last, radius_pred) + 1.0

            for tid_b, info_b in list(track_info.items()):
                if tid_b == tid_a:
                    continue
                frame_b_first = info_b["first_frame"]
                gap = frame_b_first - frame_a_last
                if gap <= 0 or gap > max_frame_gap:
                    continue
                cx_b, cy_b = _box_center(info_b["first_box"])

                short_gap = gap <= SHORT_GAP_FRAMES

                # Short gaps: generous radius (1x bbox height), no velocity check
                eff_radius_last = max(h_a, fallback_radius) if (short_gap and h_a > 0) else radius_last
                eff_radius_pred = eff_radius_last if short_gap else radius_pred

                dist_last = np.sqrt((cx_b - cx_a) ** 2 + (cy_b - cy_a) ** 2)
                pred_cx = cx_a + vx * gap
                pred_cy = cy_a + vy * gap
                dist_pred = np.sqrt((cx_b - pred_cx) ** 2 + (cy_b - pred_cy) ** 2)

                spatial_ok = (dist_last <= eff_radius_last or dist_pred <= eff_radius_pred)
                if not spatial_ok:
                    continue

                if not short_gap and has_velocity and len(info_b["entries"]) >= 3:
                    vx_b, vy_b = _estimate_track_velocity(info_b["entries"], n_tail=min(5, len(info_b["entries"])))
                    dot = vx * vx_b + vy * vy_b
                    speed_a = np.sqrt(vx**2 + vy**2)
                    speed_b = np.sqrt(vx_b**2 + vy_b**2)
                    if speed_a > 1.0 and speed_b > 1.0 and dot < 0:
                        continue

                candidate_dist = min(dist_last, dist_pred)
                if candidate_dist < best_dist:
                    best_dist = candidate_dist
                    best_b = tid_b

            if best_b is None:
                continue

            entries_a = info_a["entries"]
            entries_b = track_info[best_b]["entries"]
            box_a_last = info_a["last_box"]
            box_b_first = entries_b[0][1]
            gap_entries = _interpolate_box_sequence(
                box_a_last, box_b_first, frame_a_last, entries_b[0][0]
            )
            merged = sorted(entries_a + gap_entries + entries_b, key=lambda e: e[0])

            raw_tracks[tid_a] = merged
            del raw_tracks[best_b]

            track_info[tid_a] = {
                "first_frame": merged[0][0],
                "last_frame": merged[-1][0],
                "first_box": merged[0][1],
                "last_box": merged[-1][1],
                "entries": merged,
            }
            del track_info[best_b]
            changed = True
            break

    return raw_tracks


def coalescence_deduplicate(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    iou_thresh: float = 0.65,
    consecutive_frames: int = 10,
) -> Dict[int, List[Tuple[int, Tuple, float]]]:
    """
    Remove duplicated tracks (ghosts). If two tracks overlap highly (IoU > iou_thresh)
    for N consecutive frames, consider them the same person and delete the younger/shorter track.
    
    Returns raw_tracks modified in place.
    """
    # Restructure data to mapping of {frame: [(tid, box)...]}
    frames_dict: Dict[int, List[Tuple[int, Tuple]]] = {}
    track_age: Dict[int, int] = {}
    for tid, entries in raw_tracks.items():
        track_age[tid] = len(entries)
        for f, box, _conf in entries:
            if f not in frames_dict:
                frames_dict[f] = []
            frames_dict[f].append((tid, box))
            
    # Count consecutive overlaps between ID pairs
    overlap_counts: Dict[Tuple[int, int], int] = {}
    # tuple (tid1, tid2) where tid1 < tid2
    
    # Needs to be sorted so "consecutive" logic holds true
    for f in sorted(frames_dict.keys()):
        dets = frames_dict[f]
        current_overlaps = set()
        
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                tid1, box1 = dets[i]
                tid2, box2 = dets[j]
                
                iou = _compute_iou(box1, box2)
                if iou > iou_thresh:
                    pair = tuple(sorted([tid1, tid2]))
                    current_overlaps.add(pair)
                    
        # Update running consecutive counts
        to_delete = []
        for pair in list(overlap_counts.keys()):
            if pair not in current_overlaps:
                del overlap_counts[pair]
                
        for pair in current_overlaps:
            overlap_counts[pair] = overlap_counts.get(pair, 0) + 1

    # Find IDs to kill
    dead_ids = set()
    for (tid1, tid2), count in overlap_counts.items():
        if count >= consecutive_frames:
            # Kill the younger track (the one with fewer total frames)
            if track_age[tid1] >= track_age[tid2]:
                dead_ids.add(tid2)
            else:
                dead_ids.add(tid1)
                
    for tid in dead_ids:
        if tid in raw_tracks:
            del raw_tracks[tid]
            
    return raw_tracks


def merge_complementary_tracks(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    max_center_dist_frac: float = 0.5,
) -> Dict[int, List[Tuple[int, Tuple, float]]]:
    """
    Merge track pairs that cover complementary (non-overlapping) time segments
    of the same person — e.g. BoT-SORT assigns ID 4 for frames 0-150 and 260-385,
    and ID 17 for frames 152-258.  Stitch/re-ID miss this because neither track
    is cleanly "dead" before the other is "born".

    Criteria for merging:
      1. Zero temporal overlap (no shared frames).
      2. At every transition boundary the bbox centers are within
         max_center_dist_frac * bbox_height of each other.

    The shorter track is merged into the longer one.
    Modifies raw_tracks in place and returns it.
    """
    if len(raw_tracks) < 2:
        return raw_tracks

    tid_list = list(raw_tracks.keys())
    frame_sets: Dict[int, set] = {}
    sorted_entries: Dict[int, list] = {}
    for tid in tid_list:
        entries = sorted(raw_tracks[tid], key=lambda e: e[0])
        sorted_entries[tid] = entries
        frame_sets[tid] = {e[0] for e in entries}

    changed = True
    while changed:
        changed = False
        tid_list = list(raw_tracks.keys())
        for i in range(len(tid_list)):
            if changed:
                break
            tid_a = tid_list[i]
            if tid_a not in raw_tracks:
                continue
            for j in range(i + 1, len(tid_list)):
                tid_b = tid_list[j]
                if tid_b not in raw_tracks:
                    continue

                if frame_sets[tid_a] & frame_sets[tid_b]:
                    continue

                entries_a = sorted_entries[tid_a]
                entries_b = sorted_entries[tid_b]

                # Find transition boundaries: segments where one ends and the
                # other starts (or vice versa).  Check bbox proximity at each.
                all_entries = sorted(entries_a + entries_b, key=lambda e: e[0])
                boundaries_ok = True
                boundary_count = 0

                prev_owner = None
                prev_box = None
                for entry in all_entries:
                    fidx, box, _ = entry
                    owner = tid_a if fidx in frame_sets[tid_a] else tid_b
                    if prev_owner is not None and owner != prev_owner:
                        boundary_count += 1
                        h = max(_bbox_height(prev_box), _bbox_height(box), 1.0)
                        cx1, cy1 = _box_center(prev_box)
                        cx2, cy2 = _box_center(box)
                        dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                        if dist > max_center_dist_frac * h:
                            boundaries_ok = False
                            break
                    prev_owner = owner
                    prev_box = box

                if not boundaries_ok or boundary_count == 0:
                    continue

                # Merge: keep the longer track's ID
                if len(entries_a) >= len(entries_b):
                    keep, kill = tid_a, tid_b
                else:
                    keep, kill = tid_b, tid_a

                merged = sorted(raw_tracks[keep] + raw_tracks[kill], key=lambda e: e[0])

                # Interpolate gaps at each transition boundary
                gap_entries = []
                for k in range(len(merged) - 1):
                    f_cur = merged[k][0]
                    f_nxt = merged[k + 1][0]
                    if f_nxt - f_cur > 1:
                        gap_entries.extend(
                            _interpolate_box_sequence(merged[k][1], merged[k + 1][1], f_cur, f_nxt)
                        )
                merged = sorted(merged + gap_entries, key=lambda e: e[0])

                raw_tracks[keep] = merged
                del raw_tracks[kill]
                sorted_entries[keep] = merged
                frame_sets[keep] = {e[0] for e in merged}
                if kill in sorted_entries:
                    del sorted_entries[kill]
                if kill in frame_sets:
                    del frame_sets[kill]
                changed = True
                break

    return raw_tracks


def _get_tracker_config() -> str:
    """Return tracker config path. Prefer local occlusion-tolerant config (V3.0: track_buffer=90)."""
    cfg_dir = Path(__file__).resolve().parent
    for name in ("ocsort.yaml", "botsort.yaml"):
        p = cfg_dir / name
        if p.exists():
            return str(p)
    return "botsort.yaml"


def _iter_video_chunks(
    video_path: str,
    chunk_size: int = CHUNK_SIZE,
) -> Generator[Tuple[List[Tuple[int, np.ndarray]], int, float, int, int], None, None]:
    """
    Yield chunks of (frame_idx, frame_bgr) from video.
    Yields: (chunk_frames, chunk_start_idx, native_fps, frame_width, frame_height)
    """
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_f = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_f = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    chunk = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if chunk:
                    yield (chunk, frame_idx - len(chunk), native_fps, w_f, h_f)
                break
            chunk.append((frame_idx, frame.copy()))
            frame_idx += 1
            if len(chunk) >= chunk_size:
                yield (chunk, chunk[0][0], native_fps, w_f, h_f)
                chunk = []
    finally:
        cap.release()


def run_tracking(video_path: str) -> Tuple[Dict[int, List[Tuple[int, Tuple, float]]], int, float, Optional[List[Tuple[int, np.ndarray]]], float, int, int]:
    """
    V3.0: Run YOLO11l detection with BoT-SORT in streaming 300-frame chunks at native FPS.

    - Reads video in 300-frame chunks to avoid full RAM load
    - Processes every frame (native FPS, no decimation)
    - YOLO11l at 640x640, conf=0.25
    - BoT-SORT track_buffer=90 (3s at 30 FPS)
    - Wave 1 box stitch within 90 frames

    Returns:
        Tuple of:
        - raw_tracks: Dict[track_id, List[(frame_idx, (x1,y1,x2,y2), conf), ...]]
        - total_frames: Number of processed frames
        - output_fps: Native FPS (for downstream logic)
        - frames_list: None (V3.0 streaming — caller re-reads video for pose phase)
        - native_fps: Original video FPS
        - frame_width: Video width
        - frame_height: Video height
    """
    model_path = "yolo11m.pt"
    if Path("yolo11l.mlpackage").exists():
        model_path = "yolo11l.mlpackage"
    elif Path("yolo11m.mlpackage").exists():
        model_path = "yolo11m.mlpackage"
    
    print(f"Loading detection model: {model_path}")
    model = YOLO(model_path)
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]] = {}
    total_frames = 0
    native_fps = 30.0
    frame_width = 1920
    frame_height = 1080

    tracker_cfg = _get_tracker_config()

    # Dynamic resolution scaling initialization
    max_dancers_last_chunk = 0
    current_detect_size = DETECT_SIZE

    for chunk_frames, _chunk_start, nfps, w_f, h_f in _iter_video_chunks(video_path, CHUNK_SIZE):
        native_fps = nfps
        frame_width = w_f
        frame_height = h_f

        # Adjust detection size based on previous chunk crowd density
        if max_dancers_last_chunk > 4:
            current_detect_size = 960
            # print(f"  [Dynamic Scaling] Crowd detected. Up-scaling YOLO resolution to {current_detect_size}x{current_detect_size}")
        else:
            current_detect_size = DETECT_SIZE

        max_dancers_this_chunk = 0

        for frame_idx, frame in chunk_frames:
            if frame_idx % YOLO_DETECTION_STRIDE != 0:
                continue
            h_fr, w_fr = frame.shape[:2]
            frame_low = cv2.resize(frame, (current_detect_size, current_detect_size))
            frame_low_rgb = frame_low[:, :, ::-1]
            scale_x = w_fr / current_detect_size
            scale_y = h_fr / current_detect_size

            result = model.track(
                frame_low_rgb,
                tracker=tracker_cfg,
                classes=[0],
                conf=YOLO_CONF,
                persist=True,
                verbose=False,
            )
            result = result[0] if isinstance(result, list) else result
            boxes_data = _extract_boxes_and_ids(result)
            boxes_low = boxes_data["boxes"]
            track_ids = boxes_data["track_ids"]
            confs = boxes_data["confs"]

            valid_dancers_this_frame = 0
            for i, (box, tid, conf) in enumerate(zip(boxes_low, track_ids, confs)):
                if tid < 0:
                    continue
                valid_dancers_this_frame += 1
                x1, y1, x2, y2 = box
                box_hr = (
                    float(x1 * scale_x), float(y1 * scale_y),
                    float(x2 * scale_x), float(y2 * scale_y),
                )
                if tid not in raw_tracks:
                    raw_tracks[tid] = []
                raw_tracks[tid].append((frame_idx, box_hr, conf))
            
            max_dancers_this_chunk = max(max_dancers_this_chunk, valid_dancers_this_frame)

            if frame_idx == 0 or frame_idx == 30:
                n = len([t for t in track_ids if t >= 0])
                print(f"  Frame {frame_idx}: {n} persons (YOLO resol: {current_detect_size})")

        max_dancers_last_chunk = max_dancers_this_chunk

        total_frames += len(chunk_frames)
        del chunk_frames

    output_fps = native_fps

    # Post-tracking: stitch fragments from occlusion drop-outs
    raw_tracks = stitch_fragmented_tracks(raw_tracks, total_frames)
    
    # Wave 1.5: Coexistence deduplication to remove duplicate counts
    raw_tracks = coalescence_deduplicate(raw_tracks, iou_thresh=0.70, consecutive_frames=10)

    # Wave 1.6: Merge complementary tracks (same person, alternating IDs, zero overlap)
    raw_tracks = merge_complementary_tracks(raw_tracks)

    _fill_stride_gaps(raw_tracks, YOLO_DETECTION_STRIDE)

    return raw_tracks, total_frames, float(output_fps), None, float(native_fps), frame_width, frame_height


def iter_video_frames(video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Stream video frames one at a time for pose estimation phase.
    Yields (frame_idx, frame_bgr) in order.
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield (frame_idx, frame)
            frame_idx += 1
    finally:
        cap.release()


def _interpolate_box(
    box_prev: Tuple[float, float, float, float],
    box_next: Tuple[float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float]:
    """Linear interpolation: t=0 -> prev, t=1 -> next."""
    return (
        box_prev[0] + t * (box_next[0] - box_prev[0]),
        box_prev[1] + t * (box_next[1] - box_prev[1]),
        box_prev[2] + t * (box_next[2] - box_prev[2]),
        box_prev[3] + t * (box_next[3] - box_prev[3]),
    )


def _extract_boxes_and_ids(result) -> Dict[str, List]:
    """Extract bounding boxes, track IDs, and confidences from a YOLO result."""
    boxes = []
    track_ids = []
    confs = []

    if result.boxes is None or len(result.boxes) == 0:
        return {"boxes": boxes, "track_ids": track_ids, "confs": confs}

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    ids = result.boxes.id

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        boxes.append((float(x1), float(y1), float(x2), float(y2)))
        confs.append(float(conf[i]))
        tid = int(ids[i].item()) if ids is not None else -1
        track_ids.append(tid)

    return {"boxes": boxes, "track_ids": track_ids, "confs": confs}
