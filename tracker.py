"""
Detection & Tracking Module — YOLO11l + BoT-SORT (V3.0)

V3.0: Streaming 300-frame chunks, native FPS, YOLO11l conf=0.25.
- Reads video in chunks to avoid full RAM load
- Processes at native frame rate (30+ FPS) — no decimation
- YOLO11l (Large) at 640x640, conf=0.25 for higher recall
- BoT-SORT track_buffer=90 (3s at 30 FPS)
- Wave 1 box stitch within 90 frames

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

# Stitching params: occlusion drop-out recovery (V3.0: 90 frames = 3s @ 30 FPS)
STITCH_MAX_FRAME_GAP = 90
STITCH_MAX_PIXEL_RADIUS = 120.0
STITCH_PREDICTED_RADIUS = 180.0


def _box_center(box: Tuple) -> Tuple[float, float]:
    """(x1,y1,x2,y2) -> (cx, cy)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


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


def stitch_fragmented_tracks(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    total_frames: int,
    max_frame_gap: int = STITCH_MAX_FRAME_GAP,
    max_pixel_radius: float = STITCH_MAX_PIXEL_RADIUS,
    max_predicted_radius: float = STITCH_PREDICTED_RADIUS,
) -> Dict[int, List[Tuple[int, Tuple, float]]]:
    """
    Stitch tracks that fragmented due to occlusion. When ID_A dies and ID_B appears
    within max_frame_gap and near A's last or predicted position, merge B into A
    with linear interpolation across the gap.

    Uses velocity extrapolation: if A was moving before occlusion, predicts where
    A would reappear and accepts B if B is near that predicted position (handles
    performers moving while covered).

    Modifies raw_tracks in place and returns it.
    """
    if total_frames <= 0 or not raw_tracks:
        return raw_tracks

    # Build: track_id -> (first_frame, last_frame, first_box, last_box, entries)
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

    # Iterate until no more stitch candidates (handles A->B->C chains)
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

            # Velocity extrapolation: predict where A would be at B's first frame
            vx, vy = _estimate_track_velocity(info_a["entries"])

            best_b = None
            best_dist = max(max_pixel_radius, max_predicted_radius) + 1.0

            for tid_b, info_b in list(track_info.items()):
                if tid_b == tid_a:
                    continue
                frame_b_first = info_b["first_frame"]
                gap = frame_b_first - frame_a_last
                if gap <= 0 or gap > max_frame_gap:
                    continue
                cx_b, cy_b = _box_center(info_b["first_box"])

                # Distance from A's last position (stationary occlusion)
                dist_last = np.sqrt((cx_b - cx_a) ** 2 + (cy_b - cy_a) ** 2)
                # Distance from A's predicted position (moving occlusion)
                pred_cx = cx_a + vx * gap
                pred_cy = cy_a + vy * gap
                dist_pred = np.sqrt((cx_b - pred_cx) ** 2 + (cy_b - pred_cy) ** 2)

                # Match if B is near last position OR near predicted position
                radius_last = max_pixel_radius
                radius_pred = max_predicted_radius if (vx != 0 or vy != 0) else max_pixel_radius
                if (dist_last <= radius_last or dist_pred <= radius_pred) and min(dist_last, dist_pred) < best_dist:
                    best_dist = min(dist_last, dist_pred)
                    best_b = tid_b

            if best_b is None:
                continue

            # Merge: A + gap interpolation + B
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
            break  # Restart scan after modification

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
    model = YOLO("yolo11l.pt")
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]] = {}
    total_frames = 0
    native_fps = 30.0
    frame_width = 1920
    frame_height = 1080

    tracker_cfg = _get_tracker_config()

    for chunk_frames, _chunk_start, nfps, w_f, h_f in _iter_video_chunks(video_path, CHUNK_SIZE):
        native_fps = nfps
        frame_width = w_f
        frame_height = h_f

        for frame_idx, frame in chunk_frames:
            h_fr, w_fr = frame.shape[:2]
            frame_rgb = frame[:, :, ::-1]
            frame_low = cv2.resize(frame, (DETECT_SIZE, DETECT_SIZE))
            frame_low_rgb = frame_low[:, :, ::-1]
            scale_x = w_fr / DETECT_SIZE
            scale_y = h_fr / DETECT_SIZE

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

            for i, (box, tid, conf) in enumerate(zip(boxes_low, track_ids, confs)):
                if tid < 0:
                    continue
                x1, y1, x2, y2 = box
                box_hr = (
                    float(x1 * scale_x), float(y1 * scale_y),
                    float(x2 * scale_x), float(y2 * scale_y),
                )
                if tid not in raw_tracks:
                    raw_tracks[tid] = []
                raw_tracks[tid].append((frame_idx, box_hr, conf))

            if frame_idx == 0 or frame_idx == 30:
                n = len([t for t in track_ids if t >= 0])
                print(f"  Frame {frame_idx}: {n} persons")

        total_frames += len(chunk_frames)
        # Release chunk memory
        del chunk_frames

    output_fps = native_fps

    # Post-tracking: stitch fragments from occlusion drop-outs
    raw_tracks = stitch_fragmented_tracks(raw_tracks, total_frames)

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
