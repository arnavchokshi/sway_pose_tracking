"""
Detection & Tracking Module — YOLO26l + BoxMOT Deep OC-SORT by default (V3.4)

Hybrid SAM (default on): SWAY_HYBRID_SAM_OVERLAP=0 disables. When enabled, Ultralytics SAM2 runs on frames where
person boxes overlap heavily, tightening xyxy before DeepOCSORT (see hybrid_sam_refiner.py).

V3.0: Streaming 300-frame chunks, native FPS; default detector YOLO26l (Core ML yolo11* still used if present).
V3.3: YOLO runs on every frame for better dancer detection.
V3.4: Relative stitch radius (0.5x bbox height) + velocity-consistency check.
YOLO26 weights: pre-track DIoU-NMS is skipped (Ultralytics NMS + classical IoU-NMS at 0.50 only).

Double-Layer Tracking: Base tracker + OKS crossover refinement (see crossover.py)
handles dense overlaps when IoU > 0.6.
"""

import os
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Generator, Set

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

from sway.hybrid_sam_refiner import HybridSamRefiner, load_hybrid_sam_config
from sway.track_observation import (
    coerce_observation,
    TrackObservation,
    assign_sam_masks_to_tracker_output,
    resize_mask_to_bbox,
)

# V3.0: Streaming chunk size (10 seconds at 30 FPS)
CHUNK_SIZE = 300

# Detection resolution for YOLO
DETECT_SIZE = 640

# V3.0: YOLO confidence — lower to catch more dancers (pruning removes false positives)
# V3.6: 0.22 to improve recall for front/left dancers; pruning removes false positives
YOLO_CONF = 0.22

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


def load_tracking_runtime() -> Dict[str, Any]:
    """
    Read tracking/detection/stitch overrides from the environment (set per subprocess by Pipeline Lab).

    Defaults match module constants when env is unset.
    """
    def _iget(name: str, default: int) -> int:
        v = os.environ.get(name, "").strip()
        if not v:
            return default
        return int(v)

    def _fget(name: str, default: float) -> float:
        v = os.environ.get(name, "").strip()
        if not v:
            return default
        return float(v)

    return {
        "chunk_size": _iget("SWAY_CHUNK_SIZE", CHUNK_SIZE),
        "detect_size": _iget("SWAY_DETECT_SIZE", DETECT_SIZE),
        "yolo_conf": _fget("SWAY_YOLO_CONF", YOLO_CONF),
        "yolo_stride": _iget("SWAY_YOLO_DETECTION_STRIDE", YOLO_DETECTION_STRIDE),
        "stitch_max_gap": _iget("SWAY_STITCH_MAX_FRAME_GAP", STITCH_MAX_FRAME_GAP),
        "stitch_radius_bbox_frac": _fget("SWAY_STITCH_RADIUS_BBOX_FRAC", STITCH_RADIUS_BBOX_FRAC),
        "stitch_max_pixel_radius": _fget("SWAY_STITCH_MAX_PIXEL_RADIUS", STITCH_MAX_PIXEL_RADIUS),
        "stitch_predicted_radius_frac": _fget(
            "SWAY_STITCH_PREDICTED_RADIUS_FRAC", STITCH_PREDICTED_RADIUS_FRAC
        ),
        "short_gap_frames": _iget("SWAY_SHORT_GAP_FRAMES", SHORT_GAP_FRAMES),
        "coalescence_iou": _fget("SWAY_COALESCENCE_IOU_THRESH", 0.70),
        "coalescence_consecutive": _iget("SWAY_COALESCENCE_CONSECUTIVE_FRAMES", 8),
        "dormant_max_gap": _iget("SWAY_DORMANT_MAX_GAP", 150),
    }


def apply_post_track_stitching(
    raw_tracks: Dict[int, List[Any]],
    total_frames: int,
    *,
    ystride: Optional[int] = None,
) -> Dict[int, List[Any]]:
    """
    Doc Phase 3 — dormant registry, fragment stitch, coalescence, complementary merge,
    coexisting merge, stride gap fill. Runs after per-frame detection + tracking (Phases 1–2).
    """
    tr = load_tracking_runtime()
    if ystride is None:
        ystride = int(tr["yolo_stride"])
    raw_tracks = _apply_dormant_and_global(raw_tracks, total_frames)
    raw_tracks = stitch_fragmented_tracks(
        raw_tracks,
        total_frames,
        max_frame_gap=int(tr["stitch_max_gap"]),
        radius_bbox_frac=float(tr["stitch_radius_bbox_frac"]),
        predicted_radius_frac=float(tr["stitch_predicted_radius_frac"]),
        fallback_radius=float(tr["stitch_max_pixel_radius"]),
        short_gap_frames=int(tr["short_gap_frames"]),
    )
    raw_tracks = coalescence_deduplicate(
        raw_tracks,
        iou_thresh=float(tr["coalescence_iou"]),
        consecutive_frames=int(tr["coalescence_consecutive"]),
    )
    raw_tracks = merge_complementary_tracks(raw_tracks)
    raw_tracks = merge_coexisting_fragments(raw_tracks)
    _fill_stride_gaps(raw_tracks, ystride)
    return raw_tracks


def _use_boxmot() -> bool:
    """BoxMOT Deep OC-SORT is the default tracker path. Set SWAY_USE_BOXMOT=0 to use Ultralytics BoT-SORT."""
    v = os.environ.get("SWAY_USE_BOXMOT", "").strip().lower()
    return v not in ("0", "false", "no", "off")


def _use_global_link() -> bool:
    return os.environ.get("SWAY_GLOBAL_LINK", "").lower() in ("1", "true", "yes")


def diou_nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.6,
) -> np.ndarray:
    """DIoU-NMS: suppress overlaps penalized by center distance (torchvision box_iou + distance term)."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    device = torch.device("cpu")
    b = torch.tensor(boxes, dtype=torch.float32, device=device)
    s = torch.tensor(scores, dtype=torch.float32, device=device)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    order = torch.argsort(s, descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = box_iou(b[i : i + 1], b[rest])[0]
        enc_x1 = torch.minimum(x1[i], x1[rest])
        enc_y1 = torch.minimum(y1[i], y1[rest])
        enc_x2 = torch.maximum(x2[i], x2[rest])
        enc_y2 = torch.maximum(y2[i], y2[rest])
        enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7
        center_dist_sq = (cx[i] - cx[rest]) ** 2 + (cy[i] - cy[rest]) ** 2
        diou = iou - center_dist_sq / enc_diag_sq
        order = rest[diou < iou_threshold]
    return np.array(keep, dtype=np.int64)


# Pre-tracker: after DIoU-NMS (when used), drop near-duplicate boxes (high IoU, same person).
# 0.50 was chosen to suppress duplicate / hallucinated person boxes; 0.90 was too loose.
_PRETRACK_CLASSICAL_NMS_IOU = 0.50


def _pretrack_classical_nms_iou() -> float:
    v = os.environ.get("SWAY_PRETRACK_NMS_IOU", "").strip()
    if not v:
        return float(_PRETRACK_CLASSICAL_NMS_IOU)
    try:
        return float(v)
    except ValueError:
        return float(_PRETRACK_CLASSICAL_NMS_IOU)


def _deepocsort_extra_from_env() -> Dict[str, Any]:
    """BoxMOT DeepOcSort knobs from env (Pipeline Lab + CLI experiments)."""
    max_age = 150
    try:
        max_age = int(os.environ.get("SWAY_BOXMOT_MAX_AGE", "").strip() or "150")
    except ValueError:
        pass
    iou_thr = 0.3
    try:
        iou_thr = float(os.environ.get("SWAY_BOXMOT_MATCH_THRESH", "").strip() or "0.3")
    except ValueError:
        pass
    reid_on = os.environ.get("SWAY_BOXMOT_REID_ON", "").strip().lower() in ("1", "true", "yes")
    raw_asso = (os.environ.get("SWAY_BOXMOT_ASSOC_METRIC", "iou").strip().lower() or "iou")
    if raw_asso in ("giou",):
        asso = "giou"
    elif raw_asso in ("diou", "ciou"):
        asso = raw_asso
    else:
        asso = "iou"
    return {
        "max_age": max_age,
        "iou_threshold": iou_thr,
        "embedding_off": not reid_on,
        "asso_func": asso,
    }


def classical_nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = _PRETRACK_CLASSICAL_NMS_IOU,
) -> np.ndarray:
    """Score-sorted NMS on plain IoU — suppresses duplicate / hallucinated boxes after YOLO (and after DIoU when used)."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    device = torch.device("cpu")
    b = torch.tensor(boxes, dtype=torch.float32, device=device)
    s = torch.tensor(scores, dtype=torch.float32, device=device)
    order = torch.argsort(s, descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = box_iou(b[i : i + 1], b[rest])[0]
        order = rest[iou < iou_thresh]
    return np.array(keep, dtype=np.int64)


def _resolve_boxmot_reid_weights() -> Path:
    env = os.environ.get("SWAY_BOXMOT_REID_WEIGHTS", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    repo = Path(__file__).resolve().parent.parent
    cand = repo / "models" / "osnet_x0_25_msmt17.pt"
    if cand.is_file():
        return cand
    try:
        from boxmot.utils import WEIGHTS

        return WEIGHTS / "osnet_x0_25_msmt17.pt"
    except Exception:
        return repo / "models" / "osnet_x0_25_msmt17.pt"


def _apply_dormant_and_global(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    total_frames: int,
) -> Dict[int, List[Tuple[int, Tuple, float]]]:
    from sway.dormant_tracks import apply_dormant_merges
    from sway.global_track_link import maybe_global_stitch

    tr_rt = load_tracking_runtime()
    raw_tracks = apply_dormant_merges(
        raw_tracks, total_frames, max_gap=int(tr_rt["dormant_max_gap"])
    )
    if _use_global_link():
        raw_tracks = maybe_global_stitch(raw_tracks, total_frames)
    return raw_tracks


def _env_offline() -> bool:
    for key in ("SWAY_OFFLINE", "YOLO_OFFLINE", "ULTRALYTICS_OFFLINE"):
        if str(os.environ.get(key, "")).lower() in ("1", "true", "yes"):
            return True
    return False


# Pipeline Lab enum / SWAY_YOLO_WEIGHTS token → Ultralytics hub id or local .pt basename
# (yolo11* kept for legacy env strings; Lab UI only exposes YOLO26 + optional YOLO11 L/X fallback.)
_SWAY_YOLO_WEIGHTS_ALIASES: Dict[str, str] = {
    "yolov11n": "yolo11n.pt",
    "yolov11s": "yolo11s.pt",
    "yolov11m": "yolo11m.pt",
    "yolov11l": "yolo11l.pt",
    "yolov11x": "yolo11x.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
    "yolo26l_dancetrack": "yolo26l_dancetrack.pt",
    "yolo26x_dancetrack": "yolo26x_dancetrack.pt",
}


def resolve_yolo_model_path() -> str:
    """
    Path for ultralytics.YOLO(). Prefer on-disk weights so the pipeline can run
    air-gapped after prefetch or manual copy.

    If SWAY_YOLO_WEIGHTS is a filesystem path to a .pt, that file wins. If it is a
    Pipeline Lab enum token (e.g. yolo26l, yolov11x), it maps to a hub .pt name.

    When SWAY_YOLO_WEIGHTS is unset:
    1) On-disk .pt in models/, repo root, cwd (YOLO26 first, e.g. yolo26l.pt then yolo26l_dancetrack.pt, then YOLO11).
    2) Core ML yolo11l/yolo11m .mlpackage if present (legacy Apple path).
    3) Hub fallback yolo26l.pt.
    """
    repo = Path(__file__).resolve().parent.parent
    models_dir = repo / "models"
    cwd = Path.cwd()
    bases = (models_dir, repo, cwd)

    env_raw = os.environ.get("SWAY_YOLO_WEIGHTS")
    if env_raw:
        env_s = env_raw.strip()
        p = Path(env_s).expanduser()
        if p.is_file():
            return str(p.resolve())
        key = env_s.lower()
        mapped = _SWAY_YOLO_WEIGHTS_ALIASES.get(key)
        if mapped:
            basename = Path(mapped).name
            for base in bases:
                hit = base / basename
                if hit.is_file():
                    return str(hit.resolve())
            return mapped
        if _env_offline():
            raise FileNotFoundError(f"SWAY_YOLO_WEIGHTS is set but file not found: {env_raw}")
        return env_s

    pt_priority = (
        "yolo26l.pt",
        "yolo26l_dancetrack.pt",
        "yolo11x_dancetrack.pt",
        "yolo26m.pt",
        "yolo26x.pt",
        "yolo26s.pt",
        "yolo11l.pt",
        "yolo11m.pt",
        "yolo11x.pt",
        "yolo11n.pt",
        "yolo11x_dancetrack.pt",
    )
    for name in pt_priority:
        for base in bases:
            p = base / name
            if p.is_file():
                return str(p.resolve())

    for base in (models_dir, repo):
        for name in ("yolo11l.mlpackage", "yolo11m.mlpackage"):
            p = base / name
            if p.exists():
                return str(p.resolve())
    for c in (
        cwd / "yolo11l.mlpackage",
        cwd / "yolo11m.mlpackage",
        models_dir / "yolo11l.mlpackage",
        models_dir / "yolo11m.mlpackage",
    ):
        if c.exists():
            return str(c.resolve())

    if _env_offline():
        raise FileNotFoundError(
            "Offline mode: no YOLO weights found. While online, run:\n"
            "  python prefetch_models.py\n"
            "Or place yolo26l.pt in repo or models/, or set SWAY_YOLO_WEIGHTS."
        )
    return "yolo26l.pt"


def _yolo26_series_weights(model_path: str) -> bool:
    """True if resolved weights path or hub id is YOLO26 (l/s/x or fine-tunes with yolo26 in the name)."""
    s = str(model_path).lower().replace("\\", "/")
    return "yolo26" in s


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
        sorted_entries = sorted(entries, key=lambda e: coerce_observation(e).frame_idx)
        new_entries = []
        for i, entry in enumerate(sorted_entries):
            obs = coerce_observation(entry)
            f, box = obs.frame_idx, obs.bbox
            new_entries.append(
                TrackObservation(
                    f, tuple(box), float(obs.conf), obs.is_sam_refined, obs.segmentation_mask
                )
            )
            if i + 1 < len(sorted_entries):
                next_obs = coerce_observation(sorted_entries[i + 1])
                next_f, next_box = next_obs.frame_idx, next_obs.bbox
                for gap_f in range(f + 1, next_f):
                    t = (gap_f - f) / (next_f - f)
                    interp_box = _interpolate_box(box, next_box, t)
                    new_entries.append(
                        TrackObservation(gap_f, tuple(interp_box), 0.5, False, None)
                    )
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
        entries.append(TrackObservation(f, tuple(box), float(conf), False, None))
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
    max_speed_bbox_frac: float = 0.25,
    short_gap_frames: int = SHORT_GAP_FRAMES,
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
                h_b = _bbox_height(info_b["first_box"])

                # V3.7: Reject stitch when bbox sizes differ drastically (head vs full body).
                # Prevents merging audience head (ID 58) with late-entrant dancer.
                if h_a > 0 and h_b > 0:
                    ratio = max(h_a, h_b) / min(h_a, h_b)
                    if ratio > 1.6:
                        continue

                short_gap = gap <= short_gap_frames

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

                if (dist_last / gap) > max(h_a * max_speed_bbox_frac, 30.0):
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
    iou_thresh: float = 0.85,
    consecutive_frames: int = 15,
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
    dead_ids: Set[int] = set()  # V3.8: Accumulate across frames (ghost flew away = kill on reset)
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
        for pair in list(overlap_counts.keys()):
            if pair not in current_overlaps:
                # V3.8: Before reset — if they ever overlapped N+ frames, kill younger (ghost flew away)
                count = overlap_counts[pair]
                if count >= consecutive_frames:
                    tid1, tid2 = pair
                    if track_age[tid1] >= track_age[tid2]:
                        dead_ids.add(tid2)
                    else:
                        dead_ids.add(tid1)
                del overlap_counts[pair]
                
        for pair in current_overlaps:
            overlap_counts[pair] = overlap_counts.get(pair, 0) + 1

    # Find IDs to kill (pairs still overlapping at end)
    for (tid1, tid2), count in overlap_counts.items():
        if count >= consecutive_frames:
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
    max_speed_bbox_frac: float = 0.25,
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
                prev_fidx = None
                for entry in all_entries:
                    fidx, box, _ = entry
                    owner = tid_a if fidx in frame_sets[tid_a] else tid_b
                    if prev_owner is not None and owner != prev_owner:
                        boundary_count += 1
                        h_prev = _bbox_height(prev_box)
                        h_cur = _bbox_height(box)
                        h = max(h_prev, h_cur, 1.0)
                        if h_prev > 0 and h_cur > 0:
                            ratio = max(h_prev, h_cur) / min(h_prev, h_cur)
                            if ratio > 1.6:
                                boundaries_ok = False
                                break
                        cx1, cy1 = _box_center(prev_box)
                        cx2, cy2 = _box_center(box)
                        dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                        gap = max(1, fidx - prev_fidx)
                        speed = dist / gap
                        if dist > max_center_dist_frac * h or speed > max(h * max_speed_bbox_frac, 30.0):
                            boundaries_ok = False
                            break
                    prev_owner = owner
                    prev_box = box
                    prev_fidx = fidx

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


def merge_coexisting_fragments(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    max_center_dist_frac: float = 0.5,
    min_overlap_frames: int = 5,
    min_proximity_ratio: float = 0.6,
) -> Dict[int, List[Tuple[int, Tuple, float]]]:
    """
    Merge two tracks that COEXIST (same frames) when their bbox centers are very close.
    Catches BoT-SORT assigning two IDs to the same person (e.g. "directly above" duplicates).

    Criteria:
      1. Tracks overlap in time (share at least min_overlap_frames).
      2. For at least min_proximity_ratio of overlapping frames, centers are within
         max_center_dist_frac * bbox_height of each other.
    Keeps the longer track, merges the shorter into it.
    Modifies raw_tracks in place.
    """
    if len(raw_tracks) < 2:
        return raw_tracks

    frame_sets: Dict[int, set] = {}
    sorted_entries: Dict[int, list] = {}
    for tid in raw_tracks:
        entries = sorted(raw_tracks[tid], key=lambda e: e[0])
        sorted_entries[tid] = entries
        frame_sets[tid] = {e[0] for e in entries}

    frame_to_entries: Dict[int, Dict[int, Tuple]] = {}
    for tid, entries in sorted_entries.items():
        for f, box, conf in entries:
            if f not in frame_to_entries:
                frame_to_entries[f] = {}
            frame_to_entries[f][tid] = box

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

                overlap = frame_sets[tid_a] & frame_sets[tid_b]
                if len(overlap) < min_overlap_frames:
                    continue

                proximity_count = 0
                height_ratios = []
                iou_sum, iou_n = 0.0, 0
                for f in overlap:
                    if tid_a not in frame_to_entries.get(f, {}) or tid_b not in frame_to_entries.get(f, {}):
                        continue
                    box_a = frame_to_entries[f][tid_a]
                    box_b = frame_to_entries[f][tid_b]
                    cx_a, cy_a = _box_center(box_a)
                    cx_b, cy_b = _box_center(box_b)
                    h_a, h_b = _bbox_height(box_a), _bbox_height(box_b)
                    h = max(h_a, h_b, 1.0)
                    dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
                    if dist <= max_center_dist_frac * h:
                        proximity_count += 1
                    if h_a > 0 and h_b > 0:
                        height_ratios.append(max(h_a, h_b) / min(h_a, h_b))
                    iou_sum += _compute_iou(box_a, box_b)
                    iou_n += 1

                if proximity_count < min_proximity_ratio * len(overlap):
                    continue
                if height_ratios and np.median(height_ratios) > 1.6:
                    continue  # V3.7: Don't merge head with full-body (different people)
                # V3.7: Require substantial IoU — "same person, two IDs" = overlapping boxes.
                # Side-by-side dancers = low IoU; merging them loses real people.
                if iou_n >= 3:
                    mean_iou = iou_sum / iou_n
                    if mean_iou < 0.25:
                        continue

                keep = tid_a if len(sorted_entries[tid_a]) >= len(sorted_entries[tid_b]) else tid_b
                kill = tid_b if keep == tid_a else tid_a

                keep_entries = {e[0]: e for e in raw_tracks[keep]}
                for e in raw_tracks[kill]:
                    if e[0] not in keep_entries:
                        keep_entries[e[0]] = e
                merged = sorted(keep_entries.values(), key=lambda e: e[0])

                raw_tracks[keep] = merged
                del raw_tracks[kill]
                sorted_entries[keep] = merged
                frame_sets[keep] = {e[0] for e in merged}
                del sorted_entries[kill]
                del frame_sets[kill]
                # Rebuild frame_to_entries for remaining tracks
                frame_to_entries.clear()
                for t in raw_tracks:
                    for f, box, conf in raw_tracks[t]:
                        if f not in frame_to_entries:
                            frame_to_entries[f] = {}
                        frame_to_entries[f][t] = box
                changed = True
                break

    return raw_tracks


def _get_tracker_config() -> str:
    """Return tracker config path (single source: config/botsort.yaml)."""
    env = os.environ.get("SWAY_TRACKER_YAML", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return str(p.resolve())
    repo = Path(__file__).resolve().parent.parent
    p = repo / "config" / "botsort.yaml"
    if p.is_file():
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


def _run_tracking_boxmot_diou(video_path: str) -> Tuple[
    Dict[int, List[Any]],
    int,
    float,
    Optional[List[Tuple[int, np.ndarray]]],
    float,
    int,
    int,
    int,
]:
    """Phases 1–2: YOLO predict + pre-track NMS + BoxMOT Deep OC-SORT (no post-track stitch)."""
    from sway.boxmot_compat import apply_boxmot_kf_unfreeze_guard

    apply_boxmot_kf_unfreeze_guard()
    from boxmot import DeepOcSort

    tr = load_tracking_runtime()
    yconf = float(tr["yolo_conf"])
    ystride = int(tr["yolo_stride"])
    base_detect = int(tr["detect_size"])

    model_path = resolve_yolo_model_path()
    print(f"Loading detection model: {model_path} (BoxMOT Deep OC-SORT path)")
    if _yolo26_series_weights(model_path):
        print("  Pre-track: DIoU-NMS off for YOLO26 weights (classical IoU-NMS only).")
    model = YOLO(model_path)
    print("  YOLO weights loaded; building DeepOcSort tracker…", flush=True)
    reid_w = _resolve_boxmot_reid_weights()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _doc_kw = _deepocsort_extra_from_env()
    tracker = DeepOcSort(
        reid_weights=reid_w,
        device=dev,
        half=bool(dev.type == "cuda"),
        det_thresh=yconf,
        max_age=int(_doc_kw["max_age"]),
        min_hits=2,
        iou_threshold=float(_doc_kw["iou_threshold"]),
        asso_func=str(_doc_kw["asso_func"]),
        embedding_off=bool(_doc_kw["embedding_off"]),
    )
    hybrid_cfg = load_hybrid_sam_config()
    hybrid_refiner: Optional[HybridSamRefiner] = None
    if hybrid_cfg["enabled"]:
        hybrid_refiner = HybridSamRefiner(hybrid_cfg)
        print(
            f"  Hybrid SAM overlap refiner ON (IoU≥{hybrid_cfg['iou_trigger']}, "
            f"min_dets={hybrid_cfg['min_dets']}, weights={hybrid_cfg['weights']})",
            flush=True,
        )
    else:
        print("  Hybrid SAM overlap refiner OFF.", flush=True)
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]] = {}
    total_frames = 0
    native_fps = 30.0
    frame_width = 1920
    frame_height = 1080
    max_dancers_last_chunk = 0
    current_detect_size = base_detect
    yolo_infer_count = 0
    if dev.type == "cpu":
        print(
            "  Note: CUDA not available — Phases 1–2 (YOLO + DeepOcSort) on CPU can take "
            "**many minutes** per video with little console output between YOLO steps.",
            flush=True,
        )

    chunk_idx = -1
    for chunk_frames, _chunk_start, nfps, w_f, h_f in _iter_video_chunks(video_path, tr["chunk_size"]):
        chunk_idx += 1
        if chunk_idx == 0:
            print(
                f"  Video streaming: first chunk has {len(chunk_frames)} frames "
                f"(chunk_size={tr['chunk_size']}), ~{nfps:.2f} fps, {w_f}x{h_f}, "
                f"YOLO stride={ystride}, base detect {base_detect}px.",
                flush=True,
            )
        native_fps = nfps
        frame_width = w_f
        frame_height = h_f
        if os.environ.get("SWAY_GROUP_VIDEO", "").lower() in ("1", "true", "yes"):
            current_detect_size = max(base_detect, 960)
        elif max_dancers_last_chunk > 4:
            current_detect_size = max(base_detect, 960)
        else:
            current_detect_size = base_detect
        max_dancers_this_chunk = 0

        for frame_idx, frame in chunk_frames:
            if frame_idx % ystride != 0:
                continue
            h_fr, w_fr = frame.shape[:2]
            frame_low = cv2.resize(frame, (current_detect_size, current_detect_size))
            frame_low_rgb = frame_low[:, :, ::-1]
            scale_x = w_fr / current_detect_size
            scale_y = h_fr / current_detect_size

            yolo_infer_count += 1
            if yolo_infer_count == 1:
                print(
                    f"  First YOLO inference (frame {frame_idx}, letterbox {current_detect_size})…",
                    flush=True,
                )
            elif yolo_infer_count % 25 == 0:
                print(
                    f"  YOLO+track progress: {yolo_infer_count} det frames, video frame {frame_idx}…",
                    flush=True,
                )

            res = model.predict(
                frame_low_rgb,
                classes=[0],
                conf=yconf,
                verbose=False,
            )
            r0 = res[0] if isinstance(res, list) else res
            if r0.boxes is None or len(r0.boxes) == 0:
                dets = np.empty((0, 6), dtype=np.float32)
            else:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                conf = r0.boxes.conf.cpu().numpy()
                xyxy[:, 0] *= scale_x
                xyxy[:, 1] *= scale_y
                xyxy[:, 2] *= scale_x
                xyxy[:, 3] *= scale_y
                # YOLO26 already applies strong NMS; extra DIoU here removed valid dancers / caused duplicates.
                if not _yolo26_series_weights(model_path):
                    diou_keep = diou_nms_indices(xyxy, conf, iou_threshold=0.7)
                    xyxy = xyxy[diou_keep]
                    conf = conf[diou_keep]
                keep2 = classical_nms_indices(xyxy, conf, iou_thresh=_pretrack_classical_nms_iou())
                xyxy = xyxy[keep2]
                conf = conf[keep2]
                # BoxMOT v16+ expects (x1,y1,x2,y2,conf,cls); class 0 = person
                cls0 = np.zeros((len(xyxy), 1), dtype=np.float32)
                dets = np.hstack([xyxy, conf.reshape(-1, 1), cls0]).astype(np.float32)

            if hybrid_refiner is not None:
                dets, hmeta = hybrid_refiner.refine_person_dets(frame, dets)
            else:
                hmeta = {}

            per_det_masks = hmeta.get("per_det_masks")
            if per_det_masks is None:
                per_det_masks = [None] * int(len(dets))

            out = tracker.update(dets, frame)
            valid_dancers_this_frame = 0
            if out is not None and len(out) > 0:
                out_arr = np.atleast_2d(out)
                mask_assign = assign_sam_masks_to_tracker_output(dets, out_arr, per_det_masks)
                for row_idx, row in enumerate(out_arr):
                    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    tid = int(row[4])
                    cf = float(row[5]) if len(row) > 5 else float(yconf)
                    if tid < 0:
                        continue
                    valid_dancers_this_frame += 1
                    box_hr = (x1, y1, x2, y2)
                    is_sam, msk = mask_assign[row_idx] if row_idx < len(mask_assign) else (False, None)
                    msk_fit = resize_mask_to_bbox(msk, box_hr) if msk is not None else None
                    has_mask = msk_fit is not None and bool(np.any(msk_fit))
                    if tid not in raw_tracks:
                        raw_tracks[tid] = []
                    raw_tracks[tid].append(
                        TrackObservation(
                            frame_idx,
                            box_hr,
                            cf,
                            is_sam_refined=bool(is_sam and has_mask),
                            segmentation_mask=msk_fit if has_mask else None,
                        )
                    )
            max_dancers_this_chunk = max(max_dancers_this_chunk, valid_dancers_this_frame)
            if frame_idx == 0 or frame_idx == 30:
                extra = ""
                if hybrid_refiner is not None:
                    extra = f", maxIoU={hmeta.get('max_iou', 0):.2f}"
                    if hmeta.get("used_sam"):
                        extra += " (SAM refined dets)"
                print(
                    f"  Frame {frame_idx}: {valid_dancers_this_frame} persons "
                    f"(BoxMOT, YOLO {current_detect_size}{extra})",
                    flush=True,
                )

        max_dancers_last_chunk = max_dancers_this_chunk
        total_frames += len(chunk_frames)
        del chunk_frames

    if hybrid_refiner is not None:
        print(f"  Hybrid SAM summary: {hybrid_refiner.summary()}")

    output_fps = native_fps
    return (
        raw_tracks,
        total_frames,
        float(output_fps),
        None,
        float(native_fps),
        frame_width,
        frame_height,
        int(ystride),
    )


def _run_tracking_botsort_pre_stitch(
    video_path: str,
) -> Tuple[
    Dict[int, List[Any]],
    int,
    float,
    Optional[List[Tuple[int, np.ndarray]]],
    float,
    int,
    int,
    int,
]:
    """Phases 1–2: YOLO track() + BoT-SORT (no post-track stitch)."""
    tr = load_tracking_runtime()
    yconf = float(tr["yolo_conf"])
    ystride = int(tr["yolo_stride"])
    base_detect = int(tr["detect_size"])

    model_path = resolve_yolo_model_path()
    print(f"Loading detection model: {model_path}")
    model = YOLO(model_path)
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]] = {}
    total_frames = 0
    native_fps = 30.0
    frame_width = 1920
    frame_height = 1080

    tracker_cfg = _get_tracker_config()

    max_dancers_last_chunk = 0
    current_detect_size = base_detect

    for chunk_frames, _chunk_start, nfps, w_f, h_f in _iter_video_chunks(video_path, tr["chunk_size"]):
        native_fps = nfps
        frame_width = w_f
        frame_height = h_f

        if os.environ.get("SWAY_GROUP_VIDEO", "").lower() in ("1", "true", "yes"):
            current_detect_size = max(base_detect, 960)
        elif max_dancers_last_chunk > 4:
            current_detect_size = max(base_detect, 960)
        else:
            current_detect_size = base_detect

        max_dancers_this_chunk = 0

        for frame_idx, frame in chunk_frames:
            if frame_idx % ystride != 0:
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
                conf=yconf,
                iou=0.5,
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
                raw_tracks[tid].append(
                    TrackObservation(frame_idx, box_hr, float(conf), False, None)
                )

            max_dancers_this_chunk = max(max_dancers_this_chunk, valid_dancers_this_frame)

            if frame_idx == 0 or frame_idx == 30:
                n = len([t for t in track_ids if t >= 0])
                print(f"  Frame {frame_idx}: {n} persons (YOLO resol: {current_detect_size})")

        max_dancers_last_chunk = max_dancers_this_chunk
        total_frames += len(chunk_frames)
        del chunk_frames

    output_fps = native_fps
    return (
        raw_tracks,
        total_frames,
        float(output_fps),
        None,
        float(native_fps),
        frame_width,
        frame_height,
        int(ystride),
    )


def run_tracking_before_post_stitch(
    video_path: str,
) -> Tuple[
    Dict[int, List[Any]],
    int,
    float,
    Optional[List[Tuple[int, np.ndarray]]],
    float,
    int,
    int,
    int,
]:
    """Phases 1–2 only: streaming YOLO + tracker; caller runs apply_post_track_stitching for Phase 3."""
    if _use_boxmot():
        return _run_tracking_boxmot_diou(video_path)
    return _run_tracking_botsort_pre_stitch(video_path)


def run_tracking(
    video_path: str,
) -> Tuple[Dict[int, List[Any]], int, float, Optional[List[Tuple[int, np.ndarray]]], float, int, int]:
    """
    Phases 1–3 in one call: run_tracking_before_post_stitch + apply_post_track_stitching.

    - Default tracker: BoxMOT Deep OC-SORT (SWAY_USE_BOXMOT unset). Set SWAY_USE_BOXMOT=0 for Ultralytics BoT-SORT.
    - Streaming 300-frame chunks; native FPS (stride via SWAY_YOLO_DETECTION_STRIDE).

    Returns:
        raw_tracks, total_frames, output_fps, frames_list (None), native_fps, frame_width, frame_height
    """
    raw, tf, ofps, fl, nf, fw, fh, ys = run_tracking_before_post_stitch(video_path)
    raw = apply_post_track_stitching(raw, tf, ystride=ys)
    return raw, tf, ofps, fl, nf, fw, fh


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
