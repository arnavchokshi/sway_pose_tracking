"""
Detection & Tracking Module — YOLO26l + BoxMOT Deep OC-SORT by default (V3.4)

Hybrid SAM (default on): SWAY_HYBRID_SAM_OVERLAP=0 disables. When enabled, Ultralytics SAM2 runs on frames where
person boxes overlap heavily, tightening xyxy before DeepOCSORT (see hybrid_sam_refiner.py).

V3.0: Streaming 300-frame chunks, native FPS; default detector YOLO26l (.pt under models/ or hub).
V3.3: YOLO runs on every frame for better dancer detection.
V3.4: Relative stitch radius (0.5x bbox height) + velocity-consistency check.
YOLO26 weights: pre-track DIoU-NMS is skipped (Ultralytics NMS + classical IoU-NMS at 0.50 only).

Double-Layer Tracking: Base tracker + OKS crossover refinement (see crossover.py)
handles dense overlaps when IoU > 0.6.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

from sway.hybrid_sam_refiner import HybridSamRefiner, load_hybrid_sam_config
from sway.interp_utils import gsi_interp_scalar as _gsi_interp_scalar
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
# Odd frames filled via interpolation (linear default; optional GSI — see SWAY_BOX_INTERP_MODE).
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


def yolo_infer_batch_size() -> int:
    """
    Ultralytics YOLO ``predict`` batch size on the BoxMOT path (Phases 1–2).
    Default 1 = one letterboxed frame per GPU call. Larger values amortize host/GPU overhead;
    tune down if VRAM is tight. Ignored on the BoT-SORT path (``model.track`` is per-frame).
    Env: SWAY_YOLO_INFER_BATCH
    """
    v = os.environ.get("SWAY_YOLO_INFER_BATCH", "").strip()
    if not v:
        return 1
    try:
        return max(1, min(int(v), 32))
    except ValueError:
        return 1


def yolo_half_env_requested() -> bool:
    """True when user asked for FP16 YOLO inference (CUDA only at runtime). Env: SWAY_YOLO_HALF"""
    v = os.environ.get("SWAY_YOLO_HALF", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def yolo_predict_use_half() -> bool:
    """FP16 YOLO ``predict``/``track`` only when requested and CUDA is available (default: off)."""
    return yolo_half_env_requested() and torch.cuda.is_available()


def resolve_yolo_inference_weights() -> str:
    """
    Weights passed to ``ultralytics.YOLO()`` for Phases 1–2.

    **TensorRT**: if ``SWAY_YOLO_ENGINE`` is set to a path, that file **must** exist or this raises
    (clear failure instead of silent .pt fallback). Unset ``SWAY_YOLO_ENGINE`` to use ``.pt`` weights.
    Export: ``python -m tools.export_models --tensorrt``.
    """
    eng = os.environ.get("SWAY_YOLO_ENGINE", "").strip()
    if eng:
        p = Path(eng).expanduser()
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(
            f"SWAY_YOLO_ENGINE is set but file not found: {eng}\n"
            "Export a TensorRT engine for this GPU, e.g.:\n"
            "  python -m tools.export_models --tensorrt --device 0\n"
            "Or unset SWAY_YOLO_ENGINE to use .pt weights (SWAY_YOLO_WEIGHTS / default)."
        )
    return resolve_yolo_model_path()


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
        "yolo_infer_batch": yolo_infer_batch_size(),
        "yolo_predict_half": yolo_predict_use_half(),
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
        # Optional gap fill: linear (default) vs Gaussian-process-smoothed (RBF, 2-anchor).
        "box_interp_mode": os.environ.get("SWAY_BOX_INTERP_MODE", "linear").strip().lower() or "linear",
        "gsi_lengthscale": _fget("SWAY_GSI_LENGTHSCALE", 0.35),
    }


def apply_post_track_stitching(
    raw_tracks: Dict[int, List[Any]],
    total_frames: int,
    *,
    ystride: Optional[int] = None,
    video_path: Optional[str] = None,
    native_fps: float = 30.0,
) -> Dict[int, List[Any]]:
    """
    Doc Phase 3 — dormant registry, fragment stitch, coalescence, complementary merge,
    coexisting merge, stride gap fill. Runs after per-frame detection + tracking (Phases 1–2).

    When ``SWAY_PHASE13_MODE=dancer_registry`` and ``video_path`` is set, runs appearance-based
    dormant relinking after motion dormant (see ``sway.dancer_registry_pipeline``).
    """
    tr = load_tracking_runtime()
    if ystride is None:
        ystride = int(tr["yolo_stride"])
    raw_tracks = _apply_dormant_and_global(raw_tracks, total_frames)
    try:
        from sway.dancer_registry_pipeline import (
            apply_dancer_registry_appearance_dormant,
            phase13_dancer_registry_enabled,
        )

        if (
            phase13_dancer_registry_enabled()
            and video_path
            and str(video_path).strip()
        ):
            raw_tracks = apply_dancer_registry_appearance_dormant(
                raw_tracks,
                str(video_path),
                int(total_frames),
                float(native_fps),
            )
    except Exception as ex:  # noqa: BLE001
        print(f"  [dancer_registry] Appearance dormant skipped: {ex}", flush=True)
    raw_tracks = stitch_fragmented_tracks(
        raw_tracks,
        total_frames,
        max_frame_gap=int(tr["stitch_max_gap"]),
        radius_bbox_frac=float(tr["stitch_radius_bbox_frac"]),
        predicted_radius_frac=float(tr["stitch_predicted_radius_frac"]),
        fallback_radius=float(tr["stitch_max_pixel_radius"]),
        short_gap_frames=int(tr["short_gap_frames"]),
        box_interp_mode=str(tr["box_interp_mode"]),
        gsi_lengthscale=float(tr["gsi_lengthscale"]),
    )
    raw_tracks = coalescence_deduplicate(
        raw_tracks,
        iou_thresh=float(tr["coalescence_iou"]),
        consecutive_frames=int(tr["coalescence_consecutive"]),
    )
    raw_tracks = merge_complementary_tracks(
        raw_tracks,
        box_interp_mode=str(tr["box_interp_mode"]),
        gsi_lengthscale=float(tr["gsi_lengthscale"]),
    )
    raw_tracks = merge_coexisting_fragments(raw_tracks)
    _fill_stride_gaps(
        raw_tracks,
        ystride,
        box_interp_mode=str(tr["box_interp_mode"]),
        gsi_lengthscale=float(tr["gsi_lengthscale"]),
    )
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
    elif raw_asso in ("eiou",):
        # BoxMOT has no EIoU; CIoU is the closest built-in (center distance + aspect ratio terms).
        asso = "ciou"
    else:
        asso = "iou"
    return {
        "max_age": max_age,
        "iou_threshold": iou_thr,
        "embedding_off": not reid_on,
        "asso_func": asso,
    }


def _normalize_boxmot_tracker_kind(raw: Optional[str]) -> str:
    """Map SWAY_BOXMOT_TRACKER / aliases to deepocsort | bytetrack | ocsort | strongsort."""
    s = (raw or "deepocsort").strip().lower().replace("-", "").replace("_", "")
    if s in ("deepocsort", "deepoc", "doc", "boxmot"):
        return "deepocsort"
    if s in ("bytetrack", "byte"):
        return "bytetrack"
    if s in ("ocsort", "oc"):
        return "ocsort"
    if s in ("strongsort", "strong"):
        return "strongsort"
    return "deepocsort"


def boxmot_tracker_kind_from_env() -> str:
    return _normalize_boxmot_tracker_kind(os.environ.get("SWAY_BOXMOT_TRACKER"))


def _fps_from_video_capture(cap: cv2.VideoCapture) -> float:
    """FPS from OpenCV metadata; sane fallback when missing or absurd."""
    try:
        raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except (TypeError, ValueError):
        return 30.0
    if raw <= 1e-3 or raw > 480.0:
        return 30.0
    return raw


def probe_video_fps(video_path: str) -> float:
    """Read container FPS without decoding frames (short-lived VideoCapture)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0
    try:
        return _fps_from_video_capture(cap)
    finally:
        cap.release()


def _create_boxmot_tracker(
    kind: str,
    yconf: float,
    dev: torch.device,
    reid_w: Path,
    doc_kw: Dict[str, Any],
    *,
    tracker_frame_rate: int = 30,
) -> Any:
    """Instantiate a BoxMOT tracker; ``kind`` from ``boxmot_tracker_kind_from_env()``."""
    from boxmot import ByteTrack, DeepOcSort, OcSort, StrongSort

    half_cuda = bool(dev.type == "cuda")
    tr_fps = max(1, int(tracker_frame_rate))

    if kind == "deepocsort":
        return DeepOcSort(
            reid_weights=reid_w,
            device=dev,
            half=half_cuda,
            det_thresh=yconf,
            max_age=int(doc_kw["max_age"]),
            min_hits=2,
            iou_threshold=float(doc_kw["iou_threshold"]),
            asso_func=str(doc_kw["asso_func"]),
            embedding_off=bool(doc_kw["embedding_off"]),
        )
    if kind == "bytetrack":
        bt_match = float(os.environ.get("SWAY_BYTETRACK_MATCH_THRESH", "").strip() or "0.8")
        bt_buffer = int(os.environ.get("SWAY_BYTETRACK_TRACK_BUFFER", "").strip() or "25")
        return ByteTrack(
            min_conf=max(0.01, float(yconf) * 0.25),
            track_thresh=float(yconf),
            match_thresh=bt_match,
            track_buffer=bt_buffer,
            frame_rate=tr_fps,
        )
    if kind == "ocsort":
        oc_use_byte = os.environ.get("SWAY_OCSORT_USE_BYTE", "1").strip().lower() not in ("0", "false", "no")
        return OcSort(
            min_conf=max(0.01, float(yconf) * 0.25),
            use_byte=oc_use_byte,
        )
    if kind == "strongsort":
        if not reid_w.is_file():
            raise FileNotFoundError(
                f"StrongSORT requires Re-ID weights on disk; missing: {reid_w}\n"
                "Run: python -m tools.prefetch_models\n"
                "Or set SWAY_BOXMOT_REID_WEIGHTS to an existing OSNet .pt file."
            )
        ss_cos = float(os.environ.get("SWAY_STRONGSORT_MAX_COS_DIST", "").strip() or "0.2")
        ss_iou = float(os.environ.get("SWAY_STRONGSORT_MAX_IOU_DIST", "").strip() or "0.7")
        ss_ninit = int(os.environ.get("SWAY_STRONGSORT_N_INIT", "").strip() or "3")
        ss_budget = int(os.environ.get("SWAY_STRONGSORT_NN_BUDGET", "").strip() or "100")
        return StrongSort(
            reid_weights=reid_w,
            device=dev,
            half=half_cuda,
            min_conf=max(0.01, float(yconf) * 0.25),
            max_cos_dist=ss_cos,
            max_iou_dist=ss_iou,
            max_age=int(doc_kw["max_age"]),
            n_init=ss_ninit,
            nn_budget=ss_budget,
        )
    return _create_boxmot_tracker(
        "deepocsort", yconf, dev, reid_w, doc_kw, tracker_frame_rate=tracker_frame_rate
    )


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
        # Try as a model name relative to models/ dir
        repo = Path(__file__).resolve().parent.parent
        cand_name = repo / "models" / env
        if cand_name.is_file():
            return cand_name
        # Try BoxMOT weights cache dir — auto-download will happen at tracker init
        try:
            from boxmot.utils import WEIGHTS
            return WEIGHTS / env
        except Exception:
            pass
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


# Pipeline Lab enum / SWAY_YOLO_WEIGHTS token → Ultralytics hub id or local .pt basename (YOLO26 only).
_SWAY_YOLO_WEIGHTS_ALIASES: Dict[str, str] = {
    "yolo26s": "yolo26s.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
    "yolo26l_dancetrack": "yolo26l_dancetrack.pt",
    "yolo26l_dancetrack_crowdhuman": "yolo26l_dancetrack_crowdhuman.pt",
    "yolo26x_dancetrack": "yolo26x_dancetrack.pt",
}


def resolve_yolo_model_path() -> str:
    """
    Path for ultralytics.YOLO(). Prefer on-disk weights so the pipeline can run
    air-gapped after prefetch or manual copy.

    If SWAY_YOLO_WEIGHTS is a filesystem path to a .pt, that file wins. If it is a
    Pipeline Lab enum token (e.g. yolo26l, yolo26x), it maps to a hub .pt name.

    When SWAY_YOLO_WEIGHTS is unset:
    1) On-disk YOLO26 .pt in models/, repo root, cwd (e.g. yolo26l.pt then yolo26l_dancetrack.pt, then s/x).
    2) Hub fallback yolo26l.pt.
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
        "yolo26m.pt",
        "yolo26x.pt",
        "yolo26x_dancetrack.pt",
        "yolo26s.pt",
    )
    for name in pt_priority:
        for base in bases:
            p = base / name
            if p.is_file():
                return str(p.resolve())

    if _env_offline():
        raise FileNotFoundError(
            "Offline mode: no YOLO weights found. While online, run:\n"
            "  python -m tools.prefetch_models\n"
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


def _resolve_interp_kwargs(
    box_interp_mode: Optional[str],
    gsi_lengthscale: Optional[float],
) -> Tuple[str, float]:
    if box_interp_mode is not None and gsi_lengthscale is not None:
        return str(box_interp_mode), float(gsi_lengthscale)
    tr = load_tracking_runtime()
    return str(tr["box_interp_mode"]), float(tr["gsi_lengthscale"])


def _interp_box_at_t(
    box_prev: Tuple[float, float, float, float],
    box_next: Tuple[float, float, float, float],
    t: float,
    *,
    mode: str,
    gsi_lengthscale: float,
) -> Tuple[float, float, float, float]:
    m = (mode or "linear").strip().lower()
    if m != "gsi":
        return _interpolate_box(box_prev, box_next, t)
    return tuple(
        _gsi_interp_scalar(t, float(box_prev[i]), float(box_next[i]), gsi_lengthscale)
        for i in range(4)
    )


def _fill_stride_gaps(
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]],
    stride: int,
    *,
    box_interp_mode: Optional[str] = None,
    gsi_lengthscale: Optional[float] = None,
) -> None:
    """
    Fill missing frame entries when YOLO runs only every Nth frame.
    Inserts interpolated boxes for skipped frames (linear by default; optional GSI). Modifies in place.
    """
    bim, gsl = _resolve_interp_kwargs(box_interp_mode, gsi_lengthscale)
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
                    interp_box = _interp_box_at_t(box, next_box, t, mode=bim, gsi_lengthscale=gsl)
                    new_entries.append(
                        TrackObservation(gap_f, tuple(interp_box), 0.5, False, None)
                    )
        raw_tracks[tid] = sorted(new_entries, key=lambda e: e[0])


def _interpolate_box_sequence(
    box_last: Tuple,
    box_first: Tuple,
    frame_last: int,
    frame_first: int,
    *,
    box_interp_mode: Optional[str] = None,
    gsi_lengthscale: Optional[float] = None,
) -> List[Tuple[int, Tuple, float]]:
    """Interpolate boxes between frame_last and frame_first (exclusive)."""
    bim, gsl = _resolve_interp_kwargs(box_interp_mode, gsi_lengthscale)
    if frame_first <= frame_last + 1:
        return []
    entries = []
    conf = 0.5  # Placeholder for interpolated frames
    for f in range(frame_last + 1, frame_first):
        t = (f - frame_last) / (frame_first - frame_last)
        box = _interp_box_at_t(box_last, box_first, t, mode=bim, gsi_lengthscale=gsl)
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
    *,
    box_interp_mode: Optional[str] = None,
    gsi_lengthscale: Optional[float] = None,
) -> Dict[int, List[Tuple[int, Tuple, float]]]:
    """
    V3.4: Stitch tracks that fragmented due to occlusion. Uses relative stitch
    radius (fraction of bbox height) instead of fixed pixel radius. Adds
    velocity-consistency check to prevent merging unrelated tracks.

    Modifies raw_tracks in place and returns it.
    """
    bim, gsl = _resolve_interp_kwargs(box_interp_mode, gsi_lengthscale)
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
                box_a_last,
                box_b_first,
                frame_a_last,
                entries_b[0][0],
                box_interp_mode=bim,
                gsi_lengthscale=gsl,
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
    *,
    box_interp_mode: Optional[str] = None,
    gsi_lengthscale: Optional[float] = None,
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
    bim, gsl = _resolve_interp_kwargs(box_interp_mode, gsi_lengthscale)
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
                            _interpolate_box_sequence(
                                merged[k][1],
                                merged[k + 1][1],
                                f_cur,
                                f_nxt,
                                box_interp_mode=bim,
                                gsi_lengthscale=gsl,
                            )
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
    native_fps = _fps_from_video_capture(cap)
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


def _run_tracking_boxmot_diou(
    video_path: str,
    lab_on_infer: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[
    Dict[int, List[Any]],
    int,
    float,
    Optional[List[Tuple[int, np.ndarray]]],
    float,
    int,
    int,
    int,
    Dict[str, Any],
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
]:
    """Phases 1–2: YOLO predict + pre-track NMS + BoxMOT Deep OC-SORT (no post-track stitch)."""
    from sway.boxmot_compat import apply_boxmot_kf_unfreeze_guard

    apply_boxmot_kf_unfreeze_guard()

    tr = load_tracking_runtime()
    yconf = float(tr["yolo_conf"])
    ystride = int(tr["yolo_stride"])
    yolo_bs = int(tr["yolo_infer_batch"])
    base_detect = int(tr["detect_size"])

    model_path = resolve_yolo_inference_weights()
    bmk = boxmot_tracker_kind_from_env()
    print(f"Loading detection model: {model_path} (BoxMOT path, tracker={bmk})")
    if str(model_path).lower().endswith(".engine"):
        print("  YOLO backend: TensorRT engine (SWAY_YOLO_ENGINE).", flush=True)
    if _yolo26_series_weights(model_path):
        print("  Pre-track: DIoU-NMS off for YOLO26 weights (classical IoU-NMS only).")
    y_half = yolo_predict_use_half()
    if y_half:
        print("  YOLO inference FP16: SWAY_YOLO_HALF=1 (CUDA).", flush=True)
    elif yolo_half_env_requested() and not torch.cuda.is_available():
        print("  Note: SWAY_YOLO_HALF=1 ignored (CUDA not available).", flush=True)
    model = YOLO(model_path)
    print(f"  YOLO weights loaded; building BoxMOT {bmk} tracker…", flush=True)
    reid_w = _resolve_boxmot_reid_weights()
    # BoxMOT parses torch.device("cuda") as the string "cuda" and sets CUDA_VISIBLE_DEVICES=cuda
    # (invalid). Use an explicit ordinal so DeepOcSort/ReID backends see device "0".
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _doc_kw = _deepocsort_extra_from_env()
    src_fps = probe_video_fps(video_path)
    tr_fps = max(1, int(round(src_fps)))
    print(
        f"  Video FPS (file metadata): {src_fps:.3f}"
        + (f"; ByteTrack frame_rate={tr_fps}" if bmk == "bytetrack" else ""),
        flush=True,
    )
    tracker = _create_boxmot_tracker(
        bmk, yconf, dev, reid_w, _doc_kw, tracker_frame_rate=tr_fps
    )
    hybrid_cfg = load_hybrid_sam_config()
    hybrid_refiner: Optional[HybridSamRefiner] = None
    if hybrid_cfg["enabled"]:
        hybrid_refiner = HybridSamRefiner(hybrid_cfg)
        wc = ", weak-cue SAM gate on" if hybrid_cfg.get("weak_cues") else ""
        print(
            f"  Hybrid SAM overlap refiner ON (IoU≥{hybrid_cfg['iou_trigger']}, "
            f"min_dets={hybrid_cfg['min_dets']}, weights={hybrid_cfg['weights']}{wc})",
            flush=True,
        )
    else:
        print("  Hybrid SAM overlap refiner OFF.", flush=True)

    handshake_state = None
    try:
        from sway.handshake_tracking import SwayHandshakeState, phase13_handshake_enabled

        if phase13_handshake_enabled():
            handshake_state = SwayHandshakeState()
            print(
                "  Sway Handshake: dancer registry + SAM mask↔ID verify "
                f"(IoU≥{float(hybrid_cfg.get('iou_trigger', 0.42)):.2f}).",
                flush=True,
            )
    except Exception:
        handshake_state = None

    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]] = {}
    # frame_idx -> list of (xyxy high-res, detector confidence); pre–track-association dets.
    phase1_dets_by_frame: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    # After DIoU (if any), before classical IoU-NMS — for live IoU tuning in Lab.
    phase1_pre_classical_by_frame: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    total_frames = 0
    native_fps = 30.0
    frame_width = 1920
    frame_height = 1080
    max_dancers_last_chunk = 0
    current_detect_size = base_detect
    yolo_infer_count = 0
    if dev.type == "cpu":
        print(
            f"  Note: CUDA not available — Phases 1–2 (YOLO + BoxMOT {bmk}) on CPU can take "
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
            if yolo_bs > 1:
                print(
                    f"  YOLO infer batch size={yolo_bs} (SWAY_YOLO_INFER_BATCH); "
                    "tracker still updated once per frame in order.",
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
        det_batch: List[Tuple[int, np.ndarray, float, float, np.ndarray]] = []

        def _flush_yolo_det_batch(
            batch: List[Tuple[int, np.ndarray, float, float, np.ndarray]],
        ) -> None:
            nonlocal yolo_infer_count, max_dancers_this_chunk
            if not batch:
                return
            imgs = [b[4] for b in batch]
            res_list = model.predict(
                imgs,
                classes=[0],
                conf=yconf,
                verbose=False,
                half=y_half,
            )
            if not isinstance(res_list, (list, tuple)):
                res_list = [res_list]
            if len(res_list) != len(batch):
                raise RuntimeError(
                    f"YOLO infer batch: got {len(res_list)} result(s) for {len(batch)} image(s)"
                )
            for (frame_idx, frame, scale_x, scale_y, _rgb), r0 in zip(batch, res_list):
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

                if r0.boxes is None or len(r0.boxes) == 0:
                    dets = np.empty((0, 6), dtype=np.float32)
                    phase1_pre_classical_by_frame[frame_idx] = []
                else:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    conf = r0.boxes.conf.cpu().numpy()
                    xyxy[:, 0] *= scale_x
                    xyxy[:, 1] *= scale_y
                    xyxy[:, 2] *= scale_x
                    xyxy[:, 3] *= scale_y
                    if not _yolo26_series_weights(model_path):
                        diou_keep = diou_nms_indices(xyxy, conf, iou_threshold=0.7)
                        xyxy = xyxy[diou_keep]
                        conf = conf[diou_keep]
                    pre_pairs: List[Tuple[Tuple[float, float, float, float], float]] = []
                    for i in range(len(xyxy)):
                        r = xyxy[i]
                        pre_pairs.append(((float(r[0]), float(r[1]), float(r[2]), float(r[3])), float(conf[i])))
                    phase1_pre_classical_by_frame[frame_idx] = pre_pairs
                    keep2 = classical_nms_indices(xyxy, conf, iou_thresh=_pretrack_classical_nms_iou())
                    xyxy = xyxy[keep2]
                    conf = conf[keep2]
                    cls0 = np.zeros((len(xyxy), 1), dtype=np.float32)
                    dets = np.hstack([xyxy, conf.reshape(-1, 1), cls0]).astype(np.float32)

                # Phase-1 preview / checkpoint: YOLO + NMS only (before hybrid SAM and tracker).
                yolo_only_pairs: List[Tuple[Tuple[float, float, float, float], float]] = []
                if dets is not None and len(dets) > 0:
                    for row in dets:
                        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                        cf = float(row[4])
                        yolo_only_pairs.append(((x1, y1, x2, y2), cf))
                phase1_dets_by_frame[frame_idx] = yolo_only_pairs

                if hybrid_refiner is not None:
                    dets, hmeta = hybrid_refiner.refine_person_dets(frame, dets)
                else:
                    hmeta = {}

                _hcfg = hybrid_cfg if hybrid_refiner is not None else load_hybrid_sam_config()
                if handshake_state is not None:
                    from sway.handshake_tracking import handshake_process_frame

                    dets, hmeta = handshake_process_frame(
                        handshake_state, frame, dets, hmeta, _hcfg
                    )

                sam2_roi_tuple: Optional[tuple] = None
                if hybrid_refiner is not None and hmeta.get("used_sam"):
                    fh, fw = frame.shape[:2]
                    rb = hmeta.get("roi_box")
                    if rb is not None and len(rb) >= 4:
                        sam2_roi_tuple = tuple(float(x) for x in rb[:4])
                    else:
                        sam2_roi_tuple = (0.0, 0.0, float(fw), float(fh))

                per_det_masks = hmeta.get("per_det_masks")
                if per_det_masks is None:
                    per_det_masks = [None] * int(len(dets))

                out = tracker.update(dets, frame)
                valid_dancers_this_frame = 0
                if handshake_state is not None and out is not None and len(out) > 0:
                    handshake_state.set_prev_tracker_out(np.atleast_2d(out))
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
                                sam2_input_roi_xyxy=sam2_roi_tuple,
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
                        f"(BoxMOT/{bmk}, YOLO {current_detect_size}{extra})",
                        flush=True,
                    )

                if lab_on_infer is not None and (
                    yolo_infer_count == 1 or yolo_infer_count % 25 == 0
                ):
                    lab_on_infer(frame_idx, yolo_infer_count, valid_dancers_this_frame)

        for frame_idx, frame in chunk_frames:
            if frame_idx % ystride != 0:
                continue
            h_fr, w_fr = frame.shape[:2]
            frame_low = cv2.resize(frame, (current_detect_size, current_detect_size))
            frame_low_rgb = frame_low[:, :, ::-1]
            scale_x = w_fr / current_detect_size
            scale_y = h_fr / current_detect_size
            det_batch.append((frame_idx, frame, scale_x, scale_y, frame_low_rgb))
            if len(det_batch) >= yolo_bs:
                _flush_yolo_det_batch(det_batch)
                det_batch.clear()

        _flush_yolo_det_batch(det_batch)

        max_dancers_last_chunk = max_dancers_this_chunk
        total_frames += len(chunk_frames)
        del chunk_frames

    hybrid_stats: Dict[str, Any] = {}
    if hybrid_refiner is not None:
        hybrid_stats = hybrid_refiner.summary()
        print(f"  Hybrid SAM summary: {hybrid_stats}")

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
        hybrid_stats,
        phase1_dets_by_frame,
        phase1_pre_classical_by_frame,
    )


def run_phase1_yolo_only_boxmot(
    video_path: str,
    lab_on_infer: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[
    int,
    float,
    float,
    int,
    int,
    int,
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
]:
    """
    YOLO person detection only (BoxMOT path): NMS-scaled boxes + confidences per frame, no hybrid SAM,
    no tracker. Used for ``after_phase_1`` checkpoints.
    """
    from sway.boxmot_compat import apply_boxmot_kf_unfreeze_guard

    apply_boxmot_kf_unfreeze_guard()

    tr = load_tracking_runtime()
    yconf = float(tr["yolo_conf"])
    ystride = int(tr["yolo_stride"])
    yolo_bs = int(tr["yolo_infer_batch"])
    base_detect = int(tr["detect_size"])

    model_path = resolve_yolo_inference_weights()
    bmk = boxmot_tracker_kind_from_env()
    print(f"[phase1-only] Loading detection model: {model_path} (BoxMOT path, tracker={bmk})")
    y_half = yolo_predict_use_half()
    model = YOLO(model_path)

    phase1_dets_by_frame: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    phase1_pre_classical_by_frame: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    total_frames = 0
    native_fps = 30.0
    frame_width = 1920
    frame_height = 1080
    max_dancers_last_chunk = 0
    current_detect_size = base_detect
    yolo_infer_count = 0

    chunk_idx = -1
    for chunk_frames, _chunk_start, nfps, w_f, h_f in _iter_video_chunks(video_path, tr["chunk_size"]):
        chunk_idx += 1
        if chunk_idx == 0:
            print(
                f"  [phase1-only] chunk_size={tr['chunk_size']}, YOLO stride={ystride}, base detect {base_detect}px.",
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
        det_batch: List[Tuple[int, np.ndarray, float, float, np.ndarray]] = []

        def _flush_yolo_only_batch(
            batch: List[Tuple[int, np.ndarray, float, float, np.ndarray]],
        ) -> None:
            nonlocal yolo_infer_count, max_dancers_this_chunk
            if not batch:
                return
            imgs = [b[4] for b in batch]
            res_list = model.predict(
                imgs,
                classes=[0],
                conf=yconf,
                verbose=False,
                half=y_half,
            )
            if not isinstance(res_list, (list, tuple)):
                res_list = [res_list]
            if len(res_list) != len(batch):
                raise RuntimeError(
                    f"YOLO infer batch: got {len(res_list)} result(s) for {len(batch)} image(s)"
                )
            for (frame_idx, frame, scale_x, scale_y, _rgb), r0 in zip(batch, res_list):
                yolo_infer_count += 1
                if r0.boxes is None or len(r0.boxes) == 0:
                    dets = np.empty((0, 6), dtype=np.float32)
                    phase1_pre_classical_by_frame[frame_idx] = []
                else:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    conf = r0.boxes.conf.cpu().numpy()
                    xyxy[:, 0] *= scale_x
                    xyxy[:, 1] *= scale_y
                    xyxy[:, 2] *= scale_x
                    xyxy[:, 3] *= scale_y
                    if not _yolo26_series_weights(model_path):
                        diou_keep = diou_nms_indices(xyxy, conf, iou_threshold=0.7)
                        xyxy = xyxy[diou_keep]
                        conf = conf[diou_keep]
                    pre_pairs_ph1: List[Tuple[Tuple[float, float, float, float], float]] = []
                    for i in range(len(xyxy)):
                        r = xyxy[i]
                        pre_pairs_ph1.append(
                            ((float(r[0]), float(r[1]), float(r[2]), float(r[3])), float(conf[i]))
                        )
                    phase1_pre_classical_by_frame[frame_idx] = pre_pairs_ph1
                    keep2 = classical_nms_indices(xyxy, conf, iou_thresh=_pretrack_classical_nms_iou())
                    xyxy = xyxy[keep2]
                    conf = conf[keep2]
                    cls0 = np.zeros((len(xyxy), 1), dtype=np.float32)
                    dets = np.hstack([xyxy, conf.reshape(-1, 1), cls0]).astype(np.float32)

                yolo_pairs: List[Tuple[Tuple[float, float, float, float], float]] = []
                if dets is not None and len(dets) > 0:
                    for row in dets:
                        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                        cf = float(row[4])
                        yolo_pairs.append(((x1, y1, x2, y2), cf))
                phase1_dets_by_frame[frame_idx] = yolo_pairs
                nd = len(yolo_pairs)
                max_dancers_this_chunk = max(max_dancers_this_chunk, nd)
                if lab_on_infer is not None and (yolo_infer_count == 1 or yolo_infer_count % 25 == 0):
                    lab_on_infer(frame_idx, yolo_infer_count, nd)

        for frame_idx, frame in chunk_frames:
            if frame_idx % ystride != 0:
                continue
            h_fr, w_fr = frame.shape[:2]
            frame_low = cv2.resize(frame, (current_detect_size, current_detect_size))
            frame_low_rgb = frame_low[:, :, ::-1]
            scale_x = w_fr / current_detect_size
            scale_y = h_fr / current_detect_size
            det_batch.append((frame_idx, frame, scale_x, scale_y, frame_low_rgb))
            if len(det_batch) >= yolo_bs:
                _flush_yolo_only_batch(det_batch)
                det_batch.clear()

        _flush_yolo_only_batch(det_batch)

        max_dancers_last_chunk = max_dancers_this_chunk
        total_frames += len(chunk_frames)
        del chunk_frames

    output_fps = native_fps
    return (
        total_frames,
        float(output_fps),
        float(native_fps),
        frame_width,
        frame_height,
        int(ystride),
        phase1_dets_by_frame,
        phase1_pre_classical_by_frame,
    )


def run_boxmot_tracking_from_yolo_dets(
    video_path: str,
    yolo_dets_by_frame: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    lab_on_infer: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[
    Dict[int, List[Any]],
    int,
    float,
    Optional[List[Tuple[int, np.ndarray]]],
    float,
    int,
    int,
    int,
    Dict[str, Any],
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
]:
    """
    Phases 2 (tracker + optional hybrid SAM) from a phase-1 YOLO checkpoint (BoxMOT path only).
    Replays the video with the same chunking; for each detection frame, uses stored YOLO boxes+conf,
    then hybrid SAM + ``tracker.update`` as in the normal pipeline.
    """
    from sway.boxmot_compat import apply_boxmot_kf_unfreeze_guard

    apply_boxmot_kf_unfreeze_guard()

    tr = load_tracking_runtime()
    yconf = float(tr["yolo_conf"])
    ystride = int(tr["yolo_stride"])
    yolo_bs = int(tr["yolo_infer_batch"])
    base_detect = int(tr["detect_size"])

    model_path = resolve_yolo_inference_weights()
    bmk = boxmot_tracker_kind_from_env()
    print(f"[resume phase1→track] {model_path} BoxMOT {bmk}; hybrid SAM + tracker from checkpoint dets.")
    reid_w = _resolve_boxmot_reid_weights()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _doc_kw = _deepocsort_extra_from_env()
    src_fps = probe_video_fps(video_path)
    tr_fps = max(1, int(round(src_fps)))
    tracker = _create_boxmot_tracker(
        bmk, yconf, dev, reid_w, _doc_kw, tracker_frame_rate=tr_fps
    )
    hybrid_cfg = load_hybrid_sam_config()
    hybrid_refiner: Optional[HybridSamRefiner] = None
    if hybrid_cfg["enabled"]:
        hybrid_refiner = HybridSamRefiner(hybrid_cfg)

    handshake_state = None
    try:
        from sway.handshake_tracking import SwayHandshakeState, phase13_handshake_enabled

        if phase13_handshake_enabled():
            handshake_state = SwayHandshakeState()
    except Exception:
        handshake_state = None

    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]] = {}
    phase1_dets_by_frame: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    total_frames = 0
    native_fps = 30.0
    frame_width = 1920
    frame_height = 1080
    max_dancers_last_chunk = 0
    current_detect_size = base_detect
    yolo_infer_count = 0

    chunk_idx = -1
    for chunk_frames, _chunk_start, nfps, w_f, h_f in _iter_video_chunks(video_path, tr["chunk_size"]):
        chunk_idx += 1
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
        det_batch: List[Tuple[int, np.ndarray, float, float, np.ndarray]] = []

        def _flush_track_from_checkpoint_batch(
            batch: List[Tuple[int, np.ndarray, float, float, np.ndarray]],
        ) -> None:
            nonlocal yolo_infer_count, max_dancers_this_chunk
            if not batch:
                return
            for (frame_idx, frame, scale_x, scale_y, _rgb) in batch:
                yolo_infer_count += 1
                pairs = yolo_dets_by_frame.get(frame_idx, [])
                if not pairs:
                    dets = np.empty((0, 6), dtype=np.float32)
                else:
                    rows = []
                    for (x1, y1, x2, y2), cf in pairs:
                        rows.append([x1, y1, x2, y2, cf, 0.0])
                    dets = np.array(rows, dtype=np.float32)

                yolo_pairs = list(pairs)
                phase1_dets_by_frame[frame_idx] = yolo_pairs

                hmeta: Dict[str, Any] = {}
                if hybrid_refiner is not None:
                    dets, hmeta = hybrid_refiner.refine_person_dets(frame, dets)
                else:
                    hmeta = {}

                _hcfg = hybrid_cfg if hybrid_refiner is not None else load_hybrid_sam_config()
                if handshake_state is not None:
                    from sway.handshake_tracking import handshake_process_frame

                    dets, hmeta = handshake_process_frame(
                        handshake_state, frame, dets, hmeta, _hcfg
                    )

                sam2_roi_tuple: Optional[tuple] = None
                if hybrid_refiner is not None and hmeta.get("used_sam"):
                    fh, fw = frame.shape[:2]
                    rb = hmeta.get("roi_box")
                    if rb is not None and len(rb) >= 4:
                        sam2_roi_tuple = tuple(float(x) for x in rb[:4])
                    else:
                        sam2_roi_tuple = (0.0, 0.0, float(fw), float(fh))

                per_det_masks = hmeta.get("per_det_masks")
                if per_det_masks is None:
                    per_det_masks = [None] * int(len(dets))

                out = tracker.update(dets, frame)
                valid_dancers_this_frame = 0
                if handshake_state is not None and out is not None and len(out) > 0:
                    handshake_state.set_prev_tracker_out(np.atleast_2d(out))
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
                                sam2_input_roi_xyxy=sam2_roi_tuple,
                            )
                        )
                max_dancers_this_chunk = max(max_dancers_this_chunk, valid_dancers_this_frame)
                if lab_on_infer is not None and (yolo_infer_count == 1 or yolo_infer_count % 25 == 0):
                    lab_on_infer(frame_idx, yolo_infer_count, valid_dancers_this_frame)

        for frame_idx, frame in chunk_frames:
            if frame_idx % ystride != 0:
                continue
            h_fr, w_fr = frame.shape[:2]
            frame_low = cv2.resize(frame, (current_detect_size, current_detect_size))
            frame_low_rgb = frame_low[:, :, ::-1]
            scale_x = w_fr / current_detect_size
            scale_y = h_fr / current_detect_size
            det_batch.append((frame_idx, frame, scale_x, scale_y, frame_low_rgb))
            if len(det_batch) >= yolo_bs:
                _flush_track_from_checkpoint_batch(det_batch)
                det_batch.clear()

        _flush_track_from_checkpoint_batch(det_batch)

        max_dancers_last_chunk = max_dancers_this_chunk
        total_frames += len(chunk_frames)
        del chunk_frames

    hybrid_stats: Dict[str, Any] = {}
    if hybrid_refiner is not None:
        hybrid_stats = hybrid_refiner.summary()

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
        hybrid_stats,
        phase1_dets_by_frame,
        {},
    )


def _run_tracking_botsort_pre_stitch(
    video_path: str,
    lab_on_infer: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[
    Dict[int, List[Any]],
    int,
    float,
    Optional[List[Tuple[int, np.ndarray]]],
    float,
    int,
    int,
    int,
    Dict[str, Any],
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
]:
    """Phases 1–2: YOLO track() + BoT-SORT (no post-track stitch)."""
    tr = load_tracking_runtime()
    yconf = float(tr["yolo_conf"])
    ystride = int(tr["yolo_stride"])
    base_detect = int(tr["detect_size"])

    model_path = resolve_yolo_inference_weights()
    print(f"Loading detection model: {model_path}")
    if str(model_path).lower().endswith(".engine"):
        print("  YOLO backend: TensorRT engine (SWAY_YOLO_ENGINE).", flush=True)
    y_half = yolo_predict_use_half()
    if y_half:
        print("  YOLO inference FP16: SWAY_YOLO_HALF=1 (CUDA).", flush=True)
    elif yolo_half_env_requested() and not torch.cuda.is_available():
        print("  Note: SWAY_YOLO_HALF=1 ignored (CUDA not available).", flush=True)
    model = YOLO(model_path)
    raw_tracks: Dict[int, List[Tuple[int, Tuple, float]]] = {}
    phase1_dets_by_frame: Dict[int, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    total_frames = 0
    native_fps = 30.0
    frame_width = 1920
    frame_height = 1080

    tracker_cfg = _get_tracker_config()

    max_dancers_last_chunk = 0
    current_detect_size = base_detect
    yolo_infer_count = 0

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
            yolo_infer_count += 1
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
                half=y_half,
            )
            result = result[0] if isinstance(result, list) else result
            boxes_data = _extract_boxes_and_ids(result)
            boxes_low = boxes_data["boxes"]
            track_ids = boxes_data["track_ids"]
            confs = boxes_data["confs"]

            det_pairs_bs: List[Tuple[Tuple[float, float, float, float], float]] = []
            for box, conf in zip(boxes_low, confs):
                x1, y1, x2, y2 = box
                box_hr = (
                    float(x1 * scale_x),
                    float(y1 * scale_y),
                    float(x2 * scale_x),
                    float(y2 * scale_y),
                )
                det_pairs_bs.append((box_hr, float(conf)))
            phase1_dets_by_frame[frame_idx] = det_pairs_bs

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

            if lab_on_infer is not None and (
                yolo_infer_count == 1 or yolo_infer_count % 25 == 0
            ):
                lab_on_infer(frame_idx, yolo_infer_count, valid_dancers_this_frame)

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
        {},
        phase1_dets_by_frame,
        phase1_dets_by_frame,
    )


def run_tracking_before_post_stitch(
    video_path: str,
    lab_on_infer: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[
    Dict[int, List[Any]],
    int,
    float,
    Optional[List[Tuple[int, np.ndarray]]],
    float,
    int,
    int,
    int,
    Dict[str, Any],
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
    Dict[int, List[Tuple[Tuple[float, float, float, float], float]]],
]:
    """Phases 1–2 only: streaming YOLO + tracker; caller runs apply_post_track_stitching for Phase 3.

    The last return value maps ``frame_idx`` → list of ``(xyxy, conf)`` for every person detection
    fed to the tracker that frame (post NMS / optional hybrid SAM on BoxMOT path), i.e. Phase-1-style
    boxes without using track IDs.
    """
    if _use_boxmot():
        return _run_tracking_boxmot_diou(video_path, lab_on_infer=lab_on_infer)
    return _run_tracking_botsort_pre_stitch(video_path, lab_on_infer=lab_on_infer)


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
    raw, tf, ofps, fl, nf, fw, fh, ys, _hy, _p1, _p1pre = run_tracking_before_post_stitch(video_path)
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
