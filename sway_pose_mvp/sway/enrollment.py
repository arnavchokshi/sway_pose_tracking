"""
Dancer Enrollment System and Identity Gallery (PLAN_23 — Layer 0)

Closed-set identity gallery built from a 5-10 second temporal window.
Multi-modal feature extraction: BPBreID part embeddings, MoCos gait fingerprint,
color histograms from SAM masks, opportunistic face embeddings.

Env:
  SWAY_ENROLLMENT_ENABLED           – 0|1 (default 1)
  SWAY_ENROLLMENT_AUTO_FRAME        – 0 = auto-select; >0 = fixed frame index (default 0)
  SWAY_ENROLLMENT_MIN_SEPARATION_PX – min center distance for auto-selected frame (default 80)
  SWAY_ENROLLMENT_COLOR_BINS        – histogram bins per channel (default 32)
  SWAY_ENROLLMENT_GALLERY_SIGNALS   – comma-separated subset of {part,face,skeleton,color,spatial} (default part,face,skeleton,color,spatial)
  SWAY_ENROLLMENT_PART_MODEL        – bpbreid | paformer (default bpbreid)
  SWAY_ENROLLMENT_WINDOW_FRAMES     – temporal window in frames for multi-frame enrollment (default 300 = 10s@30fps)
  SWAY_ENROLLMENT_TOP_K_FRAMES      – how many best frames to average embeddings over (default 5)
  SWAY_ENROLLMENT_SCAN_STRIDE       – frame stride during enrollment scan (default 3)
  SWAY_ENROLLMENT_EARLY_STOP        – 0|1 enable early termination in window scan (default 0)
  SWAY_GALLERY_CACHE_ENABLED        – 0|1 enable gallery caching (default 1)
  SWAY_GALLERY_CACHE_DIR            – directory for cached galleries (default output/.gallery_cache/)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


@dataclass
class DancerGallery:
    """Per-dancer identity gallery built at enrollment time."""

    dancer_id: int
    name: Optional[str] = None

    part_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    global_embedding: Optional[np.ndarray] = None

    face_embedding: Optional[np.ndarray] = None

    color_histograms: Dict[str, np.ndarray] = field(default_factory=dict)

    skeleton_gait_embedding: Optional[np.ndarray] = None

    reference_mask_area: float = 0.0

    spatial_position: Tuple[float, float] = (0.0, 0.0)

    enrollment_frame: int = 0
    enrollment_frames_used: int = 1


@dataclass
class _FrameCandidate:
    """Scored candidate frame for enrollment."""
    frame_idx: int
    min_separation: float
    detection_count: int
    pose_quality: float
    score: float


# ---------------------------------------------------------------------------
# Multi-frame enrollment window selection
# ---------------------------------------------------------------------------

def select_enrollment_window(
    video_path: str,
    detector=None,
    max_scan_frames: Optional[int] = None,
    min_separation_px: Optional[int] = None,
    expected_count: Optional[int] = None,
    top_k: Optional[int] = None,
    quality_floor: Optional[float] = None,
) -> List[int]:
    """Scan the first N frames and return the top-K best enrollment frames.

    Scoring: separation * detection_count * pose_quality_bonus.
    Returns sorted list of frame indices (best first).

    If quality_floor is set and early stopping is enabled, scanning terminates
    once we have >= top_k candidates whose worst score exceeds quality_floor.
    """
    if max_scan_frames is None:
        max_scan_frames = _env_int("SWAY_ENROLLMENT_WINDOW_FRAMES", 300)
    if min_separation_px is None:
        min_separation_px = _env_int("SWAY_ENROLLMENT_MIN_SEPARATION_PX", 80)
    if top_k is None:
        top_k = _env_int("SWAY_ENROLLMENT_TOP_K_FRAMES", 5)

    scan_stride = _env_int("SWAY_ENROLLMENT_SCAN_STRIDE", 3)
    early_stop = _env_bool("SWAY_ENROLLMENT_EARLY_STOP", False)

    if quality_floor is None and early_stop:
        assumed_count = expected_count if expected_count is not None else 6
        quality_floor = min_separation_px * 2 * assumed_count

    if detector is None:
        return [0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video %s for enrollment scan", video_path)
        return [0]

    candidates: List[_FrameCandidate] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scan_limit = min(max_scan_frames, total)

    for fidx in range(scan_limit):
        ret, frame = cap.read()
        if not ret:
            break

        if fidx % scan_stride != 0 and fidx > 0:
            continue

        try:
            dets = detector.detect(frame, frame_idx=fidx)
        except TypeError:
            dets = detector.detect(frame)
        if isinstance(dets, tuple) and len(dets) == 2:
            dets = dets[0]
        if not dets or len(dets) < 2:
            continue

        if expected_count is not None and len(dets) != expected_count:
            continue

        centers = np.array([
            [(d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2]
            for d in dets
        ])

        n = len(centers)
        min_dist = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, d)

        if min_dist < min_separation_px:
            continue

        pose_bonus = 1.0
        for det in dets:
            if hasattr(det, "confidence"):
                pose_bonus += det.confidence * 0.1

        score = min_dist * len(dets) * pose_bonus
        candidates.append(_FrameCandidate(
            frame_idx=fidx,
            min_separation=min_dist,
            detection_count=len(dets),
            pose_quality=pose_bonus,
            score=score,
        ))

        if early_stop and quality_floor is not None and len(candidates) >= top_k:
            candidates.sort(key=lambda c: c.score, reverse=True)
            if candidates[top_k - 1].score >= quality_floor:
                logger.info(
                    "Early stop at frame %d: top-%d worst score %.1f >= floor %.1f",
                    fidx, top_k, candidates[top_k - 1].score, quality_floor,
                )
                break

    cap.release()

    if not candidates:
        logger.warning("No suitable enrollment frames found; using frame 0")
        return [0]

    candidates.sort(key=lambda c: c.score, reverse=True)
    selected = [c.frame_idx for c in candidates[:top_k]]
    logger.info(
        "Selected %d enrollment frames: %s (scores: %s)",
        len(selected), selected,
        [f"{c.score:.1f}" for c in candidates[:top_k]],
    )
    return selected


def auto_select_enrollment_frame(
    video_path: str,
    detector=None,
    max_scan_frames: int = 300,
    min_separation_px: Optional[int] = None,
    expected_count: Optional[int] = None,
) -> int:
    """Legacy single-frame selection. Returns the single best frame."""
    frames = select_enrollment_window(
        video_path, detector, max_scan_frames, min_separation_px, expected_count, top_k=1,
    )
    return frames[0]


# ---------------------------------------------------------------------------
# Multi-frame enrollment
# ---------------------------------------------------------------------------

def enroll_dancers_multiframe(
    video_path: str,
    frame_indices: List[int],
    detector=None,
    sam_model=None,
    models: Optional[Dict[str, Any]] = None,
    expected_count: Optional[int] = None,
) -> List[DancerGallery]:
    """Enroll dancers by averaging embeddings across multiple frames.

    For each selected frame, detects people, extracts all signals, then
    averages embeddings across frames for each dancer (matched by spatial proximity).
    """
    if not frame_indices:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video %s", video_path)
        return []

    per_frame_galleries: List[List[DancerGallery]] = []

    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        if detector is not None:
            try:
                dets = detector.detect(frame, frame_idx=fidx)
            except TypeError:
                dets = detector.detect(frame)
            if isinstance(dets, tuple) and len(dets) == 2:
                dets = dets[0]
        else:
            continue

        if not dets:
            continue

        sam_masks = None
        if sam_model is not None:
            try:
                sam_masks = _get_sam_masks(sam_model, frame, dets)
            except Exception as exc:
                logger.debug("SAM mask extraction failed: %s", exc)

        galleries = enroll_dancers(frame, dets, sam_masks, models, fidx)
        if galleries:
            per_frame_galleries.append(galleries)

    cap.release()

    if not per_frame_galleries:
        return []

    if len(per_frame_galleries) == 1:
        return per_frame_galleries[0]

    merged = _merge_multiframe_galleries(per_frame_galleries)
    logger.info(
        "Multi-frame enrollment: %d dancers from %d frames",
        len(merged), len(per_frame_galleries),
    )
    return merged


def _get_sam_masks(sam_model, frame: np.ndarray, dets: list) -> Dict[int, np.ndarray]:
    """Extract SAM masks for each detection."""
    masks: Dict[int, np.ndarray] = {}
    for idx, det in enumerate(dets):
        try:
            bbox = det.bbox
            result = sam_model.predict(frame, bboxes=[bbox])
            if result and hasattr(result[0], 'masks') and result[0].masks is not None:
                mask = result[0].masks.data[0].cpu().numpy().astype(bool)
                masks[idx] = mask
        except Exception:
            pass
    return masks


def _merge_multiframe_galleries(
    per_frame: List[List[DancerGallery]],
) -> List[DancerGallery]:
    """Merge galleries across frames by spatial proximity matching."""
    base = per_frame[0]

    for frame_galleries in per_frame[1:]:
        for fg in frame_galleries:
            best_match = None
            best_dist = float("inf")
            for bg in base:
                dist = np.sqrt(
                    (bg.spatial_position[0] - fg.spatial_position[0]) ** 2
                    + (bg.spatial_position[1] - fg.spatial_position[1]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_match = bg

            if best_match is not None and best_dist < 0.3:
                _accumulate_gallery(best_match, fg)

    for g in base:
        _normalize_gallery(g)

    return base


def _accumulate_gallery(target: DancerGallery, source: DancerGallery) -> None:
    """Accumulate embeddings from source into target for later averaging."""
    if source.global_embedding is not None:
        if target.global_embedding is not None:
            target.global_embedding = target.global_embedding + source.global_embedding
        else:
            target.global_embedding = source.global_embedding.copy()

    for k, v in source.part_embeddings.items():
        if k in target.part_embeddings:
            target.part_embeddings[k] = target.part_embeddings[k] + v
        else:
            target.part_embeddings[k] = v.copy()

    for k, v in source.color_histograms.items():
        if k in target.color_histograms:
            target.color_histograms[k] = target.color_histograms[k] + v
        else:
            target.color_histograms[k] = v.copy()

    if source.face_embedding is not None:
        if target.face_embedding is not None:
            target.face_embedding = target.face_embedding + source.face_embedding
        else:
            target.face_embedding = source.face_embedding.copy()

    target.enrollment_frames_used += 1


def _normalize_gallery(g: DancerGallery) -> None:
    """L2-normalize accumulated embeddings after multi-frame averaging."""
    n = max(g.enrollment_frames_used, 1)

    if g.global_embedding is not None:
        g.global_embedding = g.global_embedding / n
        g.global_embedding /= np.linalg.norm(g.global_embedding) + 1e-8

    for k in g.part_embeddings:
        g.part_embeddings[k] = g.part_embeddings[k] / n
        g.part_embeddings[k] /= np.linalg.norm(g.part_embeddings[k]) + 1e-8

    for k in g.color_histograms:
        g.color_histograms[k] = g.color_histograms[k] / n
        total = g.color_histograms[k].sum()
        if total > 0:
            g.color_histograms[k] /= total

    if g.face_embedding is not None:
        g.face_embedding = g.face_embedding / n
        g.face_embedding /= np.linalg.norm(g.face_embedding) + 1e-8


# ---------------------------------------------------------------------------
# Single-frame enrollment (used by multi-frame as inner loop)
# ---------------------------------------------------------------------------

def enroll_dancers(
    frame: np.ndarray,
    detections: list,
    sam2_masks: Optional[Dict[int, np.ndarray]] = None,
    models: Optional[Dict[str, Any]] = None,
    frame_idx: int = 0,
) -> List[DancerGallery]:
    """Create identity galleries for all detected dancers in one frame.

    Args:
        frame: BGR image.
        detections: list of Detection objects with .bbox attribute.
        sam2_masks: optional {det_index: binary_mask} from SAM.
        models: dict of loaded models {name: model} for feature extraction.
        frame_idx: frame index.

    Returns:
        List of DancerGallery objects, one per dancer.
    """
    h, w = frame.shape[:2]
    galleries: List[DancerGallery] = []
    signals = enrollment_gallery_signals()

    for idx, det in enumerate(detections):
        bbox = det.bbox.astype(int) if hasattr(det.bbox, 'astype') else np.array(det.bbox, dtype=int)
        x1, y1, x2, y2 = np.clip(bbox, 0, [w, h, w, h])

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        mask = sam2_masks.get(idx) if sam2_masks else None
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2] if mask.shape[:2] != crop.shape[:2] else mask
            if mask_crop.shape[:2] == crop.shape[:2]:
                masked_crop = crop.copy()
                masked_crop[~mask_crop] = 0
            else:
                masked_crop = crop
        else:
            masked_crop = crop

        cx = (det.bbox[0] + det.bbox[2]) / 2 if hasattr(det.bbox, '__getitem__') else 0
        cy = (det.bbox[1] + det.bbox[3]) / 2 if hasattr(det.bbox, '__getitem__') else 0
        norm_pos = (float(cx / w), float(cy / h))

        mask_area = float(mask_crop.sum()) if mask is not None and mask_crop.shape[:2] == crop.shape[:2] else float(
            (x2 - x1) * (y2 - y1)
        )

        gallery = DancerGallery(
            dancer_id=idx + 1,
            reference_mask_area=mask_area,
            spatial_position=norm_pos,
            enrollment_frame=frame_idx,
        )

        # --- Signal extraction ---

        if models and "global_reid" in models:
            try:
                emb = models["global_reid"].extract(masked_crop)
                if emb is not None:
                    gallery.global_embedding = emb / (np.linalg.norm(emb) + 1e-8)
            except Exception as exc:
                logger.debug("Global embedding extraction failed: %s", exc)

        if "part" in signals and models and "part_reid" in models:
            try:
                kp = models.get("keypoints", {}).get(idx)
                parts = models["part_reid"].extract(masked_crop, kp, mask)
                if parts is not None:
                    if hasattr(parts, "part_embs"):
                        gallery.part_embeddings = {
                            k: v / (np.linalg.norm(v) + 1e-8)
                            for k, v in parts.part_embs.items()
                            if isinstance(v, np.ndarray)
                        }
                        if parts.global_emb is not None:
                            gallery.global_embedding = parts.global_emb / (
                                np.linalg.norm(parts.global_emb) + 1e-8
                            )
                    elif isinstance(parts, dict):
                        gallery.part_embeddings = {
                            k: v / (np.linalg.norm(v) + 1e-8)
                            for k, v in parts.items()
                            if isinstance(v, np.ndarray)
                        }
            except Exception as exc:
                logger.debug("Part embedding extraction failed: %s", exc)

        if "color" in signals and models and "color_hist" in models:
            try:
                kp = models.get("keypoints", {}).get(idx)
                hists = models["color_hist"].extract(masked_crop, mask, kp)
                if hists:
                    gallery.color_histograms = hists
            except Exception as exc:
                logger.debug("Color histogram extraction failed: %s", exc)

        if "face" in signals and models and "face_reid" in models:
            try:
                face_emb = models["face_reid"].extract(crop)
                if face_emb is not None:
                    gallery.face_embedding = face_emb / (np.linalg.norm(face_emb) + 1e-8)
            except Exception as exc:
                logger.debug("Face embedding extraction failed: %s", exc)

        galleries.append(gallery)

    logger.info("Enrolled %d dancers at frame %d", len(galleries), frame_idx)
    return galleries


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _ndarray_to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def _b64_to_ndarray(s: str, shape: Tuple[int, ...] = (-1,)) -> np.ndarray:
    data = base64.b64decode(s)
    arr = np.frombuffer(data, dtype=np.float32)
    if shape != (-1,):
        arr = arr.reshape(shape)
    return arr


def save_gallery(galleries: List[DancerGallery], path: str | Path) -> None:
    """Serialize galleries to JSON + base64 numpy, and also save a binary .npz for fast I/O."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    npz_arrays: Dict[str, np.ndarray] = {}
    for idx, g in enumerate(galleries):
        entry: Dict[str, Any] = {
            "dancer_id": g.dancer_id,
            "name": g.name,
            "reference_mask_area": g.reference_mask_area,
            "spatial_position": list(g.spatial_position),
            "enrollment_frame": g.enrollment_frame,
        }
        if g.global_embedding is not None:
            entry["global_embedding"] = _ndarray_to_b64(g.global_embedding)
            entry["global_embedding_shape"] = list(g.global_embedding.shape)
            npz_arrays[f"d{idx}_global"] = g.global_embedding.astype(np.float32)
        if g.part_embeddings:
            entry["part_embeddings"] = {
                k: _ndarray_to_b64(v) for k, v in g.part_embeddings.items()
            }
            entry["part_embeddings_shapes"] = {
                k: list(v.shape) for k, v in g.part_embeddings.items()
            }
            for k, v in g.part_embeddings.items():
                npz_arrays[f"d{idx}_part_{k}"] = v.astype(np.float32)
        if g.face_embedding is not None:
            entry["face_embedding"] = _ndarray_to_b64(g.face_embedding)
            entry["face_embedding_shape"] = list(g.face_embedding.shape)
            npz_arrays[f"d{idx}_face"] = g.face_embedding.astype(np.float32)
        if g.color_histograms:
            entry["color_histograms"] = {
                k: _ndarray_to_b64(v) for k, v in g.color_histograms.items()
            }
            for k, v in g.color_histograms.items():
                npz_arrays[f"d{idx}_hist_{k}"] = v.astype(np.float32)
        if g.skeleton_gait_embedding is not None:
            entry["skeleton_gait_embedding"] = _ndarray_to_b64(g.skeleton_gait_embedding)
            npz_arrays[f"d{idx}_gait"] = g.skeleton_gait_embedding.astype(np.float32)

        data.append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    npz_path = path.with_suffix(".npz")
    if npz_arrays:
        np.savez_compressed(npz_path, **npz_arrays)
        logger.info("Gallery binary saved to %s (%.1f KB)", npz_path, npz_path.stat().st_size / 1024)

    logger.info("Gallery saved to %s (%d dancers)", path, len(galleries))


def load_gallery(path: str | Path) -> List[DancerGallery]:
    """Deserialize galleries from JSON."""
    path = Path(path)
    if not path.exists():
        logger.warning("Gallery file not found: %s", path)
        return []

    with open(path) as f:
        data = json.load(f)

    galleries = []
    for entry in data:
        g = DancerGallery(
            dancer_id=entry["dancer_id"],
            name=entry.get("name"),
            reference_mask_area=entry.get("reference_mask_area", 0.0),
            spatial_position=tuple(entry.get("spatial_position", [0.0, 0.0])),
            enrollment_frame=entry.get("enrollment_frame", 0),
        )
        if "global_embedding" in entry:
            shape = tuple(entry.get("global_embedding_shape", [-1]))
            g.global_embedding = _b64_to_ndarray(entry["global_embedding"], shape)
        if "part_embeddings" in entry:
            shapes = entry.get("part_embeddings_shapes", {})
            g.part_embeddings = {
                k: _b64_to_ndarray(v, tuple(shapes.get(k, [-1])))
                for k, v in entry["part_embeddings"].items()
            }
        if "face_embedding" in entry:
            shape = tuple(entry.get("face_embedding_shape", [-1]))
            g.face_embedding = _b64_to_ndarray(entry["face_embedding"], shape)
        if "color_histograms" in entry:
            g.color_histograms = {
                k: _b64_to_ndarray(v) for k, v in entry["color_histograms"].items()
            }
        if "skeleton_gait_embedding" in entry:
            g.skeleton_gait_embedding = _b64_to_ndarray(entry["skeleton_gait_embedding"])

        galleries.append(g)

    logger.info("Gallery loaded from %s (%d dancers)", path, len(galleries))
    return galleries


# ---------------------------------------------------------------------------
# Gallery caching
# ---------------------------------------------------------------------------

def gallery_cache_key(video_path: str) -> str:
    """Compute a deterministic hash from the first 1 MB of the video + file metadata."""
    p = Path(video_path)
    stat = p.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime).encode())
    with open(p, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()


def _gallery_cache_dir() -> Path:
    return Path(os.environ.get("SWAY_GALLERY_CACHE_DIR", "output/.gallery_cache"))


def load_cached_gallery(video_path: str, cache_dir: Optional[Path] = None) -> Optional[List[DancerGallery]]:
    """Return cached galleries for *video_path*, or None on miss."""
    if not _env_bool("SWAY_GALLERY_CACHE_ENABLED", True):
        return None
    if cache_dir is None:
        cache_dir = _gallery_cache_dir()
    try:
        key = gallery_cache_key(video_path)
    except OSError:
        return None
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        return None
    logger.info("Gallery cache hit: %s", cache_file)
    return load_gallery(cache_file)


def save_gallery_cache(galleries: List[DancerGallery], video_path: str, cache_dir: Optional[Path] = None) -> None:
    """Persist *galleries* under a content-addressed cache key for *video_path*."""
    if not _env_bool("SWAY_GALLERY_CACHE_ENABLED", True):
        return
    if cache_dir is None:
        cache_dir = _gallery_cache_dir()
    try:
        key = gallery_cache_key(video_path)
    except OSError:
        logger.debug("Cannot compute gallery cache key for %s", video_path)
        return
    cache_file = cache_dir / f"{key}.json"
    save_gallery(galleries, cache_file)
    logger.info("Gallery cached: %s", cache_file)


def is_enrollment_enabled() -> bool:
    return _env_bool("SWAY_ENROLLMENT_ENABLED", True)


def enrollment_gallery_signals() -> set:
    """Which re-ID signals to collect at enrollment time.

    Returns a set of signal names, e.g. {"part", "color", "spatial"}.
    """
    raw = os.environ.get("SWAY_ENROLLMENT_GALLERY_SIGNALS", "").strip()
    if not raw:
        return {"part", "color", "spatial"}
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def enrollment_part_model() -> str:
    """Which part-based re-ID model for enrollment embeddings."""
    return os.environ.get("SWAY_ENROLLMENT_PART_MODEL", "bpbreid").strip() or "bpbreid"
