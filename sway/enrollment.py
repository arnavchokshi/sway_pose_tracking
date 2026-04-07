"""
Dancer Enrollment System and Identity Gallery (PLAN_07)

Builds a closed-set identity gallery for every dancer before tracking begins.
Converts re-ID from open-set ("who among all possible people?") to closed-set
("which of these N enrolled dancers are you?"). Closed-set re-ID is dramatically
easier and more stable.

Env:
  SWAY_ENROLLMENT_ENABLED           – 0|1 (default 1)
  SWAY_ENROLLMENT_AUTO_FRAME        – 0 = auto-select; >0 = fixed frame index (default 0)
  SWAY_ENROLLMENT_MIN_SEPARATION_PX – min center distance for auto-selected frame (default 80)
  SWAY_ENROLLMENT_COLOR_BINS        – histogram bins per channel (default 32)
  SWAY_ENROLLMENT_GALLERY_SIGNALS   – comma-separated subset of {part,face,skeleton,color,spatial} (default part,color,spatial)
  SWAY_ENROLLMENT_PART_MODEL        – bpbreid | paformer (default bpbreid)
"""

from __future__ import annotations

import base64
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

    # Appearance embeddings (populated by PLAN_08 BPBreID or OSNet fallback)
    part_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    global_embedding: Optional[np.ndarray] = None

    # Face (populated by PLAN_11 ArcFace)
    face_embedding: Optional[np.ndarray] = None

    # Color (populated by PLAN_12)
    color_histograms: Dict[str, np.ndarray] = field(default_factory=dict)

    # Gait (populated after 60 frames by PLAN_10 MoCos)
    skeleton_gait_embedding: Optional[np.ndarray] = None

    # Reference area for state machine
    reference_mask_area: float = 0.0

    # Starting position (normalized cx/W, cy/H)
    spatial_position: Tuple[float, float] = (0.0, 0.0)

    enrollment_frame: int = 0


# ---------------------------------------------------------------------------
# Auto-select enrollment frame
# ---------------------------------------------------------------------------

def auto_select_enrollment_frame(
    video_path: str,
    detector=None,
    max_scan_frames: int = 300,
    min_separation_px: Optional[int] = None,
    expected_count: Optional[int] = None,
) -> int:
    """Scan the first N frames for the best enrollment frame.

    Best = all dancers separated with maximum pairwise center distance.

    Args:
        video_path: path to video file.
        detector: a detector with .detect(frame) method. If None, detection is skipped
                  and frame 0 is returned.
        max_scan_frames: how many frames to scan (default 300 ≈ 10s @ 30fps).
        min_separation_px: minimum center distance between any pair.
        expected_count: if set, prefer frames with exactly this many detections.

    Returns:
        Frame index of the best enrollment frame.
    """
    if min_separation_px is None:
        min_separation_px = _env_int("SWAY_ENROLLMENT_MIN_SEPARATION_PX", 80)

    if detector is None:
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video %s for enrollment scan", video_path)
        return 0

    best_frame = 0
    best_score = -1.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scan_limit = min(max_scan_frames, total)

    for fidx in range(scan_limit):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip every other frame for speed during scanning
        if fidx % 3 != 0 and fidx > 0:
            continue

        try:
            dets = detector.detect(frame, frame_idx=fidx)
        except TypeError:
            dets = detector.detect(frame)
        if isinstance(dets, tuple) and len(dets) == 2:
            dets = dets[0]
        if not dets or len(dets) < 2:
            continue

        # If expected count is known, prefer frames matching it
        if expected_count is not None and len(dets) != expected_count:
            continue

        centers = np.array([
            [(d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2]
            for d in dets
        ])

        # Compute minimum pairwise distance
        n = len(centers)
        min_dist = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, d)

        if min_dist >= min_separation_px:
            score = min_dist * len(dets)
            if score > best_score:
                best_score = score
                best_frame = fidx

    cap.release()

    if best_score < 0:
        logger.warning("No suitable enrollment frame found; using frame 0")
        return 0

    logger.info("Auto-selected enrollment frame: %d (score=%.1f)", best_frame, best_score)
    return best_frame


# ---------------------------------------------------------------------------
# Enroll dancers
# ---------------------------------------------------------------------------

def enroll_dancers(
    frame: np.ndarray,
    detections: list,
    sam2_masks: Optional[Dict[int, np.ndarray]] = None,
    models: Optional[Dict[str, Any]] = None,
    frame_idx: int = 0,
) -> List[DancerGallery]:
    """Create identity galleries for all detected dancers in the enrollment frame.

    Args:
        frame: BGR image of the enrollment frame.
        detections: list of Detection objects.
        sam2_masks: optional {det_index: binary_mask} from SAM2.
        models: optional dict of loaded models {name: model} for feature extraction.
        frame_idx: frame index.

    Returns:
        List of DancerGallery objects, one per dancer.
    """
    h, w = frame.shape[:2]
    galleries: List[DancerGallery] = []

    for idx, det in enumerate(detections):
        bbox = det.bbox.astype(int)
        x1, y1, x2, y2 = np.clip(bbox, 0, [w, h, w, h])

        # Crop
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        # Mask-isolated crop
        mask = sam2_masks.get(idx) if sam2_masks else None
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            masked_crop = crop.copy()
            masked_crop[~mask_crop] = 0
        else:
            masked_crop = crop

        cx = (det.bbox[0] + det.bbox[2]) / 2
        cy = (det.bbox[1] + det.bbox[3]) / 2
        norm_pos = (float(cx / w), float(cy / h))

        mask_area = float(mask[y1:y2, x1:x2].sum()) if mask is not None else float(
            (x2 - x1) * (y2 - y1)
        )

        gallery = DancerGallery(
            dancer_id=idx + 1,
            reference_mask_area=mask_area,
            spatial_position=norm_pos,
            enrollment_frame=frame_idx,
        )

        # Extract global embedding (OSNet fallback)
        if models and "global_reid" in models:
            try:
                emb = models["global_reid"].extract(masked_crop)
                if emb is not None:
                    gallery.global_embedding = emb / (np.linalg.norm(emb) + 1e-8)
            except Exception as exc:
                logger.debug("Global embedding extraction failed: %s", exc)

        # Extract part embeddings (PLAN_08 BPBreID)
        if models and "part_reid" in models:
            try:
                keypoints = models.get("keypoints", {}).get(idx)
                parts = models["part_reid"].extract(masked_crop, keypoints, mask)
                if parts is not None:
                    gallery.part_embeddings = {
                        k: v / (np.linalg.norm(v) + 1e-8) for k, v in parts.items()
                        if isinstance(v, np.ndarray)
                    }
            except Exception as exc:
                logger.debug("Part embedding extraction failed: %s", exc)

        # Extract color histograms (PLAN_12)
        if models and "color_hist" in models:
            try:
                keypoints = models.get("keypoints", {}).get(idx)
                hists = models["color_hist"].extract(masked_crop, mask, keypoints)
                if hists:
                    gallery.color_histograms = hists
            except Exception as exc:
                logger.debug("Color histogram extraction failed: %s", exc)

        # Face embedding (PLAN_11)
        if models and "face_reid" in models:
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
    """Serialize galleries to JSON + base64 numpy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for g in galleries:
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
        if g.part_embeddings:
            entry["part_embeddings"] = {
                k: _ndarray_to_b64(v) for k, v in g.part_embeddings.items()
            }
            entry["part_embeddings_shapes"] = {
                k: list(v.shape) for k, v in g.part_embeddings.items()
            }
        if g.face_embedding is not None:
            entry["face_embedding"] = _ndarray_to_b64(g.face_embedding)
            entry["face_embedding_shape"] = list(g.face_embedding.shape)
        if g.color_histograms:
            entry["color_histograms"] = {
                k: _ndarray_to_b64(v) for k, v in g.color_histograms.items()
            }
        if g.skeleton_gait_embedding is not None:
            entry["skeleton_gait_embedding"] = _ndarray_to_b64(g.skeleton_gait_embedding)

        data.append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

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
