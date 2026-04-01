"""
ArcFace Face Recognition Signal (PLAN_11)

EXPERIMENT ADD-ON — not lean core.

Opportunistic face recognition: when a dancer's face is visible and unoccluded,
ArcFace provides an extremely strong identity signal (99%+ on clean frontal faces).
Available on maybe 30-50% of frames in dance footage.

Env:
  SWAY_REID_FACE_MODEL    – arcface | adaface (default arcface)
  SWAY_REID_FACE_MIN_SIZE – min inter-eye distance in pixels (default 40)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


class FaceReIDExtractor:
    """Face recognition via insightface (RetinaFace detection + ArcFace embedding)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda",
        min_face_size: Optional[int] = None,
    ):
        self.model_name = model_name or _env_str("SWAY_REID_FACE_MODEL", "arcface")
        self.device = device
        self.min_face_size = min_face_size or _env_int("SWAY_REID_FACE_MIN_SIZE", 40)
        self._app = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from insightface.app import FaceAnalysis

            ctx_id = 0 if "cuda" in self.device else -1
            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("InsightFace loaded (ArcFace) on %s", self.device)
        except ImportError:
            logger.info(
                "insightface not installed; face re-ID disabled. "
                "Install with: pip install insightface onnxruntime-gpu"
            )
        except Exception as exc:
            logger.warning("InsightFace load failed: %s", exc)

    def extract(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from a person crop.

        Args:
            person_crop: BGR person crop.

        Returns:
            512-d L2-normalized face embedding, or None if no face detected.
        """
        if self._app is None:
            return None

        try:
            faces = self._app.get(person_crop)
        except Exception as exc:
            logger.debug("Face detection failed: %s", exc)
            return None

        if not faces:
            return None

        # Select the largest face with sufficient quality
        best_face = None
        best_area = 0
        for face in faces:
            if face.det_score < 0.7:
                continue
            bbox = face.bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > best_area:
                best_area = area
                best_face = face

        if best_face is None:
            return None

        # Check minimum face size (inter-eye distance proxy: bbox width)
        face_width = best_face.bbox[2] - best_face.bbox[0]
        if face_width < self.min_face_size:
            return None

        emb = best_face.embedding
        if emb is None:
            return None

        emb = emb.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def compare(self, gallery_emb: np.ndarray, query_emb: np.ndarray) -> float:
        """Cosine distance. Lower = more similar."""
        return float(1.0 - np.dot(gallery_emb, query_emb))
