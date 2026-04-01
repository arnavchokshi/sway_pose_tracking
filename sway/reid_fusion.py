"""
Multi-Signal Re-ID Fusion Engine (PLAN_13)

Confidence-gated weighted ensemble that fuses six independent re-ID signals
into a single score per candidate identity match. Each signal contributes only
when it passes a quality gate. Weights are auto-normalized after gating.

Signals:
  1. Part-based appearance (BPBreID)     — always available
  2. KPR keypoint-prompted              — only during multi-person overlap
  3. Skeleton gait (MoCos)              — requires 30+ frames of skeleton data
  4. Face (ArcFace)                     — opportunistic, ~30-50% of frames
  5. Color histograms                   — always available
  6. Spatial formation prior            — always available

Env:
  SWAY_REID_W_PART     – weight for part-based signal    (default 0.30)
  SWAY_REID_W_KPR      – weight for KPR signal          (default 0.15)
  SWAY_REID_W_SKELETON – weight for skeleton gait signal (default 0.20)
  SWAY_REID_W_FACE     – weight for face signal          (default 0.20)
  SWAY_REID_W_COLOR    – weight for color signal         (default 0.10)
  SWAY_REID_W_SPATIAL  – weight for spatial signal       (default 0.05)
  SWAY_REID_MIN_CONFIDENCE – min fused score for ID assignment (default 0.50)
  SWAY_REID_SPATIAL_DECAY  – temporal decay for spatial prior (default 0.01)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from sway.enrollment import DancerGallery

logger = logging.getLogger(__name__)

UNKNOWN_ID = -1


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


@dataclass
class ReIDQuery:
    """Features of a track requiring re-identification."""
    track_id: int
    part_embeddings: Optional[object] = None        # PartEmbeddings from PLAN_08
    kpr_embedding: Optional[np.ndarray] = None      # from PLAN_09
    gait_embedding: Optional[np.ndarray] = None     # from PLAN_10
    face_embedding: Optional[np.ndarray] = None     # from PLAN_11
    color_histograms: Optional[Dict[str, np.ndarray]] = None  # from PLAN_12
    spatial_position: Tuple[float, float] = (0.0, 0.0)
    is_multi_person_overlap: bool = False
    skeleton_window_length: int = 0


class ReIDFusionEngine:
    """Weighted ensemble of six re-ID signals with confidence gating."""

    def __init__(
        self,
        gallery: List[DancerGallery],
        weights: Optional[Dict[str, float]] = None,
        signal_modules: Optional[Dict[str, object]] = None,
    ):
        self.gallery = gallery
        self.signal_modules = signal_modules or {}

        default_weights = {
            "part": _env_float("SWAY_REID_W_PART", 0.30),
            "kpr": _env_float("SWAY_REID_W_KPR", 0.15),
            "skeleton": _env_float("SWAY_REID_W_SKELETON", 0.20),
            "face": _env_float("SWAY_REID_W_FACE", 0.20),
            "color": _env_float("SWAY_REID_W_COLOR", 0.10),
            "spatial": _env_float("SWAY_REID_W_SPATIAL", 0.05),
        }
        self.weights = weights or default_weights

        self.min_confidence = _env_float("SWAY_REID_MIN_CONFIDENCE", 0.50)
        self.spatial_decay = _env_float("SWAY_REID_SPATIAL_DECAY", 0.01)

        # Last known positions: {dancer_id: (x, y, frame)}
        self.last_known_positions: Dict[int, Tuple[float, float, int]] = {}

        # Initialize from gallery
        for g in gallery:
            self.last_known_positions[g.dancer_id] = (
                g.spatial_position[0], g.spatial_position[1], g.enrollment_frame
            )

    def update_gallery(self, gallery: List[DancerGallery]) -> None:
        self.gallery = gallery

    def update_position(self, dancer_id: int, pos: Tuple[float, float], frame: int) -> None:
        self.last_known_positions[dancer_id] = (pos[0], pos[1], frame)

    def match(self, query: ReIDQuery, current_frame: int = 0) -> Tuple[int, float]:
        """Match a query against the gallery. Returns (dancer_id, confidence).

        Returns (UNKNOWN_ID, score) if confidence is below threshold.
        """
        if not self.gallery:
            return UNKNOWN_ID, 0.0

        best_id = UNKNOWN_ID
        best_score = -1.0

        skel_min = int(_env_float("SWAY_REID_SKEL_MIN_WINDOW", 30))
        part_min = int(_env_float("SWAY_REID_PART_MIN_VISIBLE", 3))

        for dancer in self.gallery:
            signal_scores: Dict[str, float] = {}
            active_weights: Dict[str, float] = {}

            # Signal 1: Part-based appearance
            if query.part_embeddings is not None and dancer.part_embeddings:
                try:
                    from sway.bpbreid_extractor import BPBreIDExtractor
                    part_module = self.signal_modules.get("part")
                    if part_module:
                        dist = part_module.compare(
                            _gallery_to_part_emb(dancer), query.part_embeddings
                        )
                    else:
                        dist = _cosine_dist_global(dancer, query)
                    signal_scores["part"] = 1.0 - dist
                    active_weights["part"] = self.weights["part"]
                except Exception:
                    pass

            elif query.part_embeddings is not None and dancer.global_embedding is not None:
                if hasattr(query.part_embeddings, "global_emb"):
                    dist = 1.0 - float(np.dot(dancer.global_embedding, query.part_embeddings.global_emb))
                    signal_scores["part"] = 1.0 - dist
                    active_weights["part"] = self.weights["part"]

            # Signal 2: KPR (only during multi-person overlap)
            if (query.is_multi_person_overlap and query.kpr_embedding is not None
                    and dancer.global_embedding is not None):
                dist = 1.0 - float(np.dot(dancer.global_embedding, query.kpr_embedding))
                signal_scores["kpr"] = 1.0 - dist
                active_weights["kpr"] = self.weights["kpr"]

            # Signal 3: Skeleton gait
            if (query.gait_embedding is not None
                    and dancer.skeleton_gait_embedding is not None
                    and query.skeleton_window_length >= skel_min):
                dist = 1.0 - float(np.dot(dancer.skeleton_gait_embedding, query.gait_embedding))
                signal_scores["skeleton"] = 1.0 - dist
                active_weights["skeleton"] = self.weights["skeleton"]

            # Signal 4: Face
            if query.face_embedding is not None and dancer.face_embedding is not None:
                dist = 1.0 - float(np.dot(dancer.face_embedding, query.face_embedding))
                signal_scores["face"] = 1.0 - dist
                active_weights["face"] = self.weights["face"]

            # Signal 5: Color histograms (always available)
            if query.color_histograms and dancer.color_histograms:
                try:
                    color_module = self.signal_modules.get("color")
                    if color_module:
                        dist = color_module.compare(dancer.color_histograms, query.color_histograms)
                    else:
                        dist = _histogram_dist(dancer.color_histograms, query.color_histograms)
                    signal_scores["color"] = 1.0 - dist
                    active_weights["color"] = self.weights["color"]
                except Exception:
                    pass

            # Signal 6: Spatial formation prior (always available)
            spatial_score = self._compute_spatial_score(dancer.dancer_id, query, current_frame)
            signal_scores["spatial"] = spatial_score
            active_weights["spatial"] = self.weights["spatial"]

            # Fuse: re-normalize active weights to sum to 1.0
            total_weight = sum(active_weights.values())
            if total_weight <= 0:
                continue

            fused = sum(
                (active_weights[sig] / total_weight) * signal_scores[sig]
                for sig in signal_scores
            )

            if fused > best_score:
                best_score = fused
                best_id = dancer.dancer_id

        if best_score < self.min_confidence:
            logger.debug(
                "Re-ID: no confident match for track %d (best: dancer %d, score=%.3f)",
                query.track_id, best_id, best_score,
            )
            return UNKNOWN_ID, best_score

        return best_id, best_score

    def match_batch(
        self, queries: List[ReIDQuery], current_frame: int = 0
    ) -> List[Tuple[int, float]]:
        """Match multiple queries (used by collision solver PLAN_15)."""
        return [self.match(q, current_frame) for q in queries]

    def _compute_spatial_score(
        self, dancer_id: int, query: ReIDQuery, current_frame: int
    ) -> float:
        """Spatial formation prior with temporal decay."""
        if dancer_id not in self.last_known_positions:
            return 0.5

        lx, ly, lf = self.last_known_positions[dancer_id]
        qx, qy = query.spatial_position

        dist = math.sqrt((lx - qx) ** 2 + (ly - qy) ** 2)
        frames_since = max(1, current_frame - lf)

        decay = math.exp(-self.spatial_decay * frames_since)
        score = max(0.0, 1.0 - dist) * decay

        return score


def _cosine_dist_global(dancer: DancerGallery, query: ReIDQuery) -> float:
    if dancer.global_embedding is not None and hasattr(query.part_embeddings, "global_emb"):
        return 1.0 - float(np.dot(dancer.global_embedding, query.part_embeddings.global_emb))
    return 1.0


def _gallery_to_part_emb(dancer: DancerGallery):
    from sway.bpbreid_extractor import PartEmbeddings
    return PartEmbeddings(
        global_emb=dancer.global_embedding if dancer.global_embedding is not None else np.zeros(2048),
        part_embs=dancer.part_embeddings,
        visibility={k: True for k in dancer.part_embeddings},
    )


def _histogram_dist(h1: Dict[str, np.ndarray], h2: Dict[str, np.ndarray]) -> float:
    shared = set(h1.keys()) & set(h2.keys())
    if not shared:
        return 1.0
    dists = []
    for k in shared:
        a, b = h1[k], h2[k]
        if len(a) == len(b):
            d = np.sqrt(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)) / np.sqrt(2)
            dists.append(d)
    return float(np.mean(dists)) if dists else 1.0
