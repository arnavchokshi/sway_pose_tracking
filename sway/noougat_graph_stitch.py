"""
NOOUGAT-style dark-zone graph stitcher.

Builds entry/exit tracklet nodes around occlusion coalescence events and solves
identity continuity across the dark zone with global assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from sway.reid_fusion import ReIDQuery


def _env_float(key: str, default: float) -> float:
    import os

    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _center_xy(bbox: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(_norm(a), _norm(b)))


def _build_feature_vector(query: ReIDQuery) -> np.ndarray:
    """
    Compact vector for graph-level similarity.
    Prefers part global embedding, then appends face embedding when available.
    """
    parts: List[np.ndarray] = []
    if query.part_embeddings is not None and hasattr(query.part_embeddings, "global_emb"):
        ge = np.asarray(query.part_embeddings.global_emb, dtype=np.float32).flatten()
        if ge.size > 0:
            parts.append(_norm(ge))
    if query.face_embedding is not None:
        fe = np.asarray(query.face_embedding, dtype=np.float32).flatten()
        if fe.size > 0:
            parts.append(_norm(fe))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


@dataclass
class TrackletNode:
    track_id: int
    frame_idx: int
    bbox_xyxy: np.ndarray
    center_xy: Tuple[float, float]
    velocity_xy: Tuple[float, float]
    query: ReIDQuery
    feature_vec: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


@dataclass
class DarkZone:
    zone_id: int
    track_ids: List[int]
    entry_frame: int
    exit_frame: Optional[int] = None
    entry_nodes: Dict[int, TrackletNode] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class StitchResult:
    zone_id: int
    assignments: List[Tuple[int, int]]
    cost_matrix: np.ndarray
    algorithm: str


class NOOUGATGraphStitcher:
    def __init__(self):
        self._zones: Dict[int, DarkZone] = {}
        self._event_key_to_zone_id: Dict[Tuple[int, ...], int] = {}
        self._next_zone_id = 1
        self.w_app = _env_float("SWAY_NOOUGAT_W_APP", 0.60)
        self.w_spatial = _env_float("SWAY_NOOUGAT_W_SPATIAL", 0.30)
        self.w_temporal = _env_float("SWAY_NOOUGAT_W_TEMPORAL", 0.10)
        self.max_center_dist_px = _env_float("SWAY_NOOUGAT_MAX_CENTER_DIST_PX", 400.0)
        self.max_gap_frames = max(1.0, _env_float("SWAY_NOOUGAT_MAX_GAP_FRAMES", 180.0))

    @staticmethod
    def _event_key(track_ids: List[int]) -> Tuple[int, ...]:
        return tuple(sorted(int(t) for t in track_ids))

    def start_dark_zone(self, track_ids: List[int], entry_frame: int, entry_nodes: Dict[int, TrackletNode]) -> int:
        key = self._event_key(track_ids)
        existing = self._event_key_to_zone_id.get(key)
        if existing is not None:
            z = self._zones.get(existing)
            if z is not None and not z.resolved:
                z.entry_nodes.update(entry_nodes)
                return existing

        zid = self._next_zone_id
        self._next_zone_id += 1
        zone = DarkZone(
            zone_id=zid,
            track_ids=list(sorted(int(t) for t in track_ids)),
            entry_frame=int(entry_frame),
            entry_nodes=dict(entry_nodes),
            resolved=False,
        )
        self._zones[zid] = zone
        self._event_key_to_zone_id[key] = zid
        return zid

    def resolve_dark_zone(
        self,
        track_ids: List[int],
        exit_frame: int,
        exit_nodes: Dict[int, TrackletNode],
    ) -> Optional[StitchResult]:
        key = self._event_key(track_ids)
        zid = self._event_key_to_zone_id.get(key)
        if zid is None:
            return None
        zone = self._zones.get(zid)
        if zone is None or zone.resolved:
            return None
        if not zone.entry_nodes or not exit_nodes:
            zone.resolved = True
            zone.exit_frame = int(exit_frame)
            return None

        row_ids = [tid for tid in zone.track_ids if tid in zone.entry_nodes]
        col_ids = [tid for tid in sorted(exit_nodes.keys())]
        if not row_ids or not col_ids:
            zone.resolved = True
            zone.exit_frame = int(exit_frame)
            return None

        cost = np.zeros((len(row_ids), len(col_ids)), dtype=np.float64)
        for i, src_tid in enumerate(row_ids):
            src = zone.entry_nodes[src_tid]
            for j, dst_tid in enumerate(col_ids):
                dst = exit_nodes[dst_tid]
                cost[i, j] = self._pair_cost(src, dst, int(exit_frame - zone.entry_frame))

        row_ind, col_ind = linear_sum_assignment(cost)
        assigns: List[Tuple[int, int]] = []
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            assigns.append((int(row_ids[r]), int(col_ids[c])))

        zone.resolved = True
        zone.exit_frame = int(exit_frame)
        return StitchResult(
            zone_id=zid,
            assignments=assigns,
            cost_matrix=cost,
            algorithm="hungarian",
        )

    def _pair_cost(self, src: TrackletNode, dst: TrackletNode, gap_frames: int) -> float:
        app_sim = 0.0
        if src.feature_vec.size > 0 and dst.feature_vec.size > 0 and src.feature_vec.shape == dst.feature_vec.shape:
            app_sim = _cosine_sim(src.feature_vec, dst.feature_vec)
        app_cost = 1.0 - max(-1.0, min(1.0, app_sim))
        app_cost = max(0.0, min(2.0, app_cost)) * 0.5  # map to ~[0,1]

        pred_x = src.center_xy[0] + float(gap_frames) * src.velocity_xy[0]
        pred_y = src.center_xy[1] + float(gap_frames) * src.velocity_xy[1]
        dist = float(np.hypot(dst.center_xy[0] - pred_x, dst.center_xy[1] - pred_y))
        spatial_cost = min(1.0, dist / max(1.0, self.max_center_dist_px))

        temporal_cost = min(1.0, float(max(0, gap_frames)) / self.max_gap_frames)

        return (
            self.w_app * app_cost
            + self.w_spatial * spatial_cost
            + self.w_temporal * temporal_cost
        )


def make_tracklet_node(
    track_id: int,
    frame_idx: int,
    bbox_xyxy: np.ndarray,
    query: ReIDQuery,
    prev_center_xy: Optional[Tuple[float, float]] = None,
    dt_frames: int = 1,
) -> TrackletNode:
    center = _center_xy(bbox_xyxy.astype(np.float32))
    if prev_center_xy is None or dt_frames <= 0:
        vel = (0.0, 0.0)
    else:
        vel = (
            (center[0] - float(prev_center_xy[0])) / float(dt_frames),
            (center[1] - float(prev_center_xy[1])) / float(dt_frames),
        )
    return TrackletNode(
        track_id=int(track_id),
        frame_idx=int(frame_idx),
        bbox_xyxy=bbox_xyxy.astype(np.float32),
        center_xy=center,
        velocity_xy=vel,
        query=query,
        feature_vec=_build_feature_vector(query),
    )
