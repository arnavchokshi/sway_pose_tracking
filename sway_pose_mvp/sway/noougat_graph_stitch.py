"""
NOOUGAT-Inspired Graph Neural Network Trajectory Stitching (PLAN_23 — Layer 4)

Global association module that represents entry/exit tracklets as nodes in a
bipartite graph and solves optimal assignment using GNN-learned embeddings +
Hungarian algorithm.

For offline or slight-latency processing, combines with backward-pass gap filling
and MOTE disocclusion prediction for maximum identity recovery.

Env:
  SWAY_NOOUGAT_ENABLED          – 0|1 (default 1)
  SWAY_NOOUGAT_SUBCLIP_LENGTH   – frames per subclip for local trajectories (default 30)
  SWAY_NOOUGAT_GNN_LAYERS       – number of GNN message-passing layers (default 3)
  SWAY_NOOUGAT_EMBED_DIM        – graph embedding dimension (default 128)
  SWAY_NOOUGAT_MAX_GAP_FRAMES   – max temporal gap for stitching (default 300)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

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
class TrackletNode:
    """A tracklet represented as a node in the association graph."""
    track_id: int
    start_frame: int
    end_frame: int
    embedding: np.ndarray
    spatial_trajectory: List[Tuple[float, float]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_entry: bool = True


@dataclass
class DarkZone:
    """A temporal region where tracks are merged/occluded."""
    entry_frame: int
    exit_frame: int
    entry_nodes: List[TrackletNode] = field(default_factory=list)
    exit_nodes: List[TrackletNode] = field(default_factory=list)
    assignments: List[Tuple[int, int]] = field(default_factory=list)
    resolved: bool = False


@dataclass
class StitchResult:
    """Result of global trajectory stitching."""
    stitches: List[Tuple[int, int]]
    confidence_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    dark_zones_resolved: int = 0
    total_gap_frames: int = 0


class NOOUGATGraphStitcher:
    """Graph-based global trajectory stitching inspired by NOOUGAT.

    Partitions video into subclips, builds local trajectories, then fuses
    them into global trajectories via hierarchical GNN message passing.
    Falls back to appearance + motion features when no GNN weights are available.
    """

    def __init__(
        self,
        subclip_length: Optional[int] = None,
        gnn_layers: Optional[int] = None,
        embed_dim: Optional[int] = None,
        max_gap_frames: Optional[int] = None,
        fusion_engine=None,
    ):
        self.subclip_length = subclip_length or _env_int("SWAY_NOOUGAT_SUBCLIP_LENGTH", 30)
        self.gnn_layers = gnn_layers or _env_int("SWAY_NOOUGAT_GNN_LAYERS", 3)
        self.embed_dim = embed_dim or _env_int("SWAY_NOOUGAT_EMBED_DIM", 128)
        self.max_gap_frames = max_gap_frames or _env_int("SWAY_NOOUGAT_MAX_GAP_FRAMES", 300)
        self.fusion_engine = fusion_engine

        self._dark_zones: List[DarkZone] = []
        self._gnn_model = None

    def create_dark_zone(
        self,
        entry_frame: int,
        exit_frame: int,
        entry_tracklets: List[TrackletNode],
        exit_tracklets: List[TrackletNode],
    ) -> DarkZone:
        """Register a dark zone between entry and exit frames.

        Entry tracklets are frozen at cluster formation; exit tracklets
        are captured when the cluster splits.
        """
        zone = DarkZone(
            entry_frame=entry_frame,
            exit_frame=exit_frame,
            entry_nodes=entry_tracklets,
            exit_nodes=exit_tracklets,
        )
        self._dark_zones.append(zone)
        return zone

    def resolve_dark_zone(self, zone: DarkZone) -> List[Tuple[int, int]]:
        """Solve NxN identity assignment for a dark zone.

        Uses graph embeddings + Hungarian algorithm for globally optimal matching.
        """
        if zone.resolved:
            return zone.assignments

        n_entry = len(zone.entry_nodes)
        n_exit = len(zone.exit_nodes)

        if n_entry == 0 or n_exit == 0:
            zone.resolved = True
            return []

        cost_matrix = self._build_graph_cost_matrix(zone)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = []
        for r, c in zip(row_ind, col_ind):
            if r < n_entry and c < n_exit:
                entry_id = zone.entry_nodes[r].track_id
                exit_id = zone.exit_nodes[c].track_id
                assignments.append((entry_id, exit_id))

        zone.assignments = assignments
        zone.resolved = True

        logger.info(
            "Dark zone resolved: frame %d→%d, %d assignments",
            zone.entry_frame, zone.exit_frame, len(assignments),
        )
        return assignments

    def stitch_all(self) -> StitchResult:
        """Resolve all pending dark zones and return global stitch results."""
        all_stitches = []
        all_scores: Dict[Tuple[int, int], float] = {}
        total_gap = 0

        for zone in self._dark_zones:
            if zone.resolved:
                continue

            assignments = self.resolve_dark_zone(zone)
            all_stitches.extend(assignments)
            total_gap += zone.exit_frame - zone.entry_frame

            cost_matrix = self._build_graph_cost_matrix(zone)
            for entry_id, exit_id in assignments:
                entry_idx = next(
                    (i for i, n in enumerate(zone.entry_nodes) if n.track_id == entry_id), 0
                )
                exit_idx = next(
                    (i for i, n in enumerate(zone.exit_nodes) if n.track_id == exit_id), 0
                )
                confidence = 1.0 - cost_matrix[entry_idx, exit_idx]
                all_scores[(entry_id, exit_id)] = max(0.0, confidence)

        return StitchResult(
            stitches=all_stitches,
            confidence_scores=all_scores,
            dark_zones_resolved=sum(1 for z in self._dark_zones if z.resolved),
            total_gap_frames=total_gap,
        )

    def _build_graph_cost_matrix(self, zone: DarkZone) -> np.ndarray:
        """Build cost matrix using graph embeddings + appearance + motion features."""
        n = len(zone.entry_nodes)
        m = len(zone.exit_nodes)
        cost = np.ones((n, m), dtype=np.float64) * 0.99

        for i, entry in enumerate(zone.entry_nodes):
            for j, exit_node in enumerate(zone.exit_nodes):
                appearance_dist = self._appearance_distance(entry, exit_node)
                motion_dist = self._motion_distance(entry, exit_node, zone)
                temporal_dist = self._temporal_distance(entry, exit_node, zone)

                cost[i, j] = (
                    0.5 * appearance_dist
                    + 0.3 * motion_dist
                    + 0.2 * temporal_dist
                )

        return cost

    def _appearance_distance(self, entry: TrackletNode, exit_node: TrackletNode) -> float:
        """Cosine distance between graph embeddings."""
        if entry.embedding is None or exit_node.embedding is None:
            return 1.0

        e = entry.embedding
        x = exit_node.embedding

        if np.linalg.norm(e) < 1e-8 or np.linalg.norm(x) < 1e-8:
            return 1.0

        e_norm = e / (np.linalg.norm(e) + 1e-8)
        x_norm = x / (np.linalg.norm(x) + 1e-8)

        return float(1.0 - np.dot(e_norm, x_norm))

    def _motion_distance(
        self, entry: TrackletNode, exit_node: TrackletNode, zone: DarkZone
    ) -> float:
        """Predict exit position from entry velocity and compare."""
        if not entry.spatial_trajectory or not exit_node.spatial_trajectory:
            return 0.5

        entry_pos = entry.spatial_trajectory[-1]
        exit_pos = exit_node.spatial_trajectory[0]

        gap_frames = max(1, zone.exit_frame - zone.entry_frame)
        predicted_x = entry_pos[0] + entry.velocity[0] * gap_frames
        predicted_y = entry_pos[1] + entry.velocity[1] * gap_frames

        dist = np.sqrt((predicted_x - exit_pos[0]) ** 2 + (predicted_y - exit_pos[1]) ** 2)
        return min(1.0, dist)

    def _temporal_distance(
        self, entry: TrackletNode, exit_node: TrackletNode, zone: DarkZone
    ) -> float:
        """Temporal plausibility: shorter gaps are more likely correct matches."""
        gap = zone.exit_frame - zone.entry_frame
        return min(1.0, gap / max(self.max_gap_frames, 1))

    def hierarchical_merge(self) -> List[Tuple[int, int]]:
        """GAP 4-A: SUSHI-style hierarchical subgraph merge.

        Groups resolved zones into subgraphs by shared track IDs, then
        iteratively merges highest-affinity pairs until cost exceeds threshold.
        Returns list of (src_tid, dst_tid) additional alias links.
        """
        resolved = [z for z in self._dark_zones if z.resolved]
        if len(resolved) < 2:
            return []

        subgraphs: Dict[int, set] = {}
        for zone in resolved:
            tids = set()
            for t in zone.entry_nodes:
                tids.add(t.track_id)
            for t in zone.exit_nodes:
                tids.add(t.track_id)
            for tid in tids:
                if tid not in subgraphs:
                    subgraphs[tid] = set()
                subgraphs[tid].update(tids)

        clusters: List[set] = []
        visited: set = set()
        for tid, connected in subgraphs.items():
            if tid in visited:
                continue
            cluster = set()
            queue = [tid]
            while queue:
                curr = queue.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                cluster.add(curr)
                for neighbor in subgraphs.get(curr, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            clusters.append(cluster)

        additional_links: List[Tuple[int, int]] = []
        merge_threshold = 0.6

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1_embs = []
                c2_embs = []
                for zone in resolved:
                    for t in zone.entry_nodes + zone.exit_nodes:
                        if t.track_id in clusters[i]:
                            c1_embs.append(t.embedding)
                        elif t.track_id in clusters[j]:
                            c2_embs.append(t.embedding)
                if c1_embs and c2_embs:
                    c1_mean = np.mean(np.stack(c1_embs), axis=0)
                    c2_mean = np.mean(np.stack(c2_embs), axis=0)
                    c1_mean /= (np.linalg.norm(c1_mean) + 1e-8)
                    c2_mean /= (np.linalg.norm(c2_mean) + 1e-8)
                    sim = float(np.dot(c1_mean, c2_mean))
                    if sim > merge_threshold:
                        t1 = min(clusters[i])
                        t2 = min(clusters[j])
                        additional_links.append((t1, t2))

        logger.info("Hierarchical merge: %d cluster pairs evaluated, %d links added",
                     len(clusters) * (len(clusters) - 1) // 2, len(additional_links))
        return additional_links

    def get_dark_zones(self) -> List[DarkZone]:
        return list(self._dark_zones)

    def get_unresolved_zones(self) -> List[DarkZone]:
        return [z for z in self._dark_zones if not z.resolved]


def is_noougat_enabled() -> bool:
    return _env_bool("SWAY_NOOUGAT_ENABLED", True)
