"""
Group-Split Hungarian and DP Collision Solver (PLAN_15)

Replaces greedy sequential matching during group coalescence events with
global optimization. When N tracks merge and N detections re-emerge, solves
identity assignment as a single Hungarian operation over the N×N distance matrix.

For ≤5 tracks, uses exhaustive DP permutation search (N! ≤ 120).

Env:
  SWAY_COLLISION_SOLVER               – greedy | hungarian | dp (default hungarian)
  SWAY_COLLISION_MIN_TRACKS           – min tracks for global solver (default 3)
  SWAY_COLLISION_DP_MAX_PERMUTATIONS  – cap on DP search (default 120 = 5!)
  SWAY_COALESCENCE_IOU_THRESH         – IoU for coalescence detection (default 0.85)
  SWAY_COALESCENCE_CONSECUTIVE_FRAMES – consecutive frames for coalescence (default 8)
"""

from __future__ import annotations

import copy
import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


@dataclass
class CoalescenceEvent:
    """A group coalescence event: multiple tracks merging."""
    track_ids: List[int]
    entry_frame: int
    frozen_embeddings: Dict[int, object] = field(default_factory=dict)
    exit_frame: Optional[int] = None
    resolved: bool = False
    iou_counter: int = 0  # consecutive frames above threshold


class CoalescenceDetector:
    """Detects when tracks coalesce (merge) and when they split."""

    def __init__(
        self,
        iou_thresh: Optional[float] = None,
        consecutive_frames: Optional[int] = None,
    ):
        self.iou_thresh = iou_thresh or _env_float("SWAY_COALESCENCE_IOU_THRESH", 0.85)
        self.consecutive_frames = consecutive_frames or _env_int(
            "SWAY_COALESCENCE_CONSECUTIVE_FRAMES", 8
        )
        self._pending: Dict[frozenset, int] = {}  # pair → consecutive count
        self._active_events: List[CoalescenceEvent] = []

    def check(
        self,
        bboxes: Dict[int, np.ndarray],
        frame_idx: int,
        galleries: Optional[Dict[int, object]] = None,
    ) -> List[CoalescenceEvent]:
        """Check for new coalescence events.

        Returns newly detected events (not previously active).
        """
        new_events: List[CoalescenceEvent] = []
        track_ids = list(bboxes.keys())
        n = len(track_ids)

        coalescing_pairs: List[frozenset] = []

        for i in range(n):
            for j in range(i + 1, n):
                tid_a, tid_b = track_ids[i], track_ids[j]
                iou = _box_iou(bboxes[tid_a], bboxes[tid_b])

                pair = frozenset({tid_a, tid_b})
                if iou > self.iou_thresh:
                    self._pending[pair] = self._pending.get(pair, 0) + 1
                    if self._pending[pair] >= self.consecutive_frames:
                        coalescing_pairs.append(pair)
                else:
                    self._pending.pop(pair, None)

        # Merge overlapping pairs into events
        if coalescing_pairs:
            groups = self._merge_pairs(coalescing_pairs)
            for group in groups:
                if self._is_already_active(group):
                    continue

                frozen = {}
                if galleries:
                    for tid in group:
                        if tid in galleries:
                            frozen[tid] = copy.deepcopy(galleries[tid])

                event = CoalescenceEvent(
                    track_ids=sorted(group),
                    entry_frame=frame_idx,
                    frozen_embeddings=frozen,
                )
                self._active_events.append(event)
                new_events.append(event)

                logger.info(
                    "Coalescence detected: tracks %s at frame %d",
                    event.track_ids, frame_idx,
                )

        return new_events

    def check_exits(
        self, bboxes: Dict[int, np.ndarray], frame_idx: int
    ) -> List[CoalescenceEvent]:
        """Detect coalescence exit events (hysteresis at 70% of entry threshold)."""
        exit_thresh = self.iou_thresh * 0.7
        exited: List[CoalescenceEvent] = []

        for event in self._active_events:
            if event.resolved:
                continue

            all_separated = True
            for i, tid_a in enumerate(event.track_ids):
                for tid_b in event.track_ids[i + 1:]:
                    if tid_a in bboxes and tid_b in bboxes:
                        iou = _box_iou(bboxes[tid_a], bboxes[tid_b])
                        if iou >= exit_thresh:
                            all_separated = False
                            break
                if not all_separated:
                    break

            if all_separated:
                event.resolved = True
                event.exit_frame = frame_idx
                exited.append(event)
                logger.info(
                    "Coalescence exit: tracks %s at frame %d",
                    event.track_ids, frame_idx,
                )

        return exited

    def _merge_pairs(self, pairs: List[frozenset]) -> List[List[int]]:
        """Merge overlapping pairs into groups using union-find."""
        parent: Dict[int, int] = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for pair in pairs:
            items = list(pair)
            for i in range(1, len(items)):
                union(items[0], items[i])

        groups: Dict[int, List[int]] = {}
        for item in parent:
            root = find(item)
            groups.setdefault(root, []).append(item)

        return list(groups.values())

    def _is_already_active(self, group: List[int]) -> bool:
        group_set = set(group)
        for event in self._active_events:
            if not event.resolved and set(event.track_ids) == group_set:
                return True
        return False


def solve_collision(
    event: CoalescenceEvent,
    exit_embeddings: List[object],
    exit_track_ids: List[int],
    fusion_engine=None,
    solver: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Solve the identity assignment at coalescence exit.

    Args:
        event: the coalescence event with frozen embeddings.
        exit_embeddings: fresh features at exit for each detection.
        exit_track_ids: track IDs of exit detections.
        fusion_engine: ReIDFusionEngine for computing distances.
        solver: override solver choice.

    Returns:
        List of (frozen_track_id, exit_track_id) pairs.
    """
    solver = solver or _env_str("SWAY_COLLISION_SOLVER", "hungarian")
    min_tracks = _env_int("SWAY_COLLISION_MIN_TRACKS", 3)

    n_frozen = len(event.track_ids)
    n_exit = len(exit_track_ids)

    if n_frozen < min_tracks or n_exit < 2:
        return _solve_greedy(event, exit_embeddings, exit_track_ids, fusion_engine)

    dp_max_perms = _env_int("SWAY_COLLISION_DP_MAX_PERMUTATIONS", 120)
    dp_max_tracks = 5  # 5! = 120
    # If dp_max_perms is raised, allow larger N (cap at 7! = 5040)
    import math
    while dp_max_tracks < 7 and math.factorial(dp_max_tracks + 1) <= dp_max_perms:
        dp_max_tracks += 1

    if solver == "dp" and n_frozen <= dp_max_tracks:
        return _solve_dp(event, exit_embeddings, exit_track_ids, fusion_engine)
    elif solver in ("hungarian", "dp"):
        return _solve_hungarian(event, exit_embeddings, exit_track_ids, fusion_engine)
    else:
        return _solve_greedy(event, exit_embeddings, exit_track_ids, fusion_engine)


def _build_cost_matrix(
    event: CoalescenceEvent,
    exit_embeddings: List[object],
    exit_track_ids: List[int],
    fusion_engine,
) -> np.ndarray:
    """Build N×M cost matrix of re-ID distances."""
    n = len(event.track_ids)
    m = len(exit_track_ids)
    cost = np.ones((n, m), dtype=np.float64)
    if fusion_engine is None or not exit_embeddings:
        for i in range(n):
            for j in range(m):
                cost[i, j] = 0.5
        return cost

    for j in range(min(m, len(exit_embeddings))):
        q = exit_embeddings[j]
        try:
            match_id, conf = fusion_engine.match(q)
        except Exception:
            match_id, conf = -1, 0.5
        for i, frozen_id in enumerate(event.track_ids):
            if match_id == frozen_id:
                cost[i, j] = max(0.0, 1.0 - float(conf))
            else:
                cost[i, j] = min(1.0, 0.75 + 0.25 * (1.0 - float(conf)))

    return cost


def _solve_hungarian(event, exit_embs, exit_ids, fusion) -> List[Tuple[int, int]]:
    cost = _build_cost_matrix(event, exit_embs, exit_ids, fusion)
    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = []
    for r, c in zip(row_ind, col_ind):
        if r < len(event.track_ids) and c < len(exit_ids):
            assignments.append((event.track_ids[r], exit_ids[c]))

    return assignments


def _solve_dp(event, exit_embs, exit_ids, fusion) -> List[Tuple[int, int]]:
    """Exhaustive DP: enumerate all N! permutations for N ≤ 5."""
    cost = _build_cost_matrix(event, exit_embs, exit_ids, fusion)
    n = min(len(event.track_ids), len(exit_ids))

    best_cost = float("inf")
    best_perm = None

    for perm in itertools.permutations(range(n)):
        total = sum(cost[i, perm[i]] for i in range(n))
        if total < best_cost:
            best_cost = total
            best_perm = perm

    if best_perm is None:
        return []

    return [(event.track_ids[i], exit_ids[best_perm[i]]) for i in range(n)]


def _solve_greedy(event, exit_embs, exit_ids, fusion) -> List[Tuple[int, int]]:
    """Greedy sequential matching (baseline)."""
    assignments = []
    used_exit = set()

    for frozen_id in event.track_ids:
        best_j = -1
        best_dist = float("inf")
        for j, eid in enumerate(exit_ids):
            if j not in used_exit:
                if best_j < 0:
                    best_j = j
                    best_dist = 0.5
        if best_j >= 0:
            assignments.append((frozen_id, exit_ids[best_j]))
            used_exit.add(best_j)

    return assignments


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
