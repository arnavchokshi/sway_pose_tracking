"""
Global appearance-free tracklet stitching (lightweight alternative to neural AFLink).

Greedy merge when track B starts shortly after track A ends and end/start centers are close.
Enable with SWAY_GLOBAL_LINK=1.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

TrackEntry = Tuple[int, Tuple[float, float, float, float], float]


def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _track_span(raw_tracks: Dict[int, List[TrackEntry]], tid: int):
    ent = sorted(raw_tracks.get(tid, []), key=lambda e: e[0])
    if not ent:
        return None
    return {
        "start": ent[0][0],
        "end": ent[-1][0],
        "first_box": ent[0][1],
        "last_box": ent[-1][1],
        "ent": ent,
    }


def heuristic_global_stitch(
    raw_tracks: Dict[int, List[TrackEntry]],
    total_frames: int,
    max_temporal_gap: int = 30,
    max_center_dist_frac: float = 0.35,
) -> Dict[int, List[TrackEntry]]:
    """Merge B into A if A ends before B starts, small gap, nearby centers (fraction of frame diagonal)."""
    if len(raw_tracks) < 2:
        return raw_tracks
    diag = float(np.hypot(1920, 1080))
    max_dist = max_center_dist_frac * diag

    changed = True
    while changed:
        changed = False
        ids = sorted(raw_tracks.keys())
        for tid_a in ids:
            if tid_a not in raw_tracks:
                continue
            sa = _track_span(raw_tracks, tid_a)
            if sa is None:
                continue
            end_a = sa["end"]
            ent_a = sa["ent"]
            ca = _center(sa["last_box"])
            best_b = None
            best_gap = 10**9
            for tid_b in ids:
                if tid_b == tid_a or tid_b not in raw_tracks:
                    continue
                sb = _track_span(raw_tracks, tid_b)
                if sb is None:
                    continue
                start_b = sb["start"]
                if start_b <= end_a:
                    continue
                gap = start_b - end_a - 1
                if gap < 0 or gap > max_temporal_gap:
                    continue
                cb = _center(sb["first_box"])
                dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
                if dist > max_dist:
                    continue
                if gap < best_gap:
                    best_gap = gap
                    best_b = (tid_b, sb["ent"])
            if best_b is not None:
                tid_b, ent_b = best_b
                raw_tracks[tid_a] = sorted(ent_a + ent_b, key=lambda e: e[0])
                del raw_tracks[tid_b]
                changed = True
                break
    return raw_tracks
