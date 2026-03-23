"""
Dormant-style relinking: merge track B into A when BoT-SORT assigned a new ID after a gap
longer than track_buffer but within max_gap, using motion extrapolation + IoU (no pose in tracker phase).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

TrackEntry = Tuple[int, Tuple[float, float, float, float], float]


def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def _extrapolate_box(last_box: Tuple[float, float, float, float], vx: float, vy: float, dt: float):
    """Shift box by velocity * dt (pixels per frame)."""
    dx, dy = vx * dt, vy * dt
    return (
        last_box[0] + dx,
        last_box[1] + dy,
        last_box[2] + dx,
        last_box[3] + dy,
    )


def apply_dormant_merges(
    raw_tracks: Dict[int, List[TrackEntry]],
    total_frames: int,
    track_buffer: int = 90,
    max_gap: int = 150,
    spatial_sigma: float = 2.0,
    min_iou: float = 0.2,
) -> Dict[int, List[TrackEntry]]:
    if not raw_tracks:
        return raw_tracks

    def seg_meta(tid: int):
        ent = sorted(raw_tracks[tid], key=lambda e: e[0])
        if not ent:
            return None
        return {
            "tid": tid,
            "start": ent[0][0],
            "end": ent[-1][0],
            "entries": ent,
        }

    changed = True
    while changed:
        changed = False
        metas = [seg_meta(t) for t in list(raw_tracks.keys())]
        metas = [m for m in metas if m is not None]
        metas.sort(key=lambda m: m["start"])
        for A in metas:
            if A["tid"] not in raw_tracks:
                continue
            ent_a = sorted(raw_tracks[A["tid"]], key=lambda e: e[0])
            if len(ent_a) < 2:
                continue
            (_, last_box, _), (_, prev_box, _) = ent_a[-1], ent_a[-2]
            dt_obs = max(1, ent_a[-1][0] - ent_a[-2][0])
            c0, c1 = _center(prev_box), _center(last_box)
            vx, vy = (c1[0] - c0[0]) / dt_obs, (c1[1] - c0[1]) / dt_obs
            for B in metas:
                if B["tid"] == A["tid"] or B["tid"] not in raw_tracks or A["tid"] not in raw_tracks:
                    continue
                if B["start"] <= A["end"]:
                    continue
                gap = B["start"] - A["end"] - 1
                if gap <= track_buffer or gap > max_gap:
                    continue
                ent_b = sorted(raw_tracks[B["tid"]], key=lambda e: e[0])
                first_box = ent_b[0][1]
                pred_box = _extrapolate_box(last_box, vx, vy, float(gap + 1))
                diag = max(
                    1.0,
                    np.hypot(first_box[2] - first_box[0], first_box[3] - first_box[1]),
                )
                pc = _center(pred_box)
                bc = _center(first_box)
                dist = float(np.hypot(pc[0] - bc[0], pc[1] - bc[1]))
                if dist > spatial_sigma * diag:
                    continue
                if _iou(pred_box, first_box) < min_iou:
                    continue
                # Merge B -> A
                merged_list = ent_a + ent_b
                raw_tracks[A["tid"]] = merged_list
                del raw_tracks[B["tid"]]
                changed = True
                break
            if changed:
                break
    return raw_tracks
