"""
Global appearance-free tracklet stitching after dormant merges.

- Default when ``SWAY_GLOBAL_LINK=1``: **StrongSORT AFLink** (neural) if
  ``models/AFLink_epoch20.pth`` exists or ``SWAY_AFLINK_WEIGHTS`` points to a file;
  otherwise the legacy **heuristic** spatial/temporal stitch.
- Set ``SWAY_GLOBAL_AFLINK=0`` (or ``heuristic``) to force the heuristic even when
  weights are present.
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from sway.track_observation import TrackObservation, coerce_observation

TrackEntry = Union[
    Tuple[int, Tuple[float, float, float, float], float],
    TrackObservation,
]


def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _track_span(raw_tracks: Dict[int, List[TrackEntry]], tid: int):
    ent = sorted(raw_tracks.get(tid, []), key=lambda e: coerce_observation(e).frame_idx)
    if not ent:
        return None
    first = coerce_observation(ent[0])
    last = coerce_observation(ent[-1])
    return {
        "start": first.frame_idx,
        "end": last.frame_idx,
        "first_box": first.bbox,
        "last_box": last.bbox,
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
                raw_tracks[tid_a] = sorted(ent_a + ent_b, key=lambda e: coerce_observation(e).frame_idx)
                del raw_tracks[tid_b]
                changed = True
                break
    return raw_tracks


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


_GLOBAL_LINK_WARNED_MISSING_AFLINK_WEIGHTS = False


def resolve_aflink_weights() -> Path:
    env = os.environ.get("SWAY_AFLINK_WEIGHTS", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    return _repo_root() / "models" / "AFLink_epoch20.pth"


def _force_heuristic_global_link() -> bool:
    v = os.environ.get("SWAY_GLOBAL_AFLINK", "").strip().lower()
    return v in ("0", "false", "no", "off", "heuristic")


def raw_tracks_to_mot_array(
    raw_tracks: Dict[int, List[TrackEntry]],
) -> Tuple[np.ndarray, List[TrackObservation]]:
    """
    MOT-style rows: frame, id, x, y, w, h, conf, obs_row_idx, -1, -1.
    Column 7 stores the index into the parallel ``observations`` list for round-trip.
    """
    observations: List[TrackObservation] = []
    rows: List[List[float]] = []
    for tid in sorted(raw_tracks.keys()):
        for ent in sorted(raw_tracks[tid], key=lambda e: coerce_observation(e).frame_idx):
            obs = coerce_observation(ent)
            x1, y1, x2, y2 = obs.bbox
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            ridx = len(observations)
            observations.append(obs)
            rows.append(
                [
                    float(obs.frame_idx),
                    float(tid),
                    float(x1),
                    float(y1),
                    w,
                    h,
                    float(obs.conf),
                    float(ridx),
                    -1.0,
                    -1.0,
                ]
            )
    if not rows:
        return np.zeros((0, 10), dtype=np.float64), observations
    return np.array(rows, dtype=np.float64), observations


def mot_array_to_raw_tracks(
    linked: np.ndarray,
    observations: List[TrackObservation],
) -> Dict[int, List[TrackEntry]]:
    if linked.size == 0:
        return {}
    out: Dict[int, List[TrackObservation]] = defaultdict(list)
    for row in linked:
        tid = int(row[1])
        ridx = int(round(float(row[7])))
        if 0 <= ridx < len(observations):
            out[tid].append(observations[ridx])
    return {k: sorted(out[k], key=lambda o: o.frame_idx) for k in sorted(out.keys())}


def neural_global_stitch(
    raw_tracks: Dict[int, List[TrackEntry]],
    total_frames: int,
    path_AFLink: str,
    thrT: Tuple[int, int] = (0, 30),
    thrS: int = 75,
    thrP: float = 0.05,
) -> Dict[int, List[TrackEntry]]:
    """
    StrongSORT AFLink global association (no appearance embeddings).
    ``total_frames`` reserved for API parity with the heuristic stitch.
    """
    _ = total_frames
    if len(raw_tracks) < 2:
        return raw_tracks
    from sway.aflink import AFLink

    mot, observations = raw_tracks_to_mot_array(raw_tracks)
    if mot.shape[0] == 0:
        return raw_tracks
    linker = AFLink(
        path_AFLink=path_AFLink,
        thrT=thrT,
        thrS=thrS,
        thrP=thrP,
    )
    linked = linker.link(mot)
    return mot_array_to_raw_tracks(linked, observations)


def maybe_global_stitch(
    raw_tracks: Dict[int, List[TrackEntry]],
    total_frames: int,
) -> Dict[int, List[TrackEntry]]:
    """
    Neural AFLink when weights are available (unless ``SWAY_GLOBAL_AFLINK`` forces heuristic);
    otherwise heuristic_global_stitch.
    """
    global _GLOBAL_LINK_WARNED_MISSING_AFLINK_WEIGHTS
    path = resolve_aflink_weights()
    if path.is_file() and not _force_heuristic_global_link():
        try:
            return neural_global_stitch(raw_tracks, total_frames, str(path))
        except Exception as exc:
            warnings.warn(
                f"SWAY_GLOBAL_LINK: AFLink failed ({exc}); falling back to heuristic_global_stitch.",
                RuntimeWarning,
                stacklevel=2,
            )
    elif (
        os.environ.get("SWAY_GLOBAL_LINK", "").lower() in ("1", "true", "yes")
        and not path.is_file()
        and not _force_heuristic_global_link()
        and not _GLOBAL_LINK_WARNED_MISSING_AFLINK_WEIGHTS
    ):
        _GLOBAL_LINK_WARNED_MISSING_AFLINK_WEIGHTS = True
        warnings.warn(
            f"SWAY_GLOBAL_LINK: AFLink weights not found at {path}; "
            "using heuristic_global_stitch. Place AFLink_epoch20.pth from StrongSORT "
            "under models/ or set SWAY_AFLINK_WEIGHTS.",
            UserWarning,
            stacklevel=2,
        )
    return heuristic_global_stitch(raw_tracks, total_frames)
