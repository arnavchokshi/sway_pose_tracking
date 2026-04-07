"""
Post-stitch track statistics for golden / regression checks.

``main.py`` always writes ``track_stats.json`` under the run output directory after Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from sway.track_observation import TrackObservation, coerce_observation

TrackEntry = Union[
    TrackObservation,
    tuple,
]


def compute_track_quality_stats(
    raw_tracks: Dict[int, List[TrackEntry]],
    total_frames: int,
    yolo_stride: int,
) -> Dict[str, Any]:
    """
    Lightweight metrics for ID-stability / coverage heuristics (not MOT ground truth).
    """
    obs_counts: List[int] = []
    spans: List[int] = []
    first_frames: List[int] = []
    last_frames: List[int] = []

    for _tid, ents in raw_tracks.items():
        if not ents:
            continue
        obs = [coerce_observation(e) for e in ents]
        frames = sorted(int(o.frame_idx) for o in obs)
        first, last = frames[0], frames[-1]
        first_frames.append(first)
        last_frames.append(last)
        obs_counts.append(len(obs))
        spans.append(last - first + 1)

    n = len(obs_counts)
    arr_obs = np.array(obs_counts, dtype=np.float64) if n else np.zeros(0)
    arr_span = np.array(spans, dtype=np.float64) if n else np.zeros(0)

    return {
        "schema_version": 1,
        "total_frames": int(total_frames),
        "yolo_detection_stride": int(yolo_stride),
        "num_tracks": int(len(raw_tracks)),
        "tracks_with_observations": int(n),
        "total_observations": int(arr_obs.sum()) if n else 0,
        "median_observations_per_track": float(np.median(arr_obs)) if n else 0.0,
        "median_temporal_span_frames": float(np.median(arr_span)) if n else 0.0,
        "p10_temporal_span_frames": float(np.percentile(arr_span, 10)) if n else 0.0,
        "p90_temporal_span_frames": float(np.percentile(arr_span, 90)) if n else 0.0,
        "earliest_track_start_frame": int(min(first_frames)) if first_frames else None,
        "latest_track_end_frame": int(max(last_frames)) if last_frames else None,
    }


def write_track_stats_json(path: Path, stats: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
