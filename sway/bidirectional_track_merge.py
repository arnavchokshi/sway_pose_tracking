"""
Optional second tracking pass on a time-reversed copy of the video, merged into forward tracks.

**Default: off.** Enable only with ``SWAY_BIDIRECTIONAL_TRACK_PASS=1`` (or true/yes).
Requires ``ffmpeg`` on PATH (uses ``-vf reverse``).

Roadmap: Phase C robustness — forward + backward association consensus without changing
defaults for everyone (~2× Phase 1–2 wall time when enabled).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sway.track_observation import TrackObservation, coerce_observation

TrackEntry = Any


def bidirectional_track_pass_enabled() -> bool:
    v = os.environ.get("SWAY_BIDIRECTIONAL_TRACK_PASS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def bidirectional_iou_threshold() -> float:
    v = os.environ.get("SWAY_BIDIRECTIONAL_IOU_THRESH", "").strip()
    if not v:
        return 0.45
    try:
        return float(v)
    except ValueError:
        return 0.45


def bidirectional_min_match_frames() -> int:
    v = os.environ.get("SWAY_BIDIRECTIONAL_MIN_MATCH_FRAMES", "").strip()
    if not v:
        return 4
    try:
        return max(1, int(v))
    except ValueError:
        return 4


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    b1 = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    u = a1 + b1 - inter
    return float(inter / u) if u > 0 else 0.0


def reverse_video_via_ffmpeg(src: Path, dst: Path) -> None:
    """Write a time-reversed copy of ``src`` to ``dst`` (video only, no audio)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "SWAY_BIDIRECTIONAL_TRACK_PASS is on but `ffmpeg` was not found on PATH. "
            "Install ffmpeg or set SWAY_BIDIRECTIONAL_TRACK_PASS=0."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        "reverse",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed while reversing video for bidirectional tracking:\n"
            f"{r.stderr or r.stdout or r.returncode}"
        )


def remap_reverse_pass_timeline(
    raw_tracks: Dict[int, List[TrackEntry]],
    total_frames: int,
) -> Dict[int, List[TrackObservation]]:
    """
    Reverse-pass tracker uses frame indices 0..N-1 along the reversed file.
    Map those to original timeline: orig = (total_frames - 1) - r.
    """
    out: Dict[int, List[TrackObservation]] = {}
    for tid, ents in raw_tracks.items():
        mapped: List[TrackObservation] = []
        for e in ents:
            obs = coerce_observation(e)
            nf = int(total_frames - 1 - obs.frame_idx)
            mapped.append(
                TrackObservation(
                    nf,
                    obs.bbox,
                    obs.conf,
                    obs.is_sam_refined,
                    obs.segmentation_mask,
                    obs.sam2_input_roi_xyxy,
                )
            )
        out[int(tid)] = sorted(mapped, key=lambda o: o.frame_idx)
    return out


def _observations_by_frame(
    raw: Dict[int, List[TrackEntry]],
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    by_f: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for tid, ents in raw.items():
        for e in ents:
            obs = coerce_observation(e)
            f = int(obs.frame_idx)
            box = np.array(obs.bbox[:4], dtype=np.float32)
            by_f.setdefault(f, []).append((int(tid), box))
    return by_f


def merge_forward_backward_tracks(
    forward: Dict[int, List[TrackEntry]],
    reverse_remapped: Dict[int, List[TrackObservation]],
    *,
    iou_threshold: float,
    min_match_frames: int,
) -> Dict[int, List[TrackObservation]]:
    """
    Keep forward track IDs as canonical. For each reverse track ID, find the forward ID
    with the most high-IoU co-occurring frames; merge if count >= min_match_frames.
    Unmatched reverse-only observations are kept under new numeric IDs.
    """
    if not reverse_remapped:
        return {k: [coerce_observation(e) for e in v] for k, v in forward.items()}

    fwd_norm: Dict[int, List[TrackObservation]] = {
        k: [coerce_observation(e) for e in v] for k, v in forward.items()
    }

    fwd_by_frame = _observations_by_frame(fwd_norm)
    rev_by_frame = _observations_by_frame(reverse_remapped)

    match_count: Dict[Tuple[int, int], int] = {}
    for f in rev_by_frame:
        if f not in fwd_by_frame:
            continue
        for tid_r, box_r in rev_by_frame[f]:
            for tid_f, box_f in fwd_by_frame[f]:
                if _iou_xyxy(box_r, box_f) >= iou_threshold:
                    key = (tid_r, tid_f)
                    match_count[key] = match_count.get(key, 0) + 1

    rev_to_fwd: Dict[int, int] = {}
    rev_ids = set(reverse_remapped.keys())
    for tid_r in rev_ids:
        best_f: int | None = None
        best_n = 0
        for (tr, tf), n in match_count.items():
            if tr == tid_r and n > best_n:
                best_n = n
                best_f = tf
        if best_f is not None and best_n >= min_match_frames:
            rev_to_fwd[tid_r] = best_f

    out: Dict[int, List[TrackObservation]] = {k: list(v) for k, v in fwd_norm.items()}
    frames_per_fwd: Dict[int, set] = {tid: {o.frame_idx for o in obs} for tid, obs in out.items()}

    next_new_id = max(out.keys(), default=0) + 1

    for tid_r, ents in reverse_remapped.items():
        target = rev_to_fwd.get(tid_r)
        if target is None:
            out[next_new_id] = list(ents)
            frames_per_fwd[next_new_id] = {o.frame_idx for o in ents}
            next_new_id += 1
            continue
        existing = frames_per_fwd.setdefault(target, set())
        for obs in ents:
            if obs.frame_idx in existing:
                continue
            out[target].append(obs)
            existing.add(obs.frame_idx)
        out[target].sort(key=lambda o: o.frame_idx)

    return out
