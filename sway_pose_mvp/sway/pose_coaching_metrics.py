"""
Coaching-oriented metrics from 2D pose caches: segment jerk (per dancer / phrase) and pairwise synchrony.

Expects pose_cache[frame_idx][dancer_id] as (17, 3) float32 with x, y in pixels and column 2 = ViTPose score.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# COCO-17 triplets (i, vertex j, k) — angle at joint j
SEGMENT_ANGLE_DEFS: Dict[str, Tuple[int, int, int]] = {
    "left_elbow": (5, 7, 9),
    "right_elbow": (6, 8, 10),
    "left_knee": (11, 13, 15),
    "right_knee": (12, 14, 16),
    "left_shoulder": (11, 5, 7),
    "right_shoulder": (12, 6, 8),
    "left_hip": (5, 11, 13),
    "right_hip": (6, 12, 14),
}

# Midpoint velocity jerk (pixels / s^3 scale — comparable within one clip)
SEGMENT_MIDPOINT_DEFS: Dict[str, Tuple[int, int]] = {
    "left_upper_arm": (5, 7),
    "left_forearm": (7, 9),
    "right_upper_arm": (6, 8),
    "right_forearm": (8, 10),
    "left_thigh": (11, 13),
    "left_shin": (13, 15),
    "right_thigh": (12, 14),
    "right_shin": (14, 16),
}


def _angle_at_b(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a.astype(np.float64) - b.astype(np.float64)
    bc = c.astype(np.float64) - b.astype(np.float64)
    n1 = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if n1 < 1e-6:
        return float("nan")
    cos = float(np.clip(np.dot(ba, bc) / n1, -1.0, 1.0))
    return float(math.degrees(math.acos(cos)))


def _series_angle(
    pose_cache: Dict[int, Dict[int, np.ndarray]],
    dancer_id: int,
    triplet: Tuple[int, int, int],
    *,
    min_conf: float = 0.2,
) -> np.ndarray:
    """Return angles per global frame index (NaN when missing)."""
    n_frames = max(pose_cache.keys(), default=-1) + 1
    out = np.full(n_frames, np.nan, dtype=np.float64)
    ia, ib, ic = triplet
    for f in range(n_frames):
        kp = pose_cache.get(f, {}).get(dancer_id)
        if kp is None or kp.shape[0] <= max(ia, ib, ic):
            continue
        if kp[ia, 2] < min_conf or kp[ib, 2] < min_conf or kp[ic, 2] < min_conf:
            continue
        out[f] = _angle_at_b(kp[ia, :2], kp[ib, :2], kp[ic, :2])
    return out


def _series_midpoint(
    pose_cache: Dict[int, Dict[int, np.ndarray]],
    dancer_id: int,
    j1: int,
    j2: int,
    *,
    min_conf: float = 0.2,
) -> np.ndarray:
    """(T, 2) midpoints, NaN rows when invalid."""
    n_frames = max(pose_cache.keys(), default=-1) + 1
    xy = np.full((n_frames, 2), np.nan, dtype=np.float64)
    for f in range(n_frames):
        kp = pose_cache.get(f, {}).get(dancer_id)
        if kp is None or kp.shape[0] <= max(j1, j2):
            continue
        if kp[j1, 2] < min_conf or kp[j2, 2] < min_conf:
            continue
        xy[f] = 0.5 * (kp[j1, :2] + kp[j2, :2])
    return xy


def rms_jerk_1d(signal: np.ndarray, fps: float) -> float:
    """RMS of discrete third derivative (jerk proxy), ignoring NaNs."""
    x = signal[np.isfinite(signal)].astype(np.float64)
    if x.size < 4:
        return 0.0
    # scale to seconds
    s = float(fps)
    d1 = np.diff(x) * s
    d2 = np.diff(d1) * s
    d3 = np.diff(d2) * s
    if d3.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(d3 * d3)))


def rms_jerk_midpoint_xy(mid_xy: np.ndarray, fps: float) -> float:
    """Combined jerk magnitude from (x,y) midpoint track."""
    if mid_xy.shape[0] < 4:
        return 0.0
    jx = rms_jerk_1d(mid_xy[:, 0], fps)
    jy = rms_jerk_1d(mid_xy[:, 1], fps)
    return float(math.sqrt(0.5 * (jx * jx + jy * jy)))


def _phrase_ranges(num_frames: int, phrase_len: int) -> List[Tuple[int, int, str]]:
    ranges: List[Tuple[int, int, str]] = []
    lo = 0
    pid = 0
    while lo < num_frames:
        hi = min(num_frames - 1, lo + phrase_len - 1)
        ranges.append((lo, hi, f"phrase_{pid}"))
        lo = hi + 1
        pid += 1
    return ranges


def _slice_phrase(sig: np.ndarray, lo: int, hi: int) -> np.ndarray:
    chunk = sig[lo : hi + 1]
    return chunk[np.isfinite(chunk)]


def cross_correlation_best_lag(
    a: np.ndarray,
    b: np.ndarray,
    max_lag: int = 15,
) -> Tuple[int, float]:
    """Mean-centered normalized correlation; search |lag| <= max_lag. Returns (best_lag, corr_at_lag)."""
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 8:
        return 0, 0.0
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    sa = float(np.std(aa)) or 1.0
    sb = float(np.std(bb)) or 1.0
    aa /= sa
    bb /= sb
    n = min(aa.size, bb.size)
    aa = aa[:n]
    bb = bb[:n]
    best_lag, best_c = 0, -1.1
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            ca, cb = aa, bb
        elif lag > 0:
            ca, cb = aa[lag:], bb[:-lag]
        else:
            m = -lag
            ca, cb = aa[:-m], bb[m:]
        if ca.size < 6:
            continue
        c = float(np.mean(ca * cb))
        if c > best_c:
            best_c = c
            best_lag = lag
    return best_lag, best_c


def build_coaching_motion_analysis(
    pose_cache: Dict[int, Dict[int, np.ndarray]],
    *,
    fps: float,
    phrase_frames: Optional[int] = None,
    min_conf: float = 0.2,
    sync_max_lag: int = 12,
) -> Dict[str, Any]:
    """
    Build nested metrics for coaching: per-dancer segment jerk by phrase, pairwise angle sync.
    """
    if not pose_cache:
        return {"error": "empty_pose_cache"}

    n_frames = max(pose_cache.keys(), default=-1) + 1
    if phrase_frames is None:
        env_ph = (os.environ.get("SWAY_COACHING_PHRASE_FRAMES", "") or "").strip()
        if env_ph:
            phrase_frames = max(1, int(env_ph))
        else:
            phrase_frames = max(1, int(round(float(fps) * 2.0)))

    dancer_ids = sorted({did for fc in pose_cache.values() for did in fc.keys()})
    phrases = _phrase_ranges(n_frames, phrase_frames)

    per_dancer: Dict[str, Any] = {}
    global_jerks: List[float] = []

    for did in dancer_ids:
        seg_angle_jerk: Dict[str, Any] = {}
        seg_mid_jerk: Dict[str, Any] = {}

        for seg_name, trip in SEGMENT_ANGLE_DEFS.items():
            ang = _series_angle(pose_cache, int(did), trip, min_conf=min_conf)
            by_phrase: Dict[str, float] = {}
            for lo, hi, plabel in phrases:
                chunk = _slice_phrase(ang, lo, hi)
                if chunk.size < 4:
                    by_phrase[plabel] = 0.0
                else:
                    jv = rms_jerk_1d(chunk, fps)
                    by_phrase[plabel] = round(jv, 4)
            overall = rms_jerk_1d(ang, fps)
            global_jerks.append(overall)
            seg_angle_jerk[seg_name] = {"overall_rms_jerk": round(overall, 4), "by_phrase": by_phrase}

        for seg_name, (j1, j2) in SEGMENT_MIDPOINT_DEFS.items():
            mid = _series_midpoint(pose_cache, int(did), j1, j2, min_conf=min_conf)
            by_phrase_m: Dict[str, float] = {}
            for lo, hi, plabel in phrases:
                sub = mid[lo : hi + 1]
                jv = rms_jerk_midpoint_xy(sub, fps)
                by_phrase_m[plabel] = round(jv, 4)
            overall_m = rms_jerk_midpoint_xy(mid, fps)
            seg_mid_jerk[seg_name] = {"overall_rms_jerk": round(overall_m, 4), "by_phrase": by_phrase_m}

        per_dancer[str(int(did))] = {
            "joint_angle_jerk": seg_angle_jerk,
            "segment_midpoint_jerk": seg_mid_jerk,
        }

    # Pairwise synchrony on elbow angles (primary coaching signal)
    sync_pairs: List[Dict[str, Any]] = []
    for i, da in enumerate(dancer_ids):
        for db in dancer_ids[i + 1 :]:
            row: Dict[str, Any] = {"dancer_a": int(da), "dancer_b": int(db), "features": {}}
            for feat_name, trip in [
                ("left_elbow_angle", SEGMENT_ANGLE_DEFS["left_elbow"]),
                ("right_elbow_angle", SEGMENT_ANGLE_DEFS["right_elbow"]),
                ("left_knee_angle", SEGMENT_ANGLE_DEFS["left_knee"]),
                ("right_knee_angle", SEGMENT_ANGLE_DEFS["right_knee"]),
            ]:
                a = _series_angle(pose_cache, int(da), trip, min_conf=min_conf)
                b = _series_angle(pose_cache, int(db), trip, min_conf=min_conf)
                lag, corr = cross_correlation_best_lag(a, b, max_lag=sync_max_lag)
                row["features"][feat_name] = {
                    "best_lag_frames": int(lag),
                    "correlation": round(corr, 4),
                    "interpretation": (
                        f"dancer_b lags dancer_a by {lag} frames"
                        if lag > 0
                        else (
                            f"dancer_a lags dancer_b by {-lag} frames"
                            if lag < 0
                            else "best alignment at zero lag"
                        )
                    ),
                }
            sync_pairs.append(row)

    return {
        "fps": float(fps),
        "num_frames": int(n_frames),
        "phrase_frames": int(phrase_frames),
        "phrases": [{"label": lab, "start": lo, "end": hi} for lo, hi, lab in phrases],
        "dancers": per_dancer,
        "mean_temporal_jerk_angle_proxy": round(float(np.mean(global_jerks)) if global_jerks else 0.0, 4),
        "pairwise_motion_sync": sync_pairs,
    }
