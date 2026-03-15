"""
Scoring Module — Spatio-Temporal Sync (V3.0)

V3.0: circmean/circmedian Group Truth, cDTW (Sakoe-Chiba band=3), ripple->NaN.
- Group Truth: scipy.stats.circmean (angular-safe) + 5-frame rolling median
- Ripple: std > 30° → deviations = NaN (Gray), not 0
- Shape/Timing: cDTW with Sakoe-Chiba band=3 (faster, more accurate than FastDTW)
- Per-joint thresholds: Spine/Hips 10°/20°, Elbows/Knees 20°/35°
"""

from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from scipy.stats import circmean

from kinematics import (
    JOINT_NAMES,
    compute_joint_angles_vectorized,
)

# Confidence threshold for occlusion / low-quality keypoints
CONFIDENCE_THRESHOLD = 0.3

# V3.0: Ripple — std > 30° → deviations = NaN (Gray)
RIPPLE_STD_THRESHOLD = 30.0

# Group Truth: rolling median window
CONSENSUS_ROLLING_WINDOW = 5

# V3.0: cDTW 30-frame window, Sakoe-Chiba band
DTW_WINDOW_SIZE = 30
DTW_SAKOE_CHIBA_BAND = 3

# Timing threshold: > 2 frames lag/lead = Off-Beat
TIMING_ERROR_THRESHOLD = 2

# V3.0 Per-joint shape thresholds (degrees)
SHAPE_GREEN_SPINE = 10.0   # Spine/Hips: Green < 10°
SHAPE_RED_SPINE = 20.0     # Spine/Hips: Red > 20°
SHAPE_GREEN_LIMB = 20.0    # Elbows/Knees: Green < 20°
SHAPE_RED_LIMB = 35.0      # Elbows/Knees: Red > 35°


def _build_keypoints_array(
    all_frame_data: List[Dict],
) -> tuple:
    """Compile smoothed keypoints into (F, T, 17, 3)."""
    num_frames = len(all_frame_data)
    all_track_ids = set()
    for fd in all_frame_data:
        all_track_ids.update(fd.get("track_ids", []))
    all_track_ids = sorted(all_track_ids)
    num_tracks = len(all_track_ids)
    tid_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}

    kpts_arr = np.full((num_frames, num_tracks, 17, 3), np.nan, dtype=np.float64)
    track_ids_per_frame: List[List[int]] = []

    for f in range(num_frames):
        fd = all_frame_data[f]
        poses = fd.get("poses", {})
        track_ids = fd.get("track_ids", [])
        track_ids_per_frame.append(list(track_ids))
        for tid in track_ids:
            if tid not in tid_to_idx:
                continue
            t_idx = tid_to_idx[tid]
            if tid not in poses:
                continue
            data = poses[tid]
            kpts = data["keypoints"]
            if kpts.shape[0] < 17:
                continue
            kpts_arr[f, t_idx, :, :] = kpts[:17, :]

    return kpts_arr, tid_to_idx, track_ids_per_frame


def _rolling_median_1d(arr: np.ndarray, window: int = CONSENSUS_ROLLING_WINDOW) -> np.ndarray:
    """Apply 5-frame rolling median to 1D array."""
    if window <= 1 or len(arr) == 0:
        return arr.copy()
    half = window // 2
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = np.nanmedian(arr[lo:hi])
    return out


def _circmean_1d(arr: np.ndarray, high: float = 180.0, low: float = 0.0) -> float:
    """Circular mean over valid (non-NaN) values. Angles in degrees [0, 180]."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan
    return float(circmean(valid, high=high, low=low))


def _compute_group_truth(angles: np.ndarray) -> np.ndarray:
    """
    V3.0: Group Truth using circular mean (angular-safe) + 5-frame rolling median.
    angles: (F, T, 6). Returns (F, 6).
    """
    F, T, _ = angles.shape
    consensus = np.full((F, 6), np.nan, dtype=np.float64)
    for j in range(6):
        for f in range(F):
            col = angles[f, :, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                consensus[f, j] = _circmean_1d(col, high=180.0, low=0.0)
        consensus[:, j] = _rolling_median_1d(consensus[:, j], CONSENSUS_ROLLING_WINDOW)
    return consensus


def _compute_shape_and_timing_cdtw(
    track_window: np.ndarray,
    group_window: np.ndarray,
    sakoe_chiba_band: int = DTW_SAKOE_CHIBA_BAND,
) -> Tuple[float, float]:
    """
    V3.0: cDTW with Sakoe-Chiba band. Shape error = mean aligned diff, timing = mean phase shift.
    """
    try:
        from pyts.metrics import dtw
    except ImportError:
        # Fallback: simple mean diff if pyts unavailable
        diff = np.nanmean(np.abs(track_window - group_window))
        return float(diff) if not np.isnan(diff) else float("nan"), 0.0

    # NaN handling: fill with group for track, nanmean for group
    track_filled = np.where(np.isnan(track_window), group_window, track_window)
    group_filled = np.where(np.isnan(group_window), np.nanmean(group_window), group_window)
    if np.all(np.isnan(track_filled)) or np.all(np.isnan(group_filled)):
        return float("nan"), 0.0
    track_1d = np.nan_to_num(track_filled, nan=0.0).ravel()
    group_1d = np.nan_to_num(group_filled, nan=0.0).ravel()

    # pyts dtw expects 1D arrays (n_timestamps,)
    dist, path = dtw(
        track_1d,
        group_1d,
        method="sakoechiba",
        options={"window_size": sakoe_chiba_band},
        return_path=True,
    )
    if path is None or path.size == 0:
        return float("nan"), 0.0

    # path: (2, path_len) — row 0 = track indices, row 1 = group indices
    i_idx = np.asarray(path[0]).astype(int)
    j_idx = np.asarray(path[1]).astype(int)
    i_idx = np.clip(i_idx, 0, len(track_window) - 1)
    j_idx = np.clip(j_idx, 0, len(group_window) - 1)

    shape_diffs = [
        abs(track_window[i] - group_window[j])
        for (i, j) in zip(i_idx, j_idx)
        if not (np.isnan(track_window[i]) or np.isnan(group_window[j]))
    ]
    shape_err = float(np.mean(shape_diffs)) if shape_diffs else float(dist / max(1, len(i_idx)))
    timing_err = float(np.mean(j_idx.astype(float) - i_idx.astype(float))) if len(i_idx) > 0 else 0.0
    return shape_err, timing_err


def process_all_frames_scoring_vectorized(
    all_frame_data: List[Dict],
) -> Optional[Dict[str, Any]]:
    """
    V3.0 Group Truth (circmean) + cDTW scoring.
    Returns track_angles, consensus_angles, deviations, shape_errors, timing_errors.
    Ripple: std > 30° → deviations = NaN (Gray).
    """
    if not all_frame_data:
        return None

    kpts_arr, tid_to_idx, track_ids_per_frame = _build_keypoints_array(all_frame_data)
    num_frames, num_tracks, _, _ = kpts_arr.shape

    angles = compute_joint_angles_vectorized(kpts_arr, confidence_threshold=CONFIDENCE_THRESHOLD)
    group_truth = _compute_group_truth(angles)

    # V3.0 Ripple: std > 30° → deviations = NaN (not 0)
    ripple_mask = np.nanstd(angles, axis=1) > RIPPLE_STD_THRESHOLD
    consensus_bc = np.broadcast_to(group_truth[:, np.newaxis, :], (num_frames, num_tracks, 6))
    deviations = np.abs(angles - consensus_bc)
    deviations = np.where(
        np.broadcast_to(ripple_mask[:, np.newaxis, :], deviations.shape),
        np.nan,
        deviations,
    )
    deviations = np.where(np.isnan(angles), np.nan, deviations)

    # Shape and timing: cDTW per (track, joint)
    shape_errors = np.full((num_tracks, 6), np.nan, dtype=np.float64)
    timing_errors = np.zeros((num_tracks, 6), dtype=np.float64)
    w = min(DTW_WINDOW_SIZE, num_frames)
    for t in range(num_tracks):
        for j in range(6):
            start = max(0, num_frames - w)
            track_window = angles[start:, t, j].copy()
            group_window = group_truth[start:, j].copy()
            if len(track_window) < 2:
                continue
            if np.all(np.isnan(track_window)):
                continue
            se, te = _compute_shape_and_timing_cdtw(track_window, group_window)
            shape_errors[t, j] = se
            timing_errors[t, j] = te

    # Convert to per-frame structure
    track_angles_list = []
    consensus_angles_list = []
    deviations_list = []
    shape_errors_list = []
    timing_errors_list = []

    for f in range(num_frames):
        track_ids = track_ids_per_frame[f]
        ta = {}
        dev = {}
        se_dict = {}
        te_dict = {}
        for tid in track_ids:
            if tid not in tid_to_idx:
                continue
            t_idx = tid_to_idx[tid]
            ta[tid] = {}
            dev[tid] = {}
            se_dict[tid] = {}
            te_dict[tid] = {}
            for j, jname in enumerate(JOINT_NAMES):
                v = angles[f, t_idx, j]
                ta[tid][jname] = None if np.isnan(v) else float(v)
                d = deviations[f, t_idx, j]
                dev[tid][f"{jname}_diff"] = float("nan") if np.isnan(d) else float(d)
                se_dict[tid][f"{jname}_shape"] = float(shape_errors[t_idx, j]) if not np.isnan(shape_errors[t_idx, j]) else float("nan")
                te_dict[tid][f"{jname}_timing"] = float(timing_errors[t_idx, j])

        ca = {}
        for j, jname in enumerate(JOINT_NAMES):
            std_val = np.nanstd(angles[f, :, j]) if np.any(~np.isnan(angles[f, :, j])) else 0.0
            if std_val > RIPPLE_STD_THRESHOLD:
                ca[jname] = None
            else:
                ca[jname] = float(group_truth[f, j]) if not np.isnan(group_truth[f, j]) else None

        track_angles_list.append(ta)
        consensus_angles_list.append(ca)
        deviations_list.append(dev)
        shape_errors_list.append(se_dict)
        timing_errors_list.append(te_dict)

    return {
        "track_angles": track_angles_list,
        "consensus_angles": consensus_angles_list,
        "deviations": deviations_list,
        "shape_errors": shape_errors_list,
        "timing_errors": timing_errors_list,
    }


# --- Legacy compatibility ---

def calculate_joint_angles(*args, **kwargs):
    from kinematics import calculate_joint_angles as k_calc
    return k_calc(*args, **kwargs)


def compute_frame_consensus(all_tracks_angles: Dict) -> Dict:
    """Legacy consensus (median)."""
    from kinematics import JOINT_NAMES
    consensus = {}
    for jname in JOINT_NAMES:
        values = [a.get(jname) for a in all_tracks_angles.values()
                  if a.get(jname) is not None and not (isinstance(a.get(jname), float) and np.isnan(a.get(jname)))]
        consensus[jname] = float(np.median(values)) if values else float("nan")
        if values and np.std(values) > RIPPLE_STD_THRESHOLD:
            consensus[jname] = None
    return consensus


def score_frame_deviations(all_tracks_angles: Dict, consensus_angles: Dict) -> Dict:
    """Legacy deviations."""
    result = {}
    for tid, angles in all_tracks_angles.items():
        diffs = {}
        for jname, consensus_val in consensus_angles.items():
            ang = angles.get(jname)
            if ang is None or (isinstance(ang, float) and np.isnan(ang)):
                diffs[f"{jname}_diff"] = float("nan")
            elif consensus_val is None:
                diffs[f"{jname}_diff"] = float("nan")
            else:
                diffs[f"{jname}_diff"] = abs(float(ang) - float(consensus_val))
        result[tid] = diffs
    return result


def process_frame_scoring(poses: Dict):
    """Legacy per-frame scoring."""
    all_tracks_angles = {}
    for tid, data in poses.items():
        all_tracks_angles[tid] = calculate_joint_angles(
            data["keypoints"], data.get("scores")
        )
    consensus = compute_frame_consensus(all_tracks_angles)
    deviations = score_frame_deviations(all_tracks_angles, consensus)
    return all_tracks_angles, consensus, deviations
