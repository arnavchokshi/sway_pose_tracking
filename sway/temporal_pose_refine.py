"""
Lightweight temporal refinement for COCO-17 poses (post ViTPose).

This is **not** the Poseidon architecture (multi-frame ViT backbone); it is a
confidence-weighted spatial smooth over neighboring frames for the same track,
inspired by the same ±window idea. Use when single-frame ViTPose is jittery.

Default on (CLI). Disable with ``--no-temporal-pose-refine`` or ``SWAY_TEMPORAL_POSE_REFINE=0``.
Explicit ``SWAY_TEMPORAL_POSE_REFINE=1`` forces on even if CLI disables.
Radius: ``--temporal-pose-radius`` or ``SWAY_TEMPORAL_POSE_RADIUS`` (default 2).
Production ``main.py`` / Lab set ``SWAY_TEMPORAL_POSE_RADIUS=2`` after params (§14.0.1) unless
``SWAY_UNLOCK_SMOOTH_TUNING=1``.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np


def want_temporal_pose_refine(cli_flag: bool) -> bool:
    """
    Tri-state env overrides CLI default:
      SWAY_TEMPORAL_POSE_REFINE=0|false|no|off  -> off
      SWAY_TEMPORAL_POSE_REFINE=1|true|yes|on   -> on
      unset                                     -> cli_flag (default True in main.py)
    """
    raw = os.environ.get("SWAY_TEMPORAL_POSE_REFINE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return bool(cli_flag)


def temporal_pose_radius(cli_value: int) -> int:
    raw = os.environ.get("SWAY_TEMPORAL_POSE_RADIUS", "").strip()
    if raw.isdigit():
        r = int(raw)
    else:
        r = int(cli_value)
    return max(0, min(8, r))


def apply_temporal_keypoint_smoothing(
    raw_poses_by_frame: List[Dict[int, dict]],
    *,
    radius: int = 2,
    min_visibility: float = 0.05,
) -> None:
    """
    In-place: for each frame and track, replace (x, y) with a weighted average
    over frames [f - radius, f + radius] where the track exists. Weights combine
    per-joint score and a small temporal falloff (1 / (1 + |Δf|)).

    Leaves ``keypoints[:, 2]`` (ViTPose visibility) and ``scores`` unchanged
    so downstream pruning thresholds stay calibrated.
    """
    n = len(raw_poses_by_frame)
    if n == 0 or radius < 1:
        return

    # Snapshot positions only (read-only during accumulation)
    xy_by_f_tid: List[Dict[int, np.ndarray]] = []
    sc_by_f_tid: List[Dict[int, np.ndarray]] = []
    vis_by_f_tid: List[Dict[int, np.ndarray]] = []
    for f in range(n):
        frame = raw_poses_by_frame[f]
        xy_m: Dict[int, np.ndarray] = {}
        sc_m: Dict[int, np.ndarray] = {}
        vi_m: Dict[int, np.ndarray] = {}
        for tid, pdata in frame.items():
            kp = pdata.get("keypoints")
            sc = pdata.get("scores")
            if kp is None or sc is None:
                continue
            kp = np.asarray(kp, dtype=np.float32)
            sc = np.asarray(sc, dtype=np.float32)
            if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 2:
                continue
            if sc.shape[0] < 17:
                continue
            xy_m[int(tid)] = kp[:17, :2].copy()
            sc_m[int(tid)] = sc[:17].copy()
            vi_m[int(tid)] = kp[:17, 2].copy() if kp.shape[1] > 2 else np.ones(17, dtype=np.float32)
        xy_by_f_tid.append(xy_m)
        sc_by_f_tid.append(sc_m)
        vis_by_f_tid.append(vi_m)

    for f in range(n):
        frame = raw_poses_by_frame[f]
        for tid, pdata in list(frame.items()):
            tid_i = int(tid)
            if tid_i not in xy_by_f_tid[f]:
                continue
            kp = np.asarray(pdata["keypoints"], dtype=np.float32)
            if kp.ndim != 2 or kp.shape[0] < 17:
                continue
            new_xy = kp[:17, :2].copy()
            for j in range(17):
                num_x = 0.0
                num_y = 0.0
                den = 0.0
                for df in range(-radius, radius + 1):
                    f2 = f + df
                    if f2 < 0 or f2 >= n or tid_i not in xy_by_f_tid[f2]:
                        continue
                    vis = float(vis_by_f_tid[f2][tid_i][j])
                    if vis < min_visibility:
                        continue
                    w_sc = float(max(sc_by_f_tid[f2][tid_i][j], 0.0))
                    w_t = 1.0 / (1.0 + abs(float(df)))
                    w = w_sc * w_t
                    if w <= 0.0:
                        continue
                    xy = xy_by_f_tid[f2][tid_i][j]
                    num_x += w * float(xy[0])
                    num_y += w * float(xy[1])
                    den += w
                if den > 0.0:
                    new_xy[j, 0] = num_x / den
                    new_xy[j, 1] = num_y / den
            kp[:17, 0] = new_xy[:, 0]
            kp[:17, 1] = new_xy[:, 1]
            pdata["keypoints"] = kp
