"""
Shared interpolation helpers (numpy only).

Used by tracker box gap fill / stitch and optional pose-stride gap fill in main.
Default pipeline behavior remains linear unless env selects GSI.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def gsi_interp_scalar(t: float, y0: float, y1: float, lengthscale: float) -> float:
    """GP posterior mean with SE kernel at normalized anchors (0, y0) and (1, y1)."""
    t = float(np.clip(t, 0.0, 1.0))
    l = max(float(lengthscale), 1e-4)
    sn2 = 1e-6

    def k(a: float, b: float) -> float:
        return float(np.exp(-0.5 * ((a - b) / l) ** 2))

    k01 = k(0.0, 1.0)
    k_mat = np.array([[1.0 + sn2, k01], [k01, 1.0 + sn2]], dtype=np.float64)
    y = np.array([float(y0), float(y1)], dtype=np.float64)
    alpha = np.linalg.solve(k_mat, y)
    return float(k(0.0, t) * alpha[0] + k(1.0, t) * alpha[1])


def blend_scalar(t: float, y0: float, y1: float, *, mode: str, gsi_l: float) -> float:
    """Linear blend or GSI between two scalars at normalized time t in [0, 1]."""
    if (mode or "linear").strip().lower() != "gsi":
        return float(y0 + t * (y1 - y0))
    return gsi_interp_scalar(t, float(y0), float(y1), float(gsi_l))


def blend_pose_keypoints_scores(
    kp_prev: np.ndarray,
    kp_next: np.ndarray,
    sc_prev: np.ndarray,
    sc_next: np.ndarray,
    t: float,
    *,
    mode: str,
    gsi_l: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend two pose tensors at normalized time t in (0, 1); mode linear or gsi."""
    kp0 = np.asarray(kp_prev, dtype=np.float64)
    kp1 = np.asarray(kp_next, dtype=np.float64)
    s0 = np.asarray(sc_prev, dtype=np.float64)
    s1 = np.asarray(sc_next, dtype=np.float64)
    if (mode or "linear").strip().lower() != "gsi":
        w0, w1 = 1.0 - t, t
        return (w0 * kp0 + w1 * kp1).astype(np.float32), (w0 * s0 + w1 * s1).astype(np.float32)
    kp_out = np.empty_like(kp0, dtype=np.float64)
    for i in range(kp0.size):
        kp_out.flat[i] = gsi_interp_scalar(t, float(kp0.flat[i]), float(kp1.flat[i]), gsi_l)
    sc_out = np.empty_like(s0, dtype=np.float64)
    for i in range(s0.size):
        sc_out.flat[i] = gsi_interp_scalar(t, float(s0.flat[i]), float(s1.flat[i]), gsi_l)
    return kp_out.astype(np.float32), sc_out.astype(np.float32)
