"""
Temporal pose-crop helpers (see docs/FUTURE_MODULES_IDENTITY_AND_POSE_CROPS.md, Part F).

Applied in main.py Phase 5 after smart bbox expansion and before ViTPose.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

Box = Tuple[float, float, float, float]


def apply_temporal_pose_crop(
    tid: int,
    raw_box: Box,
    *,
    frame_w: int,
    frame_h: int,
    smooth_alpha: float,
    foot_bias_frac: float,
    head_bias_frac: float,
    anti_jitter_px: float,
    state: Dict[int, Box],
) -> Box:
    """
    Optionally EMA-smooth the expanded crop, damp abrupt center jumps, then apply foot/head padding bias.

    ``smooth_alpha``: 0 = off. Otherwise blend toward the new box each frame:
    ``out = (1-a)*prev + a*raw`` with ``a = min(1, max(0, smooth_alpha))``.

    ``anti_jitter_px``: when > 0 and smoothing is on, if the *raw* box center jumps farther than this
    from the *previous smoothed* box center, blend extra toward the smoothed rectangle.
    """
    raw = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
    prev = state.get(tid)

    a = min(1.0, max(0.0, float(smooth_alpha)))
    if a <= 0 or prev is None:
        cur = raw
        if a <= 0:
            state.pop(tid, None)
    else:
        cur = tuple(a * raw[i] + (1.0 - a) * prev[i] for i in range(4))
        if anti_jitter_px > 0:
            rcx = (raw[0] + raw[2]) * 0.5
            rcy = (raw[1] + raw[3]) * 0.5
            pcx = (prev[0] + prev[2]) * 0.5
            pcy = (prev[1] + prev[3]) * 0.5
            if math.hypot(rcx - pcx, rcy - pcy) > float(anti_jitter_px):
                cur = tuple(0.35 * raw[i] + 0.65 * cur[i] for i in range(4))

    x1, y1, x2, y2 = cur
    h = max(1.0, y2 - y1)
    if foot_bias_frac:
        y2 = min(float(frame_h), y2 + float(foot_bias_frac) * h)
    if head_bias_frac:
        y1 = max(0.0, y1 - float(head_bias_frac) * h)

    x1 = max(0.0, min(float(frame_w - 1), x1))
    y1 = max(0.0, min(float(frame_h - 1), y1))
    x2 = max(x1 + 1.0, min(float(frame_w), x2))
    y2 = max(y1 + 1.0, min(float(frame_h), y2))
    out = (x1, y1, x2, y2)
    if a > 0:
        state[tid] = out
    return out
