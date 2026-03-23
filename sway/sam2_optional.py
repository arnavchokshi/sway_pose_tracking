"""
Optional SAM2 mask stream (plan §3.2). Not wired into main by default — heavy deps.

If `sam2` is installed and SWAY_SAM2_MASK_POSE=1, future integration can:
  1. Run SAM2 video predictor for per-person masks
  2. Mask crops (neutral gray outside mask) before ViTPose (--pose-model huge recommended)

See README or prefetch_models for checkpoint download notes.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np


def sam2_mask_pose_enabled() -> bool:
    return os.environ.get("SWAY_SAM2_MASK_POSE", "").lower() in ("1", "true", "yes")


def try_import_sam2():
    try:
        pass  # noqa: F401 — placeholder for sam2 package
    except Exception:
        return None
    return None


def apply_mask_to_crop(
    crop_bgr: np.ndarray,
    mask_bool: np.ndarray,
    fill_value: float = 114.0,
) -> np.ndarray:
    """Zero non-dancer pixels to ImageNet mean gray for pose inference."""
    out = crop_bgr.copy().astype(np.float32)
    m = mask_bool.astype(bool)
    if m.shape[:2] != out.shape[:2]:
        raise ValueError("mask shape must match crop")
    for c in range(min(3, out.shape[2])):
        ch = out[:, :, c]
        ch[~m] = fill_value
        out[:, :, c] = ch
    return out.astype(np.uint8)
