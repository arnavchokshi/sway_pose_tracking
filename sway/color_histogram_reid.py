"""
Color Histogram Re-ID Signal (PLAN_12)

Fast, simple, and robust costume-color matching using per-region histograms
extracted from SAM2 mask-isolated pixels. Even "matching" costumes often have
subtle color differences under stage lighting angles.

This is the cheapest re-ID signal (~0.5ms per crop) and provides a useful
coarse pre-filter before expensive embedding comparisons.

Env:
  SWAY_REID_COLOR_SPACE       – hsv | lab | rgb (default hsv)
  SWAY_ENROLLMENT_COLOR_BINS  – bins per channel (default 32)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_L_HIP, _R_HIP = 11, 12
_L_ANKLE, _R_ANKLE = 15, 16
_L_SHOULDER, _R_SHOULDER = 5, 6


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


class ColorHistogramExtractor:
    """Per-region color histogram extraction for costume matching."""

    def __init__(
        self,
        color_space: Optional[str] = None,
        n_bins: Optional[int] = None,
    ):
        self.color_space = color_space or _env_str("SWAY_REID_COLOR_SPACE", "hsv")
        self.n_bins = n_bins or _env_int("SWAY_ENROLLMENT_COLOR_BINS", 32)

    def extract(
        self,
        crop: np.ndarray,
        mask: Optional[np.ndarray],
        keypoints: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Extract per-region color histograms.

        Args:
            crop: BGR person crop.
            mask: binary mask in crop dimensions. Pixels outside are ignored.
            keypoints: (17, 3) array in crop coordinates for region splitting.

        Returns:
            Dict with keys "upper", "lower", "shoes", each a histogram vector.
        """
        h, w = crop.shape[:2]

        # Convert to target color space
        if self.color_space == "hsv":
            converted = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        elif self.color_space == "lab":
            converted = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        else:
            converted = crop[:, :, ::-1]  # BGR → RGB

        # Build pixel mask
        if mask is not None and mask.shape[:2] == (h, w):
            pixel_mask = mask.astype(np.uint8)
        else:
            pixel_mask = np.ones((h, w), dtype=np.uint8)

        # Determine region boundaries from keypoints
        upper_y, lower_y, shoe_y = self._region_boundaries(h, keypoints)

        regions = {
            "upper": (0, upper_y),
            "lower": (upper_y, lower_y),
            "shoes": (lower_y, shoe_y),
        }

        result: Dict[str, np.ndarray] = {}
        for region_name, (y1, y2) in regions.items():
            y1 = max(0, min(y1, h))
            y2 = max(y1 + 1, min(y2, h))

            region_pixels = converted[y1:y2, :, :]
            region_mask = pixel_mask[y1:y2, :]

            hist = self._compute_histogram(region_pixels, region_mask)
            result[region_name] = hist

        return result

    def _region_boundaries(
        self, h: int, keypoints: Optional[np.ndarray]
    ) -> tuple:
        """Compute y-boundaries for upper/lower/shoes regions."""
        if keypoints is not None and keypoints.shape[0] >= 17:
            # Upper: above hip keypoints
            hip_ys = []
            for kid in [_L_HIP, _R_HIP]:
                if keypoints[kid, 2] > 0.3:
                    hip_ys.append(keypoints[kid, 1])
            upper_y = int(np.mean(hip_ys)) if hip_ys else int(h * 0.45)

            # Lower: hips to ankles
            ankle_ys = []
            for kid in [_L_ANKLE, _R_ANKLE]:
                if keypoints[kid, 2] > 0.3:
                    ankle_ys.append(keypoints[kid, 1])
            lower_y = int(np.mean(ankle_ys)) if ankle_ys else int(h * 0.85)

            shoe_y = min(lower_y + int((h - lower_y) * 0.8), h)
        else:
            upper_y = int(h * 0.45)
            lower_y = int(h * 0.85)
            shoe_y = h

        return upper_y, lower_y, shoe_y

    def _compute_histogram(
        self, pixels: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Compute a multi-channel histogram from masked pixels."""
        h, w, c = pixels.shape
        hists = []

        for ch in range(c):
            if self.color_space == "hsv" and ch == 0:
                hist_range = [0, 180]  # Hue range in OpenCV
            else:
                hist_range = [0, 256]

            hist = cv2.calcHist(
                [pixels], [ch], mask, [self.n_bins], hist_range
            )
            hists.append(hist.flatten())

        combined = np.concatenate(hists)

        # L1-normalize
        total = combined.sum()
        if total > 0:
            combined /= total

        return combined.astype(np.float32)

    def compare(
        self,
        gallery_hists: Dict[str, np.ndarray],
        query_hists: Dict[str, np.ndarray],
    ) -> float:
        """Compare two sets of region histograms. Lower = more similar."""
        shared = set(gallery_hists.keys()) & set(query_hists.keys())
        if not shared:
            return 1.0

        distances = []
        for region in shared:
            g = gallery_hists[region]
            q = query_hists[region]
            if len(g) != len(q):
                continue
            # Bhattacharyya distance
            dist = cv2.compareHist(
                g.reshape(-1, 1).astype(np.float32),
                q.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
            distances.append(dist)

        return float(np.mean(distances)) if distances else 1.0
