"""
FastTracker adapter for the Sway pipeline.

FastTracker (arXiv 2508.14370, August 2025) is a real-time MOT framework
with occlusion-aware tracking. It has a YOLOX-style API that is different
from BoxMOT's standard update(dets, img) → np.ndarray[M, 8] interface.

This adapter wraps FastTracker's Fasttracker class to be compatible with
the Sway pipeline's BoxMOT tracking loop.

FastTracker constructor takes (args, config_dict, frame_rate).
FastTracker.update takes (output_results, img_info, img_size).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FASTTRACKER_DIR = REPO_ROOT / "models" / "FastTracker"


class FastTrackerWrapper:
    """Wraps FastTracker to provide a BoxMOT-compatible interface."""

    def __init__(self, tracker: Any, img_size: tuple = (1280, 720)):
        self._tracker = tracker
        self._img_size = img_size

    def update(self, dets: np.ndarray, img: np.ndarray, embs: Optional[np.ndarray] = None) -> np.ndarray:
        """BoxMOT-compatible update: dets [N, 6] → outputs [M, 8]."""
        if len(dets) == 0:
            return np.empty((0, 8))

        h, w = img.shape[:2]
        img_info = [h, w]
        img_size = [h, w]

        if dets.shape[1] >= 6:
            output_results = np.column_stack([
                dets[:, :4],
                dets[:, 4],
            ])
        else:
            output_results = dets

        online_targets = self._tracker.update(output_results, img_info, img_size)

        results = []
        for t in online_targets:
            if not hasattr(t, "tlwh"):
                continue
            tlwh = t.tlwh
            tid = t.track_id
            score = t.score if hasattr(t, "score") else 1.0
            cls = 0
            x1, y1, w_box, h_box = tlwh
            x2, y2 = x1 + w_box, y1 + h_box
            results.append([x1, y1, x2, y2, tid, score, cls, 0])

        if not results:
            return np.empty((0, 8))
        return np.array(results)


def create_fasttracker(
    yconf: float,
    dev: Any,
    doc_kw: Dict[str, Any],
) -> FastTrackerWrapper:
    """Try to instantiate a FastTracker with the Sway pipeline adapter.

    Raises ImportError if the FastTracker repo is not available.
    """
    if not FASTTRACKER_DIR.is_dir():
        raise ImportError(f"FastTracker repo not found at {FASTTRACKER_DIR}")

    ft_yolox = str(FASTTRACKER_DIR)
    if ft_yolox not in sys.path:
        sys.path.insert(0, ft_yolox)

    try:
        from yolox.tracker.fasttracker import Fasttracker  # type: ignore[import]
    except ImportError as e:
        raise ImportError(f"Cannot import FastTracker: {e}")

    det_high = float(os.environ.get("SWAY_FT_DET_HIGH", str(yconf)) or str(yconf))
    det_low = float(os.environ.get("SWAY_FT_DET_LOW", "0.1") or "0.1")
    proximity_rad = float(os.environ.get("SWAY_FT_PROXIMITY_RAD", "0.3") or "0.3")
    box_enlarge = float(os.environ.get("SWAY_FT_BOX_ENLARGE", "1.1") or "1.1")
    motion_damp = float(os.environ.get("SWAY_FT_MOTION_DAMP", "0.5") or "0.5")

    config = {
        "track_thresh": det_high,
        "match_thresh": 0.8,
        "track_buffer": int(doc_kw.get("max_age", 150)),
        "reset_velocity_offset_occ": True,
        "reset_pos_offset_occ": True,
        "enlarge_bbox_occ": box_enlarge,
        "dampen_motion_occ": motion_damp,
        "active_occ_to_lost_thresh": proximity_rad,
        "init_iou_suppress": 0.5,
    }

    class _Args:
        pass

    args = _Args()

    tracker = Fasttracker(args, config, frame_rate=30)
    return FastTrackerWrapper(tracker)
