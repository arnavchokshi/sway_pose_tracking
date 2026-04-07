"""
MOTE Disocclusion Prediction Module (PLAN_21 — Module 1)

EXPERIMENT ADD-ON — default OFF.

Uses optical flow (RAFT) to predict where DORMANT tracks will re-emerge
after crossing events. When prediction matches actual re-emergence,
re-ID confidence is boosted.

Env:
  SWAY_MOTE_DISOCCLUSION      – 0|1 (default 0)
  SWAY_MOTE_FLOW_MODEL        – raft_small | raft_large (default raft_small)
  SWAY_MOTE_CONFIDENCE_BOOST  – re-ID boost on spatial match (default 0.15)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _default_flow_device(explicit: Optional[str] = None) -> str:
    """Prefer CUDA, then MPS, then CPU (never request cuda when unavailable)."""
    if explicit and explicit.strip():
        return explicit.strip()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class MOTEDisocclusion:
    """Optical-flow-based re-emergence prediction for DORMANT tracks."""

    def __init__(
        self,
        flow_model: Optional[str] = None,
        confidence_boost: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.flow_model_name = flow_model or _env_str("SWAY_MOTE_FLOW_MODEL", "raft_small")
        self.confidence_boost = confidence_boost or _env_float("SWAY_MOTE_CONFIDENCE_BOOST", 0.15)
        self.device = _default_flow_device(device)
        self._flow_model = None
        self._prev_frame = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from torchvision.models.optical_flow import raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights

            if self.flow_model_name == "raft_large":
                self._flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.device).eval()
            else:
                self._flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(self.device).eval()

            logger.info("MOTE: loaded %s for optical flow", self.flow_model_name)
        except (ImportError, Exception) as exc:
            logger.warning("MOTE: RAFT flow model load failed: %s", exc)

    def compute_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Compute dense optical flow between previous and current frame.

        Returns (H, W, 2) flow field or None.
        """
        if self._flow_model is None or self._prev_frame is None:
            self._prev_frame = frame
            return None

        try:
            import torch
            import cv2

            h, w = frame.shape[:2]
            scale = 256 / min(h, w)
            new_h, new_w = int(h * scale) // 8 * 8, int(w * scale) // 8 * 8

            prev_resized = cv2.resize(self._prev_frame, (new_w, new_h))
            curr_resized = cv2.resize(frame, (new_w, new_h))

            prev_t = torch.from_numpy(prev_resized[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            curr_t = torch.from_numpy(curr_resized[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                flows = self._flow_model(prev_t, curr_t)
                flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()

            # Scale flow back to original resolution
            flow[:, :, 0] *= w / new_w
            flow[:, :, 1] *= h / new_h
            flow = cv2.resize(flow, (w, h))

            self._prev_frame = frame
            return flow

        except Exception as exc:
            logger.debug("MOTE flow computation failed: %s", exc)
            self._prev_frame = frame
            return None

    def predict_reemergence(
        self,
        flow_field: np.ndarray,
        dormant_tracks: Dict[int, Tuple[float, float]],
    ) -> Dict[int, Tuple[float, float]]:
        """Predict re-emergence positions for DORMANT tracks.

        Args:
            flow_field: (H, W, 2) optical flow.
            dormant_tracks: {track_id: (last_x, last_y)} positions.

        Returns:
            {track_id: (predicted_x, predicted_y)}.
        """
        predictions = {}
        h, w = flow_field.shape[:2]

        for tid, (lx, ly) in dormant_tracks.items():
            ix, iy = int(lx), int(ly)
            if 0 <= ix < w and 0 <= iy < h:
                fx, fy = flow_field[iy, ix]
                # Filter out unreasonable flow (>50px)
                if abs(fx) < 50 and abs(fy) < 50:
                    predictions[tid] = (lx + fx, ly + fy)
                else:
                    predictions[tid] = (lx, ly)  # no reliable prediction

        return predictions

    def match_prediction(
        self,
        predicted_pos: Tuple[float, float],
        actual_pos: Tuple[float, float],
        threshold_px: float = 50.0,
    ) -> float:
        """Check if a re-emergence matches the predicted position.

        Returns confidence boost if matched, 0 otherwise.
        """
        dist = np.sqrt(
            (predicted_pos[0] - actual_pos[0]) ** 2 +
            (predicted_pos[1] - actual_pos[1]) ** 2
        )
        if dist < threshold_px:
            return self.confidence_boost
        return 0.0


def is_mote_enabled() -> bool:
    return _env_bool("SWAY_MOTE_DISOCCLUSION", False)
