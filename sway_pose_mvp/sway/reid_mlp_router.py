"""
Re-ID MLP Weight Router (PLAN_23 — Layer 3)

Learned dynamic weight redistribution for the 6-signal Re-ID ensemble.
When a signal is absent or below quality threshold, the MLP predicts
optimal weight redistribution across remaining signals.

Falls back to rule-based redistribution when no trained weights are available.

Env:
  SWAY_REID_MLP_CHECKPOINT  – path to trained MLP weights (default: none)
  SWAY_REID_MLP_HIDDEN_DIM  – hidden layer size (default 32)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SIGNAL_NAMES = ["part", "kpr", "skeleton", "face", "color", "spatial"]
NUM_SIGNALS = len(SIGNAL_NAMES)


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


class ReIDMLPRouter:
    """Dynamic weight router for Re-ID signal ensemble.

    Input: 12-d vector = [6 availability flags, 6 confidence scores]
    Output: 6-d weight vector (softmax-normalized)
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        checkpoint_path: Optional[str] = None,
        hidden_dim: Optional[int] = None,
    ):
        self.base_weights = base_weights or {
            "part": 0.30, "kpr": 0.15, "skeleton": 0.20,
            "face": 0.20, "color": 0.10, "spatial": 0.05,
        }
        self._hidden_dim = hidden_dim or _env_int("SWAY_REID_MLP_HIDDEN_DIM", 32)
        self._model = None

        if checkpoint_path is None:
            checkpoint_path = _env_str("SWAY_REID_MLP_CHECKPOINT", "")
        if checkpoint_path:
            self._load_model(checkpoint_path)

    def _load_model(self, path: str) -> None:
        """Load trained MLP weights."""
        p = Path(path)
        if not p.exists():
            logger.debug("MLP router checkpoint not found at %s; using rule-based fallback", p)
            return

        try:
            import torch
            state = torch.load(str(p), map_location="cpu")
            self._model = self._build_mlp()
            self._model.load_state_dict(state)
            self._model.eval()
            logger.info("ReID MLP router loaded from %s", p)
        except Exception as exc:
            logger.debug("MLP router load failed: %s", exc)

    def _build_mlp(self):
        """Build the 2-layer MLP architecture."""
        import torch
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(NUM_SIGNALS * 2, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, NUM_SIGNALS),
            nn.Softmax(dim=-1),
        )

    def compute_weights(
        self,
        signal_availability: Dict[str, bool],
        signal_confidences: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute dynamic weights given signal availability and confidence.

        Args:
            signal_availability: {signal_name: is_available}
            signal_confidences: {signal_name: confidence_score}

        Returns:
            Dict of weights summing to 1.0
        """
        if self._model is not None:
            return self._compute_learned(signal_availability, signal_confidences)
        return self._compute_rule_based(signal_availability, signal_confidences)

    def _compute_learned(
        self,
        availability: Dict[str, bool],
        confidences: Dict[str, float],
    ) -> Dict[str, float]:
        """Use trained MLP for weight prediction."""
        import torch

        avail_vec = [float(availability.get(s, False)) for s in SIGNAL_NAMES]
        conf_vec = [confidences.get(s, 0.0) for s in SIGNAL_NAMES]
        x = torch.tensor(avail_vec + conf_vec, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            weights = self._model(x).squeeze(0).numpy()

        return {s: float(w) for s, w in zip(SIGNAL_NAMES, weights)}

    def _compute_rule_based(
        self,
        availability: Dict[str, bool],
        confidences: Dict[str, float],
    ) -> Dict[str, float]:
        """Rule-based weight redistribution when no MLP is available.

        Absent signals get weight 0; their weight is redistributed proportionally
        to remaining signals, with priority given to high-confidence signals.
        """
        raw_weights: Dict[str, float] = {}
        total_available = 0.0

        for sig in SIGNAL_NAMES:
            is_avail = availability.get(sig, False)
            conf = confidences.get(sig, 0.0)

            if is_avail and conf > 0.0:
                w = self.base_weights.get(sig, 0.0) * conf
                raw_weights[sig] = w
                total_available += w
            else:
                raw_weights[sig] = 0.0

        if total_available <= 0:
            equal = 1.0 / NUM_SIGNALS
            return {s: equal for s in SIGNAL_NAMES}

        return {s: w / total_available for s, w in raw_weights.items()}

    def redistribute_on_missing(
        self,
        missing_signal: str,
        boost_signals: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Redistribute a missing signal's weight to specified boost targets.

        Used for specific rules like: "face hidden -> boost MoCos gait + spatial"
        """
        weights = dict(self.base_weights)
        missing_weight = weights.pop(missing_signal, 0.0)

        if boost_signals is None:
            boost_signals = [s for s in SIGNAL_NAMES if s != missing_signal]

        boost_total = sum(weights.get(s, 0.0) for s in boost_signals)
        if boost_total <= 0:
            boost_total = 1.0

        for s in boost_signals:
            weights[s] = weights.get(s, 0.0) + missing_weight * (weights.get(s, 0.0) / boost_total)

        weights[missing_signal] = 0.0

        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights
