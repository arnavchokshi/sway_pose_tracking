"""Post-hoc confidence temperature (plan §2.4). T=1.0 is a no-op."""

from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch


class TemperatureScaler:
    def __init__(self, T: float = 1.0) -> None:
        self.T = max(float(T), 1e-6)

    def scale_logits(self, logits: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(logits, torch.Tensor):
            return logits / self.T
        return np.asarray(logits, dtype=np.float64) / self.T

    def scale_probs_from_logits(self, logits: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        z = self.scale_logits(logits)
        if isinstance(z, torch.Tensor):
            return torch.sigmoid(z) if z.numel() == 1 else torch.softmax(z, dim=-1)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def get_temperature_scaler() -> TemperatureScaler:
    T = float(os.environ.get("SWAY_CONF_TEMPERATURE", "1.0"))
    return TemperatureScaler(T)
