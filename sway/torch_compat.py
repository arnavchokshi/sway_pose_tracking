"""PyTorch compatibility helpers (e.g. torch.load defaults changed in PyTorch 2.6+)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def torch_load_trusted(path: PathLike, *, map_location: Any = "cpu") -> Any:
    """Load a full checkpoint pickle from a trusted local file (weights + metadata)."""
    import torch

    p = Path(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)
