"""Apply params dict entries to os.environ (SWAY_* and offline flags). Shared by main.py and Lab live replay."""


from __future__ import annotations

import os
from typing import Any, Dict


def apply_sway_params_to_env(params: Dict[str, Any]) -> None:
    """Allow YAML to set SWAY_* and common offline env vars before tracking."""
    extra_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "YOLO_OFFLINE", "ULTRALYTICS_OFFLINE")
    for k, v in params.items():
        if v is None:
            continue
        if not (k.startswith("SWAY_") or k in extra_keys):
            continue
        if isinstance(v, bool):
            os.environ[k] = "1" if v else "0"
        else:
            os.environ[k] = str(v)
