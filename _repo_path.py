"""Ensure sway_pose_mvp/ is on sys.path so `import sway` works when running scripts here."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
