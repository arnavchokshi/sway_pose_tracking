#!/usr/bin/env python3
"""
Optional HMR / 4DHumans-style mesh hook — **not** integrated into ``main.py``.

The canonical 3D path in this repo remains pose lift + optional backends documented in
``docs/3D_POSE_IN_FULL_PIPELINE.md``. Use this file as a scratchpad when experimenting.

  SWAY_HMR_SMOKE=1 python -m tools.hmr_3d_optional_stub
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if os.environ.get("SWAY_HMR_SMOKE", "").strip().lower() not in ("1", "true", "yes"):
        print(
            "HMR / 4DHumans experiments are opt-in. Set SWAY_HMR_SMOKE=1 to acknowledge.",
            file=sys.stderr,
        )
        return 2
    print(
        "Stub: add HMR2 / 4DHumans inference here; keep ``main.py`` unchanged unless "
        "you add an explicit CLI flag and default-off env gate.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
