#!/usr/bin/env python3
"""
Optional GNN / group-graph tracking experiments — **not** in the default pipeline.

  SWAY_GNN_TRACK_SMOKE=1 python -m tools.gnn_group_track_stub
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if os.environ.get("SWAY_GNN_TRACK_SMOKE", "").strip().lower() not in ("1", "true", "yes"):
        print(
            "GNN group tracking is roadmap-only. Set SWAY_GNN_TRACK_SMOKE=1 to use this stub.",
            file=sys.stderr,
        )
        return 2
    print(
        "Stub: graph construction from per-frame boxes + optional pose features goes here. "
        "Wire into ``main.py`` only behind a new default-off env/CLI toggle.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
