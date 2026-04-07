#!/usr/bin/env python3
"""
Smoke test for ``sway.gnn_track_refine`` (not the default pipeline).

  SWAY_GNN_TRACK_SMOKE=1 python -m tools.gnn_group_track_stub
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if os.environ.get("SWAY_GNN_TRACK_SMOKE", "").strip().lower() not in ("1", "true", "yes"):
        print(
            "Set SWAY_GNN_TRACK_SMOKE=1 to run the GNN refine smoke test.",
            file=sys.stderr,
        )
        return 2
    os.environ["SWAY_GNN_DEVICE"] = os.environ.get("SWAY_GNN_DEVICE", "cpu")
    os.environ["SWAY_GNN_SEED"] = os.environ.get("SWAY_GNN_SEED", "0")
    from sway.gnn_track_refine import gnn_refine_raw_tracks

    box = (10.0, 20.0, 50.0, 120.0)
    raw = {
        1: [(0, box, 0.9), (1, box, 0.9)],
        2: [(0, box, 0.85), (1, box, 0.85)],
    }
    gnn_refine_raw_tracks(raw, total_frames=30, ystride=1)
    print(f"OK: merged to {len(raw)} track(s), lens={[len(v) for v in raw.values()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
