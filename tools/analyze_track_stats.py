#!/usr/bin/env python3
"""
Read ``track_stats.json`` from a pipeline run (written after Phase 3 on every run).

Does not run the pipeline. Optional helper for golden-set / CI diffing.

  python -m tools.analyze_track_stats path/to/track_stats.json
  python -m tools.analyze_track_stats path/to/track_stats.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize track_stats.json (opt-in export).")
    ap.add_argument("track_stats_json", type=Path, help="track_stats.json path")
    ap.add_argument("--json", action="store_true", help="Re-print JSON to stdout")
    args = ap.parse_args()
    p = args.track_stats_json
    if not p.is_file():
        print(f"Not found: {p}", file=sys.stderr)
        return 1
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"schema_version: {data.get('schema_version')}")
    print(f"total_frames: {data.get('total_frames')}  stride: {data.get('yolo_detection_stride')}")
    print(f"num_tracks: {data.get('num_tracks')}  observations: {data.get('total_observations')}")
    print(
        f"median obs/track: {data.get('median_observations_per_track')}  "
        f"median span (frames): {data.get('median_temporal_span_frames')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
