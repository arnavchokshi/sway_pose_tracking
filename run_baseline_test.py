#!/usr/bin/env python3
"""
Phase 2: Rigorous Testing Protocol — Baseline & Validation

Run pipeline on test videos and log metrics for comparison.
Usage:
  python run_baseline_test.py [--baseline] [--videos VIDEO1 VIDEO2 ...]
  --baseline: Establish baseline (run current pipeline, log metrics)
  (default): Run with fixes, compare to baseline

Step 1: Establish baseline — run on Video 1, IMG_2549.MP4, Video 2946
Step 2: After NMS/pruning changes — re-run, check ID 61 gone, Video 2946 >= 9
Step 3: After Re-ID — monitor IMG_2549.MP4 red vs blue
Step 4: Final validation — batch all videos
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def find_videos(patterns: List[str]) -> List[Path]:
    """Resolve video paths from patterns (relative to input/ or cwd)."""
    found = []
    for p in patterns:
        path = Path(p)
        if not path.exists():
            path = Path("input") / p
        if path.exists():
            found.append(path)
        else:
            print(f"  Warning: {p} not found")
    return found


def run_pipeline(video_path: Path, output_dir: Path, params: Optional[Path] = None) -> bool:
    """Run main pipeline, return success."""
    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable, str(script_dir / "main.py"),
        str(video_path),
        "--output-dir", str(output_dir),
    ]
    if params:
        cmd.extend(["--params", str(params)])
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Pipeline failed: {e}")
        return False


def extract_metrics(output_dir: Path) -> Dict:
    """Extract metrics from prune_log.json and data.json."""
    metrics: Dict = {"output_dir": str(output_dir)}
    prune_log = output_dir / "prune_log.json"
    data_json = output_dir / "data.json"
    if prune_log.exists():
        with open(prune_log) as f:
            pl = json.load(f)
        metrics["tracker_ids_before_prune"] = pl.get("tracker", {}).get("track_ids_before_prune", [])
        metrics["tracker_count"] = pl.get("tracker", {}).get("count", 0)
        metrics["surviving_after_pre_pose"] = pl.get("surviving_after_pre_pose", [])
        metrics["surviving_after_post_pose"] = pl.get("surviving_after_post_pose", [])
        metrics["total_frames"] = pl.get("total_frames", 0)
    if data_json.exists():
        with open(data_json) as f:
            data = json.load(f)
        frames = data.get("frames", [])
        if frames:
            meta = data.get("metadata", {})
            metrics["fps"] = meta.get("fps") or meta.get("native_fps") or 30.0
            # Tracks at start (first 60 frames)
            start_ids = set()
            for fr in frames[:60]:
                start_ids.update(str(t) for t in fr.get("tracks", {}))
            metrics["tracks_at_start"] = len(start_ids)
            # Tracks at end (last 60 frames)
            end_ids = set()
            for fr in frames[-60:] if len(frames) >= 60 else frames:
                end_ids.update(str(t) for t in fr.get("tracks", {}))
            metrics["tracks_at_end"] = len(end_ids)
            # Max concurrent tracks
            max_concurrent = 0
            for fr in frames:
                n = len(fr.get("tracks", {}))
                max_concurrent = max(max_concurrent, n)
            metrics["max_concurrent_tracks"] = max_concurrent
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Baseline test runner for Sway Pose Pipeline")
    ap.add_argument("--baseline", action="store_true", help="Establish baseline (save metrics)")
    ap.add_argument("--videos", nargs="+", default=["IMG_0256.MP4"], help="Video paths/names")
    ap.add_argument("--params", type=Path, help="YAML params override")
    ap.add_argument("--output-root", type=Path, default=Path("output"), help="Output root")
    args = ap.parse_args()

    videos = find_videos(args.videos)
    if not videos:
        print("No videos found")
        sys.exit(1)

    print(f"Running on {len(videos)} video(s)")
    results = {}
    for video in videos:
        out_dir = args.output_root / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {video.name} ---")
        ok = run_pipeline(video, out_dir, args.params)
        if ok:
            metrics = extract_metrics(out_dir)
            results[video.name] = metrics
            print(f"  Tracker IDs before prune: {metrics.get('tracker_count', '?')}")
            print(f"  Surviving post-pose: {len(metrics.get('surviving_after_post_pose', []))}")
            print(f"  Tracks at start: {metrics.get('tracks_at_start', '?')}")
            print(f"  Tracks at end: {metrics.get('tracks_at_end', '?')}")
            print(f"  Max concurrent: {metrics.get('max_concurrent_tracks', '?')}")

    # Save run log
    log_path = args.output_root / "baseline_run.json"
    with open(log_path, "w") as f:
        json.dump({"videos": results, "baseline": args.baseline}, f, indent=2)
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
