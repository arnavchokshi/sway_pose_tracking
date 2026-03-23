#!/usr/bin/env python3
"""
Benchmark runner: Run pipeline and verify outputs against ground truth.

Ground truth is defined in benchmarks/*_ground_truth.yaml.

Usage:
  # Run pipeline, then verify output
  python benchmark.py --ground-truth benchmarks/IMG_0256_ground_truth.yaml

  # Verify existing data.json without re-running pipeline
  python benchmark.py --ground-truth benchmarks/IMG_0256_ground_truth.yaml --json output/data.json

  # Run pipeline with custom output dir
  python benchmark.py --ground-truth benchmarks/IMG_0256_ground_truth.yaml --output-dir output

Exit: 0 = all checks pass, 1 = one or more checks fail
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import yaml


def load_ground_truth(path: Path) -> dict:
    """Load ground truth YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_frame_dimensions(video_path: str) -> Tuple[int, int]:
    """Get frame width and height from video."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap.release()
    return w, h


def compute_benchmark_metrics(data: dict, gt: dict) -> dict:
    """
    Compute metrics from pipeline JSON output for comparison with ground truth.
    """
    meta = data.get("metadata", {})
    frames = data.get("frames", [])
    fps = meta.get("fps") or meta.get("native_fps") or 30.0
    num_frames = meta.get("num_frames") or len(frames)
    video_path = meta.get("video_path", "")

    start_window = gt.get("start_window_frames", 60)
    end_window = gt.get("end_window_frames", 60)
    late_entrant_sec = gt.get("late_entrant_seconds", 11)
    late_tolerance_sec = gt.get("late_entrant_tolerance_seconds", 4)
    prune_region = gt.get("prune_region_bottom_right", {})
    x_min_frac = prune_region.get("x_min_frac", 0.75)
    y_min_frac = prune_region.get("y_min_frac", 0.70)

    # Build per-track first/last frame and box positions
    track_first_frame: Dict[str, int] = {}
    track_last_frame: Dict[str, int] = {}
    track_boxes_in_start: Dict[str, List[List[float]]] = {}  # track_id -> list of [x1,y1,x2,y2]

    for frame in frames:
        fidx = frame.get("frame_idx", -1)
        tracks = frame.get("tracks", {})
        for tid, tdata in tracks.items():
            tid_str = str(tid)
            if tid_str not in track_first_frame:
                track_first_frame[tid_str] = fidx
            track_last_frame[tid_str] = fidx

            if fidx < start_window and "box" in tdata:
                if tid_str not in track_boxes_in_start:
                    track_boxes_in_start[tid_str] = []
                track_boxes_in_start[tid_str].append(tdata["box"])

    # Get frame dimensions for spatial checks
    try:
        frame_w, frame_h = get_frame_dimensions(video_path)
    except Exception:
        frame_w, frame_h = 1920, 1080  # fallback

    # Tracks at start (unique in first N frames)
    tracks_in_start_window: Set[str] = set()
    for frame in frames[:start_window]:
        tracks_in_start_window.update(str(t) for t in frame.get("tracks", {}))
    tracks_at_start = len(tracks_in_start_window)

    # Tracks at end (unique in last N frames)
    tracks_in_end_window: Set[str] = set()
    for frame in frames[-end_window:] if frames else []:
        tracks_in_end_window.update(str(t) for t in frame.get("tracks", {}))
    tracks_at_end = len(tracks_in_end_window)

    # Total unique tracks
    all_track_ids: Set[str] = set()
    per_frame_counts: List[int] = []
    for frame in frames:
        tids = list(str(t) for t in frame.get("tracks", {}))
        all_track_ids.update(tids)
        per_frame_counts.append(len(tids))
    total_unique_tracks = len(all_track_ids)

    # ID consistency: max tracks in any single frame (ghost if >9; fragmentation if total>9 but max<=9)
    max_tracks_in_single_frame = max(per_frame_counts) if per_frame_counts else 0

    # Late entrant time windows (frame indices)
    late_frame_min = int((late_entrant_sec - late_tolerance_sec) * fps)
    late_frame_max = int((late_entrant_sec + late_tolerance_sec) * fps)

    # Frames with wrong count: ghost if any frame has >expected; dropped if too few
    exp_total = gt.get("expected_total_unique_tracks", 9)
    exp_start = gt.get("expected_tracks_at_start", 8)
    exp_end = gt.get("expected_tracks_at_end", 9)
    frames_with_too_many = sum(1 for c in per_frame_counts if c > exp_total)
    frames_with_too_few_before = 0
    frames_with_too_few_after = 0
    for i, c in enumerate(per_frame_counts):
        fidx = frames[i].get("frame_idx", i) if i < len(frames) else i
        if fidx < late_frame_min and c < exp_start:
            frames_with_too_few_before += 1
        elif fidx > late_frame_max and c < exp_end:
            frames_with_too_few_after += 1

    # Late entrants: first appearance in window
    late_entrants = [
        tid for tid, first in track_first_frame.items()
        if late_frame_min <= first <= late_frame_max
    ]

    # Tracks in bottom-right region during start (should be pruned)
    bottom_right_track_ids: List[str] = []
    for tid, boxes in track_boxes_in_start.items():
        if not boxes:
            continue
        # Median center of bbox
        centers_x = [(b[0] + b[2]) / 2 for b in boxes]
        centers_y = [(b[1] + b[3]) / 2 for b in boxes]
        med_x = sum(centers_x) / len(centers_x)
        med_y = sum(centers_y) / len(centers_y)
        # Normalized
        nx = med_x / frame_w if frame_w else 0
        ny = med_y / frame_h if frame_h else 0
        if nx >= x_min_frac and ny >= y_min_frac:
            bottom_right_track_ids.append(tid)

    return {
        "tracks_at_start": tracks_at_start,
        "tracks_at_end": tracks_at_end,
        "total_unique_tracks": total_unique_tracks,
        "late_entrants": late_entrants,
        "late_entrant_count": len(late_entrants),
        "bottom_right_tracks": bottom_right_track_ids,
        "all_track_ids": sorted(all_track_ids, key=lambda x: int(x) if x.isdigit() else 0),
        "track_first_frame": track_first_frame,
        "max_tracks_in_single_frame": max_tracks_in_single_frame,
        "frames_with_too_many": frames_with_too_many,
        "frames_with_too_few_before": frames_with_too_few_before,
        "frames_with_too_few_after": frames_with_too_few_after,
        "fps": fps,
        "num_frames": num_frames,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


def get_failure_reasons(metrics: dict, gt: dict) -> List[Dict[str, Any]]:
    """
    Return structured failure info for logging: check name, actual, expected, reason.
    ID consistency (one person = one ID, no drops, no ghosts) is the top priority.
    """
    failures: List[Dict[str, Any]] = []

    exp_start = gt.get("expected_tracks_at_start")
    exp_total = gt.get("expected_total_unique_tracks")
    max_tracks = metrics.get("max_tracks_in_single_frame", 0)
    exp_max = gt.get("expected_max_tracks_in_single_frame") or (exp_total if exp_total else 9)
    # Fragmentation = same person got 2 IDs (total>exp but max<=exp, so never 10 in one frame)
    is_fragmentation = (
        exp_total is not None
        and metrics["total_unique_tracks"] > exp_total
        and max_tracks <= exp_max
    )
    if exp_start is not None and metrics["tracks_at_start"] != exp_start:
        diff = metrics["tracks_at_start"] - exp_start
        reason = (
            f"Wrong ID count at start — need exactly {exp_start}. "
            "Too few: pruning too aggressive or fragments; too many: ghost/mirror."
        )
        hints = ["REID_MIN_OKS", "REID_MAX_FRAME_GAP", "min_duration_ratio", "SHORT_TRACK_MIN_FRAC", "AUDIENCE_REGION_X_MIN_FRAC"] if diff < 0 else ["EDGE_MARGIN_FRAC", "SYNC_SCORE_MIN", "MEAN_CONFIDENCE_MIN"]
        failures.append({
            "check": "tracks_at_start",
            "actual": metrics["tracks_at_start"],
            "expected": exp_start,
            "reason": reason,
            "suggested_param_hints": hints,
        })

    exp_end = gt.get("expected_tracks_at_end")
    if exp_end is not None and metrics["tracks_at_end"] != exp_end:
        diff = metrics["tracks_at_end"] - exp_end
        reason = (
            f"Wrong ID count at end — need exactly {exp_end}. "
            "Too few: dropped person or fragment; too many: ghost kept."
        )
        hints = ["REID_MIN_OKS", "REID_MAX_FRAME_GAP"] if diff < 0 else ["SYNC_SCORE_MIN", "MEAN_CONFIDENCE_MIN"]
        failures.append({
            "check": "tracks_at_end",
            "actual": metrics["tracks_at_end"],
            "expected": exp_end,
            "reason": reason,
            "suggested_param_hints": hints,
        })

    if exp_total is not None and metrics["total_unique_tracks"] != exp_total:
        diff = metrics["total_unique_tracks"] - exp_total
        if diff > 0:
            # Too many IDs: either ghost (10 in one frame) or fragmentation (same person = 2 IDs)
            if is_fragmentation:
                reason = (
                    f"ID fragmentation: {diff} extra unique track(s) — same person got multiple IDs after occlusion. "
                    "Re-ID must merge fragments: try lower REID_MIN_OKS, higher REID_MAX_FRAME_GAP."
                )
                # Re-ID is top priority for fragmentation
                hints = ["REID_MIN_OKS", "REID_MAX_FRAME_GAP", "min_duration_ratio"]
            else:
                reason = (
                    f"Ghost/extra person: {max_tracks} tracks in a single frame (expected ≤{exp_max}). "
                    "Prune mirrors/reflections: try MEAN_CONFIDENCE_MIN, SPATIAL_OUTLIER_STD_FACTOR."
                )
                hints = ["MEAN_CONFIDENCE_MIN", "SPATIAL_OUTLIER_STD_FACTOR", "SYNC_SCORE_MIN", "EDGE_MARGIN_FRAC"]
        else:
            reason = (
                "Too few unique tracks — over-pruning or fragments not merging. "
                "Relax pruning or improve Re-ID: REID_MIN_OKS, REID_MAX_FRAME_GAP."
            )
            hints = ["REID_MIN_OKS", "REID_MAX_FRAME_GAP", "min_duration_ratio", "SYNC_SCORE_MIN", "SHORT_TRACK_MIN_FRAC", "AUDIENCE_REGION_X_MIN_FRAC"]
        failures.append({
            "check": "total_unique_tracks",
            "actual": metrics["total_unique_tracks"],
            "expected": exp_total,
            "reason": reason,
            "suggested_param_hints": hints,
            "is_fragmentation": is_fragmentation if diff > 0 else False,
        })

    exp_late = gt.get("expected_late_entrants")
    if exp_late is not None and metrics["late_entrant_count"] != exp_late:
        diff = metrics["late_entrant_count"] - exp_late
        reason = (
            "Late entrant not detected — person who enters mid-video got new ID instead of being counted. "
            "Re-ID should merge occlusion fragments: lower REID_MIN_OKS, higher REID_MAX_FRAME_GAP."
            if diff < 0 else
            "Too many late-entrant-like tracks — fragments not merged; lower REID_MIN_OKS to merge."
        )
        hints = ["REID_MIN_OKS", "REID_MAX_FRAME_GAP", "min_duration_ratio"]
        failures.append({
            "check": "late_entrant_count",
            "actual": metrics["late_entrant_count"],
            "expected": exp_late,
            "reason": reason,
            "suggested_param_hints": hints,
        })

    if metrics.get("bottom_right_tracks"):
        failures.append({
            "check": "bottom_right_pruned",
            "actual": len(metrics["bottom_right_tracks"]),
            "expected": 0,
            "reason": f"Audience/head in bottom-right not pruned: {metrics['bottom_right_tracks']}; try SYNC_SCORE_MIN, spatial/bbox pruning",
            "suggested_param_hints": ["SYNC_SCORE_MIN", "EDGE_MARGIN_FRAC"],
        })

    return failures


def run_checks(metrics: dict, gt: dict) -> Tuple[bool, List[str]]:
    """Run all ground-truth checks. ID consistency (one person = one ID) is top priority."""
    messages: List[str] = []
    all_pass = True

    # ID consistency: exactly N people = exactly N IDs at each moment
    exp_total = gt.get("expected_total_unique_tracks")
    exp_max = gt.get("expected_max_tracks_in_single_frame") or exp_total
    if exp_max is not None:
        max_in_frame = metrics.get("max_tracks_in_single_frame", 0)
        ok = max_in_frame <= exp_max
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        messages.append(
            f"  {status} Max tracks in single frame: {max_in_frame} (expected ≤{exp_max}) — ghost if exceeded"
        )

    exp_start = gt.get("expected_tracks_at_start")
    if exp_start is not None:
        ok = metrics["tracks_at_start"] == exp_start
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        messages.append(
            f"  {status} Tracks at start: {metrics['tracks_at_start']} (expected {exp_start})"
        )
        if not ok:
            messages.append(f"      Track IDs in start window: {sorted(metrics.get('all_track_ids', []))}")

    exp_end = gt.get("expected_tracks_at_end")
    if exp_end is not None:
        ok = metrics["tracks_at_end"] == exp_end
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        messages.append(
            f"  {status} Tracks at end: {metrics['tracks_at_end']} (expected {exp_end})"
        )

    if exp_total is not None:
        ok = metrics["total_unique_tracks"] == exp_total
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        messages.append(
            f"  {status} Total unique tracks: {metrics['total_unique_tracks']} (expected {exp_total})"
        )
        if not ok:
            messages.append(f"      Track IDs: {metrics['all_track_ids']}")

    exp_late = gt.get("expected_late_entrants")
    if exp_late is not None:
        ok = metrics["late_entrant_count"] == exp_late
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        messages.append(
            f"  {status} Late entrants (~{gt.get('late_entrant_seconds', 11)}s): {metrics['late_entrant_count']} (expected {exp_late})"
        )
        if metrics["late_entrants"]:
            first_frames = {tid: metrics["track_first_frame"].get(tid) for tid in metrics["late_entrants"]}
            messages.append(f"      Late entrant IDs + first frame: {first_frames}")

    # Bottom-right check: we should have ZERO tracks there (that person should be pruned)
    if metrics.get("bottom_right_tracks"):
        all_pass = False
        messages.append(
            f"  ✗ Bottom-right region: {len(metrics['bottom_right_tracks'])} track(s) detected "
            f"(should be 0 — audience/head should be pruned): {metrics['bottom_right_tracks']}"
        )
    else:
        messages.append("  ✓ Bottom-right region: no tracks (audience correctly pruned)")

    return all_pass, messages


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline against ground truth")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground truth YAML (e.g. benchmarks/IMG_0256_ground_truth.yaml)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to existing data.json (skip pipeline run)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for pipeline run (default: output)",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        default=None,
        help="Run the pipeline before verification (default: True if --json not given)",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run pipeline; require --json",
    )
    parser.add_argument(
        "--trackeval",
        action="store_true",
        help="Also run TrackEval if ground truth YAML defines trackeval.gt_mot_file",
    )
    args = parser.parse_args()

    gt_path = args.ground_truth
    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(2)

    gt = load_ground_truth(gt_path)
    video_path = gt.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        print(f"Warning: Video not found at {video_path}", file=sys.stderr)

    # Determine if we run pipeline
    run_pipeline = args.run_pipeline
    if args.no_run:
        run_pipeline = False
    elif args.json is None:
        run_pipeline = True

    json_path = args.json
    if run_pipeline:
        print("Running pipeline...")
        cmd = [
            sys.executable,
            "main.py",
            video_path,
            "--output-dir",
            str(args.output_dir),
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode != 0:
            print(f"Pipeline failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(result.returncode)
        json_path = args.output_dir / "data.json"

    if not json_path or not json_path.exists():
        print(f"Error: No data.json found at {json_path}", file=sys.stderr)
        sys.exit(2)

    with open(json_path) as f:
        data = json.load(f)

    print("\n" + "=" * 60)
    print("BENCHMARK: IMG_0256 ground truth verification")
    print("=" * 60)
    print(f"Ground truth: {gt_path}")
    print(f"JSON: {json_path}")
    print()

    metrics = compute_benchmark_metrics(data, gt)
    all_pass, messages = run_checks(metrics, gt)

    for m in messages:
        print(m)
    print()
    if args.trackeval:
        try:
            root = Path(__file__).resolve().parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from sway.trackeval_runner import trackeval_from_ground_truth_yaml

            te = trackeval_from_ground_truth_yaml(gt, data)
            if te:
                print("TrackEval metrics:")
                for k in sorted(te.keys()):
                    if k != "sequence":
                        print(f"  {k}: {te[k]:.4f}" if isinstance(te[k], float) else f"  {k}: {te[k]}")
            else:
                print("TrackEval: skipped (no trackeval.gt_mot_file in ground truth YAML)")
        except Exception as ex:
            print(f"TrackEval failed: {ex}", file=sys.stderr)
    print()
    print("-" * 60)
    if all_pass:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
