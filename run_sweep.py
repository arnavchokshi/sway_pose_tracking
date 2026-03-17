#!/usr/bin/env python3
"""
Sweep driver: Run pipeline repeatedly with different parameter sets, validate against ground truth.

Fully autonomous — no input required. Just run from sway_pose_mvp/:

  python run_sweep.py

Uses benchmarks/sweep_config.yaml and IMG_0256 ground truth by default.
On each failure, appends param sets from suggested_next_params (--adaptive, default).
Stops on first PASS. Logs to output/sweep_log.jsonl.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# Run from script directory so relative paths in config work
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

from benchmark import (
    load_ground_truth,
    compute_benchmark_metrics,
    run_checks,
    get_failure_reasons,
)

# Defaults for tunable params (from track_pruning, crossover) — used for effective_params log
PARAM_DEFAULTS = {
    "min_duration_ratio": 0.20,
    "KINETIC_STD_FRAC": 0.05,
    "SYNC_SCORE_MIN": 0.10,
    "EDGE_MARGIN_FRAC": 0.10,
    "EDGE_PRESENCE_FRAC": 0.30,
    "min_lower_body_conf": 0.30,
    "REID_MAX_FRAME_GAP": 90,
    "REID_MIN_OKS": 0.35,
    "MEAN_CONFIDENCE_MIN": 0.45,
    "SPATIAL_OUTLIER_STD_FACTOR": 2.0,
    "SHORT_TRACK_MIN_FRAC": 0.20,
    "AUDIENCE_REGION_X_MIN_FRAC": 0.75,
    "AUDIENCE_REGION_Y_MIN_FRAC": 0.70,
}

# When --adaptive: step values to try when a param is suggested
ADAPTIVE_STEPS = {
    # Re-ID first — merge occlusion fragments, one person = one ID
    "REID_MIN_OKS": [0.30, 0.25, 0.20],
    "REID_MAX_FRAME_GAP": [120, 150, 180],
    "SYNC_SCORE_MIN": [0.12, 0.15, 0.18],
    "EDGE_MARGIN_FRAC": [0.08, 0.06],
    "min_lower_body_conf": [0.35, 0.40],
    "min_duration_ratio": [0.15, 0.18],
    "KINETIC_STD_FRAC": [0.04, 0.03],
    "EDGE_PRESENCE_FRAC": [0.25, 0.20],
    "MEAN_CONFIDENCE_MIN": [0.50, 0.55, 0.60],
    "SPATIAL_OUTLIER_STD_FACTOR": [1.8, 1.5],
    "SHORT_TRACK_MIN_FRAC": [0.15, 0.12, 0.10],
    "AUDIENCE_REGION_X_MIN_FRAC": [0.90, 0.95],
    "AUDIENCE_REGION_Y_MIN_FRAC": [0.90, 0.95],
}


def _effective_params(override: dict) -> dict:
    """Merge override with defaults; log all tunable params."""
    out = dict(PARAM_DEFAULTS)
    out.update(override)
    return out


def _adaptive_param_sets(suggested: list, last_params: dict, already_tried: set) -> list:
    """Generate new param sets from suggestions; avoid duplicates."""
    new_sets = []
    for param_name in suggested:
        steps = ADAPTIVE_STEPS.get(param_name)
        if not steps:
            continue
        for val in steps:
            key = (param_name, val)
            if key in already_tried:
                continue
            already_tried.add(key)
            p = dict(last_params)
            p[param_name] = val
            new_sets.append({"name": f"adaptive_{param_name}={val}", "params": p})
            break  # One new set per suggested param
    return new_sets


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline sweep with parameter tuning against ground truth"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "benchmarks" / "sweep_config.yaml",
        help="Path to sweep config YAML (default: benchmarks/sweep_config.yaml)",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run all param sets even after first PASS",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=True,
        help="On failure, append param sets from suggested_next_params and continue (default: True)",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_false",
        dest="adaptive",
        help="Disable adaptive param generation",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    ground_truth_path = Path(config.get("ground_truth", "benchmarks/IMG_0256_ground_truth.yaml"))
    video_path = config.get("video_path")
    if not video_path:
        gt = load_ground_truth(ground_truth_path)
        video_path = gt.get("video_path")
    if not video_path or not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_on_pass = config.get("stop_on_pass", True) and not args.exhaustive
    param_sets = config.get("param_sets", [])

    if not param_sets:
        print("Error: No param_sets in config", file=sys.stderr)
        sys.exit(2)

    gt = load_ground_truth(ground_truth_path)
    log_path = output_dir / "sweep_log.jsonl"
    use_adaptive = args.adaptive
    adaptive_max_extra = 5
    adaptive_tried = set()
    adaptive_count = 0

    print("=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Video: {video_path}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Param sets: {len(param_sets)}")
    print(f"Adaptive: {use_adaptive}")
    print(f"Log: {log_path}")
    print()

    passed_name = None
    passed_params = None
    all_param_sets = list(param_sets)
    idx = 0

    while idx < len(all_param_sets):
        pset = all_param_sets[idx]
        iteration = idx + 1
        name = pset.get("name", f"set_{iteration}")
        params = pset.get("params", {})

        print(f"[{iteration}/{len(all_param_sets)}] {name}")
        if params:
            for k, v in params.items():
                print(f"      {k}: {v}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            yaml.dump(params, tf, default_flow_style=False)
            params_path = tf.name

        try:
            cmd = [
                sys.executable,
                "main.py",
                str(video_path),
                "--output-dir",
                str(output_dir),
                "--params",
                params_path,
            ]
            result = subprocess.run(cmd, cwd=SCRIPT_DIR)
            if result.returncode != 0:
                print(f"      Pipeline failed (exit {result.returncode})")
                log_entry = {
                    "iteration": iteration,
                    "name": name,
                    "params_override": params,
                    "effective_params": _effective_params(params),
                    "pass": False,
                    "error": "pipeline_failed",
                    "exit_code": result.returncode,
                    "expected_tracks_at_start": gt.get("expected_tracks_at_start"),
                    "expected_tracks_at_end": gt.get("expected_tracks_at_end"),
                    "expected_total_unique_tracks": gt.get("expected_total_unique_tracks"),
                    "expected_late_entrants": gt.get("expected_late_entrants"),
                    "failure_reasons": [],
                }
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(log_entry) + "\n")
                continue

            json_path = output_dir / "data.json"
            if not json_path.exists():
                print(f"      No data.json produced")
                log_entry = {
                    "iteration": iteration,
                    "name": name,
                    "params_override": params,
                    "effective_params": _effective_params(params),
                    "pass": False,
                    "error": "no_data_json",
                    "expected_tracks_at_start": gt.get("expected_tracks_at_start"),
                    "expected_tracks_at_end": gt.get("expected_tracks_at_end"),
                    "expected_total_unique_tracks": gt.get("expected_total_unique_tracks"),
                    "expected_late_entrants": gt.get("expected_late_entrants"),
                    "failure_reasons": [],
                }
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(log_entry) + "\n")
                continue

            with open(json_path) as jf:
                data = json.load(jf)

            metrics = compute_benchmark_metrics(data, gt)
            all_pass, messages = run_checks(metrics, gt)
            failure_reasons = get_failure_reasons(metrics, gt)

            # Suggested params for next iteration — prioritize Re-ID for ID consistency
            suggested_next_params = []
            if failure_reasons:
                seen = set()
                # Re-ID params first (biggest priority: one person = one ID, merge fragments)
                reid_params = ["REID_MIN_OKS", "REID_MAX_FRAME_GAP"]
                for p in reid_params:
                    for fr in failure_reasons:
                        if p in fr.get("suggested_param_hints", []) and p not in seen:
                            seen.add(p)
                            suggested_next_params.append(p)
                            break
                # Then other hints from failures
                for fr in failure_reasons:
                    for hint in fr.get("suggested_param_hints", []):
                        if hint not in seen:
                            seen.add(hint)
                            suggested_next_params.append(hint)

            log_entry = {
                "iteration": iteration,
                "name": name,
                "params_override": params,
                "effective_params": _effective_params(params),
                "pass": all_pass,
                "tracks_at_start": metrics["tracks_at_start"],
                "tracks_at_end": metrics["tracks_at_end"],
                "total_unique_tracks": metrics["total_unique_tracks"],
                "max_tracks_in_single_frame": metrics.get("max_tracks_in_single_frame"),
                "late_entrant_count": metrics["late_entrant_count"],
                "bottom_right_count": len(metrics.get("bottom_right_tracks", [])),
                "expected_tracks_at_start": gt.get("expected_tracks_at_start"),
                "expected_tracks_at_end": gt.get("expected_tracks_at_end"),
                "expected_total_unique_tracks": gt.get("expected_total_unique_tracks"),
                "expected_late_entrants": gt.get("expected_late_entrants"),
                "failure_reasons": failure_reasons,
                "suggested_next_params": suggested_next_params,
                "track_first_frame": metrics.get("track_first_frame"),
            }

            with open(log_path, "a") as lf:
                lf.write(json.dumps(log_entry, default=str) + "\n")

            if all_pass:
                print(f"      RESULT: PASS")
                passed_name = name
                passed_params = params
                if stop_on_pass:
                    print()
                    print("Stopping on first pass (use --exhaustive to run all)")
                    break
            else:
                if use_adaptive and adaptive_count < adaptive_max_extra and suggested_next_params:
                    extra = _adaptive_param_sets(suggested_next_params, params, adaptive_tried)
                    for es in extra:
                        all_param_sets.append(es)
                        adaptive_count += 1
                print(f"      RESULT: FAIL")
                for m in messages:
                    if "✗" in m:
                        print(f"        {m.strip()}")
                for fr in failure_reasons:
                    print(f"        Reason: {fr['reason']}")
                if suggested_next_params:
                    print(f"        Suggested params to try: {suggested_next_params}")

        finally:
            Path(params_path).unlink(missing_ok=True)

        idx += 1

    print()
    print("-" * 60)
    if passed_name:
        print(f"PASS: {passed_name}")
        print(f"Params: {passed_params}")
        sys.exit(0)
    else:
        print("No param set passed. See sweep_log.jsonl for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
