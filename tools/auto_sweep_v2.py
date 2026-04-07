#!/usr/bin/env python3
"""
Optuna TPE sweep v2 for Phase 1-3 with multi-objective scoring support.

v1 behavior (MOT/TrackEval-only) is preserved when no expectation scoring is
configured.

v2 adds optional label-aware signals per sequence:
  - Benchmark expectation score from tools.benchmark ground-truth YAML.
  - Re-emergence re-ID consistency (anchor-pair labels).
  - Keypoint visibility honesty (false-visible penalty).
  - Confidence calibration (Brier-based).

Per-sequence objective:
  sequence_score = weighted mean of configured available terms.

Global objective:
  weighted harmonic mean across all sequence_score values (same as v1).

Config extends data/ground_truth/sweep_sequences.yaml with optional fields:

sequence_order:
  - bigtest
sequences:
  bigtest:
    video: data/ground_truth/bigtest/video.mp4
    gt_mot: data/ground_truth/bigtest/gt/gt.txt
    im_width: 1920
    im_height: 1080
    weight: 2.0
    benchmark_gt_yaml: benchmarks/BIGTEST_ground_truth.yaml   # optional
    objective_weights:                                        # optional
      mot: 0.75
      expectation: 0.25

objective_weights:                                            # optional defaults
  mot: 1.0
  expectation: 0.0
  reid_reemergence: 0.0
  visibility_honesty: 0.0
  confidence_calibration: 0.0
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from tools import auto_sweep as v1
from tools.benchmark import (
    compute_benchmark_metrics,
    get_failure_reasons,
    load_ground_truth,
    run_checks,
)


def _count_active_expectation_checks(gt: Dict[str, Any]) -> int:
    count = 0
    exp_total = gt.get("expected_total_unique_tracks")
    exp_max = gt.get("expected_max_tracks_in_single_frame") or exp_total
    if exp_max is not None:
        count += 1
    for k in (
        "expected_tracks_at_start",
        "expected_tracks_at_end",
        "expected_total_unique_tracks",
        "expected_late_entrants",
    ):
        if gt.get(k) is not None:
            count += 1
    # bottom-right prune check is always active in run_checks
    count += 1
    return max(count, 1)


def _expectation_score_from_data_json(
    *,
    data_json_path: Path,
    benchmark_gt_yaml_path: Path,
) -> Tuple[float, Dict[str, Any]]:
    gt = load_ground_truth(benchmark_gt_yaml_path)
    data = json.loads(data_json_path.read_text(encoding="utf-8"))
    metrics = compute_benchmark_metrics(data, gt)
    all_pass, _messages = run_checks(metrics, gt)
    failures = get_failure_reasons(metrics, gt)
    active_checks = _count_active_expectation_checks(gt)
    failed_checks = min(len(failures), active_checks)

    # Dense scalar in [0,1] so Optuna can learn gradations.
    score = max(0.0, 1.0 - (failed_checks / float(active_checks)))
    if all_pass:
        score = 1.0

    return score, {
        "all_pass": all_pass,
        "active_checks": active_checks,
        "failed_checks": failed_checks,
        "failures": failures,
        "metrics": {
            "tracks_at_start": metrics.get("tracks_at_start"),
            "tracks_at_end": metrics.get("tracks_at_end"),
            "total_unique_tracks": metrics.get("total_unique_tracks"),
            "late_entrant_count": metrics.get("late_entrant_count"),
            "max_tracks_in_single_frame": metrics.get("max_tracks_in_single_frame"),
            "bottom_right_tracks_count": len(metrics.get("bottom_right_tracks") or []),
        },
    }


def _weighted_average(terms: List[Tuple[float, float]]) -> float:
    used = [(w, x) for (w, x) in terms if w > 0]
    if not used:
        return 0.0
    denom = sum(w for (w, _) in used)
    return sum(w * x for (w, x) in used) / denom


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _frame_track_index(data: Dict[str, Any]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for fr in data.get("frames") or []:
        try:
            fidx = int(fr.get("frame_idx", 0))
        except (TypeError, ValueError):
            continue
        tracks = fr.get("tracks") or {}
        if isinstance(tracks, dict):
            out[fidx] = tracks
    return out


def _track_for_anchor(
    frame_tracks: Dict[str, Dict[str, Any]],
    x: float,
    y: float,
    max_dist_px: float,
) -> Optional[Tuple[str, Dict[str, Any], float]]:
    contained: List[Tuple[float, str, Dict[str, Any]]] = []
    nearest: Optional[Tuple[float, str, Dict[str, Any]]] = None
    for tid, tdata in frame_tracks.items():
        box = tdata.get("box")
        if not isinstance(box, list) or len(box) < 4:
            continue
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        cx, cy = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        d = math.hypot(cx - x, cy - y)
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = max(1.0, (x2 - x1) * (y2 - y1))
            contained.append((area, tid, tdata))
        if nearest is None or d < nearest[0]:
            nearest = (d, tid, tdata)

    if contained:
        contained.sort(key=lambda t: t[0])
        _area, tid, tdata = contained[0]
        box = tdata.get("box") or [0.0, 0.0, 0.0, 0.0]
        cx = 0.5 * (float(box[0]) + float(box[2]))
        cy = 0.5 * (float(box[1]) + float(box[3]))
        return tid, tdata, float(math.hypot(cx - x, cy - y))

    if nearest is not None and nearest[0] <= float(max_dist_px):
        return nearest[1], nearest[2], float(nearest[0])
    return None


def _load_sweep_v2_labels(seq_spec: Dict[str, Any], benchmark_gt_yaml_path: Optional[Path]) -> Dict[str, Any]:
    labels_path = seq_spec.get("sweep_v2_labels_yaml")
    if labels_path:
        p = v1._resolve_path(labels_path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return raw.get("sweep_v2_labels") if isinstance(raw, dict) and "sweep_v2_labels" in raw else raw
    if benchmark_gt_yaml_path is None:
        return {}
    gt = load_ground_truth(benchmark_gt_yaml_path)
    lab = gt.get("sweep_v2_labels")
    return lab if isinstance(lab, dict) else {}


def _score_reid_reemergence(
    *,
    labels: Dict[str, Any],
    frames_by_idx: Dict[int, Dict[str, Dict[str, Any]]],
) -> Tuple[Optional[float], Dict[str, Any]]:
    events = labels.get("reid_reemergence_events") or []
    if not events:
        return None, {"reason": "no_reid_reemergence_events"}
    total = len(events)
    matches = 0
    resolvable = 0
    debug_rows: List[Dict[str, Any]] = []
    for ev in events:
        try:
            bf = int(ev["before_frame"])
            af = int(ev["after_frame"])
            bx, by = float(ev["before_xy"][0]), float(ev["before_xy"][1])
            ax, ay = float(ev["after_xy"][0]), float(ev["after_xy"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            debug_rows.append({"error": "invalid_event_schema", "event": ev})
            continue
        md = float(ev.get("max_dist_px", labels.get("anchor_max_dist_px", 80.0)))
        b_track = _track_for_anchor(frames_by_idx.get(bf, {}), bx, by, md)
        a_track = _track_for_anchor(frames_by_idx.get(af, {}), ax, ay, md)
        if b_track is not None and a_track is not None:
            resolvable += 1
            ok = b_track[0] == a_track[0]
            if ok:
                matches += 1
            debug_rows.append(
                {
                    "before_frame": bf,
                    "after_frame": af,
                    "before_tid": b_track[0],
                    "after_tid": a_track[0],
                    "match": bool(ok),
                }
            )
        else:
            debug_rows.append(
                {
                    "before_frame": bf,
                    "after_frame": af,
                    "before_tid": b_track[0] if b_track else None,
                    "after_tid": a_track[0] if a_track else None,
                    "match": False,
                    "unresolved": True,
                }
            )
    # Penalize unresolved events by normalizing over total labels.
    score = float(matches) / float(max(total, 1))
    return _clamp01(score), {
        "total_events": total,
        "resolvable_events": resolvable,
        "matches": matches,
        "accuracy_resolvable_only": (float(matches) / float(resolvable)) if resolvable > 0 else None,
        "debug": debug_rows[:50],
    }


def _extract_joint_conf(track_data: Dict[str, Any], joint_index: int) -> Optional[float]:
    kpts = track_data.get("keypoints")
    if not isinstance(kpts, list) or joint_index < 0 or joint_index >= len(kpts):
        return None
    row = kpts[joint_index]
    if not isinstance(row, list) or len(row) < 3:
        return None
    try:
        return _clamp01(float(row[2]))
    except (TypeError, ValueError):
        return None


def _score_visibility_and_calibration(
    *,
    labels: Dict[str, Any],
    frames_by_idx: Dict[int, Dict[str, Dict[str, Any]]],
) -> Tuple[Optional[float], Dict[str, Any], Optional[float], Dict[str, Any]]:
    rows = labels.get("keypoint_visibility_labels") or []
    if not rows:
        return (
            None,
            {"reason": "no_keypoint_visibility_labels"},
            None,
            {"reason": "no_keypoint_visibility_labels"},
        )

    vis_thr = float(labels.get("visibility_conf_threshold", 0.35))
    total = len(rows)
    resolved = 0
    invisible_resolved = 0
    false_visible = 0
    brier_terms: List[float] = []

    for item in rows:
        try:
            fidx = int(item["frame"])
            x, y = float(item["xy"][0]), float(item["xy"][1])
            j = int(item["joint"])
            gt_visible = bool(item["visible"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        md = float(item.get("max_dist_px", labels.get("anchor_max_dist_px", 80.0)))
        picked = _track_for_anchor(frames_by_idx.get(fidx, {}), x, y, md)
        if picked is None:
            continue
        _tid, tdata, _dist = picked
        conf = _extract_joint_conf(tdata, j)
        if conf is None:
            continue
        resolved += 1
        pred_visible = conf >= vis_thr
        if not gt_visible:
            invisible_resolved += 1
            if pred_visible:
                false_visible += 1
        y_true = 1.0 if gt_visible else 0.0
        brier_terms.append((conf - y_true) ** 2)

    coverage = float(resolved) / float(max(total, 1))

    if invisible_resolved == 0:
        visibility_score = None
        visibility_details = {
            "reason": "no_invisible_labels_resolved",
            "total_labels": total,
            "resolved_labels": resolved,
            "coverage": coverage,
        }
    else:
        fp_rate = float(false_visible) / float(invisible_resolved)
        visibility_score = _clamp01((1.0 - fp_rate) * coverage)
        visibility_details = {
            "total_labels": total,
            "resolved_labels": resolved,
            "coverage": coverage,
            "invisible_resolved": invisible_resolved,
            "false_visible": false_visible,
            "false_visible_rate": fp_rate,
            "visibility_conf_threshold": vis_thr,
        }

    if not brier_terms:
        calib_score = None
        calib_details = {
            "reason": "no_resolved_labels_for_calibration",
            "total_labels": total,
            "resolved_labels": resolved,
            "coverage": coverage,
        }
    else:
        brier = sum(brier_terms) / float(len(brier_terms))
        calib_score = _clamp01((1.0 - brier) * coverage)
        calib_details = {
            "total_labels": total,
            "resolved_labels": resolved,
            "coverage": coverage,
            "brier": brier,
        }

    return visibility_score, visibility_details, calib_score, calib_details


def _resolve_objective_weights(
    seq_spec: Dict[str, Any],
    defaults: Dict[str, float],
) -> Dict[str, float]:
    seq_w = seq_spec.get("objective_weights") or {}
    keys = (
        "mot",
        "expectation",
        "reid_reemergence",
        "visibility_honesty",
        "confidence_calibration",
    )
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = max(float(seq_w.get(k, defaults.get(k, 0.0 if k != "mot" else 1.0))), 0.0)
    return out


def load_sweep_config_v2(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    order = raw.get("sequence_order")
    seqs = raw.get("sequences") or {}
    if not order:
        order = list(seqs.keys())
    for name in order:
        if name not in seqs:
            raise ValueError(f"sequence_order references missing sequence: {name}")
    defaults = raw.get("objective_weights") or {}
    default_weights = {
        "mot": float(defaults.get("mot", 1.0)),
        "expectation": float(defaults.get("expectation", 0.0)),
        "reid_reemergence": float(defaults.get("reid_reemergence", 0.0)),
        "visibility_honesty": float(defaults.get("visibility_honesty", 0.0)),
        "confidence_calibration": float(defaults.get("confidence_calibration", 0.0)),
    }
    return order, seqs, default_weights


def _score_one_sequence_v2(
    *,
    seq_name: str,
    seq_spec: Dict[str, Any],
    params_yaml: Path,
    run_dir: Path,
    timeout_s: Optional[int],
    save_phase_previews: bool,
    default_objective_weights: Dict[str, float],
    expected_tracker_engine: Optional[str],
) -> Dict[str, Any]:
    video = v1._resolve_path(seq_spec["video"])
    gt_mot = v1._resolve_path(seq_spec["gt_mot"])
    im_w = int(seq_spec.get("im_width", 1280))
    im_h = int(seq_spec.get("im_height", 720))

    mot_score, mot_metrics, runtime_meta = v1._score_one_video(
        seq_name,
        video,
        gt_mot,
        im_w,
        im_h,
        params_yaml,
        run_dir,
        timeout_s,
        save_phase_previews=save_phase_previews,
        expected_tracker_engine=expected_tracker_engine,
    )

    data_json_path = run_dir / "data.json"
    exp_score: Optional[float] = None
    exp_details: Optional[Dict[str, Any]] = None
    benchmark_gt_yaml = seq_spec.get("benchmark_gt_yaml")
    bench_path: Optional[Path] = None
    if benchmark_gt_yaml:
        bench_path = v1._resolve_path(benchmark_gt_yaml)
        if not bench_path.is_file():
            exp_details = {"error": f"benchmark_gt_yaml not found: {bench_path}"}
            exp_score = 0.0
        elif not data_json_path.is_file():
            exp_details = {"error": "missing data.json for expectation scoring"}
            exp_score = 0.0
        else:
            exp_score, exp_details = _expectation_score_from_data_json(
                data_json_path=data_json_path,
                benchmark_gt_yaml_path=bench_path,
            )

    reid_reemergence_score: Optional[float] = None
    visibility_honesty_score: Optional[float] = None
    confidence_calibration_score: Optional[float] = None
    reid_reemergence_details: Optional[Dict[str, Any]] = None
    visibility_honesty_details: Optional[Dict[str, Any]] = None
    confidence_calibration_details: Optional[Dict[str, Any]] = None

    if data_json_path.is_file():
        labels = _load_sweep_v2_labels(seq_spec, bench_path)
        if labels:
            data = json.loads(data_json_path.read_text(encoding="utf-8"))
            frames_by_idx = _frame_track_index(data)
            reid_reemergence_score, reid_reemergence_details = _score_reid_reemergence(
                labels=labels,
                frames_by_idx=frames_by_idx,
            )
            (
                visibility_honesty_score,
                visibility_honesty_details,
                confidence_calibration_score,
                confidence_calibration_details,
            ) = _score_visibility_and_calibration(
                labels=labels,
                frames_by_idx=frames_by_idx,
            )
        else:
            reid_reemergence_details = {"reason": "no_sweep_v2_labels"}
            visibility_honesty_details = {"reason": "no_sweep_v2_labels"}
            confidence_calibration_details = {"reason": "no_sweep_v2_labels"}

    obj_w = _resolve_objective_weights(seq_spec, default_objective_weights)
    terms: List[Tuple[float, float]] = [(obj_w["mot"], mot_score)]
    if exp_score is not None:
        terms.append((obj_w["expectation"], exp_score))
    if reid_reemergence_score is not None:
        terms.append((obj_w["reid_reemergence"], reid_reemergence_score))
    if visibility_honesty_score is not None:
        terms.append((obj_w["visibility_honesty"], visibility_honesty_score))
    if confidence_calibration_score is not None:
        terms.append((obj_w["confidence_calibration"], confidence_calibration_score))
    combined = _weighted_average(terms)

    return {
        "sequence_score": combined,
        "mot_score": mot_score,
        "mot_metrics": mot_metrics,
        "expectation_score": exp_score,
        "expectation_details": exp_details,
        "reid_reemergence_score": reid_reemergence_score,
        "reid_reemergence_details": reid_reemergence_details,
        "visibility_honesty_score": visibility_honesty_score,
        "visibility_honesty_details": visibility_honesty_details,
        "confidence_calibration_score": confidence_calibration_score,
        "confidence_calibration_details": confidence_calibration_details,
        "objective_weights": obj_w,
        "runtime_meta": runtime_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna Phase 1-3 multi-video sweep (v2 objective)")
    parser.add_argument(
        "--config",
        type=Path,
        default=v1.REPO_ROOT / "data" / "ground_truth" / "sweep_sequences.yaml",
        help="YAML with sequence_order + sequences (+ optional v2 objective fields)",
    )
    parser.add_argument("--n-trials", type=int, default=None, metavar="N")
    parser.add_argument("--stop-file", type=Path, default=None, metavar="PATH")
    parser.add_argument("--study-name", type=str, default="sweep_v2")
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optuna storage URL (default: sqlite under output/sweeps/optuna/)",
    )
    parser.add_argument("--show-best", action="store_true")
    parser.add_argument(
        "--timeout-per-video",
        type=int,
        default=0,
        help="Subprocess timeout seconds per video (0 = none)",
    )
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--status-json", type=Path, default=None)
    parser.add_argument("--no-status-json", action="store_true")
    parser.add_argument("--phase-previews", action="store_true")
    parser.add_argument(
        "--skip-phase13-coverage-gate",
        action="store_true",
        help="Skip fail-fast coverage gate for phase 1-3 owned keys.",
    )
    args = parser.parse_args()

    import optuna
    from optuna.pruners import NopPruner
    from optuna.samplers import TPESampler

    opt_dir = v1.REPO_ROOT / "output" / "sweeps" / "optuna"
    opt_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(opt_dir / 'sweep.db').resolve()}"

    if args.show_best:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        try:
            print(f"Best value: {study.best_value}")
            print(f"Best params: {study.best_params}")
        except ValueError:
            print("No completed trials yet.", file=sys.stderr)
            sys.exit(1)
        return

    if not args.config.is_file():
        print(
            f"Missing config: {args.config}\n"
            f"Copy {v1.REPO_ROOT / 'data' / 'ground_truth' / 'sweep_sequences.example.yaml'} "
            f"to sweep_sequences.yaml and add your videos.",
            file=sys.stderr,
        )
        sys.exit(2)

    order, seq_specs, default_obj_weights = load_sweep_config_v2(args.config)
    for sid in order:
        spec = seq_specs[sid]
        video = v1._resolve_path(spec["video"])
        gt_mot = v1._resolve_path(spec["gt_mot"])
        if not video.is_file():
            print(f"Missing video for {sid}: {video}", file=sys.stderr)
            sys.exit(2)
        if not gt_mot.is_file():
            print(f"Missing gt_mot for {sid}: {gt_mot}", file=sys.stderr)
            sys.exit(2)
        bench = spec.get("benchmark_gt_yaml")
        if bench:
            bench_path = v1._resolve_path(bench)
            if not bench_path.is_file():
                print(f"Missing benchmark_gt_yaml for {sid}: {bench_path}", file=sys.stderr)
                sys.exit(2)
        labels_yml = spec.get("sweep_v2_labels_yaml")
        if labels_yml:
            lp = v1._resolve_path(labels_yml)
            if not lp.is_file():
                print(f"Missing sweep_v2_labels_yaml for {sid}: {lp}", file=sys.stderr)
                sys.exit(2)

    log_jsonl = args.log_jsonl or (opt_dir / "sweep_log_v2.jsonl")
    status_path: Optional[Path] = None
    if not args.no_status_json:
        status_path = (args.status_json or (opt_dir / "sweep_status_v2.json")).resolve()
    available_weights = v1._available_yolo_weights()
    stop_path = (args.stop_file if args.stop_file is not None else (opt_dir / "STOP_V2")).resolve()
    coverage_report = v1.phase13_search_space_coverage_report()
    if coverage_report["missing_keys"] and not args.skip_phase13_coverage_gate:
        raise SystemExit(
            "Phase1-3 coverage gate failed. Missing keys: "
            + ", ".join(coverage_report["missing_keys"])
        )
    print(
        "[coverage] phase1-3 owned="
        f"{coverage_report['owned_key_count']} tuned={coverage_report['tuned_key_count']} "
        f"fixed={coverage_report['fixed_key_count']} full={coverage_report['is_full_coverage']}",
        flush=True,
    )
    timeout = args.timeout_per_video if args.timeout_per_video > 0 else None
    fail_fast = str(os.environ.get("SWAY_SWEEP_FAIL_FAST", "1")).strip().lower() not in {"0", "false", "no", "off"}
    prune_on_unknown_runtime = (
        str(os.environ.get("SWAY_SWEEP_PRUNE_ON_UNKNOWN_RUNTIME", "1")).strip().lower()
        not in {"0", "false", "no", "off"}
    )
    prune_on_engine_mismatch = (
        str(os.environ.get("SWAY_SWEEP_PRUNE_ON_ENGINE_MISMATCH", "1")).strip().lower()
        not in {"0", "false", "no", "off"}
    )
    floor_score_eps = float(os.environ.get("SWAY_SWEEP_FLOOR_EPS", str(v1.FLOOR_SCORE_EPS_DEFAULT)))
    bigtest_slow_seconds = float(os.environ.get("SWAY_SWEEP_BIGTEST_SLOW_SEC", str(v1.BIGTEST_SLOW_SECONDS_DEFAULT)))
    seq_weights = [float(seq_specs[n].get("weight", 1.0)) for n in order]

    status_meta: Dict[str, Any] = {
        "config": str(args.config.resolve()),
        "sequence_order": list(order),
        "log_jsonl": str(log_jsonl.resolve()),
        "storage": storage,
        "git_sha": v1._git_sha_short(),
        "objective_defaults": default_obj_weights,
        "phase13_coverage": coverage_report,
        "fail_fast": {
            "enabled": fail_fast,
            "prune_on_unknown_runtime": prune_on_unknown_runtime,
            "prune_on_engine_mismatch": prune_on_engine_mismatch,
            "floor_score_eps": floor_score_eps,
            "bigtest_slow_seconds": bigtest_slow_seconds,
            "allowed_tracker_engines": v1._allowed_tracker_engines(),
        },
    }

    n_trials_cap = args.n_trials if (args.n_trials is None or args.n_trials > 0) else None

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.time()
        env_map = v1.suggest_env_for_trial(trial, available_weights)
        per_seq_scores: List[float] = []
        seq_breakdown: Dict[str, Any] = {}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(opt_dir)
        ) as tf:
            params_path = Path(tf.name)
        try:
            v1._write_params_yaml(env_map, params_path)
            for seq_name in order:
                spec = seq_specs[seq_name]
                run_dir = opt_dir / f"trial_{trial.number:05d}_{seq_name}"
                t0 = time.time()
                seq_info = _score_one_sequence_v2(
                    seq_name=seq_name,
                    seq_spec=spec,
                    params_yaml=params_path,
                    run_dir=run_dir,
                    timeout_s=timeout,
                    save_phase_previews=bool(args.phase_previews),
                    default_objective_weights=default_obj_weights,
                    expected_tracker_engine=env_map.get("SWAY_TRACKER_ENGINE"),
                )
                seq_info["duration_s"] = round(time.time() - t0, 1)
                runtime_meta = seq_info.get("runtime_meta") or {}
                runtime_meta["floor_score_hit"] = bool(float(seq_info["sequence_score"]) <= floor_score_eps)
                runtime_meta["slow_path_hit"] = bool(
                    seq_name == "bigtest" and float(seq_info["duration_s"]) >= bigtest_slow_seconds
                )
                seq_breakdown[seq_name] = seq_info
                per_seq_scores.append(float(seq_info["sequence_score"]))

                # Persist sequence-level runtime health even if the trial gets pruned early.
                trial.set_user_attr(f"score_{seq_name}", float(seq_info["sequence_score"]))
                trial.set_user_attr(f"mot_{seq_name}", float(seq_info["mot_score"]))
                trial.set_user_attr(f"duration_s_{seq_name}", float(seq_info["duration_s"]))
                trial.set_user_attr(
                    f"runtime_mismatch_count_{seq_name}",
                    int(runtime_meta.get("requested_runtime_mismatch_count") or 0),
                )
                trial.set_user_attr(f"engine_mismatch_{seq_name}", int(bool(runtime_meta.get("engine_mismatch"))))
                trial.set_user_attr(f"unknown_runtime_hit_{seq_name}", int(bool(runtime_meta.get("unknown_runtime_hit"))))
                trial.set_user_attr(f"timeout_hit_{seq_name}", int(bool(runtime_meta.get("timeout_hit"))))
                trial.set_user_attr(f"floor_score_hit_{seq_name}", int(bool(runtime_meta.get("floor_score_hit"))))
                trial.set_user_attr(f"slow_path_hit_{seq_name}", int(bool(runtime_meta.get("slow_path_hit"))))
                if runtime_meta.get("tracker_path"):
                    trial.set_user_attr(f"runtime_tracker_{seq_name}", str(runtime_meta.get("tracker_path")))

                fail_reasons: List[str] = []
                if bool(runtime_meta.get("timeout_hit")):
                    fail_reasons.append(f"{seq_name}:timeout")
                if bool(runtime_meta.get("floor_score_hit")):
                    fail_reasons.append(f"{seq_name}:floor_score")
                if bool(runtime_meta.get("slow_path_hit")):
                    fail_reasons.append(f"{seq_name}:slow_path")
                if prune_on_unknown_runtime and bool(runtime_meta.get("unknown_runtime_hit")):
                    fail_reasons.append(f"{seq_name}:unknown_runtime")
                if prune_on_engine_mismatch and bool(runtime_meta.get("engine_mismatch")):
                    fail_reasons.append(f"{seq_name}:engine_mismatch")
                if fail_fast and fail_reasons:
                    reason = ",".join(fail_reasons)
                    trial.set_user_attr("fail_fast_pruned", 1)
                    trial.set_user_attr("fail_fast_reason", reason)
                    print(f"[sweep-v2] Trial {trial.number} fail-fast prune ({reason})", flush=True)
                    raise optuna.exceptions.TrialPruned()

            floored = [max(s, 1e-6) for s in per_seq_scores]
            total_w = sum(seq_weights)
            agg = total_w / sum(w / s for w, s in zip(seq_weights, floored))
            trial_dur = round(time.time() - trial_start, 1)

            for seq_name in order:
                seq_info = seq_breakdown[seq_name]
                trial.set_user_attr(f"score_{seq_name}", float(seq_info["sequence_score"]))
                trial.set_user_attr(f"mot_{seq_name}", float(seq_info["mot_score"]))
                runtime_meta = seq_info.get("runtime_meta") or {}
                trial.set_user_attr(
                    f"runtime_mismatch_count_{seq_name}",
                    int(runtime_meta.get("requested_runtime_mismatch_count") or 0),
                )
                if runtime_meta.get("tracker_path"):
                    trial.set_user_attr(f"runtime_tracker_{seq_name}", str(runtime_meta.get("tracker_path")))
                if seq_info["expectation_score"] is not None:
                    trial.set_user_attr(
                        f"exp_{seq_name}",
                        float(seq_info["expectation_score"]),
                    )
                if seq_info["reid_reemergence_score"] is not None:
                    trial.set_user_attr(
                        f"reid_reemerg_{seq_name}",
                        float(seq_info["reid_reemergence_score"]),
                    )
                if seq_info["visibility_honesty_score"] is not None:
                    trial.set_user_attr(
                        f"vis_honesty_{seq_name}",
                        float(seq_info["visibility_honesty_score"]),
                    )
                if seq_info["confidence_calibration_score"] is not None:
                    trial.set_user_attr(
                        f"calib_{seq_name}",
                        float(seq_info["confidence_calibration_score"]),
                    )
                trial.set_user_attr(f"duration_s_{seq_name}", float(seq_info["duration_s"]))
            trial.set_user_attr("aggregate_harmonic_mean", agg)
            trial.set_user_attr("trial_duration_s", trial_dur)
            total_mismatch = int(
                sum(
                    int(((seq_breakdown.get(n) or {}).get("runtime_meta") or {}).get("requested_runtime_mismatch_count") or 0)
                    for n in order
                )
            )
            trial.set_user_attr("runtime_mismatch_count_total", total_mismatch)
            total_engine_mismatch = int(
                sum(int(bool(((seq_breakdown.get(n) or {}).get("runtime_meta") or {}).get("engine_mismatch"))) for n in order)
            )
            total_unknown_runtime = int(
                sum(int(bool(((seq_breakdown.get(n) or {}).get("runtime_meta") or {}).get("unknown_runtime_hit"))) for n in order)
            )
            total_timeout_hit = int(
                sum(int(bool(((seq_breakdown.get(n) or {}).get("runtime_meta") or {}).get("timeout_hit"))) for n in order)
            )
            total_floor_hit = int(
                sum(int(bool(((seq_breakdown.get(n) or {}).get("runtime_meta") or {}).get("floor_score_hit"))) for n in order)
            )
            total_slow_hit = int(
                sum(int(bool(((seq_breakdown.get(n) or {}).get("runtime_meta") or {}).get("slow_path_hit"))) for n in order)
            )
            trial.set_user_attr("engine_mismatch", int(total_engine_mismatch > 0))
            trial.set_user_attr("engine_mismatch_count_total", total_engine_mismatch)
            trial.set_user_attr("unknown_runtime_hit", int(total_unknown_runtime > 0))
            trial.set_user_attr("unknown_runtime_hit_count_total", total_unknown_runtime)
            trial.set_user_attr("timeout_hit", int(total_timeout_hit > 0))
            trial.set_user_attr("timeout_hit_count_total", total_timeout_hit)
            trial.set_user_attr("floor_score_hit", int(total_floor_hit > 0))
            trial.set_user_attr("floor_score_hit_count_total", total_floor_hit)
            trial.set_user_attr("slow_path_hit", int(total_slow_hit > 0))
            trial.set_user_attr("slow_path_hit_count_total", total_slow_hit)
            trial.set_user_attr(
                "runtime_signature",
                " || ".join(
                    str((((seq_breakdown.get(n) or {}).get("runtime_meta") or {}).get("runtime_signature") or f"{n}:none"))
                    for n in order
                ),
            )

            entry = {
                "trial": trial.number,
                "git_sha": v1._git_sha_short(),
                "aggregate": agg,
                "trial_duration_s": trial_dur,
                "env": env_map,
                "per_sequence": seq_breakdown,
            }
            with open(log_jsonl, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(entry, default=str) + "\n")
            return agg
        finally:
            params_path.unlink(missing_ok=True)
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True, n_startup_trials=50),
        pruner=NopPruner(),
    )

    if status_path:
        try:
            from sway.optuna_live_status import write_live_sweep_status

            write_live_sweep_status(study, status_path, extra=status_meta)
            print(f"Live status JSON: {status_path}", flush=True)
        except Exception as ex:
            print(f"(status-json initial write failed: {ex})", flush=True)

    def _on_signal(_signum: int, _frame: Any) -> None:
        print(
            "\nStop requested - current trial will finish, then the sweep exits.",
            flush=True,
        )
        study.stop()

    try:
        signal.signal(signal.SIGINT, _on_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _on_signal)
    except ValueError:
        pass

    def _after_trial(study_obj: Any, _finished_trial: Any) -> None:
        if status_path:
            try:
                from sway.optuna_live_status import write_live_sweep_status

                write_live_sweep_status(study_obj, status_path, extra=status_meta)
            except Exception as ex:
                print(f"(status-json write failed: {ex})", flush=True)
        if stop_path.is_file():
            print(
                f"\nStop file detected ({stop_path}); stopping after this trial.",
                flush=True,
            )
            study_obj.stop()
            try:
                stop_path.unlink(missing_ok=True)
            except OSError:
                pass

    if n_trials_cap is None:
        print(
            "Sweep v2: unlimited trials. Stop with Ctrl+C / SIGTERM, or: "
            f"touch {stop_path}",
            flush=True,
        )
    else:
        print(f"Sweep v2: will stop after {n_trials_cap} trial(s).", flush=True)

    study.optimize(objective, n_trials=n_trials_cap, callbacks=[_after_trial])

    print("=== Done ===")
    try:
        print(f"Best value (harmonic mean): {study.best_value}")
        print(f"Best params: {study.best_params}")
    except ValueError:
        print("No completed trials in this study yet.", file=sys.stderr)


if __name__ == "__main__":
    main()

