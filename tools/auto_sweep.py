#!/usr/bin/env python3
"""
Optuna TPE sweep for Phase 1–3 (``--stop-after-boundary after_phase_3``).

- Multi-video objective: harmonic mean of per-video composite scores (§2.1 playbook).
- MedianPruner: ``trial.report`` after **each** video so bad configs skip remaining clips.
- Handshake SAM IoU: set via ``SWAY_*`` in ``--params`` YAML (CLI), not Lab API.

Config: copy ``data/ground_truth/sweep_sequences.example.yaml`` → ``sweep_sequences.yaml``.

  python -m tools.auto_sweep --config data/ground_truth/sweep_sequences.yaml
  python -m tools.auto_sweep --n-trials 60 --show-best
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- YOLO weights (playbook §9.1) ---
YOLO_WEIGHT_CHOICES = ("yolo26l", "yolo26l_dancetrack", "yolo26l_dancetrack_crowdhuman")

# AFLink default path (global_track_link.py)
AFLINK_DEFAULT = REPO_ROOT / "models" / "AFLink_epoch20.pth"


def _resolve_path(p: str | Path) -> Path:
    x = Path(p)
    if not x.is_absolute():
        x = (REPO_ROOT / x).resolve()
    return x


def _git_sha_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _weight_file_exists(token: str) -> bool:
    from sway.tracker import _SWAY_YOLO_WEIGHTS_ALIASES

    mapped = _SWAY_YOLO_WEIGHTS_ALIASES.get(token.lower())
    if not mapped:
        return False
    name = Path(mapped).name
    for base in (REPO_ROOT / "models", REPO_ROOT, Path.cwd()):
        if (base / name).is_file():
            return True
    return False


def _available_yolo_weights() -> List[str]:
    return [w for w in YOLO_WEIGHT_CHOICES if _weight_file_exists(w)] or list(YOLO_WEIGHT_CHOICES)


def composite_score(metrics: Dict[str, Any]) -> float:
    """
    Map TrackEval flat dict → [0,1]-ish scalar. Uses best-effort keys (see tests / boxmot ablation).
    """
    if not metrics:
        return 0.0

    def f(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            v = metrics.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        return default

    hota = f("HOTA_HOTA", default=0.0)
    if hota == 0.0:
        for k, v in metrics.items():
            if k.startswith("HOTA_HOTA") and isinstance(v, (int, float)):
                hota = float(v)
                break

    recall = f("CLEAR_CLR_Re", "CLEAR_RECALL", default=0.0)
    if recall == 0.0:
        recall = f("Identity_IDF1", default=hota)

    idsw = int(f("Identity_IDSW", "CLEAR_IDSW", default=0))
    tp = int(f("CLEAR_TP", default=0))
    fn = int(f("CLEAR_FN", default=0))
    total_gt = max(tp + fn, 1)

    frag = 0
    for k, v in metrics.items():
        if "FRAG" in k.upper() and isinstance(v, (int, float)):
            frag = int(v)
            break

    idsw_penalty = min(idsw / max(total_gt * 0.05, 1), 1.0)
    frag_penalty = min(frag / max(total_gt * 0.10, 1), 1.0)

    return (
        0.45 * hota
        + 0.25 * recall
        + 0.20 * (1.0 - idsw_penalty)
        + 0.10 * (1.0 - frag_penalty)
    )


def suggest_env_for_trial(trial: Any, available_weights: List[str]) -> Dict[str, str]:
    """Build SWAY_* env map per docs/GT_DRIVEN_SWEEP_AND_TUNING_PLAYBOOK §9."""
    env: Dict[str, str] = {}

    w = trial.suggest_categorical("SWAY_YOLO_WEIGHTS", available_weights)
    env["SWAY_YOLO_WEIGHTS"] = w

    env["SWAY_YOLO_CONF"] = str(
        trial.suggest_categorical(
            "SWAY_YOLO_CONF", ["0.15", "0.18", "0.22", "0.26", "0.30"]
        )
    )
    env["SWAY_PRETRACK_NMS_IOU"] = str(
        trial.suggest_categorical(
            "SWAY_PRETRACK_NMS_IOU", ["0.40", "0.45", "0.50", "0.55", "0.60"]
        )
    )
    env["SWAY_BOXMOT_MAX_AGE"] = str(
        trial.suggest_categorical("SWAY_BOXMOT_MAX_AGE", ["90", "120", "150", "180"])
    )
    env["SWAY_BOXMOT_MATCH_THRESH"] = str(
        trial.suggest_categorical(
            "SWAY_BOXMOT_MATCH_THRESH", ["0.20", "0.25", "0.30", "0.35"]
        )
    )
    env["SWAY_STITCH_MAX_FRAME_GAP"] = str(
        trial.suggest_categorical(
            "SWAY_STITCH_MAX_FRAME_GAP", ["45", "60", "75", "90", "120"]
        )
    )

    aflink_mode = trial.suggest_categorical(
        "sway_global_aflink_mode", ["neural_if_available", "force_heuristic"]
    )
    if aflink_mode == "force_heuristic" or not AFLINK_DEFAULT.is_file():
        env["SWAY_GLOBAL_AFLINK"] = "0"
    # neural_if_available + weights present → omit SWAY_GLOBAL_AFLINK (allow neural linker)

    mode = trial.suggest_categorical(
        "SWAY_PHASE13_MODE", ["standard", "dancer_registry", "sway_handshake"]
    )
    env["SWAY_PHASE13_MODE"] = mode

    env["SWAY_YOLO_DETECTION_STRIDE"] = "1"

    env["SWAY_USE_BOXMOT"] = "1"
    env["SWAY_BOXMOT_TRACKER"] = "deepocsort"
    env["SWAY_BOXMOT_REID_ON"] = "0"

    if mode == "standard":
        env["SWAY_HYBRID_SAM_IOU_TRIGGER"] = str(
            trial.suggest_categorical(
                "branch_std_SWAY_HYBRID_SAM_IOU_TRIGGER",
                ["0.35", "0.40", "0.42", "0.45", "0.50", "0.55", "0.60"],
            )
        )
    elif mode == "dancer_registry":
        env["SWAY_REGISTRY_TOUCH_IOU"] = str(
            trial.suggest_categorical(
                "branch_reg_SWAY_REGISTRY_TOUCH_IOU",
                ["0.05", "0.08", "0.10", "0.12", "0.15", "0.18", "0.20"],
            )
        )
        env["SWAY_REGISTRY_SWAP_MARGIN"] = str(
            trial.suggest_categorical(
                "branch_reg_SWAY_REGISTRY_SWAP_MARGIN",
                ["0.02", "0.04", "0.06", "0.08", "0.10", "0.12", "0.15"],
            )
        )
        env["SWAY_DORMANT_MAX_GAP"] = str(
            trial.suggest_categorical(
                "branch_reg_SWAY_DORMANT_MAX_GAP", ["90", "120", "150", "180", "200"]
            )
        )
    else:  # sway_handshake
        env["SWAY_HYBRID_SAM_IOU_TRIGGER"] = str(
            trial.suggest_categorical(
                "branch_hs_SWAY_HYBRID_SAM_IOU_TRIGGER",
                ["0.05", "0.10", "0.15", "0.20", "0.25"],
            )
        )
        env["SWAY_HYBRID_SAM_WEAK_CUES"] = "0"

    return env


def _write_params_yaml(env_map: Dict[str, str], path: Path) -> None:
    # main.py: only SWAY_* keys promoted to os.environ from YAML top-level
    block = {k: v for k, v in env_map.items() if k.startswith("SWAY_")}
    path.write_text(yaml.dump(block, default_flow_style=False, sort_keys=True), encoding="utf-8")


def _score_one_video(
    seq_name: str,
    video: Path,
    gt_mot: Path,
    im_w: int,
    im_h: int,
    params_yaml: Path,
    out_dir: Path,
    timeout_s: Optional[int],
) -> Tuple[float, Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        str(video),
        "--output-dir",
        str(out_dir),
        "--params",
        str(params_yaml),
        "--stop-after-boundary",
        "after_phase_3",
    ]
    kwargs: Dict[str, Any] = {"cwd": str(REPO_ROOT)}
    if timeout_s is not None:
        kwargs["timeout"] = timeout_s
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    data_json = out_dir / "data.json"
    if r.returncode != 0 or not data_json.is_file():
        return 0.0, {"error": "pipeline_failed", "stderr_tail": (r.stderr or "")[-800:]}

    data = json.loads(data_json.read_text())
    from sway.mot_format import load_mot_lines_from_file
    from sway.trackeval_runner import run_trackeval_single_sequence

    gt_lines = load_mot_lines_from_file(gt_mot)
    from sway.mot_format import data_json_to_mot_lines

    pr_lines = data_json_to_mot_lines(data)
    try:
        flat = run_trackeval_single_sequence(gt_lines, pr_lines, seq_name, im_w, im_h)
    except RuntimeError as e:
        return 0.0, {"error": str(e)}
    return composite_score(flat), flat


def load_sweep_config(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    raw = yaml.safe_load(path.read_text()) or {}
    order = raw.get("sequence_order")
    seqs = raw.get("sequences") or {}
    if not order:
        order = list(seqs.keys())
    for name in order:
        if name not in seqs:
            raise ValueError(f"sequence_order references missing sequence: {name}")
    return order, seqs


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna Phase 1–3 multi-video sweep")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "data" / "ground_truth" / "sweep_sequences.yaml",
        help="YAML with sequence_order + sequences (see sweep_sequences.example.yaml)",
    )
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--study-name", type=str, default="sway_phase13_v1")
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
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        help="Append one JSON line per completed trial (default: under output/sweeps/optuna/)",
    )
    args = parser.parse_args()

    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    opt_dir = REPO_ROOT / "output" / "sweeps" / "optuna"
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
            f"Copy {REPO_ROOT / 'data' / 'ground_truth' / 'sweep_sequences.example.yaml'} "
            f"to sweep_sequences.yaml and add your videos.",
            file=sys.stderr,
        )
        sys.exit(2)

    order, seq_specs = load_sweep_config(args.config)
    for sid in order:
        spec = seq_specs[sid]
        v = _resolve_path(spec["video"])
        g = _resolve_path(spec["gt_mot"])
        if not v.is_file():
            print(f"Missing video for {sid}: {v}", file=sys.stderr)
            sys.exit(2)
        if not g.is_file():
            print(f"Missing gt_mot for {sid}: {g}", file=sys.stderr)
            sys.exit(2)

    log_jsonl = args.log_jsonl or (opt_dir / "sweep_log.jsonl")
    available_weights = _available_yolo_weights()

    timeout = args.timeout_per_video if args.timeout_per_video > 0 else None

    def objective(trial: optuna.Trial) -> float:
        env_map = suggest_env_for_trial(trial, available_weights)
        per_video_scores: List[float] = []
        all_metrics: Dict[str, Any] = {}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(opt_dir)
        ) as tf:
            params_path = Path(tf.name)
        try:
            _write_params_yaml(env_map, params_path)

            for step, seq_name in enumerate(order):
                spec = seq_specs[seq_name]
                video = _resolve_path(spec["video"])
                gt_mot = _resolve_path(spec["gt_mot"])
                im_w = int(spec.get("im_width", 1280))
                im_h = int(spec.get("im_height", 720))
                run_dir = opt_dir / f"trial_{trial.number:05d}_{seq_name}"
                score, mflat = _score_one_video(
                    seq_name,
                    video,
                    gt_mot,
                    im_w,
                    im_h,
                    params_path,
                    run_dir,
                    timeout,
                )
                per_video_scores.append(score)
                all_metrics[seq_name] = mflat
                trial.report(score, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if min(per_video_scores) <= 0:
                agg = 0.0
            else:
                agg = statistics.harmonic_mean(per_video_scores)
            for seq_name, sc in zip(order, per_video_scores):
                trial.set_user_attr(f"score_{seq_name}", sc)
            trial.set_user_attr("aggregate_harmonic_mean", agg)

            entry = {
                "trial": trial.number,
                "git_sha": _git_sha_short(),
                "aggregate": agg,
                "per_video": dict(zip(order, per_video_scores)),
                "env": env_map,
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
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("=== Done ===")
    print(f"Best value (harmonic mean): {study.best_value}")
    print(f"Best params: {study.best_params}")
    try:
        import optuna.importance

        imp = optuna.importance.get_param_importances(study)
        print("Param importances:", imp)
    except Exception as ex:
        print("(importance analysis skipped:", ex, ")")


if __name__ == "__main__":
    main()
