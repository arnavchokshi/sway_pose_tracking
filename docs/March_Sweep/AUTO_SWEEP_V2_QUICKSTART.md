# Auto Sweep v2 Quickstart

`tools/auto_sweep_v2.py` extends the existing Optuna Phase 1-3 sweep with
label-aware objective terms for identity and critique trust.

## What v2 optimizes

- `mot_score`: existing TrackEval composite from `gt_mot`.
- `expectation_score` (optional): score derived from benchmark checks in
  `tools/benchmark.py` using a per-sequence `benchmark_gt_yaml`.
- `reid_reemergence_score` (optional): anchor-based ID consistency across
  labeled re-emergence events.
- `visibility_honesty_score` (optional): low false-visible behavior for
  invisible joints.
- `confidence_calibration_score` (optional): confidence calibration quality
  from labeled visibility points.

Per-sequence score:

- weighted mean of all configured/available metric terms.

Global score:

- weighted harmonic mean across sequences (same aggregate style as v1).

## Config format

Start from your current `data/ground_truth/sweep_sequences.yaml` and add:

```yaml
objective_weights:
  mot: 0.60
  expectation: 0.15
  reid_reemergence: 0.15
  visibility_honesty: 0.05
  confidence_calibration: 0.05

sequence_order:
  - bigtest
  - gymtest

sequences:
  bigtest:
    video: data/ground_truth/bigtest/video.mp4
    gt_mot: data/ground_truth/bigtest/gt/gt.txt
    im_width: 1920
    im_height: 1080
    weight: 2.0
    benchmark_gt_yaml: benchmarks/BIGTEST_ground_truth.yaml
    sweep_v2_labels_yaml: data/ground_truth/bigtest/sweep_v2_labels.yaml
    objective_weights:
      mot: 0.50
      expectation: 0.15
      reid_reemergence: 0.20
      visibility_honesty: 0.10
      confidence_calibration: 0.05

  gymtest:
    video: data/ground_truth/gymtest/video.mp4
    gt_mot: data/ground_truth/gymtest/gt/gt.txt
    im_width: 1920
    im_height: 1080
    weight: 1.0
    benchmark_gt_yaml: benchmarks/GYMTEST_ground_truth.yaml
```

Notes:

- If `benchmark_gt_yaml` is omitted for a sequence, v2 uses MOT-only scoring for that sequence.
- `sweep_v2_labels` can live either inside `benchmark_gt_yaml` under key
  `sweep_v2_labels`, or in a separate `sweep_v2_labels_yaml` file.
- `objective_weights` can be set globally and optionally overridden per sequence.
- Keep all paths relative to repo root for portability.

## Label schema (for new objective terms)

```yaml
sweep_v2_labels:
  # Optional default for anchor matching in pixels
  anchor_max_dist_px: 80
  # Optional confidence threshold for visible/non-visible classification
  visibility_conf_threshold: 0.35

  # Same dancer before/after occlusion or exit/re-entry.
  reid_reemergence_events:
    - before_frame: 420
      before_xy: [610, 420]
      after_frame: 465
      after_xy: [640, 430]
      max_dist_px: 90

  # Joint-level visibility labels for honesty + calibration terms.
  # joint index follows the pipeline keypoint order.
  keypoint_visibility_labels:
    - frame: 430
      xy: [612, 418]
      joint: 9
      visible: false
    - frame: 430
      xy: [612, 418]
      joint: 5
      visible: true
```

## Run

```bash
python -m tools.auto_sweep_v2 --config data/ground_truth/sweep_sequences.yaml
```

Useful flags:

- `--study-name sweep_v2`
- `--n-trials 40`
- `--phase-previews`
- `--show-best`

Outputs default to:

- `output/sweeps/optuna/sweep_log_v2.jsonl`
- `output/sweeps/optuna/sweep_status_v2.json`

## Overnight Lambda run (phase 1-3 full-tech)

Use the tmux launcher so the run keeps going until you stop it:

```bash
bash scripts/lambda_preflight.sh
export SWAY_SERVER_PERF=1
export SWEEP_STUDY_NAME=sweep_v4_phase13_fulltech
bash scripts/start_lambda_optuna_sweep.sh
```

Monitor:

- `tail -f output/sweeps/optuna/sweep_runner.log`
- `output/sweeps/optuna/sweep_status.json`

Stop in the morning:

```bash
touch output/sweeps/optuna/STOP
```

Notes:

- The launcher runs `tools.auto_sweep_v2` with phase 1-3-only trial execution (`after_phase_3`).
- Coverage gate is on by default (fails fast if phase 1-3 owned key coverage is incomplete).
- Use `SWEEP_SKIP_COVERAGE_GATE=1` only for emergency debugging.

