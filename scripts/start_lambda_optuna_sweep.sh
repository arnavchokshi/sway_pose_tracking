#!/usr/bin/env bash
# Start Phase 1–3 Optuna sweep (v2 objective) in tmux on Lambda.
#
# Usage:
#   bash scripts/start_lambda_optuna_sweep.sh
#
# Env overrides (all optional):
#   SWEEP_STUDY_NAME       Optuna study name  (default: sweep_v4_phase13_fulltech)
#   SWEEP_TMUX_SESSION     tmux session name  (default: sway_sweep)
#   SWAY_SERVER_PERF       pass 0 to disable GPU/CPU perf tweaks (default: 1)
#   SWAY_SWEEP_PHASE_PREVIEWS  pass 1 to save phase preview videos (default: 0)
#   SWEEP_CONFIG_PATH      sweep config YAML (default: data/ground_truth/sweep_sequences.yaml)
#   SWEEP_TIMEOUT_PER_VIDEO per-sequence timeout seconds (default: 420)
#   SWAY_SWEEP_ALLOWED_ENGINES comma-separated engine allow-list (default: solidtrack,matr)
#   SWEEP_SKIP_COVERAGE_GATE set 1 to pass --skip-phase13-coverage-gate
#   SWAY_YOLO_HALF         override FP16 flag (auto-set by server_runtime_perf when CUDA present)
#   SWAY_YOLO_INFER_BATCH  override YOLO batch size (auto-set to 4 on CUDA)
#
# Requires: git pull done, data/ground_truth/sweep_sequences.yaml populated, models downloaded.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SESSION="${SWEEP_TMUX_SESSION:-sway_sweep}"
STUDY="${SWEEP_STUDY_NAME:-sweep_v4_phase13_fulltech}"
CONFIG_PATH="${SWEEP_CONFIG_PATH:-data/ground_truth/sweep_sequences.yaml}"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi

export SWAY_SERVER_PERF="${SWAY_SERVER_PERF:-1}"
export SWAY_SWEEP_ALLOWED_ENGINES="${SWAY_SWEEP_ALLOWED_ENGINES:-solidtrack,matr}"
TIMEOUT_PER_VIDEO="${SWEEP_TIMEOUT_PER_VIDEO:-420}"
mkdir -p output/sweeps/optuna

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists — attach: tmux attach -t $SESSION"
  exit 1
fi

EXTRA=" --study-name $STUDY --config $CONFIG_PATH --status-json output/sweeps/optuna/sweep_status.json --log-jsonl output/sweeps/optuna/sweep_log.jsonl --stop-file output/sweeps/optuna/STOP --timeout-per-video $TIMEOUT_PER_VIDEO"
if [[ "${SWAY_SWEEP_PHASE_PREVIEWS:-0}" == "1" ]]; then
  EXTRA="$EXTRA --phase-previews"
fi
if [[ "${SWEEP_SKIP_COVERAGE_GATE:-0}" == "1" ]]; then
  EXTRA="$EXTRA --skip-phase13-coverage-gate"
fi
CMD="cd '$ROOT' && export SWAY_SERVER_PERF=1 && '$PY' -u -m tools.auto_sweep_v2$EXTRA 2>&1 | tee -a output/sweeps/optuna/sweep_runner.log"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "Started sweep:"
echo "  study:  $STUDY"
echo "  tmux:   tmux attach -t $SESSION"
echo "  log:    tail -f $ROOT/output/sweeps/optuna/sweep_runner.log"
echo "  status: $ROOT/output/sweeps/optuna/sweep_status.json (updated each trial)"
echo "  stop:   touch $ROOT/output/sweeps/optuna/STOP"
