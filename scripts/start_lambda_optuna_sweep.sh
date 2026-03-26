#!/usr/bin/env bash
# Start Phase 1–3 Optuna sweep in tmux on Lambda (repo root = $ROOT).
# Usage on instance:  bash scripts/start_lambda_optuna_sweep.sh
# Requires: git pull done, data/ground_truth/sweep_sequences.yaml, models, CUDA torch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SESSION="${SWEEP_TMUX_SESSION:-sway_sweep}"

export SWAY_SERVER_PERF="${SWAY_SERVER_PERF:-1}"
mkdir -p output/sweeps/optuna

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists — attach: tmux attach -t $SESSION"
  exit 1
fi

CMD="cd '$ROOT' && export SWAY_SERVER_PERF=1 && python -u -m tools.auto_sweep --config data/ground_truth/sweep_sequences.yaml 2>&1 | tee -a output/sweeps/optuna/sweep_runner.log"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "Started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $ROOT/output/sweeps/optuna/sweep_runner.log"
echo "  JSON:   $ROOT/output/sweeps/optuna/sweep_status.json (updated each trial)"
echo "  stop:   touch $ROOT/output/sweeps/optuna/STOP"
