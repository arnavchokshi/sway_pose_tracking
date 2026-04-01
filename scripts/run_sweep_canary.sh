#!/usr/bin/env bash
# Run a short canary sweep and enforce acceptance gates.
#
# Usage:
#   bash scripts/run_sweep_canary.sh
#
# Optional env:
#   SWEEP_STUDY_NAME
#   SWEEP_CONFIG_PATH
#   SWEEP_CANARY_TRIALS           (default: 3)
#   SWEEP_TIMEOUT_PER_VIDEO       (default: 420)
#   SWAY_SWEEP_ALLOWED_ENGINES    (default: solidtrack,matr)
#   SWEEP_STATUS_JSON             (default: output/sweeps/optuna/sweep_status.json)
#   SWEEP_LOG_JSONL               (default: output/sweeps/optuna/sweep_log.jsonl)
#   SWEEP_RUNNER_LOG              (default: output/sweeps/optuna/sweep_runner.log)
#   SWEEP_MAX_TRIAL_DURATION_S    (default: 1200)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi

STUDY="${SWEEP_STUDY_NAME:-sweep_v4_phase13_canary}"
CONFIG_PATH="${SWEEP_CONFIG_PATH:-data/ground_truth/sweep_sequences.yaml}"
CANARY_TRIALS="${SWEEP_CANARY_TRIALS:-3}"
TIMEOUT_PER_VIDEO="${SWEEP_TIMEOUT_PER_VIDEO:-420}"
MAX_TRIAL_DUR="${SWEEP_MAX_TRIAL_DURATION_S:-1200}"
STATUS_JSON="${SWEEP_STATUS_JSON:-output/sweeps/optuna/sweep_status.json}"
LOG_JSONL="${SWEEP_LOG_JSONL:-output/sweeps/optuna/sweep_log.jsonl}"
RUNNER_LOG="${SWEEP_RUNNER_LOG:-output/sweeps/optuna/sweep_runner.log}"

export SWAY_SWEEP_ALLOWED_ENGINES="${SWAY_SWEEP_ALLOWED_ENGINES:-solidtrack,matr}"
export SWAY_SWEEP_FAIL_FAST="${SWAY_SWEEP_FAIL_FAST:-1}"
export SWAY_SWEEP_PRUNE_ON_UNKNOWN_RUNTIME="${SWAY_SWEEP_PRUNE_ON_UNKNOWN_RUNTIME:-1}"
export SWAY_SWEEP_PRUNE_ON_ENGINE_MISMATCH="${SWAY_SWEEP_PRUNE_ON_ENGINE_MISMATCH:-1}"

mkdir -p output/sweeps/optuna

echo "[canary] running $CANARY_TRIALS trial(s) study=$STUDY timeout=$TIMEOUT_PER_VIDEO"
"$PY" -u -m tools.auto_sweep_v2 \
  --study-name "$STUDY" \
  --config "$CONFIG_PATH" \
  --n-trials "$CANARY_TRIALS" \
  --timeout-per-video "$TIMEOUT_PER_VIDEO" \
  --status-json "$STATUS_JSON" \
  --log-jsonl "$LOG_JSONL" \
  --stop-file output/sweeps/optuna/STOP_CANARY \
  2>&1 | tee -a "$RUNNER_LOG"

echo "[canary] validating acceptance gates"
"$PY" -m tools.validate_sweep_canary \
  --status-json "$STATUS_JSON" \
  --last-n "$CANARY_TRIALS" \
  --max-trial-duration-s "$MAX_TRIAL_DUR" \
  --api-log "$RUNNER_LOG"

echo "[canary] PASS"

