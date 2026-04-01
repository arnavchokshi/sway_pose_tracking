#!/usr/bin/env bash
# Stop (and optionally wipe) the Phase 1–3 Optuna sweep on Lambda.
# Same host/pem convention as scripts/pull_lambda_sweep_status.sh and start_lambda_optuna_sweep.sh.
#
# Usage:
#   bash scripts/stop_lambda_sweep.sh <lambda-ip> [path/to/pose-tracking.pem]
#
# Env:
#   SWEEP_STOP_MODE=kill       default — kill tmux session sway_sweep and auto_sweep processes immediately
#   SWEEP_STOP_MODE=graceful   only touch output/sweeps/optuna/STOP (current trial finishes, then exit)
#   SWEEP_CLEAR=1 CONFIRM_CLEAR=1  after stop, remove sweep.db, trial_*, logs, status under optuna/ (fresh start)
#
# Remote paths match REPO layout: ~/sway_test/sway_pose_mvp/output/sweeps/optuna/
set -euo pipefail
IP="${1:?usage: $0 <lambda-ip> [pem-path]}"
PEM="${2:-$HOME/Downloads/pose-tracking.pem}"
REMOTE_ROOT="${SWEEP_REMOTE_ROOT:-~/sway_test/sway_pose_mvp}"
OPT_REL="output/sweeps/optuna"
SESSION="${SWEEP_TMUX_SESSION:-sway_sweep}"
MODE="${SWEEP_STOP_MODE:-kill}"

SSH=(ssh -i "$PEM" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 "ubuntu@${IP}")

graceful_cmd() {
  cat <<EOF
set -e
cd ${REMOTE_ROOT}
touch ${OPT_REL}/STOP
echo "Graceful stop: touched ${OPT_REL}/STOP (current trial will finish, then sweep exits)."
EOF
}

kill_cmd() {
  cat <<EOF
set -e
cd ${REMOTE_ROOT}
# tmux session from start_lambda_optuna_sweep.sh
tmux kill-session -t ${SESSION} 2>/dev/null || true
pkill -f 'tools.auto_sweep_v2' 2>/dev/null || true
pkill -f 'tools.auto_sweep' 2>/dev/null || true
echo "Kill: tmux session '${SESSION}' (if any) and auto_sweep PIDs signaled."
EOF
}

clear_cmd() {
  cat <<EOF
set -e
cd ${REMOTE_ROOT}
O=${OPT_REL}
rm -f "\$O/STOP" "\$O/STOP_V2" 2>/dev/null || true
rm -f "\$O/sweep_status.json" "\$O/sweep_log.jsonl" "\$O/sweep_runner.log" 2>/dev/null || true
rm -f "\$O"/sweep.db* 2>/dev/null || true
rm -rf "\$O"/trial_* 2>/dev/null || true
echo "Clear: removed optuna trial dirs, DB, logs, status under \$O"
EOF
}

REMOTE_SCRIPT=""
case "$MODE" in
  graceful)
    REMOTE_SCRIPT="$(graceful_cmd)"
    ;;
  kill)
    REMOTE_SCRIPT="$(kill_cmd)"
    ;;
  *)
    echo "SWEEP_STOP_MODE must be 'kill' or 'graceful' (got: $MODE)" >&2
    exit 2
    ;;
esac

if [[ "${SWEEP_CLEAR:-0}" == "1" ]]; then
  if [[ "${CONFIRM_CLEAR:-0}" != "1" ]]; then
    echo "Refusing clear: set CONFIRM_CLEAR=1 with SWEEP_CLEAR=1 to delete remote sweep.db and trial_*." >&2
    exit 3
  fi
  REMOTE_SCRIPT="${REMOTE_SCRIPT}
$(clear_cmd)"
fi

"${SSH[@]}" bash -s <<REMOTE
$REMOTE_SCRIPT
REMOTE

echo "Done (ubuntu@${IP})."
