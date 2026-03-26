#!/usr/bin/env bash
# Pull live Optuna snapshot from a running Lambda box (for a local UI or watch loop).
# Usage:  bash scripts/pull_lambda_sweep_status.sh 150.136.111.175 [path/to/pose-tracking.pem]
set -euo pipefail
IP="${1:?usage: $0 <lambda-ip> [pem-path]}"
PEM="${2:-$HOME/Downloads/pose-tracking.pem}"
OUT="${SWEEP_STATUS_OUT:-./sweep_status.json}"
REMOTE="ubuntu@${IP}:~/sway_pose_tracking/output/sweeps/optuna/sweep_status.json"
scp -i "$PEM" -o StrictHostKeyChecking=accept-new "$REMOTE" "$OUT"
echo "Wrote $OUT"
