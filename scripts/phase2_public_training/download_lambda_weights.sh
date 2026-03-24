#!/usr/bin/env bash
# Download DanceTrack-only fine-tuned best.pt from Lambda into models/yolo26l_dancetrack.pt
#
# Usage:
#   chmod +x scripts/phase2_public_training/download_lambda_weights.sh
#   bash scripts/phase2_public_training/download_lambda_weights.sh
#
# Override host or key:
#   LAMBDA_HOST=ubuntu@132.145.211.165 PEM=~/.ssh/key.pem bash scripts/phase2_public_training/download_lambda_weights.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/models"

PEM="${PEM:-/Users/arnavchokshi/Downloads/pose-tracking.pem}"
HOST="${LAMBDA_HOST:-ubuntu@132.145.211.165}"
REMOTE="${REMOTE_PATH:-~/sway_pose_tracking/runs/detect/yolo26l_dancetrack_only/weights/best.pt}"
OUT="$ROOT/models/yolo26l_dancetrack.pt"

if [[ ! -f "$PEM" ]]; then
  echo "PEM not found: $PEM — set PEM=... or place your Lambda SSH key there." >&2
  exit 1
fi

echo "scp $HOST:$REMOTE -> $OUT"
scp -i "$PEM" -o ConnectTimeout=20 -o StrictHostKeyChecking=no \
  "$HOST:$REMOTE" "$OUT"
ls -la "$OUT"
