#!/usr/bin/env bash
# Run ON Lambda after setup_lambda_training.sh. Downloads DanceTrack from Hugging Face,
# converts to YOLO format, trains --phase dancetrack. No upload of dataset from your Mac.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "==> Repo root: $ROOT"

python3 scripts/phase2_public_training/fetch_dancetrack_hf.py
python3 scripts/phase2_public_training/convert_dancetrack_to_yolo.py
python3 scripts/phase2_public_training/train_yolo11x.py --phase dancetrack

BEST="$ROOT/runs/detect/yolo11x_dancetrack_only/weights/best.pt"
echo ""
echo "==> Training finished."
echo "    Weights: $BEST"
echo "    Copy to your Mac (from your laptop terminal, not Lambda):"
echo "    scp ubuntu@<LAMBDA_IP>:$BEST ./yolo11x_dancetrack_only_best.pt"
echo "Then: mkdir -p models && mv yolo11x_dancetrack_only_best.pt models/yolo11x_dancetrack.pt"
echo "      export SWAY_YOLO_WEIGHTS=models/yolo11x_dancetrack.pt"
