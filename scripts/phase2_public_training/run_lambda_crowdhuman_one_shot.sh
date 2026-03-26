#!/usr/bin/env bash
# One command on a fresh Lambda GPU box: venv + deps, then CrowdHuman (HF) → YOLO → train
# from your DanceTrack fine-tuned checkpoint.
#
# BEFORE running (required): upload your existing DanceTrack-trained .pt (no DanceTrack re-train). Use one of:
#   • runs/detect/yolo26l_dancetrack_only/weights/best.pt
#   • models/yolo26l_dancetrack.pt  (same name as the app / Lab)
# or set YOLO_CROWDHUMAN_PARENT_WEIGHTS=/absolute/path/to/your.pt
#
# Disk: ≥ ~22 GiB free on the repo volume before CrowdHuman fetch (see fetch_crowdhuman_hf.py).
#
# From repo root (e.g. ~/sway_pose_tracking):
#   bash scripts/phase2_public_training/run_lambda_crowdhuman_one_shot.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG="$ROOT/training_one_shot_$(date -u +%Y%m%d_%H%M%SZ).log"
ln -sf "$LOG" "$ROOT/training_one_shot_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Full one-shot log: $LOG (also: training_one_shot_latest.log)"

DT_BEST="$ROOT/runs/detect/yolo26l_dancetrack_only/weights/best.pt"
MODELS_DT="$ROOT/models/yolo26l_dancetrack.pt"
if [ -n "${YOLO_CROWDHUMAN_PARENT_WEIGHTS:-}" ]; then
  PARENT="$YOLO_CROWDHUMAN_PARENT_WEIGHTS"
elif [ -f "$DT_BEST" ]; then
  PARENT="$DT_BEST"
elif [ -f "$MODELS_DT" ]; then
  PARENT="$MODELS_DT"
else
  PARENT="$DT_BEST"
fi

echo "==> [1/3] Lambda venv + PyTorch CUDA + ultralytics (setup_lambda_training.sh)"
bash "$ROOT/scripts/phase2_public_training/setup_lambda_training.sh"

echo ""
echo "==> [2/3] DanceTrack parent checkpoint for CrowdHuman fine-tune (your saved model — skip DanceTrack training)"
if [ ! -f "$PARENT" ]; then
  echo ""
  echo "ERROR: No DanceTrack-trained weights found. Upload your existing best.pt to one of:" >&2
  echo "  $DT_BEST" >&2
  echo "  $MODELS_DT" >&2
  echo "Or: export YOLO_CROWDHUMAN_PARENT_WEIGHTS=/path/to/your.pt" >&2
  exit 1
fi
echo "    OK: $PARENT"
export YOLO_CROWDHUMAN_PARENT_WEIGHTS="$PARENT"

echo ""
echo "==> [3/3] Fetch CrowdHuman → convert → train (YOLO_LAMBDA_PIPELINE=crowdhuman)"
bash "$ROOT/scripts/phase2_public_training/run_lambda_train_crowdhuman.sh"

echo ""
echo "==> Done. Deploy weights:"
echo "    $ROOT/runs/detect/yolo26l_crowdhuman_ft/weights/best.pt"
echo "    → copy to your Mac as models/yolo26l_dancetrack.pt"
