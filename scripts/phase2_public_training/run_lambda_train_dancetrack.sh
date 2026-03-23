#!/usr/bin/env bash
# Run ON Lambda after setup_lambda_training.sh. Downloads DanceTrack from Hugging Face,
# converts to YOLO format, trains --phase dancetrack. No upload of dataset from your Mac.
#
# Full terminal output is copied to training_full_<UTC>.log in the repo root (scp it home if useful).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG="$ROOT/training_full_$(date -u +%Y%m%d_%H%M%SZ).log"
echo "==> Repo root: $ROOT"
echo "==> Logging everything to: $LOG"
echo "==> Tip: from another SSH session, run: tail -f $LOG"

{
  echo "========== START $(date -u) =========="
  echo "==> Fetch DanceTrack (HF)"
  python3 scripts/phase2_public_training/fetch_dancetrack_hf.py
  echo "==> Convert to YOLO"
  python3 scripts/phase2_public_training/convert_dancetrack_to_yolo.py
  echo "==> Train (unbuffered stdout — live progress in this log)"
  # -u: line-buffer-style prints so tee/tail -f see epochs as they run
  python3 -u scripts/phase2_public_training/train_yolo11x.py --phase dancetrack
  echo "========== END $(date -u) =========="
} 2>&1 | tee "$LOG"

BEST="$ROOT/runs/detect/yolo11x_dancetrack_only/weights/best.pt"
echo ""
echo "==> Training finished."
echo "    Weights: $BEST"
echo "    Full session log: $LOG"
echo "    Per-epoch CSV: $ROOT/runs/detect/yolo11x_dancetrack_only/results.csv"
echo "    Copy to your Mac (from your laptop terminal, not Lambda):"
echo "    scp ubuntu@<LAMBDA_IP>:$BEST ./yolo11x_dancetrack_only_best.pt"
echo "    scp ubuntu@<LAMBDA_IP>:$LOG ./lambda_training.log"
echo "Then: mkdir -p models && mv yolo11x_dancetrack_only_best.pt models/yolo11x_dancetrack.pt"
echo "      export SWAY_YOLO_WEIGHTS=models/yolo11x_dancetrack.pt"
