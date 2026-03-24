#!/usr/bin/env bash
# Run ON Lambda after setup_lambda_training.sh. Uses repo .venv Python only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY missing. Run: bash scripts/phase2_public_training/setup_lambda_training.sh" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

# Line-buffer tee so `tail` on the log shows progress while downloads/train run (glibc stdbuf).
if command -v stdbuf >/dev/null 2>&1; then
  _TEE=(stdbuf -oL -eL tee)
else
  _TEE=(tee)
fi

LOG="$ROOT/training_full_$(date -u +%Y%m%d_%H%M%SZ).log"
ln -sf "$LOG" "$ROOT/training_full_latest.log"
echo "==> Repo root: $ROOT"
echo "==> Python: $PY"
echo "==> Logging everything to: $LOG"
echo "==> Tip: from another SSH session, run: tail -f $LOG"
echo "==> Disk (repo volume):"
df -h "$ROOT" || true
echo "==> GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

{
  echo "========== START $(date -u) =========="
  echo "==> Fetch DanceTrack (HF)"
  "$PY" -u scripts/phase2_public_training/fetch_dancetrack_hf.py
  echo "==> Convert to YOLO"
  "$PY" -u scripts/phase2_public_training/convert_dancetrack_to_yolo.py
  echo "==> Train (unbuffered stdout)"
  "$PY" -u scripts/phase2_public_training/train_yolo26l.py --phase dancetrack
  echo "========== END $(date -u) =========="
} 2>&1 | "${_TEE[@]}" "$LOG"

pipe=("${PIPESTATUS[@]}")
if [[ "${pipe[0]}" -ne 0 ]]; then
  echo "========== FAILED (pipeline exit ${pipe[0]}) $(date -u) ==========" | tee -a "$LOG"
  exit "${pipe[0]}"
fi

BEST="$ROOT/runs/detect/yolo26l_dancetrack_only/weights/best.pt"
echo ""
echo "==> Training finished OK."
echo "    Weights: $BEST"
echo "    Full session log: $LOG"
echo "    Per-epoch CSV: $ROOT/runs/detect/yolo26l_dancetrack_only/results.csv"
echo "    Copy to your Mac (from your laptop terminal, not Lambda):"
echo "    scp ubuntu@132.145.211.165:$BEST ./yolo26l_dancetrack_only_best.pt"
echo "    scp ubuntu@132.145.211.165:$LOG ./lambda_training.log"
echo "Then: mkdir -p models && mv yolo26l_dancetrack_only_best.pt models/yolo26l_dancetrack.pt"
echo "      export SWAY_YOLO_WEIGHTS=models/yolo26l_dancetrack.pt"
