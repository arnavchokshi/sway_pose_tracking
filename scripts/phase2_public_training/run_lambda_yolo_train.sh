#!/usr/bin/env bash
# Run ON Lambda after setup_lambda_training.sh. Uses repo .venv Python only.
#
# Choose pipeline with YOLO_LAMBDA_PIPELINE (default: dancetrack):
#   dancetrack              — DanceTrack fetch → YOLO convert → train phase dancetrack
#   crowdhuman              — CrowdHuman fetch → convert → train phase crowdhuman
#                             (needs a DanceTrack fine-tuned .pt — default path below, or set
#                             YOLO_CROWDHUMAN_PARENT_WEIGHTS=/path/to/your_dancetrack_best.pt)
#   dancetrack_crowdhuman   — both stages in one session (full yolo26l + CrowdHuman fine-tune)
#
# Examples:
#   bash scripts/phase2_public_training/run_lambda_yolo_train.sh
#   YOLO_LAMBDA_PIPELINE=crowdhuman bash scripts/phase2_public_training/run_lambda_yolo_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PIPELINE="${YOLO_LAMBDA_PIPELINE:-dancetrack}"
case "$PIPELINE" in
  dancetrack|crowdhuman|dancetrack_crowdhuman) ;;
  *)
    echo "ERROR: YOLO_LAMBDA_PIPELINE must be dancetrack, crowdhuman, or dancetrack_crowdhuman (got: $PIPELINE)" >&2
    exit 1
    ;;
esac

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY missing. Run: bash scripts/phase2_public_training/setup_lambda_training.sh" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

if command -v stdbuf >/dev/null 2>&1; then
  _TEE=(stdbuf -oL -eL tee)
else
  _TEE=(tee)
fi

LOG="$ROOT/training_full_$(date -u +%Y%m%d_%H%M%SZ).log"
ln -sf "$LOG" "$ROOT/training_full_latest.log"
echo "==> Repo root: $ROOT"
echo "==> Pipeline: $PIPELINE"
echo "==> Python: $PY"
echo "==> Logging everything to: $LOG"
echo "==> Tip: from another SSH session, run: tail -f $LOG"
echo "==> Crash / preemption: if train_yolo26l left weights/last.pt under runs/detect/<run>/, re-run ONLY training with:"
echo "    YOLO_TRAIN_RESUME=1 $PY -u scripts/phase2_public_training/train_yolo26l.py --phase <same> --resume"
echo "==> Disk (repo volume):"
df -h "$ROOT" || true
echo "==> GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

DT_BEST="$ROOT/runs/detect/yolo26l_dancetrack_only/weights/best.pt"
MODELS_DT="$ROOT/models/yolo26l_dancetrack.pt"
CH_BEST="$ROOT/runs/detect/yolo26l_crowdhuman_ft/weights/best.pt"
# CrowdHuman-only: no DanceTrack training — use your existing DanceTrack .pt (upload before train).
# Priority: YOLO_CROWDHUMAN_PARENT_WEIGHTS, else runs/.../best.pt, else models/yolo26l_dancetrack.pt (same as app).
if [ -n "${YOLO_CROWDHUMAN_PARENT_WEIGHTS:-}" ]; then
  CROWDHUMAN_PARENT="$YOLO_CROWDHUMAN_PARENT_WEIGHTS"
elif [ -f "$DT_BEST" ]; then
  CROWDHUMAN_PARENT="$DT_BEST"
elif [ -f "$MODELS_DT" ]; then
  CROWDHUMAN_PARENT="$MODELS_DT"
else
  CROWDHUMAN_PARENT="$DT_BEST"
fi

run_dancetrack() {
  echo "==> Fetch DanceTrack (HF)"
  "$PY" -u scripts/phase2_public_training/fetch_dancetrack_hf.py
  echo "==> Convert DanceTrack to YOLO"
  "$PY" -u scripts/phase2_public_training/convert_dancetrack_to_yolo.py
  echo "==> Train phase: dancetrack"
  "$PY" -u scripts/phase2_public_training/train_yolo26l.py --phase dancetrack
}

run_crowdhuman() {
  if [ ! -f "$CROWDHUMAN_PARENT" ]; then
    echo "ERROR: CrowdHuman fine-tune needs your DanceTrack-trained weights (skip DanceTrack re-train):" >&2
    echo "  scp your best.pt to one of:" >&2
    echo "    $DT_BEST" >&2
    echo "    $MODELS_DT   (same name as pipeline / Lab)" >&2
    echo "  Or:  export YOLO_CROWDHUMAN_PARENT_WEIGHTS=/path/to/your.pt" >&2
    echo "  Then: YOLO_LAMBDA_PIPELINE=crowdhuman bash scripts/phase2_public_training/run_lambda_yolo_train.sh" >&2
    exit 1
  fi
  echo "==> CrowdHuman parent weights: $CROWDHUMAN_PARENT"
  echo "==> Fetch CrowdHuman (HF)"
  "$PY" -u scripts/phase2_public_training/fetch_crowdhuman_hf.py
  echo "==> Convert CrowdHuman to YOLO"
  "$PY" -u scripts/phase2_public_training/convert_crowdhuman_to_yolo.py
  echo "==> Train phase: crowdhuman (init from DanceTrack checkpoint)"
  "$PY" -u scripts/phase2_public_training/train_yolo26l.py --phase crowdhuman --weights "$CROWDHUMAN_PARENT"
}

{
  echo "========== START $(date -u) =========="
  case "$PIPELINE" in
    dancetrack)
      run_dancetrack
      ;;
    crowdhuman)
      run_crowdhuman
      ;;
    dancetrack_crowdhuman)
      run_dancetrack
      run_crowdhuman
      ;;
  esac
  echo "========== END $(date -u) =========="
} 2>&1 | "${_TEE[@]}" "$LOG"

pipe=("${PIPESTATUS[@]}")
if [[ "${pipe[0]}" -ne 0 ]]; then
  echo "========== FAILED (pipeline exit ${pipe[0]}) $(date -u) ==========" | tee -a "$LOG"
  exit "${pipe[0]}"
fi

echo ""
echo "==> Training finished OK ($PIPELINE)."
echo "    Full session log: $LOG"

case "$PIPELINE" in
  dancetrack)
    echo "    Weights: $DT_BEST"
    echo "    Per-epoch CSV: $ROOT/runs/detect/yolo26l_dancetrack_only/results.csv"
    echo "    Copy to your Mac (example):"
    echo "    scp ubuntu@<IP>:$DT_BEST ./yolo26l_dancetrack_only_best.pt"
    echo "Then: mv yolo26l_dancetrack_only_best.pt models/yolo26l_dancetrack.pt"
    echo "      export SWAY_YOLO_WEIGHTS=models/yolo26l_dancetrack.pt"
    echo ""
    echo "    Optional — add CrowdHuman fine-tune on this instance:"
    echo "    YOLO_LAMBDA_PIPELINE=crowdhuman bash scripts/phase2_public_training/run_lambda_yolo_train.sh"
    ;;
  crowdhuman|dancetrack_crowdhuman)
    echo "    DanceTrack-stage checkpoint: $DT_BEST"
    echo "    CrowdHuman-stage weights (deploy this for DT+CH model): $CH_BEST"
    echo "    Per-epoch CSV: $ROOT/runs/detect/yolo26l_crowdhuman_ft/results.csv"
    echo "    Copy to your Mac (example):"
    echo "    scp ubuntu@<IP>:$CH_BEST ./yolo26l_crowdhuman_ft_best.pt"
    echo "Then: mv yolo26l_crowdhuman_ft_best.pt models/yolo26l_dancetrack.pt"
    echo "      export SWAY_YOLO_WEIGHTS=models/yolo26l_dancetrack.pt"
    ;;
esac
