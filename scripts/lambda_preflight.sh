#!/usr/bin/env bash
# Run ON the Lambda host (or locally) from repo root: bash scripts/lambda_preflight.sh
# Optional: PIPELINE=1 for main.py smoke (default 90s timeout).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

echo "==> Repo: $ROOT"

if ! command -v nvidia-smi &>/dev/null; then
  echo "WARN: nvidia-smi not found (OK for local CPU-only preflight)."
else
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

echo "==> pip install -r requirements.txt"
"$PY" -m pip install -q -r "$ROOT/requirements.txt"

echo "==> PyTorch + CUDA"
if command -v nvidia-smi &>/dev/null; then
  "$PY" <<'PY'
import sys
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: GPU host but torch.cuda.is_available() is False — install CUDA wheels (see scripts/phase2_public_training/setup_lambda_training.sh).", file=sys.stderr)
    sys.exit(3)
print("device", torch.cuda.get_device_name(0))
PY
else
  "$PY" <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
fi

echo "==> Model files (Phase 1–3 sweep)"
for f in \
  models/yolo26l.pt \
  models/yolo26l_dancetrack.pt \
  models/yolo26l_dancetrack_crowdhuman.pt \
  models/AFLink_epoch20.pth \
  models/sam2.1_b.pt
do
  if [[ -f "$ROOT/$f" ]]; then
    echo "  OK $f"
  else
    echo "  MISSING $f" >&2
    exit 2
  fi
done

CFG="$ROOT/data/ground_truth/sweep_sequences.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "Missing $CFG — copy sweep_sequences.example.yaml and set paths." >&2
  exit 2
fi
echo "==> Validating sweep_sequences.yaml paths"
"$PY" <<PY
import sys
from pathlib import Path
import yaml
root = Path("$ROOT")
cfg = root / "data/ground_truth/sweep_sequences.yaml"
data = yaml.safe_load(cfg.read_text())
order = data.get("sequence_order") or []
seqs = data.get("sequences") or {}
for name in order:
    if name not in seqs:
        print(f"sequence_order missing key: {name}", file=sys.stderr)
        sys.exit(2)
    spec = seqs[name]
    for key in ("video", "gt_mot"):
        p = root / spec[key]
        if not p.is_file():
            print(f"Missing {key} for {name}: {p}", file=sys.stderr)
            sys.exit(2)
    print(f"  OK sequence {name}")
print("  all paths resolve")
PY

export SWAY_SERVER_PERF=1
echo "==> tools.smoke_server_perf_env"
"$PY" -m tools.smoke_server_perf_env

if [[ "${PIPELINE:-0}" == "1" ]]; then
  echo "==> tools.smoke_server_perf_env --pipeline (timeout ${PIPELINE_TIMEOUT:-90}s)"
  "$PY" -m tools.smoke_server_perf_env --pipeline --timeout "${PIPELINE_TIMEOUT:-90}"
fi

echo ""
echo "==> Preflight OK. Next on this host:"
echo "    export SWAY_SERVER_PERF=1"
echo "    python -m tools.auto_sweep --config data/ground_truth/sweep_sequences.yaml"
echo "    # tmux recommended; stop: touch output/sweeps/optuna/STOP  or  Ctrl+C"
echo ""
echo "==> Lambda gpu_1x_a10 (~24 GB VRAM, x86_64): use cu124 venv (setup_lambda_training.sh)."
echo "    Default YOLO infer batch=1; only raise SWAY_YOLO_INFER_BATCH after a short sweep/OOM check."
