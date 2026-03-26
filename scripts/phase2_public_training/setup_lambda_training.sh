#!/usr/bin/env bash
# Run ON the Lambda GPU instance once, from repo root (sway_pose_tracking/).
# Creates .venv/ with PyTorch (CUDA) + training deps — avoids clashes with apt
# torch/numpy/scipy on the Lambda image (common "numpy.dtype size changed" crash).
#
# aarch64 (GH200): cu128 index. x86_64 (A10, etc.): cu124 index.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "==> Repo root: $ROOT"

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Use a GPU instance image on Lambda." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ARCH="$(uname -m)"
echo "==> CPU architecture: $ARCH"

# Minimal cloud images sometimes ship without ensurepip / venv.
if [ ! -x "$PY" ]; then
  if ! python3 -m venv /tmp/.sway_check_venv 2>/dev/null; then
    echo "==> Installing python3-venv (required for python3 -m venv on this image)"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
  fi
  rm -rf /tmp/.sway_check_venv
  echo "==> Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

echo "==> Upgrading pip in venv"
"$PIP" install --upgrade pip

# x86_64: cu124. aarch64 (GH200): cu128.
PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
if [[ "$ARCH" == "aarch64" ]]; then
  PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
  echo "==> ARM64 — PyTorch cu128 index (GH200)."
else
  echo "==> x86_64 — PyTorch cu124 index."
fi

echo "==> Installing PyTorch + torchvision (CUDA) into venv"
"$PIP" install torch torchvision \
  --index-url "$PYTORCH_INDEX" \
  --extra-index-url https://pypi.org/simple

echo "==> Installing Ultralytics + helpers"
"$PIP" install -r "$ROOT/scripts/phase2_public_training/requirements-train-lambda.txt"

echo "==> Sanity check (venv only — not system Python)"
"$PY" -c "import torch; v=torch.__version__; cuda=torch.cuda.is_available(); print('torch', v, 'cuda_available=', cuda);
assert '+cpu' not in v, 'CPU-only torch in venv — wrong index?';
assert cuda, 'CUDA not available — wrong wheels or driver'"

"$PY" -c "from ultralytics import YOLO; import huggingface_hub; print('OK: ultralytics + huggingface_hub')"

echo "==> Done. Training uses: $PY"
echo "==> For inference / sweeps (main.py, tools.auto_sweep): export SWAY_SERVER_PERF=1"
echo "    Optional: SWAY_PERF_CPU_THREADS=16 on large instances; try SWAY_YOLO_INFER_BATCH=2–4 on 40GB if stable."
echo "==> Next (pick one):"
echo "    bash scripts/phase2_public_training/run_lambda_crowdhuman_skip_dancetrack.sh   # CrowdHuman only (upload DanceTrack .pt first)"
echo "    bash scripts/phase2_public_training/run_lambda_train_dancetrack.sh"
echo "    bash scripts/phase2_public_training/run_lambda_train_dancetrack_crowdhuman.sh   # train DT then CH on box"
echo "    # or: YOLO_LAMBDA_PIPELINE=crowdhuman bash .../run_lambda_yolo_train.sh"
