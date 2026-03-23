#!/usr/bin/env bash
# Run ON the Lambda GPU instance once, from repo root (sway_pose_mvp/).
# Installs PyTorch (CUDA) + training deps. Idempotent-ish: safe to re-run.
#
# IMPORTANT: Lambda GH200 (Grace Hopper) nodes are **ARM64 (aarch64)**. The usual
# x86_64 cu124 wheels will not install. This script picks the correct PyTorch index.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "==> Repo root: $ROOT"

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Use a GPU instance image on Lambda." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ARCH="$(uname -m)"
echo "==> CPU architecture: $ARCH"

echo "==> Upgrading pip"
python3 -m pip install --upgrade pip

# x86_64: cu124 is the usual Lambda / datacenter default.
# aarch64 (GH200, Grace Hopper): CUDA wheels live on cu128 (not on default PyPI — PyPI ARM = CPU-only).
PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
if [[ "$ARCH" == "aarch64" ]]; then
  PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
  echo "==> ARM64 detected — using PyTorch cu128 index (GH200 / Grace Hopper)."
else
  echo "==> Using PyTorch cu124 index (x86_64)."
fi

echo "==> Installing PyTorch + torchvision (CUDA). If this fails, see LAMBDA_TRAINING.md."
# extra-index-url: pull small deps from PyPI; torch* resolve from PyTorch index first
python3 -m pip install torch torchvision \
  --index-url "$PYTORCH_INDEX" \
  --extra-index-url https://pypi.org/simple

echo "==> Installing Ultralytics + dataset/training helpers"
python3 -m pip install -r "$ROOT/scripts/phase2_public_training/requirements-train-lambda.txt"

echo "==> Sanity check"
python3 -c "import torch; v=torch.__version__; cuda=torch.cuda.is_available(); print('torch', v, 'cuda_available=', cuda);
assert '+cpu' not in v, 'Got CPU-only torch on ARM64? See LAMBDA_TRAINING.md (GH200 / cu128).';
assert cuda, 'CUDA not available — wrong wheels or driver'"

python3 -c "from ultralytics import YOLO; import huggingface_hub; print('OK: ultralytics + huggingface_hub')"

echo "==> Done. Next: bash scripts/phase2_public_training/run_lambda_train_dancetrack.sh"
