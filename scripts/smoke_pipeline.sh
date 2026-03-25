#!/usr/bin/env bash
# Quick sanity check: unit tests + end-to-end main.py on a tiny synthetic clip.
# From repo root: ./scripts/smoke_pipeline.sh
# Requires: pip install -r requirements-dev.txt (or pip install pytest)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m pytest tests/ -q --tb=short

mkdir -p output/_pipeline_smoke
python3 <<'PY'
import cv2
import numpy as np
from pathlib import Path

out = Path("output/_pipeline_smoke/tiny.mp4")
w, h, fps, n = 640, 480, 30, 45
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
if not vw.isOpened():
    raise SystemExit("VideoWriter failed")
for i in range(n):
    f = np.full((h, w, 3), 40, dtype=np.uint8)
    x = 100 + (i * 8) % 200
    cv2.rectangle(f, (x, 180), (x + 80, 420), (200, 200, 200), -1)
    vw.write(f)
vw.release()
print("Smoke clip:", out.resolve())
PY

export SWAY_UNLOCK_HYBRID_SAM_TUNING=1
export SWAY_HYBRID_SAM_OVERLAP=0
export SWAY_YOLO_DETECTION_STRIDE=2
python3 main.py output/_pipeline_smoke/tiny.mp4 \
  --output-dir output/_pipeline_smoke/out \
  --pose-model base \
  --pose-stride 2

echo "OK — outputs under output/_pipeline_smoke/out/"
