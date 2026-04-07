#!/usr/bin/env bash
# CrowdHuman fine-tune ONLY — does NOT fetch/train DanceTrack. Use when you already
# have a DanceTrack-trained `best.pt` (or `models/yolo26l_dancetrack.pt` from the app).
#
# Upload first (pick one):
#   runs/detect/yolo26l_dancetrack_only/weights/best.pt
#   models/yolo26l_dancetrack.pt
# or: export YOLO_CROWDHUMAN_PARENT_WEIGHTS=/path/to/your.pt
#
# Then from repo root:
#   bash scripts/phase2_public_training/run_lambda_crowdhuman_skip_dancetrack.sh
#
# Same as: YOLO_LAMBDA_PIPELINE=crowdhuman bash scripts/phase2_public_training/run_lambda_yolo_train.sh
export YOLO_LAMBDA_PIPELINE=crowdhuman
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_lambda_yolo_train.sh"
