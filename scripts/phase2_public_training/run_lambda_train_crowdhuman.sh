#!/usr/bin/env bash
# CrowdHuman stage only — requires runs/detect/yolo26l_dancetrack_only/weights/best.pt on disk.
export YOLO_LAMBDA_PIPELINE=crowdhuman
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_lambda_yolo_train.sh"
