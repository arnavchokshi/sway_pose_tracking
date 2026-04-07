#!/usr/bin/env bash
# Full pipeline: DanceTrack → CrowdHuman fine-tune (deploy yolo26l_crowdhuman_ft best.pt as yolo26l_dancetrack.pt).
export YOLO_LAMBDA_PIPELINE=dancetrack_crowdhuman
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_lambda_yolo_train.sh"
