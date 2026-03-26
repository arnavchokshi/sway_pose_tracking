#!/usr/bin/env bash
# Backward-compatible entry: DanceTrack only (same as YOLO_LAMBDA_PIPELINE=dancetrack).
export YOLO_LAMBDA_PIPELINE=dancetrack
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_lambda_yolo_train.sh"
