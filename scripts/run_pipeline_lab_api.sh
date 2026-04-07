#!/usr/bin/env bash
# Start Pipeline Lab FastAPI from repo root (sway_pose_mvp/) so `import pipeline_lab` works.
# Usage (from anywhere):  bash path/to/sway_pose_mvp/scripts/run_pipeline_lab_api.sh --reload --host localhost --port 8765
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uvicorn pipeline_lab.server.app:app "$@"
