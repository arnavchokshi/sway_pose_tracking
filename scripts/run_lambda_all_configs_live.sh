#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VIDEO_PATH="${1:-/Users/arnavchokshi/Desktop/newTest.mov}"
STOP_AFTER="${STOP_AFTER_BOUNDARY:-after_phase_3}"
STAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="output/all_config_full_${STAMP}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video not found: $VIDEO_PATH" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"

echo "Starting full all-config validation"
echo "Video: $VIDEO_PATH"
echo "Output: $RUN_DIR"
echo "Stop boundary: $STOP_AFTER"

python -m tools.run_all_configurations \
  --video "$VIDEO_PATH" \
  --execute \
  --strict-coverage \
  --fail-on-unwired-extras \
  --stop-after-boundary "$STOP_AFTER" \
  --plan-out "$RUN_DIR/plan.json" \
  --results-out "$RUN_DIR/results.jsonl" \
  --failures-out "$RUN_DIR/failures.jsonl" \
  --summary-out "$RUN_DIR/summary.json" \
  --output-root "$RUN_DIR/runs" \
  | tee "$RUN_DIR/live_master.log"

echo
echo "Done. Check:"
echo "  $RUN_DIR/live_master.log"
echo "  $RUN_DIR/failures.jsonl"
echo "  $RUN_DIR/summary.json"
