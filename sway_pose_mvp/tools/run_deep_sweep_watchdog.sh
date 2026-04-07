#!/usr/bin/env bash
set -euo pipefail

# Watchdog for deep ID sweep:
# - checks every 30 minutes by default
# - if sweep process is not running, restarts it
# - writes heartbeat + status logs

ROOT_DIR="${1:-/Users/arnavchokshi/Desktop/sway_test/sway_pose_mvp}"
OUTPUT_ROOT="${2:-$ROOT_DIR/eval_results/deep_id_sweep}"
PHASES="${3:-A B C}"
INTERVAL_SECONDS="${4:-1800}"

LOG_DIR="$OUTPUT_ROOT/watchdog"
STATUS_FILE="$LOG_DIR/status.txt"
HEARTBEAT_FILE="$LOG_DIR/heartbeat.txt"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
SWEEP_LOG="$LOG_DIR/sweep.log"

mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Watchdog started" | tee -a "$WATCHDOG_LOG"
echo "ROOT_DIR=$ROOT_DIR" | tee -a "$WATCHDOG_LOG"
echo "OUTPUT_ROOT=$OUTPUT_ROOT" | tee -a "$WATCHDOG_LOG"
echo "PHASES=$PHASES" | tee -a "$WATCHDOG_LOG"
echo "INTERVAL_SECONDS=$INTERVAL_SECONDS" | tee -a "$WATCHDOG_LOG"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$ts heartbeat" > "$HEARTBEAT_FILE"

  if pgrep -f "tools/sweep_ground_truth.py --output-root $OUTPUT_ROOT --phases $PHASES" >/dev/null 2>&1; then
    echo "[$ts] sweep_running=1 action=none" | tee -a "$WATCHDOG_LOG"
    echo "$ts RUNNING" > "$STATUS_FILE"
  else
    echo "[$ts] sweep_running=0 action=start" | tee -a "$WATCHDOG_LOG"
    echo "$ts RESTARTING" > "$STATUS_FILE"
    nohup python3 tools/sweep_ground_truth.py --output-root "$OUTPUT_ROOT" --phases $PHASES >> "$SWEEP_LOG" 2>&1 &
    sleep 2
    if pgrep -f "tools/sweep_ground_truth.py --output-root $OUTPUT_ROOT --phases $PHASES" >/dev/null 2>&1; then
      echo "[$ts] restart_ok=1" | tee -a "$WATCHDOG_LOG"
      echo "$ts RUNNING_AFTER_RESTART" > "$STATUS_FILE"
    else
      echo "[$ts] restart_ok=0" | tee -a "$WATCHDOG_LOG"
      echo "$ts RESTART_FAILED" > "$STATUS_FILE"
    fi
  fi

  sleep "$INTERVAL_SECONDS"
done

