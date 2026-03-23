#!/usr/bin/env bash
# BoT-SORT (SWAY_USE_BOXMOT=0) vs BoxMOT (default) full pose pipeline + side-by-side MP4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/output/pose_compare_botsort_vs_boxmot"
mkdir -p "$OUT"

run_one() {
  local video="$1"
  local stem="$2"
  local use_boxmot="$3"
  local odir="$OUT/${stem}_$([ "$use_boxmot" = 1 ] && echo boxmot || echo botsort)"
  if [[ -f "$odir/${stem}_poses.mp4" ]]; then
    echo "  skip (exists): $odir"
    return 0
  fi
  mkdir -p "$odir"
  if [[ "$use_boxmot" = 1 ]]; then
    env -u SWAY_USE_BOXMOT python main.py "$video" --output-dir "$odir"
  else
    SWAY_USE_BOXMOT=0 python main.py "$video" --output-dir "$odir"
  fi
}

hstack() {
  local stem="$1"
  local L="$OUT/${stem}_botsort/${stem}_poses.mp4"
  local R="$OUT/${stem}_boxmot/${stem}_poses.mp4"
  local D="$OUT/${stem}_side_by_side.mp4"
  if [[ ! -f "$L" || ! -f "$R" ]]; then
    echo "missing $L or $R"
    return 1
  fi
  # OpenCV labels (this ffmpeg build often lacks drawtext)
  python3 "$ROOT/scripts/label_side_by_side_compare.py" "$L" "$R" "$D"
}

# Desktop paths (adjust if needed)
run_one "/Users/arnavchokshi/Desktop/IMG_0256.mov" "IMG_0256" 0
run_one "/Users/arnavchokshi/Desktop/IMG_0256.mov" "IMG_0256" 1
hstack "IMG_0256"

run_one "/Users/arnavchokshi/Desktop/IMG_4732.mov" "IMG_4732" 0
run_one "/Users/arnavchokshi/Desktop/IMG_4732.mov" "IMG_4732" 1
hstack "IMG_4732"

run_one "/Users/arnavchokshi/Desktop/IMG_2946.MP4" "IMG_2946" 0
run_one "/Users/arnavchokshi/Desktop/IMG_2946.MP4" "IMG_2946" 1
hstack "IMG_2946"

echo "Done. Open: $OUT"
