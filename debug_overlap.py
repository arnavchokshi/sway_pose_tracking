#!/usr/bin/env python3
"""
Debug: Trace tracks 9 and 61 frame-by-frame - positions, overlap, and late entrant.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
data_path = SCRIPT_DIR / "output" / "data.json"
if not data_path.exists():
    print("Run pipeline first", file=sys.stderr)
    sys.exit(1)

with open(data_path) as f:
    data = json.load(f)

def box_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def box_iou(b1, b2):
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2-xi1) * (yi2-yi1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0

frames = data.get("frames", [])
# Focus on frames 310-400 (around when 61 appears and late entrant enters)
print("Frame | Tracks present | 9 center | 61 center | 65 center | IoU(9,61) | Notes")
print("-" * 90)
for fr in frames:
    fidx = fr.get("frame_idx", -1)
    if fidx < 305 or fidx > 410:
        continue
    tracks = fr.get("tracks", {})
    tids = list(tracks.keys())
    t9 = tracks.get("9", tracks.get(9, {}))
    t61 = tracks.get("61", tracks.get(61, {}))
    t65 = tracks.get("65", tracks.get(65, {}))
    box9 = t9.get("box") if t9 else None
    box61 = t61.get("box") if t61 else None
    box65 = t65.get("box") if t65 else None
    c9 = box_center(box9) if box9 else "-"
    c61 = box_center(box61) if box61 else "-"
    c65 = box_center(box65) if box65 else "-"
    iou = box_iou(box9, box61) if box9 and box61 else 0
    note = ""
    if iou > 0.3:
        note = "OVERLAP!"
    if fidx == 319:
        note += " 61 first frame"
    if fidx == 334:
        note += " 65 would start"
    print(f"{fidx:5} | {sorted(tids)} | {str(c9):12} | {str(c61):12} | {str(c65):12} | {iou:.3f} | {note}")

# Summary
print("\n=== Track summary ===")
all_tids = set()
for fr in frames:
    all_tids.update(fr.get("tracks", {}))
print("All track IDs in output:", sorted(all_tids))
tid_spans = {}
for fr in frames:
    for tid in fr.get("tracks", {}):
        if tid not in tid_spans:
            tid_spans[tid] = []
        tid_spans[tid].append(fr.get("frame_idx", 0))
for tid in sorted(tid_spans.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
    spans = tid_spans[tid]
    print(f"  {tid}: frames {min(spans)}-{max(spans)} ({len(spans)} frames)")
