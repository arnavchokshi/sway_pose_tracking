# Sway Pose Tracking V3.2 — Pipeline Guide

A production-ready, M2-optimized pipeline designed to handle severe occlusions, complex choreography, and long video lengths without crashing.

---

## Overview (V3.2)

The pipeline processes video in a **linear, streaming** flow to fix the Phase 2a temporal paradox and prevent RAM bloat. It consists of **9 phases**:

| Phase | Name | Purpose |
|-------|------|---------|
| 1 | Streaming Ingestion & Detection | 300-frame chunks, native FPS, YOLO11l stride=2 (even frames only) |
| 2 | High-Tenacity Tracking & Box Stitching | BoT-SORT + manual min_hits, Wave 1 bbox stitch, Coexistence Dedup (10 frame NMS) |
| 3 | Resolution-Aware Box Pruning | 20% possible lifespan, 3% kinetic (V3.2), min bbox ≥30% global median |
| 4 | High-Fidelity Pose Estimation | ViTPose-Large, fp16 MPS, batched [N,3,256,192] |
| 4b | Completeness Audit (V3.2) | Spatial corner + Lifetime Peak: corner=max<0.35; else max<0.25, mean_shoulder>0.40 |
| 5 | Skeleton Re-Association (OKS Stitch) | Wave 2 pose stitch (OKS>0.5), crossover matching |
| 6 | Smart Mirror Pruning | Edge + inverted velocity + low lower-body conf |
| 7 | Temporal Smoothing | 1 Euro Filter with conf&lt;0.3 guard |
| 8 | Spatio-Temporal Scoring | circmean/circmedian, cDTW, per-joint thresholds |
| 9 | Export & Visualization | Per-joint heatmap (Green/Blue/Yellow/Red/Gray) |

---

## Phase 1: Streaming Ingestion & Detection

**Module:** `tracker.py`

### Goal

Fix missing dancers and prevent RAM crashes.

### What Happens

1. **Chunking:** Video read in 300-frame chunks (10 seconds at 30 FPS). Full video never held in RAM.
2. **Native FPS (30+):** Process at native frame rate. No decimation — 15 FPS would alias out fast hand strikes.
3. **YOLO11l (Large):** Detection at 640×640. **V3.0 YOLO stride=2:** YOLO runs only on *even* frames (0, 2, 4…); odd frames use BoT-SORT Kalman propagation with empty detections (2× inference speedup). Odd-frame boxes filled via interpolation.
4. **Lowered Confidence:** `conf=0.25`. Better to detect a false positive (pruned later) than miss a real dancer.
5. **Resolution splitting:** Frames resized to 640×640 for detection; boxes scaled back to original resolution.

### Output

```python
raw_tracks: Dict[track_id, List[(frame_idx, (x1,y1,x2,y2), confidence), ...]]
total_frames: int
output_fps: float  # native FPS
frames_list: List[(frame_idx, frame_bgr)]
```

---

## Phase 2: High-Tenacity Tracking & Box Stitching

**Module:** `tracker.py`, `botsort.yaml`, `ocsort.yaml`

### Goal

Stop IDs from dropping and respawning.

### What Happens

1. **BoT-SORT:** `track_buffer=90`, **`new_track_thresh=0.35`** (V3.1: lower to pick up side-dancers and late-entrants). **V3.2 Manual min_hits:** Post-tracking filter removes tracks with < 3 genuine YOLO detections (conf > 0.0) — BoT-SORT yaml often ignores min_hits.
2. **Wave 1 Stitching (BBox):** When ID_A dies and ID_B spawns within 90 frames, check pixel distance. If ID_B spawns near ID_A's last known location (or predicted trajectory), merge them. Interpolate missing boxes.
3. **Coexistence Deduplication (Inter-Track NMS):** If two tracks overlap IoU > 0.65 for **10+ consecutive frames** (V3.2: was 5), merge them. Keep older/longer track, kill younger (Kalman ghost). Prevents merging real dancers in tight crossing formations.

### Output

Same `raw_tracks` with stitched and deduplicated IDs; no fragment gaps within 90 frames; ghost duplicates merged.

---

## Phase 3: Resolution-Aware Box Pruning

**Module:** `track_pruning.py`

### Goal

Remove non-dancers (audience, chairs) without breaking at different video resolutions.

### Rules

1. **Duration (V3.2):** Must appear in &gt; **20%** of its *possible* lifespan (`possible_frames = total_frames - track_first_frame`), with hard floor of **30 frames** (V3.2: was 60). Valid late-entrants in last 1–2 seconds no longer pruned.
2. **Normalized Kinetic Pruning (V3.2):** Bounding box center must move by a standard deviation greater than **3%** of the median dancer's bounding box height (`KINETIC_STD_FRAC=0.03`). Keeps side dancers who move less.
3. **Min Bbox Height (V3.2):** Track median bbox height must be ≥ **30%** of global median (`MIN_BBOX_HEIGHT_FRAC=0.30`). Keeps side/far dancers.

### Algorithm

- Compute `median_dancer_bbox_height` across tracks that pass duration filter.
- For each track: `movement_std = sqrt(std_x² + std_y²)`; `track_median_height = median(bbox heights)`.
- Prune if `movement_std < 0.03 * median_dancer_bbox_height` OR `track_median_height < 0.30 * median_dancer_bbox_height`.

### Output

`Set[int]` of surviving track IDs.

---

## Phase 4: High-Fidelity Pose Estimation

**Module:** `pose_estimator.py`

### Goal

Get accurate enough skeletons to judge 15° wrist/elbow mistakes.

### What Happens

1. **Engine:** ViTPose-Large (`usyd-community/vitpose-plus-large`). fp16 on M2 MPS backend.
2. **Batching:** Stack cropped boxes into `[N, 3, 256, 192]` tensor for parallel processing.
3. **Bbox padding (15%):** Expand YOLO boxes before cropping to avoid cutting off limbs.
4. Keypoints converted back to global image coordinates.

### Keypoints (17 COCO)

- Face: nose, eyes, ears
- Upper body: shoulders, elbows, wrists
- Lower body: hips, knees, ankles

### Output

```python
{track_id: {"keypoints": (17, 3), "scores": (17,)}}
```

---

## Phase 5: Skeleton Re-Association (OKS Stitch)

**Module:** `crossover.py`

### Goal

Catch complex ID swaps that Phase 2 missed.

### What Happens

1. **Wave 2 Stitching (Pose):** For dead/respawned tracks Phase 2 couldn't merge: compute Object Keypoint Similarity (OKS) between dead track's last skeleton and newborn's first skeleton. If **OKS &gt; 0.5**, merge them.
2. **Crossover Matching:** During dense overlaps (IoU &gt; 0.6), temporarily suspend Box tracking and use OKS so IDs stay locked to correct bodies.
3. **Occlusion Fallback:** Total keypoint confidence &lt; 0.3 → CVM ghost the box.

---

## Phase 6: Smart Mirror + Completeness Audit

**Module:** `track_pruning.py`

### Goal

Delete reflections and decapitated head/shoulder tracks without deleting legitimate edge dancers.

### Smart Mirror Rule

Prune IF AND ONLY IF all three hold:

1. Track exists in **outer 10%** of the frame (left or right edge).
2. Its x-velocity is **inverted** relative to the group.
3. Lower-body keypoints (ankles, knees) have average confidence **&lt; 0.3** (studio mirrors cut off at floor).

### Completeness Audit (V3.2 — Spatial-Aware Corner Observers)

Runs **after Phase 4 (ViTPose) but before Phase 5** to save OKS computation. Prunes **seated corner observers** (head/shoulders only) without pruning real dancers.

- **Corner tracks** (bbox center in outer 15% x AND outer 20% y): Prune if `max(knee,ankle) < 0.35` AND `mean_shoulder > 0.35`. Stricter to catch partial persons whose ankles ViTPose may hallucinate.
- **Non-corner tracks:** Prune if `max(knee,ankle) < 0.25` AND `mean_shoulder > 0.40` (unchanged; safe for floorwork/skirts).

### Output

`Set[int]` of track IDs to remove.

---

## Phase 7: Temporal Smoothing

**Module:** `smoother.py`

### What Happens

**1 Euro Filter** applied to all `(x, y)` keypoints.

- Parameters: `min_cutoff=1.0`, `beta=0.7`, `d_cutoff=1.0`
- **Guard:** Suspended if keypoint confidence &lt; 0.3 to prevent smoothing hallucinated geometry.

### Output

Same pose structure with smoothed `(x, y)`; confidence unchanged.

---

## Phase 8: Spatio-Temporal Scoring

**Module:** `scoring.py`

### Goal

Mathematically sound scoring that accounts for human anatomy and musical tempo.

### What Happens

1. **Circular Math:** Group Truth median uses `scipy.stats.circmean` / `circmedian` with high/low boundary so 355° and 5° average correctly (not 180°).
2. **Ripple Hatch:** If group std &gt; 30° (intentional variation), deviations set to **NaN** (renders Gray), not 0 (Green).
3. **cDTW:** Constrained DTW with Sakoe-Chiba band of 3 (faster and more accurate than FastDTW for 30-frame window).
4. **Per-Joint Thresholds:**
   - **Spine/Hips:** Green &lt; 10°, Red &gt; 20°
   - **Elbows/Knees:** Green &lt; 20°, Red &gt; 35°
5. **30-frame window** per dancer/joint vs Group Truth.
6. **Timing Error:** &gt; 2 frames lag/lead = Off-Beat.

---

## Phase 9: Export & Visualization

**Module:** `visualizer.py`

### Outputs

1. **JSON** (`output/data.json`): Keypoints, boxes, joint_angles, deviations, shape_errors, timing_errors.
2. **MP4** (`output/<video_name>_poses.mp4`): Rendered at native FPS with **V3.0 heatmap**:

| Condition | Color |
|-----------|-------|
| Shape within tolerance AND Timing ≤ 2 frames | **Green** |
| Shape good, Timing &gt; 2 frames | **Blue** (Off-beat) |
| Minor Shape Error | **Yellow** |
| Major Shape Error | **Red** |
| Missing, Occluded, or Ripple | **Gray** |

Per-joint thresholds: Spine/Hips 10°/20°, Elbows/Knees 20°/35°.

---

## Data Flow Summary (V3.0)

```
Video (MP4)
    │
    ▼
[1] Streaming 300-frame chunks, native FPS → YOLO11l (stride=2, even frames only; odd frames Kalman + interpolate)
    │
    ▼
[2] BoT-SORT (track_buffer=90, new_track_thresh=0.50) + Wave 1 box stitch + coalescence_deduplicate → raw_tracks
    │
    ▼
[3] prune_tracks (20% duration, 3% kinetic, min bbox 30% global median) → surviving_ids → raw_tracks_to_per_frame
    │
    ▼
[4] ViTPose-Large (fp16 MPS) → raw_poses_by_frame
    │
    ▼
[4b] prune_completeness_audit (Lifetime Peak Lower-Body: seated observers, floorwork-safe)
    │
    ▼
[5] Wave 2: apply_occlusion_reid (OKS>0.5) + apply_crossover_refinement
    │
    ▼
[6] prune_smart_mirrors (edge+velocity+low lower-body)
    │
    ▼
[7] PoseSmoother (1 Euro, conf<0.3 guard) → smoothed poses
    │
    ▼
[8] process_all_frames_scoring_vectorized (circmean, cDTW, per-joint thresholds)
    │
    ▼
[9] render_and_export → data.json + <video>_poses.mp4 (V3.0 heatmap)
```

---

## Key Tunable Parameters (V3.0)

| Location | Parameter | Default | Effect |
|----------|-----------|---------|--------|
| `tracker.py` | `CHUNK_SIZE` | 300 | Frames per streaming chunk |
| `tracker.py` | `YOLO_CONF` | 0.25 | Detection confidence threshold |
| `tracker.py` | `YOLO_DETECTION_STRIDE` | 2 | Run YOLO only on even frames; odd frames use Kalman |
| `tracker.py` | `DEDUP_IOU_THRESH` | 0.65 | IoU threshold for coexistence deduplication |
| `tracker.py` | `DEDUP_CONSECUTIVE_FRAMES` | 10 | Consecutive frames overlapping to merge ghost tracks (V3.2) |
| `tracker.py` | `STITCH_MAX_FRAME_GAP` | 90 | Max frames for Wave 1 box stitch |
| `tracker.py` | `STITCH_MAX_PIXEL_RADIUS` | 120.0 | Max pixel distance (stationary) for stitch |
| `tracker.py` | `STITCH_PREDICTED_RADIUS` | 180.0 | Max pixel distance (velocity-predicted) |
| `track_pruning.py` | `min_duration_ratio` | 0.20 | Min fraction of *possible* frames track must appear (V3.1: late-entrant friendly) |
| `track_pruning.py` | `KINETIC_STD_FRAC` | 0.03 | Kinetic threshold = 3% of median bbox height (V3.2: side dancers) |
| `track_pruning.py` | `MIN_BBOX_HEIGHT_FRAC` | 0.30 | Min track median bbox height as fraction of global median (V3.2) |
| `track_pruning.py` | `EDGE_MARGIN_FRAC` | 0.10 | Outer 10% of frame for mirror rule |
| `track_pruning.py` | `min_lower_body_conf` | 0.3 | Lower-body conf threshold for mirror |
| `botsort.yaml` | `new_track_thresh` | 0.35 | Min conf to spawn new track ID (V3.1: pick up side-dancers) |
| `botsort.yaml` | `min_hits` | 3 | Track must be detected 3 consecutive frames to become confirmed |
| `pose_estimator.py` | `model_name` | vitpose-large | ViTPose model |
| `pose_estimator.py` | `BBOX_PADDING` | 0.15 | Expand bbox by 15% before cropping |
| `crossover.py` | `REID_MAX_FRAME_GAP` | 90 | Max frames for Wave 2 OKS stitch |
| `crossover.py` | `REID_MIN_OKS` | 0.5 | Min OKS to merge tracks |
| `smoother.py` | `SMOOTH_CONF_THRESHOLD` | 0.3 | Skip smoothing below this |
| `smoother.py` | `min_cutoff`, `beta` | 1.0, 0.7 | 1 Euro filter params |
| `scoring.py` | `RIPPLE_STD_THRESHOLD` | 30.0 | Std (deg) above which deviations=NaN |
| `scoring.py` | `DTW_SAKOE_CHIBA_BAND` | 3 | cDTW constraint band |
| `scoring.py` | `CONSENSUS_ROLLING_WINDOW` | 5 | Rolling median on Group Truth |
| `scoring.py` | `DTW_WINDOW_SIZE` | 30 | Frames for cDTW |
| `visualizer.py` | `SHAPE_GREEN_SPINE`, `SHAPE_RED_SPINE` | 10, 20 | Spine/Hips thresholds (deg) |
| `visualizer.py` | `SHAPE_GREEN_LIMB`, `SHAPE_RED_LIMB` | 20, 35 | Elbow/Knee thresholds (deg) |
| `visualizer.py` | `TIMING_OFF_BEAT_THRESHOLD` | 2 | Frames: >2 = Blue |
| `botsort.yaml` / `ocsort.yaml` | `track_buffer` | 90 | Frames to retain lost tracks |

---

## Hardware

Optimized for **Apple Silicon (M2)** using PyTorch MPS backend. fp16 inference for ViTPose-Large. Falls back to CPU if MPS is unavailable.
