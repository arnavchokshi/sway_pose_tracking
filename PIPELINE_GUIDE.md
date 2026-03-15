# Sway Pose Tracking V3.4 — Pipeline Guide

A production-ready, M2-optimized pipeline designed to handle severe occlusions, complex choreography, and long video lengths without crashing, while aggressively filtering non‑dancers and mirror ghosts.

---

## Overview (V3.4)

The pipeline processes video in a **linear, streaming** flow to avoid RAM bloat. It consists of **9 major phases**:

| Phase | Name | Purpose |
|-------|------|---------|
| 1 | Streaming Ingestion & Detection | 300‑frame chunks, native FPS, YOLO11x/m at 640/960 |
| 2 | High‑Tenacity Tracking & Box Stitching | BoT‑SORT + Wave 1 bbox stitch with relative radius & velocity consistency |
| 3 | Resolution‑Aware Box Pruning (Enhanced) | Duration, kinetic, spatial outliers, traversal (walkers), bbox size, geometric mirrors |
| 4 | High‑Fidelity Pose Estimation (Visibility‑Aware) | ViTPose, fp16 MPS, batched [N,3,256,192], visibility scoring to skip occluded tracks |
| 5 | Keypoint Collision Dedup | Remove duplicate pose overlays on the same physical person |
| 6 | Skeleton Re‑Association (Hybrid CVM) | OKS re‑ID + crossover refinement with hybrid CVM + freeze/decay |
| 7 | Post‑Pose Pruning | Sync‑score pruning (non‑dancers) + Smart Mirror pruning |
| 8 | Temporal Smoothing | 1 Euro Filter with conf&lt;0.3 guard |
| 9 | Spatio‑Temporal Scoring & Visualization | Group Truth + cDTW + per‑joint heatmaps, JSON + MP4 export |

---

## Phase 1: Streaming Ingestion & Detection

**Module:** `tracker.py`

### Goal

Stream long videos safely and detect all potential dancers at native FPS.

### What Happens

1. **Chunking:** Video read in 300‑frame chunks (≈10 s at 30 FPS). Full video never held in RAM.
2. **Native FPS (30+):** Process every frame at native FPS (no temporal decimation).
3. **YOLO11 (n/m/l/x):** Detection at 640×640 by default; resolution auto‑upsized to 960×960 if crowd density exceeds a threshold in the previous chunk.
4. **Lowered Confidence:** `YOLO_CONF=0.25`. Prefer recall (catch all dancers) and let later pruning remove non‑dancers.
5. **Resolution scaling:** Frames resized for YOLO; boxes scaled back to original resolution.

### Output

```python
raw_tracks: Dict[track_id, List[(frame_idx, (x1,y1,x2,y2), confidence), ...]]
total_frames: int
output_fps: float  # native FPS
native_fps: float
frame_width: int
frame_height: int
```

---

## Phase 2: High‑Tenacity Tracking & Box Stitching

**Module:** `tracker.py`, `botsort.yaml`, `ocsort.yaml`

### Goal

Maintain long‑lived, stable IDs through occlusions and crossings.

### What Happens

1. **BoT‑SORT:** `track_buffer=90` (≈3 s at 30 FPS), tuned thresholds in `botsort.yaml`.
2. **Wave 1 Stitching (BBox, V3.4):** When ID_A dies and ID_B appears within `STITCH_MAX_FRAME_GAP` frames:
   - Compute A’s last center and velocity.
   - Use a **relative radius** derived from A’s bbox height (e.g. 0.5× for stationary, 0.75× for predicted re‑appearance), with a small absolute fallback.
   - Merge B into A if B spawns near A’s last position or predicted position and has **velocity direction consistent** with A (dot product &gt; 0).
   - Interpolate boxes across the gap.
3. **Coexistence Deduplication:** If two tracks overlap with high IoU for many frames, keep the older/longer track and delete the ghost.

### Output

Same `raw_tracks` with stitched and deduplicated IDs; no fragment gaps within 90 frames; ghost duplicates merged.

---

## Phase 3: Resolution‑Aware Box Pruning (Enhanced)

**Module:** `track_pruning.py`

### Goal

Remove non‑dancers (audience, walkers, seated people) and obvious mirrors while keeping side/far dancers.

### Rules (V3.4)

1. **Duration:** Track must appear in &gt; **20%** of its *possible* lifespan (`possible_frames = total_frames - first_frame`), with a hard floor of **30 frames**. Late entrants are not penalized.
2. **Normalized Kinetic Pruning:** Bounding box center movement std must exceed `KINETIC_STD_FRAC = 0.03` times the median dancer bbox height (resolution‑agnostic).
3. **Spatial Outlier Filter:** Compute median center for each track that survives duration + kinetic. Compute group centroid and std across all such centers. Prune tracks whose normalized distance exceeds **2σ**, using a **minimum spread floor** (`0.05 * frame_dimension`) to avoid false positives when dancers are tightly clustered. Late entrants moving toward the group centroid are preserved.
4. **Traversal Detector (Walkers):** For each track, compute:
   - `straightness = total_displacement / path_length`
   - `traversal_frac = total_displacement / frame_width`
   - acceleration variance (2nd derivative of center positions), normalized by bbox height.  
   Prune **only** if **all** hold: `straightness > 0.7`, `traversal_frac > 0.3`, `normalized_accel_var < 0.02`. Walkers have high straightness and near‑zero acceleration; dancers accelerating/braking to new formations are kept.
5. **Bbox Size Outliers:** For surviving tracks (≥3), compute median bbox height per track and group median. Prune tracks whose median height is &lt; **40%** or &gt; **200%** of the group median.
6. **Geometric Mirrors:** Pre‑pose filtering based on edge presence and inverted x‑velocity relative to group mean, to strip obvious mirrors early and save ViTPose compute.

### Output

`surviving_ids: Set[int]` passed to downstream pose estimation.

---

## Phase 4: High‑Fidelity Pose Estimation (Visibility‑Aware)

**Modules:** `pose_estimator.py`, `crossover.py` (visibility helper), `main.py`

### Goal

Get high‑fidelity skeletons for every visible dancer while **skipping pose** for heavily occluded tracks to avoid double‑counting dancers.

### What Happens

1. **Engine:** ViTPose‑Plus Base/Large (`usyd-community/vitpose-plus-base|large`). fp16 on M2/CUDA when available.
2. **Batching:** Stack cropped person images into `[N, 3, 256, 192]` for a single forward pass per frame.
3. **Dynamic Bbox Padding:** Default 15%; shrinks to 10% for slow, stable boxes; expands to 25% for fast motion or large bbox deltas to avoid cutting off limbs.
4. **Visibility Scoring (V3.4):** For each frame:
   - Compute **containment** between boxes and use **foot Y‑position (ymax)** to infer depth: lower feet = closer to camera.
   - For each track, visibility ∈ [0,1] is reduced when its box is heavily contained inside a box whose feet are lower.
   - Tracks with visibility &lt; 0.3 **skip ViTPose**; they receive a decayed copy of their last good pose instead (confidence ×0.85 per skip).
5. Coordinates are mapped back to global image space.

### Keypoints (17 COCO)

- Face: nose, eyes, ears
- Upper body: shoulders, elbows, wrists
- Lower body: hips, knees, ankles

### Output

```python
raw_poses_by_frame: List[Dict[int, {"keypoints": (17,3), "scores": (17,)}]]
```

---

## Phase 5: Keypoint Collision Dedup

**Module:** `crossover.py`

### Goal

Prevent multiple pose overlays from being drawn on the **same physical person** when tracking fragments or crossover mis‑assignments occur.

### What Happens

For each frame:

1. For every pair of tracks with poses:
   - Compute per‑keypoint distances and take the **median** (robust to 1–2 hallucinated keypoints).
   - Compute bbox height and centers for both tracks.
2. Apply two gates:
   - **Gate 1:** median keypoint distance &lt; 0.2× bbox height → skeletons are collocated.
   - **Gate 2:** center distance &lt; 0.3× bbox height → bboxes are nearly on top of each other (protects partner work where skeletons can be close but bboxes are offset).
3. If both gates pass, suppress the track with lower average keypoint confidence.

### Output

Per‑frame `poses` dictionaries with duplicate overlays removed.

---

## Phase 6: Skeleton Re‑Association (Hybrid CVM)

**Module:** `crossover.py`

### Goal

Handle occlusion‑induced ID breaks and crossovers **after** pose estimation using OKS and a hybrid CVM model that is smooth for micro‑occlusions and robust for long occlusions.

### What Happens

1. **Wave 2 OKS Re‑ID:** `apply_occlusion_reid()` scans dead vs newborn tracks. If OKS between dead track’s last pose and newborn’s first pose ≥ 0.5 within 90 frames, newborn is merged into the old ID.
2. **Crossover Refinement (OKS):** `apply_crossover_refinement()`:
   - Detects overlapping pairs with IoU &gt; 0.6.
   - Compares how well each current pose matches each track’s **previous** pose using OKS.
   - Swaps IDs when the “swapped” assignment yields a higher OKS score.
3. **Hybrid CVM (V3.4):** For tracks whose total keypoint confidence &lt; 0.3:
   - Maintain per‑track occlusion frame counters.
   - **Frames 1–5 of occlusion:**  
     Use CVM to project bbox and keypoints forward using last known velocity, decaying velocity by 0.9 each frame (smooth micro‑occlusions, e.g., a hand passing in front of a face).
   - **Frames 6+ of occlusion:**  
     Freeze the last CVM‑propagated pose and decay its confidence by 0.85× per frame. After ~15 frames, confidence drops below rendering threshold and the pose disappears while the bbox continues via BoT‑SORT.
   - **Re‑emergence:**  
     When live keypoint confidence returns &gt;= 0.3, blend from frozen to live pose over 3 frames to avoid visual snapping.

---

---

## Phase 7: Post‑Pose Pruning (Sync + Smart Mirrors)

**Module:** `track_pruning.py`

### Goal

Remove remaining non‑dancers (standing, seated, background walkers) **after** we know how they move, and strip mirror ghosts using pose information.

### Sync‑Score Pruning (V3.4)

1. Build joint‑angle time series for all tracks vs group truth (from `scoring.py`).
2. For each track and each of the 6 joints, compute normalized correlation with the group truth across frames where both are valid.
3. Use the **maximum** absolute correlation across joints (not mean) as the track’s sync score, so soloists or subgroup dancers that synchronize at least one joint aren’t penalized.
4. Prune tracks with max correlation &lt; **0.10** — these correspond to people essentially not dancing (standing, sitting, or slow walkers).

### Smart Mirror Pruning (unchanged logic)

After sync pruning, `prune_smart_mirrors()` removes mirrors that satisfy **all**:

1. Track spends a significant fraction of time in the **outer 10%** of the frame horizontally.
2. Mean x‑velocity is **inverted** relative to the group’s mean x‑velocity.
3. Lower‑body keypoints (knees, ankles) have low average confidence (&lt; 0.3).

Output is a final set of surviving dancer track IDs.

---

## Phase 8: Temporal Smoothing

**Module:** `smoother.py`

### What Happens

**1 Euro Filter** applied to all `(x, y)` keypoints.

- Parameters: `min_cutoff=1.0`, `beta=0.7`, `d_cutoff=1.0`
- **Guard:** Suspended if keypoint confidence &lt; 0.3 to prevent smoothing hallucinated geometry.

### Output

Same pose structure with smoothed `(x, y)`; confidence unchanged.

---

## Phase 9: Spatio‑Temporal Scoring & Visualization

**Modules:** `scoring.py`, `visualizer.py`

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

### Outputs

1. **JSON** (`output/data.json`): Keypoints, boxes, joint_angles, deviations, shape_errors, timing_errors for each dancer and frame.
2. **MP4** (`output/<video_name>_poses.mp4`): Rendered at native FPS with **dual‑signal heatmap**:

| Condition | Color |
|-----------|-------|
| Shape within tolerance AND Timing ≤ 2 frames | **Green** |
| Shape good, Timing &gt; 2 frames | **Blue** (Off-beat) |
| Minor Shape Error | **Yellow** |
| Major Shape Error | **Red** |
| Missing, Occluded, or Ripple | **Gray** |

Per-joint thresholds: Spine/Hips 10°/20°, Elbows/Knees 20°/35°.

---

## Data Flow Summary (V3.4)

```
Video (MP4)
    │
    ▼
[1] Streaming 300-frame chunks, native FPS → YOLO11 (dynamic 640/960)
    │
    ▼
[2] BoT-SORT (track_buffer=90) + Wave 1 bbox stitch (relative radius + velocity-consistency) + coexistence dedup → raw_tracks
    │
    ▼
[3] prune_tracks (duration + kinetic)
    ├─ prune_by_stage_polygon (optional)
    ├─ prune_spatial_outliers (min-spread floor, late-entrant aware)
    ├─ prune_traversal_tracks (walkers)
    ├─ prune_bbox_size_outliers
    └─ prune_geometric_mirrors
        → surviving_ids → raw_tracks_to_per_frame → tracking_results
    │
    ▼
[4] ViTPose Base/Large (fp16 MPS) + dynamic bbox padding + compute_visibility_scores
    ├─ visibility < 0.3 → reuse last_good_pose with decayed confidence
    └─ visibility ≥ 0.3 → full ViTPose inference
        → raw_poses_by_frame
    │
    ▼
[5] deduplicate_collocated_poses (median keypoint distance + center gate)
    │
    ▼
[6] apply_occlusion_reid (OKS > 0.5) + apply_crossover_refinement (hybrid CVM + OKS crossover)
    │
    ▼
[7] prune_low_sync_tracks (low correlation to group truth) + prune_smart_mirrors (edge+velocity+low lower-body)
    │
    ▼
[8] PoseSmoother (1 Euro, conf<0.3 guard) → smoothed poses
    │
    ▼
[9] process_all_frames_scoring_vectorized (circmean, cDTW, per-joint thresholds)
    │
    ▼
render_and_export → data.json + <video>_poses.mp4 (dual-signal heatmap)
```

---

## Key Tunable Parameters (V3.4)

| Location | Parameter | Default | Effect |
|----------|-----------|---------|--------|
| `tracker.py` | `CHUNK_SIZE` | 300 | Frames per streaming chunk |
| `tracker.py` | `YOLO_CONF` | 0.25 | Detection confidence threshold |
| `tracker.py` | `CHUNK_SIZE` | 300 | Frames per streaming chunk |
| `tracker.py` | `STITCH_MAX_FRAME_GAP` | 180 | Max frames for Wave 1 bbox stitch |
| `tracker.py` | `STITCH_RADIUS_BBOX_FRAC` | 0.5 | Stationary stitch radius as fraction of bbox height |
| `tracker.py` | `STITCH_PREDICTED_RADIUS_FRAC` | 0.75 | Predicted stitch radius fraction when velocity is known |
| `tracker.py` | `STITCH_MAX_PIXEL_RADIUS` | 120.0 | Fallback absolute radius when bbox height unavailable |
| `track_pruning.py` | `min_duration_ratio` | 0.20 | Min fraction of *possible* frames track must appear |
| `track_pruning.py` | `KINETIC_STD_FRAC` | 0.03 | Kinetic threshold = 3% of median bbox height |
| `track_pruning.py` | `SPATIAL_MIN_SPREAD_FRAC` | 0.05 | Min spread floor (fraction of frame width/height) for spatial outliers |
| `track_pruning.py` | `SPATIAL_OUTLIER_STD_FACTOR` | 2.0 | Normalized distance threshold for spatial outliers |
| `track_pruning.py` | `TRAVERSAL_STRAIGHTNESS` | 0.7 | Straightness threshold for traversal pruning |
| `track_pruning.py` | `TRAVERSAL_MIN_FRAC` | 0.3 | Min fraction of frame width a traversal must cover |
| `track_pruning.py` | `TRAVERSAL_MAX_ACCEL_VAR` | 0.02 | Max normalized acceleration variance for walkers |
| `track_pruning.py` | `BBOX_SIZE_MIN_FRAC` | 0.40 | Min median bbox height as fraction of group median |
| `track_pruning.py` | `BBOX_SIZE_MAX_FRAC` | 2.00 | Max median bbox height as fraction of group median |
| `track_pruning.py` | `EDGE_MARGIN_FRAC` | 0.10 | Outer 10% of frame for mirror rule |
| `track_pruning.py` | `min_lower_body_conf` | 0.3 | Lower-body conf threshold for mirror |
| `track_pruning.py` | `SYNC_SCORE_MIN` | 0.10 | Min max-correlation with group truth to keep a track |
| `botsort.yaml` | `new_track_thresh` | 0.35 | Min conf to spawn new track ID (V3.1: pick up side-dancers) |
| `botsort.yaml` | `min_hits` | 3 | Track must be detected 3 consecutive frames to become confirmed |
| `pose_estimator.py` | `model_name` | vitpose-plus-base/large | ViTPose++ model |
| `pose_estimator.py` | `BBOX_PADDING` | 0.15 | Base bbox padding before cropping (dynamic in `main.py`) |
| `crossover.py` | `REID_MAX_FRAME_GAP` | 90 | Max frames for Wave 2 OKS stitch |
| `crossover.py` | `REID_MIN_OKS` | 0.5 | Min OKS to merge tracks |
| `crossover.py` | `VISIBILITY_CONTAINMENT_THRESH` | 0.7 | Containment threshold for occlusion visibility scoring |
| `crossover.py` | `VISIBILITY_MIN_SCORE` | 0.3 | Visibility cutoff used in `main.py` to skip ViTPose |
| `crossover.py` | `CVM_MAX_FRAMES` | 5 | Frames of CVM extrapolation before freezing pose |
| `crossover.py` | `CONF_DECAY_FACTOR` | 0.85 | Confidence decay per occluded frame after CVM ends |
| `crossover.py` | `BLEND_FRAMES` | 3 | Frames to blend from frozen to live pose on re-emergence |
| `crossover.py` | `COLLISION_KPT_DIST_FRAC` | 0.2 | Median keypoint distance fraction for collision gate 1 |
| `crossover.py` | `COLLISION_CENTER_DIST_FRAC` | 0.3 | Center distance fraction for collision gate 2 |
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

Optimized for **Apple Silicon (M2)** using PyTorch MPS backend. fp16 inference for ViTPose Base/Large. Falls back to CPU if MPS is unavailable. The added V3.4 logic (visibility scoring, sync pruning, spatial outliers) is pure NumPy and has negligible overhead relative to YOLO + ViTPose.
