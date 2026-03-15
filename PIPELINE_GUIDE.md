# Sway Pose Tracking V3.0 — Pipeline Guide

A production-ready, M2-optimized pipeline designed to handle severe occlusions, complex choreography, and long video lengths without crashing.

---

## Overview (V3.0)

The pipeline processes video in a **linear, streaming** flow to fix the Phase 2a temporal paradox and prevent RAM bloat. It consists of **9 phases**:

| Phase | Name | Purpose |
|-------|------|---------|
| 1 | Streaming Ingestion & Detection | 300-frame chunks, native FPS, YOLO11l conf=0.25 |
| 2 | High-Tenacity Tracking & Box Stitching | BoT-SORT track_buffer=90, Wave 1 bbox stitch |
| 3 | Resolution-Aware Box Pruning | 20% duration, normalized kinetic (5% bbox height) |
| 4 | High-Fidelity Pose Estimation | ViTPose-Large, fp16 MPS, batched [N,3,256,192] |
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
3. **YOLO11l (Large):** Detection at 640×640.
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

1. **BoT-SORT:** `track_buffer=90` — 3 full seconds at 30 FPS to wait for occluded dancer to reappear.
2. **Wave 1 Stitching (BBox):** When ID_A dies and ID_B spawns within 90 frames, check pixel distance. If ID_B spawns near ID_A's last known location (or predicted trajectory), merge them. Interpolate missing boxes.

### Output

Same `raw_tracks` with stitched IDs; no fragment gaps within 90 frames.

---

## Phase 3: Resolution-Aware Box Pruning

**Module:** `track_pruning.py`

### Goal

Remove non-dancers (audience, chairs) without breaking at different video resolutions.

### Rules

1. **Duration:** Must appear in &gt; **20%** of the video (`min_duration_ratio=0.20`).
2. **Normalized Kinetic Pruning:** Bounding box center must move by a standard deviation greater than **5%** of the median dancer's bounding box height. Resolution-agnostic across 720p, 1080p, 4K.

### Algorithm

- Compute `median_dancer_bbox_height` across tracks that pass duration filter.
- For each track: `movement_std = sqrt(std_x² + std_y²)`
- Prune if `movement_std < 0.05 * median_dancer_bbox_height`

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

## Phase 6: Smart Mirror Pruning

**Module:** `track_pruning.py`

### Goal

Delete reflections without deleting legitimate edge dancers.

### Rule

Prune IF AND ONLY IF all three hold:

1. Track exists in **outer 10%** of the frame (left or right edge).
2. Its x-velocity is **inverted** relative to the group.
3. Lower-body keypoints (ankles, knees) have average confidence **&lt; 0.3** (studio mirrors cut off at floor).

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
[1] Streaming 300-frame chunks, native FPS → YOLO11l (640×640, conf=0.25)
    │
    ▼
[2] BoT-SORT (track_buffer=90) + Wave 1 box stitch (90 frames) → raw_tracks
    │
    ▼
[3] prune_tracks (20% duration, 5% bbox-height kinetic) → surviving_ids → raw_tracks_to_per_frame
    │
    ▼
[4] ViTPose-Large (fp16 MPS) → raw_poses_by_frame
    │
    ▼
[5] Wave 2: apply_occlusion_reid (OKS>0.5) + apply_crossover_refinement
    │
    ▼
[6] prune_smart_mirrors (10% edge + inverted velocity + low lower-body conf)
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
| `tracker.py` | `STITCH_MAX_FRAME_GAP` | 90 | Max frames for Wave 1 box stitch |
| `tracker.py` | `STITCH_MAX_PIXEL_RADIUS` | 120.0 | Max pixel distance (stationary) for stitch |
| `tracker.py` | `STITCH_PREDICTED_RADIUS` | 180.0 | Max pixel distance (velocity-predicted) |
| `track_pruning.py` | `min_duration_ratio` | 0.20 | Min fraction of frames track must appear |
| `track_pruning.py` | `KINETIC_STD_FRAC` | 0.05 | Kinetic threshold = 5% of median bbox height |
| `track_pruning.py` | `EDGE_MARGIN_FRAC` | 0.10 | Outer 10% of frame for mirror rule |
| `track_pruning.py` | `min_lower_body_conf` | 0.3 | Lower-body conf threshold for mirror |
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
