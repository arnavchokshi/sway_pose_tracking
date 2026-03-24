# How 3D pose works (full pipeline context)

This document describes **where 3D pose lifting sits** in the Sway pose MVP pipeline and **what it does today**, so it can be read without digging through `main.py` and `pose_lift_3d.py`.

---

## 1. End-to-end pipeline (where 3D fits)

The CLI / `main.py` run is structured as **11 numbered steps** (see the file header in `main.py`). In order:

| Step | Phase | Role |
|------|--------|------|
| 1–2 | Detection + tracking | YOLO + tracker; boxes and track IDs per frame |
| 3 | Post-track stitching | Dormant tracks, stitch, coalesce, merge |
| 4 | Pre-pose pruning | Drop bad tracks before pose |
| 5 | Pose (ViTPose) | COCO-17 keypoints per person (pixels + confidence) |
| 6 | Association | Occlusion re-ID, crossover handling, acceleration audit |
| 7 | Collision cleanup | Keypoint dedup, bbox sanitize |
| 8 | Post-pose pruning | Tier A/B/C rules; surviving tracks |
| 9 | Temporal smoothing | 1-Euro filter on 2D poses (`PoseSmoother`) |
| **—** | **3D lift (not a separate printed phase)** | **MotionAGFormer + optional depth; runs here if enabled** |
| 10 | Spatio-temporal scoring | Deviations, angles, cDTW, etc. |
| 11 | Export | Videos, `data.json`, including optional `pose_3d` |

**Important:** 3D lifting runs **after** Phase 9 smoothing and **before** Phase 10 scoring. That means:

- The lifter sees **smoothed** 2D tracks (better temporal consistency than raw ViTPose).
- Scoring still primarily uses **`lift_xyz`** (model-space 3D from the lifter) where implemented, alongside 2D-derived metrics.

If 3D is disabled (`SWAY_3D_LIFT=0` or `--no-pose-3d-lift`), the pipeline skips lift and export of `pose_3d` (unless other code paths add it).

---

## 2. What “3D pose” means in this project

We do **not** build textured mesh avatars. We produce:

1. **`lift_xyz`** — 17×3 points per person per frame in **postprocessed MotionAGFormer space** (normalized, fixed H36M-style demo rotation in code). Used internally (e.g. 3D-aware scoring angles).
2. **`keypoints_3d`** — What downstream consumers treat as “3D for display / JSON”:
   - **Default (unified export on):** 17×`[X, Y, Z]` in a **single world-style frame** (Y up, Z as scene depth along the pinhole model), so the browser viewer can plot true 3D layout.
   - **Legacy (`SWAY_UNIFIED_3D_EXPORT=0`):** `[pixel_x, pixel_y, z_lift]` (old mixed representation).

3. **`pose_3d` blob** in `data.json` — Compact per-track series for `pose_3d_viewer.html` and `GET /api/runs/{id}/pose_3d`.

4. **2D overlay video** (`*_3d.mp4` when enabled) — Still draws skeletons at **ViTPose image (x,y)**; **`keypoints_3d`’s third component** is used mainly as a **depth shading** cue, not as screen position.

---

## 3. 3D lift pipeline (step by step)

Implementation: **`sway/pose_lift_3d.py`**. Orchestration: **`main.py`** (after smoothing).

### 3.1 Dependencies

- **Lifter backend:** `SWAY_LIFT_BACKEND=motionagformer` (default) or `poseformerv2`.
- **MotionAGFormer:** repo on `PYTHONPATH` or under `vendor/MotionAGFormer` (or `SWAY_MOTIONAGFORMER_ROOT`). Weights: `models/motionagformer-l-h36m.pth.tr` (or `.pth`), or `SWAY_MOTIONAGFORMER_WEIGHTS`. **`timm`**, PyTorch.
- **PoseFormerV2 (optional):** `vendor/PoseFormerV2` or `SWAY_POSEFORMERV2_ROOT`. Weights: e.g. `models/27_243_45.2.bin` (`python -m tools.prefetch_models --include-poseformerv2`) or `SWAY_POSEFORMERV2_WEIGHTS`. **`einops`**, **`torch-dct`**, **`timm`**. Official checkpoints are **Human3.6M** (and MPI-INF in the upstream repo)—**not AMASS**; use the same COCO→H36M→COCO bridge as their video demo.
- Optional: **Depth Anything V2** via Hugging Face (`transformers` pipeline) for depth maps — used for AugLift-style **z refinement** and for **root depth** when building unified XYZ.

### 3.2 Dense 2D sequence per track

For each track ID, we build a dense array `(T, 17, 3)` for `T = total_frames`:

- **Observed frames:** pixel `x, y` and ViTPose **confidence** in the third channel.
- **Missing frames (default `SWAY_LIFT_GAP_MODE=hold_zero`):** keep **last known (x,y)** and set **confidence to 0** (no linear interpolation through gaps). Optional `linear_interp` restores the old behavior.

Requires enough observations (≥ ~10 frames) or the track is skipped for lift.

### 3.3 MotionAGFormer (default backend)

- Input windows of up to **243** frames (shorter clips padded).
- 2D input normalized **per person, per frame**: mid-hip (COCO 11+12) at the origin, scale = 1.2× max bbox extent so coordinates sit in ~[-1, 1] (matches bbox-cropped training better than full-frame norm). Override with **`SWAY_LIFT_INPUT_NORM=screen`** for the old full-frame mapping.
- **Flip augmentation:** horizontal flip with **correct COCO left/right index pairs** (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles); averaged with the unflipped forward pass.
- **Postprocess per frame:** subtract **COCO mid-hip (11+12)** as root, fixed **H36M demo quaternion**, min-z shift, max-abs scale → **`lift_xyz`** / internal `pred_3d`.

### 3.3b PoseFormerV2 (`SWAY_LIFT_BACKEND=poseformerv2`)

- **243**-frame sliding window (pad with edge replication); per-frame **center prediction** (upstream design).
- ViTPose **COCO** pixels → **H36M-17** 2D layout (same rules as `PoseFormerV2/demo/lib/preprocess.py`), then **VideoPose3D-style** `normalize_screen_coordinates` with image `w,h`. **TTA** uses H36M left/right joint sets from their `demo/vis.py`.
- 3D output is **H36M-ordered** in the network; we map back to **COCO-17** for **`lift_xyz`** (limbs exact inverse of the demo map; face keypoints approximated from neck/head).
- Postprocess: zero **H36M pelvis (joint 0)**, same demo quaternion / min-z / max-abs scale as above for comparable scale.

### 3.4 Depth (temporal, strided)

When **`SWAY_DEPTH_DYNAMIC`** is on (default):

- **`collect_strided_depth_series`** in `sway/depth_stage.py` runs Depth Anything on every **N**th frame (`N` from `SWAY_DEPTH_STRIDE_FRAMES` or ~**output_fps**, ~1 second).
- For each frame index `t`, a full-res depth map is **linearly interpolated** between the two surrounding keyframes (per pixel, after resize).

If dynamic depth is off or no maps are produced, the code falls back to **one** depth map from the **first** video frame.

### 3.5 AugLift (z blend on model output)

For each time step and joint, sample **normalized** depth at the **pixel** `(x,y)` from the smoothed ViTPose sequence. Blend:

`refined_z = (1 - blend) * pred_z + blend * depth_z`

Blend: **`SWAY_AUGLIFT_BLEND`** (default 0.3).

### 3.6 Unified world `keypoints_3d` (default export)

When **`SWAY_UNIFIED_3D_EXPORT`** is on (default) and lift actually produced **`root_xyz`**:

- **Intrinsics:** `fx, fy, cx, cy` from frame size and **`SWAY_PINHOLE_FOV_DEG`** (~70°), or **`SWAY_FX` / `SWAY_FY`**.
- **Root depth:** sample **interpolated** normalized depth at **mid-hip**; map to camera Z with **`SWAY_DEPTH_Z_NEAR` / `SWAY_DEPTH_Z_FAR`**. If no depth map, use **`SWAY_DEFAULT_ROOT_Z`**.
- **Pelvis translation:** pinhole un-projection of the hip pixel with that Z (Y flipped to **Y-up** for Three.js-style viewers).
- **Limbs:** subtract **hip centroid in lift space** from `lift` joints, scale by a factor derived from **2D body height × Z / fy** vs lift span, optional **`SWAY_LIFT_WORLD_SCALE`**.

This is **not guaranteed metric**: monocular depth is relative per frame; the near/far mapping is a **scene-depth bridge**.

### 3.7 Export (`export_3d_for_viewer`)

- **`version`: 2** and **`camera`** { width, height, fx, fy, cx, cy, fov_deg, units, convention } only if unified export is on **and** at least one pose has **`root_xyz`** (so we never advertise v2 with pure 2D fallback keypoints).
- **`version`: 1** for legacy unified-off or no successful world lift.
- Per track: `frames`, `keypoints_3d`, optional parallel **`root_xyz`** when unified.

Embedded into **`data.json`** during export and served by Pipeline Lab as **`pose_3d`**.

---

## 4. Browser viewer

**`pipeline_lab/web/public/pose_3d_viewer.html`**

- If **`version >= 2`** and **`camera`** exist: treats **`keypoints_3d`** as **world XYZ** and **does not** fake side-by-side spacing for multiple dancers.
- Otherwise: legacy pixel-mix scaling and horizontal spread.

---

## 5. Environment variables (quick reference)

| Variable | Role |
|----------|------|
| `SWAY_3D_LIFT` / `--no-pose-3d-lift` | Master switch for lift + `pose_3d` export |
| `SWAY_UNIFIED_3D_EXPORT` | `1` = world XYZ + camera metadata (default) |
| `SWAY_DEPTH_DYNAMIC` | `1` = strided depth over time (default) |
| `SWAY_DEPTH_STRIDE_FRAMES` | Override depth keyframe stride |
| `SWAY_LIFT_GAP_MODE` | `hold_zero` (default) or `linear_interp` |
| `SWAY_PINHOLE_FOV_DEG`, `SWAY_FX`, `SWAY_FY` | Camera intrinsics |
| `SWAY_DEPTH_Z_NEAR`, `SWAY_DEPTH_Z_FAR` | Map normalized depth → Z |
| `SWAY_DEFAULT_ROOT_Z` | Z when depth unavailable |
| `SWAY_LIFT_WORLD_SCALE` | Extra scale on limb offsets |
| `SWAY_AUGLIFT_BLEND` | Depth vs model z blend |
| `SWAY_LIFT_BACKEND` | `motionagformer` (default) or `poseformerv2` |
| `SWAY_LIFT_INPUT_NORM` | `person` (default) or `screen` for 2D tensor fed to MotionAGFormer only |
| `SWAY_MOTIONAGFORMER_ROOT`, `SWAY_MOTIONAGFORMER_WEIGHTS`, `SWAY_MOTIONAGFORMER_N_LAYERS` | MotionAGFormer |
| `SWAY_POSEFORMERV2_ROOT`, `SWAY_POSEFORMERV2_WEIGHTS` | PoseFormerV2 repo + checkpoint (`.bin` with `model_pos`) |
| `SWAY_POSEFORMERV2_NFRAMES`, `SWAY_POSEFORMERV2_FRAME_KEPT`, `SWAY_POSEFORMERV2_COEFF_KEPT`, `SWAY_POSEFORMERV2_DEPTH`, `SWAY_POSEFORMERV2_EMBED_RATIO` | Must match downloaded checkpoint (defaults = 243 / 27 / 27 / 4 / 32) |

---

## 6. Related files

| File | Purpose |
|------|---------|
| `sway/pose_lift_3d.py` | Lift, unified math, export helper |
| `sway/depth_stage.py` | Depth Anything; `collect_strided_depth_series` |
| `main.py` | Runs lift after smooth; builds depth series; calls export |
| `sway/visualizer.py` | `*_3d.mp4` overlay using 2D + z shading |
| `sway/scoring.py` | Uses `lift_xyz` for 3D angle path |
| `pipeline_lab/web/public/pose_3d_viewer.html` | Three.js viewer |
| `docs/GEMINI_3D_POSE_LIFT_BRIEF.txt` | Denser technical brief for model / research handoff |

---

## 7. Limitations (honest)

- **Public lifter weights (MotionAGFormer, PoseFormerV2, MixSTE upstream)** are almost all **Human3.6M / MPI-INF**—not AMASS/AIST++. Swapping architecture helps robustness/noise, not distribution, unless you **train or fine-tune** on dance mocap.
- **Depth** is **relative** (per-frame min–max normalized); global metric scale is not solved here.
- **hold_zero + confidence** helps gaps; MotionAGFormer uses 3-channel input; PoseFormerV2 uses **2D only** (confidence ignored for that backend).

For a shorter research-oriented spec, see **`docs/GEMINI_3D_POSE_LIFT_BRIEF.txt`**.
