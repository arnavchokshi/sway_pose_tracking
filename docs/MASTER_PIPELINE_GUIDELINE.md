# Sway Pose MVP — Master pipeline guideline

This document describes the **implemented** production video pipeline in `sway_pose_mvp`: stage order, data contracts, defaults, environment variables, and where behavior is defined in code. It is the only narrative under `docs/` in this repo; all claims below are tied to the cited modules. **§21–§27** list every Lab schema field plus CLI, YAML, env-only, and module-constant parameters and how they affect the run.

**Canonical implementation sources (verify after large refactors):**

| Source | Role |
|--------|------|
| `main.py` (module docstring + progress `[1/11]`–`[11/11]`) | Orchestration order |
| `sway/pipeline_config_schema.py` | Pipeline Lab / Config UI: stages + fields → `env` / YAML / CLI keys |
| `pipeline_lab/server/app.py` | Lab API: maps `tracker_technology` + schema → subprocess `env` / `main.py` CLI |
| `sway/pipeline_matrix_presets.py` | Curated one-knob A/B recipes; `GET /api/pipeline_matrix` + `POST /api/runs/batch_path` |
| `tools/pipeline_matrix_runs.py` | CLI: queue matrix against a server-side video path (calls Lab API) |
| `sway/tracker.py` | Detection, tracking (BoxMOT Deep OC-SORT + optional BoT-SORT via env), post-track stitching |
| `sway/experimental_hooks.py` | In-pipeline optional hooks: GNN refine flag (identity pass), HMR mesh sidecar JSON |
| `sway/track_pruning.py` | Pre- and post-pose pruning rules |
| `sway/crossover.py` | Association, collision, dedup |
| `sway/pose_estimator.py` | ViTPose+ inference |
| `sway/pose_lift_3d.py` | 3D lifting (MotionAGFormer / optional PoseFormerV2), export helpers |
| `sway/hybrid_sam_refiner.py` | SAM2 overlap refinement on the BoxMOT path |

---

## 1. Executive summary

The pipeline turns one **input video** into:

- Per-frame **boxes**, **track IDs**, **2D poses** (COCO-17 keypoints + scores), optional **segmentation masks** from hybrid SAM,
- Optional **3D poses** (`lift_xyz`, unified `keypoints_3d`, `pose_3d` blob in JSON when lift succeeds),
- **Spatio-temporal scoring** (angles, deviations, shape/timing errors) when `process_all_frames_scoring_vectorized` returns data,
- **Artifacts** under `--output-dir` (default `output/`): `data.json`, MP4 visualizations, `prune_log.json`, optional montage and phase previews.

Execution is **single-process** with a clear phase order. **Configuration** stacks in this precedence (later wins where applicable):

1. **Stack defaults** in `main._apply_stack_default_env()` (only sets missing `SWAY_*`).
2. **`--params` YAML**: the full dict is kept as `params` in `main.py`. Additionally, `_apply_params_to_env()` copies only keys starting with `SWAY_` plus offline helpers (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `YOLO_OFFLINE`, `ULTRALYTICS_OFFLINE`) into `os.environ` (booleans as `"1"`/`"0"`).
3. **Explicit environment** set before launch.

---

## 2. Critical platform facts (read first)

### 2.1 Two different compute devices

- **ViTPose / RTMPose** use `get_device()` in `main.py`: **`torch.device("mps")`** if `torch.backends.mps.is_available()`, else **`cpu`**. There is **no CUDA branch** in `get_device()` — on a typical Linux + NVIDIA setup, pose runs on **CPU** unless you change this code.
- **BoxMOT trackers** (inside `sway/tracker.py`): **`cuda`** if `torch.cuda.is_available()`, else **cpu**. They do **not** use `get_device()`.

Implications: on Apple Silicon with MPS, pose can use the GPU while BoxMOT uses CUDA on an eGPU, or CPU if no CUDA. On Mac without CUDA, BoxMOT is CPU. On NVIDIA desktops, YOLO/BoxMOT may use CUDA while ViTPose stays on CPU per `get_device()`.

### 2.2 Printed phases vs. hidden substeps

The CLI prints **11 phases**. **3D lifting** is **not** a separate printed phase: it runs **after Phase 9 (1-Euro smoothing)** and **before Phase 10 (scoring)** when enabled (see §10). Failures are caught: a message `[3D Lift] Skipped: …` is printed and the run continues.

### 2.3 Default “group dance” stack

Unless you pre-set env, `main` applies:

| Variable | Default | Effect |
|----------|---------|--------|
| `SWAY_GROUP_VIDEO` | `1` | Effective YOLO input size `max(SWAY_DETECT_SIZE or 640, 960)` in `tracker.py` |
| `SWAY_GLOBAL_LINK` | `1` | Phase 3 runs global track linking (`maybe_global_stitch` inside `apply_post_track_stitching`) |
| `SWAY_AUTO_STAGE_DEPTH` | `0` | Depth-based stage polygon is opt-in |

---

## 3. Entry point, CLI, and params

### 3.1 Command

```bash
python main.py <path/to/video.mp4>
```

### 3.2 Notable CLI defaults

Defined in `main.py` `argparse` (see that block for the full set):

| Argument | Default | Notes |
|----------|---------|--------|
| `--output-dir` | `output` | All outputs |
| `--pose-model` | `base` | `base`/`large`/`huge` → ViTPose+; `rtmpose` → MMPose RTMPose-L; `sapiens` → **ViTPose-Base** with a console note that Meta Sapiens is not bundled |
| `--pose-stride` | `1` | `2` skips every other pose frame; gaps filled per §8.4 |
| `--temporal-pose-refine` | **on** | `--no-temporal-pose-refine` disables; neighbor-frame keypoint blend (not 1-Euro). `SWAY_TEMPORAL_POSE_REFINE` overrides. |
| `--temporal-pose-radius` | `2` | Clamped 0–8 via `temporal_pose_radius()` |
| `--params` | none | YAML path |
| `--montage` | off | Stitches phase clips to `montage.mp4` |
| `--save-phase-previews` | off | `phase_previews/*.mp4` |
| `--no-pose-3d-lift` | off | Disables 3D lift; same effect as `SWAY_3D_LIFT` off |

### 3.3 Params YAML → environment

`_apply_params_to_env` (in `main.py`) only promotes **`SWAY_*`** and the offline keys listed in §1 into `os.environ`. All other YAML entries remain available only through the `params` dict passed to pruning, Re-ID, smoother, etc.

### 3.4 Pipeline matrix (compare one stage at a time)

For qualitative comparison on one clip, the Lab exposes **`GET /api/pipeline_matrix`** (recipe list) and **`POST /api/runs/batch_path`** (queue many runs; the video file is copied once then hardlinked per run when the OS allows it). Recipes live in **`sway/pipeline_matrix_presets.py`**. The matrix **locks** **`sway_yolo_weights` = `yolo26l_dancetrack`** for every row; additional stacks (detection, hybrid SAM, Phase 3 stitch, pose ViTPose/3D — §6.2.1, §6.4.1, §7.0.1, §9.0.1; Phase 6–7 re-ID / dedup distance gates — §11.0.1; Phase 8 Tier C + garbage baselines + three prune weights — §13.0.1; Phase 9 min cutoff + temporal blend radius — §14.0.1) are **master-locked** in `main.py` / Lab, not matrix fields. Each recipe then changes **one** other axis.

From a terminal (API must be running on the same machine as the path):

```bash
python -m tools.pipeline_matrix_runs --video /path/to/video.mp4
python -m tools.pipeline_matrix_runs --video /path/to/video.mp4 --only baseline,pose_large,det_stride2
```

The React Lab page can queue the same matrix from a **server path** field (browser upload is unchanged).

---

## 4. Data contracts between stages

### 4.1 After Phases 1–3 (tracking)

- **`raw_tracks`**: `Dict[track_id, List[observations]]` — per-frame boxes, confidences; observations may include `is_sam_refined`, `segmentation_mask` (hybrid SAM). After `apply_post_track_stitching`, **`maybe_gnn_refine_raw_tracks`** may run (currently identity); it mutates `raw_tracks` in place before Phase 4.

### 4.2 After Phase 4 → Phase 5 input

- **`tracking_results`**: per-frame dicts with `boxes`, `track_ids` for **surviving** tracks only (`raw_tracks_to_per_frame`).

### 4.3 From Phase 5 onward (`all_frame_data_pre` / `all_frame_data`)

Per-frame dicts typically include:

- `frame_idx`, `boxes`, `track_ids`, `poses` (per-track: `keypoints`, `scores`, …)
- `embeddings` (HSV-strip for Phase 6 re-ID)
- `is_sam_refined`, `segmentation_masks`
- After Phase 10 (when scoring returns): `track_angles`, `consensus_angles`, `deviations`, `shape_errors`, `timing_errors`
- After 3D lift (when enabled and successful): per-pose fields such as `lift_xyz`, `keypoints_3d`, `root_xyz` — see module docstring and `export_3d_for_viewer` in `sway/pose_lift_3d.py`

---

## 5. Phase map (master table)

| Print | Name | Responsibility | Primary modules |
|-------|------|----------------|-----------------|
| 1–2 | Detection + tracking (streaming) | YOLO person boxes, tracker IDs, optional hybrid SAM; optional **bidirectional** second pass (`SWAY_BIDIRECTIONAL_TRACK_PASS`) | `tracker.py`, `hybrid_sam_refiner.py`, `bidirectional_track_merge.py` |
| 3 | Post-track stitching | Dormant merges, global link, fragment stitch, coalesce, stride gap fill, complementary/coexisting merges | `tracker.py`, `dormant_tracks.py`, `global_track_link.py` |
| 4 | Pre-pose pruning | Drop non-dancers before ViTPose cost | `track_pruning.py`, `depth_stage.py` |
| 5 | Pose | ViTPose+, RTMPose-L, or **sapiens CLI** (ViTPose-Base weights); visibility gating; temporal keypoint refine | `pose_estimator.py`, `rtmpose_estimator.py`, `temporal_pose_refine.py`, `main.py` |
| 6 | Association | Occlusion re-ID, crossover refinement, acceleration audit | `crossover.py`, `reid_embedder.py` |
| 7 | Collision cleanup | Collocated pose dedup, bbox/pose sanitize | `crossover.py` |
| 8 | Post-pose pruning | Tier C / Tier A (protected) / Tier B voting | `track_pruning.py` |
| 9 | Temporal smoothing | Per-joint **1-Euro** filter | `smoother.py` |
| *(unprinted)* | **3D lift** | MotionAGFormer (default) + optional PoseFormerV2; optional depth refinement | `pose_lift_3d.py`, `depth_stage.py` |
| 10 | Scoring | Vectorized spatio-temporal metrics | `scoring.py` |
| 11 | Export | JSON, MP4s, prune overlays, optional montage | `visualizer.py` |

Pipeline Lab stage IDs (`PIPELINE_STAGES` in `pipeline_config_schema.py`) group some `main.py` phases for UX (e.g. detection/tracking/hybrid SAM as separate tabs but all map to printed phases 1–2). **Printed Phase 4 (pre-pose pruning) has no Lab tab** — its YAML-tunable subset is master-locked (§8.1).

---

## 6. Phases 1–2 — Detection and tracking (single streaming pass)

### 6.1 Objective

Find people each frame (or every Nth frame), maintain **persistent track IDs**, optionally **refine overlapping boxes** with SAM2 before the tracker consumes detections.

### 6.1.1 Optional bidirectional pass

If `bidirectional_track_pass_enabled()` is true (`SWAY_BIDIRECTIONAL_TRACK_PASS`, implemented in `sway/bidirectional_track_merge.py`), after the first `run_tracking_before_post_stitch` the pipeline reverses the video with ffmpeg, tracks again, then **`merge_forward_backward_tracks`** merges timelines. Hybrid SAM stats are merged with **`_merge_hybrid_sam_stats`**. This roughly doubles Phase 1–2 wall time when enabled.

### 6.2 YOLO detection

#### 6.2.1 Master locked stack (Spot people — Lab + default `main.py`)

These environment values are **fixed for production runs** after `params` YAML is applied (`apply_master_locked_detection_env` in `main.py`, sourced from `MASTER_LOCKED_DETECTION_ENV` in `sway/pipeline_config_schema.py`). The Pipeline Lab **does not expose** controls for them; the Spot people tab shows a read-only **“Lock these down (set and forget)”** info card instead. Lab runs additionally call `freeze_lab_subprocess_detection_env` in `pipeline_lab/server/app.py` so the child process always gets this stack.

| Setting | Env | Locked value | Rationale (summary) |
|---------|-----|--------------|---------------------|
| Crowd / group mode | `SWAY_GROUP_VIDEO` | **on** (`1`) | Stronger small-body detection; with this on, letterbox uses **`max(SWAY_DETECT_SIZE, 960)`** in `tracker.py`. |
| Letterbox baseline | `SWAY_DETECT_SIZE` | **640** | Safe baseline; crowd mode raises effective size as above. |
| Stream chunk | `SWAY_CHUNK_SIZE` | **300** | Memory streaming only; does not change box placement. |
| YOLO `predict` batch (BoxMOT) | `SWAY_YOLO_INFER_BATCH` | **1** | Avoids CUDA OOM; does not improve tracking accuracy. |
| Half precision on CUDA | `SWAY_YOLO_HALF` | **off** (`0`) | Full-precision YOLO on CUDA avoids rare FP16 box numerics. |
| TensorRT engine | `SWAY_YOLO_ENGINE` | **unset** | Use `.pt` weights unless you ship a GPU-matched engine. |
| GSI lengthscale | `SWAY_GSI_LENGTHSCALE` | **0.35** | Default lengthscale whenever GSI interpolation is used (boxes, stitch, pose gaps, export tween). |

**Engineering override:** set `SWAY_UNLOCK_DETECTION_TUNING=1` before `main.py` to skip this lock (used by smoke presets such as `smoke_yolo_infer_batch_2` and `smoke_group_video_off` in `tools/validate_pipeline_e2e.py`).

#### 6.2.2 Weights and letterbox behavior

**Weights resolution** (`resolve_yolo_model_path` / `resolve_yolo_inference_weights` in `tracker.py`):

1. `SWAY_YOLO_ENGINE`: if set, path **must exist** or startup raises (TensorRT).
2. Else `SWAY_YOLO_WEIGHTS` if set: filesystem `.pt`, alias (`yolo26l` → `yolo26l.pt`), or hub string.
3. Else scan `models/`, repo root, cwd for a **priority list** (includes `yolo26l.pt`, `yolo26l_dancetrack.pt`, …).
4. Else Ultralytics hub **`yolo26l.pt`** (unless offline env forbids download — see `SWAY_OFFLINE` / `YOLO_OFFLINE` / `ULTRALYTICS_OFFLINE` handling in `tracker.py`).

| Parameter | Env / constant | Default | Meaning |
|-----------|----------------|---------|---------|
| Confidence | `SWAY_YOLO_CONF` | **0.22** | Min detection score |
| Square input size | `SWAY_DETECT_SIZE` | **640** | Letterbox baseline; **locked** in master pipeline (§6.2.1); with `SWAY_GROUP_VIDEO=1`, effective **`max(base, 960)`** |
| Detection stride | `SWAY_YOLO_DETECTION_STRIDE` | **1** | Run detector every Nth frame; gaps filled in Phase 3 |
| Chunk read | `SWAY_CHUNK_SIZE` | **300** | Frames per streaming chunk (**locked** §6.2.1) |
| Pre-track classical NMS IoU | `SWAY_PRETRACK_NMS_IOU` | **0.50** | After optional DIoU pass |
| DIoU-NMS 0.7 | internal | — | **Skipped** when `_yolo26_series_weights(model_path)` is true (classical NMS only) |
| YOLO batch (BoxMOT path) | `SWAY_YOLO_INFER_BATCH` | **1** | GPU batching for `predict` (**locked** §6.2.1) |
| FP16 on CUDA | `SWAY_YOLO_HALF` | **false** | Half precision when CUDA (**locked off** §6.2.1) |
| TensorRT | `SWAY_YOLO_ENGINE` | empty | See weights list below (**cleared** in master pipeline §6.2.1) |
| Video FPS | *(file metadata)* | **from source** | Read with OpenCV `CAP_PROP_FPS` (`probe_video_fps` / `_iter_video_chunks` in `tracker.py`). Logged at Phase 1–2 start. BoxMOT **ByteTrack** uses `max(1, round(fps))` as `frame_rate`; Deep OC-SORT / OC-SORT / StrongSORT do not use this buffer setting. |

**Box interpolation** between stride anchors: `SWAY_BOX_INTERP_MODE` = `linear` (default) or `gsi` (still configurable in Lab). GSI lengthscale `SWAY_GSI_LENGTHSCALE` is **0.35** in the master locked stack (§6.2.1).

### 6.3 Tracker backends

**Pipeline Lab (Config UI):** `tracker_technology` is either **`deep_ocsort`** (default) or **`deep_ocsort_osnet`**. Both use **BoxMOT** with **`SWAY_BOXMOT_TRACKER=deepocsort`** and **`SWAY_USE_BOXMOT=1`**. The OSNet option sets **`SWAY_BOXMOT_REID_ON=1`** and maps **`sway_boxmot_reid_model`** → **`SWAY_BOXMOT_REID_WEIGHTS`** when the preset file exists (`pipeline_lab/server/app.py`, `_normalize_tracker_technology`). Legacy saved values (`BoxMOT`, `ByteTrack`, `BoT-SORT`, …) are normalized to one of the two presets.

**Environment (CLI / YAML / power users):**

- **`SWAY_USE_BOXMOT` unset or truthy** → **BoxMOT path** (YOLO `predict` + pre-track NMS + hybrid SAM when enabled). **`SWAY_BOXMOT_TRACKER`** selects the BoxMOT class (default **`deepocsort`**). Additional kinds (`bytetrack`, `ocsort`, `strongsort`) remain implemented in `tracker.py` for scripting and experiments, but are **not** exposed in the Lab UI.

- **`SWAY_USE_BOXMOT=0`** → Ultralytics **`model.track`** **BoT-SORT** with `SWAY_TRACKER_YAML` or `config/botsort.yaml` (no hybrid SAM). Lab runs do not select this; set env manually if needed.

**Deep OC-SORT defaults** (when `SWAY_BOXMOT_TRACKER=deepocsort`), from `_deepocsort_extra_from_env` + `_create_boxmot_tracker`:

| Parameter | Default |
|-----------|---------|
| `max_age` | from `SWAY_BOXMOT_MAX_AGE` / **150** |
| `min_hits` | 2 |
| `iou_threshold` | from `SWAY_BOXMOT_MATCH_THRESH` / **0.3** |
| `embedding_off` | **True** unless `SWAY_BOXMOT_REID_ON` |
| `det_thresh` | matches YOLO conf |
| `half` | True on CUDA only |
| `device` | CUDA else CPU |

**Track-time Re-ID:** when `SWAY_BOXMOT_REID_ON` is on, weights come from `SWAY_BOXMOT_REID_WEIGHTS` if set, else `models/osnet_x0_25_msmt17.pt` / BoxMOT cache (`tracker.py`).

### 6.4 Hybrid SAM overlap refiner (default **on**, BoxMOT path only)

When **pairwise IoU** exceeds a trigger, SAM2 tightens masks/boxes **before** `tracker.update`. **Skipped** when `SWAY_USE_BOXMOT=0` (Ultralytics track path).

#### 6.4.1 Master locked stack (Overlap tab — Lab + default `main.py`)

These values are **fixed** after `params` YAML (`apply_master_locked_hybrid_sam_env` in `main.py`, from `MASTER_LOCKED_HYBRID_SAM_ENV` in `sway/pipeline_config_schema.py`). The Lab shows a read-only info card instead of controls. Lab subprocesses also call `freeze_lab_subprocess_hybrid_sam_env` in `pipeline_lab/server/app.py`.

| Setting | Env | Locked value |
|---------|-----|--------------|
| Overlap refinement | `SWAY_HYBRID_SAM_OVERLAP` | **on** (`1`) |
| Union ROI crop (not full frame) | `SWAY_HYBRID_SAM_ROI_CROP` | **on** (`1`) |
| ROI pad fraction | `SWAY_HYBRID_SAM_ROI_PAD_FRAC` | **0.1** |

**Engineering override:** `SWAY_UNLOCK_HYBRID_SAM_TUNING=1` skips this lock (smoke presets in `tools/validate_pipeline_e2e.py` disable SAM for speed).

| Setting | Env | Default |
|---------|-----|---------|
| Enable | `SWAY_HYBRID_SAM_OVERLAP` | on (`1`) — **locked** §6.4.1 |
| IoU trigger | `SWAY_HYBRID_SAM_IOU_TRIGGER` | **0.42** |
| Min detections | `SWAY_HYBRID_SAM_MIN_DETS` | **2** |
| Weights | `SWAY_HYBRID_SAM_WEIGHTS` | `sam2.1_b.pt` (resolved under `models/`) |
| Mask threshold | `SWAY_HYBRID_SAM_MASK_THRESH` | **0.5** |
| Bbox pad | `SWAY_HYBRID_SAM_BBOX_PAD` | **2** px |
| ROI union crop | `SWAY_HYBRID_SAM_ROI_CROP` | **on** — **locked** §6.4.1 |
| ROI pad fraction | `SWAY_HYBRID_SAM_ROI_PAD_FRAC` | **0.10** — **locked** §6.4.1 |

**Weak cues** (optional): `SWAY_HYBRID_SAM_WEAK_CUES` default off; when on, uses `SWAY_HYBRID_WEAK_CONF_DELTA`, `SWAY_HYBRID_WEAK_HEIGHT_FRAC`, `SWAY_HYBRID_WEAK_MATCH_IOU`.

**Contract:** SAM does not assign track IDs; row order matches dets passed to SAM; tracker may reorder rows — masks are reassigned by greedy IoU (`hybrid_sam_refiner.py` module doc).

---

## 7. Phase 3 — Post-track stitching

**Function:** `apply_post_track_stitching(raw_pre, total_frames, ystride=...)`.

### 7.0.1 Master locked stack (Merge IDs tab — Lab + default `main.py`)

After `params` YAML, **`apply_master_locked_phase3_stitch_env()`** (`main.py`, from `MASTER_LOCKED_PHASE3_STITCH_ENV` in `sway/pipeline_config_schema.py`) fixes long-range merge and AFLink internal thresholds. The Lab shows a read-only **“Lock these down”** card; **`sway_global_aflink_mode`** (neural vs. heuristic linker) stays configurable.

| Env | Locked value |
|-----|----------------|
| `SWAY_GLOBAL_LINK` | **on** (`1`) |
| `SWAY_AFLINK_THR_T0` | **0** |
| `SWAY_AFLINK_THR_T1` | **30** |
| `SWAY_AFLINK_THR_S` | **75** |
| `SWAY_AFLINK_THR_P` | **0.05** |

**Engineering override:** `SWAY_UNLOCK_PHASE3_STITCH_TUNING=1` (e.g. `smoke_global_link_off` in `tools/validate_pipeline_e2e.py`).

### 7.1 Order of operations (as coded)

Inside `apply_post_track_stitching` (`tracker.py`):

1. Load runtime stitch/coalescence parameters via `load_tracking_runtime()`.
2. **`_apply_dormant_and_global`**: `apply_dormant_merges` then, if global link on, **`maybe_global_stitch`**:
   - **Neural AFLink** when weights exist at `models/AFLink_epoch20.pth` or `SWAY_AFLINK_WEIGHTS` (see `global_track_link.py`).
   - Else **heuristic** global stitch.
   - `SWAY_GLOBAL_AFLINK=0` (or `heuristic`) forces heuristic even if weights exist.
   - Lab: `sway_global_aflink_mode` → `force_heuristic` sets `SWAY_GLOBAL_AFLINK=0`; `neural_if_available` leaves `SWAY_GLOBAL_AFLINK` unset.
3. **`stitch_fragmented_tracks`** — defaults from env or module constants: max gap **60**; radius **0.5×** bbox height; predicted radius frac **0.75**; pixel fallback **120**; short gap **20** frames.
4. **`coalescence_deduplicate`** — IoU **0.70**, **8** consecutive frames (env-overridable).
5. **`merge_complementary_tracks`**, **`merge_coexisting_fragments`**.
6. **`_fill_stride_gaps`** for YOLO stride gaps (same box interp modes as §6.2).

### 7.2 Optional GNN hook (after Phase 3 stitching, before Phase 4)

When **`SWAY_GNN_TRACK_REFINE`** is truthy (`1`/`true`/`yes` per `gnn_track_refine_enabled()`), **`maybe_gnn_refine_raw_tracks`** runs **after** `apply_post_track_stitching` returns and **before** `track_stats.json` export and phase previews. It is an **identity pass** with a console log (`experimental_hooks.py`).

**Timing note:** `dt_stitch` printed for Phase 3 does **not** include GNN wall time; GNN runs immediately after the stitch timer stops.

---

## 8. Phase 4 — Pre-pose pruning

**Goal:** Remove tracks that are unlikely to be real dancers **before** expensive pose inference.

**Order in `main.py`** (each step may log to `prune_log_entries`):

1. **`prune_tracks`** — duration + kinetic motion: `min_duration_ratio` default **0.20**; **`KINETIC_STD_FRAC`** in `track_pruning.py` = **0.02**
2. **Stage polygon** — `SWAY_STAGE_POLYGON` JSON (normalized vertices) if set; else if `SWAY_AUTO_STAGE_DEPTH` is **`1`** (env only; not exposed in Pipeline Lab), **`estimate_stage_polygon`** from first frame (`depth_stage.py`), then **`prune_by_stage_polygon`**.
3. **`prune_spatial_outliers`** — `SPATIAL_OUTLIER_STD_FACTOR` default **2.0**; min spread `SPATIAL_MIN_SPREAD_FRAC` **0.05**
4. **`prune_short_tracks`** — `SHORT_TRACK_MIN_FRAC` default **0.15**
5. **`prune_audience_region`** — `AUDIENCE_REGION_X_MIN_FRAC` **0.75**, `AUDIENCE_REGION_Y_MIN_FRAC` **0.70**, window `AUDIENCE_REGION_WINDOW_FRAMES` **120**
6. **`prune_late_entrant_short_span`** — `LATE_ENTRANT_START_FRAC` **0.35**, `LATE_ENTRANT_MAX_SPAN_FRAC` **0.17**
7. **`prune_bbox_size_outliers`** — `BBOX_SIZE_MIN_FRAC` **0.40**, `BBOX_SIZE_MAX_FRAC` **2.00**
8. **`prune_bad_aspect_ratio`** — `ASPECT_RATIO_MAX` **2.5**
9. **`prune_geometric_mirrors`** — **`main.py` calls this with no extra kwargs**, so thresholds are **`EDGE_MARGIN_FRAC` (0.15)** and **`EDGE_PRESENCE_FRAC` (0.3)** from `track_pruning.py` only. The prune log’s `cause_config` may still list `EDGE_MARGIN_FRAC` / `EDGE_PRESENCE_FRAC` from **`params`** if present — those YAML values do **not** change the filter unless code is updated to pass them through.

### 8.1 Master-locked pre-pose YAML (no Pipeline Lab stage)

**Pipeline Lab** does not show an “Early cuts” / `pre_pose_prune` flowchart step or parameter page. Phase 4 still runs in `main.py`.

After loading `params` from `--params` or from Lab `params.yaml`, **`apply_master_locked_pre_pose_prune_params`** (`sway/pipeline_config_schema.py`) **overwrites** the following keys with fixed values **unless** `SWAY_UNLOCK_PRE_POSE_PRUNE_TUNING=1` / `true` / `yes`:

| YAML key | Locked value |
|----------|----------------|
| `min_duration_ratio` | **0.20** |
| `KINETIC_STD_FRAC` | **0.02** |
| `SPATIAL_OUTLIER_STD_FACTOR` | **2.0** |
| `SHORT_TRACK_MIN_FRAC` | **0.15** |
| `AUDIENCE_REGION_X_MIN_FRAC` | **0.75** |
| `AUDIENCE_REGION_Y_MIN_FRAC` | **0.70** |

**Call sites:** `main.py` (immediately after YAML load, before `_apply_params_to_env`) and **`_build_params_yaml`** in `pipeline_lab/server/app.py` (every queued run). The same sites then call **`apply_master_locked_reid_dedup_params`** (§11.0.1) unless `SWAY_UNLOCK_REID_DEDUP_TUNING=1`, then **`apply_master_locked_post_pose_prune_params`** (§13.0.1) unless `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING=1`, then **`apply_master_locked_smooth_params`** (§14.0.1) unless `SWAY_UNLOCK_SMOOTH_TUNING=1`.

**Not in this lock** (unchanged from §8 list above): stage polygon (`SWAY_STAGE_POLYGON`, `SWAY_AUTO_STAGE_DEPTH`), bbox size / aspect / mirror / late-entrant / spatial min-spread constants where `main.py` does not read YAML — see `track_pruning.py` and the ordered steps in this section.

Then **`raw_tracks_to_per_frame`** → **`tracking_results`**.

---

## 9. Phase 5 — Pose (ViTPose+), visibility, temporal refine

### 9.0.1 Master locked stack (Body pose tab — Lab + default `main.py`)

After `params` YAML, **`apply_master_locked_pose_env()`** (`main.py`) runs unless `SWAY_UNLOCK_POSE_TUNING=1`:

| Effect | Env | Locked behavior |
|--------|-----|------------------|
| ViTPose batch cap | `SWAY_VITPOSE_MAX_PER_FORWARD` | **Unset** (treat as **0** = all people in one forward per frame; see `vitpose_max_per_forward()` in `pose_estimator.py`). |
| ViTPose precision | `SWAY_VITPOSE_FP32` | **off** (`0`) — FP16 on GPU when supported. |
| 3D lift | `SWAY_3D_LIFT` | **on** (`1`) — MotionAGFormer / viewer path unless `--no-pose-3d-lift` still disables at CLI. |

**Skeleton estimation cadence (master):** Pipeline Lab always runs **2D pose every frame** — `pose_stride` is fixed at **`1`** for Lab-launched runs (the stride control is **not** shown in the Lab UI; schema default + subprocess args stay aligned). For batch scripts, matrices, or local experiments you can still pass **`--pose-stride 2`**; gap fill remains **§9.3**.

**Engineering override:** `SWAY_UNLOCK_POSE_TUNING=1` for smoke presets (`tools/validate_pipeline_e2e.py` sets 3D off and optional chunked ViTPose).

### 9.1 Model selection

- **`SWAY_VITPOSE_MODEL`** overrides Hugging Face id when using the ViTPose path.
- **`--pose-model`**:
  - `base` / `large` / `huge` → ViTPose+ checkpoints (`usyd-community/vitpose-plus-*`).
  - `rtmpose` → **`RTMPoseEstimator`** (`sway/rtmpose_estimator.py`); requires **MMPose** stack (`requirements-rtmpose.txt`).
  - `sapiens` → **ViTPose-Base** weights for 2D keypoints **with explicit logs** that native Meta Sapiens is not bundled.
- Default pose backend for non-`rtmpose`: **`PoseEstimator(device=get_device())`** — MPS or CPU per §2.1.

### 9.2 Per-frame pipeline (conceptual)

- Stream frames; **`extract_embeddings`** with `method="hsv_strip"` when boxes exist (feeds Phase 6).
- Pose when `(frame_idx % pose_stride == 0)` and boxes exist.
- **`compute_visibility_scores`**: skip pose when visibility is below **`POSE_VISIBILITY_THRESHOLD`** (default **0.30** from `params` in `main.py`).
- Dynamic crop padding in `main.py`: **0.15** baseline; **0.25** on large motion/size change; **0.10** when small (see Phase 5 loop).
- Optional **mask-gated** crops when segmentation masks exist.

### 9.3 Pose stride gap fill

If `--pose-stride 2`, **`_interpolate_pose_gaps`** fills skipped frames using `blend_pose_keypoints_scores`:

- Mode: `SWAY_POSE_GAP_INTERP_MODE` — `linear` (default) or `gsi`
- Lengthscale: `SWAY_POSE_GSI_LENGTHSCALE` or fallback `SWAY_GSI_LENGTHSCALE` / default **0.35**

### 9.4 Temporal keypoint smoothing (≠ Phase 9 1-Euro)

- **Default on** via CLI; `SWAY_TEMPORAL_POSE_REFINE` overrides (`want_temporal_pose_refine`).
- **`apply_temporal_keypoint_smoothing`** in `temporal_pose_refine.py` — confidence-weighted blend across ±radius frames.
- Radius: `--temporal-pose-radius` or `SWAY_TEMPORAL_POSE_RADIUS`, **clamped 0–8**, default **2**; production reapplies **2** via env (§14.0.1) unless `SWAY_UNLOCK_SMOOTH_TUNING=1`.

### 9.5 Pose logging throttles

`SWAY_POSE_LOG_EVERY_SEC` (default 20), `SWAY_POSE_LOG_EVERY_N_PASSES` (8), `SWAY_POSE_SLOW_FORWARD_SEC` (4).

---

## 10. Substep: 3D lift (after Phase 9, before Phase 10)

**Not printed as `[n/11]`**, but runs here when enabled:

- **Enable:** `SWAY_3D_LIFT` **on** by default in the master locked stack (§9.0.1); `--no-pose-3d-lift` disables regardless.
- **Input:** **`all_frame_data`** after **1-Euro smoothing**.
- **Backend:** `SWAY_LIFT_BACKEND` default `motionagformer`; optional `poseformerv2` / aliases per `lift_backend()` in `pose_lift_3d.py`.
- **Depth:** `SWAY_DEPTH_DYNAMIC` default on; strided Depth Anything V2 via `collect_strided_depth_series`; stride from `SWAY_DEPTH_STRIDE_FRAMES` or default `max(1, round(output_fps))` in `main.py`. AugLift-style z blend: `SWAY_AUGLIFT_BLEND` default **0.3** (function default in `pose_lift_3d.py`).
- **Export:** `export_3d_for_viewer` → `pose_3d` blob in `data.json`; unified world coords when `SWAY_UNIFIED_3D_EXPORT` on (default in `unified_export_enabled()`).
- **Errors:** wrapped in `try`/`except` in `main.py`; failures print `[3D Lift] Skipped: …` and continue without `pose_3d`.

Further env knobs (weights paths, vendor roots, Savitzky–Golay on lift export, etc.) are documented in **`sway/pose_lift_3d.py`**.

---

## 11. Phase 6 — Association

### 11.0.1 Master locked re-ID + collocated-dedup gates (Phases 6–7 — Lab + default `main.py`)

After `apply_master_locked_pre_pose_prune_params`, **`apply_master_locked_reid_dedup_params`** (`sway/pipeline_config_schema.py`) **overwrites** the following keys in `params` **unless** `SWAY_UNLOCK_REID_DEDUP_TUNING=1` / `true` / `yes`:

| YAML key | Locked value |
|----------|----------------|
| `REID_MAX_FRAME_GAP` | **90** |
| `REID_MIN_OKS` | **0.35** |
| `COLLISION_KPT_DIST_FRAC` | **0.26** |
| `COLLISION_CENTER_DIST_FRAC` | **0.5** |
| `DEDUP_TORSO_MEDIAN_FRAC` | **0.24** |

**Call sites:** `main.py` (after Phase 4 lock, before `_apply_params_to_env`) and **`_build_params_yaml`** in `pipeline_lab/server/app.py`.

**Not in this lock** (still tunable via Lab / YAML): `DEDUP_MIN_PAIR_OKS`, `DEDUP_ANTIPARTNER_MIN_IOU`, plus module constants such as `DEDUP_KPT_TIGHT_FRAC` and collision state-machine IoU thresholds in `crossover.py`.

**Input:** `all_frame_data_pre` (mutated in place).

1. **`apply_occlusion_reid`**
   - Baseline thresholds are the locked values above (also the defaults in `crossover.py`). Override only with `SWAY_UNLOCK_REID_DEDUP_TUNING=1` and explicit YAML keys.
   - Debug: `SWAY_REID_DEBUG`
2. **`apply_crossover_refinement`** (frame dimensions passed)
3. **`apply_acceleration_audit`**

---

## 12. Phase 7 — Collision cleanup

Per frame on the post-association structure:

1. **`deduplicate_collocated_poses`** — defaults include:
   - `COLLISION_KPT_DIST_FRAC` **0.26**, `COLLISION_CENTER_DIST_FRAC` **0.5**, `DEDUP_TORSO_MEDIAN_FRAC` **0.24** — **master-locked** in `params` (§11.0.1) unless `SWAY_UNLOCK_REID_DEDUP_TUNING=1`
   - `DEDUP_ANTIPARTNER_MIN_IOU` **0.12**
   - `DEDUP_KPT_TIGHT_FRAC` **0.20**
   - `DEDUP_MIN_PAIR_OKS` **0.68**
   - Collision state machine: `COLLISION_ENTRY_IOU` **0.6** (3 consecutive frames), exit **0.3**, etc. (`crossover.py`)
2. **`sanitize_pose_bbox_consistency`** — keypoints outside bbox; logs at INFO under `sway.crossover` with `[collision]` prefix (`main._ensure_collision_cleanup_logging`).

Telemetry can append to **`prune_log_entries`**.

---

## 13. Phase 8 — Post-pose pruning (Tier C, Tier A, Tier B)

### 13.0.1 Master locked Tier C / Tier A span / garbage thresholds / three vote weights

After **`apply_master_locked_reid_dedup_params`**, **`apply_master_locked_post_pose_prune_params`** (`sway/pipeline_config_schema.py`) runs **unless** `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING=1` / `true` / `yes`:

| YAML key | Locked value |
|----------|----------------|
| `CONFIRMED_HUMAN_MIN_SPAN_FRAC` | **0.10** |
| `TIER_C_SKELETON_MEAN` | **0.15** |
| `TIER_C_LOW_FRAME_FRAC` | **0.80** |
| `MEAN_CONFIDENCE_MIN` | **0.45** |
| `EDGE_MARGIN_FRAC` | **0.15** |
| `EDGE_PRESENCE_FRAC` | **0.30** |
| `min_lower_body_conf` | **0.30** |
| `JITTER_RATIO_MAX` | **0.10** |

It also **merges** into `params["PRUNING_WEIGHTS"]`: `prune_completeness_audit` **0.6**, `prune_head_only_tracks` **0.8**, `prune_jittery_tracks` **0.5** (other weight keys from YAML are preserved when present).

**Call sites:** `main.py` (after re-ID lock, before `_apply_params_to_env`) and **`_build_params_yaml`** in `pipeline_lab/server/app.py`.

**Not in this lock** (Lab / YAML tunable): `SYNC_SCORE_MIN`, `PRUNE_THRESHOLD`, `PRUNING_WEIGHTS` entries `prune_low_sync_tracks`, `prune_smart_mirrors`, `prune_low_confidence_tracks`.

### 13.1 Tier A — Confirmed humans (exempt from Tier B vote)

**`compute_confirmed_human_set`** (`track_pruning.py`); **`main.py`** passes `min_span_frac` from YAML **`CONFIRMED_HUMAN_MIN_SPAN_FRAC`** (default **0.10**; **locked** §13.0.1 unless `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING=1`):

| Criterion | Default |
|-----------|---------|
| Mean torso keypoint confidence (shoulders/hips COCO 5,6,11,12) | ≥ **0.5** on a frame |
| Fraction of pose frames meeting above | ≥ **40%** |
| Temporal span | ≥ **0.10** of video (`min_span_frac`) |
| Edge/mirror sanity | If `frame_width > 0`: edge presence ≤ **20%** of bbox frames, margin from **`EDGE_MARGIN_FRAC`** passed from **`main.py`** (default **0.15**; **locked** with **`EDGE_PRESENCE_FRAC`** §13.0.1) |

### 13.2 Tier C — Ultra-low skeleton

**`prune_ultra_low_skeleton_tracks`**:

- `TIER_C_SKELETON_MEAN` default **0.15** (module `ULTRA_LOW_SKELETON_MEAN`; **locked** §13.0.1)
- `TIER_C_LOW_FRAME_FRAC` default **0.80** (`ULTRA_LOW_SKELETON_FRAME_FRAC`; **locked** §13.0.1)

### 13.3 Tier B — Weighted vote

**`compute_phase7_voting_prune_set`** uses **`PRUNING_WEIGHTS`** and **`PRUNE_THRESHOLD`** (default **0.65**).

Default weights (`track_pruning.py`); three are **remerged from the master lock** (§13.0.1) unless `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING=1`:

| Rule | Weight | Notes |
|------|--------|--------|
| `prune_low_sync_tracks` | 0.7 | Tunable via Lab |
| `prune_smart_mirrors` | 0.9 | Tunable via Lab |
| `prune_completeness_audit` | 0.6 | **Locked** §13.0.1 |
| `prune_head_only_tracks` | 0.8 | **Locked** §13.0.1 |
| `prune_low_confidence_tracks` | 0.5 | Tunable via Lab |
| `prune_jittery_tracks` | 0.5 | **Locked** §13.0.1 |

**Supporting thresholds** (from `main.py` `params` defaults when key absent; first six **locked** §13.0.1 unless unlock):

| Key | Default |
|-----|---------|
| `SYNC_SCORE_MIN` | 0.10 |
| `EDGE_MARGIN_FRAC` | 0.15 |
| `EDGE_PRESENCE_FRAC` | 0.3 |
| `min_lower_body_conf` | 0.30 |
| `MEAN_CONFIDENCE_MIN` | 0.45 |
| `JITTER_RATIO_MAX` | 0.10 |

Confirmed humans are **skipped** for Tier B rules via `_phase7_should_skip_confirmed`.

---

## 14. Phase 9 — Temporal smoothing (1-Euro)

### 14.0.1 Master locked 1-Euro cutoff + neighbor-blend radius

After **`apply_master_locked_post_pose_prune_params`**, **`apply_master_locked_smooth_params`** sets **`SMOOTHER_MIN_CUTOFF`** to **1.0** in `params` **unless** `SWAY_UNLOCK_SMOOTH_TUNING=1` / `true` / `yes`.

After **`apply_master_locked_pose_env()`**, **`apply_master_locked_smooth_env()`** sets **`SWAY_TEMPORAL_POSE_RADIUS`** to **2** on `os.environ` **unless** the same unlock (drives `temporal_pose_radius()` during Phase 5 neighbor blend).

**Call sites:** `main.py` (params lock with other YAML locks; env lock after pose env) and **`_build_params_yaml`** / **`_subprocess_env`** in `pipeline_lab/server/app.py`.

**Not in this lock:** `SMOOTHER_BETA`; neighbor blend on/off remains CLI / `SWAY_TEMPORAL_POSE_REFINE` / Lab **`temporal_pose_refine`**.

**`PoseSmoother`** per joint:

| Param | YAML key | Default |
|-------|----------|---------|
| Min cutoff | `SMOOTHER_MIN_CUTOFF` | **1.0** (**locked** §14.0.1 unless unlock) |
| Beta | `SMOOTHER_BETA` | **0.7** |

**Distinct from** Phase 5 temporal keypoint refine (neighbor blending). Output stream: **`all_frame_data`** with smoothed `poses`.

---

## 15. Phase 10 — Scoring

**`process_all_frames_scoring_vectorized(all_frame_data)`** returns `None` only if `all_frame_data` is empty. When non-`None`, `main.py` attaches per frame:

- `track_angles`, `consensus_angles`, `deviations`
- `shape_errors`, `timing_errors`

Degenerate slices / NaNs — see `scoring.py` and ripple handling (`RIPPLE_STD_THRESHOLD`).

---

## 16. Phase 11 — Export and visualization

### 16.1 Outputs (typical)

| Artifact | Description |
|----------|-------------|
| `data.json` | Metadata, track summaries, per-frame data; may include `pruned_overlay`, `dropped_pose_overlay`, `prune_entries`, `pose_3d` |
| `{stem}_poses.mp4` | Full visualization |
| `{stem}_track_ids.mp4` | Boxes + IDs |
| `{stem}_skeleton.mp4` | Skeleton style |
| `{stem}_sam_style.mp4` | SAM-style for hybrid-refined dets |
| `{stem}_3d.mp4` | When 3D blob is present |
| `prune_log.json` | Prune audit trail |
| `montage.mp4` | If `--montage` |
| `track_stats.json` | Always after Phase 3 (small stitched-track summary) |
| `hmr_mesh_sidecar.json` | If `SWAY_HMR_MESH_SIDECAR=1` after export (placeholder schema; Lab: `sway_hmr_mesh_sidecar`) |

Video codec commonly **mp4v**; **native_fps** from tracking; audio mux when available. Overlay temporal tweening: **`SWAY_VIS_TEMPORAL_INTERP_MODE`** (`linear`|`gsi`); optional lengthscale **`SWAY_VIS_GSI_LENGTHSCALE`** (else shared GSI lengthscale — see `_build_pipeline_diagnostics` in `main.py`).

**Run manifest (`run_manifest.json`, Lab):** when `--progress-jsonl` or `--save-phase-previews` triggers a manifest path, JSON includes **`run_context_final`**, which holds **`pipeline_diagnostics`** from `_build_pipeline_diagnostics` (`tracker_path`, `run_quality`, `experimental` flags, hybrid SAM / interpolation / AFLink summaries, `bidirectional_track_pass`, etc.).

### 16.2 Optional previews

`--save-phase-previews` writes labeled phase clips under `phase_previews/` for debugging and Lab registration.

---

## 17. GSI (Gaussian-smoothed interpolation) — cross-cutting

Used when mode is `gsi` (not default linear):

- Box stride gaps: `SWAY_BOX_INTERP_MODE`, `SWAY_GSI_LENGTHSCALE`
- Pose stride gaps: `SWAY_POSE_GAP_INTERP_MODE`, `SWAY_POSE_GSI_LENGTHSCALE` or shared lengthscale
- Export video tween: `SWAY_VIS_TEMPORAL_INTERP_MODE`, `SWAY_VIS_GSI_LENGTHSCALE` (optional)

---

## 18. Review checklist (common mistakes)

1. **Two devices** — ViTPose MPS/CPU via `get_device()` vs BoxMOT CUDA/CPU in `tracker.py`.
2. **Stack defaults** — `SWAY_GROUP_VIDEO`, `SWAY_GLOBAL_LINK`, `SWAY_AUTO_STAGE_DEPTH` applied unless pre-set; `SWAY_GLOBAL_LINK` and AFLink thresholds are then **re-locked** after params (§7.0.1) unless `SWAY_UNLOCK_PHASE3_STITCH_TUNING=1`.
3. **YOLO weights** — on-disk priority may beat hub; offline env affects downloads.
4. **Kinetic fraction** — code **`KINETIC_STD_FRAC = 0.02`**, not 0.03.
5. **Hybrid SAM** — BoxMOT `predict` path only; `SWAY_USE_BOXMOT=0` (Ultralytics track) skips it. Overlap on, union ROI crop, and ROI pad are **master-locked** in `main.py` (§6.4.1) unless `SWAY_UNLOCK_HYBRID_SAM_TUNING=1`.
6. **3D timing** — runs **after** 1-Euro, **before** scoring.
7. **TensorRT** — `SWAY_YOLO_ENGINE` must point to a real file or startup errors; unset to use `.pt`.
8. **Geometric mirror prune (Phase 4)** — uses module constants only; YAML `EDGE_*` in prune logs for that rule may not match the actual threshold unless `main.py` is changed to pass them into `prune_geometric_mirrors`.
9. **Bidirectional pass** — roughly doubles Phase 1–2 time when `SWAY_BIDIRECTIONAL_TRACK_PASS` is on.

---

## 19. Pipeline Lab / React UI — configuration map

The **Config** page and **Run** editor load **`GET /api/schema`** from `pipeline_lab/server/app.py` (payload assembled from **`sway/pipeline_config_schema.py`**).

- **Stages** (`PIPELINE_STAGES`): left-to-right flowchart tabs; field `phase` must match a stage `id` so controls appear under the right step (e.g. **tracking** → `tracker_technology`, `sway_gnn_track_refine`; **pose** → `pose_model` including Sapiens slot behavior; **export** → `sway_hmr_mesh_sidecar`, montage, GSI video blend).
- **Bindings:** `env` → copied into the subprocess environment; `yaml` → merged into `params` for `main.py`; `cli` → extra `main.py` flags (`pose_model`, `pose_stride`, previews, temporal refine); `none` + special cases: **`tracker_technology`** and **`sway_global_aflink_mode`** are interpreted only in **`_subprocess_env`** (not as `params.yaml` keys by themselves).
- **Tier 1** fields appear on the primary run strip; **tier 2/3** and **advanced** appear per-phase (Run editor) or in the full grid (Config page).
- **Watch** page: “Recipe tuning” chips use **`watchPhaseTuning.ts`** field id lists keyed by preview clip phase — keep in sync when adding high-signal knobs for Phases 1–3 / 5 / 6–7 / 8 (Phase 4 list is empty; pre-pose prune is master-locked §8.1).
- **Results:** `RunMetrics` / `PipelineImpactReport` read **`run.manifest.run_context_final.pipeline_diagnostics`**. MOT metrics **IDF1 / HOTA / IDSW** are **not** computed in-app without GT (UI shows **N/A**).

---

## 20. File index (implementation)

| Area | Files |
|------|--------|
| Orchestration | `main.py` |
| Detection / tracking / stitch | `sway/tracker.py` |
| Bidirectional track merge | `sway/bidirectional_track_merge.py` |
| Hybrid SAM | `sway/hybrid_sam_refiner.py` |
| Global link / AFLink | `sway/global_track_link.py`, `sway/aflink.py` |
| Dormant merges | `sway/dormant_tracks.py` |
| Pruning | `sway/track_pruning.py` |
| Pose | `sway/pose_estimator.py` |
| Temporal pose refine | `sway/temporal_pose_refine.py` |
| 3D lift | `sway/pose_lift_3d.py` |
| Depth / stage | `sway/depth_stage.py` |
| Association / collision | `sway/crossover.py` |
| Re-ID embeddings | `sway/reid_embedder.py` |
| Interpolation helpers | `sway/interp_utils.py` |
| Smoothing | `sway/smoother.py` |
| Scoring | `sway/scoring.py` |
| Export | `sway/visualizer.py` |
| Lab schema | `sway/pipeline_config_schema.py` |
| Lab API / env mapping | `pipeline_lab/server/app.py` |
| Optional in-pipeline hooks | `sway/experimental_hooks.py` |
| RTMPose backend | `sway/rtmpose_estimator.py` |
| Model prefetch | `tools/prefetch_models.py` |
| TensorRT export helper | `tools/export_models.py` |


---

## 21. Complete parameter reference (Lab schema + pipeline effects)

This section lists **every** field in `PIPELINE_PARAM_FIELDS` (`sway/pipeline_config_schema.py`), grouped by Lab **phase** (Config / Run UI tabs). Phase 4 pre-pose pruning is **not** in `PIPELINE_PARAM_FIELDS` — see §21.5 / §8.1. **Binding** meaning: `env` → subprocess `os.environ` (`SWAY_*`); `yaml` → written to run `params` only; `cli` → `main.py` CLI; `none` → display-only `info` or Lab-only logic in `pipeline_lab/server/app.py`; `reid_model_preset` → maps OSNet card to `SWAY_BOXMOT_REID_WEIGHTS`; `yaml_pruning_weight` → one entry inside the `PRUNING_WEIGHTS` dict in `params`.

**Precedence (main.py):** `main._apply_stack_default_env()` first, then load `params` YAML (if `--params`), then **`apply_master_locked_pre_pose_prune_params(params)`** (unless `SWAY_UNLOCK_PRE_POSE_PRUNE_TUNING=1`) — §8.1 — then **`apply_master_locked_reid_dedup_params(params)`** (unless `SWAY_UNLOCK_REID_DEDUP_TUNING=1`) — §11.0.1 — then **`apply_master_locked_post_pose_prune_params(params)`** (unless `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING=1`) — §13.0.1 — then **`apply_master_locked_smooth_params(params)`** (unless `SWAY_UNLOCK_SMOOTH_TUNING=1`) — §14.0.1 — then `_apply_params_to_env(params)`, then **`apply_master_locked_detection_env()`** (unless `SWAY_UNLOCK_DETECTION_TUNING=1`) — §6.2.1, then **`apply_master_locked_hybrid_sam_env()`** (unless `SWAY_UNLOCK_HYBRID_SAM_TUNING=1`) — §6.4.1, then **`apply_master_locked_phase3_stitch_env()`** (unless `SWAY_UNLOCK_PHASE3_STITCH_TUNING=1`) — §7.0.1, then **`apply_master_locked_pose_env()`** (unless `SWAY_UNLOCK_POSE_TUNING=1`) — §9.0.1, then **`apply_master_locked_smooth_env()`** (unless `SWAY_UNLOCK_SMOOTH_TUNING=1`) — §14.0.1. Otherwise: shell environment > `params.yaml` for keys `_apply_params_to_env` copies (`SWAY_*` + offline keys) > stack defaults.

### 21.1 Phase: `detection` — Spot people

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| *(info_detection_yolo)* | none | — | — | The pipeline draws a box around each person it thinks is in the frame. You pick how sharp that finder is (small / large / extra-large model). If a model file is missing, it may download the first time you run. Power users can point to a custom weights file via the environment. |
| *(info_detection_master_locked)* | none | — | — | Read-only: master locked detection stack (§6.2.1): `SWAY_GROUP_VIDEO`, `SWAY_DETECT_SIZE`, `SWAY_CHUNK_SIZE`, `SWAY_YOLO_INFER_BATCH`, `SWAY_YOLO_HALF`, `SWAY_YOLO_ENGINE` (cleared), `SWAY_GSI_LENGTHSCALE`. |
| `sway_yolo_weights` | env | `SWAY_YOLO_WEIGHTS` | yolo26l_dancetrack | Smaller = faster run, may miss tiny or distant people. Larger = slower, usually better on hard clips and finals footage. DanceTrack is a YOLO26l fine-tune for group dance (needs models/yolo26l_dancetrack.pt). The lab dims a card when that weight file is missing. *(choices: yolo26s, yolo26l_dancetrack, yolo26x)* |
| `sway_pretrack_nms_iou` | env | `SWAY_PRETRACK_NMS_IOU` | 0.5 | When two boxes sit on top of each other, one is thrown away before tracking. Lower = merge more (good for tight formations, fewer double-counts). Higher = keep more separate boxes (good if people stand close but are distinct). *(min 0.4 max 0.9)* |
| `sway_yolo_conf` | env | `SWAY_YOLO_CONF` | 0.22 | Higher = fewer boxes, less junk, but you might miss dim or distant dancers. Lower = more boxes, more risk of random objects being treated as people. *(min 0.05 max 0.95)* |
| `sway_yolo_detection_stride` | env | `SWAY_YOLO_DETECTION_STRIDE` | 1 | 1 = every frame (best accuracy). 2+ = skip frames between runs; the tracker fills the gap—faster but can wobble on fast motion. *(min 1 max 8)* |
| `sway_box_interp_mode` | env | `SWAY_BOX_INTERP_MODE` | linear | When YOLO runs every Nth frame, skipped frames get filled boxes before pose. Linear is the long-standing default. GSI uses a light Gaussian (RBF) smoother between anchors (also used when stitching track fragments). Off by default path = linear. *(choices: linear, gsi)* |

### 21.2 Phase: `tracking` — Track IDs

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| *(info_tracking_backends)* | none | — | — | After each frame knows where people are, this step decides “this box is still dancer 3.” Deep OC-SORT is tuned for dance. Pick motion-only matching or add track-time OSNet when outfits differ. Hybrid SAM (next tab) applies on this BoxMOT path. |
| `tracker_technology` | none | Lab → `SWAY_USE_BOXMOT`, `SWAY_BOXMOT_TRACKER`, `SWAY_BOXMOT_REID_ON`, optional `SWAY_BOXMOT_REID_WEIGHTS` in `app.py` | deep_ocsort | **Default:** Deep OC-SORT, motion/IoU only (track-time Re-ID off). **+ OSNet:** same tracker with `SWAY_BOXMOT_REID_ON` and preset weights when `models/osnet_*.pt` exists. *(choices: deep_ocsort, deep_ocsort_osnet)* |
| `sway_bidirectional_track_pass` | env | `SWAY_BIDIRECTIONAL_TRACK_PASS` | False | Off by default. When on, ffmpeg reverses the clip, the pipeline runs Phases 1–2 again, then merges boxes with the forward pass (IoU + minimum matched frames). Roughly doubles tracking time; needs ffmpeg on PATH. |
| `sway_gnn_track_refine` | env | `SWAY_GNN_TRACK_REFINE` | False | Default off. When on, ``main.py`` runs an optional post-stitch hook (identity today — logs once). Reserved for future graph-based ID association inside the pipeline. |
| `sway_boxmot_max_age` | env | `SWAY_BOXMOT_MAX_AGE` | 150 | If a dancer walks behind someone or leaves the frame, the pipeline can keep their ID warm for this many frames. Higher = better when people dip in and out; lower = less chance the wrong person inherits the ID later. Rough guide: 150 frames ≈ 5 seconds at 30 fps. *(min 60 max 300)* |
| `sway_boxmot_match_thresh` | env | `SWAY_BOXMOT_MATCH_THRESH` | 0.3 | Lower = easier to glue boxes together (fewer ID swaps, risk of merging two people). Higher = stricter match (cleaner separation, more risk of a new ID when boxes jump). *(min 0.2 max 0.5)* |
| `sway_boxmot_reid_model` | reid_model_preset | preset → `SWAY_BOXMOT_REID_WEIGHTS` in `app.py` when `tracker_technology` = `deep_ocsort_osnet` | osnet_x0_25 | OSNet checkpoint for track-time Re-ID. *(choices: osnet_x0_25, osnet_x1_0; when `tracker_technology` = `deep_ocsort_osnet`)* |

### 21.3 Phase: `hybrid_sam` — Overlap

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| *(info_hybrid_sam_master_locked)* | none | — | — | Read-only: master locked hybrid SAM stack (§6.4.1): `SWAY_HYBRID_SAM_OVERLAP`, `SWAY_HYBRID_SAM_ROI_CROP`, `SWAY_HYBRID_SAM_ROI_PAD_FRAC`. |
| `sway_hybrid_sam_iou_trigger` | env | `SWAY_HYBRID_SAM_IOU_TRIGGER` | 0.42 | Higher = only very overlapped pairs get fixed (faster, less segmentation). Lower = fix sooner (slower, cleaner lifts and partner work). With ROI crop on, lower values are cheaper than full-frame SAM. *(min 0.25 max 0.65)* |
| `sway_hybrid_sam_weak_cues` | env | `SWAY_HYBRID_SAM_WEAK_CUES` | False | Optional Hybrid-SORT–style gate: after IoU says “maybe run SAM,” compare detections to the previous output. If the worst-overlap pair is height- and confidence-stable, SAM is skipped. |
| `sway_hybrid_weak_conf_delta` | env | `SWAY_HYBRID_WEAK_CONF_DELTA` | 0.08 | Weak cue: max \|Δ confidence\| vs matched previous box *(min 0.02 max 0.35; when `sway_hybrid_sam_weak_cues`=True)* |
| `sway_hybrid_weak_height_frac` | env | `SWAY_HYBRID_WEAK_HEIGHT_FRAC` | 0.12 | Weak cue: max relative height change vs matched previous box *(min 0.03 max 0.45; when `sway_hybrid_sam_weak_cues`=True)* |
| `sway_hybrid_weak_match_iou` | env | `SWAY_HYBRID_WEAK_MATCH_IOU` | 0.25 | Weak cue: min IoU to match a box to the previous frame *(min 0.1 max 0.55; when `sway_hybrid_sam_weak_cues`=True)* |

### 21.4 Phase: `phase3_stitch` — Merge IDs

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| *(info_phase3_stitch)* | none | — | — | Sometimes the tracker splits one dancer into two IDs or loses them for a stretch. This pass tries to glue those pieces back together before body pose runs. When the optional neural linker model is installed, it can make smarter long-range guesses; otherwise simple rules are used. |
| *(info_phase3_master_locked)* | none | — | — | Read-only: `SWAY_GLOBAL_LINK`, `SWAY_AFLINK_THR_T0`/`T1`/`S`/`P` — §7.0.1. |
| `sway_stitch_max_frame_gap` | env | `SWAY_STITCH_MAX_FRAME_GAP` | 60 | If someone disappears briefly, the pipeline can still merge old and new IDs if the gap is under this. Raise for slow exits and re-entries; lower on short clips to avoid merging two different people. About 60 frames ≈ 2 seconds at 30 fps. *(min 30 max 150)* |
| `sway_global_aflink_mode` | none | Lab → `SWAY_GLOBAL_AFLINK` pop/unset in `app.py` | neural_if_available | Use the learned linker when its weights are present (recommended). Or always use simple geometry-based rules if you want predictable, lighter behavior. *(choices: neural_if_available, force_heuristic)* |

### 21.5 Phase 4 — Pre-pose pruning (removed from Lab; master-locked YAML)

There is **no** `pre_pose_prune` row in `PIPELINE_STAGES` and **no** Lab schema fields for Phase 4. The pipeline step, preview clip `02_pre_pose_prune.mp4`, and `main.py` logic are unchanged.

**Locked `params` keys** (table + unlock): **§8.1** (`MASTER_LOCKED_PRE_POSE_PRUNE_PARAMS`, `apply_master_locked_pre_pose_prune_params`, `SWAY_UNLOCK_PRE_POSE_PRUNE_TUNING`).

### 21.6 Phase: `pose` — Body pose

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| *(info_pose_models)* | none | — | — | For each person box, the pipeline estimates where shoulders, hips, knees, etc. sit in the image. That’s what you see as the skeleton overlay. Dim joints can be ignored so random noise doesn’t score as movement. |
| `pose_model` | cli | `pose_model` | ViTPose-Base | ViTPose+: larger = usually better on hard motion, slower. RTMPose-L needs MMPose (see requirements-rtmpose.txt). **Sapiens** slot uses ViTPose-Base keypoints until a native Sapiens backend is added in code. *(choices: ViTPose-Base, ViTPose-Large, ViTPose-Huge, RTMPose-L, Sapiens (ViTPose-Base fallback))* |
| *(info_pose_alternates)* | none | — | — | **Sapiens** is selectable as a pose card (runs ViTPose-Base until native Sapiens is wired). **GNN** refine flag lives under Tracking. **HMR** mesh placeholder JSON is under Export. Regression tests: ``python -m tools.golden_bench`` from a terminal (not a Lab toggle). |
| `pose_stride` | cli | `pose_stride` | **1** | **Master / Lab:** skeleton estimation **every frame**; field is **`lab_hidden`** in `pipeline_config_schema.py` (no Lab UI control). **`--pose-stride 2`** remains valid for CLI, batch matrices, and tools. *(choices: 1, 2)* |
| `sway_pose_gap_interp_mode` | env | `SWAY_POSE_GAP_INTERP_MODE` | linear | Only when **`pose_stride`=2** (not offered in Lab UI). Linear is the long-standing default. GSI uses the same RBF smoother as optional box paths; lengthscale falls back to `SWAY_GSI_LENGTHSCALE` (master stack locks this to **0.35** — §6.2.1), or set `SWAY_POSE_GSI_LENGTHSCALE` in params YAML to override for pose gaps only. *(choices: linear, gsi)* |
| *(info_pose_master_locked)* | none | — | — | Read-only: `SWAY_VITPOSE_MAX_PER_FORWARD` (unset = 0), `SWAY_VITPOSE_FP32`, `SWAY_3D_LIFT` — §9.0.1. |
| `pose_visibility_threshold` | yaml | `POSE_VISIBILITY_THRESHOLD` | 0.3 | Baseline 0.30 matches the pipeline default. Higher = fewer drawn joints in dark or blurry video; lower = show more joints (noisier). *(min 0.0 max 1.0)* |

### 21.7 Phase: `reid_dedup` — Fix IDs

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| *(info_reid_actual)* | none | — | — | Cleans up identity mistakes: when someone hides and reappears, when two dancers cross paths, or when two skeletons sit on the same body. You’ll see the effect in overlays and scores—fewer wild ID jumps and fewer double ghosts. |
| *(info_reid_dedup_master_locked)* | none | — | — | Read-only: `REID_MAX_FRAME_GAP`, `REID_MIN_OKS`, `COLLISION_KPT_DIST_FRAC`, `COLLISION_CENTER_DIST_FRAC`, `DEDUP_TORSO_MEDIAN_FRAC` — §11.0.1. |
| `dedup_min_pair_oks` | yaml | `DEDUP_MIN_PAIR_OKS` | 0.68 | How similar two colliding poses must be before one is considered a phantom duplicate. *(min 0.3 max 0.95)* |
| `dedup_antipartner_min_iou` | yaml | `DEDUP_ANTIPARTNER_MIN_IOU` | 0.12 | If IoU is lower than this, it assumes two separate people touching, bypassing deduplication. *(min 0.0 max 0.5)* |

**Locked `params` keys** (five; read-only in Lab): **§11.0.1** (`MASTER_LOCKED_REID_DEDUP_PARAMS`, `apply_master_locked_reid_dedup_params`, `SWAY_UNLOCK_REID_DEDUP_TUNING`).

### 21.8 Phase: `post_pose_prune` — Skeleton cleanup

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| *(info_post_pose_tiers)* | none | — | — | Now that skeletons exist, the pipeline removes tracks that still look wrong—wobbly noise, mirror reflections, people who never really danced in sync, etc. Trusted “real dancer” tracks get extra protection. The sliders below decide how hard each red-flag rule pulls toward “delete this track.” |
| *(info_post_pose_master_locked)* | none | — | — | Read-only: Tier C, Tier A span, edge band, mushy-skeleton + jitter thresholds, and `PRUNING_WEIGHTS` keys `prune_completeness_audit`, `prune_head_only_tracks`, `prune_jittery_tracks` — §13.0.1. |
| `sync_score_min` | yaml | `SYNC_SCORE_MIN` | 0.1 | Baseline 0.10 matches the pipeline default. Raise if you want stricter “must look like the group”; lower if soloists get unfairly cut. *(min 0.0 max 1.0)* |
| `prune_threshold` | yaml | `PRUNE_THRESHOLD` | 0.65 | Many small checks add up to a score. Higher = easier to delete a track when something looks off. Lower = keep more people even when a few checks complain (more junk may remain). *(min 0.4 max 0.9)* |
| `pruning_w_low_sync` | yaml_pruning_weight | `PRUNING_WEIGHTS['prune_low_sync_tracks']` in params | 0.7 | Raise if random audience members keep scoring; lower if real soloists get cut. *(min 0.0 max 1.0)* |
| `pruning_w_smart_mirror` | yaml_pruning_weight | `PRUNING_WEIGHTS['prune_smart_mirrors']` in params | 0.9 | High for wall-to-wall mirrors; lower if you film intentional mirrored choreography. *(min 0.0 max 1.0)* |
| `pruning_w_low_conf` | yaml_pruning_weight | `PRUNING_WEIGHTS['prune_low_confidence_tracks']` in params | 0.5 | Blurry or dark footage raises this if mushy skeletons should be dropped faster. *(min 0.0 max 1.0)* |

**Locked `params` keys / merged `PRUNING_WEIGHTS` entries:** **§13.0.1** (`MASTER_LOCKED_POST_POSE_PRUNE_*`, `apply_master_locked_post_pose_prune_params`, `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING`).

### 21.9 Phase: `smooth` — Smoothing

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| *(info_smooth_one_euro)* | none | — | — | Raw joints can wiggle frame to frame. This pass steadies them so overlays and scores look human without killing real sharp hits. Optional neighbor blending further softens noise using a few frames on either side. |
| *(info_smooth_master_locked)* | none | — | — | Read-only: `SMOOTHER_MIN_CUTOFF` **1.0**, `SWAY_TEMPORAL_POSE_RADIUS` **2** (neighbor blend ±window) — §14.0.1. |
| `temporal_pose_refine` | cli | `temporal_pose_refine` | True | Looks a few frames before/after and nudges shaky points toward a local average. Turn off for fastest runs or when you want maximum raw responsiveness. |
| `smoother_beta` | yaml | `SMOOTHER_BETA` | 0.7 | Higher = follow fast moves sooner; lower = calmer but can lag spikes. *(min 0.0 max 5.0)* |

**Locked `params` / env (read-only in Lab):** **§14.0.1** (`MASTER_LOCKED_SMOOTH_*`, `apply_master_locked_smooth_params`, `apply_master_locked_smooth_env`, `SWAY_UNLOCK_SMOOTH_TUNING`).

### 21.10 Phase: `scoring` — Scoring

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| *(info_scoring_actual)* | none | — | — | Compares dancers to each other and to the group timing: who’s early/late, which limbs differ, and rolls that into the numbers you see in exports. There isn’t a separate “mode” switch here—the math is fixed for consistent results. |

### 21.11 Phase: `export` — Export

| Lab field `id` | Binding | Key / target | Default | Effect |
|------------------|---------|--------------|---------|--------|
| `montage` | cli | `montage` | False | Concatenates phase preview clips into a single montage file when enabled. |
| `save_phase_previews` | cli | `save_phase_previews` | True | On by default in the lab: small MP4s per stage so you can scrub what changed without re-running everything. Turn off to save disk and a little export time. |
| `sway_vis_temporal_interp_mode` | env | `SWAY_VIS_TEMPORAL_INTERP_MODE` | linear | Final MP4s can run at native FPS while pose/boxes are stored at processed rate—this chooses how to blend in between. Linear is default. GSI is optional; lengthscale uses SWAY_GSI_LENGTHSCALE or SWAY_VIS_GSI_LENGTHSCALE in YAML. *(choices: linear, gsi)* |
| `sway_hmr_mesh_sidecar` | env | `SWAY_HMR_MESH_SIDECAR` | False | Default off. When on, ``main.py`` writes ``hmr_mesh_sidecar.json`` in the output folder after rendering (schema stub until HMR / mesh export exists). For experiments, see also ``tools/hmr_3d_optional_stub.py``. |

> **Schema vs. `main.py`:** `bbox_size_min_frac` / `bbox_size_max_frac` are schema YAML fields, but Phase 4 `prune_bbox_size_outliers` in `main.py` is called **without** reading those keys — the module constants `BBOX_SIZE_MIN_FRAC` / `BBOX_SIZE_MAX_FRAC` always apply until wired in code.

---

## 22. CLI parameters (`main.py` argparse)

The Lab also sets these via `_cli_from_fields` when you edit a run.

| Flag | Default | Effect |
|------|---------|--------|
| positional `video_path` | — | Input video path. |
| `--output-dir` | output | Output directory. |
| `--pose-model` | base | ViTPose+ sizes, `rtmpose`, or `sapiens` slot (ViTPose-Base + log). |
| `--pose-stride` | 1 | `1` or `2`. |
| `--temporal-pose-refine` / `--no-temporal-pose-refine` | on | Phase 5 neighbor blend; overridden by `SWAY_TEMPORAL_POSE_REFINE`. |
| `--temporal-pose-radius` | 2 | Radius 0–8 for temporal refine. |
| `--montage` | off | `montage.mp4`. |
| `--params` | none | YAML: full `params` dict; `SWAY_*` keys copied to env. |
| `--save-phase-previews` | off | `phase_previews/`; may imply default manifest path. |
| `--progress-jsonl` | none | Progress JSONL. |
| `--run-manifest` | optional | Manifest JSON path. |
| `--no-pose-3d-lift` | off | Skip 3D lift block. |

---

## 23. `params` YAML keys consumed by `main.py`

Omit a key to keep the hard-coded default in `main.py` / `track_pruning.py` / `crossover.py`.

**Phase 4 (pre-pose prune):** For `min_duration_ratio`, `KINETIC_STD_FRAC`, `SPATIAL_OUTLIER_STD_FACTOR`, `SHORT_TRACK_MIN_FRAC`, `AUDIENCE_REGION_*`, values in `params.yaml` are **overwritten** by the master lock (§8.1) unless `SWAY_UNLOCK_PRE_POSE_PRUNE_TUNING=1`.

**Phases 6–7 (re-ID + collocated dedup):** For `REID_MAX_FRAME_GAP`, `REID_MIN_OKS`, `COLLISION_KPT_DIST_FRAC`, `COLLISION_CENTER_DIST_FRAC`, `DEDUP_TORSO_MEDIAN_FRAC`, values in `params.yaml` are **overwritten** by the master lock (§11.0.1) unless `SWAY_UNLOCK_REID_DEDUP_TUNING=1`.

**Phase 8 (post-pose prune):** For the scalar keys in §13.0.1 and the three `PRUNING_WEIGHTS` entries listed there, values are **overwritten** / **merged** by the master lock unless `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING=1`.

**Phase 9 (smoothing):** `SMOOTHER_MIN_CUTOFF` is **overwritten** in `params` and `SWAY_TEMPORAL_POSE_RADIUS` is **set on the environment** unless `SWAY_UNLOCK_SMOOTH_TUNING=1` (§14.0.1).

| YAML key | Default if omitted | Effect |
|----------|-------------------|--------|
| `POSE_VISIBILITY_THRESHOLD` | 0.3 | Phase 5 visibility gate. |
| `min_duration_ratio` | 0.20 | Phase 4 duration + kinetic (`prune_tracks`); **locked** §8.1. |
| `KINETIC_STD_FRAC` | 0.02 | Phase 4 kinetic threshold; **locked** §8.1. |
| `SPATIAL_OUTLIER_STD_FACTOR` | 2.0 | Phase 4 spatial outliers; **locked** §8.1. |
| `SHORT_TRACK_MIN_FRAC` | 0.15 | Phase 4 short tracks; **locked** §8.1. |
| `AUDIENCE_REGION_X_MIN_FRAC` | 0.75 | Phase 4 audience corner (with Y); **locked** §8.1. |
| `AUDIENCE_REGION_Y_MIN_FRAC` | 0.70 | Phase 4 audience corner (with X); **locked** §8.1. |
| `REID_MAX_FRAME_GAP` | 90 | Phase 6 occlusion re-ID; **locked** §11.0.1. |
| `REID_MIN_OKS` | 0.35 | Phase 6 occlusion re-ID; **locked** §11.0.1. |
| `COLLISION_KPT_DIST_FRAC` | 0.26 | Phase 7 dedup; **locked** §11.0.1. |
| `COLLISION_CENTER_DIST_FRAC` | 0.5 | Phase 7 dedup; **locked** §11.0.1. |
| `DEDUP_ANTIPARTNER_MIN_IOU` | 0.12 | Phase 7 dedup. |
| `DEDUP_KPT_TIGHT_FRAC` | 0.20 | Phase 7 dedup. |
| `DEDUP_TORSO_MEDIAN_FRAC` | 0.24 | Phase 7 dedup; **locked** §11.0.1. |
| `DEDUP_MIN_PAIR_OKS` | 0.68 | Phase 7 dedup. |
| `EDGE_MARGIN_FRAC` | 0.15 | Tier A + Tier B + Phase 4 mirror read from `params`; **locked** §13.0.1. |
| `EDGE_PRESENCE_FRAC` | 0.3 | Tier B mirrors; **locked** §13.0.1. |
| `CONFIRMED_HUMAN_MIN_SPAN_FRAC` | 0.10 | Tier A span; **locked** §13.0.1. |
| `TIER_C_SKELETON_MEAN` | 0.15 | Tier C; **locked** §13.0.1. |
| `TIER_C_LOW_FRAME_FRAC` | 0.80 | Tier C; **locked** §13.0.1. |
| `PRUNING_WEIGHTS` | see code | Tier B: dict merging defaults; three keys **forced** §13.0.1 (see §13.0.1 table). |
| `PRUNE_THRESHOLD` | 0.65 | Tier B vote threshold. |
| `SYNC_SCORE_MIN` | 0.10 | Tier B. |
| `min_lower_body_conf` | 0.30 | Tier B smart mirror; **locked** §13.0.1. |
| `MEAN_CONFIDENCE_MIN` | 0.45 | Tier B; **locked** §13.0.1. |
| `JITTER_RATIO_MAX` | 0.10 | Tier B; **locked** §13.0.1. |
| `SMOOTHER_MIN_CUTOFF` | 1.0 | Phase 9 1-Euro; **locked** §14.0.1. |
| `SMOOTHER_BETA` | 0.7 | Phase 9 1-Euro. |

---

## 24. `SWAY_*` and other env vars **not** listed as Lab schema rows

Set via shell or `params.yaml` (`SWAY_*` only for `_apply_params_to_env`).

| Variable | Read in | Effect |
|----------|---------|--------|
| `SWAY_HYBRID_SAM_MIN_DETS` | hybrid_sam_refiner.py | Min person count before overlap logic runs. |
| `SWAY_HYBRID_SAM_WEIGHTS` | hybrid_sam_refiner.py | SAM2 checkpoint name or path. |
| `SWAY_HYBRID_SAM_MASK_THRESH` | hybrid_sam_refiner.py | Mask binarize threshold. |
| `SWAY_HYBRID_SAM_BBOX_PAD` | hybrid_sam_refiner.py | Pad on mask-derived boxes (px). |
| `SWAY_DORMANT_MAX_GAP` | tracker.py → dormant merges | Max gap for dormant registry. |
| `SWAY_STITCH_RADIUS_BBOX_FRAC` | tracker.py | Fragment stitch radius vs bbox height. |
| `SWAY_STITCH_MAX_PIXEL_RADIUS` | tracker.py | Absolute stitch radius fallback. |
| `SWAY_STITCH_PREDICTED_RADIUS_FRAC` | tracker.py | Predicted-box stitch radius scale. |
| `SWAY_SHORT_GAP_FRAMES` | tracker.py | Short gap: generous match without velocity check. |
| `SWAY_COALESCENCE_IOU_THRESH` | tracker.py | Coalescence IoU. |
| `SWAY_COALESCENCE_CONSECUTIVE_FRAMES` | tracker.py | Coalescence consecutive frames. |
| `SWAY_AFLINK_WEIGHTS` | global_track_link.py | Override AFLink `.pth` path. |
| `SWAY_GLOBAL_AFLINK` | global_track_link.py | Force heuristic linker when `0`/heuristic. |
| `SWAY_BOXMOT_REID_ON` | tracker.py | Track-time OSNet embeddings in Deep OC-SORT (Lab sets via `tracker_technology`). |
| `SWAY_BOXMOT_REID_WEIGHTS` | tracker.py / app.py | Explicit Re-ID weights path (overrides preset). |
| `SWAY_TEMPORAL_POSE_REFINE` | temporal_pose_refine.py | Overrides CLI temporal refine. |
| `SWAY_TEMPORAL_POSE_RADIUS` | temporal_pose_refine.py | Neighbor-blend half-window; master stack sets **2** (§14.0.1) unless `SWAY_UNLOCK_SMOOTH_TUNING=1`. |
| `SWAY_POSE_GAP_INTERP_MODE` | main.py | Pose stride gap interpolation. |
| `SWAY_POSE_GSI_LENGTHSCALE` | main.py | Pose GSI lengthscale. |
| `SWAY_GSI_LENGTHSCALE` | tracker.py / main.py / visualizer | Default GSI lengthscale **0.35**; master pipeline forces this after params (§6.2.1) unless `SWAY_UNLOCK_DETECTION_TUNING=1`. |
| `SWAY_UNLOCK_DETECTION_TUNING` | main.py | When `1` / `true` / `yes`, skips `apply_master_locked_detection_env` so smoke tests and advanced scripts can override §6.2.1 keys. |
| `SWAY_UNLOCK_HYBRID_SAM_TUNING` | main.py | When `1` / `true` / `yes`, skips `apply_master_locked_hybrid_sam_env` so smoke tests can set `SWAY_HYBRID_SAM_OVERLAP=0` or override ROI keys (§6.4.1). |
| `SWAY_UNLOCK_PHASE3_STITCH_TUNING` | main.py | When `1` / `true` / `yes`, skips `apply_master_locked_phase3_stitch_env` so smoke tests can set `SWAY_GLOBAL_LINK=0` or override AFLink thresholds (§7.0.1). |
| `SWAY_UNLOCK_POSE_TUNING` | main.py | When `1` / `true` / `yes`, skips `apply_master_locked_pose_env` so smoke tests can set `SWAY_3D_LIFT=0`, chunk ViTPose, or force FP32 (§9.0.1). |
| `SWAY_UNLOCK_REID_DEDUP_TUNING` | main.py / Lab `params` build | When `1` / `true` / `yes`, skips `apply_master_locked_reid_dedup_params` so `params.yaml` can override the five Phase 6–7 keys in §11.0.1. |
| `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING` | main.py / Lab `params` build | When `1` / `true` / `yes`, skips `apply_master_locked_post_pose_prune_params` so `params.yaml` can override Phase 8 locked scalars and the three forced `PRUNING_WEIGHTS` keys (§13.0.1). |
| `SWAY_UNLOCK_SMOOTH_TUNING` | main.py / Lab `params` build + subprocess env | When `1` / `true` / `yes`, skips `apply_master_locked_smooth_params` and `apply_master_locked_smooth_env` / `freeze_lab_subprocess_smooth_env` so `SMOOTHER_MIN_CUTOFF` and `SWAY_TEMPORAL_POSE_RADIUS` can differ from §14.0.1. |
| `SWAY_UNLOCK_PRE_POSE_PRUNE_TUNING` | main.py / Lab `params` build | When `1` / `true` / `yes`, skips `apply_master_locked_pre_pose_prune_params` so `params.yaml` can override the six Phase 4 keys in §8.1. |
| `SWAY_VIS_TEMPORAL_INTERP_MODE` | visualizer.py | Export MP4 tween mode. |
| `SWAY_VIS_GSI_LENGTHSCALE` | visualizer.py | Export GSI lengthscale. |
| `SWAY_STAGE_POLYGON` | main.py | JSON normalized polygon for stage prune. |
| `SWAY_VITPOSE_MODEL` | main.py | HF ViTPose id override. |
| `SWAY_POSE_LOG_EVERY_SEC / _N_PASSES / _SLOW_FORWARD_SEC` | main.py | Phase 5 logging throttle. |
| `SWAY_REID_DEBUG` | main.py | Occlusion re-ID debug. |
| `SWAY_3D_LIFT` | main.py | Disable 3D lift. |
| `SWAY_DEPTH_DYNAMIC / SWAY_DEPTH_STRIDE_FRAMES` | main.py | Depth series for lift. |
| `SWAY_USE_BOXMOT / SWAY_BOXMOT_TRACKER` | tracker.py | Ultralytics track vs BoxMOT; tracker class when BoxMOT. |
| `SWAY_BOXMOT_ASSOC_METRIC` | tracker.py | Deep OC-SORT IoU/GIoU/DIoU/CIoU. |
| `SWAY_TRACKER_YAML` | tracker.py | BoT-SORT YAML. |
| `SWAY_BIDIRECTIONAL_IOU_THRESH / SWAY_BIDIRECTIONAL_MIN_MATCH_FRAMES` | bidirectional_track_merge.py | Bidirectional merge. |
| `SWAY_VITPOSE_FP32 / SWAY_VITPOSE_MAX_PER_FORWARD` | pose_estimator.py | ViTPose inference; master stack reapplies after params (§9.0.1) unless `SWAY_UNLOCK_POSE_TUNING=1`. |
| `SWAY_RTMPose_CONFIG / SWAY_RTMPose_CHECKPOINT` | rtmpose_estimator.py | MMPose paths. |
| `SWAY_LIFT_BACKEND` | pose_lift_3d.py | Lifter backend. |
| `SWAY_MOTIONAGFORMER_* / SWAY_POSEFORMERV2_*` | pose_lift_3d.py | Vendor roots, weights, architecture knobs. |
| `SWAY_LIFT_INPUT_NORM / SWAY_LIFT_GAP_MODE` | pose_lift_3d.py | Lift input handling. |
| `SWAY_UNIFIED_3D_EXPORT / SWAY_AUGLIFT_BLEND` | pose_lift_3d.py | Export + depth blend. |
| `SWAY_FX / SWAY_FY / SWAY_PINHOLE_FOV_DEG` | pose_lift_3d.py | Camera / intrinsics fallbacks. |
| `SWAY_DEPTH_Z_NEAR / SWAY_DEPTH_Z_FAR / SWAY_DEFAULT_ROOT_Z / SWAY_LIFT_WORLD_SCALE` | pose_lift_3d.py | Depth and scale. |
| `SWAY_BONE_LENGTH_FILTER / SWAY_BONE_LENGTH_FILTER_ITERS` | pose_lift_3d.py | Bone length post-filter. |
| `SWAY_LIFT_SAVGOL / SWAY_LIFT_SAVGOL_WINDOW / SWAY_LIFT_SAVGOL_POLY` | pose_lift_3d.py | Savitzky–Golay on `lift_xyz` export. |
| `SWAY_OFFLINE / YOLO_OFFLINE / ULTRALYTICS_OFFLINE` | tracker.py, tools | Block downloads. |
| `HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE` | main.py | HF offline (via params). |
| `SWAY_SAM2_MASK_POSE` | sam2_optional.py | Mask-gated crops. |
| `SWAY_CONF_TEMPERATURE` | calibration.py | Confidence calibration helper. |

*Ad-hoc tools (not `main.py`):* `SWAY_HMR_SMOKE`, `SWAY_SAPIENS_SMOKE`, `SWAY_GNN_TRACK_SMOKE`, `SWAY_SAPIENS_HEATMAP_H/W`, `SWAY_SAPIENS_TORCHSCRIPT`.

---

## 25. `track_pruning.py` constants (when YAML does not override)

Defaults for Phase 4/8 when `main.py` does not pass a `params` override: `KINETIC_STD_FRAC`, `SPATIAL_OUTLIER_STD_FACTOR`, `SPATIAL_MIN_SPREAD_FRAC`, `SHORT_TRACK_MIN_FRAC`, `AUDIENCE_REGION_*`, `LATE_ENTRANT_*`, `BBOX_SIZE_MIN_FRAC`, `BBOX_SIZE_MAX_FRAC`, `ASPECT_RATIO_MAX`, `EDGE_MARGIN_FRAC`, `EDGE_PRESENCE_FRAC` (Phase 4 geometric mirrors always use these two), `ULTRA_LOW_SKELETON_*`, `PRUNING_WEIGHTS`, `PRUNE_THRESHOLD`. For **Phase 8**, many of these thresholds and three `PRUNING_WEIGHTS` entries are **reapplied from the master lock** in `params` (§13.0.1) unless `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING=1`.

---

## 26. `crossover.py` constants (association / collision)

The five YAML keys `REID_MAX_FRAME_GAP`, `REID_MIN_OKS`, `COLLISION_KPT_DIST_FRAC`, `COLLISION_CENTER_DIST_FRAC`, and `DEDUP_TORSO_MEDIAN_FRAC` match module defaults but are **reapplied from the master lock** in `params` (§11.0.1) unless `SWAY_UNLOCK_REID_DEDUP_TUNING=1`. Other `DEDUP_*` keys, late-entrant windows, and embedding / occlusion behavior still follow `params` or constants as documented in source.

---

## 27. `scoring.py` constants

Not exposed as YAML today: `CONFIDENCE_THRESHOLD`, `RIPPLE_STD_THRESHOLD`, `CONSENSUS_ROLLING_WINDOW`, `DTW_WINDOW_SIZE`, `DTW_SAKOE_CHIBA_BAND`, `TIMING_ERROR_THRESHOLD`, joint shape thresholds (`SHAPE_GREEN_SPINE`, etc.).

---

*After large refactors, reconcile §1–§20 against `main.py` and cited modules. Section §21 is generated from `PIPELINE_PARAM_FIELDS` — re-run the export script in the repo (or `python3 -c "from sway.pipeline_config_schema import PIPELINE_PARAM_FIELDS"`) if the schema changes.*
