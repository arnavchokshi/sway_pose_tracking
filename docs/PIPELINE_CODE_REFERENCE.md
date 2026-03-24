# Sway Pose MVP — Pipeline code reference (for review)

This document describes the **end-to-end pipeline as implemented in code**, with emphasis on **defaults** when you run:

```bash
python main.py <path/to/video.mp4>
```

(no `--params`, no optional CLI flags). Line references point at the current `sway_pose_mvp` tree.

**Canonical orchestrator:** `main.py` (module docstring lists stages `[1/11]`–`[11/11]`).

**Related docs (may drift from code):** `docs/FINAL_OPTIMIZED_PIPELINE.md`, `docs/HYBRID_SAM_PIPELINE_HANDOFF.txt`.

---

## 1. Entry point and CLI defaults

| Argument | Default | Notes |
|----------|---------|--------|
| `video_path` | *(required)* | Input MP4 |
| `--output-dir` | `output` | All artifacts |
| `--pose-model` | `base` | Maps to Hugging Face ViTPose+ id unless env overrides |
| `--pose-stride` | `1` | `1` or `2`; non-pose frames interpolated |
| `--temporal-pose-refine` | on (use `--no-temporal-pose-refine` to skip) | Lightweight ±N-frame keypoint smooth (not Poseidon) |
| `--temporal-pose-radius` | `2` | Used only if refine on; clamped 0–8 |
| `--montage` | off | Stitches phase clips to `montage.mp4` |
| `--params` | `None` | YAML → `SWAY_*` env and numeric overrides in `main` |
| `--save-phase-previews` | off | Writes `phase_previews/*.mp4` |
| `--progress-jsonl` | `None` | Pipeline Lab |
| `--run-manifest` | `None` | JSON manifest (auto path if previews/progress enabled) |

Source: `main.py` `argparse` block (~L411–485).

---

## 2. Startup side effects (before Phase 1)

### 2.1 Stack defaults (`_apply_stack_default_env`)

If not already set, `main` sets:

| Env var | Value | Effect |
|---------|-------|--------|
| `SWAY_GROUP_VIDEO` | `1` | Tracker uses **larger YOLO square resize**: `max(SWAY_DETECT_SIZE or 640, 960)` during detection |
| `SWAY_GLOBAL_LINK` | `1` | Phase 3 runs **global track stitching** after dormant merges |

Source: `main.py` ~L116–122.

### 2.2 Params YAML → environment

If `--params path.yaml` is passed, `_apply_params_to_env` copies keys starting with `SWAY_` plus offline helpers (`HF_HUB_OFFLINE`, etc.) into `os.environ`. Booleans become `"1"`/`"0"`.

Source: `main.py` ~L364–375.

### 2.3 Device selection (ViTPose and general `main` flow)

`get_device()`: **MPS** if `torch.backends.mps.is_available()`, else **CPU**.

**Important:** BoxMOT Deep OC-SORT uses **`cuda` if `torch.cuda.is_available()`, else `cpu`** inside `sway/tracker.py` — it does not follow `get_device()` for the tracker.

Source: `main.py` ~L194–198; `sway/tracker.py` `_run_tracking_boxmot_diou` ~L937.

### 2.4 Directories

`output_dir` created; `input/` and `models/` mkdir’d at repo cwd.

Source: `main.py` ~L500–503.

---

## 3. Data contract between stages

### 3.1 After tracking (Phases 1–2)

`raw_tracks`: `Dict[track_id, List[TrackObservation or legacy tuple]]` — per-frame boxes and confidences. Implementations use `TrackObservation` with optional `is_sam_refined` and `segmentation_mask`.

### 3.2 Per-frame structure used from Phase 5 onward

`all_frame_data_pre` / `all_frame_data` entries are dicts with keys including:

- `frame_idx`, `boxes`, `track_ids`, `poses` (per-track ViTPose dicts)
- `embeddings` (HSV-strip, Phase 5)
- `is_sam_refined`, `segmentation_masks`
- After scoring: `track_angles`, `consensus_angles`, `deviations`, `shape_errors`, `timing_errors`

Source: `main.py` assembly ~L980–1000, Phase 10 attach ~L1325–1332.

---

## 4. Phase 1–2: Detection + tracking (single streaming pass)

**Functions:** `run_tracking_before_post_stitch(video_path)` → returns  
`(raw_pre, total_frames, output_fps, _frames_list, native_fps, frame_width, frame_height, ystride)`.

**Default tracker path:** `SWAY_USE_BOXMOT` unset or truthy → **BoxMOT `DeepOcSort`**. Set `SWAY_USE_BOXMOT=0` → Ultralytics **BoT-SORT** via `model.track(..., tracker=config)`.

Source: `sway/tracker.py` `run_tracking_before_post_stitch` ~L1189–1204, `_use_boxmot` ~L127–130.

### 4.1 YOLO weights resolution

`resolve_yolo_model_path()`:

1. If `SWAY_YOLO_WEIGHTS` set: filesystem `.pt` path, or alias token (e.g. `yolo26l` → `yolo26l.pt`), or raw hub string.
2. Else scan `models/`, repo root, cwd for a **priority list** starting with `yolo26l.pt`, `yolo26l_dancetrack.pt`, …
3. Else Core ML `yolo11l.mlpackage` / `yolo11m.mlpackage` if present.
4. Else hub **`yolo26l.pt`** (unless offline env forbids).

Source: `sway/tracker.py` ~L251–322.

### 4.2 Streaming and detection resolution

| Constant / env | Default | Meaning |
|----------------|---------|---------|
| `CHUNK_SIZE` / `SWAY_CHUNK_SIZE` | `300` | Frames read per chunk |
| `DETECT_SIZE` / `SWAY_DETECT_SIZE` | `640` | Square resize width/height for YOLO input (before group bump) |
| `YOLO_CONF` / `SWAY_YOLO_CONF` | `0.22` | Confidence threshold |
| `YOLO_DETECTION_STRIDE` / `SWAY_YOLO_DETECTION_STRIDE` | `1` | Run detector every Nth frame; gaps filled in Phase 3 |

With `SWAY_GROUP_VIDEO=1`, effective detect size is **`max(base, 960)`**.

Source: `sway/tracker.py` top constants ~L33–45, `load_tracking_runtime` ~L58–90, `_run_tracking_boxmot_diou` loop ~L965–975.

### 4.3 Pre-tracker NMS (BoxMOT path)

1. **DIoU-NMS** `iou_threshold=0.7` (`diou_nms_indices`) — **skipped when weights path/id contains `yolo26`** (YOLO26’s own NMS + classical step only).
2. **Classical IoU NMS** `iou_thresh=0.50` (`_PRETRACK_CLASSICAL_NMS_IOU`) to drop duplicate / hallucinated boxes.

Source: `sway/tracker.py` `_PRETRACK_CLASSICAL_NMS_IOU`, `_yolo26_series_weights`, BoxMOT detection loop.

### 4.4 Deep OC-SORT parameters (defaults)

Instantiated with:

- `reid_weights` from `_resolve_boxmot_reid_weights()` (env `SWAY_BOXMOT_REID_WEIGHTS`, else `models/osnet_x0_25_msmt17.pt` or BoxMOT `WEIGHTS` path)
- `device`: CUDA if available else CPU
- `half=True` only on CUDA
- `det_thresh` = YOLO conf
- `max_age=150`, `min_hits=2`, `iou_threshold=0.3`
- **`embedding_off=True`**

Source: `sway/tracker.py` ~L198–215, ~L936–948.

### 4.5 Hybrid SAM overlap refiner (default **on**)

`load_hybrid_sam_config()` in `sway/hybrid_sam_refiner.py`:

| Key | Default | Env override |
|-----|---------|----------------|
| `enabled` | `True` | `SWAY_HYBRID_SAM_OVERLAP=0` disables |
| `iou_trigger` | `0.42` | `SWAY_HYBRID_SAM_IOU_TRIGGER` |
| `min_dets` | `2` | `SWAY_HYBRID_SAM_MIN_DETS` |
| `weights` | `sam2.1_b.pt` | `SWAY_HYBRID_SAM_WEIGHTS` (+ local resolve under `models/`) |
| `mask_thresh` | `0.5` | `SWAY_HYBRID_SAM_MASK_THRESH` |
| `bbox_pad` | `2` px | `SWAY_HYBRID_SAM_BBOX_PAD` |

When active, person dets are refined before `tracker.update`; masks are assigned back to tracks for downstream pose gating.

Source: `sway/hybrid_sam_refiner.py` ~L46–70; `sway/tracker.py` ~L949–1047.

### 4.6 BoT-SORT fallback

Uses `_get_tracker_config()` → `SWAY_TRACKER_YAML` if valid file, else `config/botsort.yaml`.

Source: `sway/tracker.py` ~L866–877, `_run_tracking_botsort_pre_stitch` ~L1080–1174.

---

## 5. Phase 3: Post-track stitching

**Function:** `apply_post_track_stitching(raw_pre, total_frames, ystride=ystride)`.

**Order (code):**

1. `load_tracking_runtime()` for stitch/coalescence params.
2. `_apply_dormant_and_global`: `apply_dormant_merges` then, if `SWAY_GLOBAL_LINK` truthy, `maybe_global_stitch` (`sway/global_track_link.py`). **Neural AFLink** runs when `models/AFLink_epoch20.pth` exists (or `SWAY_AFLINK_WEIGHTS`); otherwise heuristic stitch. Set `SWAY_GLOBAL_AFLINK=0` to force heuristic even if weights exist. Prefetch step `[6/6]` in `prefetch_models.py` documents where to obtain weights.
3. `stitch_fragmented_tracks` — max gap default **60** frames, radius **0.5×** bbox height, predicted radius frac **0.75**, pixel fallback **120**, short gap **20** frames.
4. `coalescence_deduplicate` — IoU **0.70**, **8** consecutive frames (env-overridable).
5. `merge_complementary_tracks`, `merge_coexisting_fragments`.
6. `_fill_stride_gaps` for YOLO stride gaps.

Source: `sway/tracker.py` `apply_post_track_stitching` ~L93–124, constants ~L47–55.

---

## 6. Phase 4: Pre-pose pruning

**Executed in order in `main.py`** (each step subtracts from `surviving_ids` and may append to `prune_log_entries`):

1. **`prune_tracks`** — duration + kinetic filter. Defaults: `min_duration_ratio=0.20`, `kinetic_std_frac=KINETIC_STD_FRAC` (**0.02** in `track_pruning.py`; docstring in `prune_tracks` still mentions 0.03 in prose — **code uses 0.02**). YAML: `min_duration_ratio`, `KINETIC_STD_FRAC`.
2. **Stage polygon** — `SWAY_STAGE_POLYGON` JSON if set; else if `SWAY_AUTO_STAGE_DEPTH` defaults to **`1`** in `main.py` (`os.environ.get("SWAY_AUTO_STAGE_DEPTH", "1") == "1"`), estimate via `estimate_stage_polygon(first_frame)` from `sway.depth_stage`. Then **`prune_by_stage_polygon`**.
3. **`prune_spatial_outliers`** — params `SPATIAL_OUTLIER_STD_FACTOR`.
4. **`prune_short_tracks`** — params `SHORT_TRACK_MIN_FRAC`; `main.py` logs these as tracks under 20% of the video length.
5. **`prune_audience_region`** — params `AUDIENCE_REGION_X_MIN_FRAC`, `AUDIENCE_REGION_Y_MIN_FRAC`.
6. **`prune_late_entrant_short_span`**
7. **`prune_bbox_size_outliers`**
8. **`prune_bad_aspect_ratio`**
9. **`prune_geometric_mirrors`** — pre-pose edge + inverted velocity mirrors (smart mirror with pose comes later in Tier B).

Finally **`raw_tracks_to_per_frame`** builds **`tracking_results`** for Phase 5.

Source: `main.py` ~L595–703; `sway/track_pruning.py` module docstring ~L1–27 for rule intent.

---

## 7. Phase 5: ViTPose+, visibility gating, optional stride

### 7.1 Model ID

- Env `SWAY_VITPOSE_MODEL` wins if set.
- Else CLI: `huge` → `usyd-community/vitpose-plus-huge`, `large` → `...-large`, default **`base`** → `usyd-community/vitpose-plus-base`.

**Class:** `PoseEstimator(device=device, model_name=model_id)` — device from `get_device()` (MPS/CPU).

Source: `main.py` ~L741–757.

### 7.2 Per-frame work

- Queue + `ThreadPoolExecutor(max_workers=1)` producer reads frames with `iter_video_frames`.
- **`extract_embeddings`** with `method="hsv_strip"` when boxes exist (Re-ID input for Phase 6).
- Pose runs when `(frame_idx % pose_stride == 0)` and boxes exist.
- **`compute_visibility_scores`** — skip ViTPose when visibility is below **`vis_skip`**: `float(params.get("POSE_VISIBILITY_THRESHOLD", 0.3))` (default **0.3** with no YAML).
- Dynamic crop padding: typically **0.15**, **0.25** if large movement/size change, **0.10** if small.
- **`pose_estimator.estimate_poses`** with optional `segmentation_masks` for mask-gated crops.

### 7.3 Pose stride & temporal refine

- If `--pose-stride` is `2`, **`_interpolate_pose_gaps`** fills skipped frames.
- If **`want_temporal_pose_refine`**: **default on** (`--temporal-pose-refine` / `--no-temporal-pose-refine`); env `SWAY_TEMPORAL_POSE_REFINE=0|1` overrides CLI → **`apply_temporal_keypoint_smoothing`** with radius from `temporal_pose_radius` (env `SWAY_TEMPORAL_POSE_RADIUS` or CLI, clamped 0–8).

Source: `main.py` ~L764–972; `sway/temporal_pose_refine.py`.

### 7.4 Logging env (Phase 5)

`SWAY_POSE_LOG_EVERY_SEC` (default `20`), `SWAY_POSE_LOG_EVERY_N_PASSES` (default `8`), `SWAY_POSE_SLOW_FORWARD_SEC` (default `4`).

Source: `main.py` ~L768–770.

---

## 8. Phase 6: Association

On **`all_frame_data_pre`**:

1. **`apply_occlusion_reid`** — optional YAML `REID_MAX_FRAME_GAP`, `REID_MIN_OKS`; debug via `SWAY_REID_DEBUG`.
2. **`apply_crossover_refinement`** — frame dimensions passed.
3. **`apply_acceleration_audit`**

Source: `main.py` ~L1026–1038; `sway/crossover.py`.

---

## 9. Phase 7: Collision cleanup

Per frame:

1. **`deduplicate_collocated_poses`** — `late_entrant_candidates` is **empty set** in current `main` (comment: removed protection for ghosts).
2. **`sanitize_pose_bbox_consistency`** — keypoints outside bbox handled; logging to `sway.crossover` at INFO with `[collision]` prefix.

Telemetry appended to **`prune_log_entries`**.

Source: `main.py` ~L1042–1091.

---

## 10. Phase 8: Post-pose pruning (Tier C + Tier B)

1. **`compute_confirmed_human_set`** — `edge_margin_frac` from params default **0.15** (`EDGE_MARGIN_FRAC`).
2. **Tier C:** **`prune_ultra_low_skeleton_tracks`** — optional YAML `TIER_C_SKELETON_MEAN`, `TIER_C_LOW_FRAME_FRAC`.
3. **Tier B:** **`compute_phase7_voting_prune_set`** with:
   - **`PRUNING_WEIGHTS`** / **`PRUNE_THRESHOLD`** from `sway/track_pruning.py` unless YAML overrides `PRUNING_WEIGHTS` dict or `PRUNE_THRESHOLD`.

**Module defaults** (`track_pruning.py`):

- `PRUNE_THRESHOLD = 0.65`
- Weights: `prune_low_sync_tracks` 0.7, `prune_smart_mirrors` 0.9, `prune_completeness_audit` 0.6, `prune_head_only_tracks` 0.8, `prune_low_confidence_tracks` 0.5, `prune_jittery_tracks` 0.5

**Additional thresholds passed from `main`** (YAML keys with defaults when key absent):

| Parameter key | Default in `main` |
|---------------|-------------------|
| `SYNC_SCORE_MIN` | `0.10` |
| `EDGE_MARGIN_FRAC` | `0.15` |
| `EDGE_PRESENCE_FRAC` | `0.3` |
| `min_lower_body_conf` | `0.3` |
| `MEAN_CONFIDENCE_MIN` | `0.45` |
| `JITTER_RATIO_MAX` | `0.10` |

Then build **`postprune_fd`** by filtering pruned track IDs from poses/boxes.

Source: `main.py` ~L1130–1230; `sway/track_pruning.py` ~L48–57.

---

## 11. Phase 9: Temporal smoothing (1 Euro)

**`PoseSmoother`** constructed with:

- `min_cutoff` ← `params.get("SMOOTHER_MIN_CUTOFF", 1.0)`
- `beta` ← `params.get("SMOOTHER_BETA", 0.7)`

Smoothed poses written to **`all_frame_data`** (pruned tracks and tracks without poses filtered out).

Source: `main.py` ~L1251–1292; `sway/smoother.py`.

---

## 12. Phase 10: Spatio-temporal scoring

**`process_all_frames_scoring_vectorized(all_frame_data)`** attaches per-frame `track_angles`, `consensus_angles`, `deviations`, and optionally `shape_errors`, `timing_errors`.

Source: `main.py` ~L1322–1332; `sway/scoring.py`.

---

## 13. Phase 11: Export and diagnostics

### 13.1 Overlays

- **`build_pruned_overlay_for_review`**
- **`build_dropped_pose_overlay`** (uses snapshots from pre/post dedup/sanitize)

### 13.2 `render_and_export`

Writes under `output_dir`:

| Output | Description |
|--------|-------------|
| `data.json` | Metadata, `track_summaries`, per-frame tracks; optional `pruned_overlay`, `dropped_pose_overlay`, `prune_entries` in metadata |
| `{stem}_poses.mp4` | Full visualization (boxes + skeleton + deviation styling) |
| `{stem}_track_ids.mp4` | Boxes + IDs only |
| `{stem}_skeleton.mp4` | Skeleton / heatmap style |
| `{stem}_sam_style.mp4` | SAM-style visualization for hybrid-refined detections |

Videos: OpenCV `mp4v`, **`native_fps`**; overlays **interpolated** between consecutive entries in `all_frame_data` using `processed_fps` / `native_fps` ratio (with default pipeline **`output_fps == native_fps`**, index tracks source frames 1:1). Audio muxed via `_mux_audio` when available.

Source: `sway/visualizer.py` `render_and_export` ~L842–1023.

### 13.3 `prune_log.json`

Written to **`output_dir / "prune_log.json"`** with tracker IDs before prune, surviving sets, and `prune_entries`.

Source: `main.py` ~L1436–1456.

### 13.4 Optional montage

If `--montage` and clips collected: **`stitch_montage`** → `output_dir/montage.mp4`.

Source: `main.py` ~L1401–1403.

---

## 14. Review checklist (common gotchas)

1. **Two “devices”:** ViTPose on MPS/CPU; BoxMOT tracker on CUDA/CPU — not unified.
2. **`SWAY_GROUP_VIDEO` / `SWAY_GLOBAL_LINK`** are set by `main` unless user pre-sets env — changes detection size and Phase 3 global linking.
3. **YOLO weights** depend on files in `models/` / cwd — first match in priority list wins; not always `yolo26l.pt` from hub.
4. **`prune_tracks` docstring** vs **`KINETIC_STD_FRAC`**: code default kinetic fraction is **0.02**.
5. **`visualizer.render_and_export` docstring** mentions “15 FPS” for JSON — **`main` passes `output_fps`** from the tracker, which equals **`native_fps`** in the default path; verify if downstream tools assume a fixed 15.

---

## 15. File index (primary)

| Area | Files |
|------|--------|
| Orchestration | `main.py` |
| Detection / tracking / stitch | `sway/tracker.py` |
| Hybrid SAM | `sway/hybrid_sam_refiner.py` |
| Pruning | `sway/track_pruning.py` |
| Pose | `sway/pose_estimator.py` |
| Association / collision | `sway/crossover.py` |
| Re-ID embeddings | `sway/reid_embedder.py` |
| Smoothing | `sway/smoother.py` |
| Scoring | `sway/scoring.py` |
| Export | `sway/visualizer.py` |
| Stage polygon (auto) | `sway/depth_stage.py` |
| Dormant / global link | `sway/dormant_tracks.py`, `sway/global_track_link.py` |

---

*Generated from repository snapshot; re-verify line numbers after large edits.*
