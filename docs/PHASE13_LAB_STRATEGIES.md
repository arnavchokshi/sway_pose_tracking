# Phases 1–3 Lab strategies: Default stack, Dancer registry, and Sway handshake

This document describes what the three **Phases 1–3 strategy** choices in the Pipeline Lab actually do in `main.py` (detection → track → early stitch). It reflects the **current implementation** in the repo, not marketing copy alone.

---

## Confirmation: do the three paths do different things?

**Yes.** They set different `sway_phase13_mode` values, which the Lab API turns into different subprocess environment variables and therefore different code paths in the tracker, optional post-track passes, and ViTPose cropping.


| Strategy (Lab label) | `sway_phase13_mode` | Distinct behavior                                                                                                                                                                          |
| -------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Default stack**    | `standard`          | Baseline: hybrid SAM at your Lab IoU trigger, no handshake state machine, no dancer-registry video passes.                                                                                 |
| **Dancer registry**  | `dancer_registry`   | Extra **zonal HSV** passes over the video to fix ID swaps after crossovers and to merge some dormant gaps. **ViTPose smart pad** is **master-default for all strategies** (§9.0.1 in `MASTER_PIPELINE_GUIDELINE.md`), not unique to this mode.                                                  |
| **Sway handshake**   | `sway_handshake`    | **Lower hybrid SAM IoU trigger (0.10)** and **weak cues off**; per-frame **handshake** registry + optional **Hungarian reorder** of SAM rows so masks align with appearance before BoxMOT. |


So the engines are not cosmetic: different branches run in `tracker.py`, `main.py`, `dancer_registry_pipeline.py`, `handshake_tracking.py`, and `pose_estimator.py`.

---

## Confirmation: does clicking them in the UI actually change runs?

**Yes, when you save the run and start the batch** (or single upload), the chosen strategy must appear in the queued payload and in `runs/<id>/request.json` as:

```json
"sway_phase13_mode": "standard" | "dancer_registry" | "sway_handshake"
```

**How the UI applies it**

- **Run editor — Recipe baseline:** The three buttons call `applyLabRecipe(schema, qualityTier, phase13Strategy)` in `pipeline_lab/web/src/components/RunEditorModal.tsx`, which merges `PHASE13_STRATEGY_OVERRIDES` from `configPresets.ts` and sets `sway_phase13_mode` accordingly.
- **Main choices cards:** The `sway_phase13_mode` field (`display: phase13_mode_cards`) updates the same key; both paths must agree.
- **Server:** `pipeline_lab/server/app.py` builds the child environment in `_subprocess_env()` from the merged `fields` dict. The generic loop first writes env vars for all `binding: env` fields (including `SWAY_PHASE13_MODE` from `sway_phase13_mode`), then applies **mode-specific overrides** for `dancer_registry` and `sway_handshake`.

**Important (fixed bug):** `sway_phase13_mode` must **not** be listed in `LAB_UI_ENFORCED_DEFAULTS` / `_UI_LOCK_DEFAULTS` in `sway/pipeline_config_schema.py`. If it were forced to `standard` after merge, every run would behave as Default stack regardless of the UI. That overwrite was removed so non-default strategies persist through enqueue.

---

## Shared foundation (all three strategies)

For all modes, the Lab still uses the same high-level pipeline shape:

1. **Phase 1:** YOLO person detection + pre-track NMS (BoxMOT path in `sway/tracker.py`).
2. **Phase 2:** Deep OC-SORT (BoxMOT) on the detection stream, optionally with **hybrid SAM** refinement when overlap exceeds `SWAY_HYBRID_SAM_IOU_TRIGGER` (unless overlap is disabled via env).
3. **Phase 3:** Post-track stitching (dormant merges, fragment stitch, coalescence, global link / AFLink, etc.) in `apply_post_track_stitching` and related modules.

What changes between strategies is **which optional machinery is active** around that core.

---

## 1. Default stack (`standard`)

### Intent

The baseline tuned against the main quality presets: hybrid SAM when boxes overlap enough, standard hybrid tuning from the Lab sliders, no Sway Handshake object, no dancer-registry appearance passes.

### Environment / fields (Lab API)

- `SWAY_PHASE13_MODE` is set from the schema binding (typically `standard`).
- **ViTPose smart pad** is **not** strategy-specific: `freeze_lab_subprocess_pose_env` sets `SWAY_VITPOSE_SMART_PAD=1` for every Lab run (same as `apply_master_locked_pose_env` in `main.py` unless `SWAY_UNLOCK_POSE_TUNING=1`).
- Hybrid SAM: **on** by master-locked overlap flags (`SWAY_HYBRID_SAM_OVERLAP=1`, etc. via `freeze_lab_subprocess_hybrid_sam_env`). IoU trigger comes from your `**sway_hybrid_sam_iou_trigger`** field (e.g. `0.42` on the Standard quality tier).

### Phase 1–2 tracking (`_run_tracking_boxmot_diou`)

- `phase13_handshake_enabled()` is **false** → no `SwayHandshakeState`, no `handshake_process_frame`.
- Hybrid SAM refiner runs according to `load_hybrid_sam_config()` (trigger from env / UI).

### Phase 3 (`main.py` → `_run_phase3_stitch`)

- `phase13_dancer_registry_enabled()` is **false** → **no** `apply_dancer_registry_crossover_pass` before `apply_post_track_stitching`; input to stitching is `raw_pre` unchanged by the registry module.

### `apply_post_track_stitching` (`sway/tracker.py`)

- Motion-based dormant and the rest of the stitch pipeline run as usual.
- **No** `apply_dancer_registry_appearance_dormant`.

### Pose (`sway/pose_estimator.py`)

- `vitpose_smart_pad_enabled()` is **true** under the master pose stack (`SWAY_VITPOSE_SMART_PAD=1`). Opt out with `SWAY_VITPOSE_SMART_PAD=0` / `false` / `no` / `off` (e.g. after `SWAY_UNLOCK_POSE_TUNING=1`).

---

## 2. Dancer registry (`dancer_registry`)

### Intent

Improve **identity consistency** on open-floor dance when people cross paths: build **appearance profiles** (zonal HSV histograms + aspect), detect **touch → separate** intervals, and **retroactively swap** track IDs when evidence says two IDs were exchanged. After motion-based dormant merging, optionally **relink** segments using the same kind of appearance when motion alone did not merge them. **ViTPose smart bbox padding** is the same **master-default** as Default stack and Sway handshake (not a differentiator for this recipe).

This path is **histogram / video-scan based** for the registry passes; it does **not** use SAM inside `dancer_registry_pipeline.py`.

### Environment / fields (Lab API)

From `_subprocess_env` when `sway_phase13_mode == "dancer_registry"`:

- `SWAY_PHASE13_MODE=dancer_registry`
- (No separate smart-pad line: `SWAY_VITPOSE_SMART_PAD=1` comes from `freeze_lab_subprocess_pose_env` for **all** Lab strategies.)

Hybrid SAM for BoxMOT is still governed by the same hybrid overlap machinery as standard: master lock keeps overlap **on**, and `**sway_hybrid_sam_iou_trigger`** still applies (the Dancer registry preset in `configPresets.ts` sets `0.42` like Standard). So in the **current code**, registry mode does **not** globally disable hybrid SAM during tracking; the main contrast with handshake is **when** SAM fires (handshake forces **0.10**) and the **extra registry passes** below.

### Phase 1–2 tracking

Same YOLO + optional hybrid SAM + BoxMOT loop as standard **unless** you change hybrid env manually. No handshake state.

### Phase 3 — crossover verify (`main.py`)

Before `apply_post_track_stitching`, if `phase13_dancer_registry_enabled()`:

- Logs: `[dancer_registry] Crossover verify pass (zonal HSV, no SAM)…`
- Calls `apply_dancer_registry_crossover_pass` in `sway/dancer_registry_pipeline.py`:
  - **Warm-up window** (~10 s by default, `SWAY_REGISTRY_WARMUP_SEC`): accumulate zonal features per track when the box is **spatially isolated** (no other box center within `SWAY_REGISTRY_ISOLATION_MULT` × body scale, default `1.5`).
  - After warm-up, **lock** profiles per track ID.
  - While scanning the video, detect pairs that **touch** (IoU ≥ `SWAY_REGISTRY_TOUCH_IOU`, default `0.12`) then **clear** (IoU ≤ `SWAY_REGISTRY_CLEAR_IOU`, default `0.02`) with sufficient **separation** (`SWAY_REGISTRY_SEPARATION_MULT`, default `1.5`).
  - Compare post-crossover appearance to locked profiles; if scores favor a swap by margin `SWAY_REGISTRY_SWAP_MARGIN` (default `0.06`), apply `_swap_track_interval` over the crossover window (mutates `raw_tracks` in place).

This requires a **full sequential decode** of the source video after tracking.

### Phase 3 — stitching (`apply_post_track_stitching`)

After `_apply_dormant_and_global` (motion dormant):

- If `phase13_dancer_registry_enabled()` and `video_path` is set: `apply_dancer_registry_appearance_dormant`:
  - Another video scan to sample **first isolated** appearance for segments.
  - Tries to merge segment **B** after **A** when gap is in (`SWAY_DORMANT_MAX_GAP` range, default `150` frames with a buffer) and zonal similarity ≥ `SWAY_REGISTRY_DORMANT_MATCH` (default `0.82`).

Then the usual fragment stitch, coalescence, etc. continue.

### Pose

`vitpose_smart_pad_enabled()` is **true** when the master pose stack is active (`SWAY_VITPOSE_SMART_PAD=1`). `smart_expand_bbox_xyxy` expands crops with motion lead, lift-aware top pad, and aspect nudges for ViTPose.

---

## 3. Sway handshake (`sway_handshake`)

### Intent

When **SAM2** runs on heavy overlap, **row order** of detections must stay aligned with **who is who** before `DeepOcSort.update`. Handshake maintains a **zonal appearance registry** on the **open floor** (low pairwise IoU), then on SAM frames can **permute** detection rows (and mask list) so each mask best matches the registry profile for the track ID implied by motion continuity — reducing **ID–mask slips** in crowded choreography.

### Environment / fields (Lab API)

From `_subprocess_env` when `sway_phase13_mode == "sway_handshake"`:

- `SWAY_PHASE13_MODE=sway_handshake`
- `SWAY_HYBRID_SAM_IOU_TRIGGER=0.10` (overrides a higher Lab slider for this strategy so SAM runs **much more often** on moderate overlap).
- `SWAY_HYBRID_SAM_WEAK_CUES=0` (weak-cue gate off for this path).
- **ViTPose smart pad** is unchanged vs Default stack: still **on** via `freeze_lab_subprocess_pose_env` (not cleared for handshake).

### Phase 1–2 tracking

- `phase13_handshake_enabled()` → `SwayHandshakeState` is constructed; logs mention handshake + IoU threshold.
- Per frame, after `hybrid_refiner.refine_person_dets` (if present), `handshake_process_frame` runs (`sway/handshake_tracking.py`):
  - `**update_registry_open_floor`:** When max pairwise IoU is **below** the hybrid trigger, update per-ID zonal HSV fingerprints from **isolated** boxes aligned to the **previous tracker output** (EMA-smoothed registry).
  - When `**used_sam`** is true and an overlap cluster has ≥ 2 detections, `**verify_and_reorder_sam_dets**` (if `scipy` is available): extract features from **masked** regions, match columns to track IDs via previous tracker positions, build a cost matrix from profile scores, run **Hungarian assignment**, permute `dets` and `per_det_masks` so BoxMOT sees rows consistent with appearance.
- Verification runs every `SWAY_HANDSHAKE_VERIFY_STRIDE` SAM frames (see `SwayHandshakeState`).

### Phase 3

- **No** `apply_dancer_registry_crossover_pass`.
- **No** `apply_dancer_registry_appearance_dormant`.

### Pose

Same as Default stack: **master-locked** ViTPose smart pad (`SWAY_VITPOSE_SMART_PAD=1` via `freeze_lab_subprocess_pose_env`).

---

## Side-by-side summary


| Topic                                   | Default stack           | Dancer registry                  | Sway handshake       |
| --------------------------------------- | ----------------------- | -------------------------------- | -------------------- |
| `SWAY_PHASE13_MODE`                     | `standard`              | `dancer_registry`                | `sway_handshake`     |
| Handshake registry + SAM row reorder    | No                      | No                               | Yes                  |
| Crossover zonal pass (pre-stitch)       | No                      | Yes (full video pass)            | No                   |
| Appearance dormant relink               | No                      | Yes (after motion dormant)       | No                   |
| ViTPose smart pad (Lab / master default) | **On** (§9.0.1)         | **On** (same lock)               | **On** (same lock)   |
| Hybrid SAM IoU trigger (Lab subprocess) | Your slider (e.g. 0.42) | Your slider (preset often 0.42)  | **Forced 0.10**      |
| Weak SAM cues                           | From field / locks      | From field / locks               | **Forced off**       |
| Extra video decodes                     | Baseline                | **+1–2** full scans for registry | Baseline             |


---

## Reference: main modules


| Area              | File                                    | What to read                                                                                                          |
| ----------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Lab → env         | `pipeline_lab/server/app.py`            | `_subprocess_env`, `_effective_ui_fields`, `_enqueue_run`                                                             |
| UI recipe merge   | `pipeline_lab/web/src/configPresets.ts` | `PHASE13_STRATEGY_OVERRIDES`, `applyLabRecipe`                                                                        |
| Handshake         | `sway/handshake_tracking.py`            | `phase13_handshake_enabled`, `handshake_process_frame`, `SwayHandshakeState`                                          |
| Dancer registry   | `sway/dancer_registry_pipeline.py`      | `phase13_dancer_registry_enabled`, `apply_dancer_registry_crossover_pass`, `apply_dancer_registry_appearance_dormant` |
| Tracking loop     | `sway/tracker.py`                       | `_run_tracking_boxmot_diou`, `apply_post_track_stitching`                                                             |
| Phase 3 entry     | `sway_pose_mvp/main.py`                 | `_run_phase3_stitch` (crossover branch)                                                                               |
| ViTPose crops     | `sway/pose_estimator.py`                | `vitpose_smart_pad_enabled`, `smart_expand_bbox_xyxy`                                                                 |
| Hybrid SAM config | `sway/hybrid_sam_refiner.py`            | `load_hybrid_sam_config`                                                                                              |


---

## Tunable registry / handshake environment variables (non-exhaustive)

**Dancer registry (`dancer_registry_pipeline.py`):**

- `SWAY_REGISTRY_WARMUP_SEC`, `SWAY_REGISTRY_ISOLATION_MULT`, `SWAY_REGISTRY_TOUCH_IOU`, `SWAY_REGISTRY_CLEAR_IOU`, `SWAY_REGISTRY_SEPARATION_MULT`, `SWAY_REGISTRY_SWAP_MARGIN`
- `SWAY_DORMANT_MAX_GAP`, `SWAY_REGISTRY_DORMANT_MATCH`

**Handshake:** see `SwayHandshakeState` and `handshake_tracking.py` for stride and EMA (`SWAY_HANDSHAKE_VERIFY_STRIDE`, etc., if defined in that module).

---

## When outputs might still look similar

Even with correct wiring, two strategies can produce **similar** MP4s if:

- The clip has **little occlusion** (handshake’s SAM reorder rarely triggers; crossover events are rare).
- **Dancers wear similar colors** (HSV profiles are weaker).
- **SciPy** is missing (handshake Hungarian path is skipped).
- Other locked Lab defaults dominate the visible result.

For a **sanity check**, compare `request.json` → `fields.sway_phase13_mode` and watch logs for `[dancer_registry]` or `Sway Handshake:` lines during Phase 1–3.