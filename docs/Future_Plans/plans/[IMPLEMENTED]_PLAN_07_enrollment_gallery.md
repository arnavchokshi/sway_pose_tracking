# PLAN 07: Dancer Enrollment System and Identity Gallery

> **Implementation Phase:** I1 (Foundation) · **Lean Core — highest-value change** · **Sweep Gate:** Gate 1 — bigtest HOTA > 0.55

## Objective

Implement an **enrollment** phase that builds a **closed-set identity gallery** for every dancer before (or at the start of) full-sequence tracking.

- **Open-set re-ID:** match against arbitrary people in the wild — hard.
- **Closed-set re-ID:** match only among N enrolled dancers — much easier and more stable.

Downstream re-ID, fusion, and state machines should compare queries to **curated, enrollment-time references** instead of only noisy passive accumulation. This is a high-leverage architectural change for the future pipeline.

## Inputs & Dependencies

| Item | Detail |
|------|--------|
| **Video** | Frames from the source video; ideally a short “clear” segment or user-picked frame. |
| **Detection** | PLAN_02 / PLAN_03 — boxes to localize dancers in the enrollment frame. |
| **Masks** | PLAN_04 — SAM2 masks for mask-isolated crops (preferred). |
| **Bootstrap note** | Enrollment can be prototyped with YOLO + existing SAM2 refiner before PLAN_04 is fully merged, but production path assumes stable masks from the primary tracker. |

## Step-by-Step Implementation

1. Create `sway/enrollment.py`.

2. **Dataclass `DancerGallery`** (per dancer):

   | Field | Type / purpose |
   |-------|----------------|
   | `dancer_id` | `int` |
   | `name` | `Optional[str]` — from Lab UI |
   | `part_embeddings` | `Dict[str, np.ndarray]` — keys: `head`, `torso`, `upper_arms`, `lower_arms`, `upper_legs`, `lower_legs` (PLAN_08 BPBreID or fallback) |
   | `global_embedding` | `np.ndarray` — fallback if part model unavailable |
   | `face_embedding` | `Optional[np.ndarray]` — PLAN_11 |
   | `color_histograms` | `Dict[str, np.ndarray]` — keys: `upper`, `lower`, `shoes` (PLAN_12) |
   | `skeleton_gait_embedding` | `Optional[np.ndarray]` — filled after 30–60 frames of tracking (PLAN_10) |
   | `reference_mask_area` | `float` — for state machine / scale checks |
   | `spatial_position` | `Tuple[float, float]` — normalized `(cx/W, cy/H)` |
   | `enrollment_frame` | `int` |

3. **`auto_select_enrollment_frame(video_path, detector) -> int`:**
   - Scan first 300 frames (~10 s @ 30 fps).
   - Each frame: run detector; compute pairwise distances between detection centers.
   - Choose frame where `min(pairwise_distances) > SWAY_ENROLLMENT_MIN_SEPARATION_PX` (default `80`) **and** detection count equals expected dancer count (if configured) **or** is maximum among candidates.
   - Return chosen frame index.

4. **`enroll_dancers(frame, detections, sam2_masks, models) -> List[DancerGallery]`:**
   - Per detection / mask pair: mask-isolated crop (zero outside mask).
   - Extract part embeddings (PLAN_08 BPBreID or OSNet fallback), color histograms (PLAN_12), face embedding if face detected (PLAN_11).
   - Record normalized spatial position and `reference_mask_area`.
   - Assign `dancer_id` sequentially from 1.
   - Return list of galleries.

5. **Persistence:** `save_gallery(galleries, path)` and `load_gallery(path)` — JSON + numpy as base64 (or equivalent stable serialization). Gallery should reload for the same video / project.

6. **Deferred gait update:** After first ~60 frames of tracking, compute MoCos gait embedding (PLAN_10) once per dancer and patch `skeleton_gait_embedding` — one-shot update, not EMA.

7. **Pipeline Lab API:** `POST /api/enrollment/select` — body: frame index + list of `{ dancer_id, name, bbox }` from user clicks; response: serialized gallery (or path to saved gallery).

8. **Pipeline Lab UI (React):** Show auto-selected frame; user confirms or changes frame; click each dancer, enter name, confirm; POST to backend.

9. **Environment:**
   - `SWAY_ENROLLMENT_ENABLED` — default `1` (on).
   - `SWAY_ENROLLMENT_AUTO_FRAME` — default `0` means use auto-select; non-zero can mean fixed frame index (document exact semantics in code).

10. **`main.py` (or orchestrator):** Run enrollment before the main tracking loop; pass `List[DancerGallery]` into tracker, re-ID, and state machine.

## Technical Considerations & Performance

- Gallery size is small (~order 1 KB per dancer for embeddings + histograms); keep in memory, optional disk cache — no DB required for MVP.
- Enrollment cost: roughly one frame of detector + SAM2 + feature extractors (~order 2 s wall time — measure on target GPU).

## Edge Cases & I/O

- **No good frame in first 300:** Warn in Lab UI; offer manual frame selection; optional fallback to passive gallery accumulation (legacy behavior).
- **Late-entering dancers:** New detections that do not match enrolled IDs → new gallery entry via detector re-invocation path (document ID assignment policy).

## Validation & Testing

- Enroll on bigtest (frame 0 or auto-selected). Verify: (a) all expected dancers enrolled with separable embeddings, (b) save/load round-trip, (c) downstream modules read gallery not only passive bank.
- **Targets:** Identify all N dancers in clip; gallery-based re-ID ≥ ~90% with OSNet (baseline without enrollment ~80%); with PLAN_08 parts, target ≥ ~95% (project-defined eval protocol).

## Integration & Next Steps

| Consumer | Role |
|----------|------|
| PLAN_10 | Updates `skeleton_gait_embedding` after warm-up tracking. |
| PLAN_13 | Re-ID fusion compares against gallery. |
| PLAN_14 | Pose-gated EMA may update gallery embeddings under gates. |
| Lab UI | Manual enrollment and overrides. |

**Outputs:** `DancerGallery`, enrollment pipeline, persistence, API + UI hooks, env flags.

## Open Questions / Risks

- Full value needs Lab UI; **auto-only path** must work unattended for batch runs — test auto-select on all ground-truth sequences (including tight formations).

---

*Standalone plan — no dependency on external research docs for execution.*
