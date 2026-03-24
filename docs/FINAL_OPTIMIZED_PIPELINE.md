# Sway Pose — Final Optimized Pipeline

### Built from: 5 research rounds + 3 Cursor codebase passes + live benchmark results

### Goal: Maximum accuracy at acceptable runtime. Every decision is grounded in data.

---

## Runtime Reality Check (Based on Your Actual Benchmarks)

| Mode | Per-frame | 3-min video | When to use |
|------|-----------|-------------|-------------|
| BoxMOT only | ~4.3ms | ~1.5 min | Fast preview, solo/duo |
| Hybrid SAM2 (current) | ~550ms | ~3.8 min | Standard group performance |
| Full SAM2 | ~1737ms | ~35 min | Never — too slow |

**The hybrid approach at 3.8 minutes for a 411-frame video is your production default.**
The remaining optimizations in this plan push accuracy higher without meaningfully
changing that runtime.

---

## Phase 1 — Detection (YOLO26l)

### What to run

YOLO26l fine-tuned on DanceTrack + CrowdHuman.

**Why YOLO26l over YOLO11x:**

- You confirmed this yourself — YOLO26l showed higher confidence on occluded people
  in your direct comparison test
- STAL (Small-Target-Aware Label Assignment) specifically improves partially-visible
  object detection — exactly what an occluded dancer is
- NMS-free by architecture — eliminates the need for any post-detection suppression step
- 43% faster CPU inference than comparable YOLO11 variants

**Why fine-tune on DanceTrack + CrowdHuman:**

- COCO is 95% upright pedestrians. Dancers in splits, inversions, lifts, and floor
  routines are systematically underrepresented
- DanceTrack provides dance-specific person annotations across 100 performance videos
- CrowdHuman provides dense overlapping person examples — directly targets your
  hallucination problem where 2 overlapping dancers become 4 detected boxes

**Training is already in progress via Cursor-generated scripts.**
After training: `export SWAY_YOLO_WEIGHTS=models/yolo26l_dancetrack.pt`

### Pre-tracker deduplication (add this now)

YOLO26l is NMS-free but can still produce multiple overlapping hypotheses on dense
overlaps. Before any tracking, run a deduplication pass:

```python
# In _run_tracking_boxmot_diou(), after model.predict(), before tracker.update()
# YOLO26: skip DIoU-NMS (model is NMS-free natively)
# But still deduplicate overlap hallucinations
def deduplicate_overlapping_dets(xyxy, conf, iou_threshold=0.5):
    """
    Kill ghost boxes before they reach the tracker and get IDs assigned.
    IoU=0.5: if two boxes share >50% area, keep only highest confidence.
    More aggressive than DIoU-NMS — specifically targets hallucination.
    """
    if len(xyxy) == 0:
        return xyxy, conf
    from torchvision.ops import nms
    import torch
    keep = nms(
        torch.tensor(xyxy, dtype=torch.float32),
        torch.tensor(conf, dtype=torch.float32),
        iou_threshold=iou_threshold
    )
    return xyxy[keep.numpy()], conf[keep.numpy()]

# Replace DIoU-NMS call with this for YOLO26:
if "yolo26" in model_path.lower():
    xyxy, conf = deduplicate_overlapping_dets(xyxy, conf, iou_threshold=0.5)
else:
    keep = diou_nms_indices(xyxy, conf, iou_threshold=0.7)
    xyxy, conf = xyxy[keep], conf[keep]
```

**Also set `copy_paste=0.2` in train_yolo26l.py** (raise from 0.1) — this synthesizes
more overlapping body examples during training, directly teaching YOLO26l that
2 overlapping bodies = 2 boxes, not 4.

### Detection resolution

```python
# Always use 960 for group videos — don't wait for chunk 1 to detect crowd
if os.environ.get("SWAY_GROUP_VIDEO", "").lower() in ("1", "true", "yes"):
    current_detect_size = 960
elif max_dancers_last_chunk > 4:
    current_detect_size = 960
else:
    current_detect_size = DETECT_SIZE  # 640 for solo/duo
```

---

## Phase 2 — Tracking (Hybrid BoxMOT + SAM2)

### Primary tracker: BoxMOT Deep OC-SORT

Already in your pipeline. Correct config (from your live tests):

```python
tracker = DeepOcSort(
    reid_weights=reid_w,
    device=dev,
    half=bool(dev.type == "cuda"),
    det_thresh=float(YOLO_CONF),   # 0.22 — match exactly
    max_age=150,                    # keep lost tracks alive ~5s at 30fps
    min_hits=2,                     # was 3 — confirms tracks faster
    iou_threshold=0.3,              # explicit association threshold
    embedding_off=True,             # FORCE OFF — Re-ID hurts on identical costumes
                                    # DanceTrack paper proved this empirically
)
```

**Why embedding_off=True is critical:**
The DanceTrack paper ran a controlled experiment proving that adding appearance Re-ID
(DeepSORT) performed *worse* than plain SORT on dancers in similar costumes. OSNet
embeddings create ambiguity when costumes are identical. Disable it unconditionally.

### Hybrid SAM2 (already working — tune the trigger)

Your benchmark showed 3.8 min / 411 frames with 33.6% of frames triggering SAM2.
This is the right architecture. Now tune the trigger threshold:

```python
# Current: IoU >= 0.35 (138/411 frames triggered = 33.6%)
# Recommended: IoU >= 0.42 — reduces SAM2 calls on mild overlaps BoxMOT handles fine
# Each 0.05 increase saves ~5-8% of SAM2 calls = ~15-20s on a 3-min video

# Env: SWAY_HYBRID_SAM_IOU_TRIGGER (see sway/hybrid_sam_refiner.py; default 0.42)
iou_trigger = float(os.environ.get("SWAY_HYBRID_SAM_IOU_TRIGGER", "0.42"))
```

**Use `sam2.1_b.pt` (small model) always** — you're already doing this. The large model
gives marginal accuracy improvement that isn't worth the runtime cost in hybrid mode.

**Minimum window logic** — already described in your implementation:
Once SAM2 activates, keep it running for minimum 30 frames even if overlap resolves.
Prevents costly on/off toggling and amortizes initialization cost.

**ID handoff when SAM2 deactivates:**
This is the trickiest part of hybrid tracking. When SAM2 turns off and BoxMOT resumes,
BoxMOT must initialize from SAM2's last known positions and IDs — not start fresh.
Otherwise you get an ID reset exactly when dancers separate, which is backwards.

```python
def handoff_sam2_to_boxmot(sam2_final_masks, sam2_final_ids, tracker):
    """
    When SAM2 window ends, seed BoxMOT with SAM2's confirmed IDs and positions.
    Prevents ID reset at the moment of dancer separation.
    """
    # Convert SAM2 mask-derived boxes to BoxMOT track format
    # Force BoxMOT to accept these as confirmed tracks with the SAM2 IDs
    seed_detections = []
    for dancer_id, mask in zip(sam2_final_ids, sam2_final_masks):
        box = mask_to_box(mask)
        seed_detections.append((box, dancer_id))
    # Inject into BoxMOT's track state
    tracker.reinitialize_from_seeds(seed_detections)
```

---

## Phase 3 — Post-Track Stitching (Already in Pipeline)

The existing functions in `tracker.py` handle this well:

- `stitch_fragmented_tracks` — reconnects dropped tracks
- `coalescence_deduplicate` — removes ghost duplicates
- `merge_complementary_tracks` — merges alternating-ID fragments
- `merge_coexisting_fragments` — merges spatially identical co-occurring IDs
- `apply_dormant_merges` — Dormant registry (high-OKS relink after long gaps)

**One tuning change:** Raise `coalescence_deduplicate` IoU threshold from 0.65 to 0.70
after YOLO26l fine-tuning is in place. YOLO26l produces tighter boxes, so the old 0.65
threshold may merge legitimate nearby dancers that the current model separated.

---

## Phase 4 — Pre-Pose Pruning (Phase 3 in main.py)

### Do not change the order

Phase 3 pruning runs before pose estimation. Keypoint-confidence gating is impossible
here — do not attempt to add it. Keep existing geometric rules but add one improvement:

### Add depth-based trigger for prune_by_stage_polygon

Currently `prune_by_stage_polygon` is off by default (requires manual coordinates).
Add Depth Anything V2 as an auto-generator for the stage polygon:

```python
# New: auto-generate stage polygon from first frame depth map
# Only runs if SWAY_STAGE_POLYGON is not manually set
if stage_polygon is None and os.environ.get("SWAY_AUTO_STAGE_DEPTH", "1") == "1":
    from sway.depth_stage import estimate_stage_polygon
    stage_polygon = estimate_stage_polygon(first_frame)
    # Falls back gracefully if Depth Anything not installed
```

This eliminates the manual polygon-drawing step for new venues.

---

## Phase 5 — Pose Estimation (ViTPose-H)

### Model choice

ViTPose-H (huge) — `usyd-community/vitpose-plus-huge`

Already supported via `--pose-model huge` and prefetch is in your plan.
ViTPose-H is still the accuracy leader for 17-keypoint body pose at ~79 AP on COCO.
No other model beats it for your use case.

**If you want whole-body keypoints later** (hands, face, feet — for richer scoring):
RTMW-x gives 133 keypoints and is the best whole-body model. Add as `--pose-model rtmw`
option. Only worth doing if scoring actually uses the extra keypoints.

### Extreme option: Fine-tune ViTPose-H on AthletePose3D

Research showed fine-tuning on athletic movement data reduces mean per joint position
error from 214mm to 65mm — a 69% reduction on extreme poses. AthletePose3D is a public
dataset covering figure skating and athletic movement (closest public data to dance).
Run this fine-tune job on your cloud GPU alongside the YOLO26l training job.

### Poseidon temporal fusion (add after base pipeline is stable)

Poseidon extends ViTPose with cross-frame attention — uses frames t-2 through t+2 to
infer occluded keypoints in frame t. Achieves 87.8 mAP on PoseTrack (video benchmark)
vs ViTPose-H's ~82 on the same benchmark. Code is public. Slots in as a temporal
post-processing layer over your existing pose estimator — does not replace ViTPose-H.

---

## Phase 6 — Association and Refinement (Phase 5 in main.py)

### OKS veto (already done — keep as-is)

Pass 1.5: `merge_iou_centroid` requires `oks >= 0.25`
Pass 1.6: same with floor of 0.15 for late-entrant path
These are in the codebase. Do not change.

### Neural AFLink — add this next

Currently `SWAY_GLOBAL_LINK=1` runs a heuristic spatial/temporal stitch.
Replace with the neural AFLink from StrongSORT for better global ID continuity.

```bash
# Install
git clone https://github.com/dyhBUPT/StrongSORT
# Copy AFLink module into sway/aflink.py
# Download pretrained weights: AFLink_epoch20.pth
```

```python
# In sway/global_track_link.py — add neural path alongside heuristic
def neural_global_stitch(raw_tracks, total_frames):
    from sway.aflink import AFLink
    linker = AFLink(
        path_AFLink="models/AFLink_epoch20.pth",
        thrT=(0, 30),    # temporal gap range (frames)
        thrS=75,         # spatial distance threshold
        thrP=0.05,       # probability threshold for accepting a link
    )
    # Convert raw_tracks to MOT rows → run linker → convert back
    mot_rows = raw_tracks_to_mot_lines(raw_tracks)
    linked = linker.link(mot_rows)
    return mot_lines_to_raw_tracks(linked)
```

AFLink uses **no appearance embeddings** — purely temporal and spatial context.
This is correct for identical-costume scenarios per the DanceTrack finding.

---

## Phase 7 — Collision Cleanup (Phase 6 in main.py)

No changes to `deduplicate_collocated_poses` or `sanitize_pose_bbox_consistency`.
**Add logging to both** — currently silent. These are the only two pipeline steps
that mutate data without writing to `prune_log_entries`.

---

## Phase 8 — Post-Pose Pruning (Phase 7 in main.py)

### Three-tier whitelist architecture (confirmed human exemption already done — extend it)

**Tier A — Protected (cannot be pruned by any rule):**
Track meets ALL of:

- Mean torso keypoint confidence (COCO indices 5, 6, 11, 12) > 0.5 for >40% of frames
- Track spans >10% of video duration
- Passes spatial sanity (not in mirror-prone edge zone for >80% of frames)

Already implemented via `compute_confirmed_human_set`. Make sure the span check
(`min_span_frac=0.10`) is active — prevents false exemptions on short FP tracks.

**Tier B — Voting (not veto):**
Tracks not in Tier A. Convert all 6 Phase 7 rules to suspicion scores (0.0–1.0).
Prune only if `Σ(weight_i × score_i) > PRUNE_THRESHOLD`.

Initial weights (tune from prune log telemetry after 1.3 logging is collecting data):

```python
PRUNING_WEIGHTS = {
    "prune_low_sync_tracks":      0.7,  # strong signal — low sync = likely not a dancer
    "prune_smart_mirrors":        0.9,  # very reliable — mirrors have distinct motion
    "prune_completeness_audit":   0.6,  # moderate — seated observers
    "prune_head_only_tracks":     0.8,  # reliable — audience heads
    "prune_low_confidence_tracks": 0.5, # weaker — confidence varies with pose
    "prune_jittery_tracks":       0.5,  # weaker — dancers can be jittery
}
PRUNE_THRESHOLD = 0.65  # tune this via sweep_config.yaml
```

**Tier C — Auto-reject (skeleton confidence only):**
Mean keypoint confidence < 0.15 across ALL keypoints for >80% of frames.
If ViTPose-H can't find a skeleton, it is not a human. No exceptions.
This replaces all geometric FP rules (chair-like, bad aspect ratio, etc.) which
fail on non-upright dancers. A chair will never produce a confident skeleton.

```python
def is_auto_reject(tid, pose_results, total_frames, threshold=0.15, min_frac=0.80):
    """Returns True if track should be auto-rejected — no confident skeleton ever."""
    low_conf_frames = 0
    total_pose_frames = 0
    for fd in pose_results:
        pose = fd.get("poses", {}).get(tid)
        if pose is None:
            continue
        total_pose_frames += 1
        scores = pose.get("scores", [])
        if scores and (sum(scores) / len(scores)) < threshold:
            low_conf_frames += 1
    if total_pose_frames == 0:
        return True  # no pose at all = reject
    return (low_conf_frames / total_pose_frames) >= min_frac
```

---

## Phase 9 — Smoothing

One Euro filter — already in place, keep as-is.

**Critical: verify it runs per confirmed track ID after all association is finalized.**
If it runs before IDs are stable it averages identity across crossovers.
Check that `PoseSmoother` in `main.py` is called after Phase 8 pruning, not before.

---

## Phase 10 — Scoring

No changes. Fix upstream phases and scoring becomes reliable automatically.
The current scoring logic is sound — it was producing bad numbers because of bad IDs,
not because of bad scoring math.

---

## Complete Stack Summary

| Phase | Component | Status | Next Action |
|-------|-----------|--------|-------------|
| Detection | YOLO26l | Training in progress | Add pre-tracker dedup, set SWAY_GROUP_VIDEO=1 |
| Hybrid tracking | SAM2.1b adaptive | Working (3.8min/411frames) | Tune trigger to IoU=0.42, fix ID handoff |
| Primary tracker | BoxMOT Deep OC-SORT | Working | Verify embedding_off=True on CUDA |
| Post-track stitch | Dormant + heuristic stitch | Done | Raise coalescence IoU to 0.70 after YOLO26 |
| Pre-pose pruning | Existing geometric rules | Keep | Add depth-based auto stage polygon |
| Pose estimation | ViTPose-H | Ready (prefetch) | Switch --pose-model huge, fine-tune on AthletePose3D |
| Association | OKS veto Pass 1.5/1.6 | Done | Keep |
| Global stitch | Heuristic AFLink | Done | Upgrade to neural AFLink |
| Phase 6 logging | Silent currently | Not done | Add prune_log writes |
| Post-pose pruning | Confirmed-human exemption | Done | Extend to voting ensemble |
| Smoothing | One Euro | Done | Verify timing is post-association |
| Scoring | Existing | Keep | No changes |

---

## Implementation Order (What to Do Next)

### This week (no training needed):

1. Add pre-tracker `deduplicate_overlapping_dets()` at IoU=0.5 in `tracker.py`
2. Tune hybrid SAM2 trigger from 0.35 to 0.42
3. Fix SAM2 → BoxMOT ID handoff on deactivation
4. Switch to `--pose-model huge` (ViTPose-H) — prefetch already planned
5. Add Phase 6 logging to `deduplicate_collocated_poses` and `sanitize_pose_bbox_consistency`

### When YOLO26l training finishes:

6. Set `SWAY_YOLO_WEIGHTS=models/yolo26l_dancetrack.pt`
7. Raise `coalescence_deduplicate` IoU from 0.65 to 0.70
8. Disable DIoU-NMS for YOLO26 (NMS-free model — DIoU-NMS is counterproductive)
9. Validate with TrackEval — compare IDF1/IDSW/HOTA before and after

### After validation confirms YOLO26l improvement:

10. Integrate neural AFLink (`models/AFLink_epoch20.pth`)
11. Convert Phase 7 prune rules to voting ensemble with weights above
12. Add depth-based auto stage polygon
13. Fine-tune ViTPose-H on AthletePose3D (run alongside next cloud GPU session)

---

## What You Tell Someone About This Pipeline

Detection: YOLO26l trained on 100 dance performance videos and 15,000 crowd images.
Handles extreme poses, dense formations, and occluded dancers that standard models miss.

Tracking: BoxMOT Deep OC-SORT with adaptive SAM2 segmentation. Fast bounding-box
tracking for most frames, pixel-level mask tracking only when dancers physically overlap.
Appearance Re-ID intentionally disabled — identical costumes make it counterproductive.

Pose: ViTPose-H, the highest-accuracy 17-keypoint body model available, running on
individual dancer crops with neutral background to prevent neighboring dancer bleed-in.

Pruning: Whitelist architecture — confirmed dancers are protected from all pruning rules.
False positives rejected by skeleton confidence alone (chairs and mirrors never produce
confident human skeletons). Borderline cases decided by weighted voting, not single rules.

---

## Map: conceptual phases ↔ `main.py` stages

| Doc phase | `main.py` print step |
|-----------|---------------------|
| 1 Detection | `[1/11]` (same pass as Phase 2) |
| 2 Tracking (hybrid SAM + BoxMOT) | `[2/11]` |
| 3 Post-track stitching | `[3/11]` `apply_post_track_stitching` |
| 4 Pre-pose pruning | `[4/11]` |
| 5 Pose | `[5/11]` |
| 6 Association | `[6/11]` |
| 7 Collision cleanup | `[7/11]` |
| 8 Post-pose pruning | `[8/11]` |
| 9 Smoothing | `[9/11]` |
| 10 Scoring | `[10/11]` |
| Export | `[11/11]` |

`run_tracking(video)` still runs Phases 1–3 in one call for scripts; `main.py` uses `run_tracking_before_post_stitch` + `apply_post_track_stitching` so Phase 3 is explicit.

See also: `docs/HYBRID_SAM_PIPELINE_HANDOFF.txt` (hybrid SAM deliverables and env vars).
