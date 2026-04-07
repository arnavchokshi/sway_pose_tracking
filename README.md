# SWAY Pose MVP Pipeline (v23) - In-Depth Technical README

This README documents what the current production-style pipeline actually does, phase by phase, using the exact model names/checkpoints and the real runtime logic from `sway_pose_mvp/tools/run_pipeline_v23_bigtest.py`.

The pipeline is designed for crowded, high-occlusion dance footage where identity swaps and limb confusion are common. It prioritizes correctness and temporal consistency over raw speed.

## Where the Pipeline Lives

- Main executable pipeline script: `sway_pose_mvp/tools/run_pipeline_v23_bigtest.py`
- Core modules are in `sway_pose_mvp/sway/`
- Default model directory (overrideable): `sway_pose_mvp/models/`

## Architecture at a Glance

The pipeline runs in this order:

1. Phase 1 - Detection
2. Phase 2 - Masking
3. Phase 3 - Dual pose pre-track evidence
4. Phase 3.5 - Ambiguity resolver (feature-flagged)
5. Phase 4 - Bidirectional tracking (+ early track pruning)
6. Phase 5 - Deferred enrollment (identity gallery creation)
7. Phase 6 - Multi-signal ReID fusion
8. Phase 7 - Dark-zone graph stitching and alias consolidation
9. Legacy compatibility block - 2D pose export + 3D lifting compatibility artifacts
10. Phase 8 - Final global consistency optimization

Each phase is intentional: earlier phases maximize recall and preserve alternatives; later phases add increasingly strict temporal and identity constraints.

## Exact Models and Checkpoints Used

These are the exact model names/checkpoints used in the current code path:

- Detector:
  - `yolo26l_dancetrack.pt` (fixed policy, no fallback allowed)
  - Provenance label written as `yolo26l_dancetrack`
- Segmentation:
  - Primary: `sam2.1_hiera_large.pt`
  - Fallbacks: `sam2.1_l.pt` then `sam2.1_b.pt`
  - Loaded through `ultralytics.SAM`
- Pose (top-down):
  - ViTPose wrapper model name: `usyd-community/vitpose-plus-large`
- Pose (complementary):
  - `RTMW384HybridEstimator` (rendered/logged as RTMW-X, includes full-joint output path)
- Tracker/ReID backbone for tracking:
  - BoxMOT backend default: `deepocsort`
  - Tracker ReID weights: `osnet_x0_25_msmt17.pt`
- Identity embeddings:
  - Part/global ReID: BPBreID checkpoint `bpbreid_r50_market_msmt17.pth` via `torchreid` (strict preflight)
  - Face embedding: ArcFace `buffalo_l` via InsightFace `FaceAnalysis`
  - Color descriptor: histogram extractor
- ReID fusion logic:
  - `ReIDMLPRouter` for dynamic signal weighting
  - Hungarian assignment (`scipy.optimize.linear_sum_assignment`)
- Dark-zone continuity:
  - `CoalescenceDetector` + `NOOUGATGraphStitcher`
- 3D compatibility stage:
  - `lift_poses_v23` from `sway.mamba_ssm_lifter`

Why this exact stack works in crowds:
- `yolo26l_dancetrack` gives stable person proposals for dance motion patterns.
- SAM2.1 provides instance geometry so pose and enrollment avoid background/reflection contamination.
- ViTPose + RTMW-X gives complementary confidence behavior under occlusion (robustness from redundancy).
- DeepOcSort + OSNet keeps short-term motion continuity while BPBreID/face/color fix long-range identity drift.
- Dark-zone graph stitching explicitly repairs identity paths through merges/splits and occlusion events.

## Phase-by-Phase Deep Dive

## Phase 1: Detection

What it does:
- Runs detector on every frame using YOLO wrapper with:
  - class filter: person only (`classes=[0]`)
  - detection confidence threshold: `0.3`
- Deduplicates detections and writes:
  - `phase1_detection.mp4`
  - `detections_phase1.json`
- Stores provenance per frame as `yolo26l_dancetrack`.

Exact model:
- `yolo26l_dancetrack.pt` from `models/` (hard-required).

Key runtime knobs:
- `SWAY_PHASE1_HEARTBEAT_EVERY` (progress logging cadence).

Why this phase exists:
- The rest of the pipeline is track/identity/pose constrained by these boxes.
- High-recall person detection is the foundation: missing a person here is unrecoverable downstream.
- A fixed detector policy (no detector arbitration) avoids non-deterministic handoffs between detection families.

## Phase 2: Pixel-Perfect Masking (SAM2.1)

What it does:
- Builds per-detection instance masks with multiple policies:
  1. Full SAM run on frame 0, stride frames, or overlap frames.
  2. Mask reuse when IoU with prior detection >= `SWAY_MASK_REUSE_IOU` (default `0.70`).
  3. Mask propagation by geometric shift when IoU is moderate (>= `0.30`).
  4. Optional selective masking: run SAM mainly on overlapping detections, bbox-fill for easy isolated detections.
- Writes:
  - `phase2_masks.mp4`
  - `masks_phase2.json` with quality values.

Exact model:
- `sam2.1_hiera_large.pt` -> fallback `sam2.1_l.pt` -> fallback `sam2.1_b.pt`.
- SAM3 tracker path is disabled by policy in this pipeline.

Key runtime knobs:
- `SWAY_DISABLE_SAM_MASKING`
- `SWAY_SAM2_WEIGHTS`
- `SWAY_PHASE2_HEARTBEAT_EVERY`
- `SWAY_MASK_FRAME_STRIDE`
- `SWAY_MASK_REUSE_IOU`
- `SWAY_SELECTIVE_MASKING`

Why this phase exists:
- In dense choreography, raw boxes include other people and reflections.
- Instance masks improve downstream feature purity (pose and ReID both become less contaminated).
- Reuse/propagation preserves temporal continuity and cuts SAM cost while keeping overlap robustness.

## Phase 3: Dual Pose Pre-Track Evidence (ViTPose + RTMW-X)

What it does:
- For every detection, runs both pose backends using crop-aligned masks:
  - ViTPose (`usyd-community/vitpose-plus-large`)
  - RTMW-X (`RTMW384HybridEstimator`)
- Stores dual evidence per detection:
  - `vitpose_keypoints`
  - `rtmwx_keypoints`
  - `rtmwx_keypoints_full`
  - mean confidence stats for each model
- Produces visual artifacts:
  - `phase3_vitpose.mp4`
  - `phase3_rtmwx.mp4`
  - `phase3_vitpose_overlay.json`
- Enforces a hard integrity check:
  - maximum observed RTMW full-joint cardinality must be >= 100.

Exact models:
- ViTPose model id: `usyd-community/vitpose-plus-large`
- RTMW implementation: `RTMW384HybridEstimator`

Key runtime knobs:
- `SWAY_PHASE3_VIZ_MIN_POINT_CONF` (default `0.03`)
- `SWAY_PHASE3_VIZ_MIN_BONE_CONF` (default `0.05`)
- `SWAY_PHASE3_VIZ_ADAPT_QUANTILE` (default `0.70`)

Why this phase exists:
- No single pose head is optimal in all occlusion types.
- Dual evidence gives redundancy and optional arbitration inputs for enrollment/re-ID/phase 3.5.
- Confidence-graded overlays expose failure modes early (critical for crowded-scene debugging).

## Phase 3.5: Ambiguity Resolver (Optional, Sidecar-Safe)

What it does:
- Optional disambiguation pass over phase-3 evidence.
- Enabled only when:
  - `SWAY_PHASE35_ENABLED=1`
  - `SWAY_PHASE35_MODE` is `shadow` or `active`
- In `shadow` mode: computes decisions, writes diagnostics, does not mutate pipeline state.
- In `active` mode: can replace `pretrack_pose_by_frame_det` with resolved map.
- Supports fail-open:
  - if errors/timeouts occur and `SWAY_PHASE35_FORCE_FAIL_OPEN=1`, baseline phase-3 data is retained.
- Writes:
  - `phase3_5_disambiguation.json`
  - `phase3_5_metrics.json`

Core gating config (exact env family):
- `SWAY_PHASE35_IOU_THRESH`
- `SWAY_PHASE35_LOW_CONF_THRESH`
- `SWAY_PHASE35_MIN_MARGIN`
- `SWAY_PHASE35_TEMPORAL_WEIGHT`
- Hair/hand guard controls:
  - `SWAY_PHASE35_HAIR_HAND_GUARD`
  - `SWAY_PHASE35_HAIR_HAND_CONF_THRESH`
  - `SWAY_PHASE35_HAIR_HAND_RT_CONF_THRESH`
  - `SWAY_PHASE35_HAIR_HAND_DISAGREE_PX`
  - `SWAY_PHASE35_HAIR_HAND_ARM_RATIO_MAX`
  - `SWAY_PHASE35_HAIR_HAND_TEMPORAL_JUMP_PX`
  - `SWAY_PHASE35_HAIR_HAND_LOCK_FRAMES`
  - `SWAY_PHASE35_HAIR_HAND_REPLACE_ELBOW`

Why this phase exists:
- Crowded frames produce ambiguous local evidence even with dual pose.
- Sidecar + shadow-first mode lets you validate gains safely before making decisions authoritative.
- Fail-open ensures no catastrophic regressions in long runs.

## Phase 4: Bidirectional Tracking + Early Structural Pruning

What it does:
- Forward pass tracking on detections:
  - default backend `deepocsort` (BoxMOT), with dynamic backend compatibility wrappers.
- Backward pass tracking over reversed frame order.
- Writes:
  - `phase4_tracking_forward.mp4`
  - `phase4_tracking_bidirectional.mp4`
  - `tracklets_forward.json`
  - `tracklets_backward.json`
  - `data.json` (MOT-style export)
- Calculates contamination flags when track boxes overlap heavily (IoU > 0.35).
- Runs optional tracker A/B benchmark slices in overlap-heavy windows.
- Applies "Solution C" early pruning before ReID:
  1. prune short low-confidence tracks
  2. prune reflection-like tracks using edge-presence + motion-sign conflict + relative height
  3. optional formation over-cap pruning against performer-cap expectations

Exact tracker models/checkpoints:
- BoxMOT backend default: `deepocsort`
- Tracker ReID checkpoint: `osnet_x0_25_msmt17.pt`

Key runtime knobs:
- Tracker family:
  - `SWAY_TRACKER_BACKEND`, `SWAY_TRACKER_DET_THRESH`, `SWAY_TRACKER_MAX_AGE`,
  - `SWAY_TRACKER_IOU_THRESHOLD`, `SWAY_TRACKER_DELTA_T`, `SWAY_TRACKER_INERTIA`,
  - `SWAY_TRACKER_W_ASSOC_EMB`
- A/B testing:
  - `SWAY_TRACKER_AB`, `SWAY_TRACKER_AB_BACKENDS`, `SWAY_TRACKER_AB_MIN_OVERLAP_FRAMES`
- Early prune:
  - `SWAY_SHORT_TRACK_MIN_FRAMES`, `SWAY_SHORT_TRACK_MIN_CONF`
  - `SWAY_REFLECTION_EDGE_MARGIN_FRAC`, `SWAY_REFLECTION_EDGE_PRESENCE_FRAC`
  - `SWAY_REFLECTION_MIN_SIGN_CONFLICT_FRAC`, `SWAY_REFLECTION_MAX_HEIGHT_FRAC`

Why this phase exists:
- Identity logic is only as good as track continuity.
- Bidirectional passes expose recoverable misses and improve tracklet topology.
- Early ghost/reflection pruning prevents false tracks from polluting enrollment and ReID association.

## Phase 5: Deferred Enrollment (Identity Gallery Build)

What it does:
- Enrollment is intentionally delayed until after tracking + dual pose evidence exist.
- Selects the "richest" frame (max active tracks) as enrollment anchor.
- Re-extracts SAM masks for enrollment candidates.
- Filters candidates by:
  - SAM mask quality (`SWAY_ENROLL_SAM_MASK_MIN_QUALITY`, default `0.20`)
  - either ViTPose or RTMW mean confidence thresholds
    - `SWAY_ENROLL_VITPOSE_MIN_CONF` (default `0.35`)
    - `SWAY_ENROLL_RTMWX_MIN_CONF` (default `0.35`)
- Builds gallery using:
  - part ReID (BPBreID extractor),
  - color histogram,
  - face embedding,
  - selected keypoints (ViTPose or RTMW, whichever has stronger confidence per candidate).
- Writes:
  - `gallery_identity_bank.json`

Exact models involved:
- BPBreID `bpbreid_r50_market_msmt17.pth`
- ArcFace `buffalo_l`
- Phase-3 keypoints from ViTPose/RTMW-X
- SAM2.1 masks

Why this phase exists:
- Early-frame enrollment is often low quality in crowd videos (motion blur, occlusion).
- Delayed enrollment uses stronger evidence and yields cleaner identity prototypes.
- Better prototypes directly reduce switch rate in subsequent Hungarian matching.

## Phase 6: Omni-Fusion Multi-Signal ReID

What it does:
- For each active track per frame, extracts and fuses:
  - global BPBreID embedding
  - color histogram descriptor
  - face embedding (stride-based extraction)
  - part-level embeddings (cached with IoU+TTL reuse)
  - spatial prior score
  - motion consistency score
- Computes weighted identity score matrix and solves assignment with Hungarian algorithm.
- Applies anti-switch logic:
  - cooldown gate (`SWAY_REID_SWITCH_COOLDOWN_FRAMES`)
  - hysteresis/lock state machine
  - contamination-aware conservative retention
  - cold-start spatial bonus
- Writes:
  - `phase6_reid_fusion.mp4`
  - `phase6_identity_assignments_reid.json` (+ alias copy)

Key runtime knobs:
- `SWAY_FACE_EMBED_STRIDE`
- `SWAY_PART_CACHE_TTL`, `SWAY_PART_CACHE_IOU`
- `SWAY_REID_TEMPORAL_WINDOW`
- `SWAY_REID_COLD_START_FRAMES`, `SWAY_REID_COLD_START_BONUS`
- `SWAY_REID_LOCK_CONFIRM_FRAMES`, `SWAY_REID_LOCK_SWITCH_PENALTY`, `SWAY_REID_UNLOCK_MARGIN`
- `SWAY_REID_SWITCH_MARGIN`, `SWAY_REID_SWITCH_COOLDOWN_FRAMES`

Why this phase exists:
- In dense overlap, any single feature stream (appearance only, face only, color only) fails in specific scenarios.
- Multi-signal fusion with temporal lock/cooldown is the main defense against identity thrash.
- Hungarian gives globally consistent one-to-one assignment per frame, reducing local greedy mistakes.

## Phase 7: Dark-Zone Resolution (Graph Stitching)

What it does:
- Detects coalescence events (tracks merging in overlap/dark zones) and later exits.
- Builds entry/exit tracklet nodes with embedding + trajectory context.
- Resolves identity continuation with graph stitching.
- Integrates backward pass tracklets via bbox IoU mapping to fill forward gaps.
- Applies hierarchical merge when available.
- Enforces "hard exclusion" on stitches that conflict with high-confidence gallery identity assignments.
- Builds post-stitch assignments (`reid_assignments_phase4`), optionally replacing with formation-constrained assignment logic when in formation mode.
- Writes:
  - `phase7_darkzone_resolution.mp4`

Exact algorithms/components:
- `CoalescenceDetector(iou_thresh=0.25, consecutive_frames=2)`
- `NOOUGATGraphStitcher`
- Optional formation alignment/fusion modules from `formation_identity`.

Why this phase exists:
- Classical frame-local ReID cannot reason through temporary track disappearance/merges.
- Graph-based entry/exit linking repairs continuity across occlusion dark zones.
- Hard exclusions prevent stitch operations from undoing high-confidence identity evidence.

## Legacy Compatibility Block (Pose/3D Artifacts)

What it does:
- Re-runs tracked-pose extraction for compatibility exports.
- Keeps ViTPose confidence values while computing geometry-validity diagnostics against mask/box.
- Applies temporal keypoint smoothing (`SWAY_TEMPORAL_POSE_REFINE`, radius configurable).
- Exports:
  - `pose2d_phase6.json` (keypoints + visibility + confidence-state + geometric validity flags)
- Runs 3D lifting:
  - forward `lift_poses_v23`
  - backward pass + confidence-weighted blend
- Writes:
  - `phase7_legacy_pose3d_compat.mp4`

Why it still exists:
- Some downstream tools depend on historical artifact contracts.
- This block preserves compatibility while Phase 8 uses improved identity-corrected render logic.

## Phase 8: Final Global Identity/Joint Optimization

What it does:
- Applies final correction passes to identity assignments:
  1. same-track temporal consistency correction
  2. kinematic jump correction (large displacement + low confidence)
  3. cross-track neighbor-vote correction over temporal window
- Uses configurable confidence and gating thresholds:
  - `SWAY_PHASE8_CONF_THRESH` (default `0.90`)
  - `SWAY_PHASE8_KIN_CONF_THRESH` (default `0.70`)
  - `SWAY_PHASE8_XTRACK_CONF_THRESH` (default `0.88`)
  - `SWAY_PHASE8_XTRACK_GATE_PX` (default `width*0.12`)
  - `SWAY_PHASE8_NEIGHBOR_WINDOW` (default `4`)
  - `SWAY_PHASE8_NEIGHBOR_MIN_RATIO` (default `0.50`)
  - `SWAY_PHASE8_XTRACK_TARGET_APPLY_RATIO`
- Rejects unsafe changes with explicit reason logging.
- Renders final outputs with strict source precedence:
  - final corrected assignment
  - phase4 assignment fallback
  - sticky last-valid identity fallback
  - unknown otherwise
- Writes:
  - `phase8_final_optimized.mp4`
  - `final_identity_tracks.json`
  - `phase8_reject_log.json`

Why this phase exists:
- Even after ReID + stitching, short local errors remain.
- This phase is a conservative global consistency pass: only low-confidence or contradiction cases are corrected.
- The reject-log telemetry enables data-driven threshold tuning instead of blind heuristic changes.

## Why the Whole Pipeline Works (Design Justification)

The key principle is layered uncertainty reduction:

- Phase 1-3 maximize recall and preserve competing hypotheses (do not over-prune early).
- Phase 4-6 resolve identity with multi-signal evidence and temporal memory.
- Phase 7 reasons through occlusion topology (merge/split events) rather than frame-local similarity alone.
- Phase 8 performs conservative corrections only where temporal and geometric evidence support them.

This ordering is important. If strict identity locking happens too early, the system cements mistakes. If global correction happens without prior multi-signal evidence, corrections become unstable. The current sequence is built to avoid both failure modes.

## Important Artifact Outputs

Primary outputs for debugging and analysis:

- `phase1_detection.mp4`, `detections_phase1.json`
- `phase2_masks.mp4`, `masks_phase2.json`
- `phase3_vitpose.mp4`, `phase3_rtmwx.mp4`, `phase3_vitpose_overlay.json`
- `phase3_5_disambiguation.json`, `phase3_5_metrics.json` (when enabled)
- `phase4_tracking_forward.mp4`, `phase4_tracking_bidirectional.mp4`, `tracklets_forward.json`, `tracklets_backward.json`, `data.json`
- `gallery_identity_bank.json`
- `phase6_reid_fusion.mp4`, `phase6_identity_assignments_reid.json`
- `phase7_darkzone_resolution.mp4`
- `pose2d_phase6.json`, `phase7_legacy_pose3d_compat.mp4`
- `phase8_final_optimized.mp4`, `final_identity_tracks.json`, `phase8_reject_log.json`
- Diagnostics: `evaluation_metrics.json`, `switch_event_log.json`, `run_manifest.json`

## Operational Notes

- If `SWAY_STOP_AFTER_PHASE3=1`, the run exits after tracking-side exports (for fast A/B and detector/tracker evaluation).
- If `SWAY_STOP_AFTER_PHASE4=1`, the run exits after dark-zone/formation-level identity pass.
- Phase 3.5 is built for safe rollout (`off`/`shadow`/`active` + fail-open).
- ReID path is strict by design: BPBreID preflight must pass (checkpoint exists and dimension check passes).
