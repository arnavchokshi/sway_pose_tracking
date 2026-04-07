# Highest-Accuracy Pipeline for Crowded, Heavily Occluded Multi-Person Pose

This document defines the research-backed "maximum accuracy" pipeline for 2D body keypoint estimation and tracking in crowded, heavy-occlusion scenes (dance/group performance style footage).

It is intentionally accuracy-first, not latency-first.

## 1) Goal and Design Principles

- Primary goal: maximize per-joint correctness and identity consistency under severe inter-person overlap.
- Secondary goal: preserve temporal stability (low flicker, low ID-switch impact on pose streams).
- Key research takeaway: pure top-down and pure bottom-up both fail under heavy overlap; the best practical strategy combines global disambiguation + per-person refinement + temporal reasoning.

### Why this architecture

- Top-down alone is ambiguous in overlapping boxes (wrong person inside crop).
- Bottom-up alone can mis-associate limbs between nearby people.
- Hybrid approaches (bottom-up cues conditioning top-down refinement, or multi-instance-aware top-down) consistently improve crowded benchmark performance.

## 2) "Perfect" Accuracy-First Pipeline (Stage-by-Stage)

## Stage A: Multi-Person Candidate Discovery (Global)

- Run two complementary person discovery paths per frame:
  - `A1`: strong detector (high recall human detection).
  - `A2`: bottom-up pose proposer (global keypoint evidence across full image).
- Fuse `A1` and `A2` with bipartite matching on box/pose compatibility.
- Keep multiple hypotheses in ambiguous overlap zones (do not over-prune here).

Output:
- Candidate person regions.
- Global pose cues for each candidate (coarse skeleton proposals, confidence maps, occlusion flags).

## Stage B: Occlusion-Aware Instance Disambiguation

- For each candidate crop, predict one or more pose instances using conditioning:
  - Preferred: bottom-up-conditioned top-down refinement (BUCTD-style conditioning signal).
  - Alternative/addition: multi-instance-per-box top-down head (MIPNet-style behavior).
- For each crop, rank pose hypotheses using:
  - keypoint confidence,
  - kinematic plausibility,
  - inter-person exclusivity (penalize duplicate assignment of same limb region to two people).

Output:
- 1..N pose hypotheses per person candidate with ambiguity score.

## Stage C: High-Resolution Pose Refinement

- Use highest-capacity top-down pose backbone feasible on hardware (e.g. ViTPose-G class or strongest available equivalent).
- Use larger input resolution for hard frames and small-person cases.
- Run refinement with soft priors from Stage B (conditioning heatmaps or instance selector), not hard binary cuts.
- Optional second-pass pose correction model (PoseFix-style refinement pass) for low-confidence joints only.

Output:
- Refined 2D keypoints per hypothesis with calibrated per-joint uncertainty.

## Stage D: Mask Usage Policy (Critical)

- Never use hard SAM cutout as the only pose input in crowded occlusion.
- Use masks as soft guidance:
  - keep context ring around person crop,
  - attenuate outside-mask pixels instead of full zero/gray replacement,
  - disable/relax masking automatically when mask quality is low or stale.
- Compute and track mask quality score (coverage consistency, temporal IoU, edge stability).

Output:
- Context-preserving pose input tensors with dynamic masking strength.

## Stage E: Temporal Pose-Track Inference (Video Consistency)

- Track identities using motion + appearance + pose descriptors.
- Solve short temporal windows with global optimization:
  - enforce temporal smoothness,
  - enforce anthropomorphic constraints,
  - fill missing/occluded joints from temporal neighbors with uncertainty propagation.
- Maintain per-joint visibility state machine:
  - visible,
  - self-occluded,
  - inter-person-occluded,
  - missing.

Output:
- Temporally consistent pose tracks with visibility-aware confidences.

## Stage F: Cross-Person and Group-Level Consistency

- In dense overlaps, enforce cross-person constraints:
  - one pixel region should not explain the same limb for multiple identities unless uncertainty supports it.
- Use group-level assignment pass to resolve collisions:
  - prefer assignments that maximize total temporal + kinematic consistency over a short sequence, not only current frame confidence.

Output:
- Reduced limb swapping and reduced ID-coupled pose switches.

## Stage G: Final Confidence Calibration + Export

- Calibrate per-joint confidence with validation-set reliability curves.
- Export:
  - keypoints,
  - per-joint uncertainty,
  - occlusion state,
  - correction provenance (raw/refined/temporal-filled).

This allows downstream systems to trust strong joints and treat occluded joints differently.

## 2.1) Exact Technology Choices by Stage (Accuracy-First)

Use this as the concrete "what to use for what" mapping.

## A) Person Candidate Discovery (Global)

- Primary detector: high-recall person detector fine-tuned on crowded people data (`CrowdHuman` + `CrowdPose` style data).
  - Recommended families: `CO-DETR` / strong transformer detector, or high-capacity `RTMDet-X` class if easier to maintain.
- Detector post-processing:
  - use `Soft-NMS` or weighted box fusion (do not aggressively suppress overlaps),
  - keep lower-score overlapping boxes in crowd zones.
- Bottom-up global pose proposer:
  - preferred: `HigherHRNet-W48` or equivalent strong bottom-up model,
  - alternative: `PETR`/crowd-specialized bottom-up model where available.

Why: detector gives high recall person regions; bottom-up gives global body-part context for disambiguation.

## B) Occlusion Disambiguation in Overlaps

- Primary: `BUCTD`-style conditioning (bottom-up pose proposal conditions top-down refinement).
- Complementary: `MIPNet`-style multi-instance-per-box prediction in ambiguous crops.
- Keep `Top-K` hypotheses (typically K=2 or K=3) only for overlap-heavy crops.

Why: this is the main mechanism that recovers "hidden person behind front person" failure cases.

## C) High-Resolution Top-Down Pose Refinement

- Primary highest-accuracy human model: `ViTPose-G` class at high resolution (e.g., 576x432 style regime when feasible).
- Practical highest-accuracy fallback (if memory-limited): `ViTPose-H` / `ViTPose++-H`.
- For your Hugging Face-based integration:
  - use the strongest available ViTPose checkpoint in your environment (`vitpose-plus-huge` class when supported),
  - run hard frames at a larger crop resolution.

Why: top-down transformer backbones still provide the strongest per-joint precision when ambiguity is already reduced.

## D) Segmentation / Masking Policy

- Segmentation tech: `SAM2` (or strongest SAM variant available in your stack) as a guidance signal.
- Do not hard-cut the person silhouette for pose input.
- Use soft blending outside mask and keep a context ring around the person crop.

Why: hard cutouts remove contextual cues and can clip occluded limbs; soft guidance improves robustness.

## E) Multi-Object Identity Tracking + ReID

- Tracking backbone: `BoT-SORT` or `ByteTrack` style tracker with conservative association in overlaps.
- ReID model: high-capacity person ReID model (your existing `SOLIDER`/part-aware ReID direction is aligned with this).
- Association cost should fuse:
  - motion,
  - appearance,
  - pose consistency (OKS-like or keypoint distance consistency).

Why: preventing ID switches is essential because pose quality appears to collapse when identity flips through occlusion.

## F) Temporal Pose Consistency

- Use short-window temporal optimization (5-15 frames):
  - smooth joint trajectories,
  - fill missing joints using forward/backward evidence,
  - reject kinematically implausible jumps.
- Recommended solver style:
  - per-joint confidence-weighted smoothing + visibility-state-aware interpolation,
  - global window objective preferred over frame-local greedy fixes.

Why: severe occlusion is often transient; temporal evidence restores joints not visible in a single frame.

## G) Deployment / Runtime Framework

- Training/research stack: `MMPose` + `MMDetection` ecosystem for reproducible model zoo baselines.
- Deployment: `MMDeploy` (ONNX/TensorRT/OpenVINO as needed) with explicit speed and accuracy validation.
- Always run post-conversion accuracy checks; do not trust runtime conversion blindly.

Why: this is the most mature open ecosystem for pose + detector + deploy parity testing.

## 2.2) "Gold Stack" Configuration (If You Want Maximum Accuracy)

If latency is secondary, this is the recommended end-to-end stack:

1. Detector: transformer-based high-recall person detector trained/fine-tuned with crowded-person data.
2. Global proposer: `HigherHRNet-W48`-class bottom-up model.
3. Disambiguation: `BUCTD`-style conditional top-down + `MIPNet`-style multi-instance head for overlap crops.
4. Refiner: `ViTPose-G` class (or strongest available ViTPose variant in practice).
5. Masks: `SAM2` soft guidance only (no hard binary cutout-only inputs).
6. Tracking: `BoT-SORT`/`ByteTrack` + part-aware ReID.
7. Temporal: short-window global pose-track optimization with visibility-aware joint recovery.
8. Output: keypoints + per-joint uncertainty + occlusion state + provenance.

This combination is closer to "best known system design" than any single-model substitution.

## 3) Training Strategy for Highest Accuracy

## Data

- Train/evaluate on datasets with true crowd/occlusion distribution, not only standard COCO:
  - CrowdPose,
  - OCHuman/OCHuman-Pose variants where available,
  - your in-domain dance footage with dense overlap annotations.
- Build hard-mined subsets:
  - severe overlap,
  - crossed arms/legs,
  - motion blur,
  - back-facing + partial truncation.

## Augmentation

- Prioritize realistic occlusion synthesis:
  - person-on-person cut-and-paste,
  - limb-level occluders,
  - box jitter that creates multi-person crops,
  - temporal dropout (simulate intermittent invisibility).
- Do not rely only on synthetic square erasing/masking; include real overlap compositions.

## Losses and Objectives

- Joint localization loss + uncertainty-aware weighting.
- Auxiliary losses:
  - inter-instance separation in overlap areas,
  - temporal consistency loss across adjacent frames,
  - kinematic plausibility regularization.

## Curriculum

- Start with easier scenes for convergence, then progressively upweight hard-occlusion frames.
- Maintain a fixed percentage of hard samples in each batch in late training.

## 4) Inference Policy (Accuracy First)

- Multi-pass adaptive inference:
  - pass 1: regular resolution/all persons,
  - pass 2: hard cases only (low confidence, overlap, fast motion) at higher resolution + extra refinement.
- Keep top-K hypotheses for ambiguous persons until temporal solver resolves them.
- Delay irreversible pruning to post-temporal stage.

## 5) Evaluation Protocol You Should Use

Measure all of the following, not only mAP:

- Keypoint AP by crowd level / occlusion level bins.
- AP on heavily occluded subsets (OCHuman-style hard bins).
- Temporal stability:
  - jitter metrics per joint,
  - missing-joint recovery rate after occlusion.
- Tracking-quality impact:
  - ID switches that coincide with pose corruption,
  - identity-conditioned pose continuity score.
- Error taxonomy:
  - wrong-person limb assignment,
  - left-right flips under occlusion,
  - hallucinated joints in invisible regions.

## 6) Recommended Stack for Your Current Project (Concrete)

Given your existing multi-phase pipeline design, the highest-accuracy practical evolution is:

1. Phase 1: tune detector for recall in crowds (retain overlapping candidates; avoid over-suppression).
2. Phase 2: keep SAM, but switch to soft guidance policy (no hard cutout-only input path).
3. Phase 3: keep dual-branch pose (`ViTPose` + `RTMW`), and add overlap-triggered multi-hypothesis inference.
4. Phase 3.5 (new): ambiguity resolver using BUCTD/MIPNet-style conditioning signal.
5. Phase 4/6: strengthen association with pose-consistency term in addition to motion + ReID.
6. Phase 8: add short-window global optimization for pose-track consistency and occlusion recovery.
7. Export: add per-joint uncertainty + occlusion state + correction provenance.

This yields larger gains than only swapping to a bigger backbone.

## 7) Safe Integration and Easy Rollback (Non-Disruptive by Design)

Yes, this can be structured so Phase 3.5 has near-zero blast radius and is easy to turn off.

## Core rule

- Phase 3.5 must be a pure add-on stage that reads existing Phase-3 outputs and returns an optional override.
- Existing Phase-3 path remains the default source of truth unless a flag explicitly enables Phase 3.5 output selection.

## Required feature flags

- `SWAY_PHASE35_ENABLED=0|1`
  - `0`: fully disabled (current behavior, exact fallback).
  - `1`: run Phase 3.5.
- `SWAY_PHASE35_MODE=off|shadow|active`
  - `off`: do not execute.
  - `shadow`: execute and log diagnostics, but never affect downstream outputs.
  - `active`: allow 3.5 to override Phase-3 pose hypotheses.
- `SWAY_PHASE35_FORCE_FAIL_OPEN=1` (default recommended)
  - on errors/timeouts, automatically fall back to baseline Phase-3 outputs.

## Data-contract pattern (important)

- Baseline contract stays unchanged:
  - keep current `pretrack_pose_by_frame_det` schema and current Phase-3 artifacts.
- Phase 3.5 writes sidecar artifacts only:
  - `phase3_5_disambiguation.json`
  - `phase3_5_metrics.json`
  - optional `phase3_5_overlay.mp4`
- In `shadow` mode:
  - no mutation of baseline files; only sidecar outputs.
- In `active` mode:
  - write both baseline and resolved outputs, plus a provenance tag per joint/track:
    - `source=phase3_baseline|phase3_5_override`

## Wiring pattern in code

- Add one gateway function:
  - `resolve_phase3_ambiguity(baseline_pose_map, context) -> resolved_pose_map_or_none`
- Downstream selector:
  - if mode is `active` and resolved output valid, use resolved map,
  - else use baseline map.
- Keep this selector localized at one boundary so turning Phase 3.5 off is one decision point.

## Rollout strategy (lowest risk)

1. `off` mode in production by default.
2. `shadow` mode on evaluation videos until metrics prove benefit.
3. enable `active` only behind explicit flag/profile.
4. keep a one-line rollback path: set `SWAY_PHASE35_MODE=off`.

## Validation gates before enabling `active`

- No regression in existing KPIs:
  - identity continuity,
  - switch rate,
  - keypoint AP on your internal hard set.
- Improvement must hold on occlusion-heavy slices, not only aggregate average.
- Runtime overhead within your accepted budget.

## Operational protections

- Timeout guard on Phase 3.5 execution; fail-open to baseline.
- Versioned artifact schema (`phase3_5_schema_version`) to avoid downstream breakage.
- Add run-manifest fields:
  - `phase35_enabled`,
  - `phase35_mode`,
  - `phase35_fallback_count`,
  - `phase35_override_ratio`.

With this structure, Phase 3.5 is fully reversible, testable in shadow mode, and can be disabled instantly without touching the current working pipeline behavior.

## 8) Common Failure Modes and Guardrails

- Failure: mask chops off limbs -> Guardrail: auto-disable strong masking when mask quality drops.
- Failure: box contains two people -> Guardrail: allow multi-instance hypotheses in that crop.
- Failure: per-frame confidence spikes on wrong identity -> Guardrail: temporal consistency and appearance-aware re-association.
- Failure: missing joints remain missing too long -> Guardrail: visibility-aware temporal fill with uncertainty decay.

## 9) "Perfect" Means "System", Not "Single Model"

There is no single model that is universally best for heavy occlusion and dense crowds.
The highest-accuracy solution is a system:

- global scene reasoning,
- instance disambiguation,
- high-capacity local refinement,
- temporal consistency optimization,
- uncertainty-aware outputs.

That is the architecture most aligned with current research and industry practice for this problem class.

## 10) Key References

- BUCTD (ICCV 2023): hybrid bottom-up conditioned top-down for crowded pose.
  - https://arxiv.org/abs/2306.07879
  - https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.pdf
- MIPNet (ICCV 2021): multi-instance prediction within a top-down crop.
  - https://arxiv.org/abs/2101.11223
  - https://openaccess.thecvf.com/content/ICCV2021/papers/Khirodkar_Multi-Instance_Pose_Networks_Rethinking_Top-Down_Pose_Estimation_ICCV_2021_paper.pdf
- CrowdPose benchmark/method paper (CVPR 2019): crowd-specific evaluation and global association motivation.
  - https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.pdf
- RTMPose (deployment-oriented baseline in MMPose ecosystem).
  - https://arxiv.org/abs/2303.07399
- MMPose deployment and validation workflow.
  - https://mmpose.readthedocs.io/en/latest/user_guides/how_to_deploy.html

