# Implement Visibility-Masked Pose Estimation with Per-Keypoint Confidence

> **Implementation Phase:** I4 (Pose + 3D Upgrade) · **Lean Core** · **Sweep Gate:** Gate 4 — Per-keypoint confidence accuracy > 90%
> **Swappable component:** `SWAY_POSE_MODEL` dispatches via pose factory — choices: `vitpose_large` (lean core default), `vitpose_huge`, `rtmw_l`, `rtmw_x` (experiment add-ons)

**Objective:** Upgrade pose estimation to (a) use SAM2 segmentation masks as auxiliary input to suppress keypoints from occluding people (prevents chimeric skeletons), (b) optionally upgrade to RTMW for whole-body keypoints (hands + feet + face), and (c) add a per-keypoint confidence classifier (HIGH/MEDIUM/LOW/NOT_VISIBLE) so the critique layer never gives feedback on joints it cannot actually see.

## Inputs & Dependencies

* **Upstream Data:** Person crops (BGR numpy), SAM2 masks (binary numpy per person), track state (from PLAN_01 — only run on ACTIVE and PARTIAL tracks).
* **Prior Steps:** PLAN_04 (SAM2 masks), PLAN_01 (state machine gates which tracks get pose estimation).

## Step-by-Step Implementation

1. Modify `sway/pose_estimator.py` (or create `sway/mask_guided_pose.py`).
2. **Mask-guided inference:**
   a. Before running ViTPose/RTMW on a person crop, apply the SAM2 mask:
      - Option A (simple): zero out pixels outside the mask. Set background to mean ImageNet RGB. This removes occluder pixels from the input.
      - Option B (advanced): concatenate the binary mask as a 4th input channel (requires model fine-tuning or adapter). Start with Option A.
   b. Run pose inference on the masked crop.
   c. The model should now detect keypoints only from the target person's pixels, not from the occluder.
3. **RTMW model option:**
   a. Add support for RTMW-L and RTMW-X (from MMPose). These output 133 keypoints: 17 body + 6 foot + 68 face + 42 hand (21 per hand).
   b. Install `mmpose` if not already available. Use the RTMW config and checkpoint from the official release.
   c. RTMW output format: array of (133, 3) — x, y, confidence per keypoint.
   d. When `SWAY_POSE_KEYPOINT_SET=wholebody133`: use RTMW output. When `coco17`: use existing ViTPose output (or extract the 17 body joints from RTMW).
4. **Per-keypoint confidence classifier:**
   a. For each keypoint, compute three signals:
      - `heatmap_peak`: the maximum value in the pose model's output heatmap for this joint. Higher = more confident.
      - `mask_inside`: whether the keypoint (x, y) falls inside the SAM2 mask. Boolean.
      - `temporal_consistency`: compare keypoint position to the mean position over the last `SWAY_CONFIDENCE_TEMPORAL_WINDOW` (default 5) frames. If distance > 2× the rolling std: temporally inconsistent.
   b. Classify:
      - `HIGH`: heatmap_peak ≥ `SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH` (default 0.70) AND mask_inside AND temporally consistent.
      - `MEDIUM`: heatmap_peak ≥ `SWAY_CONFIDENCE_HEATMAP_THRESH_MED` (default 0.40) AND (mask_inside OR temporally consistent).
      - `LOW`: heatmap_peak ≥ 0.10 BUT (NOT mask_inside OR NOT temporally consistent). This catches occluder keypoints — high heatmap but wrong person.
      - `NOT_VISIBLE`: heatmap_peak < 0.10 OR joint is outside the crop.
   c. **Mask gate (LOCKED):** `SWAY_CONFIDENCE_MASK_GATE=1`. If a keypoint has high heatmap score but falls OUTSIDE the SAM2 mask, downgrade to LOW regardless. This prevents chimeric skeletons.
5. Output format: extend the existing keypoint array from (17, 3) to (17, 4) — add a confidence_level column (0=NOT_VISIBLE, 1=LOW, 2=MEDIUM, 3=HIGH). Or use a separate `confidence_levels: np.ndarray` of shape (17,) with enum values.
6. Add to `data.json` output: per-frame, per-track confidence levels for each keypoint.
7. Gate critique: downstream scoring (PLAN_19) only uses keypoints with confidence ≥ MEDIUM. LOW and NOT_VISIBLE joints are excluded from critique and reported as gaps.
8. Add env vars: `SWAY_POSE_MODEL` (default `vitpose_large`, options: `vitpose_huge`, `rtmw_l`, `rtmw_x`), `SWAY_POSE_MASK_GUIDED` (default `1`), `SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH` (0.70), `SWAY_CONFIDENCE_HEATMAP_THRESH_MED` (0.40).

## Technical Considerations & Performance

* **Architecture Notes:** Mask-guided inference (Option A — background zeroing) adds ~0.5ms per crop. No model retraining needed. RTMW-L inference is ~8ms per crop on GPU (similar to ViTPose-L). The confidence classifier is pure arithmetic: ~0.1ms per crop.
* **Edge Cases & I/O:** When SAM2 mask is not available (fallback tracker without masks), skip mask guidance and use standard pose inference. Confidence classifier skips the `mask_inside` check and relies on heatmap + temporal consistency only. When the pose model detects a joint at the edge of the mask: classify as MEDIUM (partially visible).

## Validation & Testing

* **Verification:** On bigtest crossover frames: compare keypoints with vs without mask guidance. With mask guidance: (a) no chimeric skeletons (left arm from person A, right arm from person B), (b) confidence classifier correctly marks occluded joints as LOW/NOT_VISIBLE. Without mask guidance: chimeric skeletons should be observable.
* **Metrics:** Chimeric skeleton rate: ≤ 2% with mask guidance (vs ≥ 15% without). Confidence accuracy: when we say HIGH, localization error ≤ 5px in ≥ 95% of cases. Visibility false-positive rate (claiming visible when actually occluded): ≤ 3%.

## Integration & Next Steps

* **Outputs:** Extended keypoint format with per-joint confidence levels. Consumed by: PLAN_14 (EMA uses pose quality for gating), PLAN_19 (critique only uses HIGH/MEDIUM joints), PLAN_18 (3D lifting receives confidence-annotated 2D keypoints).
* **Open Questions/Risks:** ViTPose may not respond well to zeroed-out regions (it was trained on full crops). If mask guidance degrades pose quality for visible joints, consider the 4th-channel approach (Option B) which requires a small amount of fine-tuning. Note: "RTMW" as a specific named model is unverified per Perplexity fact-check — it likely refers to RTMPose-WholeBody from MMPose. Use the official MMPose config name when implementing.

## Swapability & Experimentation

**Factory integration:** Pose models implement a common interface:

```python
class BasePoseEstimator(ABC):
    def estimate(self, crop, mask=None) -> PoseResult: ...
    # PoseResult: keypoints (N, 3), confidence_levels (N,)
```

Dispatch via `SWAY_POSE_MODEL`:
- `vitpose_large` → current production model (lean core default)
- `vitpose_huge` → higher accuracy, higher cost
- `rtmw_l` / `rtmw_x` → whole-body 133 keypoints (experiment add-on, test after body-only critique works)

Mask guidance (`SWAY_POSE_MASK_GUIDED`) is orthogonal to model choice — it applies to any model by preprocessing the crop.

**Manual A/B recipe:**

```bash
# Compare pose models
python -m tools.pipeline_matrix_runs \
  --recipes pose_vitpose_large,pose_vitpose_huge,pose_rtmw_l \
  --video data/ground_truth/bigtest/BigTest.mov --compare
```

**Sweep readiness:** `SWAY_POSE_MODEL` is categorical in sweep phase S6. `SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH/MED` are continuous params sweepable after the model is chosen.
