# Implement Advanced Tracker Modules (MOTE, Sentinel, UMOT)

> **Implementation Phase:** I6 (Selective Advanced Modules) · **EXPERIMENT ADD-ONS — not lean core**
> Test ONE module at a time after the lean core (I1–I5) passes all gates.
> Per Perplexity fact-check: "MOTE and UMOT both address long-duration occlusions, albeit with different mechanisms. Sentinel's SBM and UMOT's historical backtracking both attempt to preserve tracks through weak detections. This suggests picking 1–2 modules that best match the failure modes of the dance data, rather than all of them."
> **Testing order:** MOTE first (most likely to help crossovers) → Sentinel (entrance/exit cycling) → UMOT (long dormancy). Only combine after individual impact is measured.

**Objective:** Add three optional tracker modules that target specific remaining failure modes after the core pipeline (Layers 0–5) is implemented. Each module is a standalone add-on that can be independently enabled/disabled: (1) **MOTE** — optical-flow-based disocclusion prediction for crossover events, (2) **Sentinel** — survival boosting for tracks at risk of disappearance during long occlusions, (3) **UMOT** — historical backtracking for long-term dormancy recovery. These default to OFF and are opt-in for specific failure modes.

## Inputs & Dependencies

* **Upstream Data:** Per-frame: video frames, detection bounding boxes, track IDs, track states (from PLAN_01), SAM2 masks (from PLAN_04).
* **Prior Steps:** PLAN_01 (state machine), PLAN_04 (SAM2 tracker). These modules augment the existing tracker — they are not replacements.

## Step-by-Step Implementation

### Module 1: MOTE Disocclusion Prediction

1. Create `sway/mote_disocclusion.py`.
2. Implement optical flow computation using RAFT (Recurrent All-Pairs Field Transforms):
   - **a.** Install `torchvision` (already a dependency, includes RAFT).
   - **b.** Use `torchvision.models.optical_flow.raft_small` or `raft_large` (from `SWAY_MOTE_FLOW_MODEL`, default `raft_small`).
   - **c.** For each frame pair (t-1, t): compute dense optical flow field.
3. Implement `predict_reemergence(flow_field, dormant_tracks) -> Dict[int, Tuple[float, float]]`:
   - **a.** For each DORMANT track: use its last known position and the optical flow at that position to project where it will reappear.
   - **b.** Apply softmax splatting to handle occlusion boundaries: the flow at occlusion edges indicates the direction of disocclusion.
   - **c.** Return `{track_id: (predicted_x, predicted_y)}`.
4. Integrate with re-ID (PLAN_13): when a dormant track re-emerges, compare the observed re-emergence position against MOTE's predicted position. If they match within 50px: boost re-ID confidence by `SWAY_MOTE_CONFIDENCE_BOOST` (default 0.15). If they don't match: the track may have moved unexpectedly — increase weight on appearance signals.
5. Add env var `SWAY_MOTE_DISOCCLUSION` (bool, default `0`).

### Module 2: Sentinel Survival Boosting

1. Create `sway/sentinel_sbm.py`.
2. Implement class `SurvivalBoostingMechanism`:
   - **a.** For each track, maintain a **survival score**: running average of detection confidence over the last 100 frames.
   - **b.** When a track's current-frame confidence drops below `SWAY_YOLO_CONF` but the survival score is high (track was consistently strong): grant a **grace period** of `current_max_age * SWAY_SENTINEL_GRACE_MULTIPLIER` (default 3.0) frames.
   - **c.** During the grace period: accept detections with confidence as low as `SWAY_SENTINEL_WEAK_DET_CONF` (default 0.08) if they spatially match the track's predicted position (within 2× bbox width).
   - **d.** If the track is re-found during grace: restore to ACTIVE with a boosted survival score.
   - **e.** If grace expires without re-finding: transition to DORMANT normally.
3. Integrate into the tracking loop: between detection and association, check SBM grace status for each PARTIAL/DORMANT-candidate track. Add weak detections (below normal threshold but above `SWAY_SENTINEL_WEAK_DET_CONF`) to the association candidate pool for grace-period tracks.
4. Add env var `SWAY_SENTINEL_SBM` (bool, default `0`).

### Module 3: UMOT Historical Backtracking

1. Create `sway/umot_backtrack.py`.
2. Implement `HistoricalTrajectoryBank`:
   - **a.** Store the full trajectory (position + embeddings per frame) for every track ever seen, including LOST tracks. Storage: `{track_id: List[(frame, x, y, embedding)]}`.
   - **b.** Maximum history: `SWAY_UMOT_HISTORY_LENGTH` (default 500) frames per track.
3. When a new detection appears that does not match any active track:
   - **a.** Query the trajectory bank: compute re-ID similarity between the new detection and all historical tracks.
   - **b.** If a match is found (similarity > threshold): reactivate the historical track. This handles dancers who were LOST for > max_age but have returned.
   - **c.** This replaces the simple "is the new detection near a DORMANT track?" heuristic with a full embedding-based search over all historical trajectories.
4. Add env var `SWAY_UMOT_BACKTRACK` (bool, default `0`).

## Technical Considerations & Performance

* **Architecture Notes:** MOTE is the most expensive: RAFT optical flow is ~30ms per frame on GPU. For a 3600-frame video: ~108 seconds. Only enable on videos where crossover prediction is critical (bigtest). Sentinel adds ~0.1ms per frame (just bookkeeping). UMOT adds ~1ms per new detection (embedding comparison against history).
* **Edge Cases & I/O:** MOTE's optical flow is noisy in fast-moving dance footage. Use flow confidence masking: ignore flow vectors with magnitude > 50px (likely erroneous). Sentinel must not keep "zombie tracks" alive forever — the grace multiplier caps at 5×. UMOT's trajectory bank should be pruned: remove LOST tracks older than 5× max_age.

## Validation & Testing

* **Verification:** Test each module independently on bigtest:
  - **MOTE:** log predicted vs actual re-emergence positions. Measure prediction accuracy.
  - **Sentinel:** count tracks that would have been lost without SBM but were recovered. Compare HOTA.
  - **UMOT:** count re-activations from the trajectory bank. Verify correct identity.
* **Metrics:** Each module should independently improve bigtest HOTA by ≥ 1 point when enabled. Combined: ≥ 3 points. No regressions on easy sequences (aditest, easytest).

## Integration & Next Steps

* **Outputs:** Three standalone modules with enable/disable flags. Consumed by the tracking loop in `main.py`.
* **Open Questions/Risks:** Enabling all three simultaneously may cause conflicts (e.g., MOTE predicts re-emergence but Sentinel has already kept the track alive). Define priority: **Sentinel > MOTE > UMOT** (survival boosting prevents dormancy, making MOTE less relevant). Test interactions carefully.

## Swapability & Experimentation

**Toggle pattern:** These modules are NOT factory-dispatched (they're not alternatives to each other — they augment the tracker). They use simple boolean flags:

- `SWAY_MOTE_DISOCCLUSION=0|1`
- `SWAY_SENTINEL_SBM=0|1`
- `SWAY_UMOT_BACKTRACK=0|1`

**Mutual exclusivity in sweeps:** In sweep phase S5, `suggest_advanced_module_params()` treats these as mutually exclusive by default — only one ON per trial. After individual impact is measured, a follow-up sweep can test pairwise combinations.

**Manual A/B recipes:**

```bash
# Test each module independently on bigtest
python -m tools.pipeline_matrix_runs \
  --recipes lean_core,lean_core_mote,lean_core_sentinel,lean_core_umot \
  --video data/ground_truth/bigtest/BigTest.mov --compare

# After individual winners, test the best pair
python -m tools.pipeline_matrix_runs \
  --recipes lean_core_mote,lean_core_mote_sentinel \
  --video data/ground_truth/bigtest/BigTest.mov --compare
```

**Gate 6 criterion:** Each module must independently show a statistically significant reduction in bigtest identity switches (≥ 1 HOTA point improvement, ≥ 20% fewer ID switches). If it doesn't clear the bar, keep it OFF.
