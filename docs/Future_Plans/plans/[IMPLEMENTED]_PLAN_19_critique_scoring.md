# Build Five-Dimension Critique Scoring Engine

> **Implementation Phase:** I7 (Critique Layer) · **Lean Core** · **Sweep Gate:** Gate 7 — Critique output matches human evaluation on ≥3 reference clips

**Objective:** Upgrade the critique system from the current 2-dimension analysis (angle deviation + timing errors vs group consensus) to a 5-dimension biomechanical analysis: formation accuracy, timing precision, extension/line quality, smoothness, and group synchronization. Each dimension produces a time-series score per dancer with timestamped, actionable feedback. Critically, the system respects per-keypoint confidence: only HIGH and MEDIUM joints produce feedback, and visibility gaps are explicitly reported.

## Inputs & Dependencies

* **Upstream Data:** Per-dancer: (a) 2D keypoint sequences with confidence levels (from PLAN_17), (b) 3D keypoint sequences in shared world coordinates (from PLAN_18), (c) track state time series (from PLAN_01), (d) audio track from the video (for timing precision).
* **Prior Steps:** PLAN_17 (per-keypoint confidence), PLAN_18 (multi-person 3D lifting), PLAN_01 (state machine — determines which frames have valid data).

## Step-by-Step Implementation

1. Create `sway/critique_engine.py`.
2. **Dimension 1: Formation Accuracy**
   - **a.** Input: 3D positions (hip midpoint) of all dancers per frame, plus a formation template.
   - **b.** Formation template: either user-provided (future Lab UI feature) or auto-extracted as the mean positions during the first 30 seconds (when all dancers are likely in their starting formation).
   - **c.** Per dancer, per frame: compute Euclidean distance between actual 3D position and template position.
   - **d.** Output: `formation_error_cm[dancer_id][frame]` — deviation in centimeters.
   - **e.** Requires `SWAY_LIFT_MULTI_PERSON=1` (shared coordinate frame).
3. **Dimension 2: Timing Precision**
   - **a.** Extract audio beat grid from the video's audio track using `librosa`:
     - `librosa.load(audio_path, sr=22050)`
     - `tempo, beat_frames = librosa.beat.beat_track(y, sr=sr)`
     - Convert `beat_frames` to video frame indices.
   - **b.** For each dancer: compute movement peaks (frames where total joint velocity is locally maximal). Use `scipy.signal.find_peaks` on the joint velocity magnitude time series.
   - **c.** For each beat: find the nearest movement peak. Compute `timing_offset_ms = (peak_frame - beat_frame) / fps * 1000`.
   - **d.** Output: `timing_offsets_ms[dancer_id][beat_idx]` — positive = late, negative = early.
4. **Dimension 3: Extension and Line**
   - **a.** For key joint pairs (shoulder→elbow→wrist, hip→knee→ankle), compute the joint angle at every frame.
   - **b.** At movement peaks (identified in dimension 2): compare the angle against a reference. Reference options:
     - Biomechanical maximum (e.g., 180° for full arm extension).
     - Group mean (the average angle of all dancers at that moment).
     - Teacher reference (future: from a reference video).
   - **c.** Output: `extension_deficit_deg[dancer_id][frame][joint_pair]` — how many degrees short of reference.
5. **Dimension 4: Smoothness**
   - **a.** For each joint: compute the **jerk** (third derivative of position with respect to time) using `np.gradient` applied three times.
   - **b.** Smooth the jerk signal with a window of `SWAY_CRITIQUE_JERK_WINDOW` (default 5, sweep 3–15) frames.
   - **c.** High jerk at non-impact moments indicates uncontrolled movement. At impact moments (beat hits), high jerk is expected.
   - **d.** Output: `jerk_score[dancer_id][frame][joint]` — lower is smoother.
6. **Dimension 5: Group Synchronization**
   - **a.** For each frame: compute the group mean 3D pose (average joint positions across all ACTIVE dancers).
   - **b.** Per dancer: compute `sync_deviation = ||dancer_pose - group_mean_pose||` (L2 norm across all joints).
   - **c.** Also compute timing synchronization: cross-correlation between dancer's velocity profile and group mean velocity profile. Peak offset = timing desynchronization.
   - **d.** Output: `sync_score[dancer_id][frame]` — lower is more synchronized.
7. **Confidence gating (LOCKED):**
   - **a.** For each dimension, only include joints with confidence ≥ MEDIUM.
   - **b.** When a dancer is in PARTIAL state: compute scores only for visible joints. Report: "Upper body scores available; lower body occluded — no feedback."
   - **c.** When a dancer is in DORMANT state: no scores. Report: "Dancer not visible from frame X to Y — no feedback for this interval."
8. **Report generation:**
   - **a.** For each dancer, produce a timestamped report. Example:

     ```
     Dancer 3, measures 12–16 (0:42–0:55):
     - Right arm extension: 15° short of group at peak of each cycle
     - Timing: consistently 2 frames (67ms) early on the downbeat
     - Shoulder alignment: within 3° of target (good)
     - Lower body not visible during frames 670–695 — no feedback
     ```

   - **b.** Output format: JSON with per-dancer, per-dimension, per-timestamp entries.
9. Add env vars:
   - `SWAY_CRITIQUE_DIMENSIONS` — default `formation,timing,extension,smoothness,sync`
   - `SWAY_CRITIQUE_JERK_WINDOW` — default `5`
   - `SWAY_CRITIQUE_BEAT_TOLERANCE_MS` — default `100`
   - `SWAY_CRITIQUE_MIN_CONFIDENCE` — LOCKED: `MEDIUM`
   - `SWAY_CRITIQUE_REPORT_GAPS` — LOCKED: `1`

## Technical Considerations & Performance

* **Architecture Notes:** All five dimensions are pure numpy/scipy computations — no GPU needed. The most expensive is dimension 2 (beat detection via librosa): ~2 seconds for a 2-minute audio track. Everything else is sub-second.
* **Edge Cases & I/O:** If the video has no audio track: skip dimension 2 (timing precision). If `SWAY_LIFT_MULTI_PERSON=0` (no shared coordinate frame): skip dimension 1 (formation accuracy) and use 2D-only synchronization for dimension 5.

## Validation & Testing

* **Verification:** Run on easytest (near-saturated, easy sequence): verify all 5 dimensions produce reasonable scores. Manually check 10 timestamped feedback entries against visual inspection of the video. Verify that DORMANT intervals produce zero feedback with explicit gap reports.
* **Metrics:** Timing precision: beat detection accuracy ≥ 90% (verified against manual beat annotation). Extension scoring: joint angle measurement within ±3° of manual measurement. Gap reporting: 100% of DORMANT intervals are explicitly reported.

## Integration & Next Steps

* **Outputs:** Per-dancer critique JSON with 5-dimension scores + confidence annotations + gap reports. This is the final user-facing deliverable. Consumed by the Pipeline Lab UI for display.
* **Open Questions/Risks:** The formation template (dimension 1) requires either user input or a good auto-extraction heuristic. The auto-extraction (mean positions in first 30 seconds) assumes dancers start in formation — which may not always be true.
