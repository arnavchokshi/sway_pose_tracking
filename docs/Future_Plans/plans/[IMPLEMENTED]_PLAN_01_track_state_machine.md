# Implement Occlusion-Aware Track State Machine

> **Implementation Phase:** I1 (Foundation) · **Lean Core — foundational** · **Sweep Gate:** Gate 1 — bigtest HOTA > 0.55

**Objective:** Replace the current binary tracked/lost track lifecycle with a four-state machine (ACTIVE → PARTIAL → DORMANT → LOST) so the Sway pipeline can distinguish between full visibility, partial occlusion, full occlusion, and abandoned tracks. This is foundational — every downstream component (pose estimation, re-ID, critique generation) reads track state to decide whether to process, update galleries, or report "no data."

## Inputs & Dependencies

* **Upstream Data:** Per-frame SAM2 segmentation masks for each tracked person (binary mask per track ID), plus an enrollment-time reference mask area per dancer (from the enrollment gallery, PLAN_07). Until enrollment exists, use the max mask area seen in the first 30 frames as the reference.
* **Prior Steps:** None — this is a foundation component. It defines a data contract consumed by all other plans.

## Step-by-Step Implementation

1. Create `sway/track_state.py`. Define an enum `TrackState` with values `ACTIVE`, `PARTIAL`, `DORMANT`, `LOST`.
2. Define a dataclass `TrackLifecycle` with fields:
   - `track_id: int`
   - `state: TrackState` (default ACTIVE)
   - `reference_mask_area: float` (set from enrollment or auto-computed)
   - `current_mask_area: float`
   - `frames_in_dormant: int` (counter, reset on re-emergence)
   - `visible_joint_ids: List[int]` (populated by pose estimator, empty when DORMANT)
   - `last_active_frame: int`
   - `state_history: List[Tuple[int, TrackState]]` (frame, state pairs for debugging)
3. Implement transition logic as a pure function `update_state(lifecycle, mask_area, num_visible_joints, frame_idx) -> TrackState`:
   - Compute `mask_frac = mask_area / reference_mask_area`.
   - If `mask_frac >= SWAY_STATE_PARTIAL_MASK_FRAC` (default 0.30) AND `num_visible_joints >= SWAY_STATE_PARTIAL_MIN_JOINTS` (default 5): state = ACTIVE.
   - Else if `mask_frac >= SWAY_STATE_DORMANT_MASK_FRAC` (default 0.05) AND `num_visible_joints >= 1`: state = PARTIAL.
   - Else if `mask_frac < SWAY_STATE_DORMANT_MASK_FRAC` OR mask is None: state = DORMANT. Increment `frames_in_dormant`.
   - If `frames_in_dormant > SWAY_STATE_DORMANT_MAX_FRAMES` (default 300): state = LOST.
   - On any transition, append to `state_history`.
4. Read config values from `os.environ` using the `SWAY_STATE_*` prefix. Provide defaults matching the table above.
5. Add a method `should_run_pose(state) -> bool`: returns True for ACTIVE and PARTIAL only.
6. Add a method `should_update_gallery(state) -> bool`: returns True for ACTIVE only (PARTIAL updates are handled by the pose-gated EMA module with restricted signals).
7. Add a method `should_generate_critique(state) -> bool`: returns True for ACTIVE and PARTIAL only.
8. Integrate into `main.py` frame loop: after SAM2 produces masks for frame N, call `update_state` for each track. Store the state on the track object. Downstream phases (pose, re-ID, scoring) check the state before processing.
9. Add `state` field to the per-track data in `data.json` output so visualization and scoring can use it.

### Environment variables (defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SWAY_STATE_PARTIAL_MASK_FRAC` | `0.30` | Minimum mask area fraction for ACTIVE |
| `SWAY_STATE_PARTIAL_MIN_JOINTS` | `5` | Minimum visible joints for ACTIVE |
| `SWAY_STATE_DORMANT_MASK_FRAC` | `0.05` | Threshold below which → DORMANT (when joints insufficient) |
| `SWAY_STATE_DORMANT_MAX_FRAMES` | `300` | Frames in DORMANT before LOST |

## Technical Considerations & Performance

* **Architecture Notes:** The state machine is a pure function with no I/O — it receives mask area and joint count, returns a state. Keep it stateless except for the `frames_in_dormant` counter. This makes it trivially testable.
* **Edge Cases & I/O:** When SAM2 is not the primary tracker (fallback to SolidTrack), use bounding box area ratio instead of mask area ratio. The interface is the same: `mask_area` is replaced by `bbox_area`. When a track first appears mid-video (not from enrollment), set `reference_mask_area` to the area at first detection.

## Validation & Testing

* **Verification:** Unit test `update_state` with synthetic mask area sequences: (a) a clean sequence that stays ACTIVE; (b) a sequence that transitions ACTIVE → PARTIAL → DORMANT → LOST with correct thresholds; (c) a sequence that recovers from DORMANT → ACTIVE. Integration test: run on bigtest and log state transitions per track, verify no DORMANT track generates pose data.
* **Metrics:** Zero pose/critique data generated for any frame where state is DORMANT or LOST. State transitions should match SAM2 mask area ground truth within ±2 frames.

## Integration & Next Steps

* **Outputs:** `TrackState` enum and `TrackLifecycle` dataclass, consumed by: pose estimation (PLAN_17), re-ID fusion (PLAN_13), pose-gated EMA (PLAN_14), critique scoring (PLAN_19), backward pass (PLAN_16).
* **Open Questions/Risks:** The `reference_mask_area` for the first few frames (before enrollment) may be noisy if the dancer is partially visible on entry. Mitigation: use a running max over the first 30 frames, not the first frame alone.
