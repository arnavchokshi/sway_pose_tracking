# PLAN 05: Cross-Object Interaction Module

> **Implementation Phase:** I1 (Foundation) · **Lean Core** · **Sweep Gate:** Gate 1 — bigtest HOTA > 0.55

## Objective

Add a collision-detection and memory-quarantine system (inspired by SAM2MOT) that prevents identity corruption when dancers cross over each other. When two tracked masks overlap significantly, the module detects which track is being occluded by analyzing logit score variance, then quarantines (deletes or freezes) the contaminated memory entries from SAM2's memory bank. This directly addresses a core failure mode: ID swaps caused by the tracker absorbing the wrong person's pixels during crossovers (e.g. bigtest crossover segments).

## Inputs & Dependencies

| Item | Detail |
|------|--------|
| **Upstream data** | Per-frame, per-track: SAM2 segmentation masks (binary), SAM2 logit scores (float, from the video predictor), and track IDs. |
| **SAM2 API** | The SAM2 primary tracker (PLAN_04) must expose raw logit scores per mask and provide an API to delete or freeze specific memory entries. |
| **Prior steps** | PLAN_04 — SAM2 Primary Tracker must be implemented; COI operates on SAM2's internal memory bank. |

## Step-by-Step Implementation

1. Create `sway/cross_object_interaction.py`.

2. Implement class `CrossObjectInteraction`:
   - `__init__(self, mask_iou_thresh: float, logit_variance_window: int, quarantine_mode: str)`: Read params from env — `SWAY_COI_MASK_IOU_THRESH` (default `0.25`), `SWAY_COI_LOGIT_VARIANCE_WINDOW` (default `10`), `SWAY_COI_QUARANTINE_MODE` (default `"delete"`).
   - Internal state: `logit_history: Dict[int, deque]` — per-track rolling buffer of logit scores over the last N frames.

3. Implement `check_collisions(masks: Dict[int, np.ndarray], logits: Dict[int, float], frame_idx: int) -> List[QuarantineAction]`:
   - **a.** Compute pairwise **Mask IoU** for all active tracks. Mask IoU = `intersection_area / union_area`. Use `np.logical_and` / `np.logical_or` on binary masks.
   - **b.** For each pair where Mask IoU > `mask_iou_thresh`: treat as a collision.
   - **c.** For each colliding pair `(track_A, track_B)`:
     - Look up `logit_history[track_A]` and `logit_history[track_B]` over the last `logit_variance_window` frames.
     - Compute logit score **variance** for each track.
     - The track with **higher variance** (abrupt score drop) is treated as occluded — its memory is being corrupted by the other person's pixels.
     - Emit `QuarantineAction(track_id=occluded_track, mode=quarantine_mode, start_frame=collision_start)`.

4. Implement `apply_quarantine(sam2_tracker, action: QuarantineAction)`:
   - If `mode == "delete"`: call SAM2 tracker API to remove memory entries for `track_id` from `collision_start` through current frame (aggressive; SAM2MOT-style).
   - If `mode == "freeze"`: mark the track's memory entries as frozen — they remain but are not updated (simpler, less aggressive).

5. **Collision-end detection:** When Mask IoU for a previously colliding pair drops below `mask_iou_thresh * 0.5` (hysteresis to reduce flicker), treat collision as ended. Resume normal memory updates for the quarantined track.

6. **Per-frame logit update:** `update_logits(track_id, logit_score)` appends to the deque each frame.

7. **Integration:** In `SAM2PrimaryTracker.process_frame()`: after mask propagation, call `coi.check_collisions()`, then `apply_quarantine()` for returned actions, then continue the rest of the pipeline.

8. **Logging:** Per collision, log: colliding tracks, quarantined track, frame range, and final outcome (ID maintained vs swap detected).

## Data Structures (Suggested)

- `QuarantineAction`: `track_id`, `mode` (`"delete"` | `"freeze"`), `start_frame` (and optionally `end_frame` when collision resolves).

## Technical Considerations & Performance

- **Bottleneck:** Mask IoU is O(N²) pairwise comparisons on binary masks. For N ≤ 10 at 256×256, expect sub-millisecond per frame; no special optimization required unless N or resolution grows substantially.
- **Three-way collisions:** Handle all pairwise collisions independently. Across partners, the track with highest logit variance is the primary victim. If two tracks both show high variance, quarantine both (conservative; reduces bidirectional swap risk).
- **Hysteresis:** Exit threshold at 50% of entry threshold reduces rapid quarantine / release oscillation.

## Validation & Testing

- **Verification:** Run on bigtest (or similar) crossover segments where 3+ dancers cluster. Manually check: (a) collisions detected at correct frames, (b) occluded track (not occluder) is quarantined, (c) after collision ends, quarantined track resumes with correct identity. Compare ID switches with vs without COI.
- **Targets:** ≥ 90% of crossover events correctly identify the occluded track; ID switches during crossovers drop ≥ 50% vs no-COI baseline.

## Integration & Next Steps

| Consumer | Role |
|----------|------|
| PLAN_04 | SAM2 tracker invokes COI every frame. |
| PLAN_16 | Backward pass should run COI on reversed video as well (same logic, reversed temporal order). |

**Outputs:** `CrossObjectInteraction` with `check_collisions()`, `apply_quarantine()`, `update_logits()`, and collision lifecycle (start/end with hysteresis).

## Open Questions / Risks

- SAM2's public API may not expose memory-bank deletion. **Mitigation:** Implement `"freeze"` first; add a SAM2 fork or patch if `"delete"` is required and unsupported.

---

*Standalone plan — no dependency on external research docs for execution.*
