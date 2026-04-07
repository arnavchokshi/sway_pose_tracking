# PLAN 06: MeMoSORT Memory-Augmented Motion Prediction

> **Implementation Phase:** I1 (Foundation) · **Lean Core** · **Sweep Gate:** Gate 1 — bigtest HOTA > 0.55

## Objective

Add MeMoSORT-style components — a **memory-augmented Kalman filter** and **motion-adaptive IoU matching** — as a complementary motion prediction layer alongside SAM2 mask tracking.

- SAM2 answers: which pixels belong to this person.
- MeMoSORT answers: where this person is likely to be next frame.

Together they reduce track loss when SAM2 struggles (very small / distant masks) while SAM2 still handles heavy overlap better than box-only association. The hybrid targets rapid choreographic motion: leaps, turns, abrupt direction changes.

## Inputs & Dependencies

| Item | Detail |
|------|--------|
| **Upstream data** | Per-frame bounding boxes (from SAM2 mask envelopes or a detector) and stable track IDs. |
| **Prior steps** | PLAN_04 — SAM2 Primary Tracker; MeMoSORT runs alongside it on the same boxes / IDs. |

## Step-by-Step Implementation

1. Create `sway/memosort.py`.

2. **Memory-Augmented Kalman Filter** — class `MemoryKalmanFilter`:
   - **State vector:** `[x, y, w, h, vx, vy, vw, vh]` (center x/y, width, height, and velocities).
   - **Memory buffer:** `position_memory: deque(maxlen=SWAY_MEMOSORT_MEMORY_LENGTH)` (default `30`). Store observed positions over the last N frames.
   - After each **predict** step, compare predicted position to the mean of recent memory: `err = predicted_center - mean(memory_positions[-K:])` where `K = min(10, len(memory))`.
   - If `|err|` exceeds a threshold (e.g. 2× std of memory positions): treat Kalman as diverging. **Adapt Q:** scale process noise up proportionally to error magnitude so the filter trusts measurements more than its own prediction.
   - **Intent:** On sudden direction changes, plain Kalman lags; memory-augmented correction detects divergence and adapts.

3. **Motion-Adaptive IoU** — function `adaptive_iou(box_a, box_b, velocity_a, velocity_b, alpha: float) -> float`:
   - Compute standard IoU between `box_a` and `box_b`.
   - `speed = max(|velocity_a|, |velocity_b|)` (use a consistent norm, e.g. L2 on center velocities).
   - Expand both boxes by `alpha * speed` pixels per side (or isotropic expansion — document chosen convention).
   - Recompute IoU on expanded boxes.
   - Return `max(standard_iou, expanded_iou)`.
   - `alpha` from `SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA` (default `0.50`).

4. **Class `MeMoSORT`:**
   - `__init__(self, memory_length, adaptive_iou_alpha)` (wire to env defaults).
   - `predict(track_id) -> PredictedBox`: memory-augmented Kalman predict.
   - `update(track_id, observed_box)`: Kalman update + append to memory buffer.
   - `match(predictions, detections) -> List[Tuple[int, int]]`: Hungarian assignment with cost derived from **adaptive IoU** (e.g. `1 - adaptive_iou` or gated cost).

5. **SAM2 integration:** After SAM2 propagation each frame, run MeMoSORT predict/update. Use MeMoSORT's predicted box as **fallback** when SAM2 mask confidence is below `SWAY_SAM2_CONFIDENCE_REINVOKE`. When SAM2 is confident, take bbox from mask, update MeMoSORT for future frames only.

6. **Feature flag:** `SWAY_TRACKER_ENGINE=sam2_memosort_hybrid` enables combined mode (exact string and coexistence with other engines to be aligned with `main.py` / tracker factory).

## Technical Considerations & Performance

- MeMoSORT is lightweight: Kalman + deque + O(1) corrections per track per frame on CPU. Hungarian cost is O(n³) in number of tentative matches per frame — typically small.
- **Full occlusion (mask area = 0):** Continue Kalman prediction for **re-detection hints only** — e.g. bias SAM2 re-prompt or detector ROI. Do **not** treat predicted box as ground truth for active tracking; mark track **DORMANT** per existing pipeline rules.

## Validation & Testing

- **Sequences:** Rapid direction change, large inter-frame displacement (leap), spinning (rapid aspect change).
- **A/B:** (a) SAM2-only vs (b) SAM2 + MeMoSORT hybrid on continuity metrics.
- **Target:** ≥ 20% reduction in track breaks on dynamic sequences (e.g. gymtest-style footage), measured with project-defined continuity metric.

## Integration & Next Steps

| Consumer | Role |
|----------|------|
| PLAN_04 | Hybrid tracker uses MeMoSORT for prediction and low-confidence fallback. |

**Outputs:** `MeMoSORT`, `MemoryKalmanFilter`, `adaptive_iou`, and env-configured hybrid mode.

## Open Questions / Risks

- Default memory length (30 frames ≈ 1s @ 30 fps) may be wrong for very slow vs very fast choreography. **Sweep:** 10–60 frames and log continuity + false-merge rate.

---

*Standalone plan — no dependency on external research docs for execution.*
