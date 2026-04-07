# Promote SAM2 to Primary Mask-Based Tracker

> **Implementation Phase:** I1 (Foundation) · **Lean Core** · **Sweep Gate:** Gate 1 — bigtest HOTA > 0.55
> **Swappable component:** `SWAY_TRACKER_ENGINE` dispatches via `sway/tracker_factory.py` (see FUTURE_PIPELINE.md §12.1)

**Objective:** Transform SAM2 from a helper module (currently called only for hybrid overlap refinement) into the primary tracking engine. Instead of tracking bounding boxes through a Kalman filter (SolidTrack/BotSORT), track per-person pixel-level segmentation masks via SAM2's temporal memory propagation. This eliminates bounding box overlap ambiguity — the core failure mode on bigtest.

## Inputs & Dependencies

* **Upstream Data:** Per-frame detections from the detection layer (PLAN_03 hybrid detector or standalone YOLO/DETR). Each detection is a bounding box used to prompt SAM2 for mask initialization.
* **Prior Steps:** PLAN_01 (Track State Machine — SAM2 mask area drives state transitions). PLAN_02/03 (detection backend for initialization).
* **Existing Code:** `sway/hybrid_sam_refiner.py` already loads SAM2 (`sam2.1_b.pt`) and runs single-frame mask prediction. This code will be refactored.

## Step-by-Step Implementation

1. Create `sway/sam2_tracker.py`.
2. Implement class `SAM2PrimaryTracker`:
   - `__init__(self, model_path: str, device: str, reinvoke_stride: int, confidence_reinvoke: float)`: Load SAM2 model with video predictor (not image predictor). Params from `SWAY_SAM2_REINVOKE_STRIDE` (default 30), `SWAY_SAM2_CONFIDENCE_REINVOKE` (default 0.40).
   - Internal state: `active_tracks: Dict[int, SAM2Track]` mapping track_id to track metadata (mask memory, logit history, TrackLifecycle).
3. Implement initialization flow:
   a. On frame 0 (or enrollment frame): run detector, get all person bounding boxes.
   b. For each detection, prompt SAM2 with the bounding box. SAM2 produces a segmentation mask.
   c. Create a `SAM2Track` for each mask with a unique track_id.
   d. Add all prompts to SAM2's video predictor memory bank.
4. Implement per-frame propagation:
   a. On each subsequent frame, call SAM2's `propagate_in_video()` to propagate all active masks forward.
   b. For each propagated mask: extract bounding box (min/max of mask coordinates), compute mask area, compute logit confidence score.
   c. Pass mask area to `TrackLifecycle.update_state()` (from PLAN_01) to update track states.
   d. Store the per-mask logit scores in a rolling buffer (used by Cross-Object Interaction, PLAN_05).
5. Implement detector re-invocation:
   a. Every `reinvoke_stride` frames: run detector. Compare detections against active track masks via IoU.
   b. Unmatched detections → potential new person entering. Prompt SAM2 with the new bounding box, create new track.
   c. If any active track's logit confidence drops below `confidence_reinvoke`: force detector re-invocation on the next frame regardless of stride.
6. Implement the output contract: `process_frame(frame, frame_idx) -> List[TrackResult]` where `TrackResult` contains: `track_id`, `bbox_xyxy`, `mask` (binary np.ndarray), `confidence`, `state` (from state machine), `mask_area`.
7. Add env var `SWAY_TRACKER_ENGINE` (choices: `sam2mot`, `solidtrack`, `sam2_memosort_hybrid`). When `sam2mot` or `sam2_memosort_hybrid`: use `SAM2PrimaryTracker`. When `solidtrack`: use existing BoxMOT path unchanged.
8. In `main.py`, replace the tracker instantiation based on `SWAY_TRACKER_ENGINE`. The downstream code receives the same `List[TrackResult]` regardless of engine.
9. Refactor `sway/hybrid_sam_refiner.py`: extract SAM2 model loading into a shared utility so both old and new paths can use it without loading the model twice.

### Environment variables (defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SWAY_SAM2_REINVOKE_STRIDE` | `30` | Periodic detector re-invocation interval (frames) |
| `SWAY_SAM2_CONFIDENCE_REINVOKE` | `0.40` | Below this logit confidence → force re-invoke next frame |
| `SWAY_SAM2_MEMORY_FRAMES` | `120` | Max frames retained in SAM2 memory bank (+ enrollment) |
| `SWAY_TRACKER_ENGINE` | (existing default) | `sam2mot`, `solidtrack`, or `sam2_memosort_hybrid` |

## Technical Considerations & Performance

* **Architecture Notes:** SAM2's video predictor maintains a memory bank of past frame features. Memory grows with video length. For a 2-minute video at 30fps (3600 frames), implement memory pruning: keep only the last `SWAY_SAM2_MEMORY_FRAMES` (default 120) frames in the bank, plus the enrollment frame. This bounds VRAM usage.
* **Edge Cases & I/O:** When a mask collapses to zero area (full occlusion), do NOT remove it from SAM2's memory bank — just mark the track as DORMANT. When the detector re-invocation finds a match, resume propagation from the new detection. VRAM budget: SAM2 base model ~2GB + per-frame features ~0.5MB × memory_frames. At 120 frames: ~2.06GB total. Fits alongside YOLO and pose models on 24GB GPU.

## Validation & Testing

* **Verification:** Run on bigtest. Compare mask-based tracking vs current SolidTrack box-based tracking: (a) track continuity through crossovers, (b) mask quality (visual inspection of 50 frames), (c) HOTA score. The masks should perfectly isolate each dancer even during crossings.
* **Metrics:** On bigtest, SAM2 tracker should achieve ≥ 0.55 HOTA (up from 0.495 SolidTrack). Mask IoU vs GT segmentation (if available) ≥ 0.80. Re-invocation should fire on ≤ 20% of frames.

## Swapability & Experimentation

**Factory integration:** `SAM2PrimaryTracker` implements a `BaseTracker` interface with `process_frame(frame, frame_idx) -> List[TrackResult]`. The `sway/tracker_factory.py` dispatches:

```python
# sway/tracker_factory.py
def create_tracker(env):
    engine = env.get("SWAY_TRACKER_ENGINE", "solidtrack")
    if engine == "solidtrack":
        return SolidTrackWrapper(env)       # current BoxMOT path (unchanged)
    elif engine == "sam2mot":
        return SAM2PrimaryTracker(env)      # this plan
    elif engine == "sam2_memosort_hybrid":
        return SAM2MeMoSORTHybrid(env)      # this + PLAN_06
    elif engine == "memosort":
        return MeMoSORTWrapper(env)         # PLAN_06 standalone
```

**The `BaseTracker` interface** that ALL tracker engines must implement:

```python
class BaseTracker(ABC):
    def process_frame(self, frame, frame_idx) -> List[TrackResult]: ...
    def get_active_tracks(self) -> List[int]: ...
    def get_track_state(self, track_id) -> TrackState: ...
    def get_mask(self, track_id) -> Optional[np.ndarray]: ...
```

When `SWAY_TRACKER_ENGINE=solidtrack`, `get_mask()` returns None and mask-dependent features (COI, mask-guided pose, color histograms) fall back to bbox-based alternatives. This preserves the current pipeline as a baseline.

**Manual A/B recipe:**

```bash
# Compare all tracker engines on bigtest
python -m tools.pipeline_matrix_runs \
  --recipes tracker_solidtrack,tracker_sam2mot,tracker_sam2_memosort \
  --video data/ground_truth/bigtest/BigTest.mov \
  --stop-after-boundary after_phase_3 --compare
```

**Sweep readiness:** `SWAY_TRACKER_ENGINE` is a categorical choice tested in sweep phase S1. Within SAM2MOT, the continuous params (`SWAY_SAM2_REINVOKE_STRIDE`, `SWAY_COI_MASK_IOU_THRESH`, `SWAY_SAM2_MODEL`) are sweepable via `suggest_tracking_params()`. The SolidTrack params from sweep_v3 are automatically applied when `SWAY_TRACKER_ENGINE=solidtrack`.

## Integration & Next Steps

* **Outputs:** `SAM2PrimaryTracker` class with `process_frame()` returning `List[TrackResult]` (track_id, bbox, mask, confidence, state). Consumed by: PLAN_05 (COI uses mask IoU + logit history), PLAN_07 (enrollment uses initial masks), PLAN_12 (color histograms from mask-isolated pixels), PLAN_17 (mask-guided pose estimation).
* **Open Questions/Risks:** SAM2's video predictor API may require loading the entire video first (depends on implementation). If so, need a streaming adapter that feeds frames one at a time. Check `sam2.sam2_video_predictor.SAM2VideoPredictor` API carefully.
