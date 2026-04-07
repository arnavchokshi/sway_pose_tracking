# Build Hybrid YOLO + DETR Detection Dispatcher

> **Implementation Phase:** I3 (Detection Upgrade) · **Lean Core** · **Sweep Gate:** Gate 3 — Detection AP on overlap frames improves; no regression on easy sequences
> Toggle: `SWAY_DETECTOR_HYBRID=0|1`. When OFF, single detector only. When ON, uses `sway/detector_factory.py` to get both YOLO and the precision detector.

**Objective:** Implement a smart detection dispatcher that runs YOLO on every frame as a cheap first pass and conditionally invokes Co-DETR/RT-DETR only on frames where overlapping detections are detected. This avoids paying the 3–5x DETR cost on every frame while getting its NMS-free precision exactly when it matters (during occlusion events).

## Inputs & Dependencies

* **Upstream Data:** Raw video frames (BGR numpy arrays).
* **Prior Steps:** PLAN_02 (Co-DETR backend must be implemented and loadable).

## Step-by-Step Implementation

1. Create `sway/hybrid_detector.py`.
2. Implement class `HybridDetector`:
   - `__init__(self, yolo_detector, detr_detector, overlap_iou_trigger: float)`: Takes both detector instances. `overlap_iou_trigger` defaults from `SWAY_HYBRID_OVERLAP_IOU_TRIGGER` (default 0.30).
   - `detect(self, frame: np.ndarray) -> Tuple[List[Detection], str]`: Returns detections and a tag (`"yolo"` or `"detr"`) indicating which detector produced them.
3. Implement the dispatch logic inside `detect()`:
   a. Run YOLO on the frame. Get `yolo_dets`.
   b. Compute pairwise IoU between all YOLO detection boxes using `torchvision.ops.box_iou` (already available).
   c. If `max(pairwise_iou) > overlap_iou_trigger` (any pair of boxes overlaps significantly): invoke DETR on the same frame. Return DETR detections.
   d. Else: return YOLO detections directly.
4. Add a second trigger: if SAM2 mask propagation reports low confidence on any active track (via a callback or shared state), force DETR on the next frame regardless of YOLO overlap. This handles drift recovery.
5. Add env var `SWAY_DETECTOR_HYBRID` (bool, default `1`). When `0`, use only the primary detector (no hybrid dispatch).
6. Add logging: count how many frames per sequence used YOLO vs DETR. Log at end of run: "Hybrid detection: YOLO {X}% of frames, DETR {Y}%."
7. Wire into `sway/tracker.py`: replace the single YOLO call with `HybridDetector.detect()`.

### Environment variables (defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SWAY_HYBRID_OVERLAP_IOU_TRIGGER` | `0.30` | Max pairwise IoU above this → run DETR |
| `SWAY_HYBRID_COOLDOWN_FRAMES` | `5` | After DETR fires, skip DETR for this many frames (sustained clusters) |
| `SWAY_DETECTOR_HYBRID` | `1` | `0` = primary detector only, no hybrid |

## Technical Considerations & Performance

* **Architecture Notes:** The IoU computation between YOLO boxes is O(N²) where N is the number of detections per frame (typically 2–8 dancers). This is negligible. The expensive part is the DETR call, which should only fire on 10–30% of frames in typical dance videos (only during crossovers/clusters).
* **Edge Cases & I/O:** On videos with constant overlap (e.g., very tight formations), DETR may fire on every frame, negating the speed benefit. Add a cooldown: if DETR fired on frame N, skip DETR for the next `SWAY_HYBRID_COOLDOWN_FRAMES` (default 5) frames regardless of overlap. This prevents DETR from running on every frame during sustained clusters.

## Validation & Testing

* **Verification:** Run on bigtest with logging. Verify: (a) DETR fires specifically during group occlusion segments, (b) YOLO handles all clear-separation frames, (c) detection quality on overlap frames improves vs YOLO-only. A/B compare aggregate HOTA: hybrid vs YOLO-only vs DETR-only.
* **Metrics:** Hybrid mode should achieve ≥ 95% of DETR-only detection accuracy while using DETR on ≤ 30% of frames. Processing speed should be ≤ 1.5x YOLO-only (vs 3–5x for DETR-only).

## Integration & Next Steps

* **Outputs:** `HybridDetector` class with `detect()` method returning `(detections, source_tag)`. Consumed by PLAN_04 (SAM2 tracker uses detections for initialization and re-invocation).
* **Open Questions/Risks:** The overlap IoU trigger threshold needs tuning. Start with 0.30 and sweep in range 0.15–0.50 in a future Optuna study.
