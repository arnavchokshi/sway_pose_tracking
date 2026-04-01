# Integrate Co-DETR / RT-DETR Detection Backend

> **Implementation Phase:** I3 (Detection Upgrade) · **Lean Core** · **Sweep Gate:** Gate 3 — A/B test `SWAY_DETECTOR_HYBRID=1` vs `0`; no regression on easy sequences
> **Swappable component:** `SWAY_DETECTOR_PRIMARY` dispatches via `sway/detector_factory.py` (see FUTURE_PIPELINE.md §12.1)

**Objective:** Add Co-DETR (or Co-DINO / RT-DETR) as an alternative person detection backbone alongside the existing YOLO26l. DETR-family detectors eliminate NMS entirely via learned object queries with Hungarian matching, achieving ~66 AP on COCO (vs ~55 for YOLO) with zero duplicate-detection ambiguity. This directly addresses the NMS failure mode where overlapping dancers get suppressed.

## Inputs & Dependencies

* **Upstream Data:** Raw video frames (BGR numpy arrays, same as YOLO currently receives).
* **Prior Steps:** None — this is a standalone detection backend. It produces the same output format as the YOLO detector.

## Step-by-Step Implementation

1. Create `sway/detr_detector.py`.
2. Install `detrex` (Meta's DETR ecosystem) or use the `mmdet` Co-DETR implementation. Preferred: use the official Co-DETR checkpoint from the Swin-L backbone variant (highest AP). Alternative: RT-DETR from the `ultralytics` package (already a project dependency) for a faster but still NMS-free option.
3. Implement class `DETRDetector`:
   - `__init__(self, model_name: str, device: str, conf_threshold: float)`: Load checkpoint. Supported `model_name` values: `co_detr_swinl`, `co_dino_swinl`, `rt_detr_l`, `rt_detr_x`.
   - `detect(self, frame: np.ndarray) -> List[Detection]`: Run inference. Return list of `Detection(bbox_xyxy, confidence, class_id)`. Filter to `class_id == 0` (person). Apply `conf_threshold`.
   - Internal: handle image preprocessing (resize, normalize) per model requirements.
4. Ensure output format matches existing YOLO output: `List[Detection]` where each `Detection` has `.bbox` (x1, y1, x2, y2), `.confidence`, `.class_id`. This is the contract that `tracker.py` consumes.
5. Add env var `SWAY_DETECTOR_PRIMARY` (choices: `yolo26l_dancetrack`, `co_detr`, `co_dino`, `rt_detr`). Default: `yolo26l_dancetrack` (no behavior change until explicitly opted in).
6. In `sway/tracker.py`, at the detection call site, dispatch to `DETRDetector` or existing YOLO based on `SWAY_DETECTOR_PRIMARY`.
7. When using DETR, skip all NMS-related code (`SWAY_PRETRACK_NMS_IOU` is ignored). DETR already produces deduplicated detections.
8. Add model weights to `models/` directory structure. Provide a download script `tools/download_detr_weights.py` that fetches the checkpoint from the official release URL.

## Technical Considerations & Performance

* **Architecture Notes:** Co-DETR is 3–5x slower than YOLO per frame. This is acceptable for offline processing. RT-DETR is ~1.5x slower than YOLO and may be the better default. Benchmark both on bigtest before committing.
* **Edge Cases & I/O:** DETR models expect specific image sizes (800px short side for Co-DETR). Respect `SWAY_DETECT_SIZE` by scaling the image before inference, but note that DETR's performance may differ at non-standard resolutions. GPU memory: Co-DETR Swin-L requires ~4GB VRAM at 800px; ensure this fits alongside SAM2.

## Validation & Testing

* **Verification:** Run DETR and YOLO on the same 100 frames from bigtest. Compare: (a) detection count per frame, (b) AP against GT boxes, (c) rate of duplicate/missing detections in dense clusters.
* **Metrics:** Co-DETR must achieve ≥ 60 AP on the 5 GT sequences (vs YOLO's ~55 AP). Zero NMS-related false suppressions in group occlusion frames. Inference ≤ 200ms/frame on A10 GPU.

## Swapability & Experimentation

**Factory integration:** `DETRDetector` must implement the same `detect(frame) -> List[Detection]` interface as the YOLO detector. The `sway/detector_factory.py` dispatches based on `SWAY_DETECTOR_PRIMARY`:

```python
# sway/detector_factory.py
def create_detector(env):
    primary = env.get("SWAY_DETECTOR_PRIMARY", "yolo26l_dancetrack")
    if primary.startswith("yolo"):
        return YOLODetector(env)
    elif primary in ("co_detr", "co_dino"):
        return DETRDetector(model_name=primary, ...)
    elif primary == "rt_detr":
        return DETRDetector(model_name="rt_detr", ...)
```

**Manual A/B recipe:**

```bash
# Compare YOLO-only vs Co-DETR-only on bigtest
python -m tools.pipeline_matrix_runs \
  --recipes detector_yolo,detector_codetr,detector_rtdetr \
  --video data/ground_truth/bigtest/BigTest.mov \
  --stop-after-boundary after_phase_3 --compare
```

**Sweep readiness:** After this plan lands, `SWAY_DETECTOR_PRIMARY` is a categorical sweep param (choices: `yolo26l_dancetrack`, `co_detr`, `co_dino`, `rt_detr`). Include in sweep phase S1 via `suggest_detection_params()`.

## Integration & Next Steps

* **Outputs:** `DETRDetector` class with the same `detect()` interface as YOLO. Consumed by PLAN_03 (hybrid dispatch) and PLAN_04 (SAM2 tracker initialization).
* **Open Questions/Risks:** Co-DETR Swin-L checkpoint is ~800MB. Need to verify it fits in the `models/` directory and the download script handles auth if needed. RT-DETR via ultralytics may be simpler to integrate since ultralytics is already a dependency.
