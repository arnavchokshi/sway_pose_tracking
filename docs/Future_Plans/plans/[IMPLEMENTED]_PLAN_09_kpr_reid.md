# Integrate KPR Keypoint-Prompted Re-ID

> **Implementation Phase:** I2 (Re-ID Upgrade) · **Lean Core** · **Sweep Gate:** Gate 2 — Re-ID accuracy after occlusion > 90%, bigtest HOTA > 0.58

**Objective:** Add KPR (Keypoint-Prompted Re-ID) as a specialized re-ID signal for multi-person ambiguity — the situation where multiple people are visible in the same bounding box, making standard crop-based re-ID impossible. KPR uses pose keypoints as explicit prompts to specify which person inside an occluded crop to identify. For bigtest's group occlusion frames (3–4 dancers overlapping), this is the correct tool.

## Inputs & Dependencies

* **Upstream Data:** Person crops (BGR numpy), 2D pose keypoints (COCO-17 format with x, y, confidence per joint), SAM2 masks (optional, for mask-isolated cropping).
* **Prior Steps:** Pose keypoints must be available. In the future pipeline, pose runs on ACTIVE/PARTIAL tracks (PLAN_17). For re-ID purposes, a lightweight fast pose estimate (RTMPose) can be run first.

## Step-by-Step Implementation

1. Create `sway/kpr_extractor.py`.
2. Obtain the KPR model: clone the official KPR repository or extract the model definition. KPR is built on top of a standard re-ID backbone (ResNet-50) with a keypoint-prompting module that attends to specific spatial locations.
3. Download pretrained KPR checkpoint (trained on Occluded-Duke, Market-1501 with occlusion augmentation). Store in `models/kpr_r50.pth`.
4. Implement class `KPRExtractor`:
   - `__init__(self, checkpoint_path, device)`: Load model.
   - `extract(self, crop: np.ndarray, keypoints: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray`:
     a. If mask provided: zero out pixels outside mask.
     b. Normalize keypoint coordinates relative to crop dimensions: `kp_norm = (kp_xy - crop_top_left) / crop_size`.
     c. Create a keypoint heatmap: for each visible keypoint (confidence > 0.3), place a 2D Gaussian at its normalized position. Stack into a K-channel heatmap (K=17 for COCO).
     d. Concatenate the heatmap with the RGB image as additional input channels (total: 3 + 17 = 20 channels, or adapt to the model's expected input).
     e. Run KPR forward pass. Returns a 2048-d embedding.
     f. L2-normalize the embedding.
   - `compare(self, gallery_emb: np.ndarray, query_emb: np.ndarray) -> float`: cosine distance.
5. The key insight: in a multi-person crop, standard re-ID extracts features from ALL people in the box. KPR's keypoint heatmap tells the model "focus on the person at THESE joint locations," effectively spotlighting one person.
6. Hybrid with SAM2: crop using SAM2 mask (removes other people's pixels) + KPR keypoint prompting (tells model where the target person is). Double assurance.
7. Add env var `SWAY_REID_KPR_ENABLED` (bool, default `1`). This signal is only activated during re-ID when the query crop contains multi-person overlap (detected via Mask IoU > 0.1 with any other track).
8. KPR embedding is one input to the multi-signal fusion engine (PLAN_13).

## Technical Considerations & Performance

* **Architecture Notes:** KPR's forward pass is ~8ms on GPU — slightly slower than BPBreID due to the heatmap processing. Only invoked during multi-person overlap events, not on every frame. Expected: 5–15% of frames on bigtest.
* **Edge Cases & I/O:** When all keypoints have low confidence (heavily occluded person), the keypoint heatmap is nearly blank — KPR degrades to a standard re-ID model. This is acceptable; other signals (color, spatial) carry the load in this case.

## Validation & Testing

* **Verification:** On bigtest crossover frames: extract KPR embeddings for each dancer in an overlapping cluster. Verify same-dancer embeddings cluster together (cosine distance < 0.3) even when crops contain multiple people. Compare vs BPBreID on the same crops.
* **Metrics:** KPR should achieve ≥ 85% correct identification in multi-person crops where standard re-ID drops to ≤ 60%.

## Integration & Next Steps

* **Outputs:** `KPRExtractor` class with `extract()` and `compare()`. Consumed by PLAN_13 (re-ID fusion engine uses KPR as Signal 2).
* **Open Questions/Risks:** KPR was designed for pedestrian re-ID. Dance-specific poses (splits, lifts, floor work) may produce unusual keypoint patterns not seen in training. May need fine-tuning on dance data.
