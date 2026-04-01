# Integrate MoCos Skeleton-Based Gait Re-ID

> **Implementation Phase:** I6 (Selective Advanced Modules) · **EXPERIMENT ADD-ON — not lean core**
> Implement only after lean core Re-ID (PLAN_08 + PLAN_09 + PLAN_12) passes Gate 2.
> Test as a standalone A/B experiment. Only enable permanently if it measurably improves Re-ID accuracy after occlusion on bigtest.
> **Note:** "MoCos" as a specific named model is not verified in major conference indexes per Perplexity fact-check. The general approach (skeleton-based gait re-ID via graph transformers) is well-supported. Use the best available open-source skeleton gait model when implementing.

**Objective:** Add a costume-invariant identity signal based on skeletal motion patterns. MoCos (Motif-guided Graph Transformers, AAAI 2025) processes 30–60 frames of 3D skeleton poses as a spatiotemporal graph, producing a gait identity embedding that captures biomechanical signatures — joint flexibility, stride length, shoulder width ratio, center-of-gravity shift patterns. This is orthogonal to appearance: it captures who you ARE (biomechanics), not what you LOOK LIKE (costume). Critical for dancers in matching outfits.

## Inputs & Dependencies

* **Upstream Data:** Per-dancer 3D skeleton sequences (17 COCO joints × 3 coords × T frames). The current pipeline already computes 3D poses via MotionAGFormer. Requires a minimum window of 30 frames of continuous 3D pose data.
* **Prior Steps:** 3D lifting must produce skeleton sequences (existing MotionAGFormer or future PLAN_18 MotionBERT). PLAN_07 (enrollment gallery stores gait embeddings after 60 frames).

## Step-by-Step Implementation

1. Create `sway/mocos_extractor.py`.
2. Obtain MoCos: clone the official repository or extract the model architecture. MoCos uses a Graph Transformer with two motif types: structural motifs (local joint correlations) and gait collaborative motifs (inter-joint movement patterns).
3. Download pretrained MoCos checkpoint (trained on CASIA-B gait dataset or similar). Store in `models/mocos_gait.pth`.
4. Implement class `MoCosExtractor`:
   - `__init__(self, checkpoint_path, device, min_window)`: Load model. `min_window` from `SWAY_REID_SKEL_MIN_WINDOW` (default 30).
   - `extract(self, skeleton_sequence: np.ndarray) -> Optional[np.ndarray]`:
     a. Input: `skeleton_sequence` shape (T, 17, 3) — T frames of COCO-17 3D joints.
     b. If T < `min_window`: return None (insufficient data).
     c. Normalize skeleton: center on hip midpoint, scale by torso length (shoulder-to-hip distance). This makes the embedding body-size-invariant.
     d. Construct the spatiotemporal graph: nodes = joints × frames, edges = spatial (bone connections within each frame) + temporal (same joint across adjacent frames).
     e. Run MoCos forward pass. Returns a 256-d gait identity embedding.
     f. L2-normalize.
   - `compare(self, gallery_emb, query_emb) -> float`: cosine distance.
5. Integrate with enrollment (PLAN_07): after 60 frames of tracking, extract MoCos embedding and store in `DancerGallery.skeleton_gait_embedding`. This is a one-time deferred enrollment step.
6. For re-ID: when a dancer re-emerges and accumulates ≥ 30 frames of new pose data, extract a gait embedding and compare against the gallery.
7. Add env vars: `SWAY_REID_SKEL_MODEL` (default `mocos`), `SWAY_REID_SKEL_MIN_WINDOW` (default 30, sweep 15–60).
8. Gait embedding is one input to the multi-signal fusion engine (PLAN_13). Confidence gate: only contributes when `T >= min_window`.

## Technical Considerations & Performance

* **Architecture Notes:** MoCos inference on a 60-frame skeleton is ~2ms on GPU (small model, sparse graph). The bottleneck is accumulating enough frames — gait signal is unavailable for the first 1 second after re-emergence.
* **Edge Cases & I/O:** Dancers performing identical choreography will have SIMILAR but not identical gait patterns — individual biomechanics (flexibility, proportions) create distinguishable signatures. However, discrimination may be weaker than appearance-based methods when dancers are very physically similar. This is why gait is one signal among six, not the sole identifier.

## Validation & Testing

* **Verification:** Extract gait embeddings for all 5 bigtest dancers over multiple 60-frame windows throughout the video. Compute pairwise distances. Verify: (a) same-dancer distance < 0.4 across different time windows, (b) different-dancer distance > 0.5.
* **Metrics:** Gait-only re-ID accuracy should be ≥ 75% (sufficient as a complementary signal, not required to work alone).

## Integration & Next Steps

* **Outputs:** `MoCosExtractor` class with `extract()` and `compare()`. Consumed by PLAN_07 (deferred gallery update), PLAN_13 (re-ID fusion as Signal 3).
* **Open Questions/Risks:** MoCos was trained on walking/gait data, not dance. Dance movements are more varied and complex than walking. May need fine-tuning on dance skeleton data or use a dance-specific motion encoder.
