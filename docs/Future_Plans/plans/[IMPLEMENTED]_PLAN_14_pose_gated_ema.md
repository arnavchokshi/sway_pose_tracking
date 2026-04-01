# Build Pose-Gated EMA Gallery Manager

> **Implementation Phase:** I2 (Re-ID Upgrade) · **Lean Core** · **Sweep Gate:** Gate 2 — Re-ID accuracy after occlusion > 90%

**Objective:** Implement a gallery update mechanism that only updates a dancer's identity embeddings when the current frame is trustworthy. When a dancer is spatially isolated with an extended pose, the embedding is clean — update the gallery aggressively (high EMA alpha). When the dancer is crouched in a cluster with overlapping people, the embedding is contaminated — freeze the gallery entirely (alpha = 0). This 2-parameter mechanism prevents gallery pollution, which is the primary cause of cascading identity errors.

## Inputs & Dependencies

* **Upstream Data:** Per-frame, per-track: (a) current embeddings from re-ID extractors (part-based, color, face), (b) bounding box of the dancer, (c) bounding boxes of all other dancers (to compute isolation distance), (d) pose keypoints with confidence scores (to compute pose quality), (e) track state from the state machine (PLAN_01).
* **Prior Steps:** PLAN_01 (state machine — only ACTIVE tracks update gallery), PLAN_07 (enrollment gallery is the data structure being updated), PLAN_08 (BPBreID provides part embeddings to update).

## Step-by-Step Implementation

1. Create `sway/pose_gated_ema.py`.
2. Implement class `PoseGatedEMA`:
   - `__init__(self, alpha_high, alpha_low, isolation_dist, pose_quality_thresh)`:
     - `alpha_high` from `SWAY_REID_EMA_ALPHA_HIGH` (default 0.15, sweep 0.05–0.30).
     - `alpha_low` from `SWAY_REID_EMA_ALPHA_LOW` (default 0.00, LOCKED — never update in clusters).
     - `isolation_dist` from `SWAY_REID_EMA_ISOLATION_DIST` (default 1.5, sweep 1.0–3.0). Measured in bbox-height fractions.
     - `pose_quality_thresh` from `SWAY_REID_EMA_POSE_QUALITY_THRESH` (default 0.60, sweep 0.40–0.80).
3. Implement `compute_alpha(self, dancer_bbox, all_bboxes, keypoint_confidences, track_state) -> float`:
   a. If `track_state` is DORMANT or LOST: return 0.0 (never update).
   b. If `track_state` is PARTIAL: return `alpha_high * 0.3` (reduced rate, only for visible parts).
   c. Compute `isolation_score`: min distance from this dancer's bbox center to any other dancer's bbox center, divided by this dancer's bbox height. If > `isolation_dist`: dancer is isolated.
   d. Compute `pose_quality`: mean confidence of all visible keypoints (confidence > 0.1). If < `pose_quality_thresh`: pose is low quality.
   e. If isolated AND high pose quality: return `alpha_high`.
   f. Else: return `alpha_low` (0.00 = frozen).
4. Implement `update_gallery(self, gallery: DancerGallery, new_embeddings, alpha: float)`:
   a. For each embedding type in the gallery (part_embeddings, global_embedding, color_histograms):
      - `gallery.emb = (1 - alpha) * gallery.emb + alpha * new_emb`
   b. Face embedding: only update if new face embedding is available AND has higher quality than stored one.
   c. Gait embedding: NOT updated via EMA (it's a one-time computation from a window of poses).
5. Call `compute_alpha()` and `update_gallery()` for each ACTIVE track after feature extraction in the frame loop.
6. Log: for each frame, log per-dancer: alpha value, isolation_score, pose_quality. This enables debugging cascading ID errors.

## Technical Considerations & Performance

* **Architecture Notes:** This is pure arithmetic — zero GPU cost. The alpha computation is O(N²) for pairwise bbox distances (N = number of dancers), but N ≤ 10 so this is negligible.
* **Edge Cases & I/O:** At the start of the video (enrollment phase), all dancers should be isolated. Alpha should be high, building a strong initial gallery. The critical moment is during crossovers — alpha must drop to zero before embeddings are contaminated. The isolation_dist threshold controls this boundary.

## Validation & Testing

* **Verification:** On bigtest crossover segments: log alpha values for all dancers. Verify alpha drops to 0.0 when dancers enter a cluster and recovers to alpha_high when they separate. Compare gallery drift: with EMA gating, gallery embeddings should remain stable through crossovers; without gating, they drift toward the wrong person.
* **Metrics:** Gallery embedding stability (measured as cosine distance between enrollment embedding and post-crossover embedding): with gating ≤ 0.1 drift, without gating ≥ 0.3 drift. Re-ID accuracy after crossovers with gating ≥ 95%, without gating ~80%.

## Integration & Next Steps

* **Outputs:** `PoseGatedEMA` class. Consumed by `main.py` frame loop: called after feature extraction for each ACTIVE track.
* **Open Questions/Risks:** The isolation_dist threshold needs sweep tuning. If set too low, embeddings update during partial overlap; if too high, embeddings rarely update. Start at 1.5 bbox heights.
