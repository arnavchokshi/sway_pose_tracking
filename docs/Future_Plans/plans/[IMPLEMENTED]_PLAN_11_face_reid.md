# Integrate ArcFace Face Recognition Signal

> **Implementation Phase:** I6 (Selective Advanced Modules) · **EXPERIMENT ADD-ON — not lean core**
> Implement only after lean core Re-ID (PLAN_08 + PLAN_09 + PLAN_12) passes Gate 2.
> Test as a standalone A/B experiment. Face is the strongest individual signal when available, but it's only available on 30–50% of frames. The lean core (part + KPR + color + spatial) should handle most re-ID scenarios. Add face only if accuracy is still below target after Gate 2.

**Objective:** Add face recognition as an opportunistic re-ID signal. When a dancer's face is visible and unoccluded, ArcFace provides an extremely strong identity signal (99%+ accuracy on clean frontal faces). In dance footage, faces are available on maybe 30–50% of frames — but when available, face is the single most discriminative signal. It naturally complements skeleton gait (faces visible frontally, gait strongest in profile).

## Inputs & Dependencies

* **Upstream Data:** Person crops (BGR numpy) or full-frame images with bounding boxes.
* **Prior Steps:** None for model integration. PLAN_07 (enrollment gallery stores face embeddings).

## Step-by-Step Implementation

1. Create `sway/face_reid.py`.
2. Install `insightface` (pip install insightface). This provides both face detection (RetinaFace) and face recognition (ArcFace) in one package.
3. Implement class `FaceReIDExtractor`:
   - `__init__(self, model_name, device, min_face_size)`: Load `insightface.app.FaceAnalysis`. `min_face_size` from `SWAY_REID_FACE_MIN_SIZE` (default 40, meaning min 40px inter-eye distance).
   - `extract(self, person_crop: np.ndarray) -> Optional[np.ndarray]`:
     a. Run face detection within the person crop.
     b. If no face detected, or face size (inter-eye distance) < `min_face_size`: return None.
     c. If face detected and sufficiently large: extract ArcFace embedding (512-d).
     d. L2-normalize.
     e. Return the embedding.
   - `compare(self, gallery_emb, query_emb) -> float`: cosine distance.
4. For the enrollment gallery (PLAN_07): during the first 300 frames, scan each dancer's crops for face detections. Store the highest-quality (largest, most frontal) face embedding in the gallery.
5. Face signal is highly opportunistic: it is only available when the dancer faces the camera. The fusion engine (PLAN_13) sets `w_face = 0` when no face is detected.
6. Add env vars: `SWAY_REID_FACE_MODEL` (choices: `arcface`, `adaface`, default `arcface`), `SWAY_REID_FACE_MIN_SIZE` (default 40, sweep 20–60).

## Technical Considerations & Performance

* **Architecture Notes:** Face detection + ArcFace extraction is ~10ms per crop on GPU. Only run on ACTIVE tracks (face is unlikely visible during PARTIAL occlusion). For 5 dancers: ~50ms per frame. Can be parallelized with pose estimation.
* **Edge Cases & I/O:** Side-profile faces have lower recognition accuracy. `insightface` reports a face detection confidence — only use faces with confidence > 0.7. Dancers with very similar faces (siblings) may have close embeddings — this is why face is one signal among six.

## Validation & Testing

* **Verification:** On bigtest: count how many frames per dancer have a detectable face. Extract face embeddings. Verify same-dancer face distance < 0.3, different-dancer distance > 0.5.
* **Metrics:** Face detection should fire on ≥ 25% of frames per dancer. When available, face-only re-ID accuracy should be ≥ 95%.

## Integration & Next Steps

* **Outputs:** `FaceReIDExtractor` with `extract()` and `compare()`. Consumed by PLAN_07 (enrollment), PLAN_13 (fusion Signal 4).
* **Open Questions/Risks:** Stage makeup may alter facial appearance between practices. The gallery should store multiple face embeddings (different angles, different lighting) to handle this.
