# Build Color Histogram Re-ID Signal

> **Implementation Phase:** I2 (Re-ID Upgrade) · **Lean Core** · **Sweep Gate:** Gate 2 — Re-ID accuracy after occlusion > 90%

**Objective:** Add a fast, simple, and robust costume-color matching signal. Using SAM2 masks to isolate each person's pixels, extract per-region (upper body, lower body, shoes) color histograms. Even "matching" costumes often have subtle color differences under different stage lighting angles. This is the fastest re-ID signal to compute and provides a useful coarse filter before expensive embedding comparisons.

## Inputs & Dependencies

* **Upstream Data:** Person crops (BGR numpy) + SAM2 segmentation masks (binary numpy).
* **Prior Steps:** PLAN_04 (SAM2 masks for pixel isolation). Can also work without masks (use full crop), but accuracy drops.

## Step-by-Step Implementation

1. Create `sway/color_histogram_reid.py`.
2. Implement class `ColorHistogramExtractor`:
   - `__init__(self, color_space, n_bins, regions)`: `color_space` from `SWAY_REID_COLOR_SPACE` (default `hsv`), `n_bins` from `SWAY_ENROLLMENT_COLOR_BINS` (default 32).
   - `extract(self, crop: np.ndarray, mask: np.ndarray, keypoints: np.ndarray) -> Dict[str, np.ndarray]`:
     a. Convert crop to target color space (cv2.cvtColor to HSV, LAB, or keep RGB).
     b. Apply mask: zero out pixels outside the person's mask.
     c. Split the masked region into 3 vertical zones using pose keypoints:
        - Upper: above hip keypoints (shirt/top)
        - Lower: hips to ankles (pants/skirt)
        - Shoes: below ankles (feet region)
     d. For each zone: compute a 3-channel histogram (n_bins per channel). Concatenate channels. L1-normalize.
     e. Return `{"upper": hist_upper, "lower": hist_lower, "shoes": hist_shoes}`.
   - `compare(self, gallery_hists: Dict, query_hists: Dict) -> float`:
     a. For each shared region, compute histogram intersection (cv2.compareHist with HISTCMP_INTERSECT) or Bhattacharyya distance.
     b. Return mean distance across regions. Lower = more similar.
3. Store histograms in enrollment gallery (PLAN_07) as `DancerGallery.color_histograms`.
4. Add env vars: `SWAY_REID_COLOR_SPACE` (default `hsv`), `SWAY_ENROLLMENT_COLOR_BINS` (default 32, sweep 16–64).
5. Color signal is always available (unlike face or gait, which have minimum quality requirements). It contributes to every re-ID event.

## Technical Considerations & Performance

* **Architecture Notes:** Histogram computation is pure CPU, ~0.5ms per crop. This is by far the cheapest re-ID signal. Its value is as a coarse pre-filter: if color is very different, skip expensive embedding comparisons.
* **Edge Cases & I/O:** Stage lighting changes can shift color histograms. Using HSV color space (Hue is lighting-invariant) mitigates this. If all dancers wear truly identical costumes (same color, same fabric), this signal will have zero discriminative power — but it costs nothing to compute, so include it anyway.

## Validation & Testing

* **Verification:** Extract color histograms for all 5 bigtest dancers across 100 frames each. Compute pairwise Bhattacharyya distances. Verify: (a) same-dancer distance is low, (b) if costumes differ at all, different-dancer distance is measurably higher.
* **Metrics:** If costumes have any color difference, color-only re-ID accuracy should be ≥ 70%. If costumes are truly identical, accuracy will be ~random (20% for 5 dancers) — acceptable because this is one signal among six.

## Integration & Next Steps

* **Outputs:** `ColorHistogramExtractor` with `extract()` and `compare()`. Consumed by PLAN_07 (enrollment), PLAN_13 (fusion Signal 5, always active).
* **Open Questions/Risks:** LAB color space may outperform HSV in some lighting conditions. Include as a sweep option.
