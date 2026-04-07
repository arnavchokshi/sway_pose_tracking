# Integrate MotionBERT Multi-Person 3D Lifting

> **Implementation Phase:** I4 (Pose + 3D Upgrade) · **Lean Core** (single-person lifting) · Multi-person placement is experiment add-on
> **Swappable component:** `SWAY_LIFT_BACKEND` dispatches via lifter factory — choices: `motionagformer` (current default), `motionbert` (lean core target)

**Objective:** Replace or supplement the current MotionAGFormer 3D lifter with MotionBERT (ICCV 2023), which uses a Dual-stream Spatio-temporal Transformer (DSTformer) pretrained on motion recovery from noisy 2D inputs. More importantly, add multi-person joint estimation: instead of lifting each dancer independently (current approach), estimate a shared floor plane from monocular depth, then place all 3D skeletons on that plane in a physically consistent coordinate frame. This enables formation accuracy critique ("Dancer A is 30cm too far right") which requires knowing relative 3D positions.

## Inputs & Dependencies

* **Upstream Data:** Per-dancer 2D keypoint sequences (COCO-17 format, T frames × 17 joints × 3 coords [x, y, confidence]), per-keypoint confidence levels (from PLAN_17), video frames (for monocular depth estimation).
* **Prior Steps:** PLAN_17 (mask-guided pose with confidence — provides the 2D keypoints and confidence levels that MotionBERT lifts).

## Step-by-Step Implementation

1. Create `sway/motionbert_lifter.py`.
2. Clone the MotionBERT repository or install from pip if available. Download pretrained checkpoint: `MotionBERT_pretrained.pth` (pretrained DSTformer). Store in `models/`.
3. Implement class `MotionBERTLifter`:
   - `__init__(self, checkpoint_path, device, clip_length=243)`: Load DSTformer model. MotionBERT expects temporal clips of 243 frames.
   - `lift_single(self, keypoints_2d: np.ndarray) -> np.ndarray`:
     - **a.** Input: `(T, 17, 2)` — 2D keypoint positions.
     - **b.** Pad or chunk into clips of length 243. For videos shorter than 243 frames: pad with the last frame. For longer videos: use a sliding window with 50% overlap and average predictions in overlapping regions.
     - **c.** Normalize: center on hip midpoint, scale to `[-1, 1]` range.
     - **d.** Run MotionBERT forward pass. Output: `(T, 17, 3)` — 3D joint positions per frame.
     - **e.** Return the 3D sequence.
4. Implement multi-person placement on a shared floor plane:
   - **a.** Run monocular depth estimation on the full frame using a lightweight model (Depth Anything V2 Small — already commonly available, ~5ms per frame). Extract per-dancer root depth from the depth map at each dancer's hip midpoint.
   - **b.** Estimate floor plane: use RANSAC on the bottom 20% of the depth map. The floor plane gives a world-coordinate ground reference.
   - **c.** For each dancer: place their 3D skeleton on the floor plane at the depth-estimated position. Translation: `(x_pixel * depth / focal_length, y_pixel * depth / focal_length, depth)`.
   - **d.** Result: all dancers' 3D skeletons are in a shared world coordinate frame.
5. Implement `lift_multi_person(self, all_keypoints_2d: Dict[int, np.ndarray], depth_map: np.ndarray, camera_intrinsics) -> Dict[int, np.ndarray]`:
   - **a.** For each dancer: call `lift_single()` to get 3D skeleton in camera-relative coordinates.
   - **b.** Apply floor-plane correction: project onto the estimated floor.
   - **c.** Return `{dancer_id: (T, 17, 3)}` in shared world coordinates.
6. Respect confidence levels from PLAN_17: only lift joints with confidence ≥ MEDIUM. LOW and NOT_VISIBLE joints: set their 3D coordinates to NaN and propagate the NOT_VISIBLE flag.
7. Add env vars:
   - `SWAY_LIFT_BACKEND` — options: `motionagformer`, `motionbert`, default `motionagformer`
   - `SWAY_LIFT_MULTI_PERSON` — bool, default `0`
   - `SWAY_LIFT_DEPTH_SCENE` — bool, default `0`
8. Wire into `main.py`: after pose estimation and smoothing, run 3D lifting. The output replaces the current MotionAGFormer output in `data.json`.

## Technical Considerations & Performance

* **Architecture Notes:** MotionBERT's DSTformer is ~15ms per 243-frame clip per person on GPU. For 5 dancers in a 2-min video: ~5 × (3600/243) × 15ms ≈ 1.1 seconds total — negligible. Depth Anything V2 Small is ~5ms per frame: 3600 × 5ms = 18 seconds for the full video. The depth estimation is the bottleneck; run it once and cache the full-video depth map.
* **Edge Cases & I/O:** When a dancer is DORMANT (no 2D keypoints), their 3D skeleton is NaN for those frames. Do not interpolate through DORMANT gaps — the gap is real and the critique should acknowledge it. Camera intrinsics: if not known, use the pinhole model with default FOV from `SWAY_PINHOLE_FOV_DEG` (existing env var).

## Validation & Testing

* **Verification:** Run MotionBERT on gymtest (the most dynamic sequence). Compare 3D skeleton quality vs MotionAGFormer: (a) visual inspection of 3D renders, (b) bone length consistency (MotionBERT should produce more stable bone lengths), (c) floor plane accuracy (dancers should not float or sink).
* **Metrics:** 3D bone length variance should be ≤ 5% of mean bone length (MotionBERT vs ~8% for MotionAGFormer). Multi-person floor plane error ≤ 10cm (measured by checking if dancers' feet touch the estimated floor).

## Integration & Next Steps

* **Outputs:** `{dancer_id: (T, 17, 3)}` 3D keypoint sequences in shared world coordinates. Consumed by PLAN_19 (critique scoring uses 3D positions for formation accuracy and synchronization).
* **Open Questions/Risks:** MotionBERT was trained on single-person data (Human3.6M). The multi-person placement relies on depth estimation accuracy, which may be poor in cluttered dance scenes. Validate depth estimation quality on bigtest before committing.

## Swapability & Experimentation

**Factory integration:** 3D lifters implement a common interface:

```python
class Base3DLifter(ABC):
    def lift(self, keypoints_2d: np.ndarray) -> np.ndarray: ...
    # Input: (T, 17, 2), Output: (T, 17, 3)
```

Dispatch via `SWAY_LIFT_BACKEND`:
- `motionagformer` → current production (unchanged baseline)
- `motionbert` → this plan (lean core target)

Multi-person placement (`SWAY_LIFT_MULTI_PERSON=1`) is orthogonal to the lifter choice — it's an additional post-processing step that places all per-person 3D skeletons on a shared floor plane. Test single-person MotionBERT first (A/B vs MotionAGFormer), then enable multi-person.

**Manual A/B recipe:**

```bash
# Compare 3D lifters
python -m tools.pipeline_matrix_runs \
  --recipes lift_motionagformer,lift_motionbert,lift_motionbert_multi \
  --video data/ground_truth/gymtest/gymtest.mov --compare
```

**Sweep readiness:** `SWAY_LIFT_BACKEND` is categorical (A/B test in sweep phase S6). No continuous params to sweep within the lifter itself — quality is measured by bone length consistency and floor plane accuracy.
