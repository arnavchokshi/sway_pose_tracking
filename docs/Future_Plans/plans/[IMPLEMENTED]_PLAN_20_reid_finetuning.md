# Build Contrastive Re-ID Fine-Tuning Pipeline

> **Implementation Phase:** Between I2 and sweep phase S2 · **One-time preprocessing — run before Re-ID sweeps**
> Not a sweep parameter itself. Produces fine-tuned weights that become a new `SWAY_REID_PART_MODEL` option (`bpbreid_finetuned`).

**Objective:** Fine-tune the re-ID model (BPBreID or OSNet) on the user's own dance clips to close the domain gap between the pretrained dataset (Market-1501 — pedestrians in corridors) and the target domain (dancers in performance costumes under stage lighting). Even 500 same-ID/different-ID pairs extracted from ground truth clips produce a large accuracy improvement. This is a one-time preprocessing step that runs before any sweep or production run.

## Inputs & Dependencies

* **Upstream Data:** Ground truth video clips with CVAT/MOT annotations (track IDs per frame), or enrollment data with confirmed dancer identities.
* **Prior Steps:** PLAN_08 (BPBreID model to fine-tune) or existing OSNet. PLAN_07 (enrollment data provides confirmed identity labels). Ground truth data must exist in `data/ground_truth/`.

## Step-by-Step Implementation

1. Create `tools/finetune_reid.py`.
2. **Pair extraction:**
   - **a.** For each GT video: read the MOT annotation file. For each tracked dancer, extract N random crops from different frames.
   - **b.** Create **positive pairs**: two crops of the SAME dancer from DIFFERENT frames (same `track_id`, different frame). These should span different poses, angles, and lighting.
   - **c.** Create **negative pairs**: two crops of DIFFERENT dancers from the SAME frame (different `track_id`, same frame). These are hard negatives — same background, similar costumes.
   - **d.** Target: `SWAY_REID_FINETUNE_PAIRS` (default 500) pairs total. 50% positive, 50% negative.
   - **e.** Save pairs to `data/reid_finetune/pairs/` as `pair_XXXX_{a,b}.jpg` + `pairs.csv` with columns `pair_id, path_a, path_b, same_id`.
3. **Fine-tuning:**
   - **a.** Load pretrained BPBreID (or OSNet) checkpoint.
   - **b.** Use contrastive loss (or triplet loss with hard mining):
     - Contrastive loss: `L = y * D² + (1-y) * max(0, margin - D)²` where D = embedding distance, y = 1 for same-id, 0 for different-id.
     - Margin: 0.5 (standard for re-ID).
   - **c.** Training loop: `SWAY_REID_FINETUNE_EPOCHS` (default 20) epochs, `SWAY_REID_FINETUNE_LR` (default 1e-4), batch size 32.
   - **d.** Freeze the backbone (ResNet-50) for the first 5 epochs. Then unfreeze and train end-to-end with LR / 10.
   - **e.** Data augmentation: random horizontal flip, random erasing (occlusion simulation), color jitter (stage lighting simulation).
   - **f.** Save fine-tuned weights to `models/bpbreid_r50_sway_finetuned.pth`.
4. **Validation split:** Hold out 20% of pairs for validation. Compute re-ID accuracy on the validation set at each epoch. Save the checkpoint with best validation accuracy.
5. Add env vars:
   - `SWAY_REID_FINETUNE_ENABLED` — default `0`
   - `SWAY_REID_FINETUNE_PAIRS` — `500`
   - `SWAY_REID_FINETUNE_EPOCHS` — `20`
   - `SWAY_REID_FINETUNE_LR` — `1e-4`
   - `SWAY_REID_FINETUNE_BASE_MODEL` — default `bpbreid`
6. Add CLI interface: `python -m tools.finetune_reid --gt-dir data/ground_truth/ --output models/bpbreid_r50_sway_finetuned.pth`.
7. After fine-tuning: update `SWAY_REID_PART_MODEL` default to point to the fine-tuned weights. All subsequent sweeps use the fine-tuned model.

## Technical Considerations & Performance

* **Architecture Notes:** Fine-tuning is a one-time cost: ~10 minutes on a single GPU for 500 pairs × 20 epochs. The resulting model is the same size as the original — no inference cost increase.
* **Edge Cases & I/O:** If fewer than 100 pairs are available (limited GT): use heavy augmentation to synthesize additional training data. If no GT annotations exist: use the enrollment gallery + automated pair mining from the enrollment frames (each enrolled dancer provides positive pairs across the enrollment time window).

## Validation & Testing

* **Verification:** Compare re-ID accuracy before and after fine-tuning on bigtest: extract embeddings for all 5 dancers, compute pairwise distances, measure rank-1 accuracy.
* **Metrics:** Fine-tuned model re-ID accuracy should be ≥ 10% absolute improvement over pretrained model (e.g., from 80% to 90%+ rank-1 on bigtest).

## Integration & Next Steps

* **Outputs:** Fine-tuned re-ID model weights at `models/bpbreid_r50_sway_finetuned.pth`. Drop-in replacement for the pretrained weights — no code changes needed beyond pointing `SWAY_REID_PART_MODEL` to the new checkpoint.
* **Open Questions/Risks:** Fine-tuning on a small dataset risks overfitting to the specific costumes/lighting in the GT clips. Mitigate with strong augmentation and early stopping. If the team's video library grows, re-run fine-tuning periodically.
