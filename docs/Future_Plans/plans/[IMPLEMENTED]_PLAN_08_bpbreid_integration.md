# PLAN 08: BPBreID Part-Based Re-ID Integration

> **Implementation Phase:** I2 (Re-ID Upgrade) · **Lean Core** · **Sweep Gate:** Gate 2 — Re-ID accuracy after occlusion > 90%, bigtest HOTA > 0.58
> **Swappable component:** `SWAY_REID_PART_MODEL` dispatches via `sway/reid_factory.py` — choices: `bpbreid`, `paformer`, `osnet_x0_25` (see FUTURE_PIPELINE.md §12.1)

## Objective

Replace or augment OSNet's single **global** appearance embedding with **BPBreID (Body Part-Based Re-ID)**: separate embeddings for six regions — head, torso, upper arms, lower arms, upper legs, lower legs.

**Why:** Re-ID can match on visible parts only (e.g. head + torso when legs are occluded). BPBreID is trained with adversarial occlusion (GiLt), encouraging each part embedding to be discriminative alone. In matching costumes, dancers still differ in hair, skin tone, accessories, and proportions; part-level features capture that structure.

## Inputs & Dependencies

| Item | Detail |
|------|--------|
| **Crops** | Person image crops as BGR `np.ndarray`. |
| **Mask (optional)** | SAM2 binary mask for mask-isolated cropping (PLAN_04). |
| **Pose** | COCO-17 keypoints to define part regions. |
| **Prior steps** | Model code can land independently. **Gallery** (PLAN_07) stores `part_embeddings`; **tracker** (PLAN_04) supplies masks for clean crops. |

## Step-by-Step Implementation

1. Create `sway/bpbreid_extractor.py`.

2. **Dependencies:**
   - Prefer `torchreid` if it ships BPBreID and is pip-installable for your Python/torch stack.
   - Else: vendor BPBreID under `vendor/bpbreid/` and import from there (document install steps in repo README or `requirements-optional.txt` only if the user requests docs — this plan assumes code + comment in module).

3. **Checkpoint:** Download pretrained BPBreID (e.g. ResNet-50 on Market-1501 + MSMT17). Store at `models/bpbreid_r50_market_msmt17.pth` (path configurable via env).

4. **Class `BPBreIDExtractor`:**
   - `__init__(self, checkpoint_path: str, device: str)`: load weights, `eval()`, no grad for inference.

5. **`extract(self, crop: np.ndarray, keypoints: np.ndarray, mask: Optional[np.ndarray] = None) -> PartEmbeddings`:**
   - If `mask` is not `None`: zero pixels outside mask.
   - Resize crop to **256×128** (BPBreID standard input).
   - **Part regions from keypoints:**
     - Head: above nose
     - Torso: between shoulders and hips
     - Upper arms: shoulder → elbow
     - Lower arms: elbow → wrist
     - Upper legs: hip → knee
     - Lower legs: knee → ankle  
     (Implement with COCO index mapping; handle missing keypoints gracefully.)
   - Forward through BPBreID: **global** (e.g. 2048-d) + **6× part** (e.g. 256-d each — confirm against actual checkpoint head).
   - **Visibility:** for each part, `True` if ≥ 1 defining keypoint has confidence > `0.3`.
   - Return `PartEmbeddings(global_emb, part_embs: Dict[str, np.ndarray], visibility: Dict[str, bool])`.

6. **`compare(gallery_parts: PartEmbeddings, query_parts: PartEmbeddings) -> float`:**
   - Intersect parts visible in **both** gallery and query.
   - If fewer than `SWAY_REID_PART_MIN_VISIBLE` (default `3`) shared visible parts: **fallback** to global embedding cosine distance (or weighted global — pick one and document).
   - Else: cosine distance per shared part; return **mean** distance (lower = more similar). Optionally expose weighted variant later (PLAN_13).

7. **Env:** `SWAY_REID_PART_MODEL` — choices e.g. `bpbreid`, `paformer`, `osnet_x0_25`; default `bpbreid`.

8. **Dispatch:** At existing OSNet call sites, branch on `SWAY_REID_PART_MODEL`; when `bpbreid`, use `BPBreIDExtractor`.

9. **Normalization:** L2-normalize all stored and compared embeddings to unit length.

## Technical Considerations & Performance

- Forward pass ~few ms on GPU (similar order to OSNet); keypoint → region masks on CPU is negligible.
- **No keypoints yet (e.g. early enrollment):** Use fixed **vertical stripes** on the crop: e.g. top 1/6 head, next 2/6 torso, etc. — weaker but usable.
- **No SAM2 mask:** Full bbox crop (same as current OSNet path); BPBreID still runs, with more background leakage.

## Validation & Testing

- **bigtest (5 dancers):** Pairwise cosine distances — same dancer < ~0.3, different > ~0.5 (tune thresholds on validation).
- **Partial visibility:** Evaluate using only subsets of parts; confirm ranking vs OSNet.
- **Targets:** Same-dancer accuracy ≥ ~92% on bigtest (vs ~80% OSNet baseline); upper-body-only scenario ≥ ~85% (define protocol in eval script).

## Integration & Next Steps

| Consumer | Role |
|----------|------|
| PLAN_07 | `DancerGallery.part_embeddings` populated at enrollment. |
| PLAN_13 | Fusion uses part distance as one score. |
| PLAN_14 | EMA updates may refresh part embeddings under pose gates. |

**Outputs:** `PartEmbeddings` dataclass, `BPBreIDExtractor.extract`, `BPBreIDExtractor.compare` (or module-level `compare`), env-driven model selection.

## Swapability & Experimentation

**Factory integration:** All part-based Re-ID models implement a common `PartReIDExtractor` interface:

```python
class PartReIDExtractor(ABC):
    def extract(self, crop, keypoints, mask=None) -> PartEmbeddings: ...
    def compare(self, gallery, query) -> float: ...
```

The `sway/reid_factory.py` dispatches based on `SWAY_REID_PART_MODEL`:
- `bpbreid` → `BPBreIDExtractor` (this plan — lean core default)
- `paformer` → `PAFormerExtractor` (future alternative — not verified in literature, treat as experimental)
- `osnet_x0_25` → `OSNetExtractor` (current baseline — global embedding only, no parts)

**Manual A/B recipe:**

```bash
# Compare all Re-ID backbones on bigtest
python -m tools.pipeline_matrix_runs \
  --recipes reid_bpbreid,reid_osnet,reid_bpbreid_finetuned \
  --video data/ground_truth/bigtest/BigTest.mov --compare
```

**Sweep readiness:** `SWAY_REID_PART_MODEL` is a categorical choice in sweep phase S2. After PLAN_20 (fine-tuning), the fine-tuned weights become a third option: `bpbreid_finetuned`.

## Open Questions / Risks

- **Domain gap:** BPBreID trained on street pedestrians; dance stages, lighting, and costumes may hurt. **Mitigation:** PLAN_20 contrastive fine-tuning on in-domain clips.
- **Alternative:** **PAFormer** — pose tokens and explicit visibility weighting; may be more robust for dance. Keep `SWAY_REID_PART_MODEL=paformer` as a future branch. Note: PAFormer is not clearly linked to a canonical paper (per Perplexity fact-check); treat BPBreID as the validated default.

---

*Standalone plan — no dependency on external research docs for execution.*
