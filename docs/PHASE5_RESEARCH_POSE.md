# Phase 5 — Research pose tracks (AthletePose3D, Poseidon)

Production default stays **ViTPose+** via Hugging Face (`--pose-model huge` → `usyd-community/vitpose-plus-huge`, or `SWAY_VITPOSE_MODEL` for a custom checkpoint).

This doc covers **optional, higher-cost** improvements aligned with your roadmap.

## 1. Fine-tune ViTPose-class weights on AthletePose3D

**Why:** Athletic / extreme articulation can hurt off-the-shelf COCO models; domain fine-tune can reduce error on hard poses (see AthletePose3D paper).

**Where to start**

- Dataset and official release: [github.com/calvinyeungck/AthletePose3D](https://github.com/calvinyeungck/AthletePose3D)
- Paper / citation: CVPR 2025 workshop material linked from that repo and [OpenReview](https://openreview.net/forum?id=8RJhGMsJb9).

**Reality check**

- Training is **not** implemented inside `sway_pose_mvp`; use the authors’ code (or MMPose-style ViTPose training) on a cloud GPU.
- Checkpoints are usually **PyTorch / MMPose**, not necessarily drop-in for `transformers.VitPoseForPoseEstimation`. Serving options:
  - Convert or re-export to a Hugging Face–compatible repo and set `SWAY_VITPOSE_MODEL` to that id or local path, **or**
  - Add a thin adapter that loads your `.pth` and maps outputs to the same `(17, 3)` layout the pipeline expects (larger engineering task).

**After you have weights**

1. Validate on a few of **your** dance clips (same metrics you care about for scoring / review).
2. Point inference at the new model: `export SWAY_VITPOSE_MODEL=/path/or/hf/repo` before `main.py`.

Step-by-step training commands live with the external project; see `scripts/phase5_pose_research/README.md` for a checklist only.

## 2. Poseidon (true multi-frame ViTPose extension)

**What it is:** Poseidon is a **multi-frame RGB** model (cross-attention over a temporal window), not a post-hoc smoother on 2D keypoints alone. Reported results include strong PoseTrack video benchmarks (see paper).

**References**

- Paper: [Poseidon: A ViT-based Architecture for Multi-Frame Pose Estimation…](https://arxiv.org/html/2501.08446v1) (arXiv:2501.08446).
- Code / weights: [github.com/CesareDavidePace/poseidon](https://github.com/CesareDavidePace/poseidon).

**Integration shape (future)**

- Run their stack on frame windows and bounding boxes (top-down), or refactor to emit COCO-17 tensors compatible with `main.py`’s downstream stages.
- Expect a **separate** dependency tree and checkpoint format; treat as a new backend next to `PoseEstimator`, not a flag on the current HF model.

## 3. What this repo ships today: lightweight temporal KP refinement

To reduce **jitter** without Poseidon weights, the pipeline can apply **confidence-weighted smoothing of (x, y)** over ±N frames per track **after** ViTPose and stride interpolation:

```bash
python main.py video.mp4 --pose-model huge --temporal-pose-refine --temporal-pose-radius 2
# or
export SWAY_TEMPORAL_POSE_REFINE=1
export SWAY_TEMPORAL_POSE_RADIUS=2
```

Implementation: `sway/temporal_pose_refine.py`. This is explicitly **not** Poseidon; it is a practical stopgap until a Poseidon (or similar) backend is wired in.
