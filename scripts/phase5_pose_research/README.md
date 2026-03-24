# Phase 5 pose research (external training)

Use this folder as a **pointer** only. Fine-tuning ViTPose on AthletePose3D and training Poseidon-style models happen in **separate repositories** with their own dependencies.

## AthletePose3D fine-tune — checklist

1. Clone [AthletePose3D](https://github.com/calvinyeungck/AthletePose3D) and follow their data download / license steps.
2. Train on a cloud GPU (e.g. alongside your YOLO26l job) using **their** training scripts or your MMPose recipe.
3. Export or convert checkpoints to whatever your inference path needs:
   - HF-compatible ViTPose+ → set `SWAY_VITPOSE_MODEL`.
   - Otherwise → implement a loader that still produces `(17, 3)` keypoints + `(17,)` scores per box.
4. Evaluate on held-out **dance** clips before switching production defaults.

## Poseidon — checklist

1. Clone [poseidon](https://github.com/CesareDavidePace/poseidon) and install per their README.
2. Reproduce inference on PoseTrack or your clips to confirm throughput and VRAM.
3. Design a bridge: windowed video + boxes → COCO-17 arrays in the same schema as `PoseEstimator.estimate_poses()`.

## In-repo stopgap

Temporal **keypoint** smoothing (not Poseidon): see `docs/PHASE5_RESEARCH_POSE.md` and `sway/temporal_pose_refine.py`.
