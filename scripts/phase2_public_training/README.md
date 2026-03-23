# Phase 2 — public-data-only training (no custom labels)

## YOLO11x fine-tune (DanceTrack + CrowdHuman)

1. Download [DanceTrack](https://github.com/DanceTrack/DanceTrack) or `DanceTrack/DanceTrack` from Hugging Face.
2. Register and download [CrowdHuman](https://www.crowdhuman.org/) — person detection subset.
3. Merge into a single Ultralytics `data.yaml` with `train`/`val` image and label paths.
4. Run from `sway_pose_mvp/` (with GPU):

```bash
yolo detect train model=yolo11x.pt data=path/to/merged.yaml epochs=100 imgsz=960 batch=8
```

5. Point the pipeline at the best weights:

```bash
export SWAY_YOLO_WEIGHTS=/path/to/runs/detect/train/weights/best.pt
```

## ViTPose-H fine-tune (optional)

Use OpenMMLab MMPose training configs on **COCO** keypoints (auto-downloaded by their scripts), or keep Hugging Face inference-only weights.

## AFLink (neural)

Full **AFLink** (`StrongSORT/AFLink`) expects CUDA and `newmodel_epoch20.pth`. This repo uses `SWAY_GLOBAL_LINK=1` for a lightweight appearance-free stitch in `sway/global_track_link.py` by default.
