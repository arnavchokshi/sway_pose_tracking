# Phase 2: YOLO11x Fine-Tune on DanceTrack + CrowdHuman

## What this does

Fine-tunes YOLO11x on dance-domain person detection data so it correctly detects dancers in extreme poses (splits, inversions, floor routines) and dense formations that COCO-pretrained models miss.

## What you need to do manually (cannot be automated)

### Step 1: DanceTrack

**Easiest:** from `sway_pose_mvp/`, after `pip install huggingface_hub`:

```bash
python scripts/phase2_public_training/fetch_dancetrack_hf.py
python scripts/phase2_public_training/convert_dancetrack_to_yolo.py
```

That pulls **`noahcao/dancetrack`** (official README link), merges into `datasets/dancetrack/`, then builds `datasets/dancetrack_yolo/`. Respect the [DanceTrack license](https://github.com/DanceTrack/DanceTrack/blob/main/README.md) (non-commercial research).

**Manual:** download zips from [Hugging Face](https://huggingface.co/datasets/noahcao/dancetrack) or **Baidu** (code `awew` per GitHub README). See **`REMAINING_STEPS.md`**.

Approx size: ~15 GB

### Step 2: Download CrowdHuman

1. Go to: https://www.crowdhuman.org/download.html
2. Register and download: `annotation_train.odgt`, `annotation_val.odgt`, and `Images/`
3. Place at: `datasets/crowdhuman/` under **`sway_pose_mvp/`**

Approx size: ~33 GB

### Step 3: GPU machine

Training requires:

- CUDA GPU with >= 16 GB VRAM (RTX 3090, A100, H100, etc.)
- Python >= 3.8, PyTorch with CUDA, `ultralytics`, Pillow installed (`pip install -r requirements.txt` from `sway_pose_mvp/`)
- ~50 GB free disk space total
- Estimated training time: 4–8 hours on RTX 3090, 2–4 hours on A100

Apple Silicon (MPS) will work but expect much slower training.

## Steps (after manual downloads)

All commands from **`sway_pose_mvp/`**:

```bash
cd sway_pose_mvp

# 0. Optional: verify dataset layout
python scripts/phase2_public_training/download_datasets.py

# 1. Convert datasets to YOLO format
python scripts/phase2_public_training/convert_dancetrack_to_yolo.py
python scripts/phase2_public_training/convert_crowdhuman_to_yolo.py

# 2. Run training
python scripts/phase2_public_training/train_yolo11x.py

# 3. Copy best checkpoint
cp runs/detect/yolo11x_dancetrack/weights/best.pt models/yolo11x_dancetrack.pt

# 4. Validate improvement
python scripts/phase2_public_training/validate_trained_model.py

# 5. Use in pipeline
export SWAY_YOLO_WEIGHTS=models/yolo11x_dancetrack.pt
python main.py --video your_video.mp4
```

The pipeline already reads `SWAY_YOLO_WEIGHTS` in `sway/tracker.py` (`resolve_yolo_model_path()`); no code changes are required beyond the env var.

## Expected improvement

- Better recall on extreme dance poses (splits, inversions, lifts)
- Better detection in dense formations
- Lower false positive rate on chairs/audience vs COCO baseline
- mAP50 on DanceTrack val: expect 0.80+ vs ~0.72 for COCO-only baseline (rough guide; depends on training setup)

## ViTPose-H fine-tune (optional)

Use OpenMMLab MMPose training configs on **COCO** keypoints (auto-downloaded by their scripts), or keep Hugging Face inference-only weights.

## AFLink (neural)

Full **AFLink** (`StrongSORT/AFLink`) expects CUDA and `newmodel_epoch20.pth`. This repo uses `SWAY_GLOBAL_LINK=1` for a lightweight appearance-free stitch in `sway/global_track_link.py` by default.
