# Sway Pose Tracking

M2-optimized pose pipeline for group dance videos: detection, tracking, pruning, ViTPose, scoring, and MP4/JSON export.

## Layout

| Path | Contents |
|------|----------|
| **`sway/`** | Pipeline Python package (tracker, pose, pruning, scoring, …) |
| **`config/`** | Tracker YAML (`ocsort.yaml`, `botsort.yaml`) |
| **`models/`** | YOLO weights / Core ML packages (see `models/README.md`) |
| **`input/`** | Your source videos for batch/offline runs |
| **`output/`** | Run outputs (gitignored) |
| **`review_app/`** | Static review site generator |
| **`benchmarks/`** | Optional ground-truth YAML for `benchmark.py` |
| **Entry scripts** | `main.py`, `batch_run_for_review.py`, `prefetch_models.py`, … (run from this directory) |

## Setup

```bash
cd sway_pose_mvp
pip install -r requirements.txt
```

## Run one video

```bash
python main.py input/your_clip.mp4 --output-dir output
```

## Batch + offline review

Put all source videos in **`input/`** (see `input/README.md`), then:

```bash
python batch_run_for_review.py --input-dir input --output-root output/review_batch --skip-existing
```

Open `output/review_batch/review/index.html` in a browser. When finished, use **Download JSONL** on that page to export labels.

## Offline (e.g. flight)

Inference is local once weights are cached. **While online**, prefetch once from `sway_pose_mvp/`:

```bash
python prefetch_models.py
```

That pulls YOLO (`yolo11m.pt` into `models/` or hub cache) and ViTPose base + large (Hugging Face cache, usually `~/.cache/huggingface`). If `models/yolo11l.mlpackage` or `models/yolo11m.mlpackage` exists, that Core ML bundle is used instead of `.pt`. Re-ID defaults to HSV only (no ResNet download).

**Before disconnecting**, enable strict offline mode (pick one style):

```bash
export SWAY_OFFLINE=1
```

Or standard Hugging Face / Transformers:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Optional, for YOLO path resolution only: `export YOLO_OFFLINE=1`.

Then run `main.py` or `batch_run_for_review.py` as usual. Missing cached models fail fast with a clear error instead of hanging.

**YOLO weights lookup order:** `SWAY_YOLO_WEIGHTS` → `models/yolo11l.mlpackage` / `models/yolo11m.mlpackage` (or same names in project root / cwd) → `models/yolo11m.pt`, then cwd / project root `yolo11m.pt`.

**ViTPose:** Override cache location with `HF_HOME` if models live on an external drive.

The review HTML under `output/.../review/` is static and needs no network when opened from disk.

## Hardware

Tuned for Apple Silicon (MPS). Falls back to CPU if MPS is unavailable.

## Benchmark (optional)

```bash
python benchmark.py --ground-truth benchmarks/IMG_0256_ground_truth.yaml
```
