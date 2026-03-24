# Sway Pose Tracking

M2-optimized pose pipeline for group dance videos: detection, tracking, pruning, ViTPose, scoring, and MP4/JSON export.

## Layout

| Path | Contents |
|------|----------|
| **`sway/`** | Pipeline Python package (tracker, pose, pruning, scoring, …) |
| **`config/`** | Tracker YAML (`ocsort.yaml`, `botsort.yaml`) |
| **`models/`** | YOLO, SAM2.1, OSNet, Core ML bundles (see `models/README.md`) |
| **`input/`** | Short test clips for `main.py` / small runs (gitignored blobs) |
| **`data/videos_inbox/`** | Large batch queue for `python -m tools.batch_run_for_review` (see `data/videos_inbox/README.md`) |
| **`data/`** | Other local-only data notes (`data/README.md`) |
| **`output/`** | Run outputs (gitignored) |
| **`docs/`** | Design and pipeline writeups (`docs/README.md`) |
| **`scripts/`** | Training, smoke tests, render helpers (`scripts/README.md`) |
| **`review_app/`** | Static review site generator |
| **`benchmarks/`** | Optional ground-truth YAML for `python -m tools.benchmark` |
| **Entry scripts** | `main.py` at repo root; utilities under `tools/` (e.g. `python -m tools.prefetch_models`) |

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
python -m tools.batch_run_for_review --input-dir input --output-root output/review_batch --skip-existing
```

Open `output/review_batch/review/index.html` in a browser. When finished, use **Download JSONL** on that page to export labels.

## Offline (e.g. flight)

Inference is local once weights are cached. **While online**, prefetch once from `sway_pose_mvp/`:

```bash
python -m tools.prefetch_models
```

That pulls YOLO26 (`yolo26l.pt` into `models/` or hub cache) and ViTPose base + large (Hugging Face cache, usually `~/.cache/huggingface`). On-disk YOLO26 `.pt` in `models/` / repo root is preferred. Re-ID defaults to HSV only (no ResNet download).

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

Then run `main.py` or `python -m tools.batch_run_for_review` as usual. Missing cached models fail fast with a clear error instead of hanging.

**YOLO weights lookup order:** `SWAY_YOLO_WEIGHTS` (path or Pipeline Lab token, e.g. `yolo26l`) → on-disk YOLO26 `.pt` candidates in `models/` and repo root → hub fallback `yolo26l.pt`. See `sway/tracker.py` `resolve_yolo_model_path()`.

**ViTPose:** Override cache location with `HF_HOME` if models live on an external drive.

The review HTML under `output/.../review/` is static and needs no network when opened from disk.

## Hardware

Tuned for Apple Silicon (MPS). Falls back to CPU if MPS is unavailable.

## Benchmark (optional)

```bash
python -m tools.benchmark --ground-truth benchmarks/IMG_0256_ground_truth.yaml
```
