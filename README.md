# Sway Pose Tracking V3.0

Production-ready, M2-optimized pose estimation pipeline for dance group videos. Handles severe occlusions, complex choreography, and long videos without crashing.

## Features (V3.0)

- **Streaming ingestion:** 300-frame chunks at native FPS (30+), no RAM bloat
- **Detection & Tracking:** YOLO11l (conf=0.25), BoT-SORT (track_buffer=90), Wave 1 box stitch
- **Resolution-aware pruning:** 20% duration, 5% bbox-height kinetic
- **High-fidelity pose:** ViTPose-Large, fp16 MPS, batched inference
- **Smart mirror pruning:** Edge + inverted velocity + low lower-body confidence
- **Spatio-temporal scoring:** circmean, cDTW (Sakoe-Chiba), per-joint thresholds
- **Per-joint heatmap:** Spine 10°/20°, Elbows/Knees 20°/35°

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for full documentation.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py path/to/your_video.mp4 --output-dir output
```

## Hardware

Optimized for Apple Silicon (M2) using PyTorch MPS backend. Falls back to CPU if MPS unavailable.
