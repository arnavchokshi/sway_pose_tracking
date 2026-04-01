"""
Download DETR-family model weights (PLAN_02)

Usage:
    python -m tools.download_detr_weights [--model co_detr_swinl|co_dino_swinl|rt_detr_l|rt_detr_x]

RT-DETR weights are auto-handled by ultralytics; this script is primarily for
Co-DETR / Co-DINO checkpoints from the official releases.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

CHECKPOINT_URLS = {
    # Sense-X GitHub release links for these checkpoints now 404.
    # OpenMMLab mirrors Co-DINO checkpoints and keeps stable URLs.
    # Co-DETR Swin-L public direct URLs are less stable; by default we use
    # the strong Swin-L Co-DINO checkpoint for both entries to keep
    # SWAY_DETECTOR_PRIMARY=co_detr/co_dino loadable on fresh boxes.
    "co_detr_swinl": (
        "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_1x_coco-27c13da4.pth",
        "co_detr_swinl_coco.pth",
    ),
    "co_dino_swinl": (
        "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_1x_coco-27c13da4.pth",
        "co_dino_swinl_coco.pth",
    ),
    "rt_detr_l": (
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt",
        "rtdetr-l.pt",
    ),
    "rt_detr_x": (
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-x.pt",
        "rtdetr-x.pt",
    ),
}


def download(model: str) -> None:
    if model not in CHECKPOINT_URLS:
        print(f"Unknown model: {model}. Choose from: {list(CHECKPOINT_URLS.keys())}")
        sys.exit(1)

    url, filename = CHECKPOINT_URLS[model]
    dest = MODELS_DIR / filename

    if dest.exists():
        print(f"Already exists: {dest}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model} from {url} ...")

    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))
        print(f"Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"Download failed: {exc}")
        print(f"Please manually download from {url} and place at {dest}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DETR weights")
    parser.add_argument(
        "--model",
        choices=list(CHECKPOINT_URLS.keys()),
        default=None,
        help="Specific model to download. If omitted, downloads all.",
    )
    args = parser.parse_args()

    if args.model:
        download(args.model)
    else:
        for model_name in CHECKPOINT_URLS:
            download(model_name)


if __name__ == "__main__":
    main()
