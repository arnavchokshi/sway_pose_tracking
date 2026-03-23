#!/usr/bin/env python3
"""
Compare trained YOLO11x vs COCO baseline on the merged val split (YAML).
Run after training to sanity-check the fine-tune.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "scripts" / "phase2_public_training" / "dancetrack_crowdhuman.yaml"

_CONVERTED_DIRS = (
    "datasets/dancetrack_yolo/images/train",
    "datasets/dancetrack_yolo/images/val",
    "datasets/crowdhuman_yolo/images/train",
    "datasets/crowdhuman_yolo/images/val",
)


def _converted_data_ready() -> bool:
    return all((REPO_ROOT / p).is_dir() for p in _CONVERTED_DIRS)


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate(weights_path: str, label: str) -> None:
    from ultralytics import YOLO

    device = _pick_device()
    model = YOLO(weights_path)
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=960,
        device=device,
        verbose=False,
    )
    print(f"\n{label} (device={device}):")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")


def main() -> None:
    os.chdir(REPO_ROOT)
    if not DATA_YAML.is_file():
        print(f"Missing {DATA_YAML}", file=sys.stderr)
        sys.exit(1)

    if not _converted_data_ready():
        print(
            "Converted YOLO datasets not found — skipping mAP comparison.\n"
            "After DanceTrack + CrowdHuman are in datasets/, run:\n"
            "  python scripts/phase2_public_training/convert_dancetrack_to_yolo.py\n"
            "  python scripts/phase2_public_training/convert_crowdhuman_to_yolo.py\n"
            "Then re-run this script."
        )
        best = REPO_ROOT / "models" / "yolo11x_dancetrack.pt"
        if not best.is_file():
            print(f"\nFine-tuned weights not found at {best}")
        return

    weights_coco = REPO_ROOT / "yolo11x.pt"
    if not weights_coco.is_file():
        print(
            f"Missing {weights_coco} (Ultralytics will download on first val).\n"
            "Or copy a cached yolo11x.pt into the repo root.",
            file=sys.stderr,
        )
        sys.exit(1)

    validate(str(weights_coco), "YOLO11x COCO baseline")

    best = REPO_ROOT / "models" / "yolo11x_dancetrack.pt"
    if best.is_file():
        validate(str(best), "YOLO11x DanceTrack fine-tuned")
    else:
        print(f"\nFine-tuned weights not found at {best}")
        print(
            "Run train_yolo11x.py first, then:\n"
            "  cp runs/detect/yolo11x_dancetrack/weights/best.pt models/yolo11x_dancetrack.pt"
        )


if __name__ == "__main__":
    main()
