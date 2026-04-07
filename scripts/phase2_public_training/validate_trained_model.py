#!/usr/bin/env python3
"""
Run Ultralytics YOLO val() to report mAP / precision / recall.

- Picks a **data YAML** from what is on disk: merged (DanceTrack+CrowdHuman) if both
  converted datasets exist, else DanceTrack-only or CrowdHuman-only.
- Compares **COCO-pretrained yolo26l.pt** vs your fine-tuned weights on the **same val split**
  when the baseline weights are present (use --no-baseline to skip).

Run from sway_pose_mvp/:

  python scripts/phase2_public_training/validate_trained_model.py
  python scripts/phase2_public_training/validate_trained_model.py \\
    --weights runs/detect/yolo26l_dancetrack_only/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE2 = REPO_ROOT / "scripts" / "phase2_public_training"

DT_TRAIN = REPO_ROOT / "datasets" / "dancetrack_yolo" / "images" / "train"
DT_VAL = REPO_ROOT / "datasets" / "dancetrack_yolo" / "images" / "val"
CH_TRAIN = REPO_ROOT / "datasets" / "crowdhuman_yolo" / "images" / "train"
CH_VAL = REPO_ROOT / "datasets" / "crowdhuman_yolo" / "images" / "val"


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_data_yaml() -> tuple[Path, str]:
    """Return (yaml_path, description)."""
    merged = PHASE2 / "dancetrack_crowdhuman.yaml"
    dt_only = PHASE2 / "dancetrack_only.yaml"
    ch_only = PHASE2 / "crowdhuman_only.yaml"

    merged_ok = (
        DT_TRAIN.is_dir()
        and DT_VAL.is_dir()
        and CH_TRAIN.is_dir()
        and CH_VAL.is_dir()
    )
    if merged_ok:
        return merged, "merged DanceTrack + CrowdHuman val"

    dt_ok = DT_TRAIN.is_dir() and DT_VAL.is_dir()
    if dt_ok:
        return dt_only, "DanceTrack val only"

    ch_ok = CH_TRAIN.is_dir() and CH_VAL.is_dir()
    if ch_ok:
        return ch_only, "CrowdHuman val only"

    raise SystemExit(
        "No converted YOLO val data found.\n"
        "Expected at least datasets/dancetrack_yolo/images/{train,val} and/or\n"
        "datasets/crowdhuman_yolo/images/{train,val}.\n"
        "Run the convert_*_to_yolo.py scripts after downloading raw datasets."
    )


def default_finetuned_weights() -> Path | None:
    candidates = (
        REPO_ROOT / "models" / "yolo26l_dancetrack.pt",
        REPO_ROOT / "runs" / "detect" / "yolo26l_dancetrack_only" / "weights" / "best.pt",
        REPO_ROOT / "runs" / "detect" / "yolo26l_dancetrack" / "weights" / "best.pt",
        REPO_ROOT / "runs" / "detect" / "yolo26l_crowdhuman_ft" / "weights" / "best.pt",
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def validate(weights_path: str, label: str, data_yaml: Path, imgsz: int) -> None:
    from ultralytics import YOLO

    device = _pick_device()
    model = YOLO(weights_path)
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    print(f"\n{label} (device={device}):")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO val metrics: baseline vs fine-tuned.")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Fine-tuned checkpoint (.pt). Default: first found among models/ and runs/detect/*/weights/best.pt",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override data YAML (default: auto from datasets on disk)",
    )
    p.add_argument("--imgsz", type=int, default=960, help="Val image size (match training)")
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not run COCO yolo26l.pt baseline (faster if you only care about fine-tuned mAP)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    if args.data is not None:
        if not args.data.is_file():
            print(f"Missing --data file: {args.data}", file=sys.stderr)
            sys.exit(1)
        data_yaml = args.data.resolve()
        data_desc = str(data_yaml.relative_to(REPO_ROOT))
    else:
        data_yaml, data_desc = resolve_data_yaml()

    print(f"Val split: {data_desc}")
    print(f"Data YAML: {data_yaml.relative_to(REPO_ROOT)}")

    ft = args.weights
    if ft is None:
        found = default_finetuned_weights()
        if found is None:
            print(
                "\nNo fine-tuned weights found. Pass --weights PATH, e.g.\n"
                "  --weights runs/detect/yolo26l_dancetrack_only/weights/best.pt\n"
                "or copy best.pt to models/yolo26l_dancetrack.pt",
                file=sys.stderr,
            )
            sys.exit(1)
        ft = found
    else:
        ft = ft.resolve()
        if not ft.is_file():
            print(f"Missing --weights: {ft}", file=sys.stderr)
            sys.exit(1)

    if not args.no_baseline:
        # Local yolo26l.pt or Ultralytics download on first use (cwd is REPO_ROOT).
        validate("yolo26l.pt", "YOLO26l COCO baseline", data_yaml, args.imgsz)

    validate(str(ft), f"Fine-tuned ({ft.name})", data_yaml, args.imgsz)
    print(f"\nWeights evaluated: {ft}")


if __name__ == "__main__":
    main()
