#!/usr/bin/env python3
"""
Convert CrowdHuman ODGT annotations to YOLO detection format.
Uses person class only (fbox = full-body bounding box).

Input:  datasets/crowdhuman/annotation_{train,val}.odgt
        datasets/crowdhuman/Images/{train,val}/
Output: datasets/crowdhuman_yolo/
          images/train/
          images/val/
          labels/train/
          labels/val/

Requires: Pillow (pip install pillow)

Run from sway_pose_mvp/:
  python scripts/phase2_public_training/convert_crowdhuman_to_yolo.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def convert_crowdhuman_split(
    odgt_file: Path,
    img_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
) -> None:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    lines = odgt_file.read_text().splitlines()
    converted = 0
    for line in lines:
        record = json.loads(line.strip())
        img_id = record["ID"]
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines: list[str] = []
        for ann in record.get("gtboxes", []):
            if ann.get("tag") != "person":
                continue
            fbox = ann.get("fbox")
            if not fbox or len(fbox) < 4:
                continue
            x, y, w, h = fbox
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        out_img = out_img_dir / f"{img_id}.jpg"
        if not out_img.exists():
            shutil.copy2(img_path, out_img)
        (out_lbl_dir / f"{img_id}.txt").write_text("\n".join(yolo_lines))
        converted += 1

    print(f"  Converted {converted} images from {odgt_file.name}")


def main() -> int:
    os.chdir(REPO_ROOT)
    base = REPO_ROOT / "datasets" / "crowdhuman"
    if not base.is_dir():
        print(f"Missing {base}", file=sys.stderr)
        print("Download CrowdHuman and run:", file=sys.stderr)
        print("  python scripts/phase2_public_training/download_datasets.py", file=sys.stderr)
        return 1

    for split in ("train", "val"):
        odgt = base / f"annotation_{split}.odgt"
        imgs = base / "Images" / split
        if not odgt.is_file():
            print(f"Missing {odgt}", file=sys.stderr)
            return 1
        if not imgs.is_dir():
            print(f"Missing {imgs}", file=sys.stderr)
            return 1
        convert_crowdhuman_split(
            odgt,
            imgs,
            REPO_ROOT / "datasets" / "crowdhuman_yolo" / "images" / split,
            REPO_ROOT / "datasets" / "crowdhuman_yolo" / "labels" / split,
        )

    print(f"Done. Output: {REPO_ROOT / 'datasets' / 'crowdhuman_yolo'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
