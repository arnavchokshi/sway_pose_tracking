#!/usr/bin/env python3
"""
Convert DanceTrack MOTChallenge annotations to YOLO detection format.

Input:  datasets/dancetrack/train/ and val/ (MOTChallenge gt.txt per sequence)
Output: datasets/dancetrack_yolo/
          images/train/
          images/val/
          labels/train/
          labels/val/

Run from sway_pose_mvp/:
  python scripts/phase2_public_training/convert_dancetrack_to_yolo.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def convert_mot_to_yolo(dancetrack_root: Path, output_root: Path) -> None:
    for split in ("train", "val"):
        split_dir = dancetrack_root / split
        if not split_dir.exists():
            print(f"  Skipping {split} — directory not found: {split_dir}")
            continue

        img_out = output_root / "images" / split
        lbl_out = output_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_file = seq_dir / "gt" / "gt.txt"
            img_dir = seq_dir / "img1"
            if not gt_file.exists() or not img_dir.exists():
                continue

            img_w, img_h = 1920, 1080
            seqinfo = seq_dir / "seqinfo.ini"
            if seqinfo.exists():
                for line in seqinfo.read_text().splitlines():
                    if line.startswith("imWidth="):
                        img_w = int(line.split("=", 1)[1].strip())
                    elif line.startswith("imHeight="):
                        img_h = int(line.split("=", 1)[1].strip())

            frame_boxes: dict[int, list[str]] = {}
            for line in gt_file.read_text().splitlines():
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                cls = int(parts[7]) if len(parts) > 7 else 1
                vis = float(parts[8]) if len(parts) > 8 else 1.0
                if cls != 1 or vis < 0.1:
                    continue
                x, y, w, h = (
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                frame_boxes.setdefault(frame, []).append(
                    f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                )

            for img_file in sorted(img_dir.glob("*.jpg")):
                frame_num = int(img_file.stem)
                out_img = img_out / f"{seq_dir.name}_{img_file.name}"
                if not out_img.exists():
                    shutil.copy2(img_file, out_img)
                out_lbl = lbl_out / f"{seq_dir.name}_{img_file.stem}.txt"
                lines = frame_boxes.get(frame_num, [])
                out_lbl.write_text("\n".join(lines))

        n_img = len(list(img_out.glob("*.jpg")))
        print(f"  Converted {split}: {n_img} images → {img_out}")


def main() -> int:
    os.chdir(REPO_ROOT)
    dancetrack_root = REPO_ROOT / "datasets" / "dancetrack"
    output_root = REPO_ROOT / "datasets" / "dancetrack_yolo"
    if not dancetrack_root.is_dir():
        print(f"Missing {dancetrack_root}", file=sys.stderr)
        print("Download DanceTrack and run:", file=sys.stderr)
        print("  python scripts/phase2_public_training/download_datasets.py", file=sys.stderr)
        return 1
    convert_mot_to_yolo(dancetrack_root, output_root)
    print(f"Done. Output: {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
