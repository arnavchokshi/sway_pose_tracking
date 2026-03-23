#!/usr/bin/env python3
"""
Verify DanceTrack and CrowdHuman are present under sway_pose_mvp/datasets/.

Both datasets require manual download (terms / registration); this script cannot fetch them.

Run from anywhere:
  python scripts/phase2_public_training/download_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def print_manual_instructions() -> None:
    print(
        """
================================================================================
MANUAL DOWNLOADS (cannot be automated)
================================================================================

DanceTrack
--------
  Official README (dataset + license): https://github.com/DanceTrack/DanceTrack
  Automated (HF zips + merge):  python scripts/phase2_public_training/fetch_dancetrack_hf.py
  Manual HF: https://huggingface.co/datasets/noahcao/dancetrack  (Google Drive deprecated per authors)
  Alternative: Baidu Netdisk — link and code (awew) are in the GitHub README.
  License: CC-BY-4.0 annotations; non-commercial research only (see GitHub).

  Use the zip files on Hugging Face (Files and versions), not the web dataset preview.
  Expected layout under sway_pose_mvp/datasets/dancetrack/ (same as authors' dancetrack/ folder):
    datasets/dancetrack/
      train/
        dancetrack0001/
          img1/              ← frame images (.jpg)
          gt/gt.txt          ← MOTChallenge-format ground truth
          seqinfo.ini
        dancetrack0002/
        ...
      val/
        ...
      test/
        ...

  (~40 train sequences, ~25 val sequences; ~15 GB typical)

CrowdHuman
----------
  If Google Drive links fail ("file does not exist"), use:
    python scripts/phase2_public_training/fetch_crowdhuman_hf.py
  (Hugging Face mirror sshao0516/CrowdHuman — same filenames as official.)
  Or: https://www.crowdhuman.org/download.html (Baidu; register if required)

  Expected layout:
    datasets/crowdhuman/
      annotation_train.odgt
      annotation_val.odgt
      Images/
        train/
        val/

  (~33 GB typical)

All paths are relative to the sway_pose_mvp/ directory (repo root for this project):
"""
    )
    print(f"  {REPO_ROOT}/datasets/...\n")
    print("=" * 80 + "\n")


def verify_dancetrack(root: Path) -> list[str]:
    errors: list[str] = []
    base = root / "datasets" / "dancetrack"
    for split in ("train", "val"):
        split_dir = base / split
        if not split_dir.is_dir():
            errors.append(f"DanceTrack: missing directory {split_dir}")
            continue
        seqs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not seqs:
            errors.append(f"DanceTrack: no sequence folders under {split_dir}")
            continue
        ok = 0
        for seq in seqs:
            if (seq / "gt" / "gt.txt").is_file() and (seq / "img1").is_dir():
                ok += 1
        if ok == 0:
            errors.append(
                f"DanceTrack: no valid sequences in {split_dir} "
                "(each needs gt/gt.txt and img1/)"
            )
    return errors


def verify_crowdhuman(root: Path) -> list[str]:
    errors: list[str] = []
    base = root / "datasets" / "crowdhuman"
    for name in ("annotation_train.odgt", "annotation_val.odgt"):
        p = base / name
        if not p.is_file():
            errors.append(f"CrowdHuman: missing {p}")
    for split in ("train", "val"):
        img_dir = base / "Images" / split
        if not img_dir.is_dir():
            errors.append(f"CrowdHuman: missing image directory {img_dir}")
        elif not any(img_dir.iterdir()):
            errors.append(f"CrowdHuman: empty directory {img_dir}")
    return errors


def main() -> int:
    print_manual_instructions()
    dt_err = verify_dancetrack(REPO_ROOT)
    ch_err = verify_crowdhuman(REPO_ROOT)
    all_err = dt_err + ch_err

    if not all_err:
        print("✓ DanceTrack and CrowdHuman layouts look OK under:")
        print(f"    {REPO_ROOT / 'datasets'}")
        print("\nNext steps:")
        print("  python scripts/phase2_public_training/convert_dancetrack_to_yolo.py")
        print("  python scripts/phase2_public_training/convert_crowdhuman_to_yolo.py")
        return 0

    print("Dataset check failed:\n")
    for e in all_err:
        print(f"  • {e}")
    print("\nDownload URLs:")
    print("  DanceTrack:   https://huggingface.co/datasets/noahcao/dancetrack")
    print("                https://github.com/DanceTrack/DanceTrack (README → Dataset)")
    print("  CrowdHuman:   https://www.crowdhuman.org/download.html")
    print(f"\nInstall under: {REPO_ROOT / 'datasets'}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
