#!/usr/bin/env python3
"""
Download CrowdHuman annotations + train/val zips from Hugging Face and lay out
datasets/crowdhuman/ for convert_crowdhuman_to_yolo.py.

Use this when official Google Drive links return "file does not exist" (common).

HF dataset (community mirror, same filenames as crowdhuman.org):
  https://huggingface.co/datasets/sshao0516/CrowdHuman

Requires: pip install huggingface_hub

You must still comply with CrowdHuman terms (non-commercial research, no redistribution).

Usage (from sway_pose_mvp/):
  python scripts/phase2_public_training/fetch_crowdhuman_hf.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Peak usage: one full train zip (~3 GB) + extracted tree (~3 GB) + copy to Images/train
# — need headroom beyond existing DanceTrack copies (~22 GB). Abort early if almost full.
_MIN_FREE_BYTES = 22 * 1024 * 1024 * 1024
HF_REPO = "sshao0516/CrowdHuman"
ODGTS = ("annotation_train.odgt", "annotation_val.odgt")
TRAIN_ZIPS = (
    "CrowdHuman_train01.zip",
    "CrowdHuman_train02.zip",
    "CrowdHuman_train03.zip",
)
VAL_ZIPS = ("CrowdHuman_val.zip",)


def _copy_jpgs_flat(src: Path, dest: Path) -> tuple[int, int]:
    """Copy every .jpg/.jpeg under src into dest (flat). Skip if dest name exists."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        target = dest / p.name
        if target.exists():
            skipped += 1
            continue
        shutil.copy2(p, target)
        copied += 1
    return copied, skipped


def _unzip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def _download_one(local_dir: Path, name: str) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} …", flush=True)
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=name,
                local_dir=str(local_dir),
            )
            p = local_dir / name
            if not p.is_file():
                raise FileNotFoundError(f"Missing after download: {p}")
            return p
        except Exception as e:
            last_err = e
            print(f"  HF download attempt {attempt}/3 failed: {e}", flush=True)
            if attempt < 3:
                wait = min(2**attempt, 60)
                print(f"  Retrying in {wait}s …", flush=True)
                time.sleep(wait)
    assert last_err is not None
    raise last_err


def main() -> int:
    parser = argparse.ArgumentParser(description="CrowdHuman → datasets/crowdhuman/ via Hugging Face")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "crowdhuman_hf",
        help="Where to store downloaded files during run",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep .zip files after extracting (default: delete to save ~11 GB)",
    )
    parser.add_argument(
        "--ignore-disk-check",
        action="store_true",
        help="Run even if free disk space is below the recommended minimum (may fail mid-download)",
    )
    args = parser.parse_args()
    prune_zips = not args.keep_zips

    free = shutil.disk_usage(REPO_ROOT).free
    if not args.ignore_disk_check and free < _MIN_FREE_BYTES:
        gib = free / (1024**3)
        need = _MIN_FREE_BYTES / (1024**3)
        print(
            f"Not enough free disk space on the volume containing sway_pose_mvp/ "
            f"({gib:.1f} GiB free; recommend ≥ {need:.0f} GiB before CrowdHuman fetch).\n"
            f"Free space (move datasets to an external drive, empty Trash, etc.) then re-run.\n"
            f"Or pass --ignore-disk-check to try anyway.",
            file=sys.stderr,
        )
        return 1

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        return 1

    cache_dir = args.cache_dir.resolve()
    out_base = REPO_ROOT / "datasets" / "crowdhuman"
    out_train = out_base / "Images" / "train"
    out_val = out_base / "Images" / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    for name in ODGTS:
        p = _download_one(cache_dir, name)
        dest = out_base / name
        shutil.copy2(p, dest)
        print(f"  → {dest}", flush=True)

    extract_root = REPO_ROOT / ".crowdhuman_extract_tmp"
    try:
        for zname in TRAIN_ZIPS:
            zpath = cache_dir / zname
            if not zpath.is_file():
                zpath = _download_one(cache_dir, zname)
            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True)
            print(f"Unzipping {zname} …", flush=True)
            _unzip(zpath, extract_root)
            c, s = _copy_jpgs_flat(extract_root, out_train)
            print(f"  train images +{c} (skipped existing names: {s})", flush=True)
            shutil.rmtree(extract_root)
            if prune_zips and zpath.is_file():
                zpath.unlink()

        for zname in VAL_ZIPS:
            zpath = cache_dir / zname
            if not zpath.is_file():
                zpath = _download_one(cache_dir, zname)
            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True)
            print(f"Unzipping {zname} …", flush=True)
            _unzip(zpath, extract_root)
            c, s = _copy_jpgs_flat(extract_root, out_val)
            print(f"  val images +{c} (skipped existing names: {s})", flush=True)
            shutil.rmtree(extract_root)
            if prune_zips and zpath.is_file():
                zpath.unlink()
    finally:
        if extract_root.exists():
            shutil.rmtree(extract_root, ignore_errors=True)

    print("\nDone.", flush=True)
    print(f"  {out_base}", flush=True)
    print("Next: python scripts/phase2_public_training/convert_crowdhuman_to_yolo.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
