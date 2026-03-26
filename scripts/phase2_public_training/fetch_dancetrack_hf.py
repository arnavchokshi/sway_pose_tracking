#!/usr/bin/env python3
"""
Download DanceTrack zips from Hugging Face (noahcao/dancetrack — official README link),
unzip, and merge into sway_pose_mvp/datasets/dancetrack/{train,val}/.

Requires: pip install huggingface_hub

Usage (from sway_pose_mvp/):
  python scripts/phase2_public_training/fetch_dancetrack_hf.py

Re-run is safe: skips sequences that already exist under datasets/dancetrack/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "noahcao/dancetrack"
ZIPS_TRAIN = ("train1.zip", "train2.zip")
ZIPS_VAL = ("val.zip",)


def _find_sequence_dirs(extracted: Path) -> list[Path]:
    """Return directories that look like MOT sequences (img1/ + gt/gt.txt)."""
    found: list[Path] = []
    seen: set[Path] = set()

    def consider(p: Path) -> None:
        if not p.is_dir():
            return
        if (p / "img1").is_dir() and (p / "gt" / "gt.txt").is_file():
            if p.resolve() not in seen:
                seen.add(p.resolve())
                found.append(p)

    # Common layouts: <root>/<seq>/, <root>/train/<seq>/, <root>/val/<seq>/
    for sub in (extracted, extracted / "train", extracted / "val"):
        if not sub.is_dir():
            continue
        for child in sub.iterdir():
            consider(child)
    # Deeper nesting (rare)
    if not found:
        for p in extracted.rglob("gt/gt.txt"):
            seq = p.parent.parent
            consider(seq)
    return sorted(found, key=lambda x: x.name)


def _copy_sequence(src: Path, dest_parent: Path, overwrite: bool) -> bool:
    dest = dest_parent / src.name
    if dest.exists():
        if not overwrite:
            return False
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return True


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
                raise FileNotFoundError(f"Expected file after download: {p}")
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
    parser = argparse.ArgumentParser(description="Download DanceTrack from HF and merge to datasets/dancetrack/")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing sequence folders under train/val",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "dancetrack_hf",
        help="Where to store downloaded .zip files (default: sway_pose_mvp/dancetrack_hf)",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded .zip files after merge (default: delete each zip after use to save disk)",
    )
    args = parser.parse_args()
    prune_zips = not args.keep_zips

    cache_dir = args.cache_dir.resolve()
    out_train = REPO_ROOT / "datasets" / "dancetrack" / "train"
    out_val = REPO_ROOT / "datasets" / "dancetrack" / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        return 1

    copied_train = 0
    skipped_train = 0
    copied_val = 0
    skipped_val = 0

    extract = REPO_ROOT / ".dancetrack_extract_tmp"
    try:
        for zname in ZIPS_TRAIN:
            zpath = cache_dir / zname
            if not zpath.is_file():
                zpath = _download_one(cache_dir, zname)
            if extract.exists():
                shutil.rmtree(extract)
            extract.mkdir(parents=True)
            print(f"Unzipping {zname} …", flush=True)
            _unzip(zpath, extract)
            for seq in _find_sequence_dirs(extract):
                if _copy_sequence(seq, out_train, args.overwrite):
                    copied_train += 1
                    print(f"  train ← {seq.name}", flush=True)
                else:
                    skipped_train += 1
                    print(f"  skip (exists) train/{seq.name}", flush=True)
            shutil.rmtree(extract)
            if prune_zips and zpath.is_file():
                zpath.unlink()
                print(f"  removed {zname}", flush=True)

        for zname in ZIPS_VAL:
            zpath = cache_dir / zname
            if not zpath.is_file():
                zpath = _download_one(cache_dir, zname)
            if extract.exists():
                shutil.rmtree(extract)
            extract.mkdir(parents=True)
            print(f"Unzipping {zname} …", flush=True)
            _unzip(zpath, extract)
            for seq in _find_sequence_dirs(extract):
                if _copy_sequence(seq, out_val, args.overwrite):
                    copied_val += 1
                    print(f"  val ← {seq.name}", flush=True)
                else:
                    skipped_val += 1
                    print(f"  skip (exists) val/{seq.name}", flush=True)
            shutil.rmtree(extract)
            if prune_zips and zpath.is_file():
                zpath.unlink()
                print(f"  removed {zname}", flush=True)
    finally:
        if extract.exists():
            shutil.rmtree(extract, ignore_errors=True)

    print("\nDone.", flush=True)
    print(f"  train: {out_train}  (copied {copied_train}, skipped {skipped_train})", flush=True)
    print(f"  val:   {out_val}    (copied {copied_val}, skipped {skipped_val})", flush=True)
    print("Next: python scripts/phase2_public_training/download_datasets.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
