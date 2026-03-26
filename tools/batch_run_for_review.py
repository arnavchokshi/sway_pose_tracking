#!/usr/bin/env python3
"""
Batch-run the pose pipeline on many videos into a single output root, then build
the offline static review site (review/index.html).

Run from repo root or sway_pose_mvp/; pipeline cwd is always sway_pose_mvp.

Example:
  cd sway_pose_mvp
  python -m tools.batch_run_for_review --input-dir input --output-root output/review_batch \\
      --pose-model base --skip-existing

Large batch queue (default inbox layout):
  python -m tools.batch_run_for_review --input-dir data/videos_inbox --output-root output/flight_batch \\
      --pose-model base --skip-existing

Afterwards open the review UI — for reliable video scrubbing offline, prefer:
  python review_app/serve_review.py output/flight_batch
  → http://localhost:8899/review/index.html
(or open output/.../review/index.html directly if your browser allows local video).

With --skip-existing, you can re-run the same command after sleep/crash/interrupt:
finished clips stay skipped; anything without a final *_poses.mp4 is processed again.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


VIDEO_GLOBS = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.m4v", "*.M4V", "*.webm", "*.WEBM")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def discover_videos(input_dir: Path) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for pat in VIDEO_GLOBS:
        for p in sorted(input_dir.glob(pat)):
            if p.is_file() and p.resolve() not in seen:
                seen.add(p.resolve())
                out.append(p)
    return out


def run_one(
    video: Path,
    out_dir: Path,
    extra_args: List[str],
    cwd: Path,
) -> bool:
    cmd = [
        sys.executable,
        str(cwd / "main.py"),
        str(video.resolve()),
        "--output-dir",
        str(out_dir.resolve()),
    ] + extra_args
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd), capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e}")
        return False


def has_rendered_output(out_dir: Path) -> bool:
    for p in out_dir.glob("*_poses.mp4"):
        if p.is_file() and p.stat().st_size > 0:
            return True
    return False


def write_manifest(
    output_root: Path,
    videos: List[Path],
    results: Dict[str, Any],
    extra_meta: Dict[str, Any],
) -> None:
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root.resolve()),
        **extra_meta,
        "samples": results,
    }
    path = output_root / "batch_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {path}")


def generate_review_site(output_root: Path) -> None:
    gen = _repo_root() / "review_app" / "generate_review_index.py"
    if not gen.exists():
        print("Warning: generate_review_index.py missing; skip review site.")
        return
    subprocess.run(
        [sys.executable, str(gen), str(output_root.resolve())],
        check=False,
        cwd=str(_repo_root()),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch pipeline + offline review bundle for all videos in a folder."
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input videos",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/review_batch"),
        help="All run outputs go under here: <stem>/ per video (default: output/review_batch)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a video if <output>/<stem>/*_poses.mp4 already exists (non-empty). "
            "Safe to re-run after interrupt: unfinished runs have no final *_poses.mp4."
        ),
    )
    ap.add_argument(
        "--pose-model",
        choices=["base", "large", "huge"],
        default="base",
        help="Passed to main.py --pose-model (default base; use large/huge for accuracy).",
    )
    ap.add_argument(
        "--pose-stride",
        type=int,
        choices=[1, 2],
        default=1,
    )
    ap.add_argument(
        "--params",
        type=Path,
        default=None,
        help="YAML params file passed to main.py",
    )
    ap.add_argument(
        "--montage",
        action="store_true",
        help="Pass --montage to main.py (slower; montage per video)",
    )
    ap.add_argument(
        "--no-review-site",
        action="store_true",
        help="Do not regenerate review/index.html at the end",
    )
    args = ap.parse_args()

    if str(os.environ.get("SWAY_OFFLINE", "")).lower() in ("1", "true", "yes"):
        print("SWAY_OFFLINE=1 — using local model files only (run python -m tools.prefetch_models while online if needed).")

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}")
        sys.exit(1)

    videos = discover_videos(input_dir)
    if not videos:
        print(f"No videos found in {input_dir} (tried: {', '.join(VIDEO_GLOBS)})")
        sys.exit(1)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cwd = _repo_root()
    extra: List[str] = ["--pose-model", args.pose_model, "--pose-stride", str(args.pose_stride)]
    if args.params:
        extra.extend(["--params", str(args.params.resolve())])
    if args.montage:
        extra.append("--montage")

    batch_meta = {
        "input_dir": str(input_dir),
        "cli": {
            "pose_model": args.pose_model,
            "pose_stride": args.pose_stride,
            "params": str(args.params) if args.params else None,
            "montage": args.montage,
        },
    }

    results: Dict[str, Any] = {}
    t0 = time.time()
    for i, video in enumerate(videos, 1):
        stem = video.stem
        out_dir = output_root / stem
        print(f"\n[{i}/{len(videos)}] {video.name} -> {out_dir.name}/")
        if args.skip_existing and has_rendered_output(out_dir):
            print("  skip-existing: already has *_poses.mp4")
            results[stem] = {
                "input_path": str(video.resolve()),
                "output_dir": str(out_dir.resolve()),
                "status": "skipped_existing",
            }
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = run_one(video, out_dir, extra, cwd)
        results[stem] = {
            "input_path": str(video.resolve()),
            "output_dir": str(out_dir.resolve()),
            "status": "ok" if ok else "failed",
        }

    write_manifest(output_root, videos, results, batch_meta)
    elapsed = time.time() - t0
    print(f"\nBatch finished in {elapsed/60:.1f} min")

    if not args.no_review_site:
        generate_review_site(output_root)

    print(f"\nOpen in browser: {output_root / 'review' / 'index.html'}")
    print("Export your labels from the review page when done (JSONL download).")


if __name__ == "__main__":
    main()
