#!/usr/bin/env python3
"""
Re-encode existing *_poses.mp4 files to H.264 for browser playback.

OpenCV writes MPEG-4 Part 2 video; Safari/Chrome often show black video.
Run this on an output folder (no pipeline re-run):

  python review_app/reencode_poses_for_web.py output/flight_batch

Requires ffmpeg (brew install ffmpeg).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def reencode_one(path: Path, ffmpeg: str) -> bool:
    tmp = Path(tempfile.mktemp(suffix=".mp4", dir=str(path.parent)))
    try:
        r = subprocess.run(
            [
                ffmpeg, "-y", "-i", str(path),
                "-map", "0:v:0",
                "-map", "0:a:0?",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "22",
                "-preset", "fast",
                "-movflags", "+faststart",
                "-c:a", "copy",
                str(tmp),
            ],
            capture_output=True,
            timeout=None,
        )
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(path)
            return True
        print(f"  FAIL {path.name}: {(r.stderr or b'').decode()[:200]}")
    except Exception as e:
        print(f"  FAIL {path.name}: {e}")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return False


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: reencode_poses_for_web.py <output_root>")
        sys.exit(1)
    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        sys.exit(1)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found (brew install ffmpeg)")
        sys.exit(1)
    files = sorted(root.rglob("*_poses.mp4"))
    if not files:
        print(f"No *_poses.mp4 under {root}")
        sys.exit(0)
    ok = 0
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p.relative_to(root)}")
        if reencode_one(p, ffmpeg):
            ok += 1
    print(f"Done. {ok}/{len(files)} OK.")


if __name__ == "__main__":
    main()
