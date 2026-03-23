#!/usr/bin/env python3
"""
Download and cache every weight the pipeline needs for offline (airplane) runs.

Run once while you have internet, from the sway_pose_mvp directory:

  cd sway_pose_mvp
  python prefetch_models.py

This touches:
  - Ultralytics YOLO (yolo11m.pt → models/ or hub cache)
  - Hugging Face ViTPose base + large (HF cache, usually ~/.cache/huggingface)
  - BoxMOT OSNet Re-ID weights → models/osnet_x0_25_msmt17.pt (default tracker path)

Fine-tuning YOLO11x on DanceTrack + CrowdHuman (optional): base weights `yolo11x.pt`
are pulled automatically by Ultralytics the first time you run
`scripts/phase2_public_training/train_yolo11x.py` — no separate prefetch step for
that file. See `scripts/phase2_public_training/README.md` for the full workflow.

After that, set SWAY_OFFLINE=1 when running without network (see README.md, Offline).
"""

from __future__ import annotations

import gc
import os
import sys
import urllib.request
from pathlib import Path

import _repo_path  # noqa: F401


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)
    if os.environ.get("SWAY_OFFLINE", "").lower() in ("1", "true", "yes"):
        print("Unset SWAY_OFFLINE for prefetch (needs network).", file=sys.stderr)
        sys.exit(1)

    print(f"Working directory: {root}\n")

    print("[1/4] YOLO yolo11m.pt …")
    from ultralytics import YOLO

    yolo_pt = root / "models" / "yolo11m.pt"
    YOLO(str(yolo_pt) if yolo_pt.is_file() else "yolo11m.pt")
    print("      OK\n")

    print("[2/4] BoxMOT OSNet Re-ID (osnet_x0_25_msmt17.pt) …")
    osnet_dst = root / "models" / "osnet_x0_25_msmt17.pt"
    if not osnet_dst.is_file():
        osnet_dst.parent.mkdir(parents=True, exist_ok=True)
        url = "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"
        try:
            from boxmot.reid.core.config import TRAINED_URLS

            url = TRAINED_URLS["osnet_x0_25_msmt17.pt"]
        except Exception:
            pass
        try:
            import gdown

            gdown.download(url, str(osnet_dst), quiet=False, fuzzy=False)
            if not osnet_dst.is_file() or osnet_dst.stat().st_size < 1_000_000:
                raise RuntimeError("download incomplete or not a .pt file")
            print(f"      saved {osnet_dst}\n")
        except Exception as ex:
            try:
                urllib.request.urlretrieve(url, osnet_dst)
                if osnet_dst.is_file() and osnet_dst.stat().st_size > 1_000_000:
                    print(f"      saved {osnet_dst}\n")
                else:
                    raise RuntimeError("urllib got HTML or tiny file (use gdown)") from ex
            except Exception as ex2:
                print(
                    f"      skipped ({ex2}); place weights manually or: "
                    f"pip install gdown && gdown '{url}' -O {osnet_dst}\n"
                )
    else:
        print("      already present\n")

    import torch
    from sway.pose_estimator import PoseEstimator

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for i, mid in enumerate(
        (
            "usyd-community/vitpose-plus-base",
            "usyd-community/vitpose-plus-large",
        ),
        start=3,
    ):
        print(f"[{i}/4] ViTPose {mid} …")
        est = PoseEstimator(device=device, model_name=mid)
        del est
        gc.collect()
        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        print(f"      OK\n")

    print("Prefetch complete. Before an offline run:")
    print("  export SWAY_OFFLINE=1   # macOS/Linux")
    print("  set SWAY_OFFLINE=1      # Windows cmd")
    print("See README.md (Offline section) for details.")


if __name__ == "__main__":
    main()
