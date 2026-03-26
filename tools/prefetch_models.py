#!/usr/bin/env python3
"""
Download and cache every weight the pipeline needs for offline (airplane) runs.

Run once while you have internet, from the sway_pose_mvp directory:

  cd sway_pose_mvp
  python -m tools.prefetch_models
  python -m tools.prefetch_models --include-3d   # optional: MotionAGFormer-L + notes for repo clone
  python -m tools.prefetch_models --include-poseformerv2   # optional: PoseFormerV2 243-frame H36M weights

This touches:
  - Ultralytics YOLO (yolo26l.pt → models/ or hub cache)
  - Hugging Face ViTPose base + large + huge (HF cache, usually ~/.cache/huggingface)
  - BoxMOT OSNet Re-ID weights → models/osnet_x0_25_msmt17.pt (default tracker path)
  - StrongSORT AFLink_epoch20.pth → models/ (neural global stitch; step 6 downloads from official Drive)

Fine-tuning YOLO26l on DanceTrack + CrowdHuman (optional): base weights `yolo26l.pt`
are pulled automatically by Ultralytics the first time you run
`scripts/phase2_public_training/train_yolo26l.py` — no separate prefetch step for
that file. See `scripts/phase2_public_training/README.md` for the full workflow.

After that, set SWAY_OFFLINE=1 when running without network (see README.md, Offline).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Google Drive file id for MotionAGFormer-L H36M (official README table).
_MOTIONAGFORMER_L_H36M_GDRIVE_ID = "1WI8QSsD84wlXIdK1dLp6hPZq4FPozmVZ"
# PoseFormerV2 README: 243 frames, 27 kept, MPJPE 45.2 mm (27_243_45.2.bin).
_POSEFORMERV2_243_GDRIVE_ID = "14SpqPyq9yiblCzTH5CorymKCUsXapmkg"
# StrongSORT README "Data&Model Preparation" — AFLink_epoch20.pth (hosted as zip-wrapped torch save).
_AFLINK_EPOCH20_GDRIVE_ID = "1DFMUkL-dc-j8-fibcJIq-46Xoq_bFoO9"


def prefetch_motionagformer_l(models_dir: Path) -> None:
    """Download MotionAGFormer-L H36M checkpoint to models/ (same name as upstream eval)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    dst = models_dir / "motionagformer-l-h36m.pth.tr"
    if dst.is_file() and dst.stat().st_size > 10_000_000:
        print(f"  ✓ MotionAGFormer-L already at {dst}")
        return
    print("  Downloading MotionAGFormer-L (~200MB, Google Drive)…")
    try:
        import gdown

        gdown.download(
            f"https://drive.google.com/uc?id={_MOTIONAGFORMER_L_H36M_GDRIVE_ID}",
            str(dst),
            quiet=False,
        )
    except Exception as ex:
        print(f"  MotionAGFormer download failed ({ex}). Install gdown: pip install gdown")
        print(
            f"  Manual: save checkpoint as\n    {dst}\n"
            "  Link: https://github.com/TaatiTeam/MotionAGFormer#evaluation"
        )
        return
    if not dst.is_file() or dst.stat().st_size < 10_000_000:
        print("  Download incomplete — remove partial file and retry.")
        return
    print(f"  ✓ Saved {dst}")


def prefetch_poseformerv2_243(models_dir: Path) -> None:
    """Download PoseFormerV2 243-frame H36M checkpoint (official eval table)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    dst = models_dir / "27_243_45.2.bin"
    if dst.is_file() and dst.stat().st_size > 1_000_000:
        print(f"  ✓ PoseFormerV2 243-frame checkpoint already at {dst}")
        return
    print("  Downloading PoseFormerV2 27_243_45.2.bin (Google Drive)…")
    try:
        import gdown

        gdown.download(
            f"https://drive.google.com/uc?id={_POSEFORMERV2_243_GDRIVE_ID}",
            str(dst),
            quiet=False,
        )
    except Exception as ex:
        print(f"  PoseFormerV2 download failed ({ex}). Install gdown: pip install gdown")
        print(
            f"  Manual: save as\n    {dst}\n"
            "  Link: https://github.com/QitaoZhao/PoseFormerV2#evaluation"
        )
        return
    if not dst.is_file() or dst.stat().st_size < 1_000_000:
        print("  Download incomplete — remove partial file and retry.")
        return
    print(f"  ✓ Saved {dst}")


def main() -> None:
    root = _REPO_ROOT
    os.chdir(root)
    if os.environ.get("SWAY_OFFLINE", "").lower() in ("1", "true", "yes"):
        print("Unset SWAY_OFFLINE for prefetch (needs network).", file=sys.stderr)
        sys.exit(1)

    print(f"Working directory: {root}\n")

    print("[1/6] YOLO yolo26l.pt …")
    from ultralytics import YOLO

    yolo_pt = root / "models" / "yolo26l.pt"
    YOLO(str(yolo_pt) if yolo_pt.is_file() else "yolo26l.pt")
    print("      OK\n")

    print("[2/6] BoxMOT OSNet Re-ID (osnet_x0_25_msmt17.pt) …")
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
            "usyd-community/vitpose-plus-huge",
        ),
        start=3,
    ):
        print(f"[{i}/6] ViTPose {mid} …")
        est = PoseEstimator(device=device, model_name=mid)
        del est
        gc.collect()
        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        print(f"      OK\n")

    print("[6/6] StrongSORT AFLink (neural global stitch) …")
    aflink_dst = root / "models" / "AFLink_epoch20.pth"
    if aflink_dst.is_file() and aflink_dst.stat().st_size > 1_000_000:
        print(f"      already present: {aflink_dst}\n")
    else:
        aflink_dst.parent.mkdir(parents=True, exist_ok=True)
        print("      Downloading AFLink_epoch20.pth (~4.4MB, Google Drive)…")
        try:
            import gdown

            gdown.download(
                f"https://drive.google.com/uc?id={_AFLINK_EPOCH20_GDRIVE_ID}",
                str(aflink_dst),
                quiet=False,
                fuzzy=False,
            )
        except Exception as ex:
            print(f"      failed ({ex}). pip install gdown, or download manually:\n")
            print(
                f"        {aflink_dst}\n"
                "      https://github.com/dyhBUPT/StrongSORT#datamodel-preparation\n"
            )
        if not aflink_dst.is_file() or aflink_dst.stat().st_size < 1_000_000:
            print("      incomplete — remove partial file and retry prefetch.\n")
        else:
            print(f"      saved {aflink_dst}\n")

    print("Prefetch complete. Before an offline run:")
    print("  export SWAY_OFFLINE=1   # macOS/Linux")
    print("  set SWAY_OFFLINE=1      # Windows cmd")
    print("See README.md (Offline section) for details.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prefetch Sway pose pipeline weights.")
    ap.add_argument(
        "--include-3d",
        action="store_true",
        help="Also download MotionAGFormer-L (3D lift). Requires gdown for Google Drive.",
    )
    ap.add_argument(
        "--include-poseformerv2",
        action="store_true",
        help="Also download PoseFormerV2 243-frame H36M checkpoint. Requires gdown.",
    )
    ns = ap.parse_args()
    main()
    if ns.include_3d:
        print("\n[Optional] MotionAGFormer-L (3D pose lift) …")
        prefetch_motionagformer_l(_REPO_ROOT / "models")
        print(
            "Clone MotionAGFormer for imports:\n"
            "  git clone https://github.com/TaatiTeam/MotionAGFormer.git vendor/MotionAGFormer\n"
            "  pip install timm\n"
            "Or set SWAY_MOTIONAGFORMER_ROOT to your clone path."
        )
    if ns.include_poseformerv2:
        print("\n[Optional] PoseFormerV2 (3D lift, SWAY_LIFT_BACKEND=poseformerv2) …")
        prefetch_poseformerv2_243(_REPO_ROOT / "models")
        print(
            "Clone PoseFormerV2 for imports:\n"
            "  git clone https://github.com/QitaoZhao/PoseFormerV2.git vendor/PoseFormerV2\n"
            "  pip install einops torch-dct timm\n"
            "Or set SWAY_POSEFORMERV2_ROOT. Weights are H36M-trained (not AMASS)."
        )
