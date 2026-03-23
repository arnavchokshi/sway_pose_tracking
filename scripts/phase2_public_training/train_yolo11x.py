#!/usr/bin/env python3
"""
Fine-tune YOLO11x on DanceTrack ± CrowdHuman for dance-domain person detection.

Requirements before running:
  - GPU with >=16GB VRAM recommended; CUDA for reasonable speed
  - pip install ultralytics pillow  (Pillow is in repo requirements)

Phases (disk-friendly — train one dataset at a time):
  1) DanceTrack only (no CrowdHuman):
       python scripts/phase2_public_training/train_yolo11x.py --phase dancetrack
     Best: runs/detect/yolo11x_dancetrack_only/weights/best.pt

  2) After CrowdHuman is converted, continue from (1):
       python scripts/phase2_public_training/train_yolo11x.py --phase crowdhuman \\
         --weights runs/detect/yolo11x_dancetrack_only/weights/best.pt
     (Omit --weights if that exact path exists — it is the default.)
     Best: runs/detect/yolo11x_crowdhuman_ft/weights/best.pt

  Merged (both datasets on disk, single run — same as before):
       python scripts/phase2_public_training/train_yolo11x.py --phase merged
     Best: runs/detect/yolo11x_dancetrack/weights/best.pt

Copy best.pt to models/ and set SWAY_YOLO_WEIGHTS=...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PHASE_CONFIG = {
    "merged": {
        "data_yaml": "scripts/phase2_public_training/dancetrack_crowdhuman.yaml",
        "required_dirs": (
            "datasets/dancetrack_yolo/images/train",
            "datasets/dancetrack_yolo/images/val",
            "datasets/crowdhuman_yolo/images/train",
            "datasets/crowdhuman_yolo/images/val",
        ),
        "run_name": "yolo11x_dancetrack",
    },
    "dancetrack": {
        "data_yaml": "scripts/phase2_public_training/dancetrack_only.yaml",
        "required_dirs": (
            "datasets/dancetrack_yolo/images/train",
            "datasets/dancetrack_yolo/images/val",
        ),
        "run_name": "yolo11x_dancetrack_only",
    },
    "crowdhuman": {
        "data_yaml": "scripts/phase2_public_training/crowdhuman_only.yaml",
        "required_dirs": (
            "datasets/crowdhuman_yolo/images/train",
            "datasets/crowdhuman_yolo/images/val",
        ),
        "run_name": "yolo11x_crowdhuman_ft",
    },
}

DEFAULT_CROWDHUMAN_PARENT_WEIGHTS = (
    REPO_ROOT / "runs/detect/yolo11x_dancetrack_only/weights/best.pt"
)

# ── Shared train hyperparameters ─────────────────────────────────────────────
PROJECT = "runs/detect"
EPOCHS = 80
IMGSZ = 960
BATCH = -1  # AutoBatch (~60% VRAM); set e.g. 8 for 16GB if AutoBatch fails
DEVICE = 0  # GPU index; use "mps" for Apple Silicon, "cpu" as last resort
WORKERS = 8  # reduce to 4 if dataloader OOM
# Slightly lower LR on CrowdHuman-only stage to reduce forgetting DanceTrack.
LR0_DEFAULT = 0.0005
LR0_CROWDHUMAN_PHASE = 0.00025
# Early stop if val metrics plateau (saves time; best.pt is still best epoch so far).
PATIENCE = 25
SEED = 42
# Extra checkpoint every N epochs (crash recovery); 0 disables.
SAVE_PERIOD = 10
# ─────────────────────────────────────────────────────────────────────────────


def verify_data_layout(phase: str) -> tuple[bool, str, str]:
    """Returns (ok, data_yaml_rel, run_name)."""
    import torch

    cfg = PHASE_CONFIG[phase]
    errors: list[str] = []
    for rel in cfg["required_dirs"]:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"Missing: {rel} — run the converter scripts first")

    data_yaml = cfg["data_yaml"]
    if not (REPO_ROOT / data_yaml).is_file():
        errors.append(f"Missing: {data_yaml}")

    run_name = cfg["run_name"]

    if errors:
        print("\n❌ Setup incomplete:")
        for e in errors:
            print(f"   {e}")
        return False, data_yaml, run_name

    print(f"✓ Phase: {phase}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ VRAM: {mem:.1f} GB")
    else:
        print("⚠ No CUDA GPU found — training will be very slow on CPU/MPS")
    return True, data_yaml, run_name


def resolve_weights(phase: str, weights_arg: str | None) -> str:
    if weights_arg is not None:
        p = Path(weights_arg)
        if not p.is_file():
            print(f"\n❌ --weights not found: {p.resolve()}")
            raise SystemExit(1)
        return str(p.resolve())

    if phase == "crowdhuman":
        if not DEFAULT_CROWDHUMAN_PARENT_WEIGHTS.is_file():
            print("\n❌ CrowdHuman stage needs a DanceTrack checkpoint.")
            print(f"   Expected: {DEFAULT_CROWDHUMAN_PARENT_WEIGHTS}")
            print("   Or pass:  --weights /path/to/dancetrack_only_best.pt")
            raise SystemExit(1)
        return str(DEFAULT_CROWDHUMAN_PARENT_WEIGHTS.resolve())

    # merged / dancetrack: COCO-pretrained backbone (Ultralytics downloads if missing)
    return "yolo11x.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO11x (DanceTrack / CrowdHuman / merged).")
    p.add_argument(
        "--phase",
        choices=("merged", "dancetrack", "crowdhuman"),
        default="merged",
        help="merged = both datasets; dancetrack = no CrowdHuman; crowdhuman = second stage (needs DanceTrack best.pt)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Checkpoint to start from. Default: yolo11x.pt (merged/dancetrack), or "
        f"{DEFAULT_CROWDHUMAN_PARENT_WEIGHTS.relative_to(REPO_ROOT)} (crowdhuman).",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Override Ultralytics run folder name.",
    )
    return p.parse_args()


def train() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    ok, data_yaml_rel, run_name = verify_data_layout(args.phase)
    if not ok:
        raise SystemExit(1)

    weights = resolve_weights(args.phase, args.weights)
    if args.name:
        run_name = args.name

    lr0 = LR0_CROWDHUMAN_PHASE if args.phase == "crowdhuman" else LR0_DEFAULT

    from ultralytics import YOLO

    run_dir = REPO_ROOT / PROJECT / run_name

    print(f"✓ Weights: {weights}")
    print(f"✓ Data: {data_yaml_rel}")
    print(f"✓ Run name: {run_name}")
    print(f"✓ Run directory (metrics, plots, weights):\n  {run_dir}")
    print("  • results.csv — loss & mAP per epoch (open in Numbers/Excel or tail -f)")
    print("  • results.png, confusion_matrix*.png — after training")
    print("  • weights/best.pt, weights/last.pt")
    print(f"✓ Epochs: {EPOCHS} (early stop patience={PATIENCE} if val plateaus)\n")

    model = YOLO(weights)

    model.train(
        data=str(REPO_ROOT / data_yaml_rel),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(REPO_ROOT / PROJECT),
        name=run_name,
        lr0=lr0,
        lrf=0.01,
        warmup_epochs=3,
        flipud=0.3,
        degrees=15,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        freeze=10,
        patience=PATIENCE,
        seed=SEED,
        plots=True,
        save=True,
        save_period=SAVE_PERIOD,
        exist_ok=True,
        verbose=True,
        amp=True,
    )

    best = run_dir / "weights" / "best.pt"
    print("\n✓ Training complete.")
    print(f"  Best checkpoint: {best}")
    print(f"  Per-epoch log: {run_dir / 'results.csv'}")
    print("\nTo use in pipeline:")
    print(f"  cp {best} models/yolo11x_dancetrack.pt")
    print("  export SWAY_YOLO_WEIGHTS=models/yolo11x_dancetrack.pt")
    if args.phase == "dancetrack":
        print("\nNext (after CrowdHuman is on disk and converted):")
        print(
            "  python scripts/phase2_public_training/train_yolo11x.py --phase crowdhuman \\\n"
            f"    --weights {best}"
        )


if __name__ == "__main__":
    train()
