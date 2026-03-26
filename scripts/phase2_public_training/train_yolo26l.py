#!/usr/bin/env python3
"""
Fine-tune YOLO26l on DanceTrack ± CrowdHuman for dance-domain person detection.

Requirements before running:
  - GPU with >=16GB VRAM recommended; CUDA for reasonable speed
  - pip install ultralytics pillow  (Pillow is in repo requirements; use a recent ultralytics for YOLO26)

Phases (disk-friendly — train one dataset at a time):
  1) DanceTrack only (no CrowdHuman):
       python scripts/phase2_public_training/train_yolo26l.py --phase dancetrack
     Best: runs/detect/yolo26l_dancetrack_only/weights/best.pt

  2) After CrowdHuman is converted, continue from (1):
       python scripts/phase2_public_training/train_yolo26l.py --phase crowdhuman \\
         --weights runs/detect/yolo26l_dancetrack_only/weights/best.pt
     (Omit --weights if that exact path exists — it is the default.)
     Best: runs/detect/yolo26l_crowdhuman_ft/weights/best.pt

  Resume after crash / spot preemption (same run dir, same phase):
       YOLO_TRAIN_RESUME=1 python scripts/phase2_public_training/train_yolo26l.py --phase crowdhuman --resume
     Or:  python ... --resume --resume-from runs/detect/yolo26l_crowdhuman_ft/weights/last.pt

  Merged (both datasets on disk, single run — same as before):
       python scripts/phase2_public_training/train_yolo26l.py --phase merged
     Best: runs/detect/yolo26l_dancetrack/weights/best.pt

Artifacts per run:
  - training_manifest.json — hyperparams, weights path, git sha, timestamps
  - weights/best.pt, weights/last.pt, epoch*.pt every SAVE_PERIOD epochs
  - results.csv, results.png (Ultralytics)

Env (optional):
  YOLO_TRAIN_RESUME=1       — same as --resume when last.pt exists
  YOLO_TRAIN_CACHE=disk|ram — cache images for faster epochs (disk uses more SSD; omit = off)
  YOLO_TRAIN_BATCH=N       — fix batch size (default: -1 = AutoBatch)

Copy best.pt to models/ and set SWAY_YOLO_WEIGHTS=...

Lambda (fetch + convert + train): run_lambda_yolo_train.sh with YOLO_LAMBDA_PIPELINE=
  dancetrack | crowdhuman | dancetrack_crowdhuman — see LAMBDA_TRAINING.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
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
        "run_name": "yolo26l_dancetrack",
    },
    "dancetrack": {
        "data_yaml": "scripts/phase2_public_training/dancetrack_only.yaml",
        "required_dirs": (
            "datasets/dancetrack_yolo/images/train",
            "datasets/dancetrack_yolo/images/val",
        ),
        "run_name": "yolo26l_dancetrack_only",
    },
    "crowdhuman": {
        "data_yaml": "scripts/phase2_public_training/crowdhuman_only.yaml",
        "required_dirs": (
            "datasets/crowdhuman_yolo/images/train",
            "datasets/crowdhuman_yolo/images/val",
        ),
        "run_name": "yolo26l_crowdhuman_ft",
    },
}

# CrowdHuman stage init weights (first match wins if --weights omitted):
DEFAULT_CROWDHUMAN_PARENT_WEIGHTS = (
    REPO_ROOT / "runs/detect/yolo26l_dancetrack_only/weights/best.pt"
)
# Same filename the pipeline / Lab uses for DanceTrack fine-tunes — copy your saved .pt here on Lambda.
MODELS_DANCETRACK_ALIAS = REPO_ROOT / "models" / "yolo26l_dancetrack.pt"


def resolve_crowdhuman_parent_weights() -> Path | None:
    for p in (DEFAULT_CROWDHUMAN_PARENT_WEIGHTS, MODELS_DANCETRACK_ALIAS):
        if p.is_file():
            return p
    return None

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
SAVE_PERIOD = 5
# Disable mosaic in last N epochs for cleaner metrics / less jitter.
CLOSE_MOSAIC = 10
# ─────────────────────────────────────────────────────────────────────────────


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _env_resume_requested() -> bool:
    v = os.environ.get("YOLO_TRAIN_RESUME", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _parse_cache() -> bool | str:
    """Ultralytics: False | True | 'ram' | 'disk'."""
    raw = os.environ.get("YOLO_TRAIN_CACHE", "").strip().lower()
    if not raw:
        return False
    if raw in ("1", "true", "yes", "ram"):
        return True
    if raw == "disk":
        return "disk"
    return False


def _parse_batch() -> int:
    raw = os.environ.get("YOLO_TRAIN_BATCH", "").strip()
    if not raw:
        return BATCH
    try:
        return int(raw)
    except ValueError:
        return BATCH


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
        found = resolve_crowdhuman_parent_weights()
        if found is None:
            print("\n❌ CrowdHuman stage needs your DanceTrack-trained checkpoint (skip re-training DanceTrack).")
            print(f"   Put best.pt at one of:")
            print(f"     • {DEFAULT_CROWDHUMAN_PARENT_WEIGHTS}")
            print(f"     • {MODELS_DANCETRACK_ALIAS}")
            print("   Or pass:  --weights /path/to/your_dancetrack_best.pt")
            raise SystemExit(1)
        return str(found.resolve())

    # merged / dancetrack: COCO-pretrained backbone (Ultralytics downloads if missing)
    return "yolo26l.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO26l (DanceTrack / CrowdHuman / merged).")
    p.add_argument(
        "--phase",
        choices=("merged", "dancetrack", "crowdhuman"),
        default="merged",
        help="merged = both datasets; dancetrack = no CrowdHuman; crowdhuman = second stage (needs DanceTrack best.pt)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Checkpoint to start from. Default: yolo26l.pt (merged/dancetrack), or first existing of "
        f"{DEFAULT_CROWDHUMAN_PARENT_WEIGHTS.relative_to(REPO_ROOT)}, models/yolo26l_dancetrack.pt (crowdhuman).",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Override Ultralytics run folder name.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last.pt in this run (or --resume-from). Same as YOLO_TRAIN_RESUME=1.",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        metavar="PATH",
        help="Explicit path to last.pt (default: <run_dir>/weights/last.pt).",
    )
    return p.parse_args()


def _write_manifest(
    run_dir: Path,
    *,
    phase: str,
    weights_or_resume: str,
    data_yaml_rel: str,
    resume: bool,
    train_kwargs: dict,
) -> None:
    manifest = {
        "schema": "sway_yolo_train_manifest_v1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(REPO_ROOT),
        "git_head": _git_head(),
        "phase": phase,
        "resume": resume,
        "weights_or_resume_path": weights_or_resume,
        "data_yaml": data_yaml_rel,
        "train_kwargs": {k: v for k, v in train_kwargs.items() if not callable(v)},
    }
    path = run_dir / "training_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"✓ Wrote {path.relative_to(REPO_ROOT)}")


def _oom_retry_train(model, build_kwargs, batch_fallbacks: list[int]) -> None:
    import torch

    last_err: BaseException | None = None
    for i, batch in enumerate(batch_fallbacks):
        kw = build_kwargs(batch)
        try:
            model.train(**kw)
            return
        except RuntimeError as e:
            last_err = e
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                raise
            if i == len(batch_fallbacks) - 1:
                print("\n❌ CUDA OOM after batch fallbacks. Set YOLO_TRAIN_BATCH to a small integer or lower IMGSZ in train_yolo26l.py.")
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"\n⚠ CUDA OOM — retrying with batch={batch_fallbacks[i + 1]} …\n", flush=True)
    assert last_err is not None
    raise last_err


def train() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    ok, data_yaml_rel, run_name = verify_data_layout(args.phase)
    if not ok:
        raise SystemExit(1)

    if args.name:
        run_name = args.name

    run_dir = REPO_ROOT / PROJECT / run_name
    resume_flag = args.resume or _env_resume_requested()
    resume_path: Path | None = None
    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
    elif resume_flag:
        resume_path = run_dir / "weights" / "last.pt"

    if resume_flag or args.resume_from:
        if resume_path is None or not resume_path.is_file():
            print(f"\n❌ Resume requested but checkpoint missing: {resume_path}")
            print("   Train without --resume to start fresh, or point --resume-from to weights/last.pt")
            raise SystemExit(1)
        print(f"✓ Resuming from {resume_path}")
        from ultralytics import YOLO

        _write_manifest(
            run_dir,
            phase=args.phase,
            weights_or_resume=str(resume_path),
            data_yaml_rel=data_yaml_rel,
            resume=True,
            train_kwargs={"resume": True},
        )
        try:
            model = YOLO(str(resume_path))
            model.train(resume=True)
        except Exception:
            print("\n❌ Training failed during resume:\n", file=sys.stderr)
            traceback.print_exc()
            raise SystemExit(1)

        best = run_dir / "weights" / "best.pt"
        print("\n✓ Training complete (resume).")
        print(f"  Best checkpoint: {best}")
        print(f"  Per-epoch log: {run_dir / 'results.csv'}")
        return

    weights = resolve_weights(args.phase, args.weights)
    lr0 = LR0_CROWDHUMAN_PHASE if args.phase == "crowdhuman" else LR0_DEFAULT
    cache = _parse_cache()
    batch_env = _parse_batch()

    from ultralytics import YOLO

    print(f"✓ Weights: {weights}")
    print(f"✓ Data: {data_yaml_rel}")
    print(f"✓ Run name: {run_name}")
    print(f"✓ Run directory (metrics, plots, weights):\n  {run_dir}")
    print("  • training_manifest.json — this run’s hyperparameters")
    print("  • results.csv — loss & mAP per epoch (tail -f)")
    print("  • weights/best.pt, weights/last.pt, epoch*.pt (every SAVE_PERIOD epochs)")
    print(f"✓ Epochs: {EPOCHS} (early stop patience={PATIENCE} if val plateaus)")
    print(f"✓ save_period={SAVE_PERIOD} (intermediate checkpoints for crash recovery)")
    if cache:
        print(f"✓ Image cache: {cache} (YOLO_TRAIN_CACHE)")
    print(f"✓ Batch: {batch_env} (set YOLO_TRAIN_BATCH to pin batch if AutoBatch misbehaves)\n")

    def build_train_kwargs(batch_val: int) -> dict:
        return {
            "data": str(REPO_ROOT / data_yaml_rel),
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": batch_val,
            "device": DEVICE,
            "workers": WORKERS,
            "project": str(REPO_ROOT / PROJECT),
            "name": run_name,
            "lr0": lr0,
            "lrf": 0.01,
            "warmup_epochs": 3,
            "cos_lr": True,
            "flipud": 0.3,
            "degrees": 15,
            "mixup": 0.1,
            "copy_paste": 0.1,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "close_mosaic": CLOSE_MOSAIC,
            "freeze": 10,
            "patience": PATIENCE,
            "seed": SEED,
            "plots": True,
            "save": True,
            "save_period": SAVE_PERIOD,
            "exist_ok": True,
            "verbose": True,
            "amp": True,
            "cache": cache,
        }

    train_kwargs_snapshot = build_train_kwargs(batch_env)
    _write_manifest(
        run_dir,
        phase=args.phase,
        weights_or_resume=weights,
        data_yaml_rel=data_yaml_rel,
        resume=False,
        train_kwargs=train_kwargs_snapshot,
    )

    model = YOLO(weights)

    if batch_env != -1:
        try:
            model.train(**build_train_kwargs(batch_env))
        except Exception:
            print("\n❌ Training failed:\n", file=sys.stderr)
            traceback.print_exc()
            raise SystemExit(1)
    else:
        fallbacks = [-1, 16, 12, 8, 6, 4, 2, 1]
        try:
            _oom_retry_train(model, build_train_kwargs, fallbacks)
        except Exception:
            print("\n❌ Training failed:\n", file=sys.stderr)
            traceback.print_exc()
            raise SystemExit(1)

    best = run_dir / "weights" / "best.pt"
    print("\n✓ Training complete.")
    print(f"  Best checkpoint: {best}")
    print(f"  Per-epoch log: {run_dir / 'results.csv'}")
    print("\nTo use in pipeline:")
    print(f"  cp {best} models/yolo26l_dancetrack.pt")
    print("  export SWAY_YOLO_WEIGHTS=models/yolo26l_dancetrack.pt")
    if args.phase == "dancetrack":
        print("\nNext (after CrowdHuman is on disk and converted):")
        print(
            "  python scripts/phase2_public_training/train_yolo26l.py --phase crowdhuman \\\n"
            f"    --weights {best}"
        )
    print("\nIf this run was interrupted, resume with:")
    print(f"  YOLO_TRAIN_RESUME=1 python scripts/phase2_public_training/train_yolo26l.py --phase {args.phase} --resume")


if __name__ == "__main__":
    train()
