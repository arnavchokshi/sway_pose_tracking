#!/usr/bin/env python3
"""Plot Ultralytics results.csv (loss + detection metrics vs epoch)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path, help="Path to results.csv")
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same dir as csv)",
    )
    args = p.parse_args()
    csv: Path = args.csv
    out = args.out_dir or csv.parent
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    ep = df["epoch"]

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(ep, df["train/box_loss"], label="train box", color="#2563eb")
    axes[0].plot(ep, df["train/cls_loss"], label="train cls", color="#7c3aed")
    axes[0].plot(ep, df["train/dfl_loss"], label="train dfl", color="#0891b2")
    axes[0].set_ylabel("Train loss")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("DanceTrack-only YOLO26l fine-tune — training loss")

    axes[1].plot(ep, df["val/box_loss"], label="val box", color="#1d4ed8")
    axes[1].plot(ep, df["val/cls_loss"], label="val cls", color="#6d28d9")
    axes[1].plot(ep, df["val/dfl_loss"], label="val dfl", color="#0e7490")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation loss")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    loss_path = out / "epochs_losses.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ep, df["metrics/precision(B)"], "o-", label="Precision", color="#0f766e")
    ax.plot(ep, df["metrics/recall(B)"], "s-", label="Recall", color="#b45309")
    ax.plot(ep, df["metrics/mAP50(B)"], "^-", label="mAP@0.5", color="#1e40af")
    ax.plot(ep, df["metrics/mAP50-95(B)"], "d-", label="mAP@0.5:0.95", color="#7c2d12")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.set_title("Validation metrics vs epoch")
    plt.tight_layout()
    met_path = out / "epochs_metrics.png"
    fig2.savefig(met_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote {loss_path}")
    print(f"Wrote {met_path}")


if __name__ == "__main__":
    main()
