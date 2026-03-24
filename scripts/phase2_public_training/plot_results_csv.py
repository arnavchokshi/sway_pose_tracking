#!/usr/bin/env python3
"""Plot Ultralytics YOLO results.csv: default = one validation accuracy curve; use --detailed for full panels."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, float]] = []
        for row in reader:
            if not row.get("epoch"):
                continue
            try:
                ep = int(float(row["epoch"]))
            except (TypeError, ValueError):
                continue
            out: dict[str, float] = {"epoch": float(ep)}
            for k, v in row.items():
                if k == "epoch" or v in (None, ""):
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
            rows.append(out)
        return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="Path to results.csv")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: <csv_stem>_accuracy.png or _metrics.png with --detailed)",
    )
    p.add_argument(
        "--detailed",
        action="store_true",
        help="Four-panel plot (mAP, P/R, train/val losses) instead of single accuracy chart",
    )
    args = p.parse_args()
    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No epoch rows in {args.csv}")

    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]

    def series(key: str) -> list[float]:
        return [r[key] for r in rows if key in r]

    out = args.out or (
        args.csv.parent / f"{args.csv.stem}_{'metrics' if args.detailed else 'accuracy'}.png"
    )

    if not args.detailed:
        key = "metrics/mAP50(B)"
        y_raw = series(key)
        if not y_raw or len(y_raw) != len(epochs):
            raise SystemExit(f"Missing column {key!r} for all epochs in {args.csv}")
        y_pct = [v * 100.0 for v in y_raw]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, y_pct, "o-", color="#2e7d32", linewidth=2, markersize=9)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("Validation accuracy", fontsize=13)
        y_lo, y_hi = min(y_pct), max(y_pct)
        span = y_hi - y_lo
        # Zoom y to data so small changes read clearly; pad scales with span (flat line → fixed band).
        y_pad = max(span * 0.35, 0.4) if span > 1e-9 else 0.6
        ax.set_ylim(max(0.0, y_lo - y_pad), min(100.0, y_hi + y_pad))
        x_lo, x_hi = min(epochs), max(epochs)
        x_span = max(x_hi - x_lo, 1.0)
        ax.set_xlim(x_lo - 0.25 * x_span, x_hi + 0.25 * x_span)
        ax.grid(True, alpha=0.35)
        ax.set_xticks(epochs)
        fig.subplots_adjust(bottom=0.22)
        fig.text(
            0.5,
            0.06,
            "mAP@50: box quality vs labels (≥50% overlap = correct). Higher is better.",
            ha="center",
            fontsize=9,
            color="0.35",
        )
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        fig.suptitle("YOLO training / validation (results.csv)", fontsize=12)

        ax = axes[0, 0]
        for key, label, color in (
            ("metrics/mAP50(B)", "mAP50", "C0"),
            ("metrics/mAP50-95(B)", "mAP50-95", "C1"),
        ):
            y = series(key)
            if y and len(y) == len(epochs):
                ax.plot(epochs, y, "o-", label=label, color=color)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mAP")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for key, label in (
            ("metrics/precision(B)", "precision"),
            ("metrics/recall(B)", "recall"),
        ):
            y = series(key)
            if y and len(y) == len(epochs):
                ax.plot(epochs, y, "o-", label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
            y = series(key)
            if y and len(y) == len(epochs):
                ax.plot(epochs, y, "o-", label=key.replace("train/", ""))
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss"):
            y = series(key)
            if y and len(y) == len(epochs):
                ax.plot(epochs, y, "o-", label=key.replace("val/", ""))
        ax.set_xlabel("epoch")
        ax.set_ylabel("val loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
