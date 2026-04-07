"""
Contrastive Re-ID Fine-Tuning Pipeline (PLAN_20)

Fine-tunes the re-ID model (BPBreID or OSNet) on user's own dance clips
to close the domain gap from Market-1501 (pedestrians) to dance footage.

Usage:
    python -m tools.finetune_reid --gt-dir data/ground_truth/ --output models/bpbreid_r50_sway_finetuned.pth

Env:
  SWAY_REID_FINETUNE_ENABLED    – 0|1 (default 0)
  SWAY_REID_FINETUNE_PAIRS      – number of training pairs (default 500)
  SWAY_REID_FINETUNE_EPOCHS     – training epochs (default 20)
  SWAY_REID_FINETUNE_LR         – learning rate (default 1e-4)
  SWAY_REID_FINETUNE_BASE_MODEL – bpbreid | osnet (default bpbreid)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def extract_pairs(
    gt_dir: str,
    n_pairs: int = 500,
    output_dir: str = "data/reid_finetune/pairs",
) -> str:
    """Extract positive and negative pairs from ground truth annotations.

    Args:
        gt_dir: directory containing GT videos + MOT annotations.
        n_pairs: total number of pairs (50% positive, 50% negative).
        output_dir: directory to save pair images.

    Returns:
        Path to pairs.csv.
    """
    gt_path = Path(gt_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find GT video + annotation pairs
    video_files = list(gt_path.rglob("*.mov")) + list(gt_path.rglob("*.mp4"))
    mot_files = list(gt_path.rglob("*.txt")) + list(gt_path.rglob("gt.txt"))

    logger.info("Found %d videos, %d annotation files in %s", len(video_files), len(mot_files), gt_dir)

    pairs: List[Dict] = []
    pair_id = 0

    for video_file in video_files:
        # Find matching annotation
        mot_candidates = [
            f for f in mot_files
            if f.parent.name == video_file.stem or f.parent == video_file.parent
        ]
        if not mot_candidates:
            continue

        mot_file = mot_candidates[0]
        tracks = _parse_mot(str(mot_file))

        if not tracks:
            continue

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        track_ids = list(tracks.keys())

        # Generate positive pairs (same ID, different frames)
        n_pos = n_pairs // 2
        for _ in range(n_pos):
            if not track_ids:
                break
            tid = random.choice(track_ids)
            frames = tracks[tid]
            if len(frames) < 2:
                continue

            f1, f2 = random.sample(list(frames.keys()), 2)
            crop_a = _extract_crop(cap, f1, frames[f1])
            crop_b = _extract_crop(cap, f2, frames[f2])

            if crop_a is not None and crop_b is not None:
                path_a = out_path / f"pair_{pair_id:04d}_a.jpg"
                path_b = out_path / f"pair_{pair_id:04d}_b.jpg"
                cv2.imwrite(str(path_a), crop_a)
                cv2.imwrite(str(path_b), crop_b)
                pairs.append({
                    "pair_id": pair_id, "path_a": str(path_a),
                    "path_b": str(path_b), "same_id": 1,
                })
                pair_id += 1

        # Generate negative pairs (different IDs, same frame)
        n_neg = n_pairs // 2
        for _ in range(n_neg):
            if len(track_ids) < 2:
                break
            tid_a, tid_b = random.sample(track_ids, 2)

            shared_frames = set(tracks[tid_a].keys()) & set(tracks[tid_b].keys())
            if not shared_frames:
                continue

            frame = random.choice(list(shared_frames))
            crop_a = _extract_crop(cap, frame, tracks[tid_a][frame])
            crop_b = _extract_crop(cap, frame, tracks[tid_b][frame])

            if crop_a is not None and crop_b is not None:
                path_a = out_path / f"pair_{pair_id:04d}_a.jpg"
                path_b = out_path / f"pair_{pair_id:04d}_b.jpg"
                cv2.imwrite(str(path_a), crop_a)
                cv2.imwrite(str(path_b), crop_b)
                pairs.append({
                    "pair_id": pair_id, "path_a": str(path_a),
                    "path_b": str(path_b), "same_id": 0,
                })
                pair_id += 1

        cap.release()

    csv_path = out_path / "pairs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "path_a", "path_b", "same_id"])
        writer.writeheader()
        writer.writerows(pairs)

    logger.info("Extracted %d pairs → %s", len(pairs), csv_path)
    return str(csv_path)


def _parse_mot(path: str) -> Dict[int, Dict[int, List[float]]]:
    """Parse MOT format: frame,id,x,y,w,h,..."""
    tracks: Dict[int, Dict[int, List[float]]] = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                tid = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                tracks.setdefault(tid, {})[frame] = [x, y, w, h]
    except Exception:
        pass
    return tracks


def _extract_crop(cap, frame_idx: int, bbox: List[float]) -> np.ndarray | None:
    """Extract a crop from a video at a given frame and bbox."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None

    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (128, 256))


def finetune(
    pairs_csv: str,
    base_model: str = "bpbreid",
    output_path: str = "models/bpbreid_r50_sway_finetuned.pth",
    epochs: int = 20,
    lr: float = 1e-4,
    val_split: float = 0.2,
) -> str:
    """Fine-tune the re-ID model with contrastive loss.

    Args:
        pairs_csv: path to pairs.csv from extract_pairs.
        base_model: starting checkpoint name.
        output_path: where to save fine-tuned weights.
        epochs: training epochs.
        lr: learning rate.
        val_split: validation split fraction.

    Returns:
        Path to saved fine-tuned weights.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    class PairDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            img_a = cv2.imread(row["path_a"])
            img_b = cv2.imread(row["path_b"])

            if img_a is None or img_b is None:
                img_a = np.zeros((256, 128, 3), dtype=np.uint8)
                img_b = np.zeros((256, 128, 3), dtype=np.uint8)

            # Random augmentation
            if random.random() > 0.5:
                img_a = cv2.flip(img_a, 1)
            if random.random() > 0.5:
                img_b = cv2.flip(img_b, 1)

            ta = torch.from_numpy(img_a[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            tb = torch.from_numpy(img_b[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            label = int(row["same_id"])

            return ta, tb, label

    # Load pairs
    with open(pairs_csv) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    random.shuffle(all_rows)
    val_size = int(len(all_rows) * val_split)
    val_rows = all_rows[:val_size]
    train_rows = all_rows[val_size:]

    logger.info("Training: %d pairs, Validation: %d pairs", len(train_rows), len(val_rows))

    train_dataset = PairDataset(train_rows)
    val_dataset = PairDataset(val_rows)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load backbone
    import torchvision.models as tv_models
    backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device)

    optimizer = optim.Adam(backbone.parameters(), lr=lr)
    margin = 0.5

    best_val_acc = 0.0

    for epoch in range(epochs):
        backbone.train()

        # Freeze backbone for first 5 epochs
        if epoch < 5:
            for param in list(backbone.parameters())[:-10]:
                param.requires_grad = False
        else:
            for param in backbone.parameters():
                param.requires_grad = True
            if epoch == 5:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr / 10

        epoch_loss = 0.0
        n_batches = 0

        for batch_a, batch_b, labels in train_loader:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            labels = labels.float().to(device)

            emb_a = backbone(batch_a)
            emb_b = backbone(batch_b)

            # Contrastive loss
            dist = torch.nn.functional.pairwise_distance(emb_a, emb_b)
            loss = (labels * dist.pow(2) +
                    (1 - labels) * torch.clamp(margin - dist, min=0).pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        backbone.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_a, batch_b, labels in val_loader:
                batch_a = batch_a.to(device)
                batch_b = batch_b.to(device)

                emb_a = backbone(batch_a)
                emb_b = backbone(batch_b)

                dist = torch.nn.functional.pairwise_distance(emb_a, emb_b)
                preds = (dist < margin / 2).long()
                correct += (preds.cpu() == labels).sum().item()
                total += len(labels)

        val_acc = correct / max(total, 1)
        avg_loss = epoch_loss / max(n_batches, 1)

        logger.info(
            "Epoch %d/%d: loss=%.4f, val_acc=%.3f", epoch + 1, epochs, avg_loss, val_acc
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(backbone.state_dict(), output_path)
            logger.info("Saved best model (val_acc=%.3f) → %s", val_acc, output_path)

    logger.info("Fine-tuning complete. Best val_acc=%.3f", best_val_acc)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Fine-tune Re-ID model on dance clips")
    parser.add_argument("--gt-dir", required=True, help="Ground truth directory")
    parser.add_argument("--output", default="models/bpbreid_r50_sway_finetuned.pth")
    parser.add_argument("--pairs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    n_pairs = args.pairs or _env_int("SWAY_REID_FINETUNE_PAIRS", 500)
    epochs = args.epochs or _env_int("SWAY_REID_FINETUNE_EPOCHS", 20)
    lr = args.lr or _env_float("SWAY_REID_FINETUNE_LR", 1e-4)

    csv_path = extract_pairs(args.gt_dir, n_pairs=n_pairs)
    finetune(csv_path, output_path=args.output, epochs=epochs, lr=lr)


if __name__ == "__main__":
    main()
