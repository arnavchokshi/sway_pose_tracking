"""
Meta Sapiens pose (COCO-17) via official TorchScript exports.

Download a COCO-17 ``*_coco_best_coco_AP_*_torchscript.pt2`` from the Sapiens COCO model zoo
(e.g. ``noahcao/sapiens-pose-coco`` on Hugging Face under
``sapiens_lite_host/torchscript/pose/checkpoints/``), then set:

  export SWAY_SAPIENS_TORCHSCRIPT=/path/to/sapiens_0.6b_coco_best_coco_AP_812_torchscript.pt2

Preprocessing matches the public Hugging Face Space (resize 1024×768, ImageNet-style normalize).
Heatmaps are decoded with argmax per joint; coordinates are mapped back to the **original**
person crop (then to full image), matching the Space demo logic (256×192 heatmap grid).

Requires PyTorch 2.2+. CUDA recommended; Apple MPS is not used for TorchScript here (CPU fallback).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .pose_estimator import (
    BBOX_PADDING,
    _apply_sam_mask_gate_to_expanded_crop,
    _expand_bbox,
)

# Default heatmap spatial size (H, W) for lite TorchScript COCO pose — matches HF sapiens-pose Space.
_DEFAULT_HEATMAP_H = 256
_DEFAULT_HEATMAP_W = 192


def _heatmap_hw_from_env() -> Tuple[int, int]:
    """Override if a future export uses a different resolution (H, W)."""
    try:
        h = int(os.environ.get("SWAY_SAPIENS_HEATMAP_H", str(_DEFAULT_HEATMAP_H)))
        w = int(os.environ.get("SWAY_SAPIENS_HEATMAP_W", str(_DEFAULT_HEATMAP_W)))
        return max(1, h), max(1, w)
    except ValueError:
        return _DEFAULT_HEATMAP_H, _DEFAULT_HEATMAP_W


def heatmaps_to_keypoints_xyxy(
    heatmaps: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    heatmap_h: int,
    heatmap_w: int,
) -> np.ndarray:
    """
    Decode Sapiens-style per-joint heatmaps to COCO (17, 3) in **full-image** coordinates.

    heatmaps: (J, Hm, Wm) float array.
    """
    if heatmaps.ndim != 3:
        raise ValueError(f"expected heatmaps (J,H,W), got shape {heatmaps.shape}")
    j_count, hm_h, hm_w = heatmaps.shape
    bbox_w = max(float(x2 - x1), 1.0)
    bbox_h = max(float(y2 - y1), 1.0)
    out = np.zeros((j_count, 3), dtype=np.float32)
    for i in range(j_count):
        hm = heatmaps[i]
        flat = int(np.argmax(hm))
        y, x = np.unravel_index(flat, hm.shape)
        conf = float(hm[y, x])
        x_img = x * bbox_w / float(heatmap_w) + x1
        y_img = y * bbox_h / float(heatmap_h) + y1
        out[i, 0] = x_img
        out[i, 1] = y_img
        out[i, 2] = conf
    return out


def _resolve_torch_device(device: Union[torch.device, str]) -> torch.device:
    d = device if isinstance(device, torch.device) else torch.device(device)
    if d.type == "cuda":
        return d
    if d.type == "mps":
        print(
            "  Sapiens TorchScript: MPS not used for this export — running on CPU "
            "(set CUDA or accept slower CPU inference).",
            flush=True,
        )
        return torch.device("cpu")
    return d


class SapiensTorchscriptPoseEstimator:
    """COCO-17 keypoints via Meta Sapiens lite TorchScript (top-down, tracker boxes)."""

    def __init__(
        self,
        device: Union[torch.device, str] = "cpu",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        ckpt = checkpoint_path or os.environ.get("SWAY_SAPIENS_TORCHSCRIPT", "").strip()
        if not ckpt:
            raise ValueError(
                "Sapiens TorchScript: set SWAY_SAPIENS_TORCHSCRIPT to a .pt2 file "
                "(COCO-17 export from noahcao/sapiens-pose-coco or compatible)."
            )
        p = Path(ckpt)
        if not p.is_file():
            raise FileNotFoundError(f"Sapiens TorchScript checkpoint not found: {p}")

        self.checkpoint_path = p.resolve()
        self.run_device = _resolve_torch_device(device)
        self.heatmap_h, self.heatmap_w = _heatmap_hw_from_env()

        self._model = torch.jit.load(str(self.checkpoint_path), map_location="cpu")
        self._model.eval()
        self._model.to(self.run_device)

        self._transform = transforms.Compose(
            [
                transforms.Resize((1024, 768)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[123.5 / 255, 116.5 / 255, 103.5 / 255],
                    std=[58.5 / 255, 57.0 / 255, 57.5 / 255],
                ),
            ]
        )

    def _forward_heatmaps(self, pil: Image.Image) -> np.ndarray:
        t = self._transform(pil).unsqueeze(0).to(self.run_device)
        with torch.inference_mode():
            raw = self._model(t)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        hm = raw.detach().float().cpu().numpy()
        if hm.ndim == 4:
            hm = hm[0]
        if hm.ndim != 3:
            raise RuntimeError(f"Sapiens TorchScript: expected heatmaps (J,H,W), got {hm.shape}")
        return hm

    def _one_crop(
        self,
        crop_rgb: np.ndarray,
        ox: float,
        oy: float,
        w: int,
        h: int,
        tid: int,
    ) -> Dict[int, Dict]:
        if crop_rgb.size == 0 or w <= 0 or h <= 0:
            return {}
        pil = Image.fromarray(crop_rgb)
        hm = self._forward_heatmaps(pil)
        n_j = hm.shape[0]
        if n_j != 17:
            raise RuntimeError(
                f"Sapiens TorchScript: expected 17 COCO joints, got {n_j}. "
                "Use a COCO-17 *_coco_best_coco_AP_*_torchscript.pt2 checkpoint, "
                "not whole-body or Goliath exports."
            )
        kpts = heatmaps_to_keypoints_xyxy(
            hm, 0.0, 0.0, float(w), float(h), self.heatmap_h, self.heatmap_w
        )
        kpts[:, 0] += float(ox)
        kpts[:, 1] += float(oy)
        scores = kpts[:, 2].copy()
        return {int(tid): {"keypoints": kpts, "scores": scores}}

    def _estimate_plain(
        self,
        frame_rgb: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        track_ids: List[int],
        paddings: List[float],
    ) -> Dict[int, Dict]:
        out: Dict[int, Dict] = {}
        img_h, img_w = frame_rgb.shape[:2]
        for box, tid, pad in zip(boxes, track_ids, paddings):
            ex = _expand_bbox(box, img_w, img_h, pad)
            x1, y1, x2, y2 = ex
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(img_w, int(round(x2))), min(img_h, int(round(y2)))
            crop = frame_rgb[yi1:yi2, xi1:xi2]
            cw, ch = max(0, xi2 - xi1), max(0, yi2 - yi1)
            out.update(self._one_crop(crop, float(xi1), float(yi1), cw, ch, int(tid)))
        return out

    def estimate_poses(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        track_ids: List[int],
        paddings: Optional[List[float]] = None,
        segmentation_masks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Dict[int, Dict]:
        """
        Same contract as ``PoseEstimator.estimate_poses``: RGB uint8, global COCO-17 (x,y,score).
        """
        if len(boxes) == 0:
            return {}
        assert len(boxes) == len(track_ids)
        if paddings is None:
            paddings = [BBOX_PADDING] * len(boxes)
        assert len(paddings) == len(boxes)
        if segmentation_masks is None:
            segmentation_masks = [None] * len(boxes)
        assert len(segmentation_masks) == len(boxes)

        if all(m is None for m in segmentation_masks):
            return self._estimate_plain(frame, boxes, track_ids, paddings)

        plain_idx = [i for i, m in enumerate(segmentation_masks) if m is None]
        masked_idx = [i for i, m in enumerate(segmentation_masks) if m is not None]
        out: Dict[int, Dict] = {}
        if plain_idx:
            out.update(
                self._estimate_plain(
                    frame,
                    [boxes[i] for i in plain_idx],
                    [track_ids[i] for i in plain_idx],
                    [paddings[i] for i in plain_idx],
                )
            )
        img_h, img_w = frame.shape[:2]
        for i in masked_idx:
            tid = int(track_ids[i])
            box = boxes[i]
            pad = paddings[i]
            m = segmentation_masks[i]
            if m is None or m.size == 0:
                out.update(self._estimate_plain(frame, [box], [tid], [pad]))
                continue
            crop, ox, oy = _apply_sam_mask_gate_to_expanded_crop(frame, box, pad, m)
            ch, cw = crop.shape[:2]
            out.update(self._one_crop(crop, float(ox), float(oy), cw, ch, tid))
        return out
