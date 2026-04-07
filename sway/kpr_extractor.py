"""
KPR Keypoint-Prompted Re-ID Extractor (PLAN_09)

Specialized for multi-person ambiguity — when multiple people are visible
in the same bounding box. Uses pose keypoints as explicit prompts to specify
which person inside an occluded crop to identify.

The key insight: in a multi-person crop, standard re-ID extracts features from
ALL people. KPR's keypoint heatmap tells the model "focus on the person at
THESE joint locations."

Env:
  SWAY_REID_KPR_ENABLED – 0|1 (default 1)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KPRExtractor:
    """Keypoint-Prompted Re-ID for multi-person ambiguity scenarios."""

    NUM_KEYPOINTS = 17  # COCO-17
    HEATMAP_SIGMA = 7.0

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        self._model = None

        if checkpoint_path is None:
            models_dir = Path(__file__).resolve().parent.parent / "models"
            checkpoint_path = str(models_dir / "kpr_r50.pth")

        self._checkpoint_path = checkpoint_path
        self._load_model()

    def _load_model(self) -> None:
        """Load KPR model or fall back to a ResNet-50 backbone."""
        try:
            import torch
            import torchvision.models as tv_models

            # Try loading KPR-specific checkpoint
            ckpt_path = Path(self._checkpoint_path)
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location=self.device)
                # Build backbone and load state
                backbone = tv_models.resnet50()
                # KPR modifies first conv to accept 3+17 = 20 channels
                old_conv = backbone.conv1
                backbone.conv1 = torch.nn.Conv2d(
                    20, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                # Copy pretrained RGB weights to first 3 channels
                with torch.no_grad():
                    backbone.conv1.weight[:, :3] = old_conv.weight
                    backbone.conv1.weight[:, 3:] = 0.01 * torch.randn_like(
                        backbone.conv1.weight[:, 3:]
                    )
                backbone.fc = torch.nn.Identity()

                if "model" in state:
                    backbone.load_state_dict(state["model"], strict=False)
                else:
                    backbone.load_state_dict(state, strict=False)

                backbone.eval().to(self.device)
                self._model = backbone
                logger.info("KPR model loaded from %s", ckpt_path)
                return

            # Fallback: standard ResNet-50 with 20-channel input
            backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
            old_conv = backbone.conv1
            backbone.conv1 = torch.nn.Conv2d(
                20, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                backbone.conv1.weight[:, :3] = old_conv.weight
                backbone.conv1.weight[:, 3:] = 0.0
            backbone.fc = torch.nn.Identity()
            backbone.eval().to(self.device)
            self._model = backbone
            logger.info("KPR: using fallback ResNet-50 with 20-channel input")

        except Exception as exc:
            logger.warning("KPR model load failed: %s", exc)

    def extract(
        self,
        crop: np.ndarray,
        keypoints: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Extract a keypoint-prompted embedding from a person crop.

        Args:
            crop: BGR person crop.
            keypoints: (17, 3) array [x, y, conf] in crop coordinates.
            mask: optional binary mask in crop dimensions.

        Returns:
            2048-d L2-normalized embedding, or None if extraction fails.
        """
        if self._model is None:
            return None

        if mask is not None and mask.shape[:2] == crop.shape[:2]:
            crop = crop.copy()
            crop[~mask] = 0

        h, w = crop.shape[:2]

        # Create keypoint heatmap
        heatmap = self._create_keypoint_heatmap(keypoints, h, w)

        # Resize to standard input size
        target_h, target_w = 256, 128
        rgb = cv2.resize(crop[:, :, ::-1], (target_w, target_h)).astype(np.float32) / 255.0
        heatmap_resized = cv2.resize(heatmap.transpose(1, 2, 0), (target_w, target_h))
        if heatmap_resized.ndim == 2:
            heatmap_resized = heatmap_resized[:, :, np.newaxis]

        import torch
        import torchvision.transforms.functional as F

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        rgb_tensor = F.normalize(rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        heatmap_tensor = torch.from_numpy(heatmap_resized).permute(2, 0, 1).float()

        # Concatenate: 3 RGB + 17 heatmap = 20 channels
        combined = torch.cat([rgb_tensor, heatmap_tensor], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model(combined)

        emb = features.cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def _create_keypoint_heatmap(
        self, keypoints: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Create a K-channel heatmap from keypoints.

        Each channel has a 2D Gaussian at the keypoint location.
        """
        heatmap = np.zeros((self.NUM_KEYPOINTS, h, w), dtype=np.float32)

        for k in range(min(self.NUM_KEYPOINTS, keypoints.shape[0])):
            x, y, conf = keypoints[k]
            if conf < 0.3:
                continue

            x, y = int(x), int(y)
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            sigma = self.HEATMAP_SIGMA
            size = int(3 * sigma)
            x_range = np.arange(max(0, x - size), min(w, x + size + 1))
            y_range = np.arange(max(0, y - size), min(h, y + size + 1))

            if len(x_range) == 0 or len(y_range) == 0:
                continue

            xx, yy = np.meshgrid(x_range, y_range)
            gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            heatmap[k, yy.ravel(), xx.ravel()] = gaussian.ravel()

        return heatmap

    def compare(self, gallery_emb: np.ndarray, query_emb: np.ndarray) -> float:
        """Cosine distance between two embeddings. Lower = more similar."""
        return float(1.0 - np.dot(gallery_emb, query_emb))


def is_kpr_enabled() -> bool:
    v = os.environ.get("SWAY_REID_KPR_ENABLED", "1").strip().lower()
    return v in ("1", "true", "yes", "on")
