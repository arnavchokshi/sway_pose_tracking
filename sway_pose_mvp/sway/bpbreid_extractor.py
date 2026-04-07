"""
BPBreID Part-Based Re-ID Extractor (PLAN_08)

Replaces OSNet's single global embedding with separate embeddings for six body
regions: head, torso, upper arms, lower arms, upper legs, lower legs.
When matching a partially visible person, only visible parts are compared.

Trained with adversarial occlusion (GiLt) — each part embedding is discriminative alone.

STRICT MODE (default): No fallback paths. torchreid + BPBreID checkpoint must load
or the pipeline hard-fails at startup.

Env:
  SWAY_REID_PART_MODEL      – bpbreid | paformer | osnet_x0_25 (default bpbreid)
  SWAY_REID_PART_MIN_VISIBLE – min shared visible parts for part-based comparison (default 3)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

PART_KEYPOINT_MAP = {
    "head": [_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR],
    "torso": [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP],
    "upper_arms": [_L_SHOULDER, _R_SHOULDER, _L_ELBOW, _R_ELBOW],
    "lower_arms": [_L_ELBOW, _R_ELBOW, _L_WRIST, _R_WRIST],
    "upper_legs": [_L_HIP, _R_HIP, _L_KNEE, _R_KNEE],
    "lower_legs": [_L_KNEE, _R_KNEE, _L_ANKLE, _R_ANKLE],
}

PART_VERTICAL_FRACTIONS = {
    "head": (0.0, 1 / 6),
    "torso": (1 / 6, 3 / 6),
    "upper_arms": (1 / 6, 3 / 6),
    "lower_arms": (2 / 6, 3 / 6),
    "upper_legs": (3 / 6, 5 / 6),
    "lower_legs": (5 / 6, 1.0),
}

EXPECTED_EMBEDDING_DIM = 2048


@dataclass
class PartEmbeddings:
    """Container for per-part + global embeddings with visibility flags."""
    global_emb: np.ndarray
    part_embs: Dict[str, np.ndarray] = field(default_factory=dict)
    visibility: Dict[str, bool] = field(default_factory=dict)


class PartReIDExtractor(ABC):
    """Interface for all part-based re-ID extractors."""

    @abstractmethod
    def extract(
        self, crop: np.ndarray, keypoints: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> PartEmbeddings:
        ...

    @abstractmethod
    def compare(self, gallery: PartEmbeddings, query: PartEmbeddings) -> float:
        ...


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


class BPBreIDExtractor(PartReIDExtractor):
    """Part-based re-ID using BPBreID with torchreid backbone.

    NO FALLBACK PATHS. Requires:
    1. torchreid importable
    2. BPBreID checkpoint loadable
    3. Model produces EXPECTED_EMBEDDING_DIM-d embeddings
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        self._torchreid_device = "cuda" if str(device).startswith("cuda") else "cpu"
        self._model = None
        self._backbone = None
        self.reid_feature_mode = "unknown"
        self.reid_feature_mode_reason = ""
        self.embedding_dim = EXPECTED_EMBEDDING_DIM

        fp16_env = os.environ.get("SWAY_BPBREID_FP16", "0").strip()
        self._use_fp16 = (
            fp16_env == "1"
            and self._torchreid_device in ("cuda", "mps")
        )

        if checkpoint_path is None:
            models_dir = Path(__file__).resolve().parent.parent / "models"
            checkpoint_path = str(models_dir / "bpbreid_r50_market_msmt17.pth")

        self._checkpoint_path = checkpoint_path
        self._load_model()

    def is_fallback(self) -> bool:
        return self.reid_feature_mode != "torchreid"

    def assert_not_fallback(self) -> None:
        if self.is_fallback():
            raise RuntimeError(
                "BPBreID strict mode violation: expected torchreid feature path, "
                f"got {self.reid_feature_mode} ({self.reid_feature_mode_reason or 'no reason'})"
            )

    def _load_model(self) -> None:
        """Load BPBreID via torchreid with the exact checkpoint. Hard-fail on any error."""
        import torch

        ckpt_path = Path(self._checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"BPBreID checkpoint not found: {ckpt_path}. "
                "Place bpbreid_r50_market_msmt17.pth in the models/ directory."
            )

        ckpt = torch.load(str(ckpt_path), map_location="cpu", encoding="latin1")
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            raise ValueError(
                f"BPBreID checkpoint has unexpected format: type={type(ckpt)}, "
                f"keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
            )
        sd = ckpt["state_dict"]
        if "classifier.weight" not in sd:
            raise ValueError("BPBreID checkpoint missing classifier.weight key")
        num_classes = sd["classifier.weight"].shape[0]
        emb_dim = sd["classifier.weight"].shape[1]
        if emb_dim != EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"BPBreID checkpoint embedding dim={emb_dim}, expected {EXPECTED_EMBEDDING_DIM}"
            )

        try:
            from torchreid.models.resnet import ResNet, Bottleneck
        except ImportError as exc:
            raise ImportError(
                "torchreid is required for BPBreID. Install via: "
                "pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            ) from exc

        model = ResNet(
            num_classes=num_classes,
            loss="softmax",
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=1,
            fc_dims=None,
            dropout_p=None,
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        backbone_missing = [k for k in missing if not k.startswith("classifier") and not k.startswith("fc")]
        if backbone_missing:
            raise RuntimeError(
                f"BPBreID backbone weights failed to load: {len(backbone_missing)} missing keys: "
                f"{backbone_missing[:10]}"
            )
        logger.info(
            "BPBreID weight load: missing=%d (classifier-only OK), unexpected=%d, last_stride=1",
            len(missing), len(unexpected),
        )

        model.eval()
        model.to(self._torchreid_device)
        if self._use_fp16:
            model.half()
            logger.info("BPBreID: FP16 enabled on %s", self._torchreid_device)
        self._backbone = model

        dtype = torch.float16 if self._use_fp16 else torch.float32
        dummy = torch.randn(1, 3, 256, 128, dtype=dtype).to(self._torchreid_device)
        with torch.no_grad():
            out = model(dummy)
        out_dim = out.shape[-1]
        if out_dim != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                f"BPBreID forward pass produces {out_dim}-d embeddings, "
                f"expected {EXPECTED_EMBEDDING_DIM}"
            )

        self.reid_feature_mode = "torchreid"
        self.reid_feature_mode_reason = (
            f"torchreid resnet50 + BPBreID checkpoint loaded "
            f"(num_classes={num_classes}, emb_dim={EXPECTED_EMBEDDING_DIM})"
        )
        self.embedding_dim = EXPECTED_EMBEDDING_DIM
        logger.info(
            "BPBreID loaded: %s, checkpoint=%s",
            self.reid_feature_mode_reason,
            ckpt_path.name,
        )

    def extract(
        self,
        crop: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> PartEmbeddings:
        if mask is not None and mask.shape[:2] == crop.shape[:2]:
            crop = crop.copy()
            crop[~mask] = 0

        h, w = crop.shape[:2]
        part_regions = self._compute_part_regions(h, w, keypoints)
        visibility = self._compute_visibility(keypoints)
        resized = cv2.resize(crop, (128, 256))

        part_names_ordered: List[str] = []
        part_crops: List[np.ndarray] = []
        for part_name, (y1, y2) in part_regions.items():
            py1 = int(y1 / h * 256)
            py2 = int(y2 / h * 256)
            py2 = max(py2, py1 + 8)
            pc = resized[py1:py2, :, :]
            if pc.size > 0:
                part_names_ordered.append(part_name)
                part_crops.append(cv2.resize(pc, (128, 64)))

        global_embs = self._extract_embeddings_batch([resized])
        global_emb = global_embs[0]

        part_embs: Dict[str, np.ndarray] = {}
        if part_crops:
            raw_part_embs = self._extract_embeddings_batch(part_crops)
            for name, emb in zip(part_names_ordered, raw_part_embs):
                part_embs[name] = emb / (np.linalg.norm(emb) + 1e-8)

        global_emb = global_emb / (np.linalg.norm(global_emb) + 1e-8)
        return PartEmbeddings(global_emb=global_emb, part_embs=part_embs, visibility=visibility)

    def extract_batch(
        self,
        crops: List[np.ndarray],
        keypoints_list: List[Optional[np.ndarray]],
        masks_list: List[Optional[np.ndarray]],
    ) -> List[PartEmbeddings]:
        """Extract embeddings for multiple people with minimal forward passes.

        Collects all global crops into one batch and all part crops into another,
        then runs just 2 forward passes total regardless of person count.
        """
        if not crops:
            return []

        n = len(crops)
        if len(keypoints_list) != n or len(masks_list) != n:
            raise ValueError(
                f"Length mismatch: crops={n}, keypoints={len(keypoints_list)}, "
                f"masks={len(masks_list)}"
            )

        global_images: List[np.ndarray] = []
        per_person_part_info: List[List[Tuple[str, int]]] = []
        all_part_images: List[np.ndarray] = []
        all_visibilities: List[Dict[str, bool]] = []

        for i in range(n):
            crop = crops[i]
            kp = keypoints_list[i]
            msk = masks_list[i]

            if msk is not None and msk.shape[:2] == crop.shape[:2]:
                crop = crop.copy()
                crop[~msk] = 0

            h, w = crop.shape[:2]
            resized = cv2.resize(crop, (128, 256))
            global_images.append(resized)

            part_regions = self._compute_part_regions(h, w, kp)
            all_visibilities.append(self._compute_visibility(kp))

            person_parts: List[Tuple[str, int]] = []
            for part_name, (y1, y2) in part_regions.items():
                py1 = int(y1 / h * 256)
                py2 = int(y2 / h * 256)
                py2 = max(py2, py1 + 8)
                pc = resized[py1:py2, :, :]
                if pc.size > 0:
                    person_parts.append((part_name, len(all_part_images)))
                    all_part_images.append(cv2.resize(pc, (128, 64)))
            per_person_part_info.append(person_parts)

        global_embs_raw = self._extract_embeddings_batch(global_images)

        part_embs_raw: List[np.ndarray] = []
        if all_part_images:
            part_embs_raw = self._extract_embeddings_batch(all_part_images)

        results: List[PartEmbeddings] = []
        for i in range(n):
            g = global_embs_raw[i]
            g = g / (np.linalg.norm(g) + 1e-8)

            part_embs: Dict[str, np.ndarray] = {}
            for part_name, idx in per_person_part_info[i]:
                emb = part_embs_raw[idx]
                part_embs[part_name] = emb / (np.linalg.norm(emb) + 1e-8)

            results.append(PartEmbeddings(
                global_emb=g, part_embs=part_embs, visibility=all_visibilities[i],
            ))

        return results

    def _extract_embedding(self, img: np.ndarray) -> np.ndarray:
        """Run backbone forward pass. Hard-fails if backbone is not loaded."""
        if self._backbone is None:
            raise RuntimeError("BPBreID backbone not loaded -- this should never happen in strict mode")

        import torch
        import torchvision.transforms.functional as F

        tensor = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor = tensor.unsqueeze(0).to(self._torchreid_device)

        with torch.no_grad():
            features = self._backbone(tensor)

        return features.cpu().numpy().flatten()

    def _extract_embeddings_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Run a single batched forward pass for multiple pre-resized BGR images."""
        if not images:
            return []
        if self._backbone is None:
            raise RuntimeError("BPBreID backbone not loaded -- this should never happen in strict mode")

        import torch
        import torchvision.transforms.functional as F

        _mean = [0.485, 0.456, 0.406]
        _std = [0.229, 0.224, 0.225]

        tensors = []
        for img in images:
            t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            t = F.normalize(t, mean=_mean, std=_std)
            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self._torchreid_device)
        if self._use_fp16:
            batch = batch.half()

        with torch.no_grad():
            features = self._backbone(batch)

        features_np = features.cpu().float().numpy()
        return [features_np[i].flatten() for i in range(features_np.shape[0])]

    def _compute_part_regions(
        self, h: int, w: int, keypoints: Optional[np.ndarray]
    ) -> Dict[str, Tuple[int, int]]:
        if keypoints is not None and keypoints.shape[0] >= 17:
            regions: Dict[str, Tuple[int, int]] = {}
            for part_name, kp_ids in PART_KEYPOINT_MAP.items():
                ys = []
                for kid in kp_ids:
                    if keypoints[kid, 2] > 0.3:
                        ys.append(keypoints[kid, 1])
                if len(ys) >= 1:
                    y1 = max(0, int(min(ys) - 10))
                    y2 = min(h, int(max(ys) + 10))
                    regions[part_name] = (y1, y2)
                else:
                    frac = PART_VERTICAL_FRACTIONS[part_name]
                    regions[part_name] = (int(frac[0] * h), int(frac[1] * h))
            return regions
        else:
            return {
                name: (int(frac[0] * h), int(frac[1] * h))
                for name, frac in PART_VERTICAL_FRACTIONS.items()
            }

    def _compute_visibility(self, keypoints: Optional[np.ndarray]) -> Dict[str, bool]:
        visibility: Dict[str, bool] = {}
        for part_name, kp_ids in PART_KEYPOINT_MAP.items():
            if keypoints is not None and keypoints.shape[0] >= 17:
                visible = any(keypoints[kid, 2] > 0.3 for kid in kp_ids)
            else:
                visible = True
            visibility[part_name] = visible
        return visibility

    def compare(self, gallery: PartEmbeddings, query: PartEmbeddings) -> float:
        min_visible = _env_int("SWAY_REID_PART_MIN_VISIBLE", 3)
        shared_parts = [
            name for name in gallery.part_embs
            if name in query.part_embs
            and gallery.visibility.get(name, False)
            and query.visibility.get(name, False)
        ]
        if len(shared_parts) < min_visible:
            return float(1.0 - np.dot(gallery.global_emb, query.global_emb))
        distances = []
        for name in shared_parts:
            g = gallery.part_embs[name]
            q = query.part_embs[name]
            dist = 1.0 - np.dot(g, q)
            distances.append(dist)
        return float(np.mean(distances))
