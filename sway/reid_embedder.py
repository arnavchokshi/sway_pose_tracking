"""
V3.8: Lightweight Appearance Embedder for Re-ID

Extracts feature embeddings from bbox crops to support appearance-based tracking.
Used when deciding ID assignments during occlusion (red vs blue dancer disambiguation).

- HSV color histogram: captures clothing color/texture (zero extra deps).
- Optional ResNet18 backbone via torchvision (when available).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Optional torch for ResNet backbone
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet18
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _crop_bbox(
    frame_bgr: np.ndarray,
    box: Tuple[float, float, float, float],
    padding: float = 0.05,
) -> np.ndarray:
    """Crop frame to bbox with optional padding. Returns BGR crop."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * padding
    pad_h = bh * padding
    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(w, int(x2 + pad_w))
    y2 = min(h, int(y2 + pad_h))
    return frame_bgr[y1:y2, x1:x2]


# V3.9: Per-strip minimum height; below this, fall back to global HSV
HSV_STRIP_MIN_HEIGHT = 15


def extract_hsv_histogram(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Extract normalized HSV histogram (H=8, S=4, V=4) = 128 dims.
    Captures clothing color; L2-normalized for cosine similarity.
    """
    if crop_bgr.size == 0:
        return np.zeros(128, dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [8, 4, 4],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 1e-6:
        hist = hist / norm
    return hist


def _extract_hsv_strip_histogram(crop_bgr: np.ndarray, strip_idx: int) -> np.ndarray:
    """
    Extract HSV histogram for one vertical third of crop. strip_idx in {0,1,2} = head, torso, legs.
    Same bins (8,4,4) = 128 dims. L2-normalized.
    """
    if crop_bgr.size == 0:
        return np.zeros(128, dtype=np.float32)
    h, w = crop_bgr.shape[:2]
    third = h // 3
    y0 = strip_idx * third
    y1 = (strip_idx + 1) * third if strip_idx < 2 else h
    strip_crop = crop_bgr[y0:y1, :]
    if strip_crop.size == 0:
        return np.zeros(128, dtype=np.float32)
    return extract_hsv_histogram(strip_crop)


def extract_hsv_strip_embeddings(crop_bgr: np.ndarray) -> np.ndarray:
    """
    V3.9: 3-strip HSV — head, torso, legs. Distinguishes red-shirt/black-pants
    from black-shirt/red-pants. Returns 384 dims (3 x 128), L2-normalized.
    Fallback: if any strip height < 15px, use global 128-d histogram.
    """
    if crop_bgr.size == 0:
        return np.zeros(128, dtype=np.float32)
    h = crop_bgr.shape[0]
    strip_h = h // 3
    if strip_h < HSV_STRIP_MIN_HEIGHT:
        return extract_hsv_histogram(crop_bgr)
    parts = []
    for i in range(3):
        part = _extract_hsv_strip_histogram(crop_bgr, i)
        parts.append(part)
    emb = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm
    return emb


def extract_embeddings(
    frame_bgr: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    track_ids: List[int],
    method: str = "hsv",
) -> Dict[int, np.ndarray]:
    """
    Extract appearance embeddings for each (box, track_id).

    Args:
        frame_bgr: BGR frame (e.g. from cv2)
        boxes: List of (x1,y1,x2,y2)
        track_ids: List of track IDs (one per box)
        method: "hsv" (default), "hsv_strip" (3-strip, 384d), or "resnet" (if torch available)

    Returns:
        Dict[track_id, embedding] — each embedding is L2-normalized.
    """
    out: Dict[int, np.ndarray] = {}
    for box, tid in zip(boxes, track_ids):
        crop = _crop_bbox(frame_bgr, box)
        if crop.size == 0:
            continue
        if method == "hsv":
            emb = extract_hsv_histogram(crop)
        elif method == "hsv_strip":
            emb = extract_hsv_strip_embeddings(crop)
        elif method == "resnet" and _TORCH_AVAILABLE:
            emb = _extract_resnet_embedding(crop)
        else:
            emb = extract_hsv_histogram(crop)
        out[tid] = emb
    return out


_resnet_model = None


def _extract_resnet_embedding(crop_bgr: np.ndarray, size: int = 128) -> np.ndarray:
    """Extract 512-d ResNet18 features (pooled). Requires torch/torchvision."""
    global _resnet_model
    if not _TORCH_AVAILABLE:
        return extract_hsv_histogram(crop_bgr)
    import cv2
    if _resnet_model is None:
        model = resnet18(weights="IMAGENET1K_V1")
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        _resnet_model = model
    # Preprocess: BGR -> RGB, resize to 224x224
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = (rgb - mean) / std
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        feat = _resnet_model(x)
    emb = feat.squeeze().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity [−1, 1]. 1 = identical, 0 = orthogonal, −1 = opposite."""
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 − similarity. 0 = identical, 1 = orthogonal."""
    return 1.0 - cosine_similarity(a, b)
