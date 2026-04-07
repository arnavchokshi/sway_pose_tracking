"""Optional Depth Anything V2 helper to auto-estimate stage polygon for pre-pose pruning."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_depth_pipe = None


def _depth_pipeline():
    global _depth_pipe
    if _depth_pipe is not None:
        return _depth_pipe
    try:
        import torch
        from transformers import pipeline
    except ImportError:
        return None
    device: Union[int, str] = -1
    try:
        if torch.cuda.is_available():
            device = 0
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        device = -1
    try:
        _depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )
    except Exception as e:
        logger.debug("Depth Anything pipeline unavailable: %s", e)
        _depth_pipe = None
    return _depth_pipe


def _depth_to_numpy(pipe, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    from PIL import Image

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    max_side = 768
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(rgb)
    out = pipe(pil)
    depth_obj = out["depth"] if isinstance(out, dict) else out
    d = np.asarray(depth_obj, dtype=np.float32)
    if d.size == 0:
        return None
    if d.ndim > 2:
        d = d.squeeze()
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return d


def _mask_from_depth(d: np.ndarray) -> Optional[np.ndarray]:
    h, w = d.shape[:2]
    dn = (d - float(d.min())) / (float(d.max()) - float(d.min()) + 1e-8)
    dn = cv2.GaussianBlur(dn, (5, 5), 0)
    y0 = max(1, int(h * 0.18))
    sub_u8 = np.clip(dn[y0:, :] * 255.0, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(sub_u8, (9, 9), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = (255 - binary).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def refine(mask_sub: np.ndarray) -> np.ndarray:
        m = cv2.morphologyEx(mask_sub, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
        full = np.zeros((h, w), dtype=np.uint8)
        full[y0:, :] = m
        return full

    def score(mask_full: np.ndarray) -> float:
        if mask_full.sum() < (h * w * 0.02 * 255):
            return -1.0
        touch = 1.0 if mask_full[h - 1, :].max() > 0 else 0.0
        width_at_bottom = float((mask_full[h - 1, :] > 0).sum())
        area = float((mask_full > 0).sum())
        return touch * 1e6 + width_at_bottom * 100.0 + area

    best_m = None
    best_s = -1.0
    for cand in (refine(binary), refine(inv)):
        s = score(cand)
        if s > best_s:
            best_s = s
            best_m = cand
    if best_m is None or best_s < 0:
        return None
    return best_m


def _contour_to_polygon_norm(contour: np.ndarray, w: int, h: int) -> List[Tuple[float, float]]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
    if len(approx) < 3:
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    pts = approx.reshape(-1, 2).astype(np.float32)
    poly: List[Tuple[float, float]] = []
    for x, y in pts:
        poly.append((float(np.clip(x / w, 0.0, 1.0)), float(np.clip(y / h, 0.0, 1.0))))
    return poly


def collect_strided_depth_series(
    indexed_frames: Iterable[Tuple[int, np.ndarray]],
    stride_frames: int,
) -> List[Tuple[int, np.ndarray]]:
    """Run Depth Anything on every stride_frames-th frame; return (frame_idx, normalized depth H×W).

    stride_frames must be >= 1. Skips frames when depth pipeline is unavailable.
    """
    if stride_frames < 1:
        stride_frames = 1
    out: List[Tuple[int, np.ndarray]] = []
    for fi, fr in indexed_frames:
        if int(fi) % stride_frames != 0:
            continue
        d = get_depth_array(fr)
        if d is not None:
            out.append((int(fi), d))
    return out


def get_depth_array(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return depth map (H, W) float32 from Depth Anything V2, or None if unavailable.

    Values are per-frame min–max normalized to approximately [0, 1] (relative within the
    frame, not metric depth). For unified 3D world export, root Z from this map is opt-in
    via ``SWAY_DEPTH_FOR_ROOT_Z`` (see ``sway.pose_lift_3d``); default pelvis depth uses
    ``SWAY_DEFAULT_ROOT_Z``.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    pipe = _depth_pipeline()
    if pipe is None:
        return None
    try:
        d = _depth_to_numpy(pipe, frame_bgr)
        if d is None:
            return None
        dn = (d - float(d.min())) / (float(d.max()) - float(d.min()) + 1e-8)
        return dn.astype(np.float32)
    except Exception as e:
        logger.debug("get_depth_array failed: %s", e)
        return None


def estimate_stage_polygon(frame_bgr: np.ndarray) -> Optional[List[Tuple[float, float]]]:
    """Estimate normalized stage polygon from a BGR frame using Depth Anything V2.

    Returns None if the model is unavailable or heuristics fail.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    pipe = _depth_pipeline()
    if pipe is None:
        return None
    try:
        d = _depth_to_numpy(pipe, frame_bgr)
        if d is None:
            return None
        h, w = d.shape[:2]
        mask = _mask_from_depth(d)
        if mask is None:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < (h * w * 0.03):
            return None
        return _contour_to_polygon_norm(cnt, w, h)
    except Exception as e:
        logger.debug("estimate_stage_polygon failed: %s", e)
        return None
