"""
Quality-aware RTMW 384x288 hybrid backend.

Design:
- Primary model: RTMW Cocktail14 384x288 (rtmlib/ONNXRuntime).
- Fallback model: ViTPose (same contract as PoseEstimator) on uncertain tracks.
- Fusion policy: keep RTMW when stable/clear; replace uncertain joints with fallback
  or held previous joints when motion is implausible.

Output contract matches PoseEstimator/RTMPoseEstimator: COCO-17 keypoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .pose_estimator import PoseEstimator


def _rtmlib_install_hint() -> str:
    return (
        "RTMW 384 hybrid backend requires rtmlib (ONNXRuntime). "
        "Install with: pip install rtmlib"
    )


def _expand_xyxy(
    box: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    pad: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= pad * w
    x2 += pad * w
    y1 -= pad * h
    y2 += pad * h
    x1 = max(0.0, min(float(img_w - 1), x1))
    y1 = max(0.0, min(float(img_h - 1), y1))
    x2 = max(x1 + 1.0, min(float(img_w), x2))
    y2 = max(y1 + 1.0, min(float(img_h), y2))
    return (x1, y1, x2, y2)


def _build_sam_constrained_frame(
    frame: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    segmentation_masks: Optional[List[Optional[np.ndarray]]],
) -> np.ndarray:
    """
    Build a frame where only SAM-approved person pixels are visible.
    Pixels outside all provided masks are set to neutral gray.
    """
    if not segmentation_masks or all(m is None for m in segmentation_masks):
        return frame

    h, w = frame.shape[:2]
    constrained = np.full_like(frame, 114)
    for box, mask in zip(boxes, segmentation_masks):
        if mask is None or mask.size == 0:
            continue
        x1, y1, x2, y2 = (
            int(round(float(box[0]))),
            int(round(float(box[1]))),
            int(round(float(box[2]))),
            int(round(float(box[3]))),
        )
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        roi_h = y2 - y1
        roi_w = x2 - x1
        if roi_h <= 0 or roi_w <= 0:
            continue

        m = np.asarray(mask).astype(bool)
        if m.shape[0] != roi_h or m.shape[1] != roi_w:
            m = cv2.resize(
                m.astype(np.uint8),
                (roi_w, roi_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        src_roi = frame[y1:y2, x1:x2]
        dst_roi = constrained[y1:y2, x1:x2]
        dst_roi[m] = src_roi[m]
    return constrained


def _bbox_diag(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return float(max(1.0, np.hypot(x2 - x1, y2 - y1)))


def _inside_box(
    x: float,
    y: float,
    box: Tuple[float, float, float, float],
    margin_frac: float = 0.1,
) -> bool:
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    mx = margin_frac * w
    my = margin_frac * h
    return (x1 - mx) <= x <= (x2 + mx) and (y1 - my) <= y <= (y2 + my)


class _QualityGate:
    """Joint-level certainty gate + fallback/hold fusion."""

    SENSITIVE = {0, 1, 2, 3, 4, 5, 6}  # face + shoulders in COCO-17

    def __init__(
        self,
        conf_accept: float = 0.42,
        conf_low: float = 0.28,
        jump_frac: float = 0.11,
        sensitive_jump_frac: float = 0.08,
        jump_min_px: float = 14.0,
        jump_max_px: float = 56.0,
    ):
        self.conf_accept = float(conf_accept)
        self.conf_low = float(conf_low)
        self.jump_frac = float(jump_frac)
        self.sensitive_jump_frac = float(sensitive_jump_frac)
        self.jump_min_px = float(jump_min_px)
        self.jump_max_px = float(jump_max_px)
        self.prev_final_by_tid: Dict[int, np.ndarray] = {}
        self.prev_prev_final_by_tid: Dict[int, np.ndarray] = {}

    def _jump_thr(self, diag: float, sensitive: bool) -> float:
        frac = self.sensitive_jump_frac if sensitive else self.jump_frac
        return float(np.clip(frac * diag, self.jump_min_px, self.jump_max_px))

    def _track_needs_fallback(
        self,
        tid: int,
        box: Tuple[float, float, float, float],
        rtmw_kpts: np.ndarray,
    ) -> bool:
        mean_conf = float(np.mean(rtmw_kpts[:, 2])) if rtmw_kpts.size else 0.0
        sens_conf = float(np.mean(rtmw_kpts[list(self.SENSITIVE), 2]))
        if mean_conf < 0.52 or sens_conf < 0.45:
            return True

        prev = self.prev_final_by_tid.get(int(tid))
        if prev is None or prev.shape != rtmw_kpts.shape:
            return False
        diag = _bbox_diag(box)
        # Trigger fallback if any sensitive joint jumps implausibly with low confidence.
        for j in self.SENSITIVE:
            c = float(rtmw_kpts[j, 2])
            d = float(np.linalg.norm(rtmw_kpts[j, :2] - prev[j, :2]))
            if c < self.conf_accept and d > self._jump_thr(diag, sensitive=True):
                return True
        return False

    def fuse_track(
        self,
        tid: int,
        box: Tuple[float, float, float, float],
        rtmw_kpts: np.ndarray,
        fallback_kpts: Optional[np.ndarray],
    ) -> np.ndarray:
        out = rtmw_kpts.copy()
        prev = self.prev_final_by_tid.get(int(tid))
        prev_prev = self.prev_prev_final_by_tid.get(int(tid))
        diag = _bbox_diag(box)

        for j in range(min(17, out.shape[0])):
            x, y, c = float(out[j, 0]), float(out[j, 1]), float(out[j, 2])
            sensitive = j in self.SENSITIVE
            jump_thr = self._jump_thr(diag, sensitive=sensitive)

            low_conf = c < self.conf_low
            outside = (not _inside_box(x, y, box)) and c < 0.60
            jumpy = False
            pred_deviation = False

            if prev is not None and prev.shape == out.shape:
                d_prev = float(np.linalg.norm(out[j, :2] - prev[j, :2]))
                if c < self.conf_accept and d_prev > jump_thr:
                    jumpy = True
                if (
                    prev_prev is not None
                    and prev_prev.shape == out.shape
                    and c < 0.60
                ):
                    pred_xy = prev[j, :2] + (prev[j, :2] - prev_prev[j, :2])
                    d_pred = float(np.linalg.norm(out[j, :2] - pred_xy))
                    if d_pred > (1.25 * jump_thr):
                        pred_deviation = True

            uncertain = low_conf or outside or jumpy or pred_deviation
            if not uncertain:
                continue

            # Priority: fallback (if good) -> previous stable output -> keep RTMW as last resort.
            used = False
            if fallback_kpts is not None and fallback_kpts.shape == out.shape:
                fx, fy, fc = (
                    float(fallback_kpts[j, 0]),
                    float(fallback_kpts[j, 1]),
                    float(fallback_kpts[j, 2]),
                )
                if fc >= max(c, 0.34):
                    out[j, 0], out[j, 1], out[j, 2] = fx, fy, fc
                    used = True
            if not used and prev is not None and prev.shape == out.shape:
                out[j, 0], out[j, 1] = float(prev[j, 0]), float(prev[j, 1])
                out[j, 2] = max(float(prev[j, 2]) * 0.93, min(c, 0.55))

        if prev is not None and prev.shape == out.shape:
            self.prev_prev_final_by_tid[int(tid)] = prev.copy()
        self.prev_final_by_tid[int(tid)] = out.copy()
        return out


class RTMW384HybridEstimator:
    """RTMW 384x288 with certainty gate and ViTPose fallback on uncertain data."""

    def __init__(
        self,
        device: Union[torch.device, str] = "cpu",
        model_path: Optional[Union[str, Path]] = None,
        fallback_model_name: Optional[str] = None,
    ):
        try:
            from rtmlib import RTMPose
        except ImportError as e:
            raise RuntimeError(_rtmlib_install_hint()) from e

        cache_root = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"
        default_local = cache_root / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
        env_model = os.environ.get("SWAY_RTMW_384_ONNX", "").strip()
        model_ref: Union[str, Path]
        if model_path is not None:
            model_ref = model_path
        elif env_model:
            model_ref = env_model
        elif default_local.exists():
            model_ref = default_local
        else:
            model_ref = (
                "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
                "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
            )

        self.pose_model = RTMPose(
            str(model_ref),
            model_input_size=(288, 384),
            backend="onnxruntime",
            device="cpu",
        )
        self.use_fp16 = False
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.pad_frac = float(os.environ.get("SWAY_RTMW_384_PAD_FRAC", "0.18") or 0.18)
        fb_name = (
            fallback_model_name
            or os.environ.get("SWAY_RTMW_HYBRID_FALLBACK_VITPOSE", "").strip()
            or "usyd-community/vitpose-plus-huge"
        )
        self.fallback_estimator = PoseEstimator(device=self.device, model_name=fb_name)
        self._gate = _QualityGate()

    def estimate_poses(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        track_ids: List[int],
        paddings: Optional[List[float]] = None,
        segmentation_masks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Dict[int, Dict]:
        if len(boxes) == 0:
            return {}
        assert len(boxes) == len(track_ids), "boxes and track_ids must be 1:1"
        _ = paddings
        frame_for_pose = _build_sam_constrained_frame(frame, boxes, segmentation_masks)

        img_h, img_w = frame.shape[:2]
        expanded = [_expand_xyxy(b, img_w, img_h, self.pad_frac) for b in boxes]
        bxy = np.asarray(expanded, dtype=np.float32)
        keypoints_xy, scores = self.pose_model(frame_for_pose, bboxes=bxy)

        rtmw_out: Dict[int, np.ndarray] = {}
        rtmw_out_full: Dict[int, np.ndarray] = {}
        needs_fallback_tids: List[int] = []
        needs_fallback_boxes: List[Tuple[float, float, float, float]] = []

        n = min(len(track_ids), len(keypoints_xy), len(scores))
        for i in range(n):
            tid = int(track_ids[i])
            kxy = np.asarray(keypoints_xy[i], dtype=np.float32)
            sc = np.asarray(scores[i], dtype=np.float32).reshape(-1)
            # RTMW SimCC score scale is typically >1; map to a [0,1] confidence-like band.
            sc = 1.0 / (1.0 + np.exp(-((sc - 4.0) / 0.9)))
            sc = np.clip(sc, 0.0, 1.0).astype(np.float32)

            m_full = min(kxy.shape[0], sc.shape[0])
            if m_full > 0:
                kpts_full = np.zeros((m_full, 3), dtype=np.float32)
                kpts_full[:m_full, :2] = kxy[:m_full, :2]
                kpts_full[:m_full, 2] = sc[:m_full]
            else:
                kpts_full = np.zeros((0, 3), dtype=np.float32)
            rtmw_out_full[tid] = kpts_full

            m = min(17, kxy.shape[0], sc.shape[0])
            kpts17 = np.zeros((17, 3), dtype=np.float32)
            kpts17[:m, :2] = kxy[:m, :2]
            kpts17[:m, 2] = sc[:m]
            rtmw_out[tid] = kpts17

            if self._gate._track_needs_fallback(tid, boxes[i], kpts17):
                needs_fallback_tids.append(tid)
                needs_fallback_boxes.append(boxes[i])

        fallback_out_raw: Dict[int, Dict] = {}
        if needs_fallback_tids:
            fallback_masks: Optional[List[Optional[np.ndarray]]] = None
            if segmentation_masks is not None:
                fallback_masks = []
                for tid_fb in needs_fallback_tids:
                    try:
                        src_idx = track_ids.index(tid_fb)
                    except ValueError:
                        fallback_masks.append(None)
                        continue
                    fallback_masks.append(segmentation_masks[src_idx])
            fallback_out_raw = self.fallback_estimator.estimate_poses(
                frame=frame_for_pose,
                boxes=needs_fallback_boxes,
                track_ids=needs_fallback_tids,
                paddings=None,
                segmentation_masks=fallback_masks,
            )

        out: Dict[int, Dict] = {}
        for i in range(n):
            tid = int(track_ids[i])
            rtmw_kpts = rtmw_out.get(tid)
            if rtmw_kpts is None:
                continue
            fb = fallback_out_raw.get(tid)
            fb_kpts = None
            if fb is not None and "keypoints" in fb:
                fb_kpts = np.asarray(fb["keypoints"], dtype=np.float32)
                if fb_kpts.shape != (17, 3):
                    fb_kpts = None
            fused = self._gate.fuse_track(
                tid=tid,
                box=boxes[i],
                rtmw_kpts=rtmw_kpts,
                fallback_kpts=fb_kpts,
            )
            full_kpts = rtmw_out_full.get(tid, np.zeros((0, 3), dtype=np.float32))
            out[tid] = {
                "keypoints": fused,
                "scores": fused[:, 2].copy(),
                "keypoints_full": full_kpts,
                "scores_full": full_kpts[:, 2].copy() if full_kpts.size else np.zeros((0,), dtype=np.float32),
                "joint_count_full": int(full_kpts.shape[0]),
            }
        return out
