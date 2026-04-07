"""
Optional RTMPose-L top-down backend (MMPose). Same output contract as ``PoseEstimator`` (COCO-17).

Install (separate from base ``requirements.txt`` — platform-specific MMCV wheels):

  pip install mmengine "mmcv>=2.0.0" mmpose

Default config/checkpoint resolve from the ``mmpose`` package tree. Override with:

  SWAY_RTMPose_CONFIG     — path to model config .py
  SWAY_RTMPose_CHECKPOINT — path or URL to .pth

MMPose does not use Apple MPS reliably; CUDA or CPU is selected from the pipeline device.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .pose_estimator import BBOX_PADDING

_DEFAULT_CKPT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-l_simcc-body7-pose-aic-coco_270epoch-256x192-cd4c1866_20230228.pth"
)


def _mmpose_install_hint() -> str:
    return (
        "RTMPose requires MMPose (not installed). "
        "Install with: pip install mmengine 'mmcv>=2.0.0' mmpose "
        "(see mmcv install notes for your platform), or see docs/PIPELINE_IMPROVEMENTS_ROADMAP.md."
    )


def _default_config_path() -> Path:
    import mmpose

    root = Path(mmpose.__file__).resolve().parent
    rel = (
        Path("configs")
        / "body_2d_keypoint"
        / "rtmpose"
        / "coco"
        / "rtmpose-l_8xb256-420e_coco-256x192.py"
    )
    return root / rel


def _mmengine_device(device: Union[torch.device, str]) -> str:
    if isinstance(device, str):
        d = torch.device(device)
    else:
        d = device
    if d.type == "cuda":
        idx = d.index if d.index is not None else 0
        return f"cuda:{idx}"
    return "cpu"


class RTMPoseEstimator:
    """Top-down COCO-17 keypoints via ``mmpose.apis.inference_topdown``."""

    def __init__(
        self,
        device: Union[torch.device, str] = "cpu",
        config_path: Optional[Union[str, Path]] = None,
        checkpoint: Optional[str] = None,
    ):
        try:
            from mmpose.apis import inference_topdown, init_model
        except ImportError as e:
            raise RuntimeError(_mmpose_install_hint()) from e

        self._inference_topdown = inference_topdown
        self._init_model = init_model

        cfg = config_path or os.environ.get("SWAY_RTMPose_CONFIG", "").strip()
        if not cfg:
            p = _default_config_path()
            if not p.is_file():
                raise FileNotFoundError(
                    f"RTMPose config not found at {p}. Set SWAY_RTMPose_CONFIG to your config .py."
                )
            cfg = str(p)
        ckpt = (checkpoint or os.environ.get("SWAY_RTMPose_CHECKPOINT", "").strip() or _DEFAULT_CKPT)

        mm_dev = _mmengine_device(device)
        self.device = torch.device(mm_dev)
        self.model = self._init_model(cfg, ckpt, device=mm_dev)
        self.use_fp16 = False

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
        assert len(boxes) == len(track_ids)
        if paddings is None:
            paddings = [BBOX_PADDING] * len(boxes)
        if segmentation_masks is not None and any(m is not None for m in segmentation_masks):
            # ViTPose mask-gated crops are not wired for RTMPose yet; run plain top-down on boxes.
            pass

        img_h, img_w = frame.shape[:2]
        img_bgr = frame[:, :, ::-1].copy()
        bxy = np.zeros((len(boxes), 4), dtype=np.float32)
        for i, (b, pad) in enumerate(zip(boxes, paddings)):
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            pw, ph = pad * w, pad * h
            bxy[i] = [
                max(0.0, x1 - pw),
                max(0.0, y1 - ph),
                min(float(img_w), x2 + pw),
                min(float(img_h), y2 + ph),
            ]

        results = self._inference_topdown(self.model, img_bgr, bxy, bbox_format="xyxy")
        if not results:
            return {}

        def _to_numpy(x):
            return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

        out: Dict[int, Dict] = {}

        if len(results) == len(track_ids):
            for ds, tid in zip(results, track_ids):
                pi = ds.pred_instances
                k2 = _to_numpy(pi.keypoints)
                sc = _to_numpy(
                    getattr(pi, "keypoint_scores", getattr(pi, "keypoints_visible", None))
                )
                if k2.ndim == 3:
                    k2, sc = k2[0], (sc[0] if sc is not None and sc.ndim >= 2 else sc)
                if sc is None:
                    sc = np.ones(17, dtype=np.float32)
                full_kpts = np.zeros((17, 3), dtype=np.float32)
                full_kpts[:, :2] = k2.astype(np.float32)[:17, :2]
                full_kpts[:, 2] = sc.astype(np.float32).reshape(-1)[:17]
                out[int(tid)] = {"keypoints": full_kpts, "scores": full_kpts[:, 2].copy()}
            return out

        inst = results[0].pred_instances
        k_all = _to_numpy(inst.keypoints)
        s_all = _to_numpy(
            getattr(inst, "keypoint_scores", getattr(inst, "keypoints_visible", None))
        )
        if s_all is None:
            s_all = np.ones((k_all.shape[0], 17), dtype=np.float32)

        if k_all.ndim == 2:
            k_all = k_all[np.newaxis, ...]
            s_all = s_all[np.newaxis, ...] if s_all.ndim == 1 else s_all
        n_inst = k_all.shape[0]
        n_use = min(n_inst, len(track_ids))
        for i in range(n_use):
            tid = int(track_ids[i])
            k2 = k_all[i]
            sc = s_all[i] if s_all.ndim >= 2 else s_all
            full_kpts = np.zeros((17, 3), dtype=np.float32)
            full_kpts[:, :2] = k2.astype(np.float32)[:17, :2]
            full_kpts[:, 2] = sc.astype(np.float32).reshape(-1)[:17]
            out[tid] = {"keypoints": full_kpts, "scores": full_kpts[:, 2].copy()}
        return out
