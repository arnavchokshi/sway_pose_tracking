"""
Hybrid YOLO + DETR Detection Dispatcher (PLAN_03)

Runs YOLO on every frame as a cheap first pass, then conditionally invokes
Co-DETR/RT-DETR only on frames with overlapping detections. Avoids paying
the 3-5x DETR cost on every frame while getting NMS-free precision exactly
when it matters (during occlusion events).

Env:
  SWAY_DETECTOR_HYBRID             – 0|1 (default 1)
  SWAY_HYBRID_OVERLAP_IOU_TRIGGER  – IoU above which → run DETR (default 0.30)
  SWAY_HYBRID_COOLDOWN_FRAMES      – skip DETR for N frames after it fires (default 5)
  SWAY_DETECTION_UNCERTAIN_CONF    – detections below this are flagged uncertain (default 0.50)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
from torchvision.ops import box_iou
import torch

logger = logging.getLogger(__name__)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


class HybridDetector:
    """Smart detection dispatcher: YOLO scout + DETR precision fallback.

    detect() returns the same contract as other detectors: List[Detection].
    The last backend used is exposed as ``_last_detect_source`` (``"yolo"`` or ``"detr"``)
    for logging and future-pipeline metrics.
    """

    def __init__(
        self,
        yolo_detector,
        detr_detector,
        overlap_iou_trigger: Optional[float] = None,
        cooldown_frames: Optional[int] = None,
    ):
        self.yolo = yolo_detector
        self.detr = detr_detector
        self.overlap_iou_trigger = overlap_iou_trigger or _env_float(
            "SWAY_HYBRID_OVERLAP_IOU_TRIGGER", 0.30
        )
        self.cooldown_frames = cooldown_frames or _env_int(
            "SWAY_HYBRID_COOLDOWN_FRAMES", 5
        )

        # Internal state
        self._last_detr_frame: int = -999
        self._force_detr_next: bool = False
        self._last_detect_source: str = "unknown"

        # Statistics
        self._yolo_count: int = 0
        self._detr_count: int = 0

    def request_detr_next_frame(self) -> None:
        """External trigger (e.g. SAM2 low-confidence callback) to force DETR."""
        self._force_detr_next = True

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list:
        """Run detection with smart YOLO/DETR dispatch; returns detections only (DetectorProtocol)."""
        yolo_dets = self.yolo.detect(frame)

        if not yolo_dets:
            self._yolo_count += 1
            self._last_detect_source = "yolo"
            return yolo_dets

        needs_detr = self._force_detr_next
        self._force_detr_next = False

        if not needs_detr and len(yolo_dets) >= 2:
            needs_detr = self._check_overlap(yolo_dets)

        in_cooldown = (frame_idx - self._last_detr_frame) < self.cooldown_frames
        if needs_detr and not in_cooldown and self.detr is not None:
            detr_dets = self.detr.detect(frame)
            if detr_dets:
                self._last_detr_frame = frame_idx
                self._detr_count += 1
                self._last_detect_source = "detr"
                return detr_dets

        self._yolo_count += 1
        self._last_detect_source = "yolo"
        return yolo_dets

    def _check_overlap(self, detections: list) -> bool:
        """Check if any pair of detection boxes exceeds the IoU trigger."""
        boxes = np.stack([d.bbox for d in detections], axis=0)
        boxes_t = torch.from_numpy(boxes).float()
        iou_matrix = box_iou(boxes_t, boxes_t).numpy()

        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                if iou_matrix[i, j] > self.overlap_iou_trigger:
                    return True
        return False

    def log_stats(self) -> None:
        total = self._yolo_count + self._detr_count
        if total == 0:
            return
        yolo_pct = self._yolo_count / total * 100
        detr_pct = self._detr_count / total * 100
        logger.info(
            "Hybrid detection: YOLO %.1f%% (%d frames), DETR %.1f%% (%d frames)",
            yolo_pct, self._yolo_count, detr_pct, self._detr_count,
        )

    @staticmethod
    def uncertain_conf_threshold() -> float:
        """Detections below this confidence are flagged 'uncertain' for downstream
        consumers (lower EMA weight, reduced gallery update rate)."""
        return _env_float("SWAY_DETECTION_UNCERTAIN_CONF", 0.50)

    @property
    def is_nms_free(self) -> bool:
        return False

    def reset_stats(self) -> None:
        self._yolo_count = 0
        self._detr_count = 0
        self._last_detr_frame = -999
        self._force_detr_next = False
        self._last_detect_source = "unknown"


def create_hybrid_detector(device: str = "cuda"):
    """Factory: build a HybridDetector from env config."""
    from sway.detector_factory import create_detector

    hybrid_enabled = os.environ.get("SWAY_DETECTOR_HYBRID", "1").strip()
    if hybrid_enabled not in ("1", "true", "yes", "on"):
        return create_detector(device=device)

    primary = os.environ.get("SWAY_DETECTOR_PRIMARY", "yolo26l_dancetrack")
    precision = os.environ.get("SWAY_DETECTOR_PRECISION", "rt_detr_l")

    yolo = create_detector(primary="yolo26l_dancetrack", device=device)

    if primary.startswith("yolo"):
        detr = create_detector(primary=precision, device=device)
    else:
        detr = create_detector(primary=primary, device=device)

    return HybridDetector(yolo_detector=yolo, detr_detector=detr)
