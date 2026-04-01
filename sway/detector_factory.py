"""
Detector Factory (PLAN_02 / PLAN_03)

Dispatches to the correct detection backend based on SWAY_DETECTOR_PRIMARY.
Used by tracker.py and hybrid_detector.py.

Env:
  SWAY_DETECTOR_PRIMARY – yolo26l_dancetrack | co_detr | co_dino | rt_detr_l | rt_detr_x
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, List

import numpy as np

logger = logging.getLogger(__name__)


class DetectorProtocol(Protocol):
    """Contract that all detectors must satisfy."""
    def detect(self, frame: np.ndarray) -> list: ...
    @property
    def is_nms_free(self) -> bool: ...


def create_detector(
    primary: str | None = None,
    device: str = "cuda",
) -> DetectorProtocol:
    """Instantiate the requested detector backend.

    Args:
        primary: detector name override; defaults to SWAY_DETECTOR_PRIMARY env var.
        device: torch device string.

    Returns:
        A detector instance with .detect(frame) -> List[Detection].
    """
    if primary is None:
        hybrid_enabled = os.environ.get("SWAY_DETECTOR_HYBRID", "0").strip().lower()
        if hybrid_enabled in ("1", "true", "yes", "on"):
            from sway.hybrid_detector import create_hybrid_detector

            return create_hybrid_detector(device=device)
        primary = os.environ.get("SWAY_DETECTOR_PRIMARY", "yolo26l_dancetrack")

    primary = primary.strip().lower()

    if primary.startswith("yolo"):
        return _create_yolo_detector(device)
    elif primary in ("co_detr", "co_dino"):
        from sway.detr_detector import DETRDetector

        model_name = "co_detr_swinl" if primary == "co_detr" else "co_dino_swinl"
        det = DETRDetector(model_name=model_name, device=device)
        if getattr(det, "_model", None) is None:
            logger.warning(
                "%s could not be loaded (missing detrex/mmdet/weights); using YOLO instead",
                primary,
            )
            return _create_yolo_detector(device)
        return det
    elif primary.startswith("rt_detr"):
        from sway.detr_detector import DETRDetector
        model_name = primary if primary in ("rt_detr_l", "rt_detr_x") else "rt_detr_l"
        return DETRDetector(model_name=model_name, device=device)
    else:
        logger.warning("Unknown detector '%s'; falling back to YOLO", primary)
        return _create_yolo_detector(device)


class _YOLODetectorAdapter:
    """Wraps the existing YOLO inference path to conform to DetectorProtocol."""

    def __init__(self, device: str = "cuda"):
        from ultralytics import YOLO
        from sway.tracker import resolve_yolo_inference_weights

        weight_path = resolve_yolo_inference_weights()
        self._model = YOLO(weight_path)
        self._device = device
        self._conf = float(os.environ.get("SWAY_YOLO_CONF", "0.22"))
        self._detect_size = int(os.environ.get("SWAY_DETECT_SIZE", "800"))

    def detect(self, frame: np.ndarray) -> list:
        from sway.detr_detector import Detection

        results = self._model.predict(
            frame,
            imgsz=self._detect_size,
            conf=self._conf,
            classes=[0],
            verbose=False,
            device=self._device,
        )
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for box, conf, c in zip(boxes, confs, cls):
                    if c == 0:
                        detections.append(Detection(
                            bbox=box.astype(np.float32),
                            confidence=float(conf),
                            class_id=0,
                        ))
        return detections

    @property
    def is_nms_free(self) -> bool:
        return False


def _create_yolo_detector(device: str) -> _YOLODetectorAdapter:
    return _YOLODetectorAdapter(device=device)
