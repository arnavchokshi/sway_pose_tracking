"""HybridDetector must return List[Detection] like other detectors (not a 2-tuple)."""

from __future__ import annotations

from unittest.mock import MagicMock

from sway.detr_detector import Detection
import numpy as np

from sway.hybrid_detector import HybridDetector


def test_hybrid_detect_returns_list_not_tuple(monkeypatch) -> None:
    monkeypatch.setenv("SWAY_HYBRID_OVERLAP_IOU_TRIGGER", "0.99")
    monkeypatch.setenv("SWAY_HYBRID_COOLDOWN_FRAMES", "0")

    yolo = MagicMock()
    detr = MagicMock()
    d1 = Detection(bbox=np.array([0, 0, 10, 10], dtype=np.float32), confidence=0.9, class_id=0)
    d2 = Detection(bbox=np.array([5, 5, 20, 20], dtype=np.float32), confidence=0.8, class_id=0)
    yolo.detect.return_value = [d1, d2]
    detr.detect.return_value = [d1]

    h = HybridDetector(yolo_detector=yolo, detr_detector=detr, overlap_iou_trigger=0.99)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = h.detect(frame, frame_idx=0)
    assert isinstance(out, list)
    assert not isinstance(out, tuple)
    assert len(out) == 2
    assert h._last_detect_source == "yolo"


def test_hybrid_uses_detr_when_overlap_high(monkeypatch) -> None:
    monkeypatch.setenv("SWAY_HYBRID_COOLDOWN_FRAMES", "0")

    yolo = MagicMock()
    detr = MagicMock()
    a = Detection(bbox=np.array([0, 0, 100, 100], dtype=np.float32), confidence=0.9, class_id=0)
    b = Detection(bbox=np.array([10, 10, 90, 90], dtype=np.float32), confidence=0.85, class_id=0)
    yolo.detect.return_value = [a, b]
    detr_d = Detection(bbox=np.array([1, 1, 50, 50], dtype=np.float32), confidence=0.95, class_id=0)
    detr.detect.return_value = [detr_d]

    h = HybridDetector(yolo_detector=yolo, detr_detector=detr, overlap_iou_trigger=0.01)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    out = h.detect(frame, frame_idx=1)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] is detr_d
    assert h._last_detect_source == "detr"
