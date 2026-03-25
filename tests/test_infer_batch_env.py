"""Env helpers for YOLO / ViTPose batching (no model loads)."""

import os

from sway.pose_estimator import vitpose_max_per_forward
from sway.tracker import yolo_infer_batch_size


def test_yolo_infer_batch_size_default():
    os.environ.pop("SWAY_YOLO_INFER_BATCH", None)
    assert yolo_infer_batch_size() == 1


def test_yolo_infer_batch_size_clamped():
    os.environ["SWAY_YOLO_INFER_BATCH"] = "4"
    assert yolo_infer_batch_size() == 4
    os.environ["SWAY_YOLO_INFER_BATCH"] = "999"
    assert yolo_infer_batch_size() == 32
    os.environ["SWAY_YOLO_INFER_BATCH"] = "0"
    assert yolo_infer_batch_size() == 1
    os.environ["SWAY_YOLO_INFER_BATCH"] = "nope"
    assert yolo_infer_batch_size() == 1
    os.environ.pop("SWAY_YOLO_INFER_BATCH", None)


def test_vitpose_max_per_forward():
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)
    assert vitpose_max_per_forward() == 0
    os.environ["SWAY_VITPOSE_MAX_PER_FORWARD"] = "6"
    assert vitpose_max_per_forward() == 6
    os.environ["SWAY_VITPOSE_MAX_PER_FORWARD"] = "bad"
    assert vitpose_max_per_forward() == 0
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)
