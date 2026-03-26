"""Env helpers for YOLO / ViTPose batching (no model loads)."""

import os

import pytest
import torch

from sway.pose_estimator import (
    vitpose_debug_enabled,
    vitpose_effective_max_per_forward,
    vitpose_max_per_forward,
)
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


def test_vitpose_effective_cpu_no_default_chunk():
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)
    os.environ.pop("SWAY_VITPOSE_MPS_CHUNK", None)
    assert vitpose_effective_max_per_forward(torch.device("cpu")) == 0


def test_vitpose_effective_mps_default_chunk():
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)
    os.environ.pop("SWAY_VITPOSE_MPS_CHUNK", None)
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    assert vitpose_effective_max_per_forward(torch.device("mps")) == 2


def test_vitpose_effective_env_overrides_mps():
    os.environ["SWAY_VITPOSE_MAX_PER_FORWARD"] = "2"
    os.environ.pop("SWAY_VITPOSE_MPS_CHUNK", None)
    assert vitpose_effective_max_per_forward(torch.device("mps")) == 2
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)


def test_vitpose_debug_default_off():
    os.environ.pop("SWAY_VITPOSE_DEBUG", None)
    assert vitpose_debug_enabled() is False


def test_vitpose_debug_explicit_off():
    os.environ["SWAY_VITPOSE_DEBUG"] = "0"
    assert vitpose_debug_enabled() is False
    os.environ.pop("SWAY_VITPOSE_DEBUG", None)


def test_vitpose_debug_explicit_on():
    os.environ["SWAY_VITPOSE_DEBUG"] = "1"
    assert vitpose_debug_enabled() is True
    os.environ.pop("SWAY_VITPOSE_DEBUG", None)
