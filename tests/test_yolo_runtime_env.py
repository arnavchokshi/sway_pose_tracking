"""YOLO runtime env helpers (no Ultralytics inference)."""

import os

from sway.pose_estimator import vitpose_force_fp32
from sway.tracker import (
    resolve_yolo_inference_weights,
    yolo_half_env_requested,
    yolo_predict_use_half,
)


def test_yolo_half_env_parsing():
    os.environ.pop("SWAY_YOLO_HALF", None)
    assert yolo_half_env_requested() is False
    os.environ["SWAY_YOLO_HALF"] = "1"
    assert yolo_half_env_requested() is True
    os.environ["SWAY_YOLO_HALF"] = "no"
    assert yolo_half_env_requested() is False
    os.environ.pop("SWAY_YOLO_HALF", None)


def test_yolo_predict_use_half_matches_cuda():
    os.environ["SWAY_YOLO_HALF"] = "1"
    # On CPU CI, half should not activate
    import torch

    expect = torch.cuda.is_available()
    assert yolo_predict_use_half() == expect
    os.environ.pop("SWAY_YOLO_HALF", None)


def test_resolve_yolo_engine_prefers_existing_file(tmp_path):
    eng = tmp_path / "model.engine"
    eng.write_bytes(b"\0")
    os.environ["SWAY_YOLO_ENGINE"] = str(eng)
    try:
        assert resolve_yolo_inference_weights() == str(eng.resolve())
    finally:
        os.environ.pop("SWAY_YOLO_ENGINE", None)


def test_resolve_yolo_engine_missing_raises(tmp_path):
    os.environ["SWAY_YOLO_ENGINE"] = str(tmp_path / "missing.engine")
    try:
        try:
            resolve_yolo_inference_weights()
        except FileNotFoundError as e:
            assert "SWAY_YOLO_ENGINE" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")
    finally:
        os.environ.pop("SWAY_YOLO_ENGINE", None)


def test_vitpose_fp32_env():
    os.environ.pop("SWAY_VITPOSE_FP32", None)
    assert vitpose_force_fp32() is False
    os.environ["SWAY_VITPOSE_FP32"] = "1"
    assert vitpose_force_fp32() is True
    os.environ.pop("SWAY_VITPOSE_FP32", None)
