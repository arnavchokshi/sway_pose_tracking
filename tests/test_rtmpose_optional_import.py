"""RTMPose backend is optional (full MMPose stack)."""

import pytest


def test_rtmpose_class_loads_when_mm_stack_present():
    try:
        import mmengine  # noqa: F401
    except ImportError:
        pytest.skip("MMPose / mmengine not installed")
    from sway.rtmpose_estimator import RTMPoseEstimator

    assert RTMPoseEstimator.__name__ == "RTMPoseEstimator"
