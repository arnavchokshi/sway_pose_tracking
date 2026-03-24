"""Tests for ffprobe-based camera intrinsics probing."""

from __future__ import annotations

import math

from sway.pose_lift_3d import export_3d_for_viewer, pinhole_intrinsics
from sway.video_camera_probe import (
    _find_35mm_equivalent_mm,
    fx_fy_from_35mm_equivalent,
    probe_intrinsics_from_video,
)


def test_fx_fy_from_35mm_ultrawide_vs_default_fov() -> None:
    """Ultra-wide equivalent focal length should yield smaller fx than a nominal 70° FOV."""
    w, h = 1920, 1080
    fx_uw, fy_uw, _ = fx_fy_from_35mm_equivalent(13.0, w, h)
    fx_70, fy_70, _, _, _ = pinhole_intrinsics(w, h, fov_deg=70.0)
    assert fx_uw < fx_70
    assert math.isclose(fy_uw / fx_uw, h / w, rel_tol=1e-6)


def test_find_35mm_equivalent_apple_tag() -> None:
    tags = {
        "encoder": "H264",
        "com.apple.quicktime.camera.focal_length.35mm_equivalent": "24",
    }
    mm, key = _find_35mm_equivalent_mm(tags)
    assert mm == 24.0
    assert key == "com.apple.quicktime.camera.focal_length.35mm_equivalent"


def test_probe_intrinsics_from_video_monkeypatch(tmp_path, monkeypatch) -> None:
    fake_json = {
        "format": {"tags": {}},
        "streams": [
            {
                "codec_type": "video",
                "tags": {"camera.focal_length.35mm_equivalent": "28.5"},
            }
        ],
    }

    def _fake_ffprobe(_path: str, _timeout_s: float = 15.0):
        return fake_json

    monkeypatch.setattr("sway.video_camera_probe.ffprobe_json", _fake_ffprobe)
    p = tmp_path / "dummy.mp4"
    p.write_bytes(b"")
    out = probe_intrinsics_from_video(str(p), 1280, 720)
    assert out is not None
    assert out["source_key"] is not None
    assert abs(out["fx"] - fx_fy_from_35mm_equivalent(28.5, 1280, 720)[0]) < 1e-3


def test_export_unified_includes_intrinsics_source(monkeypatch) -> None:
    import numpy as np

    monkeypatch.setenv("SWAY_UNIFIED_3D_EXPORT", "1")

    lift = np.random.rand(17, 3).astype("float64") * 0.1
    k2 = np.zeros((17, 3), dtype="float64")
    k2[:, 2] = 0.9
    frame = {
        "frame_idx": 0,
        "boxes": [[0, 0, 100, 200]],
        "track_ids": [1],
        "poses": {
            1: {
                "keypoints": k2,
                "scores": np.ones(17),
                "lift_xyz": lift,
                "keypoints_3d": [[0.0, 0.0, 0.0]] * 17,
                "root_xyz": [0.0, 1.0, 2.0],
            }
        },
    }
    vc = {
        "fx": 1400.0,
        "fy": 787.5,
        "fov_deg": 55.0,
        "source_key": "camera.focal_length.35mm_equivalent",
        "focal_length_35mm_equiv_mm": 28.0,
    }
    blob = export_3d_for_viewer([frame], [1], 1, 30.0, 1280, 720, video_camera=vc)
    assert blob["camera"]["fx"] == 1400.0
    assert blob["camera"]["intrinsics_source"] == "video_metadata"
    assert blob["camera"]["metadata_tag"] == "camera.focal_length.35mm_equivalent"
    assert blob["camera"]["focal_length_35mm_equiv_mm"] == 28.0
