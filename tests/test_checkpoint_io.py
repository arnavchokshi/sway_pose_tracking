"""Round-trip tests for sway.checkpoint_io phase-1 NPZ bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from sway.checkpoint_io import load_phase1_yolo_dets, save_phase1_yolo_dets


def test_phase1_yolo_roundtrip(tmp_path: Path) -> None:
    dets = {
        0: [((10.0, 20.0, 30.0, 40.0), 0.9), ((50.0, 60.0, 70.0, 80.0), 0.7)],
        2: [((1.0, 2.0, 3.0, 4.0), 0.5)],
    }
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"fakevideo")
    ck = tmp_path / "ck"
    save_phase1_yolo_dets(
        ck,
        dets,
        total_frames=10,
        native_fps=30.0,
        output_fps=30.0,
        frame_width=1920,
        frame_height=1080,
        ystride=1,
        video_path=vid,
        params_path=None,
    )
    out, meta = load_phase1_yolo_dets(ck)
    assert meta["total_frames"] == 10
    assert meta["ystride"] == 1
    assert len(out) == 2
    assert len(out[0]) == 2
    assert out[0][0][0] == (10.0, 20.0, 30.0, 40.0)
    assert out[0][0][1] == pytest.approx(0.9)
    assert out[2][0][0] == (1.0, 2.0, 3.0, 4.0)
    assert meta.get("phase1_pre_classical") is None


def test_phase1_pre_classical_roundtrip(tmp_path: Path) -> None:
    dets = {
        0: [((10.0, 20.0, 30.0, 40.0), 0.9)],
    }
    pre = {
        0: [((10.0, 20.0, 30.0, 40.0), 0.9), ((50.0, 60.0, 70.0, 80.0), 0.7)],
    }
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"fakevideo")
    ck = tmp_path / "ck"
    save_phase1_yolo_dets(
        ck,
        dets,
        total_frames=5,
        native_fps=30.0,
        output_fps=30.0,
        frame_width=1920,
        frame_height=1080,
        ystride=1,
        video_path=vid,
        params_path=None,
        phase1_pre_classical_by_frame=pre,
    )
    out, meta = load_phase1_yolo_dets(ck)
    assert meta.get("phase1_pre_classical") is not None
    pc = meta["phase1_pre_classical"]
    assert pc is not None
    assert len(pc[0]) == 2
    assert out[0][0][0] == (10.0, 20.0, 30.0, 40.0)
    assert out[0][0][1] == pytest.approx(0.9)
