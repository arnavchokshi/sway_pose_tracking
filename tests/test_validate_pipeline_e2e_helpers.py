"""Unit tests for validate_pipeline_e2e helpers (no full pipeline run)."""

from pathlib import Path

import pytest

from tools.validate_pipeline_e2e import (
    optimization_hints,
    parse_phase_timings_s,
    validate_data_json_deep,
    validate_log_hard_failures,
    validate_outputs,
    validate_phase_markers,
)


def test_validate_phase_markers_complete_ordered():
    log = "\n".join([f"[{n}/11] Phase {n}" for n in range(1, 12)])
    assert validate_phase_markers(log) == []


def test_validate_phase_markers_missing():
    log = "[1/11] start\n[3/11] skip 2"
    err = validate_phase_markers(log)
    assert any("[2/11]" in e for e in err)


def test_validate_phase_markers_wrong_order():
    log = "[2/11] early\n[1/11] late\n" + "\n".join(f"[{n}/11]" for n in range(3, 12))
    err = validate_phase_markers(log)
    assert any("order" in e.lower() for e in err)


def test_parse_phase_timings_s():
    # En-dash (U+2013) as in main.py between 1 and 2
    log = """
[1/11] Phase 1
[2/11] Phase 2
  └─ Phases 1–2: 12.5s
[3/11] Phase 3
  └─ 1.0s
[4/11] Phase 4
  └─ 2.5s
"""
    t = parse_phase_timings_s(log)
    assert t.get("1-2") == pytest.approx(12.5)
    assert t.get("3") == pytest.approx(1.0)
    assert t.get("4") == pytest.approx(2.5)


def test_validate_log_hard_failures_traceback():
    assert validate_log_hard_failures("ok") == []
    assert any("Traceback" in e for e in validate_log_hard_failures("Traceback (most recent call last)"))


def test_validate_data_json_deep_finite():
    data = {
        "metadata": {"fps": 30.0, "native_fps": 30.0, "num_frames": 1, "keypoint_names": []},
        "track_summaries": {},
        "frames": [
            {
                "frame_idx": 0,
                "tracks": {
                    "1": {"keypoints": [[1.0, 2.0, 0.9], [0.0, 0.0, 0.0]]},
                },
            }
        ],
    }
    errs, warns = validate_data_json_deep(data)
    assert errs == []


def test_validate_data_json_deep_nan():
    data = {
        "metadata": {"fps": 30.0, "native_fps": 30.0},
        "frames": [
            {
                "frame_idx": 0,
                "tracks": {"1": {"keypoints": [[float("nan"), 1.0, 1.0]]}},
            }
        ],
    }
    errs, _ = validate_data_json_deep(data)
    assert any("non-finite" in e for e in errs)


def test_optimization_hints_dominant_phase():
    hints = optimization_hints({"1-2": 80.0, "5": 5.0}, wall_s=100.0)
    assert any("1-2" in h or "Phases" in h for h in hints)


def test_validate_outputs_missing_files(tmp_path):
    errs, _, _ = validate_outputs(tmp_path, "nope", deep=False)
    assert any("Missing" in e for e in errs)


def test_validate_outputs_consistent_json(tmp_path):
    stem = "tiny"
    (tmp_path / f"{stem}_poses.mp4").write_bytes(b"\0" * 500)
    (tmp_path / "prune_log.json").write_text(
        '{"total_frames": 2, "tracker": {"count": 0, "track_ids_before_prune": []}, '
        '"surviving_after_pre_pose": [], "surviving_after_post_pose": [], '
        '"hybrid_sam_frame_rois": [], "prune_entries": []}',
        encoding="utf-8",
    )
    (tmp_path / "data.json").write_text(
        '{"metadata": {"num_frames": 2, "fps": 30, "native_fps": 30}, '
        '"track_summaries": {}, "frames": [{"frame_idx": 0, "tracks": {}}, {"frame_idx": 1, "tracks": {}}]}',
        encoding="utf-8",
    )
    errs, warns, _ = validate_outputs(tmp_path, stem, deep=True)
    assert errs == []
    assert any("no tracks" in w.lower() for w in warns)


@pytest.mark.skipif(
    not Path(__file__).resolve().parent.parent.joinpath("tools", "validate_pipeline_e2e.py").is_file(),
    reason="script path",
)
def test_write_synthetic_video_smoke(tmp_path):
    import cv2

    from tools.validate_pipeline_e2e import write_synthetic_video

    p = tmp_path / "s.mp4"
    write_synthetic_video(p, n_frames=8, fps=8.0)
    assert p.is_file() and p.stat().st_size > 500
    cap = cv2.VideoCapture(str(p))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    assert n >= 6
