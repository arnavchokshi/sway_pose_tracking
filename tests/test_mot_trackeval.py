import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sway.mot_format import (  # noqa: E402
    build_phase3_tracking_data_json,
    data_json_to_mot_lines,
    raw_tracks_to_mot_lines,
    xyxy_to_mot_line,
)
from sway.trackeval_runner import run_trackeval_single_sequence  # noqa: E402


def test_mot_lines_gt_format():
    gt = xyxy_to_mot_line(1, 1, 0, 0, 10, 20, 1.0, is_gt=True)
    parts = gt.split(",")
    assert len(parts) >= 8
    assert int(float(parts[7])) == 1


def test_raw_tracks_to_mot():
    raw = {3: [(0, (1.0, 2.0, 11.0, 22.0), 0.9)]}
    lines = raw_tracks_to_mot_lines(raw)
    assert len(lines) == 1
    assert lines[0].startswith("1,3,")


def test_data_json_to_mot():
    data = {
        "frames": [
            {
                "frame_idx": 0,
                "tracks": {
                    "3": {"box": [1, 2, 11, 22], "confidence": 0.9},
                },
            }
        ]
    }
    lines = data_json_to_mot_lines(data)
    assert len(lines) == 1
    assert lines[0].startswith("1,3,")


def test_phase3_tracking_data_json_matches_raw_tracks_mot():
    raw = {
        3: [(0, (1.0, 2.0, 11.0, 22.0), 0.9), (1, (2.0, 3.0, 12.0, 23.0), 0.8)],
        5: [(1, (0.0, 0.0, 5.0, 5.0), 1.0)],
    }
    dj = build_phase3_tracking_data_json(
        video_path="/tmp/x.mp4",
        raw_tracks=raw,
        total_frames=2,
        native_fps=30.0,
        output_fps=30.0,
    )
    from_lines = raw_tracks_to_mot_lines(raw)
    from_dj = data_json_to_mot_lines(dj)
    assert from_lines == from_dj


def test_trackeval_perfect_match():
    pytest.importorskip("trackeval")
    gt = [
        xyxy_to_mot_line(1, 1, 10, 10, 50, 100, 1.0, is_gt=True),
        xyxy_to_mot_line(2, 1, 11, 11, 51, 101, 1.0, is_gt=True),
    ]
    pr = [
        xyxy_to_mot_line(1, 1, 10, 10, 50, 100, 0.9),
        xyxy_to_mot_line(2, 1, 11, 11, 51, 101, 0.9),
    ]
    m = run_trackeval_single_sequence(gt, pr, "unit", 1920, 1080)
    assert m.get("Identity_IDF1", 0) >= 0.99
