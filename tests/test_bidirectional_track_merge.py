"""Bidirectional track merge helpers (no ffmpeg / YOLO)."""

from sway.bidirectional_track_merge import (
    bidirectional_iou_threshold,
    bidirectional_min_match_frames,
    bidirectional_track_pass_enabled,
    merge_forward_backward_tracks,
    remap_reverse_pass_timeline,
)
from sway.track_observation import TrackObservation


def test_remap_reverse_timeline():
    # Reverse file frame 0 = original last frame
    raw = {
        1: [
            TrackObservation(0, (10.0, 10.0, 20.0, 30.0), 0.9),
            TrackObservation(1, (11.0, 10.0, 21.0, 30.0), 0.9),
        ]
    }
    out = remap_reverse_pass_timeline(raw, total_frames=10)
    frames = {o.frame_idx for o in out[1]}
    assert frames == {8, 9}
    assert out[1][0].frame_idx == 8 and out[1][1].frame_idx == 9


def test_merge_links_reverse_id_to_forward():
    box = (100.0, 100.0, 140.0, 200.0)
    fwd = {1: [TrackObservation(0, box, 0.9), TrackObservation(1, box, 0.9), TrackObservation(2, box, 0.9)]}
    rev = {
        99: [
            TrackObservation(0, box, 0.8),
            TrackObservation(1, box, 0.8),
            TrackObservation(2, box, 0.8),
            TrackObservation(3, box, 0.8),
        ]
    }
    merged = merge_forward_backward_tracks(
        fwd,
        rev,
        iou_threshold=0.5,
        min_match_frames=3,
    )
    assert 99 not in merged
    assert len(merged[1]) == 4


def test_merge_keeps_forward_on_duplicate_frame():
    box_f = (0.0, 0.0, 10.0, 10.0)
    box_r = (1.0, 1.0, 11.0, 11.0)
    fwd = {1: [TrackObservation(0, box_f, 0.95)]}
    rev = {2: [TrackObservation(0, box_r, 0.5)]}
    merged = merge_forward_backward_tracks(
        fwd,
        rev,
        iou_threshold=0.1,
        min_match_frames=1,
    )
    assert len(merged[1]) == 1
    assert merged[1][0].conf == 0.95


def test_bidirectional_env_defaults_off():
    import os

    os.environ.pop("SWAY_BIDIRECTIONAL_TRACK_PASS", None)
    assert bidirectional_track_pass_enabled() is False
    assert bidirectional_min_match_frames() == 4
    assert bidirectional_iou_threshold() == 0.45
