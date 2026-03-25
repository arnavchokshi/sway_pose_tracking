"""track_stats export helpers (no full pipeline)."""

from sway.track_observation import TrackObservation
from sway.track_stats_export import compute_track_quality_stats


def test_compute_track_quality_stats_basic():
    raw = {
        1: [
            TrackObservation(0, (0.0, 0.0, 10.0, 10.0), 0.9),
            TrackObservation(2, (1.0, 0.0, 11.0, 10.0), 0.9),
        ],
        2: [TrackObservation(1, (5.0, 5.0, 15.0, 20.0), 0.8)],
    }
    s = compute_track_quality_stats(raw, total_frames=10, yolo_stride=1)
    assert s["schema_version"] == 1
    assert s["num_tracks"] == 2
    assert s["total_observations"] == 3
    assert s["total_frames"] == 10
    assert s["yolo_detection_stride"] == 1
    assert s["median_observations_per_track"] == 1.5
