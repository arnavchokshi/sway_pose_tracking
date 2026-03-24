"""Smoke tests for 3D pose export and lift_xyz-aware scoring (no MotionAGFormer)."""

from __future__ import annotations

import numpy as np

from sway.pose_lift_3d import COCO_BONES, COCO_KEYPOINTS, compute_joint_angles_3d, export_3d_for_viewer
from sway.scoring import process_all_frames_scoring_vectorized


def _synthetic_frame(fidx: int, tid: int = 1) -> dict:
    rng = np.random.RandomState(fidx)
    lift = rng.rand(17, 3).astype(np.float64) * 0.5 + 0.1
    k2 = np.zeros((17, 3), dtype=np.float64)
    k2[:, 0] = np.linspace(100, 200, 17) + fidx * 0.5
    k2[:, 1] = np.linspace(50, 400, 17)
    k2[:, 2] = 0.9
    k3 = [[float(k2[i, 0]), float(k2[i, 1]), float(lift[i, 2])] for i in range(17)]
    return {
        "frame_idx": fidx,
        "boxes": [[90, 40, 210, 420]],
        "track_ids": [tid],
        "poses": {
            tid: {
                "keypoints": k2.copy(),
                "scores": np.ones(17, dtype=np.float64) * 0.9,
                "lift_xyz": lift.copy(),
                "keypoints_3d": k3,
            }
        },
    }


def test_export_3d_for_viewer_structure() -> None:
    frames = [_synthetic_frame(i) for i in range(5)]
    blob = export_3d_for_viewer(frames, [1], total_frames=5, native_fps=30.0)
    assert blob["fps"] == 30.0
    assert blob["total_frames"] == 5
    assert blob["keypoint_names"] == COCO_KEYPOINTS
    assert blob["bones"] == [[a, b] for a, b in COCO_BONES]
    assert "1" in blob["tracks"]
    t1 = blob["tracks"]["1"]
    assert t1["frames"] == [0, 1, 2, 3, 4]
    assert len(t1["keypoints_3d"]) == 5
    assert len(t1["keypoints_3d"][0]) == 17
    assert len(t1["keypoints_3d"][0][0]) == 3


def test_compute_joint_angles_3d_right_angle() -> None:
    # Left elbow 5–7–9: 90° at joint 7 (shoulder on −x, wrist on +z from elbow)
    kpts = [[0.0, 0.0, 0.0] for _ in range(17)]
    kpts[5] = [0.0, 0.0, 0.0]  # shoulder
    kpts[7] = [1.0, 0.0, 0.0]  # elbow
    kpts[9] = [1.0, 0.0, 1.0]  # wrist
    ang = compute_joint_angles_3d(kpts)
    assert "left_elbow" in ang
    assert abs(ang["left_elbow"] - 90.0) < 1.0


def test_scoring_with_lift_xyz_runs() -> None:
    """Enough frames for cDTW Sakoe-Chiba window (needs len >= window)."""
    n = 35
    frames = [_synthetic_frame(i) for i in range(n)]
    out = process_all_frames_scoring_vectorized(frames)
    assert out is not None
    assert len(out["track_angles"]) == n
    assert 1 in out["track_angles"][0]
    assert "left_elbow" in out["track_angles"][0][1]
