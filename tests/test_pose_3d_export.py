"""Smoke tests for 3D pose export and lift_xyz-aware scoring (no MotionAGFormer)."""

from __future__ import annotations

import numpy as np

from sway.pose_lift_3d import (
    COCO_BONES,
    COCO_KEYPOINTS,
    apply_bone_length_filter_to_lift_sequence,
    compute_joint_angles_3d,
    export_3d_for_viewer,
    lift_backend,
    median_bone_lengths_for_sequence,
    smooth_lift_xyz_for_export,
)
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


def test_lift_backend_env(monkeypatch) -> None:
    monkeypatch.delenv("SWAY_LIFT_BACKEND", raising=False)
    assert lift_backend() == "motionagformer"
    monkeypatch.setenv("SWAY_LIFT_BACKEND", "poseformerv2")
    assert lift_backend() == "poseformerv2"
    monkeypatch.setenv("SWAY_LIFT_BACKEND", "pfv2")
    assert lift_backend() == "poseformerv2"


def test_export_3d_for_viewer_structure(monkeypatch) -> None:
    monkeypatch.setenv("SWAY_UNIFIED_3D_EXPORT", "0")
    frames = [_synthetic_frame(i) for i in range(5)]
    blob = export_3d_for_viewer(frames, [1], total_frames=5, native_fps=30.0, frame_width=640, frame_height=480)
    assert blob["fps"] == 30.0
    assert blob["total_frames"] == 5
    assert blob["version"] == 1
    assert "camera" not in blob
    assert blob["keypoint_names"] == COCO_KEYPOINTS
    assert blob["bones"] == [[a, b] for a, b in COCO_BONES]
    assert "1" in blob["tracks"]
    t1 = blob["tracks"]["1"]
    assert t1["frames"] == [0, 1, 2, 3, 4]
    assert len(t1["keypoints_3d"]) == 5
    assert len(t1["keypoints_3d"][0]) == 17
    assert len(t1["keypoints_3d"][0][0]) == 3


def test_export_3d_unified_includes_camera(monkeypatch) -> None:
    monkeypatch.setenv("SWAY_UNIFIED_3D_EXPORT", "1")
    frames = [_synthetic_frame(i) for i in range(3)]
    for fr in frames:
        fr["poses"][1]["root_xyz"] = [0.1, 1.5, 3.0]
    blob = export_3d_for_viewer(frames, [1], total_frames=3, native_fps=24.0, frame_width=1280, frame_height=720)
    assert blob["version"] == 2
    cam = blob["camera"]
    assert cam["width"] == 1280 and cam["height"] == 720
    assert "fx" in cam and "fy" in cam and cam["fx"] > 0
    assert "root_xyz" in blob["tracks"]["1"]
    assert len(blob["tracks"]["1"]["root_xyz"]) == 3
    assert blob.get("include_lift_xyz") is True
    t1 = blob["tracks"]["1"]
    assert "lift_xyz" in t1
    assert len(t1["lift_xyz"]) == len(t1["keypoints_3d"])


def test_compute_joint_angles_3d_right_angle() -> None:
    # Left elbow 5–7–9: 90° at joint 7 (shoulder on −x, wrist on +z from elbow)
    kpts = [[0.0, 0.0, 0.0] for _ in range(17)]
    kpts[5] = [0.0, 0.0, 0.0]  # shoulder
    kpts[7] = [1.0, 0.0, 0.0]  # elbow
    kpts[9] = [1.0, 0.0, 1.0]  # wrist
    ang = compute_joint_angles_3d(kpts)
    assert "left_elbow" in ang
    assert abs(ang["left_elbow"] - 90.0) < 1.0


def test_median_bone_lengths_matches_constant_arm() -> None:
    T = 20
    seq = np.zeros((T, 17, 3), dtype=np.float32)
    for t in range(T):
        seq[t, 5] = [0.0, 0.0, 0.0]
        seq[t, 7] = [0.3, 0.0, 0.0]
        seq[t, 9] = [0.3, 0.4, 0.0]
    med = median_bone_lengths_for_sequence(seq)
    # bone (5,7) index in COCO_BONES
    idx57 = COCO_BONES.index((5, 7))
    idx79 = COCO_BONES.index((7, 9))
    assert abs(med[idx57] - 0.3) < 1e-5
    assert abs(med[idx79] - 0.4) < 1e-5


def test_bone_length_filter_stabilizes_noisy_bones() -> None:
    rng = np.random.RandomState(42)
    T = 30
    base = np.zeros((17, 3), dtype=np.float32)
    base[5] = [0.0, 0.0, 0.0]
    base[7] = [0.25, 0.0, 0.0]
    base[9] = [0.25, 0.35, 0.0]
    seq = np.tile(base[np.newaxis, ...], (T, 1, 1))
    noise = rng.randn(T, 17, 3).astype(np.float32) * 0.02
    seq = seq + noise
    filtered = apply_bone_length_filter_to_lift_sequence(seq)
    idx57 = COCO_BONES.index((5, 7))
    target = float(median_bone_lengths_for_sequence(seq)[idx57])
    for t in range(T):
        d = float(np.linalg.norm(filtered[t, 5] - filtered[t, 7]))
        assert abs(d - target) < 0.02
    # Pelvis (mid-hip) should stay near original after filter
    for t in range(T):
        mp = (seq[t, 11] + seq[t, 12]) * 0.5
        mf = (filtered[t, 11] + filtered[t, 12]) * 0.5
        assert np.linalg.norm(mp - mf) < 1e-4


def test_smooth_lift_xyz_for_export_dampens_alternating_z(monkeypatch) -> None:
    monkeypatch.setenv("SWAY_LIFT_SAVGOL", "1")
    monkeypatch.setenv("SWAY_LIFT_SAVGOL_WINDOW", "5")
    monkeypatch.setenv("SWAY_LIFT_SAVGOL_POLY", "2")
    n = 25
    frames = []
    for i in range(n):
        fd = _synthetic_frame(i)
        lift = np.full((17, 3), 0.5, dtype=np.float32)
        lift[:, 2] += 0.08 * (1.0 if i % 2 == 0 else -1.0)
        fd["poses"][1]["lift_xyz"] = lift
        frames.append(fd)
    z_before = np.array([float(frames[i]["poses"][1]["lift_xyz"][0, 2]) for i in range(n)])
    assert float(np.std(z_before)) > 0.02
    assert smooth_lift_xyz_for_export(frames) == 1
    z_after = np.array([float(frames[i]["poses"][1]["lift_xyz"][0, 2]) for i in range(n)])
    assert float(np.std(z_after)) < float(np.std(z_before)) * 0.5


def test_scoring_with_lift_xyz_runs() -> None:
    """Enough frames for cDTW Sakoe-Chiba window (needs len >= window)."""
    n = 35
    frames = [_synthetic_frame(i) for i in range(n)]
    out = process_all_frames_scoring_vectorized(frames)
    assert out is not None
    assert len(out["track_angles"]) == n
    assert 1 in out["track_angles"][0]
    assert "left_elbow" in out["track_angles"][0][1]
