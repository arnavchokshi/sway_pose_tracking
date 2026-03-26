"""
Automated coverage for the Sway 3D QA protocol (geometric sanity, PBD, export).

Maps to checklist IDs 1.x–3.x. Phase 4 (full dance stress) stays manual — markers below.

Private helpers are accessed via the module to avoid widening the public API.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sway import pose_lift_3d as pl3
from sway.pose_lift_3d import (
    PBD_FILTER_BONES,
    apply_bone_length_filter_to_lift_sequence,
    median_bone_lengths_for_sequence,
    smooth_lift_xyz_for_export,
)
from sway.video_camera_probe import probe_intrinsics_from_video

# --- Phase 1: Geometric sanity ---


def test_qa_1_1_flip_augmentation_coco_swap_and_nose_x_only() -> None:
    """1.1: Asymmetric pose → _flip_data swaps L/R COCO indices; nose only flips X."""
    # (B, T, J, C) with C=3; only xy matter for flip
    X = np.zeros((1, 1, 17, 3), dtype=np.float32)
    X[0, 0, 0, 0] = 0.25  # nose x
    X[0, 0, 5, 0] = -0.8
    X[0, 0, 5, 1] = 0.5
    X[0, 0, 6, 0] = 0.9
    X[0, 0, 6, 1] = -0.3
    X[0, 0, 7, 0] = -0.5
    X[0, 0, 7, 1] = 0.1
    X[0, 0, 8, 0] = 0.4
    X[0, 0, 8, 1] = 0.2

    F = pl3._flip_data(X)

    assert float(F[0, 0, 0, 0]) == pytest.approx(-0.25)
    assert float(F[0, 0, 0, 1]) == pytest.approx(0.0)

    # After flip, old right shoulder (6) sits at left index 5
    assert np.allclose(F[0, 0, 5, :2], [-0.9, -0.3])
    assert np.allclose(F[0, 0, 6, :2], [0.8, 0.5])
    assert np.allclose(F[0, 0, 7, :2], [-0.4, 0.2])
    assert np.allclose(F[0, 0, 8, :2], [0.5, 0.1])


def test_qa_1_2_per_person_norm_hips_at_origin_corner_4k() -> None:
    """1.2: Person in bottom-right of 4K frame; mid-hip at (0,0); coords bounded.

    Note: _normalize_per_person_coordinates scales by max(keypoint span)×1.2. With hips
    near the horizontal/vertical center of the keypoint tight box, |coord| ≤ ~1.05;
    extreme limb–hip asymmetry can exceed 1.2 (not assumed here).
    """
    w, h = 3840, 2160
    # Tight human-ish layout in bottom-right; hips near bbox center
    xs = np.array([3620, 3635, 3645, 3610, 3655, 3600, 3680, 3590, 3690, 3580, 3700, 3630, 3650, 3625, 3665, 3615, 3640], dtype=np.float32)
    ys = np.array([1920, 1910, 1915, 1930, 1925, 1950, 1955, 1980, 1985, 2000, 2005, 2040, 2045, 2080, 2085, 2110, 2115], dtype=np.float32)
    X = np.zeros((1, 1, 17, 3), dtype=np.float32)
    X[0, 0, :, 0] = xs
    X[0, 0, :, 1] = ys
    X[0, 0, :, 2] = 1.0

    N = pl3._normalize_per_person_coordinates(X)
    hip_x = float((N[0, 0, 11, 0] + N[0, 0, 12, 0]) * 0.5)
    hip_y = float((N[0, 0, 11, 1] + N[0, 0, 12, 1]) * 0.5)
    assert hip_x == pytest.approx(0.0, abs=1e-5)
    assert hip_y == pytest.approx(0.0, abs=1e-5)

    xy = N[0, 0, :, :2]
    assert float(np.max(xy)) <= 1.2 + 1e-3
    assert float(np.min(xy)) >= -1.2 - 1e-3


def test_qa_1_3_pelvis_anchor_before_camera_rotation() -> None:
    """1.3: After subtracting pelvis, COCO 11/12 midpoint is origin (pre-H36M step)."""
    post = np.random.RandomState(0).randn(17, 3).astype(np.float32) * 0.5
    pelvis = (post[11, :] + post[12, :]) * 0.5
    centered = post - pelvis
    mid = (centered[11, :] + centered[12, :]) * 0.5
    assert np.allclose(mid, 0.0, atol=1e-6)

    # Full postprocess must not be assumed to leave pelvis at origin (rotation follows).
    out = pl3._postprocess_pose3d_frame(post)
    mid_after = (out[11, :] + out[12, :]) * 0.5
    assert float(np.linalg.norm(mid_after)) > 0.01


# --- Phase 2: PBD / bone filter ---


def test_qa_2_1_pbd_left_upper_arm_length_stable_over_300_frames() -> None:
    """2.1: Bone (5,7) length CV after bone-length filter < 1% over 300 frames."""
    rng = np.random.RandomState(42)
    T = 300
    seq = np.zeros((T, 17, 3), dtype=np.float32)
    for t in range(T):
        seq[t, 5] = [0.0, 0.0, 0.0]
        seq[t, 7] = [0.28 + 0.02 * rng.randn(), 0.01 * rng.randn(), 0.01 * rng.randn()]
        seq[t, 9] = seq[t, 7] + np.array([0.05 * rng.randn(), 0.35, 0.02 * rng.randn()], dtype=np.float32)
        seq[t, 11] = [-0.1, -0.5, 0.0]
        seq[t, 12] = [0.1, -0.5, 0.0]
        # minimal scaffolding for other joints
        for j in (0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 15, 16):
            seq[t, j] = seq[t, 5] * 0.1 + 0.01 * rng.randn(3)

    filt = apply_bone_length_filter_to_lift_sequence(seq)
    idx57 = PBD_FILTER_BONES.index((5, 7))
    target = float(median_bone_lengths_for_sequence(seq)[idx57])
    lengths = np.linalg.norm(filt[:, 5, :] - filt[:, 7, :], axis=1)
    rel = np.abs(lengths - target) / max(target, 1e-8)
    assert float(np.max(rel)) < 0.01


def test_qa_2_2_synthetic_tethers_stable_under_pbd() -> None:
    """2.2: Nose–shoulder and ear–shoulder edges keep low length jitter after PBD."""
    rng = np.random.RandomState(7)
    T = 120
    seq = np.zeros((T, 17, 3), dtype=np.float32)
    for t in range(T):
        base = rng.randn(17, 3).astype(np.float32) * 0.05
        base[0] += np.array([0.0, 0.6, 0.0], np.float32)
        base[3] += np.array([-0.12, 0.55, 0.02], np.float32)
        base[4] += np.array([0.12, 0.55, -0.02], np.float32)
        base[5] += np.array([-0.22, 0.35, 0.0], np.float32)
        base[6] += np.array([0.22, 0.35, 0.0], np.float32)
        base[11] = np.array([-0.1, -0.2, 0.0], np.float32)
        base[12] = np.array([0.1, -0.2, 0.0], np.float32)
        seq[t] = base

    filt = apply_bone_length_filter_to_lift_sequence(seq)

    def _cv(a: np.ndarray) -> float:
        m = float(np.mean(a))
        return float(np.std(a) / m) if abs(m) > 1e-6 else 0.0

    pairs = [(0, 5), (0, 6), (3, 5), (4, 6)]
    for i, j in pairs:
        d0 = np.linalg.norm(seq[:, i] - seq[:, j], axis=1)
        d1 = np.linalg.norm(filt[:, i] - filt[:, j], axis=1)
        assert _cv(d1) <= _cv(d0) + 1e-6
        assert float(np.max(d1)) < float(np.max(d0)) * 1.5 + 1e-3


@pytest.mark.skipif(
    pl3._ensure_poseformerv2_path() is None or not pl3._poseformerv2_weights_path().is_file(),
    reason="PoseFormerV2 vendor tree or weights missing (QA 2.3)",
)
def test_qa_2_3_poseformerv2_occlusion_gap_finite_and_moderate_curvature(monkeypatch) -> None:
    """2.3: PoseFormerV2 with a 15-frame arm confidence blackout → finite 3D, no spike collapse.

    Full “parabolic arc vs straight line” is judged visually; here we gate obvious blow-ups.
    """
    monkeypatch.setenv("SWAY_LIFT_BACKEND", "poseformerv2")

    T = 243
    w, h = 1280, 720
    seq = np.zeros((1, T, 17, 3), dtype=np.float32)
    for t in range(T):
        a = (t - 120) / 40.0
        seq[0, t, :, 2] = 1.0
        seq[0, t, 5] = [400, 300, 1]
        seq[0, t, 6] = [500, 300, 1]
        seq[0, t, 7] = [350 + 20 * math.sin(a), 280 + 8 * a, 1]
        seq[0, t, 8] = [550, 280, 1]
        seq[0, t, 9] = [320 + 18 * math.sin(a), 250 + 5 * a, 1]
        seq[0, t, 11] = [420, 500, 1]
        seq[0, t, 12] = [480, 500, 1]
        for j in (0, 1, 2, 3, 4, 10, 13, 14, 15, 16):
            seq[0, t, j] = [430 + j, 200 + 0.1 * t, 1]

    seq[0, 60:75, 7, 2] = 0.0
    seq[0, 60:75, 9, 2] = 0.0

    out = pl3._infer_poseformerv2_sequence(seq, w, h)
    assert out is not None
    assert out.shape == (T, 17, 3)
    assert np.all(np.isfinite(out))

    wrist = out[:, 9, :]
    seg = wrist[55:80]
    d2 = np.diff(seg, 2)
    assert float(np.max(np.abs(d2))) < 50.0


# --- Phase 3: Spatial alignment ---


def test_qa_3_2_probe_missing_exif_returns_none(tmp_path, monkeypatch) -> None:
    """3.2: No 35mm-equivalent tags → None (callers use default FOV); no crash."""

    def _empty(_path: str, _timeout_s: float = 15.0):
        return {"format": {"tags": {}}, "streams": [{"codec_type": "video", "tags": {}}]}

    monkeypatch.setattr("sway.video_camera_probe.ffprobe_json", _empty)
    p = tmp_path / "compressed.mp4"
    p.write_bytes(b"")
    assert probe_intrinsics_from_video(str(p), 1920, 1080) is None


def test_qa_3_3_depth_maps_world_z_ordering(monkeypatch) -> None:
    """3.3: With SWAY_DEPTH_FOR_ROOT_Z=1, larger normalized depth at hips → larger world Z."""
    monkeypatch.setenv("SWAY_DEPTH_FOR_ROOT_Z", "1")
    k2 = np.zeros((17, 2), dtype=np.float64)
    k2[:, 0] = 640.0
    k2[:, 1] = 360.0
    lift = np.zeros((17, 3), dtype=np.float64)
    lift[:, 0] = 0.1
    lift[:, 1] = 0.5
    lift[:, 2] = 0.0

    fx, fy = 800.0, 800.0
    cx, cy = 640.0, 360.0
    z_near, z_far = 1.0, 8.0

    dm_a = np.full((720, 1280), 0.2, dtype=np.float32)
    dm_b = np.full((720, 1280), 0.8, dtype=np.float32)

    w3_a, root_a = pl3._compute_unified_world_keypoints(
        k2, lift, fx, fy, cx, cy, z_near, z_far, dm_a, 1280, 720
    )
    w3_b, root_b = pl3._compute_unified_world_keypoints(
        k2, lift, fx, fy, cx, cy, z_near, z_far, dm_b, 1280, 720
    )

    assert root_b[2] > root_a[2]
    # Same lift relative shape: every world Z should scale with root depth
    assert float(w3_b[0][2] - w3_a[0][2]) == pytest.approx(root_b[2] - root_a[2], rel=0, abs=1e-4)


def test_default_root_z_ignores_depth_map(monkeypatch) -> None:
    """Without SWAY_DEPTH_FOR_ROOT_Z, depth maps do not move pelvis Z (stable export)."""
    monkeypatch.delenv("SWAY_DEPTH_FOR_ROOT_Z", raising=False)
    monkeypatch.setenv("SWAY_DEFAULT_ROOT_Z", "3.0")
    k2 = np.zeros((17, 2), dtype=np.float64)
    k2[:, 0] = 640.0
    k2[:, 1] = 360.0
    lift = np.zeros((17, 3), dtype=np.float64)
    lift[:, 0] = 0.1
    lift[:, 1] = 0.5
    dm = np.full((720, 1280), 0.99, dtype=np.float32)
    w3, root = pl3._compute_unified_world_keypoints(
        k2, lift, 800.0, 800.0, 640.0, 360.0, 1.0, 8.0, dm, 1280, 720
    )
    assert root[2] == pytest.approx(3.0, abs=1e-5)
    # If depth drove Z, root Z would be ~7.9 for d_norm=0.99, z_near=1, z_far=8
    assert root[2] < 5.0


def test_qa_3_4_savgol_reduces_z_jitter_preserves_slow_trend(monkeypatch) -> None:
    """3.4: High-frequency Z noise reduced; slow macro motion preserved."""
    monkeypatch.setenv("SWAY_LIFT_SAVGOL", "1")
    monkeypatch.setenv("SWAY_LIFT_SAVGOL_WINDOW", "11")
    monkeypatch.setenv("SWAY_LIFT_SAVGOL_POLY", "3")

    rng = np.random.RandomState(1)
    n = 80
    frames = []
    macro = np.linspace(0.0, 0.4, n)
    for i in range(n):
        lift = np.full((17, 3), 0.2, dtype=np.float32)
        lift[:, 2] = float(macro[i] + 0.05 * rng.randn())
        frames.append(
            {
                "frame_idx": i,
                "boxes": [[0, 0, 100, 100]],
                "track_ids": [1],
                "poses": {
                    1: {
                        "keypoints": np.zeros((17, 3), dtype=np.float64),
                        "scores": np.ones(17),
                        "lift_xyz": lift,
                    }
                },
            }
        )

    z_raw = np.array([float(frames[i]["poses"][1]["lift_xyz"][0, 2]) for i in range(n)])
    assert float(np.std(np.diff(z_raw, 2))) > 0.002

    assert smooth_lift_xyz_for_export(frames) == 1
    z_sm = np.array([float(frames[i]["poses"][1]["lift_xyz"][0, 2]) for i in range(n)])
    assert float(np.std(np.diff(z_sm, 2))) < float(np.std(np.diff(z_raw, 2))) * 0.6
    assert abs(z_sm[-1] - z_sm[0]) > abs(z_raw[-1] - z_raw[0]) * 0.7


# --- Phase 4: placeholders (manual / video assets) ---


@pytest.mark.skip(reason="QA 4.1–4.3: run main pipeline on reference choreography videos")
def test_qa_phase4_dance_stress_placeholder() -> None:
    assert False
