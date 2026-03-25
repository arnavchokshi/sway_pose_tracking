"""Optional GSI box interpolation and hybrid-SAM weak-cue gating (opt-in via env / config)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from sway.hybrid_sam_refiner import load_hybrid_sam_config, weak_cues_say_ambiguous
from sway.interp_utils import blend_pose_keypoints_scores, blend_scalar, gsi_interp_scalar
from sway.tracker import _interpolate_box, _interp_box_at_t


def test_gsi_matches_linear_at_endpoints() -> None:
    y0, y1 = 10.0, 30.0
    l = 0.35
    assert abs(gsi_interp_scalar(0.0, y0, y1, l) - y0) < 1e-4
    assert abs(gsi_interp_scalar(1.0, y0, y1, l) - y1) < 1e-4


def test_interp_box_linear_mode_matches_legacy() -> None:
    a = (0.0, 0.0, 10.0, 20.0)
    b = (10.0, 5.0, 20.0, 25.0)
    for t in (0.1, 0.5, 0.9):
        got = _interp_box_at_t(a, b, t, mode="linear", gsi_lengthscale=0.35)
        assert got == _interpolate_box(a, b, t)


def test_aflink_thr_from_env_defaults() -> None:
    from sway import global_track_link as gtl

    keys = ("SWAY_AFLINK_THR_T0", "SWAY_AFLINK_THR_T1", "SWAY_AFLINK_THR_S", "SWAY_AFLINK_THR_P")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        thr_t, thr_s, thr_p = gtl._aflink_thr_from_env()
        assert thr_t == (0, 30)
        assert thr_s == 75
        assert abs(thr_p - 0.05) < 1e-9
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_weak_cues_ambiguous_without_prev() -> None:
    cfg = load_hybrid_sam_config()
    cfg["weak_height_frac"] = 0.12
    cfg["weak_conf_delta"] = 0.08
    cfg["weak_match_min_iou"] = 0.25
    dets = np.array(
        [[0.0, 0.0, 50.0, 100.0, 0.9, 0.0], [5.0, 0.0, 55.0, 100.0, 0.88, 0.0]],
        dtype=np.float32,
    )
    assert weak_cues_say_ambiguous(dets, None, cfg) is True


def test_weak_cues_stable_pair_not_ambiguous() -> None:
    cfg = load_hybrid_sam_config()
    cfg["weak_height_frac"] = 0.12
    cfg["weak_conf_delta"] = 0.08
    cfg["weak_match_min_iou"] = 0.25
    dets = np.array(
        [[0.0, 0.0, 50.0, 100.0, 0.9, 0.0], [5.0, 0.0, 55.0, 100.0, 0.88, 0.0]],
        dtype=np.float32,
    )
    prev = dets.copy()
    assert weak_cues_say_ambiguous(dets, prev, cfg) is False


def test_blend_scalar_linear() -> None:
    assert abs(blend_scalar(0.25, 0.0, 10.0, mode="linear", gsi_l=0.35) - 2.5) < 1e-6


def test_interpolate_frame_data_boxes_linear() -> None:
    from sway.visualizer import _interpolate_frame_data

    fd_lo = {
        "track_ids": [1],
        "boxes": [(0.0, 0.0, 10.0, 10.0)],
        "poses": {
            1: {
                "keypoints": np.zeros((17, 2), dtype=np.float32),
                "scores": np.full(17, 0.5, dtype=np.float32),
            }
        },
    }
    fd_hi = {
        "track_ids": [1],
        "boxes": [(10.0, 10.0, 20.0, 20.0)],
        "poses": {
            1: {
                "keypoints": np.full((17, 2), 2.0, dtype=np.float32),
                "scores": np.full(17, 0.8, dtype=np.float32),
            }
        },
    }
    out = _interpolate_frame_data(fd_lo, fd_hi, 0.5, temporal_mode="linear", gsi_lengthscale=0.35)
    assert len(out["boxes"]) == 1
    assert abs(out["boxes"][0][0] - 5.0) < 1e-5
    assert float(out["poses"][1]["scores"][0]) == pytest.approx(0.8)


def test_interpolate_frame_data_phase1_nearest() -> None:
    from sway.visualizer import _interpolate_frame_data

    fd_lo = {
        "track_ids": [],
        "boxes": [],
        "poses": {},
        "phase1_boxes": [(0.0, 0.0, 10.0, 20.0)],
        "phase1_confs": [0.91],
    }
    fd_hi = {
        "track_ids": [],
        "boxes": [],
        "poses": {},
        "phase1_boxes": [(5.0, 5.0, 15.0, 25.0)],
        "phase1_confs": [0.77],
    }
    lo = _interpolate_frame_data(fd_lo, fd_hi, 0.25, temporal_mode="linear", gsi_lengthscale=0.35)
    assert lo["phase1_confs"] == [0.91]
    hi = _interpolate_frame_data(fd_lo, fd_hi, 0.75, temporal_mode="linear", gsi_lengthscale=0.35)
    assert hi["phase1_confs"] == [0.77]


def test_blend_pose_linear_matches_lerp() -> None:
    kp0 = np.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=np.float32)
    kp1 = np.array([[[20.0, 10.0], [30.0, 10.0]]], dtype=np.float32)
    s0 = np.array([[0.8, 0.9]], dtype=np.float32)
    s1 = np.array([[0.9, 0.7]], dtype=np.float32)
    t = 0.25
    kp, sc = blend_pose_keypoints_scores(kp0, kp1, s0, s1, t, mode="linear", gsi_l=0.35)
    assert np.allclose(kp, (1 - t) * kp0 + t * kp1)
    assert np.allclose(sc, (1 - t) * s0 + t * s1)


def test_blend_pose_gsi_endpoints() -> None:
    kp0 = np.ones((1, 2, 2), dtype=np.float32) * 3.0
    kp1 = np.ones((1, 2, 2), dtype=np.float32) * 5.0
    s0 = np.ones((1, 2), dtype=np.float32) * 0.5
    s1 = np.ones((1, 2), dtype=np.float32) * 0.7
    l = 0.4
    kp_a, sc_a = blend_pose_keypoints_scores(kp0, kp1, s0, s1, 0.0, mode="gsi", gsi_l=l)
    kp_b, sc_b = blend_pose_keypoints_scores(kp0, kp1, s0, s1, 1.0, mode="gsi", gsi_l=l)
    assert np.allclose(kp_a, kp0)
    assert np.allclose(kp_b, kp1)
    assert np.allclose(sc_a, s0)
    assert np.allclose(sc_b, s1)


def test_weak_cues_conf_spike_ambiguous() -> None:
    cfg = load_hybrid_sam_config()
    cfg["weak_height_frac"] = 0.12
    cfg["weak_conf_delta"] = 0.08
    cfg["weak_match_min_iou"] = 0.25
    dets = np.array(
        [[0.0, 0.0, 50.0, 100.0, 0.9, 0.0], [5.0, 0.0, 55.0, 100.0, 0.88, 0.0]],
        dtype=np.float32,
    )
    prev = dets.copy()
    prev[1, 4] = 0.5
    assert weak_cues_say_ambiguous(dets, prev, cfg) is True
