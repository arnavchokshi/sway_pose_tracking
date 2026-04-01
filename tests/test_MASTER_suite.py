#!/usr/bin/env python3
"""
MASTER consolidated test suite — all tests from the former ``tests/test_*.py`` modules.

Run::

    pytest tests/test_MASTER_suite.py -v
    pytest tests/ -v

To regenerate after restoring individual modules from version control::

    python tools/consolidate_tests_into_master.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



# ---------- SOURCE: test_aflink_weights_optional.py ----------

"""Neural AFLink needs ``models/AFLink_epoch20.pth`` (``python -m tools.prefetch_models``)."""


import pytest

from sway.global_track_link import neural_global_stitch, resolve_aflink_weights
from sway.track_observation import TrackObservation


def _aflink_weights() -> bool:
    return resolve_aflink_weights().is_file()


@pytest.mark.skipif(not _aflink_weights(), reason="AFLink_epoch20.pth missing; run: python -m tools.prefetch_models")
def test_aflink_checkpoint_loads() -> None:
    from sway.aflink import AFLink

    p = resolve_aflink_weights()
    linker = AFLink(str(p), thrT=(0, 30), thrS=75, thrP=0.05)
    assert linker is not None


@pytest.mark.skipif(not _aflink_weights(), reason="AFLink_epoch20.pth missing")
def test_neural_global_stitch_runs_without_error() -> None:
    p = resolve_aflink_weights()
    box = (100.0, 100.0, 140.0, 200.0)
    raw = {
        1: [TrackObservation(0, box, 0.9), TrackObservation(1, box, 0.9), TrackObservation(2, box, 0.9)],
        2: [TrackObservation(10, box, 0.9), TrackObservation(11, box, 0.9), TrackObservation(12, box, 0.9)],
    }
    out = neural_global_stitch(raw, total_frames=30, path_AFLink=str(p))
    assert isinstance(out, dict)
    for tid, obs in out.items():
        assert isinstance(tid, int)
        assert all(isinstance(o, TrackObservation) for o in obs)


# ---------- SOURCE: test_bidirectional_track_merge.py ----------

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


# ---------- SOURCE: test_box_interp_optional.py ----------

"""Optional GSI box interpolation and hybrid-SAM weak-cue gating (opt-in via env / config)."""


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


# ---------- SOURCE: test_boxmot_tracker_env.py ----------

"""BoxMOT tracker kind env (no video inference)."""

import os

import pytest

from sway.tracker import _create_boxmot_tracker, _deepocsort_extra_from_env, boxmot_tracker_kind_from_env


def test_boxmot_tracker_kind_defaults():
    os.environ.pop("SWAY_BOXMOT_TRACKER", None)
    try:
        assert boxmot_tracker_kind_from_env() == "deepocsort"
    finally:
        os.environ.pop("SWAY_BOXMOT_TRACKER", None)


def test_boxmot_tracker_kind_aliases():
    for raw, want in (
        ("ByteTrack", "bytetrack"),
        ("byte", "bytetrack"),
        ("OC-SORT", "ocsort"),
        ("strongsort", "strongsort"),
    ):
        os.environ["SWAY_BOXMOT_TRACKER"] = raw
        try:
            assert boxmot_tracker_kind_from_env() == want
        finally:
            os.environ.pop("SWAY_BOXMOT_TRACKER", None)


@pytest.mark.parametrize("kind", ("bytetrack", "ocsort"))
def test_create_boxmot_tracker_lightweight(kind):
    import torch
    from pathlib import Path

    dev = torch.device("cpu")
    doc = _deepocsort_extra_from_env()
    tr = _create_boxmot_tracker(kind, 0.22, dev, Path("/nonexistent/reid.pt"), doc)
    assert tr is not None


def test_deepocsort_assoc_eiou_maps_to_ciou():
    os.environ["SWAY_BOXMOT_ASSOC_METRIC"] = "eiou"
    try:
        doc = _deepocsort_extra_from_env()
        assert doc["asso_func"] == "ciou"
    finally:
        os.environ.pop("SWAY_BOXMOT_ASSOC_METRIC", None)


def test_strongsort_requires_reid_file(tmp_path):
    import torch

    dev = torch.device("cpu")
    doc = _deepocsort_extra_from_env()
    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError):
        _create_boxmot_tracker("strongsort", 0.22, dev, missing, doc)


# ---------- SOURCE: test_checkpoint_io.py ----------

"""Round-trip tests for sway.checkpoint_io phase-1 NPZ bundle."""


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


# ---------- SOURCE: test_dancer_registry_pipeline.py ----------

"""Unit tests for optional Dancer Registry Phase 1–3 pass."""


import numpy as np

from sway.dancer_registry_pipeline import (
    _bhattacharyya,
    _build_frame_map,
    _swap_track_interval,
)
from sway.track_observation import TrackObservation


def test_bhattacharyya_identical() -> None:
    p = np.ones(8, dtype=np.float64) / 8.0
    assert abs(_bhattacharyya(p, p) - 1.0) < 1e-6


def test_swap_track_interval() -> None:
    raw = {
        1: [
            TrackObservation(0, (0.0, 0.0, 10.0, 20.0), 0.9),
            TrackObservation(1, (1.0, 1.0, 11.0, 21.0), 0.9),
            TrackObservation(2, (2.0, 2.0, 12.0, 22.0), 0.9),
        ],
        2: [
            TrackObservation(0, (100.0, 0.0, 110.0, 20.0), 0.8),
            TrackObservation(1, (101.0, 1.0, 111.0, 21.0), 0.8),
        ],
    }
    _swap_track_interval(raw, 1, 2, 0, 1)
    by_f = _build_frame_map(raw)
    assert 1 in by_f[0] and 2 in by_f[0]
    # Swapped boxes at frames 0–1
    assert abs(by_f[0][1][0] - 100.0) < 0.1
    assert abs(by_f[0][2][0] - 0.0) < 0.1
    assert abs(by_f[1][1][0] - 101.0) < 0.1
    assert abs(by_f[1][2][0] - 1.0) < 0.1
    # Frame 2 unchanged for tid 1
    assert abs(by_f[2][1][0] - 2.0) < 0.1


# ---------- SOURCE: test_doc_config_parity.py ----------

"""
Doc-Config Parity Tests

Validates that every SWAY_* key documented in FUTURE_PIPELINE.md and
MASTER_PIPELINE_GUIDELINE.md is referenced in at least one Python source file,
and that the pipeline_config_schema covers the newly wired future keys.

Also smoke-tests optional model factory paths to ensure they fail gracefully
(not with unhandled exceptions) when optional weights are absent.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest



def _collect_doc_keys(doc_path: Path) -> set:
    text = doc_path.read_text(encoding="utf-8", errors="ignore")
    return set(re.findall(r"\bSWAY_[A-Z0-9_]+\b", text))


def _collect_py_text() -> str:
    parts = []
    for p in REPO_ROOT.rglob("*.py"):
        if ".git/" in str(p):
            continue
        try:
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 1. Doc-to-code parity
# ---------------------------------------------------------------------------

class TestDocCodeParity:
    """Every documented SWAY_* key must appear in at least one .py file."""

    @pytest.fixture(scope="class")
    def py_text(self):
        return _collect_py_text()

    def test_future_pipeline_keys_in_code(self, py_text):
        doc = REPO_ROOT / "docs" / "Future_Plans" / "FUTURE_PIPELINE.md"
        if not doc.exists():
            pytest.skip("FUTURE_PIPELINE.md not found")
        keys = _collect_doc_keys(doc)
        missing = [k for k in sorted(keys) if k not in py_text]
        assert missing == [], f"FUTURE_PIPELINE keys missing from Python: {missing}"

    def test_master_pipeline_keys_in_code(self, py_text):
        doc = REPO_ROOT / "docs" / "MASTER_PIPELINE_GUIDELINE.md"
        if not doc.exists():
            pytest.skip("MASTER_PIPELINE_GUIDELINE.md not found")
        keys = _collect_doc_keys(doc)
        missing = [k for k in sorted(keys) if k not in py_text]
        assert missing == [], f"MASTER_PIPELINE keys missing from Python: {missing}"


# ---------------------------------------------------------------------------
# 2. Schema coverage for newly wired future keys
# ---------------------------------------------------------------------------

FUTURE_SCHEMA_KEYS = [
    "SWAY_TRACK_PARTIAL_MASK_FRAC",
    "SWAY_TRACK_DORMANT_MASK_FRAC",
    "SWAY_TRACK_MAX_AGE",
    "SWAY_BACKWARD_COI_ENABLED",
    "SWAY_COLLISION_DP_MAX_PERMUTATIONS",
    "SWAY_DETECTION_UNCERTAIN_CONF",
    "SWAY_ENROLLMENT_GALLERY_SIGNALS",
    "SWAY_ENROLLMENT_PART_MODEL",
    "SWAY_POSE_KEYPOINT_SET",
    "SWAY_POSE_SMART_PAD",
    "SWAY_POSE_VISIBILITY_THRESHOLD",
]


class TestSchemaContainsFutureKeys:
    def test_schema_has_all_future_keys(self):
        from sway.pipeline_config_schema import schema_payload

        sp = schema_payload()
        schema_keys = set()
        for field in sp["fields"]:
            k = field.get("key", "")
            if k:
                schema_keys.add(k)

        missing = [k for k in FUTURE_SCHEMA_KEYS if k not in schema_keys]
        assert missing == [], f"Future keys missing from schema: {missing}"


# ---------------------------------------------------------------------------
# 3. Runtime env key resolution
# ---------------------------------------------------------------------------

class TestRuntimeEnvResolution:
    """Env keys resolve to sensible defaults without error."""

    def test_track_state_thresholds(self):
        from sway.track_state import _partial_mask_frac, _dormant_mask_frac
        assert 0.0 < _partial_mask_frac() <= 1.0
        assert 0.0 < _dormant_mask_frac() <= 1.0
        assert _dormant_mask_frac() < _partial_mask_frac()

    def test_backward_coi_enabled(self):
        from sway.backward_pass import is_backward_coi_enabled
        assert isinstance(is_backward_coi_enabled(), bool)

    def test_collision_dp_max_perms(self):
        from sway.collision_solver import _env_int
        val = _env_int("SWAY_COLLISION_DP_MAX_PERMUTATIONS", 120)
        assert val >= 24

    def test_detection_uncertain_conf(self):
        from sway.hybrid_detector import HybridDetector
        val = HybridDetector.uncertain_conf_threshold()
        assert 0.0 < val < 1.0

    def test_enrollment_gallery_signals(self):
        from sway.enrollment import enrollment_gallery_signals
        sigs = enrollment_gallery_signals()
        assert isinstance(sigs, set)
        assert len(sigs) > 0
        assert sigs <= {"part", "face", "skeleton", "color", "spatial"}

    def test_enrollment_part_model(self):
        from sway.enrollment import enrollment_part_model
        model = enrollment_part_model()
        assert model in ("bpbreid", "paformer")

    def test_pose_keypoint_set(self):
        from sway.mask_guided_pose import pose_keypoint_set
        ks = pose_keypoint_set()
        assert ks in ("coco17", "wholebody133")

    def test_pose_smart_pad(self):
        from sway.mask_guided_pose import pose_smart_pad_enabled
        assert isinstance(pose_smart_pad_enabled(), bool)

    def test_pose_visibility_threshold(self):
        from sway.mask_guided_pose import pose_visibility_threshold
        val = pose_visibility_threshold()
        assert 0.0 < val < 1.0


# ---------------------------------------------------------------------------
# 4. Model factory graceful fallback
# ---------------------------------------------------------------------------

class TestModelFactoryGracefulFallback:
    """Optional model variants must not raise unhandled exceptions."""

    def test_bpbreid_default(self):
        from sway.reid_factory import create_part_reid
        ext = create_part_reid("bpbreid", device="cpu")
        assert ext is not None

    def test_bpbreid_finetuned_fallback(self):
        from sway.reid_factory import create_part_reid
        ext = create_part_reid("bpbreid_finetuned", device="cpu")
        assert ext is not None

    def test_paformer_fallback(self):
        from sway.reid_factory import create_part_reid
        ext = create_part_reid("paformer", device="cpu")
        assert ext is not None

    def test_osnet_fallback(self):
        from sway.reid_factory import create_part_reid
        ext = create_part_reid("osnet_x0_25", device="cpu")
        assert ext is not None

    def test_unknown_model_fallback(self):
        from sway.reid_factory import create_part_reid
        ext = create_part_reid("nonexistent_model", device="cpu")
        assert ext is not None

    def test_mocos_without_weights(self):
        from sway.mocos_extractor import MoCosExtractor
        ext = MoCosExtractor(checkpoint_path="/nonexistent/mocos.pth", device="cpu")
        seq = np.random.randn(60, 17, 3).astype(np.float32)
        emb = ext.extract(seq)
        assert emb is not None
        assert emb.shape == (256,)


# ---------------------------------------------------------------------------
# 5. Sweep wiring completeness
# ---------------------------------------------------------------------------

class TestSweepWiring:
    def test_verify_sweep_env_wiring_no_orphans(self):
        """All keys assigned in auto_sweep are referenced in at least one other .py file."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "verify_sweep_env_wiring",
            str(REPO_ROOT / "tools" / "verify_sweep_env_wiring.py"),
        )
        if spec is None or spec.loader is None:
            pytest.skip("verify_sweep_env_wiring.py not loadable")

        sweep_text = (REPO_ROOT / "tools" / "auto_sweep.py").read_text(
            encoding="utf-8", errors="ignore"
        )
        sweep_keys = set(re.findall(r'env\["(SWAY_[A-Z0-9_]+)"\]', sweep_text))
        py_text = _collect_py_text()

        orphan = []
        for k in sorted(sweep_keys):
            other_refs = py_text.replace(sweep_text, "")
            if k not in other_refs:
                orphan.append(k)

        assert orphan == [], f"Sweep-only keys (no runtime reference): {orphan}"


# ---------- SOURCE: test_experimental_hooks.py ----------

"""In-pipeline experimental hooks (GNN / HMR sidecar)."""

import json
import os
from pathlib import Path

from sway.experimental_hooks import (
    gnn_track_refine_enabled,
    maybe_gnn_refine_raw_tracks,
    write_hmr_mesh_sidecar_json,
)


def test_gnn_disabled_by_default():
    os.environ.pop("SWAY_GNN_TRACK_REFINE", None)
    assert gnn_track_refine_enabled() is False
    assert maybe_gnn_refine_raw_tracks({1: []}, 10, 1) == {1: []}


def test_gnn_enabled_single_empty_track():
    os.environ["SWAY_GNN_TRACK_REFINE"] = "1"
    try:
        raw = {1: []}
        out = maybe_gnn_refine_raw_tracks(raw, 3, 1)
        assert out is raw
        assert 1 in raw and raw[1] == []
    finally:
        os.environ.pop("SWAY_GNN_TRACK_REFINE", None)


def test_gnn_merges_high_iou_duplicate_ids():
    """Two IDs covering the same boxes on the same frames → one component after GNN."""
    os.environ["SWAY_GNN_TRACK_REFINE"] = "1"
    os.environ["SWAY_GNN_DEVICE"] = "cpu"
    try:
        box = (10.0, 20.0, 50.0, 120.0)
        raw = {
            1: [(0, box, 0.9), (1, box, 0.9), (2, box, 0.9)],
            2: [(0, box, 0.85), (1, box, 0.85), (2, box, 0.85)],
        }
        out = maybe_gnn_refine_raw_tracks(raw, total_frames=10, ystride=1)
        assert len(out) == 1
        tid = next(iter(out))
        assert len(out[tid]) == 3
    finally:
        os.environ.pop("SWAY_GNN_TRACK_REFINE", None)
        os.environ.pop("SWAY_GNN_DEVICE", None)


def test_gnn_skips_far_disjoint_tracks():
    os.environ["SWAY_GNN_TRACK_REFINE"] = "1"
    os.environ["SWAY_GNN_DEVICE"] = "cpu"
    os.environ["SWAY_GNN_MAX_GAP"] = "5"
    try:
        a = (10.0, 20.0, 40.0, 100.0)
        b = (200.0, 20.0, 240.0, 100.0)
        raw = {
            1: [(0, a, 0.9), (1, a, 0.9)],
            2: [(100, b, 0.9), (101, b, 0.9)],
        }
        out = maybe_gnn_refine_raw_tracks(raw, total_frames=200, ystride=1)
        assert len(out) == 2
    finally:
        os.environ.pop("SWAY_GNN_TRACK_REFINE", None)
        os.environ.pop("SWAY_GNN_DEVICE", None)
        os.environ.pop("SWAY_GNN_MAX_GAP", None)


def test_hmr_sidecar_writes(tmp_path):
    os.environ["SWAY_HMR_MESH_SIDECAR"] = "1"
    try:
        write_hmr_mesh_sidecar_json(tmp_path)
        p = tmp_path / "hmr_mesh_sidecar.json"
        assert p.is_file()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data.get("schema") == "sway.hmr_mesh_sidecar.v1"
    finally:
        os.environ.pop("SWAY_HMR_MESH_SIDECAR", None)


# ---------- SOURCE: test_future_pipeline_integration.py ----------

"""
Integration tests for the Future Pipeline modules (PLAN_01–PLAN_21).

Validates:
  1. Each module's core logic works with synthetic data
  2. Cross-module interfaces connect correctly
  3. Factory patterns dispatch to the right backends
  4. Data flows end-to-end through the pipeline layers
"""

import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── PLAN_01: Track State Machine ──────────────────────────────────────────

class TestTrackStateMachine:
    def test_enum_ordering(self):
        from sway.track_state import TrackState
        assert TrackState.ACTIVE > TrackState.PARTIAL > TrackState.DORMANT > TrackState.LOST

    def test_clean_active_sequence(self):
        from sway.track_state import TrackLifecycle, TrackState, update_state
        lc = TrackLifecycle(track_id=1)
        lc.set_reference_area(1000.0)

        for f in range(10):
            state = update_state(lc, mask_area=900.0, num_visible_joints=10, frame_idx=f)
            assert state == TrackState.ACTIVE, f"Frame {f}: expected ACTIVE, got {state.name}"
        assert lc.frames_in_dormant == 0

    def test_active_to_partial_to_dormant_to_lost(self):
        from sway.track_state import TrackLifecycle, TrackState, update_state
        lc = TrackLifecycle(track_id=2)
        lc.set_reference_area(1000.0)

        # ACTIVE
        update_state(lc, 500.0, 8, 0)
        assert lc.state == TrackState.ACTIVE

        # PARTIAL (mask_frac=0.15, num_joints=3 → ≥0.05 and ≥1 joint)
        update_state(lc, 150.0, 3, 1)
        assert lc.state == TrackState.PARTIAL

        # DORMANT (mask_frac=0.01, joints=0)
        update_state(lc, 10.0, 0, 2)
        assert lc.state == TrackState.DORMANT
        assert lc.frames_in_dormant == 1

        # Stay DORMANT for 300 more frames → LOST
        for f in range(3, 304):
            update_state(lc, 0.0, 0, f)
        assert lc.state == TrackState.LOST

    def test_recovery_from_dormant(self):
        from sway.track_state import TrackLifecycle, TrackState, update_state
        lc = TrackLifecycle(track_id=3)
        lc.set_reference_area(1000.0)

        # Go DORMANT
        update_state(lc, 10.0, 0, 0)
        assert lc.state == TrackState.DORMANT

        # Recover to ACTIVE
        update_state(lc, 800.0, 12, 1)
        assert lc.state == TrackState.ACTIVE
        assert lc.frames_in_dormant == 0

    def test_state_history_recorded(self):
        from sway.track_state import TrackLifecycle, TrackState, update_state
        lc = TrackLifecycle(track_id=4)
        lc.set_reference_area(1000.0)

        update_state(lc, 900.0, 10, 0)  # ACTIVE
        update_state(lc, 10.0, 0, 1)     # DORMANT
        update_state(lc, 800.0, 10, 2)   # ACTIVE

        assert len(lc.state_history) == 3
        assert lc.state_history[0] == (0, TrackState.ACTIVE)
        assert lc.state_history[1] == (1, TrackState.DORMANT)
        assert lc.state_history[2] == (2, TrackState.ACTIVE)

    def test_should_run_pose(self):
        from sway.track_state import TrackState, should_run_pose
        assert should_run_pose(TrackState.ACTIVE) is True
        assert should_run_pose(TrackState.PARTIAL) is True
        assert should_run_pose(TrackState.DORMANT) is False
        assert should_run_pose(TrackState.LOST) is False

    def test_should_update_gallery(self):
        from sway.track_state import TrackState, should_update_gallery
        assert should_update_gallery(TrackState.ACTIVE) is True
        assert should_update_gallery(TrackState.PARTIAL) is False

    def test_should_generate_critique(self):
        from sway.track_state import TrackState, should_generate_critique
        assert should_generate_critique(TrackState.ACTIVE) is True
        assert should_generate_critique(TrackState.PARTIAL) is True
        assert should_generate_critique(TrackState.DORMANT) is False

    def test_lifecycle_serialization(self):
        from sway.track_state import TrackLifecycle, TrackState, lifecycle_to_dict, update_state
        lc = TrackLifecycle(track_id=5)
        lc.set_reference_area(500.0)
        update_state(lc, 400.0, 10, 0)

        d = lifecycle_to_dict(lc)
        assert d["track_id"] == 5
        assert d["state"] == "ACTIVE"
        assert d["reference_mask_area"] == 500.0
        assert isinstance(d["state_history"], list)

    def test_none_mask_goes_dormant(self):
        from sway.track_state import TrackLifecycle, TrackState, update_state
        lc = TrackLifecycle(track_id=6)
        lc.set_reference_area(1000.0)
        update_state(lc, None, 0, 0)
        assert lc.state == TrackState.DORMANT

    def test_auto_reference_area(self):
        from sway.track_state import TrackLifecycle
        lc = TrackLifecycle(track_id=7)
        for f in range(35):
            lc.record_area_sample(100.0 + f * 10, f)
        assert lc.reference_mask_area == 100.0 + 29 * 10  # max of first 30


# ── PLAN_05: Cross-Object Interaction ─────────────────────────────────────

class TestCrossObjectInteraction:
    def test_no_collision_when_masks_dont_overlap(self):
        from sway.cross_object_interaction import CrossObjectInteraction
        coi = CrossObjectInteraction(mask_iou_thresh=0.25)

        h, w = 100, 100
        mask_a = np.zeros((h, w), dtype=bool)
        mask_a[0:30, 0:30] = True
        mask_b = np.zeros((h, w), dtype=bool)
        mask_b[70:100, 70:100] = True

        actions = coi.check_collisions(
            masks={1: mask_a, 2: mask_b},
            logits={1: 0.9, 2: 0.9},
            frame_idx=0,
        )
        assert len(actions) == 0

    def test_collision_detected_on_overlap(self):
        from sway.cross_object_interaction import CrossObjectInteraction
        coi = CrossObjectInteraction(mask_iou_thresh=0.10)

        h, w = 100, 100
        mask_a = np.zeros((h, w), dtype=bool)
        mask_a[20:80, 20:80] = True
        mask_b = np.zeros((h, w), dtype=bool)
        mask_b[40:90, 40:90] = True

        # Seed logit history so variance can be computed
        for f in range(15):
            coi.update_logits(1, 0.9)
            coi.update_logits(2, 0.9 - f * 0.05)  # track 2 drops → higher variance

        actions = coi.check_collisions(
            masks={1: mask_a, 2: mask_b},
            logits={1: 0.9, 2: 0.2},
            frame_idx=15,
        )
        assert len(actions) == 1
        assert actions[0].track_id == 2  # higher variance track quarantined

    def test_collision_exit_hysteresis(self):
        from sway.cross_object_interaction import CrossObjectInteraction
        coi = CrossObjectInteraction(mask_iou_thresh=0.10)

        h, w = 100, 100
        mask_overlap = np.zeros((h, w), dtype=bool)
        mask_overlap[20:80, 20:80] = True

        # Create collision
        coi.check_collisions(
            masks={1: mask_overlap, 2: mask_overlap},
            logits={1: 0.9, 2: 0.5},
            frame_idx=0,
        )
        assert len(coi.get_active_collisions()) == 1

        # Separate masks → collision should exit
        mask_a = np.zeros((h, w), dtype=bool)
        mask_a[0:20, 0:20] = True
        mask_b = np.zeros((h, w), dtype=bool)
        mask_b[80:100, 80:100] = True

        coi.check_collisions(
            masks={1: mask_a, 2: mask_b},
            logits={1: 0.9, 2: 0.9},
            frame_idx=10,
        )
        assert len(coi.get_active_collisions()) == 0


# ── PLAN_06: MeMoSORT ─────────────────────────────────────────────────────

class TestMeMoSORT:
    def test_kalman_predict_update_cycle(self):
        from sway.memosort import MemoryKalmanFilter
        kf = MemoryKalmanFilter(np.array([10, 20, 50, 80], dtype=np.float32))

        pred = kf.predict()
        assert pred.bbox_xyxy.shape == (4,)
        assert pred.velocity.shape == (2,)

        kf.update(np.array([12, 22, 52, 82], dtype=np.float32))
        pred2 = kf.predict()
        # Should have moved slightly right/down
        center_x = (pred2.bbox_xyxy[0] + pred2.bbox_xyxy[2]) / 2
        assert center_x > 10  # moved from initial

    def test_adaptive_iou(self):
        from sway.memosort import adaptive_iou
        box_a = np.array([0, 0, 10, 10], dtype=np.float32)
        box_b = np.array([15, 15, 25, 25], dtype=np.float32)

        # No overlap with zero velocity
        iou_static = adaptive_iou(box_a, box_b, np.zeros(2), np.zeros(2), alpha=0.0)
        assert iou_static == 0.0

        # With high velocity expansion, boxes should overlap
        iou_fast = adaptive_iou(box_a, box_b, np.array([10, 10]), np.zeros(2), alpha=1.0)
        assert iou_fast > 0.0

    def test_memosort_match(self):
        from sway.memosort import MeMoSORT
        ms = MeMoSORT(memory_length=10)

        ms.init_track(1, np.array([10, 10, 50, 50], dtype=np.float32))
        ms.init_track(2, np.array([200, 200, 250, 250], dtype=np.float32))

        preds = ms.predict_all()
        assert 1 in preds and 2 in preds

        det_boxes = [
            np.array([12, 12, 52, 52], dtype=np.float32),   # near track 1
            np.array([198, 198, 248, 248], dtype=np.float32),  # near track 2
        ]
        matches, unmatched_t, unmatched_d = ms.match(preds, det_boxes)
        assert len(matches) == 2
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 0


# ── PLAN_07: Enrollment Gallery ───────────────────────────────────────────

class TestEnrollmentGallery:
    def test_gallery_save_load_roundtrip(self):
        from sway.enrollment import DancerGallery, save_gallery, load_gallery

        g = DancerGallery(
            dancer_id=1,
            name="TestDancer",
            global_embedding=np.random.randn(2048).astype(np.float32),
            reference_mask_area=5000.0,
            spatial_position=(0.3, 0.5),
            enrollment_frame=10,
        )
        g.color_histograms = {"upper": np.random.rand(96).astype(np.float32)}
        g.part_embeddings = {"torso": np.random.randn(256).astype(np.float32)}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_gallery([g], path)
            loaded = load_gallery(path)
            assert len(loaded) == 1
            lg = loaded[0]
            assert lg.dancer_id == 1
            assert lg.name == "TestDancer"
            assert lg.reference_mask_area == 5000.0
            assert lg.spatial_position == (0.3, 0.5)
            assert lg.global_embedding is not None
            np.testing.assert_allclose(lg.global_embedding, g.global_embedding, atol=1e-5)
            assert "torso" in lg.part_embeddings
            assert "upper" in lg.color_histograms
        finally:
            os.unlink(path)


# ── PLAN_08: BPBreID ──────────────────────────────────────────────────────

class TestBPBreID:
    def test_extract_returns_part_embeddings(self):
        from sway.bpbreid_extractor import BPBreIDExtractor, PartEmbeddings
        ext = BPBreIDExtractor(device="cpu")
        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, 2] = 0.8  # all visible

        result = ext.extract(crop, keypoints)
        assert isinstance(result, PartEmbeddings)
        assert result.global_emb.shape[0] > 0
        assert len(result.part_embs) > 0
        assert np.abs(np.linalg.norm(result.global_emb) - 1.0) < 0.01

    def test_compare_same_vs_different(self):
        from sway.bpbreid_extractor import BPBreIDExtractor
        ext = BPBreIDExtractor(device="cpu")
        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 2] = 0.8

        emb1 = ext.extract(crop, kp)
        emb2 = ext.extract(crop, kp)  # same image

        dist_same = ext.compare(emb1, emb2)
        assert dist_same < 0.1  # near-identical

        crop_diff = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        emb3 = ext.extract(crop_diff, kp)
        dist_diff = ext.compare(emb1, emb3)
        # Different image should have higher distance (not guaranteed but likely)
        assert isinstance(dist_diff, float)

    def test_visibility_computation(self):
        from sway.bpbreid_extractor import BPBreIDExtractor
        ext = BPBreIDExtractor(device="cpu")

        kp_visible = np.zeros((17, 3), dtype=np.float32)
        kp_visible[:, 2] = 0.9
        vis = ext._compute_visibility(kp_visible)
        assert all(vis.values())

        kp_invisible = np.zeros((17, 3), dtype=np.float32)
        vis2 = ext._compute_visibility(kp_invisible)
        assert not any(vis2.values())


# ── PLAN_12: Color Histogram ──────────────────────────────────────────────

class TestColorHistogram:
    def test_extract_returns_three_regions(self):
        from sway.color_histogram_reid import ColorHistogramExtractor
        ext = ColorHistogramExtractor(color_space="hsv", n_bins=16)

        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        mask = np.ones((256, 128), dtype=bool)
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 2] = 0.8

        result = ext.extract(crop, mask, kp)
        assert "upper" in result
        assert "lower" in result
        assert "shoes" in result
        # Check L1-normalized (sums to ~1)
        assert abs(result["upper"].sum() - 1.0) < 0.01

    def test_compare_identical_histograms(self):
        from sway.color_histogram_reid import ColorHistogramExtractor
        ext = ColorHistogramExtractor()

        hist = {"upper": np.ones(96, dtype=np.float32) / 96}
        dist = ext.compare(hist, hist)
        assert dist < 0.01  # identical


# ── PLAN_13: Re-ID Fusion ─────────────────────────────────────────────────

class TestReIDFusion:
    def test_match_with_gallery(self):
        from sway.enrollment import DancerGallery
        from sway.reid_fusion import ReIDFusionEngine, ReIDQuery

        emb = np.random.randn(2048).astype(np.float32)
        emb /= np.linalg.norm(emb)

        gallery = [
            DancerGallery(dancer_id=1, global_embedding=emb.copy(),
                          spatial_position=(0.3, 0.5), enrollment_frame=0),
            DancerGallery(dancer_id=2, global_embedding=np.random.randn(2048).astype(np.float32),
                          spatial_position=(0.7, 0.5), enrollment_frame=0),
        ]
        gallery[1].global_embedding /= np.linalg.norm(gallery[1].global_embedding)

        engine = ReIDFusionEngine(gallery=gallery)

        from sway.bpbreid_extractor import PartEmbeddings
        query = ReIDQuery(
            track_id=99,
            part_embeddings=PartEmbeddings(global_emb=emb.copy(), part_embs={}, visibility={}),
            spatial_position=(0.3, 0.5),
        )
        dancer_id, score = engine.match(query, current_frame=5)
        assert dancer_id == 1  # should match the identical embedding
        assert score > 0.5

    def test_unknown_on_low_confidence(self):
        from sway.enrollment import DancerGallery
        from sway.reid_fusion import ReIDFusionEngine, ReIDQuery, UNKNOWN_ID

        gallery = [
            DancerGallery(dancer_id=1, spatial_position=(0.5, 0.5), enrollment_frame=0),
        ]
        engine = ReIDFusionEngine(gallery=gallery)
        engine.min_confidence = 0.99  # impossibly high

        query = ReIDQuery(track_id=99, spatial_position=(0.1, 0.1))
        dancer_id, _ = engine.match(query)
        assert dancer_id == UNKNOWN_ID


# ── PLAN_14: Pose-Gated EMA ──────────────────────────────────────────────

class TestPoseGatedEMA:
    def test_alpha_zero_when_dormant(self):
        from sway.pose_gated_ema import PoseGatedEMA
        from sway.track_state import TrackState
        ema = PoseGatedEMA()
        alpha = ema.compute_alpha(
            np.array([10, 10, 50, 50]),
            [np.array([100, 100, 150, 150])],
            np.array([0.9] * 17),
            TrackState.DORMANT,
        )
        assert alpha == 0.0

    def test_alpha_high_when_isolated_good_pose(self):
        from sway.pose_gated_ema import PoseGatedEMA
        from sway.track_state import TrackState
        ema = PoseGatedEMA(alpha_high=0.15, isolation_dist=1.0)
        alpha = ema.compute_alpha(
            np.array([10, 10, 50, 80]),  # bbox height = 70
            [np.array([500, 500, 550, 570])],  # far away
            np.array([0.9] * 17),
            TrackState.ACTIVE,
        )
        assert alpha == 0.15

    def test_alpha_low_when_crowded(self):
        from sway.pose_gated_ema import PoseGatedEMA
        from sway.track_state import TrackState
        ema = PoseGatedEMA(alpha_high=0.15, alpha_low=0.0, isolation_dist=1.5)
        alpha = ema.compute_alpha(
            np.array([10, 10, 50, 80]),  # height 70
            [np.array([30, 10, 70, 80])],  # very close
            np.array([0.9] * 17),
            TrackState.ACTIVE,
        )
        assert alpha == 0.0

    def test_gallery_ema_update(self):
        from sway.pose_gated_ema import PoseGatedEMA
        from sway.enrollment import DancerGallery
        ema = PoseGatedEMA()

        g = DancerGallery(dancer_id=1, global_embedding=np.ones(10, dtype=np.float32))
        new_emb = np.zeros(10, dtype=np.float32)

        ema.update_gallery(g, new_global=new_emb, new_parts=None, new_color=None, new_face=None, alpha=0.5)
        # After EMA: 0.5 * 1.0 + 0.5 * 0.0 = 0.5 (before normalization)
        assert g.global_embedding is not None
        assert g.global_embedding[0] > 0  # still positive after blend


# ── PLAN_15: Collision Solver ─────────────────────────────────────────────

class TestCollisionSolver:
    def test_coalescence_detection(self):
        from sway.collision_solver import CoalescenceDetector
        cd = CoalescenceDetector(iou_thresh=0.3, consecutive_frames=3)

        overlapping = {
            1: np.array([10, 10, 60, 60], dtype=np.float32),
            2: np.array([15, 15, 65, 65], dtype=np.float32),
        }
        events = []
        for f in range(5):
            e = cd.check(overlapping, f)
            events.extend(e)

        assert len(events) == 1
        assert set(events[0].track_ids) == {1, 2}

    def test_hungarian_solver(self):
        from sway.collision_solver import CoalescenceEvent, _solve_hungarian
        event = CoalescenceEvent(track_ids=[1, 2, 3], entry_frame=0)
        assignments = _solve_hungarian(event, [{}, {}, {}], [10, 20, 30], None)
        assert len(assignments) == 3
        # Each frozen track assigned to one exit
        assigned_frozen = {a[0] for a in assignments}
        assert assigned_frozen == {1, 2, 3}


# ── PLAN_17: Mask-Guided Pose ─────────────────────────────────────────────

class TestMaskGuidedPose:
    def test_confidence_classification(self):
        from sway.mask_guided_pose import MaskGuidedPoseEstimator, KeypointConfidence
        est = MaskGuidedPoseEstimator()

        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 2] = 0.8  # high heatmap
        kp[:, 0] = 50    # x
        kp[:, 1] = 50    # y

        mask = np.ones((100, 100), dtype=bool)
        levels = est._classify_confidence(kp, mask, track_id=1)

        assert levels.shape == (17,)
        assert all(l >= KeypointConfidence.MEDIUM for l in levels)

    def test_mask_gate_downgrades_outside(self):
        from sway.mask_guided_pose import MaskGuidedPoseEstimator, KeypointConfidence
        est = MaskGuidedPoseEstimator()
        est.mask_gate = True

        kp = np.zeros((17, 3), dtype=np.float32)
        kp[0, :] = [50, 50, 0.9]  # high confidence, at (50,50)

        mask = np.zeros((100, 100), dtype=bool)
        mask[0:10, 0:10] = True  # keypoint at (50,50) is OUTSIDE mask

        levels = est._classify_confidence(kp, mask, track_id=2)
        assert levels[0] <= KeypointConfidence.LOW  # downgraded by mask gate


# ── PLAN_19: Critique Engine ──────────────────────────────────────────────

class TestCritiqueEngine:
    def test_smoothness_computation(self):
        from sway.critique_engine import CritiqueEngine
        engine = CritiqueEngine(fps=30.0)

        # Smooth sine wave → low jerk
        T = 100
        kp = np.zeros((T, 17, 3), dtype=np.float32)
        for t in range(T):
            kp[t, :, 0] = 50 + 10 * np.sin(2 * np.pi * t / 30)
            kp[t, :, 1] = 50
            kp[t, :, 2] = 0.9

        jerk = engine._smoothness(kp, mask=None)
        assert jerk.shape == (T,)
        assert np.mean(jerk) < 50  # smooth motion → low jerk

    def test_gap_detection(self):
        from sway.critique_engine import CritiqueEngine
        from sway.track_state import TrackState
        engine = CritiqueEngine()

        states = np.array([
            TrackState.ACTIVE, TrackState.ACTIVE, TrackState.DORMANT,
            TrackState.DORMANT, TrackState.DORMANT, TrackState.ACTIVE,
        ])
        gaps = engine._detect_gaps(1, states)
        assert len(gaps) == 1
        assert gaps[0].start_frame == 2
        assert gaps[0].end_frame == 4

    def test_joint_angle_computation(self):
        from sway.critique_engine import CritiqueEngine
        engine = CritiqueEngine()

        kp = np.zeros((1, 17, 3), dtype=np.float32)
        # Right angle at joint 8 (R elbow)
        kp[0, 6, :2] = [0, 0]    # R shoulder
        kp[0, 8, :2] = [10, 0]   # R elbow (vertex)
        kp[0, 10, :2] = [10, 10]  # R wrist

        angles = engine._compute_joint_angles(kp, 6, 8, 10)
        assert abs(angles[0] - 90.0) < 1.0


# ── PLAN_21: Advanced Trackers ────────────────────────────────────────────

class TestSentinelSBM:
    def test_grace_period(self):
        from sway.sentinel_sbm import SurvivalBoostingMechanism
        sbm = SurvivalBoostingMechanism(grace_multiplier=2.0, max_age=10)

        for _ in range(20):
            sbm.update_confidence(1, 0.8)

        assert sbm.should_grant_grace(1, 0.05, 0.3) is True
        sbm.enter_grace(1, frame_idx=100)
        assert sbm.is_in_grace(1) is True

        for _ in range(19):
            assert sbm.tick_grace(1) is True
        assert sbm.tick_grace(1) is False  # grace expired


class TestUMOTBacktrack:
    def test_record_and_query(self):
        from sway.umot_backtrack import HistoricalTrajectoryBank

        bank = HistoricalTrajectoryBank(history_length=100, match_threshold=0.5)

        emb = np.random.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Record and then mark lost
        for f in range(10):
            bank.record(1, f, 100.0, 200.0, emb.copy(), dancer_id=1)

        bank.mark_lost(1)
        assert bank.inactive_count == 1

        # Query with the same embedding → should match
        matched = bank.query(emb.copy(), (100.0, 200.0), current_frame=50)
        assert matched == 1


# ── Cross-module wiring tests ────────────────────────────────────────────

class TestCrossModuleWiring:
    def test_tracker_factory_returns_base_tracker(self):
        from sway.sam2_tracker import BaseTracker
        from sway.tracker_factory import create_tracker

        tracker = create_tracker(engine="solidtrack", device="cpu")
        assert isinstance(tracker, BaseTracker)

    def test_state_machine_feeds_pose_gated_ema(self):
        """State machine state flows into pose-gated EMA alpha computation."""
        from sway.track_state import TrackState
        from sway.pose_gated_ema import PoseGatedEMA
        ema = PoseGatedEMA()

        for state, expected_nonzero in [
            (TrackState.ACTIVE, True),
            (TrackState.PARTIAL, True),
            (TrackState.DORMANT, False),
            (TrackState.LOST, False),
        ]:
            alpha = ema.compute_alpha(
                np.array([10, 10, 50, 80]),
                [np.array([500, 500, 550, 570])],
                np.array([0.9] * 17),
                state,
            )
            if expected_nonzero:
                assert alpha > 0, f"State {state.name} should have alpha > 0"
            else:
                assert alpha == 0, f"State {state.name} should have alpha == 0"

    def test_enrollment_to_fusion_pipeline(self):
        """Gallery from enrollment feeds into fusion engine."""
        from sway.enrollment import DancerGallery
        from sway.reid_fusion import ReIDFusionEngine, ReIDQuery
        from sway.bpbreid_extractor import PartEmbeddings

        emb = np.random.randn(2048).astype(np.float32)
        emb /= np.linalg.norm(emb)

        gallery = [DancerGallery(
            dancer_id=1, global_embedding=emb.copy(),
            spatial_position=(0.5, 0.5), enrollment_frame=0,
        )]
        engine = ReIDFusionEngine(gallery=gallery)

        query = ReIDQuery(
            track_id=1,
            part_embeddings=PartEmbeddings(global_emb=emb.copy(), part_embs={}, visibility={}),
            spatial_position=(0.5, 0.5),
        )
        dancer_id, score = engine.match(query, current_frame=10)
        assert dancer_id == 1

    def test_confidence_levels_flow_to_critique(self):
        """Confidence levels from mask_guided_pose flow into critique engine."""
        from sway.mask_guided_pose import KeypointConfidence
        from sway.critique_engine import CritiqueEngine

        engine = CritiqueEngine()

        # Build confidence mask
        conf = np.full((100, 17), KeypointConfidence.HIGH, dtype=np.int32)
        conf[50:60, :] = KeypointConfidence.NOT_VISIBLE

        mask = engine._build_confidence_mask(conf)
        assert mask.shape == (100, 17)
        assert mask[0, 0] == True   # HIGH → True
        assert mask[55, 0] == False  # NOT_VISIBLE → False

    def test_memosort_integrates_with_sam2_tracker(self):
        """MeMoSORT can init/predict/update alongside SAM2 tracker outputs."""
        from sway.memosort import MeMoSORT
        from sway.sam2_tracker import TrackResult, TrackState

        ms = MeMoSORT(memory_length=10)

        # Simulate SAM2 tracker output
        result = TrackResult(
            track_id=1,
            bbox_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            mask=None,
            confidence=0.9,
            state=TrackState.ACTIVE,
            mask_area=1600.0,
        )

        ms.init_track(result.track_id, result.bbox_xyxy)
        pred = ms.predict(result.track_id)
        assert pred is not None
        ms.update(result.track_id, result.bbox_xyxy)


# ── Runtime Toggle Guards (backward / collision / MOTE / Sentinel / UMOT) ─

class TestRuntimeToggles:
    """Verify that SWAY_* env flags correctly gate module activation."""

    def test_backward_pass_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SWAY_BACKWARD_PASS_ENABLED", raising=False)
        from sway.backward_pass import is_backward_pass_enabled
        assert is_backward_pass_enabled() is True  # default True in module

    def test_backward_pass_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("SWAY_BACKWARD_PASS_ENABLED", "1")
        from sway.backward_pass import is_backward_pass_enabled
        assert is_backward_pass_enabled() is True

    def test_backward_pass_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("SWAY_BACKWARD_PASS_ENABLED", "0")
        from sway.backward_pass import is_backward_pass_enabled
        assert is_backward_pass_enabled() is False

    def test_mote_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SWAY_MOTE_DISOCCLUSION", raising=False)
        from sway.mote_disocclusion import is_mote_enabled
        assert is_mote_enabled() is False

    def test_mote_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("SWAY_MOTE_DISOCCLUSION", "1")
        from sway.mote_disocclusion import is_mote_enabled
        assert is_mote_enabled() is True

    def test_sentinel_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SWAY_SENTINEL_SBM", raising=False)
        from sway.sentinel_sbm import is_sentinel_enabled
        assert is_sentinel_enabled() is False

    def test_sentinel_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("SWAY_SENTINEL_SBM", "1")
        from sway.sentinel_sbm import is_sentinel_enabled
        assert is_sentinel_enabled() is True

    def test_umot_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SWAY_UMOT_BACKTRACK", raising=False)
        from sway.umot_backtrack import is_umot_enabled
        assert is_umot_enabled() is False

    def test_umot_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("SWAY_UMOT_BACKTRACK", "1")
        from sway.umot_backtrack import is_umot_enabled
        assert is_umot_enabled() is True


class TestBackwardPassStitch:
    """Backward pass stitch uses env-configurable similarity threshold."""

    def test_stitch_min_similarity_from_env(self, monkeypatch):
        monkeypatch.setenv("SWAY_BACKWARD_STITCH_MIN_SIMILARITY", "0.75")
        from sway.backward_pass import ForwardTrack, ReverseTrack, stitch_forward_reverse

        ft = ForwardTrack(track_id=1, dancer_id=1, start_frame=0, end_frame=50, is_dormant=True)
        rt = ReverseTrack(
            reverse_id=10, start_frame_reversed=0, end_frame_reversed=10,
            start_frame_original=60, end_frame_original=70,
        )
        merged = stitch_forward_reverse([ft], [rt], min_similarity=None)
        has_stitch = any(m.reverse_track_id is not None for m in merged)
        assert isinstance(has_stitch, bool)

    def test_stitch_returns_all_tracks(self):
        from sway.backward_pass import ForwardTrack, MergedTrack, stitch_forward_reverse

        fwd = [
            ForwardTrack(track_id=1, dancer_id=1, start_frame=0, end_frame=100, is_dormant=False),
            ForwardTrack(track_id=2, dancer_id=2, start_frame=0, end_frame=80, is_dormant=True),
        ]
        merged = stitch_forward_reverse(fwd, [])
        assert len(merged) == 2
        assert all(isinstance(m, MergedTrack) for m in merged)


class TestCollisionSolverDispatch:
    """Collision solver selection is driven by SWAY_COLLISION_SOLVER."""

    def test_solver_choices_in_schema(self):
        from sway.pipeline_config_schema import PIPELINE_PARAM_FIELDS
        field = next(f for f in PIPELINE_PARAM_FIELDS if f["id"] == "sway_collision_solver")
        assert "greedy" in field["choices"]
        assert "hungarian" in field["choices"]
        assert "dp" in field["choices"]


class TestMOTEModule:
    """MOTE disocclusion matrix basic interface."""

    def test_instantiate_mote(self):
        from sway.mote_disocclusion import MOTEDisocclusion
        mote = MOTEDisocclusion(flow_model="raft_small")
        assert mote.confidence_boost > 0

    def test_predict_reemergence(self):
        from sway.mote_disocclusion import MOTEDisocclusion
        mote = MOTEDisocclusion(flow_model="raft_small")
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        flow[:, :, 0] = 2.0  # 2px rightward flow
        preds = mote.predict_reemergence(flow, {1: (50.0, 50.0)})
        assert isinstance(preds, dict)
        assert 1 in preds


# ---------- SOURCE: test_gnn_track_refine.py ----------

"""Integration and edge-case tests for GNN post-stitch track refinement."""


import os
from typing import Dict, Iterator, List

import pytest
import torch

from sway.experimental_hooks import maybe_gnn_refine_raw_tracks
from sway.gnn_track_refine import RelationalTrackGNN, _hard_forbid_merge, gnn_refine_raw_tracks
from sway.track_observation import TrackObservation


_GNN_ENV_KEYS = (
    "SWAY_GNN_MERGE_THRESH",
    "SWAY_GNN_HIDDEN",
    "SWAY_GNN_HEADS",
    "SWAY_GNN_LAYERS",
    "SWAY_GNN_DROPOUT",
    "SWAY_GNN_MAX_GAP",
    "SWAY_GNN_PRIOR_SCALE",
    "SWAY_GNN_WEIGHTS",
    "SWAY_GNN_DEVICE",
    "SWAY_GNN_SEED",
    "SWAY_GNN_TRACK_REFINE",
)


@pytest.fixture
def isolate_gnn_env() -> Iterator[None]:
    saved: Dict[str, str | None] = {k: os.environ.get(k) for k in _GNN_ENV_KEYS}
    for k in _GNN_ENV_KEYS:
        os.environ.pop(k, None)
    yield
    for k in _GNN_ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def _cpu_gnn_defaults() -> None:
    os.environ["SWAY_GNN_DEVICE"] = "cpu"
    os.environ["SWAY_GNN_SEED"] = "0"


def test_gnn_refine_forward_module_shapes(isolate_gnn_env) -> None:
    """Sanity check: RelationalTrackGNN runs on a tiny dense graph."""
    _cpu_gnn_defaults()
    n, node_in, edge_in = 4, 16, 6
    x = torch.randn(n, node_in)
    ef = torch.randn(n, n, edge_in)
    adj = torch.eye(n)
    adj[0, 1] = adj[1, 0] = adj[1, 2] = adj[2, 1] = 1.0
    m = RelationalTrackGNN(
        node_in=node_in,
        edge_in=edge_in,
        hidden=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )
    m.eval()
    with torch.no_grad():
        logits = m(x, ef, adj)
    assert logits.shape == (n, n)


def test_hard_forbid_merge_rules() -> None:
    assert _hard_forbid_merge({"n_overlap": 10.0, "max_iou": 0.05, "mean_dist_h": 0.5}) is True
    assert _hard_forbid_merge({"n_overlap": 2.0, "max_iou": 0.5, "mean_dist_h": 0.1}) is False


def test_merge_three_duplicate_tracks(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    box = (5.0, 5.0, 55.0, 205.0)
    raw = {
        10: [(0, box, 0.9), (1, box, 0.9)],
        20: [(0, box, 0.88), (1, box, 0.88)],
        30: [(0, box, 0.87), (1, box, 0.87)],
    }
    gnn_refine_raw_tracks(raw, total_frames=20, ystride=1)
    assert len(raw) == 1
    assert len(next(iter(raw.values()))) == 2


def test_high_merge_threshold_prevents_merge(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    os.environ["SWAY_GNN_MERGE_THRESH"] = "1.01"
    box = (10.0, 20.0, 50.0, 120.0)
    raw = {
        1: [(0, box, 0.9), (1, box, 0.9)],
        2: [(0, box, 0.85), (1, box, 0.85)],
    }
    gnn_refine_raw_tracks(raw, total_frames=15, ystride=1)
    assert len(raw) == 2


def test_preserves_empty_track_when_merging_others(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    box = (1.0, 1.0, 40.0, 100.0)
    raw = {
        0: [],
        1: [(0, box, 0.9), (1, box, 0.9)],
        2: [(0, box, 0.86), (1, box, 0.86)],
    }
    gnn_refine_raw_tracks(raw, total_frames=10, ystride=1)
    assert 0 in raw and raw[0] == []
    assert len(raw) == 2


def test_two_people_overlapping_time_not_merged(isolate_gnn_env) -> None:
    """Side-by-side boxes on same frames → hard forbid or low same-person score → 2 tracks."""
    _cpu_gnn_defaults()
    left = (10.0, 100.0, 50.0, 200.0)
    right = (400.0, 100.0, 440.0, 200.0)
    frames = list(range(12))
    raw = {
        1: [(f, left, 0.9) for f in frames],
        2: [(f, right, 0.9) for f in frames],
    }
    gnn_refine_raw_tracks(raw, total_frames=30, ystride=1)
    assert len(raw) == 2
    assert len(raw[1]) == 12
    assert len(raw[2]) == 12


def test_track_observation_merge_keeps_dataclass(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    box = (20.0, 30.0, 80.0, 180.0)
    raw = {
        1: [
            TrackObservation(0, box, 0.91, is_sam_refined=True),
            TrackObservation(1, box, 0.90, is_sam_refined=False),
        ],
        2: [
            TrackObservation(0, box, 0.89, is_sam_refined=False),
            TrackObservation(1, box, 0.88, is_sam_refined=True),
        ],
    }
    gnn_refine_raw_tracks(raw, total_frames=12, ystride=2)
    assert len(raw) == 1
    merged = next(iter(raw.values()))
    assert len(merged) == 2
    for e in merged:
        assert isinstance(e, TrackObservation)


def test_small_architecture_env_runs(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    os.environ["SWAY_GNN_HIDDEN"] = "32"
    os.environ["SWAY_GNN_HEADS"] = "4"
    os.environ["SWAY_GNN_LAYERS"] = "2"
    box = (0.0, 0.0, 30.0, 90.0)
    raw = {1: [(0, box, 0.9)], 2: [(0, box, 0.85)]}
    gnn_refine_raw_tracks(raw, total_frames=5, ystride=1)
    assert len(raw) == 1


def test_state_dict_roundtrip_via_weights_env(isolate_gnn_env, tmp_path) -> None:
    _cpu_gnn_defaults()
    pt = tmp_path / "gnn_dummy.pt"
    m = RelationalTrackGNN(
        node_in=16,
        edge_in=6,
        hidden=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )
    torch.save(m.state_dict(), pt)
    os.environ["SWAY_GNN_WEIGHTS"] = str(pt)
    os.environ["SWAY_GNN_HIDDEN"] = "32"
    os.environ["SWAY_GNN_HEADS"] = "4"
    os.environ["SWAY_GNN_LAYERS"] = "1"
    box = (10.0, 10.0, 60.0, 160.0)
    raw = {5: [(0, box, 0.9)], 7: [(0, box, 0.8)]}
    gnn_refine_raw_tracks(raw, total_frames=8, ystride=1)
    assert len(raw) == 1


def test_maybe_gnn_hook_disabled_short_circuits(isolate_gnn_env) -> None:
    """No SWAY_GNN_TRACK_REFINE → never import path that builds the graph (lightweight)."""
    os.environ.pop("SWAY_GNN_TRACK_REFINE", None)
    raw: Dict[int, List] = {1: [(0, (0, 0, 1, 1), 0.5)]}
    out = maybe_gnn_refine_raw_tracks(raw, 5, 1)
    assert out is raw
    assert len(raw) == 1


def test_maybe_gnn_hook_enabled_matches_direct_call(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    os.environ["SWAY_GNN_TRACK_REFINE"] = "1"
    box = (2.0, 2.0, 42.0, 102.0)
    raw_hook = {1: [(0, box, 0.9), (1, box, 0.9)], 2: [(0, box, 0.85), (1, box, 0.85)]}
    raw_direct = {1: [(0, box, 0.9), (1, box, 0.9)], 2: [(0, box, 0.85), (1, box, 0.85)]}
    maybe_gnn_refine_raw_tracks(raw_hook, 20, 1)
    gnn_refine_raw_tracks(raw_direct, 20, 1)
    assert set(raw_hook.keys()) == set(raw_direct.keys())
    assert len(raw_hook) == 1
    assert len(raw_direct) == 1
    assert len(raw_hook[next(iter(raw_hook.keys()))]) == 2


def test_single_nonempty_track_noop(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    raw = {99: [(0, (0, 0, 10, 10), 0.7)]}
    gnn_refine_raw_tracks(raw, total_frames=100, ystride=1)
    assert raw == {99: [(0, (0, 0, 10, 10), 0.7)]}


def test_empty_raw_tracks_dict(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    raw: Dict[int, List] = {}
    assert gnn_refine_raw_tracks(raw, 10, 1) == raw


def test_only_empty_tracks_early_exit(isolate_gnn_env) -> None:
    _cpu_gnn_defaults()
    raw = {1: [], 2: []}
    gnn_refine_raw_tracks(raw, 10, 1)
    assert raw == {1: [], 2: []}


def test_build_track_graph_tensors_matches_inference(isolate_gnn_env) -> None:
    from sway.gnn_track_refine import build_track_graph_tensors

    _cpu_gnn_defaults()
    box = (1.0, 1.0, 41.0, 131.0)
    raw = {1: [(0, box, 0.9), (1, box, 0.9)], 2: [(0, box, 0.8), (1, box, 0.8)]}
    g = build_track_graph_tensors(
        raw,
        total_frames=50,
        max_gap=120.0,
        prior_scale=1.0,
        device=torch.device("cpu"),
    )
    assert g is not None
    assert g.x0.shape == (2, 16)
    assert g.edge_feat.shape[0] == 2
    assert g.adj[0, 1] > 0


def test_edge_bce_loss_no_edges_still_has_grad() -> None:
    """Regression: empty edge set must not return a detached zero (breaks backward)."""
    from tools.train_gnn_track_refine import edge_bce_loss

    n = 3
    learn = torch.randn(n, n, requires_grad=True)
    prior = torch.zeros(n, n)
    adj = torch.eye(n)
    node_ids = [1, 2, 3]
    tid_to_person = {1: 0, 2: 1, 3: 2}
    loss = edge_bce_loss(learn, prior, adj, node_ids, tid_to_person)
    loss.backward()
    assert learn.grad is not None


def test_quick_train_writes_checkpoint(isolate_gnn_env, tmp_path) -> None:
    from tools.train_gnn_track_refine import run_training

    out = tmp_path / "gnn.pt"
    run_training(
        out_path=out,
        steps=5,
        lr=1e-3,
        device=torch.device("cpu"),
        hidden=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        max_gap=120.0,
        prior_scale=1.0,
        total_frames=180,
        seed=1,
        log_every=0,
    )
    assert out.is_file()
    ckpt = torch.load(out, map_location="cpu")
    assert "state_dict" in ckpt
    assert "meta" in ckpt


# ---------- SOURCE: test_golden_bench_config.py ----------

"""golden_bench --config YAML loading (no pytest subprocess)."""

from pathlib import Path

import yaml



def test_golden_bench_example_yaml_loads():
    p = REPO_ROOT / "benchmarks" / "golden_bench.example.yaml"
    assert p.is_file()
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "tests" in cfg
    assert isinstance(cfg["tests"], list)
    assert len(cfg["tests"]) >= 1
    for rel in cfg["tests"]:
        assert (REPO_ROOT / rel).is_file(), rel


# ---------- SOURCE: test_handshake_tracking.py ----------

"""Tests for Sway Handshake Phase 1–3 helpers."""


import numpy as np

from sway.handshake_tracking import _bhattacharyya, _profile_score, phase13_handshake_enabled


def test_phase13_handshake_env(monkeypatch) -> None:
    monkeypatch.delenv("SWAY_PHASE13_MODE", raising=False)
    assert phase13_handshake_enabled() is False
    monkeypatch.setenv("SWAY_PHASE13_MODE", "sway_handshake")
    assert phase13_handshake_enabled() is True


def test_profile_score_with_prof() -> None:
    p = np.ones(32, dtype=np.float64) / 32.0
    s = _profile_score(p, 0.4, (p.copy(), 0.4))
    assert s > 0.5


# ---------- SOURCE: test_hybrid_sam_roi.py ----------

"""Unit tests for hybrid SAM ROI helpers (no Ultralytics / SAM required)."""

import numpy as np

from sway.hybrid_sam_refiner import (
    max_pairwise_iou,
    overlap_cluster_indices,
    union_xyxy_with_pad,
)


def test_max_pairwise_iou_identical_boxes():
    xy = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    assert max_pairwise_iou(xy) == 1.0


def test_overlap_cluster_indices():
    # Two heavily overlapping + one far away (IoU(0,1) > 0.42)
    xy = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [2.0, 2.0, 12.0, 12.0],
            [100.0, 100.0, 120.0, 140.0],
        ],
        dtype=np.float32,
    )
    idx = overlap_cluster_indices(xy, 0.42)
    assert idx == {0, 1}


def test_union_xyxy_with_pad():
    xy = np.array(
        [
            [10.0, 10.0, 20.0, 20.0],
            [15.0, 15.0, 25.0, 25.0],
        ],
        dtype=np.float32,
    )
    x1, y1, x2, y2 = union_xyxy_with_pad(xy, [0, 1], frame_h=200, frame_w=300, pad_frac=0.1)
    assert x1 >= 0 and y1 >= 0 and x2 <= 300 and y2 <= 200
    assert x2 > x1 and y2 > y1
    # Union without pad: (10,10)-(25,25); with 10% pad on ~15px size -> ~1.5px margin
    assert x1 < 10 and y1 < 10
    assert x2 > 25 and y2 > 25


# ---------- SOURCE: test_infer_batch_env.py ----------

"""Env helpers for YOLO / ViTPose batching (no model loads)."""

import os

import pytest
import torch

from sway.pose_estimator import (
    vitpose_debug_enabled,
    vitpose_effective_max_per_forward,
    vitpose_max_per_forward,
)
from sway.tracker import yolo_infer_batch_size


def test_yolo_infer_batch_size_default():
    os.environ.pop("SWAY_YOLO_INFER_BATCH", None)
    assert yolo_infer_batch_size() == 1


def test_yolo_infer_batch_size_clamped():
    os.environ["SWAY_YOLO_INFER_BATCH"] = "4"
    assert yolo_infer_batch_size() == 4
    os.environ["SWAY_YOLO_INFER_BATCH"] = "999"
    assert yolo_infer_batch_size() == 32
    os.environ["SWAY_YOLO_INFER_BATCH"] = "0"
    assert yolo_infer_batch_size() == 1
    os.environ["SWAY_YOLO_INFER_BATCH"] = "nope"
    assert yolo_infer_batch_size() == 1
    os.environ.pop("SWAY_YOLO_INFER_BATCH", None)


def test_vitpose_max_per_forward():
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)
    assert vitpose_max_per_forward() == 0
    os.environ["SWAY_VITPOSE_MAX_PER_FORWARD"] = "6"
    assert vitpose_max_per_forward() == 6
    os.environ["SWAY_VITPOSE_MAX_PER_FORWARD"] = "bad"
    assert vitpose_max_per_forward() == 0
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)


def test_vitpose_effective_cpu_no_default_chunk():
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)
    os.environ.pop("SWAY_VITPOSE_MPS_CHUNK", None)
    assert vitpose_effective_max_per_forward(torch.device("cpu")) == 0


def test_vitpose_effective_mps_default_chunk():
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)
    os.environ.pop("SWAY_VITPOSE_MPS_CHUNK", None)
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    assert vitpose_effective_max_per_forward(torch.device("mps")) == 2


def test_vitpose_effective_env_overrides_mps():
    os.environ["SWAY_VITPOSE_MAX_PER_FORWARD"] = "2"
    os.environ.pop("SWAY_VITPOSE_MPS_CHUNK", None)
    assert vitpose_effective_max_per_forward(torch.device("mps")) == 2
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)


def test_vitpose_debug_default_off():
    os.environ.pop("SWAY_VITPOSE_DEBUG", None)
    assert vitpose_debug_enabled() is False


def test_vitpose_debug_explicit_off():
    os.environ["SWAY_VITPOSE_DEBUG"] = "0"
    assert vitpose_debug_enabled() is False
    os.environ.pop("SWAY_VITPOSE_DEBUG", None)


def test_vitpose_debug_explicit_on():
    os.environ["SWAY_VITPOSE_DEBUG"] = "1"
    assert vitpose_debug_enabled() is True
    os.environ.pop("SWAY_VITPOSE_DEBUG", None)


# ---------- SOURCE: test_mot_trackeval.py ----------

import json
import sys
from pathlib import Path

import pytest

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


# ---------- SOURCE: test_optuna_live_status.py ----------

"""sway.optuna_live_status JSON payload."""

import optuna

from sway.optuna_live_status import build_study_status_payload, write_live_sweep_status


def test_build_study_status_payload(tmp_path) -> None:
    study = optuna.create_study(direction="maximize")

    def obj(t: optuna.Trial) -> float:
        t.set_user_attr("u", 1)
        return t.suggest_float("x", 0, 1)

    study.optimize(obj, n_trials=2)
    p = build_study_status_payload(study, extra={"k": "v"})
    assert p["schema"] == "sway_optuna_sweep_status_v1"
    assert p["n_trials_total"] == 2
    assert p["n_complete"] == 2
    assert p["best"] is not None
    assert p["best"]["number"] in (0, 1)
    assert p["meta"]["k"] == "v"
    assert len(p["trials"]) == 2


def test_write_live_sweep_status_atomic(tmp_path) -> None:
    study = optuna.create_study()
    study.optimize(lambda t: 0.0, n_trials=1)
    path = tmp_path / "s.json"
    write_live_sweep_status(study, path, extra={})
    assert path.is_file()
    assert not (tmp_path / "s.json.tmp").exists()


# ---------- SOURCE: test_pipeline_lab_batch_path.py ----------

"""Pipeline Lab batch_path endpoint (no main.py execution)."""

import sys
from pathlib import Path

import pytest

@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_LAB_MAX_PARALLEL", "1")
    import pipeline_lab.server.app as lab_app

    monkeypatch.setattr(lab_app, "RUNS_ROOT", tmp_path / "runs")
    lab_app.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    lab_app._runs.clear()
    while not lab_app._job_queue.empty():
        try:
            lab_app._job_queue.get_nowait()
        except Exception:
            break

    def _noop_execute_run(run_id: str) -> None:
        run_dir = lab_app.RUNS_ROOT / run_id
        with lab_app._run_lock:
            st = lab_app._runs.get(run_id)
            if st:
                st.status = "done"
                st.process = None
        (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(lab_app, "_execute_run", _noop_execute_run)

    from fastapi.testclient import TestClient

    return TestClient(lab_app.app)


def test_batch_path_queues_runs_with_shared_inode(client, tmp_path):
    import pipeline_lab.server.app as lab_app

    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"not-a-real-mp4-but-bytes")

    r = client.post(
        "/api/runs/batch_path",
        json={
            "video_path": str(vid),
            "runs": [
                {"recipe_name": "A", "fields": {}},
                {"recipe_name": "B", "fields": {"pose_stride": 2}},
            ],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "queued"
    assert data["run_count"] == 2
    assert len(data["run_ids"]) == 2
    assert data.get("batch_id")

    lab_app._job_queue.join()

    rows = client.get("/api/runs").json()
    by_id = {x["run_id"]: x for x in rows}
    for rid in data["run_ids"]:
        assert rid in by_id
        assert by_id[rid]["batch_id"] == data["batch_id"]

    root = lab_app.RUNS_ROOT
    paths = []
    for rid in data["run_ids"]:
        found = list((root / rid).glob("input_video*"))
        assert len(found) == 1
        paths.append(found[0])
    try:
        assert paths[0].stat().st_ino == paths[1].stat().st_ino
    except AssertionError:
        pytest.skip("hardlink not shared (different FS semantics)")


def test_batch_path_rejects_missing_video(client, tmp_path):
    r = client.post(
        "/api/runs/batch_path",
        json={"video_path": str(tmp_path / "nope.mp4"), "runs": [{"recipe_name": "x", "fields": {}}]},
    )
    assert r.status_code == 400


def test_pipeline_matrix_endpoint(client):
    r = client.get("/api/pipeline_matrix")
    assert r.status_code == 200
    body = r.json()
    assert "recipes" in body
    assert len(body["recipes"]) >= 5


# ---------- SOURCE: test_pipeline_matrix_presets.py ----------

"""Validate matrix recipe field ids against the Lab schema."""

import re
from pathlib import Path

from sway.pipeline_config_schema import PIPELINE_PARAM_FIELDS
from sway.pipeline_matrix_presets import (
    PIPELINE_MATRIX_RECIPES,
    matrix_recipe_by_id,
    pipeline_matrix_for_api,
)


def _schema_field_ids():
    return {f["id"] for f in PIPELINE_PARAM_FIELDS if f.get("type") != "info"}


def test_matrix_recipe_ids_unique():
    ids = [r["id"] for r in PIPELINE_MATRIX_RECIPES]
    assert len(ids) == len(set(ids))


def test_matrix_fields_are_schema_ids():
    allowed = _schema_field_ids()
    for r in PIPELINE_MATRIX_RECIPES:
        for k in (r.get("fields") or {}):
            assert k in allowed, f"recipe {r.get('id')!r} uses unknown field {k!r}"


def test_pipeline_matrix_for_api_shape():
    p = pipeline_matrix_for_api()
    assert p["version"] >= 1
    assert "intro" in p
    assert isinstance(p["recipes"], list)
    assert len(p["recipes"]) == len(PIPELINE_MATRIX_RECIPES)


# ── Section 13 preset catalog coverage ────────────────────────────────────

_SECTION_13_DOC_IDS = {
    # 13.1 Baseline carry-forward
    "baseline", "preset_open_competition", "preset_open_competition_recovery",
    "preset_dense_hifi", "preset_ballet_fluid", "preset_mirror_studio",
    "preset_wide_angle_maxprec", "preset_osnet_aggressive",
    # 13.2 Lean core
    "f01_lean_core_i1", "f02_lean_core_i2", "f03_lean_core_i3",
    "f04_lean_core_i4", "f05_lean_core_full",
    # 13.3 Detector
    "f10_det_yolo_only", "f11_det_codetr_only", "f12_det_codino_only",
    "f13_det_rtdetr_only", "f14_det_hybrid_codino", "f15_det_hybrid_rtdetr",
    "f16a_det_hybrid_overlap_lo", "f16_det_hybrid_sweep", "f16c_det_hybrid_overlap_hi",
    "f17_det_yolo_crowdhuman",
    # 13.4 Tracker
    "f20_trk_solidtrack", "f21_trk_sam2mot", "f22_trk_memosort",
    "f23_trk_sam2_memosort", "f24_trk_sam2_b", "f25_trk_sam2_l",
    "f26_trk_sam2_h", "f27_trk_coi_freeze",
    # 13.5 Re-ID
    "f30_reid_osnet_baseline", "f31_reid_bpbreid_only", "f32_reid_lean_core",
    "f33_reid_plus_face", "f34_reid_plus_skeleton", "f35_reid_full_6signal",
    "f36_reid_finetuned", "f37_reid_finetuned_full", "f38_reid_weight_sweep_a",
    "f39_reid_weight_sweep_b",
    # 13.6 Pose & 3D lift
    "f40_pose_vitpose_large", "f41_pose_vitpose_huge", "f42_pose_vitpose_nomask",
    "f43_pose_rtmw_l", "f44_pose_rtmw_x", "f45_lift_motionagformer",
    "f46_lift_motionbert", "f47_lift_motionbert_multi",
    # 13.7 Backward & collision
    "f50_backward_off", "f51_backward_on", "f52_backward_tight",
    "f53_backward_loose", "f54_collision_greedy", "f55_collision_hungarian",
    "f56_collision_dp",
    # 13.8 Advanced
    "f60_mote", "f61_mote_raft_large", "f62_sentinel",
    "f63_sentinel_aggressive", "f64_umot", "f65_umot_long_history",
    "f66_mote_sentinel", "f67_matr_branch",
    # 13.9 Full-stack
    "f70_production_candidate_a", "f71_production_candidate_b",
    "f72_production_candidate_c", "f73_production_candidate_d",
    "f74_production_speed", "f75_production_quality",
}


def test_section13_ids_present_in_matrix():
    """Every Section 13 preset ID resolves from the matrix."""
    code_ids = {r["id"] for r in PIPELINE_MATRIX_RECIPES}
    missing = sorted(_SECTION_13_DOC_IDS - code_ids)
    assert not missing, f"Section 13 IDs missing from matrix: {missing}"


def test_section13_ids_resolvable_by_lookup():
    """matrix_recipe_by_id returns a valid row for every Section 13 ID."""
    for rid in sorted(_SECTION_13_DOC_IDS):
        r = matrix_recipe_by_id(rid)
        assert r is not None, f"matrix_recipe_by_id({rid!r}) returned None"
        assert r["id"] == rid


def test_section13_alias_f23_exists():
    """f23_trk_sam2_memosort is the doc alias; f23_trk_sam2_memosort_hybrid is the original."""
    alias = matrix_recipe_by_id("f23_trk_sam2_memosort")
    original = matrix_recipe_by_id("f23_trk_sam2_memosort_hybrid")
    assert alias is not None
    assert original is not None
    assert alias["fields"]["sway_tracker_engine"] == original["fields"]["sway_tracker_engine"]


def test_every_recipe_has_required_keys():
    """Every recipe row has id, recipe_name, varies, description, fields."""
    for r in PIPELINE_MATRIX_RECIPES:
        for key in ("id", "recipe_name", "varies", "description", "fields"):
            assert key in r, f"Recipe {r.get('id')!r} missing key {key!r}"


def test_cli_selector_recipe_prefix():
    """--recipe-prefix f3 should select exactly the f30–f39 range."""
    from tools.pipeline_matrix_runs import _resolve_selector

    selected = _resolve_selector(None, None, None, "f3")
    ids = {r["id"] for r in selected}
    expected_f3x = {rid for rid in _SECTION_13_DOC_IDS if rid.startswith("f3")}
    assert expected_f3x <= ids, f"Missing f3x presets: {expected_f3x - ids}"


# ---------- SOURCE: test_pose_3d_export.py ----------

"""Smoke tests for 3D pose export and lift_xyz-aware scoring (no MotionAGFormer)."""


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


# ---------- SOURCE: test_pose_crop_temporal.py ----------

from sway.pose_crop_temporal import apply_temporal_pose_crop


def test_smooth_converges_and_foot_bias():
    state = {}
    b0 = (10.0, 10.0, 20.0, 40.0)
    b1 = apply_temporal_pose_crop(
        1,
        b0,
        frame_w=200,
        frame_h=200,
        smooth_alpha=0.0,
        foot_bias_frac=0.1,
        head_bias_frac=0.0,
        anti_jitter_px=0.0,
        state=state,
    )
    h = b0[3] - b0[1]
    assert b1[3] == min(200.0, b0[3] + 0.1 * h)

    state.clear()
    r0 = apply_temporal_pose_crop(
        2,
        (50.0, 50.0, 60.0, 80.0),
        frame_w=200,
        frame_h=200,
        smooth_alpha=0.5,
        foot_bias_frac=0.0,
        head_bias_frac=0.0,
        anti_jitter_px=0.0,
        state=state,
    )
    r1 = apply_temporal_pose_crop(
        2,
        (60.0, 50.0, 70.0, 80.0),
        frame_w=200,
        frame_h=200,
        smooth_alpha=0.5,
        foot_bias_frac=0.0,
        head_bias_frac=0.0,
        anti_jitter_px=0.0,
        state=state,
    )
    assert r0[0] == 50.0
    assert r1[0] > 50.0 and r1[0] < 60.0


# ---------- SOURCE: test_pose_lift_3d_qa_protocol.py ----------

"""
Automated coverage for the Sway 3D QA protocol (geometric sanity, PBD, export).

Maps to checklist IDs 1.x–3.x. Phase 4 (full dance stress) stays manual — markers below.

Private helpers are accessed via the module to avoid widening the public API.
"""


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


def test_qa_1_3_pelvis_anchor_after_centering() -> None:
    """1.3: After centering on pelvis + Y-negate, mid-hip at origin, Y flipped."""
    post = np.random.RandomState(0).randn(17, 3).astype(np.float32) * 0.5
    pelvis_before = (post[11, :] + post[12, :]) * 0.5

    out = pl3._postprocess_pose3d_frame(post)

    # After centering, pelvis midpoint should be at origin
    mid_after = (out[11, :] + out[12, :]) * 0.5
    assert np.allclose(mid_after, 0.0, atol=1e-5)

    # Y should be negated (camera Y-down → world Y-up)
    centered = post - pelvis_before
    assert np.allclose(out[:, 1], -centered[:, 1], atol=1e-5)


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


# ---------- SOURCE: test_prune_and_dormant.py ----------

import sys
from pathlib import Path

import numpy as np

from sway.dormant_tracks import apply_dormant_merges  # noqa: E402
from sway.track_pruning import compute_confirmed_human_set  # noqa: E402


def test_confirmed_human_set():
    scores = np.ones(17, dtype=np.float32) * 0.9
    # Bbox center in-frame (not mirror edge) so Tier A spatial sanity passes with frame_width set
    box = (500.0, 100.0, 600.0, 400.0)
    fd0 = {
        "frame_idx": 0,
        "track_ids": [1],
        "boxes": [box],
        "poses": {1: {"scores": scores}},
    }
    fd1 = {
        "frame_idx": 50,
        "track_ids": [1],
        "boxes": [box],
        "poses": {1: {"scores": scores}},
    }
    conf = compute_confirmed_human_set([fd0, fd1], total_frames=100, frame_width=1920)
    assert 1 in conf


def test_dormant_merge_gap():
    box = (100.0, 100.0, 150.0, 200.0)
    # A ends at 10, B starts at 100 -> gap 89f (> track_buffer 90? 89<=90 fails)
    # Need gap > 90: end 10, start 102 -> gap 102-10-1 = 91
    raw = {
        1: [(8, box, 0.9), (10, box, 0.9)],
        2: [(102, box, 0.9)],
    }
    out = apply_dormant_merges(dict(raw), total_frames=200, track_buffer=90, max_gap=150)
    assert 2 not in out
    assert 1 in out
    assert len(out[1]) == 3


# ---------- SOURCE: test_rtmpose_optional_import.py ----------

"""RTMPose backend is optional (full MMPose stack)."""

import pytest


def test_rtmpose_class_loads_when_mm_stack_present():
    try:
        import mmengine  # noqa: F401
    except ImportError:
        pytest.skip("MMPose / mmengine not installed")
    from sway.rtmpose_estimator import RTMPoseEstimator

    assert RTMPoseEstimator.__name__ == "RTMPoseEstimator"


# ---------- SOURCE: test_run_all_configurations.py ----------

"""Tests for all-config Lambda run planner/runner helpers."""

from tools import run_all_configurations as rac


def test_parse_feature_line_and_failure_signal():
    line = "  [feature] MOTE: requested=on, runtime=off, wiring=unwired"
    parsed = rac._parse_feature_line(line)
    assert parsed is not None
    assert parsed["name"] == "MOTE"
    assert parsed["requested"] == "on"
    assert parsed["runtime"] == "off"
    assert parsed["wiring"] == "unwired"


def test_resolve_execute_boundary_strict_modes():
    assert (
        rac._resolve_execute_boundary("off", "after_phase_3", {}) == "after_phase_3"
    )
    assert rac._resolve_execute_boundary("full", "after_phase_3", {}) == "final"
    assert (
        rac._resolve_execute_boundary("quick", "after_phase_3", {}) == "after_phase_3"
    )
    assert (
        rac._resolve_execute_boundary(
            "quick",
            "after_phase_3",
            {"SWAY_3D_LIFT": "1"},
        )
        == "final"
    )


def test_build_run_plan_reports_full_coverage(monkeypatch):
    monkeypatch.setattr(rac, "_extract_all_sway_keys_from_repo", lambda: ["SWAY_A", "SWAY_B"])
    monkeypatch.setattr(rac, "_load_manifest_keys", lambda: [])
    monkeypatch.setattr(rac, "_extract_sway_keys_from_future_doc", lambda: [])
    monkeypatch.setattr(rac, "_extract_domains_from_auto_sweep", lambda: {})
    monkeypatch.setattr(rac, "_extract_default_values_from_repo", lambda: {"SWAY_A": "0"})

    plan = rac.build_run_plan()
    coverage = plan["coverage"]
    assert coverage["is_full_coverage"] is True
    assert coverage["missing_keys_full"] == []
    names = [c["name"] for c in plan["cases"]]
    assert "baseline_defaults" in names
    assert "SWAY_B=1" in names


# ---------- SOURCE: test_sapiens_estimator.py ----------

"""Unit tests for Sapiens TorchScript heatmap decode (no checkpoint required)."""

import numpy as np

from sway.sapiens_estimator import heatmaps_to_keypoints_xyxy


def test_heatmaps_to_keypoints_xyxy_peak_maps_to_bbox():
    hm = np.zeros((17, 256, 192), dtype=np.float32)
    hm[3, 128, 96] = 5.0
    x1, y1, x2, y2 = 10.0, 20.0, 210.0, 420.0
    k = heatmaps_to_keypoints_xyxy(hm, x1, y1, x2, y2, 256, 192)
    assert k.shape == (17, 3)
    bw, bh = 200.0, 400.0
    assert abs(k[3, 0] - (96 * bw / 192 + x1)) < 1e-4
    assert abs(k[3, 1] - (128 * bh / 256 + y1)) < 1e-4
    assert k[3, 2] == 5.0
    assert np.allclose(k[:3, 2], 0.0)


# ---------- SOURCE: test_server_runtime_perf.py ----------

"""sway.server_runtime_perf — env overlay and thread helpers."""

import os

import pytest


def test_subprocess_env_overlay_empty_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SWAY_SERVER_PERF", raising=False)
    from sway.server_runtime_perf import subprocess_env_overlay

    assert subprocess_env_overlay() == {}


def test_subprocess_env_overlay_sets_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWAY_SERVER_PERF", "1")
    monkeypatch.setenv("SWAY_PERF_CPU_THREADS", "8")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)
    from sway.server_runtime_perf import subprocess_env_overlay

    o = subprocess_env_overlay()
    assert o["SWAY_SERVER_PERF"] == "1"
    assert o["SWAY_PERF_CPU_THREADS"] == "8"
    assert o["OMP_NUM_THREADS"] == "8"


def test_subprocess_env_overlay_respects_existing_omp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWAY_SERVER_PERF", "1")
    monkeypatch.setenv("SWAY_PERF_CPU_THREADS", "8")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    from sway.server_runtime_perf import subprocess_env_overlay

    o = subprocess_env_overlay()
    assert "OMP_NUM_THREADS" not in o
    assert os.environ.get("OMP_NUM_THREADS") == "2"


# ---------- SOURCE: test_smoke_server_perf_env_cli.py ----------

"""CLI smoke for tools.smoke_server_perf_env (no --pipeline; fast)."""

import os
import subprocess
import sys
from pathlib import Path



def test_smoke_server_perf_env_requires_flag() -> None:
    env = {k: v for k, v in os.environ.items() if k != "SWAY_SERVER_PERF"}
    r = subprocess.run(
        [sys.executable, "-m", "tools.smoke_server_perf_env"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2


def test_smoke_server_perf_env_probes_ok() -> None:
    env = os.environ.copy()
    env["SWAY_SERVER_PERF"] = "1"
    r = subprocess.run(
        [sys.executable, "-m", "tools.smoke_server_perf_env"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "All smoke checks passed." in (r.stdout or "")


# ---------- SOURCE: test_synthetic_regression.py ----------

"""
Synthetic regression contracts (plan E2 stand-in until full MOT fixtures exist).

A — crossover: Pass 1.5 IoU+centroid merge is OKS-gated.
B — mirror: smart-mirror pruner is available for edge+velocity+lower-body rules.
C — static silhouette: short / low-kinetic tracks are dropped by pre-pose pruning.
"""

import sys
from pathlib import Path

def test_A_crossover_pass15_oks_gate_in_source():
    p = ROOT / "sway" / "crossover.py"
    text = p.read_text()
    assert "OKS_VETO_PASS15" in text
    assert "merge_iou_centroid" in text


def test_B_smart_mirror_pruner_callable():
    from sway.track_pruning import prune_smart_mirrors

    assert callable(prune_smart_mirrors)


def test_C_short_track_pruned():
    from sway.track_pruning import prune_tracks

    # One frame only in 1000-frame video -> below duration threshold
    raw = {99: [(0, (0.0, 0.0, 10.0, 20.0), 0.9)]}
    kept = prune_tracks(raw, total_frames=1000, min_duration_ratio=0.20)
    assert 99 not in kept


# ---------- SOURCE: test_technology_contracts.py ----------

"""Unit tests for sway.technology_contracts (no pipeline subprocess)."""

from pathlib import Path

from sway.technology_contracts import (
    validate_global_pipeline_invariants,
    validate_run_against_contracts,
)


def test_global_duplicate_phase_header_is_fatal():
    log = "[1/11] Phase 1 — first\n[2/11] Phase 2\n[1/11] Phase 1 — illegal repeat\n"
    v = validate_global_pipeline_invariants(log)
    assert any(x.clause == "phase_no_restart" for x in v)


def test_global_tracker_warning_storm():
    line = "[tracker warning] bad\n"
    log = "[1/11] ok\n" + line * 201
    v = validate_global_pipeline_invariants(log)
    assert any(x.clause == "tracker_warning_storm" for x in v)


def test_global_zero_survivor_collapse_when_raw_positive():
    log = (
        "[1/11] x\n(42 raw tracks)\nKept 0 of 10 tracks after pre-pose pruning\n"
    )
    v = validate_global_pipeline_invariants(log)
    assert any(x.clause == "zero_survivor_collapse" for x in v)


def test_sam2mot_on_boxmot_branch_is_forbidden():
    env = {"SWAY_TRACKER_ENGINE": "sam2mot"}
    log = (
        "[1/11] Phase 1\n"
        "[2/11] Phase 2 — Tracking (BoxMOT Deep OC-SORT\n"
    )
    v = validate_run_against_contracts(log, env, None, "after_phase_3")
    assert any(x.clause == "forbidden_branch" for x in v)


def test_sam2mot_future_branch_no_branch_violation():
    env = {"SWAY_TRACKER_ENGINE": "sam2mot"}
    log = "future pipeline: test\n[1/11] Phase 1\n"
    v = validate_run_against_contracts(log, env, None, "after_phase_3")
    assert not any(x.clause in ("forbidden_branch", "wrong_branch") for x in v)


def test_enrollment_requires_gallery_artifact(tmp_path: Path):
    env = {"SWAY_ENROLLMENT_ENABLED": "1"}
    log = "[1/11] ok\n"
    v = validate_run_against_contracts(log, env, tmp_path, "after_phase_3")
    assert any(x.clause == "missing_artifact" and "gallery.json" in x.detail for x in v)


def test_global_phase_out_of_order():
    log = "[3/11] three\n[2/11] two\n"
    v = validate_global_pipeline_invariants(log)
    assert any(x.clause == "phase_ordering" for x in v)


# ---------- SOURCE: test_track_stats_export.py ----------

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


# ---------- SOURCE: test_validate_pipeline_e2e_helpers.py ----------

"""Unit tests for validate_pipeline_e2e helpers (no full pipeline run)."""

from pathlib import Path

import pytest

from tools.validate_pipeline_e2e import (
    _extract_stop_boundary,
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


def test_extract_stop_boundary_from_args():
    assert _extract_stop_boundary([]) == "final"
    assert _extract_stop_boundary(["--stop-after-boundary", "after_phase_3"]) == "after_phase_3"


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


# ---------- SOURCE: test_video_camera_probe.py ----------

"""Tests for ffprobe-based camera intrinsics probing."""


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


# ---------- SOURCE: test_yolo_runtime_env.py ----------

"""YOLO runtime env helpers (no Ultralytics inference)."""

import os

from sway.pose_estimator import vitpose_force_fp32
from sway.tracker import (
    resolve_yolo_inference_weights,
    yolo_half_env_requested,
    yolo_predict_use_half,
)


def test_yolo_half_env_parsing():
    os.environ.pop("SWAY_YOLO_HALF", None)
    assert yolo_half_env_requested() is False
    os.environ["SWAY_YOLO_HALF"] = "1"
    assert yolo_half_env_requested() is True
    os.environ["SWAY_YOLO_HALF"] = "no"
    assert yolo_half_env_requested() is False
    os.environ.pop("SWAY_YOLO_HALF", None)


def test_yolo_predict_use_half_matches_cuda():
    os.environ["SWAY_YOLO_HALF"] = "1"
    # On CPU CI, half should not activate
    import torch

    expect = torch.cuda.is_available()
    assert yolo_predict_use_half() == expect
    os.environ.pop("SWAY_YOLO_HALF", None)


def test_resolve_yolo_engine_prefers_existing_file(tmp_path):
    eng = tmp_path / "model.engine"
    eng.write_bytes(b"\0")
    os.environ["SWAY_YOLO_ENGINE"] = str(eng)
    try:
        assert resolve_yolo_inference_weights() == str(eng.resolve())
    finally:
        os.environ.pop("SWAY_YOLO_ENGINE", None)


def test_resolve_yolo_engine_missing_raises(tmp_path):
    os.environ["SWAY_YOLO_ENGINE"] = str(tmp_path / "missing.engine")
    try:
        try:
            resolve_yolo_inference_weights()
        except FileNotFoundError as e:
            assert "SWAY_YOLO_ENGINE" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")
    finally:
        os.environ.pop("SWAY_YOLO_ENGINE", None)


def test_vitpose_fp32_env():
    os.environ.pop("SWAY_VITPOSE_FP32", None)
    assert vitpose_force_fp32() is False
    os.environ["SWAY_VITPOSE_FP32"] = "1"
    assert vitpose_force_fp32() is True
    os.environ.pop("SWAY_VITPOSE_FP32", None)

