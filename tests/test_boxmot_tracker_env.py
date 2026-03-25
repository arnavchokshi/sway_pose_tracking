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


def test_strongsort_requires_reid_file(tmp_path):
    import torch

    dev = torch.device("cpu")
    doc = _deepocsort_extra_from_env()
    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError):
        _create_boxmot_tracker("strongsort", 0.22, dev, missing, doc)
