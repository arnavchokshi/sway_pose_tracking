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
