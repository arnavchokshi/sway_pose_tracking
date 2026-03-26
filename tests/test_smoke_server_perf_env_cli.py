"""CLI smoke for tools.smoke_server_perf_env (no --pipeline; fast)."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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
