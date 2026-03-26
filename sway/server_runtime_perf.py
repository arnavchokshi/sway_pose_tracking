"""
Tuning for high-core NVIDIA servers (e.g. Lambda 1×A100, ~30 vCPU, large RAM).

Export ``SWAY_SERVER_PERF=1`` before ``python main.py``. Sweep drivers merge the same
flags into subprocess environments when this var is already set in the parent shell.
Before long runs: ``python -m tools.smoke_server_perf_env`` (optional ``--pipeline --timeout 60``).

Optional: ``SWAY_PERF_CPU_THREADS`` (int) caps PyTorch CPU, OpenCV, and common BLAS
thread env vars when unset. ``SWAY_CUDA_PERF_DISABLE=1`` skips cuDNN/TF32 tweaks only.
"""

from __future__ import annotations

import os
from typing import Dict


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _default_cpu_threads() -> int:
    c = os.cpu_count() or 8
    tentative = max(4, min(c - 4, 16))
    return min(max(tentative, 4), c)


def recommended_cpu_thread_count() -> int:
    v = os.environ.get("SWAY_PERF_CPU_THREADS", "").strip()
    if v:
        try:
            return max(1, int(v))
        except ValueError:
            pass
    return _default_cpu_threads()


def apply_server_runtime_perf() -> None:
    if not _truthy("SWAY_SERVER_PERF"):
        return
    import torch

    n = recommended_cpu_thread_count()
    try:
        torch.set_num_threads(n)
        interop = max(1, min(4, max(1, n // 4)))
        torch.set_num_interop_threads(interop)
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(n)
    except Exception:
        pass

    if not torch.cuda.is_available():
        return
    if _truthy("SWAY_CUDA_PERF_DISABLE"):
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def subprocess_env_overlay() -> Dict[str, str]:
    """Env pairs to merge into ``subprocess.run(..., env=...)`` when sweeps spawn ``main.py``."""
    if not _truthy("SWAY_SERVER_PERF"):
        return {}
    n = str(recommended_cpu_thread_count())
    out: Dict[str, str] = {
        "SWAY_SERVER_PERF": "1",
        "SWAY_PERF_CPU_THREADS": n,
    }
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if not os.environ.get(key):
            out[key] = n
    return out
