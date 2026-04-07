"""
Tuning for high-core NVIDIA servers (e.g. Lambda **gpu_1x_a10** or 1×A100, many vCPUs, ample RAM).

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
    """Env pairs to merge into ``subprocess.run(..., env=...)`` when sweeps spawn ``main.py``.

    Passes infrastructure speed-ups to each child process:
      - CPU thread limits (PyTorch, OpenCV, BLAS) to avoid over-subscription when
        multiple videos run back-to-back in the same sweep trial.
      - YOLO FP16 (CUDA only): ~40% faster YOLO inference on A10/A100 with
        negligible accuracy difference.
      - YOLO infer batch size 4: amortises GPU call overhead when YOLO runs on
        consecutive frames (tracker.update still steps per-frame in order).

    All of these are infrastructure knobs, not tuning parameters — they are NOT
    added to Optuna's parameter space.
    """
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

    # GPU speed-ups: only inject when CUDA is reachable and caller hasn't
    # already set a preference (respect explicit overrides).
    _cuda_available: bool = False
    try:
        import torch
        _cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    if _cuda_available:
        if not os.environ.get("SWAY_YOLO_HALF"):
            out["SWAY_YOLO_HALF"] = "1"
        if not os.environ.get("SWAY_YOLO_INFER_BATCH"):
            # batch=4 amortises overhead; large enough to matter, small enough to
            # keep VRAM headroom on A10 (24 GB) alongside the ReID model.
            out["SWAY_YOLO_INFER_BATCH"] = "4"

    return out
