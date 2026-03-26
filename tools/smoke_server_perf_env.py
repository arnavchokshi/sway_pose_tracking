#!/usr/bin/env python3
"""
Check that ``SWAY_SERVER_PERF`` and the sweep merge (``subprocess_env_overlay``) reach
child processes the same way as ``tools.auto_sweep`` / ``tools.run_sweep``.

Typical pre-flight (~5–15s + optional pipeline):

  export SWAY_SERVER_PERF=1
  python -m tools.smoke_server_perf_env

Add a bounded ``main.py`` run on a synthetic clip (default 60s timeout, ``after_phase_1``):

  python -m tools.smoke_server_perf_env --pipeline --timeout 60
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sway.server_runtime_perf import subprocess_env_overlay

_ENV_PROBE = """
import json, os, sys
keys = sys.argv[1].split(",")
print(json.dumps({k: os.environ.get(k) for k in keys}))
"""

_TORCH_PROBE = """
import json, os, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
keys = sys.argv[2].split(",")
out = {k: os.environ.get(k) for k in keys}
import torch
from sway.server_runtime_perf import apply_server_runtime_perf

apply_server_runtime_perf()
out["_torch_cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available() and os.environ.get("SWAY_CUDA_PERF_DISABLE", "").strip().lower() not in (
    "1", "true", "yes", "on",
):
    out["_cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    out["_cuda_matmul_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
else:
    out["_cudnn_benchmark"] = None
    out["_cuda_matmul_tf32"] = None
print(json.dumps(out))
"""


def _parent_perf_ok() -> bool:
    return os.environ.get("SWAY_SERVER_PERF", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _merged_child_env() -> dict[str, str]:
    overlay = subprocess_env_overlay()
    if not overlay:
        return {}
    env = os.environ.copy()
    env.update(overlay)
    return env


def _run_json_subprocess(code: str, extra_args: list[str], env: dict) -> dict:
    cmd = [sys.executable, "-c", code.strip(), *extra_args]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-2500:]
        raise RuntimeError(f"probe exit {r.returncode}\n--- stderr/stdout tail ---\n{tail}")
    lines = (r.stdout or "").strip().splitlines()
    line = lines[-1] if lines else ""
    return json.loads(line)


def _make_tiny_mp4(path: Path) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h, fps, n = 640, 480, 30, 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed (codec/path)")
    for i in range(n):
        f = np.full((h, w, 3), 40, dtype=np.uint8)
        x = 100 + (i * 8) % 200
        cv2.rectangle(f, (x, 180), (x + 80, 420), (200, 200, 200), -1)
        vw.write(f)
    vw.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify SWAY_SERVER_PERF reaches subprocesses (sweep merge parity)"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run main.py on a synthetic clip with merged env (bounded by --timeout)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Seconds for --pipeline main.py subprocess (default: 60)",
    )
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=REPO_ROOT / "output" / "_server_perf_smoke",
        help="Working dir for synthetic clip + pipeline output",
    )
    args = parser.parse_args()

    if not _parent_perf_ok():
        print(
            "Set SWAY_SERVER_PERF=1 in this shell first (same as overnight sweeps).",
            file=sys.stderr,
        )
        sys.exit(2)

    overlay = subprocess_env_overlay()
    if not overlay:
        print(
            "subprocess_env_overlay() is empty — SWAY_SERVER_PERF not recognized.",
            file=sys.stderr,
        )
        sys.exit(2)

    env = _merged_child_env()
    keys = sorted(overlay.keys())

    print("1) Env probe (merged env, no sway import)…", flush=True)
    probe = _run_json_subprocess(_ENV_PROBE, [",".join(keys)], env)
    bad = [f"  {k!r}: child={probe.get(k)!r} expected {overlay[k]!r}" for k in keys if probe.get(k) != overlay[k]]
    if bad:
        print("Mismatch — subprocess did not inherit overlay:", file=sys.stderr)
        print("\n".join(bad), file=sys.stderr)
        sys.exit(1)
    print("   OK:", {k: probe[k] for k in keys}, flush=True)

    print("2) Torch + apply_server_runtime_perf in child…", flush=True)
    tprobe = _run_json_subprocess(_TORCH_PROBE, [str(REPO_ROOT), ",".join(keys)], env)
    bad2 = [f"  {k!r}: child={tprobe.get(k)!r} expected {overlay[k]!r}" for k in keys if tprobe.get(k) != overlay[k]]
    if bad2:
        print("Env mismatch after torch import:", file=sys.stderr)
        print("\n".join(bad2), file=sys.stderr)
        sys.exit(1)
    if tprobe.get("_torch_cuda_available"):
        if tprobe.get("_cudnn_benchmark") is not True:
            print(
                f"  Expected cudnn.benchmark True in CUDA child, got {tprobe.get('_cudnn_benchmark')!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        if tprobe.get("_cuda_matmul_tf32") is not True:
            print(
                f"  Expected cuda.matmul.allow_tf32 True, got {tprobe.get('_cuda_matmul_tf32')!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        print("   OK: CUDA child — cudnn.benchmark and TF32 matmul on.", flush=True)
    else:
        print("   OK: CPU-only child — skipped CUDA flag checks.", flush=True)

    if args.pipeline:
        print(f"3) main.py smoke (timeout {args.timeout}s, after_phase_1)…", flush=True)
        root = args.smoke_dir.resolve()
        clip = root / "tiny.mp4"
        out_dir = root / "out"
        _make_tiny_mp4(clip)
        out_dir.mkdir(parents=True, exist_ok=True)
        penv = dict(env)
        penv.setdefault("SWAY_UNLOCK_HYBRID_SAM_TUNING", "1")
        penv["SWAY_HYBRID_SAM_OVERLAP"] = "0"
        penv["SWAY_YOLO_DETECTION_STRIDE"] = "2"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            str(clip),
            "--output-dir",
            str(out_dir),
            "--pose-model",
            "base",
            "--pose-stride",
            "4",
            "--stop-after-boundary",
            "after_phase_1",
        ]
        try:
            r = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=penv,
                timeout=args.timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            print(
                f"main.py exceeded {args.timeout}s — raise --timeout or check GPU/CUDA setup.",
                file=sys.stderr,
            )
            sys.exit(1)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "")[-3000:]
            print(f"main.py failed rc={r.returncode}\n--- tail ---\n{tail}", file=sys.stderr)
            sys.exit(1)
        print("   OK: main.py completed with merged perf env.", flush=True)

    print("All smoke checks passed.", flush=True)


if __name__ == "__main__":
    main()
