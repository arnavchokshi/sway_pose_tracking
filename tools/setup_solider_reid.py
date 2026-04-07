#!/usr/bin/env python3
"""
One-time setup: download SOLIDER-REID Swin-Small MSMT17 weights and
export to ONNX for use with BoxMOT's ReidAutoBackend.

Usage:
    python -m tools.setup_solider_reid

Outputs:
    models/solider_swin_small_msmt17.onnx

Prerequisites:
    pip install gdown onnx torch  (gdown is a BoxMOT dependency; onnx needed for export)
    PyTorch 2.x: script injects ``torch._six`` / stub ``mmcv`` compat and uses legacy
    ``torch.onnx.export(..., dynamo=False)``; CPU-only hosts patch ``.cuda()`` during trace.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
SOLIDER_REID_DIR = MODELS_DIR / "SOLIDER-REID"
ONNX_OUTPUT = MODELS_DIR / "solider_swin_small_msmt17.onnx"

SOLIDER_REID_REPO = "https://github.com/tinyvision/SOLIDER-REID.git"
SWIN_SMALL_MSMT17_GDRIVE_ID = "1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b"
CHECKPOINT_NAME = "swin_small_msmt17.pth"


def _run(cmd: list[str], **kwargs: object) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, **kwargs)


def clone_repo() -> None:
    if SOLIDER_REID_DIR.is_dir():
        print(f"SOLIDER-REID already cloned at {SOLIDER_REID_DIR}")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", SOLIDER_REID_REPO, str(SOLIDER_REID_DIR)])


def download_weights() -> Path:
    ckpt = SOLIDER_REID_DIR / CHECKPOINT_NAME
    if ckpt.is_file():
        print(f"Weights already downloaded: {ckpt}")
        return ckpt
    import gdown
    url = f"https://drive.google.com/uc?id={SWIN_SMALL_MSMT17_GDRIVE_ID}"
    print(f"Downloading SOLIDER Swin-Small MSMT17 weights...", flush=True)
    gdown.download(url, str(ckpt), quiet=False)
    if not ckpt.is_file():
        raise RuntimeError(f"Download failed: {ckpt} not found after gdown")
    return ckpt


def _inject_torch_six_compat() -> None:
    """SOLIDER-REID targets PyTorch 1.x; ``torch._six`` was removed in PyTorch 2."""
    import collections.abc
    import types

    import torch

    if hasattr(torch, "_six"):
        return
    m = types.ModuleType("torch._six")
    m.container_abcs = collections.abc
    sys.modules["torch._six"] = m


def _inject_mmcv_stub() -> None:
    """SOLIDER-REID imports ``mmcv.runner.load_checkpoint`` but may not call it for ONNX export."""
    import types

    if "mmcv.runner" in sys.modules:
        return

    def load_checkpoint(
        model: object,
        filename: str,
        map_location: object = None,
        strict: bool = False,
        logger: object = None,
    ) -> object:
        return None

    runner = types.ModuleType("mmcv.runner")
    runner.load_checkpoint = load_checkpoint
    mmcv = types.ModuleType("mmcv")
    mmcv.runner = runner
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.runner"] = runner


def export_onnx(ckpt: Path) -> Path:
    if ONNX_OUTPUT.is_file():
        print(f"ONNX already exists: {ONNX_OUTPUT}")
        return ONNX_OUTPUT

    print("Exporting SOLIDER-REID to ONNX...", flush=True)

    solider_path = str(SOLIDER_REID_DIR)
    if solider_path not in sys.path:
        sys.path.insert(0, solider_path)

    try:
        import torch

        _inject_torch_six_compat()
        _inject_mmcv_stub()

        from config import cfg  # type: ignore[import]
        from model import make_model  # type: ignore[import]

        config_path = SOLIDER_REID_DIR / "configs" / "MSMT17" / "swin_small.yml"
        if not config_path.is_file():
            alt_paths = list(SOLIDER_REID_DIR.rglob("swin_small*.yml"))
            if alt_paths:
                config_path = alt_paths[0]
            else:
                raise FileNotFoundError(
                    f"No swin_small config found in {SOLIDER_REID_DIR}"
                )

        cfg.merge_from_file(str(config_path))
        model = make_model(
            cfg,
            num_class=1041,
            camera_num=15,
            view_num=1,
            semantic_weight=0.2,
        )
        model.load_param(str(ckpt))
        model.eval()
        model.cpu()

        dummy = torch.randn(1, 3, 256, 128)
        # SOLIDER forward calls ``.cuda()`` on internal tensors; force CPU-only export.
        _orig_t_cuda = torch.Tensor.cuda

        def _cuda_noop(self, *args: object, **kwargs: object) -> torch.Tensor:
            return self

        torch.Tensor.cuda = _cuda_noop  # type: ignore[method-assign]
        try:
            # PyTorch 2.10+ defaults to dynamo exporter (needs onnxscript); legacy avoids that.
            torch.onnx.export(
                model,
                dummy,
                str(ONNX_OUTPUT),
                opset_version=14,
                input_names=["images"],
                output_names=["features"],
                dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
                dynamo=False,
            )
        finally:
            torch.Tensor.cuda = _orig_t_cuda  # type: ignore[method-assign]

        print(f"Exported: {ONNX_OUTPUT}")
    except Exception as ex:
        print(f"ONNX export failed: {ex}", file=sys.stderr)
        print(
            "You may need to manually export the model. The weights are at:",
            ckpt,
            file=sys.stderr,
        )
        raise

    return ONNX_OUTPUT


def main() -> None:
    clone_repo()
    ckpt = download_weights()
    export_onnx(ckpt)
    print(f"\nDone. ONNX model ready at: {ONNX_OUTPUT}")
    print(f"Use in sweep: SWAY_BOXMOT_REID_WEIGHTS={ONNX_OUTPUT}")


if __name__ == "__main__":
    main()
