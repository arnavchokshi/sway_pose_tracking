#!/usr/bin/env python3
"""
Optional model exports (not run by the pipeline).

Examples:

  # CoreML (existing behavior)
  python -m tools.export_models --coreml

  # TensorRT engine on CUDA device 0 (run on the same GPU you deploy on)
  python -m tools.export_models --tensorrt --device 0

Requires local ``ultralytics`` install and, for TensorRT, CUDA + TensorRT-compatible export.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ultralytics import YOLO


def export_yolo_coreml(model_name: Optional[str] = None) -> None:
    if model_name is None:
        model_name = str(_REPO_ROOT / "models" / "yolo26l.pt")
    print(f"Exporting {model_name} to CoreML with NMS...")
    model = YOLO(model_name)
    model.export(format="coreml", nms=True)
    print("YOLO CoreML export complete.")


def export_yolo_tensorrt(model_name: Optional[str] = None, device: str = "0") -> None:
    if model_name is None:
        model_name = str(_REPO_ROOT / "models" / "yolo26l.pt")
    print(f"Exporting {model_name} to TensorRT engine (device={device})…")
    model = YOLO(model_name)
    model.export(format="engine", device=device)
    print(
        "TensorRT export finished. Point the pipeline at the generated .engine with:\n"
        "  SWAY_YOLO_ENGINE=/path/to/your.engine",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Optional YOLO exports (manual; not used by main.py).")
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to .pt weights (default: models/yolo26l.pt if present else hub id).",
    )
    p.add_argument("--coreml", action="store_true", help="Export CoreML + NMS")
    p.add_argument("--tensorrt", action="store_true", help="Export TensorRT .engine (CUDA)")
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id for TensorRT export (default: 0).",
    )
    args = p.parse_args()
    w = args.weights
    if args.tensorrt and args.coreml:
        p.error("Choose only one of --coreml or --tensorrt")
    if args.tensorrt:
        export_yolo_tensorrt(w, device=str(args.device))
    else:
        # Legacy: no flags → CoreML (matches pre–TensorRT script behavior)
        if not args.coreml:
            print("No --tensorrt: defaulting to CoreML export (same as legacy tools/export_models.py).", flush=True)
        export_yolo_coreml(w)


if __name__ == "__main__":
    main()
