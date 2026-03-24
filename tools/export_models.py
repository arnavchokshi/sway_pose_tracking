import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ultralytics import YOLO


def export_yolo(model_name: Optional[str] = None):
    if model_name is None:
        model_name = str(_REPO_ROOT / "models" / "yolo26l.pt")
    print(f"Exporting {model_name} to CoreML with NMS...")
    model = YOLO(model_name)
    # nms=True bakes Non-Maximum Suppression into the CoreML model, saving CPU time
    model.export(format="coreml", nms=True)
    print("YOLO CoreML export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", action="store_true", help="Export YOLO model")
    args = parser.parse_args()

    if args.yolo:
        export_yolo()
    else:
        export_yolo()
