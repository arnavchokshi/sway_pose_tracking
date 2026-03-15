import argparse
from ultralytics import YOLO

def export_yolo(model_name="yolo11l.pt"):
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
