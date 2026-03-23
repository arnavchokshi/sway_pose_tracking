# Model weights

Put YOLO assets here so the project root stays clean:

- **Core ML:** `yolo11l.mlpackage` or `yolo11m.mlpackage` (preferred on Apple Silicon when present)
- **PyTorch:** `yolo11m.pt` (default for `prefetch_models.py` and offline fallback)

Optional extra `.pt` sizes (`yolo11l.pt`, etc.) can live here; point `SWAY_YOLO_WEIGHTS` at the file you want if it is not the default.

ViTPose weights are cached by Hugging Face (see main README, Offline).
