# Model weights

Keep all downloadable checkpoints here so the package root stays clean:

- **YOLO (PyTorch):** `yolo26l.pt` (default for `prefetch_models.py` and hub fallback), plus any other `yolo*.pt` you use
- **Hybrid SAM2:** `sam2.1_b.pt` (default for `SWAY_HYBRID_SAM_WEIGHTS` / overlap refiner; resolved from `models/` automatically)
- **Core ML (legacy):** `yolo11l.mlpackage` or `yolo11m.mlpackage` when no `.pt` is found
- **BoxMOT Re-ID:** `osnet_x0_25_msmt17.pt` (from `prefetch_models.py`)

Point `SWAY_YOLO_WEIGHTS` or `SWAY_HYBRID_SAM_WEIGHTS` at a specific file if the default priority is not what you want.

ViTPose weights are cached by Hugging Face (see main README, Offline).
