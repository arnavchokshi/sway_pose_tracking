# Model weights

Keep all downloadable checkpoints here so the package root stays clean:

- **YOLO (PyTorch):** YOLO26 weights only, e.g. `yolo26l.pt` (default for `python -m tools.prefetch_models` and hub fallback), optional `yolo26s.pt` / `yolo26x.pt`, fine-tunes like `yolo26l_dancetrack.pt`
- **Hybrid SAM2:** `sam2.1_b.pt` (default for `SWAY_HYBRID_SAM_WEIGHTS` / overlap refiner; resolved from `models/` automatically)
- **BoxMOT Re-ID:** `osnet_x0_25_msmt17.pt` (from `python -m tools.prefetch_models`)

Point `SWAY_YOLO_WEIGHTS` or `SWAY_HYBRID_SAM_WEIGHTS` at a specific file if the default priority is not what you want.

ViTPose weights are cached by Hugging Face (see main README, Offline).
