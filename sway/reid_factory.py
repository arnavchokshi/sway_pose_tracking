"""
Re-ID Factory (PLAN_13)

Creates the re-ID ensemble based on environment configuration.
Each signal is independently togglable via SWAY_REID_*_ENABLED flags.

The fusion engine dynamically discovers available signals via this registry.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


def create_part_reid(model_name: str = "bpbreid", device: str = "cuda"):
    """Create a part-based re-ID extractor.

    Supported model names:
      bpbreid            – default BPBreID with ResNet-50 backbone
      bpbreid_finetuned  – contrastive fine-tuned BPBreID (PLAN_20)
      osnet_x0_25        – OSNet global embedding (fallback baseline)
      paformer           – PAFormer part-aware transformer (future add-on)
    """
    from pathlib import Path
    models_dir = Path(__file__).resolve().parent.parent / "models"

    if model_name in ("bpbreid", "osnet_x0_25"):
        from sway.bpbreid_extractor import BPBreIDExtractor
        return BPBreIDExtractor(device=device)
    elif model_name == "bpbreid_finetuned":
        finetuned_path = models_dir / "bpbreid_r50_sway_finetuned.pth"
        if not finetuned_path.exists():
            logger.warning(
                "Fine-tuned BPBreID weights not found at %s; falling back to base BPBreID. "
                "Run: python -m tools.finetune_reid to generate fine-tuned weights.",
                finetuned_path,
            )
        from sway.bpbreid_extractor import BPBreIDExtractor
        return BPBreIDExtractor(
            checkpoint_path=str(finetuned_path) if finetuned_path.exists() else None,
            device=device,
        )
    elif model_name == "paformer":
        paformer_path = models_dir / "paformer.pth"
        if not paformer_path.exists():
            logger.warning(
                "PAFormer weights not found at %s; falling back to BPBreID. "
                "PAFormer is a future add-on. Place weights at %s to enable.",
                paformer_path, paformer_path,
            )
        from sway.bpbreid_extractor import BPBreIDExtractor
        return BPBreIDExtractor(device=device)
    else:
        logger.warning("Unknown part re-ID model: %s; falling back to BPBreID", model_name)
        from sway.bpbreid_extractor import BPBreIDExtractor
        return BPBreIDExtractor(device=device)


def create_reid_ensemble(device: str = "cuda") -> Dict[str, object]:
    """Build the re-ID signal module registry based on env config.

    Returns dict of {signal_name: extractor_instance} for available signals.
    """
    signals: Dict[str, object] = {}

    # Always available (lean core)
    part_model = _env_str("SWAY_REID_PART_MODEL", "bpbreid")
    finetune_enabled = _env_bool("SWAY_REID_FINETUNE_ENABLED", False)
    finetune_base_model = _env_str("SWAY_REID_FINETUNE_BASE_MODEL", "bpbreid").lower()
    if finetune_enabled:
        # Promote finetune toolchain knobs into runtime model selection so sweeps
        # can truly exercise the finetune surface from env-only runs.
        if finetune_base_model == "osnet":
            if part_model in ("bpbreid", "bpbreid_finetuned"):
                part_model = "osnet_x0_25"
        else:
            if part_model == "bpbreid":
                part_model = "bpbreid_finetuned"
    try:
        signals["part"] = create_part_reid(part_model, device)
    except Exception as exc:
        logger.warning("Part re-ID init failed: %s", exc)

    if _env_bool("SWAY_REID_KPR_ENABLED", True):
        try:
            from sway.kpr_extractor import KPRExtractor
            signals["kpr"] = KPRExtractor(device=device)
        except Exception as exc:
            logger.warning("KPR init failed: %s", exc)

    # Always available
    try:
        from sway.color_histogram_reid import ColorHistogramExtractor
        signals["color"] = ColorHistogramExtractor()
    except Exception as exc:
        logger.warning("Color histogram init failed: %s", exc)

    # Experiment add-ons (default OFF)
    skel_model = os.environ.get("SWAY_REID_SKEL_MODEL", "").strip()
    if skel_model:
        try:
            from sway.mocos_extractor import MoCosExtractor
            signals["skeleton"] = MoCosExtractor(device=device)
        except Exception as exc:
            logger.warning("Skeleton gait init failed: %s", exc)

    face_model = os.environ.get("SWAY_REID_FACE_MODEL", "").strip()
    if face_model:
        try:
            from sway.face_reid import FaceReIDExtractor
            signals["face"] = FaceReIDExtractor(model_name=face_model, device=device)
        except Exception as exc:
            logger.warning("Face re-ID init failed: %s", exc)

    logger.info("Re-ID ensemble: %s", list(signals.keys()))
    return signals
