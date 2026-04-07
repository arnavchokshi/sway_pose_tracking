"""
DETR-family Detection Backend (PLAN_02)

Provides Co-DETR, Co-DINO, and RT-DETR as NMS-free person detection alternatives to YOLO.
DETR detectors use learned object queries with Hungarian matching — each person gets exactly
one predicted box with no duplicate-suppression ambiguity.

Supports:
  - co_detr_swinl  : Co-DETR with Swin-L backbone (~66 AP COCO, 3-5x slower than YOLO)
  - co_dino_swinl  : Co-DINO variant
  - rt_detr_l      : RT-DETR Large via ultralytics (NMS-free, ~1.5x YOLO cost)
  - rt_detr_x      : RT-DETR Extra-Large via ultralytics

When using DETR, all NMS-related code (SWAY_PRETRACK_NMS_IOU) is bypassed — DETR already
produces deduplicated detections.

Env:
  SWAY_DETECTOR_PRIMARY  – detector choice (default: yolo26l_dancetrack)
  SWAY_DETR_CONF         – confidence threshold for DETR (default: 0.30)
  SWAY_DETECT_SIZE       – image short-side for inference (default: 800)
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Unified detection output — matches the contract consumed by tracker.py."""
    bbox: np.ndarray    # (4,) x1, y1, x2, y2
    confidence: float
    class_id: int = 0   # 0 = person


class DETRDetector:
    """NMS-free person detector using DETR-family models.

    Implements the same detect() -> List[Detection] interface as the YOLO detector.
    """

    SUPPORTED_MODELS = {"co_detr_swinl", "co_dino_swinl", "rt_detr_l", "rt_detr_x", "deimv2_x"}

    def __init__(
        self,
        model_name: str = "rt_detr_l",
        device: str = "cuda",
        conf_threshold: Optional[float] = None,
        detect_size: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold or float(
            os.environ.get("SWAY_DETR_CONF", "0.30")
        )
        self.detect_size = detect_size or int(
            os.environ.get("SWAY_DETECT_SIZE", "800")
        )
        self._model = None
        self._is_ultralytics = model_name.startswith("rt_detr")
        self._is_deimv2 = model_name.startswith("deimv2")
        self._load_model()

    def _load_model(self) -> None:
        if self._is_deimv2:
            self._load_deimv2()
        elif self._is_ultralytics:
            self._load_ultralytics_rtdetr()
        else:
            self._load_codetr()

    def _load_deimv2(self) -> None:
        """Load DEIMv2-X from official DEIMv2 repo + HuggingFace weights."""
        try:
            deim_repo = Path(
                os.environ.get(
                    "SWAY_DEIMV2_REPO_PATH",
                    str(Path(__file__).resolve().parent.parent.parent / "third_party" / "DEIMv2"),
                )
            ).expanduser()
            if not deim_repo.is_dir():
                raise FileNotFoundError(
                    f"DEIMv2 repo not found at {deim_repo}. Set SWAY_DEIMV2_REPO_PATH to a valid clone."
                )
            repo_path = str(deim_repo.resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

            from huggingface_hub import PyTorchModelHubMixin
            from engine.backbone import DINOv3STAs
            from engine.deim import HybridEncoder, DEIMTransformer
            from engine.deim.postprocessor import PostProcessor

            class _DEIMv2Model(torch.nn.Module, PyTorchModelHubMixin):
                def __init__(self, config):
                    super().__init__()
                    self.backbone = DINOv3STAs(**config["DINOv3STAs"])
                    self.encoder = HybridEncoder(**config["HybridEncoder"])
                    self.decoder = DEIMTransformer(**config["DEIMTransformer"])
                    self.postprocessor = PostProcessor(**config["PostProcessor"])

                def forward(self, x, orig_target_sizes):
                    x = self.backbone(x)
                    x = self.encoder(x)
                    x = self.decoder(x)
                    x = self.postprocessor(x, orig_target_sizes)
                    return x

            repo_id = os.environ.get("SWAY_DEIMV2_HF_REPO", "Intellindust/DEIMv2_DINOv3_X_COCO").strip()
            logger.info("Loading DEIMv2 model from HuggingFace repo: %s", repo_id)
            self._model = _DEIMv2Model.from_pretrained(repo_id)
            self._model.to(self.device)
            self._model.eval()
            self._deimv2_input_size = int(os.environ.get("SWAY_DEIMV2_SIZE", "640") or 640)
            self._deimv2_state = {"repo_id": repo_id, "repo_path": repo_path}
            self.model_name = "deimv2_x"
            self._is_deimv2 = True
            self._is_ultralytics = False
            logger.info(
                "Loaded DEIMv2 model: %s on %s (input=%d)",
                self.model_name,
                self.device,
                self._deimv2_input_size,
            )
        except Exception as exc:
            logger.warning("DEIMv2 init failed (%s). Falling back to RT-DETR-L.", exc)
            self._is_deimv2 = False
            self._is_ultralytics = True
            self.model_name = "rt_detr_l"
            self._load_ultralytics_rtdetr()

    def _load_ultralytics_rtdetr(self) -> None:
        """Load RT-DETR via ultralytics (already a project dependency)."""
        try:
            from ultralytics import RTDETR
        except ImportError:
            from ultralytics import YOLO as RTDETR
            logger.warning(
                "ultralytics.RTDETR not available; falling back to YOLO loader for RT-DETR weights"
            )

        weight_map = {
            "rt_detr_l": "rtdetr-l.pt",
            "rt_detr_x": "rtdetr-x.pt",
        }
        weight_name = weight_map.get(self.model_name, "rtdetr-l.pt")
        models_dir = Path(__file__).resolve().parent.parent / "models"
        weight_path = models_dir / weight_name

        if weight_path.exists():
            self._model = RTDETR(str(weight_path))
        else:
            logger.info(
                "RT-DETR weight %s not found locally; ultralytics will auto-download", weight_name
            )
            self._model = RTDETR(weight_name)

        logger.info("Loaded RT-DETR model: %s on %s", self.model_name, self.device)

    def _load_codetr(self) -> None:
        """Load Co-DETR / Co-DINO via detrex or mmdet.

        This is a heavier path requiring external checkpoints.
        Falls back to a stub that raises a clear error if dependencies are missing.
        """
        models_dir = Path(__file__).resolve().parent.parent / "models"
        weight_map = {
            "co_detr_swinl": "co_detr_swinl_coco.pth",
            "co_dino_swinl": "co_dino_swinl_coco.pth",
        }
        weight_name = weight_map.get(self.model_name, "co_detr_swinl_coco.pth")
        weight_path = models_dir / weight_name
        if not weight_path.exists() and self.model_name == "co_detr_swinl":
            # Co-DETR mirrors are less stable; allow co_detr mode to use the local
            # Co-DINO Swin-L checkpoint (same family) when the Co-DETR filename is absent.
            fallback = models_dir / "co_dino_swinl_coco.pth"
            if fallback.exists():
                logger.warning(
                    "co_detr_swinl_coco.pth missing; reusing co_dino_swinl_coco.pth for co_detr mode"
                )
                weight_path = fallback
        cfg_path = self._resolve_codetr_config_path()
        if cfg_path is None:
            logger.error(
                "No Co-DETR/Co-DINO config found. Set SWAY_CODETR_REPO to a Co-DETR checkout "
                "or SWAY_MMDET_CODINO_CONFIG to a valid MMDet config path."
            )
        elif self.model_name == "co_detr_swinl":
            logger.warning(
                "Using Co-DINO Swin-L config/weights for 'co_detr' mode due upstream "
                "Co-DETR checkpoint URL instability."
            )

        try:
            from mmdet.apis import init_detector
            if cfg_path and weight_path.exists():
                self._model = init_detector(str(cfg_path), str(weight_path), device=self.device)
                logger.info("Loaded Co-DETR/Co-DINO via mmdet: %s", self.model_name)
                return
        except ImportError:
            logger.debug("mmdet not installed (pip install mmdet mmengine mmcv)")
        except Exception as exc:
            logger.warning("mmdet load failed: %s\n%s", exc, traceback.format_exc())

        if not weight_path.exists():
            logger.error(
                "Co-DETR/Co-DINO checkpoint not found at %s. "
                "Run `python tools/download_detr_weights.py` to fetch it.",
                weight_path,
            )

        self._model = None
        logger.warning(
            "Co-DETR/Co-DINO model could not be loaded. "
            "Install mmdet stack, provide a valid config, and download weights. "
            "Falling back to no-op detector."
        )

    def _codetr_config_candidates(self) -> List[Path]:
        """Ordered candidate config paths for Co-DETR/Co-DINO MMDet projects."""
        cands: List[Path] = []
        repo_root = os.environ.get("SWAY_CODETR_REPO", "").strip()
        if repo_root:
            base = Path(repo_root).expanduser().resolve()
            cands.extend(
                [
                    base / "projects" / "configs" / "co_dino" / "co_dino_5scale_swin_large_1x_coco.py",
                    base / "projects" / "CO-DETR" / "configs" / "codino" / "co_dino_5scale_swin_l_16xb1_1x_coco.py",
                    base / "projects" / "CO-DETR" / "configs" / "codino" / "co_dino_5scale_swin_l_16xb1_16e_o365tococo.py",
                ]
            )

        # Check PYTHONPATH for externally mounted mmdetection repositories.
        py_path = os.environ.get("PYTHONPATH", "")
        for raw in py_path.split(os.pathsep):
            if not raw.strip():
                continue
            p = Path(raw).expanduser().resolve()
            cands.extend(
                [
                    p / "projects" / "configs" / "co_dino" / "co_dino_5scale_swin_large_1x_coco.py",
                    p / "projects" / "CO-DETR" / "configs" / "codino" / "co_dino_5scale_swin_l_16xb1_1x_coco.py",
                    p / "projects" / "CO-DETR" / "configs" / "codino" / "co_dino_5scale_swin_l_16xb1_16e_o365tococo.py",
                ]
            )

        # Optional local mmdetection checkout in this repo.
        local_mmdet = Path(__file__).resolve().parent.parent / "third_party" / "mmdetection"
        cands.extend(
            [
                local_mmdet / "projects" / "CO-DETR" / "configs" / "codino" / "co_dino_5scale_swin_l_16xb1_1x_coco.py",
                local_mmdet / "projects" / "CO-DETR" / "configs" / "codino" / "co_dino_5scale_swin_l_16xb1_16e_o365tococo.py",
                local_mmdet / "projects" / "configs" / "co_dino" / "co_dino_5scale_swin_large_1x_coco.py",
            ]
        )
        return cands

    def _resolve_codetr_config_path(self) -> Optional[Path]:
        """Resolve Co-DETR/Co-DINO config path from env or local checkouts."""
        env_cfg = os.environ.get("SWAY_MMDET_CODINO_CONFIG", "").strip()
        if env_cfg:
            p = Path(env_cfg).expanduser().resolve()
            if p.exists():
                return p
        # Optional dedicated override for co_detr primary.
        env_codetr = os.environ.get("SWAY_MMDET_CODETR_CONFIG", "").strip()
        if env_codetr:
            p = Path(env_codetr).expanduser().resolve()
            if p.exists():
                return p

        for p in self._codetr_config_candidates():
            if p.exists():
                return p

        return None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a BGR frame. Returns person detections only."""
        if self._model is None:
            return []

        if self._is_deimv2:
            return self._detect_deimv2(frame)
        elif self._is_ultralytics:
            return self._detect_ultralytics(frame)
        else:
            return self._detect_codetr(frame)

    def _detect_deimv2(self, frame: np.ndarray) -> List[Detection]:
        """DEIMv2-X inference path via official DEIMv2 runtime."""
        if self._model is None or not hasattr(self, "_deimv2_state"):
            return self._detect_ultralytics(frame)

        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1].copy()
        x = torch.from_numpy(rgb).to(self.device).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)
        in_size = int(getattr(self, "_deimv2_input_size", 640) or 640)
        x = F.interpolate(x, size=(in_size, in_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        orig = torch.tensor([[w, h]], device=self.device)

        with torch.no_grad():
            outputs = self._model(x, orig)

        if not outputs or not isinstance(outputs, list):
            return []
        out = outputs[0]
        labels = out.get("labels")
        boxes = out.get("boxes")
        scores = out.get("scores")
        if labels is None or boxes is None or scores is None:
            return []

        labels_np = labels.detach().cpu().numpy().astype(int)
        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        detections: List[Detection] = []
        for box, conf, cls in zip(boxes_np, scores_np, labels_np):
            if cls != 0 or conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(float, box)
            x1 = max(0.0, min(x1, float(w - 1)))
            y1 = max(0.0, min(y1, float(h - 1)))
            x2 = max(0.0, min(x2, float(w - 1)))
            y2 = max(0.0, min(y2, float(h - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(bbox=np.array([x1, y1, x2, y2], dtype=np.float32), confidence=float(conf), class_id=0))
        return detections

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            frame,
            imgsz=self.detect_size,
            conf=self.conf_threshold,
            classes=[0],  # person only
            verbose=False,
            device=self.device,
        )
        detections: List[Detection] = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for box, conf, c in zip(boxes, confs, cls):
                    if c == 0:
                        detections.append(Detection(
                            bbox=box.astype(np.float32),
                            confidence=float(conf),
                            class_id=0,
                        ))
        return detections

    def _detect_codetr(self, frame: np.ndarray) -> List[Detection]:
        """Run Co-DETR / Co-DINO inference via detrex or mmdet."""
        detections: List[Detection] = []

        # mmdet path
        try:
            from mmdet.apis import inference_detector
            result = inference_detector(self._model, frame)
            if hasattr(result, "pred_instances"):
                pred = result.pred_instances
                boxes = pred.bboxes.cpu().numpy()
                scores = pred.scores.cpu().numpy()
                labels = pred.labels.cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    if label == 0 and score >= self.conf_threshold:
                        detections.append(Detection(
                            bbox=box.astype(np.float32),
                            confidence=float(score),
                            class_id=0,
                        ))
                return detections
            # mmdet 2.x returns list[np.ndarray], one array per class.
            if isinstance(result, (list, tuple)) and len(result) > 0:
                person = result[0]
                if isinstance(person, np.ndarray) and person.size > 0:
                    for row in person:
                        if row.shape[0] >= 5 and float(row[4]) >= self.conf_threshold:
                            detections.append(
                                Detection(
                                    bbox=np.asarray(row[:4], dtype=np.float32),
                                    confidence=float(row[4]),
                                    class_id=0,
                                )
                            )
                return detections
        except (ImportError, AttributeError):
            pass
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            logger.warning("Co-DETR/Co-DINO inference failed: %s", exc)

        return detections

    @property
    def is_nms_free(self) -> bool:
        """DETR detectors produce deduplicated detections — NMS should be skipped."""
        return True
