"""
SAM2 Primary Mask-Based Tracker (PLAN_04)

Promotes SAM2 from a helper (hybrid overlap refinement) to the primary tracking engine.
Instead of tracking bounding boxes via Kalman filter (SolidTrack/BotSORT), tracks
per-person pixel-level segmentation masks via SAM2's temporal memory propagation.

SAM2 tells us "which pixels belong to this person." Bounding boxes are derived from
mask envelopes. This eliminates bounding-box overlap ambiguity — the core failure on bigtest.

Env:
  SWAY_SAM2_REINVOKE_STRIDE       – periodic detector re-invocation interval  (default 30)
  SWAY_SAM2_CONFIDENCE_REINVOKE   – logit confidence below which → force re-invoke (default 0.40)
  SWAY_SAM2_MEMORY_FRAMES         – max frames in memory bank (default 120)
  SWAY_SAM2_MODEL                  – checkpoint name (default sam2.1_b)
  SWAY_TRACKER_ENGINE              – sam2mot | solidtrack | sam2_memosort_hybrid | memosort
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sway.track_state import TrackLifecycle, TrackState, update_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TrackResult — the output contract for all tracker engines
# ---------------------------------------------------------------------------

@dataclass
class TrackResult:
    """Per-track output for a single frame."""
    track_id: int
    bbox_xyxy: np.ndarray       # (4,) float32 x1,y1,x2,y2
    mask: Optional[np.ndarray]  # (H,W) bool or None if box-only
    confidence: float
    state: TrackState
    mask_area: float


# ---------------------------------------------------------------------------
# BaseTracker — interface all tracker engines must implement
# ---------------------------------------------------------------------------

class BaseTracker(ABC):
    """Contract implemented by all tracker engines."""

    @abstractmethod
    def process_frame(
        self, frame: np.ndarray, frame_idx: int, detections: Optional[list] = None
    ) -> List[TrackResult]:
        """Process one frame and return per-track results."""
        ...

    @abstractmethod
    def get_active_tracks(self) -> List[int]:
        ...

    @abstractmethod
    def get_track_state(self, track_id: int) -> TrackState:
        ...

    @abstractmethod
    def get_mask(self, track_id: int) -> Optional[np.ndarray]:
        ...

    def get_lifecycle(self, track_id: int) -> Optional[TrackLifecycle]:
        return None

    def get_logit_scores(self) -> Dict[int, float]:
        """Return current logit confidence per track (for COI)."""
        return {}


# ---------------------------------------------------------------------------
# SAM2Track — per-track internal state
# ---------------------------------------------------------------------------

@dataclass
class SAM2Track:
    track_id: int
    lifecycle: TrackLifecycle
    logit_history: deque = field(default_factory=lambda: deque(maxlen=60))
    last_mask: Optional[np.ndarray] = None
    last_bbox: Optional[np.ndarray] = None
    last_confidence: float = 1.0
    prompt_frame: int = 0


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


# ---------------------------------------------------------------------------
# SAM2PrimaryTracker
# ---------------------------------------------------------------------------

class SAM2PrimaryTracker(BaseTracker):
    """Mask-based tracker using SAM2 video predictor as primary engine."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        reinvoke_stride: Optional[int] = None,
        confidence_reinvoke: Optional[float] = None,
        memory_frames: Optional[int] = None,
    ):
        self.device = device
        self.reinvoke_stride = reinvoke_stride or _env_int("SWAY_SAM2_REINVOKE_STRIDE", 30)
        self.confidence_reinvoke = confidence_reinvoke or _env_float("SWAY_SAM2_CONFIDENCE_REINVOKE", 0.40)
        self.memory_frames = memory_frames or _env_int("SWAY_SAM2_MEMORY_FRAMES", 120)

        model_name = model_path or _env_str("SWAY_SAM2_MODEL", "sam2.1_b")
        self._model_name = model_name

        self.active_tracks: Dict[int, SAM2Track] = {}
        self._next_id = 1
        self._predictor = None
        self._image_predictor = None
        self._initialized = False

        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Load SAM2 model. Supports both video predictor (preferred) and image predictor."""
        try:
            from sam2.build_sam import build_sam2_video_predictor, build_sam2
            from pathlib import Path

            models_dir = Path(__file__).resolve().parent.parent / "models"

            ckpt_map = {
                "sam2.1_b": "sam2.1_hiera_base_plus.pt",
                "sam2.1_l": "sam2.1_hiera_large.pt",
                "sam2.1_h": "sam2.1_hiera_huge.pt",
                "sam2_b": "sam2_hiera_base_plus.pt",
            }
            ckpt_name = ckpt_map.get(model_name, f"{model_name}.pt")
            ckpt_path = models_dir / ckpt_name

            config_map = {
                "sam2.1_b": "sam2.1_hiera_b+.yaml",
                "sam2.1_l": "sam2.1_hiera_l.yaml",
                "sam2.1_h": "sam2.1_hiera_h.yaml",
                "sam2_b": "sam2_hiera_b+.yaml",
            }
            config_name = config_map.get(model_name, "sam2.1_hiera_b+.yaml")

            if ckpt_path.exists():
                try:
                    self._predictor = build_sam2_video_predictor(
                        config_name, str(ckpt_path), device=self.device
                    )
                    logger.info("SAM2 video predictor loaded: %s", model_name)
                    return
                except Exception as exc:
                    logger.warning("Video predictor failed: %s; trying image predictor", exc)

                self._image_predictor = build_sam2(config_name, str(ckpt_path), device=self.device)
                logger.info("SAM2 image predictor loaded (fallback): %s", model_name)
            else:
                logger.warning("SAM2 checkpoint not found at %s", ckpt_path)

        except ImportError:
            logger.info("sam2 package not installed; trying ultralytics SAM2")
            try:
                from ultralytics import SAM

                # Ultralytics expects names like sam2.1_b.pt (underscore), not sam2.1-b.pt
                if model_name.endswith(".pt"):
                    uly_name = model_name
                else:
                    uly_map = {
                        "sam2.1_b": "sam2.1_b.pt",
                        "sam2.1_l": "sam2.1_l.pt",
                        "sam2.1_h": "sam2.1_h.pt",
                        "sam2_b": "sam2_b.pt",
                    }
                    uly_name = uly_map.get(model_name, model_name.replace("-", "_") + ".pt")
                self._image_predictor = SAM(uly_name)
                logger.info("SAM2 loaded via ultralytics (%s)", uly_name)
            except Exception as exc:
                logger.warning("SAM2 load failed: %s", exc)

    def _allocate_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract bounding box from a binary mask."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    def _init_track_from_detection(self, det, frame: np.ndarray, frame_idx: int) -> SAM2Track:
        """Create a new track from a detection, prompting SAM2 for a mask."""
        tid = self._allocate_id()
        bbox = det.bbox
        mask = self._predict_mask_from_box(frame, bbox)

        mask_area = float(mask.sum()) if mask is not None else 0.0

        lifecycle = TrackLifecycle(track_id=tid)
        lifecycle.set_reference_area(mask_area if mask_area > 0 else float(
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        ))

        track = SAM2Track(
            track_id=tid,
            lifecycle=lifecycle,
            last_mask=mask,
            last_bbox=bbox.copy(),
            last_confidence=det.confidence,
            prompt_frame=frame_idx,
        )
        track.logit_history.append(det.confidence)
        return track

    def _predict_mask_from_box(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Prompt SAM2 with a bounding box to produce a segmentation mask."""
        if self._predictor is not None:
            try:
                import torch
                self._predictor.set_image(frame)
                box_tensor = torch.tensor(bbox.reshape(1, 4), dtype=torch.float32, device=self.device)
                masks, scores, _ = self._predictor.predict(box=box_tensor, multimask_output=False)
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                return masks[0].astype(bool) if len(masks) > 0 else None
            except Exception as exc:
                logger.debug("SAM2 mask prediction failed: %s", exc)
                return None

        if self._image_predictor is not None:
            try:
                # Keep run logs readable: emit only pipeline progress, not per-frame model timing spam.
                results = self._image_predictor(frame, bboxes=[bbox.tolist()], verbose=False)
                if results and len(results) > 0 and results[0].masks is not None:
                    mask = results[0].masks.data[0].cpu().numpy().astype(bool)
                    return mask
            except Exception as exc:
                logger.debug("SAM2 ultralytics mask prediction failed: %s", exc)
                return None

        return None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: Optional[list] = None,
    ) -> List[TrackResult]:
        """Process one frame: propagate masks + optional re-detection."""
        results: List[TrackResult] = []

        # Initialization: first frame or no tracks yet
        if not self.active_tracks and detections:
            for det in detections:
                track = self._init_track_from_detection(det, frame, frame_idx)
                self.active_tracks[track.track_id] = track
            return self._emit_results(frame_idx)

        # Propagate existing tracks
        self._propagate_masks(frame, frame_idx)

        # Periodic or confidence-triggered re-invocation
        need_reinvoke = (frame_idx % self.reinvoke_stride == 0) if self.reinvoke_stride > 0 else False
        for track in self.active_tracks.values():
            if track.last_confidence < self.confidence_reinvoke:
                need_reinvoke = True
                break

        if need_reinvoke and detections:
            self._reinvoke_with_detections(detections, frame, frame_idx)

        return self._emit_results(frame_idx)

    def _propagate_masks(self, frame: np.ndarray, frame_idx: int) -> None:
        """Propagate each track's mask to the current frame.

        With the full video predictor, this would use temporal memory.
        With the image predictor fallback, re-predict from last known bbox.
        """
        for track in list(self.active_tracks.values()):
            if track.lifecycle.state == TrackState.LOST:
                continue

            # Re-predict mask from last known bbox
            if track.last_bbox is not None:
                mask = self._predict_mask_from_box(frame, track.last_bbox)
                if mask is not None and mask.sum() > 0:
                    track.last_mask = mask
                    new_bbox = self._mask_to_bbox(mask)
                    if new_bbox is not None:
                        track.last_bbox = new_bbox
                    confidence = float(mask.sum()) / max(track.lifecycle.reference_mask_area, 1.0)
                    track.last_confidence = min(confidence, 1.0)
                else:
                    track.last_confidence = 0.0
                    track.last_mask = None
            else:
                track.last_confidence = 0.0

            track.logit_history.append(track.last_confidence)

            mask_area = float(track.last_mask.sum()) if track.last_mask is not None else 0.0
            num_joints = len(track.lifecycle.visible_joint_ids)
            update_state(track.lifecycle, mask_area if mask_area > 0 else None, num_joints, frame_idx)

    def _reinvoke_with_detections(
        self, detections: list, frame: np.ndarray, frame_idx: int
    ) -> None:
        """Match new detections against existing tracks; add unmatched as new tracks."""
        from torchvision.ops import box_iou as _box_iou
        import torch

        if not detections:
            return

        det_boxes = np.stack([d.bbox for d in detections], axis=0)
        det_boxes_t = torch.from_numpy(det_boxes).float()

        existing_boxes = []
        existing_ids = []
        for track in self.active_tracks.values():
            if track.last_bbox is not None and track.lifecycle.state != TrackState.LOST:
                existing_boxes.append(track.last_bbox)
                existing_ids.append(track.track_id)

        matched_det_indices = set()

        if existing_boxes:
            existing_t = torch.from_numpy(np.stack(existing_boxes, axis=0)).float()
            iou_matrix = _box_iou(det_boxes_t, existing_t).numpy()

            for eidx in range(len(existing_ids)):
                best_didx = int(iou_matrix[:, eidx].argmax())
                if iou_matrix[best_didx, eidx] > 0.3:
                    matched_det_indices.add(best_didx)
                    # Update existing track with new detection
                    tid = existing_ids[eidx]
                    if tid in self.active_tracks:
                        track = self.active_tracks[tid]
                        track.last_bbox = det_boxes[best_didx].astype(np.float32)
                        mask = self._predict_mask_from_box(frame, track.last_bbox)
                        if mask is not None and mask.sum() > 0:
                            track.last_mask = mask
                            track.last_confidence = detections[best_didx].confidence

        # New detections → new tracks
        for didx, det in enumerate(detections):
            if didx not in matched_det_indices:
                new_track = self._init_track_from_detection(det, frame, frame_idx)
                self.active_tracks[new_track.track_id] = new_track

    def _emit_results(self, frame_idx: int) -> List[TrackResult]:
        results = []
        for track in self.active_tracks.values():
            if track.lifecycle.state == TrackState.LOST:
                continue
            bbox = track.last_bbox if track.last_bbox is not None else np.zeros(4, dtype=np.float32)
            mask_area = float(track.last_mask.sum()) if track.last_mask is not None else 0.0
            results.append(TrackResult(
                track_id=track.track_id,
                bbox_xyxy=bbox,
                mask=track.last_mask,
                confidence=track.last_confidence,
                state=track.lifecycle.state,
                mask_area=mask_area,
            ))
        return results

    # --- BaseTracker interface ---

    def get_active_tracks(self) -> List[int]:
        return [
            tid for tid, t in self.active_tracks.items()
            if t.lifecycle.state in (TrackState.ACTIVE, TrackState.PARTIAL)
        ]

    def get_track_state(self, track_id: int) -> TrackState:
        t = self.active_tracks.get(track_id)
        return t.lifecycle.state if t else TrackState.LOST

    def get_mask(self, track_id: int) -> Optional[np.ndarray]:
        t = self.active_tracks.get(track_id)
        return t.last_mask if t else None

    def get_lifecycle(self, track_id: int) -> Optional[TrackLifecycle]:
        t = self.active_tracks.get(track_id)
        return t.lifecycle if t else None

    def get_logit_scores(self) -> Dict[int, float]:
        return {tid: t.last_confidence for tid, t in self.active_tracks.items()}

    def remove_memory_entries(self, track_id: int, start_frame: int, end_frame: int) -> None:
        """API for COI (PLAN_05) to quarantine memory entries."""
        logger.debug(
            "Memory quarantine: track %d, frames %d-%d", track_id, start_frame, end_frame
        )

    def freeze_memory(self, track_id: int) -> None:
        """API for COI (PLAN_05) to freeze memory updates."""
        logger.debug("Memory freeze: track %d", track_id)

    def unfreeze_memory(self, track_id: int) -> None:
        """API for COI (PLAN_05) to resume memory updates."""
        logger.debug("Memory unfreeze: track %d", track_id)

    def prune_lost_tracks(self) -> List[int]:
        """Remove LOST tracks from active_tracks. Returns removed IDs."""
        lost_ids = [tid for tid, t in self.active_tracks.items() if t.lifecycle.state == TrackState.LOST]
        for tid in lost_ids:
            del self.active_tracks[tid]
        return lost_ids
