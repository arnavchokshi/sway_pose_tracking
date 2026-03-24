"""
Hybrid detection refinement: BoxMOT stays the tracker; SAM2 tightens boxes when
person detections overlap heavily (occlusion / crowd).

SAM2 → BoxMOT ID handoff (verified contract):
  - SAM does **not** assign track IDs. Ultralytics SAM returns one mask per input box,
    in the **same row order** as ``dets[:, :4]`` passed to ``sam.predict``.
  - ``refine_person_dets`` writes tightened ``out[i, :4]`` and ``per_det_masks[i]`` for
    the same row index ``i`` (still N rows).
  - ``DeepOcSort.update(dets, frame)`` consumes those rows and assigns integer track IDs.
  - ``assign_sam_masks_to_tracker_output`` maps each tracker output row back to a **det**
    row by greedy IoU (tracker may reorder rows), then attaches ``per_det_masks[j]`` so
    the mask follows the person, not the row index.

Data contract (same as BoxMOT / tracker):
  dets: float32 array shape (N, 6) — columns [x1, y1, x2, y2, conf, cls].

Default on; set SWAY_HYBRID_SAM_OVERLAP=0 to disable. Optional env:
  SWAY_HYBRID_SAM_IOU_TRIGGER   — max pairwise IoU above this triggers SAM (default 0.42)
  SWAY_HYBRID_SAM_MIN_DETS        — minimum person count to consider overlap (default 2)
  SWAY_HYBRID_SAM_WEIGHTS         — Ultralytics SAM checkpoint (default sam2.1_b.pt)
  SWAY_HYBRID_SAM_MASK_THRESH     — binarize masks at this prob (default 0.5)
  SWAY_HYBRID_SAM_BBOX_PAD        — pixel pad on mask-derived boxes (default 2)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def resolve_hybrid_sam_weights(spec: str) -> str:
    """
    Map SWAY_HYBRID_SAM_WEIGHTS / default filename to an on-disk path when present:
    models/<name>, then repo root/<name>, else return spec (hub / Ultralytics id).
    """
    spec = (spec or "").strip()
    if not spec:
        spec = "sam2.1_b.pt"
    p = Path(spec).expanduser()
    if p.is_file():
        return str(p.resolve())
    repo = Path(__file__).resolve().parent.parent
    for base in (repo / "models", repo):
        cand = base / spec
        if cand.is_file():
            return str(cand.resolve())
    return spec

import cv2
import numpy as np


def load_hybrid_sam_config() -> Dict[str, Any]:
    def _truthy(name: str, default: bool = False) -> bool:
        v = os.environ.get(name, "").strip().lower()
        if not v:
            return default
        return v in ("1", "true", "yes", "on")

    def _f(name: str, default: float) -> float:
        v = os.environ.get(name, "").strip()
        return float(v) if v else default

    def _i(name: str, default: int) -> int:
        v = os.environ.get(name, "").strip()
        return int(v) if v else default

    return {
        "enabled": _truthy("SWAY_HYBRID_SAM_OVERLAP", True),
        "iou_trigger": _f("SWAY_HYBRID_SAM_IOU_TRIGGER", 0.42),
        "min_dets": _i("SWAY_HYBRID_SAM_MIN_DETS", 2),
        "weights": resolve_hybrid_sam_weights(
            os.environ.get("SWAY_HYBRID_SAM_WEIGHTS", "sam2.1_b.pt")
        ),
        "mask_thresh": _f("SWAY_HYBRID_SAM_MASK_THRESH", 0.5),
        "bbox_pad": _i("SWAY_HYBRID_SAM_BBOX_PAD", 2),
    }


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for two boxes as (4,) xyxy."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    b1 = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    u = a1 + b1 - inter
    return float(inter / u) if u > 0 else 0.0


def max_pairwise_iou(xyxy: np.ndarray) -> float:
    """Maximum IoU over all unordered pairs (i < j)."""
    n = xyxy.shape[0]
    if n < 2:
        return 0.0
    m = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            m = max(m, _iou_xyxy(xyxy[i], xyxy[j]))
    return m


def overlap_stats(xyxy: np.ndarray, cfg: Dict[str, Any]) -> Tuple[float, bool]:
    """Returns (max_pairwise_iou, should_run_sam)."""
    if xyxy.shape[0] < cfg["min_dets"]:
        return 0.0, False
    mp = max_pairwise_iou(xyxy)
    if not cfg["enabled"]:
        return mp, False
    return mp, mp >= cfg["iou_trigger"]


def _mask_to_xyxy(
    mask: np.ndarray,
    frame_h: int,
    frame_w: int,
    mask_thresh: float,
    pad: int,
) -> Optional[np.ndarray]:
    m = mask > mask_thresh
    if not np.any(m):
        return None
    ys, xs = np.where(m)
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(frame_w - 1, int(xs.max()) + pad)
    y2 = min(frame_h - 1, int(ys.max()) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def track_color_bgr(track_id: int) -> tuple[int, int, int]:
    """Stable saturated BGR per track ID (for SAM mask overlays)."""
    h = (track_id * 47) % 180
    hsv = np.uint8([[[h, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def blend_sam_masks_for_tracks(
    frame_bgr: np.ndarray,
    masks_nhw: np.ndarray,
    track_ids: list[int],
    *,
    mask_thresh: float = 0.5,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Composite SAM instance masks onto the frame. masks_nhw: (N,H,W) float, same
    order as track_ids. Pixels outside masks stay original.
    """
    out = frame_bgr.astype(np.float32)
    h, w = frame_bgr.shape[:2]
    n = min(int(masks_nhw.shape[0]), len(track_ids))
    for i in range(n):
        m = masks_nhw[i]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mb = m > mask_thresh
        col = np.array(track_color_bgr(track_ids[i]), dtype=np.float32)
        out[mb] = out[mb] * (1.0 - alpha) + col * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class HybridSamRefiner:
    """Lazy-loads SAM; refines BoxMOT-format dets when overlap is high."""

    cfg: Dict[str, Any]
    _sam: Any = None
    frames_seen: int = 0
    frames_sam_used: int = 0
    sam_calls: int = 0

    def _model(self):
        if self._sam is None:
            from ultralytics import SAM

            self._sam = SAM(self.cfg["weights"])
        return self._sam

    def refine_person_dets(
        self,
        frame_bgr: np.ndarray,
        dets: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns (dets_out, meta) where dets_out is same shape/dtype as input (possibly unchanged).
        meta: used_sam (bool), max_iou (float), n_in (int), n_out (int), per_det_masks (list aligned
        with det rows — see module docstring for BoxMOT ID / mask handoff).
        """
        self.frames_seen += 1
        meta: Dict[str, Any] = {
            "used_sam": False,
            "max_iou": 0.0,
            "n_in": int(len(dets)),
            "n_out": int(len(dets)),
        }
        if dets.size == 0 or dets.shape[0] == 0:
            return dets, meta

        nd = int(len(dets))
        meta["per_det_masks"] = [None] * nd

        xyxy = dets[:, :4].astype(np.float32)
        mp, need = overlap_stats(xyxy, self.cfg)
        meta["max_iou"] = float(mp)
        if not need:
            return dets, meta

        h, w = frame_bgr.shape[:2]
        sam = self._model()
        self.sam_calls += 1
        self.frames_sam_used += 1

        try:
            import torch

            with torch.inference_mode():
                res = sam.predict(frame_bgr, bboxes=xyxy, verbose=False)
            r0 = res[0] if isinstance(res, list) else res
        except Exception:
            return dets, meta

        if r0.masks is None or r0.masks.data is None:
            return dets, meta

        masks = r0.masks.data.detach().cpu().numpy()
        out = dets.copy()
        per_det_masks: List[Optional[np.ndarray]] = [None] * nd
        n = min(len(dets), masks.shape[0])
        thr = float(self.cfg["mask_thresh"])
        for i in range(n):
            m = masks[i]
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            tight = _mask_to_xyxy(m, h, w, self.cfg["mask_thresh"], self.cfg["bbox_pad"])
            if tight is not None:
                out[i, :4] = tight
                x1, y1, x2, y2 = (int(round(tight[0])), int(round(tight[1])), int(round(tight[2])), int(round(tight[3])))
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 > x1 and y2 > y1:
                    per_det_masks[i] = (m[y1:y2, x1:x2] > thr).astype(bool)

        meta["used_sam"] = True
        meta["n_out"] = int(len(out))
        meta["per_det_masks"] = per_det_masks
        return out.astype(np.float32), meta

    def summary(self) -> Dict[str, Any]:
        return {
            "hybrid_sam_frames_total": self.frames_seen,
            "hybrid_sam_frames_refined": self.frames_sam_used,
            "hybrid_sam_predict_calls": self.sam_calls,
        }
