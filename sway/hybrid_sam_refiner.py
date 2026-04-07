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

ROI crop (default on): when overlap triggers SAM, inference runs on the union of all
boxes that participate in a high-IoU pair (plus margin), not the full frame — then masks
are pasted back to full resolution for box tightening. Set ``SWAY_HYBRID_SAM_ROI_CROP=0``
to use the legacy full-frame path.

Data contract (same as BoxMOT / tracker):
  dets: float32 array shape (N, 6) — columns [x1, y1, x2, y2, conf, cls].

Default on; set SWAY_HYBRID_SAM_OVERLAP=0 to disable (production ``main.py`` reapplies overlap + ROI crop + pad after
params unless SWAY_UNLOCK_HYBRID_SAM_TUNING=1 — see ``MASTER_PIPELINE_GUIDELINE.md`` §6.4.1). Optional env:
  SWAY_HYBRID_SAM_IOU_TRIGGER   — max pairwise IoU above this triggers SAM (default 0.42)
  SWAY_HYBRID_SAM_MIN_DETS        — minimum person count to consider overlap (default 2)
  SWAY_HYBRID_SAM_WEIGHTS         — Ultralytics SAM checkpoint (default sam2.1_b.pt)
  SWAY_HYBRID_SAM_MASK_THRESH     — binarize masks at this prob (default 0.5)
  SWAY_HYBRID_SAM_BBOX_PAD        — pixel pad on mask-derived boxes (default 2)
  SWAY_HYBRID_SAM_ROI_CROP        — 1 = ROI union crop for SAM (default 1)
  SWAY_HYBRID_SAM_ROI_PAD_FRAC    — expand ROI union by this fraction of its size (default 0.1)
  SWAY_HYBRID_SAM_WEAK_CUES       — optional extra gate: skip SAM when overlap is high but boxes
                                    match the previous frame (conf + height stable). Default 0.
  SWAY_HYBRID_WEAK_CONF_DELTA     — max |Δconf| vs previous matched box to count as stable (default 0.08)
  SWAY_HYBRID_WEAK_HEIGHT_FRAC    — max relative height change vs previous match (default 0.12)
  SWAY_HYBRID_WEAK_MATCH_IOU      — min IoU to match a det to previous frame (default 0.25)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


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
        "roi_crop": _truthy("SWAY_HYBRID_SAM_ROI_CROP", True),
        "roi_pad_frac": _f("SWAY_HYBRID_SAM_ROI_PAD_FRAC", 0.1),
        # Optional Hybrid-SORT–style weak cues: skip SAM when overlap is high but boxes look temporally stable.
        "weak_cues": _truthy("SWAY_HYBRID_SAM_WEAK_CUES", False),
        "weak_conf_delta": _f("SWAY_HYBRID_WEAK_CONF_DELTA", 0.08),
        "weak_height_frac": _f("SWAY_HYBRID_WEAK_HEIGHT_FRAC", 0.12),
        "weak_match_min_iou": _f("SWAY_HYBRID_WEAK_MATCH_IOU", 0.25),
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


def overlap_cluster_indices(xyxy: np.ndarray, iou_trigger: float) -> Set[int]:
    """Indices that appear in any pair with IoU >= iou_trigger."""
    n = int(xyxy.shape[0])
    involved: Set[int] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if _iou_xyxy(xyxy[i], xyxy[j]) >= float(iou_trigger):
                involved.add(i)
                involved.add(j)
    return involved


def union_xyxy_with_pad(
    xyxy: np.ndarray,
    indices: List[int],
    frame_h: int,
    frame_w: int,
    pad_frac: float,
) -> Tuple[int, int, int, int]:
    """Integer ROI (x1,y1,x2,y2) clipped to the frame; pad_frac expands the union."""
    sub = xyxy[indices].astype(np.float64)
    u_x1, u_y1 = float(sub[:, 0].min()), float(sub[:, 1].min())
    u_x2, u_y2 = float(sub[:, 2].max()), float(sub[:, 3].max())
    bw = max(1.0, u_x2 - u_x1)
    bh = max(1.0, u_y2 - u_y1)
    px = float(pad_frac) * bw
    py = float(pad_frac) * bh
    cx1 = max(0, int(np.floor(u_x1 - px)))
    cy1 = max(0, int(np.floor(u_y1 - py)))
    cx2 = min(frame_w, int(np.ceil(u_x2 + px)))
    cy2 = min(frame_h, int(np.ceil(u_y2 + py)))
    if cx2 <= cx1:
        cx2 = min(frame_w, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(frame_h, cy1 + 1)
    return cx1, cy1, cx2, cy2


def overlap_stats(xyxy: np.ndarray, cfg: Dict[str, Any]) -> Tuple[float, bool]:
    """Returns (max_pairwise_iou, should_run_sam)."""
    if xyxy.shape[0] < cfg["min_dets"]:
        return 0.0, False
    mp = max_pairwise_iou(xyxy)
    if not cfg["enabled"]:
        return mp, False
    return mp, mp >= cfg["iou_trigger"]


def _bbox_height_xyxy(row: np.ndarray) -> float:
    return max(0.0, float(row[3] - row[1]))


def _greedy_match_curr_to_prev(
    xyxy_curr: np.ndarray,
    xyxy_prev: np.ndarray,
    min_iou: float,
) -> List[int]:
    """Greedy one-to-one match of current rows to previous frame by IoU."""
    n = int(xyxy_curr.shape[0])
    m = int(xyxy_prev.shape[0])
    match = [-1] * n
    if m == 0:
        return match
    used = [False] * m
    thr = float(min_iou)
    for i in range(n):
        best_j = -1
        best_iou = thr
        for j in range(m):
            if used[j]:
                continue
            iou_val = _iou_xyxy(xyxy_curr[i], xyxy_prev[j])
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            match[i] = best_j
    return match


def _max_iou_pair_indices(xyxy: np.ndarray) -> Optional[Tuple[int, int]]:
    n = int(xyxy.shape[0])
    if n < 2:
        return None
    best_iou = -1.0
    best_pair: Optional[Tuple[int, int]] = None
    for i in range(n):
        for j in range(i + 1, n):
            v = _iou_xyxy(xyxy[i], xyxy[j])
            if v > best_iou:
                best_iou = v
                best_pair = (i, j)
    return best_pair


def weak_cues_say_ambiguous(
    dets: np.ndarray,
    prev_out: Optional[np.ndarray],
    cfg: Dict[str, Any],
) -> bool:
    """
    When overlap triggers SAM, weak cues can veto if the worst-overlap pair looks
    temporally stable vs the previous frame (safe overlap). True => run SAM.
    """
    if prev_out is None or prev_out.size == 0 or int(prev_out.shape[0]) == 0:
        return True
    xyxy_c = dets[:, :4].astype(np.float32)
    xyxy_p = prev_out[:, :4].astype(np.float32)
    min_match_iou = float(cfg.get("weak_match_min_iou", 0.25))
    match = _greedy_match_curr_to_prev(xyxy_c, xyxy_p, min_match_iou)
    pair = _max_iou_pair_indices(xyxy_c)
    if pair is None:
        return True
    i, j = pair
    h_thr = float(cfg["weak_height_frac"])
    c_thr = float(cfg["weak_conf_delta"])
    for idx in (i, j):
        pj = match[idx]
        if pj < 0:
            return True
        h_c = _bbox_height_xyxy(dets[idx])
        h_p = _bbox_height_xyxy(prev_out[pj])
        denom = max(h_c, h_p, 1e-3)
        if abs(h_c - h_p) / denom > h_thr:
            return True
        conf_c = float(dets[idx, 4]) if dets.shape[1] > 4 else 1.0
        conf_p = float(prev_out[pj, 4]) if prev_out.shape[1] > 4 else 1.0
        if abs(conf_c - conf_p) > c_thr:
            return True
    return False


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


def _xyxy_to_crop_space(
    xyxy: np.ndarray,
    cx1: int,
    cy1: int,
    cw: int,
    ch: int,
) -> np.ndarray:
    """Shift boxes into crop coordinates and clamp to valid pixels."""
    b = xyxy.astype(np.float32).copy()
    b[:, [0, 2]] -= float(cx1)
    b[:, [1, 3]] -= float(cy1)
    b[:, 0] = np.clip(b[:, 0], 0.0, max(0.0, float(cw - 1)))
    b[:, 1] = np.clip(b[:, 1], 0.0, max(0.0, float(ch - 1)))
    b[:, 2] = np.clip(b[:, 2], 0.0, float(cw))
    b[:, 3] = np.clip(b[:, 3], 0.0, float(ch))
    # ensure positive area where possible
    for i in range(b.shape[0]):
        if b[i, 2] <= b[i, 0]:
            b[i, 2] = min(float(cw), b[i, 0] + 1.0)
        if b[i, 3] <= b[i, 1]:
            b[i, 3] = min(float(ch), b[i, 1] + 1.0)
    return b


def _sam_predict_masks(
    sam: Any,
    frame_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Run Ultralytics SAM on frame_bgr with boxes_xyxy (N,4) in the same pixel space as frame_bgr.
    Returns float masks stacked (N, H, W) matching frame_bgr shape, or None on failure.
    """
    h, w = frame_bgr.shape[:2]
    try:
        import torch

        with torch.inference_mode():
            res = sam.predict(frame_bgr, bboxes=boxes_xyxy, verbose=False)
        r0 = res[0] if isinstance(res, list) else res
    except Exception:
        return None
    if r0.masks is None or r0.masks.data is None:
        return None
    masks = r0.masks.data.detach().cpu().numpy()
    n = int(masks.shape[0])
    out = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
        m = masks[i]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        out[i] = m
    return out


def _apply_sam_masks_to_dets(
    dets: np.ndarray,
    masks_hw: np.ndarray,
    det_indices: List[int],
    full_h: int,
    full_w: int,
    mask_thresh: float,
    bbox_pad: int,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    """
    Tighten boxes and build per-det mask crops in full-image coordinates.
    masks_hw: (K, Hm, Wm); det_indices maps row k -> global det index.
    If roi is set, masks are defined on the ROI; they are embedded into full-frame space first.
    """
    out = dets.copy()
    nd = int(len(dets))
    per_det_masks: List[Optional[np.ndarray]] = [None] * nd
    thr = float(mask_thresh)
    for k, gi in enumerate(det_indices):
        if k >= masks_hw.shape[0]:
            break
        m = masks_hw[k]
        if roi is not None:
            rcx1, rcy1, rcx2, rcy2 = roi
            ch, cw = rcy2 - rcy1, rcx2 - rcx1
            if m.shape[0] != ch or m.shape[1] != cw:
                m = cv2.resize(m.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR)
            full_m = np.zeros((full_h, full_w), dtype=np.float32)
            full_m[rcy1:rcy2, rcx1:rcx2] = np.maximum(
                full_m[rcy1:rcy2, rcx1:rcx2], m.astype(np.float32)
            )
            m_work = full_m
        else:
            m_work = m
            if m_work.shape[0] != full_h or m_work.shape[1] != full_w:
                m_work = cv2.resize(
                    m_work.astype(np.float32), (full_w, full_h), interpolation=cv2.INTER_LINEAR
                )
        tight = _mask_to_xyxy(m_work, full_h, full_w, mask_thresh, bbox_pad)
        if tight is not None:
            out[gi, :4] = tight
            x1, y1, x2, y2 = (
                int(round(tight[0])),
                int(round(tight[1])),
                int(round(tight[2])),
                int(round(tight[3])),
            )
            x1 = max(0, min(full_w - 1, x1))
            y1 = max(0, min(full_h - 1, y1))
            x2 = max(0, min(full_w, x2))
            y2 = max(0, min(full_h, y2))
            if x2 > x1 and y2 > y1:
                per_det_masks[gi] = (m_work[y1:y2, x1:x2] > thr).astype(bool)
    return out.astype(np.float32), per_det_masks


@dataclass
class HybridSamRefiner:
    """Lazy-loads SAM; refines BoxMOT-format dets when overlap is high."""

    cfg: Dict[str, Any]
    _sam: Any = None
    frames_seen: int = 0
    frames_sam_used: int = 0
    sam_calls: int = 0
    weak_cue_skip_frames: int = 0
    _prev_dets_out: Optional[np.ndarray] = field(default=None, repr=False)

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
            "roi_crop": False,
            "roi_box": None,
            "weak_cues_skipped_sam": False,
        }
        if dets.size == 0 or dets.shape[0] == 0:
            return dets, meta

        nd = int(len(dets))
        meta["per_det_masks"] = [None] * nd

        xyxy = dets[:, :4].astype(np.float32)
        mp, need = overlap_stats(xyxy, self.cfg)
        meta["max_iou"] = float(mp)
        if not need:
            self._prev_dets_out = np.asarray(dets, dtype=np.float32).copy()
            return dets, meta

        if bool(self.cfg.get("weak_cues")) and not weak_cues_say_ambiguous(
            dets, self._prev_dets_out, self.cfg
        ):
            meta["weak_cues_skipped_sam"] = True
            self.weak_cue_skip_frames += 1
            self._prev_dets_out = np.asarray(dets, dtype=np.float32).copy()
            return dets, meta

        h, w = frame_bgr.shape[:2]
        sam = self._model()
        self.sam_calls += 1
        self.frames_sam_used += 1

        thr = float(self.cfg["mask_thresh"])
        pad = int(self.cfg["bbox_pad"])
        use_roi = bool(self.cfg.get("roi_crop", True))
        involved = sorted(overlap_cluster_indices(xyxy, float(self.cfg["iou_trigger"])))
        if not involved:
            involved = list(range(nd))

        if use_roi and involved:
            meta["roi_crop"] = True
            roi = union_xyxy_with_pad(
                xyxy,
                involved,
                h,
                w,
                float(self.cfg.get("roi_pad_frac", 0.1)),
            )
            meta["roi_box"] = list(roi)
            rcx1, rcy1, rcx2, rcy2 = roi
            crop = frame_bgr[rcy1:rcy2, rcx1:rcx2]
            ch, cw = crop.shape[:2]
            sub = xyxy[involved]
            boxes_crop = _xyxy_to_crop_space(sub, rcx1, rcy1, cw, ch)
            masks = _sam_predict_masks(sam, crop, boxes_crop)
            if masks is None:
                self._prev_dets_out = np.asarray(dets, dtype=np.float32).copy()
                return dets, meta
            out, per_det_masks = _apply_sam_masks_to_dets(
                dets,
                masks,
                involved,
                h,
                w,
                thr,
                pad,
                roi=roi,
            )
            meta["used_sam"] = True
            meta["n_out"] = int(len(out))
            meta["per_det_masks"] = per_det_masks
            self._prev_dets_out = np.asarray(out, dtype=np.float32).copy()
            return out, meta

        # Full-frame SAM (legacy or when everyone is in the overlap cluster)
        masks = _sam_predict_masks(sam, frame_bgr, xyxy)
        if masks is None:
            self._prev_dets_out = np.asarray(dets, dtype=np.float32).copy()
            return dets, meta

        all_idx = list(range(min(nd, masks.shape[0])))
        out, per_det_masks = _apply_sam_masks_to_dets(
            dets, masks, all_idx, h, w, thr, pad, roi=None
        )
        meta["used_sam"] = True
        meta["n_out"] = int(len(out))
        meta["per_det_masks"] = per_det_masks
        self._prev_dets_out = np.asarray(out, dtype=np.float32).copy()
        return out, meta

    def summary(self) -> Dict[str, Any]:
        return {
            "hybrid_sam_frames_total": self.frames_seen,
            "hybrid_sam_frames_refined": self.frames_sam_used,
            "hybrid_sam_predict_calls": self.sam_calls,
            "hybrid_sam_weak_cue_skips": int(self.weak_cue_skip_frames),
            "hybrid_sam_weak_cues_enabled": bool(self.cfg.get("weak_cues")),
        }
