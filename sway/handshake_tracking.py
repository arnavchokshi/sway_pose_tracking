"""
Sway Handshake (optional Phase 1–3): IoU-triggered SAM custody + registry verification.

Enabled when ``SWAY_PHASE13_MODE=sway_handshake``.

- **Open floor:** pairwise box IoU below the hybrid trigger — update zonal HSV fingerprints
  from isolated YOLO boxes matched to the previous frame’s tracker output.
- **Custody:** when hybrid SAM runs (IoU ≥ ``SWAY_HYBRID_SAM_IOU_TRIGGER``, preset 0.10),
  optionally permute overlapping det rows so mask pixels best match registry profiles
  (Hungarian assignment every ``SWAY_HANDSHAKE_VERIFY_STRIDE`` SAM frames).

BoxMOT still runs every frame; SAM refines boxes/masks on the choreography zone; verification
reorders detections before ``tracker.update`` to reduce ID–mask slips.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sway.hybrid_sam_refiner import (
    _greedy_match_curr_to_prev,
    max_pairwise_iou,
    overlap_cluster_indices,
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None  # type: ignore[misc, assignment]


def phase13_handshake_enabled() -> bool:
    return os.environ.get("SWAY_PHASE13_MODE", "").strip().lower() in (
        "sway_handshake",
        "handshake",
    )


def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0))))


def _aspect_sim(a: float, b: float) -> float:
    da = abs(np.log((a + 1e-6) / (b + 1e-6)))
    return float(np.exp(-da))


def _profile_score(
    feat: np.ndarray,
    asp: float,
    prof: Optional[Tuple[np.ndarray, float]],
) -> float:
    if prof is None:
        return 0.25
    pf, pa = prof
    return _bhattacharyya(feat, pf) * 0.85 + _aspect_sim(asp, pa) * 0.15


def _extract_zonal_from_masked_region(
    frame_bgr: np.ndarray,
    mask_hw: np.ndarray,
    *,
    thr: float = 0.5,
) -> Optional[Tuple[np.ndarray, float]]:
    """HSV-H zonal histogram (top/bottom split on mask bbox) using only masked pixels."""
    mb = mask_hw > float(thr)
    if not np.any(mb):
        return None
    ys, xs = np.where(mb)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    roi = frame_bgr[y1:y2, x1:x2].copy()
    mloc = mb[y1:y2, x1:x2]
    if roi.size == 0 or not np.any(mloc):
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hh, ww = hsv.shape[:2]
    mid = max(1, hh // 2)
    top_m = mloc[:mid, :]
    bot_m = mloc[mid:, :]
    top = hsv[:mid, :][top_m]
    bot = hsv[mid:, :][bot_m]
    if top.size == 0 and bot.size == 0:
        return None
    ht = cv2.calcHist([top.reshape(-1, 1, 3)], [0], None, [16], [0, 180]) if top.size else np.zeros((16, 1))
    hb = cv2.calcHist([bot.reshape(-1, 1, 3)], [0], None, [16], [0, 180]) if bot.size else np.zeros((16, 1))
    cv2.normalize(ht, ht, 1.0, 0.0, cv2.NORM_L1)
    cv2.normalize(hb, hb, 1.0, 0.0, cv2.NORM_L1)
    feat = np.concatenate([ht.flatten(), hb.flatten()]).astype(np.float64)
    s = float(feat.sum())
    if s > 1e-9:
        feat /= s
    aspect = float(x2 - x1) / float(max(y2 - y1, 1))
    return feat, aspect


def _extract_zonal_from_bbox(
    frame_bgr: np.ndarray,
    box: Tuple[float, float, float, float],
    *,
    fw: int,
    fh: int,
) -> Optional[Tuple[np.ndarray, float]]:
    x1 = int(max(0, min(fw - 1, box[0])))
    y1 = int(max(0, min(fh - 1, box[1])))
    x2 = int(max(0, min(fw, box[2])))
    y2 = int(max(0, min(fh, box[3])))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hh, ww = hsv.shape[:2]
    mid = max(1, hh // 2)
    top = hsv[:mid, :]
    bot = hsv[mid:, :]
    ht = cv2.calcHist([top], [0], None, [16], [0, 180])
    hb = cv2.calcHist([bot], [0], None, [16], [0, 180])
    cv2.normalize(ht, ht, 1.0, 0.0, cv2.NORM_L1)
    cv2.normalize(hb, hb, 1.0, 0.0, cv2.NORM_L1)
    feat = np.concatenate([ht.flatten(), hb.flatten()]).astype(np.float64)
    s = float(feat.sum())
    if s > 1e-9:
        feat /= s
    aspect = float(ww) / float(max(hh, 1))
    return feat, aspect


def _is_isolated(
    self_box: np.ndarray,
    others: List[np.ndarray],
    *,
    isolation_mult: float,
) -> bool:
    cx = float((self_box[0] + self_box[2]) * 0.5)
    cy = float((self_box[1] + self_box[3]) * 0.5)
    w = max(1.0, float(self_box[2] - self_box[0]))
    h = max(1.0, float(self_box[3] - self_box[1]))
    scale = max(w, h)
    thr = isolation_mult * scale
    for ob in others:
        ocx = float((ob[0] + ob[2]) * 0.5)
        ocy = float((ob[1] + ob[3]) * 0.5)
        if float(np.hypot(cx - ocx, cy - ocy)) < thr:
            return False
    return True


class SwayHandshakeState:
    """Per-video tracking session state (mutated across frames)."""

    def __init__(self) -> None:
        self.prev_tracker_out: Optional[np.ndarray] = None
        self.registry: Dict[int, Tuple[np.ndarray, float]] = {}
        self.sam_frames: int = 0
        self._iso = float(os.environ.get("SWAY_REGISTRY_ISOLATION_MULT", "1.5") or "1.5")
        self._verify_stride = max(1, int(os.environ.get("SWAY_HANDSHAKE_VERIFY_STRIDE", "3") or "3"))
        self._ema = 0.12

    def update_registry_open_floor(
        self,
        frame_bgr: np.ndarray,
        dets: np.ndarray,
        *,
        used_sam: bool,
        iou_trigger: float,
    ) -> None:
        """Refresh fingerprints when boxes are separated (no SAM this frame, low IoU)."""
        if used_sam or dets.size == 0 or self.prev_tracker_out is None:
            return
        xyxy = dets[:, :4].astype(np.float32)
        if xyxy.shape[0] < 1:
            return
        mp = max_pairwise_iou(xyxy)
        if mp >= float(iou_trigger) - 1e-6:
            return
        nd = int(xyxy.shape[0])
        prev = self.prev_tracker_out.astype(np.float32)
        match = _greedy_match_curr_to_prev(xyxy, prev[:, :4], 0.22)
        fh, fw = frame_bgr.shape[:2]
        for i in range(nd):
            others = [xyxy[j] for j in range(nd) if j != i]
            if not _is_isolated(xyxy[i], others, isolation_mult=self._iso):
                continue
            pj = match[i]
            if pj < 0:
                continue
            tid = int(prev[pj, 4])
            if tid < 0:
                continue
            box = tuple(float(x) for x in xyxy[i].tolist())
            zf = _extract_zonal_from_bbox(frame_bgr, box, fw=fw, fh=fh)
            if zf is None:
                continue
            feat, asp = zf
            if tid not in self.registry:
                self.registry[tid] = (feat.copy(), asp)
            else:
                pf, pa = self.registry[tid]
                nf = (1 - self._ema) * pf + self._ema * feat
                ns = float(nf.sum())
                if ns > 1e-9:
                    nf /= ns
                na = (1 - self._ema) * pa + self._ema * asp
                self.registry[tid] = (nf.astype(np.float64), float(na))

    def verify_and_reorder_sam_dets(
        self,
        frame_bgr: np.ndarray,
        dets: np.ndarray,
        per_det_masks: List[Optional[Any]],
        *,
        involved_indices: List[int],
        mask_thresh: float,
    ) -> Tuple[np.ndarray, List[Optional[Any]]]:
        """Permute overlapping rows to align SAM masks with registry (when scipy available)."""
        if (
            linear_sum_assignment is None
            or self.prev_tracker_out is None
            or not involved_indices
        ):
            return dets, per_det_masks

        self.sam_frames += 1
        if self.sam_frames % self._verify_stride != 0:
            return dets, per_det_masks

        inv = sorted(int(x) for x in involved_indices)
        n = len(inv)
        if n < 2:
            return dets, per_det_masks

        prev = self.prev_tracker_out.astype(np.float32)
        xyxy = dets[:, :4].astype(np.float32)
        fh, fw = frame_bgr.shape[:2]

        feats: List[Optional[Tuple[np.ndarray, float]]] = []
        for idx in inv:
            m = per_det_masks[idx] if idx < len(per_det_masks) else None
            if m is None or not isinstance(m, np.ndarray) or m.size == 0:
                feats.append(None)
                continue
            x1, y1, x2, y2 = [int(round(float(x))) for x in dets[idx, :4]]
            x1 = max(0, min(fw - 1, x1))
            y1 = max(0, min(fh - 1, y1))
            x2 = max(0, min(fw, x2))
            y2 = max(0, min(fh, y2))
            if x2 <= x1 or y2 <= y1:
                feats.append(None)
                continue
            full_m = np.zeros((fh, fw), dtype=np.float32)
            mh, mw = m.shape[:2]
            ch, cw = y2 - y1, x2 - x1
            if mh != ch or mw != cw:
                m = cv2.resize(m.astype(np.float32), (cw, ch), interpolation=cv2.INTER_NEAREST)
            full_m[y1:y2, x1:x2] = np.maximum(full_m[y1:y2, x1:x2], m.astype(np.float32))
            feats.append(_extract_zonal_from_masked_region(frame_bgr, full_m, thr=mask_thresh))

        match = _greedy_match_curr_to_prev(xyxy, prev[:, :4], 0.22)
        tid_for_inv: List[int] = []
        for idx in inv:
            pj = match[idx]
            tid = int(prev[pj, 4]) if pj >= 0 else -1
            tid_for_inv.append(tid)

        if any(t < 0 for t in tid_for_inv):
            return dets, per_det_masks

        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                tid = tid_for_inv[j]
                prof = self.registry.get(tid)
                fi = feats[i]
                if fi is None:
                    cost[i, j] = 0.0
                else:
                    cost[i, j] = -_profile_score(fi[0], fi[1], prof)

        r_ind, c_ind = linear_sum_assignment(cost)
        if len(r_ind) != n:
            return dets, per_det_masks

        # Hungarian: mask slot r_ind[k] assigned to identity column c_ind[k].
        # Column j (motion slot j) should receive row content from inv[r] where r is the mask index with c_ind==j.
        slot_src = [-1] * n
        for k in range(n):
            slot_src[int(c_ind[k])] = int(r_ind[k])

        new_dets = np.asarray(dets, dtype=np.float32).copy()
        new_masks: List[Optional[Any]] = list(per_det_masks)
        for j in range(n):
            src_slot = slot_src[j]
            if src_slot < 0:
                continue
            dst_row = inv[j]
            src_row = inv[src_slot]
            new_dets[dst_row, :] = dets[src_row, :]
            new_masks[dst_row] = per_det_masks[src_row] if src_row < len(per_det_masks) else None

        return new_dets, new_masks

    def set_prev_tracker_out(self, out: Optional[np.ndarray]) -> None:
        if out is None or out.size == 0:
            self.prev_tracker_out = None
            return
        self.prev_tracker_out = np.atleast_2d(out.astype(np.float32)).copy()


def handshake_process_frame(
    state: Optional[SwayHandshakeState],
    frame_bgr: np.ndarray,
    dets: np.ndarray,
    hmeta: Dict[str, Any],
    hybrid_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Update registry on open floor; reorder dets/masks after SAM when enabled.
    ``hmeta`` is mutated in place for ``per_det_masks`` consistency.
    """
    if state is None or not phase13_handshake_enabled():
        return dets, hmeta

    used_sam = bool(hmeta.get("used_sam"))
    iou_trigger = float(hybrid_cfg.get("iou_trigger", 0.42))
    per_det_masks = hmeta.get("per_det_masks")
    if not isinstance(per_det_masks, list):
        per_det_masks = [None] * int(len(dets))
        hmeta["per_det_masks"] = per_det_masks

    state.update_registry_open_floor(
        frame_bgr,
        dets,
        used_sam=used_sam,
        iou_trigger=iou_trigger,
    )

    if used_sam and len(dets) > 0:
        xyxy = dets[:, :4].astype(np.float32)
        involved = sorted(overlap_cluster_indices(xyxy, iou_trigger))
        if len(involved) >= 2:
            dets2, masks2 = state.verify_and_reorder_sam_dets(
                frame_bgr,
                dets,
                per_det_masks,
                involved_indices=involved,
                mask_thresh=float(hybrid_cfg.get("mask_thresh", 0.5)),
            )
            hmeta["per_det_masks"] = masks2
            return dets2, hmeta

    return dets, hmeta
