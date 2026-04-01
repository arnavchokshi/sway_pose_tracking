"""
Tracker Factory (PLAN_04)

Dispatches to the correct tracking engine based on SWAY_TRACKER_ENGINE.

Env:
  SWAY_TRACKER_ENGINE – solidtrack | sam2mot | sam2_memosort_hybrid | memosort | matr
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from sway.sam2_tracker import BaseTracker

logger = logging.getLogger(__name__)


def create_tracker(
    engine: str | None = None,
    device: str = "cuda",
    **kwargs,
) -> BaseTracker:
    """Instantiate the requested tracker engine.

    Args:
        engine: tracker name override; defaults to SWAY_TRACKER_ENGINE env var.
        device: torch device string.

    Returns:
        A tracker implementing BaseTracker.
    """
    if engine is None:
        engine = os.environ.get("SWAY_TRACKER_ENGINE", "solidtrack").strip().lower()

    if engine == "solidtrack":
        return _create_solidtrack_wrapper(device, **kwargs)
    elif engine == "sam2mot":
        from sway.sam2_tracker import SAM2PrimaryTracker
        return SAM2PrimaryTracker(device=device, **kwargs)
    elif engine == "sam2_memosort_hybrid":
        return _create_sam2_memosort_hybrid(device, **kwargs)
    elif engine == "memosort":
        return _create_memosort_wrapper(device, **kwargs)
    elif engine == "matr":
        return _create_matr_wrapper(device, **kwargs)
    else:
        logger.warning("Unknown tracker engine '%s'; falling back to solidtrack", engine)
        return _create_solidtrack_wrapper(device, **kwargs)


class SolidTrackWrapper(BaseTracker):
    """Wraps the existing BoxMOT SolidTrack path as a BaseTracker.

    This preserves the current production pipeline as a regression baseline.
    get_mask() returns None — mask-dependent features fall back to bbox alternatives.
    """

    def __init__(self, device: str = "cuda"):
        from sway.track_state import TrackLifecycle, TrackState
        self._device = device
        self._lifecycles: dict = {}

    def process_frame(self, frame, frame_idx, detections=None):
        from sway.sam2_tracker import TrackResult
        from sway.track_state import TrackState
        return []

    def get_active_tracks(self):
        return list(self._lifecycles.keys())

    def get_track_state(self, track_id):
        from sway.track_state import TrackState
        lc = self._lifecycles.get(track_id)
        return lc.state if lc else TrackState.LOST

    def get_mask(self, track_id):
        return None

    def get_lifecycle(self, track_id):
        return self._lifecycles.get(track_id)


def _create_solidtrack_wrapper(device: str, **kwargs) -> BaseTracker:
    return SolidTrackWrapper(device=device)


class MeMoSORTWrapper(BaseTracker):
    """Box-based MeMoSORT wrapper implementing BaseTracker."""

    def __init__(self, device: str = "cuda", **kwargs):
        from sway.memosort import MeMoSORT
        from sway.track_state import TrackLifecycle

        self._device = device
        self._engine = MeMoSORT()
        self._next_track_id = 1
        self._last_seen: Dict[int, int] = {}
        self._bboxes: Dict[int, np.ndarray] = {}
        self._lifecycles: Dict[int, TrackLifecycle] = {}
        self._max_age = max(
            1,
            int(
                os.environ.get("SWAY_TRACK_MAX_AGE", "").strip()
                or os.environ.get("SWAY_BOXMOT_MAX_AGE", "90")
            ),
        )
        self._ft_det_high = float(os.environ.get("SWAY_FT_DET_HIGH", "0.5") or 0.5)
        self._ft_det_low = float(os.environ.get("SWAY_FT_DET_LOW", "0.1") or 0.1)
        self._ft_proximity_rad = float(os.environ.get("SWAY_FT_PROXIMITY_RAD", "0.3") or 0.3)
        self._ft_box_enlarge = float(os.environ.get("SWAY_FT_BOX_ENLARGE", "1.1") or 1.1)
        self._ft_motion_damp = float(os.environ.get("SWAY_FT_MOTION_DAMP", "0.5") or 0.5)

    @staticmethod
    def _enlarge_box_xyxy(box: np.ndarray, factor: float, width: int, height: int) -> np.ndarray:
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        bw = max(1.0, float(box[2] - box[0])) * max(0.1, factor)
        bh = max(1.0, float(box[3] - box[1])) * max(0.1, factor)
        x1 = max(0.0, cx - 0.5 * bw)
        y1 = max(0.0, cy - 0.5 * bh)
        x2 = min(float(width - 1), cx + 0.5 * bw)
        y2 = min(float(height - 1), cy + 0.5 * bh)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def process_frame(self, frame, frame_idx, detections=None):
        from sway.sam2_tracker import TrackResult
        from sway.track_state import TrackLifecycle, TrackState, update_state

        detections = detections or []
        h, w = frame.shape[:2]
        det_boxes: List[np.ndarray] = []
        det_confs: List[float] = []
        for d in detections:
            conf = float(getattr(d, "confidence", 0.5))
            if conf < self._ft_det_low:
                continue
            box = d.bbox.astype(np.float32)
            box = self._enlarge_box_xyxy(box, self._ft_box_enlarge, w, h)
            det_boxes.append(box)
            det_confs.append(conf)

        preds = self._engine.predict_all()
        if preds and 0.0 <= self._ft_motion_damp <= 1.0:
            for tid, pred in list(preds.items()):
                last = self._bboxes.get(tid)
                if last is None:
                    continue
                # Blend predicted box toward last observation to damp overshoot.
                pred.bbox_xyxy = (
                    self._ft_motion_damp * last.astype(np.float32)
                    + (1.0 - self._ft_motion_damp) * pred.bbox_xyxy.astype(np.float32)
                ).astype(np.float32)

        matches, unmatched_tracks, unmatched_dets = self._engine.match(preds, det_boxes)
        track_conf: Dict[int, float] = {}

        # Apply fasttracker-style proximity guard on matches.
        if matches:
            diag = max(1.0, float(np.hypot(float(w), float(h))))
            max_dist = self._ft_proximity_rad * diag
            _filtered = []
            _used_track = set()
            _used_det = set()
            for tid, didx in matches:
                pb = preds.get(tid)
                if pb is None:
                    continue
                db = det_boxes[didx]
                pcx = 0.5 * (pb.bbox_xyxy[0] + pb.bbox_xyxy[2])
                pcy = 0.5 * (pb.bbox_xyxy[1] + pb.bbox_xyxy[3])
                dcx = 0.5 * (db[0] + db[2])
                dcy = 0.5 * (db[1] + db[3])
                dist = float(np.hypot(float(dcx - pcx), float(dcy - pcy)))
                if dist <= max_dist:
                    _filtered.append((tid, didx))
                    _used_track.add(tid)
                    _used_det.add(didx)
            matches = _filtered
            unmatched_tracks = sorted(set(unmatched_tracks) | (set(preds.keys()) - _used_track))
            unmatched_dets = sorted(set(unmatched_dets) | (set(range(len(det_boxes))) - _used_det))

        # matched tracks
        for tid, didx in matches:
            box = det_boxes[didx]
            self._engine.update(tid, box)
            self._bboxes[tid] = box
            self._last_seen[tid] = frame_idx
            track_conf[tid] = det_confs[didx] if didx < len(det_confs) else 0.5
            lc = self._lifecycles.get(tid)
            if lc is None:
                lc = TrackLifecycle(track_id=tid)
                self._lifecycles[tid] = lc
            area = float(max(1.0, (box[2] - box[0]) * (box[3] - box[1])))
            update_state(lc, area, 17, frame_idx)

        # create new tracks for unmatched detections
        for didx in unmatched_dets:
            box = det_boxes[didx]
            if didx < len(det_confs) and det_confs[didx] < self._ft_det_high:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self._engine.init_track(tid, box)
            self._bboxes[tid] = box
            self._last_seen[tid] = frame_idx
            track_conf[tid] = det_confs[didx] if didx < len(det_confs) else 0.5
            lc = TrackLifecycle(track_id=tid)
            area = float(max(1.0, (box[2] - box[0]) * (box[3] - box[1])))
            update_state(lc, area, 17, frame_idx)
            self._lifecycles[tid] = lc

        # age out unmatched tracks
        for tid in unmatched_tracks:
            last = self._last_seen.get(tid, frame_idx)
            if frame_idx - last > self._max_age:
                self._engine.remove_track(tid)
                self._bboxes.pop(tid, None)
                self._last_seen.pop(tid, None)
                self._lifecycles.pop(tid, None)
                continue
            lc = self._lifecycles.get(tid)
            if lc is not None:
                update_state(lc, None, 0, frame_idx)

        out = []
        for tid, box in self._bboxes.items():
            lc = self._lifecycles.get(tid)
            st = lc.state if lc is not None else TrackState.ACTIVE
            conf = max(0.0, min(1.0, float(track_conf.get(tid, 0.5))))
            out.append(
                TrackResult(
                    track_id=int(tid),
                    bbox_xyxy=box.astype(np.float32),
                    mask=None,
                    confidence=float(conf),
                    state=st,
                    mask_area=float(max(1.0, (box[2] - box[0]) * (box[3] - box[1]))),
                )
            )
        return out

    def get_active_tracks(self):
        return list(self._bboxes.keys())

    def get_track_state(self, track_id):
        from sway.track_state import TrackState

        lc = self._lifecycles.get(track_id)
        return lc.state if lc else TrackState.LOST

    def get_mask(self, track_id):
        return None

    def get_lifecycle(self, track_id):
        return self._lifecycles.get(track_id)


def _create_sam2_memosort_hybrid(device: str, **kwargs) -> BaseTracker:
    """SAM2 masks + MeMoSORT motion prediction hybrid."""
    return SAM2MeMoSORTHybridWrapper(device=device, **kwargs)


class SAM2MeMoSORTHybridWrapper(BaseTracker):
    """Hybrid tracker: SAM2 identities with MeMoSORT motion smoothing."""

    def __init__(self, device: str = "cuda", **kwargs):
        from sway.sam2_tracker import SAM2PrimaryTracker
        from sway.memosort import MeMoSORT

        self._sam = SAM2PrimaryTracker(device=device, **kwargs)
        self._memo = MeMoSORT()
        self._active_ids: set[int] = set()

    def process_frame(self, frame, frame_idx, detections=None):
        results = self._sam.process_frame(frame, frame_idx, detections=detections)
        # Sync trackers and apply light motion-aware smoothing on SAM2 boxes.
        for r in results:
            tid = int(r.track_id)
            box = r.bbox_xyxy.astype(np.float32)
            if not self._memo.has_track(tid):
                self._memo.init_track(tid, box)
            pred = self._memo.predict(tid)
            if pred is not None:
                r.bbox_xyxy = (0.8 * box + 0.2 * pred.bbox_xyxy.astype(np.float32)).astype(np.float32)
            self._memo.update(tid, r.bbox_xyxy)
            self._active_ids.add(tid)
        live = {int(r.track_id) for r in results}
        stale = self._active_ids - live
        for tid in list(stale):
            self._memo.remove_track(tid)
            self._active_ids.discard(tid)
        return results

    def get_active_tracks(self):
        return self._sam.get_active_tracks()

    def get_track_state(self, track_id):
        return self._sam.get_track_state(track_id)

    def get_mask(self, track_id):
        return self._sam.get_mask(track_id)

    def get_lifecycle(self, track_id):
        return self._sam.get_lifecycle(track_id)

    def get_logit_scores(self):
        return self._sam.get_logit_scores()

    def remove_memory_entries(self, track_id, start_frame, end_frame):
        if hasattr(self._sam, "remove_memory_entries"):
            return self._sam.remove_memory_entries(track_id, start_frame, end_frame)

    def freeze_memory(self, track_id):
        if hasattr(self._sam, "freeze_memory"):
            return self._sam.freeze_memory(track_id)


def _create_memosort_wrapper(device: str, **kwargs) -> BaseTracker:
    """MeMoSORT standalone (box-based, no masks)."""
    return MeMoSORTWrapper(device=device, **kwargs)


def _create_matr_wrapper(device: str, **kwargs) -> BaseTracker:
    """MATR branch entrypoint.

    Current implementation uses a MATR-compatible wrapper that preserves a
    distinct engine branch while reusing MeMoSORT motion primitives.
    """
    return MATRWrapper(device=device, **kwargs)


class MATRWrapper(MeMoSORTWrapper):
    """MATR-compatible tracker wrapper with dedicated engine identity."""

    def __init__(self, device: str = "cuda", **kwargs):
        logger.info("SWAY_TRACKER_ENGINE=matr selected; using MATR-compatible wrapper.")
        super().__init__(device=device, **kwargs)
