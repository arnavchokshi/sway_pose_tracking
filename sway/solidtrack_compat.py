"""
SolidTrack cost-matrix adapter and EMA embedding smoother.

SolidTrack (MDPI Electronics, March 2025) replaces BotSORT's weighted
IoU+embedding cost with a gated minimum:

    gated_emb = 0.5 * d_emb   if d_emb < θ_emb AND d_iou < θ_iou
                1.0            otherwise
    cost      = min(d_iou, gated_emb)

This module:

  1. Provides ``solidtrack_cost_matrix`` — pure numpy implementation.

  2. Provides ``patch_solidtrack_cost`` which ACTUALLY patches BotSort:
     - Wraps ``tracker.update`` with a context-flag.
     - Monkey-patches ``boxmot.utils.matching.iou_distance`` (or the
       matching module the BoxMOT version uses) so that during a
       SolidTrack update call the function returns our gated cost instead
       of plain IoU distance.
     - Uses a thread-local flag so concurrent trackers that are NOT
       SolidTrack remain unaffected.

  3. Provides ``apply_ema_to_tracker`` which reads ``SWAY_ST_EMA_ALPHA``
     and, after each tracker.update, applies exponential moving average
     to the per-track feature embeddings stored by BotSort.  The wrapped
     ``update`` is returned and should replace ``tracker.update``.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 1. Cost matrix (pure numpy, used by the matching patch)
# ---------------------------------------------------------------------------

def solidtrack_cost_matrix(
    iou_dist: np.ndarray,
    emb_dist: np.ndarray,
    theta_iou: float = 0.5,
    theta_emb: float = 0.25,
) -> np.ndarray:
    """Compute SolidTrack-style gated minimum cost matrix.

    gated_emb = 0.5*d_emb  if d_emb < theta_emb AND d_iou < theta_iou
                1.0         otherwise
    cost       = min(d_iou, gated_emb)
    """
    gated_emb = np.where(
        (emb_dist < theta_emb) & (iou_dist < theta_iou),
        0.5 * emb_dist,
        1.0,
    )
    return np.minimum(iou_dist, gated_emb)


# ---------------------------------------------------------------------------
# 2. BotSort matching patch (context-flag via thread-local)
# ---------------------------------------------------------------------------

_st_tls = threading.local()  # _st_tls.active = (theta_iou, theta_emb) or None

_MATCHING_MODULES = [
    "boxmot.utils.matching",
    "boxmot.trackers.botsort.botsort",   # some versions inline matching
]


def _find_matching_module() -> Optional[Any]:
    import importlib
    for mod_name in _MATCHING_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "iou_distance"):
                return mod
        except ImportError:
            continue
    return None


def _install_matching_patch(theta_iou: float, theta_emb: float) -> bool:
    """Monkey-patch the BoxMOT matching module's ``iou_distance`` once.

    The patched version checks ``_st_tls.active``; if set it delegates to
    ``solidtrack_cost_matrix`` (using the stored thetas); otherwise it
    falls through to the original.  Returns True on success.
    """
    mod = _find_matching_module()
    if mod is None:
        return False

    if getattr(mod, "_sway_st_patched", False):
        return True  # already installed

    _orig_iou_distance = mod.iou_distance

    def _st_iou_distance(atracks: Any, btracks: Any) -> np.ndarray:
        params = getattr(_st_tls, "active", None)
        if params is None:
            return _orig_iou_distance(atracks, btracks)

        t_iou, t_emb = params
        d_iou = _orig_iou_distance(atracks, btracks)

        # Try to get embedding distance; fall back gracefully
        try:
            d_emb = mod.embedding_distance(atracks, btracks)
        except Exception:
            return d_iou  # no embeddings → plain IoU

        if d_iou.shape != d_emb.shape:
            return d_iou  # shape mismatch → plain IoU

        return solidtrack_cost_matrix(d_iou, d_emb, t_iou, t_emb)

    mod.iou_distance = _st_iou_distance
    mod._sway_st_patched = True
    return True


def patch_solidtrack_cost(
    tracker: Any,
    theta_iou: float = 0.5,
    theta_emb: float = 0.25,
) -> None:
    """Monkey-patch a BotSort INSTANCE to use the SolidTrack cost matrix.

    Does two things:
      1. Installs the matching-module patch (once, class-level).
      2. Wraps ``tracker.update`` (instance-level) to set / clear the
         thread-local flag so the patched ``iou_distance`` knows when to
         apply SolidTrack logic.
    """
    # Store parameters on the tracker (introspection / logging)
    tracker._st_theta_iou = theta_iou
    tracker._st_theta_emb = theta_emb
    tracker._st_patched = True

    # Install the matching patch
    patch_ok = _install_matching_patch(theta_iou, theta_emb)
    if not patch_ok:
        # Could not locate matching module; tracker keeps default behaviour
        return

    _orig_update = tracker.update

    def _st_update(dets: Any, img: Any, *args: Any, **kwargs: Any) -> Any:
        _st_tls.active = (theta_iou, theta_emb)
        try:
            return _orig_update(dets, img, *args, **kwargs)
        finally:
            _st_tls.active = None

    tracker.update = _st_update


# ---------------------------------------------------------------------------
# 3. EMA embedding smoother
# ---------------------------------------------------------------------------

def apply_ema_to_tracker(tracker: Any, alpha: float) -> Any:
    """Wrap ``tracker.update`` to apply EMA smoothing to per-track features.

    After each call to the original update, iterates over all active tracks
    and blends their stored feature vector with the EMA-smoothed version:

        smoothed[tid] = alpha * smoothed[tid] + (1 - alpha) * current_feat

    The smoothed vector is written back to the track so BotSort uses it
    for the next association.

    Args:
        tracker: BotSort instance (already created).
        alpha: EMA decay (0 < alpha < 1). Higher = more historical weight.

    Returns:
        The original tracker (update method replaced in-place).
    """
    _ema_store: Dict[int, np.ndarray] = {}
    _orig_update = tracker.update

    def _ema_update(dets: Any, img: Any, *args: Any, **kwargs: Any) -> Any:
        out = _orig_update(dets, img, *args, **kwargs)

        # Walk active tracks and apply EMA to their embedding
        active_tracks = _get_active_tracks(tracker)
        for track in active_tracks:
            tid = _get_track_id(track)
            feat = _get_track_feat(track)
            if feat is None or tid is None:
                continue
            feat = np.asarray(feat, dtype=np.float32)
            if tid in _ema_store:
                smoothed = alpha * _ema_store[tid] + (1.0 - alpha) * feat
            else:
                smoothed = feat.copy()
            _ema_store[tid] = smoothed
            _set_track_feat(track, smoothed)

        # Prune stale track IDs from the store
        active_ids = {_get_track_id(t) for t in active_tracks if _get_track_id(t) is not None}
        stale = [k for k in list(_ema_store) if k not in active_ids]
        for k in stale:
            del _ema_store[k]

        return out

    tracker.update = _ema_update
    return tracker


# ---------------------------------------------------------------------------
# Helpers: access per-track embedding regardless of BoxMOT version
# ---------------------------------------------------------------------------

def _get_active_tracks(tracker: Any):
    """Return iterable of active track objects from a BotSort instance."""
    # BoxMOT >= 10: per_class_active_tracks is a dict class→list
    pcat = getattr(tracker, "per_class_active_tracks", None)
    if pcat is not None and isinstance(pcat, dict):
        tracks = []
        for v in pcat.values():
            tracks.extend(v)
        return tracks
    # Fallback attributes
    for attr in ("active_tracks", "tracked_stracks", "trackers"):
        v = getattr(tracker, attr, None)
        if v is not None:
            return v
    return []


def _get_track_id(track: Any) -> Optional[int]:
    for attr in ("id", "track_id"):
        v = getattr(track, attr, None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return None


def _get_track_feat(track: Any) -> Optional[np.ndarray]:
    for attr in ("smooth_feat", "features", "curr_feat", "feat"):
        v = getattr(track, attr, None)
        if v is not None:
            try:
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 1 and arr.size > 0:
                    return arr
            except (TypeError, ValueError):
                pass
    return None


def _set_track_feat(track: Any, feat: np.ndarray) -> None:
    for attr in ("smooth_feat", "features", "curr_feat", "feat"):
        if hasattr(track, attr):
            try:
                setattr(track, attr, feat)
                return
            except AttributeError:
                pass
