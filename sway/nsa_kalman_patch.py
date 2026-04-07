"""
NSA (Noise-Scale-Adaptive) Kalman filter patch for DeepOCSORT.

Scales measurement noise R by (1 - detection_confidence)^2 so that
high-confidence detections receive a larger Kalman gain (the filter
trusts the measurement more). This is the same principle used by
StrongSORT's built-in NSA-KF, ported to DeepOCSORT's XYSR state space.

Two-part implementation:
  1. KalmanFilterXYSR.update is patched to read the current-frame
     confidence from a thread-local variable and temporarily override
     self.R with the scaled measurement noise before calling the
     original update, restoring R afterwards.

  2. The DeepOcSort tracker's ``update`` method is wrapped (per-instance)
     by ``wrap_deepocsort_for_nsa`` (called from tracker.py) so that the
     mean detection confidence for each frame is injected into the
     thread-local before the tracker processes it.

Usage (tracker.py already does this):
    from sway.nsa_kalman_patch import apply_nsa_kf_patch, wrap_deepocsort_for_nsa
    apply_nsa_kf_patch()  # once — patches the class
    tracker = DeepOcSort(...)
    tracker.update = wrap_deepocsort_for_nsa(tracker.update)  # per instance
"""
from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np

# Thread-local storage: _nsa_tls.confidence is set to a float by the
# outer update wrapper each frame, read by the inner KF.update patch.
_nsa_tls = threading.local()


def set_nsa_frame_confidence(conf: Optional[float]) -> None:
    """Set the per-frame confidence to be used by the patched KF.update."""
    _nsa_tls.confidence = conf


def get_nsa_frame_confidence() -> Optional[float]:
    return getattr(_nsa_tls, "confidence", None)


def apply_nsa_kf_patch() -> bool:
    """Monkey-patch KalmanFilterXYSR.update to scale R by (1 - confidence)^2.

    Returns True if the patch was applied (or was already active).
    The patch reads confidence from the thread-local set by
    ``set_nsa_frame_confidence`` each frame; if no confidence is set
    the original update runs unchanged.
    """
    try:
        from boxmot.motion.kalman_filters.aabb import xysr_kf
    except ImportError:
        return False

    cls = xysr_kf.KalmanFilterXYSR
    if getattr(cls, "_sway_nsa_patch", False):
        return True

    _orig_update = cls.update

    def _update_nsa(self: Any, mean: Any, covariance: Any, measurement: Any, **kwargs: Any) -> Any:
        """Wrapper: temporarily scale R by (1 - confidence)^2, restore after."""
        confidence = get_nsa_frame_confidence()
        if confidence is not None and hasattr(self, "_std_weight_position"):
            h = float(mean[3]) if len(mean) > 3 else 1.0
            # Scale factor: high confidence → small noise (filter trusts measurement)
            scale = max(1.0 - float(confidence), 0.05)
            std_pos = [
                self._std_weight_position * h * scale,
                self._std_weight_position * h * scale,
                1e-2 * scale,
                self._std_weight_position * h * scale,
            ]
            R_nsa = np.diag(np.square(np.array(std_pos, dtype=np.float64)))
            _old_R = getattr(self, "R", None)
            # Some BoxMOT versions store R as a property; try direct attribute set
            try:
                self.R = R_nsa
                return _orig_update(self, mean, covariance, measurement, **kwargs)
            finally:
                if _old_R is not None:
                    self.R = _old_R
                elif hasattr(self, "R"):
                    try:
                        del self.R
                    except AttributeError:
                        pass
        return _orig_update(self, mean, covariance, measurement, **kwargs)

    cls.update = _update_nsa
    cls._sway_nsa_patch = True
    return True


def wrap_deepocsort_for_nsa(tracker_update_fn: Any) -> Any:
    """Wrap a DeepOcSort instance's ``update`` to inject mean-det confidence.

    The wrapped function computes the mean confidence of the current batch
    of detections and stores it in the thread-local before calling the
    original tracker update so every KF.update call in that frame gets
    the right R scaling.

    Args:
        tracker_update_fn: bound ``tracker.update`` method to wrap.

    Returns:
        A replacement callable with the same signature.
    """

    def _wrapped(dets: Any, img: Any, *args: Any, **kwargs: Any) -> Any:
        if dets is not None and hasattr(dets, "__len__") and len(dets) > 0:
            arr = np.asarray(dets)
            if arr.ndim == 2 and arr.shape[1] > 4:
                mean_conf = float(np.mean(arr[:, 4]))
            else:
                mean_conf = 0.5
        else:
            mean_conf = 0.5
        set_nsa_frame_confidence(mean_conf)
        try:
            return tracker_update_fn(dets, img, *args, **kwargs)
        finally:
            set_nsa_frame_confidence(None)

    return _wrapped
