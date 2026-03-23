"""
BoxMOT KalmanFilterXYSR.unfreeze can raise IndexError when fewer than two
non-None observations exist in history (DeepOcSort + short history / CPU).
Clear saved state and continue — standard KF update then applies.
"""

from __future__ import annotations


def apply_boxmot_kf_unfreeze_guard() -> bool:
    """Return True if a patch was applied."""
    try:
        from boxmot.motion.kalman_filters.aabb import xysr_kf
    except ImportError:
        return False

    cls = xysr_kf.KalmanFilterXYSR
    if getattr(cls, "_sway_unfreeze_guard", False):
        return False

    _orig = cls.unfreeze

    def unfreeze_safe(self):  # type: ignore[no-untyped-def]
        try:
            return _orig(self)
        except IndexError:
            self.attr_saved = None

    cls.unfreeze = unfreeze_safe  # type: ignore[method-assign]
    cls._sway_unfreeze_guard = True
    return True
