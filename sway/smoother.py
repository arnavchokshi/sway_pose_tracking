"""
Temporal Smoothing Module — 1 Euro Filter (V3.0)

Applies the 1 Euro Filter to (x, y) coordinates of each joint over time per track ID.
V3.0: Suspended when keypoint confidence < 0.3 to prevent smoothing hallucinated geometry.

``SMOOTHER_MIN_CUTOFF`` in ``params`` is master-locked to **1.0** (§14.0.1) unless
``SWAY_UNLOCK_SMOOTH_TUNING=1``.
"""

import math
from typing import Any, Dict, Tuple


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    """Compute alpha for exponential smoothing. τ = 1/(2π·fc), α = τ/(τ+T)."""
    if t_e <= 0:
        return 1.0
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(alpha: float, x: float, x_prev: float) -> float:
    """Apply exponential smoothing: α*x + (1-α)*x_prev."""
    return alpha * x + (1 - alpha) * x_prev


class OneEuroFilter:
    """
    One Euro Filter for smoothing a single scalar signal over time.

    Adaptive low-pass filter: reduces jitter at low speeds and lag at high speeds.
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        """
        Args:
            t0: Initial timestamp (seconds).
            x0: Initial value.
            dx0: Initial derivative estimate.
            min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing.
            beta: Speed coefficient. Higher = less lag for fast motion.
            d_cutoff: Cutoff for derivative filter (Hz).
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t: float, x: float) -> float:
        """Filter the value x at time t; returns smoothed value."""
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        # Filter the derivative
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        # Adaptive cutoff: higher when motion is fast
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    @property
    def last_output(self) -> float:
        """Return last filtered value without updating (V3.0: for low-conf skip)."""
        return self.x_prev


# V3.0: Keypoint confidence below this — suspend smoothing (prevent hallucination)
SMOOTH_CONF_THRESHOLD = 0.3


class PoseSmoother:
    """
    Applies 1 Euro Filter to all keypoint coordinates per track ID.

    Maintains one OneEuroFilter per (track_id, joint_idx, axis).
    Missing tracks in a frame are skipped (filter state persists).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
        conf_threshold: float = SMOOTH_CONF_THRESHOLD,
    ):
        """
        Args:
            min_cutoff: Minimum cutoff (Hz). Higher = less smoothing, more responsive.
            beta: Speed coefficient. Good for dance: ~0.7 reduces lag on fast moves.
            d_cutoff: Derivative filter cutoff (Hz).
            conf_threshold: V3.0 — skip smoothing when keypoint conf < this.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.conf_threshold = conf_threshold
        self._filters: Dict[Tuple[int, int, str], OneEuroFilter] = {}

    def _get_or_create_filter(
        self,
        track_id: int,
        joint_idx: int,
        axis: str,
        t: float,
        x: float,
    ) -> OneEuroFilter:
        """Get existing filter or create new one with initial value."""
        key = (track_id, joint_idx, axis)
        if key not in self._filters:
            self._filters[key] = OneEuroFilter(
                t0=t,
                x0=x,
                dx0=0.0,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff,
            )
        return self._filters[key]

    def smooth_frame(
        self,
        track_poses: Dict[int, Dict],
        frame_time: float,
    ) -> Dict[int, Dict]:
        """
        Smooth keypoints for one frame.

        Args:
            track_poses: {track_id: {"keypoints": (17,3), "scores": (17,)}}
            frame_time: Timestamp in seconds.

        Returns:
            Same structure with smoothed (x, y); scores unchanged.
        """
        result = {}
        for track_id, data in track_poses.items():
            keypoints = data["keypoints"]
            scores = data["scores"]
            smoothed_kpts = keypoints.copy()

            for j in range(min(17, keypoints.shape[0])):
                x, y = keypoints[j, 0], keypoints[j, 1]
                conf = float(keypoints[j, 2]) if keypoints.shape[1] > 2 else 1.0

                if conf < self.conf_threshold:
                    # V3.0: Suspend smoothing — use previous filtered value to avoid hallucination
                    key = (track_id, j, "x")
                    if key in self._filters:
                        smoothed_kpts[j, 0] = self._filters[key].last_output
                    else:
                        smoothed_kpts[j, 0] = x
                    key = (track_id, j, "y")
                    if key in self._filters:
                        smoothed_kpts[j, 1] = self._filters[key].last_output
                    else:
                        smoothed_kpts[j, 1] = y
                else:
                    fx = self._get_or_create_filter(track_id, j, "x", frame_time, x)
                    smoothed_kpts[j, 0] = fx(frame_time, x)
                    fy = self._get_or_create_filter(track_id, j, "y", frame_time, y)
                    smoothed_kpts[j, 1] = fy(frame_time, y)
                if keypoints.shape[1] > 2:
                    smoothed_kpts[j, 2] = keypoints[j, 2]

            entry: Dict[str, Any] = {"keypoints": smoothed_kpts, "scores": scores}
            for extra in ("keypoints_3d", "lift_xyz"):
                if extra in data:
                    entry[extra] = data[extra]
            result[track_id] = entry
        return result
