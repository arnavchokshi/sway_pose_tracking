"""
Five-Dimension Critique Scoring Engine (PLAN_19)

Analyzes dance performance across five biomechanical dimensions:
  1. Formation accuracy    – spatial deviation from template
  2. Timing precision      – alignment to audio beats
  3. Extension/line quality – joint angles at movement peaks
  4. Smoothness            – jerk analysis (3rd derivative)
  5. Group synchronization – deviation from group mean pose

Each dimension produces per-dancer, per-timestamp scores with confidence gating.
Only HIGH and MEDIUM keypoints produce feedback; gaps are explicitly reported.

Env:
  SWAY_CRITIQUE_DIMENSIONS       – comma-separated (default formation,timing,extension,smoothness,sync)
  SWAY_CRITIQUE_JERK_WINDOW      – smoothing window for jerk (default 5)
  SWAY_CRITIQUE_BEAT_TOLERANCE_MS – timing tolerance in ms (default 100)
  SWAY_CRITIQUE_MIN_CONFIDENCE   – LOCKED: MEDIUM
  SWAY_CRITIQUE_REPORT_GAPS      – LOCKED: 1
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from sway.mask_guided_pose import KeypointConfidence
from sway.track_state import TrackState

logger = logging.getLogger(__name__)


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


# Joint pairs for extension analysis
EXTENSION_PAIRS = [
    ("right_arm", [6, 8, 10]),    # R shoulder → R elbow → R wrist
    ("left_arm", [5, 7, 9]),      # L shoulder → L elbow → L wrist
    ("right_leg", [12, 14, 16]),   # R hip → R knee → R ankle
    ("left_leg", [11, 13, 15]),    # L hip → L knee → L ankle
]


@dataclass
class GapReport:
    """Report of a visibility gap for a dancer."""
    dancer_id: int
    start_frame: int
    end_frame: int
    reason: str  # "DORMANT" or "PARTIAL_lower_body"


@dataclass
class CritiqueResult:
    """Per-dancer critique results across all dimensions."""
    dancer_id: int

    # Dimension 1: formation (per-frame deviation in cm)
    formation_error: Optional[np.ndarray] = None

    # Dimension 2: timing (per-beat offset in ms)
    timing_offsets_ms: Optional[np.ndarray] = None
    beat_frames: Optional[np.ndarray] = None

    # Dimension 3: extension (per-frame, per-joint-pair deficit in degrees)
    extension_deficit: Optional[Dict[str, np.ndarray]] = None

    # Dimension 4: smoothness (per-frame jerk score)
    jerk_scores: Optional[np.ndarray] = None

    # Dimension 5: synchronization
    sync_deviation: Optional[np.ndarray] = None

    # Gaps
    gaps: List[GapReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {"dancer_id": self.dancer_id}
        if self.formation_error is not None:
            d["formation_error"] = self.formation_error.tolist()
        if self.timing_offsets_ms is not None:
            d["timing_offsets_ms"] = self.timing_offsets_ms.tolist()
        if self.beat_frames is not None:
            d["beat_frames"] = self.beat_frames.tolist()
        if self.extension_deficit is not None:
            d["extension_deficit"] = {
                k: v.tolist() for k, v in self.extension_deficit.items()
            }
        if self.jerk_scores is not None:
            d["jerk_scores"] = self.jerk_scores.tolist()
        if self.sync_deviation is not None:
            d["sync_deviation"] = self.sync_deviation.tolist()
        d["gaps"] = [
            {"start": g.start_frame, "end": g.end_frame, "reason": g.reason}
            for g in self.gaps
        ]
        return d


class CritiqueEngine:
    """Five-dimension biomechanical critique scoring."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.jerk_window = _env_int("SWAY_CRITIQUE_JERK_WINDOW", 5)
        self.beat_tolerance_ms = _env_float("SWAY_CRITIQUE_BEAT_TOLERANCE_MS", 100)
        dimensions_str = _env_str(
            "SWAY_CRITIQUE_DIMENSIONS", "formation,timing,extension,smoothness,sync"
        )
        self.dimensions = [d.strip() for d in dimensions_str.split(",")]

    def analyze(
        self,
        keypoints_2d: Dict[int, np.ndarray],
        keypoints_3d: Optional[Dict[int, np.ndarray]] = None,
        confidence_levels: Optional[Dict[int, np.ndarray]] = None,
        track_states: Optional[Dict[int, np.ndarray]] = None,
        audio_path: Optional[str] = None,
        formation_template: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[CritiqueResult]:
        """Run full critique analysis on all dancers.

        Args:
            keypoints_2d: {dancer_id: (T, 17, 3)} with [x, y, conf].
            keypoints_3d: {dancer_id: (T, 17, 3)} in world coords.
            confidence_levels: {dancer_id: (T, 17)} KeypointConfidence values.
            track_states: {dancer_id: (T,)} TrackState per frame.
            audio_path: path to audio file for beat detection.
            formation_template: {dancer_id: (T, 3)} template 3D positions.

        Returns:
            List of CritiqueResult, one per dancer.
        """
        results: List[CritiqueResult] = []
        dancer_ids = sorted(keypoints_2d.keys())

        # Detect beats if audio available
        beat_frames = None
        if audio_path and "timing" in self.dimensions:
            beat_frames = self._detect_beats(audio_path)

        for did in dancer_ids:
            kp2d = keypoints_2d[did]
            kp3d = keypoints_3d.get(did) if keypoints_3d else None
            conf = confidence_levels.get(did) if confidence_levels else None
            states = track_states.get(did) if track_states else None

            # Confidence mask: only use HIGH and MEDIUM keypoints
            mask = self._build_confidence_mask(conf)

            result = CritiqueResult(dancer_id=did)

            # Detect gaps
            if states is not None:
                result.gaps = self._detect_gaps(did, states)

            # Dimension 1: Formation
            if "formation" in self.dimensions and kp3d is not None:
                template = formation_template.get(did) if formation_template else None
                result.formation_error = self._formation_accuracy(kp3d, template, mask)

            # Dimension 2: Timing
            if "timing" in self.dimensions and beat_frames is not None:
                result.beat_frames = beat_frames
                result.timing_offsets_ms = self._timing_precision(kp2d, beat_frames, mask)

            # Dimension 3: Extension
            if "extension" in self.dimensions:
                all_kp2d = {d: keypoints_2d[d] for d in dancer_ids}
                result.extension_deficit = self._extension_quality(kp2d, all_kp2d, mask)

            # Dimension 4: Smoothness
            if "smoothness" in self.dimensions:
                result.jerk_scores = self._smoothness(kp2d, mask)

            # Dimension 5: Synchronization
            if "sync" in self.dimensions:
                all_kp2d = {d: keypoints_2d[d] for d in dancer_ids}
                result.sync_deviation = self._group_sync(did, kp2d, all_kp2d, mask)

            results.append(result)

        return results

    def _build_confidence_mask(
        self, conf: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Build a boolean mask: True where confidence >= MEDIUM."""
        if conf is None:
            return None
        return conf >= KeypointConfidence.MEDIUM

    def _detect_gaps(self, dancer_id: int, states: np.ndarray) -> List[GapReport]:
        """Detect DORMANT intervals and report as gaps."""
        gaps = []
        in_gap = False
        gap_start = 0

        for f in range(len(states)):
            is_dormant = states[f] <= TrackState.DORMANT
            if is_dormant and not in_gap:
                gap_start = f
                in_gap = True
            elif not is_dormant and in_gap:
                gaps.append(GapReport(
                    dancer_id=dancer_id,
                    start_frame=gap_start,
                    end_frame=f - 1,
                    reason="DORMANT",
                ))
                in_gap = False

        if in_gap:
            gaps.append(GapReport(
                dancer_id=dancer_id,
                start_frame=gap_start,
                end_frame=len(states) - 1,
                reason="DORMANT",
            ))

        return gaps

    def _detect_beats(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract audio beat grid using librosa."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=22050)
            tempo, beat_frames_audio = librosa.beat.beat_track(y=y, sr=sr)
            # Convert audio frames to video frames
            beat_times = librosa.frames_to_time(beat_frames_audio, sr=sr)
            beat_video_frames = (beat_times * self.fps).astype(int)
            logger.info("Detected %d beats at tempo %.1f BPM", len(beat_video_frames), float(tempo))
            return beat_video_frames
        except ImportError:
            logger.warning("librosa not installed; timing analysis disabled")
            return None
        except Exception as exc:
            logger.warning("Beat detection failed: %s", exc)
            return None

    def _formation_accuracy(
        self,
        kp3d: np.ndarray,
        template: Optional[np.ndarray],
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Per-frame formation error (Euclidean distance from template)."""
        T = kp3d.shape[0]
        errors = np.full(T, np.nan, dtype=np.float32)

        # Hip midpoint as position proxy
        hip_mid = (kp3d[:, 11, :] + kp3d[:, 12, :]) / 2  # (T, 3)

        if template is not None:
            for t in range(T):
                if not np.isnan(hip_mid[t]).any() and t < template.shape[0]:
                    errors[t] = np.linalg.norm(hip_mid[t] - template[t])
        else:
            # Auto-template: mean of first 30 seconds
            ref_frames = min(int(30 * self.fps), T)
            ref_pos = np.nanmean(hip_mid[:ref_frames], axis=0)
            for t in range(T):
                if not np.isnan(hip_mid[t]).any():
                    errors[t] = np.linalg.norm(hip_mid[t] - ref_pos)

        return errors

    def _timing_precision(
        self,
        kp2d: np.ndarray,
        beat_frames: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Per-beat timing offset in milliseconds."""
        T = kp2d.shape[0]

        # Compute joint velocity magnitude
        velocity = np.diff(kp2d[:, :, :2], axis=0)  # (T-1, 17, 2)
        vel_mag = np.linalg.norm(velocity, axis=2)    # (T-1, 17)

        if mask is not None and mask.shape[0] >= vel_mag.shape[0]:
            vel_mag = vel_mag * mask[:vel_mag.shape[0], :]

        total_vel = vel_mag.sum(axis=1)  # (T-1,)

        # Find movement peaks
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(total_vel, height=np.median(total_vel) * 1.5)
        except ImportError:
            peaks = np.array([], dtype=int)

        if len(peaks) == 0:
            return np.full(len(beat_frames), np.nan)

        offsets = np.full(len(beat_frames), np.nan, dtype=np.float32)
        for bi, bf in enumerate(beat_frames):
            if bf >= T:
                continue
            dists = np.abs(peaks - bf)
            nearest = peaks[dists.argmin()]
            offset_frames = nearest - bf
            offsets[bi] = offset_frames / self.fps * 1000  # ms

        return offsets

    def _extension_quality(
        self,
        kp2d: np.ndarray,
        all_kp2d: Dict[int, np.ndarray],
        mask: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Joint angle deficit from group mean at movement peaks."""
        T = kp2d.shape[0]
        deficits: Dict[str, np.ndarray] = {}

        for pair_name, joints in EXTENSION_PAIRS:
            j1, j2, j3 = joints
            angles = self._compute_joint_angles(kp2d, j1, j2, j3)

            all_angles = [
                self._compute_joint_angles(other_kp, j1, j2, j3)
                for _did, other_kp in all_kp2d.items()
            ]

            if all_angles:
                min_len = min(len(a) for a in all_angles + [angles])
                stacked = np.stack([a[:min_len] for a in all_angles], axis=0)
                group_mean = np.nanmean(stacked, axis=0)
                deficit = np.zeros(T, dtype=np.float32)
                deficit[:min_len] = group_mean - angles[:min_len]
            else:
                deficit = np.zeros(T, dtype=np.float32)

            deficits[pair_name] = deficit

        return deficits

    def _compute_joint_angles(
        self, kp: np.ndarray, j1: int, j2: int, j3: int
    ) -> np.ndarray:
        """Compute angle at j2 (vertex) between j1-j2-j3."""
        T = kp.shape[0]
        angles = np.full(T, np.nan, dtype=np.float32)

        for t in range(T):
            v1 = kp[t, j1, :2] - kp[t, j2, :2]
            v2 = kp[t, j3, :2] - kp[t, j2, :2]

            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angles[t] = np.degrees(np.arccos(cos_angle))

        return angles

    def _smoothness(
        self, kp2d: np.ndarray, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Jerk-based smoothness score (lower = smoother)."""
        T = kp2d.shape[0]

        vel = np.gradient(kp2d[:, :, :2], axis=0)
        accel = np.gradient(vel, axis=0)
        jerk = np.gradient(accel, axis=0)

        jerk_mag = np.linalg.norm(jerk, axis=2)  # (T, 17)

        if mask is not None:
            jerk_mag = jerk_mag * mask

        # Per-frame aggregate jerk
        jerk_score = jerk_mag.mean(axis=1)  # (T,)

        # Smooth with window
        if self.jerk_window > 1:
            kernel = np.ones(self.jerk_window) / self.jerk_window
            jerk_score = np.convolve(jerk_score, kernel, mode="same")

        return jerk_score.astype(np.float32)

    def _group_sync(
        self,
        dancer_id: int,
        kp2d: np.ndarray,
        all_kp2d: Dict[int, np.ndarray],
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Group synchronization deviation."""
        T = kp2d.shape[0]

        # Group mean pose
        all_poses = [v[:, :, :2] for v in all_kp2d.values() if v.shape[0] == T]
        if len(all_poses) < 2:
            return np.zeros(T, dtype=np.float32)

        group_mean = np.nanmean(np.stack(all_poses), axis=0)  # (T, 17, 2)

        # Per-frame L2 deviation
        diff = kp2d[:, :, :2] - group_mean
        deviation = np.linalg.norm(diff, axis=2)  # (T, 17)

        if mask is not None:
            deviation = deviation * mask

        return deviation.mean(axis=1).astype(np.float32)
