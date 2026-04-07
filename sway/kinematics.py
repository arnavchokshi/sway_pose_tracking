"""
Phase 2.5: Resilient Kinematics — High-Stability Joint Angle Calculations

Implements joint geometry with:
- Spine Vector: Mid-Shoulder (avg 5, 6) → Mid-Hip (avg 11, 12)
- Shoulders: Angle at shoulder between Elbow→Shoulder and Shoulder→Spine-Vector
- Elbows: Shoulder → Elbow → Wrist
- Knees: Hip → Knee → Ankle
- Zero-Length Guard: Return NaN if vector length < 1e-6 to prevent ZeroDivisionError

All computations vectorized with NumPy for M2 performance.
"""

from typing import Dict, Optional, Tuple

import numpy as np

# Minimum vector length; below this, return NaN
ZERO_LENGTH_THRESHOLD = 1e-6

# COCO keypoint indices
IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER = 5, 6
IDX_LEFT_ELBOW, IDX_RIGHT_ELBOW = 7, 8
IDX_LEFT_WRIST, IDX_RIGHT_WRIST = 9, 10
IDX_LEFT_HIP, IDX_RIGHT_HIP = 11, 12
IDX_LEFT_KNEE, IDX_RIGHT_KNEE = 13, 14
IDX_LEFT_ANKLE, IDX_RIGHT_ANKLE = 15, 16

# Joint names and vertex-based triplets: (p1, vertex, p2) for angle at vertex
# Shoulders: Elbow -> Shoulder -> Spine-Vector (spine = mid_shoulder to mid_hip)
# Elbows: Shoulder -> Elbow -> Wrist
# Knees: Hip -> Knee -> Ankle
# For shoulders, we use mid_hip as the "spine direction" point (angle at shoulder)
JOINT_ANGLE_DEFS = {
    "left_shoulder": (IDX_LEFT_ELBOW, IDX_LEFT_SHOULDER, "spine"),   # vertex 5, spine dir
    "right_shoulder": (IDX_RIGHT_ELBOW, IDX_RIGHT_SHOULDER, "spine"),
    "left_elbow": (IDX_LEFT_SHOULDER, IDX_LEFT_ELBOW, IDX_LEFT_WRIST),
    "right_elbow": (IDX_RIGHT_SHOULDER, IDX_RIGHT_ELBOW, IDX_RIGHT_WRIST),
    "left_knee": (IDX_LEFT_HIP, IDX_LEFT_KNEE, IDX_LEFT_ANKLE),
    "right_knee": (IDX_RIGHT_HIP, IDX_RIGHT_KNEE, IDX_RIGHT_ANKLE),
}
JOINT_NAMES = list(JOINT_ANGLE_DEFS.keys())


def _angle_at_vertex(
    p1: np.ndarray,
    vertex: np.ndarray,
    p2: np.ndarray,
    threshold: float = ZERO_LENGTH_THRESHOLD,
) -> float:
    """
    Inner angle at vertex (degrees [0, 180]). Returns NaN if any vector length < threshold.
    """
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = np.sqrt(np.sum(v1 * v1) + 1e-12)
    n2 = np.sqrt(np.sum(v2 * v2) + 1e-12)
    if n1 < threshold or n2 < threshold:
        return float("nan")
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _get_mid_point(kpts: np.ndarray, i: int, j: int) -> np.ndarray:
    """Average of keypoints i and j."""
    return (np.asarray(kpts[i, :2], dtype=np.float64) + np.asarray(kpts[j, :2], dtype=np.float64)) / 2


def calculate_joint_angles(
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.3,
) -> Dict[str, Optional[float]]:
    """
    Calculate 6 joint angles with spine vector and zero-length guards.
    Scalar version for compatibility.
    """
    result = {}
    if keypoints.shape[0] < 17:
        for name in JOINT_NAMES:
            result[name] = None
        return result

    def _score(idx: int) -> float:
        if scores is not None and idx < len(scores):
            return float(scores[idx])
        if keypoints.shape[1] > 2 and idx < keypoints.shape[0]:
            return float(keypoints[idx, 2])
        return 0.0

    mid_shoulder = _get_mid_point(keypoints, IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER)
    mid_hip = _get_mid_point(keypoints, IDX_LEFT_HIP, IDX_RIGHT_HIP)
    spine_direction = mid_hip  # Point on spine for shoulder angle

    for name, defn in JOINT_ANGLE_DEFS.items():
        if defn[2] == "spine":
            i1, i2, _ = defn
            s1 = _score(i1)
            s2 = _score(i2)
            s_spine = (_score(IDX_LEFT_SHOULDER) + _score(IDX_RIGHT_SHOULDER) + _score(IDX_LEFT_HIP) + _score(IDX_RIGHT_HIP)) / 4
            if s1 < confidence_threshold or s2 < confidence_threshold or s_spine < confidence_threshold:
                result[name] = None
                continue
            p1 = keypoints[i1, :2].astype(np.float64)
            vertex = keypoints[i2, :2].astype(np.float64)
            p2 = spine_direction
            angle = _angle_at_vertex(p1, vertex, p2)
        else:
            i1, i2, i3 = defn
            if _score(i1) < confidence_threshold or _score(i2) < confidence_threshold or _score(i3) < confidence_threshold:
                result[name] = None
                continue
            p1 = keypoints[i1, :2].astype(np.float64)
            vertex = keypoints[i2, :2].astype(np.float64)
            p2 = keypoints[i3, :2].astype(np.float64)
            angle = _angle_at_vertex(p1, vertex, p2)
        result[name] = angle if not np.isnan(angle) else None

    return result


def _vectorized_angle_at_vertex(
    p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray,
    threshold: float = ZERO_LENGTH_THRESHOLD,
) -> np.ndarray:
    """
    Vectorized inner angle. p1, vertex, p2: (..., 2) or (F, T, 2).
    Returns (...,) angles in degrees.
    """
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = np.sqrt(np.sum(v1 * v1, axis=-1) + 1e-12)
    n2 = np.sqrt(np.sum(v2 * v2, axis=-1) + 1e-12)
    bad = (n1 < threshold) | (n2 < threshold)
    cos_angle = np.sum(v1 * v2, axis=-1) / (n1 * n2 + 1e-12)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_angle))
    ang = np.where(bad, np.nan, ang)
    return ang


def compute_joint_angles_vectorized(
    kpts: np.ndarray,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """
    Vectorized joint angles. kpts: (F, T, 17, 3) — last dim (x, y, score).
    Returns (F, T, 6) with NaN for low confidence or zero-length vectors.
    """
    F, T, _, _ = kpts.shape
    angles = np.full((F, T, 6), np.nan, dtype=np.float64)

    # Mid-shoulder and mid-hip (F, T, 2)
    mid_shoulder = (kpts[:, :, IDX_LEFT_SHOULDER, :2] + kpts[:, :, IDX_RIGHT_SHOULDER, :2]) / 2
    mid_hip = (kpts[:, :, IDX_LEFT_HIP, :2] + kpts[:, :, IDX_RIGHT_HIP, :2]) / 2

    # Scores
    def _s(j: int):
        return kpts[:, :, j, 2] if kpts.shape[3] > 2 else np.ones((F, T))

    # Left shoulder: elbow(7) -> shoulder(5) -> spine(mid_hip)
    j = 0
    low = (_s(7) < confidence_threshold) | (_s(5) < confidence_threshold) | (_s(11) < confidence_threshold) | (_s(12) < confidence_threshold)
    ang = _vectorized_angle_at_vertex(
        kpts[:, :, 7, :2], kpts[:, :, 5, :2], mid_hip
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    # Right shoulder
    j = 1
    low = (_s(8) < confidence_threshold) | (_s(6) < confidence_threshold) | (_s(11) < confidence_threshold) | (_s(12) < confidence_threshold)
    ang = _vectorized_angle_at_vertex(
        kpts[:, :, 8, :2], kpts[:, :, 6, :2], mid_hip
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    # Left elbow
    j = 2
    low = (_s(5) < confidence_threshold) | (_s(7) < confidence_threshold) | (_s(9) < confidence_threshold)
    ang = _vectorized_angle_at_vertex(
        kpts[:, :, 5, :2], kpts[:, :, 7, :2], kpts[:, :, 9, :2]
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    # Right elbow
    j = 3
    low = (_s(6) < confidence_threshold) | (_s(8) < confidence_threshold) | (_s(10) < confidence_threshold)
    ang = _vectorized_angle_at_vertex(
        kpts[:, :, 6, :2], kpts[:, :, 8, :2], kpts[:, :, 10, :2]
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    # Left knee
    j = 4
    low = (_s(11) < confidence_threshold) | (_s(13) < confidence_threshold) | (_s(15) < confidence_threshold)
    ang = _vectorized_angle_at_vertex(
        kpts[:, :, 11, :2], kpts[:, :, 13, :2], kpts[:, :, 15, :2]
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    # Right knee
    j = 5
    low = (_s(12) < confidence_threshold) | (_s(14) < confidence_threshold) | (_s(16) < confidence_threshold)
    ang = _vectorized_angle_at_vertex(
        kpts[:, :, 12, :2], kpts[:, :, 14, :2], kpts[:, :, 16, :2]
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    return angles


def _vectorized_angle_at_vertex_3d(
    p1: np.ndarray,
    vertex: np.ndarray,
    p2: np.ndarray,
    threshold: float = ZERO_LENGTH_THRESHOLD,
) -> np.ndarray:
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = np.sqrt(np.sum(v1 * v1, axis=-1) + 1e-12)
    n2 = np.sqrt(np.sum(v2 * v2, axis=-1) + 1e-12)
    bad = (n1 < threshold) | (n2 < threshold)
    cos_angle = np.sum(v1 * v2, axis=-1) / (n1 * n2 + 1e-12)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_angle))
    ang = np.where(bad, np.nan, ang)
    return ang


def compute_joint_angles_vectorized_3d(
    xyz: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """
    Same six joints as compute_joint_angles_vectorized, using 3D positions (model space).
    xyz: (F, T, 17, 3), scores: (F, T, 17) from 2D detector confidence.
    """
    F, T, _, _ = xyz.shape
    angles = np.full((F, T, 6), np.nan, dtype=np.float64)

    mid_shoulder = (xyz[:, :, IDX_LEFT_SHOULDER, :] + xyz[:, :, IDX_RIGHT_SHOULDER, :]) / 2
    mid_hip = (xyz[:, :, IDX_LEFT_HIP, :] + xyz[:, :, IDX_RIGHT_HIP, :]) / 2

    def _s(j: int):
        return scores[:, :, j]

    j = 0
    low = (
        (_s(7) < confidence_threshold)
        | (_s(5) < confidence_threshold)
        | (_s(11) < confidence_threshold)
        | (_s(12) < confidence_threshold)
    )
    ang = _vectorized_angle_at_vertex_3d(
        xyz[:, :, 7, :], xyz[:, :, 5, :], mid_hip
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    j = 1
    low = (
        (_s(8) < confidence_threshold)
        | (_s(6) < confidence_threshold)
        | (_s(11) < confidence_threshold)
        | (_s(12) < confidence_threshold)
    )
    ang = _vectorized_angle_at_vertex_3d(
        xyz[:, :, 8, :], xyz[:, :, 6, :], mid_hip
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    j = 2
    low = (
        (_s(5) < confidence_threshold)
        | (_s(7) < confidence_threshold)
        | (_s(9) < confidence_threshold)
    )
    ang = _vectorized_angle_at_vertex_3d(
        xyz[:, :, 5, :], xyz[:, :, 7, :], xyz[:, :, 9, :]
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    j = 3
    low = (
        (_s(6) < confidence_threshold)
        | (_s(8) < confidence_threshold)
        | (_s(10) < confidence_threshold)
    )
    ang = _vectorized_angle_at_vertex_3d(
        xyz[:, :, 6, :], xyz[:, :, 8, :], xyz[:, :, 10, :]
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    j = 4
    low = (
        (_s(11) < confidence_threshold)
        | (_s(13) < confidence_threshold)
        | (_s(15) < confidence_threshold)
    )
    ang = _vectorized_angle_at_vertex_3d(
        xyz[:, :, 11, :], xyz[:, :, 13, :], xyz[:, :, 15, :]
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    j = 5
    low = (
        (_s(12) < confidence_threshold)
        | (_s(14) < confidence_threshold)
        | (_s(16) < confidence_threshold)
    )
    ang = _vectorized_angle_at_vertex_3d(
        xyz[:, :, 12, :], xyz[:, :, 14, :], xyz[:, :, 16, :]
    )
    ang[low] = np.nan
    angles[:, :, j] = ang

    return angles
