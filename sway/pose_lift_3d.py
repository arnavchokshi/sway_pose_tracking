"""
3D pose lifting: ViTPose 2D keypoints (+ confidence) → 3D.

Backends (SWAY_LIFT_BACKEND):
  - motionagformer (default): TaatiTeam/MotionAGFormer + COCO-17 in/out.
  - poseformerv2: QitaoZhao/PoseFormerV2 (H36M-trained checkpoints); COCO ↔ H36M
    conversion follows their demo; lift_xyz stays COCO-ordered.

MotionAGFormer: clone https://github.com/TaatiTeam/MotionAGFormer → vendor/ or
  SWAY_MOTIONAGFORMER_ROOT. Weights: models/motionagformer-l-h36m.pth.tr
  (``python -m tools.prefetch_models --include-3d``).

PoseFormerV2: clone https://github.com/QitaoZhao/PoseFormerV2 → vendor/ or
  SWAY_POSEFORMERV2_ROOT. Weights: e.g. models/27_243_45.2.bin
  (``python -m tools.prefetch_models --include-poseformerv2``). Needs einops + torch-dct.

Public checkpoints for MotionAGFormer, PoseFormerV2, and MixSTE are overwhelmingly
H36M / MPI-INF — not AMASS. For dance-heavy priors you still need AMASS/AIST++
supervised training or a community checkpoint; this file only pluggable inference.

Optional depth refinement blends Depth Anything V2 samples into z (AugLift-style).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .torch_compat import torch_load_trusted

# COCO 17 (ViTPose)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # torso sides
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Extra distance constraints for PBD only (no 18th joint; export/viewer stay COCO-17).
SYNTHETIC_NECK_BONES = [
    (0, 5),  # nose → left shoulder
    (0, 6),  # nose → right shoulder
    (3, 5),  # left ear → left shoulder (reduces head yaw wobble)
    (4, 6),  # right ear → right shoulder
]

PBD_FILTER_BONES = COCO_BONES + SYNTHETIC_NECK_BONES

# COCO-17 symmetric pairs for horizontal flip (must match ViTPose joint order).
_COCO_FLIP_LEFT = [1, 3, 5, 7, 9, 11, 13, 15]
_COCO_FLIP_RIGHT = [2, 4, 6, 8, 10, 12, 14, 16]


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


def unified_export_enabled() -> bool:
    return _env_flag("SWAY_UNIFIED_3D_EXPORT", True)


def lift_gap_mode() -> str:
    v = os.environ.get("SWAY_LIFT_GAP_MODE", "hold_zero").strip().lower()
    return v if v in ("hold_zero", "linear_interp") else "hold_zero"


def bone_length_filter_enabled() -> bool:
    """Post-process lift sequence so COCO bone lengths match clip medians (reduces jitter)."""
    return _env_flag("SWAY_BONE_LENGTH_FILTER", True)


def lift_backend() -> str:
    """motionagformer | poseformerv2"""
    v = os.environ.get("SWAY_LIFT_BACKEND", "motionagformer").strip().lower()
    if v in ("poseformerv2", "poseformer_v2", "pfv2"):
        return "poseformerv2"
    return "motionagformer"


def pinhole_intrinsics(
    width: int,
    height: int,
    fov_deg: Optional[float] = None,
    fx_fy: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float, float]:
    """Return fx, fy, cx, cy, fov_deg.

    Precedence: SWAY_FX / SWAY_FY → explicit ``fx_fy`` (e.g. from ffprobe metadata) → FOV heuristic.
    """
    cx = width * 0.5
    cy = height * 0.5
    fxs = os.environ.get("SWAY_FX", "").strip()
    fys = os.environ.get("SWAY_FY", "").strip()
    if fxs and fys:
        fx, fy = float(fxs), float(fys)
        if fov_deg is not None:
            fov = float(fov_deg)
        else:
            fov = math.degrees(2.0 * math.atan(width / (2.0 * max(fx, 1e-6))))
        return fx, fy, cx, cy, fov
    if fx_fy is not None:
        fx, fy = float(fx_fy[0]), float(fx_fy[1])
        if fov_deg is not None:
            fov = float(fov_deg)
        else:
            fov = math.degrees(2.0 * math.atan(width / (2.0 * max(fx, 1e-6))))
        return fx, fy, cx, cy, fov
    fov = float(fov_deg) if fov_deg is not None else float(os.environ.get("SWAY_PINHOLE_FOV_DEG", "70"))
    fx = 0.5 * width / max(math.tan(math.radians(fov) * 0.5), 1e-6)
    fy = fx
    return fx, fy, cx, cy, fov


def depth_z_range() -> Tuple[float, float]:
    z_near = float(os.environ.get("SWAY_DEPTH_Z_NEAR", "1.0"))
    z_far = float(os.environ.get("SWAY_DEPTH_Z_FAR", "8.0"))
    if z_far <= z_near:
        z_far = z_near + 1.0
    return z_near, z_far


def normalized_depth_to_z_cam(d_norm: float, z_near: float, z_far: float) -> float:
    d = float(np.clip(d_norm, 0.0, 1.0))
    return z_near + d * (z_far - z_near)


def screen_norm_to_pixel_xy(nx: float, ny: float, w: float, h: float) -> Tuple[int, int]:
    """Invert _normalize_screen_coordinates (x,y in pixels → nx, ny)."""
    x = (nx + 1.0) * 0.5 * w
    y = (ny + h / w) * 0.5 * w
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return xi, yi


def depth_map_for_frame_t(
    depth_series: Optional[List[Tuple[int, np.ndarray]]],
    t: int,
    target_h: int,
    target_w: int,
) -> Optional[np.ndarray]:
    """Interpolate strided depth keyframes to a single H×W map for frame index t."""
    if not depth_series:
        return None
    series = sorted(depth_series, key=lambda x: x[0])
    tw, th = target_w, target_h

    def resize(m: np.ndarray) -> np.ndarray:
        if m.shape[0] == th and m.shape[1] == tw:
            return m.astype(np.float32)
        return cv2.resize(m, (tw, th), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    if t <= series[0][0]:
        return resize(series[0][1])
    if t >= series[-1][0]:
        return resize(series[-1][1])
    for i in range(len(series) - 1):
        t0, m0 = series[i]
        t1, m1 = series[i + 1]
        if t0 <= t <= t1:
            if t1 <= t0:
                return resize(m0)
            a = (t - t0) / float(t1 - t0)
            return (1.0 - a) * resize(m0) + a * resize(m1)
    return resize(series[-1][1])


def _compute_unified_world_keypoints(
    kpts_2d: np.ndarray,
    lift: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_near: float,
    z_far: float,
    depth_map: Optional[np.ndarray],
    img_w: int,
    img_h: int,
) -> Tuple[List[List[float]], List[float]]:
    """Pelvis anchor from mid-hip un-projection + scaled lift relative to hip in lift space.

    World convention: X right, Y up, Z positive into the scene (away from camera / depth).
    """
    k2 = np.asarray(kpts_2d, dtype=np.float64)
    lift_a = np.asarray(lift, dtype=np.float64).reshape(17, 3)
    uh = (float(k2[11, 0]) + float(k2[12, 0])) * 0.5
    vh = (float(k2[11, 1]) + float(k2[12, 1])) * 0.5

    if depth_map is not None:
        h, w = depth_map.shape[:2]
        px = int(np.clip(round(uh), 0, w - 1))
        py = int(np.clip(round(vh), 0, h - 1))
        d_hip = float(depth_map[py, px])
        Z_root = normalized_depth_to_z_cam(d_hip, z_near, z_far)
    else:
        Z_root = float(os.environ.get("SWAY_DEFAULT_ROOT_Z", "2.5"))

    scale_mul = float(os.environ.get("SWAY_LIFT_WORLD_SCALE", "1.0"))
    Xr = (uh - cx) * Z_root / max(fx, 1e-6)
    Yr = -(vh - cy) * Z_root / max(fy, 1e-6)
    Zr = Z_root
    T_pelvis = np.array([Xr, Yr, Zr], dtype=np.float64)

    hip_lift = (lift_a[11] + lift_a[12]) * 0.5
    lift_rel = lift_a - hip_lift
    ankle_y = max(float(k2[15, 1]), float(k2[16, 1]))
    nose_y = float(k2[0, 1])
    raw_h = max(abs(ankle_y - nose_y), 1.0)
    approx_h_cam = raw_h * Z_root / max(fy, 1e-6)
    span_lift = float(np.max(np.linalg.norm(lift_rel, axis=1)))
    span_lift = max(span_lift, 1e-6)
    s = (approx_h_cam / span_lift) * scale_mul

    world = T_pelvis + s * lift_rel
    root_xyz = [float(T_pelvis[0]), float(T_pelvis[1]), float(T_pelvis[2])]
    return [[float(world[i, j]) for j in range(3)] for i in range(17)], root_xyz


# H36M demo: fixed camera quaternion
_H36M_DEMO_QUAT = np.array(
    [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    dtype=np.float32,
)

_MAG_MODEL: Any = None
_MAG_DEVICE: Any = None
_MAG_ROOT: Optional[Path] = None

_PFV2_MODEL: Any = None
_PFV2_DEVICE: Any = None
_PFV2_ROOT: Optional[Path] = None

# PoseFormerV2 demo: COCO-17 xy → H36M-17 xy (demo/lib/preprocess.py)
_H36M_FROM_COCO_IDX = np.array([9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3], dtype=np.int64)
_COCO_LIMB_TO_H36M = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64)
_SPPL_H36M_IDX = np.array([10, 8, 0, 7], dtype=np.int64)
# PoseFormerV2 demo TTA (H36M joint indices)
_PFV2_JOINTS_LEFT = [4, 5, 6, 11, 12, 13]
_PFV2_JOINTS_RIGHT = [1, 2, 3, 14, 15, 16]


def _coco_pixels_to_h36m17_xy(keypoints: np.ndarray) -> np.ndarray:
    """(T, 17, C≥2) COCO pixel x,y → (T, 17, 2) H36M layout (PoseFormerV2 demo/lib/preprocess.py)."""
    k = np.asarray(keypoints, dtype=np.float32)
    if k.shape[-2] != 17:
        raise ValueError("expected 17 COCO joints")
    t = k.shape[0]
    xy = k[..., :2]
    out = np.zeros((t, 17, 2), dtype=np.float32)
    htps = np.zeros((t, 4, 2), dtype=np.float32)
    htps[:, 0, 0] = np.mean(xy[:, 1:5, 0], axis=1)
    htps[:, 0, 1] = np.sum(xy[:, 1:3, 1], axis=1) - xy[:, 0, 1]
    htps[:, 1, :] = np.mean(xy[:, 5:7, :], axis=1)
    htps[:, 1, :] += (xy[:, 0, :] - htps[:, 1, :]) / 3.0
    htps[:, 2, :] = np.mean(xy[:, 11:13, :], axis=1)
    htps[:, 3, :] = np.mean(xy[:, [5, 6, 11, 12], :], axis=1)
    out[:, _SPPL_H36M_IDX] = htps
    out[:, _H36M_FROM_COCO_IDX] = xy[:, _COCO_LIMB_TO_H36M]
    out[:, 9, :] -= (out[:, 9, :] - np.mean(xy[:, 5:7, :], axis=1)) / 4.0
    out[:, 7, 0] += 2.0 * (out[:, 7, 0] - np.mean(out[:, [0, 8], 0], axis=1))
    out[:, 8, 1] -= (np.mean(xy[:, 1:3, 1], axis=1) - xy[:, 0, 1]) * (2.0 / 3.0)
    return out


def _normalize_screen_coordinates_pf(X: np.ndarray, w: float, h: float) -> np.ndarray:
    """VideoPose3D-style norm (PoseFormer common/camera.py): preserve aspect via shared w scale."""
    out = np.copy(X).astype(np.float32)
    out[..., 0] = X[..., 0] / w * 2.0 - 1.0
    out[..., 1] = X[..., 1] / w * 2.0 - h / w
    return out


def _h36m17_xyz_to_coco17(xyz: np.ndarray) -> np.ndarray:
    """H36M-ordered 3D → COCO-17 for lift_xyz (direct inverse of limb map + coarse face)."""
    h = np.asarray(xyz, dtype=np.float32).reshape(17, 3)
    c = np.zeros((17, 3), dtype=np.float32)
    c[0] = h[9]
    face = 0.65 * h[9] + 0.35 * h[10]
    c[1] = c[2] = c[3] = c[4] = face
    c[5], c[6] = h[11], h[14]
    c[7], c[8] = h[12], h[15]
    c[9], c[10] = h[13], h[16]
    c[11], c[12] = h[4], h[1]
    c[13], c[14] = h[5], h[2]
    c[15], c[16] = h[6], h[3]
    return c


def _postprocess_poseformerv2_h36m_frame(post_out: np.ndarray) -> np.ndarray:
    """Match demo/vis.py: zero pelvis (H36M 0), world rot, z floor, then scale like MotionAGFormer."""
    post_out = np.asarray(post_out, dtype=np.float32).copy()
    post_out[0, :] = 0.0
    post_out = _camera_to_world_np(post_out, _H36M_DEMO_QUAT, 0.0)
    post_out[:, 2] -= float(np.min(post_out[:, 2]))
    mx = float(np.max(np.abs(post_out)))
    if mx > 1e-8:
        post_out /= mx
    return post_out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _motionagformer_search_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.environ.get("SWAY_MOTIONAGFORMER_ROOT", "").strip()
    if env:
        roots.append(Path(env).expanduser().resolve())
    roots.append(_repo_root() / "vendor" / "MotionAGFormer")
    return roots


def _motionagformer_setup_hint_lines() -> List[str]:
    root = _repo_root()
    return [
        f"Expected clone at: {root / 'vendor' / 'MotionAGFormer'} (or SWAY_MOTIONAGFORMER_ROOT).",
        f"  cd {root}",
        "  mkdir -p vendor && git clone https://github.com/TaatiTeam/MotionAGFormer.git vendor/MotionAGFormer",
        "  pip install timm",
        "  python -m tools.prefetch_models --include-3d",
        "Disable 3D: SWAY_3D_LIFT=0 or --no-pose-3d-lift",
    ]


def _ensure_motionagformer_path() -> Optional[Path]:
    global _MAG_ROOT
    if _MAG_ROOT is not None:
        return _MAG_ROOT
    for r in _motionagformer_search_roots():
        if (r / "model" / "MotionAGFormer.py").is_file():
            p = str(r.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
            _MAG_ROOT = r.resolve()
            return _MAG_ROOT
    return None


def _poseformerv2_search_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.environ.get("SWAY_POSEFORMERV2_ROOT", "").strip()
    if env:
        roots.append(Path(env).expanduser().resolve())
    roots.append(_repo_root() / "vendor" / "PoseFormerV2")
    return roots


def _poseformerv2_setup_hint_lines() -> List[str]:
    root = _repo_root()
    return [
        f"Expected clone at: {root / 'vendor' / 'PoseFormerV2'} (or SWAY_POSEFORMERV2_ROOT).",
        f"  cd {root}",
        "  mkdir -p vendor && git clone https://github.com/QitaoZhao/PoseFormerV2.git vendor/PoseFormerV2",
        "  pip install einops torch-dct timm",
        "  python -m tools.prefetch_models --include-poseformerv2",
        "Set SWAY_LIFT_BACKEND=poseformerv2 and SWAY_POSEFORMERV2_WEIGHTS if needed.",
    ]


def _ensure_poseformerv2_path() -> Optional[Path]:
    global _PFV2_ROOT
    if _PFV2_ROOT is not None:
        return _PFV2_ROOT
    for r in _poseformerv2_search_roots():
        if (r / "common" / "model_poseformer.py").is_file():
            p = str(r.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
            _PFV2_ROOT = r.resolve()
            return _PFV2_ROOT
    return None


def _poseformerv2_weights_path() -> Path:
    root = _repo_root()
    env = os.environ.get("SWAY_POSEFORMERV2_WEIGHTS", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for name in ("27_243_45.2.bin", "poseformerv2-243.bin"):
        p = root / "models" / name
        if p.is_file():
            return p
    return root / "models" / "27_243_45.2.bin"


def _poseformerv2_build_args() -> SimpleNamespace:
    nfr = int(os.environ.get("SWAY_POSEFORMERV2_NFRAMES", "243"))
    nk = int(os.environ.get("SWAY_POSEFORMERV2_FRAME_KEPT", "27"))
    nc = os.environ.get("SWAY_POSEFORMERV2_COEFF_KEPT", "").strip()
    nc_i = int(nc) if nc else nk
    depth = int(os.environ.get("SWAY_POSEFORMERV2_DEPTH", "4"))
    ed = int(os.environ.get("SWAY_POSEFORMERV2_EMBED_RATIO", "32"))
    return SimpleNamespace(
        embed_dim_ratio=ed,
        depth=depth,
        number_of_frames=nfr,
        number_of_kept_frames=nk,
        number_of_kept_coeffs=nc_i,
    )


def _load_poseformerv2_model():
    global _PFV2_MODEL, _PFV2_DEVICE
    if _PFV2_MODEL is not None:
        return _PFV2_MODEL, _PFV2_DEVICE

    root = _ensure_poseformerv2_path()
    if root is None:
        print("  [3D Lift] PoseFormerV2 repo not found — 3D lift skipped.")
        for line in _poseformerv2_setup_hint_lines():
            print(f"  [3D Lift]   {line}")
        return None, None

    try:
        import torch
        from common.model_poseformer import PoseTransformerV2  # type: ignore
    except ImportError as e:
        print(
            f"  [3D Lift] Cannot import PoseFormerV2 ({e}). "
            "Install: pip install einops torch-dct timm && clone QitaoZhao/PoseFormerV2."
        )
        return None, None

    wpath = _poseformerv2_weights_path()
    if not wpath.is_file():
        print(f"  [3D Lift] PoseFormerV2 weights not found at {wpath}")
        print("  [3D Lift] Run: python -m tools.prefetch_models --include-poseformerv2")
        return None, None

    args_ns = _poseformerv2_build_args()
    model = PoseTransformerV2(drop_path_rate=0.0, args=args_ns)

    ckpt = torch_load_trusted(wpath, map_location="cpu")
    raw = None
    if isinstance(ckpt, dict):
        raw = ckpt.get("model_pos") or ckpt.get("model") or ckpt.get("state_dict")
    if raw is None:
        print("  [3D Lift] PoseFormerV2: unexpected checkpoint format (expected model_pos).")
        return None, None
    if not isinstance(raw, dict):
        print("  [3D Lift] PoseFormerV2: expected state dict in checkpoint.")
        return None, None
    state = _strip_module_prefix(raw)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as ex:
        print(
            f"  [3D Lift] PoseFormerV2 load_state_dict failed ({ex}). "
            "Check SWAY_POSEFORMERV2_NFRAMES / FRAME_KEPT / COEFF_KEPT / DEPTH / EMBED_RATIO vs checkpoint."
        )
        return None, None

    model.eval()
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    model = model.to(dev)

    _PFV2_MODEL = model
    _PFV2_DEVICE = dev
    print(
        f"  [3D Lift] PoseFormerV2 loaded (frames={args_ns.number_of_frames}, "
        f"kept f/c={args_ns.number_of_kept_frames}/{args_ns.number_of_kept_coeffs}, device={dev})"
    )
    return _PFV2_MODEL, _PFV2_DEVICE


def _normalize_screen_coordinates(X: np.ndarray, w: float, h: float) -> np.ndarray:
    """Legacy full-frame normalization (used only when SWAY_LIFT_INPUT_NORM=screen)."""
    assert X.shape[-1] in (2, 3)
    out = np.copy(X)
    out[..., :2] = X[..., :2] / w * 2.0 - np.array([1.0, h / w], dtype=X.dtype)
    return out


def _normalize_per_person_coordinates(X: np.ndarray) -> np.ndarray:
    """Per-frame, per-batch: center on COCO mid-hip (11,12), scale by person bbox (×1.2).

    X: (B, T, 17, C) with x,y in pixels. Channels 2+ (e.g. confidence) are unchanged.
    Output xy roughly in [-1, 1] around the pelvis, matching bbox-cropped training priors.
    """
    out = np.copy(X)
    xy = X[..., :2]
    cx = (xy[..., 11, 0] + xy[..., 12, 0]) * 0.5
    cy = (xy[..., 11, 1] + xy[..., 12, 1]) * 0.5
    min_x = np.min(xy[..., 0], axis=-1)
    max_x = np.max(xy[..., 0], axis=-1)
    min_y = np.min(xy[..., 1], axis=-1)
    max_y = np.max(xy[..., 1], axis=-1)
    scale = np.maximum(max_x - min_x, max_y - min_y) * 1.2
    scale = np.maximum(scale, 1.0)
    out[..., 0] = (X[..., 0] - cx[..., np.newaxis]) / scale[..., np.newaxis] * 2.0
    out[..., 1] = (X[..., 1] - cy[..., np.newaxis]) / scale[..., np.newaxis] * 2.0
    return out


def _lift_input_normalize(X: np.ndarray, img_w: float, img_h: float) -> np.ndarray:
    """Model 2D input normalization: per-person (default) or full-frame (compat)."""
    mode = os.environ.get("SWAY_LIFT_INPUT_NORM", "person").strip().lower()
    if mode in ("screen", "global", "frame"):
        return _normalize_screen_coordinates(X, img_w, img_h)
    return _normalize_per_person_coordinates(X)


def _qrot_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by quaternion q (w,x,y,z), numpy broadcast."""
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def _camera_to_world_np(X: np.ndarray, R: np.ndarray, t: float = 0.0) -> np.ndarray:
    Rb = np.tile(R, (*X.shape[:-1], 1))
    return _qrot_np(Rb, X) + t


def _flip_data(data: np.ndarray) -> np.ndarray:
    """Horizontal flip + swap COCO left/right limbs (test-time augmentation)."""
    import copy

    lj, rj = _COCO_FLIP_LEFT, _COCO_FLIP_RIGHT
    flipped = copy.deepcopy(data)
    flipped[..., 0] *= -1.0
    flipped[..., lj + rj, :] = flipped[..., rj + lj, :]
    return flipped


def _pad_clip_to_n(
    keypoints_btjc: np.ndarray,
    n: int = 243,
) -> Tuple[np.ndarray, int]:
    """
    keypoints_btjc: (1, L, 17, C), L <= n.
    Returns (1, n, 17, C) via edge padding and valid length L (outputs for t>=L are ignored).
    """
    L = int(keypoints_btjc.shape[1])
    if L >= n:
        return keypoints_btjc[:, :n, ...].copy(), n
    pad_n = n - L
    pad = np.repeat(keypoints_btjc[:, -1:, :, :], pad_n, axis=1)
    return np.concatenate([keypoints_btjc, pad], axis=1), L


def _postprocess_pose3d_frame(post_out: np.ndarray) -> np.ndarray:
    """post_out: (17, 3) COCO-ordered joints; root at COCO pelvis (mid-hip), not nose."""
    post_out = np.asarray(post_out, dtype=np.float32).copy()
    pelvis = (post_out[11, :] + post_out[12, :]) * 0.5
    post_out -= pelvis
    post_out = _camera_to_world_np(post_out, _H36M_DEMO_QUAT, 0.0)
    post_out[:, 2] -= float(np.min(post_out[:, 2]))
    mx = float(np.max(np.abs(post_out)))
    if mx > 1e-8:
        post_out /= mx
    return post_out


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return state
    if any(k.startswith("module.") for k in state):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _load_weights_path() -> Path:
    root = _repo_root()
    env = os.environ.get("SWAY_MOTIONAGFORMER_WEIGHTS", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for name in ("motionagformer-l-h36m.pth.tr", "motionagformer-l-h36m.pth"):
        p = root / "models" / name
        if p.is_file():
            return p
    return root / "models" / "motionagformer-l-h36m.pth.tr"


def _motionagformer_ctor_args(n_layers: int) -> Dict[str, Any]:
    import torch.nn as nn

    return {
        "n_layers": n_layers,
        "dim_in": 3,
        "dim_feat": 128,
        "dim_rep": 512,
        "dim_out": 3,
        "mlp_ratio": 4,
        "act_layer": nn.GELU,
        "attn_drop": 0.0,
        "drop": 0.0,
        "drop_path": 0.0,
        "use_layer_scale": True,
        "layer_scale_init_value": 1e-5,
        "use_adaptive_fusion": True,
        "num_heads": 8,
        "qkv_bias": False,
        "qkv_scale": None,
        "hierarchical": False,
        "use_temporal_similarity": True,
        "neighbour_num": 2,
        "temporal_connection_len": 1,
        "use_tcn": False,
        "graph_only": False,
        "n_frames": 243,
    }


def _load_motionagformer_model():
    global _MAG_MODEL, _MAG_DEVICE
    if _MAG_MODEL is not None:
        return _MAG_MODEL, _MAG_DEVICE

    root = _ensure_motionagformer_path()
    if root is None:
        print("  [3D Lift] MotionAGFormer repo not found — 3D lift skipped.")
        for line in _motionagformer_setup_hint_lines():
            print(f"  [3D Lift]   {line}")
        return None, None

    try:
        import torch
        from model.MotionAGFormer import MotionAGFormer  # type: ignore
    except ImportError as e:
        print(f"  [3D Lift] Cannot import MotionAGFormer ({e}). Install: pip install timm && clone TaatiTeam/MotionAGFormer.")
        return None, None

    wpath = _load_weights_path()
    if not wpath.is_file():
        print(f"  [3D Lift] Weights not found at {wpath}")
        print("  [3D Lift] Run: python -m tools.prefetch_models --include-3d")
        return None, None

    ckpt = torch_load_trusted(wpath, map_location="cpu")
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(raw, dict):
        print("  [3D Lift] Unexpected checkpoint format.")
        return None, None
    state = _strip_module_prefix(raw)

    n_layers = 26
    if "layers.15.norm1.weight" in state and "layers.16.norm1.weight" not in state:
        n_layers = 16
    elif os.environ.get("SWAY_MOTIONAGFORMER_N_LAYERS"):
        n_layers = int(os.environ["SWAY_MOTIONAGFORMER_N_LAYERS"])

    model = MotionAGFormer(**_motionagformer_ctor_args(n_layers))
    model.load_state_dict(state, strict=True)
    model.eval()

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    model = model.to(dev)

    _MAG_MODEL = model
    _MAG_DEVICE = dev
    print(f"  [3D Lift] MotionAGFormer loaded (n_layers={n_layers}, device={dev})")
    return _MAG_MODEL, _MAG_DEVICE


def _infer_motionagformer_sequence(
    keypoints_btc: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    """
    keypoints_btc: (1, T, 17, 3) pixel x,y + score.
    Returns (T, 17, 3) postprocessed 3D (consistent model space).
    """
    model, device = _load_motionagformer_model()
    if model is None:
        return None

    import torch

    T = int(keypoints_btc.shape[1])
    acc = np.zeros((T, 17, 3), dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)

    start = 0
    while start < T:
        raw = keypoints_btc[:, start : start + 243, ...].astype(np.float32)
        clip, valid_len = _pad_clip_to_n(raw, 243)
        norm = _lift_input_normalize(clip, float(img_w), float(img_h))
        norm_aug = _flip_data(norm)

        x = torch.from_numpy(norm).float().to(device)
        xa = torch.from_numpy(norm_aug).float().to(device)

        with torch.no_grad():
            out = (model(x) + _flip_data_tensor(model(xa))) * 0.5

        out_np = out.float().cpu().numpy()[0]
        for j in range(valid_len):
            g = start + j
            if g >= T:
                break
            acc[g] += _postprocess_pose3d_frame(out_np[j])
            cnt[g] += 1.0
        start += 243

    cnt_safe = np.maximum(cnt, 1.0)
    return (acc / cnt_safe.reshape(-1, 1, 1)).astype(np.float32)


def _infer_poseformerv2_sequence(
    keypoints_btc: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    """Sliding-window PoseFormerV2 (H36M layout); returns COCO-ordered (T,17,3)."""
    model, device = _load_poseformerv2_model()
    if model is None:
        return None

    import torch

    nfr = int(os.environ.get("SWAY_POSEFORMERV2_NFRAMES", "243"))
    pad = (nfr - 1) // 2
    jl = _PFV2_JOINTS_LEFT
    jr = _PFV2_JOINTS_RIGHT

    coco_seq = keypoints_btc[0].astype(np.float32)
    h36 = _coco_pixels_to_h36m17_xy(coco_seq)
    T = int(h36.shape[0])
    acc = np.zeros((T, 17, 3), dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)

    fw, fh = float(img_w), float(img_h)
    for i in range(T):
        start = max(0, i - pad)
        end = min(i + pad, T - 1)
        chunk = h36[start : end + 1]
        left_pad, right_pad = 0, 0
        if chunk.shape[0] != nfr:
            if i < pad:
                left_pad = pad - i
            if i > T - pad - 1:
                right_pad = i + pad - (T - 1)
            chunk = np.pad(chunk, ((left_pad, right_pad), (0, 0), (0, 0)), mode="edge")
        inp = _normalize_screen_coordinates_pf(chunk, fw, fh)
        aug = inp.copy()
        aug[:, :, 0] *= -1.0
        aug[:, jl + jr, :] = aug[:, jr + jl, :]
        x = torch.from_numpy(np.stack([inp, aug], axis=0)).float().to(device).unsqueeze(0)

        with torch.no_grad():
            o0 = model(x[:, 0])
            o1 = model(x[:, 1])
        o1 = o1.clone()
        o1[:, :, :, 0] *= -1.0
        o1[:, :, jl + jr, :] = o1[:, :, jr + jl, :]
        out = (o0 + o1) * 0.5
        out[:, :, 0, :] = 0.0
        h36_3d = out[0, 0].float().cpu().numpy()
        h36_pp = _postprocess_poseformerv2_h36m_frame(h36_3d)
        coco3 = _h36m17_xyz_to_coco17(h36_pp)
        acc[i] += coco3
        cnt[i] += 1.0

    cnt_safe = np.maximum(cnt, 1.0)
    return (acc / cnt_safe.reshape(-1, 1, 1)).astype(np.float32)


def _infer_full_sequence(
    keypoints_btc: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    """
    keypoints_btc: (1, T, 17, 3) pixel x,y + score.
    Returns (T, 17, 3) postprocessed 3D (COCO order).
    """
    if lift_backend() == "poseformerv2":
        return _infer_poseformerv2_sequence(keypoints_btc, img_w, img_h)
    return _infer_motionagformer_sequence(keypoints_btc, img_w, img_h)


def _flip_data_tensor(t) -> Any:
    """Horizontal flip + COCO left/right swap for torch tensor (B,T,J,3)."""
    import torch

    lj = _COCO_FLIP_LEFT
    rj = _COCO_FLIP_RIGHT
    flipped = t.clone()
    flipped[..., 0] *= -1.0
    flipped[..., lj, :] = t[..., rj, :].clone()
    flipped[..., rj, :] = t[..., lj, :].clone()
    return flipped


def _build_keypoint_sequence_3ch(
    all_frame_data: List[dict],
    track_id: int,
    total_frames: int,
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    """Dense (T, 17, 3): x,y in pixels, z-channel = ViTPose confidence."""
    frame_kpts: Dict[int, np.ndarray] = {}
    for fd in all_frame_data:
        pose = fd.get("poses", {}).get(track_id)
        if pose is None:
            continue
        kpts = pose.get("keypoints")
        if kpts is None or (hasattr(kpts, "shape") and kpts.shape[0] < 17):
            continue
        fidx = int(fd["frame_idx"])
        k = np.asarray(kpts[:17], dtype=np.float32)
        sc = k[:, 2] if k.shape[1] > 2 else np.ones(17, dtype=np.float32)
        pts = np.stack([k[:, 0], k[:, 1], sc], axis=-1)
        frame_kpts[fidx] = pts

    if len(frame_kpts) < 10:
        return None

    frames_sorted = sorted(frame_kpts.keys())
    T = total_frames
    seq = np.zeros((T, 17, 3), dtype=np.float32)

    if lift_gap_mode() == "linear_interp":
        for fidx in frames_sorted:
            seq[fidx] = frame_kpts[fidx]
        for kpt_idx in range(17):
            known_vals = np.array([frame_kpts[f][kpt_idx] for f in frames_sorted], dtype=np.float32)
            for dim in range(3):
                seq[:, kpt_idx, dim] = np.interp(
                    np.arange(T, dtype=np.float32),
                    np.array(frames_sorted, dtype=np.float32),
                    known_vals[:, dim],
                    left=known_vals[0, dim],
                    right=known_vals[-1, dim],
                ).astype(np.float32)
        return seq

    # hold_zero: last-known (x,y) with confidence 0 in gaps; leading/trailing use nearest keyframe.
    for kpt_idx in range(17):
        fv = frame_kpts[frames_sorted[0]][kpt_idx]
        for t in range(0, frames_sorted[0]):
            seq[t, kpt_idx, 0] = fv[0]
            seq[t, kpt_idx, 1] = fv[1]
            seq[t, kpt_idx, 2] = 0.0
        last = fv.copy()
        for t in range(frames_sorted[0], T):
            if t in frame_kpts:
                last = frame_kpts[t][kpt_idx].copy()
                seq[t, kpt_idx] = last
            else:
                seq[t, kpt_idx, 0] = last[0]
                seq[t, kpt_idx, 1] = last[1]
                seq[t, kpt_idx, 2] = 0.0
    return seq


def _apply_auglift(
    pred_3d: np.ndarray,
    seq_xy_pixels: np.ndarray,
    depth_series: Optional[List[Tuple[int, np.ndarray]]],
    img_w: int,
    img_h: int,
    blend: float = 0.3,
) -> np.ndarray:
    """Blend predicted z with depth; seq_xy_pixels is (T, 17, 2) in image pixel coordinates."""
    refined = pred_3d.copy()
    if not depth_series:
        return refined
    T, N, _ = pred_3d.shape
    b = float(os.environ.get("SWAY_AUGLIFT_BLEND", str(blend)))
    for t in range(T):
        depth_map = depth_map_for_frame_t(depth_series, t, img_h, img_w)
        if depth_map is None:
            continue
        h, w = depth_map.shape[:2]
        for kpt_idx in range(N):
            px = int(np.clip(round(float(seq_xy_pixels[t, kpt_idx, 0])), 0, img_w - 1))
            py = int(np.clip(round(float(seq_xy_pixels[t, kpt_idx, 1])), 0, img_h - 1))
            if w != img_w or h != img_h:
                sx = (px + 0.5) * w / max(img_w, 1) - 0.5
                sy = (py + 0.5) * h / max(img_h, 1) - 0.5
                px = int(np.clip(round(sx), 0, w - 1))
                py = int(np.clip(round(sy), 0, h - 1))
            depth_val = float(depth_map[py, px])
            depth_z = (depth_val - 0.5) * 2.0
            refined[t, kpt_idx, 2] = (1.0 - b) * pred_3d[t, kpt_idx, 2] + b * depth_z
    return refined


def median_bone_lengths_for_sequence(seq: np.ndarray) -> np.ndarray:
    """Per-bone median Euclidean length over time; order matches PBD_FILTER_BONES.

    seq: (T, 17, 3) model-space joints (same space as lift_xyz).
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 3 or seq.shape[1] < 17 or seq.shape[2] < 3:
        raise ValueError("seq must be (T, 17, 3)")
    out = np.zeros(len(PBD_FILTER_BONES), dtype=np.float64)
    for bi, (i, j) in enumerate(PBD_FILTER_BONES):
        d = np.linalg.norm(seq[:, i, :3] - seq[:, j, :3], axis=1)
        out[bi] = float(np.median(d))
    out = np.maximum(out, 1e-8)
    return out


def enforce_bone_lengths_frame(
    frame_xyz: np.ndarray,
    target_lengths: np.ndarray,
    bones: List[Tuple[int, int]],
    pred_anchor: np.ndarray,
    n_iter: int,
) -> np.ndarray:
    """PBD-style distance constraints + pelvis translation to match predicted mid-hip."""
    x = np.asarray(frame_xyz, dtype=np.float64).copy()
    pred = np.asarray(pred_anchor, dtype=np.float64)
    for _ in range(max(1, n_iter)):
        for bi, (i, j) in enumerate(bones):
            L = float(target_lengths[bi])
            pi, pj = x[i], x[j]
            d = pj - pi
            dist = float(np.linalg.norm(d))
            if dist < 1e-12:
                d = pred[j] - pred[i]
                dist = float(np.linalg.norm(d))
                if dist < 1e-12:
                    d = np.array([1e-6, 0.0, 0.0], dtype=np.float64)
                    dist = float(np.linalg.norm(d))
            half = 0.5 * (dist - L) / dist
            corr = d * half
            x[i] += corr
            x[j] -= corr
    mid_p = (pred[11] + pred[12]) * 0.5
    mid_x = (x[11] + x[12]) * 0.5
    x += mid_p - mid_x
    return x.astype(np.float32)


def apply_bone_length_filter_to_lift_sequence(pred_3d: np.ndarray) -> np.ndarray:
    """Enforce fixed bone lengths (clip medians) per frame while staying near predictions.

    pred_3d: (T, 17, 3) in MotionAGFormer postprocessed space (lift_xyz).
    """
    pred_3d = np.asarray(pred_3d, dtype=np.float32)
    if pred_3d.shape[0] == 0:
        return pred_3d
    T = int(pred_3d.shape[0])
    lens = median_bone_lengths_for_sequence(pred_3d)
    n_iter = int(os.environ.get("SWAY_BONE_LENGTH_FILTER_ITERS", "28"))
    bones = PBD_FILTER_BONES
    out = np.empty_like(pred_3d, dtype=np.float32)
    for t in range(T):
        out[t] = enforce_bone_lengths_frame(pred_3d[t], lens, bones, pred_3d[t], n_iter)
    return out


def lift_poses_to_3d(
    all_frame_data: List[dict],
    total_frames: int,
    frame_width: int,
    frame_height: int,
    depth_series: Optional[List[Tuple[int, np.ndarray]]] = None,
    video_camera: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Adds lift_xyz (17,3) postprocessed model-space and keypoints_3d.

    When SWAY_UNIFIED_3D_EXPORT=1 (default), keypoints_3d are world XYZ (camera-consistent)
    for the Three.js viewer. Otherwise legacy [pixel_x, pixel_y, z_lift].
    """
    if lift_backend() == "poseformerv2":
        model, _ = _load_poseformerv2_model()
        backend_name = "PoseFormerV2"
    else:
        model, _ = _load_motionagformer_model()
        backend_name = "MotionAGFormer"
    if model is None:
        return all_frame_data

    if video_camera is not None and "fx" in video_camera and "fy" in video_camera:
        fx, fy, cx, cy, _fov = pinhole_intrinsics(
            frame_width,
            frame_height,
            fov_deg=float(video_camera["fov_deg"]) if video_camera.get("fov_deg") is not None else None,
            fx_fy=(float(video_camera["fx"]), float(video_camera["fy"])),
        )
    else:
        fx, fy, cx, cy, _fov = pinhole_intrinsics(frame_width, frame_height)
    z_near, z_far = depth_z_range()
    unified = unified_export_enabled()

    all_tids = set()
    for fd in all_frame_data:
        all_tids.update(fd.get("poses", {}).keys())

    print(f"  [3D Lift] Lifting {len(all_tids)} tracks to 3D ({backend_name})…")

    for tid in sorted(all_tids):
        seq = _build_keypoint_sequence_3ch(
            all_frame_data, tid, total_frames, frame_width, frame_height
        )
        if seq is None:
            continue

        btc = seq[np.newaxis, ...]
        pred_3d = _infer_full_sequence(btc, frame_width, frame_height)
        if pred_3d is None:
            break

        seq_xy = seq[:, :, :2].copy()

        if depth_series:
            pred_3d = _apply_auglift(pred_3d, seq_xy, depth_series, frame_width, frame_height)

        if bone_length_filter_enabled():
            pred_3d = apply_bone_length_filter_to_lift_sequence(pred_3d)

        for fd in all_frame_data:
            fidx = int(fd["frame_idx"])
            pose = fd.get("poses", {}).get(tid)
            if pose is None:
                continue
            kpts_2d = np.asarray(pose.get("keypoints"), dtype=np.float64)
            lift = pred_3d[fidx]
            pose["lift_xyz"] = lift.copy()
            if unified:
                dmap = depth_map_for_frame_t(depth_series, fidx, frame_height, frame_width)
                k3, root = _compute_unified_world_keypoints(
                    kpts_2d,
                    lift,
                    fx,
                    fy,
                    cx,
                    cy,
                    z_near,
                    z_far,
                    dmap,
                    frame_width,
                    frame_height,
                )
                pose["keypoints_3d"] = k3
                pose["root_xyz"] = root
            else:
                k3 = []
                for kpt_idx in range(min(17, kpts_2d.shape[0])):
                    x2d, y2d = float(kpts_2d[kpt_idx, 0]), float(kpts_2d[kpt_idx, 1])
                    z = float(lift[kpt_idx, 2])
                    k3.append([x2d, y2d, z])
                pose["keypoints_3d"] = k3

    print("  [3D Lift] Done.")
    return all_frame_data


def lift_savgol_enabled() -> bool:
    """Non-causal Savitzky-Golay on lift_xyz at JSON export (SWAY_LIFT_SAVGOL, default on)."""
    return _env_flag("SWAY_LIFT_SAVGOL", True)


def smooth_lift_xyz_for_export(all_frame_data: List[dict]) -> int:
    """
    Apply Savitzky-Golay along time to each joint's lift_xyz (x,y,z), per track.

    Operates on the dense export timeline (indices match ``all_frame_data`` order).
    In-place; only frames that already had ``lift_xyz`` are overwritten.
    Returns the number of tracks that were smoothed.
    """
    if not all_frame_data:
        return 0
    from scipy.signal import savgol_filter

    F = len(all_frame_data)
    wl0 = int(os.environ.get("SWAY_LIFT_SAVGOL_WINDOW", "11"))
    poly = int(os.environ.get("SWAY_LIFT_SAVGOL_POLY", "3"))
    wl = max(3, min(wl0, F))
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        return 0
    poly = min(max(1, poly), wl - 1)

    all_tids = set()
    for fd in all_frame_data:
        for tid, pose in fd.get("poses", {}).items():
            if pose.get("lift_xyz") is not None:
                all_tids.add(tid)

    n_smooth = 0
    idx = np.arange(F, dtype=np.float64)
    for tid in sorted(all_tids):
        series = np.full((F, 17, 3), np.nan, dtype=np.float64)
        valid = np.zeros(F, dtype=bool)
        for i, fd in enumerate(all_frame_data):
            pose = fd.get("poses", {}).get(tid)
            if pose is None:
                continue
            lx = pose.get("lift_xyz")
            if lx is None:
                continue
            a = np.asarray(lx, dtype=np.float64).reshape(17, 3)
            series[i] = a
            valid[i] = True
        if int(np.count_nonzero(valid)) < wl:
            continue

        out = series.copy()
        for j in range(17):
            for d in range(3):
                col = series[:, j, d]
                good = ~np.isnan(col)
                if int(np.count_nonzero(good)) < wl:
                    continue
                gi = np.flatnonzero(good)
                filled = np.interp(idx, gi.astype(np.float64), col[gi], left=float(col[gi[0]]), right=float(col[gi[-1]]))
                out[:, j, d] = savgol_filter(filled, wl, poly, mode="interp")

        for i in range(F):
            if not valid[i]:
                continue
            pose = all_frame_data[i].get("poses", {}).get(tid)
            if pose is None:
                continue
            pose["lift_xyz"] = out[i].astype(np.float32).copy()
        n_smooth += 1
    return n_smooth


def video_camera_from_pose_3d_camera(cam: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Rebuild intrinsics dict for lift/export from ``pose_3d.camera`` (e.g. after JSON round-trip)."""
    if not cam or "fx" not in cam or "fy" not in cam:
        return None
    out: Dict[str, Any] = {"fx": float(cam["fx"]), "fy": float(cam["fy"])}
    if cam.get("fov_deg") is not None:
        out["fov_deg"] = float(cam["fov_deg"])
    tag = cam.get("metadata_tag")
    if tag:
        out["source_key"] = str(tag)
    if cam.get("focal_length_35mm_equiv_mm") is not None:
        out["focal_length_35mm_equiv_mm"] = float(cam["focal_length_35mm_equiv_mm"])
    return out


def refresh_keypoints_3d_from_lift(
    all_frame_data: List[dict],
    frame_width: int,
    frame_height: int,
    depth_series: Optional[List[Tuple[int, np.ndarray]]] = None,
    video_camera: Optional[Dict[str, Any]] = None,
) -> None:
    """Recompute ``keypoints_3d`` / ``root_xyz`` from current ``lift_xyz`` (after smoothing)."""
    if not all_frame_data:
        return
    if video_camera is not None and "fx" in video_camera and "fy" in video_camera:
        fx, fy, cx, cy, _fov = pinhole_intrinsics(
            frame_width,
            frame_height,
            fov_deg=float(video_camera["fov_deg"]) if video_camera.get("fov_deg") is not None else None,
            fx_fy=(float(video_camera["fx"]), float(video_camera["fy"])),
        )
    else:
        fx, fy, cx, cy, _fov = pinhole_intrinsics(frame_width, frame_height)
    z_near, z_far = depth_z_range()
    unified = unified_export_enabled()
    for fd in all_frame_data:
        fidx = int(fd["frame_idx"])
        dmap = depth_map_for_frame_t(depth_series, fidx, frame_height, frame_width) if depth_series else None
        for _tid, pose in fd.get("poses", {}).items():
            lift = pose.get("lift_xyz")
            if lift is None:
                continue
            kpts_2d = np.asarray(pose.get("keypoints"), dtype=np.float64)
            lift_a = np.asarray(lift, dtype=np.float64)
            if unified:
                k3, root = _compute_unified_world_keypoints(
                    kpts_2d,
                    lift_a,
                    fx,
                    fy,
                    cx,
                    cy,
                    z_near,
                    z_far,
                    dmap,
                    frame_width,
                    frame_height,
                )
                pose["keypoints_3d"] = k3
                pose["root_xyz"] = root
            else:
                k3 = []
                for kpt_idx in range(min(17, kpts_2d.shape[0])):
                    x2d, y2d = float(kpts_2d[kpt_idx, 0]), float(kpts_2d[kpt_idx, 1])
                    z = float(lift_a[kpt_idx, 2])
                    k3.append([x2d, y2d, z])
                pose["keypoints_3d"] = k3


def export_3d_for_viewer(
    all_frame_data: List[dict],
    track_ids: List[int],
    total_frames: int,
    native_fps: float,
    frame_width: int = 1280,
    frame_height: int = 720,
    video_camera: Optional[Dict[str, Any]] = None,
) -> dict:
    """Compact pose_3d blob for Three.js (metadata.pose_3d in data.json)."""
    tracks_out: Dict[str, Any] = {}
    for tid in track_ids:
        frames_list: List[int] = []
        kpts_list: List[Any] = []
        roots_list: List[Any] = []
        for fd in all_frame_data:
            pose = fd.get("poses", {}).get(tid)
            if pose is None:
                continue
            kpts_3d = pose.get("keypoints_3d")
            if kpts_3d is None:
                kpts_2d = pose.get("keypoints", [])
                if hasattr(kpts_2d, "tolist"):
                    kpts_2d = kpts_2d.tolist()
                kpts_3d = [[float(k[0]), float(k[1]), 0.0] for k in kpts_2d[:17]]
            frames_list.append(int(fd["frame_idx"]))
            kpts_list.append(kpts_3d)
            roots_list.append(pose.get("root_xyz"))
        if frames_list:
            entry: Dict[str, Any] = {"frames": frames_list, "keypoints_3d": kpts_list}
            if unified_export_enabled() and any(r is not None for r in roots_list):
                entry["root_xyz"] = roots_list
            tracks_out[str(tid)] = entry

    out: Dict[str, Any] = {
        "fps": float(native_fps),
        "total_frames": int(total_frames),
        "keypoint_names": list(COCO_KEYPOINTS),
        "bones": [[a, b] for a, b in COCO_BONES],
        "tracks": tracks_out,
    }
    has_world = any(
        fd.get("poses", {}).get(tid, {}).get("root_xyz") is not None
        for tid in track_ids
        for fd in all_frame_data
        if isinstance(fd.get("poses", {}).get(tid), dict)
    )
    if unified_export_enabled() and has_world:
        if video_camera is not None and "fx" in video_camera and "fy" in video_camera:
            fx, fy, cx, cy, fov = pinhole_intrinsics(
                frame_width,
                frame_height,
                fov_deg=float(video_camera["fov_deg"]) if video_camera.get("fov_deg") is not None else None,
                fx_fy=(float(video_camera["fx"]), float(video_camera["fy"])),
            )
        else:
            fx, fy, cx, cy, fov = pinhole_intrinsics(frame_width, frame_height)
        out["version"] = 2
        out["camera"] = {
            "width": int(frame_width),
            "height": int(frame_height),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "fov_deg": float(fov),
            "units": "arbitrary_scene",
            "convention": "right_handed_Y_up_Z_depth",
        }
        if video_camera and video_camera.get("source_key"):
            out["camera"]["intrinsics_source"] = "video_metadata"
            out["camera"]["metadata_tag"] = str(video_camera["source_key"])
            if video_camera.get("focal_length_35mm_equiv_mm") is not None:
                out["camera"]["focal_length_35mm_equiv_mm"] = float(video_camera["focal_length_35mm_equiv_mm"])
    else:
        out["version"] = 1
    return out


def compute_joint_angles_3d(keypoints_3d: List[List[float]]) -> Dict[str, float]:
    """Joint angles (degrees) from a single frame's 17×3 model-space points."""
    kpts = np.asarray(keypoints_3d, dtype=np.float64)
    if kpts.shape[0] < 17:
        return {}

    def angle_3d(a: int, b: int, c: int) -> float:
        v1 = kpts[a] - kpts[b]
        v2 = kpts[c] - kpts[b]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return float("nan")
        cos_angle = float(np.dot(v1, v2) / (n1 * n2))
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    mid_shoulder = (kpts[5] + kpts[6]) * 0.5
    mid_hip = (kpts[11] + kpts[12]) * 0.5

    def angle_3d_shoulder(side: str) -> float:
        if side == "left":
            elbow, shoulder = 7, 5
        else:
            elbow, shoulder = 8, 6
        v1 = kpts[elbow] - kpts[shoulder]
        v2 = mid_hip - kpts[shoulder]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return float("nan")
        cos_angle = float(np.dot(v1, v2) / (n1 * n2))
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    le = angle_3d(5, 7, 9)
    re = angle_3d(6, 8, 10)
    ls = angle_3d_shoulder("left")
    rs = angle_3d_shoulder("right")
    lk = angle_3d(11, 13, 15)
    rk = angle_3d(12, 14, 16)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    spine_vec = mid_shoulder - mid_hip
    sn = np.linalg.norm(spine_vec)
    spine_lean = float("nan")
    if sn > 1e-8:
        spine_lean = float(np.degrees(np.arccos(np.clip(float(np.dot(spine_vec, up) / sn), -1.0, 1.0))))

    sym = abs(angle_3d(9, 5, 11) - angle_3d(10, 6, 12))

    return {
        "left_elbow": le,
        "right_elbow": re,
        "left_shoulder": ls,
        "right_shoulder": rs,
        "left_knee": lk,
        "right_knee": rk,
        "spine_lean": spine_lean,
        "body_symmetry": sym,
    }
