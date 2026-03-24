"""
3D pose lifting: ViTPose 2D keypoints (+ confidence) → 3D using MotionAGFormer.

Requires a clone of https://github.com/TaatiTeam/MotionAGFormer on PYTHONPATH or at:
  sway_pose_mvp/vendor/MotionAGFormer
or set SWAY_MOTIONAGFORMER_ROOT to that directory.

Weights: models/motionagformer-l-h36m.pth.tr (see prefetch_models.py --include-3d).

Optional depth refinement blends Depth Anything V2 samples into z (AugLift-style).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# COCO 17 (ViTPose)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# H36M demo: fixed camera quaternion
_H36M_DEMO_QUAT = np.array(
    [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    dtype=np.float32,
)

_MAG_MODEL: Any = None
_MAG_DEVICE: Any = None
_MAG_ROOT: Optional[Path] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _motionagformer_search_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.environ.get("SWAY_MOTIONAGFORMER_ROOT", "").strip()
    if env:
        roots.append(Path(env).expanduser().resolve())
    roots.append(_repo_root() / "vendor" / "MotionAGFormer")
    return roots


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


def _normalize_screen_coordinates(X: np.ndarray, w: float, h: float) -> np.ndarray:
    assert X.shape[-1] in (2, 3)
    out = np.copy(X)
    out[..., :2] = X[..., :2] / w * 2.0 - np.array([1.0, h / w], dtype=X.dtype)
    return out


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
    """Test-time flip augmentation (MotionAGFormer demo)."""
    import copy

    left_joints = [1, 2, 3, 14, 15, 16]
    right_joints = [4, 5, 6, 11, 12, 13]
    flipped = copy.deepcopy(data)
    flipped[..., 0] *= -1.0
    flipped[..., left_joints + right_joints, :] = flipped[..., right_joints + left_joints, :]
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
    """post_out: (17, 3) — root-relative H36M-style; match demo/vis.py."""
    post_out = np.asarray(post_out, dtype=np.float32).copy()
    post_out[0, :] = 0.0
    post_out = _camera_to_world_np(post_out, _H36M_DEMO_QUAT, 0.0)
    post_out[:, 2] -= float(np.min(post_out[:, 2]))
    mx = float(np.max(post_out))
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
        print("  [3D Lift] MotionAGFormer repo not found. Clone to vendor/MotionAGFormer or set SWAY_MOTIONAGFORMER_ROOT.")
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
        print("  [3D Lift] Run: python prefetch_models.py --include-3d")
        return None, None

    ckpt = torch.load(wpath, map_location="cpu")
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


def _infer_full_sequence(
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
        norm = _normalize_screen_coordinates(clip, float(img_w), float(img_h))
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
    return (acc / cnt_safe[:, None]).astype(np.float32)


def _flip_data_tensor(t) -> Any:
    """Flip augmentation for torch tensor (B,T,J,3)."""
    import torch

    left_joints = [1, 2, 3, 14, 15, 16]
    right_joints = [4, 5, 6, 11, 12, 13]
    flipped = t.clone()
    flipped[..., 0] *= -1.0
    flipped[..., left_joints, :] = t[..., right_joints, :].clone()
    flipped[..., right_joints, :] = t[..., left_joints, :].clone()
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


def _apply_auglift(
    pred_3d: np.ndarray,
    seq_2d_norm_xy: np.ndarray,
    depth_map: np.ndarray,
    img_w: int,
    img_h: int,
    blend: float = 0.3,
) -> np.ndarray:
    """Blend predicted z with depth map samples (normalized depth)."""
    refined = pred_3d.copy()
    T, N, _ = pred_3d.shape
    h, w = depth_map.shape[:2]
    for t in range(T):
        for kpt_idx in range(N):
            nx = float(seq_2d_norm_xy[t, kpt_idx, 0])
            ny = float(seq_2d_norm_xy[t, kpt_idx, 1])
            px = int(nx * w) if w else 0
            py = int(ny * h) if h else 0
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            depth_val = float(depth_map[py, px])
            depth_z = (depth_val - 0.5) * 2.0
            refined[t, kpt_idx, 2] = (1.0 - blend) * pred_3d[t, kpt_idx, 2] + blend * depth_z
    return refined


def lift_poses_to_3d(
    all_frame_data: List[dict],
    total_frames: int,
    frame_width: int,
    frame_height: int,
    depth_map: Optional[np.ndarray] = None,
) -> List[dict]:
    """
    Adds lift_xyz (17,3) model-space and keypoints_3d (pixel x,y + display z) to each pose dict.
    """
    model, _ = _load_motionagformer_model()
    if model is None:
        return all_frame_data

    all_tids = set()
    for fd in all_frame_data:
        all_tids.update(fd.get("poses", {}).keys())

    print(f"  [3D Lift] Lifting {len(all_tids)} tracks to 3D (MotionAGFormer)…")

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
        seq_norm = _normalize_screen_coordinates(
            np.concatenate([seq_xy, seq[:, :, 2:3]], axis=-1)[np.newaxis, ...],
            float(frame_width),
            float(frame_height),
        )[0]

        if depth_map is not None:
            pred_3d = _apply_auglift(pred_3d, seq_norm, depth_map, frame_width, frame_height)

        for fd in all_frame_data:
            fidx = int(fd["frame_idx"])
            pose = fd.get("poses", {}).get(tid)
            if pose is None:
                continue
            kpts_2d = np.asarray(pose.get("keypoints"), dtype=np.float64)
            lift = pred_3d[fidx]
            pose["lift_xyz"] = lift.copy()
            k3 = []
            for kpt_idx in range(min(17, kpts_2d.shape[0])):
                x2d, y2d = float(kpts_2d[kpt_idx, 0]), float(kpts_2d[kpt_idx, 1])
                z = float(lift[kpt_idx, 2])
                k3.append([x2d, y2d, z])
            pose["keypoints_3d"] = k3

    print("  [3D Lift] Done.")
    return all_frame_data


def export_3d_for_viewer(
    all_frame_data: List[dict],
    track_ids: List[int],
    total_frames: int,
    native_fps: float,
) -> dict:
    """Compact pose_3d blob for Three.js (metadata.pose_3d in data.json)."""
    tracks_out: Dict[str, Any] = {}
    for tid in track_ids:
        frames_list: List[int] = []
        kpts_list: List[Any] = []
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
        if frames_list:
            tracks_out[str(tid)] = {"frames": frames_list, "keypoints_3d": kpts_list}

    return {
        "fps": float(native_fps),
        "total_frames": int(total_frames),
        "keypoint_names": list(COCO_KEYPOINTS),
        "bones": [[a, b] for a, b in COCO_BONES],
        "tracks": tracks_out,
    }


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
