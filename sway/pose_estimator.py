"""
Top-Down Pose Estimation Module — ViTPose (V3.0)

V3.0: ViTPose-Large for high-fidelity pose estimation (15° wrist/elbow accuracy).
Extracts 17 COCO keypoints per detected person. Batched [N, 3, 256, 192], fp16 on MPS unless
``SWAY_VITPOSE_FP32=1``. Production ``main.py`` reapplies ViTPose env (§9.0.1) after params unless
``SWAY_UNLOCK_POSE_TUNING=1``. Verbose Phase 5 / ViTPose timing logs are **off by default**; set ``SWAY_VITPOSE_DEBUG=1`` to enable.
(processor vs model vs post-process; hybrid SAM masked per-person forwards).
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

# COCO 17 keypoint order (for JSON export and skeleton visualization)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton edges for visualization: (joint_a, joint_b) pairs
def _local_files_only() -> bool:
    """True when HF models must not hit the network (plane / air-gapped)."""
    for key in ("SWAY_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        v = os.environ.get(key, "")
        if str(v).lower() in ("1", "true", "yes"):
            return True
    return False


COCO_SKELETON_EDGES = [
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),   # right arm
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 6), (11, 12),   # shoulders, hips
    (5, 11), (6, 12),   # torso
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
]


# Bbox padding: expand YOLO box by this fraction in all directions before cropping for ViTPose.
# Prevents limbs (e.g. fast hand raises) from being cut off at crop boundaries.
BBOX_PADDING = 0.15


def vitpose_smart_pad_enabled() -> bool:
    """
    Asymmetric crop expansion + aspect hint for ViTPose (``smart_expand_bbox_xyxy`` in ``main.py``).

    **Default on** when unset. Production and Lab subprocesses set ``SWAY_VITPOSE_SMART_PAD=1`` via
    ``apply_master_locked_pose_env`` / ``freeze_lab_subprocess_pose_env`` unless
    ``SWAY_UNLOCK_POSE_TUNING=1``. Opt out explicitly with ``SWAY_VITPOSE_SMART_PAD=0`` / ``false`` /
    ``no`` / ``off``.
    """
    v = os.environ.get("SWAY_VITPOSE_SMART_PAD", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return True


def smart_expand_bbox_xyxy(
    box: Tuple[float, float, float, float],
    prev_box: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    *,
    base_pad_frac: float = 0.12,
    target_wh_ratio: float = 0.5,
    lead_frac: float = 0.20,
    lift_top_frac: float = 0.40,
) -> Tuple[float, float, float, float]:
    """
    Expand tracker box for ViTPose: optional velocity-leading side, lift-aware top pad,
    then nudge aspect ratio toward ``target_wh_ratio`` (width / height) before a light symmetric pad.
    """
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

    if prev_box is not None:
        px1, py1, px2, py2 = prev_box
        pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        dx, dy = cx - pcx, cy - pcy
        if dx > 6.0:
            x2 += lead_frac * w
        elif dx < -6.0:
            x1 -= lead_frac * w
        if dy < -5.0 and abs(dx) < 5.0:
            y1 -= lift_top_frac * h

    # Aspect: ViTPose likes a tall-ish person crop; widen canvas (pad top/bottom) when box is too squat.
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    wh = w / h
    if wh > target_wh_ratio + 1e-3:
        need_h = w / target_wh_ratio
        extra = max(0.0, need_h - h)
        y1 -= 0.5 * extra
        y2 += 0.5 * extra
    elif wh < target_wh_ratio - 1e-3 and wh > 1e-6:
        need_w = h * target_wh_ratio
        extra = max(0.0, need_w - w)
        x1 -= 0.5 * extra
        x2 += 0.5 * extra

    pw = base_pad_frac * max(1.0, x2 - x1)
    ph = base_pad_frac * max(1.0, y2 - y1)
    x1 -= pw
    x2 += pw
    y1 -= ph
    y2 += ph

    x1 = max(0.0, min(float(img_w - 1), x1))
    y1 = max(0.0, min(float(img_h - 1), y1))
    x2 = max(x1 + 1.0, min(float(img_w), x2))
    y2 = max(y1 + 1.0, min(float(img_h), y2))
    return (x1, y1, x2, y2)


def vitpose_force_fp32() -> bool:
    """
    When True, keep ViTPose in float32 on MPS/CUDA (slower, sometimes numerically safer).
    Default False — the pipeline still uses FP16 on GPU when this is off.
    Env: SWAY_VITPOSE_FP32
    """
    v = os.environ.get("SWAY_VITPOSE_FP32", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def vitpose_use_fast_image_processor() -> bool:
    """
    Hugging Face may default to a "fast" ``VitPoseImageProcessor`` (different outputs / code path).
    Default **False** (slow processor) for stability; stalls on MPS + newer torch/transformers have
    been reported on the fast path. Opt in: ``SWAY_VITPOSE_USE_FAST=1``.
    """
    v = os.environ.get("SWAY_VITPOSE_USE_FAST", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return False


def _load_vitpose_processor(model_name: str, *, local_files_only: bool):
    use_fast = vitpose_use_fast_image_processor()
    try:
        return AutoProcessor.from_pretrained(
            model_name, local_files_only=local_files_only, use_fast=use_fast
        )
    except TypeError:
        return AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)


def vitpose_max_per_forward() -> int:
    """
    When > 0, split a single-frame multi-person ViTPose forward into chunks of at most this many
    boxes (VRAM / stability for crowded frames). 0 = one forward for all boxes (default).
    Env: SWAY_VITPOSE_MAX_PER_FORWARD

    Note: ``apply_master_locked_pose_env`` unsets this var so CUDA can batch all people in one
    forward. On Apple MPS, an unset cap plus many boxes can hang or take extreme time on the first
    forward; see ``vitpose_effective_max_per_forward``.
    """
    v = os.environ.get("SWAY_VITPOSE_MAX_PER_FORWARD", "").strip()
    if not v:
        return 0
    try:
        return max(1, int(v))
    except ValueError:
        return 0


def vitpose_effective_max_per_forward(device: Union[torch.device, str]) -> int:
    """Chunk size for ViTPose forward: explicit env, else a safe default on MPS only."""
    cap = vitpose_max_per_forward()
    if cap > 0:
        return cap
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "mps":
        try:
            # Default 2: smaller per-forward batches reduce peak MPS memory / long first-compile stalls.
            raw = int(os.environ.get("SWAY_VITPOSE_MPS_CHUNK", "2").strip())
        except ValueError:
            raw = 2
        return max(1, min(raw, 32))
    return 0


def _mps_synchronize_if_needed(device: torch.device) -> None:
    """Optional sync after MPS forwards (timing / ordering). Set SWAY_VITPOSE_MPS_SYNC=0 to skip."""
    if device.type != "mps" or not torch.backends.mps.is_available():
        return
    v = os.environ.get("SWAY_VITPOSE_MPS_SYNC", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    torch.mps.synchronize()


def vitpose_debug_enabled() -> bool:
    """Verbose ViTPose timing / tensor logs (processor → model → post). Default off; ``SWAY_VITPOSE_DEBUG=1`` enables."""
    v = os.environ.get("SWAY_VITPOSE_DEBUG", "").strip().lower()
    if not v:
        return False
    if v in ("0", "false", "no", "off"):
        return False
    return v in ("1", "true", "yes", "on")


def _vitpose_dbg(msg: str) -> None:
    if vitpose_debug_enabled():
        print(f"  [vitpose_dbg] {msg}", flush=True)


def _summarize_model_inputs(model_kwargs: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in sorted(model_kwargs.keys()):
        v = model_kwargs[k]
        if torch.is_tensor(v):
            parts.append(f"{k}={tuple(v.shape)}/{v.dtype}/{v.device.type}")
        else:
            parts.append(f"{k}={type(v).__name__}")
    return "; ".join(parts) if parts else "(empty)"


def _to_float32(obj):
    """Convert model outputs to float32 for post-processing (scipy does not support float16)."""
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if hasattr(obj, "keys"):  # dict or ModelOutput
        return type(obj)(**{k: _to_float32(v) for k, v in obj.items()})
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_float32(x) for x in obj)
    return obj


def _expand_bbox(
    box: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    padding: float = BBOX_PADDING,
) -> Tuple[float, float, float, float]:
    """Expand bbox by padding fraction in all directions; clamp to image bounds."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_w = padding * w
    pad_h = padding * h
    x1_new = max(0, x1 - pad_w)
    y1_new = max(0, y1 - pad_h)
    x2_new = min(img_w, x2 + pad_w)
    y2_new = min(img_h, y2 + pad_h)
    return (float(x1_new), float(y1_new), float(x2_new), float(y2_new))


def xyxy_to_coco(box: Tuple[float, float, float, float]) -> List[float]:
    """Convert YOLO xyxy (x1,y1,x2,y2) to COCO format (x, y, w, h)."""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _apply_sam_mask_gate_to_expanded_crop(
    frame_rgb: np.ndarray,
    box: Tuple[float, float, float, float],
    pad: float,
    mask_hw: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    """
    Expand box, crop RGB, set pixels outside SAM mask to gray (114) inside crop.
    mask_hw: bool (H_box, W_box) aligned to integer rounding of ``box``.
    Returns (crop_rgb, offset_x, offset_y) to map keypoints back to full image.
    """
    img_h, img_w = frame_rgb.shape[:2]
    ex1, ey1, ex2, ey2 = _expand_bbox(box, img_w, img_h, pad)
    ex1i = max(0, int(ex1))
    ey1i = max(0, int(ey1))
    ex2i = min(img_w, int(round(ex2)))
    ey2i = min(img_h, int(round(ey2)))
    crop = frame_rgb[ey1i:ey2i, ex1i:ex2i].copy()
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return crop, ex1i, ey1i
    gate = np.zeros((ch, cw), dtype=bool)
    x1, y1, x2, y2 = (int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3])))
    mh, mw = mask_hw.shape
    ox = x1 - ex1i
    oy = y1 - ey1i
    y1c, y2c = max(0, oy), min(ch, oy + mh)
    x1c, x2c = max(0, ox), min(cw, ox + mw)
    m_y1 = max(0, -oy)
    m_x1 = max(0, -ox)
    sl_h = y2c - y1c
    sl_w = x2c - x1c
    if sl_h > 0 and sl_w > 0:
        gate[y1c:y2c, x1c:x2c] = mask_hw[m_y1 : m_y1 + sl_h, m_x1 : m_x1 + sl_w]
    gray = np.array([114, 114, 114], dtype=np.uint8)
    crop[~gate] = gray
    return crop, ex1i, ey1i


class PoseEstimator:
    """
    ViTPose-based pose estimator for top-down keypoint extraction.

    Loads model once; call estimate_poses() per frame with boxes from tracker.
    Keypoints are returned in global image coordinates.
    """

    def __init__(
        self,
        device: Union[torch.device, str] = "cpu",
        model_name: str = "usyd-community/vitpose-plus-large",
    ):
        """
        Args:
            device: torch device (mps/cpu/cuda).
            model_name: Hugging Face model id. V3.0: vitpose-plus-large (0.4B, high fidelity).
            With SWAY_OFFLINE / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE=1, loads from local HF cache only (see README.md).
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_name = model_name

        local_only = _local_files_only()
        if not vitpose_use_fast_image_processor():
            print(
                "  ViTPose: slow image processor (default; set SWAY_VITPOSE_USE_FAST=1 for HF fast path).",
                flush=True,
            )
        if local_only:
            self.processor = _load_vitpose_processor(model_name, local_files_only=True)
            self.model = VitPoseForPoseEstimation.from_pretrained(
                model_name, local_files_only=True
            )
        else:
            # Prefer disk cache only first so repeat runs do not hit the Hub for metadata
            # or "is there an update?" checks when weights already live under HF_HOME.
            try:
                self.processor = _load_vitpose_processor(model_name, local_files_only=True)
                self.model = VitPoseForPoseEstimation.from_pretrained(
                    model_name, local_files_only=True
                )
            except Exception:
                print(
                    f"  ViTPose: local Hugging Face cache incomplete for {model_name}; "
                    "downloading / resolving from the Hub (needs network)…",
                    flush=True,
                )
                self.processor = _load_vitpose_processor(model_name, local_files_only=False)
                self.model = VitPoseForPoseEstimation.from_pretrained(
                    model_name, local_files_only=False
                )
        self.model.to(self.device)
        self.model.eval()
        # fp16 on MPS/CUDA unless SWAY_VITPOSE_FP32=1 (opt-in full precision)
        _fp32 = vitpose_force_fp32()
        use_fp16 = self.device.type in ("mps", "cuda") and not _fp32
        self.use_fp16 = use_fp16
        if _fp32 and self.device.type in ("mps", "cuda"):
            print("  ViTPose: SWAY_VITPOSE_FP32=1 — using float32 on GPU.", flush=True)
        if use_fp16:
            self.model = self.model.to(torch.float16)

        self._vitpose_chunk = vitpose_effective_max_per_forward(self.device)
        if self.device.type == "mps" and self._vitpose_chunk > 0 and not os.environ.get(
            "SWAY_VITPOSE_MAX_PER_FORWARD", ""
        ).strip():
            print(
                f"  ViTPose: MPS multi-person chunk size {self._vitpose_chunk} "
                f"(set SWAY_VITPOSE_MAX_PER_FORWARD or SWAY_VITPOSE_MPS_CHUNK to override).",
                flush=True,
            )
        if vitpose_debug_enabled():
            _vitpose_dbg(
                f"init model={model_name!r} device={self.device} fp16={self.use_fp16} "
                f"chunk_cap={self._vitpose_chunk} torch={torch.__version__}"
            )

    def _estimate_poses_batch(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        track_ids: List[int],
        paddings: List[float],
    ) -> Dict[int, Dict]:
        """Single-image ViTPose forward for N boxes (full frame)."""
        if len(boxes) == 0:
            return {}
        assert len(boxes) == len(track_ids) == len(paddings)
        cap = self._vitpose_chunk
        if cap and len(boxes) > cap:
            _vitpose_dbg(
                f"batch_split total_boxes={len(boxes)} cap={cap} -> "
                f"{(len(boxes) + cap - 1) // cap} sub-forwards"
            )
            out: Dict[int, Dict] = {}
            for start in range(0, len(boxes), cap):
                sl = slice(start, start + cap)
                _vitpose_dbg(f"batch_split chunk start={start} end={start + len(boxes[sl])} tids={list(track_ids[sl])}")
                out.update(
                    self._estimate_poses_batch(
                        frame, boxes[sl], track_ids[sl], paddings[sl]
                    )
                )
            return out
        img_h, img_w = frame.shape[:2]
        _vitpose_dbg(
            f"fullframe_batch begin n={len(boxes)} tids={list(track_ids)} "
            f"frame_hw=({img_h},{img_w})"
        )
        boxes_expanded = [
            _expand_bbox(b, img_w, img_h, pad) for b, pad in zip(boxes, paddings)
        ]
        boxes_coco = [xyxy_to_coco(b) for b in boxes_expanded]
        image = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        t0 = time.perf_counter()
        inputs = self.processor(
            image,
            boxes=[boxes_coco],
            return_tensors="pt",
        )
        _vitpose_dbg(f"processor done in {(time.perf_counter() - t0) * 1000:.1f}ms (CPU tensors)")
        t0 = time.perf_counter()
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if self.use_fp16:
            inputs = {
                k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in inputs.items()
            }
        _vitpose_dbg(f"inputs moved in {(time.perf_counter() - t0) * 1000:.1f}ms")
        batch_size = len(boxes)
        if "plus" in self.model_name.lower():
            dataset_index = torch.tensor([0] * batch_size, device=self.device, dtype=torch.long)
            model_kwargs = {**inputs, "dataset_index": dataset_index}
        else:
            model_kwargs = inputs
        _vitpose_dbg(f"model forward start | {_summarize_model_inputs(model_kwargs)}")
        t_fwd = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**model_kwargs)
        _mps_synchronize_if_needed(self.device)
        _vitpose_dbg(
            f"model forward end in {(time.perf_counter() - t_fwd) * 1000:.1f}ms (incl. MPS sync if mps)"
        )
        if self.use_fp16:
            outputs = _to_float32(outputs)
        boxes_coco_arr = np.array(boxes_coco, dtype=np.float32)
        t_post = time.perf_counter()
        pose_results = self.processor.post_process_pose_estimation(
            outputs,
            boxes=[boxes_coco_arr],
        )
        _vitpose_dbg(f"post_process_pose_estimation in {(time.perf_counter() - t_post) * 1000:.1f}ms")
        image_pose_result = pose_results[0]
        result: Dict[int, Dict] = {}
        for i, pose_dict in enumerate(image_pose_result):
            tid = track_ids[i] if i < len(track_ids) else i
            kpts = pose_dict["keypoints"].cpu().numpy()
            scores = pose_dict["scores"].cpu().numpy()
            if kpts.shape[1] == 2:
                full_kpts = np.zeros((17, 3))
                full_kpts[:, :2] = kpts
                full_kpts[:, 2] = scores
            else:
                full_kpts = kpts
            result[int(tid)] = {"keypoints": full_kpts, "scores": scores}
        _vitpose_dbg(f"fullframe_batch done n_out={len(result)}")
        return result

    def estimate_poses(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        track_ids: List[int],
        paddings: List[float] = None,
        segmentation_masks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Dict[int, Dict]:
        """
        Estimate 17 COCO keypoints for each bounding box.

        Args:
            frame: RGB image (H, W, 3), uint8.
            boxes: List of (x1, y1, x2, y2) in xyxy format.
            track_ids: List of track IDs, one per box.
            paddings: Optional list of padding bounds for each box.
            segmentation_masks: Optional, same length as boxes. When set, bool (H_box, W_box)
                aligned to ``box``; non-None entries use mask-gated expanded crops (hybrid SAM).

        Returns:
            Dict mapping track_id to {"keypoints": (17, 3), "scores": (17,)} in global coords.
        """
        if len(boxes) == 0:
            return {}

        assert len(boxes) == len(track_ids), "boxes and track_ids must be 1:1"
        if paddings is None:
            paddings = [BBOX_PADDING] * len(boxes)
        assert len(paddings) == len(boxes), "paddings and boxes must match"

        if segmentation_masks is None:
            segmentation_masks = [None] * len(boxes)
        assert len(segmentation_masks) == len(boxes)

        if all(m is None for m in segmentation_masks):
            return self._estimate_poses_batch(frame, boxes, track_ids, paddings)

        plain_idx = [i for i, m in enumerate(segmentation_masks) if m is None]
        masked_idx = [i for i, m in enumerate(segmentation_masks) if m is not None]
        _vitpose_dbg(
            f"estimate_poses hybrid_masks total={len(boxes)} plain_n={len(plain_idx)} "
            f"masked_n={len(masked_idx)} plain_tids={[int(track_ids[i]) for i in plain_idx]} "
            f"masked_tids={[int(track_ids[i]) for i in masked_idx]}"
        )
        out: Dict[int, Dict] = {}
        if plain_idx:
            _vitpose_dbg("estimate_poses -> fullframe batch for plain (no SAM mask) boxes")
            out.update(
                self._estimate_poses_batch(
                    frame,
                    [boxes[i] for i in plain_idx],
                    [track_ids[i] for i in plain_idx],
                    [paddings[i] for i in plain_idx],
                )
            )

        for mi, i in enumerate(masked_idx):
            tid = int(track_ids[i])
            box = boxes[i]
            pad = paddings[i]
            m = segmentation_masks[i]
            if m is None or m.size == 0:
                _vitpose_dbg(f"masked[{mi}] tid={tid} empty mask -> fullframe single")
                sub = self._estimate_poses_batch(frame, [box], [tid], [pad])
                out.update(sub)
                continue
            crop, ox, oy = _apply_sam_mask_gate_to_expanded_crop(frame, box, pad, m)
            if crop.size == 0:
                _vitpose_dbg(f"masked[{mi}] tid={tid} empty crop after gate -> skip")
                continue
            h, w = crop.shape[:2]
            _vitpose_dbg(
                f"masked[{mi}/{len(masked_idx)}] tid={tid} crop_hw=({h},{w}) oxoy=({ox},{oy}) "
                f"mask_hw={tuple(m.shape)}"
            )
            pil = Image.fromarray(crop)
            boxes_coco = [xyxy_to_coco((0.0, 0.0, float(w), float(h)))]
            t0 = time.perf_counter()
            inputs = self.processor(pil, boxes=[boxes_coco], return_tensors="pt")
            _vitpose_dbg(f"masked tid={tid} processor {((time.perf_counter() - t0) * 1000):.1f}ms")
            t0 = time.perf_counter()
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            if self.use_fp16:
                inputs = {
                    k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                    for k, v in inputs.items()
                }
            _vitpose_dbg(f"masked tid={tid} to_device {((time.perf_counter() - t0) * 1000):.1f}ms")
            if "plus" in self.model_name.lower():
                dataset_index = torch.tensor([0], device=self.device, dtype=torch.long)
                model_kwargs = {**inputs, "dataset_index": dataset_index}
            else:
                model_kwargs = inputs
            _vitpose_dbg(f"masked tid={tid} model forward | {_summarize_model_inputs(model_kwargs)}")
            t_fwd = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(**model_kwargs)
            _mps_synchronize_if_needed(self.device)
            _vitpose_dbg(
                f"masked tid={tid} forward done {((time.perf_counter() - t_fwd) * 1000):.1f}ms"
            )
            if self.use_fp16:
                outputs = _to_float32(outputs)
            boxes_coco_arr = np.array(boxes_coco, dtype=np.float32)
            pose_results = self.processor.post_process_pose_estimation(
                outputs,
                boxes=[boxes_coco_arr],
            )
            pose_dict = pose_results[0][0]
            kpts = pose_dict["keypoints"].cpu().numpy()
            scores = pose_dict["scores"].cpu().numpy()
            if kpts.shape[1] == 2:
                full_kpts = np.zeros((17, 3))
                full_kpts[:, :2] = kpts
                full_kpts[:, 2] = scores
            else:
                full_kpts = kpts.copy()
            full_kpts[:, 0] += float(ox)
            full_kpts[:, 1] += float(oy)
            out[tid] = {"keypoints": full_kpts, "scores": scores}

        return out
