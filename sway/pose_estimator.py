"""
Top-Down Pose Estimation Module — ViTPose (V3.0)

V3.0: ViTPose-Large for high-fidelity pose estimation (15° wrist/elbow accuracy).
Extracts 17 COCO keypoints per detected person. Batched [N, 3, 256, 192], fp16 on MPS.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

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
        if local_only:
            self.processor = AutoProcessor.from_pretrained(
                model_name, local_files_only=True
            )
            self.model = VitPoseForPoseEstimation.from_pretrained(
                model_name, local_files_only=True
            )
        else:
            # Prefer disk cache only first so repeat runs do not hit the Hub for metadata
            # or "is there an update?" checks when weights already live under HF_HOME.
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_name, local_files_only=True
                )
                self.model = VitPoseForPoseEstimation.from_pretrained(
                    model_name, local_files_only=True
                )
            except Exception:
                print(
                    f"  ViTPose: local Hugging Face cache incomplete for {model_name}; "
                    "downloading / resolving from the Hub (needs network)…",
                    flush=True,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_name, local_files_only=False
                )
                self.model = VitPoseForPoseEstimation.from_pretrained(
                    model_name, local_files_only=False
                )
        self.model.to(self.device)
        self.model.eval()
        # fp16 on MPS/CUDA (M2 optimization): ~2x speed, ~half memory
        use_fp16 = self.device.type in ("mps", "cuda")
        self.use_fp16 = use_fp16
        if use_fp16:
            self.model = self.model.to(torch.float16)

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
        img_h, img_w = frame.shape[:2]
        boxes_expanded = [
            _expand_bbox(b, img_w, img_h, pad) for b, pad in zip(boxes, paddings)
        ]
        boxes_coco = [xyxy_to_coco(b) for b in boxes_expanded]
        image = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        inputs = self.processor(
            image,
            boxes=[boxes_coco],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if self.use_fp16:
            inputs = {
                k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in inputs.items()
            }
        batch_size = len(boxes)
        if "plus" in self.model_name.lower():
            dataset_index = torch.tensor([0] * batch_size, device=self.device, dtype=torch.long)
            model_kwargs = {**inputs, "dataset_index": dataset_index}
        else:
            model_kwargs = inputs
        with torch.no_grad():
            outputs = self.model(**model_kwargs)
        if self.use_fp16:
            outputs = _to_float32(outputs)
        boxes_coco_arr = np.array(boxes_coco, dtype=np.float32)
        pose_results = self.processor.post_process_pose_estimation(
            outputs,
            boxes=[boxes_coco_arr],
        )
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
        out: Dict[int, Dict] = {}
        if plain_idx:
            out.update(
                self._estimate_poses_batch(
                    frame,
                    [boxes[i] for i in plain_idx],
                    [track_ids[i] for i in plain_idx],
                    [paddings[i] for i in plain_idx],
                )
            )

        for i in masked_idx:
            tid = int(track_ids[i])
            box = boxes[i]
            pad = paddings[i]
            m = segmentation_masks[i]
            if m is None or m.size == 0:
                sub = self._estimate_poses_batch(frame, [box], [tid], [pad])
                out.update(sub)
                continue
            crop, ox, oy = _apply_sam_mask_gate_to_expanded_crop(frame, box, pad, m)
            if crop.size == 0:
                continue
            h, w = crop.shape[:2]
            pil = Image.fromarray(crop)
            boxes_coco = [xyxy_to_coco((0.0, 0.0, float(w), float(h)))]
            inputs = self.processor(pil, boxes=[boxes_coco], return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            if self.use_fp16:
                inputs = {
                    k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                    for k, v in inputs.items()
                }
            if "plus" in self.model_name.lower():
                dataset_index = torch.tensor([0], device=self.device, dtype=torch.long)
                model_kwargs = {**inputs, "dataset_index": dataset_index}
            else:
                model_kwargs = inputs
            with torch.no_grad():
                outputs = self.model(**model_kwargs)
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
