"""
Top-Down Pose Estimation Module — ViTPose (V3.0)

V3.0: ViTPose-Large for high-fidelity pose estimation (15° wrist/elbow accuracy).
Extracts 17 COCO keypoints per detected person. Batched [N, 3, 256, 192], fp16 on MPS.
"""

from typing import Dict, List, Tuple, Union

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
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # fp16 on MPS/CUDA (M2 optimization): ~2x speed, ~half memory
        use_fp16 = self.device.type in ("mps", "cuda")
        self.use_fp16 = use_fp16
        if use_fp16:
            self.model = self.model.to(torch.float16)

    def estimate_poses(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        track_ids: List[int],
        paddings: List[float] = None,
    ) -> Dict[int, Dict]:
        """
        Estimate 17 COCO keypoints for each bounding box.

        Args:
            frame: RGB image (H, W, 3), uint8.
            boxes: List of (x1, y1, x2, y2) in xyxy format.
            track_ids: List of track IDs, one per box.
            paddings: Optional list of padding bounds for each box.

        Returns:
            Dict mapping track_id to {"keypoints": (17, 3), "scores": (17,)}.
            Each keypoint is (x, y, score). Coordinates are in global frame.
        """
        if len(boxes) == 0:
            return {}

        # Batching: ALL persons in this frame processed in ONE forward pass.
        # Do NOT loop over dancers — stacking into [N, 3, 256, 192] maximizes MPS/CUDA parallelism.
        assert len(boxes) == len(track_ids), "boxes and track_ids must be 1:1"
        img_h, img_w = frame.shape[:2]
        
        if paddings is None:
            paddings = [BBOX_PADDING] * len(boxes)
        assert len(paddings) == len(boxes), "paddings and boxes must match"

        # Expand bboxes by dynamic padding to avoid cropping off limbs
        boxes_expanded = [
            _expand_bbox(b, img_w, img_h, pad) for b, pad in zip(boxes, paddings)
        ]
        # Convert boxes to COCO format (x, y, w, h). Format: list of boxes per image for batching.
        boxes_coco = [xyxy_to_coco(b) for b in boxes_expanded]

        # ViTPose expects PIL or numpy; use numpy directly
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame

        # Batched preprocessing: processor crops, resizes, normalizes all boxes → single tensor
        # Stack shape: [N_dancers, 3, 256, 192] — single forward pass for entire frame
        inputs = self.processor(
            image,
            boxes=[boxes_coco],  # One image, N boxes → batched [N, 3, 256, 192]
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if self.use_fp16:
            inputs = {
                k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in inputs.items()
            }

        # Single batched forward pass: no per-dancer loop — all N cropped person images
        # in one model(**inputs) call (MPS/CUDA parallelizes across batch dimension)
        # ViTPose++ requires dataset_index for COCO (0)
        batch_size = len(boxes)
        if "plus" in self.model_name.lower():
            dataset_index = torch.tensor([0] * batch_size, device=self.device, dtype=torch.long)
            model_kwargs = {**inputs, "dataset_index": dataset_index}
        else:
            model_kwargs = inputs
        with torch.no_grad():
            outputs = self.model(**model_kwargs)

        # Post-process: scipy gaussian_filter does not support float16; convert outputs to float32
        if self.use_fp16:
            outputs = _to_float32(outputs)

        # Post-process: map outputs back to track_ids in global frame coordinates
        boxes_coco_arr = np.array(boxes_coco, dtype=np.float32)
        pose_results = self.processor.post_process_pose_estimation(
            outputs,
            boxes=[boxes_coco_arr],
        )

        # pose_results: list of lists (one list per image, each element = one person)
        image_pose_result = pose_results[0]
        result = {}

        for i, pose_dict in enumerate(image_pose_result):
            tid = track_ids[i] if i < len(track_ids) else i
            kpts = pose_dict["keypoints"].cpu().numpy()
            scores = pose_dict["scores"].cpu().numpy()
            # kpts shape: (17, 3) — x, y, score (or 17, 2 for coords)
            if kpts.shape[1] == 2:
                # Some outputs may only have x,y; add score column
                full_kpts = np.zeros((17, 3))
                full_kpts[:, :2] = kpts
                full_kpts[:, 2] = scores
            else:
                full_kpts = kpts
            result[int(tid)] = {"keypoints": full_kpts, "scores": scores}

        return result
