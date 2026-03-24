"""Unit tests for hybrid SAM ROI helpers (no Ultralytics / SAM required)."""

import numpy as np

from sway.hybrid_sam_refiner import (
    max_pairwise_iou,
    overlap_cluster_indices,
    union_xyxy_with_pad,
)


def test_max_pairwise_iou_identical_boxes():
    xy = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    assert max_pairwise_iou(xy) == 1.0


def test_overlap_cluster_indices():
    # Two heavily overlapping + one far away (IoU(0,1) > 0.42)
    xy = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [2.0, 2.0, 12.0, 12.0],
            [100.0, 100.0, 120.0, 140.0],
        ],
        dtype=np.float32,
    )
    idx = overlap_cluster_indices(xy, 0.42)
    assert idx == {0, 1}


def test_union_xyxy_with_pad():
    xy = np.array(
        [
            [10.0, 10.0, 20.0, 20.0],
            [15.0, 15.0, 25.0, 25.0],
        ],
        dtype=np.float32,
    )
    x1, y1, x2, y2 = union_xyxy_with_pad(xy, [0, 1], frame_h=200, frame_w=300, pad_frac=0.1)
    assert x1 >= 0 and y1 >= 0 and x2 <= 300 and y2 <= 200
    assert x2 > x1 and y2 > y1
    # Union without pad: (10,10)-(25,25); with 10% pad on ~15px size -> ~1.5px margin
    assert x1 < 10 and y1 < 10
    assert x2 > 25 and y2 > 25
