"""Unit tests for Sapiens TorchScript heatmap decode (no checkpoint required)."""

import numpy as np

from sway.sapiens_estimator import heatmaps_to_keypoints_xyxy


def test_heatmaps_to_keypoints_xyxy_peak_maps_to_bbox():
    hm = np.zeros((17, 256, 192), dtype=np.float32)
    hm[3, 128, 96] = 5.0
    x1, y1, x2, y2 = 10.0, 20.0, 210.0, 420.0
    k = heatmaps_to_keypoints_xyxy(hm, x1, y1, x2, y2, 256, 192)
    assert k.shape == (17, 3)
    bw, bh = 200.0, 400.0
    assert abs(k[3, 0] - (96 * bw / 192 + x1)) < 1e-4
    assert abs(k[3, 1] - (128 * bh / 256 + y1)) < 1e-4
    assert k[3, 2] == 5.0
    assert np.allclose(k[:3, 2], 0.0)
