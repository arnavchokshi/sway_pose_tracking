# Appearance-Free Link (AFLink) from StrongSORT — vendored for SWAY global ID stitching.
# Source: https://github.com/dyhBUPT/StrongSORT (GPL-3.0)
# Original author: Du Yunhao et al.

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .torch_compat import torch_load_trusted

INFINITY = 1e5

# Training defaults from StrongSORT AFLink/config.py
MODEL_MIN_LEN = 30
MODEL_INPUT_LEN = 30


class TemporalBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (7, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bnf = nn.BatchNorm1d(cout)
        self.bnx = nn.BatchNorm1d(cout)
        self.bny = nn.BatchNorm1d(cout)

    def bn(self, x):
        x[:, :, :, 0] = self.bnf(x[:, :, :, 0])
        x[:, :, :, 1] = self.bnx(x[:, :, :, 1])
        x[:, :, :, 2] = self.bny(x[:, :, :, 2])
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FusionBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (1, 3), bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Classifier(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.fc1 = nn.Linear(cin * 2, cin // 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(cin // 2, 2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PostLinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.TemporalModule_1 = nn.Sequential(
            TemporalBlock(1, 32),
            TemporalBlock(32, 64),
            TemporalBlock(64, 128),
            TemporalBlock(128, 256),
        )
        self.TemporalModule_2 = nn.Sequential(
            TemporalBlock(1, 32),
            TemporalBlock(32, 64),
            TemporalBlock(64, 128),
            TemporalBlock(128, 256),
        )
        self.FusionBlock_1 = FusionBlock(256, 256)
        self.FusionBlock_2 = FusionBlock(256, 256)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(256)

    def forward(self, x1, x2):
        x1 = x1[:, :, :, :3]
        x2 = x2[:, :, :, :3]
        x1 = self.TemporalModule_1(x1)
        x2 = self.TemporalModule_2(x2)
        x1 = self.FusionBlock_1(x1)
        x2 = self.FusionBlock_2(x2)
        x1 = self.pooling(x1).squeeze(-1).squeeze(-1)
        x2 = self.pooling(x2).squeeze(-1).squeeze(-1)
        y = self.classifier(x1, x2)
        if not self.training:
            y = torch.softmax(y, dim=1)
        return y


class _LinkTransform:
    """Inference-only subset of StrongSORT AFLink LinkData (transform / fill_or_cut)."""

    def __init__(self, min_len: int = MODEL_MIN_LEN, input_len: int = MODEL_INPUT_LEN):
        self.minLen = min_len
        self.inputLen = input_len

    def fill_or_cut(self, x: np.ndarray, former: bool) -> np.ndarray:
        length_x, width_x = x.shape
        if length_x >= self.inputLen:
            if former:
                x = x[-self.inputLen :]
            else:
                x = x[: self.inputLen]
        else:
            zeros = np.zeros((self.inputLen - length_x, width_x))
            if former:
                x = np.concatenate((zeros, x), axis=0)
            else:
                x = np.concatenate((x, zeros), axis=0)
        return x

    def transform(self, x1: np.ndarray, x2: np.ndarray):
        x1 = self.fill_or_cut(x1, True)
        x2 = self.fill_or_cut(x2, False)
        min_ = np.concatenate((x1, x2), axis=0).min(axis=0)
        max_ = np.concatenate((x1, x2), axis=0).max(axis=0)
        subtractor = (max_ + min_) / 2
        divisor = (max_ - min_) / 2 + 1e-5
        x1 = (x1 - subtractor) / divisor
        x2 = (x2 - subtractor) / divisor
        x1_t = torch.tensor(x1, dtype=torch.float32)
        x2_t = torch.tensor(x2, dtype=torch.float32)
        x1_t = x1_t.unsqueeze(dim=0)
        x2_t = x2_t.unsqueeze(dim=0)
        return x1_t, x2_t


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AFLink:
    """
    Global appearance-free tracklet association (StrongSORT AFLink).
    ``link()`` expects MOT-style rows: frame, id, x, y, w, h, conf, ... (≥6 columns used).
    Positions are top-left x,y plus width/height, matching MOTChallenge detections.
    """

    def __init__(
        self,
        path_AFLink: str,
        thrT: Tuple[int, int] = (0, 30),
        thrS: int = 75,
        thrP: float = 0.05,
        device: Optional[torch.device] = None,
    ):
        self.thrP = thrP
        self.thrT = thrT
        self.thrS = thrS
        self.device = device or _default_device()
        self.model = PostLinker()
        state = torch_load_trusted(path_AFLink, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.dataset = _LinkTransform()

    def gather_info(self, track: np.ndarray):
        id2info = defaultdict(list)
        t = track[np.argsort(track[:, 0])]
        for row in t:
            f, i, x, y, w, h = row[:6]
            id2info[int(i)].append([float(f), float(x), float(y), float(w), float(h)])
        id2info = {k: np.array(v, dtype=np.float64) for k, v in id2info.items()}
        return id2info, t

    def compression(self, cost_matrix: np.ndarray, ids: np.ndarray):
        mask_row = cost_matrix.min(axis=1) < self.thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]
        mask_col = cost_matrix.min(axis=0) < self.thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]
        return matrix, ids_row, ids_col

    def predict(self, track1: np.ndarray, track2: np.ndarray) -> float:
        t1, t2 = self.dataset.transform(track1, track2)
        t1 = t1.unsqueeze(0).to(self.device)
        t2 = t2.unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(t1, t2)[0, 1].detach().cpu().numpy()
        return float(1 - score)

    @staticmethod
    def deduplicate(tracks: np.ndarray) -> np.ndarray:
        _, index = np.unique(tracks[:, :2], return_index=True, axis=0)
        return tracks[index]

    def link(self, track: np.ndarray) -> np.ndarray:
        """
        Run AFLink on an (N, K) array (K ≥ 7 when using row-index passthrough in col 7).
        Returns array of same width with global IDs merged in column 1.
        """
        self.track = np.asarray(track, dtype=np.float64)
        if self.track.size == 0:
            return self.track
        if self.track.ndim == 1:
            self.track = self.track.reshape(1, -1)
        id2info, sorted_track = self.gather_info(self.track)
        num = len(id2info)
        if num < 2:
            return sorted_track
        ids = np.array(list(id2info), dtype=np.int64)
        fn_l2 = lambda x, y: float(np.sqrt(x**2 + y**2))
        cost_matrix = np.ones((num, num)) * INFINITY
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if id_i == id_j:
                    continue
                info_i, info_j = id2info[id_i], id2info[id_j]
                fi, bi = info_i[-1][0], info_i[-1][1:3]
                fj, bj = info_j[0][0], info_j[0][1:3]
                if not self.thrT[0] <= fj - fi < self.thrT[1]:
                    continue
                if self.thrS < fn_l2(bi[0] - bj[0], bi[1] - bj[1]):
                    continue
                cost = self.predict(info_i, info_j)
                if cost <= self.thrP:
                    cost_matrix[i, j] = cost
        id2id: dict = {}
        ID2ID: dict = {}
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids)
        if cost_matrix.size == 0 or cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
            return sorted_track
        indices = linear_sum_assignment(cost_matrix)
        for ii, jj in zip(indices[0], indices[1]):
            if cost_matrix[ii, jj] < self.thrP:
                id2id[int(ids_row[ii])] = int(ids_col[jj])
        for k, v in id2id.items():
            if k in ID2ID:
                ID2ID[v] = ID2ID[k]
            else:
                ID2ID[v] = k
        res = sorted_track.copy()
        for k, v in ID2ID.items():
            res[res[:, 1] == k, 1] = v
        res = self.deduplicate(res)
        return res
