"""
Graph neural track refinement (post Phase-3 stitch).

Builds a **track graph** (nodes = trajectories, edges = spatio-temporal candidates),
runs **edge-conditioned multi-head GAT** layers with residuals / LayerNorm, then
**link prediction** to propose merges. A fixed **structural prior** on edge logits
keeps behavior sane without training; by default a checkpoint at
``models/gnn_track_refine.pt`` (if present) or ``SWAY_GNN_WEIGHTS`` supplies
learned ``state_dict`` weights.

Env (optional):
  SWAY_GNN_MERGE_THRESH   sigmoid threshold for merging (default 0.55)
  SWAY_GNN_HIDDEN         hidden dim (default 128)
  SWAY_GNN_HEADS          attention heads per layer (default 8)
  SWAY_GNN_LAYERS         GAT depth (default 4)
  SWAY_GNN_DROPOUT        dropout prob (default 0.1)
  SWAY_GNN_MAX_GAP        max frame gap for candidate edges when disjoint (default 120)
  SWAY_GNN_PRIOR_SCALE    weight of hand-crafted prior logits (default 1.0)
  SWAY_GNN_WEIGHTS        path to .pt checkpoint (optional). If unset, loads
                          ``<repo>/models/gnn_track_refine.pt`` when that file exists.
  SWAY_GNN_DEVICE         cuda / cpu / auto (default auto)
  SWAY_GNN_SEED           if set, ``torch.manual_seed`` before module init (reproducible logits)

Train (synthetic BCE on edges): ``python -m tools.train_gnn_track_refine --out models/gnn_track_refine.pt``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sway.track_observation import TrackObservation, coerce_observation, iou_xyxy_np


# --- geometry helpers -----------------------------------------------------------------


def _box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _bbox_height(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[3] - box[1]))


def _bbox_wh(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return max(0.0, float(box[2] - box[0])), _bbox_height(box)


def _entries_sorted(raw: List[Any]) -> List[TrackObservation]:
    obs = [coerce_observation(e) for e in raw]
    return sorted(obs, key=lambda o: o.frame_idx)


# --- track statistics for features ----------------------------------------------------


@dataclass
class TrackStats:
    tid: int
    frames: np.ndarray  # (T,)
    boxes: np.ndarray  # (T, 4)
    confs: np.ndarray  # (T,)


def _compute_track_stats(tid: int, raw: List[Any]) -> Optional[TrackStats]:
    if not raw:
        return None
    ent = _entries_sorted(raw)
    frames = np.array([e.frame_idx for e in ent], dtype=np.int64)
    boxes = np.array([list(e.bbox) for e in ent], dtype=np.float32)
    confs = np.array([e.conf for e in ent], dtype=np.float32)
    return TrackStats(tid=tid, frames=frames, boxes=boxes, confs=confs)


def _node_feature_vector(ts: TrackStats, total_frames: int) -> np.ndarray:
    """Fixed-length raw node descriptor (before learned embedding)."""
    T = len(ts.frames)
    tf = max(int(total_frames), 1)
    x1, y1, x2, y2 = ts.boxes.T
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = np.clip(x2 - x1, 1e-3, None)
    h = np.clip(y2 - y1, 1e-3, None)
    scale = float(np.median(h))
    scale = max(scale, 1.0)

    first_f = float(ts.frames[0]) / tf
    last_f = float(ts.frames[-1]) / tf
    span = float(ts.frames[-1] - ts.frames[0] + 1) / tf
    log_len = np.log1p(T)

    vcx = np.diff(cx) if T > 1 else np.array([0.0], dtype=np.float32)
    vcy = np.diff(cy) if T > 1 else np.array([0.0], dtype=np.float32)
    speed = np.sqrt(vcx * vcx + vcy * vcy) / scale if len(vcx) else np.array([0.0])

    feat = np.array(
        [
            first_f,
            last_f,
            span,
            log_len / 6.0,
            float(np.mean(cx)) / (scale * 20.0),
            float(np.mean(cy)) / (scale * 20.0),
            float(np.std(cx)) / scale if T > 1 else 0.0,
            float(np.std(cy)) / scale if T > 1 else 0.0,
            float(np.mean(w)) / scale,
            float(np.mean(h)) / scale,
            float(np.std(h)) / scale if T > 1 else 0.0,
            float(np.mean(ts.confs)),
            float(np.std(ts.confs)) if T > 1 else 0.0,
            float(np.mean(speed)) if len(speed) else 0.0,
            float(np.std(speed)) if len(speed) > 1 else 0.0,
            float(np.median(h) / np.median(w)) if np.median(w) > 1e-3 else 1.0,
        ],
        dtype=np.float32,
    )
    return feat


def _pair_edge_stats(
    a: TrackStats,
    b: TrackStats,
    total_frames: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Scalar pairwise stats for priors + edge MLP input.
    Returns (edge_feat_np, debug_scalars).
    """
    tf = max(int(total_frames), 1)
    fa0, fa1 = int(a.frames[0]), int(a.frames[-1])
    fb0, fb1 = int(b.frames[0]), int(b.frames[-1])

    # Temporal: negative if overlap (how many overlapping frames)
    overlap_frames = set(a.frames.tolist()) & set(b.frames.tolist())
    n_overlap = len(overlap_frames)

    if n_overlap > 0:
        min_gap = 0.0
        max_iou = 0.0
        mean_dist_h = 0.0
        dist_count = 0
        for fi in overlap_frames:
            ia = int(np.where(a.frames == fi)[0][0])
            ib = int(np.where(b.frames == fi)[0][0])
            ba = a.boxes[ia]
            bb = b.boxes[ib]
            max_iou = max(max_iou, iou_xyxy_np(ba, bb))
            ca = np.array(_box_center(tuple(ba.tolist())))
            cb = np.array(_box_center(tuple(bb.tolist())))
            h = max(_bbox_height(tuple(ba.tolist())), _bbox_height(tuple(bb.tolist())), 1.0)
            mean_dist_h += float(np.linalg.norm(ca - cb) / h)
            dist_count += 1
        mean_dist_h /= max(dist_count, 1)
        disjoint = 0.0
    else:
        # min gap between segment endpoints
        if fa1 < fb0:
            min_gap = float(fb0 - fa1)
        elif fb1 < fa0:
            min_gap = float(fa0 - fb1)
        else:
            min_gap = 0.0
        max_iou = 0.0
        # nearest approach between bbox centers (any pair) scaled
        ca = np.array([_box_center(tuple(x)) for x in a.boxes])
        cb = np.array([_box_center(tuple(x)) for x in b.boxes])
        # min pairwise center dist / median height
        ha = float(np.median([_bbox_height(tuple(x)) for x in a.boxes]) or 1.0)
        hb = float(np.median([_bbox_height(tuple(x)) for x in b.boxes]) or 1.0)
        h_scale = max(ha, hb, 1.0)
        dmat = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
        mean_dist_h = float(np.min(dmat) / h_scale)
        disjoint = 1.0

    len_a, len_b = len(a.frames), len(b.frames)
    len_ratio = min(len_a, len_b) / max(len_a, len_b, 1)

    dbg = {
        "n_overlap": float(n_overlap),
        "min_gap": float(min_gap),
        "max_iou": float(max_iou),
        "mean_dist_h": float(mean_dist_h),
        "disjoint": float(disjoint),
        "len_ratio": float(len_ratio),
    }

    edge = np.array(
        [
            min_gap / float(tf),
            max_iou,
            mean_dist_h / 5.0,
            disjoint,
            len_ratio,
            float(n_overlap) / float(max(min(len_a, len_b), 1)),
        ],
        dtype=np.float32,
    )
    return edge, dbg


def _prior_logit_from_edge(edge_np: np.ndarray) -> float:
    """Hand-tuned structural prior (same-person vs not) in logit space."""
    min_gap_n, max_iou, mean_dist_h, disjoint, len_ratio, overlap_frac = [float(x) for x in edge_np]
    # High IoU overlap -> same person; large center distance -> not
    z = (
        4.2 * max_iou
        - 2.8 * mean_dist_h
        - 1.6 * min_gap_n * 8.0
        + 0.9 * overlap_frac
        + 0.35 * len_ratio
        - 0.4 * disjoint * (1.0 - max_iou)
    )
    return float(z)


def default_gnn_weights_path() -> Path:
    """Bundled checkpoint next to repo root (``sway_pose_mvp/models/``)."""
    return Path(__file__).resolve().parent.parent / "models" / "gnn_track_refine.pt"


def _load_compatible_state_dict(model: nn.Module, sd: Dict[str, Any]) -> int:
    """Apply only parameters whose shapes match (so env can change hidden/heads/layers)."""
    msd = model.state_dict()
    filtered = {
        k: v
        for k, v in sd.items()
        if k in msd and isinstance(v, torch.Tensor) and msd[k].shape == v.shape
    }
    model.load_state_dict(filtered, strict=False)
    return len(filtered)


def _load_gnn_checkpoint(model: nn.Module, device: torch.device) -> None:
    """Load ``state_dict`` from ``SWAY_GNN_WEIGHTS`` or default ``models/gnn_track_refine.pt``."""
    wpath = os.environ.get("SWAY_GNN_WEIGHTS", "").strip()
    if wpath:
        p = Path(wpath).expanduser()
        if p.is_file():
            payload = torch.load(p, map_location=device)
            sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            if not isinstance(sd, dict):
                print(f"  GNN weights: invalid checkpoint {p}", flush=True)
                return
            n = _load_compatible_state_dict(model, sd)
            if n == 0:
                print(f"  GNN weights: {p} (no matching layers for this arch; random init)", flush=True)
            else:
                print(f"  GNN weights: {p} ({n} tensors)", flush=True)
        else:
            print(f"  GNN weights: SWAY_GNN_WEIGHTS file not found ({wpath}), using random init", flush=True)
        return
    p = default_gnn_weights_path()
    if p.is_file():
        payload = torch.load(p, map_location=device)
        sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        if isinstance(sd, dict):
            n = _load_compatible_state_dict(model, sd)
            if n == 0:
                print(f"  GNN weights: {p} (no matching layers for this arch; random init)", flush=True)
            else:
                print(f"  GNN weights: {p} ({n} tensors)", flush=True)


def _hard_forbid_merge(dbg: Dict[str, float]) -> bool:
    """Reject merges that are almost certainly two people co-occurring."""
    if dbg["n_overlap"] >= 3 and dbg["max_iou"] < 0.08 and dbg["mean_dist_h"] > 0.35:
        return True
    if dbg["n_overlap"] >= 8 and dbg["max_iou"] < 0.15:
        return True
    return False


# --- neural modules --------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseEdgeGATLayer(nn.Module):
    """
    Dense multi-head attention: for each target node i, aggregate messages from j
    with attention logits ``(q_i · k_j) / sqrt(d) + a(edge_ij)`` and values
    ``v_j + U(edge_ij)`` (edge-conditioned values).
    """

    def __init__(self, dim: int, n_heads: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.dk = dim // n_heads
        self.dropout = nn.Dropout(dropout)
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.W_att_e = nn.Linear(edge_dim, n_heads)
        self.W_val_e = nn.Linear(edge_dim, dim)
        self.Wo = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (N, dim)
        edge_feat: (N, N, edge_dim)
        mask: (N, N) bool, True = valid edge (including self-loop)
        """
        n = x.size(0)
        h = self.ln(x)
        q = self.Wq(h).view(n, self.n_heads, self.dk)
        k = self.Wk(h).view(n, self.n_heads, self.dk)
        v = self.Wv(h).view(n, self.n_heads, self.dk)
        # Attention: (N, N, H) = sum_d q[i,h,d]*k[j,h,d] / sqrt(dk)
        att = torch.einsum("nhd,mhd->nmh", q, k) / (self.dk ** 0.5)
        att = att + self.W_att_e(edge_feat)
        att = att.masked_fill(~mask.unsqueeze(-1), -1e4)
        alpha = F.softmax(att, dim=1)
        alpha = self.dropout(alpha)
        # Values from source j, edge-conditioned: msg[i,j] uses v[j]
        v_b = v.unsqueeze(0).expand(n, -1, -1, -1)
        edge_val = self.W_val_e(edge_feat).view(n, n, self.n_heads, self.dk)
        msg = v_b + edge_val
        out = torch.einsum("nmh,nmhd->nhd", alpha, msg).reshape(n, self.dim)
        out = self.Wo(out)
        return x + out


class RelationalTrackGNN(nn.Module):
    """Stack of edge-conditioned GAT + link MLP."""

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_in = node_in
        self.edge_in = edge_in
        self.node_enc = MLP(node_in, hidden, hidden, dropout)
        self.edge_enc = MLP(edge_in, hidden, edge_in, dropout)
        self.layers = nn.ModuleList(
            [DenseEdgeGATLayer(hidden, n_heads, edge_in, dropout) for _ in range(n_layers)]
        )
        self.link = MLP(hidden * 2 + edge_in, hidden, 1, dropout)

    def forward(
        self,
        x0: torch.Tensor,
        edge_feat: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns pairwise logits (N, N) symmetric for i<j usage.
        """
        x = self.node_enc(x0)
        ef0 = self.edge_enc(edge_feat)
        mask = adj > 0.5
        eye = torch.eye(x.size(0), device=x.device, dtype=torch.bool)
        mask = mask | eye
        for layer in self.layers:
            x = layer(x, ef0, mask)
        n = x.size(0)
        xi = x.unsqueeze(1).expand(n, n, -1)
        xj = x.unsqueeze(0).expand(n, n, -1)
        cat = torch.cat([xi, xj, ef0], dim=-1)
        logits = self.link(cat).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        # Symmetrize
        logits = 0.5 * (logits + logits.transpose(0, 1))
        return logits


# --- merge -----------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


@dataclass
class TrackGraphTensors:
    """Dense track graph in tensor form (matches inference)."""

    x0: torch.Tensor
    edge_feat: torch.Tensor
    adj: torch.Tensor
    prior_mat: torch.Tensor
    node_ids: List[int]
    dbg_grid: List[List[Optional[Dict[str, float]]]]
    stats_map: Dict[int, TrackStats]


def build_track_graph_tensors(
    raw_tracks: Dict[int, List[Any]],
    total_frames: int,
    *,
    max_gap: float,
    prior_scale: float,
    device: torch.device,
) -> Optional[TrackGraphTensors]:
    """
    Build node/edge tensors and adjacency from stitched-style ``raw_tracks``.
    Returns ``None`` if fewer than two non-empty trajectories.
    """
    stats_map: Dict[int, TrackStats] = {}
    node_ids: List[int] = []
    for tid, seq in raw_tracks.items():
        st = _compute_track_stats(int(tid), seq)
        if st is not None:
            stats_map[int(tid)] = st
            node_ids.append(int(tid))

    if len(node_ids) < 2:
        return None

    node_ids.sort()
    n = len(node_ids)
    node_feats = np.stack(
        [_node_feature_vector(stats_map[t], total_frames) for t in node_ids],
        axis=0,
    )
    edge_in = 6

    edge_tensor = torch.zeros(n, n, edge_in, device=device)
    adj = torch.zeros(n, n, device=device)
    prior_mat = torch.zeros(n, n, device=device)
    dbg_grid: List[List[Optional[Dict[str, float]]]] = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        adj[i, i] = 1.0
        for j in range(i + 1, n):
            ti, tj = node_ids[i], node_ids[j]
            ea, eb = stats_map[ti], stats_map[tj]
            fe, dbg = _pair_edge_stats(ea, eb, total_frames)
            gap_ok = dbg["n_overlap"] > 0 or dbg["min_gap"] <= max_gap
            if not gap_ok:
                continue
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            te = torch.tensor(fe, device=device, dtype=torch.float32)
            edge_tensor[i, j] = te
            edge_tensor[j, i] = te
            pl = prior_scale * _prior_logit_from_edge(fe)
            prior_mat[i, j] = pl
            prior_mat[j, i] = pl
            dbg_grid[i][j] = dbg
            dbg_grid[j][i] = dbg

    x0 = torch.tensor(node_feats, device=device, dtype=torch.float32)
    return TrackGraphTensors(
        x0=x0,
        edge_feat=edge_tensor,
        adj=adj,
        prior_mat=prior_mat,
        node_ids=node_ids,
        dbg_grid=dbg_grid,
        stats_map=stats_map,
    )


def _merge_component(
    raw_tracks: Dict[int, List[Any]],
    tids: List[int],
) -> None:
    """Merge all tracks in tids into the longest one; delete others."""
    if len(tids) < 2:
        return
    best = max(tids, key=lambda t: len(raw_tracks.get(t) or []))
    keep_entries: Dict[int, Any] = {}
    for e in raw_tracks[best]:
        o = coerce_observation(e)
        keep_entries[o.frame_idx] = e
    for tid in tids:
        if tid == best:
            continue
        for e in raw_tracks.get(tid, []):
            o = coerce_observation(e)
            if o.frame_idx not in keep_entries:
                keep_entries[o.frame_idx] = e
    merged_list = sorted(keep_entries.values(), key=lambda e: coerce_observation(e).frame_idx)
    raw_tracks[best] = merged_list
    for tid in tids:
        if tid != best:
            raw_tracks.pop(tid, None)


def gnn_refine_raw_tracks(
    raw_tracks: Dict[int, List[Any]],
    total_frames: int,
    ystride: int,
) -> Dict[int, List[Any]]:
    """
    Apply relational GNN merge proposal on top of stitched tracks.
    Mutates and returns ``raw_tracks``.
    """
    _ = ystride  # reserved for future stride-aware edge gating
    if len(raw_tracks) < 2:
        return raw_tracks

    thresh = float(os.environ.get("SWAY_GNN_MERGE_THRESH", "0.55"))
    hidden = int(os.environ.get("SWAY_GNN_HIDDEN", "128"))
    n_heads = int(os.environ.get("SWAY_GNN_HEADS", "8"))
    n_layers = int(os.environ.get("SWAY_GNN_LAYERS", "4"))
    dropout = float(os.environ.get("SWAY_GNN_DROPOUT", "0.1"))
    max_gap = float(os.environ.get("SWAY_GNN_MAX_GAP", "120"))
    prior_scale = float(os.environ.get("SWAY_GNN_PRIOR_SCALE", "1.0"))
    dev_s = os.environ.get("SWAY_GNN_DEVICE", "auto").strip().lower()
    if dev_s == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif dev_s == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = build_track_graph_tensors(
        raw_tracks,
        total_frames,
        max_gap=max_gap,
        prior_scale=prior_scale,
        device=device,
    )
    if graph is None:
        return raw_tracks

    x0 = graph.x0
    edge_tensor = graph.edge_feat
    adj = graph.adj
    prior_mat = graph.prior_mat
    node_ids = graph.node_ids
    dbg_grid = graph.dbg_grid
    n = len(node_ids)
    node_in = x0.shape[1]
    edge_in = 6
    model = RelationalTrackGNN(
        node_in=node_in,
        edge_in=edge_in,
        hidden=hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)

    seed_s = os.environ.get("SWAY_GNN_SEED", "").strip()
    if seed_s:
        try:
            torch.manual_seed(int(seed_s))
        except ValueError:
            pass

    _load_gnn_checkpoint(model, device)

    model.eval()
    with torch.no_grad():
        learn_logits = model(x0, edge_tensor, adj)
        logits = learn_logits + prior_mat
        probs = torch.sigmoid(logits)

    uf = UnionFind(n)
    merges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] < 0.5:
                continue
            dbg = dbg_grid[i][j]
            if dbg and _hard_forbid_merge(dbg):
                continue
            p_ij = float(probs[i, j].item())
            if p_ij >= thresh:
                uf.union(i, j)
                merges += 1

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(node_ids[i])

    merged_groups = 0
    for comp in groups.values():
        if len(comp) > 1:
            _merge_component(raw_tracks, comp)
            merged_groups += 1

    off_diag = int((adj.sum().item() - n) / 2)
    print(
        f"  GNN track refine: {n} nodes, {off_diag} undirected candidate edges, "
        f"{merges} pair scores ≥ {thresh}, merged {merged_groups} component(s). "
        f"(layers={n_layers}, heads={n_heads}, hidden={hidden}, device={device})",
        flush=True,
    )
    return raw_tracks
