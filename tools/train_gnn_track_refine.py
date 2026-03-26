#!/usr/bin/env python3
"""
Quick synthetic training for ``RelationalTrackGNN`` (same tensors as inference).

Trains edge merge logits with BCE on ``learn_logits + prior`` vs. ground-truth
same-person labels on candidate edges. Saves a checkpoint loadable via
``SWAY_GNN_WEIGHTS`` (dict with ``state_dict`` key).

  python -m tools.train_gnn_track_refine --out models/gnn_track_refine.pt --steps 400

Defaults match inference env (hidden 128, 8 heads, 4 layers) unless overridden.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import _repo_path  # noqa: F401

from sway.gnn_track_refine import RelationalTrackGNN, build_track_graph_tensors


def _resolve_device(name: str) -> torch.device:
    n = name.strip().lower()
    if n == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if n in ("cuda", "auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(n)


def sample_synthetic_scene(
    rng: np.random.Generator,
    *,
    total_frames: int,
    max_gap: float,
) -> Tuple[Dict[int, List[Any]], Dict[int, int], int]:
    """
    Build ``raw_tracks`` and ``tid -> person_id`` for random fragments.

    Same person may appear as 1–2 tracks (duplicate ID / stitch failure); different
    people use separated centers so overlap edges are mostly negatives.
    """
    n_people = int(rng.integers(2, 6))
    raw: Dict[int, List[Any]] = {}
    tid_to_person: Dict[int, int] = {}
    tid = 0
    base_w = 44.0
    base_h = 130.0

    for p in range(n_people):
        n_frag = int(rng.integers(1, 4))
        cx_base = 80.0 + float(rng.uniform(0, 420)) + float(p) * rng.uniform(35, 90)
        cy_base = 90.0 + float(rng.uniform(-40, 40))

        for _ in range(n_frag):
            t_len = int(rng.integers(4, 18))
            gap_before = int(rng.integers(0, min(40, int(max_gap))))
            start = int(rng.integers(0, max(1, total_frames - t_len - gap_before - 5)))
            start += gap_before
            frames = list(range(start, min(start + t_len, total_frames)))
            if not frames:
                continue

            noise = rng.normal(0, 2.5, size=(len(frames), 2)).astype(np.float32)
            for k, f in enumerate(frames):
                cx = cx_base + float(noise[k, 0]) + 0.15 * float(k)
                cy = cy_base + float(noise[k, 1])
                x1 = cx - base_w * 0.5
                y1 = cy - base_h * 0.5
                x2 = cx + base_w * 0.5
                y2 = cy + base_h * 0.5
                conf = float(rng.uniform(0.75, 0.98))
                if tid not in raw:
                    raw[tid] = []
                raw[tid].append((int(f), (float(x1), float(y1), float(x2), float(y2)), conf))
            tid_to_person[tid] = p
            tid += 1

    # Cross-person near-miss: two different people with close time (hard negatives)
    if tid >= 2 and rng.random() < 0.35:
        f0 = int(rng.integers(10, max(11, total_frames - 20)))
        for extra in range(2):
            p_other = int(rng.integers(0, n_people))
            cx = 300.0 + float(extra) * 95.0
            cy = 200.0
            box = (cx - 22, cy - 65, cx + 22, cy + 65)
            raw[tid] = [(f0 + extra, box, 0.88), (f0 + extra + 1, box, 0.87)]
            tid_to_person[tid] = (p_other + 1 + extra) % max(n_people, 1)
            tid += 1

    return raw, tid_to_person, n_people


def edge_bce_loss(
    learn_logits: torch.Tensor,
    prior: torch.Tensor,
    adj: torch.Tensor,
    node_ids: List[int],
    tid_to_person: Dict[int, int],
) -> torch.Tensor:
    """BCE on upper-triangle candidate edges; self-loops skipped."""
    n = learn_logits.size(0)
    device = learn_logits.device
    total = torch.zeros((), device=device)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] < 0.5:
                continue
            y = 1.0 if tid_to_person[node_ids[i]] == tid_to_person[node_ids[j]] else 0.0
            logit = learn_logits[i, j] + prior[i, j].detach()
            total = total + F.binary_cross_entropy_with_logits(
                logit.unsqueeze(0),
                torch.tensor([y], device=device, dtype=torch.float32),
            )
            count += 1
    if count == 0:
        # No candidate edges (only self-loops): plain zero breaks autograd.
        return (learn_logits * 0.0).sum()
    return total / float(count)


def run_training(
    *,
    out_path: Path,
    steps: int,
    lr: float,
    device: torch.device,
    hidden: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    max_gap: float,
    prior_scale: float,
    total_frames: int,
    seed: int,
    log_every: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    node_in = 16
    edge_in = 6
    model = RelationalTrackGNN(
        node_in=node_in,
        edge_in=edge_in,
        hidden=hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    for step in range(steps):
        raw, tid_to_person, _ = sample_synthetic_scene(
            rng, total_frames=total_frames, max_gap=max_gap
        )
        g = build_track_graph_tensors(
            raw,
            total_frames,
            max_gap=max_gap,
            prior_scale=prior_scale,
            device=device,
        )
        if g is None:
            continue

        opt.zero_grad()
        learn = model(g.x0, g.edge_feat, g.adj)
        loss = edge_bce_loss(learn, g.prior_mat, g.adj, g.node_ids, tid_to_person)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"  step {step + 1}/{steps}  loss={loss.item():.4f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "meta": {
            "node_in": node_in,
            "edge_in": edge_in,
            "hidden": hidden,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "steps": steps,
            "max_gap": max_gap,
            "prior_scale": prior_scale,
            "total_frames": total_frames,
            "seed": seed,
        },
    }
    torch.save(payload, out_path)
    print(f"Wrote {out_path}", flush=True)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train GNN track refine (synthetic data).")
    p.add_argument("--out", type=Path, default=Path("models/gnn_track_refine.pt"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-gap", type=float, default=120.0)
    p.add_argument("--prior-scale", type=float, default=1.0)
    p.add_argument("--total-frames", type=int, default=240)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args(argv)

    if args.hidden % args.heads != 0:
        print("--hidden must be divisible by --heads", file=sys.stderr)
        return 2

    run_training(
        out_path=args.out.expanduser().resolve(),
        steps=args.steps,
        lr=args.lr,
        device=_resolve_device(args.device),
        hidden=args.hidden,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
        max_gap=args.max_gap,
        prior_scale=args.prior_scale,
        total_frames=args.total_frames,
        seed=args.seed,
        log_every=args.log_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
