#!/usr/bin/env python3
"""
Optuna TPE sweep for Phase 1–3 (``--stop-after-boundary after_phase_3``).

- Multi-video objective: **weighted** harmonic mean of per-video composite scores.
  Sequence weights are read from the YAML config (``weight:`` field, default 1.0).
- **NopPruner**: every trial runs all videos; no mid-trial pruning.
- **Duplicate detection**: if Optuna re-suggests an already-completed param set,
  the trial is pruned immediately (no GPU cost).
- Handshake SAM IoU: set via ``SWAY_*`` in ``--params`` YAML (CLI), not Lab API.

Config: copy ``data/ground_truth/sweep_sequences.example.yaml`` → ``sweep_sequences.yaml``.

On Lambda (**gpu_1x_a10**, A100-class, etc.) or other multi-vCPU NVIDIA hosts, export
``SWAY_SERVER_PERF=1`` so each ``main.py`` child gets cuDNN autotune, TF32, and bounded CPU
thread pools (see ``sway.server_runtime_perf``). **A10 (~24 GB):** keep default YOLO infer
batch **1** unless you have headroom. Verify propagation: ``python -m tools.smoke_server_perf_env``
(optional ``--pipeline --timeout 60``).

  python -m tools.auto_sweep --config data/ground_truth/sweep_sequences.yaml

Runs until you stop it: **Ctrl+C** (SIGINT), **SIGTERM**, or ``touch`` the stop file
(default: ``output/sweeps/optuna/STOP``) — the **current trial** finishes, then the study exits.
Optional cap: ``--n-trials 60``. Show best so far: ``--show-best``.

**Live monitoring:** by default writes atomic ``output/sweeps/optuna/sweep_status.json`` after
each trial (all trials + best); ``sweep_log.jsonl`` appends one line per completed trial.
Disable JSON with ``--no-status-json``. Refresh from DB only: ``python -m tools.export_optuna_study_status``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import signal
import statistics
import time
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sway.server_runtime_perf import subprocess_env_overlay

# --- YOLO weights (playbook §9.1) ---
YOLO_WEIGHT_CHOICES = ("yolo26l", "yolo26l_dancetrack", "yolo26l_dancetrack_crowdhuman")
DEFAULT_ALLOWED_TRACKER_ENGINES = ("solidtrack", "matr")
FLOOR_SCORE_EPS_DEFAULT = 1e-6
BIGTEST_SLOW_SECONDS_DEFAULT = 850.0

# AFLink default path (global_track_link.py)
AFLINK_DEFAULT = REPO_ROOT / "models" / "AFLink_epoch20.pth"
FEATURE_LINE_RE = re.compile(
    r"\[feature\]\s*([^:]+):\s*requested=([a-zA-Z]+),\s*runtime=([a-zA-Z]+),\s*wiring=([a-zA-Z]+)"
)


def _is_on_token(v: str) -> bool:
    return str(v).strip().lower() in {"on", "1", "true", "yes", "active", "enabled"}


def _parse_feature_states(log_text: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for m in FEATURE_LINE_RE.finditer(log_text or ""):
        out[m.group(1).strip()] = {
            "requested": m.group(2).strip().lower(),
            "runtime": m.group(3).strip().lower(),
            "wiring": m.group(4).strip().lower(),
        }
    return out


def _feature_mismatches(features: Dict[str, Dict[str, str]]) -> List[str]:
    mismatches: List[str] = []
    for name, row in features.items():
        if _is_on_token(row.get("requested", "")):
            runtime_ok = _is_on_token(row.get("runtime", ""))
            wiring_ok = str(row.get("wiring", "")).strip().lower() in {"wired", "n/a"}
            if not (runtime_ok and wiring_ok):
                mismatches.append(name)
    return sorted(mismatches)


def _load_run_manifest(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run_manifest.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _runtime_signature_from_meta(meta: Dict[str, Any]) -> str:
    if not meta:
        return "none"
    tracker_path = str(meta.get("tracker_path") or "unknown")
    features = meta.get("features") or {}
    bits = []
    for name in sorted(features.keys()):
        row = features[name]
        bits.append(
            f"{name}:{row.get('requested','?')}/{row.get('runtime','?')}/{row.get('wiring','?')}"
        )
    return tracker_path + "|" + "|".join(bits)


def _resolve_path(p: str | Path) -> Path:
    x = Path(p)
    if not x.is_absolute():
        x = (REPO_ROOT / x).resolve()
    return x


def _git_sha_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _weight_file_exists(token: str) -> bool:
    from sway.tracker import _SWAY_YOLO_WEIGHTS_ALIASES

    mapped = _SWAY_YOLO_WEIGHTS_ALIASES.get(token.lower())
    if not mapped:
        return False
    name = Path(mapped).name
    for base in (REPO_ROOT / "models", REPO_ROOT, Path.cwd()):
        if (base / name).is_file():
            return True
    return False


def _available_yolo_weights() -> List[str]:
    return [w for w in YOLO_WEIGHT_CHOICES if _weight_file_exists(w)] or list(YOLO_WEIGHT_CHOICES)


def composite_score(metrics: Dict[str, Any]) -> float:
    """
    Map TrackEval flat dict → [0,1]-ish scalar. Uses best-effort keys (see tests / boxmot ablation).
    """
    if not metrics:
        return 0.0

    def f(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            v = metrics.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        return default

    hota = f("HOTA_HOTA", default=0.0)
    if hota == 0.0:
        for k, v in metrics.items():
            if k.startswith("HOTA_HOTA") and isinstance(v, (int, float)):
                hota = float(v)
                break

    recall = f("CLEAR_CLR_Re", "CLEAR_RECALL", default=0.0)
    if recall == 0.0:
        recall = f("Identity_IDF1", default=hota)

    idsw = int(f("Identity_IDSW", "CLEAR_IDSW", default=0))
    tp = int(f("CLEAR_TP", default=0))
    fn = int(f("CLEAR_FN", default=0))
    total_gt = max(tp + fn, 1)

    frag = 0
    for k, v in metrics.items():
        if "FRAG" in k.upper() and isinstance(v, (int, float)):
            frag = int(v)
            break

    idsw_penalty = min(idsw / max(total_gt * 0.05, 1), 1.0)
    frag_penalty = min(frag / max(total_gt * 0.10, 1), 1.0)

    return (
        0.45 * hota
        + 0.25 * recall
        + 0.20 * (1.0 - idsw_penalty)
        + 0.10 * (1.0 - frag_penalty)
    )


SOLIDER_ONNX = REPO_ROOT / "models" / "solider_swin_small_msmt17.onnx"
FASTTRACKER_DIR = REPO_ROOT / "models" / "FastTracker"

# Explicit ownership map for phase 1-3 sweep coverage.
PHASE13_KEY_OWNERSHIP: Dict[str, List[str]] = {
    "detection": [
        "SWAY_YOLO_WEIGHTS",
        "SWAY_YOLO_CONF",
        "SWAY_PRETRACK_NMS_IOU",
        "SWAY_DETECT_SIZE",
        "SWAY_YOLO_DETECTION_STRIDE",
        "SWAY_DETECTOR_PRIMARY",
        "SWAY_DETECTOR_HYBRID",
        "SWAY_DETECTOR_PRECISION",
        "SWAY_DETR_CONF",
        "SWAY_HYBRID_OVERLAP_IOU_TRIGGER",
        "SWAY_DETECTION_UNCERTAIN_CONF",
    ],
    "tracking_engine": [
        "SWAY_TRACKER_ENGINE",
        "SWAY_USE_BOXMOT",
        "SWAY_TRACK_MAX_AGE",
    ],
    "boxmot_tracker_family": [
        "SWAY_BOXMOT_TRACKER",
        "SWAY_BOXMOT_MAX_AGE",
        "SWAY_BOXMOT_MATCH_THRESH",
        "SWAY_BOXMOT_REID_ON",
        "SWAY_BOXMOT_REID_WEIGHTS",
        "SWAY_BOXMOT_ASSOC_METRIC",
        "SWAY_DOC_ALPHA_EMB",
        "SWAY_DOC_AW_PARAM",
        "SWAY_DOC_AW_OFF",
        "SWAY_DOC_CMC_OFF",
        "SWAY_DOC_DELTA_T",
        "SWAY_DOC_INERTIA",
        "SWAY_DOC_W_EMB",
        "SWAY_DOC_Q_XY",
        "SWAY_DOC_Q_S",
        "SWAY_DOC_NSA_KF_ON",
        "SWAY_BYTETRACK_MATCH_THRESH",
        "SWAY_BYTETRACK_TRACK_BUFFER",
        "SWAY_OCSORT_USE_BYTE",
        "SWAY_STRONGSORT_MAX_COS_DIST",
        "SWAY_STRONGSORT_MAX_IOU_DIST",
        "SWAY_STRONGSORT_N_INIT",
        "SWAY_STRONGSORT_NN_BUDGET",
        "SWAY_BOOST_CMC_METHOD",
        "SWAY_BOOST_LAMBDA_IOU",
        "SWAY_BOOST_LAMBDA_MHD",
        "SWAY_BOOST_LAMBDA_SHP",
        "SWAY_BOOST_DLO",
        "SWAY_BOOST_DUO",
        "SWAY_BOOST_DLO_COEF",
        "SWAY_BOOST_RICH_S",
        "SWAY_BOOST_SB",
        "SWAY_BOOST_VT",
        "SWAY_BOOST_REID",
        "SWAY_BOTSORT_CMC",
        "SWAY_ST_THETA_IOU",
        "SWAY_ST_THETA_EMB",
        "SWAY_ST_EMA_ALPHA",
        "SWAY_FT_DET_HIGH",
        "SWAY_FT_DET_LOW",
        "SWAY_FT_PROXIMITY_RAD",
        "SWAY_FT_BOX_ENLARGE",
        "SWAY_FT_MOTION_DAMP",
    ],
    "sam2_memosort_and_state": [
        "SWAY_SAM2_REINVOKE_STRIDE",
        "SWAY_SAM2_CONFIDENCE_REINVOKE",
        "SWAY_SAM2_MEMORY_FRAMES",
        "SWAY_SAM2_MASK_POSE",
        "SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA",
        "SWAY_MEMOSORT_MEMORY_LENGTH",
        "SWAY_COI_MASK_IOU_THRESH",
        "SWAY_COI_LOGIT_VARIANCE_WINDOW",
        "SWAY_COI_QUARANTINE_MODE",
        "SWAY_ENROLLMENT_ENABLED",
        "SWAY_ENROLLMENT_AUTO_FRAME",
        "SWAY_ENROLLMENT_GALLERY_SIGNALS",
        "SWAY_ENROLLMENT_PART_MODEL",
        "SWAY_ENROLLMENT_COLOR_BINS",
        "SWAY_ENROLLMENT_MIN_SEPARATION_PX",
    ],
    "phase3_stitch_and_link": [
        "SWAY_STITCH_MAX_FRAME_GAP",
        "SWAY_STITCH_RADIUS_BBOX_FRAC",
        "SWAY_STITCH_MAX_PIXEL_RADIUS",
        "SWAY_STITCH_PREDICTED_RADIUS_FRAC",
        "SWAY_SHORT_GAP_FRAMES",
        "SWAY_DORMANT_MAX_GAP",
        "SWAY_COALESCENCE_IOU_THRESH",
        "SWAY_COALESCENCE_CONSECUTIVE_FRAMES",
        "SWAY_BOX_INTERP_MODE",
        "SWAY_GSI_LENGTHSCALE",
        "SWAY_GLOBAL_LINK",
        "SWAY_GLOBAL_AFLINK",
        "SWAY_AFLINK_THR_P",
        "SWAY_AFLINK_THR_S",
        "SWAY_AFLINK_THR_T0",
        "SWAY_AFLINK_THR_T1",
    ],
    "phase13_modes_and_sam": [
        "SWAY_PHASE13_MODE",
        "SWAY_HYBRID_SAM_IOU_TRIGGER",
        "SWAY_HYBRID_SAM_WEAK_CUES",
        "SWAY_HANDSHAKE_VERIFY_STRIDE",
        "SWAY_REGISTRY_TOUCH_IOU",
        "SWAY_REGISTRY_SWAP_MARGIN",
        "SWAY_REGISTRY_ISOLATION_MULT",
        "SWAY_REGISTRY_DORMANT_MATCH",
        "SWAY_HYBRID_SAM_MASK_THRESH",
        "SWAY_HYBRID_SAM_BBOX_PAD",
        "SWAY_HYBRID_SAM_ROI_PAD_FRAC",
    ],
    "advanced_phase13_modules": [
        "SWAY_BACKWARD_PASS_ENABLED",
        "SWAY_BACKWARD_STITCH_MIN_SIMILARITY",
        "SWAY_BACKWARD_STITCH_MAX_GAP",
        "SWAY_BACKWARD_COI_ENABLED",
        "SWAY_COLLISION_SOLVER",
        "SWAY_COLLISION_MIN_TRACKS",
        "SWAY_COLLISION_DP_MAX_PERMUTATIONS",
        "SWAY_MOTE_DISOCCLUSION",
        "SWAY_MOTE_FLOW_MODEL",
        "SWAY_MOTE_CONFIDENCE_BOOST",
        "SWAY_SENTINEL_SBM",
        "SWAY_SENTINEL_GRACE_MULTIPLIER",
        "SWAY_SENTINEL_WEAK_DET_CONF",
        "SWAY_UMOT_BACKTRACK",
        "SWAY_UMOT_HISTORY_LENGTH",
    ],
}

# Keys always fixed by design (phase 1-3 studies keep these constant).
PHASE13_FIXED_SWEEP_KEYS = {
    "SWAY_YOLO_DETECTION_STRIDE",
}


def _available_reid_weights() -> List[str]:
    """ReID weight choices; SOLIDER ONNX only included if the file exists on disk."""
    base = [
        "osnet_x0_25_msmt17.pt",
        "osnet_x1_0_msmt17.pt",
        "osnet_x0_25_market1501.pt",
        "osnet_ain_x1_0_msmt17.pt",
        "clip_market1501.pt",
        "lmbn_n_market.pt",
    ]
    if SOLIDER_ONNX.is_file():
        base.append(str(SOLIDER_ONNX))
    return base


def _available_tracker_types() -> List[str]:
    """Tracker types available to sweep; fasttracker only if its repo exists."""
    types = [
        "deepocsort", "bytetrack", "ocsort", "strongsort",
        "boosttrack", "botsort", "solidtrack",
    ]
    if FASTTRACKER_DIR.is_dir():
        types.append("fasttracker")
    return types


def _allowed_tracker_engines() -> List[str]:
    """
    Temporary safety gate while runtime wiring is being hardened.
    Override with SWAY_SWEEP_ALLOWED_ENGINES=comma,separated,list.
    """
    raw = os.environ.get("SWAY_SWEEP_ALLOWED_ENGINES", "").strip()
    if raw:
        allowed = [x.strip() for x in raw.split(",") if x.strip()]
        return allowed or list(DEFAULT_ALLOWED_TRACKER_ENGINES)
    return list(DEFAULT_ALLOWED_TRACKER_ENGINES)


def _engine_matches_tracker_path(expected_engine: str, tracker_path: str) -> bool:
    e = str(expected_engine or "").strip().lower()
    t = str(tracker_path or "").strip().lower()
    if not e or not t or t == "unknown":
        return False
    if e == "solidtrack":
        return (
            "boxmot:" in t
            or any(
                k in t
                for k in (
                    "deepocsort",
                    "strongsort",
                    "boosttrack",
                    "bytetrack",
                    "ocsort",
                    "botsort",
                    "solidtrack",
                    "fasttracker",
                )
            )
        )
    if e == "memosort":
        return "memosort" in t
    if e == "matr":
        return "matr" in t
    if e == "sam2mot":
        return "sam2" in t and "hybrid" not in t and "memosort" not in t
    if e == "sam2_memosort_hybrid":
        return ("sam2" in t and "hybrid" in t) or "sam2_memosort" in t
    return False


def _phase13_owned_keys() -> List[str]:
    keys = set()
    for values in PHASE13_KEY_OWNERSHIP.values():
        keys.update(values)
    return sorted(keys)


def phase13_search_space_coverage_report() -> Dict[str, Any]:
    """
    Contract: every owned phase 1-3 key must be tuned or fixed.
    """
    tuned = set(_phase13_owned_keys()) - set(PHASE13_FIXED_SWEEP_KEYS)
    owned = set(_phase13_owned_keys())
    covered = tuned | set(PHASE13_FIXED_SWEEP_KEYS)
    missing = sorted(owned - covered)
    return {
        "owned_key_count": len(owned),
        "tuned_key_count": len(tuned),
        "fixed_key_count": len(PHASE13_FIXED_SWEEP_KEYS),
        "missing_keys": missing,
        "is_full_coverage": not missing,
        "ownership": PHASE13_KEY_OWNERSHIP,
    }


def suggest_env_for_trial(trial: Any, available_weights: List[str]) -> Dict[str, str]:
    """Build SWAY_* env map for the full phase 1-3 technology sweep."""
    env: Dict[str, str] = {}

    # ── 1. Detection & detector family ──────────────────────────────────
    env["SWAY_DETECTOR_PRIMARY"] = trial.suggest_categorical(
        "SWAY_DETECTOR_PRIMARY",
        ["yolo26l_dancetrack", "rt_detr_l", "rt_detr_x", "co_detr", "co_dino"],
    )
    env["SWAY_YOLO_WEIGHTS"] = trial.suggest_categorical("SWAY_YOLO_WEIGHTS", available_weights)
    env["SWAY_YOLO_CONF"] = str(round(
        trial.suggest_float("SWAY_YOLO_CONF", 0.10, 0.32, step=0.01), 4
    ))
    env["SWAY_PRETRACK_NMS_IOU"] = str(round(
        trial.suggest_float("SWAY_PRETRACK_NMS_IOU", 0.35, 0.65, step=0.01), 4
    ))
    env["SWAY_DETECT_SIZE"] = str(
        trial.suggest_categorical("SWAY_DETECT_SIZE", ["640", "800", "960"])
    )
    env["SWAY_YOLO_DETECTION_STRIDE"] = "1"  # Fixed for stable fair comparisons.
    env["SWAY_DETR_CONF"] = str(round(
        trial.suggest_float("SWAY_DETR_CONF", 0.20, 0.50, step=0.01), 4
    ))
    env["SWAY_DETECTOR_HYBRID"] = trial.suggest_categorical("SWAY_DETECTOR_HYBRID", ["0", "1"])
    if env["SWAY_DETECTOR_HYBRID"] == "1":
        env["SWAY_DETECTOR_PRECISION"] = trial.suggest_categorical(
            "SWAY_DETECTOR_PRECISION", ["rt_detr_l", "rt_detr_x", "co_detr", "co_dino"]
        )
        env["SWAY_HYBRID_OVERLAP_IOU_TRIGGER"] = str(round(
            trial.suggest_float("SWAY_HYBRID_OVERLAP_IOU_TRIGGER", 0.15, 0.55, step=0.02), 4
        ))
        env["SWAY_DETECTION_UNCERTAIN_CONF"] = str(round(
            trial.suggest_float("SWAY_DETECTION_UNCERTAIN_CONF", 0.30, 0.70, step=0.02), 4
        ))

    # ── 2. Tracker engine tree ───────────────────────────────────────────
    engine = trial.suggest_categorical("SWAY_TRACKER_ENGINE", _allowed_tracker_engines())
    env["SWAY_TRACKER_ENGINE"] = engine
    env["SWAY_TRACK_MAX_AGE"] = str(trial.suggest_int("SWAY_TRACK_MAX_AGE", 60, 360, step=15))

    if engine == "solidtrack":
        env["SWAY_USE_BOXMOT"] = "1"
        tracker_type = trial.suggest_categorical("SWAY_BOXMOT_TRACKER", _available_tracker_types())
        env["SWAY_BOXMOT_TRACKER"] = tracker_type
        env["SWAY_BOXMOT_MAX_AGE"] = str(trial.suggest_int("SWAY_BOXMOT_MAX_AGE", 45, 210, step=15))
        env["SWAY_BOXMOT_MATCH_THRESH"] = str(round(
            trial.suggest_float("SWAY_BOXMOT_MATCH_THRESH", 0.15, 0.45, step=0.01), 4
        ))

        reid_always_on = tracker_type in ("strongsort", "solidtrack")
        if reid_always_on:
            reid_on = "1"
        elif tracker_type == "boosttrack":
            reid_on = trial.suggest_categorical("b_bst_REID", ["0", "1"])
        else:
            reid_on = trial.suggest_categorical("SWAY_BOXMOT_REID_ON", ["0", "1"])
        env["SWAY_BOXMOT_REID_ON"] = reid_on
        if reid_on == "1":
            env["SWAY_BOXMOT_REID_WEIGHTS"] = trial.suggest_categorical(
                "SWAY_BOXMOT_REID_WEIGHTS",
                _available_reid_weights(),
            )

        if tracker_type == "deepocsort":
            env["SWAY_BOXMOT_ASSOC_METRIC"] = trial.suggest_categorical("b_doc_ASSOC", ["iou", "giou", "ciou"])
            env["SWAY_DOC_ALPHA_EMB"] = str(round(trial.suggest_float("b_doc_AEMB", 0.80, 0.99, step=0.01), 4))
            env["SWAY_DOC_AW_PARAM"] = str(round(trial.suggest_float("b_doc_AWPAR", 0.2, 0.8, step=0.05), 4))
            env["SWAY_DOC_AW_OFF"] = trial.suggest_categorical("b_doc_AWOFF", ["0", "1"])
            env["SWAY_DOC_CMC_OFF"] = trial.suggest_categorical("b_doc_CMCOFF", ["0", "1"])
            env["SWAY_DOC_DELTA_T"] = str(trial.suggest_int("b_doc_DT", 1, 6))
            env["SWAY_DOC_INERTIA"] = str(round(trial.suggest_float("b_doc_IN", 0.05, 0.5, step=0.05), 4))
            env["SWAY_DOC_W_EMB"] = str(round(trial.suggest_float("b_doc_WEMB", 0.2, 0.8, step=0.05), 4))
            env["SWAY_DOC_Q_XY"] = str(round(trial.suggest_float("b_doc_QXY", 0.001, 0.05, log=True), 5))
            env["SWAY_DOC_Q_S"] = str(round(trial.suggest_float("b_doc_QS", 0.00001, 0.001, log=True), 7))
            env["SWAY_DOC_NSA_KF_ON"] = trial.suggest_categorical("b_doc_NSA", ["0", "1"])
        elif tracker_type == "bytetrack":
            env["SWAY_BYTETRACK_MATCH_THRESH"] = str(round(trial.suggest_float("b_bt_MATCH", 0.65, 0.95, step=0.05), 4))
            env["SWAY_BYTETRACK_TRACK_BUFFER"] = str(trial.suggest_int("b_bt_BUF", 10, 50, step=5))
        elif tracker_type == "ocsort":
            env["SWAY_OCSORT_USE_BYTE"] = trial.suggest_categorical("b_oc_BYTE", ["0", "1"])
        elif tracker_type == "strongsort":
            env["SWAY_STRONGSORT_MAX_COS_DIST"] = str(round(trial.suggest_float("b_ss_COS", 0.10, 0.45, step=0.05), 4))
            env["SWAY_STRONGSORT_MAX_IOU_DIST"] = str(round(trial.suggest_float("b_ss_IOU", 0.45, 0.90, step=0.05), 4))
            env["SWAY_STRONGSORT_N_INIT"] = str(trial.suggest_int("b_ss_NINIT", 1, 5))
            env["SWAY_STRONGSORT_NN_BUDGET"] = str(trial.suggest_int("b_ss_BUDGET", 50, 200, step=25))
        elif tracker_type == "boosttrack":
            env["SWAY_BOOST_CMC_METHOD"] = trial.suggest_categorical("b_bst_CMC", ["ecc", "sof", "orb"])
            env["SWAY_BOOST_LAMBDA_IOU"] = str(round(trial.suggest_float("b_bst_LIOU", 0.3, 0.9, step=0.05), 4))
            env["SWAY_BOOST_LAMBDA_MHD"] = str(round(trial.suggest_float("b_bst_LMHD", 0.1, 0.5, step=0.05), 4))
            env["SWAY_BOOST_LAMBDA_SHP"] = str(round(trial.suggest_float("b_bst_LSHP", 0.1, 0.5, step=0.05), 4))
            env["SWAY_BOOST_DLO"] = trial.suggest_categorical("b_bst_DLO", ["0", "1"])
            env["SWAY_BOOST_DUO"] = trial.suggest_categorical("b_bst_DUO", ["0", "1"])
            env["SWAY_BOOST_DLO_COEF"] = str(round(trial.suggest_float("b_bst_COEF", 0.40, 0.85, step=0.05), 4))
            env["SWAY_BOOST_RICH_S"] = trial.suggest_categorical("b_bst_RICH", ["0", "1"])
            env["SWAY_BOOST_SB"] = trial.suggest_categorical("b_bst_SB", ["0", "1"])
            env["SWAY_BOOST_VT"] = trial.suggest_categorical("b_bst_VT", ["0", "1"])
            env["SWAY_BOOST_REID"] = trial.suggest_categorical("b_bst_REID", ["0", "1"])
        elif tracker_type == "botsort":
            env["SWAY_BOTSORT_CMC"] = trial.suggest_categorical("b_bs_CMC", ["ecc", "sof"])
        elif tracker_type == "solidtrack":
            env["SWAY_ST_THETA_IOU"] = str(round(trial.suggest_float("b_st_TIOU", 0.3, 0.7, step=0.05), 4))
            env["SWAY_ST_THETA_EMB"] = str(round(trial.suggest_float("b_st_TEMB", 0.1, 0.4, step=0.025), 4))
            env["SWAY_ST_EMA_ALPHA"] = str(round(trial.suggest_float("b_st_EMA", 0.7, 0.99, step=0.01), 4))
        elif tracker_type == "fasttracker":
            env["SWAY_FT_DET_HIGH"] = str(round(trial.suggest_float("b_ft_DETH", 0.3, 0.7, step=0.05), 4))
            env["SWAY_FT_DET_LOW"] = str(round(trial.suggest_float("b_ft_DETL", 0.05, 0.3, step=0.05), 4))
            env["SWAY_FT_PROXIMITY_RAD"] = str(round(trial.suggest_float("b_ft_PRAD", 0.1, 0.5, step=0.05), 4))
            env["SWAY_FT_BOX_ENLARGE"] = str(round(trial.suggest_float("b_ft_BXEN", 1.0, 1.5, step=0.05), 4))
            env["SWAY_FT_MOTION_DAMP"] = str(round(trial.suggest_float("b_ft_MDMP", 0.3, 0.9, step=0.1), 4))
    else:
        env["SWAY_USE_BOXMOT"] = "0"
        if engine in ("sam2mot", "sam2_memosort_hybrid"):
            env["SWAY_SAM2_REINVOKE_STRIDE"] = str(
                trial.suggest_int("SWAY_SAM2_REINVOKE_STRIDE", 10, 60, step=5)
            )
            env["SWAY_SAM2_CONFIDENCE_REINVOKE"] = str(round(
                trial.suggest_float("SWAY_SAM2_CONFIDENCE_REINVOKE", 0.20, 0.60, step=0.05), 4
            ))
            env["SWAY_SAM2_MEMORY_FRAMES"] = str(
                trial.suggest_int("SWAY_SAM2_MEMORY_FRAMES", 60, 240, step=30)
            )
            env["SWAY_SAM2_MASK_POSE"] = trial.suggest_categorical("SWAY_SAM2_MASK_POSE", ["0", "1"])
        if engine in ("memosort", "sam2_memosort_hybrid", "matr"):
            env["SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA"] = str(round(
                trial.suggest_float("SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA", 0.10, 0.90, step=0.05), 4
            ))
            env["SWAY_MEMOSORT_MEMORY_LENGTH"] = str(
                trial.suggest_int("SWAY_MEMOSORT_MEMORY_LENGTH", 30, 240, step=15)
            )

    env["SWAY_COI_MASK_IOU_THRESH"] = str(round(
        trial.suggest_float("SWAY_COI_MASK_IOU_THRESH", 0.15, 0.45, step=0.05), 4
    ))
    env["SWAY_COI_LOGIT_VARIANCE_WINDOW"] = str(
        trial.suggest_int("SWAY_COI_LOGIT_VARIANCE_WINDOW", 3, 15, step=2)
    )
    env["SWAY_COI_QUARANTINE_MODE"] = trial.suggest_categorical(
        "SWAY_COI_QUARANTINE_MODE", ["hard", "soft"]
    )
    env["SWAY_ENROLLMENT_ENABLED"] = trial.suggest_categorical("SWAY_ENROLLMENT_ENABLED", ["0", "1"])
    if env["SWAY_ENROLLMENT_ENABLED"] == "1":
        env["SWAY_ENROLLMENT_AUTO_FRAME"] = trial.suggest_categorical("SWAY_ENROLLMENT_AUTO_FRAME", ["0", "1"])
        env["SWAY_ENROLLMENT_GALLERY_SIGNALS"] = trial.suggest_categorical(
            "SWAY_ENROLLMENT_GALLERY_SIGNALS",
            ["part,color,spatial", "part,color", "part"],
        )
        env["SWAY_ENROLLMENT_PART_MODEL"] = trial.suggest_categorical(
            "SWAY_ENROLLMENT_PART_MODEL",
            ["bpbreid", "osnet"],
        )
        env["SWAY_ENROLLMENT_COLOR_BINS"] = str(
            trial.suggest_int("SWAY_ENROLLMENT_COLOR_BINS", 8, 64, step=8)
        )
        env["SWAY_ENROLLMENT_MIN_SEPARATION_PX"] = str(
            trial.suggest_int("SWAY_ENROLLMENT_MIN_SEPARATION_PX", 8, 64, step=4)
        )

    # ── 3. Post-track stitching & linking ────────────────────────────────
    env["SWAY_STITCH_MAX_FRAME_GAP"] = str(
        trial.suggest_int("SWAY_STITCH_MAX_FRAME_GAP", 20, 150, step=10)
    )
    env["SWAY_STITCH_RADIUS_BBOX_FRAC"] = str(round(
        trial.suggest_float("SWAY_STITCH_RADIUS_BBOX_FRAC", 0.20, 1.20, step=0.05), 4
    ))
    env["SWAY_SHORT_GAP_FRAMES"] = str(
        trial.suggest_int("SWAY_SHORT_GAP_FRAMES", 5, 40, step=5)
    )
    env["SWAY_DORMANT_MAX_GAP"] = str(
        trial.suggest_int("SWAY_DORMANT_MAX_GAP", 60, 300, step=15)
    )
    env["SWAY_COALESCENCE_IOU_THRESH"] = str(round(
        trial.suggest_float("SWAY_COALESCENCE_IOU_THRESH", 0.50, 0.90, step=0.05), 4
    ))
    env["SWAY_COALESCENCE_CONSECUTIVE_FRAMES"] = str(
        trial.suggest_int("SWAY_COALESCENCE_CONSECUTIVE_FRAMES", 3, 20)
    )
    env["SWAY_STITCH_MAX_PIXEL_RADIUS"] = str(
        trial.suggest_int("SWAY_STITCH_MAX_PIXEL_RADIUS", 80, 420, step=20)
    )
    env["SWAY_STITCH_PREDICTED_RADIUS_FRAC"] = str(round(
        trial.suggest_float("SWAY_STITCH_PREDICTED_RADIUS_FRAC", 0.05, 0.80, step=0.05), 4
    ))
    env["SWAY_GLOBAL_LINK"] = trial.suggest_categorical("SWAY_GLOBAL_LINK", ["0", "1"])

    box_interp = trial.suggest_categorical("SWAY_BOX_INTERP_MODE", ["linear", "gsi"])
    env["SWAY_BOX_INTERP_MODE"] = box_interp
    if box_interp == "gsi":
        env["SWAY_GSI_LENGTHSCALE"] = str(round(
            trial.suggest_float("b_gsi_LS", 0.10, 0.80, step=0.05), 4
        ))

    aflink_mode = trial.suggest_categorical("sway_global_aflink_mode", ["neural_if_available", "force_heuristic"])
    if aflink_mode == "force_heuristic" or not AFLINK_DEFAULT.is_file():
        env["SWAY_GLOBAL_AFLINK"] = "0"
    else:
        env["SWAY_GLOBAL_AFLINK"] = "1"
        env["SWAY_AFLINK_THR_P"] = str(round(trial.suggest_float("SWAY_AFLINK_THR_P", 0.01, 0.20, step=0.01), 4))
        env["SWAY_AFLINK_THR_S"] = str(trial.suggest_int("SWAY_AFLINK_THR_S", 40, 120, step=5))
        env["SWAY_AFLINK_THR_T0"] = str(trial.suggest_int("SWAY_AFLINK_THR_T0", 0, 10))
        env["SWAY_AFLINK_THR_T1"] = str(trial.suggest_int("SWAY_AFLINK_THR_T1", 10, 60, step=5))

    # ── 4. Hybrid SAM + phase mode branches ──────────────────────────────
    env["SWAY_HYBRID_SAM_MASK_THRESH"] = str(round(
        trial.suggest_float("SWAY_HYBRID_SAM_MASK_THRESH", 0.35, 0.65, step=0.05), 4
    ))
    env["SWAY_HYBRID_SAM_BBOX_PAD"] = str(
        trial.suggest_int("SWAY_HYBRID_SAM_BBOX_PAD", 0, 8)
    )
    env["SWAY_HYBRID_SAM_ROI_PAD_FRAC"] = str(round(
        trial.suggest_float("SWAY_HYBRID_SAM_ROI_PAD_FRAC", 0.03, 0.25, step=0.02), 4
    ))

    mode = trial.suggest_categorical(
        "SWAY_PHASE13_MODE", ["standard", "dancer_registry", "sway_handshake"]
    )
    env["SWAY_PHASE13_MODE"] = mode

    if mode == "standard":
        env["SWAY_HYBRID_SAM_IOU_TRIGGER"] = str(round(
            trial.suggest_float("b_std_SAM_IOU", 0.25, 0.65, step=0.025), 4
        ))
    elif mode == "dancer_registry":
        env["SWAY_HYBRID_SAM_IOU_TRIGGER"] = str(round(
            trial.suggest_float("b_reg_SAM_IOU", 0.25, 0.65, step=0.025), 4
        ))
        env["SWAY_REGISTRY_TOUCH_IOU"] = str(round(
            trial.suggest_float("b_reg_TIOU", 0.04, 0.22, step=0.01), 4
        ))
        env["SWAY_REGISTRY_SWAP_MARGIN"] = str(round(
            trial.suggest_float("b_reg_SWAP", 0.01, 0.18, step=0.01), 4
        ))
        env["SWAY_REGISTRY_ISOLATION_MULT"] = str(round(
            trial.suggest_float("b_reg_ISO", 1.0, 2.2, step=0.1), 4
        ))
        env["SWAY_REGISTRY_DORMANT_MATCH"] = str(round(
            trial.suggest_float("b_reg_DORM", 0.70, 0.95, step=0.025), 4
        ))
    else:  # sway_handshake
        env["SWAY_HYBRID_SAM_IOU_TRIGGER"] = str(round(
            trial.suggest_float("b_hs_SAM_IOU", 0.03, 0.25, step=0.01), 4
        ))
        env["SWAY_HYBRID_SAM_WEAK_CUES"] = trial.suggest_categorical(
            "b_hs_WEAK", ["0", "1"]
        )
        env["SWAY_HANDSHAKE_VERIFY_STRIDE"] = str(
            trial.suggest_categorical("b_hs_VSTRIDE", ["1", "2", "3", "5"])
        )
        env["SWAY_REGISTRY_ISOLATION_MULT"] = str(round(
            trial.suggest_float("b_hs_ISO", 1.0, 2.0, step=0.1), 4
        ))

    # ── 5. Advanced phase 1-3 modules ────────────────────────────────────
    env["SWAY_BACKWARD_PASS_ENABLED"] = trial.suggest_categorical("SWAY_BACKWARD_PASS_ENABLED", ["0", "1"])
    if env["SWAY_BACKWARD_PASS_ENABLED"] == "1":
        env["SWAY_BACKWARD_STITCH_MIN_SIMILARITY"] = str(round(
            trial.suggest_float("SWAY_BACKWARD_STITCH_MIN_SIMILARITY", 0.40, 0.90, step=0.05), 4
        ))
        env["SWAY_BACKWARD_STITCH_MAX_GAP"] = str(
            trial.suggest_int("SWAY_BACKWARD_STITCH_MAX_GAP", 20, 180, step=10)
        )
        env["SWAY_BACKWARD_COI_ENABLED"] = trial.suggest_categorical("SWAY_BACKWARD_COI_ENABLED", ["0", "1"])

    env["SWAY_COLLISION_SOLVER"] = trial.suggest_categorical("SWAY_COLLISION_SOLVER", ["greedy", "hungarian", "dp"])
    env["SWAY_COLLISION_MIN_TRACKS"] = str(trial.suggest_int("SWAY_COLLISION_MIN_TRACKS", 2, 6))
    if env["SWAY_COLLISION_SOLVER"] == "dp":
        env["SWAY_COLLISION_DP_MAX_PERMUTATIONS"] = str(
            trial.suggest_int("SWAY_COLLISION_DP_MAX_PERMUTATIONS", 50, 400, step=25)
        )

    env["SWAY_MOTE_DISOCCLUSION"] = trial.suggest_categorical("SWAY_MOTE_DISOCCLUSION", ["0", "1"])
    if env["SWAY_MOTE_DISOCCLUSION"] == "1":
        env["SWAY_MOTE_FLOW_MODEL"] = trial.suggest_categorical("SWAY_MOTE_FLOW_MODEL", ["raft_small", "farneback"])
        env["SWAY_MOTE_CONFIDENCE_BOOST"] = str(round(
            trial.suggest_float("SWAY_MOTE_CONFIDENCE_BOOST", 0.0, 0.4, step=0.05), 4
        ))

    env["SWAY_SENTINEL_SBM"] = trial.suggest_categorical("SWAY_SENTINEL_SBM", ["0", "1"])
    if env["SWAY_SENTINEL_SBM"] == "1":
        env["SWAY_SENTINEL_GRACE_MULTIPLIER"] = str(round(
            trial.suggest_float("SWAY_SENTINEL_GRACE_MULTIPLIER", 1.5, 5.0, step=0.25), 4
        ))
        env["SWAY_SENTINEL_WEAK_DET_CONF"] = str(round(
            trial.suggest_float("SWAY_SENTINEL_WEAK_DET_CONF", 0.03, 0.20, step=0.01), 4
        ))

    env["SWAY_UMOT_BACKTRACK"] = trial.suggest_categorical("SWAY_UMOT_BACKTRACK", ["0", "1"])
    if env["SWAY_UMOT_BACKTRACK"] == "1":
        env["SWAY_UMOT_HISTORY_LENGTH"] = str(
            trial.suggest_int("SWAY_UMOT_HISTORY_LENGTH", 20, 200, step=10)
        )

    return env


def suggest_future_env_for_trial(
    trial,
    sweep_phase: str = "S1",
) -> Dict[str, str]:
    """Suggest future pipeline parameters for an Optuna trial.

    Sweep phases:
      S1 = Tracking (SAM2 stride, COI thresh, MeMoSORT alpha, track max age)
      S2 = Re-ID (fusion weights, EMA alpha, isolation dist)
      S3 = State (partial/dormant thresholds, confidence heatmap)
      S4 = Backward (stitch similarity, collision min tracks)
      S5 = Advanced (MoTE disocclusion, sentinel SBM, UMOT backtrack)
      S6 = Pose (model choice, critique jerk window)
    """
    env: Dict[str, str] = {}

    if sweep_phase == "S1":
        env["SWAY_TRACKER_ENGINE"] = trial.suggest_categorical(
            "tracker_engine", ["sam2mot", "sam2_memosort_hybrid"]
        )
        env["SWAY_SAM2_REINVOKE_STRIDE"] = str(trial.suggest_int(
            "sam2_reinvoke_stride", 10, 60, step=5
        ))
        env["SWAY_COI_MASK_IOU_THRESH"] = str(trial.suggest_float(
            "coi_mask_iou_thresh", 0.15, 0.40, step=0.05
        ))
        env["SWAY_TRACK_MAX_AGE"] = str(trial.suggest_int(
            "track_max_age", 60, 600, step=30
        ))

    elif sweep_phase == "S2":
        env["SWAY_REID_W_PART"] = str(trial.suggest_float(
            "reid_w_part", 0.15, 0.50, step=0.05
        ))
        env["SWAY_REID_W_COLOR"] = str(trial.suggest_float(
            "reid_w_color", 0.05, 0.25, step=0.05
        ))
        env["SWAY_REID_W_SPATIAL"] = str(trial.suggest_float(
            "reid_w_spatial", 0.00, 0.15, step=0.05
        ))
        env["SWAY_REID_EMA_ALPHA_HIGH"] = str(trial.suggest_float(
            "reid_ema_alpha_high", 0.05, 0.30, step=0.05
        ))
        env["SWAY_REID_EMA_ISOLATION_DIST"] = str(trial.suggest_float(
            "reid_ema_isolation_dist", 1.0, 3.0, step=0.25
        ))

    elif sweep_phase == "S3":
        env["SWAY_STATE_PARTIAL_MASK_FRAC"] = str(trial.suggest_float(
            "state_partial_mask_frac", 0.15, 0.50, step=0.05
        ))
        env["SWAY_STATE_DORMANT_MAX_FRAMES"] = str(trial.suggest_int(
            "state_dormant_max_frames", 60, 600, step=30
        ))
        env["SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH"] = str(trial.suggest_float(
            "confidence_heatmap_thresh_high", 0.50, 0.85, step=0.05
        ))
        env["SWAY_CONFIDENCE_HEATMAP_THRESH_MED"] = str(trial.suggest_float(
            "confidence_heatmap_thresh_med", 0.25, 0.55, step=0.05
        ))

    elif sweep_phase == "S4":
        env["SWAY_BACKWARD_STITCH_MIN_SIMILARITY"] = str(trial.suggest_float(
            "backward_stitch_min_similarity", 0.40, 0.80, step=0.05
        ))
        env["SWAY_COLLISION_MIN_TRACKS"] = str(trial.suggest_int(
            "collision_min_tracks", 2, 5
        ))
        env["SWAY_COALESCENCE_IOU_THRESH"] = str(trial.suggest_float(
            "coalescence_iou_thresh", 0.70, 0.95, step=0.05
        ))

    elif sweep_phase == "S5":
        env["SWAY_SAM2_CONFIDENCE_REINVOKE"] = str(trial.suggest_float(
            "sam2_confidence_reinvoke", 0.20, 0.60, step=0.05
        ))
        env["SWAY_SAM2_MEMORY_FRAMES"] = str(trial.suggest_int(
            "sam2_memory_frames", 60, 240, step=30
        ))

    elif sweep_phase == "S6":
        env["SWAY_POSE_MASK_GUIDED"] = str(trial.suggest_categorical(
            "pose_mask_guided", ["0", "1"]
        ))
        env["SWAY_CRITIQUE_JERK_WINDOW"] = str(trial.suggest_int(
            "critique_jerk_window", 3, 9, step=2
        ))

    return env


def _write_params_yaml(env_map: Dict[str, str], path: Path) -> None:
    # main.py: only SWAY_* keys promoted to os.environ from YAML top-level
    block = {k: v for k, v in env_map.items() if k.startswith("SWAY_")}
    path.write_text(yaml.dump(block, default_flow_style=False, sort_keys=True), encoding="utf-8")


def _score_one_video(
    seq_name: str,
    video: Path,
    gt_mot: Path,
    im_w: int,
    im_h: int,
    params_yaml: Path,
    out_dir: Path,
    timeout_s: Optional[int],
    *,
    save_phase_previews: bool = False,
    expected_tracker_engine: Optional[str] = None,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        str(video),
        "--output-dir",
        str(out_dir),
        "--params",
        str(params_yaml),
        "--stop-after-boundary",
        "after_phase_3",
        "--run-manifest",
        str(out_dir / "run_manifest.json"),
    ]
    if save_phase_previews:
        cmd.append("--save-phase-previews")
    kwargs: Dict[str, Any] = {"cwd": str(REPO_ROOT)}
    if timeout_s is not None:
        kwargs["timeout"] = timeout_s
    sub_env = os.environ.copy()
    sub_env.update(subprocess_env_overlay())
    kwargs["env"] = sub_env
    timed_out = False
    timeout_seconds = kwargs.get("timeout")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
        stdout_txt = r.stdout or ""
        stderr_txt = r.stderr or ""
    except subprocess.TimeoutExpired as ex:
        timed_out = True
        stdout_txt = (ex.stdout or "") if isinstance(ex.stdout, str) else ""
        stderr_txt = (ex.stderr or "") if isinstance(ex.stderr, str) else ""
        class _TimeoutResult:
            returncode = 124
        r = _TimeoutResult()
    log_text = stdout_txt + "\n" + stderr_txt
    features = _parse_feature_states(log_text)
    mismatches = _feature_mismatches(features)
    manifest = _load_run_manifest(out_dir)
    runtime_meta = {
        "features": features,
        "requested_runtime_mismatches": mismatches,
        "requested_runtime_mismatch_count": len(mismatches),
        "expected_tracker_engine": expected_tracker_engine,
        "tracker_path": (
            ((manifest.get("run_context_final") or {}).get("pipeline_diagnostics") or {}).get("tracker_path")
        ),
        "global_stitch": (
            ((manifest.get("run_context_final") or {}).get("pipeline_diagnostics") or {}).get("global_stitch")
        ),
    }
    runtime_meta["runtime_signature"] = _runtime_signature_from_meta(runtime_meta)
    tracker_path = str(runtime_meta.get("tracker_path") or "").strip()
    tracker_unknown = (not tracker_path) or tracker_path.lower() == "unknown"
    runtime_meta["tracker_unknown"] = tracker_unknown
    engine_mismatch = False
    if expected_tracker_engine:
        engine_mismatch = not _engine_matches_tracker_path(expected_tracker_engine, tracker_path)
    runtime_meta["engine_mismatch"] = engine_mismatch
    runtime_meta["unknown_runtime_hit"] = bool(tracker_unknown)
    if timed_out:
        runtime_meta["timeout"] = True
        runtime_meta["timeout_seconds"] = timeout_seconds
        runtime_meta["timeout_hit"] = True
    else:
        runtime_meta["timeout_hit"] = False
    data_json = out_dir / "data.json"
    if timed_out:
        return 0.0, {"error": "pipeline_timeout", "timeout_s": timeout_seconds, "stderr_tail": stderr_txt[-800:]}, runtime_meta
    if r.returncode != 0 or not data_json.is_file():
        return 0.0, {"error": "pipeline_failed", "stderr_tail": stderr_txt[-800:]}, runtime_meta

    data = json.loads(data_json.read_text())
    from sway.mot_format import load_mot_lines_from_file
    from sway.trackeval_runner import run_trackeval_single_sequence

    gt_lines = load_mot_lines_from_file(gt_mot)
    from sway.mot_format import data_json_to_mot_lines

    pr_lines = data_json_to_mot_lines(data)
    try:
        flat = run_trackeval_single_sequence(gt_lines, pr_lines, seq_name, im_w, im_h)
    except RuntimeError as e:
        return 0.0, {"error": str(e)}, runtime_meta
    return composite_score(flat), flat, runtime_meta


def load_sweep_config(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    raw = yaml.safe_load(path.read_text()) or {}
    order = raw.get("sequence_order")
    seqs = raw.get("sequences") or {}
    if not order:
        order = list(seqs.keys())
    for name in order:
        if name not in seqs:
            raise ValueError(f"sequence_order references missing sequence: {name}")
    return order, seqs


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna Phase 1–3 multi-video sweep")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "data" / "ground_truth" / "sweep_sequences.yaml",
        help="YAML with sequence_order + sequences (see sweep_sequences.example.yaml)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N completed trials (default: unlimited until signal or stop file)",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="While the sweep runs, create this file to stop after the current trial "
        "(default: output/sweeps/optuna/STOP)",
    )
    parser.add_argument("--study-name", type=str, default="sweep_v2")
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optuna storage URL (default: sqlite under output/sweeps/optuna/)",
    )
    parser.add_argument("--show-best", action="store_true")
    parser.add_argument(
        "--timeout-per-video",
        type=int,
        default=0,
        help="Subprocess timeout seconds per video (0 = none)",
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        help="Append one JSON line per completed trial (default: under output/sweeps/optuna/)",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=None,
        help="Atomic JSON after each trial for live UI (default: output/sweeps/optuna/sweep_status.json)",
    )
    parser.add_argument(
        "--no-status-json",
        action="store_true",
        help="Do not write sweep_status.json",
    )
    parser.add_argument(
        "--phase-previews",
        action="store_true",
        help="Pass --save-phase-previews to each main.py (phase_previews/*.mp4 for Lab / Optuna UI)",
    )
    parser.add_argument(
        "--skip-phase13-coverage-gate",
        action="store_true",
        help="Skip fail-fast coverage gate for phase 1-3 owned keys.",
    )
    args = parser.parse_args()

    import optuna
    from optuna.pruners import NopPruner
    from optuna.samplers import TPESampler

    opt_dir = REPO_ROOT / "output" / "sweeps" / "optuna"
    opt_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(opt_dir / 'sweep.db').resolve()}"

    if args.show_best:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        try:
            print(f"Best value: {study.best_value}")
            print(f"Best params: {study.best_params}")
        except ValueError:
            print("No completed trials yet.", file=sys.stderr)
            sys.exit(1)
        return

    if not args.config.is_file():
        print(
            f"Missing config: {args.config}\n"
            f"Copy {REPO_ROOT / 'data' / 'ground_truth' / 'sweep_sequences.example.yaml'} "
            f"to sweep_sequences.yaml and add your videos.",
            file=sys.stderr,
        )
        sys.exit(2)

    order, seq_specs = load_sweep_config(args.config)
    for sid in order:
        spec = seq_specs[sid]
        v = _resolve_path(spec["video"])
        g = _resolve_path(spec["gt_mot"])
        if not v.is_file():
            print(f"Missing video for {sid}: {v}", file=sys.stderr)
            sys.exit(2)
        if not g.is_file():
            print(f"Missing gt_mot for {sid}: {g}", file=sys.stderr)
            sys.exit(2)

    log_jsonl = args.log_jsonl or (opt_dir / "sweep_log.jsonl")
    status_path: Optional[Path] = None
    if not args.no_status_json:
        status_path = (args.status_json or (opt_dir / "sweep_status.json")).resolve()
    available_weights = _available_yolo_weights()
    stop_path = (args.stop_file if args.stop_file is not None else (opt_dir / "STOP")).resolve()
    coverage_report = phase13_search_space_coverage_report()
    if coverage_report["missing_keys"] and not args.skip_phase13_coverage_gate:
        raise SystemExit(
            "Phase1-3 coverage gate failed. Missing keys: "
            + ", ".join(coverage_report["missing_keys"])
        )
    print(
        "[coverage] phase1-3 owned="
        f"{coverage_report['owned_key_count']} tuned={coverage_report['tuned_key_count']} "
        f"fixed={coverage_report['fixed_key_count']} full={coverage_report['is_full_coverage']}",
        flush=True,
    )

    n_trials_cap = args.n_trials
    if n_trials_cap is not None and n_trials_cap <= 0:
        n_trials_cap = None

    timeout = args.timeout_per_video if args.timeout_per_video > 0 else None
    fail_fast = str(os.environ.get("SWAY_SWEEP_FAIL_FAST", "1")).strip().lower() not in {"0", "false", "no", "off"}
    prune_on_unknown_runtime = (
        str(os.environ.get("SWAY_SWEEP_PRUNE_ON_UNKNOWN_RUNTIME", "1")).strip().lower()
        not in {"0", "false", "no", "off"}
    )
    prune_on_engine_mismatch = (
        str(os.environ.get("SWAY_SWEEP_PRUNE_ON_ENGINE_MISMATCH", "1")).strip().lower()
        not in {"0", "false", "no", "off"}
    )
    floor_score_eps = float(os.environ.get("SWAY_SWEEP_FLOOR_EPS", str(FLOOR_SCORE_EPS_DEFAULT)))
    bigtest_slow_seconds = float(os.environ.get("SWAY_SWEEP_BIGTEST_SLOW_SEC", str(BIGTEST_SLOW_SECONDS_DEFAULT)))

    status_meta: Dict[str, Any] = {
        "config": str(args.config.resolve()),
        "sequence_order": list(order),
        "log_jsonl": str(log_jsonl.resolve()),
        "storage": storage,
        "git_sha": _git_sha_short(),
        "phase13_coverage": coverage_report,
        "fail_fast": {
            "enabled": fail_fast,
            "prune_on_unknown_runtime": prune_on_unknown_runtime,
            "prune_on_engine_mismatch": prune_on_engine_mismatch,
            "floor_score_eps": floor_score_eps,
            "bigtest_slow_seconds": bigtest_slow_seconds,
            "allowed_tracker_engines": _allowed_tracker_engines(),
        },
    }

    seq_weights = [float(seq_specs[n].get("weight", 1.0)) for n in order]

    def objective(trial: optuna.Trial) -> float:
        # ── Duplicate detection: skip if this exact param set already completed ──
        for prev in trial.study.trials:
            if (prev.state == optuna.trial.TrialState.COMPLETE
                    and prev.number != trial.number
                    and prev.params == trial.params):
                print(
                    f"[sweep] Trial {trial.number} duplicates #{prev.number} — pruning",
                    flush=True,
                )
                raise optuna.exceptions.TrialPruned()

        trial_start = time.time()
        env_map = suggest_env_for_trial(trial, available_weights)
        per_video_scores: List[float] = []
        per_video_durations: Dict[str, float] = {}
        all_metrics: Dict[str, Any] = {}
        runtime_by_video: Dict[str, Any] = {}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(opt_dir)
        ) as tf:
            params_path = Path(tf.name)
        try:
            _write_params_yaml(env_map, params_path)

            for step, seq_name in enumerate(order):
                spec = seq_specs[seq_name]
                video = _resolve_path(spec["video"])
                gt_mot = _resolve_path(spec["gt_mot"])
                im_w = int(spec.get("im_width", 1280))
                im_h = int(spec.get("im_height", 720))
                run_dir = opt_dir / f"trial_{trial.number:05d}_{seq_name}"
                _t0 = time.time()
                score, mflat, runtime_meta = _score_one_video(
                    seq_name,
                    video,
                    gt_mot,
                    im_w,
                    im_h,
                    params_path,
                    run_dir,
                    timeout,
                    save_phase_previews=bool(args.phase_previews),
                    expected_tracker_engine=env_map.get("SWAY_TRACKER_ENGINE"),
                )
                _video_dur = round(time.time() - _t0, 1)
                per_video_scores.append(score)
                per_video_durations[seq_name] = _video_dur
                all_metrics[seq_name] = mflat
                runtime_meta["floor_score_hit"] = bool(score <= floor_score_eps)
                runtime_meta["slow_path_hit"] = bool(seq_name == "bigtest" and _video_dur >= bigtest_slow_seconds)
                runtime_by_video[seq_name] = runtime_meta

                # Persist sequence-level health diagnostics even if this trial is pruned early.
                trial.set_user_attr(f"score_{seq_name}", score)
                trial.set_user_attr(f"duration_s_{seq_name}", _video_dur)
                trial.set_user_attr(
                    f"runtime_mismatch_count_{seq_name}",
                    int(runtime_meta.get("requested_runtime_mismatch_count") or 0),
                )
                trial.set_user_attr(f"engine_mismatch_{seq_name}", int(bool(runtime_meta.get("engine_mismatch"))))
                trial.set_user_attr(f"unknown_runtime_hit_{seq_name}", int(bool(runtime_meta.get("unknown_runtime_hit"))))
                trial.set_user_attr(f"timeout_hit_{seq_name}", int(bool(runtime_meta.get("timeout_hit"))))
                trial.set_user_attr(f"floor_score_hit_{seq_name}", int(bool(runtime_meta.get("floor_score_hit"))))
                trial.set_user_attr(f"slow_path_hit_{seq_name}", int(bool(runtime_meta.get("slow_path_hit"))))
                if runtime_meta.get("tracker_path"):
                    trial.set_user_attr(f"runtime_tracker_{seq_name}", str(runtime_meta.get("tracker_path")))

                fail_reasons: List[str] = []
                if bool(runtime_meta.get("timeout_hit")):
                    fail_reasons.append(f"{seq_name}:timeout")
                if bool(runtime_meta.get("floor_score_hit")):
                    fail_reasons.append(f"{seq_name}:floor_score")
                if bool(runtime_meta.get("slow_path_hit")):
                    fail_reasons.append(f"{seq_name}:slow_path")
                if prune_on_unknown_runtime and bool(runtime_meta.get("unknown_runtime_hit")):
                    fail_reasons.append(f"{seq_name}:unknown_runtime")
                if prune_on_engine_mismatch and bool(runtime_meta.get("engine_mismatch")):
                    fail_reasons.append(f"{seq_name}:engine_mismatch")
                if fail_fast and fail_reasons:
                    reason = ",".join(fail_reasons)
                    trial.set_user_attr("fail_fast_pruned", 1)
                    trial.set_user_attr("fail_fast_reason", reason)
                    print(f"[sweep] Trial {trial.number} fail-fast prune ({reason})", flush=True)
                    raise optuna.exceptions.TrialPruned()

            floored = [max(s, 1e-6) for s in per_video_scores]
            # Weighted harmonic mean: bigtest (weight=2.0) counts double
            total_w = sum(seq_weights)
            agg = total_w / sum(w / s for w, s in zip(seq_weights, floored))

            trial_dur = round(time.time() - trial_start, 1)
            for seq_name, sc in zip(order, per_video_scores):
                rm = runtime_by_video.get(seq_name) or {}
                # keep attrs updated on successful completion too
                trial.set_user_attr(f"score_{seq_name}", sc)
                trial.set_user_attr(f"duration_s_{seq_name}", per_video_durations.get(seq_name, 0.0))
            trial.set_user_attr("aggregate_harmonic_mean", agg)
            trial.set_user_attr("trial_duration_s", trial_dur)
            total_mismatch = int(
                sum(int((runtime_by_video.get(n) or {}).get("requested_runtime_mismatch_count") or 0) for n in order)
            )
            total_engine_mismatch = int(sum(int(bool((runtime_by_video.get(n) or {}).get("engine_mismatch"))) for n in order))
            total_unknown_runtime = int(sum(int(bool((runtime_by_video.get(n) or {}).get("unknown_runtime_hit"))) for n in order))
            total_timeout_hit = int(sum(int(bool((runtime_by_video.get(n) or {}).get("timeout_hit"))) for n in order))
            total_floor_hit = int(sum(int(bool((runtime_by_video.get(n) or {}).get("floor_score_hit"))) for n in order))
            total_slow_hit = int(sum(int(bool((runtime_by_video.get(n) or {}).get("slow_path_hit"))) for n in order))
            trial.set_user_attr("runtime_mismatch_count_total", total_mismatch)
            trial.set_user_attr("engine_mismatch", int(total_engine_mismatch > 0))
            trial.set_user_attr("engine_mismatch_count_total", total_engine_mismatch)
            trial.set_user_attr("unknown_runtime_hit", int(total_unknown_runtime > 0))
            trial.set_user_attr("unknown_runtime_hit_count_total", total_unknown_runtime)
            trial.set_user_attr("timeout_hit", int(total_timeout_hit > 0))
            trial.set_user_attr("timeout_hit_count_total", total_timeout_hit)
            trial.set_user_attr("floor_score_hit", int(total_floor_hit > 0))
            trial.set_user_attr("floor_score_hit_count_total", total_floor_hit)
            trial.set_user_attr("slow_path_hit", int(total_slow_hit > 0))
            trial.set_user_attr("slow_path_hit_count_total", total_slow_hit)
            trial.set_user_attr(
                "runtime_signature",
                " || ".join(
                    str((runtime_by_video.get(n) or {}).get("runtime_signature") or f"{n}:none")
                    for n in order
                ),
            )

            entry = {
                "trial": trial.number,
                "git_sha": _git_sha_short(),
                "aggregate": agg,
                "per_video": dict(zip(order, per_video_scores)),
                "per_video_duration_s": per_video_durations,
                "trial_duration_s": trial_dur,
                "env": env_map,
                "actual_runtime": runtime_by_video,
            }
            with open(log_jsonl, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(entry, default=str) + "\n")

            return agg
        finally:
            params_path.unlink(missing_ok=True)
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(
            seed=42,
            # multivariate=True: models correlations between params (e.g. tracker
            # type and its sub-params are highly correlated).  Significantly better
            # for conditional search spaces like ours.
            multivariate=True,
            # n_startup_trials: number of quasi-random trials before TPE switches
            # to Bayesian optimisation.  With 25-35 active params per trial and
            # 8 tracker types, 50 gives each tracker ~6 random explorations before
            # exploitation begins.
            n_startup_trials=50,
        ),
        pruner=NopPruner(),  # no pruning — all videos always run, all scores recorded
    )

    if status_path:
        try:
            from sway.optuna_live_status import write_live_sweep_status

            write_live_sweep_status(study, status_path, extra=status_meta)
            print(f"Live status JSON: {status_path}", flush=True)
        except Exception as ex:
            print(f"(status-json initial write failed: {ex})", flush=True)

    def _on_signal(_signum: int, _frame: Any) -> None:
        print(
            "\nStop requested — current trial will finish, then the sweep exits.",
            flush=True,
        )
        study.stop()

    try:
        signal.signal(signal.SIGINT, _on_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _on_signal)
    except ValueError:
        # e.g. not main thread
        pass

    def _after_trial(study_obj: Any, _finished_trial: Any) -> None:
        if status_path:
            try:
                from sway.optuna_live_status import write_live_sweep_status

                write_live_sweep_status(study_obj, status_path, extra=status_meta)
            except Exception as ex:
                print(f"(status-json write failed: {ex})", flush=True)
        if stop_path.is_file():
            print(
                f"\nStop file detected ({stop_path}); stopping after this trial.",
                flush=True,
            )
            study_obj.stop()
            try:
                stop_path.unlink(missing_ok=True)
            except OSError:
                pass

    if n_trials_cap is None:
        print(
            "Sweep: unlimited trials. Stop with Ctrl+C / SIGTERM, or: "
            f"touch {stop_path}",
            flush=True,
        )
    else:
        print(f"Sweep: will stop after {n_trials_cap} trial(s).", flush=True)

    study.optimize(objective, n_trials=n_trials_cap, callbacks=[_after_trial])

    print("=== Done ===")
    try:
        print(f"Best value (harmonic mean): {study.best_value}")
        print(f"Best params: {study.best_params}")
    except ValueError:
        print("No completed trials in this study yet.", file=sys.stderr)
    try:
        import optuna.importance

        imp = optuna.importance.get_param_importances(study)
        print("Param importances:", imp)
    except Exception as ex:
        print("(importance analysis skipped:", ex, ")")


if __name__ == "__main__":
    main()
