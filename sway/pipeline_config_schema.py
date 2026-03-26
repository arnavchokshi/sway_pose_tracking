"""
Pipeline Lab: schema of tunable parameters grouped by pipeline phase.

Used by pipeline_lab server (/api/schema) and the React UI. Many entries map to
environment variables read in tracker.py or main.py; YAML params can also set
any key starting with SWAY_ (applied in main.py before tracking).

Stage order and ``main_phases`` match ``docs/PIPELINE_CODE_REFERENCE.md`` and
``main.py`` progress lines ``[1/11]`` … ``[11/11]`` (Lab groups some steps into
one tab, e.g. phases 6–7 in ``reid_dedup``). Printed **Phase 4** has no Lab tab.

Only options that change behavior are listed; read-only ``info`` rows document
what the pipeline always does so the UI does not offer unwired presets.
Phase 4 (pre-pose prune) has no Lab stage: YAML knobs are master-locked (see
``MASTER_LOCKED_PRE_POSE_PRUNE_PARAMS``). Five Phase 6–7 re-ID / collocated-dedup YAML keys are master-locked
(``MASTER_LOCKED_REID_DEDUP_PARAMS``); finer dedup sliders stay in the Lab UI.
Tier C / Tier A span / selected Tier B thresholds and three pruning weights are
master-locked for Phase 8 (``MASTER_LOCKED_POST_POSE_PRUNE_*``); sync, vote
threshold, mirror, and low-conf weights stay tunable.
Phase 9 locks ``SMOOTHER_MIN_CUTOFF``, ``SWAY_TEMPORAL_POSE_RADIUS`` (neighbor-blend window
when enabled), and **neighbor blend off** (``SWAY_TEMPORAL_POSE_REFINE``); ``SMOOTHER_BETA``
stays tunable. Lab UI hides the neighbor-blend toggle; batch/matrix YAML may still set
``temporal_pose_refine: true`` and the Lab worker honors it. Use ``SWAY_UNLOCK_SMOOTH_TUNING=1`` to override
locked cutoff/radius from CLI or to tune without the fixed stack.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

# Canonical person-detection stack (Spot people phase). Enforced in ``main.py`` after params YAML;
# Pipeline Lab subprocess env is frozen via ``freeze_lab_subprocess_detection_env`` in ``app.py``.
# Set ``SWAY_UNLOCK_DETECTION_TUNING=1`` to allow smoke matrices / advanced overrides.
MASTER_LOCKED_DETECTION_ENV: Dict[str, str] = {
    "SWAY_GROUP_VIDEO": "1",
    "SWAY_DETECT_SIZE": "640",
    "SWAY_CHUNK_SIZE": "300",
    "SWAY_YOLO_INFER_BATCH": "1",
    "SWAY_YOLO_HALF": "0",
    "SWAY_GSI_LENGTHSCALE": "0.35",
}


def apply_master_locked_detection_env() -> None:
    """Force ``MASTER_LOCKED_DETECTION_ENV`` on ``os.environ``; clear ``SWAY_YOLO_ENGINE``."""
    v = os.environ.get("SWAY_UNLOCK_DETECTION_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    for key, val in MASTER_LOCKED_DETECTION_ENV.items():
        os.environ[key] = val
    os.environ.pop("SWAY_YOLO_ENGINE", None)


def freeze_lab_subprocess_detection_env(env: Dict[str, str]) -> None:
    """Lab API: always apply master detection env to the child process (TensorRT path cleared)."""
    for key, val in MASTER_LOCKED_DETECTION_ENV.items():
        env[key] = val
    env.pop("SWAY_YOLO_ENGINE", None)


# Hybrid SAM overlap tab (BoxMOT path). Enforced after params YAML; Lab freezes subprocess env.
# Set ``SWAY_UNLOCK_HYBRID_SAM_TUNING=1`` for smoke runs / A-B that disable SAM or use full-frame ROI.
MASTER_LOCKED_HYBRID_SAM_ENV: Dict[str, str] = {
    "SWAY_HYBRID_SAM_OVERLAP": "1",
    "SWAY_HYBRID_SAM_ROI_CROP": "1",
    "SWAY_HYBRID_SAM_ROI_PAD_FRAC": "0.1",
}


def apply_master_locked_hybrid_sam_env() -> None:
    """Force ``MASTER_LOCKED_HYBRID_SAM_ENV`` on ``os.environ``."""
    v = os.environ.get("SWAY_UNLOCK_HYBRID_SAM_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    for key, val in MASTER_LOCKED_HYBRID_SAM_ENV.items():
        os.environ[key] = val


def freeze_lab_subprocess_hybrid_sam_env(env: Dict[str, str]) -> None:
    """Lab API: always apply master hybrid SAM env to the child process."""
    for key, val in MASTER_LOCKED_HYBRID_SAM_ENV.items():
        env[key] = val


# Phase 3 global stitch / AFLink thresholds. ``sway_global_aflink_mode`` stays in the Lab UI.
# Set ``SWAY_UNLOCK_PHASE3_STITCH_TUNING=1`` to allow smoke tests (e.g. global link off).
MASTER_LOCKED_PHASE3_STITCH_ENV: Dict[str, str] = {
    "SWAY_GLOBAL_LINK": "1",
    "SWAY_AFLINK_THR_T0": "0",
    "SWAY_AFLINK_THR_T1": "30",
    "SWAY_AFLINK_THR_S": "75",
    "SWAY_AFLINK_THR_P": "0.05",
}


def apply_master_locked_phase3_stitch_env() -> None:
    """Force ``MASTER_LOCKED_PHASE3_STITCH_ENV`` on ``os.environ``."""
    v = os.environ.get("SWAY_UNLOCK_PHASE3_STITCH_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    for key, val in MASTER_LOCKED_PHASE3_STITCH_ENV.items():
        os.environ[key] = val


def freeze_lab_subprocess_phase3_stitch_env(env: Dict[str, str]) -> None:
    """Lab API: always apply master Phase 3 stitch env to the child process."""
    for key, val in MASTER_LOCKED_PHASE3_STITCH_ENV.items():
        env[key] = val


# Phase 5 pose stack (ViTPose path + 3D lift). ``pose_model`` / visibility stay in the Lab UI; pose stride is fixed
# to every frame for Lab (see ``pose_stride`` + ``lab_hidden`` in ``PIPELINE_PARAM_FIELDS``).
# Set ``SWAY_UNLOCK_POSE_TUNING=1`` for smoke (3D off, chunked ViTPose) or FP32 A/B. Smart pad is not
# then forced to ``1``; ``vitpose_smart_pad_enabled()`` still defaults on unless ``SWAY_VITPOSE_SMART_PAD=0``.
def apply_master_locked_pose_env() -> None:
    """Unset chunked ViTPose cap; FP32 off; 3D lift on; ViTPose smart bbox pad on.

    Unset cap keeps one batched forward on CUDA. On MPS, ``pose_estimator`` applies a safe default
    chunk size when the env var is absent (see ``vitpose_effective_max_per_forward``).
    """
    v = os.environ.get("SWAY_UNLOCK_POSE_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    os.environ.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)
    os.environ["SWAY_VITPOSE_FP32"] = "0"
    lift = os.environ.get("SWAY_3D_LIFT", "").strip().lower()
    if lift not in ("0", "false", "no"):
        os.environ["SWAY_3D_LIFT"] = "1"
    os.environ["SWAY_VITPOSE_SMART_PAD"] = "1"


def freeze_lab_subprocess_pose_env(env: Dict[str, str]) -> None:
    """Lab API: apply master pose env (0 = all people one ViTPose batch via unset cap)."""
    env.pop("SWAY_VITPOSE_MAX_PER_FORWARD", None)
    env["SWAY_VITPOSE_FP32"] = "0"
    env["SWAY_3D_LIFT"] = "1"
    env["SWAY_VITPOSE_SMART_PAD"] = "1"


# Phase 4 YAML keys that ``main.py`` reads from ``params`` for pre-pose pruning. Pipeline Lab does
# not expose these; values are merged into every ``params.yaml`` and forced after loading any YAML
# unless ``SWAY_UNLOCK_PRE_POSE_PRUNE_TUNING=1``. (Bbox size / aspect / mirror rules in Phase 4
# still use module constants only — not included here.)
MASTER_LOCKED_PRE_POSE_PRUNE_PARAMS: Dict[str, Any] = {
    "min_duration_ratio": 0.20,
    "KINETIC_STD_FRAC": 0.02,
    "SPATIAL_OUTLIER_STD_FACTOR": 2.0,
    "SHORT_TRACK_MIN_FRAC": 0.15,
    "AUDIENCE_REGION_X_MIN_FRAC": 0.75,
    "AUDIENCE_REGION_Y_MIN_FRAC": 0.70,
}


def apply_master_locked_pre_pose_prune_params(params: Dict[str, Any]) -> None:
    """Overwrite Phase 4 prune keys in ``params`` (mutates in place)."""
    v = os.environ.get("SWAY_UNLOCK_PRE_POSE_PRUNE_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    params.update(MASTER_LOCKED_PRE_POSE_PRUNE_PARAMS)


# Phase 6 occlusion re-ID + Phase 7 collocated-dedup distance gates (YAML via ``params``).
# Set ``SWAY_UNLOCK_REID_DEDUP_TUNING=1`` for sweeps / deliberate overrides.
# ``dedup_min_pair_oks`` and ``dedup_antipartner_min_iou`` remain Lab-tunable.
MASTER_LOCKED_REID_DEDUP_PARAMS: Dict[str, Any] = {
    "REID_MAX_FRAME_GAP": 90,
    "REID_MIN_OKS": 0.35,
    "COLLISION_KPT_DIST_FRAC": 0.26,
    "COLLISION_CENTER_DIST_FRAC": 0.5,
    "DEDUP_TORSO_MEDIAN_FRAC": 0.24,
}


def apply_master_locked_reid_dedup_params(params: Dict[str, Any]) -> None:
    """Overwrite Phase 6–7 locked re-ID / dedup keys in ``params`` (mutates in place)."""
    v = os.environ.get("SWAY_UNLOCK_REID_DEDUP_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    params.update(MASTER_LOCKED_REID_DEDUP_PARAMS)


# Phase 8 post-pose prune: Tier C, Tier A span, Tier B garbage thresholds, and three vote weights.
# Set ``SWAY_UNLOCK_POST_POSE_PRUNE_TUNING=1`` for sweeps. ``PRUNE_THRESHOLD`` is fixed in the Lab UI
# (see ``LAB_UI_ENFORCED_DEFAULTS``). Remaining tunable YAML (advanced / unlock): ``SYNC_SCORE_MIN``,
# ``pruning_w_low_sync``, ``pruning_w_smart_mirror``, ``pruning_w_low_conf``.
MASTER_LOCKED_POST_POSE_PRUNE_SCALAR_PARAMS: Dict[str, Any] = {
    "CONFIRMED_HUMAN_MIN_SPAN_FRAC": 0.10,
    "TIER_C_SKELETON_MEAN": 0.15,
    "TIER_C_LOW_FRAME_FRAC": 0.80,
    "MEAN_CONFIDENCE_MIN": 0.45,
    "EDGE_MARGIN_FRAC": 0.15,
    "EDGE_PRESENCE_FRAC": 0.30,
    "min_lower_body_conf": 0.30,
    "JITTER_RATIO_MAX": 0.10,
}

MASTER_LOCKED_POST_POSE_PRUNE_WEIGHT_KEYS: Dict[str, float] = {
    "prune_completeness_audit": 0.6,
    "prune_head_only_tracks": 0.8,
    "prune_jittery_tracks": 0.5,
}


def apply_master_locked_post_pose_prune_params(params: Dict[str, Any]) -> None:
    """Overwrite Phase 8 locked keys in ``params`` and merge locked pruning weights (mutates in place)."""
    v = os.environ.get("SWAY_UNLOCK_POST_POSE_PRUNE_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    params.update(MASTER_LOCKED_POST_POSE_PRUNE_SCALAR_PARAMS)
    base_pw = params.get("PRUNING_WEIGHTS")
    if isinstance(base_pw, dict):
        merged = {str(k): float(v) for k, v in base_pw.items()}
    else:
        merged = {}
    for k, val in MASTER_LOCKED_POST_POSE_PRUNE_WEIGHT_KEYS.items():
        merged[k] = float(val)
    params["PRUNING_WEIGHTS"] = merged


# Phase 9: 1-Euro min cutoff (YAML) + temporal keypoint refine radius + neighbor blend off (env).
# Set ``SWAY_UNLOCK_SMOOTH_TUNING=1`` for sweeps, wider neighbor windows, or forcing neighbor blend on from CLI.
MASTER_LOCKED_SMOOTH_PARAMS: Dict[str, Any] = {
    "SMOOTHER_MIN_CUTOFF": 1.0,
}

MASTER_LOCKED_SMOOTH_ENV: Dict[str, str] = {
    "SWAY_TEMPORAL_POSE_RADIUS": "2",
    "SWAY_TEMPORAL_POSE_REFINE": "0",
}


def apply_master_locked_smooth_params(params: Dict[str, Any]) -> None:
    """Overwrite Phase 9 locked 1-Euro cutoff in ``params`` (mutates in place)."""
    v = os.environ.get("SWAY_UNLOCK_SMOOTH_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    params.update(MASTER_LOCKED_SMOOTH_PARAMS)


def apply_master_locked_smooth_env() -> None:
    """Force neighbor-blend radius and ``SWAY_TEMPORAL_POSE_REFINE=0`` on ``os.environ``."""
    v = os.environ.get("SWAY_UNLOCK_SMOOTH_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    for key, val in MASTER_LOCKED_SMOOTH_ENV.items():
        os.environ[key] = val


def freeze_lab_subprocess_smooth_env(env: Dict[str, str], fields: Dict[str, Any] | None = None) -> None:
    """Lab API: apply master Phase 9 env to the child process.

    Neighbor blend defaults **off** (``SWAY_TEMPORAL_POSE_REFINE=0``). When ``fields`` is
    provided and ``temporal_pose_refine`` is true (matrix / tree recipes), set ``...=1`` so
    ``--temporal-pose-refine`` is not overridden.
    """
    v = os.environ.get("SWAY_UNLOCK_SMOOTH_TUNING", "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    env["SWAY_TEMPORAL_POSE_RADIUS"] = MASTER_LOCKED_SMOOTH_ENV["SWAY_TEMPORAL_POSE_RADIUS"]
    if fields and fields.get("temporal_pose_refine") is True:
        env["SWAY_TEMPORAL_POSE_REFINE"] = "1"
    else:
        env["SWAY_TEMPORAL_POSE_REFINE"] = MASTER_LOCKED_SMOOTH_ENV["SWAY_TEMPORAL_POSE_REFINE"]


# Intent-based Lab stages (end-user grouping). Technical ``main.py`` phases are documented in
# ``docs/MASTER_PIPELINE_GUIDELINE.md``; field rows still carry legacy phase semantics in code comments.
PIPELINE_STAGES: List[Dict[str, Any]] = [
    {
        "id": "crowd_control",
        "label": "Crowd control — how aggressively we find people",
        "short": "Crowd",
        # Printed phases in main.py progress [n/11] — see docs/MASTER_PIPELINE_GUIDELINE.md §5
        "main_phases": "1–2",
    },
    {
        "id": "handshake",
        "label": "The handshake — collisions, memory, and cross-clip identity",
        "short": "Handshake",
        "main_phases": "1–3, 6–7",
    },
    {
        "id": "pose_polish",
        "label": "Pose & polish — skeleton detail and motion feel",
        "short": "Pose",
        "main_phases": "5, 9",
    },
    {
        "id": "cleanup_export",
        "label": "Cleanup & export",
        "short": "Export",
        "main_phases": "8, 10–11",
    },
]


def _f(
    id_: str,
    phase: str,
    label: str,
    type_: str,
    default: Any,
    *,
    binding: str,
    key: str,
    description: str = "",
    min_: Any = None,
    max_: Any = None,
    choices: Any = None,
    advanced: bool = False,
    tier: int = 2,
    visible_when_field: str = "",
    visible_when_value: Any = None,
    display: str = "",
    disabled_choices: Any = None,
    lab_hidden: bool = False,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "id": id_,
        "phase": phase,
        "label": label,
        "type": type_,
        "default": default,
        "binding": binding,
        "key": key,
        "description": description,
        "advanced": advanced,
        "tier": int(tier),
    }
    if min_ is not None:
        d["min"] = min_
    if max_ is not None:
        d["max"] = max_
    if choices is not None:
        d["choices"] = choices
    if visible_when_field:
        d["visible_when_field"] = visible_when_field
        d["visible_when_value"] = visible_when_value
    if display:
        d["display"] = display
    if disabled_choices is not None:
        d["disabled_choices"] = disabled_choices
    if lab_hidden:
        d["lab_hidden"] = True
    return d


# Single list: UI groups by `phase` (order here matches coarse pipeline left-to-right).
PIPELINE_PARAM_FIELDS: List[Dict[str, Any]] = [
    # --- detection (main.py phases 1–2, YOLO side) ---
    _f(
        "info_detection_yolo",
        "detection",
        "What this step does",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "The pipeline draws a box around each person it thinks is in the frame. "
            "You pick how sharp that finder is (small / large / extra-large model). "
            "If a model file is missing, it may download the first time you run. "
            "Power users can point to a custom weights file via the environment."
        ),
    ),
    _f(
        "info_detection_master_locked",
        "detection",
        "Lock these down (set and forget)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "The master pipeline fixes these for every run — they are **not** configurable in the Lab UI. "
            "``main.py`` reapplies them after ``params`` YAML. For engineering smoke tests that must override "
            "(batch size, crowd off, etc.), set ``SWAY_UNLOCK_DETECTION_TUNING=1`` in the environment.\n\n"
            "**Crowd mode (many people on screen):** On. Under the hood this makes the detector work harder for "
            "smaller bodies; with crowd mode on, the runtime uses letterbox **max(640, 960)** (see ``tracker.py``) "
            "even though the baseline width is **640**.\n\n"
            "**Internal resize width for finding people:** **640** — safe baseline; crowd mode may bump effective size as above.\n\n"
            "**Frames processed per memory chunk:** **300** — streaming memory only; does not change where boxes land.\n\n"
            "**YOLO GPU batch size:** **1** — larger values feed the GPU faster but increase CUDA OOM risk without improving tracking accuracy.\n\n"
            "**YOLO FP16 inference on CUDA:** **Full precision** (half precision off).\n\n"
            "**TensorRT engine path (YOLO):** Unset — use standard ``.pt`` weights unless you have a matching engine for your GPU.\n\n"
            "**GSI smoothness (time, normalized):** **0.35** — tuned lengthscale when optional GSI interpolation is used "
            "(box stride gaps, stitch, pose gaps, export tween)."
        ),
    ),
    _f(
        "sway_yolo_weights",
        "detection",
        "How sharp the person finder is",
        "enum",
        "yolo26l_dancetrack",
        binding="env",
        key="SWAY_YOLO_WEIGHTS",
        choices=[
            "yolo26s",
            "yolo26l_dancetrack",
            "yolo26l_dancetrack_crowdhuman",
            "yolo26x",
        ],
        tier=1,
        display="model_cards",
        description=(
            "Smaller = faster run, may miss tiny or distant people. "
            "Larger = slower, usually better on hard clips and finals footage. "
            "DanceTrack is a YOLO26l fine-tune for group dance (needs models/yolo26l_dancetrack.pt). "
            "The lab dims a card when that weight file is missing."
        ),
    ),
    _f(
        "sway_pretrack_nms_iou",
        "detection",
        "How aggressively to merge overlapping boxes",
        "float",
        0.75,
        binding="env",
        key="SWAY_PRETRACK_NMS_IOU",
        min_=0.40,
        max_=0.90,
        tier=2,
        display="slider",
        description=(
            "When two boxes sit on top of each other, one is thrown away before tracking. "
            "Lower = merge more (good for tight formations, fewer double-counts). "
            "Higher = keep more separate boxes (good if people stand close but are distinct)."
        ),
    ),
    _f(
        "sway_yolo_conf",
        "detection",
        "Minimum “I’m sure this is a person” score",
        "float",
        0.22,
        binding="env",
        key="SWAY_YOLO_CONF",
        min_=0.05,
        max_=0.95,
        tier=2,
        display="slider",
        description=(
            "Higher = fewer boxes, less junk, but you might miss dim or distant dancers. "
            "Lower = more boxes, more risk of random objects being treated as people."
        ),
    ),
    _f(
        "sway_yolo_detection_stride",
        "detection",
        "Run person-finding every Nth frame",
        "int",
        1,
        binding="env",
        key="SWAY_YOLO_DETECTION_STRIDE",
        min_=1,
        max_=8,
        description=(
            "1 = every frame (best accuracy). "
            "2+ = skip frames between runs; the tracker fills the gap—faster but can wobble on fast motion."
        ),
        advanced=True,
        tier=3,
        display="slider",
    ),
    _f(
        "sway_box_interp_mode",
        "detection",
        "Box path between YOLO stride anchors",
        "enum",
        "linear",
        binding="env",
        key="SWAY_BOX_INTERP_MODE",
        choices=["linear", "gsi"],
        advanced=True,
        tier=3,
        display="segmented",
        description=(
            "When YOLO runs every Nth frame, skipped frames get filled boxes before pose. "
            "**Linear** (default) is the choreography-grade choice per the Pose tab verdict; GSI is optional smoothing "
            "between anchors (also used when stitching track fragments), not a substitute for dense detection cadence."
        ),
    ),
    # --- tracking (main.py phase 2) ---
    _f(
        "info_tracking_backends",
        "tracking",
        "What this step does",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "After each frame knows where people are, this step decides “this box is still dancer 3.” "
            "Deep OC-SORT is tuned for dance. You can add track-time OSNet appearance matching when outfits differ. "
            "Overlap sharpening (IoU trigger and optional weak-cue gate below) runs on this same path."
        ),
    ),
    _f(
        "tracker_technology",
        "tracking",
        "How IDs are carried frame to frame",
        "enum",
        "deep_ocsort",
        binding="none",
        key="",
        choices=["deep_ocsort", "deep_ocsort_osnet", "bytetrack"],
        tier=1,
        display="tracker_strip",
        description=(
            "**Default:** Deep OC-SORT with motion/IoU association (no track-time Re-ID). "
            "**+ OSNet:** same tracker with OSNet embeddings during tracking — better through crosses/occlusions "
            "when outfits look different; prefetch ``models/osnet_x0_25_msmt17.pt`` (or set weights via advanced preset). "
            "**ByteTrack:** faster motion-only association (no appearance embeddings); **Fast preview** uses this and turns overlap SAM off."
        ),
    ),
    _f(
        "sway_phase13_mode",
        "tracking",
        "Phases 1–3 strategy (detection → track → stitch)",
        "enum",
        "standard",
        binding="env",
        key="SWAY_PHASE13_MODE",
        choices=["standard", "dancer_registry", "sway_handshake"],
        tier=1,
        display="phase13_mode_cards",
        description=(
            "**Standard:** Baseline YOLO + Deep OC-SORT + hybrid SAM when overlap exceeds your IoU trigger + master-locked stitch. "
            "**Dancer registry (experimental):** Same hybrid SAM / overlap machinery as standard unless you change env; adds "
            "zonal HSV crossover verify and appearance-based dormant relink (full video scans — see ``docs/PHASE13_LAB_STRATEGIES.md``). "
            "Recipe preset tightens pre-track NMS and pins DanceTrack weights + ViTPose-Base. "
            "**Sway handshake (experimental):** Open-floor zonal registry + optional Hungarian reorder of SAM det rows; "
            "Lab server forces hybrid IoU trigger **0.10** and weak cues **off** at enqueue."
        ),
    ),
    _f(
        "sway_bidirectional_track_pass",
        "tracking",
        "Extra pass: track reversed video and merge IDs",
        "bool",
        False,
        binding="env",
        key="SWAY_BIDIRECTIONAL_TRACK_PASS",
        advanced=True,
        tier=3,
        description=(
            "Off by default. When on, ffmpeg reverses the clip, the pipeline runs Phases 1–2 again, "
            "then merges boxes with the forward pass (IoU + minimum matched frames). "
            "Roughly doubles tracking time; needs ffmpeg on PATH."
        ),
    ),
    _f(
        "sway_gnn_track_refine",
        "tracking",
        "GNN-style track refine pass (after Phase 3 stitch)",
        "bool",
        False,
        binding="env",
        key="SWAY_GNN_TRACK_REFINE",
        advanced=True,
        tier=3,
        description=(
            "Default off. When on, ``main.py`` runs post-stitch **graph refinement**: edge-conditioned multi-head GAT "
            "over trajectories + link logits (structural prior + optional ``SWAY_GNN_WEIGHTS``). "
            "Tune ``SWAY_GNN_MERGE_THRESH``, ``SWAY_GNN_MAX_GAP``, etc. See ``sway/gnn_track_refine.py``."
        ),
    ),
    _f(
        "sway_boxmot_max_age",
        "tracking",
        "How long to remember someone who vanishes",
        "int",
        150,
        binding="env",
        key="SWAY_BOXMOT_MAX_AGE",
        min_=60,
        max_=300,
        tier=2,
        display="slider",
        description=(
            "If a dancer walks behind someone or leaves the frame, the pipeline can keep their ID warm for this many frames. "
            "Higher = better when people dip in and out; lower = less chance the wrong person inherits the ID later. "
            "Rough guide: 150 frames ≈ 5 seconds at 30 fps."
        ),
    ),
    _f(
        "sway_boxmot_match_thresh",
        "tracking",
        "How picky matching is between frames",
        "float",
        0.30,
        binding="env",
        key="SWAY_BOXMOT_MATCH_THRESH",
        min_=0.20,
        max_=0.50,
        tier=2,
        display="slider",
        description=(
            "Lower = easier to glue boxes together (fewer ID swaps, risk of merging two people). "
            "Higher = stricter match (cleaner separation, more risk of a new ID when boxes jump)."
        ),
    ),
    _f(
        "sway_boxmot_reid_model",
        "tracking",
        "OSNet checkpoint for track-time Re-ID",
        "enum",
        "osnet_x0_25",
        binding="reid_model_preset",
        key="",
        choices=["osnet_x0_25"],
        visible_when_field="tracker_technology",
        visible_when_value="deep_ocsort_osnet",
        advanced=True,
        tier=2,
        description=(
            "Used only when **+ OSNet** is selected. ``osnet_x0_25`` (lightweight) is the preset "
            "from ``python -m tools.prefetch_models`` → ``models/osnet_x0_25_msmt17.pt``."
        ),
    ),
    # --- overlap / hybrid SAM (same Lab step as tracking; runs inside phases 1–2; BoxMOT only) ---
    _f(
        "info_hybrid_sam_master_locked",
        "tracking",
        "Lock these down (set and forget)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "The master pipeline fixes these for every run — they are **not** configurable in the Lab UI. "
            "``main.py`` reapplies them after ``params`` YAML. For fast smoke tests or deliberate A/B "
            "(e.g. SAM off), set ``SWAY_UNLOCK_HYBRID_SAM_TUNING=1`` in the environment.\n\n"
            "**Refine boxes when people overlap:** On — this is the main reason to use the BoxMOT path with hybrid SAM; "
            "turning it off skips this entire refinement path except for intentional draft runs.\n\n"
            "**Run SAM on overlap union crop (not full frame):** On — SAM runs on a tight crop around touching dancers, "
            "not the full 1920×1080 frame; same tracker math, far less VRAM. Do not use full-frame SAM in production.\n\n"
            "**ROI margin around overlapped dancers:** **0.1** — padding fraction on the union crop so SAM sees a little "
            "context; tuned default; not a tracking-accuracy knob."
        ),
    ),
    _f(
        "sway_hybrid_sam_iou_trigger",
        "tracking",
        "How much overlap before the extra pass runs",
        "float",
        0.42,
        binding="env",
        key="SWAY_HYBRID_SAM_IOU_TRIGGER",
        min_=0.10,
        max_=0.65,
        tier=1,
        display="slider",
        description=(
            "Higher = only very overlapped pairs get fixed (faster, less segmentation). "
            "Lower = fix sooner (slower, cleaner lifts and partner work). "
            "Sway Handshake preset uses **0.10** (IoU custody trigger). "
            "With ROI crop on, lower values are cheaper than full-frame SAM."
        ),
    ),
    _f(
        "sway_hybrid_sam_weak_cues",
        "tracking",
        "Skip SAM when overlap is high but boxes match the last frame",
        "bool",
        False,
        binding="env",
        key="SWAY_HYBRID_SAM_WEAK_CUES",
        advanced=True,
        tier=2,
        description=(
            "Optional Hybrid-SORT–style gate: after IoU says “maybe run SAM,” compare detections to the "
            "previous output. If the worst-overlap pair is height- and confidence-stable, SAM is skipped."
        ),
    ),
    _f(
        "sway_hybrid_weak_conf_delta",
        "tracking",
        "Weak cue: max |Δ confidence| vs matched previous box",
        "float",
        0.08,
        binding="env",
        key="SWAY_HYBRID_WEAK_CONF_DELTA",
        min_=0.02,
        max_=0.35,
        advanced=True,
        tier=3,
        display="slider",
        visible_when_field="sway_hybrid_sam_weak_cues",
        visible_when_value=True,
    ),
    _f(
        "sway_hybrid_weak_height_frac",
        "tracking",
        "Weak cue: max relative height change vs matched previous box",
        "float",
        0.12,
        binding="env",
        key="SWAY_HYBRID_WEAK_HEIGHT_FRAC",
        min_=0.03,
        max_=0.45,
        advanced=True,
        tier=3,
        display="slider",
        visible_when_field="sway_hybrid_sam_weak_cues",
        visible_when_value=True,
    ),
    _f(
        "sway_hybrid_weak_match_iou",
        "tracking",
        "Weak cue: min IoU to match a box to the previous frame",
        "float",
        0.25,
        binding="env",
        key="SWAY_HYBRID_WEAK_MATCH_IOU",
        min_=0.10,
        max_=0.55,
        advanced=True,
        tier=3,
        display="slider",
        visible_when_field="sway_hybrid_sam_weak_cues",
        visible_when_value=True,
    ),
    # --- phase3_stitch (main.py phase 3) ---
    _f(
        "info_phase3_stitch",
        "phase3_stitch",
        "What this step does",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "Sometimes the tracker splits one dancer into two IDs or loses them for a stretch. "
            "This pass tries to glue those pieces back together before body pose runs. "
            "When the optional neural linker model is installed, it can make smarter long-range guesses; "
            "otherwise simple rules are used."
        ),
    ),
    _f(
        "info_phase3_master_locked",
        "phase3_stitch",
        "Lock these down (set and forget)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "The master pipeline fixes these for every run — they are **not** configurable in the Lab UI. "
            "``main.py`` reapplies them after ``params`` YAML. For smoke tests that disable long-range merge, "
            "set ``SWAY_UNLOCK_PHASE3_STITCH_TUNING=1`` in the environment.\n\n"
            "**Try to reconnect IDs across long gaps:** On — skipping this removes the whole long-range safety net; "
            "keep it on unless the clip is unnaturally clean.\n\n"
            "**AFLink advanced sliders (min gap, max gap, max spatial distance, link probability):** Locked to "
            "StrongSORT-style defaults **0, 30, 75, 0.05**. These bound the neural linker’s search; changing them "
            "during baseline A/B makes failures ambiguous (model vs. choked radius). Use the **smart linker** "
            "switch below to test neural vs. heuristic, not these internal thresholds."
        ),
    ),
    _f(
        "sway_stitch_max_frame_gap",
        "phase3_stitch",
        "Longest gap (frames) to bridge broken IDs",
        "int",
        60,
        binding="env",
        key="SWAY_STITCH_MAX_FRAME_GAP",
        min_=30,
        max_=150,
        tier=2,
        display="slider",
        description=(
            "If someone disappears briefly, the pipeline can still merge old and new IDs if the gap is under this. "
            "Raise for slow exits and re-entries; lower on short clips to avoid merging two different people. "
            "About 60 frames ≈ 2 seconds at 30 fps."
        ),
    ),
    _f(
        "sway_global_aflink_mode",
        "phase3_stitch",
        "How smart the long-range merge should be",
        "enum",
        "neural_if_available",
        binding="none",
        key="",
        choices=["neural_if_available", "force_heuristic"],
        tier=1,
        display="segmented",
        description=(
            "Use the learned linker when its weights are present (recommended). "
            "Or always use simple geometry-based rules if you want predictable, lighter behavior."
        ),
    ),
    # --- pose (main.py phase 5) ---
    _f(
        "info_pose_choreography_verdict",
        "pose",
        "Master verdict: cadence & interpolation",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "**The Verdict:** Kill the GSI Speed Cheat. Your choreography is too fast and complex to skip frames.\n\n"
            "**The Lock:** Permanently lock in ``pose_stride`` = **1** (every single frame in the Lab) and keep "
            "your interpolation on **linear** (the default)—box stride gaps, pose gap fill (CLI stride 2 only), "
            "and export video tween."
        ),
    ),
    _f(
        "info_pose_models",
        "pose",
        "What this step does",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "For each person box, the pipeline estimates where shoulders, hips, knees, etc. sit in the image. "
            "That’s what you see as the skeleton overlay. "
            "Dim joints can be ignored so random noise doesn’t score as movement."
        ),
    ),
    _f(
        "pose_model",
        "pose",
        "Skeleton model size",
        "enum",
        "ViTPose-Base",
        binding="cli",
        key="pose_model",
        choices=[
            "ViTPose-Base",
            "ViTPose-Large",
            "ViTPose-Huge",
            "RTMPose-L",
            "Sapiens (ViTPose-Base fallback)",
        ],
        tier=1,
        display="model_cards",
        description=(
            "ViTPose+: larger = usually better on hard motion, slower. "
            "RTMPose-L needs MMPose (see requirements-rtmpose.txt). "
            "**Sapiens** slot: set ``SWAY_SAPIENS_TORCHSCRIPT`` to a COCO-17 ``.pt2`` on the API host for native "
            "TorchScript; otherwise ViTPose-Base keypoints."
        ),
    ),
    _f(
        "info_pose_alternates",
        "pose",
        "Other pose engines",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "**Sapiens** pose card: native when ``SWAY_SAPIENS_TORCHSCRIPT`` points to a ``.pt2`` file, else ViTPose-Base. "
            "**GNN** refine flag lives under Tracking. **HMR** mesh placeholder JSON is under Export. "
            "Regression tests: ``python -m tools.golden_bench`` from a terminal (not a Lab toggle)."
        ),
    ),
    _f(
        "pose_stride",
        "pose",
        "How often to run skeleton estimation",
        "enum",
        1,
        binding="cli",
        key="pose_stride",
        choices=[1, 2],
        tier=1,
        display="segmented",
        lab_hidden=True,
        description=(
            "The Lab **locks** skeleton estimation to **every frame** (``pose_stride`` **1**) per the Pose tab verdict. "
            "CLI or batch matrices may still use stride 2; that path is for experiments, not choreography-grade truth."
        ),
    ),
    _f(
        "sway_pose_gap_interp_mode",
        "pose",
        "How to fill skipped pose frames (stride 2)",
        "enum",
        "linear",
        binding="env",
        key="SWAY_POSE_GAP_INTERP_MODE",
        choices=["linear", "gsi"],
        advanced=True,
        tier=3,
        display="segmented",
        visible_when_field="pose_stride",
        visible_when_value=2,
        description=(
            "Only matters with “every other frame” pose (CLI / matrices—not the Lab). **Linear** (default) aligns "
            "with the Pose tab verdict; GSI uses the same RBF smoother as optional box paths; lengthscale is "
            "SWAY_GSI_LENGTHSCALE (detection advanced when box mode is GSI), or set SWAY_POSE_GSI_LENGTHSCALE "
            "in params YAML to override for pose gaps only."
        ),
    ),
    _f(
        "sway_pose_3d_lift",
        "pose",
        "Run 3D pose lift (MotionAGFormer / depth scoring)",
        "bool",
        True,
        binding="env",
        key="SWAY_3D_LIFT",
        tier=2,
        advanced=True,
        description=(
            "Default on for production. Turn **off** for fastest runs (2D-only; skips Phase 10 lift work). "
            "The Lab **Fast preview** recipe sets this off; ``main.py`` honors ``SWAY_3D_LIFT=0`` when the subprocess env sets it."
        ),
    ),
    _f(
        "sway_vitpose_use_fast",
        "pose",
        "Use Hugging Face fast ViTPose image processor",
        "bool",
        False,
        binding="env",
        key="SWAY_VITPOSE_USE_FAST",
        tier=3,
        advanced=True,
        description=(
            "When on, uses the faster HF ``VitPoseImageProcessor`` path (see ``SWAY_VITPOSE_USE_FAST``). "
            "Fast preview turns this on; on some devices the default slow processor avoids MPS stalls."
        ),
    ),
    _f(
        "info_pose_master_locked",
        "pose",
        "Lock these down (set and forget)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "The master pipeline fixes these for every run — they are **not** configurable in the Lab UI. "
            "``main.py`` reapplies them after ``params`` YAML. For fast smoke (no 3D, chunked ViTPose) or "
            "deliberate FP32 tests, set ``SWAY_UNLOCK_POSE_TUNING=1``.\n\n"
            "**Skeleton cadence:** Default **every frame** (``pose_stride`` **1**). **Fast preview** overrides to "
            "stride **2** for speed (gaps filled per ``SWAY_POSE_GAP_INTERP_MODE``). Use ``--pose-stride 2`` from CLI "
            "for the same idea outside the Fast recipe.\n\n"
            "**Max people per ViTPose GPU forward:** **0** (cap **unset**) — one natural batch per frame; "
            "only raise the cap if 40+ people routinely OOM your GPU.\n\n"
            "**Force ViTPose float32 on GPU:** **Off** — FP16 when the device allows; FP32 is much slower/heavier "
            "for negligible joint gains.\n\n"
            "**Smart bbox pad before ViTPose:** **On** — asymmetric / motion-aware crop expansion (``SWAY_VITPOSE_SMART_PAD``); "
            "see §9.0.1 in ``docs/MASTER_PIPELINE_GUIDELINE.md``.\n\n"
            "**Add a simple depth view (3D):** **On** — runs Phase 10 lifting for depth-aware scoring and the 3D viewer; "
            "use ``SWAY_UNLOCK_POSE_TUNING=1`` and ``SWAY_3D_LIFT=0`` only if you truly want flat 2D-only export."
        ),
    ),
    _f(
        "pose_visibility_threshold",
        "pose",
        "Ignore joints below this confidence",
        "float",
        0.30,
        binding="yaml",
        key="POSE_VISIBILITY_THRESHOLD",
        description=(
            "Baseline 0.30 matches the pipeline default. "
            "Higher = fewer drawn joints in dark or blurry video; lower = show more joints (noisier)."
        ),
        min_=0.0,
        max_=1.0,
        advanced=True,
        tier=3,
        display="slider",
    ),
    _f(
        "info_future_crop_modules",
        "pose",
        "Future-doc crop modules (identity + ViTPose crops)",
        "info",
        None,
        binding="none",
        key="",
        advanced=True,
        tier=3,
        description=(
            "Knobs aligned with ``docs/FUTURE_MODULES_IDENTITY_AND_POSE_CROPS.md`` Part F. "
            "Applied **after** smart bbox pad and **before** ViTPose. "
            "See ``sway/future_modules_registry.py`` for full catalog vs this doc."
        ),
    ),
    _f(
        "sway_pose_crop_smooth_alpha",
        "pose",
        "Temporal crop EMA (0 = off)",
        "float",
        0.0,
        binding="yaml",
        key="POSE_CROP_SMOOTH_ALPHA",
        min_=0.0,
        max_=1.0,
        advanced=True,
        tier=3,
        display="slider",
        description=(
            "**Part F — temporal crop smoothing.** 0 disables. Typical **0.15–0.35**: "
            "each frame blends toward the new expanded box with this weight (higher = more responsive)."
        ),
    ),
    _f(
        "sway_pose_crop_foot_bias_frac",
        "pose",
        "Extra crop below feet (fraction of box height)",
        "float",
        0.0,
        binding="yaml",
        key="POSE_CROP_FOOT_BIAS_FRAC",
        min_=0.0,
        max_=0.35,
        advanced=True,
        tier=3,
        display="slider",
        description="**Part F — footprint prior.** Expands the bottom of the pose crop by this fraction of box height.",
    ),
    _f(
        "sway_pose_crop_head_bias_frac",
        "pose",
        "Extra crop above head (fraction of box height)",
        "float",
        0.0,
        binding="yaml",
        key="POSE_CROP_HEAD_BIAS_FRAC",
        min_=0.0,
        max_=0.35,
        advanced=True,
        tier=3,
        display="slider",
        description="**Part F — head room prior.** Expands the top of the pose crop by this fraction of box height.",
    ),
    _f(
        "sway_pose_crop_anti_jitter_px",
        "pose",
        "Anti-jitter: damp big center jumps (px; 0 = off)",
        "float",
        0.0,
        binding="yaml",
        key="POSE_CROP_ANTI_JITTER_PX",
        min_=0.0,
        max_=120.0,
        advanced=True,
        tier=3,
        display="slider",
        description=(
            "**Part F — anti-jitter gate.** Only applies when temporal crop EMA is **on**. "
            "If the raw crop center jumps farther than this many pixels vs the previous smoothed crop, "
            "blend extra toward the smoothed rectangle."
        ),
    ),
    # --- reid_dedup (main.py phases 6–7) ---
    _f(
        "info_reid_actual",
        "reid_dedup",
        "What this step does",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "Cleans up identity mistakes: when someone hides and reappears, when two dancers cross paths, "
            "or when two skeletons sit on the same body. "
            "You’ll see the effect in overlays and scores—fewer wild ID jumps and fewer double ghosts."
        ),
    ),
    _f(
        "info_reid_dedup_master_locked",
        "reid_dedup",
        "Lock these down (set and forget)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "The master pipeline fixes five deep math gates for every run — they are **not** in the Lab UI. "
            "``main.py`` reapplies them after merging UI ``params``. For sweeps or deliberate experiments, set "
            "``SWAY_UNLOCK_REID_DEDUP_TUNING=1``.\n\n"
            "**Max frames apart to still consider “same person” after a gap:** **90** — let Phase 3 stitching "
            "handle long disappearances; keep this for short occlusions (~3 s @ 30 fps).\n\n"
            "**How similar poses must look to merge IDs:** **0.35** (OKS) — tested baseline for reconnecting a "
            "broken ID without partner false merges.\n\n"
            "**Max joint / bbox-center / torso distances for deduplication:** **0.26** / **0.5** / **0.24** "
            "(fractions of bbox height). All three must agree before a duplicate skeleton is removed — "
            "do not change one without re-tuning the set."
        ),
    ),
    _f(
        "dedup_min_pair_oks",
        "reid_dedup",
        "Min similarity (OKS) to delete duplicate",
        "float",
        0.68,
        binding="yaml",
        key="DEDUP_MIN_PAIR_OKS",
        min_=0.30,
        max_=0.95,
        advanced=True,
        tier=3,
        display="slider",
        description="How similar two colliding poses must be before one is considered a phantom duplicate.",
    ),
    _f(
        "dedup_antipartner_min_iou",
        "reid_dedup",
        "Min bbox overlap for deduplication",
        "float",
        0.12,
        binding="yaml",
        key="DEDUP_ANTIPARTNER_MIN_IOU",
        min_=0.0,
        max_=0.50,
        advanced=True,
        tier=3,
        display="slider",
        description="If IoU is lower than this, it assumes two separate people touching, bypassing deduplication.",
    ),
    # --- post_pose_prune (main.py phase 8) ---
    _f(
        "info_post_pose_tiers",
        "post_pose_prune",
        "What this step does",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "Now that skeletons exist, the pipeline removes tracks that still look wrong—"
            "wobbly noise, mirror reflections, people who never really danced in sync, etc. "
            "Trusted “real dancer” tracks get extra protection. "
            "The sliders below decide how hard each red-flag rule pulls toward “delete this track.”"
        ),
    ),
    _f(
        "info_post_pose_master_locked",
        "post_pose_prune",
        "Lock these down (set and forget)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "The master pipeline fixes Tier C, Tier A protection span, core Tier B garbage thresholds, and three "
            "vote weights — they are **not** in the Lab UI. ``main.py`` reapplies them after ``params`` YAML. "
            "For sweeps, set ``SWAY_UNLOCK_POST_POSE_PRUNE_TUNING=1``.\n\n"
            "**Tier C:** ``TIER_C_SKELETON_MEAN`` **0.15**, ``TIER_C_LOW_FRAME_FRAC`` **0.8** — drops near-empty "
            "skeletons (mics, speakers, background mush).\n\n"
            "**Mushy skeletons:** ``MEAN_CONFIDENCE_MIN`` **0.45**, ``min_lower_body_conf`` **0.3**.\n\n"
            "**Compression noise:** ``JITTER_RATIO_MAX`` **0.1**; vote weight ``prune_jittery_tracks`` **0.5**.\n\n"
            "**Missing limbs / floating heads:** ``PRUNING_WEIGHTS`` entries ``prune_completeness_audit`` **0.6**, "
            "``prune_head_only_tracks`` **0.8**.\n\n"
            "**Edge / wings:** ``EDGE_MARGIN_FRAC`` **0.15**, ``EDGE_PRESENCE_FRAC`` **0.3**.\n\n"
            "**Tier A protected span:** ``CONFIRMED_HUMAN_MIN_SPAN_FRAC`` **0.1** — on-screen ~10%+ with a solid "
            "skeleton exempts a track from harsh Tier B voting."
        ),
    ),
    _f(
        "sync_score_min",
        "post_pose_prune",
        "Minimum “in sync with the group” score to look real",
        "float",
        0.10,
        binding="yaml",
        key="SYNC_SCORE_MIN",
        description=(
            "Baseline 0.10 matches the pipeline default. "
            "Raise if you want stricter “must look like the group”; lower if soloists get unfairly cut."
        ),
        min_=0.0,
        max_=1.0,
        advanced=True,
        display="slider",
    ),
    _f(
        "prune_threshold",
        "post_pose_prune",
        "Overall strictness of the cleanup vote",
        "float",
        0.65,
        binding="yaml",
        key="PRUNE_THRESHOLD",
        description=(
            "Many small checks add up to a score. "
            "Higher = easier to delete a track when something looks off. "
            "Lower = keep more people even when a few checks complain (more junk may remain)."
        ),
        min_=0.40,
        max_=0.90,
        tier=2,
        display="slider",
    ),
    _f(
        "pruning_w_low_sync",
        "post_pose_prune",
        "How much “out of sync” counts against a track",
        "float",
        0.7,
        binding="yaml_pruning_weight",
        key="prune_low_sync_tracks",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
        description="Raise if random audience members keep scoring; lower if real soloists get cut.",
    ),
    _f(
        "pruning_w_smart_mirror",
        "post_pose_prune",
        "How hard to penalize likely mirror reflections",
        "float",
        0.9,
        binding="yaml_pruning_weight",
        key="prune_smart_mirrors",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
        description="High for wall-to-wall mirrors; lower if you film intentional mirrored choreography.",
    ),
    _f(
        "pruning_w_low_conf",
        "post_pose_prune",
        "How much weak joints count against a track",
        "float",
        0.5,
        binding="yaml_pruning_weight",
        key="prune_low_confidence_tracks",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
        description="Blurry or dark footage raises this if mushy skeletons should be dropped faster.",
    ),
    # --- smooth (main.py phase 9) ---
    _f(
        "info_smooth_one_euro",
        "smooth",
        "What this step does",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "Raw joints can wiggle frame to frame. "
            "This pass steadies them so overlays and scores look human without killing real sharp hits. "
            "Neighbor-frame blending before this step is **off** in the master stack (sharper hits); see the lock note below."
        ),
    ),
    _f(
        "info_smooth_master_locked",
        "smooth",
        "Lock these down (set and forget)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "The master pipeline fixes these for every run — they are **not** in the Lab UI. "
            "``main.py`` reapplies them after ``params`` YAML / Lab build. For deliberate experiments "
            "(e.g. neighbor blend on, wider half-window), set ``SWAY_UNLOCK_SMOOTH_TUNING=1``.\n\n"
            "**Neighbor blend (Phase 5):** **off** — ``SWAY_TEMPORAL_POSE_REFINE`` forced **0** on CLI runs; "
            "batch recipes may still pass ``temporal_pose_refine: true`` from the Lab worker.\n\n"
            "**Baseline smoothness (**``SMOOTHER_MIN_CUTOFF``**):** **1.0** — tuned to kill micro-jitter when "
            "someone is nearly still without freezing joints.\n\n"
            "**Neighbor blend half-window (when enabled):** radius **2** (``SWAY_TEMPORAL_POSE_RADIUS``) — "
            "raising it smears many frames together (“underwater” motion); keep the default."
        ),
    ),
    _f(
        "temporal_pose_refine",
        "smooth",
        "Smooth skeletons (fluid vs sharp)",
        "bool",
        False,
        binding="cli",
        key="temporal_pose_refine",
        tier=1,
        description=(
            "**On (fluid):** light neighbor-frame blend — often nicer for ballet and contemporary. "
            "**Off (sharp):** snappier hits for hip-hop and popping. One-Euro smoothing still runs after this step."
        ),
    ),
    _f(
        "smoother_beta",
        "smooth",
        "How quickly smoothing reacts to speed changes",
        "float",
        0.7,
        binding="yaml",
        key="SMOOTHER_BETA",
        min_=0.0,
        max_=5.0,
        description="Higher = follow fast moves sooner; lower = calmer but can lag spikes.",
        advanced=True,
        display="slider",
    ),
    # --- scoring (main.py phase 10) ---
    _f(
        "info_scoring_actual",
        "scoring",
        "What this step does",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "Compares dancers to each other and to the group timing: who’s early/late, which limbs differ, "
            "and rolls that into the numbers you see in exports. "
            "There isn’t a separate “mode” switch here—the math is fixed for consistent results."
        ),
    ),
    # --- export (main.py phase 11) ---
    _f(
        "montage",
        "export",
        "Also save one long “making of” video",
        "bool",
        False,
        binding="cli",
        key="montage",
        description="Concatenates phase preview clips into a single montage file when enabled.",
    ),
    _f(
        "save_phase_previews",
        "export",
        "Save short clips after major steps",
        "bool",
        True,
        binding="cli",
        key="save_phase_previews",
        description=(
            "On by default in the lab: small MP4s per stage so you can scrub what changed without re-running everything. "
            "Turn off to save disk and a little export time."
        ),
    ),
    _f(
        "sway_vis_temporal_interp_mode",
        "export",
        "Smooth overlays between saved frames (video only)",
        "enum",
        "linear",
        binding="env",
        key="SWAY_VIS_TEMPORAL_INTERP_MODE",
        choices=["linear", "gsi"],
        advanced=True,
        tier=3,
        display="segmented",
        description=(
            "Final MP4s can run at native FPS while pose/boxes are stored at processed rate—this chooses how to blend "
            "in between. **Linear** (default) matches the Pose tab verdict; GSI is optional video-only smoothing; "
            "lengthscale uses SWAY_GSI_LENGTHSCALE or SWAY_VIS_GSI_LENGTHSCALE in YAML."
        ),
    ),
    _f(
        "sway_hmr_mesh_sidecar",
        "export",
        "Write hmr_mesh_sidecar.json placeholder after export",
        "bool",
        False,
        binding="env",
        key="SWAY_HMR_MESH_SIDECAR",
        advanced=True,
        tier=3,
        description=(
            "Default off. When on, ``main.py`` writes ``hmr_mesh_sidecar.json`` in the output folder after rendering "
            "(schema stub until HMR / mesh export exists). For experiments, see also ``tools/hmr_3d_optional_stub.py``."
        ),
    ),
]

# --- Lab UI: intent-based “Sway” panel (see PIPELINE_STAGES); everything else is master-hidden in the UI. ---
_LEAN_VISIBLE_IDS = frozenset(
    {
        "sway_yolo_weights",
        "sway_phase13_mode",
        "sway_pretrack_nms_iou",
        "sway_yolo_conf",
        "sway_hybrid_sam_iou_trigger",
        "sway_boxmot_max_age",
        "sway_global_aflink_mode",
        "pose_model",
        "temporal_pose_refine",
    }
)

_LEGACY_PHASE_TO_INTENT = {
    "detection": "crowd_control",
    "tracking": "handshake",
    "phase3_stitch": "handshake",
    "pose": "pose_polish",
    "reid_dedup": "handshake",
    "post_pose_prune": "cleanup_export",
    "smooth": "pose_polish",
    "scoring": "cleanup_export",
    "export": "cleanup_export",
}

# ``sway_phase13_mode`` is omitted: Lab API ``_effective_ui_fields`` applies these after client fields;
# locking it to standard would ignore Dancer registry / Sway handshake on every enqueue.
# ``tracker_technology`` is omitted: Fast preview sets ByteTrack; locking to deep_ocsort would undo it on enqueue.
# ``sway_yolo_detection_stride`` / ``pose_stride`` / ``save_phase_previews`` omitted: Fast preview uses aggressive
# stride + no phase MP4s; locking would undo super-fast enqueue.
_UI_LOCK_DEFAULTS: Dict[str, Any] = {
    "sway_boxmot_reid_model": "osnet_x0_25",
    "sway_box_interp_mode": "linear",
    "sway_vis_temporal_interp_mode": "linear",
    "sway_gnn_track_refine": False,
    "sway_hybrid_sam_weak_cues": False,
    "sway_hybrid_weak_conf_delta": 0.08,
    "sway_hybrid_weak_height_frac": 0.12,
    "sway_hybrid_weak_match_iou": 0.25,
    "sway_pose_crop_smooth_alpha": 0.0,
    "sway_pose_crop_foot_bias_frac": 0.0,
    "sway_pose_crop_head_bias_frac": 0.0,
    "sway_pose_crop_anti_jitter_px": 0.0,
    # Post-pose cleanup vote + export outputs (no Lab sliders/toggles).
    "prune_threshold": 0.65,
    "montage": False,
    "save_phase_previews": True,
}

# Applied server-side to every Lab run / config merge so hidden master locks stay consistent.
LAB_UI_ENFORCED_DEFAULTS: Dict[str, Any] = dict(_UI_LOCK_DEFAULTS)

for _row in PIPELINE_PARAM_FIELDS:
    _fid = _row["id"]
    _row["phase"] = _LEGACY_PHASE_TO_INTENT.get(str(_row.get("phase") or ""), "crowd_control")
    if _fid in _UI_LOCK_DEFAULTS:
        _row["default"] = _UI_LOCK_DEFAULTS[_fid]
    if _row.get("type") == "info":
        _row["lab_hidden"] = True
    elif _fid in _LEAN_VISIBLE_IDS:
        _row["lab_hidden"] = False
        _row["tier"] = 1
        _row["advanced"] = False
    else:
        _row["lab_hidden"] = True


PIPELINE_PRESET_GROUPS: List[Dict[str, Any]] = [
    {
        "id": "phases_1_3",
        "label": "Detection / Tracking / Stitching",
        "phases_label": "Phases 1-3",
        "default_preset": "p13_standard",
    },
    {
        "id": "phases_4_6",
        "label": "Pose / Association",
        "phases_label": "Phases 4-6",
        "default_preset": "p46_balanced",
    },
    {
        "id": "phases_7_9",
        "label": "Cleanup / Pruning / Smoothing",
        "phases_label": "Phases 7-9",
        "default_preset": "p79_balanced",
    },
    {
        "id": "phases_10_11",
        "label": "Scoring / Export",
        "phases_label": "Phases 10-11",
        "default_preset": "p1011_standard",
    },
]


def schema_payload() -> Dict[str, Any]:
    return {
        "stages": PIPELINE_STAGES,
        "fields": PIPELINE_PARAM_FIELDS,
        "preset_groups": PIPELINE_PRESET_GROUPS,
    }
