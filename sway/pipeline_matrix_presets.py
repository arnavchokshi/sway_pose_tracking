"""
Curated pipeline A/B recipes for comparing one stage at a time.

**Fixed across all recipes (proven stack for your product):**
  - ``sway_yolo_weights`` = ``yolo26l_dancetrack``
  - Hybrid SAM overlap, ROI crop, and ROI pad are **master-locked** in ``main.py`` / Lab (not matrix fields).
  - Long-range global link and AFLink threshold env vars are **master-locked** (§7.0.1 in ``MASTER_PIPELINE_GUIDELINE.md``); use ``sway_global_aflink_mode`` to compare neural vs. heuristic linker.
  - ViTPose batch cap, FP32, and 3D lift are **master-locked** (§9.0.1); model size and pose stride remain matrix-safe.
  - Phase 6–7 re-ID frame gap, OKS merge threshold, and the three collocated-dedup distance gates are **master-locked** (§11.0.1).
  - Phase 8 Tier C / Tier A span / garbage thresholds and three Tier B weights (completeness, head-only, jittery) are **master-locked** (§13.0.1).
  - Phase 9 ``SMOOTHER_MIN_CUTOFF``, neighbor-blend radius (``SWAY_TEMPORAL_POSE_RADIUS``), and **neighbor blend off**
    (``SWAY_TEMPORAL_POSE_REFINE``) are **master-locked** on CLI (§14.0.1); Lab batch rows may still set ``temporal_pose_refine: true``.

Everything else follows Lab defaults except the **single axis** each row documents
(or a minimal pair when one knob only applies with another, e.g. stride-2 detection + box GSI vs. linear).

Use ``GET /api/pipeline_matrix`` or ``python -m tools.pipeline_matrix_runs``.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class MatrixRecipe(TypedDict, total=False):
    """One row in the matrix table."""

    id: str
    recipe_name: str
    description: str
    varies: str
    fields: Dict[str, Any]


PIPELINE_MATRIX_VERSION = 13

# Enforced on every row so matrix runs never swap detector weights.
_PROVEN_DETECTION_SAM: Dict[str, Any] = {
    "sway_yolo_weights": "yolo26l_dancetrack",
}

PIPELINE_MATRIX_INTRO = (
    "YOLO26L DanceTrack is fixed for every recipe below; hybrid SAM overlap/ROI keys are master-locked (§6.4.1); "
    "long-range merge / AFLink thresholds are master-locked (§7.0.1); ViTPose cap, FP32, and 3D lift are "
    "master-locked (§9.0.1); Phase 6–7 re-ID gap / OKS / dedup distance triple-gate are master-locked (§11.0.1); "
    "Phase 8 Tier C + garbage baseline + three prune weights are master-locked (§13.0.1); "
    "Phase 9 1-Euro min cutoff + temporal blend radius + neighbor blend off are master-locked on CLI (§14.0.1) — "
    "compare ``pose_model``, ``pose_stride``, and gap interpolation instead. "
    "Each row changes only one other knob vs. the default dance stack (Deep OC-SORT without track-time OSNet, "
    "ViTPose-Base, pose stride 1). "
    "Rows M13+ target alternate stacks and knobs not covered by earlier rows (RTMPose, bidirectional tracking, "
    "SAM weak cues, stride-2 box GSI vs. linear)."
)


def _row(
    id_: str,
    recipe_name: str,
    varies: str,
    description: str,
    fields: Dict[str, Any],
) -> MatrixRecipe:
    merged = {**_PROVEN_DETECTION_SAM, **fields}
    return {
        "id": id_,
        "recipe_name": recipe_name,
        "varies": varies,
        "description": description,
        "fields": merged,
    }


# Ordered list: stable IDs for UI checkboxes and CLI --only.
PIPELINE_MATRIX_RECIPES: List[MatrixRecipe] = [
    _row(
        "baseline",
        "M01_baseline_locked_stack",
        "none",
        "Reference: DanceTrack YOLO + hybrid SAM on + all other schema defaults (Deep OC-SORT, global link, ViTPose-Base, …).",
        {},
    ),
    _row(
        "tracker_deep_ocsort_osnet",
        "M02_tracker_Deep_OC-SORT_OSNet",
        "tracking",
        "Deep OC-SORT with track-time OSNet vs. default motion-only Deep OC-SORT (needs models/osnet_x0_25_msmt17.pt).",
        {"tracker_technology": "deep_ocsort_osnet", "sway_boxmot_reid_model": "osnet_x0_25"},
    ),
    _row(
        "aflink_heuristic",
        "M06_AFLink_heuristic_only",
        "phase3_stitch",
        "Geometry-only linker vs. neural AFLink when weights exist.",
        {"sway_global_aflink_mode": "force_heuristic"},
    ),
    _row(
        "pose_large",
        "M07_pose_ViTPose-Large",
        "pose",
        "ViTPose-Large vs. Base — quality vs. speed.",
        {"pose_model": "ViTPose-Large"},
    ),
    _row(
        "pose_stride2",
        "M08_pose_stride_2",
        "pose",
        "Pose every other frame (linear gap fill) vs. every frame.",
        {"pose_stride": 2},
    ),
    _row(
        "pose_stride2_gsi",
        "M09_pose_stride2_gap_GSI",
        "pose",
        "Stride-2 with GSI gap fill vs. stride-2 linear (isolates gap interpolation).",
        {"pose_stride": 2, "sway_pose_gap_interp_mode": "gsi"},
    ),
    _row(
        "temporal_kp_on",
        "M10_temporal_kp_refine_on",
        "pose",
        "Enable neighbor-frame keypoint blend before 1-Euro vs. master default (off).",
        {"temporal_pose_refine": True},
    ),
    _row(
        "boxmot_match_strict",
        "M11_boxmot_match_0.40",
        "tracking",
        "Stricter BoxMOT IoU match (0.40) vs. default — fewer merges, possible extra ID breaks.",
        {"sway_boxmot_match_thresh": 0.40},
    ),
    _row(
        "det_stride2",
        "M12_yolo_det_stride_2",
        "detection",
        "Run DanceTrack YOLO every 2nd frame — speed vs. dense detection (same weights).",
        {"sway_yolo_detection_stride": 2},
    ),
    # --- M13+ : alternate technologies / less-tested knobs (skip trivial YOLO size sweeps) ---
    _row(
        "pose_rtmpose",
        "M13_pose_RTMPose_L",
        "pose",
        "MMPose RTMPose-L vs. default ViTPose+ Base (different stack; needs requirements-rtmpose.txt).",
        {"pose_model": "RTMPose-L"},
    ),
    _row(
        "bidirectional_track",
        "M14_bidirectional_track_pass",
        "tracking",
        "Forward+reverse tracking merge vs. single pass (~2× Phase 1–2 time); tests ID stability on hard clips.",
        {"sway_bidirectional_track_pass": True},
    ),
    _row(
        "hybrid_sam_weak_cues",
        "M15_hybrid_SAM_weak_cues",
        "tracking",
        "Hybrid-SORT-style skip gate when overlap is stable — faster SAM phase vs. default always-evaluate overlap.",
        {"sway_hybrid_sam_weak_cues": True},
    ),
    _row(
        "det_stride2_box_gsi",
        "M16_det_stride2_box_GSI",
        "detection",
        "Stride-2 detection with GSI box gap fill — compare to recipe det_stride2 (linear) to isolate box interpolation.",
        {"sway_yolo_detection_stride": 2, "sway_box_interp_mode": "gsi"},
    ),
    # --- Phase-group preset combos (M17+): test full preset configurations end-to-end ---
    _row(
        "preset_dense_hifi",
        "M17_preset_dense_crowd_high_fidelity",
        "multi_phase",
        "Dense Crowd (phases 1-3) + High Fidelity (phases 4-6) + Balanced Cleanup (phases 7-9). For packed formations needing quality skeletons.",
        {
            "sway_yolo_weights": "yolo26x",
            "sway_pretrack_nms_iou": 0.60,
            "sway_yolo_conf": 0.18,
            "sway_hybrid_sam_iou_trigger": 0.30,
            "sway_boxmot_max_age": 200,
            "sway_boxmot_match_thresh": 0.25,
            "pose_model": "ViTPose-Large",
            "pose_visibility_threshold": 0.25,
        },
    ),
    _row(
        "preset_open_competition",
        "M18_preset_open_floor_competition",
        "multi_phase",
        "Open Floor (phases 1-3) + Competition Grade (phases 4-6) + Sharp Hip-Hop (phases 7-9). For small group competitions.",
        {
            "sway_yolo_conf": 0.30,
            "sway_pretrack_nms_iou": 0.80,
            "sway_hybrid_sam_iou_trigger": 0.50,
            "sway_boxmot_max_age": 120,
            "sway_boxmot_match_thresh": 0.35,
            "pose_model": "ViTPose-Large",
            "temporal_pose_refine": True,
            "pose_visibility_threshold": 0.22,
            "dedup_min_pair_oks": 0.75,
            "dedup_antipartner_min_iou": 0.15,
            "smoother_beta": 0.55,
            "prune_threshold": 0.60,
            "sync_score_min": 0.12,
        },
    ),
    _row(
        "preset_open_competition_recovery",
        "M23_preset_open_floor_competition_recovery",
        "multi_phase",
        "Open Floor + Competition + Sharp Hip-Hop with **recovery bias** (docs/PIPELINE_FINDINGS_AND_BEST_CONFIGS.md): "
        "DanceTrack+CrowdHuman weights, hybrid SAM IoU 0.42, box match 0.29, max_age 165 — better ID re-attach after occlusion vs M18; more SAM cost.",
        {
            "sway_yolo_weights": "yolo26l_dancetrack_crowdhuman",
            "sway_yolo_conf": 0.30,
            "sway_pretrack_nms_iou": 0.80,
            "sway_hybrid_sam_iou_trigger": 0.42,
            "sway_boxmot_max_age": 165,
            "sway_boxmot_match_thresh": 0.29,
            "pose_model": "ViTPose-Large",
            "temporal_pose_refine": True,
            "pose_visibility_threshold": 0.22,
            "dedup_min_pair_oks": 0.75,
            "dedup_antipartner_min_iou": 0.15,
            "smoother_beta": 0.55,
            "prune_threshold": 0.60,
            "sync_score_min": 0.12,
        },
    ),
    _row(
        "preset_ballet_fluid",
        "M19_preset_standard_ballet",
        "multi_phase",
        "Standard (phases 1-3) + Balanced Pose (phases 4-6) + Fluid Ballet (phases 7-9). For ballet and contemporary dance.",
        {
            "temporal_pose_refine": True,
            "smoother_beta": 0.85,
            "prune_threshold": 0.65,
            "sync_score_min": 0.08,
            "pruning_w_low_sync": 0.5,
            "pruning_w_smart_mirror": 0.7,
            "pruning_w_low_conf": 0.4,
        },
    ),
    _row(
        "preset_mirror_studio",
        "M20_preset_mirror_studio",
        "multi_phase",
        "Standard (phases 1-3) + Balanced Pose (phases 4-6) + Mirror Studio (phases 7-9). For studios with wall-to-wall mirrors.",
        {
            "pruning_w_smart_mirror": 1.0,
            "prune_threshold": 0.60,
            "pruning_w_low_sync": 0.75,
        },
    ),
    _row(
        "preset_wide_angle_maxprec",
        "M21_preset_wide_angle_max_precision",
        "multi_phase",
        "Wide Angle (phases 1-3) + Maximum Precision (phases 4-6). For wide-angle stage shots needing best skeletons.",
        {
            "sway_yolo_weights": "yolo26x",
            "sway_yolo_conf": 0.15,
            "sway_pretrack_nms_iou": 0.70,
            "sway_boxmot_max_age": 200,
            "sway_hybrid_sam_iou_trigger": 0.35,
            "pose_model": "ViTPose-Huge",
            "temporal_pose_refine": True,
            "pose_visibility_threshold": 0.20,
            "dedup_min_pair_oks": 0.72,
        },
    ),
    _row(
        "preset_osnet_aggressive",
        "M22_preset_osnet_lock_aggressive_clean",
        "multi_phase",
        "OSNet Identity Lock (phases 1-3) + High Fidelity (phases 4-6) + Aggressive Clean (phases 7-9). Maximum identity consistency with strict cleanup.",
        {
            "tracker_technology": "deep_ocsort_osnet",
            "sway_boxmot_reid_model": "osnet_x0_25",
            "sway_bidirectional_track_pass": True,
            "sway_boxmot_max_age": 180,
            "pose_model": "ViTPose-Large",
            "pose_visibility_threshold": 0.25,
            "prune_threshold": 0.55,
            "sync_score_min": 0.15,
            "pruning_w_low_sync": 0.85,
            "pruning_w_low_conf": 0.7,
        },
    ),
]


def pipeline_matrix_for_api() -> Dict[str, Any]:
    """JSON-serializable payload for ``GET /api/pipeline_matrix``."""
    return {
        "version": PIPELINE_MATRIX_VERSION,
        "intro": PIPELINE_MATRIX_INTRO,
        "proven_locks": dict(_PROVEN_DETECTION_SAM),
        "recipes": list(PIPELINE_MATRIX_RECIPES),
    }


def matrix_recipe_by_id(recipe_id: str) -> MatrixRecipe | None:
    for r in PIPELINE_MATRIX_RECIPES:
        if r.get("id") == recipe_id:
            return r
    return None
