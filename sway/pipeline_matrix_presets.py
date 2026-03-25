"""
Curated pipeline A/B recipes for comparing one stage at a time.

**Fixed across all recipes (proven stack for your product):**
  - ``sway_yolo_weights`` = ``yolo26l_dancetrack``
  - Hybrid SAM overlap, ROI crop, and ROI pad are **master-locked** in ``main.py`` / Lab (not matrix fields).
  - Long-range global link and AFLink threshold env vars are **master-locked** (§7.0.1 in ``MASTER_PIPELINE_GUIDELINE.md``); use ``sway_global_aflink_mode`` to compare neural vs. heuristic linker.
  - ViTPose batch cap, FP32, and 3D lift are **master-locked** (§9.0.1); model size and pose stride remain matrix-safe.
  - Phase 6–7 re-ID frame gap, OKS merge threshold, and the three collocated-dedup distance gates are **master-locked** (§11.0.1).
  - Phase 8 Tier C / Tier A span / garbage thresholds and three Tier B weights (completeness, head-only, jittery) are **master-locked** (§13.0.1).
  - Phase 9 ``SMOOTHER_MIN_CUTOFF`` and neighbor-blend radius (``SWAY_TEMPORAL_POSE_RADIUS``) are **master-locked** (§14.0.1).

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


PIPELINE_MATRIX_VERSION = 10

# Enforced on every row so matrix runs never swap detector weights.
_PROVEN_DETECTION_SAM: Dict[str, Any] = {
    "sway_yolo_weights": "yolo26l_dancetrack",
}

PIPELINE_MATRIX_INTRO = (
    "YOLO26L DanceTrack is fixed for every recipe below; hybrid SAM overlap/ROI keys are master-locked (§6.4.1); "
    "long-range merge / AFLink thresholds are master-locked (§7.0.1); ViTPose cap, FP32, and 3D lift are "
    "master-locked (§9.0.1); Phase 6–7 re-ID gap / OKS / dedup distance triple-gate are master-locked (§11.0.1); "
    "Phase 8 Tier C + garbage baseline + three prune weights are master-locked (§13.0.1); "
    "Phase 9 1-Euro min cutoff + temporal blend radius are master-locked (§14.0.1) — "
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
        "no_temporal_kp",
        "M10_no_temporal_kp_refine",
        "pose",
        "Disable neighbor-frame keypoint blend before 1-Euro.",
        {"temporal_pose_refine": False},
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
        "hybrid_sam",
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
