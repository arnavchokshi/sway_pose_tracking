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


PIPELINE_MATRIX_VERSION = 14

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
    # ── Future Pipeline Presets (PLAN modules) ──────────────────────────────
    _row(
        "f01_lean_core_i1",
        "F01_lean_core_I1",
        "tracker_engine+enrollment+coi+state_machine",
        "Gate 1: enrollment + SAM2MOT + COI + track state machine (lean core iteration 1).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
        },
    ),
    _row(
        "f05_lean_core_full",
        "F05_lean_core_full",
        "tracker_engine+enrollment+coi+state+reid_fusion+collision",
        "Gate 2: full lean core (SAM2MOT + enrollment + COI + state + re-ID fusion + collision solver).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
        },
    ),
    _row(
        "f10_mask_guided_pose",
        "F10_mask_guided_pose",
        "pose_mask_guided",
        "Gate 3: mask-guided pose estimation with per-keypoint confidence (PLAN_17).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_pose_mask_guided": "1",
        },
    ),
    _row(
        "f15_motionbert",
        "F15_motionbert_lift",
        "lift_backend",
        "Gate 4: MotionBERT 3D lifting backend (PLAN_18).",
        {
            "sway_lift_backend": "motionbert",
        },
    ),
    _row(
        "f20_trk_solidtrack",
        "F20_trk_solidtrack",
        "tracker_engine",
        "Regression baseline: current production pipeline (SolidTrack/BoxMOT).",
        {},
    ),
    _row(
        "f21_trk_sam2mot",
        "F21_trk_sam2mot",
        "tracker_engine",
        "SAM2-only primary tracking without enrollment or COI.",
        {
            "sway_tracker_engine": "sam2mot",
        },
    ),
    _row(
        "f22_trk_memosort",
        "F22_trk_memosort",
        "tracker_engine",
        "MeMoSORT standalone tracker.",
        {
            "sway_tracker_engine": "memosort",
        },
    ),
    _row(
        "f23_trk_sam2_memosort_hybrid",
        "F23_trk_sam2_memosort_hybrid",
        "tracker_engine",
        "SAM2 masks + MeMoSORT motion prediction hybrid tracker.",
        {
            "sway_tracker_engine": "sam2_memosort_hybrid",
        },
    ),
    _row(
        "f30_det_rtdetr_l",
        "F30_det_rtdetr_l",
        "detector_primary",
        "RT-DETR-L NMS-free detector (PLAN_02).",
        {
            "sway_detector_primary": "rt_detr_l",
            "sway_tracker_engine": "sam2mot",
        },
    ),
    _row(
        "f31_det_rtdetr_x",
        "F31_det_rtdetr_x",
        "detector_primary",
        "RT-DETR-X NMS-free detector (PLAN_02).",
        {
            "sway_detector_primary": "rt_detr_x",
            "sway_tracker_engine": "sam2mot",
        },
    ),
    _row(
        "f32_det_hybrid_yolo_detr",
        "F32_det_hybrid",
        "detector_hybrid",
        "Hybrid detector: YOLO every frame + RT-DETR on overlap frames (PLAN_03).",
        {
            "sway_detector_hybrid": "1",
            "sway_detector_primary": "rt_detr_l",
            "sway_tracker_engine": "sam2mot",
        },
    ),
    _row(
        "f40_reid_fusion",
        "F40_reid_fusion",
        "reid_fusion",
        "Re-ID fusion engine with enrollment galleries (PLAN_13).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid",
        },
    ),
    _row(
        "f45_backward_pass",
        "F45_backward_pass",
        "backward_pass",
        "Backward-pass gap filling with forward-reverse stitch (PLAN_16).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_backward_pass_enabled": "1",
        },
    ),
    _row(
        "f50_critique",
        "F50_critique_engine",
        "critique_dimensions",
        "Five-dimension biomechanical critique scoring (PLAN_19).",
        {
            "sway_critique_dimensions": "formation,timing,extension,smoothness,sync",
        },
    ),
    _row(
        "f55_full_future",
        "F55_full_future_stack",
        "all_future_modules",
        "Full future stack: SAM2MOT + enrollment + COI + mask-guided pose + re-ID fusion + collision + backward + critique.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_pose_mask_guided": "1",
            "sway_collision_solver": "hungarian",
            "sway_backward_pass_enabled": "1",
            "sway_critique_dimensions": "formation,timing,extension,smoothness,sync",
            "sway_lift_backend": "motionbert",
        },
    ),
    # ── Section 13.2: Lean Core Progression (missing I2–I4) ──────────────
    _row(
        "f02_lean_core_i2",
        "F02_lean_core_reid_upgrade",
        "tracker_engine+enrollment+coi+reid_fusion+collision",
        "I1 + BPBreID part-based Re-ID + KPR + color + spatial + Hungarian collision solver. Skeleton/face OFF.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_collision_solver": "hungarian",
        },
    ),
    _row(
        "f03_lean_core_i3",
        "F03_lean_core_hybrid_detection",
        "tracker_engine+enrollment+coi+reid_fusion+collision+hybrid_det",
        "I2 + Hybrid YOLO scout + Co-DINO precision detector on overlap frames.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_detector_hybrid": "1",
            "sway_detector_primary": "co_dino",
        },
    ),
    _row(
        "f04_lean_core_i4",
        "F04_lean_core_pose_upgrade",
        "tracker_engine+enrollment+coi+reid+collision+hybrid_det+pose+lift",
        "I3 + Mask-guided ViTPose-Large + per-keypoint confidence + MotionBERT 3D lifting.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_detector_hybrid": "1",
            "sway_detector_primary": "co_dino",
            "sway_pose_mask_guided": "1",
            "pose_model": "ViTPose-Large",
            "sway_lift_backend": "motionbert",
        },
    ),
    # ── Section 13.3: Detector A/B Presets ────────────────────────────────
    _row(
        "f10_det_yolo_only",
        "F10_detector_yolo_only",
        "detection",
        "YOLO-only, no hybrid dispatch. Baseline speed reference.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_detector_hybrid": "0",
        },
    ),
    _row(
        "f11_det_codetr_only",
        "F11_detector_codetr_only",
        "detection",
        "Co-DETR on every frame. Maximum detection AP, highest cost.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_detector_primary": "co_detr",
            "sway_detector_hybrid": "0",
        },
    ),
    _row(
        "f12_det_codino_only",
        "F12_detector_codino_only",
        "detection",
        "Co-DINO on every frame.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_detector_primary": "co_dino",
            "sway_detector_hybrid": "0",
        },
    ),
    _row(
        "f13_det_rtdetr_only",
        "F13_detector_rtdetr_only",
        "detection",
        "RT-DETR on every frame. Faster than Co-DETR, still NMS-free.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_detector_primary": "rt_detr_l",
            "sway_detector_hybrid": "0",
        },
    ),
    _row(
        "f14_det_hybrid_codino",
        "F14_detector_hybrid_yolo_codino",
        "detection",
        "Default lean core: YOLO scout + Co-DINO on overlap frames.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_detector_hybrid": "1",
            "sway_detector_primary": "co_dino",
        },
    ),
    _row(
        "f15_det_hybrid_rtdetr",
        "F15_detector_hybrid_yolo_rtdetr",
        "detection",
        "YOLO scout + RT-DETR on overlap frames. Faster hybrid variant.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_detector_hybrid": "1",
            "sway_detector_primary": "rt_detr_l",
        },
    ),
    _row(
        "f16a_det_hybrid_overlap_lo",
        "F16a_detector_hybrid_overlap_015",
        "detection",
        "Hybrid YOLO + Co-DINO with SWAY_HYBRID_OVERLAP_IOU_TRIGGER=0.15 (sweep low — DETR fires more often).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_detector_hybrid": "1",
            "sway_detector_primary": "co_dino",
            "sway_hybrid_overlap_iou_trigger": 0.15,
        },
    ),
    _row(
        "f16_det_hybrid_sweep",
        "F16_detector_hybrid_trigger_sweep",
        "detection",
        "Hybrid YOLO + Co-DINO with SWAY_HYBRID_OVERLAP_IOU_TRIGGER=0.30 (sweep mid / default).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_detector_hybrid": "1",
            "sway_detector_primary": "co_dino",
            "sway_hybrid_overlap_iou_trigger": 0.30,
        },
    ),
    _row(
        "f16c_det_hybrid_overlap_hi",
        "F16c_detector_hybrid_overlap_050",
        "detection",
        "Hybrid YOLO + Co-DINO with SWAY_HYBRID_OVERLAP_IOU_TRIGGER=0.50 (sweep high — DETR only on heavy overlap).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_detector_hybrid": "1",
            "sway_detector_primary": "co_dino",
            "sway_hybrid_overlap_iou_trigger": 0.50,
        },
    ),
    _row(
        "f17_det_yolo_crowdhuman",
        "F17_detector_yolo_crowdhuman",
        "detection",
        "YOLO with DanceTrack+CrowdHuman weights. Tests fine-tuned YOLO as a stronger scout.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_yolo_weights": "yolo26l_dancetrack_crowdhuman",
        },
    ),
    # ── Section 13.4: Tracker Engine A/B Presets (missing entries) ────────
    _row(
        "f23_trk_sam2_memosort",
        "F23_tracker_sam2_memosort_hybrid",
        "tracker_engine",
        "SAM2 masks + MeMoSORT motion prediction hybrid tracker (doc alias for f23_trk_sam2_memosort_hybrid).",
        {
            "sway_tracker_engine": "sam2_memosort_hybrid",
        },
    ),
    _row(
        "f24_trk_sam2_b",
        "F24_tracker_sam2_model_base",
        "tracker_engine",
        "SAM2MOT with sam2.1_b checkpoint (default).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_sam2_model": "sam2.1_b",
        },
    ),
    _row(
        "f25_trk_sam2_l",
        "F25_tracker_sam2_model_large",
        "tracker_engine",
        "SAM2MOT with sam2.1_l checkpoint. Better masks, slower.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_sam2_model": "sam2.1_l",
        },
    ),
    _row(
        "f26_trk_sam2_h",
        "F26_tracker_sam2_model_huge",
        "tracker_engine",
        "SAM2MOT with sam2.1_h checkpoint. Best masks, slowest.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_sam2_model": "sam2.1_h",
        },
    ),
    _row(
        "f27_trk_coi_freeze",
        "F27_tracker_coi_freeze_mode",
        "tracker_engine",
        "COI quarantine mode = freeze instead of delete. Less aggressive memory management.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_coi_enabled": "1",
            "sway_coi_quarantine_mode": "freeze",
        },
    ),
    # ── Section 13.5: Re-ID Signal A/B Presets ────────────────────────────
    _row(
        "f30_reid_osnet_baseline",
        "F30_reid_osnet_global_only",
        "reid",
        "OSNet x0.25 global embedding only (no parts). Current production Re-ID. Regression baseline.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "",
        },
    ),
    _row(
        "f31_reid_bpbreid_only",
        "F31_reid_bpbreid_parts_only",
        "reid",
        "BPBreID parts + color + spatial. No KPR, no face, no gait.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid",
        },
    ),
    _row(
        "f32_reid_lean_core",
        "F32_reid_lean_core_4signal",
        "reid",
        "Default lean core: BPBreID + KPR + color + spatial. The 4-signal ensemble.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
        },
    ),
    _row(
        "f33_reid_plus_face",
        "F33_reid_lean_plus_face",
        "reid",
        "Lean core + ArcFace face signal. Tests whether face recognition adds value.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_reid_w_face": "0.20",
        },
    ),
    _row(
        "f34_reid_plus_skeleton",
        "F34_reid_lean_plus_skeleton",
        "reid",
        "Lean core + MoCos skeleton gait signal. Tests whether gait adds value.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_reid_w_skeleton": "0.20",
        },
    ),
    _row(
        "f35_reid_full_6signal",
        "F35_reid_full_ensemble_6signal",
        "reid",
        "All 6 signals ON: part + KPR + skeleton + face + color + spatial. Maximum Re-ID power.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_reid_w_skeleton": "0.20",
            "sway_reid_w_face": "0.20",
        },
    ),
    _row(
        "f36_reid_finetuned",
        "F36_reid_bpbreid_finetuned",
        "reid",
        "BPBreID with contrastive fine-tuned weights (PLAN_20). Tests domain adaptation.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid_finetuned",
        },
    ),
    _row(
        "f37_reid_finetuned_full",
        "F37_reid_finetuned_full_ensemble",
        "reid",
        "Fine-tuned BPBreID + all 6 signals. The theoretical maximum Re-ID config.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid_finetuned",
            "sway_reid_kpr_enabled": "1",
            "sway_reid_w_skeleton": "0.20",
            "sway_reid_w_face": "0.20",
        },
    ),
    _row(
        "f38_reid_weight_sweep_a",
        "F38_reid_appearance_heavy",
        "reid",
        "W_PART=0.45, W_KPR=0.20, W_COLOR=0.05, W_SPATIAL=0.05. Appearance-dominated weighting.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_reid_w_part": "0.45",
            "sway_reid_w_kpr": "0.20",
            "sway_reid_w_color": "0.05",
            "sway_reid_w_spatial": "0.05",
        },
    ),
    _row(
        "f39_reid_weight_sweep_b",
        "F39_reid_balanced_weights",
        "reid",
        "Equal weights across all active signals. Tests whether learned weighting matters.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_reid_w_part": "0.25",
            "sway_reid_w_kpr": "0.25",
            "sway_reid_w_color": "0.25",
            "sway_reid_w_spatial": "0.25",
        },
    ),
    # ── Section 13.6: Pose & 3D Lifting A/B Presets ───────────────────────
    _row(
        "f40_pose_vitpose_large",
        "F40_pose_vitpose_large_masked",
        "pose",
        "Default lean core: ViTPose-Large with SAM2 mask guidance.",
        {
            "sway_tracker_engine": "sam2mot",
            "pose_model": "ViTPose-Large",
            "sway_pose_mask_guided": "1",
        },
    ),
    _row(
        "f41_pose_vitpose_huge",
        "F41_pose_vitpose_huge_masked",
        "pose",
        "ViTPose-Huge with mask guidance. Higher accuracy, higher cost.",
        {
            "sway_tracker_engine": "sam2mot",
            "pose_model": "ViTPose-Huge",
            "sway_pose_mask_guided": "1",
        },
    ),
    _row(
        "f42_pose_vitpose_nomask",
        "F42_pose_vitpose_large_nomask",
        "pose",
        "ViTPose-Large WITHOUT mask guidance. Tests mask guidance value.",
        {
            "sway_tracker_engine": "sam2mot",
            "pose_model": "ViTPose-Large",
            "sway_pose_mask_guided": "0",
        },
    ),
    _row(
        "f43_pose_rtmw_l",
        "F43_pose_rtmw_l_wholebody",
        "pose",
        "RTMW-L whole-body (133 keypoints). Tests hand/foot/face keypoints for critique.",
        {
            "sway_tracker_engine": "sam2mot",
            "pose_model": "RTMW-L",
        },
    ),
    _row(
        "f44_pose_rtmw_x",
        "F44_pose_rtmw_x_wholebody",
        "pose",
        "RTMW-X whole-body. Larger, more accurate.",
        {
            "sway_tracker_engine": "sam2mot",
            "pose_model": "RTMW-X",
        },
    ),
    _row(
        "f45_lift_motionagformer",
        "F45_lift_motionagformer",
        "3d_lift",
        "Current production 3D lifter. Regression baseline.",
        {
            "sway_lift_backend": "motionagformer",
        },
    ),
    _row(
        "f46_lift_motionbert",
        "F46_lift_motionbert_single",
        "3d_lift",
        "Default lean core: MotionBERT per-person 3D lifting.",
        {
            "sway_lift_backend": "motionbert",
        },
    ),
    _row(
        "f47_lift_motionbert_multi",
        "F47_lift_motionbert_multi_person",
        "3d_lift",
        "MotionBERT + multi-person shared floor plane + depth estimation.",
        {
            "sway_lift_backend": "motionbert",
            "sway_lift_multi_person": "1",
        },
    ),
    # ── Section 13.7: Backward Pass & Collision Presets ───────────────────
    _row(
        "f50_backward_off",
        "F50_backward_pass_disabled",
        "backward",
        "No backward pass. Lean core default.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_backward_pass_enabled": "0",
        },
    ),
    _row(
        "f51_backward_on",
        "F51_backward_pass_enabled",
        "backward",
        "Backward pass ON with default stitch similarity 0.60. ~2x processing time.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_backward_pass_enabled": "1",
        },
    ),
    _row(
        "f52_backward_tight",
        "F52_backward_stitch_tight",
        "backward",
        "Backward pass with tight stitch similarity (0.75). Fewer but more confident stitches.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_backward_pass_enabled": "1",
            "sway_backward_stitch_min_similarity": "0.75",
        },
    ),
    _row(
        "f53_backward_loose",
        "F53_backward_stitch_loose",
        "backward",
        "Backward pass with loose stitch similarity (0.45). More stitches, higher risk of wrong merges.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_backward_pass_enabled": "1",
            "sway_backward_stitch_min_similarity": "0.45",
        },
    ),
    _row(
        "f54_collision_greedy",
        "F54_collision_greedy_baseline",
        "collision",
        "Greedy sequential matching (current behavior). Regression baseline.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_collision_solver": "greedy",
        },
    ),
    _row(
        "f55_collision_hungarian",
        "F55_collision_hungarian",
        "collision",
        "Default lean core: Hungarian N-by-N assignment.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_collision_solver": "hungarian",
        },
    ),
    _row(
        "f56_collision_dp",
        "F56_collision_dp_full",
        "collision",
        "DP solver for <=5 tracks + Hungarian fallback for larger clusters.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_collision_solver": "dp",
        },
    ),
    # ── Section 13.8: Advanced Module Experiment Presets (I6) ─────────────
    _row(
        "f60_mote",
        "F60_advanced_mote_disocclusion",
        "advanced",
        "MOTE optical flow disocclusion matrix ON. RAFT-small. Predicts re-emergence location after crossovers.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_mote_disocclusion": "1",
        },
    ),
    _row(
        "f61_mote_raft_large",
        "F61_advanced_mote_raft_large",
        "advanced",
        "MOTE with RAFT-large (higher quality flow, slower).",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_mote_disocclusion": "1",
            "sway_mote_flow_model": "raft_large",
        },
    ),
    _row(
        "f62_sentinel",
        "F62_advanced_sentinel_sbm",
        "advanced",
        "Sentinel Survival Boosting ON. Grace multiplier 3.0, weak det conf 0.08.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_sentinel_sbm": "1",
        },
    ),
    _row(
        "f63_sentinel_aggressive",
        "F63_advanced_sentinel_aggressive",
        "advanced",
        "Sentinel with grace multiplier 5.0 and weak det conf 0.05. More aggressive track preservation.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_sentinel_sbm": "1",
            "sway_sentinel_grace_multiplier": "5.0",
            "sway_sentinel_weak_det_conf": "0.05",
        },
    ),
    _row(
        "f64_umot",
        "F64_advanced_umot_backtrack",
        "advanced",
        "UMOT Historical Backtracking ON. History length 500 frames.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_umot_backtrack": "1",
        },
    ),
    _row(
        "f65_umot_long_history",
        "F65_advanced_umot_long_history",
        "advanced",
        "UMOT with history length 1000 frames. For very long dormancy recovery.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_umot_backtrack": "1",
            "sway_umot_history_length": "1000",
        },
    ),
    _row(
        "f66_mote_sentinel",
        "F66_advanced_mote_plus_sentinel",
        "advanced",
        "MOTE + Sentinel together. Only run after F60 and F62 show individual benefit.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_mote_disocclusion": "1",
            "sway_sentinel_sbm": "1",
        },
    ),
    _row(
        "f67_matr_branch",
        "F67_advanced_matr_full_replace",
        "advanced",
        "MATR as tracker engine (full architecture replacement). Separate experiment branch.",
        {
            "sway_tracker_engine": "matr",
        },
    ),
    # ── Section 13.9: Full-Stack Experiment Presets (I5+) ─────────────────
    _row(
        "f70_production_candidate_a",
        "F70_prod_lean_backward",
        "all_future_modules",
        "Lean core + backward pass. Conservative — no advanced modules.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_backward_pass_enabled": "1",
            "sway_critique_dimensions": "formation,timing,extension,smoothness,sync",
        },
    ),
    _row(
        "f71_production_candidate_b",
        "F71_prod_lean_backward_mote",
        "all_future_modules",
        "Lean core + backward pass + MOTE. Tests the full occlusion-recovery stack.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_backward_pass_enabled": "1",
            "sway_mote_disocclusion": "1",
            "sway_critique_dimensions": "formation,timing,extension,smoothness,sync",
        },
    ),
    _row(
        "f72_production_candidate_c",
        "F72_prod_lean_backward_full_reid",
        "all_future_modules",
        "Lean core + backward pass + all 6 Re-ID signals. Maximum identity accuracy.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_backward_pass_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_reid_w_skeleton": "0.20",
            "sway_reid_w_face": "0.20",
            "sway_critique_dimensions": "formation,timing,extension,smoothness,sync",
        },
    ),
    _row(
        "f73_production_candidate_d",
        "F73_prod_full_stack",
        "all_future_modules",
        "Lean core + backward pass + best advanced module + full Re-ID + fine-tuned weights. Theoretical maximum.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_collision_solver": "hungarian",
            "sway_backward_pass_enabled": "1",
            "sway_reid_part_model": "bpbreid",
            "sway_reid_kpr_enabled": "1",
            "sway_reid_w_skeleton": "0.20",
            "sway_reid_w_face": "0.20",
            "sway_pose_mask_guided": "1",
            "sway_lift_backend": "motionbert",
            "sway_critique_dimensions": "formation,timing,extension,smoothness,sync",
        },
    ),
    _row(
        "f74_production_speed",
        "F74_prod_lean_speed_optimized",
        "all_future_modules",
        "Lean core with RT-DETR (faster) + SAM2 base + ViTPose-Large + no backward pass. Best accuracy-per-minute.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_detector_primary": "rt_detr_l",
            "sway_sam2_model": "sam2.1_b",
            "pose_model": "ViTPose-Large",
            "sway_backward_pass_enabled": "0",
            "sway_critique_dimensions": "formation,timing,extension,smoothness,sync",
        },
    ),
    _row(
        "f75_production_quality",
        "F75_prod_lean_quality_max",
        "all_future_modules",
        "Lean core with Co-DINO + SAM2 huge + ViTPose-Huge + backward pass + MOTE. Maximum quality.",
        {
            "sway_tracker_engine": "sam2mot",
            "sway_enrollment_enabled": "1",
            "sway_coi_enabled": "1",
            "sway_detector_hybrid": "1",
            "sway_detector_primary": "co_dino",
            "sway_sam2_model": "sam2.1_h",
            "pose_model": "ViTPose-Huge",
            "sway_pose_mask_guided": "1",
            "sway_collision_solver": "hungarian",
            "sway_backward_pass_enabled": "1",
            "sway_mote_disocclusion": "1",
            "sway_lift_backend": "motionbert",
            "sway_critique_dimensions": "formation,timing,extension,smoothness,sync",
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
