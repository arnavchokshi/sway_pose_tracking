"""
Pipeline Lab: schema of tunable parameters grouped by pipeline phase.

Used by pipeline_lab server (/api/schema) and the React UI. Many entries map to
environment variables read in tracker.py or main.py; YAML params can also set
any key starting with SWAY_ (applied in main.py before tracking).

Stage order and ``main_phases`` match ``docs/PIPELINE_CODE_REFERENCE.md`` and
``main.py`` progress lines ``[1/11]`` … ``[11/11]`` (Lab groups some steps into
one tab, e.g. phases 6–7 in ``reid_dedup``).

Only options that change behavior are listed; read-only ``info`` rows document
what the pipeline always does so the UI does not offer unwired presets.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Ordered stages: ``main_phases`` is shown in the Lab flowchart for A/B alignment with main.py.
PIPELINE_STAGES: List[Dict[str, Any]] = [
    {
        "id": "detection",
        "label": "Detection — YOLO & streaming",
        "short": "YOLO",
        "main_phases": "1–2",
    },
    {
        "id": "tracking",
        "label": "Tracking — BoxMOT / BoT-SORT",
        "short": "Track",
        "main_phases": "1–2",
    },
    {
        "id": "hybrid_sam",
        "label": "Hybrid SAM (overlap refiner)",
        "short": "SAM",
        "main_phases": "1–2",
    },
    {
        "id": "phase3_stitch",
        "label": "Post-track stitch & global link",
        "short": "Stitch",
        "main_phases": "3",
    },
    {
        "id": "pre_pose_prune",
        "label": "Pre-pose pruning",
        "short": "Prune",
        "main_phases": "4",
    },
    {
        "id": "pose",
        "label": "Pose (ViTPose)",
        "short": "Pose",
        "main_phases": "5",
    },
    {
        "id": "reid_dedup",
        "label": "Re-ID, crossover & collision cleanup",
        "short": "Re-ID",
        "main_phases": "6–7",
    },
    {
        "id": "post_pose_prune",
        "label": "Post-pose pruning",
        "short": "Post-prune",
        "main_phases": "8",
    },
    {
        "id": "smooth",
        "label": "Temporal smoothing",
        "short": "Smooth",
        "main_phases": "9",
    },
    {
        "id": "scoring",
        "label": "Scoring",
        "short": "Score",
        "main_phases": "10",
    },
    {
        "id": "export",
        "label": "Export",
        "short": "Export",
        "main_phases": "11",
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
    return d


# Single list: UI groups by `phase` (order here matches coarse pipeline left-to-right).
PIPELINE_PARAM_FIELDS: List[Dict[str, Any]] = [
    # --- detection (main.py phases 1–2, YOLO side) ---
    _f(
        "info_detection_yolo",
        "detection",
        "YOLO (Phase 1)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "Phase 1 in main.py: person detection. Lab default weights are COCO yolo26l (after prefetch). "
            "For DanceTrack fine-tunes, copy best.pt to models/yolo26l_dancetrack.pt and select "
            "yolo26l_dancetrack. YOLO11 L/X remain optional legacy fallbacks."
        ),
    ),
    _f(
        "sway_yolo_weights",
        "detection",
        "Detection model",
        "enum",
        "yolo26l",
        binding="env",
        key="SWAY_YOLO_WEIGHTS",
        choices=[
            "yolo26s",
            "yolo26l",
            "yolo26l_dancetrack",
            "yolo26x",
            "yolo26x_dancetrack",
            "yolov11l",
            "yolov11x",
        ],
        tier=1,
        display="model_cards",
        description=(
            "YOLO26 L/X are the primary detectors. DanceTrack fine-tunes need the matching .pt in models/ "
            "(Lab shows a badge until the file exists). YOLO11 L/X stay as optional baselines."
        ),
    ),
    _f(
        "sway_pretrack_nms_iou",
        "detection",
        "Ghost box suppression (pre-track IoU NMS)",
        "float",
        0.50,
        binding="env",
        key="SWAY_PRETRACK_NMS_IOU",
        min_=0.40,
        max_=0.90,
        tier=2,
        display="slider",
        description=(
            "Lower = drop more overlapping duplicate boxes before the tracker (dense formations). "
            "Higher = keep more raw YOLO boxes. Maps to classical IoU NMS in tracker.py."
        ),
    ),
    _f(
        "sway_group_video",
        "detection",
        "Group / crowd video mode",
        "bool",
        True,
        binding="env",
        key="SWAY_GROUP_VIDEO",
        description="Higher YOLO resolution in crowded scenes. Matches main.py stack default (setdefault SWAY_GROUP_VIDEO=1).",
        advanced=True,
        tier=3,
    ),
    _f(
        "sway_chunk_size",
        "detection",
        "Streaming chunk size (frames)",
        "int",
        300,
        binding="env",
        key="SWAY_CHUNK_SIZE",
        min_=30,
        max_=2000,
    ),
    _f(
        "sway_yolo_conf",
        "detection",
        "Detection confidence",
        "float",
        0.22,
        binding="env",
        key="SWAY_YOLO_CONF",
        min_=0.05,
        max_=0.95,
    ),
    _f(
        "sway_detect_size",
        "detection",
        "YOLO letterbox size (px)",
        "int",
        640,
        binding="env",
        key="SWAY_DETECT_SIZE",
        min_=320,
        max_=1920,
        description="tracker.py: SWAY_DETECT_SIZE (default 640; group mode may raise effective size).",
        advanced=True,
    ),
    _f(
        "sway_yolo_detection_stride",
        "detection",
        "YOLO detection stride",
        "int",
        1,
        binding="env",
        key="SWAY_YOLO_DETECTION_STRIDE",
        min_=1,
        max_=8,
        description="Run YOLO every N frames; tracker interpolates. SWAY_YOLO_DETECTION_STRIDE in tracker.py.",
        advanced=True,
    ),
    # --- tracking (main.py phase 2) ---
    _f(
        "info_tracking_backends",
        "tracking",
        "Tracker backend (Phase 2)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "Phase 2 in main.py: association in the streaming pass (BoxMOT Deep OC-SORT default, or Ultralytics BoT-SORT). "
            "Hybrid SAM refines boxes before the tracker on the BoxMOT path only. Phase 3 (global stitch) is the next Lab tab."
        ),
    ),
    _f(
        "tracker_technology",
        "tracking",
        "Tracker backend",
        "enum",
        "BoxMOT",
        binding="none",
        key="",
        choices=["BoxMOT", "BoT-SORT", "ByteTrack", "OC-SORT", "StrongSORT"],
        disabled_choices=["ByteTrack", "OC-SORT", "StrongSORT"],
        tier=1,
        description=(
            "Wired today: BoxMOT (Deep OC-SORT) or Ultralytics BoT-SORT. ByteTrack / OC-SORT / StrongSORT are "
            "planned — select them to see the label; runs are blocked until integrated."
        ),
    ),
    _f(
        "sway_boxmot_max_age",
        "tracking",
        "Lost track memory (frames)",
        "int",
        150,
        binding="env",
        key="SWAY_BOXMOT_MAX_AGE",
        min_=60,
        max_=300,
        tier=2,
        display="slider",
        description=(
            "How long the tracker keeps an ID after a dancer disappears. Raise for exit/re-entry; lower to avoid "
            "wrong reattachments. ~150 ≈ 5s at 30fps."
        ),
    ),
    _f(
        "sway_boxmot_match_thresh",
        "tracking",
        "Track assignment strictness (IoU gate)",
        "float",
        0.30,
        binding="env",
        key="SWAY_BOXMOT_MATCH_THRESH",
        min_=0.20,
        max_=0.50,
        tier=2,
        display="slider",
        description="Lower → more merges; higher → more ID fragments (BoxMOT DeepOcSort iou_threshold).",
    ),
    _f(
        "sway_boxmot_reid_on",
        "tracking",
        "Appearance Re-ID (for distinct costumes)",
        "bool",
        False,
        binding="env",
        key="SWAY_BOXMOT_REID_ON",
        tier=1,
        description=(
            "Enable when dancers wear visually distinct outfits (BoxMOT appearance embeddings). "
            "Disable for matching or uniform costumes."
        ),
    ),
    _f(
        "sway_boxmot_reid_model",
        "tracking",
        "Re-ID backbone preset",
        "enum",
        "osnet_x0_25",
        binding="reid_model_preset",
        key="",
        choices=["osnet_x0_25", "osnet_x1_0"],
        visible_when_field="sway_boxmot_reid_on",
        visible_when_value=True,
        advanced=True,
        tier=3,
        description="Used when Re-ID is on and no custom Re-ID weights path is set.",
    ),
    _f(
        "sway_boxmot_assoc_metric",
        "tracking",
        "Association distance metric",
        "enum",
        "IoU",
        binding="env",
        key="SWAY_BOXMOT_ASSOC_METRIC",
        choices=["IoU", "GIoU", "DIoU"],
        advanced=True,
        tier=3,
        description=(
            "BoxMOT association function for motion matching. (Library has no EIoU; DIoU is the closest "
            "center-aware option.)"
        ),
    ),
    _f(
        "sway_tracker_yaml",
        "tracking",
        "BoT-SORT YAML config path",
        "string",
        "",
        binding="env",
        key="SWAY_TRACKER_YAML",
        description="Only used when BoxMOT is off.",
        advanced=True,
    ),
    _f(
        "sway_boxmot_reid_weights",
        "tracking",
        "BoxMOT Re-ID weights path",
        "string",
        "",
        binding="env",
        key="SWAY_BOXMOT_REID_WEIGHTS",
        advanced=True,
    ),
    # --- hybrid_sam (runs inside the phase 1–2 pass; BoxMOT only) ---
    _f(
        "sway_hybrid_sam_overlap",
        "hybrid_sam",
        "Hybrid SAM on heavy overlap",
        "bool",
        True,
        binding="env",
        key="SWAY_HYBRID_SAM_OVERLAP",
        tier=1,
        description=(
            "BoxMOT path only: SAM2 segmentation tightens boxes when dancers overlap beyond the IoU trigger. "
            "Adds roughly 15–20% runtime on typical videos when overlap is frequent. "
            "With BoT-SORT, the Lab hides this phase (Ultralytics track() does not use the refiner)."
        ),
    ),
    _f(
        "sway_hybrid_sam_iou_trigger",
        "hybrid_sam",
        "Hybrid SAM IoU trigger",
        "float",
        0.42,
        binding="env",
        key="SWAY_HYBRID_SAM_IOU_TRIGGER",
        min_=0.25,
        max_=0.65,
        tier=2,
        display="slider",
        description=(
            "SAM runs when any pair of person boxes exceeds this IoU. "
            "Higher = fewer SAM calls (faster); lower = more overlap refinement."
        ),
    ),
    _f(
        "sway_hybrid_sam_min_dets",
        "hybrid_sam",
        "Hybrid SAM min person count",
        "int",
        2,
        binding="env",
        key="SWAY_HYBRID_SAM_MIN_DETS",
        min_=2,
        max_=30,
        advanced=True,
    ),
    _f(
        "sway_hybrid_sam_weights",
        "hybrid_sam",
        "Hybrid SAM checkpoint",
        "string",
        "sam2.1_b.pt",
        binding="env",
        key="SWAY_HYBRID_SAM_WEIGHTS",
        advanced=True,
    ),
    _f(
        "sway_hybrid_sam_mask_thresh",
        "hybrid_sam",
        "Hybrid SAM mask threshold",
        "float",
        0.5,
        binding="env",
        key="SWAY_HYBRID_SAM_MASK_THRESH",
        min_=0.05,
        max_=0.99,
        description="Binarize SAM masks at this probability (hybrid_sam_refiner.py).",
        advanced=True,
    ),
    _f(
        "sway_hybrid_sam_bbox_pad",
        "hybrid_sam",
        "Hybrid SAM bbox pad (px)",
        "int",
        2,
        binding="env",
        key="SWAY_HYBRID_SAM_BBOX_PAD",
        min_=0,
        max_=64,
        advanced=True,
    ),
    # --- phase3_stitch (main.py phase 3) ---
    _f(
        "info_phase3_stitch",
        "phase3_stitch",
        "Post-track stitching (Phase 3)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "main.py [3/11]: dormant merges, fragment stitch, coalesce, then optional global long-range link "
            "(maybe_global_stitch). Neural AFLink runs when weights exist unless forced to heuristic; see "
            "models/AFLink_epoch20.pth or SWAY_AFLINK_WEIGHTS."
        ),
    ),
    _f(
        "sway_global_link",
        "phase3_stitch",
        "Global track link",
        "bool",
        True,
        binding="env",
        key="SWAY_GLOBAL_LINK",
        tier=2,
        description="SWAY_GLOBAL_LINK: long-range stitch after dormant merges (global_track_link.py).",
    ),
    _f(
        "sway_stitch_max_frame_gap",
        "phase3_stitch",
        "Max gap to bridge between track fragments (frames)",
        "int",
        60,
        binding="env",
        key="SWAY_STITCH_MAX_FRAME_GAP",
        min_=30,
        max_=150,
        tier=2,
        display="slider",
        description=(
            "Fragment stitch horizon after post-track dormant logic. Raise for long absences; lower to avoid "
            "wrong merges on short clips. Default 60 ≈ 2s @ 30fps."
        ),
    ),
    _f(
        "sway_dormant_max_gap",
        "phase3_stitch",
        "Dormant track merge max gap (frames)",
        "int",
        150,
        binding="env",
        key="SWAY_DORMANT_MAX_GAP",
        min_=90,
        max_=300,
        advanced=True,
        tier=3,
        display="slider",
        description="Upper gap (in frames) for dormant-style relinking before fragment stitch (dormant_tracks.py).",
    ),
    _f(
        "sway_global_aflink_mode",
        "phase3_stitch",
        "Global link AFLink mode",
        "enum",
        "neural_if_available",
        binding="none",
        key="",
        choices=["neural_if_available", "force_heuristic"],
        description=(
            "neural_if_available: use AFLink when weights exist (default). "
            "force_heuristic: set SWAY_GLOBAL_AFLINK=0 so heuristic stitch is always used."
        ),
    ),
    _f(
        "sway_aflink_weights",
        "phase3_stitch",
        "AFLink weights path",
        "string",
        "",
        binding="env",
        key="SWAY_AFLINK_WEIGHTS",
        description="Override path to AFLink_epoch20.pth (see global_track_link.resolve_aflink_weights).",
        advanced=True,
    ),
    # --- pre_pose_prune (main.py phase 4, YAML) ---
    _f(
        "info_pre_pose_prune",
        "pre_pose_prune",
        "Pre-pose pruning (Phase 4)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "main.py [4/11]: prune_tracks (duration + kinetic), stage polygon, spatial outliers, short tracks, "
            "audience region, mirrors, bbox heuristics, etc. No alternate ‘heuristic mode’ switch — tune YAML keys."
        ),
    ),
    _f(
        "min_duration_ratio",
        "pre_pose_prune",
        "Min duration ratio",
        "float",
        None,
        binding="yaml",
        key="min_duration_ratio",
        description="Optional override; omit for code default inside prune_tracks.",
        min_=0.0,
        max_=1.0,
        advanced=True,
    ),
    _f(
        "kinetic_std_frac",
        "pre_pose_prune",
        "Kinetic motion threshold (KINETIC_STD_FRAC)",
        "float",
        None,
        binding="yaml",
        key="KINETIC_STD_FRAC",
        description="Optional YAML override for duration/kinetic gate (main.py passes to prune_tracks when set).",
        advanced=True,
    ),
    # --- pose (main.py phase 5) ---
    _f(
        "info_pose_models",
        "pose",
        "Pose (Phase 5)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "main.py [5/11]: ViTPose+ on crops. POSE_VISIBILITY_THRESHOLD in params gates low-vis keypoints (main.py)."
        ),
    ),
    _f(
        "pose_model",
        "pose",
        "ViTPose checkpoint size",
        "enum",
        "ViTPose-Base",
        binding="cli",
        key="pose_model",
        choices=["ViTPose-Base", "ViTPose-Large", "ViTPose-Huge"],
        tier=1,
        description="Maps to --pose-model base|large|huge.",
    ),
    _f(
        "info_pose_alternates",
        "pose",
        "Future pose backends (not wired)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "DWPose and HRNet are competitive alternatives worth wiring later. "
            "YOLO-Pose / OpenPose / MediaPipe / PoseFormer are intentionally omitted from the Lab (unwired or "
            "ill-suited to the crop-based ViTPose pipeline)."
        ),
    ),
    _f(
        "pose_stride",
        "pose",
        "Pose stride",
        "enum",
        1,
        binding="cli",
        key="pose_stride",
        choices=[1, 2],
        tier=1,
        display="segmented",
        description="1 = every frame; 2 = faster (gaps filled by interpolation downstream).",
    ),
    _f(
        "pose_visibility_threshold",
        "pose",
        "Pose visibility threshold",
        "float",
        None,
        binding="yaml",
        key="POSE_VISIBILITY_THRESHOLD",
        description="Optional; main.py default 0.3 when omitted (params.get).",
        min_=0.0,
        max_=1.0,
        advanced=True,
    ),
    _f(
        "sway_3d_lift",
        "pose",
        "3D pose lift (MotionAGFormer)",
        "bool",
        True,
        binding="env",
        key="SWAY_3D_LIFT",
        tier=2,
        description=(
            "After temporal smoothing: lift 2D keypoints to 3D for pose_3d in data.json, "
            "Three.js viewer, and hybrid angle scoring. Requires vendor/MotionAGFormer (or "
            "SWAY_MOTIONAGFORMER_ROOT) and models/motionagformer-l-h36m.pth.tr."
        ),
    ),
    # --- reid_dedup (main.py phases 6–7) ---
    _f(
        "info_reid_actual",
        "reid_dedup",
        "Association & collision (Phases 6–7)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "main.py [6/11]: occlusion re-ID, crossover, acceleration audit. [7/11]: deduplicate_collocated_poses, "
            "sanitize_pose_bbox_consistency. Phase preview 04_phases_6_7.mp4 covers both."
        ),
    ),
    _f(
        "sway_reid_debug",
        "reid_dedup",
        "Re-ID debug logging",
        "bool",
        False,
        binding="env",
        key="SWAY_REID_DEBUG",
        description="Verbose Re-ID / crossover traces (SWAY_REID_DEBUG=1).",
    ),
    _f(
        "reid_max_frame_gap",
        "reid_dedup",
        "Re-ID max frame gap",
        "int",
        None,
        binding="yaml",
        key="REID_MAX_FRAME_GAP",
        description="Passed to apply_occlusion_reid when set (main.py).",
        min_=1,
        max_=500,
        advanced=True,
    ),
    _f(
        "reid_min_oks",
        "reid_dedup",
        "Re-ID min OKS",
        "float",
        None,
        binding="yaml",
        key="REID_MIN_OKS",
        description="Passed to apply_occlusion_reid when set (main.py).",
        min_=0.0,
        max_=1.0,
        advanced=True,
    ),
    # --- post_pose_prune (main.py phase 8) ---
    _f(
        "info_post_pose_tiers",
        "post_pose_prune",
        "Post-pose pruning (Phase 8)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "main.py [8/11]: Tier C (ultra-low skeleton), Tier B weighted voting (sync, mirror, edge, jitter, …). "
            "POST_PRUNE_MODE / REID_STRATEGY-style env switches are not wired — use YAML keys below."
        ),
    ),
    _f(
        "confirmed_human_min_span_frac",
        "post_pose_prune",
        "Min video span for dancer protection",
        "float",
        0.10,
        binding="yaml",
        key="CONFIRMED_HUMAN_MIN_SPAN_FRAC",
        min_=0.05,
        max_=0.25,
        tier=2,
        description=(
            "Tier A confirmed-human whitelist: track temporal span must exceed this fraction of the video. "
            "Lower protects late entrants; higher is stricter against short false positives."
        ),
    ),
    _f(
        "tier_c_skeleton_mean",
        "post_pose_prune",
        "Tier C skeleton mean (TIER_C_SKELETON_MEAN)",
        "float",
        None,
        binding="yaml",
        key="TIER_C_SKELETON_MEAN",
        advanced=True,
    ),
    _f(
        "tier_c_low_frame_frac",
        "post_pose_prune",
        "Tier C low-frame fraction (TIER_C_LOW_FRAME_FRAC)",
        "float",
        None,
        binding="yaml",
        key="TIER_C_LOW_FRAME_FRAC",
        min_=0.0,
        max_=1.0,
        advanced=True,
    ),
    _f(
        "mean_confidence_min",
        "post_pose_prune",
        "Mean keypoint confidence min",
        "float",
        None,
        binding="yaml",
        key="MEAN_CONFIDENCE_MIN",
        description="Optional YAML override; pipeline default ~0.45 (main.py).",
        advanced=True,
    ),
    _f(
        "edge_margin_frac",
        "post_pose_prune",
        "Edge margin (EDGE_MARGIN_FRAC)",
        "float",
        None,
        binding="yaml",
        key="EDGE_MARGIN_FRAC",
        description="Confirmed-human edge band + mirror heuristics; default 0.15 in main.py.",
        min_=0.0,
        max_=0.5,
        advanced=True,
    ),
    _f(
        "edge_presence_frac",
        "post_pose_prune",
        "Edge presence (EDGE_PRESENCE_FRAC)",
        "float",
        None,
        binding="yaml",
        key="EDGE_PRESENCE_FRAC",
        min_=0.0,
        max_=1.0,
        advanced=True,
    ),
    _f(
        "min_lower_body_conf_yaml",
        "post_pose_prune",
        "Min lower-body confidence",
        "float",
        None,
        binding="yaml",
        key="min_lower_body_conf",
        description="Mirror / tier-B related; main.py default 0.3 when omitted.",
        min_=0.0,
        max_=1.0,
        advanced=True,
    ),
    _f(
        "jitter_ratio_max",
        "post_pose_prune",
        "Jitter ratio max (JITTER_RATIO_MAX)",
        "float",
        None,
        binding="yaml",
        key="JITTER_RATIO_MAX",
        min_=0.0,
        max_=1.0,
        advanced=True,
    ),
    _f(
        "sync_score_min",
        "post_pose_prune",
        "SYNC_SCORE_MIN (Tier B)",
        "float",
        None,
        binding="yaml",
        key="SYNC_SCORE_MIN",
        description="Optional YAML override for sync-strength gate; code default 0.10.",
        min_=0.0,
        max_=1.0,
        advanced=True,
    ),
    _f(
        "prune_threshold",
        "post_pose_prune",
        "Prune vote aggressiveness (PRUNE_THRESHOLD)",
        "float",
        0.65,
        binding="yaml",
        key="PRUNE_THRESHOLD",
        description=(
            "Weighted Tier-B vote threshold. Higher = prune more aggressively when rules fire. "
            "Lower = keep more tracks (risk retaining false positives)."
        ),
        min_=0.40,
        max_=0.90,
        tier=2,
        display="slider",
    ),
    _f(
        "pruning_w_low_sync",
        "post_pose_prune",
        "Rule weight · low sync",
        "float",
        0.7,
        binding="yaml_pruning_weight",
        key="prune_low_sync_tracks",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
    ),
    _f(
        "pruning_w_smart_mirror",
        "post_pose_prune",
        "Rule weight · smart mirrors",
        "float",
        0.9,
        binding="yaml_pruning_weight",
        key="prune_smart_mirrors",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
    ),
    _f(
        "pruning_w_completeness",
        "post_pose_prune",
        "Rule weight · completeness audit",
        "float",
        0.6,
        binding="yaml_pruning_weight",
        key="prune_completeness_audit",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
    ),
    _f(
        "pruning_w_head_only",
        "post_pose_prune",
        "Rule weight · head-only tracks",
        "float",
        0.8,
        binding="yaml_pruning_weight",
        key="prune_head_only_tracks",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
    ),
    _f(
        "pruning_w_low_conf",
        "post_pose_prune",
        "Rule weight · low keypoint confidence",
        "float",
        0.5,
        binding="yaml_pruning_weight",
        key="prune_low_confidence_tracks",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
    ),
    _f(
        "pruning_w_jittery",
        "post_pose_prune",
        "Rule weight · jittery tracks",
        "float",
        0.5,
        binding="yaml_pruning_weight",
        key="prune_jittery_tracks",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
    ),
    # --- smooth (main.py phase 9) ---
    _f(
        "info_smooth_one_euro",
        "smooth",
        "Smoothing (Phase 9)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "main.py [9/11]: One-Euro filter (PoseSmoother) plus optional temporal keypoint refine. "
            "Extra smoothers (Savitzky–Golay, Gaussian, learned temporal models) are experimental / not in the Lab UI yet."
        ),
    ),
    _f(
        "temporal_pose_refine",
        "smooth",
        "Temporal keypoint refine (±N frames)",
        "bool",
        True,
        binding="cli",
        key="temporal_pose_refine",
        tier=1,
        description=(
            "Light confidence-weighted (x,y) blend over neighboring frames per track. "
            "Radius sets the half-window N."
        ),
    ),
    _f(
        "temporal_pose_radius",
        "smooth",
        "Temporal refine radius (frames)",
        "int",
        2,
        binding="cli",
        key="temporal_pose_radius",
        description="Half-window for temporal refine; main.py clamps to 0–8.",
        min_=0,
        max_=8,
        tier=1,
    ),
    _f(
        "smoother_min_cutoff",
        "smooth",
        "One-Euro min_cutoff",
        "float",
        1.0,
        binding="yaml",
        key="SMOOTHER_MIN_CUTOFF",
        min_=0.01,
        max_=10.0,
        advanced=True,
    ),
    _f(
        "smoother_beta",
        "smooth",
        "One-Euro beta",
        "float",
        0.7,
        binding="yaml",
        key="SMOOTHER_BETA",
        min_=0.0,
        max_=5.0,
        advanced=True,
    ),
    # --- scoring (main.py phase 10) ---
    _f(
        "info_scoring_actual",
        "scoring",
        "Scoring (Phase 10)",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "main.py [10/11]: process_all_frames_scoring_vectorized (group circmean, cDTW, per-joint deviations). "
            "Alternate scoring modes are not exposed as switches."
        ),
    ),
    # --- export (main.py phase 11) ---
    _f(
        "montage",
        "export",
        "Stitch montage MP4",
        "bool",
        False,
        binding="cli",
        key="montage",
    ),
    _f(
        "save_phase_previews",
        "export",
        "Save phase preview clips",
        "bool",
        True,
        binding="cli",
        key="save_phase_previews",
        description="Default on in Lab. When true, worker passes --save-phase-previews (phase_previews/*.mp4).",
    ),
]


def schema_payload() -> Dict[str, Any]:
    return {
        "stages": PIPELINE_STAGES,
        "fields": PIPELINE_PARAM_FIELDS,
    }
