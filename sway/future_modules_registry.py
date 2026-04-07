"""
Catalog of ideas from ``docs/FUTURE_MODULES_IDENTITY_AND_POSE_CROPS.md``.

Each entry documents intent and how (if at all) it maps to the current codebase.
Research-only rows are listed for traceability — they are not runnable toggles.
"""

from __future__ import annotations

from typing import Any, Dict, List

FutureModuleRow = Dict[str, Any]

FUTURE_MODULES: List[FutureModuleRow] = [
    # Part B — Detection & pre-track
    {
        "id": "b_multiscale_yolo",
        "part": "B",
        "title": "Multi-scale / tiled YOLO",
        "status": "partial",
        "notes": "Use larger weights (yolo26x) + stride; true multi-scale inference not wired.",
    },
    {
        "id": "b_temporal_det_fusion",
        "part": "B",
        "title": "Temporal detection fusion",
        "status": "planned",
        "notes": "Would sit before BoxMOT; not implemented — tracker + hybrid SAM absorb some flicker.",
    },
    {
        "id": "b_choreo_nms",
        "part": "B",
        "title": "Choreography-aware NMS",
        "status": "partial",
        "notes": "Tunable via sway_pretrack_nms_iou / sway_yolo_conf.",
        "schema_fields": ["sway_pretrack_nms_iou", "sway_yolo_conf"],
    },
    {
        "id": "b_keypoint_det",
        "part": "B",
        "title": "Keypoint-assisted detection",
        "status": "not_implemented",
    },
    {
        "id": "b_stereo_depth",
        "part": "B",
        "title": "Depth / stereo person prior",
        "status": "not_applicable_single_camera",
    },
    {
        "id": "b_synthetic_mining",
        "part": "B",
        "title": "Synthetic hard-negative mining",
        "status": "training_pipeline",
    },
    # Part C — Tracking
    {
        "id": "c_reid_backbone",
        "part": "C",
        "title": "Stronger Re-ID backbones",
        "status": "partial",
        "notes": "deep_ocsort_osnet + osnet_x0_25 preset; no CLIP/DINO in-track yet.",
        "schema_fields": ["tracker_technology", "sway_boxmot_reid_model"],
    },
    {
        "id": "c_embedding_banks",
        "part": "C",
        "title": "Track-specific embedding banks",
        "status": "inside_boxmot",
        "notes": "BoxMOT internal EMA; no separate knob.",
    },
    {
        "id": "c_motion_signature",
        "part": "C",
        "title": "Motion signature features",
        "status": "partial",
        "notes": "sway_bidirectional_track_pass merges forward/reverse trajectories.",
        "schema_fields": ["sway_bidirectional_track_pass"],
    },
    {
        "id": "c_trajectory_transformer",
        "part": "C",
        "title": "Trajectory Transformer matcher",
        "status": "research",
    },
    {
        "id": "c_optical_flow_roi",
        "part": "C",
        "title": "Optical flow in track ROI",
        "status": "not_implemented",
    },
    {
        "id": "c_global_lagrangian",
        "part": "C",
        "title": "Global Lagrangian assignment",
        "status": "not_implemented",
    },
    {
        "id": "c_neural_mot",
        "part": "C",
        "title": "Neural MOT (TrackFormer-style)",
        "status": "research",
    },
    {
        "id": "c_two_stage_identity_graph",
        "part": "C",
        "title": "Two-stage identity graph",
        "status": "partial",
        "notes": "Phase 3 stitch + optional GNN refine.",
        "schema_fields": ["sway_gnn_track_refine"],
    },
    {
        "id": "c_cross_camera",
        "part": "C",
        "title": "Cross-camera Re-ID",
        "status": "not_implemented",
    },
    {
        "id": "c_choreo_script",
        "part": "C",
        "title": "Choreography script prior",
        "status": "not_implemented",
    },
    # Part D — Post-track
    {
        "id": "d_gnn_plus",
        "part": "D",
        "title": "Global identity graph (GNN++)",
        "status": "partial",
        "schema_fields": ["sway_gnn_track_refine"],
    },
    {
        "id": "d_cycle_consistency",
        "part": "D",
        "title": "Cycle consistency check",
        "status": "inside_stitch",
    },
    {
        "id": "d_crossover_swap",
        "part": "D",
        "title": "Crossover swap search",
        "status": "partial",
        "notes": "dancer_registry crossover pass; handshake Hungarian.",
        "schema_fields": ["sway_phase13_mode"],
    },
    {
        "id": "d_dormant_mobile",
        "part": "D",
        "title": "Lightweight dormant embeddings",
        "status": "partial",
        "notes": "dancer_registry mode; not MobileNet-specific.",
        "schema_fields": ["sway_phase13_mode"],
    },
    {
        "id": "d_human_verify",
        "part": "D",
        "title": "Human verification queue",
        "status": "not_implemented",
    },
    # Part E — SAM / overlap
    {
        "id": "e_handshake_v2",
        "part": "E",
        "title": "Handshake / multi-object SAM",
        "status": "partial",
        "schema_fields": ["sway_phase13_mode", "sway_hybrid_sam_iou_trigger"],
    },
    {
        "id": "e_mask_flow_loss",
        "part": "E",
        "title": "Flow-consistent masks",
        "status": "not_implemented",
    },
    # Part F — Crops (implemented knobs)
    {
        "id": "f_smart_pad",
        "part": "F",
        "title": "Smart bbox pad (motion / aspect)",
        "status": "implemented",
        "notes": "SWAY_VITPOSE_SMART_PAD / smart_expand_bbox_xyxy in pose_estimator.",
    },
    {
        "id": "f_temporal_crop_smooth",
        "part": "F",
        "title": "Temporal crop smoothing + anti-jitter + foot/head bias",
        "status": "implemented",
        "schema_fields": [
            "sway_pose_crop_smooth_alpha",
            "sway_pose_crop_foot_bias_frac",
            "sway_pose_crop_head_bias_frac",
            "sway_pose_crop_anti_jitter_px",
        ],
    },
    {
        "id": "f_joint_margin_refiner",
        "part": "F",
        "title": "Joint-visible margin second pass",
        "status": "not_implemented",
    },
    {
        "id": "f_resolution_ladder",
        "part": "F",
        "title": "Resolution ladder",
        "status": "partial",
        "notes": "pose_model size + pose_stride; no auto gated rerun.",
        "schema_fields": ["pose_model", "pose_stride"],
    },
    {
        "id": "f_pose_guided_next",
        "part": "F",
        "title": "Pose-guided crop next frame",
        "status": "partial",
        "notes": "smart_expand uses prev tracker box velocity / lift heuristics.",
    },
    # Part G — Pose model
    {
        "id": "g_temporal_refine",
        "part": "G",
        "title": "Temporal pose refine (2D smooth)",
        "status": "implemented",
        "schema_fields": ["temporal_pose_refine", "temporal_pose_radius"],
    },
    {
        "id": "g_tta",
        "part": "G",
        "title": "TTA / multi-hypothesis pose",
        "status": "not_implemented",
    },
    # Part H — Product
    {
        "id": "h_rehearsal_mode",
        "part": "H",
        "title": "Rehearsal / canonical ID",
        "status": "partial",
        "notes": "dancer_registry captures appearance snapshots.",
        "schema_fields": ["sway_phase13_mode"],
    },
    {
        "id": "h_floor_grid",
        "part": "H",
        "title": "Floor grid homography",
        "status": "not_implemented",
    },
]


def registry_summary() -> Dict[str, int]:
    from collections import Counter

    c = Counter(str(m.get("status") or "unknown") for m in FUTURE_MODULES)
    return dict(c)
