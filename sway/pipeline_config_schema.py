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
        "label": "Spotting people in the video",
        "short": "Spot people",
        "main_phases": "1–2",
    },
    {
        "id": "tracking",
        "label": "Keeping the same label on each dancer",
        "short": "Track IDs",
        "main_phases": "1–2",
    },
    {
        "id": "hybrid_sam",
        "label": "Sharper boxes when dancers touch or overlap",
        "short": "Overlap",
        "main_phases": "1–2",
    },
    {
        "id": "phase3_stitch",
        "label": "Merging broken IDs across the clip",
        "short": "Merge IDs",
        "main_phases": "3",
    },
    {
        "id": "pre_pose_prune",
        "label": "Removing obvious non-dancers early",
        "short": "Early cuts",
        "main_phases": "4",
    },
    {
        "id": "pose",
        "label": "Estimating body pose (joints)",
        "short": "Body pose",
        "main_phases": "5",
    },
    {
        "id": "reid_dedup",
        "label": "Fixing wrong IDs and duplicate outlines",
        "short": "Fix IDs",
        "main_phases": "6–7",
    },
    {
        "id": "post_pose_prune",
        "label": "Dropping weak or fake skeletons",
        "short": "Skeleton cleanup",
        "main_phases": "8",
    },
    {
        "id": "smooth",
        "label": "Smoothing shaky joints over time",
        "short": "Smoothing",
        "main_phases": "9",
    },
    {
        "id": "scoring",
        "label": "Computing dance sync and shape scores",
        "short": "Scoring",
        "main_phases": "10",
    },
    {
        "id": "export",
        "label": "Saving your outputs",
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
        "sway_yolo_weights",
        "detection",
        "How sharp the person finder is",
        "enum",
        "yolo26l",
        binding="env",
        key="SWAY_YOLO_WEIGHTS",
        choices=[
            "yolo26s",
            "yolo26l",
            "yolo26l_dancetrack",
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
        0.50,
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
        "sway_group_video",
        "detection",
        "Crowd mode (many people on screen)",
        "bool",
        True,
        binding="env",
        key="SWAY_GROUP_VIDEO",
        description=(
            "On for dance lines and groups: looks at the frame in more detail so small bodies are less likely "
            "to be missed. Slightly slower; turn off for solo or wide shots if you want maximum speed."
        ),
        advanced=True,
        tier=3,
    ),
    _f(
        "sway_chunk_size",
        "detection",
        "Frames processed per memory chunk",
        "int",
        300,
        binding="env",
        key="SWAY_CHUNK_SIZE",
        min_=30,
        max_=2000,
        tier=3,
        display="slider",
        description=(
            "Long videos are processed in chunks so memory stays stable. "
            "Only change this if you know you are hitting RAM limits or want to tune streaming."
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
        "sway_detect_size",
        "detection",
        "Internal resize width for finding people (pixels)",
        "int",
        640,
        binding="env",
        key="SWAY_DETECT_SIZE",
        min_=320,
        max_=1920,
        description=(
            "Larger = finer detail, slower. Crowd mode may bump this automatically. "
            "Leave at default unless you are tuning speed vs. small-figure detection."
        ),
        advanced=True,
        tier=3,
        display="slider",
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
            "The default engine is tuned for dance; you can switch to another engine for comparison. "
            "Overlap sharpening (next tab) only applies on the default path—not when using the alternate engine."
        ),
    ),
    _f(
        "tracker_technology",
        "tracking",
        "Which engine tracks IDs frame to frame",
        "enum",
        "BoxMOT",
        binding="none",
        key="",
        choices=["BoxMOT", "BoT-SORT", "ByteTrack", "OC-SORT", "StrongSORT"],
        disabled_choices=["ByteTrack", "OC-SORT", "StrongSORT"],
        tier=1,
        display="tracker_strip",
        description=(
            "Today you can run the built-in tracker or the alternate one from the same family as YOLO. "
            "The grayed-out names are placeholders for future work—you can’t run them yet."
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
        "sway_boxmot_reid_on",
        "tracking",
        "Use clothing / body look to tell dancers apart",
        "bool",
        False,
        binding="env",
        key="SWAY_BOXMOT_REID_ON",
        tier=1,
        description=(
            "Turn on when outfits look different—helps after crosses or occlusions. "
            "Turn off for identical costumes or uniforms so the pipeline doesn’t over-trust appearance."
        ),
    ),
    _f(
        "sway_boxmot_reid_model",
        "tracking",
        "Quality of the “who looks like who” model",
        "enum",
        "osnet_x0_25",
        binding="reid_model_preset",
        key="",
        choices=["osnet_x0_25", "osnet_x1_0"],
        visible_when_field="sway_boxmot_reid_on",
        visible_when_value=True,
        advanced=True,
        tier=3,
        description=(
            "Larger preset = heavier and usually better at telling similar people apart. "
            "Only matters when appearance matching above is on and you are not using a custom weights file."
        ),
    ),
    # --- hybrid_sam (runs inside the phase 1–2 pass; BoxMOT only) ---
    _f(
        "sway_hybrid_sam_overlap",
        "hybrid_sam",
        "Refine boxes when people overlap",
        "bool",
        True,
        binding="env",
        key="SWAY_HYBRID_SAM_OVERLAP",
        tier=1,
        description=(
            "When two dancers’ boxes sit heavily on top of each other, an extra segmentation pass tightens each box "
            "so poses don’t bleed together. Expect roughly 15–20% longer runs on clips with lots of contact. "
            "Only applies with the built-in tracker; the alternate engine path skips this step."
        ),
    ),
    _f(
        "sway_hybrid_sam_iou_trigger",
        "hybrid_sam",
        "How much overlap before the extra pass runs",
        "float",
        0.42,
        binding="env",
        key="SWAY_HYBRID_SAM_IOU_TRIGGER",
        min_=0.25,
        max_=0.65,
        tier=2,
        display="slider",
        description=(
            "Higher = only very overlapped pairs get fixed (faster, less segmentation). "
            "Lower = fix sooner (slower, cleaner lifts and partner work). "
            "With ROI crop on, lower values are cheaper than full-frame SAM."
        ),
    ),
    _f(
        "sway_hybrid_sam_roi_crop",
        "hybrid_sam",
        "Run SAM on overlap union crop (not full frame)",
        "bool",
        True,
        binding="env",
        key="SWAY_HYBRID_SAM_ROI_CROP",
        tier=2,
        advanced=True,
        description=(
            "When on, segmentation sees only a padded union of overlapped dancers — faster and "
            "usually the same accuracy. Turn off to force legacy full-frame SAM (debug / A-B)."
        ),
    ),
    _f(
        "sway_hybrid_sam_roi_pad_frac",
        "hybrid_sam",
        "ROI margin around overlapped dancers",
        "float",
        0.10,
        binding="env",
        key="SWAY_HYBRID_SAM_ROI_PAD_FRAC",
        min_=0.0,
        max_=0.35,
        advanced=True,
        tier=3,
        display="slider",
        description="Extra padding around the union box, as a fraction of union width/height.",
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
        "sway_global_link",
        "phase3_stitch",
        "Try to reconnect IDs across long gaps",
        "bool",
        True,
        binding="env",
        key="SWAY_GLOBAL_LINK",
        tier=2,
        description=(
            "On: look across the whole video for “probably the same person” after shorter-range fixes. "
            "Off: skip that pass—faster, but more duplicate IDs on messy footage."
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
        tier=2,
        display="segmented",
        description=(
            "Use the learned linker when its weights are present (recommended). "
            "Or always use simple geometry-based rules if you want predictable, lighter behavior."
        ),
    ),
    # --- pre_pose_prune (main.py phase 4, YAML) ---
    _f(
        "info_pre_pose_prune",
        "pre_pose_prune",
        "What this step does",
        "info",
        None,
        binding="none",
        key="",
        description=(
            "Before estimating joints, the pipeline drops tracks that are almost certainly not real dancers—"
            "too short on screen, barely moving, stuck in the crowd edge, etc. "
            "That saves time and keeps pose focused on people you care about. "
            "Fine tuning is through the numbers below, not a single on/off switch."
        ),
    ),
    _f(
        "min_duration_ratio",
        "pre_pose_prune",
        "Minimum fraction of the video a track must span",
        "float",
        0.20,
        binding="yaml",
        key="min_duration_ratio",
        description=(
            "Baseline 0.20 matches the pipeline default. "
            "Higher = only keep people visible for a larger share of the clip; lower = allow shorter appearances."
        ),
        min_=0.0,
        max_=1.0,
        advanced=True,
        display="slider",
    ),
    _f(
        "kinetic_std_frac",
        "pre_pose_prune",
        "How much motion counts as “alive”",
        "float",
        0.02,
        binding="yaml",
        key="KINETIC_STD_FRAC",
        min_=0.005,
        max_=0.08,
        description=(
            "Baseline 0.02 matches the pipeline default (fraction of typical box height). "
            "Raise if still objects are kept; lower if real dancers get cut as “not moving enough.”"
        ),
        advanced=True,
        display="slider",
    ),
    _f(
        "spatial_outlier_std_factor",
        "pre_pose_prune",
        "Max allowed distance from group center (Std Devs)",
        "float",
        2.0,
        binding="yaml",
        key="SPATIAL_OUTLIER_STD_FACTOR",
        min_=1.0,
        max_=5.0,
        advanced=True,
        tier=3,
        display="slider",
        description="Tracks further than this from the overall group are pruned as outliers. 2.0 = default.",
    ),
    _f(
        "bbox_size_min_frac",
        "pre_pose_prune",
        "Minimum allowed bounding box size",
        "float",
        0.40,
        binding="yaml",
        key="BBOX_SIZE_MIN_FRAC",
        min_=0.10,
        max_=1.0,
        advanced=True,
        tier=3,
        display="slider",
        description="Relative height against group median. Lower = keeps smaller boxes.",
    ),
    _f(
        "bbox_size_max_frac",
        "pre_pose_prune",
        "Maximum allowed bounding box size",
        "float",
        2.00,
        binding="yaml",
        key="BBOX_SIZE_MAX_FRAC",
        min_=1.0,
        max_=4.0,
        advanced=True,
        tier=3,
        display="slider",
        description="Relative height against group median. Higher = keeps larger boxes.",
    ),
    _f(
        "short_track_min_frac",
        "pre_pose_prune",
        "Minimum track lifespan (fraction of video)",
        "float",
        0.15,
        binding="yaml",
        key="SHORT_TRACK_MIN_FRAC",
        min_=0.0,
        max_=1.0,
        advanced=True,
        tier=3,
        display="slider",
        description="Any track shorter than this fraction of the video is removed as a ghost or passerby.",
    ),
    _f(
        "audience_region_x_min_frac",
        "pre_pose_prune",
        "Audience Corner X Start",
        "float",
        0.75,
        binding="yaml",
        key="AUDIENCE_REGION_X_MIN_FRAC",
        min_=0.0,
        max_=1.0,
        advanced=True,
        tier=3,
        display="slider",
        description="Left barrier for audience corner auto-delete (default 0.75 = right 25%).",
    ),
    _f(
        "audience_region_y_min_frac",
        "pre_pose_prune",
        "Audience Corner Y Start",
        "float",
        0.70,
        binding="yaml",
        key="AUDIENCE_REGION_Y_MIN_FRAC",
        min_=0.0,
        max_=1.0,
        advanced=True,
        tier=3,
        display="slider",
        description="Top barrier for audience corner auto-delete (default 0.70 = bottom 30%).",
    ),
    # --- pose (main.py phase 5) ---
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
        choices=["ViTPose-Base", "ViTPose-Large", "ViTPose-Huge", "RTMPose-L"],
        tier=1,
        display="model_cards",
        description=(
            "ViTPose+: larger = usually better on hard motion, slower. "
            "RTMPose-L needs a separate MMPose install (see docs/PIPELINE_IMPROVEMENTS_ROADMAP.md) "
            "and targets speed vs ViTPose-Base."
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
            "Sapiens / HMR / GNN tracking are roadmap experiments — not wired in the lab yet. "
            "RTMPose-L is available when MMPose is installed locally."
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
        description=(
            "Every frame = smoothest overlays, slowest. "
            "Every other frame = about twice as fast; missing frames are filled in later so playback still looks continuous."
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
        "sway_3d_lift",
        "pose",
        "Add a simple depth view (3D) for angles and viewer",
        "bool",
        True,
        binding="env",
        key="SWAY_3D_LIFT",
        tier=2,
        description=(
            "When on, joints get a third dimension so angled moves score more fairly and the 3D viewer has data. "
            "Needs the optional lift model and its folder installed; turn off if you only want flat 2D overlays."
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
        "reid_max_frame_gap",
        "reid_dedup",
        "Max frames apart to still consider “same person” after a gap",
        "int",
        90,
        binding="yaml",
        key="REID_MAX_FRAME_GAP",
        description=(
            "Baseline 90 ≈ 3 s at 30 fps (pipeline default). "
            "Larger = try harder to reconnect after long hides; smaller = fewer risky merges."
        ),
        min_=1,
        max_=500,
        advanced=True,
        display="slider",
    ),
    _f(
        "reid_min_oks",
        "reid_dedup",
        "How similar poses must look to merge IDs",
        "float",
        0.35,
        binding="yaml",
        key="REID_MIN_OKS",
        description=(
            "Baseline 0.35 matches the pipeline default (0–1 similarity). "
            "Higher = merge only when poses almost match; lower = merge more aggressively."
        ),
        min_=0.0,
        max_=1.0,
        advanced=True,
        display="slider",
    ),
    _f(
        "collision_kpt_dist_frac",
        "reid_dedup",
        "Max joint distance to trigger deduplication",
        "float",
        0.26,
        binding="yaml",
        key="COLLISION_KPT_DIST_FRAC",
        min_=0.10,
        max_=0.50,
        advanced=True,
        tier=3,
        display="slider",
        description="Fraction of bbox height. Median joint distance below this deletes one of the tracks.",
    ),
    _f(
        "collision_center_dist_frac",
        "reid_dedup",
        "Max bbox center distance for deduplication",
        "float",
        0.50,
        binding="yaml",
        key="COLLISION_CENTER_DIST_FRAC",
        min_=0.10,
        max_=1.0,
        advanced=True,
        tier=3,
        display="slider",
        description="Bbox centers must also be this close (fraction of height) for duplicate pose deletion.",
    ),
    _f(
        "dedup_torso_median_frac",
        "reid_dedup",
        "Max torso joint distance for deduplication",
        "float",
        0.24,
        binding="yaml",
        key="DEDUP_TORSO_MEDIAN_FRAC",
        min_=0.10,
        max_=0.50,
        advanced=True,
        tier=3,
        display="slider",
        description="Stricter distance check specifically across shoulders and hips to prevent merging partners.",
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
        "confirmed_human_min_span_frac",
        "post_pose_prune",
        "How much of the video someone must cover to be “protected”",
        "float",
        0.10,
        binding="yaml",
        key="CONFIRMED_HUMAN_MIN_SPAN_FRAC",
        min_=0.05,
        max_=0.25,
        tier=2,
        description=(
            "Long-running bodies are harder to auto-delete. "
            "Lower = late joiners still count as protected; higher = only people who are on screen a lot get that safety net."
        ),
    ),
    _f(
        "tier_c_skeleton_mean",
        "post_pose_prune",
        "Cutoff for “this skeleton is basically empty”",
        "float",
        0.15,
        binding="yaml",
        key="TIER_C_SKELETON_MEAN",
        min_=0.05,
        max_=0.45,
        description=(
            "Baseline 0.15 matches the pipeline default (mean joint confidence). "
            "Lower = drop weak skeletons sooner; higher = keep borderline bodies longer."
        ),
        advanced=True,
        display="slider",
    ),
    _f(
        "tier_c_low_frame_frac",
        "post_pose_prune",
        "How many weak frames before the harsh drop",
        "float",
        0.80,
        binding="yaml",
        key="TIER_C_LOW_FRAME_FRAC",
        description=(
            "Baseline 0.80 matches the pipeline default (share of weak frames needed to drop someone). "
            "Higher = only remove if most frames look bad (more lenient). Lower = remove with fewer weak frames (stricter)."
        ),
        min_=0.0,
        max_=1.0,
        advanced=True,
        display="slider",
    ),
    _f(
        "mean_confidence_min",
        "post_pose_prune",
        "Average joint strength must stay above this",
        "float",
        0.45,
        binding="yaml",
        key="MEAN_CONFIDENCE_MIN",
        min_=0.2,
        max_=0.85,
        description=(
            "Baseline 0.45 matches the pipeline default. "
            "Higher = prune mushy skeletons faster; lower = keep softer estimates."
        ),
        advanced=True,
        display="slider",
    ),
    _f(
        "edge_margin_frac",
        "post_pose_prune",
        "How deep the “edge of frame” band is",
        "float",
        0.15,
        binding="yaml",
        key="EDGE_MARGIN_FRAC",
        description=(
            "Baseline 0.15 matches the pipeline default. "
            "Wider band = treat more of the border as “probably audience or mirror.”"
        ),
        min_=0.0,
        max_=0.5,
        advanced=True,
        display="slider",
    ),
    _f(
        "edge_presence_frac",
        "post_pose_prune",
        "How often someone must hug the edge to count as “edge person”",
        "float",
        0.30,
        binding="yaml",
        key="EDGE_PRESENCE_FRAC",
        description=(
            "Baseline 0.30 matches the pipeline default. "
            "Works with the edge band to flag people who mostly stand at the sides."
        ),
        min_=0.0,
        max_=1.0,
        advanced=True,
        display="slider",
    ),
    _f(
        "min_lower_body_conf_yaml",
        "post_pose_prune",
        "Legs and hips must be this confident",
        "float",
        0.30,
        binding="yaml",
        key="min_lower_body_conf",
        description=(
            "Baseline 0.30 matches the pipeline default. "
            "Helps catch mirror doubles and half-visible bodies; raise if legs look trustworthy."
        ),
        min_=0.0,
        max_=1.0,
        advanced=True,
        display="slider",
    ),
    _f(
        "jitter_ratio_max",
        "post_pose_prune",
        "Allowed fraction of “twitchy” frames",
        "float",
        0.10,
        binding="yaml",
        key="JITTER_RATIO_MAX",
        description=(
            "Baseline 0.10 matches the pipeline default. "
            "Lower = stricter (noisy tracks drop sooner); higher = tolerate shaky video."
        ),
        min_=0.0,
        max_=1.0,
        advanced=True,
        display="slider",
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
        "pruning_w_completeness",
        "post_pose_prune",
        "How much “missing body parts” counts against a track",
        "float",
        0.6,
        binding="yaml_pruning_weight",
        key="prune_completeness_audit",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
        description="Broken silhouettes (torso but no legs) look less like full dancers.",
    ),
    _f(
        "pruning_w_head_only",
        "post_pose_prune",
        "How much “only a head visible” counts against a track",
        "float",
        0.8,
        binding="yaml_pruning_weight",
        key="prune_head_only_tracks",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
        description="Good for killing floating faces in the crowd; lower if your shot is honestly tight on faces.",
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
    _f(
        "pruning_w_jittery",
        "post_pose_prune",
        "How much shaky motion counts against a track",
        "float",
        0.5,
        binding="yaml_pruning_weight",
        key="prune_jittery_tracks",
        min_=0.0,
        max_=1.0,
        tier=2,
        display="pruning_weight",
        description="TV noise or compression artifacts often look like jitter—tune if real dancers get unfairly cut.",
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
            "Optional neighbor blending further softens noise using a few frames on either side."
        ),
    ),
    _f(
        "temporal_pose_refine",
        "smooth",
        "Blend each joint with nearby frames",
        "bool",
        True,
        binding="cli",
        key="temporal_pose_refine",
        tier=1,
        description=(
            "Looks a few frames before/after and nudges shaky points toward a local average. "
            "Turn off for fastest runs or when you want maximum raw responsiveness."
        ),
    ),
    _f(
        "temporal_pose_radius",
        "smooth",
        "How many frames on each side to blend",
        "int",
        2,
        binding="cli",
        key="temporal_pose_radius",
        description="Bigger window = smoother, slightly “heavier” motion; 0–8 allowed.",
        min_=0,
        max_=8,
        tier=1,
        display="slider",
    ),
    _f(
        "smoother_min_cutoff",
        "smooth",
        "Baseline smoothness (lower = more smoothing)",
        "float",
        1.0,
        binding="yaml",
        key="SMOOTHER_MIN_CUTOFF",
        min_=0.01,
        max_=10.0,
        description="Small values drag joints more; large values stay closer to the raw estimate.",
        advanced=True,
        display="slider",
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
]


def schema_payload() -> Dict[str, Any]:
    return {
        "stages": PIPELINE_STAGES,
        "fields": PIPELINE_PARAM_FIELDS,
    }
