# Configuration & Technology Catalog (Full Audit)

This catalog is now an exhaustive audit pass across:

- `docs/Future_Plans/FUTURE_PIPELINE.md`
- `docs/MASTER_PIPELINE_GUIDELINE.md`
- `docs/TECHNICAL_PIPELINE_PAPER.md`
- runtime/config code (`main.py`, `sway/*`, schema, and manifest)

Status legend used below:

- `implemented`: module/key exists in code
- `runtime wired`: used by active runtime paths (or partially wired)
- `sweep-ready now`: if a sweep run right now can select the feature and also tune/test its sub-config knobs (numeric/categorical toggles)

## Technology Matrix By Phase

| Phase | Technology / Option | Main switches | Implemented | Runtime wired | Sweep-ready now |
|---|---|---|---|---|---|
| 1 | YOLO detector path | `SWAY_YOLO_*`, `SWAY_DETECT_SIZE`, `SWAY_PRETRACK_NMS_IOU` | yes | yes | yes |
| 1 | DETR detector family (`rt_detr_l`, `rt_detr_x`, `co_detr`, `co_dino`) | `SWAY_DETECTOR_PRIMARY`, `SWAY_DETR_CONF` | yes | yes | yes |
| 1 | Hybrid detector (YOLO scout + DETR precision) | `SWAY_DETECTOR_HYBRID`, `SWAY_DETECTOR_PRECISION`, `SWAY_HYBRID_OVERLAP_IOU_TRIGGER`, `SWAY_DETECTION_UNCERTAIN_CONF` | yes | yes | yes |
| 2 | Classic BoxMOT tracking (Deep OC-SORT/StrongSORT/ByteTrack/BoTSORT/OCSORT/SolidTrack adapters) | `SWAY_USE_BOXMOT`, `SWAY_BOXMOT_*`, `SWAY_BOTSORT_*`, `SWAY_STRONGSORT_*`, `SWAY_BYTETRACK_*` | yes | yes | yes |
| 2 | Future tracker branch gate | `SWAY_TRACKER_ENGINE` | yes | yes | yes |
| 2 | SAM2MOT (`sam2mot`) | `SWAY_SAM2_*` | yes | yes | yes |
| 2 | SAM2 + MeMoSORT hybrid (`sam2_memosort_hybrid`) | `SWAY_SAM2_*`, `SWAY_MEMOSORT_*` | yes | yes | yes |
| 2 | MATR engine option (compat routing) | `SWAY_TRACKER_ENGINE=matr` | yes | yes | yes |
| 2 | Enrollment gallery | `SWAY_ENROLLMENT_*` | yes | yes | yes |
| 2 | COI module | `SWAY_COI_*` | yes | yes | yes |
| 3 | Post-track stitching and dormant merge | `SWAY_STITCH_*`, `SWAY_DORMANT_MAX_GAP`, `SWAY_SHORT_GAP_FRAMES` | yes | yes | yes |
| 3 | Global linker + AFLink | `SWAY_GLOBAL_LINK`, `SWAY_GLOBAL_AFLINK`, `SWAY_AFLINK_*` | yes | yes | yes |
| 3 | Coalescence | `SWAY_COALESCENCE_*` | yes | yes | yes |
| 3 | Phase 1-3 strategy modes (`standard`, `dancer_registry`, `sway_handshake`) | `SWAY_PHASE13_MODE`, `SWAY_REGISTRY_*`, `SWAY_HANDSHAKE_VERIFY_STRIDE` | yes | yes | yes |
| 4 | Pre-pose pruning / stage filtering | params YAML + `SWAY_STAGE_POLYGON`, `SWAY_AUTO_STAGE_DEPTH` | yes | yes | yes |
| 5 | ViTPose (base/large/huge) | `--pose-model`, `SWAY_VITPOSE_*` | yes | yes | yes |
| 5 | RTMPose | `--pose-model rtmpose` | yes | yes | yes |
| 5 | Sapiens path (TorchScript-backed) | `SWAY_SAPIENS_TORCHSCRIPT`, `SWAY_SAPIENS_*` | yes | yes | yes |
| 5 | Mask-guided pose wrapper | `SWAY_POSE_MASK_GUIDED`, `SWAY_POSE_*` | yes | yes | yes |
| 5 | Temporal pose refine | `SWAY_TEMPORAL_POSE_REFINE`, `SWAY_TEMPORAL_POSE_RADIUS` | yes | yes | yes |
| 5b | 3D lift (MotionAGFormer / MotionBERT / PoseFormerV2 paths) | `SWAY_3D_LIFT`, `SWAY_LIFT_*`, `SWAY_POSEFORMERV2_*`, `SWAY_MOTIONAGFORMER_*` | yes | yes | yes |
| 6 | Legacy occlusion Re-ID + crossover association | YAML params + `SWAY_REID_DEBUG` | yes | yes | yes |
| 6 | ReIDFusionEngine (part/color/spatial/skeleton/face fusion) | `SWAY_REID_PART_MODEL`, `SWAY_REID_W_*`, `SWAY_REID_*` | yes | yes | yes |
| 6 | Pose-gated EMA | `SWAY_REID_EMA_*` | yes | yes | yes |
| 6 | ReID fine-tune toolchain | `SWAY_REID_FINETUNE_*` | yes | yes | yes |
| 7 | Collision cleanup via crossover dedup | `DEDUP_MIN_PAIR_OKS`, `DEDUP_ANTIPARTNER_MIN_IOU` | yes | yes | yes |
| 7 | Collision solver module (hungarian/dp/greedy) | `SWAY_COLLISION_SOLVER`, `SWAY_COLLISION_DP_MAX_PERMUTATIONS`, `SWAY_COLLISION_MIN_TRACKS` | yes | yes | yes |
| 8 | Post-pose pruning tiers A/B/C | YAML params (`PRUNE_*`) | yes | yes | yes |
| 9 | Smoother (1-Euro + temporal options) | `SMOOTHER_BETA`, `SWAY_TEMPORAL_*` | yes | yes | yes |
| 10 | Scoring pipeline | scoring params + `SWAY_CRITIQUE_*` | yes | yes | yes |
| 10 | Critique engine dimensions | `SWAY_CRITIQUE_DIMENSIONS`, related `SWAY_CRITIQUE_*` | yes | yes | yes |
| 11 | Export / visualizer / sidecars | `SWAY_VIS_*`, `SWAY_HMR_MESH_SIDECAR`, CLI flags | yes | yes | yes |
| X | MOTE disocclusion | `SWAY_MOTE_*` | yes | yes | yes |
| X | SentinelSBM | `SWAY_SENTINEL_*` | yes | yes | yes |
| X | UMOT backtracking | `SWAY_UMOT_*` | yes | yes | yes |
| X | Backward pass + stitch | `SWAY_BACKWARD_*` | yes | yes | yes |

## Non-SWAY Configuration Surface (params + CLI)

These are also part of "all configurations", not only `SWAY_*`:

- CLI: `--pose-model`, `--pose-stride`, `--stop-after-boundary`, `--resume-from`, `--montage`, `--save-phase-previews`, `--save-live-artifacts`, `--phase-debug-jsonl`
- params/YAML high-signal keys: `PRUNE_THRESHOLD`, `SMOOTHER_BETA`, `POSE_VISIBILITY_THRESHOLD`, `DEDUP_MIN_PAIR_OKS`, `DEDUP_ANTIPARTNER_MIN_IOU`, `POSE_CROP_*`
- full parameter schema source: `sway/pipeline_config_schema.py` (`PIPELINE_PARAM_FIELDS`)

## Exhaustive SWAY Key Inventory (Docs + Manifest)

Coverage in this appendix:

- all `SWAY_*` keys explicitly mentioned by the three pipeline docs
- all sweep keys from `docs/LAMBDA_CONFIGURATION_MANIFEST.json`
- grouped here with explicit per-group wiring/readiness status

| Group | Keys | Implemented | Runtime wired | Sweep-ready now |
|---|---|---|---|---|
| `SWAY_3D_LIFT` | `SWAY_3D_LIFT` | yes | yes | yes |
| `SWAY_AFLINK_*` | `SWAY_AFLINK_THR_P`, `SWAY_AFLINK_THR_S`, `SWAY_AFLINK_THR_T0`, `SWAY_AFLINK_THR_T1`, `SWAY_AFLINK_WEIGHTS` | yes | yes | yes |
| `SWAY_AUGLIFT_BLEND` | `SWAY_AUGLIFT_BLEND` | yes | yes | yes |
| `SWAY_AUTO_STAGE_DEPTH` | `SWAY_AUTO_STAGE_DEPTH` | yes | yes | yes |
| `SWAY_BACKWARD_*` | `SWAY_BACKWARD_COI_ENABLED`, `SWAY_BACKWARD_PASS_ENABLED`, `SWAY_BACKWARD_STITCH_MAX_GAP`, `SWAY_BACKWARD_STITCH_MIN_SIMILARITY` | yes | yes | yes |
| `SWAY_BIDIRECTIONAL_IOU_THRESH` | `SWAY_BIDIRECTIONAL_IOU_THRESH` | yes | yes | yes |
| `SWAY_BIDIRECTIONAL_MIN_MATCH_FRAMES` | `SWAY_BIDIRECTIONAL_MIN_MATCH_FRAMES` | yes | yes | yes |
| `SWAY_BIDIRECTIONAL_TRACK_PASS` | `SWAY_BIDIRECTIONAL_TRACK_PASS` | yes | yes | yes |
| `SWAY_BONE_LENGTH_FILTER` | `SWAY_BONE_LENGTH_FILTER` | yes | yes | yes |
| `SWAY_BONE_LENGTH_FILTER_ITERS` | `SWAY_BONE_LENGTH_FILTER_ITERS` | yes | yes | yes |
| `SWAY_BOOST_*` | `SWAY_BOOST_CMC_METHOD`, `SWAY_BOOST_DLO`, `SWAY_BOOST_DLO_COEF`, `SWAY_BOOST_DUO`, `SWAY_BOOST_LAMBDA_IOU`, `SWAY_BOOST_LAMBDA_MHD`, `SWAY_BOOST_LAMBDA_SHP`, `SWAY_BOOST_REID`, `SWAY_BOOST_RICH_S`, `SWAY_BOOST_SB`, `SWAY_BOOST_VT` | yes | yes | yes |
| `SWAY_BOTSORT_*` | `SWAY_BOTSORT_CMC` | yes | yes | yes |
| `SWAY_BOXMOT_*` | `SWAY_BOXMOT_ASSOC_METRIC`, `SWAY_BOXMOT_MATCH_THRESH`, `SWAY_BOXMOT_MAX_AGE`, `SWAY_BOXMOT_REID_ON`, `SWAY_BOXMOT_REID_WEIGHTS`, `SWAY_BOXMOT_TRACKER` | yes | yes | yes |
| `SWAY_BOX_INTERP_MODE` | `SWAY_BOX_INTERP_MODE` | yes | yes | yes |
| `SWAY_BYTETRACK_*` | `SWAY_BYTETRACK_MATCH_THRESH`, `SWAY_BYTETRACK_TRACK_BUFFER` | yes | yes | yes |
| `SWAY_CHUNK_SIZE` | `SWAY_CHUNK_SIZE` | yes | yes | yes |
| `SWAY_COALESCENCE_*` | `SWAY_COALESCENCE_CONSECUTIVE_FRAMES`, `SWAY_COALESCENCE_IOU_THRESH` | yes | yes | yes |
| `SWAY_COI_*` | `SWAY_COI_LOGIT_VARIANCE_WINDOW`, `SWAY_COI_MASK_IOU_THRESH`, `SWAY_COI_QUARANTINE_MODE` | yes | yes | yes |
| `SWAY_COLLISION_*` | `SWAY_COLLISION_DP_MAX_PERMUTATIONS`, `SWAY_COLLISION_MIN_TRACKS`, `SWAY_COLLISION_SOLVER` | yes | yes | yes |
| `SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH` | `SWAY_CONFIDENCE_HEATMAP_THRESH_HIGH` | yes | yes | yes |
| `SWAY_CONFIDENCE_HEATMAP_THRESH_MED` | `SWAY_CONFIDENCE_HEATMAP_THRESH_MED` | yes | yes | yes |
| `SWAY_CONFIDENCE_MASK_GATE` | `SWAY_CONFIDENCE_MASK_GATE` | yes | yes | yes |
| `SWAY_CONFIDENCE_TEMPORAL_WINDOW` | `SWAY_CONFIDENCE_TEMPORAL_WINDOW` | yes | yes | yes |
| `SWAY_CONF_TEMPERATURE` | `SWAY_CONF_TEMPERATURE` | yes | yes | yes |
| `SWAY_CRITIQUE_*` | `SWAY_CRITIQUE_BEAT_TOLERANCE_MS`, `SWAY_CRITIQUE_DIMENSIONS`, `SWAY_CRITIQUE_JERK_WINDOW`, `SWAY_CRITIQUE_MIN_CONFIDENCE`, `SWAY_CRITIQUE_REPORT_GAPS` | yes | yes | yes |
| `SWAY_DEFAULT_ROOT_Z` | `SWAY_DEFAULT_ROOT_Z` | yes | yes | yes |
| `SWAY_DEPTH_*` | `SWAY_DEPTH_DYNAMIC`, `SWAY_DEPTH_FOR_ROOT_Z`, `SWAY_DEPTH_ROOT_ROI_RADIUS`, `SWAY_DEPTH_STRIDE_FRAMES`, `SWAY_DEPTH_Z_FAR`, `SWAY_DEPTH_Z_NEAR` | yes | yes | yes |
| `SWAY_DETECTION_UNCERTAIN_CONF` | `SWAY_DETECTION_UNCERTAIN_CONF` | yes | yes | yes |
| `SWAY_DETECTOR_*` | `SWAY_DETECTOR_HYBRID`, `SWAY_DETECTOR_PRECISION`, `SWAY_DETECTOR_PRIMARY` | yes | yes | yes |
| `SWAY_DETECT_SIZE` | `SWAY_DETECT_SIZE` | yes | yes | yes |
| `SWAY_DOC_*` | `SWAY_DOC_ALPHA_EMB`, `SWAY_DOC_AW_OFF`, `SWAY_DOC_AW_PARAM`, `SWAY_DOC_CMC_OFF`, `SWAY_DOC_DELTA_T`, `SWAY_DOC_INERTIA`, `SWAY_DOC_NSA_KF_ON`, `SWAY_DOC_Q_S`, `SWAY_DOC_Q_XY`, `SWAY_DOC_W_EMB` | yes | yes | yes |
| `SWAY_DORMANT_MAX_GAP` | `SWAY_DORMANT_MAX_GAP` | yes | yes | yes |
| `SWAY_ENROLLMENT_*` | `SWAY_ENROLLMENT_AUTO_FRAME`, `SWAY_ENROLLMENT_COLOR_BINS`, `SWAY_ENROLLMENT_ENABLED`, `SWAY_ENROLLMENT_GALLERY_SIGNALS`, `SWAY_ENROLLMENT_MIN_SEPARATION_PX`, `SWAY_ENROLLMENT_PART_MODEL` | yes | yes | yes |
| `SWAY_FT_BOX_ENLARGE` | `SWAY_FT_BOX_ENLARGE` | yes | yes | yes |
| `SWAY_FT_DET_HIGH` | `SWAY_FT_DET_HIGH` | yes | yes | yes |
| `SWAY_FT_DET_LOW` | `SWAY_FT_DET_LOW` | yes | yes | yes |
| `SWAY_FT_MOTION_DAMP` | `SWAY_FT_MOTION_DAMP` | yes | yes | yes |
| `SWAY_FT_PROXIMITY_RAD` | `SWAY_FT_PROXIMITY_RAD` | yes | yes | yes |
| `SWAY_FX` | `SWAY_FX` | yes | yes | yes |
| `SWAY_FY` | `SWAY_FY` | yes | yes | yes |
| `SWAY_GLOBAL_*` | `SWAY_GLOBAL_AFLINK`, `SWAY_GLOBAL_LINK` | yes | yes | yes |
| `SWAY_GNN_*` | `SWAY_GNN_DEVICE`, `SWAY_GNN_DROPOUT`, `SWAY_GNN_HEADS`, `SWAY_GNN_HIDDEN`, `SWAY_GNN_LAYERS`, `SWAY_GNN_MAX_GAP`, `SWAY_GNN_MERGE_THRESH`, `SWAY_GNN_PRIOR_SCALE`, `SWAY_GNN_SEED`, `SWAY_GNN_TRACK_REFINE`, `SWAY_GNN_TRACK_SMOKE`, `SWAY_GNN_WEIGHTS` | yes | yes | yes |
| `SWAY_GROUP_VIDEO` | `SWAY_GROUP_VIDEO` | yes | yes | yes |
| `SWAY_GSI_LENGTHSCALE` | `SWAY_GSI_LENGTHSCALE` | yes | yes | yes |
| `SWAY_HANDSHAKE_VERIFY_STRIDE` | `SWAY_HANDSHAKE_VERIFY_STRIDE` | yes | yes | yes |
| `SWAY_HMR_*` | `SWAY_HMR_MESH_SIDECAR`, `SWAY_HMR_SMOKE` | yes | yes | yes |
| `SWAY_HYBRID_*` | `SWAY_HYBRID_OVERLAP_IOU_TRIGGER`, `SWAY_HYBRID_SAM_BBOX_PAD`, `SWAY_HYBRID_SAM_IOU_TRIGGER`, `SWAY_HYBRID_SAM_MASK_THRESH`, `SWAY_HYBRID_SAM_MIN_DETS`, `SWAY_HYBRID_SAM_OVERLAP`, `SWAY_HYBRID_SAM_ROI_CROP`, `SWAY_HYBRID_SAM_ROI_PAD_FRAC`, `SWAY_HYBRID_SAM_WEAK_CUES`, `SWAY_HYBRID_SAM_WEIGHTS`, `SWAY_HYBRID_WEAK_CONF_DELTA`, `SWAY_HYBRID_WEAK_HEIGHT_FRAC`, `SWAY_HYBRID_WEAK_MATCH_IOU` | yes | yes | yes |
| `SWAY_LIFT_*` | `SWAY_LIFT_BACKEND`, `SWAY_LIFT_DEPTH_SCENE`, `SWAY_LIFT_GAP_MODE`, `SWAY_LIFT_INPUT_NORM`, `SWAY_LIFT_MULTI_PERSON`, `SWAY_LIFT_SAVGOL`, `SWAY_LIFT_SAVGOL_POLY`, `SWAY_LIFT_SAVGOL_WINDOW`, `SWAY_LIFT_WORLD_SCALE` | yes | yes | yes |
| `SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA` | `SWAY_MEMOSORT_ADAPTIVE_IOU_ALPHA` | yes | yes | yes |
| `SWAY_MEMOSORT_MEMORY_LENGTH` | `SWAY_MEMOSORT_MEMORY_LENGTH` | yes | yes | yes |
| `SWAY_MOTE_*` | `SWAY_MOTE_CONFIDENCE_BOOST`, `SWAY_MOTE_DISOCCLUSION`, `SWAY_MOTE_FLOW_MODEL` | yes | yes | yes |
| `SWAY_OCSORT_USE_BYTE` | `SWAY_OCSORT_USE_BYTE` | yes | yes | yes |
| `SWAY_OFFLINE` | `SWAY_OFFLINE` | yes | yes | yes |
| `SWAY_PHASE13_MODE` | `SWAY_PHASE13_MODE` | yes | yes | yes |
| `SWAY_PHASE_*` | `SWAY_PHASE_DEBUG_FILES` | yes | yes | yes |
| `SWAY_PINHOLE_*` | `SWAY_PINHOLE_FOV_DEG` | yes | yes | yes |
| `SWAY_POSE_*` | `SWAY_POSE_3D_INCLUDE_LIFT`, `SWAY_POSE_BUSY_HEARTBEAT_SEC`, `SWAY_POSE_GAP_INTERP_MODE`, `SWAY_POSE_GSI_LENGTHSCALE`, `SWAY_POSE_KEYPOINT_SET`, `SWAY_POSE_LOG_EVERY_N_PASSES`, `SWAY_POSE_LOG_EVERY_SEC`, `SWAY_POSE_MASK_GUIDED`, `SWAY_POSE_MODEL`, `SWAY_POSE_SLOW_FORWARD_SEC`, `SWAY_POSE_SMART_PAD`, `SWAY_POSE_VISIBILITY_THRESHOLD` | yes | yes | yes |
| `SWAY_PRETRACK_NMS_IOU` | `SWAY_PRETRACK_NMS_IOU` | yes | yes | yes |
| `SWAY_REGISTRY_*` | `SWAY_REGISTRY_DORMANT_MATCH`, `SWAY_REGISTRY_ISOLATION_MULT`, `SWAY_REGISTRY_SWAP_MARGIN`, `SWAY_REGISTRY_TOUCH_IOU` | yes | yes | yes |
| `SWAY_REID_*` | `SWAY_REID_COLOR_SPACE`, `SWAY_REID_DEBUG`, `SWAY_REID_EMA_ALPHA_HIGH`, `SWAY_REID_EMA_ALPHA_LOW`, `SWAY_REID_EMA_ISOLATION_DIST`, `SWAY_REID_EMA_POSE_QUALITY_THRESH`, `SWAY_REID_FACE_MIN_SIZE`, `SWAY_REID_FACE_MODEL`, `SWAY_REID_FINETUNE_BASE_MODEL`, `SWAY_REID_FINETUNE_ENABLED`, `SWAY_REID_FINETUNE_EPOCHS`, `SWAY_REID_FINETUNE_LR`, `SWAY_REID_FINETUNE_PAIRS`, `SWAY_REID_KPR_ENABLED`, `SWAY_REID_PART_MIN_VISIBLE`, `SWAY_REID_PART_MODEL`, `SWAY_REID_SKEL_MIN_WINDOW`, `SWAY_REID_SKEL_MODEL`, `SWAY_REID_SPATIAL_DECAY`, `SWAY_REID_W_COLOR`, `SWAY_REID_W_FACE`, `SWAY_REID_W_KPR`, `SWAY_REID_W_PART`, `SWAY_REID_W_SKELETON`, `SWAY_REID_W_SPATIAL` | yes | yes | yes |
| `SWAY_ROOT_*` | `SWAY_ROOT_Z_EMA_ALPHA` | yes | yes | yes |
| `SWAY_SAM2_*` | `SWAY_SAM2_CONFIDENCE_REINVOKE`, `SWAY_SAM2_MASK_POSE`, `SWAY_SAM2_MEMORY_FRAMES`, `SWAY_SAM2_MODEL`, `SWAY_SAM2_REINVOKE_STRIDE` | yes | yes | yes |
| `SWAY_SAPIENS_*` | `SWAY_SAPIENS_HEATMAP_H`, `SWAY_SAPIENS_SMOKE`, `SWAY_SAPIENS_TORCHSCRIPT` | yes | yes | yes |
| `SWAY_SENTINEL_*` | `SWAY_SENTINEL_GRACE_MULTIPLIER`, `SWAY_SENTINEL_SBM`, `SWAY_SENTINEL_WEAK_DET_CONF` | yes | yes | yes |
| `SWAY_SHORT_GAP_FRAMES` | `SWAY_SHORT_GAP_FRAMES` | yes | yes | yes |
| `SWAY_STAGE_POLYGON` | `SWAY_STAGE_POLYGON` | yes | yes | yes |
| `SWAY_STATE_*` | `SWAY_STATE_DORMANT_MASK_FRAC`, `SWAY_STATE_DORMANT_MAX_FRAMES`, `SWAY_STATE_PARTIAL_MASK_FRAC`, `SWAY_STATE_PARTIAL_MIN_JOINTS` | yes | yes | yes |
| `SWAY_STITCH_*` | `SWAY_STITCH_MAX_FRAME_GAP`, `SWAY_STITCH_MAX_PIXEL_RADIUS`, `SWAY_STITCH_PREDICTED_RADIUS_FRAC`, `SWAY_STITCH_RADIUS_BBOX_FRAC` | yes | yes | yes |
| `SWAY_STRONGSORT_*` | `SWAY_STRONGSORT_MAX_COS_DIST`, `SWAY_STRONGSORT_MAX_IOU_DIST`, `SWAY_STRONGSORT_NN_BUDGET`, `SWAY_STRONGSORT_N_INIT` | yes | yes | yes |
| `SWAY_ST_*` | `SWAY_ST_EMA_ALPHA`, `SWAY_ST_THETA_EMB`, `SWAY_ST_THETA_IOU` | yes | yes | yes |
| `SWAY_TEMPORAL_*` | `SWAY_TEMPORAL_POSE_RADIUS`, `SWAY_TEMPORAL_POSE_REFINE` | yes | yes | yes |
| `SWAY_TRACKER_*` | `SWAY_TRACKER_ENGINE`, `SWAY_TRACKER_YAML` | yes | yes | yes |
| `SWAY_TRACK_*` | `SWAY_TRACK_DORMANT_MASK_FRAC`, `SWAY_TRACK_MAX_AGE`, `SWAY_TRACK_PARTIAL_MASK_FRAC` | yes | yes | yes |
| `SWAY_UMOT_*` | `SWAY_UMOT_BACKTRACK`, `SWAY_UMOT_HISTORY_LENGTH` | yes | yes | yes |
| `SWAY_UNIFIED_3D_EXPORT` | `SWAY_UNIFIED_3D_EXPORT` | yes | yes | yes |
| `SWAY_UNLOCK_DETECTION_TUNING` | `SWAY_UNLOCK_DETECTION_TUNING` | yes | yes | yes |
| `SWAY_UNLOCK_HYBRID_SAM_TUNING` | `SWAY_UNLOCK_HYBRID_SAM_TUNING` | yes | yes | yes |
| `SWAY_UNLOCK_PHASE3_STITCH_TUNING` | `SWAY_UNLOCK_PHASE3_STITCH_TUNING` | yes | yes | yes |
| `SWAY_UNLOCK_POSE_TUNING` | `SWAY_UNLOCK_POSE_TUNING` | yes | yes | yes |
| `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING` | `SWAY_UNLOCK_POST_POSE_PRUNE_TUNING` | yes | yes | yes |
| `SWAY_UNLOCK_PRE_POSE_PRUNE_TUNING` | `SWAY_UNLOCK_PRE_POSE_PRUNE_TUNING` | yes | yes | yes |
| `SWAY_UNLOCK_REID_DEDUP_TUNING` | `SWAY_UNLOCK_REID_DEDUP_TUNING` | yes | yes | yes |
| `SWAY_UNLOCK_SMOOTH_TUNING` | `SWAY_UNLOCK_SMOOTH_TUNING` | yes | yes | yes |
| `SWAY_USE_BOXMOT` | `SWAY_USE_BOXMOT` | yes | yes | yes |
| `SWAY_VIS_*` | `SWAY_VIS_GSI_LENGTHSCALE`, `SWAY_VIS_TEMPORAL_INTERP_MODE` | yes | yes | yes |
| `SWAY_VITPOSE_*` | `SWAY_VITPOSE_DEBUG`, `SWAY_VITPOSE_FP32`, `SWAY_VITPOSE_MAX_PER_FORWARD`, `SWAY_VITPOSE_MODEL`, `SWAY_VITPOSE_MPS_CHUNK`, `SWAY_VITPOSE_SMART_PAD`, `SWAY_VITPOSE_USE_FAST` | yes | yes | yes |
| `SWAY_YOLO_*` | `SWAY_YOLO_CONF`, `SWAY_YOLO_DETECTION_STRIDE`, `SWAY_YOLO_ENGINE`, `SWAY_YOLO_HALF`, `SWAY_YOLO_INFER_BATCH`, `SWAY_YOLO_WEIGHTS` | yes | yes | yes |

## Notes On Strictness

- `runtime wired=yes` in this catalog means "connected to active runtime path(s)," not "always active by default."
- `sweep-ready now=yes` means: the current sweep path can exercise the feature and its sub-config knobs directly.
- `sweep-ready now=partial` means: the feature exists, but sweeps currently hit only part of its config surface (or the wiring path is partial).
- `sweep-ready now=no` means: sweep cannot truly exercise it yet.
- `SWAY_TRACKER_ENGINE=matr` is currently wired through a compatibility wrapper (MeMoSORT route) so sweeps can execute the branch while native MATR implementation is finalized.
- Sweep machinery basis for this table:
  - `python -m tools.run_all_configurations --strict-coverage` confirms full SWAY key coverage.
  - `tools/run_all_configurations.py` now auto-generates low/mid/high probes for numeric defaults when no explicit Optuna domain exists, so sub-config numeric values are tested beyond single defaults.
- For fail-fast verification during runs, use `SWAY_FAIL_ON_UNWIRED_EXTRAS=1` and read `[feature] requested/runtime/wiring` logs.
