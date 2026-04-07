import type { WatchPhaseId } from './watchPrune'

/**
 * Field ids (pipeline_config_schema) most useful when reviewing each phase clip.
 * Values are read from run manifest `run_context_final.fields`.
 */
export const WATCH_PHASE_TUNING_FIELD_IDS: Record<WatchPhaseId, string[]> = {
  phase1_dets: [
    'sway_yolo_weights',
    'sway_yolo_conf',
    'sway_pretrack_nms_iou',
    'sway_yolo_detection_stride',
    'sway_hybrid_sam_iou_trigger',
    'sway_hybrid_sam_weak_cues',
  ],
  track: [
    'sway_yolo_weights',
    'sway_yolo_conf',
    'sway_pretrack_nms_iou',
    'tracker_technology',
    'sway_hybrid_sam_iou_trigger',
    'sway_hybrid_sam_weak_cues',
    'sway_boxmot_max_age',
    'sway_boxmot_match_thresh',
    'sway_boxmot_reid_model',
    'sway_bidirectional_track_pass',
    'sway_gnn_track_refine',
    'sway_stitch_max_frame_gap',
    'sway_global_aflink_mode',
  ],
  // Phase 4 pre-pose prune: master-locked YAML; no Lab schema fields — leave empty.
  pre_pose: [],
  pose: [
    'pose_model',
    'pose_visibility_threshold',
    'sway_hmr_mesh_sidecar',
  ],
  collision: ['dedup_min_pair_oks', 'dedup_antipartner_min_iou'],
  post_pose: [
    'prune_threshold',
    'sync_score_min',
    'pruning_w_low_sync',
    'pruning_w_smart_mirror',
    'pruning_w_low_conf',
  ],
}
