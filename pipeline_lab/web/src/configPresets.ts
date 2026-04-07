import type { Schema, SchemaField } from './types'

// ---------------------------------------------------------------------------
// Phase-group preset system
//
// Four independent groups cover the full pipeline. Pick one preset per group;
// they override non-overlapping field sets so any combination is valid.
// ---------------------------------------------------------------------------

export type PhaseGroupId = 'phases_1_3' | 'phases_4_6' | 'phases_7_9' | 'phases_10_11'

export type Phase13PresetId =
  | 'p13_standard'
  | 'p13_fast_scan'
  | 'p13_dense_crowd'
  | 'p13_open_floor'
  | 'p13_open_floor_recovery'
  | 'p13_osnet_lock'
  | 'p13_dancer_registry'
  | 'p13_sway_handshake'
  | 'p13_wide_angle'

export type Phase46PresetId =
  | 'p46_balanced'
  | 'p46_high_fidelity'
  | 'p46_max_precision'
  | 'p46_competition'
  | 'p46_fast_skeleton'
  | 'p46_rtmpose'

export type Phase79PresetId =
  | 'p79_balanced'
  | 'p79_aggressive'
  | 'p79_conservative'
  | 'p79_fluid_ballet'
  | 'p79_sharp_hiphop'
  | 'p79_mirror_studio'

export type Phase1011PresetId = 'p1011_standard' | 'p1011_full_analysis' | 'p1011_minimal'

export interface PresetDef {
  id: string
  group: PhaseGroupId
  name: string
  phasesLabel: string
  description: string
  fields: Record<string, unknown>
}

export interface PhaseGroupMeta {
  id: PhaseGroupId
  label: string
  phasesLabel: string
  description: string
}

export const PHASE_GROUPS: PhaseGroupMeta[] = [
  {
    id: 'phases_1_3',
    label: 'Detection / Tracking / Stitching',
    phasesLabel: 'Phases 1–3',
    description: 'How people are found, followed frame-to-frame, and reconnected after gaps.',
  },
  {
    id: 'phases_4_6',
    label: 'Pose / Association',
    phasesLabel: 'Phases 4–6',
    description: 'Skeleton quality, model size, and identity cleanup after body pose.',
  },
  {
    id: 'phases_7_9',
    label: 'Cleanup / Pruning / Smoothing',
    phasesLabel: 'Phases 7–9',
    description: 'Remove fake tracks, tune pruning strictness, and set motion feel.',
  },
  {
    id: 'phases_10_11',
    label: 'Scoring / Export',
    phasesLabel: 'Phases 10–11',
    description: 'What gets written: scores, overlays, montage, debug artifacts.',
  },
]

// ---------------------------------------------------------------------------
// Group A: Phases 1-3  Detection / Tracking / Stitching
// ---------------------------------------------------------------------------

export const PHASE_13_PRESETS: PresetDef[] = [
  {
    id: 'p13_standard',
    group: 'phases_1_3',
    name: 'Standard',
    phasesLabel: 'Phases 1–3',
    description:
      'General-purpose baseline. DanceTrack YOLO-L, Deep OC-SORT motion tracking, hybrid SAM overlap refinement at IoU 0.42, neural stitch when weights exist.',
    fields: {
      sway_phase13_mode: 'standard',
      sway_yolo_weights: 'yolo26l_dancetrack',
      tracker_technology: 'deep_ocsort',
      sway_pretrack_nms_iou: 0.75,
      sway_yolo_conf: 0.22,
      sway_hybrid_sam_iou_trigger: 0.42,
      sway_hybrid_sam_weak_cues: false,
      sway_boxmot_max_age: 150,
      sway_boxmot_match_thresh: 0.30,
      sway_global_aflink_mode: 'neural_if_available',
      sway_bidirectional_track_pass: false,
      sway_yolo_detection_stride: 1,
    },
  },
  {
    id: 'p13_fast_scan',
    group: 'phases_1_3',
    name: 'Fast Scan',
    phasesLabel: 'Phases 1–3',
    description:
      'Quick preview: small YOLO, ByteTrack (no Re-ID), detection every 4th frame, heuristic stitch, no SAM overlap. Rough output for fast iteration.',
    fields: {
      sway_phase13_mode: 'standard',
      sway_yolo_weights: 'yolo26s',
      tracker_technology: 'bytetrack',
      sway_yolo_detection_stride: 4,
      sway_global_aflink_mode: 'force_heuristic',
      sway_boxmot_max_age: 90,
      sway_pretrack_nms_iou: 0.75,
      sway_yolo_conf: 0.22,
      sway_hybrid_sam_iou_trigger: 0.42,
      sway_hybrid_sam_weak_cues: false,
      sway_boxmot_match_thresh: 0.30,
      sway_bidirectional_track_pass: false,
    },
  },
  {
    id: 'p13_dense_crowd',
    group: 'phases_1_3',
    name: 'Dense Crowd',
    phasesLabel: 'Phases 1–3',
    description:
      'Optimized for 8+ dancers in tight formations. Extra-large YOLO catches small/distant bodies, lower NMS merges overlapping boxes aggressively, SAM triggers earlier, tracker remembers longer.',
    fields: {
      sway_phase13_mode: 'standard',
      sway_yolo_weights: 'yolo26x',
      tracker_technology: 'deep_ocsort',
      sway_pretrack_nms_iou: 0.60,
      sway_yolo_conf: 0.18,
      sway_hybrid_sam_iou_trigger: 0.30,
      sway_hybrid_sam_weak_cues: false,
      sway_boxmot_max_age: 200,
      sway_boxmot_match_thresh: 0.25,
      sway_global_aflink_mode: 'neural_if_available',
      sway_bidirectional_track_pass: false,
      sway_yolo_detection_stride: 1,
    },
  },
  {
    id: 'p13_open_floor',
    group: 'phases_1_3',
    name: 'Open Floor',
    phasesLabel: 'Phases 1–3',
    description:
      'For 2-4 dancers with space between them. Higher detection confidence cuts noise, SAM only fires on real overlap, shorter tracker memory prevents stale ID inheritance.',
    fields: {
      sway_phase13_mode: 'standard',
      sway_yolo_weights: 'yolo26l_dancetrack',
      tracker_technology: 'deep_ocsort',
      sway_pretrack_nms_iou: 0.80,
      sway_yolo_conf: 0.30,
      sway_hybrid_sam_iou_trigger: 0.50,
      sway_hybrid_sam_weak_cues: false,
      sway_boxmot_max_age: 120,
      sway_boxmot_match_thresh: 0.35,
      sway_global_aflink_mode: 'neural_if_available',
      sway_bidirectional_track_pass: false,
      sway_yolo_detection_stride: 1,
    },
  },
  {
    id: 'p13_open_floor_recovery',
    group: 'phases_1_3',
    name: 'Open Floor (recovery bias)',
    phasesLabel: 'Phases 1–3',
    description:
      'Recommended vs classic Open Floor when dancers duck behind others and reappear: DanceTrack+CrowdHuman weights, easier box matching (0.29), hybrid SAM at IoU 0.42, longer track memory (165). Slower Phase 1–2; see docs/PIPELINE_FINDINGS_AND_BEST_CONFIGS.md.',
    fields: {
      sway_phase13_mode: 'standard',
      sway_yolo_weights: 'yolo26l_dancetrack_crowdhuman',
      tracker_technology: 'deep_ocsort',
      sway_pretrack_nms_iou: 0.80,
      sway_yolo_conf: 0.30,
      sway_hybrid_sam_iou_trigger: 0.42,
      sway_hybrid_sam_weak_cues: false,
      sway_boxmot_max_age: 165,
      sway_boxmot_match_thresh: 0.29,
      sway_global_aflink_mode: 'neural_if_available',
      sway_bidirectional_track_pass: false,
      sway_yolo_detection_stride: 1,
    },
  },
  {
    id: 'p13_osnet_lock',
    group: 'phases_1_3',
    name: 'OSNet Identity Lock',
    phasesLabel: 'Phases 1–3',
    description:
      'Maximum identity consistency. Adds OSNet appearance embeddings during tracking + bidirectional (forward+reverse) pass to catch ID breaks. ~2x tracking time. Needs models/osnet_x0_25_msmt17.pt.',
    fields: {
      sway_phase13_mode: 'standard',
      sway_yolo_weights: 'yolo26l_dancetrack',
      tracker_technology: 'deep_ocsort_osnet',
      sway_boxmot_reid_model: 'osnet_x0_25',
      sway_bidirectional_track_pass: true,
      sway_boxmot_max_age: 180,
      sway_boxmot_match_thresh: 0.30,
      sway_pretrack_nms_iou: 0.75,
      sway_yolo_conf: 0.22,
      sway_hybrid_sam_iou_trigger: 0.42,
      sway_hybrid_sam_weak_cues: false,
      sway_global_aflink_mode: 'neural_if_available',
      sway_yolo_detection_stride: 1,
    },
  },
  {
    id: 'p13_dancer_registry',
    group: 'phases_1_3',
    name: 'Dancer Registry',
    phasesLabel: 'Phases 1–3',
    description:
      'Experimental: adds zonal HSV crossover verify + appearance-based dormant relink (extra full video scans). Fixes ID swaps after dancers cross paths. Pins DanceTrack YOLO-L + ViTPose-Base.',
    fields: {
      sway_phase13_mode: 'dancer_registry',
      sway_yolo_weights: 'yolo26l_dancetrack',
      sway_pretrack_nms_iou: 0.75,
      tracker_technology: 'deep_ocsort',
      sway_global_aflink_mode: 'neural_if_available',
      sway_hybrid_sam_iou_trigger: 0.42,
      sway_hybrid_sam_weak_cues: false,
      sway_boxmot_max_age: 150,
      sway_boxmot_match_thresh: 0.30,
      sway_bidirectional_track_pass: false,
      sway_yolo_detection_stride: 1,
      sway_yolo_conf: 0.22,
    },
  },
  {
    id: 'p13_sway_handshake',
    group: 'phases_1_3',
    name: 'Sway Handshake',
    phasesLabel: 'Phases 1–3',
    description:
      'Experimental: zonal color registry + SAM mask verification at low IoU trigger (0.10). Hungarian reorder aligns SAM rows with appearance before BoxMOT. Best for crowded choreography with frequent overlap.',
    fields: {
      sway_phase13_mode: 'sway_handshake',
      sway_yolo_weights: 'yolo26l_dancetrack',
      sway_hybrid_sam_iou_trigger: 0.1,
      sway_hybrid_sam_weak_cues: false,
      tracker_technology: 'deep_ocsort',
      sway_global_aflink_mode: 'neural_if_available',
      sway_pretrack_nms_iou: 0.75,
      sway_yolo_conf: 0.22,
      sway_boxmot_max_age: 150,
      sway_boxmot_match_thresh: 0.30,
      sway_bidirectional_track_pass: false,
      sway_yolo_detection_stride: 1,
    },
  },
  {
    id: 'p13_wide_angle',
    group: 'phases_1_3',
    name: 'Wide Angle Stage',
    phasesLabel: 'Phases 1–3',
    description:
      'Tuned for wide-angle shots where dancers appear small. Extra-large YOLO with low confidence catches distant bodies. Longer tracker memory handles slow entries/exits.',
    fields: {
      sway_phase13_mode: 'standard',
      sway_yolo_weights: 'yolo26x',
      tracker_technology: 'deep_ocsort',
      sway_yolo_conf: 0.15,
      sway_pretrack_nms_iou: 0.70,
      sway_boxmot_max_age: 200,
      sway_boxmot_match_thresh: 0.28,
      sway_hybrid_sam_iou_trigger: 0.35,
      sway_hybrid_sam_weak_cues: false,
      sway_global_aflink_mode: 'neural_if_available',
      sway_bidirectional_track_pass: false,
      sway_yolo_detection_stride: 1,
    },
  },
]

// ---------------------------------------------------------------------------
// Group B: Phases 4-6  Pose / Association
// ---------------------------------------------------------------------------

export const PHASE_46_PRESETS: PresetDef[] = [
  {
    id: 'p46_balanced',
    group: 'phases_4_6',
    name: 'Balanced Pose',
    phasesLabel: 'Phases 4–6',
    description:
      'Default: ViTPose-Base for good speed/quality tradeoff. 3D lift on for depth-aware scoring. Standard visibility and dedup thresholds.',
    fields: {
      pose_model: 'ViTPose-Base',
      sway_pose_3d_lift: true,
      temporal_pose_refine: false,
      sway_vitpose_use_fast: false,
      pose_visibility_threshold: 0.30,
      dedup_min_pair_oks: 0.68,
      dedup_antipartner_min_iou: 0.12,
    },
  },
  {
    id: 'p46_high_fidelity',
    group: 'phases_4_6',
    name: 'High Fidelity',
    phasesLabel: 'Phases 4–6',
    description:
      'ViTPose-Large for better joint accuracy on hard motion. Lower visibility threshold keeps more joints. Slower but higher quality skeletons.',
    fields: {
      pose_model: 'ViTPose-Large',
      sway_pose_3d_lift: true,
      temporal_pose_refine: false,
      sway_vitpose_use_fast: false,
      pose_visibility_threshold: 0.25,
      dedup_min_pair_oks: 0.70,
      dedup_antipartner_min_iou: 0.12,
    },
  },
  {
    id: 'p46_max_precision',
    group: 'phases_4_6',
    name: 'Maximum Precision',
    phasesLabel: 'Phases 4–6',
    description:
      'ViTPose-Huge for maximum joint accuracy. Temporal refine smooths between neighbors. Tightest dedup OKS eliminates more phantom duplicates. Slowest but best for finals.',
    fields: {
      pose_model: 'ViTPose-Huge',
      sway_pose_3d_lift: true,
      temporal_pose_refine: true,
      sway_vitpose_use_fast: false,
      pose_visibility_threshold: 0.20,
      dedup_min_pair_oks: 0.72,
      dedup_antipartner_min_iou: 0.14,
    },
  },
  {
    id: 'p46_competition',
    group: 'phases_4_6',
    name: 'Competition Grade',
    phasesLabel: 'Phases 4–6',
    description:
      'Designed for scoring competitions. ViTPose-Large with temporal refine and 3D lift. Strictest dedup thresholds prevent any phantom duplicates from affecting scores.',
    fields: {
      pose_model: 'ViTPose-Large',
      sway_pose_3d_lift: true,
      temporal_pose_refine: true,
      sway_vitpose_use_fast: false,
      pose_visibility_threshold: 0.22,
      dedup_min_pair_oks: 0.75,
      dedup_antipartner_min_iou: 0.15,
    },
  },
  {
    id: 'p46_fast_skeleton',
    group: 'phases_4_6',
    name: 'Fast Skeleton',
    phasesLabel: 'Phases 4–6',
    description:
      'Speed priority: ViTPose-Base with fast HF processor, no 3D lift, higher visibility threshold drops noisy joints. For quick previews only.',
    fields: {
      pose_model: 'ViTPose-Base',
      sway_pose_3d_lift: false,
      temporal_pose_refine: false,
      sway_vitpose_use_fast: true,
      pose_visibility_threshold: 0.35,
      save_phase_previews: false,
      dedup_min_pair_oks: 0.68,
      dedup_antipartner_min_iou: 0.12,
    },
  },
  {
    id: 'p46_rtmpose',
    group: 'phases_4_6',
    name: 'RTMPose (experimental)',
    phasesLabel: 'Phases 4–6',
    description:
      'Alternative pose stack using MMPose RTMPose-L. Requires mmengine + mmcv + mmpose installed. Compare speed vs ViTPose-Base on same clip.',
    fields: {
      pose_model: 'RTMPose-L',
      sway_pose_3d_lift: true,
      temporal_pose_refine: false,
      sway_vitpose_use_fast: false,
      pose_visibility_threshold: 0.30,
      dedup_min_pair_oks: 0.68,
      dedup_antipartner_min_iou: 0.12,
    },
  },
]

// ---------------------------------------------------------------------------
// Group C: Phases 7-9  Cleanup / Pruning / Smoothing
// ---------------------------------------------------------------------------

export const PHASE_79_PRESETS: PresetDef[] = [
  {
    id: 'p79_balanced',
    group: 'phases_7_9',
    name: 'Balanced Cleanup',
    phasesLabel: 'Phases 7–9',
    description:
      'Default pruning and smoothing. Moderate strictness keeps real dancers while removing obvious noise. Standard 1-Euro beta for general choreography.',
    fields: {
      prune_threshold: 0.65,
      sync_score_min: 0.10,
      pruning_w_low_sync: 0.7,
      pruning_w_smart_mirror: 0.9,
      pruning_w_low_conf: 0.5,
      smoother_beta: 0.70,
    },
  },
  {
    id: 'p79_aggressive',
    group: 'phases_7_9',
    name: 'Aggressive Clean',
    phasesLabel: 'Phases 7–9',
    description:
      'Strict cleanup: lower prune threshold drops marginal tracks faster, higher sync requirement filters out-of-sync bodies, stronger confidence penalty on weak skeletons.',
    fields: {
      prune_threshold: 0.55,
      sync_score_min: 0.15,
      pruning_w_low_sync: 0.85,
      pruning_w_smart_mirror: 0.95,
      pruning_w_low_conf: 0.7,
      smoother_beta: 0.70,
    },
  },
  {
    id: 'p79_conservative',
    group: 'phases_7_9',
    name: 'Conservative Keep',
    phasesLabel: 'Phases 7–9',
    description:
      'Lenient pruning: higher threshold keeps more tracks even when some checks complain. Low sync requirement avoids cutting soloists or freestyle sections.',
    fields: {
      prune_threshold: 0.78,
      sync_score_min: 0.05,
      pruning_w_low_sync: 0.4,
      pruning_w_smart_mirror: 0.5,
      pruning_w_low_conf: 0.3,
      smoother_beta: 0.70,
    },
  },
  {
    id: 'p79_fluid_ballet',
    group: 'phases_7_9',
    name: 'Fluid (Ballet / Contemporary)',
    phasesLabel: 'Phases 7–9',
    description:
      'Optimized for ballet and contemporary dance. Higher smoother beta (0.85) follows speed changes smoothly. Lower prune weights keep graceful motions that might look "out of sync" in group metrics.',
    fields: {
      prune_threshold: 0.65,
      sync_score_min: 0.08,
      pruning_w_low_sync: 0.5,
      pruning_w_smart_mirror: 0.7,
      pruning_w_low_conf: 0.4,
      smoother_beta: 0.85,
    },
  },
  {
    id: 'p79_sharp_hiphop',
    group: 'phases_7_9',
    name: 'Sharp (Hip-Hop / Popping)',
    phasesLabel: 'Phases 7–9',
    description:
      'Optimized for hip-hop and popping. Lower smoother beta (0.55) reacts faster to speed changes — critical for isolations where millisecond timing matters. Stricter pruning removes noise.',
    fields: {
      prune_threshold: 0.60,
      sync_score_min: 0.12,
      pruning_w_low_sync: 0.8,
      pruning_w_smart_mirror: 0.85,
      pruning_w_low_conf: 0.6,
      smoother_beta: 0.55,
    },
  },
  {
    id: 'p79_mirror_studio',
    group: 'phases_7_9',
    name: 'Mirror Studio',
    phasesLabel: 'Phases 7–9',
    description:
      'For studios with wall-to-wall mirrors. Maximum mirror penalty (1.0) aggressively removes reflections. Lower prune threshold for stricter overall cleanup.',
    fields: {
      prune_threshold: 0.60,
      sync_score_min: 0.10,
      pruning_w_low_sync: 0.75,
      pruning_w_smart_mirror: 1.0,
      pruning_w_low_conf: 0.5,
      smoother_beta: 0.70,
    },
  },
]

// ---------------------------------------------------------------------------
// Group D: Phases 10-11  Scoring / Export
// ---------------------------------------------------------------------------

export const PHASE_1011_PRESETS: PresetDef[] = [
  {
    id: 'p1011_standard',
    group: 'phases_10_11',
    name: 'Standard Export',
    phasesLabel: 'Phases 10–11',
    description: 'Phase preview clips saved for review. No montage. Linear overlay interpolation.',
    fields: {
      montage: false,
      save_phase_previews: true,
      sway_vis_temporal_interp_mode: 'linear',
      sway_hmr_mesh_sidecar: false,
    },
  },
  {
    id: 'p1011_full_analysis',
    group: 'phases_10_11',
    name: 'Full Analysis',
    phasesLabel: 'Phases 10–11',
    description:
      'Complete debug output: montage concatenating all phase previews, HMR mesh sidecar for 3D research. Larger disk usage.',
    fields: {
      montage: true,
      save_phase_previews: true,
      sway_vis_temporal_interp_mode: 'linear',
      sway_hmr_mesh_sidecar: true,
    },
  },
  {
    id: 'p1011_minimal',
    group: 'phases_10_11',
    name: 'Minimal Fast',
    phasesLabel: 'Phases 10–11',
    description: 'No phase previews, no montage. Fastest export — just final data.json and pose overlay.',
    fields: {
      montage: false,
      save_phase_previews: false,
      sway_vis_temporal_interp_mode: 'linear',
      sway_hmr_mesh_sidecar: false,
    },
  },
]

// ---------------------------------------------------------------------------
// All presets indexed
// ---------------------------------------------------------------------------

export const ALL_PRESETS: PresetDef[] = [
  ...PHASE_13_PRESETS,
  ...PHASE_46_PRESETS,
  ...PHASE_79_PRESETS,
  ...PHASE_1011_PRESETS,
]

export const PRESETS_BY_GROUP: Record<PhaseGroupId, PresetDef[]> = {
  phases_1_3: PHASE_13_PRESETS,
  phases_4_6: PHASE_46_PRESETS,
  phases_7_9: PHASE_79_PRESETS,
  phases_10_11: PHASE_1011_PRESETS,
}

export const DEFAULT_PRESET_IDS: Record<PhaseGroupId, string> = {
  phases_1_3: 'p13_standard',
  phases_4_6: 'p46_balanced',
  phases_7_9: 'p79_balanced',
  phases_10_11: 'p1011_standard',
}

export function presetById(id: string): PresetDef | undefined {
  return ALL_PRESETS.find((p) => p.id === id)
}

// ---------------------------------------------------------------------------
// Apply helpers
// ---------------------------------------------------------------------------

export function defaultsFromSchema(fields: SchemaField[]): Record<string, unknown> {
  const o: Record<string, unknown> = {}
  for (const f of fields) {
    if (f.default === undefined || f.default === null) continue
    if (f.type === 'string' && f.default === '') continue
    o[f.id] = f.default
  }
  return o
}

export function migrateLegacyPipelineFields(state: Record<string, unknown>): Record<string, unknown> {
  const out = { ...state }
  if (String(out.sway_boxmot_reid_model ?? '') === 'osnet_x1_0') {
    out.sway_boxmot_reid_model = 'osnet_x0_25'
  }
  return out
}

const LAB_HIDDEN_RECIPE_PRESERVE = new Set<string>([
  'pose_stride',
  'sway_yolo_detection_stride',
  'save_phase_previews',
  'sway_pose_3d_lift',
  'sway_vitpose_use_fast',
])

export function applyLabHiddenSchemaDefaults(
  fields: SchemaField[],
  state: Record<string, unknown>,
): Record<string, unknown> {
  const out = { ...state }
  for (const f of fields) {
    if (f.lab_hidden && f.default !== undefined && f.default !== null) {
      if (LAB_HIDDEN_RECIPE_PRESERVE.has(f.id)) continue
      out[f.id] = f.default
    }
  }
  return out
}

function coerceFieldValue(f: SchemaField, raw: unknown): unknown {
  if (raw === undefined) return undefined

  if (f.type === 'bool') return Boolean(raw)

  if (f.type === 'enum' && f.choices?.length) {
    let normalized = raw
    if (f.id === 'sway_boxmot_reid_model' && String(raw) === 'osnet_x1_0') {
      normalized = 'osnet_x0_25'
    }
    const hit = f.choices.find((c) => String(c) === String(normalized))
    return hit !== undefined ? hit : undefined
  }

  if (f.type === 'int') {
    const n = Number(raw)
    if (!Number.isFinite(n)) return undefined
    let v = Math.round(n)
    if (f.min !== undefined) v = Math.max(f.min, v)
    if (f.max !== undefined) v = Math.min(f.max, v)
    return v
  }

  if (f.type === 'float') {
    const n = Number(raw)
    if (!Number.isFinite(n)) return undefined
    let v = n
    if (f.min !== undefined) v = Math.max(f.min, v)
    if (f.max !== undefined) v = Math.min(f.max, v)
    return v
  }

  if (f.type === 'string') return String(raw)
  return raw
}

function mergeOverrides(
  schema: Schema,
  base: Record<string, unknown>,
  overrides: Record<string, unknown>,
): Record<string, unknown> {
  const byId = new Map(schema.fields.map((field) => [field.id, field]))
  const out = { ...base }
  for (const [id, raw] of Object.entries(overrides)) {
    const field = byId.get(id)
    if (!field) continue
    const v = coerceFieldValue(field, raw)
    if (v !== undefined) out[id] = v
  }
  return out
}

/**
 * Apply presets from all four phase groups, layered in pipeline order.
 * Each group's overrides are independent so any combination is valid.
 */
export function applyPhaseGroupPresets(
  schema: Schema,
  p13Id: string,
  p46Id: string,
  p79Id: string,
  p1011Id: string,
): Record<string, unknown> {
  const base = defaultsFromSchema(schema.fields)
  const p13 = presetById(p13Id)
  const p46 = presetById(p46Id)
  const p79 = presetById(p79Id)
  const p1011 = presetById(p1011Id)

  let state = { ...base }
  if (p13) state = mergeOverrides(schema, state, p13.fields)
  if (p46) state = mergeOverrides(schema, state, p46.fields)
  if (p79) state = mergeOverrides(schema, state, p79.fields)
  if (p1011) state = mergeOverrides(schema, state, p1011.fields)

  return applyLabHiddenSchemaDefaults(schema.fields, state)
}

// ---------------------------------------------------------------------------
// Backward-compatible legacy API (deprecated; kept for old saved runs)
// ---------------------------------------------------------------------------

/** @deprecated Single flat preset; prefer `applyPhaseGroupPresets`. */
export type ConfigPresetId =
  | 'fast_preview'
  | 'standard'
  | 'maximum_accuracy'
  | 'dancer_registry'
  | 'sway_handshake'

/** @deprecated */
export type QualityTierId = 'fast_preview' | 'standard' | 'maximum_accuracy'

/** @deprecated */
export type Phase13StrategyId = 'standard' | 'dancer_registry' | 'sway_handshake'

/** @deprecated */
export const QUALITY_TIER_ORDER: QualityTierId[] = ['fast_preview', 'standard', 'maximum_accuracy']

/** @deprecated */
export const PHASE13_STRATEGY_ORDER: Phase13StrategyId[] = ['standard', 'dancer_registry', 'sway_handshake']

export function phase13StrategyFromFields(fields: Record<string, unknown>): Phase13StrategyId {
  const raw = String(fields.sway_phase13_mode ?? 'standard')
  if (raw === 'dancer_registry' || raw === 'sway_handshake') return raw
  return 'standard'
}

/** @deprecated */
export const QUALITY_TIER_LABELS: Record<QualityTierId, string> = {
  fast_preview: 'Fast preview',
  standard: 'Standard',
  maximum_accuracy: 'Maximum accuracy',
}

/** @deprecated */
export const PHASE13_STRATEGY_LABELS: Record<Phase13StrategyId, string> = {
  standard: 'Default stack',
  dancer_registry: 'Dancer registry',
  sway_handshake: 'Sway handshake',
}

/** @deprecated */
export const QUALITY_TIER_BLURBS: Record<QualityTierId, string> = {
  fast_preview: 'Quick preview preset.',
  standard: 'Standard quality preset.',
  maximum_accuracy: 'Maximum accuracy preset.',
}

/** @deprecated */
export const PHASE13_STRATEGY_BLURBS: Record<Phase13StrategyId, string> = {
  standard: 'Default stack.',
  dancer_registry: 'Dancer registry (experimental).',
  sway_handshake: 'Sway handshake (experimental).',
}

/** @deprecated */
export const CONFIG_PRESET_ORDER: ConfigPresetId[] = [
  'fast_preview',
  'standard',
  'maximum_accuracy',
  'dancer_registry',
  'sway_handshake',
]

/** @deprecated */
export const CONFIG_PRESET_LABELS: Record<ConfigPresetId, string> = {
  fast_preview: 'Fast preview',
  standard: 'Standard',
  maximum_accuracy: 'Maximum accuracy',
  dancer_registry: 'Dancer registry',
  sway_handshake: 'Sway handshake',
}

/** @deprecated */
export const CONFIG_PRESET_BLURBS: Record<ConfigPresetId, string> = {
  fast_preview: QUALITY_TIER_BLURBS.fast_preview,
  standard: QUALITY_TIER_BLURBS.standard,
  maximum_accuracy: QUALITY_TIER_BLURBS.maximum_accuracy,
  dancer_registry: PHASE13_STRATEGY_BLURBS.dancer_registry,
  sway_handshake: PHASE13_STRATEGY_BLURBS.sway_handshake,
}

/** @deprecated Use applyPhaseGroupPresets instead. */
export function applyLabRecipe(
  schema: Schema,
  qualityTier: QualityTierId,
  phase13Strategy: Phase13StrategyId,
): Record<string, unknown> {
  const p13Map: Record<Phase13StrategyId, string> = {
    standard: 'p13_standard',
    dancer_registry: 'p13_dancer_registry',
    sway_handshake: 'p13_sway_handshake',
  }
  const qMap: Record<QualityTierId, [string, string]> = {
    fast_preview: ['p13_fast_scan', 'p46_fast_skeleton'],
    standard: ['p13_standard', 'p46_balanced'],
    maximum_accuracy: ['p13_dense_crowd', 'p46_high_fidelity'],
  }
  const [_q13, q46] = qMap[qualityTier]
  const p13 = qualityTier === 'fast_preview' ? 'p13_fast_scan' : p13Map[phase13Strategy]
  return applyPhaseGroupPresets(schema, p13, q46, 'p79_balanced', 'p1011_standard')
}

/** @deprecated */
export function applyConfigPreset(schema: Schema, preset: ConfigPresetId): Record<string, unknown> {
  const legacyMap: Record<ConfigPresetId, [QualityTierId, Phase13StrategyId]> = {
    fast_preview: ['fast_preview', 'standard'],
    standard: ['standard', 'standard'],
    maximum_accuracy: ['maximum_accuracy', 'standard'],
    dancer_registry: ['standard', 'dancer_registry'],
    sway_handshake: ['standard', 'sway_handshake'],
  }
  const [q, p] = legacyMap[preset]
  return applyLabRecipe(schema, q, p)
}
