import type { Schema, SchemaField } from './types'

export type ConfigPresetId = 'fast_preview' | 'standard' | 'maximum_accuracy'

export const CONFIG_PRESET_ORDER: ConfigPresetId[] = ['fast_preview', 'standard', 'maximum_accuracy']

export const CONFIG_PRESET_LABELS: Record<ConfigPresetId, string> = {
  fast_preview: 'Fast preview',
  standard: 'Standard',
  maximum_accuracy: 'Maximum accuracy',
}

export const CONFIG_PRESET_BLURBS: Record<ConfigPresetId, string> = {
  fast_preview:
    'Fastest rough pass: smaller finder, skeleton every other frame, no overlap fix, looser cleanup numbers, calmer smoothing.',
  standard:
    'Pipeline baseline numbers for pruning, re-linking, and smoothing—what the code defaults to when you don’t pick another preset.',
  maximum_accuracy:
    'Slowest and sharpest: biggest models, overlap fix, outfit matching, stricter cleanup and tighter smoothing.',
}

export function defaultsFromSchema(fields: SchemaField[]): Record<string, unknown> {
  const o: Record<string, unknown> = {}
  for (const f of fields) {
    if (f.default === undefined || f.default === null) continue
    if (f.type === 'string' && f.default === '') continue
    o[f.id] = f.default
  }
  return o
}

/** Partial overrides applied on top of schema defaults. */
const PRESET_OVERRIDES: Record<ConfigPresetId, Record<string, unknown>> = {
  fast_preview: {
    sway_yolo_weights: 'yolo26s',
    sway_hybrid_sam_overlap: false,
    pose_model: 'ViTPose-Base',
    pose_stride: 2,
    temporal_pose_refine: false,
    tracker_technology: 'BoxMOT',
    sway_boxmot_reid_on: false,
    // Looser / faster cleanup vs pipeline baseline (see schema defaults for “Standard”)
    min_duration_ratio: 0.17,
    kinetic_std_frac: 0.025,
    pose_visibility_threshold: 0.28,
    reid_max_frame_gap: 60,
    reid_min_oks: 0.38,
    tier_c_skeleton_mean: 0.14,
    tier_c_low_frame_frac: 0.75,
    mean_confidence_min: 0.42,
    edge_presence_frac: 0.28,
    min_lower_body_conf_yaml: 0.28,
    jitter_ratio_max: 0.12,
    sync_score_min: 0.08,
    prune_threshold: 0.62,
    confirmed_human_min_span_frac: 0.09,
    smoother_min_cutoff: 1.2,
    smoother_beta: 0.75,
  },
  standard: {
    sway_yolo_weights: 'yolo26l',
    sway_hybrid_sam_overlap: true,
    sway_hybrid_sam_iou_trigger: 0.42,
    pose_model: 'ViTPose-Base',
    pose_stride: 1,
    temporal_pose_refine: true,
    temporal_pose_radius: 2,
    tracker_technology: 'BoxMOT',
    sway_global_aflink_mode: 'neural_if_available',
    sway_global_link: true,
    sway_boxmot_reid_on: false,
    // Numeric YAML fields intentionally omitted — use schema defaults (= pipeline baseline).
  },
  maximum_accuracy: {
    sway_yolo_weights: 'yolo26x',
    sway_hybrid_sam_overlap: true,
    sway_hybrid_sam_iou_trigger: 0.35,
    pose_model: 'ViTPose-Large',
    pose_stride: 1,
    temporal_pose_refine: true,
    temporal_pose_radius: 3,
    tracker_technology: 'BoxMOT',
    sway_global_aflink_mode: 'neural_if_available',
    sway_global_link: true,
    sway_boxmot_reid_on: true,
    sway_boxmot_reid_model: 'osnet_x1_0',
    // Stricter cleanup & matching vs baseline
    min_duration_ratio: 0.23,
    kinetic_std_frac: 0.018,
    pose_visibility_threshold: 0.32,
    reid_max_frame_gap: 120,
    reid_min_oks: 0.32,
    tier_c_skeleton_mean: 0.16,
    tier_c_low_frame_frac: 0.85,
    mean_confidence_min: 0.48,
    edge_presence_frac: 0.32,
    min_lower_body_conf_yaml: 0.32,
    jitter_ratio_max: 0.08,
    sync_score_min: 0.12,
    prune_threshold: 0.68,
    confirmed_human_min_span_frac: 0.11,
    smoother_min_cutoff: 0.85,
    smoother_beta: 0.65,
  },
}

function coerceFieldValue(f: SchemaField, raw: unknown): unknown {
  if (raw === undefined) return undefined

  if (f.type === 'bool') return Boolean(raw)

  if (f.type === 'enum' && f.choices?.length) {
    const hit = f.choices.find((c) => String(c) === String(raw))
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

export function applyConfigPreset(schema: Schema, preset: ConfigPresetId): Record<string, unknown> {
  const base = defaultsFromSchema(schema.fields)
  const overrides = PRESET_OVERRIDES[preset]
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
