import type { Schema, SchemaField } from './types'

export type ConfigPresetId = 'fast_preview' | 'standard' | 'maximum_accuracy'

export const CONFIG_PRESET_ORDER: ConfigPresetId[] = ['fast_preview', 'standard', 'maximum_accuracy']

export const CONFIG_PRESET_LABELS: Record<ConfigPresetId, string> = {
  fast_preview: 'Fast preview',
  standard: 'Standard',
  maximum_accuracy: 'Maximum accuracy',
}

export const CONFIG_PRESET_BLURBS: Record<ConfigPresetId, string> = {
  fast_preview: 'Quick check on a new clip — smaller YOLO, stride 2 pose, no SAM or temporal refine.',
  standard: 'Default for most performance videos — YOLO26 L + DanceTrack when trained, hybrid SAM, temporal refine.',
  maximum_accuracy: 'Finals / archival — larger detector & pose, tighter SAM, neural AFLink, Re-ID on.',
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
  },
  standard: {
    sway_yolo_weights: 'yolo26l_dancetrack',
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
  },
  maximum_accuracy: {
    sway_yolo_weights: 'yolo26x_dancetrack',
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
