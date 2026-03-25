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
    'Fastest rough pass: smaller finder, no overlap fix, calmer smoothing. Skeleton runs every frame (master default). (Phase 4 pre-pose prune is master-locked.)',
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

/** Re-apply defaults for `lab_hidden` fields (e.g. pose stride fixed to every frame in the Lab). */
export function applyLabHiddenSchemaDefaults(
  fields: SchemaField[],
  state: Record<string, unknown>,
): Record<string, unknown> {
  const out = { ...state }
  for (const f of fields) {
    if (f.lab_hidden && f.default !== undefined && f.default !== null) {
      out[f.id] = f.default
    }
  }
  return out
}

/** Partial overrides applied on top of schema defaults. */
const PRESET_OVERRIDES: Record<ConfigPresetId, Record<string, unknown>> = {
  fast_preview: {
    sway_yolo_weights: 'yolo26s',
    pose_model: 'ViTPose-Base',
    temporal_pose_refine: false,
    tracker_technology: 'deep_ocsort',
    pose_visibility_threshold: 0.28,
    sync_score_min: 0.08,
    prune_threshold: 0.62,
    smoother_beta: 0.75,
  },
  standard: {
    sway_yolo_weights: 'yolo26l_dancetrack',
    sway_hybrid_sam_iou_trigger: 0.42,
    pose_model: 'ViTPose-Base',
    temporal_pose_refine: true,
    tracker_technology: 'deep_ocsort',
    sway_global_aflink_mode: 'neural_if_available',
    // Numeric YAML fields intentionally omitted — use schema defaults (= pipeline baseline).
  },
  maximum_accuracy: {
    sway_yolo_weights: 'yolo26x',
    sway_hybrid_sam_iou_trigger: 0.35,
    pose_model: 'ViTPose-Large',
    temporal_pose_refine: true,
    tracker_technology: 'deep_ocsort_osnet',
    sway_global_aflink_mode: 'neural_if_available',
    sway_boxmot_reid_model: 'osnet_x1_0',
    // Stricter cleanup & matching vs baseline (Phase 4 pre-pose prune is master-locked.)
    pose_visibility_threshold: 0.32,
    sync_score_min: 0.12,
    prune_threshold: 0.68,
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

  return applyLabHiddenSchemaDefaults(schema.fields, out)
}
