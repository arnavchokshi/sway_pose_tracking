import type { Schema, SchemaField } from './types'

/** Speed / quality tradeoff: detector size, pose model, scoring & smoothing — independent of Phases 1–3 strategy. */
export type QualityTierId = 'fast_preview' | 'standard' | 'maximum_accuracy'

/** Which early-pipeline stack runs in main.py Phases 1–3 (detection → track → stitch). */
export type Phase13StrategyId = 'standard' | 'dancer_registry' | 'sway_handshake'

/** @deprecated Single flat preset; prefer `applyLabRecipe` with quality + phase13. */
export type ConfigPresetId =
  | 'fast_preview'
  | 'standard'
  | 'maximum_accuracy'
  | 'dancer_registry'
  | 'sway_handshake'

export const QUALITY_TIER_ORDER: QualityTierId[] = ['fast_preview', 'standard', 'maximum_accuracy']

export const PHASE13_STRATEGY_ORDER: Phase13StrategyId[] = ['standard', 'dancer_registry', 'sway_handshake']

/** Align recipe-baseline buttons with stored ``sway_phase13_mode`` (e.g. after opening a saved draft). */
export function phase13StrategyFromFields(fields: Record<string, unknown>): Phase13StrategyId {
  const raw = String(fields.sway_phase13_mode ?? 'standard')
  if (raw === 'dancer_registry' || raw === 'sway_handshake') return raw
  return 'standard'
}

export const QUALITY_TIER_LABELS: Record<QualityTierId, string> = {
  fast_preview: 'Fast preview',
  standard: 'Standard',
  maximum_accuracy: 'Maximum accuracy',
}

export const PHASE13_STRATEGY_LABELS: Record<Phase13StrategyId, string> = {
  standard: 'Default stack',
  dancer_registry: 'Dancer registry',
  sway_handshake: 'Sway handshake',
}

/** What the speed/quality axis adjusts (Phases 1–3 strategy is chosen separately). */
export const QUALITY_TIER_BLURBS: Record<QualityTierId, string> = {
  fast_preview:
    'Smaller YOLO, calmer smoothing, looser scoring cutoffs. Does not change your Phases 1–3 strategy below.',
  standard:
    'DanceTrack YOLO-L, ViTPose-Base, neural stitch when weights exist. Hybrid SAM overlap trigger follows schema unless your strategy below overrides it.',
  maximum_accuracy:
    'Largest YOLO, ViTPose-Large, tighter overlap & cleanup thresholds. Combines with any Phases 1–3 strategy.',
}

/** What each Phases 1–3 strategy controls (orthogonal to speed/quality). */
export const PHASE13_STRATEGY_BLURBS: Record<Phase13StrategyId, string> = {
  standard:
    'Baseline detection → Deep OC-SORT → hybrid SAM + neural stitch. Same stack the quality presets were tuned against.',
  dancer_registry:
    'Experimental: extra zonal crossover + appearance-dormant passes (full video scans); hybrid SAM still uses master overlap lock and your IoU trigger (preset 0.42). Recipe raises pre-track NMS and pins DanceTrack + ViTPose-Base.',
  sway_handshake:
    'Experimental: zonal color registry + SAM mask checks at low IoU trigger (0.10) before BoxMOT to catch ID slips on open floor.',
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

export function defaultsFromSchema(fields: SchemaField[]): Record<string, unknown> {
  const o: Record<string, unknown> = {}
  for (const f of fields) {
    if (f.default === undefined || f.default === null) continue
    if (f.type === 'string' && f.default === '') continue
    o[f.id] = f.default
  }
  return o
}

/** Map removed enum values so old saved runs open cleanly in the Lab. */
export function migrateLegacyPipelineFields(state: Record<string, unknown>): Record<string, unknown> {
  const out = { ...state }
  if (String(out.sway_boxmot_reid_model ?? '') === 'osnet_x1_0') {
    out.sway_boxmot_reid_model = 'osnet_x0_25'
  }
  return out
}

/** Re-apply defaults for `lab_hidden` fields (e.g. locked stride / tracker / interpolation). */
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

/**
 * Early-pipeline strategy: full field overrides for Phases 1–3.
 * `standard` is empty — schema defaults already encode the default stack.
 */
const PHASE13_STRATEGY_OVERRIDES: Record<Phase13StrategyId, Record<string, unknown>> = {
  standard: {},
  dancer_registry: {
    sway_phase13_mode: 'dancer_registry',
    sway_yolo_weights: 'yolo26l_dancetrack',
    sway_pretrack_nms_iou: 0.75,
    tracker_technology: 'deep_ocsort',
    sway_global_aflink_mode: 'neural_if_available',
    sway_hybrid_sam_iou_trigger: 0.42,
    pose_model: 'ViTPose-Base',
  },
  sway_handshake: {
    sway_phase13_mode: 'sway_handshake',
    sway_yolo_weights: 'yolo26l_dancetrack',
    sway_hybrid_sam_iou_trigger: 0.1,
    sway_hybrid_sam_weak_cues: false,
    tracker_technology: 'deep_ocsort',
    pose_model: 'ViTPose-Base',
    sway_global_aflink_mode: 'neural_if_available',
  },
}

/**
 * Speed/quality: applied after phase13 so weights and pose model can stack on experimental strategies.
 * Does not set `sway_phase13_mode` — strategy owns that.
 * `sway_hybrid_sam_iou_trigger` only applies when strategy is `standard` (other strategies set their own overlap behavior).
 */
const QUALITY_TIER_OVERRIDES: Record<QualityTierId, Record<string, unknown>> = {
  fast_preview: {
    sway_yolo_weights: 'yolo26s',
    pose_model: 'ViTPose-Base',
    temporal_pose_refine: false,
    pose_visibility_threshold: 0.28,
    sync_score_min: 0.08,
    smoother_beta: 0.75,
  },
  standard: {
    sway_yolo_weights: 'yolo26l_dancetrack',
    sway_hybrid_sam_iou_trigger: 0.42,
    pose_model: 'ViTPose-Base',
    sway_global_aflink_mode: 'neural_if_available',
  },
  maximum_accuracy: {
    sway_yolo_weights: 'yolo26x',
    sway_hybrid_sam_iou_trigger: 0.35,
    pose_model: 'ViTPose-Large',
    sway_global_aflink_mode: 'neural_if_available',
    pose_visibility_threshold: 0.32,
    sync_score_min: 0.12,
    smoother_beta: 0.65,
  },
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
 * Apply Phases 1–3 strategy, then layer speed/quality on top so e.g. Maximum accuracy + Sway handshake is valid.
 */
export function applyLabRecipe(
  schema: Schema,
  qualityTier: QualityTierId,
  phase13Strategy: Phase13StrategyId,
): Record<string, unknown> {
  const base = defaultsFromSchema(schema.fields)
  const withPhase13 = mergeOverrides(schema, base, PHASE13_STRATEGY_OVERRIDES[phase13Strategy])

  const q = { ...QUALITY_TIER_OVERRIDES[qualityTier] }
  if (phase13Strategy !== 'standard') {
    delete q.sway_hybrid_sam_iou_trigger
  }

  const merged = mergeOverrides(schema, withPhase13, q)
  return applyLabHiddenSchemaDefaults(schema.fields, merged)
}

/** @deprecated Maps old one-shot presets to the split recipe model. */
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
