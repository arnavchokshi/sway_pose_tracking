import type { SchemaField } from '../types'

export function effectiveFieldTier(f: SchemaField): number {
  if (f.advanced) return 3
  return typeof f.tier === 'number' ? f.tier : 2
}

export function isSchemaFieldVisible(f: SchemaField, state: Record<string, unknown>): boolean {
  const wf = f.visible_when_field
  if (!wf) return true
  const want = f.visible_when_value
  const got = state[wf]
  if (typeof want === 'boolean') return Boolean(got) === want
  return String(got ?? '') === String(want)
}

const TIER1_SORT: string[] = [
  'sway_yolo_weights',
  'tracker_technology',
  'sway_hybrid_sam_overlap',
  'sway_boxmot_reid_on',
  'sway_boxmot_reid_model',
  'pose_model',
  'pose_stride',
  'temporal_pose_refine',
]

export function tier1FieldSortKey(id: string): number {
  const i = TIER1_SORT.indexOf(id)
  return i >= 0 ? i : 900
}
