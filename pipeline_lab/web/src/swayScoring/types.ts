/** API-aligned frame payload (ingest from backend). */
export type SpatialError = {
  joint: string
  error_deg?: number
  msg: string
}

export type TemporalError = {
  offset_sec: number
  msg: string
}

export type FormationError = {
  offset_x?: number
  offset_y?: number
  msg: string
}

/** Normalized [0,1] box in video frame space for weak-link spotlight. */
export type BboxNorm = { x: number; y: number; w: number; h: number }

export type DancerFramePayload = {
  id: number
  aggregate_score: number
  spatial_errors: SpatialError[]
  temporal_error?: TemporalError | null
  formation_error?: FormationError | null
  bbox_norm?: BboxNorm
  /** Golden reference keypoints normalized [0,1] for ghost overlay. */
  golden_keypoints_norm?: Record<string, [number, number]>
}

export type ScoringFramePayload = {
  frame_idx: number
  timestamp: number
  global_team_score: number
  dancers: DancerFramePayload[]
  /** Top-down minimap: dancer id → normalized stage coords [0,1]. */
  formation_targets?: Record<number, { x: number; y: number }>
  formation_actual?: Record<number, { x: number; y: number }>
}

/** ViTPose / COCO-style: x, y, optional confidence. */
export type KeypointMap = Record<string, [number, number, number?]>

export type DancerPoseFrame = {
  id: number
  keypoints: KeypointMap
  /** When set, used by golden-reference fallback (front row median). */
  row?: 'front' | 'back'
}

export type TeamPoseFrame = {
  frame_idx: number
  timestamp?: number
  dancers: DancerPoseFrame[]
}

export type BoneEdge = readonly [string, string]

export type FeedbackKind = 'spatial' | 'temporal' | 'formation'

export type FeedbackLine = {
  id: string
  timestamp: number
  kind: FeedbackKind
  label: string
  dancerId: number
}
