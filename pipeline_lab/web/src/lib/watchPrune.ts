/** Prune log + Watch phase mapping (client-side). */

export type PruneEntry = Record<string, unknown>

export type PruneLogPayload = {
  video_path?: string
  native_fps?: number
  total_frames?: number
  frame_width?: number
  frame_height?: number
  tracker?: { count?: number; track_ids_before_prune?: number[] }
  surviving_after_pre_pose?: number[]
  surviving_after_post_pose?: number[]
  prune_entries?: PruneEntry[]
}

/** Only these phase preview files appear in Watch (full-length MP4s). */
export const WATCH_PHASE_CLIP_FILES = [
  '01_tracks_post_stitch.mp4',
  '02_pre_pose_prune.mp4',
  '03_pose.mp4',
  '04_phases_6_7.mp4',
  '05_post_pose_prune.mp4',
] as const

export type WatchPhaseId = 'track' | 'pre_pose' | 'pose' | 'collision' | 'post_pose'

export const WATCH_PHASE_ROWS: {
  file: (typeof WATCH_PHASE_CLIP_FILES)[number]
  id: WatchPhaseId
  title: string
  blurb: string
}[] = [
  {
    file: '01_tracks_post_stitch.mp4',
    id: 'track',
    title: 'Phases 1–3 · Tracks after stitch',
    blurb:
      'main.py [1–2/11] YOLO + tracker (+ Hybrid SAM on BoxMOT); then [3/11] post-track stitch / global link. Matches 01_tracks_post_stitch.mp4.',
  },
  {
    file: '02_pre_pose_prune.mp4',
    id: 'pre_pose',
    title: 'Phase 4 · Pre-pose pruning',
    blurb: 'Duration, kinetic, stage polygon, spatial outliers, audience, mirrors, bbox heuristics.',
  },
  {
    file: '03_pose.mp4',
    id: 'pose',
    title: 'Phase 5 · Pose estimation',
    blurb: 'ViTPose on surviving tracks; POSE_VISIBILITY_THRESHOLD in params.yaml when set.',
  },
  {
    file: '04_phases_6_7.mp4',
    id: 'collision',
    title: 'Phases 6–7 · Association & collision cleanup',
    blurb: 'main.py [6/11] occlusion re-ID / crossover; [7/11] dedup + bbox–pose sanitize.',
  },
  {
    file: '05_post_pose_prune.mp4',
    id: 'post_pose',
    title: 'Phase 8 · Post-pose pruning',
    blurb: 'Tier C skeleton quality, Tier B weighted vote (sync, mirror, edge, jitter, …).',
  },
]

const PRE_POSE_RULES = new Set([
  'duration/kinetic',
  'stage_polygon',
  'spatial_outlier',
  'short_track',
  'audience_region',
  'late_entrant_short_span',
  'bbox_size',
  'aspect_ratio',
  'geometric_mirror',
])

export function pruneEntryRule(e: PruneEntry): string {
  return String(e.rule ?? '')
}

/** Stable key for selection / React (entries have no id in JSON). */
export function pruneEntryKey(e: PruneEntry): string {
  const r = pruneEntryRule(e)
  const tid = e.track_id != null ? String(e.track_id) : ''
  const fi = typeof e.frame_idx === 'number' ? String(Math.round(e.frame_idx)) : ''
  const fr = Array.isArray(e.frame_range) ? JSON.stringify(e.frame_range) : ''
  return `${r}|${tid}|${fi}|${fr}`
}

/** xyxy in source video pixels from prune_log.json. */
export function parseBboxXyxy(e: PruneEntry): [number, number, number, number] | null {
  const b = e.bbox_xyxy
  if (Array.isArray(b) && b.length >= 4) {
    const q = b.map((x) => Number(x)) as [number, number, number, number]
    if (q.every((n) => Number.isFinite(n))) return q
  }
  const m = e.bbox_xyxy_median
  if (Array.isArray(m) && m.length >= 4) {
    const q = m.map((x) => Number(x)) as [number, number, number, number]
    if (q.every((n) => Number.isFinite(n))) return q
  }
  return null
}

/**
 * Whether the highlighted box should be drawn at this timeline frame.
 * Per-frame events use exact frame; track-level prunes use frame_range when present.
 */
export function pruneEntryVisibleAtFrame(e: PruneEntry, frame: number): boolean {
  const r = pruneEntryRule(e)
  if (r === 'phase6_summary') return false
  const f = Math.round(frame)
  if (typeof e.frame_idx === 'number') {
    return Math.round(e.frame_idx) === f
  }
  const fr = e.frame_range as number[] | undefined
  if (Array.isArray(fr) && fr.length >= 2 && typeof fr[0] === 'number' && typeof fr[1] === 'number') {
    return f >= fr[0] && f <= fr[1]
  }
  if (Array.isArray(fr) && fr.length >= 1 && typeof fr[0] === 'number') {
    return f >= fr[0]
  }
  // Median-bbox-only rows: show for the whole clip once selected (no frame anchor).
  return parseBboxXyxy(e) != null
}

export type PruneOverlayStyle = { stroke: string; fill: string; label: string }

/** Stroke / fill for canvas overlay by prune rule (dedup vs sanitize vs Tier C vs …). */
export function pruneOverlayStyle(rule: string): PruneOverlayStyle {
  switch (rule) {
    case 'deduplicate_collocated_poses':
      return { stroke: '#fb923c', fill: 'rgba(251, 146, 60, 0.18)', label: 'Dedup (suppressed)' }
    case 'sanitize_pose_bbox_consistency':
      return { stroke: '#c084fc', fill: 'rgba(192, 132, 252, 0.18)', label: 'Sanitize (pose removed)' }
    case 'tier_c_auto_reject':
      return { stroke: '#f87171', fill: 'rgba(248, 113, 113, 0.16)', label: 'Tier C pruned' }
    case 'phase7_voting':
      return { stroke: '#dc2626', fill: 'rgba(220, 38, 38, 0.14)', label: 'Tier B vote pruned' }
    default:
      return { stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.14)', label: 'Pruned (pre/post)' }
  }
}

export function pruneEntriesForWatchPhase(phase: WatchPhaseId, entries: PruneEntry[]): PruneEntry[] {
  const list = entries ?? []
  if (phase === 'track') return []
  if (phase === 'pose') return []
  if (phase === 'pre_pose') {
    return list.filter((e) => PRE_POSE_RULES.has(pruneEntryRule(e)))
  }
  if (phase === 'collision') {
    return list.filter((e) => {
      const r = pruneEntryRule(e)
      return r === 'deduplicate_collocated_poses' || r === 'sanitize_pose_bbox_consistency' || r === 'phase6_summary'
    })
  }
  if (phase === 'post_pose') {
    return list.filter((e) => {
      const r = pruneEntryRule(e)
      return r === 'tier_c_auto_reject' || r === 'phase7_voting'
    })
  }
  return []
}

export const RULE_TITLE: Record<string, string> = {
  'duration/kinetic': 'Duration / motion',
  stage_polygon: 'Outside stage polygon',
  spatial_outlier: 'Far from group (spatial)',
  short_track: 'Too short (< fraction of video)',
  audience_region: 'Audience / edge region',
  late_entrant_short_span: 'Late entrant, short span',
  bbox_size: 'Abnormal bbox height',
  aspect_ratio: 'Non-person aspect ratio',
  geometric_mirror: 'Geometric mirror (edge + velocity)',
  deduplicate_collocated_poses: 'Duplicate pose (same person, two IDs)',
  sanitize_pose_bbox_consistency: 'Pose removed (head outside bbox)',
  tier_c_auto_reject: 'Tier C — weak skeleton',
  phase7_voting: 'Tier B — weighted vote',
  phase6_summary: 'Phase summary',
}

export function humanRule(rule: string): string {
  return RULE_TITLE[rule] ?? rule.replace(/_/g, ' ')
}

export function basenamePath(p: string): string {
  const s = p.replace(/\\/g, '/')
  const i = s.lastIndexOf('/')
  return i >= 0 ? s.slice(i + 1) : s
}

export function formatTimecode(frame: number, fps: number): string {
  if (!fps || fps <= 0 || !Number.isFinite(frame)) return `f${Math.round(frame)}`
  const t = frame / fps
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return `${m}:${s < 10 ? '0' : ''}${s.toFixed(2)}`
}

export function formatBbox(e: PruneEntry): string | null {
  const b = e.bbox_xyxy
  if (Array.isArray(b) && b.length >= 4) {
    return `[${b.map((x) => Number(x).toFixed(0)).join(', ')}]`
  }
  const m = e.bbox_xyxy_median
  if (Array.isArray(m) && m.length >= 4) {
    return `[${m.map((x) => Number(x).toFixed(0)).join(', ')}] (median)`
  }
  return null
}

export function sortPruneEntriesForUi(a: PruneEntry, b: PruneEntry): number {
  const sa = pruneEntryRule(a) === 'phase6_summary' ? 1 : 0
  const sb = pruneEntryRule(b) === 'phase6_summary' ? 1 : 0
  if (sa !== sb) return sa - sb
  const fa = primaryFrame(a)
  const fb = primaryFrame(b)
  if (fa !== fb) return fa - fb
  const ta = Number(a.track_id ?? 0)
  const tb = Number(b.track_id ?? 0)
  return ta - tb
}

function primaryFrame(e: PruneEntry): number {
  if (typeof e.frame_idx === 'number') return e.frame_idx
  const fr = e.frame_range
  if (Array.isArray(fr) && fr.length >= 1 && typeof fr[0] === 'number') return fr[0]
  return 0
}

/** Legend rows for the Watch overlay (phase tabs with prune cards). */
export const WATCH_OVERLAY_LEGEND: { label: string; color: string }[] = [
  { label: 'Pre/post prune (track removed)', color: '#ef4444' },
  { label: 'Dedup — suppressed duplicate ID', color: '#fb923c' },
  { label: 'Sanitize — pose removed (bbox)', color: '#c084fc' },
  { label: 'Tier C — weak skeleton', color: '#f87171' },
  { label: 'Tier B — weighted vote', color: '#dc2626' },
]
