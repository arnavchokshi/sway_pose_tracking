import type { WatchPhaseId } from './watchPrune'

/**
 * Checkpoint tree columns are resume-depth buckets (Phase 1 = roots, …).
 * Map each column to the phase preview that best matches “output at that stage” for Compare.
 * Deeper columns fall back to final render.
 */
const TREE_COLUMN_TO_PHASE_PREVIEW: (WatchPhaseId | 'final')[] = [
  'phase1_dets',
  'track',
  'pre_pose',
  'pose',
  'collision',
  'post_pose',
]

export function compareViewModeForTreeColumn(colIdx: number): WatchPhaseId | 'final' {
  return TREE_COLUMN_TO_PHASE_PREVIEW[colIdx] ?? 'final'
}
