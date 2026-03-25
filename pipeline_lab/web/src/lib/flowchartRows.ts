import type { SchemaStage } from '../types'

/** Collapse tracking + overlap (same main.py phase band) into one flowchart card. */
export type FlowchartRow =
  | { kind: 'single'; index: number }
  | { kind: 'merged_tracking_overlap'; trackingIndex: number; overlapIndex: number }

export function buildPipelineFlowchartRows(stages: SchemaStage[]): FlowchartRow[] {
  const out: FlowchartRow[] = []
  let i = 0
  while (i < stages.length) {
    const s = stages[i]
    const next = stages[i + 1]
    if (
      s?.id === 'tracking' &&
      next?.id === 'hybrid_sam' &&
      s.main_phases &&
      s.main_phases === next.main_phases
    ) {
      out.push({ kind: 'merged_tracking_overlap', trackingIndex: i, overlapIndex: i + 1 })
      i += 2
    } else {
      out.push({ kind: 'single', index: i })
      i += 1
    }
  }
  return out
}

export function flowchartRowLastStageIndex(row: FlowchartRow): number {
  return row.kind === 'single' ? row.index : row.overlapIndex
}

export function flowchartRowIsActive(row: FlowchartRow, activePhaseIndex: number): boolean {
  if (row.kind === 'single') return activePhaseIndex === row.index
  return activePhaseIndex === row.trackingIndex || activePhaseIndex === row.overlapIndex
}

export function flowchartRowIsDone(row: FlowchartRow, activePhaseIndex: number): boolean {
  return activePhaseIndex > flowchartRowLastStageIndex(row)
}
