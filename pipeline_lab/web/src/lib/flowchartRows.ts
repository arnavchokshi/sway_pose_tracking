import type { SchemaStage } from '../types'

/** One flowchart card per pipeline stage (tracking includes overlap / hybrid SAM in the same Lab step). */
export type FlowchartRow = { kind: 'single'; index: number }

export function buildPipelineFlowchartRows(stages: SchemaStage[]): FlowchartRow[] {
  return stages.map((_, index) => ({ kind: 'single' as const, index }))
}

export function flowchartRowLastStageIndex(row: FlowchartRow): number {
  return row.index
}

export function flowchartRowIsActive(row: FlowchartRow, activePhaseIndex: number): boolean {
  return activePhaseIndex === row.index
}

export function flowchartRowIsDone(row: FlowchartRow, activePhaseIndex: number): boolean {
  return activePhaseIndex > row.index
}
