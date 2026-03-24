import type { SchemaStage } from '../types'

/** Aligns flowchart nodes with `main.py` [1/11]…[11/11] and `docs/PIPELINE_CODE_REFERENCE.md`. */
export function mainPyPhaseCaption(stage: SchemaStage, flowStepIndex: number): string {
  if (stage.main_phases) return `main.py [${stage.main_phases}/11]`
  return `Step ${flowStepIndex + 1}`
}
