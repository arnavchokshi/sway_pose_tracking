import type { SchemaStage } from '../types'

/**
 * Printed phase label per `docs/MASTER_PIPELINE_GUIDELINE.md` §5 (e.g. "Phases 1–2", "Phase 3").
 * Ranges use plural "Phases"; single printed steps use "Phase".
 */
export function masterGuidePhaseLabel(mainPhases: string): string {
  const isRange =
    mainPhases.includes('–') ||
    mainPhases.includes('-') ||
    /^\d+\s*[-–]\s*\d+$/.test(mainPhases.trim())
  const normalized = mainPhases.replace(/(\d+)\s*-\s*(\d+)/, '$1–$2')
  return isRange ? `Phases ${normalized}` : `Phase ${mainPhases}`
}

/** Aligns flowchart nodes with master guide phase names and `main.py` [1/11]…[11/11]. */
export function mainPyPhaseCaption(stage: SchemaStage, flowStepIndex: number): string {
  if (stage.main_phases) {
    return `${masterGuidePhaseLabel(stage.main_phases)} · main.py [${stage.main_phases}/11]`
  }
  return `Step ${flowStepIndex + 1}`
}
