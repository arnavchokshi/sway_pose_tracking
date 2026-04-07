import type { SchemaStage } from '../types'

/**
 * Printed phase label per `docs/MASTER_PIPELINE_GUIDELINE.md` §5 (e.g. "Phases 1–2", "Phase 3").
 * Supports comma-separated groups: "8, 10–11", "1–3, 6–7".
 */
export function masterGuidePhaseLabel(mainPhases: string): string {
  const normalized = mainPhases.replace(/(\d+)\s*-\s*(\d+)/g, '$1–$2').trim()
  const single = /^\d+$/.test(normalized)
  if (single) return `Phase ${normalized}`
  return `Phases ${normalized}`
}

/** Flowchart subtitle: technical main.py phases when present; otherwise intent short label. */
export function mainPyPhaseCaption(stage: SchemaStage, flowStepIndex: number): string {
  if (stage.main_phases) {
    return `${masterGuidePhaseLabel(stage.main_phases)} · main.py (11 printed phases)`
  }
  return stage.short || `Step ${flowStepIndex + 1}`
}
