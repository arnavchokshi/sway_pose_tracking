/** Empty = same-origin `/api` (dev via Vite proxy or bundled UI on 8765). See `siteUrls.ts`. */
export const API = ''

export type SchemaStage = { id: string; label: string; short: string; main_phases?: string }
export type SchemaField = {
  id: string
  phase: string
  label: string
  type: string
  default?: unknown
  binding: string
  key: string
  choices?: (string | number)[]
  min?: number
  max?: number
  advanced?: boolean
  description?: string
  /** 1 = primary strip, 2 = per-phase tune, 3 = advanced (or use advanced: true). */
  tier?: number
  visible_when_field?: string
  visible_when_value?: unknown
  /** UI hint: slider | segmented | model_cards | pruning_weight | tracker_strip | phase13_mode_cards */
  display?: string
  disabled_choices?: (string | number)[]
  /** When true, Pipeline Lab does not render this control (CLI/API may still set it). */
  lab_hidden?: boolean
}

export type Schema = { stages: SchemaStage[]; fields: SchemaField[] }

export type ProgressLine = {
  stage: number
  stage_key?: string
  label?: string
  preview_relpath?: string
  status?: string
  elapsed_s?: number
  cumulative_elapsed_s?: number
  ts?: number
  extra?: Record<string, unknown>
  meta?: Record<string, unknown>
}

/** From ``GET /api/batches`` — grouped runs sharing ``batch_id`` (matrix / checkpoint tree / batch upload). */
export type BatchSummary = {
  batch_id: string
  run_count: number
  n_queued: number
  n_running_live: number
  n_running_stale: number
  n_done: number
  n_error: number
  n_cancelled: number
  has_checkpoint_tree: boolean
  latest_created: string
}

export type RunInfo = {
  run_id: string
  status: string
  recipe_name: string
  video_stem: string
  /** Present when the run was part of a batch upload or batch_path matrix queue. */
  batch_id?: string
  /** Parent run when ``checkpoint.resume_from`` points at ``../<parent>/output/checkpoints/...`` (checkpoint tree / pipeline_tree_queue). */
  parent_run_id?: string
  created?: string
  progress?: ProgressLine[]
  error?: string
  /** True only when this API process holds a live main.py Popen. False after restart or if process exited. */
  subprocess_alive?: boolean
  /** Present when run_manifest.json exists (includes Lab run_context_final). */
  manifest?: {
    run_context_final?: Record<string, unknown> & {
      track_summary?: Record<string, unknown>
      fields?: Record<string, unknown>
      pipeline_diagnostics?: Record<string, unknown>
      recipe_name?: string
    }
  }
}
