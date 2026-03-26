import type { SchemaField } from '../types'

/**
 * Mirrors `_LEGACY_PHASE_TO_INTENT` in `sway/pipeline_config_schema.py`: which schema `phase`
 * values belong under each Config / Run-editor intent stage (`PIPELINE_STAGES[].id`).
 */
export const CONFIG_STAGE_SCHEMA_PHASES: Record<string, readonly string[]> = {
  crowd_control: ['detection'],
  handshake: ['tracking', 'phase3_stitch', 'reid_dedup'],
  pose_polish: ['pose', 'smooth'],
  cleanup_export: ['post_pose_prune', 'scoring', 'export'],
}

export function schemaPhasesForConfigStage(stageId: string): readonly string[] {
  return CONFIG_STAGE_SCHEMA_PHASES[stageId] ?? []
}

export function fieldBelongsToConfigStage(f: SchemaField, stageId: string): boolean {
  return schemaPhasesForConfigStage(stageId).includes(f.phase)
}

/** Shown under the page title — how this screen relates to the Lab. */
export const CONFIG_PAGE_LEDE =
  'Pick a recipe baseline (speed/quality and Phases 1–3 strategy are two separate choices at the top), then a step. Each section explains what those sliders do. (Other pipeline defaults stay fixed for consistency — only the controls here are exposed.)'

/** One-line “what this tab is for” under the step title. */
export const CONFIG_STAGE_INTRO: Record<string, { headline: string; lede: string }> = {
  crowd_control: {
    headline: 'Who we see in the frame',
    lede: 'Detector strength, how overlapping boxes merge, and minimum confidence before someone counts as a person.',
  },
  handshake: {
    headline: 'Crosses, memory, and stitching',
    lede:
      'Start by choosing a Phases 1–3 strategy (detection → track → stitch): default stack, Dancer registry, or Sway handshake — each replaces how that whole early segment runs. Then tune overlap, memory, and long-range stitch below.',
  },
  pose_polish: {
    headline: 'Skeleton quality and motion feel',
    lede: 'Pose model size and whether joints get a light temporal blend (fluid) or stay sharp.',
  },
  cleanup_export: {
    headline: 'Cleanup & export',
    lede: 'Uses fixed pipeline defaults for this segment.',
  },
}

export type ConfigSectionDef = { title: string; blurb: string; fieldIds: string[] }

/** Groups fields so related controls sit together with plain-language section headers. */
export const CONFIG_STAGE_SECTIONS: Record<string, ConfigSectionDef[]> = {
  crowd_control: [
    {
      title: 'Person detector',
      blurb: 'Larger weights are slower and usually better on hard clips.',
      fieldIds: ['sway_yolo_weights'],
    },
    {
      title: 'Boxes before tracking',
      blurb: 'Tighter formations often need higher merge threshold; confidence trims junk detections.',
      fieldIds: ['sway_pretrack_nms_iou', 'sway_yolo_conf'],
    },
  ],
  handshake: [
    {
      title: 'Phases 1–3 · full early pipeline mode',
      blurb:
        'This enum is the same Phases 1–3 strategy as the “Phases 1–3 strategy” row at the top — combine it with a speed/quality tier there. One value sets detection, tracking, and stitch together; default stack is the baseline; Dancer registry and Sway handshake are experimental.',
      fieldIds: ['sway_phase13_mode'],
    },
    {
      title: 'Overlap refinement (SAM)',
      blurb: 'Lower values run refinement sooner when dancers overlap; higher is faster with fewer SAM calls.',
      fieldIds: ['sway_hybrid_sam_iou_trigger'],
    },
    {
      title: 'Tracker memory',
      blurb: 'How many frames an ID can stay “warm” when someone is occluded or off-screen.',
      fieldIds: ['sway_boxmot_max_age'],
    },
    {
      title: 'Long-range stitch',
      blurb: 'How the pipeline reconnects IDs after gaps — neural linker when weights exist, or geometry-only.',
      fieldIds: ['sway_global_aflink_mode'],
    },
  ],
  pose_polish: [
    {
      title: 'Skeleton model',
      blurb: 'Larger ViTPose variants cost more time and VRAM.',
      fieldIds: ['pose_model'],
    },
    {
      title: 'Temporal feel',
      blurb: 'Fluid adds light neighbor smoothing; sharp keeps hits crisp (One-Euro still runs after).',
      fieldIds: ['temporal_pose_refine'],
    },
  ],
}

const fieldById = (fields: SchemaField[], id: string) => fields.find((f) => f.id === id)

export function groupFieldsForStage(
  visibleFields: SchemaField[],
  stageId: string,
): { sections: { def: ConfigSectionDef; fields: SchemaField[] }[]; orphans: SchemaField[] } {
  const sectionsDef = CONFIG_STAGE_SECTIONS[stageId]
  const used = new Set<string>()
  const sections: { def: ConfigSectionDef; fields: SchemaField[] }[] = []

  if (sectionsDef) {
    for (const def of sectionsDef) {
      const chunk: SchemaField[] = []
      for (const fid of def.fieldIds) {
        const f = fieldById(visibleFields, fid)
        if (f) {
          chunk.push(f)
          used.add(fid)
        }
      }
      if (chunk.length > 0) {
        sections.push({ def, fields: chunk })
      }
    }
  }

  const orphans = visibleFields.filter((f) => !used.has(f.id))
  return { sections, orphans }
}
