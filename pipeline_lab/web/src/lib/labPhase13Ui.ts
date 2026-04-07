import type { Phase13StrategyId } from '../configPresets'

/** Lab server forces these when ``sway_phase13_mode`` is sway_handshake (``app.py`` `_subprocess_env`). */
export const HANDSHAKE_ENFORCED_HYBRID_IOU = 0.1

const _HANDSHAKE_HIDDEN_PREFIXES = ['info_hybrid_sam'] as const

/** Field ids hidden in the UI for Sway handshake — values are enforced at enqueue, not from sliders. */
export const HANDSHAKE_UI_HIDDEN_FIELD_IDS: ReadonlySet<string> = new Set([
  'info_hybrid_sam_master_locked',
  'sway_hybrid_sam_iou_trigger',
  'sway_hybrid_sam_weak_cues',
  'sway_hybrid_weak_conf_delta',
  'sway_hybrid_weak_height_frac',
  'sway_hybrid_weak_match_iou',
])

export function hideHandshakeEnforcedHybridField(fieldId: string): boolean {
  if (HANDSHAKE_UI_HIDDEN_FIELD_IDS.has(fieldId)) return true
  return _HANDSHAKE_HIDDEN_PREFIXES.some((p) => fieldId.startsWith(p))
}

/**
 * Hide schema controls that do not apply to the active Phases 1–3 strategy (misleading or duplicate).
 */
export function hideFieldForPhase13Strategy(fieldId: string, strategy: Phase13StrategyId): boolean {
  if (strategy === 'sway_handshake' && hideHandshakeEnforcedHybridField(fieldId)) return true
  return false
}

/** Strategy is chosen via Recipe baseline; hide duplicate enum from the “Main choices” strip. */
export const PHASE13_MODE_FIELD_ID = 'sway_phase13_mode'

export type Phase13EffectiveRow = { label: string; value: string; detail?: string }

export function phase13EffectiveRows(
  strategy: Phase13StrategyId,
  fieldsState: Record<string, unknown>,
): Phase13EffectiveRow[] {
  const weakOn = Boolean(fieldsState.sway_hybrid_sam_weak_cues)
  const sliderIou = Number(fieldsState.sway_hybrid_sam_iou_trigger ?? 0.42)
  const tt = String(fieldsState.tracker_technology ?? 'deep_ocsort')
  const isByteTrack = tt === 'bytetrack'

  if (strategy === 'standard') {
    return [
      { label: 'Early pipeline', value: 'Default stack (standard)' },
      {
        label: 'Tracking backend',
        value: isByteTrack
          ? 'BoxMOT ByteTrack (fast) — not Deep OC-SORT / OSNet'
          : 'BoxMOT Deep OC-SORT (optional track-time OSNet via pill below)',
      },
      {
        label: 'Hybrid SAM (overlap refiner)',
        value: isByteTrack
          ? 'Off — ByteTrack preview path disables SAM2 overlap refiner for speed'
          : `Runs when pairwise box IoU ≥ ${Number.isFinite(sliderIou) ? sliderIou.toFixed(2) : '0.42'} (from your slider below)`,
      },
      {
        label: 'Weak SAM gate',
        value: weakOn ? 'On — may skip SAM when boxes look stable vs last frame' : 'Off',
      },
      {
        label: 'Sway handshake',
        value: 'Off',
      },
      {
        label: 'Dancer registry passes',
        value: 'Off — no zonal crossover or appearance-dormant relink',
      },
      {
        label: 'ViTPose crop (smart pad)',
        value: 'On — master-locked (SWAY_VITPOSE_SMART_PAD=1 via Lab + main.py §9.0.1)',
      },
    ]
  }

  if (strategy === 'dancer_registry') {
    return [
      { label: 'Early pipeline', value: 'Dancer registry (experimental)' },
      {
        label: 'Tracking backend',
        value: isByteTrack
          ? 'BoxMOT ByteTrack (fast) — not Deep OC-SORT / OSNet'
          : 'BoxMOT Deep OC-SORT (optional OSNet)',
      },
      {
        label: 'Hybrid SAM (overlap refiner)',
        value: isByteTrack
          ? 'Off — ByteTrack preview path disables SAM2 overlap refiner for speed'
          : `Runs when pairwise box IoU ≥ ${Number.isFinite(sliderIou) ? sliderIou.toFixed(2) : '0.42'} (your slider still applies)`,
      },
      {
        label: 'Weak SAM gate',
        value: weakOn ? 'On' : 'Off',
      },
      {
        label: 'Sway handshake',
        value: 'Off',
      },
      {
        label: 'After Phase 2 — crossover verify',
        value: 'On — full video scan, zonal HSV profiles, may swap track IDs after crossovers (no SAM in this pass)',
      },
      {
        label: 'Phase 3 — appearance dormant',
        value: 'On after motion dormant — extra video scan to merge gaps using appearance when motion did not',
      },
      {
        label: 'ViTPose crop (smart pad)',
        value: 'On — same master lock as all strategies (asymmetric / motion-aware crops)',
      },
    ]
  }

  return [
    { label: 'Early pipeline', value: 'Sway handshake (experimental)' },
    {
      label: 'Tracking backend',
      value: isByteTrack
        ? 'BoxMOT ByteTrack — handshake still runs registry/SAM verify when overlap SAM is on; ByteTrack forces overlap SAM off'
        : 'BoxMOT Deep OC-SORT (optional OSNet)',
    },
    {
      label: 'Hybrid SAM (overlap refiner)',
      value: isByteTrack
        ? 'Off — ByteTrack disables overlap SAM (handshake open-floor registry may still use other paths per main.py)'
        : `Runs when pairwise box IoU ≥ ${HANDSHAKE_ENFORCED_HYBRID_IOU.toFixed(2)} — fixed by Lab server (slider value is not used)`,
    },
    {
      label: 'Weak SAM gate',
      value: 'Off — fixed by Lab server for this strategy',
    },
    {
      label: 'Open-floor registry',
      value: 'On — zonal HSV fingerprints when boxes are not overlapping heavily',
    },
    {
      label: 'SAM frame verify',
      value: 'On when SAM runs — may permute det rows / masks so BoxMOT sees IDs aligned with registry (Hungarian; needs SciPy)',
    },
    {
      label: 'Dancer registry passes',
      value: 'Off — no crossover or appearance-dormant passes from dancer_registry',
    },
    {
      label: 'ViTPose crop (smart pad)',
      value: 'On — same master lock as Default stack (not handshake-specific)',
    },
  ]
}
