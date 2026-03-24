/** Schema phase id for hybrid SAM env block (see `pipeline_config_schema.py`). */
export const HYBRID_SAM_PHASE_ID = 'hybrid_sam'

/** Hybrid SAM refiner runs only on the BoxMOT path in `tracker.py`; BoT-SORT ignores these env vars. */
export function hideHybridSamPhase(trackerTechnology: unknown): boolean {
  return trackerTechnology === 'BoT-SORT'
}
