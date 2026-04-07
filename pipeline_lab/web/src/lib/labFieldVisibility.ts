/** True for overlap / hybrid SAM schema fields (merged into the tracking phase in the Lab UI). */
export function isHybridSamOverlapField(fieldId: string): boolean {
  return fieldId.startsWith('info_hybrid_sam') || fieldId.startsWith('sway_hybrid_sam_') || fieldId.startsWith('sway_hybrid_weak_')
}

/** Hybrid SAM refiner runs on the BoxMOT Deep OC-SORT path; ByteTrack disables SAM server-side. */
export function hideHybridSamPhase(trackerTechnology: unknown, _phase13Mode?: unknown): boolean {
  return trackerTechnology === 'bytetrack' || trackerTechnology === 'BoT-SORT'
}
