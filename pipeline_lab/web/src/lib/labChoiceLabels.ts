/**
 * Human-readable labels for pipeline_config / Lab enum-style string values.
 * `lab` = short copy for editor pills and cards. `compare` = fuller wording on the Compare page.
 */
export type LabChoiceLabelMode = 'lab' | 'compare'

export function labChoiceDisplayLabel(fieldId: string, raw: string, mode: LabChoiceLabelMode = 'lab'): string {
  if (mode === 'compare') {
    return labChoiceDisplayLabelCompare(fieldId, raw)
  }
  return labChoiceDisplayLabelLab(fieldId, raw)
}

function labChoiceDisplayLabelCompare(fieldId: string, raw: string): string {
  switch (fieldId) {
    case 'pose_stride':
      if (raw === '1') return 'Every frame'
      if (raw === '2') return 'Every other frame'
      break
    case 'tracker_technology':
      if (raw === 'deep_ocsort') return 'Motion only (Deep OC-SORT)'
      if (raw === 'deep_ocsort_osnet') return 'Deep OC-SORT + in-track Re-ID (OSNet)'
      if (raw === 'bytetrack') return 'ByteTrack (fast preview path)'
      if (raw === 'BoxMOT') return 'Motion only (Deep OC-SORT)'
      if (raw === 'StrongSORT') return 'Deep OC-SORT + in-track Re-ID (OSNet)'
      if (raw === 'BoT-SORT' || raw === 'ByteTrack' || raw === 'OC-SORT') return 'Motion only (Deep OC-SORT)'
      break
    case 'sway_global_aflink_mode':
      if (raw === 'neural_if_available') return 'Neural long-range merge when available'
      if (raw === 'force_heuristic') return 'Heuristic / geometry long-range merge only'
      break
    case 'sway_box_interp_mode':
      if (raw === 'linear') return 'Linear'
      if (raw === 'gsi') return 'GSI (smooth)'
      break
    case 'sway_pose_gap_interp_mode':
      if (raw === 'linear') return 'Linear'
      if (raw === 'gsi') return 'GSI (smooth)'
      break
    case 'sway_vis_temporal_interp_mode':
      if (raw === 'linear') return 'Linear'
      if (raw === 'gsi') return 'GSI (smooth)'
      break
    case 'pose_model':
      if (raw === 'ViTPose-Base') return 'Base — fastest'
      if (raw === 'ViTPose-Large') return 'Large — sharper'
      if (raw === 'ViTPose-Huge') return 'Huge — heaviest'
      if (raw === 'RTMPose-L') return 'RTMPose — speed test'
      if (raw === 'Sapiens (ViTPose-Base fallback)') return 'Sapiens slot — ViT base'
      break
    case 'sway_boxmot_reid_model':
      if (raw === 'osnet_x0_25') return 'OSNet ×0.25 (light, default checkpoint)'
      if (raw === 'osnet_x1_0') return 'OSNet ×1.0 (heavier backbone, more capacity)'
      break
    case 'sway_yolo_weights':
      if (raw === 'yolo26s') return 'YOLO26s'
      if (raw === 'yolo26l') return 'YOLO26l'
      if (raw === 'yolo26l_dancetrack') return 'YOLO26l · DanceTrack'
      if (raw === 'yolo26x') return 'YOLO26x'
      break
    case 'sway_phase13_mode':
      if (raw === 'standard') return 'Standard (Phases 1–3)'
      if (raw === 'dancer_registry') return 'Dancer registry (Phases 1–3)'
      if (raw === 'sway_handshake') return 'Sway handshake (Phases 1–3)'
      break
    default:
      break
  }
  return raw
}

function labChoiceDisplayLabelLab(fieldId: string, raw: string): string {
  switch (fieldId) {
    case 'pose_stride':
      if (raw === '1') return 'Every frame'
      if (raw === '2') return 'Every other frame'
      break
    case 'tracker_technology':
      if (raw === 'deep_ocsort') return 'Default (motion only)'
      if (raw === 'deep_ocsort_osnet') return 'Deep OC-SORT + OSNet'
      if (raw === 'bytetrack') return 'ByteTrack (fast)'
      if (raw === 'BoxMOT') return 'Default (motion only)'
      if (raw === 'StrongSORT') return 'Deep OC-SORT + OSNet'
      if (raw === 'BoT-SORT' || raw === 'ByteTrack' || raw === 'OC-SORT') return 'Default (motion only)'
      break
    case 'sway_global_aflink_mode':
      if (raw === 'neural_if_available') return 'Smart linker'
      if (raw === 'force_heuristic') return 'Simple rules'
      break
    case 'sway_box_interp_mode':
      if (raw === 'linear') return 'Linear'
      if (raw === 'gsi') return 'GSI (smooth)'
      break
    case 'sway_pose_gap_interp_mode':
      if (raw === 'linear') return 'Linear'
      if (raw === 'gsi') return 'GSI (smooth)'
      break
    case 'sway_vis_temporal_interp_mode':
      if (raw === 'linear') return 'Linear'
      if (raw === 'gsi') return 'GSI (smooth)'
      break
    case 'pose_model':
      if (raw === 'ViTPose-Base') return 'Base — fastest'
      if (raw === 'ViTPose-Large') return 'Large — sharper'
      if (raw === 'ViTPose-Huge') return 'Huge — heaviest'
      if (raw === 'RTMPose-L') return 'RTMPose — speed test'
      if (raw === 'Sapiens (ViTPose-Base fallback)') return 'Sapiens slot — ViT base'
      break
    case 'sway_boxmot_reid_model':
      if (raw === 'osnet_x0_25') return 'Lightweight (default)'
      if (raw === 'osnet_x1_0') return 'Heavier OSNet (×1.0)'
      break
    case 'sway_yolo_weights':
      if (raw === 'yolo26s') return 'YOLO26s'
      if (raw === 'yolo26l') return 'YOLO26l'
      if (raw === 'yolo26l_dancetrack') return 'YOLO26l · DanceTrack'
      if (raw === 'yolo26x') return 'YOLO26x'
      break
    case 'sway_phase13_mode':
      if (raw === 'standard') return 'Standard'
      if (raw === 'dancer_registry') return 'Dancer registry'
      if (raw === 'sway_handshake') return 'Sway handshake'
      break
    default:
      break
  }
  return raw
}
