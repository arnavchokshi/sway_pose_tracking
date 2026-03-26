import type { CSSProperties } from 'react'
import { videoContainContentRect } from './videoContentRect'

/** Live overlay: dashed box + label in letterboxed video coordinates (matches burned-in preview). */
export function sam2RoiLayerStyles(
  roi: [number, number, number, number],
  video: HTMLVideoElement,
): { box: CSSProperties; label: CSSProperties } | null {
  const c = videoContainContentRect(video)
  if (!c) return null
  const [x1, y1, x2, y2] = roi
  const left = c.left + x1 * c.scale
  const top = c.top + y1 * c.scale
  const width = Math.max(0, (x2 - x1) * c.scale)
  const height = Math.max(0, (y2 - y1) * c.scale)
  const labelTop = Math.max(c.top + 4, top - 22)
  return {
    box: {
      position: 'absolute',
      left,
      top,
      width,
      height,
      pointerEvents: 'none',
      boxSizing: 'border-box',
      border: '2px dashed rgba(255, 140, 0, 0.92)',
      borderRadius: 6,
      boxShadow: '0 0 0 1px rgba(0,0,0,0.45)',
      zIndex: 3,
    },
    label: {
      position: 'absolute',
      left,
      top: labelTop,
      pointerEvents: 'none',
      fontSize: '0.68rem',
      fontWeight: 700,
      letterSpacing: '0.06em',
      textTransform: 'uppercase',
      color: '#ffb020',
      textShadow: '0 0 8px rgba(0,0,0,0.95)',
      whiteSpace: 'nowrap',
      zIndex: 4,
    },
  }
}
