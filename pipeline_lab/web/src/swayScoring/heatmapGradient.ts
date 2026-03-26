/** Map score 0–100 to RGB (red low → green high). */
export function scoreToRgb(score: number): [number, number, number] {
  const s = Math.max(0, Math.min(100, score))
  if (s >= 70) {
    const t = (s - 70) / 30
    const g = Math.round(80 + t * 95)
    const r = Math.round(30 + (1 - t) * 40)
    return [r, g, 55]
  }
  const t = s / 70
  const r = Math.round(120 + (1 - t) * 115)
  const g = Math.round(20 + t * 60)
  return [r, g, 35]
}

export function rgbToCss([r, g, b]: [number, number, number]): string {
  return `rgb(${r},${g},${b})`
}

/**
 * Build CSS linear-gradient stops along a timeline of scores (same length as segments).
 * `scores[i]` colors the segment from i/N to (i+1)/N.
 */
export function globalScoreGradientStops(scores: number[]): string {
  if (scores.length === 0) return 'linear-gradient(90deg, #444 0%, #444 100%)'
  const n = scores.length
  const stops: string[] = []
  for (let i = 0; i < n; i++) {
    const pct0 = (i / n) * 100
    const pct1 = ((i + 1) / n) * 100
    const c = rgbToCss(scoreToRgb(scores[i]!))
    stops.push(`${c} ${pct0.toFixed(3)}%`, `${c} ${pct1.toFixed(3)}%`)
  }
  return `linear-gradient(90deg, ${stops.join(', ')})`
}
