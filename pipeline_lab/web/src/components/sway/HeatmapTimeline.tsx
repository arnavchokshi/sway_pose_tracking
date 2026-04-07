import { useMemo } from 'react'
import type { CSSProperties } from 'react'
import { formatTimestampMmSs } from '../../swayScoring/formatFeedback'
import { globalScoreGradientStops } from '../../swayScoring/heatmapGradient'

type Props = {
  durationSec: number
  currentTimeSec: number
  /** Sampled along the track so any duration matches mock or real API data. */
  sampleGlobalScore: (timeSec: number) => number
  gradientSegments?: number
  onSeek: (timeSec: number) => void
}

export function HeatmapTimeline({
  durationSec,
  currentTimeSec,
  sampleGlobalScore,
  gradientSegments = 160,
  onSeek,
}: Props) {
  const dur = Math.max(0.001, durationSec)
  const shown = Math.min(dur, Math.max(0, currentTimeSec))
  const pct = (shown / dur) * 100
  const scores = useMemo(() => {
    const n = Math.max(2, gradientSegments)
    const out: number[] = []
    for (let i = 0; i < n; i++) {
      const t = (i / (n - 1)) * dur
      out.push(sampleGlobalScore(t))
    }
    return out
  }, [dur, gradientSegments, sampleGlobalScore])
  const bg = globalScoreGradientStops(scores)
  const sampleScore = sampleGlobalScore(shown)

  const trackStyle = {
    background: bg,
    '--sway-heatmap-pct': `${pct}%`,
  } as CSSProperties

  return (
    <div className="sway-heatmap">
      <div className="sway-heatmap__row">
        <div className="sway-heatmap__label">
          <span className="sway-heatmap__label-title">Team score</span>
          <span className="sway-heatmap__label-hint">Scrub to jump — red = rough sections</span>
        </div>
        <div className="sway-heatmap__meta" aria-live="polite">
          <span className="sway-heatmap__score" title="Score at playhead">
            {Math.round(sampleScore)}
          </span>
          <span className="sway-heatmap__time">
            {formatTimestampMmSs(shown)} <span className="sway-heatmap__time-sep">/</span>{' '}
            {formatTimestampMmSs(dur)}
          </span>
        </div>
      </div>
      <div className="sway-heatmap__track-wrap" style={trackStyle}>
        <input
          type="range"
          className="sway-heatmap__range"
          min={0}
          max={dur}
          step={0.01}
          value={shown}
          onChange={(e) => onSeek(Number(e.target.value))}
          aria-label="Seek video with score heatmap"
        />
      </div>
    </div>
  )
}
