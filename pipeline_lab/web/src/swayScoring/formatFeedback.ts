import type { FeedbackLine, ScoringFramePayload, SpatialError } from './types'

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n)
}

export function formatTimestampMmSs(t: number): string {
  if (!Number.isFinite(t) || t < 0) return '0:00'
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${pad2(s)}`
}

function spatialEmoji(): string {
  return '🔴'
}

function temporalEmoji(): string {
  return '🟡'
}

function formationEmoji(): string {
  return '🟦'
}

export function formatSpatialFeedLine(timestamp: number, msg: string): string {
  return `${spatialEmoji()} ${formatTimestampMmSs(timestamp)} - Spatial: "${msg}"`
}

export function formatTemporalFeedLine(timestamp: number, offsetSec: number, msg?: string): string {
  const late = offsetSec < 0
  const mag = Math.abs(offsetSec).toFixed(1)
  const hint =
    msg ??
    (late
      ? `You are ${mag}s late. Anticipate the snare.`
      : `You are ${mag}s early. Wait for the hit.`)
  return `${temporalEmoji()} ${formatTimestampMmSs(timestamp)} - Timing: "${hint}"`
}

export function formatFormationFeedLine(timestamp: number, msg: string): string {
  return `${formationEmoji()} ${formatTimestampMmSs(timestamp)} - Formation: "${msg}"`
}

/** Build scrollable feed entries from a single frame (merge timelines at page level). */
export function feedbackLinesFromFrame(frame: ScoringFramePayload): FeedbackLine[] {
  const lines: FeedbackLine[] = []
  const t = frame.timestamp
  for (const d of frame.dancers) {
    const base = `${frame.frame_idx}-${d.id}`
    for (let i = 0; i < d.spatial_errors.length; i++) {
      const e = d.spatial_errors[i]!
      const msg = e.msg || humanSpatialFromJoint(e)
      lines.push({
        id: `${base}-sp-${i}`,
        timestamp: t,
        kind: 'spatial',
        label: formatSpatialFeedLine(t, msg),
        dancerId: d.id,
      })
    }
    if (d.temporal_error) {
      const te = d.temporal_error
      lines.push({
        id: `${base}-tm`,
        timestamp: t,
        kind: 'temporal',
        label: formatTemporalFeedLine(t, te.offset_sec, te.msg),
        dancerId: d.id,
      })
    }
    if (d.formation_error?.msg) {
      lines.push({
        id: `${base}-fm`,
        timestamp: t,
        kind: 'formation',
        label: formatFormationFeedLine(t, d.formation_error.msg),
        dancerId: d.id,
      })
    }
  }
  return lines
}

function humanSpatialFromJoint(e: SpatialError): string {
  const j = e.joint.replace(/_/g, ' ')
  const deg = e.error_deg
  if (deg == null) return `Adjust your ${j}.`
  if (deg < 0) return `Raise your ${j} on the hit.`
  return `Lower your ${j} on the hit.`
}

export function mergeFeedbackTimeline(frames: ScoringFramePayload[]): FeedbackLine[] {
  const all: FeedbackLine[] = []
  for (const f of frames) {
    all.push(...feedbackLinesFromFrame(f))
  }
  all.sort((a, b) => a.timestamp - b.timestamp || a.id.localeCompare(b.id))
  return all
}
