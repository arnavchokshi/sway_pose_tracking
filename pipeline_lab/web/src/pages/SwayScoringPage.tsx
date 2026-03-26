import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useLab } from '../context/LabContext'
import { FeedbackFeed } from '../components/sway/FeedbackFeed'
import { FormationMinimap } from '../components/sway/FormationMinimap'
import { HeatmapTimeline } from '../components/sway/HeatmapTimeline'
import { SwayVideoPlayer } from '../components/sway/SwayVideoPlayer'
import { formatTimestampMmSs, mergeFeedbackTimeline } from '../swayScoring/formatFeedback'
import {
  goldenKeypointsFromTeam,
  spatialScoresVsGolden,
  temporalOffsetSeconds,
  extractDancerHistory,
  formationMeanSpacingError,
} from '../swayScoring/scoringEngine'
import type { DancerPoseFrame, TeamPoseFrame } from '../swayScoring/types'
import { frameAtTime, useSwayScoringStore } from '../swayScoring/store'

/** Tiny synthetic pose history to demonstrate client-side scoring helpers from the master doc. */
function demoEngineSnippet(): string {
  const mk = (dx: number): DancerPoseFrame['keypoints'] => ({
    nose: [100, 40],
    left_shoulder: [85, 70],
    right_shoulder: [115, 70],
    left_elbow: [75, 100],
    right_elbow: [125 + dx, 100],
    left_wrist: [70, 130],
    right_wrist: [130 + dx, 130],
    left_hip: [92, 120],
    right_hip: [108, 120],
    left_knee: [90, 160],
    right_knee: [110, 160],
    left_ankle: [88, 200],
    right_ankle: [112, 200],
  })
  const team: TeamPoseFrame[] = [0, 1, 2].map((i) => ({
    frame_idx: i,
    dancers: [
      { id: 1, row: 'front', keypoints: mk(0) },
      { id: 2, row: 'front', keypoints: mk(2) },
      { id: 3, row: 'back', keypoints: mk(-4) },
    ],
  }))
  const gold = goldenKeypointsFromTeam(team[0]!.dancers, { targetDancerId: 1 })
  const d2 = team[0]!.dancers.find((d) => d.id === 2)!
  const spatial = gold ? spatialScoresVsGolden(gold, d2.keypoints).slice(0, 4) : []
  const refHist = extractDancerHistory(team, 1)
  const d2Hist = extractDancerHistory(team, 2)
  const tempo = temporalOffsetSeconds(refHist, d2Hist, 30)
  const formErr = formationMeanSpacingError(
    [
      { id: 1, x: 0, y: 0 },
      { id: 2, x: 1.05, y: 0 },
    ],
    2,
    1,
    1,
  )
  return `Sample engine outputs — spatial (distal joints): ${spatial.map((s) => `${s.joint}=${s.percent}%`).join(', ')} · temporal Δ: ${tempo.toFixed(3)}s @30fps · formation mean spacing err: ${formErr.toFixed(3)}`
}

export function SwayScoringPage() {
  const { videoFile } = useLab()
  const objectUrl = useMemo(() => (videoFile ? URL.createObjectURL(videoFile) : undefined), [videoFile])
  const videoElRef = useRef<HTMLVideoElement | null>(null)

  const timeline = useSwayScoringStore((s) => s.timeline)
  const durationSec = useSwayScoringStore((s) => s.durationSec)
  const currentTimeSec = useSwayScoringStore((s) => s.currentTimeSec)
  const setCurrentTimeSec = useSwayScoringStore((s) => s.setCurrentTimeSec)
  const viewMode = useSwayScoringStore((s) => s.viewMode)
  const setViewMode = useSwayScoringStore((s) => s.setViewMode)
  const weakLinkEnabled = useSwayScoringStore((s) => s.weakLinkEnabled)
  const setWeakLinkEnabled = useSwayScoringStore((s) => s.setWeakLinkEnabled)
  const goldenTargetDancerId = useSwayScoringStore((s) => s.goldenTargetDancerId)
  const setGoldenTargetDancerId = useSwayScoringStore((s) => s.setGoldenTargetDancerId)
  const selectedDancerId = useSwayScoringStore((s) => s.selectedDancerId)
  const setSelectedDancerId = useSwayScoringStore((s) => s.setSelectedDancerId)
  const seekToFeedbackTimestamp = useSwayScoringStore((s) => s.seekToFeedbackTimestamp)
  const clearDrill = useSwayScoringStore((s) => s.clearDrill)
  const drillLoop = useSwayScoringStore((s) => s.drillLoop)

  useEffect(() => {
    if (!objectUrl) return
    return () => URL.revokeObjectURL(objectUrl)
  }, [objectUrl])

  const lastTimelineT = timeline[timeline.length - 1]?.timestamp ?? 0
  const heatmapDuration = Math.max(durationSec, lastTimelineT, 0.001)

  const sampleGlobalScore = useCallback(
    (t: number) => frameAtTime(timeline, t)?.global_team_score ?? 72,
    [timeline],
  )

  const currentFrame = useMemo(
    () => frameAtTime(timeline, currentTimeSec),
    [timeline, currentTimeSec],
  )

  const feedbackLines = useMemo(() => mergeFeedbackTimeline(timeline), [timeline])

  const dancerIds = useMemo(() => {
    const f = timeline[0]
    if (!f) return []
    return f.dancers.map((d) => d.id)
  }, [timeline])

  const engineHint = useMemo(() => demoEngineSnippet(), [])

  const seek = (t: number) => {
    clearDrill()
    setCurrentTimeSec(t)
    const v = videoElRef.current
    if (v && objectUrl && Number.isFinite(v.duration) && v.duration > 0) {
      try {
        v.currentTime = Math.min(Math.max(0, t), v.duration)
      } catch {
        /* ignore */
      }
    }
  }

  return (
    <div className="sway-scoring-page">
      <header className="sway-scoring-page__header">
        <div>
          <h1 className="sway-scoring-page__title">Scoring studio</h1>
          <p className="sway-scoring-page__sub">
            Captain macro view + dancer micro view — synced canvas, heatmap seek, ghost limbs, formation radar,
            and drill loops. Mock timeline drives UI until your API matches{' '}
            <code className="sway-scoring-page__code">SWAY_SCORING_UI_MASTER.md</code>.
          </p>
        </div>
        <div className="sway-scoring-page__toolbar" role="toolbar" aria-label="Scoring view controls">
          {drillLoop ? (
            <div className="sway-scoring-page__pill" role="status">
              Drill · 0.5× · {formatTimestampMmSs(drillLoop.startSec)}–{formatTimestampMmSs(drillLoop.endSec)}
            </div>
          ) : null}
          <label className="sway-scoring-page__field">
            <span>View</span>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as 'captain' | 'dancer')}
              className="sway-scoring-page__select"
            >
              <option value="captain">Captain (macro)</option>
              <option value="dancer">Dancer (micro)</option>
            </select>
          </label>
          <label className="sway-scoring-page__toggle">
            <input
              type="checkbox"
              checked={weakLinkEnabled}
              onChange={(e) => setWeakLinkEnabled(e.target.checked)}
            />
            Weak-link spotlight
          </label>
          <label className="sway-scoring-page__field">
            <span>Golden ref</span>
            <select
              value={goldenTargetDancerId ?? ''}
              onChange={(e) => {
                const v = e.target.value
                setGoldenTargetDancerId(v === '' ? null : Number(v))
              }}
              className="sway-scoring-page__select"
            >
              <option value="">Median front row (API)</option>
              {dancerIds.map((id) => (
                <option key={id} value={id}>
                  Target dancer #{id}
                </option>
              ))}
            </select>
          </label>
          {viewMode === 'dancer' ? (
            <label className="sway-scoring-page__field">
              <span>You</span>
              <select
                value={selectedDancerId ?? ''}
                onChange={(e) => setSelectedDancerId(Number(e.target.value))}
                className="sway-scoring-page__select"
              >
                {dancerIds.map((id) => (
                  <option key={id} value={id}>
                    Dancer #{id}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
        </div>
      </header>

      <div className="sway-scoring-page__grid">
        <section className="sway-scoring-page__main">
          <div className="sway-scoring-page__stage-card">
            <SwayVideoPlayer
              videoSrc={objectUrl}
              onVideoMount={(el) => {
                videoElRef.current = el
              }}
              onDuration={(d) => {
                if (Number.isFinite(d) && d > 0) {
                  const end = useSwayScoringStore.getState().timeline.at(-1)?.timestamp ?? 0
                  useSwayScoringStore.setState({ durationSec: Math.max(d, end) })
                }
              }}
            />
            <div className="sway-scoring-page__stage-card-inner">
              <HeatmapTimeline
                durationSec={heatmapDuration}
                currentTimeSec={currentTimeSec}
                sampleGlobalScore={sampleGlobalScore}
                onSeek={seek}
              />
            </div>
          </div>
          <details className="sway-scoring-page__engine">
            <summary>Scoring engine utilities (Part 1)</summary>
            <p className="sway-scoring-page__engine-body">{engineHint}</p>
          </details>
        </section>

        <aside className="sway-scoring-page__aside">
          <div className="sway-scoring-panel">
            <FormationMinimap frame={currentFrame} />
          </div>
          <div className="sway-scoring-panel">
            <FeedbackFeed
              lines={feedbackLines}
              selectedDancerId={viewMode === 'dancer' ? selectedDancerId : null}
              onPickLine={(line) => {
                seekToFeedbackTimestamp(line.timestamp)
                requestAnimationFrame(() => {
                  void videoElRef.current?.play()
                })
              }}
            />
          </div>
          <button type="button" className="btn btn--compact sway-scoring-page__clear-drill" onClick={clearDrill}>
            Clear drill loop · 1× speed
          </button>
        </aside>
      </div>
    </div>
  )
}
