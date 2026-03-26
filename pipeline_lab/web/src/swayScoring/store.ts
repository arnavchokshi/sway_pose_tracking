import { create } from 'zustand'
import { buildMockScoringTimeline } from './mockTimeline'
import type { ScoringFramePayload } from './types'

const DEMO_FPS = 30
const INITIAL_TIMELINE = buildMockScoringTimeline({
  durationSec: 14,
  fps: DEMO_FPS,
  dancerIds: [1, 2, 3, 4],
})
const INITIAL_DURATION = INITIAL_TIMELINE[INITIAL_TIMELINE.length - 1]?.timestamp ?? 0

export type ViewMode = 'captain' | 'dancer'

export type DrillLoop = {
  startSec: number
  endSec: number
}

type SwayScoringState = {
  timeline: ScoringFramePayload[]
  fps: number
  durationSec: number
  goldenTargetDancerId: number | null
  selectedDancerId: number | null
  viewMode: ViewMode
  weakLinkEnabled: boolean
  currentTimeSec: number
  isPlaying: boolean
  drillLoop: DrillLoop | null
  basePlaybackRate: number

  setTimeline: (frames: ScoringFramePayload[], fps: number) => void
  setGoldenTargetDancerId: (id: number | null) => void
  setSelectedDancerId: (id: number | null) => void
  setViewMode: (mode: ViewMode) => void
  setWeakLinkEnabled: (on: boolean) => void
  setCurrentTimeSec: (t: number) => void
  setIsPlaying: (p: boolean) => void
  setDrillLoop: (loop: DrillLoop | null) => void
  setPlaybackRate: (r: number) => void
  seekToFeedbackTimestamp: (timestampSec: number) => void
  clearDrill: () => void
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n))
}

export function frameAtTime(timeline: ScoringFramePayload[], t: number): ScoringFramePayload | null {
  if (timeline.length === 0) return null
  let best = timeline[0]!
  let bestD = Math.abs(best.timestamp - t)
  for (const f of timeline) {
    const d = Math.abs(f.timestamp - t)
    if (d < bestD) {
      best = f
      bestD = d
    }
  }
  return best
}

export const useSwayScoringStore = create<SwayScoringState>((set, get) => ({
  timeline: INITIAL_TIMELINE,
  fps: DEMO_FPS,
  durationSec: INITIAL_DURATION,
  goldenTargetDancerId: null,
  selectedDancerId: 3,
  viewMode: 'captain',
  weakLinkEnabled: true,
  currentTimeSec: 0,
  isPlaying: false,
  drillLoop: null,
  basePlaybackRate: 1,

  setTimeline: (frames, fps) => {
    const last = frames[frames.length - 1]
    const durationSec = last ? last.timestamp : 0
    set({ timeline: frames, fps, durationSec })
  },

  setGoldenTargetDancerId: (id) => set({ goldenTargetDancerId: id }),
  setSelectedDancerId: (id) => set({ selectedDancerId: id }),
  setViewMode: (mode) => set({ viewMode: mode }),
  setWeakLinkEnabled: (on) => set({ weakLinkEnabled: on }),
  setCurrentTimeSec: (t) => {
    const { durationSec } = get()
    const next = clamp(t, 0, Math.max(0, durationSec))
    set({ currentTimeSec: next })
  },
  setIsPlaying: (p) => set({ isPlaying: p }),
  setDrillLoop: (loop) => set({ drillLoop: loop }),
  setPlaybackRate: (r) => set({ basePlaybackRate: clamp(r, 0.25, 2) }),

  seekToFeedbackTimestamp: (timestampSec) => {
    const { durationSec } = get()
    const half = 2
    const start = clamp(timestampSec - half, 0, durationSec)
    const end = clamp(timestampSec + half, 0, durationSec)
    set({
      drillLoop: { startSec: start, endSec: end },
      basePlaybackRate: 0.5,
      currentTimeSec: clamp(timestampSec, start, end),
      isPlaying: true,
    })
  },

  clearDrill: () =>
    set({
      drillLoop: null,
      basePlaybackRate: 1,
    }),
}))
