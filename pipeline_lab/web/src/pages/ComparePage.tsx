import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { API } from '../types'
import type { Schema } from '../types'
import { Pause, Play, SkipBack, SkipForward, X } from 'lucide-react'
import { WATCH_PHASE_ROWS, type WatchPhaseId } from '../lib/watchPrune'

function parseCompareViewQuery(raw: string | null): 'final' | WatchPhaseId {
  const v = raw?.trim() ?? ''
  if (!v || v === 'final') return 'final'
  const row = WATCH_PHASE_ROWS.find((r) => r.id === v)
  return row ? row.id : 'final'
}
import { TrackQualitySummary, PipelineImpactSummary, PipelineImpactReport, FriendlyRunConfig } from '../components/RunMetrics'
import { formatConfigValue } from '../lib/formatConfigValue'
import { labChoiceDisplayLabel } from '../lib/labChoiceLabels'
import { verifyOutputFileExists } from '../lib/verifyOutputUrl'

type RunDetail = {
  run_id: string
  recipe_name?: string
  status?: string
  /** Input clip stem from Lab request (same field as batch list). */
  video_stem?: string
  manifest?: {
    final_video_relpath?: string
    view_variants?: Record<string, string>
    run_context_final?: {
      fields?: Record<string, unknown>
      track_summary?: Record<string, unknown>
      pipeline_diagnostics?: Record<string, unknown>
    }
  }
}

type Slot = {
  run_id: string
  label: string
  src: string | null
  error: string | null
  video_stem: string
  fields?: Record<string, unknown>
  trackSummary?: Record<string, unknown>
  pipelineDiagnostics?: Record<string, unknown>
}

function slotVideoStemKey(s: Slot): string {
  const t = (s.video_stem || '').trim()
  return t || '__none__'
}

/** Rows that differ: show tracker / Re-ID / detector / pose first so the table reads like a model comparison. */
const COMPARE_DIFF_ROW_PRIORITY = [
  'tracker_technology',
  'sway_boxmot_reid_model',
  'sway_global_aflink_mode',
  'sway_yolo_weights',
  'pose_model',
  'pose_stride',
]

const TRACK_SIGNATURE_FIELDS = ['tracker_technology', 'sway_boxmot_reid_model', 'sway_global_aflink_mode'] as const

function sortCompareDiffKeys(keys: string[]): string[] {
  return [...keys].sort((a, b) => {
    const ia = COMPARE_DIFF_ROW_PRIORITY.indexOf(a)
    const ib = COMPARE_DIFF_ROW_PRIORITY.indexOf(b)
    const ra = ia === -1 ? 1000 : ia
    const rb = ib === -1 ? 1000 : ib
    if (ra !== rb) return ra - rb
    return a.localeCompare(b)
  })
}

function trackingStackSignatureKey(slot: Slot): string {
  const f = slot.fields || {}
  return TRACK_SIGNATURE_FIELDS.map((id) => JSON.stringify(f[id])).join('|')
}

function renderCompareConfigCell(fieldId: string, val: unknown): ReactNode {
  if (val === undefined) {
    return <span style={{ color: '#ef4444' }}>Missing</span>
  }
  if (typeof val === 'string' && val.trim() !== '') {
    const nice = labChoiceDisplayLabel(fieldId, val, 'compare')
    if (nice !== val) {
      return (
        <>
          <div style={{ fontWeight: 600, color: '#f8fafc' }}>{nice}</div>
          <div
            style={{
              fontSize: '0.72rem',
              fontFamily: 'ui-monospace, monospace',
              color: '#94a3b8',
              marginTop: '0.2rem',
              lineHeight: 1.35,
            }}
            title="Value stored in run manifest"
          >
            {val}
          </div>
        </>
      )
    }
  }
  return <span>{formatConfigValue(val)}</span>
}

function useSyncedVideos(visibleRunOrder: string[], masterRunId: string | null) {
  const refs = useRef<Record<string, HTMLVideoElement | null>>({})
  const scrubbing = useRef(false)
  const [playing, setPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [current, setCurrent] = useState(0)
  const orderKey = visibleRunOrder.join('\0')

  const playingRef = useRef(playing)
  playingRef.current = playing
  const visibleOrderRef = useRef(visibleRunOrder)
  visibleOrderRef.current = visibleRunOrder
  const masterIdRef = useRef(masterRunId)
  masterIdRef.current = masterRunId

  const syncTime = useCallback(
    (t: number) => {
      const max = duration > 0 ? duration : Number.POSITIVE_INFINITY
      const clamped = Math.max(0, Math.min(t, max))
      for (const id of visibleRunOrder) {
        const el = refs.current[id]
        if (!el) continue
        if (Number.isFinite(el.duration) && el.duration > 0) {
          const cap = el.duration
          el.currentTime = Math.min(clamped, cap - 1e-3)
        } else {
          el.currentTime = clamped
        }
      }
      setCurrent(clamped)
    },
    [duration, orderKey],
  )

  const onMeta = useCallback(() => {
    const durs = visibleRunOrder.map((id) => {
      const el = refs.current[id]
      return el && el.duration && Number.isFinite(el.duration) ? el.duration : 0
    })
    const m = Math.max(...durs, 0)
    if (m > 0) setDuration(m)
  }, [orderKey])

  const togglePlay = useCallback(() => {
    const next = !playing
    setPlaying(next)
    const order = visibleRunOrder
    if (!next) {
      for (const id of order) refs.current[id]?.pause()
      return
    }
    void Promise.all(
      order.map((id) => {
        const el = refs.current[id]
        return el ? el.play() : Promise.resolve()
      }),
    )
      .then(() => {
        const mid = masterIdRef.current
        if (!mid) return
        const master = refs.current[mid]
        if (!master) return
        const t = master.currentTime
        for (const id of order) {
          if (id === mid) continue
          const el = refs.current[id]
          if (!el) continue
          if (Number.isFinite(el.duration) && el.duration > 0) {
            el.currentTime = Math.min(t, el.duration - 1e-3)
          } else {
            el.currentTime = t
          }
        }
      })
      .catch(() => setPlaying(false))
  }, [playing, visibleRunOrder])

  const onTimeUpdateMaster = useCallback((masterId: string) => {
    if (scrubbing.current) return
    const master = refs.current[masterId]
    if (!master) return
    setCurrent(master.currentTime)
  }, [])

  useEffect(() => {
    if (!playing || !masterRunId || visibleRunOrder.length < 2) return
    const master = refs.current[masterRunId]
    if (!master) return

    let cancelled = false
    const driftThreshold = 0.045
    const rVfcHandleRef = { current: 0 }
    const rafRef = { current: 0 }

    const cancelChain = () => {
      cancelled = true
      if (typeof master.cancelVideoFrameCallback === 'function' && rVfcHandleRef.current) {
        try {
          master.cancelVideoFrameCallback(rVfcHandleRef.current)
        } catch {
          /* ignore */
        }
      }
      rVfcHandleRef.current = 0
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = 0
      }
    }

    const syncSlavesTo = (t: number) => {
      const mid = masterIdRef.current
      if (!mid) return
      for (const id of visibleOrderRef.current) {
        if (id === mid) continue
        const el = refs.current[id]
        if (!el) continue
        if (Math.abs(el.currentTime - t) <= driftThreshold) continue
        if (Number.isFinite(el.duration) && el.duration > 0) {
          el.currentTime = Math.min(t, el.duration - 1e-3)
        } else {
          el.currentTime = t
        }
      }
    }

    if (typeof master.requestVideoFrameCallback === 'function') {
      const onFrame: VideoFrameRequestCallback = () => {
        if (cancelled || !playingRef.current || scrubbing.current) return
        const mid = masterIdRef.current
        const m = mid ? refs.current[mid] : null
        if (!m || m.paused) return
        syncSlavesTo(m.currentTime)
        if (!cancelled && playingRef.current && !scrubbing.current && !m.paused) {
          rVfcHandleRef.current = m.requestVideoFrameCallback(onFrame)
        }
      }
      rVfcHandleRef.current = master.requestVideoFrameCallback(onFrame)
    } else {
      const tick = () => {
        if (cancelled || !playingRef.current || scrubbing.current) return
        const mid = masterIdRef.current
        const m = mid ? refs.current[mid] : null
        if (!m || m.paused) return
        syncSlavesTo(m.currentTime)
        rafRef.current = requestAnimationFrame(tick)
      }
      rafRef.current = requestAnimationFrame(tick)
    }

    return cancelChain
  }, [playing, masterRunId, orderKey])

  const setVideoRef = useCallback((runId: string, el: HTMLVideoElement | null) => {
    refs.current[runId] = el
  }, [])

  return {
    videoRefs: refs,
    setVideoRef,
    scrubbing,
    playing,
    duration,
    current,
    syncTime,
    onMeta,
    togglePlay,
    onTimeUpdateMaster,
    setPlaying,
  }
}

/** One directed edge: winner ranks above loser (best → worst list). */
type WinEdge = { w: string; l: string }

type TinderGraphState = {
  pool: string[]
  edges: WinEdge[]
  left: string
  right: string
  endedEarly: boolean
}

function buildWinAdj(edges: WinEdge[]): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>()
  for (const { w, l } of edges) {
    if (!adj.has(w)) adj.set(w, new Set())
    adj.get(w)!.add(l)
  }
  return adj
}

function canReachWinnerToLoser(a: string, b: string, adj: Map<string, Set<string>>): boolean {
  if (a === b) return true
  const seen = new Set<string>()
  const stack = [a]
  while (stack.length) {
    const u = stack.pop()!
    if (u === b) return true
    if (seen.has(u)) continue
    seen.add(u)
    for (const v of adj.get(u) ?? []) stack.push(v)
  }
  return false
}

function knownComparable(a: string, b: string, edges: WinEdge[]): boolean {
  const adj = buildWinAdj(edges)
  return canReachWinnerToLoser(a, b, adj) || canReachWinnerToLoser(b, a, adj)
}

function pickNextIncomparablePair(pool: string[], edges: WinEdge[]): [string, string] | null {
  for (let i = 0; i < pool.length; i++) {
    for (let j = i + 1; j < pool.length; j++) {
      const a = pool[i]
      const b = pool[j]
      if (!knownComparable(a, b, edges)) return [a, b]
    }
  }
  return null
}

function countAmbiguousPairs(pool: string[], edges: WinEdge[]): number {
  let n = 0
  for (let i = 0; i < pool.length; i++) {
    for (let j = i + 1; j < pool.length; j++) {
      if (!knownComparable(pool[i], pool[j], edges)) n++
    }
  }
  return n
}

function directWinLoss(pool: string[], edges: WinEdge[]) {
  const wins = new Map<string, number>()
  const loss = new Map<string, number>()
  for (const id of pool) {
    wins.set(id, 0)
    loss.set(id, 0)
  }
  for (const { w, l } of edges) {
    wins.set(w, (wins.get(w) ?? 0) + 1)
    loss.set(l, (loss.get(l) ?? 0) + 1)
  }
  return { wins, loss }
}

/**
 * Best → worst order from edges: Kahn topological peel; on cycles, break by (wins−losses) then pool order.
 */
function rankPoolFromEdges(pool: string[], edges: WinEdge[]): { order: string[]; approximated: boolean } {
  const idx = new Map(pool.map((id, i) => [id, i]))
  const { wins, loss } = directWinLoss(pool, edges)
  const score = (id: string) => (wins.get(id) ?? 0) - (loss.get(id) ?? 0)
  const tieBreak = (a: string, b: string) => {
    const d = score(b) - score(a)
    if (d !== 0) return d
    return (idx.get(a) ?? 0) - (idx.get(b) ?? 0)
  }

  const order: string[] = []
  let workPool = [...pool]
  let workEdges = [...edges]
  let approximated = false

  while (workPool.length > 0) {
    const inDeg = new Map<string, number>()
    for (const id of workPool) inDeg.set(id, 0)
    for (const { w, l } of workEdges) {
      if (workPool.includes(w) && workPool.includes(l)) {
        inDeg.set(l, (inDeg.get(l) ?? 0) + 1)
      }
    }
    let zeros = workPool.filter((id) => (inDeg.get(id) ?? 0) === 0)
    if (zeros.length === 0) {
      approximated = true
      zeros = [...workPool]
    }
    zeros.sort(tieBreak)
    const pick = zeros[0]
    order.push(pick)
    workPool = workPool.filter((x) => x !== pick)
    workEdges = workEdges.filter((e) => e.w !== pick && e.l !== pick)
  }

  return { order, approximated }
}

function initTinderGraph(ids: string[]): TinderGraphState {
  if (ids.length < 2) {
    return { pool: [...ids], edges: [], left: '', right: '', endedEarly: true }
  }
  const pool = [...ids]
  const first = pickNextIncomparablePair(pool, [])
  if (!first) {
    return { pool, edges: [], left: '', right: '', endedEarly: true }
  }
  return { pool, edges: [], left: first[0], right: first[1], endedEarly: false }
}

type TinderGraphAction =
  | { type: 'reset'; ids: string[] }
  | { type: 'pick'; side: 'left' | 'right' }
  | { type: 'finish' }
  | { type: 'resume' }

function tinderGraphReducer(state: TinderGraphState, action: TinderGraphAction): TinderGraphState {
  if (action.type === 'reset') {
    return initTinderGraph(action.ids)
  }
  if (action.type === 'finish') {
    return { ...state, endedEarly: true, left: '', right: '' }
  }
  if (action.type === 'resume') {
    if (state.pool.length < 2) return { ...state, endedEarly: true, left: '', right: '' }
    const next = pickNextIncomparablePair(state.pool, state.edges)
    return {
      ...state,
      endedEarly: false,
      left: next?.[0] ?? '',
      right: next?.[1] ?? '',
    }
  }
  if (action.type !== 'pick' || state.endedEarly) return state
  if (!state.left || !state.right) return state

  const w = action.side === 'left' ? state.left : state.right
  const l = action.side === 'left' ? state.right : state.left
  const edges = [...state.edges, { w, l }]

  const nextPair = pickNextIncomparablePair(state.pool, edges)
  if (!nextPair) {
    return { ...state, edges, left: '', right: '', endedEarly: false }
  }
  return { ...state, edges, left: nextPair[0], right: nextPair[1] }
}

function tinderPoolSetKey(ids: string[]): string {
  if (ids.length === 0) return ''
  return [...ids].sort().join('\0')
}

function useTinderPairwiseRank(orderedRunIds: string[]) {
  const idsFingerprint = orderedRunIds.join('\x1e')
  const poolKey = useMemo(() => tinderPoolSetKey(orderedRunIds), [idsFingerprint])
  const [state, dispatch] = useReducer(tinderGraphReducer, orderedRunIds, initTinderGraph)

  useEffect(() => {
    dispatch({ type: 'reset', ids: orderedRunIds })
  }, [poolKey])

  const pairForSync = useMemo((): [string, string] | null => {
    if (state.pool.length < 2 || state.endedEarly) return null
    if (!state.left || !state.right) return null
    return [state.left, state.right]
  }, [state.pool.length, state.endedEarly, state.left, state.right])

  const ambiguousPairs = useMemo(() => countAmbiguousPairs(state.pool, state.edges), [state.pool, state.edges])

  const inferredRanking = useMemo(
    () => rankPoolFromEdges(state.pool, state.edges),
    [state.pool, state.edges],
  )

  const totalRuns = orderedRunIds.length
  const comparisonsDone = state.edges.length
  const noActivePair = !state.left || !state.right
  /** Every pair is ordered by your picks or transitive inference; no more comparisons needed. */
  const naturallyComplete = !state.endedEarly && noActivePair && state.pool.length >= 2 && ambiguousPairs === 0

  return {
    state,
    dispatch,
    pairForSync,
    totalRuns,
    ambiguousPairs,
    inferredRanking,
    comparisonsDone,
    naturallyComplete,
  }
}

export function ComparePage() {
  const [search] = useSearchParams()
  const raw = search.get('runs') || ''
  const runIds = raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)

  const viewFromUrl = parseCompareViewQuery(search.get('view'))

  const [slots, setSlots] = useState<Slot[]>([])
  const [viewMode, setViewMode] = useState<'final' | WatchPhaseId>(() => viewFromUrl)
  const [labSchema, setLabSchema] = useState<Schema | null>(null)

  useEffect(() => {
    setViewMode(viewFromUrl)
  }, [viewFromUrl, raw])

  useEffect(() => {
    let cancelled = false
    fetch(`${API}/api/schema`)
      .then((r) => (r.ok ? r.json() : null))
      .then((s) => {
        if (!cancelled && s && typeof s === 'object') setLabSchema(s as Schema)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (runIds.length < 2) {
      setSlots([])
      return
    }
    let cancelled = false
    Promise.all(
      runIds.map((id) =>
        Promise.all([
          fetch(`${API}/api/runs/${id}`).then(async (r) => {
            if (!r.ok) throw new Error(String(r.status))
            return r.json() as Promise<RunDetail>
          }),
          fetch(`${API}/api/runs/${id}/config`).then(async (r) => {
            if (!r.ok) return null
            try {
              return (await r.json()) as { fields?: Record<string, unknown> }
            } catch {
              return null
            }
          }),
        ])
          .then(([data, cfg]) => {
            let rel = data.manifest?.view_variants?.full ?? data.manifest?.final_video_relpath
            if (viewMode !== 'final') {
              const phaseRow = WATCH_PHASE_ROWS.find((r) => r.id === viewMode)
              if (phaseRow) {
                rel = `phase_previews/${phaseRow.file}`
              }
            }
            const src = rel ? `${API}/api/runs/${id}/file/output/${rel}` : null
            let error: string | null = null
            if (!rel) {
              error =
                data.status === 'done'
                  ? `No output file found for ${viewMode} view.`
                  : 'Run not finished yet.'
            }
            // Prefer /config fields: merges checkpoint-tree ancestors + schema defaults (manifest
            // run_context_final.fields only has this job's request.json slice).
            const fieldsFromConfig = cfg?.fields && typeof cfg.fields === 'object' ? cfg.fields : undefined
            const fields =
              fieldsFromConfig ?? (data.manifest?.run_context_final?.fields as Record<string, unknown> | undefined)
            const stem = typeof data.video_stem === 'string' ? data.video_stem : ''
            return {
              run_id: id,
              label: data.recipe_name || id.slice(0, 8),
              src,
              error,
              video_stem: stem,
              fields,
              trackSummary: data.manifest?.run_context_final?.track_summary,
              pipelineDiagnostics: data.manifest?.run_context_final?.pipeline_diagnostics,
            } satisfies Slot
          })
          .catch(() => ({
            run_id: id,
            label: id.slice(0, 8),
            src: null,
            error: 'Could not load run.',
            video_stem: '',
          })),
      ),
    ).then((rows) => {
      if (!cancelled) setSlots(rows)
    })
    return () => {
      cancelled = true
    }
  }, [raw, viewMode])

  const runKey = runIds.join(',')
  /** Per run: output file for current view exists on disk (phase clip may be missing for older runs). */
  const [viewAssetOk, setViewAssetOk] = useState<Record<string, boolean>>({})

  useEffect(() => {
    let cancelled = false
    if (!slots.length) {
      setViewAssetOk({})
      return
    }
    setViewAssetOk({})
    void Promise.all(
      slots.map(async (s) => {
        if (!s.src || s.error) return [s.run_id, false] as const
        const ok = await verifyOutputFileExists(s.src)
        return [s.run_id, ok] as const
      }),
    ).then((rows) => {
      if (cancelled) return
      const next: Record<string, boolean> = {}
      for (const [id, ok] of rows) next[id] = ok
      setViewAssetOk(next)
    })
    return () => {
      cancelled = true
    }
  }, [slots])

  const assetCheckPending = useMemo(
    () => slots.some((s) => s.src && !s.error && !(s.run_id in viewAssetOk)),
    [slots, viewAssetOk],
  )

  const eligibleRunIdsList = useMemo(
    () =>
      slots
        .filter((s) => s.src && !s.error && viewAssetOk[s.run_id] === true)
        .map((s) => s.run_id),
    [slots, viewAssetOk],
  )

  const skippedNoFileCount = useMemo(
    () => slots.filter((s) => s.src && !s.error && viewAssetOk[s.run_id] === false).length,
    [slots, viewAssetOk],
  )

  const eligibleSet = useMemo(() => new Set(eligibleRunIdsList), [eligibleRunIdsList])

  const [visibleRunIds, setVisibleRunIds] = useState<Set<string>>(() => new Set(runIds))
  useEffect(() => {
    setVisibleRunIds(new Set(runIds))
  }, [runKey])

  useEffect(() => {
    if (assetCheckPending) return
    if (slots.length === 0) return
    if (eligibleRunIdsList.length === 0) {
      setVisibleRunIds(new Set())
      return
    }
    const eligible = new Set(eligibleRunIdsList)
    setVisibleRunIds((prev) => {
      const next = new Set([...prev].filter((id) => eligible.has(id)))
      if (next.size === 0) return new Set(eligible)
      return next
    })
  }, [assetCheckPending, eligibleRunIdsList, slots.length])

  const [configSlot, setConfigSlot] = useState<Slot | null>(null)
  /** Empty string = all source videos; otherwise `slotVideoStemKey` value. */
  const [compareVideoStemKey, setCompareVideoStemKey] = useState('')

  const compareVideoStemOptions = useMemo(() => {
    const rows: { key: string; label: string }[] = []
    const seen = new Set<string>()
    for (const s of slots) {
      if (!eligibleSet.has(s.run_id)) continue
      const key = slotVideoStemKey(s)
      if (seen.has(key)) continue
      seen.add(key)
      rows.push({
        key,
        label: key === '__none__' ? '(Unknown source video)' : key,
      })
    }
    rows.sort((a, b) => a.label.localeCompare(b.label))
    return rows
  }, [slots, eligibleSet])

  useEffect(() => {
    if (!compareVideoStemKey) return
    if (compareVideoStemOptions.length === 0) return
    if (!compareVideoStemOptions.some((o) => o.key === compareVideoStemKey)) {
      setCompareVideoStemKey('')
    }
  }, [compareVideoStemOptions, compareVideoStemKey])

  const displaySlots = useMemo(() => {
    return slots.filter((s) => {
      if (!eligibleSet.has(s.run_id) || !visibleRunIds.has(s.run_id)) return false
      if (!compareVideoStemKey) return true
      return slotVideoStemKey(s) === compareVideoStemKey
    })
  }, [slots, eligibleSet, visibleRunIds, compareVideoStemKey])

  const visibleOrder = useMemo(() => displaySlots.map((s) => s.run_id), [displaySlots])
  const orderedRunIdsForTinder = visibleOrder

  const [compareLayout, setCompareLayout] = useState<'grid' | 'tinder'>('grid')
  const tinderRank = useTinderPairwiseRank(orderedRunIdsForTinder)
  const [tinderComments, setTinderComments] = useState<Record<string, string>>({})
  const [tinderRankingOpen, setTinderRankingOpen] = useState(false)

  const tinderPoolKeyForComments = useMemo(() => tinderPoolSetKey(orderedRunIdsForTinder), [orderedRunIdsForTinder.join('\x1e')])
  useEffect(() => {
    setTinderComments({})
  }, [tinderPoolKeyForComments])

  useEffect(() => {
    if (tinderRank.naturallyComplete || tinderRank.state.endedEarly) setTinderRankingOpen(true)
  }, [tinderRank.naturallyComplete, tinderRank.state.endedEarly])

  const visibleOrderForSync = useMemo(() => {
    if (compareLayout === 'tinder') {
      return tinderRank.pairForSync ?? []
    }
    return visibleOrder
  }, [compareLayout, tinderRank.pairForSync, visibleOrder])

  const masterRunId = visibleOrderForSync[0] ?? null

  const { videoRefs, setVideoRef, scrubbing, playing, duration, current, syncTime, onMeta, togglePlay, onTimeUpdateMaster, setPlaying } =
    useSyncedVideos(visibleOrderForSync, masterRunId)

  const tinderPairSlots = useMemo(() => {
    if (!tinderRank.pairForSync) return []
    return tinderRank.pairForSync
      .map((id) => displaySlots.find((s) => s.run_id === id))
      .filter((s): s is Slot => Boolean(s))
  }, [tinderRank.pairForSync, displaySlots])

  const slotByRunId = useMemo(() => {
    const m = new Map<string, Slot>()
    for (const s of displaySlots) m.set(s.run_id, s)
    return m
  }, [displaySlots])

  const toggleRunVisible = useCallback((id: string) => {
    if (!eligibleSet.has(id)) return
    setVisibleRunIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        const eligibleVisible = [...next].filter((x) => eligibleSet.has(x))
        if (eligibleVisible.length <= 1) return prev
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [eligibleSet])

  const selectAllEligibleRuns = useCallback(() => {
    setVisibleRunIds(new Set(eligibleRunIdsList))
  }, [eligibleRunIdsList])

  if (runIds.length < 2) {
    return (
      <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>Select at least two finished runs to compare.</p>
        <p style={{ marginTop: '1rem' }}>
          <Link to="/" className="btn primary" style={{ textDecoration: 'none' }}>
            Go to lab
          </Link>
        </p>
      </div>
    )
  }

  const transportReady =
    compareLayout === 'tinder'
      ? tinderPairSlots.length === 2 && tinderPairSlots.every((s) => s.src && !s.error)
      : displaySlots.length > 0 && displaySlots.every((s) => s.src)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div className="glass-panel" style={{ padding: '1rem 1.5rem' }}>
        <h1 style={{ fontSize: '1.5rem', margin: 0 }}>Compare</h1>
        <p className="sub" style={{ margin: 0, fontSize: '0.9rem' }}>
          {compareLayout === 'tinder'
            ? 'Two clips at a time: pick the better one, add notes, then see a full ranking when you finish.'
            : 'One playhead scrubs and plays every output together.'}
        </p>
        <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Layout:</span>
          <div
            style={{
              display: 'inline-flex',
              borderRadius: 8,
              border: '1px solid var(--glass-border)',
              overflow: 'hidden',
              fontSize: '0.82rem',
            }}
          >
            <button
              type="button"
              onClick={() => setCompareLayout('grid')}
              style={{
                padding: '0.35rem 0.65rem',
                border: 'none',
                cursor: 'pointer',
                background: compareLayout === 'grid' ? 'rgba(34, 211, 238, 0.2)' : 'rgba(0,0,0,0.35)',
                color: compareLayout === 'grid' ? '#e0f7fa' : 'var(--text-muted)',
                fontWeight: compareLayout === 'grid' ? 600 : 500,
              }}
            >
              Grid
            </button>
            <button
              type="button"
              onClick={() => setCompareLayout('tinder')}
              style={{
                padding: '0.35rem 0.65rem',
                border: 'none',
                borderLeft: '1px solid var(--glass-border)',
                cursor: 'pointer',
                background: compareLayout === 'tinder' ? 'rgba(34, 211, 238, 0.2)' : 'rgba(0,0,0,0.35)',
                color: compareLayout === 'tinder' ? '#e0f7fa' : 'var(--text-muted)',
                fontWeight: compareLayout === 'tinder' ? 600 : 500,
              }}
            >
              Tinder
            </button>
          </div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginLeft: '0.25rem' }}>View:</span>
          <select
            value={viewMode}
            onChange={(e) => {
              const v = e.target.value
              setViewMode(v === 'final' ? 'final' : parseCompareViewQuery(v))
              setConfigSlot(null)
            }}
            style={{
              background: 'rgba(0,0,0,0.4)',
              color: '#fff',
              border: '1px solid var(--glass-border)',
              borderRadius: 6,
              padding: '0.35rem 0.5rem',
              fontSize: '0.85rem',
              outline: 'none',
              maxWidth: 'min(100%, 420px)',
            }}
          >
            <option value="final">Final render (Full)</option>
            {WATCH_PHASE_ROWS.map(r => (
              <option key={r.id} value={r.id}>Phase: {r.title}</option>
            ))}
          </select>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Source video:</span>
          <select
            value={compareVideoStemKey}
            onChange={(e) => setCompareVideoStemKey(e.target.value)}
            disabled={compareVideoStemOptions.length === 0}
            style={{
              background: 'rgba(0,0,0,0.4)',
              color: '#fff',
              border: '1px solid var(--glass-border)',
              borderRadius: 6,
              padding: '0.35rem 0.5rem',
              fontSize: '0.85rem',
              outline: 'none',
              maxWidth: 'min(100%, 280px)',
            }}
            title="Filter runs by Lab input clip (video_stem)"
          >
            <option value="">All</option>
            {compareVideoStemOptions.map((o) => (
              <option key={o.key} value={o.key}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        {viewMode === 'track' && (
          <p style={{ margin: '0.5rem 0 0', fontSize: '0.76rem', color: 'var(--text-muted)', maxWidth: 720, lineHeight: 1.45 }}>
            <strong style={{ color: '#94a3b8' }}>Hybrid SAM2:</strong> colored pixels on hybrid-refined tracks are encoded in the phase MP4 when available (re-run with phase previews to refresh old clips). The orange dashed SAM2 input region is burned into the clip; when{' '}
            <code style={{ fontSize: '0.65rem' }}>prune_log.json</code> includes <code style={{ fontSize: '0.65rem' }}>hybrid_sam_frame_rois</code>, the same ROI is drawn here in letterbox sync with the shared playhead (per-run timing).
          </p>
        )}
        {compareLayout === 'grid' && (
          <p style={{ margin: '0.45rem 0 0', fontSize: '0.76rem', color: 'var(--text-muted)', maxWidth: 560, lineHeight: 1.45 }}>
            Click a run name on a tile (e.g. <code style={{ fontSize: '0.7rem' }}>tree_p2_motion_neural_sam30</code>) to see that run&apos;s Lab configuration.
          </p>
        )}
        <div style={{ marginTop: '0.6rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {assetCheckPending ? (
              <>Checking which runs have this clip…</>
            ) : (
              <>
                Shown: {displaySlots.length} run{displaySlots.length === 1 ? '' : 's'} (with clip, visible,{' '}
                {compareVideoStemKey ? 'matching source filter' : 'all sources'})
                <span style={{ color: '#94a3b8' }}>
                  {' '}
                  · {visibleRunIds.size} of {eligibleRunIdsList.length} eligible checked &ldquo;Show&rdquo;
                </span>
                {skippedNoFileCount > 0 && (
                  <span style={{ color: '#94a3b8' }}>
                    {' '}
                    ({skippedNoFileCount} run{skippedNoFileCount === 1 ? '' : 's'} missing this view file)
                  </span>
                )}
              </>
            )}
          </span>
          {!assetCheckPending && eligibleRunIdsList.length > 0 && visibleRunIds.size < eligibleRunIdsList.length && (
            <button type="button" className="btn" style={{ padding: '0.25rem 0.55rem', fontSize: '0.78rem' }} onClick={selectAllEligibleRuns}>
              Show all with clip
            </button>
          )}
        </div>
        {assetCheckPending && slots.length > 0 && (
          <p style={{ margin: '0.75rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Verifying which runs have this view&apos;s file on disk…
          </p>
        )}
        {!assetCheckPending && !transportReady && displaySlots.length > 0 && compareLayout === 'grid' && (
          <p style={{ margin: '0.75rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Waiting for players to load…
          </p>
        )}
        {!assetCheckPending &&
          compareLayout === 'tinder' &&
          tinderPairSlots.length < 2 &&
          !tinderRank.state.endedEarly &&
          !tinderRank.naturallyComplete &&
          orderedRunIdsForTinder.length >= 2 && (
          <p style={{ margin: '0.75rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Preparing pairwise view…
          </p>
        )}
        {!assetCheckPending && eligibleRunIdsList.length === 0 && slots.length > 0 && (
          <p style={{ margin: '0.75rem 0 0', fontSize: '0.85rem', color: '#fcd34d', lineHeight: 1.5, maxWidth: 640 }}>
            None of these runs have a video file for this view (missing phase preview or final output). Enable{' '}
            <strong>save phase previews</strong> in the Lab and re-run, or pick another view.
          </p>
        )}
      </div>

      <div
        className="glass-panel"
        style={{
          padding: '0.75rem 1.5rem',
          position: 'sticky',
          top: '4.5rem',
          zIndex: 40,
          background: 'rgba(20, 24, 34, 0.92)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid var(--glass-border)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', width: '100%' }}>
          <button type="button" className="btn" onClick={() => syncTime(0)} aria-label="Seek start" style={{ padding: '0.5rem' }}>
            <SkipBack size={18} />
          </button>
          <button type="button" className="btn primary" onClick={togglePlay} disabled={!transportReady} style={{ padding: '0.5rem 1rem' }}>
            {playing ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => syncTime(duration - 0.05)}
            disabled={duration <= 0}
            aria-label="Seek end"
            style={{ padding: '0.5rem' }}
          >
            <SkipForward size={18} />
          </button>
          <input
            type="range"
            min={0}
            max={duration > 0 ? duration : 1}
            step={0.01}
            value={duration > 0 ? Math.min(current, duration) : 0}
            disabled={!transportReady || duration <= 0}
            onMouseDown={() => {
              scrubbing.current = true
            }}
            onMouseUp={() => {
              scrubbing.current = false
            }}
            onTouchStart={() => {
              scrubbing.current = true
            }}
            onTouchEnd={() => {
              scrubbing.current = false
            }}
            onChange={(e) => syncTime(parseFloat(e.target.value))}
            style={{ flex: 1, minWidth: 120, accentColor: 'var(--halo-cyan)' }}
          />
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums', minWidth: '80px', textAlign: 'right' }}>
            {formatTime(current)} / {formatTime(duration)}
          </span>
          {compareLayout === 'tinder' && (
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', width: '100%', textAlign: 'right' }}>
              {orderedRunIdsForTinder.length < 2
                ? 'Need at least two visible clips with this view.'
                : tinderRank.state.endedEarly
                  ? 'Finished early — ranking uses your picks + inference (see below).'
                  : tinderRank.naturallyComplete
                    ? 'No ambiguous pairs left — order is fully determined by your comparisons.'
                    : tinderRank.pairForSync
                      ? `${tinderRank.comparisonsDone} comparison${tinderRank.comparisonsDone === 1 ? '' : 's'} · ${tinderRank.ambiguousPairs} pair${tinderRank.ambiguousPairs === 1 ? '' : 's'} still need a direct look (or use Finish early)`
                      : ''}
            </span>
          )}
        </div>
      </div>

      {compareLayout === 'tinder' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {orderedRunIdsForTinder.length < 2 ? (
            <div className="glass-panel" style={{ padding: '1.5rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Tinder mode needs at least two runs that have this clip and are checked &ldquo;Show&rdquo; in the grid. Switch to Grid,
              enable at least two tiles, then return here.
            </div>
          ) : (
            <>
              <div
                className="glass-panel"
                style={{
                  padding: '0.85rem 1.25rem',
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '0.65rem',
                  alignItems: 'center',
                }}
              >
                <button
                  type="button"
                  className="btn"
                  style={{ padding: '0.35rem 0.75rem', fontSize: '0.82rem' }}
                  onClick={() => setTinderRankingOpen((o) => !o)}
                >
                  {tinderRankingOpen ? 'Hide ranking' : 'Show ranking'}
                </button>
                <button
                  type="button"
                  className="btn"
                  style={{ padding: '0.35rem 0.75rem', fontSize: '0.82rem' }}
                  onClick={() => {
                    tinderRank.dispatch({ type: 'reset', ids: orderedRunIdsForTinder })
                    setTinderRankingOpen(false)
                  }}
                >
                  Restart ranking
                </button>
                <button
                  type="button"
                  className="btn primary"
                  style={{ padding: '0.35rem 0.75rem', fontSize: '0.82rem' }}
                  disabled={
                    tinderRank.state.endedEarly ||
                    tinderRank.naturallyComplete ||
                    orderedRunIdsForTinder.length < 2
                  }
                  onClick={() => tinderRank.dispatch({ type: 'finish' })}
                >
                  Finish early
                </button>
                {tinderRank.state.endedEarly && tinderRank.ambiguousPairs > 0 && (
                  <button
                    type="button"
                    className="btn"
                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.82rem' }}
                    onClick={() => tinderRank.dispatch({ type: 'resume' })}
                  >
                    Keep comparing
                  </button>
                )}
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.45, maxWidth: 560 }}>
                  Pairs you <strong style={{ color: '#cbd5e1' }}>don&apos;t</strong> have to see are skipped when the order already follows from
                  your earlier picks (transitive). Use <strong style={{ color: '#cbd5e1' }}>Finish early</strong> anytime — the list below
                  updates from whatever you&apos;ve compared so far. Notes stay in this browser only.
                </span>
              </div>

              {tinderRankingOpen && (
                <div className="glass-panel" style={{ padding: '1.25rem' }}>
                  <h2 style={{ fontSize: '1.05rem', margin: '0 0 0.75rem', color: '#fff' }}>Ranking & comments</h2>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '0 0 1rem', lineHeight: 1.5 }}>
                    Best → worst from your direct picks, <strong style={{ color: '#94a3b8' }}>plus transitive inference</strong> (if A beats B and
                    B beats C, we treat A above C without another click). When you stop early or some pairs never get compared, gaps are
                    filled by win counts and clip order — see the status line. Comments stay in this session only.
                  </p>
                  <div
                    style={{
                      marginBottom: '1rem',
                      padding: '0.65rem 0.9rem',
                      borderRadius: 10,
                      background: 'rgba(15, 23, 42, 0.55)',
                      border: '1px solid var(--glass-border)',
                      fontSize: '0.8rem',
                      color: '#cbd5e1',
                      lineHeight: 1.5,
                    }}
                  >
                    <strong style={{ color: '#94a3b8' }}>Status:</strong> {tinderRank.comparisonsDone} direct comparison
                    {tinderRank.comparisonsDone === 1 ? '' : 's'}
                    {tinderRank.state.endedEarly && (
                      <span style={{ color: '#fcd34d' }}> · finished early</span>
                    )}
                    {tinderRank.naturallyComplete && !tinderRank.state.endedEarly && (
                      <span style={{ color: '#a7f3d0' }}> · fully determined</span>
                    )}
                    {!tinderRank.state.endedEarly && !tinderRank.naturallyComplete && tinderRank.ambiguousPairs > 0 && (
                      <span> · {tinderRank.ambiguousPairs} pair{tinderRank.ambiguousPairs === 1 ? '' : 's'} not yet ordered by your picks</span>
                    )}
                    {tinderRank.inferredRanking.approximated && (
                      <span style={{ color: '#fca5a5' }}>
                        {' '}
                        · order partially guessed (conflicting or missing links — compare more or use Restart)
                      </span>
                    )}
                  </div>
                  {tinderRank.pairForSync && (
                    <div
                      style={{
                        marginBottom: '1rem',
                        padding: '0.75rem 1rem',
                        borderRadius: 10,
                        background: 'rgba(34, 211, 238, 0.08)',
                        border: '1px solid rgba(34, 211, 238, 0.25)',
                        fontSize: '0.84rem',
                        color: '#e2e8f0',
                        lineHeight: 1.55,
                      }}
                    >
                      <div style={{ fontWeight: 700, color: '#67e8f9', marginBottom: '0.35rem' }}>Current pair</div>
                      <div>
                        <span style={{ color: '#94a3b8' }}>Left:</span>{' '}
                        {slotByRunId.get(tinderRank.pairForSync[0])?.label ?? tinderRank.pairForSync[0].slice(0, 8)}
                        {' · '}
                        <span style={{ color: '#94a3b8' }}>Right:</span>{' '}
                        {slotByRunId.get(tinderRank.pairForSync[1])?.label ?? tinderRank.pairForSync[1].slice(0, 8)}
                      </div>
                    </div>
                  )}
                  <div
                    style={{
                      fontSize: '0.72rem',
                      fontWeight: 700,
                      color: '#94a3b8',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      marginBottom: '0.5rem',
                    }}
                  >
                    Estimated order (updates live)
                  </div>
                  <ol style={{ margin: 0, paddingLeft: '1.25rem', color: '#e2e8f0', fontSize: '0.88rem', lineHeight: 1.6 }}>
                    {tinderRank.inferredRanking.order.map((id, i) => {
                      const slot = slotByRunId.get(id)
                      return (
                        <li key={id} style={{ marginBottom: '0.85rem' }}>
                          <div style={{ fontWeight: 600, color: '#f8fafc' }}>
                            #{i + 1}{' '}
                            <button
                              type="button"
                              onClick={() => slot && setConfigSlot(slot)}
                              style={{
                                background: 'none',
                                border: 'none',
                                color: 'var(--halo-cyan)',
                                cursor: slot ? 'pointer' : 'default',
                                font: 'inherit',
                                fontWeight: 600,
                                textDecoration: 'underline',
                                textUnderlineOffset: 3,
                                padding: 0,
                              }}
                            >
                              {slot?.label ?? id.slice(0, 8)}
                            </button>
                          </div>
                          <div
                            style={{
                              fontSize: '0.72rem',
                              fontFamily: 'ui-monospace, monospace',
                              color: '#94a3b8',
                              marginTop: '0.15rem',
                              wordBreak: 'break-all',
                            }}
                          >
                            {id}
                          </div>
                          {tinderComments[id]?.trim() ? (
                            <div style={{ marginTop: '0.35rem', fontSize: '0.82rem', color: '#cbd5e1', whiteSpace: 'pre-wrap' }}>
                              {tinderComments[id].trim()}
                            </div>
                          ) : (
                            <div style={{ marginTop: '0.35rem', fontSize: '0.78rem', color: '#64748b', fontStyle: 'italic' }}>
                              No comment
                            </div>
                          )}
                        </li>
                      )
                    })}
                  </ol>
                </div>
              )}

              {tinderPairSlots.length === 2 && tinderRank.pairForSync && (
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 280px), 1fr))',
                    gap: '1rem',
                  }}
                >
                  {tinderPairSlots.map((s, colIdx) => {
                    const side: 'left' | 'right' = colIdx === 0 ? 'left' : 'right'
                    const role = colIdx === 0 ? 'Clip A (left)' : 'Clip B (right)'
                    return (
                      <div key={s.run_id} className="glass-panel" style={{ overflow: 'hidden', padding: 0, display: 'flex', flexDirection: 'column' }}>
                        <div
                          style={{
                            padding: '0.65rem 1rem',
                            borderBottom: '1px solid var(--glass-border)',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '0.2rem',
                          }}
                        >
                          <span style={{ fontSize: '0.68rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                            {role}
                          </span>
                          <button
                            type="button"
                            onClick={() => setConfigSlot(s)}
                            style={{
                              textAlign: 'left',
                              background: 'transparent',
                              border: 'none',
                              color: 'var(--halo-cyan)',
                              fontWeight: 600,
                              font: 'inherit',
                              cursor: 'pointer',
                              padding: 0,
                              textDecoration: 'underline',
                              textUnderlineOffset: 3,
                            }}
                          >
                            {s.label}
                          </button>
                        </div>
                        <div style={{ background: '#000', aspectRatio: '16/9', position: 'relative', flexShrink: 0 }}>
                          {s.error && (
                            <div style={{ color: '#f87171', padding: '1.5rem', fontSize: '0.9rem' }}>{s.error}</div>
                          )}
                          {s.src && (
                            <video
                              ref={(el) => setVideoRef(s.run_id, el)}
                              src={s.src}
                              preload="metadata"
                              playsInline
                              onLoadedMetadata={onMeta}
                              onPlay={() => setPlaying(true)}
                              onPause={() => {
                                const anyPlaying = visibleOrderForSync.some((id) => {
                                  const elv = videoRefs.current[id]
                                  return elv && !elv.paused
                                })
                                if (!anyPlaying) setPlaying(false)
                              }}
                              muted
                              onTimeUpdate={() => {
                                if (masterRunId && s.run_id === masterRunId) onTimeUpdateMaster(masterRunId)
                              }}
                              style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block', pointerEvents: 'none' }}
                            />
                          )}
                        </div>
                        <div style={{ padding: '0.75rem 1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', flex: 1 }}>
                          <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>Comment on this clip</label>
                          <textarea
                            value={tinderComments[s.run_id] ?? ''}
                            onChange={(e) => setTinderComments((prev) => ({ ...prev, [s.run_id]: e.target.value }))}
                            rows={3}
                            placeholder="Notes…"
                            style={{
                              width: '100%',
                              boxSizing: 'border-box',
                              resize: 'vertical',
                              background: 'rgba(0,0,0,0.35)',
                              border: '1px solid var(--glass-border)',
                              borderRadius: 8,
                              color: '#e2e8f0',
                              fontSize: '0.82rem',
                              padding: '0.5rem 0.65rem',
                              fontFamily: 'inherit',
                            }}
                          />
                          <button
                            type="button"
                            className="btn primary"
                            style={{ width: '100%', padding: '0.55rem', fontSize: '0.88rem' }}
                            onClick={() => tinderRank.dispatch({ type: 'pick', side })}
                          >
                            This clip is better
                          </button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}

              {(tinderRank.naturallyComplete || tinderRank.state.endedEarly) && (
                <div
                  className="glass-panel"
                  style={{
                    padding: '1rem 1.25rem',
                    fontSize: '0.88rem',
                    color: tinderRank.naturallyComplete ? '#a7f3d0' : '#fde68a',
                    lineHeight: 1.5,
                  }}
                >
                  {tinderRank.naturallyComplete && !tinderRank.state.endedEarly ? (
                    <>
                      Every pair is now ordered by your picks (including inferred links). Use <strong>Show ranking</strong> for the list and
                      comments.
                    </>
                  ) : (
                    <>
                      Stopped early — the list above is a <strong>best-effort</strong> order from {tinderRank.comparisonsDone} comparison
                      {tinderRank.comparisonsDone === 1 ? '' : 's'}. Use <strong>Keep comparing</strong> to add more, or <strong>Restart</strong>{' '}
                      to clear.
                    </>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {compareLayout === 'grid' && (
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))',
          gap: '1rem',
        }}
      >
        {!assetCheckPending && eligibleRunIdsList.length === 0 && slots.length > 0 && (
          <div className="glass-panel" style={{ padding: '2rem', gridColumn: '1 / -1', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            No tiles to show for this view — every run in the compare list is missing this output file.
          </div>
        )}
        {displaySlots.map((s) => (
          <div key={s.run_id} className="glass-panel" style={{ overflow: 'hidden', padding: 0 }}>
            <div
              style={{
                padding: '0.75rem 1rem',
                borderBottom: '1px solid var(--glass-border)',
                fontWeight: 600,
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '0.75rem',
              }}
            >
              <button
                type="button"
                onClick={() => setConfigSlot(s)}
                style={{
                  flex: 1,
                  minWidth: 0,
                  textAlign: 'left',
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--halo-cyan)',
                  fontWeight: 600,
                  font: 'inherit',
                  cursor: 'pointer',
                  padding: 0,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  textDecoration: 'underline',
                  textDecorationColor: 'rgba(34, 211, 238, 0.35)',
                  textUnderlineOffset: 3,
                }}
                title="View configuration for this run"
              >
                {s.label}
              </button>
              <label
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: '0.72rem',
                  fontWeight: 500,
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  flexShrink: 0,
                }}
              >
                <input
                  type="checkbox"
                  checked={visibleRunIds.has(s.run_id)}
                  onChange={() => toggleRunVisible(s.run_id)}
                  aria-label={`Show ${s.label}`}
                />
                Show
              </label>
            </div>
            <div
              style={{
                background: '#000',
                aspectRatio: '16/9',
                position: 'relative',
              }}
            >
              {s.error && (
                <div style={{ color: '#f87171', padding: '1.5rem', fontSize: '0.9rem' }}>{s.error}</div>
              )}
              {s.src && (
                <video
                  ref={(el) => setVideoRef(s.run_id, el)}
                  src={s.src}
                  preload="metadata"
                  playsInline
                  onLoadedMetadata={onMeta}
                  onPlay={() => setPlaying(true)}
                  onPause={() => {
                    const anyPlaying = visibleOrderForSync.some((id) => {
                      const elv = videoRefs.current[id]
                      return elv && !elv.paused
                    })
                    if (!anyPlaying) setPlaying(false)
                  }}
                  muted
                  onTimeUpdate={() => {
                    if (masterRunId && s.run_id === masterRunId) onTimeUpdateMaster(masterRunId)
                  }}
                  style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block', pointerEvents: 'none' }}
                />
              )}
            </div>
          </div>
        ))}
      </div>
      )}

      {configSlot && (
        <CompareRunConfigModal
          slot={configSlot}
          schema={labSchema}
          onClose={() => setConfigSlot(null)}
        />
      )}

      {slots.length > 0 && compareLayout === 'grid' && (
        <div style={{ marginTop: '2rem' }}>
          <h2 style={{ fontSize: '1.25rem', marginBottom: '0.35rem', color: '#fff' }}>Tracking & pipeline impact</h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', margin: '0 0 1rem', lineHeight: 1.5, maxWidth: 720 }}>
            Heuristic track cards plus manifest diagnostics so you can see how hybrid SAM, interpolation, and global stitch differed
            run-to-run — not just the final MP4.
          </p>
          {!assetCheckPending && eligibleRunIdsList.length === 0 && (
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: '0 0 1rem' }}>
              No runs have this view&apos;s file — metrics below apply only when at least one run has the clip.
            </p>
          )}
          {eligibleRunIdsList.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${displaySlots.length || 1}, minmax(320px, 1fr))`, gap: '1rem' }}>
            {displaySlots.map((s) => (
              <div key={s.run_id} className="glass-panel" style={{ padding: '1.25rem' }}>
                <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '0.5rem' }}>{s.label}</div>
                <TrackQualitySummary summary={s.trackSummary || {}} />
                <div style={{ marginTop: '0.85rem' }}>
                  <PipelineImpactSummary diagnostics={s.pipelineDiagnostics} />
                </div>
                <details style={{ marginTop: '0.65rem' }}>
                  <summary
                    style={{
                      cursor: 'pointer',
                      color: 'var(--halo-cyan)',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                    }}
                  >
                    Full impact report
                  </summary>
                  <div style={{ marginTop: '0.5rem' }}>
                    <PipelineImpactReport diagnostics={s.pipelineDiagnostics} />
                  </div>
                </details>
              </div>
            ))}
          </div>
          )}

          <h2 style={{ fontSize: '1.25rem', margin: '2.5rem 0 0.35rem', color: '#fff' }}>Configuration differences</h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', margin: '0 0 1rem', lineHeight: 1.5, maxWidth: 720 }}>
            Only keys whose values differ across the visible runs. Enum-style values show a short explanation with the raw value
            underneath. Effective settings use the same merge as the Lab config API (checkpoint-tree ancestors + defaults). Tracker +
            Re-ID + long-range merge are listed first. If several runs share the same recipe name, the run id under the header tells
            columns apart.
          </p>
          <div className="glass-panel" style={{ padding: '1.25rem', overflowX: 'auto' }}>
            {eligibleRunIdsList.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                No runs with this clip — switch view or re-run with phase previews to compare configuration here.
              </div>
            ) : (
            (() => {
              const allKeys = Array.from(new Set(displaySlots.flatMap((s) => Object.keys(s.fields || {}))))
              const diffKeys = sortCompareDiffKeys(
                allKeys.filter((k) => {
                  const vals = new Set(displaySlots.map((s) => JSON.stringify((s.fields || {})[k])))
                  return vals.size > 1
                }),
              )

              const fieldLabel = (id: string) => labSchema?.fields.find((f) => f.id === id)?.label

              const recipeLabelCounts = new Map<string, number>()
              for (const s of displaySlots) {
                recipeLabelCounts.set(s.label, (recipeLabelCounts.get(s.label) ?? 0) + 1)
              }

              const sigGroups = new Map<string, Slot[]>()
              for (const s of displaySlots) {
                const k = trackingStackSignatureKey(s)
                const arr = sigGroups.get(k) ?? []
                arr.push(s)
                sigGroups.set(k, arr)
              }

              if (diffKeys.length === 0) {
                return (
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                    No configuration differences found between the visible runs.
                  </div>
                )
              }

              return (
                <>
                  {sigGroups.size > 1 && (
                    <div
                      style={{
                        marginBottom: '1.1rem',
                        padding: '0.85rem 1rem',
                        borderRadius: 10,
                        background: 'rgba(15, 23, 42, 0.55)',
                        border: '1px solid var(--glass-border)',
                      }}
                    >
                      <div
                        style={{
                          fontSize: '0.78rem',
                          fontWeight: 700,
                          color: '#94a3b8',
                          textTransform: 'uppercase',
                          letterSpacing: '0.06em',
                          marginBottom: '0.55rem',
                        }}
                      >
                        Distinct tracking stacks ({sigGroups.size})
                      </div>
                      <p style={{ margin: '0 0 0.5rem', fontSize: '0.78rem', color: '#94a3b8', lineHeight: 1.45 }}>
                        Grouping by tracker mode, track-time Re-ID checkpoint, and global long-range merge. Useful when recipe names
                        repeat or many columns look the same at a glance.
                      </p>
                      <ul style={{ margin: 0, paddingLeft: '1.1rem', color: '#e2e8f0', fontSize: '0.84rem', lineHeight: 1.55 }}>
                        {[...sigGroups.entries()].map(([sigKey, group]) => {
                          const sample = group[0]
                          const f = sample.fields || {}
                          const line = TRACK_SIGNATURE_FIELDS.map((id) => {
                            const v = f[id]
                            if (v === undefined || v === null || String(v).trim() === '') return null
                            const raw = String(v)
                            const paramLabel = labSchema?.fields.find((x) => x.id === id)?.label ?? id
                            return `${paramLabel}: ${labChoiceDisplayLabel(id, raw, 'compare')}`
                          })
                            .filter(Boolean)
                            .join(' · ')
                          const runList = group
                            .map((g) =>
                              (recipeLabelCounts.get(g.label) ?? 0) > 1
                                ? `${g.label} (${g.run_id.slice(0, 8)}…)`
                                : g.label,
                            )
                            .join(', ')
                          return (
                            <li key={sigKey} style={{ marginBottom: '0.4rem' }}>
                              <span style={{ color: '#f8fafc' }}>{line || '—'}</span>
                              <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>
                                {' '}
                                — {group.length} run{group.length === 1 ? '' : 's'}: {runList}
                              </span>
                            </li>
                          )
                        })}
                      </ul>
                    </div>
                  )}
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                    <thead>
                      <tr>
                        <th
                          style={{
                            padding: '0.75rem',
                            borderBottom: '1px solid var(--glass-border)',
                            color: 'var(--halo-cyan)',
                            width: '28%',
                          }}
                        >
                          Parameter
                        </th>
                        {displaySlots.map((s) => (
                          <th
                            key={s.run_id}
                            style={{ padding: '0.75rem', borderBottom: '1px solid var(--glass-border)', color: '#f8fafc' }}
                          >
                            <div>{s.label}</div>
                            {(recipeLabelCounts.get(s.label) ?? 0) > 1 && (
                              <div
                                style={{
                                  fontSize: '0.68rem',
                                  fontFamily: 'ui-monospace, monospace',
                                  color: '#94a3b8',
                                  fontWeight: 500,
                                  marginTop: '0.35rem',
                                  lineHeight: 1.35,
                                  wordBreak: 'break-all',
                                }}
                                title={s.run_id}
                              >
                                {s.run_id}
                              </div>
                            )}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {diffKeys.map((k) => (
                        <tr key={k} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '0.75rem', color: 'var(--text-muted)', verticalAlign: 'top' }}>
                            <div style={{ color: '#e2e8f0', fontWeight: 600 }}>{fieldLabel(k) ?? k}</div>
                            {fieldLabel(k) && (
                              <div style={{ fontSize: '0.72rem', fontFamily: 'ui-monospace, monospace', marginTop: '0.2rem', opacity: 0.85 }}>
                                {k}
                              </div>
                            )}
                          </td>
                          {displaySlots.map((s) => {
                            const val = (s.fields || {})[k]
                            return (
                              <td key={s.run_id} style={{ padding: '0.75rem', color: '#fff', verticalAlign: 'top', lineHeight: 1.45 }}>
                                {renderCompareConfigCell(k, val)}
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )
            })()
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function CompareRunConfigModal({
  slot,
  schema,
  onClose,
}: {
  slot: Slot
  schema: Schema | null
  onClose: () => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 200,
        background: 'rgba(0,0,0,0.72)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.25rem',
      }}
      onClick={onClose}
      role="presentation"
    >
      <div
        className="glass-panel"
        style={{
          maxWidth: 800,
          width: '100%',
          maxHeight: 'min(90vh, 900px)',
          overflow: 'auto',
          padding: '1.25rem',
          position: 'relative',
        }}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="compare-run-config-title"
      >
        <button
          type="button"
          className="btn"
          aria-label="Close"
          onClick={onClose}
          style={{ position: 'absolute', top: 12, right: 12, padding: '0.35rem' }}
        >
          <X size={18} />
        </button>
        <h2 id="compare-run-config-title" style={{ fontSize: '1.15rem', margin: '0 2rem 0.35rem 0', color: '#fff' }}>
          {slot.label}
        </h2>
        <div
          style={{
            fontSize: '0.72rem',
            fontFamily: 'ui-monospace, monospace',
            color: 'var(--text-muted)',
            marginBottom: '1rem',
            wordBreak: 'break-all',
          }}
        >
          {slot.run_id}
        </div>
        <FriendlyRunConfig
          fields={slot.fields}
          schemaFields={schema?.fields}
          stages={schema?.stages}
        />
      </div>
    </div>
  )
}

function formatTime(s: number) {
  if (!Number.isFinite(s) || s < 0) return '0:00'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}
