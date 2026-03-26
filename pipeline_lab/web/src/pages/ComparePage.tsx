import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
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
  fields?: Record<string, unknown>
  trackSummary?: Record<string, unknown>
  pipelineDiagnostics?: Record<string, unknown>
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
            return {
              run_id: id,
              label: data.recipe_name || id.slice(0, 8),
              src,
              error,
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

  const displaySlots = useMemo(() => {
    return slots.filter((s) => eligibleSet.has(s.run_id) && visibleRunIds.has(s.run_id))
  }, [slots, eligibleSet, visibleRunIds])

  const visibleOrder = useMemo(() => displaySlots.map((s) => s.run_id), [displaySlots])
  const masterRunId = visibleOrder[0] ?? null

  const { videoRefs, setVideoRef, scrubbing, playing, duration, current, syncTime, onMeta, togglePlay, onTimeUpdateMaster, setPlaying } =
    useSyncedVideos(visibleOrder, masterRunId)

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

  const ready = displaySlots.length > 0 && displaySlots.every((s) => s.src)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div className="glass-panel" style={{ padding: '1rem 1.5rem' }}>
        <h1 style={{ fontSize: '1.5rem', margin: 0 }}>Compare</h1>
        <p className="sub" style={{ margin: 0, fontSize: '0.9rem' }}>
          One playhead scrubs and plays every output together.
        </p>
        <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>View:</span>
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
        </div>
        {viewMode === 'track' && (
          <p style={{ margin: '0.5rem 0 0', fontSize: '0.76rem', color: 'var(--text-muted)', maxWidth: 720, lineHeight: 1.45 }}>
            <strong style={{ color: '#94a3b8' }}>Hybrid SAM2:</strong> colored pixels on hybrid-refined tracks are encoded in the phase MP4 when available (re-run with phase previews to refresh old clips). The orange dashed SAM2 input region is burned into the clip; when{' '}
            <code style={{ fontSize: '0.65rem' }}>prune_log.json</code> includes <code style={{ fontSize: '0.65rem' }}>hybrid_sam_frame_rois</code>, the same ROI is drawn here in letterbox sync with the shared playhead (per-run timing).
          </p>
        )}
        <p style={{ margin: '0.45rem 0 0', fontSize: '0.76rem', color: 'var(--text-muted)', maxWidth: 560, lineHeight: 1.45 }}>
          Click a run name on a tile (e.g. <code style={{ fontSize: '0.7rem' }}>tree_p2_motion_neural_sam30</code>) to see that run&apos;s Lab configuration.
        </p>
        <div style={{ marginTop: '0.6rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {assetCheckPending ? (
              <>Checking which runs have this clip…</>
            ) : (
              <>
                Videos: {visibleRunIds.size} of {eligibleRunIdsList.length} with this clip
                {skippedNoFileCount > 0 && (
                  <span style={{ color: '#94a3b8' }}>
                    {' '}
                    ({skippedNoFileCount} run{skippedNoFileCount === 1 ? '' : 's'} hidden — no file for this view)
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
        {!assetCheckPending && !ready && displaySlots.length > 0 && (
          <p style={{ margin: '0.75rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Waiting for players to load…
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
          <button type="button" className="btn primary" onClick={togglePlay} disabled={!ready} style={{ padding: '0.5rem 1rem' }}>
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
            disabled={!ready || duration <= 0}
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
        </div>
      </div>

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
                    const anyPlaying = visibleOrder.some((id) => {
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

      {configSlot && (
        <CompareRunConfigModal
          slot={configSlot}
          schema={labSchema}
          onClose={() => setConfigSlot(null)}
        />
      )}

      {slots.length > 0 && (
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
