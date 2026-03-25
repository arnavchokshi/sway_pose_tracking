import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { API } from '../types'
import type { Schema } from '../types'
import { Pause, Play, SkipBack, SkipForward } from 'lucide-react'
import { WATCH_PHASE_ROWS } from '../lib/watchPrune'
import { TrackQualitySummary, PipelineImpactSummary, PipelineImpactReport } from '../components/RunMetrics'
import { formatConfigValue } from '../lib/formatConfigValue'

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

function useSyncedVideos() {
  const refs = useRef<(HTMLVideoElement | null)[]>([])
  const scrubbing = useRef(false)
  const [playing, setPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [current, setCurrent] = useState(0)

  const syncTime = useCallback(
    (t: number) => {
      const max = duration > 0 ? duration : Number.POSITIVE_INFINITY
      const clamped = Math.max(0, Math.min(t, max))
      refs.current.forEach((el) => {
        if (!el) return
        if (Number.isFinite(el.duration) && el.duration > 0) {
          const cap = el.duration
          el.currentTime = Math.min(clamped, cap - 1e-3)
        } else {
          el.currentTime = clamped
        }
      })
      setCurrent(clamped)
    },
    [duration],
  )

  const onMeta = useCallback(() => {
    const durs = refs.current.map((el) => (el && el.duration && Number.isFinite(el.duration) ? el.duration : 0))
    const m = Math.max(...durs, 0)
    if (m > 0) setDuration(m)
  }, [])

  const togglePlay = useCallback(() => {
    const next = !playing
    setPlaying(next)
    refs.current.forEach((el) => {
      if (!el) return
      if (next) void el.play().catch(() => setPlaying(false))
      else el.pause()
    })
  }, [playing])

  const onTimeUpdateMaster = useCallback(
    (idx: number) => {
      if (scrubbing.current) return
      const master = refs.current[idx]
      if (!master) return
      const t = master.currentTime
      setCurrent(t)
      refs.current.forEach((el, i) => {
        if (!el || i === idx) return
        if (Math.abs(el.currentTime - t) > 0.08) el.currentTime = t
      })
    },
    [],
  )

  return {
    refs,
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

  const [slots, setSlots] = useState<Slot[]>([])
  const [viewMode, setViewMode] = useState<string>('final')
  const [labSchema, setLabSchema] = useState<Schema | null>(null)

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
        fetch(`${API}/api/runs/${id}`)
          .then(async (r) => {
            if (!r.ok) throw new Error(String(r.status))
            return r.json() as Promise<RunDetail>
          })
          .then((data) => {
            let rel = data.manifest?.view_variants?.full ?? data.manifest?.final_video_relpath
            if (viewMode !== 'final') {
              const phaseRow = WATCH_PHASE_ROWS.find(r => r.id === viewMode)
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
            return {
              run_id: id,
              label: data.recipe_name || id.slice(0, 8),
              src,
              error,
              fields: data.manifest?.run_context_final?.fields,
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

  const { refs, scrubbing, playing, duration, current, syncTime, onMeta, togglePlay, onTimeUpdateMaster, setPlaying } =
    useSyncedVideos()

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

  const ready = slots.length > 0 && slots.every((s) => s.src)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div 
        className="glass-panel" 
        style={{ 
          padding: '1rem 1.5rem', 
          position: 'sticky', 
          top: '4.5rem', 
          zIndex: 40, 
          background: 'rgba(20, 24, 34, 0.85)' 
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem' }}>
          <div>
            <h1 style={{ fontSize: '1.5rem', margin: 0 }}>Compare</h1>
            <p className="sub" style={{ margin: 0, fontSize: '0.9rem' }}>
              One playhead scrubs and plays every output together.
            </p>
            <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>View:</span>
              <select
                value={viewMode}
                onChange={(e) => setViewMode(e.target.value)}
                style={{
                  background: 'rgba(0,0,0,0.4)',
                  color: '#fff',
                  border: '1px solid var(--glass-border)',
                  borderRadius: 6,
                  padding: '0.35rem 0.5rem',
                  fontSize: '0.85rem',
                  outline: 'none',
                }}
              >
                <option value="final">Final render (Full)</option>
                {WATCH_PHASE_ROWS.map(r => (
                  <option key={r.id} value={r.id}>Phase: {r.title}</option>
                ))}
              </select>
            </div>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', flex: 1, minWidth: 320 }}>
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
        {!ready && (
          <p style={{ margin: '0.75rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Waiting for all runs to finish and produce output video…
          </p>
        )}
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))',
          gap: '1rem',
        }}
      >
        {slots.map((s, i) => (
          <div key={s.run_id} className="glass-panel" style={{ overflow: 'hidden', padding: 0 }}>
            <div
              style={{
                padding: '0.75rem 1rem',
                borderBottom: '1px solid var(--glass-border)',
                fontWeight: 600,
                color: '#fff',
              }}
            >
              {s.label}
            </div>
            <div style={{ background: '#000', aspectRatio: '16/9' }}>
              {s.error && (
                <div style={{ color: '#f87171', padding: '1.5rem', fontSize: '0.9rem' }}>{s.error}</div>
              )}
              {s.src && (
                <video
                  ref={(el) => {
                    refs.current[i] = el
                  }}
                  src={s.src}
                  playsInline
                  onLoadedMetadata={onMeta}
                  onPlay={() => setPlaying(true)}
                  onPause={() => {
                    const anyPlaying = refs.current.some((el) => el && !el.paused)
                    if (!anyPlaying) setPlaying(false)
                  }}
                  muted
                  onTimeUpdate={() => {
                    if (i === 0) onTimeUpdateMaster(0)
                  }}
                  style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
                />
              )}
            </div>
          </div>
        ))}
      </div>

      {slots.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h2 style={{ fontSize: '1.25rem', marginBottom: '0.35rem', color: '#fff' }}>Tracking & pipeline impact</h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', margin: '0 0 1rem', lineHeight: 1.5, maxWidth: 720 }}>
            Heuristic track cards plus manifest diagnostics so you can see how hybrid SAM, interpolation, and global stitch differed
            run-to-run — not just the final MP4.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${slots.length}, minmax(320px, 1fr))`, gap: '1rem' }}>
            {slots.map((s) => (
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

          <h2 style={{ fontSize: '1.25rem', margin: '2.5rem 0 0.35rem', color: '#fff' }}>Configuration differences</h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', margin: '0 0 1rem', lineHeight: 1.5, maxWidth: 720 }}>
            Only keys that differ between runs. Labels match the Lab when the schema is loaded; technical id shown below the label.
          </p>
          <div className="glass-panel" style={{ padding: '1.25rem', overflowX: 'auto' }}>
            {(() => {
              const allKeys = Array.from(new Set(slots.flatMap((s) => Object.keys(s.fields || {}))))
              const diffKeys = allKeys
                .filter((k) => {
                  const vals = new Set(slots.map((s) => JSON.stringify((s.fields || {})[k])))
                  return vals.size > 1
                })
                .sort()

              const fieldLabel = (id: string) => labSchema?.fields.find((f) => f.id === id)?.label

              if (diffKeys.length === 0) {
                return (
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                    No configuration differences found between these runs.
                  </div>
                )
              }

              return (
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
                      {slots.map((s) => (
                        <th
                          key={s.run_id}
                          style={{ padding: '0.75rem', borderBottom: '1px solid var(--glass-border)', color: '#f8fafc' }}
                        >
                          {s.label}
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
                        {slots.map((s) => {
                          const val = (s.fields || {})[k]
                          return (
                            <td key={s.run_id} style={{ padding: '0.75rem', color: '#fff', verticalAlign: 'top', lineHeight: 1.45 }}>
                              {val !== undefined ? formatConfigValue(val) : <span style={{ color: '#ef4444' }}>Missing</span>}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )
            })()}
          </div>
        </div>
      )}
    </div>
  )
}

function formatTime(s: number) {
  if (!Number.isFinite(s) || s < 0) return '0:00'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}
