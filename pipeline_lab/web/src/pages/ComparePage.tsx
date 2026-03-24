import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { API } from '../types'
import { Pause, Play, SkipBack, SkipForward } from 'lucide-react'

type RunDetail = {
  run_id: string
  recipe_name?: string
  status?: string
  manifest?: { final_video_relpath?: string; view_variants?: Record<string, string> }
}

type Slot = {
  run_id: string
  label: string
  src: string | null
  error: string | null
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
            const rel = data.manifest?.view_variants?.full ?? data.manifest?.final_video_relpath
            const src = rel ? `${API}/api/runs/${id}/file/output/${rel}` : null
            let error: string | null = null
            if (!rel) {
              error =
                data.status === 'done'
                  ? 'No output file in manifest.'
                  : 'Run not finished yet.'
            }
            return {
              run_id: id,
              label: data.recipe_name || id.slice(0, 8),
              src,
              error,
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
  }, [raw])

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
      <div className="glass-panel" style={{ padding: '1.25rem 1.5rem' }}>
        <h1 style={{ fontSize: '1.75rem', margin: 0 }}>Compare</h1>
        <p className="sub" style={{ marginBottom: 0 }}>
          One playhead scrubs and plays every output together.
        </p>
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

      <div className="glass-panel" style={{ padding: '1rem 1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button type="button" className="btn" onClick={() => syncTime(0)} aria-label="Seek start">
            <SkipBack size={18} />
          </button>
          <button type="button" className="btn primary" onClick={togglePlay} disabled={!ready}>
            {playing ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => syncTime(duration - 0.05)}
            disabled={duration <= 0}
            aria-label="Seek end"
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
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
            {formatTime(current)} / {formatTime(duration)}
          </span>
        </div>
        {!ready && (
          <p style={{ margin: '0.75rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Waiting for all runs to finish and produce output video…
          </p>
        )}
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
