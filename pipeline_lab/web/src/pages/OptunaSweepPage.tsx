import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowLeft,
  BarChart3,
  ChevronRight,
  Film,
  Loader2,
  RefreshCw,
  Sparkles,
  VideoOff,
} from 'lucide-react'
import { API } from '../types'

type OptunaSweepMeta = {
  config?: string
  sequence_order?: string[]
  log_jsonl?: string
  storage?: string
  git_sha?: string
  hint?: string
}

type OptunaTrialRow = {
  number: number
  state: string
  value: number | null
  params: Record<string, unknown>
  user_attrs: Record<string, unknown>
}

type OptunaBest = {
  number: number
  value: number
  params: Record<string, unknown>
  user_attrs: Record<string, unknown>
}

export type OptunaSweepStatus = {
  schema?: string
  /** Present when ``sweep_status.json`` is missing (API returns 200). */
  empty_state?: boolean
  updated_unix: number
  study_name: string
  direction: string
  n_trials_total: number
  n_complete: number
  n_pruned: number
  n_other: number
  best: OptunaBest | null
  trials: OptunaTrialRow[]
  meta?: OptunaSweepMeta
}

type MediaItem = { kind: string; filename: string; path: string }

type MediaResponse = {
  trial: number
  sequence: string
  items: MediaItem[]
  note?: string
}

function pullLambdaUrl() {
  return `${API}/api/optuna-sweep/pull-lambda`
}

function statusUrl() {
  return `${API}/api/optuna-sweep/status`
}

function mediaUrl(trial: number, sequence: string) {
  return `${API}/api/optuna-sweep/trial/${trial}/sequence/${encodeURIComponent(sequence)}/media`
}

function fileUrl(trial: number, sequence: string, relPath: string) {
  const q = new URLSearchParams({ path: relPath })
  return `${API}/api/optuna-sweep/trial/${trial}/sequence/${encodeURIComponent(sequence)}/file?${q}`
}

function formatTime(ts: number) {
  try {
    return new Date(ts * 1000).toLocaleString()
  } catch {
    return '—'
  }
}

function formatValue(v: number | null | undefined) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return v.toFixed(4)
}

function ParamsTable({ params }: { params: Record<string, unknown> }) {
  const keys = Object.keys(params).sort()
  if (!keys.length) return <p className="optuna-muted">No parameters</p>
  return (
    <table className="optuna-kv">
      <tbody>
        {keys.map((k) => (
          <tr key={k}>
            <th>{k}</th>
            <td>{String(params[k])}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function formatHttpDetail(res: Response, body: unknown): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const d = (body as { detail: unknown }).detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) return d.map((x) => JSON.stringify(x)).join('; ')
  }
  return res.statusText
}

export function OptunaSweepPage() {
  const [data, setData] = useState<OptunaSweepStatus | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedTrial, setSelectedTrial] = useState<number | null>(null)
  const [sequence, setSequence] = useState<string>('')
  const [media, setMedia] = useState<MediaResponse | null>(null)
  const [mediaLoading, setMediaLoading] = useState(false)
  const [pickedClip, setPickedClip] = useState<string | null>(null)

  /**
   * 1) POST pull-lambda (scp from the Lab API’s configured Lambda → local sweep_status.json).
   * 2) GET status. Does **not** run ``auto_sweep`` on this machine.
   */
  const sync = useCallback(async () => {
    setLoading(true)
    setErr(null)
    let reachedStatusFetch = false
    try {
      const pr = await fetch(pullLambdaUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      const pj = await pr.json().catch(() => null)
      if (!pr.ok) {
        throw new Error(formatHttpDetail(pr, pj))
      }

      reachedStatusFetch = true
      const sr = await fetch(statusUrl())
      const t = await sr.text()
      if (!sr.ok) {
        throw new Error(t || sr.statusText)
      }
      setData(JSON.parse(t) as OptunaSweepStatus)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      if (reachedStatusFetch) {
        setData(null)
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    sync()
  }, [sync])

  useEffect(() => {
    const order = data?.meta?.sequence_order
    if (order?.length && !sequence) {
      setSequence(order[0])
    }
  }, [data, sequence])

  useEffect(() => {
    const id = window.setInterval(() => sync(), 8000)
    return () => window.clearInterval(id)
  }, [sync])

  useEffect(() => {
    if (selectedTrial === null || !sequence) {
      setMedia(null)
      setPickedClip(null)
      return
    }
    let cancelled = false
    ;(async () => {
      setMediaLoading(true)
      try {
        const r = await fetch(mediaUrl(selectedTrial, sequence))
        const j = (await r.json()) as MediaResponse
        if (!cancelled) {
          setMedia(j)
          const preferred =
            j.items.find((x) => x.filename.includes('01_tracks_post_stitch')) ||
            j.items.find((x) => x.kind === 'phase_preview') ||
            j.items[0] ||
            null
          setPickedClip(preferred ? preferred.path : null)
        }
      } catch {
        if (!cancelled) {
          setMedia(null)
          setPickedClip(null)
        }
      } finally {
        if (!cancelled) setMediaLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [selectedTrial, sequence])

  const sequences = data?.meta?.sequence_order ?? []

  const sortedTrials = useMemo(() => {
    const t = data?.trials ?? []
    return [...t].sort((a, b) => b.number - a.number)
  }, [data])

  const runningHint = useMemo(() => {
    const incomplete = sortedTrials.filter((x) => x.state === 'RUNNING')
    if (incomplete.length) return `Trial ${incomplete[0].number} in progress…`
    return null
  }, [sortedTrials])

  return (
    <div className="optuna-sweep-page">
      <div className="optuna-sweep-head">
        <Link to="/" className="optuna-back">
          <ArrowLeft size={18} aria-hidden />
          Lab
        </Link>
        <div className="optuna-sweep-titleblock">
          <h1 className="optuna-sweep-title">
            <BarChart3 size={26} className="optuna-sweep-title-ico" aria-hidden />
            Optuna sweep
          </h1>
          <p className="optuna-sweep-sub">
            <strong>Refresh</strong> pulls <code>sweep_status.json</code> from your gpu_1x_a10 box (
            <code>150.136.111.175</code>, key <code>pose-tracking</code>) via <code>scp</code>, then reloads the table.
            Sweep runs on Lambda only. Change the IP/key in <code>pipeline_lab/server/app.py</code> or use{' '}
            <code>SWAY_LAMBDA_SWEEP_*</code> env vars; set <code>SWAY_LAMBDA_SWEEP_HOST=</code> to disable pulls.
          </p>
        </div>
        <button type="button" className="btn primary optuna-refresh" onClick={() => sync()} disabled={loading}>
          {loading ? <Loader2 className="optuna-spin" size={18} /> : <RefreshCw size={18} />}
          Refresh
        </button>
      </div>

      {err ? (
        <div className="optuna-error-card">
          <h2>Could not refresh</h2>
          <p>{err}</p>
          <p className="optuna-muted">
            If <strong>scp</strong> failed, check that <code>pose-tracking.pem</code> exists under{' '}
            <code>~/Downloads</code> or <code>~/.ssh</code>, PEM permissions, and that the instance is running. If the
            file is not on the server yet, run the sweep on Lambda with <code>python -m tools.auto_sweep</code>. To read
            only a local JSON, set <code>SWAY_LAMBDA_SWEEP_HOST=</code> when starting the Lab API.
          </p>
        </div>
      ) : null}

      {data?.empty_state ? (
        <div className="optuna-info-card" role="status">
          <h2>No sweep data (yet)</h2>
          {data.meta?.hint ? <p>{data.meta.hint}</p> : null}
          <p className="optuna-muted">
            Hit <strong>Refresh</strong> to copy the latest snapshot from Lambda, or place <code>sweep_status.json</code>{' '}
            under <code>output/sweeps/optuna/</code> (or set <code>SWAY_OPTUNA_SWEEP_DIR</code>). Trial preview MP4s
            only play if those files exist locally (sync <code>trial_*</code> dirs separately if needed).
          </p>
        </div>
      ) : null}

      {data && !data.empty_state ? (
        <>
          <section className="optuna-summary-grid">
            <div className="optuna-stat-card">
              <span className="optuna-stat-label">Study</span>
              <span className="optuna-stat-value">{data.study_name}</span>
              <span className="optuna-stat-hint">{data.direction}</span>
            </div>
            <div className="optuna-stat-card">
              <span className="optuna-stat-label">Updated</span>
              <span className="optuna-stat-value optuna-stat-value--sm">{formatTime(data.updated_unix)}</span>
              {data.meta?.git_sha ? (
                <span className="optuna-stat-hint mono">git {data.meta.git_sha}</span>
              ) : null}
            </div>
            <div className="optuna-stat-card">
              <span className="optuna-stat-label">Progress</span>
              <span className="optuna-stat-value">
                {data.n_complete} done · {data.n_pruned} pruned · {data.n_trials_total} total
              </span>
              {runningHint ? <span className="optuna-stat-hint optuna-pulse">{runningHint}</span> : null}
            </div>
          </section>

          {data.best ? (
            <section className="optuna-best-card">
              <div className="optuna-best-head">
                <Sparkles size={22} className="optuna-best-ico" aria-hidden />
                <div>
                  <h2>Best trial</h2>
                  <p className="optuna-muted">
                    Trial <strong>#{data.best.number}</strong> · aggregate score{' '}
                    <strong>{formatValue(data.best.value)}</strong>
                  </p>
                </div>
                <button
                  type="button"
                  className="btn btn--compact"
                  onClick={() => setSelectedTrial(data.best!.number)}
                >
                  Open video
                  <ChevronRight size={16} aria-hidden />
                </button>
              </div>
              <div className="optuna-best-grid">
                <div>
                  <h3 className="optuna-h3">Per-video (user attrs)</h3>
                  <ParamsTable
                    params={Object.fromEntries(
                      Object.entries(data.best.user_attrs).filter(([k]) => k.startsWith('score_')),
                    )}
                  />
                </div>
                <div>
                  <h3 className="optuna-h3">Suggested params</h3>
                  <ParamsTable params={data.best.params} />
                </div>
              </div>
            </section>
          ) : (
            <section className="optuna-empty-best">
              <p className="optuna-muted">No completed trials yet — best will appear here.</p>
            </section>
          )}

          <section className="optuna-split">
            <div className="optuna-table-panel">
              <h2 className="optuna-h2">All trials</h2>
              <div className="optuna-table-wrap">
                <table className="optuna-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>State</th>
                      <th>Value</th>
                      <th>Aggregate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedTrials.map((t) => (
                      <tr
                        key={t.number}
                        className={selectedTrial === t.number ? 'optuna-row--active' : undefined}
                        onClick={() => setSelectedTrial(t.number)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault()
                            setSelectedTrial(t.number)
                          }
                        }}
                        tabIndex={0}
                        role="button"
                      >
                        <td className="mono">{t.number}</td>
                        <td>
                          <span className={`optuna-pill optuna-pill--${t.state.toLowerCase()}`}>{t.state}</span>
                        </td>
                        <td>{formatValue(t.value)}</td>
                        <td className="optuna-muted">
                          {typeof t.user_attrs?.aggregate_harmonic_mean === 'number'
                            ? formatValue(t.user_attrs.aggregate_harmonic_mean as number)
                            : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="optuna-video-panel">
              <h2 className="optuna-h2">
                <Film size={20} aria-hidden style={{ verticalAlign: 'middle', marginRight: 8 }} />
                Video proof
              </h2>
              <p className="optuna-muted optuna-video-help">
                Select a trial row, then a benchmark sequence. Clips are read from <strong>this Mac&apos;s</strong> sweep
                tree (same as JSON path). Remote-only previews require rsync/scp of <code>trial_*</code> dirs, or run
                previews locally. Requires <code>--phase-previews</code> on the sweep that produced the trial.
              </p>

              <div className="optuna-video-controls">
                <label className="optuna-field">
                  <span>Sequence</span>
                  <select
                    className="optuna-select"
                    value={sequence}
                    onChange={(e) => setSequence(e.target.value)}
                    disabled={!sequences.length}
                  >
                    {sequences.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="optuna-field">
                  <span>Trial</span>
                  <input
                    className="optuna-input mono"
                    type="number"
                    min={0}
                    value={selectedTrial ?? ''}
                    placeholder="#"
                    onChange={(e) => {
                      const v = e.target.value
                      setSelectedTrial(v === '' ? null : parseInt(v, 10))
                    }}
                  />
                </label>
              </div>

              {mediaLoading ? (
                <div className="optuna-video-loading">
                  <Loader2 className="optuna-spin" size={28} />
                  <span>Loading clips…</span>
                </div>
              ) : null}

              {selectedTrial !== null && media && !media.items.length ? (
                <div className="optuna-video-empty">
                  <VideoOff size={36} aria-hidden />
                  <p>No MP4s for trial {selectedTrial} / {sequence}.</p>
                  {media.note ? <p className="optuna-muted">{media.note}</p> : null}
                </div>
              ) : null}

              {selectedTrial !== null && media && media.items.length > 0 ? (
                <>
                  <label className="optuna-field optuna-field--full">
                    <span>Clip</span>
                    <select
                      className="optuna-select"
                      value={pickedClip ?? ''}
                      onChange={(e) => setPickedClip(e.target.value || null)}
                    >
                      {media.items.map((it) => (
                        <option key={it.path} value={it.path}>
                          [{it.kind}] {it.filename}
                        </option>
                      ))}
                    </select>
                  </label>
                  {pickedClip ? (
                    <video
                      className="optuna-video"
                      key={`${selectedTrial}-${sequence}-${pickedClip}`}
                      controls
                      playsInline
                      preload="metadata"
                      src={fileUrl(selectedTrial, sequence, pickedClip)}
                    />
                  ) : null}
                </>
              ) : null}

              {selectedTrial === null ? (
                <div className="optuna-video-placeholder">
                  <p className="optuna-muted">Click a trial in the table to load previews.</p>
                </div>
              ) : null}
            </div>
          </section>
        </>
      ) : null}
    </div>
  )
}
