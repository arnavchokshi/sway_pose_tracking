import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentType,
  type CSSProperties,
} from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { API } from '../types'
import type { Schema, RunInfo, ProgressLine } from '../types'
import { safeJsonPreview } from '../lib/jsonPreview'
import { isProbableVideoFile, VIDEO_ACCEPT_ATTR } from '../lib/videoFile'
import { useLab } from '../context/LabContext'
import { RunEditorModal } from '../components/RunEditorModal'
import { RunConfigModal } from '../components/RunConfigModal'
import { TrackQualitySummary } from '../components/RunMetrics'
import {
  Play,
  Plus,
  Pencil,
  Copy,
  Trash2,
  Clapperboard,
  Columns2,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  CircleDashed,
  Terminal,
  Settings2,
  StopCircle,
  Unplug,
  RotateCw,
  Upload,
} from 'lucide-react'
function formatShortId(runId: string) {
  return runId.length > 12 ? `${runId.slice(0, 8)}…` : runId
}

function formatCreated(iso?: string) {
  if (!iso) return null
  try {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return null
    return d.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
  } catch {
    return null
  }
}

function statusPresentation(status: string): {
  title: string
  subtitle: string
  color: string
  Icon: ComponentType<{ size?: number; className?: string; style?: CSSProperties }>
} {
  switch (status) {
    case 'done':
      return {
        title: 'Complete',
        subtitle: 'Outputs are ready — open Watch for phase clips and render styles.',
        color: '#10b981',
        Icon: CheckCircle2,
      }
    case 'running':
      return {
        title: 'Running',
        subtitle: 'Pipeline is executing (detection → pose → export). This can take a while.',
        color: '#0ea5e9',
        Icon: Loader2,
      }
    case 'queued':
      return {
        title: 'Queued',
        subtitle: 'Waiting to start — runs start one after another when workers are free.',
        color: '#a78bfa',
        Icon: Clock,
      }
    case 'error':
      return {
        title: 'Failed',
        subtitle: 'This run stopped with an error. Check the message below or server logs.',
        color: '#ef4444',
        Icon: XCircle,
      }
    case 'cancelled':
      return {
        title: 'Stopped',
        subtitle: 'You stopped this run from the Lab. Partial outputs may exist under the run folder.',
        color: '#fbbf24',
        Icon: StopCircle,
      }
    default:
      return {
        title: status || 'Unknown',
        subtitle: 'Status not reported yet — try refreshing in a few seconds.',
        color: '#94a3b8',
        Icon: CircleDashed,
      }
  }
}

function statusPresentationForRun(run: Pick<RunInfo, 'status' | 'subprocess_alive'>): ReturnType<typeof statusPresentation> {
  if (run.status === 'running' && run.subprocess_alive === false) {
    return {
      title: 'Stale (not attached)',
      subtitle:
        'The run folder still looks in-progress, but this API process is not running main.py for it (common after restarting uvicorn). Delete the run or start a new batch — Stop is unavailable.',
      color: '#f59e0b',
      Icon: Unplug,
    }
  }
  return statusPresentation(run.status)
}

function isCompletedProgressRow(p: ProgressLine) {
  return !p.status || p.status === 'done'
}

function isRunningProgressRow(p: ProgressLine) {
  return p.status === 'running'
}

function buildPhaseTimeline(lines: ProgressLine[]) {
  const completed = lines.filter(isCompletedProgressRow)
  const byKey = new Map<string, ProgressLine>()
  for (const row of completed) {
    const key = row.stage_key ?? `stage_${row.stage}`
    byKey.set(key, row)
  }
  return Array.from(byKey.values()).sort((a, b) => a.stage - b.stage)
}

function latestRunningRow(lines: ProgressLine[]) {
  for (let i = lines.length - 1; i >= 0; i--) {
    if (isRunningProgressRow(lines[i])) return lines[i]
  }
  return null
}


function RunPipelineDetail({
  runStatus,
  subprocessAlive,
  progressLines,
  logLines,
  trackSummary,
}: {
  runStatus: string
  /** When status is running: false = disk stale / API restarted (not actively executing here). */
  subprocessAlive?: boolean
  progressLines: ProgressLine[]
  logLines: string[]
  trackSummary?: Record<string, unknown> | null
}) {
  const timeline = useMemo(() => buildPhaseTimeline(progressLines), [progressLines])
  const live = latestRunningRow(progressLines)
  const ex = live?.extra as Record<string, unknown> | undefined
  const pct = useMemo(() => {
    if (typeof ex?.pct === 'number' && Number.isFinite(ex.pct)) return Math.round(ex.pct)
    const f = ex?.frame
    const tf = ex?.total_frames
    if (typeof f === 'number' && typeof tf === 'number' && tf > 0 && Number.isFinite(f)) {
      return Math.min(100, Math.max(0, Math.round((100 * f) / tf)))
    }
    return null
  }, [ex])
  const lastDone = useMemo(() => {
    const done = progressLines.filter(isCompletedProgressRow)
    return done.length ? done[done.length - 1] : null
  }, [progressLines])

  if (runStatus === 'queued') {
    return (
      <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Waiting in the job queue — no pipeline output yet. This run starts when earlier runs finish.
      </div>
    )
  }

  const staleRunning = runStatus === 'running' && subprocessAlive === false

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <div>
        <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
          Pipeline phases (from progress.jsonl)
        </div>
        {timeline.length === 0 ? (
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
            {staleRunning ? (
              <>
                No new progress will appear here until a <strong style={{ color: '#e2e8f0' }}>live</strong> pipeline is
                attached to this API. Use <strong style={{ color: '#fcd34d' }}>Delete</strong> to remove this folder, then
                start runs again if needed. If <code>main.py</code> is still using CPU from an old session, quit it from
                Activity Monitor.
              </>
            ) : runStatus === 'running' ? (
              <>
                No progress checkpoints yet — Phases 1–2 (YOLO + tracking) often run for minutes before the first line
                is written to <code style={{ color: 'var(--halo-cyan)' }}>progress.jsonl</code>. On{' '}
                <strong style={{ color: '#e2e8f0' }}>CPU</strong>, the first YOLO pass can sit with no new lines for a long
                time (normal). The console streams <code style={{ color: 'var(--halo-cyan)' }}>stdout.log</code> live;
                newer server builds log periodic YOLO progress there.
              </>
            ) : (
              'No phase checkpoints yet — tracking will appear here shortly.'
            )}
          </div>
        ) : (
          <ol style={{ margin: 0, paddingLeft: '1.1rem', fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
            {timeline.map((row, ti) => {
              const phaseMeta: Record<string, unknown> = {
                ...(row.meta ?? {}),
                ...(row.extra ?? {}),
              }
              const hasMeta = Object.keys(phaseMeta).length > 0
              return (
                <li key={`${row.stage}-${row.stage_key ?? ''}-${row.elapsed_s ?? ti}`} style={{ marginBottom: '0.45rem' }}>
                  <span style={{ color: '#e2e8f0' }}>{row.label ?? row.stage_key ?? `Stage ${row.stage}`}</span>
                  {row.elapsed_s != null && (
                    <span style={{ opacity: 0.85 }}> · {row.elapsed_s.toFixed(1)}s in phase</span>
                  )}
                  {row.cumulative_elapsed_s != null && (
                    <span style={{ opacity: 0.75 }}> · {row.cumulative_elapsed_s.toFixed(0)}s wall total</span>
                  )}
                  {hasMeta && (
                    <details style={{ marginTop: '0.3rem', marginLeft: '0.05rem' }}>
                      <summary
                        style={{
                          cursor: 'pointer',
                          color: 'var(--halo-cyan)',
                          fontSize: '0.7rem',
                          fontWeight: 600,
                          listStyle: 'none',
                        }}
                      >
                        Phase metadata (params, counts, prune sample)
                      </summary>
                      <pre
                        style={{
                          marginTop: '0.35rem',
                          maxHeight: 280,
                          overflow: 'auto',
                          padding: '0.5rem 0.6rem',
                          borderRadius: 8,
                          background: 'rgba(0,0,0,0.4)',
                          border: '1px solid var(--glass-border)',
                          color: '#cbd5e1',
                          fontSize: '0.65rem',
                          lineHeight: 1.35,
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
                        {safeJsonPreview(phaseMeta, 24000)}
                      </pre>
                    </details>
                  )}
                </li>
              )
            })}
          </ol>
        )}
      </div>

      {runStatus === 'done' && (
        <div style={{ fontSize: '0.8rem', color: '#6ee7b7', lineHeight: 1.45 }}>
          Pipeline finished ({timeline.length} phase checkpoints). Use Watch for videos and phase clips.
          {trackSummary && Object.keys(trackSummary).length > 0 && <TrackQualitySummary summary={trackSummary} />}
        </div>
      )}
      {runStatus === 'error' && (
        <div style={{ fontSize: '0.8rem', color: '#fca5a5', lineHeight: 1.45 }}>
          Run stopped before completion — check the console below for Python stderr/stdout near the failure.
        </div>
      )}
      {runStatus === 'cancelled' && (
        <div style={{ fontSize: '0.8rem', color: '#fcd34d', lineHeight: 1.45 }}>
          Stopped from the Lab — partial logs and any files already written stay in the run folder. You can delete the run
          to clean up.
        </div>
      )}

      {staleRunning && (
        <div
          style={{
            padding: '0.65rem 0.75rem',
            borderRadius: 10,
            background: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid rgba(245, 158, 11, 0.35)',
          }}
        >
          <div
            style={{
              fontSize: '0.72rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: '#fbbf24',
              marginBottom: '0.35rem',
            }}
          >
            Not active on this server
          </div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
            Batch summary counts this as <strong style={{ color: '#fcd34d' }}>stale</strong>, not an actively running job.
            The console below may show old <code>stdout.log</code> lines only.
          </div>
        </div>
      )}

      {runStatus === 'running' && !staleRunning && (
        <div
          style={{
            padding: '0.65rem 0.75rem',
            borderRadius: 10,
            background: 'rgba(14, 165, 233, 0.1)',
            border: '1px solid rgba(14, 165, 233, 0.28)',
          }}
        >
          <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#7dd3fc', marginBottom: '0.35rem' }}>
            Current activity
          </div>
          {live ? (
            <>
              <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#fff' }}>{live.label ?? live.stage_key}</div>
              {pct != null && (
                <div style={{ marginTop: '0.45rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>
                    <span>Phase progress</span>
                    <span>{pct}%</span>
                  </div>
                  <div style={{ height: 6, borderRadius: 4, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                    <div style={{ width: `${Math.min(100, Math.max(0, pct))}%`, height: '100%', background: 'linear-gradient(90deg, #0ea5e9, #06b6d4)', transition: 'width 0.3s ease' }} />
                  </div>
                </div>
              )}
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '0.45rem', lineHeight: 1.45 }}>
                {(typeof ex?.frame === 'number' ||
                  typeof ex?.yolo_passes === 'number' ||
                  typeof ex?.pose_pass === 'number' ||
                  typeof ex?.tracks === 'number') && (
                  <div>
                    {typeof ex.frame === 'number' && (
                      <>
                        Frame {ex.frame}
                        {typeof ex.total_frames === 'number' ? ` / ${ex.total_frames}` : ''}
                      </>
                    )}
                    {typeof ex.yolo_passes === 'number' && ` · yolo passes ${ex.yolo_passes}`}
                    {typeof ex.pose_pass === 'number' && ` · pose passes ${ex.pose_pass}`}
                    {typeof ex.tracks === 'number' && ` · tracks ${ex.tracks}`}
                  </div>
                )}
                {typeof ex?.step === 'string' && ex.step.trim() !== '' && (
                  <div style={{ marginTop: '0.2rem' }}>Step: {ex.step}</div>
                )}
                {typeof ex?.wall_s_in_phase === 'number' && <div>Time in this phase: {ex.wall_s_in_phase}s</div>}
                {typeof ex?.frame !== 'number' &&
                  typeof ex?.yolo_passes !== 'number' &&
                  typeof ex?.pose_pass !== 'number' &&
                  pct == null &&
                  !(typeof ex?.step === 'string' && ex.step.trim() !== '') &&
                  typeof ex?.wall_s_in_phase !== 'number' && (
                    <div>Live heartbeat for this phase — open the console below for full main.py logs.</div>
                  )}
              </div>
            </>
          ) : (
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
              Between checkpoints or starting up… Last finished:{' '}
              <strong style={{ color: '#e2e8f0' }}>{lastDone?.label ?? '—'}</strong>. Detailed lines appear in the console.
            </div>
          )}
        </div>
      )}

      <details style={{ fontSize: '0.78rem' }}>
        <summary
          style={{
            cursor: 'pointer',
            color: 'var(--halo-cyan)',
            fontWeight: 600,
            listStyle: 'none',
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
          }}
        >
          <Terminal size={14} /> Pipeline console (same as main.py stdout)
        </summary>
        <pre
          style={{
            marginTop: '0.5rem',
            maxHeight: 220,
            overflow: 'auto',
            padding: '0.65rem 0.75rem',
            borderRadius: 8,
            background: 'rgba(0,0,0,0.45)',
            border: '1px solid var(--glass-border)',
            color: '#cbd5e1',
            fontSize: '0.68rem',
            lineHeight: 1.4,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {logLines.length > 0 ? logLines.join('\n') : 'No log lines yet (stdout.log will fill as the run executes).'}
        </pre>
      </details>
    </div>
  )
}

export function LabPage() {
  const nav = useNavigate()
  const location = useLocation()
  const {
    videoFile,
    videoLabel,
    setVideo,
    drafts,
    addDraft,
    removeDraft,
    duplicateDraft,
    updateDraft,
    sessionRunIds,
    setSessionRunIds,
    clearDrafts,
    labHydrated,
  } = useLab()

  const [schema, setSchema] = useState<Schema | null>(null)
  const [schemaError, setSchemaError] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)

  const [batchError, setBatchError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [configRunId, setConfigRunId] = useState<string | null>(null)

  const [runRows, setRunRows] = useState<RunInfo[]>([])
  const [progressByRun, setProgressByRun] = useState<Record<string, ProgressLine[]>>({})
  const [logByRun, setLogByRun] = useState<Record<string, string[]>>({})
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null)
  const [stoppingRunId, setStoppingRunId] = useState<string | null>(null)
  const [rerunningRunId, setRerunningRunId] = useState<string | null>(null)
  const dropDepthRef = useRef(0)
  const [dropActive, setDropActive] = useState(false)

  const applyVideoFile = useCallback(
    (file?: File) => {
      if (!file) return
      if (!isProbableVideoFile(file)) {
        window.alert('Please use a video file (MP4, MOV, WebM, MKV, AVI, or M4V).')
        return
      }
      setVideo(file)
    },
    [setVideo],
  )

  const editingDraft = useMemo(
    () => drafts.find((d) => d.clientId === editingId) ?? null,
    [drafts, editingId],
  )

  useEffect(() => {
    let cancelled = false
    setSchemaError(null)
    fetch(`${API}/api/schema`)
      .then(async (r) => {
        const text = await r.text()
        if (!r.ok) throw new Error(`Schema HTTP ${r.status}`)
        return JSON.parse(text) as Schema
      })
      .then((s) => {
        if (!cancelled) setSchema(s)
      })
      .catch((e) => {
        if (!cancelled) setSchemaError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!labHydrated || !videoFile || !schema || drafts.length > 0 || sessionRunIds.length > 0) return
    addDraft()
  }, [labHydrated, videoFile, schema, drafts.length, sessionRunIds.length, addDraft])

  useEffect(() => {
    const st = location.state as { appendSessionRunIds?: string[] } | null | undefined
    const extra = st?.appendSessionRunIds
    if (!extra?.length) return
    setSessionRunIds((prev) => {
      const seen = new Set(prev)
      const out = [...prev]
      for (const rid of extra) {
        if (!seen.has(rid)) {
          seen.add(rid)
          out.push(rid)
        }
      }
      return out
    })
    nav('.', { replace: true, state: {} })
  }, [location.state, nav, setSessionRunIds])

  const refreshSessionRuns = useCallback(() => {
    if (sessionRunIds.length === 0) return
    Promise.all(
      sessionRunIds.map((id) =>
        Promise.all([
          fetch(`${API}/api/runs/${id}`)
            .then((r) => {
              if (r.status === 404) return { __missing: true }
              return r.ok ? r.json() : null
            })
            .catch(() => null),
          fetch(`${API}/api/runs/${id}/progress`)
            .then((r) => (r.ok ? r.json() : []))
            .catch(() => []),
          fetch(`${API}/api/runs/${id}/log?lines=220`)
            .then((r) => (r.ok ? r.json() : { lines: [] }))
            .catch(() => ({ lines: [] })),
        ]).then(([row, prog, logPayload]) => ({
          id,
          row,
          prog: Array.isArray(prog) ? (prog as ProgressLine[]) : [],
          lines: Array.isArray((logPayload as { lines?: string[] })?.lines)
            ? (logPayload as { lines: string[] }).lines
            : [],
        })),
      ),
    ).then((results) => {
      const rows: RunInfo[] = []
      const pm: Record<string, ProgressLine[]> = {}
      const lm: Record<string, string[]> = {}
      const missingIds: string[] = []
      
      for (const { id, row, prog, lines } of results) {
        const rowData = row as Record<string, unknown> | null
        if (rowData?.__missing) {
          missingIds.push(id)
          continue
        }
        if (row) rows.push(row as RunInfo)
        pm[id] = prog
        lm[id] = lines
      }
      setRunRows(rows)
      setProgressByRun(pm)
      setLogByRun(lm)
      
      if (missingIds.length > 0) {
        setSessionRunIds((prev) => prev.filter((id) => !missingIds.includes(id)))
      }
    })
  }, [sessionRunIds])

  useEffect(() => {
    refreshSessionRuns()
    if (sessionRunIds.length === 0) return
    const t = setInterval(refreshSessionRuns, 2000)
    return () => clearInterval(t)
  }, [sessionRunIds, refreshSessionRuns])

  const stopSessionRun = useCallback(
    async (runId: string) => {
      if (!window.confirm('Stop this pipeline run? The current main.py process will be terminated.')) return
      setStoppingRunId(runId)
      try {
        const r = await fetch(`${API}/api/runs/${runId}/stop`, { method: 'POST' })
        const text = await r.text()
        if (r.status === 409) {
          let msg = text
          try {
            const j = JSON.parse(text) as { detail?: unknown }
            if (typeof j.detail === 'string') msg = j.detail
          } catch {
            /* plain text body */
          }
          window.alert(
            msg ||
              'Cannot stop: run is not active in this API process (e.g. server restarted). Stop main.py manually if it is still running.',
          )
          return
        }
        if (!r.ok) {
          window.alert(`Stop failed (HTTP ${r.status}). ${text.slice(0, 200)}`)
          return
        }
        await refreshSessionRuns()
      } finally {
        setStoppingRunId(null)
      }
    },
    [refreshSessionRuns],
  )

  const rerunSessionRun = useCallback(
    async (runId: string) => {
      setRerunningRunId(runId)
      try {
        const r = await fetch(`${API}/api/runs/${runId}/rerun`, { method: 'POST' })
        const text = await r.text()
        if (!r.ok) {
          let msg = `Rerun failed (HTTP ${r.status})`
          try {
            const j = JSON.parse(text) as { detail?: unknown }
            if (typeof j.detail === 'string') msg = j.detail
          } catch {
            if (text.trim()) msg = text.slice(0, 280)
          }
          window.alert(msg)
          return
        }
        const j = JSON.parse(text) as { run_id: string }
        setSessionRunIds((prev) => [...prev, j.run_id])
        await refreshSessionRuns()
      } finally {
        setRerunningRunId(null)
      }
    },
    [refreshSessionRuns, setSessionRunIds],
  )

  const deleteSessionRun = useCallback(
    async (runId: string) => {
      const row = runRows.find((r) => r.run_id === runId)
      if (row?.status === 'running' && row.subprocess_alive === true) {
        window.alert('Stop the run first, or wait until it finishes — a live pipeline process is still attached.')
        return
      }
      if (
        !window.confirm(
          'Delete this run from disk? Outputs, logs, and previews will be removed. This cannot be undone.',
        )
      ) {
        return
      }
      setDeletingRunId(runId)
      try {
        const r = await fetch(`${API}/api/runs/${runId}`, { method: 'DELETE' })
        if (r.status === 409) {
          window.alert(
            'Cannot delete while the pipeline subprocess is still running. Click Stop, wait for it to exit, then try again.',
          )
          return
        }
        if (!r.ok && r.status !== 204) {
          window.alert(`Delete failed (HTTP ${r.status}).`)
          return
        }
        setSessionRunIds((prev) => prev.filter((id) => id !== runId))
        setRunRows((prev) => prev.filter((x) => x.run_id !== runId))
        setProgressByRun((prev) => {
          const next = { ...prev }
          delete next[runId]
          return next
        })
        setLogByRun((prev) => {
          const next = { ...prev }
          delete next[runId]
          return next
        })
        setConfigRunId((cur) => (cur === runId ? null : cur))
      } finally {
        setDeletingRunId(null)
      }
    },
    [runRows, setSessionRunIds],
  )



  const startAll = () => {
    if (!videoFile || !schema || drafts.length === 0 || starting) return
    setBatchError(null)
    setStarting(true)
    const runsPayload = drafts.map((d) => ({
      recipe_name: d.recipeName,
      fields: d.fields,
    }))
    const body = new FormData()
    body.append('file', videoFile)
    body.append('runs_json', JSON.stringify(runsPayload))
    fetch(`${API}/api/runs/batch_upload`, { method: 'POST', body })
      .then(async (r) => {
        const text = await r.text()
        if (!r.ok) {
          let msg = `Start failed (HTTP ${r.status})`
          try {
            const j = JSON.parse(text) as { detail?: string }
            if (typeof j.detail === 'string') msg = j.detail
          } catch {
            if (text.trim()) msg = text.slice(0, 240)
          }
          throw new Error(msg)
        }
        const j = JSON.parse(text) as { run_ids: string[] }
        setSessionRunIds(j.run_ids)
        clearDrafts()
        setEditingId(null)
      })
      .catch((e) => {
        setBatchError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => setStarting(false))
  }

  const planAnotherBatch = () => {
    setSessionRunIds([])
    setRunRows([])
    setBatchError(null)
    // useEffect adds the first draft when session is cleared and drafts stay empty
  }

  const sessionRunsOrdered = useMemo(
    () =>
      sessionRunIds.map((rid, i) => {
        const run =
          runRows.find((r) => r.run_id === rid) ??
          ({
            run_id: rid,
            status: 'queued',
            recipe_name: '',
            video_stem: '',
          } as RunInfo)
        return { run, batchIndex: i + 1, totalInBatch: sessionRunIds.length }
      }),
    [sessionRunIds, runRows],
  )

  const batchStats = useMemo(() => {
    let done = 0
    let runningLive = 0
    let staleRunning = 0
    let queued = 0
    let err = 0
    let cancelled = 0
    let other = 0
    for (const { run } of sessionRunsOrdered) {
      if (run.status === 'done') done += 1
      else if (run.status === 'running') {
        if (run.subprocess_alive === false) staleRunning += 1
        else runningLive += 1
      } else if (run.status === 'queued') queued += 1
      else if (run.status === 'error') err += 1
      else if (run.status === 'cancelled') cancelled += 1
      else other += 1
    }
    return { done, runningLive, staleRunning, queued, err, cancelled, other, total: sessionRunsOrdered.length }
  }, [sessionRunsOrdered])

  const doneIds = useMemo(
    () => sessionRunIds.filter((rid) => runRows.find((r) => r.run_id === rid)?.status === 'done'),
    [sessionRunIds, runRows],
  )
  const canCompare = doneIds.length >= 2
  const allTerminal =
    sessionRunIds.length > 0 &&
    runRows.length === sessionRunIds.length &&
    runRows.every((r) => {
      if (['done', 'error', 'cancelled'].includes(r.status)) return true
      if (r.status === 'running' && r.subprocess_alive === false) return true
      return false
    })

  if (schemaError) {
    return (
      <div className="glass-panel" style={{ padding: '2rem', maxWidth: 560, margin: '0 auto' }}>
        <p style={{ color: '#f87171' }}>{schemaError}</p>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
          Start the API from <code style={{ color: 'var(--halo-cyan)' }}>sway_pose_mvp</code>:
        </p>
        <pre
          style={{
            padding: '1rem',
            borderRadius: 12,
            background: 'rgba(0,0,0,0.35)',
            fontSize: '0.85rem',
          }}
        >
          uvicorn pipeline_lab.server.app:app --reload --host 127.0.0.1 --port 8765
        </pre>
      </div>
    )
  }

  if (!schema) {
    return (
      <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
        Loading…
      </div>
    )
  }

  if (!labHydrated) {
    return (
      <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Restoring your lab session…
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', position: 'relative' }}>
      {/* Step 1: no video yet — empty Lab was rendering nothing here before */}
      {!videoFile && (
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h2 style={{ margin: 0, fontSize: '1.15rem', color: '#fff' }}>1. Add a source video</h2>
          <p style={{ margin: '0.5rem 0 0', fontSize: '0.92rem', color: 'var(--text-muted)', lineHeight: 1.55, maxWidth: 640 }}>
            One clip is shared across every run in a batch. Use the bar above or drop a file here, then configure runs and
            start the pipeline.
          </p>
          <input
            id="sway-lab-video-input"
            type="file"
            accept={VIDEO_ACCEPT_ATTR}
            className="sr-only-file-input"
            onChange={(e) => {
              applyVideoFile(e.target.files?.[0])
              e.target.value = ''
            }}
          />
          <label
            htmlFor="sway-lab-video-input"
            className={`hero-upload ${dropActive ? 'dragover' : ''}`}
            style={{ marginTop: '1.25rem', display: 'flex' }}
            onDragEnter={(e) => {
              e.preventDefault()
              e.stopPropagation()
              dropDepthRef.current += 1
              setDropActive(true)
            }}
            onDragOver={(e) => {
              e.preventDefault()
              e.stopPropagation()
              e.dataTransfer.dropEffect = 'copy'
            }}
            onDragLeave={(e) => {
              e.preventDefault()
              e.stopPropagation()
              dropDepthRef.current = Math.max(0, dropDepthRef.current - 1)
              if (dropDepthRef.current === 0) setDropActive(false)
            }}
            onDrop={(e) => {
              e.preventDefault()
              e.stopPropagation()
              dropDepthRef.current = 0
              setDropActive(false)
              applyVideoFile(e.dataTransfer.files?.[0])
            }}
          >
            <Upload size={44} strokeWidth={1.25} style={{ color: 'var(--halo-cyan)', marginBottom: '0.35rem' }} aria-hidden />
            <h2 style={{ fontSize: '1.35rem' }}>Drop a video here</h2>
            <p>or click to browse — MP4, MOV, WebM, MKV, AVI, M4V</p>
            <span className="btn primary" style={{ pointerEvents: 'none' }}>
              Choose file
            </span>
          </label>
        </div>
      )}

      {/* Step 2: drafts */}
      {videoFile && sessionRunIds.length === 0 && (
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
            <h2 style={{ margin: 0, fontSize: '1.15rem', color: '#fff' }}>2. Runs on this video</h2>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <button type="button" className="btn" onClick={() => addDraft()}>
                <Plus size={16} /> Add run
              </button>
              <button
                type="button"
                className="btn primary"
                disabled={drafts.length === 0 || starting || !schema}
                onClick={startAll}
              >
                <Clapperboard size={16} /> {starting ? 'Starting…' : 'Start all runs'}
              </button>
            </div>
          </div>
          {batchError && (
            <p style={{ color: '#f87171', marginTop: '1rem', marginBottom: 0 }}>{batchError}</p>
          )}
          <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {drafts.map((d) => (
              <div
                key={d.clientId}
                className="glass-panel"
                style={{
                  padding: '1rem 1.25rem',
                  display: 'flex',
                  flexWrap: 'wrap',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '0.75rem',
                  border: '1px solid var(--glass-border)',
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#fff' }}>{d.recipeName}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Click edit to change pipeline parameters</div>
                </div>
                <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                  <button type="button" className="btn" style={{ padding: '0.45rem 0.75rem' }} onClick={() => setEditingId(d.clientId)}>
                    <Pencil size={14} /> Edit
                  </button>
                  <button type="button" className="btn" style={{ padding: '0.45rem 0.75rem' }} onClick={() => duplicateDraft(d.clientId)}>
                    <Copy size={14} /> Duplicate
                  </button>
                  <button
                    type="button"
                    className="btn"
                    style={{ padding: '0.45rem 0.75rem', color: '#f87171', borderColor: 'rgba(248,113,113,0.35)' }}
                    onClick={() => removeDraft(d.clientId)}
                    disabled={drafts.length <= 1}
                  >
                    <Trash2 size={14} /> Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {/* Step 3: active / completed batch */}
      {sessionRunIds.length > 0 && (
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem' }}>
            <div style={{ flex: '1 1 220px' }}>
              <h2 style={{ margin: 0, fontSize: '1.15rem', color: '#fff' }}>3. Batch progress</h2>
              <p style={{ margin: '0.5rem 0 0', fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.5, maxWidth: 560 }}>
                Each card is one full pipeline on your shared video. Jobs start in order: by default only one is{' '}
                <strong style={{ color: 'var(--text-main)', fontWeight: 600 }}>actively processing</strong> at a time;
                the rest stay <strong style={{ color: 'var(--text-main)', fontWeight: 600 }}>queued</strong> until the
                worker is free.
              </p>
              <div
                style={{
                  marginTop: '0.85rem',
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '0.5rem',
                  fontSize: '0.82rem',
                  color: 'var(--text-muted)',
                }}
              >
                <span style={{ background: 'rgba(16,185,129,0.12)', color: '#6ee7b7', padding: '0.25rem 0.55rem', borderRadius: 8 }}>
                  {batchStats.done} finished
                </span>
                <span style={{ background: 'rgba(14,165,233,0.12)', color: '#7dd3fc', padding: '0.25rem 0.55rem', borderRadius: 8 }}>
                  {batchStats.runningLive} active
                </span>
                {batchStats.staleRunning > 0 && (
                  <span style={{ background: 'rgba(245,158,11,0.14)', color: '#fcd34d', padding: '0.25rem 0.55rem', borderRadius: 8 }}>
                    {batchStats.staleRunning} stale
                  </span>
                )}
                <span style={{ background: 'rgba(167,139,250,0.15)', color: '#c4b5fd', padding: '0.25rem 0.55rem', borderRadius: 8 }}>
                  {batchStats.queued} queued
                </span>
                {batchStats.err > 0 && (
                  <span style={{ background: 'rgba(239,68,68,0.12)', color: '#fca5a5', padding: '0.25rem 0.55rem', borderRadius: 8 }}>
                    {batchStats.err} failed
                  </span>
                )}
                {batchStats.cancelled > 0 && (
                  <span style={{ background: 'rgba(251,191,36,0.12)', color: '#fcd34d', padding: '0.25rem 0.55rem', borderRadius: 8 }}>
                    {batchStats.cancelled} stopped
                  </span>
                )}
                <span style={{ padding: '0.25rem 0', color: 'var(--text-muted)' }}>· {batchStats.total} runs in this batch</span>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
              <button
                type="button"
                className="btn primary"
                disabled={!canCompare}
                onClick={() => nav(`/compare?runs=${encodeURIComponent(doneIds.join(','))}`)}
              >
                <Columns2 size={16} /> Compare outputs
              </button>
              {allTerminal && (
                <button type="button" className="btn" onClick={planAnotherBatch}>
                  <Plus size={16} /> New runs (same video)
                </button>
              )}
            </div>
          </div>

          {sessionRunIds.length >= 2 && (
            <div
              style={{
                marginTop: '1rem',
                padding: '0.85rem 1rem',
                borderRadius: 12,
                background: canCompare ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.04)',
                border: `1px solid ${canCompare ? 'rgba(16,185,129,0.25)' : 'var(--glass-border)'}`,
                fontSize: '0.88rem',
                lineHeight: 1.5,
                color: 'var(--text-muted)',
              }}
            >
              {canCompare ? (
                <>
                  <strong style={{ color: '#a7f3d0' }}>Compare is ready.</strong> You have {batchStats.done} successful
                  run{batchStats.done === 1 ? '' : 's'} — open side-by-side playback with one shared timeline.
                </>
              ) : (
                <>
                  <strong style={{ color: 'var(--text-main)' }}>Compare unlocks after 2 runs finish successfully.</strong>{' '}
                  Right now: <strong style={{ color: '#7dd3fc' }}>{batchStats.done} done</strong>,{' '}
                  <strong style={{ color: '#7dd3fc' }}>{batchStats.runningLive} active</strong>
                  {batchStats.staleRunning > 0 && (
                    <>
                      , <strong style={{ color: '#fcd34d' }}>{batchStats.staleRunning} stale</strong>
                    </>
                  )}
                  , <strong style={{ color: '#c4b5fd' }}>{batchStats.queued} queued</strong>
                  {batchStats.err > 0 && (
                    <>
                      , <strong style={{ color: '#fca5a5' }}>{batchStats.err} failed</strong>
                    </>
                  )}
                  {batchStats.cancelled > 0 && (
                    <>
                      , <strong style={{ color: '#fcd34d' }}>{batchStats.cancelled} stopped</strong>
                    </>
                  )}
                  .
                </>
              )}
            </div>
          )}

          <div
            style={{
              marginTop: '1.25rem',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 380px), 1fr))',
              gap: '1rem',
            }}
          >
            {sessionRunsOrdered.map(({ run, batchIndex, totalInBatch }) => {
              const pres = statusPresentationForRun(run)
              const { title: statusTitle, subtitle: statusSubtitle, color, Icon } = pres
              const isDone = run.status === 'done'
              const isErr = run.status === 'error'
              const isCancelled = run.status === 'cancelled'
              const isRunning = run.status === 'running'
              const subprocessLive = isRunning && run.subprocess_alive === true
              const subprocessDetached = isRunning && run.subprocess_alive === false
              const displayName =
                (run.recipe_name && run.recipe_name.trim()) || `Configuration ${batchIndex}`
              const videoLine = run.video_stem || videoLabel.replace(/\.[^/.]+$/, '') || videoLabel
              const created = formatCreated(run.created)

              return (
                <div
                  key={run.run_id}
                  className="glass-panel"
                  style={{
                    padding: '1.25rem',
                    border: `1px solid ${color}40`,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.65rem',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.75rem' }}>
                    <span
                      style={{
                        fontSize: '0.72rem',
                        fontWeight: 700,
                        letterSpacing: '0.04em',
                        textTransform: 'uppercase',
                        color: 'var(--text-muted)',
                      }}
                    >
                      Run {batchIndex} of {totalInBatch}
                    </span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'ui-monospace, monospace' }}>
                      {formatShortId(run.run_id)}
                    </span>
                  </div>

                  <div>
                    <div style={{ fontWeight: 700, color: '#fff', fontSize: '1.05rem' }}>{displayName}</div>
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                      Source video: <span style={{ color: 'var(--text-main)' }}>{videoLine}</span>
                    </div>
                    {created && (
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Started {created}</div>
                    )}
                  </div>

                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '0.65rem',
                      padding: '0.65rem 0.75rem',
                      borderRadius: 10,
                      background: `${color}14`,
                      border: `1px solid ${color}30`,
                    }}
                  >
                    <Icon
                      size={20}
                      className={subprocessLive ? 'sway-spin' : undefined}
                      style={{ flexShrink: 0, color }}
                    />
                    <div>
                      <div style={{ fontWeight: 700, color, fontSize: '0.9rem' }}>{statusTitle}</div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.2rem', lineHeight: 1.45 }}>
                        {statusSubtitle}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.35rem', opacity: 0.85 }}>
                        API status: <code style={{ color: 'var(--halo-cyan)' }}>{run.status}</code>
                        {subprocessDetached && (
                          <span style={{ color: '#94a3b8' }}> · subprocess_alive=false</span>
                        )}
                        {subprocessDetached && (
                          <span style={{ display: 'block', marginTop: '0.3rem', color: '#fcd34d', lineHeight: 1.45 }}>
                            Quit stray <code>main.py</code> in Activity Monitor if your machine is still busy from an old run.
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.65rem' }}>
                    <button
                      type="button"
                      onClick={() => setConfigRunId(run.run_id)}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: 'var(--halo-cyan)',
                        fontWeight: 600,
                        fontSize: '0.78rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.35rem',
                        padding: 0,
                        fontFamily: 'inherit',
                      }}
                    >
                      <Settings2 size={14} aria-hidden /> View configuration
                    </button>
                    {subprocessLive && (
                      <button
                        type="button"
                        className="btn"
                        disabled={stoppingRunId === run.run_id}
                        title="Send SIGTERM to the main.py subprocess for this run"
                        style={{
                          padding: '0.4rem 0.65rem',
                          fontSize: '0.76rem',
                          color: '#fcd34d',
                          borderColor: 'rgba(251,191,36,0.4)',
                        }}
                        onClick={() => void stopSessionRun(run.run_id)}
                      >
                        <StopCircle size={14} aria-hidden /> {stoppingRunId === run.run_id ? 'Stopping…' : 'Stop run'}
                      </button>
                    )}
                    <button
                      type="button"
                      className="btn"
                      disabled={
                        subprocessLive || rerunningRunId === run.run_id || deletingRunId === run.run_id
                      }
                      title={
                        subprocessLive
                          ? 'Wait until this run finishes or stop it'
                          : 'Queue a new run with the same video and settings (new run id)'
                      }
                      style={{
                        padding: '0.4rem 0.65rem',
                        fontSize: '0.76rem',
                        color: '#7dd3fc',
                        borderColor: 'rgba(125,211,252,0.35)',
                      }}
                      onClick={() => void rerunSessionRun(run.run_id)}
                    >
                      <RotateCw
                        size={14}
                        aria-hidden
                        className={rerunningRunId === run.run_id ? 'sway-spin' : undefined}
                      />{' '}
                      {rerunningRunId === run.run_id ? 'Queuing…' : 'Rerun'}
                    </button>
                    <button
                      type="button"
                      className="btn"
                      disabled={subprocessLive || deletingRunId === run.run_id}
                      title={
                        subprocessLive
                          ? 'Stop the run first, or wait until it finishes'
                          : 'Remove this run from disk and this batch'
                      }
                      style={{
                        padding: '0.4rem 0.65rem',
                        fontSize: '0.76rem',
                        color: '#f87171',
                        borderColor: 'rgba(248,113,113,0.35)',
                      }}
                      onClick={() => void deleteSessionRun(run.run_id)}
                    >
                      <Trash2 size={14} aria-hidden /> Delete run
                    </button>
                  </div>

                  <details style={{ marginTop: '0.5rem' }}>
                    <summary style={{ cursor: 'pointer', color: 'var(--halo-cyan)', fontSize: '0.85rem', fontWeight: 600, padding: '0.25rem 0', outline: 'none' }}>
                      Run details & process logs
                    </summary>
                    <div style={{ paddingTop: '0.75rem' }}>
                      <RunPipelineDetail
                        runStatus={run.status}
                        subprocessAlive={run.subprocess_alive}
                        progressLines={progressByRun[run.run_id] ?? []}
                        logLines={logByRun[run.run_id] ?? []}
                        trackSummary={
                          (run.manifest?.run_context_final?.track_summary as Record<string, unknown> | undefined) ?? null
                        }
                      />
                    </div>
                  </details>

                  {run.error && (
                    <div style={{ fontSize: '0.85rem', color: '#f87171', lineHeight: 1.45 }}>{run.error}</div>
                  )}

                  {isDone && (
                    <button
                      type="button"
                      className="btn primary"
                      style={{ marginTop: '0.25rem', width: '100%' }}
                      onClick={() => nav(`/watch/${run.run_id}`)}
                    >
                      <Play size={16} /> Watch results
                    </button>
                  )}
                  {isErr && (
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Fix the issue and start a new run from step 2, or clear the session to try again.
                    </div>
                  )}
                  {isCancelled && (
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Delete this run if you do not need partial outputs, or start a new batch from step 2.
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      <RunEditorModal
        key={editingId ?? 'closed'}
        open={Boolean(editingId && editingDraft)}
        schema={schema}
        draft={editingDraft}
        onClose={() => setEditingId(null)}
        onSave={(recipeName, fields) => {
          if (!editingId) return
          updateDraft(editingId, { recipeName, fields })
          setEditingId(null)
        }}
      />

      <RunConfigModal
        open={configRunId !== null}
        runId={configRunId}
        schema={schema}
        onClose={() => setConfigRunId(null)}
      />
    </div>
  )
}
