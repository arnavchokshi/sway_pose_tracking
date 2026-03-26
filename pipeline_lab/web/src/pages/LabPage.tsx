import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { PIPELINE_LAB_LOCAL } from '../siteUrls'
import { API } from '../types'
import type { BatchSummary, Schema, RunInfo, ProgressLine } from '../types'
import { safeJsonPreview } from '../lib/jsonPreview'
import { isProbableVideoFile, VIDEO_ACCEPT_ATTR } from '../lib/videoFile'
import { loadSessionBatchFilterId, persistSessionBatchFilterId } from '../lib/labPersistence'
import { statusPresentationForRun } from '../lib/runStatusPresentation'
import {
  computePhaseDepths,
  flattenTreeVisualOrder,
  sessionHasCheckpointTree,
  subtreeRunIdsInSession,
} from '../lib/batchTreeLayout'
import { compareViewModeForTreeColumn } from '../lib/treeColumnCompareView'
import { BatchTreeView } from '../components/BatchTreeView'
import { useLab } from '../context/LabContext'
import { RunEditorModal } from '../components/RunEditorModal'
import { RunConfigModal } from '../components/RunConfigModal'
import { TrackQualitySummary, PipelineImpactSummary } from '../components/RunMetrics'
import {
  Play,
  Plus,
  Pencil,
  Copy,
  Trash2,
  Clapperboard,
  Columns2,
  Terminal,
  Settings2,
  StopCircle,
  RotateCw,
  Upload,
  LayoutGrid,
  GitBranch,
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
  const [searchParams, setSearchParams] = useSearchParams()
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
  const [treePresets, setTreePresets] = useState<Array<{ id: string; filename: string }>>([])
  const [treePresetId, setTreePresetId] = useState('')
  const [treeVideoA, setTreeVideoA] = useState<File | null>(null)
  const [treeVideoB, setTreeVideoB] = useState<File | null>(null)
  const [treeQueueBusy, setTreeQueueBusy] = useState(false)
  const [treeQueueError, setTreeQueueError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [configRunId, setConfigRunId] = useState<string | null>(null)

  const [runRows, setRunRows] = useState<RunInfo[]>([])
  const [progressByRun, setProgressByRun] = useState<Record<string, ProgressLine[]>>({})
  const [logByRun, setLogByRun] = useState<Record<string, string[]>>({})
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null)
  const [stoppingRunId, setStoppingRunId] = useState<string | null>(null)
  const [rerunningRunId, setRerunningRunId] = useState<string | null>(null)
  const [fileDropOverlay, setFileDropOverlay] = useState(false)

  const [deletingAllRuns, setDeletingAllRuns] = useState(false)
  const [batchViewMode, setBatchViewMode] = useState<'list' | 'tree'>('list')
  /** Batches on the API when this browser has no session yet (resume without ``?batch=``). */
  const [serverBatchesHint, setServerBatchesHint] = useState<BatchSummary[]>([])
  const [treeCompareSelected, setTreeCompareSelected] = useState<Set<string>>(() => new Set())
  /** Anchor run for shift-click range select on tree Compare (last plain click on a finished run). */
  const treeCompareAnchorRef = useRef<string | null>(null)
  const urlSessionFromQueryDone = useRef(false)
  /** When set (from `?batch=`), session run list is re-synced from GET /api/runs so CLI fan-out adds new runs with the same batch_id. */
  const [sessionBatchFilterId, setSessionBatchFilterId] = useState<string | null>(null)
  const prevSessionRunCountRef = useRef(0)

  useEffect(() => {
    if (!labHydrated || urlSessionFromQueryDone.current) return
    const batch = searchParams.get('batch')?.trim()
    const runsParam = searchParams.get('runs')?.trim()
    if (!batch && !runsParam) return
    urlSessionFromQueryDone.current = true

    const stripBatchRunsParams = () => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          next.delete('batch')
          next.delete('runs')
          return next
        },
        { replace: true },
      )
    }

    if (runsParam) {
      setSessionBatchFilterId(null)
      const ids = runsParam.split(',').map((s) => s.trim()).filter(Boolean)
      if (ids.length > 0) setSessionRunIds(ids)
      stripBatchRunsParams()
      return
    }

    if (batch) {
      setSessionBatchFilterId(batch)
      stripBatchRunsParams()
    }
  }, [labHydrated, searchParams, setSearchParams, setSessionRunIds])

  useEffect(() => {
    if (!sessionBatchFilterId) return
    const syncFromBatch = () => {
      void fetch(`${API}/api/runs`)
        .then((r) => r.json())
        .then((rows: unknown) => {
          if (!Array.isArray(rows)) return
          const matched = rows.filter(
            (x): x is RunInfo =>
              typeof x === 'object' && x !== null && (x as RunInfo).batch_id === sessionBatchFilterId,
          )
          matched.sort((a, b) => {
            const ca = a.created ?? ''
            const cb = b.created ?? ''
            if (ca !== cb) return ca.localeCompare(cb)
            return a.run_id.localeCompare(b.run_id)
          })
          const nextIds = matched.map((x) => x.run_id)
          setSessionRunIds((prev) => {
            if (prev.length === nextIds.length && prev.every((id, i) => id === nextIds[i])) return prev
            return nextIds
          })
        })
        .catch(() => {
          /* offline / CORS */
        })
    }
    syncFromBatch()
    const t = setInterval(syncFromBatch, 2000)
    return () => clearInterval(t)
  }, [sessionBatchFilterId, setSessionRunIds])

  useEffect(() => {
    if (prevSessionRunCountRef.current > 0 && sessionRunIds.length === 0) {
      setSessionBatchFilterId(null)
    }
    prevSessionRunCountRef.current = sessionRunIds.length
  }, [sessionRunIds.length])

  const prevBatchFilterRef = useRef<string | null | undefined>(undefined)
  useEffect(() => {
    if (sessionBatchFilterId !== null) {
      persistSessionBatchFilterId(sessionBatchFilterId)
    } else if (prevBatchFilterRef.current !== undefined && prevBatchFilterRef.current !== null) {
      persistSessionBatchFilterId(null)
    }
    prevBatchFilterRef.current = sessionBatchFilterId
  }, [sessionBatchFilterId])

  /** After refresh on `/`, `?batch=` is gone — restore filter from localStorage unless a new `?batch=` is in the URL. */
  useEffect(() => {
    if (!labHydrated) return
    if (searchParams.get('batch')?.trim()) return
    const fromLs = loadSessionBatchFilterId()
    if (!fromLs) return
    setSessionBatchFilterId((cur) => (cur ? cur : fromLs))
  }, [labHydrated, searchParams])

  function parseBatchSummaries(rows: unknown): BatchSummary[] {
    if (!Array.isArray(rows)) return []
    const out: BatchSummary[] = []
    for (const x of rows) {
      if (!x || typeof x !== 'object') continue
      const o = x as Record<string, unknown>
      if (typeof o.batch_id !== 'string' || !o.batch_id.trim()) continue
      out.push({
        batch_id: o.batch_id.trim(),
        run_count: typeof o.run_count === 'number' ? o.run_count : 0,
        n_queued: typeof o.n_queued === 'number' ? o.n_queued : 0,
        n_running_live: typeof o.n_running_live === 'number' ? o.n_running_live : 0,
        n_running_stale: typeof o.n_running_stale === 'number' ? o.n_running_stale : 0,
        n_done: typeof o.n_done === 'number' ? o.n_done : 0,
        n_error: typeof o.n_error === 'number' ? o.n_error : 0,
        n_cancelled: typeof o.n_cancelled === 'number' ? o.n_cancelled : 0,
        has_checkpoint_tree: o.has_checkpoint_tree === true,
        latest_created: typeof o.latest_created === 'string' ? o.latest_created : '',
      })
    }
    return out
  }

  useEffect(() => {
    if (!labHydrated) return
    if (sessionBatchFilterId) return
    if (sessionRunIds.length > 0) return
    let cancelled = false
    void fetch(`${API}/api/batches`)
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: unknown) => {
        if (cancelled) return
        const parsed = parseBatchSummaries(rows)
        const active = parsed.filter((b) => b.n_queued + b.n_running_live > 0)
        if (active.length === 1) {
          setServerBatchesHint([])
          setSessionBatchFilterId(active[0].batch_id)
          return
        }
        const show = active.length > 1 ? active : parsed.slice(0, 10)
        setServerBatchesHint(show)
      })
      .catch(() => {
        if (!cancelled) setServerBatchesHint([])
      })
    return () => {
      cancelled = true
    }
  }, [labHydrated, sessionBatchFilterId, sessionRunIds.length, setSessionBatchFilterId])

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

  const isFileDragEvent = useCallback((e: DragEvent) => {
    const t = e.dataTransfer?.types
    if (!t) return false
    for (let i = 0; i < t.length; i++) {
      if (t[i] === 'Files') return true
    }
    return false
  }, [])

  useEffect(() => {
    if (!schema || schemaError || !labHydrated) return

    const onEnter = (e: DragEvent) => {
      if (!isFileDragEvent(e)) return
      e.preventDefault()
      setFileDropOverlay(true)
    }

    const onOver = (e: DragEvent) => {
      if (!isFileDragEvent(e)) return
      e.preventDefault()
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy'
    }

    const onLeave = (e: DragEvent) => {
      if (!isFileDragEvent(e)) return
      const rel = e.relatedTarget as Node | null
      if (rel && document.documentElement.contains(rel)) return
      setFileDropOverlay(false)
    }

    const onDrop = (e: DragEvent) => {
      if (!isFileDragEvent(e)) return
      e.preventDefault()
      setFileDropOverlay(false)
      applyVideoFile(e.dataTransfer?.files?.[0])
    }

    const clearOverlay = () => setFileDropOverlay(false)

    document.addEventListener('dragenter', onEnter, true)
    document.addEventListener('dragover', onOver, true)
    document.addEventListener('dragleave', onLeave, true)
    document.addEventListener('drop', onDrop, false)
    document.addEventListener('dragend', clearOverlay, true)
    window.addEventListener('blur', clearOverlay)

    return () => {
      document.removeEventListener('dragenter', onEnter, true)
      document.removeEventListener('dragover', onOver, true)
      document.removeEventListener('dragleave', onLeave, true)
      document.removeEventListener('drop', onDrop, false)
      document.removeEventListener('dragend', clearOverlay, true)
      window.removeEventListener('blur', clearOverlay)
    }
  }, [applyVideoFile, isFileDragEvent, labHydrated, schema, schemaError])

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
    if (!schema) return
    let cancelled = false
    fetch(`${API}/api/tree_presets`)
      .then(async (r) => {
        const text = await r.text()
        if (!r.ok) throw new Error(`tree presets HTTP ${r.status}`)
        return JSON.parse(text) as unknown
      })
      .then((rows) => {
        if (cancelled || !Array.isArray(rows)) return
        const ok = rows.filter(
          (x): x is { id: string; filename: string } =>
            typeof x === 'object' &&
            x !== null &&
            typeof (x as { id?: unknown }).id === 'string' &&
            typeof (x as { filename?: unknown }).filename === 'string',
        )
        setTreePresets(ok)
        setTreePresetId((cur) => (cur && ok.some((p) => p.id === cur) ? cur : ok[0]?.id ?? ''))
      })
      .catch(() => {
        if (!cancelled) setTreePresets([])
      })
    return () => {
      cancelled = true
    }
  }, [schema])

  useEffect(() => {
    if (!labHydrated || !videoFile || !schema || drafts.length > 0 || sessionRunIds.length > 0) return
    if (searchParams.get('batch')?.trim() || searchParams.get('runs')?.trim()) return
    addDraft()
  }, [labHydrated, videoFile, schema, drafts.length, sessionRunIds.length, addDraft, searchParams])

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

  const performDeleteSessionRun = useCallback(
    async (runId: string): Promise<boolean> => {
      const row = runRows.find((r) => r.run_id === runId)
      if (row?.status === 'running' && row.subprocess_alive === true) {
        window.alert('Stop the run first, or wait until it finishes — a live pipeline process is still attached.')
        return false
      }
      setDeletingRunId(runId)
      try {
        const r = await fetch(`${API}/api/runs/${runId}`, { method: 'DELETE' })
        if (r.status === 409) {
          window.alert(
            'Cannot delete while the pipeline subprocess is still running. Click Stop, wait for it to exit, then try again.',
          )
          return false
        }
        if (!r.ok && r.status !== 204) {
          window.alert(`Delete failed (HTTP ${r.status}).`)
          return false
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
        return true
      } finally {
        setDeletingRunId(null)
      }
    },
    [runRows, setSessionRunIds],
  )

  const deleteSessionRun = useCallback(
    async (runId: string) => {
      if (
        !window.confirm(
          'Delete this run from disk? Outputs, logs, and previews will be removed. This cannot be undone.',
        )
      ) {
        return
      }
      await performDeleteSessionRun(runId)
    },
    [performDeleteSessionRun],
  )

  const startCheckpointTreeTwoVideos = useCallback(() => {
    if (!treeVideoA || !treeVideoB || !treePresetId || treeQueueBusy) return
    setTreeQueueError(null)
    setTreeQueueBusy(true)
    const body = new FormData()
    body.append('video_0', treeVideoA)
    body.append('video_1', treeVideoB)
    body.append('preset', treePresetId)
    fetch(`${API}/api/runs/tree_checkpoint_upload`, { method: 'POST', body })
      .then(async (r) => {
        const text = await r.text()
        if (!r.ok) {
          let msg = `Tree queue failed (HTTP ${r.status})`
          try {
            const j = JSON.parse(text) as { detail?: unknown }
            if (typeof j.detail === 'string') msg = j.detail
          } catch {
            if (text.trim()) msg = text.slice(0, 280)
          }
          throw new Error(msg)
        }
        const j = JSON.parse(text) as { batch_id?: string }
        const bid = typeof j.batch_id === 'string' ? j.batch_id : ''
        if (!bid) throw new Error('No batch_id in response')
        setSessionBatchFilterId(bid)
        setSessionRunIds([])
        nav('/', { replace: true })
      })
      .catch((e) => {
        setTreeQueueError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => setTreeQueueBusy(false))
  }, [treeVideoA, treeVideoB, treePresetId, treeQueueBusy, setSessionRunIds, setSessionBatchFilterId, nav])

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
    setSessionBatchFilterId(null)
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

  const deleteAllSessionRuns = useCallback(async () => {
    if (sessionRunIds.length === 0) return
    const blocking = sessionRunsOrdered.filter(
      ({ run }) => run.status === 'running' && run.subprocess_alive === true,
    )
    if (blocking.length > 0) {
      window.alert(
        `Stop ${blocking.length} active run(s) first, or wait until they finish — a live pipeline process is still attached.`,
      )
      return
    }
    if (
      !window.confirm(
        `Delete all ${sessionRunIds.length} run(s) in this batch from disk? Outputs, logs, and previews will be removed. This cannot be undone.`,
      )
    ) {
      return
    }
    setDeletingAllRuns(true)
    try {
      const ids = [...sessionRunIds]
      const remaining: string[] = []
      for (const id of ids) {
        const r = await fetch(`${API}/api/runs/${id}`, { method: 'DELETE' })
        if (r.status === 204 || r.status === 404) continue
        remaining.push(id)
      }
      setSessionRunIds(remaining)
      setRunRows((prev) => prev.filter((x) => remaining.includes(x.run_id)))
      setProgressByRun((prev) => {
        const next = { ...prev }
        for (const id of ids) {
          if (!remaining.includes(id)) delete next[id]
        }
        return next
      })
      setLogByRun((prev) => {
        const next = { ...prev }
        for (const id of ids) {
          if (!remaining.includes(id)) delete next[id]
        }
        return next
      })
      setConfigRunId((cur) => (cur && remaining.includes(cur) ? cur : null))
      if (remaining.length > 0) {
        window.alert(
          `Could not remove ${remaining.length} run folder(s) (still running, or server error). Short IDs: ${remaining.map(formatShortId).join(', ')}`,
        )
      }
    } finally {
      setDeletingAllRuns(false)
    }
  }, [sessionRunIds, sessionRunsOrdered, setSessionRunIds])

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

  const runMap = useMemo(() => {
    const m = new Map<string, RunInfo>()
    for (const r of runRows) m.set(r.run_id, r)
    return m
  }, [runRows])

  const deleteSessionRunSubtree = useCallback(
    async (rootId: string) => {
      const ids = subtreeRunIdsInSession(rootId, sessionRunIds, runMap)
      const depths = computePhaseDepths(sessionRunIds, runMap)
      ids.sort((a, b) => (depths.get(b) ?? 0) - (depths.get(a) ?? 0))
      const blocking = ids.filter((id) => {
        const row = runRows.find((r) => r.run_id === id)
        return row?.status === 'running' && row.subprocess_alive === true
      })
      if (blocking.length > 0) {
        window.alert(
          `Stop ${blocking.length} active run(s) in this subtree first, or wait until they finish — a live pipeline process is still attached.`,
        )
        return
      }
      for (const id of ids) {
        const ok = await performDeleteSessionRun(id)
        if (!ok) break
      }
    },
    [performDeleteSessionRun, runRows, sessionRunIds, runMap],
  )

  const hasCheckpointTree = useMemo(
    () => sessionHasCheckpointTree(sessionRunIds, runMap),
    [sessionRunIds, runMap],
  )

  const treeCompareVisualOrder = useMemo(
    () => flattenTreeVisualOrder(sessionRunIds, runMap),
    [sessionRunIds, runMap],
  )

  useEffect(() => {
    if (hasCheckpointTree) setBatchViewMode('tree')
  }, [hasCheckpointTree])

  useEffect(() => {
    setTreeCompareSelected(new Set())
    treeCompareAnchorRef.current = null
  }, [sessionRunIds.join('\0')])

  const onTreeCompareToggle = useCallback(
    (runId: string, { shiftKey }: { shiftKey: boolean }) => {
      const row = runMap.get(runId)
      if (row?.status !== 'done') return

      if (shiftKey && treeCompareVisualOrder.length > 0) {
        const iClick = treeCompareVisualOrder.indexOf(runId)
        if (iClick < 0) return
        const anchor = treeCompareAnchorRef.current
        const iAnchor = anchor != null ? treeCompareVisualOrder.indexOf(anchor) : -1
        const i0 = iAnchor < 0 ? iClick : iAnchor
        const lo = Math.min(i0, iClick)
        const hi = Math.max(i0, iClick)
        setTreeCompareSelected((prev) => {
          const next = new Set(prev)
          for (let i = lo; i <= hi; i++) {
            const id = treeCompareVisualOrder[i]
            if (runMap.get(id)?.status === 'done') next.add(id)
          }
          return next
        })
        return
      }

      treeCompareAnchorRef.current = runId
      setTreeCompareSelected((prev) => {
        const next = new Set(prev)
        if (next.has(runId)) next.delete(runId)
        else next.add(runId)
        return next
      })
    },
    [runMap, treeCompareVisualOrder],
  )

  const selectedDoneCompareIds = useMemo(() => {
    return [...treeCompareSelected].filter((id) => runRows.find((r) => r.run_id === id)?.status === 'done')
  }, [treeCompareSelected, runRows])

  const canCompareSelected = selectedDoneCompareIds.length >= 2

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
          uvicorn pipeline_lab.server.app:app --reload --host localhost --port 8765
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
      {fileDropOverlay && (
        <div
          role="presentation"
          aria-hidden
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 20000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(8, 12, 20, 0.94)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            pointerEvents: 'none',
          }}
        >
          <div
            className="hero-upload dragover"
            style={{
              pointerEvents: 'none',
              maxWidth: 520,
              width: 'min(520px, calc(100vw - 3rem))',
              margin: '0 1.5rem',
              padding: '3rem 2rem',
            }}
          >
            <Upload size={48} strokeWidth={1.25} style={{ color: 'var(--halo-cyan)', marginBottom: '0.35rem' }} aria-hidden />
            <h2 style={{ fontSize: 'clamp(1.35rem, 4vw, 1.75rem)' }}>Drop a video here</h2>
            <p style={{ marginBottom: 0 }}>MP4, MOV, WebM, MKV, AVI, M4V</p>
          </div>
        </div>
      )}
      {sessionRunIds.length === 0 && serverBatchesHint.length > 0 && (
        <div className="glass-panel" style={{ padding: '1.25rem 1.5rem' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.6rem', marginBottom: '0.65rem' }}>
            <LayoutGrid size={20} strokeWidth={1.5} style={{ color: 'var(--halo-cyan)' }} aria-hidden />
            <h2 style={{ margin: 0, fontSize: '1.05rem', color: '#fff' }}>Batches on this API</h2>
          </div>
          <p style={{ margin: '0 0 0.85rem', fontSize: '0.86rem', color: 'var(--text-muted)', lineHeight: 1.5, maxWidth: 720 }}>
            Open one to attach it to this browser — no need for a special URL. Checkpoint trees switch to tree view
            automatically once runs load.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {serverBatchesHint.map((b) => {
              const active = b.n_queued + b.n_running_live
              const short = b.batch_id.length > 10 ? `${b.batch_id.slice(0, 8)}…` : b.batch_id
              const label =
                active > 0
                  ? `${short} · ${active} active, ${b.run_count} total`
                  : `${short} · ${b.run_count} run${b.run_count === 1 ? '' : 's'}`
              return (
                <button
                  key={b.batch_id}
                  type="button"
                  className="btn"
                  style={{ fontSize: '0.78rem', padding: '0.4rem 0.65rem', textAlign: 'left' }}
                  onClick={() => {
                    setSessionBatchFilterId(b.batch_id)
                    setServerBatchesHint([])
                  }}
                >
                  {b.has_checkpoint_tree ? 'Tree · ' : ''}
                  {label}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {sessionRunIds.length === 0 && (
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <GitBranch size={22} strokeWidth={1.5} style={{ color: 'var(--halo-cyan)' }} aria-hidden />
            <h2 style={{ margin: 0, fontSize: '1.15rem', color: '#fff' }}>Checkpoint tree — two videos</h2>
          </div>
          <p style={{ margin: '0.25rem 0 0', fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.55, maxWidth: 720 }}>
            Upload two clips and run the <strong style={{ color: '#e2e8f0', fontWeight: 600 }}>same</strong> preset tree on
            both, <strong style={{ color: '#e2e8f0', fontWeight: 600 }}>one after the other</strong> (all jobs share one batch
            so you can track everything together). Video A completes every stage before video B starts.
          </p>
          {treePresets.length === 0 ? (
            <p style={{ marginTop: '1rem', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
              No presets found. Add <code style={{ color: 'var(--halo-cyan)' }}>*.yaml</code> under{' '}
              <code style={{ color: 'var(--halo-cyan)' }}>pipeline_lab/tree_presets/</code>, or use the CLI{' '}
              <code style={{ color: 'var(--halo-cyan)' }}>python -m tools.pipeline_tree_queue --video … --video …</code>.
            </p>
          ) : (
            <div style={{ marginTop: '1.1rem', display: 'flex', flexDirection: 'column', gap: '0.85rem', maxWidth: 520 }}>
              <label style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Tree preset
                <select
                  className="btn"
                  style={{ padding: '0.5rem 0.65rem', cursor: 'pointer', textAlign: 'left' }}
                  value={treePresetId}
                  onChange={(e) => setTreePresetId(e.target.value)}
                >
                  {treePresets.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.filename}
                    </option>
                  ))}
                </select>
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 200px), 1fr))', gap: '0.75rem' }}>
                <label
                  htmlFor="sway-tree-video-a"
                  style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.82rem', color: 'var(--text-muted)', cursor: 'pointer' }}
                >
                  Video A (runs first)
                  <input
                    type="file"
                    accept={VIDEO_ACCEPT_ATTR}
                    className="sr-only-file-input"
                    id="sway-tree-video-a"
                    onChange={(e) => {
                      const f = e.target.files?.[0]
                      if (f && !isProbableVideoFile(f)) {
                        window.alert('Please use a video file (MP4, MOV, WebM, MKV, AVI, or M4V).')
                        e.target.value = ''
                        return
                      }
                      setTreeVideoA(f ?? null)
                      e.target.value = ''
                    }}
                  />
                  <span
                    className="btn"
                    style={{ padding: '0.45rem 0.65rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                  >
                    {treeVideoA ? treeVideoA.name : 'Choose file…'}
                  </span>
                </label>
                <label
                  htmlFor="sway-tree-video-b"
                  style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.82rem', color: 'var(--text-muted)', cursor: 'pointer' }}
                >
                  Video B (runs second)
                  <input
                    type="file"
                    accept={VIDEO_ACCEPT_ATTR}
                    className="sr-only-file-input"
                    id="sway-tree-video-b"
                    onChange={(e) => {
                      const f = e.target.files?.[0]
                      if (f && !isProbableVideoFile(f)) {
                        window.alert('Please use a video file (MP4, MOV, WebM, MKV, AVI, or M4V).')
                        e.target.value = ''
                        return
                      }
                      setTreeVideoB(f ?? null)
                      e.target.value = ''
                    }}
                  />
                  <span
                    className="btn"
                    style={{ padding: '0.45rem 0.65rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                  >
                    {treeVideoB ? treeVideoB.name : 'Choose file…'}
                  </span>
                </label>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.5rem' }}>
                <button
                  type="button"
                  className="btn primary"
                  disabled={!treeVideoA || !treeVideoB || treeQueueBusy}
                  onClick={startCheckpointTreeTwoVideos}
                >
                  <GitBranch size={16} /> {treeQueueBusy ? 'Queueing tree…' : 'Queue tree on both videos'}
                </button>
              </div>
              <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
                The API runs the orchestrator in the background (it must reach itself at{' '}
                <code style={{ color: 'var(--halo-cyan)' }}>PIPELINE_LAB_INTERNAL_URL</code>, default{' '}
                  <code style={{ color: 'var(--halo-cyan)' }}>{PIPELINE_LAB_LOCAL.apiOrigin}</code>).
              </p>
              {treeQueueError && <p style={{ margin: 0, color: '#f87171', fontSize: '0.88rem' }}>{treeQueueError}</p>}
            </div>
          )}
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
              {hasCheckpointTree && (
                <div
                  role="group"
                  aria-label="Batch layout"
                  style={{
                    display: 'inline-flex',
                    borderRadius: 10,
                    border: '1px solid var(--glass-border)',
                    overflow: 'hidden',
                  }}
                >
                  <button
                    type="button"
                    className="btn"
                    onClick={() => setBatchViewMode('list')}
                    style={{
                      borderRadius: 0,
                      border: 'none',
                      background: batchViewMode === 'list' ? 'rgba(34, 211, 238, 0.2)' : 'transparent',
                      color: batchViewMode === 'list' ? '#e2e8f0' : 'var(--text-muted)',
                    }}
                  >
                    <LayoutGrid size={16} aria-hidden /> List
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => setBatchViewMode('tree')}
                    style={{
                      borderRadius: 0,
                      border: 'none',
                      borderLeft: '1px solid var(--glass-border)',
                      background: batchViewMode === 'tree' ? 'rgba(34, 211, 238, 0.2)' : 'transparent',
                      color: batchViewMode === 'tree' ? '#e2e8f0' : 'var(--text-muted)',
                    }}
                  >
                    <GitBranch size={16} aria-hidden /> Tree
                  </button>
                </div>
              )}
              <button
                type="button"
                className="btn primary"
                disabled={!canCompare}
                onClick={() => nav(`/compare?runs=${encodeURIComponent(doneIds.join(','))}`)}
              >
                <Columns2 size={16} /> Compare all finished
              </button>
              {hasCheckpointTree && batchViewMode === 'tree' && (
                <button
                  type="button"
                  className="btn"
                  disabled={!canCompareSelected}
                  title={
                    canCompareSelected
                      ? 'Open Compare with the runs you checked in the tree'
                      : 'Check at least two finished runs in the tree (Compare column)'
                  }
                  onClick={() =>
                    nav(`/compare?runs=${encodeURIComponent(selectedDoneCompareIds.join(','))}`)
                  }
                >
                  <Columns2 size={16} /> Compare selected ({selectedDoneCompareIds.length})
                </button>
              )}
              {allTerminal && (
                <button type="button" className="btn" onClick={planAnotherBatch}>
                  <Plus size={16} /> New runs (same video)
                </button>
              )}
              <button
                type="button"
                className="btn"
                disabled={deletingAllRuns}
                style={{ color: '#f87171', borderColor: 'rgba(248,113,113,0.35)' }}
                onClick={() => void deleteAllSessionRuns()}
              >
                <Trash2 size={16} /> {deletingAllRuns ? 'Deleting…' : 'Delete all in batch'}
              </button>
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

          {hasCheckpointTree && batchViewMode === 'tree' && (
            <p
              style={{
                marginTop: '0.85rem',
                marginBottom: 0,
                fontSize: '0.88rem',
                color: 'var(--text-muted)',
                lineHeight: 1.5,
                maxWidth: 720,
              }}
            >
              Depth still matches checkpoint stages (roots = first stage); the canvas lays branches out like a tree so
              children sit under their parent instead of one tall triangle. Scroll horizontally and vertically in the
              desk area. Arrows show resume-from-parent relationships. Click a{' '}
              <strong style={{ color: 'var(--text-main)' }}>Phase N — compare column</strong>{' '}
              header to open Compare with every <strong style={{ color: 'var(--text-main)' }}>finished</strong> run in
              that phase and the phase preview mapped to that depth (override the clip from Compare’s View menu). Or use
              the <strong style={{ color: 'var(--text-main)' }}>Compare</strong> control on each run (not the card
              background — that highlights the path from root and any downstream children of the clicked run).{' '}
              <strong style={{ color: 'var(--text-main)' }}>Shift-click Compare</strong>{' '}
              selects a range; then <strong style={{ color: 'var(--text-main)' }}>Compare selected</strong>.
            </p>
          )}

          {batchViewMode === 'tree' && hasCheckpointTree ? (
            <BatchTreeView
              orderedIds={sessionRunIds}
              runMap={runMap}
              videoStemFallback={videoLabel.replace(/\.[^/.]+$/, '') || videoLabel}
              treeCompareSelected={treeCompareSelected}
              onToggleCompare={onTreeCompareToggle}
              onOpenConfig={setConfigRunId}
              onWatch={(runId) => nav(`/watch/${runId}`)}
              onStop={(runId) => void stopSessionRun(runId)}
              onRerun={(runId) => void rerunSessionRun(runId)}
              onDelete={(runId) => void deleteSessionRun(runId)}
              onDeleteSubtree={(rootId) => void deleteSessionRunSubtree(rootId)}
              stoppingRunId={stoppingRunId}
              rerunningRunId={rerunningRunId}
              deletingRunId={deletingRunId}
              onComparePhaseColumn={(colIdx, ids) => {
                const done = ids.filter((id) => runRows.find((r) => r.run_id === id)?.status === 'done')
                if (done.length < 2) {
                  window.alert(
                    `Need at least two finished runs in this phase to compare (this column has ${done.length}).`,
                  )
                  return
                }
                const view = compareViewModeForTreeColumn(colIdx)
                const params = new URLSearchParams()
                params.set('runs', done.join(','))
                if (view !== 'final') params.set('view', view)
                nav(`/compare?${params.toString()}`)
              }}
            />
          ) : (
            <div
              style={{
                marginTop: '1.25rem',
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 380px), 1fr))',
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
                    <>
                      <div
                        style={{
                          marginTop: '0.65rem',
                          padding: '0.65rem 0.75rem',
                          borderRadius: 10,
                          border: '1px solid rgba(34, 211, 238, 0.2)',
                          background: 'rgba(15, 23, 42, 0.45)',
                        }}
                      >
                        <div
                          style={{
                            fontSize: '0.62rem',
                            fontWeight: 700,
                            textTransform: 'uppercase',
                            letterSpacing: '0.06em',
                            color: '#94a3b8',
                            marginBottom: '0.35rem',
                          }}
                        >
                          Pipeline impact (manifest)
                        </div>
                        <PipelineImpactSummary
                          diagnostics={
                            run.manifest?.run_context_final?.pipeline_diagnostics as
                              | Record<string, unknown>
                              | undefined
                          }
                        />
                      </div>
                      <button
                        type="button"
                        className="btn primary"
                        style={{ marginTop: '0.65rem', width: '100%' }}
                        onClick={() => nav(`/watch/${run.run_id}`)}
                      >
                        <Play size={16} /> Watch results
                      </button>
                    </>
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
          )}
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
