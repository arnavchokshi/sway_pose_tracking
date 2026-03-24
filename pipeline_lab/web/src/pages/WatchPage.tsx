import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { API } from '../types'
import type { ProgressLine } from '../types'
import {
  WATCH_OVERLAY_LEGEND,
  WATCH_PHASE_CLIP_FILES,
  WATCH_PHASE_ROWS,
  type PruneEntry,
  type PruneLogPayload,
  type WatchPhaseId,
  basenamePath,
  formatBbox,
  formatTimecode,
  humanRule,
  parseBboxXyxy,
  pruneEntriesForWatchPhase,
  pruneEntryKey,
  pruneEntryRule,
  pruneEntryVisibleAtFrame,
  pruneOverlayStyle,
  sortPruneEntriesForUi,
} from '../lib/watchPrune'
import { ArrowLeft, Box, Clapperboard, Film, Search } from 'lucide-react'

type RunDetail = {
  run_id: string
  recipe_name?: string
  status?: string
  manifest?: {
    final_video_relpath?: string
    view_variants?: Record<string, string>
  }
}

const VARIANT_LABELS: Record<string, string> = {
  full: 'Full — boxes, IDs & heatmap skeleton',
  track_ids: 'Track IDs — boxes and labels only',
  skeleton: 'Skeleton only — heatmap, no boxes',
  segmentation_style: 'SAM-style — colored pixels only on hybrid-SAM frames + IDs',
}

function videoUrl(runId: string, outputRel: string) {
  const rel = outputRel.replace(/^\//, '')
  return `${API}/api/runs/${runId}/file/output/${rel}`
}

function pruneLogUrl(runId: string) {
  return `${API}/api/runs/${runId}/file/output/prune_log.json`
}

function clipFilename(rel: string): string {
  const i = rel.lastIndexOf('/')
  return i >= 0 ? rel.slice(i + 1) : rel
}

/** Map bbox from video pixel space onto the letterboxed `object-fit: contain` video element. */
function videoContainContentRect(video: HTMLVideoElement): {
  left: number
  top: number
  w: number
  h: number
  scale: number
} | null {
  const vw = video.videoWidth
  const vh = video.videoHeight
  const cw = video.clientWidth
  const ch = video.clientHeight
  if (!vw || !vh || !cw || !ch) return null
  const scale = Math.min(cw / vw, ch / vh)
  const dw = vw * scale
  const dh = vh * scale
  return {
    left: (cw - dw) / 2,
    top: (ch - dh) / 2,
    w: dw,
    h: dh,
    scale,
  }
}

function TierBBlock({ e }: { e: PruneEntry }) {
  const raw = e.tier_b_vote as
    | { weighted_sum?: number; prune_threshold?: number; rule_hits?: Record<string, number> }
    | undefined
  if (!raw) return null
  const hits = raw.rule_hits ?? {}
  const lines = Object.entries(hits)
    .filter(([, v]) => Number(v) > 0)
    .map(([k]) => k.replace(/^prune_/, '').replace(/_tracks$/, ''))
  return (
    <div style={{ marginTop: '0.45rem', fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
      <div>
        Weighted sum <span style={{ color: '#e2e8f0' }}>{raw.weighted_sum}</span>
        {raw.prune_threshold != null && (
          <>
            {' '}
            vs threshold <span style={{ color: '#e2e8f0' }}>{raw.prune_threshold}</span>
          </>
        )}
      </div>
      {lines.length > 0 && <div>Rules firing: {lines.join(', ')}</div>}
    </div>
  )
}

function PruneEventCard({
  e,
  fps,
  onSeek,
  selected,
  onToggleHighlight,
}: {
  e: PruneEntry
  fps: number
  onSeek: (frame: number) => void
  selected: boolean
  onToggleHighlight: (entry: PruneEntry) => void
}) {
  const rule = String(e.rule ?? 'unknown')
  if (rule === 'phase6_summary') {
    return (
      <div
        style={{
          padding: '0.65rem 0.75rem',
          borderRadius: 10,
          background: 'rgba(14, 165, 233, 0.08)',
          border: '1px solid rgba(14, 165, 233, 0.25)',
          fontSize: '0.78rem',
          color: 'var(--text-muted)',
          lineHeight: 1.5,
        }}
      >
        <div style={{ fontWeight: 600, color: '#7dd3fc', marginBottom: '0.25rem' }}>Collision phase summary</div>
        <div>Dedup removed pose rows: {String(e.dedup_removed_poses ?? '—')}</div>
        <div>Sanitize removed poses: {String(e.sanitize_removed_poses ?? '—')}</div>
        <div>Keypoints zeroed (limb): {String(e.sanitize_keypoints_zeroed ?? '—')}</div>
        <div>Per-frame event rows logged: {String(e.per_event_log_count ?? '—')}</div>
      </div>
    )
  }

  const tid = e.track_id != null ? String(e.track_id) : '—'
  const frameIdx = typeof e.frame_idx === 'number' ? e.frame_idx : null
  const fr = e.frame_range as number[] | undefined
  const rangeLabel =
    fr && fr.length >= 2 ? `Frames ${fr[0]}–${fr[1]}` : fr && fr.length >= 1 ? `From frame ${fr[0]}` : null
  const seekFrame = frameIdx ?? (fr && fr.length >= 1 ? fr[0] : null)

  const bboxStr = formatBbox(e)
  const title = humanRule(rule)
  const canHighlight = rule !== 'phase6_summary' && parseBboxXyxy(e) != null

  return (
    <button
      type="button"
      onClick={() => {
        if (seekFrame != null) onSeek(seekFrame)
        if (canHighlight) onToggleHighlight(e)
      }}
      disabled={seekFrame == null && !canHighlight}
      style={{
        display: 'block',
        width: '100%',
        textAlign: 'left',
        padding: '0.7rem 0.8rem',
        borderRadius: 10,
        background: selected ? 'rgba(14, 165, 233, 0.12)' : 'rgba(0,0,0,0.35)',
        border: selected ? '2px solid var(--halo-cyan)' : '1px solid var(--glass-border)',
        color: 'inherit',
        font: 'inherit',
        cursor: seekFrame != null || canHighlight ? 'pointer' : 'default',
        marginBottom: '0.5rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '0.5rem' }}>
        <span style={{ fontWeight: 700, color: '#f8fafc', fontSize: '0.88rem' }}>Track {tid}</span>
        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
          {frameIdx != null ? (
            <>
              f{frameIdx} · {formatTimecode(frameIdx, fps)}
            </>
          ) : rangeLabel ? (
            <>
              {rangeLabel}
              {fps > 0 && fr && fr.length >= 1 ? ` · ${formatTimecode(fr[0], fps)}` : ''}
            </>
          ) : (
            '—'
          )}
        </span>
      </div>
      <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: '0.2rem' }}>{title}</div>
      {String(e.decision ?? '') && (
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
          Decision: {String(e.decision)}
        </div>
      )}
      {bboxStr && (
        <div style={{ fontSize: '0.7rem', color: '#a5b4fc', marginTop: '0.25rem', fontFamily: 'ui-monospace, monospace' }}>
          Bbox xyxy {bboxStr}
        </div>
      )}
      {e.n_frames != null && rule !== 'deduplicate_collocated_poses' && rule !== 'sanitize_pose_bbox_consistency' && (
        <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
          Detections on track: {String(e.n_frames)}
          {e.pos_pct != null && Array.isArray(e.pos_pct) && (
            <span>
              {' '}
              · median pos {e.pos_pct[0]}%, {e.pos_pct[1]}%
            </span>
          )}
        </div>
      )}
      <TierBBlock e={e} />
      {seekFrame != null && (
        <div style={{ fontSize: '0.65rem', color: '#64748b', marginTop: '0.35rem' }}>Click to seek video</div>
      )}
      {canHighlight && (
        <div style={{ fontSize: '0.65rem', color: selected ? '#7dd3fc' : '#64748b', marginTop: '0.25rem' }}>
          {selected ? 'Highlight on video (this stage) — click again to clear' : 'Click to show bbox overlay on video'}
        </div>
      )}
    </button>
  )
}

export function WatchPage() {
  const { id } = useParams<{ id: string }>()
  const nav = useNavigate()
  const videoRef = useRef<HTMLVideoElement>(null)
  const videoWrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [run, setRun] = useState<RunDetail | null>(null)
  const [progress, setProgress] = useState<ProgressLine[]>([])
  const [title, setTitle] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [mode, setMode] = useState<'phase' | 'final'>('phase')
  const [phasePick, setPhasePick] = useState(0)
  const [finalKey, setFinalKey] = useState<string>('full')
  const [pruneLog, setPruneLog] = useState<PruneLogPayload | null>(null)
  const [pruneLogErr, setPruneLogErr] = useState<string | null>(null)
  const [pruneQuery, setPruneQuery] = useState('')
  const [selectedPruneKey, setSelectedPruneKey] = useState<string | null>(null)
  const initialModeSet = useRef(false)

  useEffect(() => {
    initialModeSet.current = false
  }, [id])

  useEffect(() => {
    if (!id) return
    let cancelled = false
    setErr(null)
    setRun(null)
    setProgress([])
    setPruneLog(null)
    setPruneLogErr(null)

    const load = () => {
      Promise.all([
        fetch(`${API}/api/runs/${id}`).then(async (r) => {
          if (!r.ok) throw new Error(`Run HTTP ${r.status}`)
          return r.json() as Promise<RunDetail>
        }),
        fetch(`${API}/api/runs/${id}/progress`).then(async (r) => {
          if (!r.ok) return [] as ProgressLine[]
          return r.json() as Promise<ProgressLine[]>
        }),
      ])
        .then(([data, prog]) => {
          if (cancelled) return
          setRun(data)
          setTitle(data.recipe_name || data.run_id.slice(0, 8))
          setProgress(Array.isArray(prog) ? prog : [])
        })
        .catch((e) => {
          if (!cancelled) setErr(e instanceof Error ? e.message : String(e))
        })
    }

    load()
    const t = setInterval(load, 5000)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [id])

  useEffect(() => {
    if (!id) return
    let cancelled = false
    fetch(pruneLogUrl(id))
      .then((r) => {
        if (r.status === 404) return null
        if (!r.ok) throw new Error(`prune log HTTP ${r.status}`)
        return r.json() as Promise<PruneLogPayload>
      })
      .then((j) => {
        if (cancelled) return
        if (j) setPruneLog(j)
        else setPruneLog(null)
        setPruneLogErr(null)
      })
      .catch((e) => {
        if (!cancelled) {
          setPruneLog(null)
          setPruneLogErr(e instanceof Error ? e.message : String(e))
        }
      })
    return () => {
      cancelled = true
    }
  }, [id, run?.status])

  const phaseTabs = useMemo(() => {
    const allowed = new Set(WATCH_PHASE_CLIP_FILES as unknown as string[])
    const seen = new Set<string>()
    const have = new Set<string>()
    for (const p of progress) {
      const rel = p.preview_relpath
      if (!rel || rel.includes('_poses')) continue
      const fn = clipFilename(rel)
      if (!allowed.has(fn) || seen.has(fn)) continue
      seen.add(fn)
      have.add(fn)
    }
    return WATCH_PHASE_ROWS.filter((row) => have.has(row.file)).map((row) => ({
      ...row,
      src: videoUrl(id!, `phase_previews/${row.file}`),
    }))
  }, [progress, id])

  const finalVariants = useMemo(() => {
    const m = run?.manifest
    const vv = m?.view_variants
    if (vv && Object.keys(vv).length > 0) {
      return Object.entries(vv).map(([key, rel]) => ({
        key,
        label: VARIANT_LABELS[key] || key,
        rel,
        src: videoUrl(id!, rel),
      }))
    }
    const rel = m?.final_video_relpath
    if (rel) {
      return [{ key: 'full', label: VARIANT_LABELS.full, rel, src: videoUrl(id!, rel) }]
    }
    return []
  }, [run, id])

  const isDone = run?.status === 'done'
  const activePhase = phaseTabs[phasePick]
  const activeSrc =
    mode === 'phase' ? activePhase?.src ?? null : finalVariants.find((v) => v.key === finalKey)?.src ?? finalVariants[0]?.src ?? null

  const nativeFps = pruneLog?.native_fps && pruneLog.native_fps > 0 ? pruneLog.native_fps : 30

  const showPruneAside = mode === 'phase' && Boolean(activePhase)
  const gridTemplateColumns =
    mode === 'final'
      ? 'minmax(0, 1fr) minmax(260px, 340px)'
      : showPruneAside
        ? 'minmax(0, 1fr) minmax(280px, 380px)'
        : '1fr'

  const seekToFrame = useCallback(
    (frame: number) => {
      const el = videoRef.current
      if (!el || !nativeFps) return
      const t = Math.max(0, frame / nativeFps)
      el.currentTime = t
      void el.play().catch(() => {})
    },
    [nativeFps],
  )

  const onTogglePruneHighlight = useCallback((entry: PruneEntry) => {
    if (pruneEntryRule(entry) === 'phase6_summary') return
    if (!parseBboxXyxy(entry)) return
    const k = pruneEntryKey(entry)
    setSelectedPruneKey((prev) => (prev === k ? null : k))
  }, [])

  useEffect(() => {
    if (mode === 'phase' && phasePick >= phaseTabs.length) setPhasePick(0)
  }, [mode, phaseTabs.length, phasePick])

  useEffect(() => {
    if (mode === 'final' && finalVariants.length && !finalVariants.some((v) => v.key === finalKey)) {
      setFinalKey(finalVariants[0].key)
    }
  }, [mode, finalVariants, finalKey])

  useEffect(() => {
    if (!run || initialModeSet.current) return
    const hasPhase = phaseTabs.length > 0
    const hasFinal = finalVariants.length > 0
    if (!hasPhase && !hasFinal) return
    setMode(hasPhase ? 'phase' : 'final')
    initialModeSet.current = true
  }, [run, phaseTabs.length, finalVariants.length])

  const currentPhaseId: WatchPhaseId | null = activePhase?.id ?? null
  const phasePruneEntries = useMemo(() => {
    if (!currentPhaseId) return []
    const raw = pruneLog?.prune_entries ?? []
    return [...pruneEntriesForWatchPhase(currentPhaseId, raw)].sort(sortPruneEntriesForUi)
  }, [currentPhaseId, pruneLog?.prune_entries])

  const filteredPrune = useMemo(() => {
    const q = pruneQuery.trim().toLowerCase()
    if (!q) return phasePruneEntries
    return phasePruneEntries.filter((e) => {
      const blob = JSON.stringify(e).toLowerCase()
      return blob.includes(q)
    })
  }, [phasePruneEntries, pruneQuery])

  const highlightedPruneEntry = useMemo(() => {
    if (!selectedPruneKey) return null
    return phasePruneEntries.find((e) => pruneEntryKey(e) === selectedPruneKey) ?? null
  }, [phasePruneEntries, selectedPruneKey])

  useEffect(() => {
    setSelectedPruneKey(null)
  }, [phasePick, mode])

  useEffect(() => {
    if (!selectedPruneKey) return
    const stillVisible = filteredPrune.some((e) => pruneEntryKey(e) === selectedPruneKey)
    if (!stillVisible) setSelectedPruneKey(null)
  }, [filteredPrune, selectedPruneKey])

  const redrawPruneOverlay = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const wrap = videoWrapRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || !wrap) return

    const dpr = window.devicePixelRatio || 1
    const cw = Math.max(1, Math.floor(wrap.clientWidth))
    const ch = Math.max(1, Math.floor(wrap.clientHeight))
    canvas.width = Math.floor(cw * dpr)
    canvas.height = Math.floor(ch * dpr)
    canvas.style.width = `${cw}px`
    canvas.style.height = `${ch}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cw, ch)

    if (mode !== 'phase' || !highlightedPruneEntry || !video) return

    const frame = video.currentTime * nativeFps
    if (!pruneEntryVisibleAtFrame(highlightedPruneEntry, frame)) return

    const bbox = parseBboxXyxy(highlightedPruneEntry)
    if (!bbox) return

    const content = videoContainContentRect(video)
    if (!content) return

    const [x1, y1, x2, y2] = bbox
    const px1 = content.left + x1 * content.scale
    const py1 = content.top + y1 * content.scale
    const px2 = content.left + x2 * content.scale
    const py2 = content.top + y2 * content.scale

    const rule = pruneEntryRule(highlightedPruneEntry)
    const { stroke, fill, label } = pruneOverlayStyle(rule)

    const w = px2 - px1
    const h = py2 - py1
    if (w < 2 || h < 2) return

    ctx.fillStyle = fill
    ctx.fillRect(px1, py1, w, h)
    ctx.strokeStyle = stroke
    ctx.lineWidth = 3
    ctx.strokeRect(px1, py1, w, h)

    ctx.font = '600 13px system-ui, sans-serif'
    const tid = highlightedPruneEntry.track_id != null ? ` id ${highlightedPruneEntry.track_id}` : ''
    const cap = `${label}${tid}`
    const pad = 6
    const tw = Math.min(ctx.measureText(cap).width + pad * 2, cw - px1 - 4)
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)'
    ctx.fillRect(px1, Math.max(0, py1 - 26), tw, 22)
    ctx.fillStyle = '#f8fafc'
    ctx.fillText(cap, px1 + pad, Math.max(14, py1 - 10))
  }, [mode, highlightedPruneEntry, nativeFps])

  useEffect(() => {
    redrawPruneOverlay()
  }, [redrawPruneOverlay, activeSrc, highlightedPruneEntry])

  useEffect(() => {
    const video = videoRef.current
    const wrap = videoWrapRef.current
    if (!video || !wrap) return
    const onFrame = () => redrawPruneOverlay()
    video.addEventListener('timeupdate', onFrame)
    video.addEventListener('loadedmetadata', onFrame)
    video.addEventListener('seeked', onFrame)
    const ro = new ResizeObserver(() => redrawPruneOverlay())
    ro.observe(wrap)
    return () => {
      video.removeEventListener('timeupdate', onFrame)
      video.removeEventListener('loadedmetadata', onFrame)
      video.removeEventListener('seeked', onFrame)
      ro.disconnect()
    }
  }, [redrawPruneOverlay, activeSrc])

  const runSummary = useMemo(() => {
    if (!pruneLog) return null
    const w = pruneLog.frame_width
    const h = pruneLog.frame_height
    const tf = pruneLog.total_frames
    const fps = pruneLog.native_fps
    const tr = pruneLog.tracker
    return {
      videoLabel: pruneLog.video_path ? basenamePath(pruneLog.video_path) : null,
      resolution: w && h ? `${w}×${h}` : null,
      frames: tf,
      fps,
      tracksIn: tr?.count,
    }
  }, [pruneLog])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxWidth: 1400, margin: '0 auto' }}>
      <header
        className="glass-panel"
        style={{
          padding: '0.75rem 1.25rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button type="button" className="btn" onClick={() => nav(-1)}>
            <ArrowLeft size={18} /> Back
          </button>
          <div>
            <h1 style={{ fontSize: '1.2rem', margin: 0, fontWeight: 600 }}>{title || 'Output'}</h1>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              {id?.slice(0, 8)}…{run?.status ? ` · ${run.status}` : ''}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            className={`btn ${mode === 'phase' ? 'primary' : ''}`}
            onClick={() => setMode('phase')}
            disabled={phaseTabs.length === 0}
          >
            <Clapperboard size={16} /> Pipeline
          </button>
          <button
            type="button"
            className={`btn ${mode === 'final' ? 'primary' : ''}`}
            onClick={() => setMode('final')}
            disabled={finalVariants.length === 0}
          >
            <Film size={16} /> Final render
          </button>
        </div>
      </header>

      {err && (
        <div className="glass-panel" style={{ padding: '1rem', color: '#f87171' }}>
          {err}
        </div>
      )}

      {!err && run && mode === 'phase' && phaseTabs.length > 0 && (
        <div className="glass-panel" style={{ padding: '0.65rem 1rem' }}>
          <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Stage preview
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
            {phaseTabs.map((tab, i) => (
              <button
                key={tab.file}
                type="button"
                className={`btn ${i === phasePick ? 'primary' : ''}`}
                style={{ fontSize: '0.82rem', padding: '0.45rem 0.75rem', borderRadius: 999 }}
                onClick={() => setPhasePick(i)}
              >
                {tab.title}
              </button>
            ))}
          </div>
          {activePhase && (
            <p style={{ margin: '0.6rem 0 0', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5, maxWidth: 720 }}>
              {activePhase.blurb} Full-length clip, no title card. Prune details → right panel.
            </p>
          )}
        </div>
      )}

      {!err && run && mode === 'final' && (
        <div className="glass-panel" style={{ padding: '0.85rem 1rem' }}>
          <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Export variant
          </div>
          {!isDone && finalVariants.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', margin: 0 }}>Final videos appear when the run completes.</p>
          ) : finalVariants.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', margin: 0 }}>No final video in manifest.</p>
          ) : (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
              {finalVariants.map((v) => (
                <button
                  key={v.key}
                  type="button"
                  className={`btn ${v.key === finalKey ? 'primary' : ''}`}
                  style={{ fontSize: '0.82rem', borderRadius: 999 }}
                  onClick={() => setFinalKey(v.key)}
                >
                  {v.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns,
          gap: '0.75rem',
          alignItems: 'start',
        }}
        className="watch-page-grid"
      >
        <div className="glass-panel" style={{ padding: 0, overflow: 'hidden', background: '#000' }}>
          {activeSrc ? (
            <>
              <div
                ref={videoWrapRef}
                style={{ position: 'relative', width: '100%', lineHeight: 0 }}
              >
                <video
                  ref={videoRef}
                  key={activeSrc}
                  src={activeSrc}
                  controls
                  playsInline
                  style={{
                    width: '100%',
                    maxHeight: 'min(72vh, 820px)',
                    display: 'block',
                    margin: '0 auto',
                    objectFit: 'contain',
                  }}
                />
                <canvas
                  ref={canvasRef}
                  aria-hidden
                  style={{
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                  }}
                />
              </div>
              {showPruneAside &&
                (currentPhaseId === 'pre_pose' ||
                  currentPhaseId === 'collision' ||
                  currentPhaseId === 'post_pose') && (
                  <div
                    style={{
                      padding: '0.5rem 0.85rem 0.65rem',
                      fontSize: '0.72rem',
                      color: 'var(--text-muted)',
                      lineHeight: 1.5,
                      borderTop: '1px solid rgba(148, 163, 184, 0.2)',
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: '0.5rem 1rem',
                      alignItems: 'center',
                    }}
                  >
                    <span style={{ fontWeight: 600, color: '#94a3b8' }}>Prune overlay</span>
                    {WATCH_OVERLAY_LEGEND.map((row) => (
                      <span key={row.label} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        <span
                          style={{
                            width: 10,
                            height: 10,
                            borderRadius: 2,
                            background: row.color,
                            boxShadow: `0 0 0 1px ${row.color}`,
                          }}
                        />
                        {row.label}
                      </span>
                    ))}
                  </div>
                )}
            </>
          ) : (
            !err && (
              <div style={{ color: 'var(--text-muted)', padding: '2.5rem', textAlign: 'center' }}>
                {run?.status === 'running' || run?.status === 'queued'
                  ? 'Run in progress — previews will appear here.'
                  : 'Nothing to play for this selection.'}
              </div>
            )
          )}
        </div>

        {showPruneAside && activePhase && (
          <aside className="glass-panel" style={{ padding: '0.85rem 1rem', maxHeight: 'min(78vh, 860px)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
              Run data
            </div>
            {runSummary && (
              <div
                style={{
                  fontSize: '0.78rem',
                  color: '#cbd5e1',
                  lineHeight: 1.55,
                  marginBottom: '0.75rem',
                  padding: '0.55rem 0.65rem',
                  borderRadius: 8,
                  background: 'rgba(0,0,0,0.35)',
                  border: '1px solid var(--glass-border)',
                }}
              >
                {runSummary.videoLabel && (
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Video </span>
                    {runSummary.videoLabel}
                  </div>
                )}
                {runSummary.resolution && (
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Resolution </span>
                    {runSummary.resolution}
                  </div>
                )}
                {runSummary.frames != null && (
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Frames </span>
                    {runSummary.frames}
                    {runSummary.fps != null && (
                      <span style={{ color: 'var(--text-muted)' }}> @ {runSummary.fps} fps</span>
                    )}
                  </div>
                )}
                {runSummary.tracksIn != null && (
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Tracks (pre–pre-pose) </span>
                    {runSummary.tracksIn}
                  </div>
                )}
              </div>
            )}
            {!pruneLog && !pruneLogErr && (
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Loading prune log…</div>
            )}
            {pruneLogErr && (
              <div style={{ fontSize: '0.76rem', color: '#fca5a5' }}>{pruneLogErr}</div>
            )}

            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', margin: '0.5rem 0' }}>
              {currentPhaseId === 'track' && 'After this stage'}
              {currentPhaseId === 'pre_pose' && 'Removals this stage'}
              {currentPhaseId === 'pose' && 'Pose stage'}
              {currentPhaseId === 'collision' && 'Collision & sanitize'}
              {currentPhaseId === 'post_pose' && 'Post-pose removals'}
            </div>

            {currentPhaseId === 'track' && pruneLog?.tracker && (
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: '0 0 0.5rem' }}>
                Per-track pruning starts in <strong style={{ color: '#e2e8f0' }}>Pre-pose pruning</strong>. Here you should see{' '}
                {pruneLog.tracker.count ?? '—'} raw tracks after stitch. Full JSON:{' '}
                <code style={{ fontSize: '0.85em' }}>output/prune_log.json</code>.
              </p>
            )}

            {currentPhaseId === 'pose' && (
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: '0 0 0.5rem' }}>
                No track-level prune rows are logged during ViTPose itself. Occluded / skipped frames are reflected in the
                video and in Lab phase metadata. Tune pose stride in the recipe if needed.
              </p>
            )}

            {(currentPhaseId === 'pre_pose' || currentPhaseId === 'collision' || currentPhaseId === 'post_pose') && (
              <>
                <div style={{ position: 'relative', marginBottom: '0.5rem' }}>
                  <Search
                    size={14}
                    style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: '#64748b' }}
                  />
                  <input
                    type="search"
                    className="text-input"
                    placeholder="Filter by track id, rule, bbox…"
                    value={pruneQuery}
                    onChange={(e) => setPruneQuery(e.target.value)}
                    style={{ width: '100%', paddingLeft: 34, fontSize: '0.8rem' }}
                  />
                </div>
                <div style={{ overflowY: 'auto', flex: 1, paddingRight: 4 }}>
                  {filteredPrune.length === 0 ? (
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>No matching events for this stage.</div>
                  ) : (
                    filteredPrune.map((e, idx) => (
                      <PruneEventCard
                        key={`${pruneEntryKey(e)}-${idx}`}
                        e={e}
                        fps={nativeFps}
                        onSeek={seekToFrame}
                        selected={pruneEntryKey(e) === selectedPruneKey}
                        onToggleHighlight={onTogglePruneHighlight}
                      />
                    ))
                  )}
                </div>
              </>
            )}
          </aside>
        )}

        {mode === 'final' && (
          <aside className="glass-panel" style={{ padding: '0.85rem 1rem' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Full export. SAM-style uses colored fills on hybrid-SAM frames only. Open <strong>Pipeline</strong> for stage
              clips and prune cards; complete audit trail stays in <code style={{ fontSize: '0.85em' }}>prune_log.json</code>.
            </div>
            {id && (
              <a
                href={`/pose_3d_viewer.html?runId=${encodeURIComponent(id)}`}
                target="_blank"
                rel="noreferrer"
                className="btn-secondary"
                style={{
                  marginTop: '0.75rem',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8,
                  textDecoration: 'none',
                  fontSize: '0.8rem',
                }}
              >
                <Box size={16} />
                View 3D poses
              </a>
            )}
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: 1.45 }}>
              Opens when <code style={{ fontSize: '0.85em' }}>data.json</code> includes <code style={{ fontSize: '0.85em' }}>pose_3d</code>{' '}
              (3D lift enabled in recipe).
            </p>
          </aside>
        )}
      </div>

      <style>{`
        @media (max-width: 960px) {
          .watch-page-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  )
}
