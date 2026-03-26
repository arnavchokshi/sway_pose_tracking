import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
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
  parseOtherBboxXyxy,
  pruneEntriesForWatchPhase,
  pruneEntryKey,
  pruneEntryRule,
  pruneEntryVisibleAtFrame,
  sortPruneEntriesForUi,
  hybridSamRoiMap,
} from '../lib/watchPrune'
import { WATCH_PHASE_TUNING_FIELD_IDS } from '../lib/watchPhaseTuning'
import { formatConfigValue } from '../lib/formatConfigValue'
import { videoContainContentRect } from '../lib/videoContentRect'
import { sam2RoiLayerStyles } from '../lib/sam2RoiOverlay'
import { ArrowLeft, Box, Clapperboard, Film, RotateCw, Search, Gauge } from 'lucide-react'
import { TrackQualitySummary, PipelineImpactReport, FriendlyRunConfig, RunOverviewStrip } from '../components/RunMetrics'
import { CauseConfigBlock, TierBBlock } from '../components/PruneInspectBlocks'
import type { Schema } from '../types'

type RunDetail = {
  run_id: string
  recipe_name?: string
  status?: string
  subprocess_alive?: boolean
  manifest?: {
    final_video_relpath?: string
    view_variants?: Record<string, string>
    /** Subset of SWAY_* env written for run diffing (see Lab manifest). */
    env?: Record<string, unknown>
    run_context_final?: {
      track_summary?: Record<string, unknown>
      fields?: Record<string, unknown>
      pipeline_diagnostics?: Record<string, unknown>
      recipe_name?: string
    }
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

function WatchPhaseTuningBlock({
  phaseId,
  fields,
  schema,
}: {
  phaseId: WatchPhaseId
  fields: Record<string, unknown>
  schema: Schema | null
}) {
  const ids = WATCH_PHASE_TUNING_FIELD_IDS[phaseId]
  const labelById = new Map(schema?.fields.map((f) => [f.id, f.label]) ?? [])
  const rows = ids
    .map((id) => ({ id, label: labelById.get(id) ?? id, raw: fields[id] }))
    .filter((r) => Object.prototype.hasOwnProperty.call(fields, r.id))
  if (rows.length === 0) {
    return (
      <div
        style={{
          marginBottom: '0.75rem',
          padding: '0.5rem 0.65rem',
          borderRadius: 8,
          background: 'rgba(15, 23, 42, 0.35)',
          border: '1px solid rgba(148, 163, 184, 0.18)',
        }}
      >
        <div
          style={{
            fontSize: '0.65rem',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            color: '#64748b',
            marginBottom: '0.35rem',
          }}
        >
          Recipe knobs for this stage
        </div>
        <p style={{ margin: 0, fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
          {!schema?.fields?.length ? (
            <>
              Start the Lab API so <code style={{ fontSize: '0.65rem' }}>/api/schema</code> loads field labels.
            </>
          ) : (
            <>
              No settings from your recipe map to this preview (defaults only, or not in the run snapshot). Open{' '}
              <strong style={{ color: '#94a3b8' }}>Metrics &amp; Config</strong> for the full effective configuration.
            </>
          )}
        </p>
      </div>
    )
  }
  return (
    <div
      style={{
        marginBottom: '0.75rem',
        padding: '0.55rem 0.65rem',
        borderRadius: 8,
        background: 'rgba(14, 165, 233, 0.06)',
        border: '1px solid rgba(14, 165, 233, 0.22)',
      }}
    >
      <div
        style={{
          fontSize: '0.65rem',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          color: '#7dd3fc',
          marginBottom: '0.45rem',
        }}
      >
        Recipe knobs for this stage
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', maxHeight: 220, overflowY: 'auto' }}>
        {rows.map((r) => (
          <div
            key={r.id}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'baseline',
              gap: '0.5rem',
              fontSize: '0.72rem',
              lineHeight: 1.4,
            }}
            title={r.id}
          >
            <span style={{ color: 'var(--text-muted)', flex: '1 1 auto' }}>{r.label}</span>
            <span style={{ color: '#e2e8f0', fontFamily: 'ui-monospace, monospace', flexShrink: 0 }}>
              {formatConfigValue(r.raw)}
            </span>
          </div>
        ))}
      </div>
      {!schema?.fields?.length && (
        <div style={{ fontSize: '0.65rem', color: '#94a3b8', marginTop: '0.35rem', lineHeight: 1.4 }}>
          Start the Lab API (<code style={{ fontSize: '0.6rem' }}>uvicorn …</code>) so labels load from{' '}
          <code style={{ fontSize: '0.6rem' }}>/api/schema</code>.
        </div>
      )}
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
  const otherBboxStr =
    rule === 'deduplicate_collocated_poses' ? formatBbox({ ...e, bbox_xyxy: e.other_bbox_xyxy } as PruneEntry) : null
  const keptTid =
    rule === 'deduplicate_collocated_poses' && e.kept_track_id != null ? String(e.kept_track_id) : null
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
      <CauseConfigBlock e={e} />
      {keptTid != null && (
        <div style={{ fontSize: '0.74rem', color: '#86efac', marginTop: '0.25rem', lineHeight: 1.45 }}>
          Same frame: kept track <span style={{ fontWeight: 700 }}>{keptTid}</span> (this row is the duplicate that was
          removed).
        </div>
      )}
      {String(e.decision ?? '') && (
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
          Decision: {String(e.decision)}
        </div>
      )}
      {rule === 'deduplicate_collocated_poses' &&
        (typeof e.bbox_iou === 'number' ||
          typeof e.median_kpt_dist_px === 'number' ||
          typeof e.median_dist_over_bbox_h === 'number') && (
          <div
            style={{
              fontSize: '0.68rem',
              color: 'var(--text-muted)',
              marginTop: '0.2rem',
              lineHeight: 1.45,
              fontFamily: 'ui-monospace, monospace',
            }}
          >
            {typeof e.bbox_iou === 'number' && <div>BBox IoU {Number(e.bbox_iou).toFixed(3)}</div>}
            {typeof e.median_kpt_dist_px === 'number' && typeof e.median_dist_over_bbox_h === 'number' && (
              <div>
                Median kpt distance {Number(e.median_kpt_dist_px).toFixed(1)} px (
                {(Number(e.median_dist_over_bbox_h) * 100).toFixed(1)}% of bbox height)
              </div>
            )}
            {typeof e.torso_median_dist_px === 'number' &&
              typeof e.torso_median_over_bbox_h === 'number' && (
                <div>
                  Torso (shldr+hip) median {Number(e.torso_median_dist_px).toFixed(1)} px (
                  {(Number(e.torso_median_over_bbox_h) * 100).toFixed(1)}% of bbox height)
                </div>
              )}
            {typeof e.dedup_pair_oks === 'number' && (
              <div>Pairwise OKS {Number(e.dedup_pair_oks).toFixed(3)}</div>
            )}
            {typeof e.mean_kpt_conf_suppressed === 'number' && typeof e.mean_kpt_conf_kept === 'number' && (
              <div>
                Mean kpt conf — suppressed {Number(e.mean_kpt_conf_suppressed).toFixed(3)}, kept{' '}
                {Number(e.mean_kpt_conf_kept).toFixed(3)}
              </div>
            )}
            {typeof e.tie_break === 'string' && (
              <div>
                Why kept:{' '}
                {e.tie_break === 'longer_track_history'
                  ? 'longer track history'
                  : e.tie_break === 'higher_mean_kpt_conf'
                    ? 'higher mean keypoint confidence'
                    : e.tie_break === 'lower_track_id_tiebreak'
                      ? 'lower track ID (equal frame counts — avoids ghost winning then pruned)'
                      : String(e.tie_break)}
              </div>
            )}
          </div>
        )}
      {bboxStr && (
        <div style={{ fontSize: '0.7rem', color: '#fdba74', marginTop: '0.25rem', fontFamily: 'ui-monospace, monospace' }}>
          Suppressed bbox xyxy {bboxStr}
        </div>
      )}
      {otherBboxStr && (
        <div style={{ fontSize: '0.7rem', color: '#86efac', marginTop: '0.15rem', fontFamily: 'ui-monospace, monospace' }}>
          Kept (other ID) bbox xyxy {otherBboxStr}
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
          {selected
            ? 'Focus zoom on this region — click again to clear'
            : 'Also click to focus-zoom (boxes are baked into the phase preview video)'}
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
  const [run, setRun] = useState<RunDetail | null>(null)
  const [progress, setProgress] = useState<ProgressLine[]>([])
  const [title, setTitle] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [mode, setMode] = useState<'phase' | 'final' | 'metrics'>('phase')
  const [phasePick, setPhasePick] = useState(0)
  const [finalKey, setFinalKey] = useState<string>('full')
  const [pruneLog, setPruneLog] = useState<PruneLogPayload | null>(null)
  const [pruneLogErr, setPruneLogErr] = useState<string | null>(null)
  const [pruneQuery, setPruneQuery] = useState('')
  const [selectedPruneKey, setSelectedPruneKey] = useState<string | null>(null)
  const [videoFocusStyle, setVideoFocusStyle] = useState<CSSProperties>({})
  const [rerunning, setRerunning] = useState(false)
  /** Playback frame index (for hybrid SAM ROI overlay on track phase). */
  const [playFrame, setPlayFrame] = useState(0)
  /** Bumps when the video wrapper resizes so SAM2 overlay rescales with letterbox. */
  const [samOverlayLayout, setSamOverlayLayout] = useState(0)
  const initialModeSet = useRef(false)
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
          setTitle(
            data.recipe_name ||
              (data.manifest?.run_context_final?.recipe_name as string | undefined) ||
              data.run_id.slice(0, 8),
          )
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
    mode === 'phase'
      ? activePhase?.src ?? null
      : mode === 'final'
        ? finalVariants.find((v) => v.key === finalKey)?.src ?? finalVariants[0]?.src ?? null
        : null

  const nativeFps = pruneLog?.native_fps && pruneLog.native_fps > 0 ? pruneLog.native_fps : 30

  const samRoiMap = useMemo(
    () => hybridSamRoiMap(pruneLog?.hybrid_sam_frame_rois),
    [pruneLog?.hybrid_sam_frame_rois],
  )

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
      const wasPlaying = !el.paused
      const t = Math.max(0, frame / nativeFps)
      el.currentTime = t
      if (wasPlaying) void el.play().catch(() => {})
      else el.pause()
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

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const bump = () => setPlayFrame(Math.max(0, Math.floor(v.currentTime * nativeFps)))
    bump()
    v.addEventListener('timeupdate', bump)
    v.addEventListener('seeked', bump)
    v.addEventListener('loadedmetadata', bump)
    return () => {
      v.removeEventListener('timeupdate', bump)
      v.removeEventListener('seeked', bump)
      v.removeEventListener('loadedmetadata', bump)
    }
  }, [activeSrc, nativeFps])

  useEffect(() => {
    const w = videoWrapRef.current
    if (!w) return
    const ro = new ResizeObserver(() => setSamOverlayLayout((n) => n + 1))
    ro.observe(w)
    return () => ro.disconnect()
  }, [activeSrc])

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

  const updatePruneFocusZoom = useCallback(() => {
    const video = videoRef.current
    if (!video || mode !== 'phase' || !highlightedPruneEntry) {
      setVideoFocusStyle({})
      return
    }
    const frame = video.currentTime * nativeFps
    if (!pruneEntryVisibleAtFrame(highlightedPruneEntry, frame)) {
      setVideoFocusStyle({})
      return
    }
    const main = parseBboxXyxy(highlightedPruneEntry)
    if (!main) {
      setVideoFocusStyle({})
      return
    }
    const rule = pruneEntryRule(highlightedPruneEntry)
    const other = rule === 'deduplicate_collocated_poses' ? parseOtherBboxXyxy(highlightedPruneEntry) : null
    let cx = (main[0] + main[2]) / 2
    let cy = (main[1] + main[3]) / 2
    if (other) {
      cx = (cx + (other[0] + other[2]) / 2) / 2
      cy = (cy + (other[1] + other[3]) / 2) / 2
    }
    const content = videoContainContentRect(video)
    if (!content || video.clientWidth < 8 || video.clientHeight < 8) {
      setVideoFocusStyle({})
      return
    }
    const px = content.left + cx * content.scale
    const py = content.top + cy * content.scale
    const ox = (100 * px) / video.clientWidth
    const oy = (100 * py) / video.clientHeight
    setVideoFocusStyle({
      transform: 'scale(1.14)',
      transformOrigin: `${ox}% ${oy}%`,
      transition: 'transform 0.35s ease-out',
    })
  }, [mode, highlightedPruneEntry, nativeFps])

  useEffect(() => {
    updatePruneFocusZoom()
  }, [updatePruneFocusZoom, highlightedPruneEntry, activeSrc])

  useEffect(() => {
    const video = videoRef.current
    const wrap = videoWrapRef.current
    if (!video || !wrap) return
    const onFrame = () => updatePruneFocusZoom()
    video.addEventListener('timeupdate', onFrame)
    video.addEventListener('loadedmetadata', onFrame)
    video.addEventListener('seeked', onFrame)
    const ro = new ResizeObserver(() => updatePruneFocusZoom())
    ro.observe(wrap)
    return () => {
      video.removeEventListener('timeupdate', onFrame)
      video.removeEventListener('loadedmetadata', onFrame)
      video.removeEventListener('seeked', onFrame)
      ro.disconnect()
    }
  }, [updatePruneFocusZoom, activeSrc])

  const onRerun = useCallback(async () => {
    if (!id) return
    const live = run?.status === 'running' && run.subprocess_alive === true
    if (live) {
      window.alert('Wait for this run to finish or stop it from the Pipeline Lab before rerunning.')
      return
    }
    setRerunning(true)
    try {
      const r = await fetch(`${API}/api/runs/${id}/rerun`, { method: 'POST' })
      const text = await r.text()
      if (!r.ok) {
        let msg = `Rerun failed (HTTP ${r.status})`
        try {
          const j = JSON.parse(text) as { detail?: string }
          if (typeof j.detail === 'string') msg = j.detail
        } catch {
          if (text.trim()) msg = text.slice(0, 280)
        }
        window.alert(msg)
        return
      }
      const j = JSON.parse(text) as { run_id: string }
      nav('/', { state: { appendSessionRunIds: [j.run_id] } })
    } finally {
      setRerunning(false)
    }
  }, [id, nav, run?.status, run?.subprocess_alive])

  const runSummary = useMemo(() => {
    if (!pruneLog) return null
    const w = pruneLog.frame_width
    const h = pruneLog.frame_height
    const tf = pruneLog.total_frames
    const fps = pruneLog.native_fps
    const tr = pruneLog.tracker
    const rois = pruneLog.hybrid_sam_frame_rois
    const hybridSamFrames = Array.isArray(rois) ? rois.length : 0
    return {
      videoLabel: pruneLog.video_path ? basenamePath(pruneLog.video_path) : null,
      resolution: w && h ? `${w}×${h}` : null,
      frames: tf,
      fps,
      tracksIn: tr?.count,
      hybridSamFrames,
    }
  }, [pruneLog])

  const manifestFields = run?.manifest?.run_context_final?.fields

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
          {isDone && id ? (
            <Link to={`/watch/${id}/live`} className="btn">
              Live sandbox
            </Link>
          ) : null}
          <div>
            <h1 style={{ fontSize: '1.2rem', margin: 0, fontWeight: 600 }}>{title || 'Output'}</h1>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              {id?.slice(0, 8)}…{run?.status ? ` · ${run.status}` : ''}
            </div>
            {isDone && (manifestFields || run?.manifest?.run_context_final?.pipeline_diagnostics) ? (
              <RunOverviewStrip
                fields={manifestFields ?? null}
                diagnostics={run.manifest?.run_context_final?.pipeline_diagnostics ?? null}
              />
            ) : null}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {id && (
            <button
              type="button"
              className="btn"
              disabled={
                rerunning ||
                (run?.status === 'running' && run.subprocess_alive === true) ||
                !run
              }
              title="Queue a new run with the same video and settings"
              onClick={() => void onRerun()}
            >
              <RotateCw size={16} className={rerunning ? 'sway-spin' : undefined} aria-hidden />
              {rerunning ? ' Queuing…' : ' Rerun'}
            </button>
          )}
          <button
            type="button"
            className={`btn ${mode === 'phase' ? 'primary' : ''}`}
            onClick={() => setMode('phase')}
            disabled={!isDone && phaseTabs.length === 0}
            title={
              phaseTabs.length === 0 && isDone
                ? 'No phase preview MP4s — enable phase previews in the Lab recipe (Export) and re-run'
                : undefined
            }
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
          <button
            type="button"
            className={`btn ${mode === 'metrics' ? 'primary' : ''}`}
            onClick={() => setMode('metrics')}
          >
            <Gauge size={16} /> Metrics & Config
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
              {activePhase.blurb} Full-length clip, no title card. Event list → right; pruned regions are drawn in the video itself
              for pre-pose, collision, and post-pose stages (re-run with phase previews enabled if clips look old).
            </p>
          )}
        </div>
      )}

      {!err && run && mode === 'phase' && phaseTabs.length === 0 && (
        <div className="glass-panel" style={{ padding: '0.85rem 1rem' }}>
          <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.45rem' }}>
            Stage preview
          </div>
          <p style={{ margin: 0, fontSize: '0.85rem', color: '#e2e8f0', lineHeight: 1.55, maxWidth: 640 }}>
            No phase preview videos for this run. In the Lab, open your recipe&apos;s <strong style={{ color: '#7dd3fc' }}>Export</strong>{' '}
            step and enable phase preview MP4s, then re-run. Use <strong style={{ color: '#7dd3fc' }}>Final render</strong> or{' '}
            <strong style={{ color: '#7dd3fc' }}>Metrics &amp; Config</strong> meanwhile.
          </p>
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

      {!err && run && mode === 'metrics' && (
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h2 style={{ fontSize: '1.1rem', margin: '0 0 0.75rem', color: '#fff' }}>Run metrics & configuration</h2>
          <p
            style={{
              fontSize: '0.82rem',
              color: 'var(--text-muted)',
              margin: '0 0 1.25rem',
              lineHeight: 1.55,
              maxWidth: 760,
            }}
          >
            <strong style={{ color: '#e2e8f0' }}>Tracks</strong> — counts and heuristics only (no MOT ground truth).{' '}
            <strong style={{ color: '#e2e8f0' }}>What changed</strong> — hybrid SAM, interpolation, stitch, stage timing, and
            experimental flags from the manifest. <strong style={{ color: '#e2e8f0' }}>Configuration</strong> — same field groups as
            the Lab, plus raw <code style={{ fontSize: '0.78rem' }}>SWAY_*</code> env below.
          </p>

          <div style={{ marginBottom: '1.75rem' }}>
            <TrackQualitySummary summary={run.manifest?.run_context_final?.track_summary || {}} />
          </div>

          <div style={{ marginBottom: '1.75rem' }}>
            <PipelineImpactReport diagnostics={run.manifest?.run_context_final?.pipeline_diagnostics} />
          </div>

          <details style={{ marginBottom: '1.25rem' }}>
            <summary
              style={{
                cursor: 'pointer',
                color: 'var(--halo-cyan)',
                fontSize: '0.85rem',
                fontWeight: 600,
                marginBottom: '0.35rem',
              }}
            >
              Effective SWAY_* environment (subset)
            </summary>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: '0 0 0.5rem', lineHeight: 1.45 }}>
              Non-empty variables recorded on the manifest for diffing runs. Empty defaults are omitted.
            </p>
            <pre
              style={{
                margin: 0,
                maxHeight: 280,
                overflow: 'auto',
                padding: '0.65rem 0.75rem',
                borderRadius: 10,
                background: 'rgba(0,0,0,0.45)',
                border: '1px solid var(--glass-border)',
                color: '#cbd5e1',
                fontSize: '0.68rem',
                lineHeight: 1.4,
              }}
            >
              {JSON.stringify(run.manifest?.env ?? {}, null, 2)}
            </pre>
          </details>

          <FriendlyRunConfig
            fields={run.manifest?.run_context_final?.fields as Record<string, unknown> | undefined}
            schemaFields={labSchema?.fields}
            stages={labSchema?.stages}
          />
        </div>
      )}

      {mode !== 'metrics' && (
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
                style={{ position: 'relative', width: '100%', lineHeight: 0, overflow: 'hidden' }}
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
                    ...videoFocusStyle,
                  }}
                />
                {currentPhaseId === 'track' &&
                  samRoiMap.size > 0 &&
                  (() => {
                    const v = videoRef.current
                    const roi = samRoiMap.get(playFrame)
                    if (!v || !roi) return null
                    const layers = sam2RoiLayerStyles(roi, v)
                    if (!layers) return null
                    return (
                      <div key={`samroi-${playFrame}-${samOverlayLayout}`} style={{ pointerEvents: 'none' }}>
                        <div style={layers.box} title="Region passed to SAM2 (hybrid overlap refiner)" />
                        <div style={layers.label}>SAM2 input</div>
                      </div>
                    )
                  })()}
              </div>
              {mode === 'phase' && currentPhaseId === 'track' && (
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
                  <span style={{ fontWeight: 600, color: '#94a3b8' }}>Hybrid SAM2</span>
                  {samRoiMap.size > 0 ? (
                    <>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                        <span
                          style={{
                            width: 28,
                            height: 18,
                            borderRadius: 4,
                            border: '2px dashed rgba(255, 140, 0, 0.9)',
                            boxShadow: '0 0 0 1px rgba(0,0,0,0.35)',
                            flexShrink: 0,
                          }}
                          aria-hidden
                        />
                        <span>
                          Orange dashed box = the image region passed to SAM2 on overlap frames (ROI crop or full frame).
                          It is drawn in the preview MP4 and, when <code style={{ fontSize: '0.65rem' }}>prune_log.json</code>{' '}
                          includes <code style={{ fontSize: '0.65rem' }}>hybrid_sam_frame_rois</code>, the same box tracks playback
                          above the video.
                        </span>
                      </span>
                    </>
                  ) : (
                    <span style={{ fontSize: '0.7rem', opacity: 0.9 }}>
                      No hybrid SAM frame metadata in this run&apos;s prune log (hybrid off, or re-export from an older pipeline).
                      Re-run with hybrid SAM enabled to record ROI boxes.
                    </span>
                  )}
                </div>
              )}
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
                    <span style={{ fontWeight: 600, color: '#94a3b8' }}>Phase preview</span>
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
                    <span style={{ flex: '1 1 100%', fontSize: '0.68rem', opacity: 0.9 }}>
                      Legend matches tags drawn in the MP4. Click an event to seek and mild zoom on that region.
                    </span>
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
                {currentPhaseId === 'track' && runSummary.hybridSamFrames > 0 && (
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Hybrid SAM ROI frames in log </span>
                    {runSummary.hybridSamFrames}
                    <span style={{ color: 'var(--text-muted)' }}> (xyxy per frame in prune_log)</span>
                  </div>
                )}
              </div>
            )}
            {manifestFields && currentPhaseId && (
              <WatchPhaseTuningBlock phaseId={currentPhaseId} fields={manifestFields} schema={labSchema} />
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
                No track-level prune rows are logged during pose itself. The Lab locks{' '}
                <strong style={{ color: '#e2e8f0' }}>pose_stride=1</strong> (every frame) and{' '}
                <strong style={{ color: '#e2e8f0' }}>linear</strong> interpolation by default — skip-frame + GSI shortcuts are
                not the choreography path. Check the recipe snapshot for model (ViTPose vs RTMPose-L), visibility threshold, and
                3D lift; adjust on the Lab Config page if overlays look soft or sparse.
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
      )}

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
