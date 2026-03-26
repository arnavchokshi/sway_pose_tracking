import { formatConfigValue } from '../lib/formatConfigValue'

type Diag = Record<string, unknown>

function num(d: Diag, key: string): number | null {
  const v = d[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function str(d: Diag, key: string): string | null {
  const v = d[key]
  return typeof v === 'string' ? v : null
}

/** One-line + chips for the Lab run list (no schema needed). */
export function PipelineImpactSummary({ diagnostics }: { diagnostics?: Diag | null }) {
  if (!diagnostics || typeof diagnostics !== 'object') {
    return (
      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
        Diagnostics will appear here after this run completes (Lab manifest).
      </div>
    )
  }
  const hybrid = (diagnostics.hybrid_sam as Diag) || {}
  const interp = (diagnostics.interpolation as Diag) || {}
  const gst = (diagnostics.global_stitch as Diag) || {}
  const total = num(hybrid, 'hybrid_sam_frames_total')
  const refined = num(hybrid, 'hybrid_sam_frames_refined')
  const skips = num(hybrid, 'hybrid_sam_weak_cue_skips')
  const pct =
    total != null && total > 0 && refined != null ? Math.round((100 * refined) / total) : null
  const chips: string[] = []
  const tp = str(diagnostics as Diag, 'tracker_path')
  if (tp) {
    if (tp === 'boxmot:deepocsort+osnet') chips.push('BoxMOT · Deep OC-SORT + OSNet')
    else if (tp === 'boxmot:deepocsort') chips.push('BoxMOT · Deep OC-SORT')
    else chips.push(tp.replace(/^boxmot:/, 'BoxMOT · ').replace(/^botsort$/, 'BoT-SORT'))
  }
  if (pct != null && refined != null && refined > 0) chips.push(`SAM ${pct}% of det frames`)
  if (skips != null && skips > 0) chips.push(`weak-cue −${skips} SAM`)
  const boxMode = str(interp, 'box_mode')
  const poseGap = str(interp, 'pose_stride_gap_mode')
  const visMode = str(interp, 'export_video_temporal')
  if (boxMode === 'gsi') chips.push('box GSI')
  if (poseGap === 'gsi') chips.push('pose GSI')
  if (visMode === 'gsi') chips.push('export GSI')
  const af = str(gst, 'aflink_effective')
  if (af === 'neural') chips.push('AFLink neural')
  else if (af === 'heuristic' && gst.enabled === true) chips.push('AFLink heuristic')
  if (diagnostics.bidirectional_track_pass === true) chips.push('bidirectional')
  const rq = diagnostics.run_quality as Diag | undefined
  const sumS = rq && typeof rq.sum_stages_s === 'number' ? rq.sum_stages_s : null
  const nSt = rq && typeof rq.count === 'number' ? rq.count : null
  if (sumS != null && sumS > 0 && nSt != null && nSt > 0) {
    chips.push(`stages ~${sumS}s (${nSt})`)
  }

  return (
    <div style={{ fontSize: '0.72rem', lineHeight: 1.5 }}>
      <div style={{ color: '#cbd5e1', marginBottom: chips.length ? '0.35rem' : 0 }}>
        {pct != null && total != null
          ? `Hybrid SAM: ${refined}/${total} frames refined (${pct}%).`
          : 'Hybrid SAM: no counter (Ultralytics track path, or overlap off).'}
        {skips != null && skips > 0 && (
          <span style={{ color: '#86efac' }}> Weak-cue saved {skips} SAM run(s).</span>
        )}
      </div>
      {chips.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
          {chips.map((c) => (
            <span
              key={c}
              style={{
                fontSize: '0.62rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.04em',
                color: '#a5f3fc',
                border: '1px solid rgba(34,211,238,0.35)',
                borderRadius: 999,
                padding: '0.12rem 0.45rem',
                background: 'rgba(34,211,238,0.08)',
              }}
            >
              {c}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

/** What actually happened for hybrid SAM, interpolation, AFLink — from `pipeline_diagnostics` in manifest. */
export function PipelineImpactReport({ diagnostics }: { diagnostics?: Diag | null }) {
  if (!diagnostics || typeof diagnostics !== 'object') {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', lineHeight: 1.5 }}>
        No pipeline diagnostics on this manifest (older run or non-Lab CLI). Re-run from the Lab to capture
        hybrid SAM counts, interpolation modes, and stitch path.
      </div>
    )
  }

  const hybrid = (diagnostics.hybrid_sam as Diag) || {}
  const interp = (diagnostics.interpolation as Diag) || {}
  const gst = (diagnostics.global_stitch as Diag) || {}
  const notes = (diagnostics.notes as Diag) || {}

  const total = num(hybrid, 'hybrid_sam_frames_total')
  const refined = num(hybrid, 'hybrid_sam_frames_refined')
  const skips = num(hybrid, 'hybrid_sam_weak_cue_skips')
  const predCalls = num(hybrid, 'hybrid_sam_predict_calls')
  const frac = num(hybrid, 'hybrid_sam_refined_frac')
  const weakOn = hybrid.hybrid_sam_weak_cues_enabled === true

  const refinedPct =
    total != null && total > 0 && refined != null ? Math.round((100 * refined) / total) : null

  const rows: { title: string; body: string; tone?: 'ok' | 'muted' | 'accent' }[] = []

  const trackerPath = str(diagnostics as Diag, 'tracker_path')
  if (trackerPath) {
    const isBoxmot = trackerPath === 'boxmot' || trackerPath.startsWith('boxmot:')
    const variant = trackerPath.startsWith('boxmot:') ? trackerPath.slice('boxmot:'.length) : 'deepocsort'
    let body: string
    if (trackerPath === 'boxmot:deepocsort+osnet') {
      body = 'BoxMOT Deep OC-SORT with track-time OSNet; hybrid SAM when overlap refiner is on.'
    } else if (isBoxmot) {
      const label = variant === 'deepocsort' ? 'Deep OC-SORT (motion only)' : variant
      body = `BoxMOT (${label}) + hybrid SAM when overlap refiner is on.`
    } else {
      body = 'Ultralytics BoT-SORT — hybrid SAM and BoxMOT-only batch/YOLO options do not apply.'
    }
    rows.push({
      title: 'Tracking path',
      body,
      tone: 'muted',
    })
  }

  if (total != null && total > 0) {
    rows.push({
      title: 'Hybrid SAM2 (overlap refiner)',
      body: [
        `${refined ?? 0} of ${total} detector frames ran SAM (${refinedPct ?? '—'}%).`,
        predCalls != null ? `${predCalls} SAM predict call(s) total.` : '',
        weakOn && skips != null && skips > 0
          ? `Weak-cue gate skipped SAM on ${skips} frame(s) (saved work when overlap looked stable vs previous frame).`
          : weakOn
            ? 'Weak-cue gate was on; no skips (every high-IoU frame still ran SAM or had no history).'
            : 'Weak-cue gate off — high IoU alone triggers SAM when hybrid overlap is enabled.',
        frac != null && frac > 0 ? `Refined fraction (stored): ${frac}.` : '',
      ]
        .filter(Boolean)
        .join(' '),
      tone: refined != null && refined > 0 ? 'accent' : 'ok',
    })
  } else {
    rows.push({
      title: 'Hybrid SAM2',
      body: 'No hybrid SAM counters (e.g. SWAY_USE_BOXMOT=0 path or overlap refiner off).',
      tone: 'muted',
    })
  }

  const boxMode = str(interp, 'box_mode')
  const poseGap = str(interp, 'pose_stride_gap_mode')
  const poseStride = interp.pose_stride_cli
  const visMode = str(interp, 'export_video_temporal')
  if (boxMode || poseGap || visMode) {
    const parts = [
      boxMode && boxMode !== 'linear' ? `Box stitch/stride gaps: ${boxMode.toUpperCase()}` : null,
      poseGap && poseGap !== 'linear' && typeof poseStride === 'number' && poseStride > 1
        ? `Pose stride gaps: ${poseGap.toUpperCase()} (stride ${poseStride})`
        : poseGap && poseGap !== 'linear'
          ? `Pose stride gaps: ${poseGap.toUpperCase()} (enable stride 2 to apply)`
          : null,
      visMode && visMode !== 'linear' ? `Export MP4 blend: ${visMode.toUpperCase()}` : null,
    ].filter(Boolean)
    rows.push({
      title: 'Interpolation (GSI / linear)',
      body:
        parts.length > 0
          ? `${parts.join(' · ')}. JSON keyframes stay on discrete processed frames; GSI mainly smooths boxes, pose fill, or video export. Master choreography stance: pose every frame + linear defaults (Pose tab verdict).`
          : 'All linear (default): box gaps, pose stride gaps (if any), and export-time video blend — matches the master lock (pose_stride=1 in Lab, linear interpolation).',
      tone: parts.length > 0 ? 'accent' : 'ok',
    })
  }

  const af = str(gst, 'aflink_effective')
  const glOn = gst.enabled === true
  if (af) {
    rows.push({
      title: 'Global ID stitch (Phase 3)',
      body:
        !glOn
          ? 'Long-range merge disabled (SWAY_GLOBAL_LINK off).'
          : af === 'neural'
            ? 'Neural AFLink ran (weights found, smart linker not forced to heuristic).'
            : af === 'heuristic'
              ? 'Heuristic spatial/temporal stitch only (no weights, or “simple rules” selected, or AFLink failed at runtime).'
              : 'Disabled.',
      tone: af === 'neural' ? 'accent' : 'ok',
    })
  }

  if (diagnostics.bidirectional_track_pass === true) {
    rows.push({
      title: 'Bidirectional tracking',
      body: 'Forward + reverse pass merged — expect ~2× Phase 1–2 time; can reduce ID fragmentation on hard clips.',
      tone: 'accent',
    })
  }

  rows.push({
    title: 'Track stats JSON',
    body: 'Always written after Phase 3 as track_stats.json (small summary — use tools/analyze_track_stats).',
    tone: 'ok',
  })

  const rq = diagnostics.run_quality as Diag | undefined
  if (rq && typeof rq === 'object') {
    const sumS = typeof rq.sum_stages_s === 'number' ? rq.sum_stages_s : null
    const cnt = typeof rq.count === 'number' ? rq.count : null
    const per = rq.per_stage_elapsed_s as Record<string, number> | undefined
    if (sumS != null && cnt != null && cnt > 0) {
      const top = per
        ? Object.entries(per)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4)
            .map(([k, v]) => `${k}: ${v}s`)
            .join(' · ')
        : ''
      rows.push({
        title: 'Stage timing (from progress log)',
        body: [
          `Sum of recorded major stages ≈ ${sumS}s across ${cnt} checkpoint(s).`,
          top ? `Slowest: ${top}.` : '',
          'Phases 1–2 may span one combined line; not a full profiler.',
        ]
          .filter(Boolean)
          .join(' '),
        tone: 'muted',
      })
    }
  }

  const exp = diagnostics.experimental as Diag | undefined
  if (exp && typeof exp === 'object') {
    const bits: string[] = []
    if (exp.gnn_track_refine === true) bits.push('GNN track refine (post-stitch graph merge)')
    if (exp.sapiens_pose_cli === true) {
      if (exp.sapiens_native_torchscript === true) {
        bits.push('Sapiens pose slot → native TorchScript')
      } else {
        bits.push('Sapiens pose slot → ViTPose-Base fallback')
      }
    }
    if (exp.hmr_mesh_sidecar === true) bits.push('HMR mesh sidecar JSON written')
    if (bits.length > 0) {
      rows.push({
        title: 'Experimental flags',
        body: bits.join(' · '),
        tone: 'accent',
      })
    }
  }

  const noteGsi = notes.gsi
  const noteAf = notes.aflink
  const noteBlock =
    typeof noteGsi === 'string' || typeof noteAf === 'string'
      ? [typeof noteGsi === 'string' ? noteGsi : '', typeof noteAf === 'string' ? noteAf : '']
          .filter(Boolean)
          .join(' ')
      : ''

  return (
    <div style={{ marginTop: '0.5rem' }}>
      <div
        style={{
          fontSize: '0.72rem',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          color: '#94a3b8',
          marginBottom: '0.55rem',
        }}
      >
        What changed this run
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
        {rows.map((r) => (
          <div
            key={r.title}
            style={{
              padding: '0.65rem 0.75rem',
              borderRadius: 10,
              background: 'rgba(15, 23, 42, 0.55)',
              border: `1px solid ${
                r.tone === 'accent' ? 'rgba(34, 211, 238, 0.35)' : 'rgba(148, 163, 184, 0.22)'
              }`,
            }}
          >
            <div style={{ fontSize: '0.68rem', fontWeight: 700, color: '#94a3b8', marginBottom: '0.25rem' }}>{r.title}</div>
            <div style={{ fontSize: '0.82rem', color: '#e2e8f0', lineHeight: 1.5 }}>{r.body}</div>
          </div>
        ))}
      </div>
      {noteBlock && (
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.65rem', lineHeight: 1.45 }}>
          {noteBlock}
        </div>
      )}
    </div>
  )
}

type SchemaField = {
  id: string
  phase: string
  label: string
  type: string
  advanced?: boolean
  lab_hidden?: boolean
}
type Stage = { id: string; label: string }

function fileBasename(p: string): string {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  return i >= 0 ? p.slice(i + 1) : p
}

/** Compact header strip: recipe choices + effective tracker path + timing hint. */
export function RunOverviewStrip({
  fields,
  diagnostics,
}: {
  fields?: Record<string, unknown> | null
  diagnostics?: Diag | null
}) {
  const chips: { k: string; v: string }[] = []
  const tr = fields?.tracker_technology
  const pose = fields?.pose_model
  const yolo = fields?.sway_yolo_weights
  if (typeof tr === 'string') {
    const tv =
      tr === 'deep_ocsort'
        ? 'Deep OC-SORT (default)'
        : tr === 'deep_ocsort_osnet'
          ? 'Deep OC-SORT + OSNet'
          : tr
    chips.push({ k: 'Tracker', v: tv })
  }
  if (typeof pose === 'string') chips.push({ k: 'Pose', v: pose.length > 28 ? `${pose.slice(0, 26)}…` : pose })
  if (typeof yolo === 'string') {
    const short = fileBasename(yolo)
    chips.push({ k: 'YOLO', v: short.length > 32 ? `${short.slice(0, 30)}…` : short })
  }
  const tp = diagnostics && str(diagnostics, 'tracker_path')
  if (tp && !chips.some((c) => c.k === 'Tracker')) {
    chips.push({ k: 'Engine', v: tp })
  } else if (tp) {
    chips.push({ k: 'Path', v: tp })
  }
  const rq = diagnostics?.run_quality as Diag | undefined
  const sumS = rq && typeof rq.sum_stages_s === 'number' ? rq.sum_stages_s : null
  if (sumS != null && sumS > 0) chips.push({ k: 'Stages Σ', v: `~${sumS}s logged` })

  if (chips.length === 0) return null
  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0.4rem 0.65rem',
        alignItems: 'center',
        padding: '0.5rem 0.75rem',
        borderRadius: 10,
        background: 'rgba(15, 23, 42, 0.5)',
        border: '1px solid rgba(148, 163, 184, 0.22)',
        marginTop: '0.35rem',
      }}
    >
      {chips.map((c) => (
        <span key={c.k} style={{ fontSize: '0.74rem', color: '#e2e8f0', lineHeight: 1.35 }}>
          <span style={{ color: '#94a3b8', marginRight: 6 }}>{c.k}</span>
          <span style={{ fontWeight: 600 }}>{c.v}</span>
        </span>
      ))}
    </div>
  )
}

/** Group Lab fields by pipeline stage using schema labels (readable vs raw JSON keys). */
export function FriendlyRunConfig({
  fields,
  schemaFields,
  stages,
}: {
  fields?: Record<string, unknown>
  schemaFields?: SchemaField[]
  stages?: Stage[]
}) {
  if (!fields || Object.keys(fields).length === 0) {
    return <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No configuration snapshot on this run.</div>
  }
  if (!schemaFields?.length) {
    return <RunConfigRawFallback fields={fields} />
  }

  const stageOrder = stages?.map((s) => s.id) ?? []
  const byPhase = new Map<string, SchemaField[]>()
  for (const f of schemaFields) {
    if (f.type === 'info') continue
    if (f.lab_hidden) continue
    const arr = byPhase.get(f.phase) || []
    arr.push(f)
    byPhase.set(f.phase, arr)
  }

  const orderedPhases = [
    ...stageOrder.filter((id) => byPhase.has(id)),
    ...[...byPhase.keys()].filter((id) => !stageOrder.includes(id)).sort(),
  ]

  const stageTitle = (id: string) => stages?.find((s) => s.id === id)?.label ?? id

  return (
    <div style={{ marginTop: '0.5rem' }}>
      <div
        style={{
          fontSize: '0.72rem',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          color: '#94a3b8',
          marginBottom: '0.55rem',
        }}
      >
        Configuration (from Lab UI)
      </div>
      <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: '0 0 0.65rem', lineHeight: 1.45, maxWidth: 720 }}>
        Values you set for this run, grouped by the same pipeline tabs as the Lab. Stages with only defaults may be
        omitted from the snapshot.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {orderedPhases.map((phaseId) => {
          const flist = byPhase.get(phaseId) ?? []
          const withValues = flist.filter((f) => Object.prototype.hasOwnProperty.call(fields, f.id))
          if (withValues.length === 0) return null
          return (
            <div key={phaseId}>
              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#cbd5e1', marginBottom: '0.4rem' }}>
                {stageTitle(phaseId)}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 240px), 1fr))', gap: '0.45rem' }}>
                {withValues.map((f) => (
                  <div
                    key={f.id}
                    title={f.id}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      background: 'rgba(0,0,0,0.2)',
                      padding: '0.45rem 0.55rem',
                      borderRadius: 8,
                      border: '1px solid var(--glass-border)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem', lineHeight: 1.3 }}>{f.label}</span>
                      {f.advanced ? (
                        <span
                          style={{
                            fontSize: '0.58rem',
                            fontWeight: 700,
                            textTransform: 'uppercase',
                            letterSpacing: '0.04em',
                            color: '#a78bfa',
                            border: '1px solid rgba(167,139,250,0.4)',
                            borderRadius: 4,
                            padding: '0.05rem 0.3rem',
                          }}
                        >
                          Advanced
                        </span>
                      ) : null}
                    </div>
                    <span style={{ color: '#fff', fontFamily: 'ui-monospace, monospace', fontSize: '0.8rem', marginTop: '0.15rem' }}>
                      {formatConfigValue(fields[f.id])}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function RunConfigRawFallback({ fields }: { fields: Record<string, unknown> }) {
  return (
    <div style={{ marginTop: '0.5rem' }}>
      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.45rem' }}>
        Schema not loaded — raw keys:
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 200px), 1fr))', gap: '0.4rem', fontSize: '0.75rem' }}>
        {Object.entries(fields)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([k, v]) => (
            <div key={k} style={{ display: 'flex', flexDirection: 'column', background: 'rgba(0,0,0,0.2)', padding: '0.4rem 0.5rem', borderRadius: 6, border: '1px solid var(--glass-border)' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem', marginBottom: '0.1rem', wordBreak: 'break-all' }}>{k}</span>
              <span style={{ color: '#fff', fontFamily: 'ui-monospace, monospace' }}>{formatConfigValue(v)}</span>
            </div>
          ))}
      </div>
    </div>
  )
}
