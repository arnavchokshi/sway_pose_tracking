import type { ReactNode } from 'react'

/**
 * Tiny reactive visuals for pipeline config sliders — shows how “strong” or “wide” a setting is.
 * Keep height compact so grids stay scannable.
 */

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x))
}

function norm(value: number, min: number, max: number) {
  if (max <= min) return 0.5
  return clamp01((value - min) / (max - min))
}

/** Two boxes — overlap grows when the normalized value is high (tweak per field). */
export function OverlapPairVisual({
  value,
  min,
  max,
  invert,
}: {
  value: number
  min: number
  max: number
  invert?: boolean
}) {
  let t = norm(value, min, max)
  if (invert) t = 1 - t
  const gap = 8 + (1 - t) * 22
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap,
        height: 44,
        marginBottom: 6,
      }}
      aria-hidden
    >
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: 6,
          border: '2px solid rgba(6, 182, 212, 0.85)',
          background: 'rgba(6, 182, 212, 0.12)',
          boxShadow: '0 0 12px rgba(6, 182, 212, 0.15)',
        }}
      />
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: 6,
          border: '2px solid rgba(236, 72, 153, 0.85)',
          background: 'rgba(236, 72, 153, 0.1)',
          boxShadow: '0 0 12px rgba(236, 72, 153, 0.12)',
        }}
      />
    </div>
  )
}

/** Horizontal fill — “how much time / memory / bridge”. */
export function SpanBarVisual({ value, min, max, hue = 195 }: { value: number; min: number; max: number; hue?: number }) {
  const t = norm(value, min, max)
  return (
    <div style={{ marginBottom: 6 }} aria-hidden>
      <div
        style={{
          height: 8,
          borderRadius: 4,
          background: 'rgba(255,255,255,0.06)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${t * 100}%`,
            height: '100%',
            borderRadius: 4,
            background: `linear-gradient(90deg, hsla(${hue}, 70%, 45%, 0.5), hsla(${hue}, 85%, 55%, 0.95))`,
            transition: 'width 0.12s ease-out',
          }}
        />
      </div>
    </div>
  )
}

/** Single bar — confidence / strictness. */
export function LevelMeterVisual({
  value,
  min,
  max,
  lowLabel,
  highLabel,
}: {
  value: number
  min: number
  max: number
  lowLabel: string
  highLabel: string
}) {
  const t = norm(value, min, max)
  return (
    <div style={{ marginBottom: 6 }} aria-hidden>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: 4 }}>
        <span>{lowLabel}</span>
        <span>{highLabel}</span>
      </div>
      <div style={{ height: 10, borderRadius: 5, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
        <div
          style={{
            width: `${t * 100}%`,
            height: '100%',
            borderRadius: 5,
            background: `linear-gradient(90deg, #34d399, #fbbf24, #f97316)`,
            backgroundSize: '200% 100%',
            backgroundPosition: `${(1 - t) * 100}% 0`,
            transition: 'width 0.12s ease-out',
          }}
        />
      </div>
    </div>
  )
}

/** Frame with inset margin — edge band. */
export function EdgeFrameVisual({ value, min, max }: { value: number; min: number; max: number }) {
  const t = norm(value, min, max)
  const inset = 4 + t * 14
  return (
    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 6 }} aria-hidden>
      <div
        style={{
          position: 'relative',
          width: 72,
          height: 48,
          borderRadius: 6,
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(148,163,184,0.25)',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: inset,
            left: inset,
            right: inset,
            bottom: inset,
            borderRadius: 4,
            border: '1px dashed rgba(6, 182, 212, 0.45)',
            background: 'rgba(6, 182, 212, 0.06)',
            transition: 'all 0.12s ease-out',
          }}
        />
      </div>
    </div>
  )
}

/** Jitter / noise spark. */
export function JitterSparkVisual({ value, min, max }: { value: number; min: number; max: number }) {
  const t = norm(value, min, max)
  const h = [8, 18, 12, 22, 14, 20, 10, 16].map((base, i) => base + (i % 3) * t * 8)
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: 3, height: 36, marginBottom: 4 }} aria-hidden>
      {h.map((height, i) => (
        <div
          key={i}
          style={{
            width: 4,
            height,
            borderRadius: 2,
            background: `rgba(251, 191, 36, ${0.35 + t * 0.5})`,
            transition: 'height 0.12s ease-out',
          }}
        />
      ))}
    </div>
  )
}

/** Motion / kinetic — wave amplitude. */
export function MotionWaveVisual({ value, min, max }: { value: number; min: number; max: number }) {
  const t = norm(value, min, max)
  const amp = 4 + t * 14
  const w = 80
  const mid = 20
  const d = `M 0 ${mid} Q 20 ${mid - amp} 40 ${mid} T 80 ${mid}`
  return (
    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 4 }} aria-hidden>
      <svg width={w} height={32} style={{ overflow: 'visible' }}>
        <path d={d} fill="none" stroke="rgba(6, 182, 212, 0.85)" strokeWidth={2} strokeLinecap="round" />
      </svg>
    </div>
  )
}

/** Smoothing — path rough vs smooth. */
export function SmoothPathVisual({ cutoff, beta, cutoffMin, cutoffMax, betaMin, betaMax }: {
  cutoff: number
  beta: number
  cutoffMin: number
  cutoffMax: number
  betaMin: number
  betaMax: number
}) {
  const smoothness = 1 - norm(cutoff, cutoffMin, cutoffMax) * 0.55 + norm(beta, betaMin, betaMax) * 0.25
  const rough = Math.max(0.05, 1 - smoothness)
  
  // Create a visually rich line
  const pts: [number, number][] = []
  for (let x = 8; x <= 76; x += 4) {
    const jitter = Math.sin(x * 0.5) * rough * 12 + Math.cos(x * 1.3) * rough * 5
    pts.push([x, 20 + jitter])
  }
  
  const d = `M 4 20 ` + pts.map(p => `L ${p[0]} ${p[1]}`).join(' ')
  
  return (
    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 4 }} aria-hidden>
      <svg width={80} height={40}>
        {/* Glow */}
        <path d={d} fill="none" stroke="rgba(129, 140, 248, 0.3)" strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
        <path d={d} fill="none" stroke="rgba(129, 140, 248, 0.9)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

/** Resolution box for detect size */
export function ResolutionBoxVisual({ value, min, max }: { value: number; min: number; max: number }) {
  const t = norm(value, min, max)
  const size = 16 + t * 24
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 44, marginBottom: 4 }} aria-hidden>
      <div
        style={{
          width: 44,
          height: 44,
          border: '1px dashed rgba(255,255,255,0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative'
        }}
      >
        <div
          style={{
            width: size,
            height: size,
            border: '2px solid rgba(16, 185, 129, 0.8)',
            background: 'rgba(16, 185, 129, 0.15)',
            boxShadow: '0 0 10px rgba(16, 185, 129, 0.2)',
            transition: 'all 0.15s ease-out'
          }}
        />
      </div>
    </div>
  )
}

/** Neighbor blend radius — dots. */
export function NeighborDotsVisual({ radius, maxR }: { radius: number; maxR: number }) {
  const t = maxR <= 0 ? 0 : radius / maxR
  const spread = 6 + t * 18
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 40, marginBottom: 4, gap: spread }} aria-hidden>
      <div style={{ width: 8, height: 8, borderRadius: 999, background: 'rgba(148,163,184,0.5)' }} />
      <div style={{ width: 10, height: 10, borderRadius: 999, background: 'rgba(6, 182, 212, 0.95)', boxShadow: '0 0 10px rgba(6,182,212,0.4)' }} />
      <div style={{ width: 8, height: 8, borderRadius: 999, background: 'rgba(148,163,184,0.5)' }} />
    </div>
  )
}

/** Detection stride — tick marks. */
export function StrideTicksVisual({ stride, maxS }: { stride: number; maxS: number }) {
  const n = Math.min(8, Math.max(2, Math.round(maxS)))
  const st = Math.max(1, Math.round(stride))
  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: 4, height: 32, alignItems: 'flex-end', marginBottom: 4 }} aria-hidden>
      {Array.from({ length: n }, (_, i) => {
        const active = i % st === 0
        return (
          <div
            key={i}
            style={{
              width: 5,
              height: active ? 24 : 8,
              borderRadius: 2,
              background: active ? 'rgba(6, 182, 212, 0.9)' : 'rgba(148,163,184,0.25)',
              transition: 'height 0.15s ease-out',
            }}
          />
        )
      })}
    </div>
  )
}

/** Count of people before overlap mode kicks in. */
export function MinPeopleVisual({ count, cap }: { count: number; cap: number }) {
  const slots = Math.min(8, Math.max(2, cap))
  const lit = Math.min(slots, Math.max(0, Math.round(count)))
  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: 5, height: 36, alignItems: 'center', marginBottom: 4 }} aria-hidden>
      {Array.from({ length: slots }, (_, i) => (
        <div
          key={i}
          style={{
            width: i < lit ? 12 : 10,
            height: i < lit ? 12 : 10,
            borderRadius: 999,
            background: i < lit ? 'rgba(6, 182, 212, 0.85)' : 'rgba(148,163,184,0.15)',
            border: i < lit ? 'none' : '1px solid rgba(148,163,184,0.25)',
            transition: 'all 0.12s ease-out',
          }}
        />
      ))}
    </div>
  )
}

/** Prune / vote strength. */
export function PruneStrengthVisual({ value, min, max }: { value: number; min: number; max: number }) {
  const t = norm(value, min, max)
  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: 5, height: 40, alignItems: 'flex-end', marginBottom: 4 }} aria-hidden>
      {[0.25, 0.5, 0.75, 1].map((h, i) => (
        <div
          key={i}
          style={{
            width: 10,
            height: 12 + h * 22 * t,
            borderRadius: 3,
            background: t > h - 0.2 ? 'rgba(248, 113, 113, 0.75)' : 'rgba(148,163,184,0.2)',
            transition: 'all 0.12s ease-out',
          }}
        />
      ))}
    </div>
  )
}

/** Weight slider — single vertical bar. */
export function WeightBarVisual({ value, min, max }: { value: number; min: number; max: number }) {
  const t = norm(value, min, max)
  return (
    <div style={{ display: 'flex', justifyContent: 'center', height: 36, marginBottom: 2 }} aria-hidden>
      <div style={{ width: 14, height: '100%', borderRadius: 7, background: 'rgba(255,255,255,0.06)', position: 'relative', overflow: 'hidden' }}>
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            height: `${t * 100}%`,
            borderRadius: 7,
            background: 'linear-gradient(180deg, rgba(6,182,212,0.4), rgba(14,165,233,0.95))',
            transition: 'height 0.1s ease-out',
          }}
        />
      </div>
    </div>
  )
}

export function renderConfigVisual(
  fieldId: string,
  value: number,
  min: number,
  max: number,
  allFields?: Record<string, unknown>,
): ReactNode {
  switch (fieldId) {
    case 'sway_hybrid_sam_iou_trigger':
      return <OverlapPairVisual value={value} min={min} max={max} invert />
    case 'sway_pretrack_nms_iou':
      return <OverlapPairVisual value={value} min={min} max={max} />
    case 'sway_boxmot_match_thresh':
      return <OverlapPairVisual value={value} min={min} max={max} invert />
    case 'sway_boxmot_max_age':
    case 'sway_stitch_max_frame_gap':
    case 'sway_dormant_max_gap':
    case 'reid_max_frame_gap':
      return <SpanBarVisual value={value} min={min} max={max} hue={195} />
    case 'sway_chunk_size':
      return <SpanBarVisual value={value} min={min} max={max} hue={265} />
    case 'sway_yolo_conf':
    case 'pose_visibility_threshold':
    case 'mean_confidence_min':
    case 'tier_c_skeleton_mean':
    case 'sway_hybrid_sam_mask_thresh':
    case 'min_lower_body_conf_yaml':
    case 'reid_min_oks':
      return <LevelMeterVisual value={value} min={min} max={max} lowLabel="looser" highLabel="stricter" />
    case 'sync_score_min':
      return <LevelMeterVisual value={value} min={min} max={max} lowLabel="forgiving" highLabel="tight" />
    case 'edge_margin_frac':
      return <EdgeFrameVisual value={value} min={min} max={max} />
    case 'edge_presence_frac':
      return <LevelMeterVisual value={value} min={min} max={max} lowLabel="rare" highLabel="often" />
    case 'jitter_ratio_max':
      return <JitterSparkVisual value={value} min={min} max={max} />
    case 'kinetic_std_frac':
      return <MotionWaveVisual value={value} min={min} max={max} />
    case 'prune_threshold':
      return <PruneStrengthVisual value={value} min={min} max={max} />
    case 'confirmed_human_min_span_frac':
    case 'min_duration_ratio':
    case 'tier_c_low_frame_frac':
      return <SpanBarVisual value={value} min={min} max={max} hue={160} />
    case 'sway_yolo_detection_stride':
      return <StrideTicksVisual stride={Math.round(value)} maxS={max} />
    case 'sway_detect_size':
      return <ResolutionBoxVisual value={value} min={min} max={max} />
    case 'sway_hybrid_sam_min_dets':
      return <MinPeopleVisual count={value} cap={max} />
    case 'sway_hybrid_sam_bbox_pad':
      return <EdgeFrameVisual value={value} min={0} max={Math.max(max, 1)} />
    case 'sway_hybrid_sam_roi_pad_frac':
      return <LevelMeterVisual value={value} min={min} max={max} lowLabel="tight" highLabel="loose" />
    case 'temporal_pose_radius':
      return <NeighborDotsVisual radius={value} maxR={max} />
    case 'smoother_min_cutoff': {
      const beta = typeof allFields?.smoother_beta === 'number' ? (allFields.smoother_beta as number) : 0.7
      return (
        <SmoothPathVisual
          cutoff={value}
          beta={beta}
          cutoffMin={min}
          cutoffMax={max}
          betaMin={0}
          betaMax={5}
        />
      )
    }
    case 'smoother_beta': {
      const cut = typeof allFields?.smoother_min_cutoff === 'number' ? (allFields.smoother_min_cutoff as number) : 1
      return (
        <SmoothPathVisual
          cutoff={cut}
          beta={value}
          cutoffMin={0.01}
          cutoffMax={5}
          betaMin={min}
          betaMax={max}
        />
      )
    }
    default:
      if (fieldId.startsWith('pruning_w_')) {
        return <WeightBarVisual value={value} min={min} max={max} />
      }
      return <SpanBarVisual value={value} min={min} max={max} hue={200} />
  }
}
