import type { SchemaField } from '../types'
import { renderConfigVisual } from './ConfigFieldVisual'

const YOLO_CARD_META: Record<string, { accent: string; badge: string; hint: string }> = {
  yolo26s: { accent: '#34d399', badge: 'Fastest run', hint: 'Drafts & quick checks' },
  yolo26l: { accent: '#fbbf24', badge: 'Balanced', hint: 'Most performances' },
  yolo26l_dancetrack: {
    accent: '#a78bfa',
    badge: 'DanceTrack FT',
    hint: 'Group lines & crowds (local .pt)',
  },
  yolo26x: { accent: '#f97316', badge: 'Slowest, sharpest', hint: 'Hard lighting & small bodies' },
}

const POSE_CARD_META: Record<string, { accent: string; badge: string; hint: string }> = {
  'ViTPose-Base': { accent: '#34d399', badge: 'Light', hint: 'Default speed' },
  'ViTPose-Large': { accent: '#fbbf24', badge: 'Heavy', hint: 'Sharper poses' },
  'ViTPose-Huge': { accent: '#f97316', badge: 'Max', hint: 'Best, slowest' },
  'RTMPose-L': { accent: '#22d3ee', badge: 'MMPose', hint: 'Fast (optional install)' },
  'Sapiens (ViTPose-Base fallback)': {
    accent: '#c084fc',
    badge: 'Sapiens slot',
    hint: 'ViTPose-Base until native Sapiens is wired',
  },
}

const BOOL_STATUS: Record<string, { on: string; off: string }> = {
  sway_hybrid_sam_weak_cues: { on: 'Stable-overlap gate on', off: 'IoU only' },
  sway_bidirectional_track_pass: { on: 'Forward + reverse pass', off: 'Forward only' },
  sway_gnn_track_refine: { on: 'GNN hook on (stub)', off: 'Off' },
  sway_hmr_mesh_sidecar: { on: 'hmr_mesh_sidecar.json', off: 'Off' },
  montage: { on: 'Montage clip', off: 'No montage' },
  save_phase_previews: { on: 'Phase clips saved', off: 'Skip previews' },
  temporal_pose_refine: { on: 'Neighbor blend on', off: 'Raw only' },
}

function choiceDisplayLabel(fieldId: string, raw: string): string {
  switch (fieldId) {
    case 'pose_stride':
      if (raw === '1') return 'Every frame'
      if (raw === '2') return 'Every other frame'
      break
    case 'tracker_technology':
      if (raw === 'deep_ocsort') return 'Default (motion only)'
      if (raw === 'deep_ocsort_osnet') return 'Deep OC-SORT + OSNet'
      if (raw === 'BoxMOT') return 'Default (motion only)'
      if (raw === 'StrongSORT') return 'Deep OC-SORT + OSNet'
      if (raw === 'BoT-SORT' || raw === 'ByteTrack' || raw === 'OC-SORT') return 'Default (motion only)'
      break
    case 'sway_global_aflink_mode':
      if (raw === 'neural_if_available') return 'Smart linker'
      if (raw === 'force_heuristic') return 'Simple rules'
      break
    case 'sway_box_interp_mode':
      if (raw === 'linear') return 'Linear'
      if (raw === 'gsi') return 'GSI (smooth)'
      break
    case 'sway_pose_gap_interp_mode':
      if (raw === 'linear') return 'Linear'
      if (raw === 'gsi') return 'GSI (smooth)'
      break
    case 'sway_vis_temporal_interp_mode':
      if (raw === 'linear') return 'Linear'
      if (raw === 'gsi') return 'GSI (smooth)'
      break
    case 'pose_model':
      if (raw === 'ViTPose-Base') return 'Base — fastest'
      if (raw === 'ViTPose-Large') return 'Large — sharper'
      if (raw === 'ViTPose-Huge') return 'Huge — heaviest'
      if (raw === 'RTMPose-L') return 'RTMPose — speed test'
      if (raw === 'Sapiens (ViTPose-Base fallback)') return 'Sapiens slot — ViT base'
      break
    case 'sway_boxmot_reid_model':
      if (raw === 'osnet_x0_25') return 'Lighter'
      if (raw === 'osnet_x1_0') return 'Heavier, finer'
      break
    case 'sway_yolo_weights':
      if (raw === 'yolo26s') return 'YOLO26s'
      if (raw === 'yolo26l') return 'YOLO26l'
      if (raw === 'yolo26l_dancetrack') return 'YOLO26l · DanceTrack'
      if (raw === 'yolo26x') return 'YOLO26x'
      break
    default:
      break
  }
  return raw
}

function sliderStep(f: SchemaField): number {
  if (f.type === 'int') {
    return 1
  }
  const span = (f.max ?? 1) - (f.min ?? 0)
  if (span <= 1) return 0.01
  if (span <= 0.2) return 0.005
  return 0.01
}

function BoolToggle({
  checked,
  onChange,
  fieldId,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  fieldId: string
}) {
  const labels = BOOL_STATUS[fieldId] ?? { on: 'On', off: 'Off' }
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="config-bool-toggle"
      style={{
        display: 'flex',
        width: '100%',
        maxWidth: 280,
        padding: 3,
        borderRadius: 999,
        border: '1px solid var(--glass-border)',
        background: 'rgba(0,0,0,0.35)',
        cursor: 'pointer',
        gap: 0,
      }}
    >
      <span
        style={{
          flex: 1,
          textAlign: 'center',
          padding: '0.4rem 0.6rem',
          borderRadius: 999,
          fontSize: '0.78rem',
          fontWeight: 700,
          transition: 'background 0.15s ease, color 0.15s ease',
          background: checked ? 'rgba(6, 182, 212, 0.35)' : 'transparent',
          color: checked ? '#ecfeff' : 'var(--text-muted)',
        }}
      >
        {labels.on}
      </span>
      <span
        style={{
          flex: 1,
          textAlign: 'center',
          padding: '0.4rem 0.6rem',
          borderRadius: 999,
          fontSize: '0.78rem',
          fontWeight: 700,
          transition: 'background 0.15s ease, color 0.15s ease',
          background: !checked ? 'rgba(148, 163, 184, 0.2)' : 'transparent',
          color: !checked ? '#e2e8f0' : 'var(--text-muted)',
        }}
      >
        {labels.off}
      </span>
    </button>
  )
}

export function InlineFieldInput({
  f,
  value,
  onChange,
  modelsStatus,
  allFields,
}: {
  f: SchemaField
  value: unknown
  onChange: (v: unknown) => void
  modelsStatus?: Record<string, boolean> | null
  /** For cross-field visuals (e.g. smoother). */
  allFields?: Record<string, unknown>
}) {
  const v = value === undefined ? f.default : value

  if (f.type === 'bool') {
    return <BoolToggle checked={Boolean(v)} onChange={(x) => onChange(x)} fieldId={f.id} />
  }

  if (f.type === 'enum' && f.choices) {
    const disabled = new Set(f.disabled_choices?.map((c) => String(c)) ?? [])

    if (f.display === 'segmented') {
      return (
        <div style={{ display: 'flex', borderRadius: 10, overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
          {f.choices.map((c) => {
            const s = String(c)
            const isSelected = String(v ?? '') === s
            const isDis = disabled.has(s)
            return (
              <button
                key={s}
                type="button"
                disabled={isDis}
                onClick={() => {
                  if (isDis) return
                  const num = Number(s)
                  onChange(Number.isNaN(num) ? s : num)
                }}
                style={{
                  flex: 1,
                  padding: '0.5rem 0.65rem',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  border: 'none',
                  cursor: isDis ? 'not-allowed' : 'pointer',
                  opacity: isDis ? 0.45 : 1,
                  background: isSelected ? 'rgba(14, 165, 233, 0.4)' : 'rgba(0,0,0,0.25)',
                  color: '#e2e8f0',
                  transition: 'background 0.12s ease',
                }}
              >
                {choiceDisplayLabel(f.id, s)}
              </button>
            )
          })}
        </div>
      )
    }

    if (f.display === 'tracker_strip') {
      return (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
          {f.choices.map((c) => {
            const s = String(c)
            const isSelected = String(v ?? '') === s
            const isDis = disabled.has(s)
            return (
              <button
                key={s}
                type="button"
                disabled={isDis}
                onClick={() => {
                  if (isDis) return
                  onChange(s)
                }}
                className={`config-tracker-pill ${isSelected ? 'selected' : ''}`}
                style={{
                  padding: '0.55rem 1rem',
                  borderRadius: 12,
                  border: isSelected ? '2px solid rgba(6, 182, 212, 0.8)' : '1px solid var(--glass-border)',
                  background: isSelected ? 'rgba(6, 182, 212, 0.15)' : 'rgba(0,0,0,0.3)',
                  color: '#f8fafc',
                  fontWeight: 600,
                  fontSize: '0.82rem',
                  cursor: isDis ? 'not-allowed' : 'pointer',
                  opacity: isDis ? 0.45 : 1,
                  transition: 'border 0.12s ease, background 0.12s ease',
                }}
              >
                {choiceDisplayLabel(f.id, s)}
                {isDis && (
                  <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginTop: 2 }}>soon</div>
                )}
              </button>
            )
          })}
        </div>
      )
    }

    if (f.display === 'model_cards' && f.id === 'sway_yolo_weights') {
      return (
        <div
          className="options-grid"
          style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '0.55rem' }}
        >
          {f.choices.map((c) => {
            const s = String(c)
            const isSelected = String(v ?? '') === s
            const meta = YOLO_CARD_META[s] ?? { accent: '#fbbf24', badge: 'Model', hint: '' }
            const fileKey = `${s}.pt`
            const missing = modelsStatus && modelsStatus[fileKey] === false
            const recommended = s === 'yolo26l' && modelsStatus && modelsStatus['yolo26l.pt'] === true
            const title = choiceDisplayLabel('sway_yolo_weights', s)
            return (
              <div
                key={s}
                className={`option-card ${isSelected ? 'selected' : ''}`}
                onClick={() => onChange(s)}
                style={{
                  opacity: missing ? 0.55 : 1,
                  position: 'relative',
                  padding: '0.65rem 0.55rem',
                  minHeight: 88,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  gap: '0.35rem',
                }}
              >
                <div style={{ fontWeight: 700, fontSize: '0.78rem', lineHeight: 1.25, color: '#f8fafc' }}>{title}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', alignItems: 'center' }}>
                  <span
                    style={{
                      fontSize: '0.62rem',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      letterSpacing: '0.04em',
                      padding: '0.12rem 0.35rem',
                      borderRadius: 6,
                      background: 'rgba(0,0,0,0.35)',
                      color: meta.accent,
                    }}
                  >
                    {meta.badge}
                  </span>
                  {meta.hint && (
                    <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>{meta.hint}</span>
                  )}
                </div>
                {missing && (
                  <span
                    style={{
                      fontSize: '0.58rem',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      color: '#94a3b8',
                      letterSpacing: '0.05em',
                    }}
                  >
                    File not downloaded
                  </span>
                )}
                {recommended && (
                  <span
                    style={{
                      position: 'absolute',
                      top: 6,
                      right: 6,
                      fontSize: '0.55rem',
                      fontWeight: 800,
                      textTransform: 'uppercase',
                      color: '#6ee7b7',
                      letterSpacing: '0.06em',
                    }}
                  >
                    Recommended
                  </span>
                )}
              </div>
            )
          })}
        </div>
      )
    }

    if (f.display === 'model_cards' && f.id === 'pose_model') {
      return (
        <div
          className="options-grid"
          style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '0.55rem' }}
        >
          {f.choices.map((c) => {
            const s = String(c)
            const isSelected = String(v ?? '') === s
            const meta = POSE_CARD_META[s] ?? { accent: '#fbbf24', badge: s, hint: '' }
            return (
              <div
                key={s}
                className={`option-card ${isSelected ? 'selected' : ''}`}
                onClick={() => onChange(s)}
                style={{
                  padding: '0.65rem 0.5rem',
                  minHeight: 80,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  gap: '0.3rem',
                }}
              >
                <div style={{ fontWeight: 700, fontSize: '0.75rem', color: '#f8fafc' }}>
                  {choiceDisplayLabel(f.id, s)}
                </div>
                <span
                  style={{
                    fontSize: '0.62rem',
                    fontWeight: 700,
                    padding: '0.1rem 0.32rem',
                    borderRadius: 6,
                    background: 'rgba(0,0,0,0.35)',
                    color: meta.accent,
                    alignSelf: 'flex-start',
                  }}
                >
                  {meta.badge}
                </span>
                {meta.hint && <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>{meta.hint}</span>}
              </div>
            )
          })}
        </div>
      )
    }

    return (
      <div className="options-grid">
        {f.choices.map((c) => {
          const s = String(c)
          const isSelected = String(v ?? '') === s
          const isDis = disabled.has(s)
          return (
            <div
              key={s}
              className={`option-card ${isSelected ? 'selected' : ''} ${isDis ? 'disabled' : ''}`}
              style={isDis ? { opacity: 0.45, cursor: 'not-allowed' } : undefined}
              onClick={() => {
                if (isDis) return
                const num = Number(s)
                onChange(Number.isNaN(num) ? s : num)
              }}
              title={isDis ? 'Not available in this version yet' : undefined}
            >
              {choiceDisplayLabel(f.id, s)}
              {isDis && (
                <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>coming soon</div>
              )}
            </div>
          )
        })}
      </div>
    )
  }

  if ((f.type === 'int' || f.type === 'float') && (f.display === 'slider' || f.display === 'pruning_weight')) {
    const min = f.min ?? 0
    const max = f.max ?? 1
    const num = v === undefined || v === null || v === '' ? Number(f.default ?? min) : Number(v)
    const safe = Number.isFinite(num) ? num : Number(min)
    const step = sliderStep(f)
    const viz =
      f.display === 'pruning_weight' || f.display === 'slider'
        ? renderConfigVisual(f.id, safe, min, max, allFields)
        : null

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
        {viz}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={safe}
          onChange={(e) => {
            const x = f.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value)
            onChange(x)
          }}
          className="config-range-input"
          style={{ width: '100%', accentColor: 'var(--halo-cyan, #06b6d4)' }}
        />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{min}</span>
          <input
            type="number"
            className="text-input config-slider-num"
            step={step}
            min={f.min}
            max={f.max}
            value={safe}
            onChange={(e) => {
              const raw = f.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value)
              if (!Number.isFinite(raw)) return
              let out = raw
              if (f.min !== undefined) out = Math.max(f.min, out)
              if (f.max !== undefined) out = Math.min(f.max, out)
              if (f.type === 'int') out = Math.round(out)
              onChange(out)
            }}
            style={{
              width: 88,
              padding: '0.35rem 0.5rem',
              fontSize: '0.85rem',
              fontWeight: 700,
              textAlign: 'center',
              borderRadius: 8,
            }}
          />
          <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{max}</span>
        </div>
      </div>
    )
  }

  if (f.type === 'int' || f.type === 'float') {
    const fallback =
      typeof f.default === 'number' && Number.isFinite(f.default)
        ? f.default
        : f.type === 'int'
          ? 0
          : 0
    const effective =
      v === undefined || v === null || (typeof v === 'number' && Number.isNaN(v)) ? fallback : v
    const safeNum = typeof effective === 'number' && Number.isFinite(effective) ? effective : fallback
    return (
      <input
        className="text-input"
        type="number"
        step={f.type === 'float' ? 'any' : 1}
        min={f.min}
        max={f.max}
        value={String(safeNum)}
        onChange={(e) => {
          const s = e.target.value
          if (s === '' || s === '-' || s === '.' || s === '-.') {
            onChange(fallback)
            return
          }
          const parsed = f.type === 'int' ? parseInt(s, 10) : parseFloat(s)
          if (!Number.isFinite(parsed)) {
            onChange(fallback)
            return
          }
          let out = parsed
          if (f.min !== undefined) out = Math.max(f.min, out)
          if (f.max !== undefined) out = Math.min(f.max, out)
          if (f.type === 'int') out = Math.round(out)
          onChange(out)
        }}
        style={{ padding: '0.3rem 0.5rem', fontSize: '0.8rem' }}
      />
    )
  }

  return (
    <input
      className="text-input"
      type="text"
      value={v === undefined || v === null ? '' : String(v)}
      onChange={(e) => onChange(e.target.value || undefined)}
      placeholder={String(f.default ?? '')}
      style={{ padding: '0.3rem 0.5rem', fontSize: '0.8rem' }}
    />
  )
}
