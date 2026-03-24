import type { SchemaField } from '../types'

const YOLO_CARD_META: Record<
  string,
  { speed: 'fast' | 'med' | 'slow'; acc: 'base' | 'high' | 'max'; hint?: string }
> = {
  yolo26s: { speed: 'fast', acc: 'base', hint: 'Preview' },
  yolo26l: { speed: 'med', acc: 'high', hint: 'COCO L' },
  yolo26l_dancetrack: { speed: 'med', acc: 'high', hint: 'DanceTrack fine-tune' },
  yolo26x: { speed: 'slow', acc: 'max', hint: 'COCO X' },
  yolo26x_dancetrack: { speed: 'slow', acc: 'max', hint: 'DanceTrack X' },
  yolov11l: { speed: 'med', acc: 'base', hint: 'Legacy baseline' },
  yolov11x: { speed: 'slow', acc: 'high', hint: 'Legacy baseline' },
}

function speedAccent(s: string) {
  if (s === 'fast') return '#34d399'
  if (s === 'med') return '#fbbf24'
  return '#f97316'
}

export function InlineFieldInput({
  f,
  value,
  onChange,
  modelsStatus,
}: {
  f: SchemaField
  value: unknown
  onChange: (v: unknown) => void
  modelsStatus?: Record<string, boolean> | null
}) {
  const v = value === undefined ? f.default : value

  if (f.type === 'bool') {
    return (
      <label className="checkbox-label" style={{ marginTop: '0.2rem' }}>
        <input
          type="checkbox"
          checked={Boolean(v)}
          onChange={(e) => onChange(e.target.checked)}
          style={{ display: 'none' }}
        />
        <div className="checkbox-visual" style={{ width: 14, height: 14 }}></div>
      </label>
    )
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
                  padding: '0.45rem 0.65rem',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  border: 'none',
                  cursor: isDis ? 'not-allowed' : 'pointer',
                  opacity: isDis ? 0.45 : 1,
                  background: isSelected ? 'rgba(14, 165, 233, 0.35)' : 'rgba(0,0,0,0.25)',
                  color: '#e2e8f0',
                }}
              >
                {s}
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
            const meta = YOLO_CARD_META[s] ?? { speed: 'med', acc: 'high' }
            const fileKey =
              s === 'yolo26l_dancetrack'
                ? 'yolo26l_dancetrack.pt'
                : s === 'yolo26x_dancetrack'
                  ? 'yolo26x_dancetrack.pt'
                  : null
            const missing = fileKey && modelsStatus && modelsStatus[fileKey] === false
            const recommended = s === 'yolo26l_dancetrack' && modelsStatus && modelsStatus['yolo26l_dancetrack.pt'] === true
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
                <div style={{ fontWeight: 700, fontSize: '0.78rem', lineHeight: 1.25, color: '#f8fafc' }}>{s}</div>
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
                      color: speedAccent(meta.speed),
                    }}
                  >
                    {meta.speed} · {meta.acc}
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
                    Weights missing
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
              title={isDis ? 'Coming soon — not wired in this build' : undefined}
            >
              {s}
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
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
        <input
          type="range"
          min={min}
          max={max}
          step={f.type === 'int' ? 1 : 0.01}
          value={safe}
          onChange={(e) => {
            const x = f.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value)
            onChange(x)
          }}
          style={{ width: '100%', accentColor: 'var(--halo-cyan, #06b6d4)' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          <span>{min}</span>
          <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{safe}</span>
          <span>{max}</span>
        </div>
        {f.id === 'sway_hybrid_sam_iou_trigger' && (
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            SAM runs when any pair of dancer boxes exceeds IoU {safe.toFixed(2)}.
          </div>
        )}
        {f.id === 'sway_boxmot_match_thresh' && (
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            Left = more merges · right = more fragments.
          </div>
        )}
        {f.id === 'sway_pretrack_nms_iou' && (
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            Lower = stronger ghost suppression before tracking.
          </div>
        )}
        {f.id === 'sway_boxmot_max_age' && (
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            ≈ {(safe / 30).toFixed(1)}s @ 30fps — adjust for your video frame rate.
          </div>
        )}
        {f.id === 'sway_stitch_max_frame_gap' && (
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            ≈ {(safe / 30).toFixed(1)}s @ 30fps bridge horizon for fragment stitch.
          </div>
        )}
      </div>
    )
  }

  if (f.type === 'int' || f.type === 'float') {
    return (
      <input
        className="text-input"
        type="number"
        step={f.type === 'float' ? 'any' : 1}
        min={f.min}
        max={f.max}
        value={v === undefined || v === null ? '' : String(v)}
        onChange={(e) => {
          const s = e.target.value
          if (s === '') return onChange(undefined)
          onChange(f.type === 'int' ? parseInt(s, 10) : parseFloat(s))
        }}
        placeholder={String(f.default ?? '')}
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
