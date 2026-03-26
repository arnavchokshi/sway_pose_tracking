import type { Phase13StrategyId } from '../configPresets'
import { phase13EffectiveRows } from '../lib/labPhase13Ui'

export function Phase13EffectivePanel({
  strategy,
  fieldsState,
}: {
  strategy: Phase13StrategyId
  fieldsState: Record<string, unknown>
}) {
  const rows = phase13EffectiveRows(strategy, fieldsState)
  return (
    <div
      style={{
        borderRadius: 12,
        border: '1px solid rgba(34, 211, 238, 0.35)',
        background: 'rgba(6, 78, 95, 0.18)',
        padding: '0.85rem 1rem',
      }}
    >
      <div
        style={{
          fontSize: '0.68rem',
          fontWeight: 800,
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          color: '#7dd3fc',
          marginBottom: '0.55rem',
        }}
      >
        Effective Phases 1–3 configuration
      </div>
      <p style={{ margin: '0 0 0.65rem', fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
        What actually runs for the strategy above (including values the Lab API fixes for you). Sliders elsewhere only apply
        when listed here.
      </p>
      <ul style={{ margin: 0, paddingLeft: '1.1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {rows.map((r) => (
          <li key={r.label} style={{ fontSize: '0.78rem', color: '#e2e8f0', lineHeight: 1.45 }}>
            <span style={{ fontWeight: 700, color: '#f0f9ff' }}>{r.label}: </span>
            {r.value}
            {r.detail ? (
              <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>{r.detail}</span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  )
}
