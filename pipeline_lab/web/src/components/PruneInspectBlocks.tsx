import type { PruneEntry } from '../lib/watchPrune'

function formatCauseParamValue(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  if (typeof v === 'string') return v
  if (Array.isArray(v)) return JSON.stringify(v)
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

export function CauseConfigBlock({ e }: { e: PruneEntry }) {
  const cc = e.cause_config as
    | { filter?: string; summary?: string; params?: Record<string, unknown> }
    | undefined
  if (!cc || typeof cc !== 'object') return null
  const paramEntries =
    cc.params && typeof cc.params === 'object' ? Object.entries(cc.params) : []
  if (!cc.summary && !cc.filter && paramEntries.length === 0) return null
  return (
    <div
      style={{
        marginTop: '0.35rem',
        padding: '0.45rem 0.55rem',
        borderRadius: 8,
        background: 'rgba(30, 41, 59, 0.55)',
        border: '1px solid rgba(148, 163, 184, 0.2)',
        fontSize: '0.7rem',
        lineHeight: 1.45,
        color: '#cbd5e1',
      }}
      onClick={(ev) => ev.stopPropagation()}
    >
      <div style={{ fontWeight: 600, color: '#93c5fd', marginBottom: '0.2rem' }}>Filter / configuration</div>
      {cc.filter != null && cc.filter !== '' && (
        <div>
          <span style={{ color: 'var(--text-muted)' }}>id </span>
          {cc.filter}
        </div>
      )}
      {cc.summary != null && cc.summary !== '' && (
        <div style={{ marginTop: '0.15rem' }}>{cc.summary}</div>
      )}
      {paramEntries.length > 0 && (
        <div
          style={{
            marginTop: '0.25rem',
            fontFamily: 'ui-monospace, monospace',
            fontSize: '0.65rem',
            color: '#e2e8f0',
            wordBreak: 'break-word',
          }}
        >
          {paramEntries.map(([k, v]) => (
            <div key={k}>
              {k}: {formatCauseParamValue(v)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function TierBBlock({ e }: { e: PruneEntry }) {
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
