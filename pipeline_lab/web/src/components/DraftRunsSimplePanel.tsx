import type { DraftRun } from '../context/LabContext'
import { Pencil, Copy, Trash2, LayoutGrid } from 'lucide-react'

export function DraftRunsSimplePanel({
  roots,
  onEdit,
  onDuplicate,
  onRemoveRootBranch,
}: {
  roots: DraftRun[]
  onEdit: (id: string) => void
  onDuplicate: (id: string) => void
  onRemoveRootBranch: (rootClientId: string) => void
}) {
  return (
    <div style={{ marginTop: '1.15rem' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          marginBottom: '0.65rem',
          fontSize: '0.72rem',
          fontWeight: 700,
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
        }}
      >
        <LayoutGrid size={14} aria-hidden />
        Runs in this batch
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 300px), 1fr))',
          gap: '0.85rem',
        }}
      >
        {roots.map((d, idx) => (
          <div
            key={d.clientId}
            className="glass-panel"
            style={{
              padding: '0.95rem 1.05rem',
              borderRadius: 14,
              border: '1px solid var(--glass-border)',
              background: 'rgba(15, 23, 42, 0.35)',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.65rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.75rem' }}>
              <span
                style={{
                  fontSize: '0.65rem',
                  fontWeight: 800,
                  letterSpacing: '0.06em',
                  padding: '0.2rem 0.45rem',
                  borderRadius: 6,
                  border: '1px solid rgba(56, 189, 248, 0.45)',
                  background: 'rgba(14, 165, 233, 0.14)',
                  color: '#7dd3fc',
                  flexShrink: 0,
                }}
              >
                Run {idx + 1}
              </span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', justifyContent: 'flex-end' }}>
                <button type="button" className="btn" style={{ fontSize: '0.78rem', padding: '0.35rem 0.55rem' }} onClick={() => onEdit(d.clientId)}>
                  <Pencil size={14} aria-hidden /> Edit
                </button>
                <button
                  type="button"
                  className="btn"
                  style={{ fontSize: '0.78rem', padding: '0.35rem 0.55rem' }}
                  onClick={() => onDuplicate(d.clientId)}
                >
                  <Copy size={14} aria-hidden /> Duplicate
                </button>
                <button
                  type="button"
                  className="btn"
                  style={{ fontSize: '0.78rem', padding: '0.35rem 0.55rem', color: '#f87171', borderColor: 'rgba(248,113,113,0.35)' }}
                  onClick={() => onRemoveRootBranch(d.clientId)}
                  disabled={roots.length <= 1}
                  title={roots.length <= 1 ? 'Keep at least one run' : 'Remove this run (and its checkpoint children, if any)'}
                >
                  <Trash2 size={14} aria-hidden /> Remove
                </button>
              </div>
            </div>
            <div>
              <div style={{ fontWeight: 700, color: '#fff', fontSize: '1rem' }}>{d.recipeName}</div>
              <p style={{ margin: '0.35rem 0 0', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
                Full pipeline — one queued job from detect through export. Use the editor for all stages.
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
