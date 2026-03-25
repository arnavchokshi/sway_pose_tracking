import type { SchemaField } from '../types'
import type { ReactNode } from 'react'
import { SchemaFieldVisual } from './SchemaVisuals'

/** Compact label + optional help (full text in native tooltip). */
export function ConfigFieldWrap({
  field,
  value,
  children,
  variant = 'default',
}: {
  field: SchemaField
  value?: any
  children: ReactNode
  variant?: 'default' | 'dense'
}) {
  const pad = variant === 'dense' ? '0.85rem' : '1rem'
  return (
    <div
      className="config-field-card"
      style={{
        background: 'rgba(0,0,0,0.22)',
        padding: pad,
        borderRadius: 12,
        border: '1px solid var(--glass-border)',
      }}
    >
      {field.type !== 'info' && (
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: '0.5rem',
            marginBottom: '0.55rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap', minWidth: 0 }}>
            <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '0.88rem', lineHeight: 1.35 }}>{field.label}</span>
            {(field.advanced === true || field.tier === 3) && (
              <span
                style={{
                  fontSize: '0.58rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  color: '#a5b4fc',
                  border: '1px solid rgba(165, 180, 252, 0.45)',
                  borderRadius: 4,
                  padding: '0.1rem 0.35rem',
                  flexShrink: 0,
                }}
              >
                Advanced
              </span>
            )}
          </div>
          {field.description ? (
            <span
              className="config-field-help"
              title={field.description}
              tabIndex={0}
              role="note"
              style={{
                flexShrink: 0,
                width: 22,
                height: 22,
                borderRadius: 999,
                display: 'grid',
                placeItems: 'center',
                fontSize: '0.72rem',
                fontWeight: 700,
                color: 'var(--text-muted)',
                border: '1px solid rgba(148,163,184,0.35)',
                cursor: 'help',
                background: 'rgba(0,0,0,0.25)',
              }}
            >
              ?
            </span>
          ) : null}
        </div>
      )}
      <SchemaFieldVisual fieldId={field.id} value={value} />
      {children}
    </div>
  )
}

/** Collapsible info strip — keeps phase intros one line until expanded. */
export function ConfigInfoFold({ title, body }: { title: string; body: string }) {
  return (
    <details
      className="config-info-fold"
      style={{
        borderRadius: 10,
        border: '1px solid rgba(148, 163, 184, 0.22)',
        background: 'rgba(0,0,0,0.18)',
        padding: '0.5rem 0.75rem',
        marginBottom: '0.75rem',
      }}
    >
      <summary
        style={{
          cursor: 'pointer',
          fontSize: '0.78rem',
          fontWeight: 600,
          color: '#94a3b8',
          listStyle: 'none',
        }}
      >
        {title}
      </summary>
      <p style={{ margin: '0.5rem 0 0', fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{body}</p>
    </details>
  )
}
