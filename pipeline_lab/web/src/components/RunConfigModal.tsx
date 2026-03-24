import { useEffect, useMemo, useState } from 'react'
import { API } from '../types'
import type { Schema } from '../types'
import { X, ClipboardList, Loader2 } from 'lucide-react'

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return v ? 'On' : 'Off'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

type ConfigPayload = {
  recipe_name: string
  video_stem: string
  fields: Record<string, unknown>
  request_meta: Record<string, unknown>
  params_yaml: Record<string, unknown> | null
}

export function RunConfigModal({
  open,
  runId,
  schema,
  onClose,
}: {
  open: boolean
  runId: string | null
  schema: Schema | null
  onClose: () => void
}) {
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [data, setData] = useState<ConfigPayload | null>(null)

  useEffect(() => {
    if (!open || !runId) {
      setData(null)
      setErr(null)
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setErr(null)
    setData(null)
    fetch(`${API}/api/runs/${runId}/config`)
      .then(async (r) => {
        const text = await r.text()
        if (!r.ok) {
          let msg = `HTTP ${r.status}`
          try {
            const j = JSON.parse(text) as { detail?: string }
            if (typeof j.detail === 'string') msg = j.detail
          } catch {
            if (text.trim()) msg = text.slice(0, 200)
          }
          throw new Error(msg)
        }
        return JSON.parse(text) as ConfigPayload
      })
      .then((j) => {
        if (!cancelled) setData(j)
      })
      .catch((e) => {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, runId])

  const orderedRows = useMemo(() => {
    if (!data?.fields) return []
    const fields = data.fields
    const seen = new Set<string>()
    const rows: { id: string; label: string; value: string }[] = []
    if (schema) {
      for (const f of schema.fields) {
        if (!(f.id in fields)) continue
        seen.add(f.id)
        rows.push({ id: f.id, label: f.label, value: formatValue(fields[f.id]) })
      }
    }
    for (const id of Object.keys(fields).sort()) {
      if (seen.has(id)) continue
      rows.push({ id, label: id, value: formatValue(fields[id]) })
    }
    return rows
  }, [data, schema])

  if (!open || !runId) return null

  const shortId = runId.length > 12 ? `${runId.slice(0, 8)}…` : runId

  return (
    <div
      role="dialog"
      aria-modal
      aria-labelledby="run-config-title"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        background: 'rgba(0,0,0,0.65)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem',
        overflow: 'auto',
      }}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="glass-panel"
        style={{
          width: 'min(640px, 100%)',
          maxHeight: 'min(88vh, 820px)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: '1rem',
            padding: '1rem 1.15rem',
            borderBottom: '1px solid var(--glass-border)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.65rem' }}>
            <ClipboardList size={22} style={{ color: 'var(--halo-cyan)', flexShrink: 0, marginTop: 2 }} />
            <div>
              <h2 id="run-config-title" style={{ margin: 0, fontSize: '1.1rem', color: '#fff' }}>
                Run configuration
              </h2>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem', fontFamily: 'ui-monospace, monospace' }}>
                {shortId}
              </div>
            </div>
          </div>
          <button
            type="button"
            aria-label="Close"
            onClick={onClose}
            style={{
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid var(--glass-border)',
              borderRadius: 8,
              padding: '0.35rem',
              cursor: 'pointer',
              color: 'var(--text-muted)',
              lineHeight: 0,
            }}
          >
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '1rem 1.15rem', overflow: 'auto', flex: 1 }}>
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              <Loader2 size={18} className="sway-spin" /> Loading…
            </div>
          )}
          {err && !loading && <div style={{ color: '#f87171', fontSize: '0.9rem', lineHeight: 1.5 }}>{err}</div>}
          {data && !loading && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
              <div>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                  Overview
                </div>
                <div style={{ fontSize: '0.88rem', color: '#e2e8f0', lineHeight: 1.55 }}>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Recipe name:</span>{' '}
                    <strong>{data.recipe_name?.trim() ? data.recipe_name : '—'}</strong>
                  </div>
                  <div style={{ marginTop: '0.25rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Source video (stem):</span>{' '}
                    <strong>{data.video_stem || '—'}</strong>
                  </div>
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                  Settings used for this run
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.45rem', lineHeight: 1.45 }}>
                  Effective parameters: schema defaults merged with what was queued for this run. Expand{' '}
                  <span style={{ fontStyle: 'italic' }}>Raw request.json</span> below if the stored file still has an empty{' '}
                  <span style={{ fontFamily: 'ui-monospace, monospace' }}>fields</span> object from an older Lab version.
                </div>
                {orderedRows.length === 0 ? (
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No tunable fields to show.</div>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                    <tbody>
                      {orderedRows.map((row) => (
                        <tr key={row.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                          <td
                            style={{
                              padding: '0.45rem 0.5rem 0.45rem 0',
                              color: 'var(--text-muted)',
                              verticalAlign: 'top',
                              width: '42%',
                            }}
                          >
                            {row.label}
                          </td>
                          <td style={{ padding: '0.45rem 0', color: '#f1f5f9', verticalAlign: 'top', wordBreak: 'break-word' }}>
                            {row.value}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>

              {data.params_yaml && Object.keys(data.params_yaml).length > 0 && (
                <div>
                  <div style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                    Resolved YAML params (main.py --params)
                  </div>
                  <pre
                    style={{
                      margin: 0,
                      padding: '0.65rem 0.75rem',
                      borderRadius: 8,
                      background: 'rgba(0,0,0,0.45)',
                      border: '1px solid var(--glass-border)',
                      color: '#cbd5e1',
                      fontSize: '0.72rem',
                      lineHeight: 1.45,
                      overflow: 'auto',
                      maxHeight: 200,
                    }}
                  >
                    {JSON.stringify(data.params_yaml, null, 2)}
                  </pre>
                </div>
              )}

              <details style={{ fontSize: '0.78rem' }}>
                <summary style={{ cursor: 'pointer', color: 'var(--halo-cyan)', fontWeight: 600 }}>
                  Raw request.json (debug)
                </summary>
                <pre
                  style={{
                    marginTop: '0.5rem',
                    padding: '0.65rem 0.75rem',
                    borderRadius: 8,
                    background: 'rgba(0,0,0,0.45)',
                    border: '1px solid var(--glass-border)',
                    color: '#94a3b8',
                    fontSize: '0.68rem',
                    lineHeight: 1.4,
                    overflow: 'auto',
                    maxHeight: 220,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {JSON.stringify(data.request_meta, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
