import { useEffect, useState, useMemo } from 'react'
import { API } from '../types'
import type { Schema, SchemaField } from '../types'
import {
  applyConfigPreset,
  CONFIG_PRESET_LABELS,
  CONFIG_PRESET_ORDER,
  defaultsFromSchema,
} from '../configPresets'
import { InlineFieldInput } from '../components/InlineFieldInput'
import { HYBRID_SAM_PHASE_ID, hideHybridSamPhase } from '../lib/labFieldVisibility'
import { mainPyPhaseCaption } from '../lib/schemaStageCaption'
import { Save, Copy } from 'lucide-react'

export function ConfigPage() {
  const [schema, setSchema] = useState<Schema | null>(null)
  const [schemaError, setSchemaError] = useState<string | null>(null)
  const [activePhaseIndex, setActivePhaseIndex] = useState(0)
  const [configName, setConfigName] = useState('My Config')
  const [fieldsState, setFieldsState] = useState<Record<string, unknown>>({})
  const [showAdvanced, setShowAdvanced] = useState(false)

  useEffect(() => {
    let cancelled = false
    setSchemaError(null)
    fetch(`${API}/api/schema`)
      .then(async (r) => {
        const text = await r.text()
        if (!r.ok) {
          if (r.status === 502 || r.status === 503) {
            throw new Error(
              'Pipeline API is not reachable (bad gateway). The Vite dev server proxies /api to http://127.0.0.1:8765 — start uvicorn there in another terminal.',
            )
          }
          throw new Error(`Schema request failed (HTTP ${r.status}).`)
        }
        if (!text.trim()) throw new Error('Empty response from /api/schema.')
        try {
          return JSON.parse(text) as Schema
        } catch {
          throw new Error('Invalid JSON from /api/schema.')
        }
      })
      .then((s) => {
        if (!cancelled) {
          setSchema(s)
          setFieldsState(defaultsFromSchema(s.fields))
        }
      })
      .catch((e) => {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e)
          setSchemaError(msg)
          console.error(e)
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  const fieldsByPhase = useMemo(() => {
    if (!schema) return new Map<string, SchemaField[]>()
    const m = new Map<string, SchemaField[]>()
    for (const f of schema.fields) {
      if (f.advanced && !showAdvanced) continue
      const arr = m.get(f.phase) || []
      arr.push(f)
      m.set(f.phase, arr)
    }
    return m
  }, [schema, showAdvanced])

  if (schemaError) {
    return (
      <div className="glass-panel" style={{ padding: '2rem 2.5rem', maxWidth: 640, margin: '0 auto' }}>
        <p style={{ color: '#f87171', margin: '0 0 1rem', lineHeight: 1.5 }}>{schemaError}</p>
        <p style={{ color: 'var(--text-muted)', margin: '0 0 1rem', fontSize: '0.95rem', lineHeight: 1.5 }}>
          From the <code style={{ color: 'var(--halo-cyan)' }}>sway_pose_mvp</code> directory (after installing{' '}
          <code style={{ color: 'var(--halo-cyan)' }}>pipeline_lab/server/requirements.txt</code>):
        </p>
        <pre
          style={{
            margin: 0,
            padding: '1rem',
            borderRadius: 12,
            background: 'rgba(0,0,0,0.35)',
            border: '1px solid var(--glass-border)',
            color: 'var(--text-main)',
            fontSize: '0.85rem',
            overflow: 'auto',
            lineHeight: 1.45,
          }}
        >
          {`cd sway_pose_mvp
uvicorn pipeline_lab.server.app:app --reload --host 127.0.0.1 --port 8765`}
        </pre>
        <p style={{ color: 'var(--text-muted)', margin: '1rem 0 0', fontSize: '0.9rem' }}>
          Keep that running while <code style={{ color: 'var(--halo-cyan)' }}>npm run dev</code> is active.
        </p>
      </div>
    )
  }

  if (!schema) {
    return <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>Loading schema...</div>
  }

  const stages = schema.stages.filter(s => s.id !== 'export')
  const activeStage = stages[activePhaseIndex]
  const activeFields = fieldsByPhase.get(activeStage?.id) || []
  const hybridSamHidden =
    activeStage?.id === HYBRID_SAM_PHASE_ID && hideHybridSamPhase(fieldsState.tracker_technology)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
      {/* Header */}
      <div className="glass-panel" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 2rem' }}>
        <input 
          value={configName} 
          onChange={e => setConfigName(e.target.value)}
          style={{ background: 'transparent', border: 'none', borderBottom: '1px solid var(--halo-cyan)', color: '#fff', fontSize: '1.5rem', fontWeight: 600, outline: 'none', padding: '0.2rem 0' }}
        />
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
          {CONFIG_PRESET_ORDER.map((id) => (
            <button
              key={id}
              type="button"
              className="btn"
              style={{ whiteSpace: 'nowrap' }}
              onClick={() => setFieldsState(applyConfigPreset(schema, id))}
            >
              {CONFIG_PRESET_LABELS[id]}
            </button>
          ))}
          <button className="btn"><Copy size={16}/> Duplicate</button>
          <button className="btn primary"><Save size={16}/> Save Config</button>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', flex: 1 }}>
        {/* Top: Flowchart */}
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h2 style={{ margin: 0, fontSize: '1.2rem', color: '#fff' }}>Pipeline Flow</h2>
              <p style={{ margin: '0.35rem 0 0', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
                Phase labels match <code style={{ color: 'var(--halo-cyan)' }}>main.py</code> [1/11]…[11/11] and{' '}
                <code style={{ color: 'var(--halo-cyan)' }}>docs/PIPELINE_CODE_REFERENCE.md</code>.
              </p>
            </div>
            <label className="checkbox-label" style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem 1rem', borderRadius: '8px' }}>
              <input type="checkbox" checked={showAdvanced} onChange={e => setShowAdvanced(e.target.checked)} style={{ display: 'none' }}/>
              <div className="checkbox-visual"></div>
              Show Advanced Params
            </label>
          </div>
          
          <div className="flowchart-board" style={{ position: 'relative', overflowX: 'auto', padding: '2rem 1rem', display: 'flex', alignItems: 'center', gap: '1.5rem', minHeight: 'auto', background: 'transparent' }}>
            {stages.map((stage, idx) => {
              const isActive = idx === activePhaseIndex
              const isDone = idx < activePhaseIndex
              
              return (
                <div key={stage.id} style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                  <div 
                    onClick={() => setActivePhaseIndex(idx)}
                    className={`node-box ${isActive ? 'selected' : ''} ${isDone ? 'status-done' : ''}`}
                    style={{
                      border: isActive ? '2px solid var(--halo-cyan)' : '1px solid var(--glass-border)',
                      background: isActive ? 'rgba(14, 165, 233, 0.15)' : 'rgba(15, 20, 30, 0.7)',
                      padding: '1.2rem 1.5rem',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '0.5rem',
                      minWidth: '200px'
                    }}
                  >
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>{mainPyPhaseCaption(stage, idx)}</div>
                    <div style={{ fontWeight: 600, color: isActive ? '#fff' : 'var(--text-main)', fontSize: '1rem' }}>{stage.label}</div>
                  </div>
                  
                  {/* Arrow to next node */}
                  {idx < stages.length - 1 && (
                    <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
                      <div style={{ height: '2px', width: '30px', background: isDone ? 'var(--halo-cyan)' : 'var(--glass-border)' }}></div>
                      <div style={{ 
                        width: 0, height: 0, 
                        borderTop: '6px solid transparent', 
                        borderBottom: '6px solid transparent', 
                        borderLeft: `8px solid ${isDone ? 'var(--halo-cyan)' : 'var(--glass-border)'}`,
                        marginLeft: '-1px'
                      }}></div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Bottom Active Params */}
        <div className="glass-panel" style={{ padding: '2rem', flex: 1 }}>
          <h2 style={{ marginTop: 0, marginBottom: '2rem', color: '#fff', fontSize: '1.5rem' }}>{activeStage?.label} Parameters</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '2rem' }}>
            {hybridSamHidden ? (
              <div
                style={{
                  gridColumn: '1 / -1',
                  padding: '1.15rem 1.35rem',
                  borderRadius: 12,
                  border: '1px solid rgba(148, 163, 184, 0.35)',
                  background: 'rgba(30, 41, 59, 0.45)',
                  color: 'var(--text-muted)',
                  fontSize: '0.92rem',
                  lineHeight: 1.55,
                }}
              >
                <strong style={{ color: '#e2e8f0' }}>Hybrid SAM</strong> runs only on the{' '}
                <strong style={{ color: '#e2e8f0' }}>BoxMOT</strong> path. With <strong style={{ color: '#e2e8f0' }}>BoT-SORT</strong>{' '}
                selected, these settings are hidden (they would have no effect). Switch Tracker backend to BoxMOT to tune
                the overlap refiner.
              </div>
            ) : (
              <>
                {activeFields.map((f) => (
                  <div key={f.id} style={{ background: 'rgba(0,0,0,0.2)', padding: '1.2rem', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <label style={{ color: '#fff', fontWeight: 500 }}>{f.label}</label>
                    </div>
                    {f.type !== 'info' && f.description && (
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem', fontStyle: 'italic' }}>{f.description}</div>
                    )}
                    {f.type === 'info' ? (
                      <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>{f.description}</div>
                    ) : (
                      <InlineFieldInput
                        f={f}
                        value={fieldsState[f.id]}
                        onChange={(v) => setFieldsState((prev) => ({ ...prev, [f.id]: v }))}
                      />
                    )}
                  </div>
                ))}
                {activeFields.length === 0 && (
                  <div style={{ color: 'var(--text-muted)' }}>No configurable parameters for this phase.</div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
