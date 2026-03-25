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
import { ConfigFieldWrap, ConfigInfoFold } from '../components/ConfigFieldWrap'
import { HYBRID_SAM_PHASE_ID, hideHybridSamPhase } from '../lib/labFieldVisibility'
import { isSchemaFieldVisible } from '../lib/labFieldMeta'
import { mainPyPhaseCaption } from '../lib/schemaStageCaption'
import {
  buildPipelineFlowchartRows,
  flowchartRowIsActive,
  flowchartRowIsDone,
} from '../lib/flowchartRows'
import { ComplexityVisualizer } from '../components/ComplexityVisualizer'

export function ConfigPage() {
  const [schema, setSchema] = useState<Schema | null>(null)
  const [schemaError, setSchemaError] = useState<string | null>(null)
  const [activePhaseIndex, setActivePhaseIndex] = useState(0)
  const [configName, setConfigName] = useState('My Config')
  const [fieldsState, setFieldsState] = useState<Record<string, unknown>>({})
  const [activePreset, setActivePreset] = useState<string | null>('standard')

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
      const arr = m.get(f.phase) || []
      arr.push(f)
      m.set(f.phase, arr)
    }
    return m
  }, [schema])

  const flowchartRows = useMemo(() => (schema ? buildPipelineFlowchartRows(schema.stages) : []), [schema])

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

  const stages = schema.stages
  const activeStage = stages[activePhaseIndex]
  const activeFieldsRaw = fieldsByPhase.get(activeStage?.id) || []
  const hybridSamHidden =
    activeStage?.id === HYBRID_SAM_PHASE_ID && hideHybridSamPhase(fieldsState.tracker_technology)
  const activeFields = hybridSamHidden
    ? activeFieldsRaw
    : activeFieldsRaw.filter((f) => f.type === 'info' || isSchemaFieldVisible(f, fieldsState))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', height: '100%' }}>
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
              style={{ 
                whiteSpace: 'nowrap',
                borderColor: activePreset === id ? '#fff' : undefined,
                background: activePreset === id ? 'rgba(255,255,255,0.15)' : undefined
              }}
              onClick={() => {
                setFieldsState(applyConfigPreset(schema, id))
                setActivePreset(id)
              }}
            >
              {CONFIG_PRESET_LABELS[id]}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', flex: 1 }}>
        {/* Top: Flowchart */}
        <div className="glass-panel" style={{ padding: '1.25rem 1.5rem' }}>
          <div style={{ marginBottom: '1rem' }}>
            <h2 style={{ margin: 0, fontSize: '1.2rem', color: '#fff' }}>Pipeline Flow</h2>
            <p style={{ margin: '0.35rem 0 0', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.45, maxWidth: 820 }}>
              Left to right is execution order. Captions line up with <code style={{ fontSize: '0.75rem' }}>main.py</code> phase
              prints. This page is a reusable template — the Lab attaches a video per run. Use the Lab recipe&apos;s{' '}
              <strong style={{ color: '#cbd5e1' }}>Export</strong> step for phase preview MP4s, final view variants, and optional
              mesh sidecar flags.
            </p>
          </div>
          
          <div className="flowchart-board" style={{ position: 'relative', overflowX: 'auto', padding: '1rem 0.5rem', display: 'flex', alignItems: 'center', gap: '1rem', minHeight: 'auto', background: 'transparent' }}>
            <ComplexityVisualizer fieldsState={fieldsState} />
            {flowchartRows.map((row, rowIdx) => {
              const isRowActive = flowchartRowIsActive(row, activePhaseIndex)
              const isRowDone = flowchartRowIsDone(row, activePhaseIndex)

              const connector = rowIdx < flowchartRows.length - 1 && (
                <svg width="40" height="24" viewBox="0 0 40 24" style={{ overflow: 'visible', flexShrink: 0 }}>
                  <path
                    d="M 0 12 C 15 12, 25 12, 34 12"
                    className={`flow-path ${isRowDone ? 'done' : ''} ${isRowActive ? 'active' : ''}`}
                  />
                  <polygon
                    points="32,7 40,12 32,17"
                    fill={isRowDone || isRowActive ? 'var(--halo-cyan)' : 'var(--glass-border)'}
                    style={{ transition: 'fill 0.5s' }}
                  />
                </svg>
              )

              if (row.kind === 'single') {
                const stage = stages[row.index]
                const idx = row.index
                const isActive = idx === activePhaseIndex
                const isDone = idx < activePhaseIndex
                return (
                  <div key={stage.id} style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                    <div
                      onClick={() => setActivePhaseIndex(idx)}
                      className={`node-box ${isActive ? 'selected' : ''} ${isDone ? 'status-done' : ''}`}
                      style={{
                        padding: '1rem 1.2rem',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '0.35rem',
                        minWidth: '180px',
                      }}
                    >
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
                        {mainPyPhaseCaption(stage, idx)}
                      </div>
                      <div style={{ fontWeight: 600, color: isActive ? '#fff' : 'var(--text-main)', fontSize: '1rem' }}>
                        {stage.label}
                      </div>
                    </div>
                    {connector}
                  </div>
                )
              }

              const track = stages[row.trackingIndex]
              const overlap = stages[row.overlapIndex]
              const lineActive = (i: number) => i === activePhaseIndex
              return (
                <div key="merged-tracking-overlap" style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                  <div
                    className={`node-box ${isRowActive ? 'selected' : ''} ${isRowDone ? 'status-done' : ''}`}
                    style={{
                      padding: '1rem 1.2rem',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'stretch',
                      gap: '0.35rem',
                      minWidth: '220px',
                    }}
                  >
                    <div
                      style={{
                        fontSize: '0.72rem',
                        color: 'var(--text-muted)',
                        letterSpacing: '0.04em',
                        textAlign: 'center',
                      }}
                    >
                      {mainPyPhaseCaption(track, row.trackingIndex)}
                    </div>
                    <button
                      type="button"
                      onClick={() => setActivePhaseIndex(row.trackingIndex)}
                      style={{
                        fontWeight: lineActive(row.trackingIndex) ? 600 : 500,
                        color: lineActive(row.trackingIndex) ? '#fff' : 'var(--text-main)',
                        fontSize: '1rem',
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '0.15rem 0',
                        textAlign: 'center',
                        fontFamily: 'inherit',
                      }}
                    >
                      {track.label}
                    </button>
                    <div
                      style={{
                        height: 1,
                        margin: '0.1rem 0',
                        background: 'rgba(148, 163, 184, 0.35)',
                        alignSelf: 'stretch',
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => setActivePhaseIndex(row.overlapIndex)}
                      style={{
                        fontWeight: lineActive(row.overlapIndex) ? 600 : 500,
                        color: lineActive(row.overlapIndex) ? '#fff' : 'var(--text-main)',
                        fontSize: '1rem',
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '0.15rem 0',
                        textAlign: 'center',
                        fontFamily: 'inherit',
                      }}
                    >
                      {overlap.label}
                    </button>
                  </div>
                  {connector}
                </div>
              )
            })}
          </div>
        </div>

        {/* Bottom Active Params */}
        <div className="glass-panel" style={{ padding: '1.5rem 1.75rem 2rem', flex: 1 }}>
          <h2 style={{ marginTop: 0, marginBottom: '1.5rem', color: '#fff', fontSize: '1.35rem', fontWeight: 600 }}>{activeStage?.label} — Parameters</h2>
          <div 
            key={activeStage?.id}
            style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1.25rem' }}
          >
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
                Overlap sharpening only runs with the <strong style={{ color: '#e2e8f0' }}>built-in tracker</strong>.{' '}
                With the <strong style={{ color: '#e2e8f0' }}>alternate tracker</strong> selected, these options are hidden
                because they would not apply. Switch back to tune overlap behavior.
              </div>
            ) : (
              <>
                {activeFields.map((f, idx) =>
                  f.type === 'info' ? (
                    <div key={f.id} className="animate-slide-up" style={{ gridColumn: '1 / -1', animationDelay: `${idx * 0.04}s` }}>
                      <ConfigInfoFold title={`ℹ ${f.label}`} body={f.description ?? ''} />
                    </div>
                  ) : (
                    <div key={f.id} className="animate-slide-up" style={{ animationDelay: `${idx * 0.04}s` }}>
                      <ConfigFieldWrap field={f} value={fieldsState[f.id]}>
                        <InlineFieldInput
                          f={f}
                          value={fieldsState[f.id]}
                          onChange={(v) => {
                            setFieldsState((prev) => ({ ...prev, [f.id]: v }))
                            setActivePreset(null)
                          }}
                          allFields={fieldsState}
                        />
                      </ConfigFieldWrap>
                    </div>
                  ),
                )}
                {activeFields.length === 0 && (
                  <div style={{ color: 'var(--text-muted)' }}>Nothing to change on this step.</div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
