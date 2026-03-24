import { useEffect, useMemo, useState, type ReactNode } from 'react'
import type { DraftRun } from '../context/LabContext'
import type { Schema, SchemaField } from '../types'
import { API } from '../types'
import {
  applyConfigPreset,
  CONFIG_PRESET_BLURBS,
  CONFIG_PRESET_LABELS,
  CONFIG_PRESET_ORDER,
  defaultsFromSchema,
  type ConfigPresetId,
} from '../configPresets'
import { HYBRID_SAM_PHASE_ID, hideHybridSamPhase } from '../lib/labFieldVisibility'
import { effectiveFieldTier, isSchemaFieldVisible, tier1FieldSortKey } from '../lib/labFieldMeta'
import { mainPyPhaseCaption } from '../lib/schemaStageCaption'
import { InlineFieldInput } from './InlineFieldInput'
import { X, Zap, Scale, Trophy } from 'lucide-react'

const PRESET_ICONS: Record<ConfigPresetId, typeof Zap> = {
  fast_preview: Zap,
  standard: Scale,
  maximum_accuracy: Trophy,
}

function fieldShell(f: SchemaField, inner: ReactNode, opts?: { noDescription?: boolean }) {
  return (
    <div
      key={f.id}
      style={{
        background: 'rgba(0,0,0,0.2)',
        padding: '1rem',
        borderRadius: 12,
        border: '1px solid var(--glass-border)',
      }}
    >
      <div style={{ color: '#fff', fontWeight: 500, marginBottom: '0.35rem' }}>{f.label}</div>
      {!opts?.noDescription && f.type !== 'info' && f.description && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.75rem', lineHeight: 1.45 }}>
          {f.description}
        </div>
      )}
      {inner}
    </div>
  )
}

export function RunEditorModal({
  open,
  schema,
  draft,
  onClose,
  onSave,
}: {
  open: boolean
  schema: Schema
  draft: DraftRun | null
  onClose: () => void
  onSave: (recipeName: string, fields: Record<string, unknown>) => void
}) {
  const [recipeName, setRecipeName] = useState('')
  const [fieldsState, setFieldsState] = useState<Record<string, unknown>>({})
  const [activePhaseIndex, setActivePhaseIndex] = useState(0)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [modelsStatus, setModelsStatus] = useState<Record<string, boolean> | null>(null)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    fetch(`${API}/api/models/status`)
      .then((r) => (r.ok ? r.json() : {}))
      .then((j) => {
        if (!cancelled && j && typeof j === 'object') setModelsStatus(j as Record<string, boolean>)
      })
      .catch(() => {
        if (!cancelled) setModelsStatus(null)
      })
    return () => {
      cancelled = true
    }
  }, [open])

  useEffect(() => {
    if (!open || !draft) return
    setRecipeName(draft.recipeName)
    const base = defaultsFromSchema(schema.fields)
    setFieldsState({ ...base, ...draft.fields })
    setActivePhaseIndex(0)
  }, [open, draft?.clientId, draft?.recipeName, draft?.fields, schema.fields])

  const hybridHidden = hideHybridSamPhase(fieldsState.tracker_technology)

  const visible = useMemo(
    () => (f: SchemaField) => {
      if (f.type === 'info') return false
      if (hybridHidden && f.phase === HYBRID_SAM_PHASE_ID) return false
      return isSchemaFieldVisible(f, fieldsState)
    },
    [fieldsState, hybridHidden],
  )

  const tier1Fields = useMemo(() => {
    const skip = new Set(['temporal_pose_radius', 'temporal_pose_refine'])
    return schema.fields
      .filter((f) => visible(f) && effectiveFieldTier(f) === 1 && !skip.has(f.id))
      .sort((a, b) => tier1FieldSortKey(a.id) - tier1FieldSortKey(b.id))
  }, [schema.fields, visible])

  const applyPreset = (id: ConfigPresetId) => {
    setFieldsState(applyConfigPreset(schema, id))
  }

  if (!open || !draft) return null

  const stages = schema.stages.filter((s) => s.id !== 'export')
  const activeStage = stages[activePhaseIndex]

  const tuneFieldsForPhase = (phaseId: string) =>
    schema.fields
      .filter(
        (f) =>
          f.phase === phaseId &&
          f.type !== 'info' &&
          visible(f) &&
          effectiveFieldTier(f) === 2,
      )
      .filter((f) => !['temporal_pose_refine', 'temporal_pose_radius'].includes(f.id))
      .filter((f) => f.display !== 'pruning_weight')

  const advancedFieldsForPhase = (phaseId: string) =>
    schema.fields.filter(
      (f) => f.phase === phaseId && f.type !== 'info' && visible(f) && effectiveFieldTier(f) === 3,
    )

  const temporalRefineField = schema.fields.find((f) => f.id === 'temporal_pose_refine')
  const temporalRadiusField = schema.fields.find((f) => f.id === 'temporal_pose_radius')
  const poseModelField = schema.fields.find((f) => f.id === 'pose_model')

  const pruningWeightFields = schema.fields.filter(
    (f) => f.display === 'pruning_weight' && f.phase === 'post_pose_prune' && visible(f),
  )

  const renderFieldInput = (f: SchemaField) => (
    <InlineFieldInput
      f={f}
      value={fieldsState[f.id]}
      onChange={(v) => setFieldsState((prev) => ({ ...prev, [f.id]: v }))}
      modelsStatus={modelsStatus}
    />
  )

  return (
    <div
      role="dialog"
      aria-modal
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
          width: 'min(960px, 100%)',
          maxHeight: 'min(90vh, 900px)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div
          style={{
            padding: '1rem 1.25rem',
            borderBottom: '1px solid var(--glass-border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
          }}
        >
          <input
            value={recipeName}
            onChange={(e) => setRecipeName(e.target.value)}
            placeholder="Run name"
            style={{
              flex: 1,
              background: 'rgba(0,0,0,0.25)',
              border: '1px solid var(--glass-border)',
              borderRadius: 10,
              color: '#fff',
              fontSize: '1.1rem',
              fontWeight: 600,
              padding: '0.5rem 0.75rem',
              outline: 'none',
            }}
          />
          <button type="button" className="btn" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '1rem 1.25rem', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
            Primary controls stay visible; expand <strong style={{ color: '#e2e8f0' }}>Tune</strong> per phase for
            thresholds; enable <strong style={{ color: '#e2e8f0' }}>Advanced</strong> for paths and env-style overrides.
          </div>

          <div>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
              Presets
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.65rem' }}>
              {CONFIG_PRESET_ORDER.map((pid) => {
                const Icon = PRESET_ICONS[pid]
                return (
                  <button
                    key={pid}
                    type="button"
                    className="btn"
                    onClick={() => applyPreset(pid)}
                    style={{
                      textAlign: 'left',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      gap: '0.35rem',
                      padding: '0.85rem 1rem',
                      borderRadius: 12,
                      border: '1px solid var(--glass-border)',
                      background: 'rgba(15, 23, 42, 0.65)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <Icon size={18} style={{ color: 'var(--halo-cyan)' }} />
                      <span style={{ fontWeight: 700, color: '#f8fafc' }}>{CONFIG_PRESET_LABELS[pid]}</span>
                    </div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                      {CONFIG_PRESET_BLURBS[pid]}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
              Main settings
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.85rem' }}>
              {tier1Fields.map((f) => fieldShell(f, renderFieldInput(f)))}
              {temporalRefineField && temporalRadiusField && visible(temporalRefineField) && (
                <div
                  style={{
                    background: 'rgba(0,0,0,0.2)',
                    padding: '1rem',
                    borderRadius: 12,
                    border: '1px solid var(--glass-border)',
                    gridColumn: '1 / -1',
                  }}
                >
                  <div style={{ color: '#fff', fontWeight: 500, marginBottom: '0.35rem' }}>Temporal pose refine</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.75rem', lineHeight: 1.45 }}>
                    {temporalRefineField.description}
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '1rem' }}>
                    <label className="checkbox-label" style={{ gap: '0.5rem' }}>
                      <input
                        type="checkbox"
                        checked={Boolean(fieldsState.temporal_pose_refine ?? temporalRefineField.default)}
                        onChange={(e) => setFieldsState((p) => ({ ...p, temporal_pose_refine: e.target.checked }))}
                        style={{ display: 'none' }}
                      />
                      <div className="checkbox-visual" style={{ width: 14, height: 14 }} />
                      <span style={{ fontSize: '0.88rem', color: '#e2e8f0' }}>Enable refine</span>
                    </label>
                    {Boolean(fieldsState.temporal_pose_refine ?? temporalRefineField.default) && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Radius</span>
                        {renderFieldInput(temporalRadiusField)}
                      </div>
                    )}
                  </div>
                </div>
              )}
              {poseModelField &&
                visible(poseModelField) &&
                (fieldsState.pose_model === 'ViTPose-Large' || fieldsState.pose_model === 'ViTPose-Huge') && (
                  <div
                    style={{
                      gridColumn: '1 / -1',
                      fontSize: '0.78rem',
                      color: '#a5b4fc',
                      padding: '0.5rem 0.75rem',
                      borderRadius: 10,
                      border: '1px solid rgba(129, 140, 248, 0.35)',
                      background: 'rgba(79, 70, 229, 0.12)',
                      lineHeight: 1.45,
                    }}
                  >
                    AthletePose3D-style fine-tunes on athletic movement improve extreme poses — large/huge ViTPose
                    checkpoints are the closest analogue in this build.
                  </div>
                )}
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Phase flow · {mainPyPhaseCaption(activeStage, activePhaseIndex)}
            </span>
            <label className="checkbox-label" style={{ background: 'rgba(0,0,0,0.3)', padding: '0.4rem 0.8rem', borderRadius: 8 }}>
              <input
                type="checkbox"
                checked={showAdvanced}
                onChange={(e) => setShowAdvanced(e.target.checked)}
                style={{ display: 'none' }}
              />
              <div className="checkbox-visual" />
              Advanced params
            </label>
          </div>

          <div className="flowchart-board" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem' }}>
            {stages.map((stage, idx) => {
              const isActive = idx === activePhaseIndex
              const isDone = idx < activePhaseIndex
              return (
                <div key={stage.id} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <button
                    type="button"
                    onClick={() => setActivePhaseIndex(idx)}
                    className={`node-box ${isActive ? 'selected' : ''} ${isDone ? 'status-done' : ''}`}
                    style={{
                      border: isActive ? '2px solid var(--halo-cyan)' : '1px solid var(--glass-border)',
                      background: isActive ? 'rgba(14, 165, 233, 0.15)' : 'rgba(15, 20, 30, 0.7)',
                      padding: '0.75rem 1rem',
                      cursor: 'pointer',
                      borderRadius: 12,
                      color: 'inherit',
                      font: 'inherit',
                    }}
                  >
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
                      {mainPyPhaseCaption(stage, idx)}
                    </div>
                    <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{stage.label}</div>
                  </button>
                  {idx < stages.length - 1 && (
                    <div
                      style={{
                        width: 20,
                        height: 2,
                        background: isDone ? 'var(--halo-cyan)' : 'var(--glass-border)',
                      }}
                    />
                  )}
                </div>
              )
            })}
          </div>

          <div>
            <h3 style={{ margin: '0 0 0.75rem', color: '#fff', fontSize: '1.05rem' }}>{activeStage?.label}</h3>
            {activeStage?.id === HYBRID_SAM_PHASE_ID && hybridHidden ? (
              <div
                style={{
                  padding: '1rem 1.15rem',
                  borderRadius: 12,
                  border: '1px solid rgba(148, 163, 184, 0.35)',
                  background: 'rgba(30, 41, 59, 0.45)',
                  color: 'var(--text-muted)',
                  fontSize: '0.88rem',
                  lineHeight: 1.55,
                }}
              >
                <strong style={{ color: '#e2e8f0' }}>Hybrid SAM</strong> runs only on the{' '}
                <strong style={{ color: '#e2e8f0' }}>BoxMOT</strong> path. Switch tracker to BoxMOT to tune overlap
                refinement.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <details open style={{ borderRadius: 12, border: '1px solid var(--glass-border)', padding: '0.65rem 0.85rem', background: 'rgba(0,0,0,0.15)' }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 600, color: '#e2e8f0', listStyle: 'none' }}>
                    ⚙ Tune · {activeStage?.short ?? 'phase'}
                  </summary>
                  <div style={{ marginTop: '0.85rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.85rem' }}>
                    {tuneFieldsForPhase(activeStage?.id ?? '').map((f) => fieldShell(f, renderFieldInput(f)))}
                    {activeStage?.id === 'post_pose_prune' && pruningWeightFields.length > 0 && (
                      <details
                        style={{
                          gridColumn: '1 / -1',
                          borderRadius: 10,
                          border: '1px dashed rgba(148, 163, 184, 0.4)',
                          padding: '0.65rem 0.75rem',
                          background: 'rgba(0,0,0,0.2)',
                        }}
                      >
                        <summary style={{ cursor: 'pointer', fontWeight: 600, color: '#cbd5e1', fontSize: '0.88rem' }}>
                          Tier B rule weights (PRUNING_WEIGHTS)
                        </summary>
                        <div
                          style={{
                            marginTop: '0.75rem',
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
                            gap: '0.65rem',
                          }}
                        >
                          {pruningWeightFields.map((f) => fieldShell(f, renderFieldInput(f), { noDescription: true }))}
                        </div>
                      </details>
                    )}
                    {tuneFieldsForPhase(activeStage?.id ?? '').length === 0 &&
                      !(activeStage?.id === 'post_pose_prune' && pruningWeightFields.length > 0) && (
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No tune sliders in this phase.</div>
                      )}
                  </div>
                </details>

                {showAdvanced && advancedFieldsForPhase(activeStage?.id ?? '').length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#94a3b8', marginBottom: '0.5rem' }}>
                      Advanced · {activeStage?.short}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.85rem' }}>
                      {advancedFieldsForPhase(activeStage?.id ?? '').map((f) => fieldShell(f, renderFieldInput(f)))}
                    </div>
                  </div>
                )}

                {schema.fields
                  .filter((f) => f.phase === activeStage?.id && f.type === 'info')
                  .map((f) => (
                    <div
                      key={f.id}
                      style={{
                        padding: '0.85rem 1rem',
                        borderRadius: 12,
                        border: '1px solid var(--glass-border)',
                        background: 'rgba(0,0,0,0.18)',
                        fontSize: '0.78rem',
                        color: 'var(--text-muted)',
                        lineHeight: 1.55,
                      }}
                    >
                      <div style={{ fontWeight: 600, color: '#cbd5e1', marginBottom: '0.35rem' }}>{f.label}</div>
                      {f.description}
                    </div>
                  ))}
              </div>
            )}
          </div>
        </div>

        <div
          style={{
            padding: '1rem 1.25rem',
            borderTop: '1px solid var(--glass-border)',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: '0.75rem',
          }}
        >
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn primary"
            onClick={() => onSave(recipeName.trim() || 'Untitled run', { ...fieldsState })}
          >
            Save run
          </button>
        </div>
      </div>
    </div>
  )
}
