import { useEffect, useMemo, useState, type ReactNode } from 'react'
import type { DraftRun } from '../context/LabContext'
import type { Schema, SchemaField } from '../types'
import { API } from '../types'
import {
  applyConfigPreset,
  applyLabHiddenSchemaDefaults,
  CONFIG_PRESET_BLURBS,
  CONFIG_PRESET_LABELS,
  CONFIG_PRESET_ORDER,
  defaultsFromSchema,
  type ConfigPresetId,
} from '../configPresets'
import { HYBRID_SAM_PHASE_ID, hideHybridSamPhase } from '../lib/labFieldVisibility'
import { effectiveFieldTier, isSchemaFieldVisible, tier1FieldSortKey } from '../lib/labFieldMeta'
import { mainPyPhaseCaption } from '../lib/schemaStageCaption'
import {
  buildPipelineFlowchartRows,
  flowchartRowIsActive,
  flowchartRowIsDone,
} from '../lib/flowchartRows'
import { InlineFieldInput } from './InlineFieldInput'
import { ConfigFieldWrap, ConfigInfoFold } from './ConfigFieldWrap'
import { X, Zap, Scale, Trophy } from 'lucide-react'

const PRESET_ICONS: Record<ConfigPresetId, typeof Zap> = {
  fast_preview: Zap,
  standard: Scale,
  maximum_accuracy: Trophy,
}

function fieldShell(f: SchemaField, inner: ReactNode) {
  return <ConfigFieldWrap field={f}>{inner}</ConfigFieldWrap>
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
    setFieldsState(applyLabHiddenSchemaDefaults(schema.fields, { ...base, ...draft.fields }))
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
    const skip = new Set(['temporal_pose_refine'])
    return schema.fields
      .filter((f) => visible(f) && effectiveFieldTier(f) === 1 && !skip.has(f.id))
      .sort((a, b) => tier1FieldSortKey(a.id) - tier1FieldSortKey(b.id))
  }, [schema.fields, visible])

  const flowchartRows = useMemo(() => buildPipelineFlowchartRows(schema.stages), [schema.stages])

  const applyPreset = (id: ConfigPresetId) => {
    setFieldsState(applyConfigPreset(schema, id))
  }

  if (!open || !draft) return null

  const stages = schema.stages
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
      .filter((f) => f.id !== 'temporal_pose_refine')
      .filter((f) => f.display !== 'pruning_weight')

  const advancedFieldsForPhase = (phaseId: string) =>
    schema.fields.filter(
      (f) => f.phase === phaseId && f.type !== 'info' && visible(f) && effectiveFieldTier(f) === 3,
    )

  const temporalRefineField = schema.fields.find((f) => f.id === 'temporal_pose_refine')
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
      allFields={fieldsState}
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
            Big choices up top; open <strong style={{ color: '#e2e8f0' }}>Tune</strong> on each step for sliders.{' '}
            File paths and expert overrides sit under <strong style={{ color: '#e2e8f0' }}>Advanced</strong>.{' '}
            <strong style={{ color: '#e2e8f0' }}>Export</strong> controls phase preview clips, final montage variants, and optional{' '}
            HMR mesh sidecar JSON — enable those if you want them in Watch after the run.
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
              Main choices
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.85rem' }}>
              {tier1Fields.map((f) => fieldShell(f, renderFieldInput(f)))}
              {temporalRefineField && visible(temporalRefineField) && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <ConfigFieldWrap field={temporalRefineField}>
                    <InlineFieldInput
                      f={temporalRefineField}
                      value={fieldsState.temporal_pose_refine}
                      onChange={(v) => setFieldsState((p) => ({ ...p, temporal_pose_refine: v }))}
                      allFields={fieldsState}
                    />
                  </ConfigFieldWrap>
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
                    Large / huge skeleton models handle wild angles and speed better—closer to what pro sports research
                    uses—but need more time and GPU memory.
                  </div>
                )}
              {poseModelField &&
                visible(poseModelField) &&
                fieldsState.pose_model === 'RTMPose-L' && (
                  <div
                    style={{
                      gridColumn: '1 / -1',
                      fontSize: '0.78rem',
                      color: '#67e8f9',
                      padding: '0.5rem 0.75rem',
                      borderRadius: 10,
                      border: '1px solid rgba(34, 211, 238, 0.35)',
                      background: 'rgba(6, 78, 95, 0.2)',
                      lineHeight: 1.45,
                    }}
                  >
                    RTMPose-L runs through MMPose (not bundled). Install mmengine, mmcv, and mmpose locally, then compare
                    speed vs ViTPose-Base on the same clip. See docs/PIPELINE_IMPROVEMENTS_ROADMAP.md.
                  </div>
                )}
              {poseModelField &&
                visible(poseModelField) &&
                fieldsState.pose_model === 'Sapiens (ViTPose-Base fallback)' && (
                  <div
                    style={{
                      gridColumn: '1 / -1',
                      fontSize: '0.78rem',
                      color: '#e9d5ff',
                      padding: '0.5rem 0.75rem',
                      borderRadius: 10,
                      border: '1px solid rgba(192, 132, 252, 0.4)',
                      background: 'rgba(46, 16, 101, 0.25)',
                      lineHeight: 1.45,
                    }}
                  >
                    Set <strong style={{ color: '#f5f3ff' }}>SWAY_SAPIENS_TORCHSCRIPT</strong> on the API host to a COCO-17
                    Sapiens lite <code style={{ color: '#e9d5ff' }}>.pt2</code> (e.g. from Hugging Face{' '}
                    <code style={{ color: '#e9d5ff' }}>noahcao/sapiens-pose-coco</code>) for native Meta Sapiens inference.
                    If unset or the file is missing, the pipeline uses <strong style={{ color: '#f5f3ff' }}>ViTPose-Base</strong>{' '}
                    with the same COCO-17 output shape.
                  </div>
                )}
            </div>
          </div>

          <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {mainPyPhaseCaption(activeStage, activePhaseIndex)}
          </div>

          <div className="flowchart-board" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem' }}>
            {flowchartRows.map((row, rowIdx) => {
              const isRowActive = flowchartRowIsActive(row, activePhaseIndex)
              const isRowDone = flowchartRowIsDone(row, activePhaseIndex)

              const bar =
                rowIdx < flowchartRows.length - 1 ? (
                  <div
                    style={{
                      width: 20,
                      height: 2,
                      background: isRowDone ? 'var(--halo-cyan)' : 'var(--glass-border)',
                    }}
                  />
                ) : null

              if (row.kind === 'single') {
                const stage = stages[row.index]
                const idx = row.index
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
                    {bar}
                  </div>
                )
              }

              const track = stages[row.trackingIndex]
              const overlap = stages[row.overlapIndex]
              const linePick = (i: number) => i === activePhaseIndex
              return (
                <div key="merged-tracking-overlap" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div
                    className={`node-box ${isRowActive ? 'selected' : ''} ${isRowDone ? 'status-done' : ''}`}
                    style={{
                      border: isRowActive ? '2px solid var(--halo-cyan)' : '1px solid var(--glass-border)',
                      background: isRowActive ? 'rgba(14, 165, 233, 0.15)' : 'rgba(15, 20, 30, 0.7)',
                      padding: '0.75rem 1rem',
                      borderRadius: 12,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'stretch',
                      gap: '0.35rem',
                      minWidth: '200px',
                    }}
                  >
                    <div
                      style={{
                        fontSize: '0.68rem',
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
                        fontWeight: linePick(row.trackingIndex) ? 600 : 500,
                        fontSize: '0.9rem',
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '0.1rem 0',
                        color: 'inherit',
                        fontFamily: 'inherit',
                        textAlign: 'center',
                      }}
                    >
                      {track.label}
                    </button>
                    <div style={{ height: 1, background: 'rgba(148, 163, 184, 0.35)', margin: '0.05rem 0' }} />
                    <button
                      type="button"
                      onClick={() => setActivePhaseIndex(row.overlapIndex)}
                      style={{
                        fontWeight: linePick(row.overlapIndex) ? 600 : 500,
                        fontSize: '0.9rem',
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '0.1rem 0',
                        color: 'inherit',
                        fontFamily: 'inherit',
                        textAlign: 'center',
                      }}
                    >
                      {overlap.label}
                    </button>
                  </div>
                  {bar}
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
                Overlap sharpening only runs with the <strong style={{ color: '#e2e8f0' }}>built-in tracker</strong>.{' '}
                Switch back from the alternate engine to change these options.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <details open style={{ borderRadius: 12, border: '1px solid var(--glass-border)', padding: '0.65rem 0.85rem', background: 'rgba(0,0,0,0.15)' }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 600, color: '#e2e8f0', listStyle: 'none' }}>
                    ⚙ Sliders · {activeStage?.short ?? 'phase'}
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
                          How strongly each “looks fake” rule counts
                        </summary>
                        <div
                          style={{
                            marginTop: '0.75rem',
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
                            gap: '0.65rem',
                          }}
                        >
                          {pruningWeightFields.map((f) => fieldShell(f, renderFieldInput(f)))}
                        </div>
                      </details>
                    )}
                    {tuneFieldsForPhase(activeStage?.id ?? '').length === 0 &&
                      !(activeStage?.id === 'post_pose_prune' && pruningWeightFields.length > 0) && (
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                          Nothing to slide on this step—defaults are fine unless you open Expert below.
                        </div>
                      )}
                  </div>
                </details>

                {advancedFieldsForPhase(activeStage?.id ?? '').length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#94a3b8', marginBottom: '0.5rem' }}>
                      Expert / paths · {activeStage?.short}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.85rem' }}>
                      {advancedFieldsForPhase(activeStage?.id ?? '').map((f) => fieldShell(f, renderFieldInput(f)))}
                    </div>
                  </div>
                )}

                {schema.fields
                  .filter((f) => f.phase === activeStage?.id && f.type === 'info')
                  .map((f) => (
                    <ConfigInfoFold key={f.id} title={`ℹ ${f.label}`} body={f.description ?? ''} />
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
