import { useEffect, useMemo, useState, type ReactNode } from 'react'
import type { DraftRun } from '../context/LabContext'
import type { Schema, SchemaField } from '../types'
import { API } from '../types'
import {
  applyLabHiddenSchemaDefaults,
  applyPhaseGroupPresets,
  defaultsFromSchema,
  migrateLegacyPipelineFields,
  phase13StrategyFromFields,
  DEFAULT_PRESET_IDS,
  type PhaseGroupId,
} from '../configPresets'
import { hideHybridSamPhase, isHybridSamOverlapField } from '../lib/labFieldVisibility'
import {
  hideFieldForPhase13Strategy,
  PHASE13_MODE_FIELD_ID,
} from '../lib/labPhase13Ui'
import { effectiveFieldTier, isSchemaFieldVisible, tier1FieldSortKey } from '../lib/labFieldMeta'
import { mainPyPhaseCaption } from '../lib/schemaStageCaption'
import { fieldBelongsToConfigStage } from '../lib/configPageUi'
import { buildPipelineFlowchartRows, flowchartRowIsDone } from '../lib/flowchartRows'
import { InlineFieldInput } from './InlineFieldInput'
import { ConfigFieldWrap } from './ConfigFieldWrap'
import { PresetGroupSelector } from './PresetGroupSelector'
import { X } from 'lucide-react'

function fieldShell(f: SchemaField, inner: ReactNode) {
  return <ConfigFieldWrap field={f}>{inner}</ConfigFieldWrap>
}

export function RunEditorModal({
  open,
  schema,
  draft,
  schemaPhaseAllowlist,
  onClose,
  onSave,
}: {
  open: boolean
  schema: Schema
  draft: DraftRun | null
  schemaPhaseAllowlist?: ReadonlySet<string> | null
  onClose: () => void
  onSave: (recipeName: string, fields: Record<string, unknown>) => void
}) {
  const [recipeName, setRecipeName] = useState('')
  const [fieldsState, setFieldsState] = useState<Record<string, unknown>>({})
  const [selectedPresets, setSelectedPresets] = useState<Record<PhaseGroupId, string>>({
    ...DEFAULT_PRESET_IDS,
  })
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
    const merged = applyLabHiddenSchemaDefaults(
      schema.fields,
      migrateLegacyPipelineFields({ ...base, ...draft.fields }),
    )
    setFieldsState(merged)
    setActivePhaseIndex(0)
    setSelectedPresets({ ...DEFAULT_PRESET_IDS })
  }, [open, draft?.clientId, draft?.recipeName, draft?.fields, schema.fields])

  const hybridHidden = hideHybridSamPhase(fieldsState.tracker_technology, fieldsState.sway_phase13_mode)

  const activePhase13 = useMemo(() => phase13StrategyFromFields(fieldsState), [fieldsState])

  const visible = useMemo(
    () => (f: SchemaField) => {
      if (schemaPhaseAllowlist && !schemaPhaseAllowlist.has(f.phase)) return false
      if (f.type === 'info') return false
      if (hybridHidden && isHybridSamOverlapField(f.id)) return false
      if (hideFieldForPhase13Strategy(f.id, activePhase13)) return false
      return isSchemaFieldVisible(f, fieldsState)
    },
    [fieldsState, hybridHidden, activePhase13, schemaPhaseAllowlist],
  )

  const tier1Fields = useMemo(() => {
    const skip = new Set(['temporal_pose_refine', PHASE13_MODE_FIELD_ID])
    return schema.fields
      .filter((f) => visible(f) && effectiveFieldTier(f) === 1 && !skip.has(f.id))
      .sort((a, b) => tier1FieldSortKey(a.id) - tier1FieldSortKey(b.id))
  }, [schema.fields, visible])

  const flowchartRows = useMemo(() => buildPipelineFlowchartRows(schema.stages), [schema.stages])

  const handlePresetSelect = (groupId: PhaseGroupId, presetId: string) => {
    const next = { ...selectedPresets, [groupId]: presetId }
    setSelectedPresets(next)
    setFieldsState(
      applyPhaseGroupPresets(schema, next.phases_1_3, next.phases_4_6, next.phases_7_9, next.phases_10_11),
    )
  }

  if (!open || !draft) return null

  const hideRecipeBaselineStrip =
    schemaPhaseAllowlist != null &&
    schemaPhaseAllowlist.size === 1 &&
    schemaPhaseAllowlist.has('detection')

  const stages = schema.stages
  const activeStage = stages[activePhaseIndex]

  const tuneFieldsForStage = (stageId: string) =>
    schema.fields
      .filter(
        (f) =>
          fieldBelongsToConfigStage(f, stageId) &&
          f.type !== 'info' &&
          visible(f) &&
          effectiveFieldTier(f) === 2,
      )
      .filter((f) => f.id !== 'temporal_pose_refine')
      .filter((f) => f.display !== 'pruning_weight')

  const advancedFieldsForStage = (stageId: string) =>
    schema.fields.filter(
      (f) =>
        fieldBelongsToConfigStage(f, stageId) && f.type !== 'info' && visible(f) && effectiveFieldTier(f) === 3,
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
      onChange={(v) => {
        setFieldsState((prev) => ({ ...prev, [f.id]: v }))
      }}
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
          width: 'min(1240px, 100%)',
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
            Pick presets per phase group, then open <strong style={{ color: '#e2e8f0' }}>Tune</strong> on each step for sliders.{' '}
            File paths and expert overrides sit under <strong style={{ color: '#e2e8f0' }}>Advanced</strong>.
            {schemaPhaseAllowlist && (
              <span>
                {' '}
                <strong style={{ color: '#7dd3fc' }}>Segment editor:</strong> only parameters for this checkpoint stage are
                shown; other phases keep schema defaults until a later tree node changes them.
              </span>
            )}
          </div>

          <div style={hideRecipeBaselineStrip ? { display: 'none' } : undefined}>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              Pipeline presets
            </div>
            <p style={{ margin: '0 0 0.65rem', fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
              Pick one preset per phase group. They combine independently — any combination is valid. Manual field edits
              clear preset highlights.
            </p>
            <PresetGroupSelector
              selectedPresets={selectedPresets}
              onSelect={handlePresetSelect}
              compact
            />
          </div>

          <div>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
              Main choices
            </div>
            <p style={{ margin: '0 0 0.55rem', fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
              {hideRecipeBaselineStrip ? (
                <>
                  Detection-only segment: preset cards are hidden. Tracking / pose knobs apply on later tree nodes after this
                  run writes the Phase-1 checkpoint.
                </>
              ) : (
                <>
                  These controls are the most impactful across strategies. Preset selections above set these values;
                  you can override them here.
                </>
              )}
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '0.85rem' }}>
              {tier1Fields.map((f) => fieldShell(f, renderFieldInput(f)))}
              {temporalRefineField && visible(temporalRefineField) && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <ConfigFieldWrap field={temporalRefineField}>
                    <InlineFieldInput
                      f={temporalRefineField}
                      value={fieldsState.temporal_pose_refine}
                      onChange={(v) => {
                        setFieldsState((p) => ({ ...p, temporal_pose_refine: v }))
                      }}
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
                    Large / huge skeleton models handle wild angles and speed better — closer to what pro sports research
                    uses — but need more time and GPU memory.
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
                    speed vs ViTPose-Base on the same clip.
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
                    Sapiens lite <code style={{ color: '#e9d5ff' }}>.pt2</code> for native Meta Sapiens inference.
                    If unset, the pipeline uses <strong style={{ color: '#f5f3ff' }}>ViTPose-Base</strong>.
                  </div>
                )}
            </div>
          </div>

          <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {mainPyPhaseCaption(activeStage, activePhaseIndex)}
          </div>

          <div className="flowchart-board" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem' }}>
            {flowchartRows.map((row, rowIdx) => {
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

              const stage = stages[row.index]
              const idx = row.index
              const isActive = idx === activePhaseIndex
              const isDone = isRowDone
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
                      maxWidth: 'min(320px, 90vw)',
                      textAlign: 'center',
                    }}
                  >
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
                      {mainPyPhaseCaption(stage, idx)}
                    </div>
                    <div style={{ fontWeight: 600, fontSize: '0.9rem', lineHeight: 1.35, marginTop: '0.25rem' }}>{stage.label}</div>
                  </button>
                  {bar}
                </div>
              )
            })}
          </div>

          <div>
            <h3 style={{ margin: '0 0 0.75rem', color: '#fff', fontSize: '1.05rem' }}>{activeStage?.label}</h3>
            {activeStage?.id === 'handshake' && hybridHidden && (
              <div
                style={{
                  marginBottom: '0.75rem',
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
                Switch back from ByteTrack to see those options; tracking sliders below still apply.
              </div>
            )}
            {activeStage?.id === 'handshake' && activePhase13 === 'sway_handshake' && !hybridHidden && (
              <div
                style={{
                  marginBottom: '0.75rem',
                  padding: '1rem 1.15rem',
                  borderRadius: 12,
                  border: '1px solid rgba(45, 212, 191, 0.35)',
                  background: 'rgba(6, 95, 70, 0.2)',
                  color: 'var(--text-muted)',
                  fontSize: '0.88rem',
                  lineHeight: 1.55,
                }}
              >
                <strong style={{ color: '#e2e8f0' }}>Sway handshake</strong> fixes hybrid SAM IoU at <strong style={{ color: '#e2e8f0' }}>0.10</strong> and turns{' '}
                <strong style={{ color: '#e2e8f0' }}>weak cues off</strong> at enqueue — overlap sliders are hidden.
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <details open style={{ borderRadius: 12, border: '1px solid var(--glass-border)', padding: '0.65rem 0.85rem', background: 'rgba(0,0,0,0.15)' }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 600, color: '#e2e8f0', listStyle: 'none' }}>
                    Sliders · {activeStage?.short ?? 'phase'}
                  </summary>
                  <div style={{ marginTop: '0.85rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '0.85rem' }}>
                    {tuneFieldsForStage(activeStage?.id ?? '').map((f) => fieldShell(f, renderFieldInput(f)))}
                    {activeStage?.id === 'cleanup_export' && pruningWeightFields.length > 0 && (
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
                          How strongly each "looks fake" rule counts
                        </summary>
                        <div
                          style={{
                            marginTop: '0.75rem',
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 220px), 1fr))',
                            gap: '0.65rem',
                          }}
                        >
                          {pruningWeightFields.map((f) => fieldShell(f, renderFieldInput(f)))}
                        </div>
                      </details>
                    )}
                    {tuneFieldsForStage(activeStage?.id ?? '').length === 0 &&
                      !(activeStage?.id === 'cleanup_export' && pruningWeightFields.length > 0) && (
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                          Nothing to slide on this step — defaults are fine unless you open Expert below.
                        </div>
                      )}
                  </div>
                </details>

                {advancedFieldsForStage(activeStage?.id ?? '').length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#94a3b8', marginBottom: '0.5rem' }}>
                      Expert / paths · {activeStage?.short}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '0.85rem' }}>
                      {advancedFieldsForStage(activeStage?.id ?? '').map((f) => fieldShell(f, renderFieldInput(f)))}
                    </div>
                  </div>
                )}

            </div>
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
