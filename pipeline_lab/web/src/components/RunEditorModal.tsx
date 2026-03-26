import { useEffect, useMemo, useState, type ReactNode } from 'react'
import type { DraftRun } from '../context/LabContext'
import type { Schema, SchemaField } from '../types'
import { API } from '../types'
import {
  applyLabHiddenSchemaDefaults,
  applyLabRecipe,
  defaultsFromSchema,
  migrateLegacyPipelineFields,
  phase13StrategyFromFields,
  PHASE13_STRATEGY_BLURBS,
  PHASE13_STRATEGY_LABELS,
  PHASE13_STRATEGY_ORDER,
  QUALITY_TIER_BLURBS,
  QUALITY_TIER_LABELS,
  QUALITY_TIER_ORDER,
  type Phase13StrategyId,
  type QualityTierId,
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
import { Phase13EffectivePanel } from './Phase13EffectivePanel'
import { Handshake, Layers, Users, X, Zap, Scale, Trophy } from 'lucide-react'

const QUALITY_ICONS: Record<QualityTierId, typeof Zap> = {
  fast_preview: Zap,
  standard: Scale,
  maximum_accuracy: Trophy,
}

const PHASE13_ICONS: Record<Phase13StrategyId, typeof Layers> = {
  standard: Layers,
  dancer_registry: Users,
  sway_handshake: Handshake,
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
  const [qualityTier, setQualityTier] = useState<QualityTierId>('standard')
  const [recipeCustom, setRecipeCustom] = useState(true)
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
    setQualityTier('standard')
    setRecipeCustom(true)
  }, [open, draft?.clientId, draft?.recipeName, draft?.fields, schema.fields])

  const hybridHidden = hideHybridSamPhase(fieldsState.tracker_technology, fieldsState.sway_phase13_mode)

  const activePhase13 = useMemo(() => phase13StrategyFromFields(fieldsState), [fieldsState])

  const visible = useMemo(
    () => (f: SchemaField) => {
      if (f.type === 'info') return false
      if (hybridHidden && isHybridSamOverlapField(f.id)) return false
      if (hideFieldForPhase13Strategy(f.id, activePhase13)) return false
      return isSchemaFieldVisible(f, fieldsState)
    },
    [fieldsState, hybridHidden, activePhase13],
  )

  const tier1Fields = useMemo(() => {
    const skip = new Set(['temporal_pose_refine', PHASE13_MODE_FIELD_ID])
    return schema.fields
      .filter((f) => visible(f) && effectiveFieldTier(f) === 1 && !skip.has(f.id))
      .sort((a, b) => tier1FieldSortKey(a.id) - tier1FieldSortKey(b.id))
  }, [schema.fields, visible])

  const flowchartRows = useMemo(() => buildPipelineFlowchartRows(schema.stages), [schema.stages])

  const applyQualityTier = (id: QualityTierId) => {
    setFieldsState(applyLabRecipe(schema, id, phase13StrategyFromFields(fieldsState)))
    setQualityTier(id)
    setRecipeCustom(false)
  }

  const applyPhase13 = (id: Phase13StrategyId) => {
    setFieldsState(applyLabRecipe(schema, qualityTier, id))
    setRecipeCustom(false)
  }

  if (!open || !draft) return null

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
        setRecipeCustom(true)
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
            Big choices up top; open <strong style={{ color: '#e2e8f0' }}>Tune</strong> on each step for sliders.{' '}
            File paths and expert overrides sit under <strong style={{ color: '#e2e8f0' }}>Advanced</strong>.
          </div>

          <div>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              Recipe baseline
            </div>
            <p style={{ margin: '0 0 0.65rem', fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
              Pick speed/quality and Phases 1–3 strategy separately; they combine. The panel below lists the exact early-pipeline
              behavior for your strategy. Editing a field clears speed/quality highlights only.
            </p>
            <div style={{ fontSize: '0.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              Speed &amp; quality
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 200px), 1fr))', gap: '0.65rem', marginBottom: '0.85rem' }}>
              {QUALITY_TIER_ORDER.map((pid) => {
                const Icon = QUALITY_ICONS[pid]
                const selected = !recipeCustom && qualityTier === pid
                return (
                  <button
                    key={pid}
                    type="button"
                    className="btn"
                    title={QUALITY_TIER_BLURBS[pid]}
                    onClick={() => applyQualityTier(pid)}
                    style={{
                      textAlign: 'left',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      gap: '0.35rem',
                      padding: '0.85rem 1rem',
                      borderRadius: 12,
                      border: selected ? '1px solid rgba(255,255,255,0.55)' : '1px solid var(--glass-border)',
                      background: selected ? 'rgba(255,255,255,0.1)' : 'rgba(15, 23, 42, 0.65)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <Icon size={18} style={{ color: 'var(--halo-cyan)' }} />
                      <span style={{ fontWeight: 700, color: '#f8fafc' }}>{QUALITY_TIER_LABELS[pid]}</span>
                    </div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>{QUALITY_TIER_BLURBS[pid]}</span>
                  </button>
                )
              })}
            </div>
            <div style={{ fontSize: '0.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              Phases 1–3 strategy
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 200px), 1fr))', gap: '0.65rem' }}>
              {PHASE13_STRATEGY_ORDER.map((pid) => {
                const Icon = PHASE13_ICONS[pid]
                const selected = activePhase13 === pid
                return (
                  <button
                    key={pid}
                    type="button"
                    className="btn"
                    title={PHASE13_STRATEGY_BLURBS[pid]}
                    onClick={() => applyPhase13(pid)}
                    style={{
                      textAlign: 'left',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      gap: '0.35rem',
                      padding: '0.85rem 1rem',
                      borderRadius: 12,
                      border: selected ? '1px solid rgba(255,255,255,0.55)' : '1px solid var(--glass-border)',
                      background: selected ? 'rgba(255,255,255,0.1)' : 'rgba(15, 23, 42, 0.65)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <Icon size={18} style={{ color: 'var(--halo-cyan)' }} />
                      <span style={{ fontWeight: 700, color: '#f8fafc' }}>{PHASE13_STRATEGY_LABELS[pid]}</span>
                    </div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                      {PHASE13_STRATEGY_BLURBS[pid]}
                    </span>
                  </button>
                )
              })}
            </div>
            <div style={{ marginTop: '0.85rem' }}>
              <Phase13EffectivePanel strategy={activePhase13} fieldsState={fieldsState} />
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
              Main choices
            </div>
            <p style={{ margin: '0 0 0.55rem', fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
              Phases 1–3 strategy is set only by the cards above (not repeated here). Remaining knobs are shared across strategies
              except hybrid overlap sliders, which are hidden for Sway handshake because the server fixes them.
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
                        setRecipeCustom(true)
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
                Switch back from the alternate engine to change those options; tracking sliders below still apply.
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
                <strong style={{ color: '#e2e8f0' }}>weak cues off</strong> at enqueue — overlap sliders are hidden so the UI matches
                the subprocess. Open-floor registry + SAM verify still run; see <strong style={{ color: '#e2e8f0' }}>Effective Phases 1–3</strong> above.
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <details open style={{ borderRadius: 12, border: '1px solid var(--glass-border)', padding: '0.65rem 0.85rem', background: 'rgba(0,0,0,0.15)' }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 600, color: '#e2e8f0', listStyle: 'none' }}>
                    ⚙ Sliders · {activeStage?.short ?? 'phase'}
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
                          How strongly each “looks fake” rule counts
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
                          Nothing to slide on this step—defaults are fine unless you open Expert below.
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
