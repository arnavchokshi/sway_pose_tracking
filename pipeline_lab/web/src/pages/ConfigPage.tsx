import { useEffect, useState, useMemo } from 'react'
import { PIPELINE_LAB_LOCAL } from '../siteUrls'
import { API } from '../types'
import type { Schema, SchemaField } from '../types'
import {
  applyLabRecipe,
  defaultsFromSchema,
  phase13StrategyFromFields,
  PHASE13_STRATEGY_BLURBS,
  PHASE13_STRATEGY_LABELS,
  PHASE13_STRATEGY_ORDER,
  QUALITY_TIER_BLURBS,
  QUALITY_TIER_LABELS,
  QUALITY_TIER_ORDER,
  type QualityTierId,
} from '../configPresets'
import { InlineFieldInput } from '../components/InlineFieldInput'
import { ConfigFieldWrap } from '../components/ConfigFieldWrap'
import { Phase13EffectivePanel } from '../components/Phase13EffectivePanel'
import { hideFieldForPhase13Strategy, PHASE13_MODE_FIELD_ID } from '../lib/labPhase13Ui'
import { hideHybridSamPhase, isHybridSamOverlapField } from '../lib/labFieldVisibility'
import { isSchemaFieldVisible } from '../lib/labFieldMeta'
import {
  CONFIG_PAGE_LEDE,
  CONFIG_STAGE_INTRO,
  fieldBelongsToConfigStage,
  groupFieldsForStage,
} from '../lib/configPageUi'
import { masterGuidePhaseLabel } from '../lib/schemaStageCaption'
import { ComplexityVisualizer } from '../components/ComplexityVisualizer'

export function ConfigPage() {
  const [schema, setSchema] = useState<Schema | null>(null)
  const [schemaError, setSchemaError] = useState<string | null>(null)
  const [activePhaseIndex, setActivePhaseIndex] = useState(0)
  const [configName, setConfigName] = useState('My Config')
  const [fieldsState, setFieldsState] = useState<Record<string, unknown>>({})
  const [qualityTier, setQualityTier] = useState<QualityTierId>('standard')
  /** When true, preset buttons are not highlighted (user diverged from last preset apply). */
  const [recipeCustom, setRecipeCustom] = useState(false)

  useEffect(() => {
    let cancelled = false
    setSchemaError(null)
    fetch(`${API}/api/schema`)
      .then(async (r) => {
        const text = await r.text()
        if (!r.ok) {
          if (r.status === 502 || r.status === 503) {
            throw new Error(
              `Pipeline API is not reachable (bad gateway). The dev UI proxies /api to ${PIPELINE_LAB_LOCAL.apiOrigin} — start uvicorn there in another terminal.`,
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

  const activePhase13 = useMemo(() => phase13StrategyFromFields(fieldsState), [fieldsState])

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
uvicorn pipeline_lab.server.app:app --reload --host localhost --port 8765`}
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
  const activeFieldsRaw = schema.fields.filter((f) => fieldBelongsToConfigStage(f, activeStage.id))
  const hideOverlapBecauseTracker = hideHybridSamPhase(
    fieldsState.tracker_technology,
    fieldsState.sway_phase13_mode,
  )
  const activeFields = activeFieldsRaw.filter((f) => {
    if (f.type === 'info') return false
    if (f.id === PHASE13_MODE_FIELD_ID) return false
    if (hideOverlapBecauseTracker && activeStage?.id === 'handshake' && isHybridSamOverlapField(f.id)) {
      return false
    }
    if (hideFieldForPhase13Strategy(f.id, activePhase13)) return false
    return isSchemaFieldVisible(f, fieldsState)
  })
  const showOverlapHiddenBanner = activeStage?.id === 'handshake' && hideOverlapBecauseTracker

  const controlFields = activeFields.filter((f) => f.type !== 'info')
  const intro = activeStage ? CONFIG_STAGE_INTRO[activeStage.id] : undefined
  const { sections, orphans } = groupFieldsForStage(controlFields, activeStage?.id ?? '')

  const renderField = (f: SchemaField, idx: number) => (
    <div key={f.id} className="animate-slide-up" style={{ animationDelay: `${idx * 0.03}s` }}>
      <ConfigFieldWrap field={f} value={fieldsState[f.id]}>
        <InlineFieldInput
          f={f}
          value={fieldsState[f.id]}
          onChange={(v) => {
            setFieldsState((prev) => ({ ...prev, [f.id]: v }))
            setRecipeCustom(true)
          }}
          allFields={fieldsState}
        />
      </ConfigFieldWrap>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', height: '100%' }}>
      <div className="glass-panel config-page-settings-panel">
        <header className="config-page-settings-header">
          <div className="config-page-settings-header__row">
            <h1 className="config-page-settings-title">Pipeline settings</h1>
            <div className="config-page-identity">
              <label className="config-page-preset-label" htmlFor="config-name-input">
                Config name
              </label>
              <input
                id="config-name-input"
                value={configName}
                onChange={(e) => setConfigName(e.target.value)}
                placeholder="e.g. Finals — tight stage"
                className="config-page-name-input"
              />
            </div>
          </div>
          <p className="config-page-settings-lede">{CONFIG_PAGE_LEDE}</p>
        </header>

        <div className="config-page-preset-card">
          <div className="config-page-preset-label">Recipe baseline</div>
          <p className="config-page-preset-blurb">
            Two independent choices: <strong style={{ color: '#cbd5e1' }}>speed &amp; quality</strong> (weights, pose, scoring)
            vs <strong style={{ color: '#cbd5e1' }}>Phases 1–3 strategy</strong> (early detection → track → stitch). Combine
            them — e.g. Maximum accuracy + Sway handshake. Manual edits hide the highlights until you press a preset again.
            Phases 1–3 strategy is only changed here (the handshake step no longer duplicates that enum).
          </p>
          <div className="config-page-preset-rows">
            <div>
              <div
                style={{
                  fontSize: '0.68rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  color: 'var(--text-muted)',
                  marginBottom: '0.3rem',
                }}
              >
                Speed &amp; quality
              </div>
              <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center', flexWrap: 'wrap' }}>
                {QUALITY_TIER_ORDER.map((id) => (
                  <button
                    key={id}
                    type="button"
                    className="btn"
                    title={QUALITY_TIER_BLURBS[id]}
                    style={{
                      whiteSpace: 'nowrap',
                      borderColor: !recipeCustom && qualityTier === id ? '#fff' : undefined,
                      background: !recipeCustom && qualityTier === id ? 'rgba(255,255,255,0.15)' : undefined,
                    }}
                    onClick={() => {
                      setFieldsState(applyLabRecipe(schema, id, phase13StrategyFromFields(fieldsState)))
                      setQualityTier(id)
                      setRecipeCustom(false)
                    }}
                  >
                    {QUALITY_TIER_LABELS[id]}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div
                style={{
                  fontSize: '0.68rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  color: 'var(--text-muted)',
                  marginBottom: '0.3rem',
                }}
              >
                Phases 1–3 strategy
              </div>
              <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center', flexWrap: 'wrap' }}>
                {PHASE13_STRATEGY_ORDER.map((id) => (
                  <button
                    key={id}
                    type="button"
                    className="btn"
                    title={PHASE13_STRATEGY_BLURBS[id]}
                    style={{
                      whiteSpace: 'nowrap',
                      borderColor: activePhase13 === id ? '#fff' : undefined,
                      background: activePhase13 === id ? 'rgba(255,255,255,0.15)' : undefined,
                    }}
                    onClick={() => {
                      setFieldsState(applyLabRecipe(schema, qualityTier, id))
                      setRecipeCustom(false)
                    }}
                  >
                    {PHASE13_STRATEGY_LABELS[id]}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div style={{ marginTop: '0.85rem' }}>
            <Phase13EffectivePanel strategy={activePhase13} fieldsState={fieldsState} />
          </div>
        </div>
      </div>

      <div
        className="glass-panel config-page-steps-panel"
        style={{ padding: '0.85rem 1rem' }}
        aria-label="Pipeline steps; workload estimate shown as background motion"
      >
        <div className="config-page-workload-bg" aria-hidden>
          <ComplexityVisualizer fieldsState={fieldsState} />
        </div>
        <div className="config-page-steps-inner">
          <div className="config-page-steps-heading">
            <h2>Steps</h2>
            <span className="config-page-steps-pill" title="main.py prints [1/11] through [11/11]; the UI groups knobs into four intents">
              4 setup steps · 11 engine phases
            </span>
          </div>
          <p className="config-page-steps-lede">
            Pick a step to see only the controls for that part of the pipeline. The CLI still prints{' '}
            <strong>eleven phases</strong> — each card shows which printed phase numbers your knobs target. Pre-pose prune (
            <strong>phase 4</strong>) is master-locked in code (no tab here). 3D lift runs between smoothing and scoring when
            enabled but is not a separate printed phase.
          </p>

          <div className="config-page-flow" role="tablist" aria-label="Pipeline steps">
            {stages.map((stage, idx) => {
              const isActive = idx === activePhaseIndex
              const isDone = idx < activePhaseIndex
              const copy = CONFIG_STAGE_INTRO[stage.id]
              return (
                <button
                  key={stage.id}
                  type="button"
                  role="tab"
                  aria-selected={isActive}
                  className={`config-page-flow-node ${isActive ? 'config-page-flow-node--active' : ''} ${isDone ? 'config-page-flow-node--done' : ''}`}
                  onClick={() => setActivePhaseIndex(idx)}
                >
                  <div className="config-page-flow-node-top">
                    <span className="config-page-flow-badge" aria-hidden>
                      {idx + 1}
                    </span>
                    <div className="config-page-flow-step">
                      {stage.short}
                    </div>
                  </div>
                  <div className="config-page-flow-name">{copy?.headline ?? stage.label}</div>
                  {stage.main_phases ? (
                    <div className="config-page-flow-mainpy">{masterGuidePhaseLabel(stage.main_phases)}</div>
                  ) : null}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '1rem 1.25rem 1.25rem', flex: 1 }}>
        <h2 className="config-page-panel-title">
          {intro?.headline ?? activeStage?.label}
          {activeStage ? (
            <span style={{ fontWeight: 500, color: '#64748b', fontSize: '0.85rem', marginLeft: '0.5rem' }}>
              ({activeStage.short}
              {activeStage.main_phases ? (
                <>
                  {' '}
                  ·{' '}
                  <span style={{ color: 'var(--halo-cyan)' }}>{masterGuidePhaseLabel(activeStage.main_phases)}</span>
                </>
              ) : null}
              )
            </span>
          ) : null}
        </h2>
        <p className="config-page-panel-lede">{intro?.lede ?? 'Adjust parameters for this part of the run.'}</p>

        <div key={activeStage?.id}>
          {showOverlapHiddenBanner && (
            <div
              style={{
                marginBottom: '1.25rem',
                padding: '1rem 1.15rem',
                borderRadius: 12,
                border: '1px solid rgba(148, 163, 184, 0.35)',
                background: 'rgba(30, 41, 59, 0.45)',
                color: 'var(--text-muted)',
                fontSize: '0.88rem',
                lineHeight: 1.55,
              }}
            >
              Overlap refinement only applies on the default tracker path. These controls are hidden when an alternate engine
              is selected.
            </div>
          )}
          {activeStage?.id === 'handshake' && activePhase13 === 'sway_handshake' && !hideOverlapBecauseTracker && (
            <div
              style={{
                marginBottom: '1.25rem',
                padding: '1rem 1.15rem',
                borderRadius: 12,
                border: '1px solid rgba(45, 212, 191, 0.35)',
                background: 'rgba(6, 95, 70, 0.2)',
                color: 'var(--text-muted)',
                fontSize: '0.88rem',
                lineHeight: 1.55,
              }}
            >
              <strong style={{ color: '#e2e8f0' }}>Sway handshake</strong> forces hybrid SAM IoU to{' '}
              <strong style={{ color: '#e2e8f0' }}>0.10</strong> and weak cues off at enqueue — overlap sliders are hidden.
              See <strong style={{ color: '#e2e8f0' }}>Effective Phases 1–3 configuration</strong> above.
            </div>
          )}

          {sections.map((sec, sIdx) => (
            <div key={sec.def.title} className="config-page-section">
              <h3 className="config-page-section-title">{sec.def.title}</h3>
              <p className="config-page-section-blurb">{sec.def.blurb}</p>
              <div className="config-page-section-grid">
                {sec.fields.map((f, i) => renderField(f, sIdx * 10 + i))}
              </div>
            </div>
          ))}

          {orphans.length > 0 && (
            <div className="config-page-section">
              <h3 className="config-page-section-title">Other</h3>
              <div className="config-page-section-grid">{orphans.map((f, i) => renderField(f, 80 + i))}</div>
            </div>
          )}

          {controlFields.length === 0 && (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Nothing to change on this step.</div>
          )}
        </div>
      </div>
    </div>
  )
}
