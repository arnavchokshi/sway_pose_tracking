import type { PhaseGroupId, PresetDef } from '../configPresets'
import { PHASE_GROUPS, PRESETS_BY_GROUP } from '../configPresets'

export function PresetGroupSelector({
  selectedPresets,
  onSelect,
  compact,
}: {
  selectedPresets: Record<PhaseGroupId, string>
  onSelect: (groupId: PhaseGroupId, presetId: string) => void
  compact?: boolean
}) {
  return (
    <div
      className="preset-group-selector"
      style={{ display: 'flex', flexDirection: 'column', gap: compact ? '0.75rem' : '1.1rem' }}
    >
      {PHASE_GROUPS.map((group) => {
        const presets = PRESETS_BY_GROUP[group.id]
        const selectedId = selectedPresets[group.id]
        const selectedPreset = presets.find((p) => p.id === selectedId)
        return (
          <PresetGroupRow
            key={group.id}
            groupId={group.id}
            label={group.label}
            phasesLabel={group.phasesLabel}
            description={group.description}
            presets={presets}
            selectedId={selectedId}
            selectedPreset={selectedPreset}
            onSelect={(presetId) => onSelect(group.id, presetId)}
            compact={compact}
          />
        )
      })}
    </div>
  )
}

function PresetGroupRow({
  groupId: _groupId,
  label,
  phasesLabel,
  description,
  presets,
  selectedId,
  selectedPreset,
  onSelect,
  compact,
}: {
  groupId: PhaseGroupId
  label: string
  phasesLabel: string
  description: string
  presets: PresetDef[]
  selectedId: string
  selectedPreset: PresetDef | undefined
  onSelect: (presetId: string) => void
  compact?: boolean
}) {
  return (
    <div
      style={{
        borderRadius: 14,
        border: '1px solid var(--glass-border)',
        background: 'rgba(15, 23, 42, 0.45)',
        padding: compact ? '0.65rem 0.85rem' : '0.85rem 1rem',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginBottom: '0.25rem' }}>
        <span
          style={{
            fontSize: '0.62rem',
            fontWeight: 800,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: '#7dd3fc',
            padding: '0.15rem 0.4rem',
            borderRadius: 5,
            background: 'rgba(14, 165, 233, 0.15)',
            border: '1px solid rgba(14, 165, 233, 0.25)',
            whiteSpace: 'nowrap',
          }}
        >
          {phasesLabel}
        </span>
        <span style={{ fontWeight: 700, fontSize: compact ? '0.82rem' : '0.88rem', color: '#e2e8f0' }}>{label}</span>
      </div>
      {!compact && (
        <p style={{ margin: '0 0 0.5rem', fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
          {description}
        </p>
      )}

      <div
        style={{
          display: 'flex',
          gap: '0.4rem',
          flexWrap: 'wrap',
          marginBottom: selectedPreset ? '0.5rem' : 0,
        }}
      >
        {presets.map((preset) => {
          const isSelected = preset.id === selectedId
          return (
            <button
              key={preset.id}
              type="button"
              className="btn preset-group-preset-btn"
              title={preset.description}
              onClick={() => onSelect(preset.id)}
              style={{
                fontSize: compact ? '0.72rem' : '0.78rem',
                padding: compact ? '0.3rem 0.55rem' : '0.4rem 0.7rem',
                borderRadius: 8,
                border: isSelected ? '1px solid rgba(255,255,255,0.55)' : '1px solid var(--glass-border)',
                background: isSelected ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.2)',
                color: isSelected ? '#f8fafc' : '#94a3b8',
                fontWeight: isSelected ? 700 : 500,
              }}
            >
              {preset.name}
            </button>
          )
        })}
      </div>

      {selectedPreset && (
        <div
          style={{
            fontSize: '0.72rem',
            color: 'var(--text-muted)',
            lineHeight: 1.45,
            padding: '0.45rem 0.6rem',
            borderRadius: 8,
            background: 'rgba(0,0,0,0.15)',
            border: '1px solid rgba(148, 163, 184, 0.15)',
          }}
        >
          <span style={{ fontWeight: 700, color: '#cbd5e1' }}>{selectedPreset.name}:</span>{' '}
          {selectedPreset.description}
        </div>
      )}
    </div>
  )
}
