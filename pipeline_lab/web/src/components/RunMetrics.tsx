export { PipelineImpactReport, FriendlyRunConfig, PipelineImpactSummary, RunOverviewStrip } from './PipelineImpactReport'

export function TrackQualitySummary({ summary }: { summary: Record<string, unknown> }) {
  const count = typeof summary.track_count === 'number' ? summary.track_count : null
  const med = typeof summary.median_track_observations === 'number' ? summary.median_track_observations : null
  const mean = typeof summary.mean_track_observations === 'number' ? summary.mean_track_observations : null
  const jumps = typeof summary.internal_timeline_jumps === 'number' ? summary.internal_timeline_jumps : null
  const note = typeof summary.note === 'string' ? summary.note : null
  const cards: { label: string; value: string; hint: string }[] = [
    {
      label: 'Track count',
      value: count != null ? String(count) : '—',
      hint: 'Unique IDs after post-track stitch (before pruning).',
    },
    {
      label: 'Median track length',
      value: med != null ? `${med.toFixed(1)} obs` : '—',
      hint: 'Frames with a detection per track (median).',
    },
    {
      label: 'Mean track length',
      value: mean != null ? `${mean.toFixed(1)} obs` : '—',
      hint: 'Average observations per track.',
    },
    {
      label: 'Timeline jumps (heuristic)',
      value: jumps != null ? String(jumps) : '—',
      hint: 'Large gaps inside a track — rough proxy for fragmentation (not MOT IDSW).',
    },
    {
      label: 'IDF1 / HOTA',
      value: 'N/A',
      hint: 'Not computed in the Lab UI. Need MOT ground-truth labels; run TrackEval offline (tools.benchmark_trackeval, run_trackeval_boxmot_ablation).',
    },
    {
      label: 'IDSW',
      value: 'N/A',
      hint: 'Not computed without GT. “Timeline jumps” above is only a no-GT heuristic.',
    },
  ]
  return (
    <div style={{ marginTop: '0.5rem' }}>
      <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#94a3b8', marginBottom: '0.45rem' }}>
        Track stats (no MOT / GT labels)
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '0.5rem' }}>
        {cards.map((c) => (
          <div
            key={c.label}
            style={{
              padding: '0.55rem 0.65rem',
              borderRadius: 10,
              background: 'rgba(15, 23, 42, 0.65)',
              border: '1px solid rgba(148, 163, 184, 0.25)',
            }}
            title={c.hint}
          >
            <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              {c.label}
            </div>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc', marginTop: '0.2rem' }}>{c.value}</div>
          </div>
        ))}
      </div>
      {note && (
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem', lineHeight: 1.45 }}>{note}</div>
      )}
    </div>
  )
}
