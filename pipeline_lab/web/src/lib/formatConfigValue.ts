/** Human-readable values for Lab config / manifest display. */
export function formatConfigValue(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return v ? 'On' : 'Off'
  if (typeof v === 'number') {
    return Number.isFinite(v) && !Number.isInteger(v) ? String(roundSmart(v)) : String(v)
  }
  if (typeof v === 'string') return v.trim() === '' ? '—' : v
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function roundSmart(n: number): number {
  const a = Math.abs(n)
  if (a >= 100) return Math.round(n)
  if (a >= 10) return Math.round(n * 10) / 10
  return Math.round(n * 1000) / 1000
}
