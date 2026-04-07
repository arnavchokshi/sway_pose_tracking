import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowLeft,
  ArrowUpRight,
  BarChart3,
  ChevronRight,
  Film,
  GitBranch,
  GitCompare,
  Layers,
  Lightbulb,
  Loader2,
  RefreshCw,
  ScatterChart,
  Timer,
  Sparkles,
  Target,
  TrendingUp,
  VideoOff,
  X,
  Zap,
} from 'lucide-react'
import { PIPELINE_LAB_LOCAL } from '../siteUrls'
import { API } from '../types'

// ─── Types ────────────────────────────────────────────────────────────────────

type OptunaSweepMeta = {
  config?: string
  sequence_order?: string[]
  log_jsonl?: string
  storage?: string
  git_sha?: string
  hint?: string
  v4?: boolean
  v4_mode?: string
  has_stages?: boolean
  stage?: string
  stage_expected_counts?: Record<string, number>
  planned_trials_total?: number
  planned_video_runs_total?: number
  stage_planned_trials_total?: number
  stage_planned_video_runs_total?: number
  videos_per_trial?: number
  objective_components?: Record<string, number>
}

type OptunaTrialRow = {
  number: number
  state: string
  value: number | null
  /** Wall time for completed trials (from Optuna), seconds */
  duration_s?: number | null
  params: Record<string, unknown>
  user_attrs: Record<string, unknown>
}

type OptunaBest = {
  number: number
  value: number
  params: Record<string, unknown>
  user_attrs: Record<string, unknown>
}

export type OptunaSweepStatus = {
  schema?: string
  empty_state?: boolean
  updated_unix: number
  study_name: string
  direction: string
  n_trials_total: number
  n_complete: number
  n_pruned: number
  n_other: number
  best: OptunaBest | null
  trials: OptunaTrialRow[]
  meta?: OptunaSweepMeta
}

type MediaItem = { kind: string; filename: string; path: string }

type MediaResponse = {
  trial: number
  sequence: string
  items: MediaItem[]
  note?: string
}

type SweepMode = 'optuna' | 'v4'

// ─── URL helpers ─────────────────────────────────────────────────────────────

function sweepApiBase(mode: SweepMode) {
  return `${API}/api/${mode === 'v4' ? 'v4-sweep' : 'optuna-sweep'}`
}
function pullLambdaUrl(mode: SweepMode) {
  return `${sweepApiBase(mode)}/pull-lambda`
}
function statusUrl(mode: SweepMode) {
  return `${sweepApiBase(mode)}/status`
}
function mediaUrl(mode: SweepMode, trial: number, sequence: string) {
  return `${sweepApiBase(mode)}/trial/${trial}/sequence/${encodeURIComponent(sequence)}/media`
}
function fileUrl(mode: SweepMode, trial: number, sequence: string, relPath: string) {
  const q = new URLSearchParams({ path: relPath })
  const base =
    import.meta.env.DEV && typeof window !== 'undefined' ? PIPELINE_LAB_LOCAL.apiOrigin : API
  const ns = mode === 'v4' ? 'v4-sweep' : 'optuna-sweep'
  return `${base}/api/${ns}/trial/${trial}/sequence/${encodeURIComponent(sequence)}/file?${q}`
}

// ─── Formatting ───────────────────────────────────────────────────────────────

function formatTime(ts: number) {
  try {
    return new Date(ts * 1000).toLocaleString()
  } catch {
    return '—'
  }
}

function formatValue(v: number | null | undefined) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return v.toFixed(4)
}

/** Column id for the All trials table: built-ins or a video name (per-clip score). */
type TrialTableSortCol = 'trial' | 'state' | 'score' | string

function compareTrialTableRows(
  a: OptunaTrialRow,
  b: OptunaTrialRow,
  col: TrialTableSortCol,
  dir: 'asc' | 'desc',
): number {
  const mult = dir === 'asc' ? 1 : -1
  if (col === 'state') {
    const c = a.state.localeCompare(b.state)
    if (c !== 0) return mult * c
  } else if (col === 'trial') {
    if (a.number !== b.number) return mult * (a.number - b.number)
  } else if (col === 'score') {
    const na = typeof a.value === 'number' ? a.value : null
    const nb = typeof b.value === 'number' ? b.value : null
    if (na === null && nb === null) {
      /* tie-break below */
    } else if (na === null) return 1
    else if (nb === null) return -1
    else if (na !== nb) return mult * (na - nb)
  } else {
    const ra = a.user_attrs[`score_${col}`]
    const rb = b.user_attrs[`score_${col}`]
    const va = typeof ra === 'number' ? ra : null
    const vb = typeof rb === 'number' ? rb : null
    if (va === null && vb === null) {
      /* tie-break below */
    } else if (va === null) return 1
    else if (vb === null) return -1
    else if (va !== vb) return mult * (va - vb)
  }
  return b.number - a.number
}

function SortableTrialTh({
  col,
  label,
  active,
  dir,
  onSort,
  style,
}: {
  col: TrialTableSortCol
  label: ReactNode
  active: boolean
  dir: 'asc' | 'desc'
  onSort: (c: TrialTableSortCol) => void
  style?: CSSProperties
}) {
  return (
    <th
      scope="col"
      aria-sort={active ? (dir === 'asc' ? 'ascending' : 'descending') : 'none'}
      style={style}
      className={active ? 'optuna-table-th--sortable optuna-table-th--sorted' : 'optuna-table-th--sortable'}
    >
      <button
        type="button"
        className="optuna-table-sort-btn"
        onClick={() => onSort(col)}
      >
        <span>{label}</span>
        {active ? (
          <span className="optuna-table-sort-icon" aria-hidden>
            {dir === 'asc' ? '↑' : '↓'}
          </span>
        ) : null}
      </button>
    </th>
  )
}

function valuesEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true
  if (typeof a === 'number' && typeof b === 'number' && Number.isFinite(a) && Number.isFinite(b)) {
    return Math.abs(a - b) < 1e-9
  }
  if (a == null && b == null) return true
  if (a == null || b == null) return false
  try {
    return JSON.stringify(a) === JSON.stringify(b)
  } catch {
    return false
  }
}

function diffParamKeys(a: Record<string, unknown>, b: Record<string, unknown>): Set<string> {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)])
  const diff = new Set<string>()
  for (const k of keys) {
    if (!valuesEqual(a[k], b[k])) diff.add(k)
  }
  return diff
}

function compactParamValue(k: string, v: unknown): string {
  const raw = String(v ?? '')
  if ((k.includes('weight') || k.includes('path')) && raw.includes('/')) {
    return raw.split('/').pop() || raw
  }
  if (raw.length > 42) return `${raw.slice(0, 42)}…`
  return raw
}

function trialConfigDiffSummary(
  trial: OptunaTrialRow,
  best: OptunaBest | null,
): { count: number; rows: string[] } {
  if (!best || trial.number === best.number) return { count: 0, rows: ['baseline (best)'] }
  const keys = [...diffParamKeys(best.params, trial.params)].sort()
  const rows: string[] = []
  for (const k of keys) {
    const from = compactParamValue(k, best.params[k])
    const to = compactParamValue(k, trial.params[k])
    rows.push(`${k}: ${from} -> ${to}`)
  }
  return { count: rows.length, rows: rows.slice(0, 4) }
}

function trialProgressSummary(t: OptunaTrialRow): { label: string; detail: string; tone: 'running' | 'complete' | 'failed' | 'queued' } {
  const ua = t.user_attrs ?? {}
  const curVideo = String(ua.progress_current_video ?? '').trim()
  const curPhase = String(ua.progress_current_phase ?? '').trim()
  const idxRaw = ua.progress_video_index
  const totalRaw = ua.progress_video_total
  const idx = typeof idxRaw === 'number' ? idxRaw : Number(idxRaw)
  const total = typeof totalRaw === 'number' ? totalRaw : Number(totalRaw)
  const videoPart = Number.isFinite(idx) && Number.isFinite(total) && total > 0 ? `video ${Math.max(0, idx)}/${Math.max(1, total)}` : 'video —'
  if (t.state === 'COMPLETE') return { label: 'done', detail: `${videoPart} · phase: final`, tone: 'complete' }
  if (t.state === 'FAIL' || t.state === 'ERROR') return { label: 'failed', detail: `${videoPart} · video: ${curVideo || 'n/a'} · phase: ${curPhase || 'n/a'}`, tone: 'failed' }
  if (t.state === 'RUNNING') return { label: 'running', detail: `${videoPart} · video: ${curVideo || 'pending'} · phase: ${curPhase || 'starting'}`, tone: 'running' }
  return { label: t.state.toLowerCase(), detail: `${videoPart} · phase: queued`, tone: 'queued' }
}

function phaseLabel(raw: unknown): string {
  const p = String(raw ?? '').trim().toLowerCase()
  if (!p || p === 'starting') return 'phase0_enrollment'
  return p
}

function trialProgressNumbers(t: OptunaTrialRow): {
  videoIndex: number
  videoTotal: number
  phasePct: number | null
  overallPct: number | null
} {
  const ua = t.user_attrs ?? {}
  const phase = String(ua.progress_current_phase ?? '').toLowerCase()
  const state = String(t.state ?? '').toUpperCase()
  const idxRaw = ua.progress_video_index
  const totalRaw = ua.progress_video_total
  const pctRaw = ua.progress_current_phase_pct
  const idx = typeof idxRaw === 'number' ? idxRaw : Number(idxRaw)
  const total = typeof totalRaw === 'number' ? totalRaw : Number(totalRaw)
  const phasePctParsed = typeof pctRaw === 'number' ? pctRaw : Number(pctRaw)
  let phasePct = Number.isFinite(phasePctParsed) ? Math.max(0, Math.min(100, phasePctParsed)) : null
  // Avoid sticky 100% visuals while a trial is still RUNNING and not finished.
  if (state === 'RUNNING' && phasePct != null && phasePct >= 99.9 && !['done', 'failed', 'phase8_final'].includes(phase)) {
    phasePct = 99.0
  }
  if (!(Number.isFinite(idx) && Number.isFinite(total) && total > 0)) {
    return { videoIndex: 0, videoTotal: 0, phasePct, overallPct: null }
  }
  const i = Math.max(0, idx)
  const ttot = Math.max(1, total)
  const base = Math.max(0, Math.min(ttot, i - 1)) / ttot
  const frac = (phasePct ?? 0) / 100.0 / ttot
  const overall = Math.max(0, Math.min(100, (base + frac) * 100))
  return { videoIndex: i, videoTotal: ttot, phasePct, overallPct: overall }
}

function formatHttpDetail(res: Response, body: unknown): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const d = (body as { detail: unknown }).detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) return d.map((x) => JSON.stringify(x)).join('; ')
  }
  return res.statusText
}

// Score colors: 0 → red, 0.5 → yellow, 1 → green
function scoreColor(v: number): string {
  if (v >= 0.75) return '#4ade80'
  if (v >= 0.55) return '#86efac'
  if (v >= 0.4) return '#fde047'
  return '#f87171'
}

function isMaximizeStudy(direction: string): boolean {
  const d = direction.toLowerCase()
  return d.includes('max') && !d.includes('min')
}

/** Map raw metric to heat color depending on whether higher or lower is better */
function scoreColorDirected(v: number, direction: string): string {
  const x = isMaximizeStudy(direction) ? v : 1 - v
  return scoreColor(Math.min(1, Math.max(0, x)))
}

function formatDurationSec(s: number): string {
  if (s < 60) return `${s.toFixed(0)}s`
  const m = Math.floor(s / 60)
  const sec = Math.round(s % 60)
  return sec ? `${m}m ${sec}s` : `${m}m`
}

/** Short labels for Y-axis ticks (may be large spans). */
function formatDurationAxisTick(sec: number, spanSec: number): string {
  if (!Number.isFinite(sec)) return '—'
  if (spanSec < 120) return sec < 10 ? sec.toFixed(1) : `${Math.round(sec)}s`
  if (spanSec < 3600) return `${(sec / 60).toFixed(sec < 600 ? 1 : 0)}m`
  const h = sec / 3600
  return `${h < 10 ? h.toFixed(1) : Math.round(h)}h`
}

/** JSON sometimes stringifies floats; Optuna categoricals stay strings — only treat as numeric when coercible */
function coerceParamNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string') {
    const t = v.trim()
    if (t === '') return null
    const n = Number(t)
    if (Number.isFinite(n)) return n
  }
  return null
}

/** Charts only show successful completed trials (excludes pruned, failed, running, etc.) */
function isChartTrial(t: OptunaTrialRow): boolean {
  return t.state === 'COMPLETE' && typeof t.value === 'number'
}

function coercePositiveSeconds(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v) && v > 0) return v
  if (typeof v === 'string') {
    const n = parseFloat(v)
    if (Number.isFinite(n) && n > 0) return n
  }
  return null
}

/**
 * Wall-clock trial duration (seconds): `duration_s` from JSON (Optuna or coalesced server-side),
 * else `user_attrs.trial_duration_s`, else sum of `duration_s_*` per sequence from auto_sweep.
 */
function resolveTrialWallDurationSec(t: OptunaTrialRow): number | null {
  const fromTop = coercePositiveSeconds(t.duration_s as unknown)
  if (fromTop != null) return fromTop
  const ua = t.user_attrs
  if (!ua || typeof ua !== 'object') return null
  const rec = ua as Record<string, unknown>
  const trialDur = coercePositiveSeconds(rec.trial_duration_s)
  if (trialDur != null) return Math.round(trialDur * 10) / 10
  let sum = 0
  for (const [k, v] of Object.entries(rec)) {
    if (!k.startsWith('duration_s_')) continue
    const n = coercePositiveSeconds(v)
    if (n != null) sum += n
  }
  return sum > 0 ? Math.round(sum * 10) / 10 : null
}

function safeMean(values: number[]): number | null {
  if (!values.length) return null
  const s = values.reduce((a, b) => a + b, 0)
  return s / values.length
}

/** Y-axis (or X) range with padding — zooms into narrow data bands instead of pinning to 0 */
function chartAxisDomain(dataMin: number, dataMax: number): { lo: number; hi: number } {
  const padFrac = 0.06
  const minPad = 1e-6
  if (!Number.isFinite(dataMin) || !Number.isFinite(dataMax)) return { lo: 0, hi: 1 }
  const rawSpan = dataMax - dataMin
  if (rawSpan <= 0) {
    const d = Math.max(Math.abs(dataMax) * 0.02, minPad)
    return { lo: dataMax - d, hi: dataMax + d }
  }
  const pad = Math.max(rawSpan * padFrac, minPad)
  return { lo: dataMin - pad, hi: dataMax + pad }
}

const VIDEO_COLORS: Record<string, string> = {
  bigtest: '#38bdf8',    // sky blue
  gymtest: '#a78bfa',   // violet
  mirrortest: '#34d399', // emerald
  aditest: '#f97316',    // orange
  easytest: '#f472b6',   // pink (distinct from violet gymtest)
}

// ─── Chart: Score Progress Chart (bar per trial) ─────────────────────────────

function ScoreProgressChart({
  trials,
  bestNumber,
  onSelectTrial,
  selectedTrial,
  chartMode = 'bar',
  direction = 'minimize',
  showRollingAvg = false,
}: {
  trials: OptunaTrialRow[]
  bestNumber: number | null
  onSelectTrial: (n: number) => void
  selectedTrial: number | null
  chartMode?: 'bar' | 'scatter'
  direction?: string
  showRollingAvg?: boolean
}) {
  const completed = useMemo(
    () => trials.filter(isChartTrial).sort((a, b) => a.number - b.number),
    [trials],
  )
  const maxM = useMemo(() => isMaximizeStudy(direction), [direction])

  const W = 920
  const H = 248
  const PAD = { top: 22, right: 24, bottom: 40, left: 48 }
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom

  const vals = completed.map((t) => t.value as number)
  const dom =
    completed.length >= 1 ? chartAxisDomain(Math.min(...vals), Math.max(...vals)) : { lo: 0, hi: 1 }
  const span = Math.max(dom.hi - dom.lo, 1e-9)
  const yForVal = (v: number) =>
    PAD.top + innerH - ((v - dom.lo) / span) * innerH * 0.92 - innerH * 0.04

  const rollingWindow = Math.min(5, Math.max(2, Math.ceil(Math.max(completed.length, 1) / 6)))
  const rollingPts = useMemo(() => {
    if (!showRollingAvg || chartMode !== 'scatter' || completed.length < rollingWindow) return null
    const out: { cx: number; cy: number }[] = []
    for (let i = 0; i < completed.length; i++) {
      const a = Math.max(0, i - rollingWindow + 1)
      const slice = completed.slice(a, i + 1)
      const mean = slice.reduce((s, t) => s + (t.value as number), 0) / slice.length
      const cx = PAD.left + (i / (completed.length - 1 || 1)) * innerW
      const cy = PAD.top + innerH - ((mean - dom.lo) / span) * innerH * 0.92 - innerH * 0.04
      out.push({ cx, cy })
    }
    return out
  }, [
    showRollingAvg,
    chartMode,
    completed,
    rollingWindow,
    innerW,
    innerH,
    PAD.left,
    PAD.top,
    dom.lo,
    span,
  ])

  const [tooltip, setTooltip] = useState<{
    trial: number
    value: number
    x: number
    y: number
    duration?: number | null
  } | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  if (!completed.length) {
    return (
      <div className="optuna-chart-empty">
        <TrendingUp size={32} opacity={0.3} />
        <p>No completed trials yet</p>
      </div>
    )
  }

  const barW = Math.max(4, Math.min(36, innerW / completed.length - 3))

  const gridTicks = [0, 1, 2, 3, 4].map((i) => dom.lo + (i / 4) * (dom.hi - dom.lo))
  const tickDecimals = dom.hi - dom.lo < 0.05 ? 4 : dom.hi - dom.lo < 0.2 ? 3 : 2

  return (
    <div className="optuna-chart-wrap optuna-chart-wrap--framed" style={{ position: 'relative' }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="optuna-chart-svg"
        onMouseLeave={() => setTooltip(null)}
      >
        <defs>
          <linearGradient id="optunaScoreAreaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.22} />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
          </linearGradient>
          <filter id="optunaGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Y-axis gridlines (match min–max range) */}
        {gridTicks.map((tick) => {
          const y = yForVal(tick)
          return (
            <g key={tick}>
              <line
                x1={PAD.left} x2={PAD.left + innerW}
                y1={y} y2={y}
                stroke="rgba(148,163,184,0.08)" strokeDasharray="1 0"
              />
              <text x={PAD.left - 8} y={y + 4} textAnchor="end" fontSize={10} fill="#94a3b8" fontWeight={500}>
                {tick.toFixed(tickDecimals)}
              </text>
            </g>
          )
        })}

        {/* ── BAR MODE ── */}
        {chartMode === 'bar' && completed.map((t, i) => {
          const val = t.value as number
          const y0 = yForVal(val)
          const yBase = PAD.top + innerH
          const barH = yBase - y0
          const x = PAD.left + (i / completed.length) * innerW + (innerW / completed.length - barW) / 2
          const y = y0
          const isBest = t.number === bestNumber
          const isSelected = t.number === selectedTrial
          const color = isBest ? '#a78bfa' : isSelected ? '#38bdf8' : scoreColorDirected(val, direction)

          return (
            <g
              key={t.number}
              style={{ cursor: 'pointer' }}
              onClick={() => onSelectTrial(t.number)}
              onMouseEnter={() =>
                setTooltip({
                  trial: t.number,
                  value: val,
                  x: x + barW / 2,
                  y,
                  duration: resolveTrialWallDurationSec(t),
                })
              }
            >
              {isBest && (
                <rect
                  x={x - 1}
                  y={PAD.top}
                  width={barW + 2}
                  height={innerH}
                  fill="rgba(167,139,250,0.07)"
                  rx={4}
                />
              )}
              <rect
                x={x}
                y={y}
                width={barW}
                height={barH}
                rx={4}
                fill={isBest || isSelected ? color : scoreColorDirected(val, direction)}
                opacity={isSelected ? 1 : isBest ? 0.92 : 0.82}
                stroke={isSelected || isBest ? color : 'rgba(34,211,238,0.18)'}
                strokeWidth={isSelected || isBest ? 1.2 : 0.45}
                style={{ filter: isSelected || isBest ? 'drop-shadow(0 0 8px rgba(34,211,238,0.35))' : undefined }}
              />
              {isBest && (
                <text x={x + barW / 2} y={y - 5} textAnchor="middle" fontSize={9} fill="#c4b5fd" fontWeight={700}>
                  ★
                </text>
              )}
              <text
                x={x + barW / 2}
                y={H - 8}
                textAnchor="middle"
                fontSize={10}
                fill={isSelected ? '#38bdf8' : '#94a3b8'}
                fontWeight={isSelected ? 700 : 500}
              >
                {t.number}
              </text>
            </g>
          )
        })}

        {/* ── SCATTER/LINE MODE ── */}
        {chartMode === 'scatter' &&
          (() => {
            const points = completed.map((t, i) => {
              const val = t.value as number
              const cx = PAD.left + (i / (completed.length - 1 || 1)) * innerW
              const cy = yForVal(val)
              return { t, cx, cy, val }
            })
            const areaPath =
              points.length > 1
                ? `M ${points[0].cx},${PAD.top + innerH} L ${points.map((p) => `${p.cx},${p.cy}`).join(' L ')} L ${points[points.length - 1].cx},${PAD.top + innerH} Z`
                : ''
            return (
              <g>
                {areaPath ? <path d={areaPath} fill="url(#optunaScoreAreaGrad)" /> : null}
                {points.length > 1 && (
                  <polyline
                    points={points.map((p) => `${p.cx},${p.cy}`).join(' ')}
                    fill="none"
                    stroke="rgba(34,211,238,0.45)"
                    strokeWidth={2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                )}
                {rollingPts && rollingPts.length > 1 && (
                  <polyline
                    points={rollingPts.map((p) => `${p.cx},${p.cy}`).join(' ')}
                    fill="none"
                    stroke="#fbbf24"
                    strokeWidth={1.75}
                    strokeDasharray="6 4"
                    strokeLinejoin="round"
                    opacity={0.95}
                  />
                )}
                {points.length > 1 &&
                  (() => {
                    let frontier = maxM ? -Infinity : Infinity
                    const bpts = points.map((p) => {
                      frontier = maxM ? Math.max(frontier, p.val) : Math.min(frontier, p.val)
                      return `${p.cx},${yForVal(frontier)}`
                    })
                    return (
                      <polyline
                        points={bpts.join(' ')}
                        fill="none"
                        stroke="rgba(167,139,250,0.55)"
                        strokeWidth={2}
                        strokeDasharray="6 4"
                        strokeLinejoin="round"
                        filter="url(#optunaGlow)"
                      />
                    )
                  })()}
                {points.map(({ t, cx, cy, val }) => {
                  const isBest = t.number === bestNumber
                  const isSelected = t.number === selectedTrial
                  const color = isBest ? '#a78bfa' : isSelected ? '#38bdf8' : scoreColorDirected(val, direction)
                  return (
                    <g
                      key={t.number}
                      style={{ cursor: 'pointer' }}
                      onClick={() => onSelectTrial(t.number)}
                      onMouseEnter={() =>
                        setTooltip({
                          trial: t.number,
                          value: val,
                          x: cx,
                          y: cy,
                          duration: resolveTrialWallDurationSec(t),
                        })
                      }
                    >
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isSelected ? 9 : isBest ? 8 : 6}
                        fill={color}
                        opacity={0.95}
                        stroke="rgba(15,23,42,0.9)"
                        strokeWidth={1.5}
                        style={{
                          filter: isSelected || isBest ? `drop-shadow(0 0 6px ${color})` : undefined,
                        }}
                      />
                      {isBest && (
                        <text x={cx} y={cy - 11} textAnchor="middle" fontSize={9} fill="#c4b5fd" fontWeight={700}>
                          ★
                        </text>
                      )}
                      <text
                        x={cx}
                        y={H - 8}
                        textAnchor="middle"
                        fontSize={10}
                        fill={isSelected ? '#38bdf8' : '#94a3b8'}
                        fontWeight={isSelected ? 700 : 500}
                      >
                        {t.number}
                      </text>
                    </g>
                  )
                })}
              </g>
            )
          })()}

        <line
          x1={PAD.left}
          x2={PAD.left + innerW}
          y1={PAD.top + innerH}
          y2={PAD.top + innerH}
          stroke="rgba(148,163,184,0.2)"
        />
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + innerH} stroke="rgba(148,163,184,0.2)" />

        <text x={PAD.left + innerW / 2} y={H - 2} textAnchor="middle" fontSize={9} fill="#64748b" fontWeight={400}>
          Trial order (chronological)
        </text>
        <text
          x={12}
          y={PAD.top + innerH / 2}
          textAnchor="middle"
          fontSize={10}
          fill="#94a3b8"
          fontWeight={600}
          transform={`rotate(-90, 12, ${PAD.top + innerH / 2})`}
        >
          Objective
        </text>
      </svg>

      {tooltip && (
        <div
          className="optuna-chart-tooltip"
          style={{
            left: `${(tooltip.x / W) * 100}%`,
            top: `${(tooltip.y / H) * 100}%`,
          }}
        >
          <span className="optuna-chart-tooltip-label">Trial #{tooltip.trial}</span>
          <span className="optuna-chart-tooltip-value">{tooltip.value.toFixed(4)}</span>
          {tooltip.duration != null && tooltip.duration > 0 ? (
            <span className="optuna-chart-tooltip-meta">{formatDurationSec(tooltip.duration)}</span>
          ) : null}
        </div>
      )}
    </div>
  )
}

// ─── Chart: Per-Video Performance ─────────────────────────────────────────────

function PerVideoChart({
  trials,
  videoNames,
  bestNumber,
  onSelectTrial,
  selectedTrial,
}: {
  trials: OptunaTrialRow[]
  videoNames: string[]
  bestNumber: number | null
  onSelectTrial: (n: number) => void
  selectedTrial: number | null
}) {
  const completed = useMemo(
    () => trials.filter(isChartTrial).sort((a, b) => a.number - b.number),
    [trials],
  )

  const W = 920
  const H = 256
  const PAD = { top: 24, right: 22, bottom: 48, left: 44 }
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom

  const [tooltip, setTooltip] = useState<{ label: string; sub?: string; x: number; y: number } | null>(null)

  const scoreDomain = useMemo(() => {
    const all: number[] = []
    for (const t of completed) {
      for (const vn of videoNames) {
        const v = t.user_attrs[`score_${vn}`]
        if (typeof v === 'number') all.push(v)
      }
    }
    if (!all.length) return { lo: 0, hi: 1 }
    return chartAxisDomain(Math.min(...all), Math.max(...all))
  }, [completed, videoNames])

  const sSpan = Math.max(scoreDomain.hi - scoreDomain.lo, 1e-9)
  const yBottom = PAD.top + innerH - innerH * 0.04
  const yForScore = (val: number) =>
    PAD.top + innerH - ((val - scoreDomain.lo) / sSpan) * innerH * 0.92 - innerH * 0.04

  if (!completed.length || !videoNames.length) {
    return (
      <div className="optuna-chart-empty">
        <Target size={32} opacity={0.3} />
        <p>No per-video data yet</p>
      </div>
    )
  }

  const gridTicksPv = [0, 1, 2, 3, 4].map((i) => scoreDomain.lo + (i / 4) * (scoreDomain.hi - scoreDomain.lo))
  const tickDecimalsPv = scoreDomain.hi - scoreDomain.lo < 0.05 ? 4 : scoreDomain.hi - scoreDomain.lo < 0.2 ? 3 : 2

  const groupW = innerW / completed.length
  const barCount = videoNames.length
  const barW = Math.max(3, Math.min(20, (groupW - 8) / barCount))
  const groupPad = (groupW - barW * barCount) / 2

  return (
    <div style={{ position: 'relative' }}>
      <div className="optuna-per-video-legend">
        {videoNames.map((vn) => (
          <span
            key={vn}
            className="optuna-legend-pill"
            style={{ '--pill-color': VIDEO_COLORS[vn] ?? '#94a3b8' } as React.CSSProperties}
          >
            <span className="optuna-legend-dot" />
            {vn}
          </span>
        ))}
        <span className="optuna-legend-note">Bar height = score on that clip · grey whisker = min→max spread</span>
      </div>

      <div className="optuna-chart-wrap optuna-chart-wrap--framed" style={{ position: 'relative' }}>
        <svg viewBox={`0 0 ${W} ${H}`} className="optuna-chart-svg" onMouseLeave={() => setTooltip(null)}>
          <defs>
            {videoNames.map((vn) => (
              <linearGradient key={vn} id={`optunaVidGrad-${vn.replace(/[^a-zA-Z0-9]/g, '_')}`} x1="0" y1="1" x2="0" y2="0">
                <stop offset="0%" stopColor={VIDEO_COLORS[vn] ?? '#64748b'} stopOpacity={0.45} />
                <stop offset="100%" stopColor={VIDEO_COLORS[vn] ?? '#64748b'} stopOpacity={1} />
              </linearGradient>
            ))}
          </defs>

          {gridTicksPv.map((tick) => {
            const y = yForScore(tick)
            return (
              <g key={tick}>
                <line
                  x1={PAD.left}
                  x2={PAD.left + innerW}
                  y1={y}
                  y2={y}
                  stroke="rgba(148,163,184,0.08)"
                  strokeDasharray="4 3"
                />
                <text x={PAD.left - 8} y={y + 4} textAnchor="end" fontSize={10} fill="#94a3b8" fontWeight={500}>
                  {tick.toFixed(tickDecimalsPv)}
                </text>
              </g>
            )
          })}

          {completed.map((t, gi) => {
            const gx = PAD.left + gi * groupW
            const isBest = t.number === bestNumber
            const isSelected = t.number === selectedTrial
            const scores = videoNames
              .map((vn) => t.user_attrs[`score_${vn}`])
              .filter((v): v is number => typeof v === 'number')
            const vmin = scores.length ? Math.min(...scores) : 0
            const vmax = scores.length ? Math.max(...scores) : 0
            const yTopW = yForScore(vmax)
            const yBotW = yForScore(vmin)
            const cxWhisk = gx + groupW / 2

            return (
              <g key={t.number} style={{ cursor: 'pointer' }} onClick={() => onSelectTrial(t.number)}>
                {(isBest || isSelected) && (
                  <rect
                    x={gx + 1}
                    y={PAD.top}
                    width={groupW - 2}
                    height={innerH}
                    fill={isSelected ? 'rgba(56,189,248,0.06)' : 'rgba(167,139,250,0.06)'}
                    rx={4}
                  />
                )}
                {scores.length >= 2 && (
                  <line
                    x1={cxWhisk}
                    x2={cxWhisk}
                    y1={yTopW}
                    y2={yBotW}
                    stroke="rgba(148,163,184,0.35)"
                    strokeWidth={2}
                    strokeLinecap="round"
                  />
                )}
                {videoNames.map((vn, bi) => {
                  const scoreKey = `score_${vn}`
                  const val = t.user_attrs[scoreKey]
                  if (typeof val !== 'number') return null
                  const barTop = yForScore(val)
                  const barH = yBottom - barTop
                  const bx = gx + groupPad + bi * barW
                  const by = barTop
                  const gid = `optunaVidGrad-${vn.replace(/[^a-zA-Z0-9]/g, '_')}`
                  return (
                    <rect
                      key={vn}
                      x={bx}
                      y={by}
                      width={Math.max(barW - 1.5, 1)}
                      height={barH}
                      rx={3}
                      fill={`url(#${gid})`}
                      stroke="rgba(15,23,42,0.35)"
                      strokeWidth={0.5}
                      onMouseEnter={() =>
                        setTooltip({
                          label: `${vn}: ${val.toFixed(4)}`,
                          sub:
                            scores.length >= 2
                              ? `spread ${vmin.toFixed(3)}–${vmax.toFixed(3)} · trial #${t.number}`
                              : `trial #${t.number}`,
                          x: bx + barW / 2,
                          y: by,
                        })
                      }
                    />
                  )
                })}
                <text
                  x={gx + groupW / 2}
                  y={H - 8}
                  textAnchor="middle"
                  fontSize={10}
                  fill={isSelected ? '#38bdf8' : isBest ? '#c4b5fd' : '#94a3b8'}
                  fontWeight={isSelected || isBest ? 700 : 500}
                >
                  {t.number}
                </text>
                {isBest && (
                  <text x={gx + groupW / 2} y={H - 22} textAnchor="middle" fontSize={8} fill="#a78bfa" fontWeight={700}>
                    best
                  </text>
                )}
              </g>
            )
          })}

          <line
            x1={PAD.left}
            x2={PAD.left + innerW}
            y1={PAD.top + innerH}
            y2={PAD.top + innerH}
            stroke="rgba(148,163,184,0.2)"
          />
          <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + innerH} stroke="rgba(148,163,184,0.2)" />
          <text x={PAD.left + innerW / 2} y={H - 2} textAnchor="middle" fontSize={9} fill="#64748b">
            Trial # · click group for video
          </text>
        </svg>

        {tooltip && (
          <div
            className="optuna-chart-tooltip"
            style={{ left: `${(tooltip.x / W) * 100}%`, top: `${(tooltip.y / H) * 100}%` }}
          >
            <span className="optuna-chart-tooltip-value">{tooltip.label}</span>
            {tooltip.sub ? <span className="optuna-chart-tooltip-meta">{tooltip.sub}</span> : null}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Video trend lines (same data, easier to see which clip drags the score) ─

function VideoTrendChart({
  trials,
  videoNames,
  bestNumber,
  onSelectTrial,
  selectedTrial,
}: {
  trials: OptunaTrialRow[]
  videoNames: string[]
  bestNumber: number | null
  onSelectTrial: (n: number) => void
  selectedTrial: number | null
}) {
  const completed = useMemo(
    () => trials.filter(isChartTrial).sort((a, b) => a.number - b.number),
    [trials],
  )

  const W = 920
  const H = 228
  const PAD = { top: 20, right: 22, bottom: 44, left: 44 }
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom

  const [tooltip, setTooltip] = useState<{ label: string; x: number; y: number } | null>(null)

  const trendDomain = useMemo(() => {
    const all: number[] = []
    for (const t of completed) {
      for (const vn of videoNames) {
        const v = t.user_attrs[`score_${vn}`]
        if (typeof v === 'number') all.push(v)
      }
    }
    if (!all.length) return { lo: 0, hi: 1 }
    return chartAxisDomain(Math.min(...all), Math.max(...all))
  }, [completed, videoNames])
  const tSpan = Math.max(trendDomain.hi - trendDomain.lo, 1e-9)
  const yForTrend = (val: number) =>
    PAD.top + innerH - ((val - trendDomain.lo) / tSpan) * innerH * 0.92 - innerH * 0.04

  if (!completed.length || !videoNames.length) {
    return (
      <div className="optuna-chart-empty">
        <Layers size={28} opacity={0.3} />
        <p>No data for trend view</p>
      </div>
    )
  }

  const gridTicksTr = [0, 1, 2, 3, 4].map((i) => trendDomain.lo + (i / 4) * (trendDomain.hi - trendDomain.lo))
  const tickDecimalsTr = trendDomain.hi - trendDomain.lo < 0.05 ? 4 : trendDomain.hi - trendDomain.lo < 0.2 ? 3 : 2

  return (
    <div className="optuna-chart-wrap optuna-chart-wrap--framed" style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="optuna-chart-svg" onMouseLeave={() => setTooltip(null)}>
        {gridTicksTr.map((tick) => {
          const y = yForTrend(tick)
          return (
            <g key={tick}>
              <line x1={PAD.left} x2={PAD.left + innerW} y1={y} y2={y} stroke="rgba(148,163,184,0.08)" />
              <text x={PAD.left - 8} y={y + 4} textAnchor="end" fontSize={10} fill="#94a3b8">
                {tick.toFixed(tickDecimalsTr)}
              </text>
            </g>
          )
        })}
        {videoNames.map((vn) => {
          const pts = completed.map((t, i) => {
            const v = t.user_attrs[`score_${vn}`]
            const val = typeof v === 'number' ? v : 0
            const cx = PAD.left + (i / (completed.length - 1 || 1)) * innerW
            const cy = yForTrend(val)
            return { cx, cy, t, val }
          })
          const color = VIDEO_COLORS[vn] ?? '#94a3b8'
          return (
            <g key={vn}>
              {pts.length > 1 && (
                <polyline
                  points={pts.map((p) => `${p.cx},${p.cy}`).join(' ')}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  strokeLinejoin="round"
                  opacity={0.85}
                />
              )}
              {pts.map(({ cx, cy, t, val }) => {
                const isSel = t.number === selectedTrial
                const isBest = t.number === bestNumber
                return (
                  <circle
                    key={`${vn}-${t.number}`}
                    cx={cx}
                    cy={cy}
                    r={isSel ? 7 : isBest ? 6 : 4}
                    fill={color}
                    stroke={isSel ? '#f8fafc' : 'rgba(15,23,42,0.8)'}
                    strokeWidth={isSel ? 2 : 1}
                    opacity={0.92}
                    style={{ cursor: 'pointer' }}
                    onClick={() => onSelectTrial(t.number)}
                    onMouseEnter={() =>
                      setTooltip({ label: `${vn} · #${t.number}: ${val.toFixed(4)}`, x: cx, y: cy })
                    }
                  />
                )
              })}
            </g>
          )
        })}
        <line
          x1={PAD.left}
          x2={PAD.left + innerW}
          y1={PAD.top + innerH}
          y2={PAD.top + innerH}
          stroke="rgba(148,163,184,0.2)"
        />
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + innerH} stroke="rgba(148,163,184,0.2)" />
        <text x={PAD.left + innerW / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#64748b">
          Trial order · one line per benchmark clip
        </text>
      </svg>
      {tooltip && (
        <div className="optuna-chart-tooltip" style={{ left: `${(tooltip.x / W) * 100}%`, top: `${(tooltip.y / H) * 100}%` }}>
          <span className="optuna-chart-tooltip-value">{tooltip.label}</span>
        </div>
      )}
    </div>
  )
}

// ─── Trial duration (wall time from Optuna) ───────────────────────────────────

function DurationParamViewToggle({
  view,
  onViewChange,
}: {
  view: 'duration' | 'scatter'
  onViewChange: (v: 'duration' | 'scatter') => void
}) {
  return (
    <div className="optuna-chart-toolbar optuna-chart-toolbar--duration-param">
      <span className="optuna-chart-toolbar-label">View</span>
      <div className="optuna-chart-toggle">
        <button
          type="button"
          className={`optuna-chart-toggle-btn optuna-chart-toggle-btn--labeled${view === 'duration' ? ' active' : ''}`}
          onClick={() => onViewChange('duration')}
          title="Wall time per trial"
        >
          <Timer size={14} />
          <span className="optuna-chart-toggle-text">Duration</span>
        </button>
        <button
          type="button"
          className={`optuna-chart-toggle-btn optuna-chart-toggle-btn--labeled${view === 'scatter' ? ' active' : ''}`}
          onClick={() => onViewChange('scatter')}
          title="Parameter vs objective"
        >
          <ScatterChart size={14} />
          <span className="optuna-chart-toggle-text">Param vs score</span>
        </button>
      </div>
    </div>
  )
}

function TrialDurationChart({
  trials,
  onSelectTrial,
  selectedTrial,
}: {
  trials: OptunaTrialRow[]
  onSelectTrial: (n: number) => void
  selectedTrial: number | null
}) {
  const rows = useMemo(() => {
    const out: { t: OptunaTrialRow; duration: number }[] = []
    for (const t of trials) {
      if (!isChartTrial(t)) continue
      const duration = resolveTrialWallDurationSec(t)
      if (duration != null && duration > 0) out.push({ t, duration })
    }
    out.sort((a, b) => a.t.number - b.t.number)
    return out
  }, [trials])

  const W = 920
  /** Taller plot area so the chart fills the card when shown full-width (matches param scatter ~aspect). */
  const H = 340
  const PAD = { top: 20, right: 22, bottom: 34, left: 58 }
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom

  const [tooltip, setTooltip] = useState<{ label: string; x: number; y: number } | null>(null)

  if (!rows.length) {
    return (
      <div className="optuna-chart-empty optuna-chart-empty--sm">
        <Timer size={26} opacity={0.35} />
        <p>No wall-clock duration for completed trials (Optuna timing and sweep user_attrs were empty).</p>
      </div>
    )
  }

  const maxD = Math.max(...rows.map((r) => r.duration))
  const barW = Math.max(5, Math.min(40, innerW / rows.length - 4))
  const yTicks = [0, 0.25, 0.5, 0.75, 1]
  const durSpan = maxD

  return (
    <div className="optuna-chart-wrap optuna-chart-wrap--framed optuna-chart-wrap--duration" style={{ position: 'relative' }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="optuna-chart-svg"
        preserveAspectRatio="xMidYMid meet"
        onMouseLeave={() => setTooltip(null)}
        role="img"
        aria-label={`Wall time per trial, max ${formatDurationSec(maxD)}`}
      >
        <defs>
          <linearGradient id="optunaDurGrad" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stopColor="#6366f1" stopOpacity={0.85} />
            <stop offset="100%" stopColor="#a78bfa" stopOpacity={0.95} />
          </linearGradient>
        </defs>
        {yTicks.map((fr) => {
          const val = fr * maxD
          const y = PAD.top + innerH - fr * innerH * 0.92
          return (
            <g key={`dur-y-${fr}`}>
              <line x1={PAD.left} x2={PAD.left + innerW} y1={y} y2={y} stroke="rgba(148,163,184,0.1)" />
              <text x={PAD.left - 8} y={y + 4} textAnchor="end" fontSize={11} fill="#cbd5e1" fontWeight={500}>
                {formatDurationAxisTick(val, durSpan)}
              </text>
            </g>
          )
        })}
        {rows.map(({ t, duration: d }, i) => {
          const h = (d / maxD) * innerH * 0.92
          const x = PAD.left + (i / rows.length) * innerW + (innerW / rows.length - barW) / 2
          const y = PAD.top + innerH - h
          const isSel = t.number === selectedTrial
          const showDurOnBar = h >= 22
          return (
            <g
              key={t.number}
              style={{ cursor: 'pointer' }}
              onClick={() => onSelectTrial(t.number)}
              onMouseEnter={() => setTooltip({ label: `#${t.number} · ${formatDurationSec(d)}`, x: x + barW / 2, y })}
            >
              <rect
                x={x}
                y={y}
                width={barW}
                height={h}
                rx={4}
                fill="url(#optunaDurGrad)"
                opacity={isSel ? 1 : 0.75}
                stroke={isSel ? '#c4b5fd' : 'none'}
                strokeWidth={1.5}
              />
              {showDurOnBar ? (
                <text
                  x={x + barW / 2}
                  y={y + 13}
                  textAnchor="middle"
                  fontSize={10}
                  fill="rgba(248,250,252,0.92)"
                  fontWeight={600}
                  style={{ pointerEvents: 'none' }}
                >
                  {formatDurationSec(d)}
                </text>
              ) : null}
              <text
                x={x + barW / 2}
                y={H - 6}
                textAnchor="middle"
                fontSize={11}
                fill={isSel ? '#e2e8f0' : '#94a3b8'}
                fontWeight={isSel ? 700 : 500}
              >
                {t.number}
              </text>
            </g>
          )
        })}
        <line
          x1={PAD.left}
          x2={PAD.left + innerW}
          y1={PAD.top + innerH}
          y2={PAD.top + innerH}
          stroke="rgba(148,163,184,0.28)"
        />
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top + innerH * 0.08} y2={PAD.top + innerH} stroke="rgba(148,163,184,0.28)" />
        <text
          x={10}
          y={PAD.top + innerH / 2}
          textAnchor="middle"
          fontSize={12}
          fill="#e2e8f0"
          fontWeight={600}
          transform={`rotate(-90, 10, ${PAD.top + innerH / 2})`}
        >
          Wall time
        </text>
      </svg>
      <p className="optuna-duration-foot" role="note">
        Taller bars = more compute per trial. Values from Optuna timing or sweep <code className="mono">trial_duration_s</code>.
      </p>
      {tooltip && (
        <div className="optuna-chart-tooltip" style={{ left: `${(tooltip.x / W) * 100}%`, top: `${(tooltip.y / H) * 100}%` }}>
          <span className="optuna-chart-tooltip-value">{tooltip.label}</span>
        </div>
      )}
    </div>
  )
}

// ─── One numeric hyperparameter vs objective ─────────────────────────────────

function formatScatterTick(val: number, span: number): string {
  if (!Number.isFinite(val)) return '—'
  if (span < 1e-9) return val.toFixed(6)
  if (span < 0.0001) return val.toFixed(6)
  if (span < 0.01) return val.toFixed(5)
  if (span < 0.1) return val.toFixed(4)
  if (span < 1) return val.toFixed(3)
  return val.toFixed(2)
}

function ParamVsScoreScatter({
  trials,
  paramKey,
  direction,
  bestNumber,
  onSelectTrial,
  selectedTrial,
}: {
  trials: OptunaTrialRow[]
  paramKey: string
  direction: string
  bestNumber: number | null
  onSelectTrial: (n: number) => void
  selectedTrial: number | null
}) {
  const pts = useMemo(() => {
    const out: { t: OptunaTrialRow; px: number; py: number }[] = []
    for (const t of trials) {
      if (!isChartTrial(t)) continue
      const px = coerceParamNumber(t.params[paramKey])
      if (px === null) continue
      out.push({ t, px, py: t.value as number })
    }
    return out
  }, [trials, paramKey])

  const W = 960
  const H = 360
  const PAD = { top: 28, right: 28, bottom: 72, left: 72 }
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom

  const [tooltip, setTooltip] = useState<{
    trial: number
    paramLine: string
    scoreLine: string
    x: number
    y: number
  } | null>(null)

  const goalHint = isMaximizeStudy(direction) ? 'Higher is better' : 'Lower is better'

  if (!paramKey || pts.length < 2) {
    return (
      <div className="optuna-chart-empty optuna-chart-empty--sm">
        <ScatterChart size={28} opacity={0.35} />
        <p>Need ≥2 completed trials with numeric <code className="mono">{paramKey || 'param'}</code></p>
      </div>
    )
  }

  const xs = pts.map((p) => p.px)
  const ys = pts.map((p) => p.py)
  const xDom = chartAxisDomain(Math.min(...xs), Math.max(...xs))
  const yDom = chartAxisDomain(Math.min(...ys), Math.max(...ys))
  const spanX = Math.max(xDom.hi - xDom.lo, 1e-9)
  const spanY = Math.max(yDom.hi - yDom.lo, 1e-9)

  const project = (px: number, py: number) => ({
    cx: PAD.left + ((px - xDom.lo) / spanX) * innerW * 0.92 + innerW * 0.04,
    cy: PAD.top + innerH - ((py - yDom.lo) / spanY) * innerH * 0.92 - innerH * 0.04,
  })

  const yTicks = [0, 0.25, 0.5, 0.75, 1]
  const xTicks = [0, 0.25, 0.5, 0.75, 1]
  const xLabFs = paramKey.length > 48 ? 9 : paramKey.length > 36 ? 10 : 11

  return (
    <div className="optuna-chart-wrap optuna-chart-wrap--framed optuna-chart-wrap--scatter" style={{ position: 'relative' }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="optuna-chart-svg"
        preserveAspectRatio="xMidYMid meet"
        onMouseLeave={() => setTooltip(null)}
        role="img"
        aria-label={`${paramKey} versus objective score for ${pts.length} trials`}
      >
        {yTicks.map((fr) => {
          const val = yDom.lo + fr * spanY
          const y = PAD.top + innerH - ((val - yDom.lo) / spanY) * innerH * 0.92 - innerH * 0.04
          return (
            <g key={`y-${fr}`}>
              <line x1={PAD.left} x2={PAD.left + innerW} y1={y} y2={y} stroke="rgba(148,163,184,0.11)" />
              <text x={PAD.left - 10} y={y + 4} textAnchor="end" fontSize={12} fill="#cbd5e1" fontWeight={500}>
                {formatScatterTick(val, spanY)}
              </text>
            </g>
          )
        })}
        {xTicks.map((fr) => {
          const vx = xDom.lo + fr * spanX
          const x = PAD.left + innerW * 0.04 + fr * innerW * 0.92
          return (
            <g key={`x-${fr}`}>
              <line x1={x} x2={x} y1={PAD.top} y2={PAD.top + innerH} stroke="rgba(148,163,184,0.08)" />
              <text x={x} y={H - 44} textAnchor="middle" fontSize={12} fill="#cbd5e1" fontWeight={500}>
                {formatScatterTick(vx, spanX)}
              </text>
            </g>
          )
        })}
        {pts.map(({ t, px, py }) => {
          const { cx, cy } = project(px, py)
          const isSel = t.number === selectedTrial
          const isBest = bestNumber != null && t.number === bestNumber
          return (
            <g
              key={t.number}
              style={{ cursor: 'pointer' }}
              onClick={() => onSelectTrial(t.number)}
              onMouseEnter={() =>
                setTooltip({
                  trial: t.number,
                  paramLine: `${paramKey} = ${formatScatterTick(px, spanX)}`,
                  scoreLine: `Objective = ${formatScatterTick(py, spanY)}`,
                  x: cx,
                  y: cy,
                })
              }
            >
              <title>{`Trial #${t.number}: ${paramKey}=${px}, score=${py}`}</title>
              <circle
                cx={cx}
                cy={cy}
                r={isSel ? 10 : isBest ? 9 : 7}
                fill={isBest ? '#a78bfa' : scoreColorDirected(py, direction)}
                stroke={isSel ? '#f8fafc' : isBest ? '#ddd6fe' : 'rgba(15,23,42,0.85)'}
                strokeWidth={isSel ? 2.5 : isBest ? 2 : 1.2}
                opacity={0.92}
              />
            </g>
          )
        })}
        <line
          x1={PAD.left}
          x2={PAD.left + innerW}
          y1={PAD.top + innerH}
          y2={PAD.top + innerH}
          stroke="rgba(148,163,184,0.28)"
        />
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + innerH} stroke="rgba(148,163,184,0.28)" />
        <text x={PAD.left + innerW / 2} y={H - 14} textAnchor="middle" fontSize={xLabFs} fill="#e2e8f0" fontWeight={600}>
          {paramKey}
        </text>
        <text
          x={14}
          y={PAD.top + innerH / 2}
          textAnchor="middle"
          transform={`rotate(-90, 14, ${PAD.top + innerH / 2})`}
        >
          <tspan x={14} dy="-0.45em" fontSize={12} fill="#e2e8f0" fontWeight={600}>
            Objective score
          </tspan>
          <tspan x={14} dy="1.25em" fontSize={11} fill="#94a3b8" fontWeight={500}>
            {goalHint}
          </tspan>
        </text>
      </svg>
      {tooltip && (
        <div
          className="optuna-chart-tooltip optuna-chart-tooltip--multiline"
          style={{ left: `${(tooltip.x / W) * 100}%`, top: `${(tooltip.y / H) * 100}%` }}
        >
          <span className="optuna-chart-tooltip-label">Trial #{tooltip.trial}</span>
          <span className="optuna-chart-tooltip-value">{tooltip.paramLine}</span>
          <span className="optuna-chart-tooltip-meta">{tooltip.scoreLine}</span>
        </div>
      )}
    </div>
  )
}

// ─── Score bar (horizontal) ────────────────────────────────────────────────────

function ScoreBar({
  label,
  value,
  color,
  highlightDiff,
}: {
  label: string
  value: number
  color: string
  highlightDiff?: boolean
}) {
  return (
    <div className={`optuna-score-bar-row${highlightDiff ? ' optuna-score-bar-row--diff' : ''}`}>
      <span className="optuna-score-bar-label">{label}</span>
      <div className="optuna-score-bar-track">
        <div
          className="optuna-score-bar-fill"
          style={{ width: `${Math.min(value * 100, 100)}%`, background: color }}
        />
      </div>
      <span className="optuna-score-bar-num">{value.toFixed(4)}</span>
    </div>
  )
}

// ─── Radial gauge for aggregate score ─────────────────────────────────────────

function ScoreGauge({ value, label, highlightDiff }: { value: number; label: string; highlightDiff?: boolean }) {
  const r = 36
  const cx = 48
  const cy = 48
  const stroke = 7
  const circ = 2 * Math.PI * r
  // Only draw 75% of the circle (270 degrees) and start from bottom-left
  const arcFraction = 0.75
  const filled = Math.min(value, 1.0) * arcFraction
  const color = scoreColor(value)

  return (
    <div className={`optuna-gauge-wrap${highlightDiff ? ' optuna-gauge-wrap--diff' : ''}`}>
      <svg width={96} height={96} viewBox="0 0 96 96">
        {/* Track arc */}
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke="rgba(148,163,184,0.12)"
          strokeWidth={stroke}
          strokeDasharray={`${circ * arcFraction} ${circ}`}
          strokeDashoffset={circ * 0.125}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
        />
        {/* Fill arc */}
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={`${circ * filled} ${circ}`}
          strokeDashoffset={circ * 0.125}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
          style={{ filter: `drop-shadow(0 0 6px ${color}80)`, transition: 'stroke-dasharray 0.6s ease' }}
        />
        {/* Center text */}
        <text x={cx} y={cy - 4} textAnchor="middle" fontSize={15} fontWeight={700} fill="#f8fafc">
          {(value * 100).toFixed(1)}
        </text>
        <text x={cx} y={cy + 11} textAnchor="middle" fontSize={9} fill="#64748b">
          / 100
        </text>
      </svg>
      <span className="optuna-gauge-label">{label}</span>
    </div>
  )
}

// ─── Best / selected trial detail cards (config + scores) ─────────────────────

type TrialCardModel = {
  number: number
  state: string
  value: number | null
  params: Record<string, unknown>
  userAttrs: Record<string, unknown>
}

function trialRowToModel(t: OptunaTrialRow): TrialCardModel {
  return {
    number: t.number,
    state: t.state,
    value: typeof t.value === 'number' ? t.value : null,
    params: t.params,
    userAttrs: t.user_attrs,
  }
}

function bestToModel(b: OptunaBest): TrialCardModel {
  return {
    number: b.number,
    state: 'COMPLETE',
    value: typeof b.value === 'number' ? b.value : null,
    params: b.params,
    userAttrs: b.user_attrs,
  }
}

function TrialConfigCard({
  variant,
  model,
  heading,
  sameAsBest,
  paramDiffKeys,
  aggregateDiff,
  videoDiff,
  videoNames,
  onOpenVideo,
  onClearSelection,
}: {
  variant: 'best' | 'compare'
  model: TrialCardModel
  heading: string
  sameAsBest?: boolean
  paramDiffKeys: Set<string>
  aggregateDiff: boolean
  videoDiff: Record<string, boolean>
  videoNames: string[]
  onOpenVideo: () => void
  /** Shown on compare card only — clears table/chart selection */
  onClearSelection?: () => void
}) {
  const Icon = variant === 'best' ? Sparkles : GitCompare
  const icoClass = variant === 'best' ? 'optuna-best-ico' : 'optuna-compare-ico'
  return (
    <section
      className={variant === 'best' ? 'optuna-best-card' : 'optuna-best-card optuna-best-card--compare'}
    >
      <div className="optuna-best-head">
        <Icon size={22} className={icoClass} aria-hidden />
        <div className="optuna-best-head__body">
          <h2>{heading}</h2>
          <p className="optuna-muted">
            Trial <strong>#{model.number}</strong>
            {model.state !== 'COMPLETE' && (
              <>
                {' '}
                ·{' '}
                <span className={`optuna-pill optuna-pill--${model.state.toLowerCase()}`}>{model.state}</span>
              </>
            )}
            {typeof model.value === 'number' ? (
              <>
                {' '}
                · aggregate score <strong>{formatValue(model.value)}</strong>
              </>
            ) : (
              <> · score unavailable</>
            )}
          </p>
          {variant === 'compare' && sameAsBest ? (
            <p className="optuna-compare-same-hint">Same trial as best — no differences below.</p>
          ) : null}
        </div>
        {variant === 'compare' && onClearSelection ? (
          <button
            type="button"
            className="optuna-trial-deselect"
            onClick={onClearSelection}
            aria-label="Deselect trial"
            title="Deselect trial"
          >
            <X size={18} aria-hidden />
          </button>
        ) : null}
        <button type="button" className="btn btn--compact" onClick={onOpenVideo}>
          Open video
          <ChevronRight size={16} aria-hidden />
        </button>
      </div>

      <div className="optuna-best-score-wrap">
        {typeof model.value === 'number' ? (
          <ScoreGauge value={model.value} label="aggregate" highlightDiff={aggregateDiff} />
        ) : (
          <div className="optuna-gauge-placeholder">
            <span className="optuna-muted">No aggregate score</span>
          </div>
        )}
        <div className="optuna-best-video-bars">
          <h3 className="optuna-h3">Video breakdown</h3>
          {videoNames.length ? (
            videoNames.map((vn) => {
              const val = model.userAttrs[`score_${vn}`]
              if (typeof val !== 'number') return null
              return (
                <ScoreBar
                  key={vn}
                  label={vn}
                  value={val}
                  color={VIDEO_COLORS[vn] ?? '#94a3b8'}
                  highlightDiff={videoDiff[vn] ?? false}
                />
              )
            })
          ) : (
            Object.entries(model.userAttrs)
              .filter(([k]) => k.startsWith('score_'))
              .map(([k, v]) => {
                if (typeof v !== 'number') return null
                const name = k.replace('score_', '')
                return (
                  <ScoreBar
                    key={k}
                    label={name}
                    value={v}
                    color={VIDEO_COLORS[name] ?? '#94a3b8'}
                    highlightDiff={videoDiff[name] ?? false}
                  />
                )
              })
          )}
        </div>
      </div>

      <div style={{ marginTop: '1rem' }}>
        <h3 className="optuna-h3">Suggested params</h3>
        <div className="optuna-params-grid">
          {Object.entries(model.params)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([k, v]) => {
              const full = String(v)
              const isPathLike = (k.includes('weight') || k.includes('path')) && full.includes('/')
              const compact = isPathLike ? (full.split('/').pop() || full) : full
              return (
                <div
                  key={k}
                  className={`optuna-param-pill${paramDiffKeys.has(k) ? ' optuna-param-pill--diff' : ''}`}
                >
                  <span className="optuna-param-key">{k}</span>
                  <span className="optuna-param-val" title={full}>{compact}</span>
                  {isPathLike ? <span className="optuna-param-subval" title={full}>{full}</span> : null}
                </div>
              )
            })}
        </div>
      </div>
    </section>
  )
}

// ─── Insight Panel — param correlation analysis ────────────────────────────────

type ParamInsight = {
  param: string
  direction: 'higher' | 'lower' | 'categorical'
  topValue: string
  correlation: 'strong' | 'weak'
  description: string
}

function analyzeParamInsights(trials: OptunaTrialRow[]): ParamInsight[] {
  const completed = trials.filter(isChartTrial)
  if (completed.length < 2) return []

  // Split into top half / bottom half by score (for MAXIMIZE)
  const sorted = [...completed].sort((a, b) => (b.value as number) - (a.value as number))
  const half = Math.ceil(sorted.length / 2)
  const topTrials = sorted.slice(0, half)
  const bottomTrials = sorted.slice(half)

  const allParams = new Set<string>()
  completed.forEach((t) => Object.keys(t.params).forEach((k) => allParams.add(k)))

  const insights: ParamInsight[] = []

  for (const param of allParams) {
    const topVals = topTrials.map((t) => t.params[param]).filter((v) => v !== undefined && v !== null)
    const botVals = bottomTrials.map((t) => t.params[param]).filter((v) => v !== undefined && v !== null)

    if (!topVals.length || !botVals.length) continue

    const isNumeric = topVals.every((v) => typeof v === 'number') && botVals.every((v) => typeof v === 'number')

    if (isNumeric) {
      const topAvg = (topVals as number[]).reduce((s, v) => s + v, 0) / topVals.length
      const botAvg = (botVals as number[]).reduce((s, v) => s + v, 0) / botVals.length
      const diff = topAvg - botAvg
      const relDiff = Math.abs(diff) / (Math.max(topAvg, botAvg, 0.001))

      if (relDiff < 0.04) continue // negligible difference

      insights.push({
        param,
        direction: diff > 0 ? 'higher' : 'lower',
        topValue: topAvg.toFixed(3),
        correlation: relDiff > 0.15 ? 'strong' : 'weak',
        description: `Best trials use ${diff > 0 ? 'higher' : 'lower'} values (avg ${topAvg.toFixed(3)} vs ${botAvg.toFixed(3)})`,
      })
    } else {
      // Categorical: find most common value in top trials
      const topCounts: Record<string, number> = {}
      topVals.forEach((v) => {
        const k = String(v)
        topCounts[k] = (topCounts[k] ?? 0) + 1
      })
      const topMode = Object.entries(topCounts).sort((a, b) => b[1] - a[1])[0]
      if (!topMode) continue

      const botCounts: Record<string, number> = {}
      botVals.forEach((v) => {
        const k = String(v)
        botCounts[k] = (botCounts[k] ?? 0) + 1
      })
      const topModeInBot = (botCounts[topMode[0]] ?? 0) / botVals.length
      const topModeInTop = topMode[1] / topVals.length

      if (topModeInTop - topModeInBot < 0.25) continue

      insights.push({
        param,
        direction: 'categorical',
        topValue: topMode[0],
        correlation: topModeInTop - topModeInBot > 0.5 ? 'strong' : 'weak',
        description: `Best trials prefer "${topMode[0]}" (${Math.round(topModeInTop * 100)}% of top vs ${Math.round(topModeInBot * 100)}% of bottom)`,
      })
    }
  }

  // Sort: strong first, then alphabetical
  return insights.sort((a, b) => {
    if (a.correlation !== b.correlation) return a.correlation === 'strong' ? -1 : 1
    return a.param.localeCompare(b.param)
  })
}

function InsightSignalList({ insights }: { insights: ParamInsight[] }) {
  return (
    <>
      {insights.map((ins) => (
        <div
          key={ins.param}
          className={`optuna-insight-item optuna-insight-item--${ins.correlation}`}
        >
          <div className="optuna-insight-item-top">
            <span className="optuna-insight-param">{ins.param}</span>
            {ins.direction === 'higher' && (
              <span className="optuna-insight-dir optuna-insight-dir--up">
                <ArrowUpRight size={12} /> higher
              </span>
            )}
            {ins.direction === 'lower' && (
              <span className="optuna-insight-dir optuna-insight-dir--down">
                <ArrowUpRight size={12} style={{ transform: 'rotate(90deg)' }} /> lower
              </span>
            )}
            {ins.direction === 'categorical' && (
              <span className="optuna-insight-dir optuna-insight-dir--cat">
                <Zap size={12} /> {ins.topValue}
              </span>
            )}
          </div>
          <p className="optuna-insight-desc">{ins.description}</p>
        </div>
      ))}
    </>
  )
}

function InsightPanel({
  trials,
  columnHeightPx,
}: {
  trials: OptunaTrialRow[]
  columnHeightPx: number | null
}) {
  const [modalOpen, setModalOpen] = useState(false)
  const insights = useMemo(() => analyzeParamInsights(trials), [trials])

  const completed = trials.filter(isChartTrial).length

  const matchCol = columnHeightPx != null && columnHeightPx > 0
  const panelStyle: React.CSSProperties | undefined = matchCol
    ? { height: columnHeightPx, maxHeight: columnHeightPx }
    : undefined

  useEffect(() => {
    if (!modalOpen) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [modalOpen])

  useEffect(() => {
    if (!modalOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setModalOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [modalOpen])

  if (completed < 2) {
    return (
      <div
        className={`optuna-insight-panel${matchCol ? ' optuna-insight-panel--match-col' : ''}`}
        style={panelStyle}
      >
        <div className="optuna-insight-header">
          <Lightbulb size={16} />
          <span>What's helping?</span>
        </div>
        <p className="optuna-muted" style={{ fontSize: '0.82rem', margin: '0.75rem 0 0' }}>
          Need at least 2 completed trials to detect parameter trends.
        </p>
      </div>
    )
  }

  if (!insights.length) {
    return (
      <div
        className={`optuna-insight-panel${matchCol ? ' optuna-insight-panel--match-col' : ''}`}
        style={panelStyle}
      >
        <div className="optuna-insight-header">
          <Lightbulb size={16} />
          <span>What's helping?</span>
        </div>
        <p className="optuna-muted" style={{ fontSize: '0.82rem', margin: '0.75rem 0 0' }}>
          No strong parameter trends detected yet — run more trials to surface insights.
        </p>
      </div>
    )
  }

  return (
    <>
      <div
        className={`optuna-insight-panel optuna-insight-panel--interactive${matchCol ? ' optuna-insight-panel--match-col' : ''}`}
        style={panelStyle}
        role="button"
        tabIndex={0}
        title="Click to expand all signals"
        aria-label="Open signals in full screen"
        onClick={() => setModalOpen(true)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            setModalOpen(true)
          }
        }}
      >
        <div className="optuna-insight-header">
          <Lightbulb size={16} />
          <span>What's helping?</span>
          <span className="optuna-insight-badge">{insights.length} signals</span>
        </div>
        <div className="optuna-insight-list">
          <InsightSignalList insights={insights} />
        </div>
      </div>

      {modalOpen ? (
        <div
          className="optuna-insight-modal-backdrop"
          role="presentation"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setModalOpen(false)
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="optuna-insight-modal-title"
            className="optuna-insight-modal-panel glass-panel"
          >
            <div className="optuna-insight-modal-toolbar">
              <div className="optuna-insight-header" style={{ marginBottom: 0 }}>
                <Lightbulb size={16} />
                <span id="optuna-insight-modal-title">What's helping?</span>
                <span className="optuna-insight-badge">{insights.length} signals</span>
              </div>
              <button
                type="button"
                className="optuna-insight-modal-close"
                onClick={() => setModalOpen(false)}
                aria-label="Close"
              >
                <X size={20} />
              </button>
            </div>
            <div className="optuna-insight-modal-scroll">
              <InsightSignalList insights={insights} />
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}

// ─── Prune reason helper ─────────────────────────────────────────────────────

function getPruneReason(t: OptunaTrialRow): string | null {
  // Look in user_attrs for common error/prune signals the sweep tool writes
  const ua = t.user_attrs
  if (!ua) return null
  // Direct error key
  if (typeof ua.error === 'string' && ua.error) return ua.error
  if (typeof ua.prune_reason === 'string' && ua.prune_reason) return ua.prune_reason
  if (typeof ua.fail_reason === 'string' && ua.fail_reason) return ua.fail_reason
  // Score of 0 on all videos → pipeline crashed (produced no output)
  const scoreKeys = Object.keys(ua).filter((k) => k.startsWith('score_'))
  if (scoreKeys.length > 0 && scoreKeys.every((k) => ua[k] === 0)) {
    return 'All video scores are 0 — pipeline likely crashed or produced no output'
  }
  // Very low partial score at prune time
  if (typeof ua.partial_score === 'number') {
    return `Intermediate score ${(ua.partial_score as number).toFixed(4)} was below threshold`
  }
  // Fall back to the trial value itself if PRUNED
  if (t.state === 'PRUNED' && typeof t.value === 'number') {
    return `Score at prune: ${t.value.toFixed(4)}`
  }
  return null
}

// ─── Optuna Direction Panel ──────────────────────────────────────────────────

type ParamRange = {
  param: string
  min: number
  max: number
  bestVal: number
  recentAvg: number
  globalAvg: number
  converging: boolean
  direction: 'up' | 'down' | 'stable'
}

function buildParamRanges(trials: OptunaTrialRow[], bestParams: Record<string, unknown>): ParamRange[] {
  const completed = trials.filter(isChartTrial)
  if (completed.length < 2) return []
  // Recent = last 40%
  const recentCutoff = Math.max(1, Math.floor(completed.length * 0.6))
  const recentTrials = completed.slice(recentCutoff)
  const earlyTrials = completed.slice(0, recentCutoff)

  const numericParams = new Set<string>()
  completed.forEach((t) => {
    Object.entries(t.params).forEach(([k, v]) => {
      if (coerceParamNumber(v) !== null) numericParams.add(k)
    })
  })

  const ranges: ParamRange[] = []
  for (const param of numericParams) {
    const allVals = completed
      .map((t) => coerceParamNumber(t.params[param]))
      .filter((v): v is number => v !== null)
    if (allVals.length < 2) continue
    const recentVals = recentTrials
      .map((t) => coerceParamNumber(t.params[param]))
      .filter((v): v is number => v !== null)
    const earlyVals = earlyTrials
      .map((t) => coerceParamNumber(t.params[param]))
      .filter((v): v is number => v !== null)
    const globalAvg = allVals.reduce((s, v) => s + v, 0) / allVals.length
    const recentAvg = recentVals.length ? recentVals.reduce((s, v) => s + v, 0) / recentVals.length : globalAvg
    const earlyAvg = earlyVals.length ? earlyVals.reduce((s, v) => s + v, 0) / earlyVals.length : globalAvg
    const min = Math.min(...allVals)
    const max = Math.max(...allVals)
    const spread = max - min
    // Check if recent trials are converging (smaller std dev)
    const recentSpread = recentVals.length > 1
      ? Math.max(...recentVals) - Math.min(...recentVals)
      : spread
    const converging = spread > 0.001 && recentSpread < spread * 0.55
    const drift = recentAvg - earlyAvg
    const direction = Math.abs(drift) < spread * 0.1 ? 'stable' : drift > 0 ? 'up' : 'down'
    const bestRaw = bestParams[param]
    const bestCoerced = coerceParamNumber(bestRaw)
    ranges.push({
      param,
      min, max,
      bestVal: bestCoerced ?? globalAvg,
      recentAvg,
      globalAvg,
      converging,
      direction,
    })
  }
  return ranges.sort((a, b) => {
    // Converging params first (most interesting signal), then stable, then others
    if (a.converging !== b.converging) return a.converging ? -1 : 1
    return a.param.localeCompare(b.param)
  })
}

type CategoricalShift = { param: string; earlyMode: string; recentMode: string }

function buildCategoricalShifts(trials: OptunaTrialRow[]): CategoricalShift[] {
  const completed = trials.filter(isChartTrial).sort((a, b) => a.number - b.number)
  if (completed.length < 3) return []
  const recentCutoff = Math.max(1, Math.floor(completed.length * 0.6))
  const recentTrials = completed.slice(recentCutoff)
  const earlyTrials = completed.slice(0, recentCutoff)
  const keys = new Set<string>()
  completed.forEach((t) => Object.keys(t.params).forEach((k) => keys.add(k)))
  const out: CategoricalShift[] = []

  const modeOf = (subset: OptunaTrialRow[], param: string): string => {
    const counts: Record<string, number> = {}
    subset.forEach((t) => {
      const v = t.params[param]
      if (v === undefined || v === null) return
      if (coerceParamNumber(v) !== null) return
      const k = String(v)
      counts[k] = (counts[k] ?? 0) + 1
    })
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
    return top ? top[0] : '—'
  }

  for (const param of keys) {
    const anyNonNumeric = completed.some((t) => {
      const v = t.params[param]
      if (v === undefined || v === null) return false
      return coerceParamNumber(v) === null
    })
    if (!anyNonNumeric) continue
    const em = modeOf(earlyTrials, param)
    const rm = modeOf(recentTrials, param)
    if (em === '—' && rm === '—') continue
    out.push({ param, earlyMode: em, recentMode: rm })
  }
  return out.sort((a, b) => a.param.localeCompare(b.param))
}

/** Objective vs trial order + best-so-far — always meaningful even when all hparams are categorical */
function ObjectiveTrajectoryMini({
  trials,
  studyDirection,
}: {
  trials: OptunaTrialRow[]
  studyDirection: string
}) {
  const completed = useMemo(
    () => trials.filter(isChartTrial).sort((a, b) => a.number - b.number),
    [trials],
  )
  const maxM = useMemo(() => isMaximizeStudy(studyDirection), [studyDirection])

  if (completed.length < 2) return null

  const W = 900
  const H = 128
  const PAD = { top: 16, right: 18, bottom: 44, left: 46 }
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom
  const vals = completed.map((t) => t.value as number)
  const dom = chartAxisDomain(Math.min(...vals), Math.max(...vals))
  const span = Math.max(dom.hi - dom.lo, 1e-9)
  const yAt = (v: number) => PAD.top + innerH - ((v - dom.lo) / span) * innerH * 0.92 - innerH * 0.04

  const points = completed.map((t, i) => {
    const v = t.value as number
    const cx = PAD.left + (i / (completed.length - 1 || 1)) * innerW
    const cy = yAt(v)
    return { cx, cy, v, n: t.number }
  })

  let frontier = maxM ? -Infinity : Infinity
  const frontierPts = points.map((p) => {
    frontier = maxM ? Math.max(frontier, p.v) : Math.min(frontier, p.v)
    return `${p.cx},${yAt(frontier)}`
  })

  return (
    <div className="optuna-dir-trajectory">
      <p className="optuna-dir-trajectory-caption">
        Objective value after each completed trial (order = time). Purple dashed = best-so-far ({maxM ? 'maximize' : 'minimize'}
        ).
      </p>
      <div className="optuna-chart-wrap optuna-chart-wrap--framed">
        <svg viewBox={`0 0 ${W} ${H}`} className="optuna-chart-svg" aria-hidden>
          {[0, 0.5, 1].map((fr) => {
            const lab = dom.lo + fr * (dom.hi - dom.lo)
            const y = yAt(lab)
            return (
              <g key={fr}>
                <line x1={PAD.left} x2={PAD.left + innerW} y1={y} y2={y} stroke="rgba(148,163,184,0.07)" />
                <text x={PAD.left - 8} y={y + 4} textAnchor="end" fontSize={9} fill="#64748b">
                  {lab.toFixed(3)}
                </text>
              </g>
            )
          })}
          {points.length > 1 && (
            <polyline
              points={frontierPts.join(' ')}
              fill="none"
              stroke="rgba(167,139,250,0.65)"
              strokeWidth={2}
              strokeDasharray="5 4"
              strokeLinejoin="round"
            />
          )}
          {points.length > 1 && (
            <polyline
              points={points.map((p) => `${p.cx},${p.cy}`).join(' ')}
              fill="none"
              stroke="rgba(34,211,238,0.45)"
              strokeWidth={1.75}
              strokeLinejoin="round"
            />
          )}
          {points.map((p) => (
            <circle key={p.n} cx={p.cx} cy={p.cy} r={4} fill="#22d3ee" opacity={0.85} />
          ))}
          <line
            x1={PAD.left}
            x2={PAD.left + innerW}
            y1={PAD.top + innerH}
            y2={PAD.top + innerH}
            stroke="rgba(148,163,184,0.2)"
          />
          <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + innerH} stroke="rgba(148,163,184,0.2)" />
          <text x={PAD.left + innerW / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="#64748b">
            Completed trials (chronological)
          </text>
        </svg>
      </div>
    </div>
  )
}

function OptunaDirectionPanel({
  trials,
  bestParams,
  studyDirection,
}: {
  trials: OptunaTrialRow[]
  bestParams: Record<string, unknown>
  studyDirection: string
}) {
  const completed = trials.filter(isChartTrial).length
  const pruned = trials.filter((t) => t.state === 'PRUNED').length
  const ranges = useMemo(() => buildParamRanges(trials, bestParams), [trials, bestParams])
  const catShifts = useMemo(() => buildCategoricalShifts(trials), [trials])

  return (
    <div className="optuna-direction-panel">
      <div className="optuna-direction-header">
        <GitBranch size={16} />
        <span>Optuna&apos;s direction</span>
        <span className="optuna-insight-badge">{completed} complete · {pruned} pruned</span>
      </div>

      {completed >= 2 ? (
        <ObjectiveTrajectoryMini trials={trials} studyDirection={studyDirection} />
      ) : (
        <p className="optuna-muted" style={{ fontSize: '0.82rem', margin: '0.35rem 0 0' }}>
          Finish at least 2 complete trials to see the objective trajectory.
        </p>
      )}

      {completed < 3 ? (
        <p className="optuna-muted" style={{ fontSize: '0.82rem', margin: '0.85rem 0 0' }}>
          Run at least 3 complete trials for per-parameter tracks (early vs recent batches).
        </p>
      ) : (
        <>
          <div
            className="optuna-direction-explainer"
            role="region"
            aria-labelledby="optuna-direction-howto-title"
          >
            <p id="optuna-direction-howto-title" className="optuna-direction-explainer-title">
              How to read each slider
            </p>
            <ul className="optuna-direction-explainer-list">
              <li>
                The bar is the <strong>observed range</strong> for this parameter: left = minimum, right = maximum seen in{' '}
                <strong>any completed trial</strong> so far.
              </li>
              <li className="optuna-direction-explainer-dots">
                <span>
                  <span className="optuna-dir-dot optuna-dir-dot--best" aria-hidden />
                  <strong>Purple</strong> = this parameter&apos;s value in the <strong>current best trial</strong>.
                </span>
                <span>
                  <span className="optuna-dir-dot optuna-dir-dot--recent" aria-hidden />
                  <strong>Cyan</strong> = <strong>mean</strong> of this parameter across the <strong>latest ~40%</strong> of
                  completed trials (most recent batch).
                </span>
              </li>
              <li>
                <span className="optuna-dir-converging">◆ Narrowing</span> — recent trials fall in a{' '}
                <strong>tighter min→max band</strong> than the full study (spread shrank): the search is{' '}
                <em>homing in</em> on a region, not “converging toward” the other dot.
              </li>
              <li>
                <span className="optuna-dir-up">↑</span> / <span className="optuna-dir-down">↓</span> — the{' '}
                <strong>recent batch average</strong> moved up or down vs the <strong>earlier batch</strong> average.{' '}
                <span className="optuna-dir-stable">— Stable</span> — little shift between those averages.
              </li>
            </ul>
          </div>

          <div className="optuna-direction-legend-chips" aria-hidden>
            <span className="optuna-dir-chip optuna-dir-chip--conv">◆ narrowing</span>
            <span className="optuna-dir-chip optuna-dir-chip--up">↑ recent avg up</span>
            <span className="optuna-dir-chip optuna-dir-chip--down">↓ recent avg down</span>
            <span className="optuna-dir-chip optuna-dir-chip--st">— stable</span>
          </div>

          {ranges.length > 0 ? (
            <div className="optuna-direction-list">
              {ranges.map((r) => {
                const pct = r.min === r.max ? 0.5 : (r.recentAvg - r.min) / (r.max - r.min)
                const bestPct = r.min === r.max ? 0.5 : (r.bestVal - r.min) / (r.max - r.min)
                return (
                  <div key={r.param} className={`optuna-dir-row${r.converging ? ' optuna-dir-row--converging' : ''}`}>
                    <div className="optuna-dir-row-head">
                      <span className="optuna-dir-param">{r.param}</span>
                      <span className={`optuna-dir-badge optuna-dir-badge--${r.direction}${r.converging ? ' optuna-dir-badge--conv' : ''}`}>
                        {r.converging ? '◆' : r.direction === 'up' ? '↑' : r.direction === 'down' ? '↓' : '—'}
                        {r.converging
                          ? ' narrowing'
                          : r.direction === 'stable'
                            ? ' stable'
                            : r.direction === 'up'
                              ? ' recent avg up'
                              : ' recent avg down'}
                      </span>
                    </div>
                    <div
                      className="optuna-dir-track"
                      title="Min→max of this parameter across all completed trials in this snapshot"
                    >
                      <div
                        className="optuna-dir-best-marker"
                        style={{ left: `${Math.min(Math.max(bestPct * 100, 0), 100)}%` }}
                        title={`Best trial: ${r.bestVal.toFixed(4)} (this parameter in the best-scoring trial)`}
                      />
                      <div
                        className="optuna-dir-recent-marker"
                        style={{ left: `${Math.min(Math.max(pct * 100, 0), 100)}%` }}
                        title={`Recent batch mean: ${r.recentAvg.toFixed(4)} (average in latest ~40% of completed trials)`}
                      />
                    </div>
                    <div className="optuna-dir-row-vals">
                      <span>{r.min.toFixed(3)}</span>
                      <span className="optuna-dir-recent-val">recent avg {r.recentAvg.toFixed(3)}</span>
                      <span>{r.max.toFixed(3)}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : catShifts.length > 0 ? (
            <div className="optuna-dir-categorical">
              <p className="optuna-muted optuna-dir-categorical-intro">
                No numeric hyperparameters in this snapshot (or all values are categorical). Showing the most common value in{' '}
                <strong>early</strong> vs <strong>recent</strong> trials for each categorical param:
              </p>
              <ul className="optuna-dir-cat-list">
                {catShifts.map((c) => (
                  <li key={c.param}>
                    <span className="optuna-dir-param">{c.param}</span>
                    <span className="optuna-dir-cat-val">
                      <code>{c.earlyMode}</code>
                      <span aria-hidden> → </span>
                      <code>{c.recentMode}</code>
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="optuna-muted" style={{ fontSize: '0.82rem', margin: '0.35rem 0 0' }}>
              No hyperparameters were recorded on trials, or values could not be parsed — only the objective chart above applies.
            </p>
          )}
        </>
      )}
    </div>
  )
}

// ─── Main Page ─────────────────────────────────────────────────────────────────

export function OptunaSweepPage() {
  const [sweepMode, setSweepMode] = useState<SweepMode>('v4')
  const [data, setData] = useState<OptunaSweepStatus | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedTrial, setSelectedTrial] = useState<number | null>(null)
  const [sequence, setSequence] = useState<string>('')
  const [media, setMedia] = useState<MediaResponse | null>(null)
  const [mediaLoading, setMediaLoading] = useState(false)
  const [proofMode, setProofMode] = useState<'video' | 'progress'>('video')
  const [pickedClip, setPickedClip] = useState<string | null>(null)
  const [chartMode, setChartMode] = useState<'bar' | 'scatter'>('bar')
  const [showRollingAvg, setShowRollingAvg] = useState(true)
  const [perVideoMode, setPerVideoMode] = useState<'bars' | 'lines'>('bars')
  const [durationVsParamView, setDurationVsParamView] = useState<'duration' | 'scatter'>('duration')
  const [scatterParam, setScatterParam] = useState<string>('')
  const analyticsBestColRef = useRef<HTMLDivElement>(null)
  const [analyticsBestColHeight, setAnalyticsBestColHeight] = useState<number | null>(null)
  const lastStatusRef = useRef<{ updatedUnix: number; nComplete: number } | null>(null)
  const [trialTableSort, setTrialTableSort] = useState<{ col: TrialTableSortCol; dir: 'asc' | 'desc' }>({
    col: 'trial',
    dir: 'desc',
  })
  const [stageFilter, setStageFilter] = useState<string>('all')

  const sync = useCallback(async (opts?: { syncTrialArtifacts?: boolean }) => {
    const syncTrialArtifacts = opts?.syncTrialArtifacts ?? false
    setLoading(true)
    setErr(null)
    let pullWarn: string | null = null
    try {
      try {
        const pr = await fetch(pullLambdaUrl(sweepMode), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sync_trial_artifacts: syncTrialArtifacts }),
        })
        const pj = await pr.json().catch(() => null)
        if (!pr.ok) {
          pullWarn = formatHttpDetail(pr, pj)
        }
      } catch (e) {
        pullWarn = e instanceof Error ? e.message : String(e)
      }

      const sr = await fetch(statusUrl(sweepMode))
      const t = await sr.text()
      if (!sr.ok) throw new Error(t || sr.statusText)
      const parsed = JSON.parse(t) as OptunaSweepStatus
      // #region agent log
      fetch('http://127.0.0.1:7840/ingest/2e89ee7e-7e09-4947-9405-6f6b907db65f',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'1f2ff8'},body:JSON.stringify({sessionId:'1f2ff8',runId:'slow-sweep-check',hypothesisId:'H1_H3_H4_H5',location:'OptunaSweepPage.tsx:sync',message:'status_poll_snapshot',data:{mode:sweepMode,updated_unix:parsed.updated_unix,n_trials_total:parsed.n_trials_total,n_complete:parsed.n_complete,n_other:parsed.n_other,running:(parsed.trials||[]).filter((r)=>r.state==='RUNNING').slice(0,3).map((r)=>({n:r.number,video:r.user_attrs?.progress_current_video,phase:r.user_attrs?.progress_current_phase,pct:r.user_attrs?.progress_current_phase_pct,duration_s:r.duration_s}))},timestamp:Date.now()})}).catch(()=>{});
      // #endregion
      const prev = lastStatusRef.current
      if (prev && (parsed.n_complete < prev.nComplete || parsed.updated_unix < prev.updatedUnix)) {
        // #region agent log
        fetch('http://127.0.0.1:7840/ingest/2e89ee7e-7e09-4947-9405-6f6b907db65f',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'1f2ff8'},body:JSON.stringify({sessionId:'1f2ff8',runId:'slow-sweep-check',hypothesisId:'H4',location:'OptunaSweepPage.tsx:sync',message:'status_counter_regression_detected',data:{prev_updated_unix:prev.updatedUnix,prev_n_complete:prev.nComplete,cur_updated_unix:parsed.updated_unix,cur_n_complete:parsed.n_complete},timestamp:Date.now()})}).catch(()=>{});
        // #endregion
      }
      lastStatusRef.current = { updatedUnix: parsed.updated_unix, nComplete: parsed.n_complete }
      setData(parsed)
      if (pullWarn) {
        setErr(`Live pull warning (showing latest local status): ${pullWarn}`)
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [sweepMode])

  useEffect(() => {
    sync({ syncTrialArtifacts: false })
  }, [sync])

  useEffect(() => {
    const order = data?.meta?.sequence_order
    if (order?.length && !sequence) setSequence(order[0])
  }, [data, sequence])

  useEffect(() => {
    const id = window.setInterval(() => sync({ syncTrialArtifacts: false }), 8000)
    return () => window.clearInterval(id)
  }, [sync])

  useEffect(() => {
    if (selectedTrial === null || !sequence) {
      setMedia(null)
      setPickedClip(null)
      return
    }
    let cancelled = false
    ;(async () => {
      setMediaLoading(true)
      try {
        const r = await fetch(mediaUrl(sweepMode, selectedTrial, sequence))
        const j = (await r.json()) as MediaResponse
        if (!cancelled) {
          setMedia(j)
          const preferred =
            j.items.find((x) => x.filename.includes('phase8_final_optimized')) ||
            j.items.find((x) => x.filename.includes('phase3_tracking_bidirectional')) ||
            j.items.find((x) => x.filename.includes('01_tracks_post_stitch')) ||
            j.items.find((x) => x.kind === 'phase_preview') ||
            j.items[0] ||
            null
          setPickedClip(preferred ? preferred.path : null)
        }
      } catch {
        if (!cancelled) { setMedia(null); setPickedClip(null) }
      } finally {
        if (!cancelled) setMediaLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [selectedTrial, sequence, sweepMode])

  const sequences = data?.meta?.sequence_order ?? []
  const isV4Data = sweepMode === 'v4' || Boolean(data?.meta?.v4)
  const stageOrder = ['stage_a', 'stage_b', 'stage_c'] as const
  const hasV4Stages = useMemo(() => {
    if (!isV4Data) return false
    if (data?.meta?.has_stages === false) return false
    const raw = data?.meta?.stage_expected_counts
    if (!raw || typeof raw !== 'object') return false
    return stageOrder.some((s) => {
      const v = (raw as Record<string, unknown>)[s]
      return typeof v === 'number' && Number.isFinite(v) && v > 0
    })
  }, [isV4Data, data?.meta?.has_stages, data?.meta?.stage_expected_counts])

  const stageExpected = useMemo<Record<string, number>>(() => {
    if (!hasV4Stages) return {}
    const raw = data?.meta?.stage_expected_counts
    if (!raw || typeof raw !== 'object') {
      return { stage_a: 16, stage_b: 10, stage_c: 1 }
    }
    const out: Record<string, number> = {}
    for (const s of stageOrder) {
      const v = (raw as Record<string, unknown>)[s]
      out[s] = typeof v === 'number' && Number.isFinite(v) && v >= 0 ? Math.floor(v) : 0
    }
    return out
  }, [data?.meta?.stage_expected_counts, hasV4Stages])

  const stageCounts = useMemo(() => {
    const out: Record<string, { total: number; complete: number; fail: number }> = {}
    for (const s of stageOrder) out[s] = { total: 0, complete: 0, fail: 0 }
    for (const t of data?.trials ?? []) {
      const st = String((t.user_attrs?.stage as string) || '').toLowerCase()
      if (!out[st]) out[st] = { total: 0, complete: 0, fail: 0 }
      out[st].total += 1
      if (t.state === 'COMPLETE') out[st].complete += 1
      if (t.state === 'FAIL' || t.state === 'ERROR') out[st].fail += 1
    }
    return out
  }, [data?.trials])

  const plannedTrialsTotal = useMemo(() => {
    if (typeof data?.meta?.planned_trials_total === 'number') return data.meta.planned_trials_total
    if (!hasV4Stages) return data?.n_trials_total ?? 0
    return Object.values(stageExpected).reduce((a, b) => a + Math.max(0, b), 0)
  }, [data?.meta?.planned_trials_total, stageExpected, hasV4Stages, data?.n_trials_total])

  const plannedVideoRunsTotal = useMemo(() => {
    if (typeof data?.meta?.planned_video_runs_total === 'number') return data.meta.planned_video_runs_total
    const videosPerTrial = typeof data?.meta?.videos_per_trial === 'number'
      ? data.meta.videos_per_trial
      : (data?.meta?.sequence_order?.length ?? 0)
    return plannedTrialsTotal * videosPerTrial
  }, [data?.meta?.planned_video_runs_total, data?.meta?.videos_per_trial, data?.meta?.sequence_order?.length, plannedTrialsTotal])

  const videosPerTrialCount = useMemo(() => {
    if (typeof data?.meta?.videos_per_trial === 'number') return data.meta.videos_per_trial
    return sequences.length
  }, [data?.meta?.videos_per_trial, sequences.length])

  const avgCompletedTrialDurationSec = useMemo(() => {
    const durations = (data?.trials ?? [])
      .filter((t) => t.state === 'COMPLETE')
      .map((t) => resolveTrialWallDurationSec(t))
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0)
    return safeMean(durations)
  }, [data?.trials])

  const startedTrialsCount = useMemo(() => {
    const rows = data?.trials ?? []
    if (!rows.length) return 0
    return rows.filter((t) => String(t.state).toUpperCase() !== 'QUEUED').length
  }, [data?.trials])

  const queuedTrialsCount = useMemo(() => {
    const rows = data?.trials ?? []
    if (!rows.length) return 0
    return rows.filter((t) => String(t.state).toUpperCase() === 'QUEUED').length
  }, [data?.trials])

  const remainingTrials = useMemo(() => {
    if (!data) return null
    return Math.max(0, plannedTrialsTotal - startedTrialsCount)
  }, [data, plannedTrialsTotal, startedTrialsCount])

  const etaSec = useMemo(() => {
    if (remainingTrials == null || avgCompletedTrialDurationSec == null) return null
    return Math.max(0, remainingTrials * avgCompletedTrialDurationSec)
  }, [remainingTrials, avgCompletedTrialDurationSec])

  const failureReasonRollup = useMemo(() => {
    const counts = new Map<string, number>()
    for (const t of data?.trials ?? []) {
      if (!(t.state === 'FAIL' || t.state === 'ERROR')) continue
      const sd = t.user_attrs?.sequence_details
      if (!sd || typeof sd !== 'object') {
        counts.set('unknown_failure', (counts.get('unknown_failure') ?? 0) + 1)
        continue
      }
      let had = false
      for (const row of Object.values(sd as Record<string, unknown>)) {
        if (!row || typeof row !== 'object') continue
        const err = String((row as Record<string, unknown>).error ?? '').trim()
        if (!err) continue
        counts.set(err, (counts.get(err) ?? 0) + 1)
        had = true
      }
      if (!had) counts.set('unknown_failure', (counts.get('unknown_failure') ?? 0) + 1)
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
  }, [data?.trials])

  const cycleTrialTableSort = useCallback((col: TrialTableSortCol) => {
    setTrialTableSort((prev) => {
      if (prev.col === col) {
        return { col, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
      }
      return { col, dir: 'asc' }
    })
  }, [])

  const stageFilteredTrials = useMemo(() => {
    const t = data?.trials ?? []
    if (!isV4Data || !hasV4Stages || stageFilter === 'all') return t
    return t.filter((row) => String((row.user_attrs?.stage as string) || '').toLowerCase() === stageFilter)
  }, [data?.trials, isV4Data, hasV4Stages, stageFilter])

  const sortedTrials = useMemo(() => {
    const t = stageFilteredTrials
    const { col, dir } = trialTableSort
    return [...t].sort((a, b) => compareTrialTableRows(a, b, col, dir))
  }, [stageFilteredTrials, trialTableSort])

  const runningHint = useMemo(() => {
    const incomplete = sortedTrials.filter((x) => x.state === 'RUNNING')
    if (incomplete.length) return `Trial ${incomplete[0].number} in progress…`
    if (isV4Data && hasV4Stages && data?.meta?.stage) return `Active stage: ${String(data.meta.stage).replace('_', ' ').toUpperCase()}`
    if (isV4Data && !hasV4Stages && data?.meta?.v4_mode) return `Mode: ${String(data.meta.v4_mode)}`
    return null
  }, [sortedTrials, isV4Data, hasV4Stages, data?.meta?.stage, data?.meta?.v4_mode])

  // Infer video names from best trial user_attrs
  const videoNames = useMemo(() => {
    const allAttrs = (data?.trials ?? []).flatMap((t) => Object.keys(t.user_attrs))
    const names = new Set<string>()
    allAttrs.forEach((k) => {
      if (k.startsWith('score_')) names.add(k.replace('score_', ''))
    })
    return Array.from(names).sort()
  }, [data])

  const numericParamKeys = useMemo(() => {
    const keys = new Set<string>()
    for (const t of data?.trials ?? []) {
      if (!isChartTrial(t)) continue
      for (const [k, v] of Object.entries(t.params)) {
        if (coerceParamNumber(v) !== null) keys.add(k)
      }
    }
    return Array.from(keys).sort()
  }, [data?.trials])

  useEffect(() => {
    if (!scatterParam && numericParamKeys.length) setScatterParam(numericParamKeys[0])
  }, [scatterParam, numericParamKeys])

  useEffect(() => {
    setStageFilter('all')
  }, [sweepMode])

  const bestTrial = data?.best ?? null

  const selectedTrialRow = useMemo(() => {
    if (selectedTrial === null || !data) return null
    return data.trials.find((t) => t.number === selectedTrial) ?? null
  }, [selectedTrial, data])

  const compareDiff = useMemo(() => {
    if (!bestTrial || !selectedTrialRow) {
      return {
        paramDiffKeys: new Set<string>(),
        aggregateDiff: false,
        videoDiff: {} as Record<string, boolean>,
      }
    }
    const paramDiffKeys = diffParamKeys(bestTrial.params, selectedTrialRow.params)
    const aggregateDiff = !valuesEqual(bestTrial.value, selectedTrialRow.value)
    const videoDiff: Record<string, boolean> = {}
    for (const vn of videoNames) {
      videoDiff[vn] = !valuesEqual(
        bestTrial.user_attrs[`score_${vn}`],
        selectedTrialRow.user_attrs[`score_${vn}`],
      )
    }
    return { paramDiffKeys, aggregateDiff, videoDiff }
  }, [bestTrial, selectedTrialRow, videoNames])

  useLayoutEffect(() => {
    const el = analyticsBestColRef.current
    if (!el) return
    const update = () => setAnalyticsBestColHeight(el.getBoundingClientRect().height)
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [data?.updated_unix, bestTrial?.number, data?.n_complete])

  return (
    <div className="optuna-sweep-page optuna-sweep-page--touch">
      {/* ── Header ── */}
      <div className="optuna-sweep-head">
        <Link to="/" className="optuna-back">
          <ArrowLeft size={18} aria-hidden />
          Lab
        </Link>
        <div className="optuna-sweep-titleblock">
          <h1 className="optuna-sweep-title">
            <BarChart3 size={26} className="optuna-sweep-title-ico" aria-hidden />
            {sweepMode === 'v4' ? 'V4 sweep' : 'Optuna sweep'}
          </h1>
        </div>
        <div className="optuna-sweep-actions">
          <select
            value={sweepMode}
            onChange={(e) => setSweepMode((e.target.value as SweepMode) || 'v4')}
            title="Sweep source"
            style={{ marginRight: 8 }}
          >
            <option value="v4">V4</option>
            <option value="optuna">Optuna</option>
          </select>
          <button
            type="button"
            className="btn primary optuna-refresh"
            onClick={() => sync({ syncTrialArtifacts: false })}
            disabled={loading}
          >
            {loading ? <Loader2 className="optuna-spin" size={18} /> : <RefreshCw size={18} />}
            Refresh
          </button>
          <button
            type="button"
            className="btn optuna-refresh"
            onClick={() => sync({ syncTrialArtifacts: true })}
            disabled={loading}
            title="scp status + rsync trial_* dirs (slower; required for preview MP4s)"
          >
            {loading ? <Loader2 className="optuna-spin" size={18} /> : <Film size={18} />}
            Full sync (MP4s)
          </button>
        </div>
      </div>

      {/* ── Error ── */}
      {err ? (
        <div className="optuna-error-card">
          <h2>Could not refresh</h2>
          <p>{err}</p>
          <p className="optuna-muted">
            If <strong>scp</strong> failed, check that <code>pose-tracking.pem</code> exists under{' '}
            <code>~/Downloads</code> or <code>~/.ssh</code>, PEM permissions, and that the instance is running. If the
            file is not on the server yet, run the sweep on Lambda with{' '}
            <code>{sweepMode === 'v4' ? 'python -m tools.run_v4_lambda_sweep' : 'python -m tools.auto_sweep'}</code>. To read
            only a local JSON, set <code>SWAY_LAMBDA_SWEEP_HOST=</code> when starting the Lab API.
          </p>
        </div>
      ) : null}

      {/* ── Empty state ── */}
      {data?.empty_state ? (
        <div className="optuna-info-card" role="status">
          <h2>No sweep data (yet)</h2>
          {data.meta?.hint ? <p>{data.meta.hint}</p> : null}
          <p className="optuna-muted">
            Hit <strong>Refresh</strong> for <code>sweep_status.json</code>, or <strong>Full sync (MP4s)</strong> to also
            rsync <code>trial_*</code> from Lambda. You can also place files under{' '}
            <code>{sweepMode === 'v4' ? 'output/sweeps/v4_lambda_sweep/' : 'output/sweeps/optuna/'}</code> or set{' '}
            <code>{sweepMode === 'v4' ? 'SWAY_V4_SWEEP_DIR' : 'SWAY_OPTUNA_SWEEP_DIR'}</code> on the API process.
          </p>
        </div>
      ) : null}

      {data && !data.empty_state ? (
        <>
          {/* ── Stat cards ── */}
          <section className="optuna-summary-grid">
            <div className="optuna-stat-card">
              <span className="optuna-stat-label">Study</span>
              <span className="optuna-stat-value">{data.study_name}</span>
              <span className="optuna-stat-hint">{data.direction}</span>
            </div>
            <div className="optuna-stat-card">
              <span className="optuna-stat-label">Updated</span>
              <span className="optuna-stat-value optuna-stat-value--sm">{formatTime(data.updated_unix)}</span>
              {data.meta?.git_sha ? (
                <span className="optuna-stat-hint mono">git {data.meta.git_sha}</span>
              ) : null}
            </div>
            <div className="optuna-stat-card">
              <span className="optuna-stat-label">Progress</span>
              <span className="optuna-stat-value">
                {isV4Data && !hasV4Stages
                  ? `${data.n_complete} done · ${startedTrialsCount} started · ${queuedTrialsCount} queued`
                  : `${data.n_complete} done · ${data.n_pruned} pruned · ${data.n_trials_total} total`}
              </span>
              {runningHint ? <span className="optuna-stat-hint optuna-pulse">{runningHint}</span> : null}
            </div>
            {bestTrial ? (
              <div className="optuna-stat-card optuna-stat-card--best">
                <span className="optuna-stat-label">Best score</span>
                <span className="optuna-stat-value" style={{ color: '#a78bfa' }}>
                  {formatValue(bestTrial.value)}
                </span>
                <span className="optuna-stat-hint">Trial #{bestTrial.number}</span>
              </div>
            ) : null}
          </section>

          {isV4Data && hasV4Stages ? (
            <section className="optuna-info-card" style={{ marginBottom: '1rem' }}>
              <h2 style={{ marginTop: 0 }}>V4 stage progress</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '0.75rem' }}>
                {stageOrder.map((st) => {
                  const c = stageCounts[st] ?? { total: 0, complete: 0, fail: 0 }
                  const exp = stageExpected[st] ?? Math.max(1, c.total)
                  const pct = Math.min(100, Math.round((c.total / Math.max(1, exp)) * 100))
                  return (
                    <div key={st} className="optuna-stat-card">
                      <span className="optuna-stat-label">{st.replace('_', ' ').toUpperCase()}</span>
                      <span className="optuna-stat-value">{c.complete} complete · {c.fail} failed</span>
                      <span className="optuna-stat-hint">{c.total}/{exp} trials</span>
                      <div className="optuna-table-score" style={{ marginTop: 6 }}>
                        <div className="optuna-table-score-bar" style={{ width: `${pct}%`, background: '#38bdf8' }} />
                        <span>{pct}%</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </section>
          ) : null}

          {isV4Data && hasV4Stages ? (
            <section className="optuna-info-card" style={{ marginBottom: '1rem' }}>
              <h2 style={{ marginTop: 0 }}>Execution roadmap</h2>
              <p className="optuna-muted" style={{ marginTop: 0 }}>
                Shows what is done, what is running, and what happens next across Stage A/B/C.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '0.75rem', marginBottom: '0.75rem' }}>
                <div className="optuna-stat-card">
                  <span className="optuna-stat-label">Planned trials</span>
                  <span className="optuna-stat-value">{startedTrialsCount}/{plannedTrialsTotal}</span>
                  <span className="optuna-stat-hint">{remainingTrials ?? 0} remaining · {queuedTrialsCount} queued</span>
                </div>
                <div className="optuna-stat-card">
                  <span className="optuna-stat-label">Planned video-runs</span>
                  <span className="optuna-stat-value">
                    {(startedTrialsCount * videosPerTrialCount).toLocaleString()} / {plannedVideoRunsTotal.toLocaleString()}
                  </span>
                  <span className="optuna-stat-hint">{videosPerTrialCount} videos per trial</span>
                </div>
                <div className="optuna-stat-card">
                  <span className="optuna-stat-label">ETA</span>
                  <span className="optuna-stat-value">
                    {etaSec != null ? formatDurationSec(etaSec) : '—'}
                  </span>
                  <span className="optuna-stat-hint">
                    {avgCompletedTrialDurationSec != null
                      ? `avg ${formatDurationSec(avgCompletedTrialDurationSec)} / completed trial`
                      : 'need completed trials first'}
                  </span>
                </div>
              </div>
              <div style={{ display: 'grid', gap: '0.5rem' }}>
                {stageOrder.map((st) => {
                  const c = stageCounts[st] ?? { total: 0, complete: 0, fail: 0 }
                  const exp = Math.max(0, stageExpected[st] ?? 0)
                  const active = String(data.meta?.stage || '').toLowerCase() === st
                  const done = exp > 0 && c.total >= exp
                  const state = done ? 'completed' : active ? 'running' : c.total > 0 ? 'partial' : 'pending'
                  return (
                    <div key={st} className="optuna-table-wrap" style={{ padding: '0.5rem 0.75rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', alignItems: 'center' }}>
                        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                          <span className={`optuna-pill optuna-pill--${state === 'completed' ? 'complete' : state === 'running' ? 'running' : 'queued'}`}>
                            {state}
                          </span>
                          <strong>{st.replace('_', ' ').toUpperCase()}</strong>
                        </div>
                        <span className="optuna-muted mono">{c.total}/{exp} trials</span>
                      </div>
                    </div>
                  )
                })}
              </div>
              {failureReasonRollup.length ? (
                <div style={{ marginTop: '0.75rem' }}>
                  <strong style={{ fontSize: '0.9rem' }}>Top failure reasons</strong>
                  <ul style={{ margin: '0.4rem 0 0 1rem' }}>
                    {failureReasonRollup.map(([reason, count]) => (
                      <li key={reason} className="optuna-muted" style={{ fontSize: '0.82rem' }}>
                        {reason} ({count})
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </section>
          ) : null}

          {isV4Data && !hasV4Stages ? (
            <section className="optuna-info-card" style={{ marginBottom: '1rem' }}>
              <h2 style={{ marginTop: 0 }}>V4 full-grid progress</h2>
              <p className="optuna-muted" style={{ marginTop: 0 }}>
                All configurations run fully through the pipeline with no stage split.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '0.75rem', marginBottom: '0.75rem' }}>
                <div className="optuna-stat-card">
                  <span className="optuna-stat-label">Planned trials</span>
                  <span className="optuna-stat-value">{startedTrialsCount}/{plannedTrialsTotal}</span>
                  <span className="optuna-stat-hint">{remainingTrials ?? 0} remaining · {queuedTrialsCount} queued</span>
                </div>
                <div className="optuna-stat-card">
                  <span className="optuna-stat-label">Planned video-runs</span>
                  <span className="optuna-stat-value">
                    {(startedTrialsCount * videosPerTrialCount).toLocaleString()} / {plannedVideoRunsTotal.toLocaleString()}
                  </span>
                  <span className="optuna-stat-hint">{videosPerTrialCount} videos per trial</span>
                </div>
                <div className="optuna-stat-card">
                  <span className="optuna-stat-label">ETA</span>
                  <span className="optuna-stat-value">
                    {etaSec != null ? formatDurationSec(etaSec) : '—'}
                  </span>
                  <span className="optuna-stat-hint">
                    {avgCompletedTrialDurationSec != null
                      ? `avg ${formatDurationSec(avgCompletedTrialDurationSec)} / completed trial`
                      : 'need completed trials first'}
                  </span>
                </div>
              </div>
              {failureReasonRollup.length ? (
                <div style={{ marginTop: '0.75rem' }}>
                  <strong style={{ fontSize: '0.9rem' }}>Top failure reasons</strong>
                  <ul style={{ margin: '0.4rem 0 0 1rem' }}>
                    {failureReasonRollup.map(([reason, count]) => (
                      <li key={reason} className="optuna-muted" style={{ fontSize: '0.82rem' }}>
                        {reason} ({count})
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {data.meta?.objective_components ? (
                <div style={{ marginTop: '0.75rem' }}>
                  <strong style={{ fontSize: '0.9rem' }}>Scoring objective</strong>
                  <p className="optuna-muted" style={{ margin: '0.4rem 0 0' }}>
                    MOT identity/detection: {((Number(data.meta.objective_components.mot_identity_detection_weight) || 0) * 100).toFixed(0)}% ·
                    joint tracking quality: {((Number(data.meta.objective_components.joint_tracking_weight) || 0) * 100).toFixed(0)}%
                  </p>
                </div>
              ) : null}
            </section>
          ) : null}

          {/* ── Score progress chart ── */}
          <section className="optuna-chart-section">
            <div className="optuna-chart-section-head">
              <TrendingUp size={16} />
              <h2 className="optuna-h2" style={{ margin: 0 }}>Score progression</h2>
              <span className="optuna-chart-hint">Click a point/bar to load video proof below</span>
              <div className="optuna-chart-toolbar">
                <div className="optuna-chart-toggle">
                  <button
                    type="button"
                    className={`optuna-chart-toggle-btn${chartMode === 'bar' ? ' active' : ''}`}
                    onClick={() => setChartMode('bar')}
                    title="Bar chart"
                  >
                    <BarChart3 size={14} />
                  </button>
                  <button
                    type="button"
                    className={`optuna-chart-toggle-btn${chartMode === 'scatter' ? ' active' : ''}`}
                    onClick={() => setChartMode('scatter')}
                    title="Line + points"
                  >
                    <ScatterChart size={14} />
                  </button>
                </div>
                {chartMode === 'scatter' ? (
                  <label className="optuna-chart-check">
                    <input
                      type="checkbox"
                      checked={showRollingAvg}
                      onChange={(e) => setShowRollingAvg(e.target.checked)}
                    />
                    Rolling avg
                  </label>
                ) : null}
              </div>
            </div>
            <ScoreProgressChart
              trials={data.trials}
              bestNumber={bestTrial?.number ?? null}
              onSelectTrial={setSelectedTrial}
              selectedTrial={selectedTrial}
              chartMode={chartMode}
              direction={data.direction}
              showRollingAvg={showRollingAvg}
            />
          </section>

          {/* ── Per-video performance chart ── */}
          {videoNames.length > 0 && (
            <section className="optuna-chart-section">
              <div className="optuna-chart-section-head">
                <Target size={16} />
                <h2 className="optuna-h2" style={{ margin: 0 }}>Per-video performance</h2>
                <span className="optuna-chart-hint">Grouped bars vs trend lines</span>
                <div className="optuna-chart-toggle">
                  <button
                    type="button"
                    className={`optuna-chart-toggle-btn${perVideoMode === 'bars' ? ' active' : ''}`}
                    onClick={() => setPerVideoMode('bars')}
                    title="Grouped bars"
                  >
                    <BarChart3 size={14} />
                  </button>
                  <button
                    type="button"
                    className={`optuna-chart-toggle-btn${perVideoMode === 'lines' ? ' active' : ''}`}
                    onClick={() => setPerVideoMode('lines')}
                    title="Trend lines"
                  >
                    <TrendingUp size={14} />
                  </button>
                </div>
              </div>
              {perVideoMode === 'bars' ? (
                <PerVideoChart
                  trials={data.trials}
                  videoNames={videoNames}
                  bestNumber={bestTrial?.number ?? null}
                  onSelectTrial={setSelectedTrial}
                  selectedTrial={selectedTrial}
                />
              ) : (
                <VideoTrendChart
                  trials={data.trials}
                  videoNames={videoNames}
                  bestNumber={bestTrial?.number ?? null}
                  onSelectTrial={setSelectedTrial}
                  selectedTrial={selectedTrial}
                />
              )}
            </section>
          )}

          {/* ── Trial duration ↔ Parameter vs score (full width, toggle) ── */}
          <section className="optuna-chart-section optuna-chart-section--duration-param">
            {durationVsParamView === 'duration' ? (
              <div className="optuna-chart-section-head">
                <Timer size={16} />
                <h2 className="optuna-h2" style={{ margin: 0 }}>Trial duration</h2>
                <span className="optuna-chart-hint">Click a bar to select that trial · source details below the chart</span>
                <DurationParamViewToggle view={durationVsParamView} onViewChange={setDurationVsParamView} />
              </div>
            ) : (
              <>
                <div className="optuna-chart-section-head optuna-chart-section-head--scatter-param">
                  <div className="optuna-chart-section-head-row">
                    <Sparkles size={16} />
                    <h2 className="optuna-h2" style={{ margin: 0 }}>Parameter vs score</h2>
                  </div>
                  <DurationParamViewToggle view={durationVsParamView} onViewChange={setDurationVsParamView} />
                </div>
                <div className="optuna-chart-section-subhead optuna-chart-section-subhead--scatter-param">
                  <span className="optuna-chart-hint">
                    {isMaximizeStudy(data.direction) ? 'Higher score is better' : 'Lower score is better'}
                    {' · '}
                    pick a param, then click a point to compare trials and open video proof
                  </span>
                  {numericParamKeys.length > 0 ? (
                    <select
                      className="optuna-select optuna-select--chart"
                      value={scatterParam}
                      onChange={(e) => setScatterParam(e.target.value)}
                      aria-label="Parameter for scatter plot"
                    >
                      {numericParamKeys.map((k) => (
                        <option key={k} value={k}>
                          {k}
                        </option>
                      ))}
                    </select>
                  ) : null}
                </div>
              </>
            )}
            {durationVsParamView === 'duration' ? (
              <TrialDurationChart
                trials={data.trials}
                onSelectTrial={setSelectedTrial}
                selectedTrial={selectedTrial}
              />
            ) : (
              <ParamVsScoreScatter
                trials={data.trials}
                paramKey={scatterParam}
                direction={data.direction}
                bestNumber={bestTrial?.number ?? null}
                onSelectTrial={setSelectedTrial}
                selectedTrial={selectedTrial}
              />
            )}
          </section>

          {/* ── Optuna direction panel (full width) ── */}
          {data.trials.filter(isChartTrial).length >= 2 && (
            <OptunaDirectionPanel
              trials={data.trials}
              bestParams={bestTrial?.params ?? {}}
              studyDirection={data.direction}
            />
          )}

          {/* ── Best trial (+ selected compare) + Insights ── */}
          <div
            className={`optuna-analytics-row${bestTrial && selectedTrialRow ? ' optuna-analytics-row--compare' : ''}`}
          >
            <div ref={analyticsBestColRef} className="optuna-analytics-row__col">
              <div
                className={`optuna-compare-cards${bestTrial && selectedTrialRow ? ' optuna-compare-cards--split' : ''}`}
              >
                {bestTrial ? (
                  <TrialConfigCard
                    variant="best"
                    model={bestToModel(bestTrial)}
                    heading="Best trial"
                    paramDiffKeys={selectedTrialRow ? compareDiff.paramDiffKeys : new Set()}
                    aggregateDiff={selectedTrialRow ? compareDiff.aggregateDiff : false}
                    videoDiff={selectedTrialRow ? compareDiff.videoDiff : {}}
                    videoNames={videoNames}
                    onOpenVideo={() => setSelectedTrial(bestTrial.number)}
                  />
                ) : (
                  <section className="optuna-empty-best">
                    <p className="optuna-muted">No completed trials yet — best will appear here.</p>
                  </section>
                )}
                {bestTrial && selectedTrialRow ? (
                  <TrialConfigCard
                    variant="compare"
                    model={trialRowToModel(selectedTrialRow)}
                    heading="Selected trial"
                    sameAsBest={selectedTrialRow.number === bestTrial.number}
                    paramDiffKeys={compareDiff.paramDiffKeys}
                    aggregateDiff={compareDiff.aggregateDiff}
                    videoDiff={compareDiff.videoDiff}
                    videoNames={videoNames}
                    onOpenVideo={() => setSelectedTrial(selectedTrialRow.number)}
                    onClearSelection={() => setSelectedTrial(null)}
                  />
                ) : null}
              </div>
              {bestTrial && selectedTrialRow ? (
                <p className="optuna-compare-diff-legend" role="note">
                  Amber outline: differs from the other trial (params, aggregate, or per-video scores).
                </p>
              ) : null}
            </div>

            <InsightPanel trials={data.trials} columnHeightPx={analyticsBestColHeight} />
          </div>

          {/* ── All trials + Video proof ── */}
          <section className="optuna-split">
            <div className="optuna-table-panel">
              <h2 className="optuna-h2">All trials</h2>
              {isV4Data && hasV4Stages ? (
                <div style={{ marginBottom: '0.5rem' }}>
                  <label className="optuna-field">
                    <span>Stage filter</span>
                    <select className="optuna-select" value={stageFilter} onChange={(e) => setStageFilter(e.target.value)}>
                      <option value="all">all stages</option>
                      {stageOrder.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </label>
                </div>
              ) : null}
              <p className="optuna-table-scroll-hint" role="note">
                On a narrow screen, swipe the table horizontally to see every column (trial # stays pinned).
              </p>
              <div className="optuna-table-wrap">
                <table className="optuna-table">
                  <thead>
                    <tr>
                      <SortableTrialTh
                        col="trial"
                        label="#"
                        active={trialTableSort.col === 'trial'}
                        dir={trialTableSort.dir}
                        onSort={cycleTrialTableSort}
                      />
                      <SortableTrialTh
                        col="state"
                        label="State"
                        active={trialTableSort.col === 'state'}
                        dir={trialTableSort.dir}
                        onSort={cycleTrialTableSort}
                      />
                      {isV4Data && hasV4Stages ? <th scope="col">Stage</th> : null}
                      <SortableTrialTh
                        col="score"
                        label="Score"
                        active={trialTableSort.col === 'score'}
                        dir={trialTableSort.dir}
                        onSort={cycleTrialTableSort}
                      />
                      <th scope="col">Progress</th>
                      <th scope="col">Config diff vs best</th>
                      {videoNames.map((vn) => (
                        <SortableTrialTh
                          key={vn}
                          col={vn}
                          label={vn}
                          style={{ color: VIDEO_COLORS[vn] ?? undefined }}
                          active={trialTableSort.col === vn}
                          dir={trialTableSort.dir}
                          onSort={cycleTrialTableSort}
                        />
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedTrials.map((t) => {
                      const isBest = t.number === (bestTrial?.number ?? -1)
                      const cfgDiff = trialConfigDiffSummary(t, bestTrial)
                      const progress = trialProgressSummary(t)
                      return (
                        <tr
                          key={t.number}
                          className={selectedTrial === t.number ? 'optuna-row--active' : isBest ? 'optuna-row--best' : undefined}
                          onClick={() => setSelectedTrial(t.number)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              setSelectedTrial(t.number)
                            }
                          }}
                          tabIndex={0}
                          role="button"
                        >
                          <td className="mono">
                            {t.number}
                            {isBest && <span className="optuna-best-badge">★</span>}
                          </td>
                          <td>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                              <span className={`optuna-pill optuna-pill--${t.state.toLowerCase()}`}>{t.state}</span>
                              {t.state === 'PRUNED' && (() => {
                                const reason = getPruneReason(t)
                                return reason ? (
                                  <span className="optuna-prune-reason" title={reason}>
                                    ✂ {reason.length > 42 ? reason.slice(0, 42) + '…' : reason}
                                  </span>
                                ) : null
                              })()}
                            </div>
                          </td>
                          {isV4Data && hasV4Stages ? (
                            <td className="mono">{String((t.user_attrs?.stage as string) || '—')}</td>
                          ) : null}
                          <td>
                            <div className="optuna-table-score">
                              {typeof t.value === 'number' ? (
                                <>
                                  <div
                                    className="optuna-table-score-bar"
                                    style={{
                                      width: `${Math.min((t.value ?? 0) * 100, 100)}%`,
                                      background: scoreColorDirected(t.value, data.direction),
                                    }}
                                  />
                                  <span>{formatValue(t.value)}</span>
                                </>
                              ) : '—'}
                            </div>
                          </td>
                          <td>
                            <div className="optuna-table-config-diff">
                              <span className={`optuna-pill optuna-pill--${progress.tone}`}>
                                {progress.label}
                              </span>
                              <span className="optuna-muted optuna-table-config-diff-text" title={progress.detail}>
                                {progress.detail}
                              </span>
                            </div>
                          </td>
                          <td>
                            <div className="optuna-table-config-diff">
                              <span className={`optuna-pill ${cfgDiff.count === 0 ? 'optuna-pill--complete' : 'optuna-pill--running'}`}>
                                {cfgDiff.count === 0 ? 'baseline' : `${cfgDiff.count} change${cfgDiff.count === 1 ? '' : 's'}`}
                              </span>
                              {cfgDiff.count > 0 ? (
                                <span className="optuna-muted optuna-table-config-diff-text" title={cfgDiff.rows.join(' | ')}>
                                  {cfgDiff.rows.join(' • ')}
                                </span>
                              ) : null}
                            </div>
                          </td>
                          {videoNames.map((vn) => {
                            const val = t.user_attrs[`score_${vn}`]
                            if (typeof val !== 'number') return <td key={vn} className="optuna-muted">—</td>
                            return (
                              <td key={vn}>
                                <div className="optuna-table-dot-wrap" title={val.toFixed(4)}>
                                  <div
                                    className="optuna-table-dot"
                                    style={{
                                      background: VIDEO_COLORS[vn] ?? '#94a3b8',
                                      opacity: 0.35 + val * 0.65,
                                      width: `${8 + val * 12}px`,
                                      height: `${8 + val * 12}px`,
                                    }}
                                  />
                                  <span className="optuna-table-dot-val">{val.toFixed(3)}</span>
                                </div>
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Video proof panel */}
            <div className="optuna-video-panel">
              <h2 className="optuna-h2">
                <Film size={20} aria-hidden style={{ verticalAlign: 'middle', marginRight: 8 }} />
                {proofMode === 'video' ? 'Video proof' : 'Progress check'}
              </h2>
              <div className="optuna-video-controls" style={{ marginBottom: '0.5rem' }}>
                <label className="optuna-field">
                  <span>Panel mode</span>
                  <select className="optuna-select" value={proofMode} onChange={(e) => setProofMode((e.target.value as 'video' | 'progress') || 'video')}>
                    <option value="video">Video proof</option>
                    <option value="progress">Progress check</option>
                  </select>
                </label>
              </div>
              {proofMode === 'video' ? (
                <p className="optuna-muted optuna-video-help">
                  Select a trial row or click a chart bar, then pick a benchmark sequence. Clips are read from{' '}
                  <strong>this Mac&apos;s</strong> sweep tree (same as JSON path). Remote-only previews require
                  rsync/scp of <code>trial_*</code> dirs.
                </p>
              ) : (
                <p className="optuna-muted optuna-video-help">
                  Live trial monitor showing active video, active phase, and phase/overall percent progress.
                </p>
              )}

              <div className="optuna-video-controls">
                <label className="optuna-field">
                  <span>Sequence</span>
                  <select
                    className="optuna-select"
                    value={sequence}
                    onChange={(e) => setSequence(e.target.value)}
                    disabled={!sequences.length}
                  >
                    {sequences.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </label>
                <label className="optuna-field optuna-field--trial">
                  <span>Trial</span>
                  <div className="optuna-trial-input-row">
                    <input
                      className="optuna-input mono"
                      type="number"
                      min={0}
                      value={selectedTrial ?? ''}
                      placeholder="#"
                      onChange={(e) => {
                        const v = e.target.value
                        setSelectedTrial(v === '' ? null : parseInt(v, 10))
                      }}
                      aria-label="Selected trial number"
                    />
                    {selectedTrial !== null ? (
                      <button
                        type="button"
                        className="optuna-trial-deselect optuna-trial-deselect--field"
                        onClick={() => setSelectedTrial(null)}
                        aria-label="Clear selected trial"
                        title="Clear selection"
                      >
                        <X size={16} aria-hidden />
                      </button>
                    ) : null}
                  </div>
                </label>
              </div>

              {/* Per-video scores for selected trial */}
              {proofMode === 'video' && selectedTrial !== null && (() => {
                const t = data.trials.find((x) => x.number === selectedTrial)
                if (!t || !videoNames.length) return null
                return (
                  <div className="optuna-selected-trial-scores">
                    <span className="optuna-h3" style={{ marginBottom: '0.5rem', display: 'block' }}>
                      Trial #{selectedTrial} scores
                      {isV4Data && hasV4Stages ? (
                        <span className="optuna-pill optuna-pill--complete" style={{ marginLeft: 8 }}>
                          {String((t.user_attrs?.stage as string) || 'n/a')}
                        </span>
                      ) : null}
                    </span>
                    {videoNames.map((vn) => {
                      const val = t.user_attrs[`score_${vn}`]
                      if (typeof val !== 'number') return null
                      return <ScoreBar key={vn} label={vn} value={val} color={VIDEO_COLORS[vn] ?? '#94a3b8'} />
                    })}
                    {typeof t.value === 'number' && (
                      <ScoreBar label="aggregate" value={t.value} color="#a78bfa" />
                    )}
                  </div>
                )
              })()}

              {proofMode === 'video' && mediaLoading ? (
                <div className="optuna-video-loading">
                  <Loader2 className="optuna-spin" size={28} />
                  <span>Loading clips…</span>
                </div>
              ) : null}

              {proofMode === 'video' && selectedTrial !== null && media && !media.items.length ? (
                <div className="optuna-video-empty-card">
                  <div className="optuna-video-empty-icon">
                    <VideoOff size={28} aria-hidden />
                  </div>
                  <div className="optuna-video-empty-body">
                    <p className="optuna-video-empty-title">
                      No clips for Trial #{selectedTrial} / {sequence}
                    </p>
                    {media.note ? (
                      <p className="optuna-muted" style={{ margin: '0.25rem 0 0', fontSize: '0.8rem' }}>{media.note}</p>
                    ) : (
                      <p className="optuna-muted" style={{ margin: '0.25rem 0 0', fontSize: '0.8rem' }}>
                        Phase preview MP4s haven&apos;t been synced from Lambda yet, or this trial
                        was run without <code>--phase-previews</code>.
                      </p>
                    )}
                    <button
                      type="button"
                      className="btn btn--compact optuna-video-sync-btn"
                      onClick={() => sync({ syncTrialArtifacts: true })}
                      disabled={loading}
                      style={{ marginTop: '0.75rem' }}
                    >
                      {loading ? <Loader2 className="optuna-spin" size={14} /> : <Film size={14} />}
                      Full sync (rsync MP4s from Lambda)
                    </button>
                  </div>
                </div>
              ) : null}

              {proofMode === 'video' && selectedTrial !== null && media && media.items.length > 0 ? (
                <>
                  <label className="optuna-field optuna-field--full">
                    <span>Clip</span>
                    <select
                      className="optuna-select"
                      value={pickedClip ?? ''}
                      onChange={(e) => setPickedClip(e.target.value || null)}
                    >
                      {media.items.map((it) => (
                        <option key={it.path} value={it.path}>
                          [{it.kind}] {it.filename}
                        </option>
                      ))}
                    </select>
                  </label>
                  {pickedClip ? (
                    <video
                      className="optuna-video"
                      key={`${selectedTrial}-${sequence}-${pickedClip}`}
                      controls
                      playsInline
                      preload="metadata"
                      src={fileUrl(sweepMode, selectedTrial, sequence, pickedClip)}
                    />
                  ) : null}
                </>
              ) : null}

              {proofMode === 'video' && selectedTrial === null ? (
                <div className="optuna-video-placeholder">
                  <p className="optuna-muted">Click a trial in the table or a chart bar to load previews.</p>
                </div>
              ) : null}

              {proofMode === 'progress' ? (
                <div className="optuna-selected-trial-scores">
                  {selectedTrial === null ? (
                    <p className="optuna-muted" style={{ margin: 0 }}>Select a trial to monitor progress.</p>
                  ) : (() => {
                    const t = data.trials.find((x) => x.number === selectedTrial)
                    if (!t) return <p className="optuna-muted" style={{ margin: 0 }}>Selected trial not found.</p>
                    const ua = t.user_attrs ?? {}
                    const curVideo = String(ua.progress_current_video ?? '').trim() || 'pending'
                    const curPhase = phaseLabel(ua.progress_current_phase)
                    const nums = trialProgressNumbers(t)
                    const transitionLikely =
                      t.state === 'RUNNING' &&
                      nums.phasePct != null &&
                      nums.phasePct >= 99 &&
                      !['done', 'failed', 'phase8_final'].includes(curPhase.toLowerCase())
                    return (
                      <div>
                        <span className="optuna-h3" style={{ marginBottom: '0.5rem', display: 'block' }}>
                          Trial #{selectedTrial} live status
                        </span>
                        <p className="optuna-muted" style={{ margin: '0 0 0.5rem' }}>
                          Selected trial: <strong>#{selectedTrial}</strong> · State: <strong>{t.state}</strong>
                        </p>
                        <p className="optuna-muted" style={{ margin: '0 0 0.5rem' }}>
                          Current video: <strong>{curVideo}</strong> ({nums.videoIndex}/{nums.videoTotal || 0})
                        </p>
                        <p className="optuna-muted" style={{ margin: '0 0 0.6rem' }}>
                          Current pipeline phase: <strong>{curPhase}</strong>
                        </p>
                        <div className="optuna-table-score" style={{ marginBottom: '0.35rem' }}>
                          <div className="optuna-table-score-bar" style={{ width: `${nums.phasePct ?? 0}%`, background: '#22c55e' }} />
                          <span>
                            Pipeline phase progress: {nums.phasePct != null ? `${nums.phasePct.toFixed(1)}%` : '—'}
                            {transitionLikely ? ' (transitioning)' : ''}
                          </span>
                        </div>
                        <div className="optuna-table-score">
                          <div className="optuna-table-score-bar" style={{ width: `${nums.overallPct ?? 0}%`, background: '#38bdf8' }} />
                          <span>
                            Video progress: {nums.videoIndex}/{nums.videoTotal || 0}
                            {nums.overallPct != null ? ` (${nums.overallPct.toFixed(1)}%)` : ''}
                          </span>
                        </div>
                      </div>
                    )
                  })()}
                  <div style={{ marginTop: '0.75rem' }}>
                    <strong style={{ fontSize: '0.9rem' }}>Running trials</strong>
                    <ul style={{ margin: '0.4rem 0 0 1rem' }}>
                      {data.trials
                        .filter((r) => r.state === 'RUNNING')
                        .slice(0, 8)
                        .map((r) => {
                          const ua = r.user_attrs ?? {}
                          const curVideo = String(ua.progress_current_video ?? '').trim() || 'pending'
                          const curPhase = phaseLabel(ua.progress_current_phase)
                          const nums = trialProgressNumbers(r)
                          return (
                            <li key={r.number} className="optuna-muted" style={{ fontSize: '0.82rem' }}>
                              Trial {r.number}{r.number === selectedTrial ? ' (selected)' : ''}: {curVideo} · {curPhase}
                              {nums.phasePct != null ? ` · ${nums.phasePct.toFixed(1)}%` : ''}
                            </li>
                          )
                        })}
                    </ul>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        </>
      ) : null}
    </div>
  )
}
