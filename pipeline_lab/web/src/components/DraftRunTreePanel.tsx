import { useCallback, useLayoutEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import type { DraftParentRef, DraftRun } from '../context/LabContext'
import { parentRefSummary, treeSegmentLabel } from '../lib/runTreeDraft'
import {
  Pencil,
  Copy,
  Trash2,
  CornerDownRight,
  ArrowRight,
  Users,
  GitBranch,
  ChevronDown,
  MoreHorizontal,
} from 'lucide-react'

const MAX_TREE_LEVEL = 4
const COL_GAP = 28
const NODE_MIN_W = 132
const NODE_MAX_W = 168

const SEGMENT_BADGE: Record<number, { bg: string; border: string; color: string; short: string }> = {
  0: { bg: 'rgba(14, 165, 233, 0.14)', border: 'rgba(56, 189, 248, 0.45)', color: '#7dd3fc', short: 'P1' },
  1: { bg: 'rgba(167, 139, 250, 0.14)', border: 'rgba(196, 181, 253, 0.4)', color: '#c4b5fd', short: 'P2' },
  2: { bg: 'rgba(52, 211, 153, 0.12)', border: 'rgba(110, 231, 183, 0.4)', color: '#6ee7b7', short: 'P3' },
  3: { bg: 'rgba(251, 191, 36, 0.1)', border: 'rgba(252, 211, 77, 0.45)', color: '#fcd34d', short: 'P4' },
  4: { bg: 'rgba(248, 250, 252, 0.08)', border: 'rgba(226, 232, 240, 0.35)', color: '#e2e8f0', short: 'P5' },
}

function phaseBadge(level: number, compact?: boolean) {
  const b = SEGMENT_BADGE[Math.min(level, 4)] ?? SEGMENT_BADGE[4]
  return (
    <span
      title={treeSegmentLabel(level)}
      style={{
        fontSize: compact ? '0.58rem' : '0.65rem',
        fontWeight: 800,
        letterSpacing: '0.06em',
        padding: compact ? '0.12rem 0.32rem' : '0.2rem 0.45rem',
        borderRadius: 5,
        border: `1px solid ${b.border}`,
        background: b.bg,
        color: b.color,
        flexShrink: 0,
      }}
    >
      {b.short}
    </span>
  )
}

function isFanOutDraft(d: DraftRun) {
  return d.parentRef.kind === 'all_roots' || d.parentRef.kind === 'all_at_level'
}

function newClientId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `run-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

function collectSingleParentSubtreeIds(targetId: string, all: DraftRun[]): Set<string> {
  const acc = new Set<string>([targetId])
  let growing = true
  while (growing) {
    growing = false
    for (const d of all) {
      if (acc.has(d.clientId)) continue
      if (d.parentRef.kind === 'single' && acc.has(d.parentRef.parentClientId)) {
        acc.add(d.clientId)
        growing = true
      }
    }
  }
  return acc
}

function rootsOf(drafts: DraftRun[]) {
  return drafts.filter((d) => d.parentRef.kind === 'none')
}

function fanOutNodes(drafts: DraftRun[]) {
  return drafts.filter((d) => d.parentRef.kind === 'all_roots' || d.parentRef.kind === 'all_at_level')
}

function singleParentChildren(parentId: string, drafts: DraftRun[]): DraftRun[] {
  return drafts
    .filter((d) => d.parentRef.kind === 'single' && d.parentRef.parentClientId === parentId)
    .sort((a, b) => drafts.indexOf(a) - drafts.indexOf(b))
}

type GraphEdge = { from: string; to: string }

/** Column order (top → bottom) per phase + edges for connectors. */
function buildColumnsAndEdges(drafts: DraftRun[]): { columns: DraftRun[][]; edges: GraphEdge[] } {
  const roots = rootsOf(drafts)
  const seen = new Set<string>()
  const columns: DraftRun[][] = Array.from({ length: MAX_TREE_LEVEL + 1 }, () => [])
  const edges: GraphEdge[] = []

  const pushCol = (d: DraftRun) => {
    const L = d.treeLevel
    if (L < 0 || L > MAX_TREE_LEVEL) return
    if (seen.has(d.clientId)) return
    seen.add(d.clientId)
    columns[L].push(d)
  }

  function visit(d: DraftRun) {
    if (seen.has(d.clientId)) return
    pushCol(d)
    for (const c of singleParentChildren(d.clientId, drafts)) {
      edges.push({ from: d.clientId, to: c.clientId })
      visit(c)
    }
  }

  for (const r of roots) {
    visit(r)
  }

  const fans = fanOutNodes(drafts)
  for (const f of fans) {
    if (!seen.has(f.clientId)) {
      pushCol(f)
    }
    if (f.parentRef.kind === 'all_roots') {
      for (const r of roots) {
        edges.push({ from: r.clientId, to: f.clientId })
      }
    } else if (f.parentRef.kind === 'all_at_level') {
      const lvl = f.parentRef.level
      const parents = drafts.filter((d) => d.treeLevel === lvl && !isFanOutDraft(d))
      for (const p of parents) {
        edges.push({ from: p.clientId, to: f.clientId })
      }
    }
    for (const c of singleParentChildren(f.clientId, drafts)) {
      if (!seen.has(c.clientId)) {
        edges.push({ from: f.clientId, to: c.clientId })
        visit(c)
      }
    }
  }

  return { columns, edges }
}

function CompactNodeCard({
  d,
  drafts,
  ctx,
}: {
  d: DraftRun
  drafts: DraftRun[]
  ctx: {
    onEdit: (id: string) => void
    onDuplicate: (id: string) => void
    onRemove: (id: string) => void
    onAddSameSegment: (anchor: DraftRun) => void
    onAddNextSingle: (anchor: DraftRun) => void
    onAddNextAllRoots: () => void
    onAddNextAllAtLevel: (anchor: DraftRun) => void
    removeDisabled: boolean
  }
}) {
  const isFanOut = d.parentRef.kind === 'all_roots' || d.parentRef.kind === 'all_at_level'
  const roots = rootsOf(drafts)
  const showAllRoots = !isFanOut && roots.length > 1 && d.treeLevel === 0 && d.treeLevel < MAX_TREE_LEVEL
  const peersAtLevel = drafts.filter((x) => x.treeLevel === d.treeLevel && !isFanOutDraft(x))
  const showAllAtLevel =
    !isFanOut && peersAtLevel.length > 1 && d.treeLevel > 0 && d.treeLevel < MAX_TREE_LEVEL

  return (
    <div
      data-tree-node={d.clientId}
      className="glass-panel sway-draft-tree-node"
      style={{
        padding: '0.42rem 0.5rem',
        marginBottom: 0,
        border: `1px solid ${isFanOut ? 'rgba(167, 139, 250, 0.32)' : 'rgba(148, 163, 184, 0.22)'}`,
        borderRadius: 10,
        background: isFanOut ? 'rgba(88, 28, 135, 0.12)' : 'rgba(15, 23, 42, 0.65)',
        minWidth: NODE_MIN_W,
        maxWidth: NODE_MAX_W,
        boxSizing: 'border-box',
        position: 'relative',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '0.35rem' }}>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap', marginBottom: '0.2rem' }}>
            {phaseBadge(d.treeLevel, true)}
            {isFanOut && (
              <span
                style={{
                  fontSize: '0.52rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  color: '#c4b5fd',
                  padding: '0.1rem 0.28rem',
                  borderRadius: 4,
                  border: '1px solid rgba(167, 139, 250, 0.35)',
                  background: 'rgba(167, 139, 250, 0.08)',
                }}
              >
                Fan
              </span>
            )}
          </div>
          <div
            style={{
              fontWeight: 600,
              color: '#f1f5f9',
              fontSize: '0.72rem',
              lineHeight: 1.3,
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
              wordBreak: 'break-word',
            }}
          >
            {d.recipeName}
          </div>
          <div style={{ fontSize: '0.58rem', color: 'var(--text-muted)', marginTop: '0.18rem', lineHeight: 1.35, opacity: 0.9 }}>
            {parentRefSummary(d, drafts)}
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', flexShrink: 0 }}>
          <button
            type="button"
            className="btn"
            style={{ padding: '0.22rem', minWidth: 0, borderRadius: 6 }}
            title="Edit"
            onClick={() => ctx.onEdit(d.clientId)}
          >
            <Pencil size={12} aria-hidden />
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: '0.22rem', minWidth: 0, borderRadius: 6 }}
            title="Duplicate"
            onClick={() => ctx.onDuplicate(d.clientId)}
          >
            <Copy size={12} aria-hidden />
          </button>
          <button
            type="button"
            className="btn"
            style={{ padding: '0.22rem', minWidth: 0, borderRadius: 6, color: '#f87171', borderColor: 'rgba(248,113,113,0.3)' }}
            title="Remove"
            disabled={ctx.removeDisabled}
            onClick={() => ctx.onRemove(d.clientId)}
          >
            <Trash2 size={12} aria-hidden />
          </button>
        </div>
      </div>

      <details style={{ marginTop: '0.38rem' }}>
        <summary
          style={{
            cursor: 'pointer',
            listStyle: 'none',
            display: 'flex',
            alignItems: 'center',
            gap: '0.25rem',
            fontSize: '0.58rem',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            color: 'var(--text-muted)',
          }}
        >
          <MoreHorizontal size={11} aria-hidden />
          Plan
        </summary>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.28rem', marginTop: '0.35rem', paddingTop: '0.35rem', borderTop: '1px solid rgba(148,163,184,0.1)' }}>
          <button type="button" className="btn" style={{ padding: '0.28rem 0.4rem', fontSize: '0.62rem' }} onClick={() => ctx.onAddSameSegment(d)}>
            <CornerDownRight size={11} /> Sibling
          </button>
          {d.treeLevel < MAX_TREE_LEVEL && !isFanOut ? (
            <>
              <button
                type="button"
                className="btn"
                style={{
                  padding: '0.28rem 0.4rem',
                  fontSize: '0.62rem',
                  borderColor: 'rgba(34, 211, 238, 0.3)',
                  background: 'rgba(34, 211, 238, 0.06)',
                }}
                onClick={() => ctx.onAddNextSingle(d)}
              >
                <ArrowRight size={11} /> Next branch
              </button>
              {showAllRoots && (
                <button type="button" className="btn" style={{ padding: '0.28rem 0.4rem', fontSize: '0.62rem' }} onClick={() => ctx.onAddNextAllRoots()}>
                  <Users size={11} /> All P1
                </button>
              )}
              {showAllAtLevel && (
                <button type="button" className="btn" style={{ padding: '0.28rem 0.4rem', fontSize: '0.62rem' }} onClick={() => ctx.onAddNextAllAtLevel(d)}>
                  <Users size={11} /> All here
                </button>
              )}
            </>
          ) : isFanOut ? (
            <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
              Continues from multiple parents. Add next segment from a single-parent node.
            </span>
          ) : (
            <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)' }}>Leaf — runs through export.</span>
          )}
        </div>
      </details>
    </div>
  )
}

function TreeConnectorSvg({
  width,
  height,
  segments,
}: {
  width: number
  height: number
  segments: Array<{ x1: number; y1: number; x2: number; y2: number; dashed?: boolean }>
}) {
  if (width <= 0 || height <= 0 || segments.length === 0) return null
  return (
    <svg
      width={width}
      height={height}
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
        pointerEvents: 'none',
        overflow: 'visible',
      }}
      aria-hidden
    >
      {segments.map((s, i) => {
        const midX = s.x1 + (s.x2 - s.x1) * 0.55
        const d = `M ${s.x1} ${s.y1} C ${midX} ${s.y1}, ${midX} ${s.y2}, ${s.x2} ${s.y2}`
        return (
          <path
            key={i}
            d={d}
            fill="none"
            stroke="rgba(34, 211, 238, 0.35)"
            strokeWidth={1.25}
            strokeDasharray={s.dashed ? '4 3' : undefined}
            vectorEffect="non-scaling-stroke"
          />
        )
      })}
    </svg>
  )
}

export function DraftRunTreePanel({
  drafts,
  setDrafts,
  addDraft,
  duplicateDraft,
  setEditingId,
  totalDraftCount,
}: {
  drafts: DraftRun[]
  setDrafts: Dispatch<SetStateAction<DraftRun[]>>
  addDraft: (initial?: Partial<DraftRun>) => void
  duplicateDraft: (clientId: string) => void
  setEditingId: (id: string | null) => void
  totalDraftCount: number
}) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLDivElement>(null)
  const [lineSegments, setLineSegments] = useState<Array<{ x1: number; y1: number; x2: number; y2: number; dashed?: boolean }>>([])
  const [svgSize, setSvgSize] = useState({ w: 0, h: 0 })

  const { columns, edges } = useMemo(() => buildColumnsAndEdges(drafts), [drafts])

  const nonEmptyColIndices = useMemo(() => columns.map((c, i) => (c.length > 0 ? i : -1)).filter((i) => i >= 0), [columns])

  const removeNode = useCallback(
    (clientId: string) => {
      if (totalDraftCount <= 1) return
      const drop = collectSingleParentSubtreeIds(clientId, drafts)
      setDrafts((prev) => prev.filter((d) => !drop.has(d.clientId)))
    },
    [drafts, setDrafts, totalDraftCount],
  )

  const removeDisabled = totalDraftCount <= 1

  const onAddSameSegment = useCallback(
    (anchor: DraftRun) => {
      if (anchor.parentRef.kind === 'none') {
        addDraft({ treeLevel: 0, parentRef: { kind: 'none' } })
        return
      }
      const ref: DraftParentRef = anchor.parentRef
      const n = drafts.filter((x) => x.parentRef.kind === 'none').length + 1
      setDrafts((prev) => [
        ...prev,
        {
          clientId: newClientId(),
          recipeName: `Same segment ${n}`,
          fields: {},
          treeLevel: anchor.treeLevel,
          parentRef: ref,
        },
      ])
    },
    [addDraft, drafts, setDrafts],
  )

  const onAddNextSingle = useCallback(
    (anchor: DraftRun) => {
      const nextLevel = Math.min(anchor.treeLevel + 1, MAX_TREE_LEVEL)
      const n = drafts.length + 1
      setDrafts((prev) => [
        ...prev,
        {
          clientId: newClientId(),
          recipeName: `Segment ${nextLevel + 1} (${n})`,
          fields: {},
          treeLevel: nextLevel,
          parentRef: { kind: 'single', parentClientId: anchor.clientId },
        },
      ])
    },
    [drafts.length, setDrafts],
  )

  const onAddNextAllRoots = useCallback(() => {
    const nextLevel = 1
    const n = drafts.length + 1
    setDrafts((prev) => [
      ...prev,
      {
        clientId: newClientId(),
        recipeName: `Segment 2 — all Phase-1 roots (${n})`,
        fields: {},
        treeLevel: nextLevel,
        parentRef: { kind: 'all_roots' },
      },
    ])
  }, [drafts.length, setDrafts])

  const onAddNextAllAtLevel = useCallback(
    (anchor: DraftRun) => {
      const nextLevel = Math.min(anchor.treeLevel + 1, MAX_TREE_LEVEL)
      const n = drafts.length + 1
      setDrafts((prev) => [
        ...prev,
        {
          clientId: newClientId(),
          recipeName: `Segment ${nextLevel + 1} — all at L${anchor.treeLevel + 1} (${n})`,
          fields: {},
          treeLevel: nextLevel,
          parentRef: { kind: 'all_at_level', level: anchor.treeLevel },
        },
      ])
    },
    [drafts.length, setDrafts],
  )

  const ctx = useMemo(
    () => ({
      onEdit: (id: string) => setEditingId(id),
      onDuplicate: (id: string) => duplicateDraft(id),
      onRemove: removeNode,
      onAddSameSegment,
      onAddNextSingle,
      onAddNextAllRoots,
      onAddNextAllAtLevel,
      removeDisabled,
    }),
    [
      duplicateDraft,
      onAddNextAllAtLevel,
      onAddNextAllRoots,
      onAddNextSingle,
      onAddSameSegment,
      removeDisabled,
      removeNode,
      setEditingId,
    ],
  )

  const relayoutLines = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const r = canvas.getBoundingClientRect()
    if (r.width < 10 || r.height < 10) return

    const segs: Array<{ x1: number; y1: number; x2: number; y2: number; dashed?: boolean }> = []
    const fanIds = new Set(fanOutNodes(drafts).map((d) => d.clientId))

    for (const e of edges) {
      const a = canvas.querySelector(`[data-tree-node="${CSS.escape(e.from)}"]`)
      const b = canvas.querySelector(`[data-tree-node="${CSS.escape(e.to)}"]`)
      if (!(a instanceof HTMLElement) || !(b instanceof HTMLElement)) continue
      const ar = a.getBoundingClientRect()
      const br = b.getBoundingClientRect()
      const x1 = ar.right - r.left
      const y1 = ar.top - r.top + ar.height / 2
      const x2 = br.left - r.left
      const y2 = br.top - r.top + br.height / 2
      const dashed = fanIds.has(e.to) || edges.filter((x) => x.to === e.to).length > 1
      segs.push({ x1, y1, x2, y2, dashed })
    }

    setSvgSize({ w: r.width, h: r.height })
    setLineSegments(segs)
  }, [drafts, edges])

  useLayoutEffect(() => {
    relayoutLines()
  }, [relayoutLines, columns, drafts])

  useLayoutEffect(() => {
    const outer = scrollRef.current
    const canvas = canvasRef.current
    if (!outer || !canvas) return
    const ro = new ResizeObserver(() => relayoutLines())
    ro.observe(outer)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [relayoutLines])

  return (
    <div style={{ marginTop: '1.15rem' }}>
      <p style={{ margin: '0 0 0.65rem', fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.5, maxWidth: 720 }}>
        Phases flow <strong style={{ color: '#7dd3fc' }}>left → right</strong>; branches stack{' '}
        <strong style={{ color: '#7dd3fc' }}>top → bottom</strong> within each phase. Open <strong style={{ color: '#e2e8f0' }}>Plan</strong>{' '}
        on a node to add siblings or the next segment. Curves show resume / fan-out links.
      </p>
      <details
        className="sway-lab-tree-help"
        style={{
          marginBottom: '0.85rem',
          borderRadius: 10,
          border: '1px solid rgba(148, 163, 184, 0.2)',
          background: 'rgba(0,0,0,0.2)',
          overflow: 'hidden',
        }}
      >
        <summary
          style={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.55rem 0.85rem',
            fontSize: '0.78rem',
            fontWeight: 600,
            color: '#cbd5e1',
          }}
        >
          <ChevronDown className="sway-lab-tree-help-chevron" size={15} style={{ flexShrink: 0, opacity: 0.8 }} aria-hidden />
          How sibling vs fan-out works
        </summary>
        <p style={{ margin: 0, padding: '0 0.85rem 0.75rem', fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6, maxWidth: 820 }}>
          <strong style={{ color: '#e2e8f0' }}>Sibling</strong> — same segment (another root or branch sharing a parent).{' '}
          <strong style={{ color: '#e2e8f0' }}>Next branch</strong> — one child resuming from this node only.{' '}
          <strong style={{ color: '#e2e8f0' }}>All P1 / All here</strong> — fan-out to every root or every node at this segment.{' '}
          Dashed curves usually indicate multi-parent fan-out.
        </p>
      </details>

      <div
        ref={scrollRef}
        className="sway-draft-tree-canvas"
        style={{
          overflowX: 'auto',
          overflowY: 'hidden',
          padding: '0.75rem 0.5rem 1rem',
          borderRadius: 12,
          border: '1px solid rgba(148, 163, 184, 0.15)',
          background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.5) 0%, rgba(2, 6, 23, 0.65) 100%)',
          minHeight: 120,
        }}
      >
        <div
          ref={canvasRef}
          style={{
            position: 'relative',
            display: 'inline-block',
            verticalAlign: 'top',
            minWidth: '100%',
          }}
        >
          <TreeConnectorSvg width={svgSize.w} height={svgSize.h} segments={lineSegments} />

          <div
            style={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'flex-start',
              gap: COL_GAP,
              position: 'relative',
              zIndex: 1,
              minWidth: 'min-content',
            }}
          >
          {nonEmptyColIndices.map((levelIdx) => (
            <div
              key={levelIdx}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'stretch',
                gap: '0.45rem',
                flexShrink: 0,
                paddingTop: '0.15rem',
              }}
            >
              <div
                style={{
                  fontSize: '0.58rem',
                  fontWeight: 800,
                  letterSpacing: '0.12em',
                  textTransform: 'uppercase',
                  color: 'rgba(148, 163, 184, 0.85)',
                  marginBottom: '0.1rem',
                  paddingLeft: '0.1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                }}
              >
                <GitBranch size={11} style={{ opacity: 0.7 }} aria-hidden />
                Phase {levelIdx + 1}
              </div>
              {columns[levelIdx].map((d) => (
                <CompactNodeCard key={d.clientId} d={d} drafts={drafts} ctx={ctx} />
              ))}
            </div>
          ))}
          </div>
        </div>
      </div>
    </div>
  )
}
