import { useCallback, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  CheckSquare,
  Square,
  Play,
  Settings2,
  StopCircle,
  RotateCw,
  Trash2,
} from 'lucide-react'
import type { RunInfo } from '../types'
import {
  ancestorPathToRoot,
  columnBuckets,
  computePhaseDepths,
  computeTreeCanvasLayout,
  descendantRunIdsInSession,
  pathSegmentKeys,
  treeEdges,
  TREE_LAYOUT_CARD_WIDTH,
} from '../lib/batchTreeLayout'
import { statusPresentationForRun } from '../lib/runStatusPresentation'

function formatShortId(runId: string) {
  return runId.length > 12 ? `${runId.slice(0, 8)}…` : runId
}

function formatCreated(iso?: string) {
  if (!iso) return null
  try {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return null
    return d.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
  } catch {
    return null
  }
}

/** Smooth left→right connector with horizontal tangents (readable flow-chart style). */
function edgePathD(x1: number, y1: number, x2: number, y2: number): string {
  const span = Math.max(0, x2 - x1)
  const dx = Math.min(72, Math.max(28, span * 0.28))
  return `M ${x1} ${y1} C ${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`
}

export type BatchTreeViewProps = {
  orderedIds: string[]
  runMap: Map<string, RunInfo>
  videoStemFallback: string
  treeCompareSelected: Set<string>
  onToggleCompare: (runId: string, detail: { shiftKey: boolean }) => void
  onOpenConfig: (runId: string) => void
  onWatch: (runId: string) => void
  onStop?: (runId: string) => void
  onRerun?: (runId: string) => void
  onDelete?: (runId: string) => void
  /** Delete this run and every session-local child (after in-tree confirm); omit to only use ``onDelete``. */
  onDeleteSubtree?: (rootId: string) => void | Promise<void>
  stoppingRunId?: string | null
  rerunningRunId?: string | null
  deletingRunId?: string | null
  /** Open Compare with finished runs in this column and the phase preview mapped to this tree depth. */
  onComparePhaseColumn?: (columnIndex: number, runIdsInColumn: string[]) => void
}

export function BatchTreeView({
  orderedIds,
  runMap,
  videoStemFallback,
  treeCompareSelected,
  onToggleCompare,
  onOpenConfig,
  onWatch,
  onStop,
  onRerun,
  onDelete,
  onDeleteSubtree,
  stoppingRunId,
  rerunningRunId,
  deletingRunId,
  onComparePhaseColumn,
}: BatchTreeViewProps) {
  const uid = useId().replace(/:/g, '')
  const arrowMarkerId = `${uid}-mk`
  const arrowMarkerHiId = `${uid}-mk-hi`
  const arrowMarkerDelId = `${uid}-mk-del`
  const [focusedRunId, setFocusedRunId] = useState<string | null>(null)
  const [pendingDeleteRootId, setPendingDeleteRootId] = useState<string | null>(null)

  const depths = useMemo(() => computePhaseDepths(orderedIds, runMap), [orderedIds, runMap])
  const phases = useMemo(() => columnBuckets(orderedIds, depths), [orderedIds, depths])
  const canvasLayout = useMemo(() => computeTreeCanvasLayout(orderedIds, runMap), [orderedIds, runMap])
  const edges = useMemo(() => treeEdges(orderedIds, runMap), [orderedIds, runMap])

  const DESK_PADDING = 120
  const PHASE_HEADER = 56

  const focusPath = useMemo(() => {
    if (!focusedRunId) return []
    return ancestorPathToRoot(focusedRunId, orderedIds, runMap)
  }, [focusedRunId, orderedIds, runMap])

  /** Ancestors → focused node, plus every session-local child/descendant (for card + edge highlight). */
  const { pathNodeSet, pathSegmentSet } = useMemo(() => {
    if (!focusedRunId) {
      return { pathNodeSet: new Set<string>(), pathSegmentSet: new Set<string>() }
    }
    const descendants = descendantRunIdsInSession(focusedRunId, orderedIds, runMap)
    const subtreeNodes = new Set<string>([focusedRunId, ...descendants])
    const nodes = new Set<string>([...focusPath, ...descendants])
    const segments = pathSegmentKeys(focusPath)
    for (const e of edges) {
      if (subtreeNodes.has(e.from) && subtreeNodes.has(e.to)) segments.add(`${e.from}-${e.to}`)
    }
    return { pathNodeSet: nodes, pathSegmentSet: segments }
  }, [focusedRunId, focusPath, orderedIds, runMap, edges])

  const pendingDeleteChildIds = useMemo(() => {
    if (!pendingDeleteRootId) return new Set<string>()
    return new Set(descendantRunIdsInSession(pendingDeleteRootId, orderedIds, runMap))
  }, [pendingDeleteRootId, orderedIds, runMap])

  const pendingDeleteSubtreeEdgeKeys = useMemo(() => {
    if (!pendingDeleteRootId) return new Set<string>()
    const subtreeNodes = new Set<string>([pendingDeleteRootId, ...pendingDeleteChildIds])
    const s = new Set<string>()
    for (const e of edges) {
      if (subtreeNodes.has(e.from) && subtreeNodes.has(e.to)) s.add(`${e.from}-${e.to}`)
    }
    return s
  }, [pendingDeleteRootId, pendingDeleteChildIds, edges])

  /** Horizontal scroll port (viewport). */
  const scrollRef = useRef<HTMLDivElement>(null)
  /** Same layer as columns (`width: max-content`) so edge coords match the scrollable canvas. */
  const edgesCanvasRef = useRef<HTMLDivElement>(null)
  const nodeRefs = useRef<Record<string, HTMLDivElement | null>>({})
  const [svgPaths, setSvgPaths] = useState<Array<{ d: string; key: string; from: string; to: string }>>([])

  const measure = useCallback(() => {
    const layer = edgesCanvasRef.current
    if (!layer) return
    const wr = layer.getBoundingClientRect()
    const paths: Array<{ d: string; key: string; from: string; to: string }> = []
    for (const e of edges) {
      const a = nodeRefs.current[e.from]
      const b = nodeRefs.current[e.to]
      if (!a || !b) continue
      const ar = a.getBoundingClientRect()
      const br = b.getBoundingClientRect()
      const x1 = ar.right - wr.left
      const y1 = ar.top + ar.height / 2 - wr.top
      const x2 = br.left - wr.left
      const y2 = br.top + br.height / 2 - wr.top
      const d = edgePathD(x1, y1, x2, y2)
      paths.push({ d, key: `${e.from}-${e.to}`, from: e.from, to: e.to })
    }
    setSvgPaths(paths)
  }, [edges])

  useLayoutEffect(() => {
    const run = () => {
      requestAnimationFrame(() => {
        requestAnimationFrame(measure)
      })
    }
    run()
    const layer = edgesCanvasRef.current
    const scrollEl = scrollRef.current
    if (!layer) return undefined
    const ro = new ResizeObserver(run)
    ro.observe(layer)
    if (scrollEl) {
      scrollEl.addEventListener('scroll', run, { passive: true })
    }
    return () => {
      ro.disconnect()
      if (scrollEl) scrollEl.removeEventListener('scroll', run)
    }
  }, [measure, canvasLayout, orderedIds, runMap, focusedRunId, pendingDeleteRootId])

  const setNodeRef = useCallback((id: string, el: HTMLDivElement | null) => {
    nodeRefs.current[id] = el
  }, [])

  const requestDeleteInTree = useCallback(
    (runId: string) => {
      const children = descendantRunIdsInSession(runId, orderedIds, runMap)
      if (children.length === 0) {
        onDelete?.(runId)
        return
      }
      if (!onDeleteSubtree) {
        onDelete?.(runId)
        return
      }
      setPendingDeleteRootId(runId)
      window.setTimeout(() => {
        const n = children.length
        const ok = window.confirm(
          `Delete this run and ${n} child run${n === 1 ? '' : 's'} (${n + 1} total) from disk? Outputs, logs, and previews for all of them will be removed. This cannot be undone.`,
        )
        setPendingDeleteRootId(null)
        if (ok) void onDeleteSubtree(runId)
      }, 0)
    },
    [onDelete, onDeleteSubtree, orderedIds, runMap],
  )

  const { positions, canvasWidth, canvasHeight, depthXs, maxDepth } = canvasLayout
  const innerMinHeight = `max(62vh, ${PHASE_HEADER + canvasHeight + 2 * DESK_PADDING}px)`

  return (
    <div
      ref={scrollRef}
      className="batch-tree-desk"
      style={{
        position: 'relative',
        overflow: 'auto',
        padding: '0.75rem 0 1rem',
        maxHeight: 'min(88vh, 1400px)',
        minHeight: 280,
        borderRadius: 12,
        border: '1px solid rgba(148, 163, 184, 0.12)',
        background: 'radial-gradient(ellipse 120% 80% at 50% 0%, rgba(34, 211, 238, 0.06), transparent 55%), rgba(15, 23, 42, 0.25)',
        WebkitOverflowScrolling: 'touch',
        overscrollBehavior: 'contain',
      }}
    >
      {focusPath.length > 0 && focusedRunId && (
        <div
          style={{
            position: 'relative',
            zIndex: 1,
            marginBottom: '0.85rem',
            padding: '0.65rem 0.85rem',
            borderRadius: 10,
            border: '1px solid rgba(34, 211, 238, 0.28)',
            background: 'rgba(15, 23, 42, 0.92)',
            fontSize: '0.82rem',
            lineHeight: 1.55,
            color: 'var(--text-muted)',
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: '0.5rem 0.75rem',
          }}
        >
          <span style={{ color: 'var(--halo-cyan)', fontWeight: 700 }}>Path from root</span>
          <span style={{ color: '#e2e8f0' }}>
            {focusPath.map((id, i) => {
              const label =
                runMap.get(id)?.recipe_name?.trim() || formatShortId(id)
              const step = focusedRunId === id
              return (
                <span key={id}>
                  {i > 0 && <span style={{ color: 'var(--text-muted)', margin: '0 0.2rem' }}>→</span>}
                  <span
                    style={{
                      fontWeight: step ? 700 : 500,
                      color: step ? 'var(--halo-cyan)' : '#e2e8f0',
                      fontFamily: step ? 'inherit' : 'ui-monospace, monospace',
                      fontSize: step ? '0.9rem' : '0.78rem',
                    }}
                  >
                    {label}
                  </span>
                </span>
              )
            })}
          </span>
          <button
            type="button"
            className="btn"
            style={{ padding: '0.25rem 0.55rem', fontSize: '0.72rem', marginLeft: 'auto' }}
            onClick={() => setFocusedRunId(null)}
          >
            Clear path
          </button>
        </div>
      )}

      {pendingDeleteRootId && (
        <div
          style={{
            position: 'relative',
            zIndex: 1,
            marginBottom: '0.85rem',
            padding: '0.65rem 0.85rem',
            borderRadius: 10,
            border: '1px solid rgba(248, 113, 113, 0.45)',
            background: 'rgba(127, 29, 29, 0.35)',
            fontSize: '0.82rem',
            lineHeight: 1.55,
            color: '#fecaca',
          }}
        >
          <strong style={{ color: '#fca5a5' }}>Delete preview</strong>{' '}
          The selected run (stronger outline) and every child run highlighted in red will be removed if you confirm in the
          dialog.
        </div>
      )}

      <div
        ref={edgesCanvasRef}
        style={{
          position: 'relative',
          boxSizing: 'border-box',
          width: '100%',
          minWidth: `max(100%, ${canvasWidth + 2 * DESK_PADDING}px)`,
          minHeight: innerMinHeight,
          padding: DESK_PADDING,
        }}
      >
        <svg
          aria-hidden
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            overflow: 'visible',
            zIndex: 0,
            shapeRendering: 'geometricPrecision',
          }}
        >
          <defs>
            <marker
              id={arrowMarkerId}
              markerUnits="userSpaceOnUse"
              markerWidth="9"
              markerHeight="9"
              refX="8"
              refY="4.5"
              orient="auto"
              viewBox="0 0 9 9"
            >
              <path d="M0,0 L9,4.5 L0,9 z" fill="rgba(120, 190, 205, 0.65)" />
            </marker>
            <marker
              id={arrowMarkerHiId}
              markerUnits="userSpaceOnUse"
              markerWidth="11"
              markerHeight="11"
              refX="10"
              refY="5.5"
              orient="auto"
              viewBox="0 0 11 11"
            >
              <path d="M0,0 L11,5.5 L0,11 z" fill="rgb(34, 211, 238)" />
            </marker>
            <marker
              id={arrowMarkerDelId}
              markerUnits="userSpaceOnUse"
              markerWidth="11"
              markerHeight="11"
              refX="10"
              refY="5.5"
              orient="auto"
              viewBox="0 0 11 11"
            >
              <path d="M0,0 L11,5.5 L0,11 z" fill="rgb(248, 113, 113)" />
            </marker>
          </defs>
          {svgPaths.map(({ d, key }) => {
            const deleteEdge = pendingDeleteRootId !== null && pendingDeleteSubtreeEdgeKeys.has(key)
            const onPath = focusedRunId !== null && pathSegmentSet.has(key)
            const dim = focusedRunId !== null && !deleteEdge
            let stroke = 'rgba(100, 180, 200, 0.42)'
            let sw = 1.75
            let markerEnd: string | undefined
            if (deleteEdge) {
              stroke = 'rgba(248, 113, 113, 0.88)'
              sw = 2.65
              markerEnd = `url(#${arrowMarkerDelId})`
            } else if (!dim) {
              markerEnd = undefined
            } else if (onPath) {
              stroke = 'rgba(34, 211, 238, 0.92)'
              sw = 2.75
              markerEnd = `url(#${arrowMarkerHiId})`
            } else {
              stroke = 'rgba(100, 180, 200, 0.14)'
              sw = 1.15
              markerEnd = `url(#${arrowMarkerId})`
            }
            return (
              <path
                key={key}
                d={d}
                fill="none"
                stroke={stroke}
                strokeWidth={sw}
                strokeLinecap="round"
                strokeLinejoin="round"
                markerEnd={markerEnd}
              />
            )
          })}
        </svg>

        {Array.from({ length: maxDepth + 1 }, (_, colIdx) => {
          const colIds = phases[colIdx] ?? []
          const left = depthXs.get(colIdx) ?? colIdx * (TREE_LAYOUT_CARD_WIDTH + 56)
          return onComparePhaseColumn ? (
            <button
              key={`phase-h-${colIdx}`}
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                onComparePhaseColumn(colIdx, colIds)
              }}
              title="Compare every finished run in this column, using the phase preview aligned to this checkpoint stage (see Compare view menu to change clip)."
              style={{
                position: 'absolute',
                zIndex: 2,
                left,
                top: 6,
                width: TREE_LAYOUT_CARD_WIDTH,
                textAlign: 'left',
                fontSize: '0.68rem',
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--halo-cyan)',
                padding: '0.35rem 0.15rem 0.5rem',
                margin: 0,
                border: 'none',
                borderBottom: '1px solid var(--glass-border)',
                background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.92) 0%, rgba(15, 23, 42, 0.65) 100%)',
                cursor: 'pointer',
                fontFamily: 'inherit',
                borderRadius: 8,
              }}
            >
              Phase {colIdx + 1} — compare column
            </button>
          ) : (
            <div
              key={`phase-h-${colIdx}`}
              style={{
                position: 'absolute',
                zIndex: 2,
                left,
                top: 10,
                width: TREE_LAYOUT_CARD_WIDTH,
                fontSize: '0.68rem',
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--text-muted)',
                padding: '0 0.15rem',
                borderBottom: '1px solid var(--glass-border)',
                paddingBottom: '0.45rem',
              }}
            >
              Phase {colIdx + 1}
            </div>
          )
        })}

        {orderedIds.map((runId) => {
          const pos = positions.get(runId)
          const run = runMap.get(runId)
          if (!run || !pos) return null
          const pres = statusPresentationForRun(run)
          const { color, Icon } = pres
          const isDone = run.status === 'done'
          const isRunning = run.status === 'running'
          const subprocessLive = isRunning && run.subprocess_alive === true
          const displayName = (run.recipe_name && run.recipe_name.trim()) || 'Run'
          const videoLine = run.video_stem || videoStemFallback
          const created = formatCreated(run.created)
          const compareOn = treeCompareSelected.has(runId)
          const pathHighlight = focusedRunId !== null && pathNodeSet.has(runId)
          const isFocusedCard = focusedRunId === runId
          const isDeleteRootPreview = pendingDeleteRootId === runId
          const isDeleteChildPreview = pendingDeleteChildIds.has(runId)
          let cardBorder = `1px solid ${compareOn ? 'rgba(34, 211, 238, 0.55)' : `${color}35`}`
          if (isDeleteRootPreview) cardBorder = '2px solid rgba(248, 113, 113, 0.92)'
          else if (isDeleteChildPreview) cardBorder = '1px solid rgba(248, 113, 113, 0.72)'
          else if (isFocusedCard) cardBorder = '2px solid rgba(34, 211, 238, 0.9)'
          else if (pathHighlight) cardBorder = '1px solid rgba(34, 211, 238, 0.55)'

          const cardBg = isDeleteRootPreview
            ? 'rgba(248, 113, 113, 0.14)'
            : isDeleteChildPreview
              ? 'rgba(248, 113, 113, 0.08)'
              : isFocusedCard
                ? 'rgba(34, 211, 238, 0.1)'
                : pathHighlight
                  ? 'rgba(34, 211, 238, 0.05)'
                  : compareOn
                    ? 'rgba(34, 211, 238, 0.06)'
                    : 'rgba(15, 23, 42, 0.35)'

          return (
            <div
              key={runId}
              ref={(el) => setNodeRef(runId, el)}
              role="button"
              tabIndex={0}
              aria-pressed={isFocusedCard}
              aria-label={`${displayName}. Click to highlight path from root and this run's children in the batch; click again to clear.`}
              className="glass-panel"
              onClick={() => setFocusedRunId((cur) => (cur === runId ? null : runId))}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  setFocusedRunId((cur) => (cur === runId ? null : runId))
                }
              }}
              style={{
                position: 'absolute',
                left: pos.left,
                top: PHASE_HEADER + pos.top,
                width: TREE_LAYOUT_CARD_WIDTH,
                boxSizing: 'border-box',
                zIndex: 1,
                padding: '1rem 1.25rem',
                border: cardBorder,
                background: cardBg,
                display: 'flex',
                flexDirection: 'column',
                gap: '0.5rem',
                cursor: 'pointer',
                outline: 'none',
                boxShadow: isDeleteRootPreview
                  ? '0 0 0 1px rgba(248, 113, 113, 0.4)'
                  : isFocusedCard
                    ? '0 0 0 1px rgba(34, 211, 238, 0.35)'
                    : undefined,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
                <button
                  type="button"
                  disabled={!isDone}
                  title={
                    isDone
                      ? compareOn
                        ? 'Remove from compare. Shift-click to add every finished run from here to the last Compare you clicked.'
                        : 'Add to side-by-side compare. Shift-click to select a range (with the last Compare you clicked).'
                      : 'Compare is available when this run finishes successfully'
                  }
                  onClick={(e) => {
                    e.stopPropagation()
                    onToggleCompare(runId, { shiftKey: e.shiftKey })
                  }}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    padding: '0.2rem 0.35rem',
                    borderRadius: 8,
                    border: `1px solid ${isDone ? (compareOn ? 'rgba(34,211,238,0.5)' : 'rgba(148,163,184,0.35)') : 'rgba(148,163,184,0.2)'}`,
                    background: isDone ? 'rgba(0,0,0,0.2)' : 'transparent',
                    color: isDone ? '#e2e8f0' : 'var(--text-muted)',
                    cursor: isDone ? 'pointer' : 'not-allowed',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                  }}
                >
                  {isDone ? (
                    compareOn ? (
                      <CheckSquare size={15} strokeWidth={2.2} style={{ color: 'var(--halo-cyan)' }} />
                    ) : (
                      <Square size={15} strokeWidth={2} />
                    )
                  ) : (
                    <Square size={15} strokeWidth={2} style={{ opacity: 0.35 }} />
                  )}
                  Compare
                </button>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: 'ui-monospace, monospace' }}>
                  {formatShortId(runId)}
                </span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <Icon
                  size={16}
                  className={subprocessLive ? 'sway-spin' : undefined}
                  style={{ flexShrink: 0, color }}
                />
                <div style={{ fontWeight: 700, color: '#fff', fontSize: '0.92rem', lineHeight: 1.25 }}>{displayName}</div>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                {videoLine}
                {created ? <span style={{ display: 'block', marginTop: '0.2rem', opacity: 0.9 }}>{created}</span> : null}
              </div>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    onOpenConfig(runId)
                  }}
                  style={{
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: 'var(--halo-cyan)',
                    fontWeight: 600,
                    fontSize: '0.72rem',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.25rem',
                    padding: 0,
                    fontFamily: 'inherit',
                  }}
                >
                  <Settings2 size={12} aria-hidden /> Config
                </button>
                {subprocessLive && onStop && (
                  <button
                    type="button"
                    className="btn"
                    disabled={stoppingRunId === runId}
                    style={{
                      padding: '0.3rem 0.5rem',
                      fontSize: '0.7rem',
                      color: '#fcd34d',
                      borderColor: 'rgba(251,191,36,0.4)',
                    }}
                    onClick={(e) => {
                      e.stopPropagation()
                      onStop(runId)
                    }}
                  >
                    <StopCircle size={12} aria-hidden /> {stoppingRunId === runId ? '…' : 'Stop'}
                  </button>
                )}
                {onRerun && (
                  <button
                    type="button"
                    className="btn"
                    disabled={subprocessLive || rerunningRunId === runId || deletingRunId === runId}
                    style={{
                      padding: '0.3rem 0.5rem',
                      fontSize: '0.7rem',
                      color: '#7dd3fc',
                      borderColor: 'rgba(125,211,252,0.35)',
                    }}
                    onClick={(e) => {
                      e.stopPropagation()
                      onRerun(runId)
                    }}
                  >
                    <RotateCw size={12} className={rerunningRunId === runId ? 'sway-spin' : undefined} aria-hidden />{' '}
                    {rerunningRunId === runId ? '…' : 'Rerun'}
                  </button>
                )}
                {onDelete && (
                  <button
                    type="button"
                    className="btn"
                    disabled={subprocessLive || deletingRunId === runId}
                    style={{
                      padding: '0.3rem 0.5rem',
                      fontSize: '0.7rem',
                      color: '#f87171',
                      borderColor: 'rgba(248,113,113,0.35)',
                    }}
                    onClick={(e) => {
                      e.stopPropagation()
                      requestDeleteInTree(runId)
                    }}
                  >
                    <Trash2 size={12} aria-hidden />
                  </button>
                )}
              </div>

              {isDone && (
                <button
                  type="button"
                  className="btn primary"
                  style={{ width: '100%', padding: '0.45rem', fontSize: '0.78rem' }}
                  onClick={(e) => {
                    e.stopPropagation()
                    onWatch(runId)
                  }}
                >
                  <Play size={14} aria-hidden /> Watch
                </button>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
