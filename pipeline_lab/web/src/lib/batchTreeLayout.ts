import type { RunInfo } from '../types'

/** True when at least one run in the session resumes from another run in the same session (checkpoint fan-out tree). */
export function sessionHasCheckpointTree(orderedIds: string[], runMap: Map<string, RunInfo>): boolean {
  const set = new Set(orderedIds)
  for (const id of orderedIds) {
    const p = runMap.get(id)?.parent_run_id?.trim()
    if (p && set.has(p)) return true
  }
  return false
}

export function computePhaseDepths(orderedIds: string[], runMap: Map<string, RunInfo>): Map<string, number> {
  const set = new Set(orderedIds)
  const memo = new Map<string, number>()

  function depthOf(id: string): number {
    if (memo.has(id)) return memo.get(id)!
    const row = runMap.get(id)
    const raw = row?.parent_run_id?.trim()
    const parent = raw && set.has(raw) ? raw : null
    const d = parent ? depthOf(parent) + 1 : 0
    memo.set(id, d)
    return d
  }

  for (const id of orderedIds) depthOf(id)
  const out = new Map<string, number>()
  for (const id of orderedIds) out.set(id, memo.get(id) ?? 0)
  return out
}

export function columnBuckets(orderedIds: string[], depths: Map<string, number>): string[][] {
  let maxD = 0
  for (const id of orderedIds) maxD = Math.max(maxD, depths.get(id) ?? 0)
  const cols: string[][] = Array.from({ length: maxD + 1 }, () => [])
  for (const id of orderedIds) {
    const d = depths.get(id) ?? 0
    cols[d].push(id)
  }
  return cols
}

export function treeEdges(orderedIds: string[], runMap: Map<string, RunInfo>): Array<{ from: string; to: string }> {
  const set = new Set(orderedIds)
  const out: Array<{ from: string; to: string }> = []
  for (const id of orderedIds) {
    const p = runMap.get(id)?.parent_run_id?.trim()
    if (p && set.has(p)) out.push({ from: p, to: id })
  }
  return out
}

/**
 * Runs whose ``parent_run_id`` chain reaches ``rootId`` within the session (excludes ``rootId``).
 * Used when deleting a checkpoint node and every downstream resume in the same batch.
 */
export function descendantRunIdsInSession(
  rootId: string,
  orderedIds: string[],
  runMap: Map<string, RunInfo>,
): string[] {
  const set = new Set(orderedIds)
  const out: string[] = []
  for (const id of orderedIds) {
    if (id === rootId) continue
    let cur: string | null = id
    const seen = new Set<string>()
    while (cur && set.has(cur) && !seen.has(cur)) {
      seen.add(cur)
      const raw: string | undefined = runMap.get(cur)?.parent_run_id?.trim()
      const p: string | null = raw && set.has(raw) ? raw : null
      if (!p) break
      if (p === rootId) {
        out.push(id)
        break
      }
      cur = p
    }
  }
  return out
}

/** ``rootId`` plus every session-local descendant (same order as ``descendantRunIdsInSession`` + root first). */
export function subtreeRunIdsInSession(
  rootId: string,
  orderedIds: string[],
  runMap: Map<string, RunInfo>,
): string[] {
  return [rootId, ...descendantRunIdsInSession(rootId, orderedIds, runMap)]
}

/** Root → … → ``runId`` along ``parent_run_id`` links within the session (empty if ``runId`` not in session). */
export function ancestorPathToRoot(
  runId: string,
  orderedIds: string[],
  runMap: Map<string, RunInfo>,
): string[] {
  const set = new Set(orderedIds)
  const up: string[] = []
  let cur: string | null = runId
  const seen = new Set<string>()
  while (cur && set.has(cur) && !seen.has(cur)) {
    seen.add(cur)
    up.push(cur)
    const rawParent: string | undefined = runMap.get(cur)?.parent_run_id?.trim()
    cur = rawParent && set.has(rawParent) ? rawParent : null
  }
  up.reverse()
  return up
}

/** Edge keys ``from-to`` for consecutive nodes on a root→leaf path (for SVG highlighting). */
export function pathSegmentKeys(pathRootToLeaf: string[]): Set<string> {
  const s = new Set<string>()
  for (let i = 0; i < pathRootToLeaf.length - 1; i++) {
    s.add(`${pathRootToLeaf[i]}-${pathRootToLeaf[i + 1]}`)
  }
  return s
}

/** Phase-column order as rendered in the tree (left→right columns, top→bottom within each). */
export function flattenTreeVisualOrder(orderedIds: string[], runMap: Map<string, RunInfo>): string[] {
  const depths = computePhaseDepths(orderedIds, runMap)
  const columns = columnBuckets(orderedIds, depths)
  const out: string[] = []
  for (const col of columns) {
    for (const id of col) out.push(id)
  }
  return out
}

/** Pixel geometry for absolutely positioned checkpoint tree (open-canvas layout). */
export type TreeCanvasLayout = {
  positions: Map<string, { left: number; top: number }>
  canvasWidth: number
  canvasHeight: number
  /** Unique depths 0..max for phase band labels */
  depthXs: Map<number, number>
  maxDepth: number
}

/** Exported for tree view cards to match edge geometry. */
export const TREE_LAYOUT_CARD_WIDTH = 360
/** Layout slot height — keep in sync with typical card stack (actions + Watch). */
export const TREE_LAYOUT_CARD_HEIGHT = 248
const HORIZONTAL_STEP = TREE_LAYOUT_CARD_WIDTH + 56
const VERTICAL_SLOT = TREE_LAYOUT_CARD_HEIGHT + 32

function childrenByParent(orderedIds: string[], runMap: Map<string, RunInfo>): Map<string | null, string[]> {
  const set = new Set(orderedIds)
  const m = new Map<string | null, string[]>()
  const index = new Map<string, number>()
  orderedIds.forEach((id, i) => index.set(id, i))
  for (const id of orderedIds) {
    const raw = runMap.get(id)?.parent_run_id?.trim()
    const p = raw && set.has(raw) ? raw : null
    const arr = m.get(p)
    if (arr) arr.push(id)
    else m.set(p, [id])
  }
  for (const [, arr] of m) {
    arr.sort((a, b) => (index.get(a) ?? 0) - (index.get(b) ?? 0))
  }
  return m
}

type SubtreeBox = { top: number; bottom: number }

/**
 * Hierarchical positions: roots stack top-to-bottom; each parent is vertically centered on its subtree.
 * Reads as a real tree (branches spread) instead of rigid phase columns aligned to the top (triangle silhouette).
 */
export function computeTreeCanvasLayout(orderedIds: string[], runMap: Map<string, RunInfo>): TreeCanvasLayout {
  const depths = computePhaseDepths(orderedIds, runMap)
  const byParent = childrenByParent(orderedIds, runMap)
  const roots = byParent.get(null) ?? []
  const positions = new Map<string, { left: number; top: number }>()

  let leafCursor = 0

  function layoutSubtree(id: string): SubtreeBox {
    const d = depths.get(id) ?? 0
    const left = d * HORIZONTAL_STEP
    const kids = byParent.get(id) ?? []

    if (kids.length === 0) {
      const top = leafCursor * VERTICAL_SLOT
      leafCursor += 1
      positions.set(id, { left, top })
      return { top, bottom: top + TREE_LAYOUT_CARD_HEIGHT }
    }

    let minT = Infinity
    let maxB = -Infinity
    for (const c of kids) {
      const box = layoutSubtree(c)
      minT = Math.min(minT, box.top)
      maxB = Math.max(maxB, box.bottom)
    }
    const top = (minT + maxB - TREE_LAYOUT_CARD_HEIGHT) / 2
    positions.set(id, { left, top })
    return { top, bottom: top + TREE_LAYOUT_CARD_HEIGHT }
  }

  for (const r of roots) {
    layoutSubtree(r)
  }

  let maxDepth = 0
  let maxRight = 0
  let maxBottom = 0
  for (const id of orderedIds) {
    const pos = positions.get(id)
    if (!pos) continue
    const depth = depths.get(id) ?? 0
    maxDepth = Math.max(maxDepth, depth)
    maxRight = Math.max(maxRight, pos.left + TREE_LAYOUT_CARD_WIDTH)
    maxBottom = Math.max(maxBottom, pos.top + TREE_LAYOUT_CARD_HEIGHT)
  }

  const depthXs = new Map<number, number>()
  for (let di = 0; di <= maxDepth; di++) {
    depthXs.set(di, di * HORIZONTAL_STEP)
  }

  return {
    positions,
    canvasWidth: maxRight,
    canvasHeight: maxBottom,
    depthXs,
    maxDepth,
  }
}
