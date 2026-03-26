/**
 * Draft run tree: Phase-1 roots, then child segments that resume from parent checkpoints.
 * Expanded to Lab batch_upload specs with parent_batch_index (resolved server-side).
 */

import type { DraftRun } from '../context/LabContext'

/** Boundaries written when a segment stops (matches main.py checkpoint dirs). */
export function stopBoundaryForTreeLevel(level: number): string | undefined {
  const m = ['after_phase_1', 'after_phase_3', 'after_phase_4', 'after_phase_5']
  return m[level]
}

/** Schema `phase` strings allowed in Run editor for this tree segment. */
export function allowedSchemaPhasesForTreeLevel(level: number): Set<string> {
  const byLevel: ReadonlyArray<ReadonlyArray<string>> = [
    ['detection'],
    ['tracking', 'phase3_stitch', 'reid_dedup'],
    ['pose', 'smooth'],
    ['post_pose_prune', 'scoring'],
    ['export'],
  ]
  const i = Math.min(level, byLevel.length - 1)
  return new Set(byLevel[i] ?? byLevel[byLevel.length - 1])
}

export function treeSegmentLabel(level: number): string {
  const labels = ['Phase 1 — detect (stop after YOLO)', 'Phase 2 — track', 'Phase 3 — pose', 'Phase 4 — prune / score', 'Phase 5 — export / finish']
  return labels[Math.min(level, labels.length - 1)] ?? `Segment ${level + 1}`
}

export function dependencyClientIds(node: DraftRun, all: DraftRun[]): string[] {
  if (node.parentRef.kind === 'none') return []
  if (node.parentRef.kind === 'single') return [node.parentRef.parentClientId]
  if (node.parentRef.kind === 'all_roots') {
    return all.filter((n) => n.parentRef.kind === 'none').map((n) => n.clientId)
  }
  if (node.parentRef.kind === 'all_at_level') {
    return all.filter((n) => n.treeLevel === node.parentRef.level).map((n) => n.clientId)
  }
  return []
}

export function topoSortDrafts(drafts: DraftRun[]): DraftRun[] {
  const idSet = new Set(drafts.map((d) => d.clientId))
  const inDegree = new Map<string, number>()
  const children = new Map<string, string[]>()

  for (const d of drafts) {
    inDegree.set(d.clientId, 0)
    children.set(d.clientId, [])
  }
  for (const t of drafts) {
    const ds = dependencyClientIds(t, drafts).filter((id) => idSet.has(id))
    inDegree.set(t.clientId, ds.length)
    for (const u of ds) {
      const ch = children.get(u) ?? []
      ch.push(t.clientId)
      children.set(u, ch)
    }
  }

  const orderIdx = new Map(drafts.map((d, i) => [d.clientId, i]))
  const queue = drafts
    .filter((d) => inDegree.get(d.clientId) === 0)
    .sort((a, b) => (orderIdx.get(a.clientId)! - orderIdx.get(b.clientId)!))
  const out: DraftRun[] = []

  while (queue.length) {
    const d = queue.shift()!
    out.push(d)
    for (const tid of children.get(d.clientId) ?? []) {
      const inc = (inDegree.get(tid) ?? 0) - 1
      inDegree.set(tid, inc)
      if (inc === 0) {
        const node = drafts.find((x) => x.clientId === tid)
        if (node) queue.push(node)
      }
    }
    queue.sort((a, b) => (orderIdx.get(a.clientId)! - orderIdx.get(b.clientId)!))
  }

  if (out.length !== drafts.length) {
    throw new Error('Run tree has a cycle or missing parent — fix parent links and try again.')
  }
  return out
}

function buildCheckpointForChild(parentLevel: number, childLevel: number, parentBatchIndex: number): Record<string, unknown> {
  const expect = stopBoundaryForTreeLevel(parentLevel)
  if (!expect) {
    throw new Error(`Invalid parent level ${parentLevel} for checkpoint child`)
  }
  const stop = stopBoundaryForTreeLevel(childLevel)
  const ck: Record<string, unknown> = {
    expect_boundary: expect,
    resume_checkpoint_subdir: expect,
    force_checkpoint_load: true,
    parent_batch_index: parentBatchIndex,
  }
  if (stop) ck.stop_after_boundary = stop
  return ck
}

export type ExpandedBatchRunSpec = {
  recipe_name: string
  fields: Record<string, unknown>
  checkpoint: Record<string, unknown>
}

/**
 * Turn UI tree nodes into API run specs (order matches topological sort).
 * Fan-out nodes (`all_roots`, `all_at_level`) become multiple specs; they are not valid single parents for other nodes.
 */
export function expandDraftsToBatchSpecs(drafts: DraftRun[]): ExpandedBatchRunSpec[] {
  if (drafts.length === 0) return []
  const sorted = topoSortDrafts(drafts)
  const byId = new Map(drafts.map((d) => [d.clientId, d]))
  const idToBatchIndex = new Map<string, number>()
  const specs: ExpandedBatchRunSpec[] = []

  const roots = drafts.filter((d) => d.parentRef.kind === 'none')

  for (const node of sorted) {
    if (node.parentRef.kind === 'none') {
      const stop = stopBoundaryForTreeLevel(node.treeLevel)
      if (!stop) {
        throw new Error('Root runs must be Phase-1 segments (stop after detect).')
      }
      idToBatchIndex.set(node.clientId, specs.length)
      specs.push({
        recipe_name: node.recipeName,
        fields: node.fields,
        checkpoint: { stop_after_boundary: stop },
      })
      continue
    }

    if (node.parentRef.kind === 'single') {
      const parent = byId.get(node.parentRef.parentClientId)
      if (!parent) throw new Error(`Missing parent for run "${node.recipeName}".`)
      const pIdx = idToBatchIndex.get(node.parentRef.parentClientId)
      if (pIdx === undefined) {
        throw new Error(
          `Parent "${parent.recipeName}" cannot be used as a single parent — it fans out to multiple batch jobs. Add a Phase-2 node per parent instead.`,
        )
      }
      const ck = buildCheckpointForChild(parent.treeLevel, node.treeLevel, pIdx)
      idToBatchIndex.set(node.clientId, specs.length)
      specs.push({
        recipe_name: node.recipeName,
        fields: node.fields,
        checkpoint: ck,
      })
      continue
    }

    if (node.parentRef.kind === 'all_roots') {
      if (roots.length === 0) throw new Error('“All Phase-1 parents” needs at least one Phase-1 root.')
      const parentLevel = 0
      for (const r of roots) {
        const pIdx = idToBatchIndex.get(r.clientId)
        if (pIdx === undefined) continue
        const ck = buildCheckpointForChild(parentLevel, node.treeLevel, pIdx)
        specs.push({
          recipe_name: node.recipeName,
          fields: { ...node.fields },
          checkpoint: ck,
        })
      }
      continue
    }

    if (node.parentRef.kind === 'all_at_level') {
      const parents = drafts.filter((d) => d.treeLevel === node.parentRef.level)
      if (parents.length === 0) {
        throw new Error(`No nodes at level ${node.parentRef.level} for “all parents at level”.`)
      }
      for (const p of parents) {
        const pIdx = idToBatchIndex.get(p.clientId)
        if (pIdx === undefined) {
          throw new Error(
            `Parent "${p.recipeName}" fans out and cannot be used in “all at level” — use single-parent nodes.`,
          )
        }
        const ck = buildCheckpointForChild(p.treeLevel, node.treeLevel, pIdx)
        specs.push({
          recipe_name: node.recipeName,
          fields: { ...node.fields },
          checkpoint: ck,
        })
      }
    }
  }

  return specs
}

export function parentRefSummary(node: DraftRun, drafts: DraftRun[]): string {
  if (node.parentRef.kind === 'none') return 'Root — full video from frame 0'
  if (node.parentRef.kind === 'single') {
    const p = drafts.find((d) => d.clientId === node.parentRef.parentClientId)
    return p ? `Continues from: ${p.recipeName}` : 'Continues from: (missing parent)'
  }
  if (node.parentRef.kind === 'all_roots') {
    const n = drafts.filter((d) => d.parentRef.kind === 'none').length
    return `One run per Phase-1 root (${n} parent${n === 1 ? '' : 's'})`
  }
  const n = drafts.filter((d) => d.treeLevel === node.parentRef.level).length
  return `One run per node at segment level ${node.parentRef.level + 1} (${n} parents)`
}
