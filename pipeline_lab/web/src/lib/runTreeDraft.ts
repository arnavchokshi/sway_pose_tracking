/**
 * Draft run tree: Phase-1 roots, then child segments that resume from parent checkpoints.
 * Expanded to Lab batch_upload specs with parent_batch_index (resolved server-side).
 */

import type { DraftRun } from '../context/LabContext'

function isFanOutNode(d: DraftRun) {
  return d.parentRef.kind === 'all_roots' || d.parentRef.kind === 'all_at_level'
}

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

/** Root plus every draft that chains from it via ``single`` parents only (for removing a branch in simple view). */
export function descendantIdsIncludingRoot(rootId: string, all: DraftRun[]): Set<string> {
  const acc = new Set<string>([rootId])
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

export function dependencyClientIds(node: DraftRun, all: DraftRun[]): string[] {
  const pr = node.parentRef
  switch (pr.kind) {
    case 'none':
      return []
    case 'single':
      return [pr.parentClientId]
    case 'all_roots':
      return all.filter((n) => n.parentRef.kind === 'none').map((n) => n.clientId)
    case 'all_at_level':
      return all.filter((n) => n.treeLevel === pr.level && !isFanOutNode(n)).map((n) => n.clientId)
    default:
      return []
  }
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
  /** Omitted or empty = full pipeline (no checkpoint stop / resume). */
  checkpoint?: Record<string, unknown>
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
      if (node.treeLevel !== 0) {
        throw new Error(
          `Run "${node.recipeName}" is a root but uses segment ${node.treeLevel + 1}. Roots must stay on Phase 1 — edit or remove this node.`,
        )
      }
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
      if (node.treeLevel !== parent.treeLevel + 1) {
        throw new Error(
          `Run "${node.recipeName}" is segment ${node.treeLevel + 1} but its parent "${parent.recipeName}" is segment ${parent.treeLevel + 1}. Use Next segment from the parent card to keep the ladder aligned.`,
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
      if (node.treeLevel !== 1) {
        throw new Error(
          `“All Phase-1 roots” continuations must be segment 2 (track). "${node.recipeName}" is segment ${node.treeLevel + 1}.`,
        )
      }
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
      const lvl = node.parentRef.level
      if (node.treeLevel !== lvl + 1) {
        throw new Error(
          `“All at segment ${lvl + 1}” node "${node.recipeName}" must be exactly one segment deeper (expected segment ${lvl + 2}, got ${node.treeLevel + 1}).`,
        )
      }
      const parents = drafts.filter((d) => d.treeLevel === lvl && !isFanOutNode(d))
      if (parents.length === 0) {
        throw new Error(`No nodes at level ${lvl} for “all parents at level”.`)
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

/** One batch job per Phase-1 root: full pipeline, same video (no checkpoint tree). */
export function expandSimpleRootsToBatchSpecs(drafts: DraftRun[]): ExpandedBatchRunSpec[] {
  return drafts
    .filter((d) => d.parentRef.kind === 'none')
    .map((d) => ({
      recipe_name: d.recipeName,
      fields: d.fields,
    }))
}

export function parentRefSummary(node: DraftRun, drafts: DraftRun[]): string {
  const pr = node.parentRef
  switch (pr.kind) {
    case 'none':
      return 'Root — full video from frame 0'
    case 'single': {
      const p = drafts.find((d) => d.clientId === pr.parentClientId)
      return p ? `Continues from: ${p.recipeName}` : 'Continues from: (missing parent)'
    }
    case 'all_roots': {
      const n = drafts.filter((d) => d.parentRef.kind === 'none').length
      return `One run per Phase-1 root (${n} parent${n === 1 ? '' : 's'})`
    }
    case 'all_at_level': {
      const n = drafts.filter((d) => d.treeLevel === pr.level && !isFanOutNode(d)).length
      return `One run per node at segment level ${pr.level + 1} (${n} parents)`
    }
    default:
      return ''
  }
}
