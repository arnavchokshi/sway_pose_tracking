import { COCO_BONE_EDGES, EXTREMITY_JOINTS } from './bones'
import type { BoneEdge, DancerPoseFrame, KeypointMap, TeamPoseFrame } from './types'

function vec2(a: [number, number], b: [number, number]): [number, number] {
  return [b[0] - a[0], b[1] - a[1]]
}

function len2(v: [number, number]): number {
  return Math.hypot(v[0], v[1])
}

/** Cosine similarity in [-1, 1]. */
export function cosineSimilarity2D(u: [number, number], v: [number, number]): number {
  const Lu = len2(u)
  const Lv = len2(v)
  if (Lu < 1e-8 || Lv < 1e-8) return 0
  return (u[0] * v[0] + u[1] * v[1]) / (Lu * Lv)
}

/** Map cosine to 0–100 (opposite → 0, same direction → 100). */
export function cosineToPercentMatch(cos: number): number {
  const clamped = Math.max(-1, Math.min(1, cos))
  return Math.round(((clamped + 1) / 2) * 100)
}

export function boneVector(kp: KeypointMap, from: string, to: string): [number, number] | null {
  const a = kp[from]
  const b = kp[to]
  if (!a || !b) return null
  return vec2([a[0], a[1]], [b[0], b[1]])
}

/**
 * Golden reference keypoints: target dancer if set; else median x/y per joint
 * across front-row dancers (or all dancers if no row tags).
 */
export function goldenKeypointsFromTeam(
  dancers: DancerPoseFrame[],
  opts: { targetDancerId?: number | null },
): KeypointMap | null {
  if (dancers.length === 0) return null
  let pool = dancers
  if (opts.targetDancerId != null) {
    const t = dancers.find((d) => d.id === opts.targetDancerId)
    if (t) return t.keypoints
  }
  const front = dancers.filter((d) => d.row === 'front')
  pool = front.length > 0 ? front : dancers

  const names = new Set<string>()
  for (const d of pool) {
    Object.keys(d.keypoints).forEach((k) => names.add(k))
  }
  const out: KeypointMap = {}
  for (const name of names) {
    const xs: number[] = []
    const ys: number[] = []
    const cs: number[] = []
    for (const d of pool) {
      const p = d.keypoints[name]
      if (!p) continue
      xs.push(p[0])
      ys.push(p[1])
      if (p[2] != null) cs.push(p[2]!)
    }
    if (xs.length === 0) continue
    xs.sort((a, b) => a - b)
    ys.sort((a, b) => a - b)
    const mx = xs[Math.floor(xs.length / 2)]!
    const my = ys[Math.floor(ys.length / 2)]!
    const mc = cs.length ? cs.reduce((a, b) => a + b, 0) / cs.length : 1
    out[name] = [mx, my, mc]
  }
  return Object.keys(out).length ? out : null
}

export type SpatialJointScore = { joint: string; percent: number; cos: number }

/**
 * Per-bone cosine vs golden; score attributed to the distal joint (bone[1]).
 */
export function spatialScoresVsGolden(
  golden: KeypointMap,
  dancer: KeypointMap,
  edges: BoneEdge[] = COCO_BONE_EDGES,
): SpatialJointScore[] {
  const rows: SpatialJointScore[] = []
  for (const [from, to] of edges) {
    const gv = boneVector(golden, from, to)
    const dv = boneVector(dancer, from, to)
    if (!gv || !dv) continue
    const cos = cosineSimilarity2D(gv, dv)
    rows.push({ joint: to, percent: cosineToPercentMatch(cos), cos })
  }
  return rows
}

function keypointSpeed(kp0: KeypointMap, kp1: KeypointMap, joint: string): number {
  const a = kp0[joint]
  const b = kp1[joint]
  if (!a || !b) return 0
  return Math.hypot(b[0] - a[0], b[1] - a[1])
}

/** Sum of extremity displacements between consecutive frames (proxy for velocity). */
export function extremityMotionSeries(frames: KeypointMap[]): number[] {
  if (frames.length < 2) return []
  const out: number[] = []
  for (let i = 1; i < frames.length; i++) {
    let s = 0
    for (const j of EXTREMITY_JOINTS) {
      s += keypointSpeed(frames[i - 1]!, frames[i]!, j)
    }
    out.push(s)
  }
  return out
}

export function argmax(arr: number[]): number {
  if (arr.length === 0) return -1
  let mi = 0
  for (let i = 1; i < arr.length; i++) {
    if (arr[i]! > arr[mi]!) mi = i
  }
  return mi
}

/**
 * Peak frame index in original frame list (motion between peak-1 and peak).
 * Returns offset into `teamHistory` (same index as frame_idx).
 */
export function peakExtremityFrameIndex(frames: KeypointMap[]): number {
  const series = extremityMotionSeries(frames)
  if (series.length === 0) return 0
  const rel = argmax(series)
  return rel + 1
}

export function temporalOffsetSeconds(
  refFrames: KeypointMap[],
  dancerFrames: KeypointMap[],
  fps: number,
): number {
  const pr = peakExtremityFrameIndex(refFrames)
  const pd = peakExtremityFrameIndex(dancerFrames)
  return (pd - pr) / fps
}

export type BboxCenter = { id: number; x: number; y: number }

/**
 * Compare pairwise distances to a regular grid of `expectedSpacing` (same units as x,y).
 */
export function formationMeanSpacingError(
  actual: BboxCenter[],
  gridCols: number,
  _gridRows: number,
  expectedSpacing: number,
): number {
  if (actual.length < 2) return 0
  const sorted = [...actual].sort((a, b) => a.id - b.id)
  let sum = 0
  let n = 0
  for (let i = 0; i < sorted.length; i++) {
    for (let j = i + 1; j < sorted.length; j++) {
      const d = Math.hypot(sorted[j]!.x - sorted[i]!.x, sorted[j]!.y - sorted[i]!.y)
      const rowI = Math.floor(i / gridCols)
      const colI = i % gridCols
      const rowJ = Math.floor(j / gridCols)
      const colJ = j % gridCols
      const manhattan = Math.abs(rowI - rowJ) + Math.abs(colI - colJ)
      const expected = manhattan > 0 ? manhattan * expectedSpacing : 0
      if (expected > 0) {
        sum += Math.abs(d - expected)
        n++
      }
    }
  }
  return n > 0 ? sum / n : 0
}

export function extractDancerHistory(team: TeamPoseFrame[], dancerId: number): KeypointMap[] {
  return team.map((f) => {
    const d = f.dancers.find((x) => x.id === dancerId)
    return d?.keypoints ?? {}
  })
}
