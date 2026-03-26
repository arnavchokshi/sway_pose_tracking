import { edgesForErrorJoints } from './bones'
import type { BoneEdge } from './types'

type DrawOpts = {
  ctx: CanvasRenderingContext2D
  /** Content rect inside canvas (letterboxed video area). */
  contentLeft: number
  contentTop: number
  contentW: number
  contentH: number
  /** Normalized [0,1] keypoints */
  keypointsNorm: Record<string, [number, number]>
  /** Only draw bones connected to these joints (spatial errors). */
  errorJointNames: string[]
  glowColor?: string
  lineWidth?: number
}

/**
 * Draw semi-transparent cyan "ghost" skeleton for erroneous limbs only.
 */
export function drawGhostSkeletonPartial(opts: DrawOpts): void {
  const {
    ctx,
    contentLeft,
    contentTop,
    contentW,
    contentH,
    keypointsNorm,
    errorJointNames,
    glowColor = 'rgba(34, 211, 238, 0.55)',
    lineWidth = 4,
  } = opts

  if (errorJointNames.length === 0) return

  const edges = edgesForErrorJoints(errorJointNames)

  const toCanvas = (nx: number, ny: number): [number, number] => {
    return [contentLeft + nx * contentW, contentTop + ny * contentH]
  }

  ctx.save()
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.strokeStyle = glowColor
  ctx.lineWidth = lineWidth
  ctx.shadowColor = 'rgba(34, 211, 238, 0.9)'
  ctx.shadowBlur = 18

  for (const [from, to] of edges) {
    const pa = keypointsNorm[from]
    const pb = keypointsNorm[to]
    if (!pa || !pb) continue
    const [x0, y0] = toCanvas(pa[0], pa[1])
    const [x1, y1] = toCanvas(pb[0], pb[1])
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.lineTo(x1, y1)
    ctx.stroke()
  }

  ctx.shadowBlur = 0
  for (const [from, to] of edges) {
    const pa = keypointsNorm[from]
    const pb = keypointsNorm[to]
    if (!pa || !pb) continue
    const [x0, y0] = toCanvas(pa[0], pa[1])
    const [x1, y1] = toCanvas(pb[0], pb[1])
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.lineTo(x1, y1)
    ctx.strokeStyle = 'rgba(255,255,255,0.35)'
    ctx.lineWidth = Math.max(1, lineWidth - 2)
    ctx.stroke()
  }

  ctx.restore()
}

/** Full skeleton (e.g. debug) — all edges that have both keypoints. */
export function drawGhostSkeletonFull(
  ctx: CanvasRenderingContext2D,
  contentLeft: number,
  contentTop: number,
  contentW: number,
  contentH: number,
  keypointsNorm: Record<string, [number, number]>,
  edges: BoneEdge[],
  glowColor = 'rgba(34, 211, 238, 0.45)',
): void {
  drawGhostSkeletonPartial({
    ctx,
    contentLeft,
    contentTop,
    contentW,
    contentH,
    keypointsNorm,
    errorJointNames: edges.flatMap((e) => [e[0], e[1]]),
    glowColor,
  })
}
