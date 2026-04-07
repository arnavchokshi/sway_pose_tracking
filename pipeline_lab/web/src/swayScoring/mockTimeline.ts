import type { DancerFramePayload, ScoringFramePayload } from './types'

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Toy normalized keypoints: standing T-pose-ish, jittered per dancer. */
function baseGoldenNorm(seed: number): Record<string, [number, number]> {
  const ox = 0.02 * Math.sin(seed * 1.7)
  const oy = 0.02 * Math.cos(seed * 1.3)
  const cx = 0.5 + ox
  const headY = 0.22 + oy
  const shoulderY = 0.3 + oy
  const hipY = 0.52 + oy
  const kneeY = 0.68 + oy
  const ankleY = 0.9 + oy
  return {
    nose: [cx, headY - 0.02],
    left_eye: [cx - 0.02, headY - 0.025],
    right_eye: [cx + 0.02, headY - 0.025],
    left_ear: [cx - 0.04, headY],
    right_ear: [cx + 0.04, headY],
    left_shoulder: [cx - 0.12, shoulderY],
    right_shoulder: [cx + 0.12, shoulderY],
    left_elbow: [cx - 0.18, shoulderY + 0.12],
    right_elbow: [cx + 0.18, shoulderY + 0.12],
    left_wrist: [cx - 0.2, shoulderY + 0.28],
    right_wrist: [cx + 0.2, shoulderY + 0.28],
    left_hip: [cx - 0.06, hipY],
    right_hip: [cx + 0.06, hipY],
    left_knee: [cx - 0.07, kneeY],
    right_knee: [cx + 0.07, kneeY],
    left_ankle: [cx - 0.08, ankleY],
    right_ankle: [cx + 0.08, ankleY],
  }
}

function distortArm(
  kp: Record<string, [number, number]>,
  side: 'left' | 'right',
  amount: number,
): Record<string, [number, number]> {
  const elbow = `${side}_elbow` as const
  const wrist = `${side}_wrist` as const
  const out = { ...kp }
  if (out[elbow]) {
    out[elbow] = [out[elbow]![0], out[elbow]![1] + amount * 0.08]
  }
  if (out[wrist]) {
    out[wrist] = [out[wrist]![0] + amount * 0.04, out[wrist]![1] + amount * 0.12]
  }
  return out
}

export function buildMockScoringTimeline(opts: {
  durationSec: number
  fps: number
  dancerIds?: number[]
}): ScoringFramePayload[] {
  const fps = opts.fps
  const durationSec = opts.durationSec
  const dancerIds = opts.dancerIds ?? [1, 2, 3, 4]
  const totalFrames = Math.max(1, Math.round(durationSec * fps))
  const frames: ScoringFramePayload[] = []

  for (let i = 0; i < totalFrames; i++) {
    const t = i / fps
    const u = i / Math.max(1, totalFrames - 1)
    const wave = 0.5 + 0.5 * Math.sin(u * Math.PI * 4 + 0.3)
    const dip = u > 0.35 && u < 0.5 ? -22 : 0
    const globalScore = Math.round(lerp(55, 96, wave) + dip)

    const dancers: DancerFramePayload[] = dancerIds.map((id, di) => {
      const personal = globalScore - di * 6 - (id === 3 ? 12 : 0)
      const agg = Math.max(40, Math.min(100, Math.round(personal)))
      const golden = baseGoldenNorm(id + i * 0.01)

      const rightArmBad = id === 3 && (u > 0.25 && u < 0.65)
      const goldenDraw = rightArmBad ? distortArm(golden, 'right', 1) : golden

      const spatial_errors = rightArmBad
        ? [
            {
              joint: 'right_elbow',
              error_deg: -25,
              msg: 'Right arm too low',
            },
            {
              joint: 'right_wrist',
              error_deg: 18,
              msg: 'Raise your right arm higher on the hit.',
            },
          ]
        : []

      const temporal_error =
        id === 3 && Math.abs(u - 0.42) < 0.02
          ? { offset_sec: -0.3, msg: 'Slightly late' }
          : null

      const formation_error =
        id === 2 && Math.abs(u - 0.55) < 0.015
          ? { offset_x: -0.5, offset_y: 0.1, msg: 'Shift right' }
          : null

      const col = di % 2
      const row = Math.floor(di / 2)
      const bx = 0.18 + col * 0.28 + id * 0.01
      const by = 0.12 + row * 0.38

      return {
        id,
        aggregate_score: agg,
        spatial_errors,
        temporal_error,
        formation_error,
        bbox_norm: { x: bx, y: by, w: 0.14, h: 0.42 },
        golden_keypoints_norm: goldenDraw,
      }
    })

    const formation_targets: Record<number, { x: number; y: number }> = {}
    const formation_actual: Record<number, { x: number; y: number }> = {}
    dancerIds.forEach((id, di) => {
      const col = di % 2
      const row = Math.floor(di / 2)
      formation_targets[id] = {
        x: 0.2 + col * 0.35,
        y: 0.2 + row * 0.35,
      }
      const jitter = 0.04 * Math.sin(u * 8 + id)
      formation_actual[id] = {
        x: formation_targets[id]!.x + jitter + (id === 2 ? 0.08 : 0),
        y: formation_targets[id]!.y + jitter * 0.5,
      }
    })

    frames.push({
      frame_idx: i,
      timestamp: t,
      global_team_score: globalScore,
      dancers,
      formation_targets,
      formation_actual,
    })
  }

  return frames
}
