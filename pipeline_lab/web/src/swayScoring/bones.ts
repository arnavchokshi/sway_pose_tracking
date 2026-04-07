import type { BoneEdge } from './types'

/** Standard COCO-17 limb pairs for drawing / vector math. */
export const COCO_BONE_EDGES: BoneEdge[] = [
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
  ['nose', 'left_eye'],
  ['nose', 'right_eye'],
  ['left_eye', 'left_ear'],
  ['right_eye', 'right_ear'],
]

export const EXTREMITY_JOINTS = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle'] as const

/** Bones that include `joint` as an endpoint (for partial ghost draw). */
export function bonesTouchingJoint(joint: string, edges: BoneEdge[] = COCO_BONE_EDGES): BoneEdge[] {
  return edges.filter(([a, b]) => a === joint || b === joint)
}

export function jointSetFromSpatialErrors(joints: string[]): Set<string> {
  return new Set(joints.map((j) => j.toLowerCase()))
}

/** All edges that should be drawn when these joints have high error. */
export function edgesForErrorJoints(
  errorJointNames: string[],
  edges: BoneEdge[] = COCO_BONE_EDGES,
): BoneEdge[] {
  const err = jointSetFromSpatialErrors(errorJointNames)
  const out: BoneEdge[] = []
  for (const e of edges) {
    if (err.has(e[0].toLowerCase()) || err.has(e[1].toLowerCase())) {
      out.push(e)
    }
  }
  return out
}
