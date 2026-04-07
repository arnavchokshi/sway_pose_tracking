// @ts-nocheck
import { Fn, If, Loop, float, vec3, uniform, instanceIndex, atomicLoad, instancedArray } from 'three/tsl'

export class SphereGridAO {
  intensityUniform: any
  aoRadiusUniform: any
  aoBuffer: any
  computeAO: any

  constructor({ physics, count, radius, intensity = 1, aoRadius = 0.3 }: any) {
    this.intensityUniform = uniform(intensity)
    this.aoRadiusUniform = uniform(aoRadius)
    const radiusSq = radius * radius

    const {
      positions,
      gridCounters,
      gridParticles,
      diameter,
      maxPerCell,
      gridResX,
      gridResY,
      gridResZ,
      gridOriginX,
      gridOriginY,
      gridOriginZ,
    } = physics

    // Per-instance: xyz = weighted occlusion direction, w = total occlusion weight
    const aoBuffer = instancedArray(count, 'vec4')
    this.aoBuffer = aoBuffer
    const aoRadiusU = this.aoRadiusUniform

    // Compute shader: per-instance directional AO using physics grid
    this.computeAO = Fn(() => {
      const myPos = positions.element(instanceIndex)
      const occDir = vec3(0, 0, 0).toVar()
      const occWeight = float(0).toVar()

      const cx = myPos.x.sub(gridOriginX).div(diameter).floor().toInt()
      const cy = myPos.y.sub(gridOriginY).div(diameter).floor().toInt()
      const cz = myPos.z.sub(gridOriginZ).div(diameter).floor().toInt()

      Loop(27, ({ i: n }) => {
        const dx = n.toInt().mod(3).sub(1)
        const dy = n.toInt().div(3).mod(3).sub(1)
        const dz = n.toInt().div(9).sub(1)

        const nx = cx.add(dx)
        const ny = cy.add(dy)
        const nz = cz.add(dz)

        If(
          nx
            .greaterThanEqual(0)
            .and(nx.lessThan(gridResX))
            .and(ny.greaterThanEqual(0))
            .and(ny.lessThan(gridResY))
            .and(nz.greaterThanEqual(0))
            .and(nz.lessThan(gridResZ)),
          () => {
            const cellIdx = nx.add(ny.mul(gridResX)).add(nz.mul(gridResX * gridResY))
            const cellCount = atomicLoad(gridCounters.element(cellIdx))

            Loop(maxPerCell, ({ i: s }) => {
              If(s.toUint().lessThan(cellCount), () => {
                const otherId = gridParticles.element(cellIdx.mul(maxPerCell).add(s))

                If(otherId.toInt().notEqual(instanceIndex.toInt()), () => {
                  const otherPos = positions.element(otherId)
                  const diff = myPos.sub(otherPos)
                  const distSq = diff.dot(diff)
                  const aoRadSq = aoRadiusU.mul(aoRadiusU)

                  If(distSq.lessThan(aoRadSq).and(distSq.greaterThan(radiusSq)), () => {
                    const dist = distSq.sqrt()
                    const t = float(1.0).sub(dist.div(aoRadiusU))
                    const occ = t.mul(t)
                    // Accumulate direction toward neighbor, weighted by occlusion
                    occDir.addAssign(diff.negate().normalize().mul(occ))
                    occWeight.addAssign(occ)
                  })
                })
              })
            })
          },
        )
      })

      aoBuffer.element(instanceIndex).assign(vec3(occDir).toVec4(occWeight))
    })().compute(count)
  }

  update(renderer: any) {
    renderer.compute(this.computeAO)
  }
}
