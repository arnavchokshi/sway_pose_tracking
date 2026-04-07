// @ts-nocheck
import {
  Fn,
  If,
  Loop,
  float,
  uint,
  vec3,
  uniform,
  instancedArray,
  instanceIndex,
  atomicAdd,
  atomicStore,
  atomicLoad,
  hash,
} from 'three/tsl'

export class BallPhysicsGPU {
  #forces
  #dtUniform
  #computeInit
  #computeClearGrid
  #computeBuildGrid
  #computeForces
  #computeIntegrate
  #computeCorrection

  constructor({
    renderer,
    count,
    radius,
    bounds = { x: 12, z: 12, y: 12 },
    maxPerCell = 4,
    gravity = -20,
    stiffness = 4000,
    damping = 25,
    friction = 0.4,
    restitution = 0.3,
    cohesion = 0,
    correctionStrength = 0.5,
    correctionDamping = 0.3,
  }) {
    this.renderer = renderer
    this.count = count
    this.radius = radius
    this.diameter = radius * 2
    this.cohesionRange = radius * 4
    this.bounds = bounds
    this.maxPerCell = maxPerCell

    this.gridResX = Math.ceil(bounds.x / this.diameter)
    this.gridResY = Math.ceil(bounds.y / this.diameter)
    this.gridResZ = Math.ceil(bounds.z / this.diameter)
    this.gridTotal = this.gridResX * this.gridResY * this.gridResZ

    // Exact baseline from your shared code
    this.gridOriginX = -bounds.x / 2
    this.gridOriginY = 0
    this.gridOriginZ = -bounds.z / 2

    this.positions = instancedArray(count, 'vec3')
    this.velocities = instancedArray(count, 'vec3')
    this.#forces = instancedArray(count, 'vec3')
    this.externalForces = instancedArray(count, 'vec3')
    this.gridCounters = instancedArray(this.gridTotal, 'uint').toAtomic()
    this.gridParticles = instancedArray(this.gridTotal * maxPerCell, 'uint')

    this.#dtUniform = uniform(0)
    this.gravityUniform = uniform(gravity)
    this.stiffnessUniform = uniform(stiffness)
    this.dampingUniform = uniform(damping)
    this.frictionUniform = uniform(friction)
    this.restitutionUniform = uniform(restitution)
    this.cohesionUniform = uniform(cohesion)
    this.correctionStrengthUniform = uniform(correctionStrength)
    this.correctionDampingUniform = uniform(correctionDamping)

    this.#computeInit = this.#buildInitShader()
    this.#computeClearGrid = this.#buildClearGridShader()
    this.#computeBuildGrid = this.#buildBuildGridShader()
    this.#computeForces = this.#buildForcesShader()
    this.#computeIntegrate = this.#buildIntegrateShader()
    this.#computeCorrection = this.#buildCorrectionShader()
  }

  async init() {
    await this.renderer.computeAsync(this.#computeInit)
  }

  compute(dt, { correctionPass = true } = {}) {
    this.#dtUniform.value = dt
    this.renderer.compute(this.#computeClearGrid)
    this.renderer.compute(this.#computeBuildGrid)
    this.renderer.compute(this.#computeForces)
    this.renderer.compute(this.#computeIntegrate)
    if (correctionPass) {
      this.renderer.compute(this.#computeCorrection)
    }
  }

  #buildInitShader() {
    const { count, radius, diameter, bounds, positions, velocities } = this

    return Fn(() => {
      const pos = positions.element(instanceIndex)
      const vel = velocities.element(instanceIndex)

      const spreadX = float(bounds.x - diameter * 2)
      const spreadZ = float(bounds.z - diameter * 2)

      pos.x.assign(hash(instanceIndex).mul(spreadX).sub(spreadX.mul(0.5)))
      const layersNeeded = Math.ceil(count / ((bounds.x / diameter) * (bounds.z / diameter)))
      pos.y.assign(hash(instanceIndex.add(count)).mul(layersNeeded * diameter).add(radius))
      pos.z.assign(hash(instanceIndex.add(count * 2)).mul(spreadZ).sub(spreadZ.mul(0.5)))

      vel.assign(vec3(0))
    })().compute(count)
  }

  #buildClearGridShader() {
    const { gridTotal } = this
    const gridCounters = this.gridCounters

    return Fn(() => {
      atomicStore(gridCounters.element(instanceIndex), uint(0))
    })().compute(gridTotal)
  }

  #buildBuildGridShader() {
    const {
      count,
      diameter,
      maxPerCell,
      gridResX,
      gridResY,
      gridResZ,
      gridOriginX,
      gridOriginY,
      gridOriginZ,
      positions,
    } = this
    const gridCounters = this.gridCounters
    const gridParticles = this.gridParticles

    return Fn(() => {
      const pos = positions.element(instanceIndex)

      const cx = pos.x.sub(gridOriginX).div(diameter).floor().toInt()
      const cy = pos.y.sub(gridOriginY).div(diameter).floor().toInt()
      const cz = pos.z.sub(gridOriginZ).div(diameter).floor().toInt()

      If(
        cx
          .greaterThanEqual(0)
          .and(cx.lessThan(gridResX))
          .and(cy.greaterThanEqual(0))
          .and(cy.lessThan(gridResY))
          .and(cz.greaterThanEqual(0))
          .and(cz.lessThan(gridResZ)),
        () => {
          const cellIdx = cx.add(cy.mul(gridResX)).add(cz.mul(gridResX * gridResY))
          const slot = atomicAdd(gridCounters.element(cellIdx), uint(1))

          If(slot.lessThan(uint(maxPerCell)), () => {
            gridParticles.element(cellIdx.mul(maxPerCell).add(slot.toInt())).assign(instanceIndex)
          })
        },
      )
    })().compute(count)
  }

  #buildForcesShader() {
    const {
      count,
      diameter,
      cohesionRange,
      maxPerCell,
      gridResX,
      gridResY,
      gridResZ,
      gridOriginX,
      gridOriginY,
      gridOriginZ,
      positions,
      velocities,
      stiffnessUniform,
      dampingUniform,
      cohesionUniform,
    } = this
    const forces = this.#forces
    const gridCounters = this.gridCounters
    const gridParticles = this.gridParticles

    return Fn(() => {
      const myPos = positions.element(instanceIndex)
      const myVel = velocities.element(instanceIndex)
      const force = forces.element(instanceIndex)

      force.assign(vec3(0))

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

                If(otherId.notEqual(instanceIndex), () => {
                  const otherPos = positions.element(otherId)
                  const diff = myPos.sub(otherPos)
                  const distSq = diff.dot(diff)
                  const maxRangeSq = float(cohesionRange * cohesionRange)

                  If(distSq.lessThan(maxRangeSq).and(distSq.greaterThan(0.00000001)), () => {
                    const dist = distSq.sqrt()
                    const normal = diff.div(dist)

                    If(dist.lessThan(diameter), () => {
                      const penetration = float(diameter).sub(dist)
                      force.addAssign(normal.mul(stiffnessUniform.mul(penetration)))

                      const otherVel = velocities.element(otherId)
                      const relVel = myVel.sub(otherVel)
                      force.subAssign(normal.mul(dampingUniform.mul(relVel.dot(normal))))
                    })

                    If(dist.greaterThanEqual(diameter).and(dist.lessThan(cohesionRange)), () => {
                      const t = float(1).sub(dist.sub(diameter).div(cohesionRange - diameter))
                      const falloff = t.mul(t).mul(float(3).sub(t.mul(2)))
                      force.subAssign(normal.mul(cohesionUniform.mul(falloff)))

                      const otherVelC = velocities.element(otherId)
                      const relVelC = myVel.sub(otherVelC)
                      force.subAssign(normal.mul(float(10).mul(relVelC.dot(normal)).mul(falloff)))
                    })
                  })
                })
              })
            })
          },
        )
      })
    })().compute(count)
  }

  #buildIntegrateShader() {
    const {
      count,
      radius,
      bounds,
      positions,
      velocities,
      externalForces,
      gravityUniform,
      frictionUniform,
      restitutionUniform,
    } = this
    const forces = this.#forces
    const dtUniform = this.#dtUniform

    return Fn(() => {
      const pos = positions.element(instanceIndex)
      const vel = velocities.element(instanceIndex)
      const force = forces.element(instanceIndex)
      const extForce = externalForces.element(instanceIndex)
      const dt = dtUniform

      vel.addAssign(force.mul(dt))
      vel.addAssign(extForce.mul(dt))
      vel.y.addAssign(gravityUniform.mul(dt))
      vel.mulAssign(0.999)

      const speed = vel.length()
      If(speed.greaterThan(20), () => {
        vel.assign(vel.div(speed).mul(20))
      })

      pos.addAssign(vel.mul(dt))

      const r = float(radius)
      If(pos.y.lessThan(r), () => {
        pos.y.assign(r)
        If(vel.y.lessThan(0), () => {
          vel.y.assign(vel.y.negate().mul(restitutionUniform))
          vel.x.mulAssign(float(1).sub(frictionUniform))
          vel.z.mulAssign(float(1).sub(frictionUniform))
        })
      })

      const wallMinX = float(-bounds.x / 2).add(r)
      const wallMaxX = float(bounds.x / 2).sub(r)
      const wallMinZ = float(-bounds.z / 2).add(r)
      const wallMaxZ = float(bounds.z / 2).sub(r)
      const wallMaxY = float(bounds.y).sub(r)

      If(pos.x.lessThan(wallMinX), () => {
        pos.x.assign(wallMinX)
        If(vel.x.lessThan(0), () => {
          vel.x.assign(vel.x.negate().mul(restitutionUniform))
        })
      })
      If(pos.x.greaterThan(wallMaxX), () => {
        pos.x.assign(wallMaxX)
        If(vel.x.greaterThan(0), () => {
          vel.x.assign(vel.x.negate().mul(restitutionUniform))
        })
      })
      If(pos.z.lessThan(wallMinZ), () => {
        pos.z.assign(wallMinZ)
        If(vel.z.lessThan(0), () => {
          vel.z.assign(vel.z.negate().mul(restitutionUniform))
        })
      })
      If(pos.z.greaterThan(wallMaxZ), () => {
        pos.z.assign(wallMaxZ)
        If(vel.z.greaterThan(0), () => {
          vel.z.assign(vel.z.negate().mul(restitutionUniform))
        })
      })
      If(pos.y.greaterThan(wallMaxY), () => {
        pos.y.assign(wallMaxY)
        If(vel.y.greaterThan(0), () => {
          vel.y.assign(vel.y.negate().mul(restitutionUniform))
        })
      })
    })().compute(count)
  }

  #buildCorrectionShader() {
    const {
      count,
      radius,
      diameter,
      bounds,
      maxPerCell,
      gridResX,
      gridResY,
      gridResZ,
      gridOriginX,
      gridOriginY,
      gridOriginZ,
      positions,
      velocities,
      correctionStrengthUniform,
      correctionDampingUniform,
    } = this
    const gridCounters = this.gridCounters
    const gridParticles = this.gridParticles

    return Fn(() => {
      const myPos = positions.element(instanceIndex)
      const vel = velocities.element(instanceIndex)
      const correction = vec3(0).toVar()
      const neighborCount = float(0).toVar()

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

                If(otherId.notEqual(instanceIndex), () => {
                  const otherPos = positions.element(otherId)
                  const diff = myPos.sub(otherPos)
                  const distSq = diff.dot(diff)

                  If(distSq.lessThan(diameter * diameter).and(distSq.greaterThan(0.00000001)), () => {
                    const dist = distSq.sqrt()
                    const penetration = float(diameter).sub(dist)
                    const normal = diff.div(dist)
                    correction.addAssign(normal.mul(penetration.mul(correctionStrengthUniform)))
                    neighborCount.addAssign(1)
                  })
                })
              })
            })
          },
        )
      })

      If(neighborCount.greaterThan(0), () => {
        const avgCorrection = correction.div(neighborCount)
        myPos.addAssign(avgCorrection)

        const corrLen = avgCorrection.length()
        If(corrLen.greaterThan(0.00001), () => {
          const corrDir = avgCorrection.div(corrLen)
          const velAlongCorr = vel.dot(corrDir)
          If(velAlongCorr.lessThan(0), () => {
            vel.subAssign(corrDir.mul(velAlongCorr.mul(correctionDampingUniform)))
          })
        })
      })

      const r = float(radius)
      const wallMinX = float(-bounds.x / 2).add(r)
      const wallMaxX = float(bounds.x / 2).sub(r)
      const wallMinZ = float(-bounds.z / 2).add(r)
      const wallMaxZ = float(bounds.z / 2).sub(r)
      const wallMaxY = float(bounds.y).sub(r)

      If(myPos.y.lessThan(r), () => {
        myPos.y.assign(r)
      })
      If(myPos.x.lessThan(wallMinX), () => {
        myPos.x.assign(wallMinX)
      })
      If(myPos.x.greaterThan(wallMaxX), () => {
        myPos.x.assign(wallMaxX)
      })
      If(myPos.z.lessThan(wallMinZ), () => {
        myPos.z.assign(wallMinZ)
      })
      If(myPos.z.greaterThan(wallMaxZ), () => {
        myPos.z.assign(wallMaxZ)
      })
      If(myPos.y.greaterThan(wallMaxY), () => {
        myPos.y.assign(wallMaxY)
      })
    })().compute(count)
  }
}
