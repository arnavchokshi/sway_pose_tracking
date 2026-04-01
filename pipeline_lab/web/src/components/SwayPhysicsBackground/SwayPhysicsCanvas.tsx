// @ts-nocheck
import { useEffect, useRef } from 'react'
import * as THREE from 'three/webgpu'
import {
  Fn,
  If,
  color,
  instanceIndex,
  positionLocal,
  uniform,
  instancedArray,
  storage,
  struct,
  mix,
  smoothstep,
  vec3,
  vec4,
  float,
  uint,
  abs,
  pow,
  sign,
  normalWorld,
  mx_noise_float,
  atomicAdd,
  atomicStore,
  cos,
  sin,
  vec2,
  min,
  length,
  clamp,
} from 'three/tsl'

import { FixedTimestep } from './FixedTimestep'
import { BallPhysicsGPU } from './BallPhysicsGPU'
import { SphereGridAO } from './SphereGridAO'

interface SwayPhysicsCanvasProps {
  debug?: boolean
  sphereCount?: number
  geometryDetail?: number
  glowPattern?: number
  color1?: string
  color2?: string
}

export function SwayPhysicsCanvas({ 
  debug = false, 
  sphereCount = 21000, 
  geometryDetail = 0, 
  glowPattern = 0,
  color1 = '#6366f1',
  color2 = '#22d3ee'
}: SwayPhysicsCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null)



  useEffect(() => {
    if (!containerRef.current) return

    let mounted = true
    const container = containerRef.current

    // ─── Constants & Params ──────────────────────────────────────────────────
    const SPHERE_COUNT = sphereCount
    const SPHERE_RADIUS = 0.08
    const BOUNDS = { x: 20, z: 15, y: 12 }

    const params = {
      debug,
      gravity: -20,
      stiffness: 4000,
      damping: 25,
      friction: 0.4,
      restitution: 0.3,
      cohesion: 0,
      correctionStrength: 0.5,
      correctionDamping: 0.3,
      correctionPass: true,
      paused: false,
      visualLerp: 0.5,
      noiseAmplitude: 25,
      noiseFrequency: 0.4,
      noiseSpeed: 0.07,
      attractorHeight: 0.4,
      attractorRadius: 1.5,
      attractorStrength: 40,
      fov: 30,
      backgroundColor: '#040610', // From Singularity HTML
      ballColor: '#11111a',
      metalness: 0.1,
      materialType: 'standard',
      ambientType: 'discs',
      environmentIntensity: 0.15,
      lightIntensity: 0.7,
      roughness: 0.8,
      shadows: true,
      shadowType: 'vsm',
      shadowBias: -0.001,
      shadowNormalBias: 0.02,
      shadowMapSize: 1024,
      lightX: 5,
      lightY: 3,
      lightZ: -3,
      emissivePower: 5,
      emissiveEnabled: true,
      emissivePaused: false,
      // User dynamic accent colors
      emissiveColor1: color1,
      emissiveColor2: color2,
      emissiveIntensity: 4.5,
      emissiveAoIntensity: 0.2,
      emissiveAoRadius: 1.5,
      emissiveSpeed: 0.3,
      emissiveScale: 0.08, // Widened from 0.05 so the core color actually has pixels to draw onto
      emissiveMin: -0.3,
      emissiveMax: 0.9,
      bounce: true,
      bounceIntensity: 0.2,
      bounceScale: 0.26,
      bouncePower: 4,
      ao: true,
      aoIntensity: 0.5,
      aoRadius: 0.3,
      culling: true,
      lodDistanceNear: 4,
      lodDistanceFar: 8,
      colliderRadius: 1,
      colliderStrength: 200,
      colliderDistance: -4.5,
      colliderY: 0.33,
      pointLightIntensity: 0.6,
      pointLightColor: '#4a7cff',
      pointLightRange: 0,
      pointLightDecay: 3.6,
      pointLightDistance: -3.28,
      pointLightY: 0.1,
      collider2Radius: 0.7,
      collider2Strength: 200,
      lightFadeIn: 0.3,
      lightFadeDelay: 1.2,
    }

    // ─── Scene setup ───────────────────────────────────────────────────────────
    const renderer = new THREE.WebGPURenderer({ antialias: true, alpha: true })
    renderer.setClearColor(0x000000, 0)
    renderer.toneMapping = THREE.AgXToneMapping
    renderer.toneMappingExposure = 1.0
    renderer.shadowMap.enabled = params.shadows
    renderer.shadowMap.type = THREE.VSMShadowMap
    
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.domElement.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;z-index:2;pointer-events:none;'
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    // Background removed to allow CSS background & underlying elements (confetti) to show through

    const camera = new THREE.PerspectiveCamera(params.fov, window.innerWidth / window.innerHeight, 0.01, 200)
    // Restoring deep 3D perspective by slightly angling down instead of tilting up! 
    camera.position.set(0, 4.0, 16)
    
    // Looking slightly low keeps the horizon centered and spheres strictly taking up the dense bottom half natively
    // Nudged to 1.60 to physically shift the floor graphic further down (total ~60px drop from original)
    const cameraTarget = new THREE.Vector3(0, 1.60, 0)
    camera.lookAt(cameraTarget)


    // ─── Elegant Infinite Floor (if Stage Mode enabled) ──────────────────────────
    // ─── Lights ────────────────────────────────────────────────────────────────
    scene.environmentIntensity = params.environmentIntensity

    const ambientLight = new THREE.AmbientLight('#222233', 0)
    scene.add(ambientLight)

    const rimLight = new THREE.DirectionalLight('#ffffff', 0)
    rimLight.position.set(params.lightX, params.lightY, params.lightZ)
    rimLight.castShadow = params.shadows
    rimLight.shadow.mapSize.setScalar(params.shadowMapSize)
    rimLight.shadow.bias = params.shadowBias
    rimLight.shadow.normalBias = params.shadowNormalBias
    rimLight.shadow.camera.near = 0.1
    rimLight.shadow.camera.far = 30
    rimLight.shadow.camera.left = -BOUNDS.x / 2
    rimLight.shadow.camera.right = BOUNDS.x / 2
    rimLight.shadow.camera.top = BOUNDS.z / 2
    rimLight.shadow.camera.bottom = -BOUNDS.z / 2
    scene.add(rimLight)

    // ─── Physics ──────────────────────────────────────────────────────────────
    const physics = new BallPhysicsGPU({
      renderer,
      count: SPHERE_COUNT,
      radius: SPHERE_RADIUS,
      bounds: BOUNDS,
      maxPerCell: 4,
      gravity: params.gravity,
      stiffness: params.stiffness,
      damping: params.damping,
      friction: params.friction,
      restitution: params.restitution,
      cohesion: params.cohesion,
      correctionStrength: params.correctionStrength,
      correctionDamping: params.correctionDamping,
    })

    const ao = new SphereGridAO({
      physics,
      count: SPHERE_COUNT,
      radius: SPHERE_RADIUS,
      intensity: params.aoIntensity,
      aoRadius: params.aoRadius,
    })

    const visualPositions = instancedArray(SPHERE_COUNT, 'vec3')
    const visualLerpUniform = uniform(0.1)

    const lerpVisualPositions = Fn(() => {
      const target = physics.positions.element(instanceIndex)
      const current = visualPositions.element(instanceIndex)
      visualPositions.element(instanceIndex).assign(mix(current, target, visualLerpUniform))
    })().compute(SPHERE_COUNT)

    async function snapVisualPositions() {
      visualLerpUniform.value = 1
      await renderer.computeAsync(lerpVisualPositions)
      visualLerpUniform.value = params.visualLerp
    }

    // ─── External forces (noise + attractor) ──────────────────────────────────
    const noiseAmplitudeUniform = uniform(params.noiseAmplitude)
    const noiseFrequencyUniform = uniform(params.noiseFrequency)
    const noiseSpeedUniform = uniform(params.noiseSpeed)
    const glowPatternUniform = uniform(glowPattern)
    const timeUniform = uniform(0)
    const NOISE_TIME_OFFSET = 270
    const NOISE_SCALE = 0.5
    const NOISE_BIAS = 0.275

    function noiseSamplePos(px: any, pz: any) {
      return vec3(
        px.mul(noiseFrequencyUniform),
        pz.mul(noiseFrequencyUniform),
        timeUniform.mul(noiseSpeedUniform).add(NOISE_TIME_OFFSET),
      )
    }

    function sampleNoise(px: any, pz: any) {
      const mode = glowPatternUniform.toInt()
      const time = timeUniform.mul(noiseSpeedUniform).add(NOISE_TIME_OFFSET)
      const raw = float(0).toVar()

      If(mode.equal(0), () => {
        // Blob / Noise (Original)
        raw.assign(mx_noise_float(noiseSamplePos(px, pz)))
      }).Else(() => {
        If(mode.equal(1), () => {
          // Snaking Pattern (Sine wave on X * Cos Z)
          raw.assign(sin(px.mul(noiseFrequencyUniform.mul(6)).add(time)).mul(cos(pz.mul(noiseFrequencyUniform.mul(8)).add(time))))
        }).Else(() => {
          If(mode.equal(2), () => {
            // Concentric Rings / Ripples
            const dist = vec2(px, pz).length()
            raw.assign(sin(dist.mul(noiseFrequencyUniform.mul(8)).sub(time.mul(4))))
          }).Else(() => {
            If(mode.equal(3), () => {
              // Scanline Sweep
              raw.assign(sin(pz.mul(noiseFrequencyUniform.mul(10)).add(time.mul(3))))
            }).Else(() => {
              // Interference Grid (Default / 4)
              const waveX = sin(px.mul(noiseFrequencyUniform.mul(12)).add(time.mul(2)))
              const waveZ = sin(pz.mul(noiseFrequencyUniform.mul(12)).add(time.mul(2)))
              raw.assign(waveX.add(waveZ).mul(0.5)) // Normalize [-2, 2] to [-1, 1]
            })
          })
        })
      })

      return raw.mul(NOISE_SCALE).add(NOISE_BIAS)
    }

    const attractorPosUniform = uniform(vec3(0, 0, 0))
    const attractorRadiusUniform = uniform(params.attractorRadius)
    const attractorStrengthUniform = uniform(params.attractorStrength)
    const attractorActiveUniform = uniform(0)
    const colliderPosUniform = uniform(vec3(0, 0, 0))
    const colliderRadiusUniform = uniform(params.colliderRadius)
    const colliderStrengthUniform = uniform(params.colliderStrength)

    const collider2PosUniform = uniform(vec3(0, 0, 0))
    const collider2RadiusUniform = uniform(params.collider2Radius)
    const collider2StrengthUniform = uniform(params.collider2Strength)

    // --- Spotlight Formations Uniforms ---
    const currentFormationUniform = uniform(0)
    const nextFormationUniform = uniform(1)
    const wipeThresholdUniform = uniform(0)

    const getMask = Fn(([x_param, z_param, typeIndex]) => {
      const x = float(x_param)
      const z = float(z_param)
      
      // Massively spacing out the formation performers! 
      const gridSpacing = float(2.2)
      // Safely offset by 100 to avoid negative mod issues
      const localX = x.add(100.0).mod(gridSpacing).sub(gridSpacing.div(2.0))
      const localZ = z.add(100.0).mod(gridSpacing).sub(gridSpacing.div(2.0))
      const distToSpotlight = length(vec2(localX, localZ))
      
      // We dramatically tighten the radius (0.35) and use a rock-hard mathematical cutoff instead of a smooth fade.
      // Since the spheres are densely packed, this acts as a 'laser pointer' perfectly targeting only the 1-2 spheres in the exact dead-center!
      const isSpotlight = distToSpotlight.lessThan(0.35).select(1.0, 0.0)
      
      const val = float(0).toVar()
      
      If(isSpotlight.greaterThan(0), () => {
        If(typeIndex.equal(0), () => {
          // V shape (Cheer/Dance classic)
          const dist = abs(z.sub(abs(x.mul(1.2)).sub(3.0)))
          val.assign(dist.lessThan(1.8).select(1.0, 0.0))
        }).ElseIf(typeIndex.equal(1), () => {
          // Core Block (Tight squad)
          const isBlock = abs(x).lessThan(5.0).and(abs(z).lessThan(3.5))
          val.assign(isBlock.select(1.0, 0.0))
        }).ElseIf(typeIndex.equal(2), () => {
          // X-Shape (Crossing Diagonals)
          const distToX = abs(abs(x.mul(1.2)).sub(abs(z)))
          val.assign(distToX.lessThan(1.8).select(1.0, 0.0))
        }).ElseIf(typeIndex.equal(3), () => {
          // Three Columns (Marching band style)
          const centerCol = abs(x).lessThan(1.5)
          const sideCols = abs(abs(x).sub(6.0)).lessThan(1.5)
          const validZ = abs(z).lessThan(6.0)
          val.assign(centerCol.or(sideCols).and(validZ).select(1.0, 0.0))
        })
      })
      
      return val.mul(isSpotlight)
    })

    const computeExternalForces = Fn(() => {
      const pos = physics.positions.element(instanceIndex)
      const extForce = physics.externalForces.element(instanceIndex)

      const noiseVal = sampleNoise(pos.x, pos.z)
      const force = vec3(0, noiseVal.mul(noiseAmplitudeUniform), 0).toVar()

      const toAttractor = attractorPosUniform.sub(pos).toVar()
      const dist = toAttractor.length()
      const falloff = dist.div(attractorRadiusUniform).clamp(0, 1).oneMinus().mul(attractorActiveUniform)
      force.addAssign(toAttractor.normalize().mul(attractorStrengthUniform).mul(falloff))

      // Original chaotic soft spherical center colliders
      const toCenter = pos.sub(colliderPosUniform).toVar()
      const centerDist = toCenter.length()
      const penetration = colliderRadiusUniform.sub(centerDist)
      If(penetration.greaterThan(0), () => {
        const pushDir = toCenter.normalize()
        const pushForce = pushDir.mul(penetration.mul(colliderStrengthUniform))
        force.addAssign(pushForce)
      })

      const toCenter2 = pos.sub(collider2PosUniform).toVar()
      const centerDist2 = toCenter2.length()
      const penetration2 = collider2RadiusUniform.sub(centerDist2)
      If(penetration2.greaterThan(0), () => {
        const pushDir2 = toCenter2.normalize()
        const pushForce2 = pushDir2.mul(penetration2.mul(collider2StrengthUniform))
        force.addAssign(pushForce2)
      })

      extForce.assign(force)
    })().compute(SPHERE_COUNT)

    // ─── Instance Culling ─────────────────────────────────────────────────────
    const boundingSphere = new THREE.Sphere(
      new THREE.Vector3(0, BOUNDS.y / 2, 0),
      Math.hypot(BOUNDS.x, BOUNDS.y, BOUNDS.z) / 2,
    )

    function getIndexedGeometry(geo: THREE.BufferGeometry) {
      if (!geo.index) {
        const count = geo.attributes.position.count
        const indices = new Uint32Array(count)
        for (let i = 0; i < count; i++) indices[i] = i
        geo.setIndex(new THREE.BufferAttribute(indices, 1))
      }
      return geo
    }

    const sphereGeoNear = getIndexedGeometry(new THREE.IcosahedronGeometry(SPHERE_RADIUS, geometryDetail))
    const sphereGeoMid = getIndexedGeometry(new THREE.IcosahedronGeometry(SPHERE_RADIUS, Math.max(0, geometryDetail - 1)))
    const sphereGeoFar = getIndexedGeometry(new THREE.IcosahedronGeometry(SPHERE_RADIUS, Math.max(0, geometryDetail - 2)))
    sphereGeoNear.boundingSphere = boundingSphere.clone()
    sphereGeoMid.boundingSphere = boundingSphere.clone()
    sphereGeoFar.boundingSphere = boundingSphere.clone()

    const nearIndirect = new THREE.IndirectStorageBufferAttribute(new Uint32Array([sphereGeoNear.index.count, 0, 0, 0, 0]), 5)
    const midIndirect = new THREE.IndirectStorageBufferAttribute(new Uint32Array([sphereGeoMid.index.count, 0, 0, 0, 0]), 5)
    const farIndirect = new THREE.IndirectStorageBufferAttribute(new Uint32Array([sphereGeoFar.index.count, 0, 0, 0, 0]), 5)

    const nearOutIdSSBO = new THREE.StorageBufferAttribute(new Uint32Array(SPHERE_COUNT), 1)
    const midOutIdSSBO = new THREE.StorageBufferAttribute(new Uint32Array(SPHERE_COUNT), 1)
    const farOutIdSSBO = new THREE.StorageBufferAttribute(new Uint32Array(SPHERE_COUNT), 1)
    const nearOutIdNode = storage(nearOutIdSSBO, 'uint', SPHERE_COUNT)
    const midOutIdNode = storage(midOutIdSSBO, 'uint', SPHERE_COUNT)
    const farOutIdNode = storage(farOutIdSSBO, 'uint', SPHERE_COUNT)

    const DrawIndexed = struct({
      indexCount: 'uint',
      instanceCount: { type: 'uint', atomic: true },
      firstIndex: 'uint',
      baseVertex: 'uint',
      firstInstance: 'uint',
    }, 'DrawIndexed')
    const nearDrawStruct = storage(nearIndirect, DrawIndexed)
    const midDrawStruct = storage(midIndirect, DrawIndexed)
    const farDrawStruct = storage(farIndirect, DrawIndexed)

    const cullCamView = uniform(new THREE.Matrix4())
    const cullCamProj = uniform(new THREE.Matrix4())
    const cullCamPos = uniform(new THREE.Vector3())
    const cullingEnabledUniform = uniform(1)
    const lodDistanceNearUniform = uniform(params.lodDistanceNear)
    const lodDistanceFarUniform = uniform(params.lodDistanceFar)

    const clearCulling = Fn(() => {
      atomicStore(nearDrawStruct.get('instanceCount'), uint(0))
      atomicStore(midDrawStruct.get('instanceCount'), uint(0))
      atomicStore(farDrawStruct.get('instanceCount'), uint(0))
    })().compute(1)

    const cullAndPack = Fn(() => {
      const idx = instanceIndex
      const pos = visualPositions.element(idx)

      const clipPos = cullCamProj.mul(cullCamView).mul(vec4(pos, 1)).toVar()
      const w = clipPos.w

      const rx = float(SPHERE_RADIUS).mul(cullCamProj.element(0).x)
      const ry = float(SPHERE_RADIUS).mul(cullCamProj.element(1).y)
      const rz = float(SPHERE_RADIUS).mul(cullCamProj.element(2).z.abs())

      const inFrustum = w
        .greaterThan(0)
        .and(clipPos.x.abs().lessThanEqual(w.add(rx)))
        .and(clipPos.y.abs().lessThanEqual(w.add(ry)))
        .and(clipPos.z.greaterThanEqual(rz.negate()))
        .and(clipPos.z.lessThanEqual(w.add(rz)))

      const visible = inFrustum.or(cullingEnabledUniform.equal(0))

      If(visible, () => {
        const dist = cullCamPos.sub(pos).length()
        If(dist.lessThan(lodDistanceNearUniform), () => {
          const slot = atomicAdd(nearDrawStruct.get('instanceCount'), uint(1))
          If(slot.lessThan(uint(SPHERE_COUNT)), () => {
            nearOutIdNode.element(slot).assign(uint(idx))
          })
        }).Else(() => {
          If(dist.lessThan(lodDistanceFarUniform), () => {
            const slot = atomicAdd(midDrawStruct.get('instanceCount'), uint(1))
            If(slot.lessThan(uint(SPHERE_COUNT)), () => {
              midOutIdNode.element(slot).assign(uint(idx))
            })
          }).Else(() => {
            const slot = atomicAdd(farDrawStruct.get('instanceCount'), uint(1))
            If(slot.lessThan(uint(SPHERE_COUNT)), () => {
              farOutIdNode.element(slot).assign(uint(idx))
            })
          })
        })
      })
    })().compute(SPHERE_COUNT)

    sphereGeoNear.setIndirect(nearIndirect)
    sphereGeoMid.setIndirect(midIndirect)
    sphereGeoFar.setIndirect(farIndirect)

    const nearRemappedIndex = nearOutIdNode.element(instanceIndex)
    const midRemappedIndex = midOutIdNode.element(instanceIndex)
    const farRemappedIndex = farOutIdNode.element(instanceIndex)

    // ─── Visualization ─────────────────────────────────────────────────────────
    const ballColorUniform = uniform(color(params.ballColor))
    const roughnessUniform = uniform(params.roughness)
    const metalnessUniform = uniform(params.metalness)

    const emissiveColor1Uniform = uniform(color(params.emissiveColor1))
    const emissiveColor2Uniform = uniform(color(params.emissiveColor2))
    const emissiveIntensityUniform = uniform(params.emissiveIntensity)
    const emissiveScaleUniform = uniform(params.emissiveScale)
    const emissivePowerUniform = uniform(params.emissivePower)
    const emissiveAoIntensityUniform = uniform(params.emissiveAoIntensity)
    const emissiveAoRadiusUniform = uniform(params.emissiveAoRadius)
    const emissiveMinUniform = uniform(params.emissiveMin)
    const emissiveMaxUniform = uniform(params.emissiveMax)
    const emissiveTimeUniform = uniform(0)

    const bounceIntensityUniform = uniform(params.bounceIntensity)
    const bounceScaleUniform = uniform(params.bounceScale)
    const bouncePowerUniform = uniform(params.bouncePower)
    
    const noiseDataBuffer = instancedArray(SPHERE_COUNT, 'vec4')

    const computeNoiseData = Fn(() => {
      const pos = visualPositions.element(instanceIndex)
      const eps = float(0.1)

      const nVal = sampleNoise(pos.x, pos.z)
      const gradX = sampleNoise(pos.x.add(eps), pos.z).sub(sampleNoise(pos.x.sub(eps), pos.z))
      const gradZ = sampleNoise(pos.x, pos.z.add(eps)).sub(sampleNoise(pos.x, pos.z.sub(eps)))

      noiseDataBuffer.element(instanceIndex).assign(vec4(nVal, gradX, 0, gradZ))
    })().compute(SPHERE_COUNT)

    const bandCenter = cos(emissiveTimeUniform)
      .mul(0.5)
      .add(0.5)
      .mul(emissiveMaxUniform.sub(emissiveMinUniform))
      .add(emissiveMinUniform)

    function createMaterialNodes(remappedIndex: any) {
      const instancePos = positionLocal.add(visualPositions.element(remappedIndex))

      const noiseData = noiseDataBuffer.element(remappedIndex)
      const noiseVal = noiseData.x
      const aoData = ao.aoBuffer.element(remappedIndex)
      const occWeight = aoData.w
      const occDir = vec3(aoData.x, aoData.y, aoData.z)
      const aoFacing = normalWorld.dot(occDir.normalize()).mul(0.5).add(0.5).mul(occWeight).mul(ao.intensityUniform)
      const aoNode = float(1.0).sub(aoFacing).clamp(0.0, 1.0)

      const emissiveAoDarkening = float(1.0).sub(aoNode)
      const emissiveAo = float(1.0)
        .sub(smoothstep(0, emissiveAoRadiusUniform, emissiveAoDarkening).mul(emissiveAoIntensityUniform))
        .clamp(0, 1)
      
      // Removed the ambient `bandCenter` nerve sweep! 
      // The background sea is now completely pitch black as requested, so the ONLY thing glowing are the spotlight dancers!
      const normalEmissive = vec3(0.0)
      const normalColor = ballColorUniform.mul(aoNode)

      // --- Overhead Spotlight Projectors ---
      const activeFormationGlow = Fn(() => {
        const localPos = visualPositions.element(remappedIndex)
        const nVal = noiseData.x
        
        const isPassed = nVal.greaterThan(wipeThresholdUniform)
        
        const currentMask = getMask(localPos.x, localPos.z, currentFormationUniform)
        const nextMask = getMask(localPos.x, localPos.z, nextFormationUniform)
        
        const activeMask = isPassed.select(nextMask, currentMask)
        
        const distToWipe = abs(nVal.sub(wipeThresholdUniform))
        const wipeGlow = smoothstep(0.12, 0.0, distToWipe).mul(3.5)
        
        const spotlightColor = emissiveColor1Uniform.mul(activeMask).mul(4.0)
        const transitionColor = emissiveColor2Uniform.mul(wipeGlow)
        
        return spotlightColor.add(transitionColor).mul(emissiveIntensityUniform)
      })()

      // The original organic blobs and bounce nodes are intact. The spotlights strictly burst over them!
      const finalEmissive = normalEmissive.add(activeFormationGlow)

      return { positionNode: instancePos, colorNode: normalColor, emissiveNode: finalEmissive }
    }

    function createMaterials(remappedIndex: any) {
      const nodes = createMaterialNodes(remappedIndex)
      const standard = new THREE.MeshStandardNodeMaterial()
      standard.positionNode = nodes.positionNode
      standard.colorNode = nodes.colorNode
      standard.emissiveNode = nodes.emissiveNode
      standard.roughnessNode = roughnessUniform
      return { standard }
    }

    const nearMaterials = createMaterials(nearRemappedIndex)
    const midMaterials = createMaterials(midRemappedIndex)
    const farMaterials = createMaterials(farRemappedIndex)

    nearMaterials.standard.roughnessNode = roughnessUniform
    nearMaterials.standard.metalnessNode = metalnessUniform
    midMaterials.standard.roughnessNode = roughnessUniform
    midMaterials.standard.metalnessNode = metalnessUniform
    farMaterials.standard.roughnessNode = roughnessUniform
    farMaterials.standard.metalnessNode = metalnessUniform

    const sphereMeshNear = new THREE.InstancedMesh(sphereGeoNear, nearMaterials.standard, SPHERE_COUNT)
    sphereMeshNear.castShadow = true
    sphereMeshNear.receiveShadow = true
    scene.add(sphereMeshNear)

    const sphereMeshMid = new THREE.InstancedMesh(sphereGeoMid, midMaterials.standard, SPHERE_COUNT)
    sphereMeshMid.castShadow = true
    sphereMeshMid.receiveShadow = true
    scene.add(sphereMeshMid)

    const sphereMeshFar = new THREE.InstancedMesh(sphereGeoFar, farMaterials.standard, SPHERE_COUNT)
    sphereMeshFar.castShadow = true
    sphereMeshFar.receiveShadow = true
    scene.add(sphereMeshFar)

    // Center colliders & lights
    const cameraDir = new THREE.Vector3().subVectors(cameraTarget, camera.position).normalize()
    const colliderPos = new THREE.Vector3().copy(cameraTarget).addScaledVector(cameraDir, params.colliderDistance)
    colliderPos.y += params.colliderY

    const pointLight = new THREE.PointLight(params.pointLightColor, 0, params.pointLightRange, params.pointLightDecay)
    const pointLightPos = new THREE.Vector3().copy(cameraTarget).addScaledVector(cameraDir, params.pointLightDistance)
    pointLightPos.y += params.pointLightY
    pointLight.position.copy(pointLightPos)
    scene.add(pointLight)

    colliderPosUniform.value.copy(colliderPos)
    collider2PosUniform.value.copy(pointLight.position)

    // ─── Setup and animation ──────────────────────────────────────────────────
    const fixedTimestep = new FixedTimestep({ dt: 1 / 120, maxSteps: 6 })

    let emissiveTime = 0
    let prevTimestamp = 0
    let elapsedTime = 0

    // Async initializer
    async function initPhysics() {
      try {
        await renderer.init()
        
        await physics.init()
        await snapVisualPositions()
        await renderer.compileAsync(scene, camera)
      } catch (err) {
        console.warn('WebGPU not supported or context lost.', err)
        return
      }

      const raycaster = new THREE.Raycaster()
      const mouse = new THREE.Vector2()
      const attractorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -params.attractorHeight)
      const attractorPos = new THREE.Vector3(0, 0, 0)
      let mouseOnGround = false

      const onPointerMove = (e: PointerEvent) => {
        mouse.x = (e.clientX / window.innerWidth) * 2 - 1
        mouse.y = -(e.clientY / window.innerHeight) * 2 + 1
        raycaster.setFromCamera(mouse, camera)
        attractorPlane.constant = -params.attractorHeight
        const hit = raycaster.ray.intersectPlane(attractorPlane, attractorPos)
        mouseOnGround = hit !== null
      }
      window.addEventListener('pointermove', onPointerMove)

      renderer.setAnimationLoop(async (timestamp) => {
        if (!mounted) return
        const dt = (timestamp - prevTimestamp) / 1000
        prevTimestamp = timestamp
        elapsedTime += dt
        
        attractorActiveUniform.value = mouseOnGround ? 1 : 0
        attractorPosUniform.value.copy(attractorPos)

        const fadeElapsed = elapsedTime - params.lightFadeDelay
        if (fadeElapsed < 0) {
          rimLight.intensity = 0
          pointLight.intensity = 0
          ambientLight.intensity = 0
        } else if (fadeElapsed < params.lightFadeIn) {
          const t = fadeElapsed / params.lightFadeIn
          rimLight.intensity = params.lightIntensity * t
          pointLight.intensity = params.pointLightIntensity * t
          ambientLight.intensity = 0.4 * t
        } else {
          rimLight.intensity = params.lightIntensity
          pointLight.intensity = params.pointLightIntensity
          ambientLight.intensity = 0.4
        }
        
        // --- Spotlight Formations Timer ---
        // Increased times for a slower, more deliberate Nerve Sweep
        const FORMATION_DURATION = 8.0 
        const WIPE_DURATION = 3.5
        
        const cycleTime = elapsedTime % (FORMATION_DURATION * 4)
        const formationStage = Math.floor(cycleTime / FORMATION_DURATION)
        const localTime = cycleTime % FORMATION_DURATION
        
        currentFormationUniform.value = formationStage
        nextFormationUniform.value = (formationStage + 1) % 4
        
        // Organic sweep transition across the noise pattern
        if (localTime < WIPE_DURATION) {
           const t = localTime / WIPE_DURATION
           // The geometric noise values strictly exist between ~[-0.2, 0.8]. 
           // Sweeping from 0.9 down to -0.3 ensures the wave's visible speed actually matches the full duration!
           wipeThresholdUniform.value = 0.9 - (t * 1.2) 
        } else {
           // Park safely WAY out of noise range so the neon cyan transition edge never randomly triggers!
           wipeThresholdUniform.value = -10.0
        }

        const steps = params.paused ? 0 : fixedTimestep.update(timestamp)

        if (steps > 0) {
          timeUniform.value = timestamp / 1000

          renderer.compute(computeExternalForces)
          for (let i = 0; i < steps; i++) {
            physics.compute(fixedTimestep.dt, { correctionPass: params.correctionPass })
          }
        }

        ao.update(renderer)

        visualLerpUniform.value = params.visualLerp
        renderer.compute(lerpVisualPositions)
        renderer.compute(computeNoiseData)

        cullCamView.value.copy(camera.matrixWorldInverse)
        cullCamProj.value.copy(camera.projectionMatrix)
        cullCamPos.value.copy(camera.position)
        renderer.compute(clearCulling)
        renderer.compute(cullAndPack)

        if (!params.emissivePaused && fadeElapsed >= 0) {
          emissiveTime += dt * params.emissiveSpeed
        }
        emissiveTimeUniform.value = emissiveTime

        renderer.render(scene, camera)

        await renderer.resolveTimestampsAsync('render')
        await renderer.resolveTimestampsAsync('compute')
      })
    }
    
    initPhysics()

    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
    }
    window.addEventListener('resize', onResize)

    return () => {
      mounted = false
      window.removeEventListener('resize', onResize)
      renderer.setAnimationLoop(null)
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  return <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }} />
}
