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
} from 'three/tsl'
import { FixedTimestep } from './FixedTimestep'
import { BallPhysicsGPU } from './BallPhysicsGPU'
import { SphereGridAO } from './SphereGridAO'

interface SwayPhysicsCanvasProps {
  sphereCount?: number
  glowPattern?: number
  color1?: string
  color2?: string
  noiseAmplitude?: number
  attractorStrength?: number
  cohesion?: number
}

export function SwayPhysicsCanvas({
  sphereCount = 4096 * 2 * 2 * 2 * 1.5,
  glowPattern = 0,
  color1 = '#1144ff',
  color2 = '#ffffff',
  noiseAmplitude = 25,
  attractorStrength = 40,
  cohesion = 30,
}: SwayPhysicsCanvasProps) {
  void glowPattern
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current
    let mounted = true

    const SPHERE_COUNT = Math.floor(sphereCount)
    const SPHERE_RADIUS = 0.07
    const BOUNDS = { x: 12, z: 12, y: 12 }

    const params = {
      gravity: -10,
      stiffness: 4000,
      damping: 10,
      friction: 0.4,
      restitution: 0.3,
      cohesion,
      correctionStrength: 0.5,
      correctionDamping: 0.3,
      correctionPass: true,
      paused: false,
      visualLerp: 0.5,
      noiseAmplitude,
      noiseFrequency: 0.4,
      noiseSpeed: 0.15,
      attractorHeight: 0.6,
      attractorRadius: 1.5,
      attractorStrength,
      fov: 30,
      ballColor: '#000000',
      roughness: 0.8,
      metalness: 0,
      lightIntensity: 0.7,
      shadowMapSize: 1024,
      shadowBias: -0.001,
      shadowNormalBias: 0.02,
      lightX: 5,
      lightY: 3,
      lightZ: -3,
      emissivePower: 4,
      emissiveIntensity: 2,
      emissiveAoIntensity: 0.4,
      emissiveAoRadius: 1.5,
      emissiveSpeed: 0.3,
      emissiveScale: 0.15,
      emissiveMin: -0.3,
      emissiveMax: 0.9,
      bounceIntensity: 0.2,
      bounceScale: 0.26,
      bouncePower: 4,
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
      pointLightColor: '#ffffff',
      pointLightRange: 0,
      pointLightDecay: 3.6,
      pointLightDistance: -3.28,
      pointLightY: 0.1,
      collider2Radius: 0.7,
      collider2Strength: 200,
      lightFadeIn: 0.3,
      lightFadeDelay: 1.2,
    }

    const renderer = new THREE.WebGPURenderer({ antialias: true, alpha: true })
    renderer.setClearColor(0x000000, 0)
    renderer.toneMapping = THREE.AgXToneMapping
    renderer.toneMappingExposure = 1.0
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.VSMShadowMap
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.domElement.style.cssText =
      'position:absolute;top:0;left:0;width:100%;height:100%;z-index:2;pointer-events:none;'
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(params.fov, window.innerWidth / window.innerHeight, 0.01, 100)
    camera.position.set(-5.5, 2.2, 6.7)
    const cameraTarget = new THREE.Vector3(-0.92, 0.41, -0.09)
    camera.lookAt(cameraTarget)

    const ambientLight = new THREE.AmbientLight('#ffffff', 0)
    scene.add(ambientLight)

    const rimLight = new THREE.DirectionalLight('#ffffff', 0)
    rimLight.position.set(params.lightX, params.lightY, params.lightZ)
    rimLight.castShadow = true
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

    const physics = new BallPhysicsGPU({
      renderer,
      count: SPHERE_COUNT,
      radius: SPHERE_RADIUS,
      bounds: BOUNDS,
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

    const noiseAmplitudeUniform = uniform(params.noiseAmplitude)
    const noiseFrequencyUniform = uniform(params.noiseFrequency)
    const noiseSpeedUniform = uniform(params.noiseSpeed)
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
      return mx_noise_float(noiseSamplePos(px, pz)).mul(NOISE_SCALE).add(NOISE_BIAS)
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

    const computeExternalForces = Fn(() => {
      const pos = physics.positions.element(instanceIndex)
      const extForce = physics.externalForces.element(instanceIndex)

      const noiseVal = sampleNoise(pos.x, pos.z)
      const force = vec3(0, noiseVal.mul(noiseAmplitudeUniform), 0).toVar()

      const toAttractor = attractorPosUniform.sub(pos).toVar()
      const dist = toAttractor.length()
      const falloff = dist.div(attractorRadiusUniform).clamp(0, 1).oneMinus().mul(attractorActiveUniform)
      force.addAssign(toAttractor.normalize().mul(attractorStrengthUniform).mul(falloff))

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

    const boundingSphere = new THREE.Sphere(
      new THREE.Vector3(0, BOUNDS.y / 2, 0),
      Math.hypot(BOUNDS.x, BOUNDS.y, BOUNDS.z) / 2,
    )

    const sphereGeoNear = new THREE.SphereGeometry(SPHERE_RADIUS, 8, 8)
    const sphereGeoMid = new THREE.SphereGeometry(SPHERE_RADIUS, 6, 6)
    const sphereGeoFar = new THREE.SphereGeometry(SPHERE_RADIUS, 4, 4)
    sphereGeoNear.boundingSphere = boundingSphere.clone()
    sphereGeoMid.boundingSphere = boundingSphere.clone()
    sphereGeoFar.boundingSphere = boundingSphere.clone()

    const nearIndirect = new THREE.IndirectStorageBufferAttribute(
      new Uint32Array([sphereGeoNear.index.count, 0, 0, 0, 0]),
      5,
    )
    const midIndirect = new THREE.IndirectStorageBufferAttribute(new Uint32Array([sphereGeoMid.index.count, 0, 0, 0, 0]), 5)
    const farIndirect = new THREE.IndirectStorageBufferAttribute(new Uint32Array([sphereGeoFar.index.count, 0, 0, 0, 0]), 5)

    const nearOutIdSSBO = new THREE.StorageBufferAttribute(new Uint32Array(SPHERE_COUNT), 1)
    const midOutIdSSBO = new THREE.StorageBufferAttribute(new Uint32Array(SPHERE_COUNT), 1)
    const farOutIdSSBO = new THREE.StorageBufferAttribute(new Uint32Array(SPHERE_COUNT), 1)
    const nearOutIdNode = storage(nearOutIdSSBO, 'uint', SPHERE_COUNT)
    const midOutIdNode = storage(midOutIdSSBO, 'uint', SPHERE_COUNT)
    const farOutIdNode = storage(farOutIdSSBO, 'uint', SPHERE_COUNT)

    const DrawIndexed = struct(
      {
        indexCount: 'uint',
        instanceCount: { type: 'uint', atomic: true },
        firstIndex: 'uint',
        baseVertex: 'uint',
        firstInstance: 'uint',
      },
      'DrawIndexed',
    )
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

    const ballColorUniform = uniform(color(params.ballColor))
    const roughnessUniform = uniform(params.roughness)
    const metalnessUniform = uniform(params.metalness)
    const emissiveColor1Uniform = uniform(color(color1))
    const emissiveColor2Uniform = uniform(color(color2))
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
      const gradDir = vec3(noiseData.y, 0, noiseData.w).normalize().mul(sign(bandCenter.sub(noiseVal)))

      const bDistance = abs(noiseVal.sub(bandCenter)).div(emissiveScaleUniform).clamp(0, 1)
      const gColor = mix(emissiveColor2Uniform, emissiveColor1Uniform, smoothstep(0, 0.5, bDistance))
      const gAlpha = pow(smoothstep(1.0, 0.7, bDistance), emissivePowerUniform)

      const aoData = ao.aoBuffer.element(remappedIndex)
      const occDir = vec3(aoData.x, aoData.y, aoData.z)
      const occWeight = aoData.w
      const aoFacing = normalWorld.dot(occDir.normalize()).mul(0.5).add(0.5).mul(occWeight).mul(ao.intensityUniform)
      const aoNode = float(1.0).sub(aoFacing).clamp(0.0, 1.0)

      const hFactor = pow(normalWorld.dot(gradDir).mul(0.5).add(0.5), bouncePowerUniform)
      const bDist = abs(noiseVal.sub(bandCenter)).div(bounceScaleUniform).clamp(0, 1)
      const bProximity = smoothstep(1.0, 0.3, bDist)
      const bMask = gAlpha.oneMinus()
      const bounceNode = gColor.mul(bounceIntensityUniform).mul(bProximity).mul(hFactor).mul(bMask).mul(aoNode)

      const emissiveAoDarkening = float(1.0).sub(aoNode)
      const emissiveAo = float(1.0)
        .sub(smoothstep(0, emissiveAoRadiusUniform, emissiveAoDarkening).mul(emissiveAoIntensityUniform))
        .clamp(0, 1)
      const emissiveNode = gColor.mul(emissiveIntensityUniform).mul(emissiveAo).mul(gAlpha).add(bounceNode)
      const aoColorNode = ballColorUniform.mul(aoNode)
      return { positionNode: instancePos, colorNode: aoColorNode, emissiveNode }
    }

    function createMaterials(remappedIndex: any) {
      const nodes = createMaterialNodes(remappedIndex)
      const standard = new THREE.MeshStandardNodeMaterial()
      standard.positionNode = nodes.positionNode
      standard.colorNode = nodes.colorNode
      standard.emissiveNode = nodes.emissiveNode
      standard.roughnessNode = roughnessUniform
      standard.metalnessNode = metalnessUniform
      return { standard }
    }

    const nearMaterials = createMaterials(nearRemappedIndex)
    const midMaterials = createMaterials(midRemappedIndex)
    const farMaterials = createMaterials(farRemappedIndex)

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

    const cameraDir = new THREE.Vector3().subVectors(cameraTarget, camera.position).normalize()
    const colliderPos = new THREE.Vector3().copy(cameraTarget).addScaledVector(cameraDir, params.colliderDistance)
    colliderPos.y += params.colliderY

    const pointLight = new THREE.PointLight(
      params.pointLightColor,
      0,
      params.pointLightRange,
      params.pointLightDecay,
    )
    pointLight.position.copy(cameraTarget).addScaledVector(cameraDir, params.pointLightDistance)
    pointLight.position.y += params.pointLightY
    scene.add(pointLight)

    colliderPosUniform.value.copy(colliderPos)
    collider2PosUniform.value.copy(pointLight.position)

    const fixedTimestep = new FixedTimestep({ dt: 1 / 120, maxSteps: 6 })
    let emissiveTime = 0
    let prevTimestamp = 0
    let elapsedTime = 0

    async function initPhysics() {
      try {
        await renderer.init()
        await physics.init()
        await snapVisualPositions()
        await renderer.compileAsync(scene, camera)
      } catch {
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
      const onPointerLeave = () => {
        mouseOnGround = false
      }
      window.addEventListener('pointermove', onPointerMove)
      window.addEventListener('mouseleave', onPointerLeave)

      renderer.setAnimationLoop((timestamp) => {
        if (!mounted) return
        const dt = (timestamp - prevTimestamp) / 1000
        prevTimestamp = timestamp
        elapsedTime += dt

        const fadeElapsed = elapsedTime - params.lightFadeDelay
        if (fadeElapsed < 0) {
          rimLight.intensity = 0
          pointLight.intensity = 0
        } else if (fadeElapsed < params.lightFadeIn) {
          const t = fadeElapsed / params.lightFadeIn
          rimLight.intensity = params.lightIntensity * t
          pointLight.intensity = params.pointLightIntensity * t
        } else {
          rimLight.intensity = params.lightIntensity
          pointLight.intensity = params.pointLightIntensity
        }

        const steps = params.paused ? 0 : fixedTimestep.update(timestamp)
        if (steps > 0) {
          physics.gravityUniform.value = params.gravity
          physics.stiffnessUniform.value = params.stiffness
          physics.dampingUniform.value = params.damping
          physics.frictionUniform.value = params.friction
          physics.restitutionUniform.value = params.restitution
          physics.cohesionUniform.value = params.cohesion
          physics.correctionStrengthUniform.value = params.correctionStrength
          physics.correctionDampingUniform.value = params.correctionDamping
          noiseAmplitudeUniform.value = params.noiseAmplitude
          noiseFrequencyUniform.value = params.noiseFrequency
          noiseSpeedUniform.value = params.noiseSpeed
          timeUniform.value = timestamp / 1000
          attractorRadiusUniform.value = params.attractorRadius
          attractorStrengthUniform.value = params.attractorStrength
          attractorActiveUniform.value = mouseOnGround ? 1 : 0
          attractorPosUniform.value.copy(attractorPos)
          renderer.compute(computeExternalForces)

          for (let i = 0; i < steps; i++) {
            physics.compute(fixedTimestep.dt, { correctionPass: params.correctionPass })
          }
        }

        ao.update(renderer)
        visualLerpUniform.value = params.visualLerp
        renderer.compute(lerpVisualPositions)
        renderer.compute(computeNoiseData)

        cullingEnabledUniform.value = params.culling ? 1 : 0
        lodDistanceNearUniform.value = params.lodDistanceNear
        lodDistanceFarUniform.value = params.lodDistanceFar
        cullCamView.value.copy(camera.matrixWorldInverse)
        cullCamProj.value.copy(camera.projectionMatrix)
        cullCamPos.value.copy(camera.position)
        renderer.compute(clearCulling)
        renderer.compute(cullAndPack)

        emissiveTime += dt * params.emissiveSpeed
        emissiveTimeUniform.value = emissiveTime
        renderer.render(scene, camera)
      })

      return () => {
        window.removeEventListener('pointermove', onPointerMove)
        window.removeEventListener('mouseleave', onPointerLeave)
      }
    }

    let physicsTeardown: (() => void) | undefined
    initPhysics().then((cleanup) => {
      physicsTeardown = cleanup
    })

    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
    }
    window.addEventListener('resize', onResize)

    return () => {
      mounted = false
      physicsTeardown?.()
      window.removeEventListener('resize', onResize)
      renderer.setAnimationLoop(null)
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [sphereCount, color1, color2, noiseAmplitude, attractorStrength, cohesion])

  return <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }} />
}
