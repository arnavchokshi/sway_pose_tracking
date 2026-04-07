import { useEffect, useRef, useMemo } from 'react'

/** Muted slate / cool-gray only — keeps the viz readable, not rainbow */
const PALETTE = ['#64748b', '#6b7280', '#788396', '#5c6b7a', '#7d8694']

type Metrics = {
  numNodes: number
  nodeSpeed: number
  glowAlpha: number
  linkDist: number
}

function calculateComplexityMetrics(fields: Record<string, unknown>): Metrics {
  let score = 0
  
  // yolo size
  const yolo = String(fields.sway_yolo_weights || '')
  if (yolo === 'yolo26x') score += 5
  else if (yolo === 'yolo26l' || yolo === 'yolo26l_dancetrack') score += 3
  else score += 1

  // pose mode
  const pose = String(fields.pose_model || '')
  if (pose === 'ViTPose-Huge') score += 5
  else if (pose === 'ViTPose-Large') score += 3
  else score += 1

  // Master stack: SWAY_DETECT_SIZE=640 + SWAY_GROUP_VIDEO → effective letterbox ≥960 in tracker.py
  const detLetterbox = 960
  score += detLetterbox > 1280 ? 4 : detLetterbox > 960 ? 3 : detLetterbox > 640 ? 2 : 1

  // tracker
  const track = String(fields.tracker_technology || '')
  if (track === 'deep_ocsort_osnet' || track === 'StrongSORT') score += 2
  else if (track === 'bytetrack') score += 0
  else if (track === 'BoxMOT' || track === 'deep_ocsort') score += 1

  // Hybrid SAM overlap refiner (master stack; off when ByteTrack fast path disables it)
  if (track !== 'bytetrack') {
    score += 3
  }

  const yoloStride = Number(fields.sway_yolo_detection_stride) || 1
  if (yoloStride > 1) score += 1

  if (Number(fields.pose_stride) === 2) score += 1

  if (fields.sway_bidirectional_track_pass) score += 4

  if (fields.sway_gnn_track_refine) score += 1
  if (fields.sway_hmr_mesh_sidecar) score += 1

  if (String(fields.pose_model || '').includes('Sapiens')) score += 1

  // temporal refine
  if (fields.temporal_pose_refine) score += 2

  // Score conceptually ranges from ~5 to ~25
  const normalized = Math.max(0, Math.min(1, (score - 5) / 20)) // 0.0 to 1.0
  
  return {
    numNodes: Math.floor(20 + normalized * 80), // 20 to 100 nodes
    nodeSpeed: 0.15 + normalized * 0.85, 
    glowAlpha: 0.15 + normalized * 0.45, 
    linkDist: 110 + normalized * 70
  }
}

class Node {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  pulseSpeed: number
  pulseOffset: number
  colorIndex: number
  
  constructor(w: number, h: number, speed: number) {
    this.x = Math.random() * w
    this.y = Math.random() * h
    const angle = Math.random() * Math.PI * 2
    const s = (Math.random() * 0.5 + 0.5) * speed
    this.vx = Math.cos(angle) * s
    this.vy = Math.sin(angle) * s
    this.radius = Math.random() * 2 + 1.5
    this.pulseSpeed = Math.random() * 0.03 + 0.01
    this.pulseOffset = Math.random() * Math.PI * 2
    this.colorIndex = Math.floor(Math.random() * PALETTE.length)
  }

  update(w: number, h: number, speedMult: number, mouseX: number, mouseY: number) {
    this.x += this.vx * speedMult
    this.y += this.vy * speedMult
    
    // Repel from mouse
    if (mouseX > 0 && mouseY > 0) {
      const dx = this.x - mouseX
      const dy = this.y - mouseY
      const distSq = dx*dx + dy*dy
      if (distSq < 40000) { // 200px radius
        const dist = Math.sqrt(distSq)
        const force = (200 - dist) / 200
        this.x += (dx / dist) * force * 2
        this.y += (dy / dist) * force * 2
      }
    }
    
    // Bounce off walls smoothly (with a margin, so they don't pop instantly at edges)
    const margin = 50
    if (this.x < -margin) { this.x = -margin; this.vx *= -1 }
    if (this.x > w + margin) { this.x = w + margin; this.vx *= -1 }
    if (this.y < -margin) { this.y = -margin; this.vy *= -1 }
    if (this.y > h + margin) { this.y = h + margin; this.vy *= -1 }
  }
}

class DataPacket {
  progress: number
  speed: number
  sourceIdx: number
  targetIdx: number
  
  constructor(sourceIdx: number, targetIdx: number) {
    this.progress = 0
    this.speed = Math.random() * 0.02 + 0.01 // 1% to 3% per frame
    this.sourceIdx = sourceIdx
    this.targetIdx = targetIdx
  }
}

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? [
    parseInt(result[1], 16),
    parseInt(result[2], 16),
    parseInt(result[3], 16)
  ] : [0, 0, 0];
}

export function ComplexityVisualizer({ fieldsState, className = '' }: { fieldsState: Record<string, unknown>, className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const metrics = useMemo(() => calculateComplexityMetrics(fieldsState), [fieldsState])
  
  // Animation state references
  const nodesRef = useRef<Node[]>([])
  const packetsRef = useRef<DataPacket[]>([])
  const metricsRef = useRef(metrics)
  const timeRef = useRef(0)
  const mouseRef = useRef({ x: -1000, y: -1000 })
  
  useEffect(() => {
    metricsRef.current = metrics
  }, [metrics])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // Need bounding rect of canvas to get relative coords
      const canvas = canvasRef.current
      if (canvas) {
        const rect = canvas.getBoundingClientRect()
        mouseRef.current = {
          x: e.clientX - rect.left,
          y: e.clientY - rect.top
        }
      }
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    let w = canvas.width
    let h = canvas.height
    
    const rgbCache = PALETTE.map(hexToRgb)
    
    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect()
      if (rect) {
        // High DPI canvas support
        const dpr = window.devicePixelRatio || 1
        canvas.width = rect.width * dpr
        canvas.height = rect.height * dpr
        ctx.scale(dpr, dpr)
        canvas.style.width = `${rect.width}px`
        canvas.style.height = `${rect.height}px`
        
        w = rect.width
        h = rect.height
        
        // Re-populate nodes if too few
        if (nodesRef.current.length === 0) {
           nodesRef.current = Array.from({ length: 150 }).map(() => new Node(w, h, 1))
        }
      }
    }
    
    window.addEventListener('resize', resize)
    resize()
    
    let reqId: number
    const render = () => {
      reqId = requestAnimationFrame(render)
      timeRef.current += 1
      
      const m = metricsRef.current
      const currentNodes = m.numNodes
      
      ctx.clearRect(0, 0, w, h)
      
      // Subtle neutral vignette (intensity still tracks workload)
      const cx = w * 0.5
      const cy = h * 0.5
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h) * 0.85)
      grad.addColorStop(0, `rgba(100, 116, 139, ${m.glowAlpha * 0.12})`)
      grad.addColorStop(0.55, `rgba(71, 85, 105, ${m.glowAlpha * 0.04})`)
      grad.addColorStop(1, 'rgba(0,0,0,0)')
      
      ctx.fillStyle = grad
      ctx.fillRect(0, 0, w, h)
      
      const pool = nodesRef.current
      const activeNodes = pool.slice(0, currentNodes)
      const mouse = mouseRef.current
      
      // Update node positions
      for (let i = 0; i < currentNodes; i++) {
        activeNodes[i].update(w, h, m.nodeSpeed, mouse.x, mouse.y)
      }
      
      // Build links dynamically based on proximity
      const links: [number, number, number][] = [] // idxA, idxB, distance
      const linkSq = m.linkDist * m.linkDist
      
      for (let i = 0; i < currentNodes; i++) {
        const node = activeNodes[i]
        for (let j = i + 1; j < currentNodes; j++) {
          const other = activeNodes[j]
          const dx = node.x - other.x
          const dy = node.y - other.y
          const distSq = dx*dx + dy*dy
          
          if (distSq < linkSq) {
            links.push([i, j, Math.sqrt(distSq)])
          }
        }
      }
      
      // Spawn data packets along active links randomly
      // Packets spawn more often when network is denser / complex
      if (links.length > 0 && Math.random() < (0.05 + m.glowAlpha * 0.1)) {
        const link = links[Math.floor(Math.random() * links.length)]
        if (Math.random() > 0.5) packetsRef.current.push(new DataPacket(link[0], link[1]))
        else packetsRef.current.push(new DataPacket(link[1], link[0]))
      }
      
      // Clean up packets that reached their destination or whose nodes were removed
      packetsRef.current = packetsRef.current.filter(p => 
        p.progress < 1 && p.sourceIdx < currentNodes && p.targetIdx < currentNodes
      )
      
      // Rendering Layer 1: The connecting links
      ctx.lineCap = 'round'
      for (const [i, j, dist] of links) {
        const node = activeNodes[i]
        const other = activeNodes[j]
        const alpha = Math.pow(1 - dist / m.linkDist, 1.5) // non-linear fade for organic feel
        
        ctx.beginPath()
        ctx.moveTo(node.x, node.y)
        ctx.lineTo(other.x, other.y)
        
        const linkGrad = ctx.createLinearGradient(node.x, node.y, other.x, other.y)
        const [r1, g1, b1] = rgbCache[node.colorIndex]
        const [r2, g2, b2] = rgbCache[other.colorIndex]
        const baseA = alpha * (0.12 + m.glowAlpha * 0.22)
        linkGrad.addColorStop(0, `rgba(${r1}, ${g1}, ${b1}, ${baseA})`)
        linkGrad.addColorStop(1, `rgba(${r2}, ${g2}, ${b2}, ${baseA})`)
        
        ctx.strokeStyle = linkGrad
        ctx.lineWidth = 1 + alpha * 1.5
        ctx.stroke()
      }
      
      // Rendering Layer 2: Moving data packets traversing the links
      for (const packet of packetsRef.current) {
        packet.progress += packet.speed * m.nodeSpeed
        if (packet.progress > 1) continue
        
        const source = activeNodes[packet.sourceIdx]
        const target = activeNodes[packet.targetIdx]
        
        const px = source.x + (target.x - source.x) * packet.progress
        const py = source.y + (target.y - source.y) * packet.progress
        
        const [r, g, b] = rgbCache[target.colorIndex]
        const alpha = Math.sin(packet.progress * Math.PI) // fade in and out at ends
        
        ctx.beginPath()
        ctx.arc(px, py, 2.5, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.9})`
        ctx.fill()
        
        // Packet trails/glow
        ctx.beginPath()
        ctx.arc(px, py, 6, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha * 0.35})`
        ctx.fill()
      }

      // Rendering Layer 3: The nodes themselves
      for (let i = 0; i < currentNodes; i++) {
        const node = activeNodes[i]
        const [r, g, b] = rgbCache[node.colorIndex]
        
        // Outer pulsing ring for aesthetic
        const pulseR = node.radius * 2.5 + Math.sin(timeRef.current * node.pulseSpeed + node.pulseOffset) * 2
        ctx.beginPath()
        ctx.arc(node.x, node.y, Math.max(0.1, pulseR), 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.08 + m.glowAlpha * 0.12})`
        ctx.fill()

        const coreAlpha = 0.35 + m.glowAlpha * 0.25
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(226, 232, 240, ${coreAlpha})`
        ctx.fill()

        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius * 1.5, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${coreAlpha * 0.35})`
        ctx.fill()
      }
    }
    
    reqId = requestAnimationFrame(render)
    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(reqId)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        opacity: 0.85,
        transition: 'opacity 0.8s ease',
        zIndex: 0,
      }}
    />
  )
}
