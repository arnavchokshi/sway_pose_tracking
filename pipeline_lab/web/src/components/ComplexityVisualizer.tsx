import { useEffect, useRef, useMemo } from 'react'

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

  // detection resolution
  const detSize = Number(fields.sway_detect_size) || 1280
  score += detSize > 1280 ? 4 : (detSize < 1280 ? 1 : 2)

  // tracker 
  const track = String(fields.tracker_technology || '')
  if (track === 'BoT-SORT') score += 3
  else if (track === 'BoxMOT') score += 1

  // hybrid sam
  if (fields.sway_hybrid_sam_overlap) score += 3
  
  // temporal refine
  if (fields.temporal_pose_refine) score += 2

  // Score conceptually ranges from ~5 to ~25
  const normalized = Math.max(0, Math.min(1, (score - 5) / 20)) // 0.0 to 1.0
  
  return {
    numNodes: Math.floor(20 + normalized * 70), // 20 to 90 nodes
    nodeSpeed: 0.2 + normalized * 0.8, // 0.2 to 1.0 speed
    glowAlpha: 0.1 + normalized * 0.4, // brightens
    linkDist: 100 + normalized * 60 // 100 to 160 link distance
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
  
  constructor(w: number, h: number, speed: number) {
    this.x = Math.random() * w
    this.y = Math.random() * h
    const angle = Math.random() * Math.PI * 2
    const s = (Math.random() * 0.5 + 0.5) * speed
    this.vx = Math.cos(angle) * s
    this.vy = Math.sin(angle) * s
    this.radius = Math.random() * 2 + 1
    this.pulseSpeed = Math.random() * 0.05 + 0.01
    this.pulseOffset = Math.random() * Math.PI * 2
  }

  update(w: number, h: number, speedMult: number) {
    this.x += this.vx * speedMult
    this.y += this.vy * speedMult
    
    if (this.x < 0) { this.x = 0; this.vx *= -1 }
    if (this.x > w) { this.x = w; this.vx *= -1 }
    if (this.y < 0) { this.y = 0; this.vy *= -1 }
    if (this.y > h) { this.y = h; this.vy *= -1 }
  }
}

export function ComplexityVisualizer({ fieldsState, className = '' }: { fieldsState: Record<string, unknown>, className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const metrics = useMemo(() => calculateComplexityMetrics(fieldsState), [fieldsState])
  
  // Keep refs for animation loop
  const nodesRef = useRef<Node[]>([])
  const metricsRef = useRef(metrics)
  const timeRef = useRef(0)
  
  useEffect(() => {
    metricsRef.current = metrics
  }, [metrics])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    let w = canvas.width
    let h = canvas.height
    
    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect()
      if (rect) {
        canvas.width = rect.width
        canvas.height = rect.height
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
      
      // Draw a highly transparent background that glows
      const cx = w * 0.8
      const cy = h * 0.2
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h))
      grad.addColorStop(0, `rgba(6, 182, 212, ${m.glowAlpha * 0.2})`)
      grad.addColorStop(1, 'rgba(0,0,0,0)')
      
      ctx.fillStyle = grad
      ctx.fillRect(0, 0, w, h)
      
      // We keep a pool of max nodes, but only render up to currentNodes
      const pool = nodesRef.current
      
      for (let i = 0; i < currentNodes; i++) {
        const node = pool[i]
        node.update(w, h, m.nodeSpeed)
        
        // pulsing radius
        const r = node.radius + Math.sin(timeRef.current * node.pulseSpeed + node.pulseOffset) * 1.5
        
        ctx.beginPath()
        ctx.arc(node.x, node.y, Math.max(0.5, r), 0, Math.PI * 2)
        ctx.fillStyle = `rgba(14, 165, 233, ${0.4 + m.glowAlpha})`
        ctx.fill()
        
        // draw links
        for (let j = i + 1; j < currentNodes; j++) {
          const other = pool[j]
          const dx = node.x - other.x
          const dy = node.y - other.y
          const distSq = dx*dx + dy*dy
          const linkSq = m.linkDist * m.linkDist
          
          if (distSq < linkSq) {
            const alpha = 1 - Math.sqrt(distSq) / m.linkDist
            ctx.beginPath()
            ctx.moveTo(node.x, node.y)
            ctx.lineTo(other.x, other.y)
            ctx.strokeStyle = `rgba(6, 182, 212, ${alpha * (0.3 + m.glowAlpha * 0.5)})`
            ctx.lineWidth = 1 + alpha
            ctx.stroke()
          }
        }
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
        top: 0,
        left: 0,
        pointerEvents: 'none',
        opacity: 0.45,
        mixBlendMode: 'screen',
        transition: 'opacity 0.5s ease',
        zIndex: 0
      }} 
    />
  )
}
