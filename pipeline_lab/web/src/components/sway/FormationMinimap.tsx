import { useEffect, useRef } from 'react'
import type { ScoringFramePayload } from '../../swayScoring/types'

type Props = {
  frame: ScoringFramePayload | null
  width?: number
  height?: number
}

export function FormationMinimap({ frame, width = 220, height = 220 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')
    if (!ctx) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    c.width = Math.round(width * dpr)
    c.height = Math.round(height * dpr)
    c.style.width = `${width}px`
    c.style.height = `${height}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)

    ctx.fillStyle = 'rgba(8, 10, 20, 0.92)'
    ctx.fillRect(0, 0, width, height)
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)'
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1)

    if (!frame?.formation_targets || !frame.formation_actual) {
      ctx.fillStyle = 'rgba(161, 161, 170, 0.8)'
      ctx.font = '12px Plus Jakarta Sans, system-ui, sans-serif'
      ctx.fillText('No formation data', 14, height / 2)
      return
    }

    const toX = (nx: number) => nx * width
    const toY = (ny: number) => ny * height

    const ids = Object.keys(frame.formation_targets).map(Number)
    for (const id of ids) {
      const tgt = frame.formation_targets[id]
      const act = frame.formation_actual[id]
      if (!tgt) continue

      ctx.beginPath()
      ctx.arc(toX(tgt.x), toY(tgt.y), 9, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(255,255,255,0.85)'
      ctx.lineWidth = 2
      ctx.stroke()

      if (act) {
        const dx = act.x - tgt.x
        const dy = act.y - tgt.y
        const err = Math.hypot(dx, dy)
        ctx.beginPath()
        ctx.arc(toX(act.x), toY(act.y), 6, 0, Math.PI * 2)
        const hue = id % 360
        ctx.fillStyle = `hsla(${hue}, 70%, 58%, 0.95)`
        ctx.fill()

        if (err > 0.06) {
          ctx.beginPath()
          ctx.moveTo(toX(tgt.x), toY(tgt.y))
          ctx.lineTo(toX(act.x), toY(act.y))
          ctx.strokeStyle = 'rgba(248, 113, 113, 0.85)'
          ctx.lineWidth = 2
          ctx.shadowColor = 'rgba(248, 113, 113, 0.5)'
          ctx.shadowBlur = 6
          ctx.stroke()
          ctx.shadowBlur = 0
        }
      }
    }

    ctx.fillStyle = 'rgba(226, 232, 240, 0.75)'
    ctx.font = '10px Plus Jakarta Sans, system-ui, sans-serif'
    ctx.fillText('○ target  ● you', 10, height - 10)
  }, [frame, width, height])

  return (
    <div className="sway-minimap">
      <div className="sway-minimap__title">Formation radar</div>
      <canvas ref={canvasRef} className="sway-minimap__canvas" width={width} height={height} />
    </div>
  )
}
