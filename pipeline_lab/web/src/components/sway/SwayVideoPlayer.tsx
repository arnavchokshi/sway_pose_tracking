import { useCallback, useEffect, useRef } from 'react'
import { videoContainContentRect } from '../../lib/videoContentRect'
import { drawGhostSkeletonPartial } from '../../swayScoring/ghostDraw'
import { frameAtTime, useSwayScoringStore } from '../../swayScoring/store'
import type { ScoringFramePayload } from '../../swayScoring/types'

type Props = {
  videoSrc: string | undefined
  onDuration: (sec: number) => void
  onVideoMount?: (el: HTMLVideoElement | null) => void
}

function teamAverageScore(frame: ScoringFramePayload | null): number {
  if (!frame?.dancers.length) return 0
  const s = frame.dancers.reduce((a, d) => a + d.aggregate_score, 0)
  return s / frame.dancers.length
}

export function SwayVideoPlayer({ videoSrc, onDuration, onVideoMount }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const canvasLayoutRef = useRef<{ cw: number; ch: number; dpr: number }>({ cw: 0, ch: 0, dpr: 0 })
  const paintRef = useRef<() => void>(() => {})

  const timeline = useSwayScoringStore((s) => s.timeline)
  const viewMode = useSwayScoringStore((s) => s.viewMode)
  const weakLinkEnabled = useSwayScoringStore((s) => s.weakLinkEnabled)
  const selectedDancerId = useSwayScoringStore((s) => s.selectedDancerId)
  const drillLoop = useSwayScoringStore((s) => s.drillLoop)
  const basePlaybackRate = useSwayScoringStore((s) => s.basePlaybackRate)
  const setCurrentTimeSec = useSwayScoringStore((s) => s.setCurrentTimeSec)
  const currentTimeSec = useSwayScoringStore((s) => s.currentTimeSec)

  const bindVideoRef = useCallback(
    (el: HTMLVideoElement | null) => {
      videoRef.current = el
      onVideoMount?.(el)
    },
    [onVideoMount],
  )

  const paint = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    const rect = wrapRef.current?.getBoundingClientRect()
    const cw = Math.round(rect?.width ?? video.clientWidth)
    const ch = Math.round(rect?.height ?? video.clientHeight)
    if (cw <= 0 || ch <= 0) return

    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const lay = canvasLayoutRef.current
    if (lay.cw !== cw || lay.ch !== ch || lay.dpr !== dpr) {
      lay.cw = cw
      lay.ch = ch
      lay.dpr = dpr
      canvas.width = Math.round(cw * dpr)
      canvas.height = Math.round(ch * dpr)
      canvas.style.width = `${cw}px`
      canvas.style.height = `${ch}px`
    }

    const ctx2 = canvas.getContext('2d')
    if (!ctx2) return
    ctx2.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx2.clearRect(0, 0, cw, ch)

    const content =
      videoContainContentRect(video) ??
      (() => {
        const scale = 1
        return { left: 0, top: 0, w: cw, h: ch, scale }
      })()

    const hasFrame =
      Boolean(videoSrc) &&
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      Number.isFinite(video.duration) &&
      video.duration > 0
    const vDur = hasFrame ? video.duration : 0
    /** When scrubbed past clip end, keep overlays on API timeline while video shows last frame. */
    const t =
      hasFrame && currentTimeSec <= vDur + 1e-3 ? video.currentTime : currentTimeSec
    const frame = frameAtTime(timeline, t)
    const dimPaused =
      (hasFrame ? video.paused : true) && viewMode === 'captain' && weakLinkEnabled
    if (dimPaused) {
      ctx2.fillStyle = 'rgba(0,0,0,0.35)'
      ctx2.fillRect(content.left, content.top, content.w, content.h)
    }

    if (viewMode === 'captain' && weakLinkEnabled && (hasFrame ? video.paused : true) && frame) {
      const avg = teamAverageScore(frame)
      ctx2.save()
      for (const d of frame.dancers) {
        if (d.aggregate_score >= avg) continue
        const b = d.bbox_norm
        if (!b) continue
        const x = content.left + b.x * content.w
        const y = content.top + b.y * content.h
        const w = b.w * content.w
        const h = b.h * content.h
        ctx2.strokeStyle = 'rgba(248, 113, 113, 0.95)'
        ctx2.lineWidth = 3
        ctx2.shadowColor = 'rgba(248, 113, 113, 0.75)'
        ctx2.shadowBlur = 22
        ctx2.strokeRect(x, y, w, h)
      }
      ctx2.restore()
    }

    if (viewMode === 'dancer' && frame && selectedDancerId != null) {
      const dancer = frame.dancers.find((x) => x.id === selectedDancerId)
      const joints = dancer?.spatial_errors.map((e) => e.joint) ?? []
      const kp = dancer?.golden_keypoints_norm
      if (dancer && kp && joints.length > 0) {
        drawGhostSkeletonPartial({
          ctx: ctx2,
          contentLeft: content.left,
          contentTop: content.top,
          contentW: content.w,
          contentH: content.h,
          keypointsNorm: kp,
          errorJointNames: joints,
        })
      }
    }
  }, [timeline, viewMode, weakLinkEnabled, selectedDancerId, currentTimeSec, videoSrc])

  useEffect(() => {
    paintRef.current = paint
  }, [paint])

  useEffect(() => {
    let id = 0
    const loop = () => {
      paint()
      id = requestAnimationFrame(loop)
    }
    id = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(id)
  }, [paint])

  useEffect(() => {
    const el = wrapRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => paintRef.current())
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const onResize = () => paintRef.current()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const onTimeUpdate = () => {
    const video = videoRef.current
    if (!video) return
    const vDur = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : Infinity
    let t = video.currentTime
    if (drillLoop) {
      const loopEnd = Math.min(drillLoop.endSec, vDur)
      const loopStart = Math.min(drillLoop.startSec, loopEnd)
      if (t < loopStart) {
        t = loopStart
        video.currentTime = t
      } else if (loopEnd > loopStart && t >= loopEnd - 0.05) {
        t = loopStart
        video.currentTime = t
      }
    }
    const capped = Number.isFinite(vDur) ? Math.min(t, vDur) : t
    setCurrentTimeSec(capped)
  }

  useEffect(() => {
    const v = videoRef.current
    if (v) v.playbackRate = basePlaybackRate
  }, [basePlaybackRate])

  useEffect(() => {
    const v = videoRef.current
    if (!v || !videoSrc || !Number.isFinite(v.duration) || v.duration <= 0) return
    if (currentTimeSec > v.duration + 0.02) return
    if (Math.abs(v.currentTime - currentTimeSec) > 0.1) {
      v.currentTime = currentTimeSec
    }
  }, [currentTimeSec, videoSrc])

  return (
    <div
      ref={wrapRef}
      className={`sway-video-wrap${viewMode === 'dancer' ? ' sway-video-wrap--dancer' : ''}`}
    >
      <video
        ref={bindVideoRef}
        className="sway-video"
        src={videoSrc}
        playsInline
        controls
        onLoadedMetadata={(e) => {
          const d = e.currentTarget.duration
          if (Number.isFinite(d)) onDuration(d)
        }}
        onTimeUpdate={onTimeUpdate}
        onPlay={() => useSwayScoringStore.getState().setIsPlaying(true)}
        onPause={() => useSwayScoringStore.getState().setIsPlaying(false)}
      />
      <canvas ref={canvasRef} className="sway-video-canvas" aria-hidden />
    </div>
  )
}
