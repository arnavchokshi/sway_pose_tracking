import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Sparkles } from 'lucide-react'
import { API } from '../types'

type LiveCaps = {
  has_phase3_bundle: boolean
  has_phase1_npz: boolean
  has_phase1_pre_classical: boolean
  input_video_relpath: string | null
}

type LiveFrame = {
  frame_idx: number
  boxes: number[][]
  track_ids?: number[]
  confs?: number[]
}

function fileUrl(runId: string, rel: string) {
  return `${API}/api/runs/${runId}/file/${rel.replace(/^\//, '')}`
}

export function LiveSandboxPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [caps, setCaps] = useState<LiveCaps | null>(null)
  const [mode, setMode] = useState<'phase1_nms' | 'phase4_prune'>('phase1_nms')
  const [iou, setIou] = useState(0.5)
  const [shortFrac, setShortFrac] = useState(0.2)
  const [frames, setFrames] = useState<LiveFrame[]>([])
  const [totalFrames, setTotalFrames] = useState(300)
  const [frameEnd, setFrameEnd] = useState(120)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!id) return
    fetch(`${API}/api/runs/${id}/live_capabilities`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<LiveCaps>
      })
      .then(setCaps)
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
  }, [id])

  useEffect(() => {
    if (!id || !caps) return
    if (mode === 'phase1_nms' && !caps.has_phase1_pre_classical) return
    if (mode === 'phase4_prune' && !caps.has_phase3_bundle) return
    const t = window.setTimeout(() => {
      setBusy(true)
      setErr(null)
      const body =
        mode === 'phase1_nms'
          ? {
              mode: 'phase1_nms',
              sway_pretrack_nms_iou: iou,
              frame_start: 0,
              frame_end: Math.min(frameEnd, 400),
            }
          : {
              mode: 'phase4_prune',
              params_overrides: { SHORT_TRACK_MIN_FRAC: shortFrac },
              frame_start: 0,
              frame_end: Math.min(frameEnd, 400),
            }
      fetch(`${API}/api/runs/${id}/live_preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
        .then(async (r) => {
          if (!r.ok) {
            const tx = await r.text()
            throw new Error(tx || `HTTP ${r.status}`)
          }
          return r.json() as Promise<{ frames: LiveFrame[]; total_frames?: number }>
        })
        .then((j) => {
          setFrames(Array.isArray(j.frames) ? j.frames : [])
          if (typeof j.total_frames === 'number' && j.total_frames > 0) {
            setTotalFrames(j.total_frames)
          }
        })
        .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
        .finally(() => setBusy(false))
    }, 220)
    return () => window.clearTimeout(t)
  }, [id, caps, mode, iou, shortFrac, frameEnd])

  const drawOverlay = useCallback(() => {
    const v = videoRef.current
    const c = canvasRef.current
    if (!v || !c || !frames.length) return
    const fps = 30
    const fi = Math.max(0, Math.min(Math.floor(v.currentTime * fps), totalFrames - 1))
    const row = frames.find((f) => f.frame_idx === fi) ?? frames[Math.min(frames.length - 1, fi)]
    if (!row) return
    const vw = v.videoWidth
    const vh = v.videoHeight
    if (!vw || !vh) return
    const cw = v.clientWidth
    const ch = v.clientHeight
    if (!cw || !ch) return
    const dpr = window.devicePixelRatio || 1
    c.width = Math.floor(cw * dpr)
    c.height = Math.floor(ch * dpr)
    const ctx = c.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cw, ch)
    const sx = cw / vw
    const sy = ch / vh
    ctx.lineWidth = 2
    ctx.strokeStyle = 'rgba(0, 255, 180, 0.95)'
    ctx.font = '14px ui-sans-serif, system-ui'
    for (let i = 0; i < row.boxes.length; i++) {
      const b = row.boxes[i]
      if (!b || b.length < 4) continue
      const [x1, y1, x2, y2] = b
      const X1 = x1 * sx
      const Y1 = y1 * sy
      const X2 = x2 * sx
      const Y2 = y2 * sy
      ctx.strokeRect(X1, Y1, X2 - X1, Y2 - Y1)
      const tid = row.track_ids?.[i]
      const cf = row.confs?.[i]
      const lab =
        tid !== undefined ? `id ${tid}` : cf !== undefined ? `${cf.toFixed(2)}` : ''
      if (lab) {
        ctx.fillStyle = 'rgba(0,0,0,0.55)'
        ctx.fillRect(X1, Y1 - 16, Math.min(120, lab.length * 8), 16)
        ctx.fillStyle = '#e8fff8'
        ctx.fillText(lab, X1 + 2, Y1 - 3)
      }
    }
  }, [frames, totalFrames])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const on = () => drawOverlay()
    v.addEventListener('timeupdate', on)
    v.addEventListener('seeked', on)
    v.addEventListener('loadeddata', on)
    return () => {
      v.removeEventListener('timeupdate', on)
      v.removeEventListener('seeked', on)
      v.removeEventListener('loadeddata', on)
    }
  }, [drawOverlay])

  if (!id) return null

  const vidSrc = caps?.input_video_relpath ? fileUrl(id, caps.input_video_relpath) : null
  const canPhase1 = caps?.has_phase1_pre_classical
  const canPhase4 = caps?.has_phase3_bundle

  const fieldLabel: CSSProperties = {
    fontSize: '0.72rem',
    fontWeight: 700,
    letterSpacing: '0.07em',
    textTransform: 'uppercase',
    color: 'var(--text-muted)',
  }

  return (
    <div
      className="watch-page live-sandbox"
      style={{
        maxWidth: 1100,
        margin: '0 auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem',
      }}
    >
      <header
        className="glass-panel"
        style={{
          padding: '0.85rem 1.15rem',
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          flexWrap: 'wrap',
        }}
      >
        <button type="button" className="btn btn--compact" onClick={() => navigate(`/watch/${id}`)}>
          <ArrowLeft size={18} aria-hidden /> Watch
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', minWidth: 0 }}>
          <Sparkles size={20} style={{ flexShrink: 0, color: 'var(--halo-cyan)', opacity: 0.95 }} aria-hidden />
          <h1 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600, letterSpacing: '-0.02em' }}>Live sandbox</h1>
        </div>
      </header>

      <p className="sub" style={{ marginTop: 0, marginBottom: 0 }}>
        Recompute detection NMS (IoU) or pre-pose pruning on cached pipeline data. Scrub the video to refresh overlays
        for each frame in the preview.
      </p>

      {err && (
        <div className="glass-panel" style={{ padding: '0.85rem 1rem', color: '#f87171', borderColor: 'rgba(248, 113, 113, 0.35)' }}>
          {err}
        </div>
      )}

      {!caps && (
        <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '0.9rem' }}>Loading capabilities…</p>
      )}

      {caps && (
        <>
          <div
            className="glass-panel"
            style={{
              padding: '1.15rem 1.35rem',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '1.25rem 1.75rem',
              alignItems: 'end',
            }}
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', minWidth: 0 }}>
              <span style={fieldLabel}>Mode</span>
              <select
                className="text-input"
                value={mode}
                onChange={(e) => setMode(e.target.value as 'phase1_nms' | 'phase4_prune')}
                disabled={busy}
                style={{ padding: '0.5rem 0.75rem', fontSize: '0.875rem', cursor: busy ? 'wait' : 'pointer' }}
              >
                <option value="phase1_nms" disabled={!canPhase1}>
                  Phase 1 — pre-track NMS (IoU){!canPhase1 ? ' (no pre-NMS cache)' : ''}
                </option>
                <option value="phase4_prune" disabled={!canPhase4}>
                  Phase 4 — pre-pose prune{!canPhase4 ? ' (no phase3 bundle)' : ''}
                </option>
              </select>
            </div>
            {mode === 'phase1_nms' && (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem',
                  gridColumn: 'span 2',
                  minWidth: 'min(100%, 320px)',
                }}
              >
                <span style={fieldLabel}>Classical NMS IoU</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                  <input
                    type="range"
                    className="config-range-input"
                    min={0.15}
                    max={0.95}
                    step={0.01}
                    value={iou}
                    onChange={(e) => setIou(parseFloat(e.target.value))}
                    style={{ flex: 1, minWidth: 0, accentColor: 'var(--halo-purple)' }}
                  />
                  <span
                    style={{
                      fontVariantNumeric: 'tabular-nums',
                      minWidth: '2.75rem',
                      textAlign: 'right',
                      fontSize: '0.9rem',
                      color: 'var(--text-main)',
                    }}
                  >
                    {iou.toFixed(2)}
                  </span>
                </div>
              </div>
            )}
            {mode === 'phase4_prune' && (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem',
                  gridColumn: 'span 2',
                  minWidth: 'min(100%, 320px)',
                }}
              >
                <span style={fieldLabel}>SHORT_TRACK_MIN_FRAC</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                  <input
                    type="range"
                    className="config-range-input"
                    min={0.05}
                    max={0.45}
                    step={0.01}
                    value={shortFrac}
                    onChange={(e) => setShortFrac(parseFloat(e.target.value))}
                    style={{ flex: 1, minWidth: 0, accentColor: 'var(--halo-purple)' }}
                  />
                  <span
                    style={{
                      fontVariantNumeric: 'tabular-nums',
                      minWidth: '2.75rem',
                      textAlign: 'right',
                      fontSize: '0.9rem',
                    }}
                  >
                    {shortFrac.toFixed(2)}
                  </span>
                </div>
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', minWidth: 0 }}>
              <span style={fieldLabel}>Preview frames (0–N)</span>
              <input
                type="number"
                className="text-input config-slider-num"
                min={30}
                max={800}
                step={10}
                value={frameEnd}
                onChange={(e) => setFrameEnd(parseInt(e.target.value, 10) || 120)}
                style={{ width: '100%', maxWidth: 120, padding: '0.5rem 0.75rem', fontSize: '0.875rem' }}
              />
            </div>
            {busy && (
              <div style={{ display: 'flex', alignItems: 'flex-end', paddingBottom: '0.35rem' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Updating…</span>
              </div>
            )}
          </div>

          {vidSrc ? (
            <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
              <video
                ref={videoRef}
                src={vidSrc}
                controls
                playsInline
                style={{ width: '100%', maxHeight: '72vh', background: '#111' }}
              />
              <canvas
                ref={canvasRef}
                style={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                }}
              />
            </div>
          ) : (
            <p>No input video on disk for this run.</p>
          )}
        </>
      )}
    </div>
  )
}
