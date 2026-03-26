/** Map video pixel space onto the letterboxed `object-fit: contain` video element. */

export function videoContainContentRect(video: HTMLVideoElement): {
  left: number
  top: number
  w: number
  h: number
  scale: number
} | null {
  const vw = video.videoWidth
  const vh = video.videoHeight
  const cw = video.clientWidth
  const ch = video.clientHeight
  if (!vw || !vh || !cw || !ch) return null
  const scale = Math.min(cw / vw, ch / vh)
  const dw = vw * scale
  const dh = vh * scale
  return {
    left: (cw - dw) / 2,
    top: (ch - dh) / 2,
    w: dw,
    h: dh,
    scale,
  }
}

export function clientPointToVideoPixel(
  video: HTMLVideoElement,
  clientX: number,
  clientY: number,
): [number, number] | null {
  const rect = video.getBoundingClientRect()
  const relX = clientX - rect.left
  const relY = clientY - rect.top
  const c = videoContainContentRect(video)
  if (!c) return null
  const x = (relX - c.left) / c.scale
  const y = (relY - c.top) / c.scale
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  if (x < 0 || y < 0 || x > video.videoWidth || y > video.videoHeight) return null
  return [x, y]
}
