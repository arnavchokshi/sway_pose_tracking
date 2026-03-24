const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'])

/** `accept` attribute for `<input type="file">` — keep in sync with drop validation. */
export const VIDEO_ACCEPT_ATTR = 'video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v'

export function isProbableVideoFile(file: File): boolean {
  if (file.type.startsWith('video/')) return true
  const i = file.name.lastIndexOf('.')
  if (i < 0) return false
  return VIDEO_EXTENSIONS.has(file.name.slice(i + 1).toLowerCase())
}
