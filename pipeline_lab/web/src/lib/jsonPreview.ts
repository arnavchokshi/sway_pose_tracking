/** Pretty-print JSON for UI (cap length for huge prune lists). */
export function safeJsonPreview(obj: unknown, maxLen: number): string {
  try {
    const s = JSON.stringify(obj, null, 2)
    return s.length > maxLen ? `${s.slice(0, maxLen)}\n… (truncated for display)` : s
  } catch {
    return String(obj)
  }
}
