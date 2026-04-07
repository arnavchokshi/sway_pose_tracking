/**
 * Check that a run output URL is reachable (file exists on server).
 * Uses a tiny Range request so we do not download full video bodies.
 */
export async function verifyOutputFileExists(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, {
      method: 'GET',
      headers: { Range: 'bytes=0-0' },
      cache: 'no-store',
    })
    if (r.status === 404) return false
    return r.ok || r.status === 206
  } catch {
    return false
  }
}
