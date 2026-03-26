/**
 * Restore lab UI after refresh: session run IDs + drafts (localStorage), video blob (IndexedDB).
 */

type PersistedParentRef =
  | { kind: 'none' }
  | { kind: 'single'; parentClientId: string }
  | { kind: 'all_roots' }
  | { kind: 'all_at_level'; level: number }

export type PersistedDraftRun = {
  clientId: string
  recipeName: string
  fields: Record<string, unknown>
  treeLevel?: number
  parentRef?: PersistedParentRef
}

const LS_RUN_IDS = 'sway_lab_session_run_ids'
const LS_BATCH_FILTER = 'sway_lab_session_batch_filter_id'
const LS_DRAFTS = 'sway_lab_drafts'

const IDB_NAME = 'sway-pipeline-lab-v1'
const IDB_STORE = 'kv'
const IDB_VIDEO_KEY = 'lab-input-video'

export type RestoredLabSession = {
  sessionRunIds: string[]
  drafts: PersistedDraftRun[]
  videoFile: File | null
}

function openIdb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1)
    req.onerror = () => reject(req.error ?? new Error('indexedDB open failed'))
    req.onsuccess = () => resolve(req.result)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE)
      }
    }
  })
}

type VideoRow = { name: string; type: string; buffer: ArrayBuffer }

export async function saveLabVideoFile(file: File): Promise<void> {
  const buffer = await file.arrayBuffer()
  const row: VideoRow = {
    name: file.name,
    type: file.type || 'video/mp4',
    buffer,
  }
  const db = await openIdb()
  try {
    await new Promise<void>((res, rej) => {
      const tx = db.transaction(IDB_STORE, 'readwrite')
      tx.oncomplete = () => res()
      tx.onerror = () => rej(tx.error ?? new Error('idb write'))
      tx.objectStore(IDB_STORE).put(row, IDB_VIDEO_KEY)
    })
  } finally {
    db.close()
  }
}

export async function loadLabVideoFile(): Promise<File | null> {
  const db = await openIdb()
  try {
    const row = await new Promise<VideoRow | undefined>((res, rej) => {
      const tx = db.transaction(IDB_STORE, 'readonly')
      const r = tx.objectStore(IDB_STORE).get(IDB_VIDEO_KEY)
      r.onsuccess = () => res(r.result as VideoRow | undefined)
      r.onerror = () => rej(r.error ?? new Error('idb read'))
    })
    if (!row?.buffer || row.buffer.byteLength === 0) return null
    return new File([row.buffer], row.name, { type: row.type || 'video/mp4' })
  } finally {
    db.close()
  }
}

export async function clearLabVideoFile(): Promise<void> {
  const db = await openIdb()
  try {
    await new Promise<void>((res, rej) => {
      const tx = db.transaction(IDB_STORE, 'readwrite')
      tx.oncomplete = () => res()
      tx.onerror = () => rej(tx.error ?? new Error('idb delete'))
      tx.objectStore(IDB_STORE).delete(IDB_VIDEO_KEY)
    })
  } finally {
    db.close()
  }
}

function safeParseRunIds(raw: string | null): string[] {
  if (!raw) return []
  try {
    const v = JSON.parse(raw) as unknown
    if (!Array.isArray(v)) return []
    return v.filter((x): x is string => typeof x === 'string' && x.length > 0)
  } catch {
    return []
  }
}

function safeParseDrafts(raw: string | null): PersistedDraftRun[] {
  if (!raw) return []
  try {
    const v = JSON.parse(raw) as unknown
    if (!Array.isArray(v)) return []
    const out: PersistedDraftRun[] = []
    for (const item of v) {
      if (!item || typeof item !== 'object') continue
      const o = item as Record<string, unknown>
      const clientId = typeof o.clientId === 'string' ? o.clientId : ''
      const recipeName = typeof o.recipeName === 'string' ? o.recipeName : 'Run'
      const fields = o.fields && typeof o.fields === 'object' && o.fields !== null ? (o.fields as Record<string, unknown>) : {}
      if (!clientId) continue
      const treeLevel = typeof o.treeLevel === 'number' && Number.isFinite(o.treeLevel) ? Math.max(0, Math.floor(o.treeLevel)) : 0
      let parentRef: PersistedParentRef = { kind: 'none' }
      const pr = o.parentRef
      if (pr && typeof pr === 'object' && pr !== null) {
        const p = pr as Record<string, unknown>
        const k = p.kind
        if (k === 'single' && typeof p.parentClientId === 'string') {
          parentRef = { kind: 'single', parentClientId: p.parentClientId }
        } else if (k === 'all_roots') {
          parentRef = { kind: 'all_roots' }
        } else if (k === 'all_at_level' && typeof p.level === 'number' && Number.isFinite(p.level)) {
          parentRef = { kind: 'all_at_level', level: Math.max(0, Math.floor(p.level)) }
        }
      }
      out.push({ clientId, recipeName, fields, treeLevel, parentRef })
    }
    return out
  } catch {
    return []
  }
}

export function loadSessionRunIds(): string[] {
  try {
    return safeParseRunIds(localStorage.getItem(LS_RUN_IDS))
  } catch {
    return []
  }
}

export function loadDrafts(): PersistedDraftRun[] {
  try {
    return safeParseDrafts(localStorage.getItem(LS_DRAFTS))
  } catch {
    return []
  }
}

export function persistSessionRunIds(ids: string[]): void {
  try {
    if (ids.length === 0) localStorage.removeItem(LS_RUN_IDS)
    else localStorage.setItem(LS_RUN_IDS, JSON.stringify(ids))
  } catch {
    /* quota / private mode */
  }
}

/** Persisted so a refresh on `/` still expands the same Lab batch (CLI adds runs with the same batch_id). */
export function loadSessionBatchFilterId(): string | null {
  try {
    const raw = localStorage.getItem(LS_BATCH_FILTER)
    const t = raw?.trim()
    return t && t.length > 0 ? t : null
  } catch {
    return null
  }
}

export function persistSessionBatchFilterId(id: string | null): void {
  try {
    const t = id?.trim()
    if (!t) localStorage.removeItem(LS_BATCH_FILTER)
    else localStorage.setItem(LS_BATCH_FILTER, t)
  } catch {
    /* quota / private mode */
  }
}

export function persistDrafts(drafts: PersistedDraftRun[]): void {
  try {
    if (drafts.length === 0) localStorage.removeItem(LS_DRAFTS)
    else localStorage.setItem(LS_DRAFTS, JSON.stringify(drafts))
  } catch {
    /* quota */
  }
}

export function clearLabLocalStorage(): void {
  try {
    localStorage.removeItem(LS_RUN_IDS)
    localStorage.removeItem(LS_BATCH_FILTER)
    localStorage.removeItem(LS_DRAFTS)
  } catch {
    /* ignore */
  }
}

export async function restoreLabSession(): Promise<RestoredLabSession> {
  const [videoFile, sessionRunIds, drafts] = await Promise.all([
    loadLabVideoFile().catch(() => null),
    Promise.resolve(loadSessionRunIds()),
    Promise.resolve(loadDrafts()),
  ])
  return { videoFile, sessionRunIds, drafts }
}
