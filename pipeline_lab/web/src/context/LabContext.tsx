import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from 'react'
import {
  clearLabLocalStorage,
  clearLabVideoFile,
  persistDrafts,
  persistSessionRunIds,
  restoreLabSession,
  saveLabVideoFile,
} from '../lib/labPersistence'

export type DraftParentRef =
  | { kind: 'none' }
  | { kind: 'single'; parentClientId: string }
  /** Queue one continuation run per Phase-1 root (same recipe/fields). */
  | { kind: 'all_roots' }
  /** Queue one run per draft node at the given tree level (0 = same as all_roots for typical trees). */
  | { kind: 'all_at_level'; level: number }

export type DraftRun = {
  clientId: string
  recipeName: string
  fields: Record<string, unknown>
  /** Checkpoint segment: 0 = stop after phase 1, then 1, 2, … for later boundaries. */
  treeLevel: number
  parentRef: DraftParentRef
}

type LabContextValue = {
  videoFile: File | null
  videoLabel: string
  setVideo: (file: File | null) => void
  drafts: DraftRun[]
  setDrafts: React.Dispatch<React.SetStateAction<DraftRun[]>>
  updateDraft: (
    clientId: string,
    patch: Partial<Pick<DraftRun, 'recipeName' | 'fields' | 'treeLevel' | 'parentRef'>>,
  ) => void
  addDraft: (initial?: Partial<DraftRun>) => void
  removeDraft: (clientId: string) => void
  duplicateDraft: (clientId: string) => void
  sessionRunIds: string[]
  setSessionRunIds: Dispatch<SetStateAction<string[]>>
  clearDrafts: () => void
  clearSession: () => void
  /** False until restore from localStorage/IndexedDB finishes (avoid flashing empty UI). */
  labHydrated: boolean
}

const LabContext = createContext<LabContextValue | null>(null)

function newClientId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `run-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

export function LabProvider({ children }: { children: ReactNode }) {
  const [videoFile, setVideoFileState] = useState<File | null>(null)
  const [videoLabel, setVideoLabel] = useState('')
  const [drafts, setDrafts] = useState<DraftRun[]>([])
  const [sessionRunIds, setSessionRunIds] = useState<string[]>([])
  const [labHydrated, setLabHydrated] = useState(false)

  const persistReady = useRef(false)
  const draftsDebounce = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    let cancelled = false
    restoreLabSession()
      .then(({ videoFile: vf, sessionRunIds: ids, drafts: d }) => {
        if (cancelled) return
        if (vf) {
          setVideoFileState(vf)
          setVideoLabel(vf.name)
        }
        if (ids.length > 0) setSessionRunIds(ids)
        if (d.length > 0) {
          setDrafts(
            d.map((row) => ({
              clientId: row.clientId,
              recipeName: row.recipeName,
              fields: row.fields,
              treeLevel: row.treeLevel ?? 0,
              parentRef: row.parentRef ?? { kind: 'none' },
            })),
          )
        }
      })
      .catch(() => {
        /* private mode / IDB blocked */
      })
      .finally(() => {
        if (!cancelled) {
          persistReady.current = true
          setLabHydrated(true)
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!persistReady.current) return
    persistSessionRunIds(sessionRunIds)
  }, [sessionRunIds])

  useEffect(() => {
    if (!persistReady.current) return
    if (draftsDebounce.current) clearTimeout(draftsDebounce.current)
    draftsDebounce.current = setTimeout(() => {
      persistDrafts(drafts)
    }, 350)
    return () => {
      if (draftsDebounce.current) clearTimeout(draftsDebounce.current)
    }
  }, [drafts])

  useEffect(() => {
    if (!persistReady.current) return
    if (videoFile) {
      void saveLabVideoFile(videoFile).catch(() => {
        /* quota */
      })
    } else {
      void clearLabVideoFile().catch(() => {})
    }
  }, [videoFile])

  const setVideo = useCallback((file: File | null) => {
    setVideoFileState(file)
    setVideoLabel(file?.name ?? '')
    setDrafts([])
    setSessionRunIds([])
  }, [])

  const updateDraft = useCallback(
    (
      clientId: string,
      patch: Partial<Pick<DraftRun, 'recipeName' | 'fields' | 'treeLevel' | 'parentRef'>>,
    ) => {
      setDrafts((prev) => prev.map((d) => (d.clientId === clientId ? { ...d, ...patch } : d)))
    },
    [],
  )

  const addDraft = useCallback((initial?: Partial<DraftRun>) => {
    setDrafts((prev) => {
      const n = prev.filter((x) => x.parentRef.kind === 'none').length + 1
      const d: DraftRun = {
        clientId: newClientId(),
        recipeName: initial?.recipeName ?? `Phase 1 — ${n}`,
        fields: { ...(initial?.fields ?? {}) },
        treeLevel: initial?.treeLevel ?? 0,
        parentRef: initial?.parentRef ?? { kind: 'none' },
      }
      return [...prev, d]
    })
  }, [])

  const removeDraft = useCallback((clientId: string) => {
    setDrafts((prev) => {
      if (prev.length <= 1) return prev
      return prev.filter((d) => d.clientId !== clientId)
    })
  }, [])

  const duplicateDraft = useCallback((clientId: string) => {
    setDrafts((prev) => {
      const src = prev.find((d) => d.clientId === clientId)
      if (!src) return prev
      const copy: DraftRun = {
        clientId: newClientId(),
        recipeName: `${src.recipeName} (copy)`,
        fields: { ...src.fields },
        treeLevel: src.treeLevel,
        parentRef: src.parentRef,
      }
      const idx = prev.findIndex((d) => d.clientId === clientId)
      const next = [...prev]
      next.splice(idx + 1, 0, copy)
      return next
    })
  }, [])

  const clearDrafts = useCallback(() => {
    setDrafts([])
  }, [])

  const clearSession = useCallback(() => {
    clearLabLocalStorage()
    void clearLabVideoFile()
    setVideoFileState(null)
    setVideoLabel('')
    setDrafts([])
    setSessionRunIds([])
  }, [])

  const value = useMemo(
    () => ({
      videoFile,
      videoLabel,
      setVideo,
      drafts,
      setDrafts,
      updateDraft,
      addDraft,
      removeDraft,
      duplicateDraft,
      sessionRunIds,
      setSessionRunIds,
      clearDrafts,
      clearSession,
      labHydrated,
    }),
    [
      videoFile,
      videoLabel,
      setVideo,
      drafts,
      updateDraft,
      addDraft,
      removeDraft,
      duplicateDraft,
      sessionRunIds,
      clearDrafts,
      clearSession,
      labHydrated,
    ],
  )

  return <LabContext.Provider value={value}>{children}</LabContext.Provider>
}

export function useLab() {
  const ctx = useContext(LabContext)
  if (!ctx) throw new Error('useLab must be used within LabProvider')
  return ctx
}
