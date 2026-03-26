import { useMemo, useRef, useEffect } from 'react'
import type { FeedbackLine } from '../../swayScoring/types'

type Props = {
  lines: FeedbackLine[]
  selectedDancerId: number | null
  onPickLine: (line: FeedbackLine) => void
}

export function FeedbackFeed({ lines, selectedDancerId, onPickLine }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)

  const filtered = useMemo(() => {
    if (selectedDancerId == null) return lines
    return lines.filter((l) => l.dancerId === selectedDancerId)
  }, [lines, selectedDancerId])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [filtered.length])

  return (
    <div className="sway-feed">
      <div className="sway-feed__title">Feedback</div>
      <div ref={scrollRef} className="sway-feed__scroll" role="log" aria-live="polite">
        {filtered.length === 0 ? (
          <p className="sway-feed__empty">No cues for this dancer at loaded frames.</p>
        ) : (
          filtered.map((line) => (
            <button
              key={line.id}
              type="button"
              className="sway-feed__line"
              onClick={() => onPickLine(line)}
              title="Drill: 4s loop @ 0.5×"
            >
              {line.label}
            </button>
          ))
        )}
      </div>
    </div>
  )
}
