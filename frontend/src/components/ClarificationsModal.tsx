import { useEffect } from 'react'
import type { ClarificationLogRound } from '../types'

function stageLabel(stage: number): string {
  if (stage === 0) return 'No questions'
  if (stage === 2) return 'Stage 2'
  return 'Stage 1'
}

function featureKey(r: ClarificationLogRound): string {
  return `${r.feature_index ?? 1}|${r.feature_total ?? 1}|${r.feature_label ?? ''}`
}

export function ClarificationsModal({
  open,
  onClose,
  log,
}: {
  open: boolean
  onClose: () => void
  log: ClarificationLogRound[]
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const hasMulti =
    log.length > 0 &&
    log.some((r) => (r.feature_total ?? 1) > 1 || (r.feature_index != null && r.feature_index > 1))

  return (
    <div
      className="ba-modal-overlay"
      role="presentation"
      onClick={onClose}
      aria-hidden={!open}
    >
      <div
        className="ba-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="ba-clar-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="ba-modal-header">
          <h2 id="ba-clar-modal-title">View Clarifications</h2>
          <button type="button" className="ba-modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="ba-modal-body">
          {!log.length ? (
            <p className="hint-muted">No clarification answers were recorded for this run yet.</p>
          ) : (
            <div className="ba-clar-log">
              {log.map((round, ri) => {
                const prev = ri > 0 ? log[ri - 1] : null
                const fk = featureKey(round)
                const prevFk = prev ? featureKey(prev) : ''
                const showFeatureHeader = hasMulti && fk !== prevFk
                return (
                  <section key={ri} className="ba-clar-round">
                    {showFeatureHeader ? (
                      <h3 className="ba-clar-feature-heading">
                        Feature {round.feature_index ?? 1} of {round.feature_total ?? 1}
                        {round.feature_label ? `: ${round.feature_label}` : ''}
                      </h3>
                    ) : null}
                    <p className="ba-clar-stage">
                      <strong>{stageLabel(round.stage)}</strong>
                    </p>
                    {round.note && !round.items.length ? (
                      <p className="hint-muted">{round.note}</p>
                    ) : null}
                    {round.items.map((it, ii) => (
                      <div key={ii} className="ba-clar-qa">
                        <p className="ba-clar-q">
                          <span className="ba-clar-q-label">Q:</span> {it.question}
                        </p>
                        <p className="ba-clar-a">
                          <span className="ba-clar-a-label">A:</span> {it.answer}
                        </p>
                      </div>
                    ))}
                  </section>
                )
              })}
            </div>
          )}
        </div>
        <footer className="ba-modal-footer">
          <button type="button" className="btn btn-primary" onClick={onClose}>
            Close
          </button>
        </footer>
      </div>
    </div>
  )
}
