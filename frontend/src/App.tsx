import { useCallback, useEffect, useId, useRef, useState } from 'react'
import {
  continueEmptyClarification,
  createSession,
  fetchHealth,
  fetchMeta,
  generateRequirement,
  resetSession,
  submitClarification,
} from './api'
import { ArtifactsPanel } from './components/ArtifactsPanel'
import { SiteFooter } from './components/SiteFooter'
import { SiteHeader } from './components/SiteHeader'
import type { ApiMeta, ClarificationQuestion, PipelineState } from './types'

const PLACEHOLDER = '— Select an option —'

/** First segment before fallback description (em/en dash), trimmed for UI. */
function summarizeLlmRouting(llm: string): string {
  const s = (llm || '').trim()
  if (!s) return ''
  const parts = s.split(/\s+[—–-]\s+/)
  const main = (parts[0] ?? s).trim()
  return main.length > 120 ? `${main.slice(0, 117)}…` : main
}

function UnderstandingView({
  u,
  heading = 'Requirement understanding',
}: {
  u: Record<string, unknown>
  heading?: string
}) {
  const impact = u.impact
  const list = Array.isArray(impact)
    ? (impact as unknown[]).map(String).filter(Boolean)
    : typeof impact === 'string' && impact.trim()
      ? impact.split(',').map((s) => s.trim()).filter(Boolean)
      : []
  return (
    <section className="card ba-understanding-card">
      <h3>{heading}</h3>
      <dl className="kv">
        <dt>Requirement type</dt>
        <dd>{String(u.type ?? '—')}</dd>
        <dt>Actor</dt>
        <dd>{String(u.actor ?? '—')}</dd>
        {u.secondary_actor ? (
          <>
            <dt>Secondary actor</dt>
            <dd>{String(u.secondary_actor)}</dd>
          </>
        ) : null}
        <dt>Action</dt>
        <dd>{String(u.action ?? '—')}</dd>
        <dt>Domain</dt>
        <dd>{String(u.domain ?? '—')}</dd>
      </dl>
      <h4>Impacted systems</h4>
      {list.length ? (
        <ul>
          {list.map((x, i) => (
            <li key={i}>{x}</li>
          ))}
        </ul>
      ) : (
        <p className="hint-muted">No impacted systems identified.</p>
      )}
    </section>
  )
}

function ClarificationForm({
  stage,
  questions,
  stage2Notes,
  otherLabel,
  busy,
  onSubmit,
}: {
  stage: number
  questions: ClarificationQuestion[]
  stage2Notes: string[] | null
  otherLabel: string
  busy: boolean
  onSubmit: (answers: Record<string, string>) => void
}) {
  const [selection, setSelection] = useState<Record<string, string>>({})
  const [otherText, setOtherText] = useState<Record<string, string>>({})

  useEffect(() => {
    const initSel: Record<string, string> = {}
    const initO: Record<string, string> = {}
    for (const q of questions) {
      initSel[q.category] = PLACEHOLDER
      initO[q.category] = ''
    }
    setSelection(initSel)
    setOtherText(initO)
  }, [questions, stage])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const answers: Record<string, string> = {}
    for (const q of questions) {
      const sel = (selection[q.category] ?? '').trim()
      if (!sel || sel === PLACEHOLDER) {
        answers[q.category] = ''
      } else if (sel === otherLabel) {
        answers[q.category] = (otherText[q.category] ?? '').trim()
      } else {
        answers[q.category] = sel
      }
    }
    onSubmit(answers)
  }

  return (
    <form className="card clarification" onSubmit={handleSubmit}>
      <h3>Clarification {stage === 2 ? '(stage 2)' : '(stage 1)'}</h3>
      <p className="hint-muted">
        Choose an option for each question. Leave <strong>{PLACEHOLDER}</strong> if you do not want
        to answer that item. Use <strong>{otherLabel}</strong> only when you need a custom answer.
      </p>
      {stage2Notes?.length ? (
        <aside className="ba-clarification-notes">
          <strong>Stage 2 context</strong>
          <ul>
            {stage2Notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </aside>
      ) : null}
      {questions.map((q) => {
        const opts = [PLACEHOLDER, ...q.options, otherLabel]
        const sel = selection[q.category] ?? PLACEHOLDER
        return (
          <fieldset key={q.category} className="ba-q-block">
            <legend>
              <span className="ba-q-cat">{q.category}</span>
            </legend>
            <p className="ba-q-text">{q.question}</p>
            <label className="sr-only" htmlFor={`sel-${q.category}`}>
              Answer for {q.category}
            </label>
            <select
              id={`sel-${q.category}`}
              value={sel}
              onChange={(e) =>
                setSelection((s) => ({ ...s, [q.category]: e.target.value }))
              }
            >
              {opts.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
            {sel === otherLabel ? (
              <textarea
                rows={2}
                className="ba-other-input"
                placeholder="Custom answer"
                value={otherText[q.category] ?? ''}
                onChange={(e) =>
                  setOtherText((t) => ({ ...t, [q.category]: e.target.value }))
                }
              />
            ) : null}
          </fieldset>
        )
      })}
      <div className="btn-group">
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? 'Working…' : 'Submit answers & continue'}
        </button>
      </div>
    </form>
  )
}

function AppShell({
  subtitle,
  children,
}: {
  subtitle?: string
  children: React.ReactNode
}) {
  return (
    <>
      <SiteHeader subtitle={subtitle} />
      <main className="app-main ba-main">
        <div className="app-main-inner">{children}</div>
      </main>
      <SiteFooter />
    </>
  )
}

export default function App() {
  const [meta, setMeta] = useState<ApiMeta | null>(null)
  const [state, setState] = useState<PipelineState | null>(null)
  const [bootError, setBootError] = useState<string | null>(null)
  const [text, setText] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [artifactLabel, setArtifactLabel] = useState('All Artifacts')
  const [acFormat, setAcFormat] = useState('declarative')
  const [busy, setBusy] = useState(false)
  const [apiReachable, setApiReachable] = useState(true)
  const fileInputId = useId()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const otherLabel = meta?.clarification_other_option_label ?? 'Other (type your answer below)'

  const bootstrap = useCallback(async () => {
    setBootError(null)
    try {
      const m = await fetchMeta()
      setMeta(m)
      if (m.artifact_scope_labels.length) {
        setArtifactLabel(m.artifact_scope_labels[0])
      }
      const defAc = m.acceptance_criteria_formats[0]?.value ?? 'declarative'
      setAcFormat(defAc)
      const { session_id, state: s } = await createSession()
      setApiReachable(true)
      setState({ ...s, session_id })
    } catch (e) {
      setApiReachable(false)
      setBootError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    void bootstrap()
  }, [bootstrap])

  useEffect(() => {
    if (!state?.session_id) return
    let cancelled = false
    const check = async () => {
      const ok = await fetchHealth()
      if (!cancelled) setApiReachable(ok)
    }
    void check()
    const t = window.setInterval(() => void check(), 30_000)
    return () => {
      cancelled = true
      window.clearInterval(t)
    }
  }, [state?.session_id])

  const sessionId = state?.session_id

  /** Clear prior run output so a new Generate does not show stale artifacts until the response arrives. */
  const clearPipelineResultState = useCallback(() => {
    setState((s) =>
      s
        ? {
            ...s,
            error: null,
            intake_open_items: [],
            original_requirement_text: '',
            multi_feature: null,
            multi_feature_results: null,
            understood: null,
            clarification: null,
            needs_continue_empty_clarification: false,
            refined_text: null,
            refined: null,
            artifacts: null,
            clarification_context: null,
            intake: {},
          }
        : s,
    )
  }, [])

  const runGenerate = async () => {
    if (!sessionId) return
    clearPipelineResultState()
    setBusy(true)
    try {
      const next = await generateRequirement(sessionId, {
        requirementText: text,
        artifactScopeLabel: artifactLabel,
        acceptanceCriteriaFormat: acFormat,
        file,
      })
      setApiReachable(true)
      setState(next)
    } catch (e) {
      void fetchHealth().then((ok) => setApiReachable(ok))
      setState((s) =>
        s ? { ...s, error: e instanceof Error ? e.message : String(e) } : s,
      )
    } finally {
      setBusy(false)
    }
  }

  const runClarification = async (answers: Record<string, string>) => {
    if (!sessionId) return
    setBusy(true)
    try {
      const next = await submitClarification(sessionId, answers)
      setApiReachable(true)
      setState(next)
    } catch (e) {
      void fetchHealth().then((ok) => setApiReachable(ok))
      setState((s) =>
        s ? { ...s, error: e instanceof Error ? e.message : String(e) } : s,
      )
    } finally {
      setBusy(false)
    }
  }

  const runContinueEmpty = async () => {
    if (!sessionId) return
    setBusy(true)
    try {
      const next = await continueEmptyClarification(sessionId)
      setApiReachable(true)
      setState(next)
    } catch (e) {
      void fetchHealth().then((ok) => setApiReachable(ok))
      setState((s) =>
        s ? { ...s, error: e instanceof Error ? e.message : String(e) } : s,
      )
    } finally {
      setBusy(false)
    }
  }

  const runReset = async () => {
    if (!sessionId) return
    setBusy(true)
    try {
      const next = await resetSession(sessionId)
      setApiReachable(true)
      setState(next)
      setText('')
      setFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
    } catch (e) {
      void fetchHealth().then((ok) => setApiReachable(ok))
      setState((s) =>
        s ? { ...s, error: e instanceof Error ? e.message : String(e) } : s,
      )
    } finally {
      setBusy(false)
    }
  }

  if (bootError) {
    return (
      <AppShell subtitle="API disconnected">
        <div className="panel-hero">
          <h2>Connect the API</h2>
          <p>
            Start the FastAPI server from the <code>backend</code> folder (port 8000), ensure the Vite
            dev proxy is enabled, then retry.
          </p>
          <p className="hint-muted">
            <code className="schema-preview" style={{ maxHeight: 'none', marginTop: 0 }}>
              uvicorn api_app:app --reload --port 8000
            </code>
          </p>
        </div>
        <div className="msg err" role="alert">
          {bootError}
        </div>
        <div className="btn-group">
          <button type="button" className="btn btn-primary" onClick={() => void bootstrap()}>
            Retry
          </button>
        </div>
      </AppShell>
    )
  }

  if (!meta || !state) {
    return (
      <AppShell subtitle="Loading workspace">
        <div className="loading">Loading workspace…</div>
      </AppShell>
    )
  }

  const showResults =
    (state.multi_feature_results && state.multi_feature_results.length > 0) ||
    (state.refined && state.artifacts)

  return (
    <AppShell subtitle="Structured requirements → BA artifacts">
      <div className="panel-hero">
        <h2>From rough ideas to testable artifacts</h2>
        <p>
          Paste or upload a requirement, run the pipeline, answer any clarification prompts, then
          review understanding, refinement, epics, user stories, and acceptance criteria
        </p>
      </div>

      {state.error ? (
        <div className="msg err" role="alert">
          {state.error}
        </div>
      ) : null}

      {state.intake_open_items?.length ? (
        <div className="msg info" role="status">
          <strong>Preprocessing:</strong> {state.intake_open_items.length} line(s) treated as pending
          discussion / clarification.
        </div>
      ) : null}

      {state.multi_feature?.active ? (
        <div className="msg info" role="status">
          Multi-feature pipeline: feature <strong>{state.multi_feature.feature_index + 1}</strong> of{' '}
          <strong>{state.multi_feature.total}</strong> (each unit runs understanding → clarification →
          refinement → artifacts).
        </div>
      ) : null}

      <div className="status-bar" id="ba-pipeline">
        <span className={apiReachable ? 'pill ok' : 'pill err'}>
          {apiReachable ? 'API ready' : 'API unreachable'}
        </span>
        <span className={state.error ? 'pill err' : 'pill ok'}>
          {state.error ? 'Pipeline error' : 'Pipeline idle'}
        </span>
      </div>



      <section className="card" id="ba-input">
        <h2>Input</h2>
        <label htmlFor="ba-req-text">Requirement (natural language)</label>
        <textarea
          id="ba-req-text"
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe the requirement, or upload a document below."
        />

        <label htmlFor={fileInputId} style={{ marginTop: 'var(--s-20)' }}>
          Document (optional — overrides text when uploaded)
        </label>
        <div className="file-input-wrapper">
          <input
            ref={fileInputRef}
            id={fileInputId}
            type="file"
            className="file-input-hidden"
            accept=".pdf,.docx,.txt,.md,.text"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <label htmlFor={fileInputId} className="file-input-choose-btn">
            Choose file
          </label>
          <span className="file-input-label">
            {file?.name ?? 'No file selected — PDF, Word, or plain text'}
          </span>
        </div>

        <div className="form-row" style={{ marginTop: 'var(--s-24)' }}>
          <div>
            <label htmlFor="ba-artifact-scope">Artifact scope</label>
            <select
              id="ba-artifact-scope"
              value={artifactLabel}
              onChange={(e) => setArtifactLabel(e.target.value)}
            >
              {meta.artifact_scope_labels.map((lbl) => (
                <option key={lbl} value={lbl}>
                  {lbl}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="ba-ac-format">Acceptance criteria format</label>
            <select
              id="ba-ac-format"
              value={acFormat}
              onChange={(e) => setAcFormat(e.target.value)}
            >
              {meta.acceptance_criteria_formats.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="btn-group">
          <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void runGenerate()}>
            {busy ? 'Running…' : 'Generate'}
          </button>
          <button type="button" className="btn btn-outline" disabled={busy} onClick={() => void runReset()}>
            Clear and start over
          </button>
        </div>
      </section>

      {state.original_requirement_text ? (
        <section className="card">
          <h3>Received requirement</h3>
          <pre className="ba-pre">{state.original_requirement_text}</pre>
        </section>
      ) : null}

      <div id="ba-results">
        {state.multi_feature_results && state.multi_feature_results.length > 0 ? (
          <header className="ba-multi-results-header">
            <h2 className="ba-section-title">Results by feature</h2>
            <div className="msg ok ba-multi-success" role="status">
              <strong>{state.multi_feature_results.length} independent features</strong> were processed. Each went
              through clarification before refinement and artifacts.
            </div>
          </header>
        ) : null}

        {state.multi_feature_results?.map((bundle) => (
          <div key={bundle.index} className="bundle ba-feature-result">
            <h2 className="ba-bundle-title">
              Feature {bundle.index} of {bundle.total}
              {bundle.feature_label ? `: ${bundle.feature_label}` : ''}
            </h2>
            {bundle.requirement_level ? (
              <p className="ba-intake-classification">
                <strong>Intake classification:</strong> {bundle.requirement_level}
              </p>
            ) : null}
            <details className="ba-details">
              <summary>Sub-requirement text</summary>
              <pre className="ba-pre">{bundle.unit_text}</pre>
            </details>
            {bundle.understood ? <UnderstandingView u={bundle.understood} heading="Understanding" /> : null}
            <details className="ba-details ba-llm-routing">
              <summary>Model routing (primary &amp; fallbacks)</summary>
              <p className="hint-muted ba-llm-summary">Primary: {summarizeLlmRouting(state.llm)}</p>
              <pre className="ba-pre">{state.llm}</pre>
            </details>
            <section className="card">
              <h3>Refinement</h3>
              <pre className="ba-pre">{bundle.refined_text}</pre>
            </section>
            <h2 className="ba-section-title ba-generated-heading">Generated artifacts</h2>
            <ArtifactsPanel artifacts={bundle.artifacts} intakeLevel={bundle.requirement_level} />
            {bundle.clarification_context ? (
              <details className="ba-details ba-clarification-block">
                <summary>Clarification captured (this feature)</summary>
                <pre className="ba-pre">{bundle.clarification_context}</pre>
              </details>
            ) : null}
          </div>
        ))}

        {!state.multi_feature_results?.length && state.understood ? (
          <UnderstandingView u={state.understood} />
        ) : null}

        {state.intake && Object.keys(state.intake).length ? (
          <p className="ba-intake-line">
            {state.intake.level ? <>Intake level: {state.intake.level} · </> : null}
            {state.intake.feature_label ? <>Feature label: {state.intake.feature_label}</> : null}
          </p>
        ) : null}

        {state.needs_continue_empty_clarification ? (
          <section className="card">
            <p>
              No clarification questions for this feature. If refinement did not run automatically,
              continue below.
            </p>
            <div className="btn-group">
              <button
                type="button"
                className="btn btn-primary"
                disabled={busy}
                onClick={() => void runContinueEmpty()}
              >
                Continue — refine &amp; generate artifacts
              </button>
            </div>
          </section>
        ) : null}

        {state.clarification ? (
          <ClarificationForm
            stage={state.clarification.stage}
            questions={state.clarification.questions}
            stage2Notes={state.clarification.stage2_notes}
            otherLabel={otherLabel}
            busy={busy}
            onSubmit={(a) => void runClarification(a)}
          />
        ) : null}

        {!state.multi_feature_results?.length && state.refined_text ? (
          <section className="card">
            <h3>Refinement</h3>
            <pre className="ba-pre">{state.refined_text}</pre>
          </section>
        ) : null}

        {!state.multi_feature_results?.length && state.artifacts ? (
          <>
            <h2 className="ba-section-title">Generated artifacts</h2>
            <ArtifactsPanel
              artifacts={state.artifacts}
              intakeLevel={
                state.refined && typeof state.refined.requirement_level === 'string'
                  ? state.refined.requirement_level
                  : null
              }
            />
          </>
        ) : null}

        {state.clarification_context && !state.clarification && showResults ? (
          <details className="card ba-details">
            <summary>Clarification summary</summary>
            <pre className="ba-pre">{state.clarification_context}</pre>
          </details>
        ) : null}
      </div>
    </AppShell>
  )
}
