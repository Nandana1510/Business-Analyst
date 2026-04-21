import type { ArtifactsPayload } from '../types'

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null && !Array.isArray(x)
}

function stringList(val: unknown): string[] {
  if (!Array.isArray(val)) return []
  return val.map((x) => String(x).trim()).filter(Boolean)
}

type EpicBlockProps = {
  epic: unknown
  /** When true, no outer `.card` — use inside another card/section. */
  embedded?: boolean
  /** Main heading (e.g. "Epic" or "Epic (this feature)"). */
  title?: string
}

function EpicBlock({ epic, embedded, title = 'Epic' }: EpicBlockProps) {
  if (!epic) return null
  if (typeof epic === 'string' && epic.trim()) {
    const inner = <pre className="ba-pre">{epic}</pre>
    const wrapped = (
      <details className="ba-expander">
        <summary>{title}</summary>
        {embedded ? <div className="ba-epic-embedded ba-epic-under-expander">{inner}</div> : inner}
      </details>
    )
    return embedded ? wrapped : <section className="card ba-artifact-collapsible-card">{wrapped}</section>
  }
  if (!isRecord(epic)) return null
  const epicTitle = String(epic.epic_title ?? epic.title ?? '')
  if (!epicTitle && !Object.keys(epic).length) return null

  const goals = stringList(epic.goals_and_objectives)
  const caps = stringList(epic.key_capabilities)
  const outcomes = stringList(epic.business_outcomes)
  const metrics = stringList(epic.success_metrics)

  const body = (
    <>
      <dl className="kv">
        {epicTitle ? (
          <>
            <dt>Title</dt>
            <dd>{epicTitle}</dd>
          </>
        ) : null}
        {epic.epic_summary ? (
          <>
            <dt>Summary</dt>
            <dd className="ba-dd-plain">{String(epic.epic_summary)}</dd>
          </>
        ) : null}
        {epic.epic_description ? (
          <>
            <dt>Description</dt>
            <dd className="ba-dd-plain">{String(epic.epic_description)}</dd>
          </>
        ) : null}
        {epic.business_problem ? (
          <>
            <dt>Business problem</dt>
            <dd className="ba-dd-plain">{String(epic.business_problem)}</dd>
          </>
        ) : null}
        {goals.length ? (
          <>
            <dt>Goals &amp; objectives</dt>
            <dd>
              <ul>
                {goals.map((g, i) => (
                  <li key={i}>{g}</li>
                ))}
              </ul>
            </dd>
          </>
        ) : null}
        {caps.length ? (
          <>
            <dt>Key capabilities</dt>
            <dd>
              <ul>
                {caps.map((g, i) => (
                  <li key={i}>{g}</li>
                ))}
              </ul>
            </dd>
          </>
        ) : null}
        {outcomes.length ? (
          <>
            <dt>Business outcomes</dt>
            <dd>
              <ul>
                {outcomes.map((g, i) => (
                  <li key={i}>{g}</li>
                ))}
              </ul>
            </dd>
          </>
        ) : null}
        {metrics.length ? (
          <>
            <dt>Success metrics</dt>
            <dd>
              <ul>
                {metrics.map((g, i) => (
                  <li key={i}>{g}</li>
                ))}
              </ul>
            </dd>
          </>
        ) : null}
      </dl>
    </>
  )

  const wrapped = (
    <details className="ba-expander">
      <summary>{title}</summary>
      {embedded ? <div className="ba-epic-embedded ba-epic-under-expander">{body}</div> : body}
    </details>
  )
  return embedded ? wrapped : <section className="card ba-artifact-collapsible-card">{wrapped}</section>
}

function BugReport({ br }: { br: Record<string, unknown> }) {
  const desc = String(br.bug_description ?? '').trim()
  const steps = Array.isArray(br.steps_to_reproduce) ? br.steps_to_reproduce : []
  if (!desc && !steps.length) return null
  return (
    <section className="card ba-artifact-collapsible-card">
      <details className="ba-expander">
        <summary>Bug report</summary>
        <div className="ba-collapsible-body">
          {desc ? (
            <p className="hint-muted" style={{ whiteSpace: 'pre-wrap' }}>
              {desc}
            </p>
          ) : null}
          {steps.length > 0 ? (
            <ol>
              {(steps as unknown[]).map((s, i) => (
                <li key={i}>{String(s)}</li>
              ))}
            </ol>
          ) : null}
          {br.expected_behavior ? (
            <p>
              <strong>Expected:</strong> {String(br.expected_behavior)}
            </p>
          ) : null}
          {br.actual_behavior ? (
            <p>
              <strong>Actual:</strong> {String(br.actual_behavior)}
            </p>
          ) : null}
        </div>
      </details>
    </section>
  )
}

function StoryBlock({ story, i }: { story: Record<string, unknown>; i: number }) {
  const text = String(story.story ?? '')
  const ref = String(story.story_ref ?? '').trim()
  const acs = Array.isArray(story.acceptance_criteria) ? story.acceptance_criteria : []
  const refLabel = ref && ref !== 'unspecified' ? ref : `US${i + 1}`
  return (
    <article className="ba-story">
      <h4>
        User story {i + 1} <code className="ba-ref">{refLabel}</code>
      </h4>
      <p style={{ whiteSpace: 'pre-wrap' }}>{text}</p>
      {acs.length > 0 ? (
        <>
          <h5 className="ba-ac-heading">Acceptance criteria</h5>
          <ul>
            {acs.map((ac, j) => {
              const row = isRecord(ac) ? ac : { text: String(ac) }
              const t = String(row.text ?? '').trim()
              return t.includes('\n') ? (
                <li key={j}>
                  <pre className="ba-ac-pre ba-pre">{t}</pre>
                </li>
              ) : (
                <li key={j}>{t}</li>
              )
            })}
          </ul>
        </>
      ) : null}
    </article>
  )
}

function JourneyBlock({ title, steps }: { title: string; steps: string[] }) {
  if (!steps.length) return null
  return (
    <details className="ba-expander">
      <summary>
        User journey — {title}
      </summary>
      <ol className="ba-journey-ol">
        {steps.map((step, i) => (
          <li key={i}>{step}</li>
        ))}
      </ol>
    </details>
  )
}

function GapBlock({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null
  return (
    <details className="ba-expander">
      <summary>Gap analysis — {title}</summary>
      <ul className="ba-gap-ul">
        {items.map((g, i) => (
          <li key={i}>{g}</li>
        ))}
      </ul>
    </details>
  )
}

function UserStoriesBlock({ title, stories }: { title: string; stories: unknown[] }) {
  const list = stories.filter(isRecord)
  if (!list.length) return null
  return (
    <details className="ba-expander">
      <summary>
        User stories — {title} ({list.length})
      </summary>
      <div className="ba-stories-under-expander">
        {list.map((s, j) => (
          <StoryBlock key={j} story={s} i={j} />
        ))}
      </div>
    </details>
  )
}

function FeatureBlock({
  feat,
  index1,
  showEpic,
}: {
  feat: Record<string, unknown>
  index1: number
  /** Epics are only shown for product-level requirements. */
  showEpic: boolean
}) {
  const name = String(feat.feature_name ?? `Feature ${index1}`)
  const stories = Array.isArray(feat.user_stories) ? feat.user_stories : []
  const uj = stringList(feat.user_journey)
  const ga = stringList(feat.gap_analysis)
  const epic = feat.epic

  return (
    <div className="ba-feature-block">
      <h4 className="ba-feature-heading">
        Feature {index1}: {name}
      </h4>
      {String(feat.feature_summary ?? '').trim() ? (
        <p className="hint-muted ba-feature-summary">{String(feat.feature_summary)}</p>
      ) : null}
      {showEpic && (isRecord(epic) || (typeof epic === 'string' && String(epic).trim())) ? (
        <EpicBlock epic={epic} embedded title="Epic (this feature)" />
      ) : null}
      <JourneyBlock title={name} steps={uj} />
      <GapBlock title={name} items={ga} />
      {stories.length > 0 ? (
        <UserStoriesBlock title={name} stories={stories} />
      ) : (
        <p className="hint-muted">No user stories returned for this feature.</p>
      )}
    </div>
  )
}

export type ArtifactsPanelProps = {
  artifacts: ArtifactsPayload
  /** Intake level from refined requirement (e.g. `product`) — drives product-level callouts. */
  intakeLevel?: string | null
}

function featuresHaveBucketedStories(features: unknown[] | undefined): boolean {
  if (!Array.isArray(features) || !features.length) return false
  return features.some(
    (f) => isRecord(f) && Array.isArray(f.user_stories) && (f.user_stories as unknown[]).length > 0,
  )
}

export function ArtifactsPanel({ artifacts, intakeLevel }: ArtifactsPanelProps) {
  const br = artifacts.bug_report
  const lvl = (intakeLevel || '').trim().toLowerCase()
  const isProduct = lvl === 'product'
  const isFeatureLike = lvl === 'feature' || lvl === 'enhancement'
  const featureList = Array.isArray(artifacts.features) ? (artifacts.features as unknown[]) : []
  const hasPerFeatureStories = featuresHaveBucketedStories(featureList)
  /** Flat `user_stories` duplicates per-feature lists when both are present; show flat only as fallback. */
  const showFlatUserStories =
    Array.isArray(artifacts.user_stories) &&
    artifacts.user_stories.length > 0 &&
    (!featureList.length || !hasPerFeatureStories)

  return (
    <div className="ba-artifacts">
      {br && isRecord(br) ? <BugReport br={br} /> : null}

      {isProduct ? <EpicBlock epic={artifacts.epic} title="Epic" /> : null}

      {Array.isArray(artifacts.features) && artifacts.features.length > 0 ? (
        <section className="card ba-features-section">
          <h3>Features &amp; user stories</h3>
          <details className="ba-expander ba-section-guide-expander">
            <summary>How this section is organized</summary>
            <p className="hint-muted ba-features-lead">
              Each <strong>feature</strong> can include
              {isProduct ? <> its own epic, </> : null} user stories, journey, and gap analysis. Epics, journeys, gaps,
              and stories open on click. Stories include acceptance criteria when the model returns them.
            </p>
          </details>
          {featureList.map((feat, fi) =>
            isRecord(feat) ? (
              <FeatureBlock key={fi} feat={feat} index1={fi + 1} showEpic={isProduct} />
            ) : null,
          )}
        </section>
      ) : null}

      {showFlatUserStories ? (
        <section className="card ba-artifact-collapsible-card">
          <details className="ba-expander">
            <summary>
              User stories — flat list ({(artifacts.user_stories as unknown[]).length})
            </summary>
            <div className="ba-collapsible-body">
              <p className="hint-muted ba-features-lead">
                Used when the model did not return per-feature buckets. Open to read each story and acceptance criteria.
              </p>
              <div className="ba-stories-under-expander">
                {(artifacts.user_stories as unknown[]).map((s, j) =>
                  isRecord(s) ? <StoryBlock key={j} story={s} i={j} /> : null,
                )}
              </div>
            </div>
          </details>
        </section>
      ) : null}

      {artifacts.user_journey?.length ? (
        <section className="card ba-artifact-collapsible-card">
          <details className="ba-expander">
            <summary>
              {isFeatureLike
                ? `User journey — ${artifacts.user_journey.length} steps`
                : `User journey (global) — ${artifacts.user_journey.length} steps`}
            </summary>
            <div className="ba-collapsible-body">
              <p className="hint-muted">
                {isFeatureLike
                  ? 'Ordered steps for this requirement (feature / enhancement intake).'
                  : 'End-to-end journey across the full scope (e.g. product or sprint container).'}
              </p>
              <ol className="ba-journey-ol">
                {artifacts.user_journey.map((step, i) => (
                  <li key={i}>{step}</li>
                ))}
              </ol>
            </div>
          </details>
        </section>
      ) : null}

      {artifacts.gap_analysis?.length ? (
        <section className="card ba-artifact-collapsible-card">
          <details className="ba-expander">
            <summary>
              {isFeatureLike
                ? `Gap analysis — ${artifacts.gap_analysis.length} items`
                : `Gap analysis (global) — ${artifacts.gap_analysis.length} items`}
            </summary>
            <div className="ba-collapsible-body">
              <p className="hint-muted">
                {isFeatureLike
                  ? 'Open questions, edge cases, and risks for this requirement.'
                  : 'Cross-cutting gaps and risks across the full scope.'}
              </p>
              <ul className="ba-gap-ul">
                {artifacts.gap_analysis.map((g, i) => (
                  <li key={i}>{g}</li>
                ))}
              </ul>
            </div>
          </details>
        </section>
      ) : null}
    </div>
  )
}
