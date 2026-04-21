import type { ApiMeta, PipelineState } from './types'

const JSON_HEADERS = { 'Content-Type': 'application/json' }

/** True when the FastAPI server responds to `/api/health`. */
export async function fetchHealth(): Promise<boolean> {
  try {
    const r = await fetch('/api/health', { method: 'GET' })
    return r.ok
  } catch {
    return false
  }
}

export async function fetchMeta(): Promise<ApiMeta> {
  const r = await fetch('/api/meta')
  if (!r.ok) throw new Error(`Meta failed: ${r.status}`)
  return r.json()
}

export async function createSession(): Promise<{ session_id: string; state: PipelineState }> {
  const r = await fetch('/api/sessions', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({}),
  })
  if (!r.ok) throw new Error(`Create session failed: ${r.status}`)
  return r.json()
}

export async function getSession(sessionId: string): Promise<PipelineState> {
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`)
  if (!r.ok) throw new Error(`Get session failed: ${r.status}`)
  return r.json()
}

export async function resetSession(sessionId: string): Promise<PipelineState> {
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/reset`, {
    method: 'POST',
  })
  if (!r.ok) throw new Error(`Reset failed: ${r.status}`)
  return r.json()
}

export async function generateRequirement(
  sessionId: string,
  params: {
    requirementText: string
    artifactScopeLabel: string
    acceptanceCriteriaFormat: string
    file?: File | null
  },
): Promise<PipelineState> {
  const fd = new FormData()
  fd.set('requirement_text', params.requirementText)
  fd.set('artifact_scope_label', params.artifactScopeLabel)
  fd.set('acceptance_criteria_format', params.acceptanceCriteriaFormat)
  if (params.file) fd.set('file', params.file)
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/generate`, {
    method: 'POST',
    body: fd,
  })
  if (!r.ok) throw new Error(`Generate failed: ${r.status}`)
  return r.json()
}

export async function submitClarification(
  sessionId: string,
  answers: Record<string, string>,
): Promise<PipelineState> {
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/clarification`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ answers }),
  })
  if (!r.ok) throw new Error(`Clarification failed: ${r.status}`)
  return r.json()
}

export async function continueEmptyClarification(sessionId: string): Promise<PipelineState> {
  const r = await fetch(
    `/api/sessions/${encodeURIComponent(sessionId)}/continue-empty-clarification`,
    { method: 'POST' },
  )
  if (!r.ok) throw new Error(`Continue failed: ${r.status}`)
  return r.json()
}
