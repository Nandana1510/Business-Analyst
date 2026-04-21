export type ClarificationQuestion = {
  category: string
  question: string
  options: string[]
}

export type ClarificationState = {
  stage: number
  questions: ClarificationQuestion[]
  stage2_notes: string[] | null
}

export type MultiFeatureProgress = {
  active: boolean
  feature_index: number
  total: number
}

export type FeatureBundle = {
  index: number
  total: number
  unit_text: string
  understood: Record<string, unknown>
  refined_text: string
  refined: Record<string, unknown>
  artifacts: ArtifactsPayload
  clarification_context: string
  feature_label: string
  requirement_level: string
}

export type ArtifactsPayload = {
  epic: unknown
  features: unknown[]
  user_stories: unknown[]
  user_journey: string[]
  gap_analysis: string[]
  bug_report: Record<string, unknown> | null
}

export type PipelineState = {
  session_id: string
  error: string | null
  llm: string
  intake_open_items: string[]
  original_requirement_text: string
  multi_feature: MultiFeatureProgress | null
  multi_feature_results: FeatureBundle[] | null
  understood: Record<string, unknown> | null
  clarification: ClarificationState | null
  needs_continue_empty_clarification: boolean
  refined_text: string | null
  refined: Record<string, unknown> | null
  artifacts: ArtifactsPayload | null
  clarification_context: string | null
  intake: { level?: string; feature_label?: string }
}

export type ApiMeta = {
  artifact_scope_labels: string[]
  artifact_mode_by_label: Record<string, string>
  acceptance_criteria_formats: { label: string; value: string }[]
  clarification_other_option_label: string
}
