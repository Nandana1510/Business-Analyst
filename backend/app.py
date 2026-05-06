"""
Streamlit frontend for the AI Business Analyst pipeline.
Reuses existing stages: entry, understanding, clarification, refinement, artifacts.

Single- and multi-feature runs use the same per-feature flow: understanding, then two-stage clarification,
then refinement and artifacts. Multi-unit inputs process units sequentially with isolated context per unit.

Traceability (rule sources, BR/US links) is kept on ``RefinedRequirement`` and artifact models
and in ``.to_dict()`` / ``traceability_metadata()`` — it is not rendered in the UI. For
debugging, inspect ``st.session_state.refined``, ``st.session_state.artifacts``, or log those
structures from a temporary hook in code.
"""

from dataclasses import dataclass

import streamlit as st

from stages.requirement_entry import (
    DocumentExtractionError,
    RawRequirement,
    raw_requirement_from_file,
    raw_requirement_from_manual_text,
)
from stages.requirement_intake import NormalizedRequirementUnit, analyze_intake
from stages.requirement_understanding import (
    UnderstoodRequirement,
    understand_requirement,
    get_llm_provider_and_model,
)
from stages.requirement_clarification import (
    CLARIFICATION_MAX_TOTAL_QUESTIONS,
    CLARIFICATION_STAGE1_TARGET_MAX,
    CLARIFICATION_STAGE1_TARGET_MIN,
    ClarifiedRequirement,
    OTHER_OPTION_LABEL,
    clarification_needs_stage2,
    detect_missing_fields,
    effective_requirement_level_for_artifacts,
    generate_stage1_questions,
    generate_stage2_questions,
    merge_stage1_questions_with_optional_granularity,
    normalize_responses_with_questions,
    stage1_effective_budget_count,
)
from stages.clarification_consistency import validate_clarification_answers
from stages.requirement_refinement import RefinedRequirement, refine_requirement
from stages.artifact_generation import (
    AC_FORMAT_BDD,
    AC_FORMAT_DECLARATIVE,
    ARTIFACT_GENERATION_CHOICES,
    ARTIFACT_MODE_BY_LABEL,
    AdvancedArtifacts,
    generate_artifacts_for_mode,
)
from stages.run_output_storage import persist_multi_feature_run, persist_single_feature_run
# First selectbox entry: Streamlit always pre-selects index 0, so this must not be a real answer.
CLARIFICATION_SELECT_PLACEHOLDER = "— Select an option —"


def _session_clarification_requirement_context() -> str:
    """Active intake unit / sub-requirement text only — not the full original upload."""
    return (st.session_state.get("raw_text") or "").strip()


def _session_collapse_from_active_unit() -> bool:
    au = st.session_state.get("active_intake_unit")
    if isinstance(au, NormalizedRequirementUnit):
        return bool(getattr(au, "collapse_product_feature_decomposition", False))
    return False


def _show_clarification_validation(clarified: ClarifiedRequirement) -> bool:
    """
    Display validation issues. Returns True if refinement may proceed (no errors).
    """
    vr = validate_clarification_answers(clarified)
    for sev, msg in vr.issues:
        if sev == "error":
            st.error(msg)
        elif sev == "warning":
            st.warning(msg)
    return not vr.has_errors


def _render_requirement_understanding(understood) -> None:
    """Present structured understanding without raw JSON (label: value + impacted systems list)."""
    d = understood.to_dict()
    t = d.get("type") or "—"
    actor = d.get("actor") or "—"
    sec = d.get("secondary_actor") or ""
    action = d.get("action") or "—"
    domain = d.get("domain") or "—"
    actor_line = f"**Actor:** {actor}"
    if sec:
        actor_line += f"\n\n**Secondary actor:** {sec}"
    st.markdown(
        f"**Requirement Type:** {t}\n\n"
        f"{actor_line}\n\n"
        f"**Action:** {action}\n\n"
        f"**Domain:** {domain}"
    )

    st.markdown("**Impacted Systems:**")
    impact = d.get("impact") or []
    if isinstance(impact, str):
        impact = [s.strip() for s in impact.split(",") if s.strip()] if impact.strip() else []
    if isinstance(impact, list) and any(str(x).strip() for x in impact):
        st.markdown("\n".join(f"- {item}" for item in impact if str(item).strip()))
    else:
        st.caption("No impacted systems identified.")


@dataclass
class FeaturePipelineBundle:
    """One feature unit after running the full pipeline (including clarification)."""

    index: int
    total: int
    unit_text: str
    understood: UnderstoodRequirement
    refined: RefinedRequirement
    artifacts: AdvancedArtifacts
    clarified: ClarifiedRequirement | None = None
    feature_label: str = ""
    requirement_level: str = ""


def _multi_feature_active() -> bool:
    return bool(st.session_state.get("multi_feature_units"))


def _clarification_key_suffix() -> str:
    """Disambiguate Streamlit widget keys across multi-feature rounds."""
    if _multi_feature_active():
        return str(st.session_state.multi_feature_index)
    return "0"


def _persist_multi_feature_run_if_ready() -> None:
    """Append-only JSON export for a completed multi-feature run (best-effort)."""
    bundles = st.session_state.get("multi_feature_results")
    if not bundles:
        return
    persist_multi_feature_run(
        original_requirement=st.session_state.get("original_requirement_text") or "",
        open_items=list(st.session_state.get("intake_open_items") or []),
        artifact_scope_mode=st.session_state.get("pipeline_artifact_mode") or "",
        acceptance_criteria_format=st.session_state.get("pipeline_acceptance_criteria_format") or "",
        bundles=list(bundles),
    )


def _persist_single_feature_completed() -> None:
    """Persist a finished single-feature run (not used while a multi-feature batch is active)."""
    if _multi_feature_active():
        return
    refined = st.session_state.get("refined")
    artifacts = st.session_state.get("artifacts")
    understood = st.session_state.get("understood")
    if refined is None or artifacts is None or understood is None:
        return
    persist_single_feature_run(
        original_requirement=st.session_state.get("original_requirement_text")
        or (st.session_state.raw_text or ""),
        open_items=list(st.session_state.get("intake_open_items") or []),
        artifact_scope_mode=st.session_state.get("pipeline_artifact_mode") or "",
        acceptance_criteria_format=st.session_state.get("pipeline_acceptance_criteria_format") or "",
        understood=understood,
        refined=refined,
        artifacts=artifacts,
        clarified=st.session_state.get("clarified"),
    )


def _multi_feature_finish_cleanup() -> None:
    _persist_multi_feature_run_if_ready()
    st.session_state.multi_feature_units = None
    st.session_state.multi_feature_index = 0
    st.session_state.understood = None
    st.session_state.clarification_questions = []
    st.session_state.clarification_stage1_questions = []
    st.session_state.clarification_stage1_budget_exclusions = 0
    st.session_state.clarification_cl_stage = 1
    st.session_state.clarification_pending_answers_s1 = None
    st.session_state.clarification_stage2_notes = None
    st.session_state.clarified = None
    st.session_state.refined = None
    st.session_state.artifacts = None


def _append_completed_bundle(
    refined: RefinedRequirement,
    artifacts: AdvancedArtifacts,
    clarified: ClarifiedRequirement,
) -> None:
    units = st.session_state.multi_feature_units
    idx = st.session_state.multi_feature_index
    au = st.session_state.get("active_intake_unit")
    fl = (au.feature_name if isinstance(au, NormalizedRequirementUnit) else "") or ""
    rl = (getattr(refined, "requirement_level", None) or "").strip() or (
        (au.requirement_level if isinstance(au, NormalizedRequirementUnit) else "") or ""
    )
    st.session_state.multi_feature_results.append(
        FeaturePipelineBundle(
            index=idx + 1,
            total=len(units),
            unit_text=st.session_state.raw_text or "",
            understood=st.session_state.understood,
            refined=refined,
            artifacts=artifacts,
            clarified=clarified,
            feature_label=fl,
            requirement_level=rl,
        )
    )


def _run_understanding_for_unit(unit: NormalizedRequirementUnit) -> None:
    supp = (getattr(unit, "supplementary_constraints", None) or "").strip()
    requirement = RawRequirement(
        text=unit.text,
        intake_feature_label=unit.feature_name or None,
        requirement_level=unit.requirement_level or None,
        supplementary_constraints=supp,
    )
    understood = understand_requirement(requirement)
    st.session_state.understood = understood
    st.session_state.raw_text = unit.text
    st.session_state.active_intake_unit = unit
    oi = st.session_state.get("intake_open_items") or []
    try:
        core = generate_stage1_questions(understood, unit.text, {}, open_items=oi)
    except Exception as e:
        st.session_state.error = (
            f"Clarification Stage 1 failed (you can retry after fixing LLM access): {e}"
        )
        core = []
    qs, budget_excl = merge_stage1_questions_with_optional_granularity(
        core,
        understood,
        unit.text,
        unit.requirement_level,
        len(st.session_state.multi_feature_units or []) or 1,
        product_intent_source_text=st.session_state.get("original_requirement_text")
        or st.session_state.get("raw_text")
        or "",
        intake_supplementary_constraints=supp or None,
    )
    st.session_state.clarification_questions = qs
    st.session_state.clarification_stage1_questions = list(qs)
    st.session_state.clarification_stage1_budget_exclusions = budget_excl
    st.session_state.clarification_cl_stage = 1
    st.session_state.clarification_pending_answers_s1 = None
    st.session_state.clarification_stage2_notes = None
    st.session_state.clarified = None
    st.session_state.refined = None
    st.session_state.artifacts = None


def _multi_feature_auto_advance_through_empty_clarification(
    artifact_mode: str,
    acceptance_criteria_format: str,
) -> None:
    """
    For units with no clarification questions: refine, generate artifacts, append bundle,
    move to next unit or finish. May process several units in one run.
    """
    units = st.session_state.multi_feature_units
    while True:
        if not st.session_state.clarification_questions:
            clarified = ClarifiedRequirement.from_answers(st.session_state.understood, {})
            au = st.session_state.get("active_intake_unit")
            intake_lbl = au.feature_name if isinstance(au, NormalizedRequirementUnit) else None
            lvl = au.requirement_level if isinstance(au, NormalizedRequirementUnit) else None
            oi = st.session_state.get("intake_open_items") or []
            refined = refine_requirement(
                st.session_state.understood,
                clarification_context=clarified.to_refinement_block(),
                intake_feature_label=intake_lbl,
                requirement_level=lvl,
                open_items=oi,
                full_requirement_narrative=_session_clarification_requirement_context() or None,
                collapse_product_feature_decomposition=_session_collapse_from_active_unit(),
            )
            artifacts = generate_artifacts_for_mode(
                refined,
                artifact_mode,
                acceptance_criteria_format=acceptance_criteria_format,
            )
            st.session_state.clarified = clarified
            _append_completed_bundle(refined, artifacts, clarified)
            if st.session_state.multi_feature_index + 1 < len(units):
                st.session_state.multi_feature_index += 1
                _run_understanding_for_unit(units[st.session_state.multi_feature_index])
                continue
            _multi_feature_finish_cleanup()
            return
        return


ACCEPTANCE_CRITERIA_FORMAT_LABELS: list[tuple[str, str]] = [
    ("Declarative", AC_FORMAT_DECLARATIVE),
    ("BDD (Given-When-Then)", AC_FORMAT_BDD),
]


def _render_generated_artifacts(
    arts: AdvancedArtifacts,
    intake_level: str | None = None,
) -> None:
    # Traceability (story_ref, traces_to, rule source/source_id) stays on objects and in
    # .to_dict() / refine_requirement().traceability_metadata() — not shown in the UI.
    _il = (intake_level or "").strip().lower()
    if _il == "product":
        st.info(
            "**Product-level output:** **One program epic** (above when generated); **features** below group "
            "related **user stories**. **One** consolidated **user journey** and **one** **gap analysis** apply "
            "across the backlog areas—not a separate epic or journey under each feature row."
        )
    if arts.bug_report:
        br = arts.bug_report
        st.markdown("### Bug report")
        if (br.get("bug_description") or "").strip():
            st.markdown("**Description**")
            st.markdown(str(br["bug_description"]).strip())
        steps = br.get("steps_to_reproduce") or []
        if isinstance(steps, list) and steps:
            st.markdown("**Steps to reproduce**")
            for si, step in enumerate(steps, 1):
                st.markdown(f"{si}. {step}")
        if (br.get("expected_behavior") or "").strip():
            st.markdown("**Expected behavior**")
            st.markdown(str(br["expected_behavior"]).strip())
        if (br.get("actual_behavior") or "").strip():
            st.markdown("**Actual behavior**")
            st.markdown(str(br["actual_behavior"]).strip())
        st.divider()
    ed = arts.epic_document
    if ed:
        st.markdown("### Epic")
        st.markdown(f"**Epic title:** {ed.title}")
        if (ed.epic_summary or "").strip():
            st.markdown("**Epic summary**")
            st.markdown(ed.epic_summary.strip())
        if (ed.epic_description or "").strip():
            st.markdown("**Epic description**")
            st.markdown(ed.epic_description.strip())
        if (ed.business_problem or "").strip():
            st.markdown("**Business problem**")
            st.markdown(ed.business_problem.strip())
        if ed.goals_and_objectives:
            st.markdown("**Goals & objectives**")
            for g in ed.goals_and_objectives:
                st.markdown(f"- {g}")
        if ed.key_capabilities:
            st.markdown("**Key capabilities / scope**")
            for k in ed.key_capabilities:
                st.markdown(f"- {k}")
        if ed.business_outcomes:
            st.markdown("**Business outcomes**")
            for o in ed.business_outcomes:
                st.markdown(f"- {o}")
        if ed.success_metrics:
            st.markdown("**Success metrics**")
            for m in ed.success_metrics:
                st.markdown(f"- {m}")
    elif (arts.epic or "").strip():
        st.markdown("### Epic")
        st.markdown(arts.epic)
    else:
        il = _il
        if il == "product" and arts.features:
            st.caption(
                "No structured epic document was returned — feature buckets below may still list stories. "
                "Retry artifact generation if this persists."
            )
        elif il == "product":
            st.warning(
                "No feature buckets were returned (generation may have failed). Check user stories below if any."
            )
        elif il in ("feature", "enhancement"):
            st.caption(
                "No epic for this intake level: **feature** and **enhancement** produce user stories (+ AC) only."
            )
        elif il == "bug":
            st.caption("No epic — **bug** level uses a structured bug report and optional fix-oriented story.")
        elif il == "sprint":
            st.caption(
                "No epic — **sprint** outputs **features** and user stories per feature only (no global journey or gap)."
            )
    if arts.features:
        st.markdown("### Features & user stories")
        st.caption(
            "Each **feature** row groups **user stories** (acceptance criteria per story). "
            + (
                "Per-feature mini-epics are not used at product level — see the program epic above."
                if _il == "product"
                else "Optional per-feature epic/journey/gap may appear for sprint-level runs."
            )
        )
        for fi, feat in enumerate(arts.features, 1):
            st.markdown(f"#### Feature {fi}: {feat.feature_name}")
            if (feat.feature_summary or "").strip():
                st.markdown(feat.feature_summary.strip())
            fed = getattr(feat, "epic_document", None)
            if fed:
                st.markdown("**Epic (this feature)**")
                st.markdown(f"**Title:** {fed.title}")
                if (fed.epic_summary or "").strip():
                    st.markdown(fed.epic_summary.strip())
                if (fed.epic_description or "").strip():
                    st.markdown("**Description**")
                    st.markdown(fed.epic_description.strip())
                if (fed.business_problem or "").strip():
                    st.markdown("**Business problem**")
                    st.markdown(fed.business_problem.strip())
                if fed.goals_and_objectives:
                    st.markdown("**Goals & objectives**")
                    for g in fed.goals_and_objectives:
                        st.markdown(f"- {g}")
                if fed.key_capabilities:
                    st.markdown("**Key capabilities**")
                    for k in fed.key_capabilities:
                        st.markdown(f"- {k}")
            if getattr(feat, "user_journey", None):
                with st.expander(f"User journey — {feat.feature_name}"):
                    for si, step in enumerate(feat.user_journey, 1):
                        st.markdown(f"{si}. {step}")
            if getattr(feat, "gap_analysis", None):
                with st.expander(f"Gap analysis — {feat.feature_name}"):
                    for gi, gap in enumerate(feat.gap_analysis, 1):
                        st.markdown(f"{gi}. {gap}")
            if not feat.user_stories:
                st.caption("No user stories returned for this feature.")
                st.divider()
                continue
            for j, row in enumerate(feat.user_stories, 1):
                ref = (row.story_ref or "").strip() or f"US{j}"
                st.markdown(f"**User story {j}** (`{ref}`)")
                st.markdown(row.story)
                if row.acceptance_criteria:
                    st.markdown("**Acceptance criteria**")
                    for c in row.acceptance_criteria:
                        t = (c.text or "").strip()
                        if "\n" in t:
                            st.code(t, language=None)
                        else:
                            st.markdown(f"- {t}")
                else:
                    st.caption("No acceptance criteria returned for this story.")
                st.divider()
    elif arts.user_stories:
        st.markdown("### User stories")
        if _il == "bug":
            st.caption("Optional fix-oriented story; bug report above is primary.")
        else:
            st.caption("Each story includes acceptance criteria scoped to that story only.")
        for i, row in enumerate(arts.user_stories, 1):
            st.markdown(f"#### User Story {i}")
            st.markdown(row.story)
            if row.acceptance_criteria:
                st.markdown("**Acceptance criteria**")
                for c in row.acceptance_criteria:
                    t = (c.text or "").strip()
                    if "\n" in t:
                        st.code(t, language=None)
                    else:
                        st.markdown(f"- {t}")
            else:
                st.caption("No acceptance criteria returned for this story.")
            st.divider()
    if arts.user_journey:
        _uj = "User journey (global)" if _il == "product" else "User journey"
        with st.expander(_uj):
            for i, step in enumerate(arts.user_journey, 1):
                st.markdown(f"{i}. {step}")
    if arts.gap_analysis:
        _ga = (
            "Gap analysis — global (edge cases & risks)"
            if _il == "product"
            else "Gap analysis (edge cases)"
        )
        with st.expander(_ga):
            for i, gap in enumerate(arts.gap_analysis, 1):
                st.markdown(f"{i}. {gap}")


st.set_page_config(page_title="AI Business Analyst", page_icon="📋", layout="wide")
st.title("📋 AI Business Analyst")
st.caption("Enter a requirement, click Generate, answer clarification questions if shown, and view outputs.")

# Session state for pipeline outputs
if "understood" not in st.session_state:
    st.session_state.understood = None
if "clarification_questions" not in st.session_state:
    st.session_state.clarification_questions = []
if "clarified" not in st.session_state:
    st.session_state.clarified = None
if "refined" not in st.session_state:
    st.session_state.refined = None
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None
if "error" not in st.session_state:
    st.session_state.error = None
if "multi_feature_results" not in st.session_state:
    st.session_state.multi_feature_results = None
if "multi_feature_units" not in st.session_state:
    st.session_state.multi_feature_units = None
if "multi_feature_index" not in st.session_state:
    st.session_state.multi_feature_index = 0
if "clarification_cl_stage" not in st.session_state:
    st.session_state.clarification_cl_stage = 1
if "clarification_stage1_questions" not in st.session_state:
    st.session_state.clarification_stage1_questions = []
if "clarification_stage1_budget_exclusions" not in st.session_state:
    st.session_state.clarification_stage1_budget_exclusions = 0
if "clarification_pending_answers_s1" not in st.session_state:
    st.session_state.clarification_pending_answers_s1 = None
if "clarification_stage2_notes" not in st.session_state:
    st.session_state.clarification_stage2_notes = None
if "intake_open_items" not in st.session_state:
    st.session_state.intake_open_items = []

# Input — file takes priority over the text box when both are present
uploaded_requirement_file = st.file_uploader(
    "Upload requirement document (optional)",
    type=["pdf", "docx", "txt", "md", "text"],
    help="PDF, Word (.docx), or plain text (.txt, .md). Legacy .doc is not supported. "
    "If you upload a file, its extracted text is used instead of the box below.",
)
requirement_text = st.text_area(
    "Requirement (natural language)",
    placeholder="e.g. User should be able to pause meal subscription. Or upload a document above.",
    height=100,
)
_artifact_labels = [label for _, label in ARTIFACT_GENERATION_CHOICES]
artifact_scope_label = st.selectbox(
    "Artifact generation scope",
    options=_artifact_labels,
    index=0,
    help="**Product:** per-feature epic, stories, user journey, and gap analysis (no single whole-product epic). "
    "**Sprint:** features + user stories per feature only (no epic, journey, or gap). "
    "**Feature / enhancement:** user stories with acceptance criteria per story. **Bug:** bug report + optional fix story.",
)
artifact_mode = ARTIFACT_MODE_BY_LABEL[artifact_scope_label]
_ac_fmt_display = [lbl for lbl, _ in ACCEPTANCE_CRITERIA_FORMAT_LABELS]
_ac_fmt_default_idx = 0
acceptance_criteria_format_label = st.selectbox(
    "Acceptance criteria format",
    options=_ac_fmt_display,
    index=_ac_fmt_default_idx,
    help="Declarative: short checklist-style lines. BDD: each criterion uses Given / When / Then (user stories and business rules are unchanged).",
)
acceptance_criteria_format = dict(ACCEPTANCE_CRITERIA_FORMAT_LABELS)[acceptance_criteria_format_label]


def _run_refinement_after_clarification(clarified: ClarifiedRequirement) -> None:
    """Persist clarification, run refinement + artifacts, advance multi-feature if active.

    Only sets ``clarified`` in session **after** refinement + artifacts succeed. Otherwise (e.g. LLM
    quota error) the clarification form stays available for retry and the UI does not look 'stuck'
    after understanding only.
    """
    st.session_state.clarification_cl_stage = 1
    st.session_state.clarification_pending_answers_s1 = None
    st.session_state.clarification_stage2_notes = None
    with st.spinner("Refining requirement and generating artifacts..."):
        au2 = st.session_state.get("active_intake_unit")
        ilab = au2.feature_name if isinstance(au2, NormalizedRequirementUnit) else None
        ilvl = au2.requirement_level if isinstance(au2, NormalizedRequirementUnit) else None
        oi = st.session_state.get("intake_open_items") or []
        eff_level = effective_requirement_level_for_artifacts(clarified, ilvl)
        refined = refine_requirement(
            st.session_state.understood,
            clarification_context=clarified.to_refinement_block(),
            intake_feature_label=ilab,
            requirement_level=eff_level,
            open_items=oi,
            full_requirement_narrative=_session_clarification_requirement_context() or None,
            collapse_product_feature_decomposition=_session_collapse_from_active_unit(),
        )
        artifacts = generate_artifacts_for_mode(
            refined,
            artifact_mode,
            acceptance_criteria_format=acceptance_criteria_format,
        )
    st.session_state.clarified = clarified
    st.session_state.clarification_questions = []
    st.session_state.clarification_stage1_questions = []
    st.session_state.clarification_stage1_budget_exclusions = 0
    if _multi_feature_active():
        _append_completed_bundle(refined, artifacts, clarified)
        units = st.session_state.multi_feature_units
        if st.session_state.multi_feature_index + 1 < len(units):
            st.session_state.multi_feature_index += 1
            _run_understanding_for_unit(units[st.session_state.multi_feature_index])
            _multi_feature_auto_advance_through_empty_clarification(
                artifact_mode,
                acceptance_criteria_format,
            )
        else:
            _multi_feature_finish_cleanup()
    else:
        st.session_state.refined = refined
        st.session_state.artifacts = artifacts
        _persist_single_feature_completed()


run_clicked = st.button("Generate")

if run_clicked:
    requirement: RawRequirement | None = None
    try:
        if uploaded_requirement_file is not None:
            requirement = raw_requirement_from_file(
                uploaded_requirement_file.name,
                uploaded_requirement_file.getvalue(),
            )
        else:
            if not (requirement_text and requirement_text.strip()):
                st.warning("Please enter a requirement in the text box or upload a document.")
                requirement = None
            else:
                requirement = raw_requirement_from_manual_text(requirement_text)
    except DocumentExtractionError as e:
        st.error(f"Could not read the file: {e}")
        requirement = None
    except ValueError as e:
        st.warning(str(e))
        requirement = None
    except Exception as e:
        st.error(f"Could not process input: {e}")
        requirement = None

    if requirement is not None:
        raw = requirement.text
        st.session_state.original_requirement_text = raw
        st.session_state.pipeline_artifact_mode = artifact_mode
        st.session_state.pipeline_acceptance_criteria_format = acceptance_criteria_format
        st.session_state.intake_open_items = list(requirement.open_items or [])
        if requirement.open_items:
            st.info(
                "**Preprocessing:** "
                + str(len(requirement.open_items))
                + " line(s) classified as **pending discussion / clarification** (excluded from feature extraction); "
                "they are routed to clarification and gap analysis."
            )
        if uploaded_requirement_file is not None and requirement.source_filename:
            st.info(f"Using extracted text from **{requirement.source_filename}** (preprocessed).")
        st.session_state.error = None
        st.session_state.multi_feature_units = None
        st.session_state.multi_feature_index = 0
        st.session_state.multi_feature_results = None
        try:
            units: list[NormalizedRequirementUnit] = analyze_intake(raw)
        except Exception as e:
            st.session_state.error = f"Pre-processing failed: {e}"
            units = [NormalizedRequirementUnit(text=raw, requirement_level="feature")]

        if not units:
            st.session_state.error = "No processable requirement text."
        elif len(units) > 1:
            try:
                with st.spinner(
                    f"Detected {len(units)} independent features — starting feature 1 "
                    "(understanding → clarification → refinement → artifacts for each)…"
                ):
                    st.session_state.multi_feature_units = units
                    st.session_state.multi_feature_index = 0
                    st.session_state.multi_feature_results = []
                    _run_understanding_for_unit(units[0])
                    _multi_feature_auto_advance_through_empty_clarification(
                        artifact_mode,
                        acceptance_criteria_format,
                    )
            except Exception as e:
                st.session_state.error = str(e)
                st.session_state.multi_feature_units = None
                st.session_state.multi_feature_index = 0
                st.session_state.multi_feature_results = None
                st.session_state.understood = None
                st.session_state.clarification_questions = []
                st.session_state.clarification_stage1_questions = []
                st.session_state.clarification_stage1_budget_exclusions = 0
                st.session_state.clarification_cl_stage = 1
                st.session_state.clarification_pending_answers_s1 = None
                st.session_state.clarification_stage2_notes = None
                st.session_state.clarified = None
                st.session_state.refined = None
                st.session_state.artifacts = None
        else:
            with st.spinner("Generating requirement understanding..."):
                try:
                    u0 = units[0]
                    st.session_state.active_intake_unit = u0
                    st.session_state.intake_level_display = u0.requirement_level
                    oi = st.session_state.get("intake_open_items") or []
                    understood = understand_requirement(
                        RawRequirement(
                            text=u0.text,
                            intake_feature_label=u0.feature_name or None,
                            requirement_level=u0.requirement_level or None,
                            supplementary_constraints=(getattr(u0, "supplementary_constraints", None) or ""),
                            open_items=oi,
                        )
                    )
                    st.session_state.understood = understood
                    st.session_state.raw_text = u0.text
                    supp0 = (getattr(u0, "supplementary_constraints", None) or "").strip()
                    core = generate_stage1_questions(
                        understood,
                        st.session_state.raw_text,
                        {},
                        open_items=oi,
                    )
                    questions, budget_excl = merge_stage1_questions_with_optional_granularity(
                        core,
                        understood,
                        st.session_state.raw_text,
                        u0.requirement_level,
                        len(units),
                        product_intent_source_text=st.session_state.get("original_requirement_text")
                        or st.session_state.raw_text
                        or "",
                        intake_supplementary_constraints=supp0 or None,
                    )
                    st.session_state.clarification_questions = questions
                    st.session_state.clarification_stage1_questions = list(questions)
                    st.session_state.clarification_stage1_budget_exclusions = budget_excl
                    st.session_state.clarification_cl_stage = 1
                    st.session_state.clarification_pending_answers_s1 = None
                    st.session_state.clarification_stage2_notes = None
                    if not questions:
                        clarified = ClarifiedRequirement.from_answers(understood, {})
                        refined = refine_requirement(
                            understood,
                            clarification_context=clarified.to_refinement_block(),
                            intake_feature_label=u0.feature_name or None,
                            requirement_level=u0.requirement_level,
                            open_items=oi,
                            full_requirement_narrative=_session_clarification_requirement_context()
                            or None,
                            collapse_product_feature_decomposition=bool(
                                getattr(u0, "collapse_product_feature_decomposition", False)
                            ),
                        )
                        artifacts = generate_artifacts_for_mode(
                            refined,
                            artifact_mode,
                            acceptance_criteria_format=acceptance_criteria_format,
                        )
                        st.session_state.clarified = clarified
                        st.session_state.refined = refined
                        st.session_state.artifacts = artifacts
                        _persist_single_feature_completed()
                except Exception as e:
                    st.session_state.error = str(e)
                    st.session_state.understood = None
                    st.session_state.clarification_questions = []
                    st.session_state.clarification_stage1_questions = []
                    st.session_state.clarification_stage1_budget_exclusions = 0
                    st.session_state.clarification_cl_stage = 1
                    st.session_state.clarification_pending_answers_s1 = None
                    st.session_state.clarification_stage2_notes = None
                    st.session_state.clarified = None
                    st.session_state.refined = None
                    st.session_state.artifacts = None

# Show error
if st.session_state.error:
    st.error(st.session_state.error)

# Multi-feature in progress: each unit uses its own raw text for clarification context
if _multi_feature_active():
    u = st.session_state.multi_feature_units
    idx = st.session_state.multi_feature_index
    st.divider()
    cur = st.session_state.get("active_intake_unit")
    lbl = ""
    if isinstance(cur, NormalizedRequirementUnit) and cur.feature_name:
        lbl = f" **{cur.feature_name}** —"
    st.info(
        f"**Multi-feature pipeline:** you are on **feature {idx + 1} of {len(u)}**.{lbl} "
        "Each feature runs **understanding → clarification → refinement → artifacts** with **separate** clarification context."
    )
    if st.session_state.multi_feature_results:
        with st.expander(
            f"Completed features ({len(st.session_state.multi_feature_results)}) — summary",
            expanded=False,
        ):
            for b in st.session_state.multi_feature_results:
                cap = f"**Feature {b.index} of {b.total}**"
                if b.feature_label:
                    cap += f" — _{b.feature_label}_"
                if b.requirement_level:
                    cap += f" ({b.requirement_level})"
                st.markdown(f"{cap} (refinement + artifacts recorded).")

# Final multi-feature outputs (all units finished; interactive clarification was run per unit)
if st.session_state.multi_feature_results and not _multi_feature_active():
    st.divider()
    st.subheader("Results by feature")
    n = len(st.session_state.multi_feature_results)
    st.success(
        f"**{n} independent features** were processed. Each went through clarification before refinement and artifacts."
    )
    for b in st.session_state.multi_feature_results:
        st.divider()
        title = f"## Feature {b.index} of {b.total}"
        if b.feature_label:
            title += f": {b.feature_label}"
        st.markdown(title)
        if b.requirement_level:
            st.caption(f"Intake classification: **{b.requirement_level}**")
        with st.expander("Sub-requirement text", expanded=len(st.session_state.multi_feature_results) <= 2):
            st.text(b.unit_text)
        st.subheader("Understanding")
        _render_requirement_understanding(b.understood)
        st.caption(f"LLM: {get_llm_provider_and_model()}")
        st.subheader("Refinement")
        st.text(b.refined.format_output())
        st.subheader("Generated artifacts")
        _render_generated_artifacts(b.artifacts, intake_level=b.refined.requirement_level)
        if b.clarified and (b.clarified.to_clarification_context() or "").strip():
            with st.expander("Clarification captured (this feature)"):
                st.text(b.clarified.to_clarification_context())

# Single-feature or current multi-feature unit: requirement understanding
elif st.session_state.understood is not None:
    st.divider()
    st.subheader("1️⃣ Requirement understanding")
    ild = st.session_state.get("intake_level_display")
    au1 = st.session_state.get("active_intake_unit")
    if ild or (isinstance(au1, NormalizedRequirementUnit) and au1.feature_name):
        parts = []
        if ild:
            parts.append(f"**Intake level:** {ild}")
        if isinstance(au1, NormalizedRequirementUnit) and au1.feature_name:
            parts.append(f"**Feature label:** {au1.feature_name}")
        st.caption(" · ".join(parts))
    _render_requirement_understanding(st.session_state.understood)
    st.caption(f"LLM: {get_llm_provider_and_model()}")

# Multi-feature: empty Stage 1 questions — clarification form is skipped (falsy list);
# auto-advance may not have run (LLM error / refresh). Offer an explicit continue.
if (
    _multi_feature_active()
    and st.session_state.understood is not None
    and not st.session_state.clarification_questions
    and st.session_state.clarified is None
    and st.session_state.refined is None
):
    st.divider()
    st.subheader("2️⃣ Clarification")
    st.info(
        "**No clarification questions** were returned for this feature (the model may treat it as "
        "already clear). If refinement did not run automatically (e.g. LLM quota error or page refresh), "
        "use the button below to **refine and generate artifacts** for this feature."
    )
    idx_mf = int(st.session_state.multi_feature_index or 0)
    if st.button(
        "Continue — refine & generate artifacts (this feature)",
        key=f"mf_empty_cl_{idx_mf}",
        type="primary",
    ):
        st.session_state.error = None
        try:
            with st.spinner("Refining requirement and generating artifacts…"):
                _multi_feature_auto_advance_through_empty_clarification(
                    artifact_mode,
                    acceptance_criteria_format,
                )
        except Exception as e:
            st.session_state.error = str(e)
        st.rerun()

# Clarification: Stage 1 + optional Stage 2 until clarified is set
if (
    st.session_state.understood is not None
    and st.session_state.clarification_questions
    and st.session_state.clarified is None
):
    st.divider()
    st.subheader("2️⃣ Clarification")
    cl_stage = int(st.session_state.get("clarification_cl_stage", 1) or 1)
    if cl_stage == 1:
        st.caption(
            f"**Stage 1 —** **{CLARIFICATION_STAGE1_TARGET_MIN}–{CLARIFICATION_STAGE1_TARGET_MAX}** questions "
            f"(scope, flow, constraints, assumptions). If needed, **Stage 2** adds targeted follow-up "
            f"(**max {CLARIFICATION_MAX_TOTAL_QUESTIONS}** total). Consistency is checked before refinement."
        )
    else:
        st.caption(
            "**Stage 2 —** targeted follow-up. Stage 1 answers are kept. Submit below, then answers are validated."
        )
        st.info(
            "This step is **normal** when Stage 1 surfaced ambiguities, possible contradictions, "
            "gaps, or extra detail the model wants to lock down—not an error."
        )
        notes = st.session_state.get("clarification_stage2_notes")
        if notes:
            with st.expander("What Stage 2 is addressing (from Stage 1 review)", expanded=False):
                for line in notes:
                    st.markdown(f"- {line}")
    st.markdown(
        "Choose an option from each dropdown (nothing is pre-selected as your answer). "
        "A **custom answer** field appears only when you select **Other**. "
        "You can leave a question unset by keeping **— Select an option —**."
    )
    suf = _clarification_key_suffix()
    for cq in st.session_state.clarification_questions:
        st.markdown(f"**Q ({cq.category}):** {cq.question}")
        dropdown_choices = (
            [CLARIFICATION_SELECT_PLACEHOLDER] + list(cq.options) + [OTHER_OPTION_LABEL]
        )
        sel_key = f"clar_sel_{suf}_s{cl_stage}_{cq.category}"
        st.selectbox(
            "Choose an option",
            options=dropdown_choices,
            index=0,
            key=sel_key,
        )
        if st.session_state.get(sel_key) == OTHER_OPTION_LABEL:
            st.text_input(
                "Custom answer",
                key=f"clar_other_{suf}_s{cl_stage}_{cq.category}",
                placeholder="Type your answer here",
            )
    submitted = st.button("Submit answers & continue", key="clarification_submit")
    if submitted:
        understood = st.session_state.understood
        raw_txt = _session_clarification_requirement_context()
        cur_questions = st.session_state.clarification_questions
        answers_dict: dict[str, str] = {}
        for cq in cur_questions:
            sel = (st.session_state.get(f"clar_sel_{suf}_s{cl_stage}_{cq.category}") or "").strip()
            custom = (st.session_state.get(f"clar_other_{suf}_s{cl_stage}_{cq.category}") or "").strip()
            if sel == CLARIFICATION_SELECT_PLACEHOLDER or not sel:
                answers_dict[cq.category] = ""
            elif sel == OTHER_OPTION_LABEL:
                answers_dict[cq.category] = custom
            else:
                answers_dict[cq.category] = sel

        if cl_stage == 1:
            s1qs = st.session_state.clarification_stage1_questions or cur_questions
            with st.spinner("Normalizing answers and checking consistency…"):
                norm1 = normalize_responses_with_questions(answers_dict, s1qs)
                cr1 = ClarifiedRequirement.from_answers(understood, norm1, s1qs)
                vr1 = validate_clarification_answers(cr1)
                need2 = clarification_needs_stage2(
                    understood,
                    raw_txt,
                    cr1,
                    validation_has_errors=vr1.has_errors,
                    validation_has_warnings=vr1.has_warnings,
                )
                missing = detect_missing_fields(understood, norm1)
                cap2 = max(
                    0,
                    CLARIFICATION_MAX_TOTAL_QUESTIONS
                    - stage1_effective_budget_count(
                        s1qs, st.session_state.get("clarification_stage1_budget_exclusions") or 0
                    ),
                )
                q2: list = []
                if need2 and cap2 > 0:
                    with st.spinner("Preparing Stage 2 follow-up questions (if needed)…"):
                        oi2 = st.session_state.get("intake_open_items") or []
                        q2 = generate_stage2_questions(
                            understood,
                            raw_txt,
                            norm1,
                            list(s1qs),
                            list(vr1.issues),
                            missing,
                            max_additional=cap2,
                            open_items=oi2,
                        )
            if need2 and q2:
                note_lines: list[str] = [msg for _sev, msg in vr1.issues]
                if missing:
                    note_lines.append(
                        "Required areas still empty or generic: " + ", ".join(missing)
                    )
                if not note_lines:
                    note_lines = [
                        "A short follow-up was suggested to remove ambiguity before refinement."
                    ]
                st.session_state.clarification_stage2_notes = note_lines
                st.session_state.clarification_pending_answers_s1 = norm1
                st.session_state.clarification_questions = q2
                st.session_state.clarification_cl_stage = 2
                st.rerun()
            if need2 and not q2 and (vr1.has_errors or vr1.has_warnings or missing):
                st.error(
                    "Follow-up clarification was required (consistency checks or missing fields), but "
                    "no extra questions could be added—usually because the **question budget is full** "
                    f"(max {CLARIFICATION_MAX_TOTAL_QUESTIONS} total). "
                    "Use **Clear and start over**, or shorten answers and resubmit Stage 1."
                )
                for sev, msg in vr1.issues:
                    if sev == "error":
                        st.error(msg)
                    else:
                        st.warning(msg)
                st.stop()
            if not _show_clarification_validation(cr1):
                st.stop()
            _run_refinement_after_clarification(cr1)
            st.rerun()

        elif cl_stage == 2:
            with st.spinner("Merging Stage 2 answers and validating…"):
                norm2 = normalize_responses_with_questions(answers_dict, cur_questions)
                pending = st.session_state.clarification_pending_answers_s1 or {}
                full_answers = {**pending, **norm2}
                s1qs2 = st.session_state.clarification_stage1_questions or []
                all_qs = list(s1qs2) + list(cur_questions)
                clarified = ClarifiedRequirement.from_answers(understood, full_answers, all_qs)
            if not _show_clarification_validation(clarified):
                st.stop()
            _run_refinement_after_clarification(clarified)
            st.rerun()

# Refinement output
if st.session_state.refined is not None:
    st.divider()
    st.subheader("3️⃣ Refinement")
    st.text(st.session_state.refined.format_output())

# Single-feature artifacts only (multi-feature renders artifacts per bundle when complete)
if st.session_state.artifacts is not None and not st.session_state.multi_feature_results:
    st.divider()
    st.subheader("4️⃣ Generated artifacts")
    rl = st.session_state.refined.requirement_level if st.session_state.refined else None
    _render_generated_artifacts(st.session_state.artifacts, intake_level=rl)

# Clear state button
if (
    st.session_state.understood is not None
    or st.session_state.error
    or st.session_state.multi_feature_results
    or _multi_feature_active()
):
    st.divider()
    if st.button("Clear and start over"):
        for key in list(st.session_state.keys()):
            if key in (
                "understood",
                "clarification_questions",
                "clarification_stage1_questions",
                "clarification_stage1_budget_exclusions",
                "clarification_cl_stage",
                "clarification_pending_answers_s1",
                "clarification_stage2_notes",
                "clarified",
                "refined",
                "artifacts",
                "error",
                "raw_text",
                "multi_feature_results",
                "multi_feature_units",
                "multi_feature_index",
                "active_intake_unit",
                "intake_level_display",
            ) or key.startswith("clar_"):
                del st.session_state[key]
        st.rerun()
