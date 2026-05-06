"""
Server-side pipeline state for the REST API (mirrors ``app.py`` Streamlit flow).
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from stages.artifact_generation import (
    AC_FORMAT_BDD,
    AC_FORMAT_DECLARATIVE,
    AdvancedArtifacts,
    generate_artifacts_for_mode,
)
from stages.clarification_consistency import validate_clarification_answers
from stages.requirement_clarification import (
    CLARIFICATION_MAX_TOTAL_QUESTIONS,
    ClarifiedRequirement,
    ClarificationQuestion,
    clarification_needs_stage2,
    detect_missing_fields,
    effective_requirement_level_for_artifacts,
    format_prior_clarification_log_for_prompt,
    generate_stage1_questions,
    generate_stage2_questions,
    merge_stage1_questions_with_optional_granularity,
    normalize_responses_with_questions,
    stage1_effective_budget_count,
)
from stages.requirement_entry import (
    DocumentExtractionError,
    RawRequirement,
    raw_requirement_from_file,
    raw_requirement_from_manual_text,
)
from stages.requirement_intake import NormalizedRequirementUnit, analyze_intake
from stages.requirement_refinement import RefinedRequirement, refine_requirement
from stages.requirement_understanding import (
    UnderstoodRequirement,
    understand_requirement,
    get_llm_provider_and_model,
)
from stages.gap_to_requirement import (
    build_combined_requirement_text,
    convert_gaps_to_requirement_statements,
    parse_gap_focus_bullet_statements,
    per_unit_gap_focus_blocks,
    split_core_and_gap_supplement,
)
from stages.pipeline_logging import pipeline_log
from stages.run_output_storage import persist_multi_feature_run, persist_single_feature_run


def _string_lines_from_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        t = str(x).strip()
        if t:
            out.append(t)
    return out


def _gaps_from_feature_dict(f: dict[str, Any]) -> list[str]:
    """
    Gaps on one feature object: ``gap_analysis`` / ``gapAnalysis``, then any nested ``features`` /
    ``sub_features`` buckets (same ordering as the UI flatten).
    """
    out: list[str] = []
    out.extend(_string_lines_from_list(f.get("gap_analysis")))
    out.extend(_string_lines_from_list(f.get("gapAnalysis")))
    nested = f.get("features") or f.get("sub_features")
    if isinstance(nested, list):
        for sub in nested:
            if isinstance(sub, dict):
                out.extend(_gaps_from_feature_dict(sub))
    return out


def _gap_lines_from_artifacts_dict(d: dict[str, Any]) -> list[str]:
    """Top-level gaps (``gap_analysis`` / ``gapAnalysis``), then each root ``features[]`` tree in order."""
    out: list[str] = []
    out.extend(_string_lines_from_list(d.get("gap_analysis")))
    out.extend(_string_lines_from_list(d.get("gapAnalysis")))
    feats = d.get("features")
    if isinstance(feats, list):
        for f in feats:
            if isinstance(f, dict):
                out.extend(_gaps_from_feature_dict(f))
    return out


def _validate_gap_selection(selected: list[str], allowed: list[str]) -> list[str] | None:
    """Return ordered validated lines, or None if any selection is not in the multiset ``allowed``."""
    pool = Counter(allowed)
    out: list[str] = []
    for raw in selected:
        s = (raw or "").strip()
        if not s:
            continue
        if pool[s] <= 0:
            return None
        pool[s] -= 1
        out.append(s)
    return out if out else None


def _gap_lines_from_multi_feature_bundles(bundles: list[FeaturePipelineBundle]) -> list[str]:
    """Same order as the UI: each bundle's flattened gaps (top-level then per-feature), in bundle order."""
    out: list[str] = []
    for b in bundles:
        out.extend(_gap_lines_from_artifacts_dict(b.artifacts.to_dict()))
    return out


@dataclass
class FeaturePipelineBundle:
    index: int
    total: int
    unit_text: str
    understood: UnderstoodRequirement
    refined: RefinedRequirement
    artifacts: AdvancedArtifacts
    clarified: ClarifiedRequirement | None = None
    feature_label: str = ""
    requirement_level: str = ""


@dataclass
class PipelineSession:
    """One browser/API session — in-memory only (lost on server restart)."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    understood: UnderstoodRequirement | None = None
    clarification_questions: list[ClarificationQuestion] = field(default_factory=list)
    clarification_stage1_questions: list[ClarificationQuestion] = field(default_factory=list)
    # Stage-1 slots excluded from CLARIFICATION_MAX_TOTAL_QUESTIONS (e.g. optional granularity question).
    clarification_stage1_budget_exclusions: int = 0
    clarification_cl_stage: int = 1
    clarification_pending_answers_s1: dict[str, str] | None = None
    clarification_stage2_notes: list[str] | None = None
    clarified: ClarifiedRequirement | None = None
    refined: RefinedRequirement | None = None
    artifacts: AdvancedArtifacts | None = None
    error: str | None = None
    multi_feature_units: list[NormalizedRequirementUnit] | None = None
    multi_feature_index: int = 0
    multi_feature_results: list[FeaturePipelineBundle] | None = None
    raw_text: str = ""
    original_requirement_text: str = ""
    intake_open_items: list[str] = field(default_factory=list)
    active_intake_unit: NormalizedRequirementUnit | None = None
    intake_level_display: str | None = None
    pipeline_artifact_mode: str = "all"
    pipeline_acceptance_criteria_format: str = AC_FORMAT_DECLARATIVE
    # Completed clarification rounds for the current run only (cleared on reset / new generate).
    clarification_log: list[dict[str, Any]] = field(default_factory=list)
    # Gap-derived / supplementary bullets: refinement + understanding enrichment only (not intake).
    pipeline_gap_focus_block: str = ""
    # When intake splits into multiple units, each index matches ``multi_feature_units`` order only.
    pipeline_gap_focus_by_unit: list[str] | None = None
    # Formatted prior clarification history for Stage 1 (gap regeneration); cleared on reset.
    stage1_prior_clarification_for_prompt: str = ""

    def _multi_feature_active(self) -> bool:
        return self.multi_feature_units is not None

    def _gap_focus_for_current_unit(self) -> str:
        """
        Intake supplementary (e.g. capability bucket list) plus gap-focus bullets for the active unit.
        """
        bu = self.pipeline_gap_focus_by_unit
        gap_part = ""
        if bu and self._multi_feature_active() and self.multi_feature_units:
            if len(bu) == len(self.multi_feature_units):
                i = self.multi_feature_index
                if 0 <= i < len(bu):
                    gap_part = (bu[i] or "").strip()
        else:
            gap_part = (self.pipeline_gap_focus_block or "").strip()
        au = self.active_intake_unit
        intake_sup = ""
        if isinstance(au, NormalizedRequirementUnit):
            intake_sup = (getattr(au, "supplementary_constraints", None) or "").strip()
        parts = [p for p in (intake_sup, gap_part) if p]
        return "\n\n".join(parts)

    def _collapse_from_active_unit(self) -> bool:
        au = self.active_intake_unit
        if isinstance(au, NormalizedRequirementUnit):
            return bool(getattr(au, "collapse_product_feature_decomposition", False))
        return False

    def _clarification_requirement_context(self) -> str:
        """
        **Sub-requirement only:** active intake unit text (``raw_text``). Per-feature isolation —
        do not use the full original paste / combined requirement here (clarification, refinement
        narrative, artifact grounding for this slice).
        """
        return (self.raw_text or "").strip()

    def _persist_multi_feature_run_if_ready(self) -> None:
        bundles = self.multi_feature_results
        if not bundles:
            return
        persist_multi_feature_run(
            original_requirement=self.original_requirement_text or "",
            open_items=list(self.intake_open_items or []),
            artifact_scope_mode=self.pipeline_artifact_mode or "",
            acceptance_criteria_format=self.pipeline_acceptance_criteria_format or "",
            bundles=list(bundles),
        )

    def _persist_single_feature_completed(self) -> None:
        if self._multi_feature_active():
            return
        if self.refined is None or self.artifacts is None or self.understood is None:
            return
        persist_single_feature_run(
            original_requirement=self.original_requirement_text or (self.raw_text or ""),
            open_items=list(self.intake_open_items or []),
            artifact_scope_mode=self.pipeline_artifact_mode or "",
            acceptance_criteria_format=self.pipeline_acceptance_criteria_format or "",
            understood=self.understood,
            refined=self.refined,
            artifacts=self.artifacts,
            clarified=self.clarified,
        )

    def _multi_feature_finish_cleanup(self) -> None:
        self._persist_multi_feature_run_if_ready()
        self.multi_feature_units = None
        self.multi_feature_index = 0
        self.understood = None
        self.clarification_questions = []
        self.clarification_stage1_questions = []
        self.clarification_stage1_budget_exclusions = 0
        self.clarification_cl_stage = 1
        self.clarification_pending_answers_s1 = None
        self.clarification_stage2_notes = None
        self.clarified = None
        self.refined = None
        self.artifacts = None
        n = len(self.multi_feature_results or [])
        if n:
            pipeline_log().info(
                "[%s] multi-feature run finished: %d bundle(s) persisted",
                self.session_id,
                n,
            )

    def _append_completed_bundle(
        self,
        refined: RefinedRequirement,
        artifacts: AdvancedArtifacts,
        clarified: ClarifiedRequirement,
    ) -> None:
        units = self.multi_feature_units
        if not units:
            return
        idx = self.multi_feature_index
        au = self.active_intake_unit
        fl = (au.feature_name if isinstance(au, NormalizedRequirementUnit) else "") or ""
        rl = (getattr(refined, "requirement_level", None) or "").strip() or (
            (au.requirement_level if isinstance(au, NormalizedRequirementUnit) else "") or ""
        )
        assert self.multi_feature_results is not None
        self.multi_feature_results.append(
            FeaturePipelineBundle(
                index=idx + 1,
                total=len(units),
                unit_text=self.raw_text or "",
                understood=self.understood,  # type: ignore[arg-type]
                refined=refined,
                artifacts=artifacts,
                clarified=clarified,
                feature_label=fl,
                requirement_level=rl,
            )
        )

    def _run_understanding_for_unit(self, unit: NormalizedRequirementUnit) -> None:
        nu = self.multi_feature_units
        if nu:
            pipeline_log().info(
                "[%s] multi-feature: unit %s/%s — understanding + clarification prep (%s)",
                self.session_id,
                self.multi_feature_index + 1,
                len(nu),
                (unit.feature_name or "—")[:48],
            )
        self.active_intake_unit = unit
        self.raw_text = unit.text
        requirement = RawRequirement(
            text=unit.text,
            intake_feature_label=unit.feature_name or None,
            requirement_level=unit.requirement_level or None,
            supplementary_constraints=self._gap_focus_for_current_unit(),
        )
        understood = understand_requirement(requirement)
        self.understood = understood
        oi = self.intake_open_items or []
        clar_raw = (unit.text or "").strip()
        gap_s1 = self._gap_focus_for_current_unit()
        try:
            core = generate_stage1_questions(
                understood,
                clar_raw,
                {},
                open_items=oi,
                prior_clarification_for_prompt=self.stage1_prior_clarification_for_prompt,
                gap_derived_scope_text=gap_s1,
            )
        except Exception as e:
            self.error = f"Clarification Stage 1 failed (you can retry after fixing LLM access): {e}"
            core = []
        intake_n = len(self.multi_feature_units) if self.multi_feature_units else 1
        intake_blob = (self.original_requirement_text or clar_raw or "").strip()
        supp_u = (getattr(unit, "supplementary_constraints", None) or "").strip()
        qs, budget_excl = merge_stage1_questions_with_optional_granularity(
            core,
            understood,
            clar_raw,
            unit.requirement_level,
            intake_n,
            product_intent_source_text=intake_blob,
            intake_supplementary_constraints=supp_u or None,
        )
        self.clarification_questions = qs
        self.clarification_stage1_questions = list(qs)
        self.clarification_stage1_budget_exclusions = budget_excl
        self.clarification_cl_stage = 1
        self.clarification_pending_answers_s1 = None
        self.clarification_stage2_notes = None
        self.clarified = None
        self.refined = None
        self.artifacts = None

    def _multi_feature_auto_advance_through_empty_clarification(self) -> None:
        units = self.multi_feature_units
        if not units:
            return
        artifact_mode = self.pipeline_artifact_mode
        acceptance_criteria_format = self.pipeline_acceptance_criteria_format
        while True:
            if not self.clarification_questions:
                if self.understood is None:
                    return
                self._log_no_clarification_for_feature()
                clarified = ClarifiedRequirement.from_answers(self.understood, {})
                au = self.active_intake_unit
                intake_lbl = au.feature_name if isinstance(au, NormalizedRequirementUnit) else None
                lvl = au.requirement_level if isinstance(au, NormalizedRequirementUnit) else None
                oi = self.intake_open_items or []
                refined = refine_requirement(
                    self.understood,
                    clarification_context=clarified.to_refinement_block(),
                    intake_feature_label=intake_lbl,
                    requirement_level=lvl,
                    open_items=oi,
                    gap_focus_block=self._gap_focus_for_current_unit() or None,
                    full_requirement_narrative=self._clarification_requirement_context() or None,
                    collapse_product_feature_decomposition=self._collapse_from_active_unit(),
                )
                artifacts = generate_artifacts_for_mode(
                    refined,
                    artifact_mode,
                    acceptance_criteria_format=acceptance_criteria_format,
                )
                self.clarified = clarified
                self._append_completed_bundle(refined, artifacts, clarified)
                pipeline_log().info(
                    "[%s] multi-feature: bundle %s/%s stored",
                    self.session_id,
                    self.multi_feature_index + 1,
                    len(units),
                )
                if self.multi_feature_index + 1 < len(units):
                    self.multi_feature_index += 1
                    self._run_understanding_for_unit(units[self.multi_feature_index])
                    continue
                self._multi_feature_finish_cleanup()
                return
            return

    def _run_refinement_after_clarification(self, clarified: ClarifiedRequirement) -> None:
        self.clarification_cl_stage = 1
        self.clarification_pending_answers_s1 = None
        self.clarification_stage2_notes = None
        au2 = self.active_intake_unit
        ilab = au2.feature_name if isinstance(au2, NormalizedRequirementUnit) else None
        ilvl = au2.requirement_level if isinstance(au2, NormalizedRequirementUnit) else None
        oi = self.intake_open_items or []
        assert self.understood is not None
        eff_level = effective_requirement_level_for_artifacts(clarified, ilvl)
        refined = refine_requirement(
            self.understood,
            clarification_context=clarified.to_refinement_block(),
            intake_feature_label=ilab,
            requirement_level=eff_level,
            open_items=oi,
            gap_focus_block=self._gap_focus_for_current_unit() or None,
            full_requirement_narrative=self._clarification_requirement_context() or None,
            collapse_product_feature_decomposition=self._collapse_from_active_unit(),
        )
        artifacts = generate_artifacts_for_mode(
            refined,
            self.pipeline_artifact_mode,
            acceptance_criteria_format=self.pipeline_acceptance_criteria_format,
        )
        self.clarified = clarified
        self.clarification_questions = []
        self.clarification_stage1_questions = []
        self.clarification_stage1_budget_exclusions = 0
        if self._multi_feature_active():
            units = self.multi_feature_units
            assert units is not None
            self._append_completed_bundle(refined, artifacts, clarified)
            pipeline_log().info(
                "[%s] multi-feature: bundle %s/%s stored (after clarification)",
                self.session_id,
                self.multi_feature_index + 1,
                len(units),
            )
            if self.multi_feature_index + 1 < len(units):
                self.multi_feature_index += 1
                self._run_understanding_for_unit(units[self.multi_feature_index])
                self._multi_feature_auto_advance_through_empty_clarification()
            else:
                self._multi_feature_finish_cleanup()
        else:
            self.refined = refined
            self.artifacts = artifacts
            self._persist_single_feature_completed()
            pipeline_log().info(
                "[%s] clarification path complete — refinement + artifacts saved",
                self.session_id,
            )

    def reset_pipeline(self) -> None:
        self.clarification_log = []
        self.pipeline_gap_focus_block = ""
        self.pipeline_gap_focus_by_unit = None
        self.stage1_prior_clarification_for_prompt = ""
        self.understood = None
        self.clarification_questions = []
        self.clarification_stage1_questions = []
        self.clarification_stage1_budget_exclusions = 0
        self.clarification_cl_stage = 1
        self.clarification_pending_answers_s1 = None
        self.clarification_stage2_notes = None
        self.clarified = None
        self.refined = None
        self.artifacts = None
        self.error = None
        self.raw_text = ""
        self.multi_feature_results = None
        self.multi_feature_units = None
        self.multi_feature_index = 0
        self.active_intake_unit = None
        self.intake_level_display = None

    def _feature_meta_for_log(self) -> tuple[int | None, int | None, str]:
        """1-based feature index, total features, label — for grouping clarification history."""
        if self._multi_feature_active() and self.multi_feature_units:
            idx = self.multi_feature_index + 1
            tot = len(self.multi_feature_units)
            au = self.active_intake_unit
            fl = (au.feature_name if isinstance(au, NormalizedRequirementUnit) else "") or ""
            return idx, tot, fl
        au = self.active_intake_unit
        fl = (au.feature_name if isinstance(au, NormalizedRequirementUnit) else "") or ""
        return 1, 1, fl

    def _append_clarification_round(
        self,
        stage: int,
        questions: list[ClarificationQuestion],
        answers: dict[str, str],
    ) -> None:
        """Record one submitted clarification stage (Q + selected/normalized answers)."""
        fi, ft, fl = self._feature_meta_for_log()
        items: list[dict[str, str]] = []
        for q in questions:
            raw = (answers.get(q.category) or "").strip()
            items.append(
                {
                    "category": q.category,
                    "question": q.question,
                    "answer": raw if raw else "(not answered)",
                }
            )
        self.clarification_log.append(
            {
                "stage": stage,
                "feature_index": fi,
                "feature_total": ft,
                "feature_label": fl,
                "items": items,
            }
        )

    def _log_no_clarification_for_feature(self) -> None:
        """Pipeline skipped clarification (no questions) for the current feature slice."""
        fi, ft, fl = self._feature_meta_for_log()
        self.clarification_log.append(
            {
                "stage": 0,
                "feature_index": fi,
                "feature_total": ft,
                "feature_label": fl,
                "items": [],
                "note": "No clarification questions were generated for this feature.",
            }
        )

    def generate(
        self,
        requirement_text: str | None,
        file_name: str | None,
        file_bytes: bytes | None,
        artifact_mode: str,
        acceptance_criteria_format: str,
        *,
        intake_analyze_text: str | None = None,
        gap_focus_block: str | None = None,
        prior_clarification_log: list[dict[str, Any]] | None = None,
    ) -> None:
        self.reset_pipeline()
        self.stage1_prior_clarification_for_prompt = format_prior_clarification_log_for_prompt(
            prior_clarification_log or []
        )
        self.pipeline_artifact_mode = artifact_mode
        self.pipeline_acceptance_criteria_format = acceptance_criteria_format
        requirement: RawRequirement | None = None
        try:
            if file_bytes is not None and file_name:
                requirement = raw_requirement_from_file(file_name, file_bytes)
            else:
                if not (requirement_text and requirement_text.strip()):
                    self.error = "Please enter a requirement or upload a document."
                    return
                requirement = raw_requirement_from_manual_text(requirement_text)
        except DocumentExtractionError as e:
            self.error = f"Could not read the file: {e}"
            return
        except ValueError as e:
            self.error = str(e)
            return
        except Exception as e:
            self.error = f"Could not process input: {e}"
            return

        raw = requirement.text
        self.original_requirement_text = raw
        self.intake_open_items = list(requirement.open_items or [])
        self.multi_feature_units = None
        self.multi_feature_index = 0
        self.multi_feature_results = None
        from_file = file_bytes is not None and bool(file_name)
        intake_src = raw
        gap_fb = ""
        if not from_file and gap_focus_block is None and intake_analyze_text is None:
            core_auto, supp_auto = split_core_and_gap_supplement(raw)
            if supp_auto:
                intake_src = (core_auto or raw).strip() or raw
                gap_fb = supp_auto
        if gap_focus_block is not None:
            gap_fb = (gap_focus_block or "").strip()
        if intake_analyze_text is not None:
            t = (intake_analyze_text or "").strip()
            intake_src = t if t else raw
        try:
            units: list[NormalizedRequirementUnit] = analyze_intake(intake_src)
        except Exception as e:
            self.error = f"Pre-processing failed: {e}"
            units = [NormalizedRequirementUnit(text=intake_src, requirement_level="feature")]

        if not units:
            self.error = "No processable requirement text."
            return

        self.pipeline_gap_focus_by_unit = None
        stmts = parse_gap_focus_bullet_statements(gap_fb)
        if not stmts and (gap_fb or "").strip():
            stmts = [(gap_fb or "").strip()]
        if len(units) > 1 and stmts:
            self.pipeline_gap_focus_by_unit = per_unit_gap_focus_blocks(stmts, units)
            self.pipeline_gap_focus_block = ""
        else:
            self.pipeline_gap_focus_block = gap_fb

        pipeline_log().info(
            "[%s] generate: %d intake unit(s); intake_chars=%d; gap_supplement=%s; per_unit_gaps=%s; file=%s; mode=%s",
            self.session_id,
            len(units),
            len(intake_src or ""),
            bool((gap_fb or "").strip()),
            self.pipeline_gap_focus_by_unit is not None,
            from_file,
            artifact_mode,
        )

        if len(units) > 1:
            try:
                pipeline_log().info(
                    "[%s] generate: multi-feature — processing units sequentially",
                    self.session_id,
                )
                self.multi_feature_units = units
                self.multi_feature_index = 0
                self.multi_feature_results = []
                self._run_understanding_for_unit(units[0])
                self._multi_feature_auto_advance_through_empty_clarification()
            except Exception as e:
                self.error = str(e)
                pipeline_log().warning("[%s] generate: multi-feature error — %s", self.session_id, e)
                self.clarification_log = []
                self.multi_feature_units = None
                self.multi_feature_index = 0
                self.multi_feature_results = None
                self.understood = None
                self.clarification_questions = []
                self.clarification_stage1_questions = []
                self.clarification_stage1_budget_exclusions = 0
                self.clarification_cl_stage = 1
                self.clarification_pending_answers_s1 = None
                self.clarification_stage2_notes = None
                self.clarified = None
                self.refined = None
                self.artifacts = None
        else:
            try:
                pipeline_log().info("[%s] generate: single intake unit", self.session_id)
                u0 = units[0]
                self.active_intake_unit = u0
                self.intake_level_display = u0.requirement_level
                understood = understand_requirement(
                    RawRequirement(
                        text=u0.text,
                        intake_feature_label=u0.feature_name or None,
                        requirement_level=u0.requirement_level or None,
                        supplementary_constraints=self._gap_focus_for_current_unit(),
                        open_items=list(self.intake_open_items or []),
                    )
                )
                self.understood = understood
                self.raw_text = u0.text
                oi = self.intake_open_items or []
                clar_raw = (self.raw_text or "").strip()
                gap_s1 = self._gap_focus_for_current_unit().strip()
                core = generate_stage1_questions(
                    understood,
                    clar_raw,
                    {},
                    open_items=oi,
                    prior_clarification_for_prompt=self.stage1_prior_clarification_for_prompt,
                    gap_derived_scope_text=gap_s1,
                )
                intake_blob = (self.original_requirement_text or clar_raw or "").strip()
                supp0 = (getattr(u0, "supplementary_constraints", None) or "").strip()
                questions, budget_excl = merge_stage1_questions_with_optional_granularity(
                    core,
                    understood,
                    clar_raw,
                    u0.requirement_level,
                    len(units),
                    product_intent_source_text=intake_blob,
                    intake_supplementary_constraints=supp0 or None,
                )
                self.clarification_questions = questions
                self.clarification_stage1_questions = list(questions)
                self.clarification_stage1_budget_exclusions = budget_excl
                self.clarification_cl_stage = 1
                self.clarification_pending_answers_s1 = None
                self.clarification_stage2_notes = None
                if not questions:
                    self._log_no_clarification_for_feature()
                    clarified = ClarifiedRequirement.from_answers(understood, {})
                    refined = refine_requirement(
                        understood,
                        clarification_context=clarified.to_refinement_block(),
                        intake_feature_label=u0.feature_name or None,
                        requirement_level=u0.requirement_level,
                        open_items=oi,
                        gap_focus_block=self._gap_focus_for_current_unit() or None,
                        full_requirement_narrative=self._clarification_requirement_context() or None,
                        collapse_product_feature_decomposition=bool(
                            getattr(u0, "collapse_product_feature_decomposition", False)
                        ),
                    )
                    artifacts = generate_artifacts_for_mode(
                        refined,
                        artifact_mode,
                        acceptance_criteria_format=acceptance_criteria_format,
                    )
                    self.clarified = clarified
                    self.refined = refined
                    self.artifacts = artifacts
                    self._persist_single_feature_completed()
                    pipeline_log().info(
                        "[%s] generate: done (skipped clarification); artifacts ready",
                        self.session_id,
                    )
                else:
                    pipeline_log().info(
                        "[%s] generate: awaiting user clarification (%d questions)",
                        self.session_id,
                        len(questions),
                    )
            except Exception as e:
                self.error = str(e)
                pipeline_log().warning("[%s] generate: single-unit error — %s", self.session_id, e)
                self.clarification_log = []
                self.understood = None
                self.clarification_questions = []
                self.clarification_stage1_questions = []
                self.clarification_stage1_budget_exclusions = 0
                self.clarification_cl_stage = 1
                self.clarification_pending_answers_s1 = None
                self.clarification_stage2_notes = None
                self.clarified = None
                self.refined = None
                self.artifacts = None

    def continue_empty_clarification(self) -> None:
        self.error = None
        pipeline_log().info("[%s] continue_empty_clarification (multi-feature advance)", self.session_id)
        try:
            self._multi_feature_auto_advance_through_empty_clarification()
        except Exception as e:
            self.error = str(e)
            pipeline_log().warning("[%s] continue_empty_clarification failed — %s", self.session_id, e)

    def regenerate_with_selected_gaps(self, selected_gap_texts: list[str]) -> None:
        """
        Convert selected gap lines into requirement statements, compose a new requirement body from
        the stored original requirement, and **re-run the full pipeline** (``generate``): intake,
        understanding, clarification, refinement, artifacts.

        Gap lines may come from the current single-feature ``artifacts`` or from **completed**
        multi-feature bundle outputs (same flattened order as the UI).
        """
        self.error = None
        if self._multi_feature_active():
            self.error = "Finish the multi-feature run before regenerating from gaps."
            return
        pipeline_log().info(
            "[%s] regenerate_with_gaps: %d gap line(s) selected",
            self.session_id,
            len(list(selected_gap_texts or [])),
        )
        if self.artifacts is not None:
            allowed = _gap_lines_from_artifacts_dict(self.artifacts.to_dict())
        elif self.multi_feature_results:
            allowed = _gap_lines_from_multi_feature_bundles(self.multi_feature_results)
        else:
            self.error = "Generate artifacts first before regenerating from gaps."
            return
        if not allowed:
            self.error = "No gap analysis lines are available to regenerate from."
            return
        validated = _validate_gap_selection(list(selected_gap_texts or []), allowed)
        if validated is None:
            self.error = (
                "Invalid gap selection: pick one or more items exactly as shown in the current "
                "gap analysis list."
            )
            return
        base = (self.original_requirement_text or self.raw_text or "").strip()
        if not base:
            self.error = "Original requirement text is missing; run a full generate first."
            return
        try:
            statements = convert_gaps_to_requirement_statements(validated)
            core_base, _ = split_core_and_gap_supplement(base)
            if not (core_base or "").strip():
                core_base = base
            combined = build_combined_requirement_text(core_base, statements)
            gap_lines = "\n".join(f"- {s}" for s in statements)
            prior_log = list(self.clarification_log or [])
            self.generate(
                combined,
                None,
                None,
                self.pipeline_artifact_mode or "all",
                self.pipeline_acceptance_criteria_format or AC_FORMAT_DECLARATIVE,
                intake_analyze_text=core_base,
                gap_focus_block=gap_lines,
                prior_clarification_log=prior_log,
            )
        except Exception as e:
            self.error = str(e)
            pipeline_log().warning("[%s] regenerate_with_gaps failed — %s", self.session_id, e)

    def submit_clarification(self, answers: dict[str, str]) -> tuple[bool, str | None]:
        """
        Process clarification submit like Streamlit.
        Returns (ok, error_message_if_any).
        """
        if self.understood is None or not self.clarification_questions or self.clarified is not None:
            return False, "No clarification step is active."
        pipeline_log().info(
            "[%s] clarification: submitting stage %s (%d questions in round)",
            self.session_id,
            self.clarification_cl_stage,
            len(self.clarification_questions),
        )
        cl_stage = int(self.clarification_cl_stage or 1)
        cur_questions = self.clarification_questions
        answers_dict: dict[str, str] = {}
        for cq in cur_questions:
            raw_ans = (answers.get(cq.category) or "").strip()
            answers_dict[cq.category] = raw_ans

        if cl_stage == 1:
            s1qs = self.clarification_stage1_questions or cur_questions
            norm1 = normalize_responses_with_questions(answers_dict, s1qs)
            cr1 = ClarifiedRequirement.from_answers(self.understood, norm1, s1qs)
            vr1 = validate_clarification_answers(cr1)
            need2 = clarification_needs_stage2(
                self.understood,
                self._clarification_requirement_context(),
                cr1,
                validation_has_errors=vr1.has_errors,
                validation_has_warnings=vr1.has_warnings,
            )
            missing = detect_missing_fields(self.understood, norm1)
            cap2 = max(
                0,
                CLARIFICATION_MAX_TOTAL_QUESTIONS
                - stage1_effective_budget_count(s1qs, self.clarification_stage1_budget_exclusions),
            )
            q2: list[ClarificationQuestion] = []
            if need2 and cap2 > 0:
                oi2 = self.intake_open_items or []
                q2 = generate_stage2_questions(
                    self.understood,
                    self._clarification_requirement_context(),
                    norm1,
                    list(s1qs),
                    list(vr1.issues),
                    missing,
                    max_additional=cap2,
                    open_items=oi2,
                )
            if need2 and q2:
                self._append_clarification_round(1, list(s1qs), norm1)
                note_lines: list[str] = [msg for _sev, msg in vr1.issues]
                if missing:
                    note_lines.append(
                        "Required areas still empty or generic: " + ", ".join(missing)
                    )
                if not note_lines:
                    note_lines = [
                        "A short follow-up was suggested to remove ambiguity before refinement."
                    ]
                self.clarification_stage2_notes = note_lines
                self.clarification_pending_answers_s1 = norm1
                self.clarification_questions = q2
                self.clarification_cl_stage = 2
                return True, None
            if need2 and not q2 and (vr1.has_errors or vr1.has_warnings or missing):
                msg = (
                    "Follow-up clarification was required, but no extra questions could be added—"
                    f"often because the question budget is full (max {CLARIFICATION_MAX_TOTAL_QUESTIONS} total). "
                    "Clear and start over, or shorten answers and resubmit Stage 1."
                )
                issues_text = "; ".join(f"[{s}] {m}" for s, m in vr1.issues)
                return False, f"{msg} Details: {issues_text}"
            vr = validate_clarification_answers(cr1)
            if vr.has_errors:
                return False, "; ".join(m for sev, m in vr.issues if sev == "error") or "Validation failed."
            self._append_clarification_round(1, list(s1qs), norm1)
            try:
                self._run_refinement_after_clarification(cr1)
            except Exception as e:
                return False, str(e)
            return True, None

        if cl_stage == 2:
            norm2 = normalize_responses_with_questions(answers_dict, cur_questions)
            pending = self.clarification_pending_answers_s1 or {}
            full_answers = {**pending, **norm2}
            s1qs2 = self.clarification_stage1_questions or []
            all_qs = list(s1qs2) + list(cur_questions)
            clarified = ClarifiedRequirement.from_answers(self.understood, full_answers, all_qs)
            vr = validate_clarification_answers(clarified)
            if vr.has_errors:
                return False, "; ".join(m for sev, m in vr.issues if sev == "error") or "Validation failed."
            self._append_clarification_round(2, list(cur_questions), norm2)
            try:
                self._run_refinement_after_clarification(clarified)
            except Exception as e:
                return False, str(e)
            return True, None

        return False, "Unknown clarification stage."

    def _bundle_to_dict(self, b: FeaturePipelineBundle) -> dict[str, Any]:
        return {
            "index": b.index,
            "total": b.total,
            "unit_text": b.unit_text,
            "understood": b.understood.to_dict(),
            "refined_text": b.refined.format_output(),
            "refined": b.refined.to_dict(),
            "artifacts": b.artifacts.to_dict(),
            "clarification_context": (b.clarified.to_clarification_context() or "").strip()
            if b.clarified
            else "",
            "feature_label": b.feature_label,
            "requirement_level": b.requirement_level,
        }

    def public_state(self) -> dict[str, Any]:
        """JSON-serializable snapshot for the React client."""
        llm = get_llm_provider_and_model()
        multi_active = self._multi_feature_active()
        clar: dict[str, Any] | None = None
        if (
            self.understood is not None
            and self.clarification_questions
            and self.clarified is None
        ):
            clar = {
                "stage": self.clarification_cl_stage,
                "questions": [
                    {
                        "category": q.category,
                        "question": q.question,
                        "options": list(q.options),
                    }
                    for q in self.clarification_questions
                ],
                "stage2_notes": self.clarification_stage2_notes,
            }
        intake_info: dict[str, str] = {}
        if self.intake_level_display:
            intake_info["level"] = self.intake_level_display
        au = self.active_intake_unit
        if isinstance(au, NormalizedRequirementUnit) and au.feature_name:
            intake_info["feature_label"] = au.feature_name

        needs_continue = bool(
            multi_active
            and self.understood is not None
            and not self.clarification_questions
            and self.clarified is None
            and self.refined is None
        )

        multi_results_out = None
        if self.multi_feature_results and not multi_active:
            multi_results_out = [self._bundle_to_dict(b) for b in self.multi_feature_results]

        single_artifacts = None
        if self.artifacts is not None and not self.multi_feature_results:
            single_artifacts = self.artifacts.to_dict()

        return {
            "session_id": self.session_id,
            "error": self.error,
            "llm": llm,
            "intake_open_items": list(self.intake_open_items or []),
            "original_requirement_text": self.original_requirement_text,
            "multi_feature": (
                {
                    "active": True,
                    "feature_index": self.multi_feature_index,
                    "total": len(self.multi_feature_units or []),
                }
                if multi_active
                else None
            ),
            "multi_feature_results": multi_results_out,
            "understood": self.understood.to_dict() if self.understood else None,
            "clarification": clar,
            "needs_continue_empty_clarification": needs_continue,
            "refined_text": self.refined.format_output() if self.refined else None,
            "refined": self.refined.to_dict() if self.refined else None,
            "artifacts": single_artifacts,
            "clarification_context": (self.clarified.to_clarification_context() or "").strip()
            if self.clarified
            else None,
            "intake": intake_info,
            "clarification_log": list(self.clarification_log),
        }
