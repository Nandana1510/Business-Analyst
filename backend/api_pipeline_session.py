"""
Server-side pipeline state for the REST API (mirrors ``app.py`` Streamlit flow).
"""

from __future__ import annotations

import uuid
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
    generate_stage1_questions,
    generate_stage2_questions,
    normalize_responses_with_questions,
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
from stages.run_output_storage import persist_multi_feature_run, persist_single_feature_run


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

    def _multi_feature_active(self) -> bool:
        return self.multi_feature_units is not None

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
        self.clarification_cl_stage = 1
        self.clarification_pending_answers_s1 = None
        self.clarification_stage2_notes = None
        self.clarified = None
        self.refined = None
        self.artifacts = None

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
        rl = (au.requirement_level if isinstance(au, NormalizedRequirementUnit) else "") or ""
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
        requirement = RawRequirement(
            text=unit.text,
            intake_feature_label=unit.feature_name or None,
            requirement_level=unit.requirement_level or None,
        )
        understood = understand_requirement(requirement)
        self.understood = understood
        self.raw_text = unit.text
        self.active_intake_unit = unit
        oi = self.intake_open_items or []
        try:
            qs = generate_stage1_questions(understood, unit.text, {}, open_items=oi)
        except Exception as e:
            self.error = f"Clarification Stage 1 failed (you can retry after fixing LLM access): {e}"
            qs = []
        self.clarification_questions = qs
        self.clarification_stage1_questions = list(qs)
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
                )
                artifacts = generate_artifacts_for_mode(
                    refined,
                    artifact_mode,
                    acceptance_criteria_format=acceptance_criteria_format,
                )
                self.clarified = clarified
                self._append_completed_bundle(refined, artifacts, clarified)
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
        refined = refine_requirement(
            self.understood,
            clarification_context=clarified.to_refinement_block(),
            intake_feature_label=ilab,
            requirement_level=ilvl,
            open_items=oi,
        )
        artifacts = generate_artifacts_for_mode(
            refined,
            self.pipeline_artifact_mode,
            acceptance_criteria_format=self.pipeline_acceptance_criteria_format,
        )
        self.clarified = clarified
        self.clarification_questions = []
        self.clarification_stage1_questions = []
        if self._multi_feature_active():
            units = self.multi_feature_units
            assert units is not None
            self._append_completed_bundle(refined, artifacts, clarified)
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

    def reset_pipeline(self) -> None:
        self.understood = None
        self.clarification_questions = []
        self.clarification_stage1_questions = []
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

    def generate(
        self,
        requirement_text: str | None,
        file_name: str | None,
        file_bytes: bytes | None,
        artifact_mode: str,
        acceptance_criteria_format: str,
    ) -> None:
        self.reset_pipeline()
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
        try:
            units: list[NormalizedRequirementUnit] = analyze_intake(raw)
        except Exception as e:
            self.error = f"Pre-processing failed: {e}"
            units = [NormalizedRequirementUnit(text=raw, requirement_level="feature")]

        if not units:
            self.error = "No processable requirement text."
            return

        if len(units) > 1:
            try:
                self.multi_feature_units = units
                self.multi_feature_index = 0
                self.multi_feature_results = []
                self._run_understanding_for_unit(units[0])
                self._multi_feature_auto_advance_through_empty_clarification()
            except Exception as e:
                self.error = str(e)
                self.multi_feature_units = None
                self.multi_feature_index = 0
                self.multi_feature_results = None
                self.understood = None
                self.clarification_questions = []
                self.clarification_stage1_questions = []
                self.clarification_cl_stage = 1
                self.clarification_pending_answers_s1 = None
                self.clarification_stage2_notes = None
                self.clarified = None
                self.refined = None
                self.artifacts = None
        else:
            try:
                u0 = units[0]
                self.active_intake_unit = u0
                self.intake_level_display = u0.requirement_level
                understood = understand_requirement(RawRequirement(text=u0.text))
                self.understood = understood
                self.raw_text = u0.text
                oi = self.intake_open_items or []
                questions = generate_stage1_questions(
                    understood,
                    self.raw_text,
                    {},
                    open_items=oi,
                )
                self.clarification_questions = questions
                self.clarification_stage1_questions = list(questions)
                self.clarification_cl_stage = 1
                self.clarification_pending_answers_s1 = None
                self.clarification_stage2_notes = None
                if not questions:
                    clarified = ClarifiedRequirement.from_answers(understood, {})
                    refined = refine_requirement(
                        understood,
                        clarification_context=clarified.to_refinement_block(),
                        intake_feature_label=u0.feature_name or None,
                        requirement_level=u0.requirement_level,
                        open_items=oi,
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
            except Exception as e:
                self.error = str(e)
                self.understood = None
                self.clarification_questions = []
                self.clarification_stage1_questions = []
                self.clarification_cl_stage = 1
                self.clarification_pending_answers_s1 = None
                self.clarification_stage2_notes = None
                self.clarified = None
                self.refined = None
                self.artifacts = None

    def continue_empty_clarification(self) -> None:
        self.error = None
        try:
            self._multi_feature_auto_advance_through_empty_clarification()
        except Exception as e:
            self.error = str(e)

    def submit_clarification(self, answers: dict[str, str]) -> tuple[bool, str | None]:
        """
        Process clarification submit like Streamlit.
        Returns (ok, error_message_if_any).
        """
        if self.understood is None or not self.clarification_questions or self.clarified is not None:
            return False, "No clarification step is active."
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
                self.raw_text or "",
                cr1,
                validation_has_errors=vr1.has_errors,
                validation_has_warnings=vr1.has_warnings,
            )
            missing = detect_missing_fields(self.understood, norm1)
            cap2 = max(0, CLARIFICATION_MAX_TOTAL_QUESTIONS - len(s1qs))
            q2: list[ClarificationQuestion] = []
            if need2 and cap2 > 0:
                oi2 = self.intake_open_items or []
                q2 = generate_stage2_questions(
                    self.understood,
                    self.raw_text or "",
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
        }
