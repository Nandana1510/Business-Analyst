"""
Phase 2: Clarification mechanism (two-stage adaptive)

Stage 1: 4–5 base questions (business rules, user flow, constraints, assumptions).
Stage 2: optional follow-up when consistency, required-field gaps, or clarity eval
say more detail is needed. Total questions capped at ``CLARIFICATION_MAX_TOTAL_QUESTIONS``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from stages.requirement_understanding import UnderstoodRequirement, call_llm
from stages.pipeline_logging import agent_log

# UI / CLI: label for free-text when preset options don't fit
OTHER_OPTION_LABEL = "Other (type your answer below)"

# Internal Stage-2 metadata — must not become business rules or appear in saved artifacts.
EXCLUDE_FROM_REFINEMENT_AND_EXPORT: frozenset[str] = frozenset({"stage2_followup_resolution"})

# Fallback when JSON options are missing; each is one outcome, no overlap in meaning.
DEFAULT_SUGGESTED_OPTIONS = [
    "Apply the default policy for this scenario",
    "Apply a different explicit policy (use Other to describe)",
    "Out of scope — not applicable for this requirement",
]

CLARIFICATION_MAX_TOTAL_QUESTIONS = 10
CLARIFICATION_STAGE1_TARGET_MIN = 4
CLARIFICATION_STAGE1_TARGET_MAX = 5

# --- Step 1: Required fields (for detecting what is still missing) ---

REQUIRED_FIELDS_BASE = ["actor", "action"]
CONDITIONAL_FIELDS = {
    "timing": ["subscription", "billing", "delivery", "payment"],
    "billing_behavior": ["subscription", "billing", "payment"],
}
MISSING_VALUES = {"", "unknown", "n/a", "general", "none"}


def format_open_items_for_clarification_prompt(open_items: list[str] | None) -> str:
    """Block appended to clarification prompts when preprocessing routed lines to clarification/gap."""
    items = [x.strip() for x in (open_items or []) if (x or "").strip()]
    if not items:
        return ""
    lines = "\n".join(f"- {it}" for it in items)
    return (
        "\n**Pending discussion / clarification from source (excluded from feature extraction — use to inform "
        "questions; do **not** treat as confirmed product scope unless the requirement text already reflects it):**\n"
        f"{lines}\n"
    )


def _is_empty_or_generic(value: str) -> bool:
    if not value or not str(value).strip():
        return True
    return str(value).strip().lower() in MISSING_VALUES


def _domain_triggers_field(domain: str, trigger_keywords: list[str]) -> bool:
    d = (domain or "").lower()
    return any(kw in d for kw in trigger_keywords)


# --- Step 2: Identify missing information ---


def get_required_fields_for(understood: UnderstoodRequirement) -> list[str]:
    """Return list of field names that are required for this requirement."""
    required = list(REQUIRED_FIELDS_BASE)
    domain = (understood.domain or "").strip()
    for field_name, keywords in CONDITIONAL_FIELDS.items():
        if _domain_triggers_field(domain, keywords):
            required.append(field_name)
    return required


def get_current_values(understood: UnderstoodRequirement, clarified: dict) -> dict:
    """Build a single dict of current values (from understood + clarified)."""
    return {
        "actor": understood.actor or "",
        "secondary_actor": understood.secondary_actor or "",
        "action": understood.action or "",
        "domain": understood.domain or "",
        "timing": clarified.get("timing", ""),
        "billing_behavior": clarified.get("billing_behavior", ""),
        **{k: v for k, v in clarified.items() if k not in ("timing", "billing_behavior")},
    }


def detect_missing_fields(
    understood: UnderstoodRequirement,
    clarified: dict,
) -> list[str]:
    """
    Compare extracted + clarified data with required fields.
    Return list of field names that are missing or too generic.
    """
    required = get_required_fields_for(understood)
    values = get_current_values(understood, clarified)
    missing = []
    for field_name in required:
        val = values.get(field_name, "")
        if isinstance(val, str) and _is_empty_or_generic(val):
            missing.append(field_name)
    return missing


# --- Step 3: Two-stage clarification ---

CLARIFICATION_STAGE1_PROMPT = """You are a business analyst. **Stage 1 — initial clarification:** generate a **base set** of clarification questions with SHORT suggested answers for dropdowns.

**Original requirement (user's words):**
{raw_requirement_text}
{open_items_block}
**Extracted structure:**
- Action: {action}
- Domain: {domain}
- Actor (primary): {actor}
- Secondary actor (human touchpoint): {secondary_actor}
- System impact: {impact}

**Already clarified (do not ask again):**
{already_clarified}

**Stage 1 scope (mandatory):**
- Output **exactly {stage1_min} to {stage1_max}** questions (inclusive). Prefer **{stage1_min}** if the requirement is very narrow; otherwise use **{stage1_max}** when several aspects need decisions.
- Across the set, **cover** where relevant (combine when tightly related):
  - **Business rules / policies**
  - **User flow**
  - **Constraints**
  - **Key assumptions**
- Each question MUST add **meaningful** value; **no** overlapping intent across questions.

**Relevance:**
- Base questions ONLY on the action, domain, and original requirement.
- **Domain-specific wording:** Options must use vocabulary that fits **this** domain (e.g. subscriptions, deliveries, meals)—**not** generic placeholders that could apply to any IT project.
- When **Actor** is **System**, do **not** ask as if the **User** performs automated monitoring; ask about policies, thresholds, or human follow-up only if relevant.
- Keep the user’s concrete terminology in questions and options.

**No invention:** Do **not** ask about accessibility, WCAG, analytics, or topics **not** in the requirement.

**No process meta in questions:** Do **not** ask about "interpretation strategy", "wording priorities", or how to resolve model ambiguity—only **product** decisions the business must make.

If fully explicit, return **{stage1_min}**–**{stage1_max}** short **scope_confirmation** questions—**without** adding new scope.

**Output format — return ONLY valid JSON (no markdown, no extra text):**
{{
  "questions": [
    {{
      "category": "snake_case_label",
      "question": "Clear question text",
      "options": ["Short option 1", "Short option 2", "Short option 3", "Short option 4"]
    }}
  ]
}}

**Options rules (every question):**
- Include exactly 3 or 4 strings in "options".
- **Mutually exclusive**, **logically consistent** sets per question.
- Do **not** use "TBD", "Unclear", "It depends", or "Both" as options.
- Do NOT include an "Other" option — the app adds that separately.
- Categories must be unique snake_case."""


CLARIFICATION_STAGE2_PROMPT = """You are a business analyst. **Stage 2 — targeted follow-up only.** The user already answered an initial clarification set. Generate **additional** questions **only** to resolve gaps below—**no** repetition of Stage 1.

**Original requirement:**
{raw_requirement_text}
{open_items_block}
**Extracted structure:**
- Action: {action}
- Domain: {domain}
- Actor: {actor}
- Secondary actor: {secondary_actor}
- Impact: {impact}

**Stage 1 questions (do NOT repeat themes or categories):**
{stage1_summary}

**Current answers (merged):**
{answers_summary}

**Consistency / gap signals (address when non-empty):**
{validation_and_gaps}

**Mandatory output rule:**
- If the **Consistency / gap signals** block above lists **any** `[error]`, `[warning]`, or missing-field line (i.e. not only the “none” placeholders), you **must** output **at least 1** and **at most {max_additional}** questions. **Never** return `"questions": []` in that case—each question must help resolve those signals with **new** categories.
- Only if signals are **truly empty** (no inconsistencies, no missing-field line, only “none” placeholders) may you return `"questions": []`.

**Rules (strict):**
- Output **at most {max_additional}** new questions; **total** Stage 1 + Stage 2 must not exceed {max_total}.
- Each new **category** must be **new snake_case** not in: {forbidden_categories}
- **Do NOT** re-ask Stage 1 questions verbatim; paraphrase and narrow to the **unresolved** signal.
- Same JSON shape as Stage 1 (3–4 options per question, no Other in JSON).
- **Product-focused only:** questions and options must stay in the **same domain** as the requirement — **no** generic IT templates, **no** meta questions about interpretation or resolution strategy.

Return ONLY valid JSON: {{"questions": [ ... ]}}"""


CLARIFICATION_CLARITY_EVAL_PROMPT = """After initial clarification answers, is **more** clarification needed before writing formal business rules?

**Requirement:**
{raw_requirement_text}

**Extracted summary:**
{understood_summary}

**Clarification answers (summary):**
{clarification_summary}

Return ONLY JSON: {{"need_followup": true or false, "reason": "one short phrase"}}

**need_followup** = **true** only if important ambiguity or missing decisions would **change** implementation. **false** if a BA could write concrete rules now."""


def _parse_question_lines(text: str) -> list[tuple[str, str]]:
    """Parse LLM output for lines 'Q (category): question text'. Returns list of (category_snake_case, question_text)."""
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.upper().startswith("Q "):
            continue
        # Match: Q (category): question text  or  Q (category): question text
        m = re.match(r"Q\s*\(\s*([^)]+)\s*\)\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            category = m.group(1).strip().replace(" ", "_").replace("-", "_").lower()
            question = m.group(2).strip()
            if category and question:
                results.append((category, question))
    return results


def _extract_json_from_text(text: str) -> dict:
    """Extract a JSON object from LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
    raise ValueError("No valid JSON found in clarification response")


def _dedupe_options_case_insensitive(options: list[str]) -> list[str]:
    """Drop duplicate meanings (case-insensitive), preserve first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for s in options:
        key = s.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _normalize_option_list(raw: object) -> list[str]:
    if isinstance(raw, str):
        raw = [s.strip() for s in raw.split(";") if s.strip()]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for o in raw:
        s = str(o).strip()
        if not s:
            continue
        low = s.lower()
        if low in ("other", "other (specify)", "other — specify", "tbd", "unclear", "it depends", "depends"):
            continue
        # Drop obvious hedging-only options
        if low in ("both", "both apply", "n/a — tbd"):
            continue
        out.append(s)
    out = _dedupe_options_case_insensitive(out)
    # 3–4 options for dropdown
    if len(out) > 4:
        out = out[:4]
    while len(out) < 3:
        for d in DEFAULT_SUGGESTED_OPTIONS:
            if d.lower() not in {x.lower() for x in out}:
                out.append(d)
            if len(out) >= 3:
                break
    return out[:4]


@dataclass
class ClarificationQuestion:
    """One clarification question with a small set of suggested answers (UI dropdown + Other)."""

    category: str
    question: str
    options: list[str]  # 3–4 presets; "Other" is added in the UI only


def _questions_from_llm_json(data: dict, *, max_questions: int) -> list[ClarificationQuestion]:
    arr = data.get("questions", [])
    if not isinstance(arr, list):
        return []
    seen: set[str] = set()
    out: list[ClarificationQuestion] = []
    cap = max(1, min(max_questions, CLARIFICATION_MAX_TOTAL_QUESTIONS))
    for item in arr:
        if len(out) >= cap:
            break
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category", "")).strip().replace(" ", "_").replace("-", "_").lower()
        q = str(item.get("question", "")).strip()
        opts = _normalize_option_list(item.get("options", []))
        if not opts:
            opts = list(DEFAULT_SUGGESTED_OPTIONS)
        if not cat or not q:
            continue
        if cat in seen:
            continue
        seen.add(cat)
        out.append(ClarificationQuestion(category=cat, question=q, options=opts))
    return out


def _run_clarification_llm_json(prompt: str, agent_label: str) -> dict | None:
    with agent_log(agent_label):
        content = call_llm(prompt)
        if not content:
            return None
        try:
            return _extract_json_from_text(content)
        except (ValueError, json.JSONDecodeError, TypeError):
            return None


def generate_stage1_questions(
    understood: UnderstoodRequirement,
    raw_requirement_text: str,
    clarified: dict | None = None,
    open_items: list[str] | None = None,
) -> list[ClarificationQuestion]:
    """
    Stage 1: ``CLARIFICATION_STAGE1_TARGET_MIN``–``MAX`` questions
    (business rules, user flow, constraints, assumptions where relevant).
    """
    clarified = clarified or {}
    already = json.dumps(clarified, indent=0) if clarified else "None yet."
    oi_block = format_open_items_for_clarification_prompt(open_items)
    prompt = CLARIFICATION_STAGE1_PROMPT.format(
        raw_requirement_text=raw_requirement_text or "(none)",
        open_items_block=oi_block,
        action=understood.action or "",
        domain=understood.domain or "",
        actor=understood.actor or "",
        secondary_actor=(understood.secondary_actor or "").strip() or "(none)",
        impact=json.dumps(understood.impact) if understood.impact else "[]",
        already_clarified=already,
        stage1_min=CLARIFICATION_STAGE1_TARGET_MIN,
        stage1_max=CLARIFICATION_STAGE1_TARGET_MAX,
    )
    cap = CLARIFICATION_STAGE1_TARGET_MAX
    with agent_log("Clarification Stage 1"):
        content = call_llm(prompt)
    if not content:
        return []
    try:
        data = _extract_json_from_text(content)
        parsed = _questions_from_llm_json(data, max_questions=cap)
        if parsed:
            return parsed
    except (ValueError, json.JSONDecodeError, TypeError):
        pass
    legacy = _parse_question_lines(content)
    seen = set()
    out: list[ClarificationQuestion] = []
    for cat, q in legacy:
        if len(out) >= cap:
            break
        if cat not in seen and q and q.strip():
            seen.add(cat)
            out.append(
                ClarificationQuestion(
                    category=cat,
                    question=q.strip(),
                    options=list(DEFAULT_SUGGESTED_OPTIONS),
                )
            )
    return out


def _stage2_signals_require_questions(
    validation_issues: list[tuple[str, str]],
    missing_fields: list[str],
) -> bool:
    """True when Stage 2 must not be empty (consistency flags or required-field gaps)."""
    return bool(validation_issues) or bool(missing_fields)


def _fallback_stage2_questions(
    validation_issues: list[tuple[str, str]],
    missing_fields: list[str],
    forbidden_lower: set[str],
    max_count: int,
) -> list[ClarificationQuestion]:
    """
    Last resort when the LLM returns no usable Stage 2 questions but logic said follow-up
    was required—avoids a dead-end UI/CLI.
    """
    if max_count <= 0:
        return []
    base_cat = "stage2_followup_resolution"
    cat = base_cat
    suffix = 0
    while cat.lower() in forbidden_lower:
        suffix += 1
        cat = f"{base_cat}_{suffix}"
    parts: list[str] = []
    for sev, msg in validation_issues[:3]:
        parts.append(f"[{sev}] {msg}")
    if missing_fields:
        parts.append("Missing or generic fields: " + ", ".join(missing_fields))
    summary = " ".join(parts) if parts else "Additional alignment is needed before refinement."
    # Keep question readable; Streamlit/CLI show full text
    qtext = (
        "Automated checks flagged issues in your Stage 1 answers. "
        "Which approach should the formal requirement take? "
        f"Context: {summary}"[:900]
    )
    return [
        ClarificationQuestion(
            category=cat,
            question=qtext,
            options=[
                "Use the strictest interpretation so answers do not contradict",
                "Treat conflicting lines as applying in different contexts (e.g. timing windows)",
                "Prioritize operational / delivery behavior over billing wording",
                "Prioritize billing / payment wording over delivery wording",
            ],
        )
    ]


def generate_stage2_questions(
    understood: UnderstoodRequirement,
    raw_requirement_text: str,
    clarified_flat: dict,
    stage1_questions: list[ClarificationQuestion],
    validation_issues: list[tuple[str, str]],
    missing_fields: list[str],
    *,
    max_additional: int,
    open_items: list[str] | None = None,
) -> list[ClarificationQuestion]:
    """Stage 2: targeted follow-ups; total questions never exceed ``CLARIFICATION_MAX_TOTAL_QUESTIONS``."""
    max_additional = max(0, min(max_additional, CLARIFICATION_MAX_TOTAL_QUESTIONS))
    if max_additional <= 0:
        return []
    forbidden = [q.category for q in stage1_questions]
    forbidden_set = {c.lower() for c in forbidden}
    stage1_lines = "\n".join(f"- [{q.category}] {q.question}" for q in stage1_questions) or "(none)"
    if validation_issues:
        vlines = "\n".join(f"- [{sev}] {msg}" for sev, msg in validation_issues)
    else:
        vlines = "(none — no consistency flags)"
    if missing_fields:
        gap_line = "Missing or generic structured fields after Stage 1: " + ", ".join(missing_fields)
    else:
        gap_line = "(no required-field gaps flagged)"
    must_have_questions = _stage2_signals_require_questions(validation_issues, missing_fields)
    oi_block = format_open_items_for_clarification_prompt(open_items)
    prompt = CLARIFICATION_STAGE2_PROMPT.format(
        raw_requirement_text=raw_requirement_text or "(none)",
        open_items_block=oi_block,
        action=understood.action or "",
        domain=understood.domain or "",
        actor=understood.actor or "",
        secondary_actor=(understood.secondary_actor or "").strip() or "(none)",
        impact=json.dumps(understood.impact) if understood.impact else "[]",
        stage1_summary=stage1_lines,
        answers_summary=json.dumps(clarified_flat, ensure_ascii=False, indent=0) if clarified_flat else "{}",
        validation_and_gaps=vlines + "\n" + gap_line,
        max_additional=max_additional,
        max_total=CLARIFICATION_MAX_TOTAL_QUESTIONS,
        forbidden_categories=", ".join(forbidden) if forbidden else "(none)",
    )
    extra: list[ClarificationQuestion] = []
    data = _run_clarification_llm_json(prompt, "Clarification Stage 2")
    if data:
        raw = _questions_from_llm_json(data, max_questions=max_additional)
        extra = [q for q in raw if q.category.lower() not in forbidden_set]

    if must_have_questions and not extra:
        extra = _fallback_stage2_questions(
            validation_issues, missing_fields, forbidden_set, max_additional
        )
    return extra[:max_additional]


def _llm_clarity_insufficient(
    understood: UnderstoodRequirement,
    raw_requirement_text: str,
    clarification_summary: str,
) -> bool:
    understood_summary = json.dumps(understood.to_dict(), ensure_ascii=False, indent=0)
    prompt = CLARIFICATION_CLARITY_EVAL_PROMPT.format(
        raw_requirement_text=raw_requirement_text or "(none)",
        understood_summary=understood_summary,
        clarification_summary=clarification_summary or "(no clarification text)",
    )
    data = _run_clarification_llm_json(prompt, "Clarification clarity eval")
    if not data:
        return False
    val = data.get("need_followup")
    if isinstance(val, str):
        return val.strip().lower() in ("true", "yes", "1")
    return bool(val)


def clarification_needs_stage2(
    understood: UnderstoodRequirement,
    raw_requirement_text: str,
    clarified: ClarifiedRequirement,
    *,
    validation_has_errors: bool,
    validation_has_warnings: bool,
) -> bool:
    """True when Stage 2 should run: consistency flags, required-field gaps, or LLM sees ambiguity."""
    if validation_has_errors or validation_has_warnings:
        return True
    missing = detect_missing_fields(understood, clarified_to_response_dict(clarified))
    if missing:
        return True
    summary = clarified.to_clarification_context()
    return _llm_clarity_insufficient(understood, raw_requirement_text, summary)


def clarified_to_response_dict(clarified: ClarifiedRequirement) -> dict:
    """Flat answers dict for ``detect_missing_fields``."""
    d = dict(clarified.additional)
    if clarified.timing:
        d["timing"] = clarified.timing
    if clarified.billing_behavior:
        d["billing_behavior"] = clarified.billing_behavior
    return d


def generate_questions_context_aware(
    understood: UnderstoodRequirement,
    raw_requirement_text: str,
    clarified: dict,
    missing_fields_hint: list[str] | None = None,
) -> list[ClarificationQuestion]:
    """Backward-compatible alias: Stage 1 only. ``missing_fields_hint`` is ignored."""
    _ = missing_fields_hint
    return generate_stage1_questions(understood, raw_requirement_text, clarified)


# --- Normalization: free-text ("Other") clarification answers ---

NORMALIZE_CUSTOM_CLARIFICATION_PROMPT = """A user answered a clarification question in free text (they did not pick a preset option). Rewrite their answer as ONE clear business rule sentence.

Requirements:
- Single declarative sentence, same style as a concise policy line in a requirements workshop (parallel to preset dropdown options).
- Remove hedging, filler, vague phrasing, and internal contradictions; keep intent and scope faithful to the user.
- Keep concrete values from the user's answer exactly as stated (themes, product names, labels, numbers, channels); do not replace with vaguer or generic equivalents.
- No bullet points, prefixes like "Rule:", or wrapping quotation marks in your output.

Question: {question}
User's answer: {answer}

Rewritten business rule (one sentence only):"""


def _light_cleanup_free_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def normalize_custom_clarification_answer(raw: str, question_text: str = "") -> str:
    """
    Turn a custom clarification response into one clear business-rule sentence.
    Preset-style answers should be filtered by the caller and not passed here when unchanged.
    """
    raw = _light_cleanup_free_text(raw or "")
    if len(raw) < 2:
        return raw
    q = (question_text or "").strip() or "(clarification context not provided)"
    prompt = NORMALIZE_CUSTOM_CLARIFICATION_PROMPT.format(question=q, answer=raw)
    try:
        out = (call_llm(prompt) or "").strip()
        out = out.split("\n")[0].strip().strip('"').strip("'")
        if len(out) < 3:
            return raw
        return _light_cleanup_free_text(out)
    except Exception:
        return raw


def normalize_responses_with_questions(
    responses: dict[str, str],
    questions: list[ClarificationQuestion],
) -> dict[str, str]:
    """
    For each answer, if it is not exactly one of the question's preset options, normalize via LLM.
    Preset matches are left unchanged (case-insensitive).
    """
    if not questions:
        return dict(responses or {})
    by_cat = {q.category: q for q in questions}
    out: dict[str, str] = {}
    for cat, val in (responses or {}).items():
        v = (val or "").strip()
        if not v:
            out[cat] = ""
            continue
        cq = by_cat.get(cat)
        if cq is None:
            out[cat] = v
            continue
        opts_lower = {o.lower().strip() for o in cq.options if o}
        if v.lower() in opts_lower:
            out[cat] = next(o for o in cq.options if o.lower().strip() == v.lower())
            continue
        out[cat] = normalize_custom_clarification_answer(v, cq.question)
    return out


# --- Step 4: Capture user responses ---


def capture_responses(questions: list[ClarificationQuestion]) -> dict[str, str]:
    """Ask each question via console: numbered options plus Other for free text."""
    responses: dict[str, str] = {}
    for cq in questions:
        print(f"\n  Q ({cq.category}): {cq.question}")
        choices = list(cq.options) + [OTHER_OPTION_LABEL]
        for idx, label in enumerate(choices, start=1):
            print(f"    {idx}. {label}")
        answer = ""
        try:
            raw = input(f"  Choose 1–{len(choices)} (or press Enter to skip): ").strip()
        except EOFError:
            raw = ""
        if not raw:
            responses[cq.category] = ""
            continue
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(choices):
                picked = choices[n - 1]
                if picked == OTHER_OPTION_LABEL:
                    try:
                        answer = input("  Your custom answer: ").strip()
                    except EOFError:
                        answer = ""
                else:
                    answer = picked
        else:
            answer = raw
        responses[cq.category] = answer or ""
    return responses


# --- Step 5: Update structured data ---


def update_clarified_data(clarified: dict, responses: dict) -> dict:
    """Merge user responses into the clarified data dict."""
    out = dict(clarified)
    for k, v in (responses or {}).items():
        if v:
            out[k] = v
    return out


# --- Step 6: Iterative completion ---


@dataclass
class ClarifiedRequirement:
    """
    Requirement after clarification: understood data plus clarified slots
    (timing, billing_behavior, and any extra Q&A).
    """

    understood: UnderstoodRequirement
    timing: str = ""
    billing_behavior: str = ""
    additional: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        add = {
            k: v
            for k, v in self.additional.items()
            if k not in EXCLUDE_FROM_REFINEMENT_AND_EXPORT
        }
        return {
            **self.understood.to_dict(),
            "timing": self.timing,
            "billing_behavior": self.billing_behavior,
            **add,
        }

    def to_clarification_context(self) -> str:
        """Short summary for display."""
        parts = []
        if self.timing:
            parts.append(f"Timing: {self.timing}")
        if self.billing_behavior:
            parts.append(f"Billing: {self.billing_behavior}")
        for k, v in self.additional.items():
            if v and k not in EXCLUDE_FROM_REFINEMENT_AND_EXPORT:
                parts.append(f"{k}: {v}")
        return "; ".join(parts) if parts else ""

    def to_refinement_block(self) -> str:
        """
        Full structured clarification answers for the refinement stage.
        One line per category so refinement can map each to business rules.
        """
        lines = []
        if self.timing:
            lines.append(f"- timing: {self.timing}")
        if self.billing_behavior:
            lines.append(f"- billing_behavior: {self.billing_behavior}")
        for k, v in self.additional.items():
            if v and k not in EXCLUDE_FROM_REFINEMENT_AND_EXPORT:
                lines.append(f"- {k}: {v}")
        if not lines:
            return ""
        return "Clarification answers (user responses — must be reflected in business rules):\n" + "\n".join(lines)

    @classmethod
    def from_answers(
        cls,
        understood: UnderstoodRequirement,
        answers: dict,
        questions: list[ClarificationQuestion] | None = None,
    ) -> "ClarifiedRequirement":
        """
        Build ClarifiedRequirement from understood + category -> answer.
        If ``questions`` is provided, free-text answers (not matching preset options) are normalized to business-rule sentences.
        """
        answers = dict(answers or {})
        if questions:
            answers = normalize_responses_with_questions(answers, questions)
        timing = (answers.get("timing") or "").strip()
        billing_behavior = (answers.get("billing_behavior") or "").strip()
        additional = {
            k: (v or "").strip()
            for k, v in answers.items()
            if k not in ("timing", "billing_behavior") and (v or "").strip()
        }
        return cls(understood=understood, timing=timing, billing_behavior=billing_behavior, additional=additional)


def run_clarification(
    understood: UnderstoodRequirement,
    raw_requirement_text: str = "",
    max_rounds: int = 2,
    open_items: list[str] | None = None,
) -> ClarifiedRequirement:
    """
    Two-stage clarification (CLI): Stage 1 (4–5 questions), then Stage 2 if needed
    (consistency, gaps, clarity eval). Total ≤ ``CLARIFICATION_MAX_TOTAL_QUESTIONS``.
    ``max_rounds`` is ignored (backward compatibility).
    """
    _ = max_rounds
    from stages.clarification_consistency import validate_clarification_answers

    clarified: dict[str, str] = {}
    q1 = generate_stage1_questions(
        understood, raw_requirement_text, {}, open_items=open_items
    )
    if not q1:
        return ClarifiedRequirement(
            understood=understood,
            timing="",
            billing_behavior="",
            additional={},
        )
    print("\n--- Clarification Stage 1 ---")
    responses = capture_responses(q1)
    responses = normalize_responses_with_questions(responses, q1)
    clarified = update_clarified_data(clarified, responses)
    cr1 = ClarifiedRequirement.from_answers(understood, clarified, q1)
    vr1 = validate_clarification_answers(cr1)
    all_questions: list[ClarificationQuestion] = list(q1)

    cap2 = CLARIFICATION_MAX_TOTAL_QUESTIONS - len(q1)
    need2 = clarification_needs_stage2(
        understood,
        raw_requirement_text,
        cr1,
        validation_has_errors=vr1.has_errors,
        validation_has_warnings=vr1.has_warnings,
    )
    missing = detect_missing_fields(understood, clarified)
    if need2 and cap2 > 0:
        q2 = generate_stage2_questions(
            understood,
            raw_requirement_text,
            clarified,
            q1,
            list(vr1.issues),
            missing,
            max_additional=cap2,
            open_items=open_items,
        )
        if q2:
            print("\n--- Clarification Stage 2 (follow-up) ---")
            if vr1.issues:
                print("  Notes from consistency check (address in follow-up):")
                for sev, msg in vr1.issues:
                    print(f"    [{sev.upper()}] {msg}")
            r2 = capture_responses(q2)
            r2 = normalize_responses_with_questions(r2, q2)
            clarified = update_clarified_data(clarified, r2)
            all_questions.extend(q2)

    return ClarifiedRequirement.from_answers(understood, clarified, all_questions)
