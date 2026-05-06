"""
Phase 2: Clarification mechanism (two-stage adaptive)

Stage 1: 4–5 base questions (business rules, user flow, constraints, assumptions), optionally
plus one **hierarchy / granularity** question when backlog shape materially affects artifacts (Epic-style
program hierarchy vs flat feature / user stories vs keep intake). Not counted toward the Stage 2 cap.
Stage 2: optional follow-up when consistency, required-field gaps, or clarity eval
say more detail is needed. Total questions capped at ``CLARIFICATION_MAX_TOTAL_QUESTIONS``.

**Hierarchy question gating** (see ``assess_need_for_granularity_clarification``): **never** for **bug**;
**product** when multiple capabilities are detected; **sprint** when multi-capability or (time-box bundle
gray-zone LLM); **enhancement** only with multiple capability areas; **feature** only when multiple
capabilities suggest hierarchy matters—skip obvious tiny single-feature scopes. **2–3 intake slices** still
run the LLM gate (umbrella epic vs separate deliverables). Intake supplementary text still counts
toward multi-capability. Does **not** count toward the Stage 2 cap (see ``merge_stage1_questions_with_optional_granularity``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from stages.requirement_intake import normalize_requirement_level, parse_explicit_capability_labels
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

# Reserved Stage-1 category: optional granularity (not counted against Stage 2 budget).
REQUIREMENT_GRANULARITY_CATEGORY = "requirement_granularity_scope"
GRANULARITY_OPTION_USE_INTAKE = "Use intake classification (automation default)"
GRANULARITY_OPTION_EPIC = "Treat as Epic-level (product-style backlog)"
GRANULARITY_OPTION_FEATURE = "Treat as Feature-level (single capability slice)"
REQUIREMENT_GRANULARITY_QUESTION_TEXT = (
    "Backlog hierarchy: keep the intake default, or structure outputs as a program-level Epic with "
    "grouped capability buckets vs a single feature / flat user stories?"
)

_PRODUCT_LEVEL_INTENT_RES = [
    re.compile(r"\bbuild\s+(?:a|an)\s+system\b", re.I),
    re.compile(r"\bcreate\s+(?:a|an)\s+app\b", re.I),
    re.compile(r"\bdevelop\s+(?:a|an)\s+platform\b", re.I),
    re.compile(r"\bbuild\s+(?:a|an)\s+platform\b", re.I),
    re.compile(r"\bdevelop\s+(?:a|an)\s+application\b", re.I),
    re.compile(r"\bbuild\s+(?:a|an)\s+application\b", re.I),
    re.compile(r"\bdevelop\s+(?:a|an)\s+app\b", re.I),
    re.compile(r"\bcreate\s+(?:a|an)\s+platform\b", re.I),
]


_TIMEBOX_OR_BUNDLE_LANG_RES = [
    re.compile(r"\bthis\s+sprint\b", re.I),
    re.compile(r"\bnext\s+sprint\b", re.I),
    re.compile(r"\blast\s+sprint\b", re.I),
    re.compile(r"\bthis\s+release\b", re.I),
    re.compile(r"\bnext\s+release\b", re.I),
    re.compile(r"\bthis\s+iteration\b", re.I),
    re.compile(r"\bnext\s+iteration\b", re.I),
    re.compile(r"\btime[- ]?box", re.I),
]


def detect_timeboxed_bundle_language(raw_requirement_text: str) -> bool:
    """True when text reads like a sprint/release bundle (several items in one time box)."""
    t = (raw_requirement_text or "").strip()
    if not t:
        return False
    return any(rx.search(t) for rx in _TIMEBOX_OR_BUNDLE_LANG_RES)


def detect_product_level_intent(raw_requirement_text: str) -> bool:
    """
    True when wording suggests a program / product build framing (not only a single tweak).
    Used with 2–3 intake slices to re-open the granularity gate despite confident intake labels.
    """
    t = (raw_requirement_text or "").strip()
    if not t:
        return False
    return any(rx.search(t) for rx in _PRODUCT_LEVEL_INTENT_RES)


def _hierarchy_combined_text(
    raw_requirement_text: str,
    product_intent_source_text: str | None,
    intake_supplementary_constraints: str | None,
) -> str:
    """Merge sources for multi-capability detection (include intake bucket lists)."""
    parts: list[str] = []
    for blob in (product_intent_source_text, raw_requirement_text, intake_supplementary_constraints):
        s = (blob or "").strip()
        if s and s not in parts:
            parts.append(s)
    return "\n\n".join(parts)


def _intake_bucket_bullet_count(text: str) -> int:
    """Count `-` bucket lines (e.g. named capability areas from product intake supplementary lists)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip().startswith("- ")]
    if len(lines) < 2:
        return 0
    if not all(len(ln) < 160 for ln in lines):
        return 0
    return len(lines)


def _prose_multi_capability_hint(text: str) -> bool:
    """Heuristic when strict comma-list parsing misses clear parallel capabilities."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    if " with " in low:
        _, _, tail = t.partition(" with ")
        tail = tail.strip()
        if tail.count(",") >= 2:
            return True
        if tail.count(",") >= 1 and re.search(r"\band\b", low):
            return True
    if re.search(r"\b(including|such\s+as|e\.g\.|eg\.)\b", low) and (
        "," in t or re.search(r"\band\b", low)
    ):
        return True
    return False


def hierarchy_multi_capability_detected(combined_text: str) -> bool:
    """
    True when text or supplementary material suggests **two or more** business capabilities
    (explicit lists, ``with`` tails, ``including``, or intake bucket bullets).
    """
    t = (combined_text or "").strip()
    if not t:
        return False
    labs = parse_explicit_capability_labels(t)
    if labs and len(labs) >= 2:
        return True
    if _intake_bucket_bullet_count(t) >= 2:
        return True
    if _prose_multi_capability_hint(t):
        return True
    return False


_TRIVIAL_ISOLATED_MAX_WORDS = 14
_TRIVIAL_ISOLATED_MAX_CHARS = 200


def _obviously_isolated_tiny_feature_scope(text: str) -> bool:
    """Very small scope with no enumeration — Epic vs stories choice would not change structure meaningfully."""
    t = " ".join((text or "").split())
    if not t:
        return False
    if len(t) > _TRIVIAL_ISOLATED_MAX_CHARS:
        return False
    if len(t.split()) > _TRIVIAL_ISOLATED_MAX_WORDS:
        return False
    if any(c in t for c in (",", ";", "•", "·")):
        return False
    if parse_explicit_capability_labels(t):
        return False
    low = t.lower()
    if " with " in low and ("," in t or " and " in low):
        return False
    return True


REQUIREMENT_GRANULARITY_GATE_PROMPT = """You decide whether the UI should show **one optional backlog hierarchy control** (program Epic + grouped buckets vs single feature / flat user stories vs keep intake default).

**Pre-classified intake level:** {intake_level}
**Preprocessing unit count** (intake slices this run): **{intake_unit_total}**
**Multi-capability signals** (comma / with / including / intake buckets): **{multi_cap_detected}**
**Product-level intent (broad build language):** {product_intent_detected}
**Time-box / sprint-bundle language:** {timeboxed_detected}

**Original requirement text (this slice):**
---
{raw_requirement_text}
---

**Structured extraction (JSON):**
{understood_json}

Return ONLY valid JSON (no markdown):
{{
  "needs_granularity_question": true or false,
  "reason": "one short phrase"
}}

**Authoritative rules (follow strictly):**
- **bug**: **needs_granularity_question** MUST be **false** (caller should not invoke this for bug — if you see bug, answer **false**).
- **Single slice** (`intake_unit_total` == 1): Set **true** when backlog hierarchy choice **materially** changes artifacts **and**:
  - Intake level is **product** AND **multi-capability signals** is **yes** — lean **true** unless scope is unmistakably one capability.
  - Intake level is **sprint** AND (**multi-capability signals** is **yes** OR time-box language suggests several deliverables whose grouping is unclear).
  - Intake level is **enhancement** AND **multi-capability signals** is **yes** (several major areas).
  - Intake level is **feature** AND **multi-capability signals** is **yes** (suggests mis-scope or parallel capabilities worth separating).
- Set **false** for **single slice** when scope is **obviously** one tiny isolated action (reset password, single button, one validation) **and** multi-capability is **no**.
- **2–3 slices**: Often ambiguous whether to backlog as **one umbrella Epic** vs **separate feature-level** work — lean **true** unless every slice is clearly an unrelated product **or** the text explicitly fixes backlog shape.
- If genuinely unsure on **single slice**, prefer **false** — **except** **product** + multi-capability signals **yes**, where you should prefer **true**.

JSON only:"""

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


def format_prior_clarification_log_for_prompt(log: list[dict[str, Any]] | None) -> str:
    """
    Turn ``clarification_log`` entries into a block for Stage 1 prompts (gap regeneration / evolution).
    """
    if not log:
        return ""
    lines: list[str] = []
    for rnd in log:
        if not isinstance(rnd, dict):
            continue
        stage = rnd.get("stage")
        fi = rnd.get("feature_index")
        ft = rnd.get("feature_total")
        fl = (rnd.get("feature_label") or "").strip()
        note = (rnd.get("note") or "").strip()
        parts: list[str] = [f"**Stage {stage}"]
        if fi is not None and ft is not None:
            parts.append(f"feature {fi} of {ft}")
        if fl:
            parts.append(f"“{fl}”")
        header = " — ".join(parts) + "**"
        if note:
            header += f" _({note})_"
        lines.append(header)
        items = rnd.get("items")
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                cat = (it.get("category") or "").strip()
                q = (it.get("question") or "").strip()
                ans = (it.get("answer") or "").strip()
                if cat or q or ans:
                    lines.append(f"  - `{cat}` **Q:** {q}")
                    lines.append(f"    **A:** {ans}")
        lines.append("")
    return "\n".join(lines).strip()


def _stage1_prior_clarification_block(formatted_log: str) -> str:
    """Wrap prior-round Q&A for Stage 1 prompts (gap regeneration / refinement loops)."""
    t = (formatted_log or "").strip()
    if not t:
        return ""
    return (
        "\n**Prior clarification rounds (already answered — do not repeat questions whose intent is "
        "fully settled unless new scope below contradicts or reopens them; prioritize unresolved, vague, "
        "or newly introduced topics):**\n"
        f"{t}\n\n"
    )


def _stage1_gap_derived_block(gap_text: str) -> str:
    """Optional emphasis block for gap-derived scope (may overlap full requirement text — still use for focus)."""
    t = (gap_text or "").strip()
    if not t:
        return ""
    return (
        "\n**Scope emphasized from selected gaps / supplementary constraints (treat as part of the requirement; "
        "ask targeted questions for new risks — e.g. retries, limits, security, edge cases — where relevant):**\n"
        f"{t}\n\n"
    )


CLARIFICATION_STAGE1_PROMPT = """You are a business analyst. **Stage 1 — initial clarification:** generate a **base set** of clarification questions with SHORT suggested answers for dropdowns.

**Original requirement (user's words):**
{raw_requirement_text}
{open_items_block}
{prior_clarification_rounds_block}{gap_derived_scope_block}
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

**Adaptive clarification:** When **Prior clarification rounds** or **Scope emphasized from gaps** appear above, the requirement may have **evolved**. Base questions on the **full** original requirement text **and** any gap emphasis. Add questions that cover **new** concerns introduced there; **do not** near-duplicate prior questions when those topics already have clear, specific answers—unless new scope makes them incomplete.

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


def build_requirement_granularity_clarification_question() -> ClarificationQuestion:
    """Fixed Stage-1 question; option strings must match ``effective_requirement_level_for_artifacts``."""
    return ClarificationQuestion(
        category=REQUIREMENT_GRANULARITY_CATEGORY,
        question=REQUIREMENT_GRANULARITY_QUESTION_TEXT,
        options=[
            GRANULARITY_OPTION_USE_INTAKE,
            GRANULARITY_OPTION_EPIC,
            GRANULARITY_OPTION_FEATURE,
        ],
    )


def _intake_level_for_granularity_gate(
    intake_level: str | None,
    understood: UnderstoodRequirement,
) -> str:
    """Prefer real intake; fall back to coarse mapping from understanding ``type`` (CLI / tests)."""
    if (intake_level or "").strip():
        return normalize_requirement_level(intake_level)
    t = (understood.type or "").lower()
    if "bug" in t:
        return "bug"
    if "product" in t or "platform" in t:
        return "product"
    if "sprint" in t:
        return "sprint"
    if "enhancement" in t:
        return "enhancement"
    return "feature"


def _granularity_gate_llm(
    understood: UnderstoodRequirement,
    raw_requirement_text: str,
    *,
    intake_level: str,
    intake_unit_total: int,
    multi_cap: bool,
    product_intent: bool,
    timeboxed: bool,
    fallback_if_llm_fails: bool,
) -> bool:
    prompt = REQUIREMENT_GRANULARITY_GATE_PROMPT.format(
        intake_level=intake_level,
        intake_unit_total=intake_unit_total,
        raw_requirement_text=(raw_requirement_text or "").strip() or "(none)",
        understood_json=json.dumps(understood.to_dict(), ensure_ascii=False, indent=0),
        product_intent_detected="yes" if product_intent else "no",
        timeboxed_detected="yes" if timeboxed else "no",
        multi_cap_detected="yes" if multi_cap else "no",
    )
    data = _run_clarification_llm_json(prompt, "Requirement granularity gate")
    if not data:
        return fallback_if_llm_fails
    v = data.get("needs_granularity_question", data.get("needs_granularity"))
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def assess_need_for_granularity_clarification(
    understood: UnderstoodRequirement,
    raw_requirement_text: str,
    intake_level: str | None,
    *,
    intake_unit_total: int = 1,
    product_intent_source_text: str | None = None,
    intake_supplementary_constraints: str | None = None,
) -> bool:
    """
    When to show the optional hierarchy question (Epic / grouped buckets vs flat feature / intake default).

    **Bug:** never. **≥4** slices: never (separate pipeline units). **2–3** slices: LLM gate with
    fallback **true**. **Single slice:** **product** iff multi-capability detected; **sprint** if
    multi-capability **or** (time-box gray zone via LLM); **enhancement** iff multi-capability;
    **feature** iff multi-capability, skipping obvious tiny isolated scopes without enumeration.
    Intake supplementary text (including focus hints on split units) counts toward multi-capability.
    """
    lvl = _intake_level_for_granularity_gate(intake_level, understood)
    if lvl == "bug":
        return False
    n = max(1, int(intake_unit_total or 1))
    if n >= 4:
        return False

    combined = _hierarchy_combined_text(
        raw_requirement_text or "",
        product_intent_source_text,
        intake_supplementary_constraints,
    )
    pi_blob = (product_intent_source_text or "").strip() or (raw_requirement_text or "").strip()
    product_intent = detect_product_level_intent(pi_blob) or detect_product_level_intent(combined)
    timeboxed = detect_timeboxed_bundle_language(pi_blob) or detect_timeboxed_bundle_language(
        combined
    )
    multi_cap = hierarchy_multi_capability_detected(combined)

    if n in (2, 3):
        return _granularity_gate_llm(
            understood,
            raw_requirement_text or "",
            intake_level=lvl,
            intake_unit_total=n,
            multi_cap=multi_cap,
            product_intent=product_intent,
            timeboxed=timeboxed,
            fallback_if_llm_fails=True,
        )

    if lvl == "product":
        return multi_cap
    if lvl == "enhancement":
        if not multi_cap and _obviously_isolated_tiny_feature_scope(combined):
            return False
        return multi_cap
    if lvl == "feature":
        if not multi_cap and _obviously_isolated_tiny_feature_scope(combined):
            return False
        return multi_cap
    if lvl == "sprint":
        if multi_cap:
            return True
        if timeboxed:
            return _granularity_gate_llm(
                understood,
                raw_requirement_text or "",
                intake_level=lvl,
                intake_unit_total=n,
                multi_cap=False,
                product_intent=product_intent,
                timeboxed=True,
                fallback_if_llm_fails=True,
            )
        return False

    return _granularity_gate_llm(
        understood,
        raw_requirement_text or "",
        intake_level=lvl,
        intake_unit_total=n,
        multi_cap=multi_cap,
        product_intent=product_intent,
        timeboxed=timeboxed,
        fallback_if_llm_fails=False,
    )


def merge_stage1_questions_with_optional_granularity(
    core_questions: list[ClarificationQuestion],
    understood: UnderstoodRequirement,
    raw_requirement_text: str,
    intake_level: str | None,
    intake_unit_total: int = 1,
    *,
    product_intent_source_text: str | None = None,
    intake_supplementary_constraints: str | None = None,
) -> tuple[list[ClarificationQuestion], int]:
    """
    Optionally prepend the granularity question. Drops any LLM duplicate of the reserved category.

    ``intake_unit_total`` should be ``len(analyze_intake(...))`` for this run (``1`` for a single slice).
    ``product_intent_source_text`` optional full user document for phrase detection (multi-feature
    slices often omit program-level wording).
    ``intake_supplementary_constraints`` optional intake-only bucket list — included in hierarchy detection.

    Returns ``(questions_for_ui, budget_exclusions)`` where ``budget_exclusions`` is ``1`` if the
    granularity question was added — that many slots are **excluded** from
    ``CLARIFICATION_MAX_TOTAL_QUESTIONS`` when computing Stage 2 headroom.
    """
    core_clean = [
        q
        for q in core_questions
        if q.category.strip().lower() != REQUIREMENT_GRANULARITY_CATEGORY.lower()
    ]
    if not assess_need_for_granularity_clarification(
        understood,
        raw_requirement_text or "",
        intake_level,
        intake_unit_total=intake_unit_total,
        product_intent_source_text=product_intent_source_text,
        intake_supplementary_constraints=intake_supplementary_constraints,
    ):
        return (list(core_clean), 0)
    gq = build_requirement_granularity_clarification_question()
    return ([gq] + core_clean, 1)


def stage1_effective_budget_count(
    stage1_questions: list[ClarificationQuestion],
    budget_exclusions: int,
) -> int:
    """Slots that count toward ``CLARIFICATION_MAX_TOTAL_QUESTIONS`` (Stage 1 + Stage 2)."""
    excl = max(0, int(budget_exclusions or 0))
    return max(0, len(stage1_questions) - excl)


def effective_requirement_level_for_artifacts(
    clarified: ClarifiedRequirement | None,
    intake_level: str | None,
) -> str:
    """
    Intake level unless the user answered the optional granularity question — then force
    ``product`` (epic path), ``feature`` (stories-only path), or keep intake when they chose the
    automation default.
    """
    base = normalize_requirement_level(intake_level)
    if clarified is None:
        return base
    raw = (clarified.additional.get(REQUIREMENT_GRANULARITY_CATEGORY) or "").strip()
    if not raw:
        return base
    if raw == GRANULARITY_OPTION_USE_INTAKE:
        return base
    if raw == GRANULARITY_OPTION_EPIC:
        return "product"
    if raw == GRANULARITY_OPTION_FEATURE:
        return "feature"
    # Legacy labels (older sessions / exports)
    if "Epic" in raw and "Treat" not in raw:
        return "product"
    if "Specific feature" in raw or ("user stories" in raw and "Treat" not in raw):
        return "feature"
    return base


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
    *,
    prior_clarification_for_prompt: str = "",
    gap_derived_scope_text: str = "",
) -> list[ClarificationQuestion]:
    """
    Stage 1: ``CLARIFICATION_STAGE1_TARGET_MIN``–``MAX`` questions
    (business rules, user flow, constraints, assumptions where relevant).
    """
    clarified = clarified or {}
    already = json.dumps(clarified, indent=0) if clarified else "None yet."
    oi_block = format_open_items_for_clarification_prompt(open_items)
    prior_block = _stage1_prior_clarification_block(prior_clarification_for_prompt)
    gap_block = _stage1_gap_derived_block(gap_derived_scope_text)
    prompt = CLARIFICATION_STAGE1_PROMPT.format(
        raw_requirement_text=raw_requirement_text or "(none)",
        open_items_block=oi_block,
        prior_clarification_rounds_block=prior_block,
        gap_derived_scope_block=gap_block,
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
    core = generate_stage1_questions(
        understood, raw_requirement_text, {}, open_items=open_items
    )
    q1, budget_excl = merge_stage1_questions_with_optional_granularity(
        core, understood, raw_requirement_text, None
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

    cap2 = max(
        0,
        CLARIFICATION_MAX_TOTAL_QUESTIONS - stage1_effective_budget_count(q1, budget_excl),
    )
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
