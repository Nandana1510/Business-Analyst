"""
Stage 3: Requirement Refinement

Transform structured (understood) requirement and clarification into a formal definition:
feature name, description, and business rules with traceability metadata (conservative).
"""

import json
import re
from dataclasses import dataclass, field

from stages.requirement_understanding import UnderstoodRequirement, call_llm
from stages.pipeline_logging import agent_log
from stages.requirement_intake import normalize_requirement_level


def _extract_json_from_text(text: str) -> dict:
    """Extract a JSON object from LLM response (direct, code block, or balanced braces)."""
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
    raise ValueError("No valid JSON found in response")


@dataclass
class BusinessRule:
    """Business rule text with traceability to requirement vs clarification (source_id = category slug, etc.)."""

    text: str
    source: str  # "requirement" | "clarification"
    source_id: str  # e.g. clarification category snake_case, or understanding:action, understanding:impact:0

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "source_id": self.source_id}

    def with_rule_id(self, index_1_based: int) -> dict:
        d = self.to_dict()
        d["rule_id"] = f"BR{index_1_based}"
        return d


def _parse_business_rules_from_llm(rules_raw: object) -> list[BusinessRule]:
    if isinstance(rules_raw, str):
        rules_raw = [s.strip() for s in rules_raw.split("\n") if s.strip()]
    if not isinstance(rules_raw, list):
        return []
    out: list[BusinessRule] = []
    seen_text: set[str] = set()
    for item in rules_raw:
        if isinstance(item, str):
            t = item.strip()
            if not t or t.lower() in seen_text:
                continue
            seen_text.add(t.lower())
            out.append(BusinessRule(text=t, source="requirement", source_id="unspecified"))
        elif isinstance(item, dict):
            t = str(item.get("text", item.get("rule", ""))).strip()
            if not t or t.lower() in seen_text:
                continue
            seen_text.add(t.lower())
            src = str(item.get("source", "requirement")).strip().lower()
            if src not in ("requirement", "clarification"):
                src = "requirement"
            sid = str(item.get("source_id", item.get("source_ref", ""))).strip() or "unspecified"
            out.append(BusinessRule(text=t, source=src, source_id=sid))
        if len(out) >= 8:
            break
    return out


@dataclass
class RefinedRequirement:
    """Formal requirement definition after refinement (includes actor from understanding stage)."""

    feature_name: str
    actor: str
    description: str
    business_rules: list[BusinessRule]
    secondary_actor: str = ""
    requirement_level: str = "feature"  # product | sprint | feature | enhancement | bug (from intake)
    domain: str = ""  # business domain from understanding stage (artifact prompts must stay aligned)
    # Preprocessing: lines routed to clarification/gap (not treated as functional requirements)
    open_items: list[str] = field(default_factory=list)
    # Intake: multi-unit product slice — use one epic + single ``features[]`` row (no nested LLM split).
    collapse_product_feature_decomposition: bool = False

    def to_dict(self) -> dict:
        d = {
            "feature_name": self.feature_name,
            "actor": self.actor,
            "description": self.description,
            "business_rules": [br.with_rule_id(i) for i, br in enumerate(self.business_rules, 1)],
            "requirement_level": self.requirement_level,
            "domain": self.domain,
        }
        if self.secondary_actor:
            d["secondary_actor"] = self.secondary_actor
        if self.collapse_product_feature_decomposition:
            d["collapse_product_feature_decomposition"] = True
        return d

    def traceability_metadata(self) -> dict:
        """Structured traceability for APIs / collapsible UI (not shown in format_output)."""
        return {
            "business_rules": [br.with_rule_id(i) for i, br in enumerate(self.business_rules, 1)],
        }

    def format_output(self) -> str:
        """Clean, readable output: Feature, Actor, Description, Business Rules (numbered) — no trace clutter."""
        lines = [
            f"Feature: {self.feature_name}",
            f"Actor: {self.actor}",
        ]
        if (self.secondary_actor or "").strip():
            lines.append(f"Secondary actor: {self.secondary_actor.strip()}")
        if (self.domain or "").strip():
            lines.append(f"Domain: {self.domain.strip()}")
        lines.extend(
            [
                f"Description: {self.description}",
                "",
                "Business Rules:",
            ]
        )
        for i, rule in enumerate(self.business_rules, 1):
            lines.append(f"  {i}. {rule.text}")
        return "\n".join(lines)


REFINEMENT_PROMPT = """You are a business analyst. Turn the following into a formal requirement definition using **strict, conservative interpretation**.
{intake_hint_block}
{narrative_block}
**Structured requirement (authoritative source):**
- Type: {type}
- Actor (primary): {actor}
- Secondary actor (human touchpoint, if any): {secondary_actor}
- Action: {action}
- Domain: {domain}
- Impact: {impact}
{clarification_block}

{open_items_block}
**Grounding rules (mandatory):**
0. **Assumption-free (mandatory):** **No new business rules.** Every sentence in **feature_name**, **description**, and **business_rules** must restate or combine information that already appears in the structured requirement and/or clarification above. If something is **not** stated there, **do not** add it—even if it would be "normal" in industry practice.
1. **Traceability:** Every business rule MUST be directly traceable to the structured requirement and/or the clarification answers above. Do **not** introduce new policies, constraints, deadlines, SLAs, limits, validations, edge cases, security rules, or conditions that are **not** explicitly stated in that text. Do not use "common sense," industry defaults, or unstated assumptions. When in doubt, omit—do not guess or pad the requirement.
2. **No invention:** Do not infer additional restrictions (e.g. time limits, frequency caps, approval steps, notifications, compliance) unless the user or clarification **explicitly** mentioned them. Rephrasing is allowed; adding substance is not.
3. **Clarification coverage:** Every clarification answer listed above must appear in at least one business rule (merged with related text if needed). Do not drop a stated clarification point.
3b. **Clarification interpretation — preserve modality (mandatory):** Read each clarification line for **what the user actually decided**, not the strongest possible reading. **Do not** convert **conditional**, **context-specific**, **optional**, **preferred**, **example**, or **"when applicable"** answers into **universal must/always/shall** rules unless the answer text was explicitly mandatory (e.g. fixed policy, always-on behavior). Use **may**, **when**, **if**, **where configured**, or **optional** in the rule wording when the clarification left flexibility. **Do not** stack extra constraints onto a clarification answer beyond what it stated. If an answer is **"not applicable"** or **out of scope**, do **not** invent obligations for that topic.
4. **Structured requirement coverage:** The feature name, description, and rules must reflect the stated type, actor, action, domain, and impact only as given—no broader scope than the sources support.
4b. **Impact / integrations are not separate features:** **Impact** list items (impacted systems, external services, platforms) are **integration boundaries and dependencies** for **this** capability—**not** additional backlog features. Keep **one** cohesive **feature_name** anchored on the **primary user-facing action**; reference named systems in **description** or **business_rules** as integration context only, unless the user explicitly scoped **building or owning** a named system as its **own** deliverable.
4c. **Constraints and SLAs are not new features:** Time windows, delivery guarantees, payment rules, and similar **constraints** belong in **business_rules** and/or **description** as testable obligations—**do not** widen scope into multiple **feature_name** candidates or separate "mini-features" per constraint line.
5. **No duplication:** One rule per distinct grounded point. Merge overlapping or redundant phrasing into a single rule.
6. **Concise wording:** One sentence per rule where possible. Plain, testable language. No filler.
6b. **Natural description:** The **description** field must read like a **short stakeholder-facing summary** (active voice, concrete who/what/outcome)—the kind a BA would paste into Confluence or a PRD. Avoid stiff templates, duplicated clauses, or repeating the feature_name verbatim. It should sound **human and clear**, not like a generic spec generator.
7. **Literal fidelity:** When the structured fields or clarifications name concrete specifics (exact labels, themes, colors, product names, amounts, thresholds, versions, integrations, etc.), carry those terms through **verbatim** in feature_name, description, and business_rules. Do not normalize to generic equivalents (e.g. do not replace "black theme" with "dark theme" or "appropriate styling") unless the source text already used only the broader wording.
8. **No unstated NFRs or UX extras:** Do **not** add accessibility, WCAG, color pickers, themes beyond what was stated, internationalization, analytics, SEO, branding, or generic "security hardening" unless the user or clarification **explicitly** asked for them.
9. **No generic filler rules:** Do **not** emit business rules that only restate vague "domain" or "system impact" lists without a **testable** policy grounded in the sources. Every rule must be an implementable check or obligation from the text above.
10. **No meta or process language:** Do **not** phrase rules as **clarification mechanics**, **resolution strategies**, or internal QA (e.g. "prioritize … wording", "strictest interpretation", "resolution logic"). Only **business-facing** obligations a stakeholder would sign off on.

**Domain coherence:** **feature_name**, **description**, and **business_rules** must stay in the **same** business domain as **Domain** and **Action** above—do not introduce unrelated domains.

**Actor consistency:** Use **Actor** as the **primary** role. When **Actor** is **System**, describe what the platform automates; do **not** rewrite the primary actor as **User** unless the sources say the user performs the work. If **Secondary actor** is set, use it only for human steps (notifications, review queues, configuration)—do **not** claim the user continuously performs system-level monitoring or scoring the **System** does.

**Per-rule traceability (mandatory in JSON):**
- For each business rule include **source**: `"requirement"` if it comes only from the structured fields (type, actor, action, domain, impact) above, or `"clarification"` if it reflects (wholly or in part) a clarification answer.
- Include **source_id**:
  - If **source** is `clarification`: use the **exact** clarification category key from the block (the token before the colon on lines like `- timing_and_billing: ...`).
  - If **source** is `requirement`: use a short id such as `understanding:action`, `understanding:domain`, `understanding:type`, or `understanding:impact` (or `understanding:impact:0` if tied to a specific impact list item).

**Count:** Output **only as many** business rules as needed—**maximum 8**.

Return ONLY a valid JSON object with these keys (no other text, no markdown):
- "feature_name": short title grounded in the action/domain (no invented scope); use **title-style** wording a product team would ship
- "secondary_actor": same human role as in the structured source above, or **""** if none—must not invent a secondary actor
- "description": one or two **natural** sentences—who benefits, what they can do or what changes, and why it matters—**only** from the sources; no robotic repetition
- "business_rules": array of objects, each: {{"text": "<rule sentence>", "source": "requirement" or "clarification", "source_id": "<id as above>"}}

JSON only:"""


def refine_requirement(
    understood: UnderstoodRequirement,
    clarification_context: str | None = None,
    intake_feature_label: str | None = None,
    requirement_level: str | None = None,
    open_items: list[str] | None = None,
    gap_focus_block: str | None = None,
    full_requirement_narrative: str | None = None,
    *,
    collapse_product_feature_decomposition: bool = False,
) -> RefinedRequirement:
    """
    Produce a formal requirement from the understood requirement and optional clarification.
    Business rules include traceability metadata; display text stays clean via format_output().

    ``gap_focus_block``: optional free-text block (e.g. user-selected gap lines) appended after
    clarification for a targeted refinement pass.

    ``full_requirement_narrative``: optional full user requirement text (e.g. combined body after
    gap regeneration) for literal terms and scope boundaries alongside structured fields.
    """
    narrative_block = ""
    narr = (full_requirement_narrative or "").strip()
    if narr:
        narrative_block = (
            "\n**Original requirement text (user's words — use for literal terms, labels, and scope "
            "boundaries together with the structured fields below; do not contradict clarification when "
            "both apply):**\n---\n"
            f"{narr}\n---\n"
        )
    clarification_block = ""
    if clarification_context and clarification_context.strip():
        clarification_block = "\n" + clarification_context.strip()
    else:
        clarification_block = ""
    gf = (gap_focus_block or "").strip()
    if gf:
        clarification_block += (
            "\n\n**User-selected gap analysis (prioritize reflecting these concerns where they align "
            "with the structured requirement and clarification above; treat them as **constraints, rules, "
            "or refinements** on the existing capability—**not** as separate products, unrelated epics, or "
            "wholly new features unless a line clearly names a distinct capability. Use **source** "
            "`clarification` and **source_id** `gap_focus:n` for rules driven primarily by gap line n "
            "below):**\n"
            f"{gf}\n"
        )
    oi = [x.strip() for x in (open_items or []) if (x or "").strip()]
    open_items_block = ""
    if oi:
        lines = "\n".join(f"- {it}" for it in oi)
        open_items_block = (
            "\n**Source notes (pending discussion / clarification — not confirmed product scope; "
            "do **not** encode as business rules unless already reflected above):**\n"
            f"{lines}\n"
        )
    hint = (intake_feature_label or "").strip()
    intake_hint_block = ""
    if hint:
        intake_hint_block = (
            f"\n**Upstream feature label (hint only; must match the sources—adjust if inaccurate):** {hint}\n"
        )
    sec_act = (understood.secondary_actor or "").strip()
    prompt = REFINEMENT_PROMPT.format(
        intake_hint_block=intake_hint_block,
        narrative_block=narrative_block,
        type=understood.type,
        actor=understood.actor,
        secondary_actor=sec_act if sec_act else "— none —",
        action=understood.action,
        domain=understood.domain,
        impact=json.dumps(understood.impact),
        clarification_block=clarification_block,
        open_items_block=open_items_block,
    )
    with agent_log("Refinement"):
        content = call_llm(prompt)
        if not content:
            raise ValueError("LLM returned empty response for refinement")
        data = _extract_json_from_text(content)
        feature_name = str(data.get("feature_name", "")).strip() or "Unnamed feature"
        description = str(data.get("description", "")).strip() or ""
        actor = (understood.actor or "").strip()
        if not actor or actor.lower() in ("unknown", "n/a", ""):
            actor = "User"
        # Reject meta-actors like "User builds system" — align with interaction role
        if re.search(r"(?i)user\s+builds?\s+(the\s+)?system", actor):
            actor = "User"
        secondary = str(data.get("secondary_actor", "")).strip()
        if not secondary:
            secondary = (understood.secondary_actor or "").strip()
        business_rules = _parse_business_rules_from_llm(data.get("business_rules", []))
        level = normalize_requirement_level(requirement_level)
        dom = (understood.domain or "").strip()
        return RefinedRequirement(
            feature_name=feature_name,
            actor=actor,
            description=description,
            business_rules=business_rules,
            secondary_actor=secondary,
            requirement_level=level,
            domain=dom,
            open_items=list(oi),
            collapse_product_feature_decomposition=bool(collapse_product_feature_decomposition),
        )
