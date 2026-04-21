"""
Stage 4: Artifact generation (multi-agent)

**Requirement level** (from intake — enforced in code, not by the LLM) selects which agents run:

- **product:** ``features[]`` with user stories per feature; **each** feature also gets its own structured **epic**, **user journey**, and **gap analysis** (no single whole-product epic or global journey/gap).
- **sprint:** ``features[]`` + user stories per feature only — **no** epic, **no** journey, **no** gap analysis.
- **feature** / **enhancement:** flat user stories + acceptance criteria only.
- **bug:** bug report (description, steps, expected vs actual) + optional fix-oriented user story — **no** epic, features, journey, or gap.

Orchestration: ``generate_all_artifacts`` / ``generate_advanced_artifacts`` for the Streamlit UI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from stages.pipeline_logging import agent_log
from stages.requirement_understanding import call_llm
from stages.requirement_refinement import RefinedRequirement
from stages.requirement_intake import normalize_requirement_level

# Acceptance criteria wording in user-story JSON (does not change stories or business rules).
AC_FORMAT_DECLARATIVE = "declarative"
AC_FORMAT_BDD = "bdd"


def normalize_acceptance_criteria_format(value: str | None) -> str:
    """Return ``declarative`` or ``bdd`` (default: declarative)."""
    v = (value or AC_FORMAT_DECLARATIVE).strip().lower()
    return AC_FORMAT_BDD if v == AC_FORMAT_BDD else AC_FORMAT_DECLARATIVE


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
    raise ValueError("No valid JSON found in response")


def _requirement_block(refined: RefinedRequirement) -> str:
    """Serialize refined requirement including numbered business rules for trace links (BR1..BRn)."""
    rules_payload = [
        {
            "rule_id": f"BR{i}",
            "text": br.text,
            "source": br.source,
            "source_id": br.source_id,
        }
        for i, br in enumerate(refined.business_rules, 1)
    ]
    sec = (refined.secondary_actor or "").strip()
    lvl = normalize_requirement_level(getattr(refined, "requirement_level", None))
    dom = (getattr(refined, "domain", None) or "").strip()
    lines = [
        f"Requirement level (intake): {lvl}",
        f"Feature: {refined.feature_name}",
        f"Actor: {refined.actor}",
    ]
    if sec:
        lines.append(f"Secondary actor: {sec}")
    if dom:
        lines.append(f"Business domain (from understanding — keep all artifacts in this domain): {dom}")
    lines.append(f"Description: {refined.description}")
    lines.append(
        f"Business rules (numbered — use rule_id in traces_to): {json.dumps(rules_payload, ensure_ascii=False)}"
    )
    return "\n".join(lines)


@dataclass
class EpicDocument:
    """Structured business epic (produced for **product**-level artifact generation)."""

    title: str
    epic_summary: str = ""
    epic_description: str = ""
    business_problem: str = ""
    goals_and_objectives: list[str] = field(default_factory=list)
    key_capabilities: list[str] = field(default_factory=list)
    business_outcomes: list[str] = field(default_factory=list)
    success_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {
            "epic_title": self.title.strip(),
            "epic_summary": (self.epic_summary or "").strip(),
            "epic_description": (self.epic_description or "").strip(),
            "business_problem": (self.business_problem or "").strip(),
            "goals_and_objectives": list(self.goals_and_objectives),
            "key_capabilities": list(self.key_capabilities),
        }
        if self.business_outcomes:
            d["business_outcomes"] = list(self.business_outcomes)
        if self.success_metrics:
            d["success_metrics"] = list(self.success_metrics)
        return d

    def to_markdown(self) -> str:
        """Single readable block (CLI / legacy)."""
        parts: list[str] = [f"## {self.title.strip()}", ""]
        if self.epic_summary.strip():
            parts.append(self.epic_summary.strip())
            parts.append("")
        if self.epic_description.strip():
            parts.append("### Description")
            parts.append(self.epic_description.strip())
            parts.append("")
        if self.business_problem.strip():
            parts.append("### Business problem")
            parts.append(self.business_problem.strip())
            parts.append("")
        if self.goals_and_objectives:
            parts.append("### Goals & objectives")
            parts.extend(f"- {x}" for x in self.goals_and_objectives)
            parts.append("")
        if self.key_capabilities:
            parts.append("### Key capabilities / scope")
            parts.extend(f"- {x}" for x in self.key_capabilities)
            parts.append("")
        if self.business_outcomes:
            parts.append("### Business outcomes")
            parts.extend(f"- {x}" for x in self.business_outcomes)
            parts.append("")
        if self.success_metrics:
            parts.append("### Success metrics")
            parts.extend(f"- {x}" for x in self.success_metrics)
            parts.append("")
        return "\n".join(parts).strip()

    def alignment_block_for_stories(self) -> str:
        """Compact context so user stories stay under the epic umbrella."""
        lines = [
            f"Epic title: {self.title.strip()}",
            f"Epic summary: {(self.epic_summary or '').strip()}",
        ]
        if self.key_capabilities:
            lines.append("Key capabilities (stories should map to these, not invent scope outside them):")
            lines.extend(f"  - {c}" for c in self.key_capabilities[:12])
        return "\n".join(lines)

    @classmethod
    def from_llm_dict(cls, data: dict) -> EpicDocument | None:
        title = str(
            data.get("epic_title", data.get("title", data.get("Epic Title", "")))
        ).strip()
        if not title:
            return None
        return cls(
            title=title,
            epic_summary=str(data.get("epic_summary", data.get("summary", ""))).strip(),
            epic_description=str(data.get("epic_description", data.get("description", ""))).strip(),
            business_problem=str(data.get("business_problem", "")).strip(),
            goals_and_objectives=_ensure_string_list(data.get("goals_and_objectives", data.get("goals", []))),
            key_capabilities=_ensure_string_list(
                data.get("key_capabilities", data.get("key_capabilities_scope", data.get("scope", [])))
            ),
            business_outcomes=_ensure_string_list(data.get("business_outcomes", [])),
            success_metrics=_ensure_string_list(data.get("success_metrics", [])),
        )

    @classmethod
    def from_legacy_title(cls, title: str) -> EpicDocument | None:
        t = (title or "").strip()
        if not t:
            return None
        return cls(title=t, epic_summary="")


def _ensure_string_list(val) -> list[str]:
    if isinstance(val, str):
        return [s.strip() for s in val.split("\n") if s.strip()]
    if isinstance(val, list):
        return [str(x).strip() for x in val if x and str(x).strip()]
    return []


@dataclass
class AcceptanceCriterion:
    """Acceptance test line with trace link to a business rule (BR:n) and/or user story (US:n)."""

    text: str
    traces_to: str = ""  # e.g. "BR:2", "US:1", or "BR:1;US:1"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "traces_to": self.traces_to if self.traces_to else "unspecified",
        }


def _parse_acceptance_criteria_raw(ac_raw: object) -> list[AcceptanceCriterion]:
    if isinstance(ac_raw, str):
        ac_raw = [s.strip() for s in ac_raw.split("\n") if s.strip()]
    if not isinstance(ac_raw, list):
        return []
    out: list[AcceptanceCriterion] = []
    for x in ac_raw:
        if isinstance(x, str) and x.strip():
            out.append(AcceptanceCriterion(text=x.strip(), traces_to=""))
        elif isinstance(x, dict):
            t = str(x.get("text", "")).strip()
            if not t:
                continue
            tr = str(x.get("traces_to", x.get("trace", x.get("maps_to", "")))).strip()
            out.append(AcceptanceCriterion(text=t, traces_to=tr))
    return out


@dataclass
class UserStoryWithCriteria:
    """One user story with acceptance criteria and trace metadata."""

    story: str
    story_ref: str = ""
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "story": self.story,
            "story_ref": self.story_ref or "unspecified",
            "acceptance_criteria": [c.to_dict() for c in self.acceptance_criteria],
        }

    @classmethod
    def from_any(cls, obj: object) -> UserStoryWithCriteria | None:
        if isinstance(obj, cls):
            return obj if obj.story.strip() else None
        if isinstance(obj, dict):
            story = str(obj.get("story", "")).strip()
            if not story:
                return None
            ref = str(obj.get("story_ref", obj.get("storyRef", ""))).strip()
            ac = _parse_acceptance_criteria_raw(
                obj.get("acceptance_criteria", obj.get("acceptanceCriteria", []))
            )
            return cls(story=story, story_ref=ref, acceptance_criteria=ac)
        if isinstance(obj, str) and obj.strip():
            return cls(story=obj.strip(), acceptance_criteria=[])
        return None

    @classmethod
    def parse_list(cls, raw: list) -> list[UserStoryWithCriteria]:
        out: list[UserStoryWithCriteria] = []
        for item in raw:
            u = cls.from_any(item)
            if u is not None:
                out.append(u)
        for i, u in enumerate(out, 1):
            if not u.story_ref:
                u.story_ref = f"US{i}"
        return out


@dataclass
class FeatureWithStories:
    """Feature bucket: optional per-feature epic, user stories, journey, and gap analysis (product-level)."""

    feature_name: str
    feature_summary: str = ""
    user_stories: list[UserStoryWithCriteria] = field(default_factory=list)
    epic_document: EpicDocument | None = None
    user_journey: list[str] = field(default_factory=list)
    gap_analysis: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {
            "feature_name": self.feature_name,
            "feature_summary": (self.feature_summary or "").strip(),
            "user_stories": [u.to_dict() for u in self.user_stories],
        }
        if self.epic_document is not None:
            d["epic"] = self.epic_document.to_dict()
        if self.user_journey:
            d["user_journey"] = list(self.user_journey)
        if self.gap_analysis:
            d["gap_analysis"] = list(self.gap_analysis)
        return d

    @classmethod
    def from_llm_dict(cls, obj: object) -> FeatureWithStories | None:
        if not isinstance(obj, dict):
            return None
        fn = str(obj.get("feature_name", obj.get("name", ""))).strip()
        if not fn:
            return None
        fs = str(obj.get("feature_summary", obj.get("summary", ""))).strip()
        raw = obj.get("user_stories", [])
        stories = UserStoryWithCriteria.parse_list(raw) if isinstance(raw, list) else []
        return cls(feature_name=fn, feature_summary=fs, user_stories=stories)

    @classmethod
    def from_dict(cls, obj: object) -> FeatureWithStories | None:
        if not isinstance(obj, dict):
            return None
        fn = str(obj.get("feature_name", "")).strip()
        if not fn:
            return None
        fs = str(obj.get("feature_summary", "")).strip()
        raw = obj.get("user_stories") or []
        stories = UserStoryWithCriteria.parse_list(raw) if isinstance(raw, list) else []
        epic_raw = obj.get("epic")
        epic_doc: EpicDocument | None = None
        if isinstance(epic_raw, dict):
            epic_doc = EpicDocument.from_llm_dict(epic_raw)
        uj = _ensure_string_list(obj.get("user_journey", []))
        ga = _ensure_string_list(obj.get("gap_analysis", []))
        return cls(
            feature_name=fn,
            feature_summary=fs,
            user_stories=stories,
            epic_document=epic_doc,
            user_journey=uj,
            gap_analysis=ga,
        )


def _flatten_stories_from_features(features: list[FeatureWithStories]) -> list[UserStoryWithCriteria]:
    out: list[UserStoryWithCriteria] = []
    for f in features:
        out.extend(f.user_stories)
    return out


def _parse_user_stories_response(data: dict) -> list[UserStoryWithCriteria]:
    raw = data.get("user_stories", [])
    if not isinstance(raw, list):
        return []
    return UserStoryWithCriteria.parse_list(raw)


def _llm_json(agent_name: str, prompt: str) -> dict:
    with agent_log(agent_name):
        content = call_llm(prompt)
        if not content:
            raise ValueError(f"LLM returned empty response ({agent_name})")
        return _extract_json_from_text(content)


# --- Agent prompts (one task each; JSON only) ---

_LITERAL_FIDELITY_BLOCK = """**Literal fidelity (mandatory):** Reuse the requirement's exact wording for every concrete specific—product and feature names, UI labels, visual themes and styling (e.g. keep "black theme" if that is what was stated), colors, channels, currencies, numeric thresholds, dates, versions, regions, and named systems or integrations. Do not substitute synonyms, euphemisms, or vaguer generalizations unless the requirement text itself only used a broad term.

**No invention:** Do not add user stories, journey steps, gaps, accessibility, color pickers, or policies that are **not** supported by the requirement and business rules below.

**Domain alignment (mandatory):** The **business domain** is anchored by the requirement block (**Business domain** line when present, plus description and business rules). **Do NOT** introduce unrelated business domains, industries, or problem spaces. **Maintain the same domain** across **all** outputs: epic, **every** feature grouping, user stories, acceptance criteria, user journey, and gap analysis. Features are **in-domain** sub-capabilities or themes—**not** switches to a different vertical or unrelated product area unless the source text explicitly requires it.

**Acceptance criteria vs business rules:** Acceptance criteria must state **observable, testable outcomes** (what we verify in UAT), not copy-paste the business-rule sentence. **Do not** open with stiff spec filler such as "The system shall…"; use **natural** wording. If a criterion reflects a business rule, phrase it as a **check** the user or QA can perform—not as a restatement of policy wording.

**No pipeline meta in artifacts:** Do **not** include clarification-process language, "resolution logic", "strictest interpretation", or how the model reconciled options. Output **only** customer- and business-facing content.

**Output size (mandatory):** Right-size volume to complexity. **Simple** scope → fewer stories and **fewer** criteria per story; **complex** scope → still cap at roughly **3–5 user stories** for a single feature slice unless the requirement block clearly separates **distinct** goals. **Limit** acceptance criteria per story to **what is needed for test coverage**—typically **2–5** per story; **no** padding or repetition across criteria.

"""

SPRINT_EPIC_GATE_PROMPT = (
    """You decide whether this **sprint-level** requirement has ONE clear, shared **business capability** worthy of a single umbrella epic (not unrelated tickets bundled by time).

Requirement:
{requirement}

Return ONLY JSON: {{"unifying_business_capability": true or false, "rationale": "<= 25 words, plain English>", "suggested_capability_theme": "<2-8 words, Title Case, only if true; else empty string>"}}
- **true** only if work clearly rolls up to one business theme users/stakeholders would name as one program.
- **false** if capabilities are unrelated or only technically in the same sprint.
- **suggested_capability_theme**: short umbrella name when true; **""** when false.
No other keys."""
)


STRUCTURED_EPIC_PROMPT = (
    """You are a principal business analyst. Produce one **structured epic** as a roadmap artifact (not a feature rename).

Requirement:
{requirement}
{theme_hint_block}
"""
    + _LITERAL_FIDELITY_BLOCK
    + """**Epic rules (mandatory):**
- **epic_title**: A **broad business capability** (2–8 words, Title Case). It MUST **not** duplicate or lightly rephrase the **Feature** line in the requirement; step up one level of abstraction (example: feature "Pause subscription" → epic "Subscription Lifecycle Management"). This JSON is **one** epic only—**not** a per-feature epic title.
- **epic_summary**: Exactly 1–2 **short** sentences: what this epic delivers.
- **epic_description**: What the capability does and how it improves the product; concrete; must **not** repeat the summary verbatim.
- **business_problem**: The real-world gap—inefficiency, risk, or limitation in the **current** way of working; wording must differ from epic_description.
- **goals_and_objectives**: JSON array of short bullets—outcomes (efficiency, experience, flexibility, performance) **only** grounded in the requirement.
- **key_capabilities**: JSON array of **grouped** major functional themes **under this single epic** (each line is a scope theme—not a separate epic). The **product-level** hint block below may require **every** feature area to appear here when the whole product is in scope.
- **business_outcomes** (optional): JSON array; expected business improvements—omit key or use [] if not applicable.
- **success_metrics** (optional): JSON array; measurable indicators—omit or [] if not applicable.

Each section: **distinct** content (no repeated sentences across fields). No vague filler. Reasonable inference from thin inputs is allowed; do not invent unrelated scope.

Return ONLY JSON with keys:
{{"epic_title":"...","epic_summary":"...","epic_description":"...","business_problem":"...","goals_and_objectives":["..."],"key_capabilities":["..."],"business_outcomes":[],"success_metrics":[]}}

Use [] for optional arrays you skip. No markdown, no other keys."""
)

# Nested **user_stories** under product/sprint **features** must match standalone User Stories (JSON + rules).
_USER_STORIES_NESTED_SAME_AS_STANDALONE = """**User stories under each feature (mandatory — same artifact as standalone User Stories):** Each object in **user_stories** must use the **same structure** as the feature-level User Stories agent: **story_ref**, **story**, **acceptance_criteria** (array of objects, each **text** + **traces_to**). **Story format (mandatory):** Each **story** MUST be exactly: **As a [human user or stakeholder], I want [action] so that [benefit].** — **no** system-as-actor phrasing, **no** "The system shall …", **no** alternate templates (**not** "In order to …"). **Story actor (mandatory):** The **story** must use a **real person or stakeholder** as the subject of **As a …** (e.g. **User**, **Customer**, **Admin**, support or operations roles)—**never** **As the System**, **As the platform**, or **As a service** as the primary actor. If the structured requirement’s **Actor** is **System**, still phrase the story from the **human who benefits**; put automated behavior in **acceptance criteria**. **Traceability:** **traces_to** = **BR:n** or **US:n** as defined there; **story_ref** = US1, US2, … **restart at US1 within each feature**. Apply the **ac_instructions** block below for declarative vs BDD criterion **text** shape.

**Do NOT** duplicate the same story under two features."""


PRODUCT_HIERARCHY_PROMPT = (
    """You are a senior business analyst. Build a **single backlog hierarchy** for a **product-level** requirement.

Requirement (intake level is **product** — describe the whole product/system scope):
{requirement}

"""
    + _LITERAL_FIDELITY_BLOCK
    + """**Product-level output contract (mandatory — this JSON implements 1–3 only):**
- **ONE Epic** — Exactly **one** top-level **`epic`** object for the entire product/system.
- **Features** — **`features`** array: Feature 1, Feature 2, Feature 3, … (one object per distinct capability area).
- **User stories** — **Only** nested under each feature: **`features[].user_stories`** (each with **acceptance_criteria**). **Do not** add a separate top-level **`user_stories`** key to this JSON.

**Separate pipeline steps (do not include here):**
- **ONE User Journey (global)** — Produced in a **later** LLM call after this response; **one** end-to-end journey across **all** features. **Do not** output **`user_journey`** in this JSON.
- **ONE Gap Analysis (global)** — Produced **last**, after journey; **one** consolidated list for the whole product. **Do not** output **`gap_analysis`** in this JSON.

**Mandatory hierarchy (exactly ONE epic for the whole system — never one epic per feature):**
1. **Epic (single, top-level only)** — Output **exactly ONE** `epic` object for the **entire product/system**. It is the **only** epic in your JSON. **Do NOT** create a separate epic per feature, **do NOT** nest epics under features, and **do NOT** name features as if they were epics.
2. **Features (grouping only)** — Treat **each detected functionality** (each distinct user-facing capability you infer from the text) as **one Feature**. Use **Features only as grouping units**: they bucket related backlog items; they are **not** epics and **not** where product-level problem/goals/scope belong. Each has **feature_name** + one-line **feature_summary**. **Every** feature must stay in the **same business domain** as the requirement (see **Business domain** in the block above)—**do not** label a feature as if it belonged to an unrelated industry or problem space.

**Input coverage (mandatory):** Read the **full** requirement and ensure **every distinct user-facing capability** the text **names or clearly calls out** appears in **`features[]`** (each **feature_name** should reflect source wording when specific—e.g. "room search" stays room-search focused). **Do not** drop a named capability. **Do not** create **separate** features for **implementation choices** or **sub-steps** of the same capability (e.g. in-house couriers vs tracking are part of **one** delivery feature unless the text treats them as unrelated products).

3. **Under each Feature** — For **every** feature, output **user_stories**. Each **story** text must follow **As a [user], I want [action] so that [benefit]** (human actor only—see nested rules). Each story must include **acceptance_criteria** (per **ac_instructions**): **user stories** carry the goals; **acceptance criteria** carry the testable checks (**text** + **traces_to** on each criterion). Stories belong **only** to that feature’s bucket. **Do not** emit system-voice or technical-spec lines as **story** text.

**Epic object keys:** epic_title, epic_summary, epic_description, business_problem, goals_and_objectives (array), key_capabilities (array), business_outcomes (array, optional), success_metrics (array, optional).

**Epic = whole product scope (mandatory):** The **epic** must **represent the complete product** implied by the requirement—end-to-end value, not a single capability slice. **epic_summary** and **epic_description** must read as **full-product** scope (what the product delivers overall), not as one feature’s scope.

**Epic includes every feature (mandatory):** **key_capabilities** MUST list **every** identified **features[].feature_name** (same count; 1:1 alignment—each feature appears once in **key_capabilities**). The epic’s scope **includes all** of those feature areas; nothing in **features** may be missing from **key_capabilities**. No orphan features; no epic capability line that does not map to a feature you output.

**Per-feature object (strict — product vs feature split):** Each item in **features** may use **only** these keys: **feature_name**, **feature_summary**, **user_stories**. **Do NOT** put any of the following **inside** a feature object (they belong **only** on the top-level **epic**): **epic** / **epic_title**, **business_problem**, **goals** / **goals_and_objectives**, **key_capabilities** / **scope** / program-scope lists, **epic_summary**, **epic_description**, **business_outcomes**, **success_metrics**, or any other roadmap/epic/PRD block. Features are **grouping + user stories only**—one line summary at most, not a mini-epic.

"""
    + _USER_STORIES_NESTED_SAME_AS_STANDALONE
    + """
{ac_instructions}

**Quality:** Distinct epic sections (no copy-paste). Epic = product-level narrative; **features[]** = grouping only, each with **user_stories** + per-story **acceptance_criteria**. **Verify** before returning: every capability explicitly mentioned in the requirement is represented in **features[]** (see **Input coverage**).

**User journey (not in this JSON):** User journey is produced **only after** this response (after features exist), in a **separate** LLM call—**do NOT** put **user_journey** inside **features**. Downstream, the journey combines flows across the feature areas into **one** end-to-end path. This response is epic + features + user stories only.

**Gap analysis (not in this JSON):** Gap analysis is generated **last** in the pipeline, **after** epic, features, stories, and user journey—**one** consolidated list for the **entire system**, using the same feature decomposition to find **missing detail across features**, **edge cases**, and **risks**. **Do NOT** put **gap_analysis** inside **features** or per feature.

Return ONLY JSON with this top-level shape (no markdown, no other keys):
{{"epic": {{...}}, "features": [{{"feature_name":"...","feature_summary":"...","user_stories":[...]}}, ...]}}
Each **features** element: **only** the three keys above.
"""
)


SPRINT_FEATURE_HIERARCHY_PROMPT = (
    """You are a senior business analyst. Intake level is **sprint** — decompose into **multiple Features** (functional groupings), each with its own user stories. **Do not** invent an epic here.

Requirement:
{requirement}

{epic_alignment_block}
"""
    + _LITERAL_FIDELITY_BLOCK
    + """**Features (mandatory):** Emit **features** array. **Input coverage (mandatory):** Every distinct capability, flow, or functional area **named or clearly described** in the sprint text must appear as **its own** `features[]` item (e.g. if the text mentions **room search**, include a **room search** feature—**do not** drop or merge it away). Prefer **Literal fidelity** for **feature_name** from the source wording. When the sprint bundles **multiple** capabilities, use **at least** one feature per distinct area—often **2 or more** entries; use **one** feature **only** when the sprint truly describes a **single** cohesive slice with **no** separate named capabilities. All features must remain in the **same business domain** as the requirement (**Business domain** line above)—no unrelated domains.

**Per-feature object (strict):** **feature_name**, **feature_summary**, **user_stories** only—no **epic**, **business_problem**, **goals**, **scope** / **key_capabilities**, or other epic-level fields inside a feature (this response has no top-level epic; still do not duplicate PRD blocks per feature).

"""
    + _USER_STORIES_NESTED_SAME_AS_STANDALONE
    + """
{ac_instructions}

**User journey:** Do **not** include **user_journey** in this JSON or per feature—journeys are produced elsewhere as **one** system-wide list.

**Gap analysis:** Do **not** include **gap_analysis** in this JSON or per feature—gaps are produced elsewhere as **one** consolidated list for the whole scope.

Return ONLY JSON: {{"features": [{{"feature_name":"...","feature_summary":"...","user_stories":[...]}}, ...]}}
Each feature object: **only** those three keys. No other keys."""
)


_AC_INSTRUCTIONS_DECLARATIVE = """**Acceptance criterion format (declarative):** Each criterion's **text** is a **short** line that reads as a **test** or **observation** a tester can mark pass/fail—**not** a verbatim repeat of a business-rule policy line. Prefer **user-visible** outcomes ("After pausing, the user sees…", "Scheduled deliveries are skipped for the pause window") over generic policy restatement. Avoid **"The system shall…"** openers; use direct, concrete language. Optional: **The user can…** / **When… then…** / **…is shown**. One focal check per criterion when possible; **do not** duplicate the same check with different words."""

_AC_INSTRUCTIONS_BDD = """**Acceptance criterion format (BDD — mandatory):** Each criterion's **text** MUST be **exactly three lines** with these prefixes (use a real newline between lines **inside** the JSON string value):
Given <initial context>
When <action>
Then <expected outcome>

- **Given:** Use stated context from the requirement when present. If **missing**, infer a **minimal, realistic** context for this feature (e.g. "the user is on the settings page", "the user has an active session")—domain-appropriate, not generic filler.
- **When:** **One** clear user or system action per criterion; do **not** chain many unrelated actions—split into extra criteria if needed.
- **Then:** **One** testable, observable outcome.
- **Simple language** throughout. **traces_to** on each criterion object follows the same rules as declarative mode.
- Generate acceptance criteria **only** for **that** story's scope—never mix another story's criteria into this story's array."""

USER_STORIES_AGENT_TEMPLATE = (
    """You write user stories and acceptance criteria the way a **senior product owner** would for backlog refinement: clear, testable, and **not** copy-pasted. **Only real user or stakeholder goals** become stories—never domains, impacted systems, or architecture as stories. Each criterion must apply ONLY to its paired story.

**Story format (mandatory — every story):** Each **story** string MUST be **exactly one sentence** in this shape, with **no** extra clauses before **As a** or after the benefit:
**As a [human user or stakeholder], I want [action/capability they need] so that [value or benefit].**
- **[human user or stakeholder]** = a real role (e.g. User, Customer, Admin, merchant, support agent)—**never** System, platform, service, API, backend, or automation as the **As a** subject.
- **Do NOT** use alternate openers (**not** "In order to …", **not** "I need …" without **I want**, **not** system-voice statements like "The system shall …").
- **Do NOT** phrase the **story** as a system requirement or technical spec; system behavior belongs only in **acceptance_criteria**.

**Story actor (mandatory):** The **subject** of every user story (**As a …**) MUST be a **real person or stakeholder**—e.g. **User**, **Customer**, **Admin**, merchant, support agent, operations analyst, or another human role appropriate to the requirement. **NOT allowed as primary story actor:** **System**, **platform**, **service**, **API**, **backend**, or any non-human protagonist (**do not** write **"As the System, I want …"** or **"As the platform …"**). If the structured **Actor** in the requirement is **System**, still express the story from the **human who benefits** (e.g. **As a customer, I want …** so that …) and describe automation, validation, or background processing in **acceptance criteria** (declarative or BDD), not as the story’s **As a** role.

Requirement (includes numbered business rules with rule_id BR1, BR2, … and source metadata):
{requirement}

{epic_alignment_block}
"""
    + _LITERAL_FIDELITY_BLOCK
    + """Return ONLY JSON:
{{"user_stories": [
  {{
    "story_ref": "US1",
    "story": "As a <human stakeholder>, I want <action> so that <benefit>.",
    "acceptance_criteria": [
      {{"text": "<testable criterion>", "traces_to": "BR:1"}},
      {{"text": "<criterion>", "traces_to": "US:1"}}
    ]
  }}
]}}

The **user_stories** array holds **as many entries as are justified** by distinct goals (often one for a small feature, more when flows or actors' outcomes genuinely differ)—**not** a fixed length.

**Valid user stories only — filtering (mandatory):**
- A **valid** story is **only** the single sentence **As a … I want … so that …** with a **human** **As a** role. **Do NOT** output system-based statements as **story** text (e.g. **not** "The system shall …", **not** "The service must …", **not** passive technical specs)—those belong in **acceptance_criteria** under a human-goal story.
- A **valid** story expresses a **real user intention**: answer **"What does the user (or the named secondary human actor) want to achieve?"** with a **valuable outcome**—not a system label, module name, or integration inventory.
- **Do NOT** create user stories for: **abstract "domains" or areas** without a goal (e.g. "the Settings domain", "the subscription module"); **internal system structure** (layers, services, databases, pipelines, "the backend"); **impacted / named systems** listed for traceability (Billing, CRM, Delivery scheduling)—those are **context**, not backlog goals; **technical or architectural work** (APIs, schemas, refactors, infra) unless reframed as a **user- or business-visible** outcome—and even then prefer **one** goal-level story with **technical detail in acceptance criteria**.
- **Do NOT** emit **one story per impacted system** or **one per component**; merge into **fewer** goal-based stories. Automated or system behavior belongs in **acceptance criteria** under a **human**-goal story (what the person needs to observe or rely on), not as a **System**-as-actor story.
- **Where non-user-facing detail belongs:** Keep it in **acceptance criteria** (with **traces_to: BR:n** when it verifies or reflects a stated business rule). Do **not** duplicate the **business rules** list as pseudo-stories—**cover** rules through criteria under the relevant **goal** stories instead.

**Traceability (mandatory):**
- Assign **story_ref** per story: US1, US2, US3, … **sequentially in order** (use **only** as many numbers as you actually output—there is **no** target or minimum number of stories).
- Each acceptance criterion MUST be an object with **text** and **traces_to**:
  - **traces_to** = **BR:n** (n = integer) when the criterion verifies or derives from business rule **BRn** from the requirement above.
  - **traces_to** = **US:n** when the criterion only states an aspect of **that story’s** user goal (same n as this story’s story_ref number, e.g. US1 for the first story).
  - If both apply, use a single primary link (prefer **BR:n** when a rule clearly covers it).

{ac_instructions}

**Story count — adaptive (mandatory):** Think like a **business analyst** deciding backlog shape from the requirement—not from a template.
- **Rough sizing (use judgment):** **Single simple goal** → commonly **1–2** user stories. **Moderate** flows (several steps or conditions) → **2–4** stories. **Complex** scope with clearly separate goals → up to **3–5** stories—**do not** exceed **5** stories for one feature-level requirement unless the requirement block explicitly lists **separate** major capabilities that cannot share one story set. **Never** pad to fill a range; **never** split one goal across many stories.
- **Decide from the requirement:** Before writing, note **distinct user goals**, **clearly different flows** (e.g. setup vs ongoing vs exception path), and **complexity**. Let the **smallest** natural set of **separate goals** drive count—a **simple** feature often has **one** strong story.
- **One story per goal:** Add a **new** story **only** when it expresses a **separate** user intention or outcome. **Do not** split **one** goal across multiple stories for granularity.
- **Avoid over-fragmentation:** Do **not** create standalone stories for **technical constraints**, **internal system rules**, **minor field validations**, or **incidental side effects** (e.g. notifications sent, audit/logging, background status updates). Capture those inside **acceptance criteria** under the story whose user goal they support.
- **Goal-oriented & consolidated:** Each story states **one** clear, meaningful user intention; **combine** behaviors that belong to the **same** goal. Remove **repetitive** or **overlapping** stories—prefer **fewer**, sharper stories that each add **distinct** value.
- **Acceptance criteria own the detail:** Keep user stories **high-level** and **goal-focused**. Put **detailed logic**, **validations**, **edge cases**, and **conditions** in **acceptance criteria** only (testable, traceable)—**never** spin those into extra stories.
- **Coverage without padding:** The set of stories must **fully** cover the feature and business rules (**no gaps**), using the **smallest** natural set—**quality over quantity**.

**Style:**
- **Fixed pattern:** Every **story** uses **only** **As a … I want … so that …** (see **Story format** above). You may still vary **word choice** inside the three brackets—**not** the sentence structure.
- **Real-world tone:** Short, professional, **plain business English**.
- **Logical completeness** over volume; **do not** invent scope beyond the requirement.

Do not output a separate global criteria list.

JSON only."""
)


def _build_user_stories_prompt(
    requirement_block: str,
    ac_format: str,
    epic_alignment_block: str = "",
) -> str:
    fmt = normalize_acceptance_criteria_format(ac_format)
    ac_inst = _AC_INSTRUCTIONS_BDD if fmt == AC_FORMAT_BDD else _AC_INSTRUCTIONS_DECLARATIVE
    align = (epic_alignment_block or "").strip()
    if align:
        align = (
            "**Epic umbrella (product-level — user stories must fall under this scope as sub-capabilities; "
            "do not invent work outside it):**\n"
            + align
        )
    return USER_STORIES_AGENT_TEMPLATE.format(
        requirement=requirement_block,
        epic_alignment_block=align,
        ac_instructions=ac_inst,
    )


USER_JOURNEY_PROMPT = (
    """You map **one** user journey only: ordered steps from entry to success. Labels should read like **plain steps** a BA would write on a whiteboard (verb + object), not repetitive "User then user then".

**Single journey — not per feature (mandatory):** Return **exactly one** JSON array **`user_journey`**. **Do NOT** emit multiple journeys, **do NOT** output one journey per feature or per capability, and **do NOT** split the response into parallel flows for different parts of the system. **One** ordered list per response.

**Product-level — end-to-end UX:** When **Requirement level (intake): product**, produce **one** journey **after** treating the requirement as decomposed into feature areas (if a **Features** block appears below, it lists those areas). The steps MUST **combine flows across features** into a **single end-to-end user experience**: one coherent path from entry to outcome that touches the major capability areas as a unified story—not a separate mini-journey per feature, not siloed flows. When level is **sprint**, one coherent flow for the **entire sprint scope** (combine sprint capabilities when a Features block is present). When level is **feature** or **enhancement**, a flow local to that feature is appropriate.

Requirement:
{requirement}

{post_features_context}
"""
    + _LITERAL_FIDELITY_BLOCK
    + """Return ONLY JSON: {{"user_journey": ["Step 1 label", "Step 2 label", ...]}}
Include **only** as many short step labels as the flow needs for logical completeness (no fixed count). Vary phrasing between steps; avoid copy-paste openers. No other keys."""
)


GAP_ANALYSIS_PROMPT = (
    """You identify gaps only: edge cases, failures, ambiguities, missing detail. Phrase each item as a **clear, specific** question or risk tied to **this** requirement—**not** generic "need more detail" or boilerplate.

**One consolidated gap analysis (mandatory):** Return **exactly one** JSON array **`gap_analysis`**. **Do NOT** emit separate gap lists per feature, per capability, or per epic slice—**one** consolidated list per response.

**Product-level — at the end, across all features:** When **Requirement level (intake): product**, this artifact is the **single final** consolidated gap view (downstream, it runs **after** epic, features, stories, and user journey). When a **feature-areas** block appears below, you must surface **missing details across those features** (interfaces, dependencies, sequencing, data ownership, NFRs that span areas). Explicitly **highlight edge cases** (boundary conditions, failure modes, exception paths) and **risks** (technical, operational, security, compliance, UX) that affect the **whole** product or the **boundaries between** capabilities—not isolated nitpicks for one feature only.

**Whole-system coverage:** When level is **product**, prioritize cross-cutting and integration gaps; when **sprint**, cover the **full sprint scope**; when **feature** / **enhancement**, local gaps are appropriate.

**Quality bar:** Each gap must be **actionable** and **unique**—do **not** repeat the same concern with different wording. Avoid **vague** items ("clarify requirements", "confirm stakeholders"). Tie each line to something **concrete** missing from the sources (data, rules, timing, roles, error handling, integration points).

Requirement:
{requirement}

{open_items_block}
{post_features_context}
"""
    + _LITERAL_FIDELITY_BLOCK
    + """Return ONLY JSON: {{"gap_analysis": ["...", ...]}}
Include **only** genuine gaps implied by the requirement and feature context—**no fixed count**; omit padding and repetitive wording. No other keys."""
)


# --- Agents ---


def _sprint_epic_gate(refined: RefinedRequirement) -> tuple[bool, str]:
    """Returns (passes_gate, suggested_capability_theme from model, may be empty)."""
    prompt = SPRINT_EPIC_GATE_PROMPT.format(requirement=_requirement_block(refined))
    try:
        data = _llm_json("Sprint epic gate", prompt)
    except ValueError:
        return False, ""
    v = data.get("unifying_business_capability", data.get("unifying", False))
    if isinstance(v, str):
        ok = v.strip().lower() in ("true", "yes", "1")
    else:
        ok = bool(v)
    hint = str(data.get("suggested_capability_theme", "")).strip()
    return ok, hint


def generate_structured_epic_if_applicable(refined: RefinedRequirement) -> EpicDocument | None:
    """
    Product level: always a full structured epic.
    Feature / enhancement / bug: no epic.
    Sprint: epic only when the sprint gate finds one unifying business capability.
    """
    level = normalize_requirement_level(getattr(refined, "requirement_level", None))
    if level in ("feature", "enhancement", "bug"):
        return None
    theme_hint_block = ""
    if level == "product":
        theme_hint_block = (
            "\n**Product-level (mandatory):** Generate **exactly ONE** epic for the **entire system/product**—the **complete** "
            "product scope. **Do NOT** generate an epic per feature or multiple epics; features belong in backlog grouping only "
            "(elsewhere in the pipeline), not as separate epics here. This epic must **represent the full product** and "
            "**include all** major capability areas / features you infer from the requirement in **key_capabilities** "
            "(whole-product coverage). Do **not** narrow the epic to a single refined **Feature** line when the text "
            "supports a wider product.\n"
        )
    elif level == "sprint":
        gate_ok, sprint_theme = _sprint_epic_gate(refined)
        if not gate_ok:
            return None
        theme_hint_block = (
            "\n**Sprint note:** Intake level is sprint—name the epic after the **single** umbrella "
            "business program that unifies this bundle (not the sprint label itself).\n"
        )
        if sprint_theme:
            theme_hint_block += (
                f"\n**Working theme (from sprint analysis — refine, do not paste verbatim):** {sprint_theme}\n"
            )
    prompt = STRUCTURED_EPIC_PROMPT.format(
        requirement=_requirement_block(refined),
        theme_hint_block=theme_hint_block,
    )
    data = _llm_json("Structured epic", prompt)
    doc = EpicDocument.from_llm_dict(data)
    return doc


def _build_sprint_feature_hierarchy_prompt(
    requirement_block: str,
    acceptance_criteria_format: str | None,
    epic_alignment_block: str,
) -> str:
    fmt = normalize_acceptance_criteria_format(acceptance_criteria_format)
    ac_inst = _AC_INSTRUCTIONS_BDD if fmt == AC_FORMAT_BDD else _AC_INSTRUCTIONS_DECLARATIVE
    return SPRINT_FEATURE_HIERARCHY_PROMPT.format(
        requirement=requirement_block,
        epic_alignment_block=epic_alignment_block or "",
        ac_instructions=ac_inst,
    )


def generate_product_hierarchy(
    refined: RefinedRequirement,
    acceptance_criteria_format: str | None = None,
) -> tuple[EpicDocument | None, list[FeatureWithStories]]:
    """
    One LLM call: single system-wide epic + multiple features, each with user stories + AC.
    """
    fmt = normalize_acceptance_criteria_format(acceptance_criteria_format)
    ac_inst = _AC_INSTRUCTIONS_BDD if fmt == AC_FORMAT_BDD else _AC_INSTRUCTIONS_DECLARATIVE
    prompt = PRODUCT_HIERARCHY_PROMPT.format(
        requirement=_requirement_block(refined),
        ac_instructions=ac_inst,
    )
    data = _llm_json("Product hierarchy", prompt)
    epic_doc: EpicDocument | None = None
    epic_raw = data.get("epic", {})
    if isinstance(epic_raw, dict):
        epic_doc = EpicDocument.from_llm_dict(epic_raw)
    features: list[FeatureWithStories] = []
    raw_feats = data.get("features", [])
    if isinstance(raw_feats, list):
        for item in raw_feats:
            f = FeatureWithStories.from_llm_dict(item)
            if f is not None:
                features.append(f)
    if not features:
        epic_ctx = epic_doc.alignment_block_for_stories() if epic_doc else None
        stories = generate_user_stories(
            refined,
            acceptance_criteria_format=acceptance_criteria_format,
            epic_context=epic_ctx,
        )
        features = [
            FeatureWithStories(
                feature_name=(refined.feature_name or "Primary capability").strip(),
                feature_summary=(refined.description or "")[:400],
                user_stories=stories,
            )
        ]
    if epic_doc is None and features:
        caps = [f.feature_name for f in features]
        epic_doc = EpicDocument(
            title=f"{(refined.feature_name or 'Product').strip()} — Program",
            epic_summary="Program epic synthesized from identified feature areas.",
            epic_description="",
            business_problem="",
            goals_and_objectives=[],
            key_capabilities=caps or ["Core capability"],
        )
    return epic_doc, features


def generate_sprint_features_and_stories(
    refined: RefinedRequirement,
    acceptance_criteria_format: str | None = None,
    epic_alignment: str | None = None,
) -> list[FeatureWithStories]:
    """Sprint: multiple feature buckets, each with user stories (no epic in this call)."""
    prompt = _build_sprint_feature_hierarchy_prompt(
        _requirement_block(refined),
        acceptance_criteria_format,
        epic_alignment or "",
    )
    data = _llm_json("Sprint feature hierarchy", prompt)
    features: list[FeatureWithStories] = []
    raw = data.get("features", [])
    if isinstance(raw, list):
        for item in raw:
            f = FeatureWithStories.from_llm_dict(item)
            if f is not None:
                features.append(f)
    if not features:
        stories = generate_user_stories(
            refined,
            acceptance_criteria_format=acceptance_criteria_format,
            epic_context=epic_alignment,
        )
        features = [
            FeatureWithStories(
                feature_name=(refined.feature_name or "Sprint scope").strip(),
                feature_summary="",
                user_stories=stories,
            )
        ]
    return features


def _open_items_block_for_gap(refined: RefinedRequirement) -> str:
    items = [x.strip() for x in (getattr(refined, "open_items", None) or []) if (x or "").strip()]
    if not items:
        return ""
    lines = "\n".join(f"- {it}" for it in items)
    return (
        "**Pre-identified open items from source (pending decisions / clarification — validate as gaps, not as features):**\n"
        f"{lines}\n"
    )


def _refined_slice_for_feature(parent: RefinedRequirement, feature: FeatureWithStories) -> RefinedRequirement:
    """Narrow parent requirement to one feature area for per-feature journey/gap/epic agents."""
    desc = (
        f"Feature area: {feature.feature_name}\n"
        f"Summary: {(feature.feature_summary or '').strip()}\n\n"
        f"Parent product context:\n{parent.description}"
    )
    return RefinedRequirement(
        feature_name=feature.feature_name,
        actor=parent.actor,
        description=desc,
        business_rules=list(parent.business_rules),
        secondary_actor=parent.secondary_actor,
        requirement_level="feature",
        domain=parent.domain,
        open_items=list(parent.open_items) if getattr(parent, "open_items", None) else [],
    )


def generate_structured_epic_for_feature(
    refined_parent: RefinedRequirement,
    feature: FeatureWithStories,
) -> EpicDocument | None:
    """Structured epic scoped to a single feature area (not a whole-product rollup)."""
    slice_r = _refined_slice_for_feature(refined_parent, feature)
    theme_hint_block = (
        "\n**Scope:** This epic covers **only** the feature area in the requirement block "
        "(one slice of a larger product). **epic_title** should abstract that area, not the entire product.\n"
    )
    prompt = STRUCTURED_EPIC_PROMPT.format(
        requirement=_requirement_block(slice_r),
        theme_hint_block=theme_hint_block,
    )
    data = _llm_json("Structured epic (feature)", prompt)
    return EpicDocument.from_llm_dict(data)


def generate_product_per_feature_artifacts(
    refined: RefinedRequirement,
    acceptance_criteria_format: str | None = None,
) -> list[FeatureWithStories]:
    """
    Product-level: decompose into features + user stories, then for each feature generate a structured
    epic, user journey, and gap analysis. Top-level ``epic`` / global journey / gap are **not** used.
    """
    features = generate_sprint_features_and_stories(
        refined,
        acceptance_criteria_format,
        epic_alignment=None,
    )
    for f in features:
        f.epic_document = generate_structured_epic_for_feature(refined, f)
        slice_r = _refined_slice_for_feature(refined, f)
        f.user_journey = generate_user_journey(slice_r)
        f.gap_analysis = generate_gap_analysis(slice_r)
    return features


def generate_user_stories(
    refined: RefinedRequirement,
    acceptance_criteria_format: str | None = None,
    epic_context: str | None = None,
) -> list[UserStoryWithCriteria]:
    """
    Agent: user stories with per-story acceptance criteria (structured; one LLM call).
    Story count is **not** fixed—the prompt asks the model to derive it from goals, flows, and complexity.
    ``acceptance_criteria_format``: ``declarative`` (default) or ``bdd`` (Given/When/Then in each criterion text).
    """
    align = (epic_context or "").strip()
    prompt = _build_user_stories_prompt(
        _requirement_block(refined),
        acceptance_criteria_format,
        epic_alignment_block=align,
    )
    data = _llm_json("User Stories", prompt)
    return _parse_user_stories_response(data)


def _post_features_context_for_journey(features: list[FeatureWithStories]) -> str:
    """
    After **product** feature decomposition, pass feature names into the User Journey agent
    so one E2E path can combine flows across features.
    """
    if not features:
        return ""
    lines = [
        "**Context — features already identified (weave into ONE end-to-end journey):** The backlog is grouped into the areas below. Your **single** `user_journey` must **combine** these into **one** ordered path representing **end-to-end user experience** across capabilities—not one subsection per feature.",
        "",
    ]
    for f in features:
        fs = (f.feature_summary or "").strip()
        lines.append(f"- **{f.feature_name}**" + (f" — {fs}" if fs else ""))
    return "\n".join(lines) + "\n"


def _post_features_context_for_gap(features: list[FeatureWithStories]) -> str:
    """
    After feature decomposition, pass feature areas into Gap Analysis so one list can cover
    missing detail across features, plus cross-cutting edge cases and risks.
    """
    if not features:
        return ""
    lines = [
        "**Context — consolidated gap analysis (feature areas below):** This is **one** analysis covering the **whole** product. Identify **missing details across all features** (unclear hand-offs, unspecified interfaces between areas, integration gaps, conflicting assumptions). Explicitly **highlight edge cases** and **risks** (operational, technical, compliance, UX) that span or sit at boundaries between capabilities—not a separate gap list per feature.",
        "",
    ]
    for f in features:
        fs = (f.feature_summary or "").strip()
        lines.append(f"- **{f.feature_name}**" + (f" — {fs}" if fs else ""))
    return "\n".join(lines) + "\n"


def generate_user_journey(
    refined: RefinedRequirement,
    post_features_context: str | None = None,
) -> list[str]:
    """Agent: one ordered journey. For **product**, pass ``post_features_context`` after hierarchy so the path is E2E across features."""
    extra = (post_features_context or "").strip()
    prompt = USER_JOURNEY_PROMPT.format(
        requirement=_requirement_block(refined),
        post_features_context=("\n" + extra + "\n") if extra else "",
    )
    data = _llm_json("User Journey", prompt)
    return _ensure_string_list(data.get("user_journey", []))


def generate_gap_analysis(
    refined: RefinedRequirement,
    post_features_context: str | None = None,
) -> list[str]:
    """Agent: one consolidated gap list. For **product**, pass ``post_features_context`` after decomposition."""
    extra = (post_features_context or "").strip()
    oi = _open_items_block_for_gap(refined).strip()
    oi_block = ("\n" + oi + "\n") if oi else ""
    prompt = GAP_ANALYSIS_PROMPT.format(
        requirement=_requirement_block(refined),
        open_items_block=oi_block,
        post_features_context=("\n" + extra + "\n") if extra else "",
    )
    data = _llm_json("Gap Analysis", prompt)
    return _ensure_string_list(data.get("gap_analysis", []))


BUG_ARTIFACT_PROMPT = (
    """You are a senior QA engineer and business analyst. From the structured requirement below, produce a **bug report** suitable for a backlog or defect tracker.

**Requirement level (intake) is bug** — focus on incorrect behavior, missing behavior, or regression.

Return ONLY valid JSON (no markdown):
{{
  "bug_description": "<concise statement of the defect>",
  "steps_to_reproduce": ["<step 1>", "<step 2>"],
  "expected_behavior": "<what should happen>",
  "actual_behavior": "<what happens instead>",
  "fix_oriented_user_story": "<optional: single sentence As a ... I want ... so that ... if helpful; else empty string>"
}}

Ground every field in the requirement; use **minimal** reasonable inference only for steps if not fully specified.

Requirement:
{requirement}
"""
    + _LITERAL_FIDELITY_BLOCK
)


def generate_bug_artifacts(
    refined: RefinedRequirement,
    acceptance_criteria_format: str | None = None,
) -> tuple[dict, list[UserStoryWithCriteria]]:
    """
    Bug-level artifacts: structured bug report + optional fix-oriented story (no AC unless paired elsewhere).
    ``acceptance_criteria_format`` is reserved for future use.
    """
    _ = acceptance_criteria_format
    prompt = BUG_ARTIFACT_PROMPT.format(requirement=_requirement_block(refined))
    data = _llm_json("Bug artifacts", prompt)
    fix = str(data.get("fix_oriented_user_story", "")).strip()
    report = {
        "bug_description": str(data.get("bug_description", "")).strip(),
        "steps_to_reproduce": _ensure_string_list(data.get("steps_to_reproduce", [])),
        "expected_behavior": str(data.get("expected_behavior", "")).strip(),
        "actual_behavior": str(data.get("actual_behavior", "")).strip(),
        "fix_oriented_user_story": fix,
    }
    stories: list[UserStoryWithCriteria] = []
    if fix:
        stories.append(
            UserStoryWithCriteria(
                story=fix,
                story_ref="US1",
                acceptance_criteria=[],
            )
        )
    return report, stories


def generate_all_artifacts(
    refined: RefinedRequirement,
    acceptance_criteria_format: str | None = None,
) -> dict:
    """
    Orchestrator: run artifact agents and return one structured dict.
    Branching is **only** by ``refined.requirement_level`` (normalized).
    """
    level = normalize_requirement_level(getattr(refined, "requirement_level", None))
    ac_fmt = acceptance_criteria_format

    if level in ("feature", "enhancement"):
        stories = generate_user_stories(refined, ac_fmt, epic_context=None)
        return {
            "epic": None,
            "features": [],
            "user_stories": [s.to_dict() for s in stories],
            "user_journey": [],
            "gap_analysis": [],
            "bug_report": None,
        }

    if level == "bug":
        report, stories = generate_bug_artifacts(refined, ac_fmt)
        return {
            "epic": None,
            "features": [],
            "user_stories": [s.to_dict() for s in stories],
            "user_journey": [],
            "gap_analysis": [],
            "bug_report": report,
        }

    if level == "product":
        features = generate_product_per_feature_artifacts(refined, ac_fmt)
        flat = _flatten_stories_from_features(features)
        return {
            "epic": None,
            "features": [f.to_dict() for f in features],
            "user_stories": [s.to_dict() for s in flat],
            "user_journey": [],
            "gap_analysis": [],
            "bug_report": None,
        }

    if level == "sprint":
        features = generate_sprint_features_and_stories(refined, ac_fmt, epic_alignment=None)
        flat = _flatten_stories_from_features(features)
        return {
            "epic": None,
            "features": [f.to_dict() for f in features],
            "user_stories": [s.to_dict() for s in flat],
            "user_journey": [],
            "gap_analysis": [],
            "bug_report": None,
        }

    stories = generate_user_stories(refined, ac_fmt, epic_context=None)
    return {
        "epic": None,
        "features": [],
        "user_stories": [s.to_dict() for s in stories],
        "user_journey": [],
        "gap_analysis": [],
        "bug_report": None,
    }


@dataclass
class AdvancedArtifacts:
    """Bundle of all advanced artifacts (for UI / convenience)."""

    epic_document: EpicDocument | None = None
    features: list[FeatureWithStories] = field(default_factory=list)
    user_stories: list[UserStoryWithCriteria] = field(default_factory=list)
    user_journey: list[str] = field(default_factory=list)
    gap_analysis: list[str] = field(default_factory=list)
    bug_report: dict | None = None

    @property
    def epic(self) -> str:
        """Markdown epic for backward compatibility; empty when no epic."""
        return self.epic_document.to_markdown() if self.epic_document else ""

    def to_dict(self) -> dict:
        return {
            "epic": self.epic_document.to_dict() if self.epic_document else None,
            "features": [f.to_dict() for f in self.features],
            "user_stories": [s.to_dict() for s in self.user_stories],
            "user_journey": self.user_journey,
            "gap_analysis": self.gap_analysis,
            "bug_report": self.bug_report,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AdvancedArtifacts:
        features: list[FeatureWithStories] = []
        raw_feats = d.get("features")
        if isinstance(raw_feats, list):
            for item in raw_feats:
                f = FeatureWithStories.from_dict(item)
                if f is not None:
                    features.append(f)

        raw_us = d.get("user_stories") or []
        if features:
            user_stories = _flatten_stories_from_features(features)
        elif isinstance(raw_us, list):
            user_stories = UserStoryWithCriteria.parse_list(raw_us)
        else:
            user_stories = []

        raw_epic = d.get("epic")
        epic_document: EpicDocument | None = None
        if isinstance(raw_epic, dict):
            epic_document = EpicDocument.from_llm_dict(raw_epic)
        elif isinstance(raw_epic, str) and raw_epic.strip():
            epic_document = EpicDocument.from_legacy_title(raw_epic.strip())
        br = d.get("bug_report")
        bug_report: dict | None = br if isinstance(br, dict) else None
        return cls(
            epic_document=epic_document,
            features=features,
            user_stories=user_stories,
            user_journey=_ensure_string_list(d.get("user_journey", [])),
            gap_analysis=_ensure_string_list(d.get("gap_analysis", [])),
            bug_report=bug_report,
        )


def generate_advanced_artifacts(
    refined: RefinedRequirement,
    acceptance_criteria_format: str | None = None,
) -> AdvancedArtifacts:
    """
    Same outputs as generate_all_artifacts, as a dataclass (used by Streamlit app).
    """
    return AdvancedArtifacts.from_dict(
        generate_all_artifacts(refined, acceptance_criteria_format=acceptance_criteria_format)
    )


# (internal_mode_key, display_label) — order = Streamlit dropdown order
ARTIFACT_GENERATION_CHOICES: list[tuple[str, str]] = [
    ("all", "All Artifacts"),
    ("epic", "Epic"),
    ("user_stories", "User Stories"),
    ("acceptance_criteria", "Acceptance Criteria"),
    ("user_journey", "User Journey"),
    ("gap_analysis", "Gap Analysis"),
]

ARTIFACT_MODE_BY_LABEL: dict[str, str] = {label: key for key, label in ARTIFACT_GENERATION_CHOICES}


def generate_artifacts_for_mode(
    refined: RefinedRequirement,
    mode: str,
    acceptance_criteria_format: str | None = None,
) -> AdvancedArtifacts:
    """
    Run all artifact agents, or a single agent. Branching uses ``refined.requirement_level`` only.
    User Stories and Acceptance Criteria both use the combined per-story agent (except **bug**).
    """
    m = (mode or "all").strip().lower()
    ac_fmt = acceptance_criteria_format
    lvl = normalize_requirement_level(getattr(refined, "requirement_level", None))

    if m == "all":
        return generate_advanced_artifacts(refined, acceptance_criteria_format=ac_fmt)

    if m == "epic":
        if lvl == "product":
            feats = generate_product_per_feature_artifacts(refined, ac_fmt)
            flat = _flatten_stories_from_features(feats)
            return AdvancedArtifacts(epic_document=None, features=feats, user_stories=flat)
        return AdvancedArtifacts(epic_document=None)

    if m == "user_stories" or m == "acceptance_criteria":
        if lvl == "product":
            feats = generate_product_per_feature_artifacts(refined, ac_fmt)
            flat = _flatten_stories_from_features(feats)
            return AdvancedArtifacts(epic_document=None, features=feats, user_stories=flat)
        if lvl == "sprint":
            feats = generate_sprint_features_and_stories(refined, ac_fmt, epic_alignment=None)
            flat = _flatten_stories_from_features(feats)
            return AdvancedArtifacts(
                epic_document=None,
                features=feats,
                user_stories=flat,
            )
        if lvl == "bug":
            report, stories = generate_bug_artifacts(refined, ac_fmt)
            return AdvancedArtifacts(user_stories=stories, bug_report=report)
        return AdvancedArtifacts(
            user_stories=generate_user_stories(
                refined,
                acceptance_criteria_format=ac_fmt,
                epic_context=None,
            )
        )

    if m == "user_journey":
        if lvl == "product":
            feats = generate_product_per_feature_artifacts(refined, ac_fmt)
            flat = _flatten_stories_from_features(feats)
            return AdvancedArtifacts(epic_document=None, features=feats, user_stories=flat)
        return AdvancedArtifacts(user_journey=[])

    if m == "gap_analysis":
        if lvl == "product":
            feats = generate_product_per_feature_artifacts(refined, ac_fmt)
            flat = _flatten_stories_from_features(feats)
            return AdvancedArtifacts(epic_document=None, features=feats, user_stories=flat)
        return AdvancedArtifacts(gap_analysis=[])

    return generate_advanced_artifacts(refined, acceptance_criteria_format=ac_fmt)
