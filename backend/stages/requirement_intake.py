"""
Requirement intake: classify level (product / sprint / feature / enhancement / bug), normalize text,
and optionally split into intake units. **Product/program** scopes with several named capability areas
stay **one** unit so downstream runs a single pipeline with multi-bucket hierarchy inside artifact generation.

Runs before understanding. Downstream pipeline runs **per intake unit**:
understanding → clarification → refinement → artifacts.

When intake yields **multiple units** at **product** level (explicit capability lists or
``multiple_independent_features``), downstream treats each unit as its **own** program slice:
full clarification per slice, then artifacts with **one epic + one backlog bucket** for that slice
(no second-pass ``features[]`` decomposition inside artifact generation).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from stages.pipeline_logging import agent_log
from stages.requirement_understanding import call_llm

_MAX_UNITS = 5
# Explicit comma/keyword lists may enumerate more slices than the generic LLM multi-cap limit.
_MAX_EXPLICIT_CAPABILITIES = 12
_MIN_UNIT_TEXT = 12
# Segments longer than this are not treated as plain list items (likely prose, not a capability token).
_MAX_WORDS_PER_EXPLICIT_SEGMENT = 4
_STOPWORD_CAPABILITIES = frozenset(
    {"yes", "no", "maybe", "tbd", "n/a", "na", "none", "todo", "ok", "okay"}
)


@dataclass
class NormalizedRequirementUnit:
    """One normalized slice of input for the unified pipeline."""

    text: str
    feature_name: str = ""
    requirement_level: str = "feature"  # product | sprint | feature | enhancement | bug
    # Intake-only hints (focus slice when one pasted narrative fans out to multiple runs).
    supplementary_constraints: str = ""
    # When True (multi-unit product scope): artifact generation uses one epic + single feature row — no extra LLM ``features[]`` split.
    collapse_product_feature_decomposition: bool = False


def _extract_json_object(text: str) -> dict:
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
    raise ValueError("No valid JSON in intake response")


INTAKE_PROMPT = """You are a senior business analyst. **Classify** the requirement **level**, then **split and normalize** for downstream processing.

**Input:**
---
{text}
---

**Levels (pick exactly one). Prioritize explicit scope or time-box phrases over vague action verbs:**
- **product**: Whole application, platform, or system (e.g. "build an app", "platform", "the system", "end-to-end").
- **sprint**: Time-boxed bundle (e.g. "this sprint", "this release", "next iteration", "for Q2 release")—may contain several capabilities.
- **feature**: One primary user-facing capability (e.g. "user should be able to …", single clear user goal).
- **enhancement**: Improves or extends something existing ("add", "improve", "enhance", "extend") without implying a whole new product.
- **bug**: Defect or incorrect behavior ("bug", "broken", "not working", "error", "fix", "regression", "fails when").

**Split & normalize:**
1. **product** or **sprint**: Use **one** unit with the **full** narrative only when the scope is truly **one** delivery without distinct primary capabilities that stakeholders would backlog separately. When the text lists **several primary capabilities or modules** that warrant **their own** refinement/clarification passes (e.g. unrelated shopper vs admin areas, or clearly separate modules named by the user), set `"multiple_independent_features": **true**` and emit **2** to {_max} **units** — one **feature_name** per unit (Title Case, 2–5 words), **standalone** **text** per unit (trim boilerplate only), exact literals preserved. If the list is **themes under one indivisible delivery**, **false** with **one** unit holding the full narrative. Prefer **splitting** when multiple **user-facing** outcomes are materially independent.
2. **feature** or **enhancement**: Set `"multiple_independent_features": false` **unless** there are **clearly separate unrelated** capabilities (e.g. unrelated admin vs customer flows)—then **true** with one unit per capability.
3. **Split** when multiple **independent user-facing capabilities** are clearly present—**not** when the text only lists **external systems**, **integrations**, or **impacted platforms** (see **Hard rules**).

Return ONLY valid JSON (no markdown):
{{
  "requirement_level": "product" | "sprint" | "feature" | "enhancement" | "bug",
  "multiple_independent_features": true or false,
  "units": [
    {{ "feature_name": "<short title; use \"\" only for a single generic unit>", "text": "<normalized standalone requirement>" }}
  ]
}}

**Hard rules:**
- **One product story:** All **units** must belong to the **same** product or system the user described—do **not** mix unrelated domains or separate products in one intake response. Each **feature_name** + **text** should read as part of **one** coherent scope.
- **Capabilities only:** Classify and split **system or user-facing behaviors** in the text. Do **not** emit a unit whose **text** is only a **coordination note**, **placeholder** (e.g. standalone TBD), or **stakeholder discussion** line—those are excluded upstream.
- **External systems vs features (mandatory):** **Impacted systems**, **integrations**, **dependencies**, or **named platforms** (e.g. **Order Management System**, **Notification Service**, **Payment Gateway**, **CRM**) describe **where** a capability touches another product—they are **integration / impact context**, **not** standalone user-facing capabilities. **Do not** emit **one intake unit per** named external system. Fold those names into the **parent** capability’s **text** (e.g. one **Delivery Tracking** unit that states integration with OMS and Notification Service). **Exception:** emit a **separate** unit **only** when the source explicitly demands **building or owning** that external system as its **own** deliverable.
- **Single primary capability (mandatory):** **Do not** create **separate intake units** from **constraints**, **SLAs / time guarantees**, **NFRs**, **payment methods**, **checkout steps**, or **supporting sub-flows** when they clearly belong to **one** user-facing capability (e.g. **Food Delivery** stays **one** unit that **includes** ordering flow, payment options, and delivery-time promises in its **text**—**not** three units like "Food Ordering", "Delivery Time Assurance", "Payment Methods"). Split **only** when the text describes **unrelated** primary goals (e.g. unrelated admin vs shopper journeys).
- **Split quality:** Create **multiple** units when there are **clearly separate primary capabilities** (distinct backlog-worthy outcomes). **Do not** split **implementation details** under **one** user-facing capability. **Enumerated areas:** comma lists, **including …**, **with …**, **with … and …** often imply **separate** backlog slices — use `"multiple_independent_features": true` and **one unit per distinct primary capability** unless the narrative is explicitly **one** inseparable delivery; preserve **all** named areas across units or in shared context as appropriate. **Do not** drop explicit capability names.
- If **multiple_independent_features** is **false**: exactly **one** object in **units**; **text** must preserve full intent (only trim meaningless boilerplate).
- If **true**: **2** to {_max} units; each **text** at least ~15 characters; **feature_name** required per unit (non-empty).
- **Gap-style appendices:** If the text includes a block such as **Additional requirements:** with bullets, treat those lines as **constraints, rules, or clarifications** on the same product scope—**do not** create **one unit per bullet** unless a bullet clearly describes a **separate unrelated** user-facing capability. Prefer merging them into the unit(s) that match the main requirement above that block.
- Never return an empty **units** array.
"""


def _clean(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _title_feature_name(label: str) -> str:
    t = _clean(label)
    if not t:
        return ""
    return " ".join(w.capitalize() for w in t.split())


def _segment_short_enough_for_explicit_list(seg: str) -> bool:
    seg = re.sub(r"^\s*and\s+", "", seg, flags=re.I).strip()
    if not seg:
        return False
    words = seg.split()
    if len(words) > _MAX_WORDS_PER_EXPLICIT_SEGMENT:
        return False
    if len(seg) > 72:
        return False
    low = seg.lower()
    if low in _STOPWORD_CAPABILITIES:
        return False
    return True


def _normalize_explicit_segments(parts: list[str]) -> list[str]:
    out: list[str] = []
    for p in parts:
        p = _clean(p)
        p = re.sub(r"^\s*and\s+", "", p, flags=re.I).strip()
        if not p:
            continue
        out.append(p)
    return out


def _parse_comma_list_body(body: str) -> list[str] | None:
    """Parse comma-separated segments if every segment looks like a parallel capability phrase."""
    body = body.rstrip(",.;").strip()
    if "," not in body:
        return None
    raw = [s.strip() for s in body.split(",")]
    segs = _normalize_explicit_segments(raw)
    if len(segs) < 2:
        return None
    if not all(_segment_short_enough_for_explicit_list(s) for s in segs):
        return None
    stop_hits = sum(1 for s in segs if s.lower() in _STOPWORD_CAPABILITIES)
    if stop_hits > len(segs) // 2:
        return None
    return segs


def _explicit_labels_whole_text(text: str) -> list[str] | None:
    """Entire requirement is a comma-separated list of capabilities."""
    t = text.strip()
    return _parse_comma_list_body(t)


def _explicit_labels_after_colon(text: str) -> list[str] | None:
    m = re.search(
        r"(?is)\b(?:capabilities|capability|features|feature\s+areas|must\s+include|including)\s*:\s*(.+)$",
        text.strip(),
    )
    if not m:
        return None
    return _parse_comma_list_body(m.group(1))


def _explicit_labels_including_phrase(text: str) -> list[str] | None:
    m = re.search(r"(?is)\b(?:including|such\s+as|e\.g\.|eg\.)\s+(.+)$", text.strip().rstrip("."))
    if not m:
        return None
    tail = m.group(1).strip()
    if "," in tail:
        return _parse_comma_list_body(tail)
    if re.search(r"\s+and\s+", tail, re.I):
        parts = re.split(r"\s+and\s+", tail, flags=re.I)
        segs = _normalize_explicit_segments(parts)
        if len(segs) >= 2 and all(_segment_short_enough_for_explicit_list(s) for s in segs):
            return segs
    return None


def _explicit_labels_support_tail(text: str) -> list[str] | None:
    """
    Phrases like 'must support cart, payment, tracking' — list after support/include/deliver/needs/cover/with.
    """
    m = re.search(
        r"(?is)\b(?:support|supports|include|includes|deliver|delivers|need|needs|must\s+have|"
        r"implement|implements|cover|covers|provide|provides|with)\s+(.+)$",
        text.strip().rstrip("."),
    )
    if not m:
        return None
    tail = m.group(1).strip()
    if "," not in tail:
        return None
    return _parse_comma_list_body(tail)


def parse_explicit_capability_labels(text: str) -> list[str] | None:
    """
    Return ordered capability labels when ``text`` contains an explicit enumeration
    (comma-separated and/or clear keyword patterns). Otherwise None.
    """
    if not (text or "").strip():
        return None
    for fn in (
        _explicit_labels_after_colon,
        _explicit_labels_support_tail,
        _explicit_labels_including_phrase,
        _explicit_labels_whole_text,
    ):
        labels = fn(text)
        if labels and len(labels) >= 2:
            return labels[:_MAX_EXPLICIT_CAPABILITIES]
    return None


_SCOPE_CONTAINER_TERMS = frozenset(
    {
        "platform",
        "system",
        "application",
        "portal",
        "dashboard",
        "product",
        "suite",
        "ecosystem",
        "solution",
        "software",
        "website",
        "marketplace",
        "store",
    }
)
_APP_TOKEN = re.compile(
    r"\b(app|webapp|web\s+app|mobile\s+app|microservice|microservices)\b",
    re.I,
)
_SCOPE_VERB = re.compile(
    r"\b(build|develop|create|implement|design|deliver|launch|roll\s*out|stand\s*up)\b",
    re.I,
)
_DOMAIN_OR_STACK = re.compile(
    r"\b(e-?commerce|ecommerce|saas|erp|crm|b2b|b2c|fintech|edtech|telehealth)\b",
    re.I,
)


def _text_signals_single_program_scope(text: str) -> bool:
    """
    True when the narrative reads as one container initiative (product/system/app), using
    vocabulary and structure—not keyword-only checks.
    """
    if not (text or "").strip():
        return False
    low = text.lower()
    words = low.split()
    if any(t in low for t in _SCOPE_CONTAINER_TERMS):
        return True
    if _APP_TOKEN.search(text):
        return True
    if _DOMAIN_OR_STACK.search(text):
        return True
    if _SCOPE_VERB.search(text) and len(words) >= 8:
        return True
    if low.count(",") >= 2 and len(text) >= 45:
        return True
    return False


def _derive_program_feature_label(text: str) -> str:
    """Short Title Case label for the overall program (from the user's lead phrase)."""
    t = _clean(text)
    if not t:
        return "Product Scope"
    low = t.lower()
    if " with " in low:
        cut = low.index(" with ")
        head = t[:cut].strip()
    else:
        head = t
    head = re.sub(r"^(build|develop|create|implement|design|deliver)\s+", "", head, flags=re.I)
    head = re.sub(r"^(an?\s+|the\s+)+", "", head, flags=re.I)
    words = head.split()[:10]
    head = " ".join(words).strip(" ,.;:")
    label = _title_feature_name(head)
    return label if label else "Product Scope"


def _program_scope_with_explicit_capabilities(original: str, labels: list[str]) -> bool:
    """One program-shaped requirement + several distinct named areas → single product run."""
    if len(labels) < 2:
        return False
    if not _text_signals_single_program_scope(original):
        return False
    if any(len(l.split()) > 8 for l in labels):
        return False
    return True


def coalesce_intake_units_if_product_program(
    original: str, units: list[NormalizedRequirementUnit]
) -> list[NormalizedRequirementUnit]:
    """
    Legacy hook — **no longer coalesces**. Multiple intake units are preserved so each capability
    runs its own clarification + artifact pass (see module docstring).
    """
    return units


def _single_slice_supplementary_focus(capability_title: str) -> str:
    return (
        f"**Primary capability for this pipeline run:** {_title_feature_name(capability_title) or capability_title}\n"
        "**Other named areas** in the source are **context only** — scope refinement and artifacts "
        "to **this** capability; **do not** add sibling backlog **features[]** for other areas in this run."
    )


def _mark_collapse_for_split_product_units(
    units: list[NormalizedRequirementUnit],
) -> list[NormalizedRequirementUnit]:
    """Product-level units in a multi-unit intake share one backlog bucket per run (no nested feature explosion)."""
    if len(units) < 2:
        return units
    for u in units:
        if normalize_requirement_level(u.requirement_level) == "product":
            u.collapse_product_feature_decomposition = True
    return units


def fix_explicit_capability_extraction(raw_text: str) -> list[NormalizedRequirementUnit] | None:
    """
    When the requirement explicitly lists capability labels **and** reads as one program scope,
    return **one intake unit per listed primary capability** (same full narrative + focus hint each)
    so the pipeline runs clarification + artifacts **per** capability.

    Bare comma lists without program scope are left to LLM intake (returns None).
    """
    original = _clean(raw_text or "")
    if not original:
        return None
    labels = parse_explicit_capability_labels(original)
    if not labels or len(labels) < 2:
        return None
    if not _program_scope_with_explicit_capabilities(original, labels):
        return None
    out: list[NormalizedRequirementUnit] = []
    for lab in labels[:_MAX_EXPLICIT_CAPABILITIES]:
        disp = _title_feature_name(lab) or lab
        out.append(
            NormalizedRequirementUnit(
                text=original,
                feature_name=disp,
                requirement_level="product",
                supplementary_constraints=_single_slice_supplementary_focus(lab),
                collapse_product_feature_decomposition=True,
            )
        )
    return _mark_collapse_for_split_product_units(out)


def _fallback_structural_split(text: str) -> list[str]:
    """Last-resort two-part split for forced multi scenarios."""
    t = text.strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n+", t) if len(p.strip()) >= _MIN_UNIT_TEXT]
    if len(parts) >= 2:
        return parts[:_MAX_UNITS]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) >= _MIN_UNIT_TEXT]
    if len(sents) >= 2:
        mid = max(1, len(sents) // 2)
        a, b = " ".join(sents[:mid]), " ".join(sents[mid:])
        if len(a) >= _MIN_UNIT_TEXT and len(b) >= _MIN_UNIT_TEXT:
            return [a, b]
    n = len(t)
    if n >= 30:
        mid = n // 2
        cut = t.rfind(" ", max(0, mid - 60), mid + 60)
        if cut <= 0:
            cut = t.find(" ", mid)
        if cut <= 0:
            cut = mid
        a, b = t[:cut].strip(), t[cut:].strip()
        if len(a) >= _MIN_UNIT_TEXT and len(b) >= _MIN_UNIT_TEXT:
            return [a, b]
    return []


def normalize_requirement_level(raw: object) -> str:
    """Canonical level for pipeline and epic gating: product | sprint | feature | enhancement | bug."""
    s = str(raw or "").strip().lower()
    if s in ("product", "sprint", "feature", "enhancement", "bug"):
        return s
    return "feature"


def _normalize_level(raw: object) -> str:
    return normalize_requirement_level(raw)


def _parse_units(data: dict) -> list[NormalizedRequirementUnit]:
    level = _normalize_level(data.get("requirement_level"))
    multi = data.get("multiple_independent_features", False)
    if isinstance(multi, str):
        multi = multi.strip().lower() in ("true", "yes", "1")
    raw_units = data.get("units", [])
    if not isinstance(raw_units, list):
        return []
    out: list[NormalizedRequirementUnit] = []
    for item in raw_units:
        if isinstance(item, str):
            tx = _clean(item)
            if len(tx) >= _MIN_UNIT_TEXT:
                out.append(NormalizedRequirementUnit(text=tx, requirement_level=level))
        elif isinstance(item, dict):
            tx = _clean(str(item.get("text", "")))
            fn = _clean(str(item.get("feature_name", "")))
            if len(tx) >= _MIN_UNIT_TEXT:
                out.append(
                    NormalizedRequirementUnit(text=tx, feature_name=fn, requirement_level=level)
                )
    if not multi and len(out) > 1:
        merged = _clean(" ".join(u.text for u in out))
        out = [
            NormalizedRequirementUnit(
                text=merged or out[0].text,
                feature_name=out[0].feature_name,
                requirement_level=level,
            )
        ]
    if multi and len(out) < 2:
        out = []
    if multi and len(out) > _MAX_UNITS:
        out = out[:_MAX_UNITS]
    return out


def analyze_intake(raw_text: str) -> list[NormalizedRequirementUnit]:
    """
    Classify, normalize, and optionally split input into intake units.
    Returns one unit for a single scope, or **several** when the source names multiple
    **primary** capabilities (see ``INTAKE_PROMPT`` and ``fix_explicit_capability_extraction``).
    """
    original = _clean(raw_text or "")
    if not original:
        return []

    explicit = fix_explicit_capability_extraction(original)
    if explicit:
        return explicit

    prompt = INTAKE_PROMPT.format(text=original, _max=_MAX_UNITS)
    try:
        with agent_log("Requirement intake"):
            content = call_llm(prompt)
        if not content:
            return [NormalizedRequirementUnit(text=original, requirement_level="feature")]
        data = _extract_json_object(content)
        units = _parse_units(data)
        if not units:
            fb = _fallback_structural_split(original)
            if len(fb) >= 2:
                fb_units = [
                    NormalizedRequirementUnit(
                        text=t,
                        feature_name=f"Capability {i + 1}",
                        requirement_level=_normalize_level(data.get("requirement_level")),
                    )
                    for i, t in enumerate(fb)
                ]
                return _mark_collapse_for_split_product_units(fb_units)
            return [NormalizedRequirementUnit(text=original, requirement_level="feature")]
        if len(units) == 1:
            u = units[0]
            if not u.requirement_level:
                u.requirement_level = _normalize_level(data.get("requirement_level"))
            return [u]
        for i, u in enumerate(units):
            if not u.feature_name:
                u.feature_name = f"Capability {i + 1}"
        return _mark_collapse_for_split_product_units(units)
    except Exception:
        fb = _fallback_structural_split(original)
        if len(fb) >= 2:
            fb_units = [
                NormalizedRequirementUnit(
                    text=t,
                    feature_name=f"Capability {i + 1}",
                    requirement_level="product",
                )
                for i, t in enumerate(fb)
            ]
            return _mark_collapse_for_split_product_units(fb_units)
        return [NormalizedRequirementUnit(text=original, requirement_level="feature")]
