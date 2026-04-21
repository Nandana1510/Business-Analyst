"""
Requirement intake: classify level (product / sprint / feature / enhancement / bug), normalize text,
and split into independent feature units with optional short feature names.

Runs before understanding. Downstream pipeline runs per unit:
understanding → clarification → refinement → artifacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from stages.pipeline_logging import agent_log
from stages.requirement_understanding import call_llm

_MAX_UNITS = 5
_MIN_UNIT_TEXT = 12


@dataclass
class NormalizedRequirementUnit:
    """One normalized slice of input for the unified pipeline."""

    text: str
    feature_name: str = ""
    requirement_level: str = "feature"  # product | sprint | feature | enhancement | bug


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
1. **product** or **sprint**: Set `"multiple_independent_features": true` when the text is high-level (app/platform/system/sprint container) **or** clearly contains **multiple distinct capabilities**. Emit **2** to {_max} **units**. Each unit needs a concrete **feature_name** (Title Case, 2–5 words, e.g. "Cart Management", "Payment Processing"). **Remove boilerplate** from each unit's **text** (phrases like "we need to build an app that", "the platform must", "this sprint we will") so **text** reads as a **standalone capability**; **keep exact literals** from the source (proper names, colors like "black", numbers, integrations).
2. **feature** or **enhancement**: Set `"multiple_independent_features": false` **unless** there are **clearly separate unrelated** capabilities (e.g. unrelated admin vs customer flows)—then **true** with one unit per capability.
3. **Always split** when multiple independent actions/features are clearly present, regardless of level.

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
- **Split quality:** Create **multiple** units **only** when there are **clearly separate user-facing capabilities** (different goals or workflows). **Do not** split **implementation details** (e.g. "in-house delivery" vs "delivery tracking") into separate units when they are **the same** delivery capability; keep them in **one** unit. **Do not** miss an **explicit** capability named in the text—each distinct capability the user called out should appear **somewhere** in **units** (either its own unit or clearly inside another unit’s **text**).
- If **multiple_independent_features** is **false**: exactly **one** object in **units**; **text** must preserve full intent (only trim meaningless boilerplate).
- If **true**: **2** to {_max} units; each **text** at least ~15 characters; **feature_name** required per unit (non-empty).
- Never return an empty **units** array.
"""


def _clean(s: str) -> str:
    return " ".join((s or "").split()).strip()


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
    Classify, normalize, and optionally split input into feature units.
    Always returns at least one unit when input is non-empty.
    """
    original = _clean(raw_text or "")
    if not original:
        return []

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
                return [
                    NormalizedRequirementUnit(
                        text=t,
                        feature_name=f"Capability {i + 1}",
                        requirement_level=_normalize_level(data.get("requirement_level")),
                    )
                    for i, t in enumerate(fb)
                ]
            return [NormalizedRequirementUnit(text=original, requirement_level="feature")]
        if len(units) == 1:
            u = units[0]
            if not u.requirement_level:
                u.requirement_level = _normalize_level(data.get("requirement_level"))
            return [u]
        # Multi: ensure feature names
        for i, u in enumerate(units):
            if not u.feature_name:
                u.feature_name = f"Capability {i + 1}"
        return units
    except Exception:
        fb = _fallback_structural_split(original)
        if len(fb) >= 2:
            return [
                NormalizedRequirementUnit(
                    text=t,
                    feature_name=f"Capability {i + 1}",
                    requirement_level="product",
                )
                for i, t in enumerate(fb)
            ]
        return [NormalizedRequirementUnit(text=original, requirement_level="feature")]
