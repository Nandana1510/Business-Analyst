"""
Convert gap-analysis lines into short requirement statements and compose a new requirement body.

Used when re-running the full pipeline from user-selected gaps (iterative refinement).
"""

from __future__ import annotations

import json
import re

from stages.pipeline_logging import agent_log
from stages.requirement_intake import NormalizedRequirementUnit
from stages.requirement_understanding import call_llm


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
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
    raise ValueError("No valid JSON object in LLM response")


def _fallback_statement(gap: str) -> str:
    g = " ".join((gap or "").split()).strip()
    if not g:
        return "The system should satisfy the stated business need."
    if g.endswith("?"):
        stem = g[:-1].strip()
        low = stem.lower()
        if low.startswith("what ") or low.startswith("which ") or low.startswith("how "):
            return f"The system should define and document: {stem}"
        if low.startswith("can ") or low.startswith("could ") or low.startswith("should "):
            return f"The system should allow or support: {stem}"
        return f"The system should clarify and specify: {stem}"
    return f"The system should ensure: {g}"


def convert_gaps_to_requirement_statements(gaps: list[str]) -> list[str]:
    """
    One backlog-style sentence per gap (same order). Uses an LLM batch call; falls back per line on failure.
    """
    clean = [(" ".join((g or "").split()).strip()) for g in gaps]
    clean = [g for g in clean if g]
    if not clean:
        return []
    n = len(clean)
    numbered = "\n".join(f"{i + 1}. {g}" for i, g in enumerate(clean))
    prompt = (
        "You convert **gap analysis** lines (open questions, risks, missing detail) into concise "
        "**requirement statements** suitable for a PRD backlog.\n\n"
        "**Rules:**\n"
        "- Output **exactly one** declarative sentence per gap, **same count and order** as the numbered list.\n"
        "- Prefer **The system should …** when describing product/platform behavior (define, support, allow, "
        "record, enforce, expose).\n"
        "- For **yes/no** or **can users …** style gaps, phrase as capability: e.g. "
        '"Can users cancel orders?" → "The system should allow users to cancel orders."\n'
        "- For **policy/definition** gaps: e.g. "
        '"What is the refund policy?" → "The system should define a refund policy."\n'
        "- **Do not** invent specific numbers, deadlines, or policies not hinted at in the gap; stay faithful "
        "to the gap’s intent.\n"
        "- No bullets inside a sentence; one sentence only each.\n\n"
        f"**Gaps ({n}):**\n{numbered}\n\n"
        'Return ONLY valid JSON (no markdown): {"statements": ["...", ...]} '
        f"with **exactly** {n} strings."
    )
    try:
        with agent_log("Gap to requirement statements"):
            raw = call_llm(prompt)
        data = _extract_json_object(raw)
        arr = data.get("statements", [])
        if not isinstance(arr, list):
            raise ValueError("statements not a list")
        out: list[str] = []
        for i in range(n):
            s = str(arr[i]).strip() if i < len(arr) else ""
            if not s:
                s = _fallback_statement(clean[i])
            out.append(s)
        return out
    except Exception:
        return [_fallback_statement(g) for g in clean]


def build_combined_requirement_text(original_requirement: str, statements: list[str]) -> str:
    """Append ``Additional requirements`` block after the original requirement text."""
    base = (original_requirement or "").strip()
    stmts = [(" ".join((s or "").split()).strip()) for s in statements]
    stmts = [s for s in stmts if s]
    if not stmts:
        return base
    lines = "\n".join(f"- {s}" for s in stmts)
    if not base:
        return f"Additional requirements:\n{lines}\n"
    return f"{base}\n\nAdditional requirements:\n{lines}\n"


_ADDITIONAL_BLOCK = re.compile(
    r"(?is)\n{1,2}\s*Additional requirements:\s*\n(?P<body>.+)$",
)
_LEADING_ADDITIONAL = re.compile(
    r"(?is)^\s*Additional requirements:\s*\n(?P<body>.+)$",
)


def split_core_and_gap_supplement(full_text: str) -> tuple[str, str]:
    """
    Split text composed with :func:`build_combined_requirement_text` into intake-safe **core**
    and the **Additional requirements** body (bullet lines). Used so gap-derived lines do not
    drive feature intake / splitting as separate capabilities.
    """
    s = (full_text or "").rstrip()
    if not s:
        return "", ""
    m = _ADDITIONAL_BLOCK.search(s)
    if m:
        core = s[: m.start()].strip()
        body = (m.group("body") or "").strip()
        return core, body
    m2 = _LEADING_ADDITIONAL.match(s)
    if m2:
        return "", (m2.group("body") or "").strip()
    return s, ""


def parse_gap_focus_bullet_statements(gap_focus_block: str) -> list[str]:
    """
    Split a gap-focus / supplementary block into individual requirement lines.
    Expects ``- …`` bullets (as produced for refinement); falls back to one chunk if there are no bullets.
    """
    raw = (gap_focus_block or "").strip()
    if not raw:
        return []
    bullets: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("- "):
            bullets.append(s[2:].strip())
        elif s.startswith("-") and len(s) > 1 and not s.startswith("--"):
            bullets.append(s[1:].strip())
        elif s.startswith("• "):
            bullets.append(s[2:].strip())
    if bullets:
        return [b for b in bullets if b]
    return [raw]


def _format_gap_focus_bullet_block(statements: list[str]) -> str:
    stmts = [(" ".join((s or "").split()).strip()) for s in statements if (s or "").strip()]
    if not stmts:
        return ""
    return "\n".join(f"- {s}" for s in stmts)


def _best_unit_by_token_overlap(statement: str, units: list[NormalizedRequirementUnit]) -> int:
    """Heuristic fallback: pick the intake unit whose text + label overlaps the statement most."""
    st = set(re.findall(r"[a-z0-9]+", (statement or "").lower()))
    if not st:
        return 0
    best_i, best_score = 0, -1.0
    for i, u in enumerate(units):
        blob = f"{u.feature_name or ''} {u.text or ''}"
        ut = set(re.findall(r"[a-z0-9]+", blob.lower()))
        inter = len(st & ut)
        score = inter / (1 + len(st))
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def assign_additional_statements_to_units(
    statements: list[str],
    units: list[NormalizedRequirementUnit],
) -> list[list[str]]:
    """
    For each additional requirement line, pick **one** intake unit it primarily extends.
    Returns ``len(units)`` lists of statements (order preserved within each bucket).
    """
    n = len(units)
    if n <= 1:
        return [list(statements)]
    if not statements:
        return [[] for _ in units]
    numbered_st = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    numbered_u = "\n".join(
        f"{i}. feature_name={json.dumps((u.feature_name or '')[:120])} "
        f"scope_preview={json.dumps((u.text or '')[:400])}"
        for i, u in enumerate(units)
    )
    prompt = (
        "You map **additional requirement** lines (gap-derived refinements) to **exactly one** "
        "intake feature each.\n\n"
        "**Intake features (indices 0–"
        f"{n - 1}"
        ", use each line's index only):**\n"
        f"{numbered_u}\n\n"
        "**Additional requirement statements (same order as below):**\n"
        f"{numbered_st}\n\n"
        "**Rules:**\n"
        "- Output **one** unit index per statement, **same count and order** as the statements.\n"
        "- Choose the feature whose **user-facing capability** the statement most clearly tightens "
        "(e.g. email rules → registration/sign-up; login validation → login).\n"
        "- If a line fits **multiple** features, pick the **single best** primary fit — never assign "
        "to all features.\n"
        "- If unclear, prefer the feature whose **scope_preview** mentions the same domain nouns.\n\n"
        'Return ONLY valid JSON: {"assignments": [0, 1, ...]} — **exactly '
        f"{len(statements)}"
        " integers, each in 0.."
        f"{n - 1}"
        "}.\n"
    )
    assignments: list[int] = []
    try:
        with agent_log("Gap statements → intake units"):
            raw = call_llm(prompt)
        data = _extract_json_object(raw)
        arr = data.get("assignments", [])
        if not isinstance(arr, list) or len(arr) != len(statements):
            raise ValueError("bad assignments length")
        for x in arr:
            j = int(x)
            if j < 0 or j >= n:
                raise ValueError("assignment out of range")
            assignments.append(j)
    except Exception:
        assignments = [_best_unit_by_token_overlap(s, units) for s in statements]

    buckets: list[list[str]] = [[] for _ in range(n)]
    for s, j in zip(statements, assignments):
        buckets[j].append(s)
    return buckets


def per_unit_gap_focus_blocks(
    statements: list[str],
    units: list[NormalizedRequirementUnit],
) -> list[str]:
    """Build one bullet block per intake unit (only that unit's lines)."""
    buckets = assign_additional_statements_to_units(statements, units)
    return [_format_gap_focus_bullet_block(b) for b in buckets]
