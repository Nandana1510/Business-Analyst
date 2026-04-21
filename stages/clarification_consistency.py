"""
Business-logic consistency checks on clarification answers.

Flags unrealistic combinations and contradictions (especially billing, fulfillment,
cancellation) before refinement. Does not replace LLM quality; augments with
deterministic rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from stages.requirement_clarification import ClarifiedRequirement


@dataclass
class ClarificationValidationResult:
    """Outcome of validating clarification answers for internal consistency."""

    issues: list[tuple[str, str]] = field(default_factory=list)  # (severity, message)

    @property
    def has_errors(self) -> bool:
        return any(sev == "error" for sev, _ in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(sev == "warning" for sev, _ in self.issues)


def _chunks(clarified: ClarifiedRequirement) -> list[str]:
    """Text segments to scan (per-field and combined)."""
    parts: list[str] = []
    if clarified.timing:
        parts.append(clarified.timing)
    if clarified.billing_behavior:
        parts.append(clarified.billing_behavior)
    for _k, v in clarified.additional.items():
        if v:
            parts.append(str(v))
    return parts


def _full_text(clarified: ClarifiedRequirement) -> str:
    return " ".join(_chunks(clarified)).lower()


def _self_contradiction_in_chunk(chunk: str, patterns_a, patterns_b, message: str) -> str | None:
    if not chunk or not chunk.strip():
        return None
    t = chunk.lower()
    if patterns_a.search(t) and patterns_b.search(t):
        return message
    return None


# Billing: pause / no charge vs full price continues (same statement)
_RE_BILLING_PAUSE = re.compile(
    r"no charge|not charged|billing pauses?|pause.*bill|bill.*pause|"
    r"zero charge|suspend.*(payment|billing)|charges? stop",
    re.I,
)
_RE_BILLING_FULL_CONTINUE = re.compile(
    r"full (subscription )?price continues|charges? continue|billing continues( at full)?|"
    r"unchanged (subscription )?(price|billing)|still charged full|full rate continues",
    re.I,
)

# Refund
_RE_NO_REFUND = re.compile(r"no refund|non-?refundable|refunds? (are )?not (allowed|offered)", re.I)
_RE_FULL_REFUND = re.compile(r"full refund|complete refund|100\s*% refund|refund in full", re.I)

# Cancel vs ship pending
_RE_IMMEDIATE_CANCEL = re.compile(
    r"cancel(s|lation)? (effective )?immediately|immediate(ly)? cancel|effective immediately",
    re.I,
)
_RE_FULFILL_PENDING = re.compile(
    r"ship (all )?pending|fulfill (all )?(open|pending) orders|complete (all )?scheduled (delivery|orders)",
    re.I,
)

# Delivery pause vs ship
_RE_NO_DELIVERY = re.compile(
    r"no deliveries? during|deliveries? pause|skip (all )?deliveries|hold deliveries?",
    re.I,
)
_RE_DELIVERY_CONTINUES = re.compile(
    r"deliveries? continue|still (receive|get) deliveries|ship(s|ping)? during pause",
    re.I,
)


def validate_clarification_answers(clarified: ClarifiedRequirement) -> ClarificationValidationResult:
    """
    Check clarification answers for contradictory or unrealistic combinations.

    Returns errors (block downstream) for clear logical clashes; warnings for
    cases that need human verification.
    """
    issues: list[tuple[str, str]] = []
    chunks = _chunks(clarified)
    if not chunks:
        return ClarificationValidationResult(issues=[])

    # 1) Same-field billing contradiction
    for ch in chunks:
        msg = _self_contradiction_in_chunk(
            ch,
            _RE_BILLING_PAUSE,
            _RE_BILLING_FULL_CONTINUE,
            "Clarification mixes “no / paused charges” with “full price continues unchanged” in the same "
            "answer—pick one policy per period or clarify scope (e.g. during pause vs after resume).",
        )
        if msg:
            issues.append(("error", msg))

    # 2) Refund contradiction (anywhere in combined answers)
    full = _full_text(clarified)
    if _RE_NO_REFUND.search(full) and _RE_FULL_REFUND.search(full):
        issues.append(
            (
                "error",
                "Answers imply both no refunds and a full refund—align cancellation/refund policy with "
                "real-world billing rules.",
            )
        )

    # 3) Immediate cancellation vs fulfilling all pending orders (cross-answers)
    if _RE_IMMEDIATE_CANCEL.search(full) and _RE_FULFILL_PENDING.search(full):
        issues.append(
            (
                "warning",
                "Immediate cancellation is combined with fulfilling all pending orders—confirm cutoff "
                "for production/fulfillment vs billing cancellation so behavior matches operations.",
            )
        )

    # 4) Delivery paused vs deliveries continue
    for ch in chunks:
        msg = _self_contradiction_in_chunk(
            ch,
            _RE_NO_DELIVERY,
            _RE_DELIVERY_CONTINUES,
            "Same answer suggests deliveries both stop and continue during a pause—clarify fulfillment "
            "for the paused window.",
        )
        if msg:
            issues.append(("error", msg))

    # De-duplicate identical (severity, message)
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return ClarificationValidationResult(issues=unique)
