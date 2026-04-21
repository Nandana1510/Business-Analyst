"""
Pre-processing: classify requirement level, normalize, and split into feature units.

All logic lives in ``requirement_intake.analyze_intake``. This module keeps the legacy
``analyze_and_split_requirement_units`` name for callers that expect ``list[str]``.
"""

from __future__ import annotations

from stages.requirement_intake import NormalizedRequirementUnit, analyze_intake

__all__ = ["NormalizedRequirementUnit", "analyze_intake", "analyze_and_split_requirement_units"]


def analyze_and_split_requirement_units(raw_text: str) -> list[str]:
    """Return normalized unit texts for the pipeline (backward-compatible API)."""
    return [u.text for u in analyze_intake(raw_text)]
