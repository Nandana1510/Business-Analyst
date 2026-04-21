"""
Stage 1: Requirement Entry

Accepts unstructured product requirements in natural language (typed, pasted, or from PDF / Word / text files).
No predefined format; any scope level (product, feature, sprint, enhancement, bug fix).

File bytes are normalized via ``requirement_document_input`` (extract + preprocess) before the rest of the pipeline.
"""

from dataclasses import dataclass, field

from stages.requirement_document_input import (
    DocumentExtractionError,
    extract_text_from_bytes,
    preprocess_requirement_with_classification,
)


@dataclass
class RawRequirement:
    """Holds the user's raw requirement text as submitted."""

    text: str
    source: str = "user"  # "user" | "file"
    source_filename: str | None = None  # set when ``source == "file"``
    intake_feature_label: str | None = None  # optional hint from intake / split stage
    requirement_level: str | None = None  # from intake: product | sprint | feature | enhancement | bug
    # Lines classified as pending discussion / clarification (excluded from feature intake)
    open_items: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.text and self.text.strip())


def accept_requirement_from_console() -> RawRequirement:
    """
    Prompt the user to enter a requirement in free text.
    Supports single or multiple lines; user finishes with an empty line.
    """
    print("\n--- Requirement Entry ---")
    print(
        "Enter your requirement in natural language (any scope: product, feature, sprint, enhancement, bug fix)."
    )
    print("One line or multiple lines; press Enter twice when done.\n")

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
            continue
        lines.append(line)

    text = "\n".join(lines).strip()
    pre = preprocess_requirement_with_classification(text)
    if not pre.functional_text.strip() and pre.open_items:
        raise ValueError(
            "No actionable system requirements found after preprocessing. "
            "The following were treated as pending discussion or clarification only: "
            + "; ".join(pre.open_items)
        )
    if not pre.functional_text.strip():
        raise ValueError("Requirement text is empty after preprocessing.")
    return RawRequirement(
        text=pre.functional_text,
        open_items=list(pre.open_items),
        source="user",
    )


def accept_requirement_text(text: str) -> RawRequirement:
    """
    Create a RawRequirement from a string (e.g. for API or tests).
    Applies the same preprocessing as the Streamlit entry path for consistency.
    """
    pre = preprocess_requirement_with_classification(text or "")
    if not pre.functional_text.strip() and pre.open_items:
        raise ValueError(
            "No actionable system requirements found after preprocessing. "
            "The following were treated as pending discussion or clarification only: "
            + "; ".join(pre.open_items)
        )
    if not pre.functional_text.strip():
        raise ValueError("Requirement text is empty after preprocessing.")
    return RawRequirement(text=pre.functional_text, open_items=list(pre.open_items), source="user")


def text_from_uploaded_file(filename: str, file_bytes: bytes) -> str:
    """
    Extract text from a supported document and run preprocessing.
    Raises ``DocumentExtractionError`` if the format cannot be read, or ``ValueError`` if text is empty.
    """
    raw = extract_text_from_bytes(filename, file_bytes)
    pre = preprocess_requirement_with_classification(raw)
    if not pre.functional_text.strip() and pre.open_items:
        raise ValueError(
            "No actionable system requirements found after preprocessing. "
            "The following were treated as pending discussion or clarification only: "
            + "; ".join(pre.open_items)
        )
    if not pre.functional_text.strip():
        raise ValueError(
            "Extracted text is empty after preprocessing. Try another file or paste the requirement as text."
        )
    return pre.functional_text


def raw_requirement_from_file(filename: str, file_bytes: bytes) -> RawRequirement:
    """Build ``RawRequirement`` with ``source="file"`` after extract + preprocess."""
    raw = extract_text_from_bytes(filename, file_bytes)
    pre = preprocess_requirement_with_classification(raw)
    if not pre.functional_text.strip() and pre.open_items:
        raise ValueError(
            "No actionable system requirements found after preprocessing. "
            "The following were treated as pending discussion or clarification only: "
            + "; ".join(pre.open_items)
        )
    if not pre.functional_text.strip():
        raise ValueError(
            "Extracted text is empty after preprocessing. Try another file or paste the requirement as text."
        )
    return RawRequirement(
        text=pre.functional_text,
        open_items=list(pre.open_items),
        source="file",
        source_filename=(filename or "").strip() or None,
    )


def raw_requirement_from_manual_text(text: str) -> RawRequirement:
    """Typed/pasted text with the same preprocessing as file extracts."""
    pre = preprocess_requirement_with_classification(text or "")
    if not pre.functional_text.strip() and pre.open_items:
        raise ValueError(
            "No actionable system requirements found after preprocessing. "
            "The following were treated as pending discussion or clarification only: "
            + "; ".join(pre.open_items)
        )
    if not pre.functional_text.strip():
        raise ValueError(
            "Requirement text is empty after preprocessing. Enter substantive text or upload a document."
        )
    return RawRequirement(text=pre.functional_text, open_items=list(pre.open_items), source="user")
