"""
Persist full pipeline outputs to ``outputs/output_<timestamp>.json`` and ``output_<timestamp>.md``.

JSON: machine-readable, reloadable. Markdown: human-readable report with the same content.

Failures are logged and never raise into the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stages.artifact_generation import AdvancedArtifacts

logger = logging.getLogger(__name__)

# Project root: stages/ → parent.parent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "outputs"


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _md_plain(text: object) -> str:
    """Escape markdown table pipes; keep readable body text."""
    s = str(text) if text is not None else ""
    return s.replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n")


def _md_quote_block(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.strip().split("\n")) if text.strip() else "_empty_"


def _append_meta_list(lines: list[str], meta: dict[str, Any]) -> None:
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    order = [
        "saved_at_utc",
        "pipeline_kind",
        "requirement_level",
        "requirement_levels_all",
        "artifact_generation_scope",
        "acceptance_criteria_format",
        "feature_count",
        "user_story_count",
    ]
    done: set[str] = set()
    for k in order:
        if k in meta and meta[k] not in (None, "", []):
            v = meta[k]
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            lines.append(f"| {k} | {_md_plain(v)} |")
            done.add(k)
    for k, v in sorted(meta.items()):
        if k in done or v in (None, "", []):
            continue
        if isinstance(v, (dict, list)):
            continue
        lines.append(f"| {k} | {_md_plain(v)} |")


def _kv_lines_flat(prefix: str, d: dict[str, Any], skip: set[str]) -> list[str]:
    out: list[str] = []
    for k in sorted(d.keys()):
        if k in skip:
            continue
        v = d[k]
        if v is None or v == "" or v == []:
            continue
        label = k.replace("_", " ").title()
        if isinstance(v, list):
            out.append(f"- **{label}:**")
            for item in v:
                out.append(f"  - {_md_plain(item)}")
        else:
            out.append(f"- **{label}:** {_md_plain(v)}")
    return out


def _render_epic_block(ep: object, lines: list[str]) -> None:
    if ep is None or ep == "":
        return
    if isinstance(ep, str):
        lines.append(ep.strip())
        return
    if not isinstance(ep, dict):
        lines.append(_md_plain(ep))
        return
    title = (ep.get("epic_title") or "").strip()
    if title:
        lines.append(f"**Epic title:** {_md_plain(title)}")
    for key, label in (
        ("epic_summary", "Summary"),
        ("epic_description", "Description"),
        ("business_problem", "Business problem"),
    ):
        t = (ep.get(key) or "").strip()
        if t:
            lines.append(f"- **{label}:** {_md_plain(t)}")
    for key, label in (
        ("goals_and_objectives", "Goals & objectives"),
        ("key_capabilities", "Key capabilities"),
        ("business_outcomes", "Business outcomes"),
        ("success_metrics", "Success metrics"),
    ):
        items = ep.get(key)
        if isinstance(items, list) and items:
            lines.append(f"- **{label}:**")
            for it in items:
                lines.append(f"  - {_md_plain(it)}")


def _render_user_story_block(st: dict[str, Any], lines: list[str], idx: int) -> None:
    ref = (st.get("story_ref") or "").strip()
    head = f"**Story {idx}**" + (f" (`{ref}`)" if ref else "")
    lines.append(head)
    lines.append("")
    lines.append(_md_plain(st.get("story", "")))
    lines.append("")
    acs = st.get("acceptance_criteria") or []
    if isinstance(acs, list) and acs:
        lines.append("**Acceptance criteria**")
        for j, ac in enumerate(acs, 1):
            if isinstance(ac, dict):
                t = (ac.get("text") or "").strip()
                tr = (ac.get("traces_to") or "").strip()
                line = f"{j}. {t}"
                if tr and tr != "unspecified":
                    line += f" _(→ {tr})_"
            else:
                line = f"{j}. {_md_plain(ac)}"
            lines.append(line)
        lines.append("")


def _render_artifacts_dict(artifacts: dict[str, Any], lines: list[str]) -> None:
    epic = artifacts.get("epic")
    if epic:
        lines.append("### Epic")
        lines.append("")
        _render_epic_block(epic, lines)
        lines.append("")

    feats = artifacts.get("features") or []
    if isinstance(feats, list) and feats:
        for fi, fblock in enumerate(feats, 1):
            if not isinstance(fblock, dict):
                continue
            fname = (fblock.get("feature_name") or f"Feature {fi}").strip()
            lines.append(f"### Feature {fi}: {_md_plain(fname)}")
            if (fblock.get("feature_summary") or "").strip():
                lines.append("")
                lines.append(f"_{_md_plain(fblock['feature_summary'])}_")
                lines.append("")
            epf = fblock.get("epic")
            if epf:
                lines.append("#### Feature epic")
                _render_epic_block(epf, lines)
                lines.append("")
            ujf = fblock.get("user_journey") or []
            if isinstance(ujf, list) and ujf:
                lines.append("#### User journey")
                for i, step in enumerate(ujf, 1):
                    lines.append(f"{i}. {_md_plain(step)}")
                lines.append("")
            gaf = fblock.get("gap_analysis") or []
            if isinstance(gaf, list) and gaf:
                lines.append("#### Gap analysis")
                for i, gap in enumerate(gaf, 1):
                    lines.append(f"{i}. {_md_plain(gap)}")
                lines.append("")
            stories = fblock.get("user_stories") or []
            if isinstance(stories, list) and stories:
                lines.append("#### User stories")
                lines.append("")
                for j, st in enumerate(stories, 1):
                    if isinstance(st, dict):
                        _render_user_story_block(st, lines, j)

    stories_top = artifacts.get("user_stories") or []
    if isinstance(stories_top, list) and stories_top and not feats:
        lines.append("### User stories")
        lines.append("")
        for j, st in enumerate(stories_top, 1):
            if isinstance(st, dict):
                _render_user_story_block(st, lines, j)

    uj = artifacts.get("user_journey") or []
    if isinstance(uj, list) and uj and not feats:
        lines.append("### User journey (top-level)")
        for i, step in enumerate(uj, 1):
            lines.append(f"{i}. {_md_plain(step)}")
        lines.append("")

    ga = artifacts.get("gap_analysis") or []
    if isinstance(ga, list) and ga and not feats:
        lines.append("### Gap analysis (top-level)")
        for i, gap in enumerate(ga, 1):
            lines.append(f"{i}. {_md_plain(gap)}")
        lines.append("")

    br = artifacts.get("bug_report")
    if isinstance(br, dict):
        desc = (br.get("bug_description") or "").strip()
        steps = br.get("steps_to_reproduce") or []
        exp = (br.get("expected_behavior") or "").strip()
        act = (br.get("actual_behavior") or "").strip()
        if desc or exp or act or (isinstance(steps, list) and steps):
            lines.append("### Bug report")
            lines.append("")
            if desc:
                lines.append(f"**Description:** {_md_plain(desc)}")
            if isinstance(steps, list) and steps:
                lines.append("")
                lines.append("**Steps to reproduce:**")
                for i, s in enumerate(steps, 1):
                    lines.append(f"{i}. {_md_plain(s)}")
            if exp:
                lines.append(f"**Expected:** {_md_plain(exp)}")
            if act:
                lines.append(f"**Actual:** {_md_plain(act)}")
            lines.append("")


def _render_refined_block(ref: dict[str, Any], lines: list[str]) -> None:
    if not ref:
        return
    for key, label in (
        ("feature_name", "Feature"),
        ("actor", "Actor"),
        ("secondary_actor", "Secondary actor"),
        ("domain", "Domain"),
        ("description", "Description"),
    ):
        v = ref.get(key)
        if v and str(v).strip():
            lines.append(f"**{label}:** {_md_plain(v)}")
            lines.append("")
    rules = ref.get("business_rules") or []
    if isinstance(rules, list) and rules:
        lines.append("**Business rules**")
        lines.append("")
        lines.append("| ID | Text | Source | Source ID |")
        lines.append("| --- | --- | --- | --- |")
        for br in rules:
            if not isinstance(br, dict):
                lines.append(f"| | {_md_plain(br)} | | |")
                continue
            lines.append(
                f"| {_md_plain(br.get('rule_id', ''))} | {_md_plain(br.get('text', ''))} | "
                f"{_md_plain(br.get('source', ''))} | {_md_plain(br.get('source_id', ''))} |"
            )
        lines.append("")


def payload_to_markdown(payload: dict[str, Any]) -> str:
    """
    Full narrative report: metadata, requirements, understanding, clarification,
    refinement, and all artifacts — suitable for demos, Confluence import, or Git.
    """
    lines: list[str] = []
    meta = payload.get("metadata") or {}
    lines.append("# AI Business Analyst — pipeline output report")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    _append_meta_list(lines, meta)
    lines.append("")

    lines.append("## Original requirement")
    lines.append("")
    orig = (payload.get("original_requirement") or "").strip()
    lines.append(_md_quote_block(orig) if orig else "_None_")
    lines.append("")

    oi = payload.get("open_items_from_preprocess") or []
    if oi:
        lines.append("## Open items (pre-process / intake)")
        lines.append("")
        for item in oi:
            lines.append(f"- {_md_plain(item)}")
        lines.append("")

    kind = meta.get("pipeline_kind") or payload.get("pipeline_kind") or ""

    if kind == "multi_feature":
        feats_payload = payload.get("features") or []
        if isinstance(feats_payload, list):
            for block in feats_payload:
                if not isinstance(block, dict):
                    continue
                idx = block.get("feature_index", "?")
                tot = block.get("feature_total", "?")
                label = (block.get("feature_label") or "").strip()
                lbl = f" — {_md_plain(label)}" if label else ""
                lvl = (block.get("requirement_level") or "").strip()
                lvl_s = f" ({_md_plain(lvl)})" if lvl else ""
                lines.append(
                    f"## Feature {idx} of {tot}{lbl}{lvl_s}"
                )
                lines.append("")
                ut = (block.get("unit_text") or "").strip()
                if ut:
                    lines.append("### Sub-requirement text")
                    lines.append("")
                    lines.append(_md_quote_block(ut))
                    lines.append("")
                udb = block.get("understanding") or {}
                if isinstance(udb, dict) and udb:
                    lines.append("### Understanding")
                    lines.append("")
                    lines.extend(_kv_lines_flat("", udb, set()))
                    lines.append("")
                clb = block.get("clarification") or {}
                if isinstance(clb, dict) and clb:
                    lines.append("### Clarification answers")
                    lines.append("")
                    lines.extend(_kv_lines_flat("", clb, set()))
                    lines.append("")
                rfb = block.get("refined_requirement") or {}
                if isinstance(rfb, dict) and rfb:
                    lines.append("### Refined requirement")
                    lines.append("")
                    _render_refined_block(rfb, lines)
                arts = block.get("artifacts") or {}
                if isinstance(arts, dict) and arts:
                    lines.append("### Generated artifacts")
                    lines.append("")
                    _render_artifacts_dict(arts, lines)
                lines.append("---")
                lines.append("")
    else:
        ud = payload.get("understanding") or {}
        if isinstance(ud, dict) and ud:
            lines.append("## Understanding")
            lines.append("")
            lines.extend(_kv_lines_flat("", ud, set()))
            lines.append("")

        cl = payload.get("clarification") or {}
        if isinstance(cl, dict) and cl:
            lines.append("## Clarification answers")
            lines.append("")
            lines.extend(_kv_lines_flat("", cl, set()))
            lines.append("")

        ref = payload.get("refined_requirement") or {}
        if isinstance(ref, dict) and ref:
            lines.append("## Refined requirement")
            lines.append("")
            _render_refined_block(ref, lines)

        arts = payload.get("artifacts") or {}
        if isinstance(arts, dict) and arts:
            lines.append("## Generated artifacts")
            lines.append("")
            _render_artifacts_dict(arts, lines)

    footer = (
        "\n---\n\n_Generated by the AI Business Analyst pipeline. "
        "JSON export with identical timestamp retains the same structured data for tools and APIs._\n"
    )
    body = "\n".join(lines).strip() + footer
    # Normalise excessive blank lines
    body = re.sub(r"\n{4,}", "\n\n\n", body)
    return body


def _count_user_stories(artifacts: AdvancedArtifacts) -> int:
    n = len(artifacts.user_stories)
    for f in artifacts.features:
        n += len(f.user_stories)
    return n


def _feature_count(artifacts: AdvancedArtifacts) -> int:
    if artifacts.features:
        return len(artifacts.features)
    return 1 if artifacts.user_stories or artifacts.epic_document else 0


def build_payload_single(
    *,
    original_requirement: str,
    open_items: list[str],
    artifact_scope_mode: str,
    acceptance_criteria_format: str,
    understood: Any,
    refined: Any,
    artifacts: AdvancedArtifacts,
    clarified: Any | None = None,
) -> dict[str, Any]:
    """Structured document for one feature / one completed single-feature run."""
    meta = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_kind": "single_feature",
        "requirement_level": getattr(refined, "requirement_level", None) or "",
        "artifact_generation_scope": artifact_scope_mode,
        "acceptance_criteria_format": acceptance_criteria_format,
        "feature_count": _feature_count(artifacts),
        "user_story_count": _count_user_stories(artifacts),
    }
    body: dict[str, Any] = {
        "metadata": meta,
        "original_requirement": original_requirement,
        "open_items_from_preprocess": list(open_items or []),
        "understanding": understood.to_dict() if understood is not None else {},
        "refined_requirement": refined.to_dict() if refined is not None else {},
        "artifacts": artifacts.to_dict(),
    }
    if clarified is not None and hasattr(clarified, "to_dict"):
        body["clarification"] = clarified.to_dict()
    return body


def build_payload_multi(
    *,
    original_requirement: str,
    open_items: list[str],
    artifact_scope_mode: str,
    acceptance_criteria_format: str,
    bundles: list[Any],
) -> dict[str, Any]:
    """``bundles``: list of objects with ``index``, ``total``, ``unit_text``, ``feature_label``,
    ``requirement_level``, ``understood``, ``refined``, ``artifacts``, ``clarified`` (Streamlit bundles)."""
    total_stories = 0
    features_out: list[dict[str, Any]] = []
    levels: list[str] = []
    for b in bundles:
        ar = b.artifacts
        total_stories += _count_user_stories(ar)
        levels.append(getattr(b, "requirement_level", "") or "")
        block: dict[str, Any] = {
            "feature_index": getattr(b, "index", 0),
            "feature_total": getattr(b, "total", 0),
            "feature_label": getattr(b, "feature_label", "") or "",
            "unit_text": getattr(b, "unit_text", "") or "",
            "requirement_level": getattr(b, "requirement_level", "") or "",
            "understanding": b.understood.to_dict(),
            "refined_requirement": b.refined.to_dict(),
            "artifacts": b.artifacts.to_dict(),
        }
        cl = getattr(b, "clarified", None)
        if cl is not None and hasattr(cl, "to_dict"):
            block["clarification"] = cl.to_dict()
        features_out.append(block)

    meta = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_kind": "multi_feature",
        "requirement_level": levels[0] if levels else "",
        "requirement_levels_all": levels,
        "artifact_generation_scope": artifact_scope_mode,
        "acceptance_criteria_format": acceptance_criteria_format,
        "feature_count": len(bundles),
        "user_story_count": total_stories,
    }
    return {
        "metadata": meta,
        "original_requirement": original_requirement,
        "open_items_from_preprocess": list(open_items or []),
        "features": features_out,
    }


def save_run_outputs(payload: dict[str, Any]) -> None:
    """
    Write ``output_<timestamp>.json`` (structured) and ``output_<timestamp>.md`` (readable report).
    JSON and Markdown use the same stem; failures in one format do not block the other.
    """
    try:
        _ensure_output_dir()
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = OUTPUT_DIR / f"output_{stamp}"
    except OSError as e:
        logger.error("Could not prepare outputs directory: %s", e)
        logger.debug("%s", traceback.format_exc())
        return

    try:
        json_path = base.with_suffix(".json")
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved pipeline JSON to %s", json_path)
    except OSError as e:
        logger.error("Failed to save pipeline JSON: %s", e)
        logger.debug("%s", traceback.format_exc())
    except (TypeError, ValueError) as e:
        logger.error("Failed to serialize pipeline JSON: %s", e)
        logger.debug("%s", traceback.format_exc())

    try:
        md_path = base.with_suffix(".md")
        md_path.write_text(payload_to_markdown(payload), encoding="utf-8")
        logger.info("Saved pipeline Markdown report to %s", md_path)
    except OSError as e:
        logger.error("Failed to save Markdown report (filesystem): %s", e)
        logger.debug("%s", traceback.format_exc())
    except Exception as e:
        logger.error("Failed to build or write Markdown report: %s", e)
        logger.debug("%s", traceback.format_exc())


def persist_single_feature_run(
    *,
    original_requirement: str,
    open_items: list[str],
    artifact_scope_mode: str,
    acceptance_criteria_format: str,
    understood: Any,
    refined: Any,
    artifacts: AdvancedArtifacts,
    clarified: Any | None = None,
) -> None:
    """Best-effort save; never raises."""
    try:
        payload = build_payload_single(
            original_requirement=original_requirement,
            open_items=open_items,
            artifact_scope_mode=artifact_scope_mode,
            acceptance_criteria_format=acceptance_criteria_format,
            understood=understood,
            refined=refined,
            artifacts=artifacts,
            clarified=clarified,
        )
        save_run_outputs(payload)
    except Exception as e:
        logger.error("Unexpected error building single-feature payload: %s", e)
        logger.debug("%s", traceback.format_exc())


def persist_multi_feature_run(
    *,
    original_requirement: str,
    open_items: list[str],
    artifact_scope_mode: str,
    acceptance_criteria_format: str,
    bundles: list[Any],
) -> None:
    """Best-effort save; never raises."""
    try:
        payload = build_payload_multi(
            original_requirement=original_requirement,
            open_items=open_items,
            artifact_scope_mode=artifact_scope_mode,
            acceptance_criteria_format=acceptance_criteria_format,
            bundles=bundles,
        )
        save_run_outputs(payload)
    except Exception as e:
        logger.error("Unexpected error building multi-feature payload: %s", e)
        logger.debug("%s", traceback.format_exc())


def _count_stories_and_features_cli(artifacts_dict: dict[str, Any]) -> tuple[int, int]:
    """Avoid double-counting when both ``features`` and top-level ``user_stories`` are populated."""
    feats = artifacts_dict.get("features") or []
    top_stories = artifacts_dict.get("user_stories") or []
    if isinstance(feats, list) and feats:
        n_feat = len(feats)
        nested = sum(len(f.get("user_stories") or []) for f in feats if isinstance(f, dict))
        n_stories = nested if nested else len(top_stories)
        return n_feat, n_stories
    n_stories = len(top_stories)
    br = artifacts_dict.get("bug_report")
    has_content = n_stories or (
        isinstance(br, dict)
        and (
            (str(br.get("bug_description") or "").strip())
            or (str(br.get("expected_behavior") or "").strip())
            or (str(br.get("actual_behavior") or "").strip())
            or (isinstance(br.get("steps_to_reproduce"), list) and len(br.get("steps_to_reproduce")) > 0)
        )
    )
    n_feat = 1 if has_content else 0
    return n_feat, n_stories


def persist_cli_run(
    *,
    original_requirement: str,
    open_items: list[str],
    understood: Any,
    refined: Any,
    artifacts_dict: dict[str, Any],
    acceptance_criteria_format: str,
    artifact_scope_key: str = "generate_all_artifacts",
) -> None:
    """Save from CLI when artifacts are a plain dict from ``generate_all_artifacts``."""
    try:
        fc, uc = _count_stories_and_features_cli(artifacts_dict)
        meta = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_kind": "cli",
            "requirement_level": getattr(refined, "requirement_level", "") or "",
            "artifact_generation_scope": artifact_scope_key,
            "acceptance_criteria_format": acceptance_criteria_format,
            "feature_count": fc,
            "user_story_count": uc,
        }
        payload = {
            "metadata": meta,
            "original_requirement": original_requirement,
            "open_items_from_preprocess": list(open_items or []),
            "understanding": understood.to_dict() if understood is not None else {},
            "refined_requirement": refined.to_dict() if refined is not None else {},
            "artifacts": artifacts_dict,
        }
        save_run_outputs(payload)
    except Exception as e:
        logger.error("CLI persist error: %s", e)
        logger.debug("%s", traceback.format_exc())
