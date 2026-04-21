"""
AI Business Analyst System

Entry → optional multi-feature split → Understanding → Clarification → Refinement → Artifacts.
"""

import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

from stages.requirement_entry import accept_requirement_from_console, RawRequirement
from stages.requirement_intake import NormalizedRequirementUnit, analyze_intake
from stages.requirement_understanding import understand_requirement, get_llm_provider_and_model
from stages.requirement_clarification import run_clarification
from stages.clarification_consistency import validate_clarification_answers
from stages.requirement_refinement import refine_requirement
from stages.artifact_generation import generate_all_artifacts, normalize_acceptance_criteria_format
from stages.run_output_storage import persist_cli_run

# For clear 403 vs 401 diagnosis when using OpenAI-compatible APIs (e.g. xAI)
try:
    from openai import PermissionDeniedError as OpenAIPermissionDeniedError
except ImportError:
    class OpenAIPermissionDeniedError(BaseException):
        """Placeholder when openai not installed; never raised."""

    pass


def _print_artifacts(artifacts: dict) -> None:
    """Print artifact dict; level-specific keys may be empty."""
    ep = artifacts.get("epic")
    if isinstance(ep, dict) and (ep.get("epic_title") or "").strip():
        print("Epic (structured):")
        print(f"  Title: {ep.get('epic_title', '').strip()}")
        if (ep.get("epic_summary") or "").strip():
            print(f"  Summary: {str(ep.get('epic_summary')).strip()}")
        if (ep.get("epic_description") or "").strip():
            print(f"  Description: {str(ep.get('epic_description')).strip()}")
        if (ep.get("business_problem") or "").strip():
            print(f"  Business problem: {str(ep.get('business_problem')).strip()}")
        for label, key in (
            ("Goals & objectives", "goals_and_objectives"),
            ("Key capabilities", "key_capabilities"),
            ("Business outcomes", "business_outcomes"),
            ("Success metrics", "success_metrics"),
        ):
            items = ep.get(key)
            if isinstance(items, list) and items:
                print(f"  {label}:")
                for it in items:
                    print(f"    - {it}")
    elif isinstance(ep, str) and ep.strip():
        print("Epic:", ep.strip())
    else:
        print("Epic (whole product): (none — use per-feature epics under each feature when present)")
    feats = artifacts.get("features")
    if isinstance(feats, list) and feats:
        print("Features & user stories (BA hierarchy):")
        for fi, fblock in enumerate(feats, 1):
            if not isinstance(fblock, dict):
                continue
            fname = str(fblock.get("feature_name", "") or f"Feature {fi}")
            print(f"  Feature {fi}: {fname}")
            if (fblock.get("feature_summary") or "").strip():
                print(f"    Summary: {str(fblock.get('feature_summary')).strip()}")
            epf = fblock.get("epic")
            if isinstance(epf, dict) and (epf.get("epic_title") or "").strip():
                print(f"    Epic: {str(epf.get('epic_title')).strip()}")
                if (epf.get("epic_summary") or "").strip():
                    print(f"      Summary: {str(epf.get('epic_summary')).strip()}")
            ujf = fblock.get("user_journey") or []
            if isinstance(ujf, list) and ujf:
                print("    User journey:")
                for ui, step in enumerate(ujf, 1):
                    print(f"      {ui}. {step}")
            gaf = fblock.get("gap_analysis") or []
            if isinstance(gaf, list) and gaf:
                print("    Gap analysis:")
                for gi, gap in enumerate(gaf, 1):
                    print(f"      {gi}. {gap}")
            for j, block in enumerate(fblock.get("user_stories") or [], 1):
                if isinstance(block, dict):
                    story = block.get("story", "")
                    acs = block.get("acceptance_criteria") or []
                else:
                    story = str(block)
                    acs = []
                print(f"    [{j}] {story}")
                if acs:
                    for k, ac in enumerate(acs, 1):
                        if isinstance(ac, dict):
                            t = str(ac.get("text", ""))
                            tr = str(ac.get("traces_to", "")).strip()
                        else:
                            t = str(ac)
                            tr = ""
                        line = f"        • AC {k}: {t}"
                        if tr and tr != "unspecified":
                            line += f"  [→ {tr}]"
                        print(line)
                else:
                    print("        (no acceptance criteria)")
    else:
        print("User stories (each with acceptance criteria):")
        for i, block in enumerate(artifacts.get("user_stories") or [], 1):
            if isinstance(block, dict):
                story = block.get("story", "")
                acs = block.get("acceptance_criteria") or []
            else:
                story = str(block)
                acs = []
            print(f"  [{i}] {story}")
            if acs:
                for j, ac in enumerate(acs, 1):
                    if isinstance(ac, dict):
                        t = str(ac.get("text", ""))
                        tr = str(ac.get("traces_to", "")).strip()
                    else:
                        t = str(ac)
                        tr = ""
                    line = f"      • AC {j}: {t}"
                    if tr and tr != "unspecified":
                        line += f"  [→ {tr}]"
                    print(line)
            else:
                print("      (no acceptance criteria)")
    br = artifacts.get("bug_report")
    _steps = (br.get("steps_to_reproduce") or []) if isinstance(br, dict) else []
    if isinstance(br, dict) and (
        (br.get("bug_description") or "").strip()
        or (br.get("expected_behavior") or "").strip()
        or (br.get("actual_behavior") or "").strip()
        or (isinstance(_steps, list) and len(_steps) > 0)
    ):
        print("Bug report:")
        if (br.get("bug_description") or "").strip():
            print(f"  Description: {str(br['bug_description']).strip()}")
        steps = br.get("steps_to_reproduce") or []
        if isinstance(steps, list) and steps:
            print("  Steps to reproduce:")
            for i, s in enumerate(steps, 1):
                print(f"    {i}. {s}")
        if (br.get("expected_behavior") or "").strip():
            print(f"  Expected: {str(br['expected_behavior']).strip()}")
        if (br.get("actual_behavior") or "").strip():
            print(f"  Actual: {str(br['actual_behavior']).strip()}")
    uj = artifacts.get("user_journey") or []
    if uj:
        print("User journey (top-level):")
        for i, step in enumerate(uj, 1):
            print(f"  {i}. {step}")
    elif isinstance(feats, list) and any(
        isinstance(f, dict) and (f.get("user_journey") or []) for f in feats
    ):
        print("User journey: (per feature — see under each feature above)")
    else:
        print("User journey: (none)")
    ga = artifacts.get("gap_analysis") or []
    if ga:
        print("Gap analysis (top-level):")
        for i, gap in enumerate(ga, 1):
            print(f"  {i}. {gap}")
    elif isinstance(feats, list) and any(
        isinstance(f, dict) and (f.get("gap_analysis") or []) for f in feats
    ):
        print("Gap analysis: (per feature — see under each feature above)")
    else:
        print("Gap analysis: (none)")


def _run_pipeline_for_single_requirement(requirement: RawRequirement) -> bool:
    """
    Run understanding → clarification → refinement → artifacts.
    Returns False if a fatal error should stop the program (matches prior main behavior).
    """
    print(f"\n--- Requirement Understanding (LLM extraction) [using: {get_llm_provider_and_model()}] ---")
    try:
        understood = understand_requirement(requirement)
        print(json.dumps(understood.to_dict(), indent=2))
        print("--- End ---")
    except ImportError as e:
        print(f"Error: {e}")
        return False
    except RuntimeError as e:
        print(f"Error: {e}")
        return False
    except OpenAIPermissionDeniedError as e:
        _print_403_diagnosis(e)
        return False
    except Exception as e:
        if _is_403_credits_error(e):
            _print_403_diagnosis(e)
        else:
            print(f"Understanding failed: {e}")
            if "\nRaw LLM response:" in str(e):
                print("\n(Full error above includes raw LLM response for debugging.)")
            else:
                traceback.print_exc(file=sys.stderr)
        return False

    clarified = run_clarification(
        understood,
        raw_requirement_text=requirement.text,
        open_items=requirement.open_items,
    )
    vr = validate_clarification_answers(clarified)
    if vr.has_errors:
        print("\n--- Clarification consistency check failed ---")
        for sev, msg in vr.issues:
            print(f"  [{sev.upper()}] {msg}")
        print(
            "Fix contradictory clarification answers (billing, refunds, fulfillment) and rerun."
        )
        return False
    for sev, msg in vr.issues:
        if sev == "warning":
            print(f"[Clarification warning] {msg}")
    if clarified.to_clarification_context():
        print("\n--- Clarification summary ---")
        print(clarified.to_clarification_context())
        print("--- End ---")

    print("\n--- Requirement Refinement (formal definition) ---")
    try:
        refined = refine_requirement(
            understood,
            clarification_context=clarified.to_refinement_block(),
            intake_feature_label=requirement.intake_feature_label,
            requirement_level=requirement.requirement_level,
            open_items=requirement.open_items,
        )
        print(refined.format_output())
        print("--- End ---")
    except Exception as e:
        print(f"Refinement failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return False

    print("\n--- Artifact Generation (multi-agent) ---")
    try:
        ac_fmt = normalize_acceptance_criteria_format(os.environ.get("ACCEPTANCE_CRITERIA_FORMAT"))
        artifacts = generate_all_artifacts(refined, acceptance_criteria_format=ac_fmt)
        _print_artifacts(artifacts)
        persist_cli_run(
            original_requirement=requirement.text,
            open_items=list(requirement.open_items or []),
            understood=understood,
            refined=refined,
            artifacts_dict=artifacts,
            acceptance_criteria_format=ac_fmt,
        )
        print("--- End ---")
    except Exception as e:
        print(f"Artifact generation failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return False

    return True


def main() -> None:
    requirement: RawRequirement = accept_requirement_from_console()

    if requirement.is_empty():
        print("No requirement entered. Exiting.")
        return

    print("\n--- Received requirement ---")
    print(requirement.text)
    if requirement.open_items:
        print("--- Pending discussion / clarification (routed out of feature text) ---")
        for li in requirement.open_items:
            print(f"  - {li}")
    print("--- End ---")

    units: list[NormalizedRequirementUnit] = analyze_intake(requirement.text)
    if not units:
        print("No processable requirement text. Exiting.")
        return

    if len(units) > 1:
        print(
            f"\n--- Pre-processing: {len(units)} feature units (after classification & normalize) — pipeline each ---"
        )
    else:
        lvl = units[0].requirement_level or "feature"
        print(f"\n--- Pre-processing: single unit — level: {lvl} — standard pipeline ---")

    for feat_idx, unit in enumerate(units, 1):
        if len(units) > 1:
            print("\n" + "=" * 72)
            print(f"  FEATURE {feat_idx} OF {len(units)}")
            if unit.feature_name:
                print(f"  Label: {unit.feature_name}  |  Level: {unit.requirement_level}")
            print("=" * 72)
            print("\n--- Sub-requirement text ---")
            print(unit.text)
            print("--- End sub-requirement ---\n")

        sub = RawRequirement(
            text=unit.text,
            intake_feature_label=unit.feature_name or None,
            requirement_level=unit.requirement_level or None,
            open_items=list(requirement.open_items or []),
        )
        ok = _run_pipeline_for_single_requirement(sub)
        if not ok:
            return


def _is_403_credits_error(e: Exception) -> bool:
    s = str(e).lower()
    return "403" in s and ("credits" in s or "licenses" in s or "permission" in s)


def _print_403_diagnosis(e: Exception) -> None:
    print("Understanding failed: API returned 403 (Permission Denied).")
    print()
    print("--- DIAGNOSIS ---")
    print("NOT an API key issue. Your key is valid (server accepted it).")
    print("401 = wrong/invalid key  →  you do not have this.")
    print("403 = key valid but request not allowed  →  this is what you have.")
    print()
    print("Cause: Your xAI team has no credits or licenses.")
    print("Fix: Open https://console.x.ai/ → your team → add credits/licenses.")
    print("---")


if __name__ == "__main__":
    main()
