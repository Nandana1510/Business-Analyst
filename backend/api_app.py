"""
FastAPI server for the Vite + React frontend. Run: ``uvicorn api_app:app --reload --app-dir backend``
(or from ``backend``: ``uvicorn api_app:app --reload``).

Pipeline activity (intake, each LLM agent, refinement, artifacts, session milestones) is logged to
**stderr** under the name ``ba.pipeline`` (see ``PIPELINE_LOG_LEVEL`` in ``.env.example``).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from stages.artifact_generation import (
    AC_FORMAT_BDD,
    AC_FORMAT_DECLARATIVE,
    ARTIFACT_GENERATION_CHOICES,
    ARTIFACT_MODE_BY_LABEL,
    normalize_acceptance_criteria_format,
)
from stages.pipeline_logging import configure_pipeline_logging, pipeline_log
from stages.requirement_clarification import OTHER_OPTION_LABEL

from api_pipeline_session import PipelineSession

SESSIONS: dict[str, PipelineSession] = {}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    configure_pipeline_logging()
    pipeline_log().info(
        "API startup — pipeline steps log to stderr as [ba.pipeline] "
        "(set PIPELINE_LOG_LEVEL=DEBUG for more detail)"
    )
    yield


app = FastAPI(title="AI Business Analyst API", version="1.0.0", lifespan=_lifespan)

_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()] or ["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionBody(BaseModel):
    """Optional defaults; can be overridden on each generate request."""

    artifact_scope_label: str | None = Field(
        default=None,
        description="Display label e.g. 'All Artifacts' — maps to internal mode key.",
    )
    acceptance_criteria_format: str | None = Field(
        default=None, description="'declarative' or 'bdd'"
    )


class ClarificationSubmitBody(BaseModel):
    """Map category (slug) -> answer text; empty string means 'not answered'."""

    answers: dict[str, str] = Field(default_factory=dict)


class RegenerateGapsBody(BaseModel):
    """Exact gap lines from the current artifact output (order matches UI flattening)."""

    selected_gap_texts: list[str] = Field(default_factory=list)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    labels = [label for _, label in ARTIFACT_GENERATION_CHOICES]
    return {
        "artifact_scope_labels": labels,
        "artifact_mode_by_label": ARTIFACT_MODE_BY_LABEL,
        "acceptance_criteria_formats": [
            {"label": "Declarative", "value": AC_FORMAT_DECLARATIVE},
            {"label": "BDD (Given-When-Then)", "value": AC_FORMAT_BDD},
        ],
        "clarification_other_option_label": OTHER_OPTION_LABEL,
    }


@app.post("/api/sessions")
def create_session(body: CreateSessionBody | None = None) -> dict[str, Any]:
    body = body or CreateSessionBody()
    s = PipelineSession()
    if body.artifact_scope_label and body.artifact_scope_label in ARTIFACT_MODE_BY_LABEL:
        s.pipeline_artifact_mode = ARTIFACT_MODE_BY_LABEL[body.artifact_scope_label]
    if body.acceptance_criteria_format:
        s.pipeline_acceptance_criteria_format = normalize_acceptance_criteria_format(
            body.acceptance_criteria_format
        )
    SESSIONS[s.session_id] = s
    return {"session_id": s.session_id, "state": s.public_state()}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s.public_state()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"status": "deleted"}


@app.post("/api/sessions/{session_id}/reset")
def reset_session(session_id: str) -> dict[str, Any]:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    s.reset_pipeline()
    s.original_requirement_text = ""
    s.intake_open_items = []
    s.multi_feature_results = None
    return s.public_state()


@app.post("/api/sessions/{session_id}/generate")
async def generate(
    session_id: str,
    requirement_text: str = Form(""),
    artifact_scope_label: str = Form("All Artifacts"),
    acceptance_criteria_format: str = Form("declarative"),
    file: UploadFile | None = File(None),
) -> dict[str, Any]:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    pipeline_log().info("[%s] HTTP POST generate (file=%s)", session_id, bool(file and file.filename))
    mode = ARTIFACT_MODE_BY_LABEL.get(artifact_scope_label, "all")
    ac_fmt = normalize_acceptance_criteria_format(acceptance_criteria_format)
    file_bytes: bytes | None = None
    file_name: str | None = None
    if file is not None and file.filename:
        file_bytes = await file.read()
        file_name = file.filename
    s.generate(
        requirement_text=requirement_text or "",
        file_name=file_name,
        file_bytes=file_bytes,
        artifact_mode=mode,
        acceptance_criteria_format=ac_fmt,
    )
    return s.public_state()


@app.post("/api/sessions/{session_id}/clarification")
def submit_clarification(session_id: str, body: ClarificationSubmitBody) -> dict[str, Any]:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    pipeline_log().info("[%s] HTTP POST clarification (stage=%s)", session_id, s.clarification_cl_stage)
    ok, err = s.submit_clarification(body.answers)
    if not ok and err:
        s.error = err
    elif ok:
        s.error = None
    return s.public_state()


@app.post("/api/sessions/{session_id}/continue-empty-clarification")
def continue_empty_clarification(session_id: str) -> dict[str, Any]:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    pipeline_log().info("[%s] HTTP POST continue-empty-clarification", session_id)
    s.continue_empty_clarification()
    return s.public_state()


@app.post("/api/sessions/{session_id}/regenerate-with-gaps")
def regenerate_with_gaps(session_id: str, body: RegenerateGapsBody) -> dict[str, Any]:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    pipeline_log().info(
        "[%s] HTTP POST regenerate-with-gaps (%d selected)", session_id, len(body.selected_gap_texts or [])
    )
    s.regenerate_with_selected_gaps(body.selected_gap_texts)
    return s.public_state()
