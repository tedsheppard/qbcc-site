"""Isolated routes for the local Sopal v2 prototype.

This module intentionally does not alter existing live Sopal routes. It serves
the single-page prototype at /sopal-v2/* and exposes only /api/sopal-v2/*
endpoints for prototype AI calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
SOPAL_V2_PAGE = ROOT / "site" / "sopal-v2.html"

page_router = APIRouter(tags=["sopal-v2-page"])
router = APIRouter(prefix="/api/sopal-v2", tags=["sopal-v2"])


@page_router.get("/sopal-v2", include_in_schema=False)
@page_router.get("/sopal-v2/{path:path}", include_in_schema=False)
async def sopal_v2_page(path: str = "") -> FileResponse:
    """Serve the isolated local SPA for all /sopal-v2 routes."""
    if not SOPAL_V2_PAGE.exists():
        raise HTTPException(status_code=404, detail="Sopal v2 prototype page not found")
    return FileResponse(str(SOPAL_V2_PAGE), media_type="text/html")


class SopalV2AgentRequest(BaseModel):
    agentType: str | None = Field(default=None, max_length=80)
    mode: Literal["review", "draft"] | None = None
    message: str = Field(default="", max_length=120_000)
    files: list[dict[str, Any]] = Field(default_factory=list)
    projectContext: str | None = Field(default=None, max_length=40_000)


AGENT_LABELS: dict[str, str] = {
    "payment-claims": "Payment Claims",
    "payment-schedules": "Payment Schedules",
    "eots": "EOTs",
    "variations": "Variations",
    "delay-costs": "Delay Costs",
    "adjudication-application": "Adjudication Application",
    "adjudication-response": "Adjudication Response",
}


AGENT_INSTRUCTIONS: dict[tuple[str, str], str] = {
    ("payment-claims", "review"): (
        "Review a payment claim for potential issues under the applicable security of payment framework, "
        "especially BIF Act style requirements where relevant. Identify compliance issues, missing information, "
        "date problems, repeat claim issues, excluded or problematic amounts if applicable, and practical amendments. "
        "Do not state definitive legal conclusions beyond available information. Ask for missing documents where needed."
    ),
    ("payment-claims", "draft"): (
        "Draft a payment claim or supporting cover content based on user instructions and provided context. "
        "Include placeholders where evidence is missing. Be precise and legally careful."
    ),
    ("payment-schedules", "review"): (
        "Review a payment schedule for adequacy, timing, reasons for withholding, clarity of scheduled amount, "
        "jurisdictional and compliance risks, and whether reasons may be too vague or missing. Identify risks for "
        "both claimant and respondent where relevant."
    ),
    ("payment-schedules", "draft"): (
        "Draft a payment schedule, including scheduled amount, reasons for withholding, itemised disputed amounts, "
        "and reservation of rights where appropriate. Include placeholders for evidence."
    ),
    ("eots", "review"): (
        "Review an extension of time notice or claim against the contract requirements and general construction "
        "claims logic. Identify trigger event, notice timing, causation, critical delay, supporting documents, "
        "and time bar risks."
    ),
    ("eots", "draft"): (
        "Draft an EOT notice or EOT claim using the user's contract and project facts. Include event description, "
        "contractual basis, delay period, causation, evidence, and reservations."
    ),
    ("variations", "review"): (
        "Review a variation notice or claim. Identify whether there is a direction or change, contractual basis, "
        "notice compliance, valuation method, evidence, and risks."
    ),
    ("variations", "draft"): (
        "Draft a variation notice or claim. Include direction or change, contract clause placeholder, scope change, "
        "cost and time impact, evidence, and reservation of rights."
    ),
    ("delay-costs", "review"): (
        "Review a delay cost, prolongation, or disruption claim. Identify entitlement basis, compensable delay, "
        "causal link, quantum support, duplication risks, and evidence gaps."
    ),
    ("delay-costs", "draft"): (
        "Draft a delay cost claim with sections for entitlement, causation, delay period, quantum, supporting evidence, "
        "and reservation of rights."
    ),
    ("adjudication-application", "review"): (
        "Review draft adjudication application material for structure, jurisdictional and compliance risks, statutory "
        "timing, claim and payment schedule alignment, evidence gaps, and clarity."
    ),
    ("adjudication-application", "draft"): (
        "Draft an adjudication application submission structure and content based on provided claim, schedule, contract, "
        "and evidence. Include chronology, jurisdiction, issues, entitlement, quantum, and annexure or evidence references."
    ),
    ("adjudication-response", "review"): (
        "Review draft adjudication response material for structure, jurisdictional objections, alignment with the payment "
        "schedule, reasons not previously raised risk, evidence gaps, and clarity."
    ),
    ("adjudication-response", "draft"): (
        "Draft an adjudication response structure and content based on the payment schedule, application, contract, "
        "and evidence. Include jurisdictional objections, response to each claim item, evidentiary references, and "
        "reasons previously raised."
    ),
}


BASE_SYSTEM_PROMPT = """You are Sopal, a professional construction law and security of payment assistant.
Use clear Australian English. Be practical, precise, and legally careful.
Do not invent facts, cases, document contents, statistics, dates, or project records.
If the user has not provided enough information, identify the missing information and explain why it matters.
Do not state definitive legal conclusions beyond the information provided.
Do not claim that uploaded or selected files were read unless their text is present in the user message or project context.
Format the answer with concise headings and bullet points where helpful.
Include a short note where appropriate that Sopal assists with legal and contract analysis but does not replace professional legal advice."""


def _build_messages(payload: SopalV2AgentRequest, *, assistant_only: bool = False) -> list[dict[str, str]]:
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    agent_key = (payload.agentType or "").strip()
    mode = payload.mode or "review"
    if assistant_only:
        task_prompt = (
            "You are helping inside the Sopal v2 project assistant. Answer using only the user's typed instructions "
            "and any explicit project context provided. If no project documents are available, say so plainly."
        )
        label = "Project Assistant"
    else:
        if agent_key not in AGENT_LABELS:
            raise HTTPException(status_code=400, detail="Unknown agent type")
        if mode not in {"review", "draft"}:
            raise HTTPException(status_code=400, detail="Mode must be review or draft")
        task_prompt = AGENT_INSTRUCTIONS[(agent_key, mode)]
        label = AGENT_LABELS[agent_key]

    file_note = ""
    if payload.files:
        file_names = [str(f.get("name", "")).strip() for f in payload.files if f.get("name")]
        if file_names:
            file_note = (
                "\n\nSelected file names, not parsed by this prototype: "
                + ", ".join(file_names[:12])
                + ". Do not treat these as document contents."
            )

    context = (payload.projectContext or "").strip()
    context_block = f"\n\nProject context provided by user:\n{context}" if context else ""

    user_content = (
        f"Workspace: {label}\n"
        f"Mode: {mode.title() if not assistant_only else 'Assistant'}\n\n"
        f"User instructions and pasted text:\n{message}"
        f"{context_block}"
        f"{file_note}"
    )

    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT + "\n\n" + task_prompt},
        {"role": "user", "content": user_content},
    ]


def _complete(messages: list[dict[str, str]]) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="AI is not configured. Add OPENAI_API_KEY to .env.local or the server environment.",
        )

    from services.claim_check.llm_config import CostCapExceededError, complete

    try:
        return complete(
            messages=messages,
            reasoning_effort="medium",
            tier="default",
            max_output_tokens=2200,
            temperature=0.2,
        )
    except HTTPException:
        raise
    except CostCapExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI request failed: {exc}") from exc


@router.get("/health")
async def sopal_v2_health() -> dict[str, Any]:
    return {"ok": True, "openaiConfigured": bool(os.getenv("OPENAI_API_KEY"))}


@router.post("/agent")
async def sopal_v2_agent(payload: SopalV2AgentRequest) -> dict[str, Any]:
    result = _complete(_build_messages(payload))
    return {
        "answer": result["content"],
        "model": result.get("model"),
        "lowConfidence": result.get("low_confidence", False),
    }


@router.post("/chat")
async def sopal_v2_chat(payload: SopalV2AgentRequest) -> dict[str, Any]:
    result = _complete(_build_messages(payload, assistant_only=True))
    return {
        "answer": result["content"],
        "model": result.get("model"),
        "lowConfidence": result.get("low_confidence", False),
    }
