"""Isolated routes for the local Sopal v2 prototype.

This module intentionally does not alter existing live Sopal routes. It serves
the single-page prototype at /sopal-v2/* and exposes only /api/sopal-v2/*
endpoints for prototype AI calls.
"""

from __future__ import annotations

import os
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, File, HTTPException, UploadFile
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


REVIEW_OUTPUT_FRAME = """Structure the response with these sections:
1. Executive summary
2. Issues identified
3. Compliance / entitlement analysis
4. Missing information
5. Evidence required
6. Recommended amendments
7. Suggested next steps
8. Risk rating

Be specific to the document/facts provided. If the supplied text is inadequate, say exactly what is missing."""

DRAFT_OUTPUT_FRAME = """Structure the response with these sections:
1. Draft document / letter / submission
2. Assumptions
3. Placeholders to complete
4. Evidence schedule
5. Optional alternative wording
6. Risks / notes

Draft in usable professional wording. Use bracketed placeholders where facts or evidence are missing."""


AGENT_INSTRUCTIONS: dict[tuple[str, str], str] = {
    ("payment-claims", "review"): (
        "Review a payment claim for potential issues under the applicable security of payment framework, "
        "especially BIF Act style requirements where relevant. Identify compliance issues, missing information, "
        "date problems, repeat claim issues, excluded or problematic amounts if applicable, and practical amendments. "
        "Focus on whether the document appears to be a payment claim, the claimed amount, identification of work or "
        "related goods/services, request for payment, service/date issues, reference date or claim date issues if relevant, "
        "repeat claim risks, problematic amounts, and supporting documents. "
        + REVIEW_OUTPUT_FRAME
    ),
    ("payment-claims", "draft"): (
        "Draft payment claim content and a cover letter/email if useful. Include claim items if facts are supplied, "
        "statutory wording where appropriate, and placeholders for amount, work, dates, recipient, contract, and project. "
        + DRAFT_OUTPUT_FRAME
    ),
    ("payment-schedules", "review"): (
        "Review a payment schedule for adequacy, timing, reasons for withholding, clarity of scheduled amount, "
        "jurisdictional and compliance risks, and whether reasons may be too vague or missing. Identify risks for "
        "both claimant and respondent where relevant. Focus on timing, scheduled amount, itemisation, reasons for "
        "withholding, reasons not properly raised, and likely adjudication risk. "
        + REVIEW_OUTPUT_FRAME
    ),
    ("payment-schedules", "draft"): (
        "Draft a payment schedule, including scheduled amount, reasons for withholding, itemised disputed amounts, "
        "contractual/statutory basis, evidence references, and reservation of rights where appropriate. "
        + DRAFT_OUTPUT_FRAME
    ),
    ("eots", "review"): (
        "Review an extension of time notice or claim against the contract requirements and general construction "
        "claims logic. Identify trigger event, notice timing, causation, critical delay, supporting documents, "
        "time bar risks, and whether the claim also raises variation or delay cost issues. "
        + REVIEW_OUTPUT_FRAME
    ),
    ("eots", "draft"): (
        "Draft an EOT notice or EOT claim using the user's contract and project facts. Include event description, "
        "contractual basis, delay period, causation, supporting documents, and reservations. "
        + DRAFT_OUTPUT_FRAME
    ),
    ("variations", "review"): (
        "Review a variation notice or claim. Identify whether there is a direction or change, contractual basis, "
        "scope change, notice compliance, valuation method, evidence, time/cost impact, and time bar risk. "
        + REVIEW_OUTPUT_FRAME
    ),
    ("variations", "draft"): (
        "Draft a variation notice or claim. Include direction or change, contract clause placeholder, scope change, "
        "valuation, cost and time impact, supporting evidence, and reservation of rights. "
        + DRAFT_OUTPUT_FRAME
    ),
    ("delay-costs", "review"): (
        "Review a delay cost, prolongation, or disruption claim. Identify entitlement basis, compensable delay, "
        "causal link, quantum support, duplication risks, notice compliance, and evidence gaps. "
        + REVIEW_OUTPUT_FRAME
    ),
    ("delay-costs", "draft"): (
        "Draft a delay cost claim with sections for entitlement, causation, delay period, quantum, supporting evidence, "
        "and reservation of rights. "
        + DRAFT_OUTPUT_FRAME
    ),
    ("adjudication-application", "review"): (
        "Review draft adjudication application material for structure, jurisdictional and compliance risks, statutory "
        "timing, payment claim validity, payment schedule issues, claim/schedule alignment, evidence gaps, annexures, "
        "and clarity. "
        + REVIEW_OUTPUT_FRAME
    ),
    ("adjudication-application", "draft"): (
        "Draft an adjudication application submission structure and content based on provided claim, schedule, contract, "
        "and evidence. Include chronology, jurisdiction, statutory framework, contract background, payment claim, "
        "payment schedule, issues, entitlement, quantum, anticipated objections, and annexure/evidence references. "
        + DRAFT_OUTPUT_FRAME
    ),
    ("adjudication-response", "review"): (
        "Review draft adjudication response material for structure, jurisdictional objections, alignment with the payment "
        "schedule, reasons previously raised, new reasons risk, response structure, claimant arguments, evidence gaps, "
        "and clarity. "
        + REVIEW_OUTPUT_FRAME
    ),
    ("adjudication-response", "draft"): (
        "Draft an adjudication response structure and content based on the payment schedule, application, contract, "
        "and evidence. Include jurisdictional objections, response to each claim item, evidentiary references, and "
        "reasons previously raised. "
        + DRAFT_OUTPUT_FRAME
    ),
}


BASE_SYSTEM_PROMPT = """You are Sopal, a professional construction law and security of payment assistant.
Use clear Australian English. Be practical, precise, and legally careful.
For Queensland, distinguish the Building Industry Fairness (Security of Payment) Act 2017 (Qld) ("BIF Act") from the repealed Building and Construction Industry Payments Act 2004 (Qld) ("BCIPA"). Do not call BCIPA the BIF Act. Apply the BIF Act where the user is asking about current Queensland SOPA workflows unless the provided facts point to an older BCIPA-era matter.
Do not invent facts, cases, document contents, statistics, dates, or project records.
If the user has not provided enough information, identify the missing information and explain why it matters.
Do not state definitive legal conclusions beyond the information provided.
Do not claim that uploaded or selected files were read unless their text is present in the user message or project context.
Format the answer in clean Markdown with headings, bullets, and tables where useful.
Include a short note where appropriate that Sopal assists with legal and contract analysis but does not replace professional legal advice."""


def _current_date_context() -> str:
    """Keep isolated Sopal v2 AI calls anchored to the user's local product context."""
    now = datetime.now(ZoneInfo("Australia/Brisbane"))
    return f"Current date: {now.strftime('%-d %B %Y')} (Australia/Brisbane)."


def _build_messages(payload: SopalV2AgentRequest, *, assistant_only: bool = False) -> list[dict[str, str]]:
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    agent_key = (payload.agentType or "").strip()
    mode = payload.mode or "review"
    if assistant_only:
        task_prompt = (
            "You are helping inside the Sopal v2 project assistant. Use the user's typed instructions, pasted project "
            "context, and extracted document text if provided. Distinguish contract context from project library context "
            "where possible. If no project documents are available, say so plainly, then answer from the typed instructions. "
            "For analysis questions, use: direct answer, relevant contract/SOPA issues, missing context, and next steps. "
            "For drafting questions, provide usable wording plus assumptions and placeholders."
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
                "\n\nSelected file names supplied with this request: "
                + ", ".join(file_names[:12])
                + ". Only rely on file contents if extracted text is included in the message or project context."
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
        {"role": "system", "content": BASE_SYSTEM_PROMPT + "\n\n" + _current_date_context() + "\n\n" + task_prompt},
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


@router.get("/search")
async def sopal_v2_search(
    q: str = "",
    limit: int = 20,
    offset: int = 0,
    sort: str = "relevance",
    startDate: str | None = None,
    endDate: str | None = None,
    minClaim: float | None = None,
    maxClaim: float | None = None,
) -> dict[str, Any]:
    """v2-internal decision search. Same FTS as /search_fast, no auth gate.
    Owner-controlled v2 sandbox; deferred import keeps routes/sopal_v2.py
    importable at server boot."""
    from server import con, preprocess_sqlite_query, normalize_query  # type: ignore
    import sqlite3

    q_norm = normalize_query(q or "")
    nq2 = preprocess_sqlite_query(q_norm) if q_norm else ""

    # Outer (non-FTS) filters are applied after we've already narrowed to FTS hits.
    outer_clauses: list[str] = [
        "a.decision_date IS NOT NULL",
        "a.decision_date != ''",
        "LOWER(TRIM(a.decision_date)) != 'null'",
        "a.claimant_name IS NOT NULL",
        "a.claimant_name != ''",
        "LOWER(TRIM(a.claimant_name)) != 'not specified'",
    ]
    outer_params: list[Any] = []
    if startDate:
        outer_clauses.append("a.decision_date >= ?")
        outer_params.append(startDate)
    if endDate:
        outer_clauses.append("a.decision_date <= ?")
        outer_params.append(endDate)
    if minClaim is not None:
        outer_clauses.append("CAST(a.claimed_amount AS REAL) >= ?")
        outer_params.append(minClaim)
    if maxClaim is not None:
        outer_clauses.append("CAST(a.claimed_amount AS REAL) <= ?")
        outer_params.append(maxClaim)
    outer_where = "WHERE " + " AND ".join(outer_clauses)

    if not q_norm and sort == "relevance":
        sort = "newest"
    order_clauses = {
        "newest": "ORDER BY a.decision_date DESC",
        "oldest": "ORDER BY a.decision_date ASC",
        "claim_high": "ORDER BY CASE WHEN a.claimed_amount IS NULL OR a.claimed_amount = 'N/A' OR a.claimed_amount = '' THEN -1 ELSE CAST(a.claimed_amount AS REAL) END DESC",
        "claim_low": "ORDER BY CASE WHEN a.claimed_amount IS NULL OR a.claimed_amount = 'N/A' OR a.claimed_amount = '' THEN 9999999999 ELSE CAST(a.claimed_amount AS REAL) END ASC",
        "adj_high": "ORDER BY CASE WHEN a.adjudicated_amount IS NULL OR a.adjudicated_amount = 'N/A' OR a.adjudicated_amount = '' THEN -1 ELSE CAST(a.adjudicated_amount AS REAL) END DESC",
        "adj_low": "ORDER BY CASE WHEN a.adjudicated_amount IS NULL OR a.adjudicated_amount = 'N/A' OR a.adjudicated_amount = '' THEN 9999999999 ELSE CAST(a.adjudicated_amount AS REAL) END ASC",
    }
    # bm25/rank can't be projected out of an FTS subquery either. Easiest path:
    # do a two-step search — first a tiny FTS-only query to get matched rowids
    # (and capture rank order if needed), then a regular JOIN to fetch the rows.
    fts_params: list[Any] = []
    if nq2:
        # Pull a generous slice — we only display 20 at a time, but allow the
        # outer filters (date range, claim caps) to whittle the set without
        # forcing a second round-trip. 600 hits is plenty for a paged UI.
        ranked_rows = con.execute(
            "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT 600",
            (nq2,),
        ).fetchall()
        rowids = [r[0] for r in ranked_rows]
        if not rowids:
            return {"total": 0, "items": []}
        # Capture relevance order for later sort.
        rowid_rank = {rid: idx for idx, rid in enumerate(rowids)}
    else:
        rowids = None
        rowid_rank = None

    if rowids is not None:
        placeholders = ",".join("?" for _ in rowids)
        rowid_clause = f"d.rowid IN ({placeholders})"
        rowid_params = list(rowids)
    else:
        rowid_clause = "1=1"
        rowid_params = []

    snippet_expr = (
        "snippet(fts, '<mark>', '</mark>', ' … ', 0, 30)"
        if nq2
        else "substr(d.full_text, 1, 200) || '...'"
    )

    if sort == "relevance" and nq2:
        # We'll sort in Python using rowid_rank because SQLite would otherwise
        # need a CASE-by-rowid construct to recreate bm25 ordering.
        order_clause = "ORDER BY a.decision_date DESC"
        post_sort = "relevance"
    else:
        order_clause = order_clauses.get(sort, "ORDER BY a.decision_date DESC")
        post_sort = None

    join_block = (
        "FROM fts JOIN docs_fresh d ON fts.rowid = d.rowid "
        "LEFT JOIN decision_details a ON d.ejs_id = a.ejs_id "
    )
    extra_clauses = [rowid_clause] + outer_clauses
    where_sql = "WHERE " + " AND ".join(extra_clauses)
    base_params = tuple(rowid_params + outer_params)

    try:
        total = con.execute(
            f"SELECT COUNT(DISTINCT d.rowid) {join_block} {where_sql}",
            base_params,
        ).fetchone()[0]

        if post_sort == "relevance":
            # Fetch the full set in arbitrary order, sort by relevance in Python,
            # then slice. The set is bounded by the 600-row cap above.
            fetch_sql = f"""
                SELECT DISTINCT d.rowid AS rowid, {snippet_expr} AS snippet,
                       a.claimant_name, a.respondent_name, a.adjudicator_name,
                       a.act_category, d.reference, d.pdf_path, d.ejs_id,
                       a.claimed_amount, a.adjudicated_amount,
                       a.decision_date
                {join_block}
                {where_sql}
            """
            rows = con.execute(fetch_sql, base_params).fetchall()
            rows = sorted(rows, key=lambda r: rowid_rank.get(r["rowid"], 10**9))
            rows = rows[offset:offset + limit]
        else:
            sql = f"""
                SELECT DISTINCT d.rowid AS rowid, {snippet_expr} AS snippet,
                       a.claimant_name, a.respondent_name, a.adjudicator_name,
                       a.act_category, d.reference, d.pdf_path, d.ejs_id,
                       a.claimed_amount, a.adjudicated_amount,
                       a.decision_date
                {join_block}
                {where_sql}
                {order_clause}
                LIMIT ? OFFSET ?
            """
            rows = con.execute(sql, base_params + (limit, offset)).fetchall()
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=500, detail=f"Search query failed: {exc}") from exc

    items: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["id"] = d.get("ejs_id", d.get("rowid"))
        d["claimant"] = d.get("claimant_name")
        d["respondent"] = d.get("respondent_name")
        d["adjudicator"] = d.get("adjudicator_name")
        d["decision_date_norm"] = d.get("decision_date")
        d["act"] = d.get("act_category")
        snippet_raw = r["snippet"]
        if snippet_raw and len(snippet_raw) > 350:
            truncated = snippet_raw[:300]
            last_period = truncated.rfind(".")
            last_space = truncated.rfind(" ")
            if last_period > 200:
                snippet_raw = truncated[: last_period + 1] + " ..."
            elif last_space > 200:
                snippet_raw = truncated[:last_space] + " ..."
            else:
                snippet_raw = truncated + " ..."
        d["snippet"] = snippet_raw
        items.append(d)

    return {"total": total, "items": items}


@router.post("/extract")
async def sopal_v2_extract(file: UploadFile = File(...)) -> dict[str, Any]:
    """Extract text for the local Sopal v2 workspace without persisting files."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File is too large. Limit is 25MB.")

    name = file.filename
    lower = name.lower()
    try:
        if lower.endswith(".txt"):
            text = content.decode("utf-8", errors="replace")
        elif lower.endswith(".pdf"):
            import PyPDF2

            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif lower.endswith(".docx"):
            import docx

            document = docx.Document(io.BytesIO(content))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        else:
            raise HTTPException(status_code=400, detail="Supported file types: PDF, DOCX, TXT.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not extract text from {name}: {exc}") from exc

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="No text could be extracted from this file.")
    return {"filename": name, "text": text[:120_000], "characters": len(text), "truncated": len(text) > 120_000}
