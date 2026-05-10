"""Isolated routes for the local Sopal v2 prototype.

This module intentionally does not alter existing live Sopal routes. It serves
the single-page prototype at /sopal-v2/* and exposes only /api/sopal-v2/*
endpoints for prototype AI calls plus the per-user project persistence layer
under /api/sopal-v2/projects/*.
"""

from __future__ import annotations

import json
import os
import io
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
SOPAL_V2_PAGE = ROOT / "site" / "sopal-v2.html"

page_router = APIRouter(tags=["sopal-v2-page"])
router = APIRouter(prefix="/api/sopal-v2", tags=["sopal-v2"])


# ---------- Per-user project persistence ----------
#
# Sopal v2's SPA is local-first (projects live in the browser's localStorage)
# but a paid user expects their work to survive a browser reset and to follow
# them across devices. This block adds a light-touch persistence layer over
# the existing purchases.db sqlite database, keyed by the same purchase user
# email that the marketing site authenticates with.
#
# The client opts into syncing by sending its purchase_token JWT in the
# Authorization header. Anonymous requests are rejected so we never
# accidentally store one user's project under another user's email.

_SECRET_KEY = os.getenv("LEXIFILE_SECRET_KEY", "dev-secret-key")
_JWT_ALGORITHM = "HS256"
_USE_PERSISTENT_DISK = os.path.isdir("/var/data") and os.access("/var/data", os.W_OK)
_PURCHASES_DB_PATH = (
    "/var/data/adjudicator_purchases.db"
    if _USE_PERSISTENT_DISK
    else str(ROOT / "_local_data" / "adjudicator_purchases.db")
)
# DB init is wrapped in a try/except so that, in the worst case where the
# disk path is not writable on a particular deploy, the server still boots
# and the rest of the prototype keeps working. The persistence endpoints
# below check `_sopal_v2_con` is not None before touching the DB and return
# a 503 if persistence is offline.
_sopal_v2_con: sqlite3.Connection | None = None
try:
    os.makedirs(os.path.dirname(_PURCHASES_DB_PATH), exist_ok=True)
    _conn = sqlite3.connect(_PURCHASES_DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sopal_v2_projects (
            user_email TEXT NOT NULL,
            project_id TEXT NOT NULL,
            data TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_email, project_id)
        )
        """
    )
    _conn.execute(
        "CREATE INDEX IF NOT EXISTS sopal_v2_projects_user ON sopal_v2_projects(user_email)"
    )
    # One row per user, holding the firm-wide branding settings (firm name,
    # letterhead, footer text, logo data URL, body font, page size, accent
    # colour, heading numbering style). The shape is owned by the SPA — the
    # server treats `data` as opaque JSON capped at 1 MB so an oversized logo
    # cannot wedge persistence.
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sopal_v2_firm (
            user_email TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    _conn.commit()
    _sopal_v2_con = _conn
except Exception as _exc:  # pragma: no cover - defensive, runs once at boot
    print(f"[sopal_v2] WARNING: project persistence offline ({_exc}); endpoints will return 503.")


def _require_persistence() -> sqlite3.Connection:
    if _sopal_v2_con is None:
        raise HTTPException(status_code=503, detail="Project persistence is temporarily unavailable.")
    return _sopal_v2_con


def _current_user_email(authorization: str | None = Header(default=None)) -> str:
    """Decode the purchase_token JWT and return the user's email.

    Returns 401 with a clear message if the token is missing, malformed, or
    expired. We do not look the user up in purchase_users here because the
    JWT is signed by the same SECRET_KEY this server uses; if the signature
    is valid the email is trustworthy for the duration of the token.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Sign in to sync this project to your account.")
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization header must be 'Bearer <token>'.")
    token = parts[1].strip()
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {exc}") from exc
    email = (payload.get("sub") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=401, detail="Token is missing the 'sub' email claim.")
    return email


class SopalV2ProjectPut(BaseModel):
    """Body for PUT /projects/{id}. The whole project blob is sent up.

    The server treats the blob as opaque JSON; the SPA owns the schema.
    Cap the payload at 5 MB to keep accidental runaways out of the DB.
    """

    data: dict[str, Any]


@router.get("/projects")
def list_projects(email: str = Depends(_current_user_email)) -> dict[str, Any]:
    """Return the lightweight index of every project for the current user.

    The full data blob is NOT included to keep the response small. Use
    GET /projects/{id} to pull a single project's full content.
    """
    rows = _require_persistence().execute(
        "SELECT project_id, updated_at, length(data) AS size_bytes FROM sopal_v2_projects WHERE user_email = ? ORDER BY updated_at DESC",
        (email,),
    ).fetchall()
    return {
        "projects": [
            {"id": r["project_id"], "updatedAt": r["updated_at"], "sizeBytes": r["size_bytes"]}
            for r in rows
        ]
    }


@router.get("/projects/{project_id}")
def get_project(project_id: str, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    row = _require_persistence().execute(
        "SELECT data, updated_at FROM sopal_v2_projects WHERE user_email = ? AND project_id = ?",
        (email, project_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found for this user.")
    try:
        data = json.loads(row["data"])
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Stored project blob is corrupted.")
    return {"id": project_id, "updatedAt": row["updated_at"], "data": data}


@router.put("/projects/{project_id}")
def upsert_project(
    project_id: str,
    payload: SopalV2ProjectPut,
    email: str = Depends(_current_user_email),
) -> dict[str, Any]:
    if not project_id or len(project_id) > 128:
        raise HTTPException(status_code=400, detail="project_id must be 1 to 128 characters.")
    blob = json.dumps(payload.data, separators=(",", ":"), ensure_ascii=False)
    if len(blob.encode("utf-8")) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Project blob is over the 5 MB cap.")
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    con = _require_persistence()
    con.execute(
        """
        INSERT INTO sopal_v2_projects (user_email, project_id, data, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_email, project_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at
        """,
        (email, project_id, blob, now),
    )
    con.commit()
    return {"id": project_id, "updatedAt": now, "sizeBytes": len(blob)}


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    con = _require_persistence()
    cur = con.execute(
        "DELETE FROM sopal_v2_projects WHERE user_email = ? AND project_id = ?",
        (email, project_id),
    )
    con.commit()
    return {"deleted": cur.rowcount}


# ---------- Firm-wide branding (one row per user) ----------
#
# The firm settings (logo, letterhead, footer, fonts, page size, accent
# colour, heading numbering style) drive how the AA master document and
# the six standalone drafting agents render. Stored opaquely as JSON so
# the SPA can evolve the shape without a migration.

class SopalV2FirmPut(BaseModel):
    data: dict[str, Any]


@router.get("/firm")
def get_firm(email: str = Depends(_current_user_email)) -> dict[str, Any]:
    row = _require_persistence().execute(
        "SELECT data, updated_at FROM sopal_v2_firm WHERE user_email = ?",
        (email,),
    ).fetchone()
    if not row:
        return {"data": None, "updatedAt": None}
    try:
        data = json.loads(row["data"])
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Stored firm blob is corrupted.")
    return {"data": data, "updatedAt": row["updated_at"]}


@router.put("/firm")
def upsert_firm(payload: SopalV2FirmPut, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    blob = json.dumps(payload.data, separators=(",", ":"), ensure_ascii=False)
    # 1 MB cap is generous given the only large field is the base64 logo,
    # which we already downscale client-side to ~250 KB.
    if len(blob.encode("utf-8")) > 1_000_000:
        raise HTTPException(status_code=413, detail="Firm settings are over the 1 MB cap (downscale your logo).")
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    con = _require_persistence()
    con.execute(
        """
        INSERT INTO sopal_v2_firm (user_email, data, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_email) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at
        """,
        (email, blob, now),
    )
    con.commit()
    return {"updatedAt": now, "sizeBytes": len(blob)}


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
    projectContext: str | None = Field(default=None, max_length=120_000)
    # Review-workspace fields. When `structured` is True the model must return
    # strict JSON with the per-check breakdown the v2 UI renders.
    structured: bool = False
    reviewSubmode: str | None = Field(default=None, max_length=40)
    checks: list[str] = Field(default_factory=list)
    chatFollowup: bool = False


AGENT_LABELS: dict[str, str] = {
    "payment-claims": "Payment Claims",
    "payment-schedules": "Payment Schedules",
    "eots": "EOTs",
    "variations": "Variations",
    "delay-costs": "Delay Costs",
    "general-correspondence": "General Correspondence",
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
    ("general-correspondence", "draft"): (
        "Draft general project correspondence (letter / email / notice / RFI / show-cause / suspension / default / "
        "reservation of rights / chase-up / settlement). Identify the document type from the user's instructions, "
        "use professional Australian English suitable for a construction-contract context, and ground every factual "
        "statement in the project context if provided. Include: a clear subject line, an opening that identifies the "
        "contract / project, the substantive body, any contractual or statutory references the user supplied, and a "
        "professional sign-off block with bracketed placeholders for sender details. If specific contract clauses "
        "have not been supplied, leave bracketed placeholders rather than inventing clause numbers. "
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


REVIEW_STRUCTURED_FRAME = """You are running a structured BIF Act / SOPA review of the document the user pasted.
Return ONLY a single JSON object — no surrounding prose, no Markdown fences.

Schema:
{
  "summary": "2-4 sentence executive summary",
  "checks": [
    {
      "title": "<exact check title from the list provided>",
      "status": "pass" | "fail" | "warn" | "info",
      "detail": "concise plain-English explanation, 2-5 sentences max, naming exact wording / clauses / dates from the document where possible. If the document doesn't address this check, set status to 'info' and explain what's needed."
    }
  ],
  "recommendations": ["short imperative actions the user should take next"],
  "missing": ["specific facts / documents you need to make a firm call"]
}

Status meanings:
- pass = clearly compliant or in order
- fail = clear non-compliance / fatal defect / strong adverse risk
- warn = compliance arguable / risk worth fixing / unclear
- info = the document is silent on this and the user must supply more material

Cover EVERY check title in the order given. Do not invent extra checks. Use Australian English.
Do not add a top-level "advice" field; do not add disclaimers; do not include code fences."""

REVIEW_CHAT_FRAME = """You are answering a follow-up question about the document the user is reviewing.
Ground every answer in the document text and the project context. Be concise, practical, and quote
short snippets from the document when useful. Use Markdown."""


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
        if payload.structured and mode == "review":
            check_block = "\n".join(f"- {c}" for c in (payload.checks or []))
            submode_line = f"\nReview perspective: {payload.reviewSubmode}" if payload.reviewSubmode else ""
            task_prompt = (
                AGENT_INSTRUCTIONS[(agent_key, mode)]
                + "\n\n"
                + REVIEW_STRUCTURED_FRAME
                + submode_line
                + "\n\nCheck titles to use verbatim:\n"
                + check_block
            )
        elif payload.chatFollowup and mode == "review":
            task_prompt = REVIEW_CHAT_FRAME
        else:
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


# Drafting-workspace endpoint. Drafting agents in the v2 sidebar render a
# Word-style editor on the left and an AI chat on the right; this endpoint is
# the bridge — the client sends the current document HTML plus the user's
# instruction, the model returns the FULL updated document HTML plus a one
# or two sentence summary that's surfaced in the chat stream.
class SopalV2EditDraftRequest(BaseModel):
    agentType: str = Field(default="", max_length=80)
    currentDocumentHtml: str = Field(default="", max_length=300_000)
    message: str = Field(default="", max_length=120_000)
    projectContext: str | None = Field(default=None, max_length=120_000)


EDIT_DRAFT_SYSTEM_PROMPT = """You are Sopal Drafting, editing a construction-law / SOPA / BIF Act draft document on the user's behalf.

You will receive:
1. The CURRENT DOCUMENT (HTML) — the user's working draft, including their existing edits.
2. A user instruction describing the change(s) they want applied.
3. Optional project context (parties, contract form, contract clauses, library documents).

You must return STRICT JSON with exactly two fields:
{
  "documentHtml": "...",   // The FULL UPDATED document HTML — not a diff, not a snippet.
  "summary": "..."          // One or two sentence plain-English summary of what you changed.
}

Rules for documentHtml:
- Return the WHOLE document, not a fragment. The client replaces the editor's content with this string.
- Keep the same structural elements the user is using (h1, h2, p, table, ul, ol, strong, em, br).
- Do NOT add inline styles, scripts, or non-document elements.
- Preserve the user's existing wording wherever possible. Only change what the instruction asks for.
- Preserve [bracketed placeholders] the user has not filled in. If the user instructs you to fill one in, fill it.
- Do not invent facts (case names, sums, dates, parties) that are not in the current document or the project context.
- Use clear Australian English. Be legally careful.
- The current Queensland security-of-payment legislation is the **Building Industry Fairness (Security of Payment) Act 2017 (Qld)**, referred to as the "BIF Act". The repealed BCIPA 2004 (Qld) is no longer the current Act and should not be cited as the operative statute for any payment claim, schedule, adjudication application or response from after 17 December 2018. If the user's draft refers to "BCIPA" or the older Act, ask whether to update it rather than silently changing.
- Quote section numbers and contract clause numbers explicitly when the source provides them (e.g. "section 75 of the BIF Act", "cl 36.2(a) of the Contract"). Avoid floating assertions without a source reference.

Rules for summary:
- One or two sentences, written to the user (\"I've added a section on …\", \"Updated the claimed amount to …\").
- No code fences, no JSON, no Markdown headings.

Return only the JSON object. No surrounding prose, no code fences."""


@router.post("/agent/edit-draft")
async def sopal_v2_edit_draft(payload: SopalV2EditDraftRequest) -> dict[str, Any]:
    agent_key = (payload.agentType or "").strip()
    if agent_key not in AGENT_LABELS:
        raise HTTPException(status_code=400, detail="Unknown agent type")
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    document_html = (payload.currentDocumentHtml or "").strip()
    if not document_html:
        raise HTTPException(status_code=400, detail="Current document is required")

    label = AGENT_LABELS[agent_key]
    project_context = (payload.projectContext or "").strip()
    context_block = f"\n\nProject context provided by user:\n{project_context}" if project_context else ""

    user_content = (
        f"Drafting workspace: {label}\n\n"
        f"User instruction:\n{message}\n\n"
        f"CURRENT DOCUMENT (HTML):\n{document_html[:200_000]}"
        f"{context_block}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                BASE_SYSTEM_PROMPT
                + "\n\n"
                + _current_date_context()
                + "\n\n"
                + EDIT_DRAFT_SYSTEM_PROMPT
            ),
        },
        {"role": "user", "content": user_content},
    ]

    result = _complete(messages)
    raw = (result.get("content") or "").strip()
    # Tolerate the occasional code-fence wrapper from the model.
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()
    try:
        import json as _json

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        parsed = _json.loads(raw[start : end + 1])
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=502, detail=f"Could not parse drafting agent output: {exc}") from exc

    return {
        "documentHtml": parsed.get("documentHtml") or "",
        "summary": parsed.get("summary") or "Updated the draft.",
        "model": result.get("model"),
    }


# Complex Agent — Adjudication Application. See
# docs/complex-adjudication-application-plan.md for the full architecture.
# v1 ships two endpoints: a deterministic-shaped document parser (for the
# Stage 1 → 2 transition) and a single "engine" that handles RFI generation,
# follow-ups, and per-thread drafting.
class AAParseRequest(BaseModel):
    paymentClaimText: str = Field(default="", max_length=300_000)
    paymentScheduleText: str = Field(default="", max_length=300_000)
    s79Scenario: str = Field(default="less-than-claimed", max_length=40)
    projectMeta: dict[str, Any] = Field(default_factory=dict)


AA_S79_FRAMING: dict[str, str] = {
    "no-schedule": (
        "S 79 SCENARIO: 'No payment schedule received and no payment made' — s 79(2)(a). "
        "There is NO Payment Schedule. Treat every line item in the Payment Claim as DISPUTED for the purposes of this "
        "structured extract; psReasons should be empty for each item. Set scheduledAmount = 0. The application timing "
        "is 30 BD after the LATER of (i) the day the amount became payable under the contract; or (ii) the last day "
        "a payment schedule could have been given (15 BD after the PC). psReasonsUniverse should be empty."
    ),
    "less-than-claimed": (
        "S 79 SCENARIO: 'Schedule received — scheduled amount LESS than claimed' — s 79(2)(b). The PS schedules a "
        "lower amount than the PC and offers reasons. Capture every reason against the line item it relates to. "
        "Application timing is 30 BD after receipt of the PS."
    ),
    "scheduled-but-unpaid": (
        "S 79 SCENARIO: 'Schedule received — scheduled amount EQUAL to claim, but not paid' — s 79(2)(c). The PS "
        "scheduled the full amount (or part of it) but the respondent has not paid the scheduled amount by the due "
        "date. Mark items 'disputed' on the basis of non-payment, not on the basis of valuation. psReasons may be "
        "empty (the dispute is timing, not amount). Application timing is 20 BD after the day on which payment was "
        "due under the contract."
    ),
}


AA_PARSE_SYSTEM_PROMPT = """You are extracting a structured snapshot of a Queensland BIF Act adjudication matter from the documents the user has supplied.

Return STRICT JSON with exactly this shape:
{
  "parties":            { "claimant": "...", "respondent": "..." },
  "contractReference":  "string",
  "referenceDate":      "YYYY-MM-DD or empty string",
  "claimedAmount":      number,
  "scheduledAmount":    number,
  "lineItems": [
    {
      "label":       "short label, eg 'Variation V14'",
      "description": "one to three sentence description from the PC",
      "claimed":     number,
      "scheduled":   number,
      "psReasons":   "verbatim or close-paraphrase of the respondent's reasons for any difference, from the PS (empty for the no-schedule scenario)",
      "status":      "disputed" | "admitted" | "partial" | "jurisdictional",
      "issueType":   "variation" | "eot" | "delay-costs" | "defects" | "set-off" | "retention" | "prevention" | "scope" | "valuation" | "other"
    }
  ],
  "psReasonsUniverse":  "all of the respondent's reasons concatenated — this is the s 82(4) ceiling of arguments the respondent can later run (empty for the no-schedule scenario)",
  "warnings": [{ "code": "string", "message": "human-readable" }]
}

Rules:
- Do NOT invent line items, parties, dates, or dollar amounts. If a field isn't in the documents, leave it empty / 0 / [].
- The active s 79 scenario is given to you in the user message — apply the framing accordingly.
- If the reference date appears to be in the future, add a warning {"code":"ref-date-future","message":"..."}.
- Use Australian English. Numbers are plain numbers (no $ symbols, no commas).
- Return only the JSON object. No commentary. No code fences."""


@router.post("/complex/aa/parse-documents")
async def aa_parse_documents(payload: AAParseRequest) -> dict[str, Any]:
    pc = (payload.paymentClaimText or "").strip()
    ps = (payload.paymentScheduleText or "").strip()
    if not pc:
        raise HTTPException(status_code=400, detail="Payment Claim text is required")
    scenario = (payload.s79Scenario or "less-than-claimed").strip()
    if scenario not in AA_S79_FRAMING:
        scenario = "less-than-claimed"
    if scenario != "no-schedule" and not ps:
        raise HTTPException(status_code=400, detail="Payment Schedule text is required for this scenario")

    project_meta = payload.projectMeta or {}
    project_block = (
        f"Project: {project_meta.get('name') or '[unnamed]'}\n"
        f"Contract form: {project_meta.get('contractForm') or '[unspecified]'}\n"
        f"Claimant (project record): {project_meta.get('claimant') or ''}\n"
        f"Respondent (project record): {project_meta.get('respondent') or ''}\n"
    )

    user_content = (
        project_block
        + "\n"
        + AA_S79_FRAMING[scenario]
        + "\n\nPAYMENT CLAIM:\n"
        + pc[:120_000]
        + (
            "\n\nPAYMENT SCHEDULE:\n" + ps[:120_000]
            if ps
            else "\n\nPAYMENT SCHEDULE: (none received — s 79(2)(a) scenario)"
        )
    )

    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT + "\n\n" + _current_date_context() + "\n\n" + AA_PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    result = _complete(messages)
    raw = (result.get("content") or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    import json as _json

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        raise HTTPException(status_code=502, detail="Could not parse the model output as JSON")
    try:
        parsed = _json.loads(raw[start : end + 1])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not parse the model output: {exc}") from exc
    return parsed


# ---------------- AA engine ----------------
class AAEngineRequest(BaseModel):
    mode: str = Field(default="rfi-next", max_length=40)
    threadKind: str = Field(default="shared", max_length=20)
    threadLabel: str = Field(default="", max_length=200)
    disputeId: str | None = None
    dispute: dict[str, Any] | None = None
    rounds: list[dict[str, Any]] = Field(default_factory=list)
    existingSubmissions: str = Field(default="", max_length=120_000)
    parties: dict[str, Any] = Field(default_factory=dict)
    contractReference: str = ""
    referenceDate: str = ""
    claimedAmount: float = 0
    scheduledAmount: float = 0
    psReasonsUniverse: str = ""
    s79Scenario: str = Field(default="less-than-claimed", max_length=40)
    definitions: dict[str, Any] = Field(default_factory=dict)
    contractDocs: list[dict[str, Any]] = Field(default_factory=list)
    libraryDocs: list[dict[str, Any]] = Field(default_factory=list)
    projectMeta: dict[str, Any] = Field(default_factory=dict)
    # Cover-page extras (ABN, contact details, contract date, site address,
    # ANA, etc.). Optional. Only forwarded for context where the engine can
    # use them (introduction / background threads). Empty dict is fine.
    coverMeta: dict[str, Any] = Field(default_factory=dict)


AA_ISSUE_TYPE_RFI_HINTS: dict[str, str] = {
    "variation": (
        "Variation-specific RFI priorities (ask one focused question at a time, in this rough order):\n"
        "1. Was a written direction or instruction given (date, author, mode — email / instruction notice / drawing rev)?\n"
        "2. Was the variation directed under the contract's variation clause? Which clause?\n"
        "3. How does the varied scope differ from the original contract scope?\n"
        "4. How is the variation valued — Schedule of Rates, day-work, lump sum, contractor's rates? Is the build-up available?\n"
        "5. What is the time impact, and is a separate EOT being claimed?\n"
        "6. Time-bar / waiver risk — did the contractor give the contractually required notice within time?"
    ),
    "eot": (
        "EOT-specific RFI priorities:\n"
        "1. What is the qualifying cause of delay? Cite the contract clause that lists it.\n"
        "2. When did the contractor first become aware of the cause?\n"
        "3. Was contractual notice given within the prescribed period (date, content, recipient)?\n"
        "4. What is the impact on the critical path — what programme analysis supports the period claimed?\n"
        "5. Are there concurrent or parallel delays caused by the contractor? How are they apportioned?\n"
        "6. What mitigation steps were taken?"
    ),
    "delay-costs": (
        "Delay-cost / prolongation-specific RFI priorities:\n"
        "1. What is the entitlement basis — contract clause, breach, prevention principle?\n"
        "2. What is the compensable delay period? How is it bounded against any non-compensable periods?\n"
        "3. What quantum methodology applies — preliminaries, Hudson, Emden, measured-mile?\n"
        "4. What records support the quantum — payroll, plant logs, subcontractor invoices, off-site overhead allocation?\n"
        "5. Is there overlap with EOT or variation claims (avoid double recovery)?\n"
        "6. Was contractual notice of the cost claim given within time?"
    ),
    "defects": (
        "Defect / set-off RFI priorities:\n"
        "1. Was a defect notice issued? Date, content, particulars given.\n"
        "2. What is the precise nature and location of the alleged defect? Is photographic / inspection evidence available?\n"
        "3. How is the rectification cost quantified — quotes, actual rectification, contract rates?\n"
        "4. Was the contractor given a contractually-compliant opportunity to rectify before the respondent engaged others?\n"
        "5. Is the alleged defect actually a defect, or a design / scope dispute?"
    ),
    "set-off": (
        "Set-off RFI priorities:\n"
        "1. Contractual basis for the set-off — clause cited.\n"
        "2. Particulars of the underlying claim being set off (date, amount, basis).\n"
        "3. Notice / particulars given to the contractor under the contract before the set-off was applied.\n"
        "4. Is the set-off amount quantified or merely asserted?"
    ),
    "retention": (
        "Retention RFI priorities:\n"
        "1. Contractual basis for the retention — clause cited.\n"
        "2. Has the trigger for release of retention been met (Practical Completion, end of DLP, certificate)?\n"
        "3. Has any retention been substituted by an unconditional undertaking?\n"
        "4. Have any defects on which retention is being held been particularised?"
    ),
    "prevention": (
        "Prevention principle RFI priorities:\n"
        "1. What act or omission of the principal is alleged to have prevented timely completion?\n"
        "2. Was the act / omission a 'qualifying' cause of delay under the contract, or one for which the contract bars an extension?\n"
        "3. What programme analysis demonstrates the act / omission delayed the critical path?\n"
        "4. Was contractual notice given despite the prevention argument?"
    ),
    "scope": (
        "Scope-dispute RFI priorities:\n"
        "1. What is the relevant contract scope — drawings, specifications, schedule of rates?\n"
        "2. Why does the contractor say the disputed work is outside the scope? Why does the respondent say it is within?\n"
        "3. Is there a written instruction / direction that converts the work into a variation?"
    ),
    "valuation": (
        "Valuation RFI priorities:\n"
        "1. What rates source applies — Schedule of Rates, market rate, contractor's quote, day-work?\n"
        "2. What is the build-up showing labour / plant / materials / overheads / margin?\n"
        "3. What supporting documents are available — quotes, invoices, dockets, time sheets?\n"
        "4. Has the respondent itemised any disagreement with the build-up?"
    ),
    "other": (
        "Generic per-item RFI guidance:\n"
        "1. What is the contractual basis for the entitlement?\n"
        "2. What are the relevant facts and dates?\n"
        "3. What records or correspondence support the claim?\n"
        "4. What is the quantum methodology and supporting documents?"
    ),
}


def _aa_thread_brief(payload: AAEngineRequest) -> str:
    if payload.threadKind == "dispute" and payload.dispute:
        d = payload.dispute
        issue = (d.get("issueType") or "other").strip()
        hints = AA_ISSUE_TYPE_RFI_HINTS.get(issue, AA_ISSUE_TYPE_RFI_HINTS["other"])
        return (
            f"Thread: PER-ITEM DISPUTE — '{d.get('item') or payload.threadLabel}'\n"
            f"Issue type: {issue}\n"
            f"Status: {d.get('status') or 'disputed'}\n"
            f"Claimed: {d.get('claimed') or 0}\n"
            f"Scheduled: {d.get('scheduled') or 0}\n"
            f"Description: {d.get('description') or ''}\n"
            f"Respondent's reasons (from PS): {d.get('psReasons') or ''}\n\n"
            "STRICT SCOPE: Every RFI and every line of submissions you produce must be about THIS specific item only. "
            "Do not ask jurisdictional questions here. Do not ask about other items. Do not draft general background here.\n\n"
            f"{hints}"
        )
    if payload.threadKind == "shared" and "jurisdiction" in payload.threadLabel.lower():
        return (
            "Thread: SHARED — JURISDICTIONAL.\n\n"
            "STRICT SCOPE: Every RFI and every line of submissions you produce must be a JURISDICTIONAL question or "
            "submission only — i.e. about whether the adjudicator has jurisdiction to decide this application at all. "
            "DO NOT ask about the substantive merits of any disputed item (variations, EOTs, delay costs, defects, "
            "valuation). Those have their own per-item threads.\n\n"
            "Jurisdictional topics, in priority order:\n"
            "1. Construction contract — does the contract fall within s 64 BIF Act?\n"
            "2. Reference date — is the date relied on in the PC valid under the contract / s 67?\n"
            "3. Claimant not excluded — s 88 (excluded persons / second-tier subcontractors / commercial vs domestic).\n"
            "4. PC content compliance — did the PC identify the work, claim an amount, and request payment (s 68)?\n"
            "5. PC service — when, by what method, evidence of service.\n"
            "6. PS content & timing — when received, within s 76 window, content compliance with s 69.\n"
            "7. Application within the s 79 window — calculation of the deadline.\n"
            "8. ANA selection — which ANA, eligibility.\n"
            "Ask one focused JURISDICTIONAL question at a time."
        )
    if payload.threadKind == "shared" and ("general" in payload.threadLabel.lower() or "background" in payload.threadLabel.lower()):
        return (
            "Thread: SHARED — BACKGROUND / GENERAL.\n\n"
            "STRICT SCOPE: Every RFI here must establish background facts that frame the master document — project, "
            "parties' relationship, contract execution, key personnel, defined terms the user wants used throughout, "
            "lodgement deadline. DO NOT ask about jurisdiction (that has its own thread). DO NOT ask about the "
            "substantive merits of any disputed item (those have their own per-item threads). One focused question at a time."
        )
    return f"Thread: shared — {payload.threadLabel}."


def _aa_rounds_brief(rounds: list[dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(rounds, start=1):
        lines.append(f"RFI {i} (asked): {r.get('question') or ''}")
        lines.append(f"RFI {i} (answer): {r.get('answer') or '(unanswered)'}")
    return "\n".join(lines) if lines else "No RFIs yet."


AA_ENGINE_SYSTEM_PROMPT = """You are Sopal Complex Agent — Adjudication Application. You run the iterative lawyer workflow for a single RFI thread on a Queensland BIF Act adjudication application.

You will receive context for ONE thread at a time:
- A short brief (which thread, jurisdictional / general / per-item dispute).
- The matter context (parties, claimed / scheduled, reference date, contract reference, the respondent's full reasons universe from the PS).
- The full RFI Q&A history for this thread.
- The current draft submissions HTML for this thread (may be empty).
- The shared definitions dictionary.

You will be asked to do ONE of:
- mode = "rfi-next": Ask the next targeted RFI question for this thread. Be specific, lawyer-grade, and tailored to the thread (issue-type aware for per-item disputes). Do NOT ask multi-part questions in one shot — one focused question at a time.
- mode = "rfi-followup": A user just answered the latest RFI. Either (a) ask another follow-up RFI if you still need information, or (b) re-draft the submissions HTML for this thread now that you have enough.
- mode = "draft": Draft / re-draft the submissions HTML for this thread using everything you have, even if some RFIs are unanswered (note any gaps in the draft as bracketed placeholders).

Return STRICT JSON with this shape:
{
  "appendRfi":         "string|null",  // a new RFI question to append. null if you didn't ask one.
  "submissionsHtml":   "string",       // the FULL updated submissions HTML for THIS thread (may be empty).
  "evidenceIndex":     [{ "ref": "SOE-1", "desc": "...", "location": "para 4.1.3" }],
  "statDecContent":    "string",
  "definitions":       { "term": "definition" },
  "isReady":           true|false      // whether this thread is "drafted enough" to advance.
}

DRAFTING STYLE — voice and rhythm:
- Voice is restrained, professional and measured. Assertive on substance ("the Claimant rejects", "the Respondent's contention is plainly wrong", "is not sustainable") but never theatrical, never sarcastic, never colloquial.
- Defer to the adjudicator's role: "the Adjudicator is invited to find …", "the Adjudicator should determine …", "with respect", "the Claimant respectfully submits".
- Concede the opponent's good points where they are good: "the Claimant accepts that …", "to that extent the Claimant agrees". A measured concession strengthens the rest of the submissions.
- Australian English throughout. BIF Act not BCIPA. Do NOT call BCIPA the BIF Act.

DRAFTING STYLE — paragraph and clause craft:
- Numbered paragraphs at the top level. Use <p><strong>1.1</strong> …</p>, <p><strong>1.2</strong> …</p>. Each paragraph addresses ONE proposition. Aim for 1–4 sentences per paragraph; never wall-of-text.
- Multi-strand answers go into sub-paragraphs (a)(b)(c) and (i)(ii)(iii) using nested HTML lists or indented <p> blocks. The reader should be able to scan the structure.
- ANCHOR every assertion. Every factual claim should reference a paragraph number, contract clause, document name, statutory provision or authority — e.g. "cl 36.2(a) of the Contract", "the Payment Schedule at [4]", "s 75(2) of the BIF Act", "Tab 3 of this Application". Floating assertions are weak.
- Direct quotes from contract clauses or statutes are short, indented, and (where relevant) marked "(emphasis added)" if the Claimant has added emphasis.

DRAFTING STYLE — defined terms:
- Use the shared Definitions consistently throughout (the Claimant, the Respondent, the Contract, the Payment Claim, the Payment Schedule, the BIF Act, the Reference Date). Define a term once and reuse it — never alternate between "the contractor" and "the Claimant".
- New defined Terms you introduce should be capitalised and quoted on first use, then added to the definitions dict so other threads can reuse them.

DRAFTING STYLE — concession-management language (use sparingly, where appropriate):
- Where the Claimant has not addressed every line of the Respondent's reasoning: "Where the Claimant has not replied directly to a particular submission of the Respondent, that is not to be taken as any admission or concession."
- For matters not addressed in the interests of brevity: "In the interests of brevity the Claimant does not propose to respond to every allegation. Any matter not addressed should not be taken to have been admitted."

DRAFTING STYLE — opening signpost:
- Where the thread is per-item, open with a short Introduction (one or two paragraphs) that identifies the item, its claimed and scheduled amounts, and the Respondent's reasons (with paragraph references back to the Payment Schedule).
- For overarching threads, open with the scope of what this part will address ("In this part the Claimant addresses …").

DRAFTING STYLE — closing signpost:
- Each thread ends with a short Conclusion / Summary stating, in one or two sentences, what the Adjudicator should find on this issue. For per-item threads, restate the precise dollar figure the Claimant submits should be allowed.

Rules:
- Submissions are professional adjudication application submissions: assertive, evidence-anchored, structured around the respondent's PS reasons (s 82(4) ceiling), citation-light but precise where used.
- HEADING HIERARCHY: do NOT use <h1> or <h2> in submissionsHtml — those are reserved for the master document's top-level section numbering (e.g. '2. Jurisdiction', '4.1 Variation V14'). Use <h3> for top-level subheadings within your submissions and <h4> for any finer divisions. Do NOT repeat the section title (the master assembler supplies it).
- Do NOT use generic templates — adapt to this matter. Length and depth fluid: a thin item gets a short reply; a substantive item gets a fuller reply.
- Do NOT invent facts. If a fact isn't supplied, leave a [bracketed placeholder] in the submissions and add another RFI to fill it.
- For per-item threads: focus the submissions on THIS item only. The master assembler stitches all items together.
- For jurisdictional thread: produce a structured set of jurisdictional submissions with subheadings per s 64 / s 67 / s 68 / s 69 / s 75 / s 76 / s 79 / s 88 as applicable.
- For general thread: produce parties / background / contract / project facts.
- Definitions you introduce (defined Terms in capitalised quoted form) should also be added to the definitions dict.
- USE THE PROVIDED CONTRACT + LIBRARY DOCS. When the user has uploaded contract clauses or correspondence, quote / cite them directly in submissions where useful — e.g. 'cl 36 of the Contract provides that …' — instead of emitting [bracketed placeholders]. Only fall back to placeholders when the relevant fact genuinely isn't in any of the provided docs. Don't invent — paraphrase what is in the docs faithfully.
- Return only the JSON object. No commentary. No code fences."""


@router.post("/complex/aa/engine")
async def aa_engine(payload: AAEngineRequest) -> dict[str, Any]:
    if payload.mode not in {"rfi-next", "rfi-followup", "draft"}:
        raise HTTPException(status_code=400, detail="Unknown engine mode")

    thread_brief = _aa_thread_brief(payload)
    rounds_brief = _aa_rounds_brief(payload.rounds)
    definitions_lines = "\n".join(f"- {k}: {v}" for k, v in (payload.definitions or {}).items()) or "(none yet)"
    scenario_id = (payload.s79Scenario or "less-than-claimed").strip()
    if scenario_id not in AA_S79_FRAMING:
        scenario_id = "less-than-claimed"
    scenario_block = AA_S79_FRAMING[scenario_id]

    # Stitch the project's uploaded contract + library docs into the prompt
    # so the model can quote actual contract clauses + correspondence rather
    # than emitting [bracketed placeholders]. Total cap to keep latency sane.
    def _format_docs(docs: list[dict[str, Any]], label: str, per_doc_cap: int, total_cap: int) -> str:
        if not docs:
            return f"{label}: (none uploaded)"
        out: list[str] = [label + ":"]
        running = 0
        for d in docs:
            name = (d.get("name") or "Untitled").strip()[:200]
            text = (d.get("text") or "").strip()
            if not text:
                continue
            text = text[:per_doc_cap]
            if running + len(text) > total_cap:
                out.append(f"- {name} (omitted — total cap reached)")
                continue
            running += len(text)
            out.append(f"--- {name} ---\n{text}")
        return "\n".join(out)

    contract_block = _format_docs(payload.contractDocs, "Contract documents (uploaded by user)", 25_000, 60_000)
    library_block = _format_docs(payload.libraryDocs, "Project library (correspondence / programme / claims / schedules)", 18_000, 50_000)

    # Cover-page extras the user filled in (ABN, addresses, contract date,
    # site address, ANA). Optional. Folded into the matter context only when
    # present, so the engine can reference them in introduction / background
    # threads where useful, instead of leaving placeholders.
    cover_meta_lines: list[str] = []
    cm = payload.coverMeta or {}
    cm_label_map = [
        ("contractDate", "Contract executed on"),
        ("siteAddress", "Project / site address"),
        ("claimantAbn", "Claimant ABN"),
        ("claimantAddress", "Claimant address"),
        ("claimantContact", "Claimant contact"),
        ("claimantPhone", "Claimant phone"),
        ("claimantEmail", "Claimant email"),
        ("respondentAbn", "Respondent ABN"),
        ("respondentAddress", "Respondent address"),
        ("respondentContact", "Respondent contact"),
        ("respondentPhone", "Respondent phone"),
        ("respondentEmail", "Respondent email"),
        ("ana", "Authorised Nominating Authority"),
        ("anaReference", "ANA reference"),
        ("pcDate", "Payment claim served"),
        ("psDate", "Payment schedule served"),
        ("applicationDate", "Application date"),
    ]
    for k, label in cm_label_map:
        v = cm.get(k)
        if isinstance(v, str) and v.strip():
            cover_meta_lines.append(f"- {label}: {v.strip()[:200]}")
    cover_meta_block = (
        "Cover-page extras the user has confirmed (use in introduction / background submissions where natural; DO NOT invent these if they are not listed):\n"
        + "\n".join(cover_meta_lines)
        if cover_meta_lines else
        "Cover-page extras: (none provided. Leave [bracketed placeholders] in the draft for any field you want)"
    )

    user_content = (
        f"Mode: {payload.mode}\n\n"
        f"{thread_brief}\n\n"
        f"{scenario_block}\n\n"
        f"Matter context:\n"
        f"- Project: {payload.projectMeta.get('name') or ''}\n"
        f"- Contract form: {payload.projectMeta.get('contractForm') or ''}\n"
        f"- Claimant: {payload.parties.get('claimant') or ''}\n"
        f"- Respondent: {payload.parties.get('respondent') or ''}\n"
        f"- Contract reference: {payload.contractReference or ''}\n"
        f"- Reference date: {payload.referenceDate or ''}\n"
        f"- Claimed amount: {payload.claimedAmount}\n"
        f"- Scheduled amount: {payload.scheduledAmount}\n"
        f"- s 82(4) PS reasons universe: {payload.psReasonsUniverse[:8000] or '(empty — no PS in this scenario)'}\n\n"
        f"{cover_meta_block}\n\n"
        f"Definitions (shared):\n{definitions_lines}\n\n"
        f"{contract_block}\n\n"
        f"{library_block}\n\n"
        f"RFI history for this thread:\n{rounds_brief}\n\n"
        f"Current draft submissions HTML for this thread (may be empty):\n{payload.existingSubmissions[:60_000]}"
    )

    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT + "\n\n" + _current_date_context() + "\n\n" + AA_ENGINE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    result = _complete(messages)
    raw = (result.get("content") or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    parsed = _aa_safe_parse_engine_output(raw, payload.existingSubmissions or "")

    return {
        "appendRfi": parsed.get("appendRfi") if parsed.get("appendRfi") else None,
        "submissionsHtml": parsed.get("submissionsHtml") or payload.existingSubmissions or "",
        "evidenceIndex": parsed.get("evidenceIndex") or [],
        "statDecContent": parsed.get("statDecContent") or "",
        "definitions": parsed.get("definitions") or {},
        "isReady": bool(parsed.get("isReady")),
    }


def _aa_safe_parse_engine_output(raw: str, fallback_submissions: str) -> dict[str, Any]:
    """Parse the engine's JSON envelope, tolerating common model quirks.

    Tries strict JSON first, then progressively-more-forgiving fallbacks so
    a single malformed comma doesn't blow up the whole turn. If the model
    returned no JSON at all we treat the entire blob as submissionsHtml so
    the user at least sees the model's draft.
    """
    import json as _json
    import re as _re

    start = raw.find("{")
    end = raw.rfind("}")

    # Pass 1: strict JSON between the first { and last }.
    if start != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            return _json.loads(candidate)
        except Exception:
            pass
        # Pass 2: try removing trailing commas (a common model error).
        try:
            cleaned = _re.sub(r",\s*([}\]])", r"\1", candidate)
            return _json.loads(cleaned)
        except Exception:
            pass

    # Pass 3: degraded — treat whatever the model produced as the new
    # submissionsHtml. Strip any leading/trailing code fences. This keeps the
    # workflow flowing rather than throwing 502 on a single quote-escape slip.
    body = raw.strip()
    if body.startswith("```"):
        body = body.split("```", 2)[-1].lstrip()
        if body.startswith("json"):
            body = body[4:].lstrip()
        body = body.rstrip("`").strip()
    return {
        "appendRfi": None,
        "submissionsHtml": body or fallback_submissions or "",
        "evidenceIndex": [],
        "statDecContent": "",
        "definitions": {},
        "isReady": False,
    }


# ---------------- AA exec-summary pass ----------------
# After the per-item threads have been drafted, run one consolidated pass that
# distils the headlines into a 4-6 paragraph executive summary suitable for the
# top of the master document. Kept separate from the engine so it can be re-run
# cheaply when an item is re-drafted or a new item is added.
class AAExecSummaryRequest(BaseModel):
    parties: dict[str, Any] = Field(default_factory=dict)
    contractReference: str = ""
    referenceDate: str = ""
    claimedAmount: float = 0
    scheduledAmount: float = 0
    s79Scenario: str = Field(default="less-than-claimed", max_length=40)
    threadDigest: list[dict[str, Any]] = Field(default_factory=list)
    projectMeta: dict[str, Any] = Field(default_factory=dict)


AA_EXEC_SUMMARY_SYSTEM_PROMPT = """You are Sopal Complex Agent — Adjudication Application. You write the EXECUTIVE SUMMARY that sits near the top of an adjudication application master document.

You will receive:
- Matter context (parties, claimed/scheduled, reference date, contract).
- The active s 79 BIF Act scenario.
- A digest of every drafted thread (label, issue type, claimed, scheduled, headline of the Claimant's position).

Write a SHORT executive summary in HTML — 4 to 6 numbered paragraphs that:
1. State the application in one sentence (who, against whom, payment claim amount, scheduled amount, s 79(2)(a)/(b)/(c) basis).
2. Frame the dispute at a high level (what the Claimant is, what the project is, what the Respondent's position is at the broadest level).
3. Identify any threshold / jurisdictional issues briefly (only if the digest shows a jurisdictional thread is in play).
4. Summarise the substantive items in one or two paragraphs — group like with like (variations, EOTs, delay costs, defects). Refer to specific items by their label (e.g. "Variation V14"). Do NOT regurgitate the per-item sections; this is a tour at altitude.
5. State the relief sought — the precise dollar amount the Claimant submits the Adjudicator should determine.

Style:
- Restrained, professional, measured. Same voice and rhythm as the per-item submissions.
- Numbered HTML paragraphs <p><strong>1.</strong> …</p>.
- Use the defined Terms (the Claimant, the Respondent, the Contract, the Payment Claim, the Payment Schedule).
- Australian English. BIF Act, not BCIPA.
- HEADING HIERARCHY: do NOT emit <h1> or <h2>. The master assembler supplies the section heading. You may use a single <h3> sub-heading sparingly if the summary genuinely needs one.
- Do NOT invent facts. If a number isn't in the digest, leave it out rather than guess.

Return STRICT JSON: { "summaryHtml": "..." }
No commentary. No code fences."""


@router.post("/complex/aa/exec-summary")
async def aa_exec_summary(payload: AAExecSummaryRequest) -> dict[str, Any]:
    scenario_id = (payload.s79Scenario or "less-than-claimed").strip()
    if scenario_id not in AA_S79_FRAMING:
        scenario_id = "less-than-claimed"
    scenario_block = AA_S79_FRAMING[scenario_id]

    digest_lines: list[str] = []
    for entry in payload.threadDigest or []:
        label = (entry.get("label") or "").strip()[:200]
        kind = (entry.get("kind") or "").strip()[:40]
        issue = (entry.get("issueType") or "").strip()[:40]
        claimed = entry.get("claimed") or 0
        scheduled = entry.get("scheduled") or 0
        status = (entry.get("status") or "").strip()[:40]
        headline = (entry.get("headline") or "").strip()[:1500]
        digest_lines.append(
            f"- [{kind}] {label} (issue: {issue}, status: {status}, claimed: {claimed}, scheduled: {scheduled})\n"
            f"  Headline: {headline}"
        )
    digest_block = "\n".join(digest_lines) or "(no drafted threads supplied)"

    user_content = (
        f"Matter context:\n"
        f"- Project: {payload.projectMeta.get('name') or ''}\n"
        f"- Contract form: {payload.projectMeta.get('contractForm') or ''}\n"
        f"- Claimant: {payload.parties.get('claimant') or ''}\n"
        f"- Respondent: {payload.parties.get('respondent') or ''}\n"
        f"- Contract reference: {payload.contractReference or ''}\n"
        f"- Reference date: {payload.referenceDate or ''}\n"
        f"- Claimed amount: {payload.claimedAmount}\n"
        f"- Scheduled amount: {payload.scheduledAmount}\n\n"
        f"{scenario_block}\n\n"
        f"Drafted thread digest:\n{digest_block}"
    )

    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT + "\n\n" + _current_date_context() + "\n\n" + AA_EXEC_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    result = _complete(messages)
    raw = (result.get("content") or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    import json as _json
    import re as _re

    start = raw.find("{")
    end = raw.rfind("}")
    summary_html = ""
    if start != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            parsed = _json.loads(candidate)
            summary_html = parsed.get("summaryHtml") or ""
        except Exception:
            try:
                cleaned = _re.sub(r",\s*([}\]])", r"\1", candidate)
                parsed = _json.loads(cleaned)
                summary_html = parsed.get("summaryHtml") or ""
            except Exception:
                summary_html = ""
    if not summary_html:
        # Degraded fallback: treat the whole response as HTML so the user at
        # least sees something they can edit, rather than throwing 502.
        body = raw
        if body.startswith("```"):
            body = body.split("```", 2)[-1].lstrip()
            if body.startswith("json"):
                body = body[4:].lstrip()
            body = body.rstrip("`").strip()
        summary_html = body
    return {"summaryHtml": summary_html}


# Project-less research chat surfaced by the Research Agent in the v2 sidebar.
# Different system prompt focus from /chat (which is the project assistant) —
# this one is for general construction-law / SOPA / BIF Act questions.
class SopalV2ResearchRequest(BaseModel):
    message: str = Field(default="", max_length=120_000)
    history: list[dict[str, Any]] = Field(default_factory=list)
    jurisdiction: str | None = Field(default="qld", max_length=8)


RESEARCH_SYSTEM_PROMPT_BASE = """You are Sopal Research, a research-only assistant for Australian construction-law and security-of-payment questions.

Be practical, precise, and cite section numbers where relevant. Do NOT invent case names, decision references, or statistics — if you don't know, say so. Use clear Australian English.

For questions about specific decisions: explain that the user can use the Decision search tool in Sopal to look up real decisions, and offer to interpret the legal principles instead. Do not fabricate decision summaries.

Format the answer in clean Markdown with headings, bullets, and tables where useful. Add a short note that Sopal Research assists with legal analysis but does not replace professional legal advice."""

# Jurisdiction-specific framing. Sopal's decision corpus + structured agent
# prompts are QLD-only today; for the other states we explicitly tell the
# model it lacks integrated case data and must rely on general knowledge.
RESEARCH_JURISDICTION_FRAMING: dict[str, str] = {
    "qld": (
        "Active jurisdiction: Queensland. Apply the Building Industry Fairness (Security of Payment) Act 2017 (Qld) "
        "(\"BIF Act\") to current matters. Distinguish it from the repealed Building and Construction Industry "
        "Payments Act 2004 (Qld) (\"BCIPA\") — do NOT call BCIPA the BIF Act. Use current QLD section numbering."
    ),
    "nsw": (
        "Active jurisdiction: New South Wales. Apply the Building and Construction Industry Security of Payment Act "
        "1999 (NSW). Note: Sopal's decision corpus is QLD-only — you do NOT have access to NSW-specific decisions, "
        "so do not fabricate them. Answer from general knowledge of the NSW Act, flag any answer that would normally "
        "rely on case law as needing verification, and recommend the user check NSW Caselaw / NCAT / Supreme Court "
        "decisions directly."
    ),
    "vic": (
        "Active jurisdiction: Victoria. Apply the Building and Construction Industry Security of Payment Act 2002 "
        "(Vic). Note: Sopal's decision corpus is QLD-only — you do NOT have access to VIC-specific decisions, so do "
        "not fabricate them. Answer from general knowledge of the Vic Act, flag any answer that would normally rely "
        "on case law as needing verification, and recommend the user check VCAT / Supreme Court Vic / VBA decisions "
        "directly."
    ),
    "wa": (
        "Active jurisdiction: Western Australia. Apply the Building and Construction Industry (Security of Payment) "
        "Act 2021 (WA) for matters arising on or after 1 August 2022, and the Construction Contracts Act 2004 (WA) "
        "for older matters where it still applies. Note: Sopal's decision corpus is QLD-only — you do NOT have "
        "access to WA-specific decisions, so do not fabricate them. Answer from general knowledge of the WA regime, "
        "flag any answer that would normally rely on case law as needing verification, and recommend the user check "
        "SAT WA / Supreme Court WA decisions directly."
    ),
    "sa": (
        "Active jurisdiction: South Australia. Apply the Building and Construction Industry Security of Payment Act "
        "2009 (SA). Note: Sopal's decision corpus is QLD-only — you do NOT have access to SA-specific decisions, so "
        "do not fabricate them. Answer from general knowledge of the SA Act, flag any answer that would normally "
        "rely on case law as needing verification, and recommend the user check SACAT / Supreme Court SA decisions "
        "directly."
    ),
}


@router.post("/research")
async def sopal_v2_research(payload: SopalV2ResearchRequest) -> dict[str, Any]:
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    jur = (payload.jurisdiction or "qld").strip().lower()
    framing = RESEARCH_JURISDICTION_FRAMING.get(jur, RESEARCH_JURISDICTION_FRAMING["qld"])
    system_prompt = RESEARCH_SYSTEM_PROMPT_BASE + "\n\n" + framing

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt + "\n\n" + _current_date_context()},
    ]
    # Replay the prior turns so the model can answer follow-ups, capped to the
    # most-recent ~20 messages (the client also caps; this is defence-in-depth).
    history = payload.history[-20:] if payload.history else []
    for turn in history:
        role = (turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:60_000]})

    # If the latest message wasn't already in history, append it. Otherwise it
    # was already added via history replay.
    if not (history and (history[-1].get("role") == "user") and ((history[-1].get("content") or "").strip() == message)):
        messages.append({"role": "user", "content": message})

    result = _complete(messages)
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
    # Two-step search: first run an FTS-only query for matching rowids (FTS5
    # `rank` only works in a query directly against the FTS table). Then join
    # the metadata tables. No cap, full corpus returned — the UI paginates at
    # 10/page and Ted wants every match available for deep-browsing /
    # cross-referencing on broad queries.
    rowid_rank: dict[int, int] = {}
    rowid_snippet: dict[int, str] = {}
    rowids: list[int] | None
    if nq2:
        # Capture snippet inside the FTS-only query so MATCH is in WHERE — that's
        # the only context where snippet() is allowed to render highlights.
        snippet_call = "snippet(fts, '<mark>', '</mark>', ' … ', 0, 30)"
        try:
            ranked = con.execute(
                f"SELECT rowid, {snippet_call} AS snip FROM fts WHERE fts MATCH ? ORDER BY rank",
                (nq2,),
            ).fetchall()
        except sqlite3.OperationalError:
            ranked = con.execute(
                f"SELECT rowid, {snippet_call} AS snip FROM fts WHERE fts MATCH ?",
                (nq2,),
            ).fetchall()
        rowids = [r[0] for r in ranked]
        if not rowids:
            return {"total": 0, "items": []}
        rowid_rank = {rid: idx for idx, (rid, _s) in enumerate(ranked)}
        rowid_snippet = {rid: (snip or "") for rid, snip in ranked}
    else:
        rowids = None

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
        # Prefer the highlighted snippet captured in the FTS-only first step;
        # fall back to the placeholder snippet from the join query.
        snippet_raw = rowid_snippet.get(d.get("rowid")) or r["snippet"] or ""
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
