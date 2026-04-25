"""APIRouter for the Contract Assist feature.

Endpoints:
  POST /api/contract-assist/ingest         — upload a contract; ingest into RAG
  POST /api/contract-assist/chat           — JSON in, SSE out (streamed answer)
  POST /api/contract-assist/draft-export   — export a draft response as DOCX
  POST /api/contract-assist/clear          — drop a session's chunks
  GET  /api/contract-assist/health         — liveness

Privacy:
  - Documents held in memory ONLY. Subagent-2 retrieval uses ChromaDB
    EphemeralClient — sessions vanish with the process.
  - Per the spec: rate limits are 10 uploads / IP / day and 60 chats / IP / day.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from typing import Any, AsyncIterator

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

log = logging.getLogger("contract_assist.routes")

router = APIRouter(prefix="/api/contract-assist", tags=["contract-assist"])

MAX_CONTRACT_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_CHAT_PER_DAY = 60
MAX_INGEST_PER_DAY = 10
WINDOW_SECONDS = 24 * 60 * 60

# In-memory rate limiter (process-local; fine for single-worker Render setup,
# matches the pattern in routes/claim_check.py).
_rl_lock = threading.Lock()
_rl_hits: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))


def _client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip() or "unknown"
    if req.client:
        return req.client.host
    return "unknown"


def _rate_limit(endpoint: str, ip: str, cap: int) -> None:
    now = time.time()
    cutoff = now - WINDOW_SECONDS
    with _rl_lock:
        hits = _rl_hits[endpoint][ip]
        hits[:] = [t for t in hits if t >= cutoff]
        if len(hits) >= cap:
            retry_after = int(hits[0] + WINDOW_SECONDS - now)
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit reached ({cap}/day). Try again in ~{max(1, retry_after // 60)} minutes.",
            )
        hits.append(now)


def _safe_session_id(s: Any) -> str:
    s = (s or "").strip() if isinstance(s, str) else ""
    if not s:
        raise HTTPException(status_code=400, detail="Missing session_id.")
    # Conservative: collection names need to be reasonable.
    import re
    if not re.fullmatch(r"[A-Za-z0-9_\-]{4,64}", s):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    return s


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@router.post("/ingest")
async def ingest(
    request: Request,
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    ip = _client_ip(request)
    _rate_limit("ingest", ip, MAX_INGEST_PER_DAY)
    sid = _safe_session_id(session_id)

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_CONTRACT_BYTES:
        raise HTTPException(status_code=413, detail="Contract is too large. Max 25 MB.")

    name = (file.filename or "").lower()
    if not (name.endswith(".pdf") or name.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only PDF or DOCX accepted.")

    log.info("contract-assist ingest: ip=%s sid=%s name=%s size=%d", ip, sid, file.filename, len(raw))

    try:
        from services.contract_assist.retrieval import ingest as ingest_contract
        result = ingest_contract(file_bytes=raw, filename=file.filename, session_id=sid)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        msg = str(e)
        if "OPENAI_API_KEY" in msg:
            raise HTTPException(status_code=503, detail="Contract Assist is temporarily unavailable (LLM not configured).")
        log.exception("ingest failed (RuntimeError)")
        raise HTTPException(status_code=502, detail=msg)
    except Exception as e:
        log.exception("ingest failed")
        raise HTTPException(status_code=500, detail=f"Could not ingest contract: {e}")

    return {
        "session_id": result.get("session_id") or sid,
        "filename": result.get("filename") or file.filename,
        "page_count": result.get("page_count"),
        "chunk_count": result.get("chunk_count", 0),
        "identified_form": result.get("identified_form"),
        "source_size": len(raw),
        "elapsed_ms": result.get("elapsed_ms"),
    }


# ---------------------------------------------------------------------------
# Chat (SSE)
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(request: Request, payload: dict = Body(...)) -> StreamingResponse:
    ip = _client_ip(request)
    _rate_limit("chat", ip, MAX_CHAT_PER_DAY)

    sid = _safe_session_id(payload.get("session_id"))
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")
    history = payload.get("history") or []
    contract_meta = payload.get("contract_meta") or None
    attachments = payload.get("attachments") or []
    if not isinstance(attachments, list):
        attachments = []

    log.info("contract-assist chat: ip=%s sid=%s chars=%d attachments=%d", ip, sid, len(message), len(attachments))

    from services.contract_assist import chatbot

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev_name, ev_data in chatbot.stream_chat(
                session_id=sid,
                message=message,
                history=history,
                contract_meta=contract_meta,
                attachments=attachments,
            ):
                yield _sse(ev_name, ev_data)
        except Exception as e:
            log.exception("chat stream failed (outer)")
            yield _sse("error", {"message": str(e)})
            yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Draft export
# ---------------------------------------------------------------------------

@router.post("/draft-export")
async def draft_export(request: Request, payload: dict = Body(...)) -> Response:
    ip = _client_ip(request)
    _rate_limit("ingest", ip, MAX_INGEST_PER_DAY)  # share the upload bucket — drafts are light

    sid = _safe_session_id(payload.get("session_id"))
    draft = payload.get("draft") or {}
    if not isinstance(draft, dict):
        raise HTTPException(status_code=400, detail="Invalid draft payload.")
    kind = str(draft.get("kind") or "Draft document")
    content = str(draft.get("content") or "")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Empty draft content.")

    from services.contract_assist import draft_exporter
    citations_contract, citations_bif = draft_exporter.extract_citations(content)

    try:
        docx_bytes = draft_exporter.build_docx(
            kind=kind,
            content=content,
            citations_contract=citations_contract,
            citations_bif=citations_bif,
        )
    except Exception as e:
        log.exception("draft export failed")
        raise HTTPException(status_code=500, detail=f"Could not build DOCX: {e}")

    # ASCII-safe filename + RFC 5987 extension (same pattern as Claim Assist report).
    date_str = time.strftime("%Y-%m-%d")
    raw_stem = (kind or "draft").strip()[:60]
    safe_ascii = "".join(ch if ord(ch) < 128 and ch not in '"\\\r\n' else "_" for ch in raw_stem).strip() or "draft"
    ascii_filename = f"Sopal Assist - {safe_ascii} - {date_str}.docx"
    pretty_filename = f"Sopal Assist — {raw_stem} — {date_str}.docx"
    from urllib.parse import quote as _urlquote

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_filename}"; '
                f"filename*=UTF-8''{_urlquote(pretty_filename)}"
            ),
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# Clear session
# ---------------------------------------------------------------------------

@router.post("/clear")
async def clear(request: Request, payload: dict = Body(...)) -> dict:
    sid = _safe_session_id(payload.get("session_id"))
    try:
        from services.contract_assist.retrieval import clear as clear_session
        clear_session(sid)
    except Exception as e:
        log.warning("clear failed for %s: %s", sid, e)
    return {"ok": True}
