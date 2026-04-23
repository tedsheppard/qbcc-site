"""APIRouter for the /claim-check feature.

Endpoints:
  GET  /api/claim-check/health         — liveness
  GET  /api/claim-check/checks         — list of checks per mode (from rules.md)
  POST /api/claim-check/analyse        — run extraction + all checks (non-streaming)
  POST /api/claim-check/analyse-stream — same, but SSE-streamed progress + results
  POST /api/claim-check/chat           — contextual chat
  GET  /api/claim-check/qbcc-search    — proxy to Queensland CKAN QBCC register
  POST /api/claim-check/preview        — server-side preview helper for DOCX

Privacy: documents held in memory only for the request. Metadata (filename,
size, mode, IP) is logged for rate limiting; content is never logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("claim_check.routes")

router = APIRouter(prefix="/api/claim-check", tags=["claim-check"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per spec Section 1
MAX_PASTE_CHARS = 200_000
MAX_ANALYSES_PER_DAY = 20
MAX_CHATS_PER_DAY = 80
MAX_QBCC_PER_DAY = 200
WINDOW_SECONDS = 24 * 60 * 60

VALID_MODES = {
    "payment_claim_serving",
    "payment_claim_received",
    "payment_schedule_giving",
    "payment_schedule_received",
}

# Rate limiting (in-memory per-process).
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


# ---------------------------------------------------------------------------
# Health + checks
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/checks")
async def list_checks(mode: str = Query(...)) -> dict:
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode!r}")
    from services.claim_check import rules_parser
    try:
        items = rules_parser.checks_summary_for_mode(mode)
    except rules_parser.RulesParseError as e:
        raise HTTPException(status_code=500, detail=f"Rules file error: {e}")
    return {"mode": mode, "checks": items}


# ---------------------------------------------------------------------------
# Analyse (non-streaming)
# ---------------------------------------------------------------------------

async def _ingest_document(
    file: UploadFile | None,
    pasted_text: str | None,
) -> tuple[str, str, int, dict[str, Any]]:
    """Returns (document_text, source_name, source_size, extras).

    ``extras`` may include structured XLSX data under key "xlsx_structure".
    """
    from services.claim_check import extractor

    if file is not None and file.filename:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File is too large. Max 50 MB.")
        name = file.filename
        try:
            text, extras = extractor.extract_rich(name, raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("Extraction failed for %s", name)
            raise HTTPException(status_code=500, detail=f"Could not read document: {e}")
        return text, name, len(raw), extras

    if pasted_text and pasted_text.strip():
        if len(pasted_text) > MAX_PASTE_CHARS:
            raise HTTPException(status_code=413, detail="Pasted text is too long.")
        text = pasted_text.strip()
        return text, "Pasted text", len(text.encode("utf-8")), {}

    raise HTTPException(status_code=400, detail="Provide either a file or pasted text.")


@router.post("/analyse")
async def analyse(
    request: Request,
    mode: str = Form(...),
    file: Optional[UploadFile] = File(None),
    pasted_text: Optional[str] = Form(None),
    user_answers: Optional[str] = Form(None),
) -> dict:
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode!r}")

    ip = _client_ip(request)
    _rate_limit("analyse", ip, MAX_ANALYSES_PER_DAY)

    document_text, source_name, source_size, _extras = await _ingest_document(file, pasted_text)

    log.info(
        "claim-check analyse: mode=%s ip=%s name=%s size=%d chars=%d",
        mode, ip, source_name, source_size, len(document_text),
    )

    answers: dict[str, Any] = {}
    if user_answers:
        try:
            parsed = json.loads(user_answers)
            if isinstance(parsed, dict):
                answers = {str(k): v for k, v in parsed.items() if v not in (None, "")}
        except json.JSONDecodeError:
            pass

    from services.claim_check import case_law, llm_config, rule_engine

    try:
        result = rule_engine.run_checks(mode, document_text, answers)
    except llm_config.CostCapExceededError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        msg = str(e)
        if "OPENAI_API_KEY" in msg:
            raise HTTPException(status_code=503, detail="Analysis is temporarily unavailable (LLM not configured).")
        raise HTTPException(status_code=502, detail=msg)
    except Exception as e:
        log.exception("Rule engine failed (unexpected)")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    checks = result.get("checks", [])

    # Best-effort citations per check.
    try:
        for c in checks:
            q = c.get("query") or c.get("title") or ""
            if q:
                c["decisions"] = case_law.relevant_decisions(q, limit=3)
    except Exception:
        log.warning("Case law attachment failed (continuing)", exc_info=True)

    return {
        "mode": mode,
        "source_name": source_name,
        "source_size": source_size,
        "chars": len(document_text),
        "summary": result.get("summary", ""),
        "checks": checks,
        "document_text": document_text,
    }


# ---------------------------------------------------------------------------
# Analyse (streaming)
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/analyse-stream")
async def analyse_stream(
    request: Request,
    mode: str = Form(...),
    file: Optional[UploadFile] = File(None),
    pasted_text: Optional[str] = Form(None),
    user_answers: Optional[str] = Form(None),
) -> StreamingResponse:
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode!r}")

    ip = _client_ip(request)
    _rate_limit("analyse", ip, MAX_ANALYSES_PER_DAY)

    document_text, source_name, source_size, extras = await _ingest_document(file, pasted_text)

    answers: dict[str, Any] = {}
    if user_answers:
        try:
            parsed = json.loads(user_answers)
            if isinstance(parsed, dict):
                answers = {str(k): v for k, v in parsed.items() if v not in (None, "")}
        except json.JSONDecodeError:
            pass

    from services.claim_check import case_law, llm_config, rule_engine, rules_parser

    rules = rules_parser.rules_for_mode(mode)

    pages = extras.get("pages") or None
    scanned_flag = bool(extras.get("scanned"))

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("status", {"message": "Reading the document…"})
        yield _sse("meta", {
            "source_name": source_name,
            "source_size": source_size,
            "chars": len(document_text),
            "pages": pages,
            "scanned": scanned_flag,
            "summary": rule_engine._derive_summary(mode, document_text),
        })

        loop = asyncio.get_event_loop()

        for rule in rules:
            yield _sse("status", {"message": f"Checking {rule.get('act_reference', '')} — {rule['title']}…"})
            await asyncio.sleep(0)  # cooperate with the event loop so the status flushes
            try:
                result = await loop.run_in_executor(
                    None, rule_engine.run_single_rule, mode, rule, document_text, answers
                )
            except llm_config.CostCapExceededError as e:
                yield _sse("error", {"message": str(e)})
                return
            except Exception as e:
                log.exception("streaming rule failed for %s", rule.get("id"))
                result = {
                    "id": rule["id"],
                    "status": "warning",
                    "status_summary": rule_engine.STATUS_SUMMARY["warning"],
                    "title": rule["title"],
                    "section": rule.get("act_reference", ""),
                    "explanation": f"Automated check could not run: {e}",
                    "quote": "",
                    "query": rule.get("search_query") or rule["title"],
                }

            # Optional citation surfacing (best-effort; fast).
            yield _sse("status", {"message": "Searching Sopal for relevant decisions…"})
            try:
                q = result.get("query") or result.get("title") or ""
                if q:
                    result["decisions"] = await loop.run_in_executor(
                        None, case_law.relevant_decisions, q, 3
                    )
            except Exception:
                pass

            yield _sse("check_result", result)

        yield _sse("status", {"message": "Compiling analysis…"})
        yield _sse("complete", {"document_text": document_text})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable proxy buffering if present
        },
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(request: Request, payload: dict = Body(...)) -> dict:
    ip = _client_ip(request)
    _rate_limit("chat", ip, MAX_CHATS_PER_DAY)

    mode = payload.get("mode")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode!r}")

    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")

    document_text = str(payload.get("document_text") or "")
    history = payload.get("history") or []
    check_results = payload.get("checks") or []

    from services.claim_check import chatbot as chatbot_mod, llm_config

    try:
        reply = chatbot_mod.chat(mode, document_text, check_results, history, message)
    except llm_config.CostCapExceededError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        msg = str(e)
        if "OPENAI_API_KEY" in msg:
            raise HTTPException(status_code=503, detail="Chat is temporarily unavailable (LLM not configured).")
        raise HTTPException(status_code=502, detail=msg)
    except Exception as e:
        log.exception("Chat failed (unexpected)")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    return {"reply": reply}


# ---------------------------------------------------------------------------
# QBCC CKAN lookup
# ---------------------------------------------------------------------------

@router.get("/qbcc-search")
async def qbcc_search(request: Request, q: str = Query(..., min_length=2, max_length=120)) -> dict:
    ip = _client_ip(request)
    _rate_limit("qbcc", ip, MAX_QBCC_PER_DAY)

    from services.claim_check import qbcc
    try:
        results = qbcc.search(q, limit=10)
        return {"query": q, "results": results, "stale_warning": qbcc.STALE_NOTICE}
    except qbcc.QBCCUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# DOCX preview (server-side conversion)
# ---------------------------------------------------------------------------

@router.post("/preview")
async def preview(
    request: Request,
    file: UploadFile = File(...),
) -> JSONResponse:
    """Convert a DOCX to PDF for the browser viewer, if LibreOffice is available.

    Returns JSON:
        {"kind": "pdf", "bytes_base64": "...", "bytes": <len>}
        — or —
        {"kind": "unavailable", "reason": "LibreOffice not installed"}
    """
    ip = _client_ip(request)
    _rate_limit("analyse", ip, MAX_ANALYSES_PER_DAY)  # same bucket

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large. Max 50 MB.")
    name = (file.filename or "").lower()
    if not name.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only DOCX is accepted by this endpoint.")

    from services.claim_check import docx_to_pdf
    result = docx_to_pdf.convert(raw)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# PDF report — streamed, never persisted
# ---------------------------------------------------------------------------

from fastapi.responses import Response

@router.post("/report")
async def report(request: Request, payload: dict = Body(...)) -> Response:
    ip = _client_ip(request)
    _rate_limit("analyse", ip, MAX_ANALYSES_PER_DAY)  # same bucket as analysis

    mode = payload.get("mode")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode!r}")

    source_name = payload.get("source_name") or payload.get("document_name") or "document"
    summary = payload.get("summary") or ""
    checks = payload.get("checks") or []
    user_answers = payload.get("user_answers") or {}
    check_input_labels = payload.get("check_input_labels") or {}

    if not isinstance(checks, list):
        raise HTTPException(status_code=400, detail="checks must be a list")
    if not checks:
        raise HTTPException(status_code=400, detail="Nothing to report on yet — run an analysis first.")

    from services.claim_check import report_generator

    try:
        pdf_bytes = report_generator.build_report_pdf(
            mode=mode,
            source_name=source_name,
            summary=summary,
            checks=checks,
            user_answers=user_answers,
            check_input_labels=check_input_labels,
        )
    except Exception as e:
        log.exception("PDF report build failed")
        raise HTTPException(status_code=500, detail=f"Could not build PDF report: {e}")

    date_str = time.strftime("%Y-%m-%d")
    safe_name = (source_name or "document").rsplit(".", 1)[0][:60].replace('"', "")
    filename = f"Sopal Claim Check — {safe_name} — {date_str}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )
