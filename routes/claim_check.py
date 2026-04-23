"""APIRouter for the /claim-check feature.

All HTTP endpoints for /claim-check live here under the /api/claim-check
prefix. The page itself (site/claim-check.html) is served by server.py's
existing catch-all static handler — this router only owns the JSON API.

Endpoints:
  GET  /api/claim-check/health   — liveness (used by frontend readiness)
  POST /api/claim-check/analyse  — run compliance checks on a document
  POST /api/claim-check/chat     — contextual chat about the document

Privacy:
  - Uploaded documents are held in memory for the duration of the request only.
  - We log metadata (filename, size, mode, IP) for rate limiting, never content.

Rate limiting (in-memory, per IP, rolls every 24h):
  - /analyse:  MAX_ANALYSES_PER_DAY
  - /chat:     MAX_CHATS_PER_DAY
Restarting the process clears counters; for a multi-process deployment the
counters are per-process. For the current single-worker Render setup that is
acceptable; later stages can move to Redis / slowapi if needed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Optional

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File, Body

log = logging.getLogger("claim_check.routes")

router = APIRouter(prefix="/api/claim-check", tags=["claim-check"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PASTE_CHARS = 200_000
MAX_ANALYSES_PER_DAY = 20
MAX_CHATS_PER_DAY = 80
WINDOW_SECONDS = 24 * 60 * 60

VALID_MODES = {
    "payment_claim_outgoing",
    "payment_claim_incoming",
    "payment_schedule_outgoing",
    "payment_schedule_incoming",
}

# {endpoint: {ip: [timestamps...]}}
_rl_lock = threading.Lock()
_rl_hits: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))


def _client_ip(req: Request) -> str:
    # Respect X-Forwarded-For set by Render's proxy.
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
        # drop old
        hits[:] = [t for t in hits if t >= cutoff]
        if len(hits) >= cap:
            retry_after = int(hits[0] + WINDOW_SECONDS - now)
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit reached ({cap}/day). Try again in ~{max(1, retry_after // 60)} minutes.",
            )
        hits.append(now)


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/analyse")
async def analyse(
    request: Request,
    mode: str = Form(...),
    file: Optional[UploadFile] = File(None),
    pasted_text: Optional[str] = Form(None),
    user_answers: Optional[str] = Form(None),  # JSON-encoded dict
) -> dict:
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode!r}")

    ip = _client_ip(request)
    _rate_limit("analyse", ip, MAX_ANALYSES_PER_DAY)

    # Local imports so this route can load even if one of these modules errors.
    from services.claim_check import extractor, rule_engine, case_law

    # --- gather document text ------------------------------------------------
    document_text = ""
    source_name = "Pasted text"
    source_size = 0

    if file is not None and file.filename:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File is too large. Max 10 MB.")
        source_name = file.filename
        source_size = len(raw)
        try:
            document_text = extractor.extract_text(file.filename, raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("Extraction failed for %s", file.filename)
            raise HTTPException(status_code=500, detail=f"Could not read document: {e}")
    elif pasted_text and pasted_text.strip():
        if len(pasted_text) > MAX_PASTE_CHARS:
            raise HTTPException(status_code=413, detail="Pasted text is too long.")
        document_text = pasted_text.strip()
        source_size = len(document_text.encode("utf-8"))
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or pasted text.")

    # Log metadata only (per privacy spec — never content).
    log.info("claim-check analyse: mode=%s ip=%s name=%s size=%d chars=%d",
             mode, ip, source_name, source_size, len(document_text))

    # --- parse user_answers --------------------------------------------------
    answers: dict[str, Any] = {}
    if user_answers:
        import json
        try:
            parsed = json.loads(user_answers)
            if isinstance(parsed, dict):
                answers = {str(k): v for k, v in parsed.items() if v not in (None, "")}
        except json.JSONDecodeError:
            pass  # ignore malformed user_answers; treat as none

    # --- run checks ----------------------------------------------------------
    try:
        result = rule_engine.run_checks(mode, document_text, answers)
    except RuntimeError as e:
        msg = str(e)
        if "OPENAI_API_KEY" in msg:
            raise HTTPException(status_code=503, detail="Analysis is temporarily unavailable (LLM not configured).")
        log.exception("Rule engine failed")
        raise HTTPException(status_code=502, detail=msg)
    except Exception as e:
        log.exception("Rule engine failed (unexpected)")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    checks = result.get("checks", [])

    # --- attach case citations (best-effort, non-fatal) ----------------------
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
        "document_text": document_text,  # returned so the chatbot can use it client-side
    }


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

    from services.claim_check import chatbot as chatbot_mod

    try:
        reply = chatbot_mod.chat(mode, document_text, check_results, history, message)
    except RuntimeError as e:
        msg = str(e)
        if "OPENAI_API_KEY" in msg:
            raise HTTPException(status_code=503, detail="Chat is temporarily unavailable (LLM not configured).")
        log.exception("Chat failed")
        raise HTTPException(status_code=502, detail=msg)
    except Exception as e:
        log.exception("Chat failed (unexpected)")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    return {"reply": reply}


@router.post("/report")
async def report() -> dict:
    # PDF report arrives in a later stage. Kept here so the frontend button
    # can be visibly disabled with a clear message rather than 404ing.
    raise HTTPException(status_code=501, detail="PDF report coming in a later stage.")
