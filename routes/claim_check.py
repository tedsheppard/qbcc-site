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
import os
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

log = logging.getLogger("claim_check.routes")

router = APIRouter(prefix="/api/claim-check", tags=["claim-check"])

# ---------------------------------------------------------------------------
# Server-side run log (Payment Claim / Payment Schedule verifier admin tab)
# ---------------------------------------------------------------------------

_RUNS_DB_PATH = Path(os.environ.get(
    "CLAIM_CHECK_RUNS_DB",
    "/var/data/claim_check_runs.sqlite"
        if os.path.isdir("/var/data")
        else str(Path(__file__).resolve().parent.parent / "services" / "claim_check" / "runs.sqlite"),
))
_runs_con: sqlite3.Connection | None = None
_runs_lock = threading.Lock()


def _runs_con_init() -> sqlite3.Connection:
    _RUNS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_RUNS_DB_PATH), check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS claim_check_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at   REAL,
            finished_at  REAL,
            mode         TEXT,
            kind         TEXT,
            filename     TEXT,
            n_chars      INTEGER,
            n_pages      INTEGER,
            scanned      INTEGER,
            n_pass       INTEGER,
            n_warn       INTEGER,
            n_fail       INTEGER,
            n_input      INTEGER,
            summary      TEXT,
            title        TEXT,
            ip           TEXT,
            user_email   TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_ccr_started ON claim_check_runs(started_at DESC)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ccr_kind ON claim_check_runs(kind)")
    con.commit()
    return con


def _runs_con() -> sqlite3.Connection:
    global _runs_con
    if _runs_con is None:
        with _runs_lock:
            if _runs_con is None:
                _runs_con = _runs_con_init()
    return _runs_con


def _kind_for_mode(mode: str) -> str:
    if mode.startswith("payment_claim_"):
        return "claim"
    if mode.startswith("payment_schedule_"):
        return "schedule"
    return "other"


def _decode_user_email_from_request(request: Request) -> str:
    """Best-effort: decode the JWT in Authorization to surface a signed
    user email on admin run rows. Returns "" if anonymous or invalid."""
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return ""
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return ""
    try:
        from jose import jwt  # python-jose is already a dep
    except Exception:
        return ""
    secret = (
        os.getenv("LEXIFILE_SECRET_KEY")
        or os.getenv("SECRET_KEY")
        or "dev-secret-key"
    )
    alg = os.getenv("JWT_ALG") or "HS256"
    try:
        payload = jwt.decode(token, secret, algorithms=[alg])
    except Exception:
        return ""
    email = (payload.get("email") or payload.get("sub") or "").strip()
    return email if "@" in email else ""


def _log_claim_check_run(
    *,
    started_at: float,
    finished_at: float,
    mode: str,
    filename: str,
    n_chars: int,
    n_pages: int | None,
    scanned: bool,
    results: list[dict],
    summary: str,
    title: str,
    ip: str,
    user_email: str,
) -> None:
    """Insert a single run row. Best-effort — never raise into the
    user-facing response path."""
    n_pass = sum(1 for r in results if r.get("status") == "pass")
    n_warn = sum(1 for r in results if r.get("status") == "warning")
    n_fail = sum(1 for r in results if r.get("status") == "fail")
    n_input = sum(1 for r in results if r.get("status") == "input")
    try:
        _runs_con().execute(
            """INSERT INTO claim_check_runs
               (started_at, finished_at, mode, kind, filename, n_chars,
                n_pages, scanned, n_pass, n_warn, n_fail, n_input,
                summary, title, ip, user_email)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                started_at, finished_at, mode, _kind_for_mode(mode), filename,
                int(n_chars or 0),
                int(n_pages) if n_pages else None,
                1 if scanned else 0,
                n_pass, n_warn, n_fail, n_input,
                (summary or "")[:500],
                (title or "")[:120],
                ip or "",
                user_email or "",
            ),
        )
        _runs_con().commit()
    except Exception as e:
        log.warning("claim_check_runs insert failed: %s", e)


# Legacy URL redirects after the 2026-04-25 Sopal Assist pivot.
# /claim-check is the canonical URL again; bookmarks created during the brief
# Sopal Assist period are funnelled back to it.
redirect_router = APIRouter()


@redirect_router.get("/assist")
async def _redirect_legacy_assist_landing() -> RedirectResponse:
    return RedirectResponse(url="/claim-check", status_code=301)


@redirect_router.get("/assist/claim")
async def _redirect_legacy_assist_claim() -> RedirectResponse:
    return RedirectResponse(url="/claim-check", status_code=301)


@redirect_router.get("/assist/contract")
async def _redirect_legacy_assist_contract() -> RedirectResponse:
    return RedirectResponse(url="/claim-check", status_code=301)

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

    started_at = time.time()
    user_email = _decode_user_email_from_request(request)
    summary_text = rule_engine._derive_summary(mode, document_text)

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("status", {"message": "Reading the document…"})
        yield _sse("meta", {
            "source_name": source_name,
            "source_size": source_size,
            "chars": len(document_text),
            "pages": pages,
            "scanned": scanned_flag,
            "summary": summary_text,
        })

        loop = asyncio.get_event_loop()
        # Captured here so we can log the run row at the end.
        run_results: list[dict] = []

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

            run_results.append({"status": result.get("status")})

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

        # Persist the run for the admin tabs. Title is filled in
        # asynchronously by the AI-title endpoint a moment later (see
        # below) — this row holds the placeholder until then.
        _log_claim_check_run(
            started_at=started_at,
            finished_at=time.time(),
            mode=mode,
            filename=source_name or "",
            n_chars=len(document_text),
            n_pages=pages,
            scanned=scanned_flag,
            results=run_results,
            summary=summary_text or "",
            title="",
            ip=ip,
            user_email=user_email,
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable proxy buffering if present
        },
    )


# ---------------------------------------------------------------------------
# Recheck — re-run a subset of checks against new answers (no re-extraction)
# ---------------------------------------------------------------------------

@router.post("/recheck")
async def recheck(request: Request, payload: dict = Body(...)) -> dict:
    ip = _client_ip(request)
    _rate_limit("analyse", ip, MAX_ANALYSES_PER_DAY)

    mode = payload.get("mode")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode!r}")
    document_text = str(payload.get("document_text") or "")
    user_answers = payload.get("user_answers") or {}
    if not isinstance(user_answers, dict):
        user_answers = {}
    raw_ids = payload.get("check_ids") or []
    if not isinstance(raw_ids, list):
        raw_ids = []
    check_ids = {str(x) for x in raw_ids if x}

    from services.claim_check import case_law, llm_config, rule_engine, rules_parser
    rules = rules_parser.rules_for_mode(mode)
    if check_ids:
        rules = [r for r in rules if r["id"] in check_ids]
    if not rules:
        return {"checks": []}

    results: list[dict[str, Any]] = []
    for rule in rules:
        try:
            res = rule_engine.run_single_rule(mode, rule, document_text, user_answers)
        except llm_config.CostCapExceededError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            log.exception("recheck failed for %s", rule.get("id"))
            res = {
                "id": rule["id"],
                "status": "warning",
                "status_summary": rule_engine.STATUS_SUMMARY["warning"],
                "title": rule["title"],
                "section": rule.get("act_reference", ""),
                "explanation": f"Recheck failed: {e}",
                "quote": "",
                "reasoning_trace": "",
                "query": rule.get("search_query") or rule["title"],
            }
        # Best-effort citations.
        try:
            q = res.get("query") or res.get("title") or ""
            if q:
                res["decisions"] = case_law.relevant_decisions(q, limit=3)
        except Exception:
            pass
        results.append(res)

    return {"checks": results}


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
    user_answers = payload.get("user_answers") or {}
    if not isinstance(user_answers, dict):
        user_answers = {}

    from services.claim_check import chatbot as chatbot_mod, llm_config

    log.info(
        "claim-check chat: mode=%s msg_len=%d doc_len=%d hist=%d checks=%d answers=%d",
        mode, len(message), len(document_text or ""),
        len(history) if isinstance(history, list) else 0,
        len(check_results) if isinstance(check_results, list) else 0,
        len(user_answers),
    )
    try:
        reply = chatbot_mod.chat(mode, document_text, check_results, history, message, user_answers=user_answers)
    except llm_config.CostCapExceededError as e:
        log.warning("claim-check chat: cost cap exceeded — %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        msg = str(e)
        log.warning("claim-check chat: RuntimeError — %s", msg)
        if "OPENAI_API_KEY" in msg:
            raise HTTPException(status_code=503, detail="Chat is temporarily unavailable (LLM not configured).")
        # "All models in chain failed" surfaces here. Pass the detail
        # through so the frontend can show something actionable.
        raise HTTPException(status_code=502, detail=msg[:240])
    except Exception as e:
        log.exception("claim-check chat: unexpected failure")
        raise HTTPException(status_code=500, detail=f"Chat failed: {type(e).__name__}: {str(e)[:200]}")

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
# Contract scan (Section 4) — upload a contract, return reference-date clauses
# ---------------------------------------------------------------------------

MAX_CONTRACT_BYTES = 20 * 1024 * 1024  # 20 MB per spec Section 4


@router.post("/contract-scan")
async def contract_scan(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    ip = _client_ip(request)
    _rate_limit("analyse", ip, MAX_ANALYSES_PER_DAY)

    raw = await file.read()
    if len(raw) > MAX_CONTRACT_BYTES:
        raise HTTPException(status_code=413, detail="Contract is too large. Max 20 MB.")
    name = (file.filename or "").lower()
    if not any(name.endswith(ext) for ext in (".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Upload the contract as PDF, DOCX, or TXT.")

    from services.claim_check import contract_scanner, extractor
    try:
        text, _extras = extractor.extract_rich(file.filename, raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Contract extraction failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Could not read contract: {e}")

    try:
        result = contract_scanner.find_reference_date_clauses(text)
    except Exception as e:
        log.exception("Contract scan failed")
        raise HTTPException(status_code=502, detail=f"Contract scan failed: {e}")

    return {
        "filename": file.filename,
        "size": len(raw),
        "chars": len(text),
        **result,
    }


# ---------------------------------------------------------------------------
# Session-title generator (used by the SopalAI sidebar Recents)
# ---------------------------------------------------------------------------

@router.post("/title")
async def generate_session_title(request: Request, payload: dict = Body(...)) -> dict:
    """Generate a 3-6 word title for a verify session, given mode + a
    snippet of the document (and optionally the analysis summary).
    Cheap default-tier call. Used by the SopalAI sidebar Recents."""
    mode = (payload.get("mode") or "").strip()
    doc = (payload.get("document_text") or "")[:1500]
    summary = (payload.get("summary") or "")[:500]
    filename = (payload.get("filename") or "").strip()
    if not mode and not doc and not summary and not filename:
        return {"title": "Claim check"}

    mode_label = {
        "payment_claim_serving":    "claim being served",
        "payment_claim_received":   "claim received",
        "payment_schedule_giving":  "schedule being given",
        "payment_schedule_received":"schedule received",
    }.get(mode, mode or "claim check")

    sys = (
        "You write concise 3-6 word labels for security-of-payment "
        "compliance-check sessions. The label must identify the parties "
        "or project where possible, then the document type. No quotes. "
        "No trailing punctuation. Example outputs: 'Devcon claim — Civils', "
        "'MWB schedule — Stage 2', 'Acme progress claim 17'."
    )
    user_parts = [f"Mode: {mode_label}"]
    if filename: user_parts.append(f"Filename: {filename}")
    if doc:      user_parts.append(f"Document excerpt:\n---\n{doc}\n---")
    if summary:  user_parts.append(f"Analyst summary:\n{summary}")
    user_parts.append("Return ONLY the label text, nothing else.")
    user = "\n\n".join(user_parts)

    from services.claim_check import llm_config
    try:
        resp = llm_config.complete(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user",   "content": user},
            ],
            reasoning_effort="low",
            tier="default",
            max_output_tokens=300,
        )
    except llm_config.CostCapExceededError as e:
        raise HTTPException(429, str(e))
    except Exception as e:
        log.exception("title generation failed")
        return {"title": ""}

    title = (resp.get("content") or "").strip().strip('"').strip("'")
    title = title.split("\n")[0].strip()
    if len(title) > 80:
        title = title[:77].rstrip() + "…"

    # Best-effort: stamp this title onto the most recent untitled run
    # row for the same mode (so the admin tabs show a meaningful label).
    if title and mode:
        try:
            _runs_con().execute(
                """UPDATE claim_check_runs SET title = ?
                   WHERE id = (
                       SELECT id FROM claim_check_runs
                       WHERE mode = ? AND (title IS NULL OR title = '')
                       ORDER BY started_at DESC LIMIT 1
                   )""",
                (title[:120], mode),
            )
            _runs_con().commit()
        except Exception as e:
            log.info("failed to attach title to run row: %s", e)

    return {"title": title or ""}


# ---------------------------------------------------------------------------
# Admin: list run rows for the /admin Payment Claim / Schedule tabs
# ---------------------------------------------------------------------------

_MODE_LABELS_ADMIN = {
    "payment_claim_serving":     "Claim — about to serve",
    "payment_claim_received":    "Claim — received",
    "payment_schedule_giving":   "Schedule — about to give",
    "payment_schedule_received": "Schedule — received",
}


def _require_admin_email(request: Request) -> str:
    """Mirror of bif_research's admin gate — same JWT secret, same email
    allow-list. Raises 401/403 otherwise."""
    email = _decode_user_email_from_request(request)
    if not email:
        raise HTTPException(401, "Sign in required")
    allow = {
        e.strip().lower()
        for e in os.environ.get("BIF_ADMIN_EMAILS", "edwardsheppard5@gmail.com").split(",")
        if e.strip()
    }
    if email.lower() not in allow:
        raise HTTPException(403, "Admin only")
    return email.lower()


@router.get("/admin/runs")
def admin_list_runs(
    request: Request,
    kind: str = Query("claim", regex="^(claim|schedule|all)$"),
    limit: int = Query(500, ge=1, le=2000),
) -> dict:
    """Return recent verifier runs for the admin tabs. `kind` filters
    Payment Claim runs vs Payment Schedule runs (or all)."""
    _require_admin_email(request)
    where = "" if kind == "all" else "WHERE kind = ?"
    params: tuple = () if kind == "all" else (kind,)
    rows = _runs_con().execute(
        f"""SELECT id, started_at, finished_at, mode, kind, filename,
                   n_chars, n_pages, scanned, n_pass, n_warn, n_fail,
                   n_input, summary, title, ip, user_email
            FROM claim_check_runs
            {where}
            ORDER BY started_at DESC
            LIMIT ?""",
        (*params, limit),
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "started_at": r[1],
            "finished_at": r[2],
            "mode": r[3],
            "mode_label": _MODE_LABELS_ADMIN.get(r[3] or "", r[3] or ""),
            "kind": r[4],
            "filename": r[5] or "",
            "n_chars": r[6] or 0,
            "n_pages": r[7],
            "scanned": bool(r[8]),
            "n_pass": r[9] or 0,
            "n_warn": r[10] or 0,
            "n_fail": r[11] or 0,
            "n_input": r[12] or 0,
            "summary": r[13] or "",
            "title": r[14] or "",
            "ip": r[15] or "",
            "user_email": r[16] or "",
        })
    return {"runs": out, "n": len(out)}


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
    raw_stem = (source_name or "document").rsplit(".", 1)[0][:60]
    # HTTP headers must be ASCII per RFC 7230. Strip non-ASCII for the
    # Content-Disposition header value while keeping a friendly filename.
    safe_ascii = "".join(ch if ord(ch) < 128 and ch not in '"\\\r\n' else "_" for ch in raw_stem).strip() or "document"
    ascii_filename = f"Sopal Claim Check - {safe_ascii} - {date_str}.pdf"
    # Also offer a Unicode filename via the RFC 5987 filename* extension so
    # the original characters survive when the browser supports it.
    from urllib.parse import quote as _urlquote
    pretty_filename = f"Sopal Claim Check — {raw_stem} — {date_str}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_filename}"; '
                f"filename*=UTF-8''{_urlquote(pretty_filename)}"
            ),
            "Cache-Control": "no-store",
        },
    )
