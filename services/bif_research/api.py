"""FastAPI server for Sopal BIF Research.

Endpoints:
  POST /api/ask                         answer a question (streaming SSE)
  GET  /api/conversations               list recent conversations
  GET  /api/conversations/{id}          load one
  POST /api/conversations               create new (returns id)
  GET  /api/sources/{chunk_id}          fetch a source chunk by id
  GET  /api/health                      diagnostics

  GET  /                                serves the SPA shell
  GET  /static/...                      static assets

Run:
    OPENAI_API_KEY=sk-... uvicorn services.bif_research.api:app --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import budget
from .indexer import CHUNKS_DB, BM25_PATH, CHROMA_PATH
from . import llm_config, name_index
from . import planner as planner_module
from .hard_pipeline import HardQuestionPipeline
from .pipeline import (
    FullPipeline,
    NaivePipeline,
    _format_history_for_answerer,
    _rewrite_with_history,
    route_question,
)


_TITLE_SYSTEM_PROMPT = (
    "Return a concise 3-6 word title for this Queensland construction-law "
    "research question. Title-case. No punctuation. No quotes. Output ONLY "
    "the title text — no preamble, no explanation."
)


def _generate_chat_title(question: str) -> str:
    """Generate a short title for a new conversation. Falls back to a
    truncated copy of the question if the LLM call fails. Cheap chain."""
    try:
        text, _ = llm_config.complete_chat(
            messages=[
                {"role": "system", "content": _TITLE_SYSTEM_PROMPT},
                {"role": "user", "content": question[:500]},
            ],
            kind="default",
            operation="chat-title",
            max_output_tokens=400,
        )
        title = (text or "").strip().strip('"').strip("'").strip()
        # Single line, no newlines
        title = title.split("\n")[0].strip()
        if 0 < len(title) <= 80:
            return title
    except Exception as e:
        log.info(f"chat-title generation failed: {e}")
    # Fallback: truncated question
    short = question.strip().split("\n")[0]
    if len(short) > 60:
        short = short[:57].rstrip() + "…"
    return short


def _resolved_case_citations(named_authorities: list[str]) -> list[dict]:
    """Resolve planner-named authorities to citation labels for the
    reading_cases status event. Each entry: {input, citation, case_id,
    case_name}. Misses get marked with case_id=None."""
    out: list[dict] = []
    seen_case_ids: set[str] = set()
    for name in named_authorities or []:
        m = name_index.lookup_case(name)
        if m and m.case_id not in seen_case_ids:
            seen_case_ids.add(m.case_id)
            out.append({
                "input": name,
                "case_id": m.case_id,
                "case_name": m.case_name,
                "citation": m.citation or m.case_name,
                "in_corpus": True,
            })
        elif not m:
            out.append({
                "input": name,
                "case_id": None,
                "case_name": "",
                "citation": name,
                "in_corpus": False,
            })
    return out

log = logging.getLogger("bif_research.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

# Conversations DB lives on the persistent disk in production. Override via
# BIF_CONV_DB env var. Falls back to the in-package store/ for local dev.
_default_conv_db = ROOT / "store" / "conversations.sqlite"
CONV_DB = Path(os.environ.get("BIF_CONV_DB", str(_default_conv_db)))

# Anonymous-user lifetime question cap (no sign-in required)
ANON_QUESTION_LIMIT = int(os.environ.get("BIF_ANON_LIMIT", "4"))
# Signed-in user per-UTC-day cap before requiring enterprise contact
SIGNED_DAILY_LIMIT = int(os.environ.get("BIF_SIGNED_DAILY_LIMIT", "30"))
# JWT secret + algorithm — must match the values server.py uses to sign
# `purchase_token` so we can verify it without a circular import. server.py
# reads `LEXIFILE_SECRET_KEY` and falls back to "dev-secret-key"; we mirror
# that exactly so signed users get recognised in production.
JWT_SECRET = (
    os.environ.get("LEXIFILE_SECRET_KEY")
    or os.environ.get("SECRET_KEY")
    or os.environ.get("JWT_SECRET")
    or "dev-secret-key"
)
JWT_ALG = os.environ.get("JWT_ALG", "HS256")


# ---------------------------------------------------------------------------
# Conversation store
# ---------------------------------------------------------------------------

def init_conv_db() -> sqlite3.Connection:
    CONV_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(CONV_DB), check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at REAL,
            updated_at REAL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content_json TEXT,
            created_at REAL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)")
    # Per-query cost ledger. One row per /api/ask call.
    # Not surfaced to the frontend; admin endpoints in /api/admin/costs* expose it.
    con.execute("""
        CREATE TABLE IF NOT EXISTS query_costs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            question TEXT,
            started_at REAL,
            finished_at REAL,
            elapsed_ms INTEGER,
            n_propositions INTEGER,
            n_sources INTEGER,
            confidence TEXT,
            refused INTEGER,
            cumulative_usd_before REAL,
            cumulative_usd_after REAL,
            total_cost_usd REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            n_api_calls INTEGER,
            by_operation_json TEXT,
            calls_json TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_query_costs_conv ON query_costs(conversation_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_query_costs_started ON query_costs(started_at)")
    # Per-identifier query quota ledger (anon + signed). One row per asked
    # question. We compute lifetime/daily totals at check-time; trivially
    # cheap with the index.
    con.execute("""
        CREATE TABLE IF NOT EXISTS query_quota (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT NOT NULL,   -- 'anon:<uuid>' or 'user:<email>'
            is_signed INTEGER NOT NULL,
            asked_at REAL NOT NULL,
            question TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_quota_id_time ON query_quota(identifier, asked_at)")
    con.commit()
    return con


_conv_con: sqlite3.Connection | None = None
_fast_pipeline: FullPipeline | None = None
_hard_pipeline: HardQuestionPipeline | None = None


def conv_con() -> sqlite3.Connection:
    global _conv_con
    if _conv_con is None:
        _conv_con = init_conv_db()
    return _conv_con


def get_pipelines() -> tuple[FullPipeline, HardQuestionPipeline]:
    """Lazy-init both pipelines. Routing happens in /api/ask after the
    planner runs once."""
    global _fast_pipeline, _hard_pipeline
    if _fast_pipeline is None:
        _fast_pipeline = FullPipeline(k_chunks=16, real_planner=True)
    if _hard_pipeline is None:
        _hard_pipeline = HardQuestionPipeline()
    return _fast_pipeline, _hard_pipeline


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the conversation DB
    conv_con()
    log.info("bif_research API ready")
    yield


app = FastAPI(lifespan=lifespan, title="Sopal BIF Research")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    pipeline_mode: str = "hybrid"   # "bm25" | "dense" | "hybrid"


class ConvCreateRequest(BaseModel):
    title: str | None = None


# ---------------------------------------------------------------------------
# Auth + per-identifier quota
# ---------------------------------------------------------------------------

def _decode_user_email(authorization_header: Optional[str]) -> Optional[str]:
    """Decode the JWT in an Authorization: Bearer <token> header. Returns
    the user's email if valid, else None. Mirrors server.py's purchase
    token format."""
    if not authorization_header:
        return None
    h = authorization_header.strip()
    if h.lower().startswith("bearer "):
        h = h[7:].strip()
    if not h:
        return None
    try:
        from jose import jwt as _jwt, JWTError
        payload = _jwt.decode(h, JWT_SECRET, algorithms=[JWT_ALG])
        # server.py stores email under 'sub' (standard) or 'email'
        return payload.get("sub") or payload.get("email")
    except Exception:
        return None


def _identify_request(authorization: Optional[str], anon_id: Optional[str]) -> tuple[str, bool, Optional[str]]:
    """Resolve who's asking. Returns (identifier, is_signed, email_or_none).

    Identifier format:
      'user:<email>'  if the JWT in Authorization decodes to an email
      'anon:<uuid>'   otherwise; uses X-Anon-ID header (client-generated)
    """
    email = _decode_user_email(authorization)
    if email:
        return f"user:{email.lower()}", True, email
    aid = (anon_id or "").strip()
    if not aid:
        # No anon-id supplied — generate one server-side. Client SHOULD send
        # X-Anon-ID for stable counting; absent that we can't track properly
        # so we still rate-limit but every request looks like a new user.
        # In practice the frontend always sends one.
        aid = uuid.uuid4().hex
    return f"anon:{aid}", False, None


def _quota_check(identifier: str, is_signed: bool) -> tuple[bool, dict]:
    """Return (allowed, info). info contains:
      - kind: 'anon' | 'signed'
      - used:  questions used so far (lifetime for anon, today UTC for signed)
      - limit: cap
      - remaining: max(0, limit-used)
    If not allowed, frontend should show the upgrade/contact gate.
    """
    con = conv_con()
    if is_signed:
        # Count today UTC
        now = datetime.now(timezone.utc)
        start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()
        used = con.execute(
            "SELECT COUNT(*) FROM query_quota WHERE identifier=? AND asked_at>=?",
            (identifier, start_of_day),
        ).fetchone()[0]
        info = {"kind": "signed", "used": used, "limit": SIGNED_DAILY_LIMIT,
                "remaining": max(0, SIGNED_DAILY_LIMIT - used)}
        return used < SIGNED_DAILY_LIMIT, info
    used = con.execute(
        "SELECT COUNT(*) FROM query_quota WHERE identifier=?",
        (identifier,),
    ).fetchone()[0]
    info = {"kind": "anon", "used": used, "limit": ANON_QUESTION_LIMIT,
            "remaining": max(0, ANON_QUESTION_LIMIT - used)}
    return used < ANON_QUESTION_LIMIT, info


def _record_quota(identifier: str, is_signed: bool, question: str) -> None:
    con = conv_con()
    con.execute(
        "INSERT INTO query_quota (identifier, is_signed, asked_at, question) VALUES (?,?,?,?)",
        (identifier, 1 if is_signed else 0, time.time(), (question or "")[:500]),
    )
    con.commit()


def _gate_message(info: dict) -> str:
    if info.get("kind") == "anon":
        return ("You've used your free preview questions. Sign in (free) to "
                "get 30 questions per day.")
    return ("You've reached today's 30-question limit. For unlimited access, "
            "please contact info@sopal.com.au to arrange enterprise access.")


# ---------------------------------------------------------------------------
# Conversation endpoints
# ---------------------------------------------------------------------------

@app.get("/api/conversations")
def list_conversations(limit: int = 30):
    rows = conv_con().execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "ORDER BY updated_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]


@app.post("/api/conversations")
def create_conversation(payload: ConvCreateRequest):
    conv_id = uuid.uuid4().hex[:12]
    now = time.time()
    title = payload.title or "New chat"
    conv_con().execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conv_id, title, now, now),
    )
    conv_con().commit()
    return {"id": conv_id, "title": title}


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str):
    conv = conv_con().execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    msgs = conv_con().execute(
        "SELECT role, content_json, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id", (conv_id,)
    ).fetchall()
    return {
        "id": conv[0],
        "title": conv[1],
        "created_at": conv[2],
        "updated_at": conv[3],
        "messages": [
            {"role": r[0], "content": json.loads(r[1] or "{}"), "created_at": r[2]}
            for r in msgs
        ],
    }


# ---------------------------------------------------------------------------
# Source endpoint
# ---------------------------------------------------------------------------

@app.get("/api/sources/{chunk_id}")
def get_source(chunk_id: str):
    if not CHUNKS_DB.exists():
        raise HTTPException(503, "Index not built")
    con = sqlite3.connect(str(CHUNKS_DB), check_same_thread=False)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT chunk_id, source_id, source_type, header, text, metadata_json "
        "FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(404, "Chunk not found")
    return {
        "chunk_id": row["chunk_id"],
        "source_id": row["source_id"],
        "source_type": row["source_type"],
        "header": row["header"],
        "text": row["text"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }


# ---------------------------------------------------------------------------
# Ask endpoint (SSE)
# ---------------------------------------------------------------------------

@app.get("/api/usage")
def get_usage(request: Request):
    """Return the caller's quota state. Anon callers should send X-Anon-ID."""
    identifier, is_signed, email = _identify_request(
        request.headers.get("authorization"),
        request.headers.get("x-anon-id"),
    )
    allowed, info = _quota_check(identifier, is_signed)
    return {
        "is_signed": is_signed,
        "email": email,
        "allowed": allowed,
        **info,
    }


@app.post("/api/ask")
async def ask(payload: AskRequest, request: Request):
    """Stream the answer flow as SSE.

    Events emitted:
      status   - {"phase": "planning"|"retrieving"|"answering"|"verifying", "msg": str}
      answer   - the final dict from pipeline.answer(...)
      title    - {"conversation_id": str, "title": str}
      error    - {"error": str}
      done     - {}

    Authorization:
      Bearer JWT in the Authorization header for signed users (30/day cap).
      X-Anon-ID header (UUID) for anonymous users (4-question lifetime cap).
    """
    # Quota gate first — fail fast before spinning up the pipeline
    identifier, is_signed, _email = _identify_request(
        request.headers.get("authorization"),
        request.headers.get("x-anon-id"),
    )
    allowed, info = _quota_check(identifier, is_signed)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "quota_exceeded",
                "kind": info["kind"],
                "used": info["used"],
                "limit": info["limit"],
                "message": _gate_message(info),
            },
        )

    fast_pipe, hard_pipe = get_pipelines()

    # Persist user msg. Note: we DO NOT auto-set title from the question here.
    # The AI-generated title is set after the answer is composed and emitted
    # via SSE so the sidebar can slide it in.
    conv_id = payload.conversation_id
    is_first_turn = False
    if conv_id:
        # Check if this is the first turn so we know to generate a title
        existing_count = conv_con().execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()[0]
        is_first_turn = existing_count == 0
        conv_con().execute(
            "INSERT INTO messages (conversation_id, role, content_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conv_id, "user", json.dumps({"text": payload.question}), time.time()),
        )
        conv_con().execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (time.time(), conv_id),
        )
        conv_con().commit()

    # Load conversation history (prior turns) so the pipeline can interpret
    # follow-up questions in context. Without this, "is it sufficient?" is
    # treated as a brand-new search and matches against any chunk containing
    # "sufficient" in the corpus.
    history = []
    if conv_id:
        rows = conv_con().execute(
            "SELECT role, content_json FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
        for r in rows:
            try:
                content = json.loads(r[1] or "{}")
            except json.JSONDecodeError:
                content = {}
            # User messages store {"text": "..."}; assistant messages store the
            # full structured answer dict — strip to summary text downstream.
            if r[0] == "user":
                text = content.get("text", "")
                if text:
                    history.append({"role": "user", "content": text})
            else:
                history.append({"role": "assistant", "content": content})

    def gen():
        def emit(event: str, data: dict) -> bytes:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
        try:
            # Record this question against the caller's quota now (before any
            # work). If the pipeline crashes the user still spent a quota
            # slot — that's the right tradeoff vs giving free retries.
            try:
                _record_quota(identifier, is_signed, payload.question)
            except Exception as qerr:
                log.warning(f"failed to record quota: {qerr}")

            started_at = time.time()
            cum_before = budget.cumulative_usd()
            cap = budget.CostCapture()

            # NB: do NOT yield inside a `with cap:` block. Starlette runs
            # successive iterations of this sync generator in a thread pool;
            # the contextvar token created in __enter__ is invalidated when
            # the next iteration lands on a different thread. We keep each
            # `with cap:` block yield-free and let cap.calls accumulate
            # across multiple enter/exit cycles in different threads.

            yield emit("status", {"phase": "planning",
                                  "msg": "Planning — naming provisions and authorities…"})

            # Phase A — planner + routing (no yields inside)
            with cap:
                rewritten = _rewrite_with_history(payload.question, history)
                plan = planner_module.plan(rewritten, real=True)
                route = route_question(plan, rewritten)

            yield emit("status", {"phase": "routing",
                                  "msg": f"Routed to {route} path"})
            if route == "hard":
                resolved_cases = _resolved_case_citations(plan.named_authorities or [])
                in_corpus = [c for c in resolved_cases if c["in_corpus"]]
                n_cases = len(in_corpus)
                yield emit("status", {
                    "phase": "reading_cases",
                    "msg": f"Reading {n_cases} judgment(s) in parallel",
                    "cases": [c["citation"] for c in in_corpus],
                    "missed": [c["citation"] for c in resolved_cases if not c["in_corpus"]],
                })
            else:
                yield emit("status", {"phase": "retrieving",
                                      "msg": "Retrieving sources…"})

            # Phase B — answer composition + title generation (no yields inside)
            generated_title: str | None = None
            with cap:
                if route == "hard":
                    answer = hard_pipe.answer_with_plan(
                        rewritten, plan, history=history,
                        original_question=payload.question,
                    )
                else:
                    answer = fast_pipe.answer_with_plan(
                        rewritten, plan, history=history,
                        original_question=payload.question,
                    )
                # Generate a chat title for new conversations
                if conv_id and is_first_turn:
                    generated_title = _generate_chat_title(payload.question)
            cum_after = budget.cumulative_usd()
            elapsed = time.time() - started_at
            answer["_elapsed_ms"] = int(elapsed * 1000)
            answer["_route"] = route

            # Persist + emit the title before the answer so the sidebar
            # updates while the user is reading
            if generated_title and conv_id:
                try:
                    conv_con().execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (generated_title, conv_id),
                    )
                    conv_con().commit()
                except Exception as title_err:
                    log.warning(f"failed to persist chat title: {title_err}")
                yield emit("title", {
                    "conversation_id": conv_id,
                    "title": generated_title,
                })

            yield emit("status", {"phase": "verifying", "msg": "Verifying citations…"})
            yield emit("answer", answer)
            cap_summary = cap.summary()
            # Persist per-query cost row (server-side only, not in answer payload)
            try:
                conv_con().execute(
                    """INSERT INTO query_costs
                       (conversation_id, question, started_at, finished_at, elapsed_ms,
                        n_propositions, n_sources, confidence, refused,
                        cumulative_usd_before, cumulative_usd_after, total_cost_usd,
                        input_tokens, output_tokens, n_api_calls,
                        by_operation_json, calls_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        conv_id,
                        payload.question,
                        started_at,
                        time.time(),
                        int(elapsed * 1000),
                        len(answer.get("propositions") or []),
                        len(answer.get("sources") or []),
                        answer.get("confidence", ""),
                        1 if answer.get("refused") else 0,
                        cum_before,
                        cum_after,
                        cap_summary["total_usd"],
                        cap_summary["input_tokens"],
                        cap_summary["output_tokens"],
                        cap_summary["n_api_calls"],
                        json.dumps(cap_summary["by_operation"]),
                        json.dumps(cap_summary["calls"]),
                    ),
                )
                conv_con().commit()
                log.info(
                    f"query cost: ${cap_summary['total_usd']:.4f} "
                    f"({cap_summary['n_api_calls']} api calls, "
                    f"{cap_summary['input_tokens']}/{cap_summary['output_tokens']} tok, "
                    f"{int(elapsed*1000)}ms) — '{payload.question[:60]}'"
                )
            except Exception as cost_err:
                log.warning(f"failed to persist query_costs row: {cost_err}")

            if conv_id:
                conv_con().execute(
                    "INSERT INTO messages (conversation_id, role, content_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (conv_id, "assistant", json.dumps(answer), time.time()),
                )
                conv_con().execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (time.time(), conv_id),
                )
                conv_con().commit()
            yield emit("done", {})
        except budget.BudgetExceeded as e:
            yield emit("error", {"error": "Budget cap reached.", "detail": str(e)})
        except Exception as e:
            log.exception("ask failed")
            yield emit("error", {"error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Admin: per-query cost endpoints (NOT linked from the frontend).
# Designed to slot into the Sopal admin console later. No auth here yet —
# expose only on localhost. When integrated into the Sopal site, wrap
# behind the existing /admin/page-style auth gate.
# ---------------------------------------------------------------------------

@app.get("/api/admin/costs")
def admin_costs(limit: int = 100, conversation_id: str | None = None):
    """Recent per-query cost rows. Most recent first."""
    where = ""
    params: list = []
    if conversation_id:
        where = "WHERE conversation_id = ?"
        params.append(conversation_id)
    params.append(limit)
    rows = conv_con().execute(
        f"""SELECT id, conversation_id, question, started_at, finished_at, elapsed_ms,
                   n_propositions, n_sources, confidence, refused,
                   total_cost_usd, input_tokens, output_tokens, n_api_calls,
                   by_operation_json
            FROM query_costs
            {where}
            ORDER BY started_at DESC
            LIMIT ?""",
        params,
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0], "conversation_id": r[1], "question": r[2],
            "started_at": r[3], "finished_at": r[4], "elapsed_ms": r[5],
            "n_propositions": r[6], "n_sources": r[7],
            "confidence": r[8], "refused": bool(r[9]),
            "total_cost_usd": r[10],
            "input_tokens": r[11], "output_tokens": r[12], "n_api_calls": r[13],
            "by_operation": json.loads(r[14] or "{}"),
        })
    return {"queries": out, "n": len(out)}


@app.get("/api/admin/costs/summary")
def admin_costs_summary():
    """Aggregate cost stats — total, today, last 24h, average per query."""
    now = time.time()
    day_ago = now - 24 * 3600
    week_ago = now - 7 * 24 * 3600
    midnight = now - (now % 86400)  # crude UTC midnight
    rows = conv_con().execute(
        "SELECT COUNT(*), COALESCE(SUM(total_cost_usd), 0), "
        "COALESCE(AVG(total_cost_usd), 0), COALESCE(AVG(elapsed_ms), 0), "
        "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) "
        "FROM query_costs"
    ).fetchone()
    today = conv_con().execute(
        "SELECT COUNT(*), COALESCE(SUM(total_cost_usd), 0) FROM query_costs WHERE started_at >= ?",
        (midnight,),
    ).fetchone()
    last24 = conv_con().execute(
        "SELECT COUNT(*), COALESCE(SUM(total_cost_usd), 0) FROM query_costs WHERE started_at >= ?",
        (day_ago,),
    ).fetchone()
    last7 = conv_con().execute(
        "SELECT COUNT(*), COALESCE(SUM(total_cost_usd), 0) FROM query_costs WHERE started_at >= ?",
        (week_ago,),
    ).fetchone()
    return {
        "all_time": {
            "n_queries": rows[0],
            "total_cost_usd": round(rows[1], 6),
            "avg_cost_per_query_usd": round(rows[2], 6),
            "avg_elapsed_ms": round(rows[3], 0),
            "input_tokens": rows[4],
            "output_tokens": rows[5],
        },
        "today_utc": {"n_queries": today[0], "cost_usd": round(today[1], 6)},
        "last_24h": {"n_queries": last24[0], "cost_usd": round(last24[1], 6)},
        "last_7d": {"n_queries": last7[0], "cost_usd": round(last7[1], 6)},
        "cumulative_spend_usd": budget.cumulative_usd(),
        "remaining_usd": budget.remaining_usd(),
    }


@app.get("/api/admin/costs/{cost_id}")
def admin_cost_detail(cost_id: int):
    """Full detail of one cost row, including per-call breakdown."""
    r = conv_con().execute(
        "SELECT id, conversation_id, question, started_at, finished_at, elapsed_ms, "
        "n_propositions, n_sources, confidence, refused, "
        "cumulative_usd_before, cumulative_usd_after, total_cost_usd, "
        "input_tokens, output_tokens, n_api_calls, "
        "by_operation_json, calls_json "
        "FROM query_costs WHERE id = ?",
        (cost_id,),
    ).fetchone()
    if not r:
        raise HTTPException(404, "Not found")
    return {
        "id": r[0], "conversation_id": r[1], "question": r[2],
        "started_at": r[3], "finished_at": r[4], "elapsed_ms": r[5],
        "n_propositions": r[6], "n_sources": r[7],
        "confidence": r[8], "refused": bool(r[9]),
        "cumulative_usd_before": r[10], "cumulative_usd_after": r[11],
        "total_cost_usd": r[12],
        "input_tokens": r[13], "output_tokens": r[14], "n_api_calls": r[15],
        "by_operation": json.loads(r[16] or "{}"),
        "calls": json.loads(r[17] or "[]"),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    info = {
        "ok": True,
        "cumulative_spend_usd": budget.cumulative_usd(),
        "remaining_usd": budget.remaining_usd(),
        "chunks_db_present": CHUNKS_DB.exists(),
        "bm25_present": BM25_PATH.exists(),
        "chroma_present": (CHROMA_PATH / "chroma.sqlite3").exists(),
    }
    if CHUNKS_DB.exists():
        con = sqlite3.connect(str(CHUNKS_DB), check_same_thread=False)
        n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        by_type = {
            r[0]: r[1]
            for r in con.execute(
                "SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type"
            ).fetchall()
        }
        info["chunks_total"] = n
        info["chunks_by_type"] = by_type
        con.close()
    return info


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if WEB.exists():
    # /static/* -> any file under web/ (styles.css, app.js, future assets).
    # When the app is mounted at /sopalai in production, these become
    # /sopalai/static/styles.css etc — matching the absolute paths in
    # index.html. Local-dev clients hit /static/styles.css with no prefix.
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(WEB / "index.html"))
