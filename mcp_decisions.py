"""Adjudication Decisions MCP server.

Exposes the full corpus of Australian construction-payment adjudication
decisions (the same data behind sopal.com.au's search) to Claude as a remote
MCP connector, over streamable HTTP. Mounted onto the main FastAPI app in
server.py at /dmcp; auth is enforced there (single shared key, either as
'Authorization: Bearer <key>' on /dmcp/mcp or embedded in the path as
/dmcp/<key>/mcp for claude.ai custom connectors — see server.py).

Data sources (read-only) in qbcc.db:
  - docs_fresh        full decision text + pdf path  (FTS index: `fts`)
  - decision_details  structured metadata per decision (parties, amounts, ...)

The core query helpers (_search/_get/_stats) are plain sync functions so they
can be unit-tested without the mcp package; the @mcp.tool() wrappers just push
them through asyncio.to_thread to keep the event loop free.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import struct
import threading
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Read-only connection to qbcc.db
# ─────────────────────────────────────────────────────────────────────────────
_DB_PATH = os.environ.get("DECISIONS_DB_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qbcc.db"
)

_con = sqlite3.connect(_DB_PATH, check_same_thread=False)
_con.row_factory = sqlite3.Row
_lock = threading.Lock()  # serialise access; sqlite connection is shared


def _rank(raw: bytes) -> float:
    """FTS4 relevance score from matchinfo(fts) in default 'pcx' format.

    Layout: [phrase_count, col_count, then 3 ints per (phrase, col):
    (hits_this_row, hits_all_rows, docs_with_hits)]. We sum hits_this_row
    across phrases/cols and negate so ORDER BY ... ASC puts best matches first.
    """
    try:
        ints = struct.unpack("@%dI" % (len(raw) // 4), raw)
    except struct.error:
        return 0.0
    p, c = ints[0], ints[1]
    score = 0
    for phrase in range(p):
        for col in range(c):
            base = 2 + (phrase * c + col) * 3
            if base < len(ints):
                score += ints[base]
    return -float(score)


_con.create_function("rank", 1, _rank)


# ─────────────────────────────────────────────────────────────────────────────
# FTS query handling
# ─────────────────────────────────────────────────────────────────────────────
def _fts_expr(query: str) -> str:
    """Turn a natural-language query into a safe FTS4 MATCH expression.

    Passes the query through as-is first (so power users can use AND/OR/NEAR/
    "phrases"/-exclusions); the caller retries with this sanitised form if the
    raw expression raises an FTS syntax error.
    """
    import re

    tokens = re.findall(r'"[^"]+"|\S+', query.strip())
    safe = []
    for tok in tokens:
        if tok.startswith('"') and tok.endswith('"'):
            safe.append(tok)
        else:
            cleaned = re.sub(r'[^0-9A-Za-z]+', " ", tok).strip()
            for word in cleaned.split():
                safe.append(f'"{word}"')
    return " ".join(safe)


_SORTS = {
    "relevance": "ORDER BY rank(matchinfo(fts))",
    "newest": "ORDER BY a.decision_date DESC",
    "oldest": "ORDER BY a.decision_date ASC",
    "claim_high": "ORDER BY CAST(NULLIF(a.claimed_amount,'') AS REAL) DESC",
    "claim_low": "ORDER BY CAST(NULLIF(a.claimed_amount,'') AS REAL) ASC",
    "adj_high": "ORDER BY CAST(NULLIF(a.adjudicated_amount,'') AS REAL) DESC",
    "adj_low": "ORDER BY CAST(NULLIF(a.adjudicated_amount,'') AS REAL) ASC",
}


def _search(
    query: str = "",
    adjudicator: Optional[str] = None,
    claimant: Optional[str] = None,
    respondent: Optional[str] = None,
    act: Optional[str] = None,
    outcome: Optional[str] = None,
    section: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_claim: Optional[float] = None,
    max_claim: Optional[float] = None,
    min_adjudicated: Optional[float] = None,
    max_adjudicated: Optional[float] = None,
    sort: str = "relevance",
    limit: int = 20,
) -> dict:
    query = (query or "").strip()
    limit = max(1, min(int(limit or 20), 50))

    where = [
        "a.decision_date IS NOT NULL",
        "a.decision_date != ''",
        "LOWER(TRIM(a.decision_date)) != 'null'",
    ]
    params: list = []

    has_text = bool(query)
    if has_text:
        where.append("fts MATCH ?")
        params.append(query)

    def like(col: str, val: Optional[str]):
        if val:
            where.append(f"LOWER(a.{col}) LIKE ?")
            params.append(f"%{val.lower()}%")

    like("adjudicator_name", adjudicator)
    like("claimant_name", claimant)
    like("respondent_name", respondent)
    like("act_category", act)
    like("outcome", outcome)
    like("sections_referenced", section)

    if date_from:
        where.append("a.decision_date >= ?"); params.append(date_from)
    if date_to:
        where.append("a.decision_date <= ?"); params.append(date_to)
    if min_claim is not None:
        where.append("CAST(NULLIF(a.claimed_amount,'') AS REAL) >= ?"); params.append(min_claim)
    if max_claim is not None:
        where.append("CAST(NULLIF(a.claimed_amount,'') AS REAL) <= ?"); params.append(max_claim)
    if min_adjudicated is not None:
        where.append("CAST(NULLIF(a.adjudicated_amount,'') AS REAL) >= ?"); params.append(min_adjudicated)
    if max_adjudicated is not None:
        where.append("CAST(NULLIF(a.adjudicated_amount,'') AS REAL) <= ?"); params.append(max_adjudicated)

    base_join = (
        "FROM fts "
        "JOIN docs_fresh d ON fts.rowid = d.rowid "
        "LEFT JOIN decision_details a ON d.ejs_id = a.ejs_id"
    )
    where_sql = "WHERE " + " AND ".join(where)

    # Sorting: relevance needs a text match; otherwise default to newest.
    if sort == "relevance" and not has_text:
        sort = "newest"
    order_sql = _SORTS.get(sort, _SORTS["relevance" if has_text else "newest"])

    snippet_sql = (
        "snippet(fts, '«', '»', ' … ', -1, 30)" if has_text
        else "substr(d.full_text, 1, 240) || '…'"
    )

    select = f"""
        SELECT d.ejs_id, d.reference, d.pdf_path, {snippet_sql} AS snippet,
               a.claimant_name, a.respondent_name, a.adjudicator_name,
               a.act_category, a.outcome, a.decision_date,
               a.claimed_amount, a.adjudicated_amount,
               a.jurisdiction_upheld, a.sections_referenced
        {base_join}
        {where_sql}
        {order_sql}
        LIMIT ?
    """
    count_sql = f"SELECT COUNT(*) {base_join} {where_sql}"

    with _lock:
        try:
            total = _con.execute(count_sql, tuple(params)).fetchone()[0]
            rows = _con.execute(select, tuple(params) + (limit,)).fetchall()
        except sqlite3.OperationalError:
            # Most likely an FTS syntax error in a raw query — retry sanitised.
            if not has_text:
                raise
            params[params.index(query)] = _fts_expr(query)
            total = _con.execute(count_sql, tuple(params)).fetchone()[0]
            rows = _con.execute(select, tuple(params) + (limit,)).fetchall()

    items = []
    for r in rows:
        items.append({
            "ejs_id": r["ejs_id"],
            "reference": r["reference"],
            "decision_date": r["decision_date"],
            "claimant": r["claimant_name"],
            "respondent": r["respondent_name"],
            "adjudicator": r["adjudicator_name"],
            "act": r["act_category"],
            "outcome": r["outcome"],
            "claimed_amount": r["claimed_amount"],
            "adjudicated_amount": r["adjudicated_amount"],
            "jurisdiction_upheld": r["jurisdiction_upheld"],
            "sections_referenced": r["sections_referenced"],
            "snippet": r["snippet"],
        })
    return {"total": total, "returned": len(items), "items": items}


_GCS_BUCKET = "sopal-bucket"


def _pdf_url(pdf_path: Optional[str]) -> Optional[str]:
    if not pdf_path:
        return None
    return f"https://storage.googleapis.com/{_GCS_BUCKET}/pdfs/{os.path.basename(pdf_path)}"


def _get(ejs_id: str) -> Optional[dict]:
    ejs_id = (ejs_id or "").strip()
    if not ejs_id:
        return None
    with _lock:
        doc = _con.execute(
            "SELECT ejs_id, reference, pdf_path, full_text FROM docs_fresh "
            "WHERE ejs_id = ? COLLATE NOCASE", (ejs_id,)
        ).fetchone()
        if doc is None:
            return None
        meta = _con.execute(
            "SELECT * FROM decision_details WHERE ejs_id = ? COLLATE NOCASE",
            (doc["ejs_id"],),
        ).fetchone()

    out = {
        "ejs_id": doc["ejs_id"],
        "reference": doc["reference"],
        "pdf_url": _pdf_url(doc["pdf_path"]),
        "full_text": doc["full_text"],
    }
    if meta is not None:
        m = dict(meta)
        m.pop("raw_json", None)  # internal; redundant with the columns
        out.update({
            "adjudicator": m.get("adjudicator_name"),
            "claimant": m.get("claimant_name"),
            "respondent": m.get("respondent_name"),
            "decision_date": m.get("decision_date"),
            "act": m.get("act_category"),
            "outcome": m.get("outcome"),
            "claimed_amount": m.get("claimed_amount"),
            "payment_schedule_amount": m.get("payment_schedule_amount"),
            "adjudicated_amount": m.get("adjudicated_amount"),
            "jurisdiction_upheld": m.get("jurisdiction_upheld"),
            "fee_claimant_proportion": m.get("fee_claimant_proportion"),
            "fee_respondent_proportion": m.get("fee_respondent_proportion"),
            "sections_referenced": m.get("sections_referenced"),
            "keywords": m.get("keywords"),
            "project_type": m.get("project_type"),
            "contract_type": m.get("contract_type"),
            "doc_length_pages": m.get("doc_length_pages"),
        })
    return out


def _stats() -> dict:
    with _lock:
        total = _con.execute("SELECT COUNT(*) FROM docs_fresh").fetchone()[0]
        dates = _con.execute(
            "SELECT MIN(decision_date), MAX(decision_date) FROM decision_details "
            "WHERE decision_date LIKE '20%'"
        ).fetchone()
        acts = _con.execute(
            "SELECT act_category, COUNT(*) c FROM decision_details "
            "WHERE act_category IS NOT NULL AND act_category != '' "
            "GROUP BY act_category ORDER BY c DESC LIMIT 8"
        ).fetchall()
        outcomes = _con.execute(
            "SELECT outcome, COUNT(*) c FROM decision_details "
            "WHERE outcome IS NOT NULL AND outcome != '' "
            "GROUP BY outcome ORDER BY c DESC LIMIT 8"
        ).fetchall()
        adjudicators = _con.execute(
            "SELECT adjudicator_name, COUNT(*) c FROM decision_details "
            "WHERE adjudicator_name IS NOT NULL AND adjudicator_name != '' "
            "GROUP BY adjudicator_name ORDER BY c DESC LIMIT 15"
        ).fetchall()
    return {
        "total_decisions": total,
        "earliest_decision": dates[0],
        "latest_decision": dates[1],
        "act_categories": [{"act": r[0], "count": r[1]} for r in acts],
        "outcomes": [{"outcome": r[0], "count": r[1]} for r in outcomes],
        "top_adjudicators": [{"name": r[0], "count": r[1]} for r in adjudicators],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCP protocol layer — hand-rolled streamable HTTP (JSON responses), no SDK.
#
# We deliberately do NOT depend on the `mcp` package: it forces pydantic>=2.11,
# which breaks this app's pinned fastapi==0.111.0 at startup. The streamable-HTTP
# MCP surface we need (initialize / tools/list / tools/call / ping) is a small
# JSON-RPC 2.0 protocol, implemented directly below as a bare ASGI app that
# server.py mounts behind its key-auth wrapper.
# ─────────────────────────────────────────────────────────────────────────────
import json

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "Sopal Adjudication Decisions", "version": "1.0.0"}
INSTRUCTIONS = (
    "Sopal Adjudication Decisions is a full-text database of Australian "
    "construction security-of-payment adjudication decisions (Queensland BCIPA "
    "2004 and BIF Act 2017). Use search_decisions to find relevant decisions by "
    "keyword, party, adjudicator, section of the Act, amount or date; "
    "get_decision to retrieve the complete verbatim text of one decision by its "
    "ejs_id; and corpus_stats to see coverage. Quote decisions verbatim — never "
    "paraphrase statutory text or an adjudicator's reasons as if they were a "
    "direct quote."
)

TOOLS = [
    {
        "name": "search_decisions",
        "description": (
            "Search the adjudication-decision corpus by full text and/or "
            "metadata. All arguments are optional and combine with AND; omit "
            "`query` to run a pure metadata filter. Full-text `query` uses "
            "SQLite FTS4 syntax: plain words are ANDed (reference date); "
            "\"quoted phrases\" match verbatim; OR between terms; minus "
            "excludes (variation -residential); NEAR/n proximity. Returns "
            "{total, returned, items}; each item has ejs_id, parties, "
            "adjudicator, act, outcome, decision_date, claimed_amount, "
            "adjudicated_amount, jurisdiction_upheld, sections_referenced and a "
            "snippet (matched terms wrapped in «guillemets»). Pass an item's "
            "ejs_id to get_decision to read it in full."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Full-text FTS4 search terms. Optional."},
                "adjudicator": {"type": "string", "description": "Adjudicator name, case-insensitive substring (e.g. 'Davenport')."},
                "claimant": {"type": "string", "description": "Claimant party name, case-insensitive substring."},
                "respondent": {"type": "string", "description": "Respondent party name, case-insensitive substring."},
                "act": {"type": "string", "description": "Act substring. Dominant: 'BCIPA 2004 (Qld)' (pre-2018) and 'BIF Act 2017 (Qld)'."},
                "outcome": {"type": "string", "description": "Outcome substring. Common: 'Claimant Fully Successful', 'Unsuccessful', 'Partly Successful'."},
                "section": {"type": "string", "description": "Section of the Act referenced, substring (e.g. '75' or 's 88')."},
                "date_from": {"type": "string", "description": "Earliest decision date, ISO YYYY-MM-DD."},
                "date_to": {"type": "string", "description": "Latest decision date, ISO YYYY-MM-DD."},
                "min_claim": {"type": "number", "description": "Minimum claimed amount (AUD)."},
                "max_claim": {"type": "number", "description": "Maximum claimed amount (AUD)."},
                "min_adjudicated": {"type": "number", "description": "Minimum adjudicated (awarded) amount (AUD)."},
                "max_adjudicated": {"type": "number", "description": "Maximum adjudicated (awarded) amount (AUD)."},
                "sort": {"type": "string", "enum": ["relevance", "newest", "oldest", "claim_high", "claim_low", "adj_high", "adj_low"], "description": "Result ordering. 'relevance' requires a query; defaults to newest otherwise."},
                "limit": {"type": "integer", "description": "Max results (default 20, max 50)."},
            },
        },
    },
    {
        "name": "get_decision",
        "description": (
            "Retrieve one adjudication decision in full, including its complete "
            "verbatim text (full_text — can be long; use it for verbatim "
            "quotation), structured metadata and the source PDF link (pdf_url). "
            "If no decision matches, returns {found: false}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ejs_id": {"type": "string", "description": "Decision id from search_decisions (e.g. 'EJS01234')."},
            },
            "required": ["ejs_id"],
        },
    },
    {
        "name": "corpus_stats",
        "description": (
            "Summarise coverage of the database: total decision count, "
            "earliest/latest decision dates, and the most common Acts, outcomes "
            "and adjudicators. Useful for grounding a query and discovering "
            "valid filter values before calling search_decisions."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _run_tool(name: str, args: dict) -> dict:
    """Execute a tool synchronously and return its result dict."""
    if name == "search_decisions":
        keys = ("query", "adjudicator", "claimant", "respondent", "act",
                "outcome", "section", "date_from", "date_to", "min_claim",
                "max_claim", "min_adjudicated", "max_adjudicated", "sort", "limit")
        kwargs = {k: args[k] for k in keys if k in args and args[k] is not None}
        return _search(**kwargs)
    if name == "get_decision":
        decision = _get(args.get("ejs_id", ""))
        if decision is None:
            return {"found": False, "message": (
                f"No decision found with ejs_id {args.get('ejs_id')!r}. "
                "Use search_decisions to find the correct id.")}
        return {"found": True, **decision}
    if name == "corpus_stats":
        return _stats()
    raise KeyError(name)


def _jsonrpc_result(rpc_id, result):
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _jsonrpc_error(rpc_id, code, message):
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


async def _handle_message(msg: dict):
    """Dispatch a single JSON-RPC message; return a response dict, or None for
    notifications (no id / notifications.*)."""
    if not isinstance(msg, dict):
        return _jsonrpc_error(None, -32600, "Invalid Request")
    rpc_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    is_notification = "id" not in msg

    try:
        if method == "initialize":
            client_ver = params.get("protocolVersion") or PROTOCOL_VERSION
            result = {
                "protocolVersion": client_ver,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS,
            }
        elif method == "ping":
            result = {}
        elif method and method.startswith("notifications/"):
            return None  # client notifications need no response
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                data = await asyncio.to_thread(_run_tool, name, args)
            except KeyError:
                return _jsonrpc_result(rpc_id, {
                    "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                    "isError": True,
                })
            except Exception as exc:  # tool failure -> reported in result, not protocol error
                return _jsonrpc_result(rpc_id, {
                    "content": [{"type": "text", "text": f"Tool error: {exc}"}],
                    "isError": True,
                })
            result = {"content": [{"type": "text", "text": json.dumps(data, indent=2, default=str)}]}
        else:
            if is_notification:
                return None
            return _jsonrpc_error(rpc_id, -32601, f"Method not found: {method}")
    except Exception as exc:
        return _jsonrpc_error(rpc_id, -32603, f"Internal error: {exc}")

    if is_notification:
        return None
    return _jsonrpc_result(rpc_id, result)


async def _read_body(receive) -> bytes:
    chunks = []
    while True:
        event = await receive()
        if event["type"] == "http.request":
            chunks.append(event.get("body", b""))
            if not event.get("more_body"):
                break
        elif event["type"] == "http.disconnect":
            break
    return b"".join(chunks)


async def _send_json(send, status: int, obj, extra_headers=None):
    body = json.dumps(obj).encode() if obj is not None else b""
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"mcp-protocol-version", PROTOCOL_VERSION.encode()),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def mcp_asgi(scope, receive, send):
    """Bare ASGI app implementing MCP streamable HTTP (JSON responses).

    Stateless: every POST is a self-contained JSON-RPC request (single object
    or batch array). GET/DELETE (the optional SSE stream / session teardown)
    are not supported in JSON mode and return 405, which compliant clients
    (including claude.ai's connector) tolerate.
    """
    if scope["type"] != "http":
        return
    method = scope.get("method", "")
    if method != "POST":
        await _send_json(send, 405, _jsonrpc_error(None, -32600, "Use POST for MCP requests."),
                         extra_headers=[(b"allow", b"POST")])
        return

    raw = await _read_body(receive)
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        await _send_json(send, 400, _jsonrpc_error(None, -32700, "Parse error"))
        return

    if isinstance(payload, list):  # JSON-RPC batch
        responses = []
        for item in payload:
            r = await _handle_message(item)
            if r is not None:
                responses.append(r)
        if not responses:
            await _send_json(send, 202, None)  # all notifications
        else:
            await _send_json(send, 200, responses)
        return

    response = await _handle_message(payload)
    if response is None:
        await _send_json(send, 202, None)  # notification
    else:
        await _send_json(send, 200, response)
