#!/usr/bin/env python3
"""ANA coding v2 — recover Authorised Nominating Authorities that v1 missed.

v1 coded post-2014 decisions where the QBCC Registrar referred the application
as ana='QBCC Registry' (or null), ignoring the adjudicator's ANA when it
appeared in an agent/letterhead role (e.g. "served ... by the ABC Dispute
Resolution Service as my agent"). That was a systematic miss: this pass
re-codes those rows, capturing the ANA in ANY role plus the referring body.

Targets decision_details rows with decision_date >= 2015-01-01 AND
(ana='QBCC Registry' OR ana IS NULL OR ana_source='ai_v2').
Writes ana, ana_role, ana_source='ai_v2'/'unknown', ana_evidence, and exports
exports/ana_ai_v2.json.

Requires OPENAI_API_KEY in the environment. Never hard-code the key here.
"""

import argparse
import asyncio
import json
import os
import random
import re
import sqlite3
import sys
import time

import aiohttp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "qbcc.db")
EXPORT_PATH = os.path.join(REPO_ROOT, "exports", "ana_ai_v2.json")

DEFAULT_MODEL = "gpt-5.4-nano"
CONCURRENCY = 30
MAX_RETRIES = 5
BATCH_COMMIT = 100
FALLBACK_CHARS_PER_PAGE = 2600
DATE_CUTOFF = "2015-01-01"

CANONICAL = [
    "Adjudicate Today",
    "Institute of Arbitrators & Mediators Australia (IAMA)",
    "RICS Dispute Resolution Service",
    "ABC Dispute Resolution Service",
    "Australian Solutions Centre",
    "LEADR",
    "Queensland Law Society",
    "Master Builders Queensland",
    "Nominator",
    "Able Adjudication Pty Ltd",
    "JDLA Pty Ltd",
]

SYSTEM_PROMPT = """You are coding Queensland construction adjudication decisions \
(BCIPA 2004 or BIF Act 2017). You will be given excerpts (first ~3 pages and the \
last page) of one decision's OCR text. OCR output often has stray spaces breaking \
words (e.g. "Con struction", "Octo ber", "Adjudi cate") - mentally rejoin broken \
words when reading.

Background: since December 2014 the QBCC Adjudication Registrar (Queensland \
Building and Construction Commission) refers adjudication applications to \
adjudicators. However, adjudicators often belong to an Authorised Nominating \
Authority (ANA), and that ANA may still appear in the decision in one of these \
roles:
(a) "referrer" - the body that received / was served with the application and \
referred it to the adjudicator (typical in pre-amendment matters);
(b) "agent" - acting as the adjudicator's AGENT, e.g. "Notice of the acceptance \
was served ... by the ABC Dispute Resolution Service as my agent", "X, my \
authorised agent", "X as agent for the adjudicator", "I directed my agent to \
contact the parties";
(c) "letterhead" - appearing only in the letterhead, coversheet branding, \
signature block, or as the adjudicator's stated affiliation / trust account.

Your task:
1. referring_body: the body that referred the application to the adjudicator or \
received/was served with the adjudication application. Post-2014 this is usually \
the QBCC Adjudication Registrar / Registry - return exactly "QBCC Registry" for \
any QBCC / Adjudication Registrar variant. If an ANA received the application, \
name the ANA. Null if not identifiable.
2. ana: the Authorised Nominating Authority associated with this decision in ANY \
of roles (a), (b), (c). Capture it even when the QBCC Registrar referred the \
application - an agent or letterhead mention still counts. Null only if no ANA \
appears in any of those roles. The QBCC Registry is a statutory registrar, NOT \
an ANA - never return it as ana.
3. ana_role: which role the ANA appeared in - "referrer", "agent", or \
"letterhead". If it appears in several, prefer referrer > agent > letterhead. \
Null if ana is null.
4. evidence: a short verbatim quote from the excerpt (max 140 characters) \
supporting the ana finding (or the referring_body finding if ana is null). Null \
if nothing found.

Do NOT treat mentions of ANAs inside quoted contract clauses, cited cases, or \
background legal discussion as involvement in this adjudication.

Known ANAs - when the body matches one of these, return the canonical name \
exactly, mapping obvious variants (e.g. "Australian Building & Construction \
Dispute Resolution Service" or "ABCDRS" -> "ABC Dispute Resolution Service"; \
"adjudicate.today" -> "Adjudicate Today"):
- Adjudicate Today
- Institute of Arbitrators & Mediators Australia (IAMA)
- RICS Dispute Resolution Service
- ABC Dispute Resolution Service
- Australian Solutions Centre
- LEADR
- Queensland Law Society
- Master Builders Queensland
- Nominator
- Able Adjudication Pty Ltd
- JDLA Pty Ltd
If a genuinely different body acted in one of the roles, return its name verbatim."""

JSON_SCHEMA = {
    "name": "ana_coding_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "referring_body": {"type": ["string", "null"]},
            "ana": {"type": ["string", "null"]},
            "ana_role": {
                "anyOf": [
                    {"type": "string",
                     "enum": ["referrer", "agent", "letterhead"]},
                    {"type": "null"},
                ]
            },
            "evidence": {"type": ["string", "null"]},
        },
        "required": ["referring_body", "ana", "ana_role", "evidence"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------


def build_excerpt(full_text: str, pages) -> str:
    """First ~3 pages + last page."""
    n = len(full_text)
    if pages and pages > 0:
        cpp = max(500, n // pages)
    else:
        cpp = FALLBACK_CHARS_PER_PAGE
    head = 3 * cpp
    tail = cpp
    if n <= head + tail:
        return full_text
    return (
        full_text[:head]
        + "\n\n[... middle of decision omitted ...]\n\n"
        + full_text[n - tail:]
    )


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# Normalisation: map obvious variants onto canonical names.
_NORM_RULES = [
    (lambda q: "adjudicatetoday" in q or "adjudicate2day" in q, "Adjudicate Today"),
    (lambda q: "iama" in q or ("institute" in q and "arbitrator" in q),
     "Institute of Arbitrators & Mediators Australia (IAMA)"),
    (lambda q: q.startswith("rics") or "royalinstitutionofcharteredsurveyors" in q,
     "RICS Dispute Resolution Service"),
    (lambda q: q.startswith("abc") and ("dispute" in q or "resolution" in q
                                        or q.startswith("abcdrs")),
     "ABC Dispute Resolution Service"),
    (lambda q: "australianbuilding" in q and "disputeresolution" in q,
     "ABC Dispute Resolution Service"),
    (lambda q: "australiansolutionscentre" in q or "australiansolutioncentre" in q,
     "Australian Solutions Centre"),
    (lambda q: q == "leadr" or q.startswith("leadrassociation") or "leadrmediators" in q
     or q == "leadriama" or "resolutioninstitute" in q, "LEADR"),
    (lambda q: "queenslandlawsociety" in q or q == "qls", "Queensland Law Society"),
    (lambda q: "masterbuilders" in q, "Master Builders Queensland"),
    (lambda q: q == "nominator" or q.startswith("nominatorptyltd") or q == "thenominator",
     "Nominator"),
    (lambda q: "ableadjudication" in q, "Able Adjudication Pty Ltd"),
    (lambda q: q.startswith("jdla"), "JDLA Pty Ltd"),
    (lambda q: "qbcc" in q or "queenslandbuildingandconstructioncommission" in q
     or "adjudicationregistr" in q or "bciparegistr" in q, "QBCC Registry"),
]


def normalise(name):
    """Return canonical ANA name, 'QBCC Registry', the verbatim name, or None."""
    if name is None:
        return None
    name = name.strip().strip('."“”')
    if not name:
        return None
    q = _squash(name)
    if q in ("null", "none", "na", "notidentifiable", "unknown", "notstated",
             "bifa", "bcipa"):
        return None
    if name.lower().startswith("adjudicator "):
        return None
    for canon in CANONICAL + ["QBCC Registry"]:
        if _squash(canon) == q:
            return canon
    for pred, canon in _NORM_RULES:
        if pred(q):
            return canon
    return name


def classify(referring_body, ana, ana_role, evidence):
    """Map a model response onto DB values.

    Returns (ana, ana_role, ana_source, ana_evidence)."""
    norm_ana = normalise(ana)
    norm_ref = normalise(referring_body)
    evidence = (evidence or "").strip()[:140] or None
    if norm_ana == "QBCC Registry":
        # Registrar is not an ANA; treat as registry referral.
        norm_ana = None
        norm_ref = norm_ref or "QBCC Registry"
    if norm_ana is not None:
        role = ana_role if ana_role in ("referrer", "agent", "letterhead") else "referrer"
        return norm_ana, role, "ai_v2", evidence
    if norm_ref == "QBCC Registry":
        return "QBCC Registry", "registry", "ai_v2", evidence
    return None, None, "unknown", None


# ---------------------------------------------------------------------------


class Stats:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.done = 0
        self.errors = 0


async def call_openai(session, sem, api_key, model, ejs_id, excerpt, stats):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "DECISION EXCERPT:\n\n" + excerpt},
        ],
        "response_format": {"type": "json_schema", "json_schema": JSON_SCHEMA},
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    async with sem:
        delay = 2.0
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=body, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status in (429, 500, 502, 503, 504):
                        text = await resp.text()
                        if attempt == MAX_RETRIES:
                            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                        await asyncio.sleep(delay + random.uniform(0, 1))
                        delay *= 2
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
                    data = await resp.json()
                usage = data.get("usage", {})
                stats.prompt_tokens += usage.get("prompt_tokens", 0)
                stats.completion_tokens += usage.get("completion_tokens", 0)
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return ejs_id, parsed, None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == MAX_RETRIES:
                    return ejs_id, None, f"network: {e}"
                await asyncio.sleep(delay + random.uniform(0, 1))
                delay *= 2
            except (json.JSONDecodeError, KeyError) as e:
                return ejs_id, None, f"parse: {e}"
            except RuntimeError as e:
                return ejs_id, None, str(e)
    return ejs_id, None, "exhausted retries"


# ---------------------------------------------------------------------------


def ensure_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(decision_details)")}
    for col in ("ana", "ana_source", "ana_evidence", "ana_role"):
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE decision_details ADD COLUMN {col} TEXT")
                conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise


TARGET_FILTER = ("dd.decision_date >= ? AND "
                 "(dd.ana = 'QBCC Registry' OR dd.ana IS NULL "
                 "OR dd.ana_source = 'ai_v2')")


def load_rows(conn, limit=None, ids=None):
    sql = f"""SELECT dd.ejs_id, df.full_text, dd.doc_length_pages
              FROM decision_details dd
              JOIN docs_fresh df ON dd.ejs_id = df.ejs_id
              WHERE {TARGET_FILTER}"""
    params = [DATE_CUTOFF]
    if ids:
        sql += " AND dd.ejs_id IN (%s)" % ",".join("?" * len(ids))
        params.extend(ids)
    sql += " ORDER BY dd.ejs_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def write_batch(db_path, batch):
    """Short write transaction (another agent may be using this DB).
    Only touches target rows (re-guarded in the WHERE)."""
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for ejs_id, ana, role, source, evidence in batch:
            conn.execute(
                f"""UPDATE decision_details AS dd
                    SET ana = ?, ana_role = ?, ana_source = ?, ana_evidence = ?
                    WHERE dd.ejs_id = ? AND {TARGET_FILTER}""",
                (ana, role, source, evidence, ejs_id, DATE_CUTOFF),
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ids", default=None,
                    help="comma-separated ejs_ids (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="call the API but skip DB writes and export")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set")

    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None

    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_columns(conn)
    rows = load_rows(conn, args.limit, ids)
    conn.close()
    print(f"{len(rows)} decisions to code with model {args.model}", flush=True)

    stats = Stats()
    sem = asyncio.Semaphore(CONCURRENCY)
    results = {}  # ejs_id -> record dict
    pending_db = []
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(call_openai(
                session, sem, api_key, args.model, ejs_id,
                build_excerpt(text or "", pages), stats))
            for ejs_id, text, pages in rows
        ]
        for fut in asyncio.as_completed(tasks):
            ejs_id, parsed, err = await fut
            stats.done += 1
            if err:
                stats.errors += 1
                if stats.errors <= 20:
                    print(f"  ERROR {ejs_id}: {err}", flush=True)
                # Leave the row untouched on error.
                results[ejs_id] = {"ejs_id": ejs_id, "error": err}
            else:
                ana, role, source, evidence = classify(
                    parsed.get("referring_body"), parsed.get("ana"),
                    parsed.get("ana_role"), parsed.get("evidence"))
                results[ejs_id] = {
                    "ejs_id": ejs_id,
                    "referring_body": normalise(parsed.get("referring_body")),
                    "ana": ana,
                    "ana_role": role,
                    "ana_source": source,
                    "ana_evidence": evidence,
                }
                if not args.dry_run:
                    pending_db.append((ejs_id, ana, role, source, evidence))
            if len(pending_db) >= BATCH_COMMIT:
                write_batch(DB_PATH, pending_db)
                pending_db = []
            if stats.done % 100 == 0:
                el = time.time() - t0
                print(f"  {stats.done}/{len(rows)} done "
                      f"({el:.0f}s, {stats.prompt_tokens} in / "
                      f"{stats.completion_tokens} out tokens, "
                      f"{stats.errors} errors)", flush=True)

    if pending_db:
        write_batch(DB_PATH, pending_db)

    for r in sorted(results.values(), key=lambda x: x["ejs_id"])[:40] if ids else []:
        print(" ", json.dumps(r, ensure_ascii=False), flush=True)

    if not args.dry_run and not ids:
        os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
        export = [results[k] for k in sorted(results)]
        with open(EXPORT_PATH, "w") as f:
            json.dump(export, f, indent=1, ensure_ascii=False)
        print(f"\nWrote {len(export)} records to {EXPORT_PATH}", flush=True)

    total_tok = stats.prompt_tokens + stats.completion_tokens
    print(f"\nTokens: {stats.prompt_tokens} in, {stats.completion_tokens} out "
          f"({total_tok} total). Errors: {stats.errors}. "
          f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
