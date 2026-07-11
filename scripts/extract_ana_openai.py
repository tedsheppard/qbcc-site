#!/usr/bin/env python3
"""Code which Authorised Nominating Authority (ANA) handled each adjudication
decision dated >= 2015-01-01, using an OpenAI nano-tier model over the first
~2 pages and last page of each decision's OCR text.

Writes results into decision_details (ana, ana_source, ana_evidence) and
exports/ana_ai.json.

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
EXPORT_PATH = os.path.join(REPO_ROOT, "exports", "ana_ai.json")

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
    "QBCC Registry",
]

SYSTEM_PROMPT = """You are coding Queensland construction adjudication decisions.
You will be given excerpts (first ~2 pages and last page) of one decision's text.
The text is OCR output and often has stray spaces breaking words (e.g. "Con struction",
"Adjudi cate") - mentally rejoin broken words when reading.

Task: identify the Authorised Nominating Authority (ANA) that actually RECEIVED /
handled this adjudication application. It is usually named on the coversheet, in the
letterhead, in a statement like "the application was made to / lodged with / referred
by ...", or in the closing/signature block. Carefully DISTINGUISH it from ANAs that are
merely mentioned in contract clauses, cited cases, or background discussion.

If the referring body is the QBCC Adjudication Registry / Queensland Building and
Construction Commission Registry (the post-2014 Queensland statutory referrer, sometimes
"the Registrar" or "Adjudication Registrar"), return exactly "QBCC Registry".

When the identified body matches one of these canonical names, return the canonical
form exactly:
- Adjudicate Today
- Institute of Arbitrators & Mediators Australia (IAMA)
- RICS Dispute Resolution Service
- ABC Dispute Resolution Service
- Australian Solutions Centre
- LEADR
- Queensland Law Society
- Master Builders Queensland
- Nominator
- QBCC Registry

If a different/unknown authority handled the application, return its name verbatim.
If no handling authority is identifiable, return null.

Respond with JSON: {"ana": <name or null>, "evidence": <short verbatim quote from the
excerpt, max 120 characters, supporting your answer; empty string if ana is null>}."""

JSON_SCHEMA = {
    "name": "ana_coding",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ana": {"type": ["string", "null"]},
            "evidence": {"type": "string"},
        },
        "required": ["ana", "evidence"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------


def build_excerpt(full_text: str, pages) -> str:
    n = len(full_text)
    if pages and pages > 0:
        cpp = max(500, n // pages)
    else:
        cpp = FALLBACK_CHARS_PER_PAGE
    head = 2 * cpp
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
    # (predicate on squashed name, canonical)
    (lambda q: "adjudicatetoday" in q or "adjudicate2day" in q, "Adjudicate Today"),
    (lambda q: "iama" in q or ("institute" in q and "arbitrator" in q),
     "Institute of Arbitrators & Mediators Australia (IAMA)"),
    (lambda q: q.startswith("rics") or "royalinstitutionofcharteredsurveyors" in q,
     "RICS Dispute Resolution Service"),
    (lambda q: q.startswith("abc") and ("dispute" in q or "resolution" in q or q == "abcdrs"),
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
    (lambda q: "qbcc" in q or "queenslandbuildingandconstructioncommission" in q
     or "adjudicationregistr" in q or "bciparegistr" in q, "QBCC Registry"),
]


def normalise(name):
    if name is None:
        return None
    name = name.strip().strip('."“”')
    if not name:
        return None
    q = _squash(name)
    if q in ("null", "none", "na", "notidentifiable", "unknown", "notstated",
             "bifa", "bcipa"):  # Act names, not authorities
        return None
    if name.lower().startswith("adjudicator "):  # an adjudicator, not an ANA
        return None
    for canon in CANONICAL:
        if _squash(canon) == q:
            return canon
    for pred, canon in _NORM_RULES:
        if pred(q):
            return canon
    return name


# ---------------------------------------------------------------------------


class Stats:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.done = 0
        self.errors = 0


async def call_openai(session, sem, api_key, model, ejs_id, excerpt, stats,
                      use_schema=True):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "DECISION EXCERPT:\n\n" + excerpt},
        ],
    }
    if use_schema:
        body["response_format"] = {"type": "json_schema", "json_schema": JSON_SCHEMA}
    else:
        body["response_format"] = {"type": "json_object"}

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
                ana = parsed.get("ana")
                evidence = (parsed.get("evidence") or "")[:120]
                return ejs_id, normalise(ana), evidence, None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == MAX_RETRIES:
                    return ejs_id, None, "", f"network: {e}"
                await asyncio.sleep(delay + random.uniform(0, 1))
                delay *= 2
            except (json.JSONDecodeError, KeyError) as e:
                return ejs_id, None, "", f"parse: {e}"
            except RuntimeError as e:
                return ejs_id, None, "", str(e)
    return ejs_id, None, "", "exhausted retries"


# ---------------------------------------------------------------------------


def ensure_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(decision_details)")}
    for col in ("ana", "ana_source", "ana_evidence"):
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE decision_details ADD COLUMN {col} TEXT")
                conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise


def load_rows(conn, limit=None, only_uncoded=False):
    sql = """SELECT dd.ejs_id, df.full_text, dd.doc_length_pages
             FROM decision_details dd
             JOIN docs_fresh df ON dd.ejs_id = df.ejs_id
             WHERE dd.decision_date >= ?"""
    if only_uncoded:
        sql += " AND dd.ana_source IS NULL"
    sql += " ORDER BY dd.ejs_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, (DATE_CUTOFF,)).fetchall()


def write_batch(db_path, batch):
    """Short write transaction; only touches rows >= cutoff by construction
    (ejs_ids were selected with the date filter), but re-guard anyway."""
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for ejs_id, ana, evidence in batch:
            source = "ai" if ana is not None else "unknown"
            conn.execute(
                """UPDATE decision_details
                   SET ana = ?, ana_source = ?, ana_evidence = ?
                   WHERE ejs_id = ? AND decision_date >= ?""",
                (ana, source, evidence if ana is not None else None,
                 ejs_id, DATE_CUTOFF),
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------


async def list_models(api_key):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as resp:
            data = await resp.json()
    return sorted(m["id"] for m in data.get("data", []))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--only-uncoded", action="store_true",
                    help="skip rows whose ana_source is already set")
    ap.add_argument("--no-schema", action="store_true",
                    help="use json_object instead of strict json_schema")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set")

    if args.list_models:
        for m in await list_models(api_key):
            print(m)
        return

    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_columns(conn)
    rows = load_rows(conn, args.limit, args.only_uncoded)
    conn.close()
    print(f"{len(rows)} decisions to code with model {args.model}", flush=True)

    stats = Stats()
    sem = asyncio.Semaphore(CONCURRENCY)
    results = {}  # ejs_id -> (ana, evidence)
    pending_db = []
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(call_openai(
                session, sem, api_key, args.model, ejs_id,
                build_excerpt(text or "", pages), stats,
                use_schema=not args.no_schema))
            for ejs_id, text, pages in rows
        ]
        for fut in asyncio.as_completed(tasks):
            ejs_id, ana, evidence, err = await fut
            if err:
                stats.errors += 1
                if stats.errors <= 20:
                    print(f"  ERROR {ejs_id}: {err}", flush=True)
            results[ejs_id] = (ana, evidence)
            pending_db.append((ejs_id, ana, evidence))
            stats.done += 1
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

    # JSON export
    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    export = [
        {"ejs_id": e, "ana": a,
         "ana_source": "ai" if a is not None else "unknown",
         "ana_evidence": ev if a is not None else None}
        for e, (a, ev) in sorted(results.items())
    ]
    with open(EXPORT_PATH, "w") as f:
        json.dump(export, f, indent=1, ensure_ascii=False)
    print(f"\nWrote {len(export)} records to {EXPORT_PATH}", flush=True)

    # Distribution
    conn = sqlite3.connect(DB_PATH, timeout=60)
    print("\n=== ANA distribution (rows >= %s) ===" % DATE_CUTOFF)
    for ana, cnt in conn.execute(
        """SELECT COALESCE(ana, '(null)') , COUNT(*)
           FROM decision_details WHERE decision_date >= ?
           GROUP BY ana ORDER BY COUNT(*) DESC""", (DATE_CUTOFF,)):
        print(f"  {cnt:5d}  {ana}")

    print("\n=== 25 random samples ===")
    for r in conn.execute(
        """SELECT ejs_id, decision_date, ana, ana_evidence
           FROM decision_details
           WHERE decision_date >= ? AND ana_source IS NOT NULL
           ORDER BY RANDOM() LIMIT 25""", (DATE_CUTOFF,)):
        print(f"  {r[0]}  {r[1]}  ana={r[2]!r}  evidence={r[3]!r}")
    conn.close()

    total_tok = stats.prompt_tokens + stats.completion_tokens
    print(f"\nTokens: {stats.prompt_tokens} in, {stats.completion_tokens} out "
          f"({total_tok} total). Errors: {stats.errors}. "
          f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
