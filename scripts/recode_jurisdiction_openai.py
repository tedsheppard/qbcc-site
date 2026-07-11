#!/usr/bin/env python3
"""Recode decision_details.jurisdiction_upheld for decisions >= 2015-01-01.

Intended semantics (matches pre-2015 coding): 1 = a jurisdictional objection
succeeded, i.e. the adjudicator found they LACKED jurisdiction over the whole
application and no substantive adjudicated amount was decided. The post-2015
upload batches inverted this ("adjudicator's jurisdiction upheld"), so ~75%
of post-2015 rows are 1. This pass re-codes every post-2015 row from the
decision text via gpt-5.4-nano.

Consistency guard: model=true but adjudicated_amount > 0, or model=false but
the coded outcome plainly records a jurisdiction decline -> conflict; writes
NULL and logs, rather than silently writing a contradictory value.

Writes jurisdiction_upheld (1/0/NULL) in short batched transactions and
exports/jurisdiction_recode.json for replaying onto the live GCS DB.

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
EXPORT_PATH = os.path.join(REPO_ROOT, "exports", "jurisdiction_recode.json")

DEFAULT_MODEL = "gpt-5.4-nano"
CONCURRENCY = 30
MAX_RETRIES = 5
BATCH_COMMIT = 100
FALLBACK_CHARS_PER_PAGE = 2600
DATE_CUTOFF = "2015-01-01"

SYSTEM_PROMPT = """You are coding Queensland construction adjudication decisions \
(BCIPA 2004 or BIF Act 2017). You will be given excerpts (first ~2 pages and the \
last ~2 pages) of one decision's OCR text. OCR output often has stray spaces \
breaking words (e.g. "Con struction", "juris diction", "Adjudi cate") - mentally \
rejoin broken words when reading.

Question: did the adjudicator decide that they LACKED jurisdiction to decide the \
adjudication application, such that NO substantive adjudicated amount was decided \
on the merits?

Answer no_jurisdiction = true only when the adjudicator wholly declined \
jurisdiction over the application, for example:
- the adjudication application was found invalid (out of time, no valid payment \
claim, no reference date, not served properly);
- the contract or claim was outside the Act (e.g. s 61(2)(b) BIF Act residential \
building work exclusion, no "construction work");
- the adjudicator resigned or ceased to act for want of jurisdiction;
- the decision states there is "no jurisdiction to decide/adjudicate the \
application" and stops there.

Answer no_jurisdiction = false when the adjudicator proceeded to decide the \
application on the merits (valuing the claim, deciding an adjudicated amount - \
even $0 or a partial amount). This includes cases where a jurisdictional \
objection was raised and REJECTED, and cases where jurisdiction was declined \
only for SOME claim items but the rest were decided on the merits.

Answer no_jurisdiction = null only when the excerpt genuinely does not show \
either way.

evidence: a short verbatim quote from the excerpt (max 140 characters) \
supporting your answer; null if no_jurisdiction is null."""

JSON_SCHEMA = {
    "name": "jurisdiction_coding",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "no_jurisdiction": {"type": ["boolean", "null"]},
            "evidence": {"type": ["string", "null"]},
        },
        "required": ["no_jurisdiction", "evidence"],
        "additionalProperties": False,
    },
}

# Outcome strings that plainly record a whole-application jurisdiction decline.
DECLINE_OUTCOME = re.compile(
    r"no jurisdiction|not have jurisdiction|without jurisdiction|"
    r"lack(s|ed)? +jurisdiction", re.I)

# ---------------------------------------------------------------------------


def build_excerpt(full_text: str, pages) -> str:
    """First ~2 pages + last ~2 pages."""
    n = len(full_text)
    if pages and pages > 0:
        cpp = max(500, n // pages)
    else:
        cpp = FALLBACK_CHARS_PER_PAGE
    head = 2 * cpp
    tail = 2 * cpp
    if n <= head + tail:
        return full_text
    return (
        full_text[:head]
        + "\n\n[... middle of decision omitted ...]\n\n"
        + full_text[n - tail:]
    )


class Stats:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.done = 0
        self.errors = 0
        self.conflicts = 0


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


def load_rows(conn, limit=None, ids=None):
    sql = """SELECT dd.ejs_id, df.full_text, dd.doc_length_pages,
                    dd.adjudicated_amount, dd.outcome
             FROM decision_details dd
             JOIN docs_fresh df ON dd.ejs_id = df.ejs_id
             WHERE dd.decision_date >= ?"""
    params = [DATE_CUTOFF]
    if ids:
        sql += " AND dd.ejs_id IN (%s)" % ",".join("?" * len(ids))
        params.extend(ids)
    sql += " ORDER BY dd.ejs_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def apply_guard(no_jur, adjudicated_amount, outcome, stats):
    """Return (db_value, conflict_reason|None)."""
    try:
        amount = float(adjudicated_amount) if adjudicated_amount is not None else None
    except (TypeError, ValueError):
        amount = None
    if no_jur is True and amount is not None and amount > 0:
        stats.conflicts += 1
        return None, f"model=true but adjudicated_amount={amount}"
    if no_jur is False and outcome and DECLINE_OUTCOME.search(outcome):
        stats.conflicts += 1
        return None, f"model=false but outcome={outcome!r}"
    if no_jur is True:
        return 1, None
    if no_jur is False:
        return 0, None
    return None, None


def write_batch(db_path, batch):
    """Short write transaction; another agent may be using this DB."""
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for ejs_id, value in batch:
            conn.execute(
                """UPDATE decision_details
                   SET jurisdiction_upheld = ?
                   WHERE ejs_id = ? AND decision_date >= ?""",
                (value, ejs_id, DATE_CUTOFF),
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
    rows = load_rows(conn, args.limit, ids)
    conn.close()
    print(f"{len(rows)} decisions to code with model {args.model}", flush=True)

    meta = {r[0]: (r[3], r[4]) for r in rows}  # ejs_id -> (amount, outcome)
    stats = Stats()
    sem = asyncio.Semaphore(CONCURRENCY)
    results = {}
    conflicts = []
    pending_db = []
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(call_openai(
                session, sem, api_key, args.model, ejs_id,
                build_excerpt(text or "", pages), stats))
            for ejs_id, text, pages, _amt, _out in rows
        ]
        for fut in asyncio.as_completed(tasks):
            ejs_id, parsed, err = await fut
            stats.done += 1
            if err:
                stats.errors += 1
                if stats.errors <= 20:
                    print(f"  ERROR {ejs_id}: {err}", flush=True)
                results[ejs_id] = {"ejs_id": ejs_id, "error": err}
                # leave the row untouched on API error
            else:
                no_jur = parsed.get("no_jurisdiction")
                evidence = (parsed.get("evidence") or "").strip()[:140] or None
                amount, outcome = meta[ejs_id]
                value, conflict = apply_guard(no_jur, amount, outcome, stats)
                rec = {"ejs_id": ejs_id, "no_jurisdiction": no_jur,
                       "evidence": evidence}
                if conflict:
                    rec["conflict"] = conflict
                    rec["no_jurisdiction"] = None  # what gets written
                    conflicts.append((ejs_id, conflict))
                    print(f"  CONFLICT {ejs_id}: {conflict}", flush=True)
                results[ejs_id] = rec
                if not args.dry_run:
                    pending_db.append((ejs_id, value))
            if len(pending_db) >= BATCH_COMMIT:
                write_batch(DB_PATH, pending_db)
                pending_db = []
            if stats.done % 200 == 0:
                el = time.time() - t0
                print(f"  {stats.done}/{len(rows)} done "
                      f"({el:.0f}s, {stats.prompt_tokens} in / "
                      f"{stats.completion_tokens} out tokens, "
                      f"{stats.errors} errors, "
                      f"{stats.conflicts} conflicts)", flush=True)

    if pending_db:
        write_batch(DB_PATH, pending_db)

    if ids:
        for r in sorted(results.values(), key=lambda x: x["ejs_id"]):
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
          f"Conflicts: {stats.conflicts}. Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
