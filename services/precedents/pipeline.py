"""Firm Precedent Vault — background ingestion pipeline.

Runs on the server's existing APScheduler (wired in server.py):
  worker_tick() every 60s:
    1. extract  — queued documents -> per-page text (pdfplumber; OCR fallback),
                  encrypted into doc_pages.
    2. submit   — extracted documents -> one Anthropic Message Batch (Sonnet 5),
                  50% batch pricing.
    3. poll     — processing batches -> parse section tags, write sections + FTS,
                  record cost, mark ready.
  nightly_backup() daily: consistent sqlite backup uploaded to GCS, 14 kept.

Every stage is defensive: a failure marks the document failed/retryable and
never propagates; the tick itself is wrapped so scheduler jobs can't die.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
from datetime import datetime
from typing import Any

from . import core
from .taxonomy import TAXONOMY_VERSION, build_system_prompt, valid_pair

MODEL = os.getenv("PRECEDENTS_MODEL", "claude-sonnet-5")
# USD per 1M tokens at 50% batch pricing; override via env when prices change.
RATE_IN = float(os.getenv("PRECEDENTS_RATE_IN", "1.5"))
RATE_OUT = float(os.getenv("PRECEDENTS_RATE_OUT", "7.5"))

MAX_PAGES = 500
MAX_DOC_CHARS = 700_000          # ~200k tokens; Sonnet 5 has 1M context
MAX_SECTION_CHARS = 60_000       # cap of plaintext indexed per section
MAX_ATTEMPTS = 3
BATCH_MAX_DOCS = 20
OCR_MIN_AVG_CHARS = 150          # below this avg chars/page, assume scanned

_tick_lock = threading.Lock()


def _log(msg: str) -> None:
    print(f"[precedents] {msg}", file=sys.stderr)


# ---------------------------------------------------------------- extraction

def _extract_pdf_pages(data: bytes) -> list[str]:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages[:MAX_PAGES]:
            try:
                pages.append((page.extract_text() or "").strip())
            except Exception:
                pages.append("")
    return pages


def _ocr_pdf_pages(data: bytes, existing: list[str]) -> list[str]:
    """OCR only the pages that came back (near-)empty from pdfplumber."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except Exception as exc:
        _log(f"OCR unavailable ({exc}); keeping extracted text as-is")
        return existing
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        _log(f"OCR open failed: {exc}")
        return existing
    out = list(existing)
    for i in range(min(len(doc), MAX_PAGES)):
        if i < len(out) and len(out[i]) >= OCR_MIN_AVG_CHARS:
            continue
        try:
            pix = doc[i].get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img) or ""
            if i < len(out):
                out[i] = text.strip() or out[i]
            else:
                out.append(text.strip())
        except Exception:
            continue
    doc.close()
    return out


def _extract_docx_pages(data: bytes) -> list[str]:
    import docx2txt
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        text = docx2txt.process(tmp.name) or ""
    # DOCX has no fixed pagination; chunk into pseudo-pages so page refs
    # in the taxonomy output still roughly locate content.
    text = text.strip()
    if not text:
        return []
    chunk = 4500
    return [text[i : i + chunk] for i in range(0, min(len(text), MAX_DOC_CHARS), chunk)]


def _extract_one(con, doc) -> None:
    doc_id = doc["id"]
    con.execute(
        "UPDATE documents SET status='extracting', updated_at=? WHERE id=?",
        (core.now_iso(), doc_id),
    )
    con.commit()
    data = core.read_blob(doc["storage"], doc["blob_path"])
    name = (doc["filename"] or "").lower()
    if name.endswith(".docx"):
        pages = _extract_docx_pages(data)
    else:
        pages = _extract_pdf_pages(data)
        total = sum(len(p) for p in pages)
        if pages and total / max(len(pages), 1) < OCR_MIN_AVG_CHARS:
            _log(f"doc {doc_id}: low text density, running OCR fallback")
            pages = _ocr_pdf_pages(data, pages)
    pages = [p[:20_000] for p in pages]
    total_chars = sum(len(p) for p in pages)
    if total_chars < 200:
        raise RuntimeError("No extractable text (extraction and OCR both came back empty)")
    with core._con_lock:
        con.execute("DELETE FROM doc_pages WHERE doc_id=?", (doc_id,))
        for i, text in enumerate(pages, start=1):
            con.execute(
                "INSERT INTO doc_pages (doc_id, page_no, text_enc, chars) VALUES (?,?,?,?)",
                (doc_id, i, core.encrypt_text(text), len(text)),
            )
        con.execute(
            "UPDATE documents SET status='extracted', pages=?, tokens_est=?, error='', updated_at=? WHERE id=?",
            (len(pages), int(total_chars / 3.5), core.now_iso(), doc_id),
        )
        con.commit()


def _stage_extract(con) -> None:
    rows = con.execute(
        "SELECT * FROM documents WHERE status IN ('queued','extracting') AND attempts < ? ORDER BY created_at LIMIT 3",
        (MAX_ATTEMPTS,),
    ).fetchall()
    for doc in rows:
        try:
            _extract_one(con, doc)
            _log(f"extracted doc {doc['id']} ({doc['filename']})")
        except Exception as exc:
            _log(f"extract failed for doc {doc['id']}: {exc}")
            with core._con_lock:
                con.execute(
                    "UPDATE documents SET status=?, error=?, attempts=attempts+1, updated_at=? WHERE id=?",
                    (
                        "failed" if doc["attempts"] + 1 >= MAX_ATTEMPTS else "queued",
                        str(exc)[:500],
                        core.now_iso(),
                        doc["id"],
                    ),
                )
                con.commit()


# ---------------------------------------------------------------- batch submit

def _doc_content(con, doc_id: str) -> str:
    pages = con.execute(
        "SELECT page_no, text_enc FROM doc_pages WHERE doc_id=? ORDER BY page_no", (doc_id,)
    ).fetchall()
    parts, running = [], 0
    for p in pages:
        text = core.decrypt_text(p["text_enc"])
        if running + len(text) > MAX_DOC_CHARS:
            text = text[: max(0, MAX_DOC_CHARS - running)]
        running += len(text)
        parts.append(f"[PAGE {p['page_no']}]\n{text}")
        if running >= MAX_DOC_CHARS:
            break
    return "\n\n".join(parts)


def _anthropic_client():
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    return anthropic.Anthropic()


def _stage_submit(con) -> None:
    docs = con.execute(
        "SELECT id, firm_id FROM documents WHERE status='extracted' ORDER BY created_at LIMIT ?",
        (BATCH_MAX_DOCS,),
    ).fetchall()
    if not docs:
        return
    client = _anthropic_client()
    if client is None:
        _log("ANTHROPIC_API_KEY not set; leaving extracted docs pending")
        return
    system_prompt = build_system_prompt()
    requests = []
    for d in docs:
        content = _doc_content(con, d["id"])
        requests.append(
            {
                "custom_id": d["id"],
                "params": {
                    "model": MODEL,
                    "max_tokens": 8000,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": content}],
                },
            }
        )
    batch = client.messages.batches.create(requests=requests)
    with core._con_lock:
        con.execute(
            "INSERT OR REPLACE INTO batches (batch_id, status, doc_count, created_at) VALUES (?,?,?,?)",
            (batch.id, "processing", len(docs), core.now_iso()),
        )
        for d in docs:
            con.execute(
                "UPDATE documents SET status='classifying', batch_id=?, taxonomy_version=?, updated_at=? WHERE id=?",
                (batch.id, TAXONOMY_VERSION, core.now_iso(), d["id"]),
            )
        con.commit()
    _log(f"submitted batch {batch.id} with {len(docs)} docs")


# ---------------------------------------------------------------- batch poll

_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)


def _extract_json(text: str) -> dict[str, Any] | None:
    m = _FENCE_RE.search(text)
    candidate = text[m.end():] if m else text
    end_fence = candidate.find("```")
    if end_fence != -1:
        candidate = candidate[:end_fence]
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    raw = candidate[start : end + 1]
    for attempt in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _section_text(con, doc_id: str, page_start: int, page_end: int) -> str:
    rows = con.execute(
        "SELECT text_enc FROM doc_pages WHERE doc_id=? AND page_no BETWEEN ? AND ? ORDER BY page_no",
        (doc_id, page_start, page_end),
    ).fetchall()
    text = "\n".join(core.decrypt_text(r["text_enc"]) for r in rows)
    return text[:MAX_SECTION_CHARS]


def _apply_result(con, doc, parsed: dict[str, Any]) -> int:
    doc_id, firm_id = doc["id"], doc["firm_id"]
    meta = parsed.get("document") or {}
    sections = parsed.get("sections") or []
    n_pages = int(doc["pages"] or 1)
    written = 0
    with core._con_lock:
        con.execute("DELETE FROM sections_fts WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM sections WHERE doc_id=?", (doc_id,))
        for s in sections:
            if not isinstance(s, dict):
                continue
            cat, sub = valid_pair(str(s.get("category", "")), str(s.get("subcategory", "")))
            try:
                p1 = max(1, min(int(s.get("page_start", 1)), n_pages))
                p2 = max(p1, min(int(s.get("page_end", p1)), n_pages))
            except Exception:
                p1, p2 = 1, n_pages
            heading = str(s.get("heading", ""))[:300]
            summary = str(s.get("summary", ""))[:2000]
            try:
                confidence = max(0.0, min(float(s.get("confidence", 0)), 1.0))
            except Exception:
                confidence = 0.0
            cur = con.execute(
                "INSERT INTO sections (doc_id, firm_id, category, subcategory, heading, summary, page_start, page_end, confidence, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (doc_id, firm_id, cat, sub, heading, summary, p1, p2, confidence, core.now_iso()),
            )
            section_id = cur.lastrowid
            con.execute(
                "INSERT INTO sections_fts (section_text, summary, heading, section_id, doc_id, firm_id) VALUES (?,?,?,?,?,?)",
                (_section_text(con, doc_id, p1, p2), summary, heading, str(section_id), doc_id, firm_id),
            )
            written += 1
        con.execute(
            "UPDATE documents SET status='ready', title=?, doc_type=?, party_side=?, act=?, doc_date_hint=?, error='', updated_at=? WHERE id=?",
            (
                str(meta.get("title", ""))[:300],
                str(meta.get("doc_type", ""))[:120],
                str(meta.get("party_side", ""))[:40],
                str(meta.get("act", ""))[:40],
                str(meta.get("date_hint", ""))[:40],
                core.now_iso(),
                doc_id,
            ),
        )
        con.commit()
    return written


def _record_cost(con, doc, usage) -> None:
    try:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cost = input_tokens * RATE_IN / 1e6 + output_tokens * RATE_OUT / 1e6
        with core._con_lock:
            con.execute(
                "INSERT INTO cost_ledger (firm_id, doc_id, model, input_tokens, output_tokens, cost_usd, created_at) VALUES (?,?,?,?,?,?,?)",
                (doc["firm_id"], doc["id"], MODEL, input_tokens, output_tokens, round(cost, 6), core.now_iso()),
            )
            con.commit()
    except Exception as exc:
        _log(f"cost ledger write failed: {exc}")


def _stage_poll(con) -> None:
    batches = con.execute("SELECT batch_id FROM batches WHERE status='processing'").fetchall()
    if not batches:
        return
    client = _anthropic_client()
    if client is None:
        return
    for b in batches:
        batch_id = b["batch_id"]
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except Exception as exc:
            _log(f"batch retrieve failed {batch_id}: {exc}")
            continue
        if batch.processing_status != "ended":
            continue
        try:
            for result in client.messages.batches.results(batch_id):
                doc = con.execute("SELECT * FROM documents WHERE id=?", (result.custom_id,)).fetchone()
                if doc is None or doc["batch_id"] != batch_id:
                    continue
                rtype = result.result.type
                if rtype == "succeeded":
                    message = result.result.message
                    text = next((blk.text for blk in message.content if getattr(blk, "type", "") == "text"), "")
                    parsed = _extract_json(text)
                    _record_cost(con, doc, getattr(message, "usage", None))
                    if parsed is None:
                        with core._con_lock:
                            con.execute(
                                "UPDATE documents SET status='needs_review', error='Model returned unparseable classification', raw_result_enc=?, updated_at=? WHERE id=?",
                                (core.encrypt_text(text[:200_000]), core.now_iso(), doc["id"]),
                            )
                            con.commit()
                    else:
                        n = _apply_result(con, doc, parsed)
                        with core._con_lock:
                            con.execute(
                                "UPDATE documents SET raw_result_enc=? WHERE id=?",
                                (core.encrypt_text(text[:200_000]), doc["id"]),
                            )
                            con.commit()
                        _log(f"doc {doc['id']} classified: {n} sections")
                elif rtype in ("errored", "expired", "canceled"):
                    retryable = rtype != "canceled"
                    next_status = "extracted" if (retryable and doc["attempts"] + 1 < MAX_ATTEMPTS) else "failed"
                    with core._con_lock:
                        con.execute(
                            "UPDATE documents SET status=?, error=?, attempts=attempts+1, batch_id='', updated_at=? WHERE id=?",
                            (next_status, f"batch result: {rtype}", core.now_iso(), doc["id"]),
                        )
                        con.commit()
            with core._con_lock:
                con.execute(
                    "UPDATE batches SET status='ended', ended_at=? WHERE batch_id=?",
                    (core.now_iso(), batch_id),
                )
                con.commit()
            _log(f"batch {batch_id} completed")
        except Exception as exc:
            _log(f"batch results processing failed {batch_id}: {exc}")


# ---------------------------------------------------------------- entry points

def worker_tick() -> None:
    """Scheduler entry point. Never raises; never overlaps itself."""
    if not _tick_lock.acquire(blocking=False):
        return
    try:
        con = core.get_con()
        _stage_extract(con)
        _stage_submit(con)
        _stage_poll(con)
    except Exception as exc:
        _log(f"worker_tick error: {exc}")
    finally:
        _tick_lock.release()


def nightly_backup() -> None:
    """Consistent sqlite backup of precedents.db uploaded to GCS; keep 14."""
    try:
        import sqlite3

        client = core._get_gcs_client()
        if client is None:
            _log("backup skipped: GCS unavailable")
            return
        src = core.get_con()
        tmp_path = "/tmp/precedents_backup.db"
        dst = sqlite3.connect(tmp_path)
        with dst:
            src.backup(dst)
        dst.close()
        stamp = datetime.utcnow().strftime("%Y%m%d")
        bucket = client.bucket(core.GCS_BUCKET)
        bucket.blob(f"{core.GCS_PREFIX}_backups/precedents-{stamp}.db").upload_from_filename(tmp_path)
        os.remove(tmp_path)
        backups = sorted(
            bucket.list_blobs(prefix=f"{core.GCS_PREFIX}_backups/"), key=lambda b: b.name
        )
        for old in backups[:-14]:
            old.delete()
        _log(f"backup uploaded ({stamp}), {min(len(backups), 14)} retained")
    except Exception as exc:
        _log(f"nightly_backup error: {exc}")
