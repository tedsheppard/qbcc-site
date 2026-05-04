"""QBCC adjudication decision chunker.

Source: qbcc.db — 7,483 rows in `docs_fresh` (full_text) joined to
`decision_details` (metadata). 7,465 of these have full_text > 1000 chars.

OCR quality is mixed. The chunker is conservative — splits on numbered
paragraph markers when present, otherwise on character windows of ~2400
chars. Each chunk's header line includes decision_id, parties, adjudicator,
date so the chunk is identifiable even with garbled body text.

Schema (from AUDIT.md):
  docs_fresh: ejs_id PK, reference, pdf_path, full_text, id
  decision_details: ejs_id PK, adjudicator_name, claimant_name,
                    respondent_name, decision_date, sections_referenced,
                    outcome, raw_json, ...
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterator

from .base import Chunk

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = ROOT / "qbcc.db"

MAX_CHARS_PER_CHUNK = 2400
HARD_MAX = MAX_CHARS_PER_CHUNK * 3  # 7,200 chars
PARA_RE = re.compile(r"^\s*\[?(\d{1,4})[\]\.\)]\s+", re.MULTILINE)
MIN_FULL_TEXT_CHARS = 1000


def _is_para_num(n: int) -> bool:
    return n < 1000 or n > 2200


def _chunk_decision_text(text: str) -> list[tuple[int, int, str]]:
    """Split decision body. Returns (para_start, para_end, chunk_text)."""
    matches: list[tuple[int, int]] = []
    for m in PARA_RE.finditer(text):
        n = int(m.group(1))
        if _is_para_num(n):
            matches.append((n, m.start()))

    if not matches:
        # No paragraph markers — char-window
        out = []
        i = 0
        while i < len(text):
            out.append((0, 0, text[i:i + HARD_MAX].strip()))
            i += HARD_MAX
        return out

    spans: list[tuple[int, int, int]] = []
    for i, (n, start) in enumerate(matches):
        end = matches[i + 1][1] if i + 1 < len(matches) else len(text)
        spans.append((n, start, end))

    out: list[tuple[int, int, str]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_text = ""

    def flush():
        nonlocal cur_text, cur_start, cur_end
        if cur_text.strip() and cur_start is not None:
            out.append((cur_start, cur_end or cur_start, cur_text.strip()))
        cur_text = ""
        cur_start = None
        cur_end = None

    for para_num, span_start, span_end in spans:
        piece = text[span_start:span_end]
        if len(piece) > HARD_MAX:
            if cur_text:
                flush()
            i = 0
            while i < len(piece):
                out.append((para_num, para_num, piece[i:i + HARD_MAX].strip()))
                i += HARD_MAX
            continue
        if cur_text and len(cur_text) + len(piece) > MAX_CHARS_PER_CHUNK:
            flush()
        if not cur_text:
            cur_start = para_num
        cur_end = para_num
        cur_text += piece
    flush()
    return out


def chunk_all(limit: int | None = None) -> Iterator[Chunk]:
    if not DB_PATH.exists():
        return
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    sql = """
        SELECT d.ejs_id, d.reference, d.full_text,
               dd.adjudicator_name, dd.claimant_name, dd.respondent_name,
               dd.decision_date, dd.sections_referenced, dd.outcome,
               dd.act_category
        FROM docs_fresh d
        LEFT JOIN decision_details dd ON d.ejs_id = dd.ejs_id
        WHERE d.full_text IS NOT NULL
          AND length(d.full_text) >= ?
        ORDER BY d.ejs_id
    """
    params: list = [MIN_FULL_TEXT_CHARS]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    cursor = con.execute(sql, params)

    for row in cursor:
        ejs_id = row["ejs_id"]
        reference = row["reference"] or ejs_id
        full_text = row["full_text"] or ""
        if len(full_text) < MIN_FULL_TEXT_CHARS:
            continue

        adjudicator = (row["adjudicator_name"] or "").strip()
        claimant = (row["claimant_name"] or "").strip()
        respondent = (row["respondent_name"] or "").strip()
        date = (row["decision_date"] or "").strip()

        parties_short = ""
        if claimant or respondent:
            parties_short = f"{claimant or '?'} v {respondent or '?'}"

        chunks = _chunk_decision_text(full_text)
        for ci, (ps, pe, chunk_text) in enumerate(chunks):
            if not chunk_text or len(chunk_text) < 80:
                continue
            # Always include chunk index ci to keep ids unique
            if ps and pe and ps != pe:
                sid = f"dec_{ejs_id}_p{ps}-{pe}_c{ci}"
            elif ps:
                sid = f"dec_{ejs_id}_p{ps}_c{ci}"
            else:
                sid = f"dec_{ejs_id}_part{ci}"
            header_parts = [f"Adjudication Decision {reference}"]
            if parties_short:
                header_parts.append(f"— {parties_short}")
            if date:
                header_parts.append(f"({date})")
            if ps:
                header_parts.append(f"at [{ps}]" if ps == pe else f"at [{ps}]–[{pe}]")
            header = " ".join(header_parts)

            yield Chunk(
                source_id=sid,
                source_type="decision",
                header=header,
                text=chunk_text,
                metadata={
                    "decision_id": reference,
                    "ejs_id": ejs_id,
                    "parties": parties_short,
                    "claimant": claimant,
                    "respondent": respondent,
                    "adjudicator": adjudicator,
                    "decision_date": date,
                    "sections_referenced": row["sections_referenced"],
                    "outcome": row["outcome"],
                    "paragraph_start": ps if ps else None,
                    "paragraph_end": pe if pe else None,
                },
            )
    con.close()


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    n = 0
    sizes = []
    for c in chunk_all(limit=limit):
        n += 1
        sizes.append(len(c.text))
        if n % 5000 == 0:
            print(f"  ...{n} chunks", file=sys.stderr)
    print(f"total decision chunks: {n}")
    if sizes:
        print(f"size: avg={sum(sizes)//len(sizes)}  max={max(sizes)}  min={min(sizes)}")
