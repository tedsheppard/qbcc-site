"""Pure SQLite reads for full statutory provisions and full case judgments.

Used by the read-full-cases (HardQuestionPipeline) path to assemble the raw
text the reader-model and reasoner work over. No LLM calls, no scoring —
just deterministic fan-out from name_index keys to the chunk text in
chunks.sqlite, joined back in corpus order.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHUNKS_DB = ROOT / "store" / "chunks.sqlite"
NAME_DB = ROOT / "store" / "name_index.sqlite"

# Same map as name_index.py — duplicated here so this module has no
# import-time dependency on name_index.
ACT_SHORT_TO_KEY = {
    "BIF Act":  "bif_act",
    "BIF Reg":  "bif_reg",
    "QBCC Act": "qbcc_act",
    "QBCC Reg": "qbcc_reg",
    "AIA":      "aia",
}

_chunks_conn: sqlite3.Connection | None = None
_name_conn: sqlite3.Connection | None = None


def _chunks_db() -> sqlite3.Connection:
    global _chunks_conn
    if _chunks_conn is None:
        if not CHUNKS_DB.exists():
            raise FileNotFoundError(f"chunks.sqlite missing: {CHUNKS_DB}")
        _chunks_conn = sqlite3.connect(CHUNKS_DB)
        _chunks_conn.row_factory = sqlite3.Row
    return _chunks_conn


def _name_db() -> sqlite3.Connection:
    global _name_conn
    if _name_conn is None:
        if not NAME_DB.exists():
            raise FileNotFoundError(
                f"name_index.sqlite missing: {NAME_DB} — "
                "run `python3 -m services.bif_research.name_index build`"
            )
        _name_conn = sqlite3.connect(NAME_DB)
        _name_conn.row_factory = sqlite3.Row
    return _name_conn


def _norm_section(section: str) -> str:
    return re.sub(r"[^\w]", "", str(section)).lower()


def _join_chunk_texts(rows: list[sqlite3.Row]) -> str:
    parts = []
    for r in rows:
        header = r["header"] or ""
        text = r["text"] or ""
        parts.append(f"{header}\n\n{text}" if header else text)
    return "\n\n".join(parts)


def fetch_section_full(act_short: str, section: str) -> dict | None:
    """Return all chunks for a (act, section) joined as a single text block,
    plus metadata. Uses provision_index plus chunks.sqlite. Returns None
    if not in corpus.

    Output:
      {"act": str, "section": str, "header": str, "full_text": str,
       "chunk_ids": list[str]}
    """
    act_key = ACT_SHORT_TO_KEY.get(act_short)
    if not act_key:
        return None
    provision_key = f"{act_key}_s{_norm_section(section)}"

    name = _name_db()
    prov = name.execute(
        "SELECT * FROM provision_index WHERE provision_key=?",
        (provision_key,),
    ).fetchone()
    if not prov:
        return None

    chunk_id_rows = name.execute(
        "SELECT chunk_id FROM provision_chunks WHERE provision_key=? "
        "ORDER BY rank_within_section",
        (provision_key,),
    ).fetchall()
    chunk_ids = [r["chunk_id"] for r in chunk_id_rows]
    if not chunk_ids:
        return None

    chunks = _chunks_db()
    placeholders = ",".join("?" for _ in chunk_ids)
    raw = chunks.execute(
        f"SELECT chunk_id, header, text FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    by_id = {r["chunk_id"]: r for r in raw}
    ordered = [by_id[cid] for cid in chunk_ids if cid in by_id]

    section_title = prov["section_title"] or ""
    header = (
        f"{prov['act_short']} s {prov['section_number']}"
        + (f" — {section_title}" if section_title else "")
    )

    return {
        "act": prov["act_short"],
        "section": prov["section_number"],
        "header": header,
        "full_text": _join_chunk_texts(ordered),
        "chunk_ids": chunk_ids,
    }


def fetch_case_full(case_id: str) -> dict | None:
    """Return all chunks for a case_id joined as a single text block in
    paragraph order. Uses case_index + chunks.sqlite. Returns None if
    not in corpus.

    Output:
      {"case_id": str, "case_name": str, "citation": str,
       "court": str, "year": int, "full_text": str,
       "chunk_ids": list[str], "paragraph_count": int}
    """
    name = _name_db()
    case = name.execute(
        "SELECT * FROM case_index WHERE case_id=?",
        (case_id,),
    ).fetchone()
    if not case:
        return None

    chunk_rows = name.execute(
        "SELECT chunk_id, paragraph_end FROM case_chunks WHERE case_id=? "
        "ORDER BY rank_within_case",
        (case_id,),
    ).fetchall()
    chunk_ids = [r["chunk_id"] for r in chunk_rows]
    if not chunk_ids:
        return None

    max_para_row = name.execute(
        "SELECT MAX(paragraph_end) AS max_para FROM case_chunks WHERE case_id=?",
        (case_id,),
    ).fetchone()
    max_para = max_para_row["max_para"] if max_para_row else None
    paragraph_count = int(max_para) if max_para else len(chunk_ids)

    chunks = _chunks_db()
    placeholders = ",".join("?" for _ in chunk_ids)
    raw = chunks.execute(
        f"SELECT chunk_id, header, text FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    by_id = {r["chunk_id"]: r for r in raw}
    ordered = [by_id[cid] for cid in chunk_ids if cid in by_id]

    try:
        year = int(case["year"]) if case["year"] else 0
    except (TypeError, ValueError):
        year = 0

    return {
        "case_id": case["case_id"],
        "case_name": case["case_name"] or "",
        "citation": case["citation"] or "",
        "court": case["court"] or "",
        "year": year,
        "full_text": _join_chunk_texts(ordered),
        "chunk_ids": chunk_ids,
        "paragraph_count": paragraph_count,
    }
