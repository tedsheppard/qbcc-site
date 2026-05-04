"""Queensland judgment chunker.

Source files in `exports/qld_judgments_txt/` start with a 3-line header:
    CITATION: <case name> [YEAR] COURT NN
    COURT: <COURT_CODE>
    YEAR: <YYYY>

Followed by a structured court header (PARTIES, JUDGES, ORDERS, CATCHWORDS).
The body uses paragraph markers `[1]`, `[2]`, ... at the start of each
numbered paragraph. The same `[N]` notation also appears for year refs
(e.g. `[2008]`) so the regex must accept only paragraph-number ranges.

Filename convention: COURT_YYYY_N.txt (e.g. QSC_2017_85.txt).

Strategy:
- Read header (first 3 lines) for citation, court, year.
- Locate body start (first occurrence of `[1]` at line start).
- Walk paragraph markers; group consecutive paragraphs up to ~500 tokens
  (~2000 chars) per chunk while preserving all paragraph markers.
- Header line per chunk: e.g. "Lean Field Developments v E & I Global Solutions [2014] QSC 293 at [30]–[34]".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .base import Chunk

ROOT = Path(__file__).resolve().parent.parent.parent.parent
JUDGMENTS_DIR = ROOT / "exports" / "qld_judgments_txt"

# Paragraph-number marker at the start of a line. Accept up to 4 digits but
# reject 4-digit years (1000-2999).
PARA_RE = re.compile(r"^\s*\[(\d{1,4})\]\s", re.MULTILINE)

CITATION_RE = re.compile(r"^CITATION:\s*(.+?)\s*$", re.MULTILINE)
COURT_RE = re.compile(r"^COURT:\s*(.+?)\s*$", re.MULTILINE)
YEAR_RE = re.compile(r"^YEAR:\s*(\d{4})\s*$", re.MULTILINE)

MAX_CHARS_PER_CHUNK = 2400  # ~600 tokens
MIN_TARGET_CHARS = 800

# Cap on judgments to index — keep all 393 by default; can reduce for cost.
MAX_JUDGMENTS = None  # None = all


def _is_paragraph_number(n: int) -> bool:
    """True if n looks like a paragraph number (not a year)."""
    return n < 1000 or n > 2200


def _parse_header(text: str) -> tuple[str, str, str]:
    """Return (citation, court, year)."""
    head_block = text[:600]
    cite = CITATION_RE.search(head_block)
    court = COURT_RE.search(head_block)
    year = YEAR_RE.search(head_block)
    return (
        cite.group(1).strip() if cite else "",
        court.group(1).strip() if court else "",
        year.group(1).strip() if year else "",
    )


def _chunk_judgment_text(text: str) -> list[tuple[int, int, str]]:
    """Split the body into (para_start, para_end, chunk_text) tuples.

    Walks paragraph markers and groups consecutive paragraphs up to
    MAX_CHARS_PER_CHUNK while preserving the markers.
    """
    matches = []
    for m in PARA_RE.finditer(text):
        n = int(m.group(1))
        if _is_paragraph_number(n):
            matches.append((n, m.start()))

    if not matches:
        # No numbered paragraphs found. Fall back to char-window chunking,
        # treating each window as paragraphs ranged 0-0.
        out: list[tuple[int, int, str]] = []
        i = 0
        while i < len(text):
            chunk = text[i:i + MAX_CHARS_PER_CHUNK]
            out.append((0, 0, chunk))
            i += MAX_CHARS_PER_CHUNK
        return out

    # Build paragraph spans
    spans: list[tuple[int, int, str]] = []  # (para_num, start, end)
    for i, (n, start) in enumerate(matches):
        end = matches[i + 1][1] if i + 1 < len(matches) else len(text)
        spans.append((n, start, end))

    # Pack consecutive paragraphs up to MAX_CHARS
    out: list[tuple[int, int, str]] = []
    cur_start_para = None
    cur_end_para = None
    cur_text = ""
    cur_offset = None

    def flush():
        nonlocal cur_text, cur_start_para, cur_end_para, cur_offset
        if cur_text.strip() and cur_start_para is not None:
            out.append((cur_start_para, cur_end_para or cur_start_para, cur_text.strip()))
        cur_text = ""
        cur_start_para = None
        cur_end_para = None
        cur_offset = None

    HARD_MAX = MAX_CHARS_PER_CHUNK * 3  # 7,200 chars — embed API still happy
    for para_num, span_start, span_end in spans:
        piece = text[span_start:span_end]
        # Edge case: single paragraph exceeds the soft window. Hard-split it
        # at char boundaries so we don't produce 100K-char chunks (which blow
        # embedding cost and degrade retrieval).
        if len(piece) > HARD_MAX:
            if cur_text:
                flush()
            i = 0
            while i < len(piece):
                slice_text = piece[i:i + HARD_MAX]
                out.append((para_num, para_num, slice_text.strip()))
                i += HARD_MAX
            continue
        if cur_text and len(cur_text) + len(piece) > MAX_CHARS_PER_CHUNK:
            flush()
        if not cur_text:
            cur_start_para = para_num
            cur_offset = span_start
        cur_end_para = para_num
        cur_text += piece
    flush()
    return out


def _parse_filename(filename: str) -> tuple[str, str, str]:
    """e.g. QSC_2017_85.txt -> ("QSC", "2017", "85")"""
    stem = filename.replace(".txt", "")
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return "", "", ""


def chunk_all(limit: int | None = MAX_JUDGMENTS) -> Iterator[Chunk]:
    files = sorted(JUDGMENTS_DIR.glob("*.txt"))
    if limit:
        files = files[:limit]

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        citation, court, year = _parse_header(text)
        if not citation:
            citation = f.stem.replace("_", " ")
        court_code, year_code, num_code = _parse_filename(f.name)

        # Extract case name (before the [YYYY] citation)
        m = re.match(r"^(.*?)\s*\[(\d{4})\]\s*([A-Z]+)\s*(\d+)\s*$", citation)
        if m:
            case_name = m.group(1).strip()
        else:
            case_name = citation.split("[")[0].strip()

        # Find body start (skip header + court boilerplate)
        # Body starts at the first paragraph marker [1] (or fallback)
        body_match = re.search(r"^\s*\[1\]\s", text, re.MULTILINE)
        body_start = body_match.start() if body_match else min(2000, len(text) // 3)
        body = text[body_start:]

        chunks = _chunk_judgment_text(body)
        for ci, (ps, pe, chunk_text) in enumerate(chunks):
            if not chunk_text.strip():
                continue
            # Always include the chunk index to guarantee uniqueness even when
            # paragraph spans repeat (e.g. judgments without [N] markers).
            if ps and pe and ps != pe:
                para_label = f"at [{ps}]–[{pe}]"
                source_id = f"{court_code}_{year_code}_{num_code}_p{ps}-{pe}_c{ci}"
            elif ps:
                para_label = f"at [{ps}]"
                source_id = f"{court_code}_{year_code}_{num_code}_p{ps}_c{ci}"
            else:
                para_label = ""
                source_id = f"{court_code}_{year_code}_{num_code}_part{ci}"

            header = f"{citation} {para_label}".strip()

            yield Chunk(
                source_id=source_id,
                source_type="judgment",
                header=header,
                text=chunk_text,
                metadata={
                    "case_name": case_name,
                    "citation": citation,
                    "court": court_code,
                    "year": year_code,
                    "case_number": num_code,
                    "paragraph_start": ps if ps else None,
                    "paragraph_end": pe if pe else None,
                    "source_file": f.name,
                },
            )


if __name__ == "__main__":
    n = 0
    no_paras = 0
    sizes = []
    for c in chunk_all():
        n += 1
        sizes.append(len(c.text))
        if c.metadata.get("paragraph_start") is None:
            no_paras += 1
    print(f"total judgment chunks: {n}")
    print(f"chunks without paragraph markers: {no_paras}")
    if sizes:
        print(f"size: avg={sum(sizes)//len(sizes)}  max={max(sizes)}  min={min(sizes)}")
