"""Structure-aware chunker for construction contracts.

The chunker targets ~1200 tokens (~4800 characters at the 4 chars/token
approximation) per chunk with ~200 tokens (~800 chars) of overlap. It
prefers to split on clause / section boundaries first, then on
double-newline paragraph breaks, then on sentence breaks, then on plain
newlines, and finally on a hard char boundary.

We deliberately avoid pulling in ``tiktoken`` here — chars/4 is good
enough for sizing decisions at this granularity and keeps the dep set
light, in line with the rest of the contract_assist subsystem.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Sizing constants (chars, not tokens; 4 chars ≈ 1 token).
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4
TARGET_TOKENS = 1200
OVERLAP_TOKENS = 200

TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN          # 4800
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN        # 800

# Allow up to ~25% slack before forcing a hard split, so the structure-aware
# splitter has room to land on a clean boundary.
MAX_CHARS = int(TARGET_CHARS * 1.25)                    # 6000

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# A "clause boundary" is a line whose first non-space token is a clause/
# section number followed by a capital-letter heading. Matches things like:
#   "34 Variations"
#   "34.1 Notice of variation"
#   "Clause 7 — Time"
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?m)^\s*(?:Clause\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*)\s+[A-Z]"
)

# Any clause reference, used both for section_heading detection (first hit
# in chunk) and for clause_numbers metadata extraction (all hits).
_CLAUSE_NUMBER_RE = re.compile(
    r"(?:^|\b)(?:Clause\s+)?(\d+(?:\.\d+){0,3})\b"
)

# "Heading-shaped" line — a numbered or "Clause N" line, followed by some
# title text. Used to populate section_heading.
_HEADING_LINE_RE = re.compile(
    r"(?m)^\s*(?:Clause\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*)\s+[A-Za-z][^\n]{0,160}"
)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    *,
    document_name: str = "",
    page_offsets: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Split ``text`` into structure-aware overlapping chunks.

    Args:
      text: the full document text to chunk.
      document_name: stored on each chunk for downstream display.
      page_offsets: optional list of character offsets at which each PDF
        page begins (page_offsets[i] = start index of page i+1). When
        provided, each chunk is tagged with ``page_number`` (1-based).

    Returns a list of dicts (one per chunk) each containing:
      - ``chunk_index`` (int, 0-based)
      - ``document_name`` (str)
      - ``content`` (str)
      - ``section_heading`` (str | None)
      - ``clause_numbers`` (list[str])
      - ``page_number`` (int | None)
    """
    if not text:
        return []

    raw = text.strip()
    if not raw:
        return []

    spans = _split_into_spans(raw)
    chunks: list[dict[str, Any]] = []

    for idx, (start, end) in enumerate(spans):
        body = raw[start:end].strip()
        if not body:
            continue

        chunks.append({
            "chunk_index": idx,
            "document_name": document_name,
            "content": body,
            "section_heading": _detect_section_heading(body),
            "clause_numbers": _extract_clause_numbers(body),
            "page_number": _page_for_offset(start, page_offsets),
        })

    # Re-index in case any chunks were dropped due to whitespace-only bodies.
    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    return chunks


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _split_into_spans(text: str) -> list[tuple[int, int]]:
    """Return [(start, end)] character spans, with overlap between adjacent spans."""
    n = len(text)
    if n <= TARGET_CHARS:
        return [(0, n)]

    spans: list[tuple[int, int]] = []
    cursor = 0

    while cursor < n:
        # Tentative end of this chunk.
        ideal_end = cursor + TARGET_CHARS
        hard_end = min(cursor + MAX_CHARS, n)

        if ideal_end >= n:
            spans.append((cursor, n))
            break

        # Search window: TARGET_CHARS .. MAX_CHARS for a good boundary.
        window_start = cursor + (TARGET_CHARS // 2)
        window_end = hard_end
        boundary = _find_boundary(text, window_start, window_end)
        if boundary is None or boundary <= cursor:
            boundary = hard_end

        spans.append((cursor, boundary))

        if boundary >= n:
            break

        # Step forward, leaving an overlap. Never go backwards past the
        # previous start.
        next_cursor = max(cursor + 1, boundary - OVERLAP_CHARS)
        # If overlap would re-emit the same boundary, just advance.
        if next_cursor <= cursor:
            next_cursor = boundary
        cursor = next_cursor

    return spans


def _find_boundary(text: str, lo: int, hi: int) -> int | None:
    """Find the best split point within [lo, hi] using the structure priority.

    Returns the character index just *after* the boundary character, so the
    caller can use it as a half-open span end. None means "no good
    boundary found"; the caller will fall back to ``hi``.
    """
    if hi <= lo:
        return None
    window = text[lo:hi]

    # 1) Clause / section boundary (highest priority).
    last = None
    for m in _CLAUSE_BOUNDARY_RE.finditer(window):
        last = m
    if last is not None:
        return lo + last.start()

    # 2) Double-newline paragraph break — prefer the latest one in the window.
    pos = window.rfind("\n\n")
    if pos != -1:
        return lo + pos + 2

    # 3) Sentence break — '. ' followed by a capital letter is the safest.
    sentence_re = re.compile(r"\.\s+(?=[A-Z])")
    last_sent = None
    for m in sentence_re.finditer(window):
        last_sent = m
    if last_sent is not None:
        return lo + last_sent.end()

    # 4) Plain newline.
    pos = window.rfind("\n")
    if pos != -1:
        return lo + pos + 1

    # 5) Whitespace.
    pos = window.rfind(" ")
    if pos != -1:
        return lo + pos + 1

    return None


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _detect_section_heading(chunk_body: str) -> str | None:
    """Return the first heading-shaped line in the chunk, if any."""
    m = _HEADING_LINE_RE.search(chunk_body)
    if not m:
        return None
    line = m.group(0).strip()
    # Cap heading length so metadata stays tidy.
    if len(line) > 160:
        line = line[:160].rstrip() + "…"
    return line


def _extract_clause_numbers(chunk_body: str) -> list[str]:
    """Return a deduplicated list of clause numbers found in the chunk.

    Uses an ordered set so the first-seen ordering is preserved; this is
    handy for the section_heading vs body alignment but not load-bearing.
    """
    seen: dict[str, None] = {}
    for m in _CLAUSE_NUMBER_RE.finditer(chunk_body):
        token = m.group(1)
        # Bare integers are fairly noisy on their own (years, page nos, lists).
        # Keep only references that either contain a dot OR appear right after
        # the literal "Clause "/"clause ".
        is_dotted = "." in token
        prefix_start = max(0, m.start() - 8)
        prefix = chunk_body[prefix_start:m.start()].lower()
        is_after_clause_word = "clause" in prefix
        if not (is_dotted or is_after_clause_word):
            continue
        seen.setdefault(token, None)
    return list(seen.keys())


def _page_for_offset(offset: int, page_offsets: list[int] | None) -> int | None:
    """Return 1-based page number containing ``offset``, or None."""
    if not page_offsets:
        return None
    # page_offsets[i] is the start of page i+1.
    page = 1
    for i, start in enumerate(page_offsets):
        if offset >= start:
            page = i + 1
        else:
            break
    return page
