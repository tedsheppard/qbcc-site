"""Annotated BIF Act chunker.

Source files in `bif_guide_build/v3/source/annotated/` are one-per-section,
already extracted from the v29 (March 2026) annotated DOCX. Each file looks
like:

    # Annotated BIF Act source — Section 68
    # Chapter: CHAPTER 3 – Progress payments
    # Section title: Meaning of payment claim
    # DOCX paragraphs: 1347-1565

    [2 Com-BIFSOPA Heading 1] SECTION 68 – Meaning of payment claim
    [2.1 Com-BIFSOPA Heading 2] A    Legislation
    [1 BIFSOPA Heading] 68    Meaning of payment claim
    [1.3 BIFSOPA level 1 (CDI)] A payment claim, for a progress payment, ...

We chunk one chunk per file (per section). The style-tag prefixes are stripped
from the chunk text but a tag-frequency summary is kept in metadata for
debugging. If a section file is huge (>3000 chars), split into ~2400-char
windows aligned to paragraph boundaries.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .base import Chunk

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANNOTATED_DIR = ROOT / "bif_guide_build" / "v3" / "source" / "annotated"

HEADER_LINE_RE = re.compile(r"^#\s*(.*)$", re.MULTILINE)
STYLE_TAG_RE = re.compile(r"^\[([^\]]+)\]\s*", re.MULTILINE)
SECTION_FILENAME_RE = re.compile(r"section_(\d+[A-Z]?)\.txt$")

MAX_CHARS = 2400


def _strip_style_tags(text: str) -> str:
    """Remove [Style Name] prefixes from each line."""
    return STYLE_TAG_RE.sub("", text)


def _chunks_for_text(body: str) -> list[str]:
    if len(body) <= MAX_CHARS:
        return [body]
    HARD_MAX = MAX_CHARS * 3  # 7,200 chars
    # split at paragraph boundaries (double newline) first; fall back to
    # single-newline; finally hard-split at char boundaries for any piece
    # still too big.
    def split_or_pack(pieces: list[str]) -> list[str]:
        out: list[str] = []
        cur = ""
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            if len(p) > HARD_MAX:
                if cur:
                    out.append(cur)
                    cur = ""
                i = 0
                while i < len(p):
                    out.append(p[i:i + HARD_MAX])
                    i += HARD_MAX
                continue
            if cur and len(cur) + len(p) + 2 > MAX_CHARS:
                out.append(cur)
                cur = p
            else:
                cur = (cur + "\n\n" + p) if cur else p
        if cur:
            out.append(cur)
        return out

    out = split_or_pack(body.split("\n\n"))
    # Pieces still over HARD_MAX (rare) get re-split at single-newline
    refined: list[str] = []
    for piece in out:
        if len(piece) > HARD_MAX:
            refined.extend(split_or_pack(piece.split("\n")))
        else:
            refined.append(piece)
    return refined


def chunk_all() -> Iterator[Chunk]:
    if not ANNOTATED_DIR.exists():
        return
    for path in sorted(ANNOTATED_DIR.glob("section_*.txt")):
        m = SECTION_FILENAME_RE.search(path.name)
        if not m:
            continue
        section_num = m.group(1).lstrip("0") or "0"
        raw = path.read_text(encoding="utf-8", errors="replace")

        # Strip header `#` lines
        body_lines = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("#")]
        body_raw = "\n".join(body_lines).strip()

        # Strip style-tag prefixes, keep prose
        body = _strip_style_tags(body_raw).strip()

        if not body:
            continue

        # Title from raw header lines
        section_title = ""
        for ln in raw.splitlines():
            if ln.startswith("# Section title:"):
                section_title = ln.split(":", 1)[1].strip()
                break

        chunks = _chunks_for_text(body)
        for i, chunk_text in enumerate(chunks):
            sid = f"annot_s{section_num}"
            if len(chunks) > 1:
                sid += f"_p{i+1}"
            header = f"Annotated BIF Act — commentary on s {section_num}"
            if section_title:
                header += f" ({section_title})"
            yield Chunk(
                source_id=sid,
                source_type="annotated",
                header=header,
                text=chunk_text,
                metadata={
                    "act_short": "BIF Act (annotated)",
                    "section_number": section_num,
                    "section_title": section_title,
                    "source_file": path.name,
                    "source_type_detail": "v29 annotated BIF Act, March 2026",
                },
            )


if __name__ == "__main__":
    n = 0
    sizes = []
    for c in chunk_all():
        n += 1
        sizes.append(len(c.text))
    print(f"total annotated chunks: {n}")
    if sizes:
        print(f"size: avg={sum(sizes)//len(sizes)}  max={max(sizes)}  min={min(sizes)}")
