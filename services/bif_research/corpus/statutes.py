"""Statute and regulation chunker.

Source files in `exports/qld_acts_txt/` contain pre-processed text where each
section is preceded by a `--- Section NN: <title> ---` separator and the
section body follows on the next line. Each section body is typically a single
long line with subsections inline as `(1)... (2)...`.

Strategy:
- Use the `--- Section NN: ---` separators as canonical section boundaries.
- For each section, split into chunks of <= MAX_TOKENS (~600 tokens) at
  subsection boundaries when possible; fall back to character splits otherwise.
- Header line per chunk: e.g. "BIF Act s 68 — Meaning of payment claim".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .base import Chunk

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ACTS_DIR = ROOT / "exports" / "qld_acts_txt"

# (filename, act_short, act_name, source_type)
ACT_SOURCES = [
    ("act-2017-043.txt", "BIF Act", "Building Industry Fairness (Security of Payment) Act 2017", "statute"),
    ("sl-2018-0016.txt", "BIF Reg", "Building Industry Fairness (Security of Payment) Regulation 2018", "regulation"),
    ("act-1991-098.txt", "QBCC Act", "Queensland Building and Construction Commission Act 1991", "statute"),
    ("sl-2018-0138.txt", "QBCC Reg", "Queensland Building and Construction Commission Regulation 2018", "regulation"),
    ("act-1954-003.txt", "AIA", "Acts Interpretation Act 1954", "statute"),
]

SECTION_HEADER_RE = re.compile(
    r"^---\s*Section\s+(?P<num>[0-9]+[A-Z]{0,2}(?:\.[0-9]+)?)\s*:\s*(?P<title>[^-]+?)\s*---\s*$",
    re.MULTILINE,
)

# Subsection split points within a long section body, e.g. "(2)Body text..."
SUBSECTION_RE = re.compile(r"(?<=\.)\((\d{1,3}[A-Z]?)\)")

# Approximate token budget per chunk (4 chars/token)
MAX_CHARS = 2400  # ~600 tokens
MIN_SPLIT_CHARS = 800  # don't bother splitting a section under this size


def _split_into_subchunks(body: str) -> list[tuple[str, str]]:
    """Split a long section body into subchunks at subsection boundaries.

    Returns list of (subsection_marker, chunk_text). Returns [("", body)]
    if no split is needed or possible.
    """
    if len(body) <= MAX_CHARS:
        return [("", body)]
    matches = list(SUBSECTION_RE.finditer(body))
    if not matches:
        # fallback to character-window
        chunks = []
        i = 0
        while i < len(body):
            chunks.append(("", body[i:i + MAX_CHARS]))
            i += MAX_CHARS
        return chunks
    # Build subchunks at subsection boundaries, packing up to MAX_CHARS each
    boundaries = [m.start() for m in matches] + [len(body)]
    sub_pieces = []
    for i, m in enumerate(matches):
        sub_marker = m.group(1)
        start = m.start()
        end = boundaries[i + 1]
        piece = body[start:end]
        sub_pieces.append((f"({sub_marker})", piece))
    # The pre-(1) prefix (if any) becomes part of the first chunk
    pre = body[:matches[0].start()]
    if pre.strip():
        sub_pieces[0] = (sub_pieces[0][0], pre + sub_pieces[0][1])

    # Pack into chunks not exceeding MAX_CHARS
    packed: list[tuple[str, str]] = []
    cur_marker = ""
    cur_text = ""
    for marker, text in sub_pieces:
        if cur_text and len(cur_text) + len(text) > MAX_CHARS:
            packed.append((cur_marker, cur_text))
            cur_marker = marker
            cur_text = text
        else:
            if not cur_text:
                cur_marker = marker
            cur_text += text
    if cur_text:
        packed.append((cur_marker, cur_text))
    return packed


def _parse_act(filename: str, act_short: str, act_name: str, source_type: str) -> Iterator[Chunk]:
    path = ACTS_DIR / filename
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")

    # Find all section headers and their offsets
    headers = list(SECTION_HEADER_RE.finditer(text))
    if not headers:
        return

    for i, m in enumerate(headers):
        sec_num = m.group("num").strip()
        sec_title = m.group("title").strip()
        body_start = m.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        raw_body = text[body_start:body_end].strip()
        if not raw_body:
            continue

        subchunks = _split_into_subchunks(raw_body)
        for j, (sub_marker, chunk_text) in enumerate(subchunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            # Strip the leading "NN<title>" prefix that appears at the start of
            # the body line (we already have the title in metadata)
            # e.g. body starts "68Meaning of payment claim(1)A payment claim..."
            # We keep this in chunk_text because it's actually informative for
            # retrieval — section number adjacent to the body helps BM25.

            # Always append a positional suffix when there is more than one
            # subchunk so source_ids stay unique even if subsection markers
            # repeat or the same marker spans multiple packed chunks.
            sec_id = f"{filename.replace('.txt', '').replace('-', '_')}_s{sec_num}"
            if len(subchunks) > 1:
                if sub_marker:
                    sec_id += f"_{sub_marker.strip('()').lower()}_p{j+1}"
                else:
                    sec_id += f"_p{j+1}"
            elif sub_marker:
                sec_id += f"_{sub_marker.strip('()').lower()}"

            header_parts = [f"{act_short} s {sec_num}"]
            if sub_marker:
                header_parts[0] += sub_marker
            header_parts.append(f"— {sec_title}")
            header = " ".join(header_parts)

            yield Chunk(
                source_id=sec_id,
                source_type=source_type,  # type: ignore[arg-type]
                header=header,
                text=chunk_text,
                metadata={
                    "act_short": act_short,
                    "act_name": act_name,
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "subsection_path": sub_marker,
                    "source_file": filename,
                },
            )


def chunk_all() -> Iterator[Chunk]:
    for filename, act_short, act_name, source_type in ACT_SOURCES:
        yield from _parse_act(filename, act_short, act_name, source_type)


if __name__ == "__main__":
    n = 0
    by_act: dict[str, int] = {}
    for c in chunk_all():
        n += 1
        by_act[c.metadata["act_short"]] = by_act.get(c.metadata["act_short"], 0) + 1
    print(f"total statute chunks: {n}")
    for k, v in sorted(by_act.items()):
        print(f"  {k}: {v}")
