"""Loader for rules/bif_act_annotations.md.

The annotations file is the user's own commentary on the BIF Act (extracted
from the v29 annotated edition). The rule engine consults it when running
semantic checks whose act_reference points at a section that has an
annotation.

Authority ranking per spec Section 13:
  legislation > user's v29 annotations > LLM prior knowledge.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("claim_check.annotations")

ANNOTATIONS_PATH = Path(__file__).resolve().parents[2] / "rules" / "bif_act_annotations.md"

# Per-prompt cap so we don't blow the context window. ~3500 chars ≈ ~900 tokens.
DEFAULT_EXCERPT_CHARS = 3500


def load_annotations(path: Path | str | None = None) -> dict[int, str]:
    """Return {section_number: annotation_text} parsed from the markdown file."""
    p = Path(path) if path else ANNOTATIONS_PATH
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    out: dict[int, str] = {}

    # Sections are level-2 headings "## s NN".
    pattern = re.compile(r"^##\s+s\s+(\d+)\b.*$", re.MULTILINE)
    positions = [(m.start(), int(m.group(1))) for m in pattern.finditer(text)]
    positions.append((len(text), -1))

    for i in range(len(positions) - 1):
        start, num = positions[i]
        end = positions[i + 1][0]
        if num < 0:
            continue
        block = text[start:end]
        # Strip the heading and the trailing "---" separator.
        block = re.sub(r"^##\s+s\s+\d+.*?\n", "", block, count=1)
        block = block.strip()
        if block.endswith("---"):
            block = block[:-3].strip()
        out[num] = block

    return out


def primary_section(act_reference: str | None) -> int | None:
    """Parse 's 68(1)(a) BIF Act' → 68. Returns None if no match."""
    if not act_reference:
        return None
    m = re.search(r"\bs\s*(\d+)\b", act_reference)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def annotation_excerpt_for_act_reference(
    act_reference: str | None,
    max_chars: int = DEFAULT_EXCERPT_CHARS,
    *,
    keyword_hint: str | None = None,
) -> str | None:
    """Return a prompt-ready excerpt of the annotation for the section.

    If ``keyword_hint`` is given, prefers the passage of the annotation
    containing the hint. Otherwise returns the opening block.
    """
    num = primary_section(act_reference)
    if num is None:
        return None
    annotations = load_annotations()
    body = annotations.get(num)
    if not body:
        return None

    if keyword_hint:
        # Find the keyword and return a window around it.
        pos = body.lower().find(keyword_hint.lower())
        if pos != -1:
            half = max_chars // 2
            start = max(0, pos - half)
            end = min(len(body), start + max_chars)
            snippet = body[start:end].strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(body):
                snippet = snippet + "…"
            return snippet

    if len(body) <= max_chars:
        return body
    return body[:max_chars].rsplit("\n\n", 1)[0] + "\n…"
