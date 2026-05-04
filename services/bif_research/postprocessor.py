"""Mechanical quote extraction and citation rendering.

Takes the answerer's structured JSON and the retrieved chunks, then:
  1. Resolves each citation chunk_id to a real chunk (drops invalid).
  2. For each quote, locates the span in the chunk and extracts the verbatim
     text. If the span cannot be located (whitespace-tolerant), the quote
     is dropped and a flag is recorded for the verifier.
  3. Renders the final answer as HTML with inline [n] citation markers and
     a sources list keyed by [n].
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

from .corpus.base import normalise_ws
from .retriever import Hit

# Hard rule: only legislation and case law may appear as a cited end-source.
# Annotated commentary and adjudication decisions can be retrieved (the
# answerer sees them as context) but they MUST be stripped from any cite.
CITABLE_SOURCE_TYPES = {"statute", "regulation", "judgment"}


@dataclass
class FinalQuote:
    chunk_id: str
    text: str            # the verbatim extracted quote
    located: bool        # True if span_start/end resolved


@dataclass
class FinalProposition:
    text: str
    citation_indices: list[int]   # 1-based indices into the sources list
    quotes: list[FinalQuote]


@dataclass
class FinalAnswer:
    propositions: list[FinalProposition]
    sources: list[Hit]            # ordered; index n-1 corresponds to citation [n]
    answer_summary: str
    confidence: str
    answer_html: str
    refused: bool
    flags: list[str]              # any post-process warnings


def _locate_span(chunk_text: str, start: str, end: str) -> str | None:
    """Find a span in chunk_text from start to end. Whitespace-tolerant.

    Returns the verbatim text from chunk_text (preserving original
    whitespace) or None if not found.
    """
    if not start or not end:
        return None

    # Build a tolerant pattern for the start and end strings — collapse runs
    # of whitespace in the needle to \s+ in the regex.
    def to_tolerant(needle: str) -> re.Pattern:
        parts = [re.escape(p) for p in needle.split()]
        if not parts:
            return None
        return re.compile(r"\s+".join(parts), re.IGNORECASE)

    start_pat = to_tolerant(start)
    end_pat = to_tolerant(end)
    if start_pat is None or end_pat is None:
        return None

    sm = start_pat.search(chunk_text)
    if not sm:
        return None
    # Search end AFTER the start match
    em = end_pat.search(chunk_text, sm.end())
    if not em:
        # Try same position — quote might be a single sentence where start
        # and end overlap
        em = end_pat.search(chunk_text, sm.start())
        if not em or em.end() <= sm.start():
            return None
    extracted = chunk_text[sm.start():em.end()]
    # Sanity: cap quote length to ~1000 chars
    if len(extracted) > 1000:
        return None
    return extracted


def _chunk_lookup(sources: list[Hit]) -> dict[str, Hit]:
    return {h.chunk_id: h for h in sources}


def process(answer_json: dict, retrieved: list[Hit], *, refused: bool = False) -> FinalAnswer:
    """Apply mechanical quote extraction + render the final answer."""
    flags: list[str] = []

    if refused:
        return FinalAnswer(
            propositions=[],
            sources=[],
            answer_summary=answer_json.get("answer_summary",
                "I could not find sufficient sources to answer this question with confidence."),
            confidence="low",
            answer_html=_render_refusal(),
            refused=True,
            flags=flags,
        )

    raw_props = answer_json.get("propositions", []) or []
    summary = answer_json.get("answer_summary", "") or ""
    confidence = answer_json.get("confidence", "medium")

    by_id = _chunk_lookup(retrieved)
    # Build the ordered sources list — only include chunks that get cited
    # AND are of a citable source_type (legislation or case law). Any
    # citation pointing at annotated commentary or an adjudication
    # decision is silently stripped here as a defensive measure on top
    # of the answerer prompt's CONTEXT-ONLY tagging.
    cited_ids: list[str] = []
    seen: set[str] = set()
    for p in raw_props:
        for cid in p.get("citations", []) or []:
            if cid in seen:
                continue
            if cid not in by_id:
                continue
            if by_id[cid].source_type not in CITABLE_SOURCE_TYPES:
                flags.append(
                    f"stripped non-citable citation {cid} "
                    f"(source_type={by_id[cid].source_type})"
                )
                continue
            cited_ids.append(cid)
            seen.add(cid)
    sources = [by_id[cid] for cid in cited_ids]
    cid_to_index = {cid: i + 1 for i, cid in enumerate(cited_ids)}

    final_props: list[FinalProposition] = []
    for p in raw_props:
        prop_text = (p.get("text", "") or "").strip()
        if not prop_text:
            continue
        cits = []
        for cid in p.get("citations", []) or []:
            if cid in cid_to_index:
                cits.append(cid_to_index[cid])
            else:
                flags.append(f"unresolved citation {cid} in proposition '{prop_text[:50]}…'")

        # Mechanical quote extraction
        final_quotes: list[FinalQuote] = []
        for q in p.get("quotes", []) or []:
            qcid = q.get("chunk_id", "")
            if qcid not in by_id:
                flags.append(f"quote chunk_id {qcid} not in retrieved set; dropping")
                final_quotes.append(FinalQuote(qcid, "", located=False))
                continue
            if by_id[qcid].source_type not in CITABLE_SOURCE_TYPES:
                # Silently drop quotes from CONTEXT-ONLY chunks
                flags.append(
                    f"dropped quote from non-citable {qcid} "
                    f"(source_type={by_id[qcid].source_type})"
                )
                continue
            chunk_text = by_id[qcid].text
            extracted = _locate_span(chunk_text, q.get("span_start", ""), q.get("span_end", ""))
            if extracted is None:
                flags.append(f"quote span not found in {qcid}; dropping")
                final_quotes.append(FinalQuote(qcid, "", located=False))
                continue
            final_quotes.append(FinalQuote(qcid, extracted, located=True))

        final_props.append(FinalProposition(
            text=prop_text,
            citation_indices=cits,
            quotes=final_quotes,
        ))

    if not final_props:
        # Model returned propositions that all failed resolution — treat as refusal
        return FinalAnswer(
            propositions=[],
            sources=[],
            answer_summary=summary or "I could not produce a supported answer.",
            confidence="low",
            answer_html=_render_refusal(),
            refused=True,
            flags=flags,
        )

    answer_html = _render_html(summary, final_props, sources)
    return FinalAnswer(
        propositions=final_props,
        sources=sources,
        answer_summary=summary,
        confidence=confidence,
        answer_html=answer_html,
        refused=False,
        flags=flags,
    )


def _render_refusal() -> str:
    return (
        '<div class="answer-refusal">'
        '<p>I could not find sufficient sources in the available material to answer '
        'this question with confidence. Try rephrasing or narrowing the question.</p>'
        '</div>'
    )


def _render_html(summary: str, props: list[FinalProposition], sources: list[Hit]) -> str:
    parts: list[str] = []
    if summary:
        parts.append(f'<div class="answer-summary"><p>{html.escape(summary)}</p></div>')
    parts.append('<div class="answer-body">')
    for p in props:
        cite_markers = "".join(
            f'<sup><a href="#src-{i}" class="cite-marker" data-cite="{i}">[{i}]</a></sup>'
            for i in p.citation_indices
        )
        prop_html = f'<p>{html.escape(p.text)} {cite_markers}</p>'
        parts.append(prop_html)
        for q in p.quotes:
            if q.located and q.text:
                cite_idx = next(
                    (i for i, s in enumerate(sources, 1) if s.chunk_id == q.chunk_id),
                    None,
                )
                cite_link = (
                    f'<sup><a href="#src-{cite_idx}" class="cite-marker">[{cite_idx}]</a></sup>'
                    if cite_idx else ""
                )
                parts.append(
                    f'<blockquote class="answer-quote">'
                    f'<p>{html.escape(q.text.strip())} {cite_link}</p>'
                    f'</blockquote>'
                )
    parts.append('</div>')
    return "\n".join(parts)
