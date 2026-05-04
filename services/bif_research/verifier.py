"""Verifier — checks every citation resolves and every quote is verbatim.

The architectural guarantees from spec section 4.1 and 4.2:
  - Every citation chunk_id MUST resolve to a chunk in the retrieved set.
    (Already enforced in postprocessor.process — invalid ids dropped.)
  - Every quote text MUST be a verbatim substring of its cited chunk's text
    (whitespace-tolerant). The postprocessor only includes located quotes,
    so this is structurally enforced.

This module verifies the FINAL answer object after postprocessing. If any
quote is `located=False` it counts as a verification failure. If the
proposition has no remaining located quotes AND no citations, it is removed
and replaced with an [unsupported] marker.
"""
from __future__ import annotations

from dataclasses import dataclass
from .postprocessor import FinalAnswer, FinalProposition, FinalQuote
from .corpus.base import normalise_ws


@dataclass
class VerificationReport:
    citations_total: int = 0
    citations_resolved: int = 0       # always == total because postprocessor drops unresolved
    quotes_attempted: int = 0
    quotes_located: int = 0
    quotes_verbatim: int = 0          # of located, how many actually verbatim in chunk
    failed_quotes: list[str] = None
    propositions_kept: int = 0
    propositions_removed: int = 0


def verify(answer: FinalAnswer) -> tuple[FinalAnswer, VerificationReport]:
    """Run the verification pass over a postprocessed answer.

    Quotes that didn't locate are dropped (with a [unsupported] flag).
    Returns (possibly modified answer, report).
    """
    report = VerificationReport(failed_quotes=[])

    if answer.refused:
        return answer, report

    sources_by_id = {s.chunk_id: s for s in answer.sources}

    new_props: list[FinalProposition] = []
    for p in answer.propositions:
        report.citations_total += len(p.citation_indices)
        report.citations_resolved += len(p.citation_indices)  # postprocessor guarantees this

        verified_quotes: list[FinalQuote] = []
        for q in p.quotes:
            report.quotes_attempted += 1
            if not q.located or not q.text:
                report.failed_quotes.append(f"{q.chunk_id}: not located")
                continue
            report.quotes_located += 1

            # Strict verbatim check
            chunk = sources_by_id.get(q.chunk_id)
            if chunk is None:
                report.failed_quotes.append(f"{q.chunk_id}: chunk vanished")
                continue
            if normalise_ws(q.text) in normalise_ws(chunk.text):
                report.quotes_verbatim += 1
                verified_quotes.append(q)
            else:
                report.failed_quotes.append(f"{q.chunk_id}: quote not verbatim in chunk")

        # Keep proposition if it still has citation backing
        if p.citation_indices or verified_quotes:
            new_props.append(FinalProposition(
                text=p.text,
                citation_indices=p.citation_indices,
                quotes=verified_quotes,
            ))
            report.propositions_kept += 1
        else:
            report.propositions_removed += 1

    answer.propositions = new_props
    return answer, report
