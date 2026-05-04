"""Eval question schema for bif_research."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CATEGORIES = (
    "statutory_test",
    "definition",
    "procedural_deadline",
    "case_law",
    "section_precision",
    "cross_reference",
    "amendment_currency",
)


@dataclass
class ExpectedSource:
    """A source the system is expected to retrieve and cite."""
    type: Literal["statute", "case", "decision", "annotated", "regulation"]
    # Statute/regulation: act + section
    act: str = ""           # e.g. "BIF Act", "QBCC Act", "BIF Reg"
    section: str = ""       # e.g. "68", "75(2)"
    # Case
    name_pattern: str = ""  # case-insensitive substring match against case_name
    paragraph_range: tuple[int, int] | None = None  # inclusive
    # Annotated
    annotated_section: str = ""  # the BIF Act section the commentary is about


@dataclass
class EvalQuestion:
    id: str
    question: str
    category: str
    expected_sources: list[ExpectedSource] = field(default_factory=list)
    expected_answer_summary: str = ""
    must_not_contain: list[str] = field(default_factory=list)
    notes: str = ""


def question_to_dict(q: EvalQuestion) -> dict:
    return {
        "id": q.id,
        "question": q.question,
        "category": q.category,
        "expected_sources": [
            {k: v for k, v in src.__dict__.items() if v not in ("", None)}
            for src in q.expected_sources
        ],
        "expected_answer_summary": q.expected_answer_summary,
        "must_not_contain": q.must_not_contain,
        "notes": q.notes,
    }
