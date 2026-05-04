"""Common chunk dataclass and helpers for all four chunkers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal


SourceType = Literal["statute", "regulation", "judgment", "decision", "annotated"]


@dataclass
class Chunk:
    source_id: str          # stable per-source id (e.g. "bifact_s68", "QSC_2017_85_p42-45")
    source_type: SourceType
    header: str             # rendered header line (e.g. "BIF Act s 68 — Meaning of payment claim")
    text: str               # body text only (no header)
    metadata: dict = field(default_factory=dict)

    def indexed_text(self) -> str:
        """Text passed to embedder + BM25 (header + body so retrieval matches both)."""
        return f"{self.header}\n\n{self.text}".strip()

    def render_citation(self) -> str:
        """Human-readable citation rendered from metadata."""
        m = self.metadata
        if self.source_type in ("statute", "regulation"):
            act = m.get("act_short") or m.get("act_name") or ""
            sec = m.get("section_number") or ""
            sub = m.get("subsection_path") or ""
            base = f"{act} s {sec}".strip()
            if sub:
                base += sub  # e.g. "BIF Act s 75(2)"
            return base
        if self.source_type == "judgment":
            cite = m.get("citation") or m.get("case_name") or ""
            ps = m.get("paragraph_start")
            pe = m.get("paragraph_end")
            if ps is not None and pe is not None and ps != pe:
                return f"{cite} at [{ps}]–[{pe}]"
            if ps is not None:
                return f"{cite} at [{ps}]"
            return cite
        if self.source_type == "decision":
            ref = m.get("decision_id") or ""
            parties = m.get("parties") or ""
            return f"Adjudication Decision {ref} — {parties}".strip(" —")
        if self.source_type == "annotated":
            sec = m.get("section_number") or ""
            return f"Annotated BIF Act — commentary on s {sec}".strip()
        return self.header


def normalise_ws(text: str) -> str:
    """Collapse runs of whitespace to single spaces. Used for verifier comparisons."""
    return " ".join(text.split())
