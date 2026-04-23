"""PDF analysis report generator for /claim-check.

Uses reportlab (already in requirements.txt) to build a streamed PDF
containing the Sopal logo, document summary, all check results, the
user's inline answers, and relevant case citations. The PDF is generated
on demand and never written to disk.

Stage 11 will implement this.
"""

from __future__ import annotations


def build_report_pdf(
    mode: str,
    filename: str | None,
    document_summary: str,
    check_results: list[dict],
    user_answers: dict,
    citations: list[dict],
) -> bytes:
    raise NotImplementedError("report_generator.build_report_pdf — implemented in stage 11")
