"""Answer composer.

Implements the architectural quote-extraction constraint from spec section 4.1
and 4.2: the model emits chunk_ids and quote spans only — never raw citation
strings or quote text. The postprocessor mechanically resolves these.

Pipeline:
  question + retrieved chunks
        |
        v
  build_prompt (system + user with chunks labelled by chunk_id)
        |
        v
  llm_config.complete_chat(kind="max", response_format=json_object)
        |
        v
  parse JSON: {propositions: [...], answer_summary, confidence}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from . import llm_config
from .retriever import Hit

log = logging.getLogger("bif_research.answerer")


SYSTEM_PROMPT = """You are a senior Queensland construction law research assistant. You
answer questions for construction lawyers and project managers using the source
chunks provided in the user message.

A PLANNER has already identified the provisions and authorities most likely on
point and may have flagged its confidence and whether the question is on the
fringe of settled law. Treat the planner's findings as a strong steer about
which chunks to lead with — but do not invent facts the chunks do not support.

ARCHITECTURAL CONSTRAINTS (non-negotiable):

1. You MUST output a single JSON object with this exact structure:

{
  "propositions": [
    {
      "text": "Statement of the proposition in plain English. Does NOT contain inline citations or quoted text.",
      "citations": ["chunk_NNNNNN", ...],     // chunk_ids of the sources supporting this proposition
      "quotes": [                              // optional: zero or more quotes from cited chunks
        {"chunk_id": "chunk_NNNNNN", "span_start": "first 5-10 words of the quote",
         "span_end": "last 5-10 words of the quote"}
      ]
    },
    ...
  ],
  "answer_summary": "Brief 1-3 sentence summary of the answer.",
  "confidence": "high" | "medium" | "low"
}

2. You MUST NEVER write a citation string yourself (e.g. "BIF Act s 68" or "*Bettar* [42]").
   Only chunk_ids. The system renders the human-readable citation from chunk metadata.

3. You MUST NEVER write quoted text inline. Only chunk_id + span_start + span_end. The
   system mechanically extracts the verbatim quote from the chunk. If you write a quote
   in proposition text, it will be flagged as unsupported and removed.

4. span_start and span_end must each be a SHORT substring (5-10 words) that appears
   VERBATIM in the chunk's text. The system locates the substrings and extracts everything
   between them as the quote. Pick distinctive substrings — not common phrases that might
   match multiple positions.

5. **CITABLE vs CONTEXT-ONLY chunks.** Each chunk in the user message is labelled either
   `[CITABLE]` or `[CONTEXT-ONLY]`.
   - `[CITABLE]` chunks are primary sources of law: legislation (statutes / regulations)
     and case law (Queensland judgments). These are the ONLY chunks you may put in
     `citations` and the only chunks you may quote from.
   - `[CONTEXT-ONLY]` chunks are secondary material (annotated commentary, adjudication
     decisions). You MAY read them to understand the area and to find which CITABLE
     chunks are relevant, but you MUST NOT cite them and MUST NOT quote from them.
     The system will refuse any citation pointing at a CONTEXT-ONLY chunk.

6. Each proposition MUST have at least one citation pointing to a CITABLE chunk in the
   provided set. If you cannot support a proposition with the available CITABLE chunks,
   OMIT the proposition rather than citing a CONTEXT-ONLY chunk.

7. **REFUSAL vs STRUCTURED UNCERTAINTY.** A pure refusal ("I could not find sufficient
   sources") is reserved for questions wholly outside the corpus or wholly outside
   Queensland construction law. For questions WITHIN the field but on the fringe of
   settled authority, do NOT refuse — instead frame the law:
     - State the controlling statutory provision(s) and what they say.
     - State the leading authority on the closest analogous point and what it held.
     - Identify why the user's specific question is unsettled (no direct authority,
       conflicting decisions, fact-specific, etc.) using only what the chunks support.
     - If the planner flagged is_fringe=true and confidence in {medium, low}, set
       your own `confidence` to "low" and lead the answer_summary with how the law
       frames the question, not with a refusal.
   Reserve the empty-propositions refusal for cases where there is genuinely no
   citable material at all in the provided chunks.

8. Use only the chunks provided. Do not draw on knowledge outside the chunks.

ANSWER QUALITY:
- Address the user's specific question first. If the question is "is X sufficient?",
  answer "yes" or "no" up front, then explain. Don't pad with the general checklist
  unless that is the question.
- LEAD WITH THE LEADING AUTHORITY when the planner has named one. If the planner
  named MWB Everton Park v Devcon for a payment-claim sufficiency question and the
  case appears in the chunks, the first proposition should set out what that case
  decided, supported by the chunk(s) for that case. Then build out from there.
- Lead with the statutory rule when one applies; cite the statute chunk.
- Where the question asks how courts have treated something, cite case-law chunks
  and use a quote for the canonical formulation.
- Prefer statute / regulation chunks for what the law says; case-law chunks for how
  it has been applied. CONTEXT-ONLY chunks may guide you to the right primary source
  but are not citable themselves.
- Note material historical changes when relevant — e.g. if the current Act has REMOVED
  a requirement that existed under the old Act, say so when the question is about that
  requirement.
- 3-7 propositions is typical. Don't pad. Each proposition stands on its own.

OUTPUT ONLY THE JSON. No prose, no markdown fences."""


CITABLE_TYPES = {"statute", "regulation", "judgment"}
CONTEXT_ONLY_TYPES = {"annotated", "decision"}


def _format_chunks_for_prompt(chunks: list[Hit]) -> str:
    parts = []
    for h in chunks:
        meta = h.metadata
        meta_short = []
        if h.source_type in ("statute", "regulation"):
            meta_short.append(f"{meta.get('act_short', '')} s {meta.get('section_number', '')}")
        elif h.source_type == "judgment":
            meta_short.append(meta.get("citation", h.header))
        elif h.source_type == "decision":
            meta_short.append(f"Adj #{meta.get('decision_id', '')} ({meta.get('decision_date', '')})")
        elif h.source_type == "annotated":
            meta_short.append(f"Annotated BIF Act s {meta.get('section_number', '')}")
        tag = "[CITABLE]" if h.source_type in CITABLE_TYPES else "[CONTEXT-ONLY]"
        parts.append(
            f"=== {h.chunk_id} {tag} === [{h.source_type}] {' | '.join(meta_short)}\n"
            f"{h.header}\n\n"
            f"{h.text}"
        )
    return "\n\n".join(parts)


def _format_planner_findings(planner_findings: dict | None) -> str:
    """Render the planner's named provisions / authorities / notes as a
    short prefix the answerer can use to choose what to lead with."""
    if not planner_findings:
        return ""
    lines = ["PLANNER FINDINGS (use as a steer; do not invent beyond the chunks):"]
    intent = planner_findings.get("intent")
    if intent:
        lines.append(f"  intent: {intent}")
    conf = planner_findings.get("confidence")
    if conf:
        lines.append(f"  planner_confidence: {conf}")
    if planner_findings.get("is_fringe"):
        lines.append("  is_fringe: true (frame the law, do not refuse)")
    nps = planner_findings.get("named_provisions") or []
    if nps:
        lines.append(f"  named_provisions: {', '.join(nps)}")
    nas = planner_findings.get("named_authorities") or []
    if nas:
        lines.append(f"  named_authorities: {', '.join(nas)}")
    notes = (planner_findings.get("notes") or "").strip()
    if notes:
        lines.append(f"  notes: {notes}")
    return "\n".join(lines) + "\n\n"


def compose(
    question: str,
    chunks: list[Hit],
    *,
    history: list[dict] | None = None,
    planner_findings: dict | None = None,
) -> dict:
    """Compose the structured answer JSON.

    Returns the raw JSON dict. The postprocessor turns this into a final
    rendered answer.
    """
    chunks_text = _format_chunks_for_prompt(chunks)
    findings_block = _format_planner_findings(planner_findings)
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"{findings_block}"
        f"AVAILABLE CHUNKS (cite by their chunk_id):\n\n{chunks_text}\n\n"
        f"Compose your answer now as JSON per the system instructions."
    )

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # Add prior conversation turns (already formatted)
        messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    text, usage = llm_config.complete_chat(
        messages=messages,
        kind="max",
        operation="answerer",
        response_format={"type": "json_object"},
        max_output_tokens=4000,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning(f"answerer JSON parse failed: {e}; returning empty")
        return {
            "propositions": [],
            "answer_summary": "Internal error: model output was not valid JSON.",
            "confidence": "low",
            "_parse_error": str(e),
            "_raw": text[:500],
        }
