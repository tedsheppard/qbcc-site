"""Read-full-cases pipeline for nuanced questions.

The fast `FullPipeline` (in pipeline.py) compresses everything via
BM25/dense retrieval before the reasoner sees it; that's right for simple
questions but loses doctrinal nuance on hard ones (sub-issues, fringe
authorities, jurisdictional-fact debates, etc.).

This pipeline trades latency and a few cents for thoroughness:

  1. Conversation-aware query rewrite (reused).
  2. Knowledge-augmented planner (reused) — names provisions and cases.
  3. fetch_section_full() for each named provision (no LLM).
  4. resolve named_authorities -> case_ids via name_index.
  5. parallel_read_cases() runs cheap reader workers per case in parallel;
     each worker gets the FULL case text and emits verbatim on-point
     paragraphs (no compression of text fidelity).
  6. retrieve_three_channel(k=6) for breadth coverage of anything the
     planner didn't name.
  7. reasoner_compose() — Opus 4.7 — given full statute text + verbatim
     case extracts + 6 breadth chunks, emits the same JSON schema as the
     fast-path answerer.
  8. postprocessor + verifier (reused).

Quote extraction stays mechanical: the reasoner emits chunk_ids and span
markers, postprocessor pulls verbatim spans from the chunks.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from . import (
    answerer,
    corpus_fetch,
    llm_config,
    name_index,
    parallel_reader,
    planner,
    postprocessor,
    verifier,
)
from .pipeline import (
    _format_history_for_answerer,
    _planner_summary,
    _rewrite_with_history,
    _to_eval_format,
)
from .retriever import Hit, Retriever

log = logging.getLogger("bif_research.hard_pipeline")


# Map planner-emitted "BIF Act s 75" to (act_short, section) for fetch.
_ACT_NAME_PATTERNS = [
    (re.compile(r"\bbif\s*act\b|\bbifa\b|\bbuilding\s+industry\s+fairness", re.I), "BIF Act"),
    (re.compile(r"\bbif\s*reg\b|\bbifr\b|\bbuilding\s+industry\s+fairness.*regulation", re.I), "BIF Reg"),
    (re.compile(r"\bqbcc\s*act\b|\bqueensland\s+building\s+and\s+construction\s+commission\s+act", re.I), "QBCC Act"),
    (re.compile(r"\bqbcc\s*reg\b|\bqueensland\s+building\s+and\s+construction\s+commission\s+regulation", re.I), "QBCC Reg"),
    (re.compile(r"\bacts?\s+interpretation\s+act\b|\baia\b", re.I), "AIA"),
]
_SECTION_PATTERN = re.compile(
    r"(?:section|sec|s|sch)\s*\.?\s*([0-9]+[A-Za-z]*)",
    re.I,
)


def _parse_provision_for_fetch(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    act_short = None
    for pat, label in _ACT_NAME_PATTERNS:
        if pat.search(text):
            act_short = label
            break
    m = _SECTION_PATTERN.search(text)
    if not m:
        return act_short, None
    return act_short, m.group(1).strip()


REASONER_SYSTEM_PROMPT = """You are a senior Queensland construction law research
assistant composing a final answer from full case excerpts and full statutory
provisions.

You are reasoning over FULL CASE EXCERPTS rather than chunked snippets. The
extracts in the user message are verbatim paragraphs that a reader-model
identified as on-point for the question. The reader did NOT compose
analysis — it only quoted. Your job is to do the doctrinal synthesis the
reader did not do.

ARCHITECTURAL CONSTRAINTS (non-negotiable):

1. Output a single JSON object with this exact structure:

{
  "propositions": [
    {
      "text": "Statement of the proposition in plain English. Does NOT contain inline citations or quoted text.",
      "citations": ["chunk_NNNNNN", ...],
      "quotes": [
        {"chunk_id": "chunk_NNNNNN",
         "span_start": "first 5-10 words of the quote",
         "span_end": "last 5-10 words of the quote"}
      ]
    },
    ...
  ],
  "answer_summary": "Brief 1-3 sentence summary of the answer.",
  "confidence": "high" | "medium" | "low"
}

2. NEVER write a citation string yourself. Only chunk_ids. The system
   renders the human-readable citation from chunk metadata.

3. NEVER write quoted text inline. Only chunk_id + span_start + span_end.
   The system mechanically extracts the verbatim quote from the chunk.

4. span_start and span_end must each be a SHORT substring (5-10 words)
   that appears VERBATIM in the chunk's text. Pick distinctive substrings
   that are unlikely to match multiple positions.

REASONING RULES:

5. Lead with the leading authority on the question's specific sub-issue.
   The reader extracts will tell you which case has the operative holding
   (look for is_holding=true and primary_holding_for_question fields).

6. When you cite a case, the chunk_ids you may cite are exactly the
   chunk_ids listed under that case's "case chunk index" in the user
   message. The reader's extracts list paragraph numbers; map those
   paragraph numbers to the chunk_ids that cover them. Do NOT cite
   chunk_ids that weren't fed to you.

7. When you quote, the span_start and span_end MUST be substrings of the
   verbatim text the reader provided for that case (or for statutory
   provisions, of the statutory text shown). The mechanical quote
   extractor will verify this against the underlying chunks.

8. For genuinely fringe questions where the reader extracts don't squarely
   answer the question, use structured uncertainty mode — frame the law
   from the statutory text and the closest analogous authority among the
   extracts. Set confidence to "low" or "medium" accordingly. Do NOT
   refuse just because there is no perfect on-point case; refusal is
   only for questions wholly outside the corpus.

9. If the planner named authorities that the system flagged as NOT
   INDEXED in the corpus, you may mention them by name in the answer
   ("the leading interstate authority is Brodyn v Davenport [2004]
   NSWCA 394, which is not in this corpus") but you may NOT cite or
   quote them.

10. Use only material from the user message. Do not draw on outside
    knowledge for substantive propositions.

ANSWER QUALITY:
- Address the user's specific question first. Yes/no questions get a yes
  or no in the answer_summary up front, then explanation.
- 3-7 propositions is typical. Don't pad. Each proposition stands on
  its own.
- Note material qualifications (the qualifications field of an extract
  matters — fold it into the proposition rather than dropping it).

OUTPUT ONLY THE JSON. No prose, no markdown fences."""


def _format_planner_findings_block(
    plan: planner.Plan,
    resolved_cases: list[dict],
    missed_authorities: list[str],
) -> str:
    lines = ["PLANNER FINDINGS:"]
    lines.append(f"  intent: {plan.intent}")
    lines.append(f"  confidence: {plan.confidence}")
    lines.append(f"  is_fringe: {plan.is_fringe}")
    if plan.notes:
        lines.append(f"  notes: {plan.notes}")
    if resolved_cases:
        lines.append("  named_authorities (resolved against corpus):")
        for c in resolved_cases:
            lines.append(f"    - {c['case_name']} [{c['case_id']}]")
    if missed_authorities:
        lines.append("  named_authorities (NOT INDEXED — may mention by name only, not cite):")
        for a in missed_authorities:
            lines.append(f"    - {a}")
    return "\n".join(lines)


def _format_section_block(section: dict) -> str:
    chunk_ids = ", ".join(section["chunk_ids"])
    return (
        f"[{section['header']}]\n"
        f"  chunk_ids: {chunk_ids}\n\n"
        f"{section['full_text']}"
    )


def _build_case_chunk_index(case: dict, case_chunk_meta: dict[str, dict]) -> str:
    """Build a 'chunk_id -> paragraph range' index string for the reasoner.

    case_chunk_meta is {chunk_id: {paragraph_start, paragraph_end}}.
    """
    lines = []
    for cid in case["chunk_ids"]:
        meta = case_chunk_meta.get(cid, {})
        ps = meta.get("paragraph_start")
        pe = meta.get("paragraph_end")
        if ps and pe and ps != pe:
            lines.append(f"    {cid} covers paragraphs [{ps}]–[{pe}]")
        elif ps:
            lines.append(f"    {cid} covers paragraph [{ps}]")
        else:
            lines.append(f"    {cid}")
    return "\n".join(lines)


def _format_reader_block(
    reader_result: dict,
    case_full: dict,
    case_chunk_meta: dict[str, dict],
) -> str:
    case_name = reader_result.get("case_name") or case_full.get("case_name", "")
    citation = case_full.get("citation", "") or case_name
    case_id = reader_result.get("case_id") or case_full.get("case_id", "")
    chunk_index = _build_case_chunk_index(case_full, case_chunk_meta)

    lines = [f"[{citation}]  (case_id: {case_id})"]
    lines.append("  case chunk index:")
    lines.append(chunk_index)

    if not reader_result.get("is_on_point"):
        lines.append("")
        notes = reader_result.get("notes") or "reader did not find on-point material"
        lines.append(f"  reader_status: not on point ({notes})")
        return "\n".join(lines)

    primary = reader_result.get("primary_holding_for_question")
    if primary:
        lines.append(f"\n  primary_holding_for_question: {primary}")

    for i, extract in enumerate(reader_result.get("extracts", []) or [], 1):
        paras = extract.get("paragraphs") or []
        is_holding = extract.get("is_holding")
        why = extract.get("why_relevant") or ""
        qual = extract.get("qualifications")
        verbatim = extract.get("verbatim") or ""
        para_label = ", ".join(f"[{p}]" for p in paras) if paras else "(unspecified)"
        lines.append(f"\n  extract {i} (paragraphs {para_label}) [is_holding={is_holding}]:")
        if why:
            lines.append(f"    why_relevant: {why}")
        if qual:
            lines.append(f"    qualifications: {qual}")
        lines.append("    verbatim:")
        for line in verbatim.splitlines():
            lines.append(f"      {line}")

    follows = reader_result.get("follows") or []
    distinguishes = reader_result.get("distinguishes") or []
    if follows:
        lines.append(f"\n  follows: {', '.join(follows)}")
    if distinguishes:
        lines.append(f"  distinguishes: {', '.join(distinguishes)}")
    rnotes = reader_result.get("notes")
    if rnotes:
        lines.append(f"  reader_notes: {rnotes}")

    return "\n".join(lines)


def _format_breadth_chunk(h: Hit) -> str:
    meta_short = []
    if h.source_type in ("statute", "regulation"):
        meta_short.append(f"{h.metadata.get('act_short', '')} s {h.metadata.get('section_number', '')}")
    elif h.source_type == "judgment":
        meta_short.append(h.metadata.get("citation", h.header))
    elif h.source_type == "annotated":
        meta_short.append(f"Annotated BIF Act s {h.metadata.get('section_number', '')}")
    elif h.source_type == "decision":
        meta_short.append(f"Adj #{h.metadata.get('decision_id', '')}")
    tag = "[CITABLE]" if h.source_type in {"statute", "regulation", "judgment"} else "[CONTEXT-ONLY]"
    return (
        f"=== {h.chunk_id} {tag} === [{h.source_type}] {' | '.join(meta_short)}\n"
        f"{h.header}\n\n"
        f"{h.text}"
    )


class HardQuestionPipeline:
    """Read-full-cases pipeline (see module docstring)."""
    name = "hard"

    def __init__(self, k_breadth: int = 6):
        self.retriever = Retriever()
        self.k_breadth = k_breadth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str, history: list | None = None) -> dict:
        rewritten = _rewrite_with_history(question, history)
        plan = planner.plan(rewritten, real=True)
        return self.answer_with_plan(
            rewritten, plan, history=history, original_question=question,
        )

    def answer_with_plan(
        self,
        question: str,
        plan: planner.Plan,
        history: list | None = None,
        original_question: str | None = None,
    ) -> dict:
        """Compose an answer for `question` (already-rewritten) given a
        pre-computed plan. Skips re-running the planner."""
        log.info(
            "hard pipeline: intent=%s conf=%s fringe=%s provisions=%s authorities=%s",
            plan.intent, plan.confidence, plan.is_fringe,
            plan.named_provisions, plan.named_authorities,
        )

        # Step 1: full statutory sections
        section_blocks: list[dict] = []
        for prov_name in plan.named_provisions or []:
            act, section = _parse_provision_for_fetch(prov_name)
            if not act or not section:
                continue
            fetched = corpus_fetch.fetch_section_full(act, section)
            if not fetched:
                continue
            if any(s["chunk_ids"] == fetched["chunk_ids"] for s in section_blocks):
                continue
            section_blocks.append(fetched)

        # Step 2: resolve named authorities -> case_ids
        resolved_cases: list[dict] = []
        missed_authorities: list[str] = []
        seen_case_ids: set[str] = set()
        for auth_name in plan.named_authorities or []:
            match = name_index.lookup_case(auth_name)
            if not match or match.case_id in seen_case_ids:
                if not match:
                    missed_authorities.append(auth_name)
                continue
            seen_case_ids.add(match.case_id)
            resolved_cases.append({
                "input": auth_name,
                "case_id": match.case_id,
                "case_name": match.case_name,
                "citation": match.citation,
                "confidence": match.confidence,
            })

        # Step 3: fan out reader workers
        case_ids = [c["case_id"] for c in resolved_cases]
        reader_results = parallel_reader.parallel_read_cases(
            case_ids, question, plan.notes or "",
        )
        reader_by_case_id = {r.get("case_id"): r for r in reader_results}

        # Pre-fetch full case data + per-chunk paragraph metadata for
        # building chunk indexes in the prompt and for postprocessor input.
        case_fulls: dict[str, dict] = {}
        case_chunk_meta: dict[str, dict[str, dict]] = {}
        for cid in case_ids:
            full = corpus_fetch.fetch_case_full(cid)
            if not full:
                continue
            case_fulls[cid] = full
            # Pull the per-chunk paragraph_start/end from name_index so we
            # can label chunk_ids with paragraph ranges in the prompt.
            db = name_index._db()
            rows = db.execute(
                "SELECT chunk_id, paragraph_start, paragraph_end "
                "FROM case_chunks WHERE case_id=? ORDER BY rank_within_case",
                (cid,),
            ).fetchall()
            case_chunk_meta[cid] = {
                r["chunk_id"]: {
                    "paragraph_start": r["paragraph_start"],
                    "paragraph_end": r["paragraph_end"],
                }
                for r in rows
            }

        # Step 4: breadth via the existing 3-channel retriever (capped small)
        breadth_hits, breadth_diag = self.retriever.retrieve_three_channel(
            queries=plan.queries or [question],
            intent=plan.intent,
            named_provisions=[],   # already pulled in step 1
            named_authorities=[],  # already pulled in step 2/3
            k=self.k_breadth,
        )

        # Step 5: build the chunk list passed to postprocessor
        all_chunks = self._build_chunk_list(section_blocks, case_fulls, breadth_hits)

        # If we have nothing at all to work with, refuse
        if not all_chunks:
            empty = postprocessor.FinalAnswer(
                propositions=[], sources=[],
                answer_summary="I could not find sufficient sources for this question.",
                confidence="low",
                answer_html=postprocessor._render_refusal(),
                refused=True,
                flags=["hard pipeline: no statute, no resolved cases, no breadth chunks"],
            )
            result = _to_eval_format(empty)
            result["_pipeline"] = self.name
            result["_planner"] = _planner_summary(plan)
            result["_resolved_cases"] = resolved_cases
            result["_missed_authorities"] = missed_authorities
            return result

        # Step 6: reasoner compose
        answer_json = self._reasoner_compose(
            question=question,
            plan=plan,
            section_blocks=section_blocks,
            reader_by_case_id=reader_by_case_id,
            resolved_cases=resolved_cases,
            missed_authorities=missed_authorities,
            case_fulls=case_fulls,
            case_chunk_meta=case_chunk_meta,
            breadth_hits=breadth_hits,
            history=history,
        )

        # Step 7: postprocessor + verifier
        final = postprocessor.process(answer_json, all_chunks, refused=False)
        verified, _ = verifier.verify(final)
        result = _to_eval_format(verified)
        result["_pipeline"] = self.name
        result["_planner"] = _planner_summary(plan)
        result["_resolved_cases"] = resolved_cases
        result["_missed_authorities"] = missed_authorities
        result["_reader_summary"] = [
            {
                "case_id": r.get("case_id"),
                "is_on_point": r.get("is_on_point"),
                "n_extracts": len(r.get("extracts") or []),
                "primary_holding_for_question": r.get("primary_holding_for_question"),
                "notes": r.get("notes", "")[:200],
            }
            for r in reader_results
        ]
        if original_question is not None and question != original_question:
            result["_rewritten_question"] = question
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_chunk_list(
        self,
        section_blocks: list[dict],
        case_fulls: dict[str, dict],
        breadth_hits: list[Hit],
    ) -> list[Hit]:
        """Build a deduped list of Hit objects covering every chunk_id the
        reasoner is allowed to cite. Postprocessor uses these to extract
        verbatim quotes by span markers."""
        wanted_chunk_ids: list[str] = []
        seen: set[str] = set()
        for sec in section_blocks:
            for cid in sec["chunk_ids"]:
                if cid not in seen:
                    seen.add(cid)
                    wanted_chunk_ids.append(cid)
        for full in case_fulls.values():
            for cid in full["chunk_ids"]:
                if cid not in seen:
                    seen.add(cid)
                    wanted_chunk_ids.append(cid)
        for h in breadth_hits:
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                wanted_chunk_ids.append(h.chunk_id)

        if not wanted_chunk_ids:
            return []

        self.retriever._ensure_loaded()
        chunk_objs = self.retriever._fetch_chunks(wanted_chunk_ids)
        ordered: list[Hit] = []
        for i, cid in enumerate(wanted_chunk_ids):
            h = chunk_objs.get(cid)
            if h is None:
                continue
            h.rank = i + 1
            ordered.append(h)
        return ordered

    def _reasoner_compose(
        self,
        *,
        question: str,
        plan: planner.Plan,
        section_blocks: list[dict],
        reader_by_case_id: dict[str, dict],
        resolved_cases: list[dict],
        missed_authorities: list[str],
        case_fulls: dict[str, dict],
        case_chunk_meta: dict[str, dict[str, dict]],
        breadth_hits: list[Hit],
        history: list | None,
    ) -> dict:
        findings = _format_planner_findings_block(plan, resolved_cases, missed_authorities)

        statute_text = ""
        if section_blocks:
            statute_text = (
                "=== STATUTORY CONTEXT (full text of named provisions) ===\n\n"
                + "\n\n---\n\n".join(_format_section_block(s) for s in section_blocks)
            )

        case_text = ""
        case_blocks = []
        for cid in [c["case_id"] for c in resolved_cases]:
            full = case_fulls.get(cid)
            if not full:
                continue
            reader = reader_by_case_id.get(cid) or {
                "case_id": cid, "case_name": full.get("case_name", ""),
                "is_on_point": False, "extracts": [], "follows": [],
                "distinguishes": [], "notes": "(no reader output)",
            }
            case_blocks.append(_format_reader_block(reader, full, case_chunk_meta.get(cid, {})))
        if case_blocks:
            case_text = (
                "=== CASE READER EXTRACTS (verbatim paragraphs identified by reader) ===\n\n"
                + "\n\n---\n\n".join(case_blocks)
            )

        breadth_text = ""
        if breadth_hits:
            breadth_text = (
                "=== BREADTH CHUNKS (hybrid retrieval, for coverage of anything the planner did not name) ===\n\n"
                + "\n\n".join(_format_breadth_chunk(h) for h in breadth_hits)
            )

        sections = [s for s in (findings, statute_text, case_text, breadth_text) if s]
        user_msg = (
            f"QUESTION:\n{question}\n\n"
            + "\n\n".join(sections)
            + "\n\nCompose your answer now as JSON per the system instructions."
        )

        messages: list[dict] = [{"role": "system", "content": REASONER_SYSTEM_PROMPT}]
        if history:
            messages.extend(_format_history_for_answerer(history))
        messages.append({"role": "user", "content": user_msg})

        text, _usage = llm_config.complete_chat(
            messages=messages,
            kind="reasoner",
            operation="reasoner",
            response_format={"type": "json_object"},
            max_output_tokens=4000,
        )
        try:
            import json as _json
            return _json.loads(text)
        except Exception as e:
            log.warning("reasoner JSON parse failed: %s; returning empty", e)
            return {
                "propositions": [],
                "answer_summary": "Internal error: reasoner output was not valid JSON.",
                "confidence": "low",
                "_parse_error": str(e),
                "_raw": (text or "")[:500],
            }
