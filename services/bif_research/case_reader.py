"""Reader-worker that sends a full judgment to a cheap model and gets back
verbatim on-point paragraphs. The reader does not paraphrase or summarise;
it identifies which paragraphs bear on the question and quotes them exactly.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from . import corpus_fetch, llm_config

log = logging.getLogger("bif_research.case_reader")


READER_SYSTEM_PROMPT = """You are a legal research reader. You will be given the FULL TEXT of a single
Queensland judgment and a research question. Your job is NOT to answer the
question. Your job is to identify which paragraphs of THIS judgment bear on
the question and emit them VERBATIM with their paragraph numbers.

CRITICAL RULES:

1. Quote paragraphs EXACTLY. Copy text character-for-character from the
   judgment. Do not paraphrase. Do not summarise. Do not "clean up" the
   text. Preserve [N] paragraph markers, capitalisation, punctuation,
   internal quotation marks, everything.

2. Only include paragraphs that genuinely bear on the question. If a
   paragraph is on-point, include the full paragraph (or the contiguous
   range of paragraphs needed to make the holding intelligible). Do not
   excerpt single sentences out of paragraphs.

3. If the case is not on point at all, set is_on_point=false and return
   empty extracts. Do not force-fit irrelevant material.

4. Distinguish ratio from dicta. Set is_holding=true for paragraphs that
   contain the operative holding on the question; is_holding=false for
   passing dicta or background discussion that's still useful context.

5. Note any qualifications. If the holding is qualified ("subject to...",
   "save where...", "this does not extend to..."), capture the
   qualification text in the qualifications field of the same extract.

6. follows / distinguishes: list cases by name as they appear in the
   judgment. These help the reasoner trace doctrine.

7. primary_holding_for_question is YOUR one-sentence label of what the
   case did that matters for the question. This is the only place you
   write in your own words. Everything else is verbatim.

OUTPUT STRICT JSON only matching the schema given. No prose, no fences."""


def _safe_load_json(text: str) -> dict:
    """Parse JSON, tolerating accidental ```json fences or trailing prose."""
    if not text:
        raise ValueError("empty model response")
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    if not s.startswith("{"):
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            s = m.group(0)
    return json.loads(s)


def _stub(case_id: str, case_name: str, notes: str) -> dict:
    return {
        "case_id": case_id,
        "case_name": case_name,
        "is_on_point": False,
        "primary_holding_for_question": None,
        "extracts": [],
        "follows": [],
        "distinguishes": [],
        "notes": notes,
    }


def read_case_for_question(case_id: str, question: str,
                           planner_notes: str = "") -> dict:
    """Send the full case text + question to the reader-model chain.

    Returns a structured dict of verbatim on-point paragraphs. On any
    failure (case not in corpus, oversized text, model error, JSON
    parse failure) returns a safe stub with is_on_point=false.
    """
    case = corpus_fetch.fetch_case_full(case_id)
    if case is None:
        log.info("case %s not in corpus", case_id)
        return _stub(case_id, "", "case not in corpus")

    case_name = case.get("case_name", "") or ""
    full_text = case.get("full_text", "") or ""

    if len(full_text) > 60_000:
        log.info("skipping %s: full text %d chars exceeds 60000", case_id, len(full_text))
        return _stub(
            case_id,
            case_name,
            f"skipped — case full text {len(full_text)} chars > 60000",
        )

    citation = case.get("citation", "") or case_name
    user_msg = (
        f"=== QUESTION ===\n{question}\n\n"
        f"=== PLANNER NOTES ===\n{planner_notes or '(none)'}\n\n"
        f"=== CASE ===\n"
        f"case_id: {case_id}\n"
        f"case_name: {case_name}\n"
        f"citation: {citation}\n\n"
        f"=== FULL JUDGMENT TEXT ===\n{full_text}\n\n"
        f"Now emit the JSON object per the system instructions."
    )

    try:
        text, usage = llm_config.complete_chat(
            messages=[
                {"role": "system", "content": READER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            kind="reader",
            operation=f"read-case:{case_id}",
            response_format={"type": "json_object"},
            max_output_tokens=4000,
        )
    except Exception as e:
        log.warning("reader call failed for %s: %s", case_id, e)
        return _stub(case_id, case_name, f"read failed: {e}")

    try:
        data = _safe_load_json(text)
    except Exception as e:
        log.warning("reader JSON parse failed for %s: %s", case_id, e)
        return _stub(case_id, case_name, f"read failed: json parse: {e}")

    if not isinstance(data, dict):
        return _stub(case_id, case_name, "read failed: model returned non-object")

    data["case_id"] = case_id
    data["case_name"] = case_name
    data.setdefault("is_on_point", False)
    data.setdefault("primary_holding_for_question", None)
    data.setdefault("extracts", [])
    data.setdefault("follows", [])
    data.setdefault("distinguishes", [])
    data.setdefault("notes", "")
    return data
