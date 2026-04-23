"""Scan an uploaded construction contract for reference-date clause language.

Used by PC-005 (reference date) when the user uploads their contract via
the "More ways to clarify" expander. We do a keyword pre-filter to find
candidate paragraphs (so the prompt stays small), then ask the LLM to
identify and extract reference-date clauses, returning clause numbers and
the exact clause text.

Per spec Section 4: contract file is held in memory for the request only.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from . import llm_config

log = logging.getLogger("claim_check.contract_scanner")

KEYWORDS = (
    "reference date",
    "reference dates",
    "progress claim date",
    "progress claim",
    "progress payment",
    "claim date",
    "claim month",
    "date of the claim",
    "monthly claim",
)
CLAUSE_HINTS = ("clause 37", "clause 38", "clause 39", "clause 42", "clause 45")

MAX_CHUNK_CHARS = 3500
MAX_CHUNKS = 6
CONTEXT_WINDOW = 900  # chars around each keyword match


def _find_candidate_windows(text: str) -> list[str]:
    """Return up to MAX_CHUNKS text windows likely to contain reference-date language."""
    text = text or ""
    positions: list[int] = []
    lower = text.lower()
    for kw in KEYWORDS + CLAUSE_HINTS:
        start = 0
        while True:
            idx = lower.find(kw, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(kw)
    if not positions:
        return []
    positions.sort()

    windows: list[tuple[int, int]] = []
    for p in positions:
        s = max(0, p - CONTEXT_WINDOW // 2)
        e = min(len(text), p + CONTEXT_WINDOW // 2)
        if windows and s <= windows[-1][1] + 200:
            # merge
            windows[-1] = (windows[-1][0], max(windows[-1][1], e))
        else:
            windows.append((s, e))
        if len(windows) >= MAX_CHUNKS:
            break

    return [text[s:e] for s, e in windows]


def find_reference_date_clauses(contract_text: str) -> dict[str, Any]:
    """Returns:
        {
          "found": bool,
          "clauses": [{"clause_ref": "Clause 37.1", "text": "..."}],
          "notes": "..."  # optional (e.g. "no reference-date language detected")
        }
    """
    text = (contract_text or "").strip()
    if not text:
        return {"found": False, "clauses": [], "notes": "No text was extracted from the contract."}

    windows = _find_candidate_windows(text)
    if not windows:
        return {"found": False, "clauses": [], "notes": "No reference-date language detected in the contract. This may mean the contract is silent on reference dates, in which case s 67(2) BIF Act's statutory defaults apply."}

    corpus = "\n\n--- WINDOW ---\n\n".join(windows)[: MAX_CHUNK_CHARS * MAX_CHUNKS]

    system = (
        "You extract reference-date clause language from construction contracts under the "
        "Queensland BIF Act. Return JSON only. If no relevant clause appears, return an empty clauses array."
    )
    user = (
        "Find clauses that govern the REFERENCE DATE or the timing of progress claims under the contract. "
        "Common terms: 'reference date', 'progress claim', 'progress payment', clause 37 / 38 / 42 / 45 of AS 4000, AS 4902, AS 2124 etc. "
        "Return JSON with this schema:\n"
        '{ "found": true|false, "clauses": [{ "clause_ref": "...", "text": "<verbatim>" }], "notes": "..." }\n\n'
        "Constraints:\n"
        "- The 'text' for each clause MUST be verbatim from the contract — do not paraphrase.\n"
        "- Prefer shorter, more on-point extracts over whole-page dumps.\n"
        "- If no reference-date language appears in the windows below, return found:false with a short note.\n\n"
        f"CONTRACT WINDOWS:\n---\n{corpus}\n---"
    )

    try:
        resp = llm_config.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            reasoning_effort="medium",
            tier="default",
            response_format={"type": "json_object"},
            max_output_tokens=900,
        )
    except Exception as e:
        log.exception("contract scan LLM failed")
        return {"found": False, "clauses": [], "notes": f"Automated scan failed: {e}"}

    try:
        data = json.loads(resp["content"] or "{}")
    except json.JSONDecodeError:
        return {"found": False, "clauses": [], "notes": "Automated scan returned an unparseable response. Please paste the clause text manually."}

    clauses_raw = data.get("clauses") or []
    clauses: list[dict[str, str]] = []
    for c in clauses_raw:
        if not isinstance(c, dict):
            continue
        ref = str(c.get("clause_ref") or "").strip()
        text_c = str(c.get("text") or "").strip()
        if not text_c:
            continue
        clauses.append({"clause_ref": ref, "text": text_c[:2000]})

    return {
        "found": bool(data.get("found")) and bool(clauses),
        "clauses": clauses,
        "notes": str(data.get("notes") or "").strip(),
    }
