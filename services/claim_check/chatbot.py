"""Contextual chatbot for /claim-check.

The system prompt is assembled at request time from:
  - The user's selected mode
  - A truncated view of the extracted document text
  - The current list of check results (titles + statuses + explanations)
  - A strict scope instruction (BIF Act / payment claim / payment schedule only)
  - A compulsory disclaimer to end substantive answers

Heavy imports deferred so this module loads cheaply at server startup.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("claim_check.chatbot")

MODEL = "gpt-4-turbo-preview"
MAX_DOC_CHARS = 12_000
MAX_HISTORY_TURNS = 8
MAX_OUTPUT_TOKENS = 700

MODE_LABELS = {
    "payment_claim_outgoing": "a payment claim the user is about to serve",
    "payment_claim_incoming": "a payment claim the user has received",
    "payment_schedule_outgoing": "a payment schedule the user is about to give",
    "payment_schedule_incoming": "a payment schedule the user has received",
}


def _build_system_prompt(mode: str, document_text: str, check_results: list[dict]) -> str:
    mode_label = MODE_LABELS.get(mode, "a payment claim or payment schedule")

    doc_snippet = (document_text or "").strip()[:MAX_DOC_CHARS]
    if len(document_text or "") > MAX_DOC_CHARS:
        doc_snippet += "\n\n[...document truncated for context...]"

    check_lines = []
    for c in (check_results or [])[:25]:
        status = c.get("status", "?")
        title = c.get("title", "")
        section = c.get("section", "")
        explanation = c.get("explanation", "")
        check_lines.append(f"- [{status.upper()}] {title} ({section}) — {explanation}")
    checks_block = "\n".join(check_lines) if check_lines else "(No checks run yet.)"

    return f"""You are a specialist assistant for the Queensland Building Industry Fairness (Security of Payment) Act 2017 ("BIF Act") and related construction-law matters in Queensland (including the QBCC Act 1991).

You are talking to a user about {mode_label}. The user has uploaded their document (shown below) and has already seen a structured compliance analysis (also shown below).

SCOPE:
You answer ONLY questions related to:
  - The BIF Act (payment claims, payment schedules, adjudication, service, reference dates, interest, etc.)
  - The QBCC Act and licensing issues that bear on BIF Act claims
  - The specific document and check results in front of the user
If asked about anything outside this scope, politely say you only help with BIF Act / payment claim / payment schedule matters and redirect.

STYLE:
  - Australian legal terminology and spelling.
  - Cite BIF Act sections as "s 68(1)(a)" and QBCC Act sections as "s 42 QBCC Act".
  - Quote passages from the user's document where directly relevant (use short verbatim quotes in italics).
  - Do not invent facts or sections. If you are unsure, say so.
  - Do not provide legal advice for the user's specific decisions. You can explain what the Act says and what the check flagged.

MANDATORY TRAILING DISCLAIMER:
End every substantive answer with exactly this line on its own paragraph:
"General information only — not legal advice. For your specific situation, consult a construction lawyer."

USER'S DOCUMENT (truncated):
---
{doc_snippet}
---

CHECK RESULTS FROM THE MOST RECENT ANALYSIS:
{checks_block}
"""


def chat(
    mode: str,
    document_text: str,
    check_results: list[dict],
    history: list[dict],
    user_message: str,
) -> str:
    if not (user_message or "").strip():
        raise ValueError("Empty message.")

    from openai import OpenAI  # deferred

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured on the server.")

    client = OpenAI(api_key=api_key)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(mode, document_text, check_results)}
    ]

    # Keep the last N turns (user+assistant pairs) to bound token use.
    trimmed = [m for m in (history or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    for m in trimmed[-(2 * MAX_HISTORY_TURNS):]:
        messages.append({"role": m["role"], "content": str(m.get("content", ""))})

    messages.append({"role": "user", "content": user_message})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as e:
        log.exception("OpenAI call failed in chat")
        raise RuntimeError(f"Chat failed: {e}") from e

    return (resp.choices[0].message.content or "").strip()
