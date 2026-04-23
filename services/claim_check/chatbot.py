"""Contextual chatbot for /claim-check.

Uses ``llm_config.complete`` so the same model selection / reasoning /
fallback / cost-logging logic applies here as to the rule engine.

Escalation: chat messages that contain multi-step indicators or exceed
200 characters bump reasoning_effort from medium to high.
"""

from __future__ import annotations

import logging
from typing import Any

from . import llm_config

log = logging.getLogger("claim_check.chatbot")

MAX_DOC_CHARS = 12_000
MAX_HISTORY_TURNS = 8
MAX_OUTPUT_TOKENS = 700

MODE_LABELS = {
    "payment_claim_serving":    "a payment claim the user is about to serve",
    "payment_claim_received":   "a payment claim the user has received",
    "payment_schedule_giving":  "a payment schedule the user is about to give",
    "payment_schedule_received": "a payment schedule the user has received",
}


def _build_system_prompt(mode: str, document_text: str, check_results: list[dict]) -> str:
    mode_label = MODE_LABELS.get(mode, "a payment claim or payment schedule")
    doc_snippet = (document_text or "").strip()[:MAX_DOC_CHARS]
    if len(document_text or "") > MAX_DOC_CHARS:
        doc_snippet += "\n\n[...document truncated for context...]"

    check_lines: list[str] = []
    for c in (check_results or [])[:25]:
        status = c.get("status", "?")
        title = c.get("title", "")
        section = c.get("section", "")
        explanation = c.get("explanation", "")
        check_lines.append(f"- [{status.upper()}] {title} ({section}) — {explanation}")
    checks_block = "\n".join(check_lines) if check_lines else "(No checks run yet.)"

    return f"""You are a specialist assistant for the Queensland Building Industry Fairness (Security of Payment) Act 2017 ("BIF Act") and related construction-law matters in Queensland (including the QBCC Act 1991).

You are talking to a user about {mode_label}. The user has uploaded their document (shown below) and has already seen a structured compliance analysis (also shown below).

SCOPE — answer questions about any of:
  - The BIF Act (payment claims, payment schedules, adjudication, service, reference dates, interest, etc.)
  - The QBCC Act and its payment-related provisions, including licensing that bears on the BIF Act
  - The user's specific uploaded document and the check results above — including why a check was flagged, what the document does or does not say, what would change the outcome, and how to fix drafting weaknesses
  - Related Queensland construction-law topics where they directly bear on the above
Decline and redirect only on topics that are clearly off-topic for a BIF Act / QBCC research tool (e.g. personal legal advice for the user's decisions, other jurisdictions' security-of-payment regimes unless clearly relevant, medical / tax / unrelated subjects). When you decline, be brief and point them back to what this tool can help with.

STYLE:
  - Australian legal terminology and spelling.
  - Cite BIF Act sections as "s 68(1)(a)" and QBCC Act sections as "s 42 QBCC Act".
  - Quote passages from the user's document where directly relevant (short verbatim quotes in italics or block-quote markdown).
  - HEDGE your language — do not speak in guarantees. Prefer "appears to", "on its face", "likely", "arguable" over absolutes.
  - Do not invent facts or sections. If you are unsure, say so plainly.
  - You may use markdown formatting: **bold**, *italic*, bullet lists, numbered lists, inline `code`, and > blockquotes. The UI renders these safely. Do NOT emit raw HTML.
  - You provide general information about the Act; you do not provide legal advice on the user's specific decisions.

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

    reasoning = llm_config.reasoning_for_chat(user_message)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(mode, document_text, check_results)}
    ]
    trimmed = [m for m in (history or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    for m in trimmed[-(2 * MAX_HISTORY_TURNS):]:
        messages.append({"role": m["role"], "content": str(m.get("content", ""))})
    messages.append({"role": "user", "content": user_message})

    resp = llm_config.complete(
        messages=messages,
        reasoning_effort=reasoning,
        tier="default",
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    return resp["content"] or ""
