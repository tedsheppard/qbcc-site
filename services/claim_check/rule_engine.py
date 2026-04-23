"""BIF Act rule engine for /claim-check.

Current implementation:
  - LLM-based holistic compliance check per mode, using the model's
    knowledge of the BIF Act (Qld). Returns a structured list of checks.
  - Matches the four modes defined on the frontend.

Future (stage 5+):
  - Parse rules/bif_act_rules.md and execute deterministic / semantic /
    user-input checks per the authored rule set. The run_checks entry
    point will remain the same so the frontend does not change.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger("claim_check.rule_engine")

MODEL = "gpt-4-turbo-preview"
MAX_DOC_CHARS_FOR_PROMPT = 18_000  # ~12K tokens; leaves headroom for the system prompt + output
MAX_OUTPUT_TOKENS = 2200


MODE_LABELS = {
    "payment_claim_outgoing": "A payment claim the user is about to serve under the BIF Act (Qld)",
    "payment_claim_incoming": "A payment claim the user has received under the BIF Act (Qld)",
    "payment_schedule_outgoing": "A payment schedule the user is about to give under the BIF Act (Qld)",
    "payment_schedule_incoming": "A payment schedule the user has received under the BIF Act (Qld)",
}


CHECKLIST_BY_MODE = {
    "payment_claim_outgoing": """Check ALL of the following requirements for a valid payment claim:
1. s 68(1)(a) — Identifies the construction work (or related goods/services) to which the claim relates.
2. s 68(1)(b) — States the amount claimed.
3. s 68(1)(c) — Requests payment of the claimed amount.
4. s 75 — Served under the right contract and within the statutory timeframe (needs contract date from user).
5. s 75(4) — Only one payment claim per reference date (ask user if earlier claims have been served).
6. s 67 — Reference date is valid under the contract (ask user when the contract was entered into).
7. QBCC Act s 42 — Claimant appears to be properly licensed for the work described. If nothing in the document indicates licensing, flag as needing user confirmation.
8. Correct form of claim — the document is identifiable as a "payment claim" (need not use those exact words post-2018 amendments but must substantively request payment under the Act).

For this mode, identify DRAFTING WEAKNESSES that the user should fix before serving.""",

    "payment_claim_incoming": """Check ALL of the following requirements and also identify any defences / knockout arguments:
1. s 68(1)(a) — Identifies the construction work. Flag vague or missing identification.
2. s 68(1)(b) — States the amount claimed (unambiguously).
3. s 68(1)(c) — Requests payment.
4. s 75 — Served under the correct contract, within timeframe (needs contract date + service date from user).
5. s 75(4) — Only one claim per reference date (ask user if earlier claims were received).
6. s 67 — Reference date is valid (ask user when the contract was entered into).
7. QBCC Act s 42 / unlicensed work — if the claim is for building work and the claimant does not appear licensed, this can invalidate the claim.
8. Service requirements under s 102 and the contract.

For each check, if failed or warning, state the specific KNOCKOUT ARGUMENT the respondent could make.""",

    "payment_schedule_outgoing": """Check ALL of the following requirements for a valid payment schedule:
1. s 76(2)(a) — Identifies the payment claim to which it responds.
2. s 76(2)(b) — States the amount of the payment the respondent proposes to make ("scheduled amount").
3. s 76(3) — If the scheduled amount is less than the claimed amount, states the respondent's reasons for the difference AND the reasons for withholding payment.
4. s 76(1) — Served within the required timeframe (ask user when the payment claim was received and whether the contract prescribes a shorter period).
5. Completeness — every reason for withholding payment is stated; later adjudication is limited to reasons in the schedule (s 82(4)).

For this mode, identify GAPS that would limit the respondent's adjudication defence.""",

    "payment_schedule_incoming": """Check ALL of the following requirements AND assess whether the schedule is VALID AT ALL:
1. s 76(2)(a) — Identifies the payment claim it responds to.
2. s 76(2)(b) — States a scheduled amount. If missing, the schedule is likely invalid and the respondent defaults to owing the full claimed amount (see s 77).
3. s 76(3) — Reasons for any withholding are stated. Vague / boilerplate reasons may be insufficient.
4. s 76(1) — Given within the statutory timeframe (ask user when the payment claim was served).
5. Reasons scope — reasons outside the schedule cannot be relied on at adjudication (s 82(4)).

Explicitly state in the explanation of check #2 whether the schedule appears VALID or INVALID.""",
}


SYSTEM_PREFIX = """You are an expert in Queensland's Building Industry Fairness (Security of Payment) Act 2017 ("BIF Act") and the QBCC Act 1991. You analyse payment-claim-related documents for compliance.

Australian legal terminology and spelling. Cite BIF Act sections as "s 68(1)(a) BIF Act" and QBCC Act sections as "s 42 QBCC Act".

Output MUST be valid JSON matching this schema exactly:

{
  "summary": "<one-sentence summary of the document's nature and apparent claimed/scheduled amount if stated>",
  "checks": [
    {
      "id": "<short stable id like 'pc-68-1-a'>",
      "status": "pass" | "warning" | "fail" | "input",
      "title": "<one-line plain-English check title>",
      "section": "<Act reference, e.g. 's 68(1)(a)'>",
      "explanation": "<plain-English reason for the status, 1-3 sentences>",
      "quote": "<optional: a short exact quote from the document supporting the status; leave empty string if none>",
      "query": "<Meilisearch-friendly keyword query to find relevant adjudication decisions for this check>",
      "prompt": "<only if status='input': the question to ask the user>",
      "input_type": "date" | "yes-no" | "text"
    }
  ]
}

Rules for statuses:
- "pass"   — clearly satisfied on the face of the document.
- "warning"— arguable, vague, or technically satisfied but drafting-weak.
- "fail"   — clearly not satisfied, or missing entirely.
- "input"  — cannot be determined from the document alone; needs a fact from the user. The "prompt" field must contain a concise question.

Rules for output:
- Include EVERY check listed below, even if the answer is "input".
- Do not invent sections the Act does not contain.
- If you are unsure, return "warning" with a clear explanation, rather than inventing facts.
- The "quote" must be verbatim text from the document or an empty string. Never paraphrase inside quote.
- Return JSON only — no prose before or after.
"""


def run_checks(mode: str, document_text: str, user_answers: dict | None = None) -> dict[str, Any]:
    """Run the compliance checks for the given mode and document text.

    Returns: {"summary": str, "checks": [ {id, status, title, section, explanation, quote, query, prompt?, input_type?}, ... ]}
    """
    if mode not in CHECKLIST_BY_MODE:
        raise ValueError(f"Unknown mode: {mode!r}")

    from openai import OpenAI  # deferred

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured on the server.")

    client = OpenAI(api_key=api_key)

    doc_for_prompt = (document_text or "")[:MAX_DOC_CHARS_FOR_PROMPT]
    mode_label = MODE_LABELS[mode]
    checklist = CHECKLIST_BY_MODE[mode]

    answers_block = ""
    if user_answers:
        # User answers feed back into the analysis so previously-'input' checks can resolve.
        answer_lines = [f"- {k}: {v}" for k, v in user_answers.items() if v not in (None, "")]
        if answer_lines:
            answers_block = "\n\nPREVIOUS USER ANSWERS (apply to the relevant checks):\n" + "\n".join(answer_lines)

    user_content = (
        f"Mode: {mode_label}\n\n"
        f"{checklist}"
        f"{answers_block}\n\n"
        f"DOCUMENT TEXT:\n---\n{doc_for_prompt}\n---\n\n"
        "Return the JSON per the schema."
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PREFIX},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        log.exception("OpenAI call failed in run_checks")
        raise RuntimeError(f"Analysis failed: {e}") from e

    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.error("Model returned invalid JSON: %s", raw[:500])
        raise RuntimeError("Analysis returned invalid JSON. Please retry.")

    checks = data.get("checks") or []
    # Normalise/guardrail the output so the frontend never has to defensively parse.
    normalised = []
    for i, c in enumerate(checks):
        if not isinstance(c, dict):
            continue
        status = c.get("status")
        if status not in ("pass", "warning", "fail", "input"):
            status = "warning"
        item = {
            "id": str(c.get("id") or f"check-{i+1}"),
            "status": status,
            "title": str(c.get("title") or "").strip() or "Untitled check",
            "section": str(c.get("section") or "").strip(),
            "explanation": str(c.get("explanation") or "").strip(),
            "quote": str(c.get("quote") or "").strip(),
            "query": str(c.get("query") or c.get("title") or "").strip(),
        }
        if status == "input":
            item["prompt"] = str(c.get("prompt") or "Please provide more information.").strip()
            it = c.get("input_type") or "text"
            item["input_type"] = it if it in ("date", "yes-no", "text") else "text"
        normalised.append(item)

    return {
        "summary": str(data.get("summary") or "").strip(),
        "checks": normalised,
    }
