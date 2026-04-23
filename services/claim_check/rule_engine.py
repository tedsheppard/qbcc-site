"""BIF Act rule engine for /claim-check.

Reads rules from rules/bif_act_rules.md (via rules_parser). For each rule
applicable to the selected mode, runs the appropriate check:

  - `deterministic`  — evaluated in Python (e.g. date arithmetic)
  - `user_input`     — presence/shape check on user-provided answers
  - `semantic`       — LLM call using the document text + rule criteria
  - `qbcc_lookup`    — consume a licensee record selected by the user

A single rule can combine multiple implementations (e.g. `semantic +
user_input` for the reference-date rule); the engine merges sub-results.

Hedged result language (spec Section 7) is applied here — the engine
returns status plus a short plain-English summary line the frontend uses
verbatim above the rule explanation.

The engine also supports running a single rule (for streaming / parallel
execution under Section 4); see ``run_single_rule``.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

from . import llm_config, rules_parser

log = logging.getLogger("claim_check.rule_engine")

MODE_LABELS = {
    "payment_claim_serving":    "a payment claim the user is about to serve",
    "payment_claim_received":   "a payment claim the user has received",
    "payment_schedule_giving":  "a payment schedule the user is about to give",
    "payment_schedule_received": "a payment schedule the user has received",
}

# Hedged status summary lines per spec Section 7.
STATUS_SUMMARY = {
    "pass":    "No issues detected based on the information provided",
    "warning": "Potential issue identified — review recommended",
    "fail":    "Likely non-compliant — this requires attention",
    "input":   "Additional information required to complete this check",
}

MAX_DOC_CHARS_FOR_PROMPT = 18_000


SEMANTIC_SYSTEM_PROMPT = """You are a compliance checker for Queensland's Building Industry Fairness (Security of Payment) Act 2017 ("BIF Act") and the QBCC Act 1991. You assess ONE rule at a time against the document text provided.

Australian legal terminology and spelling. Cite BIF Act sections as "s 68(1)(a)" and QBCC Act sections as "s 42 QBCC Act".

You MUST return a JSON object with this schema:

{
  "status": "pass" | "warning" | "fail",
  "explanation": "<plain-English reasoning, 1-3 sentences. HEDGED language — do not speak in guarantees. Use phrases like 'appears to', 'on its face', 'likely satisfies' rather than absolutes>",
  "quote": "<verbatim short quote from the document supporting the finding, OR empty string if none>"
}

Rules:
- Evaluate ONLY against the pass/warning/fail criteria provided for this rule.
- If the document text is silent on the point, prefer "warning" over guessing "pass".
- The quote must be verbatim text from the document or empty string. Do not paraphrase.
- Return JSON only. No prose before or after."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_checks(mode: str, document_text: str, user_answers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run every applicable rule for the mode and return a structured result.

    Kept synchronous for backwards-compat with the existing /analyse endpoint.
    The SSE streaming endpoint (Section 4) calls ``run_single_rule`` per rule
    in parallel; this function is the sequential equivalent for non-streaming
    callers.
    """
    rules = rules_parser.rules_for_mode(mode)
    if not rules:
        return {"summary": "", "checks": []}

    user_answers = user_answers or {}
    checks: list[dict[str, Any]] = []
    for rule in rules:
        checks.append(run_single_rule(mode, rule, document_text, user_answers))

    # A lightweight summary line. Rule engine doesn't need its own model call for this;
    # we derive from the document text if we can find a claimed/scheduled amount.
    summary = _derive_summary(mode, document_text)

    return {"summary": summary, "checks": checks}


def run_single_rule(
    mode: str,
    rule: dict[str, Any],
    document_text: str,
    user_answers: dict[str, Any] | None,
) -> dict[str, Any]:
    """Execute one rule and return a structured check result."""
    user_answers = user_answers or {}
    impl = rule.get("implementation") or []

    # Start with a neutral result; the strongest component wins.
    components: list[tuple[str, str, str]] = []  # (status, explanation, quote)

    if "user_input" in impl or "qbcc_lookup" in impl:
        comp = _evaluate_user_input(rule, user_answers)
        if comp:
            components.append(comp)

    if "deterministic" in impl:
        comp = _evaluate_deterministic(rule, user_answers)
        if comp:
            components.append(comp)

    semantic_result: dict[str, Any] | None = None
    if "semantic" in impl and document_text.strip():
        semantic_result = _evaluate_semantic(rule, document_text)
        if semantic_result:
            components.append(
                (semantic_result["status"], semantic_result["explanation"], semantic_result.get("quote", ""))
            )

    if not components:
        # No implementation ran — treat as input-needed so the UI prompts the user.
        status = "input"
        explanation = rule.get("pass_criteria") or "Awaiting document and inputs."
        quote = ""
    else:
        status, explanation, quote = _combine_components(components)

    # Hedge the explanation if it reads like a guarantee.
    explanation = _hedge(explanation)

    return {
        "id": rule["id"],
        "status": status,
        "status_summary": STATUS_SUMMARY[status],
        "title": rule["title"],
        "section": rule.get("act_reference", ""),
        "explanation": explanation,
        "quote": quote or "",
        "query": rule.get("search_query") or rule["title"],
        "escalated": (rule.get("escalate") or "").lower() == "high-reasoning",
        "model": semantic_result.get("model") if semantic_result else None,
        "reasoning": semantic_result.get("reasoning") if semantic_result else None,
    }


# ---------------------------------------------------------------------------
# Component evaluators
# ---------------------------------------------------------------------------

def _evaluate_user_input(rule: dict[str, Any], answers: dict[str, Any]) -> tuple[str, str, str] | None:
    """Check whether required user inputs are present. Evaluates conditional visibility."""
    inputs = rule.get("inputs") or []
    if not inputs:
        return None

    missing: list[str] = []
    for q in inputs:
        if not q.get("required"):
            continue
        if not _show_if_satisfied(q.get("show_if"), answers):
            continue
        val = answers.get(q["id"])
        if val in (None, "", []):
            missing.append(q.get("question") or q["id"])

    if missing:
        q_list = "; ".join(missing[:3])
        more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        return ("input", f"Answer required: {q_list}{more}", "")

    return None


def _evaluate_deterministic(rule: dict[str, Any], answers: dict[str, Any]) -> tuple[str, str, str] | None:
    """A couple of built-in deterministic checks for rules that rely on date arithmetic.

    We keep these targeted — only PC-004 (6-month window) and PS-004 (15 business days).
    Everything else falls through to the semantic engine.
    """
    rid = rule["id"]
    if rid == "PC-004":
        last_work = _parse_date(answers.get("last_work_date"))
        served = _parse_date(answers.get("served_date"))
        if not last_work or not served:
            return None
        delta_days = (served - last_work).days
        is_final = (answers.get("is_final_claim") or "").lower().startswith("yes")
        if delta_days <= 180:
            return ("pass", f"Served {delta_days} days after the last day of work — within the 6-month window in s 75(4)(a).", "")
        elif is_final:
            return ("warning", f"Served {delta_days} days after the last day of work. Longer final-claim windows in s 75(4)(b)–(d) may apply — verify against the contract.", "")
        else:
            return ("fail", f"Served {delta_days} days after the last day of work — more than the 6 months permitted by s 75(4)(a).", "")

    if rid == "PS-004":
        claim_received = _parse_date(answers.get("claim_received_date"))
        schedule_served = _parse_date(answers.get("schedule_served_date"))
        contract_days = answers.get("contract_timeframe_days")
        try:
            contract_days = int(contract_days) if contract_days not in (None, "") else None
        except (TypeError, ValueError):
            contract_days = None
        if not claim_received:
            return None
        deadline_days = min(contract_days, 15) if contract_days else 15
        if not schedule_served:
            return ("warning", f"Deadline to give the payment schedule is {deadline_days} business day(s) after receipt of the claim (s 76(1)). Confirm scheduled service date.", "")
        # Approximate business days between dates.
        business_days = _business_days_between(claim_received, schedule_served)
        if business_days <= deadline_days:
            return ("pass", f"Given {business_days} business day(s) after receipt of the payment claim — within the {deadline_days}-day period.", "")
        return ("fail", f"Given {business_days} business day(s) after receipt of the payment claim — outside the {deadline_days}-day period in s 76(1). Respondent may be liable for the full claimed amount under s 77.", "")

    return None


def _evaluate_semantic(rule: dict[str, Any], document_text: str) -> dict[str, Any] | None:
    """Send the rule criteria + document text to the LLM for adjudication."""
    reasoning = llm_config.reasoning_for_rule(rule, default="medium")
    doc = (document_text or "")[:MAX_DOC_CHARS_FOR_PROMPT]

    criteria_block = _format_criteria(rule)

    user = (
        f"RULE: {rule['id']} — {rule['title']}\n"
        f"ACT REFERENCE: {rule.get('act_reference', '')}\n\n"
        f"CRITERIA:\n{criteria_block}\n\n"
    )
    if rule.get("quote_requirement"):
        user += f"QUOTE GUIDANCE: {rule['quote_requirement']}\n\n"
    user += f"DOCUMENT TEXT:\n---\n{doc}\n---\n\nReturn JSON per the schema."

    try:
        resp = llm_config.complete(
            messages=[
                {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            reasoning_effort=reasoning,
            tier="default",
            response_format={"type": "json_object"},
            max_output_tokens=700,
        )
    except llm_config.CostCapExceededError:
        raise
    except Exception as e:
        log.exception("semantic check failed for %s", rule["id"])
        return {
            "status": "warning",
            "explanation": f"Automated check could not run ({e}). Review manually.",
            "quote": "",
            "model": None,
            "reasoning": None,
        }

    try:
        data = json.loads(resp["content"] or "{}")
    except json.JSONDecodeError:
        return {
            "status": "warning",
            "explanation": "Automated check returned an unparseable response. Review manually.",
            "quote": "",
            "model": resp.get("model"),
            "reasoning": resp.get("reasoning"),
        }

    status = data.get("status")
    if status not in ("pass", "warning", "fail"):
        status = "warning"

    return {
        "status": status,
        "explanation": str(data.get("explanation") or "").strip(),
        "quote": str(data.get("quote") or "").strip(),
        "model": resp.get("model"),
        "reasoning": resp.get("reasoning"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _show_if_satisfied(show_if: str | None, answers: dict[str, Any]) -> bool:
    if not show_if:
        return True
    m = rules_parser._SHOW_IF_RE.match(show_if) if isinstance(show_if, str) else None
    if not m:
        return True
    target_id, target_val = m.group(1), m.group(2)
    return str(answers.get(target_id, "")) == target_val


def _combine_components(components: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    """Combine multiple component results into one. Priority: input > fail > warning > pass."""
    priority = {"input": 4, "fail": 3, "warning": 2, "pass": 1}
    winner = max(components, key=lambda c: priority.get(c[0], 0))
    status = winner[0]
    explanations = [c[1] for c in components if c[1]]
    quote = next((c[2] for c in components if c[2]), "")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq = []
    for e in explanations:
        if e in seen:
            continue
        seen.add(e)
        uniq.append(e)
    return status, " ".join(uniq), quote


def _format_criteria(rule: dict[str, Any]) -> str:
    parts: list[str] = []
    if rule.get("pass_criteria"):
        parts.append(f"PASS if: {rule['pass_criteria']}")
    if rule.get("warning_criteria"):
        parts.append(f"WARNING if: {rule['warning_criteria']}")
    if rule.get("fail_criteria"):
        parts.append(f"FAIL if: {rule['fail_criteria']}")
    return "\n\n".join(parts)


_GUARANTEE_PATTERNS = [
    ("passes the requirement", "appears to satisfy the requirement"),
    ("passes the requirements", "appears to satisfy the requirements"),
    ("this passes", "this appears to satisfy the check"),
    ("the claim passes", "the claim appears to satisfy the check"),
    ("the schedule passes", "the schedule appears to satisfy the check"),
    ("is compliant", "appears to be compliant"),
    ("is valid", "appears to be valid"),
    ("will succeed", "is likely to succeed"),
    ("will fail", "is likely to fail"),
    ("guarantees", "suggests"),
]


def _hedge(text: str) -> str:
    t = text or ""
    low = t.lower()
    for needle, replacement in _GUARANTEE_PATTERNS:
        if needle in low:
            i = low.find(needle)
            t = t[:i] + replacement + t[i + len(needle):]
            low = t.lower()
    return t


def _parse_date(v: Any) -> date | None:
    if not v:
        return None
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _business_days_between(start: date, end: date) -> int:
    if end < start:
        return 0
    days = 0
    cur = start
    while cur < end:
        cur = cur.fromordinal(cur.toordinal() + 1)
        if cur.weekday() < 5:
            days += 1
    return days


def _derive_summary(mode: str, text: str) -> str:
    """Cheap summary line without an LLM call. The frontend displays it at the top of the checklist."""
    label = {
        "payment_claim_serving": "Draft payment claim",
        "payment_claim_received": "Received payment claim",
        "payment_schedule_giving": "Draft payment schedule",
        "payment_schedule_received": "Received payment schedule",
    }.get(mode, "Document")
    import re
    m = re.search(r"\$\s*[\d,]+(?:\.\d{2})?", text or "")
    if m:
        return f"{label} — amount referenced: {m.group(0).strip()}"
    return label
