"""Explanatory chatbot for /claim-check (Claim Assist).

The chatbot is a strict interpreter of the rule engine's analysis. It is NOT
an independent legal analyst. Every response must be grounded in the four
authoritative sources injected into the system prompt:

  1. The analysis state (rule engine output) for this document
  2. rules/bif_act_rules.md (re-read at request time)
  3. rules/bif_act_annotations.md (re-read at request time, section excerpts)
  4. The user's uploaded document text

This file used to permit free-form legal reasoning; that licence produced
output that contradicted the rule engine and fabricated case-law-flavoured
fixes. The current architecture removes that licence at the prompt level
and at the context level (by feeding the model the engine's reasoning,
quotes, and citations directly so it has nothing to fall back to).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from . import annotations as annotations_mod
from . import llm_config

log = logging.getLogger("claim_check.chatbot")

# ---------------------------------------------------------------------------
# Token / size budget
# ---------------------------------------------------------------------------

# Soft target for the assembled prompt. gpt-5.4-mini's context is generous,
# but we keep the prompt deliberately compact so the model focuses on the
# authoritative sources rather than chasing scattered context.
MAX_PROMPT_CHARS = 110_000  # ≈ 28K tokens at 4 chars/token

# Per-piece allocations. These are upper bounds; the actual prompt is usually
# smaller. Truncation order, per spec:
#   (i) keep the system prompt in full
#   (ii) keep all check results in full
#   (iii) keep relevant v29 annotation sections in full
#   (iv) keep rules/bif_act_rules.md in full
#   (v) truncate the user's document text from the middle if needed
MAX_RULES_CHARS = 24_000           # rules.md is ~12 KB today; cap is generous
MAX_ANNOTATION_PER_SECTION = 18_000  # ≈ 4.5K tokens per relevant section
MAX_DOC_HEAD_CHARS = 30_000
MAX_DOC_TAIL_CHARS = 20_000
MAX_HISTORY_TURNS = 8
MAX_OUTPUT_TOKENS = 1500

MODE_LABELS = {
    "payment_claim_serving":    "a payment claim the user is about to serve",
    "payment_claim_received":   "a payment claim the user has received",
    "payment_schedule_giving":  "a payment schedule the user is about to give",
    "payment_schedule_received": "a payment schedule the user has received",
}


# ---------------------------------------------------------------------------
# The system prompt — verbatim per spec, no paraphrasing
# ---------------------------------------------------------------------------

CHATBOT_SYSTEM_PROMPT = """You are Sopal Claim Assist's explanatory chatbot. Your role is narrow: you explain and elaborate on an analysis that has already been performed by Sopal's rule engine. You do not perform independent legal analysis.

AUTHORITATIVE SOURCES — IN ORDER OF PRECEDENCE

The analysis state provided in this prompt. The rule engine has evaluated specific checks against the user's document. Those evaluations are authoritative. You must not contradict them.
rules/bif_act_rules.md — the codified rules the engine applied. Provided in this prompt.
rules/bif_act_annotations.md — the user's annotated BIF Act commentary, including authority on s 68, s 70, s 75, s 76. Provided in this prompt.
The user's uploaded document text. Provided in this prompt.

You may not draw on legal knowledge from your general training that is not present in sources 1-4. If a question requires reasoning beyond these sources, decline and recommend the user speak with a construction lawyer.
WHEN THE USER ASKS ABOUT A REQUIREMENT THE RULE ENGINE EVALUATED
Look up the corresponding check in the analysis state. Reference its status, summary, and reasoning. Explain the rule engine's conclusion in plain language. If the user asks "does it identify the construction work?" and the rule engine flagged that as a warning, your answer must reflect that warning — not contradict it. Quote the rule engine's reasoning where helpful.
You must NEVER answer "yes it satisfies X" when the rule engine has flagged a warning or failure on X. You must NEVER suggest fixes that go beyond what rules/bif_act_rules.md or rules/bif_act_annotations.md actually require. You must NEVER fabricate suggested wording, drafting language, or claim improvements not grounded in the rules or annotations.
WHEN THE USER ASKS SOMETHING THE RULE ENGINE DID NOT EVALUATE
Reason only from rules/bif_act_rules.md and rules/bif_act_annotations.md. If those sources do not address the question directly, say so and recommend speaking with a construction lawyer. Do not extrapolate. Do not infer general legal principles from your training data.
WHEN THE USER ASKS HOW TO IMPROVE A CLAIM OR DOCUMENT
You may only suggest improvements that are explicitly grounded in:

A specific rule in rules/bif_act_rules.md, or
A specific passage in rules/bif_act_annotations.md, or
A specific case authority cited in either.

You must quote or cite the source of any suggested improvement. If you cannot ground a suggestion in those sources, do not make it.
CITATION DISCIPLINE
Every legal claim must cite its source inline using:

[s N(M) BIF Act] for statutory references
[rule {ID} from rules/bif_act_rules.md] when referencing a specific rule the engine applied
[v29 annotation on s N] when referencing the annotated commentary
Case names with citations (e.g., Luikens v Multiplex [2003] NSWSC 1140) only when those cases appear in the rules or annotations

If you cannot cite, you cannot assert. Say "I do not have authority for that in the sources available to me" instead of guessing.
CONTRADICTION CHECK BEFORE EVERY RESPONSE
Before sending any response, check whether your response contradicts the rule engine's analysis. If it does, you must revise. The rule engine's findings are the ceiling of your conclusions — you can explain them, soften them with plain-language framing, or note user uncertainty, but you cannot reach a conclusion more favourable to the user than the rule engine reached.
SCOPE
You answer questions about: the user's uploaded document, the rule engine's analysis of it, the BIF Act sections covered by the analysis, and Queensland security of payment law as covered by the rules and annotations.
You do not answer: general legal questions, questions about other jurisdictions, questions about strategy or litigation, questions about contract drafting beyond what the rules cover, or any question whose answer requires legal training data outside the four authoritative sources.
DISCLAIMER
End every response that touches anything specific with: "General information only — not legal advice. For your specific situation, consult a construction lawyer."
"""


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Match "s 68", "s 68(1)", "s 68(1)(a)", "section 68" — captures the leading int.
_SECTION_RE = re.compile(r"\bs(?:ection)?\s*(\d{1,3})\b", re.IGNORECASE)


def _sections_from_check(check: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for field in ("section", "section_ref", "act_reference"):
        v = check.get(field) or ""
        for m in _SECTION_RE.finditer(str(v)):
            try:
                out.add(int(m.group(1)))
            except ValueError:
                continue
    return out


def _sections_from_text(text: str) -> set[int]:
    out: set[int] = set()
    for m in _SECTION_RE.finditer(text or ""):
        try:
            out.add(int(m.group(1)))
        except ValueError:
            continue
    return out


def _relevant_section_set(checks: list[dict[str, Any]], user_message: str) -> list[int]:
    """Pick which annotation sections to inject.

    Per spec: include any section referenced by a check OR mentioned in the
    user's message. Plus an always-on rule: if any s 68 check is present
    (whatever sub-paragraph), include the s 68 annotation — that's where the
    Luikens / KDV / Bridgeman authority lives, and it must never be missed.
    """
    sections: set[int] = set()
    has_any_s68_check = False
    for c in checks or []:
        secs = _sections_from_check(c)
        sections.update(secs)
        if 68 in secs:
            has_any_s68_check = True
    sections.update(_sections_from_text(user_message or ""))
    if has_any_s68_check:
        sections.add(68)  # belt-and-braces — already added above, but explicit
    # Order: priority sections (the ones with most BIF Act commentary) first.
    priority = [68, 75, 76, 70]
    ordered = [s for s in priority if s in sections]
    extras = sorted(s for s in sections if s not in priority)
    return ordered + extras


# ---------------------------------------------------------------------------
# Authoritative source rendering
# ---------------------------------------------------------------------------

def _render_check_results(checks: list[dict[str, Any]], user_answers: dict[str, Any]) -> str:
    """Render the analysis state as the model will see it. Keep ALL substantive
    fields per check — status, summary, reasoning_trace, quote, citations,
    confidence — so the chatbot has the engine's actual reasoning, not just
    its conclusion."""
    if not checks:
        return "(No analysis has been run yet on this document. The user has not produced any check results to ground answers in.)"

    lines: list[str] = []
    for i, c in enumerate(checks, 1):
        cid = str(c.get("id") or f"check-{i}")
        title = str(c.get("title") or "(untitled)")
        section = str(c.get("section") or "")
        status = str(c.get("status") or "?").upper()
        status_summary = str(c.get("status_summary") or "")
        explanation = str(c.get("explanation") or "")
        reasoning = str(c.get("reasoning_trace") or "")
        quote = str(c.get("quote") or "").strip()
        confidence = str(c.get("confidence") or "")
        decisions = c.get("decisions") or []
        input_questions = c.get("input_questions") or []

        lines.append(f"### {cid}: {title}")
        lines.append(f"  Section: {section}")
        lines.append(f"  Status: {status}")
        if status_summary:
            lines.append(f"  Status summary: {status_summary}")
        if explanation:
            lines.append(f"  Engine explanation (shown in UI): {explanation}")
        if reasoning:
            # Multi-line reasoning trace — indent for readability.
            indent = "    "
            wrapped = "\n".join(indent + ln for ln in reasoning.splitlines())
            lines.append(f"  Engine reasoning trace (the chain-of-reasoning the engine used):\n{wrapped}")
        if quote:
            lines.append(f'  Document quote relied on: "{quote}"')
        if confidence:
            lines.append(f"  Engine confidence: {confidence}")
        if decisions:
            for d in decisions[:3]:
                t = (d or {}).get("title") or ""
                if t:
                    lines.append(f"  Engine surfaced citation: {t}")

        # If this check has any user answers, include them so the chatbot can
        # ground answers like "based on what you told me, the reference date
        # is 31 March".
        relevant_answers: list[tuple[str, Any]] = []
        for q in input_questions:
            qid = (q or {}).get("id")
            if qid and qid in (user_answers or {}):
                v = user_answers[qid]
                if v not in (None, "", []):
                    relevant_answers.append((q.get("question") or qid, v))
        if relevant_answers:
            lines.append("  User-provided answers used by this check:")
            for q_text, v in relevant_answers:
                lines.append(f"    Q: {q_text}")
                if isinstance(v, dict):
                    # Likely a licensee_lookup record — render compactly.
                    display = v.get("display") or v.get("entity_name") or "(unnamed)"
                    extras = []
                    if v.get("licence_number"): extras.append(f"licence {v['licence_number']}")
                    if v.get("licence_status"): extras.append(v["licence_status"])
                    if v.get("licence_classes"): extras.append(", ".join(v["licence_classes"][:3]))
                    extra_str = (" — " + " · ".join(extras)) if extras else ""
                    lines.append(f"    A: {display}{extra_str}")
                else:
                    lines.append(f"    A: {v}")
        lines.append("")

    return "\n".join(lines)


def _read_rules_md() -> str:
    """Re-read rules/bif_act_rules.md at request time. No caching."""
    p = Path(__file__).resolve().parents[2] / "rules" / "bif_act_rules.md"
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        log.warning("rules.md read failed: %s", e)
        return "(rules/bif_act_rules.md unavailable)"


def _build_annotation_section(section_num: int, max_chars: int, *, keyword_hint: str | None = None) -> str:
    """Pull the v29 annotation for a single section with a budget."""
    body = annotations_mod.load_annotations().get(section_num)
    if not body:
        return ""
    if len(body) <= max_chars:
        return body
    if keyword_hint:
        # Window around the first hit of the keyword; fall back to head if not found.
        pos = body.lower().find(keyword_hint.lower())
        if pos != -1:
            half = max_chars // 2
            start = max(0, pos - half)
            end = min(len(body), start + max_chars)
            snippet = body[start:end].strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(body):
                snippet = snippet + "…"
            return snippet
    # No hint or hint not found — head excerpt.
    return body[:max_chars].rsplit("\n\n", 1)[0] + "\n…"


def _truncate_doc_text_middle(doc: str, head: int, tail: int) -> str:
    """Keep the start (where identification language usually sits) and the
    end (where claimed amount + signature live). Truncate the middle."""
    if not doc:
        return ""
    if len(doc) <= head + tail:
        return doc
    return (
        doc[:head]
        + f"\n\n[…document middle truncated, {len(doc) - head - tail:,} chars omitted…]\n\n"
        + doc[-tail:]
    )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _build_full_prompt(
    *,
    mode: str,
    document_text: str,
    check_results: list[dict[str, Any]],
    user_answers: dict[str, Any],
    user_message: str,
) -> str:
    mode_label = MODE_LABELS.get(mode, "a payment claim or payment schedule")

    # 1. Render the analysis state.
    analysis_block = _render_check_results(check_results or [], user_answers or {})

    # 2. Read rules.md fresh at request time.
    rules_md = _read_rules_md()
    if len(rules_md) > MAX_RULES_CHARS:
        rules_md = rules_md[:MAX_RULES_CHARS] + "\n\n[…rules.md truncated for prompt budget…]"

    # 3. Pick relevant annotation sections + render each.
    section_nums = _relevant_section_set(check_results or [], user_message or "")
    annotation_blocks: list[str] = []
    for s in section_nums:
        # Keyword hints sharpen the window when the section is too big.
        hint = None
        if s == 68:
            hint = "KDV Sport"  # ensures the Luikens/KDV/Bridgeman discussion is in the window
        elif s == 75:
            hint = "75(4)"
        elif s == 76:
            hint = "76(2)"
        block = _build_annotation_section(s, MAX_ANNOTATION_PER_SECTION, keyword_hint=hint)
        if block:
            annotation_blocks.append(f"### v29 annotation on s {s}\n\n{block}")
    annotations_block = "\n\n".join(annotation_blocks) if annotation_blocks else "(No v29 annotation sections were selected for this question.)"

    # 4. Document text — stays last so we can truncate from the middle if the
    #    overall prompt exceeds budget.
    doc = (document_text or "").strip()

    # Assemble.
    def _assemble(doc_section: str) -> str:
        return f"""{CHATBOT_SYSTEM_PROMPT}

────────────────────────────────────────────────
CONTEXT — {mode_label}
────────────────────────────────────────────────

[Source 1 of 4 — ANALYSIS STATE]
The rule engine has produced the following per-check analysis on this document.
These conclusions are authoritative. You may not reach a conclusion more favourable
to the user than the rule engine reached on any check.

{analysis_block}

[Source 2 of 4 — rules/bif_act_rules.md]
The codified rules the engine applied. Cite specific rules using
[rule <ID> from rules/bif_act_rules.md].

{rules_md}

[Source 3 of 4 — rules/bif_act_annotations.md (v29 annotated BIF Act, relevant sections)]
The user's annotated BIF Act commentary. Cite specific passages using
[v29 annotation on s N]. Case authority is grounded in these annotations only.

{annotations_block}

[Source 4 of 4 — USER'S UPLOADED DOCUMENT]
The text the rule engine analysed. Quote it directly when referring to the
document; do not paraphrase quotes.

---
{doc_section}
---
"""

    # First pass: full document.
    candidate = _assemble(doc)
    if len(candidate) <= MAX_PROMPT_CHARS:
        return candidate

    # Truncate the document text from the middle (preserve start and end).
    over = len(candidate) - MAX_PROMPT_CHARS
    # Compute how much to keep of doc such that the new candidate fits.
    target_doc_len = max(2000, len(doc) - over - 200)  # 200 char buffer for the truncation marker
    # Split into head + tail.
    head_share = min(MAX_DOC_HEAD_CHARS, target_doc_len * 3 // 5)
    tail_share = max(target_doc_len - head_share, 0)
    truncated_doc = _truncate_doc_text_middle(doc, head_share, tail_share)
    candidate = _assemble(truncated_doc)
    if len(candidate) <= MAX_PROMPT_CHARS:
        return candidate

    # If still over (extreme cases), trim the annotations block evenly.
    log.warning("Chat prompt exceeds budget after doc truncation (%d chars). Trimming annotations.", len(candidate))
    if annotation_blocks:
        per_section_cap = max(4_000, MAX_ANNOTATION_PER_SECTION // 2)
        slim_annotations: list[str] = []
        for s in section_nums:
            block = _build_annotation_section(s, per_section_cap, keyword_hint="KDV Sport" if s == 68 else None)
            if block:
                slim_annotations.append(f"### v29 annotation on s {s}\n\n{block}")
        annotations_block = "\n\n".join(slim_annotations) if slim_annotations else "(No v29 annotation sections fit the prompt budget.)"
        candidate = _assemble(truncated_doc)
    return candidate[:MAX_PROMPT_CHARS]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chat(
    mode: str,
    document_text: str,
    check_results: list[dict],
    history: list[dict],
    user_message: str,
    *,
    user_answers: dict[str, Any] | None = None,
) -> str:
    if not (user_message or "").strip():
        raise ValueError("Empty message.")

    user_answers = user_answers or {}
    reasoning = llm_config.reasoning_for_chat(user_message)

    system_content = _build_full_prompt(
        mode=mode,
        document_text=document_text,
        check_results=check_results,
        user_answers=user_answers,
        user_message=user_message,
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]

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
