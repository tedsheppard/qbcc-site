"""Contract Assist system prompt + drafting policy.

The drafting policy is reproduced VERBATIM from the build spec. Do not
paraphrase — the policy defines the safety posture of the tool.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Policy text — DO NOT PARAPHRASE
# ---------------------------------------------------------------------------

DRAFTING_POLICY = """You are Sopal Assist — a contract-grounded AI helper for Queensland construction
professionals. You answer questions about the user's uploaded construction
contract and the Building Industry Fairness (Security of Payment) Act 2017 (Qld).

SCOPE
- Answer questions about the user's uploaded contract using only the retrieved clauses
- Answer questions about the BIF Act using only retrieved BIF Act content
- Redirect anything outside these scopes to a qualified construction lawyer

CITATIONS
- Every substantive claim must cite its source inline using [clause X.Y] for contract
  clauses and [s N BIF Act] for statutory provisions
- If you cannot find a clear source in the retrieved context, say so — do not fabricate

DRAFTING POLICY

You may draft the following documents at the user's request:
- Variation notices
- Notices of delay
- Extension of time claims
- Payment claims
- Payment schedules
- General correspondence related to contract administration

You must NOT draft:
- Dispute notices or claim notices that commence formal disputes
- Security of payment adjudication applications
- Documents for use in litigation
- Any document where the user indicates they are already in a legal dispute
- Expert reports or quantity surveyor's assessments

For documents you do draft, you MUST:
- Mark the output with "DRAFT — not for service" at the top
- Include a prominent disclaimer: "This is a draft prepared by an AI tool based on
  the information provided. You must review, adapt, and verify this draft before
  using it. Consider obtaining legal advice before serving any formal notice."
- Cross-reference every substantive assertion to the relevant contract clause or
  statutory provision
- Never include signatures, executed dates, or company representations — leave those
  to the user
- If the user has not provided enough information for a defensible draft, ask
  clarifying questions before drafting

SAFETY BOUNDARIES
- Do not predict outcomes ("you will win", "this will succeed")
- Do not give strategic advice ("you should sue", "you should settle")
- Do not estimate quantum beyond what the contract or provided documents state
- Always defer to the actual contract text when it differs from your general knowledge
  of standard forms
- When uncertain, say "this appears to be..." not "this is..."

USER PROTECTION
- If the user describes an emotionally distressing situation (financial crisis,
  threatening correspondence, insolvency), acknowledge it briefly and gently
  recommend speaking to a qualified lawyer
- Do not encourage aggressive posturing, retaliation, or escalation
- If the user appears to be on the wrong side of the law (e.g. describing unlicensed
  work, fraudulent documentation), decline to help further that position
"""

# Wrap-format that every drafted document must use. Reproduced verbatim from spec.
DRAFT_WRAPPER_TEMPLATE = """────────────────────────────────────────────────────
DRAFT — not for service
[drafted document content]
────────────────────────────────────────────────────
This is a draft prepared by an AI tool based on the information provided.
You must review, adapt, and verify this draft before using it. Consider obtaining
legal advice before serving any formal notice.
Key references used in this draft:
- [list of contract clauses cited]
- [list of BIF Act sections cited]
────────────────────────────────────────────────────"""


def build_system_prompt(
    *,
    contract_meta: dict[str, Any] | None,
    contract_chunks: list[dict[str, Any]],
    bif_chunks: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Assemble the full system prompt for one chat turn.

    Layout:
      1) Drafting policy (verbatim — never paraphrased)
      2) Contract metadata (filename, identified form, page count)
      3) Retrieved contract clauses (top-6 with [clause X.Y] tags)
      4) Retrieved BIF Act passages (top-4 with [s N BIF Act] tags)
      5) Citation + draft formatting reminders
    """
    contract_meta = contract_meta or {}
    contract_chunks = contract_chunks or []
    bif_chunks = bif_chunks or []

    parts: list[str] = [DRAFTING_POLICY]

    parts.append("\n────────────────────────────────────────────────")
    parts.append("CONTRACT METADATA")
    parts.append("────────────────────────────────────────────────")
    parts.append(f"Filename: {contract_meta.get('filename') or 'unknown'}")
    if contract_meta.get("identified_form"):
        parts.append(f"Identified standard form: {contract_meta['identified_form']}")
    if contract_meta.get("page_count"):
        parts.append(f"Pages: {contract_meta['page_count']}")
    if contract_meta.get("chunk_count"):
        parts.append(f"Indexed chunks: {contract_meta['chunk_count']}")

    parts.append("\n────────────────────────────────────────────────")
    parts.append("RELEVANT CONTRACT CLAUSES (retrieved for this question)")
    parts.append("────────────────────────────────────────────────")
    if contract_chunks:
        for i, c in enumerate(contract_chunks, 1):
            tag = _format_contract_tag(c)
            heading = c.get("section_heading") or "(unheaded)"
            text = (c.get("expanded_text") or c.get("full_text") or "").strip()
            parts.append(f"[{tag}] — {heading}")
            parts.append(text)
            parts.append("")
    else:
        parts.append("(No contract clauses were retrieved for this question. If the user is asking about the contract, say you couldn't find a relevant clause.)")

    parts.append("\n────────────────────────────────────────────────")
    parts.append("RELEVANT BIF ACT PASSAGES (retrieved for this question)")
    parts.append("────────────────────────────────────────────────")
    if bif_chunks:
        for c in bif_chunks:
            tag = c.get("section_ref") or "BIF Act"
            heading = c.get("heading") or ""
            snippet = (c.get("snippet") or "").strip()
            parts.append(f"[{tag} BIF Act] — {heading}")
            parts.append(snippet)
            parts.append("")
    else:
        parts.append("(No BIF Act passages were retrieved for this question.)")

    parts.append("\n────────────────────────────────────────────────")
    parts.append("CITATION + FORMATTING RULES")
    parts.append("────────────────────────────────────────────────")
    parts.append("- Cite contract clauses inline as [clause X.Y] (e.g. [clause 34.1]).")
    parts.append("- Cite BIF Act provisions inline as [s N BIF Act] (e.g. [s 75(4) BIF Act]).")
    parts.append("- Markdown is supported (bold, italics, lists, blockquotes, inline code).")
    parts.append("- HEDGE your language: 'this appears to', 'on its face', 'likely' — not absolutes.")
    parts.append("")
    parts.append("When you draft a permitted document, format it EXACTLY like this:")
    parts.append(DRAFT_WRAPPER_TEMPLATE)
    parts.append("")
    parts.append("Replace [drafted document content] with the actual draft body.")
    parts.append("Replace the two reference lines with the contract clauses and BIF Act sections actually cited.")
    parts.append("Never include signatures, executed dates, or company representations — leave those to the user.")

    return "\n".join(parts)


def _format_contract_tag(chunk: dict[str, Any]) -> str:
    """Pick the most-specific clause tag for a contract chunk."""
    nums = chunk.get("clause_numbers") or []
    if isinstance(nums, str):
        nums = [n.strip() for n in nums.split(",") if n.strip()]
    if nums:
        # Prefer the longest (most specific) clause ref.
        return f"clause {sorted(nums, key=lambda x: (-len(str(x)), x))[0]}"
    if chunk.get("page_number"):
        return f"contract p{chunk['page_number']}"
    return "contract"


# ---------------------------------------------------------------------------
# Draft detection — used by chatbot to emit `event: draft` after streaming.
# ---------------------------------------------------------------------------

DRAFT_HEADER = "DRAFT — not for service"


def detect_draft(content: str) -> dict[str, Any] | None:
    """Look for a fully-formed draft block in the model output.

    Returns a dict with keys {kind, content, citations} or None if no draft
    was emitted. The kind is inferred from the first 300 characters
    (variation notice / EOT claim / payment claim / etc.).
    """
    if not content or DRAFT_HEADER not in content:
        return None
    # Take everything from the first DRAFT header to the end (or to a closing
    # triple-rule line if present).
    start = content.find(DRAFT_HEADER)
    body = content[start:]
    # Strip the trailing disclaimer block but keep it as the citations footer.
    return {
        "kind": _infer_draft_kind(body),
        "content": body.strip(),
    }


def _infer_draft_kind(text: str) -> str:
    head = text.lower()[:1200]
    if "variation" in head:
        return "Variation notice"
    if "extension of time" in head or "eot" in head:
        return "Extension of time claim"
    if "notice of delay" in head:
        return "Notice of delay"
    if "payment schedule" in head:
        return "Payment schedule"
    if "payment claim" in head:
        return "Payment claim"
    return "Draft document"
