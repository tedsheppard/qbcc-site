"""Taxonomy v2 for submission categorisation + the Sonnet 5 prompt.

v2 (2026-07-11): categories reorganised around how adjudication submissions
are actually argued —
  - Jurisdictional subs are grouped by the question they attack (right to a
    progress payment / payment claim validity / payment schedule validity /
    adjudication application validity / adjudicator's powers) rather than a
    flat topic list; the specific point (e.g. "no valid reference date")
    lives in the section heading.
  - New top-level categories: Security, Adjudicator's Fees, Legal Arguments.
  - Every section carries a STANCE: whether the submission argues FOR the
    entitlement/point or AGAINST it — a firm searching "reference date"
    precedents needs to know which side the extract runs.

The taxonomy is versioned: documents record the taxonomy_version they were
classified under, so re-running v2 over v1-classified documents is a cheap
batch backfill (Reclassify button / bulk re-run).
"""

from __future__ import annotations

import json
from typing import Any

TAXONOMY_VERSION = "v2"

STANCES = {"for": "For", "against": "Against", "mixed": "Mixed"}

# slug -> {label, subs: {slug: {label, hint}}}. `hint` steers the model on
# what belongs in the bucket; it is prompt-only and not shown in the UI.
TAXONOMY: dict[str, dict[str, Any]] = {
    "jurisdictional": {
        "label": "Jurisdictional",
        "subs": {
            "right_to_payment": {
                "label": "Right to a progress payment",
                "hint": "reference dates, existence of a construction contract, whether work is construction work / related goods and services, carve-outs, claimant standing, solvency, licensing (s 42 QBCC Act etc.)",
            },
            "payment_claim_validity": {
                "label": "Payment claim validity",
                "hint": "identification of the work, stating the claimed amount, request for payment / endorsement under the Act, timing of the claim, service of the payment claim, one-claim-per-reference-date",
            },
            "payment_schedule_validity": {
                "label": "Payment schedule validity",
                "hint": "whether the schedule identifies the payment claim, states the scheduled amount, includes reasons for withholding, timing of the schedule, service of the payment schedule",
            },
            "adjudication_application_validity": {
                "label": "Adjudication application validity",
                "hint": "timing of the application, service, compliance with the Act, whether the application exceeds the scope of the payment claim, new material",
            },
            "adjudicator_powers": {
                "label": "Adjudicator's powers & procedure",
                "hint": "power to consider reasons not in the payment schedule, natural justice / procedural fairness, pay-when-paid provisions, jurisdictional error, adequacy of reasons",
            },
        },
    },
    "contract_works": {
        "label": "Contract Works",
        "subs": {
            "scope": {"label": "Scope of works", "hint": "what is inside/outside the contracted scope"},
            "valuation": {"label": "Valuation of work", "hint": "measurement, rates, percentage complete, valuation methodology"},
            "defects": {"label": "Defects & set-off", "hint": "defective work, rectification costs, back-charges, set-off against the claim"},
            "completion": {"label": "Completion & quality", "hint": "practical completion, quality standards, incomplete work"},
        },
    },
    "variations": {
        "label": "Variations",
        "subs": {
            "general": {"label": "General entitlement", "hint": "entitlement to be paid for a variation generally"},
            "scope": {"label": "Whether a variation / within scope", "hint": "whether the work was a variation at all or within original scope"},
            "direction": {"label": "Direction & approval", "hint": "whether a direction was given, who gave it, authority, writing requirements"},
            "time_bars": {"label": "Time bars & notices", "hint": "notice preconditions and time bars for variation claims"},
            "quantum": {"label": "Quantum & valuation", "hint": "valuation of the variation, cost build-ups, margins"},
        },
    },
    "eots": {
        "label": "EOTs",
        "subs": {
            "entitlement": {"label": "Entitlement & general", "hint": "entitlement to an extension of time generally, qualifying causes"},
            "time_bars": {"label": "Time bars & notices", "hint": "EOT notice preconditions and time bars"},
            "responsibility": {"label": "Causation & responsibility", "hint": "who caused the delay, critical path, delay analysis"},
            "concurrent": {"label": "Concurrent delay", "hint": "concurrency arguments and apportionment"},
            "delay_costs": {"label": "Delay costs & prolongation", "hint": "prolongation costs, disruption, delay damages quantum"},
        },
    },
    "liquidated_damages": {
        "label": "Liquidated Damages",
        "subs": {
            "general": {"label": "General entitlement", "hint": "entitlement to levy or resist LDs generally, rate, calculation"},
            "penalty": {"label": "Penalty doctrine", "hint": "whether the LD rate is a penalty"},
            "prevention": {"label": "Prevention principle", "hint": "principal-caused delay defeating LDs, time at large"},
            "set_off": {"label": "Set-off & deduction", "hint": "whether LDs can be set off against the claimed amount in the adjudication"},
        },
    },
    "security": {
        "label": "Security",
        "subs": {
            "retention": {"label": "Retention", "hint": "retention moneys, release of retention, claiming retention"},
            "bank_guarantees": {"label": "Bank guarantees", "hint": "bank guarantees / insurance bonds as security"},
            "recourse": {"label": "Recourse to security", "hint": "entitlement to have recourse / calls on security, notice requirements"},
            "release": {"label": "Release of security", "hint": "when security must be released or reduced"},
        },
    },
    "interest": {
        "label": "Interest",
        "subs": {
            "statutory": {"label": "Statutory interest", "hint": "interest under the Act or penalty interest"},
            "contractual": {"label": "Contractual interest", "hint": "interest under the contract"},
            "calculation": {"label": "Rate & calculation", "hint": "applicable rate, compounding, period"},
        },
    },
    "adjudicators_fees": {
        "label": "Adjudicator's Fees",
        "subs": {
            "apportionment": {"label": "Apportionment", "hint": "who should bear the adjudicator's fees and in what proportion"},
            "quantum": {"label": "Quantum", "hint": "reasonableness / amount of the adjudicator's fees"},
        },
    },
    "legal_arguments": {
        "label": "Legal Arguments",
        "subs": {
            "waiver": {"label": "Waiver", "hint": "waiver of rights, requirements or objections"},
            "estoppel": {"label": "Estoppel", "hint": "estoppel by convention, representation, promissory estoppel"},
            "misleading_conduct": {"label": "Misleading or deceptive conduct", "hint": "ACL s 18 and related arguments"},
            "reagitation": {"label": "Re-agitation & repeat claims", "hint": "re-agitating decided issues, repeat claims, issue estoppel in adjudication"},
            "interpretation": {"label": "Contract interpretation", "hint": "construction of contract clauses, implied terms"},
            "misc": {"label": "Other legal arguments", "hint": "any legal argument not covered elsewhere"},
        },
    },
    "other": {
        "label": "Other",
        "subs": {
            "misc": {"label": "Uncategorised", "hint": "anything that genuinely fits nowhere else"},
        },
    },
}


def valid_pair(category: str, subcategory: str) -> tuple[str, str]:
    """Coerce a model-suggested (category, subcategory) onto the taxonomy."""
    cat = (category or "").strip().lower()
    sub = (subcategory or "").strip().lower()
    if cat not in TAXONOMY:
        return "other", "misc"
    if sub not in TAXONOMY[cat]["subs"]:
        first = next(iter(TAXONOMY[cat]["subs"]))
        return cat, first
    return cat, sub


def valid_stance(stance: str) -> str:
    s = (stance or "").strip().lower()
    return s if s in STANCES else ""


def sub_label(category: str, subcategory: str) -> str:
    node = TAXONOMY.get(category, {}).get("subs", {}).get(subcategory)
    return node["label"] if node else subcategory


def taxonomy_public() -> dict[str, Any]:
    return {
        "version": TAXONOMY_VERSION,
        "stances": [{"slug": k, "label": v} for k, v in STANCES.items()],
        "categories": [
            {
                "slug": slug,
                "label": node["label"],
                "subs": [{"slug": s, "label": v["label"]} for s, v in node["subs"].items()],
            }
            for slug, node in TAXONOMY.items()
        ],
    }


_RESULT_SHAPE = {
    "document": {
        "title": "short human title for the document",
        "doc_type": "e.g. adjudication application submissions | adjudication response submissions | reply submissions | payment claim | payment schedule | other",
        "party_side": "claimant | respondent | unknown",
        "act": "BIF | BCIPA | NSW SOPA | other | unknown",
        "date_hint": "the document's date if identifiable, ISO format (YYYY-MM-DD), else empty string",
    },
    "matter": {
        "claimant": "full name of the claimant party, else empty string",
        "respondent": "full name of the respondent party, else empty string",
        "claimant_lawyers": "law firm acting for the claimant if identifiable, else empty string",
        "respondent_lawyers": "law firm acting for the respondent if identifiable, else empty string",
        "claimed_amount": "the claimed amount in AUD as a plain number (no $ or commas), else null",
        "scheduled_amount": "the scheduled amount in AUD as a plain number, else null",
    },
    "sections": [
        {
            "category": "<category slug from the taxonomy>",
            "subcategory": "<subcategory slug from the taxonomy>",
            "stance": "for | against | mixed — see rules",
            "heading": "the specific point, e.g. 'No valid reference date arose under cl 37.1'",
            "summary": "2-3 sentence summary of the submission being made",
            "page_start": 1,
            "page_end": 1,
            "confidence": 0.9,
        }
    ],
}


def build_system_prompt() -> str:
    tax_lines = []
    for slug, node in TAXONOMY.items():
        subs = "; ".join(f"{s} = {v['label']} ({v['hint']})" for s, v in node["subs"].items())
        tax_lines.append(f"- {slug} ({node['label']}): {subs}")
    tax_block = "\n".join(tax_lines)
    return f"""You are an expert Australian construction-law analyst classifying a law firm's precedent document — typically adjudication submissions under the Building Industry Fairness (Security of Payment) Act 2017 (Qld), BCIPA 2004 (Qld), or an interstate SOP Act.

The document is supplied as numbered pages marked [PAGE n]. You must do two jobs:

JOB 1 — MATTER METADATA. From the document itself, identify the claimant, the respondent, the law firm acting for each side (often in headers, footers, signature blocks or service details), the claimed amount and the scheduled amount. Use empty string / null where the document does not reveal a value. Never guess party or firm names.

JOB 2 — SECTION CLASSIFICATION. Identify every DISTINCT submission or argument section and classify it against this fixed taxonomy (use the slugs exactly; taxonomy version {TAXONOMY_VERSION}):

{tax_block}

Rules:
- A section is a contiguous run of pages advancing one identifiable submission (e.g. "the payment claim was not served on a valid reference date"). Long documents commonly contain many sections.
- The specific point goes in `heading`; the taxonomy slugs place it in the right bucket.
- STANCE captures which way the submission cuts on its subject matter: "for" = argues in favour of the entitlement, validity or right in question (e.g. the payment claim IS valid, the variation IS payable, LDs ARE claimable); "against" = argues against it (the claim is INVALID, there is NO entitlement, the security must NOT be called); "mixed" = genuinely runs both ways. Judge stance by the argument itself, not by which party makes it.
- page_start/page_end refer to the [PAGE n] markers and must be within the document.
- Cover the substantive submissions; skip covers, indexes, signature blocks and annexure lists.
- If a section genuinely fits no taxonomy entry, use category "other".
- confidence is 0.0-1.0, your own confidence in the classification.
- Never invent content that is not in the document.

Return ONLY a single JSON object, no markdown fences, no prose, exactly this shape:
{json.dumps(_RESULT_SHAPE, indent=2)}"""
