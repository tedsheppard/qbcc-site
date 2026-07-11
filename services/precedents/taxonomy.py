"""Taxonomy v1 for submission categorisation + the Sonnet 5 prompt.

The taxonomy is versioned: documents record the taxonomy_version they were
classified under, so a future v2 can be re-run as a cheap batch backfill
over documents whose version differs.
"""

from __future__ import annotations

import json
from typing import Any

TAXONOMY_VERSION = "v1"

# slug -> {label, subs: {slug: label}}
TAXONOMY: dict[str, dict[str, Any]] = {
    "jurisdictional": {
        "label": "Jurisdictional",
        "subs": {
            "reference_date": "No valid reference date",
            "invalid_payment_claim": "Invalid payment claim (form / content)",
            "invalid_service": "Invalid or out-of-time service",
            "no_construction_contract": "No construction contract / carve-outs",
            "out_of_time": "Application out of time",
            "new_reasons": "New reasons barred (s 82(4) / s 24(4))",
            "natural_justice": "Natural justice / procedural fairness",
            "adjudicator_jurisdiction": "Adjudicator jurisdiction (other)",
        },
    },
    "payment_process": {
        "label": "Payment process",
        "subs": {
            "payment_claim_validity": "Payment claim validity (supporting)",
            "payment_schedule_adequacy": "Payment schedule adequacy / reasons",
            "due_date": "Due date for payment",
            "reference_date_args": "Reference date arguments (merits)",
        },
    },
    "variations": {
        "label": "Variations",
        "subs": {
            "directed": "Directed variations",
            "constructive": "Constructive / implied variations",
            "latent_conditions": "Latent conditions",
            "omissions": "Omissions / descoping",
            "valuation": "Valuation of variations",
            "time_bars": "Variation time bars / notice compliance",
        },
    },
    "delay_eot": {
        "label": "Delay / EOT",
        "subs": {
            "eot_entitlement": "EOT entitlement",
            "causation": "Delay events & causation",
            "concurrent_delay": "Concurrent delay",
            "prolongation": "Prolongation costs",
            "disruption": "Disruption / loss of productivity",
            "time_bars": "Delay time bars / notices",
        },
    },
    "liquidated_damages": {
        "label": "Liquidated damages",
        "subs": {
            "entitlement": "LD entitlement & rate",
            "penalties": "Penalties doctrine",
            "prevention": "Prevention principle",
            "set_off": "Set-off of LDs",
        },
    },
    "contract_works": {
        "label": "Contract works",
        "subs": {
            "valuation": "Valuation of works / scope",
            "defects": "Defective works / set-off",
            "completion": "Quality / completion disputes",
            "retention_security": "Retention & security",
        },
    },
    "interest": {
        "label": "Interest",
        "subs": {
            "statutory": "Statutory interest",
            "contractual": "Contractual interest",
        },
    },
    "fees_costs": {
        "label": "Fees & costs",
        "subs": {
            "adjudicator_fees": "Adjudicator fees apportionment",
            "legal_costs": "Legal costs",
        },
    },
    "other": {
        "label": "Other",
        "subs": {
            "estoppel_conduct": "Estoppel / misleading conduct",
            "interpretation": "Contract interpretation",
            "misc": "Miscellaneous",
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


def taxonomy_public() -> dict[str, Any]:
    return {
        "version": TAXONOMY_VERSION,
        "categories": [
            {
                "slug": slug,
                "label": node["label"],
                "subs": [{"slug": s, "label": l} for s, l in node["subs"].items()],
            }
            for slug, node in TAXONOMY.items()
        ],
    }


_RESULT_SHAPE = {
    "document": {
        "title": "short human title for the document",
        "doc_type": "e.g. adjudication application submissions | adjudication response submissions | payment claim | payment schedule | other",
        "party_side": "claimant | respondent | unknown",
        "act": "BIF | BCIPA | NSW SOPA | other | unknown",
        "date_hint": "any document date you can identify, ISO format, else empty string",
    },
    "sections": [
        {
            "category": "<category slug from the taxonomy>",
            "subcategory": "<subcategory slug from the taxonomy>",
            "heading": "the heading or a short label for this part of the document",
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
        subs = ", ".join(f"{s} ({l})" for s, l in node["subs"].items())
        tax_lines.append(f"- {slug} ({node['label']}): {subs}")
    tax_block = "\n".join(tax_lines)
    return f"""You are an expert Australian construction-law analyst classifying a law firm's precedent document — typically adjudication submissions under the Building Industry Fairness (Security of Payment) Act 2017 (Qld), BCIPA 2004 (Qld), or an interstate SOP Act.

The document is supplied as numbered pages marked [PAGE n]. Identify every DISTINCT submission or argument section in the document and classify each against this fixed taxonomy (use the slugs exactly; taxonomy version {TAXONOMY_VERSION}):

{tax_block}

Rules:
- A section is a contiguous run of pages advancing one identifiable submission or argument (e.g. "the payment claim was not served on a valid reference date"). Long documents commonly contain many sections.
- page_start/page_end refer to the [PAGE n] markers and must be within the document.
- Cover the substantive submissions; skip covers, indexes, signature blocks and annexure lists.
- If a section genuinely fits no taxonomy entry, use category "other".
- confidence is 0.0-1.0, your own confidence in the classification.
- Never invent content that is not in the document.

Return ONLY a single JSON object, no markdown fences, no prose, exactly this shape:
{json.dumps(_RESULT_SHAPE, indent=2)}"""
