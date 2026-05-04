"""Knowledge-augmented query planner.

The planner is the brain of the upgraded system. Given a user question,
it draws on the model's prior legal knowledge to:

  1. Detect intent (statutory / case_law / procedural / definitional / general)
  2. Name the provisions that are likely controlling (e.g. "BIF Act s 75")
  3. Name the leading Queensland authorities on point (e.g. "MWB Everton
     Park v Devcon [2024] QCA 94")
  4. Reformulate the question into 1-3 retrieval queries that match how
     the answer would actually be written in statute or judgment text
  5. Flag uncertainty (is_fringe / confidence) so the answerer knows
     whether to write a settled-law answer or a structured "frame the
     law" answer for unsettled territory

The retriever then resolves named provisions/authorities deterministically
against the name index (no scoring noise), and merges those chunks with
the hybrid retrieval pool. The answerer composes from the retrieved
chunks only — the planner's knowledge is steering retrieval, not
authoring claims, so chunk-level no-hallucination guarantees are
preserved.

Two modes:
  - real=False -> heuristic baseline (intent + raw query, no LLM call)
  - real=True  -> Claude Opus 4.7 (falls back to GPT chain if unavailable)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from . import llm_config

log = logging.getLogger("bif_research.planner")


@dataclass
class Plan:
    intent: str = "general"
    queries: list[str] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    named_provisions: list[str] = field(default_factory=list)
    named_authorities: list[str] = field(default_factory=list)
    confidence: str = "medium"   # "high" | "medium" | "low"
    is_fringe: bool = False
    notes: str = ""              # planner's free-text reasoning trace
    raw: dict = field(default_factory=dict)


# Heuristic intent fallback (when LLM call fails)
INTENT_PATTERNS = [
    ("statutory",    re.compile(r"\b(test|requirement|elements?|valid|when is|under (the )?act|under section)\b", re.I)),
    ("definitional", re.compile(r"\b(define|definition|what is|what does|meaning of|means)\b", re.I)),
    ("procedural",   re.compile(r"\b(time limit|deadline|how long|within|days|business days|when must|service|serve)\b", re.I)),
    ("case_law",     re.compile(r"\b(courts? have|case law|leading authority|how have|treated|applied)\b", re.I)),
]


def _heuristic_intent(question: str) -> str:
    for intent, pat in INTENT_PATTERNS:
        if pat.search(question):
            return intent
    return "general"


PLAN_SYSTEM_PROMPT = """You are a senior Queensland construction lawyer
acting as the *planner* for a legal research engine. You have read the
Building Industry Fairness (Security of Payment) Act 2017 (Qld), the
Queensland Building and Construction Commission Act 1991, both
regulations, and the body of Queensland appellate and Supreme Court
decisions interpreting those Acts. You also know the leading interstate
authorities (Brodyn, Chase Oyster Bar, Probuild, Southern Han, John
Holland, Bitannia etc) that Queensland courts treat as persuasive.

Your job is NOT to answer the user's question. Your job is to *plan*
retrieval. Specifically:

  1. Detect the user's intent.
  2. NAME the statutory or regulatory provisions most likely to be
     controlling — the specific sections a senior practitioner would
     open first. Be over-inclusive: if the question is about payment
     claims, name s 68 (form), s 75 (timing), s 76 (response), s 77
     (entitlement on no schedule), s 79 (adjudication), s 100 (debt)
     even if the user asked about only one of those facets.
  3. NAME the leading Queensland authorities directly on point. Prefer
     binding QCA decisions, then QSC. Include the citation if you
     remember it (e.g. "MWB Everton Park v Devcon [2024] QCA 94"). If
     a leading interstate authority would be the first thing a Qld
     judge would cite, name it too — Qld retrieval will surface it via
     the second channel.
  4. Reformulate the user's question into 1-3 retrieval queries that
     would match the wording of statute or a judgment headnote on
     point. Keep them short (under 20 words) and use the actual
     statutory or judicial vocabulary.
  5. Set `confidence`:
       - "high"   = you are sure the named provisions and authorities
                    are the right ones; this is settled law.
       - "medium" = there's a reasonable list but some judgment calls;
                    other authorities might be relevant.
       - "low"    = the question is genuinely on the fringe and the
                    leading-authority list is your best guess.
  6. Set `is_fringe = true` if the question is unsettled, novel, or
     turns on facts the cases haven't squarely addressed. False if
     it's a settled question (most BIF Act mechanics are settled).
  7. In `notes`, write 1-3 sentences explaining your reasoning — what
     the question is really asking, which provisions/cases speak to
     it, and any pivotal authority you'd lead with.

Return STRICT JSON only, with exactly these top-level keys:

{
  "intent": "statutory" | "case_law" | "procedural" | "definitional" | "general",
  "queries": [string, ...],            // 1-3 retrieval reformulations
  "named_provisions": [string, ...],   // e.g. ["BIF Act s 75", "BIF Act s 100"]
  "named_authorities": [string, ...],  // e.g. ["MWB Everton Park v Devcon [2024] QCA 94"]
  "entities": {
    "section_refs": [string, ...],
    "case_names":  [string, ...],
    "concepts":    [string, ...]
  },
  "confidence": "high" | "medium" | "low",
  "is_fringe": true | false,
  "notes": string
}

Output ONLY the JSON object. No prose, no code fences.

EXAMPLES (study these patterns):

Q: "Is it sufficient to request payment by stating 'this is a progress claim under the BIF Act'?"
Plan:
  intent: case_law
  named_provisions: ["BIF Act s 68", "BIF Act s 68(1)(c)", "BIF Act s 68(3)"]
  named_authorities: ["MWB Everton Park Pty Ltd v Devcon Building Co Pty Ltd [2024] QCA 94", "Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering [2007] QSC 333"]
  notes: This is the request-for-payment sub-issue under s 68(1)(c), not the identification-of-work sub-issue under s 68(1)(a). MWB v Devcon is the leading QCA authority: a bare statement that the document is made under the Act is not sufficient; something amounting to a request for payment must be express or necessarily and clearly implied. Minimax is the source of the "no particular form of words is necessary" formulation MWB applies. Distinguish from KDV Sport / Buckley which are about identification of work, not request for payment.

Q: "How have Queensland courts treated the requirement that a payment claim identify the construction work?"
Plan:
  intent: case_law
  named_provisions: ["BIF Act s 68(1)(a)"]
  named_authorities: ["KDV Sport Pty Ltd v Muggeridge Constructions [2019] QSC 178", "T & M Buckley P/L v 57 Moss Rd P/L [2010] QCA 381", "Neumann Contractors Pty Ltd v Peet Beachton Syndicate"]
  notes: This is the identification-of-work sub-issue under s 68(1)(a). The test is objective and practical — the work must be identified sufficiently for the recipient to understand what the claim is for, taking into account the parties' background knowledge. Distinguish from MWB v Devcon which is about request for payment, not identification.

Q: "When must a respondent give a payment schedule under the BIF Act?"
Plan:
  intent: statutory
  named_provisions: ["BIF Act s 76", "BIF Act s 77", "BIF Act s 78"]
  named_authorities: []
  notes: Pure statutory question — s 76 sets the timing (earlier of contractual period or 15 business days). s 77 and s 78 are the consequence provisions if the schedule isn't given.

Q: "Can a contractor bank unused reference dates after termination?"
Plan:
  intent: case_law
  named_provisions: ["BIF Act s 67", "BIF Act s 70"]
  named_authorities: ["Southern Han Breakfast Point Pty Ltd v Lewence Construction Pty Ltd [2016] HCA 52", "Parrwood Pty Ltd v Trinity Constructions (Aust) Pty Ltd [2020] QSC 211"]
  notes: Contested doctrine. Southern Han is the HCA authority on accrual of reference dates; Parrwood is the leading Qld treatment. The "banking" question turns on contract construction and the timing of accrual relative to termination.

Q: "What is the test for jurisdictional error in a Queensland adjudication decision?"
Plan:
  intent: case_law
  named_provisions: ["BIF Act s 88", "BIF Act s 100", "BIF Act s 101"]
  named_authorities: ["Brodyn Pty Ltd v Davenport [2004] NSWCA 394", "Probuild Constructions (Aust) Pty Ltd v Shade Systems Pty Ltd [2018] HCA 4", "Chase Oyster Bar Pty Ltd v Hamo Industries Pty Ltd [2010] NSWCA 190"]
  notes: Brodyn / Chase Oyster line is the leading interstate authority Qld courts apply. Probuild settled the HCA position on non-jurisdictional error of law on the face of the record. Look for Qld treatments applying these.
"""


def _safe_load_json(text: str) -> dict:
    """Parse JSON, tolerating accidental code fences or trailing prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    # If the model added prose after the object, extract the first {...} block
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    return json.loads(text)


def plan(question: str, *, real: bool = True) -> Plan:
    """Build a retrieval plan.

    real=False is the heuristic baseline (no LLM call). real=True calls
    Claude Opus 4.7 first, falling back through Anthropic and OpenAI
    chains as configured in llm_config.CHAIN_PLANNER_KA.
    """
    if not real:
        return Plan(
            intent=_heuristic_intent(question),
            queries=[question],
            entities={},
        )

    try:
        text, usage = llm_config.complete_chat(
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            kind="planner_ka",
            operation="planner-ka",
            response_format={"type": "json_object"},
            # GPT-5.x family burns hidden reasoning tokens before output;
            # under-provisioning here yields empty content and JSON parse
            # failure. 4000 covers ~1.5k reasoning + ~1k output comfortably.
            max_output_tokens=4000,
        )
        data = _safe_load_json(text)
    except Exception as e:
        log.warning(f"knowledge-augmented planner failed ({e}); falling back to heuristic")
        return Plan(
            intent=_heuristic_intent(question),
            queries=[question],
            entities={},
            confidence="low",
            notes=f"planner-failed: {e}",
        )

    # Always include the original question as a query so case names or
    # statutory phrases the LLM rewrote away still get a shot.
    llm_queries = [q for q in (data.get("queries") or []) if isinstance(q, str) and q.strip()]
    merged_queries = [question] + [q for q in llm_queries if q != question]

    return Plan(
        intent=data.get("intent", "general") or "general",
        queries=merged_queries[:4] or [question],
        entities=data.get("entities") or {},
        named_provisions=[s for s in (data.get("named_provisions") or []) if isinstance(s, str)],
        named_authorities=[s for s in (data.get("named_authorities") or []) if isinstance(s, str)],
        confidence=(data.get("confidence") or "medium").lower(),
        is_fringe=bool(data.get("is_fringe", False)),
        notes=str(data.get("notes") or ""),
        raw=data,
    )
