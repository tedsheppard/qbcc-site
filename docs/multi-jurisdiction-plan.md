# Multi-jurisdiction expansion plan (Sopal v2)

Sopal v2 is currently QLD-only. This is the staged plan for adding NSW, VIC, WA, and SA. The Research Agent jurisdiction picker (already shipped) is the visible front door — every other jurisdiction-aware feature plugs in behind it.

## What "QLD-only" means today

- **Decision corpus**: `qbcc.db` is QLD adjudication decisions sourced from the QBCC public register. ~thousands of rows, FTS-indexed under `fts`.
- **Adjudicator stats**: derived from QLD decisions; adjudicators are registered through the QBCC under the BIF Act / BCIPA regime.
- **System prompts**: BIF Act 2017 (Qld) is the default frame. The Research Agent now scopes the prompt by jurisdiction, but for non-QLD it explicitly tells the model "no decision corpus, rely on general knowledge, flag uncertainty".
- **Calculators**: Due Date Calculator hardcodes BIF Act sections (s 76 / 79 / 83 / 85 / 87) and QLD public + regional show holidays. Interest Calculator uses QBCC s 67P (10% + RBA) — that's a QLD-specific statutory rate.
- **Drafting templates**: Each agent's starter doc cites BIF Act sections (e.g. payment claim cites s 75).

## Per-jurisdiction shopping list

| Jurisdiction | Primary act | Current source today | What we'd need |
|---|---|---|---|
| NSW | Building and Construction Industry Security of Payment Act 1999 (NSW) | None | NSW Caselaw + ANA registers (Adjudicate Today, RICS, IAMA) for the decision corpus. Different timing rules (10 BD schedule, 20 BD adjudication application). |
| VIC | Building and Construction Industry Security of Payment Act 2002 (Vic) | None | VCAT + Supreme Court Vic + VBA decisions. "Excluded amounts" concept (s 10B) has no QLD analogue. |
| WA | Building and Construction Industry (Security of Payment) Act 2021 (WA) — and Construction Contracts Act 2004 (WA) for older matters | None | SAT WA + Supreme Court WA. WA shifted regimes on 1 Aug 2022 — the corpus has to be tagged by which act applied. |
| SA | Building and Construction Industry Security of Payment Act 2009 (SA) | None | SACAT + Supreme Court SA decisions. Smaller corpus than NSW/VIC. |

Each jurisdiction also has its own:
- ANAs and adjudicator registration scheme → adjudicator stats need a jurisdiction column on the source table.
- Public-holiday schedule (already structured as `HOLIDAYS[location]` in `sopal-v2.js`; needs NSW/VIC/WA/SA arrays).
- Statutory interest rate (NSW: Civil Procedure Act 2005 s 101 + UCPR; VIC: Penalty Interest Rates Act 1983; WA: Civil Judgments Enforcement Act 2004; SA: Supreme Court Act 1935 s 30C). The current QBCC s 67P live-fetch pipeline is the "easy" reference shape — the others are mostly fixed-rate.

## Schema and data work

1. **Decisions table** — add a `jurisdiction` column (`qld`/`nsw`/`vic`/`wa`/`sa`). Backfill the existing rows to `qld`. Add a per-jurisdiction FTS index OR a single FTS table with a `jurisdiction` filter applied at query time.
2. **Adjudicators** — already keyed by name. Add a `jurisdictions` array (an adjudicator can be registered in more than one). The aggregated stats need to be computed per jurisdiction.
3. **Decision text storage** — same shape, just tagged.
4. **Sourcing pipeline** — each jurisdiction needs its own scraper / ingestion step. Realistic effort: NSW Caselaw is feasible but volume is large; VIC has sparse public access; WA and SA have small registers. Budget: 3–6 weeks per jurisdiction to do data sourcing + extraction + QA properly. This is the bulk of the cost.

## Frontend wiring

1. **Global jurisdiction context** — surface the currently-selected jurisdiction at the workspace level (today it's only on Research Agent). Suggested storage: `store.activeJurisdiction`. Persist in localStorage. Show a small jurisdiction pill in the top header so users always know which lens they're using.
2. **Decision Search** — add `&jurisdiction=` to the `/api/sopal-v2/search` query. Filters dropdown gets a Jurisdiction picker. URL-driven.
3. **Adjudicator Statistics** — same: query `/api/adjudicators?jurisdiction=`. The list and stats recompute per jurisdiction.
4. **Tools / calculators** — calculators load per-jurisdiction config (sections, day rules, holidays, interest rate source). Hide rate options that don't apply (e.g. QBCC s 67P only shows on QLD).
5. **Drafting agents** — each agent's starter template + system instructions branch on jurisdiction. Templates would live in a `templates/{jurisdiction}/{agent-key}.html` map.

## Server / prompt wiring

1. `/api/sopal-v2/search`, `/api/adjudicators`, `/api/adjudicator/{name}` accept an optional `jurisdiction` param and apply a SQL filter. Default to `qld` for back-compat.
2. `/api/sopal-v2/agent` and `/api/sopal-v2/agent/edit-draft` accept a `jurisdiction` field. The system prompt loads jurisdiction-specific framing (already wired for the Research Agent — extend the same pattern).
3. **Calculator endpoints** — `/get_interest_rate` becomes `/get_interest_rate?jurisdiction=`, returning the right rate series. New jurisdictions need their own rate sources.

## Phasing recommendation

- **Phase 1 (cheap, ship in days)**: jurisdiction-aware system prompts everywhere they aren't already (drafting agents, project assistant). Keep the corpus / stats / calculators QLD-only and surface a clear "Limited support" banner when a non-QLD jurisdiction is selected. This is what the Research Agent already does. Lets users at least get prompt-only assistance for NSW/VIC/WA/SA matters without us shipping bad data.
- **Phase 2 (weeks)**: add NSW first — it has the largest corpus and best public sources. Schema migration + scraper + per-jurisdiction calculator config + per-jurisdiction templates.
- **Phase 3 (weeks per state)**: VIC, then WA, then SA. Each is a self-contained data-engineering project with a small follow-up of frontend tweaks.

## Out of scope (for now)

- Multi-jurisdiction projects (a single project applying different state acts across head/sub) — uncommon, deal with it in v3.
- Federal Comcare / NTAA matters — vanishingly rare in this domain.
- Cross-jurisdiction comparison views ("how would this claim be treated in NSW vs QLD") — interesting but blocked on having two real corpora to compare.

## Anchor files (current code)

- Decision search route: [routes/sopal_v2.py:331](../routes/sopal_v2.py)
- Research Agent jurisdiction framing: [routes/sopal_v2.py:`RESEARCH_JURISDICTION_FRAMING`](../routes/sopal_v2.py)
- Holiday tables: [site/assets/sopal-v2/sopal-v2.js — `HOLIDAYS`](../site/assets/sopal-v2/sopal-v2.js)
- Drafting templates: [site/assets/sopal-v2/sopal-v2.js — `AGENT_TEMPLATES`](../site/assets/sopal-v2/sopal-v2.js)
