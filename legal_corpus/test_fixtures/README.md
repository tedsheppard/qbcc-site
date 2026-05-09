# Sopal v2 — public-contract test fixtures

Hand-picked, ethically-sourced Australian construction-law artefacts used to dogfood the Sopal v2 Adjudication Application parser and the drafting agents' templates. Nothing here is private user data or commercially-licensed content — every file here is either:

- A government / Crown-published model contract or template (Crown copyright, freely usable for research and operational use), OR
- A peak-body / industry-association template explicitly published for free public use.

We do **not** include Standards Australia AS 4000 / 4902 / 2124 — those are commercially licensed (~A$200–600 per editable seat) and must not be redistributed. Where the parser needs to handle AS-style language, we rely on the heavily-quoted versions inside AustLII-published judgments (fair-dealing).

Per [docs/multi-jurisdiction-plan.md](../../docs/multi-jurisdiction-plan.md), fixtures are tagged by jurisdiction so the Sopal parser can scope tests by state.

## What's here

| Slug | Source | Type | Jurisdiction |
|---|---|---|---|
| `nsw-gc21-ed2/` | NSW Government — info.buy.nsw.gov.au | Head contract — Construction Contracts (GC21) Edition 2 | NSW |
| `nsw-mw21/` | NSW Government — info.buy.nsw.gov.au | Minor Works contract suite (MW21) | NSW |
| `act-mw21/` | ACT Government — Infrastructure Canberra | Minor Works contract suite (MW21) | ACT |
| `qbcc-pc-template/` | Queensland Building & Construction Commission | Payment Claim template (BIF Act compliant) | QLD |
| `mbq-bif3-ps/` | Master Builders Queensland | Payment Schedule template (BIF Act compliant) | QLD |

Each subdirectory contains:
- `source.txt` — the original URL, retrieval date, and licence notes
- the original PDF / DOCX as downloaded
- (optional) `extracted.txt` for plaintext-extracted content if the source was a PDF

## Acquisition

`scripts/fetch_public_contracts.sh` fetches the directly-downloadable items in the table above. Run from the repo root:

```bash
bash scripts/fetch_public_contracts.sh
```

The script never overwrites an existing file — it skips anything already present. Per-source licence text is captured in each `source.txt`.

## Curated AustLII judgments (separate ingest)

For the parser-stress tests we hand-pick about 20 NSWSC / QSC SOPA / BIF Act judgments that quote contract clauses + PC/PS extracts at length. Seed cases:
- *Probuild Constructions (Aust) Pty Ltd v Shade Systems Pty Ltd* [2018] HCA 4
- *Roberts Co v Sharvain Facades* [2025] NSWCA 161
- *Niclin Constructions Pty Ltd v SHA Premier Constructions Pty Ltd* (Qld)
- *Multiplex v Luikens* (NSW)
- *John Holland v TAC Pacific* (Qld)

These are pulled separately as needed (AustLII has stable URLs; manual selection is more useful than crawling).

## Don't add

- Standards Australia AS-series contracts (commercial)
- HIA / MBA member-only templates (member paywall)
- Court registry pleadings (per-file fee + confidentiality risk)
- Anything containing real party names, real project addresses, or real PII outside what's already in the published judgments

## Use

Sopal v2 ingestion is opt-in: nothing here gets loaded into the live app automatically. To use one of these as a test parse:
1. Open the Adjudication Application Complex Agent in any project
2. Drop the relevant PDF into the Stage 1 intake area
3. The parser will treat it like any user-supplied document
