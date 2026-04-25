# Sopal Assist — Contract Assist (archived 2026-04-25)

This directory holds the full Contract Assist build that briefly shipped as part
of "Sopal Assist", the two-product suite at `/assist`. Contract Assist was a
RAG-grounded chatbot for Queensland construction contracts, with permitted-draft
generation (variation notices, NODs, EOT claims, payment claims/schedules, general
correspondence) and DOCX export with a watermark on every page.

## Why this is archived

Render's free tier (the active hosting plan as at 2026-04-25) gives the
process 512 MB of RAM. Contract ingestion (pdfplumber + chunker + ChromaDB +
OpenAI embedder client) was overrunning that ceiling on real-world contracts
of even modest size, killing the worker. Rather than aggressively rewrite the
ingestion path while still on free tier, the decision was made to:

1. Park Contract Assist as-is so it can be brought back unchanged once Sopal
   migrates to a paid Render tier (or another host with sufficient RAM).
2. Restore Claim Assist as the canonical, standalone tool at `/claim-check`.
3. Remove the Sopal Assist landing page and the "Assist" top-nav item.

The pivot is structural-only: nothing about Contract Assist's behaviour is being
changed, just where the code lives.

## Branch reference

The pivot point — main exactly as it shipped to production with Contract Assist
live — is preserved at:

    archive/sopal-assist-contract-2026-04-25

That branch contains the live-served version of this code (frontend wired up,
backend orchestration in `routes/contract_assist.py`, the `register_startup`
hook in `server.py`, the cross-product breadcrumb in `/assist/claim`, and the
`/assist` landing page). Anything that this directory cannot be restored from
on its own (e.g. the precise `server.py` wiring, the gitignore entry that
preserved the BIF Act ChromaDB cache path, the `/claim-check` → `/assist/claim`
301 redirect that is being reversed in the pivot) lives on that branch.

To inspect or resurrect the live version verbatim:

    git fetch origin
    git checkout archive/sopal-assist-contract-2026-04-25

## What is in this directory

```
_archive/contract-assist-2026-04-25/
  README.md                                  ← this file
  routes/
    contract_assist.py                       ← FastAPI router (5 endpoints under /api/contract-assist/*)
  services/
    contract_assist/
      __init__.py
      chatbot.py                             ← SSE-streaming orchestrator with intent classifier + thinking states
      draft_exporter.py                      ← python-docx generator with VML watermark on every page
      prompts.py                             ← system prompt + verbatim drafting policy
      bif_act_index/
        __init__.py                          ← register_startup(app) hook + public retrieve_bif()
        builder.py                           ← parses /bif-act-guide and embeds 138 chunks into ChromaDB
        retriever.py                         ← cosine-similarity retrieval API
      retrieval/
        __init__.py
        chunker.py                           ← structure-aware chunker (1200-token, 200-overlap)
        embedder.py                          ← OpenAI text-embedding-3-small batched
        service.py                           ← public ingest/retrieve/clear API; hybrid retrieval
        store.py                             ← ChromaDB EphemeralClient (session-scoped, in-process)
  site/
    assist.html                              ← Sopal Assist landing page (two product cards + drop zones)
    assist/
      contract.html                          ← Contract Assist UI (two-column desktop, mobile tabs, viewer + chat)
    assets/
      contract-assist/
        contract-assist.css                  ← page-specific styles, source-pill design, shimmer
        contract-assist.js                   ← state, viewer wiring, SSE consumer, source-pill injection
  tests/
    contract_assist/
      __init__.py
      test_retrieval.py                      ← 8 tests for the retrieval pipeline
      test_bif_act_index.py                  ← 5 tests for the BIF Act index builder + retriever
```

External dependencies that the live code referenced and are NOT in this archive:

- `services/claim_check/extractor.py` — Contract Assist reused this for PDF/DOCX
  extraction. It stays in the live tree because Claim Assist still uses it.
- `site/assets/claim-check/viewers/` — pdf.js + DOCX→PDF viewer modules, reused
  by Contract Assist via `/assets/claim-check/viewers/...` script tags. Stay
  in the live tree because Claim Assist depends on them.
- `site/assets/claim-check/components/modal.js` — bespoke modal component,
  reused by Contract Assist. Stays in the live tree because Claim Assist
  depends on it.

The Astruct reference codebase that informed the patterns is not stored here
either; it sat at `/astruct/` during the build (gitignored). The patterns doc
that Subagent 1 produced (`/tmp/astruct_patterns.md`) is also not bundled
because it was a working artefact, not source.

## Restoration notes — what would need to be re-wired to bring Contract Assist back

If you want to bring Contract Assist live again, the cleanest path is:

1. Confirm the Render tier has at least 2 GB RAM (Standard plan or higher).
   Contract ingestion will OOM on the free tier without aggressive rework.
2. Either:
   - Cherry-pick the matching commits from `archive/sopal-assist-contract-2026-04-25`
     onto the current main, OR
   - Move every directory under this `_archive/contract-assist-2026-04-25/`
     back to its original path (mirror the directory structure under repo
     root: `services/contract_assist/`, `routes/contract_assist.py`,
     `site/assist.html`, `site/assist/contract.html`,
     `site/assets/contract-assist/`, `tests/contract_assist/`).
3. Re-add to `server.py`:
   ```python
   from routes.contract_assist import router as _contract_assist_router
   app.include_router(_contract_assist_router)
   from services.contract_assist.bif_act_index import register_startup as _ca_bif_register_startup
   _ca_bif_register_startup(app)
   ```
4. Re-add the cross-product breadcrumb to `site/claim-check.html` (the version
   on the archive branch shows the exact HTML and inline switcher script).
5. Re-add the "Assist" item to the top nav across nav-bearing pages, between
   "Due Date Calculator" and "How-To Guide". Update the legacy /claim-check
   redirect: it should once again point to /assist/claim, and add the new
   /assist/* → /claim-check redirects in reverse.
6. Re-add `_archive/contract-assist-2026-04-25/services/contract_assist/bif_act_index/chroma/`
   (or the live path the index will use) to `.gitignore`.
7. If running on a tier with a persistent disk (`/var/data`), consider moving
   the BIF Act ChromaDB cache there so it survives restarts. Contract session
   collections deliberately use `EphemeralClient` and should remain in-memory.
8. Run `python3 -m pytest tests/contract_assist/ -v` to confirm all 13 tests
   pass before redeploying.

## Date and decision provenance

- Built: 2026-04-24 (overnight)
- Live in production: ~24 hours (2026-04-24 → 2026-04-25)
- Pivot decision: 2026-04-25 (Render free tier OOM on real contracts)
- Archived: 2026-04-25
