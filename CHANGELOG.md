# Sopal — changelog

Notable structural and behavioural changes to the Sopal codebase. Tooling
fixes, copy edits, and small bug fixes are not catalogued here — see
`git log` for the full history.

## 2026-04-25 — Sopal Assist suite shelved; Claim Assist restored to standalone

**What changed**

- The Sopal Assist suite (`/assist`, `/assist/claim`, `/assist/contract`)
  is gone from the live site. The "Assist" item is removed from the top
  nav across every nav-bearing page.
- Claim Assist is back at its canonical URL `/claim-check` as a
  standalone product. The page H1 stays as **Claim Assist** (the rename
  from "Claim Check" was an improvement worth keeping regardless of
  suite branding). The cross-product breadcrumb and the "Switch to
  Contract Assist" button are removed.
- Tagline tightened on `/claim-check` to: *"Check BIF Act compliance of
  payment claims and payment schedules — before you serve, or after you
  receive."*
- `/assist`, `/assist/claim`, and `/assist/contract` now return a 301
  redirect to `/claim-check` — bookmarks created during the brief
  Sopal Assist period (≈24 hours) are funnelled cleanly back to Claim
  Assist instead of 404-ing.
- Reverted the temporary `/claim-check` → `/assist/claim` 301 that
  shipped with the suite. `/claim-check` is canonical again.
- Claim Assist localStorage cap restored to 10 sessions and now only
  counts Claim Assist entries when enforcing the cap. Any Contract
  Assist localStorage entries left over from the brief Sopal Assist
  period are silently ignored — never displayed in the Previous
  Sessions panel, never counted toward the cap, and never deleted.

**Why**

Render's free tier (the active hosting plan) gives the Python process
512 MB of RAM. Contract ingestion (pdfplumber + chunker + ChromaDB +
OpenAI embedder client) was overrunning that ceiling on real-world
contracts of even modest size, killing the worker. Rather than rewrite
the ingestion path while still on free tier, the decision was made to
park Contract Assist as-is and revisit once Sopal moves to a paid tier
(or onto a different host with sufficient memory).

The pivot is **structural-only**: nothing about Contract Assist's
behaviour was changed, just where the code lives. Existing Claim Assist
defects (reasoning-trace coverage, debug-metadata leakage, PDF rotation,
generic-error rendering) are tracked separately and will land in a
focused Claim Assist iteration after this revert.

**How to restore Contract Assist later**

Two recovery points:

1. **Branch:** `archive/sopal-assist-contract-2026-04-25` — preserves
   the live-served version of the suite at the moment of pivot. Check
   out this branch, or cherry-pick its commits onto a future main, to
   bring Contract Assist back exactly as it shipped.

2. **Directory in main:** `_archive/contract-assist-2026-04-25/`
   contains the full Contract Assist build (services, routes, frontend,
   tests, system prompt, drafting policy, DOCX exporter). The
   `README.md` inside that directory documents what's in it and the
   exact server-side wiring required to reactivate it (the
   `app.include_router(...)` call in `server.py`, the BIF Act index
   `register_startup` hook, the cross-product breadcrumb in the page
   template, the `_archive/` gitignore note).

Contract Assist will return when:

1. Render is on a tier with at least 2 GB RAM (Standard plan or
   higher), OR
2. Sopal is hosted somewhere other than Render with sufficient memory.

Until then, the directory and the branch sit dormant. Nothing in
`_archive/` is imported by live code — it's tracked for posterity only.

**Files moved (not deleted)**

```
services/contract_assist/        →  _archive/contract-assist-2026-04-25/services/contract_assist/
routes/contract_assist.py        →  _archive/contract-assist-2026-04-25/routes/contract_assist.py
site/assist.html                 →  _archive/contract-assist-2026-04-25/site/assist.html
site/assist/contract.html        →  _archive/contract-assist-2026-04-25/site/assist/contract.html
site/assets/contract-assist/     →  _archive/contract-assist-2026-04-25/site/assets/contract-assist/
tests/contract_assist/           →  _archive/contract-assist-2026-04-25/tests/contract_assist/
```

`site/assist/claim.html` was moved back to `site/claim-check.html` via
`git mv` so the file's history (including the Phase-2-onwards
iteration) is preserved across the rename in both directions.
