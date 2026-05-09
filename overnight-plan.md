# Overnight plan — 2026-05-10

**Branch**: `overnight/app-eastbrook-pass`  
**Rollback anchor**: `pre-overnight-2026-05-10` (tag) at `e89707a`  
**Worktree**: `.claude/worktrees/sopal-overnight/`  
**Scope**: app.sopal.com.au only (no sopal.com.au paths). No push to origin.

## Status

| # | Task | Status |
|---|------|--------|
| 1 | RFI panel → table layout | ✅ committed (e57d335) |
| 2 | Read the live app, list visible issues | next |
| 3 | Convert Eastbrook bundle .md → .pdf so it can be uploaded through AA workflow | pending |
| 4 | Stage 1 intake quality-of-life improvements | pending |
| 5 | Drafting agent contenteditable polish (per CC's earlier work) | pending |
| 6 | General housekeeping in sopal-v2.js (7777 lines — look for dead code / refactor opportunities) | pending |
| 7 | Verify the AA exec-summary endpoint with a real master-doc dataset | pending |

## Operating rules

1. Never touch sopal.com.au paths — only `site/sopal-v2.html`, `site/assets/sopal-v2/*`, `routes/sopal_v2.py`, and adjacent app.sopal-only files.
2. Never push to origin. All commits stay on the `overnight/` branch.
3. No mock/seed data. Use real AS 4902 / BIF Act content.
4. Cache-bust HTML version on every js/css change.
5. Commit early and often so the morning review is diff-by-diff readable.
6. Pause if Claude usage on claude.ai goes above 89% — wait for reset, then resume.

## Notes captured from yesterday's CC session

- Master doc 7-section structure shipped at v49: cover / intro / exec summary / jurisdiction / overarching / per-item / conclusion (all optional-when-empty).
- Engine prompt now carries NZ-reply drafting cues (defined-terms discipline, anchoring assertions, restrained-but-assertive voice). Don't undo these.
- BIF Act 2017 not BCIPA. AU English. AS 4902 is the typical D&C contract. AS 4000 the typical construct-only.
- Heading hierarchy: master uses h1/h2; per-thread submissions h3/h4 only.
- s79(2)(a)/(b)/(c) scenarios drive parser, engine, and master content. Don't break.
