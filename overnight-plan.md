# Overnight plan — 2026-05-10

**Branch**: `overnight/app-eastbrook-pass`  
**Rollback anchor**: `pre-overnight-2026-05-10` (tag) at `e89707a`  
**Worktree**: `.claude/worktrees/sopal-overnight/`  
**Scope**: app.sopal.com.au only (no sopal.com.au paths). No push to origin.

## Status

| #  | Commit  | What |
|----|---------|------|
| 1  | e57d335 | RFI panel rebuilt as a table — # / Sopal's question / Your response / Submit, plus Edit affordance and Cmd+Enter to submit |
| 2  | a7e2264 | Stage 5 actions: stack title + actions vertically, wrap export buttons two-per-row so nothing clips |
| 3  | 59e2061 | Stage 2 dispute table: `$` prefix on Claimed / Scheduled cells |
| 4  | 278e01d | Stage 1 intake: replace `alert()` validation + parse-failure modals with an inline error pane |
| 5  | 3dd712f | Stage 2 dispute table: actions wrap; table itself horizontal-scrolls inside card-body for mobile |
| 6  | 16c2db6 | Cover page: add Claimant / Respondent ABN fields (cover meta editor + master cover table) |
| 7  | e847577 | Cover page: add Contract executed date + Project / site address fields |
| 8  | f0bddac | Stage 2: new "Matter details" editor (Claimant / Respondent / Contract ref / Reference date) |
| 9  | 6e4ded0 | AA engine: forward cover-page extras (ABN, contract date, site address, ANA, contacts) so introductions / background threads can weave them in |
| 10 | e8c2fea | overnight-plan.md created |
| 11 | 54da2a2 | EDIT_DRAFT_SYSTEM_PROMPT: explicit BIF Act 2017 not BCIPA rule + cite section/clause numbers |
| 12 | 4e4b840 | Stage 2 status cell colour-coded (disputed=red, admitted=green, partial=amber, jurisdictional=purple) |
| 13 | 5693787 | Stage 1 deadline: "Calculate from dates →" link to the Due Date Calculator preset to s 79 BIF Act |
| 14 | 0aa7c00 | Cover-page editor: group fields into Application context / Claimant / Respondent sections |

Total: ~560 insertions, ~80 deletions across 5 files.

## Operating rules

1. Never touch sopal.com.au paths — only `site/sopal-v2.html`, `site/assets/sopal-v2/*`, `routes/sopal_v2.py`, and adjacent app.sopal-only files.
2. Never push to origin. All commits stay on the `overnight/` branch.
3. No mock / seed data.
4. Cache-bust HTML version on every js/css change (currently at v20260510-12).
5. Commit early and often so the morning review is diff-by-diff readable.
6. Pause if Claude usage on claude.ai goes above 89% — wait for reset, then resume.

## Notes / decisions

- Existing CC worktree (vibrant-shirley) untouched — Ted's other Claude Code session is unaffected.
- Engine prompt drafting-style cues (NZ-reply derived: defined-terms, anchoring, restrained-but-assertive, BIF Act not BCIPA, AU English) carry through unchanged from v47.
- Master document fluid 7-section structure (cover / intro / exec summary / jurisdiction / overarching / per-item / conclusion) stays — only optional sections render when content exists.
- Cover meta now has 17 fields, grouped Application context / Claimant / Respondent. All optional; empty fields don't render in master.
- Status colours use existing CSS variables and BIF-friendly tones (no aggressive saturation).

## Visual coverage of changes (what to look for during review)

- Stage 1 Intake: inline parse-error pane replaces alert; "Calculate from dates →" link next to deadline.
- Stage 2 Dispute Table: `$` prefixes; status cell colour-coded; "Matter details" button in card-head; mobile horizontal-scrolls.
- Stage 3-4 Workspace: RFI table with #/Q/Response/Submit columns; Edit on answered rows; Cmd+Enter submits.
- Stage 5 Review: action-row wraps two-per-row in narrow column.
- Master modal cover page: ABN row, Contract executed row, Project / site address row.
- Cover-page editor: grouped sections.

## Rollback

```bash
cd /Users/tedsheppard/Desktop/Projects/qbcc-site
# View what's on the overnight branch:
git log --oneline pre-overnight-2026-05-10..overnight/app-eastbrook-pass
# Discard everything overnight and reset main:
git checkout main
git reset --hard pre-overnight-2026-05-10
git branch -D overnight/app-eastbrook-pass
git worktree prune
```

Or if you want to merge into main (review first):

```bash
git checkout main
git merge --no-ff overnight/app-eastbrook-pass
# inspect, run tests, then push when happy
```
