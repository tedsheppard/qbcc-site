# Sopal v2 UX audit — 2026-05-11

Live URL audited: https://app.sopal.com.au/sopal-v2
Worktree branch: `claude/sharp-meninsky-b447a3` (pushed to `main`)
Cache-bust at start: `v=20260511-3` (then bumped twice during audit to `-7` and `-8`)

The signed-in test account did not have a `purchase_token` in this browser's localStorage at the time of audit, so most of the walk was done as a guest. Guest mode still exercises every route, every drafting agent landing page, the AA workspace shell, all standalone tools, all Settings cards, and every help article, because Sopal v2 stores state per-browser without sign-in. Sign-in only enables cloud sync.

While I was working, `main` advanced (someone shipped the docx/pdf export and the first-run onboarding overlay). I rebased onto that and kept both new things intact.

---

## Commits shipped

| SHA | Summary |
|-----|---------|
| `55b44d7` | Sopal v2 UX audit fixes: decision click P0, adjudicator scroll, dark-mode contrast on empty-state + radio, em-dash sweep across user-visible copy |
| `15e776b` | Sopal v2: dark-mode contrast P0 on decision detail body + result rows |

Cache-bust went `20260511-3` → `20260511-7` → `20260511-8`.

---

## P0 bugs fixed

### P0-1. Decision search result click did nothing
**File**: `site/assets/sopal-v2/sopal-v2.js` (search results click handler).
**Cause**: The click handler called `loadDecisionDetail(...)`, which rendered into `#decision-detail`. That mount only exists on the *detail* route — not on the search results page. So clicking a result silently returned at the `if (!mount || !id) return;` guard. The bug was hidden behind a stub that *did* run before the guard: `trackRecentDecision`, which is why "Recently viewed decisions" on home worked while the click itself did nothing.
**Fix**: Result-row click now navigates to `/sopal-v2/research/decisions/{id}` via the SPA `navigate(...)` helper. `trackRecentDecision` is still called inline so home keeps populating.
**Verified live**: Clicking a result on `/sopal-v2/research/decisions?q=delay` now opens `/sopal-v2/research/decisions/EJS00001` and the decision body renders.

### P0-2. Decision detail body text near-invisible in dark mode
**File**: `site/assets/sopal-v2/sopal-v2.css`.
**Cause**: `.decision-text { color: #2c2c2c }` — hardcoded dark grey. On the dark-mode panel (`#1f1d1b`) the body text rendered near-black-on-black. The result-row title also relied on inheriting from a hardcoded `background:#fff` on `.result-row` which left titles dim too.
**Fix**: `.decision-text` now uses `var(--main-fg)`. `.result-row` uses `var(--main-panel)` / `var(--main-fg)`. Hover uses `var(--main-muted)`. Snippet `p` uses `var(--muted-fg)`.
**Verified live** on `v=20260511-8`: `.decision-text` computed color is `rgb(236, 236, 234)` on the dark panel (was `rgb(44, 44, 44)`), and `.result-row` background is now `rgb(24, 23, 21)`. The decision body renders cleanly.

---

## P1 bugs fixed

### P1-1. Adjudicator detail rendered below the fold with no scroll
**File**: `site/assets/sopal-v2/sopal-v2.js` (`loadAdjudicatorDetail`).
**Cause**: The page renders a single-column `research-grid` on viewports under ~980px (most laptops), so clicking an adjudicator card filled `#adj-detail` *below* the 10-card adjudicator grid. The click felt dead — you had to scroll a long way down to find the result.
**Fix**: `loadAdjudicatorDetail` now `scrollIntoView({behavior:"smooth",block:"start"})` on the mount immediately so the click registers visually.

### P1-2. New Project modal: "The respondent" radio + empty-state card stuck on white in dark mode
**File**: `site/assets/sopal-v2/sopal-v2.css`.
**Cause**: `.card-empty` and `.radio-option` both had `background: #fff` hardcoded, so the "Create your first project" empty card on `/sopal-v2/projects`, and the "You act for" radio cards in the New Project modal, appeared as bright white tiles in dark mode.
**Fix**: Both now use `var(--main-panel)`.

### P1-3. Em dashes in user-visible copy (Ted's standing rule)
**File**: `site/assets/sopal-v2/sopal-v2.js`.
**What I touched** (verbatim BIF Act sections and verbatim decision text were left alone per the verbatim-legal-text memory):
- All 8 `AGENT_DESCRIPTIONS` (visible as drafting agent landing-page subtitles).
- `COMPLEX_AGENT_DESCRIPTIONS["adjudication-application"]`.
- Review checklists for payment schedules (s 76 line), EOTs (Trigger event / Notice timing / Causation / Delay period), variations (Direction or instruction / Notice compliance / Time impact), and timing checks for AA and AR.
- Due-date scenario sub-labels (`s 79(2)(a)`, `s 79(2)(b)`, `s 79(2)(c)`, and the three "Schedule received" labels).
- Due-date dynamic page title: `${meta.title}: due date`.
- Interest calculator section header: `BIF Act s 73: interest on overdue progress payments`.
- AA document titles in the `.doc`/`.docx`/`.pdf` exports — `${project.name}: Adjudication Application`, `: Statutory Declaration`, `: Index of Supporting Evidence`.
- Assistant/review heading prefix (`{Agent} review: {Project}`).
- Drafting-agent doc title `{project.name}: {AGENT_LABELS[key]}`.
- AA stage prompts: Item thread prefix, "No AA to snapshot yet…" alert, "No disputed items in the table. Lock anyway?" confirm, the Stage 5 cross-cutting-arguments hint.
- File-upload helper "Click or drop one or more PDF / DOCX / TXT files. Each becomes a separate entry."
- Suggested prompt chips ("Audit a payment claim served on me", "Compare this claim against the prior claim").
- Adjudicator citation suffix in chat citations.
- "What's new" entries for jurisdiction selector, pinned context, and bulk upload.
- Parse-fallback raw-response banner.

**Not touched**: verbatim BIF Act section quotes (s 73, s 76, s 79, s 83, s 85 etc.) and verbatim citations from decisions. Those still have the em dashes that exist in the underlying statute / judgement, by your standing rule.

---

## Verified live

- Decision search: `/sopal-v2/research/decisions?q=delay` returns 1,714 results, snippet highlights, pagination. Clicking a result now navigates to `/sopal-v2/research/decisions/{id}` and renders the body.
- Adjudicator stats: `/sopal-v2/research/adjudicators` shows 301 adjudicators with metrics, filter, sort. Clicking a card now scrolls the detail panel into view.
- Research agent: jurisdiction switcher works; non-QLD shows the "sources aren't yet integrated" banner.
- Tools: Payment Claim Reviewer, Payment Schedule Reviewer, Due Date Calculator (Payment Schedule scenario tested with empty submit → "Enter the date the payment claim was given." inline error), Interest Calculator. All load.
- Settings: Account (guest banner correct), Cloud sync (correct off state for guest), Firm (all fields present), Data and storage (counters render), Appearance (theme toggle works).
- Help: index renders, the Getting Started article renders, breadcrumbs work.
- Projects: Empty-state quickstart card renders. New-project modal works. Project create lands on the four-step Quick Start overview.
- Drafting agents: Payment Claims landing page renders the Word-style editor with the prefilled Payment Claim template.

---

## P2 + open punch list

These are documented but not shipped this session. None block the demo.

1. **Adjudicator detail "decisions by this adjudicator" rows navigate to a re-search by title**, not to the decision detail directly. The data fixture for each `mini-item` has the decision title but not the EJS id, which is why this was wired as a search. Real fix: have `/api/adjudicator/{name}` return the EJS id per decision and switch the click to `navigate(/research/decisions/{id})`. Until then, the current behaviour is a deliberate fallback, not a bug.

2. **Adjudicator detail panel only sits in a sidebar on viewports ≥980px**. On laptops it stacks under the list. Fixed the "you have to scroll" symptom with `scrollIntoView`, but the long-term answer is a sticky right-column at ≥1100px or a slide-in drawer at narrower widths. Out of scope for a one-line fix.

3. **Hardcoded `background: #fff` survives in 18 other rules** in `sopal-v2.css` (lines 421, 585, 617, 690, 715, 927, 963, 1151, 1234, 1270, 1398, 1515, 1538, 1557, 1606, 1661, 1715, and a handful past 2000). The most user-visible ones are now fixed (`.card-empty`, `.radio-option`, `.result-row`, `.decision-text`). The rest are mostly modal/drawer surfaces that *also* have a `[data-theme="dark"]` override later, so they look fine, but a systematic pass would replace every literal `#fff` with `var(--main-panel)` and remove the dark-mode overrides.

4. **`.recent-decision-card`, `.tile`, `.adj-card`, `.project-row`, `.help-card`** were not individually verified for dark-mode hover/active states. Spot-checks looked OK; recommend a one-pass screenshot review at some point.

5. **Mobile viewport (~390px) was not browser-verified** because the chrome MCP virtual viewport stays at 800px regardless of `resize_window`. Code review of the @media rules at 380/620/720/780 looks reasonable — sidebar collapses to drawer at <620px, grids stack — but a real device check before any public push is warranted.

6. **`.dark-button` text in the new-project modal looks white**. That's by design (`--button: #f5f2ed` in dark mode) but it sits on the same near-white `#fff` background a screenshot apart from the rest of the modal, so visually it reads as "low contrast button". Not a bug, but if you want, the dark-mode `.dark-button` could pick up a 1-px ring to give it edge.

7. **Recently viewed decisions** on `/sopal-v2` (the home page) shows a card whose click likely routes correctly now (it has `data-nav` href). I did not regression-check, but the same handler is used elsewhere and worked.

8. **Decision search "Save search" button** is present but I did not exercise it. The handler exists in the JS but the persistence path was not walked.

9. **"What's new" modal** was not opened. Three of its lines previously contained em dashes (fixed). The Cmd+K palette also was not exercised.

10. **AA workspace stages 1–5 and the AA master modal** were not deeply walked because exercising them requires pasting in real claim text. The static audit (file:line review) showed every modal has Escape + backdrop click + Cancel handlers — that was verified by the parallel codebase pass. Stage transitions and the live-master rendering remain visually unverified end-to-end.

11. **Console**: no JS errors observed during navigation. The chrome MCP console tracker starts late, so anything that fires during initial paint may not be captured.

12. **Sidebar layout when "Drafting agents" + "Complex agents" list extends below `sidebar-foot`**: looked OK after fix — `.sidebar-scroll` does overflow-y:auto. The "guest auth" card pinned to the foot is correctly above the sidebar bottom.
