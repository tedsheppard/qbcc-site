# Overnight test plan

A walkthrough you (or anyone reviewing the branch) can follow to manually verify every change in `overnight/app-eastbrook-pass`. Steps are grouped by area and reference the commit that introduced the change.

## Pre-requisites

1. `git checkout overnight/app-eastbrook-pass` (or `git worktree list` to see the existing worktree at `.claude/worktrees/sopal-overnight/`).
2. Run the FastAPI server locally: `uvicorn server:app --reload --port 8000`.
3. Open `http://localhost:8000/sopal-v2` in a browser.
4. Sign in with your purchase account (the SidebarAuth row in the bottom-left should say "Signed in" with your name).

## 1. RFI table redesign (e57d335)

1. Open project "t" → Adjudication Application → Stage 3 RFI.
2. Confirm the right pane shows a TABLE with columns: `#`, `Sopal's question`, `Your response`, action.
3. The first row should be RFI 1 with a textarea + Submit button. Type an answer, hit Submit; confirm a new row appears for RFI 2.
4. Click Edit on an answered row; the textarea should reappear pre-filled with your prior answer.
5. Press Cmd+Enter inside a textarea; confirm it submits without clicking the button.
6. Resize the browser to under 720 px wide; rows should stack into a card-per-row layout.

## 2. Stage 5 actions wrap (a7e2264)

1. Stage 5 Review with project "t".
2. Confirm the action buttons (Export master, Stat dec, Evidence index, Print, Copy, Draft all) wrap two-per-row in the narrow summary column instead of clipping off the right edge.

## 3. Dispute table currency prefix and mobile (59e2061, 3dd712f)

1. Stage 2 Dispute table.
2. Each Claimed and Scheduled cell shows a `$` prefix to the left of the number input. Editing still works as a number.
3. Resize to under ~900 px; the table should stay readable via a horizontal scroll inside the card body. The Add row / Lock buttons should wrap, not clip.

## 4. Inline parse error (278e01d)

1. Stage 1 Intake. Try Parse documents with the Payment Claim slot empty.
2. Confirm a red bordered error pane appears just below the action row (not a browser alert).
3. Add the PC text and hit Parse again; the error should clear before parsing starts.

## 5. ABN, Contract date, Site address fields (16c2db6, e847577)

1. Stage 5 Review → View master document → Cover page.
2. Confirm the modal has fields for Claimant ABN, Respondent ABN, Contract executed on, Project / site address, plus the existing fields.
3. Fill in (e.g.) Claimant ABN "12 345 678 901" and Save. Re-open the master; the cover-page Claimant table should now have an "ABN" row above Contact.

## 6. Matter details editor (f0bddac)

1. Stage 2 Dispute table → click "Matter details" in the card head.
2. Edit the Claimant name from "Acme Builders" to "Acme Builders Pty Ltd" and Save.
3. Confirm the Stage 2 summary line now shows the new name. View the master document; the cover page Claimant Name should also update.

## 7. AA engine forwarded cover meta (6e4ded0)

1. Fill in cover-page Claimant ABN, Contract executed date, and Site address (Stage 5 cover modal).
2. Open Stage 3, pick the Background / General thread, click "Ask another RFI" or run a draft. The engine should now have the cover-meta values available to weave into the introduction paragraph rather than leaving placeholders.

## 8. EDIT_DRAFT_SYSTEM_PROMPT tighten (54da2a2)

1. Open the Variations drafting agent in any project.
2. In the chat composer, type: "Change every reference to BCIPA to BIF Act."
3. Confirm Sopal updates the document accordingly. (Note: this is a backend prompt change, no visible UI difference.)

## 9. Status colour-coding (4e4b840)

1. Stage 2 Dispute table.
2. Change the status column on a row to `admitted`. Cell background should be a soft green.
3. Try `partial` (amber), `jurisdictional` (purple). Reset to `disputed` (soft red).

## 10. Calculate from dates link (5693787)

1. Stage 1 Intake.
2. Click "Calculate from dates →" beside the Lodgement deadline input.
3. Confirm it opens the Due Date Calculator preset to the Adjudication Application scenario.

## 11. Cover-page editor grouping (0aa7c00)

1. Open the cover-page editor (Stage 5 → master modal → Cover page).
2. Confirm the fields are grouped under three headings: Application context, Claimant, Respondent.

## 12. Contract / Library placeholder fix (f8e1cfc)

1. Open Project Library (project sub-nav).
2. Textarea placeholder reads "Paste correspondence, RFIs, claims, schedules, programme notes, or facts."
3. Open Contract.
4. Textarea placeholder reads "Paste contract clauses or terms here. For example: cl 36 (Variations), cl 41 (Default), cl 42 (Payment)."

## 13. Help and Support system (664b7d4)

1. Sidebar foot → click "Help and support".
2. Confirm an index page with cards grouped under Start here / Workflows / Research / Tools / Reference.
3. Open at least three articles. Each article should be long-form, no em dashes anywhere, with related links and a back link at the bottom.
4. Press Cmd+K and type "FAQ"; the FAQ article should appear in the palette under Help.

## 14. Auth row in sidebar foot (e448fd8)

1. Open `/sopal-v2`.
2. Sidebar foot should show "Checking sign-in..." briefly, then either:
   - "You are using Sopal as a guest. Sign in to keep your work tied to your account." plus a Sign in button (when no token).
   - Your name plus an Account link and a Sign out button (when signed in).
3. Click Sign out; you should be redirected to `/login?redirect=/sopal-v2`.

## 15. Em dash sweep (d470b5f)

1. Open Stage 1 Intake. The validation error and parse failure messages should not contain em dashes.
2. Open Project Library textarea and Contract textarea; placeholders should not contain em dashes.

## 16. notFoundPage overrides (d1a8df2)

1. Open `/sopal-v2/help/non-existent-slug`.
2. Confirm the page reads "We could not find that help article." with a CTA to the help index, not "Project not found."

## 17. Help articles in palette (6755898)

1. Cmd+K → type "adjudication application end". The "Adjudication Application: end-to-end guide" article should appear under Help.

## 18. Server-side persistence backend (41d55ff)

1. With dev server running, sign in.
2. Open the browser DevTools, take note of your purchase_token from localStorage.
3. Run: `curl -H "Authorization: Bearer <token>" http://localhost:8000/api/sopal-v2/projects`.
4. Confirm a JSON response with `"projects": [...]`.

## 19. Cloud sync wiring (f9eeaf8)

1. Sign in. Make any project edit (e.g. add a row to the dispute table).
2. Wait ~2 seconds for the debounce.
3. Re-run the curl from step 18; the project's `updatedAt` should be a fresh timestamp.
4. Open a different browser (or incognito), sign in with the same account, and open `/sopal-v2`. The project should appear in the sidebar list within a few seconds (pull-missing on boot).

## 20. Settings page (86cf73b)

1. Sidebar foot → Settings (or `/sopal-v2/settings`, or Cmd+K → "Settings").
2. Confirm four cards: Account, Cloud sync, Data and storage, Appearance.
3. Click "Push all local projects to cloud now"; confirm the status line reports the count pushed.
4. Click "Switch to Dark" (or Light); theme should change.

## 21. Project Quick start panel (2310650)

1. Create a brand new empty project.
2. Project Overview should show a "Quick start for a fresh project" card with four numbered steps and deep-link buttons.
3. Open the Contract page and add anything; come back to Project Overview; the Quick start card should be gone.

## Rollback if anything is broken

```bash
git checkout main
git reset --hard pre-overnight-2026-05-10
git branch -D overnight/app-eastbrook-pass
git worktree prune
```
