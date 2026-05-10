# Sopal v2 — UX audit, 2026-05-10

Reviewer: senior product designer hat on, the day before a Series A pitch.
Surface walked: every page reachable from the v2 sidebar plus every modal opened, in both light and dark mode, with the "Eastbrook Apartments" project (real local-storage state, 2 contracts, 3 library items, AA workspace seeded to RFI stage). Mobile tested at ~390 px and ~600 px viewports.

Caveat — during the audit window the production frontend (Render) intermittently 502'd for headless requests using the default Playwright UA but worked for real Chrome UAs. Not a UX bug per se, but worth noting that the upstream is sensitive enough to bot-block legitimate-looking traffic and return a generic Render-branded 502 page (a brand surface for a paying customer).

---

## P0 — must fix before any external user sees this

- **[Cmd+K palette] The "search" does nothing.** The palette opens, accepts input, the input value updates in the React state — but the result list never filters. Empty input, "FAQ", "RFI", "stage 2", "Nyholt" (a real recently-viewed decision), "due date" (a real tool name) and "xyzzyplugh" (gibberish) all return the identical 16-item static list (6 Drafting agents + 10 Help/action cards). Confirmed by inspecting DOM under `.palette-list .palette-label` after each query. This is the single feature that ships in your "What's new" notes ("Nine long-form help articles routed under /sopal-v2/help and indexed in the Cmd+K palette") and it is non-functional. Screenshot: `01-cmdk-palette-search-broken.png`.

- **[AA → Stage 2 Dispute table] Money values render as raw integers.** `$ 1269450`, `$ 1254870`, `$ 241840`, `$ 257460`, `$ 171180`, `$ 165380` in the Claimed and Scheduled inputs. The summary bar two rows above formats correctly: `Claimed: $2,094,000  Scheduled: $2,058,000`. Adjudicators read these. PMs cross-check these. An unformatted seven-figure number is the classic "this software was built by engineers" smell. Screenshot: `02-aa-dispute-table-unformatted-numbers.png`.

- **[Mobile] No usable mobile layout.** At a 390 px viewport the sidebar is rendered at the same fixed width as on desktop (~250 px) and the main content area is squeezed into the remaining ~140 px. "Welcome to Sopal v2" title wraps onto three lines, the tools grid still tries to render four columns, "Recently viewed decisions" header wraps. There are media queries in the bundle (down to `(max-width: 380px)`) but the sidebar-collapse breakpoint isn't wired to them. Screenshot: `03-mobile-390-no-responsive.png`. A lawyer paying $200/mo will absolutely open this on their phone in a hearing corridor — and abandon the next time they do.

- **[Sign-in flow] Tossing the user from v2 design system into a v1 marketing-site login page.** Clicking the sidebar "Sign in" link navigates to `/login?redirect=%2Fsopal-v2`, which is the legacy white/green-button form with the old top-nav (`Search · Adjudicators · SopalAI · Interest Calculator · Due Date Calculator · How-To Guide · Contact`) and the old logo mark. Brand whiplash mid-flow. The "Sign In" link in the top-right of that page is also so low-contrast it's nearly invisible. Screenshot: `04-signin-page-v1-design-mismatch.png`.

- **[Sign-in form] Submitting the form with both fields blank silently does nothing.** No HTML5 validation, no inline error, no toast. Form just sits there. The user assumes the page is broken and bounces.

- **[AA → "← Back a stage"] The forward stages in the stepper look clickable but aren't.** Hover state lights up on Stage 3/4/5; mouse cursor changes; no click feedback, no advance. The only way forward is the primary button at the bottom of each stage ("Lock dispute table →"). The stepper should either be visibly disabled for un-reached stages, or it should let the user jump back and forth. Right now it teaches users the click did nothing.

## P1 — visibly amateurish

- **[Em dashes everywhere in your own copy.]** Standing rule: no em dashes in user-visible copy. Spotted, all of them in Sopal-authored copy (not pasted legal text):
  - `/sopal-v2/projects/.../contract` placeholder: "Click or drop one or more PDF / DOCX / TXT files **—** each becomes a separate entry"
  - AA Stage 1 scenario card subtitles: "s 79(2)(a) **—** 30 BD after the LATER of (i) day amount became payable…", "s 79(2)(b) **—** 30 BD after receipt of the payment schedule.", "s 79(2)(c) **—** 20 BD after the day on which payment is due under the contract."
  - AA Stage 2 summary bar: "s 79 scenario: Schedule received **—** scheduled amount LESS than claimed"
  - Due Date Calculator pane heading: "Payment Schedule **—** due date"
  - Interest Calculator collapsible header: "BIF Act s 73 **—** interest on overdue progress payments"
  - Drafting agents (Payment Claims) tagline: "Review or draft payment claim material **—** BIF Act compliance, work identification, dates, service, evidence."
  - What's new modal, "Research Agent: jurisdiction selector" entry: "Other states show a 'Limited support **—** general knowledge only' banner"
  - The empty-value fallback "—" in Project details (`Reference`, `Claimant`, `Respondent` rows of the "t" project) is also an em dash; arguably forgivable, but at minimum inconsistent with the no-em-dash rule.
  Screenshot of em dashes in the live scenario copy: `05-em-dash-in-scenario-copy.png`.

- **[Pluralisation: "1 issues", "1 warnings"]** Payment Claim Reviewer status badges always print the bare count + "s". With one finding it reads `1 issues` / `1 warnings`. Use `Intl.PluralRules` or just `count === 1 ? 'issue' : 'issues'`.

- **[AA → Stage 2 Dispute table] Item column truncates without ellipsis.** "Retention adjustm" / "Margin/Risk" / "Preliminaries" — the "Retention adjustment" cell loses its last three characters with no `…`. Either widen the column or add `text-overflow: ellipsis` so the user knows there's hidden content.

- **[AA → Stage 3 RFI] "RFIs" rendered as "2 RFIS" all-caps.** The pluralised acronym in the right-pane header. CSS uppercase on a string that already has the plural-s preserves it unchanged → "RFIS". Same word ten lines below in the breadcrumb prose is "RFIs". Pick one casing rule.

- **[AA workspace] Progress pill "0/5 drafted · 0%" is misleading.** Even when Jurisdictional shows "1/2 answered" in the items rail, the pill reads 0%. It's measuring drafted artefacts, not RFI progress, but the user reads it as "how far am I in this workflow", which is wrong. Either rename ("0/5 drafts · 0%") or pick a metric that moves with the work the user just did.

- **[AA → Master document modal] Toolbar wraps to a second row and packs eleven actions in.** "Cover page · Introduction · Generate summary · Edit summary · Overarching · Export .doc · Stat dec · Evidence index · Print · Copy as Markdown · Rebuild · ✕". "Stat dec" is unprofessional shorthand for "Statutory declaration". "Generate summary" + "Edit summary" sit next to each other and the user can't tell which to click first. Group: artefact actions on the left (Cover page, Introduction, Overarching, Stat dec, Evidence index), output actions on the right (Export .doc, Print, Copy as Markdown, Rebuild). Drop "Stat dec" → "Statutory declaration".

- **[Project Overview → Documents widget] Same document listed twice (looks like a bug, is data).** "AS 4902 cl 36 (Variations) - Eastbrook HC-2024-EAB" appears as two identical 2,324-char entries; "VAR-001 Latent rock cost summary" appears twice as 1,048 chars. The user did upload them twice (Stage-1 paste + manual library add), but the overview widget should de-dupe by content hash or at minimum show a "Duplicate of …" marker so the user knows it isn't a render bug.

- **[Project Overview] "You are: Claimant" vs Edit-modal "You act for: The claimant".** Two different labels for the same field on the same screen pair. Pick one — preferably "You act for" (better legal voice).

- **[Sidebar collapsed state] No tooltips on icons.** Hover the search/users/sparkle icons after collapsing — nothing. With twelve unlabelled icons in a vertical strip the user has to expand and re-collapse just to remember which is which. Add a `title=` or floating tooltip on hover.

- **[Sidebar collapsed state] Section headers wrap and break.** "Drafting agents" wraps as "Draftin agents" mid-word in the collapsed rail because the column is narrower than the heading. Either hide section headers when collapsed or shorten to "Agents".

- **[Help index] One-card "TOOLS" group looks unfinished.** The 3-column grid contains a single article ("BIF Act calculators") under TOOLS, leaving two thirds of the row blank. Either move the card into a 2-up row with another, or render TOOLS as a single full-width row.

- **[Help index] "Read ↗" arrow icon implies external link but they're internal.** The up-right arrow is the universal "leaves the site" cue. Drop the arrow or swap to a right-arrow `→`.

- **[Help → /help/legal] 404'd — the index links to "Legal disclaimer" but the URL slug is `/help/legal-disclaimer`.** "Legal" alone returns the friendly 404 ("We could not find that help article."). Old URL? Make `/help/legal` redirect to `/help/legal-disclaimer`.

- **[What's new modal] Reads like a CHANGELOG.md, not a customer release-notes doc.** Examples from the live entries: "alert() validation", "debounced PUTs to /api/sopal-v2/projects", "Cmd+Enter to submit", "B/I/U/H1/H2/¶/lists", "Three s 79 BIF Act scenarios", "(NEW)" tag inline in titles, "AA:" and "PC + PS" abbreviations dropped without expansion. PMs and lawyers will see this and the only emotion it produces is "they've shipped a lot, but I have no idea what any of it does for me." Either delete the modal entirely until you can write it like a customer-facing changelog, or strip the engineering vocabulary.

- **[What's new modal — copy]** "Three **s 79** BIF Act scenarios" has an extra space between "s" and "79"; "Three s79 BIF Act scenarios" is what you mean.

- **[What's new modal — text shadows of internal endpoints]** First entry literally exposes the storage key (`The login flow stores a JWT in your browser's local storage under the key "purchase_token"`) and the API path (`/api/sopal-v2/projects/{id}`) in user-facing help. That's not a security bug — both are observable already — but it's a vibe-killer. Help articles for paying customers shouldn't read like internal Notion.

- **[Payment Claim Reviewer] "I'm received".** The radio chooser reads "I'm about to serve" / "I'm received". The second one is grammatically broken. Use "I've received one" or "Reviewing one received from the other side".

- **[Payment Claim Reviewer] Inconsistent button label casing.** "Copy as markdown" here vs. "Copy as Markdown" in the master-document modal. Pick one.

- **[Top-right header] Two nearly-identical buttons next to each other.** A "Dark"/"Light" toggle and a "⌘ K" hint. Both have the same shape, the same border radius, same background, and yet one is interactive and the other is also interactive but does something completely different. Add visual hierarchy — for example, the K button should either look like a tooltip-styled hint with a thin outline, or be replaced by the actual icon button used to open the palette.

- **[Sidebar guest banner] The "You are using Sopal as a guest" panel is glued to the bottom of the sidebar and uses the same colour as the sidebar background.** It reads as part of the chrome, not as a callout. The black "Sign in" CTA inside it is the visually loudest thing in the entire sidebar — louder than "Eastbrook Apartments" — which is exactly the wrong attention hierarchy for a paying customer who's already logged in.

- **[Project list sidebar] A project literally named "t" with reference "AS 4000".** This is your test data, fine, but the UI reveals that a single-letter project name is allowed. The Edit-project modal should require a minimum length (or at least suggest one) — "t" is unidentifiable in the project switcher.

- **[Empty project Overview] The "Notes" textarea has a placeholder "Free-form scratchpad. Chronology, key dates, open questions. Saved automatically." but the textarea's container has no visible label.** The label is the placeholder, which disappears the moment focus lands. Use `<label>` so the field stays self-describing while typing.

- **[Empty project Overview] The Recent conversations card spits out a draft with placeholder syntax exposed.** "Draft Document / Letter / Submission Subject: Follow-Up on Outstanding Payment for Project [Project Name] [Date] [Recipient's Name…". The square-bracket template variables should be substituted (or hidden until substituted) before the doc shows up in the user's recent list.

- **[Settings → Account card] "OFF (SIGN IN)" and "GUEST" status pills are styled with parens.** "OFF (SIGN IN)" is awkward — the parenthetical is a CTA inside a status badge. Use two elements: a status pill ("Off") and a separate "Sign in to enable" link.

- **[Settings → Appearance → "Switch to Dark"]** Duplicates the toggle that already exists in the global header. If you keep both, at least make the page-level button reflect the current state with the same iconography (sun/moon) as the header.

- **[Due Date Calculator → Location dropdown]** The only option is "Brisbane". The product story is "Queensland security-of-payment workspace" — but a single Brisbane option implies regional variation that the dropdown doesn't actually offer. Either remove the dropdown until you have multi-region timing rules, or show the available locations as a flat radio.

- **[Interest Calculator → tabs]** "QBCC s 67P rate" / "Contractual rate" tabs use almost identical visual weight. The active tab has a slightly lighter background, but contrast is sub-AA. Add an underline or a stronger active-state shadow.

- **[Research Agent → jurisdiction]** Yellow dots on NSW/VIC/WA/SA are a nice signal but unlabelled. A user has to click and read the warning banner ("Victoria sources aren't yet integrated. Answers rely on general knowledge only…") to learn what the dot means. Add a tooltip on the dot ("Limited coverage — general knowledge only").

- **[Drafting agents → Payment Claims editor]** "Saved" indicator next to "Copy HTML / Download .doc / Reset" — ambiguous. Saved when? Just now? Two minutes ago? Use a relative timestamp ("Saved 12s ago") or leave it as just an icon when nothing is dirty and a "Saving…" / "Saved 0s ago" only on activity.

- **[Drafting agents → Payment Claims template]** Body still has `[Item description]` and `$[Amount]` literal placeholders in the table after the AI ran. The AI fills the cover details but doesn't substitute the table rows; the user is left with literal square-bracket placeholders that they will paste into a real document. Either fill them or strip the placeholder rows entirely.

## P2 — nice to fix

- **["Welcome to Sopal v2" hero copy]** "Search adjudication decisions, run BIF Act calculators, and manage SOPA workflows project by project." — the "project by project" tail reads awkwardly. Suggest: "Search adjudication decisions, run BIF Act calculators, and run SOPA workflows for each project you take on." Or more terse: "Search decisions, run the BIF Act calculators, and work each SOPA matter from one project workspace."

- **[Recently viewed decisions]** "Pick up where you left off." shows on home for guest users with no signed-in history, anchored to local-storage. Fine for now, but for a brand-new visitor who has only just opened the page, the line reads as a presumption.

- **[Sidebar "What's new" button on home only]** The button only appears in the hero of the home page; on every other route it's gone. If "what's new" is important enough to ship a button for, it deserves a permanent slot in the header next to the Light/Dark and ⌘K controls.

- **[Top-bar "sopal.com.au"]** The bare domain in the top-right is a confusing nav element — does it leave the v2 app? Does it surface marketing? Either label it ("Marketing site →") or replace it with an account-menu trigger.

- **[V2 pill next to logo]** Tiny, currently always present. Once V2 is the default product, drop it; while it's an opt-in beta, make it a more obvious "BETA" pill so users self-attribute weirdness to early-access status.

- **[Project switcher in sidebar]** A single dropdown caret on "Eastbrook Apartments" — clicking the caret opens nothing visible; you have to click the project name. Either make the whole row a click target or move the caret next to the project name with a clear affordance.

- **[Notes textarea]** No visible save indicator. Placeholder claims "Saved automatically" but the user gets zero feedback that the autosave actually fired.

- **[Adjudicator Statistics cards]** The "X% avg award" green pill colour-codes by percentage — good — but the colour scale isn't documented anywhere and a 79% "good" is the same green as a 86% "great". Add a quick legend or tooltip on the pill.

- **[Decision search empty state]** Two side-by-side empty cards ("Enter a search.", "Select a decision.") at the top of the page is unusual; consider one centred state until a search is run.

- **[Master document modal]** No backdrop click-to-close; only the X button or Escape closes it. Backdrop click works on the Edit-project and What's-new modals. Inconsistent.

- **[Master document → live ToC]** The ToC is rendered as a static numbered list at the top of the doc instead of a scroll-following sidebar; for a long doc the user loses orientation halfway through.

- **[Sidebar "Drafting agents" header]** Always lower-case "Drafting agents" but everywhere else section headers are sentence-case-Capitalised ("Research", "Tools", "Projects"). Pick one.

- **["I am" radios in Edit project modal]** The selected option ("The claimant") has a subtle cream background — easy to miss in dark mode where the contrast against the dark surface is weaker than the contrast in light mode.

- **[Settings → Data and storage]** "Browser storage used: 161.8 KB of about 5.00 MB (3%)". The "5.00 MB" is a soft estimate (browsers actually allow ~10 MB in most cases); calling it "about" is honest, but a bar chart would communicate the percentage better than text.

- **[Stage 1 Intake → "Calculate from dates →"]** Tertiary link next to the Lodgement deadline input — the arrow suggests it opens elsewhere. Clicking it inline-expands a calc widget. Drop the arrow.

- **[Drafting agent template values that don't get replaced]** EOTs/Variations/Payment Schedules drafts in the local-storage dump still have `[Contractor name]`, `[Project name]`, `[Contract reference]`, `[Date]` etc. in the saved HTML even after the project's parties and reference are known. The drafts should pick up project context the moment they're created.

- **["Open contract / Open library / Open assistant" buttons on Project Overview]** Three identical-weight buttons. The Documents widget already has a "library items" tile; one of these buttons feels redundant.

- **[Research Agent body] "Sopal research agent" centred title in body has lower-case 'r' for "research agent" but the page title in the header bar is "Research agent" with capital R. Internal vs external title inconsistency.**

- **[Decision Search] Page title in top-bar is "Decision search"; sidebar label is "Decision Search". Same word, different case.**

- **[Adjudicator Statistics] "Most decisions" sort dropdown — only one option visible by default. Add a label "Sort by" outside the dropdown so the dropdown doesn't read as a single-option select.**

- **[Cover-page editor modal]** No min-height on form fields — empty fields stack tightly. Also the helper text mentions "Leave blank to fall back to the project-level value" but there's no visual indication of which fields actually have a project-level fallback (vs which are AA-specific).

- **[AA Stage 2 Reset button]** Top-right red "Reset" with no confirm. Pressing this nukes all AA work in seconds. At minimum prompt "This clears your dispute table, RFIs and master document. Continue?"

- **[Project Overview → Delete button]** Red text "Delete" — same risk. Confirm dialog needed.

- **[AA workspace top bar]** The pill `0/5 drafted · 0%` and the text buttons `← Back a stage` and `Reset` sit at three different visual weights but in one row. Either make them all icon+text or all text — currently the eye doesn't know where the primary action is.

## Surprising things that are GOOD

- **The localStorage-only data model with the explicit "Stored in this browser only." line on every project doc page.** Most products bury "your data is in your browser" inside Settings → Privacy. Sopal v2 puts it in front of the user where the data lives. That's honest and trust-building. Same for Settings → "Cloud sync becomes available once you sign in. Until then, every project lives in this browser only." — the disclosure is the design.

- **The "Drafting agents: when to use which" help article.** Six agents explained in one screen, each with the BIF Act section, the template payload, and the use case. This is the kind of content that competitors charge consultants to produce.

- **Help article URL stability and the friendly 404 ("We could not find that help article. → Open the help index").** The 404 is a single soft card, not a stack-trace, not a Render-branded gateway page, and it offers exactly one useful next action. Treat that as the model for every other empty / error state.

- **The Adjudicator Statistics cards.** Real money figures, real award rates, paginated 301 deep, with sort controls and a filter. This is a research surface a working adjudication lawyer will actually use, and the numbers ARE comma-formatted (`$387,829,960`), which makes the AA dispute-table miss above more glaring.

- **AA Stage 1 → "Parsing extracts the parties, amounts, claim line items, and (if a PS was given) the respondent's reasons. You'll review and edit the result on the next stage."** This is the kind of helper text most AI products ship as "Click to parse." Sopal tells the user what's about to happen and what they'll do next. Keep doing that.

- **AA Stage 2 → Matter details modal helper text.** "These are the core fields the engine and master document use. Edit if the parser got something wrong, or to clean up names (e.g. 'Acme' → 'Acme Builders Pty Ltd'). Leave blank to fall back to the project-level value." Good plain-English explanation of fallback behaviour without using the word "fallback".

- **Live `0/5 drafted · 0%` pill, even though I called it misleading above.** The instinct to put a progress pill in the workspace header is exactly right for a multi-stage workflow; just relabel it.

- **Settings → "Browser storage used: 161.8 KB of about 5.00 MB (3%)"**. Few products tell you how much room you've got left in your local store. This is a small honest detail that plays well with the no-server story.

- **The Master Document modal toolbar packs a lot but `Copy as Markdown` is a power-user feature that competitors don't ship.** Lawyers email Markdown around more than people admit; surface it more prominently.

---

## Verdict

**Score: 6.0 / 10** for "would I show this to a lawyer paying $200/mo".

The shape of the product is impressive — the multi-stage AA workflow, the master document assembly, the drafting-agent library, the decision corpus, the calculators, all in one workspace, with an honest browser-local data model and a help-and-support surface that a real customer can read. The content quality (help articles, Matter-details copy, scenario explanations) is markedly better than the average legal-tech V1.

But the polish layer is missing in three specific ways and a paying user will hit each within their first session:

1. **The Cmd+K palette is broken.** That's the one feature you marketed in your most recent release-note entry, and it's the discoverability layer for everything else. Fixing it is one afternoon and lifts the whole product.
2. **Money in the dispute table doesn't get commas.** A construction lawyer reads `$ 1269450` and silently downgrades you. This is one `Intl.NumberFormat` call.
3. **Mobile is untouched and sign-in throws you back to a v1 page.** Both are first-impression killers in a Series A demo, where the partner will absolutely pull out their phone to "just have a look later."

If I were the partner: I'd write the cheque on the Adjudication Application workflow alone. I'd also withhold a quarter of it until the three items above are fixed. Ship the fixes for those plus the em dashes, the "1 issues" plural, and the dispute-table column truncation, and you're at an 8 — comfortably ahead of the legal-tech median. Ignore them and a sharp customer will close the tab during onboarding.

---

### Screenshots

- `sopal_ux_audit_screens/01-cmdk-palette-search-broken.png` — palette with "xyzzyplugh nothing matches" typed; the result list is the same default 16 items.
- `sopal_ux_audit_screens/02-aa-dispute-table-unformatted-numbers.png` — Stage 2 with raw 7-figure ints in $ inputs while summary bar above shows comma-formatted totals.
- `sopal_ux_audit_screens/03-mobile-390-no-responsive.png` — 390 px viewport showing the desktop sidebar consuming more than half of the screen and the home page main content squashed to nothing.
- `sopal_ux_audit_screens/04-signin-page-v1-design-mismatch.png` — the legacy v1 sign-in page that v2 redirects to.
- `sopal_ux_audit_screens/05-em-dash-in-scenario-copy.png` — AA Stage 1 with em dashes in the s 79(2) scenario card subtitles, against the standing rule.
