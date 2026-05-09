# Sopal v2 — Complex Agent: Adjudication Application

The big plan. This is the architecture and execution plan for the first **Complex Agent** in Sopal v2: a guided, multi-stage adjudication application drafter. It's fundamentally different from the drafting agents (which are single-shot Word-style document editors). This one orchestrates an iterative, item-by-item lawyer workflow and stitches the output into a live master document.

This plan supersedes the user's initial sketch — same vision, fleshed out enough that a Claude Code agent can actually build it.

---

## 1. Where it sits

Sidebar (under each Project):
```
PROJECTS
  └ Contract
  └ Project Library
  └ Assistant
  ├ DRAFTING AGENTS
  │   ├ Payment Claims
  │   ├ Payment Schedules
  │   ├ EOTs
  │   ├ Variations
  │   ├ Delay Costs
  │   └ General Correspondence
  └ COMPLEX AGENTS
      └ Adjudication Application       ← this plan
```

Future Complex Agents (out of scope for v1): Adjudication Response, Court Originating Process, Witness Statement Bundle.

Routes: `/sopal-v2/projects/{id}/complex/adjudication-application`

---

## 2. State model (per project)

```ts
project.complexApps = {
  "adjudication-application": {
    stage: "intake" | "dispute-table" | "rfi" | "draft" | "review",
    deadline: string | null,                  // user-entered ISO date for s 79 lodgement
    documents: {
      paymentClaim:    { name, text, parsed }, // see §3
      paymentSchedule: { name, text, parsed },
      contract:        { name, text } | null,  // optional uploads
      programme:       { name, text } | null,
      drawings:        { name, text }[] | null,
      priorClaims:     { name, text }[] | null,
    },
    disputes: Dispute[],                       // see §4
    jurisdictionalRfis: RFIThread,             // shared thread
    generalRfis:        RFIThread,             // shared thread
    definitions: Record<string, string>,       // central glossary; see §6
    masterHtml: string,                        // assembled live; see §7
    paragraphMode: "section" | "global",       // numbering scheme
    auditLog: AuditEntry[],                    // every RFI Q&A timestamped
    updatedAt: number,
  }
}

type Dispute = {
  id: string,
  item: string,                                // "Variation V14"
  description: string,
  claimed: number, scheduled: number,
  psReasons: string,                           // verbatim from PS
  status: "admitted" | "partial" | "disputed" | "jurisdictional",
  issueType: "variation" | "eot" | "delay-costs" | "defects"
           | "set-off" | "retention" | "prevention" | "scope"
           | "valuation" | "other",
  rfis: RFIThread,
  submissions: string,                         // HTML — live updated
  evidenceIndex: { ref: string, desc: string, location: string }[],
  statDecContent: string,                      // first-person factual statements
  updatedAt: number,
};

type RFIThread = { rounds: { question: string, answer: string|null, askedAt: number, answeredAt: number|null }[] };
```

State lives in localStorage (under the project record). All AI calls receive only the slice they need (per-item slice + master definitions + parties), never the whole blob.

---

## 3. Document parsing (Stage 1 → Stage 2 transition)

User uploads or pastes the **Payment Claim** and **Payment Schedule**. Optional: contract, programme, drawings, prior PC/PS cycles.

Server endpoint `/api/sopal-v2/complex/aa/parse-documents` takes both texts and returns a structured extract:
```json
{
  "parties": { "claimant": "...", "respondent": "..." },
  "contractReference": "...",
  "referenceDate": "YYYY-MM-DD",
  "claimedAmount": 487250.00,
  "scheduledAmount": 0.00,
  "lineItems": [{ "id": "auto-1", "label": "...", "amount": 0, "description": "..." }],
  "psReasons": [{ "lineItemId": "auto-1", "reason": "..." }],
  "psReasonsUniverse": "...",                   // all reasons concatenated — defines the s 82(4) ceiling
  "warnings": [{ "code": "ref-date-future", "message": "..." }]
}
```

This is the canonical extract — it seeds the dispute table in Stage 2 and is shown to the user for verification before they commit.

**s 82(4) BIF Act guardrail:** the parser explicitly captures `psReasonsUniverse` so later stages can flag "new reasons" (arguments not foreshadowed in the PS that the respondent will be barred from raising in their adjudication response). This is surfaced as a UI hint, not a hard block — the claimant might still want to raise the issue.

---

## 4. Dispute table (Stage 2)

Editable table seeded from the parser:

| ✓ | Item | Description | Claimed | Scheduled | Status | Issue type | Rough size |

User actions:
- Edit any cell.
- **Merge** two rows into one (when the PC item splits artificially what is really one dispute).
- **Split** one row into multiple (when one PC item conceals two issues — e.g. variation + delay).
- **Reclassify** issue type.
- **Mark jurisdictional** to lift a row out of the substantive disputes into the jurisdictional section.
- **Add manual dispute** (e.g. a counter to a respondent set-off that wasn't a PC line item).

User clicks **Lock dispute table** to advance to Stage 3. Locking creates one `Dispute` per row + the two shared RFI threads (jurisdictional, general).

The user can re-open the table later — re-opening invalidates the assembled master and forces a re-draft of any affected sections (with a clear "Re-running affected items" prompt).

---

## 5. RFI phase (Stage 3) — three streams

Each stream is its own chat thread. The active stream is selected from the left dispute list (jurisdictional / general / per-item).

### 5a. Jurisdictional RFIs
Generated once, fixed list (the AI may add follow-ups based on answers):
1. **Construction contract** — does the contract fall within s 64 BIF Act?
2. **Reference date** — is the date from the PC valid under the contract / s 67?
3. **Claimant not excluded** — is the claimant excluded under s 88?
4. **PC content compliance** — did the PC identify the work, claim an amount, and request payment?
5. **PC service** — when, by what method, evidence of service?
6. **PS service & timing** — when received, within s 76 window, content compliance with s 69?
7. **Application within s 79 window** — confirm the deadline calculation.
8. **ANA selection** — which ANA is the user nominating?
9. **Excluded amount risks** — any items that may attract exclusion arguments?

Each answer is timestamped. AI may follow up if the answer is incomplete.

### 5b. General RFIs
Open-ended questions that establish the master document's spine:
1. **Project background** — site, scope, value, parties' relationship.
2. **Key personnel** — superintendent, project manager, claimant's representative.
3. **Contract execution** — date, formal contract or letter of agreement, any side deeds.
4. **Defined terms** — what abbreviations / defined terms does the user want used throughout (Contract, Project, Superintendent, etc.)?
5. **Lodgement deadline** — input date for the deadline countdown UI.
6. **ANA preference and adjudicator preferences** — any history with particular adjudicators?
7. **Tone preference** — assertive vs measured.

### 5c. Per-dispute RFIs
Tailored to issue type. The AI keeps asking until it has enough to draft. Templates per type:

#### Variation
- Was there a written direction? Date, author, mode (email / instruction notice / drawing rev)?
- Was prior approval given (clause [#] type)?
- Scope: how does this differ from the contract scope of work?
- Valuation method (Schedule of Rates, day-work, lump sum) and basis?
- Time impact, and is a separate EOT being claimed?

#### EOT
- Qualifying cause of delay (clause / event)?
- When did the contractor become aware?
- Was contractual notice given (date, content)?
- Critical path impact and programme analysis basis?
- Concurrent / parallel delays?
- Mitigation steps?

#### Delay costs / prolongation
- Entitlement basis (clause, breach, prevention)?
- Compensable delay period?
- Quantum methodology (preliminaries, Hudson, Emden, measured-mile)?
- Records (payroll, plant, subbies)?
- Overlap with EOT or variation claims?

#### Defects / set-off
- Was a defect notice issued? When?
- Particulars of the defect — location, nature, evidence?
- Quantification basis (rectification quotes, actual rectification cost)?
- Was the contractor given an opportunity to rectify?

#### Quantum / valuation
- Rates source (Schedule of Rates, market rate, contractor's own quote)?
- Build-up showing labour / plant / materials / overheads / margin?
- Supporting docs?

The RFI engine is **not rules-based** — these are seed templates the AI uses as starting points. It adapts to the answers, asks follow-ups, and stops when it has enough. UX target: 3–8 rounds per item for a typical dispute.

---

## 6. Definitions (centralised)

A separate small panel — **Definitions** — is accessible from any RFI screen. Whenever the AI introduces a defined term in any item's draft (e.g. "Contract", "Site", "Variation Notice"), it's also written to `definitions[]`. The Definitions panel shows the live glossary and lets the user:
- Rename a term ("Variation Notice" → "VN") with propagation.
- Edit a definition's wording.
- Lock a term so the AI can't redefine it.

Implementation: every per-item draft call sends `definitions` in the system prompt and the AI is instructed to use them verbatim.

---

## 7. Master document (Stages 4 + 5 — live)

The master `.html` is assembled from the per-item drafts plus the shared threads, in this rough conventional order. The order is **not rules-based**; the AI proposes the running order based on the matter and the user can override (drag to reorder).

```
1. Cover page
2. Table of Contents (auto-generated)
3. Executive Summary             ← drafted last; refreshed on each item completion
4. Parties and Contract
5. Jurisdiction
   - Construction contract
   - Reference date
   - PC content & service
   - PS content, service & timing
   - s 79 window
   - ANA
6. Background
7. Definitions                   ← propagated from §6
8. Item-by-item Submissions
   - One section per locked dispute
   - User-orderable (default: jurisdictionally critical first, then highest value)
9. Quantum Summary
10. Conclusion and Amount Sought
11. Index of Supporting Evidence
   - SOE-1, SOE-2, … one row per evidence item
12. Statutory Declaration         ← rendered into a separate exportable doc
```

**Live update mechanic:**
- Every per-item RFI answer fires a server call to regenerate that item's submission, evidence index entries, and stat-dec content **in isolation** — but with the master's `definitions`, parties block, and PS reasons in context.
- The master assembler runs after each item update: concatenates the latest per-item HTML, regenerates the ToC, refreshes the executive summary (only if at least N items have completed since the last refresh — to avoid wasted tokens).
- The master is shown live in the right pane, scroll-locked to the section being worked on. A pulse animation marks paragraphs that just changed.

**Numbering:** user picks at lock time — sequential within sections (1, 1.1, 1.2 / 2, 2.1) or globally sequential (1, 2, 3, 4 …). Locked at table-lock time; can't be changed without re-rendering.

**No rule-based templates.** The AI must know what good adjudication submissions look like — assertive, evidence-anchored, structured around the respondent's PS reasons rather than free-floating argument, citation-light but precise where used. We give it strong context (PC, PS, item-specific RFI Q&A, definitions, parties) and a high-quality system prompt; we do not feed it a template fill-in.

---

## 8. Server endpoints (new)

All under `/api/sopal-v2/complex/aa/`:

| Endpoint | Purpose |
|---|---|
| `POST /parse-documents` | Stage 1 → 2: parse PC + PS, return structured extract + warnings |
| `POST /seed-jurisdictional-rfis` | Generate the initial jurisdictional RFI list |
| `POST /seed-general-rfis` | Generate the initial general RFI list |
| `POST /seed-item-rfis` | For an item, generate the first RFI question (issue-type-aware) |
| `POST /answer-rfi` | User submitted an answer; AI may write a follow-up RFI or signal "ready to draft" |
| `POST /draft-item` | Draft / re-draft submissions + evidence index + stat dec for one dispute (returns 3 fields) |
| `POST /assemble-master` | Re-assemble the master document HTML from current per-item state |
| `POST /export-master` | Stream a `.docx` (or `.doc` HTML envelope) of the master |
| `POST /export-stat-dec` | Stream a separate stat-dec `.docx` |
| `POST /export-soe-index` | Stream the SOE index `.docx` |

All endpoints take the project ID + the slice they need. None ever receive the full state.

System prompts are jurisdiction-aware (currently QLD-only — see [docs/multi-jurisdiction-plan.md](multi-jurisdiction-plan.md)).

---

## 9. UX (three panes)

When a dispute is selected:

```
┌──────────────┬───────────────────────────┬───────────────────────┐
│ Disputes nav │ RFI chat (active item)    │ Master doc (live)     │
│              │                           │                       │
│ • Jurisd.    │ AI: Was there a written…  │ 5. Jurisdiction       │
│   [3/9 ✓]    │ User: Yes — email from S. │   5.1 Construction…   │
│ • General    │ AI: Date and content?     │   5.2 Reference date  │
│   [4/7 ✓]    │ User: 3 March 2026 …      │ ...                   │
│ • V14 var.   │                           │ 8.1 Variation V14 ←   │
│ • V15 ssteel │ [composer]                │   ¶ 8.1.1 The Var…    │
│ • Prol.      │                           │   ¶ 8.1.2 …           │
│              │                           │                       │
└──────────────┴───────────────────────────┴───────────────────────┘
```

Top bar:
- **Stage indicator** — Intake → Dispute Table → RFI → Draft → Review (current highlighted)
- **Items completed** — `4/12 items locked`
- **Jurisdictional traffic light** — green/amber/red based on jurisdictional RFI completion
- **Days remaining to lodgement** — countdown if user provided a deadline

Every chat turn has a **Jump to master** affordance that scrolls the right pane to where the answer landed.

Audit log is accessible from a hamburger menu — every Q&A timestamped.

When NO dispute is selected (or pre-Stage 2): right pane shows whatever stage we're at (intake form, dispute table, etc.).

---

## 10. Phased build plan

### Phase A — v1 skeleton (this session)
Goal: end-to-end usable on a real matter for a single dispute. Master assembly is concatenation; AI drafts each item from the RFI history.

- Sidebar: COMPLEX AGENTS group with Adjudication Application
- Page: `/sopal-v2/projects/{id}/complex/adjudication-application`
- Stage 1 UI: PC + PS upload/paste with server-side parse → extract preview
- Stage 2 UI: dispute table (rendered from extract; manual editing of each row)
- Stage 3 UI: per-item chat with seed RFIs → user answers → AI either follows up or signals "ready to draft"
- Stage 4: per-item draft generation (submissions + evidence index + stat dec — three fields in one model call returning JSON)
- Stage 5: master HTML = concatenation of preamble + jurisdictional + general + per-item drafts + conclusion. ToC and exec summary are placeholders ("auto-generated on export").
- Stage 6: master export to `.doc` (HTML envelope, like the existing drafting workspace)
- Persisted in `project.complexApps["adjudication-application"]` localStorage slice

### Phase B — quality pass (later)
- Live ToC generation
- Executive summary refreshed on item completion
- Numbering scheme (sequential vs section)
- Drag-to-reorder dispute sections
- Jurisdictional traffic light + deadline countdown
- New reasons / s 82(4) flagging in the dispute table
- Definitions panel with rename + lock
- Audit log viewer
- Stat dec separate export
- SOE index separate export

### Phase C — polish (later)
- ToC + exec summary as proper AI passes (not concatenation)
- Multi-round RFI templates per issue type with finer follow-up logic
- Cross-reference detection (when item N's submission references item M)
- Track-changes diffs when re-drafting after user edits the master
- Optional inline citations to decisions in Sopal's corpus

---

## 11. Out of scope for v1

- Lodgement to ANA — this app does not lodge.
- Adjudication response drafting (separate Complex Agent later).
- Limitation / time-bar calculations beyond the s 79 deadline countdown.
- Multi-jurisdiction (QLD-only — see multi-jurisdiction-plan).
- Real `.docx` engine (we use the HTML-as-`.doc` shortcut Word opens natively; full python-docx pipeline is Phase C).

---

## 12. Test data — public construction-law contract corpus

Separate task. We need real ANA-published or court-disclosure contracts (or near-equivalents) to dogfood the parser and the AA flow. Candidate sources:

- **AustLII** — adjudication decisions often have the contract reproduced in extracts; useful for testing the parser on real wording.
- **QBCC's adjudication register** — already the source of `qbcc.db`; some decisions reproduce contract clauses verbatim.
- **AS 4000 / AS 4902 freely available drafts** — Standards Australia publishes excerpts; some commentary sites reproduce the conditions.
- **Federal Court of Australia open registry** — judgments occasionally annex contract terms (esp. Contract Act / SOPA appeals).
- **AFCC, MBA Qld, HIA model contracts** — partial public availability.
- **CIDA / Building Code 2024** — Department of Employment and Workplace Relations published Code-compliant model conditions.

Plan: spawn a research task to enumerate sources, verify licensing (we cannot ingest copyrighted commercial contracts), and draft a small ingestion script. Anything ingested should be tagged `source: "public/<source>"` so it never gets confused with user-uploaded contracts.

---

## 13. Anchor files

- New JS: `site/assets/sopal-v2/sopal-v2.js` — `ComplexAdjudicationPage`, `bindComplexAA`, `AA_ISSUE_TYPES`, parser/draft helpers
- New CSS: `site/assets/sopal-v2/sopal-v2.css` — `.aa-shell`, `.aa-three-pane`, `.aa-disputes-nav`, `.aa-rfi-chat`, `.aa-master`
- New server: `routes/sopal_v2.py` — endpoints listed in §8
- Sidebar wiring: `Sidebar()` — new `COMPLEX_AGENT_KEYS` and group

---

## 14. Honest scope note

This is a 2–4 week project to do right. The Phase A skeleton in this session gets us to "a senior lawyer could use it on a small matter and accept it as a starting point". Phases B and C are where it becomes the artefact described above (live ToC, live exec summary, definitions panel, deadline countdown, real `.docx` export, cross-reference detection). I'll ship Phase A live and then we iterate.
