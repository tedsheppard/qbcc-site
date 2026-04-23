# BIF Act rule set for /claim-check

> **Authoring note.** This file is the single source of truth for the
> /claim-check analysis engine. Edit it freely — the engine reloads on
> every request in production, so changes land without a redeploy. The
> baseline below was seeded from the checklists the engine was already
> running in code; replace, rewrite, or extend rules as needed.

## Rule file schema (how the engine parses this file)

Each rule is a level-2 heading of the form:
`## <ID>: <Check title>`

Then a block of bolded fields, any subset of:

- **Applies to:** comma-separated modes. Accepted modes:
  `payment_claim_serving`, `payment_claim_received`,
  `payment_schedule_giving`, `payment_schedule_received`
- **Act reference:** e.g. `s 68(1)(a) BIF Act`
- **Search query:** keyword query used for the "See relevant decisions" link and for Meilisearch-backed citation surfacing
- **Implementation:** any combination of `semantic`, `user_input`, `deterministic`, `qbcc_lookup` joined with `+` (e.g. `semantic + user_input`)
- **Escalate:** optional. `high-reasoning` lifts the LLM call for this rule to the higher reasoning tier
- **Quote requirement:** plain-English note to the LLM about what to quote

Then criteria sections (Markdown subheadings), any of:

- `**Pass criteria:**` — when the check records "No issues detected"
- `**Warning criteria:**` — when the check flags a potential issue
- `**Fail criteria:**` — when the check flags likely non-compliance
- `**Inputs required:**` — a YAML-ish list of inline questions (see below)

Input questions are authored under `**Inputs required:**` as a list. Each entry is four-space-indented with one of these per-question keys:

```
- id: <stable_id>
  question: "<what to ask>"
  type: date | radio | text | number | licensee_lookup
  options: ["Yes", "No", "Unsure"]        # only for type: radio
  required: true | false                  # default false
  show_if: <other_id> == "<value>"        # optional conditional visibility
  no_future: true                          # optional: for type: date, reject future dates
```

Anything else between rules is treated as comments and ignored by the parser.

---

## PC-001: Identifies the construction work or related goods/services

**Applies to:** payment_claim_serving, payment_claim_received
**Act reference:** s 68(1)(a) BIF Act
**Search query:** identify construction work payment claim s 68
**Implementation:** semantic
**Quote requirement:** quote the passage that identifies the works or related goods and services.

**Pass criteria:**
The document identifies, with enough particularity that the respondent can understand what is being claimed for, the construction work (or related goods and services) to which the claim relates.

**Warning criteria:**
Identification is present but vague, highly generic, or lumps together distinct items in a way that would make it hard for the respondent to respond line-by-line.

**Fail criteria:**
No identification of the work or goods/services is apparent, or the "description" is so bare that s 68(1)(a) is not met on its face.

---

## PC-002: States the amount claimed

**Applies to:** payment_claim_serving, payment_claim_received
**Act reference:** s 68(1)(b) BIF Act
**Search query:** amount claimed payment claim s 68
**Implementation:** semantic
**Quote requirement:** quote the line or figure stating the claimed amount.

**Pass criteria:**
A clear, single claimed amount is stated (either inclusive or exclusive of GST, clearly labelled).

**Warning criteria:**
An amount is stated but is internally inconsistent across the document (e.g. cover page vs schedule) or the GST position is unclear.

**Fail criteria:**
No dollar figure is stated as "the amount claimed".

---

## PC-003: Requests payment of the claimed amount

**Applies to:** payment_claim_serving, payment_claim_received
**Act reference:** s 68(1)(c) BIF Act
**Search query:** request payment payment claim s 68
**Implementation:** semantic
**Quote requirement:** quote the sentence that requests payment.

**Pass criteria:**
The document requests that the respondent pay the claimed amount (post-2018 amendments do not require magic words, but substantive request for payment must appear).

**Warning criteria:**
Request for payment is implied rather than express, or is buried in a way that a reasonable recipient might miss.

**Fail criteria:**
No request for payment appears; document reads as a progress report, estimate, or invoice-in-draft.

---

## PC-004: Served within the permitted time window

**Applies to:** payment_claim_serving, payment_claim_received
**Act reference:** s 75(4) BIF Act
**Search query:** timing payment claim six months s 75
**Implementation:** deterministic + user_input
**Quote requirement:** no quote required.

**Inputs required:**
- id: last_work_date
  question: "When was construction work last carried out under this contract?"
  type: date
  required: true
  no_future: true
- id: served_date
  question: "On what date was (or will) the payment claim be served?"
  type: date
  required: true
- id: is_final_claim
  question: "Is this a final payment claim?"
  type: radio
  options: ["Yes", "No", "Unsure"]
  required: false

**Pass criteria:**
The claim is (or will be) served within 6 months after the construction work to which the claim relates was last carried out (s 75(4)(a)).

**Warning criteria:**
The claim is served close to the 6-month boundary (within 14 days of it), or the user is unsure whether this is a final claim — in which case the longer final-claim windows in s 75(4)(b)–(d) may apply and should be verified against the contract.

**Fail criteria:**
The claim is served more than 6 months after the last day of work and no final-claim exception under s 75(4)(b)–(d) appears to apply.

---

## PC-005: Reference date is valid under the contract

**Applies to:** payment_claim_serving, payment_claim_received
**Act reference:** s 67 BIF Act (reference dates); s 70 BIF Act (progress payment)
**Search query:** reference date validity s 67 s 70
**Implementation:** semantic + user_input
**Escalate:** high-reasoning
**Quote requirement:** quote any reference date text stated in the document.

**Inputs required:**
- id: contract_mechanism
  question: "Does the contract specify reference date mechanics?"
  type: radio
  options: ["Yes — specific mechanism", "No — silent on reference dates", "Unsure"]
  required: true
- id: ref_date_claimed
  question: "What reference date is the claim made for?"
  type: date
  required: false
- id: contract_mechanism_text
  question: "In a sentence, what does the contract say about reference dates?"
  type: text
  show_if: contract_mechanism == "Yes — specific mechanism"
  required: false

**Pass criteria:**
If the contract specifies a reference date mechanism and the claimed reference date matches it, the check passes. If the contract is silent, the default in s 67(2) applies (last day of the named month, unless the Act specifies otherwise) and the claimed reference date is a month-end date that has not been the subject of an earlier claim.

**Warning criteria:**
The user is unsure about the contract mechanism, or the claimed reference date is plausibly valid but requires verification against the actual contract text.

**Fail criteria:**
The claimed reference date does not align with either the stated contract mechanism or the statutory default.

---

## PC-006: Only one payment claim for this reference date

**Applies to:** payment_claim_serving, payment_claim_received
**Act reference:** s 75(4) BIF Act
**Search query:** one payment claim per reference date s 75
**Implementation:** user_input
**Quote requirement:** no quote required.

**Inputs required:**
- id: earlier_claims_served
  question: "Have any earlier payment claims been served for this reference date?"
  type: radio
  options: ["Yes", "No", "Unsure"]
  required: true
- id: earlier_claim_date
  question: "When was the earlier payment claim served?"
  type: date
  show_if: earlier_claims_served == "Yes"
  required: false
  no_future: true

**Pass criteria:**
No earlier payment claim has been served for the same reference date under the same contract.

**Warning criteria:**
The user is unsure; the drafter should check contract records and prior claims before serving.

**Fail criteria:**
An earlier payment claim was served for the same reference date. A second claim for the same reference date is generally barred (s 75(4)) unless it falls within recognised exceptions.

---

## PC-007: Claimant is properly licensed for the work

**Applies to:** payment_claim_serving, payment_claim_received
**Act reference:** s 42 QBCC Act; s 75(2) BIF Act
**Search query:** unlicensed building work QBCC licence s 42
**Implementation:** user_input + qbcc_lookup
**Quote requirement:** no quote required from the document; output quotes the QBCC register record.

**Inputs required:**
- id: claimant_licensee
  question: "Select the claimant's entry on the QBCC Licensed Contractors Register"
  type: licensee_lookup
  required: true
- id: licence_covers_work
  question: "Do the selected licence class(es) cover the scope of work in this claim?"
  type: radio
  options: ["Yes", "No", "Unsure"]
  required: false

**Pass criteria:**
A matching entity is found on the QBCC register, the licence status is Active, and at least one current licence class plausibly covers the work described.

**Warning criteria:**
A matching entity is found but the user is unsure whether the licence classes cover the work; OR the licensee's class coverage is marginal. Note that the register reflects current status as at the dataset's last update, not historical status at the time the work was performed.

**Fail criteria:**
No matching entity found on the register, the matched entity's licence is Cancelled / Surrendered / Suspended, or the licence classes plainly do not cover the work described. Unlicensed building work is barred from recovery under s 42(4) QBCC Act and s 75(2) BIF Act.

---

## PS-001: Identifies the payment claim it responds to

**Applies to:** payment_schedule_giving, payment_schedule_received
**Act reference:** s 76(2)(a) BIF Act
**Search query:** payment schedule identify claim s 76
**Implementation:** semantic
**Quote requirement:** quote the passage that identifies the claim being responded to.

**Pass criteria:**
The schedule identifies the payment claim it responds to (by date, reference, amount, or similar specifics sufficient for the claimant to know).

**Warning criteria:**
Identification is generic (e.g. "your claim") without distinguishing detail — may be sufficient if only one claim is in issue but should be tightened.

**Fail criteria:**
No identification of the claim being responded to.

---

## PS-002: States the scheduled amount

**Applies to:** payment_schedule_giving, payment_schedule_received
**Act reference:** s 76(2)(b) BIF Act
**Search query:** scheduled amount payment schedule s 76
**Implementation:** semantic
**Quote requirement:** quote the scheduled amount figure.

**Pass criteria:**
The schedule states the amount the respondent proposes to pay ("scheduled amount"), including zero if applicable, clearly labelled.

**Warning criteria:**
A figure is stated but is not clearly labelled as the scheduled amount, or the relationship between figures (e.g. net of set-offs) is unclear.

**Fail criteria:**
No scheduled amount is stated. Under s 77 this would typically expose the respondent to the full claimed amount.

---

## PS-003: Reasons for withholding are stated (when applicable)

**Applies to:** payment_schedule_giving, payment_schedule_received
**Act reference:** s 76(3) BIF Act
**Search query:** reasons for withholding payment schedule s 76
**Implementation:** semantic
**Quote requirement:** quote the reasons (or their absence).

**Pass criteria:**
If the scheduled amount is less than the claimed amount, the schedule states the respondent's reasons both for the difference AND for withholding payment. Reasons are specific enough to put the claimant on notice of each ground.

**Warning criteria:**
Reasons are stated but are boilerplate, generic, or plainly under-particularised. Adjudication is limited to reasons in the schedule (s 82(4)); thin reasons narrow the respondent's defence.

**Fail criteria:**
Scheduled amount is less than claimed and no reasons appear; OR reasons given do not correspond to the withholding in any intelligible way.

---

## PS-004: Given within the statutory timeframe

**Applies to:** payment_schedule_giving, payment_schedule_received
**Act reference:** s 76(1) BIF Act
**Search query:** time for payment schedule s 76 fifteen days
**Implementation:** deterministic + user_input
**Quote requirement:** no quote required.

**Inputs required:**
- id: claim_received_date
  question: "When was the payment claim received by the respondent?"
  type: date
  required: true
  no_future: true
- id: contract_timeframe_days
  question: "Does the contract specify a shorter timeframe for the payment schedule (business days)?"
  type: number
  required: false
- id: schedule_served_date
  question: "When was (or will) the payment schedule be given?"
  type: date
  required: false

**Pass criteria:**
The schedule is given within the earlier of (a) any contract-specified period, or (b) 15 business days after the payment claim was received (s 76(1)).

**Warning criteria:**
The schedule is given within but close to the deadline, or the user cannot locate the contract's schedule period.

**Fail criteria:**
The schedule is given outside the permitted period. Under s 77 the respondent becomes liable to pay the claimed amount.

---

## PS-005: Reasons scope — what can be relied on at adjudication

**Applies to:** payment_schedule_giving, payment_schedule_received
**Act reference:** s 82(4) BIF Act
**Search query:** reasons adjudication response limited schedule s 82
**Implementation:** semantic
**Quote requirement:** quote any reasons that may be borderline in scope.

**Pass criteria:**
Every reason the respondent is likely to rely on at adjudication is already articulated in the schedule with enough particularity to be usable.

**Warning criteria:**
Some likely adjudication grounds are hinted at but not articulated. These will be unavailable at adjudication unless fleshed out in the schedule itself.

**Fail criteria:**
Major grounds (e.g. set-off, defective work, abandonment) are plainly missing from the schedule despite appearing relevant on the face of the document.

---

## PS-006: Schedule is a valid payment schedule overall

**Applies to:** payment_schedule_received
**Act reference:** s 76, s 77 BIF Act
**Search query:** valid payment schedule s 76 s 77
**Implementation:** semantic
**Escalate:** high-reasoning
**Quote requirement:** quote any language that casts doubt on validity.

**Pass criteria:**
Taken as a whole, the document would likely be characterised by a court as a valid payment schedule under s 76.

**Warning criteria:**
The document is labelled a payment schedule but has one or more gaps that an adjudicator could seize on — e.g. missing scheduled amount, ambiguous reasons, or late service. Validity is arguable.

**Fail criteria:**
The document fails one or more essential requirements of s 76 to the point it would likely not be a valid payment schedule — giving the claimant an s 77 entitlement.
