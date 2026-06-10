# NSW Security of Payment — Research Spec for Sopal

**Prepared:** 10 June 2026.
**Primary source:** current consolidated text of the Act captured from legislation.nsw.gov.au (consolidation in force **from 20 August 2024 to date**; capture dated 23 January 2026 — the most recent in-force version as at the date of this spec). Regulation text captured from legislation.nsw.gov.au (consolidation current from 1 March 2021; capture dated 24 April 2025; no later substantive amendment found as at June 2026).
**Currency warning:** the *Fair Trading and Building Legislation Amendment Bill 2026* (NSW) is before Parliament (passed Legislative Assembly 11 Feb 2026; before the Legislative Council, not yet assented as at 10 June 2026 per the live NSW Parliament bill page). Its only SOP Act change is terminological — replacing the term "business day" with "working day" to align with the Interpretation Act, **without changing the substantive definition**. Re-check status before launch; all day-counts below are unaffected.

---

## 1. Act metadata

| Item | Detail |
|---|---|
| Full title | **Building and Construction Industry Security of Payment Act 1999 No 46 (NSW)** |
| Short citation form | "s 13 SOP Act (NSW)" / "*Building and Construction Industry Security of Payment Act 1999* (NSW)". Commonly "NSW SOPA" or "the SOP Act". |
| Regulation | **Building and Construction Industry Security of Payment Regulation 2020 (NSW)** (legislation.nsw.gov.au id `sl-2020-0504`; commenced 1 Sep 2020, Sch 2 commenced 1 Mar 2021) |
| Current Act version | In force from **20 August 2024** (last amended by the *Better Regulation Legislation Amendment (Miscellaneous) Act 2024* No 53, which inserted s 8(2)) |
| Administering body | NSW Fair Trading / Building Commission NSW |
| Adjudication model | Claimant chooses a private **authorised nominating authority (ANA)** (s 17(3)(b)) — *not* a state registrar as in QLD |

### Key definitions (s 4(1), quoted verbatim)

**business day** — note the NSW-specific December exclusion (only 27–31 Dec, much narrower than QLD's 22 Dec–10 Jan):

> "business day means any day other than—(a) a Saturday, Sunday or public holiday, or
> (b) 27, 28, 29, 30 or 31 December."

**named month** — not defined in the SOP Act; defined in the *Interpretation Act 1987* (NSW) s 21(1):

> "named month means January, February, March, April, May, June, July, August, September, October, November or December."

**Other s 4(1) definitions (verbatim):**

> "claimed amount means an amount of a progress payment claimed to be due for construction work carried out, or for related goods and services supplied, as referred to in section 13."

> "due date, in relation to a progress payment, means the due date for the progress payment, as referred to in section 11."

> "head contractor means the person who is to carry out construction work or supply related goods and services for the principal under a construction contract (the main contract) and for whom construction work is to be carried out or related goods and services supplied under a construction contract as part of or incidental to the work or goods and services carried out or supplied under the main contract."
> (Note in Act: "There is no head contractor when the principal contracts directly with subcontractors.")

> "principal means the person for whom construction work is to be carried out or related goods and services supplied under a construction contract (the main contract) and who is not themselves engaged under a construction contract to carry out construction work or supply related goods and services as part of or incidental to the work or goods and services carried out or supplied under the main contract."

> "subcontractor means a person who is to carry out construction work or supply related goods and services under a construction contract otherwise than as head contractor."

> "scheduled amount means the amount of a progress payment that is proposed to be made under a payment schedule, as referred to in section 14."

> "exempt residential construction contract means—(a) a construction contract that is connected with an owner occupier construction contract, or
> (b) any other type of construction contract for the carrying out of residential building work that is prescribed by the regulations for the purposes of this definition."

> "owner occupier construction contract means a construction contract for the carrying out of residential building work within the meaning of the Home Building Act 1989 on such part of any premises as the party for whom the work is carried out resides or proposes to reside in."

> "progress payment means a payment to which a person is entitled under section 8, and includes (without affecting any such entitlement)—(a) the final payment for construction work carried out (or for related goods and services supplied) under a construction contract, or
> (b) a single or one-off payment for carrying out construction work (or for supplying related goods and services) under a construction contract, or
> (c) a payment that is based on an event or date (known in the building and construction industry as a “milestone payment”)."

### Reference dates — ABOLISHED for contracts entered into on/after 21 October 2019

The *Building and Construction Industry Security of Payment Amendment Act 2018* No 78 commenced **21 October 2019** and removed the "reference date" concept. For contracts entered into **on or after 21 Oct 2019**, the entitlement is a direct statutory monthly service right under s 13(1A)–(1C) (last day of each named month, or earlier contractual date, or on/from termination). Contracts entered into **before** 21 Oct 2019 remain governed by the old reference-date regime (s 8(1)–(2) as previously in force; *Southern Han* applies). The 2018 Act also: reinstated the s 13(2)(c) endorsement, cut the subcontractor payment cap from 30 to 20 business days, introduced s 13(1C) (termination claims) and s 17A (withdrawal), and removed the owner-occupier exclusion from the Act itself (s 7(2)(b) now "(Repealed)").

### Right to progress payments — s 8 (current, verbatim; note new s 8(2) from 20 Aug 2024)

> "(1) A person who, under a construction contract, has undertaken to carry out construction work or to supply related goods and services is entitled to receive a progress payment.
> (2) A person is not entitled to a progress payment under subsection (1) if the construction contract—(a) does not comply with the Home Building Act 1989, section 4, or
> (b) involves construction work that is residential building work done in contravention of the Home Building Act 1989, section 92."

s 8(2) (inserted by 2024 No 53 Sch 1.2, **commenced 20 August 2024**) bars unlicensed residential builders (HBA s 4) and uninsured residential building work (HBA s 92 — HBCF insurance) from any SOPA progress payment entitlement. A Payment Claim Reviewer for NSW should ask whether the contract involves residential building work and, if so, flag licensing/insurance as a threshold validity issue.

### Penalty units

NSW penalty unit = **$110** (*Crimes (Sentencing Procedure) Act 1999* (NSW) s 17). So 1,000 units = $110,000; 200 units = $22,000.

---

## 2. Payment claims (for Payment Claim Reviewer)

### Validity requirements — s 13(2) (verbatim)

> "(2) A payment claim—(a) must identify the construction work (or related goods and services) to which the progress payment relates, and
> (b) must indicate the amount of the progress payment that the claimant claims to be due (the claimed amount), and
> (c) must state that it is made under this Act."

**Endorsement requirement CONFIRMED CURRENT.** The "made under this Act" statement (removed in 2014 by the 2013 amendments) was **reinstated by the 2018 Amendment Act (Sch 1[10]) with effect from 21 October 2019** and remains in the current consolidation (verified against the in-force text captured January 2026). Every NSW payment claim for a contract entered into on or after 21 Oct 2019 **must** carry the endorsement, e.g. "This is a payment claim made under the *Building and Construction Industry Security of Payment Act 1999* (NSW)". (Contracts entered into between 21 April 2014 and 20 October 2019: endorsement not required.) Note some secondary web sources still wrongly say NSW needs no endorsement — the statute is unambiguous.

**Claimed amount can include** (s 13(3), verbatim):

> "(3) The claimed amount may include any amount—(a) that the respondent is liable to pay the claimant under section 27(2A), or
> (b) that is held under the construction contract by the respondent and that the claimant claims is due for release."

(s 27(2A) = loss/expense from lawful suspension; s 13(3)(b) = retention/security release claims.)

### When a claim can be served — s 13(1A)–(1C) (verbatim)

> "(1A) A payment claim may be served on and from the last day of the named month in which the construction work was first carried out (or the related goods and services were first supplied) under the contract and on and from the last day of each subsequent named month."

> "(1B) However, if the construction contract concerned makes provision for an earlier date for the serving of a payment claim in any particular named month, the claim may be served on and from that date instead of on and from the last day of that month."

> "(1C) In the case of a construction contract that has been terminated, a payment claim may be served on and from the date of termination."

Logic for the reviewer: default monthly entitlement opens on the **last day of each named month** (calendar month); the contract can only bring the date *earlier* in the month, never later; on termination a claim can be served immediately (including a final claim) regardless of the monthly cycle.

### Outer time limit — s 13(4) (verbatim)

> "(4) A payment claim may be served only within—(a) the period determined by or in accordance with the terms of the construction contract, or
> (b) the period of 12 months after the construction work to which the claim relates was last carried out (or the related goods and services to which the claim relates were last supplied),
> whichever is the later."

**12 months** (vs QLD's 6 months) after the work claimed was last carried out, or any longer contractual period.

### One claim per named month — s 13(5)–(6) (verbatim)

> "(5) Except as otherwise provided for in the construction contract, a claimant may only serve one payment claim in any particular named month for construction work carried out or undertaken to be carried out (or for related goods and services supplied or undertaken to be supplied) in that month.
> (6) Subsection (5) does not prevent the claimant from—(a) serving a single payment claim in respect of more than one progress payment, or
> (b) including in a payment claim an amount that has been the subject of a previous claim, or
> (c) serving a payment claim in a particular named month for construction work carried out or undertaken to be carried out (or for related goods and services supplied or undertaken to be supplied) in a previous named month."

So: one claim per named month (unless the contract allows more), but resubmission of previously claimed amounts and claiming for earlier months' work is expressly permitted.

### Supporting statement — s 13(7)–(9) (verbatim, head contractors only)

> "(7) A head contractor must not serve a payment claim on the principal unless the claim is accompanied by a supporting statement that indicates that it relates to that payment claim.
> Maximum penalty—1,000 penalty units in the case of a corporation or 200 penalty units in the case of an individual.
> (8) A head contractor must not serve a payment claim on the principal accompanied by a supporting statement knowing that the statement is false or misleading in a material particular in the particular circumstances.
> Maximum penalty—1,000 penalty units in the case of a corporation or 200 penalty units or 3 months imprisonment (or both) in the case of an individual.
> (9) In this section—supporting statement means a statement that is in the form approved by the Secretary and (without limitation) that includes a declaration to the effect that all subcontractors, if any, have been paid all amounts that have become due and payable in relation to the construction work concerned."

- Applies **only** to a head contractor claiming on the principal (three-tier chain). Subcontractors and direct principal–subcontractor chains: no supporting statement.
- Regulation cl 19 narrows scope: *"The requirement for a head contractor to provide a supporting statement under section 13(7) of the Act relates only to those subcontractors or suppliers directly engaged by the head contractor."*
- Form: the approved form is published by NSW Fair Trading ("Supporting statement" form on the NSW Government security-of-payment pages).
- Penalties: corporation $110,000; individual $22,000 (and 3 months imprisonment for a knowingly false statement).
- **Failure to attach does NOT invalidate the claim**: *TFM Epping Land Pty Ltd v Decon Australia Pty Ltd* [2020] NSWCA 93 — s 13(7) is penal, not a validity condition. The reviewer should flag a missing supporting statement as an offence risk, not as claim invalidity.

### Excluded / special contracts

- s 7(1): Act applies to any construction contract, written or oral, even if expressed to be governed by another jurisdiction's law.
- s 7(2)(a): loan/guarantee/insurance contracts of recognised financial institutions; s 7(2)(b) **"(Repealed)"** (formerly the owner-occupier exclusion — removed by 2018 No 78 Sch 1[3]); s 7(2)(c): contracts where consideration is not calculated by reference to the value of work ("non-value" consideration).
- s 7(3): employee arrangements; lending/guarantee/indemnity provisions.
- s 7(4): work carried out outside NSW.
- s 7(5): contracts prescribed by the regulations — **currently NONE prescribed.** Clause 4 of the 2020 Regulation (which prescribed owner occupier construction contracts under s 7(5)) was **repealed effective 1 March 2021** (cl 4: "Rep 2020 (504), Sch 2[2]").
- **Owner-occupier position CONFIRMED CURRENT (June 2026):** since 1 March 2021 the Act **applies** to owner-occupier construction contracts. But they are prescribed as "exempt residential construction contracts" by Regulation cl 3(1A) (verbatim, including the source's spelling of "contact"):

> "(1A) For the purposes of paragraph (b) of the definition of exempt residential construction contact in section 4(1) of the Act, an owner occupier construction contract is prescribed."

  Consequence: claims CAN be served on owner-occupier homeowners, but the s 11(1A)/(1B) maximum payment terms do not apply — due date is per contract, or 10 business days in default (s 11(1C)).
- s 8(2) (from 20 Aug 2024): no entitlement at all where the contract breaches HBA s 4 (licensing/contract requirements) or the work is residential building work done in contravention of HBA s 92 (insurance) — see §1 above.

### Service (s 31, verbatim heads)

> "(1) Any document that by or under this Act is authorised or required to be served on a person may be served on the person—(a) by delivering it to the person personally, or
> (b) by lodging it during normal office hours at the person’s ordinary place of business, or
> (c) by sending it by post addressed to the person’s ordinary place of business, or
> (d) by email to an email address specified by the person for the service of documents of that kind, or
> (d1) by any other method authorised by the regulations for the service of documents of that kind, or
> (e) in the case of service by a party to a construction contract on another party to the construction contract—in the manner that may be provided under the construction contract."

s 31(2): postal service to a place of business "is taken to have been effected when the document is received at that place". s 31(3): the section is additional to other laws on service. **Note:** contractual deeming clauses that *delay* the effective time of service are void under s 34 (*Roberts Co v Sharvain Facades* [2025] NSWCA 161), and email to an address habitually used on the project can be good service even without express designation (*Rewais v BPB Earthmoving* [2025] NSWCA 103).

### Key case law on payment claim validity (one-line holdings)

1. **Southern Han Breakfast Point Pty Ltd (in liq) v Lewence Construction Pty Ltd [2016] HCA 52** — existence of a reference date under the contract is a precondition to a valid payment claim (governs NSW contracts entered into before 21 Oct 2019; for later contracts the monthly s 13(1A) entitlement replaces reference dates).
2. **Brodyn Pty Ltd v Davenport [2004] NSWCA 394** — only failure of the Act's "basic and essential requirements" (e.g. a payment claim under s 13) vitiates an adjudication (as recast by *Chase Oyster Bar*).
3. **Clarence Street Pty Ltd v Isis Projects Pty Ltd [2005] NSWCA 391** — a claim sufficiently "identifies" the work if it is reasonably comprehensible to a recipient with knowledge of the project; itemised schedules referencing the contract suffice.
4. **Nepean Engineering Pty Ltd v Total Process Services Pty Ltd (in liq) [2005] NSWCA 409** — a claim is invalid for want of identification only if it is not a bona fide/reasonable attempt to identify the work; alleged deficiencies are to be raised in the payment schedule, not by ignoring the claim.
5. **Protectavale Pty Ltd v K2K Pty Ltd [2008] FCA 1248** — Finkelstein J's much-applied test: the claim must be reasonably comprehensible to the party receiving it, judged objectively in context.
6. **Dualcorp Pty Ltd v Remo Constructions Pty Ltd [2009] NSWCA 69** — a claimant cannot repeatedly re-agitate the same adjudicated claim; issue estoppel/abuse of process limits repeat claims (resubmission under s 13(6)(b) is not a licence to re-adjudicate).
7. **Falgat Constructions Pty Ltd v Equity Australia Corporation Pty Ltd [2006] NSWCA 259** — service is effected when the claim is received; a document can simultaneously be an invoice and a payment claim.
8. **Chase Oyster Bar Pty Ltd v Hamo Industries Pty Ltd [2010] NSWCA 190** — compliance with s 17(2) (the second-chance notice window) is jurisdictional; certiorari lies for jurisdictional error.
9. **TFM Epping Land Pty Ltd v Decon Australia Pty Ltd [2020] NSWCA 93** — failure to accompany a head contractor's claim with a s 13(7) supporting statement attracts a penalty but does not invalidate the claim.
10. **EnerMech Pty Ltd v Acciona Infrastructure Projects Australia Pty Ltd [2024] NSWCA 162** — a payment claim need not be narrowly "for construction work"; a claim for money owing on an account under a construction contract (there, recourse to a bank guarantee) can be a valid payment claim.
11. **Manariti Plumbing Pty Ltd v Universal Property Group Pty Ltd [2025] NSWCA 135** — technical defects do not invalidate a claim that reasonably identifies the work and the amount; principals ignore claims at their peril (summary judgment for the claimant).
12. **Claire Rewais and Osama Rewais t/as McVitty Grove v BPB Earthmoving Pty Ltd [2025] NSWCA 103** — service by email to an address repeatedly used for project communications was valid service.
13. **Roberts Co (NSW) Pty Ltd v Sharvain Facades Pty Ltd (administrators appointed) [2025] NSWCA 161** — a contractual clause deeming claims served after a cut-off to be served the next day was void under s 34 (it postponed the statutory timeline); the late payment schedule was invalid.
14. **CBEM Holdings Pty Ltd v Sunshine East Pty Ltd [2025] NSWCA 250** — SOPA payments are provisional only; restitution may be ordered for amounts paid on an interim basis for work not performed.

---

## 3. Payment schedules (for Payment Schedule Reviewer)

### Requirements — s 14(1)–(3) (verbatim)

> "(1) A person on whom a payment claim is served (the respondent) may reply to the claim by providing a payment schedule to the claimant.
> (2) A payment schedule—(a) must identify the payment claim to which it relates, and
> (b) must indicate the amount of the payment (if any) that the respondent proposes to make (the scheduled amount).
> (3) If the scheduled amount is less than the claimed amount, the schedule must indicate why the scheduled amount is less and (if it is less because the respondent is withholding payment for any reason) the respondent’s reasons for withholding payment."

### Time to provide — s 14(4) (verbatim)

> "(4) If—(a) a claimant serves a payment claim on a respondent, and
> (b) the respondent does not provide a payment schedule to the claimant—(i) within the time required by the relevant construction contract, or
> (ii) within 10 business days after the payment claim is served,
> whichever time expires earlier,
> the respondent becomes liable to pay the claimed amount to the claimant on the due date for the progress payment to which the payment claim relates."

**Deadline = the EARLIER of the contract period and 10 business days after service.** (QLD users: NSW is 10 BD, not 15 BD.)

### Consequence of no schedule — s 14(4) + s 15

- s 14(4): the respondent becomes **liable for the full claimed amount** on the due date.
- s 15 (where no schedule AND no payment by the due date), verbatim core:

> "(2) In those circumstances, the claimant—(a) may—(i) recover the unpaid portion of the claimed amount from the respondent, as a debt due to the claimant, in any court of competent jurisdiction, or
> (ii) make an adjudication application under section 17(1)(b) in relation to the payment claim, and
> (b) may serve notice on the respondent of the claimant’s intention to suspend carrying out construction work (or to suspend supplying related goods and services) under the construction contract."

- In s 15(4) debt proceedings the respondent **cannot cross-claim or raise any contract defence**.
- s 16 mirrors this where a schedule was given but the scheduled amount goes unpaid (debt recovery or adjudication under s 17(1)(a)(ii), plus suspension rights).
- Additional gate at adjudication: s 20(2A)–(2B) — a respondent who never provided a schedule (within s 14(4) or the s 17(2)(b) window) **cannot lodge an adjudication response at all**, and any response cannot include reasons not in the schedule.

### s 17(2) second-chance notice regime (verbatim)

> "(2) An adjudication application to which subsection (1)(b) applies cannot be made unless—(a) the claimant has served written notice on the respondent, within the period of 20 business days immediately following the due date for payment, of the claimant’s intention to apply for adjudication of the payment claim, and
> (b) the respondent has been given an opportunity to provide a payment schedule to the claimant within 5 business days after receiving the claimant’s notice."

Where the respondent gave **no schedule and didn't pay**: before adjudicating, the claimant must (a) serve a "s 17(2) notice" within **20 business days after the due date**, and (b) allow the respondent **5 business days** from receipt to provide a (late) schedule. Compliance is jurisdictional (*Chase Oyster Bar*). QLD has no equivalent — this is a major trap for QLD-trained users in both directions.

### Key case law on payment schedules / adequacy of reasons

1. **Multiplex Constructions Pty Ltd v Luikens [2003] NSWSC 1140** (Palmer J) — the leading statement: reasons must be indicated with sufficient particularity to enable the claimant to understand the case it must meet and decide whether to adjudicate; terse reasons may suffice between parties who know the context, but the respondent is confined in adjudication to the reasons stated.
2. **Witron Australia Pty Ltd v Turnkey Innovative Engineering Pty Ltd [2023] NSWCA 305** — a response failing to give any reason directed to a **distinct and substantial component** of the claim (there, variations "to be reviewed later") fails s 14(3) and is not a valid payment schedule for that purpose.
3. **Style Timber Floor Pty Ltd v Krivosudsky [2019] NSWCA 171** — an email offering to explain reasons in the future is not a payment schedule; s 14(3) requires actual disclosure of reasons, even when the respondent proposes to pay nothing.
4. **Quasar Constructions v Demtech Pty Ltd [2004] NSWSC 116** — whether a document is a payment schedule is assessed in substance, but it must actually identify the claim and indicate the amount proposed to be paid (even if nil); a bare dispute that does neither is not a schedule.
5. **Clarence Street Pty Ltd v Isis Projects Pty Ltd [2005] NSWCA 391** — context (parties' prior dealings and project knowledge) informs the adequacy of both claims and schedule reasons.
6. **John Holland Pty Ltd v Roads and Traffic Authority of NSW [2007] NSWCA 19** — s 20(2B) bars new reasons at adjudication; the adjudicator may only consider submissions "duly made" by reference to the schedule.
7. **Pacific General Securities Ltd v Soliman & Sons Pty Ltd [2006] NSWSC 13** — even with no schedule and no response, the adjudicator must still be satisfied the claimed amount is properly payable — no automatic rubber stamp.
8. **Probuild Constructions (Aust) Pty Ltd v Shade Systems Pty Ltd [2018] HCA 4** — no judicial review of an adjudicator's non-jurisdictional error of law; reinforces that the schedule + response is effectively the respondent's only merits forum.
9. **Roberts Co (NSW) Pty Ltd v Sharvain Facades Pty Ltd [2025] NSWCA 161** — schedule served outside the statutory timeframe (computed without the void deeming clause) is invalid; s 34 voids contractual attempts to stretch the schedule window.
10. **Builtcom Constructions Pty Ltd v VSD Investments Pty Ltd [2025] NSWCA 93** — an adjudicator's approach to whether submissions were "duly made" can amount to jurisdictional error depending on how the opinion was formed.

---

## 4. Due dates (for Due Date Calculator)

All counts are **business days** as defined in s 4(1) (excludes Sat, Sun, NSW public holidays, and 27–31 December). The Interpretation Act 1987 s 36 rules apply to computation (exclude the trigger day; if a deadline falls on a non-business day it rolls per s 36(2)). NSW does **not** exclude 22 Dec–10 Jan (that's QLD) — only 27–31 December.

### 4.1 Due date for payment of progress payment — s 11

Decision tree (contract date ≥ 21 Oct 2019):

1. **Is the contract an exempt residential construction contract?** (= owner-occupier contract per Reg cl 3(1A), or a contract connected with one.)
   - YES → s 11(1C): due "(a) on the date on which the payment becomes due and payable in accordance with the terms of the contract, or (b) if the contract makes no express provision with respect to the matter, on the date occurring 10 business days after a payment claim is made under Part 3 in relation to the payment." → **contract date, else 10 BD after claim.**
2. **Is the payment from a principal to a head contractor?**
   - YES → s 11(1A): due on "(a) the date occurring 15 business days after a payment claim is made under Part 3 in relation to the payment, except to the extent paragraph (b) applies, or (b) an earlier date as provided in accordance with the terms of the contract." → **15 BD after claim, or earlier contract date.**
3. **Is the payment to a subcontractor (i.e. by a head contractor, or by a principal contracting directly)?**
   - YES → s 11(1B): "(a) the date occurring 20 business days after a payment claim is made under Part 3 in relation to the payment, except to the extent paragraph (b) applies, or (b) an earlier date as provided in accordance with the terms of the contract." → **20 BD after claim, or earlier contract date.**

- **Contract-earlier rule:** the contract can only accelerate, never extend. s 11(8) (verbatim): *"A provision in a construction contract has no effect to the extent it allows for payment of a progress payment later than the relevant date it becomes due and payable under subsection (1A) or (1B)."*
- **Transitional:** contracts entered into 21 Apr 2014 – 20 Oct 2019: subcontractor cap was **30 BD** (not 20); head contractor cap 15 BD (unchanged). Contracts before 21 Apr 2014: due per contract or 10 BD default. The calculator should branch on contract date if older contracts must be supported.
- The clock runs from when the payment claim "is made" (served) — service rules in s 31; deeming clauses that delay service are void (*Roberts Co v Sharvain* [2025] NSWCA 161).

### 4.2 Payment schedule deadline — s 14(4)

**Earlier of**: (i) the time required by the construction contract, and (ii) **10 business days after the payment claim is served**. (Verbatim text in §3 above.)

### 4.3 Adjudication application deadlines — s 17(3) (all sub-scenarios)

Statutory text (verbatim, relevant limbs):

> "(c) in the case of an application under subsection (1)(a)(i)—must be made within 10 business days after the claimant receives the payment schedule, and
> (d) in the case of an application under subsection (1)(a)(ii)—must be made within 20 business days after the due date for payment, and
> (e) in the case of an application under subsection (1)(b)—must be made within 10 business days after the end of the 5-day period referred to in subsection (2)(b), and"

| Scenario | Trigger | Deadline |
|---|---|---|
| **Schedule given, scheduled amount < claimed amount** — s 17(1)(a)(i) | Receipt of the payment schedule | **10 BD after the claimant receives the schedule** (s 17(3)(c)) |
| **Schedule given, scheduled amount not paid in full by due date** — s 17(1)(a)(ii) | Due date for payment | **20 BD after the due date** (s 17(3)(d)) |
| **No schedule and claimed amount not paid** — s 17(1)(b) | Step 1: serve s 17(2)(a) notice of intention **within 20 BD after the due date**. Step 2: respondent has **5 BD after receiving the notice** to provide a schedule (s 17(2)(b)). | **10 BD after the end of that 5-BD period** (s 17(3)(e)) |

Other s 17 mechanics: application in writing, to an ANA chosen by the claimant, identifying the claim and schedule, with the ANA's fee; copy must be served on the respondent (s 17(5)); ANA must refer to an adjudicator ASAP (s 17(6)). Withdrawal: s 17A.

### 4.4 Adjudication response deadline — s 20 (verbatim core)

> "(1) Subject to subsection (2A), the respondent may lodge with the adjudicator a response to the claimant’s adjudication application (the adjudication response) at any time within—(a) 5 business days after receiving a copy of the application, or
> (b) 2 business days after receiving notice of an adjudicator’s acceptance of the application,
> whichever time expires later."

> "(2A) The respondent may lodge an adjudication response only if the respondent has provided a payment schedule to the claimant within the time specified in section 14(4) or 17(2)(b).
> (2B) The respondent cannot include in the adjudication response any reasons for withholding payment unless those reasons have already been included in the payment schedule provided to the claimant."

**Later of 5 BD after receiving the application and 2 BD after notice of the adjudicator's acceptance** — and no response right at all without a (timely or second-chance) schedule. No extension mechanism (contrast QLD).

### 4.5 Adjudicator's decision deadline — s 21(3) (verbatim)

> "(3) Subject to subsections (1) and (2), an adjudicator is to determine an adjudication application as expeditiously as possible and, in any case—(a) within 10 business days after—(i) if the respondent is entitled to lodge an adjudication response under section 20—the date on which the respondent lodges the response or, if a response is not lodged, the end of the period within which the respondent was entitled to lodge a response, or
> (ii) in any other case—the date on which notice of the adjudicator’s acceptance of the application is served on the claimant and the respondent, or
> (b) within such further time as the claimant and the respondent may agree."

**10 BD** from response lodgment / expiry of the response window (or from notice of acceptance where there is no response entitlement), extendable only by agreement of both parties.

### 4.6 Downstream dates the calculator may want

- **Payment of adjudicated amount — s 23**: due on or before the "relevant date" = **5 BD after the determination is served** on the respondent, or any later date determined by the adjudicator under s 22(1)(b).
- **New application where adjudicator goes silent — s 26**: if no notice of acceptance within **4 BD** of application, or no determination within s 21(3) time, claimant may withdraw and re-apply within **5 BD** of becoming entitled to withdraw (s 26(3)).
- **Suspension of work — s 27**: at least **2 business days' notice** after the relevant non-payment (s 24(1)(b)/s 15(2)(b)/s 16(2)(b) notice must state it is made under the Act).

### 4.7 Business day definition and excluded days (calculator rules)

- Excluded: Saturdays, Sundays, NSW public holidays (state-wide; include the NSW-specific Bank Holiday only if it's a public holiday — it is a bank holiday, not a general public holiday; standard NSW public holidays: New Year's Day, Australia Day, Good Friday, Easter Saturday, Easter Sunday, Easter Monday, Anzac Day, King's Birthday (June), Labour Day (first Monday October), Christmas Day, Boxing Day, plus proclaimed additional days), **and 27, 28, 29, 30, 31 December**.
- Practical effect: the SOPA clock stops from 25 December through 1 January inclusive every year (25–26 Dec public holidays, 27–31 Dec statutory exclusion, 1 Jan public holiday), plus weekend roll-ins and substituted holidays.
- Local-only public holidays (part-day or local government area holidays) are a known edge case — a NSW "public holiday" under the Public Holidays Act 2010 can be local; conservative approach is to compute on state-wide holidays and flag local-holiday risk.
- Pending: the Fair Trading and Building Legislation Amendment Bill 2026 renames "business day" to "working day" **with the same definition** — no calculator change needed when it passes; update display terminology only.

---

## 5. Interest (for Interest Calculator)

### Statutory rule — s 11(2) (verbatim)

> "(2) Interest is payable on the unpaid amount of a progress payment that has become due and payable at the rate—(a) prescribed under section 101 of the Civil Procedure Act 2005, or
> (b) specified under the construction contract,
> whichever is the greater."

In adjudication, the adjudicator must determine "the rate of interest payable" on the adjudicated amount (s 22(1)(c)). Interest on an adjudicated amount can be added to the adjudication certificate (s 24(4)).

### How the prescribed rate works

- s 101 *Civil Procedure Act 2005* (NSW) = interest **after judgment**, "at the prescribed rate", defined by **UCPR r 36.7** (Uniform Civil Procedure Rules 2005 (NSW)).
- UCPR r 36.7: the prescribed rate is **6% above the RBA cash rate** last published before the half-year period commencing **1 January** or **1 July** in which the interest accrues. (This is the nationally harmonised post-judgment rate; the pre-judgment harmonised rate is cash + 4% but that is NOT the s 11(2) rate — s 11(2)(a) points to the s 101/post-judgment rate.)
- The rate therefore resets every 1 January and 1 July. Daily simple interest on the unpaid amount from (and excluding) the due date is the orthodox computation.

### Current and recent rates (NSW Local Court official table, fetched 10 June 2026)

| Period | Post-judgment rate (= s 101 prescribed rate, used by SOPA s 11(2)(a)) | Pre-judgment (reference only) |
|---|---|---|
| **1 Jan 2026 – 30 Jun 2026** | **9.60%** | 7.60% |
| 1 Jul 2025 – 31 Dec 2025 | 9.85% | 7.85% |
| 1 Jan 2025 – 30 Jun 2025 | 10.35% | 8.35% |
| 1 Jul 2024 – 31 Dec 2024 | 10.35% | 8.35% |
| 1 Jan 2024 – 30 Jun 2024 | 10.35% | 8.35% |
| 1 Jul 2023 – 31 Dec 2023 | 10.10% | 8.10% |
| 1 Jan 2023 – 30 Jun 2023 | 9.10% | 7.10% |
| 1 Jul 2022 – 31 Dec 2022 | 6.85% | 4.85% |
| 1 Jan 2022 – 30 Jun 2022 | 6.10% | 4.10% |

**Current rate as at June 2026: 9.60% p.a.** (RBA cash rate 3.60% as last published before 1 Jan 2026, + 6%).

### Where to look it up (for ongoing maintenance)

- NSW Local Court "Interest rates" page (civil jurisdiction) — half-yearly table of pre/post-judgment rates.
- Supreme Court of NSW / District Court practice pages publish the same harmonised rates.
- Council of Chief Justices of Australia and New Zealand "Interest Rates" page (harmonisation source).
- RBA cash rate target (rba.gov.au) — the input; rate for each half-year = cash rate last published before 1 Jan / 1 Jul + 6%.

### Calculator logic

1. Determine due date (Section 4 rules).
2. Interest runs on the unpaid amount from the day after the due date until payment.
3. For each half-year period overlapped, apply max(contract rate, prescribed rate for that period) — the comparison is at the rate level per s 11(2); if the contract specifies a rate, use the greater for each period (where the contract is silent, prescribed rate applies).
4. Simple interest, daily accrual: amount × rate × days/365.

---

## 6. Differences vs QLD BIF Act (traps for a QLD-trained user)

- **Endorsement:** NSW claims **must state "made under this Act"** (s 13(2)(c), reinstated 21 Oct 2019). QLD abolished the endorsement requirement (an invoice can be a payment claim under BIF s 68). A QLD-style unendorsed invoice is **not** a valid NSW payment claim.
- **Reference dates:** QLD still runs on reference dates (BIF s 67); NSW abolished them for contracts from 21 Oct 2019 — entitlement to serve arises on/from the **last day of each named month** (or earlier contract date, or termination).
- **Business days:** NSW excludes only **27–31 December** (plus weekends/public holidays); QLD excludes **22 December – 10 January**. Christmas-period calculations differ materially between the two states.
- **Payment schedule deadline:** NSW **10 BD** (or earlier contract period); QLD **15 BD** (or earlier contract period).
- **No schedule → second chance:** NSW retains the **s 17(2) notice** path (20 BD window to give notice; respondent gets 5 BD to schedule; then 10 BD to apply). QLD has **no second-chance notice** — failing to schedule is an offence and the claimant proceeds directly. Conversely, in QLD the respondent who missed the schedule gets no late opportunity; in NSW the claimant **must** give the s 17(2) notice before adjudicating on a no-schedule claim (jurisdictional — *Chase Oyster Bar*).
- **Claim service window:** NSW **12 months** after work last carried out (or longer contract period); QLD generally **6 months** (longer for final claims). NSW has no special final-claim window beyond s 13(1C)/(4).
- **Maximum payment terms:** NSW caps live in the SOP Act itself — **15 BD** principal→head contractor, **20 BD** to subcontractors, longer terms void (s 11(8)). QLD's caps (15 BD/25 BD) sit in the QBCC Act (ss 67U/67W) for building contracts. NSW exempt residential (owner-occupier) contracts: contract terms govern, default 10 BD.
- **Adjudication application windows:** NSW: 10 BD (schedule shortfall), 20 BD (unpaid scheduled amount), 10 BD after the second-chance window (no schedule). QLD: 30 BD / 20 BD / 30 BD. NSW windows are much tighter for disputed schedules.
- **Adjudication response:** NSW — later of 5 BD after application / 2 BD after acceptance, **no response at all if no schedule was given**, no new reasons (s 20(2A)-(2B)). QLD — 10 BD (standard) / 15 BD (complex) with possible extension, and new reasons are similarly restricted for standard claims via the schedule rule.
- **Decision deadline:** NSW 10 BD from response/expiry. QLD 10 BD (standard) / 15 BD (complex) with party-agreed extensions.
- **Who runs adjudication:** NSW — claimant picks a private **ANA**; QLD — applications go to the **QBCC Adjudication Registrar**.
- **Supporting statement:** NSW head contractors must attach a subcontractor-payment declaration to claims on the principal (s 13(7), $110k corporate penalty). **No QLD equivalent.**
- **Interest:** NSW — greater of contract rate and the **s 101 CPA prescribed rate (RBA cash + 6%; 9.60% for Jan–Jun 2026)**, resetting half-yearly. QLD — greater of contract rate and the Civil Proceedings Act 2011 (Qld) s 59(3) money order debt rate, but for "building contracts" the QBCC Act s 67P penalty rate (10% + RBA 90-day bill rate) applies. Different statutes, different numbers, different reset mechanics.
- **Owner-occupiers:** NSW SOPA **applies** to owner-occupier contracts since 1 Mar 2021 (with relaxed due-date rules as "exempt residential construction contracts"); QLD BIF excludes contracts with resident owners. A NSW homeowner can be served a payment claim and must schedule within 10 BD.
- **Licensing gate (new, 20 Aug 2024):** NSW s 8(2) removes the progress-payment entitlement where the contract breaches HBA s 4 or the work is uninsured residential building work (HBA s 92) — comparable in effect to QLD's QBCC Act s 42, but now expressly inside the NSW SOP Act.
- **Trust accounts:** NSW retention trust applies to head contractors on main contracts ≥ **$20 million** (Reg Pt 2); QLD has the much broader project/retention trust account framework (BIF Ch 2). Don't transplant QLD trust assumptions.
- **Terminology watch:** the pending NSW Fair Trading and Building Legislation Amendment Bill 2026 will rename "business day" to "working day" (same definition).

---

## 7. Sources

Statute and regulation (primary, fetched as Wayback Machine captures of legislation.nsw.gov.au because the live site Cloudflare-blocks automated fetching):

- https://legislation.nsw.gov.au/view/whole/html/inforce/current/act-1999-046 — Act, current consolidation (in force from 20 Aug 2024; captured via https://web.archive.org/web/20260123091038/https://legislation.nsw.gov.au/view/whole/html/inforce/current/act-1999-046 on 23 Jan 2026) — all verbatim Act quotes above.
- https://legislation.nsw.gov.au/view/whole/html/inforce/current/sl-2020-0504 — Regulation 2020, current consolidation (captured via Wayback 24 Apr 2025) — cl 3(1A), cl 4 (Repealed), cl 6 ($20m threshold), cl 19 (supporting statement scope).
- https://legislation.nsw.gov.au/view/whole/html/inforce/current/act-1987-015 — Interpretation Act 1987 (NSW) (Wayback capture) — "named month" definition (s 21(1)).
- https://classic.austlii.edu.au/au/legis/nsw/consol_act/bacisopa1999606/ — AustLII consolidated Act (cross-reference; blocked from this network, used via search snippets only).

Bill / amendment status:

- https://www.parliament.nsw.gov.au/bills/Pages/bill-details.aspx?pk=18857 — Fair Trading and Building Legislation Amendment Bill 2026 status (fetched live 10 Jun 2026: passed LA 11 Feb 2026; in LC, not assented).
- https://www.holdingredlich.com/residential-focus-what-s-new-for-regulation-in-the-nsw-fair-trading-and-building-legislation-amendment-bill-2026 — Bill 2026 SOP Act change (business day → working day), 3 Mar 2026.
- https://piperalderman.com.au/insight/nsw-security-of-payment-act-amendments-loophole-closed-for-unlicensed-residential-builders/ — s 8(2) amendment (Better Regulation Legislation Amendment (Miscellaneous) Act 2024 No 53), commenced 20 Aug 2024.
- https://www.bartier.com.au/insights/articles/security-of-payment-regime-changes-to-commence-21-oct-2019 and https://www.fairtrading.nsw.gov.au/consultation-tool/security-of-payment-reforms — 2018 Amendment Act commencement 21 Oct 2019 and contract-date transition.
- https://www.sparke.com.au/insights/changes-to-nsw-security-of-payment-regulation/ — Regulation Sch 2 changes incl. owner-occupier exemption removal from 1 Mar 2021.
- https://www.nsw.gov.au/housing-and-construction/compliance-and-regulation/security-of-payment/changes-to-laws — NSW Government SOP changes page (supporting statement form, guidance).

Interest rates:

- https://localcourt.nsw.gov.au/about-us/jurisdictions0/civil-jurisdiction/interest-rates.html — NSW Local Court official pre/post-judgment rate table (fetched 10 Jun 2026; source of the 9.60% Jan–Jun 2026 rate).
- https://ccjanz.gov.au/interest-rates — Council of Chief Justices harmonised rates framework (cash rate + 6% post-judgment).
- https://classic.austlii.edu.au/au/legis/nsw/consol_act/cpa2005167/s101.html — Civil Procedure Act 2005 s 101.

Case law and commentary:

- https://www.holdingredlich.com/nsw-security-of-payment-decisions-in-2025-learnings-for-claims-this-year — 2025 NSW decisions roundup (CBEM Holdings [2025] NSWCA 250; Roberts Co v Sharvain [2025] NSWCA 161; Rewais v BPB Earthmoving [2025] NSWCA 103; Manariti Plumbing [2025] NSWCA 135; SE Ware Street v Kwik Flo [2025] NSWSC 160; Builtcom v VSD [2025] NSWCA 93; DECC Credit [2025] NSWSC 826; Black Label v McMenemy [2025] NSWCA 114), 20 Apr 2026.
- https://www.tglaw.com.au/insights/nsw-court-of-appeal-upholds-that-a-valid-payment-schedule-must-answer-each-distinct-and-substantial-component-of-a-payment-claim and https://constructionlawmadeeasy.com/security-of-payment/a-payment-schedule-must-address-each-component-of-a-payment-claim/ — Witron v Turnkey [2023] NSWCA 305.
- https://eralegal.com.au/2019/09/11/the-devil-in-the-detail-of-payment-schedules-under-the-security-of-payment-legislation/ and https://kreisson.com.au/what-constitutes-a-valid-payment-schedule/ — Style Timber v Krivosudsky [2019] NSWCA 171.
- https://cdilawyers.com.au/contractors-can-breathe-a-sigh-of-relief-nsw-court-of-appeal-reinforces-payment-rights-under-security-of-payment-law-and-what-constitutes-a-valid-payment-claim/ — Manariti Plumbing [2025] NSWCA 135.
- https://www.allens.com.au/insights-news/insights/2025/06/tomorrow-starts-today-nsw-supreme-court-finds-deeming-clause-for-service-of-notices-void-under-security-of-payment-legislation/ — deeming-clause service line of authority (first-instance decision behind Roberts Co v Sharvain).
- Classic authorities (settled law, cited from professional knowledge, citations verified format): Southern Han Breakfast Point v Lewence [2016] HCA 52; Brodyn v Davenport [2004] NSWCA 394; Chase Oyster Bar v Hamo Industries [2010] NSWCA 190; Probuild v Shade Systems [2018] HCA 4; Clarence Street v Isis Projects [2005] NSWCA 391; Nepean Engineering v Total Process Services [2005] NSWCA 409; Protectavale v K2K [2008] FCA 1248; Dualcorp v Remo Constructions [2009] NSWCA 69; Falgat v Equity Australia [2006] NSWCA 259; TFM Epping Land v Decon [2020] NSWCA 93; EnerMech v Acciona [2024] NSWCA 162; Multiplex v Luikens [2003] NSWSC 1140; Quasar v Demtech [2004] NSWSC 116; Pacific General Securities v Soliman [2006] NSWSC 13; John Holland v RTA [2007] NSWCA 19.

QLD comparison:

- https://ableadjudication.com.au/qld-interest-rates/ and https://subbiesunited.com.au/67p-motivates-builders-pay/ — QBCC Act s 67P penalty rate mechanics (10% + RBA 90-day bill rate).

Raw captures retained at `docs/sopa-research/raw/` (act.txt, reg.txt, wayback-act.html, wayback-reg.html, interp.html) for verbatim verification.
