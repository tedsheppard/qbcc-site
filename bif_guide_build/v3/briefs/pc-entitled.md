# Element-page brief — pc-entitled
**Title:** The claimant must be entitled to a progress payment
**Breadcrumb:** Requirements of a payment claim
**Anchor id:** `pc-entitled`
**Output file:** `bif_guide_build/v3/pages/page_pc-entitled.html`

---
# Global rules (same for every element-page)

1. **Match the format model exactly.** Read the model below and replicate
   the structure: breadcrumb h4, h2 title, opening paragraph, statute
   excerpt, prose body with topical h3 sub-headings, optional case
   extract block, "What this means on a project" bullet list,
   footnotes ordered list at the bottom.

2. **No headings labelling cases.** Do NOT use sub-headings like
   "Key authority", "Leading authority", "Key case" or "Important
   decision". Discuss the case naturally in prose under a topical
   sub-heading instead.

3. **Section reference style.** Always use `s 80` with a space, not
   `s80`. Apply this to every section reference in your prose, your
   footnotes, and your statute-excerpt summary text. The ONE exception
   is the verbatim statute body itself, where the gazetted heading
   number stays as printed (e.g. `<strong>61 Application of chapter</strong>`).

4. **Plain prose default.** Do NOT use `<div class="analysis">`. The
   only special blocks you may use are:
   - `<details class="statute-excerpt" open>` for verbatim statute
   - `<div class="case-excerpt">` for a single judicial extract (use
     this at most ONCE per page, only where the extract earns its
     place — i.e. it's the leading proposition or contains words the
     courts repeatedly quote)
   - `<ul>` / `<ol>` for lists
   - `<ol class="footnotes">` for the footnotes block at the bottom

5. **Verbatim source.** Statutory text, case names, citations, judicial
   block quotes and adjudication reference numbers must be reproduced
   word-for-word from the source. Curly→straight quote conversion is
   fine; everything else is not. Do NOT silently "correct" what looks
   like a typo or wrong cross-reference in the source — reproduce as
   written.

6. **No hallucination.** If a case, section or proposition is not in
   the source bundled below, do not include it.

7. **Audience and register.** Construction project manager / project
   director. Plain English, active voice, "you" framing where natural.
   Avoid "pursuant to", "notwithstanding", "inter alia", "it is
   submitted", "the authorities establish".

8. **Length.** Aim for 400–800 words of prose plus a tight footnote
   block. Don't pad. Don't truncate either if the source has substance.

9. **Case selection.** Pick at most TWO cases to discuss in-text. The
   rest go in the footnotes with citation, judge in parentheses, and a
   one-line gloss of what the case is for. The two in-text cases should
   be either:
   - the leading appellate authority (HCA, QCA, recent NSWCA), or
   - a case whose facts directly illustrate the point.
   Use AT MOST ONE `<div class="case-excerpt">` block on the page,
   reserved for a passage the courts treat as the canonical formulation.

10. **HTML entities.** Use `&mdash;`, `&ndash;`, `&hellip;`,
    `&ldquo;`/`&rdquo;`, `&amp;` rather than literal characters where
    appropriate.

11. **Output ONE `<section id="SLUG">…</section>` block.** No `<html>`,
    `<head>`, `<body>`, `<style>`, `<script>` wrappers. No markdown
    fences. No preamble or explanation outside the HTML.

---

## Format model — replicate this structure exactly

```html
        <section id="pc-construction-contract">
          <h4 class="breadcrumb-heading">Requirements of a payment claim</h4>
          <h2>A construction contract must exist</h2>

          <p>The BIF Act payment regime only switches on if there is a &ldquo;construction contract&rdquo; between you and the other side. Without one, no payment claim can be made and any adjudication that follows has no foundation.</p>

          <details class="statute-excerpt" open>
            <summary>Section 64 BIF Act &mdash; &ldquo;construction contract&rdquo;</summary>
            <div class="statute-body">
              <p><em>construction contract</em> means a contract, agreement or other arrangement under which 1 party undertakes to carry out construction work for, or to supply related goods and services to, another party.</p>
            </div>
          </details>

          <h3>What counts</h3>

          <p>The definition does three things at once. It captures any contract or agreement; it extends to an &ldquo;other arrangement&rdquo; beyond a contract; and it removes any requirement that the agreement be in writing. An oral agreement is enough, and so is a partly-oral / partly-written exchange &mdash; for example, a series of emails confirming scope and price, or a sequence of agreed meeting minutes.<sup><a href="#fn-pc-cc-1">1</a></sup></p>

          <p>The work being undertaken must be construction work, or the supply of related goods and services in connection with construction work. Those concepts are defined separately at s 65 and s 66 and are dealt with on the next two pages.</p>

          <h3>The &ldquo;other arrangement&rdquo; limb</h3>

          <p>The &ldquo;other arrangement&rdquo; words exist to catch dealings that don&rsquo;t quite fit the legal definition of a contract &mdash; for instance, a head contractor&rsquo;s direction to a subcontractor to perform variation work where the parties are operating informally &mdash; without letting in every situation where someone has done work for someone else. Drawing the line has produced the most case law under this part of the Act.</p>

          <p>The current position is that an &ldquo;other arrangement&rdquo; must give rise to either:</p>

          <ul>
            <li>a legally binding obligation (not necessarily a contract &mdash; an estoppel or other source of legal compulsion can suffice, provided some legal consequence flows from it); or</li>
            <li>a &ldquo;concluded state of affairs&rdquo; involving reciprocity &mdash; that is, both parties have accepted mutual rights and obligations relating to payment or price for the works.</li>
          </ul>

          <p>The most recent appellate guidance is <em>Bettar Holdings Pty Ltd trading as Hunt Collaborative v RWC Brookvale Investment Pty Ltd as trustee for Brookvale Development Trust</em> [2025] NSWCA 242. The pleading in <em>Bettar</em> alleged only a common law contract or, in the alternative, an estoppel preventing denial of a contract. The NSW Court of Appeal held that wasn&rsquo;t enough &mdash; what was missing was a separate allegation that the claimant had &ldquo;undertaken&rdquo; to perform work on some basis other than a common law contract. McHugh JA (Bell CJ and Kirk JA agreeing) put the test for an &ldquo;other arrangement&rdquo; in these terms:</p>

          <div class="case-excerpt">
            <p><span class="para-number">[139]</span> a legally binding obligation&hellip; or a concluded state of affairs&hellip; involving some element of reciprocity or acceptance of mutual rights and obligations relating to payment or price for the works.</p>
          </div>

          <p>His Honour expressly declined to resolve the earlier first-instance conflict between <em>Lendlease Engineering Pty Ltd v Timecon Pty Ltd</em> [2019] NSWSC 685 (Ball J &mdash; required a legally binding obligation) and <em>Crown Green Square Pty Ltd v Transport for NSW</em> [2018] NSWSC 1080 (McDougall J &mdash; adopted a less stringent reciprocity-based test),<sup><a href="#fn-pc-cc-2">2</a></sup> because the pleading in <em>Bettar</em> failed under either test. So which test applies in close cases remains open at appellate level.</p>

          <p>The practical consequence is that a bare quantum-meruit claim &mdash; where work has been done but there is no agreement at all about price or payment terms &mdash; is unlikely to qualify. The &ldquo;arrangement&rdquo; requires reciprocal payment obligations, not merely the fact that work was performed.</p>

          <h3>What this means on a project</h3>

          <ul>
            <li>If you have a signed contract, a construction contract exists. Move on to the next element.</li>
            <li>If your engagement is by exchange of emails, purchase orders or agreed meeting minutes, that&rsquo;s still capable of being a construction contract &mdash; provided there is offer, acceptance and an intention to be bound. Document what was agreed and when; you may need to prove it later.</li>
            <li>For variation or out-of-scope work without a written variation order, capture the instruction and your response in writing as soon as possible. Most &ldquo;other arrangement&rdquo; disputes turn on whether there is evidence of agreement on price or rate.</li>
            <li>A bare quantum-meruit case &mdash; just a claim for the value of work done, with no agreement &mdash; will probably fall outside the Act. You&rsquo;d need to show reciprocal payment obligations, not just the fact of work done.</li>
            <li>If you anticipate a challenge to the existence of a contract, plead the arrangement carefully: the material facts of the agreement, the undertaking to carry out work, and the payment obligation. After <em>Bettar</em> the appellate courts are scrutinising those pleadings closely.</li>
          </ul>

          <ol class="footnotes">
            <li id="fn-pc-cc-1">That a written instrument is not required follows from the words &ldquo;contract, agreement or other arrangement&rdquo; in s 64: see <em>Okaroo Pty Ltd v Vos Construction &amp; Joinery Pty Ltd</em> [2005] NSWSC 45 (Nicholas J); <em>Machkevitch v Andrew Building Constructions Pty Ltd</em> [2012] NSWSC 546 (McDougall J).</li>
            <li id="fn-pc-cc-2"><em>Lendlease Engineering Pty Ltd v Timecon Pty Ltd</em> [2019] NSWSC 685, [29]&ndash;[33] (Ball J); <em>Crown Green Square Pty Ltd v Transport for NSW</em> [2018] NSWSC 1080 (McDougall J). For Queensland, <em>Capricorn Quarries Pty Ltd v Devcon Pty Ltd</em> [2010] QSC 28 (Daubney J) is the leading reading of the contract limb of the BIF Act definition.</li>
          </ol>

        </section>

```

---

## Verbatim BIF Act statute

### `statute/chapter_3/section_070.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 70 — Right to progress payments
From each reference date under a construction contract, a
person is entitled to a progress payment if the person has
carried out construction work, or supplied related goods and
services, under the contract.

```

### `statute/chapter_3/section_071.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 71 — Amount of progress payment
The amount of a progress payment to which a person is
entitled under a construction contract is—
(a) if the contract provides for the matter—the amount
calculated in accordance with the contract; or
(b) if the contract does not provide for the matter—the
amount calculated on the basis of the value of
construction work carried out, or related goods and
services supplied, by the person in accordance with the
contract.

```

### `statute/chapter_3/section_073.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 73 — Due date for payment
(1) A progress payment under a construction contract becomes
payable—
(a) if the contract provides for the matter—on the day on
which the payment becomes payable under the contract;
or
Notes—

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_070.txt`
```
# Annotated BIF Act source — Section 70
# Chapter: CHAPTER 3 – Progress payments
# Section title: Right to progress payments
# DOCX paragraphs: 1702-1865

[2 Com-BIFSOPA Heading 1] SECTION 70 – Right to progress payments
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 70    Right to progress payments
[1.1 BIFSOPA Body Text] From each reference date under a construction contract, a person is entitled to a progress payment if the person has carried out construction work, or supplied related goods and services, under the contract.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.4 Com-BIFSOPA CDI Normal Body Text] The Act amends previous wording under section 12 of BCIPA by simplifying the phrase, “a person is entitled to a progress payment if the person has undertaken to carry out construction work” to “a person who has carried out construction work”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In the Act, the previous wording in relation to the right to progress payment pursuant to section 12 of BCIPA was amended from “a person is entitled to a progress payment if the person has undertaken to carry out construction work” to  “a person who has carried out construction work”.
[2.4 Com-BIFSOPA CDI Normal Body Text] Whilst the explanatory notes do not explicitly explain the amendment to the wording, it does explain that the wording in section 70 “essentially replicates” section 12 of BCIPA. Accordingly, it is highly likely that any earlier decisions regarding this wording will still apply.
[2.2 Com-BIFSOPA Heading 3] 70.1    ‘From’ each reference date
[2.4 Com-BIFSOPA CDI Normal Body Text] In Queensland, there are differing authorities as to whether a payment claim may be made ‘from’ or ‘on’ each reference date.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Reed Constructions (Qld) Pty Ltd v Martinek Holdings Pty Ltd [2009] QSC 345 the latter approach of ‘on’ each reference date was considered by Daubney J to be the correct approach. His Honour concluded (at [26]):
[2.5 Com-BIFSOPA Normal Body Quote] Accordingly, I consider that the correct interpretation is that of the respondent; that the contract contemplates that the progress claim be made on the first of the month. On this interpretation, there can only be one opportunity per month to make a progress claim. However, as was noted by Fryberg J in the Rubikcon case, where a progress claim is not made on the first of the month, the right to claim has not been lost entirely, but merely postponed until the first day of the following month.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, in Tenix Alliance Pty Ltd v Magaldi Power Pty Ltd [2010] QSC 7, Fryberg J considered, and expressly declined to follow Daubney J in Reed Constructions (Qld) Pty Ltd v Martinek Holdings Pty Ltd [2009] QSC 345. Fryberg J said:
[2.5 Com-BIFSOPA Normal Body Quote] On the face of s 12, the opening words from each reference date would seem to suggest the existence of an entitlement on and after each date, which was a reference date. Magaldi submitted however, that that was not the correct approach.
[2.5 Com-BIFSOPA Normal Body Quote] In support of that decision, it relied upon the decision of Daubney J, in Reed Constructions (Qld) Pty Ltd v Martinek Holdings Pty Ltd [2009] QSC 345. As far as I am aware, that is the only case which could give any support to the submission.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] His Honour reached the conclusion that the Act allowed a claim to be served only on each reference date, not on and after the date. He did so because he thought that acceptance of the applicant's contention would mean that more than one progress claim could be served in relation to each reference date.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] With great respect, I am not able to agree with his Honour's interpretation. It is true that the Act does not permit two claims to be made in respect of the one reference date. It is, however a considerable jump from that proposition to conclude that the Act does not permit one claim to be made after the relevant reference date but in respect of that date. It does not seem to me that the words of the Act require such a conclusion.
[2.4 Com-BIFSOPA CDI Normal Body Text] This divergence of views was noted by Douglas J in Simcorp Developments and Constructions Pty Ltd v Gold Coast Titans Property Pty Ltd [2010] QSC 162 where his Honour recognised this ‘difference of opinion’ but found that it was not necessary to reach a conclusion on the facts of the case.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, in McNab Developments (Qld) Pty Ltd v more Construction Services Pty Ltd & Ors [2013] QSC 293, Mullins J considered that the statutory right ‘from each reference date’ under section 12 BCIPA gave the claimant ‘the right to make the statutory claim at any time after’ the date under the relevant contract, subject to the limitation under the then section 17(4) BCIPA (which was amended to become section 17A, and now falls under section 75 of the Act).
[2.4 Com-BIFSOPA CDI Normal Body Text] The equivalent in the NSW Act (section 8(1)) provides for a reference date to be ‘[o]n and from’ each reference date. In this context, a payment claim may be served ‘on and from’ a reference date: see eg Hill (as Trustee for Ashmore Superannuation Benefit Fund) v Halo Architectural Design Services Pty Ltd [2013] NSWSC 865; Walter Construction Group Ltd v CPL (Surry Hills) Pty Ltd [2003] NSWSC 266; All Seasons Air Pty Ltd v Regal Consulting Services Pty Ltd [2017] NSWCA 289.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Rapid Concrete Developments Pty Ltd v Lorem Constructions Pty Ltd [2020] VCC 858, Judicial Registrar Burchell held that nine separate payment claims were invalid because they were made without a valid reference date. In that case, the subcontractor submitted nine payment claims in total: a payment claim for Contact One, four payment claims for Contract Two, and four payment claims for asserted 'additional works’. The analysis first turned to the subcontractor’s submission that there were six separate contracts arising out of emails and oral discussions between it and the builder. Judicial Registrar Burchell was unpersuaded and formed the view that there were only two contracts, such that the payment claims for the “additional works” were properly considered to be variations as the claimed amounts fell within the scope of Contracts One and Two.
[2.4 Com-BIFSOPA CDI Normal Body Text] In respect of the remaining five payment claims, Judicial Registrar Burchell held that a valid reference date did not arise at the time the payment claims were made as required by section 9(1) of the SOP Act (equivalent to section 70 of BIFSOPA). On a proper construction of each contract, the date for payment was “in accordance with our payment terms being on or before 30 days from invoice date”, while a reference date would accrue upon completion of the works. A reference date had therefore not arisen since the claims were submitted before completion of the works. Further, section 9(2)(b) of the Vic Act (equivalent to section 67 BIFSOPA) was not relevant as a statutory reference date would only arise if the construction contract was silent as to a reference date, which it was not on the facts. Finally, Judicial Registrar Burchell considered that a reference did not arise after termination, meaning that three of the payment claims made after the date of termination would nonetheless still be considered invalid.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Roar Fire Systems Pty Ltd v The Construction Studio [2020] VCC 1576, the court was to determine whether a reference date had arisen under the contract for the purpose of section 9 of the Vic Act (the equivalent of section 70 BIFSOPA) and whether the plaintiff’s payment claim, submitted prior to a reference date, was valid. In that case, the plaintiff relied on a clause of the Contract that the date for claims was ‘by the 25th day of each month’ to support its interpretation that a valid reference date may arise on any date between the first to the 25th day of each month. 
[2.4 Com-BIFSOPA CDI Normal Body Text] The Judicial Registrar held that the plain language of the clause operated to require the plaintiff to issue payment claims on a date by the 25th of the relevant month. In finding that the plaintiff had validly issued the claim in respect of a valid reference date, the Judicial Registrar held that:
[2.4 Com-BIFSOPA CDI Normal Body Text] 1) in relation to construction of terms of a contract, one must consider what the ‘reasonable businessman would have understood the terms of the contract to mean’;
[2.4 Com-BIFSOPA CDI Normal Body Text] 2) the objective and reasonable business person would not construe ‘by’ to mean ‘on’. Rather, the ordinary meaning of ‘by’ means ‘on or before’ or ‘not later than’ and indicates a period of time in which an action may be performed, having regard to the defined meaning; and this interpretation would not result in an uncommercial interpretation nor a ‘capricious, unreasonable, inconvenient or unjust’ outcome.
[2.4 Com-BIFSOPA CDI Normal Body Text] In MPA Construction Pty Ltd v Profine Construction Pty Ltd (No 2) [2020] VCC 1254, Woodward J upheld the decision of the Judicial Registrar declaring that a payment claim was valid despite claiming for advanced payments. In that case, the defendant contended that the applicant’s payment claim was partially invalid for claiming advanced payment and work done after the reference date. Woodward J rejected this argument on the basis that a reference date may include work ‘on’ or ‘from’ the reference date under section 9 of the SOP Act (Vic) (equivalent to section 70 of the BIFSOPA Act). Woodward J upheld the principles in Southern Han [Breakfast Point Pty Ltd (in liq) v Lewence Construction Pty Ltd [2016] HCA 52 at [67] and Seymour Whyte Constructions Pty Ltd v Ostwald Bros Pty Ltd (in liq) [2019] NSWCA 11 at [228] that the reference date can encompass work to be undertaken to be done where there is an element of advance payment in the contract.

[2.3 Com-BIFSOPA Heading 4] Concurrent right to claim under the contract
[2.4 Com-BIFSOPA CDI Normal Body Text] A construction contract will generally provide for the right to give a progress claim in accordance with the contract, as distinct from the statutory right to make a payment claim under the Act. One issue is in relation to the timing of progress claims made under the contract. While the statutory right has been held to be ‘from’ the reference date, judicial decisions have grappled with the proper construction, with a divergence of opinion emerging as a result of the drafting of contract clauses.
[2.4 Com-BIFSOPA CDI Normal Body Text] A narrow approach to a contractor’s right to claim under the contract was adopted by the Queensland Court of Appeal in John Holland Pty Ltd v Costal Dredging and Construction Pty Ltd [2012] 2 Qd R 435. The court held that a distinctively restrictive clause had the effect that a claim could be submitted “only on each reference date.” This decision remains authoritative for contracts which expressly adopt restrictive language.
[2.4 Com-BIFSOPA CDI Normal Body Text] A divergence of opinion emerged in Reed Constructions (Qld) Pty Ltd v Martinek Holdings Pty Ltd [2009] QSC 345. In that case, cl 37.1 and Item 28 of the contract provided:
[2.5 Com-BIFSOPA Normal Body Quote] “The Contractor shall claim payment progressively in accordance with Item 28. An early progress claim shall be deemed to have been made on the date for making that claim. Each progress claim shall be given in writing to the Superintendent and shall include details of the WUC [Work Under Contract] done and may include details of other moneys then due to the Contractor pursuant to the provisions of the Contract.”
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] “Time for progress claims: first day of each month for Work Under Contract to the last day of that month to which the claim relates.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In considering whether the progress claim served two days late was valid, Justice Daubney held that the progress payment had to be served “on” the first day of the month, going as far to import the word “on” to describe when the progress claim had to be submitted where it did not already appear in the language of the clause (emphasis original):
[2.5 Com-BIFSOPA Normal Body Quote] [23] … The critical words found in Item 28 of the Schedule provide the time for making progress claims is the “first day of each month for Work Under Contract to the last day of that month to which the claim relates” (emphasis added). There is no evidence to suggest that this contractual provision was altered by agreement between the parties.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] [26] Accordingly, I consider that the correct interpretation is that of the respondent; that the contract contemplates that the progress claim be made on the first of the month. On this interpretation, there can only be one opportunity per month to make a progress claim. However, as was noted by Fryberg J in the Rubikcon case, where a progress claim is not made on the first of the month, the right to claim has not been lost entirely, but merely postponed until the first day of the following month.
[2.4 Com-BIFSOPA CDI Normal Body Text] Part of Daubney J’s judgement was distinguished a year later in Tenix Alliance P/L v Magaldi Power P/L [2010] QSC 7, however, the distinguishing feature was that there was no reference date supplied in the contract. Tenix also considered the statutory entitlement as opposed to the contractual entitlement.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Queensland District Court expressly rejected the approach to contractual entitlement contemplated in Reed Constructions in the decision of CKP Constructions Pty Ltd v Gabba Holdings Pty Ltd [2016] QDC 356. In that decision, McGill SC DCJ remarked in obiter that the clause before the court did not properly deal with the situation where a progress claim is made late (after the day which was stipulated in Item 33 of the contract), unlike John Holland. In such circumstances, his Honour held that the entitlement arose on the day stipulated under the contract (at [35]):
[2.5 Com-BIFSOPA Normal Body Quote] “My conclusion is that the correct interpretation of the contract is that a right to make a claim arose on the 25th of each month, but that there was nothing in the contract which provided that that right had to be exercised, if at all, only on that day, or that if a claim in respect of that month was submitted after that day, it stood as an early claim for the following month.  In effect, I am being asked to imply such a restriction into the contract, and I consider there are no grounds for doing so.  In my opinion, the right to make a claim under cl. 37.1 does not have to be exercised on the day specified, so long as it is clear that what is being claimed involves an exercise of the right in relation to the particular specified day for a month.”
[2.2 Com-BIFSOPA Heading 3] 70.2    Preconditions to progress payments and reference dates
[2.4 Com-BIFSOPA CDI Normal Body Text] There is a distinction drawn in the case law between:
[2.4 Com-BIFSOPA CDI Normal Body Text] (1) preconditions imposed on the making of a payment claim where a reference date has arisen; and
[2.4 Com-BIFSOPA CDI Normal Body Text] (2) where a reference date is conditional and will not arise until certain conditions are fulfilled.
[2.4 Com-BIFSOPA CDI Normal Body Text] This notion was explored in Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd & Anor [2014] QSC 293 at [66], Applegarth J said:
[2.5 Com-BIFSOPA Normal Body Quote] There is a distinction between two categories of case. The first is where a reference date has arisen and the contract purports to require a payment claim to meet certain conditions in making a payment claim or provides that if certain conditions are not met, the payment claim is ineffective and the reference date is deferred. In such a case, the statutory reference date is not conditional upon compliance with the condition, such as the delivery of a statutory declaration. Also the contract is ineffective to defer what would have otherwise been the contractor’s statutory entitlement to a progress payment from a reference date. The second category of case, and the one with which I am concerned, is where a reference date has not arisen because the contract purports to provide that no reference date will arise until certain conditions are fulfilled.  Authorities about the first category of case do not determine what the position should be in the second category.
[2.5 Com-BIFSOPA Normal Body Quote] (footnotes omitted)
[2.3 Com-BIFSOPA Heading 4] Preconditions to the making of a payment claim
[2.5 Com-BIFSOPA Normal Body Quote] A provision in a contract which operates to defer what would otherwise have been a claimant’s statutory entitlement to a progress payment from a reference date ascertained in accordance with the Act will be void. A provision of this kind “purports to … modify … or otherwise change the effect of a provision of this Act, or would otherwise have the effect of … modifying, or otherwise changing the effect of a provision of this Act”: John Holland Pty Ltd v Coastal Dredging & Construction Pty Ltd [2012] QCA 150; [2012] 2 Qd R 435, 443 [21] (Fraser JA, White JA and Peter Lyons J agreeing), where such a requirement was reconciled with s 99(2)(b) BCIPA, now section 200 of the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] Citing John Holland Pty Ltd v Coastal Dredging & Construction Pty Ltd [2012] QCA 150; [2012] 2 Qd R 435 at [21], Applegarth J in McConnell Dowell Constructors (Aust) Pty Ltd v Heavy Plant Leasing Pty Ltd [2013] QSC 269 said at [26]: 
[2.5 Com-BIFSOPA Normal Body Quote] If a “reference date” arises under a construction contract and a party has a statutory entitlement to a progress payment under the contract in relation to that reference date, then a contractual provision which would operate to defer what otherwise would have been the party’s statutory entitlement will be ineffective to do so since it would change the effect of a provision of the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also BHW Solutions Pty Ltd v Altitude Constructions Pty Ltd [2012] QSC 214 where Mullins J applied John Holland Pty Ltd v Coastal Dredging & Construction Pty Ltd [2012] QCA 150; [2012] 2 Qd R 435 to hold there was no impediment to summary judgment where the respondent sought to rely on the absence of a statutory declaration with the payment claim, which was imposed as a precondition to payment under the contract.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] John Holland Pty Ltd v Coastal Dredging & Construction Pty Ltd [2012] QCA 150; [2012] 2 Qd R 435. Clause 12.6 of the construction contract contained a number of warranties from the claimant to the respondent on payment claims, including a warranty that payment claims will be in the format the respondent requires, including the provision of a statutory declaration. Failure to comply with the requirements of clause 12.6 had the effect that the payment claim was void, and the reference date is to be the same day on the following month. The Court of Appeal held clause 12.6 was void under section 99(2)(b) of BCIPA, now section 200 of the Act.
[2.3 Com-BIFSOPA Heading 4] Preconditions to reference dates
[2.4 Com-BIFSOPA CDI Normal Body Text] There is a distinction between a precondition to the entitlement to a progress payment, and a precondition to the arising of a reference date.
[2.4 Com-BIFSOPA CDI Normal Body Text] In John Holland Pty Ltd v Coastal Dredging & Construction Pty Ltd [2012] QCA 150; [2012] 2 Qd R 435, after referring to provisions under sections 13, 14, 15, 17 and 18 of BCIPA, Fraser JA said:
[2.5 Com-BIFSOPA Normal Body Quote] None of those provisions bear upon the date upon which the statutory entitlement to progress payments accrues. Bearing in mind the statutory object and the role of s 12 and the definition of “reference date” in giving effect to that object, those provisions are incapable of justifying an implication that the date upon which the statutory entitlement to a progress payment accrues may be qualified by contractual provisions other than those captured by the unambiguous terms of the definition of “reference date”. 
[2.4 Com-BIFSOPA CDI Normal Body Text] The words ‘other than those captured by the unambiguous terms of the definition of “reference date”’ indicates that a statutory entitlement to a progress payment can be qualified to this extent.
[2.4 Com-BIFSOPA CDI Normal Body Text] In State of Qld v T & M Buckley Pty Ltd [2012] QSC 265, this distinction was considered, but, on the facts of the case, as the precondition in question was held to condition the contractual right to a progress payment, and not the statutory right, the relevant precondition to a reference date did not fall to be determined in the context of section 99 of BCIPA (and section 200 of the Act).
[2.4 Com-BIFSOPA CDI Normal Body Text] Whether a particular clause providing for a precondition to a reference date in question will be invalidated by operation of section 200 of the Act will depend on an assessment of the reference date against the purpose of the Act: see Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd & Anor [2014] QSC 293 (per Applegarth J):
[2.5 Com-BIFSOPA Normal Body Quote] The purpose of the Act and the role which s 99 plays in constraining the contractual freedom of parties to modify the entitlements which the Act confers means that the parties do not have an unconstrained freedom to contract about when a reference date will arise. The Act does not allow the contract to impose onerous conditions which make a reference date more of a theoretical possibility than an actuality. Likewise, the example given by Hodgson JA, of a contract which provides for an annual reference date, would seem to be inimical to the purpose of the Act and invalidated by s 99.
[2.5 Com-BIFSOPA Normal Body Quote] The extent to which a particular condition is contrary to the Act or purports to change the effect of the Act depends upon its content and practical consequences.  In assessing the validity of such a condition a useful inquiry is whether it facilitates or impedes the purpose of the Act.  A provision which has the purpose of regulating contractual rights to progress payments may not be appropriate to condition a statutory right to a progress payment.  A condition is likely to be contrary to the Act or unjustifiably change the effect of the Act’s provisions where it does not facilitate a statutory entitlement to progress payments or the resolution of payment claims made under the Act.  This is likely to be the case where the condition impedes the making of a payment claim with no corresponding benefit in achieving the Act’s purpose.
[2.5 Com-BIFSOPA Normal Body Quote] The inquiry into validity requires the identification of the condition or conditions in the absence of which there would be a statutory entitlement to a progress payment.  Even a condition which has some utility in a contractor making a payment claim and receiving a progress payment may be excessively onerous and be invalid because of its unjustified effect in denying a party what otherwise would be a statutory entitlement.  A condition which has no significant utility in terms of the scheme created by the Act may be invalid, not because it is particularly onerous, but because it impedes a statutory entitlement without any corresponding benefit.
[2.4 Com-BIFSOPA CDI Normal Body Text] Applegarth J revisited the question of what constitutes a valid reference date in BRB Modular Pty Ltd v AWX Constructions Pty Ltd & Ors [2015] QSC 218. Referring to his decision in Lean Field Developments, his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] I had occasion to consider the operation of s 99 of the Act in Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd and observed that in assessing the validity of a condition, a useful inquiry is whether it facilitates or impedes the purpose of the Act. That observation was not intended to place a gloss upon s 99 or to be a substitute for the words of the statute. I accept BRB’s submission that a contractual provision could not be contrary to the Act simply because it does not further the objects of the Act. In considering whether a provision of a contract is contrary to the provisions of the Act or otherwise is ineffective by reason of s 99, it is necessary to be specific about how the Act and its operation are said to be affected by the contractual provision. As I observed in Lean Field, the extent to which a particular condition is contrary to the Act, or purports to change the effect of the Act, depends upon its content and practical consequences.  A provision which has the purpose of regulating contractual rights to progress payments may not be appropriate to condition a statutory right to a progress payment.  The condition is likely to be contrary to the Act or unjustifiably change the effect of the Act’s provisions “where it does not facilitate a statutory entitlement to progress payments or the resolution of payment claims made under the Act”. This is likely to be the case where the condition impedes the making of a payment claim with no corresponding benefit in achieving the Act’s purpose.
[2.5 Com-BIFSOPA Normal Body Quote] If, absent a contractual provision, a contractor would have a statutory entitlement to make a claim for a progress payment under the Act, then the provision will have the effect of excluding, modifying, restricting or otherwise changing the effect of the Act.  The position is otherwise where, even absent the provision, there would be no entitlement under the Act, for example, because no reference date will have arisen. 
[2.4 Com-BIFSOPA CDI Normal Body Text] On the facts of the case in BRB Modular, Applegarth J held that the precondition in question (a statutory declaration on payments to subcontractors and suppliers) was void under section 99 BCIPA as it imposed a burden on the claimant without any ‘real utility in advancing the purpose of [BCIPA]’ (at [70]).
[2.4 Com-BIFSOPA CDI Normal Body Text] In East End Projects Pty Ltd v GJ Building and Contracting Pty Ltd [2020] NSWSC 819, Ball J considered the validity of a contractual provision which required a “draft progress claim” to be issued before a reference date would arise under the NSW Act and whether such a provision unduly restricted the operation the Act. More specifically, the construction contract in question stipulated the following requirements in order for a reference date to arise:
[2.4 Com-BIFSOPA CDI Normal Body Text] (1) The claimant was to provide a “draft progress claim” in writing to the superintendent on or before the 25th day of each month “for WUC done to the last day of that month”.
[2.4 Com-BIFSOPA CDI Normal Body Text] (2) The superintendent, after receiving the draft progress claim, to then issue a preliminary assessment of the amount which was due in relation to the draft progress claim. 
[2.4 Com-BIFSOPA CDI Normal Body Text] (3) The claimant to then issue a progress claim in final form seven business days after the draft progress claim was issued. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Ball J distinguished the decision in Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd & Anor [2016] 1 Qd R 30; [2014] QSC 293 and held that the requirement for a draft payment claim to be served before a reference date arose was a “serious restriction” on the operation of section 8 of the NSW Act (equivalent to section 70 BIFSOPA). Ball J observed that the draft and final payment claims in this case will or are likely to cover the same work period and that this was an important distinction from the facts in Lean Field Developments. However, his Honour held that the precondition was still void because the requirement to serve a draft payment claim “on the 25th day of a month or before”, meant that the claimant would lose its entitlement to serve a payment claim at all for that month if it failed to serve the draft by the 25th day of that month. As a result, Ball J held that the effect of such provision was a serious restriction on the operation of the NSW Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Statutory declarations. State of Qld v T & M Buckley Pty Ltd [2012] QSC 265. Reference dates under the relevant construction contract accrued monthly, on the monthly anniversary of the commencement of work. The contract required a completed statutory declaration, in the form attached to the contract, to be provided prior to the making of a payment claim under the contract. The Court held that the reference date under BCIPA was the monthly anniversary, regardless of whether the claimant had provided a completed statutory declaration.
[2.6 Com-BIFSOPA bullet] In BRB Modular Pty Ltd v AWX Constructions Pty Ltd & Ors [2015] QSC 218, it was held that a precondition to a reference date of a completed and signed statutory declaration in the form under a schedule to the contract on payments to subcontractors and suppliers, was not a valid precondition to a reference date under BCIPA.
[2.6 Com-BIFSOPA bullet] In J Hutchinson Pty Ltd v Glavcom Pty Ltd [2016] NSWSC 126, Ball J held that a provision which made the occurrence of a reference date conditional on the provision of a statutory declaration by a subcontractor concerning the payment of other amounts owed by the subcontractor, was a provision which sought to add an additional condition on the right to obtain a progress payment, and did not further the purpose of the NSW Act, and was therefore void pursuant to s 34 of the NSW Act. 
[2.6 Com-BIFSOPA bullet] ‘Forming an opinion’. In Castle Constructions Pty Ltd v Ghossayn Group Pty Ltd [2017] NSWSC 1317, Stevenson J held that ‘a contractual provision which states that a contractor’s right to payment is contingent upon a third party forming an opinion, or certifying a particular state of affairs, is one which purports to or has the effect of excluding, modifying or restricting the operation of the NSW Act or might reasonably be construed as an attempt to deter a person from taking action under the NSW Act and is therefore void by reason of section 34 of the NSW Act. In consideration of previous authorities, Stevenson J held that a precondition to a reference date will or may be invalidated where it:
[2.4 Com-BIFSOPA CDI Normal Body Text] imposes conditions on the occurrence of a reference date (Hutchinson Pty Ltd v Glavcom Pty Ltd [2016] NSWSC 126 at [26]);
[2.4 Com-BIFSOPA CDI Normal Body Text] modifies or restricts the circumstances in which a contractor is entitled to a progress claim (Hutchinson Pty Ltd v Glavcom Pty Ltd [2016] NSWSC 126 at [26]);
[2.4 Com-BIFSOPA CDI Normal Body Text] inordinately delays or effectively prevents a reference date from arising (Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd [2014] QSC 293 at [55]);
[2.4 Com-BIFSOPA CDI Normal Body Text] unjustifiably impeaches the making of a payment claim or renders the statutory entitlement practically illusory (Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd [2014] QSC 293at [68]);
[2.4 Com-BIFSOPA CDI Normal Body Text] imposes onerous conditions which make a reference date more of a theoretical possibility than an actuality (Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd [2014] QSC 293at [73]); or does not facilitate a statutory entitlement to a progress payment (Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd [2014] QSC 293at [74]).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Wärtsilä Australia Pty Ltd v Primero Group Ltd  [2020] SASC 162, Stanley J quashed an adjudication determination on the basis that there was no reference date under section 8 of the SOP Act (SA) (equivalent), to s 70 of BIFSOPA because the claimant failed to comply with the contractual requirements for a valid reference date to arise (i.e. the requirements for ‘SW Completion’ to ‘provide’ and make documents ‘available for inspection’). In this case, the subcontract provided the following relevant requirements for ‘SW Completion’:

[2.5 Com-BIFSOPA Normal Body Quote] ‘(2) the tests, inspections and communications required by this subcontract (including Schedule 3) to have been carried out before SW Completion have been carried out, passed and the results of the tests, inspections and commissioning provided to [Wärtsilä]
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] (8) the completed quality assurance documentation … is available for inspection by [Wärtsilä] at the Facility Land’ 

[2.4 Com-BIFSOPA CDI Normal Body Text] On 28 February 2020, Primero emailed Wärtsilä a Dropbox link to the relevant documents, but Wärtsilä was unable to completely download the documents. Primero proceeded to adjudication and the adjudicator determined Primero’s payment claim was valid, awarding $15,269,674.30. Key to the adjudicator’s determination was that the payment claim was supported by a reference date of 28 February 2020 when the Dropbox link was sent to Wärtsilä. On the application before Stanley J, his Honour held:

[Normal] in relation to SW Completion item (2), “the provision of the hyperlink merely provided a means by which Wärtsilä was permitted to download the documents stored in the cloud. Until it did so, those documents had not been provided.”

[Normal] in relation to SW Completion item (8), “the hyperlink did not amount to making the documents available for inspection… because until all the documents were downloaded, they were not capable of being inspected at the facility land.”

[Normal] Stanley J, relying on the case of Conveyor & General Engineering v Basetec Services & Anor [2015] 1 Qd R 265, found that sending the Dropbox link to the documents was insufficient to satisfy the requirements of SW Completion. Accordingly, the adjudication determination was quashed because it was not made with reference to a valid payment claim and the $15M award to Primero was nullified.
[2.2 Com-BIFSOPA Heading 3] 70.3    ‘Under a construction contract’
[2.3 Com-BIFSOPA Heading 4] Work preceding the creation of the contract
[2.4 Com-BIFSOPA CDI Normal Body Text] ‘Work which was not carried out under the contract, because it preceded the making of the contract, would appear to be outside the operation of BCIPA’: Walton Construction (Qld) Pty Ltd v Robert Salce [2008] QSC 235, [22] (McMurdo J).
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer also to the commentary under the Schedule 2 definition of ‘construction contract’.
[2.4 Com-BIFSOPA CDI Normal Body Text] Example
[2.6 Com-BIFSOPA bullet] Novation as a new construction contract. Ball Construction Pty Ltd v Conart Pty Ltd [2014] QSC 124. Parties to a construction contract entered into a deed of assignment, assigning the obligations of one party to an incoming builder. Subsequent to this deed of assignment, the incoming party included amounts in a payment claim that were referable to the period before the deed of assignment. The Court held that the deed of assignment had created a new construction contract, such that the incoming party did not have an entitlement in relation to work done before the deed of assignment; the work had not been carried out by the incoming party.
[2.3 Com-BIFSOPA Heading 4] Payment claim made after termination of the contract
[2.4 Com-BIFSOPA CDI Normal Body Text] Where a contract provides for reference dates, they will not otherwise accrue after termination of the contract: Walton Construction (Qld) Pty Ltd v Corrosion Control Technology Pty Ltd & Ors [2011] QSC 67, [42]-[45] (Peter Lyons J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Walton Construction (Qld) Pty Ltd v Corrosion Control Technology Pty Ltd & Ors [2011] QSC 67, Peter Lyons J concluded:
[2.5 Com-BIFSOPA Normal Body Quote] The use of the expression “under a construction contract” found in the Queensland definition makes it somewhat more difficult to conclude that a reference date occurs after termination.  There is then no longer a contract “under” which there might be a reference date. The conclusion that a reference date does not occur after termination of a contract is, in my view, also consistent with the general nature of the payments for which provision is made by the BCIP Act, that is to say, payments which are of a provisional nature, made over the life of the contract.
[2.5 Com-BIFSOPA Normal Body Quote] The second difference which I have noted between the two definitions is also of significance.  The language used in the BCIP Act gives greater primacy to the provisions of the contract dealing with the making of a claim for a progress payment than does the language of the New South Wales Act.
[2.5 Com-BIFSOPA Normal Body Quote] For these reasons, I am not prepared to adopt the statement from the judgment of Hodgson JA in Brodyn as reflecting the effect of the definition of the expression “reference date” in the BCIP Act.
[2.5 Com-BIFSOPA Normal Body Quote] In my view, the contract provides for reference dates, by both enabling their identification, and by providing in effect that there is no right to make a progress claim after the contract is terminated under clause 44.10, with the consequence that no further reference date of this kind would then accrue.
[2.5 Com-BIFSOPA Normal Body Quote] The approach of Peter Lyons J in Walton Construction (Qld) Pty Ltd v Corrosion Control Technology Pty Ltd & Ors [2011] QSC 67 was adopted by the Chief Justice in McNab NQ Pty Ltd v Walkrete Pty Ltd & Ors [2013] QSC 128.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, where a reference date has previously accrued prior to the date of termination, but this reference date is unused, the claimant does not lose an entitlement to make a payment claim following termination of the contract: see McNab Developments (Qld) Pty Ltd v MAK Construction Services Pty Ltd & Ors [2013] QSC 293, [15]-[18] (Mullins J).
[2.4 Com-BIFSOPA CDI Normal Body Text] This distinction, between Walton Construction (Qld) Pty Ltd v Corrosion Control Technology Pty Ltd & Ors [2011] QSC 67 and McNab Developments (Qld) Pty Ltd v MAK Construction Services Pty Ltd & Ors [2013] QSC 293, was acknowledged by Douglas J in Kellett Street Partners Pty Ltd v Pacific Rim Trading Co Pty Ltd and Ors [2013] QSC 298.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also McConnell Dowell Constructors (Aust) Pty Ltd v Heavy Plant Leasing Pty Ltd [2013] QSC 269.
[2.3 Com-BIFSOPA Heading 4] Termination for convenience
[2.4 Com-BIFSOPA CDI Normal Body Text] Under the NSW Act, it has been held that, where a contract was terminated for convenience, and as a matter of construction the reference dates under the contract did not survive termination, the claimant was unable to claim for work done between the last reference date under the contract and the date of termination: see Patrick Stevedores Operations No 2 Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2014] NSWSC 1413 where Ball J held:
[2.5 Com-BIFSOPA Normal Body Quote] That leaves the question of what happens in relation to work done after the last reference date under the contract and before termination. In my opinion, the contract still provided a reference date in respect of that work at the time the work was performed because, at that time, the contract was still on foot. Consequently, there is no room for the operation of s 8(2)(b). The fact that the contract was terminated before the reference date in respect of that work arrived does not alter the position. It simply means that no reference date in respect of that work can arise. That result does not seem to me to be inconsistent with the purpose of the Security of Payment Act. As I have said, the purpose of the Act is not to ensure that a contractor is paid for work as soon as it is done. Nor do I think it is to ensure that a contractor is paid everything it is owed promptly. Rather, the purpose of the Act is to provide a practical mechanism to ensure that contractors receive progress payments for the work that they do. It seems to me that purpose is achieved even if, because of the way in which the contract and Act operate, the contractor is not entitled to use the mechanism provided for by the Act to recover a payment for work done shortly before termination.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Omega House Pty Ltd v Khouzame [2014] NSWSC 1837.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, on one view, per Ann Lyons J in Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd & Ors [2012] QSC 373, a termination for convenience clause which provides for certain amounts to be payable in the event of termination, may, of itself, create a reference date. Her Honour rejected a submission from the respondent on the basis of Walton Construction (Qld) Pty Ltd v Corrosion Control Technology Pty Ltd & Ors [2011] QSC 67.
[2.2 Com-BIFSOPA Heading 3] 70.4    ‘Supply related goods and services’
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to the commentary under section 66, above.
[2.2 Com-BIFSOPA Heading 3] 70.5    For work done to that month
[2.4 Com-BIFSOPA CDI Normal Body Text] In Spankie & Ors v James Trowse Constructions Pty Ltd [2010] QCA 355, the relevant construction contract provided that a reference date arose on ‘the 28th day of each month for WUC done to the end of that month’. The Court of Appeal held that the claimant was entitled to make a progress claim ‘on the 28th day of each month in respect of work done up to the end of that month, whether or not the work was done in that month’. On this issue, Fraser JA said (at [24]) that:
[2.5 Com-BIFSOPA Normal Body Quote] …there is no necessary relationship between the reference date upon which a claim is made and the time when the work the subject of the claim was carried out.
[2.4 Com-BIFSOPA CDI Normal Body Text] In St Hilliers Property Pty Limited v ACT Projects Pty Ltd and Simon Wilson [2017] ACTSC 177, there was confusion as to whether there was an available reference date in respect of a payment claim that was served. Acting Judge Walmsley held that a new reference date could accrue when no construction work had been done in the current month, only where a valid payment claim had not been submitted in the previous month but where work had been done in that previous month. As the builder in this case had not performed any work in the current month and had already claimed for works done in the month prior, there was no reference date. Acting Judge Walmsley followed the established principle set out in Southern Han Breakfast Point Pty Ltd v Lewence Construction Pty Ltd & Ors [2016] HCA 52 (that a reference date is a prerequisite to lodging a valid payment claim) and held that as there was no reference date, the claim was invalid. Interestingly, the adjudicator was ordered to pay 20% of costs here “because of his active role in the defence of his position.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Hanson Construction Materials Pty Ltd v Brolton Group Pty Limited [2019] NSWSC 1641, Ball J observed, in obiter, that a payment claim supported by a reference date could relate to work done after the reference date in respect of which the payment claim was served. His Honour said at [32] that:
[2.5 Com-BIFSOPA Normal Body Quote] “[t]here is no requirement in s 13 that the work in respect of which the payment claim is made must be performed before the claim is made. As the High Court held in Southern Han, there is a requirement that the date from which a progress claim may be made (the reference date) has occurred. Where the contract simply specifies that a claim for a progress payment may be made on a particular date each month, without tying that date to the performance of particular work, then, assuming the date has occurred, it is primarily a question for the Adjudicator what work may be made the subject of a progress claim in accordance with the contract by reference to that date.”
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Fyntray Constructions Pty Ltd v Macind Drainage & Hydraulic Services Pty Ltd [2002] NSWCA 238, [53] (Heydon JA).
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer also to the further commentary under section 75, below.
[2.2 Com-BIFSOPA Heading 3] 70.6    ‘Progress payment’
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to the commentary under the section 64 definition of ‘progress payment’
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer also to the further commentary under section 72, below.
[2.2 Com-BIFSOPA Heading 3] 70.7    Progress payments and insolvency
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court in Façade Treatment Engineering Pty Ltd (in liq) v Brookfield Multiplex Constructions Pty Ltd [2016] VSCA 247 held that section 9(1) of the VIC Act (section 70 of the Act) does not create an entitlement to progress payments for persons who are in liquidation. 
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court cited the judgment of Young CJ in Brodyn Pty Ltd v Dasein Constructions Pty Ltd [2004] NSWSC 1230 and said that where Young CJ ‘focused on the purposes underlying the [NSW] Act, rather than the text used in the provisions of the [NSW] Act’, the starting point for statutory construction should instead be the statutory text. The Court said:
[2.5 Com-BIFSOPA Normal Body Quote] Section 1 states that the purpose of the Act ‘is to provide for entitlements to progress payments for persons who carry out construction work or who supply related goods and services under construction contracts’ (emphasis added).  Thus, the very first provision of the BCISP Act suggests that the protection it offers is targeted to persons who are acting pursuant to a construction contract.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] Unlike s 1, s 9(1) refers to a person who ‘has undertaken’ to do certain things, rather than a person who is doing those things in the present.  But in its ordinary meaning, the word ‘undertake’ connotes an expectation of performance.  It is also significant that s 9(1), in creating a person’s entitlement to progress payments under the BCISP Act, emphasises that person’s undertaking to perform actions under the construction contract, rather than, say, the person’s status as a party to the contract.  The text of s 9(1) accords importance to the actions of carrying out construction work and supplying related goods and services.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court interpreted s 9(1) of the VIC Act in the following terms:
[2.5 Com-BIFSOPA Normal Body Quote] In our view, therefore, s 9(1) creates an entitlement to progress payments only for persons who have undertaken to, and continue to, carry out construction work or supply related goods and services.  The term ‘the claimant’ used throughout pt 3 is commensurately limited.  Consequently, the payment regime in pt 3 of the BCISP Act is not available to companies in liquidation, since such companies cannot carry out construction work or supply goods and services, and thus do not satisfy the requirements for ‘a claimant’.
[2.4 Com-BIFSOPA CDI Normal Body Text] In the decision at first instance, Vickery J held that inconsistency arose between section 16 of the VIC Act (equivalent to section 77 of the Act) and s 553C of the Corporations Act 2001 (Cth).
[2.4 Com-BIFSOPA CDI Normal Body Text] Relevantly, his Honour held:
[2.5 Com-BIFSOPA Normal Body Quote] A company to which s 553C of the Corporations Act 2001 applies, subject to s 553C(2), is precluded from entering any judgment pursuant to s 16(2)(a)(i) of the BCISP Act in respect of the debt due to it under that Act, and is further precluded from relying on s 16(4)(b) as a bar to a respondent under the BCISP Act from bringing any cross-claim against the company or raising any defence by way of set-off in relation to matters arising under the relevant construction contract.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court of Appeal affirmed the decision of Vickery J in respect of the statutory inconsistency issue in obiter:Once a company has gone into liquidation, and where there are mutual dealings so that s 553C is engaged, a payment claim cannot be enforced by means of a summary judgment under s 16(2)(a)(i) of the VIC Act, and there is no scope for the ousting of the cross-claims or defences under s 16(4)(b) of the VIC Act. The decision in Façade Treatment Engineering Pty Ltd (in liq) v Brookfield Multiplex Constructions Pty Ltd [2016] VSCA 247 was followed by Reid DCJ in Tantallon Constructions Pty Ltd (in liq) v Santos GLNG & Anor [2016] QDC 324 where his Honour commented in obiter that:
[2.5 Com-BIFSOPA Normal Body Quote] In my view, consideration of the provisions of BCIPA and of the decisions of Neller and of Facade make it clear that, certainly upon the plaintiff being placed into liquidation, and very probably upon its being earlier placed into administration, its ability to make and progress a claim under BCIPA was lost.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Seymour Whyte Constructions Pty Ltd v Ostwald Bros Pty Ltd (In liquidation) [2019] NSWCA 11, the NSW Court of Appeal unanimously held that the NSW Act operates for the benefit of claimants in liquidation. In overturning the decision of Stevenson J at first instance, where it was held that a company in liquidation is unable to exercise the rights afforded to a “claimant”, the Court of Appeal held that there is nothing in the NSW Act to support a finding that a company in liquidation is unable to meet the preconditions to progress payments in s 8(1) of the NSW Act (equivalent to s 70 of the Act). In the first instance, the Court held that s 8(1) of the NSW Act: 
[2.5 Com-BIFSOPA Normal Body Quote] “read in light of in light of the analysis in Southern Han, creates an entitlement to a progress payment… on satisfaction of two conditions:
[2.5 Com-BIFSOPA Normal Body Quote] 1.    A person has undertaken to carry out construction work under the contract; and
[2.5 Com-BIFSOPA Normal Body Quote] 2.    A reference date under the contract has arisen.”
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court of Appeal held that: 
[2.4 Com-BIFSOPA CDI Normal Body Text] 1.    the first condition is a reference to a contractual undertaking, not to the physical performance of the work; and
[2.4 Com-BIFSOPA CDI Normal Body Text] 2.    in respect of the second condition, a reference date can arise regardless of whether a claimant is actually continuing to carry out construction work under the contract, as they are set by “contractual force”. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Accordingly, the NSW Court of Appeal held that there was nothing in the language of the NSW Act to support a finding that either of the two conditions could not be met where a company had entered into liquidation.
[2.4 Com-BIFSOPA CDI Normal Body Text] In MPA Construction Pty Ltd v Profine Construction Pty Ltd (No 2) [2020] VCC 1254, Woodward J upheld the decision of the Judicial Registrar declaring that a payment claim was valid despite claiming for advanced payments. In that case, the defendant contended that the applicant’s payment claim was partially invalid for claiming advanced payment and work done after the reference date. Woodward J rejected this argument on the basis that a reference date may include work ‘on’ or ‘from’ the reference date under section 9 of the SOP Act (Vic) (equivalent to section 70 of the BIFSOPA Act). Woodward J upheld the principles in Southern Han [Breakfast Point Pty Ltd (in liq) v Lewence Construction Pty Ltd [2016] HCA 52 at [67] and Seymour Whyte Constructions Pty Ltd v Ostwald Bros Pty Ltd (in liq) [2019] NSWCA 11 at [228] that the reference date can encompass work to be undertaken to be done where there is an element of advance payment in the contract.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Hastie Group Limited (in liq) v Multiplex Constructions Pty Ltd (Formerly Brookfield Multiplex Constructions Pty Ltd) [2020] FCA 1824, Middleton J noted in obiter that the ratio’s in Façade and Seymour White were inconsistent. His Honour observed that where a respondent to a payment claim is in liquidation, the failure to stay the claim, or alternatively the execution of any judgment that might be based upon it, would have the effect of transforming a payment that is intended to be interim in nature into a permanent payment. Middleton J explained that this would be contrary to the intention of the security of payment legislation that payments be on account.

[2.4 Com-BIFSOPA CDI Normal Body Text] In Kennedy Civil Contracting Pty Ltd (Administrators Appointed) v Richard Crookes Construction Pty Ltd; In the matter of Kennedy Civil Contracting Pty Ltd [2023] NSWSC 99, a subcontractor who was in liquidation, executed a deed of company arrangement pursuant to s 445(d)(1) of the Corporations Act 2001 (Cth). The deed preserved the subcontractor’s rights to recover unpaid payment claims under the NSW SOP Act, as s 32B provides that a corporation in liquidation cannot serve a payment claim or take action to enforce a payment claim, or an adjudication determination. Ball J held that it is not an abuse of process for companies in liquidation to enter into a deed of company arrangement so that their affairs fall outside the scope of s 32B of the NSW SOP Act. There is no equivalent s 32B in the BIFSOPA, however in Tantallon Constructions Pty Ltd (in liq) v Santos GLNG [2016] QDC 324, the Queensland District Court denied a claimant, its ability to make a progress claim under the former Queensland Security of Payment Act (BCIPA), due to the claimant being in liquidation.

```

### `annotated/section_071.txt`
```
# Annotated BIF Act source — Section 71
# Chapter: CHAPTER 3 – Progress payments
# Section title: Amount of progress payment
# DOCX paragraphs: 1866-1905

[2 Com-BIFSOPA Heading 1] SECTION 71 – Amount of progress payment
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 71    Amount of progress payment 
[1.3 BIFSOPA level 1 (CDI)] The amount of a progress payment to which a person is entitled under a construction contract is—
[1.4 BIFSOPA level 2 (CDI)] if the contract provides for the matter—the amount calculated in accordance with the contract; or
[1.4 BIFSOPA level 2 (CDI)] if the contract does not provide for the matter—the amount calculated on the basis of the value of construction work carried out, or related goods and services supplied, by the person in accordance with the contract.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 71.1    Section 71(a)
[2.3 Com-BIFSOPA Heading 4] Relationship to the amount certified under the contract
[2.4 Com-BIFSOPA CDI Normal Body Text] Under BCIPA, ‘amount calculated under the contract’ was not the amount that was in fact calculated by the Superintendent (or other certifying person) under the construction contract. Such a submission was expressly rejected in Hervey Bay (JV) Pty Ltd v Civil Mining and Construction Pty Ltd & Ors [2008] QSC 58, [24] (McMurdo J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Abacus v Davenport & Ors [2003] NSWSC 1027, McDougall J said:
[2.5 Com-BIFSOPA Normal Body Quote] It cannot be correct to say that an adjudicator under the Act is bound by the terms of any progress certificate issued, under a contractual regime of the kind that I have described, by the architect or someone in the position of the architect. That would mean that an adjudicator could not make a determination that was inconsistent with a certificate that was (for example) manifestly wrong. Indeed, it would mean that an adjudicator could not make a determination that was inconsistent with a certificate that had been issued in bad faith, or as the result of fraudulent collusion to the disadvantage of the builder.
[2.5 Com-BIFSOPA Normal Body Quote] Further, as was submitted for Renascent, it is not uncommon for building contracts to provide that it is the proprietor, or someone who is the proprietor’s alter ego or agent, to occupy the certifying role that, under the form of contract presently under consideration, is occupied by the architect. In those circumstances, if the submission for Abacus be correct, an adjudicator would be bound by a certificate issued by a proprietor, or by its agent or alter ego, in bad faith, or one that flatly and obviously disregarded the rights of the builder.
[2.5 Com-BIFSOPA Normal Body Quote] Such a construction would undermine in a very serious way the evident intention of the legislature that is embodied in the Act. It would enable an unscrupulous proprietor (either by itself, if the contract so permitted, or with the collusion of an unscrupulous certifier) to set at nought the entitlement to progress payments that the Act provides and protects.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Transgrid v Siemens Ltd [2004] NSWSC 87, [62]-[63] (Master Macready).
[2.4 Com-BIFSOPA CDI Normal Body Text] Rather, the ‘amount calculated under the contract’ as expressed by section 13(a) BCIPA was the amount ‘calculated on the criteria established by the contract’: Hervey Bay (JV) Pty Ltd v Civil Mining and Construction Pty Ltd & Ors [2008] QSC 58, [24] (McMurdo J).
[2.4 Com-BIFSOPA CDI Normal Body Text] The mere adoption of a superintendent’s certificate, without more, is ‘inconsistent with the adjudicator’s statutory task of independently assessing the evidence of value’: SSC Plenty Road Pty Ltd v Construction Engineering (Aust) Pty Ltd [2015] VSC 631, [116] (Vickery J). This reasoning was approved by the Victorian Court of Appeal in SSC Plenty Road Pty Ltd v Construction Engineering (Aust) Pty Ltd [2016] VSCA 119 where the court considered the matters which the adjudicator must consider in making a determination. In dismissing the appeal, the court held:
[2.5 Com-BIFSOPA Normal Body Quote] “Requiring an adjudicator to adopt a price stipulated by the superintendent is inconsistent with the making of a determination required by the Act. Moreover, the provisions of the Act prevail over those in the contract.”
[2.4 Com-BIFSOPA CDI Normal Body Text] However, an adjudicator may take into account the certificate of the superintendent (or other certifying person) as part of the body of evidence in making their ultimate assessment of value: SSC Plenty Road Pty Ltd v Construction Engineering (Aust) Pty Ltd [2015] VSC 631, [118] (Vickery J).
[2.3 Com-BIFSOPA Heading 4] ‘Calculated under the contract’
[2.4 Com-BIFSOPA CDI Normal Body Text] ‘Calculated under the contract’ was considered in Transgrid v Siemens & Anor [2004] NSWSC 87, where Master Macready said that:
[2.5 Com-BIFSOPA Normal Body Quote] The word ‘calculate’, in its definition in the Shorter Oxford Dictionary and the Macquarie Dictionary, has primary meanings which comprehend both the making of a mathematical calculation and the making of an estimate. Thus on the plain meaning of the word ‘calculated’ in s 9 (a) the word includes the making of an estimate which under the contract is to be done by the superintendent. The words “in accordance with the terms of the contract” should also be noted. Those words are used elsewhere in the Act in particular in s 8(2)(a) in respect of the concept of reference date. The actual words used there are “determined by or in accordance with the terms of the contract”. This dual expression indicates that a date determined “by” the contract is one fixed self-referentially by its terms in contrast to a date determined “in accordance with” the contract which can be ascertained by some external mechanism performed pursuant to the contract. This wider meaning of the latter expression supports the view that a calculation “in accordance with” the terms of the contract for the purposes of s 9(a) is not merely a reference as was held by McDougall J in Abacus [at 38] to “the contractual mechanism for determination of that which is to be calculated”.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, this view was doubted on appeal to the New South Wales Court of Appeal in Transgrid v Siemens Ltd & Anor [2004] NSWCA 395, where Hodgson JA (Mason P and Giles JA) said:
[2.5 Com-BIFSOPA Normal Body Quote] Accordingly, it is not necessary to decide whether, on the true construction of s.9(a) and the contract, the amount “calculated in accordance with the terms of the contract” is the amount certified (cl.42.2 of the contract) or the value of the work less deductions (cl.42.3 of the contract). However I would express the view that the latter follows from what I think is a preferable interpretation of s.9(a) and the contract, consistent with the use of the word “calculation” and consistent with the provisions against contracting out (s.34); that is, on this matter, I prefer the view of McDougall J in Abacus v. Davenport [2003] NSWSC 1027 to that tentatively expressed by the Master in the present case.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Clyde Gergemann v Varley Power [2011] NSWSC 1039.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Where the amount claimed under the contract is by reference to a proportion of the work. CKP Constructions Pty Ltd v Evolution Piling Pty Ltd & Ors (Unreported, Supreme Court of Queensland, Dalton J, 24 July 2015). Under a subcontract for the building of foundation works, a claimant (a subcontractor) was entitled to progress payments for the proportion of the subcontract sum which represented the part of the works satisfactorily completed at the time of the claim. A claimant made a payment claim for both construction work (claims CW1 to CW6) and for variation claims. Claims CW2, 3 and 6 were for materials and preparation work. Referring to Patrick Stevedores Operations (No 2) Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2014] NSWSC 1413, [35], Dalton J held that these claims were not ‘for’ construction work (at 4). Claims CW1, 4 and 5 were held by Dalton J not to represent the proportion of the work actually completed. As such, Dalton J held that ‘the entirety of the claims CW1-6 were claims for money on some basis other than that allowed by the contract’ (at 4).
[2.6 Com-BIFSOPA bullet] Where the amount payable is ‘as certified by the Superintendent’. SSC Plenty Road Pty Ltd v Construction Engineering (Aust) Pty Ltd [2015] VSC 631. The construction contract in question provided that the respondent was to pay the claimant, in respect of variations, ‘the amount ascertained by the superintendent and certified in a relevant progress certificate’ using the order of precedence provided under the contract. The contract also provided that the amount payable for progress claims was ‘[a]s certified by the Superintendent’. Concluding that the VIC Act does not merely allow an adjudicator to adopt the superintendent’s certificate, Vickery J held that the construction contract made no express provision for the valuation of progress claims. 
[2.6 Com-BIFSOPA bullet] Not properly stating the basis on which amounts payable were decided. SSC Plenty Road Pty Ltd v Construction Engineering (Aust) Pty Ltd [2015] VSC 631. In an adjudication under the VIC Act, either the claimant or the respondent had relied on rates under the contract schedule of rates in valuing 19 disputed variation claims. The adjudicator did not state whether he adopted the schedule of rates in assessing these 19 disputed variation claims. Vickery J found that the adjudicator did not properly state ‘the basis on which the amounts determined to be payable had been decided’ (at [151]). Vickery J held that the adjudicator fell into jurisdictional error in respect of these 19 disputed variation claims.
[2.2 Com-BIFSOPA Heading 3] 71.2    Section 71(b)
[2.4 Com-BIFSOPA CDI Normal Body Text] The Act modifies the wording from ‘calculated under the contract’ to ‘in accordance with the contract.’ Parliament identified in the explanatory notes that section 71 of the Act “replicates section 13 of the repealed BCIPA.” Judicial interpretations of the previous provision will thus remain relevant. 
[2.3 Com-BIFSOPA Heading 4] ‘Does not provide for the matter’
[2.4 Com-BIFSOPA CDI Normal Body Text] Under the equivalent wording under the NSW Act, it has been held for the word ‘matter’ in this context that ‘it is clear that the trigger for the application of subsection (b) is the fact that the “contract makes no express provision with respect to the matter”. The relevant matter is “the amount of the progress payment”’: Transgrid v Siemens & Anor [2004] NSWSC 87, [28].
[2.4 Com-BIFSOPA CDI Normal Body Text] In the context of the NSW Act, in Quasar Constructions NSW Pty Ltd v Demtech Pty Ltd [2004] NSWSC 116 Barrett J said (at [20], [21]):
[2.5 Com-BIFSOPA Normal Body Quote] …If the contract makes no express provision with respect to “the matter” (being, clearly enough, the matter of “the amount of a progress payment to which a person is entitled in respect of a construction contract”), the amount to which the s 8 entitlement extends is to be calculated under s 9(b). If, on the other hand, there is such an express provision in the contract, the amount to which the s 8 entitlement relates is “the amount calculated in accordance with the terms of the contract” as specified in s 9(a). Again, application of the s 4 definition of ‘progress payment’ would mean that s 9(a) applied only if the contract made express provision for the calculation of the amount to which an entitlement under s 8 of the Building and Construction Industry Security of Payment Act related and would not apply where the contract merely created a progress payment regime in the way commonly encountered in the construction industry.
[2.5 Com-BIFSOPA Normal Body Quote] ...Having regard to the objects of the Act stated in s 3, it must be accepted that there was an intention that the statutory mechanisms should underwrite the contractual system of progress payments where the parties had adopted one. Such parties were not meant to be denied their agreed system and forced on to the timing and quantification in ss 8(2)(b) and 9(b) just because the express terms of their contract did not adopt the statutory definition of “progress payment”. It must follow that ss 8 and 9 (and, as to actual calculation, s 10) are to be approached on the footing that where there is an express contractual regime for the claiming and making of progress payments in the ordinary sense, divorced from the defined meaning arising from s 4, the matters of timing and quantification with which ss 8 and 9 are concerned are to be determined in accordance with the express contractual provisions on the subject. This is the approach that commended itself to McDougall J in Musico v Davenport (above). It involves the conclusion that, to the extent that the expression “progress payment” plays a part in deciding whether ss 8(2)(a) and 9(a) apply, the context is, as contemplated by s 6 of the Interpretation Act, one indicating a meaning other than that derived from the s 4 definition.
[2.3 Com-BIFSOPA Heading 4] Set off
[2.4 Com-BIFSOPA CDI Normal Body Text] In J Hutchinson Pty Ltd v Glavcom Pty Ltd [2016] NSWSC 126, Ball J considered an argument made by a subcontractor that an adjudicator was not entitled to deduct liquidated damages claimed by the head contractor when determining the amount of a progress payment pursuant to s 9(b) of the NSW Act (equivalent to section 71(b) of the Act). Whilst his Honour did not reach a conclusion on this point, his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] Section 9(b) says nothing about a setoff. It says that the amount of a progress payment is to be determined by the value of the relevant work. It is difficult to see how a right of setoff can be implied in the face of the clear language of the section. There is nothing inherently uncommercial or absurd in a provision which does not permit Hutchinson to setoff amounts due to it against amounts due to Glavcom for the work that it has done. If the parties had wanted to allow for a right of setoff, they were free to do so by specifically providing in the contract for the calculation of a progress payment which included a right of setoff. The parties did not do that in this case.

```

### `annotated/section_073.txt`
```
# Annotated BIF Act source — Section 73
# Chapter: CHAPTER 3 – Progress payments
# Section title: Due date for payment
# DOCX paragraphs: 1958-1994

[2 Com-BIFSOPA Heading 1] SECTION 73 – Due date for payment
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 73    Due date for payment
[1.3 BIFSOPA level 1 (CDI)] A progress payment under a construction contract becomes payable—
[1.4 BIFSOPA level 2 (CDI)] if the contract provides for the matter—on the day on which the payment becomes payable under the contract; or
[1.1 BIFSOPA Body Text] Notes— 
[Style1] A ‘pay when paid’ provision in a construction contract has no effect, see section 74.
[Style1] A provision in a construction management trade contract or subcontract providing for payment of a progress payment later than 25 business days is void, see the Queensland Building and Construction Commission Act 1991, section 67U. 
[Style1] A provision in a commercial building contract providing for payment of a progress payment later than 15 business days is void, see the Queensland Building and Construction Commission Act 1991, section 67W.
[1.4 BIFSOPA level 2 (CDI)] if the contract does not provide for the matter—on the day that is 10 business days after the day a payment claim for the progress payment is made under part 3.
[1.3 BIFSOPA level 1 (CDI)] Interest for a construction contract is payable on the unpaid amount of a progress payment that has become payable at the greater of the following rates— 
[1.4 BIFSOPA level 2 (CDI)] the rate stated in the contract;
[1.4 BIFSOPA level 2 (CDI)] the rate prescribed under the Civil Proceedings Act 2011, section 59(3) for a money order debt.
[1.3 BIFSOPA level 1 (CDI)] However, for a construction contract to which the Queensland Building and Construction Commission Act 1991, section 67P applies because it is a building contract, interest is payable at the penalty rate under that section.
[1.3 BIFSOPA level 1 (CDI)] Each of the following construction contracts are taken to be a contract to which subsection (1)(b) applies—
[1.4 BIFSOPA level 2 (CDI)] a construction contract that includes a ‘pay when paid’ provision; 
[1.4 BIFSOPA level 2 (CDI)] a construction management trade contract or subcontract mentioned in the Queensland Building and Construction Commission Act 1991, section 67U;
[1.4 BIFSOPA level 2 (CDI)] a commercial building contract mentioned in the Queensland Building and Construction Commission Act 1991, section 67W. 
[1.3 BIFSOPA level 1 (CDI)] In this section—
[1.1 BIFSOPA Body Text] ‘pay when paid’ provision, of a construction contract, see section 74.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 73.1    Changes introduced to section 73
[2.4 Com-BIFSOPA CDI Normal Body Text] The introduction of section 73 has restructured section 15 of BCIPA which previously stipulated the due date for the payment of a progress payment.
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 73(1)(a) mirrors section 15(1)(a) and specifies that a progress payment will become payable on the day in which the payment becomes payable under the contract. However:
[2.6 Com-BIFSOPA bullet] Note 1 voids a ‘pay when paid’ provision as defined in section 74. 
[2.6 Com-BIFSOPA bullet] Note 2 makes void a provision which provides for payment later than 25 business days under a construction management trade contract or subcontract (consistent with section 67U of the Queensland Building and Construction Commission Act 1991)
[2.6 Com-BIFSOPA bullet] Note 3 makes void a provision which provides for payment later than 15 business days for a commercial building contract under 67W of the Queensland Building and Construction Commission Act 1991.
[2.4 Com-BIFSOPA CDI Normal Body Text] If the contract does not provide for the due date for payment, section 73(1)(b) operates to deem the due date for payment to be 10 business days after a payment claim for a progress payment is made. Where a provision listed under Notes 1-3 is rendered void, Section 73(4) deems the due date consistent with section 73(1)(b).
[2.4 Com-BIFSOPA CDI Normal Body Text] The following commentary remains applicable.
[2.2 Com-BIFSOPA Heading 3] 73.2    Due date for payment
[2.4 Com-BIFSOPA CDI Normal Body Text] See Isis Projects Pty Ltd v Clarence Street Ltd [2004] NSWSC 222, where Einstein J held that a claimant must prove that a respondent’s liability has accrued in terms of a ‘valid payment claim’ and prove what is the due date for the progress payment (at [33]). His Honour rejected a construction of the due date as the ‘claimed’ due date.
[2.4 Com-BIFSOPA CDI Normal Body Text] The appeal from this decision was dismissed: See Clarence Street Pty Ltd v Isis Projects Pty Ltd [2005] NSWCA 391.
[2.2 Com-BIFSOPA Heading 3] 73.3    Interest payable
[2.4 Com-BIFSOPA CDI Normal Body Text] In McCarthy v State of Queensland [2013] QCA 313, Muir JA said of BCIPA that:
[2.5 Com-BIFSOPA Normal Body Quote] Sections 17 to 20 inclusive do not contemplate the recovery of interest on the amount claimed in a payment claim, unless that interest is part of the progress payment in respect of which a payment claim is made.
[QBS Body Text] In Duff Kennedy Pty Ltd v Galileo Miranda Nominee Pty Ltd [2020] NSWCA 25, the New South Wales Court of Appeal affirmed that the definition of “scheduled amount” does not include interest payable under section 11 of the NSW Act (equivalent to section 73 of the Act) on the unpaid amount of a progress payment unless that amount is included in a “scheduled amount”. In that case, the respondent failed to pay a sum by the deadline, and interest accrued for which the appellant asserted became payable under section 11 of the NSW Act. The respondent purported to exercise its right to suspend works due to the default in payment, and after the works failed to commence, the respondent issued a show cause notice. In upholding the decision of Parker J at trial and dismissing the appeal, White and Brereton JJA and Barrett AJA held that the appellant was in breach by unjustifiably suspending the works since the interest did not fall within the definition of “scheduled amount”.

```
