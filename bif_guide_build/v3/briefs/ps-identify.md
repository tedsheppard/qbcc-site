# Element-page brief — ps-identify
**Title:** Identify the payment claim
**Breadcrumb:** Requirements of a payment schedule
**Anchor id:** `ps-identify`
**Output file:** `bif_guide_build/v3/pages/page_ps-identify.html`

## Statute scope note
Show only s 69(a) and chapeau.

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

### `statute/chapter_3/section_069.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 69 — Meaning of payment schedule
A payment schedule, responding to a payment claim, is a
written document that—
Page 99 Current as at 27 April 2025

(a) identifies the payment claim to which it responds; and
(b) states the amount of the payment, if any, that the
respondent proposes to make; and
(c) if the amount proposed to be paid is less than the
amount stated in the payment claim—states why the
amount proposed to be paid is less, including the
respondent’s reasons for withholding any payment; and
(d) includes the other information prescribed by regulation.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_069.txt`
```
# Annotated BIF Act source — Section 69
# Chapter: CHAPTER 3 – Progress payments
# Section title: Meaning of payment schedule
# DOCX paragraphs: 1566-1701

[2 Com-BIFSOPA Heading 1] SECTION 69 – Meaning of payment schedule 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 69    Meaning of payment schedule 
[1.1 BIFSOPA Body Text] A payment schedule, responding to a payment claim, is a written document that—
[1.4 BIFSOPA level 2 (CDI)] identifies the payment claim to which it responds; and
[1.4 BIFSOPA level 2 (CDI)] states the amount of the payment, if any, that the respondent proposes to make; and
[1.4 BIFSOPA level 2 (CDI)] if the amount proposed to be paid is less than the amount stated in the payment claim—states why the amount proposed to be paid is less, including the respondent’s reasons for withholding any payment; and
[1.4 BIFSOPA level 2 (CDI)] includes the other information prescribed by regulation.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 69.1    Introduction of section 69
[2.4 Com-BIFSOPA CDI Normal Body Text] Similarly to section 68, section 69 specifies the meaning of a payment schedule while the process for submitting a payment schedule in accordance with the Act now appears under section 76 of the Act. Previously, these requirements appeared under section 18 and section 18A BCIPA.
[2.2 Com-BIFSOPA Heading 3] 69.2    Form of a payment schedule
[2.4 Com-BIFSOPA CDI Normal Body Text] No particular form for a payment schedule was required under BCIPA. The relevant principles for determining whether a document is a payment schedule for the purposes of BCIPA were set out in Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA Pty Ltd) & Ors [2007] QSC 33 where Chesterman J said:
[2.5 Com-BIFSOPA Normal Body Quote] The Act emphasises speed and informality. Accordingly one should not approach the question whether a document satisfies the description of a payment schedule (or payment claim for that matter) from an unduly critical viewpoint. No particular form is required. One is concerned only with whether the content of the document in question satisfies the statutory description. To constitute a payment schedule the applicant’s email of 14 December had to:
[2.5 Com-BIFSOPA Normal Body Quote] (i)    identify the payment claim to which it related, and
[2.5 Com-BIFSOPA Normal Body Quote] (ii)    state any amount which the recipient of the payment claim proposed to make in response to it; and
[2.5 Com-BIFSOPA Normal Body Quote] (iii)    importantly, if that amount is less than the amount claimed the payment schedule it must state why it is less.
[2.5 Com-BIFSOPA Normal Body Quote] If these three criteria are satisfied the document will be a payment schedule. How they are expressed, with what formality or lack of it, and with what felicity or awkwardness, will not matter.
[2.4 Com-BIFSOPA CDI Normal Body Text] Importantly, this passage was approved by the Queensland Court of Appeal in Neumann Contractors Pty Ltd v Traspunt No 5 Pty Ltd [2010] QCA 119 at [24] per Muir JA (Holmes and Chesterman JJA agreeing).
[2.4 Com-BIFSOPA CDI Normal Body Text] The requirements of BCIPA are substantially the same and accordingly, to constitute a payment schedule, a document must:
[2.6 Com-BIFSOPA bullet] identifies the payment claim to which it responds; and
[2.6 Com-BIFSOPA bullet] states the amount of the payment, if any, that the respondent proposes to make; and
[2.6 Com-BIFSOPA bullet] if the amount proposed to be paid is less than the amount stated in the payment claim—states why the amount proposed to be paid is less, including the respondent’s reasons for withholding any payment; and
[2.6 Com-BIFSOPA bullet] includes the other information prescribed by regulation.
[2.4 Com-BIFSOPA CDI Normal Body Text] If these requirements are satisfied, the document will be a payment schedule, and it does not matter “[h]ow they are expressed, with what formality or lack of it, and with what felicity or awkwardness”. 
[2.4 Com-BIFSOPA CDI Normal Body Text] A document will be sufficient to be a payment schedule if it "achieves the basic objective of putting the claiming party on notice as to how much the party making payment intends to pay with reference to the payment claim and the reasons why that amount is lower than that claimed, if that is the case": Linke Developments Pty Ltd v 21st Century Developments Pty Ltd [2014] SASC 203 at [45] (Nicholson J).
[2.5 Com-BIFSOPA Normal Body Quote] To be a payment schedule, a document does not need to be described internally as a ‘payment schedule’: Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd & Ors [2013] QCA 386, [11] (Muir JA, Gotterson and Morrision JJA agreeing).
[2.4 Com-BIFSOPA CDI Normal Body Text] The requirements of form for a payment schedule are influenced by its role which (as per Chesterman J, above) is to define the issues in dispute between the parties. In Multiplex Constructions Pty Ltd v Luikens & Anor [2003] NSWSC 1140, Palmer J said that: 
[2.5 Com-BIFSOPA Normal Body Quote] A payment claim and a payment schedule are, in many cases, given and received by parties who are experienced in the building industry and are familiar with the particular building contract, the history of construction of the project and the broad issues which have produced the dispute as to the claimant’s payment claim. a payment claim and a payment schedule must be produced quickly; much that is contained therein in an abbreviated form which would be meaningless to the uninformed reader will be understood readily by the parties themselves. A payment claim and a payment schedule should not, therefore, be required to be as precise and as particularised as a pleading in the Supreme Court. Nevertheless, precision and particularity must be required to a degree reasonably sufficient to apprise the parties of the real issues in the dispute.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, a document, or the first document received after submission of a payment claim that meets the requirements of section 69 of the Act, is not necessarily a payment schedule: Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2013] QCA 386, [24] (Muir JA, Gotterson and Morrison JJA agreeing); McConnell Dowell Constructors (Aust) Pty Ltd v Heavy Plant Leasing Pty Ltd [2013] QSC 223, [48] (Boddice J).
[2.4 Com-BIFSOPA CDI Normal Body Text] It may be necessary to consider the surrounding circumstances in determining what documents, if any, constitute a payment schedule: see Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2013] QCA 386 at [23], where the Court of Appeal said:
[2.5 Com-BIFSOPA Normal Body Quote] If, as the parties accepted, only one payment schedule may be served in response to a payment claim, as a general proposition, whether a document is a payment schedule is to be determined by reference to the contents of the document and, in some circumstances, any accompanying or supplementary document. The fact that, as is the case here, a later document is described as a payment schedule and the earlier one is not would not normally bear on whether the earlier document is a payment schedule. The above proposition is qualified because it may be possible that the later document has some evidentiary value. Take, for example, circumstances in which there has been a course of dealing between the parties in which documents such as payment claims and payment schedules have been clearly identified in a particular way and the document alleged by the recipient to be a payment schedule lacks such identification. It may well be that the later document provided and described as a payment schedule would constitute some evidence of adherence to and continuance of the previous course of conduct.
[2.4 Com-BIFSOPA CDI Normal Body Text] A document meeting the requirements of section 69 of the Act is not necessarily the payment schedule for the purposes of the Act: see Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2013] QCA 386, where the Court of Appeal, following the discussion in the above extract in relation to section 18 BCIPA, said:
[2.5 Com-BIFSOPA Normal Body Quote] It follows from the foregoing discussion that I do not accept as correct the unqualified proposition that a document which meets the requirements of s 18 of the Act is necessarily a payment schedule. Section 18 does not so provide. It merely stipulates the matters which must be incorporated in a payment schedule…A document such as a payment schedule does not necessarily take its character only from its own form and content.
[2.4 Com-BIFSOPA CDI Normal Body Text] The trial judge in McConnell Dowell Constructors (Aust) Pty Ltd v Heavy Plant Leasing Pty Ltd [2013] QSC 223 took a similar view, considering that Minimax:
[2.5 Com-BIFSOPA Normal Body Quote] is not authority for a proposition that the first correspondence sent by a party following receipt of a payment claim is its payment schedule. Whether that response is the payment schedule will depend upon all of the circumstances of the case. Minimax is also not authority for the proposition that a document which does not purport to be a payment schedule must be considered to be a payment schedule if it was delivered within the statutory timeframe.
[2.4 Com-BIFSOPA CDI Normal Body Text] That a document may not necessarily be a payment schedule may also depend on whether the language of the payment schedule is ‘provisional in any way’ or suggests that ‘a further and more detailed response to the payment claim would be forthcoming’: see Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2013] QCA 386, [37].
[2.4 Com-BIFSOPA CDI Normal Body Text] In Style Timber Floor Pty Ltd v Krivosudsky [2019] NSWCA 171, the New South Wales Court of Appeal held that while a payment schedule under section 14 of the NSW Act (equivalent of section 69 of the BIF Act) does not require the degree of formality required in other legal contexts, use of the NSW Act “is not a licence for informality or an excuse for vague, generalised objections to payment.” In its payment schedule, Style Timber asserted that it had ‘many emails, photos, videos, back charges from builders and other trades, complains (sic) from my clients” that were relied upon to justify non-payment. The Court found that “documents to be so incorporated would need to be identified with sufficient particularity so that the recipient of the schedule knew what was being incorporated.”
[2.6 Com-BIFSOPA bullet] Examples
[2.6 Com-BIFSOPA bullet] A ‘relatively informal’ email. Gisley Investments Pty Ltd v Williams & Anor [2010] QSC 178. An email considered to be ‘relatively informal’ which was sent in response to a payment claim received that day, was held to be a payment schedule under BCIPA. This email referred to ‘paperwork, stating I owe you this outrageous amount of money’, stated that ‘whilst the job continues to remain unfinished, final payment is not yet owed’, and provided other reasons for non-payment. The Court held that this email was sufficient to be a payment schedule.
[2.6 Com-BIFSOPA bullet] Denham Constructions Pty Ltd v Islamic Republic of Pakistan (No 2) [2016] ACTSC 215. An email was held not to be a payment schedule for the purposes of ss 16(2) and (3) of the ACT Act as Mossop AsJ held that the email did not identify the payment claim to which it related. The email was merely directed at a claim for a variation which formed one of many components of the payment claim served by the plaintiff. Mossop AsJ held that despite this connection, the email was:
[2.5 Com-BIFSOPA Normal Body Quote] …  not sufficient to make it clear that it is a response to the payment claim as opposed to being a communication in relation to another document generated for the purposes of the underlying contract which happened to relate to one component of the payment claim. 
[2.6 Com-BIFSOPA bullet] A letter not referring to the payment claim or stating to be a ‘payment schedule’. Springs Golf Club Pty Ltd v Profile Golf Club Pty Ltd [2006] NSWSC 344. The letter purporting to be a payment schedule provided a summary of an earlier meeting and confirmed the amount that the claimant was offered at the end of the meeting. The letter was not described as being a ‘payment schedule’. The Court held the letter to be a payment schedule, being in response to the payment claim and stating the amount that the respondent was, and continues to be, prepared to pay.
[2.6 Com-BIFSOPA bullet] Linke Developments Pty Ltd v 21st Century Developments Pty Ltd [2014] SASC 203. The document in question in Linke was a letter which referred to the payment claim, enclosed a cheque for an amount less than the payment claim, and provided reasons for the lower amount. The Court held this sufficient to constitute a payment schedule.
[2.6 Com-BIFSOPA bullet] T & T Building Pty Ltd v GMW Group Pty Ltd & Ors [2010] QSC 211. Judgment was sought in respect of five payment claims. In relation to one of them, claim 22, there was a response by email which read: ‘We do not agree with the above claim as the variations schedule is incorrect’. The respondent paid approximately half of claim 22. The Court held that the part payment made it clear that ‘there was not an objection to the entire amount’ and that the respondent had not satisfied the requirement to state why the scheduled amount is less.
[2.6 Com-BIFSOPA bullet] A later document served in time. Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2013] QCA 386. Within time, the respondent served two documents on the claimant, the 6 March 2013 document, and the 8 March 2013 document. The adjudicator concluded that the 6 March 2013 document was ‘the payment schedule’. In the circumstances, the Court of Appeal held the 8 March 2013 document to be the payment schedule for the purposes of the Act. The Court of Appeal held that it is not necessarily the case that the first document which satisfies the requirements of section 18 of BCIPA is ‘the’ payment schedule.
[2.2 Com-BIFSOPA Heading 3] 69.3    Payment certificates
[2.4 Com-BIFSOPA CDI Normal Body Text] That a payment certificate is issued under a contract does not necessarily mean that it will be the payment schedule for the purposes of the Act: see McConnell Dowell Constructors (Aust) Pty Ltd v Heavy Plant Leasing Pty Ltd [2013] QSC 223 where Boddice J said having regard to BCIPA (at [49]):
[2.5 Com-BIFSOPA Normal Body Quote] The first respondent contended significance was to be attached to the reference in the 6 March document to the “payment certificate”, as Part B of the schedule to the subcontract provides that a payment schedule for the purposes of the Payments Act is a payment certificate. However, that provision allows a party to contend that the payment schedule delivered by it is a payment certificate under the contract. The provision does not mean every payment certificate must be the party’s payment schedule in respect of a payment claim.
[2.2 Com-BIFSOPA Heading 3] 69.4    Section 69(a)
[2.3 Com-BIFSOPA Heading 4] Identify the payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] In Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA) Pty Ltd [2007] QSC 333, Chesterman J of the Queensland Supreme Court, in considering the requirement contained in s 18(3) of BCIPA held that a payment schedule was invalid where the payment claim contained three items and the payment schedule only provided reasons for non-payment in relation to two of those items. Relevantly, his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] The machinery for prompt payment and enforcement of payment would break down if a document, said to be a payment schedule, took issue with part only of a claim but was silent as to what it proposed to pay in respect of the balance.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Gisley Investments Pty Ltd v Williams & Anor [2010] QSC 178, an email sent by the respondent to a payment claim, on the same date the payment claim was received, which referred to “paperwork, stating I owe you this outrageous amount of money” was held to be a sufficient reference to the payment claim served that day for the purposes of BCIPA. The Court noted that there ‘was no other document identified with which it could have been confused’.
[2.4 Com-BIFSOPA CDI Normal Body Text] In BCS Infrastructure Support Pty Ltd v Jones Lang Lasalle (NSW) Pty Ltd [2020] VSC 739, Stynes J found that the payment schedule in question satisfied the requirements under section 15(2)(a) of the NSW SOP Act (the equivalent of section 69(a) of the BIF Act) on the basis that it made reference to a letter of demand, which contained the payment claim within it, as well as a figure that clearly related to the disputed amount.
[2.4 Com-BIFSOPA CDI Normal Body Text] Example
[2.6 Com-BIFSOPA bullet] A letter not referring to the payment claim. Springs Gold Club Pty Ltd v Profile Golf Club Pty Ltd [2006] NSWSC 344. A payment claim was submitted on 29 July 2005, seeking payment for outstanding invoices. A letter referred to ‘your letter dated Friday 29 July 2005’. The letter was held to be a valid payment schedule.
[2.2 Com-BIFSOPA Heading 3] 69.5    Section 69(b)
[2.3 Com-BIFSOPA Heading 4] ‘Scheduled amount’
[2.4 Com-BIFSOPA CDI Normal Body Text] Parliament has omitted defining the “the amount of the payment, if any, that the respondent proposes to make” as the ‘scheduled amount’. However, there is no change in meaning and ‘scheduled amount’ is likely to continue to have application from earlier judicial interpretation.
[2.4 Com-BIFSOPA CDI Normal Body Text] ‘Scheduled amount’ is the overall amount that the respondent proposes to pay (if any): Cornerstone Danks Street v Parkview Constructions [2014] NSWSC 866, [25] (McDougall J).
[2.3 Com-BIFSOPA Heading 4] States the amount
[2.4 Com-BIFSOPA CDI Normal Body Text] On the requirement that a payment schedule ‘states the amount of the payment, it is likely sufficient that an amount of "nothing", "nil" or "zero" is sufficient for the purposes of section 18(2)(b) BCIPA: Barclay Mowlem Construction Ltd v Tesrol Walsh Bay Pty Ltd [2004] NSWSC 1232 at [15] (McDougall J); Façade Treatment Engineering Ltd v Brookfield Multiplex Constructions Pty Ltd [2015] VSC 41 at [37] (Vickery J). In Façade Treatment Engineering, the document in question was an email which referred to the payment claim and alleged that the payment claim was not submitted as valid. Vickery J held that this email was clear in stating the amount the respondent proposed to pay, being nil, on the basis of the alleged invalidity of the payment claim.
[2.4 Com-BIFSOPA CDI Normal Body Text] An amount may also be sufficiently stated if it can be clearly inferred from the document (or documents). In Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA Pty Ltd) [2007] QSC 333, the Qld Supreme Court held that the requirement to state the amount of the payment is satisfied if, although no actual amount is stated, it can be clearly inferred from the document as a whole that no amount is to be paid to the Claimant.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, for such an inference to be made, it appears to be the case that the inferred amount of nil be able to be interpreted as addressing the whole of the payment claim, and not merely part of it. A document addressing part of a payment claim was the case in Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA Pty Ltd) [2007] QSC 333 where the payment claim was an invoice for three items, the document in question proposed to pay a nil amount in respect of the payment claim, and the document in question provided reasons for why a nil amount was payable for the first item of the payment claim. The document in question did not provide reasons for items 2 or 3 of the payment claim. On this issue Chesterman J concluded:
[2.5 Com-BIFSOPA Normal Body Quote] There are two possible constructions of the 14 December email. The first is that the applicant did, in fact, object to paying anything for items 2 and 3, and proposed to pay nothing in respect of them. In that case for the document to be a payment schedule the applicant had to give reasons for its objection. The email is silent on this point. The second, more likely, construction is that the email does not address that part of the payment claim at all. It did not state the amount of the payment which the applicant proposed to make. If this be the true construction the email did not satisfy the second criterion in s 18 of the Act. On either view the 14 December email did not comply with the definition. The email is incomplete if it is intended to be a payment schedule. It had to address the claim made and not only a part of it. I think this is clear.
[2.5 Com-BIFSOPA Normal Body Quote] A payment schedule which complies with the Act will set out the amount it proposes to pay in response to a claim. By s 20 a respondent who does not pay the amount its payment schedule proposes to pay can suffer summary judgment in a court of competent jurisdiction and enforce the amount as a judgment debt. The machinery for prompt payment and enforcement of payment would break down if a document, said to be a payment schedule, took issue with part only of a claim but was silent as to what it proposed to pay in respect of the balance. The contractor could not enter judgment. The respondent’s reticence could frustrate the operation of the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Brimar Electrical Services Pty Ltd v Zi-Argus Australia Pty Ltd [2014] QDC 78, the payment claim was comprised of five invoices, each of which was a payment claim endorsed under the Act. The Court held that although the respondent was required to state why the overall scheduled amount was less than the claimed amount, there was no need for the Respondent to specifically address each invoice that comprised the claim.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Cornerstone Danks Street v Parkview Constructions [2014] NSWSC 866, McDougall J held at [25] that the term "scheduled amount" is not intended to be used in respect of each discrete item set out in a payment schedule. The scheduled amount is the overall amount that the respondent proposes to pay.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Melaleuca View Pty Ltd v Sutton Constructions Pty Ltd & Ors [2019] QSC 226, Brown J found that section 69 of the Qld Act required that a payment schedule must “state” [the amount of the payment] rather than “indicate,” suggesting that a greater level of detail in the reasons was required than under the NSW Act. The court held that Melaleuca had failed to respond with a payment schedule since it did not identify the initial payment claim. Brown J reasoned that while such a conclusion may be seen to be harsh, the onus was on a respondent to ensure it had complied with the requirements of the Qld Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Inferring a nil amount. In Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA Pty Ltd) [2007] QSC 333 the relevant passage of the email in question was as follows: ‘For this reasons [sic] we accept not your invoice but we suggest to have a meeting on site next year to clarify the situation and to find a solution for both sides.’ This was sufficient to infer a scheduled amount of nil. However, because the email in question did not address all items of the payment claim, it was not a valid payment schedule under the Act.
[2.6 Com-BIFSOPA bullet] ‘final payment is not yet owed’. Gisley Investments Pty Ltd v Williams & Anor [2010] QSC 178. An email in response to the payment claim served that day provided reasons for non-payment and stated that ‘whilst the job continues to remain unfinished, final payment is not yet owed’. The Court held that the email constituted a payment schedule within the meaning of the Act.
[2.6 Com-BIFSOPA bullet] Without prejudice letters. National Vegetation Management Solutions Pty Ltd v Shekar Plant Hire Pty Ltd [2010] QSC 3. It has been held that a “without prejudice” letter does not constitute a payment schedule under the Act. This is because the letter did not state a “scheduled amount” for payment, but rather was a proposal for an offer to settle.
[2.6 Com-BIFSOPA bullet] Conditional payment. Tenix Alliance Pty Ltd v Magaldi Power Pty Ltd [2010] QSC 7. A document purporting to be a payment schedule provided that two conditions were to be satisfied before an amount was to be paid. The Court held that the document did comply with section 18(2)(b) of the Act. The Court held that it could not be said that there was a statement that nothing would be paid, or that the amount stated on the document would be paid.
[2.4 Com-BIFSOPA CDI Normal Body Text] In RHG Construction Fitout and Maintenance Pty Ltd v Kangaroo Point Developments MP Property Pty Ltd & Ors [2021] QCA 117, the Queensland Court of Appeal overturned the Supreme Court’s decision and held a “deeming clause” (reflecting the common deeming provision whereby an assessment of a payment claim by the superintendent is deemed a payment schedule for the purpose of the BIF Act) to be valid. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In the Supreme Court decision, Justice Dalton found that the superintendent’s assessment did not constitute a payment schedule because it did not state the amount that the respondent proposed to make, but rather was the superintendent’s opinion as to the amount payable. Dalton J determined the deeming clause void because the provision could not deem a document, which did not comply with section 69 of the BIF Act, as a payment schedule.
[2.4 Com-BIFSOPA CDI Normal Body Text] In overturning the decision, and allowing the appeal, the Court of Appeal held:
[2.4 Com-BIFSOPA CDI Normal Body Text] interpreting the provision in such a way would constitute a failure to give the contract a business-like interpretation and would be ignoring the parties’ use of long-standing statutory terms; and 
[2.4 Com-BIFSOPA CDI Normal Body Text] the superintendent’s schedule, when read with the deeming provision, meant that the amount stated in the schedule was the amount the respondent proposed to pay and therefore section 69(b) of the Act was satisfied.
[2.4 Com-BIFSOPA CDI Normal Body Text] In ACP Properties (Townsville) Pty Ltd v Rodrigues Construction Group Pty Ltd & Anor [2021] QSC 45, Bradley J determined that two emails sent from the respondent stating that it considered no amount was payable in respect of the claimant's invoices, and that it proposed to pay an amount of $nil as a result of their contractual arrangement capping the amount payable, was sufficient to constitute a payment schedule for the purposes of section 69(b) and (c) of the BIF Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] In BCS Infrastructure Support Pty Ltd v Jones Lang Lasalle (NSW) Pty Ltd [2020] VSC 739, Stynes J held that a ‘nil’ amount can be inferred for the purpose of section 15(2)(b) of the SOP Act (equivalent to section 69(b) of the BIF Act), where the respondent does not dispute the validity of a payment claim and expresses an intention not to pay the claimant any amount
[2.2 Com-BIFSOPA Heading 3] 69.6    Section 69(c) – reasons for withholding payment
[2.4 Com-BIFSOPA CDI Normal Body Text] As the purpose of a payment schedule is to clearly define the issues in dispute, some degree of “[p]recision and particularly is required in the payment schedule to a degree reasonably sufficient to apprise the parties of the real issues in the dispute”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Multiplex Constructions Pty Ltd v Luikens & Anor [2003] NSWSC 1140, Palmer J said:
[2.5 Com-BIFSOPA Normal Body Quote] The evident purpose of s.13(1) and (2), s.14(1), (2) and (3), and s.20(2B) is to require the parties to define clearly, expressly and as early as possible what are the issues in dispute between them; the issues so defined are the only issues which the parties are entitled to agitate in their dispute and they are the only issues which the adjudicator is entitled to determine under s.22. It would be entirely inimical to the quick and efficient adjudication of disputes which the scheme of the Act envisages if a respondent were able to reject a payment claim, serve a payment schedule which said nothing except that the claim was rejected, and then “ambush” the claimant by disclosing for the first time in its adjudication response that the reasons for the rejection were founded upon a certain construction of the contractual terms or upon a variety of calculations, valuations and assessments said to be made in accordance with the contractual terms but which the claimant has had no prior opportunity of checking or disputing. In my opinion, the express words of s.14(3) and s.20(2B) are designed to prevent this from happening.
[2.5 Com-BIFSOPA Normal Body Quote] Section 14(3) requires that if the respondent to a payment claim has “any reason” for “withholding payment”, it must indicate that reason in the payment schedule. To construe the phrase “withholding payment” as meaning “withholding payment only by reason of a set-off or cross claim” is to put a gloss on the words which their plain meaning cannot justify. The phrase, in the context of the subsection as a whole, simply means “withholding payment of all or any part of the claimed amount in the payment claim”. If the respondent has any reason whatsoever for withholding payment of all or any part of the payment claim, s.14(3) requires that that reason be indicated in the payment schedule and s.20(2B) prevents the respondent from relying in its adjudication response upon any reason not indicated in the payment schedule. Correspondingly,s.22(d) requires the adjudicator to have regard only to those submissions which have been “duly made” by the respondent in support of the payment schedule, that is, made in support of a reason for withholding payment which has been indicated in the payment schedule in accordance with s.14(3).
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] …For a respondent merely to state in its payment schedule that a claim is rejected is no more informative than to say merely that payment of the claim is “withheld”: the result is stated but not the reason for arriving at the result. Section 14(3) requires that reasons for withholding payment of a claim be indicated in the payment schedule with sufficient particularity to enable the claimant to understand, at least in broad outline, what is the issue between it and the respondent. This understanding is necessary so that the claimant may decide whether to pursue the claim and may know what is the nature of the respondent’s case which it will have to meet if it decides to pursue the claim by referring it to adjudication.
[2.5 Com-BIFSOPA Normal Body Quote] (emphasis original)
[2.4 Com-BIFSOPA CDI Normal Body Text] However, provided that the reasons are part of the payment schedule, section 18(3) of BCIPA was held to have no bearing on the adequacy or sufficiency of the reasons: Barclay Mowlem v Tesrol Walsh Bay [2004] NSWSC 1232, [26] (McDougall J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In the context of the above comments, an important difference between the NSW Act and the Act, is the procedure for both ‘new reasons’ and a reply to such under complex payment claims.
[2.4 Com-BIFSOPA CDI Normal Body Text] Failure to give a payment schedule can be fatal to disputing a claim. In TFM Epping Land Pty Ltd v Decon Australia Pty Ltd [2020] NSWCA 93, the Court of Appeal considered three ‘triable issues’ arising from a summary judgement made in favour of a claimant. In granting conditional leave to appeal the summary judgement, the three issues were whether: (1) variation claims arose under the construction contract or were quantum meruit claims; (2) there was an available reference date for the claim; and (3) the claim was validly served in absence of a supporting statement. Basten JA, with whom Meagher JA and Emmett AJA agreed, held, in dismissing each of the grounds of the appeal in turn: 
[2.4 Com-BIFSOPA CDI Normal Body Text] (1) the claims were for work done under the construction contract and the respondent should have disputed the amounts in the payment schedule pursuant to section 14 of the NSW Act (equivalent to section 69 BIFSOPA); 
[2.4 Com-BIFSOPA CDI Normal Body Text] (2) upon a proper construction of the construction contract, there was an available reference date for the payment claim as required by section 13(5) of the NSW Act (equivalent to section 75(4) BIFSOPA), and the inclusion of interest which accrued after the reference date was permissible; and 
[2.4 Com-BIFSOPA CDI Normal Body Text] (3) the failure to provide a supporting statement under section 13(7) of the NSW Act (a requirement not present in BIFSOPA) did not result in the invalidity of the payment claim because there was no basis to imply a legislative intention otherwise.
[2.4 Com-BIFSOPA CDI Normal Body Text] In BCS Infrastructure Support Pty Ltd v Jones Lang Lasalle (NSW) Pty Ltd [2020] VSC 739, Stynes J determined that “the substantive dispute between parties with regards to pricing ” was a valid reason for withholding payment under section 15(3) of the SOP Act (equivalent to section 69(c) of the BIF Act) on the basis that: 
[2.4 Com-BIFSOPA CDI Normal Body Text] it was clear that the payment schedule in issue was in response to the payment claims;
[2.4 Com-BIFSOPA CDI Normal Body Text] the scope of the dispute was described with some particularity; and 
[2.4 Com-BIFSOPA CDI Normal Body Text] it is irrelevant whether the claimant believes that there was a dispute between the parties.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Joye Group Pty Ltd v Cemco Projects Pty Ltd [2021] NSWCA 211, the NSW Court of Appeal overturned the District Court’s  findings that a refusal to pay two payment claims until work was complete was sufficient reasoning for withholding payment under the equivalent of section 69(c) of the BIF Act. In the District Court decision, Strathdee DCJ found that a valid payment schedule was given by the respondent, as the background communications between the parties clarified the works that were incomplete and that thus, the respondent provided reasons for withholding payment. In overturning the District Court’s finding and determining that the payment schedule was invalid, the Court of Appeal held:
[2.4 Com-BIFSOPA CDI Normal Body Text] the statement that “we will not pay your claim until …” does not indicate that part or all of the claim will not be paid, but merely a statement as to when some or all may be paid;
[2.4 Com-BIFSOPA CDI Normal Body Text] even if there was only one item that was the subject of the payment claim, a refusal to make payment until work had been completed does not demonstrate which of the items identified in the payment claim had been carried out before the reference date for which the Claimant would be entitled to payment; and
[2.4 Com-BIFSOPA CDI Normal Body Text] a payment schedule is not to be reconstructed by reference to external materials, where there is no ambiguity, to give it a degree of particularity which it simply did not enjoy.
[2.4 Com-BIFSOPA CDI Normal Body Text] There are two further issues relating to reasons in a payment schedule, namely:
[2.6 Com-BIFSOPA bullet] the requirement under section 69(c) to ‘state’ the reasons for withholding payment; and
[2.6 Com-BIFSOPA bullet] the incorporation of documents in a payment schedule.
[2.4 Com-BIFSOPA CDI Normal Body Text] These two issues are considered in the following two sections.
[2.5 Com-BIFSOPA Normal Body Quote] In Turnkey Innovative Engineering Pty Ltd v Witron Australia Pty Ltd [2023] NSWSC 981, Stevenson J considered whether a payment schedule was invalid if it failed to “indicate” reasons for withholding payment, for a specific part of the Payment Claim. 
[2.5 Com-BIFSOPA Normal Body Quote] Relevantly, part of the Payment Claim was for amounts which were the subject of an increase to the Contract Sum – which the Principal had not confirmed. Stevenson J held that the words “please adjust your claim accordingly and resubmit for approval” was sufficient to indicate to the Contractor that it’s reasons for non-payment was that it disagreed with the repricing of the contract.
[2.5 Com-BIFSOPA Normal Body Quote] However, the Payment Claim also included a claim for variations. Stevenson J found that the Payment Schedule did not indicate at all why there was to be no payment for those aspects of the Payment Claim, nor any indication of any reasons for withholding payment. Stevenson J held that a payment schedule must address the entire Payment Claim, as the purpose of a Payment Schedule is to identify each of the amounts in dispute and why.  On that basis, the Court held that the Payment Schedule was invalid.
[2.5 Com-BIFSOPA Normal Body Quote] In Roberts Construction Group Pty Ltd v Drummond Carpentry Services Pty Ltd [2024] VSC 246, Niall JA found that an email was not a payment schedule within the meaning of section 15 of the Vic Act (equivalent to section 69 of the QLD Act) on the basis that it did not provide a substantive reason for withholding payment as required by s 15(3) of the Vic Act (equivalent to section 69(c) of the BIF Act). Rather, it was held that ‘[t]he emails amount to little more than a bare assertion that further substantiation is required’, and Niall JA also observed at [98] that:
[2.5 Com-BIFSOPA Normal Body Quote] [98] When a respondent to a payment claim disputes the quantum of a payment claim and seeks that the claimant submit another ‘accurate realistic figure’, that does not state what it proposes to pay but invites a further, lesser claim to be made. The purpose of ensuring the timely payment of progress payments and narrowing the area of dispute would not be served by treating such a communication as sufficient to constitute the giving of reasons as required by s 15(3).
[2.5 Com-BIFSOPA Normal Body Quote] In Actif Concrete v Design Builders [2025] VCC 275, Lauritsen J held that an email requesting clarification about a payment claim did not constitute a payment schedule under sections 15(2)(b) and (3) of the Victorian Act (equivalent to sections 69(b) and (c) BIFSOPA). His Honour emphasised that for a document to qualify as a payment schedule, it must indicate the amount the respondent proposes to pay, if any, and provide reasons for withholding payment with sufficient particularity.
[2.5 Com-BIFSOPA Normal Body Quote] Lauritsen J held that the email, which stated “Could you expand on the difference between these 2 quotes as the first item has gone from 332,130 to 388,550 against the same wording: “the boys in Melbourne have asked me to check it & need a few more words against the new value”, did not indicate an intention to dispute the claim or specify a proposed payment amount. In coming to this decision, Lauritsen J applied Façade Treatment Engineering Pty Ltd v Brookfield Multiplex Constructions Pty Ltd [2016] VSCA 247, where the Victorian Court of Appeal reaffirmed for a document to qualify as a Payment Schedule, it must do more than merely question a claim or seek additional information.
[2.5 Com-BIFSOPA Normal Body Quote] In CPB Contractors Pty Ltd & Ors v MSS Projects (NSW) Pty Ltd t/as MSS Steel & Ors [2025] QSC 239, Johnstone J held that a payment schedule satisfied the requirements of s 69(c) of the BIF Act where the respondent’s reasons for paying less than the sum claimed were stated with sufficient clarity to define the scope of the dispute. In this case, both parties were aware that a component of the payment claim had been descoped by direction, and the respondent expressly stated that it proposed to pay $0 in relation to that descoping claim, expressly referring to the direction. Furthermore, the respondent adequately explained its calculations of liability for variation claims by reference to the figures in the “Certified this Month” section of Schedule 2 of the payment schedule. Johnstone J agreed with the reasoning of Chesterman J in Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA) Pty Ltd & Ors [2007] QSC 333 that there is no particular way the respondent must respond and the respondent is not required to provide a detailed spreadsheet in response to a payment claim.
[2.2 Com-BIFSOPA Heading 3] 69.7    ‘States why’ as opposed to ‘indicates’
[2.4 Com-BIFSOPA CDI Normal Body Text] The requirement of section 69(c) is for the respondent to ‘state’ any reasons for withholding payment.
[2.4 Com-BIFSOPA CDI Normal Body Text] Interpreting the NSW Act, in Multiplex Constructions Pty Ltd v Luikens & Anor [2003] NSWSC 1140, Palmer J said: 
[2.5 Com-BIFSOPA Normal Body Quote] A respondent to a payment claim cannot always content itself with cryptic or vague statements in its payment schedule as to its reasons for withholding payment on the assumption that the claimant will know what issue is sought to be raised. Sometimes the issue is so straightforward or has been so expansively agitated in prior correspondence that the briefest reference in the payment schedule will suffice to identify it clearly.
[2.4 Com-BIFSOPA CDI Normal Body Text] In contrast to the requirement to ‘state’ reasons under the Act, section 14(3) of the NSW Act requires a respondent to ‘indicate’ the reasons for withholding payment. In Multiplex Constructions Pty Ltd v Luikens & Anor [2003] NSWSC 1140, Palmer J said the following of this requirement: 
[2.5 Com-BIFSOPA Normal Body Quote] Section 14(3) of the Act, in requiring a respondent to “indicate” its reasons for withholding payment, does not require that a payment schedule give full particulars of those reasons. The use of the word “indicate” rather than “state”, “specify” or “set out”, conveys an impression that some want of precision and particularity is permissible as long as the essence of “the reason” for withholding payment is made known sufficiently to enable the claimant to make a decision whether or not to pursue the claim and to understand the nature of the case it will have to meet in an adjudication.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Perform (NSW) Pty Ltd v MEV-AUS Pty Ltd [2009] NSWCA 157, the New South Wales Court of Appeal stated that it considered the word ‘indicate’ to be an ordinary word "to be applied in a common sense practical manner and often where the provider and the recipient of the payment schedule will have debated a claim to payment in prior correspondence" (at [49]). Accordingly, and in an extension of the principles outlined in Multiplex Constructions Pty Ltd v Luikens [2003] NSWSC 1140, the New South Wales Court of Appeal in Perform held that the Act does not exclude incorporation by reference to material extrinsic to a payment schedule, particularly in the case of reference to the contract or prior correspondence between the provider and the recipient of the payment schedule.
[2.5 Com-BIFSOPA Normal Body Quote] See also Pacific General Securities Ltd v Soliman and Sons Pty Ltd [2006] NSWSC 13, [71] per Brereton J.
[2.2 Com-BIFSOPA Heading 3] 69.8    Multiple documents
[2.4 Com-BIFSOPA CDI Normal Body Text] In Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2013] QCA 386, the Queensland Court of Appeal said that: 
[2.5 Com-BIFSOPA Normal Body Quote] It is not implicit in s 18 that a payment schedule must consist of only one document. It would be surprising if the parliament had so intended. In substantial construction disputes, it is common place for documents disputing the validity of payment claims to contain schedules or appendices. Where more than one document is alleged to constitute a payment schedule, the validity or otherwise of the allegation must be tested by an analysis of the documents in question. It may be that one such document, although not part of a payment schedule, will be relevant to the assessment of whether the other documents constitute a payment schedule.
[2.4 Com-BIFSOPA CDI Normal Body Text] Multiple documents may, and often will, constitute a payment schedule. On the relevant approach to be taken under BCIPA, the Queensland Court of Appeal said: 
[2.5 Com-BIFSOPA Normal Body Quote] If, as the parties accepted, only one payment schedule may be served in response to a payment claim, as a general proposition, whether a document is a payment schedule is to be determined by reference to the contents of the document and, in some circumstances, any accompanying or supplementary document.
[2.4 Com-BIFSOPA CDI Normal Body Text] Accompanying documents may be provided at the same time as the payment schedule. This was the case in Baulderstone Hornibrook Pty Ltd v Queensland Investment Corporation [2006] NSWSC 522, where a box containing eight folders of documents was served upon the claimant, with the top document in the top folder specifically indicating that it was a payment schedule, the fact that the covering letter, which was delivered with the box of documents, did not state that what was being served was a payment schedule, did not result in the payment schedule not being a valid payment schedule for the purposes of the NSW Act.
[2.2 Com-BIFSOPA Heading 3] 69.9    Incorporation of documents
[2.4 Com-BIFSOPA CDI Normal Body Text] On the incorporation into a payment schedule of documents which have been provided earlier and incorporated by reference, in Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd [2013] QCA 386 the Queensland Court of Appeal accepted under BCIPA ‘that there can be no objection in principle to an obviously incomplete purported payment schedule being completed, within time, by the provision of the omitted material’. However, the court qualified this statement by providing that: 
[2.5 Com-BIFSOPA Normal Body Quote] A claimant cannot be expected on its own initiative to compose a payment schedule for the respondent by assembling miscellaneous documents received from the respondent until the assembled materials satisfy the requirements of s 18 of the Act.
[2.5 Com-BIFSOPA Normal Body Quote] In the context of the NSW Act, in Perform (NSW) Pty Ltd v MEV-AUS Pty Ltd & Anor [2009] NSWCA 157, Giles JA (McColl and Young JJA agreeing) held that ‘indication within s 14(3) does not exclude what the adjudicator described as incorporation by reference of material extrinsic to the payment schedule’.

```
