# Element-page brief — ps-within-time
**Title:** Within time (15 BD or shorter contractual)
**Breadcrumb:** Requirements of a payment schedule
**Anchor id:** `ps-within-time`
**Output file:** `bif_guide_build/v3/pages/page_ps-within-time.html`

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

### `statute/chapter_3/section_076.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 76 — Responding to payment claim
(1) If given a payment claim, a respondent must respond to the
payment claim by giving the claimant a payment schedule
within whichever of the following periods ends first—
(a) the period, if any, within which the respondent must
give the payment schedule under the relevant
construction contract;
(b) 15 business days after the payment claim is given to the
respondent.
Maximum penalty—100 penalty units.
Note—
A failure to give a payment schedule as required under this section is
also grounds for taking disciplinary action under the Queensland
Building and Construction Commission Act 1991.
(2) However, the respondent is not required to give the claimant
the payment schedule if the amount claimed in the payment
claim is paid in full on or before the due date for the progress
payment to which the payment claim relates.
(3) If the respondent gives the claimant a payment schedule, the
respondent must pay the claimant the amount proposed in the
payment schedule no later than the due date for the progress
payment to which the payment schedule relates.
Maximum penalty—100 penalty units.

(4) Subsection (3) does not apply to an amount to the extent the
respondent is required to retain the amount under chapter 3,
part 4A.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_076.txt`
```
# Annotated BIF Act source — Section 76
# Chapter: CHAPTER 3 – Progress payments
# Section title: Responding to payment claim
# DOCX paragraphs: 2260-2313

[2 Com-BIFSOPA Heading 1] SECTION 76 – Responding to payment claim 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 76    Responding to payment claim 
[1.3 BIFSOPA level 1 (CDI)] If given a payment claim, a respondent must respond to the payment claim by giving the claimant a payment schedule within whichever of the following periods ends first— 
[1.4 BIFSOPA level 2 (CDI)] the period, if any, within which the respondent must give the payment schedule under the relevant construction contract; 
[1.4 BIFSOPA level 2 (CDI)] 15 business days after the payment claim is given to the respondent. 
[1.1 BIFSOPA Body Text] Maximum penalty—100 penalty units. 
[1.6 BIFSPOA example/note] Note—
[1.7 BIFSOPA example/note body] A failure to give a payment schedule as required under this section is also grounds for taking disciplinary action under the Queensland Building and Construction Commission Act 1991. 
[1.3 BIFSOPA level 1 (CDI)] However, the respondent is not required to give the claimant the payment schedule if the amount claimed in the payment claim is paid in full on or before the due date for the progress payment to which the payment claim relates.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 76 has combined the sections that previously appeared under section 18 and section 18A BCIPA and has inserted the definition of a payment schedule under section 69.
[2.2 Com-BIFSOPA Heading 3] 76.1    Purpose of a payment schedule
[2.4 Com-BIFSOPA CDI Normal Body Text] The purpose of a payment claim and payment schedule, once exchanged, is to define clearly and as early as possible, the issues in dispute between the parties.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA Pty Ltd) & Ors [2007] QSC 33 (on the role of a payment schedule under BCIPA) Chesterman J said the following: 
[2.5 Com-BIFSOPA Normal Body Quote] The whole purpose of such a document is to identify what amounts are in dispute and why. The delivery of a payment claim and a payment schedule is meant to identify, at an early stage, the parameters of a dispute about payment for the quick and informal adjudication process for which the Act provides. If a builder wishes to take advantage of the Act to dispute the claim it must comply with its provisions and must, relevantly, take the trouble to respond to a payment claim in the manner required by the Act. The process is not difficult. The applicant was required to identify those parts of the claim which it objected to paying and to say what the grounds of its objection were.
[2.4 Com-BIFSOPA CDI Normal Body Text] The function of a payment schedule is to ‘identify the scope of the dispute’: Clarence Street Pty Ltd v Isis Projects Pty Ltd [2005] NSWCA 391, [31] (Mason P, Giles and Santon JJA agreeing).
[2.4 Com-BIFSOPA CDI Normal Body Text] However, the case law does need to be considered in the context of the introduction of the right in an adjudication on a complex payment claim, for an adjudication response to contain “new reasons”, and for a claimant to submit a reply to such new reasons.
[2.2 Com-BIFSOPA Heading 3] 76.2    What constitutes a payment schedule?
[2.4 Com-BIFSOPA CDI Normal Body Text] In NewGrow Pty Ltd v Buxton Constructions (Vic) Pty Ltd [2019] VCC 464, Cosgrave J was required to consider the validity of a number of payment schedules served by the defendant and determine whether they were issued in compliance with section 15 of the VIC Act (equivalent of section 76 of the Act). The defendant had issued a schedule in response to a payment claim within the time frame imposed by the VIC Act and reissued a second schedule for the same claim outside the time frame. For the first schedule, the court held that given the significant number of errors contained in the payment schedule, a reasonable person would be justifiably confused about whether the schedule related to the corresponding claim, such that the schedule failed to satisfy section 15 of the VIC Act. As a result of the first schedule’s invalidity, the second schedule which revised the first could not be valid. Further, it was served out of time and could not be a schedule in its own right.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Vannella Pty Limited atf Capitalist Family Trust v TFM Epping Land Pty Ltd [2019] NSWSC 1379, Henry J held that a valid payment schedule does not need to be a formal document, but it must at the very least identify the payment claim to which it is responding, what the respondent proposes to pay instead, and what parts of the claim are objected to and why. In this case, Decon had served a progress claim on the defendants by email under cover of a letter from Decon’s solicitors referring to “our client’s Progress Claim dated 3 June 2019”. The defendant’s responded on 14 June with an email, “Your Client’s Claims”. Decon subsequently sought summary judgment against the defendants for the amount claimed, on the basis that the defendants had failed to submit a valid payment schedule. The Supreme Court held that an email entitled “Your Client’s Claims” was not a valid payment schedule because it was far too broad and had not directly responded to the payment claim, as it could be construed to refer to more than one of the past claims.
[2.2 Com-BIFSOPA Heading 3] 76.3    Section 76(1) – ‘giving the claimant a payment schedule’
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 18(1) and section 18A(1) of BCIPA was drafted in terms of the respondent ‘serving a payment schedule’ on the claimant. This contrasts with section 14(1) of the NSW Act, which is drafted in terms of the respondent ‘providing’ the payment schedule. The introduction of the Act has again change the phrase, which now reads as ‘giving’. 
[2.4 Com-BIFSOPA CDI Normal Body Text] This distinction should be noted, particularly as the use of the word ‘provide’ under the NSW Act has been held to be substantive. In Falgat Constructions Pty Ltd v Equity Australia Corporation Pty Ltd [2006] NSWCA 259, Hodgson JA said:
[2.5 Com-BIFSOPA Normal Body Quote] The use of the word “provide” rather than the word “serve” does carry a suggestion that a different meaning is intended, and that accordingly s.31 does not apply in the case of the word “provide”. Against this, however, I do not think the legislature would have (1) used a problematic word like “provide” with the intention that it have a different meaning from “serve”, (2) given useful instructions as to how service may be effected, yet (3) given no instructions whatsoever as to how provision may be effected. When this consideration is combined with the consideration raised in the previous paragraph, in my opinion this justifies the conclusion, reached by the primary judge in this case, that “provide” does not mean anything different from “serve”, and that s.31 applies to “provision” as well as to “service”.
[2.3 Com-BIFSOPA Heading 4] Service by a solicitor
[2.4 Com-BIFSOPA CDI Normal Body Text] In Emag Constructions Pty Ltd v Highrise Concrete Contractors (Aust) Pty Ltd [2003] NSWSC 903, Einstein J considered that (at [59]):
[2.5 Com-BIFSOPA Normal Body Quote] In my view the character of the subject legislation is such that general principles of actual or ostensible authority in solicitors to receive service of copies of relevant notices must yield to the strictures of the strict requirement to prove service. The service provisions of the Act require to be complied with in terms. Prudence dictates that those responsible for complying with the service provisions take steps to be in a position to strictly prove service in the usual way. One only example of the difficulties which may arise is where a solicitor who may have been instructed to act in relation to an adjudication application has his/her instructions withdrawn. There are no provisions similar to those to be found in the Supreme Court Rules 1970 for notices of ceasing to act and the like. The Act here under consideration simply proceeds by requiring particular steps to be taken by the parties and by the adjudicator and proof of strict compliance with the Act is necessary for the achievement of the quick and efficient recovery of progress payments and resolution of disputes in that regard.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour applied these comments in Taylor Projects Group Pty Ltd v Brick Dept Pty Ltd [2005] NSWSC 439, holding that a solicitor was unable to serve a purported payment schedule, as the general principles of actual or ostensible authority ‘must yield to the strictures of the Act’.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, in Baulderstone Hornibrook Pty Ltd v Queensland Investment Corporation [2007] NSWCA 9, the Court of Appeal rejected a submission that a payment schedule was not validly served by the respondent because it was served by the respondent’s solicitors. In holding that the respondent’s solicitor’s had authority to serve the payment schedule, the Court of Appeal considered (after a detailed review of the retained agreement) the nature of the retainer, prior conduct, and the commercial sense of the conclusion.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer also to the further commentary under section 102, below.
[2.3 Com-BIFSOPA Heading 4] Service by email outside of business hours
[2.4 Com-BIFSOPA CDI Normal Body Text] In CKP Constructions Pty Ltd v Gabba Holdings Pty Ltd [2016] QDC 356, the applicant commenced proceedings seeking judgment of the full amount of a progress claim submitted on 25 August 2016 pursuant to s 19(3)(a)(i) of BCIPA (equivalent to section 78(2)(a) of the Act). There was no dispute that the progress claim was a payment claim under BCIPA and that GH initially did not serve a payment schedule in response.
[2.4 Com-BIFSOPA CDI Normal Body Text] As required by section 20A of BCIPA (equivalent to section 99 of the Act), CKP Constructions Pty Ltd (CKP) served a notice on Gabba Holdings Pty Ltd (GH) of its intention to apply for judgment of the payment claim. GH then had 5 business days to serve a payment schedule. At 11:45pm on the 5th business day after receiving the section 20A notice, GH sent an email attaching a payment schedule to CKP. 
[2.4 Com-BIFSOPA CDI Normal Body Text] McGill DCJ held that the payment schedule had been validly served because the Contract permitted service by email and, pursuant to section 24(1)(a) of the Electronic Transactions Act 2001 (Qld), the payment schedule was served at the time when it became capable of being retrieved by GH at its email address, which was 11:45pm.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour held that, as GH had served a payment schedule, CKP was not entitled to judgment under s 19(3)(a)(i) of the Act. 18A.
[2.2 Com-BIFSOPA Heading 3] 76.4    Section 76(1) – ‘giving…within whichever of the following periods ends first’
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been held that, displacement of the statutory time period for the service of a payment schedule by the contract requires ‘clear contextual support for a necessary implication’ of such displacement: Theiss Pty Ltd v Lane Cove Tunnell Nominee Company Pty Ltd [2008] NSWSC 729, [21] (Hammerschlag J); Tailored Projects Pty Ltd v Jedfire Pty Ltd [2009] 2 Qd R 171; [2009] QSC 32, [21]-[22] (Douglas J).
[2.4 Com-BIFSOPA CDI Normal Body Text] The relevant test was set out in Theiss Pty Ltd v Lane Cove Tunnell Nominee Company Pty Ltd [2008] NSWSC 729 by Hammerschlag J was, in the context of the NSW Act, as follows:
[2.5 Com-BIFSOPA Normal Body Quote] The test is thus whether there is clear contextual support for a necessary implication that the Contract has supplanted the ten-day period in s 14(4)(b)(ii) with the four business day period in cl 14.3A.
[2.5 Com-BIFSOPA Normal Body Quote] Undoubtedly for a Contract to require a particular time as contemplated by s 14(4)(b)(i) it need not make express reference to that section.
[2.4 Com-BIFSOPA CDI Normal Body Text] The decision of Hammerschlag J was upheld on appeal in Theiss Pty Ltd v Lane Cove Tunnel Nominee Company Pty Ltd [2009] NSWCA 53.
[2.4 Com-BIFSOPA CDI Normal Body Text] This approach to the construction of the period in which to serve a payment schedule was accepted by Douglas J in Tailored Projects Pty Ltd v Jedfire Pty Ltd [2009] QSC 32; [2009] 2 Qd R 171, 179 [23].
[2.4 Com-BIFSOPA CDI Normal Body Text] In Allencon Pty Ltd v Palmgrove Holdings Pty Ltd trading as Carruthers Contracting [2023] QCA 6, the Queensland Court of Appeal considered when a payment schedule was due based on whether the relevant contract prescribed a period for the delivery of a payment schedule or whether the period in the BIFSOPA applied.  
[2.4 Com-BIFSOPA CDI Normal Body Text] As per s 76(1) of the BIFSOPA, a respondent must respond to the payment claim by providing the claimant with a payment schedule within whichever of the following periods end first:
[2.4 Com-BIFSOPA CDI Normal Body Text] The period, if any, within which the respondent must give the payment schedule under the relevant construction contract; or
[2.4 Com-BIFSOPA CDI Normal Body Text] 15 business days after the payment claim is given to the respondent.
[2.4 Com-BIFSOPA CDI Normal Body Text] The relevant extract of the clause in the contract was as follows: 

[2.5 Com-BIFSOPA Normal Body Quote] Within 21 days after receipt of a claim for payment, the Main Contractor’s Representative shall issue to the Main Contractor and to the Subcontractor a payment certificate stating the payment which, in the opinion of the Main Contractor’s Representative, is to be made by the Main Contractor to the Subcontractor or by the Subcontractor to the Main Contractor. The Main Contractor’s Representative shall set out in the certificate the calculations employed to arrive at the amount and, if the amount is more or less than the amount claimed by the Subcontractor, the reasons for the difference.
[2.4 Com-BIFSOPA CDI Normal Body Text] The claimant asserted that the respondent failed to provide its payment schedule before the earlier expiry date which arose under the contract. The respondent argued that the contract did not provide a timeframe to deliver the payment schedule and contended that it had provided the payment schedule prior to the expiry of the date which arose under the operation s 76(1)(b) BIFSOPA.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court held that because a period of 21 calendar days was stipulated in the contract to deliver a payment schedule, this was the date which the payment schedule had to be served. Therefore, the respondent had failed to provide the payment schedule by this date and was liable to pay the full amount.
[2.5 Com-BIFSOPA Normal Body Quote] The Court of Appeal distinguished the contract in this case from the contract in Thiess Pty Ltd v Lane Cove Tunnel Nominee Co Pty Ltd [2009] NSWCA 53. The Court explained that the contract in Theiss provided a requirement for an Independent Verifier’s Certificate to be accompanied with any progress payment, which indicated that the contract clause related to a payment certificate under the contract and not a payment schedule “reply to the contractual progress claim and not a reply to a statutory payment claim”.
[2.5 Com-BIFSOPA Normal Body Quote] In Roberts Co (NSW) Pty Ltd v Sharvain Facades Pty Ltd (Administrators Appointed) [2025] NSWCA 161, Hammerschlag CJ held that the 10-business day period in section 14(4)(b)(ii) of the NSW SOP Act (equivalent to section 76(1)(b) of the QLD Act) commences from the date a payment claim is served, and that parties may only agree to shorten, not lengthen, that period. The Court found that Sharvain’s payment claim was served on 28 February 2025 and the payment schedule issued by Roberts on 17 March 2025 was therefore, certainly out of time. In coming to this conclusion, his Honour rejected Roberts’ reliance on a deeming clause that postponed service to 3 March 2025, holding that such a clause could not delay the operation of the statutory timeframe. McHugh JA agreed, confirming that the contractual deeming clause had no effect on the operation of section 14(4), and that Roberts became liable for the full amount claimed.

```
