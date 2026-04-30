# Element-page brief — ps-served
**Title:** Served on the claimant
**Breadcrumb:** Requirements of a payment schedule
**Anchor id:** `ps-served`
**Output file:** `bif_guide_build/v3/pages/page_ps-served.html`

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

### `statute/chapter_3/section_102.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 102 — Service of notices
(1) A notice or other document that, under this chapter, is
authorised or required to be given to a person may be given to
the person in the way, if any, provided under the relevant
construction contract.
Example—
A contract may allow for the service of notices by email.
(2) Subsection (1) is in addition to, and does not limit or exclude,
the Acts Interpretation Act 1954, section39 or the provisions
of any other law about the giving of notices.
Page 139 Current as at 27 April 2025

(3) To remove any doubt, it is declared that nothing in this Act—
(a) excludes the proper service of notices or documents by a
person’s agent; or
(b) requires a person’s acknowledgement of a notice or
document properly given to the person.

chapter

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

### `annotated/section_102.txt`
```
# Annotated BIF Act source — Section 102
# Chapter: CHAPTER 3 – Progress payments
# Section title: Service of notices
# DOCX paragraphs: 4013-4108

[2 Com-BIFSOPA Heading 1] SECTION 102 – Service of notices
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 102    Services of notices
[1.3 BIFSOPA level 1 (CDI)] A notice or other document that, under this chapter, is authorised or required to be given to a person may be given to the person in the way, if any, provided under the relevant construction contract. 
[1.6 BIFSPOA example/note] Example—
[1.7 BIFSOPA example/note body] A contract may allow for the service of notices by email.
[1.3 BIFSOPA level 1 (CDI)] Subsection (1) is in addition to, and does not limit or exclude, the Acts Interpretation Act 1954, section 39 or the provisions of any other law about the giving of notices.
[1.3 BIFSOPA level 1 (CDI)] To remove any doubt, it is declared that nothing in this Act— 
[1.4 BIFSOPA level 2 (CDI)] excludes the proper service of notices or documents by a person’s agent; or
[1.4 BIFSOPA level 2 (CDI)] requires a person’s acknowledgement of a notice or document properly given to the person.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 102.1    A ‘facultative’ provision
[2.4 Com-BIFSOPA CDI Normal Body Text] In Neumann Contractors Pty Ltd v Traspunt No 5 Pty Ltd [2010] QCA 119, Muir JA (Holmes and Chesterman JJA agreeing) considered that section 103 of the Act “appears to be facultative rather than mandatory in nature”.
[2.2 Com-BIFSOPA Heading 3] 102.2    Time limitations under the Act
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been said that the whole of the rationale under the NSW Act is such that strict compliance within the time limitations is critical: see Emag Constructions Pty Ltd v Highrise Concrete Contractors (Aust) Pty Ltd [2003] NSWSC 903 where Einstein J considered that:
[2.5 Com-BIFSOPA Normal Body Quote] Plainly enough the whole of the rationale underpinning the procedures laid down by the Act is directed at providing a quick and efficient set of procedures permitting recovery of progress payments and the quick resolution of disputes in that regard. Time limits under the Act are strict. The consequences of not complying with the stipulated time limits can be significant…
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] Service being effected in accordance with the Act is critical as it governs the commencement of the time limitations following such service. The consequence of non-compliance with the time limitation periods is harsh. As was submitted to the court by counsel for the plaintiff, the Act exhibits “zero tolerance” for delay. To borrow a phrase from the world of contract, and in particular conveyancing, in a real sense time is of the essence.
[2.2 Com-BIFSOPA Heading 3] 102.3    Service under the Acts Interpretation Act 1954
[2.4 Com-BIFSOPA CDI Normal Body Text] In Penfold Projects Pty Ltd v Securcorp Limited [2011] QDC 77, Irwin DCJ held email to be a ‘similar facility’ within the meaning of section 39(1) of the Act. This view was doubted in Conveyor & General Engineering Pty Ltd v Basetec Services Pty Ltd & Anor [2014] QSC 30.
[2.2 Com-BIFSOPA Heading 3] 102.4    Service on a corporation
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 109X of the Corporations Act 2001 (Cth) provides that a document may be served on a company by:
[2.6 Com-BIFSOPA bullet] leaving it at, or posting it to, the company's registered office (section 109X(1)(a)); or
[2.6 Com-BIFSOPA bullet] delivering a copy of the document personally to a director of the company who resides in Australia or in an external Territory (section 109X(1)(b)).
[2.4 Com-BIFSOPA CDI Normal Body Text] Further provision is made under section 109X for service where a liquidator or administrator of the company has been appointed.
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 109X is facultative and does not represent an exclusive code for service of documents upon corporations: Howship Holdings Pty Ltd v Leslie (No 2) (1996) 41 NSWLR 542.
[2.4 Com-BIFSOPA CDI Normal Body Text] Pursuant to section 109X(6)(a) of the Corporations Act 2001 (Cth), section 109X does not affect any other provision of the Corporations Act, or any provision of another law, that permits a document to be served in a different way. 
[2.2 Com-BIFSOPA Heading 3] 102.5    Service on a post office box
[2.4 Com-BIFSOPA CDI Normal Body Text] Service on a post office box is not valid service in accordance with either section 39 of the Acts Interpretation Act 1954 (Qld) or section 109X of the Corporations Act 2001 (Cth): CMF Projects Pty Ltd v Masic Pty Ltd & Ors [2014] QSC 209, [23] (Daubney J).
[2.4 Com-BIFSOPA CDI Normal Body Text] Accordingly, on this basis, unless service of a notice under the Act to a post office pox is provided for under the relevant construction contract, service of a notice under the Act to a post office box will not, by itself, be valid service: see CMF Projects Pty Ltd v Masic Pty Ltd & Ors [2014] QSC 209, [16], [23] (Daubney J).
[2.2 Com-BIFSOPA Heading 3] 102.6    Service by email    
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 24(1) of the Electronic Transactions (Queensland) Act 2001 (Qld) provides that, unless otherwise agreed, the time of receipt of an electronic communication is:
[2.6 Com-BIFSOPA bullet] ‘the time the electronic communication becomes capable of being retrieved by the addressee at an electronic address designated by the addressee’ (section 24(1)(a)); or
[2.6 Com-BIFSOPA bullet] at another electronic address of the addressee, at the time when both ‘the electronic communication has become capable of being retrieved by the addressee at that address’ and ‘the addressee has become aware that the electronic communication has been sent to that address’ (section 24(1)(b)).
[2.4 Com-BIFSOPA CDI Normal Body Text] The phrase ‘capable of being retrieved’ in the context of service of an adjudication response under the NSW Act was considered in Bauen Constructions Pty Ltd v Sky General Services Pty Ltd [2012] NSWSC 1123. Sackar J held (at [77]):
[2.5 Com-BIFSOPA Normal Body Quote] The words "capable of being retrieved" are ample in their reach. They certainly do not require an email to be opened, let alone read. Again the Oxford dictionary defines "retrieve" in its primary sense as "to get or bring back from somewhere". In its secondary sense it is said to mean "to find or extract (information stored in a computer)". 
[2.4 Com-BIFSOPA CDI Normal Body Text] A document will be ‘capable of being retrieved’ notwithstanding that the document in question was caught by the spam filter on the computer system in question: Bauen Constructions Pty Ltd v Sky General Services Pty Ltd [2012] NSWSC 1123, [77] (Sackar J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In MN Builders Pty Ltd v MMM Cement Rendering Pty Ltd [2019] NSWDC 734, Abadee DCJ held that once it is known that a payment claim was actually received by the respondent, compliance with section 31 of the NSW Act (equivalent of section 102 of the Act), regarding service of the payment claim, does not come into consideration. In this case, the subcontractor had issued its claim to the builder’s foreman, who was not a ‘person’ authorised to be served with a claim under section 31. The subcontractor contended that its invoice constituted a 'payment claim' for the purposes of the NSW Act and because the builder did not serve a payment schedule within 10 business days they were liable to default judgement by reason of section 15 of the NSW Act. The builder denied liability on the basis that the payment claim had not been validly served pursuant to section 31, because the subcontract stipulated that claims were to be sent by email to the builder’s accounts team and not the foreman. Abadee DCJ held that the court did not need to concern itself with the content of the construction contract before the subcontractor’s rights to bring payment claims arose under sections 8 and 13 of the NSW Act. The District Court held that, as the claim was actually received by the builder, the payment claim had been validly served and granted summary judgment for the subcontractor pursuant to the NSW Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Pelligra Build Pty Ltd v Australian Crane & Machinery Pty Ltd [2020] VCC 545, Archer J held that service of a payment claim via email was valid despite the method of service being expressly unauthorised under the construction contract. In that case, the service of “notices” by email was expressly prohibited under the construction contract and the payment claim the subject of the dispute was served by email. Despite this, the relevant payment claim had come to the attention of the respondent even though it was not made by authorised means of service. Archer J, relying on the reasoning by Vickery J in Amasya Enterprises Pty Ltd v Asta Developments (Aust) Pty Ltd (No 2) [2015] VSC 500, reasoned there is no mandatory requirement expressed in section 14 of the SOP Act (Vic) (equivalent to section 75 of BIFSOPA) for payment claims to be served exclusively by the methods set out in section 50 (equivalent to section 102 of BIFSOPA), which requires that a notice or document required to be given under the Vic Act is to be served on a person by delivering it personally; by post; or by lodging it at the person’s ordinary place of business. Accordingly, Archer J held that a standardised contractual clause will not frustrate a claimant’s right to serve a payment claim under the SOP Act (Vic).  
[2.4 Com-BIFSOPA CDI Normal Body Text] In Spirito Development Pty Ltd v Sinjen Group Pty Ltd [2020] VCC 1368, Justice Woodward declared that a payment claim was invalid on the basis that it was served prior to the relevant reference date. In that case, the claimant argued that it emailed the payment claim at 10:17 pm on a Friday, with the next reference date due to accrue on the following Monday. The contractual notice provision was silent as to the calculation of time for acts performed outside business hours. In construing section 50 of the Vic Act (equivalent to section 102 BIFSOPA) his Honour noted that it “says nothing about service by emails”, while it says that a facsimile received after 4.00 pm on any day “must be taken to have been received on the next business day”. In holding that the payment claim was invalid for being served prior to the relevant reference date, his Honour considered that emails are instantaneous and, in application of the reasons in MKA Bowen v Carelli Constructions [2019] VSC 436, “if a reference date is not available on the date that a payment claim is taken to have been served, the payment claim is invalid regardless of when it was received”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Piety Constructions Pty Ltd v Hville FCP Pty Ltd [2022] NSWSC 1318, Stevenson J held that if a document is electronically served, it is taken to have been “provided” per s 14 of the NSW SOP Act when it is accessed and viewed. In this case, the contract contained a clause that stated that any documents delivered after 4:30pm on a business day, were “deemed to be given at 9:30am on the next business day”. Stevenson J held that to the extent that this clause purported to amend the requirements in the Act for a document, such as a payment claim, to be “provided” within a certain timeframe, the clause was invalid as it purported to change the operation of the word “provide” under the NSW Act. Therefore, a document required to be provided under security of payment, which was delivered electronically, will arguably be “provided” on the day it was accessed and viewed irrespective of any deeming clause in a contract. Notably, this decision is inconsistent with earlier authorities regarding service, which the Court was not taking to.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Fredon Infrastructure Pty Ltd v Hitachi Rail GTS Australia Pty Ltd [2024] NSWSC 1244, Stevenson J considered when two Payment Claims delivered electronically were taken to have been effectively served. There were two Payment Claims because the parties were dealing under two different contracts for the one project.
[2.4 Com-BIFSOPA CDI Normal Body Text] Fredon (the Claimant) delivered the Payment Claims electronically (via Aconex), on 21 February 2024. Hitachi (the Respondent) did not issue Payment Schedules until 7 March 2024, which fell outside the 10 Business Day timeframe to respond to the Payment Claims delivered on 21 February 2024. Fredon’s position was that this meant the Payment Schedule was invalid. However, Hitachi argued that it did not become aware of the Payment Claim until 22 February 2024, meaning its Payment Schedule was valid.
[2.4 Com-BIFSOPA CDI Normal Body Text] Stevenson J was satisfied that as a matter of fact, Hitachi “became aware” of the Aconex messages that contained the Payment Claims on 21 February 2024 because:
[2.6 Com-BIFSOPA bullet] the Hitachi representatives whom the Aconex messages were delivered to were present at work at the time the 21 February 2024 Payment Claims were served;
[2.6 Com-BIFSOPA bullet] one of the Hitachi representatives whom the Aconex messages were delivered left a voice message at 8:33am on 22 February 2024 suggesting receipt occurred on the previous day; and
[2.6 Com-BIFSOPA bullet] one of the Hitachi representatives whom the Aconex messages were delivered managed a register which recorded when payment claims were received, but failed to depose as to the contents of that register under cross examination.
[2.4 Com-BIFSOPA CDI Normal Body Text] Although Stevenson J had already determined the Payment Claim was served on 21 February 2024 (based on the abovementioned facts), His Honour also applied section 13A(1) of the Electronic Transactions Act 2000 (NSW) (the ETA).
[2.4 Com-BIFSOPA CDI Normal Body Text] Stevenson J concluded that because the Aconex correspondence was sent to an electronic address other than one which was designated under the Contract, section 13A(1) of the ETA was applicable. His Honour noted that the “time of receipt” would be the time that Fredon became aware of the communications and when the communication had reached the electronic address’.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Claire Rewais and Osama Rewais t/as McVitty Grove v BPB Earthmoving Pty Ltd [2024] NSWSC 1271, McGrath J affirmed and reiterated the principles regarding service under the NSW Act. In essence, the onus of proof is on the claimant who must show either:
[2.6 Com-BIFSOPA bullet] the payment claim was served in accordance with the facultative provisions in s 31 of the NSW Act (in which case the time of service will be the time at which the s 31 requirements were satisfied); or
[2.6 Com-BIFSOPA bullet] if they were not served in accordance with s 31, that they were in fact brought to the respondent’s attention and the time of service will be the time at which the respondent became aware of the payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] In this case, McGrath J held that the Claimant failed to prove the payment claim was served in accordance with s31(1)(d) of the NSW Act. That section requires a claimant to prove that a respondent ‘specified’ its email address for the purpose of ‘service of documents of that kind’, being documents required to be served on him under the NSW Act. In this case, it was found the Respondent merely provided his email address for the purpose of receiving a quote or indication of costing for the works, but a quote is not a document required to be served pursuant to the NSW Act. Further, the Claimant also failed to prove that the payment claim was brought to the Respondent’s attention. At paragraph [143] His Honour held that:
[2.5 Com-BIFSOPA Normal Body Quote] I consider that BPB has also failed to prove that the payment claim was brought to the Rewaises’ attention prior to 11 June 2024. While the evidence shows that Dr Rewais would, from time-to-time, send emails from and receive emails to his account, there is no evidence of any email communications from Dr Rewais to BPB between 24 April 2024 and 11 June 2024, no evidence that BPB notified the Rewaises over text message or in person of the payment claim that had been emailed, no evidence that he opened the 24 April emails, no evidence that he was aware the 24 April emails were in his inbox and no evidence that Dr Rewais read the 24 April emails prior to 11 June 2024. Moreover, Dr Rewais was not challenged in cross-examination to the contrary. More is required for BPB to meet its onus of proof.”
[2.5 Com-BIFSOPA Normal Body Quote] In Roberts Co (NSW) Pty Ltd v Sharvain Facades Pty Ltd (Administrators Appointed) [2025] NSWCA 161, McHugh JA held that service under section 31(1)(d) of the NSW SOP Act (equivalent to section 102(1) of the QLD Act) occurs when a document is emailed to an address specified by the recipient for service and becomes capable of being retrieved. The Court found that Sharvain’s payment claim was served on 28 February 2025 when it was uploaded to Payapps and sent to Robert’s nominated email address. His Honour confirmed that there is no requirement under section 31 for the recipient to actually access or become aware of the communication. In coming to this conclusion, McHugh JA followed followed Claire Rewais and Osama Rewais t/as McVitty Grove v BPB Earthmoving Pty Ltd [2025] NSWCA 103 and distinguished Parkview Constructions v Total Lifestyle Windows Pty Ltd [2017] NSWSC 194. The Court reaffirmed that section 31 governs the mechanics of valid service under the Act, irrespective of awareness.
[2.5 Com-BIFSOPA Normal Body Quote] In Kumar v Frankies Cranes [2025] NSWSC 1264, Peden J held that handing over a business card, "without more", does not amount to specification of an email address for service under s 31(1)(d) of the NSW Act (equivalent to s 102 of the Act). Applying Rewais v BPB Earthmoving Pty Ltd, her Honour confirmed that "specification" may be express or inferred from conduct, and quoted McHugh JA's guidance that given "the jurisdictional significance of service", parties should obtain express confirmation before using email addresses for service under the Act. Her Honour also applied QC Communications NSW Pty Ltd v CivComm Pty Ltd and BCFK Holdings Pty Ltd v Rork Projects Pty Ltd, confirming that in the absence of specification the claimant must prove actual receipt, which was not established.

[2.2 Com-BIFSOPA Heading 3] 102.7    Dropbox and Cloud
[2.4 Com-BIFSOPA CDI Normal Body Text] In Conveyor & General Engineering Pty Ltd v Basetec Services Pty Ltd & Anor [2014] QSC 30, Philip McMurdo J held that service of an adjudication application by ‘Dropbox’ was not effective service under either section 39 of the Acts Interpretation Act 1954 (Qld) or 109X of the Corporations Act 2001 (Cth) because:
[2.6 Com-BIFSOPA bullet] the electronic communication (an email) did not contain the adjudication application, but rather the means to access the adjudication application such that the whole adjudication application had not been ‘sent’ to the relevant office (at [29]-[30]); and
[2.6 Com-BIFSOPA bullet] the adjudication application was not ‘left’ or ‘sent’ to the relevant office at least until it had been opened on the Dropbox site ‘and probably not until its contents had been downloaded to a computer’ at the relevant office (at [32]).
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour further held that actual service had not occurred within the statutory time period as ‘it was insufficient for the document and its whereabouts to be identified absent something in the nature of its receipt’.
[2.4 Com-BIFSOPA CDI Normal Body Text] This case likely applies to other forms of similar ‘cloud storage’ document hosting services.
[2.4 Com-BIFSOPA CDI Normal Body Text] In BCS Infrastructure Support Pty Ltd v Jones Lang Lasalle (NSW) Pty Ltd [2020] VSC 739, Stynes J found that service of a payment claim via a cloud based information system was not effective until the payment claim was identified and read by the respondent. In that case, the claimant uploaded the payment claim to Corrigo. It was stored and available for the respondent to retrieve from Corrigo from the time it was uploaded. An email was sent by Corrigo to the respondent notifying its existences, but the respondent did not become aware of the payment claim until a later period. The Court found that because Corrigo was an electronic address for the purposes of the Electronic Transactions Act 2000 (Vic), the time for receipt of the payment claim by the respondent arose when the respondent became aware that the payment claim was sent to Corrigo.
[2.2 Com-BIFSOPA Heading 3] 102.8    ‘Service’ and ‘receipt’ of documents under the Act
[2.4 Com-BIFSOPA CDI Normal Body Text] There was a distinction between ‘service’ and ‘receipt’ of documents under BCIPA. The distinction is no longer present in the Act because of the attempt to use simple, everyday language. The following commentary remains relevant for context where its use appears in other jurisdictions.
[2.4 Com-BIFSOPA CDI Normal Body Text] The distinction was considered in Falgat Constructions Pty Ltd v Equity Australia Corporation Pty Ltd [2006] NSWCA 259, where Hodgson JA stated:
[2.5 Com-BIFSOPA Normal Body Quote] There remains the question of whether the time at which service or provision has taken effect is also the time at which a document is received, for the purposes of s.17(3)(b) (and see also s.17(2)(b) and s.21). I note that the word “receive” is also used in s.31(2), but used in the context of “received at that place”. In my opinion, mail delivered to a registered office or place of business is received at that place when it is put into the mail box of that registered office or place of business, without the necessity of anyone actually seeing it.
[2.5 Com-BIFSOPA Normal Body Quote] In my opinion, the word “receive” in s.17(3)(c) does not necessarily require that the document come to the notice of a person authorised to deal with the document on behalf of the claimant. In general, in my opinion, it would be satisfied once the document has arrived at the claimant’s registered office or place of business and is there during normal office hours. There may be circumstances in which service or provision has been effected within s.109X of the Corporations Act or s.31 of the Act, but the document has not been received, but I find it difficult to identify any such circumstances.
[2.4 Com-BIFSOPA CDI Normal Body Text] The decision of the New South Wales Court of Appeal in Falgat Constructions Pty Ltd v Equity Australia Corporation Pty Ltd [2006] NSWCA 259 was a departure from the previous view on ‘receipt’ under BCIPA, as expressed by McDougall J in Pacific General Securities Ltd v Soliman & Sons Pty Ltd [2005] NSWSC 378, where his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] It will be seen that there is some distinction drawn between the concept of service and the concept of receipt, although for some purposes at least, the two will coincide. Thus, in s 19, the operative concept is that of service, whilst in s 20, the operative concept is that of receipt. Again, where s 31(1)(c) is availed of, the operative concept, by virtue of subs (2), is receipt. It is receipt that both sets the clock running under s 20 and following and that provides both the fact and the time of service, at least where s 31(1)(c) is relied upon.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] The verb “receive” in its ordinary meaning denotes the taking of something into one’s hand or possession, of something given or delivered, or having something delivered or brought to one. I see no reason why the word “receive”, and its cognate forms in the Act, should not be given that ordinary English meaning. This does not mean, in the case of a corporation (at least absent any contractual stipulation to the contrary) a document must come to a particular person within a corporation before it can be received. It means that the document must come into the hand or possession of, or be delivered or brought to, someone on behalf of the corporation; or, perhaps, that otherwise somehow it comes into the hand or possession of, or is delivered or brought to, the corporation.
[2.4 Com-BIFSOPA CDI Normal Body Text] In CMF Projects Pty Ltd v Masic Pty Ltd & Ors [2014] QSC 209, Daubney J concluded on the current authorities that:
[2.5 Com-BIFSOPA Normal Body Quote] It seems now to be clear enough on the authorities that the word “receive” connotes that, whilst the document in question need not come to the attention of a particular person within the relevant office, it nevertheless does actually need to have arrived at, and thereby been received’, at the recipient’s registered office, or place of business, and be there during normal office hours.
[2.4 Com-BIFSOPA CDI Normal Body Text] Ball J applied the current authorities in QC Communications NSW Pty Ltd v CivComm Pty Ltd [2016] NSWSC 1095, stating that: 
[2.5 Com-BIFSOPA Normal Body Quote] A document will be served in accordance with the requirements of the [Building and Construction Industry Security of Payment Act 1999 (NSW)] if it actually comes to the attention of the person to be served. It is not necessary that it be served in accordance with s 31 [of the Building and Construction Industry Security of Payment Act 1999 (NSW)].
[2.4 Com-BIFSOPA CDI Normal Body Text] In arriving at his conclusion, Ball J cited Hodgson JA in Falgat Constructions Pty Ltd v Equity Australia Corporation Ltd [2006] NSWCA 259 at [58]:
[2.5 Com-BIFSOPA Normal Body Quote] In the first place, in my opinion it is clear that if a document has actually been received and come to the attention of a person to be served or provided with the document, or of a person with authority to deal with such a document on behalf of a person or corporation to be served or provided with the document, it does not matter whether or not any facultative regime has been complied with: see Howship Holdings Pty Limited v Leslie (1996) 41 NSWLR 542; Mohamed v Farah [2004] NSWSC 482 at [42]-[44]. In such a case, there has been service, provision and receipt.
[2.4 Com-BIFSOPA CDI Normal Body Text] In MGW engineering Pty Ltd (t/as forefront services) v CMOC mining Pty Ltd [2021] NSWSC 514, Justice Stevenson held that service of a payment claim on an employee of the respondent did not constitute effective service until the document came to the attention of the relevant representative of the respondent. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In that case, the payment claims were addressed to respondents representative and the superintendent under the construction contract. The court found that because both parties were not present on the day of service, it could not have been said to have been delivered personally or lodged until the payment claim came to the attention of either. The court considered section 31(1)(a) and (b) of the NSW SOP Act (no equivalent in the BIF Act) which related to the service of a document being served personally and lodged.
[2.4 Com-BIFSOPA CDI Normal Body Text] The court held that for a document to be served “personally” and “lodged” pursuant to section 31(1)(a) and (b) of the NSW SOP Act (no equivalent in the BIF Act) “more is required than simply leaving the document with an employee, no matter what that employee’s functions were nor how junior that employee was, at any location within the corporation’s business premises. Some further step, the effect to which would likely be bringing the Payment Claim to the attention of the relevantly responsible person, is necessary”.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] CMF Projects Pty Ltd v Masic Pty Ltd & Ors [2014] QSC 209. An adjudicator’s notice of acceptance was sent not to the respondent’s registered office, but to a post office box of the respondent. Daubney J noted the distinction between ‘service’ and ‘receipt’ of documents under the Act and held that the adjudicator had erred in equating deposit of the notice into the post office box with receipt of the notice.
[2.2 Com-BIFSOPA Heading 3] 102.9    Service on or by an agent of the Principal
[2.4 Com-BIFSOPA CDI Normal Body Text] In Bucklands Convalescent Hospital v Taylor Projects Group [2007] NSWSC 1514, Hammerschlag J followed the Court of Appeal in Baulderstone Hornibrook Pty Ltd v Queensland Investment Corporation [2007] NSWCA 9, holding:
[2.5 Com-BIFSOPA Normal Body Quote] …As a matter of law it does not seem to me that a person who is a Superintendent under a contract and who has certifying functions under it is incapable of being appointed as agent to respond to a payment claim under the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Lucas Stuart v Hemmes Hermitage [2009] NSWSC 477, McDougall J held an assessment by a quantity surveyor did not constitute a payment schedule. Relevantly, his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] Under the Act, a claimant is entitled to have a written reply from (or binding on) the respondent, so that the claimant can decide whether to accept the scheduled amount or to proceed to adjudication. A claimant should not be required to act upon the basis of informal indications by those who advise the respondent, even if (in a particular case) that adviser is trusted, and its advice has always been followed in the past.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Director of Housing v Structx Pty Ltd t/as Bzibuilders & Anor [2011] VSC 410, an architect issued what was purported to be a payment schedule under the VIC Act. There was no contractual provision conferring such right, however, by a letter, the architect had been appointed as a representative of the superintendent. This letter also stated that the architect has authority to issue progress certificates. In the circumstances, the Court held that the architect did possess the relevant authority.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Lewance Construction Pty Ltd v Southern Han Breakfast Point Pty Ltd [2014] NSWSC 1726, Nicholas AJ held that the superintendent’s representative had the authority to receive progress claims and payment claims, and respond by providing both a progress certificate and a payment schedule. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer also to the commentary under section 75.3 in relation to service of a payment claim by an agent of the principal.
[2.2 Com-BIFSOPA Heading 3] 102.10    Service on a firm of solicitors
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to the commentary under section 76 of the Act.

[2a BIFSOPA Chapter Heading] CHAPTER 4 – Subcontractors’ charges

```

## Other authority

### `other/aia_act_s39_service.txt`
```
# Source: Acts Interpretation Act 1954 (Qld)
# Section 39 — Service of documents

39 Service of documents
(1) If an Act requires or permits a document to be served on a
person, the document may be served—
(a) on an individual—
(i) by delivering it to the person personally; or
(ii) by leaving it at, or by sending it by post, telex,
facsimile or similar facility to, the address of the
Page 60 Current as at 1 February 2024

place of residence or business of the person last
known to the person serving the document; or
(b) on a body corporate—by leaving it at, or sending it by
post, telex, facsimile or similar facility to, the head
office, a registered office or a principal office of the
body corporate.
(2) Subsection (1) applies whether the expression ‘deliver’,
‘give’, ‘notify’, ‘send’ or ‘serve’ or another expression is
used.
(3) Nothing in subsection(1)—
(a) affects the operation of another law that authorises the
service of a document otherwise than as provided in the
subsection; or
(b) affects the power of a court or tribunal to authorise
service of a document otherwise than as provided in the
subsection.
39A Meaning of service by post etc.
(1) If an Act requires or permits a document to be served by post,
service—
(a) may be effected by properly addressing, prepaying and
posting the document as a letter; and
(b) is taken to have been effected at the time at which the
letter would be delivered in the ordinary course of post,
unless the contrary is proved.
(2) If an Act requires or permits a document to be served by a
particular postal method, the requirement or permission is
taken to be satisfied if the document is posted by that method
or, if that method is not available, by the equivalent, or nearest
equivalent, method provided for the time being by Australia
Post.
(3) Subsections (1) and (2) apply whether the expression
‘deliver’, ‘give’, ‘notify’, ‘send’ or ‘serve’ or another
expression is used.

proceedings

```
