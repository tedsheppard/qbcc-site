# Element-page brief — ar-form-content
**Title:** Form and content
**Breadcrumb:** Requirements of an adjudication response
**Anchor id:** `ar-form-content`
**Output file:** `bif_guide_build/v3/pages/page_ar-form-content.html`

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

### `statute/chapter_3/section_082.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 82 — Adjudication response
(1) After being given notice of an adjudicator’s acceptance of an
adjudication application under section 81, the respondent may

give the adjudicator a response to the adjudication application
(the adjudication response).
(2) However, the respondent must not give an adjudication
response if the respondent failed to give the claimant a
payment schedule as required under section76.
(3) The adjudication response—
(a) must be in writing; and
(b) must identify the adjudication application to which it
relates; and
(c) may include the submissions relevant to the response the
respondent chooses to include.
(4) However, the adjudication response must not include any
reasons (new reasons) for withholding payment that were not
included in the payment schedule when given to the claimant.
(5) The adjudicator may require the respondent to resubmit the
adjudication response without the new reasons.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_082.txt`
```
# Annotated BIF Act source — Section 82
# Chapter: CHAPTER 3 – Progress payments
# Section title: Adjudication response
# DOCX paragraphs: 2494-2568

[2 Com-BIFSOPA Heading 1] SECTION 82 – Adjudication response 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 82    Adjudication Response
[1.3 BIFSOPA level 1 (CDI)] After being given notice of an adjudicator’s acceptance of an adjudication application under section 81, the respondent may give the adjudicator a response to the adjudication application (the adjudication response). 
[1.3 BIFSOPA level 1 (CDI)] However, the respondent must not give an adjudication response if the respondent failed to give the claimant a payment schedule as required under section 76. 
[1.3 BIFSOPA level 1 (CDI)] The adjudication response— 
[1.4 BIFSOPA level 2 (CDI)] must be in writing; and
[1.4 BIFSOPA level 2 (CDI)] must identify the adjudication application to which it relates; and 
[1.4 BIFSOPA level 2 (CDI)] may include the submissions relevant to the response the respondent chooses to include.
[1.3 BIFSOPA level 1 (CDI)] However, the adjudication response must not include any reasons (new reasons) for withholding payment that were not included in the payment schedule when given to the claimant. 
[1.3 BIFSOPA level 1 (CDI)] The adjudicator may require the respondent to resubmit the adjudication response without the new reasons.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 82.1    Section 82(3)(c) – ‘May include the submissions relevant to the response’
[2.3 Com-BIFSOPA Heading 4] Interaction between the payment schedule and adjudication response: reasons
[2.4 Com-BIFSOPA CDI Normal Body Text] The submissions contained in an adjudication response are not limited ‘word for word’ to the payment schedule: see TQM v Dasein [2004] NSWSC 1216 at [30] where McDougall J said in the context of the NSW Act:
[2.5 Com-BIFSOPA Normal Body Quote] It was submitted for Dasein that Mr Davenport had in any event considered the payment schedule provided by TQM, so that (because of s 20(2B) of the Act) he had in substance considered its case. I do not accept that submission. It is correct to that, by s 20(2B), a respondent cannot include in its adjudication response any reason for withholding payment that has not been included in its payment schedule. However, it would be open to a respondent to argue, either in greater detail or with perhaps more supporting evidence, reasons that had been advanced. If it were intended that an adjudication response should do no more than mirror, word for word, a payment schedule, then there is little point to the ability under s 20(1) for a respondent to lodge an adjudication response. In that context, it may be noted that an adjudication response by s 20(2)(c) "may contain such submissions relevant to the response as the respondent chooses to include". That must be qualified by subs (2B) so that the submission is not only relevant to the response but also in amplification of reasons earlier given; but it is a clear indication that the adjudication response may do more than parrot the terms of the payment schedule.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to the further commentary in relation to section 82(4) below.
[2.3 Com-BIFSOPA Heading 4] Interaction between the payment schedule and adjudication response: documents
[2.4 Com-BIFSOPA CDI Normal Body Text] In Syntech Resources Pty Ltd v Peter Campbell Earthmoving (Aust) Pty Ltd & Ors [2011] QSC 293, Daubney J said that:
[2.5 Com-BIFSOPA Normal Body Quote] The word “submissions” in s 26(2)(d) is not to be construed narrowly; indeed, the words of the section show specifically that the submissions may include relevant documentation in support.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Trysams Pty Ltd v Club Constructions (NSW) Pty Ltd [2007] NSWSC 941, an expert report was included in the adjudication response that was not contained in the payment schedule. In the circumstances, Hammerschlag J held that the report was either part of a submission or relevant documentation in support of a submission “duly made” by the plaintiff in support of the schedule (at [49]). His Honour considered that the expert report included in the response was ‘foreshadowed’ in the payment schedule, and that ‘[t]he report did not represent any shift from that position. It was provided in support of it’ (at [56]). In holding that the adjudicator erred in determining that the report was not part of a submission ‘duly made’, his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] Despite the interim nature of an adjudication, natural justice nevertheless clearly required the adjudicator to consider the report unless (even if erroneously) he determined that it was, or was part of, a submission not duly made. In this case, he made no such determination. This amounted to a substantial failure to afford natural justice which worked practical injustice on the plaintiff and rendered the whole adjudication void.
[2.5 Com-BIFSOPA Normal Body Quote] Section 22(2)(d) required the adjudicator, in the circumstances, to consider the report because it was part of submissions duly made. In so far as a failure to do so was jurisdictional error in the sense that a legislative requirement essential to the existence of a determination was not met, he made such an error.
[2.4 Com-BIFSOPA CDI Normal Body Text] Similarly, in Wiggins Island Coal Export Terminal Pty Ltd v Monadelphous Engineering Pty Ltd & Ors [2015] QSC 307, the respondent had included in its adjudication response an expert report (on the claimed extensions of time) that was not contained in the payment schedule. In issue was whether this report went beyond the reasons contained in the payment schedule. Philip McMurdo J was unable to find that the export report did not raise a reason for withholding payment not included in the payment schedule: see at [65]. 
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Syntech Resources Pty Ltd v Peter Campbell Earthmoving (Aust) Pty Ltd & Ors [2011] QSC 293, where Daubney J referred to Trysams and held there was not new reasons for withholding payment, but ‘explanatory’ reasons only (at [25]).
[2.4 Com-BIFSOPA CDI Normal Body Text] Similarly, in Owners Strata Plan 61172 v Stratabuild Ltd [2011] NSWSC 1000, Macready AJ held that in relation to testing that was undertaken for defective work, the payment schedule was ‘appropriately worded and specific enough to 'sufficiently indicate' the reasons for withholding payment’.  His Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] However, even if the adjudicator was incorrect, had he correctly considered section 20(2B) of the Act at paragraphs 31 to 35 of his reasons? I think the defendant is correct in submitting that one must focus on the substance rather than the form of the adjudicator's statements. The adjudicator expressed his understanding of ss 14(3) and 20(2B) of the Act. The adjudicator's view was that, consistent with the policies of fairness and promptness, the full reports should be included with the payment schedule to allow a claimant to properly articulate an adjudication application and prevent a fresh case being brought about in reply after that application was made. The defendant says that the adjudicator did not misconstrue the Act in a way that led to a misconception of his functions.
[2.5 Com-BIFSOPA Normal Body Quote] Unfortunately, the adjudicator did not appreciate how the courts have interpreted the use of the word "indicate" in s 14(3) and the difference between "reasons" and submissions" when used in the legislation.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] A spreadsheet not contained in the payment schedule. Syntech Resources Pty Ltd v Peter Campbell Earthmoving (Aust) Pty Ltd & Ors [2011] QSC 293. Two of the respondent’s reasons for withholding payment in the payment schedule was that there should be a deduction of $142,134.25 for the wet hire payments and that it has taken a review of all previous payments and has determined that the claimant has been overpaid to the amount of $127,618.40. In its adjudication response, the respondent included spreadsheets with calculations supporting these reasons. The adjudicator determined that these spreadsheets could not be relied on by the respondent, referring to natural justice considerations (the adjudicator considered that the respondent must have prepared the spreadsheet or something similar before submitting the payment schedule). One submission by the respondent was that the adjudicator was required to consider the spreadsheets by reason of section 26(2). The Court held that the adjudicator’s failure to consider the spreadsheets was a breach of an essential requirement and jurisdictional error.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer also to the commentary under section 88.5, below.
[2.2 Com-BIFSOPA Heading 3] 82.2    Section 82(4) – ‘must not include any reasons (new reasons) for withholding payment’
[2.4 Com-BIFSOPA CDI Normal Body Text] In John Holland Pty Ltd v Walz Marine Services Pty Ltd & Ors [2011] QSC 39, Margaret Wilson J in considering section 24 of BCIPA (equivalent to section 88 of the Act) said:
[2.5 Com-BIFSOPA Normal Body Quote] Thus, the issues to be decided by the adjudicator are defined in the payment claim and the payment schedule. The respondent must state its reasons for wholly or partially withholding payment in the payment schedule. While the parties may make submissions in support of their respective positions in the adjudication application and the adjudication response respectively, the respondent cannot use the adjudication response as a vehicle to advance reasons for withholding payment which it failed to include in its payment schedule.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Thiess Pty Ltd & John Holland Pty Ltd v Civil Works Australia Pty Ltd [2010] QSC 187; [2011] 2 Qd R 276, Daubney J held that an adjudicator did not commit jurisdictional error in not considering clauses that were raised in the adjudication response, but not in the payment schedule.
[2.4 Com-BIFSOPA CDI Normal Body Text] In GW Enterprises Pty Ltd v Xentex Industries Pty Ltd [2006] QSC 399, Lyons J held that the adjudicator took the correct approach in deciding that ‘he could not take into account any matters raised by the current applicant which they had not raised in the payment schedule’ (at [36]). His Honour held that: 
[2.5 Com-BIFSOPA Normal Body Quote] it is imperative that any reasons for withholding payment must be raised in the payment schedule or they cannot be raised at all. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In State Water Corporation v Civil Team Engineering Pty Ltd [2013] NSWSC 1879 at [63] Sackar J said:
[2.5 Com-BIFSOPA Normal Body Quote] … Once a payment claim and payment schedule have been exchanged, the claimant's decision of whether or not to take the further step of lodging an adjudication application will depend on its assessment of the respondent's reasons for non-payment as disclosed in its payment schedule. There are obvious and powerful policy considerations in ensuring that the respondent discloses all of its reasons for non-payment in its payment schedule. The difficulties a claimant would experience if faced with an adjudication response which raised reasons for non-payment which were not included in the payment schedule, are amplified by the fact that the Act does not grant to the claimant a right to reply to the respondent's adjudication response.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Brambles Australia Ltd v Davenport [2004] NSWSC 120, Einstein J held that it was appropriate for an adjudicator to disregard the respondent’s reliance on a clause in the adjudication response that was not raised in the payment schedule. Here, the respondent relied on a clause 24.1 of the contract where there was no reference to the clause in the payment schedule. Einstein J held:
[2.5 Com-BIFSOPA Normal Body Quote] In my own view, the proper approach, bearing in mind the time constraints built into the scheme, is to be reasonably strict where, as here for example, enforcing the injunction to be found in section 20(2B). Precision and particularity was required in the payment schedule to a degree reasonably sufficient to apprise the parties of the real issues in the dispute. If clause 24.1 was to have been relied upon it had to be expressly identified. Insofar as the adjudication response here went outside the reasons for withholding payment which were included in the payment schedule, it was appropriate for the adjudicator to disregard the adjudication response.
[2.5 Com-BIFSOPA Normal Body Quote] In BuiltCom Construction Pty Ltd v VSD Investments Pty Ltd atf the VSD Investments Trust; VSD Investments Pty Ltd atf The VSD Investments Trust v Builtcom Construction Pty Ltd [2025] NSWSC 250, Peden J upheld the adjudicator’s refusal to consider VSD’s additional site inspection documents that were not referenced in its payment schedule. Under s 20(2B) (equivalent to section 82(4) BIFSOPA) respondents are prohibited from introducing new reason for withholding payment that were not included in the payment schedule. The Court followed EnerMech Pty Ltd v Acciona Infrastructure Projects Australia Pty Ltd (2024) 115 NSWLR 56, where Basten AJA confirmed that claims for damages or restitution are generally outside an adjudicator’s jurisdiction. Peden J agreed that VSD’s defect setoffs were not particularised and amounted to global claims for damages, which the adjudicator had no power to assess.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer also to the further commentary under section 88, below in relation to an adjudicator’s role on issues of ‘real relevance’, as raised by John Holland Pty Ltd v Roads & Traffic Authority of New South Wales & Ors [2007] NSWCA 19 and Thiess Pty Ltd & John Holland Pty Ltd v Civil Works Australia Pty Ltd [2010] QSC 187; [2011] 2 Qd R 276.
[2.3 Com-BIFSOPA Heading 4] Distinction between a reason as to quantum and an ‘other’ reason for withholding payment
[2.4 Com-BIFSOPA CDI Normal Body Text] In John Holland Pty Ltd v Roads & Traffic Authority of New South Wales & Ors [2007] NSWCA 19, counsel for the respondent observed that there is a distinction under section 14(3) of the NSW Act (section 69(c) of the Act) between ‘why the scheduled amount is less’ and ‘the respondent’s reasons for withholding payment’. Hodgson JA said, as to this distinction:
[2.5 Com-BIFSOPA Normal Body Quote] In my opinion, this distinction does not justify a narrow view as to what amounts to reasons for withholding payment. If a respondent does not propose to pay any amount included in the payment claim for any reasons said to justify non-payment of that amount, then in my opinion that is withholding payment and the reasons are reasons for withholding payment. It does not matter whether the reasons relate to non-performance of work, bad work, set-offs or cross-claims of any kind, contractual provisions limiting the claimant’s right to payment or statutory provisions limiting the claimant’s right to payment, or indeed any other suggested justification. Any other view would do violence to the language “withholding payment for any reason”, and be contrary to the plain purpose of s.20(2B) to avoid new submissions being introduced late in a process going ahead on a brief and strict timetable. I agree with what Palmer J said on this matter in Mulitiplex Constructions Pty. Limited v. Luikens [2003] NSWSC 1140 at [65]-[68].
[2.4 Com-BIFSOPA CDI Normal Body Text] This passage was followed by Philip McMurdo J in Wiggins Island Coal Export Terminal Pty Ltd v Monadelphous Engineering Pty Ltd & Ors [2015] QSC 307, [52]. His Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] The evident intent of s 24 is to prevent the unfairness to a claimant which could follow from a respondent being allowed to contest its alleged liability for a reason which it had not advanced ahead of the adjudication application. If WICET’s argument was accepted, that beneficial operation of s 24(4) would be confined to one kind of reason for non-payment which would have no logical distinction from another.
[2.5 Com-BIFSOPA Normal Body Quote] In Baguley Build Pty Ltd v Olcon Concrete & Construction Pty Ltd [2025] QSC 126, Copley J held that the adjudicator did not fall into jurisdictional error by concluding that the applicant’s adjudication response included “new reasons” for withholding payment that had not been stated in the payment schedule, thereby not contravening s 82(4) of the BIF Act. The adjudicator found that the response attempted to rely on reasons derived from earlier payment schedules without specifying them in relation to the new consolidated claim, and thus determined that they could not be considered. His Honour affirmed that it is a matter “for the adjudicator to determine” whether reasons are new or properly included, and relied on Civmec Electrical & Instrumental Pty Ltd v Southern Cross Electrical Engineering Ltd [2019] QSC 300, where Mullins J drew a distinction between failing to consider properly made submissions (which may amount to jurisdictional error) and mischaracterising submissions (which is an error within jurisdiction). This approach was also supported by Applegarth J in Niclin Constructions Pty Ltd v Robotic Steel Fab Pty Ltd (2023) 16 QR 336.

[2.3 Com-BIFSOPA Heading 4] Reasons going to the adjudicator’s jurisdiction
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been held that the prohibition on reasons for withholding payment not contained in the payment schedule does not preclude reasons going to the jurisdiction of the adjudicator: see Rail Corporation (NSW) v Nebax Constructions [2012] NSWSC 6 where McDougall J said:
[2.5 Com-BIFSOPA Normal Body Quote] More significantly, it appears to me, s 20(2B) talks of "reasons for withholding payment". That is not what is at issue here. The question that was raised for the adjudicator's consideration was whether he had any jurisdiction to hear the multiple applications that were lodged.
[2.5 Com-BIFSOPA Normal Body Quote] In Olympia Group (NSW) Pty Ltd v Hansen Yuncken Pty Ltd [2011] NSWSC 165, Ball J said at [11], of a similar argument put to him for consideration, that s 20(2B) prevented the respondent:
[2.5 Com-BIFSOPA Normal Body Quote] "from raising in its adjudication response a reason for not making a payment that was not raised in its payment schedule. It did not prevent it from raising grounds on which it was asserted that the adjudicator did not have jurisdiction to make a determination".
[2.5 Com-BIFSOPA Normal Body Quote] I agree. The point could not have been taken in the payment schedule. Thus, the alternative submission referred to at [8] above and the submission referred to at [10] above must fail.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Watkins Contracting Pty Ltd v Hyatt Ground Engineering Pty Ltd [2018] QSC 65, Brown J held that the respondent, Watkins, had not raised ‘new reasons’ in its adjudication response, in circumstances where:
[2.6 Com-BIFSOPA bullet] the payment schedule stated that Watkins proposed to pay ‘nil’, because there was no reference date to support the payment claim, as the contract had been terminated or frustrated; and
[2.6 Com-BIFSOPA bullet] the adjudication response included detailed submissions as to the grounds of termination, including that the contract contained implied terms.
[2.4 Com-BIFSOPA CDI Normal Body Text] Her Honour held that Watkins’ reasons were merely an elaboration of matters already raised and were, therefore, not new. Further, in obiter, Brown J said that even if Watkins’ reasons were new, as they were directed at jurisdiction, the adjudicator was bound to consider them and section 24(4) BCIPA would not apply. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In Bega Valley Shire Council v Kenpass Pty Ltd [2024] NSWSC 399, Nixon J declined to set aside an adjudication determination on the alleged grounds of jurisdictional error. In this case, one of the alleged grounds Bega relied upon was that the adjudicator wrongfully concluded that there were no valid reasons in the Payment Schedule and thus any reasons advanced in the Adjudication Response was “new” under s 20(2B). Nixon J at [49] held that “any error by the Adjudicator in interpreting the Payment Schedule, or in understanding the reasons given in it, is not a reviewable error”. This decision reinforced the authority that whether new reasons are raised is a matter for the adjudicator to determine.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Acciona Agua Australia Pty Ltd v Monadelphous Engineering Pty Ltd [2020] QSC 133, Bond J considered three distinct grounds of alleged jurisdictional error. First, His Honour declared that the adjudicator had fallen into jurisdictional error by considering “new reasons” that should not have been considered under section 88(3)(b) of BIFSOPA. In that case, the adjudicator accepted two bases in the Contractor’s adjudication response to withhold payment arising out of the Subcontractor’s purported termination of the construction contract: first, that the Subcontractor should be prevented from taking advantage of its breach, and second, an alleged breach of an implied contractual term by the Subcontractor entitled the Contractor to a setoff. Without considering the merits of the arguments, Justice Bond found “no mention” of the alleged breaches of the implied contractual term in the payment schedule explaining the legal basis for withholding payment. Hence, the adjudicator’s consideration of “new reasons” went against the policy of BIFSOPA by infringing the prohibition under section 82(4) of BIFSOPA, resulting in part of the determination being declared void with the balance remaining binding on the parties.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Total Lifestyle Windows Pty Ltd v Aniko Constructions Pty Ltd & Anor [2021] QSC 92, Justice Martin declared part of an adjudicator’s determination void because the adjudicator had considered new reasons in the adjudication response that were not included in the payment schedule. In that case, the adjudicator applied a clause of the construction contract that had been relied upon in the respondent’s adjudication response but had not been raised in the payment schedule. In finding that the adjudicator had fallen into jurisdictional error, Martin J held that:
[2.4 Com-BIFSOPA CDI Normal Body Text] section 69(c) of the BIF Act requires a respondent to include all of its reasons for withholding payment in its payment schedule and not just those that are specifically raised or prompted by the payment claim;
[2.4 Com-BIFSOPA CDI Normal Body Text] section 82(3)(c) of the BIF Act must be read subject to section 82(4) which provides that an adjudication response must not include any reasons for withholding payment that were not included in the payment schedule when given to the claimant; and
[2.4 Com-BIFSOPA CDI Normal Body Text] once an adjudicator considers a matter that it is not permitted to pursuant to section 82 of the BIF Act, then section 88(3)(b) has been breached, whether or not the consideration leads to a particular decision.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Total Lifestyle Windows Pty Ltd v Aniko Constructions Pty Ltd (No. 2) [2021] QSC 231, Freeburn J declared parts of a revised adjudication determination void because the adjudicator had, in contravention of section 88(3) of the BIF Act, considered a defence for withholding payment not included in the payment schedule.
[2.4 Com-BIFSOPA CDI Normal Body Text] In that case, the Claimant suspended the works under the contract in response to the Respondent’s non-payment of two payment claims. Unbeknownst to the Claimant, the Respondent was in the process of engaging a new contractor to carry out the remaining works. The adjudicator was required to make a factual assessment as to whether the Respondent had removed works, thereby triggering the right to loss and expense under section 98(3) of the BIF Act. In the adjudication response, the Respondent submitted that it had engaged the new contractor after the suspension period and therefore it had not removed any works under the contract. The adjudicator agreed with the Respondent’s submission and was of the view that the Claimant bore the onus of proving that works under the contract had been removed by the Respondent. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Although the payment schedule included multiple reasons for withholding payment, it did not include the Respondents submission concerning the engagement of the new contractor. Justice Freeburn determined that the adjudicator had considered a new defence that was not included in the payment schedule and therefore had fallen into jurisdictional error.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Castle Constructions Pty Ltd v Napoli Excavations and Civil Pty Ltd [2023] NSWSC 348, the NSW Supreme Court quashed an adjudication determination for jurisdictional error, finding that the adjudicator failed to consider a submission made in the adjudication response. Both parties accepted that the contract was terminated; with the contention being the date termination occurred. The Court held that the adjudicator did not consider any submission the respondent made regarding termination of the contract, because, in the adjudicator’s determination, the adjudicator proceeded on the basis that the works had been suspended and not terminated. The Court held that the respondent’s submission was duly made and accordingly, the failure to consider the respondent’s submissions was a material failure.
[2.2 Com-BIFSOPA Heading 3] 82.3    Section 24(5) BCIPA
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 24(5) provided that where an adjudication application is about a complex payment claim, a respondent may include reasons for withholding payment not included in the payment schedule.
[2.4 Com-BIFSOPA CDI Normal Body Text] This subsection was unique in Queensland under BCIPA, introduced in December 2014 by operation of the Amendment Act. Cases under equivalent legislation in other jurisdictions, and cases under BCIPA in Queensland before 15 December 2014, when the Amendment Act took effect, and after the Act commenced, should be considered in this context.

```
