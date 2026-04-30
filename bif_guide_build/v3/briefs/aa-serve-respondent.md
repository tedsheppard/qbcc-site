# Element-page brief — aa-serve-respondent
**Title:** Serve a copy on the respondent
**Breadcrumb:** Requirements of an adjudication application
**Anchor id:** `aa-serve-respondent`
**Output file:** `bif_guide_build/v3/pages/page_aa-serve-respondent.html`

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

### `statute/chapter_3/section_079.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 79 — Application for adjudication
(1) A claimant may apply to the registrar for adjudication of a
payment claim (an adjudication application) if—
(a) the claimant is entitled to apply for adjudication under
section 78(2)(b) because of a failure by the respondent
to pay an amount owed to the claimant by the due date
for the payment; or
(b) the amount stated in the payment schedule, given in
response to the payment claim, is less than the amount
stated in the payment claim.
(2) An adjudication application—
(a) must be in the approved form; and
(b) must be made within—
(i) for an application relating to a failure to give a
payment schedule and pay the full amount stated in
the payment claim—30 business days after the
later of the following days—
(A) the day of the due date for the progress
payment to which the claim relates;
(B) the last day the respondent could have given
the payment schedule under section76; or

(ii) for an application relating to a failure to pay the
full amount stated in the payment schedule—20
business days after the due date for the progress
payment to which the claim relates; or
(iii) for an application relating to the amount stated in
the payment schedule being less than the amount
stated in the payment claim—30 business days
after the claimant receives the payment schedule;
and
(c) must identify the payment claim and the payment
schedule, if any, to which it relates; and
(d) must be accompanied by the fee prescribed by
regulation for the application.
(3) The adjudication application may be accompanied by
submissions relevant to the application.
(4) The claimant must give the following documents to the
respondent within 4 business days after making the
adjudication application—
(a) a copy of the adjudication application;
(b) a copy of the submissions, if any, accompanying the
application under subsection(3).
(5) The registrar must, within 4 business days after the
application is received, refer the application to a person
eligible to be an adjudicator under section 80.
(6) In this section—
copy, of an adjudication application, includes a document
containing details of the application given to the claimant by
the registrar for the purpose of the claimant complying with
the claimant’s obligation under subsection(4)(a).

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

### `annotated/section_079.txt`
```
# Annotated BIF Act source — Section 79
# Chapter: CHAPTER 3 – Progress payments
# Section title: Application for adjudication
# DOCX paragraphs: 2356-2464

[2 Com-BIFSOPA Heading 1] SECTION 79 – Application for adjudication 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 79    Application for adjudication 
[1.3 BIFSOPA level 1 (CDI)] A claimant may apply to the registrar for adjudication of a payment claim (an adjudication application) if—
[1.4 BIFSOPA level 2 (CDI)] the claimant is entitled to apply for adjudication under section 78(2)(b) because of a failure by the respondent to pay an amount owed to the claimant by the due date for the payment; or 
[1.4 BIFSOPA level 2 (CDI)] the amount stated in the payment schedule, given in response to the payment claim, is less than the amount stated in the payment claim. 
[1.3 BIFSOPA level 1 (CDI)] An adjudication application—
[1.4 BIFSOPA level 2 (CDI)] must be in the approved form; and 
[1.4 BIFSOPA level 2 (CDI)] must be made within—
[1.5 BIFSOPA level 3 (CDI)] for an application relating to a failure to give a payment schedule and pay the full amount stated in the payment claim—30 business days after the later of the following days— 
[1.6 BIFSOPA level 4 (CDI)] the day of the due date for the progress payment to which the claim relates; 
[1.6 BIFSOPA level 4 (CDI)] the last day the respondent could have given the payment schedule under section 76; or 
[1.5 BIFSOPA level 3 (CDI)] for an application relating to a failure to pay the full amount stated in the payment schedule—20 business days after the due date for the progress payment to which the claim relates; or
[1.5 BIFSOPA level 3 (CDI)] for an application relating to the amount stated in the payment schedule being less than the amount stated in the payment claim—30 business days after the claimant receives the payment schedule; and 
[1.4 BIFSOPA level 2 (CDI)] must identify the payment claim and the payment schedule, if any, to which it relates; and 
[1.4 BIFSOPA level 2 (CDI)] must be accompanied by the fee prescribed by regulation for the application; and
[1.3 BIFSOPA level 1 (CDI)] The adjudication application may be accompanied by submissions relevant to the application.
[1.3 BIFSOPA level 1 (CDI)] The claimant must give the following documents to the respondent within 4 business days after making the adjudication application—
[1.4 BIFSOPA level 2 (CDI)] a copy of the adjudication application;
[1.4 BIFSOPA level 2 (CDI)] a copy of the submissions, if any, accompanying the application under subsection (3).
[1.3 BIFSOPA level 1 (CDI)] The registrar must, within 4 business days after the application is received, refer the application to a person eligible to be an adjudicator under section 80.
[1.3 BIFSOPA level 1 (CDI)] In this section—
[1.4 BIFSOPA level 2 (CDI)] copy, of an adjudication application, includes a document containing details of the application given to the claimant by the registrar for the purpose of the claimant complying with the claimant’s obligation under subsection (4)(a).”
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 79.1    Section 79(2)
[2.4 Com-BIFSOPA CDI Normal Body Text] In Nebmas Pty Ltd v Sub Divide Pty Ltd & Ors [2009] QSC 92, after referring to the decision in Brodyn, McMurdo J said:
[2.5 Com-BIFSOPA Normal Body Quote] As that judgment makes clear, the conclusion in Kell & Rigby, that the requirement for a notice under s 21(2) is mandatory, does not answer the present question, which is whether the adjudication is void where the adjudicator has decided, albeit wrongly, that there had been compliance with that provision. The judgment of Hodgson JA in Brodyn indicates that this requirement for a notice under s 21(2) is not an essential requirement, in the sense that it was an essential precondition of the existence of an adjudicator’s determination. With regard to the purpose of this legislation, there is no reason why this requirement within s 21(2) should be an essential requirement although, for example, the time limit within s 21(3) should not be essential: cf Project Blue Sky Inc v Australian Broadcasting Authority. Of course s 21(2) provides that an adjudication application “can not be made” unless there is such a notice and these precise words are not replicated in s 21(3). Nevertheless, s 21(3) provides that an adjudication application “must be made” within the times there set out. In each case the words are “emphatic”, and in my view there is no basis for distinguishing between the two provisions on the basis of this difference in words.
[2.5 Com-BIFSOPA Normal Body Quote] Accordingly, the fact, as I have found, that there was non-compliance with s 21(2) does not have the result for which the present applicant contends. The declaratory relief which it seeks must be refused. It also follows that the interlocutory injunction restraining the first respondent from filing the adjudicator’s certificate should be discharged.
[2.4 Com-BIFSOPA CDI Normal Body Text] This decision was followed by Fryberg J in De Neefe Signs Pty Ltd v Build1 (Qld) Pty Ltd; Traffic Technologies Traffic Hire Pty Ltd v Build1 (Qld) Pty Ltd [2010] QSC 279.
[2.4 Com-BIFSOPA CDI Normal Body Text] As to the effect of non-compliance, in De Neefe Signs Pty Ltd v Build1 (Qld) Pty Ltd; Traffic Technologies Traffic Hire Pty Ltd v Build1 (Qld) Pty Ltd [2010] QSC 279, Fryberg held that compliance with section 21(2) to be a precondition to an adjudicator’s jurisdiction. On this point, his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] That question must be answered by construing the opening words of the subsection. I acknowledge that they must be construed in the light of the two conditions, particularly the nature and content of those conditions. I acknowledge the force of the view that those conditions seem on their face purely procedural matters aimed at ensuring the provision of natural justice and perhaps reducing the chance that the case will actually go to adjudication. There is however no reason why Parliament should not if it so chooses make fulfilment of conditions of that nature essential to the making of an application. In my judgment, the words “an adjudication application … cannot be made” produce that result. They should be given their natural meaning. If by statute an application cannot be made, then anything purporting to be one cannot be an application. The existence of an application is a basic and essential requirement of an adjudicator’s determination.
[2.5 Com-BIFSOPA Normal Body Quote] That is the same conclusion as was reached in Kell & Rigby Pty Ltd v Guardian International Properties Pty Ltd.
[2.5 Com-BIFSOPA Normal Body Quote] The effect of the contrary conclusion is the re-wording of s 21(2) to read “unless the adjudicator thinks” in place of “unless” (and it consequentially requires a change of tense in the two conditions). While the subject matter of BCIPA can hardly be compared with the liberty of the subject, there is powerful (albeit dissenting) authority against adopting that technique of statutory interpretation.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Parkview Constructions Pty Limited v Total Lifestyle Windows Pty Ltd t/a Total Concept Group [2017] NSWSC 194, Hammerschlag J considered the requirements of service of an adjudication application under section 17 of the NSW Act, the equivalent of section 21 of the Act. Specifically, his Honour found it is plain that what is served on the respondent must itself be in writing. In this case, it was found service via a USB stick did not equate to service of writing stored on it within the meaning of the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] In National Management Group Pty Ltd v Biriel Industries Pty Ltd trading as Master Steel & Ors [2019] QSC 219, Wilson J held that an adjudication application made by the claimant satisfied section 79(2)(c) of the BIF Act despite a number of errors, because “the adjudication application document refers to the project name, the reference date and the amount due.”
[2.2 Com-BIFSOPA Heading 3] 79.1A    Section 79(2)(a) – ‘the approved form’
[2.4 Com-BIFSOPA CDI Normal Body Text] In Iris Broadbeach Business Pty Ltd v Descon Group Australia Pty Ltd & Anor [2023] QSC 290, Williams J held that the form the claimant gave the respondent as part of the adjudication application was not the ‘approved form’ under s 79(2)(a). The Claimant had provided the ‘QBCC PDF Form’, which is automatically generated when the ‘Electronic Form’ is submitted on the QBCC website.
[2.4 Com-BIFSOPA CDI Normal Body Text] Williams J explained that the ‘approved form’ is the form approved for the purpose of the Claimant applying to the registrar for adjudication. It is the adjudication application document itself which the claimant lodges, either electronically or in hardcopy, with the QBCC to commence the process which is required to be in the approved form and given to the Respondent. Accordingly, the QBCC PDF Form, which is provided as a receipt of the adjudication application being made is not ‘the approved form’.
[2.4 Com-BIFSOPA CDI Normal Body Text] Note:     (i) the QBCC Registry has since made changes to the way in which it processes adjudication applications;
[2.4 Com-BIFSOPA CDI Normal Body Text] (ii) on 6 June 2024, the the Residential Tenancies and Rooming Accommodation and Other Legislation Amendment Act 2024 was passed by the Queensland Parliament, containing significant amendments to the BIF Act. The Explanatory Note to the Third Reading explicitly provides that these amendments are in response to the Queensland Supreme Court’s decision in Iris Broadbeach Business Pty Ltd v Descon Group Australia Pty Ltd & Anor [2023] QSC 290.
[2.4 Com-BIFSOPA CDI Normal Body Text] Those amendments to the BIF Act in June 2024 have rendered this legal principle from Iris Broachbeach (i.e. the approved form issue) nugatory.
[2.2 Com-BIFSOPA Heading 3] 79.2    Section 79(2)(b) – ‘must be made within’
[2.4 Com-BIFSOPA CDI Normal Body Text] In Gisley Investments Pty Ltd v Williams & Anor [2010] QSC 178, Douglas J considered the question of whether an adjudication application made out of time invalidated the adjudication decision. His Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] It is also possible, however, to approach the question as an exercise in statutory construction by asking whether the failure to seek the adjudication earlier, because the email should have been regarded as a payment schedule, makes the later application, premised on the absence of a valid payment schedule, invalid. To use the language in Project Blue Sky Inc v ABC was it a purpose of the legislation that an act done in breach of s 21(3)(c)(i) should be invalid?
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] Here, although the adjudication decision should have been made pursuant to an earlier notice, Mr Williams’ failure to give such a notice seems to have been caused by the imprecision of Gisley Investments’ own email in failing to identify itself as a payment schedule explicitly on its face. The consequence of the delay was that Gisley Investments had a further opportunity to provide a payment schedule and a later notification of the adjudication application that it decided to ignore. In either case, the real objects of the Act of providing notice of the claim and an opportunity to respond by provision of a schedule and by appearing before the adjudicator have been met and there is no good reason why the adjudicator’s decision should be treated as a nullity.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Adjudication application made out of time. Gisley Investments Pty Ltd v Williams & Anor [2010] QSC 178. The claimant was required to make an adjudication application by 3 December 2009. Due to the claimant not considering a ‘relatively informal’ email to be a payment schedule, an adjudication application was not made until 1 February 2010. Daubney J upheld the validity of the adjudication decision.
[2.2 Com-BIFSOPA Heading 3] 79.3    Section 79(2)(b)(iii) – ‘receives the payment schedule’
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 79(2)(b)(iii) is drafted in terms of when the ‘claimant receives the payment schedule’. There is a distinction between ‘service’ and ‘receipt’ under the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to section 102 for further commentary on this distinction, and the meaning of ‘receipt’ under the Act.
[2.2 Com-BIFSOPA Heading 3] 79.4    Section 79(2)(c) – must identify the payment claim and the payment schedule, if any, to which it relates
[2.4 Com-BIFSOPA CDI Normal Body Text] In Kangaroo Point Developments MP Property Pty Ltd v RHG Construction Fitout and Maintenance Pty Ltd [2021] QSC 30, Dalton J held that an adjudication application did not satisfy section 79(2)(c) of the BIF Act on the basis that the claimant did not identify the payment schedule that was issued by the respondent, and instead it identified another document. Justice Dalton held that the word “must” in each of section 79(2)(a) and 79(2)(c) of the BIF Act denotes a mandatory requirement of the Act whereas, in contrast, the word “may” in section 79(2)(e) of the BIF Act is used when the applicant for adjudication in fact has a choice as to whether or not to provide submissions.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Kuatro Build Pty Ltd v Elite Formwork Group Pty Ltd [2025] NSWSC 372 Hmelnitsky J held that Elite’s adjudication application was validly made under s 17 of the SOP Act (equivalent to section 79 BIFSOPA). Elite served its payment claim on 28 November 2024 and lodged its adjudication application on 3 January 2025, which fell within the statutory timeframe. The Court did not identify any issue with compliance under s 17, and no jurisdictional objection was raised in respect of the lodgement or timing of the adjudication application.

[2.2 Com-BIFSOPA Heading 3] 79.5    Section 79(2)(e) – ‘submissions’
[2.3 Com-BIFSOPA Heading 4] ‘May include the submissions’
[2.4 Com-BIFSOPA CDI Normal Body Text] The meaning of ‘submission’ was considered by McDougall J in Austruc Constructions Ltd v ACA Developments Pty Ltd; ACA Developments Pty Ltd v Sarlos [2004] NSWSC 131. The plaintiff (a respondent under the NSW Act) submitted that there is a distinction between ‘submission’ and ‘evidence’, with the former being the theory or reasoning propounded. This distinction was rejected by McDougall J. Relevantly, his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] I do not think that the word “submissions”, in either s 17(3) or s 22(2), should be limited as Mr Corsaro submitted. Firstly, I do not think that the ordinary English meaning of the word “submission” is limited in the way that ACA contends. It is certainly correct to say that one of the definitions given by the Shorter Oxford English Dictionary is “[t]he theory of a case put forward by an advocate”. However, the same dictionary also defines the word to mean “the act of submitting a matter to a person for decision or consideration”; and it gives other definitions as well. Further, the Macquarie Dictionary defines “submission” as including “the act of submitting ... the condition of having submitted … submissive conduct or attitude … that which is submitted … law an agreement to abide by a decision or obey an authority in some matter referred to arbitration … “.
[2.5 Com-BIFSOPA Normal Body Quote] It is apparent from the definitions given by both dictionaries that the “ordinary English meaning” for which ACA contends is a specific application of the more general meaning, to the effect of “that which is submitted for decision or consideration”.
[2.5 Com-BIFSOPA Normal Body Quote] Secondly, I think that the better view of s 17(3) is that it does not limit the matters that may be put to an adjudicator in an adjudication application. In this context, I think that the contrast between the mandatory language of paragraphs (a) to (g), and the discretionary language of paragraph (h), is clear.
[2.5 Com-BIFSOPA Normal Body Quote] Thirdly, and in any event, I think that it is s 22(2) that governs the situation. It will be recalled that that subsection specifies the only matters that an adjudicator may take into account. Those matters include, through paragraph (c), the relevant payment claim “together with all submissions (including relevant documentation) …”. Not only do the parenthesised words show that the legislature had in mind that the word ”submissions” was not to be construed narrowly, as ACA contends; they show specifically that the submissions may include relevant documentation in support.
[2.5 Com-BIFSOPA Normal Body Quote] It follows, I think, that if a claimant chooses to include, as part of the relevant documentation supporting its payment claim, a statutory declaration whereby relevant matters are, in effect, verified, then that statutory declaration will form part of the material to be considered by the adjudicator. Equally, if a claimant includes such a statutory declaration in its adjudication application, that is part of the “submission” to be considered.
[2.4 Com-BIFSOPA CDI Normal Body Text] This approach was followed in Queensland in Syntech Resources Pty Ltd v Peter Campbell Earthmoving (Aust) Pty Ltd [2011] QSC 293, [32] (Daubney J).
[2.3 Com-BIFSOPA Heading 4] ‘Relevant to the application’
[2.4 Com-BIFSOPA CDI Normal Body Text] In John Holland v Cardno MBK (NSW) Pty Ltd [2004] NSWSC 258, Einstein J considered the NSW equivalent of section 21(3) and the introduction of new reasons in an adjudication application, holding that:
[2.5 Com-BIFSOPA Normal Body Quote] The adjudication application will relate to a particular payment claim and payment schedule [section 17 (3) (f)]. The central significance of the entitlement of the applicant to include submissions as part of its adjudication application is because those submissions have to be supportive of the payment claim. Those submissions cannot constitute a payment claim or part of it. The central significance of the entitlement of the respondent to include submissions as part of its adjudication response is because those submissions have to be supportive of the payment schedule. Those submissions cannot constitute a payment schedule or part of it.
[2.4 Com-BIFSOPA CDI Normal Body Text] The inclusion of submissions by a claimant is directory rather than mandatory: Abacus Funds Management Ltd v Davenport [2003] NSWSC 1027, [19] (McDougall J).
[2.2 Com-BIFSOPA Heading 3] 79.6    New reasons in an adjudication application
[2.4 Com-BIFSOPA CDI Normal Body Text] Whether there is an equivalent prohibition to section 82(4) of the Act prohibiting new reasons from appearing in the adjudication response which were not raised in the payment schedule applying in the context of an adjudication application was considered in John Holland v Cardno MBK (NSW) Pty Ltd [2004] NSWSC 258 and Minister for Commerce v Contrax Plumbing (NSW) Pty Ltd [2004] NSWSC 823.
[2.4 Com-BIFSOPA CDI Normal Body Text] In John Holland v Cardno MBK (NSW) Pty Ltd [2004] NSWSC 258, Einstein J said in the context of section 20(2B) of the NSW Act:
[2.5 Com-BIFSOPA Normal Body Quote] The primary touchstone it seems to me, is section 20 (2B). Whilst a claimant which provides the most minimal amount of information in its payment claim may even so, be seen to technically comply with section 13, such a claimant will expose itself to an abortive adjudication determination if it be that:
[2.5 Com-BIFSOPA Normal Body Quote] the respondent is simply unable to discern from the content of the payment claim, sufficient detail of that claim to be in a position to meaningfully verify or reject the claim: hence not then being in a position to do otherwise than to reject the whole of the claim on the basis of its inability to verify any part of the claim;
[2.5 Com-BIFSOPA Normal Body Quote] the claimant then elects to include the missing detail in the adjudication application with the inexorable consequence that the respondent is barred by section 20 (2B) from dealing with that detail/matter in its adjudication response;
[2.5 Com-BIFSOPA Normal Body Quote] the adjudicator relies in determining the adjudication application upon the detail supportive of the payment claim which first emerged as part of the adjudication application.
[2.5 Com-BIFSOPA Normal Body Quote] For those reasons whilst it is not permissible to construe section 13 as providing that in order to be a valid payment claim, such a claim must do more than satisfy the requirements stipulated for by subsection 2 (a), (b) and (c), the consequence to a claimant which does not include sufficient detail of that claim to be in a position to permit the respondent to meaningfully verify or reject the claim, may indeed be to abort any determination.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour then considered the provision of documents in an adjudication application that were not provided in the payment claim. On this point, his Honour said that:
[2.5 Com-BIFSOPA Normal Body Quote] The deploying for the first time in the adjudication application, of supporting documentation will require careful attention and becomes a matter of degree and detail. However in the main I do not see that a respondent which, by reason of insufficient information supplied with the payment claim, is unable to verify that claim, and says as much in the payment schedule [only later to receive as part of the adjudication application, the supporting documentation which should have been earlier supplied in order to permit a meaningful payment schedule response], will be otherwise than barred by section 20 (2B) from including in its adjudication response reasons for withholding payment arising by reference to the later supporting documentation. It could not be said that those reasons were already included in the payment schedule provided to the claimant. A complaint about inability to verify a claim because of insufficient information is not synonymous with reasons for dealing with a properly supported claim.
[2.4 Com-BIFSOPA CDI Normal Body Text] The view of Einstein J in John Holland v Cardno were revisited in Minister for Commerce v Contrax Plumbing (NSW) Pty Ltd [2004] NSWSC 823 by McDougall J. His Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] In John Holland, it was submitted to Einstein J that considerations of procedural fairness demanded that a restriction similar to that contained in s 20(2B) be read into s 17(3), to the effect that a claimant in an adjudication application is restricted to raising only matters canvassed in its payment claim. While his Honour thought that considerations of "logic", and "consistency" with the situation of respondents, suggested that this submission be accepted (at [4]), his conclusion was that the "accepted principles of statutory construction" would not permit the suggested implication to be made (at [21]). This conclusion was bolstered by the fact that, in contrast to the situation of respondents when preparing payment schedules pursuant to s 14(3), it is not an "essential condition" of s 13 that the claimant include any reasons whatsoever in a payment claim (at [18]).
[2.4 Com-BIFSOPA CDI Normal Body Text] Einstein J dealt with the problem in a different way. He said that when an adjudication application put a claim on a basis that had not been advanced in the payment claim, the adjudicator, as a matter of jurisdiction, could not deal with it; and there would also be denial of natural justice (at [41]). That was because (as his Honour explained at [40]), s 20(2B) would prevent the respondent from including in its adjudication response any reasons relating to the new claim; but it could not deal with a new claim except by doing that which was prevented by s 20(2B). To determine such a new claim upon a basis that the respondent could not answer was, his Honour said, a denial of natural justice.
[2.4 Com-BIFSOPA CDI Normal Body Text] McDougall J interpreted the decision by Einstein J in John Holland v Cardno in the following manner:
[2.5 Com-BIFSOPA Normal Body Quote] What Einstein J said in John Holland was that a claimant that did not provide sufficient details in its payment claim to enable the respondent to verify or reject (ie, assess) the claim could not include the missing details in its adjudication application. That was because, since the respondent was barred by s 20(2B) from replying to those details (ie, of responding in its adjudication response in a way that did deal with the merits of the claim) the result “may indeed be to abort any determination”: at [23]. His Honour said, alternatively, that an adjudicator did not have power to consider materials supplied by a claimant in its adjudication application which went outside the materials provided in the payment: at [24]. Materials would go outside what had already been provided if they fell outside the ambit or scope of that earlier material.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, his Honour rejected that an adjudication application could not address matters raised in the payment schedule. On this point, his Honour said that:
[2.5 Com-BIFSOPA Normal Body Quote] It would be quite extraordinary if the statutory regime, on its proper construction, prevented an applicant for adjudication from dealing with issues raised by the respondent to adjudication in its payment schedule. Such a construction would mean, in effect, that the applicant would be required to anticipate in its payment claim, and deal with at length, every possible argument that the respondent might rely upon. That would have the effect of increasing enormously the complexity and expense of the statutory procedure: something quite at odds with the statutory objects set out in s 3 and reinforced in the Second Reading Speech. It would also mean that, notwithstanding the best attempts of the applicant to foresee and answer all possible arguments, it might be defeated if the ingenuity of the respondent or its lawyers turned up yet further arguments.
[2.5 Com-BIFSOPA Normal Body Quote] I do not believe that the legislature intended such consequences to flow from the scheme that it enacted. Nor do I think that there is anything in what Einstein J said in John Holland that requires me to conclude, notwithstanding the views that I have expressed, that the legislature did intend such bizarre consequences to follow.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer also to the commentary under section 88, below.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Probuild Constructions (Aust) Pty Ltd v Shade Systems Pty Ltd [2016] NSWSC 770, Emmett AJA held that there was no denial of procedural fairness in circumstances where an adjudicator made a determination for an amount greater than the amount claimed in an adjudication application.
[2.4 Com-BIFSOPA CDI Normal Body Text] The amount claimed in the adjudication application was of a lesser amount than the amount claimed in the payment claim due to a number of matters being conceded by the applicant. Emmett AJA held that on a fair reading of the payment claim, payment schedule, the adjudication application and adjudication response, there had been no denial of procedural fairness:
[2.5 Com-BIFSOPA Normal Body Quote] In all of the circumstances, I do not consider that, on a fair reading of the Payment Claim, the Payment Schedule, the Adjudication Application, and the Adjudication Response, Probuild was denied procedural fairness.
[2.2 Com-BIFSOPA Heading 3] 79.7    Section 79(3) – ‘adjudication application must be given to the respondent’
[2.4 Com-BIFSOPA CDI Normal Body Text] In Niclin Constructions Pty Ltd v SHA Premier Constructions Pty Ltd & Anor [2019] QSC 91, Ryan J held that, in order for an adjudication application to be validly served on the respondent in accordance with section 21(5) BCIPA (equivalent to section 79(3) of the Act):
[2.6 Com-BIFSOPA bullet] a copy of the QBCC form 6 (now form S79) must be served on the respondent as part of the adjudication application; and 
[2.6 Com-BIFSOPA bullet] the adjudication application, including the QBCC form 6 (now form S79), must be served on the respondent as soon as possible. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Her Honour reasoned that given the “brutally fast timeframes” imposed by BCIPA, service of the form was significant because it informed the adjudicator when to determine the case in the knowledge that he or she has all the material before them, such as an adjudication response which would follow the application and be due within 10 business days in accordance with BCIPA. This decision is currently being appealed to the Court of Appeal.
[2.4 Com-BIFSOPA CDI Normal Body Text] This decision was subsequently upheld on appeal. In Niclin Constructions Pty Ltd v SHA Premier Constructions Pty Ltd & Anor [2019] QCA 177, the Queensland Court of Appeal upheld the decision of the primary judge that s 21(5) BCIPA (equivalent of 79 of the Qld Act) requires service of an adjudication application upon the respondent “as soon as possible” after the application is lodged with the registrar.  
[2.4 Com-BIFSOPA CDI Normal Body Text] In McCarthy v TKM Builders Pty Ltd & Anor [2020] QSC 301, Martin J determined that an adjudication application was not “given to the respondent” in accordance with section 79(3) of the BIF act on the basis that service of the submissions relevant to the adjudication application was invalid. In that case, the submissions to the adjudication application could only be obtained by opening a Dropbox link that was sent via email to the respondent. The court applied Justice McMurdo’s reasoning in Basetec and held that the respondent did not become aware of the contents of the document merely by being referred to a link to a Dropbox file. His Honour found that it was not enough that it could be shown the respondents solicitors saw the submissions when the respondent forwarded the email, the court required a positive action on the part of the receiver.

[Normal] In Equinox Construction Pty Ltd v Henning & Anor [2021] QSC 223, Justice Ryan declared an adjudication determination void because the Claimant failed to “give” a copy of the adjudication application to the Respondent pursuant to section 79(3) of the BIF Act. The Claimant contended that it had posted the adjudication application in accordance with the sections 39 and 29A of the Acts Interpretation Act 1954 (Qld). The Respondent asserted that it did not receive it. In determining that the service of the adjudication application was invalid, Ryan J held that a Claimant must be in a position to strictly prove service so that there can be quick and efficient resolution of disputes by way of adjudication. The Court found that: 
[1.3 BIFSOPA level 1 (CDI)] evidence of having used a "postage paid" envelope is not sufficient under section 39 or 39A  because the Claimant did not explain how postage paid envelopes were to be used when paying for postage of items contained in them and failed to give evidence about how the postage was calculated/paid for; and 
[1.3 BIFSOPA level 1 (CDI)] evidence of attending a post office on a specific date is not sufficient to establish how the posting was effected. 
[Normal] In Equa Building Services Pty Ltd v KLG Trading Pty Ltd [2021] NSWSC 1674, Stevenson J declared that the claimant had sufficiently served a copy of the adjudication application on the respondent, even though it was not an exact reproduction, on the basis that the differences were trivial and inconsequential. In that case, the claimant had served the adjudication application electronically via an online “lockbox” system and provided a hard copy to the respondent. The respondent argued that the hard copy was different to the electronic copy provided to the adjudicator because: the respondent was unable to access a video provided on a USB; it was missing three documents; contained illegible photographs; and documents were mislabelled.
[1.3 BIFSOPA level 1 (CDI)] Justice Stevenson held that the operation of section 17(5) of the NSW SOP Act (the equivalent of s 79(3) of the BIF Act)  is capable of degrees of non-compliance if it does not prejudice the substantial effect of the objectives of the Act. In assessing the differences, Stevenson J found that the differences were trivial because: the information stored on the USB was available to be accessed and the claimant was unable to access due to software differences in the computer used; the missing documents was a template of the payment claim and an ASIC search of the developer; it was clear that that the photographs were copies of the document uploaded electronically; and the mislabelling was of no consequence in that the documents themselves were contained in both versions. Accordingly, Stevenson J found that it would not be consistent with that legislative object for s 17(5) of the NSW SOP Act (the equivalent of s 79(3) of the BIF Act)  to be read as having the effect that the slightest difference between the adjudication application as submitted to the adjudicator and the copy served on the respondent rendered invalid the purported service of the copy adjudication application, let alone an adjudication determination based on it.
[1.3 BIFSOPA level 1 (CDI)] In Iris Broadbeach Business Pty Ltd v Descon Group Australia Pty Ltd & Anor [2023] QSC 290, Williams J held that an adjudicator did not have jurisdiction because the claimant had not provided ‘a complete copy’ of the adjudication application to the respondent. The Claimant had provided the ‘QBCC PDF Form’, which was automatically generated by the QBCC when the ‘approved form’ is submitted on the QBCC website and that this is not the approved form. 
[1.3 BIFSOPA level 1 (CDI)] Williams J found that the differences between the forms were ‘not trivial’ and therefore were ‘such as to result in the copy of the QBCC PDF form not being a “copy” of the Electronic Form’. It was possible to give a full copy of the form electronically completed, and there was no evidence to suggest doing so was ‘onerous or time-consuming such as to make it inconsistent with the timeframes in the statutory regime’. The Court noted this document could be produced by taking a screen shot of each page of the QBCC’s platform when electronically making an adjudication application.
[1.3 BIFSOPA level 1 (CDI)] Note: The amendments to the BIF Act in June 2024 have rendered this legal principle from Iris Broachbeach (i.e. the approved form issue) nugatory.

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
