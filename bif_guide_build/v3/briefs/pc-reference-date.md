# Element-page brief — pc-reference-date
**Title:** A valid reference date must have arisen
**Breadcrumb:** Requirements of a payment claim
**Anchor id:** `pc-reference-date`
**Output file:** `bif_guide_build/v3/pages/page_pc-reference-date.html`

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

### `statute/chapter_3/section_067.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 67 — Meaning of reference date
(1) A reference date, for a construction contract, means—
(a) a date stated in, or worked out under, the contract as the
date on which a claim for a progress payment may be
made for construction work carried out, or related goods
and services supplied, under the contract; or
(b) if the contract does not provide for the matter—

(i) the last day of the month in which the construction
work was first carried out, or the related goods and
services were first supplied, under the contract; and
(ii) the last day of each later month.
(2) However, if a construction contract is terminated and the
contract does not provide for, or purports to prevent, a
reference date surviving beyond termination, the final
reference date for the contract is the date the contract is
terminated.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_067.txt`
```
# Annotated BIF Act source — Section 67
# Chapter: CHAPTER 3 – Progress payments
# Section title: Meaning of reference date
# DOCX paragraphs: 1293-1346

[2 Com-BIFSOPA Heading 1] SECTION 67 – Meaning of reference date
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 67    Meaning of reference date 
[1.3 BIFSOPA level 1 (CDI)] A reference date, for a construction contract, means—
[1.4 BIFSOPA level 2 (CDI)] a date stated in, or worked out under, the contract as the date on which a claim for a progress payment may be made for construction work carried out, or related goods and services supplied, under the contract; or
[1.4 BIFSOPA level 2 (CDI)] if the contract does not provide for the matter—
[1.5 BIFSOPA level 3 (CDI)] the last day of the month in which the construction work was first carried out, or the related goods and services were first supplied, under the contract; and
[1.5 BIFSOPA level 3 (CDI)] the last day of each later month.
[1.3 BIFSOPA level 1 (CDI)] However, if a construction contract is terminated and the contract does not provide for, or purports to prevent, a reference date surviving beyond termination, the final reference date for the contract is the date the contract is terminated.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 67.1    Introduction of section 67
[2.4 Com-BIFSOPA CDI Normal Body Text] The Act has introduced a new provision which extends the definition of a ‘reference date’. The definition largely mirrors the previous definition under Sch 2 of BCIPA; however, the Act:
[2.4 Com-BIFSOPA CDI Normal Body Text] omits the words “undertaken to be carried out” from the previous definition in BCIPA; and 
[2.4 Com-BIFSOPA CDI Normal Body Text] modifies “named month” by reducing it to “month”. 
[2.4 Com-BIFSOPA CDI Normal Body Text] The phrase ‘named month’ was previously used in the paragraph (b) definition of ‘reference date’ under BCIPA. The modification is likely to be an attempt to simplify the language of the Act. Whilst the reduced wording is yet to be judicially considered, it is unlikely to impact the application of the Courts’ interpretation in the proceeding judgments.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Act inserts subsection 2 to the meaning of ‘reference date’ which creates a statutory reference date upon termination of the contract, being the date upon the contract is terminated. In relation this amendment, the explanatory notes for the Act explained:
[2.5 Com-BIFSOPA Normal Body Quote] Clause 67 is a new provision defining the term ‘reference date’ for the purposes of this chapter. The definition of reference date also provides for when a construction contract is terminated and does not provide for a reference date surviving beyond termination. In such cases, the final reference date for when payment claim may be made is the day the contract is terminated. 
[2.4 Com-BIFSOPA CDI Normal Body Text] See section 70 for further explanation of ‘reference date’ in the context of the right to a progress payment.
[2.2 Com-BIFSOPA Heading 3] 67.2    ‘Reference date’
[2.4 Com-BIFSOPA CDI Normal Body Text] The date stated in, or worked out under, the contract.
[2.4 Com-BIFSOPA CDI Normal Body Text] Under paragraph (a) of the definition of reference date, the ‘reference date’ is ‘a date stated in, or worked out under, the contract’. This passage was considered in State of Queensland v T & M Buckley Pty Ltd [2012] QSC 265 at [17]-[19] per Margaret Wilson J:
[2.5 Com-BIFSOPA Normal Body Quote] By paragraph (a) of the statutory definition, “reference date” means a date “stated in” or “worked out under” the contract. Whether a date is one “stated in” the contract depends on the proper construction of the words used in the contract without reference to extrinsic evidence (with the possible exception of evidence of the matrix of facts).
[2.5 Com-BIFSOPA Normal Body Quote] According to the Oxford English Dictionary Online, one of the meanings of “work out” is –
[2.5 Com-BIFSOPA Normal Body Quote] “To go through a process of calculation or consideration so as to arrive at the solution of (a problem or question), to solve; also, to reckon out, calculate.”
[2.5 Com-BIFSOPA Normal Body Quote] Just as a mathematical problem may be solved or “worked out” by applying values to a given formula, so may a “reference date” be “worked out under the contract” by applying facts to a formula found within the contract. In my view counsel for the first respondent’s submission ignores the true import of the expression “worked out under”. If it were correct, there would arguably be no distinction between a date “stated in” the contract and one “worked out under” it, because in each case the date would be one ascertained by construction of the contract without reference to extrinsic evidence.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd & Anor [2014] QSC 293, Applegarth J agreed that a reference date may be ‘worked out’ by reference to facts or conduct, (at [30]) where his Honour said (in the context of BCIPA):
[2.5 Com-BIFSOPA Normal Body Quote] The term “worked out” is not defined in the Act. The ordinary meaning of “worked out” in this context connotes a process of calculation. Absent some contextual basis to not apply that ordinary meaning, the statutory definition of “reference date” would seem to allow a reference date to be worked out by applying a formula to facts that are capable of being ascertained. For example, it might provide that a claim for a progress payment may be made 14 days after event X occurs. There is no compelling reason why the fact in question should not be something within the power of a party to bring about. For instance, a contract might provide that a claim for a progress payment may be made 60 days after the contractor commences the works and every 60 days thereafter. The first and subsequent reference dates are worked out by reference to the conduct of the contractor after the formation of the contract in commencing work. In such a case the existence of a reference date is conditioned by the conduct of one of the parties. There is nothing absurd about such an outcome, and it might be inconvenient if the parties could not agree such a provision. It would be odd if the parties could not negotiate, subject to limits imposed by s 99, a formula to work out a reference date which applied to the conduct of one of the parties, rather than rely on the default position set by sub-paragraph (b) of the statutory definition. Subject to limits imposed by s 99 and the assumption in s 12 that there will be a reference date under a construction contract, the parties should be free to agree provisions for the timing of claims for progress payments by reference to post-formation events, including events that depend on a party’s conduct.
[2.5 Com-BIFSOPA Normal Body Quote] (footnotes omitted)
[2.4 Com-BIFSOPA CDI Normal Body Text] A reference date may be ‘worked out’ by reference to post-contract conduct, including by reference to post-contract conduct which is an event which is not certain to occur: Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd & Anor [2014] QSC 293 at [31] per Applegarth J:
[2.5 Com-BIFSOPA Normal Body Quote] The Act appears to allow the parties to agree a reference date that is worked out by reference to an event which may not occur, for example, a milestone that is reached when a certain length of pipe is laid. It is hard to see any valid distinction between a contractual milestone and an event which is similarly within the ability of a party to bring about, such as the submission of a certain document.  In such cases, post-formation facts to which the contractual provision for the working out of the reference date applies permit a reference date to be worked out.  There seems no reason in principle, and nothing in the Act, as to why the parties could not provide for reference dates to be worked out according to the post-formation conduct of one of the parties.  A reference date which is worked out according to a post-formation event which is not certain to occur may make a reference date, and therefore the statutory entitlement, conditional.  But there is nothing in the definition of “reference date” or elsewhere in the Act which prohibits a reference date being worked according to an event which is not certain to occur.
[2.4 Com-BIFSOPA CDI Normal Body Text] That a reference date may be ‘worked out’ is subject to the limits imposed by section 99 of BCIPA (equivalent to section 200 of the Act): Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd & Anor [2014] QSC 293, [30], [34] (Applegarth J).
[2.4 Com-BIFSOPA CDI Normal Body Text] The limits imposed by section 99 of BCIPA (equivalent to section 200 of the Act) may invalidate conditions ‘which effectively prevent or inordinately delay a reference date arising’: Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd & Anor [2014] QSC 293 at [34] where his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] Section 99 may operate to invalidate certain contractual conditions which effectively prevent or inordinately delay a reference date arising. For example, it has been said that if a contract provided for yearly reference dates the provision would be so inimical to the statutory entitlement as to be avoided by a provision like s 99. However, the possibility that certain contractual provisions, which effectively prevent the accrual of a reference date, may be invalidated by s 99 of the Act does not require “worked out” to be given the narrow meaning for which the first respondent contends. It simply means that s 99 may have work to do in the context of an Act which grants an entitlement to progress payments and which, in the section defining the entitlement, assumes that there will be a reference date.
[2.5 Com-BIFSOPA Normal Body Quote] (footnotes omitted)
[2.4 Com-BIFSOPA CDI Normal Body Text] A contractual precondition which renders the reference date illusory may be invalidated by section 200 of the Act: Lean Field Developments Pty Ltd v E & I Global Solutions (Aust) Pty Ltd & Anor [2014] QSC 293, [33] (Applegarth J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Regal Consulting Services Pty Ltd v All Seasons Air Pty Ltd [2017] NSWSC 613, McDougall J held that, a deeming clause of a contract which stipulates that a payment claim which is made earlier than the reference date is deemed to have been made on the reference date, does not convert the determined date to some earlier date, and will not render payment claims made before the reference date valid for the purposes of the Act. This proposition was upheld by the New South Wales Court of Appeal in All Seasons Air Pty Ltd v Regal Consulting Services Pty Ltd [2017] NSWCA 289.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Castle Constructions Pty Ltd v N & R Younis Plumbing Pty Ltd [2019] NSWSC 225, Parker J declared an adjudication determination void on the basis that the relevant payment claim was not made from a valid reference date as it was submitted prior to a reference date accruing. In that case, the contract provided that payment claims were to be made “by the 28th day of the month”. At adjudication, the adjudicator interpreted this to mean that a payment claim could be served prior to the 28th day of the month and that any works claimed would be deemed as being made to the reference date of the 28th of the month. Parker J, in setting aside the adjudicator’s determination, followed the decision of the NSW Court of Appeal in All Seasons Air Pty Ltd v Regal Consulting Services Pty Ltd [2017] NSWCA 289 (above), as authority that a clause in a contract cannot deem an early payment claim as valid and that a right to progress payments depends on the progress payment being made “on and from each reference date”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Waco Kwikform Ltd v Complete Access Scaffolding (NSW) Pty Ltd [2020] NSWSC 1702, Justice Stevenson held that for a contractual clause to attract the benefit of the equivalent to section 67(1)(a) of the BIF Act,  the term must be one which expressly nominates the date by which the claim for a progress payment may be made. In that case, the clause in question merely conferred upon the claimant the discretion to “choose if and when to make a claim”. His Honour distinguished this clause from the clause before Ball J in Patrick Stevedores which entitled the claimant to claim payment progressively on the last day of each month for work done to the second last day of the month. As a result, His Honour concluded that the reference date could not be determined under the construction contract.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Whitehorse Box Hill Pty Ltd v Alliance CG Pty Ltd and Anor [2022] VSC 22, Stynes J upheld an adjudicator’s determination of a final payment claim on the basis that the adjudicator had correctly determined that the reference date arose under the construction contract. In that case, the parties had entered into a deed of amendment whereby both parties agreed that an additional reference date would arise 8 days after the execution of the deed for either party to make a claim under the contract. The respondent submitted that the final payment claim was premature on the basis that the deed of amendment did not amend the terms of the contract concerning the making of a final payment claim, only monthly payment claims.

[2.4 Com-BIFSOPA CDI Normal Body Text] In dismissing this respondent’s submission, Stynes J explained that on a literal reading, the term ‘claim’ encompasses monthly progress claims and any final payment claim. Stynes J outlined that had the parties intended to exclude an entitlement to a final payment claim, they could have done so expressly by, for instance, using the words ‘monthly payment claim’ instead of ‘claim’. The fact that the parties did not, having regard to the context and purpose of the amending deed, strongly supported the objective intention that the reference date could apply to a final payment claim.
[2.3 Com-BIFSOPA Heading 4] In Argyle Building Services Pty Ltd v Dalanex Pty Ltd (No 2) [2022] VSC 452, Delany J held that if “surgical intervention” (to change the wording/language of the contract) by an adjudicator is required in order to determine the reference date under the contract, then there is clearly no “express provision” for a reference date under the contract and the default position in the Act will then apply (i.e., the last day of the named month).
[2.4 Com-BIFSOPA CDI Normal Body Text] ‘Named month’
[2.4 Com-BIFSOPA CDI Normal Body Text] In the context of the NSW Act, in Veer Build Pty Ltd v TCA Electrical and Communication Pty Ltd [2015] NSWSC 864, it was held by Darke J that the phrase ‘named month’ is to be given the statutory meaning as under the Interpretation Act 1987 (NSW), section 21, to mean ‘January, February, March, April, May, June, July, August, September, October, November or December’. This view has been subsequently approved: Broadview Windows Pty Ltd v Architectural Project Specialists Pty Ltd [2015] NSWSC 955, [34] (McDougall J).
[2.4 Com-BIFSOPA CDI Normal Body Text] Whether construction work was carried out in a named month
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been held in the context of the NSW Act that it is necessary that some construction work be undertaken in a named month for a reference date to arise of the last day of that month: Veer Build Pty Ltd v TCA Electrical and Communication Pty Ltd [2015] NSWSC 864, [45] (Darke J), citing Grid Projects NSW Pty Ltd v Proyalbi Organic Set Plaster Pty Ltd [2012] NSWSC 1571, [23]-[39].
[2.4 Com-BIFSOPA CDI Normal Body Text] In Dickson Developments Precinct 5 Pty Ltd v Core Building Group Pty Ltd [2024] FCA 86, Jackman J held that an adjudication determination was void because the adjudicator found that there was a valid reference date where the only available reference dated had been used for the previous payment claim. In this case, Jackman J found that s15(5) of the Building and Construction Industry (Security of Payment) Act 2009 (ACT) (Queensland equivalent BIFSOPA section 75(3)) precluded Core from making the claim in the payment claim. Core could not rely upon the default date contemplated in s 10(3)(b) of the Act. Jackman J referenced Dualcorp Pty Ltd v Remo Constructions Pty Ltd [2009] NSWCA 69 at [65], cited with approval in Southern Han Breakfast Point Pty Ltd (in liq) v Lewence Construction Pty Ltd [2016] HCA 52, to emphasise that the effect of s 15(5) of the Act “is that a document purporting to be a payment claim that is in respect of the same reference date as a previous claim is not a payment claim under the Act and does not attract the recovery regime under the Act”.
[2.3 Com-BIFSOPA Heading 4] Use of inference in determining the reference date
[2.4 Com-BIFSOPA CDI Normal Body Text] In Burrell v JGE Machinery [2014] NSWSC 32, in a proceeding for an interlocutory injunction to prevent the enforcement of an adjudication determination, McDougall J held there to be a serious question to be tried as to whether the adjudication proceeded without jurisdiction in determining the reference date in reliance on ‘probability’ and an inference from business practice. 
[2.3 Com-BIFSOPA Heading 4] Post-contract conduct in varying the reference date
[2.4 Com-BIFSOPA CDI Normal Body Text] In T & T Building Pty Ltd v GMW Group Pty Ltd [2010] QSC 211, Martin J held that parties had agreed to vary the terms of the contract regarding the times on which the claimant may submit progress claims. Martin J drew this implication on the basis of an implied term through the test in BP Refinery (Westernport) Pty Ltd v Shire of Hastings [1977] HCA 40; (1977) 180 CLR 266.
[2.4 Com-BIFSOPA CDI Normal Body Text] Payment claims post termination of contract.
[2.4 Com-BIFSOPA CDI Normal Body Text] In BCFK Holdings Pty Ltd v Rork Projects Pty Ltd [2022] NSWSC 1706, Stevenson J confirmed that the correct construction of section 13(1C) of the NSW SOP Act, (which contains wording similar to section 67(1)(2) BIFSOPA), is that once a contract has been terminated, only one payment claim can be made. The Court’s reasoning was largely influenced by the NSW Parliament’s Second Reading Speech. The minister stated that the “new subsection 13(1)(c) aims to create an entitlement to make a payment claim for a progress payment in circumstances where a contract had been terminated. The reform closes a loophole identified in the High Court’s decision in Southern Han Breakfast Point Pty Ltd (in liq) v Lewence Construction Pty Ltd (2016) 260 CLR 340; 91 ALJR 233; [2016] HCA 52 ”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Taylor Construction Group Pty Ltd v Adcon Structural Group Pty Ltd [2023] NSWSC 723, Rees J granted a permanent injunction to restrain the subcontractor from making adjudication applications in respect of its second payment claims after termination. Section 13(1C) of the NSW SOP Act (similar to s 67(2) BIFSOPA) only permits a single payment claim to be served following termination of a construction contract. Rees J confirmed that section 13(1C) applies strictly “on its terms” without discretion unless the previous payment claim made after termination is withdrawn.

```
