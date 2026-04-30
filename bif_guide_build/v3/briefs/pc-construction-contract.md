# Element-page brief — pc-construction-contract
**Title:** A construction contract must exist
**Breadcrumb:** Requirements of a payment claim
**Anchor id:** `pc-construction-contract`
**Output file:** `bif_guide_build/v3/pages/page_pc-construction-contract.html`

## Statute scope note
Show only the "construction contract" definition from s 64. Other defined terms in s 64 belong on later pages.

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

### `statute/chapter_3/section_064.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 64 — Definitions for chapter
In this chapter—
adjudicated amount see section88(1).
adjudication application see section 79(1).
adjudication certificate see section91(1).

adjudication response see section 82(1).
adjudicator, in relation to an adjudication application, means
an adjudicator appointed under section81 to decide the
application.
carry out construction work means—
(a) carry out construction work personally; or
(b) directly or indirectly, cause construction work to be
carried out; or
(c) provide advisory, administrative, management or
supervisory services for carrying out construction work.
claimant see section75(1).
complex payment claim means a payment claim for an
amount more than $750,000 or, if a greater amount is
prescribed by regulation, the amount prescribed.
construction contract means a contract, agreement or other
arrangement under which 1 party undertakes to carry out
construction work for, or to supply related goods and services
to, another party.
construction work see section65.
due date, for a progress payment, means the day the progress
payment becomes payable under section73.
payment claim see section 68(1).
payment schedule see section69.
progress payment means a payment to which a person is
entitled under section 70, and includes, without affecting any
entitlement under the section—
(a) the final payment for construction work carried out, or
for related goods and services supplied, under a
construction contract; or
(b) a single or one-off payment for carrying out
construction work, or for supplying related goods and
services, under a construction contract; or
Page 95 Current as at 27 April 2025

(c) a payment that is based on an event or date, known in
the building and construction industry as a ‘milestone
payment’.
reference date see section67.
related goods and services see section 66.
relevant construction contract, for a progress payment or
payment claim, means the construction contract to which the
progress payment, or to which the payment claim, relates.
respondent see section 75(1).
standard payment claim means a payment claim that is not a
complex payment claim.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_064.txt`
```
# Annotated BIF Act source — Section 64
# Chapter: CHAPTER 3 – Progress payments
# Section title: Definitions for chapter
# DOCX paragraphs: 1094-1196

[2 Com-BIFSOPA Heading 1] SECTION 64 – Definitions for chapter
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 64    Definitions for chapter
[1.3 BIFSOPA level 1 (CDI)] In this chapter— 
[1.1 BIFSOPA Body Text] adjudicated amount see section 88(1). 
[1.1 BIFSOPA Body Text] adjudication application see section 79(1). 
[1.1 BIFSOPA Body Text] adjudication certificate see section 91(1). 
[1.1 BIFSOPA Body Text] adjudication response see section 82(1). 
[1.1 BIFSOPA Body Text] adjudicator, in relation to an adjudication application, means an adjudicator appointed under section 81 to decide the application. 
[1.1 BIFSOPA Body Text] carry out construction work means— 
[1.4 BIFSOPA level 2 (CDI)] carry out construction work personally; or 
[1.4 BIFSOPA level 2 (CDI)] directly or indirectly, cause construction work to be carried out; or
[1.4 BIFSOPA level 2 (CDI)] provide advisory, administrative, management or supervisory services for carrying out construction work. 
[1.1 BIFSOPA Body Text] claimant see section 75(1). 
[1.1 BIFSOPA Body Text] complex payment claim means a payment claim for an amount more than $750,000 or, if a greater amount is prescribed by regulation, the amount prescribed.
[1.1 BIFSOPA Body Text] construction contract means a contract, agreement or other arrangement under which 1 party undertakes to carry out construction work for, or to supply related goods and services to, another party. 
[1.1 BIFSOPA Body Text] construction work see section 65.
[1.1 BIFSOPA Body Text] due date, for a progress payment, means the day the progress payment becomes payable under section 73.
[1.1 BIFSOPA Body Text] payment claim see section 68(1). 
[1.1 BIFSOPA Body Text] payment schedule see section 69.
[1.1 BIFSOPA Body Text] progress payment means a payment to which a person is entitled under section 70, and includes, without affecting any entitlement under the section—
[1.4 BIFSOPA level 2 (CDI)] the final payment for construction work carried out, or for related goods and services supplied, under a construction contract; or
[1.4 BIFSOPA level 2 (CDI)] a single or one-off payment for carrying out construction work, or for supplying related goods and services, under a construction contract; or 
[1.4 BIFSOPA level 2 (CDI)] a payment that is based on an event or date, known in the building and construction industry as a ‘milestone payment’. 
[1.1 BIFSOPA Body Text] reference date see section 67.
[1.1 BIFSOPA Body Text] related goods and services see section 66. 
[1.1 BIFSOPA Body Text] relevant construction contract, for a progress payment or payment claim, means the construction contract to which the progress payment, or to which the payment claim, relates.
[1.1 BIFSOPA Body Text] respondent see section 75(1). 
[1.1 BIFSOPA Body Text] standard payment claim means a payment claim that is not a complex payment claim.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 64.1    ‘Construction contract’
[2.3 Com-BIFSOPA Heading 4] The distinction between a ‘contract’, ‘agreement’ and ‘other arrangement’
[2.4 Com-BIFSOPA CDI Normal Body Text] The definition makes a distinction between a ‘contract’, an ‘agreement’ and an ‘other arrangement’. The making of this distinction may suggest that an ‘other arrangement’ is broader than a ‘contract’. This was the view of Nicholas J in Okaroo Pty Ltd v Vos Constructions and Joinery Pty Ltd [2005] NSWSC 45, where his Honour said: 
[2.5 Com-BIFSOPA Normal Body Quote] With regard to the authorities, and to its context in the Act, in my opinion the term “arrangement” in the definition is a wide one, and encompasses transactions or relationships which are not legally enforceable agreements. The distinction in the definition between “a contract” and “other arrangement” is intended by the legislature to be one of substance so that under the Act construction contracts include agreements which are legally enforceable and transactions which are not. Thus in distinguishing between these relationships I understand the legislature intends that “contract” is to be given its common law meaning and that “arrangement” means a transaction or relationship which is not enforceable at law as a contract would be. Accordingly I reject the submission for Okaroo that the term “arrangement” should be understood to mean an agreement which is tantamount to a contract enforceable at law.
[2.4 Com-BIFSOPA CDI Normal Body Text] This approach was approved by Douglas J in Bezzina Developers Pty Ltd v Deemah Stone (Qld) Pty Ltd [2007] QSC 286.
[2.4 Com-BIFSOPA CDI Normal Body Text] In the context of the NSW Act, where a ‘construction contract’ is defined in terms of ‘a contract or other arrangement’, in Machkevitch v Andrew Building Commissions [2012] NSWSC 546, McDougall J, considered the use of the term ‘other arrangement’ in the following manner:
[2.5 Com-BIFSOPA Normal Body Quote] As a matter of language, it seems to be clear that the legislature intended that a "construction contract" could include both a "contract" (as that concept is known to and understood in the law) and some "other arrangements" that would not in law be regarded as contracts.
[2.5 Com-BIFSOPA Normal Body Quote] It seems to me, as a simple matter of reading the legislative words, that the concept of "other arrangement" is something which goes beyond the concept of "contract".
[2.5 Com-BIFSOPA Normal Body Quote] No doubt, the legislature had in mind that, from time to time, work would be done pursuant to arrangements which might not be susceptible to classification as contracts, formal or informal. Clearly, it did not intend that the entitlement to payment should depend on the degree of formality in the arrangements pursuant to which work should be done. In this respect, the legislative intention could be contrasted with the intention underlying s 10 of the Home Building Act 1989 (NSW), under which a builder is not entitled to enforce a contract unless it is licensed, and carries out work pursuant to a written contract.
[2.5 Com-BIFSOPA Normal Body Quote] The word "arrangement" may be thought to be a somewhat strange one in the context of the Act. It its primary meaning, it denotes the ordering or disposition of things (see, for example, the online editions of the Macquarie Dictionary and the Oxford English Dictionary). But the same reference sources suggest that a secondary meaning of "arrangement" denotes measures or preparations, or plans for the accomplishment of some purpose.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Crown Green Square Pty Ltd v Transport for NSW [2021] NSWSC 1557, Henry J declared that the claimant did not have a right to progress payments on the basis that the claimant was not engaged by the respondent under a construction contract or “other arrangement”. The respondent had entered into a construction contract with a related entity of the claimant. The claimant sought judgment of its payment claim following the respondent's failure to provide a payment schedule. The respondent contended that the claimant was not a party to the construction contract and therefore the payment claim was invalid. The claimant submitted that the entitlement arose under an ‘other arrangement’ which under section 4(1) of the NSW SOP Act (the equivalent of section 64 of the BIF Act) constitutes a construction contract.  Justice Henry dismissed the claimant’s submission on the basis that the requirement to carry out the claimed works arose under a construction contract to which the claimant was not a party and the alleged agreement between the claimant and the respondents didn’t constitute an ‘other arrangement’ as it could not be reconciled with the existence of a concluded state of affairs or meeting of minds that gave rise to a valid arrangement. In obiter, Henry J observed that an ‘other arrangement’ does not necessarily require a legally binding obligation but does require something more than a party undertaking to carry out construction work for another. His Honour explained that there “must be a concluded state of affairs between two or more parties involving some element of reciprocity or acceptance of mutual rights and obligations relating to payment or price for the works (which may or may not be legally binding obligations) under which one party undertakes to carry out construction work for another party to the arrangement.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Ventia Australia Pty Ltd v BSA Advanced Property Solutions (Fire) Pty Ltd [2021] NSWSC 1534, Rees J in obiter explained that an ‘arrangement’ should be used where there is a lack of precision about the parties agreement to carry out construction work which is not captured by the express terms of a contract. In that case, the terms of the contract expressly outlined that each ‘work order’ issued by the respondent constituted the formation of a new construction contract. Rees J was of the view that the Court cannot overlook an express contractual regime in assessing whether an ‘arrangement’ arises.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Bettar Holdings Pty Ltd v RWC Brookvale Investments Pty Ltd, Cole DCJ held that a Claimant was not entitled to a progress payment under section 8(1) of the NSW Act (equivalent to section 70 BIFSOPA) as there was no “construction contract” or “other arrangement”. The Court found that while some terms of a potential future contract were agreed, negotiations remained “subject to a comprehensive agreement being arrived at and recorded in a written and executed contract” (which never actualised). Cole DCJ cited BSA Advanced Property Solutions (Fire) Pty Ltd v Ventia Australia Pty Ltd (2022) 108 NSWLR 350, in affirming that an “other arrangement” must involve an undertaking to perform work for consideration, which the Claimant failed to establish. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In Justar Property Group Pty Ltd v Chase Building Group (Canberra) Pty Ltd [2020] ACTSC 231, Mossop J considered whether the claimant and respondent had a valid “construction contract” or “arrangement” for the purposes of determining whether the adjudicator had jurisdiction in making its adjudication determination. In that case, the respondent contended that it never entered into a “construction contract” with the claimant, but rather that the contract was with a different company, Maxon Group Pty Ltd which the respondent was associated with under the ‘Maxon Group’. Mossop J held that the adjudicator was entitled to accept that there was some “other arrangement” between the claimant and respondent, and that there was no error of law which was manifest on the face of the adjudication determination. Mossop J applied the decision in Machkevitch v Andrew Building Constructions [2012] NSWSC 546 and found that when determining whether or not there is an “arrangement” which is sufficient to constitute a construction contract under the SOP Act, the court must look for “a concluded state of affairs, which is bilateral at least, which can amount to an arrangement under which one party to it undertakes to perform construction work for another party to it”.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour agreed with the view of Nicholas J in Okaroo Pty Ltd v Vos Constructions and Joinery Pty Ltd [2005] NSWSC 45 that an ‘arrangement encompasses transactions or relationships which are not legally enforceable’ and that:
[2.5 Com-BIFSOPA Normal Body Quote] the only matter for consideration is whether it is one under which one party undertakes to carry out construction work, or to supply related goods and services, for another party. There is no other requirement or qualification which is expressly or by implication included in the definition which must be satisfied. It may be safely assumed that had the legislature intended any additional requirement or qualification it would have included it in the definition.
[2.4 Com-BIFSOPA CDI Normal Body Text] On the term ‘arrangement’, McDougall J held that:
[2.5 Com-BIFSOPA Normal Body Quote] In my view, what is required is that there be something more than a mere undertaking; or something which can be said to give rise to an engagement, although not a legally enforceable engagement, between two parties; or a state of affairs under which one party undertakes to the other to do something; or an arrangement between parties to like effect.
[2.5 Com-BIFSOPA Normal Body Quote] In those circumstances, the court must look for a concluded state of affairs, which is bilateral at least, which can amount to an arrangement under which one party to it undertakes to perform construction work for another party to it. It is not necessary that the arrangement be legally enforceable; but an "arrangement" which is legally enforceable may be, a priori, a construction contract.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Ratcliffe v Horizon Glass & Aluminium Pty Ltd [2023] NSWSC 196, Rees J set aside an adjudication determination because there was no “construction contract” between the claimant and the respondent. In this case, the claimant had incorrectly named the individual director of the principal, as the respondent to their payment claim, when the construction contract was with the Principal Company, and not the individual director. In arriving at her decision, Rees J referred to Brereton JA’s summary in Mills v Walsh [2022] NSWCA 255 at [73], in how the Court ought to determine the parties to a construction contract:
[2.4 Com-BIFSOPA CDI Normal Body Text] “The parties to a contract are identified according to the objective theory of contract, which involves ascertaining the intention of the parties from their communications and the circumstances in their mutual knowledge, including their evident commercial aims and expectations; their subjective beliefs and intentions are irrelevant, save insofar as they are manifest and shared. However, the post-contractual conduct of the parties may more readily be resorted to for this purpose than for the purpose of construing contractual terms.”
[2.4 Com-BIFSOPA CDI Normal Body Text] Rees J held that on the balance of probabilities, the parties had intended that the principal company, and not the individual director, enter the construction contract with the claimant. Therefore, the determination was set aside as a construction contract did not exist between the claimant and respondent.
[2.4 Com-BIFSOPA CDI Normal Body Text] In GCB Constructions Pty Ltd v SEQ Formwork Pty Ltd & Ors [2023] QSC 71, the Queensland Supreme Court considered what may constitute a construction contract, and particularly the wording in the definition which refers to “other arrangement”, for the purposes of s 64 BIFSOPA. The proceedings arose from the respondent challenging the validity of an adjudicator’s decision, as there was not a valid “construction contract” or “other arrangement” between the parties. The parties had agreed, via a discussion on site, for the claimant to carry out additional works. The adjudicator concluded that the discussion constituted an “other arrangement”. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In reaching his decision, Burns J noted that to constitute an “other arrangement” within the meaning of s 64 BIFSOPA, there must be at least a bilateral concluded state of affairs, and a sufficient degree of mutuality to serve the purposes for which the arrangement is required under the Act, including enough settled detail to enable the work and/or materials to be claimed with precision. His Honour noted the following reasons as to why in these circumstances the discussion was not an “other arrangement”:
[2.6 Com-BIFSOPA bullet] The payment claim was for the supply of material, however, there was nothing in the discussion regarding the supply of, or payment for material; 
[2.6 Com-BIFSOPA bullet] There was no specification of the rates to be charged for labour; and 
[2.6 Com-BIFSOPA bullet] Nothing was agreed as to the time for performance for any of the work, the making of the claim, the time for payment or retention money.
[2.4 Com-BIFSOPA CDI Normal Body Text] Consequently, as there was no construction contract or other arrangement between the parties, Burns J held that the adjudicator’s decision was void.
[2.4 Com-BIFSOPA CDI Normal Body Text] In EnerMech Pty Ltd v Acciona Infrastructure Projects Australia Pty Ltd [2024] NSWCA 162, Basten AJA (with whom Meagher JA and Griffiths AJA agreed) considered the definitions of “construction contract” and “progress payment” in section 4 of the Building and Construction Industry Security of Payment Act 1999 (NSW). With reference to these definitions, his Honour stated (at [16]) that:
[2.5 Com-BIFSOPA Normal Body Quote] “Two points are immediately apparent from these provisions: first, the Security of Payment Act does not purport to limit the amount or nature of a payment to which a party is entitled under a construction contract; secondly, there is a risk in compartmentalising payments according to their character, regardless of the terms of the contract, so as to contend that some fall within the concept of a progress payment for which a claim may be made, and some do not”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Class Electrical Services v Go Electrical [2013] NSWSC 363, McDougall J clarified his earlier comments in Machkevitch v Andrew Building Commissions [2012] NSWSC 546 in relation to the use of the word ‘undertaking’, stating (at [28]-[29]):
[2.5 Com-BIFSOPA Normal Body Quote] The first use of the word "undertaking" seems to me now to be somewhat unfortunate, having regard to the definition of "construction contract". It was intended to pick up undertakings of the kind said to have been given by Mr Machkevitch to the builder, as I explained above. It was not intended to be "an undertaking" in a cognate sense to the verb "undertakes" as it is used in the definition of "construction contract".
[2.5 Com-BIFSOPA Normal Body Quote] Thus, properly understood, I do not think that anything that I said in Machkevitch focused on what is required to satisfy or demonstrate the concept of undertaking to do construction work or supply related goods and services. It was concerned with the existence of a contract or arrangement.
[2.5 Com-BIFSOPA Normal Body Quote] In Bettar Holdings Pty Ltd trading as Hunt Collaborative v RWC Brookvale Investment Pty Ltd as trustee for Brookvale Development Trust [2025] NSWCA 242, McHugh JA (Bell CJ and Kirk JA agreeing) held that an "other arrangement" under section 4 of the NSW Act (equivalent to section 64 of the Act) must be properly pleaded with the material facts constituting the arrangement and the basis on which the party had "undertaken" to carry out work [135] – [136]. It also requires either a legally binding obligation or a "concluded state of affairs" involving "some element of reciprocity or acceptance of mutual rights and obligations relating to payment or price for the works" [139]. 
[2.5 Com-BIFSOPA Normal Body Quote] In coming to this decision, his Honour discussed the conflicting authorities in Lendlease Engineering Pty Ltd v Timecon Pty Ltd and Crown Green Square Pty Ltd v Transport for NSW [140], declining to resolve the conflict but finding that under either test, there was no "other arrangement" on the facts where the pleading only alleged a common law contract or estoppel preventing denial thereof, and there was no allegation that Hunt had "undertaken" to perform work on any basis other than pursuant to a common law contract. His Honour also found that any case based on quantum meruit was foreclosed by the pleading that the contract existed no later than 27 October 2023, as there was no request for work or performance of services giving rise to restitutionary entitlement until after that date [141] – [142].
[2.5 Com-BIFSOPA Normal Body Quote] In Bloc Constructions (NSW) Pty Limited v Alorra Piling (NSW) Pty Ltd [2025] NSWSC 1324, Peden J affirmed that a "construction contract" under the NSW Act does not require a formal, signed agreement. At [51], the Court held that even if a formal contract could not be proven, the objective conduct of the parties – specifically, the Principal requesting the piling work, the Contractor performing the work, and the Principal making payments – was sufficient to establish an "arrangement" between them within the meaning of section 4 of the NSW Act (equivalent to section 64 of the Act). The decision reinforces that the Act’s remedial purpose allows it to operate based on the commercial reality of work performed, potentially even where formal legal enforceability might be questionable, noting that the claimant would likely have a legal entitlement to restitution in any event.
[2.4 Com-BIFSOPA CDI Normal Body Text] Both Okaroo and Machkevitch, as they relate to the definition of ‘arrangement’, were summarised by Stevenson J in IWN No 2 Pty Ltd v Level Orange Pty Ltd [2012] NSWSC 1439, at [25].
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Vis Constructions Ltd & Anor v Cockburn & Anor [2006] QSC 416 where Jones J considered that to establish the existence of a contract, agreement or other arrangement within the meaning of Schedule 2 BCIPA, “[i]t must be established that there was an undertaking by one party…to carry out construction work or supply related goods to another party”.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] References to an earlier contract. Kaycee Trucking Pty Ltd v M and C Rogers Transport Pty Ltd & Ors [2014] QSC 185. There were two relevant contracts between the parties, a 2012 contract and a 2013 contract. The payment claim in question contained reference to both the 2012 and 2013 contracts. It was submitted that the payment claim should be characterised as seeking payment under or relating to two contracts, and invalid on that basis. The payment claim did not claim for any work done under the 2012 contract. The Court considered that the references to 2012 should not be seen as anything other than background, holding that the payment claim was made under the 2013 agreement.
[2.6 Com-BIFSOPA bullet] Concluded state of affairs. In SMLXL Projects Pty Ltd v RIIS Retail A/S [2017] NSWDC 131, Dicker SC DCJ found that an email sent by the claimant to the respondent which set out various services which had been carried out and a further email from the respondent accepting the works, was sufficient to constitute a construction contact under the NSW act as the works were agreed and amounted to “a concluded state of affairs”. 
[2.6 Com-BIFSOPA bullet] The distinction was considered in Lendlease Engineering Pty Ltd v Timecon Pty Ltd [2019] NSWSC 685. Justice Ball held that for a construction contract to constitute an ‘other arrangement’ under section 4 of the NSW Act (equivalent to section 5 of the Act), it must give rise to a legally binding obligation. The legally binding obligation need not be contractual in nature. Lendlease contended that in circumstances where there was no ‘contract or other arrangement’ between the parties, there was no right for Timecon to claim a progress payment under the NSW Act. Justice Ball’s judgment distinguishes previous authority which found that an ‘other arrangement’ need not be a legally enforceable arrangement to satisfy the NSW Act.
[2.3 Com-BIFSOPA Heading 4] ‘Construction contract’    
[2.4 Com-BIFSOPA CDI Normal Body Text] In Capricorn Quarries Pty Ltd v Inline Communication Construction Pty Ltd [2012] QSC 388; [2013] 2 Qd R 1, Jackson J rejected a submission that BCIPA should be given a ‘remedial’ construction. His Honour held that, in construing the definitions of ‘construction contract’, ‘construction work’ and ‘related goods and services’, a ‘natural’ construction should be given. His Honour held:
[2.5 Com-BIFSOPA Normal Body Quote] Having regard to the context as discussed, in my view, a “natural” construction of the relevant definitions of BCIPA in this case is to be preferred to an approach which seeks to extend the operation of BCIPA by a “liberal interpretation” to be engaged in with the purpose of increasing the width of the class of persons who are entitled to the benefit of a payment claim and correspondingly increasing the width of the class of persons who are subject to BCIPA’s restriction and obligations.
[2.5 Com-BIFSOPA Normal Body Quote] The language chosen by Parliament to define “construction contract”, “construction work” and “related goods and services” has no purpose other than to draw the line between who is in and who is out of those classes.  There seems to be little logic in seeking to stretch that language either way.  In saying this, I take a “natural” construction to be that arrived at by the usual process of the application of the common law of statutory interpretation, as affected by statute, but without a presumptive approach.
[2.3 Com-BIFSOPA Heading 4] A ‘mixed’ contract
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been held that a ‘construction contract’ may include work that is both ‘construction work’ and not ‘construction work’: see Brian Leigh Smith & Anor v Coastivity Pty Ltd [2008] NSWSC 313 where McDougall J held (at [35]):
[2.5 Com-BIFSOPA Normal Body Quote] The definition of “construction contract” does not require that all the work undertaken to be performed, or all the goods and services undertaken to be provided, pursuant to the contract should be construction work or related goods and services as defined. It is sufficient if some of them fall within the definition. Thus, it is not to the point that some of the obligations undertaken by Coastivity (for example, in relation to raising finance) do not amount to related services. The question must be whether any of the other obligations undertaken fall within the statutory definition.
[2.4 Com-BIFSOPA CDI Normal Body Text] Similarly, in Thiess Pty Ltd v Warren Brothers Earthmoving Pty Ltd & Anor [2012] QCA 276, Philippides J said (at [56]):
[2.5 Com-BIFSOPA Normal Body Quote] In my view, his Honour was clearly correct in stating at [77] that “a contract or arrangement is a construction contract if it contains an undertaking of the type specified in the definition of construction contract, notwithstanding that it contains other undertakings or imposes other obligations not within the definition”. It also accords with the approach taken in New South Wales in respect of equivalent legislation: see Brian Leigh Smith v Coastivity Pty Ltd [2008] NSWSC 313 at [35]; HM Australia Holdings Pty Ltd v Edelbrand Pty Ltd [2011] NSWSC 604 at [30]. What is required for the purposes of the definition of “construction contract” is that it is one under which a party undertakes to carry out some “construction work”.
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been held that ‘the adjudicator’s determination as to the extent and value of the “construction work” or “related goods and services” the subject of the payment claim does not…concern a matter of jurisdictional fact’: Thiess Pty Ltd v Warren Brothers Earthmoving Pty Ltd & Anor [2012] QCA 276, [103] (Philippides J, Holmes and White JJA agreeing).
[2.3 Com-BIFSOPA Heading 4] Multiple contracts?
[2.4 Com-BIFSOPA CDI Normal Body Text] A payment claim must relate only to one construction contract, and ‘there can only be one adjudication application for any particular payment claim for any particular contract’: Matrix Projects (Qld) Pty Ltd v Luscombe [2013] QSC 4, [17]-[18] (Douglas J); Rail Corporation of NSW v Nebax Constructions [2012] NSWSC 6, [44] (McDougall J).
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] A payment claim relating to more than one ‘construction contract’. Matrix Projects (Qld) Pty Ltd v Luscombe [2013] QSC 4. Construction work was carried out under a period subcontract and for ‘do and charge’ rectification work. The Court rejected a submission that ‘arrangement’ was sufficiently wide to cover both, holding that the payment claim was not made under a single construction contract, being fatal to its validity.
[2.6 Com-BIFSOPA bullet] Trinco (NSW) Pty Ltd v Alpha A Group Pty Ltd [2018] NSWSC 239. In that case, McDougall J declared void an adjudication determination on the basis that the relevant payment claim was not supported by a valid reference date. However, his Honour held that in the instance that his interpretation of the contracts is found to be incorrect, the adjudicator’s determination would be void on the basis that the relevant payment claim related to works done both under the first and second contract.
[2.3 Com-BIFSOPA Heading 4] Side deeds
[2.4 Com-BIFSOPA CDI Normal Body Text] In APN DF2 Project 2 Pty Ltd v Grocon Constructors (Victoria) Pty Ltd [2014] VSC 596, a side deed was entered into after execution of a design and construct contract, varying the design and construct contract. Vickery J held the side deed to be a ‘construction contract’ under the VIC Act. His Honour concluded that:
[2.5 Com-BIFSOPA Normal Body Quote] It is plainly obvious that the concept of “construction contract” as defined in s 4 of the Act, encompasses both an original contract entered into between the parties (in this case the D&C Contract) and any instrument which validly effects a variation to the original contract (in this case the Side Deed). The Act could not properly operate in a commercial context if this was not the case.
[2.5 Com-BIFSOPA Normal Body Quote] In this case the Side Deed in law varied the D & C Contract, including the payment regime in critical respects. Further, the D & C Contract remained in place following entry into the Side Deed, subject to the alterations reflected in the variations. This resulted in the parties making two contracts which co-existed: the D & C Contract as amended by the Side Deed; and the Side Deed itself.
[2.2 Com-BIFSOPA Heading 3] 64.2    ‘Progress payment’
[2.4 Com-BIFSOPA CDI Normal Body Text] Under the equivalent definition of ‘progress payment’ under section 4 of the NSW Act, in Quasar Constructions NSW Pty Ltd v Demtech Pty Ltd [2004] NSWSC 116, Barrett J considered that the definition of ‘progress payment’:
[2.5 Com-BIFSOPA Normal Body Quote] can only have that character if it is “for” work done or, where some element of advance payment has been agreed, “for” work undertaken to be done. The relevant concepts do not extend to damages for breach of contract, including damages for the loss of an opportunity to receive in full a contracted lump sum price. Compensation of that kind does not bear to actual work the relationship upon which the “progress payment” concept is founded.
[2.4 Com-BIFSOPA CDI Normal Body Text] By the definition of ‘progress payment’ and section 71 of the Act, a payment claim under the Act could be a claim “for the whole of the contract price”.
[2.2 Com-BIFSOPA Heading 3] 64.3    ‘Complex payment claim’
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 64 defines ‘complex payment claim’, being a payment claim for an amount more than $750,000 or, if a greater amount is presciribed by regulation, the amount prescribed. It does not expressly state in the BIF act whether this is inclusive or exclusive of GST. 
[2.4 Com-BIFSOPA CDI Normal Body Text] The Explanatory Notes to the Building Industry Fairness (Security of Payment) and Other Legislation Amendment Act 2020 provides that:
[2.5 Com-BIFSOPA Normal Body Quote] Clause 64 Amendment of s 64 (Definitions for chapter)
[2.5 Com-BIFSOPA Normal Body Quote] Clause 64 amends the definition of ‘complex payment claim’ to remove reference to the amount being exclusive of GST. This has been amended to reflect industry practice where it is common for a payment claim to include a claim for GST as well as for the value of the work undertaken. Additionally, the definition of progress payment does not provide for GST to be excluded.
[2.5 Com-BIFSOPA Normal Body Quote] As a result, the value of the payment claim is the total amount stated (i.e. it will include GST where it is claimed and exclude GST where it is not claimed).

```
