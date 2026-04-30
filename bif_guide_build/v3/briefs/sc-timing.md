# Element-page brief — sc-timing
**Title:** Timing of the notice
**Breadcrumb:** Valid subcontractor's charge
**Anchor id:** `sc-timing`
**Output file:** `bif_guide_build/v3/pages/page_sc-timing.html`

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

### `statute/chapter_4/section_122.txt`
```
# Source: BIF Act 2017 — Chapter 4
# Section 122 — Notice of claim
(1) To claim a subcontractor’s charge over money payable to the
contractor under the contract, the subcontractor must give
written notice (a notice of claim) to the person obliged to pay
the money under the contract.
(2) The notice of claim must be made in the approved form and—
(a) state the amount of the claim; and
(b) include details of the work done by the subcontractor,
certified as prescribed by a qualified person; and
(c) include the other information prescribed by regulation.
(3) The amount of the claim must be certified by a qualified
person, as prescribed by regulation.
(4) The notice of claim may be given even if the work is not
completed, or payment of the money relating to the charge is
not yet due.
(5) However, if the work has been completed, the notice of claim
must be given within 3 months after practical completion for
the work.
(6) The claim may relate only to—
(a) money payable to the subcontractor by the date the
notice is given; and
(b) money to become payable to the subcontractor after the
date the notice is given if the money is for work done by
the subcontractor before that date.

(7) To remove any doubt, it is declared that a subcontractor’s
charge on money payable under the contract includes a charge
on a retention amount for the contract.
(8) If the notice of claim relates only to a retention amount for the
contract, the notice—
(a) may be given at any time while work under the contract
is being performed; and
(b) must be given within 3 months after the expiration of the
defects liability period for the contract.
(9) If the notice of claim is not given in compliance with this
section, the notice is of no effect and the subcontractor’s
charge does not attach.
(10) In this section—
qualified person see section147(1).

```

### `statute/chapter_4/section_125.txt`
```
# Source: BIF Act 2017 — Chapter 4
# Section 125 — Withdrawing a notice of claim
(1) A subcontractor may at any time withdraw, wholly or partly, a
notice of claim by giving notice of the withdrawal, in the
approved form, to the person to whom the subcontractor gave
the notice of claim.
(2) If a subcontractor withdraws, wholly or partly, a notice of
claim under subsection(1), the subcontractor must give a
copy of the notice of withdrawal to each of the persons to
whom the subcontractor gave a copy of the notice of claim.

claim

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_122.txt`
```
# Annotated BIF Act source — Section 122
# Chapter: CHAPTER 4 – Subcontractors’ charges
# Section title: Notice of claim
# DOCX paragraphs: 4501-4583

[2 Com-BIFSOPA Heading 1] SECTION 122 – Notice of claim
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 122    Notice of claim 
[1.3 BIFSOPA level 1 (CDI)] To claim a subcontractor’s charge over money payable to the contractor under the contract, the subcontractor must give written notice (a notice of claim) to the person obliged to pay the money under the contract.
[1.3 BIFSOPA level 1 (CDI)] The notice of claim must be made in the approved form and— 
[1.4 BIFSOPA level 2 (CDI)] state the amount of the claim; and 
[1.4 BIFSOPA level 2 (CDI)] include details of the work done by the subcontractor, certified as prescribed by a qualified person; and 
[1.4 BIFSOPA level 2 (CDI)] include the other information prescribed by regulation. 
[1.3 BIFSOPA level 1 (CDI)] The amount of the claim must be certified by a qualified person, as prescribed by regulation.
[1.3 BIFSOPA level 1 (CDI)] The notice of claim may be given even if the work is not completed, or payment of the money relating to the charge is not yet due. 
[1.3 BIFSOPA level 1 (CDI)] However, if the work has been completed, the notice of claim must be given within 3 months after practical completion for the work. 
[1.3 BIFSOPA level 1 (CDI)] The claim may relate only to—
[1.4 BIFSOPA level 2 (CDI)] money payable to the subcontractor by the date the notice is given; and
[1.4 BIFSOPA level 2 (CDI)] money to become payable to the subcontractor after the date the notice is given if the money is for work done by the subcontractor before that date.
[1.3 BIFSOPA level 1 (CDI)] To remove any doubt, it is declared that a subcontractor’s charge on money payable under the contract includes a charge on a retention amount for the contract.
[1.3 BIFSOPA level 1 (CDI)] If the notice of claim relates only to a retention amount for the contract, the notice— 
[1.4 BIFSOPA level 2 (CDI)] may be given at any time while work under the contract is being performed; and
[1.4 BIFSOPA level 2 (CDI)] must be given within 3 months after the expiration of the defects liability period for the contract. 
[1.3 BIFSOPA level 1 (CDI)] If the notice of claim is not given in compliance with this section, the notice is of no effect and the subcontractor’s charge does not attach. 
[1.3 BIFSOPA level 1 (CDI)] In this section—
[1.1 BIFSOPA Body Text] qualified person see section 147(1).
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 122.1    General
[2.4 Com-BIFSOPA CDI Normal Body Text] Reading this section as a whole, a charge may be claimed either on retention money, or secondly on moneys otherwise payable under the contract, or thirdly on retention moneys and moneys otherwise payable under the contract. The only material consequence flowing from the third categorisation is that such notice would have to be given within three months of the completion of the work, and not within the extended period permissible when the charge is sought against retention moneys only: Road Surfaces Group Pty Ltd v Brown [1987] 2 Qd R 792 per Williams J at p 804.
[2.4 Com-BIFSOPA CDI Normal Body Text] The SCA distinguishes between “claim”, “notice of claim of charge” and “charge”. McPherson J has observed that these expressions are generally treated as distinct in the SCA: Ronnor Pty Ltd v D & R Fabrications Pty Ltd [1983] 2 Qd R 455 at 458. See also Re University of Queensland [2003] 2 Qd R 577; [2003] QSC 158.
[2.2 Com-BIFSOPA Heading 3] 122.2    Retention
[2.4 Com-BIFSOPA CDI Normal Body Text] The legislative intent appears to be that if a claim is to be made in respect of retention money only then the word “retention” does not change form. A claimant would be concerned to claim in respect of retention money only, in circumstances where he sought to give a notice more than three months after completion of the work. In such a case a notice of claim of charge could not validly be given, unless it was expressed in regards to the retention of money only. The legislation gives the subcontractor the option of giving a notice of claim of charge in respect of money payable or to become payable under the contract generally, which includes retention money subject to prior claims upon it under the contract; or alternatively a notice of claim of charge in respect to the retention of money only, thus excluding any other moneys payable under the contract from the charge. The fact that the only moneys remaining payable under a contract may be retention moneys, does not mean that in order to charge those moneys the notice of charge needs to be restricted in terms to retention money. Retention money would also be charged under a notice of claim expressed simply in relation to money payable under the contract. To be effective, however, such a notice needs to be given within three months after completion of the work, and not within three months after the expiration of the period of maintenance provided for by the contract: Re Bunnerin Pty Ltd (in liq) [1990] 1 Qd R 525 per Senior Master McLauchlan QC (as he then was) at pp 526 to 7.
[2.2 Com-BIFSOPA Heading 3] 122.3    Paragraph (a)
[2.4 Com-BIFSOPA CDI Normal Body Text] The requirement to give to the contractor a notice of having made the claim is to be viewed as something to be done with as little delay as possible. It is a question of fact in each case, whether such notice has been given with all convenient speed. It should, however, be done with a sense of commercial urgency. The application of all convenient speed to the giving of a notice under s 10(1)(b) (now see s 122(1)BIFSOPA (Qld)) should commonly result in that notice being given to the contractor within a few days of the notice of claim of charge being given to the employer. It is desirable that the notice should state the time when the work was done, although this is not a pre-requisite to a valid charge. If an issue is raised as to whether the charge is given within the requisite time after completion, the time of completion will be decided as a question of fact: Re Austco Pty Ltd [1985] 2 Qd R 1 per Thomas J (as His Honour then was) at p 4.
[2.4 Com-BIFSOPA CDI Normal Body Text] On appeal in Ed Ahern Plumbing (Gold Coast) Pty Ltd v JM Kelly (Project Builders) Pty Ltd [2008] 2 Qd R 123; [2007] QCA 452, Holmes JA held that a charge does attach to a notice where the notice of claim of charge is given to the contractor under s 10(1)(b) (now see s 122(1)BIFSOPA (Qld)) before the claim of charge has been made to the employer under s 10(1)(a) (now see s 122(1) – (3)BIFSOPA (Qld)). It was confirmed that once a subcontractor has taken the necessary steps to transmit the claim to the employer, the subcontractor has “made” the claim, and only then should the subcontractor give notice to the contractor of having made the claim, which they should do as soon as possible (at [46]).
[2.4 Com-BIFSOPA CDI Normal Body Text] In McConnell Dowell Constructors (Aust) Pty Ltd v Heavy Plant Leasing Pty Ltd [2013] QSC 223, the Supreme Court found that the notice “Monies payable to claimant pursuant to subcontract for the earthworks, concrete and pond liner works on the project” was insufficient information to allow the identification of the work subject of the claim and failed to comply with the s 10 requirements of the SCA.
[2.2 Com-BIFSOPA Heading 3] 122.4    Test for compliance
[2.4 Com-BIFSOPA CDI Normal Body Text] The contents of the notice given must be sufficient in substance to indicate the moneys sought to be charged: Re Austco Pty Ltd [1985] 2 Qd R 1 per Thomas J (as he then was), at p 5 cf, Re FFE Group (Qld) Pty Ltd [1984] 1 Qd R 267.
[2.2 Com-BIFSOPA Heading 3] 122.5    The meaning of “employer”
[2.4 Com-BIFSOPA CDI Normal Body Text] Generally failure to give notice of the claim to the actual employer will normally be fatal. For the purpose of s 10(1)(a), the employer is the person who contracted with the contractor for building work ultimately performed by the subcontractor: see Walsh & Donaghay v Queensland (unreported, Dist Ct Mackay, Wylie QCJ, No 194 of 1986, 20 May 1987). Thus in that case, a notice of intention to claim charge purported to comply with the requirement to show the name of the employer by showing “The Manager, State Works Department” when it should have shown “State of Queensland” was held to suffer from an inaccuracy being a reference only to the relevant executive branch of the legal persona which was the employer rather than the former entity, or in the alternative, that in the notice suffered from a want of form and that the name of the employer did not appear. However, as the notice effectively came into the possession of the employer through its agents it was held that nothing turned on the notice's deficiencies and the notice was valid. See also Stumann v Spansteel Engineering Pty Ltd [1986] 2 Qd R 471. A notice containing the name of the wholly owned subsidiary of an employer is not affected by the inaccuracy and therefore the notice should not be treated as invalid: Paks Contractors Pty Ltd v Maruko Inc [1993] 1 Qd R 119, per McKenzie J at p 124.
[2.2 Com-BIFSOPA Heading 3] 122.6    The significance of “notice of claim of charge”
[2.4 Com-BIFSOPA CDI Normal Body Text] It is clear that this phrase refers to the notice that is required to be given to the employer or superior contractor under s 10(1)(a) (now s 122(1)BIFSOPA (Qld)). That phrase is mentioned in the heading to s 10 (s 122BIFSOPA (Qld)), repeated in subss (2), (3) and (5), and in ss 11(1), 12(1) and 15(1)(b). Furthermore, the reference to “a notice of having made the claim” is plainly not a reference to the original notice of claim, but to the notice required to be given to the contractor by s 10(1)(b): Re Austco Pty Ltd [1985] 2 Qd R 1 at p 2. Thus, where notice of having made the claim is not served on the contractor at the contractors place of residence or registered place of business, no charge will attach: Re Peter Bushnell & Co Pty Ltd [1985] 2 Qd R 383.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Parker Bros Constructions Pty Ltd v Hancock [2001] QCA 397 the Court of Appeal held that giving a copy of the Notice of Intention to Claim a Charge to the contractor was sufficient notice that the claim had been made.
[2.4 Com-BIFSOPA CDI Normal Body Text] The expression “claim” can be described as referring to an asserted entitlement to be paid under the subcontract for specified work performed by the subcontractor under the subcontract, upon which a notice of claim of charge given under s 10(1) of the SCA (now s 122BIFSOPA (Qld)) is based, in order to obtain the entitlement of the charge which arises by operation of s 5(1) of the SCA (now s 109BIFSOPA (Qld)). One “claim” can support two notices of claim of charge and as a result ss 10(7) and (8) of the amended SCA have no application. There is nothing in the SCA to preclude a subcontractor lodging two notices of claim of charge based on the one claim where each of the notices is given to a different contractor or employer. The extent to which the claim is satisfied under one charge must affect the amount of the claim, which can be pursued under the other charge: Re University of Queensland [2003] 2 Qd R 577; [2003] QSC 158.
[2.4 Com-BIFSOPA CDI Normal Body Text] The amendments to s 10(7) and (8) do not reverse the effect of Hewitt Nominees Pty Ltd v Commissioner for Railways [1979] Qd R 256 in recognising the validity of a leapfrogging charge: Re University of Queensland [2003] 2 Qd R 577; [2003] QSC 158.
[2.2 Com-BIFSOPA Heading 3] 122.7    Successive notices
[2.4 Com-BIFSOPA CDI Normal Body Text] A subcontractor may serve, within time, a second set of notices creating a charge where the first set of notices, because of irregularities, did not create a charge: Re Peter Bushnell & Co Pty Ltd [1985] 2 Qd R 383. Compare this with the situation where a charge created by the first set of notices is extinguished by operation of s 15(3)SCA it is then necessary to commence the requisite proceedings within two months in which case, a further set of notices cannot be given creating a charge: Re Castley (unreported, QSC, Lucas SPJ, 18 September 1980); and Evans Deakin Industries Ltd v Commonwealth [1985] 2 Qd R 152 at p 153.
[2.2 Com-BIFSOPA Heading 3] 122.8    Subcontractors as “secured creditors”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Re Stockport (NQ) Pty Ltd (2003) 127 FCR 291; [2003] FCA 31 some subcontractors issued notices of charge after signing a deed of arrangement. Mansfield J held that those subcontractors could not both issue a notice as a secured creditor under the SCA and participate in any distribution under the deed.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Surfers Paradise Investments Pty Ltd (in liq) v Davoren Nominees Pty Ltd [2004] 1 Qd R; [2003] QCA 458, the Court said that for an election to surrender security there must be unequivocal conduct amounting to an election to surrender the security and prove as an unsecured creditor. The inference, however, could be displaced by evidence to the contrary, thus even if a subcontractor also lodges a proof of debt in error, it might, depending on the facts, be able to be withdrawn.
[2.4 Com-BIFSOPA CDI Normal Body Text] Mansfield J, in Re Stockport (NQ) Pty Ltd (2003) 127 FCR 291; [2003] FCA 31 said that s 5 of the SCA (equivalent to s 109BIFSOPA (Qld)) operates to create a set of rights in a subcontractor in relation to the money payable by an employer to the principal or superior contractor, although there is no crystallisation of those rights so as to create specific proprietary interest in the moneys, or the chose in action, being the entitlement to those moneys, by attaching the rights to the specific property until notice is given under s 10 (equivalent to s 122BIFSOPA (Qld)). His Honour was of the view that s 5(2)SCA creates what is in effect a floating charge.
[2.2 Com-BIFSOPA Heading 3] 122.9    Retention money
[2.4 Com-BIFSOPA CDI Normal Body Text] It seems plain from the definition in s 5SCA (s 109BIFSOPA (Qld)) that during the course of a subcontractors work there would be no retention money for the purposes of s 10(3) (equivalent of s 122(5)BIFSOPA (Qld)): Rees v Mount Isa City Council [1979] Qd R 553 per Matthews J at p 556.
[2.2 Com-BIFSOPA Heading 3] 122.10    The meaning of “the work”
[2.4 Com-BIFSOPA CDI Normal Body Text] For the purposes of this subsection, “the work” means the subcontractors work rather than that of the head contractor: see Stucoid Pty Ltd v Stadiums Pty Ltd (1960) 107 CLR 521; and Rees v Mount Isa City Council [1979] Qd R 553 per Matthews J at p 555.
[2.2 Com-BIFSOPA Heading 3] 122.11    Completion of the work
[2.4 Com-BIFSOPA CDI Normal Body Text] In Multiplex Constructions Pty Ltd v Abigroup Contractors Pty Ltd [2004] QSC 198 the Court said that completion of work means completion of the whole of the subcontract works and not just completion of the work in respect of which the claim was made. Therefore, three months notice must be given from completion of the subcontract works as a whole: per Chesterman J at [35]. This decision has been upheld in the Court of Appeal see Multiplex Constructions Pty Ltd v Abigroup Contractors Pty Ltd [2005] 1 Qd R 610; [2005] QCA 61, (Multiplex 4), per Jerrard JA at [21] – [22].
[2.2 Com-BIFSOPA Heading 3] 122.12    Notice “given within 3 months after such completion”
[2.4 Com-BIFSOPA CDI Normal Body Text] Multiplex Constructions v Abigroup Contractors Pty Ltd [2004] QSC 198, (Multiplex 3), involved a dispute as to the construction of s 10SCA. Does the section require the notice of claim of charge to be given within three months of the completion of the works the subject of the claim, or within three months of the completion of the subcontract works as a whole? Chesterman J held that work must mean the “whole of the subcontract work” in order for s 10(1A) and s 10(2)SCA to be reconciled. By reading:
[2.5 Com-BIFSOPA Normal Body Quote] s 10(2) in this way, the time limit does not commence until all the work required by the subcontract has been done. To read s 10(2) otherwise would mean that there would be different time limits in respect of each and every different item of work as it is completed. His Honour also said the SCA was designed to protect the interests of subcontractors and is remedial in character. Giving three months from the completion of the subcontract works to claim a charge reduces scope for confusion, lessens the burden on the subcontractor to give notice when it is busy doing the work, and provides additional time for giving notice. This decision has been upheld in the Court of Appeal see Multiplex Constructions Pty Ltd v Abigroup Contractors Pty Ltd [2005] 1 Qd R 610; [2005] QCA 61, (Multiplex 4), per Jerrard JA at [21] – [22].
[2.2 Com-BIFSOPA Heading 3] 122.13    Consequence of non-compliance
[2.4 Com-BIFSOPA CDI Normal Body Text] If the subcontractor fails to give notice of having made the claim to the contractor to whom the money is payable, a charge does not attach: Re Austco Pty Ltd (unreported, QSC, Weld M, OS 502 of 1984, 27 April 1984) at pp 2 – 3; and Evans Deakin Industries Ltd v Commonwealth [1985] 2 Qd R 152 at p 153.
[2.2 Com-BIFSOPA Heading 3] 122.14    General
[2.4 Com-BIFSOPA CDI Normal Body Text] This provision, as it states clearly, is concerned with preserving the validity of the notice where there is:
[BCIPA Bullet Point] any inaccuracy; or
[BCIPA Bullet Point] any want of form.
[2.4 Com-BIFSOPA CDI Normal Body Text] If there is any inaccuracy or want of form the validity of the notice is not affected if:
[BCIPA Bullet Point] the money sought to be charged can be ascertained with reasonable certainty from the notice; and
[BCIPA Bullet Point] the amount of claim can be ascertained with reasonable certainty from the notice: Paks Contractors Pty Ltd v Maruko Inc [1993] 1 Qd R 119 per McKenzie J at pp 122 to 124.
[2.2 Com-BIFSOPA Heading 3] 122.15    Money ascertainable
[2.4 Com-BIFSOPA CDI Normal Body Text] The moneys sought to be charged will be sufficiently identified if it is described as the money payable to a named contractor or superior contractor under its contract with the person to whom the notice is given: Rees v Mount Isa City Council [1979] Qd R 553 per Connolly J at p 562. A notice claiming a charge upon “money and retention money that is now or will be payable” constitutes a departure from the form which is more than a mere inaccuracy or want of form, and therefore is a notice not given pursuant to this section: Re Galaxy Investments Pty Ltd (in liq) (unreported, QSC, White J, 92 of 0152, 17 August 1993).
[2.4 Com-BIFSOPA CDI Normal Body Text] Where the claim made by the subcontractor only relates to work performed in respect of a single contract between the contractor and that contractor's employer, the invoice numbers and amounts would enable the money sought to be charged and the amount of the claim to be ascertained. However, where there are two separate contracts between the contractor and that contractor's employer and the notice of claim by the subcontractor relates to work done under both contracts, the money sought to be charged cannot be ascertained from the notice. On receipt of the notice, the builder would be unable to ascertain from the notice what amount was sought to be charged from the moneys payable under the separate contracts. As a result, the notice is not one given pursuant to s 10 of the SCA: CND Earthmoving Pty Ltd v Packer & Jonasson (unreported, Dist Ct Bne, 13 January 1995).
[2.4 Com-BIFSOPA CDI Normal Body Text] McPherson J (as he then was) in Re FFE Group (Qld) Pty Ltd [1984] 1 Qd R 267 was faced with a description of money to which a charge was to attach as “money and/or retention money”. His Honour concluded that such usage usually gives rise to three possibilities:
[BCIPA Bullet Point] money;
[BCIPA Bullet Point] retention money; or
[BCIPA Bullet Point] money and retention money.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour held, at p 271, that such a formula led to uncertainty about which money was subject to the statutory charge and so liable to be retained and would therefore eliminate the distinction maintained between the two by ss 10 and 15 of the SCA which, was clearly the intention of the legislature. It was accordingly a matter of substance and not merely a matter of form.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Road Surfaces Group Pty Ltd v Brown [1987] 2 Qd R 792 the notice referred to “all moneys, including retention moneys”. Williams J said, at p 804, that reading s 10 of the SCA as a whole, he was of the view that a charge may be claimed either on retention money, on moneys otherwise payable under the contract or on both. The only material consequence resulting from the third categorisation is that such notice would have to be given within three months of the completion of the work, and not within the extended period permissible when the charge is sought against retention moneys only.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Walter Construction Group Ltd v J & L Schmider Investments Pty Ltd [2001] QSC 124, Muir J struck out a notice of charge because it was impossible to detect, from the claim, how it was made up, the basis of the claim and whether it was capable of being a claim for money payable or to become payable - it was deficient and did not meet the requirements of s 10(i).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Quality Concrete Pty Ltd v Honeycombes Townsville Pty Ltd [2005] QSC 192 the main contractor argued that the notice of claim of charge given by the subcontractor was defective and incapable of creating a charge under the SCA because a claim was made in respect of two subcontracts. However, the two subcontracts were inter-related to such an extent that they were to be read together in order to determine the works to be carried out by the subcontractor in relation to the project. In these circumstances Cullinane J, at [20], held that the notice of claim was valid as both the amount of the claim and the moneys sought to be charged were clear on the face of the notice.
[2.2 Com-BIFSOPA Heading 3] 122.16    Want of form
[2.4 Com-BIFSOPA CDI Normal Body Text] In Re Spantech Pty Ltd (unreported, QSC, Derrington J, 91 of 0298, 4 & 10 April 1991), Derrington J held that a notice given purportedly pursuant to Pt 11 of the Wages Act 1918 was void and of no force or effect whatsoever because it did not constitute a notice under this section.
[2.4 Com-BIFSOPA CDI Normal Body Text] McPherson J (as His Honour then was) pointed out at p 270 in Re FFE Group (Qld) Pty Ltd [1984] 1 Qd R 267, that the provisions of the Acts Interpretation Act 1954 (Qld) are displaced by the terms of s 10(5) which selects not want of form, which is the criterion in s 40 of the Acts Interpretation Act 1954, but whether the money sought to be charged can be ascertained with reasonable certainty from the notice.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Parker Bros Constructions Pty Ltd v Hancock [2001] QCA 397 the Court of Appeal said the section was “permissive” and there is no particular statutory requirement regarding the use of a particular form.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Ed Ahern Plumbing (Gold Coast) Pty Ltd v JM Kelly (Project Builders) Pty Ltd [2008] 2 Qd R 123; [2007] QCA 452 the subcontractor had initially sent a notice of claim of charge to the contractor that purported to attach an annexure containing a statement of account, which was not attached. It was held at [3] and [42] that the document that was initially provided did not constitute a notice since the particulars as required by s 10(1)(a)SCA (see also s 123BIFSOPA (Qld)) were inadequate which is a matter of substance, rather than of form or inaccuracy. However, the annexure that was subsequently sent to the contractor had the effect that a notice was given to the contractor in accordance with s 10(1)(a)SCA (see also s 123BIFSOPA (Qld)) when the missing annexure was later provided. This conclusion was not a result of statutory interpretation, but rather due to the fact that the document which was initially sent was plainly incomplete, in that it signalled that the appellants notice of claim of charge included further documents which had not yet been given by the appellant: at [4]. Consequently, it was held that the time period in which the subcontractor must institute proceedings after giving a notice, did not start to run in this instance, until the missing annexure was given.
[2.2 Com-BIFSOPA Heading 3] 122.17    Leapfrogging charge
[2.4 Com-BIFSOPA CDI Normal Body Text] In Re University of Queensland [2003] 2 Qd R 577; [2003] QSC 158, Mullins J held that there was nothing in the SCA to preclude a subcontractor lodging two notices of claim of charge based on the one claim where each of the notices is given to a different contractor or employer. The extent to which the claim is satisfied under one charge must affect the amount of the claim which can be pursued under the other charge.

```

### `annotated/section_125.txt`
```
# Annotated BIF Act source — Section 125
# Chapter: CHAPTER 4 – Subcontractors’ charges
# Section title: Withdrawing a notice of claim
# DOCX paragraphs: 4601-4605

[2 Com-BIFSOPA Heading 1] SECTION 125 – Withdrawing a notice of claim 
[1 BIFSOPA Heading] 125    Withdrawing a notice of claim 
[1.3 BIFSOPA level 1 (CDI)] A subcontractor may at any time withdraw, wholly or partly, a notice of claim by giving notice of the withdrawal, in the approved form, to the person to whom the subcontractor gave the notice of claim. 
[1.3 BIFSOPA level 1 (CDI)] If a subcontractor withdraws, wholly or partly, a notice of claim under subsection (1), the subcontractor must give a copy of the notice of withdrawal to each of the persons to whom the subcontractor gave a copy of the notice of claim.

```
