# Element-page brief — sc-service
**Title:** Service of the notice
**Breadcrumb:** Valid subcontractor's charge
**Anchor id:** `sc-service`
**Output file:** `bif_guide_build/v3/pages/page_sc-service.html`

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

### `statute/chapter_4/section_123.txt`
```
# Source: BIF Act 2017 — Chapter 4
# Section 123 — Copy of notice of claim to contractor
(1) This section applies if a subcontractor gives a notice of claim
to a person obliged to pay money to a contractor under a
contract.
(2) The subcontractor must—
(a) give the contractor a copy of the notice of claim; and
(b) advise the contractor of the name and address of the
person given the notice of claim.
(3) If the subcontractor does not comply with subsection(2), the
notice is of no effect and the subcontractor’s charge does not
attach.

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

### `annotated/section_123.txt`
```
# Annotated BIF Act source — Section 123
# Chapter: CHAPTER 4 – Subcontractors’ charges
# Section title: Copy of notice of claim to contractor
# DOCX paragraphs: 4584-4591

[2 Com-BIFSOPA Heading 1] SECTION 123 – Copy of notice of claim to contractor
[1 BIFSOPA Heading] 123    Copy of notice of claim to contractor 
[1.3 BIFSOPA level 1 (CDI)] This section applies if a subcontractor gives a notice of claim to a person obliged to pay money to a contractor under a contract.
[1.3 BIFSOPA level 1 (CDI)] The subcontractor must—
[1.4 BIFSOPA level 2 (CDI)] give the contractor a copy of the notice of claim; and 
[1.4 BIFSOPA level 2 (CDI)] advise the contractor of the name and address of the person given the notice of claim.
[1.3 BIFSOPA level 1 (CDI)] If the subcontractor does not comply with subsection (2), the notice is of no effect and the subcontractor’s charge does not attach.

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
