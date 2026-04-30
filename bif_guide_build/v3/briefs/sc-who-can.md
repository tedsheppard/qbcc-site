# Element-page brief — sc-who-can
**Title:** Who can give a notice of charge
**Breadcrumb:** Valid subcontractor's charge
**Anchor id:** `sc-who-can`
**Output file:** `bif_guide_build/v3/pages/page_sc-who-can.html`

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

### `statute/chapter_4/section_104.txt`
```
# Source: BIF Act 2017 — Chapter 4
# Section 104 — Definitions
In this chapter—
contractor, for a contract, means the party to the contract who
is required to carry out work under the contract.
contract price, for a contract, means the amount the
contractor is entitled to be paid under the contract or, if the
amount can not be accurately calculated, the reasonable

estimate of the amount the contractor is entitled to be paid
under the contract.
court means the Magistrates Court, District Court or Supreme
Court in which a proceeding may be taken under this chapter.
land means any land within Queensland and includes land
under water.
notice of claim see section122(1).
person includes an unincorporated association.
security, for a contract, means something—
(a) given to, or for the direct or indirect benefit of, the party
to the contract for whom the work the subject of the
contract is to be performed, by the contractor; and
(b) intended to secure, wholly or partly, the performance of
the contract; and
(c) in the form of either, or a combination of both, of the
following—
(i) an amount of money, other than an amount held as
a retention amount;
(ii) 1 or more valuable instruments, whether or not
exchanged for, or held instead of, a retention
amount.
structure, for a structure on land under water, includes a
structure made up of component parts that include—
(a) component parts fixed to the land; and
(b) component parts that rise and fall with the rise and fall
of the water, and that are otherwise confined in their
location by component parts fixed to the land.
Example of a structure included under this definition—
A marina made up of fixed pylons, and pontoons that rise and fall with
the water level that are otherwise confined in their location by the
pylons.
subcontractor’s charge see section109(4).
valuable instrument means any of the following—
Page 141 Current as at 27 April 2025

(a) banker’s undertaking;
(b) bond;
(c) inscribed stock;
(d) guarantee policy;
(e) interest-bearing deposit;
(f) another instrument, to the extent it is convertible into an
amount of money.
variation, of a contract, means an addition to, or an omission
from, the work required to be carried out under the contract.
work see section105.

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

### `annotated/section_104.txt`
```
# Annotated BIF Act source — Section 104
# Chapter: CHAPTER 4 – Subcontractors’ charges
# Section title: Definitions
# DOCX paragraphs: 4114-4245

[2 Com-BIFSOPA Heading 1] SECTION 104 – Definitions
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 104    Definitions
[1.3 BIFSOPA level 1 (CDI)] In this chapter— 
[1.1 BIFSOPA Body Text] contractor, for a contract, means the party to the contract who is required to carry out work under the contract. 
[1.1 BIFSOPA Body Text] contract price, for a contract, means the amount the contractor is entitled to be paid under the contract or, if the amount can not be accurately calculated, the reasonable estimate of the amount the contractor is entitled to be paid under the contract.
[1.1 BIFSOPA Body Text] court means the Magistrates Court, District Court or Supreme Court in which a proceeding may be taken under this chapter. 
[1.1 BIFSOPA Body Text] land means any land within Queensland and includes land under water. notice of claim see section 122(1). 
[1.1 BIFSOPA Body Text] person includes an unincorporated association. 
[1.1 BIFSOPA Body Text] security, for a contract, means something— 
[1.4 BIFSOPA level 2 (CDI)] given to, or for the direct or indirect benefit of, the party to the contract for whom the work the subject of the contract is to be performed, by the contractor; and
[1.4 BIFSOPA level 2 (CDI)] intended to secure, wholly or partly, the performance of the contract; and 
[1.4 BIFSOPA level 2 (CDI)] in the form of either, or a combination of both, of the following—
[1.5 BIFSOPA level 3 (CDI)] an amount of money, other than an amount held as a retention amount; 
[1.5 BIFSOPA level 3 (CDI)] 1 or more valuable instruments, whether or not exchanged for, or held instead of, a retention amount.
[1.3 BIFSOPA level 1 (CDI)] structure, for a structure on land under water, includes a structure made up of component parts that include— 
[1.4 BIFSOPA level 2 (CDI)] component parts fixed to the land; and 
[1.4 BIFSOPA level 2 (CDI)] component parts that rise and fall with the rise and fall of the water, and that are otherwise confined in their location by component parts fixed to the land. 
[1.6 BIFSPOA example/note] Example of a structure included under this definition—
[1.7 BIFSOPA example/note body] A marina made up of fixed pylons, and pontoons that rise and fall with the water level that are otherwise confined in their location by the pylons. 
[1.1 BIFSOPA Body Text] subcontractor’s charge see section 109(4). valuable instrument means any of the following— 
[1.4 BIFSOPA level 2 (CDI)] banker’s undertaking;
[1.4 BIFSOPA level 2 (CDI)] bond; 
[1.4 BIFSOPA level 2 (CDI)] inscribed stock; 
[1.4 BIFSOPA level 2 (CDI)] guarantee policy; 
[1.4 BIFSOPA level 2 (CDI)] interest-bearing deposit; 
[1.4 BIFSOPA level 2 (CDI)] another instrument, to the extent it is convertible into an amount of money. 
[1.1 BIFSOPA Body Text] variation, of a contract, means an addition to, or an omission from, the work required to be carried out under the contract. 
[1.1 BIFSOPA Body Text] work see section 105.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 104.1    General
[2.4 Com-BIFSOPA CDI Normal Body Text] What must first be determined are terms of the agreement between the applicant and the respondent in each case. Having determined the terms of the agreement in each case, it is necessary to determine if what was contracted to do constituted work. It is not the correct approach to look at what was actually done: see Dowstress (Qld) Pty Ltd v Mission Congregation Servants of the Holy Spirit [1987] 1 Qd R 150 per Derrington J at pp 151-152 and Moynihan J (Kelly ACJ agreeing) at p 157. What characterises a subcontractor as defined is not what he does but that he contracts with a contractor for the performance of works: Re Gradeline Contracting Pty Ltd [1998] 2 Qd R 251.
[2.4 Com-BIFSOPA CDI Normal Body Text] Subcontractors are put in a favoured position by the provision of the BIFSOPA and its predecessor SCA: see Lucas J in Re Castley (unreported, QSC, Lucas SPJ, 18 September 1980) referred to by Thomas J (as His Honour then was) in KGK Constructions Pty Ltd v Tabray Pty Ltd [1985] 2 Qd R 173 at p 176. The whole tenor of decisions interpreting its sometimes elusive meaning is to require strict compliance with its terms: Stumann v Spansteel Engineering Pty Ltd [1986] 2 Qd R 471 at p 477.
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 104 defines “contractor” as meaning “the party to the contract who is required to carry out work under the contract”. Its predecessor, s 3(1) of the SCA defined “subcontractor” as meaning “a person who contracts with a contractor or with another subcontractor for the performance of work”. The definition of work in the SCA was found to be an inclusive definition. References to work, or labour done, or commenced on the land, and to the supply of material used or brought on premises, are included in the definition of work: Sun Engineering (Qld) Pty Ltd v Dynac Pty Ltd [2000] QSC 213.
[2.2 Com-BIFSOPA Heading 3] 104.2    Land in Queensland
[2.4 Com-BIFSOPA CDI Normal Body Text] In Transfield Pty Ltd v Fondside Australia Pty Ltd [2000] QSC 480, the applicant builder brought an application to cancel a notice of charge given by the receiver of the respondent subcontractor for work carried out on the construction of a gas pipeline between Victoria and NSW. Mullins J found, that as a matter of construction, the SCA applied where the land on which the work under the head contract is being performed is within Queensland. The reference in a Queensland Act to “land” must prima facie be taken to be a reference to land within the jurisdiction and there were no other indications in the SCA as to any other basis for giving some territorial limitation to its operation. This is reflected in s 104 of BIFSOPA's definition of “land” meaning “any land within Queensland and includes land under water”.
[2.2 Com-BIFSOPA Heading 3] 104.3    Approach to interpretation
[2.4 Com-BIFSOPA CDI Normal Body Text] The previous SCA conferred special rights and privileges, the enforcement of which depended on strict compliance with its terms. Where the requirements of the SCA were not complied with, the court had no jurisdiction to enforce them: Queensland Tiling Service Pty Ltd v Sisters of St Joseph of Most Sacred Heart of Jesus [1962] QWN 46 cited in Stumann v Spansteel Engineering Pty Ltd [1986] 2 Qd R 471 at p 477.
[2.2 Com-BIFSOPA Heading 3] 104.4    Definition of subcontractor's charge
[2.4 Com-BIFSOPA CDI Normal Body Text] The definition of subcontractor's charge can be found in section 109 of the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] In the Common Law, a charge has been determined to be:
[2.5 Com-BIFSOPA Normal Body Quote] a well-known thing. If one man owes a debt to another, a creditor of the latter can, by bringing in the debtor, charge the debt in his hands so as to prevent him from paying it to his own creditor, and oblige him to pay it to the creditor who obtains the charge. Why is that a charge? Because the debt is in the hands of the man who has to pay it.
[2.4 Com-BIFSOPA CDI Normal Body Text] See Re Potts; Ex parte Taylor [1893] 1 QB 648, per Lord Esher.
[2.2 Com-BIFSOPA Heading 3] 104.5    Effect of charge
[2.4 Com-BIFSOPA CDI Normal Body Text] I think it is correct to say that the Act creates rights over such moneys in the hands of the employer as well as personal rights against him; and does more than preserve this property in medio until the rights of the claimant are determined.
[2.4 Com-BIFSOPA CDI Normal Body Text] See Stapleton v FTS O'Donnell Griffin & Co (Qld) Pty Ltd (1961) 108 CLR 106 per McTiernan J at p 119 (in relation to the Contractors' and Workmen's Liens Acts 1906 to 1921 (Qld)). See also Seka Pty Ltd v Fabric Dyeworks (Australia) Pty Ltd (No 2) (1991) 4 ACSR 455 and Clark v Trevilyan [1963] QWN 11.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Isakka v South Australian Asset Management Corp [2002] QCA 549 it was held that a mortgagee making payments under its securities over land and assets through a receiver and manager was not doing so as “employer”. It is not until a cheque is presented and paid that there is any money to which the statutory charge is capable of attaching. Before then, it is simply a chose in action vested in the mortgagee against its Bank in which a subcontractor has no proprietary or other right or interest. It was only when the money was paid to the employer or to the contractor that there was any “money” to which a charge was capable of attaching.
[2.2 Com-BIFSOPA Heading 3] 104.6    Definition of work
[2.4 Com-BIFSOPA CDI Normal Body Text] The approach of the Supreme Court of New Zealand has been to emphasise the nature of the definition in the New Zealand Act as a denotative definition and to concentrate attention upon the section of the New Zealand Act which imposes the charge. This approach was maintained when considering the SCA see: Re Austco Pty Ltd (unreported, QSC, Weld M, OS 502 of 1984, 27 April 1984) per Master Weld at p 4. “Work” is defined in s 105 of the BIFSOPA. It would appear that the approach to interpretation of s 105BIFSOPA (Qld) would be the same as to s 3 of the previous SCA as the definition in most respects is identical. Whether what was done on or in respect of the land amounts in a given case to work within the meaning of s 21 of the previous SCA must depend upon the meaning of that word as ordinarily understood: Caldow Properties Ltd v H J G Low & Assoc Ltd [1971] NZLR 311, per Turner at p 314.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Leach v Dynac Pty Ltd [2000] QSC 300 the Supreme Court held that work done by a subcontractor off the land on which the structure in question is being constructed is not work within the meaning of that term in the SCA and that when a substantial part of the work to be performed is not work within the meaning of that term in the SCA, none of the work is then within the scheme of the SCA. As to the first proposition in this case, see the previous s 3AASCA (now in s 105BIFSOPA (Qld)) which has extended the definition of work to include particular manufacture or fabrication and supply of labour of project specific components for a contract or subcontract. The second proposition is still relevant in that if a substantial part of the work to be performed is not work, then it is arguable that none of the work would then fall within the scheme of the SCA, now contained in BIFSOPA (Qld).
[2.4 Com-BIFSOPA CDI Normal Body Text] Timber clearing service on land was found to be work within the definition of the SCA. “The clearing of timber is a necessary and integral part of the construction process”, Griffiths Powerline Maintenance Pty Ltd v IDS Consulting Services Pty Ltd [2006] QDC 59.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Re Bulk Materials (Coal Handling) Pty Ltd (Administrators Appointed) (unreported, QSC, Demack J, No 3 of 1997, 7 February 1997), Demack J held that as a charge applies only in respect of work done under the subcontract and a contract to supply goods on which work of manufacture is performed by a subcontractor is not a contract for work within the meaning of the SCA, the entitlement to the charge arises from a contract to perform work alone, unless the subcontractor is performing work on the land the subject of contract. In these circumstances, the materials used on that land are embraced within the inclusive definition of work in s 3. That was not the case in this matter. Whilst the subcontractors contribution to the whole project of constructing an ore sorting magnesium plant was very substantial this was is not a factor that has any significance under the SCA.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Griffiths Powerline Maintenance Pty Ltd v IDS Consulting Services Pty Ltd [2006] QDC 59, the Court held (following Re Leighton Contractors Pty Ltd [1985] 2 Qd R 377) that the clearing of trees for the erection of transmission towers was a necessary and integral part of the construction process and constituted work for the purposes of the SCA. Boulton DCJ further held that the use of the term “hire” in the contract documents was not determinative of the nature of the contractual arrangement and that the subcontract was not for a mere hire of machinery but a contract for services, utilising the machines and operators.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Sempac Pty Ltd v Stockport (Q) Pty Ltd [2004] QDC 87, Wall DCJ followed the interpretation of Holmes J in Sun Engineering (Qld) Pty Ltd v Dynac Pty Ltd [2000] QSC 213 that the definition of work is an inclusive definition and pointed out that the term “includes” was used rather than “means” in the definition.
[2.2 Com-BIFSOPA Heading 3] 104.7    Ordinary meaning
[2.4 Com-BIFSOPA CDI Normal Body Text] The ordinary meaning of the word work is no doubt capable of referring to anything which is done by a person in the course of his trade or calling: Caldow Properties Ltd v H J G Low & Assoc Ltd [1971] NZLR 311, per Richmond J at p 320. The views expressed by Richmond J in Caldow Properties were cited with approval by de Jersey J (as His Honour then was) in Re Leighton Contractors Pty Ltd [1985] 2 Qd R 377 at pp 379-380. It is not the correct approach to the problem of defining the term work to go to the Oxford English Dictionary for an appropriate definition and then remove the exclusions from its ambit. The remedy created by the SCA is an unusual remedy in favour of a restricted class of persons. It is a statutory remedy and must be found within the four corners of the statute. That is not to say that the given definition of work is exhaustive: Re Peter Fardoulys Pty Ltd [1983] 1 Qd R 345 at p 347.
[2.2 Com-BIFSOPA Heading 3] 104.8    The test
[2.4 Com-BIFSOPA CDI Normal Body Text] The test is not necessarily whether what has been done has been done upon or in respect of the land; it is whether what was contracted to be done is work upon or in respect of the land: Caldow Properties Ltd v H J G Low & Assoc Ltd [1971] NZLR 311, per Turner at p 315. See also Moynihan J in Dowstress (Qld) Pty Ltd v Mission Congregation Servants of the Holy Spirit [1987] 1 Qd R 150 at p 157. The insertion of a clause in a contract cannot bring within the statutory and ordinary meaning of work the supply of plant and machinery which would not be within that meaning if those clauses did not appear in the contract: Bayliss v Wellington City Corp [1957] NZLR 836, per Barrowclough CJ at p 843.
[2.2 Com-BIFSOPA Heading 3] 104.9    The enquiry
[2.4 Com-BIFSOPA CDI Normal Body Text] The enquiry must always be: first, what is the work the contractor agreed to perform; and second, did the subcontractor agree to perform part of it: Bayliss v Wellington City Corp [1957] NZLR 836 per Barrowclough CJ at p 843.
[2.2 Com-BIFSOPA Heading 3] 104.10    Use of the word “includes”
[2.4 Com-BIFSOPA CDI Normal Body Text] The result of the choice of the word “includes” is to retain the ordinary and natural meaning of the word work and to give it also the meaning defined in the statute: Watkins v Scott [1928] NZLR 628; Haynes v McKillop (1905) 24 NZLR 833; Brown v E G Laurie (in liq) [1930] NZLR 23. Cf Bayliss v Wellington City Corp [1957] NZLR 836, per Barrowclough CJ at p 839.
[2.2 Com-BIFSOPA Heading 3] 104.11    The extent of work
[2.4 Com-BIFSOPA CDI Normal Body Text] It would require courage, if not temerity, to suggest that today the word work is confined to manual labour only: Brown v E G Laurie (in liq) [1930] NZLR 23 per Blair J at p 26. The ingredient is the “supplying of labour, manually or by brain, and supplying material for the purpose of incorporation” as specifically provided for in the last sentence of the definition of work: Winstone Ltd v Wellington City [1972] NZLR 399, per Woodhouse J at p 403.
[2.2 Com-BIFSOPA Heading 3] 104.12    The meaning of “in connection with”
[2.4 Com-BIFSOPA CDI Normal Body Text] The introductory phrase “in connection with” should be given the meaning “as an integral part of”: Caldow Properties Ltd v H J G Low & Assoc Ltd [1971] NZLR 311, per Richmond J at p 322. Thus, work includes also the supply of materials used or brought on premises to be used by a subcontractor in connection with other work, the subject of his contract or subcontract: Re Austco Pty Ltd (unreported, QSC, Weld M, OS 502 of 1984, 27 April 1984) per Master Weld at p 6. The provision of catering, janitorial and associated managerial services to construction workers is not something done upon the land “in connection with” the construction work but would be better characterised as something done in connection with the provision and maintenance of the workforce for the construction: Re Leighton Contractors Pty Ltd [1985] 2 Qd R 377. The repair of earthmoving equipment used on site is not work done upon the land in connection with the development: Re Turners Engineering (unreported, QSC, de Jersey J, 275 of 1988, 6 April 1988) at p 3.
[2.4 Com-BIFSOPA CDI Normal Body Text] The phrase “in connection with” bears a very wide connotation and perhaps is similar to “in respect of”, its extent in the end must be regulated by the particular context in which it appears. It is not wide enough to link services with actual construction: per de Jersey J (as His Honour then was) in Re Leighton Contractors Pty Ltd [1985] 2 Qd R 377 at p 380.
[2.2 Com-BIFSOPA Heading 3] 104.13    Paragraph (a) – “construction”
[2.4 Com-BIFSOPA CDI Normal Body Text] The expression “construction” is appropriate to describe an activity which makes a building where none existed before. It matters not that it is made out of another building structure: Fletcher Construction Australia Ltd v Southside Tower Developments Pty Ltd (unreported, VSC, Byrne J, 6668 of 96, 9 October 1996).
[2.2 Com-BIFSOPA Heading 3] 104.14    Demolition
[2.4 Com-BIFSOPA CDI Normal Body Text] The previous statutory definition of work speaks only of “construction, decoration, alteration or repair” with demolition being inferred. See Caldow Properties Ltd v H J G Low & Assoc Ltd [1971] NZLR 311 per Richmond J, at p 322; approved by White J in Winstone Ltd v Wellington City [1972] NZLR 399 at p 404 wherein it was found work may well embrace work purely of demolition. Subsequently, “demolition, removal or relocation of a building or other structure” has been included in the definition in s 105BIFSOPA (Qld) at s 105(1)(a)(v).
[2.2 Com-BIFSOPA Heading 3] 104.15    Decoration
[2.4 Com-BIFSOPA CDI Normal Body Text] The decoration of a building or other structure upon land may be rightly regarded as within the definition of work: Re Austco Pty Ltd (unreported, QSC, Weld M, OS 502 of 1984, 27 April 1984) at p 6.
[2.2 Com-BIFSOPA Heading 3] 104.16    The second paragraph
[2.4 Com-BIFSOPA CDI Normal Body Text] The grammatical construction of the second paragraph in the definition of work, however, must be read as though the word work were repeated in it, so that grammatically the sentence must read “and”work shall include the supply of “material used, or brought on the premises to be used, in connection with the work”: Re Official Assignee (1899) 17 NZLR 712, per Edwards J at p 718.
[2.4 Com-BIFSOPA CDI Normal Body Text] The second paragraph appears to have been included to explain the exclusionary part of the definition and therefore allow a subcontractor who is already performing work of such a kind (as will create a charge in his favour) to include in that charge the cost of supply of materials ancillary to that other work: Re Peter Fardoulys Pty Ltd [1983] 1 Qd R 345 at p 347.
[2.2 Com-BIFSOPA Heading 3] 104.17    Paragraph (c)
[2.4 Com-BIFSOPA CDI Normal Body Text] The drafting of the previous SCA s 3(e) is clumsy and has not been rectified in its current form (in same terms) in s 105(1)(c)(i)BIFSOPA (Qld). To begin, the exclusion has little effect because the definition is inclusive only, and if the delivery of goods upon which work has been performed by the subcontractors does not come within the ordinary meaning of a contract for work, then the exclusion, and any implications from the word “mere”, does not affect that. It is probably intended that the express statement of the exclusion should indicate, where the contract does no more than impose upon subcontractors the obligation to supply goods, that it is not a contract for work: Dowstress (Qld) Pty Ltd v Mission Congregation Servants of the Holy Spirit [1987] 1 Qd R 150 per Derrington J, at p 153.
[2.2 Com-BIFSOPA Heading 3] 104.18    Fabrication and delivery of chattels
[2.4 Com-BIFSOPA CDI Normal Body Text] Last Reviewed: 8-2-2019
[2.4 Com-BIFSOPA CDI Normal Body Text] The SCA does not create a charge when the head contract is merely for fabrication and delivery of chattels, with no work to be done after delivery. Nor would a charge arise, irrespective of the nature of the head-contract, when the subcontract is merely for the fabrication and delivery of chattels, with no work to be done after delivery: Re Peter Fardoulys Pty Ltd [1983] 1 Qd R 345 at p 348. This is maintained in BIFSOPA (Qld).
[2.2 Com-BIFSOPA Heading 3] 104.19    Supply of materials
[2.4 Com-BIFSOPA CDI Normal Body Text] If a contract is for the supply of materials generally, without specifying their purpose, so that the contractor could do what he pleases with the materials in question, the mere fact that they were used in a particular building would not have created a charge in favour of the person supplying the same. Where a contract is entered into for the supply of materials, however, for the purpose of work in connection with a particular building, it has become an essential term of the contract that such materials should be used only for the purpose for which they are supplied and it is wholly immaterial that, for the sake of convenience, the work has actually been performed in the supplier's own factory: Re Official Assignee (1899) 17 NZLR 712 per Edwards J at pp 718-9. The mere delivery of ordered goods cannot constitute work under the SCA, nor its replacement BIFSOPA (Qld). The delivery of gravel is not work under the SCA: per McMurdo DJC (as Her Honour then was) in Re Gradeline Contracting Pty Ltd [1998] QDC 099.
[2.2 Com-BIFSOPA Heading 3] 104.20    Supply and working on goods
[2.4 Com-BIFSOPA CDI Normal Body Text] Where there is an obligation under the subcontract to do work on goods supplied by the contractor, it cannot be said of the subcontractor that the subject of the head contract is the mere delivery of goods sold by a vendor under a contract for the sale of goods. It is one for the performance of work upon goods supplied by the contactor: Re DA Story Pty Ltd [1993] 2 Qd R 355, per Ryan J at p 359, cf Dowstress (Qld) Pty Ltd v Mission Congregation Servants of the Holy Spirit [1987] 1 Qd R 150. It is well established that the supply of timber is work within the SCA: Ball v Scott Timber Co Ltd [1929] NZLR 570, per Smith J at p 582. See [SCA 3.80].
[2.2 Com-BIFSOPA Heading 3] 104.21    Paragraph (1)(c)(vi) – the “hire of materials, plant or machinery”
[2.4 Com-BIFSOPA CDI Normal Body Text] Hiring within this section, and its previous incarnation in the SCA, embraces a contract which gives rise to a bailment. Thus, where trucks were not given to the possession of the contractor but remained with the subcontractor with only limited control to direct drivers to the unloading site (those drivers at all times remaining the servants of the subcontractor), there is not a hiring within the section: APA Transport Pty Ltd v Lobegeier Earthmoving Pty Ltd (unreported, Dist Ct Bne, Miller QC DCJ, 580 of 1987, 21 August 1987). A person who hires plant and machinery out to another for use has performed no work within the meaning of the New Zealand statute: Winstone Ltd v Wellington City [1972] NZLR 399, per White J at p 404.
[2.4 Com-BIFSOPA CDI Normal Body Text] When the hire charges relate to the hire of plant not intended to be incorporated in work, then the costs cannot be the subject of a charge under the SCA: per Williams J in Walter Wright Pty Ltd v Ronnor Pty Ltd (unreported, QSC, N Williams J, 3318 of 1983, 21 November 1983). See also Re Turners Engineering (unreported, QSC, de Jersey J, 275 of 1988, 6 April 1988). Followed by White J in John Holland Construction & Engineering Pty Ltd v Seovic Civil Engineering Pty Ltd (unreported, QSC, White J, OS 202 of 1995, 3 April 1995).
[2.2 Com-BIFSOPA Heading 3] 104.22    Hire of machinery with operator
[2.4 Com-BIFSOPA CDI Normal Body Text] The supply of earthmoving machines with men to operate them, though called hiring between the parties, may be the carrying out of work within the meaning of the definition contained in s 4 of the New Zealand Act: Farrier-Waimak Ltd v Hornby Development Ltd (in liq) [1962] NZLR 635 per Henry J at p 639.
[2.4 Com-BIFSOPA CDI Normal Body Text] The hiring of equipment and delivery of material to the site cannot be looked at in isolation: Farrier-Waimak Ltd v Hornby Development Ltd. Where the subcontractor provided men and equipment to excavate and remove material from the site and replace it with fill supplied by the contractor, and where an excavator was used to effect that work, those services fall within the definition of work in the SCA (which includes the supply of materials used or brought on premises to be used by a subcontractor in connection with other work the subject of his contract): per Hoath DCJ in CND Earthmoving Pty Ltd v Packer & Jonasson (unreported, Dist Ct Bne, 13 January 1995). See also discussion of SCA below which is applicable to the interpretation of BIFSOPA (Qld).
[2.4 Com-BIFSOPA CDI Normal Body Text] A subcontractor who hired his machine and operator only for the time the machine actually worked and who received instruction from the contractor on site as to what work was to be done, contracted to perform work within the meaning of the SCA. The supply of the grader and operator was simply a step towards the performance of the contract between contractor and subcontractor. It entitled the subcontractor to no payment at all until he operated the grader. Work was therefore an essential term of the agreement and the agreed obligations were performed only when that work was done: Re Hacking & Sons Pty Ltd (unreported, Dist Ct Bne, Skoien ACJDC, 157 of 1993, 24 August 1993).
[2.4 Com-BIFSOPA CDI Normal Body Text] A subcontractor requiring the provision of plant, machinery and two operators, who, during the standby times, could be reassigned to other suitable work on the project by the contractor is not work under the SCA. The contractors under the subcontract had full possession of the plant and machinery during the contract period and were obliged to maintain and repair it, returning it at the end of the job to the subcontractor. The subcontractor's operators by inference performed their work completely under the supervision and direction of the contractor: John Holland Construction & Engineering Pty Ltd v Seovic Civil Engineering Pty Ltd (unreported, QSC, White J, OS 202 of 1995, 3 April 1995).
[2.4 Com-BIFSOPA CDI Normal Body Text] The removing of fill from a work site where possession and control of the trucks remained with the subcontractor in circumstances where the subcontractor drove the trucks onto the site, waited whilst they were loaded by the contractor, and drove the vehicles with their loads from the site so that a building could be erected constituted work within the meaning of the SCA. The material was taken away and dumped at the direction of the contractor, not the subcontractor. There was no hiring as the agreement did not give rise to any bailment or possession: APA Transport Pty Ltd v Lobegeier Earthmoving Pty Ltd (unreported, Dist Ct Bne, Miller QC DCJ, 580 of 1987, 21 August 1987).
[2.4 Com-BIFSOPA CDI Normal Body Text] A subcontract partially written and partially oral, to supply a scraper and operator to perform all remaining scraper work on the site at an hourly rate, where payment was only to be for work done and the operator was not to be used on other work, was not a mere hiring, but a subcontract for work within the meaning of the SCA. It does not matter if the term “hire” is used in the subcontract if what was done was in fact work. The subcontractor was contracted to work on the land where the contract was being performed in connection with the construction, alteration or repair of a road on the land, per McMurdo DCJ (as Her Honour then was) in Re Gradeline Contracting Pty Ltd [1998] QDC 099.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Re Turners Engineering (unreported, QSC, de Jersey J, 275 of 1988, 6 April 1988), de Jersey J (as His Honour then was) said (at p 2) that:
[2.4 Com-BIFSOPA CDI Normal Body Text] without more, the provision of … machinery by the subcontractor would not amount to work within the meaning of the Act because of the final paragraph, para (g) of the definition of work in s 3 of the Act.
[2.2 Com-BIFSOPA Heading 3] 104.23    Compressors
[2.4 Com-BIFSOPA CDI Normal Body Text] The hiring out of compressors may well be a supplying of plant machinery, but it is not a supplying of material used or brought on the premises to be used in connection with the work: Bayliss v Wellington City Corp [1957] NZLR 836.
[2.2 Com-BIFSOPA Heading 3] 104.24    Fuel and oil
[2.4 Com-BIFSOPA CDI Normal Body Text] The supply of fuel and oil may not, in certain circumstances, be the supply of “material”. See Motor Rebuilds Ltd v Bollard [1956] NZLR 954; and Bayliss v Wellington City Corp [1957] NZLR 836.
[2.2 Com-BIFSOPA Heading 3] 104.25    Electricity
[2.4 Com-BIFSOPA CDI Normal Body Text] The supply of electrical energy to operate machinery and plant used in the erection of a mining dredge may have been work within the meaning of the Wages Protection and Contractors Liens Act 1908 (NZ): Kanieri Electric Ltd v Hansford & Mills Construction Co Ltd [1931] GLR 446.
[2.2 Com-BIFSOPA Heading 3] 104.26    Plant
[2.4 Com-BIFSOPA CDI Normal Body Text] Plant was never intended by the Legislature to be regarded as “material”. The hiring out of plant and equipment is not “work or labour” within the meaning of that phrase in the first line of the definition of work. It is a supplying or furnishing of plant so that work may be done, but it is not part of the work itself. Such a hiring can, therefore, fall within the definition of work only if it is consistent with the concluding words of that definition, or can ordinarily be regarded as work - for the definition begins “Work includes”. See Bayliss v Wellington City Corp [1957] NZLR 836, per Barrowclough CJ at p 840.
[2.2 Com-BIFSOPA Heading 3] 104.27    Machinery
[2.4 Com-BIFSOPA CDI Normal Body Text] The supply of machinery cannot, in the ordinary acceptance of the word, be regarded as work or labour: Bayliss v Wellington City Corp [1957] NZLR 836, per Barrowclough at p 841.
[2.2 Com-BIFSOPA Heading 3] 104.28    Road works
[2.4 Com-BIFSOPA CDI Normal Body Text] The formation, metalling, and construction of a road was work within the meaning of that term in s 48 of the Wages Protection and Contractors Liens Act 1908 (NZ): Black v Shaw (1913) 33 NZLR 194; referred to in Farrier-Waimak Ltd v Hornby Development Ltd (in liq) [1962] NZLR 635, at p 640.
[2.2 Com-BIFSOPA Heading 3] 104.29    Landscaping
[2.4 Com-BIFSOPA CDI Normal Body Text] Work, at least to the extent that it included paving, the placement of logs, sleeper steps, timber edge strips and log retaining walls, was within the clear intention of the definition. Although it is more doubtful that the supply of various landscaping materials is within the extended part of the definition which follows the paras (a) – (d) SCA (see now s 105(1)(a)BIFSOPA (Qld)): Re Austco Pty Ltd (unreported, QSC, Weld M, OS 502 of 1984, 27 April 1984) per Master Weld at p 6.
[2.2 Com-BIFSOPA Heading 3] 104.30    Supply of workmen
[2.4 Com-BIFSOPA CDI Normal Body Text] When a claim is brought by the subcontractor stating that the contractor had some contractual obligations for (i) the wages payable to the riggers when riggers were supplied from the workforce, and (ii) the subcontractor, to perform work under the direction of the contractor, not on behalf of the subcontractor but on behalf of the head contractor, there is no basis for the claim of a charge: Walter Wright Pty Ltd v Ronnor Pty Ltd (unreported, QSC, N Williams J, 3318 of 1983, 21 November 1983).
[2.4 Com-BIFSOPA CDI Normal Body Text] A subcontractor who was obliged to provide both equipment and an operator, with the operator to work under the head contractor's direction, was not considered work under the SCA. The agreement obliged the subcontractor to supply the equipment and operator but not to do any more specific work on the site. The subcontract was complete upon the supply of the equipment and driver. See: Re Turners Engineering (unreported, QSC, de Jersey J, 275 of 1988, 6 April 1988), at p 2.
[2.4 Com-BIFSOPA CDI Normal Body Text] The provision of an excavator and operator to excavate and remove material from the site and replace it with fill also supplied by the subcontractor was work within the definition of work in the SCA: CND Earthmoving Pty Ltd v Packer & Jonasson (unreported, Dist Ct Bne, 13 January 1995).
[2.2 Com-BIFSOPA Heading 3] 104.31    Cartage
[2.4 Com-BIFSOPA CDI Normal Body Text] The driving of trucks onto a site, waiting there with them while they are being loaded, and the driving of the vehicles with their loads on the site so that a building could be erected on the site, constitutes work: APA Transport Pty Ltd v Lobegeier Earthmoving Pty Ltd (unreported, Dist Ct Bne, Miller QC DCJ, 580 of 1987, 21 August 1987).
[2.2 Com-BIFSOPA Heading 3] 104.32    Architecture
[2.4 Com-BIFSOPA CDI Normal Body Text] Last Reviewed: 8-2-2019
[2.4 Com-BIFSOPA CDI Normal Body Text] The work of an architect who draws the plans from which a building is constructed and actually supervises the work while it is being done, ought to find its place within the BIFSOPA. But it may be doubtful whether work done in drawing the plans only can so be made the subject of a claim for lien, if someone other than the draughtsman has completed the superintendence: Caldow Properties Ltd v H J G Low & Assoc Ltd [1971] NZLR 311, per Turner J at p 315. An architect in designing and superintending the erection of a building unquestionably performs work, although the result is attained mainly by the brain and the manual element is negligible: Brown v E G Laurie (in liq) [1930] NZLR 23, per Blair J said at p 26. In Arnoldi v Guin (1875) 22 Gr 314 at p 316, Proudfoot VC concluded that:
[2.4 Com-BIFSOPA CDI Normal Body Text] The man who designs the building and superintends its erection actually does work upon it as if he had carried a hod … in common use the signification is, I think, somewhat more restricted, and perhaps would not embrace the duties of an architect in designing the building.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Scott & Furnadjieff v Lindstrom 48 DLR (2d) 286, Swencisky Co Ct J remarked that because the architect prepared plans and supervised construction in Arnoldi's, the observation by Proudfoot VC is obiter dictum. Section 6 of the Ontario Mechanics' and Wage Earners' Lien Act 1896 gave to:
[2.5 Com-BIFSOPA Normal Body Quote] any person who performs any work or service upon or in respect of … the … Erecting … or any building … for any owner, contractor or sub-contractor … a lien for the price of such work, service or materials upon the … building … and the land occupied thereby it enjoyed therewith, or upon or in respect of which such work or service is performed … limited, however, in amount to the sum justly owning, except as herein provided, by the owner.
[2.5 Com-BIFSOPA Normal Body Quote] (emphasis added)
[2.4 Com-BIFSOPA CDI Normal Body Text] In Read v Whitney (1919) 48 DLR 305, the plaintiff, a Toronto architect was, according to the usual (if not universal) custom, employed by the principal architect to superintend the building and act as assistant architect. On an appeal from the judgment of the Assistant Master in Ordinary in an action to enforce a lien under the SCA, Riddell J in the Ontario Supreme Court said at p 309:
[2.4 Com-BIFSOPA CDI Normal Body Text] I can see no reason why superintending the building is any less “service upon” the building than carrying bricks and mortar to the bricklayers, and I agree with the Vice-Chancellor (22 Gr. 315, 316) that drawing plans etc., is an essential thing “to be done in the construction of the work”, and that he who draws such plans for a building “actually does work upon it as if he had carried a hod”.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Mechanics' Lien Act 1960, limits the right of lien to a “workman, material-man, contractor, or sub-contractor”. The Court in Scott & Furnadjieff v Lindstrom 48 DLR (2d) 286, granted an application for orders that the clam for a lien be dismissed, and for cancellation of the mechanics' lien on the ground that the claim is for professional services rendered in preparing plans, and does not entitle the plaintiff to a lien under the SCA. Section 4(1) of the Mechanics' Lien Act 1960 (Canada, c 64), provided for a lien to “a person who (a) does or causes to be done any work upon or in respect of an improvement”. In Inglewood Plumbing & Gasfitting Ltd v Northgate Development Ltd (1966) 54 DLR 509 at 512, it was held that an architect had a valid and subsisting lien for services performed in the nature of the preparation of plans. Compare the situation of some engineers and quantity surveyors under para (f).
[2.2 Com-BIFSOPA Heading 3] 104.33    Damages
[2.4 Com-BIFSOPA CDI Normal Body Text] Damages for breach of contract cannot be the subject of a charge under the SCA: Groutco (Aust) Pty Ltd v Thiess Contractors Pty Ltd [1985] 1 Qd R 238; and Milgun Pty Ltd v Austco Pty Ltd [1988] 2 Qd R 670. Costs and interest on a judgment sum are not work done by a subcontractor under a subcontract. Nor are damages for conversion work done by a subcontractor under a subcontract: Prince Constructions Pty Ltd v Yendex Pty Ltd (unreported, QSC, 12 July 1989).
[2.2 Com-BIFSOPA Heading 3] 104.34    Restitutionary awards
[2.4 Com-BIFSOPA CDI Normal Body Text] Where the actual claim makes it plain that the claim is not in damages then it is distinguishable. McPherson J (as he then was) mentioned without deciding the question, whether a quantum meruit was recoverable by the old action of debt or only in an action of indebitatus assumpsit, and whether the latter was in substance an action of debt. His Honour noted that in Alexander v Ajax Insurance Co Ltd [1956] VLR 436 (at 445), Sholl J said that debt extended to quantum meruit for work although later on the same page said that such a claim is covered by the indebitatus or common count: Re Northbuild Pty Ltd (unreported, QSC, Shepherdson J, OS 6399 of 1998, 25 September 1998). In that decision the Court held inferentially and by way of obiter dicta, that a claim set out in detail as a debt, being in effect, a liquidated demand, was capable of founding a claim of charge. Cf Milgun Pty Ltd v Austco Pty Ltd [1988] 2 Qd R 670 at pp 671-672; and Morton Engineering Co Pty Ltd v Stork Wescon Australia Pty Ltd (1998) 15 BCL 192 at p 192.

```
