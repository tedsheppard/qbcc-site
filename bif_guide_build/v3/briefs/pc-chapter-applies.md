# Element-page brief — pc-chapter-applies
**Title:** Chapter 3 must apply (and not be excluded)
**Breadcrumb:** Requirements of a payment claim
**Anchor id:** `pc-chapter-applies`
**Output file:** `bif_guide_build/v3/pages/page_pc-chapter-applies.html`

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

### `statute/chapter_3/section_061.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 61 — Application of chapter
(1) Subject to subsections (2) to (4), this chapter applies to
construction contracts—

(a) whether written or oral, or partly written and partly oral;
and
(b) whether expressed to be governed by the law of
Queensland or a jurisdiction other than Queensland; and
(c) whether entered into before or after the commencement
of this section, other than to the extent the repealed
Building and Construction Industry Payments Act 2004
continues to apply to unfinished matters under
section 205.
(2) This chapter does not apply to—
(a) a construction contract to the extent that it forms part of
a loan agreement, a contract of guarantee or a contract
of insurance under which a recognised financial
institution undertakes—
(i) to lend an amount or to repay an amount lent; or
(ii) to guarantee payment of an amount owing or
repayment of an amount lent; or
(iii) to provide an indemnity relating to construction
work carried out, or related goods and services
supplied, under the construction contract; or
(b) a construction contract for the carrying out of domestic
building work if a resident owner is a party to the
contract, to the extent the contract relates to a building
or part of a building where the resident owner resides or
intends to reside; or
(c) a construction contract under which it is agreed that the
consideration payable for construction work carried out
under the contract, or for related goods and services
supplied under the contract, is to be calculated other
than by reference to the value of the work carried out or
the value of the goods and services supplied.
(3) This chapter does not apply to a construction contract to the
extent it includes—
(a) provisions under which a party undertakes to carry out
construction work, or supply related goods and services
Page 91 Current as at 27 April 2025

in relation to construction work, as an employee of the
party for whom the work is to be carried out or the
related goods and services are to be supplied; or
(b) provisions under which a party undertakes to carry out
construction work, or to supply related goods and
services in relation to construction work, as a condition
of a loan agreement with a recognised financial
institution; or
(c) provisions under which a party undertakes—
(i) to lend an amount or to repay an amount lent; or
(ii) to guarantee payment of an amount owing or
repayment of an amount lent; or
(iii) to provide an indemnity relating to construction
work carried out, or related goods and services
supplied, under the construction contract.
(4) This chapter does not apply to a construction contract to the
extent it deals with construction work carried out outside
Queensland or related goods and services supplied for
construction work carried out outside Queensland.
(5) In this section—
resident owner, in relation to a construction contract for
carrying out domestic building work, means a resident owner
under the Queensland Building and Construction Commission
Act 1991, schedule 1B, section1, but does not include a
person—
(a) who holds, or should hold, an owner-builder permit
under the Queensland Building and Construction
Commission Act 1991 relating to the work; or
(b) who is a building contractor within the meaning of the
Queensland Building and Construction Commission Act
1991.

```

### `statute/chapter_3/section_062.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 62 — Effect of giving notice of claim for subcontractors’
charges
(1) This section applies if a person gives a notice of claim under
chapter 4 in relation to construction work or related goods and
services the subject of a construction contract.
(2) Proceedings or other action may not be started or continued
by the person under part 3 for all or part of the construction
work or related goods and services.
(3) Without limiting subsection(2), if the person gave a
respondent a payment claim for all or part of the construction
work or related goods and services before or at the same time
as giving the notice of claim—
(a) the respondent is not required to pay an amount to the
person under section 77(2) in relation to the claim; and
(b) amounts may not be recovered by the person as a debt
owing to the person in any court of competent
jurisdiction in relation to the claim; and
(c) if the person made an adjudication application in
relation to the claim and the application has not been
decided by an adjudicator before the notice of claim is
given, the person is taken to have withdrawn the
application; and
(d) if the person made an adjudication application in
relation to the claim and the application has been
decided by an adjudicator before the notice of claim was
given—
(i) the respondent to the application is not required to
pay the adjudicated amount under section90; and
(ii) the registrar must not give the person an
adjudication certificate under section91 relating to
the adjudication; and
(iii) any adjudication certificate provided in relation to
the adjudication can not be enforced by the person
under section 93 as a judgment of a court; and
Page 93 Current as at 27 April 2025

(e) the person may not suspend, or continue to suspend,
carrying out all or part of the construction work or the
supply of the related goods and services under
section 98.
(4) This section does not affect the operation of section 95 and an
adjudication application taken to have been withdrawn by the
person under subsection(3)(c) is taken to have been
withdrawn for the purpose of section95.
(5) This section does not stop the person serving, under this
chapter, a payment claim in relation to all or part of the
construction work or related goods and services and taking
other action under this chapter in relation to that claim, if the
notice of claim in so far as it relates to the construction work
or related goods and services, or part, is withdrawn.
(6) In this section—
notice of claim see section122(1).

```

### `statute/chapter_3/section_063.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 63 — Act does not limit claimant’s other rights
A claimant’s entitlements and remedies under this chapter do
not limit—
(a) another entitlement a claimant may have under a
construction contract; or
(b) any remedy a claimant may have for recovering the
other entitlement.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_061.txt`
```
# Annotated BIF Act source — Section 61
# Chapter: CHAPTER 3 – Progress payments
# Section title: Application of chapter
# DOCX paragraphs: 963-1054

[2 Com-BIFSOPA Heading 1] SECTION 61 – Application of chapter
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 61    Application of chapter
[1.3 BIFSOPA level 1 (CDI)] Subject to subsections (2) to (4), this chapter applies to construction contracts—
[1.4 BIFSOPA level 2 (CDI)] whether written or oral, or partly written and partly oral; and
[1.4 BIFSOPA level 2 (CDI)] whether expressed to be governed by the law of Queensland or a jurisdiction other than Queensland; and 
[1.4 BIFSOPA level 2 (CDI)] whether entered into before or after the commencement of this section, other than to the extent the repealed Building and Construction Industry Payments Act 2004 continues to apply to unfinished matters under section 205.
[1.3 BIFSOPA level 1 (CDI)] This chapter does not apply to— 
[1.4 BIFSOPA level 2 (CDI)] a construction contract to the extent that it forms part of a loan agreement, a contract of guarantee or a contract of insurance under which a recognised financial institution undertakes— 
[1.5 BIFSOPA level 3 (CDI)] to lend an amount or to repay an amount lent; or
[1.5 BIFSOPA level 3 (CDI)] to guarantee payment of an amount owing or repayment of an amount lent; or
[1.5 BIFSOPA level 3 (CDI)] to provide an indemnity relating to construction work carried out, or related goods and services supplied, under the construction contract; or
[1.4 BIFSOPA level 2 (CDI)] a construction contract for the carrying out of domestic building work if a resident owner is a party to the contract, to the extent the contract relates to a building or part of a building where the resident owner resides or intends to reside; or 
[1.4 BIFSOPA level 2 (CDI)] a construction contract under which it is agreed that the consideration payable for construction work carried out under the contract, or for related goods and services supplied under the contract, is to be calculated other than by reference to the value of the work carried out or the value of the goods and services supplied. 
[1.3 BIFSOPA level 1 (CDI)] This chapter does not apply to a construction contract to the extent it includes—
[1.4 BIFSOPA level 2 (CDI)] provisions under which a party undertakes to carry out construction work, or supply related goods and services in relation to construction work, as an employee of the party for whom the work is to be carried out or the related goods and services are to be supplied; or
[1.4 BIFSOPA level 2 (CDI)] provisions under which a party undertakes to carry out construction work, or to supply related goods and services in relation to construction work, as a condition of a loan agreement with a recognised financial institution; or
[1.4 BIFSOPA level 2 (CDI)] provisions under which a party undertakes—
[1.5 BIFSOPA level 3 (CDI)] to lend an amount or to repay an amount lent; or
[1.5 BIFSOPA level 3 (CDI)] to guarantee payment of an amount owing or repayment of an amount lent; or
[1.5 BIFSOPA level 3 (CDI)] to provide an indemnity relating to construction work carried out, or related goods and services supplied, under the construction contract.
[1.3 BIFSOPA level 1 (CDI)] This chapter does not apply to a construction contract to the extent it deals with construction work carried out outside Queensland or related goods and services supplied for construction work carried out outside Queensland.
[1.3 BIFSOPA level 1 (CDI)] In this section—
[1.1 BIFSOPA Body Text] resident owner, in relation to a construction contract for carrying out domestic building work, means a resident owner under the Queensland Building and Construction Commission Act 1991, schedule 1B, section 1, but does not include a person—
[1.4 BIFSOPA level 2 (CDI)] who holds, or should hold, an owner-builder permit under the Queensland Building and Construction Commission Act 1991 relating to the work; or
[1.4 BIFSOPA level 2 (CDI)] who is a building contractor within the meaning of the Queensland Building and Construction Commission Act 1991.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 61.1    A ‘rough and ready’ process
[2.4 Com-BIFSOPA CDI Normal Body Text] In Musico v Davenport [2003] NSWSC 977, the NSW Act was described by McDougall J as (at [107]):
[2.5 Com-BIFSOPA Normal Body Quote] a somewhat rough and ready way of assessing a builder’s entitlement to progress claims. It may also be accepted that the procedure is intended not only to be swift, but also to be carried out with the minimum amount of formality and expense.
[2.2 Com-BIFSOPA Heading 3] 61.2    A parallel regime
[2.4 Com-BIFSOPA CDI Normal Body Text] BCIPA has been held to create a ‘dual’ or ‘parallel’ contractual and statutory system: see Beckhaus v Brewarrina Council [2002] NSWSC 960, [60] (Macready AJ); Falgat Constructions Pty Ltd v Equity Australia Corp Pty Ltd [2005] NSWCA 49; (2005) 62 NSWLR 385.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer further to the commentary under section 101.2 in relation to this dual regime.
[2.2 Com-BIFSOPA Heading 3] 61.3    Strict compliance with the Act?
[2.4 Com-BIFSOPA CDI Normal Body Text] The statutory regime under BCIPA required ‘strict’ compliance. In F.K. Gardner & Sons Pty Ltd v Dimin Pty Ltd [2007] 1 Qd R 10; [2006] QSC 243, Lyons J said at [24]:
[2.5 Com-BIFSOPA Normal Body Quote] The Act sets up a statutory regime for the recovery of progress claims and it is dependent on a series of steps being completed. There must be a valid statutory entitlement to a progress payment before a payment claim can be made and then if a payment schedule does not issue within time the unpaid portion of the claim becomes a debt. Such a statutory regime depends on strict compliance with the provisions in the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] In De Neefe Signs Pty Ltd v Build1 (Qld) Pty Ltd; Traffic Technologies Traffic Hire Pty Ltd v Build1 (Qld) Pty Ltd [2010] QSC 279 Fryberg J, in response to a submission based on Dimin and strict compliance with the requirements of BCIPA, proceeded on the assumption that this submission was correct.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also the similar comments in Walter Construction Group Ltd v CPL (Surry Hills) Pty Ltd [2003] NSWSC 266 at [59] (Nicholas J):
[2.5 Com-BIFSOPA Normal Body Quote] It has been said that the Act offers to a claimant special statutory rights which override general contractual rights and place the claimant in a privileged position (Jemzone Pty Ltd v Trytan Pty Ltd (2002) NSWSC 395 at para 41 per Austin J). In order to exercise these rights compliance with the requirements which establish them are necessary.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Tailored Projects Pty Ltd v Jedfire Pty Ltd [2009] 2 Qd R 171; [2009] QSC 32 (at 176, [18] per Douglas J).
[2.4 Com-BIFSOPA CDI Normal Body Text] The need for ‘strict’ compliance has been considered in light of the statutory rights created by BCIPA: see Simcorp Development and Constructions Pty Ltd v Gold Coast Titans Property Pty Ltd [2010] QSC 162 at [26] where Douglas J said that:
[2.5 Com-BIFSOPA Normal Body Quote] Failure to adhere strictly to the statutory regime for the recovery of claims has been held to preclude reliance on the special statutory rights available under the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] Similar comments have also been made in the context of the NSW Act and the VIC Act. Under the NSW Act, Einstein J said in Taylor Projects Group Pty Ltd v Brick Dept Pty Ltd [2005] NSWSC 439 at [49]:
[2.5 Com-BIFSOPA Normal Body Quote] In my view it is simply critical for a rigid approach to be taken to compliance with the terms of the Act, particularly for the reason that the legislation provides for a fast dual-track interim determination, reserving the parties' final legal entitlements for subsequent determination.
[2.4 Com-BIFSOPA CDI Normal Body Text] Similarly, under the VIC Act, in Saville v Hallmarc Construction Pty Ltd [2015] VSCA 318, Warren CJ and Tate JA, referring to the decision of the New South Wales Court of Appeal in Chase Oyster Bar Pty Ltd v Hamo Industries Pty Ltd [2010] NSWCA 190; (2010) 78 NSWLR 393, said:
[2.5 Com-BIFSOPA Normal Body Quote] The considerations identified by the New South Wales Court of Appeal, as described above, indicate that the strict observance of the procedural steps in the statutory scheme, including time limits, was mandated by the legislature as the price by which a claimant can take the benefit of the regime. The observance of certain procedural steps are essential to the assumption of jurisdiction by an adjudicator. An assessment of whether those procedural steps have been observed may require matters of evaluation that travel beyond a mechanical process.
[2.4 Com-BIFSOPA CDI Normal Body Text] Contrast the above authorities with considerations, arising in the context of payment claims, that ‘reasonable’ compliance may be sufficient: see T&M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QCA 381, [33] (Philippides J; Fraser and White JJA agreeing); Nepean Engineering Pty Ltd v Total Process Services Pty Ltd (in liq) [2005] NSWCA 409 at [34] (Hodgson JA), [76] (Ipp JA).
[2.4 Com-BIFSOPA CDI Normal Body Text] Contrast also the view of the New South Wales Court of Appeal in Hawkins Construction (Australia) Pty Ltd v Mac’s Industrial Pipework Pty Ltd [2002] NSWCA 136 where Davies AJA considered that the equivalent to section 17(2) under the NSW Act:
[2.5 Com-BIFSOPA Normal Body Quote] should not be approached in an unduly technical manner… The terms used by subs (2) of s 13 are well understood words of the English language. They should be given their normal and natural meaning. As the words are used in relation to events occurring in the construction industry, they should be applied in a common sense practical manner.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Amasya Enterprises Pty Ltd & Anor v Asta Developments (Aust) Pty Ltd & Anor (No 2) [2015] VSC 500, [87] (Vickery J), citing Protectavale Pty Ltd v K2K Pty Ltd [2008] FCA 1248, [10]-[11] (Finkelstein J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Hawkesbury City Council v The Civil Experts Pty Ltd trading as TCE Contracting [2023] NSWSC 962, the New South Wales Supreme Court considered whether a claim which included costs for off-site overheads was a claim for “construction work”. Ball J noted that the respondent agreed to pay “Cost + 25%” for construction work and held that in the absence of a definition, the adjudicator was entitled to interpret “cost” as a reference to all costs incurred by the claimant in carrying out its construction business.
[2.2 Com-BIFSOPA Heading 3] 61.4    ‘Construction contract’
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to the commentary under 64.1 for the definition of ‘construction contract’.
[2.2 Com-BIFSOPA Heading 3] 61.5    Section 61(2) – exceptions
[2.4 Com-BIFSOPA CDI Normal Body Text] In Walter Construction Group Limited v CPL (Surry Hills) Pty Ltd [2003] NSWSC 266, Nicholas J set out the following approach to the exceptions (in the context of section 7(2) of the NSW Act):
[2.5 Com-BIFSOPA Normal Body Quote] It seems clear to me that the intention of the legislature is that the Act applies to any construction contract, s 7(1) being a statement of general application. It is also clear that its intention is that the classes of contract described in s 7(2) are exceptions to the general rule so that if a party wishes to avoid its application it must demonstrate that the contract is within an excepted class. Obviously, it is probable that the features of a contract which might bring it within an excepted class will be within the knowledge of the party seeking to avoid the Act. Consistent with the principles referred to, that circumstance is further indication of the intention that the onus of proof concerning the exception is on the party claiming that the contract is within it.
[2.2 Com-BIFSOPA Heading 3] 61.6    Section 61(2)(a) – ‘Forms part of a loan agreement’
[2.4 Com-BIFSOPA CDI Normal Body Text] The phrase ‘forms part of’ was considered in the context of the NSW Act in Consolidated Constructions Pty Ltd v Ettamogah Pub [2004] NSWSC 110. McDougall J considered that:
[2.5 Com-BIFSOPA Normal Body Quote] As a matter of ordinary English usage, something may be said to “form part of” another if the first thing is included or incorporated within the second. The first thing may form part of the second as a result of some natural process or as a result of some artificial process (for example, a process of manufacture). In general terms, the words “forms part of” seem to me to connote something akin to inclusion, as opposed to association. In a particular case, however, it may be difficult to discern the point at which association changes to inclusion: that is to say, the point at which one thing may be said to form part of, rather than merely to be associated with, another.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] In both ordinary English usage and legal usage, the words “forms part of” therefore seem to indicate a relationship that is more than ancillary or associative. It is not enough to say only that the two things in question are in some way connected, for example because the one bears in some way on the other. The point at which connection becomes inclusion – at which the ancillary becomes integral – may not be easy to discern, and will in any event depend upon the facts of the particular case and the terms of the particular documents.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour held that a ‘construction contract will not form part of a loan agreement unless, in some way, the former is included in, or incorporated into, the latter’ (at [21]).
[2.2 Com-BIFSOPA Heading 3] 61.7    Section 61(2)(a) – “indemnity relating to construction work carried out”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Forte Sydney Carlingford Pty Ltd v Li [2022] FCA 1499, Stewart J granted an injunction to restrain the claimant from re-submitting an adjudication application, where it had previously submitted the same application to adjudication on two occasions. The claimant argued that on both occasions, the adjudicator had committed jurisdictional error in determining that the whole payment claim was excluded from adjudication under s 7(3)(c)(iii) of the NSW SOP Act. In granting the injunction, Stewart J found that the adjudicator had not erred in his finding that the whole of the payment claim was excluded from adjudication under s 7(3)(c)(iii) of the NSW SOP Act act, as the entire claim was based on an indemnity under the construction contract and not a claim for construction works carried out.  Stewart J explained that the operation of s 7(3)(c)(iii) is to exclude any part of a payment claim which falls under such an indemnity from adjudication and therefore, the payment claim was invalid. In the BIFSOPA, s61(3)(c)(iii) is equivalent to s 7(3)(c)(iii) of the NSW SOP Act. 61.8    Section 61(2)(b) – “for the carrying out of domestic building work”.
[2.4 Com-BIFSOPA CDI Normal Body Text] This exception under section 61(2)(b) of the Act is subject to the definition of ‘resident owner’ as defined under section 61(5) of the Act. Under this definition, the section 61(2)(b) exemption does not apply if either section 61(5)(a) or (b) applies, that is, if the person:
[2.6 Com-BIFSOPA bullet] holds, or should hold, an owner-builder permit; or
[2.6 Com-BIFSOPA bullet] is a building contractor,
[2.6 Com-BIFSOPA bullet] under the QBCC Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] The definition of ‘resident owner’ in the context of section 3(5) of BCIPA was considered by the Court of Appeal in R J Neller Building Pty Ltd v Ainsworth [2008] QCA 397; [2009] 1 Qd R 390. Rejecting a submission that, for the purposes of BCIPA, an owner may have only one residence, Keane JA held:
[2.5 Com-BIFSOPA Normal Body Quote] In my respectful opinion, it is not possible to detect in the provisions of the BCIP Act an intention on the part of the legislature that the concept of "resident owner" should be confined in the way urged by Neller. The evident concern of the legislature in excluding contracts between builders and resident owners from the scope of the BCIP Act is to ensure that the ultimate consumers of the goods and services provided by the building and construction industry should not be subject to a regime which, among other things, imposes certain financial risks upon building owners who are participants in that industry to the advantage of the builders who contract with them. I shall discuss the effect of the BCIP Act upon the allocation of these financial risks in due course, but for the moment it is sufficient to say that, this being the concern which can be seen to inform the exclusion of "resident owners" from the scope of the legislation, there is little reason to think that it was any part of the legislative intention to alter the usual understanding that a person may be a resident of several locations at the same time.
[2.4 Com-BIFSOPA CDI Normal Body Text] The second exemption to the definition of “resident owner” was considered in Vis Constructions Ltd & Anor v Cockburn & Anor [2006] QSC 416 in relation to BCIPA. In this case, the owners of the land, a husband and wife, engaged a builder to construct a dwelling, which the owners intended to use as their residence. The husband held a licence to carry out plastering work under the-then QBSA Act. Notwithstanding that the joint owners were both the husband and the wife (who held no such relevant licence), the adjudicator concluded that the husband and wife were not a ‘resident owner’ under the second of the exclusions to the definition. In the proceedings to set aside the adjudicator’s decision, Jones J held that the decision ‘might well be wrong in law’, however, the adjudicator had turned his mind to the provisions of BCIPA and that ‘an error of law of this kind is a non-jurisdictional error and is not reviewable in these proceedings’.
[2.4 Com-BIFSOPA CDI Normal Body Text] In BWAY Group Pty v Pasiopoulos [2019] VCC 691, Marks J held that in determining whether a party to a domestic building contract is ‘in the business of building residences’ for the purposes of s7(2)(b) of the VIC Act (equivalent to section 61 of the Act), courts will consider a variety of salient features including a party’s commercial history. Only ‘objective indicia in evidence’ will be used to determine this fact, evidence to the subjective intention of the parties will not be relevant.
[2.2 Com-BIFSOPA Heading 3] 61.9    Section 61(2)(c) – ‘calculated other than by reference to…’
[2.4 Com-BIFSOPA CDI Normal Body Text] In the context of the NSW Act, in Brambles Australia Limited v Philip Davenport & Ors [2004] NSWSC 120, Einstein J said of the equivalent exclusion to section 61(2)(c) of the Act:
[2.5 Com-BIFSOPA Normal Body Quote] “Clearly enough, this sub-section serves to exclude certain construction contracts from the reach of the Act. The construction contracts so excluded are identified by reference to whether or not the consideration payable for construction work carried out under the contract or for related goods and services supplied under the contract is to be calculated.”
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour considered that the correct approach to be applied in considering this exemption was that of Nicholas J in Walter Construction Group Limited v CPL (Surry Hills) Pty Ltd [2003] NSWSC 266.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Sheppard Homes Pty Ltd v FADL Industrial Pty Ltd [2010] QSC 228, the amounts to be paid under the contract in question were determined not by the value of the services supplied between the builder and consultant, but by reference to the amounts payable by the owner to the builder. Fryberg J held that section 3(2)(c) of BCIPA applied to the contract between the builder and the consultant.
[2.2 Com-BIFSOPA Heading 3] 61.10    Section 61(3)(c) – ‘to guarantee payment’
[2.4 Com-BIFSOPA CDI Normal Body Text] In Walton Construction (Qld) Pty Ltd v Robert Salce & Ors [2008] QSC 235, an undertaking to pay money owed was held to fall clearly within the exclusion under section 3(3)(c)(ii) of BCIPA.
[2.2 Com-BIFSOPA Heading 3] 61.11    Section 61(4) – construction work or related goods or services outside Queensland
[2.4 Com-BIFSOPA CDI Normal Body Text] The purpose of the equivalent provision under BCIPA was described in Wiggins Island Coal Export Terminal Pty Ltd v Monadelphous Engineering Pty Ltd & Ors [2015] QSC 307 by Philip McMurdo J in the following terms: 
[2.5 Com-BIFSOPA Normal Body Quote] “The evident purpose of s 3(4) is to confine the operation of the Act to circumstances which have a relevant association with Queensland. It is of the essence of construction work, as defined, that it involves work in relation to certain land. It is therefore logical that the Act would be given a territorial limitation which is defined according to whether the relevant land is in Queensland.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Wiggins Island Coal Export Terminal Pty Ltd v Monadelphous Engineering Pty Ltd & Ors [2015] QSC 307, a contract provided for the supply, installation and commissioning of a shiploader and associated piece of equipment called a ‘tripper’ to the Wiggins Island Coal Export Terminal near Gladstone. Both the shiploader and the tripper were fabricated in Malaysia. In proceedings challenging an adjudication decision, the respondent alleged that, pursuant to section 3(4), BCIPA did not apply to a payment claim made about four weeks before the shiploader arrived at Wiggins Island. In rejecting this submission, Philip McMurdo J held:
[2.5 Com-BIFSOPA Normal Body Quote] The work of the fabrication of this equipment was carried out outside Queensland.  But that work was construction work only because it was an integral part of construction work undertaken inside Queensland.  In my view, that fabrication could not be relevantly characterised both as construction work carried out outside Queensland and an integral part of construction work carried out within Queensland. 
[2.5 Com-BIFSOPA Normal Body Quote] In requiring the fabrications of this equipment, the contract was providing for part of the construction of a structure in Queensland and was, in the sense of s 3(4), “dealing with” construction work carried out in Queensland. It follows that the application of the Act was not displaced by s 3(4).
[2.4 Com-BIFSOPA CDI Normal Body Text] In South City Plaster Pty Ltd v Modscape Pty Ltd [2018] VCC 1576, Macnamara J held that the VIC Act applied to a construction contract, despite that some of the work was carried out in Tasmania, because the contract did not propose for any work to be done outside the State of Victoria and the work done in Tasmania under the contract was “not an integral part of construction work.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Lendlease Building Pty Ltd v BCS Airport Systems Pty Ltd & Ors [2024] QSC 164, the head contractor, Lendlease, sought to quash an adjudication determination on the basis that the payment claim issued by BSC was invalid as part of the works were carried out in NSW (and the adjudication was made under the Qld BIF Act). In this case, the site, Gold Coast Airport, straddles the Qld-NSW border, and some of works were carried out in NSW.
[2.4 Com-BIFSOPA CDI Normal Body Text] Sullivan J rejected Lendlease’s argument. It was held that, under s 61(4) of the Act, the BIF Act applied as part of the works were carried out in Queensland. That section would only apply, precluding the operation of the BIF Act, if the construction work carried out is ‘wholly’ outside Queensland. In terms of the construction work carried out outside of Queensland, what matters is the location of the building, structure or works, as opposed to where each activity might take place disconnected from the location of the ultimate building, structure or works involved. Furthermore, the Act does not require a payment claim in a cross-border project to purport, in a reasonable way, to identify the location of the construction work. As such, the mere fact that a payment claim may include a component of ‘excluded construction work’ does not mean it cannot be a payment claim within the meaning of section 61(4) of the BIF Act. Sullivan J held that the payment claim submitted by the sub-contractor for construction work carried out in both Queensland and New South Wales was valid.
[2.2 Com-BIFSOPA Heading 3] 61.12    Final payments
[2.4 Com-BIFSOPA CDI Normal Body Text] The Act applies to final payments under a construction contract. Refer to the section 75(3) definition of ‘final payment’ and the section 64 definition of ‘progress payment’.

```

### `annotated/section_062.txt`
```
# Annotated BIF Act source — Section 62
# Chapter: CHAPTER 3 – Progress payments
# Section title: Effect of giving notice of claim for subcontractors’ charges
# DOCX paragraphs: 1055-1080

[2 Com-BIFSOPA Heading 1] SECTION 62 – Effect of giving notice of claim for subcontractors’ charges 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 62    Effect of giving notice of claim for subcontractors’ charges
[1.3 BIFSOPA level 1 (CDI)] This section applies if a person gives a notice of claim under chapter 4 in relation to construction work or related goods and services the subject of a construction contract. 
[1.3 BIFSOPA level 1 (CDI)] Proceedings or other action may not be started or continued by the person under part 3 for all or part of the construction work or related goods and services.
[1.3 BIFSOPA level 1 (CDI)] Without limiting subsection (2), if the person gave a respondent a payment claim for all or part of the construction work or related goods and services before or at the same time as giving the notice of claim— 
[1.4 BIFSOPA level 2 (CDI)] the respondent is not required to pay an amount to the person under section 77(2) in relation to the claim; and
[1.4 BIFSOPA level 2 (CDI)] amounts may not be recovered by the person as a debt owing to the person in any court of competent jurisdiction in relation to the claim; and
[1.4 BIFSOPA level 2 (CDI)] if the person made an adjudication application in relation to the claim and the application has not been decided by an adjudicator before the notice of claim is given, the person is taken to have withdrawn the application; and
[1.4 BIFSOPA level 2 (CDI)] if the person made an adjudication application in relation to the claim and the application has been decided by an adjudicator before the notice of claim was given—
[1.5 BIFSOPA level 3 (CDI)] the respondent to the application is not required to pay the adjudicated amount under section 90; and
[1.5 BIFSOPA level 3 (CDI)] the registrar must not give the person an adjudication certificate under section 91 relating to the adjudication; and
[1.5 BIFSOPA level 3 (CDI)] any adjudication certificate provided in relation to the adjudication can not be enforced by the person under section 93 as a judgment of a court; and
[1.4 BIFSOPA level 2 (CDI)] the person may not suspend, or continue to suspend, carrying out all or part of the construction work or the supply of the related goods and services under section 98. 
[1.3 BIFSOPA level 1 (CDI)] This section does not affect the operation of section 95 and an adjudication application taken to have been withdrawn by the person under subsection (3)(c) is taken to have been withdrawn for the purpose of section 95. 
[1.3 BIFSOPA level 1 (CDI)] This section does not stop the person serving, under this chapter, a payment claim in relation to all or part of the construction work or related goods and services and taking other action under this chapter in relation to that claim, if the notice of claim in so far as it relates to the construction work or related goods and services, or part, is withdrawn. 
[1.3 BIFSOPA level 1 (CDI)] In this section—
[1.1 BIFSOPA Body Text] notice of claim see section 122(1).
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 62.1    Notice of claim of charge
[2.4 Com-BIFSOPA CDI Normal Body Text] In Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd & Ors [2013] QCA 386, the Court of Appeal said that it was implicit in the words of section 4(1) of BCIPA that a notice of claim or charge must be valid.
[2.2 Com-BIFSOPA Heading 3] 62.2    Withdrawn
[2.4 Com-BIFSOPA CDI Normal Body Text] In Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd & Ors [2013] QCA 386 the Court of Appeal said, in reference to the first instance decision and in the context of BCIPA, that:
[2.4 Com-BIFSOPA CDI Normal Body Text] There is an even stronger argument that the reference to a notice of claim of charge being “withdrawn” in s 4(6) of the Act is a reference to the withdrawal of a notice of claim of charge in such a way that is effective in respect of all relevant parties regardless of whether it is in the approved form.

```

### `annotated/section_063.txt`
```
# Annotated BIF Act source — Section 63
# Chapter: CHAPTER 3 – Progress payments
# Section title: Act does not limit claimant’s other rights
# DOCX paragraphs: 1081-1093

[2 Com-BIFSOPA Heading 1] SECTION 63 – Act does not limit claimant’s other rights
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 63    Act does not limit claimant other rights
[1.1 BIFSOPA Body Text] A claimant’s entitlements and remedies under this chapter do not limit— 
[1.4 BIFSOPA level 2 (CDI)] another entitlement a claimant may have under a construction contract; or 
[1.4 BIFSOPA level 2 (CDI)] any remedy a claimant may have for recovering the other entitlement
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 63.1    Other rights preserved
[2.4 Com-BIFSOPA CDI Normal Body Text] In Jemzone Pty Ltd v Trytan Pty Ltd [2002] NSWSC 395, Austin J observed that the structure (of the NSW Act):
[2.5 Com-BIFSOPA Normal Body Quote] generally leaves it to the construction contract to define the rights of the parties but makes “default provisions” to fill in the contractual gaps.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to the commentary under 101.2 in relation to the interaction between the contractual and statutory rights.

```
