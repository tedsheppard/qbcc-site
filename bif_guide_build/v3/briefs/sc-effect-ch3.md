# Element-page brief — sc-effect-ch3
**Title:** Effect on the Chapter 3 process
**Breadcrumb:** Valid subcontractor's charge
**Anchor id:** `sc-effect-ch3`
**Output file:** `bif_guide_build/v3/pages/page_sc-effect-ch3.html`

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

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

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
