# Element-page brief — d-valuation-later
**Title:** Bind valuation in later applications
**Breadcrumb:** Requirements of an adjudicator's decision
**Anchor id:** `d-valuation-later`
**Output file:** `bif_guide_build/v3/pages/page_d-valuation-later.html`

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

### `statute/chapter_3/section_087.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 87 — Valuation of work etc. in later adjudication application
(1) This section applies if, in deciding an adjudication
application, an adjudicator has decided the value of—
(a) any construction work carried out under a construction
contract; or
(b) any related goods and services supplied under a
construction contract.
Note—
See section72 for the valuation of construction work and related goods
and services.
(2) Any adjudicator must, in any later adjudication application
that involves the working out of the value of the construction
work or of the related goods and services, give the work, or
the goods and services, the same value as that previously
decided by the adjudicator unless the claimant or respondent
satisfies the adjudicator concerned that the value of the work,
or the goods and services, has changed since the previous
decision.
(3) However, if a decision or order of a court changes the value of
the construction work or of the related goods and services, the
adjudicator must give the work, or the goods and services, the
same value as changed by the decision or order unless the
Page 115 Current as at 27 April 2025

claimant or respondent satisfies the adjudicator concerned that
the value of the work, or the goods and services, has changed
since the decision or order.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_087.txt`
```
# Annotated BIF Act source — Section 87
# Chapter: CHAPTER 3 – Progress payments
# Section title: Valuation of work etc. in later adjudication application
# DOCX paragraphs: 2767-2801

[2 Com-BIFSOPA Heading 1] SECTION 87 – Valuation of work etc. in later adjudication application 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 87    Valuation of work etc. in later adjudication application 
[1.3 BIFSOPA level 1 (CDI)] This section applies if, in deciding an adjudication application, an adjudicator has decided the value of—
[1.4 BIFSOPA level 2 (CDI)] any construction work carried out under a construction contract; or 
[1.4 BIFSOPA level 2 (CDI)] any related goods and services supplied under a construction contract.
[1.6 BIFSPOA example/note] Note—
[1.7 BIFSOPA example/note body] See section 72 for the valuation of construction work and related goods and services.
[1.3 BIFSOPA level 1 (CDI)] Any adjudicator must, in any later adjudication application that involves the working out of the value of the construction work or of the related goods and services, give the work, or the goods and services, the same value as that previously decided by the adjudicator unless the claimant or respondent satisfies the adjudicator concerned that the value of the work, or the goods and services, has changed since the previous decision. 
[1.3 BIFSOPA level 1 (CDI)] However, if a decision or order of a court changes the value of the construction work or of the related goods and services, the adjudicator must give the work, or the goods and services, the same value as changed by the decision or order unless the claimant or respondent satisfies the adjudicator concerned that the value of the work, or the goods and services, has changed since the decision or order.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 87.1    Valuation-Entitlement distinction
[2.4 Com-BIFSOPA CDI Normal Body Text] In Rothnere Pty Ltd v Quasar Constructions NSW Pty Ltd [2004] NSWSC 1151, McDougall J said that a decision by an adjudicator may involve both questions of valuation (or quantification) and entitlement, or it may involve one or the other.
[2.4 Com-BIFSOPA CDI Normal Body Text] This distinction was clarified by McDougall J in John Goss Projects Pty Ltd v Leighton Contractors Pty Ltd [2006] NSWSC 789; (2006) 66 NSWLR 707; where his Honour, in the context of the NSW Act, said:
[2.5 Com-BIFSOPA Normal Body Quote] Sections 9 and 10 make it clear that there is a distinction between the calculation of the amount of a progress payment (which is, ultimately, what the adjudicator is required to do) and the valuation of construction work. That is the distinction that I sought to point out (on reflection, in a way that was perhaps unduly brief and somewhat delphic) in para [43] of my decision in Rothnere.
[2.4 Com-BIFSOPA CDI Normal Body Text] The approach of McDougall J in Rothnere and John Goss Projects has been approved both by:
[2.6 Com-BIFSOPA bullet] the New South Wales Court of Appeal in Dualcorp Pty Ltd v Remo Constructions Pty Ltd [2009] NSWCA 69; (2009) 74 NSWLR 190;;  and
[2.6 Com-BIFSOPA bullet] the Queensland Supreme Court in ACN 060 559 971 Pty Ltd v O’Brien [2007] QSC 91; [2008] 2 Qd R 396;.
[2.4 Com-BIFSOPA CDI Normal Body Text] Accordingly, on this distinction, where an amount affecting the calculation of the amount due under a payment claim is not referrable to a particular item of construction work (and, by extension under section 87(1) of the Act, related goods and services), such an amount will not attract the operation of section 87 of the Act. This was the approach taken by Mullins J in ACN 060 559 971 Pty Ltd v O’Brien [2007] QSC 91; [2008] 2 Qd R 396.
[2.4 Com-BIFSOPA CDI Normal Body Text] The approach by Mullins J in ACN 060 559 971 Pty Ltd v O’Brien [2007] QSC 91; [2008] 2 Qd R 396 has been subsequently approved by the Queensland Supreme Court and Court of Appeal.
[2.2 Com-BIFSOPA Heading 3] 87.2    Valuation in later adjudication application
[2.4 Com-BIFSOPA CDI Normal Body Text] In Hitachi Ltd v O’Donnell Griffin Pty Ltd [2008] QSC 135, two adjudication decisions were in issue before the court. In the first adjudication, due to the large number of variation claims (113), the adjudicator had adopted a selective approach, assessing only 12 variation claims.
[2.4 Com-BIFSOPA CDI Normal Body Text] As to the second adjudication decision, the adjudicator noted the effect of section 27 of BCIPA (equivalent to section 87 of the Act), but concluded that the first adjudicator, notwithstanding that all of the claims were not addressed, had in fact made a valuation of all of the claims. On this basis, and on the basis that the adjudicated amount from the first adjudication had been paid, the second adjudicator assessed the adjudicated amount at $nil. Skoien AJ held that the first adjudicator had, for the purposes of section 27 of BCIPA, valued only six variation claims, none of which were in the later payment claim.
[2.4 Com-BIFSOPA CDI Normal Body Text] In obiter, Douglas J said in Ball Construction Pty Ltd v Conart Pty Ltd [2014] QSC 124, in relation to a section 27 of BCIPA issue that was not necessary to decide, that were it necessary, he ‘would have concluded that the private decision of the parties not to enforce an earlier decision does not affect what the subsequent adjudicator must take into account for the purposes of s 27’.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Bezzina Developers Pty Ltd v Deemah Stone (Qld) Pty Ltd & Ors [2008] QCA 213; [2008] 2 Qd R 495, the Court of Appeal held that, where an adjudicator:
[2.5 Com-BIFSOPA Normal Body Quote] does not know that there is, or that there probably is, a prior adjudication decision, no occasion arises for the exercise of the power to call for further submissions concerning any prior adjudication decision.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court of Appeal also held that:
[2.5 Com-BIFSOPA Normal Body Quote] an obligation to seek further submissions from the parties on that topic should not be implied in s 27 when an express power to seek further submissions exists in s 25(4)(a). Implying such an obligation would also create practical difficulties in view of the very limited time available to adjudicators to make decisions and the absence of compulsive powers to obtain any prior adjudication. If the parties fail to inform the adjudicator of an earlier adjudication decision, it seems too much to ask of the adjudicator that he or she should delay a decision for the purpose merely of making enquiries and at the risk of losing any right to remuneration.
[2.4 Com-BIFSOPA CDI Normal Body Text] Iris Broadbeach Business Pty Ltd v Descon Group Australia Pty Ltd [2024] QSC 16, Wilson J found that the adjudicator fell into jurisdictional error by failing to engage with Iris’s submissions as required under section 87(2) of the Act. In this case, the adjduicator had, in a previous adjudication, valued work completed by an architect at $200,000. In the adjudication the subject of the proceeding, Iris submitted an invoice suggesting only $183,780 was payable. The invoce was dated after the previous adjudication decision, whereas the adjudicator applied her previous valuation with no consideration for Iris’s submissions. The Court was satified that Iris’s evidence indicated that the architect item should havevalued less than $200,000 and therefore the adjduicator fell into jurisidication error.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Goyder Wind Farm 1 Pty Ltd v GE Renewable Energy Australia Pty Ltd & Ors [2024] SASC 108, Stein J dismissed the application to quash the adjudicator's determination. In this case, the contractor issued three separate payment claims, and two of which proceeded to adjudication determinations. The payment claims related to the EOTs and associated delay costs. The Respondent to the adjudication argued that the adjudicator failed to adopt the earlier valuation of the work as determined in a previous adjudication, given that the delay costs arising from the EOTs constitute a singular claim for delay costs. In dismissing the application, Stein J found that the delay costs did not constitute a singular claim for the same construction work; hence, the adjudicator was not bound in the second adjudication to apply the valuation from the first adjudication. His Honour also observed that, in any event, the valuation issues concerning s 22(4) of the SA Act alleged by the Principal was not a matter of jurisdiction, even if it had been made out.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Liquidated damages. In ACN 060 559 971 Pty Ltd v O’Brien [2007] QSC 91; [2008] 2 Qd R 396 it was held that liquidated damages under a construction contract were not referrable to any particular item of construction work. In the circumstances, the Court held that the subsequent adjudicator had erred in law in relying on section 27 of the Act in adopting the decision of the previous adjudicator on liquidated damages.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Harlech Enterprises Pty Ltd v Beno Excavations Pty Ltd [2022] ACTCA 42, the Court found that the s 24(4) ACT SOP Act (which is equivalent to s 87(2) BIFSOPA), provides that “an adjudicator’s valuation of work may only be disturbed by a subsequent adjudicator in limited circumstances”, and emphasised that the SOP Act’s purpose is not to protect any broader findings of adjudicators by applying the doctrine of issue estoppel.

```
