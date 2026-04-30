# Element-page brief — ec-contractor-response
**Title:** The contractor's response
**Breadcrumb:** Enforcing a subcontractor's charge
**Anchor id:** `ec-contractor-response`
**Output file:** `bif_guide_build/v3/pages/page_ec-contractor-response.html`

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

### `statute/chapter_4/section_128.txt`
```
# Source: BIF Act 2017 — Chapter 4
# Section 128 — Contractor given copy of notice of claim must respond
(1) This section applies if a subcontractor gives a contractor a
copy of a notice of claim.
(2) The contractor must give both of the following persons a
written response to the claim within 10 business days after the
contractor is given the copy of the notice of claim, unless the
contractor has a reasonable excuse—
(a) the person given the notice of claim;
(b) the subcontractor.
Maximum penalty—20 penalty units.
(3) The response to the claim must be made in the approved form
and—
(a) accept liability to pay the amount claimed; or
(b) accept liability to pay an amount stated in the response,
but otherwise dispute the claim; or
(c) dispute the claim.

```

### `statute/chapter_4/section_132.txt`
```
# Source: BIF Act 2017 — Chapter 4
# Section 132 — Authority of court for security
(1) This section applies if the holder of a security for a contract—
(a) is retaining the security under section130(2)(a) or
131(2)(a) for a subcontractor’s charge; or
(b) has paid an amount for the security into court under
section 131(2)(b) for a subcontractor’s charge.
(2) The court may make the order it considers appropriate for
enforcing the subcontractor’s charge over the security,
including an order for realising the security.
(3) However, the court may make an order for realising the
security only if the holder of the security is no longer entitled,
under any contract or other arrangement about the security,
including under the security itself, to use the security—
(a) for securing the performance of the contract; or
(b) in some other way provided for in the contract.
(4) Without limiting the orders the court may make under
subsection(2), the court may order the holder of the security
to produce the security to the court.
(5) A precondition or expiry provision for the security is of no
effect to the extent that the provision purports to stop the
realisation of a security under subsection(2).

(6) In this section—
expiry provision, for a security, means a provision of a
contract or another arrangement about the security, including
a provision of the security itself, under which the security
stops, wholly or partly, having effect.
precondition provision, for a security, means a provision of a
contract or another arrangement about the security, including
a provision of the security itself, stating the circumstances that
are to apply before the holder of the security may exercise an
entitlement to use the security for securing the performance of
a contract.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_128.txt`
```
# Annotated BIF Act source — Section 128
# Chapter: CHAPTER 4 – Subcontractors’ charges
# Section title: Contractor given copy of a notice of claim must respond
# DOCX paragraphs: 4628-4639

[2 Com-BIFSOPA Heading 1] SECTION 128 – Contractor given copy of a notice of claim must respond 
[1 BIFSOPA Heading] 128    Contractor given copy of a notice of claim must respond 
[1.3 BIFSOPA level 1 (CDI)] This section applies if a subcontractor gives a contractor a copy of a notice of claim.
[1.3 BIFSOPA level 1 (CDI)] The contractor must give both of the following persons a written response to the claim within 10 business days after the contractor is given the copy of the notice of claim, unless the contractor has a reasonable excuse— 
[1.4 BIFSOPA level 2 (CDI)] the person given the notice of claim;
[1.4 BIFSOPA level 2 (CDI)] the subcontractor.
[1.1 BIFSOPA Body Text] Maximum penalty—20 penalty units.
[1.3 BIFSOPA level 1 (CDI)] The response to the claim must be made in the approved form and—
[1.4 BIFSOPA level 2 (CDI)] accept liability to pay the amount claimed; or
[1.4 BIFSOPA level 2 (CDI)] accept liability to pay an amount stated in the response, but otherwise dispute the claim; or
[1.4 BIFSOPA level 2 (CDI)] dispute the claim.

```

### `annotated/section_132.txt`
```
# Annotated BIF Act source — Section 132
# Chapter: CHAPTER 4 – Subcontractors’ charges
# Section title: Authority of court for security
# DOCX paragraphs: 4716-4730

[2 Com-BIFSOPA Heading 1] SECTION 132 – Authority of court for security 
[1 BIFSOPA Heading] 132    Authority of court for security 
[1.3 BIFSOPA level 1 (CDI)] This section applies if the holder of a security for a contract—
[1.4 BIFSOPA level 2 (CDI)] is retaining the security under section 130(2)(a) or 131(2)(a) for a subcontractor’s charge; or 
[1.4 BIFSOPA level 2 (CDI)] has paid an amount for the security into court under section 131(2)(b) for a subcontractor’s charge. 
[1.3 BIFSOPA level 1 (CDI)] The court may make the order it considers appropriate for enforcing the subcontractor’s charge over the security, including an order for realising the security. 
[1.3 BIFSOPA level 1 (CDI)] However, the court may make an order for realising the security only if the holder of the security is no longer entitled, under any contract or other arrangement about the security, including under the security itself, to use the security—
[1.4 BIFSOPA level 2 (CDI)]  for securing the performance of the contract; or 
[1.4 BIFSOPA level 2 (CDI)]  in some other way provided for in the contract.
[1.3 BIFSOPA level 1 (CDI)] Without limiting the orders the court may make under subsection (2), the court may order the holder of the security to produce the security to the court. 
[1.3 BIFSOPA level 1 (CDI)] A precondition or expiry provision for the security is of no effect to the extent that the provision purports to stop the realisation of a security under subsection (2).
[1.3 BIFSOPA level 1 (CDI)] In this section—
[1.1 BIFSOPA Body Text] expiry provision, for a security, means a provision of a contract or another arrangement about the security, including a provision of the security itself, under which the security stops, wholly or partly, having effect. 
[1.1 BIFSOPA Body Text] precondition provision, for a security, means a provision of a contract or another arrangement about the security, including a provision of the security itself, stating the circumstances that are to apply before the holder of the security may exercise an entitlement to use the security for securing the performance of a contract.

```
