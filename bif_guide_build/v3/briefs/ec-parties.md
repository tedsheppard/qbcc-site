# Element-page brief — ec-parties
**Title:** Necessary parties / joinder
**Breadcrumb:** Enforcing a subcontractor's charge
**Anchor id:** `ec-parties`
**Output file:** `bif_guide_build/v3/pages/page_ec-parties.html`

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

### `statute/chapter_4/section_131.txt`
```
# Source: BIF Act 2017 — Chapter 4
# Section 131 — Use of security for benefit of subcontractor if contractor
does not accept liability for all claims
(1) This section applies if—
(a) a subcontractor gives a notice of claim to a person under
section 122 because the person is obliged to pay money
to a contractor under a contract; and
(b) the subcontractor gives a copy of the notice of claim to
the contractor under section 123; and
(c) in the contractor’s response to the claim under
section 128, the contractor disputes the claim (does not
accept liability to pay the amount claimed); and
(d) the unsatisfied amount for the contract is more than the
retained amount for the contract.
(2) The holder of a security for the contract must—
(a) retain the security until the court in which the
subcontractor’s claim is heard makes an order under
section 132 about enforcing the subcontractor’s charge
over the security; or
(b) instead of retaining the security—
(i) if the security is held as an amount of money—pay
the amount, up to the difference amount for the
contract, into court; or
(ii) if the security is not held as an amount of money
but may be converted into an amount of

money—convert the security, wholly or partly, into
an amount of money and pay the amount, up to the
difference amount for the contract, into court.
(3) If the holder of the security does not comply with
subsection(2), the holder is personally liable to pay the
subcontractor the amount of the subcontractor’s claim to the
extent that the security would have been capable under this
chapter, if the holder had complied with the subsection, of
satisfying the claim.
(4) A payment of an amount under subsection(2)(b) discharges
the holder of the security of all further liability for the amount
paid and of the costs of any proceeding in relation to the
amount paid.
(5) Subsections (2) and (3) do not stop the holder of the security
from exercising an entitlement to use the security for securing
the performance of the contract, including by keeping control
of the security until the security would be required to be
surrendered, wholly or partly, if this section did not apply.
(6) A provision of the contract, or of another arrangement, about
the security, including a provision of the security itself, and
including a provision providing for the surrender, wholly or
partly, of the security, is of no effect to the extent it purports—
(a) to stop the holder of the security from complying with
subsection(2); or
(b) to operate to the detriment of a person if the holder
complies with subsection(2).
(7) An amount paid into court under subsection(2)(b) may be
paid out only under an order of the court.
(8) In this section—
difference amount, for a contract, means the amount by
which the unsatisfied amount for the contract is more than the
retained amount for the contract.
retained amount, for a contract, means the total of—
(a) all amounts a person is retaining under section 126(2)
for the contract; and
Page 161 Current as at 27 April 2025

(b) all amounts a person has paid into court under
section 126(4) for the contract; and
(c) all amounts the holder of a security for the contract has
paid into court under subsection(2)(b) in relation to the
security.
unsatisfied amount, for a contract, means the total of all
amounts of claims of charge for the contract for which a
notice of claim has been given, other than amounts that have
been—
(a) satisfied by payment under section129(2) or (3); or
(b) the subject of a notice of claim that has been withdrawn.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_131.txt`
```
# Annotated BIF Act source — Section 131
# Chapter: CHAPTER 4 – Subcontractors’ charges
# Section title: Use of security for benefits of subcontractor if contractor does not accept liability for all claims
# DOCX paragraphs: 4688-4715

[2 Com-BIFSOPA Heading 1] SECTION 131 – Use of security for benefits of subcontractor if contractor does not accept liability for all claims 
[1 BIFSOPA Heading] 131    Use of security for benefits of subcontractor if contractor does not accept liability for all claims 
[1.3 BIFSOPA level 1 (CDI)] This section applies if—
[1.4 BIFSOPA level 2 (CDI)] a subcontractor gives a notice of claim to a person under section 122 because the person is obliged to pay money to a contractor under a contract; and 
[1.4 BIFSOPA level 2 (CDI)] the subcontractor gives a copy of the notice of claim to the contractor under section 123; and
[1.4 BIFSOPA level 2 (CDI)] in the contractor’s response to the claim under section 128, the contractor disputes the claim (does not accept liability to pay the amount claimed); and
[1.4 BIFSOPA level 2 (CDI)] the unsatisfied amount for the contract is more than the retained amount for the contract. 
[1.3 BIFSOPA level 1 (CDI)] The holder of a security for the contract must—
[1.4 BIFSOPA level 2 (CDI)] retain the security until the court in which the subcontractor’s claim is heard makes an order under section 132 about enforcing the subcontractor’s charge over the security; or 
[1.4 BIFSOPA level 2 (CDI)] instead of retaining the security— 
[1.5 BIFSOPA level 3 (CDI)] if the security is held as an amount of money—pay the amount, up to the difference amount for the contract, into court; or
[1.5 BIFSOPA level 3 (CDI)] if the security is not held as an amount of money but may be converted into an amount of money— convert the security, wholly or partly, into an amount of money and pay the amount, up to the difference amount for the contract, into court. 
[1.3 BIFSOPA level 1 (CDI)] If the holder of the security does not comply with subsection (2), the holder is personally liable to pay the subcontractor the amount of the subcontractor’s claim to the extent that the security would have been capable under this chapter, if the holder had complied with the subsection, of satisfying the claim. 
[1.3 BIFSOPA level 1 (CDI)] A payment of an amount under subsection (2)(b) discharges the holder of the security of all further liability for the amount paid and of the costs of any proceeding in relation to the amount paid.
[1.3 BIFSOPA level 1 (CDI)] Subsections (2) and (3) do not stop the holder of the security from exercising an entitlement to use the security for securing the performance of the contract, including by keeping control of the security until the security would be required to be surrendered, wholly or partly, if this section did not apply. 
[1.3 BIFSOPA level 1 (CDI)] A provision of the contract, or of another arrangement, about the security, including a provision of the security itself, and including a provision providing for the surrender, wholly or partly, of the security, is of no effect to the extent it purports— 
[1.4 BIFSOPA level 2 (CDI)] to stop the holder of the security from complying with subsection (2); or 
[1.4 BIFSOPA level 2 (CDI)] to operate to the detriment of a person if the holder complies with subsection (2). 
[1.3 BIFSOPA level 1 (CDI)] An amount paid into court under subsection (2)(b) may be paid out only under an order of the court. 
[1.3 BIFSOPA level 1 (CDI)] In this section— 
[1.1 BIFSOPA Body Text] difference amount, for a contract, means the amount by which the unsatisfied amount for the contract is more than the retained amount for the contract. 
[1.1 BIFSOPA Body Text] retained amount, for a contract, means the total of— 
[1.4 BIFSOPA level 2 (CDI)] all amounts a person is retaining under section 126(2) for the contract; and
[1.4 BIFSOPA level 2 (CDI)] all amounts a person has paid into court under section 126(4) for the contract; and
[1.4 BIFSOPA level 2 (CDI)] all amounts the holder of a security for the contract has paid into court under subsection (2)(b) in relation to the security.
[1.1 BIFSOPA Body Text] unsatisfied amount, for a contract, means the total of all amounts of claims of charge for the contract for which a notice of claim has been given, other than amounts that have been—
[1.4 BIFSOPA level 2 (CDI)] satisfied by payment under section 129(2) or (3); or 
[1.4 BIFSOPA level 2 (CDI)] the subject of a notice of claim that has been withdrawn. subsection (2)(b) in relation to the security.

```
