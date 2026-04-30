# Element-page brief — sus-when-begin
**Title:** When suspension may begin
**Breadcrumb:** Statutory suspension of works
**Anchor id:** `sus-when-begin`
**Output file:** `bif_guide_build/v3/pages/page_sus-when-begin.html`

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

### `statute/chapter_3/section_098.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 98 — Claimant’s right to suspend work
(1) A claimant may suspend carrying out construction work, or
supplying related goods and services, under a construction
contract if at least 2 business days have passed since the
claimant gave notice of intention to do so to the respondent
under section 78 or 92.
(2) The right conferred under subsection(1) exists until the day
on which the claimant receives payment from the respondent
of the amount mentioned in section78(1) or 92(1), and

continues for another 3 business days immediately following
that day.
(3) If, in exercising the right to suspend carrying out construction
work or supplying related goods and services under a
construction contract, the claimant incurs a loss or expense
because the respondent removes any part of the work or
supply from the contract, the respondent is liable to pay the
claimant the amount of the loss or expense.
(4) A claimant who suspends carrying out construction work, or
supplying related goods and services under a construction
contract under subsection(1) is not liable for any loss or
damage suffered by the respondent, or by any person claiming
through the respondent, because of the claimant not carrying
out that work or not supplying those goods and services,
during the suspension.

recovery

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_098.txt`
```
# Annotated BIF Act source — Section 98
# Chapter: CHAPTER 3 – Progress payments
# Section title: Claimaint’s right to suspend work
# DOCX paragraphs: 3833-3849

[2 Com-BIFSOPA Heading 1] SECTION 98 – Claimaint’s right to suspend work 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 98    Claimant’s right to suspend work 
[1.3 BIFSOPA level 1 (CDI)] A claimant may suspend carrying out construction work, or supplying related goods and services, under a construction contract if at least 2 business days have passed since the claimant gave notice of intention to do so to the respondent under section 78 or 92. 
[1.3 BIFSOPA level 1 (CDI)] The right conferred under subsection (1) exists until the day on which the claimant receives payment from the respondent of the amount mentioned in section 78(1) or 92(1), and continues for another 3 business days immediately following that day. 
[1.3 BIFSOPA level 1 (CDI)] If, in exercising the right to suspend carrying out construction work or supplying related goods and services under a construction contract, the claimant incurs a loss or expense because the respondent removes any part of the work or supply from the contract, the respondent is liable to pay the claimant the amount of the loss or expense.
[1.3 BIFSOPA level 1 (CDI)] (4) A claimant who suspends carrying out construction work, or supplying related goods and services under a construction contract under subsection (1) is not liable for any loss or damage suffered by the respondent, or by any person claiming through the respondent, because of the claimant not carrying out that work or not supplying those goods and services, during the suspension.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 98.1    Section 33(2) – payment
[2.4 Com-BIFSOPA CDI Normal Body Text] In Beckhaus Civil Pty Ltd v Brewarrina Shire Council No. 2 [2004] NSWSC 1160, Master Macready defined the verb ‘pay’ to mean:
[2.5 Com-BIFSOPA Normal Body Quote] To give (a person) what is due in discharge of a debt, or as a return for services done, or goods received, or in compensation for injury done; to remunerate, recompense.” In turn, ‘payment’ is defined as “[t]he action, or an act, of paying.” From this it can readily be concluded that the touchstone of the ordinary and natural meaning of the expressions ‘to pay’ and ‘receives payment’ in the present context is the act of a debtor transferring by way of satisfaction whatever is owed by him or her to a creditor.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] Thus an analysis of both the general and specific purposes of the Act permits the court to identify an alternative construction to the ordinary meaning of ‘pay’ and ‘payment.’ That is that a claimant will not have ‘received payment’ pursuant to s 27(2), and that a respondent will still have failed ‘to pay’ pursuant to s 15(1)(b), in circumstances where the claimant is unable to obtain the benefit of that payment due to conditions, such as the provision of a secured bank guarantee, placed on its receipt. The adoption of such a construction does not constitute a ‘re-writing’ of ss 15 and 27 as proscribed by the dictum of Dawson J above, but is rather a departure from the ordinary meaning of ‘payment’ in circumstances where the context and purpose of its use demands such a modification.
[2.2 Com-BIFSOPA Heading 3] 98.2    Claimant’s right to suspend work
[2.4 Com-BIFSOPA CDI Normal Body Text] In On Forbes Developments Pty Ltd v Chase Building Group (Canberra) Pty Ltd [2020] ACTSC 163, Crowe AJ held, following an adjudication determination made in favour of a claimant, that a claimant’s right to suspend work continues until a respondent gives an unequivocal acknowledgement of the discharge of the adjudication amount. In that case, the claimant exercised its right to suspended work pursuant to section 26(2)(b) of the ACT Act (equivalent to 98(1) of BIFSOPA) following the respondent’s failure to pay an adjudicated amount. A subsequent expert determination under the construction contract while the proceedings were on foot concluded that the claimant owed the respondent more than the adjudicated amount. The respondent argued the claimant’s right to suspend work existed only until the date of expert determination. Crowe AJ held that the claimant’s right to suspend work continued until either the outstanding amount was set aside, or there was an unequivocal communication to the claimant which acknowledged discharge of the payment claim, adjudicated amount, or in this case, provisional judgment debt. On the facts, constructive receipt of the adjudication amount being set aside was held to have been after consent orders were made by Crowe AJ on the date of the hearing, after which the claimant’s right to suspend work ended 3 business days after that date.

```
