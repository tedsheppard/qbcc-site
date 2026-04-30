# Element-page brief — ar-copy-claimant
**Title:** Copy to the claimant within 2 BD
**Breadcrumb:** Requirements of an adjudication response
**Anchor id:** `ar-copy-claimant`
**Output file:** `bif_guide_build/v3/pages/page_ar-copy-claimant.html`

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

### `statute/chapter_3/section_083.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 83 — Time for making adjudication response
(1) If responding to a standard payment claim, the respondent
must give the adjudicator the adjudication response within the
later of the following periods to end—
(a) 10 business days after receiving the documents
mentioned in section79(4);
(b) 7 business days after receiving notice of the
adjudicator’s acceptance of the adjudication application.
(2) If responding to a complex payment claim, the respondent
must give the adjudicator the adjudication response within the
later of the following to end—
(a) 15 business days after receiving the documents
mentioned in section79(4);
(b) 12 business days after receiving notice of the
adjudicator’s acceptance of the adjudication application.
Page 111 Current as at 27 April 2025

(3) However, if responding to a complex payment claim, the
respondent may apply to the adjudicator for an extension of
time, of up to 15 additional business days, to give the
adjudication response to the adjudicator.
(4) The application must—
(a) be in writing; and
(b) be made within the later of the following periods to
end—
(i) 5 business days after receiving the documents
mentioned in section79(4);
(ii) 2 business days after receiving notice of the
adjudicator’s acceptance of the adjudication
application; and
(c) include the reasons for requiring the extension of time.
(5) If the application is granted, the respondent may give the
adjudicator the adjudication response no later than the end of
the extension of time granted by the adjudicator.
(6) If the respondent gives the adjudicator an adjudication
response under this section, the respondent must give a copy
of the response to the claimant not more than 2 business days
after giving the response to the adjudicator.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_083.txt`
```
# Annotated BIF Act source — Section 83
# Chapter: CHAPTER 3 – Progress payments
# Section title: Time for making adjudication response
# DOCX paragraphs: 2569-2595

[2 Com-BIFSOPA Heading 1] SECTION 83 – Time for making adjudication response 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 83    Time for making adjudication response 
[1.3 BIFSOPA level 1 (CDI)] If responding to a standard payment claim, the respondent must give the adjudicator the adjudication response within the later of the following periods to end—
[1.4 BIFSOPA level 2 (CDI)] 10 business days after receiving the documents mentioned in section 79(4);
[1.4 BIFSOPA level 2 (CDI)] 7 business days after receiving notice of the adjudicator’s acceptance of the adjudication application. 
[1.3 BIFSOPA level 1 (CDI)] If responding to a complex payment claim, the respondent must give the adjudicator the adjudication response within the later of the following to end—
[1.4 BIFSOPA level 2 (CDI)] 15 business days after receiving the documents mentioned in section 79(4); 
[1.4 BIFSOPA level 2 (CDI)] 12 business days after receiving notice of the adjudicator’s acceptance of the adjudication application.
[1.3 BIFSOPA level 1 (CDI)] However, if responding to a complex payment claim, the respondent may apply to the adjudicator for an extension of time, of up to 15 additional business days, to give the adjudication response to the adjudicator. 
[1.3 BIFSOPA level 1 (CDI)] The application must— 
[1.4 BIFSOPA level 2 (CDI)] be in writing; and 
[1.4 BIFSOPA level 2 (CDI)] be made within the later of the following periods to end—
[1.5 BIFSOPA level 3 (CDI)] 5 business days after receiving the documents mentioned in section 79(4);
[1.5 BIFSOPA level 3 (CDI)] 2 business days after receiving notice of the adjudicator’s acceptance of the adjudication application; and
[1.4 BIFSOPA level 2 (CDI)] include the reasons for requiring the extension of time. 
[1.3 BIFSOPA level 1 (CDI)] If the application is granted, the respondent may give the adjudicator the adjudication response no later than the end of the extension of time granted by the adjudicator.
[1.3 BIFSOPA level 1 (CDI)] If the respondent gives the adjudicator an adjudication response under this section, the respondent must give a copy of the response to the claimant not more than 2 business days after giving the response to the adjudicator.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 83.1    After ‘receiving’ a copy
[2.4 Com-BIFSOPA CDI Normal Body Text] Timeframes for an adjudication response under this section are defined by reference to the date on which the adjudication application was ‘received’. There is a distinction between ‘service’ and ‘receipt’ under the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to section 102 for further commentary on this distinction, and the meaning of ‘receipt’ under the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] CMF Projects Pty Ltd v Masic Pty Ltd & Ors [2014] QSC 209. An adjudicator’s notice of acceptance was sent not to the respondent’s registered office, but to a post office box of the respondent. Daubney J noted the distinction between ‘service’ and ‘receipt’ of documents under BCIPA and held that the adjudicator had erred in equating deposit of the notice into the post office box with receipt of the notice.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Rise Constructions v El-Hajj [2019] VSC 818, Digby J dismissed an application seeking to have an adjudication determination quashed for jurisdictional error and want of procedural fairness, by reason of the Adjudicator’s alleged failure to consider an Adjudication Response. In that case, Digby J concluded that the Adjudicator’s decision that a Notice of Acceptance was served on a particular date was made within jurisdiction and based on sufficient materials before the Adjudicator. Accordingly, the Adjudicator validly decided not to consider the Adjudication Response which, consequential to the finding of service of the Notice of Acceptance, was made outside the time required by section 21(1)(b) of the Vic Act (equivalent to section 83(1)(b) of the Act).

```
