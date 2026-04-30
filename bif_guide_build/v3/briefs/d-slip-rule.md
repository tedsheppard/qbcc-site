# Element-page brief — d-slip-rule
**Title:** Slip-rule corrections
**Breadcrumb:** Requirements of an adjudicator's decision
**Anchor id:** `d-slip-rule`
**Output file:** `bif_guide_build/v3/pages/page_d-slip-rule.html`

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

### `statute/chapter_3/section_089.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 89 — Adjudicator may correct clerical mistakes etc.
(1) This section applies if the adjudicator’s decision includes—
(a) a clerical mistake; or
(b) an error arising from an accidental slip or omission; or
(c) a material miscalculation of figures or a material
mistake in the description of a person, thing or matter
mentioned in the decision; or
(d) a defect of form.
Page 117 Current as at 27 April 2025

(2) The adjudicator may, on the adjudicator’s own initiative or on
the application of the claimant or respondent, correct the
decision.
(3) The adjudicator may, if requested by the registrar, correct the
decision.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_089.txt`
```
# Annotated BIF Act source — Section 89
# Chapter: CHAPTER 3 – Progress payments
# Section title: Adjudicator may correct clerical mistakes etc.
# DOCX paragraphs: 3569-3615

[2 Com-BIFSOPA Heading 1] SECTION 89 – Adjudicator may correct clerical mistakes etc. 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 89    Adjudicator may correct clerical mistakes etc. 
[1.3 BIFSOPA level 1 (CDI)] This section applies if the adjudicator’s decision includes—
[1.4 BIFSOPA level 2 (CDI)] a clerical mistake; or
[1.4 BIFSOPA level 2 (CDI)] an error arising from an accidental slip or omission; or
[1.4 BIFSOPA level 2 (CDI)] a material miscalculation of figures or a material mistake in the description of a person, thing or matter mentioned in the decision; or 
[1.4 BIFSOPA level 2 (CDI)] a defect of form. 
[1.3 BIFSOPA level 1 (CDI)] The adjudicator may, on the adjudicator’s own initiative or on the application of the claimant or respondent, correct the decision.
[1.3 BIFSOPA level 1 (CDI)] The adjudicator may, if requested by the registrar, correct the decision.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 89.1    Section 89(1)(a) – ‘mistake’
[2.4 Com-BIFSOPA CDI Normal Body Text] The legislature recognises that mistakes can be corrected upon a final hearing. McDougall J in Musico v Davenport [2003] NSWSC 977 said:
[2.5 Com-BIFSOPA Normal Body Quote] It is, I think, clear that a mistake of the kind referred to in s22 (5) (s28 BCIPA) would not vitiate a determination made by an adjudicator. It is equally clear that the power given by s22 (5) (s28 Qld) is permissive rather than mandatory. That is no doubt because the legislature recognised that any mistakes could be corrected upon a final hearing.     
[2.4 Com-BIFSOPA CDI Normal Body Text] The adjudicator has power to correct the determination, on his own initiative or on the application of either party. This power to correct the determination remains even if there is a significant delay. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In McNab Developments (Qld) Pty Ltd v MAK Construction Services Pty Ltd & Ors [2014] QCA 232, Jackson J said that the presence of section 28 of BCIPA (equivalent to section 89 the Act) reflects the legislature’s view that the accidental or erroneous omission to consider part of a claim does not invalidate the adjudicator’s decision for jurisdictional error.
[2.4 Com-BIFSOPA CDI Normal Body Text] The adjudicator’s decision is one which may be corrected. As Jackson J said:
[2.5 Com-BIFSOPA Normal Body Quote] Applying the Project Blue Sky v Australian Broadcasting Authority [1998] HCA 28 approach, in my view, it follows that non-compliance with the s26(2) requirement to consider the payment schedule and submissions and relevant documentation which comprises an accidental slip or omission does not invalidate the adjudicator’s decision for jurisdictional error. An administrative decision which is invalid for jurisdictional error may in law amount to no decision at all. However, since s 28 (2) preserves the operation of an adjudicator’s decision to be corrected, there is no other reason, in my view, why the decision should be treated as invalid and of no effect until that power is exercised. It may be that none of the relevant parties considers that a particular error is important enough to warrant correction.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Thiess Pty Ltd v Warren Brothers Earthmoving Pty Ltd & Ors [2012] QSC 373, Lyons J held that where an error is not in the nature of an ‘accidental slip up’ it cannot be corrected under section 28 of BCIPA.
[2.4 Com-BIFSOPA CDI Normal Body Text] In CPB Contractors Pty Limited v Heyday5 Pty Limited [2020] NSWSC 1625, Hammerschlag J upheld the validity of an adjudication determination on the basis that: 
[2.4 Com-BIFSOPA CDI Normal Body Text] the adjudicator’s error in the first line of its adjudication decision was a typographical error and therefore did not justify intervention by the courts; and
[2.4 Com-BIFSOPA CDI Normal Body Text] the adjudicator was not under any obligation to tell the respondent how it proposed to determine whether the claimant was entitled to variations.
[2.4 Com-BIFSOPA CDI Normal Body Text] In this case, the adjudicator determined that the respondent’s directions, namely, enforcing compliance with recent Work Health and Safety (WHS) legislation whilst undertaking construction work, constituted a variation under the contract. The respondent argued that it was denied procedural fairness because it was deprived the opportunity of submitting that it did not matter that the safety progress statements did not comply with the new WHS legislation. However, Hammerschlag J held that the respondent had every opportunity to raise submissions in relation to the variation claim as part of its adjudication response. As a result, the court held that the adjudicator did not deny the respondent procedural fairness by finding against the respondent.  
[2.2 Com-BIFSOPA Heading 3] 89.2    Section 89(1)(c) – ‘Miscalculation’
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been held that a correction cannot be described as the correction of a ‘miscalculation’ where the ‘correction’ would require the adjudicator ‘to adopt a completely different method of making the calculations of the amounts’ or to ‘apply a completely different chain of reasoning and calculation…to reach results which are quite different from those calculated’ under the original decision: Uniting Church in Australia Property Trust (Qld) v Davenport & Anor [2009] QSC 134, [37] (Daubney J); Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd & Ors [2012] QSC 373, [69] (Ann Lyons J).
[2.4 Com-BIFSOPA CDI Normal Body Text] Daubney J observed that Thiess Pty Ltd v Warren Brothers Earthmoving Pty Ltd & Ors [2012] QSC 373 is analogous to the decision of Uniting Church in Australia Property Trust (Qld) v Davenport & Anor [2009] QSC 134 in that where an adjudicator has a complete change of reasoning and adopts a different method of making calculations; that cannot be a ‘miscalculation.’ 
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Uniting Church in Australia Property Trust (Qld) v Davenport & Anor [2009] QSC 134.
[2.6 Com-BIFSOPA bullet] Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd & Ors [2012] QSC 373.
[2.2 Com-BIFSOPA Heading 3] 89.3    Section 28(2) – ‘correct the decision’
[2.4 Com-BIFSOPA CDI Normal Body Text] It is a matter for the adjudicator whether any correction under section 89 of the Act is to be made. If an error exists and the adjudicator fails to correct a mistake, or declines there is a mistake, an error within jurisdiction will arise.
[2.4 Com-BIFSOPA CDI Normal Body Text] On consideration of the NSW equivalent under section 89 of the Act, in Musico v Davenport [2003] NSWSC 977 McDougall J said:
[2.5 Com-BIFSOPA Normal Body Quote] It might be thought that, in the ordinary case, a proper exercise of the discretion conferred by s22 (5) (s28 BCIPA) would favour the correction of mistakes of the kind referred to therein. However, it is a matter for the adjudicator whether or not any such correction is to be made. If an adjudicator declines to make any correction – for example, because he or she thinks there is no mistake – then the error (if any) will be an error within jurisdiction. This is really another way of saying that, under s22 (5) (s28 BCIPA) a party to an adjudication has the right to request the adjudicator to consider the exercise of power under s22 (5) (s28 BCIPA), not the right to have the power exercised in a particular way.  
[2.4 Com-BIFSOPA CDI Normal Body Text] In Cawood v Infraworth Pty Ltd [1990] 2 Qd R 114, Macrossan CJ said at 122:
[2.5 Com-BIFSOPA Normal Body Quote] Inadvertence, as distinguished from an error or mistake resulting from deliberate decision, is the basis of the jurisdiction to correct under the slip rule.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Uniting Church in Australia Property Trust (Qld) v Davenport & Anor [2009] QSC 134, Daubney J held:
[2.5 Com-BIFSOPA Normal Body Quote] When one looks at s 28 of the BCIPA, however, the only one of the discrete elements referred to in s 28(1) which imports the notion of inadvertence is that mentioned in s 28(1)(b), namely ‘an error arising from an accidental slip or omission. It was not suggested that the mistake which the adjudicator would seek to correct in each decision was a ‘clerical mistake’; on the common understanding of that term, it clearly was not, and is not sought to be painted as such. Nor were the adjudicator's mistakes suggested to be, nor could they sensibly be seen to be, defects of form.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Hansen Yuncken Pty Ltd & Anor v Yuanda Australia Pty Ltd & Anor [2018] SASC 158, Lovell J held that if an adjudicator declines to exercise their discretion under the “slip rule” pursuant to section 22(5) of the ACT Act (section 28 of the Qld Act), that decision will be within the adjudicator’s jurisdiction, and thus not reviewable for jurisdictional error.
[2.2 Com-BIFSOPA Heading 3] 89.4    Rule 388 of the UCPR
[2.4 Com-BIFSOPA CDI Normal Body Text] Rule 388 of the UCPR provides a power to correct a clerical mistake in an order or certificate of the court or an error in a record of an order or a certificate of the court if the mistake or error resulted from an accidental slip or omission.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Uniting Church in Australia Property Trust (Qld) v Davenport & Anor [2009] QSC 134, Daubney J rejected an analogy between section 28 of BCIPA (equivalent to section 89 of the Act) and rule 388 of the UCPR, considering such an analogy to be ‘inapt’. His Honour said, in the context of BCIPA:
[2.5 Com-BIFSOPA Normal Body Quote] Section 28(1) prescribes, in disjunctive terms, four discrete circumstances, any one of which may found an exercise of the adjudicator’s discretion under s 28(2). Rule 388(1), on the other hand, contains two subparagraphs which must be read conjunctively such that, to the extent that there is similarity in wording between Rule 388 and s 28, the slip rule applies if “there is a clerical mistake in an order ... of the Court ... and ... the mistake ... resulted from an accidental slip or omission.” In Cawood v Infraworth Pty Ltd [1990] 2 Qd R 114, Macrossan CJ, with whom Kelly SPJ agreed, said at 122:
[2.5 Com-BIFSOPA Normal Body Quote] “Inadvertence, as distinguished from an error or mistake resulting from deliberate decision, is the basis of the jurisdiction to correct under the slip rule.”
[2.5 Com-BIFSOPA Normal Body Quote] When one looks at s 28 of the BCIPA, however, the only one of the discrete elements referred to in s 28(1) which imports the notion of inadvertence is that mentioned in s 28(1)(b), namely “an error arising from an accidental slip or omission”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Theiss Pty Ltd v Warren Brothers Earthmoving Pty Ltd & Ors [2012] QSC 373, Ann Lyons J distinguished Queensland Pork Pty Ltd v Lott [2003] QCA 271, a case which arose in in relation to rule 388 of the Supreme Court Rules 1999 (Qld) and relied on by the claimant, on the grounds that, in Theiss, something more than a different final figure needs to be inserted.

```
