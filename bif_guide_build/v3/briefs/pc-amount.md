# Element-page brief — pc-amount
**Title:** State the claimed amount
**Breadcrumb:** Requirements of a payment claim
**Anchor id:** `pc-amount`
**Output file:** `bif_guide_build/v3/pages/page_pc-amount.html`

## Statute scope note
Show only s 68(1)(b) and the introductory chapeau.

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

### `statute/chapter_3/section_068.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 68 — Meaning of payment claim
(1) A payment claim, for a progress payment, is a written
document that—
(a) identifies the construction work or related goods and
services to which the progress payment relates; and
(b) states the amount (the claimed amount) of the progress
payment that the claimant claims is payable by the
respondent; and
(c) requests payment of the claimed amount; and
(d) includes the other information prescribed by regulation.
(2) The amount claimed in the payment claim may include an
amount that—
(a) the respondent is liable to pay the claimant under
section 98(3); or
(b) is held under the construction contract by the respondent
and that the claimant claims is due for release.
(3) A written document bearing the word ‘invoice’ is taken to
satisfy subsection(1)(c).

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_068.txt`
```
# Annotated BIF Act source — Section 68
# Chapter: CHAPTER 3 – Progress payments
# Section title: Meaning of payment claim
# DOCX paragraphs: 1347-1565

[2 Com-BIFSOPA Heading 1] SECTION 68 – Meaning of payment claim 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 68    Meaning of payment claim 
[1.3 BIFSOPA level 1 (CDI)] A payment claim, for a progress payment, is a written document that— 
[1.4 BIFSOPA level 2 (CDI)] identifies the construction work or related goods and services to which the progress payment relates; and 
[1.4 BIFSOPA level 2 (CDI)] states the amount (the claimed amount) of the progress payment that the claimant claims is payable by the respondent; and 
[1.4 BIFSOPA level 2 (CDI)] requests payment of the claimed amount; and
[1.4 BIFSOPA level 2 (CDI)] includes the other information prescribed by regulation.
[1.3 BIFSOPA level 1 (CDI)] The amount claimed in the payment claim may include an amount that—
[1.4 BIFSOPA level 2 (CDI)] the respondent is liable to pay the claimant under section 98(3); or 
[1.4 BIFSOPA level 2 (CDI)] is held under the construction contract by the respondent and that the claimant claims is due for release. 
[1.3 BIFSOPA level 1 (CDI)] A written document bearing the word ‘invoice’ is taken to satisfy subsection (1)(c).
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 68.1    Introduction of section 68
[2.4 Com-BIFSOPA CDI Normal Body Text] The definition of what constitutes a valid payment claim under the Act appears in this section, whilst the procedure for making a claim for payment is outlined in section 75 of the Act. 
[2.4 Com-BIFSOPA CDI Normal Body Text] The explanatory note for the Act provides the following explanation for inserting the new provision:
[2.5 Com-BIFSOPA Normal Body Quote] Clause 68 is a new provision defining the term ‘payment claim’ for the purposes of this chapter. The claim must be a written document that (a) identifies the construction work or related goods and services to which the progress claim relates; (b) states the amount of the progress payment that the claimant claims is payable by the respondent; and (c) requests payment of the claimed amount; and (d) includes the other information prescribed by regulation. A request for payment may be, for example, in the form of an invoice that contains a due date.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Act has introduced a new requirement for a payment claim to “request payment of the claimed amount” under section 68(1)(c) of the Act, with section 68(3) providing that “a written document bearing the word ‘invoice’ is taken to satisfy subsection (1)(c)”. The explanatory notes do not explain this amendment, nor is there any indication as to whether the ‘written document’ can bear any other words, akin to ‘invoice’, in order to satisfy the requirement in section 68(1)(c). Further, no other security of payment regime includes this, or a similar requirement. Whilst a valid payment claim under the Act must now “request payment of the claimed amount”, the courts will likely continue to consider the requirements of a valid payment claim “not from an unduly critical viewpoint,” an expression previously adopted by Henry J when considering what constitutes a valid payment claim under BCIPA in Camporeale Holdings Pty Ltd v Mortimer Construction Pty Ltd & Anor [2015] QSC 211.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Act also omits the word ‘must’ and instead defines a valid payment claim as a written document which includes the information set out in section 68.
[2.4 Com-BIFSOPA CDI Normal Body Text] Further, a valid payment claim does not need to state that it is made under the Act.
[2.2 Com-BIFSOPA Heading 3] 68.2    Formal requirements of a payment claim
[2.3 Com-BIFSOPA Heading 4] Mandatory requirements of a payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 68(1) imposes four requirements of a payment claim under the Act, namely, that a payment claim:
[2.4 Com-BIFSOPA CDI Normal Body Text] identifies the construction work or related goods and services to which the progress payment relates; and 
[2.4 Com-BIFSOPA CDI Normal Body Text] states the amount (the claimed amount) of the progress payment that the claimant claims is payable by the respondent; and 
[2.4 Com-BIFSOPA CDI Normal Body Text] requests payment of the claimed amount; and
[2.4 Com-BIFSOPA CDI Normal Body Text] includes the other information prescribed by regulation.
[2.6 Com-BIFSOPA bullet] The issues surrounding the validity of a payment claim are considered in the following section.
[2.3 Com-BIFSOPA Heading 4] Compliance with the Act in the making of a payment claim under the Act
[2.4 Com-BIFSOPA CDI Normal Body Text] Much of the following commentary relates to the previous regime under BCIPA. As a matter of policy, the new regime under the Act adopts a less stringent test for what constitutes a valid payment claim. For example, the Act now deems a document bearing the word ‘invoice’ to satisfy the requirements of section 68(1)(c). The commentary below should be considered in this context.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Camporeale Holdings Pty Ltd v Mortimer Construction Pty Ltd & Anor [2015] QSC 211, Henry J said in the context of the previous regime under BCIPA: 
[2.5 Com-BIFSOPA Normal Body Quote] It is well settled that because of the mandatory consequences of the BCIPA regime the content of a progress payment claim must strictly comply with the requirements of that statutory regime. However, whether a purported claim does satisfy the description of a payment claim is not to be approached from an unduly critical viewpoint and the only concern is whether the content of the purported claim satisfies the statutory description.
[2.4 Com-BIFSOPA CDI Normal Body Text] In the above passage, Henry J suggested that he should not be adopting ‘an unduly critical viewpoint’, citing Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA Pty Ltd) & Ors [2007] QSC 333 at [20] where Chesterman J said:
[2.5 Com-BIFSOPA Normal Body Quote] The Act emphasises speed and informality. Accordingly one should not approach the question whether a document satisfies the description of a payment schedule (or payment claim for that matter) from an unduly critical viewpoint. No particular form is required. One is concerned only with whether the content of the document in question satisfies the statutory description.
[2.4 Com-BIFSOPA CDI Normal Body Text] This passage has been approved by the Court of Appeal as being the correct approach to the construction of payment claims, and consistent with the approaches taken in equivalent jurisdiction: Neumann Contractors Pty Ltd v Traspunt No 5 Pty Ltd [2010] QCA 119; [2011] 2 Qd R 114, 122 [24] (Muir JA, Holmes and Chesterman JJA agreeing).
[2.4 Com-BIFSOPA CDI Normal Body Text] The New South Wales Court of Appeal shared a similar view in Hawkins Construction (Australia) Pty Ltd v Mac’s Industrial Pipework Pty Ltd [2002] NSWCA 136. Davies AJA considered that the equivalent to section 68(1) under the NSW Act:
[2.5 Com-BIFSOPA Normal Body Quote] should not be approached in an unduly technical manner… The terms used by subs (2) of s 13 are well understood words of the English language. They should be given their normal and natural meaning. As the words are used in relation to events occurring in the construction industry, they should be applied in a common sense practical manner.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Watpac Constructions (NSW) Pty Ltd v Charter Hall Funds Management Limited [2017] NSWSC 865, [63] (Palmer J), Mackie Pty Ltd v Counahan & Anor [2013] VSC 694, [46] (Vickery J); Protectavale Pty Ltd v K2K Pty Ltd [2008] FCA 1248, [11] (Finkelstein J); Leighton Contractors Pty Ltd v Campbelltown Catholic Club Limited [2003] NSWSC 1103, [54] (Einstein J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In contrast, F.K. Gardner & Sons Pty Ltd v Dimin Pty Ltd Ltd [2006] QSC 243; [2007] 1 Qd R 10 is often cited to support the proposition that “strict” compliance is necessary in the making of a payment claim (and, more generally, in compliance with the statutory requirements of the Act).
[2.2 Com-BIFSOPA Heading 3] 68.3    Validity of a payment claim
[2.3 Com-BIFSOPA Heading 4] The pleading metaphor
[2.4 Com-BIFSOPA CDI Normal Body Text] In Brookhollow Pty Ltd v R&R Consultants Pty Ltd & Anor [2006] NSWSC 1, Palmer J said (at [44]-[45]):
[2.5 Com-BIFSOPA Normal Body Quote] A payment claim under the Act is, in many respects, like a Statement of Claim in litigation. In pleading a Statement of Claim, the plaintiff sets out only the facts and circumstances required to establish entitlement to the relief sought; the Statement of Claim does not attempt to negative in advance all possible defences to the claim. It is for the defendant to decide which defences to raise; the plaintiff, in a reply, answers only those defences which the defendant has pleaded.
[2.5 Com-BIFSOPA Normal Body Quote] In my opinion, a payment claim under the Act works the same way. If it purports reasonably on its face to state what s.13(2)(a) and (b) require it to state, it will have disclosed the critical elements of the claimant’s claim. It is then for the respondent either to admit the claim or to decide what defences to raise.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, this analogy has been doubted on the basis that the role of a payment claim is not to identify the scope of a dispute: see T & M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QCA 381, [30]-[32]; Clarence Street Pty Ltd v Isis Projects Pty Ltd [2005] NSWCA 391, [31].
[2.4 Com-BIFSOPA CDI Normal Body Text] In Watpac Constructions (NSW) Pty Ltd v Charter Hall Funds Management Limited [2017] NSWSC 865, McDougall J held that, although a disputed payment claim did not indicate what work had been paid for and for what, the payment claim did not fail to comply with s 13(2)(a) of the NSW Act (s 17(2)(c) equivalent) because there was no reasonable ambiguity between the parties. In that case, the first 32 progress claims were not valid payment claims for the purposes of the Act. Consequently, the corresponding schedules were also invalid, as they did not respond to a valid payment claim. Notwithstanding, payment was made. The following four payment claims were the subject of dispute. Charter Hall’s principal submission was that the claims did not identify the construction work to which the claim related, specifically that they did not indicate what work had been paid for and what, for which payment had not been made, was the subject of the claim. McDougall J held that it should have been in reasonable contemplation of both parties as to what monies had previously been paid, and what had not. Therefore, his Honour held that the claim did not fail to comply with the s 17(2)(c) equivalent of the Act. 
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been suggested that the obligation on a claimant to identify the construction work or related goods and services to which the payment claim relates should be considered in the context of a respondent being precluded from raising reasons for withholding payment not contained in the payment schedule (as per section 24(4) of the Act for a “standard payment claim”).
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer also to the commentary under section 70 regarding a claimant’s entitlement to submit a payment claim.
[2.2 Com-BIFSOPA Heading 3] 68.4    Section 68(1)(a)
[2.3 Com-BIFSOPA Heading 4] Identify the construction work or related goods and services: general view
[2.4 Com-BIFSOPA CDI Normal Body Text] A payment claim must provide ‘sufficient identification “to enable the respondent to understand the basis of the claim”’: T & M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QCA 381, [35] (Philippides J, Fraser and White JJA agreeing), quoting Coordinated Construction Co Pty Ltd v Climatech (Canberra) Pty Ltd & Ors [2005] NSWCA 229, [25].In the context of the NSW Act, in Estate Property Holdings Pty Ltd v Barclay Mowlem Construction Ltd [2004] NSWCA 393, Hodgson JA (Mason P and Giles JA agreeing) held that the requirement to identify the construction work or related goods and services:
[2.5 Com-BIFSOPA Normal Body Quote] requires that a payment claim identify the construction work for which payment is claimed in the claim, not merely the construction work as a whole that is being carried out under the relevant construction contract. I think this is indicated by the words “construction work … to which the progress payment relates”; and strongly confirmed by the consideration that, unless a progress claim identified the particular work for which payment was claimed, it would be impossible for a respondent to provide a meaningful payment schedule supported by reasons.
[2.4 Com-BIFSOPA CDI Normal Body Text] The test for the identification of construction work or related goods and services has been held to be in terms of being reasonably comprehensible to the respondent to a payment claim.
[2.4 Com-BIFSOPA CDI Normal Body Text] This was the view of Finkelstein J in Protectavale Pty Ltd v K2K Pty Ltd [2008] FCA 1248 where his Honour said, in the context of the VIC Act:
[2.5 Com-BIFSOPA Normal Body Quote] Nonetheless a payment claim must be sufficiently detailed to enable the principal to understand the basis of the claim.  If a reasonable principal is unable to ascertain with sufficient certainty the work to which the claim relates, he will not be able to provide a meaningful payment schedule.  That is to say, a payment claim must put the principal in a position where he is able to decide whether to accept or reject the claim and, if the principal opts for the latter, to respond appropriately in a payment schedule:  Nepean Engineering Pty Ltd v Total Process Services Pty Ltd (in liq) (2005) 64 NSWLR 462, 477; John Holland Pty Ltd v Cardno MBK (NSW) Pty Ltd [2004] NSWSC 258 at [18]-[21]. That is not an unreasonable price to pay to obtain the benefits of the statute.
[2.4 Com-BIFSOPA CDI Normal Body Text] This view was adopted by White J in Neumann Contractors Pty Ltd v Peet Beachton Syndicate Ltd [2011] 1 Qd R 17; [2009] QSC 376 and followed in Krongold Constructions (Aust) v SR RS Wales [2016] VSC 94 where Vickery J held that a payment claim was invalid as it failed to identify the work to which it relates with sufficient certainty to enable the principal to understand the basis of the claim and provide a considered response to it. Further, his Honour held that despite the principal having familiarity with the work being claimed for, it is still entitled to a payment claim which describes the work being claimed for.
[2.4 Com-BIFSOPA CDI Normal Body Text] In T & M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QCA 381, the Court of Appeal said, after a review of the authorities:
[2.5 Com-BIFSOPA Normal Body Quote] In Nepean Engineering Pty Ltd, Hodgson JA (with whom Ipp JA agreed) in considering the degree of identification required for a payment claim, had regard to the observations in Clarence Street Pty Ltd concerning the different functions of a payment claim and payment schedule and to his statements in Climatech Pty Ltd (at [25]) that what was required was sufficient identification “to enable the respondent to understand the basis of the claim”. His Honour noted Basten JA’s statements in Climatech Pty Ltd (at [42]) that to be valid a claim must be reasonably comprehensible to the other party, and expressed the degree of identification required in terms of whether in all the circumstances, the material in the payment claim was sufficient to convey to the recipient just what was the work for which payment was claimed (at [28]). What was required was that the payment claim purport in a reasonable way to identify the work the subject of the claim, and a payment claim was not a nullity for failure to identify the work unless the failure was patent on its face. The payment claim did not cease to satisfy the requirement concerning identification because it could be subsequently shown that the payment claim was not entirely successful in identifying all of the work.
[2.5 Com-BIFSOPA Normal Body Quote] Santow JA (at [47]-[48]) expressed the view that, in respect of the minimum necessary to satisfy the identification requirement that the payment claim “purport in a reasonable way to identify the work” there must be “sufficient specificity in the payment claim for its recipient actually to be able to identify a ‘payment claim’ for the purpose of determining whether to pay, or to respond by way of a payment schedule indicating the extent of payment, if any.” But having said that, his Honour stated his agreement with what Hodgson JA said in Climatech Pty Ltd that what was required was sufficient identification “to enable the respondent to understand the basis of the claim” and disavowed the notion that there was a legal necessity to include any material directed merely to persuading a respondent to accept a payment claim (at [25]).
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] …The issue for determination was not whether the payment claim explained in every respect the means by which a particular claim item had been calculated, but whether the relevant construction work or related goods and services was sufficiently identified as explained above. That is, whether the payment claim reasonably identified the construction work to which it related such that the basis of the claim was reasonably comprehensible to the applicant.
[2.5 Com-BIFSOPA Normal Body Quote] In Canberra Building and Maintenance Pty Ltd v WNA Construction Pty Ltd [2023] ACTSC 153, the Supreme Court considered whether a payment claim adequately identified construction work pursuant to section 15(2) of the ACT SOP Act (equivalent of section 68(1) BIFSOPA). Curtin JA applied Nepean Engineering Pty Ltd v Total Process Services Pty Ltd (in liq) [2005] NSWCA 409 and held that the payment claim adequately identified the construction on the basis that:
[List Paragraph] the payment claim had to be interpreted in the common practical sense, taking into account the speed with which it had to be prepared [95]; and

[List Paragraph] to require a higher degree of preciseness within the payment claim would frustrate the purposes of the security of payment legislation [96].
[2.4 Com-BIFSOPA CDI Normal Body Text] In Queensland then, the test as set out by the Court of Appeal under the previous regime in BCIPA was whether the payment claim is ‘reasonably comprehensible’ to the respondent.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Wetlock Industries Pty Ltd trading as Wetlock Waterproofing ABN 45 151 807 359 v Body Corporate for City Link [2016] QMC 9, Magistrate Carmody considered the degree of specificity required in a payment claim and held: 
[2.5 Com-BIFSOPA Normal Body Quote] Under s 17(2) a payment claim must identify the work to which it relates with reasonable clarity. Reasonableness is a matter of judgment or a question of fact and degree.
[2.4 Com-BIFSOPA CDI Normal Body Text] Further, his Honour held that when considering if a payment claim satisfied the requirements of section 17(2) of BCIPA:
[2.5 Com-BIFSOPA Normal Body Quote] If major aspects of a payment claim can be critiqued and rejected for including items of work unlicensed, unfinished or reiterations it probably adequately and reasonably identifies the construction work claimed.
[2.4 Com-BIFSOPA CDI Normal Body Text] In A-Tech Australia Pty Ltd v Top Pacific Construction Aust Pty Ltd [2019] NSWSC 404, Parker J, held, in construing section 13(2) of the NSW Act (equivalent of section 68 of the Act), that two invoices served on the respondent were valid payment claims. His Honour declared that although the invoice narratives were very brief, indicating only that the works related to glazing and locks and the relevant levels of the buildings, because the respondents correspondence did not profess any difficulty understanding what the plaintiff was claiming for, the invoices were valid payment claims.
[2.4 Com-BIFSOPA CDI Normal Body Text] In ACP Properties (Townsville) Pty Ltd v Rodrigues Construction Group Pty Ltd & Anor [2021] QSC 45, Bradley J determined that an invoice for “labour fees and supply” sufficiently identified the construction work or related goods and services to which the progress payment related, as required by section 68(1)(a) of the BIF Act, and thus was a valid payment claim, on the basis that: 
[2.4 Com-BIFSOPA CDI Normal Body Text] although the timesheets which were attached to the invoice did not, with precision, identify which parts of the project each employee was working on the total amount claimed was modest in the sense that it enabled the respondent to reasonably comprehend the basis of the claim and form a judgment whether to reject or pay;  
[2.4 Com-BIFSOPA CDI Normal Body Text] contractual obligations, requiring a claimant to reconcile a supplied amount with a Quantity Surveying report, cannot affect the validity of a payment claim that satisfies the requirements under the BIF Act; and 
[2.4 Com-BIFSOPA CDI Normal Body Text] where the value of the building work is so significant (i.e. for $2.7 million), a principal party should ordinarily expect to have to exert itself to monitor, review and consider payment claims against the contract and against other information it has relevant to the progress, sufficiency, completeness and quality of the work.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Piety Constructions Pty Ltd v Megacrane Holdings Pty Ltd [2023] NSWSC 309, Richmond J refused an application to set aside an adjudication determination on the basis that a claimant failed to issue a valid payment claim. In this case, the respondent argued that a letter issued by the claimant that was titled “a letter of demand”, was not a valid payment claim and was instead, a letter of demand. Richmond J disagreed with the respondent and upon a fair reading of the letter as a whole, held that as per the “relatively undemanding” requirements for a valid payment claim contained in s13(2) of the NSW SOP Act (equivalent s 68(1) BIFSOPA), the letter constituted a valid payment claim. The key considerations which Richmond J took into consideration in making his decision, were that the letter stated that it was a payment claim, and that it sufficiently identified the construction work to which it related.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Panel Concepts Pty Ltd v Tomkins Commercial & Industrial Builders Pty Ltd [2021] QDC 322, Porter DCJ declared a payment claim invalid on the basis that the payment claim did not sufficiently identify the relevant construction work as required by section 68(1)(a) of the BIF Act. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In that case, the disputed payment claim described the relevant work as a “claim for completion of tilt panels”. The respondent challenged the validity of the payment claim on the basis that the description of the work claimed did not sufficiently identify the work. The respondent argued the word “completion” was ambiguous, as it was unclear whether the payment claim was for completion of the whole of the subcontracted works or a portion of those works. The claimant argued that the description of work related to the completion of the fabrication of the tilt panels, was sufficiently clear in the context of the “objective facts”, including the previous two payment claims which described the work claimed as “fabrication of tilt panels”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In determining that the payment claim did not sufficiently identify the works claimed, Judge Porter adopted the principles set out KDV Sports Pty Ltd where Brown J found that “providing the percentage of work carried out in total is insufficient to reasonably identify the construction work in respect of the claim”. Accordingly, Porter DCJ found the lack of particularity in the word “completion” meant that the payment claim failed to sufficiently identify the works claimed.
[2.4 Com-BIFSOPA CDI Normal Body Text] In EnerMech Pty Ltd v Acciona Infrastructure Projects Australia Pty Ltd [2024] NSWCA 162, Basten AJA (with whom Meagher JA and Griffiths AJA agreed) explained that it is a matter for the adjudicator to determine whether a payment claim satisfies the requirements set out in section 13(2) of the NSW Act (equivalent to s 68(1) of the Qld Act). In particular, his Honour stated the following (at [19]):
[2.5 Com-BIFSOPA Normal Body Quote] “Section 13(2) provides, in obligatory terms, three matters which a payment claim must contain. However, the respondents accepted that the use of the imperative did not necessarily indicate an essential precondition to validity, and did not do so in s 13(2). These are factors of which an adjudicator will need to be satisfied in the case of an adjudication. (In most cases, these matters will not be in issue)”.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour also observed that the primary judge erred in finding that the Act requires a payment claim (as a pre-condition to its validity) to be a claim for “construction work” (as defined by section 5 of the Act – equivalent to section 65 of the Building Industry Fairness (Security of Payment) Act 2017 (Qld)). According to Basten AJA (at [62]):
[2.5 Com-BIFSOPA Normal Body Quote] “Whether an amount is in fact payable will depend on the proper construction of the contract, identification of the work which has been carried out and a determination as to whether that work has already been paid for. These are matters to be determined by an adjudicator in the case of a dispute; they are not preconditions to the validity of a claim. If the entitlement is claimed to arise “under” a construction contract, as required by s 13(1) of the Act, the soundness of a claim denied in the payment schedule is for the adjudicator. Use of the word “for” to describe the required relationship between the amount and the basis of liability to pay is not a reason to substitute that word for the language of the statute, let alone to create a novel precondition to engagement”.
[2.5 Com-BIFSOPA Normal Body Quote] In Bettar Holdings Pty Ltd v RWC Brookvale Investments Pty Ltd, Cole DCJ held that a Payment Claim did not meet the requirements of section 13(2) of the NSW Act (equivalent to section 68(1) BIFSOPA), as it lacked “precision and particularity” required to sufficiently identify the construction work. Cole DCJ noted that the specific services relating to the construction work must be identified, however, the Payment Claim merely relied upon a general description. In the first part of the Payment Claim, no monetary claim was made. In the later parts, specific services are not identified nor is there any attempt at all to identify services relating to construction works.
[2.5 Com-BIFSOPA Normal Body Quote] In Bridgeman Agencies Pty Ltd v S.E QLD Plumbing & Drainage Pty Ltd [2025] QSC 167, Kelly J referring to section 68(1)(a) of the BIFSOPA said:
[2.5 Com-BIFSOPA Normal Body Quote] “the question whether the relevant construction work is identified sufficiently, cannot be answered by confining the inquiry to the actual document or documents comprising the payment claim.” 
[2.5 Com-BIFSOPA Normal Body Quote] This statement quite clearly dispels narrow arguments that allege, any documentation referred to in a payment claim must be attached to it, in order to validate the claim. Furthermore, his Honour reiterated words of Brown J in KDV Sport Pty Ltd v Muggeridge Constructions Pty Ltd [2019] QSC 178, stating that:
[2.5 Com-BIFSOPA Normal Body Quote] “the particular work [identified by way of percentage valuation] was being claimed from a zero base… the use of the percentage figure in this case was sufficient to identify the basis on which this part of the payment claim was being advanced.”
[2.4 Com-BIFSOPA CDI Normal Body Text] This passage confirms the position that percentage-based claims are not inherently inadequate in identifying construction work. Rather, where the percentage is explained by refence to a benchmark, the description will likely be sufficient in identifying the relevant work.The following section explores a stricter view propounded under this requirement, framed in terms of apprising the parties of the real issues in dispute.
[2.3 Com-BIFSOPA Heading 4] Identify the construction work or related goods and services: a stricter view
[2.4 Com-BIFSOPA CDI Normal Body Text] In Isis Projects v Clarence Street [2004] NSWSC 714, McDougall J approved of the observations of Palmer J in Multiplex Constructions Pty Ltd v Luikens & Anor [2003] NSWSC 1140 at [76]-[78] as being equally applicable to payment claims. On such observations of Palmer J, the identification of construction work or related goods or services under a payment claim must have a degree of precision and particularity “required to a degree reasonably sufficient to apprise the parties of the real issues in the dispute”.
[2.4 Com-BIFSOPA CDI Normal Body Text] This approach was adopted by Daubney J in Baxbex Pty Ltd v Bickle [2009] QSC 194, and followed by Samios DCJ in T & M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QDC 60 at [21].
[2.4 Com-BIFSOPA CDI Normal Body Text] However, the Court of Appeal in T & M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QCA 381 held that the approach adopted by the primary judge was too narrow. The Court said (at [31]-[32], [37]):
[2.5 Com-BIFSOPA Normal Body Quote] In my view there is merit in the respondent’s submissions. In Clarence Street Pty Ltd [2005] NSWCA 391 Mason P (with whom Giles JA and Santow JA agreed) observed (at [30] - [32]) that, in Multiplex Constructions, Palmer J was in fact considering the requirements for a valid payment schedule which has a different function from a payment claim. A payment schedule is required to identify the payment claim to which it relates, indicate the amount of the payment (if any) that is proposed to be made and why payment in full is withheld. The joinder of issue thus achieved sets the parameters for the matters that may be contested if an adjudication ensues. On the other hand, as Mason P pointed out (at [31]), when it is the validity of a payment claim that is in issue, it must be borne in mind that:
[2.5 Com-BIFSOPA Normal Body Quote] “… a ‘payment claim’ is no more than a claim. It must comply with s 13, but (unlike a payment schedule) it is not its function to identify the scope of a dispute. Many claims will not be disputed, but if they are, it is a matter for the respondent to the payment claim to state the extent and reasons for failing to pay the sum withheld.”
[2.5 Com-BIFSOPA Normal Body Quote] It is apparent that, in relying on the statement quoted from Baxbex Pty Ltd, the learned judge overlooked the matters referred to by Mason P and set too high a bar in respect of what was required for the purposes of s 17(2)(a) BCIPA. Moreover, he adopted an approach that was inconsistent with the other line of authority to which he had referred.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] The approach taken in Climatech Pty Ltd and Nepean Engineering Pty Ltd indicates what is required in determining whether there has been sufficient identification for the purpose of s 17(2)(a). In the present case, however, the learned judge applied a more stringent test, by focusing on the dicta quoted from Baxbex Pty Ltd that the degree of identification required for a valid payment claim was such as to apprize the parties of the real issues in dispute.
[2.3 Com-BIFSOPA Heading 4] Identification
[2.4 Com-BIFSOPA CDI Normal Body Text] In Isis Projects v Clarence Street [2004] NSWSC 714, McDougall J set down the following four requirements for the identification of construction work:
[2.5 Com-BIFSOPA Normal Body Quote] In principle, I think, the requirement in s 13(2)(a) that a payment claim must identify the construction work to which the progress payment relates is capable of being satisfied where:
[2.5 Com-BIFSOPA Normal Body Quote] (1) The payment claim gives an item reference which, in the absence of evidence to the contrary, is to be taken as referring to the contractual or other identification of the work;

(2) That reference is supplemented by a single line item description of the work;

(3) Particulars are given of the amount previously completed and claimed and the amount now said to be complete;

(4) There is a summary that pulls all the details together and states the amount claimed.
[2.4 Com-BIFSOPA CDI Normal Body Text] This approach was approved on appeal by Mason P in Clarence Street Pty Ltd v Isis Projects Pty Ltd [2005] NSWCA 391 at [33]-[34].
[2.4 Com-BIFSOPA CDI Normal Body Text] In Coordinated Construction Co Pty Ltd v Climatech (Canberra) Pty Ltd & Ors [2005] NSWCA 229, Hodgson JA said:
[2.5 Com-BIFSOPA Normal Body Quote] In my opinion, the relevant construction work or related goods and services must be identified sufficiently to enable the respondent to understand the basis of the claim…
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] In my opinion, failure adequately to set out in a payment claim the basis of the claim could be a ground on which an adjudicator could exclude a relevant amount from the determination. Further, even if in such a case a claimant adequately set out the basis of the claim in submissions put to the adjudicator, the adjudicator could take the view that, because the respondent was unable adequately to respond to this subsequent material (because of the provisions of s.20(2B) and s.22(2)(c) of the Act), he or she is not appropriately satisfied of the claimant’s entitlement.
[2.4 Com-BIFSOPA CDI Normal Body Text] Hodgson JA suggested that the construction work or related goods and services must be “identified sufficiently”, as per the above extract. Basten JA, referring to this passage by Hodgson JA, said that the term “identify”:
[2.5 Com-BIFSOPA Normal Body Quote] should be given a purposive construction: what must be done must be sufficient to draw the attention of the principal to the fact that an entitlement to a payment is asserted, arising under the contract to which both the contractor and the principal are parties. In that sense, the claim, to be valid, must be reasonably comprehensible to the other party. If the entitlement does not arise absent the supply of supporting documentation, then the claim must be accompanied by that documentation, unless it has already been provided. 
[2.4 Com-BIFSOPA CDI Normal Body Text] It was also stated by Basten JA that a payment claim need not identify the precise contractual basis for the entitlement, but rather:
[2.5 Com-BIFSOPA Normal Body Quote] the claim should assert, in full, the factual basis upon which it is made, including the provision of documents where necessary, whereas the reliance on a relevant contractual provision may be dealt with by way of submissions, if the matter comes before an adjudicator. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In T & M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QCA 381, the Queensland Court of Appeal considered the position of the parties to be relevant, stating that:
[2.5 Com-BIFSOPA Normal Body Quote] the evaluation of the sufficiency of the identification takes into account the background knowledge of each of the parties derived from their past dealings and exchanges of documentation.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Laimo Masonry Services Pty Ltd v TP Projects Pty Ltd [2015] NSWSC 127, Ball J said that the level of identification:
[2.5 Com-BIFSOPA Normal Body Quote] will depend very much on the circumstances of the case, and may depend on what the terms of the contract between the parties require. The services do not have to be identified completely or accurately. Rather, they must be identified sufficiently so that the respondent is able to respond to the claim
[2.3 Com-BIFSOPA Heading 4] A failure in identification
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been held by the New South Wales Court of Appeal that ‘[w]hether a payment claim identifies the construction work or related goods and services to which the payment relates…is generally a matter for the adjudicator to determine’: Coordinated Construction Co Pty Ltd v Climatech (Canberra) Pty Ltd [2005] NSWCA 229, [26] (Hodgson JA).
[2.5 Com-BIFSOPA Normal Body Quote] In Queensland, this view was adopted in De Neefe Signs Pty Ltd v Build1 (Qld) Pty Ltd; Traffic Technologies Traffic Hire Pty Ltd v Build1 (Qld) Pty Ltd [2010] QSC 279 at [43] per Fryberg J.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Nepean Engineering Pty Ltd v Total Process Services Pty Ltd (in liq) [2005] NSWCA 409, Hodgson JA said (at [36]):
[2.5 Com-BIFSOPA Normal Body Quote] I do not think a payment claim can be treated as a nullity for failure to comply with s.13(2)(a) of the Act, unless the failure is patent on its face; and this will not be the case if the claim purports in a reasonable way to identify the particular work in respect of which the claim is made.
[2.5 Com-BIFSOPA Normal Body Quote] However, ‘[t]he corollary of this proposition is that a payment claim can be treated as a nullity if it does not on its face reasonably purport to comply with s.13(2)(a)’: Brookhollow Pty Ltd v R&R Consultants Pty Ltd & Anor [2006] NSWSC 1, [33] (Palmer J) (emphasis original).
[2.4 Com-BIFSOPA CDI Normal Body Text] The Queensland Court of Appeal, referring to Nepean, in T & M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QCA 381, held:
[2.5 Com-BIFSOPA Normal Body Quote] What was required was that the payment claim purport in a reasonable way to identify the work the subject of the claim, and a payment claim was not a nullity for failure to identify the work unless the failure was patent on its face. The payment claim did not cease to satisfy the requirement concerning identification because it could be subsequently shown that the payment claim was not entirely successful in identifying all of the work.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Richard Kirk Architect Pty Ltd v ABC [2012] QSC 177, Daubney J said that:
[2.5 Com-BIFSOPA Normal Body Quote] … a payment claim must, in a reasonable way, identify the work the subject of the claim so that the person receiving it can understand the basis of the claim.  It almost goes without saying that this requirement is for the benefit of the recipient…A failure to provide a payment claim which meets that requirement would, if the matter later proceeded to adjudication, impugn the jurisdiction to adjudicate if the payment claim did not meet that necessary purpose and effect; any consequent adjudication decision would have failed to comply with the basic requirements of the BCIPA.
[2.4 Com-BIFSOPA CDI Normal Body Text] In John Beever (Aust) Pty Ltd v Paper Australia Pty Ltd [2019] VSC 126 at [65], Lyons J summarised the authorities above and established the following principles for the VIC Act which are relevant when considering whether a payment claim adequately identifies the construction work to which a progress claim relates:
[2.4 Com-BIFSOPA CDI Normal Body Text] 1.    the test of whether a claim is a payment claim for the purpose of the Act is objective;
[2.4 Com-BIFSOPA CDI Normal Body Text] 2.    however, the manner in which compliance is tested is not overly demanding and should not be approached in an unduly technical manner or from an unduly critical point of view;
[2.4 Com-BIFSOPA CDI Normal Body Text] 3.    for the purposes of the identification requirement, it is necessary that the payment claim reasonably identifies the construction work to which it relates such that the basis of the claim is reasonably comprehensible to the recipient party when considered objectively i.e. from the perspective of a reasonable party who is in the position of the recipient; and
[2.4 Com-BIFSOPA CDI Normal Body Text] 4.    in evaluating the sufficiency of the identification of the work, it is appropriate to take into account the background knowledge of the parties from their past dealings and prior exchanges of information including correspondence passing between them before and at the time of the payment claim. To that extent, the Court may go beyond the face of the document itself.
[2.3 Com-BIFSOPA Heading 4] In that case, whilst the payment claim did not of itself contain sufficient information to satisfy s 14(2)(c) VIC Act (equivalent to section 17(2)(a) BCIPA; section 68(1)(a) of the Act), Lyons J held that by having regard to the objective context and circumstances in which the payment claims were prepared, the principal should reasonably have been able to identify the construction work to which the contractor’s claim related.
[2.3 Com-BIFSOPA Heading 4] ‘Reasonably comprehensible to the reasonable principle’: the rule in KDV Sport
[2.4 Com-BIFSOPA CDI Normal Body Text] The recent Queensland decision of KDV Sport Pty Ltd v Muggeridge Constructions Pty Ltd & Ors [2019] QSC 178 (KDV Sport) clarified the requirements in Queensland around what is required of a Claimant to sufficiently identify the construction work.
[2.4 Com-BIFSOPA CDI Normal Body Text] In KDV Sport, Muggeridge Constructions Pty Ltd served a payment claim for $2,365,432.00, which included:
[2.6 Com-BIFSOPA bullet] the percentage of work completed;
[2.6 Com-BIFSOPA bullet] a reference to a “Trade Breakdown Schedule” describing the categories of work completed; and
[2.6 Com-BIFSOPA bullet] claim for variations that Muggeridge had no submitted.
[2.4 Com-BIFSOPA CDI Normal Body Text] Below is an excerpt of the payment claim the subject of KDV Sport:

[2.4 Com-BIFSOPA CDI Normal Body Text] KDV Sport Pty Ltd certified $0 in the payment schedule. Muggeridge applied for adjudication of the payment claim under the Act and the adjudicator awarded Muggeridge $802,198.59. 
[2.4 Com-BIFSOPA CDI Normal Body Text] KDV applied to the Supreme Court to set aside the decision, submitting that the payment claim was invalid as it failed to identify the construction work to which it related. Muggeridge submitted that the Trade Breakdown Schedule and KDV’s knowledge on the project enabled KDV to identify the completed construction work, therefore the payment claim was valid. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Justice Brown agreed with the argument presented by KDV and found the payment claim had not sufficiently identified the construction work. In summary, her Honour found that the payment claim was not reasonably comprehensible to a reasonable principal, as:
[2.6 Com-BIFSOPA bullet] the payment claim did not contain enough information for parties with background knowledge of the Trade Breakdown Schedule to identify the actual work, understand the claim and respond; 
[2.6 Com-BIFSOPA bullet] a mere reference to a completed percentage of a category of work does not identify the actual work that the claim relates to sufficiently; 
[2.6 Com-BIFSOPA bullet] the lack of utility of the percentages was compounded by the principal’s inability to reconcile various amounts of work claimed and the cost thereof; and
[2.6 Com-BIFSOPA bullet] Muggeridge did not submit the variations by the date of the payment claim, support the variations with further information or describe the work that the variations relate to. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In drawing her conclusions, her Honour did not find the Muggeridge’s argument compelling, suggesting that, at [36]:
[2.5 Com-BIFSOPA Normal Body Quote] “While it may be accepted, as Muggeridge contends, that KDV is aware of the content of the Trade Breakdown Schedule, where there are some 51 categories of work in a sizeable contract with a number of components in the work to be undertaken, merely referring to the category of work does not identify the construction work itself to which the claim relates.”
[2.4 Com-BIFSOPA CDI Normal Body Text] Her Honour considered that, although there may be a commonsense order in which the work was to be done which would indicate to KDV what work the claim related to, many factors could have affected this. As such, it was insufficient to provide a percentage of work completed for each category to identify the works that the payment claim related to. Her Honour considered that it was not possible to identify what ‘20% completion’ of the works represented as opposed to ‘30% completion’.
[2.4 Com-BIFSOPA CDI Normal Body Text] At paragraph [38], her honour stated:
[2.5 Com-BIFSOPA Normal Body Quote] “The lack of description of the work and the inability to reconcile the figures to the amount claimed support the fact that the construction work which is subject to the claim cannot be identified with any certainty and the claim does not purport in a reasonable way to identify the work.”
[2.4 Com-BIFSOPA CDI Normal Body Text] Her Honour noted that errors in individual items are not generally sufficient to justify a finding of invalidity, but the extent of the ambiguity in the payment claim in dispute was indicative that the claim was not reasonably comprehensible, as a whole, to a reasonable principal.
[2.4 Com-BIFSOPA CDI Normal Body Text] An additional issue was raised concerning the number of mathematical errors in the claim, with KDV alleging that the claim could not be reconciled. Muggeridge contended that mathematical errors did not invalidate a claim as the principal was at liberty, in their payment schedule, to not indicate payment for that portion. Her Honour agreed with this reasoning but stated that where the number of such errors was so great so as to prevent the identification of the construction work to which the claim related, the payment clam would be invalidated.
[2.4 Com-BIFSOPA CDI Normal Body Text] Additionally, at [49], her Honour suggested that through KDV’s background knowledge of the contract it is reasonable to infer that KDV had knowledge of the overall work to be done in relation to each of the 51 categories in the payment claim, but this did not demonstrate that KDV was able to identify the construction work that was the subject of this specific payment claim. Further, although she accepted that KDV could potentially have identified what the current claim was for by reconstructing all previous claims and attempting to determine the balance of the payment, Brown J considered, at [49], the decision in Neumann Contractors Pty Ltd:
[2.5 Com-BIFSOPA Normal Body Quote] “Requiring a respondent to a payment claim to undertake that kind of research which would be subject to error and within the time constraint of 10 days under the Payments Act leads me to conclude that the payment claim does not identify the construction work to which that claim relates and does not fulfil the requirements of s 17(2) of the Payments Act.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In conclusion, her Honour stated, at [63]:
[2.5 Com-BIFSOPA Normal Body Quote] “The matters identified by KDV do not simply relate to the interpretation of the payment claim and its scope and nature, as contended by Muggeridge, but rather its comprehensibility. There is no description of the work which is the subject of the claim. References to the total percentages of work claimed to date and the amount the subject of the claim does not sufficiently identify the work done”
[2.5 Com-BIFSOPA Normal Body Quote] In Denbrook Constructions Pty Ltd v CBO Developments Pty Ltd [2022] QDC 184, Porter DCJ dismissed an application for summary judgment of a payment claim on the basis that the payment claim in dispute failed to sufficiently identify the construction work to which it related as required by section 68(1)(a) of the BIF Act.
[2.5 Com-BIFSOPA Normal Body Quote] In that case, the payment claim contained no particulars identifying the work relating to the claim and primarily consisted of extracts from a spreadsheet regarding the claimants’ costs of works and balance due with reference to previous payment claims. In determining that the payment claim did not sufficiently identify the construction work, Porter DCJ confirmed that the test to determine whether a payment claim satisfies section 68(1)(a) is set out in KDV Sport Pty Ltd v Muggeridge Constructions Pty Ltd [2019] QSC 178. His Honour explained that the payment claim failed to satisfy section 68(1)(a) because: it contained no particulars identifying the work relating to the claim; reference to the previous payment claim cannot be relied on to substantiate a payment claim as it did not use words which purported to incorporate the previous claim; even if the previous payment claim could be used, prior claims says nothing about the work the subject of the claim; and the contents of the payment schedule could not be considered in determining whether the work was reasonably identified such that it was comprehensible to the respondent.
[2.5 Com-BIFSOPA Normal Body Quote] In Iris Broadbeach Business Pty Ltd v Descon Group Australia Pty Ltd & Anor [2023] QSC 290, Williams J held that the payment claim in dispute was not a valid payment claim having regard to s 68(1)(a) of the BIF Act. Her Honour identified the key issue by following Brown J’s comments in KDV Sport Pty Ltd v Muggeridge Constructions Pty Ltd & Ors [2019] QSC 178. The issue was whether substantial compliance with s 68(1)(a) of the BIF Act is sufficient for validity.  Her Honour observed [at 160] that ‘if “big ticket items” are adequately described, on this approach the payment claim would be valid in its entirety. However, if the items that are not sufficiently described add up to a substantial amount of the total claimed in the payment claim, then the payment claim would be invalid’.
[2.5 Com-BIFSOPA Normal Body Quote] In this case, the items identified in the disputed payment claim were considered to be insufficiently particularised as required by s 68(1)(a). This was because [at 167] all of the items fall within one or more of the following categories: 
[2.5 Com-BIFSOPA Normal Body Quote] No description or identification of the actual work undertaken.
[2.5 Com-BIFSOPA Normal Body Quote] Not supported by information or material provided at or prior to the delivery of [the payment claim].
[2.5 Com-BIFSOPA Normal Body Quote] Not supported by background knowledge of the parties through past dealings and exchange of information.
[2.3 Com-BIFSOPA Heading 4] Document in support of a payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] In T & M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QCA 381, the Queensland Court of Appeal, referring to the decision of the primary judge below, held that (at [41]):
[2.5 Com-BIFSOPA Normal Body Quote] Baxbex Pty Ltd does not stand for the proposition implicit in the judge’s reasons that a failure to provide documents referred to in a payment claim or attachment thereto per se results in there being a deficiency in identification for the purposes of s 17(2)(a). In the present case, the fact that the attachment referred to certain supplier invoices without also attaching them did not detract from the identification that was provided being sufficient.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Incorporation of an earlier spreadsheet. Unifor Australia Pty Ltd v Katrd Pty Ltd atf Morshan Unit Trust t/as Beyond Completion Projects [2012] QSC 252. The claimant served three invoices on the respondent, each endorsed as a payment claim under the Act. Relevantly one tax invoice (dated 3 July 2012) referred to ‘our earlier variation claim presented…on 20 June 2012, with comprehensive documentary evidence’. On 20 July 2012, the claimant had sent the respondent an email attaching a ‘Summary of Costed Variations’ which had, in fact, attached the wrong document. The correct spreadsheet was provided on 27 June 2015. The Court held that the tax invoice in question sufficiently incorporated the spreadsheet as part of the claim, and accordingly, was reasonably comprehensible to the respondent.
[2.6 Com-BIFSOPA bullet] Further particulars or details sent contemporaneously with payment claim. Richard Kirk Architect Pty Ltd v Australian Broadcasting Corporation & Ors [2012] QSC 177. An email from the claimant to the respondent attached (1) a without prejudice letter; (2) a notice of dispute; and (3) a document purporting to be a payment claim. The document purporting to be a payment claim stated a claimed amount and referred to the ‘method of calculation’ in the attachments to the payment claim. There were no attachments to this document, but, the notice of dispute attached to the same email provided a detailed itemisation of amounts claimed, which added up to the amount of the purported payment claim. The Court held that it was clear that the contemporaneous notice of dispute formed part of the payment claim.
[2.6 Com-BIFSOPA bullet] Information provided after the service of payment claim
[2.6 Com-BIFSOPA bullet] In Iris Broadbeach Business Pty Ltd v Descon Group Australia Pty Ltd & Anor [2023] QSC 290, Williams J [at 159] concluded that ‘whether s 68(1)(a) of the BIF Act is complied with is to be considered in light of information and material provided at the time of delivery of Payment Claim’, not additional information provided after the delivery of the payment claim. Her Honour followed the decision in Coordinated Construction Co Pty Ltd v Climatech (Canberra) Pty Ltd [2005] NSWCA 229 in which Basten JA stated that: ‘… the claim, to be valid, must be reasonably comprehensible to the other party. If the entitlement does not arise absent the supply of supporting documentation, then the claim must be accompanied be that documentation, unless it has already been provided. … the claim should assert, in full, the factual basis upon which it is made, including the provision of documents where necessary, whereas the reliance on a relevant contractual provision may be dealt with by way of submission, if the matter comes before an adjudicator. …’. 
[2.6 Com-BIFSOPA bullet] Furthermore, Williams J refused to accept the Respondent’s submission that a previous practice (without more) is sufficient to find a basis to consider additional information provided after the delivery of payment claim. In doing so, Her Honour [at 157-158] noted that this submission was not put on the basis of an estoppel, so to accept the submission would be contrary to the reasoning in Coordinated Construction as well as lead to commercial uncertainty
[2.3 Com-BIFSOPA Heading 4] The inclusion of an erroneous or unrelated item in a payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] It has been held that, as an adjudicator is to determine the extent and value of construction work, ‘the inclusion of a claim for an obviously irrelevant item for what is not construction work does not deprive the adjudicator of that jurisdiction’: Matrix Projects (Qld) Pty Ltd v Luscombe [2013] QSC 4, [24] (Douglas J).
[2.4 Com-BIFSOPA CDI Normal Body Text] A similar view was reached in Walter Construction Group Ltd v CPL (Surry Hills) Pty Ltd [2003] NSWSC 266, [66]-[67] (Nicholas J).
[2.3 Com-BIFSOPA Heading 4] Severance of an insufficiently identified item in a payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] In Tenix Alliance Pty Ltd v Magaldi Power Pty Ltd [2010] QSC 7, Fryberg J said:
[2.5 Com-BIFSOPA Normal Body Quote] it does not seem to me that the inclusion of one aspect of a claim incorrectly in the claim is sufficient to invalidate the whole claim. The requirements for a valid claim are set out in s 17(2) of the Act, and in my judgment the erroneous inclusion of the amount to which I have just been discussing, is not sufficient to take the claim outside the statutory description.
[2.4 Com-BIFSOPA CDI Normal Body Text] A similar approach was adopted in Gantley Pty Ltd v Phoenix International Group Pty Ltd [2010] VSC 106 where, following a review of the authorities, Vickery J concluded (at [115]-[116]):
[2.5 Com-BIFSOPA Normal Body Quote] …The question should be whether the Act, either expressly or impliedly, operates to exclude the common law doctrine of severance. I find that it does not. Indeed, the purposes and objects of the Act earlier described are best served by processes which, so far as possible, ought to accommodate reasonable flexibility and avoid unnecessary technicality.
[2.5 Com-BIFSOPA Normal Body Quote] Severance in this case would operate to achieve the purpose and objects of the Act and would not operate to diminish the attainment of these goals. A respondent to a payment claim and an adjudicator, if appointed, should be able to assess the valid part of this progress claim which sufficiently describexs the work for which payment is claimed, and provide a rational response or adjudication determination in respect of that part of the claim, and exclude from consideration that part of the claim which does not comply.
[2.3 Com-BIFSOPA Heading 4] Amounts not considered “construction work or related goods and services”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Grocon (Belgrave St) Developer Pty Ltd v Construction Profile Pty Ltd [2020] NSWSC 409, the Court was required to consider whether a claimant was entitled to claim in a payment claim two bank guarantees, which the respondent had recourse to under the contract. In permanently restraining the Claimant from seeking an adjudication determination in respect of the purported payment claim, Justice Ball held that the amounts were not “construction work or for related goods and services” under section 13 of the NSW Act (equivalent to section 68 BIFSOPA). Further, his Honour held that while the other amounts in the claim were for “construction work or for related goods and services”, they did not validate the payment claim, as they were small in quantum and did not provide a basis for the Court to refuse relief.
[2.2 Com-BIFSOPA Heading 3] 68.5    Section 68(1)(b)
[2.3 Com-BIFSOPA Heading 4] ‘States the amount’ (the claimed amount): more than one amount?
[2.4 Com-BIFSOPA CDI Normal Body Text] The judgement in Camporeale Holdings Pty Ltd v Mortimer Construction Pty Ltd & Anor [2015] QSC 211 suggests that a payment claim need not expressly state a ‘claimed amount’, but that it is sufficient if it states amounts of progress payments which, together, constitute the claimed amount under the payment claim. Relevantly, Henry J said:
[2.5 Com-BIFSOPA Normal Body Quote] The applicant submitted the covering email here should have cited a single total amount claimed. However, it is obvious when the documents are considered together that the amount claimed was the total of the four amounts stated in the invoices. The absence of a statement of the mathematical total of those amounts does not make the statements of the four amounts considered in combination any less a statement of the amount of the progress payment claimed. I am fortified in reaching that conclusion by s 32C of the Acts Interpretation Act 1954 (Qld) which provides that in an act words in the singular include the plural and vice versa. Thus a payment claim which states the amounts of the progress payment claimed to be payable will comply with s 17(2)(b), as long as those amounts are stated as part of one payment claim.
[2.5 Com-BIFSOPA Normal Body Quote] When collective consideration is given to the content of all of the documents served it is readily discernible that the combined total of the invoices served was the amount of the progress payment claimed, that what had been served was one payment claim and that it was said to be made under the Act. The documents were served together and it would have been obvious to their recipient that they fell for consideration collectively. When they are considered collectively it can be seen that what was served was one payment claim and that it met the requirements of s 17.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Iris Broadbeach Business Pty Ltd v Descon Group Australia Pty Ltd [2024] QSC 16, Iris raised five grounds to apply for orders that the adjudication decision be set aside for jurisdictional error. One of the grounds was that the payment claim did not comply with the requirements under section 68(1) of the Act because it did not expressly request payment of the claimed amount and did not include the word ‘invoice’. Wilson J held that the payment claim objectively requested for payment, despite not expressly including such request, on the basis that it stated that it was submitted under the BIF; it was titled “progress claim”, and it included amounts for the “Total Progress Claim Value for the Month”. Further, another factor the Court took into account was that the Claimant’s managing director had provided a statutory declaration which stated that:
[2.5 Com-BIFSOPA Normal Body Quote] “I am making this statutory declaration in connection with the payment of Progress Claim No. 7 dated 28 February 2023 pursuant to the Contract.”
[2.4 Com-BIFSOPA CDI Normal Body Text] Therefore, the cumulative effect of these factors satisfied Wilson J that the claim was a request for payment and met the requirements of s 68(1).
[2.4 Com-BIFSOPA CDI Normal Body Text] In MWB Everton Park Pty Ltd as trustee for MWB Everton Park Unit Trust v Devcon Building Co Pty Ltd [2024] QCA 94, the Queensland Court of Appeal considered whether two sets of documents sent by the builder to the principal constituted a payment claim within the meaning of section 68(1) of the Qld Act. Ultimately, Dalton JA (Brown J and Kelly J agreeing) provided three independent reasons of why the documents do not meet the statutory definition of a payment claims:
[2.4 Com-BIFSOPA CDI Normal Body Text] The documents did not identify the construction work to which the progress claim related within the meaning of s 68(1)(a) of the Act. In this regard, the Court at [25] affirmed the reasoning in KDV Sport Pty Ltd v Muggeridge Constructions Pty Ltd [2019] QSC 178. Dalton JA held that, as the contract was for construction of 56 townhouses, it was meaningless to say that 5% of concerting or 12% of plumbing had been completed. For the purpose of s 68(1)(a), more description was needed. However, His Honour also made an observation at Footnote [5] that ‘in different factual circumstances, a trade summary or trade breakdown might be sufficient. Hypothetically, if the contract was to build a single domestic dwelling and the only concreting was to the driveway, a description that 50% of the concreting had been achieved would allow the party receiving the claim to understand what work the builder said had been done.’
[2.4 Com-BIFSOPA CDI Normal Body Text] The documents could not be considered as a payment claim having regard to s 68(1)(b). This was because that the documents attached to the email did not state an amount which was the claimed amount of the progress payment.
[2.4 Com-BIFSOPA CDI Normal Body Text] The documents did not request payment of the claimed amount as per section 68(1)(c) of the Act. In this regard, the Court of Appeal disagreed with the Wilson J’s analysis at [132] in Iris Broadbeach Business Pty Ltd v Descon Group Australia Pty Ltd [2024] QSC. At paragraph [36] it was held that:
[2.4 Com-BIFSOPA CDI Normal Body Text] [36] … In [Iris Broadbeach] the judge was prepared to find that the documentation impliedly included a request for payment ... While I cannot see any relevant distinction between the documents in this case and the documents in Iris Broadbeach Business, I am afraid that I am not persuaded by the analysis in that case. I do not think it can be said that the documents comprising the email and attachments of 30 June 2023 made a request for payment, independently of identifying the amount the subject of the claim.
[2.3 Com-BIFSOPA Heading 4] ‘Claims’ to be payable
[2.4 Com-BIFSOPA CDI Normal Body Text] In De Neefe Signs Pty Ltd v Build1 (Qld) Pty Ltd; Traffic Technologies Traffic Hire Pty Ltd v Build1 (Qld) Pty Ltd [2010] QSC 279, Fryberg J held that it is not a requirement that a payment claim correctly states  the amount payable and that “[i]t is sufficient to state the amount which the claimant claims to be payable.”
[2.3 Com-BIFSOPA Heading 4] ‘Payable’
[2.5 Com-BIFSOPA Normal Body Quote] In De Neefe Signs Pty Ltd v Build1 (Qld) Pty Ltd; Traffic Technologies Traffic Hire Pty Ltd v Build1 (Qld) Pty Ltd [2010] QSC 279, Fryberg J was not persuaded that ‘payable’ under section 17 of the Act means ‘due and payable’. 
[2.2 Com-BIFSOPA Heading 3] 68.6    No requirement to state that it is made under the Act (section 17(2)(c) BCIPA)
[2.4 Com-BIFSOPA CDI Normal Body Text] The Act has abolished the requirement for a payment claim to state that it is made under the Act. This has been regarded as a policy choice which favours subcontractor payment by removing the requirement to ensure each claim states it has been made under the Act. On the removal of this requirement, the Minister said in his second reading speech of the Building Industry Fairness (Security of Payment) Bill 2017:
[2.5 Com-BIFSOPA Normal Body Quote] These reforms expedite the payment system, by providing that all invoices are automatically deemed to be statutory claims for payment. These provisions will restore accountability on respondents, by restoring the penalty which used to apply when there was a failure to provide a payment schedule when one was due
[2.4 Com-BIFSOPA CDI Normal Body Text] Whilst the Act has removed this requirement, the requirement for payment claims to be endorsed still remains in other jurisdictions and therefore, commentary on this requirement has been included below. 
[2.4 Com-BIFSOPA CDI Normal Body Text] On the previous requirement to state that a claim was made pursuant to BCIPA, In Leighton Contractors Pty Ltd v Campbelltown Catholic Club Ltd [2003] NSWSC 1103, Einstein J said in the context of the NSW Act that:
[2.5 Com-BIFSOPA Normal Body Quote] There is no room for ambiguity of any type and it is critical that the recipient of a payment claim be made aware by the terms of that claim that the provisions of the Act have been engaged.
[2.4 Com-BIFSOPA CDI Normal Body Text] Einstein J approved the test from Nicholas J in Walter Construction Group Ltd v CPL (Surry Hills) Pty Ltd [2003] NSWSC 266 that:
[2.5 Com-BIFSOPA Normal Body Quote] The test is an objective one. In deciding the meaning conveyed by a notice a court will ask whether a reasonable person who had considered the notice as a whole and given fair and proper consideration would be left in any doubt as to its meaning.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, strict compliance was held to not necessarily be a requirement under section 17(2)(c) BCIPA as substantial compliance may be sufficient: see Eco Steel Homes Pty Ltd v Hippo’s Concreting Pty Ltd & Ors [2014] QSC 135, [8] (Daubney J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Hawkins Construction v Mac’s Industrial Pipework [2001] NSWSC 815, under the NSW Act, it was submitted that because a payment claim was endorsed as under the ‘Building Construction Ind Security of Payments Act 1999’, it was not a valid payment claim. As to this submission, Windeyer J said:
[2.5 Com-BIFSOPA Normal Body Quote] This argument might have had some weight in 1800. In 2001, an argument based on the absence of the word “and” and the letters “ustry” has no merit. It should not have been put.
[2.4 Com-BIFSOPA CDI Normal Body Text] The purpose of this requirement was considered in Eco Steel Homes Pty Ltd v Hippo’s Concreting Pty Ltd & Ors [2014] QSC 135, where Daubney J said that:
[2.5 Com-BIFSOPA Normal Body Quote] The purpose of that subsection is clear – it is to ensure that notice is given on the face of a payment claim that the claimant is invoking the procedure for recovering progress payments contained in Part 3 of BCIPA.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Eco Steel Homes Pty Ltd v Hippo’s Concreting Pty Ltd & Ors [2014] QSC 135. The payment claim in question stated ‘This is a payment claim made under the Building and Construction Building Act 2004’. This was held by the Court to be sufficiently clear to give notice to the respondent.
[2.6 Com-BIFSOPA bullet] South East Civil and Drainage Contractors Pty Ltd v AMGW Pty Ltd [2013] QSC 45; [2013] 2 Qd R 189. The payment claim in question stated ‘This is a claim under the Building and Construction Industry Security of Payment Act 2004 Queensland’. This was held by the Court to be sufficient, with a reader not left in any doubt that it was made under the Act.

```
