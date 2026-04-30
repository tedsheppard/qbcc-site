# Element-page brief — d-within-time
**Title:** Decided within time
**Breadcrumb:** Requirements of an adjudicator's decision
**Anchor id:** `d-within-time`
**Output file:** `bif_guide_build/v3/pages/page_d-within-time.html`

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

### `statute/chapter_3/section_085.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 85 — Time for deciding adjudication application
(1) Subject to section86, an adjudicator must decide an
adjudication application no later than—
(a) for a standard payment claim—10 business days after
the response date; or
(b) for a complex payment claim—15 business days after
the response date.
(2) The response date is—
(a) if the adjudicator is given an adjudication response
under section 83—the day on which the adjudicator
receives the response; or
(b) if the respondent is prevented from giving the
adjudicator an adjudication response under
section 82(2)—the last day on which the respondent
could have given the adjudicator an adjudication
Page 113 Current as at 27 April 2025

response under section 83 had it not been prevented
from doing so under section82(2); or
(c) otherwise—the last day on which the respondent could
give the adjudicator an adjudication response under
section 83.
(3) An adjudicator must not decide an adjudication application
before the end of the period that the respondent may give an
adjudication response to the adjudicator under section 83,
unless—
(a) the adjudicator decides he or she does not have
jurisdiction to adjudicate the application; or
(b) the adjudicator decides the application is frivolous or
vexatious.

```

### `statute/chapter_3/section_086.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 86 — Extending time for deciding adjudication application
(1) The claimant and respondent for an adjudication application
may, before or after the end of the maximum period for
deciding the application under section85(1), agree in writing
that the adjudicator has additional time to decide the
application.
(2) Despite section85(1), an adjudicator may decide an
adjudication application within a longer period if—
(a) the claimant and respondent have informed the
adjudicator that they have agreed under subsection(1)
that the adjudicator has additional time to decide the
application; or
(b) the application relates to a complex payment claim and,
in the opinion of the adjudicator, the claimant and
respondent have failed to reach an agreement mentioned
in subsection(1).
(3) The longer period is—
(a) if subsection(2)(a) applies—the additional time agreed
to by the claimant and respondent under subsection(1);
or

(b) if subsection(2)(b) applies—5 business days after the
time the adjudicator would otherwise have to decide the
application under section85(1).
(4) If the adjudicator has additional time to decide an adjudication
application under this section, the adjudicator must notify the
registrar of the additional time to decide the application within
4 business days after—
(a) if subsection(2)(a) applies—the day the claimant and
respondent agreed under subsection(1); or
(b) if subsection(2)(b) applies—the day the adjudicator
decided he or she had additional time under
subsection(2)(b).

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_085.txt`
```
# Annotated BIF Act source — Section 85
# Chapter: CHAPTER 3 – Progress payments
# Section title: Time for deciding adjudication application
# DOCX paragraphs: 2694-2750

[2 Com-BIFSOPA Heading 1] SECTION 85 – Time for deciding adjudication application 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 85    Time for deciding adjudication application 
[1.3 BIFSOPA level 1 (CDI)] Subject to section 86, an adjudicator must decide an adjudication application no later than—
[1.4 BIFSOPA level 2 (CDI)] for a standard payment claim—10 business days after the response date; or
[1.4 BIFSOPA level 2 (CDI)] for a complex payment claim—15 business days after the response date.
[1.3 BIFSOPA level 1 (CDI)] The response date is—
[1.4 BIFSOPA level 2 (CDI)] if the adjudicator is given an adjudication response under section 83—the day on which the adjudicator receives the response; or
[1.4 BIFSOPA level 2 (CDI)] otherwise—the last day on which the respondent could give the adjudicator an adjudication response under section 83. 
[1.3 BIFSOPA level 1 (CDI)] An adjudicator must not decide an adjudication application before the end of the period that the respondent may give an adjudication response to the adjudicator under section 83, unless—
[1.4 BIFSOPA level 2 (CDI)] the adjudicator decides he or she does not have jurisdiction to adjudicate the application; or
[1.4 BIFSOPA level 2 (CDI)]  the adjudicator decides the application is frivolous or vexatious.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 85.1    Section 85(1) – ‘Subject to section 86’ (extensions of time)
[2.4 Com-BIFSOPA CDI Normal Body Text] In Allpro v Micos [2010] NSWSC 453, McDougall J held that the adjudicator erred in taking the view that:
[2.5 Com-BIFSOPA Normal Body Quote] it was open to him firstly to impose a very short time limit on the parties to indicate their attitude to the request for an extension of time and secondly to assume that consent was given if the parties did not respond within the time so limited. There may conceivably be circumstances where a party to an adjudication application should be taken to have agreed to an extension of time under s 21 (3)(b) without expressly having so indicated. But that could not arise in circumstances such as those shown on the evidence in this case where, as I have said, the time limited by the adjudicator was extremely short. Section 21 (3)(b) requires agreement. That means real or actual agreement. It does not in my view entitle an adjudicator to purport to impose agreement of parties in the way that I have outlined.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, section 86 of the Act allows a period of a further five business days in making a decision on a complex payment claim, or any other period agreed between the parties, where the requirements of section 86(1) of the Act are satisfied.
[2.2 Com-BIFSOPA Heading 3] 85.2    Section 85(1) – deciding within time
[2.4 Com-BIFSOPA CDI Normal Body Text] In the context of the previous section 25(3) BCIPA (before the commencement of the Amendment Act), the Court of Appeal in Heavy Plant Leasing Pty Ltd v McConnell Dowell Constructors (Aust) Pty Ltd & Ors [2013] QCA 386 said that:
[2.5 Com-BIFSOPA Normal Body Quote] Unless the mechanism of s 32 is engaged, there is nothing in the Act which directly or indirectly relieves the adjudicator of the obligation to decide an adjudication within the times allowed by s 25(3).
[2.4 Com-BIFSOPA CDI Normal Body Text] The question of an adjudication decision handed down beyond the statutory time frame (and the effectiveness of such decision) was considered in MPM Constructions v Trepcha Constructions [2004] NSWSC 103. McDougall J said:
[2.5 Com-BIFSOPA Normal Body Quote] In accordance with what the majority said in Project Blue Sky at 397 [97], I start from the proposition that it is unlikely in this case that the legislature intended that an act done in breach of the time limit set out s 21(3)(a) would be invalid. I say that because the consequences of invalidity seem to me to be susceptible of undermining the purpose of the legislation both as set out in s 3 and as expanded upon by the Minister in the Second Reading Speech to which Bergin J referred in Paynter Dixon and to which I referred in Musico. That is to my view confirmed because where a determination is not made within the relevant time limit, the only relief that a claimant seems to have under the Act is that for which s 26 applies, namely, to withdraw the application and make a new application. However, by s 26(3) that course may only be taken within five business days after the claimant becomes entitled to withdraw the previous adjudication application. Relevantly for present purposes, that means that the application must have been withdrawn within five days of the expiry of the s 21(3) time period. It would be, to put it mildly, anomalous if the effect of non-compliance with the time period allowed by s 21(3) were to render any subsequent adjudication a nullity, but if because of non-compliance of s 26(3) the claimants were unable to seek adjudication of the dispute. That would not seem to me an outcome consistent with the evident objects of the legislation and, therefore, something to be avoided unless no other view is available.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour held that the challenge to the adjudicator’s determination on the basis that it was made outside of the relevant time failed, and that the requirement to make a decision within time is not jurisdictional (at [24]).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Cranbrook School v JA Bradshaw Civil Contracting [2013] NSWSC 430, McDougall J followed his decision in MPM Constructions v Trepcha Constructions [2004] NSWSC 103, relevantly stating:
[2.5 Com-BIFSOPA Normal Body Quote] I dealt with this in MPM Constructions Pty Ltd v Trepcha Constructions Pty Ltd [2004] NSWSC 103. I concluded that the requirements of the section were not jurisdictional. I noted, among other things, that the Act contained its own provision for non-compliance on the part of the adjudicator, with the time requirements of s 21(3).
[2.5 Com-BIFSOPA Normal Body Quote] To my mind, it would be quite extraordinary if the legislature intended that a builder or subcontractor who had got through the various hurdles that the Act imposes, in the path of obtaining a successful determination, up until the point of receipt of the adjudicator's reasons, should be disqualified from the benefit of a determination in its favour simply because the adjudicator did not comply with the statutory time limit.
[2.4 Com-BIFSOPA CDI Normal Body Text] In MT Lewis Estate Pty Ltd v Metricon Homes Pty Ltd [2017] NSWSC 1121, Hammerschlag J applied McDougall J’s reasoning in Cranbrook School v JA Bradshaw Civil Contracting [2013] NSWSC 430, and found that an out of time adjudication decision will not invalidate the decision. At [61] his Honour added to the considerations against invalidity to those identified by McDougall J:
[2.5 Com-BIFSOPA Normal Body Quote] [U]nder s 31 [of the NSW Act], an adjudicator is exempt from liability for anything omitted to be done in good faith. This would undoubtedly extend to a failure to deliver on time. If the time limit in s 29(4) is a guillotine, the obligation on an adjudicator to deliver a determination, whilst it may have been breached, would come to an end with no redress against her or him unless the failure was not in good faith. The adjudicator would be relieved of the burden of producing, albeit that she or he could not charge for work done. If a later adjudication is nevertheless valid, the adjudicator’s duty would continue and the sanction of not being paid is more real. This position is more conducive to the prompt delivery by adjudicators and fulfilment of the overall objects of the Act. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Similarly, in Ian Street Developer Pty Ltd v Arrow International Pty Ltd [2018] VSC 14, Riordan J held that an adjudicator’s decision was valid, notwithstanding that it was made outside of the time period required by the VIC Act. After considering the case law on this issue, including MT Lewis Estate Pty Ltd, Riordan J concluded that:
[2.6 Com-BIFSOPA bullet] invalidating an adjudicator’s determination would be inconsistent with the objects of the Victorian Act;
[2.6 Com-BIFSOPA bullet] section 23(2B) of the Victorian Act provides that the adjudicator’s determination is void in certain circumstances, arising out of post-appointment conduct, which do not include non-compliance with s 22(4);
[2.6 Com-BIFSOPA bullet] If an out of time determination was a nullity there would be no purpose in the Act providing that the claimant may withdraw the application in the circumstances;
[2.6 Com-BIFSOPA bullet] to deprive the parties of the benefit of an adjudication determination after they have completed their submissions would result in an inconvenience and the likely costs of a further adjudication; and
[2.6 Com-BIFSOPA bullet] if sub-s 22(4) of the Victorian Act was intended to impose a jurisdictional requirement, then an adjudication determination, which was given within the 10 day period, could still be invalid if it was not given as ‘expeditiously as possible’. A requirement to do an act as ‘expeditiously as possible’ is less likely to be jurisdictional because it does not have a ‘rule-like quality which can be easily identified and applied’ (in the equivalent Queensland section an adjudicator must decide an adjudication application ‘as quickly as possible’).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Ian Street Developer Pty Ltd v Arrow International Pty Ltd [2018] VSCA 294, the Victorian Supreme Court of Appeal upheld the decision of Riordan J (above), confirming that where an adjudicator fails to determine an adjudication application within the time prescribed under s 22(4) of the VIC Act (equivalent to s 85(1) of the Act), the determination is nonetheless valid. Maxwell P (McLeish and Niall JJA agreeing) agreed with the judgment at first instance, finding that “to hold that non-compliance with the time limit invalidated the adjudication decision would be inconsistent both with the express provisions of the [VIC] Act governing the adjudication process and with the objects of the statutory scheme as a whole”. On this point, Riordan J applied the reasoning of McDougall J and Hammerschlag J in Cranbrook School v JA Bradshaw Civil Contracting [2013] NSWSC 430 and MT Lewis Estate Pty Ltd v Metricon Homes Pty Ltd [2017] NSWSC 1121 (respectively) and held at [95]:
[2.5 Com-BIFSOPA Normal Body Quote] I agree with McDougall J and Hammerschlag J that a failure to comply with the time limits for determining an adjudication application does not invalidate the adjudication determination for the reasons given by them, referred to above. In my opinion, I can have regard to this reasoning in interpreting the Victorian Act because the relevant provisions are substantially identical. I would also add the following considerations:
[2.5 Com-BIFSOPA Normal Body Quote] (a) Section 22(4) of the Act regulates ‘the exercise of functions already conferred on [the adjudicator], rather than impos[ing] essential preliminaries to the exercise of those functions’. This is a strong indicator of an intention that non-compliance does not deprive the adjudicator of his or her jurisdiction.
[2.5 Com-BIFSOPA Normal Body Quote] (b) Section 23(2B) of the Act provides that the adjudicator’s determination is void in certain circumstances, arising out of post-appointment conduct, which do not include non-compliance with s 22.
[2.4 Com-BIFSOPA CDI Normal Body Text] This position has since been distinguished in Queensland. Justice Dalton of the Queensland Supreme Court held in Galaxy Developments Pty Ltd v Civil Contractors (Aust) Pty Ltd t/a CCA Winslow & Ors [2020] QSC 51, that an adjudication decision handed down outside of the timeframe prescribed by section 85 of BIFSOPA will be void. In arriving at this conclusion, her Honour reasoned in that particular language choices of the legislature demonstrated “particularity and precision” with respect to time limits. Further, the existence of section 86 which specifically contemplated the extension of time limits indicated of the importance of the statutory time limits under section 85 BIFSOPA. Her Honour’s decision distinguished the appellate court authority of Ian Street Developer Pty Ltd v Arrow International Pty Ltd & Anor [2018] VSCA 294 which found that the purpose of the Vic Act would be frustrated if the expiry of the time limit terminated the adjudicator’s jurisdiction. In the context of BIFSOPA, her Honour said that “the changes to the current Queensland legislation were made after all the extant interstate decisions on similar provisions”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Civil Contractors (Aust) Pty Ltd v Galaxy Developments Pty Ltd & Ors; Jones v Galaxy Developments Pty Ltd & Ors [2021] QCA 10, the Queensland Court of Appeal upheld the primary judge’s decision that an adjudication decision was void because the adjudicator failed to deliver its determination within the timeframe prescribed by section 85 of BIFSOPA. The Court of Appeal considered both the NSW and Victorian authorities in relation to the time limits for deciding an adjudication application but was able to distinguish these authorities on the basis that the wording of the BIF Act is different.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Demex Pty Ltd v Marine Civil Contractors Pty Ltd & Anor [2022] QSC 141 , Bowskill CJ dismissed an interlocutory application to restrain the enforcement of an adjudication determination on the basis that the applicant failed to establish a prima facie case and that the balance of convenience did not favour the granting of an injunction. The applicant argued that an adjudication determination was void for jurisdictional error, because it was not provided to the parties within the statutory timeframe set out under section 21(3) of the NSW SOP Act (similar to section 85(1) of the BIF Act)Bowskill CJ held that although there  there was a question to be tried, it was a weak one. The adjudicator made their determination within the statutory timeframe, however the determination was not provided to the parties until the next available business day, which fell outside the statutory timeframe. In determining that it would be difficult to establish jurisdictional error on this basis, Bowskill CJ explained that as matter was dealt with under the NSW SOP Act,established line of authorities in New South Wales, establish that the timing of an adjudication determination is not ‘jurisdictional’, and that a failure to provide the determination by the time provided does not mean the determination is invalid. However, His Honour outlined that it is arguable that in Queensland the determination may have been void on the basis of the decision inGalaxy Developments Pty Ltd v Civil Contractors (Aust) Pty Ltd t/a CCA Winslow & Ors [2020] QSC 51, noting there are distinct differences between the NSW SOP Act and the BIF Act.


[2.4 Com-BIFSOPA CDI Normal Body Text] In Karam Group Pty Ltd as Trustee for Karam (No. 1) Family Trust v HCA Queensland Pty Ltd & Ors [2022] QSC 290, Williams J held that an Adjudicator will have failed to make their determination within the statutory timeframe if they fail to attend to all aspects specified in s 88 BIFSOPA, including notifying the parties that the decision had been made. In this case, the adjudicator purportedly made their decision in time but did not notify the parties until after the time prescribed in s 85(1) BIFSOPA surpassed. The Court held that the failure to notify the parties mean that the adjudicator’s decision was not made within the time and was invalid.  
[2.2 Com-BIFSOPA Heading 3] 85.3    ‘Gives’
[2.4 Com-BIFSOPA CDI Normal Body Text] The deadline under section 85(2) is defined by reference to the date on which the adjudicator ‘gives the response or the reply. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to section 102 for further commentary on this distinction.
[2.2 Com-BIFSOPA Heading 3] 85.4    Section 25A(3) – ‘minimum consideration period’
[2.4 Com-BIFSOPA CDI Normal Body Text] On the context of BCIPA, Section 25A(1) provided that an adjudicator must not decide an adjudication application before the end of the minimum consideration period. Pursuant to section 25A(3), this minimum consideration period will be, at a ‘minimum’, the period within which the respondent may give an adjudication response to the adjudicator under section 24A.
[2.4 Com-BIFSOPA CDI Normal Body Text] Prior to the Amendment Act, there was an equivalent limitation under the former section 25(1) of the Act which read:
[2.5 Com-BIFSOPA Normal Body Quote] An adjudicator must not decide an adjudication application until after the end of the period within which the respondent may give an adjudication response to the adjudicator.
[2.4 Com-BIFSOPA CDI Normal Body Text] This restriction under the former section 25(1) was considered in Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA Pty Ltd) [2007] QSC 333 where Chesterman J held that where the respondent was not entitled to give an adjudication response (because no payment schedule was served on the claimant within the time allowed), the adjudicator was not required to wait five business days to deliver the adjudicator decision, being the end of the period described under the-then section 25(1). Relevantly, his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] The applicant therefore could not give an adjudication response. Was the second respondent nevertheless obliged to wait for the expiration of the five days? The answer must be negative. Such a construction of the Act would achieve no purpose. There was no period within which the applicant could give an adjudication response because of s 24(3). Section 25(1) is not dealing with abstractions. It provides the procedure that an adjudicator must follow in particular cases. The time within which an adjudicator may or may not decide an application will depend upon the particular facts on which the Act operates. Where a respondent may not deliver an adjudication response the adjudicator does not have to pretend that he can, and wait for a period during which nothing can happen. The adjudication was not premature.
[2.4 Com-BIFSOPA CDI Normal Body Text] Although the former restriction under section 25(1) is now replaced by the ‘minimum consideration period’ under section 25A(3), given the parallels between the former section 25(1) and current section 25A(1) of the Act, it is an open question as to whether the reasoning of Chesterman J in Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (WA Pty Ltd) [2007] QSC 333 will apply to the minimum consideration period where a respondent has no entitlement to deliver an adjudication response.
[2.4 Com-BIFSOPA CDI Normal Body Text] Under the NSW Act, in Emag Constructions Pty Ltd v Highrise Concrete Contractors (Aust) Pty Ltd [2003] NSWSC 903, Einstein J held that an adjudication determination on the last day on which an adjudication response might have been lodged, vitiated the validity of the adjudicator’s determination.

```

### `annotated/section_086.txt`
```
# Annotated BIF Act source — Section 86
# Chapter: CHAPTER 3 – Progress payments
# Section title: Extending time for deciding adjudication application
# DOCX paragraphs: 2751-2766

[2 Com-BIFSOPA Heading 1] SECTION 86 – Extending time for deciding adjudication application
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 86    Extending time for deciding adjudication application 
[1.3 BIFSOPA level 1 (CDI)] The claimant and respondent for an adjudication application may, before or after the end of the maximum period for deciding the application under section 85(1), agree in writing that the adjudicator has additional time to decide the application. 
[1.3 BIFSOPA level 1 (CDI)] Despite section 85(1), an adjudicator may decide an adjudication application within a longer period if—
[1.4 BIFSOPA level 2 (CDI)] the claimant and respondent have informed the adjudicator that they have agreed under subsection (1) that the adjudicator has additional time to decide the application; or
[1.4 BIFSOPA level 2 (CDI)] the application relates to a complex payment claim and, in the opinion of the adjudicator, the claimant and respondent have failed to reach an agreement mentioned in subsection (1). 
[1.3 BIFSOPA level 1 (CDI)] The longer period is—
[1.4 BIFSOPA level 2 (CDI)] if subsection (2)(a) applies—the additional time agreed to by the claimant and respondent under subsection (1); or
[1.4 BIFSOPA level 2 (CDI)] if subsection (2)(b) applies—5 business days after the time the adjudicator would otherwise have to decide the application under section 85(1).
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 86.1    Additional time to decide adjudication
[2.4 Com-BIFSOPA CDI Normal Body Text] In Galaxy Developments Pty Ltd v Civil Contractors (Aust) Pty Ltd t/a CCA Winslow & Ors [2020] QSC 51, Dalton J stated, in obiter, that the power of parties to extend the adjudicator’s time to decide an application under section 86(1) of BIFSOPA must be regarded as the legislature expressly giving the parties to an adjudication the “power to confer additional jurisdiction on the adjudicator in circumstances where he or she would not otherwise have jurisdiction, i.e. in circumstances where the original jurisdiction conferred had expired because no decision had been delivered within the statutory time.”

```
