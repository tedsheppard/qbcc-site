# Element-page brief — e-statutory-demand
**Title:** Statutory demand (insolvency pathway)
**Breadcrumb:** Enforcing an adjudicator's decision
**Anchor id:** `e-statutory-demand`
**Output file:** `bif_guide_build/v3/pages/page_e-statutory-demand.html`

## Extra guidance for this element
The statutory-demand pathway uses Corporations Act s 459E. The annotated s 93 commentary discusses the cases.

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

### `statute/chapter_3/section_093.txt`
```
# Source: BIF Act 2017 — Chapter 3
# Section 93 — Filing of adjudication certificate as judgment debt
(1) An adjudication certificate may be filed as a judgment for a
debt, and may be enforced, in a court of competent
jurisdiction.
Page 119 Current as at 27 April 2025

(2) An adjudication certificate can not be filed under this section
unless it is accompanied by an affidavit by the claimant
stating that the whole or a part of the adjudicated amount has
not been paid to the claimant at the time the certificate is filed.
(3) If the affidavit states that part of the adjudicated amount has
been paid, the judgment is for the unpaid part of the amount
only.
(4) If the respondent commences proceedings to have the
judgment set aside—
(a) the respondent is not, in those proceedings, entitled—
(i) to bring any counterclaim against the claimant; or
(ii) to raise any defence in relation to matters arising
under the construction contract to which the
adjudication certificate relates; or
(iii) to challenge the adjudicator’s decision; and
(b) the respondent is required to pay into the court, as
security, the unpaid portion of the adjudicated amount
pending the final decision in those proceedings.

```

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_093.txt`
```
# Annotated BIF Act source — Section 93
# Chapter: CHAPTER 3 – Progress payments
# Section title: Filing of adjudication certificate as judgment debt
# DOCX paragraphs: 3648-3739

[2 Com-BIFSOPA Heading 1] SECTION 93 – Filing of adjudication certificate as judgment debt 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 93    Filing of adjudication certificate as judgment debt 
[1.3 BIFSOPA level 1 (CDI)] An adjudication certificate may be filed as a judgment for a debt, and may be enforced, in a court of competent jurisdiction. 
[1.3 BIFSOPA level 1 (CDI)] An adjudication certificate can not be filed under this section unless it is accompanied by an affidavit by the claimant stating that the whole or a part of the adjudicated amount has not been paid to the claimant at the time the certificate is filed. 
[1.3 BIFSOPA level 1 (CDI)] If the affidavit states that part of the adjudicated amount has been paid, the judgment is for the unpaid part of the amount only. 
[1.3 BIFSOPA level 1 (CDI)] If the respondent commences proceedings to have the judgment set aside—
[1.4 BIFSOPA level 2 (CDI)] the respondent is not, in those proceedings, entitled—
[1.5 BIFSOPA level 3 (CDI)] to bring any counterclaim against the claimant; or 
[1.5 BIFSOPA level 3 (CDI)] to raise any defence in relation to matters arising under the construction contract to which the adjudication certificate relates; or 
[1.5 BIFSOPA level 3 (CDI)] to challenge the adjudicator’s decision; and 
[1.4 BIFSOPA level 2 (CDI)] the respondent is required to pay into the court, as security, the unpaid portion of the adjudicated amount pending the final decision in those proceedings.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 93.1    Filing of the adjudication certificate
[2.4 Com-BIFSOPA CDI Normal Body Text] Pursuant to the principles determined in Australian Broadcasting Corporation v O’Neill [2006] HCA 46; (2006) 227 CLR 57, where an applicant seeks an interlocutory injunction on the filing of an adjudication certificate as a judgment debt, the applicant must establish:
[2.6 Com-BIFSOPA bullet] a prima facie case; and
[2.6 Com-BIFSOPA bullet] that the balance of convenience favours the granting of an injunction.
[2.4 Com-BIFSOPA CDI Normal Body Text] The phrase ‘prima facie case’ in this context does not mean that it is more probable than not that the plaintiff will succeed at trial, but:
[2.5 Com-BIFSOPA Normal Body Quote] it is sufficient that the plaintiff show a sufficient likelihood of success to justify in the circumstances the preservation of the status quo pending the trial.
[2.4 Com-BIFSOPA CDI Normal Body Text] In deciding whether the balance of convenience favours the granting of an injunction, the legislative policy under the Act as to the balance of risk is a relevant factor: see R J Neller Building Pty Ltd v Ainsworth [2008] QCA 397; [2009] 1 Qd R 390; Wiggins Island Coal Export Terminal Pty Ltd v Sun Engineering (Qld) Pty Ltd [2014] QSC 170; BRB Modular Pty Ltd v AWX Constructions Pty Ltd & Ors [2015] QSC 222.
[2.4 Com-BIFSOPA CDI Normal Body Text] The leading Queensland authority on this point is R J Neller Building Pty Ltd v Ainsworth [2008] QCA 397; [2009] 1 Qd R 390 where the Court of Appeal had regard to the legislative policy, and moreover, to the risk that a builder might not be able to refund money ultimately found to be due to the owner, which the Court of Appeal considered legislature assigned to the owner. Per Keane JA (Fraser JA and Fryberg J agreeing):
[2.5 Com-BIFSOPA Normal Body Quote] It is evidently the intention of the BCIP Act, and, in particular, s 31 and s 100 to which reference has been made, that the process of adjudication established under that Act should provide a speedy and effective means of ensuring cash flow to builders from the parties with whom they contract, where those parties operate in a commercial, as opposed to a domestic, context. This intention reflects an appreciation on the part of the legislature that an assured cash flow is essential to the commercial survival of builders, and that if a payment the subject of an adjudication is withheld pending the final resolution of the builder's entitlement to the payment, the builder may be ruined.
[2.5 Com-BIFSOPA Normal Body Quote] The BCIP Act proceeds on the assumption that the interruption of a builder's cash flow may cause the financial failure of the builder before the rights and wrongs of claim and counterclaim between builder and owner can be finally determined by the courts. On that assumption, the BCIP Act seeks to preserve the cash flow to a builder notwithstanding the risk that the builder might ultimately be required to refund the cash in circumstances where the builder's financial failure, and inability to repay, could be expected to eventuate. Accordingly, the risk that a builder might not be able to refund moneys ultimately found to be due to a non-residential owner after a successful action by the owner must, I think, be regarded as a risk which, as a matter of policy in the commercial context in which the BCIP Act applies, the legislature has, prima facie at least, assigned to the owner.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Parkview Constructions Pty Limited v Total Lifestyle Windows Pty Ltd [2016] NSWSC 1911, Justice Slattery cited RJ Neller Building Pty Ltd v Ainsworth [2009] 1 Qd R 390 in determining whether to grant an interlocutory injunction restraining the claimant from applying for an adjudication certificate or filing any adjudication certificate that may have been obtained. The underlying issue was whether the Second Defendant (the adjudicator) erred in determining that the Plaintiff’s response was out of time under the NSW Act and thus denied the Plaintiff procedural fairness by not having regard to the Plaintiff’s response submissions during the adjudication process.
[2.4 Com-BIFSOPA CDI Normal Body Text] In determining that there was a serious question to be tried, Slattery J said:
[2.5 Com-BIFSOPA Normal Body Quote] In this case, it does seem to me that there is a serious factual issue to be tried. It is a simple issue, but one which will have decisive consequences for each side depending on how it is decided. If Parkview does succeed in showing that delivery of this application to it occurred on the 10th November, then it has a strong prospects of success. If on the other hand the adjudicator's determination is upheld as a matter of fact, then Windows has sound chances of success, notwithstanding the alternative legal arguments that have been deployed.
[2.4 Com-BIFSOPA CDI Normal Body Text] As to the balance of convenience, Slattery J held that:
[2.5 Com-BIFSOPA Normal Body Quote] In my view the balance of convenience favours the payment of these funds into Court rather than their payment to Windows in accordance with the filed adjudication certificate. This is principally because the factual issue in the case is so simple that it can very quickly be readied for hearing on both sides and be heard very early in the new term…
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour concluded that the applicant ought to be granted an interlocutory injunction restraining the respondent from filing an adjudication certificate procured as a result of the adjudication determination.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, R J Neller Building Pty Ltd v Ainsworth [2008] QCA 397; [2009] 1 Qd R 390 has been the subject of judicial consideration.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Low v MCC Pty Ltd & Ors; MCC Pty Ltd v Low [2018] QSC 6, Jackson J granted an injunction restraining MCC from enforcing two adjudication certificates, and enforcement warrants issued to enforce those certificates, on the basis that:
[2.6 Com-BIFSOPA bullet] there was a serious question to be tried as to whether the payment claim, the subject of each adjudication, was served in respect of a valid reference date; and 
[2.6 Com-BIFSOPA bullet] the balance of convenience favoured granting the injunction, particularly because Low had given an undertaking to lodge a bank guarantee with the registrar of the court as security for the unpaid portion of the adjudicated amount pending the final decision of the Court regarding the validity of the adjudicator’s decision.
[2.4 Com-BIFSOPA CDI Normal Body Text] The facts of the case where that there was doubt, with supporting evidence, as to whether the relevant milestone had been reached, and thus, whether the relevant reference date had accrued, with his Honour concluding that this constituted a serious question to be tried. In relation to the balance of convenience, his Honour held that it favoured granting the injunction, because of the circumstances of the case (including whether MCC could repay the adjudicated amount if overturned) and because of the fact that Low had given the undertaking the lodge the bank guarantee. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In Wiggins Island Coal Export Terminal Pty Ltd v Sun Engineering (Qld) Pty Ltd [2014] QSC 170, Daubney J considered R J Neller, noting that the decision was before the High Court’s judgment in Kirk v Industrial Court of New South Wales [2010] HCA 1; (2010) 239 CLR 531 on jurisdictional error. However, Daubney J concluded that nothing in Kirk detracted from the applicability of R J Neller and the legislative intention on the apportionment of risk.
[2.4 Com-BIFSOPA CDI Normal Body Text] This approach was followed in Sunshine Coast Regional Council v Earthpro Pty Ltd & Ors [2014] QSC 271, [15] (Peter Lyons J). In so doing, Peter Lyons J considered that the level of risk that the applicant might not be repaid was that of ‘substantial risk’: at [23].
[2.4 Com-BIFSOPA CDI Normal Body Text] However, the decision in Wiggins Island Coal Export Terminal Pty Ltd v Sun Engineering (Qld) Pty Ltd [2014] QSC 170 was doubted and not followed in BRB Modular Pty Ltd v AWX Constructions Pty Ltd & Ors [2015] QSC 222. Here, Bond J drew a distinction between valid adjudication decisions, and adjudication decisions affected by jurisdictional error. Bond J said that, a prima facie case that an adjudication decision is affected by jurisdictional error is relevant in deciding whether an interlocutory injunction is to be granted. This distinction was drawn in the following terms:
[2.5 Com-BIFSOPA Normal Body Quote] The intention of the Act – and in particular the intention revealed in ss 31 and 100 of the Act, to which Keane JA adverted in R J Neller at [39] – must be taken to be an intention revealed in relation to decisions of an adjudicator, made within jurisdiction.  The policy of the Act described by Keane JA in R J Neller only applies in its full force to valid adjudication decisions.  To my mind, it could not possibly be intended to be the policy of the Act that the risk is so passed in relation to decisions which are void and liable to be quashed by the Courts. 
[2.5 Com-BIFSOPA Normal Body Quote] It follows that I respectfully disagree with the approach taken by Daubney J in the WICET decision.  It seems to me that the appropriate analytical significance of the Act’s policy in cases such as the present is that the policy characterises the nature of the right that is in question in the interlocutory injunction application.  So if, on an interlocutory injunction application of this nature, the applicant has established a prima facie case that the adjudication decision is void for jurisdictional error, in considering the question which arises on the balance of convenience, in considering where the lower risk lies, it is appropriate to consider the fact that granting an injunction would deny to someone a right, the nature of which is that described by Keane JA in R J Neller.  The way in which this plays out in the exercise of the discretion is in the balancing process described by Hansen J in Kellogg Brown & Root Pty Ltd. 
[2.4 Com-BIFSOPA CDI Normal Body Text] A similar approach has been taken by the Supreme Courts of New South Wales and the Australian Capital Territory respectively, whereby, in circumstances where the applicant for injunctive relief has established a prime facie case (or that there is a serious issue to be tried) the Courts have been willing to grant an interlocutory injunction, but only on the condition that the amount in dispute be paid into court (or an agreed managed fund) by the applicant, pending final determination of the dispute: see Filadelfia Projects Pty Limited v EntirlTy Business Services [2010] NSWSC 473; Nazero Group Pty Limited v Top Quality Construction Pty Ltd [2015] NSWSC 232; Milburn Lake Pty Ltd v Andritz Pty Ltd [2016] VSC 3. Note that where a party is unable to presently make payment into court of the amount in dispute, an undertaking to do so may be sufficient for the court to grant an interlocutory injunction: see Filadelfia Projects Pty Limited v EntirlTy Business Services [2010] NSWSC 473.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Hakea Holdings Pty Limited v Denham Constructions Pty Ltd; BaptistCare NSW & ACT v Denham Constructions Pty Ltd [2016] NSWSC 1120, the Supreme Court of New South Wales reaffirmed their position when Ball J held that an injunction preventing the respondent from enforcing an adjudication determination should be continued. Ball J held that the applicants had established prima facie cases that they did not owe the amounts subject of the adjudication determination under their respective contracts and there was little prospect of  the applicants being able to recover any amount paid to the respondent due to its current and future solvency. His Honour took into account several factors, including the financial position of the respondent and the strength of the applicants’ cases and held that those considerations strongly favoured a stay and the continuation of the injunctions preventing the respondent from enforcing the adjudication determination. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Superb Build Pty Ltd v Petrosyan [2023] NSWDC 2 concerned an application to extend an interim stay on a judgment which gave effect to an adjudication determination. 
[2.4 Com-BIFSOPA CDI Normal Body Text] Andronos J referred to the factors that Ball J listed in Hakea Holdings Pty Ltd v Denham Constructions Pty [2016] NSWSC 1120 and held that the balance of convenience favoured the extension of the interim stay, and ultimately extended the interim stay to the date of the Related Proceedings. The factors Andronos J took into consideration in making his decision were:
[2.4 Com-BIFSOPA CDI Normal Body Text] the strength of the Respondent’s case; 
[2.4 Com-BIFSOPA CDI Normal Body Text] significant risk of irreparable financial harm to the Respondent if the stay was not extended; and
[2.4 Com-BIFSOPA CDI Normal Body Text] no evidence that the Claimant would become insolvent if the interim stay was extended.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Richard Crookes Constructions Pty Ltd v CES Projects (Aust) Pty Ltd [2016] NSWSC 1119 McDougall J held that the adjudicator had arguably failed to perform his statutory function of assessing the amount of construction work that had been performed and then valuing that work.  His Honour was satisfied that the plaintiff (contractor) had therefore made out that there was a prima facie case for the grant of relief and the balance of convenience favoured the grant of interlocutory injunctive relief restraining enforcement of the determination until there was a final hearing.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Bellerive Homes Pty Ltd v FW Projects Pty Ltd [2018] NSWSC 1435, N Adams J dismissed an application for a stay of a judgment for an adjudicated amount on the basis that would have been inconsistent with the underlying policy of the NSW Act (i.e. that successful applicants be paid promptly). In dismissing the stay application, N Adams J recognised the discretionary nature of granting such a stay, which is to be based on the facts of each case, and provided the following examples of where a stay may be granted:
[2.6 Com-BIFSOPA bullet] Failure to order a stay would “convert an amount which ought to be an interim payment into a final payment” (i.e. where there is a certainty that the party against who the stay is sought will suffer irreparable prejudice).
[2.6 Com-BIFSOPA bullet] A respondent can demonstrate that the result produced by the NSW Act is unjust and there is a substantial risk the money paid over would be irrecoverable and that proceedings for a final resolution are being pursued.
[2.6 Com-BIFSOPA bullet] The builder may have engaged in tactics calculated to delay the ultimate determination of the rights and liabilities of the parties so as to unfairly increase the owner’s exposure to the risk the builder’s insolvency.
[2.6 Com-BIFSOPA bullet] The builder may have restructured its finances after the making of the contract to increase the risk to the owner of the possible inability of the builder to meet its liabilities to the owner when they are ultimately declared by the courts.
[2.4 Com-BIFSOPA CDI Normal Body Text] Whilst N Adams J recognised that Bellerive’s financial records raised “some concern as to whether [Bellerive] has any future income stream”, his Honour did not consider that such concerns warranted the granting of the stay, nor did they fall within any of the examples provided (above) and therefore his Honour dismissed the stay.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Paul Kennedy McCullough trading as The Green Side Up v New Start Australia Pty Limited [2023] NSWDC 82, Dicker SC DCJ held that a court should not order a stay of the enforcement of an adjudication certificate, where “the whole purpose of the (Security of Payment) Act would… be circumvented if the adjudication certificate was not given proper force consistently with its registration in accordance with s 25” of the NSW SOP Act (equivalent to s 93 BIFSOPA), unless,  “failure to do so would [make] permanent that which the Act intended to be only interim”, and  “there is a likelihood of “irreparable prejudice”…”, referencing Veolia Water Solutions & Technologies (Australia) Pty Ltd v Kruger Engineering Australia Pty Ltd (No 3) [2007] NSWSC 459”.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Kuatro Build Pty Ltd v Elite Formwork Group Pty Ltd [2025] NSWSC 372, Hmelnitsky J held that although interlocutory orders had initially restrained enforcement under s 24(1)(a) the stay should not be continued (relevant to section 92(1) BIFSOPA). His Honour applied Martinus Rail Pty Ltd v Qube RE Services (No 2) [2024] NSWSC 1223 and Martinus [2025] NSWCA 49, reiterating that “considerable caution should attend the grant of such an injunction or a stay,” as it risks undermining the SOP Act’s policy of immediate cash flow. Elite’s financial evidence, including evidence of trading viability and tax repayment capacity, was unchallenged, and Kuatro’s insolvency concerns were insufficient to maintain the stay.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Fitz Jersey Pty Ltd v Atlas Construction Group Pty Ltd [2017] NSWCA 53, Beazley ACJ, Basten JA and Leeming JA considered section 25(4) of the BCIPA (NSW), equivalent to section 19 of the Act. Specifically, the Court considered if the section created a right to notify an affected party of a judgment debt before commencing proceedings to enforce, in circumstances where the resulting adjudication determination was filed as a judgment debt. Their Honour’s held a judgment can be enforced without notice to the party affected. This case involved no circumstances requiring the builder to notify the court of the proceedings commenced by the developer when applying for the garnishee order. In coming to this conclusion, Basten J reasoned at [70]:
[2.5 Com-BIFSOPA Normal Body Quote] Although s25(4) of the Act is said not to confer any right to have a judgment set aside, the developer’s claim was that it recognized an entitlement which otherwise existed to have the judgment set aside. It would be incorrect to suggest that s25 (4) put the respondent to a claim in any better position than an unsuccessful respondent against whom a judgment debt has been obtained in civil proceedings in a court.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Kennedy Civil Contracting Pty Ltd (subject to Deed of Company Arrangement) v Linx Constructions Pty Ltd [2024] NSWSC 366, Wright J held that the Respondent’s payment to a third party was to be taken into account in determining the “unpaid portion of the claimed amount” under s 15(2)(a)(i) of the SOP Act (equivalent to s 93(3) BIFSOPA). The Magistrate determined, as “simply a factual determination”, that the payment did reduce the amount outstanding under the payment claim in the circumstances. In doing so, Wright J noted that the words are not specifically defined and there was nothing to suggest they are used other than in their ordinary meaning. Wright J also rejected that such payment was not a payment “to the claimant” within the meaning of s 14(4) of the SOP Act (equivalent to s 77(2) BIFSOPA). Therefore, the Court held that amount of the unpaid portion was a question of fact or a mixed question of fact and law and refused to grant leave to appeal.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Kennedy Civil Contracting Pty Ltd v Linx Constructions Pty Ltd [2024] NSWCA 243, the New South Wales Court of Appeal held that a respondent’s payment to a third-party supplier did not constitute a cross-claim under section 15(4)(b)(ii) of the NSW Act, (equivalent to section 93 BIFSOPA). 
[2.4 Com-BIFSOPA CDI Normal Body Text] In this case, the respondent failed to issue a payment schedule in response to a $147,000 payment claim, which included approximately $78,000 for goods and services by a third-party supplier, however, it subsequently made payments totalling the amount of $102,800. The Claimant commenced proceedings to recover the unpaid portion. The Respondent argued that it had satisfied part of the debt by making a direct $30,000 payment to the third-party supplier, and that the debt should be reduced by that amount. 
[2.4 Com-BIFSOPA CDI Normal Body Text] The Magistrate found that based on the Claimant’s acknowledgment that the third-party supplier’s invoices were part of the claim, that the debt had been partially satisfied. Wright J of the Supreme Court upheld that finding, describing it as a “factual determination” rather than an impermissible defence. The Court of Appeal affirmed Wright J’s decision, and also confirmed that while the Act strictly limits defences in the absence of a payment schedule, reductions based on accepted third-party payments may be considered in limited, fact-specific circumstances.
[2.2 Com-BIFSOPA Heading 3] 93.2    Statutory demands
[2.4 Com-BIFSOPA CDI Normal Body Text] The current authority on the application of section 459H(1) of the Corporations Act 2001 (Cth) in the context of security of payment legislation is as per the decision in Re Douglas Aerospace Pty Ltd [2015] NSWSC 167 that:
[2.6 Com-BIFSOPA bullet] a genuine dispute is not founded on an application to set aside the debt; but
[2.6 Com-BIFSOPA bullet] an offsetting claim can exist on the presence of a counterclaim, set-off or cross demand that does not deny the debt, but asserts a countervailing liability.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, doubts about the correctness of the current approach were expressed by Robb J in Re J Group Constructions Pty Ltd [2015] NSWSC 1607.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Re J Group Constructions Pty Ltd [2015] NSWSC 1607, Robb J considered the line of authority from Diploma Construction (WA) Pty Ltd v KPA Architects Pty Ltd [2014] WASCA 91 and Re Douglas Aerospace Pty Ltd [2015] NSWSC 167. Upon a review of the authorities, Robb J followed the decision in Re Douglas Aerospace Pty Ltd [2015] NSWSC 167.However, despite following the decision in Re Douglas Aerospace Pty Ltd [2015] NSWSC 167, Robb J suggested that there are ‘reasons for doubting’ the applicability of the principles in Deputy Commissioner of Taxation v Broadbeach Properties Pty Ltd [2008] HCA 41; (2008) 237 CLR 473. These principles were considered and referred to in both Diploma and Re Douglas Aerospace. In particular, his Honour said that there is a question as to whether it is correct to apply the Broadbeach principles where the debts created by operation of the security of payment acts have a nature different to the tax debts that were the subject of the proceedings in Broadbeach.
[2.4 Com-BIFSOPA CDI Normal Body Text] In In the matter of Rockwall Homes Pty Limited [2017] NSWSC 223, Black J made reference to the reasoning of Brereton J in Re Douglas Aerospace Pty Ltd, when making a determination as to the submission that contended ss 15 (2) and (4) of the NSW Act did not prevent a company served with a creditors statutory demand from raising a genuine dispute for the purposes of setting aside that demand under section 459G of the Corporations Act. Specifically, his Honour made reference to paragraph [76] of the Brereton J’s reasoning:
[2.5 Com-BIFSOPA Normal Body Quote] some debts are not capable of genuine dispute — because they are judgment debts or conclusive debts — but may still be amenable to an offsetting claim. It is difficult to see how a debt arising under s 14(4) (“SOPA”), which creates a statutory liability upon the failure to provide a payment schedule, could be the subject of a genuine dispute, if the conditions in s 15(1) are satisfied, regardless of any underlying dispute. As it seems to me, the only way in which a “genuine dispute” could be raised in respect of such a debt would be by disputing whether the circumstances referred to in s 15(1) existed.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour found this reasoning was not applicable to the claim at hand, in that SOPA is only capable of applying where a payment claim is served upon a party to the construction contract and not exclude a dispute as to whether that requirement is satisfied. In coming to this conclusion, reference was nonetheless made to the reasoning of set out in CGI Information Systems & Management Consultants Pty Ltd v APRA Consulting Pty Ltd [2003] NSWSC 728, where Barrett J observed; 
[2.5 Com-BIFSOPA Normal Body Quote] “[T]he task faced by the company challenging a statutory demand on the genuine dispute grounds is by no means at all a difficult or demanding one. The company will fail in that task only if it is found, upon the hearing of its s 459G application, that the contentions upon which it seeks to rely in mounting its challenge are so devoid of substance that no further investigation is warranted. Once the company shows that even one issue has a sufficient degree of cogency to be arguable, a finding of genuine dispute must follow. The court does not engage in any form of balancing exercise between the strengths of competing contentions. If it sees any factor that on rational grounds indicates an arguable case on the part of the company, it must find that a genuine dispute exists, even where any case apparently available to be advanced against the company seems stronger.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Re Roberts Construction Group Pty Ltd [2024] VSC 679, Steffensen AsJ set aside a statutory demand in which a claimant issued in respect of the respondent’s failure to pay a scheduled amount under the Victorian SOP Act. Steffensen AsJ found the statutory demand insufficiently specified that the statutory debt arose was insufficiently described the legal basis of the debt, being one that due and payable under the Victorian SOP Act.  This is because the statutory demand did not refer to the ‘scheduled amount’, the document under which it was due, or the SOP Act. In coming to her decision, Steffensen AsJ referred to Re Three Pillars Lynbrook Pty Ltd [2022] VSC 540, where it was found a statutory demand is required to be unambiguous. Additionally, in CM Luxury Pty Ltd v Menzies Civil Australia Pty Ltd [2023] WASC 340, it was found that a statutory demand must accurately describe the legal basis of the debt, otherwise it can cause substantial injustice to the receiver of the demand. Therefore, statutory demands under the SOP Act must expressly state the basis of the demand for section 17 to apply.
[2.2 Com-BIFSOPA Heading 3] 93.3    Challenging an Adjudication Determination
[2.4 Com-BIFSOPA CDI Normal Body Text] In Amasya Enterprises Pty Ltd & Anor v Asta Developments (Aust) Pty Ltd & Anor [2015] VSC 233, Vickery J considered the application of section 28R(5)(a)(iii) of the VIC Act, the equivalent of section 31(4)(a)(iii) of the Act, in circumstances where the plaintiffs sought to challenge the adjudicator’s determination and held that:
[2.5 Com-BIFSOPA Normal Body Quote] … if the Plaintiffs are successful in a challenge to the Adjudicator’s Determination founded upon jurisdictional error, the Supreme Court may make an order in the nature of certiorari or a declaration to remove the legal consequences or purported legal consequences of the Adjudication Determination under the Act, with the result that the judgment founded upon it, must also be set aside.
[2.4 Com-BIFSOPA CDI Normal Body Text] In arriving at that conclusion, his Honour stated:
[2.5 Com-BIFSOPA Normal Body Quote] …the operation of the privative clause in s 28R(5)(a)(iii) is confined to denying relief being granted by a court in Victoria, including the Supreme Court, in the course of a proceeding to set aside a judgment entered pursuant to s 28R, where the error relied upon is an error on the face of the record in an adjudication determination which is the foundation of the judgment. In other words, pursuant to s 28R(5)(a)(iii) of the Act, it is not open to challenge an adjudication determination (or a review determination) in a proceeding to have a s 28R judgment set aside, on the basis or an error on the face of the record in the relevant determination.
[Normal] It follows in this case that, if the Plaintiffs are able to establish a jurisdictional error in the Adjudication Determination, they are not precluded by the operation of s 28R(5)(a)(iii) of the Act from challenging the Adjudication Determination on that basis.
[2.2 Com-BIFSOPA Heading 3] 93.4    Section 93(4)(a) – Respondent is not entitled to counterclaim, defence, and chanllenge adjduicator’s decision
[2.4 Com-BIFSOPA CDI Normal Body Text] In Infinity Formwork Group Pty Ltd v Adobe Construction Group Pty Ltd [2023] NSWDC 220, the Court considered whether an equitable setoff fell within the scope of section 15(4)(b)(ii) of the NSW SOP Act (equivalent s 93(4)(a)(ii) BIFSOPA) and was excluded as an option available to the respondent in challenging the claimant’s application for summary judgment. In this case, the respondent failed to provide a payment schedule in response to a payment claim, and the claimant sought summary judgment for the amount claimed. Adronos SC DCJ held that equitable setoff was within the scope of the exclusion from raising a defence in section 15(4) and was not available to the respondent. In reaching this conclusion, Adronos SC DCJ stated that it was “incumbent on the defendant, if it wished to rely on the defence, to raise it in a payment schedule”. The application for summary judgment was successful, with the respondent being liable for the full amount under the payment claim.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Marques Group Pty Ltd v Parkview Constructions Pty Ltd [2023] NSWSC 625, Rees J considered a respondent’s defence to an application for summary judgment made under section 16(2)(a)(i) of the NSW SOP Act, for failure to pay a scheduled amount. The Court noted that section 15(4)(b) and 16(4)(b) (equivalent to section 93(4)(a) BIFSOPA), precludes a respondent from bringing a crossclaim or raising a defence to matters arising under the construction contract. Rees J applied Bitannia Pty Ltd v Parkline Constructions Pty Ltd (2006) 67 NSWLR 9 and confirmed that a misleading and deceptive conduct argument under the ACL is not a defence in relation to matters under the construction contract and therefore the respondent was not precluded from making their misleading and deceptive conduct argument. Ultimately, the Court held that it was “arguable” that the defence could be made out, and therefore the claimant’s application for summary judgment was rejected.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Kennedy Civil Contracting Pty Ltd (subject to Deed of Company Arrangement) v Linx Constructions Pty Ltd [2024] NSWSC 366, Wright J held that the Respondent’s reliance on a payment to a third party to reduce the “unpaid portion of the claimed amount” did not amount to an impermissible cross-claim within s 15(4)(b)(i) of the SOP Act (equivalent to s 93(4)(a)(i) BIFSOPA). At first instance, the Magistrate rejected the Claimant’s contention that the reliance on the payment amounted to a cross-claim. The Magistrate also found that the Claimant should not be entitled to be paid an amount it never intended to retain.
[2.4 Com-BIFSOPA CDI Normal Body Text] In upholding the Magistrate’s conclusion, the Court emphasised that the Respondent had not advanced a claim for independent or additional relief that could be given in separate proceedings, which is inherent in the concept of a cross-claim. The payment was merely a factual contention going towards what level of debt was outstanding under the payment claim.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Harlech Enterprises Pty Ltd v Beno Excavations Pty Ltd t/as Benex Pipeline [2024] NSWDC 151, Montgomery DCJ considered whether to set aside a judgement and subsequent garnishee order entered into the court in early 2021. The judgement and garnishee order related to an adjudication determination that was subsequently quashed by the ACT Supreme Court on the basis of jurisdictional error in Beno Excavations Pty Ltd v Harlech Enterprises (No 2) [2021] ACTSC 269 per Mossop J.
[2.4 Com-BIFSOPA CDI Normal Body Text] In this case, Montgomery DCJ affirmed the decision in Brodyn Pty Ltd t/as Time Cost and Quality v Davenport & Anor [2004] NSWCA 394 citing [61] of that decision: “…until it [adjudication determination] is filed, proceedings can appropriately be brought in a court with jurisdiction to grant declarations and injunctions to establish that it is void and to prevent it being filed. However, once it has been filed, the resulting judgement is not void. An application can be made to set aside the judgement… it is not contrary to s 25(4)(a)(iii) to do so on the basis that there is in truth no adjudicator’s determination”. Accordingly, Montgomery DCJ found that it was not contrary to s 25(4)(a)(iii) of the SOP ACT (NSW) [equivalent to BIFSOPA s 93(4)(iii)] to do so, because the statutory scheme of the ACT Act did not contemplate the enforcement of a judgement debt where the determination was infected with jurisdictional error. As such, His Honour ordered that the judgement and garnishee order be set aside.
[2.4 Com-BIFSOPA CDI Normal Body Text] In BRP Industries Pty Ltd v Hynash Constructions Pty Ltd [2024] NSWDC 392, a Claimant applied to the NSW District Court for summary judgment for $234,286.80 in respect of a payment claim which the Respondent failed to provide a payment schedule. In its defence, the Respondent pleaded that it had paid $110,756.80 as an upfront payment, and that the amount it was liable for was in fact the difference between $234,286.80 and $110,756.80. The issue before the NSW District Court was whether the Respondent’s argument was a defence “in relation to matters arising under the construction contract” and therefore meant that the Respondent was barred from making that argument under section 15(4)(b)(ii) of the Building and Construction Industry Security of Payment Act 1999 (NSW) (equivalent BIFSOPA section 93(4)(ii)). 
[2.4 Com-BIFSOPA CDI Normal Body Text] Russell SC DCJ held that the set-off asserted by the Respondent could not be raised to reduce the amount of the Payment Claim, because it did in fact relate to a matter under the contract. This was because the contract provided that the Respondent was to pay an upfront 10% establishment and mobilisation fee. The upfront payment of $110,756.80, was the 10% establishment and mobilisation fee. As such, His Honour held that the Respondent sought to raise a defence by way of set-off in relation to a matter arising under the construction contract, as the obligation to make the upfront payment arose under the contract.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Hynash Constructions Pty Ltd v BRP Industries Pty Ltd [2025] NSWCA 14, the NSW Court of Appeal considered the scope of section 15(4)(b)(ii) of the NSW Act (equivalent to section 93(4)(a)(ii) BIFSOPA), which prohibits a respondent from raising a defence in relation to matters arising under the construction contract in proceedings brought by a claimant for an “unpaid portion” of a payment claim. In this case, the Respondent sought to argue it was entitled to reimbursement of an up-front payment that it made to the Claimant, because it was partial payment of the debt which arose from the payment claim, and therefore the “unpaid portion” to the Claimant amount ought to be reduced by the up-front payment. The Court of Appeal dismissed this argument and held that the “unpaid portion” in section 15(2)(a)(i) refers to amounts unpaid in response to the payment claim, and not unrelated prior payments. If the respondent wanted a prior payment credited, it must have done so raised in the payment schedule, which it failed to do.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Martinus Rail Pty Ltd v Qube RE Services (No.2) Pty Ltd [2025] NSWCA 49, Payne JA confirmed that entry of judgement under section 25 (equivalent to section 93(1) BIFSOPA) is not precluded by a pending arbitration or by the mere possibility of repayment risk. The Court upheld the primary of the SOP Act’s enforcement mechanism, rejecting Qube’s argument that enforcement should be stayed pending resolution of a separate arbitration. The Court emphasised that the adjudicated amounts became payable by the statutory due date and that “the grant of a stay until the end of arbitration would be contrary to the statutory purpose of the SOP Act.” The Court applied Kennedy Civil Contracting Pty Ltd v Richard Crookes Construction Pty Ltd [2023] NSWSC 99 and Probuild in affirming the swift enforcement intended under s 25.

```
