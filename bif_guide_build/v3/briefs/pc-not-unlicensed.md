# Element-page brief — pc-not-unlicensed
**Title:** Not for unlicensed work
**Breadcrumb:** Requirements of a payment claim
**Anchor id:** `pc-not-unlicensed`
**Output file:** `bif_guide_build/v3/pages/page_pc-not-unlicensed.html`

## Extra guidance for this element
The relevant rule is QBCC Act s 42 (unlicensed work cannot be the subject of a recoverable claim). Discuss how this interacts with a BIF Act payment claim. Source the cases from the annotated s 75 commentary on illegality.

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

## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)

### `annotated/section_075.txt`
```
# Annotated BIF Act source — Section 75
# Chapter: CHAPTER 3 – Progress payments
# Section title: Making payment claim
# DOCX paragraphs: 2014-2259

[2 Com-BIFSOPA Heading 1] SECTION 75 – Making payment claim 
[2.1 Com-BIFSOPA Heading 2] A    Legislation
[1 BIFSOPA Heading] 75    Making payment claim 
[1.3 BIFSOPA level 1 (CDI)] A person (the claimant) who is, or who claims to be, entitled to a progress payment may give a payment claim to the person (the respondent) who, under the relevant construction contract, is or may be liable to make the payment. 
[1.3 BIFSOPA level 1 (CDI)] Unless the payment claim relates to a final payment, the claim must be given before the end of whichever of the following periods is the longest—
[1.4 BIFSOPA level 2 (CDI)] the period, if any, worked out under the construction contract; 
[1.4 BIFSOPA level 2 (CDI)] the period of 6 months after the construction work to which the claim relates was last carried out or the related goods and services to which the claim relates were last supplied. 
[1.3 BIFSOPA level 1 (CDI)] If the payment claim relates to a final payment, the claim must be given before the end of whichever of the following periods is the longest— 
[1.4 BIFSOPA level 2 (CDI)] the period, if any, worked out under the relevant construction contract;
[1.4 BIFSOPA level 2 (CDI)] 28 days after the end of the last defects liability period for the construction contract; 
[1.4 BIFSOPA level 2 (CDI)] 6 months after the completion of all construction work to be carried out under the construction contract;
[1.4 BIFSOPA level 2 (CDI)] 6 months after the complete supply of related goods and services to be supplied under the construction contract. 
[1.3 BIFSOPA level 1 (CDI)] The claimant can not make more than 1 payment claim for each reference date under the construction contract.
[1.3 BIFSOPA level 1 (CDI)] A payment claim may include an amount that was included in a previous payment claim. 
[1.3 BIFSOPA level 1 (CDI)] In this section— 
[1.1 BIFSOPA Body Text] final payment means a progress payment that is the final payment for construction work carried out, or for related goods and services supplied, under a construction contract.
[2.1 Com-BIFSOPA Heading 2] B    Commentary
[2.2 Com-BIFSOPA Heading 3] 75.1    Changes introduced to section 75
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 75 creates the right to make a payment claim and incorporates relevant commentary from section 17 and section 17A of BCIPA. Commentary on the requirements for a valid payment claim appear under section 68 of the Act.
[2.2 Com-BIFSOPA Heading 3] 75.2    A requirement of good faith?
[2.4 Com-BIFSOPA CDI Normal Body Text] In Neumann Contractors Pty Ltd v Traspunt No 5 Pty Ltd [2010] QCA 119; [2011] 2 Qd R 114, the Court of Appeal rejected a submission that a payment claim must be made in good faith. Muir JA said:
[2.5 Com-BIFSOPA Normal Body Quote] …The Act enables the respondent to a payment claim to serve a payment schedule and for the payment claim and payment schedule to go to adjudication.  The adjudicator addresses and determines the merits of the parties' dispute as articulated in the payment claim and payment schedule.  No enquiry into the claimant's bona fides is mentioned and it is difficult to see why a claimant's bona fides should be any more fundamental to a valid payment claim than a plaintiff's bona fides would be in relation to a valid writ, claim, or statement of claim.
[2.5 Com-BIFSOPA Normal Body Quote] In Bitannia, Basten JA, noted that, "… the beliefs or motivations of the plaintiff in proceedings have generally been treated as irrelevant, unless they reach the stage of an improper purpose" which would constitute an abuse of process.  An abuse of process arises where proceedings are brought, not for the purpose of prosecuting them to a conclusion, but to use them as a means of obtaining some advantage for which they are not designed, or some collateral advantage beyond that offered by the law. The "improper purpose" must be the predominant purpose.
[2.4 Com-BIFSOPA CDI Normal Body Text] A similar conclusion on good faith was reached by Vickery J in 470 St Kilda Road Pty Ltd v Reed Constructions Australia Pty Ltd [2012] VSC 235, [43]-[44].
[2.4 Com-BIFSOPA CDI Normal Body Text] In the above extract from Neumann Contractors, Muir JA referred to the case of Bitannia Pty Ltd & Anor v Parkline Constructions Pty Ltd [2006] NSWCA 238; (2006) 67 NSWLR 9, where it was held by the New South Wales Court of Appeal that there is no requirement that a payment claim be made in a bona fide belief in the truth of the facts asserted. In The New South Wales Netball Association Ltd v Probuild Construction (Aust) Pty Ltd [2015] NSWSC 1339, Stevenson J held that Bitannia: 
[2.5 Com-BIFSOPA Normal Body Quote] does not support the proposition that the service of a payment claim involves the making of representations as to the truth of the matters contained in the payment claim.
[2.2 Com-BIFSOPA Heading 3] 75.3    Section 75(1) – ‘to the person who, under the relevant construction contract, is or may be liable’
[2.3 Com-BIFSOPA Heading 4] To the person…liable to make the payment
[2.4 Com-BIFSOPA CDI Normal Body Text] There is differing authority on who a ‘person…liable to make the payment’ may be: see The Owners Strata Plan 56597 v Consolidated Quality Projects [2009] NSWSC 1476; Metacorp Australia Pty Ltd v Andeco Construction Group Pty Ltd (2010) 30 VR 141; [2010] VSC 199; Penfold Projects Pty Ltd v Securcorp Limited [2011] QDC 77.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples
[2.6 Com-BIFSOPA bullet] Service of a payment claim on a quantity surveyor as agent of the Principal. Abigroup Contractors Pty Ltd v River Street Developments Pty Ltd [2006] VSC 425. The construction contract required claims for payment to be delivered to the quantity surveyor and the superintendent. The contract provided that both the quantity surveyor and the principal act as agent of the principal. In an application for summary judgment, the applicant (the claimant) submitted that the VIC Act did not displace the common law principles of agency for service of payment claims. Habersberger J considered that there was a serious question to be tried in relation to a number of issues, including service of the payment claim.
[2.6 Com-BIFSOPA bullet] Service on a superintendent with a prior course of dealing. The Owners Strata Plan 56597 v Consolidated Quality Projects [2009] NSWSC 1476. The contract required service of progress claims to the superintendent. Before the claim in question, the claimant had served 25 progress claims to the superintendent under the contract, each being endorsed as a payment claim. The superintendent was not stated as being the agent of the principal under the contract, but was to act ‘reasonably and in good faith’. The Court held that the parties ‘appear to have engaged a regime whereby documents purporting to be both progress claims under the contract and payment claims under the Act were delivered, as one document, to the [respondent] in care of [the superintendent]’ (at [31]).
[2.6 Com-BIFSOPA bullet] Service on a superintendent as agent of the principal. Metacorp Pty Ltd v Andeco Construction Group Pty Ltd [2010] VSC 199. The payment claim in question (payment claim 15) was served on the superintendent. The contract defined the superintendent as the agent of the principal. There was evidence that each prior payment claim had been served in the same manner as payment claim 15, to the superintendent by email. The Court held that the superintendent had the ostensible authority of the principal to receive service of payment claims under the VIC Act. The Court held that the superintendent received payment claim 15 as a person who ‘is or may be liable’.
[2.6 Com-BIFSOPA bullet] Service on a project manager as agent of the principal. Penfold Projects Pty Ltd v Securcorp Limited [2011] QDC 77. A payment claim was served on the construction manager, who was stated under the contract to be the agent of the principal, except where the contract otherwise provides. Irwin DCJ distinguished both Abigroup and Owners Strata and held that, in the absence of an express contractual provision, service on the construction manager was not valid service under BCIPA.
[2.6 Com-BIFSOPA bullet] In Canadian Solar Construction Pty Ltd v Re Oakey Pty Ltd [2023] QSC 288, Freeburn J found that a payment claim had been effectively served and given to the Respondent via its project manager and other senior project personnel. His Honour also declared that the objective of Mode of Service clauses is ‘to facilitate service, and not to restrict the means by which service might be effected’.
[2.6 Com-BIFSOPA bullet] In this case, an issue was whether service of the payment claim via email had come to the attention of the Respondent. The payment claim was issued by email to the director of the Respondent as well as five individuals involved in the project. The director of the Respondent did not receive the email. However, the five individuals did receive the email, and four of them were representatives of the Project Manager/Superintendent who were able to exercise all the functions of the Respondent. Freeburn J concluded that the payment claim had been given to the Respondent on the basis that the payment claim was received by five of the six recipients and that four out of them had authority to assess payment claim.
[2.6 Com-BIFSOPA bullet] His Honour agreed with the reasoning in BCFK Holdings Pty Ltd v Rork Projects Pty Ltd [2022] NSWSC 1706. That is, ‘a party that actually receives a payment claim should not be entitled to assert that service did not ever happen because of a shortcoming, perhaps technical, in the manner in which the claimant purported to effect service’.  
[2.3 Com-BIFSOPA Heading 4] Under the construction contract
[2.4 Com-BIFSOPA CDI Normal Body Text] Under section 75(1) of the Act, the ‘respondent’ must be liable ‘under the relevant construction contract’: see, re the NSW Act, Grave v Blazevic Holdings Pty Ltd (2010) 79 NSWLR 132; [2010] NSWCA 324 at [26]-[28] per McDougall J (Allsop P and Macfarlan JA agreeing):
[2.5 Com-BIFSOPA Normal Body Quote] The person on whom the progress claim may be served is someone who, under the construction contract concerned, is or may be liable to make the payment. The words “is or may be liable” may be capable of referring, for example, to primary or secondary liability (the latter, for example, as guarantor). They may be capable of encompassing disputes as to quantification; and indeed, disputes as to whether, on the proper construction of the contract and the Security of Payment Act, any payment is due at all. However, whatever is the nature and amount of the liability sought to be enforced, it must be a liability “under the construction contract concerned”. If the proposed recipient of the payment claim is not a party to or liable under the construction contract, then it falls outside that statutory description.
[2.5 Com-BIFSOPA Normal Body Quote] Section 14 deals with the way in which the recipient of a payment claim may dispute, in whole or in part, its liability. The consequences of failure to take advantage of the dispute process are spelled out in s 14(4). The respondent to the payment claim “becomes liable to pay the claimed amount to the claimant on the due date for the progress payment to which the payment claim relates”.
[2.5 Com-BIFSOPA Normal Body Quote] The person referred to in s 14(4) as “the respondent” is, by reference to subs (1), the person on whom the payment claim is served. That in turn directs attention back to s 13(1). The person must be someone who under the construction contract is or may be liable to make the payment. I do not think that it is the correct construction of the statutory liability that may come into existence pursuant to s 14(4) that it extends beyond those who, by s 13(1), are denoted as being susceptible to that liability.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Collective Crane Hire Pty Ltd v ICR Steel Pty Ltd [2021] VCC 132, Justice Woodward considered McDougall J’s decision in Grave v Blazevic Holdings [2011] NSWSC 287, where His Honour held that the context of section 13 of the NSW SOP Act (equivalent to section 75 of the BIF Act) “makes it clear that liability is enforced through the mechanism of judgment and the starting point is that it arises under a construction contract.” Woodward J held that: 
[2.4 Com-BIFSOPA CDI Normal Body Text] Grave is not authority for the proposition that, “only by serving a payment claim on the person who, is in the strict or formal sense, the ‘other party’ to the construction contract” will a claim be valid; 
[2.4 Com-BIFSOPA CDI Normal Body Text] “the gravamen of Graves is that the person on whom a payment claim is served must be someone who is not a stranger to the contract”; and 
[2.4 Com-BIFSOPA CDI Normal Body Text] the appropriate approach is that any person who, under a construction contract, is at least arguably likely to be liable to make payment under the payment claim, may potentially be liable within the meaning of section 14 of the Vic Act (equivalent to section 75 of the BIF Act).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Nefiko Pty Ltd v Statewide Form Pty Ltd (No 2) [2014] NSWSC 840, Ball J held that conduct subsequent to the formation of the contract may be taken into account in determining who the parties to the contract were.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Martinus Rail Pty Ltd v Qube RE Services (No.2) Pty Ltd [2025] NSWCA 49, Payne JA did not directly dispute the validity of the payment claims under s 13 but confirmed that the adjudicator was entitled to determine the scope and content of both the Payment Claim and Payment Schedule (relevant to section 82(2) BIFSOPA). The Court noted that Martinus’s claims arose after the termination of the contracts and involved significant claims for variations and post-termination work. Payne JA held that the adjudicator’s approach in assessing the payment claims- despite their complexity- was within jurisdiction and consistent with the object of the Act: to allow contractors to receive progress payments promptly. The Court followed John Holland Pty Ltd v Roads and Traffic Authority of New South Wales [2007] NSWCA 19 and Icon Co (NSW) Pty Ltd v Australia Avenue Developments Pty Ltd [2018] NSWCA 339 in recognising that the scope of the claim is a matter for the adjudicator to assess based on the material before them.

[2.3 Com-BIFSOPA Heading 4] Is or may be liable
[2.4 Com-BIFSOPA CDI Normal Body Text] The statutory right under section 70 to serve a payment claim under section 75(1), does not extend the right to serve a payment claim on a person ‘who “may” be a party’: see Mansouri & Anor v Aquamist Pty Ltd [2010] QCA 209, [13] where the Queensland Court of Appeal said:
[2.5 Com-BIFSOPA Normal Body Quote] Read in light of the definition of “construction contract”, s 12 of BCIPA gives the statutory right to a progress payment only to a person who “undertakes to carry out construction work for”, or to supply related goods and services to, another party under a contract, agreement or other arrangement. The provisions in Part 3 were designed to facilitate the enforcement of that statutory right, not to broaden it. Such a right is not established by proof only that a payment claim was served on a person who “may” be a party for whom the claimant undertook to perform construction work or to whom the claimant undertook to supply related goods and services. Rather, the effect of the expression in s 17(1) “under the construction contract concerned”, again read in light of the definition of “construction contract”, is that a payment claim may validly be served only on a person who is a party for whom the claimant undertook to perform construction work, or to whom the claimant undertook to supply related goods and services. 
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court of Appeal held that ‘may be liable’ refers to the claimant’s entitlement for progress payments, and not a ‘liberalisation’ of who may be the respondent:
[2.5 Com-BIFSOPA Normal Body Quote] The expression “may be liable” does not qualify that requirement. It serves a different purpose. Progress payments under BCIPA are made on account of a liability which may or may not ultimately be established. That is so under conventional building contracts and s 100 makes it clear that the same is true under BCIPA. It was therefore necessary for s 17(1) of BCIPA to include a person “who claims to be entitled” to a progress payment within the description of persons entitled to serve a payment claim and correspondingly to include a person who “may be liable” within the description of persons upon whom a payment claim might be served. Neither expression suggests any liberalisation of the requirement that “the respondent” must be a person for whom the claimant undertook to carry out construction work or to whom the claimant undertook to supply related goods and services. The entitlement to serve a payment schedule under s 18 is conferred only upon such a person and the remedies in s 19(2) are given only against such a person. So much is consistent also with s 19(4)(a), which requires the court to be satisfied that “the respondent” has become liable under s 18, and with s 19(4)(b)(ii), which operates to preclude defences which “the construction contract” otherwise might have provided to “the respondent”. 
[2.2 Com-BIFSOPA Heading 3] 75.4    Section 75(1) – ‘A person…who is or who claims to be entitled’
[2.3 Com-BIFSOPA Heading 4] A ‘claimed’ entitlement to a progress payment
[2.4 Com-BIFSOPA CDI Normal Body Text] Section 75(1) refers to a person “who is or who claims to be entitled to a progress payment” (emphasis added).
[2.4 Com-BIFSOPA CDI Normal Body Text] It was submitted on behalf of the claimant in F.K. Gardner & Sons Pty Ltd v Dimin Pty Ltd Ltd [2007] 1 Qd R 10; [2006] QSC 243 that, under section 17(1) of BCIPA, “there need not be an enquiry as to actual entitlement but simply an assertion of entitlement by a claim and this then extends to who may make a payment claim”. This submission was rejected by Lyons J, who held that section 17(1) of BCIPA, in conferring a right to “[a] person mentioned in section 12”, refers “to a person who is entitled under s. 12 to a progress payment or claims an entitlement under s. 12” (equivalent to section 70 of the Act).
[2.4 Com-BIFSOPA CDI Normal Body Text] A broader approach has been taken in Victoria under the VIC Act. Referring to the difference to the former NSW Act as in force in December 2002, which did not contain the phrase ‘who claims to be entitled’, in Metacorp Pty Ltd v Andeco Construction Group Pty Ltd [2010] VSC 199, Vickery J held:
[2.5 Com-BIFSOPA Normal Body Quote] In contrast, under the legislation as it now stands, the class of persons who may serve a payment claim has been extended to include persons ‘who claim to be entitled’ to a progress payment, in addition to those who may actually be so entitled. In my view, provided that a person makes a claim to be entitled to a progress payment, and that claim is made bona fide, the claimant is permitted to serve its payment claim pursuant to s. 14(1) of the Victorian Act, and this is so, whether or not there existed an actual entitlement to payment at the time when the payment claim was served.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour followed this approach again in Seabay Properties Pty Ltd v Galvin Construction Pty Ltd [2011] VSC 183.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, a similar approach to that of Lyons J in Dimin has been taken in New South Wales (regarding section 8 of the NSW Act). In Southern Han Breakfast Point Pty Ltd v Lewence Construction Pty Ltd [2015] NSWSC 502, Ball J, referring to the introduction of the phrase ‘or who claims to be’, considered that:
[2.5 Com-BIFSOPA Normal Body Quote] A person may claim to be entitled to a progress payment but not actually be entitled to such a payment for a variety of reasons. The claim may not be under a construction contract. The work in respect of which the claim is made may not be for construction work (or for related goods and services). A reference date may not have arisen under the contract. The work may not have been done. The claimant may not be entitled to be paid for that work under the contract (because, for example, the contract is a lump sum contract and the work is not properly work done under a variation to the contracted-for work). The claimant may already have been paid for the work. No doubt, there are other examples. Contrary to Lewence's submission, it is not correct to say that the words “or claims to be” have no work to do if the phrase “[a] person referred to in section 8(1)” is interpreted broadly to include a person having the right to make a progress claim because that person satisfies the requirements of s 8(1). The words “or claims to be” would still cover cases where, for example, a person claims to have done work that has not been done, or claims to be entitled to be paid for work where no such entitlement exists or where payment has already been made.
[2.5 Com-BIFSOPA Normal Body Quote] Section 8 of the Act sets out the essential requirements that a person must satisfy in order to become entitled to a progress payment. However, it is concerned with that right at an abstract level. Whether a claimant is actually entitled to a payment depends on the particular work that has been done and the terms of the relevant construction contract. Section 13 must be interpreted as saying that a person who meets the essential requirements set out in s 8 is entitled to make a progress claim and other sections of the Act set out how that claim is to be resolved. The entitlement to make a claim does not depend on the success or otherwise of the claim. But it does depend on satisfying the essential requirements. Section 13(1) uses the words “or claims to be” to address the first of these points. It uses the words “[a] person referred to in section 8(1)” to address the second. It is apparent from the wording of s 8 that the occurrence of a reference date is as essential as the existence of a construction contract and the performance of construction work or the supply of related goods and services under that contract.
[2.4 Com-BIFSOPA CDI Normal Body Text] Ball J considered that there were two possible interpretations of the phrase ‘[a] person referred to in section 8(1)’ under section 13(1) of the NSW Act (relevantly, ‘[a] person mentioned in section 70’ is the equivalent phrase under section 75(1) of the Act):
[2.6 Com-BIFSOPA bullet] that a person has undertaken to carry out construction work under a construction contract or to supply related goods and services under a construction contract; or
[2.6 Com-BIFSOPA bullet] that a person has undertaken to carry out construction work under a construction contract or to supply related goods and services under a construction contract and a reference date has arisen in respect of such.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour preferred the latter approach.
[2.4 Com-BIFSOPA CDI Normal Body Text] On appeal, in Lewence Construction Pty Ltd v Southern Han Breakfast Point Pty Ltd [2015] NSWCA 288. the New South Wales Court of Appeal preferred the former approach. On this point, Sackville AJA concluded that:
[2.5 Com-BIFSOPA Normal Body Quote] It is a strained interpretation of the introductory words to s 13(1) of the BCI Act (“[a] person referred to in section 8(1) who is or who claims to be entitled to a progress payment”) to read them as referring to a claimant who not only satisfies either sub-par (a) or sub-par (b) of s 8(1), but who is also able to show that the relevant reference date under the construction contract has in fact arrived. The very point of the procedure created by Pt 3 of the BCI Act is to establish a mechanism, in the event of a dispute, for an adjudicator to determine precisely that question.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Canberra Drilling Rigs Pty Ltd v Haides Pty Ltd [2019] ACTCA 15 the Court of Appeal held, consistent with the primary judge’s determination, that the language of section 15 of the ACT Act (equivalent to section 75 of the Act) means that liability under a construction contract is something which may be claimed rather than established as a jurisdictional fact. While the existence of a construction contact is a jurisdictional fact made clear in the language of section 15(4)(a) of the ACT Act, an assertion that the work was performed under the construction contract can be distinguished and is not a preliminary jurisdictional issue (at [39]). The court believed this conclusion accorded with “rough and ready” scheme, by ensuring the avoidance of contested questions of contractual interpretation before an adjudicator.
[2.3 Com-BIFSOPA Heading 4] Early payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] In F.K. Gardner & Sons Pty Ltd v Dimin Pty Ltd Ltd [2007] 1 Qd R 10; [2006] QSC 243, Lyons J held that, where the relevant contract made provision for reference dates, and the payment claim in question was submitted prior to the contractual reference date, the would-be claimant had no statutory entitlement under section 17 of BCIPA to submit a payment claim.
[2.4 Com-BIFSOPA CDI Normal Body Text] Again, a broader approach has been taken in Victoria under the VIC Act. In Metacorp Pty Ltd v Andeco Construction Group Pty Ltd [2010] VSC 199, on the basis of a broader approach to the phrase ‘who claims to be’, Vickery J held an early payment claim to be valid, considering that:
[2.5 Com-BIFSOPA Normal Body Quote] A payment claim which is delivered shortly prior to its reference date, even a few days before, would not, in the usual case, evidence lack of bona fides on the part of the person making the claim because the work carried out in respect of which the claim is made in all likelihood would have been done, or substantially completed.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour followed this approach again in Seabay Properties Pty Ltd v Galvin Construction Pty Ltd [2011] VSC 183.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Cool Logic Pty Ltd v Citi-Con (Vic) Pty Ltd [2020] VCC 1261, Woodward J held that where a payment claim is served shortly before a reference date, the claimant cannot argue that the payment claim is valid on the basis that it was issued in good faith. Woodward J held that such position is now limited by the findings of Digby J in Southern Han, and by the application of those findings in All Seasons and MKA Bowen. Relevantly, the majority of the NSWCA in All Seasons applied Southern Han and held that: 
[2.5 Com-BIFSOPA Normal Body Quote]  “[A] statutory payment claim served before the relevant reference date is not served ‘on or from’ that reference date for the purpose of the NSW SOP Act, the effect being that the phrase ‘on or from’ when properly construed, is to be understood as meaning ‘on or after’ the relevant reference date.”
[2.2 Com-BIFSOPA Heading 3] 75.5    The identification of an improper claimant
[2.4 Com-BIFSOPA CDI Normal Body Text] In G W Enterprises Pty Ltd v Xentex Industries Pty Ltd & Ors [2006] QSC 399, it was alleged by the applicant that the payment claim in question was not valid because it had misdescribed the correct entity that had carried out the construction work in question. The party that carried out the work was identified as “Xentex Industries”, with an ABN, on invoices issued. A payment claim and adjudication application was made by “Xentex Industries Pty Ltd”. The applicant’s issue with the claimant’s identity was not raised in the payment schedule or adjudication response. Lyons J held that it was clear to the applicant that the payment claim was made by the entity to the construction contract. His Honour also “did not consider that a misdescription of a party should defeat the whole adjudication particularly when the issue is not raised at the time of the adjudication”: at [46]. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In Quickway Constructions Pty Ltd v Electrical Energy Pty Ltd [2017] NSWSC 1140, Parker J held that the NSW security of payment legislation still applied between the subcontractor and principal, notwithstanding that the subcontractor had assigned their rights to payment to a third party (BFS). In that case, the subcontractor entered into a factoring agreement with BFS to assign all debts owing to the subcontractor to BFS. The subcontractor submitted their invoices to the principal, along with a notation that – “it had been assigned” to BFS. The principal’s main contention was that, due to the assignment, the payment claim submitted to the principal was, for all intents and purposes, from BFS, and BFS not being a contractor, did not have standing under security of payment legislation to make a claim pursuant to section 13(1) of the NSW Act (equivalent to section 75(1) of the Act).
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour held that, regardless of the principal’s contentions, the notation on the invoice could not in itself, have had the effect of assigning the subcontractor’s statutory rights to BFS. Parker J held that assignment does not occur until a statutory entitlement had crystallised i.e. as a result of a failure to provide a payment schedule, or the provision of a payment schedule in a particular amount. His honour held in this case that it was not until the adjudication determination was made in favour of the subcontractor that the debt had become assignable at law. Prior to the date of the determination, the subcontractor was entitled to pursue the claim for itself. Parker J, recognising that security of payment legislation is intended to create statutory rights to payment, which can be enforced independently of, and in some cases inconsistently with the parties’ contractual rights, held that as the assignment had not been effected until the determination, the claim and corresponding assignment was valid.
[2.4 Com-BIFSOPA CDI Normal Body Text] In the decision of Quickway Constructions Pty Ltd v Electrical Energy Pty Ltd [2017] NSWCA 337, the Court of Appeal (NSW) overturned Parker J’s decision in the first instance and held that if claimant assigns its rights to payment for a payment claim to a third party (i.e. not the claimant), the Act may not be used to pursue that payment claim.  
[2.4 Com-BIFSOPA CDI Normal Body Text] In allowing the appeal, and overturning the adjudication decision, the majority (Gleeson and Leeming JJA) explained that that at the moment the payment claim, which included a notice of assignment of the “invoice” to the financier, was received by the respondent, the claimant ceased to be a creditor of the respondent, thus ceasing any entitlement to pursue the payment claim under the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] In SJ Higgins Pty Ltd v The Bays Healthcare Group Inc [2018] VCC 805, Woodward J held that a payment claim served on the superintendent instead of the principal is not valid, in circumstances where the contract explicitly provided that the superintendent was not authorised to receive claims under the VIC Act. In that case, Woodward J held that the contract was explicit in requiring that SJ Higgins ensure that any document in relation to the VIC Act be served directly to the principal.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Shells Venture Management v Agresta [2019] VSC 863, Digby J held that section 14(1) of the Vic Act (equivalent to section 75(1) of the Act) is not limited in operation to those parties to a contract. In that case, Digby J considered that the Payment Claim in question was validly served by a person not part of the Contract because they were acting as a servant or agent of a party to the Contract, and because the Contract did not preclude the employment of an agent.
[2.2 Com-BIFSOPA Heading 3] 75.6    Construction work to which the claim relates
[2.4 Com-BIFSOPA CDI Normal Body Text] In Leighton Contractors Pty Ltd v Campbelltown Catholic Club Ltd [2003] NSWSC 1103, Einstein J considered the equivalent limitation to Section 17A(2)(b) in the context of the NSW Act:
[2.5 Com-BIFSOPA Normal Body Quote] “Construction work involves as the norm, work which may be expected to be carried out over a period of time. The laying of foundations is an example. This is unlikely to take one hour and may take days or weeks or even longer.
[2.5 Com-BIFSOPA Normal Body Quote] Presumably the same may be said in relation to the provision of related goods and services.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] By definition or real world practice it would be likely that a payment claim would cover disparate forms of construction work carried out over the same or different periods/brackets of time and would similarly likely cover the provision of related goods and services over the same or different periods/brackets of time.
[2.5 Com-BIFSOPA Normal Body Quote] It would seem unlikely that the legislature would have intended that a payment claim in respect of any particular item of construction work [as for example the laying of a particular brick] could only be served within the period of 12 months after completion of the work comprising that particular item.
[2.5 Com-BIFSOPA Normal Body Quote] Possibly the same may also be said in relation to it being unlikely that the legislature would have intended that a payment claim in respect of a particular unit of construction work [as for example the laying of a brick course or concourse] could only be served within the period of 12 months after completion of the work comprising that unit of construction work. On the other hand perhaps it is arguable that the legislature may have so intended.
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] To my mind properly construed the subsection in its reference to "the construction work to which the claim relates" should be regarded as referring in a general way to the construction work or to the related goods and services. Hence as long as any item of construction work to which the claim relates [in that general sense], was carried on during the 12 month period prior to the service of a payment claim, that payment claim could also, unexceptionally, include items of construction work carried on prior to that 12 month period.
[2.5 Com-BIFSOPA Normal Body Quote] The same proposition would hold good in terms of the supply of related goods and services.”
[2.4 Com-BIFSOPA CDI Normal Body Text] This test was narrowed by the New South Wales Court of Appeal in Estate Property Holdings Pty Ltd v Barclay Mowlem Construction Ltd [2004] NSWCA 393.
[2.4 Com-BIFSOPA CDI Normal Body Text] Before the trial judge below (Einstein J in Barclay Mowlem Construction Ltd v Estate Property Holdings Pty Ltd [2004] NSWSC 649), it was submitted that there were three possible interpretations as to the relevant time limit:
[2.6 Com-BIFSOPA bullet] 12 months from any construction work under the contract was last carried out (interpretation 1);
[2.6 Com-BIFSOPA bullet] 12 months from any construction work forming part of the payment claim was last carried out (interpretation 2); or
[2.6 Com-BIFSOPA bullet] requiring all items of construction work under the payment claim to have been carried out in the last 12 months (interpretation 3).
[2.4 Com-BIFSOPA CDI Normal Body Text] Einstein J accepted interpretation 1, consistent with the decision in Leighton Contractors Pty Ltd v Campbelltown Catholic Club Ltd [2003] NSWSC 1103.
[2.4 Com-BIFSOPA CDI Normal Body Text] On appeal, the New South Wales Court of Appeal in Estate Property Holdings Pty Ltd v Barclay Mowlem Construction Ltd [2004] NSWCA 393 held interpretation 2 to be the correct approach. Hodgson JA said:
[2.5 Com-BIFSOPA Normal Body Quote] In my opinion, s.13(2)(a) of the Act requires that a payment claim identify the construction work for which payment is claimed in the claim, not merely the construction work as a whole that is being carried out under the relevant construction contract. I think this is indicated by the words “construction work … to which the progress payment relates”; and strongly confirmed by the consideration that, unless a progress claim identified the particular work for which payment was claimed, it would be impossible for a respondent to provide a meaningful payment schedule supported by reasons. This in turn would make wholly unreasonable s.20(2B) of the Act, which prevents a respondent relying, in an adjudication of a payment claim, on reasons not included in the payment schedule.
[2.5 Com-BIFSOPA Normal Body Quote] Consistently with this, in my opinion “construction work … to which the claim relates” in s.13(4)(b) is also the construction work for which payment is claimed in the claim; and accordingly, the requirement of s.13(4)(b) is that some of that construction work be carried out in the relevant twelve month period. Accordingly, in my opinion, the primary judge’s acceptance of the first of the three interpretations of s.13(4)(b) which he identified, both in this case and in Leighton, was incorrect.
[2.5 Com-BIFSOPA Normal Body Quote] However, in my opinion ss.13-15 of the Act do not provide any basis for dividing up the construction work to which the claim relates into items which may be considered discrete, and asking in respect of each such item whether some work was carried out in the twelve month period. Section 13(2)(b) refers to “the amount” of the progress payment, s.14(4) refers to liability to pay “the claimed amount”, and s.15(4) refers to “the unpaid portion of the claimed amount”: these provisions weigh against the idea that separate consideration should be given to individual items that make up the claimed amount.
[2.5 Com-BIFSOPA Normal Body Quote] Further, the distinction between discrete items of construction work and continuous processes of construction is not clear-cut. The piling work in this case could be considered clearly distinct from other items of construction work; but many other examples can be given where this would be far from clear. For example, is brickwork a distinct item, or part of a continuous process that extends to completing the external fabric of a building? In my opinion, there is no good reason to introduce an unclear distinction of this kind into the operation of the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Pacific General Securities Ltd & Anor v Soliman & Sons Pty Ltd & Ors [2006] NSWSC 13; (2006) 62 NSWLR 421, [37] (Brereton J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Canberra Drilling Rigs Pty Ltd v Haides Pty Ltd [2018] ACTSC 282, McWilliam AsJ followed the decision in Estate Property Holdings Pty Ltd v Barclay Mowlem Constructions Ltd [2004] NSWCA 393 and held that as the payment claim included some work carried out by Canberra Drilling Rigs within 12 months of the payment claim being given to the respondent under section 15 of the ACT Act, the payment claim was valid.
[2.2 Com-BIFSOPA Heading 3] 75.7    The period of 6 months after
[2.3 Com-BIFSOPA Heading 4] Differences between the BCIPA and the Act
[2.4 Com-BIFSOPA CDI Normal Body Text] The limitation under section 17(4) of BCIPA before the Amendment Act provided that a payment claim ‘may be served only within the later of…the period of 12 months after the construction work to which the claim relates…’. This rule was later revised under section 17A(2)-(3) BCIPA and is now equivalent to section 75(2)-(3) of the Act. Relevantly, under the present section 75(2)-(3) of the Act:
[2.6 Com-BIFSOPA bullet] there is a limitation of 6 months (regarding both a payment claim and a final payment claim); and
[2.6 Com-BIFSOPA bullet] the phrase ‘may be served only within the later of’ was replaced by the Amendment Act with ‘must be served within the later of’, and further modified by the Act to read ‘must be given before the end of whichever of the following periods is the longest’.
[2.4 Com-BIFSOPA CDI Normal Body Text] In the context section 17(4) of the pre-amendment BCIPA (currently section 75(2)-(3) of the Act), Jackson J held in South East Civil and Drainage Contractors Pty Ltd v AMGW Pty Ltd [2013] QSC 45; [2013] 2 Qd R 189 that:
[2.5 Com-BIFSOPA Normal Body Quote] a respondent’s failure to take the point of non-compliance with s 17(4) in a payment schedule does not authorise an adjudicator to ignore the point, where it is apparent on the face of the material which the adjudicator is obliged to consider under s 26(2).
[2.4 Com-BIFSOPA CDI Normal Body Text] In so doing, Jackson J distinguished the approach of Palmer J in Brookhollow Pty Ltd v R & R Consultants Pty Ltd [2006] NSWSC 1.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples
[2.6 Com-BIFSOPA bullet] Not ‘apparent on the face of the record’. GW Enterprises Pty Ltd v Xentex Industries Pty Ltd [2006] QSC 399. A payment claim was served on 12 July 2006. The respondent submitted, in its adjudication response, that the payment claim was made in excess of 12 months from the date the last painting works were carried out. The adjudicator did not accept this submission. In proceedings challenging the adjudicator’s decision, Lyons J concluded:
[2.5 Com-BIFSOPA Normal Body Quote] I am not satisfied that the adjudicator has failed to address the issue. The material which was submitted to the adjudicator pursuant to s 26 of the BCIPA included the construction contract which clearly set out the date for commencement as 15 October 2004. This material also included the submission by the applicant to the adjudication annexing material showing evidence of work being done on the sites as late as 8 August 2005. In addition the payment schedule submitted from the respondent (to the adjudication) shows an item listed as “Amount paid for painter to go back”[2] (with respect to three invoices). The respondent’s written submission also specifically included a reference to a “6 month maintenance period”.[3] Furthermore the adjudicator’s decision clearly stated that “the respondent stops short of saying that the claimant did not carry out any work in the 12 months preceding the making of the claim”. I am not satisfied therefore that the adjudicator has failed to address the issue and I am not satisfied that the adjudication decision is a nullity on this basis.
[2.6 Com-BIFSOPA bullet] In Canberra Drilling Rigs Pty Ltd v Haides Pty Ltd [2018] ACTSC 282, McWilliam AsJ followed the decision in Estate Property Holdings Pty Ltd v Barclay Mowlem Constructions Ltd [2004] NSWCA 393 and held that as the payment claim included some work carried out by Canberra Drilling Rigs within 12 months of the payment claim being given to the respondent under section 15 of the ACT Act, the payment claim was valid.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Commercial Fitouts Australia Pty Ltd v Miracle Ceilings (Aust) Pty Ltd & Ors [2020] SASC 11, Stanley J of the South Australian Supreme Court made orders to quash an adjudication determination for jurisdictional error. The plaintiff alleged that the defendant’s payment claim failed to satisfy the “6 month rule” under section 13(4)(b) of the SA Act (equivalent to section 75(2)(b) of the Act). The Court concluded that the work the plaintiff was asserting as being within the 6 month period had already been paid in a previous invoice. This meant that the remaining works asserted in the claim were unrecoverable under the SA Act because they were completed greater than 6 months before the claim was made. As a result, the adjudication determination had been made beyond jurisdiction.
[2.2 Com-BIFSOPA Heading 3] 75.8    Non-compliance with service time limits
[2.4 Com-BIFSOPA CDI Normal Body Text] In the context of section 17(4) of BCIPA before the Amendment Act, Fryberg J in De Neefe Signs Pty Ltd v Build1 (Qld) Pty Ltd; Traffic Technologies Traffic Hire Pty Ltd v Build1 (Qld) Pty Ltd [2010] QSC 279 held that ‘compliance is not a condition precedent to the existence of, nor an essential element of a valid payment claim’. As such, his Honour held that a payment claim which did not comply with these time requirements ‘does not mean that the claim is not one answering the description “payment claim” in s 17(1)’.
[2.4 Com-BIFSOPA CDI Normal Body Text] However, a different approach was taken by the Supreme Court of the Australian Capital Territory in Winyu Pty Ltd v King & Anor [2015] ACTSC 387, wherein Mossop AsJ declared an adjudication decision invalid because the payment claim the subject of the adjudication decision was not given within the period required by section 15(4) of the ACT Act. His Honour concluded: 
[2.5 Com-BIFSOPA Normal Body Quote] I am therefore satisfied that the second defendant did not undertake work pursuant to the building contract or supply goods under that contract on or after 30 June 2014. As a consequence the requirement in s 15(4) of the SOP Act that the works or related goods and services to which the claim relates must have been carried out or supplied within one year of the payment claim is not satisfied and the adjudicator did not have jurisdiction under the SOP Act. 
[Normal] In Iridium Developments Pty Ltd v A-Civil Aust Pty Ltd [2021] NSWSC 1601, William J upheld an adjudicator’s determination on the basis that an adjudicator’s opinion that a payment claim has been served within the time limits prescribed by section 13(4)(b) of the NSW SOP Act (similar to section 75(2)(b) of the BIF Act) is sufficient to enliven the adjudicator's jurisdiction to determine the adjudication application regardless of whether that finding was erroneous.

[2.4 Com-BIFSOPA CDI Normal Body Text] In that case, the respondent challenged the validity of an adjudication determination on the basis that the adjudicator erred in determining that the payment claim related to works carried out within the requisite 12 month period set out under section 13(4)(b) of the NSW SOP Act (similar to section 75(2)(b) of the BIF Act) and that it was a jurisdictional error. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In upholding the adjudicator’s determination and determining that the adjudicator had not fallen into jurisdictional error, Williams J held that section 13(4)(b) of the NSW SOP Act (similar to section 75(2)(b) of the BIF act) is a jurisdictional fact only in the sense that an adjudicator must form an opinion and that the court is not entitled to determine whether the finding was correct or erroneous (this was described as the second category of jurisdictional fact in Icon Co (NSW) Pty Ltd v Australia Avenue Developments Pty Ltd [2018] NSWCA 339). His Honour explained that jurisdictional facts which fall within the second category are less likely to render an adjudication determination void and that the court is only able to review whether the adjudicator’s opinion was lawfully formed not whether the finding was erroneous. In that case, the respondent simply contended that the finding was erroneous. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In Eq Constructions Pty Ltd v A-Civil Aust Pty Ltd [2021] NSWSC 1604, Williams J declared an adjudication determination void on the basis that the adjudicator failed to form an opinion as to whether some of the construction work to which the payment claim related was carried out during the 12 month period stipulated in section 13(4)(b) of the NSW SOP Act  (similar to section 75(2)(b) of the BIF act)
[2.4 Com-BIFSOPA CDI Normal Body Text] In that case, the respondent contended that the adjudicator failed to lawfully form an opinion that the construction work to which the payment claim relates included at least some construction work that was last carried out within 12 months before service of the payment claim. His Honour found that the adjudicator had, despite having correctly stated the effect of s 13(4)(b) of the NSW SOP Act, considered the incorrect question in assessing the payment claim. Namely, the adjudicator considered whether some work had been performed during the 12 month period prior to service of the payment claim rather than whether the payment claim related to any such construction work performed during that period.
[2.4 Com-BIFSOPA CDI Normal Body Text] In determining that the adjudicator had fallen into jurisdictional error, Williams J held that an erroneous finding of the question posited by s 13(4)(b) of the NSW SOP Act would not constitute jurisdictional error however, the adjudicator made no finding of that question in the case and therefore did not enliven his jurisdiction to determine the adjudication application.  His Honour explained that s13(4)(b) is a jurisdictional fact and that, consistent with his decision in Iridium, an adjudicator will fall into jurisdictional error if it is not established. 

[2.2 Com-BIFSOPA Heading 3] 75.9    Section 75(3) – Final payment claims
[2.4 Com-BIFSOPA CDI Normal Body Text] In EHome Construction Pty Ltd v GCB Constructions Pty Ltd [2020] QSC 291 Bond J determined that a payment claim which claimed the release of retention amounts was “plainly” one which fell within the definition of section 68 of the BIF Act, on the basis that retention amounts had been deducted from the value of previous claims during the course of the contract and were therefore a claim for progress payment for construction work carried out, or the supply of related goods and services. In that case, the claimant sought a declaration that the payment claim was invalid and that the adjudication decision was therefore made without jurisdiction on the basis that it was impermissible to include retention in the payment claim since it did not fall within a claim “for” construction work. Bond J found that the payment claim in question satisfied the characteristics of a final payment claim within the meaning of section 75(3) and was given before the end of the longest of the periods referred to in ss 75(3)(a), (b), (c) or (d). Bond J held that it was permissible to include retention in the payment claim for the payment of the completed works because retention amounts were amounts that had been deducted from the value of construction work already completed, and so a claim expressed as this one was simply cannot be characterised as anything other than a payment claim within the meaning of the Act.  
[2.4 Com-BIFSOPA CDI Normal Body Text] In S.H.A. Premier Constructions Pty Ltd v Niclin Constructions Pty Ltd [2020] QSC 307, Bond J upheld an adjudicator’s determination of three adjudications on the basis that the adjudicator appropriately undertook his statutory task in making his determination. In this case, Niclin Constructions Pty Ltd (Niclin), lodged three separate adjudication applications for three separate contracts against S.H.A. Premier Constructions (SHA) and was awarded a total of $850k by the adjudicator. SHA sought declarations that each of the adjudication decisions were void for jurisdictional error by reference to three grounds:
[2.4 Com-BIFSOPA CDI Normal Body Text] Ground 1: misconception of the nature of the adjudicator’s function and misapprehension of the limited of his functions or powers; 
[2.4 Com-BIFSOPA CDI Normal Body Text] Ground 2: failure to undertake the statutory task; and 
[2.4 Com-BIFSOPA CDI Normal Body Text] Ground 3: the adjudication application were vexatious, coupled with the absence of a necessary precondition.
[2.4 Com-BIFSOPA CDI Normal Body Text] In upholding the adjudicator’s determinations, Bond J held:
[2.4 Com-BIFSOPA CDI Normal Body Text] 1) it is an essential part of the adjudicator’s task that the adjudicator address, as a threshold question, whether the adjudication application is frivolous or vexatious, as required in s. 84(2)(a)(ii) of the Act. However, the valid exercise of an adjudicator’s jurisdiction is simply conditioned on the adjudicator having a particular state of mind, namely, the state of mind of having decided that the application is not frivolous or vexatious;
[2.4 Com-BIFSOPA CDI Normal Body Text] 2) the valid exercise of an adjudicator’s jurisdiction is conditioned on the payment claim having been given before the end of the longest of the periods provided by s 75(3) of the Act;
[2.4 Com-BIFSOPA CDI Normal Body Text] 3) whilst a construction contract remains on foot, there continues to be a potential for construction work to be carried out under them, given the possibility of further “rectification work” being required. This is because rectification works will constitute “construction work” where the defect liability period has not expired. However, termination brings about the “completion” of all construction work to be carried out under the construction contracts and the commencement of the 6-month timeframe under s 75(3)(c) (unless a Court can be persuaded that, on a particular earlier date, the works had been completed without defects);
[2.4 Com-BIFSOPA CDI Normal Body Text] 4) where an adjudicator has identified their statutory task correctly in relation to the valuation of the work and defects and has said that they would come back to perform it, but then fails to do so in relation to some defect items, because they either accidentally omitted to do so, or because they wrongly regarded the assessment, “such an accidental or erroneous omission by an adjudicator cannot be characterised as anything other than an error within jurisdiction”.
[2.5 Com-BIFSOPA Normal Body Quote] In Ventralia Pty Ltd v Redline Water Infrastructure Pty Ltd [2023] QSC 291, Freeburn J declared that section 75(3)(c) should not be read as if the time limit was six months from the date the contract was terminated. In this case, the Claimant sought to challenge an adjudication decision whereby the Claimant was required to pay the sum of $241.359 for the payment claim. One of the issues was whether the payment claim was served within the required time, the adjudicator held that the time bar under s 75(3)(c) began when the contract was terminated by the Respondent. 
[2.5 Com-BIFSOPA Normal Body Quote] Freeburn J concluded that the commencement of the six months can only sensibly be measured from the point when work is complete. His Honour also noted that caution is needed when applying SHA Premier Constructions Pty Ltd v Niclin Constructions Pty Ltd [2020] QSC 307. This is because the termination of a construction contract must, or is likely to, set the outer limit for any further work under the contract. Freeburn J did, however, observe that such an inquiry is useful for determining when the work was complete.
[2.2 Com-BIFSOPA Heading 3] 75.9    Section 75(4) – Multiple payment claims
[2.3 Com-BIFSOPA Heading 4] More than one payment claim issued for a reference date
[2.4 Com-BIFSOPA CDI Normal Body Text] In Tailored Projects Pty Ltd v Jedfire Pty Ltd [2009] QSC 32; [2009] 2 Qd R 171, Douglas J accepted a submission that ‘where more than one payment claim issues for a reference date, the first must be considered to be the payment claim for the reference period so that any subsequent claims for that reference period are invalid’.
[2.4 Com-BIFSOPA CDI Normal Body Text] A similar conclusion has been reached under the NSW Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples
[2.6 Com-BIFSOPA bullet] Kellett Street Partners Pty Ltd v Pacific Rim Trading Co Pty Ltd and Ors [2013] QSC 298. Construction work was carried out under an oral construction contract. The Court held that the obligation to perform work had terminated by 28 March 2012 (at [11]). The claimant submitted payment claim 10 on 31 August 2012. The claimant further submitted payment claims 11, 12, 13 and 14. The Court held that payment claim 10 was made in relation to the March 2012 reference date, with the effect that no other payment claims could be validly served by the claimant.
[2.6 Com-BIFSOPA bullet] Refer also the further commentary under the section 64 definition of ‘construction contract’.
[2.3 Com-BIFSOPA Heading 4] Identical payment claims
[2.4 Com-BIFSOPA CDI Normal Body Text] In Doolan v Rubikcon (QLD) Pty Ltd [2007] QSC 168; [2008] 2 Qd R 117, Fryberg J concluded that section 17 of BCIPA does not permit a claimant to serve identical payment claims on a respondent. Referring to a passage by Hodgson JA in Brodyn Pty Ltd t/as Time Coast and Quality v Davenport & Ors (2004) 61 NSWLR 421; [2004] NSWCA 394 at [63]-[64], his Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] “However, it is in my judgment reasonably clear that his Honour in saying that successive payment claims do not necessarily have to be in respect of additional work was not saying that successive payment claims may be identical.  That is because payment claims may include not only work but claims for goods and services and also claims for liability under section 33(3). 
[2.5 Com-BIFSOPA Normal Body Quote] …
[2.5 Com-BIFSOPA Normal Body Quote] …Subsection 17(6) permits also the inclusion of an amount which has been the subject of a previous claim but that does not mean that a previous claim can be the sole item included in the later claim.  
[2.5 Com-BIFSOPA Normal Body Quote] No case has been cited to me where such a claim was permitted.  To allow it seems to me to fly in the face of the words of  ss 12 and 17.”
[2.4 Com-BIFSOPA CDI Normal Body Text] However, the Court of Appeal in Spankie & Ors v James Trowse Constructions Pty Ltd [2010] QCA 355, held that where the Court of Appeal held that the effect of the former section 17(5) of BCIPA was not to be construed to prevent a claimant from including an amount that has been the subject of a previous claim. Fraser JA said:
[2.5 Com-BIFSOPA Normal Body Quote] In my respectful opinion, the text of s 17(6) is wholly inapt to impose any restriction upon the generally expressed entitlement in s 17(1). When s 17(6) is read in the context of the preceding provisions, as it should be, its effect is merely to ensure that no implication may be drawn that s 17(5) precludes a claimant from making a payment claim for an unpaid amount claimed in a previous claim. Section 17(6) does not provide that a payment claim may not claim only an unpaid amount of a previous payment claim and it should not be given that construction. The prospect that claimants might cause respondents undue expense and inconvenience by using payment claims as rehearsals for a final claim is an insufficient basis for such a construction. That concern should not be exaggerated. Claimants will ordinarily have a powerful interest in putting their best foot forward in the earliest payment claim so as to obtain prompt payment. In any event, even on the appellant’s construction s 17(6) permits repetitive claims if the subsequent payment claim also claims another amount. That requires rejection of the appellant’s argument that there is a general implication in BCIPA against any “re-agitation” of a payment claim in a subsequent payment claim, even where there has been no adjudication determination.
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court of Appeal suggested that ‘the re-agitation of a payment claim may be impermissible for other reasons’ (at [25]). The Court of Appeal referred to case law on issue estoppel and case law relied on by the respondent in relation to abuse of process. On this, refer to the commentary below under section 88.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Ware Building Pty Ltd v Centre Projects Pty Ltd & Anor (No 1) [2011] QSC 424.
[2.4 Com-BIFSOPA CDI Normal Body Text] The restriction under section 17(4) prevents multiple servings of different payment claims. Multiple servings of the same payment claim does not contravene this restriction: Eco Steel Homes Pty Ltd v Hippo’s Concreting Pty Ltd & Ors [2014] QSC 135, [10] (Daubney J). This may also apply even where the same payment claim is ‘reasonably supplemented’ with additional information: see Amasya Enterprises Pty Ltd & Anor v Asta Developments (Aust) Pty Ltd & Anor (No 2) [2015] VSC 500, [86] where Vickery J said that:
[2.5 Com-BIFSOPA Normal Body Quote] “the re-sending of the same payment claim, even if reasonably supplemented with additional material and information, does not offend these objectives of the Act. Indeed, it would be inimical to these objectives to inhibit reasonable corrections to be made to payment claims if they are called for. A realistic degree of tolerance needs to be observed to adjust for such shortcomings or mistakes made in the course of submitting a payment claim.”
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Two invoices, only one endorsed as under the Act. In JAG Projects Qld Pty Ltd v Total Cool Pty Ltd & Anor [2015] QSC 229, two invoices were served by the claimant, one dated 14 January 2015 and another dated 31 December 2014. The invoices covered the same work. The 14 January 2015 invoice did not state that it was made under the Act, but the 31 December 2014 invoice did. Notwithstanding the dates on the invoices, the 31 December 2014 was the later invoice served on the respondent. The Court held that, because the 14 January 2015 invoice did not state that it was made under BCIPA, it did not constitute a valid payment claim under BCIPA. Accordingly, only one payment claim was served on the reference date. Similarly, in Kyle Bay Removals Pty Ltd v Dynabuild Project Services Pty Ltd [2016] NSWSC 334, Meagher JA held that where two payment claims are served in respect of the same reference date but only one payment claim complies with the service requirements of the NSW Act, there is only one valid payment claim and therefore section 13(5) of the NSW Act (the equivalent of s75(4)-(5) of the Act) had not been contravened. 
[2.6 Com-BIFSOPA bullet] Resending the same payment claim. Amasya Enterprises Pty Ltd & Anor v Asta Developments (Aust) Pty Ltd & Anor (No 2) [2015] VSC 500. On 7 October 2014, the claimant served a payment claim on the respondent by facsimile, comprising of three invoices. On 9 October 2014 (at 6:17 pm), the claimant re-sent the payment claim to the respondent by facsimile, including invoices which were not attached previously. By section 50(3) of the VIC Act, this facsimile was received on 10 October 2014. The Court held that the 7 October 2014 and 10 October 2014 transmissions together constituted one payment claim. Alternatively, the Court held that the second facsimile was served in substitution of the earlier facsimile.
[2.3 Com-BIFSOPA Heading 4] Construction work done before a particular reference date claimed at a subsequent reference date
[2.4 Com-BIFSOPA CDI Normal Body Text] In Doolan v Rubikcon (QLD) Pty Ltd [2007] QSC 168; [2008] 2 Qd R 117, Fryberg J considered that section 17 of BCIPA allows work which may be claimed at a particular reference date, to be claimed at a subsequent reference date. His Honour said:
[2.5 Com-BIFSOPA Normal Body Quote] That is not to say that the claim must include all work done up to that date.  If something is omitted from a claim, notwithstanding that it could have been claimed on a particular reference date, there is no reason why it cannot be claimed on the next reference date.  Likewise anything further which gives rise to a claim after the first of such reference dates, may also be included in the next claim.  Subsection 17(6) permits also the inclusion of an amount which has been the subject of a previous claim but that does not mean that a previous claim can be the sole item included in the later claim.  
[2.3 Com-BIFSOPA Heading 4] Payment claims after practical completion
[2.4 Com-BIFSOPA CDI Normal Body Text] In Tailored Projects Pty Ltd v Jedfire Pty Ltd [2009] QSC 32; [2009] 2 Qd R 171, the construction contract provided that the reference dates arose at ‘(i) the times stated in the Schedule (or, if any time stated in the Schedule is not a Business Day, the next Business day) or the last Business Day of each month, whichever is the earlier; and (ii) on the Works reaching Practical Completion.’ The contract also provided for the making of a final claim. Douglas J held that, on the drafting of the contract, reference dates arose on the date for practical completion and on the date for making the final claim, but not in between these two dates.
[2.4 Com-BIFSOPA CDI Normal Body Text] A different conclusion was reached in City of Ryde Council v AMFM Constructions [2011] NSWSC 1469. Brereton J held that monthly reference dates continued to accrue, considering that there was ‘no obvious or necessary requirement that monthly claims end upon practical completion’.
[2.2 Com-BIFSOPA Heading 3] 75.10    Section 75(4) – Multiple invoices as one payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] A payment claim may consist of a covering document (such as an email) attaching multiple invoices: Tailored Projects Pty Ltd v Jedfire Pty Ltd [2009] QSC 32; [2009] 2 Qd R 171, 176 [18] (Douglas J); Camporeale Holdings Pty Ltd v Mortimer Construction Pty Ltd & Anor [2015] QSC 211, [29] (Henry J).
[2.4 Com-BIFSOPA CDI Normal Body Text] In Tailored Projects Pty Ltd v Jedfire Pty Ltd [2009] QSC 32; [2009] 2 Qd R 171, the claimant served a claim consisting of a covering document, stating to be a payment claim under BCIPA, which referred to 18 invoices which (with supporting documents) were attached behind this covering document. Each invoice also stated to be a payment claim under the Act. Douglas J held that, together, the documents consisted of one payment claim. His Honour held that each invoice being duly endorsed as under BCIPA should not lead to the conclusion that 19 separate claims were served on the respondent. His Honour said that (at [18]):
[2.5 Com-BIFSOPA Normal Body Quote] To conclude otherwise would require the triumph of form over substance, even in an area where adherence to form and strict compliance with the Act is important.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Camporeale Holdings Pty Ltd v Mortimer Construction Pty Ltd & Anor [2015] QSC 211, the claimant served an email on the respondent, described as being “Claim #8” which contained five attachments to the email. Four of these attachments were separate tax invoices, three for variations, and one labelled as “Progress claim #8”. Each invoice stated that it was a payment claim made under BCIPA, but no such statement appeared on the covering email. The respondent to the payment claim sought to distinguish the case of Tailored Projects based on the covering document not containing such a statement that it was a payment claim under BCIPA. On this submission, Henry J held (at [29]):
[2.5 Com-BIFSOPA Normal Body Quote] The four invoices here were received as one set of documents in the one act of service as attachments to a single covering email. To interpret s 17 as requiring that the covering email here had to explicitly state the attachments formed part of one payment claim would be to ignore the collective state in which they were received and referenced by the covering email. Similarly, to interpret s 17 as requiring the covering email to state that it was made under the Act would be to ignore that each of the invoices stated they were made under the Act. Considered collectively, those invoices and the covering email constituted one payment claim which was stated to be made under the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] See also Alan Conolly & Co v Commercial Indemnity [2005] NSWSC 339, [14]-[23]; Ardnas (No 1) Pty Ltd v J Group (Aust) Pty Ltd [2012] NSWSC 805, [11].
[2.4 Com-BIFSOPA CDI Normal Body Text] In Sought After Investments Pty Ltd v Unicus Homes Pty Ltd [2019] NSWSC 600, Ball J, in the context of section 13(5) of the NSW Act (equivalent of section 75(4) of the Act), held that a payment claim will be valid despite having four separate invoices attached so long as it is objectively clear that the recipient would have understood that a single claim was being made for payment of the full amount of the four invoices. His Honour believed this position was consistent with the policy of the NSW Act not to vex a principal with multiple claims served at different times because
[2.5 Com-BIFSOPA Normal Body Quote] there was no difference between receiving a single letter enclosing four invoices and a single document consolidating the four invoices or claims.
[2.4 Com-BIFSOPA CDI Normal Body Text] Ball J held, at [32], that the service of more than one supporting statement in respect of a payment claim did not undermine the policy of the NSW Act or the requirement to serve a supporting statement imposed by s 13(7) of the NSW Act. On the contrary, the prescribed form of the supporting statement requires the person who signs it to state that the person is “in a position to know the truth of the matters that are contained in this supporting statement”. It is quite possible that in some cases a single person will not be in a position to make that statement in respect of all subcontractors, in which case it would be necessary to serve more than one supporting statement in order to meet the requirements of the prescribed form.
[2.4 Com-BIFSOPA CDI Normal Body Text] His Honour did not determine the issue of a failure to comply with section 13(7) of the NSW Act given that it did not arise on the facts, but remarked in obiter that the present authority lends to the conclusion that where there is non-compliance, an adjudicator does not have jurisdiction to deal with the claim: see Kitchen Xchange v Formacon Building Services [2014] NSWSC 1602; Duffy Kennedy Pty Ltd v Lainson Holdings Pty Ltd [2016] NSWSC 371; Kyle Bay Removals Pty Ltd v Dynabuild Project Services Pty Ltd [2016] NSWSC 334; Greenwood Futures Pty Ltd v DSD Builders Pty Ltd [2018] NSWSC 1407. Ball J acknowledged his departure from the dominant view in Central Projects Pty Ltd v Davidson [2018] NSWSC 523 but decided not to rule on the issue.
[2.4 Com-BIFSOPA CDI Normal Body Text] In J.R. & L.M. Trackson Pty Ltd v NCP Contracting Pty Ltd & Ors [2019] QSC 201, Ryan J held, first, that undue emphasis should not to be placed on the form of a payment claim, and second, that an email attaching three separate invoices will constitute one payment claim in accordance with section 17(4) BCIPA (equivalent of s 75(4) of the Qld Act). In finding that the invoices constituted a payment claim, the court placed emphasis on the fact that the invoices were sent collectively under one email cover which concerned work under the same construction contract and referenced the same project.
[2.4 Com-BIFSOPA CDI Normal Body Text] The argument of multiple invoices constituting jurisdictional error recently arose in Modog Pty Ltd v ZS Constructions (Queenscliff) Pty Ltd [2019] NSWSC 1743. Henry J held that multiple invoices accompanying a ‘Payment Summary Sheet” will still constitute a single payment claim under the NSW Act. On the facts, ZS Constructions had served a payment claim via email to Modog attaching a document referred to as a “Payment Summary Sheet”. Six further separate emails were then sent which attached invoices - all of which were referred to in the Payment Summary Sheet. Importantly, the state of the law in NSW at the time meant that payment claims were not required to endorse that they were made under the NSW Act in order to be valid (similar to the current state of the law under the Act). Modog contended that even if the email and Payment Summary Sheet was a payment claim within the meaning of section 13(1) of the NSW Act (equivalent of section 75(1) of the Act), it was one of many payment claims received from ZS Constructions on the same day, which was therefore contrary to section 13(5) of the NSW Act. In dismissing the application, Digby J held that “unclear drafting in a document, which a party purports to be a payment claim and which may lead to confusion due to it being open to multiple interpretations, does not…necessarily raise a jurisdictional error.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Taringa Property Group Pty Ltd v Kenik Pty Ltd [2024] QSC 298, Hindman J held that a final invoice issued on 8 September 2023 was the only valid payment claim under section 68(1) BIFSOPA. The dispute arose after the Claimant submitted eleven variation claims via email, followed by a twelfth email containing a formal invoice labelled “Payment Claim 31,” supported by a signed statement and demanding nearly $9.7 million. The Respondent argued this constituted multiple payment claims for the same reference date, in breach of section 75(4) BIFSOPA, which limits claimants to one payment claim per reference date. Hindman J rejected the Respondent’s argument and held that only the final invoice constituted a valid payment claim under section 68(1) BIFSOPA. Her Honour noted that earlier emails lacked the features of a payment claim, such as a clear demand for payment and a supporting statement and were should have been construed as supporting documents for the payment claim. Her Honour emphasised that a payment claim must be objectively recognisable as a single demand for payment.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Adjudication on half of the invoices in the payment claim. Downsouth Constructions v Jigsaw Corporate Childcare [2007] NSWSC 597. A payment claim consisted of more than 40 invoices. Each invoice was duly endorsed as being a payment claim under the NSW Act. The claimant lodged an adjudication application on 21 of the invoices. In proceedings to set aside the adjudicator’s determination, the respondent submitted that all of the invoices constituted ‘the payment claim’, and thus, ‘the payment claim’ was not referred to adjudication. This submission was rejected by McDougall J, noting the practice adopted by the claimant was likely to lead to efficiency and savings of time and money.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Spirito Development Pty Ltd v Sinjen Group Pty Ltd [2020] VCC 1368, Woodward J determined that two invoices sent separately could not constitute one payment claim and therefore the relevant payment claim’s infringed the prohibition under section 14(8) of the Vic Act (the equivalent of section 75(4) of BIFSOPA). His Honour reasoned that the Vic Act permits only one payment claim in respect of each reference date, and in the circumstances (at [24]):
[2.5 Com-BIFSOPA Normal Body Quote] “… where emails attaching discrete invoices are separated in time even by only a few minutes, and there is nothing on their face to confirm that they constitute a single payment claim, I consider that they must be viewed as separate payment claims …”
[2.2 Com-BIFSOPA Heading 3] 75.11    Section 75(4) – ‘more than one payment claim … under the construction contract’
[2.4 Com-BIFSOPA CDI Normal Body Text] A payment claim must relate only to one construction contract, and ‘there can only be one adjudication application for any particular payment claim for any particular contract’: Matrix Projects (Qld) Pty Ltd v Luscombe [2013] QSC 4, [17]-[18] (Douglas J); Rail Corporation of NSW v Nebax Constructions [2012] NSWSC 6, [44] (McDougall J); Trinco (NSW) Pty Ltd v Alpha A Group Pty Ltd [2018] NSWSC 239.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Acciona Infrastructure Australia Pty Ltd v Holcim (Australia) Pty Ltd [2020] NSWSC 1330, Hammerschlag J declared an adjudication determination void on the basis that the relevant payment claim was for work completed under multiple contracts. In that case, the payment claim was for the value of goods supplied pursuant to several purchase orders made by the respondent. Relevantly, the contract provided that “upon the issue of a Purchase Order a separate contract will come into existence”. The court applied McDougall J findings in Trinco and held that “each time [the claimant] placed a purchase order, a separate contract for discrete work with a separate payment date came into existence.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Ausipile Pty Ltd v Bothar Boring and Tunnelling (Australia) Pty Ltd [2021] QSC 39, Wilson J determined that a payment claim was invalid on the basis that it was for works completed under more than one construction contract. In that case, the claimant served a payment claim on the respondent which included a consolidated amount owing under two prior payment claim’s and an additional amount for the hire of the claimant’s crawler crane. The claimant argued that the wet hire arrangement was simply a variation to the subcontract and that the legislature could not have intended for the Court to have to engage in an analysis of the existence of a new construction contract every time a principal argued that some variation was not a variation under the existing contract.
[2.4 Com-BIFSOPA CDI Normal Body Text]  In determining that the payment claim contained claims in relation to two separate contracts, Wilson J held:
[2.4 Com-BIFSOPA CDI Normal Body Text] “In my view, the [claimant’s] approach does not give effect to the purpose of the Act. It was the legislatures intention that a payment claim must only relate to one contract. If a payment claim in substances concerns more than one contract, that fact is fatal to its validity under the Act, whether or not the claimant makes this patent on the face of the payment claim.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Ausipile Pty Ltd v Bothar Boring and Tunnelling (Australia) Pty Ltd [2021] QCA 223, the Queensland Court of Appeal overturned Justice Wilson’s finding that the payment claim in issue related to works carried out under multiple contracts and in doing so granted the Claimant summary judgment of its payment claim to which the Respondent failed to provide a payment schedule in response.
[2.4 Com-BIFSOPA CDI Normal Body Text] In determining that the payment claim related to a variation of the contract and not multiple contracts, the Court of Appeal found that it would be contrary to the intention of two commercial parties to enter into a commercial relationship outside the provisions of the contract. The contract contained provisions concerning the date for a progress claim, the date when claims had to be met, insurance, workers compensation and indemnities covering damage. The Court of Appeal dismissed the Respondent’s contention that the payment claim related to a separate contract on the basis that the alleged separate contract did not have a start or end date, there was no date for invoices to be presented, nor any payment terms outlining when the amount was to be paid, and it was silent as to how and whether non-payment would attract interest or penalties. In relation to a court’s consideration of the validity of a payment claim, the Court of Appeal held:
[2.4 Com-BIFSOPA CDI Normal Body Text] “A payment claim should not be treated as a nullity for failure to comply with s 75(1) of the Act, unless that failure is patent on its face. Where a payment claim purports to be made under one contract, it is not rendered invalid simply because at a later time (either during the adjudication or otherwise) it is determined that part of the claim was, in fact, a claim under a different contract. Provided a payment claim is made in good faith and purports to comply with s 75(1) of the Act, the merits of that claim, including questions as to whether it complies with s 75(1), is a matter for adjudication after having been raised in a payment schedule. A recipient of a payment claim cannot simply sit by and raise that point later, if it is not put in a payment schedule in response.”
[2.4 Com-BIFSOPA CDI Normal Body Text] In Ventia Australia Pty Ltd v BSA Advanced Property Solutions (Fire) Pty Ltd [2021] NSWSC 1534, Rees J declared an adjudication determination void on the basis that the payment claim in issue related to works carried out under multiple construction contracts. The claimant and respondent were parties to a fire asset maintenance subcontract which enabled the respondent to issue ‘work orders’ to the claimant for certain works. Under the subcontract, the issue of a ‘work order’ operated to create a new agreement between the parties. The payment claim related to works carried out under more than one ‘work order’. In quashing the determination, Rees J held that the requirement that a payment claim is made in respect of one construction contract only is a jurisdictional fact which is necessary to the existence of an adjudicator’s jurisdiction to make an adjudication determination. [Note annotation will be deleted in next version because of the appeal]
[2.4 Com-BIFSOPA CDI Normal Body Text] In BSA Advanced Property Solutions (Fire) Pty Ltd v Ventia Australia Pty Ltd [2022] NSWCA 82, the NSW Court of Appeal overturned Rees J's decision and found that the adjudicator had not fallen into jurisdictional error because the payment claim in dispute related to one construction contract. The Court of Appeal determined that, as a matter of contract interpretation, the provision of the contract that stated a separate agreement would come into existence each time a work order was issued was inconsistent with the balance of the contract. The Court explained that whilst the work orders may have been an integral part of some aspects of the contractual relationship, the creation of a separate contract was insufficient to ascertain the real legal effect of the issuance of each work order. 
[2.4 Com-BIFSOPA CDI Normal Body Text] The Court of Appeal observed in obiter that the “one contract” rule is not a precondition to the validity of a payment claim and that the NSW SOP Act does not make any strict restriction. In making this observation the Court of Appeal noted:

[List Paragraph] the object of the NSW SOP Act is to ensure that persons carrying out work obtain regular payments on account and are subject to a final reckoning;

[List Paragraph] the stated requirements for a valid payment claim under the SOP Act do not include the identification of the source of the obligation to carry out the work or the source of payment; and


[List Paragraph] the phrase "one contract rule" conveys a degree of precision as to its meaning, which fails to capture the expansive scope of practical commercial arrangements under which goods and services may be supplied.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Cosmo Cranes & Rigging Pty Ltd (Cosmo Cranes) v Eq Constructions Pty Ltd [2022] NSWDC 6, Abadee DCJ refused to grant the claimant summary judgment on its payment claim, following the respondent's failure to provide a payment schedule, on the basis that it was arguable that the payment claim related to works carried out under multiple construction contracts. 
[2.4 Com-BIFSOPA CDI Normal Body Text] In that case, the claimant submitted that the respondent was prohibited from disputing that the payment claim related to multiple construction contracts as the respondent had failed to provide a payment schedule. In dismissing the claimant’s submission, Abadee DCJ explained that although section 15(4)(b) of the NSW SOP Act (no equivalent provision in the BIF Act) limits matters which a respondent may raise in opposition to a proceeding to recover a debt, it is still necessary for a claimant to satisfy the test for awarding summary judgement.
[2.4 Com-BIFSOPA CDI Normal Body Text] In determining that it was arguable that the payment claim in issue related to multiple construction contracts, Abadee DCJ explained that the authorities require a high level of proof or clear evidence such that it was inarguable that the payment claim related to a single construction contract, and that the claimant's failure to establish so was sufficient basis to dismiss the application for summary judgment.
[2.4 Com-BIFSOPA CDI Normal Body Text] Refer to the further commentary under the section 64 definition of ‘construction contract’.
[2.2 Com-BIFSOPA Heading 3] 75.12    Section 75(5) – An amount that was included in a previous payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] The current section 75(5) of the Act was previously covered by section 17(4) of BCIPA and before that, subsections 17(5) and (6) of pre-Amendment Act BCIPA.
[2.4 Com-BIFSOPA CDI Normal Body Text] In the context of the former subsections 17(5) and (6), in VK Property Group Pty Ltd & Ors v A A D Design Pty Ltd & Anor [2011] QSC 54, Boddice J said:
[2.5 Com-BIFSOPA Normal Body Quote] In any event, s 17(6) of the Act does not impose any restriction upon the generally expressed entitlement in s 17(1), namely an entitlement to claim an unpaid amount of work done earlier before an earlier reference date, whether or not it was claimed in an earlier payment claim. The effect of s 17(6) of the Act is merely to ensure that no implication may be drawn that s 17(5) precludes a claimant from making a payment claim for an unpaid amount claimed in a previous claim. There is no general implication in the Act against “re-agitation” of a payment claim in a subsequent payment claim where there has been no adjudication determination.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Denham Constructions Pty Ltd v Islamic Republic of Pakistan (No 2) [2016] ACTSC 215 Mossop AsJ said the following in relation to section 15(6) of the ACT Act, which is the equivalent of section 75(5) of the Act:
[2.5 Com-BIFSOPA Normal Body Quote] A decision not to seek adjudication of a claim which has not been paid does not involve an election to not further pursue that claim. First, everything that occurs under the SOP Act occurs on an interim basis without affecting substantive rights under the contract: s 38. Second, so long as it has not been adjudicated, an amount previously claimed may be claimed again subsequently: s 15(6). The Act does not imply a need to immediately adjudicate all claims or abandon them. Rather, claims may be repeated and adjudicated subsequently although this comes at the cost of delay so far as the claimant is concerned.
[2.5 Com-BIFSOPA Normal Body Quote] In Harlech Enterprises Pty Ltd v Beno Excavations Pty Ltd [2022] ACTCA 41, Lee J (with whom Elkaim J agreed), the Court of Appeal confirmed that s 15(6) of the ACT SOP Act (equivalent to s 75(5) BIFSOPA ), allows a claimant to claim an amount in a payment claim that has been the subject of a previous payment claim. Here, the Court found that this section expressly permits the making of cumulative payment claims.
[2.5 Com-BIFSOPA Normal Body Quote] In Civil & Civic Corporation Pty Ltd v Nova Builders Pty Ltd [2023] ACTCA 30, the ACT Court of Appeal confirmed that section 15(6) of the ACT SOP Act (equivalent to section 75(5) BIFSOPA), allows for a payment claim to claim for construction work which was the subject of a previous payment claim so long as the new payment claim is made from a new and valid reference date. 
[2.5 Com-BIFSOPA Normal Body Quote] In this case, the respondent appealed against the Primary Judge’s finding that an adjudicator had not erred by finding that a payment claim was not invalid, by way of claiming construction work which had already been claimed, in a previous payment claim.
[2.5 Com-BIFSOPA Normal Body Quote] The Court of Appeal held that because the payment claim contained a new reference date, it was valid under the SOP Act, and was able to claim for construction work claimed under a previous payment claim.In Kennedy Civil Contracting Pty Ltd (subject to a Deed of Company Arrangement) v Total Construction Pty Ltd [2023] NSWDC 325, the Court considered whether a payment claim comprising multiple invoices including money the subject of prior claims was valid. Abadee DCJ held that the documents comprised a single payment claim as the covering letter had made it clear that the claimant was bringing a single claim for money for the total of the combination of invoices.
[2.2 Com-BIFSOPA Heading 3] 75.13    Withdrawal of a payment claim
[2.4 Com-BIFSOPA CDI Normal Body Text] A payment claim can be withdrawn and replaced by the consent of both parties, which may arise by necessary implication through conduct: see NC Refractories Pty Ltd v Consultant Bricklaying Pty Ltd [2013] NSWSC 842, [39] (Hammerschlag J); Veer Build Pty Ltd v TCA Electrical and Communication Pty Ltd [2015] NSWSC 864, [32] (Darke J); Reitsma Constructions Pty Ltd v Davies Engineering Pty Ltd t/as In City Steel [2015] NSWSC 343, [21] (Ball J).
[2.4 Com-BIFSOPA CDI Normal Body Text] There is doubt about whether a claimant may unilaterally withdraw a payment claim: see Kitchen Xchange v Formacon Building Services [2014] NSWSC 1602, [17], [26], where McDougall J said:
[2.5 Com-BIFSOPA Normal Body Quote] Whether or not implied unilateral withdrawal is sufficient is a difficult question. It seems to me, in particular given the serious consequences that follow if a respondent does not reply to a payment claim by providing a payment schedule, that the circumstances must make it very clear to the respondent that a payment claim is to be withdrawn, if it is intended that withdrawal should occur.
[2.5 Com-BIFSOPA Normal Body Quote] In the present case, the evidence falls far short of that. All the evidence suggests is that, having received the letter which in my view was a payment schedule and considered what it said, the first defendant decided, as it were, to up the ante and put in a payment claim which not only reinstated the hitherto conceded amounts, but also introduced a totally new and hitherto unheralded claim.
[2.5 Com-BIFSOPA Normal Body Quote] In my view, that is exactly the sort of action that s 13(5) is intended to prohibit. The vice of submission of repetitive payment claims is obvious. It was considered in Dualcorp, not only by Allsop P but also by Macfarlan JA (with whom Handley AJA agreed). It is clearly deleterious to a respondent to be forced to reply individually, often at the expense of time, labour and money, to repetitive payment claims which all relate to the same reference date. On the other hand, should the respondent take the view that it has done enough, it is courting the risk that a particular document will be held to be valid and, thus, sufficient to initiate the process of recovery.
[2.4 Com-BIFSOPA CDI Normal Body Text] In Promax Building Developments Pty Ltd v 167 Lower Heidelberg Road Pty Ltd [2016] VCC 1960, Justice Anderson relied on the reasoning of McDougall J in Kitchen Xchange v Formacon Building Services [2014] NSWSC 1602 at [17], [26] and [27] when determining whether the original progress claim served on 15 November 2016 was later withdrawn by the Contractor when it subsequently served a revised invoice on 22 November 2016. The Contractor sought judgment against the Owner in respect of the original progress claim in the sum of $299,028.83, pursuant to section 16(2)(a)(i) of the VIC Act (equivalent to section 78(2)(a) of the Act).
[2.4 Com-BIFSOPA CDI Normal Body Text] In distinguishing the facts of Kitchen Xchange from that of Promax, Anderson J said:
[2.5 Com-BIFSOPA Normal Body Quote] In the present case, the Owner was fully involved in the process by which the revised invoice was submitted; it being the process which had been followed for the course of the project in relation to all progress claims. In the circumstances, the Owner was entitled to assume that the revised claim served upon it was intended as a valid replacement claim, in respect of which the Owner had the right to respond with a payment schedule in accordance with the Act.
[2.4 Com-BIFSOPA CDI Normal Body Text] Accordingly, Anderson J determined that the application for judgment against the Owner be dismissed and that the Contractor’s appropriate course was to pursue the dispute by adjudication processes under the security of payment act (Vic). 
[2.4 Com-BIFSOPA CDI Normal Body Text] Contrast the circumstances in Amasya Enterprises Pty Ltd & Anor v Asta Developments (Aust) Pty Ltd & Anor (No 2) [2015] VSC 500, [89] where Vickery J held the conduct of the claimant to the sufficient such that the claimant had ‘substituted’ an earlier document served on 7 October 2014 with a later document served on 10 October 2014.
[2.4 Com-BIFSOPA CDI Normal Body Text] Examples:
[2.6 Com-BIFSOPA bullet] Revised tax invoice. NC Refractories Pty Ltd v Consultant Bricklaying Pty Ltd [2013] NSWSC 842. On 11 December 2012, the claimant served a payment claim on the respondent, in the form of a tax invoice, the majority of which was for construction work (brickwork) at a rate of $70 per hour. By email on the same day, the respondent said (inter alia) that they ‘will pay’ $50 per each ‘Ord Hour’. The following day the claimant submitted a revised tax invoice ‘reflecting reduction in rates’. The Court held that, by necessary implication, the earlier invoice had been withdrawn, leaving only the later one: at [39].
[2.6 Com-BIFSOPA bullet] Kitchen Xchange v Formacon Building Services [2014] NSWSC 1602. On 4 June 2014, the claimant served a payment claim on the respondent. Following discussions between representatives of the parties, it was agreed that this payment claim be withdrawn. A further payment claim was served on 12 June 2014. On 13 June 2014, the claimant received correspondence from the respondent’s solicitors on this further payment claim. On 20 June 2014, the claimant served a second further document purporting to be a payment claim. The Court held that the further payment claim (of 12 June 2014) had not been withdrawn by the claimant. The Court held that there was no evidence of an implied agreement as to withdrawal and that unilateral withdrawal (if accepted as being available to a claimant) requires some words or conduct from which the respondent should understand that the earlier payment claim is not being relied on, which was not present in the circumstances: at [26]-[28].
[2.6 Com-BIFSOPA bullet] Veer Build Pty Ltd v TCA Electrical and Communication Pty Ltd [2015] NSWSC 864. On 27 October 2014, the claimant served a payment claim on the respondent, described as being “Claim No 2”. On 10 November 2014, the respondent requested a revision to the payment claim for a number of reasons. On 18 November 2014, the claimant sent a revised spreadsheet (and other documents) to the respondent in an email described as ‘progress calim [sic] 2 Hoyts October 14 revised’. On 25 November 2014, the respondent requested a revision to the contents of the 18 November 2014 email. On 27 November 2014, the claimant sent an email to the respondent described as ‘claim 2 rev 3’, attaching a further revised spreadsheet. The claimant sent a further email (described as ‘Hoyts Broadway Claim 2 – Veer Group’) to the respondent on 1 December 2014, attaching the further revised spreadsheet and other supporting documentation. On 8 December 2014, the respondent repeated its earlier position from its 25 November 2014 email that the work was completed ‘only 70% at best’. A payment claim was made by the claimant on 7 January 2015. The Court held that the 27 October 2014 payment claim was withdrawn by agreement, and secondly, that the 18 November 2014 and 27 November 2014 emails were merely submissions of a draft claim for approval.
[2.6 Com-BIFSOPA bullet] Reitsma Constructions Pty Ltd v Davies Engineering Pty Ltd t/as In City Steel [2015] NSWSC 343. Following the submission of a payment claim, there was a telephone conversation between the claimant and the respondent about outstanding variation claims. The Court accepted the claimant’s evidence of the details of this conversation that the respondent had told the claimant to add the outstanding variations ‘to the current payment claim and send me a revised invoice’. The Court held that there was an agreement that the subsequent payment claim would be served in substitution of the earlier payment claim.
[2.6 Com-BIFSOPA bullet] In Citi-Con (Vic) Pty Ltd v Punton’s Shoes Pty Ltd [2020] VCC 804, Marks J held that a payment claim was invalid under the equivalent of section 75(4) of BIFSOPA on the basis that it was made under the same reference date as an earlier payment claim. In that case, the claimant unilaterally withdrew an earlier payment claim and served a new claim, relying on the same reference date. Marks J distinguished the case from NC Refractories Pty Limited v Consultant Bricklaying Pty Limited [2013] NSWSC 842, where Hammerschlag J held that a claimant may withdraw and reissue a payment claim in respect of the same reference date with consent of the respondent. His Honour held that where there is no such agreement between parties regarding withdrawal, section 14(8) of the Vic Act (equivalent to section 75(4) BIFSOPA) is a clear prohibition on serving more than one claim with respect to a single reference date.

```

## Other authority

### `other/qbcc_act_s42_unlicensed_work.txt`
```
# Source: Queensland Building and Construction Commission Act 1991
# Section 42 (unlawful carrying out of building work — illegality affects PC validity under BIF Act)

42 Unlawful carrying out of building work
(1) Unless exempt under schedule1A, a person must not carry
out, or undertake to carry out, building work unless the person
holds a contractor’s licence of the appropriate class under this
Act.
Maximum penalty—
(a) for a first offence—250 penalty units; or
(b) for a second offence—300 penalty units; or
(c) for a third or later offence, or if the building work
carried out is tier 1 defective work—350 penalty units or
1 year’s imprisonment.
(2) An individual who contravenes subsection(1) and is liable to
a maximum penalty of 350 penalty units or 1 year’s
imprisonment, commits a crime.
(3) Subject to subsection(4), a person who carries out building
work in contravention of this section is not entitled to any
monetary or other consideration for doing so.
(4) A person is not stopped under subsection(3) from claiming
reasonable remuneration for carrying out building work, but
only if the amount claimed—
(a) is not more than the amount paid by the person in
supplying materials and labour for carrying out the
building work; and
(b) does not include allowance for any of the following—
(i) the supply of the person’s own labour;
Page 72 Current as at 1 February 2026

(ii) the making of a profit by the person for carrying
out the building work;
(iii) costs incurred by the person in supplying materials
and labour if, in the circumstances, the costs were
not reasonably incurred; and
(c) is not more than any amount agreed to, or purportedly
agreed to, as the price for carrying out the building
work; and
(d) does not include any amount paid by the person that
may fairly be characterised as being, in substance, an
amount paid for the person’s own direct or indirect
benefit.
42A Temporary exemption from s 42 for up to 12 months for
new classes of licence
(1) This section applies if—
(a) on or after 1 July 2000, a class of licence (the new class
of licence) is established under this Act; and
(b) immediately before the establishment of the new class
of licence a person (the relevant person) was carrying
on a business that included carrying out work (relevant
work) within the scope of work for the new class of
licence; and
(c) either of the following applied immediately before the
establishment of the new class of licence—
(i) no relevant work was building work;
(ii) some relevant work was building work, but its
carrying out was incidental to the carrying out of
all other relevant work.
(2) Section42 does not apply to the relevant person for carrying
out, or undertaking to carry out, relevant work for the first 6
months after the new class of licence is established.
(3) Also, if the relevant person applies for the new class of licence
within the 6-month period mentioned in subsection(2),

section 42 does not apply in relation to the person for carrying
out, or undertaking to carry out, relevant work until the
application is decided or withdrawn.
(4) To remove any doubt, it is declared that while section 42 does
not apply to the relevant person under subsection(2) or (3)—
(a) the person is not prohibited from carrying out, or
undertaking to carry out, relevant work under
section 42(1); and
(b) the person is not stopped from being entitled to
monetary or other consideration for carrying out
relevant work under section42(3).
(5) If the application mentioned in subsection(3) is not decided
or withdrawn within 12 months after the new class of licence
is established, the application is taken to be refused at the end
of the 12 months.
42B Carrying out building work without a nominee
(1) A licensee that is a company must not carry out, or undertake
to carry out, building work unless the licensee has a nominee
who holds a contractor’s licence or a nominee supervisor’s
licence for the building work carried out, or undertaken to be
carried out, under the company’s class of licence.
Maximum penalty—
(a) for a first offence—250 penalty units; or
(b) for a second offence—300 penalty units; or
(c) for a third or later offence, or if the building work
carried out is tier 1 defective work—350 penalty units.
(2) However, a licensee does not commit an offence against
subsection(1) if the period the licensee has not had a nominee
is less than 28 days.
(3) An individual who contravenes subsection(1) and is liable to
a maximum penalty of 350 penalty units, commits a crime.
Page 74 Current as at 1 February 2026

42C Unlawful carrying out of fire protection work
(1) An individual must not personally carry out, or personally
supervise, fire protection work unless the individual—
(a) holds a fire protection occupational licence; or
(b) holds a licence, registration or authorisation under this
or another Act that allows the person to personally carry
out or personally supervise the work.
Maximum penalty—
(a) for a first offence—250 penalty units; or
(b) for a second offence—300 penalty units; or
(c) for a third or later offence, or if the fire protection work
carried out is tier 1 defective work—350 penalty units or
1 year’s imprisonment.
(2) Subsection (1) does not apply to an individual who personally
carries out fire protection work if—
(a) the fire protection work is a type prescribed under a
regulation; and
(b) the individual has the technical qualifications prescribed
under a regulation for the type of fire protection work;
and
(c) the individual carries out the fire protection work for a
licensed contractor who holds a licence of the relevant
class for the work.
(3) Also, subsection(1) does not apply to—
(a) an apprentice who personally carries out fire protection
work in a calling that requires the apprentice to carry out
the work; or
(b) a trainee who personally carries out fire protection work
in a calling that requires the trainee to carry out the
work; or
(c) a student who personally carries out fire protection work
as part of training under the supervision of teaching staff
at—

[s 42CA]
(i) a university; or
(ii) a college, school or similar institution conducted,
approved or accredited by the State or the
Commonwealth; or
(d) a student who, for work experience, personally carries
out fire protection work as part of a pre-vocational
course.
(3A) Further, subsection(1) does not apply to a person who,
immediately before the establishment of a new class of
licence, registration or authorisation mentioned in
subsection(1), was carrying out work within the scope of
work for the new class of licence, registration or authorisation
in a circumstance prescribed by regulation.
(4) An individual who contravenes subsection(1) and is liable to
a maximum penalty of 350 penalty units or 1 year’s
imprisonment, commits a crime.
42CA Unlawful carrying out of mechanical services work
(1) An individual must not personally carry out, or personally
supervise, mechanical services work unless the individual—
(a) holds a mechanical services occupational licence; or
(b) holds a licence, registration or authorisation under this
Act or another Act that allows the person to personally
carry out or personally supervise the work.
Maximum penalty—
(a) for a first offence—250 penalty units; or
(b) for a second offence—300 penalty units; or
(c) for a third or later offence, or if the mechanical services
work carried out is tier 1 defective work—350 penalty
units or 1 year’s imprisonment.
(2) Subsection (1) does not apply to an individual who personally
carries out mechanical services work if the mechanical
services work is a type prescribed by regulation.
Page 76 Current as at 1 February 2026

(3) Also, subsection(1) does not apply to—
(a) an apprentice who personally carries out mechanical
services work in a calling that requires the apprentice to
carry out the work; or
(b) a trainee who personally carries out mechanical services
work in a calling that requires the trainee to carry out the
work; or
(c) a student who personally carries out mechanical
services work as part of training under the supervision
of teaching staff at—
(i) a university; or
(ii) a college, school or similar institution conducted,
approved or accredited by the State or the
Commonwealth; or
(d) a student who, for work experience, personally carries
out mechanical services work as part of a pre-vocational
course.
(3A) Further, subsection(1) does not apply to a person who,
immediately before the establishment of a new class of
licence, registration or authorisation mentioned in
subsection(1), was carrying out work within the scope of
work for the new class of licence, registration or authorisation
in a circumstance prescribed by regulation.
(4) An individual who contravenes subsection(1) and is liable to
a maximum penalty of 350 penalty units or 1 year’s
imprisonment commits a crime.
42D Licensed contractor must not engage or direct
unauthorised person for fire protection work
(1) A licensed contractor must not engage or direct an employee
to carry out fire protection work unless the employee is
authorised to carry out the work under this or another Act.
Maximum penalty—
(a) for a first offence—250 penalty units; or

[s 42DA]
(b) for a second offence—300 penalty units; or
(c) for a third or later offence, or if the fire protection work
carried out is tier 1 defective work—350 penalty units or
1 year’s imprisonment.
(2) An individual who contravenes subsection(1) and is liable to
a maximum penalty of 350 penalty units or 1 year’s
imprisonment, commits a crime.
Note—
This provision is an executive liability provision—see section111B.
42DA Licensed contractor must not engage or direct
unauthorised person for mechanical services work
(1) A licensed contractor must not engage or direct an employee
to carry out mechanical services work unless the employee is
authorised to carry out the work under this Act or another Act.
Maximum penalty—
(a) for a first offence—250 penalty units; or
(b) for a second offence—300 penalty units; or
(c) for a third or later offence, or if the mechanical services
work carried out is tier 1 defective work—350 penalty
units or 1 year’s imprisonment.
(2) An individual who contravenes subsection(1) and is liable to
a maximum penalty of 350 penalty units or 1 year’s
imprisonment commits a crime.
Note—
This provision is an executive liability provision—see section111B.
42E Avoidance of contractual obligations causing significant
financial loss
(1) This section applies to a person who is a party to a building
contract.
(2) The person must not, without reasonable excuse, cause
another party to the building contract to suffer a significant
Page 78 Current as at 1 February 2026

financial loss because of the person’s deliberate
non-compliance with the contract.
Maximum penalty—350 penalty units.
Note—
See also the Justices Act 1886, section76.

```
