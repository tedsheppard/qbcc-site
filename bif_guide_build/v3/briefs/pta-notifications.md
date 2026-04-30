# Element-page brief — pta-notifications
**Title:** Notifications to QBCC, principal and subcontractors
**Breadcrumb:** Project trust accounts
**Anchor id:** `pta-notifications`
**Output file:** `bif_guide_build/v3/pages/page_pta-notifications.html`

## Extra guidance for this element
No annotated commentary. Cover: notification to commissioner of opening / closing / changes; subcontractor notification of trust establishment; principal information.

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

### `statute/chapter_2/section_007.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 7 — Purpose of chapter
The main purpose of this chapter is to ensure that funds paid
to the contracted party for particular contracts are held in a
trust to protect the interests of subcontractors.

```

### `statute/chapter_2/section_008.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 8 — Definitions for chapter
In this chapter—
amendment, of a contract, includes a variation of the contract
or a change in the contract price.
approved financial institution means a financial institution
approved by the commissioner under section55.
building means a fixed structure that is wholly or partly
enclosed by walls or is roofed.
contract administration, in relation to project trust work
wholly or partly designed by a person, includes the
following—
(a) preparing tender documentation and calling and
selecting tenders;
(b) preparing, or helping the person’s clients with the
preparation of, contracts;
(c) preparing additional documentation for the person’s
clients or building contractors;
(d) arranging and conducting on-site meetings and
inspections;
(e) arranging progress payments;
(f) arranging for certificates, including certificates from a
local government, to be issued;

(g) providing advice and help to the person’s clients
including during the maintenance period allowed under
a contract.
contracted party, for a contract, means the party to the
contract who is required to perform work under the contract,
whether by—
(a) carrying out the work personally; or
(b) directly or indirectly causing the work to be carried out.
contracted work, for a contract, means the work required to
be carried out under the contract.
contracting party, for a contract, means the party to the
contract for whom the contracted work is to be carried out.
contract price see section9.
hospital and health service means a Hospital and Health
Service established under the Hospital and Health Boards Act
2011, section17.
mechanical services work see the Queensland Building and
Construction Commission Act 1991, schedule 2.
project trust contract means a contract for which a project
trust is required under section12.
project trust subcontract see section9A.
project trust work see section8A.
State authority—
(a) means—
(i) an agency, authority, commission, corporation,
instrumentality, office, or other entity, established
under an Act or by authority of the State for a
public or State purpose; or
(ii) a corporation that is—
(A) owned or controlled by the State, a local
government or an entity mentioned in
subparagraph(i); and
Page 19 Current as at 27 April 2025

(B) prescribed by regulation to be a State
authority; or
(iii) a subsidiary of a corporation mentioned in
subparagraph(ii); or
(iv) a part of an entity mentioned in subparagraphs (i)
to (iii); or
(v) a hospital and health service; but
(b) does not include an entity prescribed by regulation not
to be a State authority.
trust records means records required to be kept and retained
under section 52.
variation, of a contract, means an addition to, or an omission
from, the contracted work.

```

### `statute/chapter_2/section_009.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 9 — Meaning of contract price
(1) The contract price, for a contract, means the amount the
contracted party is entitled to be paid under the contract or, if
the amount can not be accurately calculated, the reasonable
estimate of the amount the contracted party is entitled to be
paid under the contract.
(2) In working out the amount under subsection(1), an amount
for GST is not to be included.

```

### `statute/chapter_2/section_010.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 10 — Definitions for part
In this part—
head contract means a contract for project trust work that is
not also a subcontract of another contract.
project trust account means the account for a project trust at a
financial institution.
subcontractor beneficiary, for a project trust, means a
subcontractor who is a beneficiary of the trust under
section 11A.

```

### `statute/chapter_2/section_011.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 11 — What is a project trust
A project trust is a trust—
(a) over amounts—
(i) payable in connection with a project trust contract
or project trust subcontract; and
(ii) required to be deposited in the project trust account
under this chapter; and
(b) primarily for the benefit of subcontractors for the project
trust contract.
Page 27 Current as at 27 April 2025

```

### `statute/chapter_2/section_012.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 12 — When project trust required for a contract
(1) This section applies to a contract entered into on or after the
commencement of this section.
(2) A project trust is required for a contract if—
(a) the contract is eligible for a project trust under
subdivision2; and
(b) the contract is not exempted under subdivision3; and
(c) the contracted party enters into a subcontract for all or
part of the contracted work.
(3) The requirement starts on the first day a project trust is
required under subsection(2).
(4) The requirement continues until the project trust is dissolved
under section21, regardless of any of the following
changes—
(a) a variation, or any other amendment, of the contract;
(b) a change in the contract price;
(c) a change in the contracted work.
Page 29 Current as at 27 April 2025

(5) If a project trust is required for a contract and a project trust is
also required for a subcontract of the contract, separate project
trusts are required for the contract and the subcontract.

```

### `statute/chapter_2/section_013.txt`
```
[MISSING SOURCE FILE: source/statute/chapter_2/section_013.txt]
```

### `statute/chapter_2/section_014.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 14 — Eligibility of contract for project trust when
contract entered into
A contract is eligible for a project trust if, when it
is entered into—
(a) more than 50% of the contract price is for
project trust work; and
Page 223 Current as at 27 April 2025

(b) the contract price is $1 million or more.

```

### `statute/chapter_2/section_015.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 15 — Subcontracts generally
A project trust is not required for a subcontract unless it is a
type of subcontract to which section14C or 14E apply.

```

### `statute/chapter_2/section_016.txt`
```
[MISSING SOURCE FILE: source/statute/chapter_2/section_016.txt]
```

### `statute/chapter_2/section_017.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 17 — Establishment of project trust
Once a project trust is required for a contract under section 12,
the trust is established by the first of the following being made
after the trust is required—
(a) payment of an amount from the contracting party to the
contracted party under the contract;
(b) payment of an amount from the contracted party to a
subcontractor beneficiary for subcontracted work under
the contract;
(c) a deposit in the project trust account as required under
this chapter.

```

### `statute/chapter_2/section_018.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 18 — Contracted party must open project trust account
(1) If a project trust is required for a contract under section12, the
contracted party must open an account at a financial
institution for the trust as required by this section.
Maximum penalty—500 penalty units.
(2) The project trust account must be opened within 20 business
days after the contracted party enters into the first subcontract
for the contract.
(3) However—
(a) if a project trust is not required for the contract until
after an amendment of the contract; and
Note—
See section14A about amendments of contracts affecting the
requirement to establish a project trust.
Page 37 Current as at 27 April 2025

(b) the contracted party entered into a subcontract for the
contract before the amendment of the contract;
the project trust account must be opened within 20 business
days after the day the contract is amended.
(4) The project trust account must not be a virtual account or
subordinate to any other account at a financial institution.
(5) There must not be more than 1 project trust account for the
project trust.
(6) A provision of a contract that provides that the project trust
account must be opened less than 20 business days after the
contract is entered into is of no effect.

```

### `statute/chapter_2/section_019.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 19 — All payments from contracting party to be deposited in
project trust account
(1) This section applies to any of the following amounts paid by
the contracting party to the contracted party in connection
with a project trust contract—
(a) an amount paid in accordance with the terms of the
contract;
(b) an amount paid because the contracting party is liable
under section77 to pay the amount to the contracted
party in connection with the contract;
(c) an amount paid under chapter3, part4 because of an
adjudication of a disputed progress payment relating to
the contract;
(d) an amount paid because of a final and binding dispute
resolution process relating to the contract;
(e) an amount paid because of a court order relating to the
contract;
(f) an amount, paid for any other reason, that reduces the
unpaid amount of the contract price for the contract.
(2) The contracting party must deposit the amount into the project
trust account for the contract (the deposit obligation) unless—
(a) the amount was due to be paid before the trust was
established; or
(b) the amount is paid into court; or
(c) the amount is to be withheld because of a payment
withholding request given to the contracting party under
section 97B; or

(d) the amount is paid directly to a person under chapter 4
in connection with a subcontractor’s charge; or
(e) the contracting party has a reasonable excuse for failing
to deposit the amount into the account.
Maximum penalty—200 penalty units.
(3) Once the amount is deposited into the project trust account,
the deposit is taken to be a payment made by the contracting
party to the contracted party and discharges the contracting
party’s liability to pay that amount to the contracted party.
(4) If an amount is paid to the contracted party or its agent in
contravention of the deposit obligation, the contracted party
must deposit the amount into the project trust account as soon
as practicable after receiving the amount.
Maximum penalty for subsection(4)—200 penalty units or 2
years imprisonment.

```

### `statute/chapter_2/section_020.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 20 — All payments to subcontractor beneficiaries to be paid
from project trust account
(1) The contracted party for a project trust contract may only pay
an amount to a subcontractor beneficiary of the project trust—
(a) from the project trust account; and
(b) by depositing the amount into the account of a financial
institution nominated by the beneficiary.
Maximum penalty—200 penalty units or 1 year’s
imprisonment.
(2) To remove any doubt, it is declared that the obligation to pay
an amount from the project trust account applies whether or
not the amount is held in the account when it is to be paid.
Note—
See section51 about covering shortfalls.
(3) The account nominated by the subcontractor beneficiary
under subsection(1)(b) must be—
(a) controlled by the beneficiary; and
(b) if the beneficiary is also required to establish a project
trust for its subcontract—the account for the project
trust for the subcontract.
(4) This section does not apply to—
(a) a retention amount withheld from payment to a
subcontractor beneficiary if the amount is deposited into
a retention trust account of which the subcontractor is,
or will be, a beneficiary; and
(b) a retention amount to be released to a subcontractor
beneficiary from a retention trust account.

```

### `statute/chapter_2/section_021.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 21 — Ending project trust
(1) Once a project trust is established for a contract, the trustee
may dissolve the trust only if—
(a) there are no longer any subcontractor beneficiaries for
the trust; or
(b) the only remaining work to be carried out under the
contract is maintenance work.
(2) A project trust is dissolved by the trustee—
Page 45 Current as at 27 April 2025

(a) closing the project trust account; and
(b) giving written notice to the commissioner of the trust
having been dissolved.
(3) A trustee is taken not to dissolve the project trust by closing
the project trust account if the account was only closed for the
purpose of transferring the account to another financial
institution under section 18C.
(4) When dissolving the project trust, the trustee may pay itself
the following amounts—
(a) any amount for interest that the trustee is entitled to
under section 51D;
(b) any remaining amount that is not owing to a
subcontractor beneficiary.
(5) In this section—
maintenance work see section15D(2).

```

### `statute/chapter_2/section_022.txt`
```
[MISSING SOURCE FILE: source/statute/chapter_2/section_022.txt]
```

### `statute/chapter_2/section_023.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 23 — Notice of project trust before entering subcontracts
(1) The contracted party for a project trust contract must give
each subcontractor a notice about the use of a project trust
account (notice of project trust) as required by this section.
Maximum penalty—200 penalty units or 1 year’s
imprisonment.
(2) The notice of project trust must—
(a) be in writing; and
(b) include a statement that a project trust will be used for
making payments to the subcontractor; and
(c) include the information prescribed by regulation.
(3) The notice of project trust must be given to the
subcontractor—
(a) if the project trust is not yet established when the
contracted party and the subcontractor enter into a
subcontract—within 10 business days after the trust is
established; or
(b) if the project trust is already established when the
contracted party and the subcontractor enter into a
subcontract—before the contracted party and the
subcontractor enter into a subcontract.
(4) However—
(a) if a project trust is not required for a contract until after
an amendment of the contract; and
Note—
See section14A about amendments of a contract affecting the
requirement to establish a project trust.
(b) the contracted party entered into a subcontract for the
contract before the amendment of the contract;
the notice of project trust must be given within 10 business
days after opening the project trust account.
Page 47 Current as at 27 April 2025

```

### `statute/chapter_2/section_024.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 24 — Contracting party to report related entities
(1) This section applies if—
(a) a project trust is established for a contract; and
(b) the contracting party knows, or ought reasonably to
know, that a subcontractor beneficiary is a related entity
for the contracted party.
(2) The contracting party must, using an approved way, inform
the commissioner of the matter within 5 business days after
the party first becomes aware, or ought reasonably to have
become aware, of the matter.
Maximum penalty—50 penalty units.

```

### `statute/chapter_2/section_025.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 25 — Contracted party to report related entities
(1) This section applies if—
(a) a project trust is established for a contract; and
(b) the contracted party enters into a subcontract with a
related entity for the party.
(2) The contracted party must, using an approved way, inform the
commissioner about entering into the subcontract with the
related entity within 5 business days after entering into the
subcontract.
Maximum penalty—200 penalty units.

```

### `statute/chapter_2/section_026.txt`
```
[MISSING SOURCE FILE: source/statute/chapter_2/section_026.txt]
```

### `statute/chapter_2/section_027.txt`
```
[MISSING SOURCE FILE: source/statute/chapter_2/section_027.txt]
```

### `statute/chapter_2/section_028.txt`
```
[MISSING SOURCE FILE: source/statute/chapter_2/section_028.txt]
```

### `statute/chapter_2/section_029.txt`
```
[MISSING SOURCE FILE: source/statute/chapter_2/section_029.txt]
```

### `statute/chapter_2/section_030.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 30 — Definitions for part
In this part—
building contract see Queensland Building and Construction
Commission Act 1991, section67AAA.
minimum contract price means the contract price amount
prescribed by regulation.
retention trust see section31.
retention trust account means the account for a retention trust
at a financial institution.

```

### `statute/chapter_2/section_031.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 31 — What is a retention trust
A retention trust is a trust—
(a) over the following amounts—
(i) retention amounts withheld in the form of cash
under particular contracts, inclusive of any GST
related to those amounts;
(ii) deposits in the retention trust account as required
under this chapter; and
(b) primarily for the benefit of the party who will be entitled
to the retention amount.

```

### `statute/chapter_2/section_032.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 32 — When retention trust required
(1) A retention trust is required for a retention amount withheld
from payment under a contract if—
Page 53 Current as at 27 April 2025

(a) the contract is a withholding contract at the time the
retention amount is withheld; and
(b) the retention amount is withheld by the contracting
party in the form of cash.
(2) Also, a retention trust is required for a retention amount
withheld from payment under a contract if—
(a) the contract was not a withholding contract at the time
the retention amount was withheld but the contract has
since become a withholding contract; and
(b) the contract is a first tier subcontract; and
(c) the retention amount was withheld by the contracting
party in the form of cash; and
(d) the retention amount had not been released to the parties
entitled to it at the time the contract became a
withholding contract.
(3) The requirement starts, or is taken to have started, on the first
day the contracting party withholds the retention amount from
payment.
(4) The requirement continues until all of the retention amount
has been released to the parties entitled to it under the
withholding contract, regardless of any of the following
changes—
(a) a variation, or any other amendment, of the contract;
(b) a change in the contracted work.
(5) This section does not apply to a retention amount withheld
from payment under a contract if—
(a) the contracting party is the State, the Commonwealth, a
state authority, a local government or another entity
prescribed by regulation; or
(b) the contract price for the contract is at least the
minimum contract price.
(6) In this section—
withholding contract means—

(a) a project trust contract that is—
(i) a head contract; or
(ii) a subcontract that is eligible for a project trust
under section 14C or 14D; or
(b) a project trust subcontract for a project trust contract
mentioned in paragraph(a).

```

## BIF Regulation

### `regs/reg_010.txt`
```
# Source: BIF Regulation 2018
# Section 10 — Assessment 2—Assignment: Mock adjudication decision
Schedule 2
Schedule 2 Fees
section 39
Fee units

```

### `regs/reg_011.txt`
```
# Source: BIF Regulation 2018
# Section 11 — Approval of registrar’s policies—Act, s 155
For section155(2) of the Act—
(a) version 1 of the document called ‘Adjudicator referral
policy’ and published on the commission’s website is
approved as a policy about administering chapter3 of
the Act; and
(b) version 1 of the document called ‘Continuing
professional development for adjudicators policy’ and
published on the commission’s website is approved as a
policy about administering chapter5 of the Act.

```

### `regs/reg_012.txt`
```
# Source: BIF Regulation 2018
# Section 12 — Approval of code of conduct—Act, s 181
For section181(2) of the Act, version 1 of the document
called ‘Code of conduct for adjudicators’ and published on the
commission’s website is approved.
Page 20 Current as at 1 July 2025

```

### `regs/reg_013.txt`
```
# Source: BIF Regulation 2018
# Section 13 — Conflicts of interest—Act, s 80
(1) For section 80(b) of the Act, an adjudicator has a conflict of
interest if—
(a) the adjudicator or a family member of the adjudicator—
(i) is, or is contracted to be, employed or otherwise
engaged by the claimant or respondent for the
adjudication application; or
(ii) is an owner of a building, structure or land in
relation to which construction work or the supply
of related goods and services to which the
adjudication application relates is being carried
out; or
(iii) is carrying out construction work or the supply of
related goods and services in relation to a building,
structure or land to which the adjudication
application relates; or
(iv) has a direct or indirect pecuniary or other interest
in a matter to be considered during the adjudication
that could conflict with the proper performance of
the adjudicator in adjudicating the adjudication
application; or
(b) the claimant or respondent for the adjudication
application is a family member of the adjudicator.
(2) For subsection(1), a person is a family member of the
adjudicator if the person is—
(a) the adjudicator’s spouse; or
(b) a grandparent, parent, uncle, aunt, brother, sister, cousin,
child, nephew, niece or grandchild of—
(i) the adjudicator; or
(ii) the adjudicator’s spouse; or
(c) a spouse of a person mentioned in paragraph(b).

(3) In this section—
freehold land see the Land Act 1994, schedule 6.
owner—
(a) of a building or structure, means the owner of the
building or structure under the Building Act 1975; or
(b) of land, means the following—
(i) if the land is freehold land—the registered owner
of the land;
(ii) if the land is the subject of a lease registered under
the Land Title Act 1994—the lessee of the land;
(iii) if the land is the subject of a lease registered under
the Land Act 1994—the lessee of the land;
(iv) if the land is a reserve—the trustee of the reserve;
(v) if a person has occupation rights in relation to the
land under a licence or permit—the licensee or
permittee.
reserve see the Land Act 1994, schedule 6.

```

### `regs/reg_014.txt`
```
# Source: BIF Regulation 2018
# Section 14 — Maximum fees and expenses for particular adjudication
applications—Act, s 95
(1) This section prescribes, for section 95(2) of the Act, the
maximum amount for fees and expenses an adjudicator is
entitled to be paid for adjudicating an adjudication application
relating to a payment claim for a progress payment of not
more than $25,000.
(2) The maximum amount is—
(a) if the progress payment is not more than $5,000—$620;
or
(b) if the progress payment is more than $5,000 but not
more than $15,000—$930; or
(c) if the progress payment is more than $15,000 but not
more than $20,000—$1,860; or
Page 22 Current as at 1 July 2025

(d) if the progress payment is more than $20,000 but not
more than $25,000—$2,070.
(3) To remove any doubt, it is declared that a maximum amount
mentioned in subsection(2) includes both fees and expenses.

```

### `regs/reg_015.txt`
```
# Source: BIF Regulation 2018
# Section 15 — Time for lodgement—Act, s 201
(1) For section201(2)(f)(i) of the Act, an adjudication application
must be lodged with the registrar no later than 5p.m. on a
business day.
(2) An application lodged after 5p.m. is taken to be lodged on the
next business day.

```
