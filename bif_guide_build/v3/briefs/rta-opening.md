# Element-page brief — rta-opening
**Title:** Opening the retention trust account
**Breadcrumb:** Retention trust accounts
**Anchor id:** `rta-opening`
**Output file:** `bif_guide_build/v3/pages/page_rta-opening.html`

## Extra guidance for this element
No annotated commentary. Cover: timing, account name, financial institution.

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

### `statute/chapter_2/section_033.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 33 — Establishment of retention trust
(1) This section applies if, under section 32, a retention trust is
required for a retention amount withheld from payment under
a contract.
(2) The retention trust is established by the contracting party
withholding the retention amount from payment.

```

### `statute/chapter_2/section_033A.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 33A — Charge over retention amounts held in retention trust
(1) This section applies if, under section 32, a retention trust is
required for a retention amount withheld from payment under
a contract.
(2) In addition to the retention trust, the retention amount is also
subject to a charge in favour of the contracted party for
securing the release of the amount when the party becomes
entitled to the amount.
(3) However, if and when the contracting party becomes entitled
to be paid part of the retention amount under the contract, the
charge is released over that part of the retention amount.
(4) The contracted party may enforce the charge as if the charge
had been given to it under a written agreement between it and
the contracting party.
Page 55 Current as at 27 April 2025

(5) An act done to defeat, or purporting to operate so as to defeat,
the charge is of no effect against the contracted party.
(6) The charge is declared to be a statutory interest to which the
Personal Property Securities Act 2009 (Cwlth), section 73(2)
applies.
(7) The charge is inclusive of any GST related to the amount.

```

### `statute/chapter_2/section_034.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 34 — Contracting party withholding retention amount must
open retention trust account
(1) This section applies if, under section 32, a retention trust is
required for a retention amount withheld from payment under
a contract.
(2) The contracting party must open an account for the retention
trust at a financial institution—
(a) if the retention trust is required under
section 32(1)—before the retention amount is withheld;
or
(b) if the retention trust is required under
section 32(2)—within 20 business days after the
contract becomes a withholding contract.
Maximum penalty—500 penalty units.
(3) However, the contracting party need only establish 1 retention
trust account for all retention amounts withheld by the party
under any number of contracts for which it is the contracting
party.

```

### `statute/chapter_2/section_034A.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 34A — Restrictions for retention trust account
(1) A trustee must ensure the retention trust account is held at an
approved financial institution.
Maximum penalty—200 penalty units.

(2) A trustee must ensure the retention trust account is held under
a name that includes the trustee’s name and the word ‘trust’.
Maximum penalty—200 penalty units.
(3) A trustee must ensure that deposits of amounts to, and
withdrawals of amounts from, the retention trust account are
made using only methods that create an electronic record of
the transfer.
Maximum penalty—500 penalty units.
(4) A trustee must not close the retention trust account unless—
(a) all retention amounts held in the account have been
released to the parties entitled to it under the relevant
contracts; or
(b) the account is transferred to an alternative financial
institution under section 34C.
Maximum penalty—200 penalty units.

```

### `statute/chapter_2/section_034B.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 34B — Notice of retention trust account’s opening, closing or
name change
(1) This section applies if a trustee, or another person on behalf of
the trustee, takes any of the following actions in relation to the
retention trust account—
(a) opens the account;
(b) changes the name of the account;
(c) closes the account;
(d) transfers the account.
(2) Within 5 business days after taking the action, the trustee
must, using an approved way, give the commissioner a
notice—
(a) stating the action taken; and
(b) including the information prescribed by regulation.
Maximum penalty—200 penalty units.
Page 57 Current as at 27 April 2025

```

### `statute/chapter_2/section_034C.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 34C — Change of financial institution
(1) A trustee must not transfer the retention trust account to an
alternative financial institution unless—
(a) the alternative financial institution is an approved
financial institution; and
(b) all amounts held in the account are transferred with the
account to the alternative financial institution; and
(c) the trustee informs all contracted parties, from whom
retention amounts held in the account have been
withheld from payment, about the transfer as prescribed
by regulation.
Maximum penalty—200 penalty units.
Note—
See, also, section34B for the trustee’s obligation to inform the
commissioner of closing and opening a retention trust account.
(2) When transferring the retention trust account to an alternative
financial institution, the trustee may withdraw the amounts of
interest credited to the account by a financial institution.
(3) In this section—
alternative financial institution, for a retention trust account,
means a financial institution that is not the financial institution
at which the account is currently kept.

```

### `statute/chapter_2/section_035.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 35 — All retention amounts withheld must be deposited in
retention trust account
(1) This section applies if, under section 32, a retention trust is
required for a retention amount withheld from payment under
a contract.
(2) The contracting party must deposit the retention amount
withheld under the contract in a retention trust account (the
deposit obligation) as follows—

(a) if the retention trust is required under
section 32(1)—when the retention amount is withheld;
(b) if the retention trust is required under section32(2) and
there is an existing retention trust account into which the
retention amount may be deposited—within 5 business
days after the contract becomes a withholding contract;
(c) if the retention trust is required under section32(2) and
there is no existing retention trust account into which the
retention amount may be deposited—when the retention
trust account is opened.
Maximum penalty—200 penalty units or 2 years
imprisonment.
(3) A term of a contract is of no effect to the extent it is
inconsistent with the deposit obligation.
(4) The retention amount to which the deposit obligation relates is
inclusive of any GST related to the amount.

```

### `statute/chapter_2/section_035A.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 35A — Limited purposes for which money may be deposited in
retention trust account
(1) A trustee must not cause an amount to be deposited into the
retention trust account for any purpose other than—
(a) withholding a retention amount from payment under a
contract for which the trustee is the contracting party; or
(b) repaying an amount withdrawn in error.
Maximum penalty—200 penalty units or 1 year’s
imprisonment.
(2) This section does not apply to a deposit of an amount that is
interest earned on amounts held in a retention trust account.
Page 59 Current as at 27 April 2025

account

```

### `statute/chapter_2/section_036.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 36 — Limited purposes for which money may be withdrawn
from retention trust account
(1) A trustee must not withdraw an amount from the retention
trust account for any purpose other than—
(a) paying a beneficiary who is a contracted party from
whom a retention amount was withheld from payment;
or
(b) paying the trustee, as contracting party, for the purpose
of correcting defects or omissions in contracted work, or
otherwise to secure, wholly or partly, the performance of
a contract; or
(c) paying another person for the purpose of correcting
defects or omissions in contracted work.
Note—
As the contracting party for a contract, the trustee’s ability to make a
payment mentioned in paragraph(a), (b) or (c) would be governed by
the contract under which the retention amount was withheld.
Maximum penalty—300 penalty units or 2 years
imprisonment.
(2) A trustee must not withdraw an amount from the retention
trust account for a payment mentioned in subsection(1)(b)
until after the defects liability period, applying to the amount,
ends.
Maximum penalty—300 penalty units or 2 years
imprisonment.
(3) The trustee must repay all amounts the trustee withdraws in
contravention of subsection(1) as soon as practicable after
withdrawing the amount.
Maximum penalty—300 penalty units or 2 years
imprisonment.

(4) The trustee is taken to have withdrawn an amount from the
retention trust account if—
(a) the trustee authorises any person to make the
withdrawal; or
(b) the trustee knowingly contributes to the withdrawal
being made.
(5) This section does not apply to that part of a retention amount
withheld from payment under a contract that is beyond the
amount that may be lawfully withheld under the contract.
Note—
See the Queensland Building and Construction Commission Act 1991,
part4A for limits on retention amounts that may be withheld from
payment under a contract.

```

### `statute/chapter_2/section_036A.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 36A — All retention amounts withheld to be released from
retention trust account
(1) This section applies if a contracted party becomes entitled to
the release of a retention amount held in a retention trust.
(2) The trustee must not release the retention amount other than
by—
(a) withdrawing the amount from the retention trust
account; and
(b) depositing the amount into the contracted party’s
account at a financial institution.
Maximum penalty—200 penalty units or 1 year’s
imprisonment.

```

### `statute/chapter_2/section_037.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 37 — Ending retention trust
(1) If a retention trust is established for a retention amount
withheld from payment under a contract, the trust is dissolved
Page 61 Current as at 27 April 2025

if all of the amount has been released to the parties entitled to
it under the contract.
(2) On the dissolution of a retention trust, the trustee may pay
itself all amounts held in the trust that are not owing to
another beneficiary.

```

### `statute/chapter_2/section_038.txt`
```
[MISSING SOURCE FILE: source/statute/chapter_2/section_038.txt]
```

### `statute/chapter_2/section_039.txt`
```
[MISSING SOURCE FILE: source/statute/chapter_2/section_039.txt]
```

### `statute/chapter_2/section_040.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 40 — Notice of retention trust before withholding retention
amount
(1) If, under section 32, a retention trust is required for a retention
amount withheld from payment under a contract, the
contracting party must give the contracted party a notice about
the use of a retention trust (notice of retention trust) as
required by this section.
Maximum penalty—200 penalty units or 1 year’s
imprisonment.
(2) The notice of retention trust must—
(a) be in writing; and
(b) include a statement that a retention trust will be used for
withholding retention amounts under the contract; and

(c) include the information prescribed by regulation.
(3) The notice of retention trust must be given to the contracted
party before withholding the retention amount.
(4) However, if the retention trust account was not opened before
the retention trust is required under section32, the notice of
retention trust must be given to the contracted party within 5
business days after opening the account.

```

### `statute/chapter_2/section_040A.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 40A — Beneficiary to be informed of transactions affecting
retention amount
(1) This section applies if a trustee—
(a) deposits a retention amount into the retention trust
account; or
(b) withdraws all or part of a retention amount held in the
retention trust account.
(2) Within 5 business days after making the deposit or
withdrawal, the trustee must give to the contracted party from
whom the retention amount was withheld a notice of the
deposit or withdrawal that includes the information prescribed
by regulation, unless the trustee has a reasonable excuse.
Maximum penalty—100 penalty units.
(3) This section does not apply to the deposit of a retention
amount if the trustee has informed the contracted party of the
deposit under section23A.

```

### `statute/chapter_2/section_041.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 41 — Retention trust training
(1) This section applies if, under section 32, a retention trust is
required for a retention amount withheld from payment under
a contract.
(2) If the trustee will not be responsible for administering the
retention trust account, the trustee must nominate a person
who is responsible for administering the retention trust
account on behalf of the trustee.
(3) The trustee may change the nomination mentioned in
subsection(2) at any time and must make another nomination
if the previous nominee is no longer responsible for
administering the retention trust account.

(4) The trustee must, using an approved way, inform the
commissioner of each nomination made under subsection(2).
(5) The trustee must ensure each person nominated under
subsection(2) completes the training prescribed by regulation
(the retention trust training) within the period required by
regulation.
Maximum penalty—100 penalty units.
(6) If the trustee does not nominate a person under subsection(2),
the trustee must complete retention trust training within the
period required by regulation.
Maximum penalty—100 penalty units.
(7) A regulation may—
(a) prescribe a fee for retention trust training; or
(b) provide for—
(i) an extension of time for a trustee or nominee to
complete retention trust training; or
(ii) an exemption of a trustee from complying with
subsection(5) or (6).
(8) Subsections (5) and (6) apply to a trustee subject to an
extension or exemption under subsection(7) applying to the
trustee.
(9) The trustee is liable for all costs associated with the trustee or
a nominated person completing the retention trust training.

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
