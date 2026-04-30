# Element-page brief — rta-records
**Title:** Records, reporting, training and audit
**Breadcrumb:** Retention trust accounts
**Anchor id:** `rta-records`
**Output file:** `bif_guide_build/v3/pages/page_rta-records.html`

## Extra guidance for this element
No annotated commentary. Cover: training requirements for trustee, annual reviews, retention of records.

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

### `statute/chapter_2/section_050.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 50 — Definitions for part
In this part—
account review report see section 57A(1).
auditor means—
(a) a person registered as an auditor under the Corporations
Act, part9.2; or
(b) a person prescribed by regulation to be an auditor.
chapter 2 requirement means a requirement, applying to a
trustee under chapter 2, in relation to the administration of a
trust account.
trust account means a project trust account or retention trust
account.

for trustees

```

### `statute/chapter_2/section_051.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 51 — Trustee to cover shortfalls
(1) This section applies if there is an insufficient amount available
in a trust account to pay an amount due to be paid to a
beneficiary of the trust.
(2) The trustee must immediately deposit an amount equal to the
shortfall in the trust account (a shortfall deposit).
Maximum penalty—100 penalty units or 1 year’s
imprisonment.
(3) In this section—

shortfall, in a trust account, means an amount equal to the
difference between the amount available in the trust account
for payment and the amount to be paid from the account.

```

### `statute/chapter_2/section_052.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 52 — Trust records
The trustee for a project trust or retention trust must—
(a) keep records for the trust as prescribed by regulation;
and
(b) retain the records for at least 7 years after the trust is
dissolved.
Maximum penalty—300 penalty units or 1 year’s
imprisonment.

commissioner

```

### `statute/chapter_2/section_053.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 53 — Register of project trusts and retention trusts
(1) The commissioner must maintain a register of the project
trusts and retention trusts of which the commissioner has been
notified.

(2) The commissioner may publish information about the project
trusts and retention trusts in the way decided by the
commissioner.

```

### `statute/chapter_2/section_054.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 54 — Definitions for division
In this division—

accepted representations see section54C(2).
show cause notice see section54B(2).
show cause period see section54B(2)(d).

```

### `statute/chapter_2/section_055.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 55 — Approval of financial institutions
(1) The commissioner may approve the financial institutions at
which trust accounts may be kept.
(2) However, the commissioner may approve a financial
institution under subsection(1) only if the financial institution
has entered into an agreement with the commission about
providing financial services for trust accounts.
(3) The agreement may provide for the following matters—
(a) a requirement to inform the commissioner of amounts
held in trust accounts;
(b) the auditing of trust accounts;
(c) other matters prescribed by regulation.
(4) The commissioner may, if the financial institution agrees,
amend or revoke the agreement.
(5) The commissioner must publish the names of all financial
institutions approved under section (1) on the commission’s
website.

```

### `statute/chapter_2/section_056.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 56 — Application of Personal Property Securities Act 2009
(Cwlth)
(1) A project trust or retention trust—
(a) is declared to be a statutory interest to which
section 73(2) of the Personal Property Securities Act
2009 (Cwlth) applies; and

(b) has priority over all security interests in relation to all
funds held in trust for the project trust or retention trust.
(2) In this section—
security interest has the meaning given by the Personal
Property Securities Act 2009 (Cwlth), section 12.

```

### `statute/chapter_2/section_057.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 57 — Engaging auditor for review of trust account
(1) The trustee for a project trust or retention trust must engage an
auditor to carry out a review of the trust account as required
by this section.
Maximum penalty—200 penalty units or 1 year’s
imprisonment.
Page 83 Current as at 27 April 2025

(2) The engagement of the auditor must comply with the
requirements prescribed by regulation.
(3) The review must be carried out at the times prescribed by
regulation.
(4) The period of the review (the review period) is the period
prescribed by regulation.
(5) The review must be complete within 40 business days after
starting the review.
(6) The review must be carried out by an auditor that is
independent of the trustee and has not been excluded by the
commissioner under section54E.
(7) An auditor is independent of the trustee if the auditor is not
any of the following—
(a) an employee of the trustee;
(b) if the trustee is a company—an executive officer,
investor or shareholder for the company;
(c) if the trustee is a partnership—a partner in the
partnership;
(d) a related entity for the trustee.
(8) The trustee need not engage an auditor to carry out a review of
the trust account—
(a) if—
(i) a retention amount was not held in the account
during the review period; and
(ii) within 10 business days after the end of the review
period the trustee gave the commissioner a written
statement, using an approved way, as to why the
trustee did not engage an auditor to carry out the
review; or
(b) if—
(i) there are no transactions or changes for the account
during the review period; and

(ii) within 10 business days after the end of the review
period the trustee gave the commissioner a written
statement, using an approved way, confirming the
matter mentioned in subparagraph(i); or
(c) in the circumstances prescribed by regulation.

```

### `statute/chapter_2/section_058.txt`
```
# Source: BIF Act 2017 — Chapter 2
# Section 58 — Commissioner may give redacted information to
professional bodies
(1) This section applies if the commissioner reasonably suspects
the conduct of an auditor engaged under section 57 breaches a
professional standard or condition applying to the auditor.

(2) The commissioner may inform the relevant professional body
of the conduct and give the body any information necessary to
investigate the conduct.
(3) The commissioner must redact from any information given to
the professional body all information identifying the trustee of
a project trust or retention trust.
(4) In this section—
condition means a condition imposed as part of registration as
an auditor or accountant.
professional body means—
(a) an entity of which an auditor is a member as an auditor
or accountant; or
(b) an entity that registers or licenses a person as an auditor.
professional standard means a standard about auditing or
accounting made, or adopted, by a professional body.

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
