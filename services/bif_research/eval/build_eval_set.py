"""Generate eval_set.jsonl with 30 hand-curated questions across 7 categories.

Hand-curated rather than LLM-generated because (a) the categories are tightly
specified, (b) each question's expected_sources need to actually exist in the
corpus, and (c) the corpus is well-known to the author from the BIF Act guide
v3 build. Saves $0.20-$0.50 of LLM spend versus auto-generation.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import EvalQuestion, ExpectedSource, question_to_dict


# ---------------------------------------------------------------------------
# Statutory test (6)
# ---------------------------------------------------------------------------

STATUTORY_TEST = [
    EvalQuestion(
        id="q01",
        question="What is the test for a valid payment claim under the BIF Act?",
        category="statutory_test",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="68"),
            ExpectedSource(type="statute", act="BIF Act", section="75"),
        ],
        expected_answer_summary=(
            "A payment claim under s 68 must be a written document that identifies the construction "
            "work or related goods and services, states the amount claimed, and requests payment. "
            "It must be served by an entitled person under s 75(1), within the time limits in s 75(2), "
            "and only one claim per reference date is permitted (s 75(4))."
        ),
        must_not_contain=["s 13", "Building and Construction Industry Security of Payment Act 1999"],
    ),
    EvalQuestion(
        id="q02",
        question="What is the test for a valid payment schedule?",
        category="statutory_test",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="69"),
            ExpectedSource(type="statute", act="BIF Act", section="76"),
        ],
        expected_answer_summary=(
            "A payment schedule under s 69 must identify the payment claim, state the scheduled "
            "amount the respondent proposes to pay (which may be nil), and where the scheduled amount "
            "is less than the claimed amount, state the reasons for withholding. It must be given "
            "within the time in s 76."
        ),
    ),
    EvalQuestion(
        id="q03",
        question="What test does an adjudicator apply to determine whether construction work has been validly identified in a payment claim?",
        category="statutory_test",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="68"),
            ExpectedSource(type="case", name_pattern="MWB Everton Park"),
            ExpectedSource(type="case", name_pattern="KDV Sport"),
        ],
        expected_answer_summary=(
            "The work must be 'reasonably comprehensible' to the recipient — that is, identified "
            "with sufficient particularity for the respondent to determine whether to pay. "
            "Trade summaries with percentage completions are usually insufficient. The test is not "
            "whether the claim is correct but whether the work to which it relates is identified."
        ),
    ),
    EvalQuestion(
        id="q04",
        question="When is a contract a 'construction contract' for the purposes of the BIF Act?",
        category="statutory_test",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="64"),
            ExpectedSource(type="case", name_pattern="Capricorn Quarries"),
        ],
        expected_answer_summary=(
            "Section 64 defines construction contract as a contract, agreement or other arrangement "
            "under which one party undertakes to carry out construction work for, or supply related "
            "goods and services to, another party. The Queensland authority Capricorn Quarries v "
            "Devcon adopted a broad reading of the contract limb."
        ),
    ),
    EvalQuestion(
        id="q05",
        question="What constitutes 'construction work' under the BIF Act?",
        category="statutory_test",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="65"),
        ],
        expected_answer_summary=(
            "Section 65 defines construction work broadly: construction, alteration, repair, "
            "restoration, maintenance or extension of buildings or structures, demolition, "
            "site preparation, integral or preparatory operations, etc. Mining extraction is "
            "excluded by s 65(3) but tunnelling/boring not for the actual purpose of extraction "
            "remains construction work."
        ),
    ),
    EvalQuestion(
        id="q06",
        question="What test do the courts apply to determine whether an adjudicator's decision contains jurisdictional error?",
        category="statutory_test",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="88"),
            ExpectedSource(type="case", name_pattern="Brodyn"),
            ExpectedSource(type="case", name_pattern="Chase Oyster Bar"),
        ],
        expected_answer_summary=(
            "Following Brodyn and Chase Oyster Bar, jurisdictional error includes failures to comply "
            "with the basic and essential requirements of the Act, denial of a substantial measure of "
            "natural justice, lack of a bona fide attempt to exercise the power, or deciding outside "
            "the matters listed in s 88(2). Mere errors of fact or law within jurisdiction are not "
            "jurisdictional error."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Definition (4)
# ---------------------------------------------------------------------------

DEFINITION = [
    EvalQuestion(
        id="q07",
        question="What is a 'reference date' under the BIF Act?",
        category="definition",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="67"),
        ],
        expected_answer_summary=(
            "Section 67 defines reference date as a date stated in or worked out under the contract "
            "for making a payment claim. If the contract makes no provision, the default reference "
            "date is the last day of the month in which the construction work was first carried out, "
            "and the last day of each later named month."
        ),
    ),
    EvalQuestion(
        id="q08",
        question="What does 'related goods and services' mean under the BIF Act?",
        category="definition",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="66"),
        ],
        expected_answer_summary=(
            "Section 66 defines related goods and services as goods or services for use in connection "
            "with construction work, including materials and components forming part of buildings, "
            "plant or materials for hire, professional services such as design and project management, "
            "and labour."
        ),
    ),
    EvalQuestion(
        id="q09",
        question="What is a 'complex payment claim' under the BIF Act?",
        category="definition",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="64"),
        ],
        expected_answer_summary=(
            "A complex payment claim is defined in s 64 as a payment claim for an amount more than "
            "$750,000 (or a greater amount prescribed by regulation). Complex claims attract longer "
            "timeframes for adjudication response and decision."
        ),
    ),
    EvalQuestion(
        id="q10",
        question="What is a 'progress payment' under the BIF Act?",
        category="definition",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="64"),
            ExpectedSource(type="statute", act="BIF Act", section="70"),
        ],
        expected_answer_summary=(
            "Section 64 defines progress payment to include a payment to which a person is entitled "
            "under s 70, the final payment, a single or one-off payment, or a milestone payment. "
            "Section 70 sets the entitlement: from each reference date, a person who has carried out "
            "construction work or supplied related goods and services is entitled to a progress payment."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Procedural deadline (5)
# ---------------------------------------------------------------------------

PROCEDURAL_DEADLINE = [
    EvalQuestion(
        id="q11",
        question="How long does a respondent have to give a payment schedule in response to a payment claim under the BIF Act?",
        category="procedural_deadline",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="76"),
        ],
        expected_answer_summary=(
            "Section 76 requires the payment schedule to be given within the period stated in the "
            "construction contract, or 15 business days after the payment claim is given, whichever "
            "is earlier."
        ),
    ),
    EvalQuestion(
        id="q12",
        question="How long does an adjudicator have to decide a standard payment claim?",
        category="procedural_deadline",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="85"),
            ExpectedSource(type="statute", act="BIF Act", section="86"),
        ],
        expected_answer_summary=(
            "Section 85 sets 10 business days from the response date for a standard claim, with the "
            "ability under s 86 for the adjudicator to extend by agreement of the parties or for "
            "complex claims."
        ),
    ),
    EvalQuestion(
        id="q13",
        question="What is the time limit for serving a payment claim under the BIF Act?",
        category="procedural_deadline",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="75"),
        ],
        expected_answer_summary=(
            "Section 75(2) requires the claim to be given before the end of the longer of the period "
            "(if any) worked out under the contract, or 6 months after the construction work was last "
            "carried out (or related goods and services last supplied). Section 75(3) extends this for "
            "final payment claims."
        ),
    ),
    EvalQuestion(
        id="q14",
        question="How long does a respondent have to give an adjudication response for a standard payment claim?",
        category="procedural_deadline",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="83"),
        ],
        expected_answer_summary=(
            "Section 83 requires the response to be given within 10 business days after the respondent "
            "receives a copy of the adjudication application, or 7 business days after the respondent "
            "receives notice from the adjudicator of acceptance, whichever is the later."
        ),
    ),
    EvalQuestion(
        id="q15",
        question="When is an adjudicated amount required to be paid under the BIF Act?",
        category="procedural_deadline",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="90"),
        ],
        expected_answer_summary=(
            "Section 90 requires the respondent to pay the adjudicated amount on or before 5 business "
            "days after the adjudicator's decision is given to the respondent, or such later date as "
            "the adjudicator may decide."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Case law (6)
# ---------------------------------------------------------------------------

CASE_LAW = [
    EvalQuestion(
        id="q16",
        question="How have Queensland courts treated the requirement that a payment claim 'identify' the construction work to which it relates?",
        category="case_law",
        expected_sources=[
            ExpectedSource(type="case", name_pattern="MWB Everton Park"),
            ExpectedSource(type="case", name_pattern="KDV Sport"),
            ExpectedSource(type="case", name_pattern="T & M Buckley"),
            ExpectedSource(type="statute", act="BIF Act", section="68"),
        ],
        expected_answer_summary=(
            "The leading Queensland appellate authority is T & M Buckley v 57 Moss Rd, adopting the "
            "'reasonably comprehensible' test. MWB Everton Park v Devcon held that trade-summary "
            "percentages without reference to identifiable scopes are insufficient. KDV Sport v "
            "Muggeridge applied this where a one-page trade breakdown failed the identification limb."
        ),
    ),
    EvalQuestion(
        id="q17",
        question="What is the test for an 'other arrangement' in the BIF Act definition of construction contract under Queensland authority?",
        category="case_law",
        expected_sources=[
            ExpectedSource(type="case", name_pattern="Capricorn Quarries"),
            ExpectedSource(type="statute", act="BIF Act", section="64"),
        ],
        expected_answer_summary=(
            "Capricorn Quarries v Devcon (Daubney J) is the leading Queensland reading of the "
            "contract limb of the BIF Act definition. The 'other arrangement' words require some "
            "legal nexus — typically a binding obligation or a concluded state of affairs with "
            "reciprocity around payment. NSW first-instance authority has divided on the point but "
            "Queensland courts adopt a broad reading consistent with Capricorn Quarries."
        ),
    ),
    EvalQuestion(
        id="q18",
        question="What are the consequences if a respondent fails to give a payment schedule within time?",
        category="case_law",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="77"),
            ExpectedSource(type="statute", act="BIF Act", section="78"),
            ExpectedSource(type="case", name_pattern="Fyntray"),
        ],
        expected_answer_summary=(
            "Under s 77 and s 78, the respondent becomes liable to pay the claimed amount as a debt. "
            "The claimant may either commence court proceedings to recover the unpaid amount as a debt "
            "(s 78(2)(a)) or apply for adjudication after a warning notice (s 99). Fyntray Constructions "
            "is the leading authority on the strict consequences of non-compliance."
        ),
    ),
    EvalQuestion(
        id="q19",
        question="What is the leading authority on the requirement that an adjudicator make a 'bona fide' or honest attempt to exercise the power?",
        category="case_law",
        expected_sources=[
            ExpectedSource(type="case", name_pattern="Brodyn"),
            ExpectedSource(type="statute", act="BIF Act", section="88"),
        ],
        expected_answer_summary=(
            "Brodyn Pty Ltd v Davenport [2004] NSWCA 394 established that a bona fide attempt to "
            "exercise the power is one of the basic and essential requirements; absence of such an "
            "attempt is jurisdictional error. The principle has been applied in Queensland and "
            "remains good law for those features that survived Chase Oyster Bar."
        ),
    ),
    EvalQuestion(
        id="q20",
        question="How have the courts treated the 'reference date' requirement after Southern Han?",
        category="case_law",
        expected_sources=[
            ExpectedSource(type="case", name_pattern="Southern Han"),
            ExpectedSource(type="case", name_pattern="Lean Field"),
            ExpectedSource(type="statute", act="BIF Act", section="67"),
        ],
        expected_answer_summary=(
            "Southern Han Breakfast Point v Lewence Construction [2016] HCA 52 confirmed at the High "
            "Court level that a reference date arising under the contract is a precondition to a valid "
            "payment claim, not merely a timing rule. In Queensland, Lean Field Developments v E & I "
            "Global Solutions [2014] QSC 293 is the leading exposition of 'worked out under' the "
            "contract, allowing post-formation events to operate the formula."
        ),
    ),
    EvalQuestion(
        id="q21",
        question="What approach do Queensland courts take to severance of an adjudicator's decision where part of the decision is affected by jurisdictional error?",
        category="case_law",
        expected_sources=[
            ExpectedSource(type="case", name_pattern="Niclin"),
            ExpectedSource(type="case", name_pattern="Probuild"),
            ExpectedSource(type="statute", act="BIF Act", section="88"),
        ],
        expected_answer_summary=(
            "Queensland courts have allowed severance of an adjudicator's decision where the affected "
            "part can be cleanly separated from the unaffected part. Niclin Constructions v Robotic "
            "Steel Fab [2019] QCA 177 considered severance principles. Where severance is not possible, "
            "the entire decision must be set aside."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Section precision (4)
# ---------------------------------------------------------------------------

SECTION_PRECISION = [
    EvalQuestion(
        id="q22",
        question="What does s 75(4) of the BIF Act say about multiple payment claims for the same reference date?",
        category="section_precision",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="75"),
        ],
        expected_answer_summary=(
            "Section 75(4) provides that the claimant cannot make more than 1 payment claim for each "
            "reference date under the construction contract. Section 75(5) clarifies that a payment "
            "claim may include an amount that was included in a previous payment claim."
        ),
    ),
    EvalQuestion(
        id="q23",
        question="What matters must an adjudicator consider under s 88(2) of the BIF Act?",
        category="section_precision",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="88"),
        ],
        expected_answer_summary=(
            "Section 88(2) requires the adjudicator to consider only the provisions of the BIF Act, "
            "the construction contract, the payment claim and the documents accompanying it, the "
            "payment schedule (if any) and the documents accompanying it, the adjudication application "
            "and the documents accompanying it, the adjudication response and the documents "
            "accompanying it (other than reasons not in the schedule), and the results of any "
            "inspection."
        ),
    ),
    EvalQuestion(
        id="q24",
        question="What does s 102 of the BIF Act provide about service of notices?",
        category="section_precision",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="102"),
        ],
        expected_answer_summary=(
            "Section 102 sets out methods of service: in person, by post to the recipient's usual or "
            "last known place of business or residence, by leaving at that place, by fax or other "
            "method specified in the contract. Service by other means is also permitted under the "
            "Acts Interpretation Act 1954 (Qld), s 39."
        ),
    ),
    EvalQuestion(
        id="q25",
        question="What does s 93(4) of the BIF Act say about defences a respondent cannot raise when an adjudication certificate is filed as a judgment debt?",
        category="section_precision",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="93"),
        ],
        expected_answer_summary=(
            "Section 93(4) bars the respondent from raising any counterclaim, raising any defence in "
            "relation to matters arising under the construction contract, or challenging the "
            "adjudicator's decision in those proceedings. The respondent is limited to challenging "
            "the certificate on jurisdictional grounds in separate proceedings."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Cross-reference (3)
# ---------------------------------------------------------------------------

CROSS_REFERENCE = [
    EvalQuestion(
        id="q26",
        question="How does s 62 of the BIF Act interact with Chapter 4 (subcontractors' charges)?",
        category="cross_reference",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="62"),
            ExpectedSource(type="case", name_pattern="Heavy Plant Leasing"),
        ],
        expected_answer_summary=(
            "Section 62 provides that if a person gives a notice of claim of charge under Chapter 4 "
            "in relation to construction work or related goods and services, that person cannot start "
            "or continue Chapter 3 proceedings for the same scope. Pending adjudications are taken "
            "to be withdrawn; decided adjudications cannot be enforced. The freeze ends if the notice "
            "is effectively withdrawn (Heavy Plant Leasing v McConnell Dowell)."
        ),
    ),
    EvalQuestion(
        id="q27",
        question="What is the relationship between s 78 and s 92 of the BIF Act in terms of the suspension trigger?",
        category="cross_reference",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="78"),
            ExpectedSource(type="statute", act="BIF Act", section="92"),
            ExpectedSource(type="statute", act="BIF Act", section="98"),
        ],
        expected_answer_summary=(
            "Both s 78 (failure to pay scheduled amount) and s 92 (failure to pay adjudicated amount) "
            "operate as triggers for the s 98 right to suspend. Section 78(3) and s 92 both require a "
            "2-business-day notice of intention to suspend before the s 98 right can be exercised."
        ),
    ),
    EvalQuestion(
        id="q28",
        question="How does QBCC Act s 42 interact with a payment claim under the BIF Act for unlicensed building work?",
        category="cross_reference",
        expected_sources=[
            ExpectedSource(type="statute", act="QBCC Act", section="42"),
            ExpectedSource(type="statute", act="BIF Act", section="75"),
        ],
        expected_answer_summary=(
            "Section 42(3) of the QBCC Act provides that a person who carries out building work without "
            "the appropriate licence is not entitled to monetary or other consideration for that work. "
            "This bar applies to BIF Act payment claims: a payment claim for unlicensed work cannot "
            "succeed because the underlying entitlement is barred. Section 42(4) provides a limited "
            "carve-out for reasonable remuneration on certain heads."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Amendment currency (2)
# ---------------------------------------------------------------------------

AMENDMENT_CURRENCY = [
    EvalQuestion(
        id="q29",
        question="What is the current threshold for a 'complex payment claim' under the BIF Act?",
        category="amendment_currency",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="64"),
        ],
        expected_answer_summary=(
            "The threshold in the s 64 definition is more than $750,000, or such greater amount as "
            "is prescribed by regulation. The system should not assert a different figure unless the "
            "BIF Regulation expressly prescribes one."
        ),
    ),
    EvalQuestion(
        id="q30",
        question="Does the BIF Act still use 'reference date' terminology, or has it been replaced by recent amendments?",
        category="amendment_currency",
        expected_sources=[
            ExpectedSource(type="statute", act="BIF Act", section="67"),
            ExpectedSource(type="statute", act="BIF Act", section="70"),
        ],
        expected_answer_summary=(
            "The BIF Act (current as at April 2025) retains the 'reference date' terminology in s 67 "
            "and s 70. The 2020/2021 amendments overhauled Chapter 2 (project trust accounts) but did "
            "not remove the reference-date concept from Chapter 3."
        ),
    ),
]


ALL_QUESTIONS = (
    STATUTORY_TEST + DEFINITION + PROCEDURAL_DEADLINE +
    CASE_LAW + SECTION_PRECISION + CROSS_REFERENCE + AMENDMENT_CURRENCY
)


def main() -> None:
    out_path = Path(__file__).resolve().parent / "eval_set.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for q in ALL_QUESTIONS:
            f.write(json.dumps(question_to_dict(q), ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for q in ALL_QUESTIONS:
        counts[q.category] = counts.get(q.category, 0) + 1
    print(f"Wrote {len(ALL_QUESTIONS)} questions to {out_path}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
