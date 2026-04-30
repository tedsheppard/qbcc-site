"""
Element-page configuration for v3 build.

Each entry is one clickable sidebar item. The order in this list determines
the order in the sidebar. ``slug`` becomes the section anchor (id="...").
``sources`` lists which extracted source files the brief should bundle.

Source path conventions (relative to bif_guide_build/v3/source/):
  statute/chapter_N/section_NNN.txt   verbatim BIF Act
  annotated/section_NNN.txt           annotated commentary
  regs/reg_NNN.txt                    BIF Regulation
  other/qbcc_act_s42_unlicensed_work.txt
  other/aia_act_s39_service.txt
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Element:
    slug: str                  # e.g. "pc-construction-contract"
    title: str                 # e.g. "A construction contract must exist"
    breadcrumb: str            # e.g. "Requirements of a payment claim"
    statute: list[str] = field(default_factory=list)
    annotated: list[str] = field(default_factory=list)
    regs: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    statute_scope: str = ""    # optional note on what to extract from the statute
    extra: str = ""             # optional extra note for the agent


@dataclass
class Category:
    slug: str
    title: str
    elements: list[Element] = field(default_factory=list)


def s(num: str, ch: int) -> str:
    """Helper for statute path."""
    return f"statute/chapter_{ch}/section_{num}.txt"


def ann(num: str) -> str:
    """Helper for annotated path."""
    return f"annotated/section_{num}.txt"


# ---------------------------------------------------------------------------
# CATEGORY 1: Requirements of a payment claim
# ---------------------------------------------------------------------------
CAT1 = Category("pc", "Requirements of a payment claim", [
    Element("pc-construction-contract", "A construction contract must exist",
            "Requirements of a payment claim",
            statute=[s("064", 3)],
            annotated=[ann("064")],
            statute_scope='Show only the "construction contract" definition from s 64. Other defined terms in s 64 belong on later pages.'),
    Element("pc-construction-work", "The work must be construction work or related goods and services",
            "Requirements of a payment claim",
            statute=[s("065", 3), s("066", 3)],
            annotated=[ann("065"), ann("066")]),
    Element("pc-chapter-applies", "Chapter 3 must apply (and not be excluded)",
            "Requirements of a payment claim",
            statute=[s("061", 3), s("062", 3), s("063", 3)],
            annotated=[ann("061"), ann("062"), ann("063")]),
    Element("pc-entitled", "The claimant must be entitled to a progress payment",
            "Requirements of a payment claim",
            statute=[s("070", 3), s("071", 3), s("073", 3)],
            annotated=[ann("070"), ann("071"), ann("073")]),
    Element("pc-reference-date", "A valid reference date must have arisen",
            "Requirements of a payment claim",
            statute=[s("067", 3)],
            annotated=[ann("067")]),
    Element("pc-identify-work", "Identify the construction work",
            "Requirements of a payment claim",
            statute=[s("068", 3)],
            annotated=[ann("068")],
            statute_scope='Show only s 68(1)(a) and the introductory chapeau.'),
    Element("pc-amount", "State the claimed amount",
            "Requirements of a payment claim",
            statute=[s("068", 3)],
            annotated=[ann("068")],
            statute_scope='Show only s 68(1)(b) and the introductory chapeau.'),
    Element("pc-request-payment", "Request payment",
            "Requirements of a payment claim",
            statute=[s("068", 3)],
            annotated=[ann("068")],
            statute_scope='Show only s 68(1)(c) and the introductory chapeau.'),
    Element("pc-six-months", "Within six months",
            "Requirements of a payment claim",
            statute=[s("075", 3)],
            annotated=[ann("075")],
            statute_scope='Show only s 75(2)–(3) (the timing rule, including final-payment variant).'),
    Element("pc-one-claim", "Only one claim per reference date",
            "Requirements of a payment claim",
            statute=[s("075", 3)],
            annotated=[ann("075")],
            statute_scope='Show only s 75(4)–(5).'),
    Element("pc-not-unlicensed", "Not for unlicensed work",
            "Requirements of a payment claim",
            statute=[],
            annotated=[ann("075")],
            other=["other/qbcc_act_s42_unlicensed_work.txt"],
            extra='The relevant rule is QBCC Act s 42 (unlicensed work cannot be the subject of a recoverable claim). Discuss how this interacts with a BIF Act payment claim. Source the cases from the annotated s 75 commentary on illegality.'),
    Element("pc-served", "Served on the right person, the right way",
            "Requirements of a payment claim",
            statute=[s("075", 3), s("102", 3)],
            annotated=[ann("075"), ann("102")],
            other=["other/aia_act_s39_service.txt"],
            statute_scope='Show s 75(1) (who can give a claim and on whom) and s 102 (service of notices).'),
])

# ---------------------------------------------------------------------------
# CATEGORY 2: Requirements of a payment schedule
# ---------------------------------------------------------------------------
CAT2 = Category("ps", "Requirements of a payment schedule", [
    Element("ps-identify", "Identify the payment claim",
            "Requirements of a payment schedule",
            statute=[s("069", 3)],
            annotated=[ann("069")],
            statute_scope='Show only s 69(a) and chapeau.'),
    Element("ps-amount", "State the scheduled amount",
            "Requirements of a payment schedule",
            statute=[s("069", 3)],
            annotated=[ann("069")],
            statute_scope='Show only s 69(b) and chapeau.'),
    Element("ps-reasons", "Reasons for withholding payment",
            "Requirements of a payment schedule",
            statute=[s("069", 3)],
            annotated=[ann("069"), ann("076")],
            statute_scope='Show only s 69(c) and chapeau.'),
    Element("ps-within-time", "Within time (15 BD or shorter contractual)",
            "Requirements of a payment schedule",
            statute=[s("076", 3)],
            annotated=[ann("076")]),
    Element("ps-served", "Served on the claimant",
            "Requirements of a payment schedule",
            statute=[s("076", 3), s("102", 3)],
            annotated=[ann("076"), ann("102")],
            other=["other/aia_act_s39_service.txt"]),
])

# ---------------------------------------------------------------------------
# CATEGORY 3: Requirements of an adjudication application
# ---------------------------------------------------------------------------
CAT3 = Category("aa", "Requirements of an adjudication application", [
    Element("aa-trigger", "Trigger condition (no schedule / partial payment / scheduled less than claimed)",
            "Requirements of an adjudication application",
            statute=[s("078", 3), s("079", 3)],
            annotated=[ann("078"), ann("079")],
            statute_scope='Focus on s 79(1) and the related s 78 trigger.'),
    Element("aa-within-time", "Within time (windows depend on trigger)",
            "Requirements of an adjudication application",
            statute=[s("079", 3)],
            annotated=[ann("079")],
            statute_scope='Show s 79(2) (the time-window subsection).'),
    Element("aa-form-registrar", "On the approved form to the registrar",
            "Requirements of an adjudication application",
            statute=[s("079", 3)],
            annotated=[ann("079")]),
    Element("aa-identify", "Identify the payment claim and schedule",
            "Requirements of an adjudication application",
            statute=[s("079", 3)],
            annotated=[ann("079")]),
    Element("aa-fee", "Pay the prescribed fee",
            "Requirements of an adjudication application",
            statute=[s("079", 3)],
            annotated=[ann("079")],
            regs=["regs/reg_001.txt"],
            extra='The fee schedule sits in the Regulation. Reproduce the schedule (or the current top of it) verbatim, then explain in plain English what fee tier applies based on the claim amount.'),
    Element("aa-submission-limits", "Submission limits and the no-new-reasons rule",
            "Requirements of an adjudication application",
            statute=[s("079", 3)],
            annotated=[ann("079")],
            statute_scope='Show s 79(3) (limits on submissions for standard/complex claims).'),
    Element("aa-serve-respondent", "Serve a copy on the respondent",
            "Requirements of an adjudication application",
            statute=[s("079", 3), s("102", 3)],
            annotated=[ann("079"), ann("102")],
            other=["other/aia_act_s39_service.txt"]),
])

# ---------------------------------------------------------------------------
# CATEGORY 4: Requirements of an adjudication response
# ---------------------------------------------------------------------------
CAT4 = Category("ar", "Requirements of an adjudication response", [
    Element("ar-time-standard", "Within time — standard claims (10 BD)",
            "Requirements of an adjudication response",
            statute=[s("083", 3)],
            annotated=[ann("083")],
            statute_scope='Show s 83(1).'),
    Element("ar-time-complex", "Within time — complex claims (15 + 15 BD)",
            "Requirements of an adjudication response",
            statute=[s("083", 3)],
            annotated=[ann("083")],
            statute_scope='Show s 83(1)(b) and s 83(2).'),
    Element("ar-confined-reasons", "Confined to reasons in the payment schedule",
            "Requirements of an adjudication response",
            statute=[s("082", 3)],
            annotated=[ann("082")],
            statute_scope='Show s 82(4) and the surrounding subsections that fence off new reasons.'),
    Element("ar-form-content", "Form and content",
            "Requirements of an adjudication response",
            statute=[s("082", 3)],
            annotated=[ann("082")]),
    Element("ar-copy-claimant", "Copy to the claimant within 2 BD",
            "Requirements of an adjudication response",
            statute=[s("083", 3)],
            annotated=[ann("083")]),
])

# ---------------------------------------------------------------------------
# CATEGORY 5: Requirements of an adjudicator's decision
# ---------------------------------------------------------------------------
CAT5 = Category("d", "Requirements of an adjudicator's decision", [
    Element("d-eligibility", "The adjudicator must be eligible",
            "Requirements of an adjudicator's decision",
            statute=[s("080", 3)],
            annotated=[ann("080")]),
    Element("d-acceptance", "The adjudicator must accept the referral",
            "Requirements of an adjudicator's decision",
            statute=[s("081", 3)],
            annotated=[ann("081")]),
    Element("d-within-time", "Decided within time",
            "Requirements of an adjudicator's decision",
            statute=[s("085", 3), s("086", 3)],
            annotated=[ann("085"), ann("086")]),
    Element("d-natural-justice", "Procedural fairness / natural justice",
            "Requirements of an adjudicator's decision",
            statute=[s("084", 3)],
            annotated=[ann("084"), ann("088")],
            extra='Natural-justice issues run through the case law on s 88 challenges. Pull the relevant authorities from both s 84 (procedure) and s 88 (decision) annotated commentary.'),
    Element("d-applied-for", "Decide only what was applied for (jurisdictional limits)",
            "Requirements of an adjudicator's decision",
            statute=[s("088", 3)],
            annotated=[ann("088")],
            statute_scope='Show s 88(1) and (2).'),
    Element("d-must-consider", "Consider only what is permitted",
            "Requirements of an adjudicator's decision",
            statute=[s("088", 3)],
            annotated=[ann("088")],
            statute_scope='Show s 88(2)(a)–(d).'),
    Element("d-must-not-consider", "Not consider what is prohibited",
            "Requirements of an adjudicator's decision",
            statute=[s("088", 3)],
            annotated=[ann("088")],
            statute_scope='Show s 88(3).'),
    Element("d-reasons", "Give written reasons",
            "Requirements of an adjudicator's decision",
            statute=[s("088", 3)],
            annotated=[ann("088")],
            statute_scope='Show s 88(4)–(5).'),
    Element("d-valuation-later", "Bind valuation in later applications",
            "Requirements of an adjudicator's decision",
            statute=[s("087", 3)],
            annotated=[ann("087")]),
    Element("d-slip-rule", "Slip-rule corrections",
            "Requirements of an adjudicator's decision",
            statute=[s("089", 3)],
            annotated=[ann("089")]),
])

# ---------------------------------------------------------------------------
# CATEGORY 6: Enforcing an adjudicator's decision
# ---------------------------------------------------------------------------
CAT6 = Category("e", "Enforcing an adjudicator's decision", [
    Element("e-due-date", "Due date for paying the adjudicated amount",
            "Enforcing an adjudicator's decision",
            statute=[s("090", 3)],
            annotated=[ann("090")]),
    Element("e-certificate", "Adjudication certificate",
            "Enforcing an adjudicator's decision",
            statute=[s("091", 3)],
            annotated=[ann("091")]),
    Element("e-judgment-debt", "Filing the certificate as a judgment debt",
            "Enforcing an adjudicator's decision",
            statute=[s("093", 3)],
            annotated=[ann("093")],
            statute_scope='Show s 93 (1)–(3).'),
    Element("e-defence-bar", "What the respondent cannot raise as a defence",
            "Enforcing an adjudicator's decision",
            statute=[s("093", 3)],
            annotated=[ann("093")],
            statute_scope='Show s 93(4).'),
    Element("e-statutory-demand", "Statutory demand (insolvency pathway)",
            "Enforcing an adjudicator's decision",
            statute=[s("093", 3)],
            annotated=[ann("093")],
            extra='The statutory-demand pathway uses Corporations Act s 459E. The annotated s 93 commentary discusses the cases.'),
    Element("e-stays-injunctions", "Stays and injunctions",
            "Enforcing an adjudicator's decision",
            statute=[s("093", 3)],
            annotated=[ann("093")]),
])

# ---------------------------------------------------------------------------
# CATEGORY 7: Statutory suspension of works
# ---------------------------------------------------------------------------
CAT7 = Category("sus", "Statutory suspension of works", [
    Element("sus-trigger-scheduled", "Trigger — failure to pay scheduled amount",
            "Statutory suspension of works",
            statute=[s("078", 3)],
            annotated=[ann("078")],
            statute_scope='Show s 78(2)(a).'),
    Element("sus-trigger-adjudicated", "Trigger — failure to pay adjudicated amount",
            "Statutory suspension of works",
            statute=[s("092", 3)],
            annotated=[ann("092")]),
    Element("sus-notice", "Notice of intention to suspend (2 BD)",
            "Statutory suspension of works",
            statute=[s("078", 3), s("092", 3)],
            annotated=[ann("078"), ann("092"), ann("098")]),
    Element("sus-form", "Form and content of the notice",
            "Statutory suspension of works",
            statute=[s("098", 3)],
            annotated=[ann("098")]),
    Element("sus-when-begin", "When suspension may begin",
            "Statutory suspension of works",
            statute=[s("098", 3)],
            annotated=[ann("098")]),
    Element("sus-protection", "Protection from liability while suspended",
            "Statutory suspension of works",
            statute=[s("098", 3)],
            annotated=[ann("098")]),
    Element("sus-resumption", "Resumption when payment is made",
            "Statutory suspension of works",
            statute=[s("098", 3)],
            annotated=[ann("098")]),
])

# ---------------------------------------------------------------------------
# CATEGORY 8: Project trust accounts (Chapter 2 — current Act)
# ---------------------------------------------------------------------------
# Section numbers in Chapter 2 of the CURRENT Act differ from the older
# project-bank-account regime. The agent will be given the full Chapter 2
# statute as source and asked to scope to the relevant sections itself.
PTA_STATUTE_FULL = [s(f"{n:03d}", 2) for n in range(7, 33)]  # broad slice
PTA_REGS_FULL = ["regs/reg_010.txt", "regs/reg_011.txt", "regs/reg_012.txt", "regs/reg_013.txt", "regs/reg_014.txt", "regs/reg_015.txt"]

CAT8 = Category("pta", "Project trust accounts", [
    Element("pta-when-required", "When a project trust is required",
            "Project trust accounts",
            statute=PTA_STATUTE_FULL,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary is available for the current Chapter 2 (Statutory trusts) regime. Work directly from the statute and the regulations. Focus on Chapter 2 Part 2 sections setting out eligibility (project value thresholds, contract type carve-outs).'),
    Element("pta-opening", "Opening the project trust account",
            "Project trust accounts",
            statute=PTA_STATUTE_FULL,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: timing for opening, financial institution requirements, account name requirements, notification of opening.'),
    Element("pta-beneficiaries", "Subcontractor beneficiaries",
            "Project trust accounts",
            statute=PTA_STATUTE_FULL,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: who is a subcontractor beneficiary, when they become one, related-entity rules.'),
    Element("pta-deposits", "Deposits into the project trust",
            "Project trust accounts",
            statute=PTA_STATUTE_FULL,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: which payments must be deposited, timing of deposits, treatment of advance/upfront payments.'),
    Element("pta-withdrawals-subbie", "Withdrawals to pay subcontractor beneficiaries",
            "Project trust accounts",
            statute=PTA_STATUTE_FULL,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: when withdrawals to pay subbies are permitted, payment instruction process, order of payment.'),
    Element("pta-withdrawals-hc", "Withdrawals to pay the contracting party",
            "Project trust accounts",
            statute=PTA_STATUTE_FULL,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: when the contracting party (head contractor) can be paid from the trust, what conditions apply.'),
    Element("pta-notifications", "Notifications to QBCC, principal and subcontractors",
            "Project trust accounts",
            statute=PTA_STATUTE_FULL,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: notification to commissioner of opening / closing / changes; subcontractor notification of trust establishment; principal information.'),
    Element("pta-records", "Records and reporting",
            "Project trust accounts",
            statute=[s(f"{n:03d}", 2) for n in range(50, 59)],
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: bookkeeping requirements, trust account reviews / audits, retention of records.'),
    Element("pta-closing", "Closing the project trust",
            "Project trust accounts",
            statute=PTA_STATUTE_FULL,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: when a trust must be closed, distribution of remaining funds, notice of closure.'),
    Element("pta-penalties", "Penalties for non-compliance",
            "Project trust accounts",
            statute=PTA_STATUTE_FULL,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Survey the offence provisions in Chapter 2 — maximum penalties, executive officer liability, common compliance traps.'),
])

# ---------------------------------------------------------------------------
# CATEGORY 9: Retention trust accounts
# ---------------------------------------------------------------------------
RTA_STATUTE_FULL = [s(f"{n:03d}{l}", 2) for n in range(33, 42) for l in ("", "A", "B", "C") if l == "" or n in (33, 34, 35, 36, 40)]
# Add specific files we know exist; missing ones will be skipped by the brief generator
RTA_STATUTE_EXPLICIT = [
    s("033", 2), s("033A", 2), s("034", 2), s("034A", 2), s("034B", 2), s("034C", 2),
    s("035", 2), s("035A", 2), s("036", 2), s("036A", 2), s("037", 2), s("038", 2),
    s("039", 2), s("040", 2), s("040A", 2), s("041", 2),
]

CAT9 = Category("rta", "Retention trust accounts", [
    Element("rta-when-required", "When a retention trust is required",
            "Retention trust accounts",
            statute=RTA_STATUTE_EXPLICIT,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: project value thresholds for retention trust requirement; relationship to project trust; contract type carve-outs.'),
    Element("rta-what-goes-in", "What money is held in retention trust",
            "Retention trust accounts",
            statute=RTA_STATUTE_EXPLICIT,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: which cash retention amounts must be held in trust; treatment of bank guarantees.'),
    Element("rta-opening", "Opening the retention trust account",
            "Retention trust accounts",
            statute=RTA_STATUTE_EXPLICIT,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: timing, account name, financial institution.'),
    Element("rta-beneficiaries", "Beneficiaries (whose retention it is)",
            "Retention trust accounts",
            statute=RTA_STATUTE_EXPLICIT,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary.'),
    Element("rta-releasing", "Releasing retention back to the subcontractor",
            "Retention trust accounts",
            statute=RTA_STATUTE_EXPLICIT,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: when retention must be released, notification to subbie, dispute procedure.'),
    Element("rta-set-off", "Set-off / contract claims against retention",
            "Retention trust accounts",
            statute=RTA_STATUTE_EXPLICIT,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: when retention can be applied to contract claims, dispute mechanism.'),
    Element("rta-notifications", "Notifications",
            "Retention trust accounts",
            statute=RTA_STATUTE_EXPLICIT,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary.'),
    Element("rta-records", "Records, reporting, training and audit",
            "Retention trust accounts",
            statute=[s(f"{n:03d}", 2) for n in range(50, 59)],
            regs=PTA_REGS_FULL,
            extra='No annotated commentary. Cover: training requirements for trustee, annual reviews, retention of records.'),
    Element("rta-penalties", "Penalties for non-compliance",
            "Retention trust accounts",
            statute=RTA_STATUTE_EXPLICIT,
            regs=PTA_REGS_FULL,
            extra='No annotated commentary.'),
])

# ---------------------------------------------------------------------------
# CATEGORY 10: Valid subcontractor's charge
# ---------------------------------------------------------------------------
CAT10 = Category("sc", "Valid subcontractor's charge", [
    Element("sc-who-can", "Who can give a notice of charge",
            "Valid subcontractor's charge",
            statute=[s("122", 4), s("104", 4)],
            annotated=[ann("122"), ann("104")]),
    Element("sc-what-money", "What money can be charged",
            "Valid subcontractor's charge",
            statute=[s("113", 4), s("122", 4)],
            annotated=[ann("113"), ann("122")]),
    Element("sc-content", "Content of the notice of claim",
            "Valid subcontractor's charge",
            statute=[s("122", 4), s("123", 4), s("124", 4)],
            annotated=[ann("122"), ann("123"), ann("124")]),
    Element("sc-supporting-cert", "Supporting certificate (s 124)",
            "Valid subcontractor's charge",
            statute=[s("124", 4)],
            annotated=[ann("124")]),
    Element("sc-timing", "Timing of the notice",
            "Valid subcontractor's charge",
            statute=[s("122", 4), s("125", 4)],
            annotated=[ann("122"), ann("125")]),
    Element("sc-service", "Service of the notice",
            "Valid subcontractor's charge",
            statute=[s("122", 4), s("123", 4)],
            annotated=[ann("122"), ann("123"), ann("102")],
            other=["other/aia_act_s39_service.txt"]),
    Element("sc-effect-ch3", "Effect on the Chapter 3 process",
            "Valid subcontractor's charge",
            statute=[s("062", 3)],
            annotated=[ann("062")]),
    Element("sc-withdrawal", "Withdrawal of a notice",
            "Valid subcontractor's charge",
            statute=[s("125", 4)],
            annotated=[ann("125")]),
])

# ---------------------------------------------------------------------------
# CATEGORY 11: Enforcing a subcontractor's charge
# ---------------------------------------------------------------------------
CAT11 = Category("ec", "Enforcing a subcontractor's charge", [
    Element("ec-court-proceedings", "Court proceedings to recover",
            "Enforcing a subcontractor's charge",
            statute=[s("130", 4), s("131", 4)],
            annotated=[ann("130"), ann("131")]),
    Element("ec-time-limit", "Time limit to commence",
            "Enforcing a subcontractor's charge",
            statute=[s("130", 4)],
            annotated=[ann("130")]),
    Element("ec-parties", "Necessary parties / joinder",
            "Enforcing a subcontractor's charge",
            statute=[s("131", 4)],
            annotated=[ann("131")]),
    Element("ec-contractor-response", "The contractor's response",
            "Enforcing a subcontractor's charge",
            statute=[s("128", 4), s("132", 4)],
            annotated=[ann("128"), ann("132")]),
    Element("ec-set-off", "Set-off and counterclaim by the contractor",
            "Enforcing a subcontractor's charge",
            statute=[s("133", 4)],
            annotated=[ann("133")]),
    Element("ec-discharge", "Discharge of the charge",
            "Enforcing a subcontractor's charge",
            statute=[s("135", 4)],
            annotated=[ann("135")]),
    Element("ec-insolvency", "Effect of insolvency of the contractor",
            "Enforcing a subcontractor's charge",
            statute=[s("137", 4)],
            annotated=[ann("137")]),
    Element("ec-costs", "Costs",
            "Enforcing a subcontractor's charge",
            statute=[s("138", 4), s("145", 4)],
            annotated=[ann("138"), ann("145")]),
])

ALL_CATEGORIES: list[Category] = [CAT1, CAT2, CAT3, CAT4, CAT5, CAT6, CAT7, CAT8, CAT9, CAT10, CAT11]


def all_elements() -> list[Element]:
    out: list[Element] = []
    for c in ALL_CATEGORIES:
        out.extend(c.elements)
    return out


if __name__ == "__main__":
    cats = ALL_CATEGORIES
    total = sum(len(c.elements) for c in cats)
    print(f"{len(cats)} categories, {total} elements")
    for c in cats:
        print(f"  [{c.slug}] {c.title}: {len(c.elements)} elements")
