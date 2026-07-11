#!/usr/bin/env python3
"""
Keyword-code the Authorised Nominating Authority (ANA) for pre-2015 BCIPA
decisions in qbcc.db.

ANAs referred adjudication applications to adjudicators under BCIPA 2004 (Qld)
until the Dec-2014 amendments moved referrals to the QBCC Adjudication
Registry, so only rows with decision_date < 2015-01-01 are touched here.
Rows from 2015 onward are coded separately (ana_source='ai').

Matching strategy
-----------------
PDF text extraction breaks words with stray spaces/newlines ("Con struction"),
so we strip ALL whitespace and lowercase before substring matching, keeping an
offset map back to the original text for evidence snippets. Acronyms that
would false-positive inside whitespace-stripped text (RICS in "fabrics", IAMA,
LEADR in "available ADR", Nominator in "denominator") are matched with
word-boundary regexes on the ORIGINAL text instead.

Priority: earliest match in the first ~6000 normalised chars (coversheet /
letterhead names the referring ANA) wins; failing that the last ~3000 chars;
failing that anywhere in the document (recorded as lower confidence:
ana_source='keyword_lowconf').

Outputs
-------
- decision_details.ana / ana_source / ana_evidence updated (short batched
  transactions; WAL handles the concurrent >=2015 writer).
- exports/ana_keyword.json with {ejs_id, ana, ana_source, ana_evidence} for
  every scanned row, for replay against the live GCS DB.
"""

import json
import os
import random
import re
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "qbcc.db")
EXPORT_PATH = os.path.join(ROOT, "exports", "ana_keyword.json")

HEAD_CHARS = 6000   # ~ first 2 pages of whitespace-stripped text
TAIL_CHARS = 3000
EVIDENCE_RADIUS = 80

# Canonical ANA name -> matching rules.
#   "norm":         substrings searched in whitespace-stripped, lowercased text
#   "norm_reject":  a norm hit is discarded if the text immediately before it
#                   ends with one of these (kills e.g. "not APPLIC-able
#                   adjudication" for Able Adjudication)
#   "norm_context": if present, a norm hit only counts when the ~60 chars
#                   before it contain one of these referral phrases (kills
#                   contract-form mentions like "Master Builders subcontract")
#   "regex":        compiled patterns searched in the ORIGINAL text (for
#                   acronyms that need word boundaries)
REFERRAL_CONTEXT = (
    "nominatingauthority", "nominatedby", "nominatedme", "lodgedwith",
    "applicationto", "applicationwasmadeto", "referredby", "(ana)", "ana:",
)
ANA_RULES = {
    "Adjudicate Today": {
        "norm": ["adjudicatetoday"],
        "regex": [],
    },
    "Institute of Arbitrators & Mediators Australia (IAMA)": {
        "norm": ["instituteofarbitrators"],
        "regex": [re.compile(r"\bIAMA\b")],
    },
    "RICS Dispute Resolution Service": {
        "norm": [
            "ricsdisputeresolution",
            "ricsaustralia",
            "ricsoceania",
            "royalinstitutionofcharteredsurveyors",
        ],
        "regex": [re.compile(r"\bRICS\b")],
    },
    "ABC Dispute Resolution Service": {
        "norm": ["abcdisputeresolution"],
        "regex": [],
    },
    "Able Adjudication": {
        # Brand often appears lowercase in letterheads. The normalised form
        # collides with "(not) applicable adjudication", hence norm_reject.
        "norm": ["ableadjudication"],
        "norm_reject": ["applic"],
        "regex": [re.compile(r"\bAble\s+Adjudication\b|ABLE\s+ADJUDICATION")],
    },
    "Australian Solutions Centre": {
        "norm": ["australiansolutionscentre", "australiansolutionscenter"],
        "regex": [],
    },
    "Australian Institute of Quantity Surveyors (AIQS)": {
        # Bare "aiqs" false-positives on post-nominals ("FAIQS"), so anchor
        # the normalised forms and use a word boundary on the original.
        "norm": ["aiqsana", "aiqsadjudication", "aiqs–adjudication",
                 "aiqs-adjudication", "instituteofquantitysurveyors"],
        "regex": [re.compile(r"\bAIQS\b")],
    },
    "LEADR": {
        # "leadr" as a bare normalised substring false-positives on
        # "available ADR" -> "availableadr"; use word boundaries on original.
        # PDF extraction sometimes splits the acronym ("LEAD R"), so allow
        # whitespace between the (uppercase) letters.
        "norm": [],
        "regex": [
            re.compile(r"\bL\s*E\s*A\s*D\s*R\b"),
            re.compile(r"\bLEADR\b", re.IGNORECASE),
        ],
    },
    "Queensland Law Society": {
        # "queensiand" = common OCR mangling of "Queensland"
        "norm": ["queenslandlawsociety", "queensiandlawsociety", "qldlawsociety"],
        "regex": [],
    },
    "Master Builders Queensland": {
        # "Master Builders" overwhelmingly refers to their standard-form
        # contracts, so only count it in a referral context.
        "norm": ["masterbuilders"],
        "norm_context": REFERRAL_CONTEXT,
        "regex": [],
    },
    "Nominator": {
        # Company name; word boundary excludes "denominator". Capitalised
        # forms only, so the generic noun mid-sentence doesn't fire.
        "norm": ["nominatorptyltd", "nominator(qld)", "nominatorqld"],
        "regex": [re.compile(r"\bNOMINATOR\b|\bNominator\b")],
    },
}


def normalise(text):
    """Strip all whitespace + lowercase. Returns (norm_text, offsets) where
    offsets[i] is the index in the ORIGINAL text of norm_text[i]."""
    chars = []
    offsets = []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch.lower())
            offsets.append(i)
    return "".join(chars), offsets


def find_matches_in_region(norm, offsets, text, nws_prefix, start, end):
    """Return {ana: earliest_norm_pos} for matches whose normalised position
    falls within [start, end)."""
    hits = {}
    region = norm[start:end]
    for ana, rules in ANA_RULES.items():
        best = None
        for needle in rules["norm"]:
            p = region.find(needle)
            while p != -1:
                pos = start + p
                rejected = False
                for bad in rules.get("norm_reject", ()):
                    if norm[max(0, pos - len(bad)):pos] == bad:
                        rejected = True
                        break
                if not rejected and "norm_context" in rules:
                    before = norm[max(0, pos - 60):pos]
                    if not any(c in before for c in rules["norm_context"]):
                        rejected = True
                if not rejected:
                    if best is None or pos < best:
                        best = pos
                    break  # earliest surviving occurrence in region is enough
                p = region.find(needle, p + 1)
        for rx in rules["regex"]:
            for m in rx.finditer(text):
                npos = nws_prefix[m.start()]  # normalised position of match
                if npos < start or npos >= end:
                    continue
                if best is None or npos < best:
                    best = npos
                break  # finditer is ordered; first in-region hit is earliest
        if best is not None:
            hits[ana] = best
    return hits


def evidence_snippet(text, orig_pos):
    lo = max(0, orig_pos - EVIDENCE_RADIUS)
    hi = min(len(text), orig_pos + EVIDENCE_RADIUS)
    snip = re.sub(r"\s+", " ", text[lo:hi]).strip()
    return ("..." if lo > 0 else "") + snip + ("..." if hi < len(text) else "")


def code_document(text):
    """Return (ana, ana_source, ana_evidence) for one document's text."""
    if not text:
        return None, "unknown", None
    norm, offsets = normalise(text)
    if not norm:
        return None, "unknown", None

    # nws_prefix[i] = number of non-whitespace chars strictly before text[i]
    # (= normalised position of the char at original index i).
    nws_prefix = [0] * (len(text) + 1)
    c = 0
    for i, ch in enumerate(text):
        nws_prefix[i] = c
        if not ch.isspace():
            c += 1
    nws_prefix[len(text)] = c

    regions = [
        (0, min(HEAD_CHARS, len(norm)), "keyword"),                      # head
        (max(0, len(norm) - TAIL_CHARS), len(norm), "keyword"),          # tail
        (0, len(norm), "keyword_lowconf"),                               # full
    ]
    for start, end, source in regions:
        if start >= end:
            continue
        hits = find_matches_in_region(norm, offsets, text, nws_prefix, start, end)
        if hits:
            ana, pos = min(hits.items(), key=lambda kv: kv[1])
            return ana, source, evidence_snippet(text, offsets[pos])
    return None, "unknown", None


def ensure_columns(con):
    existing = {r[1] for r in con.execute("PRAGMA table_info(decision_details)")}
    for col in ("ana", "ana_source", "ana_evidence"):
        if col not in existing:
            try:
                con.execute(f"ALTER TABLE decision_details ADD COLUMN {col} TEXT")
                print(f"Added column decision_details.{col}")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise  # concurrent writer may have added it; that's fine


def main():
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout = 60000")
    ensure_columns(con)
    con.commit()

    rows = con.execute(
        """
        SELECT dd.ejs_id, df.full_text
        FROM decision_details dd
        LEFT JOIN docs_fresh df ON dd.ejs_id = df.ejs_id
        WHERE dd.decision_date < '2015-01-01'
        """
    ).fetchall()
    print(f"Scanning {len(rows)} pre-2015 decisions...")

    results = []
    for row in rows:
        ana, source, evidence = code_document(row["full_text"])
        results.append(
            {"ejs_id": row["ejs_id"], "ana": ana, "ana_source": source,
             "ana_evidence": evidence}
        )

    # Write back in short batched transactions (concurrent WAL writer safe).
    BATCH = 500
    for i in range(0, len(results), BATCH):
        batch = results[i:i + BATCH]
        con.execute("BEGIN IMMEDIATE")
        con.executemany(
            "UPDATE decision_details SET ana=?, ana_source=?, ana_evidence=? "
            "WHERE ejs_id=?",
            [(r["ana"], r["ana_source"], r["ana_evidence"], r["ejs_id"])
             for r in batch],
        )
        con.commit()
    con.close()

    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    with open(EXPORT_PATH, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Wrote {len(results)} rows to {EXPORT_PATH}")

    # ----- report -----
    dist = {}
    for r in results:
        key = r["ana"] or "(unknown)"
        dist[key] = dist.get(key, 0) + 1
    lowconf = sum(1 for r in results if r["ana_source"] == "keyword_lowconf")
    coded = sum(1 for r in results if r["ana"])

    print("\nDistribution:")
    for name, n in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {name}")
    print(f"\nCoded: {coded} / {len(results)}  "
          f"(of which low-confidence full-text matches: {lowconf})")

    print("\n15 random samples with evidence:")
    random.seed(42)
    for r in random.sample([r for r in results if r["ana"]],
                           min(15, coded)):
        print(f"- {r['ejs_id']} [{r['ana_source']}] {r['ana']}\n"
              f"    {r['ana_evidence']}")


if __name__ == "__main__":
    main()
