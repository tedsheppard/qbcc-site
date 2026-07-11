"""Attribute an ANA to registry-referred / uncoded decisions via the adjudicator.

Post-Dec-2014 Qld referrals come from the QBCC Registry, but the adjudicator's
ANA still administers the matter and is frequently not named in the decision.
Where an adjudicator appears in document-coded decisions (keyword/ai sources)
with a specific ANA, that affiliation is projected onto their registry-referred
or uncoded decisions: the ANA they used most within +/-3 years of the decision,
falling back to their most frequent overall, requiring >=2 supporting decisions.
Inferred rows get ana_source='inferred_adjudicator' so the UI can label them
approximate; the referral origin is preserved in referred_by.
"""
import json
import sqlite3
from collections import Counter

MIN_SUPPORT = 2
WINDOW_YEARS = 3

con = sqlite3.connect("qbcc.db")
con.row_factory = sqlite3.Row

cols = [r[1] for r in con.execute("PRAGMA table_info(decision_details)")]
if "referred_by" not in cols:
    con.execute("ALTER TABLE decision_details ADD COLUMN referred_by TEXT")

# Preserve the referral fact before we overwrite ana on registry rows.
con.execute("""UPDATE decision_details SET referred_by='QBCC Registry'
               WHERE ana='QBCC Registry' AND (referred_by IS NULL OR referred_by='')""")
# v2 export captured referring_body for the rows it processed.
try:
    for r in json.load(open("exports/ana_ai_v2.json")):
        rb = (r.get("referring_body") or "").strip()
        if rb:
            con.execute(
                "UPDATE decision_details SET referred_by=? WHERE ejs_id=? AND (referred_by IS NULL OR referred_by='')",
                (rb, r["ejs_id"]),
            )
except FileNotFoundError:
    pass

# Document-coded adjudicator->ANA evidence (never from inferred rows).
evidence = con.execute("""
    SELECT adjudicator_name, ana, decision_date
    FROM decision_details
    WHERE ana IS NOT NULL AND ana != 'QBCC Registry'
      AND ana_source IN ('keyword','keyword_lowconf','ai','ai_v2')
      AND adjudicator_name IS NOT NULL AND TRIM(adjudicator_name) != ''
""").fetchall()

by_adj = {}
for row in evidence:
    by_adj.setdefault(row["adjudicator_name"].strip(), []).append(
        (row["ana"], (row["decision_date"] or "")[:4])
    )

targets = con.execute("""
    SELECT ejs_id, adjudicator_name, decision_date
    FROM decision_details
    WHERE (ana='QBCC Registry' OR ana IS NULL)
      AND adjudicator_name IS NOT NULL AND TRIM(adjudicator_name) != ''
""").fetchall()

inferred, skipped_support, skipped_unknown_adj = 0, 0, 0
dist = Counter()
for t in targets:
    adj = t["adjudicator_name"].strip()
    pool = by_adj.get(adj)
    if not pool:
        skipped_unknown_adj += 1
        continue
    year = None
    d = t["decision_date"] or ""
    if len(d) >= 4 and d[:4].isdigit():
        year = int(d[:4])
    windowed = [a for a, y in pool
                if year and y.isdigit() and abs(int(y) - year) <= WINDOW_YEARS]
    counts = Counter(windowed if len(windowed) >= MIN_SUPPORT else [a for a, _ in pool])
    ana, support = counts.most_common(1)[0]
    if sum(counts.values()) < MIN_SUPPORT:
        skipped_support += 1
        continue
    basis = "window" if len(windowed) >= MIN_SUPPORT else "overall"
    con.execute(
        """UPDATE decision_details SET ana=?, ana_source='inferred_adjudicator',
           ana_role='inferred', ana_evidence=? WHERE ejs_id=?""",
        (ana, f"inferred from adjudicator '{adj}': {support} of {sum(counts.values())} "
              f"document-coded decisions with this ANA ({basis})", t["ejs_id"]),
    )
    inferred += 1
    dist[ana] += 1

con.commit()
print(f"targets: {len(targets)}, inferred: {inferred}, "
      f"no-affiliation adjudicator: {skipped_unknown_adj}, insufficient support: {skipped_support}")
for ana, n in dist.most_common():
    print(f"  {ana}: {n}")
print("remaining registry-only:",
      con.execute("SELECT COUNT(*) FROM decision_details WHERE ana='QBCC Registry'").fetchone()[0])
print("remaining uncoded:",
      con.execute("SELECT COUNT(*) FROM decision_details WHERE ana IS NULL").fetchone()[0])
con.close()
