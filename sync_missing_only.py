import sqlite3
import requests
import os

# ---------------- CONFIG ----------------
MEILI_URL = "https://meilisearch-v1-9-3xaz.onrender.com"
MEILI_KEY = os.getenv("MEILI_MASTER_KEY")  # make sure set in Render env vars
SQLITE_PATH = "/tmp/qbcc.db"
BATCH_SIZE = 500
# ----------------------------------------

def get_meili_ids():
    """Fetch just IDs from Meili in batches."""
    ids = []
    offset = 0
    while True:
        url = f"{MEILI_URL}/indexes/decisions/documents?limit={BATCH_SIZE}&offset={offset}&fields=ejs_id"
        resp = requests.get(url, headers={"Authorization": f"Bearer {MEILI_KEY}"})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        ids.extend([d["ejs_id"] for d in batch if "ejs_id" in d])
        offset += BATCH_SIZE
    return set(ids)

def get_sqlite_ids(con):
    cur = con.cursor()
    cur.execute("SELECT ejs_id FROM docs_fresh")
    return {row[0] for row in cur.fetchall()}

def fetch_doc(ejs_id):
    """Fetch a single document by ID from Meili."""
    url = f"{MEILI_URL}/indexes/decisions/documents/{ejs_id}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {MEILI_KEY}"})
    resp.raise_for_status()
    return resp.json()

def insert_doc(cur, doc):
    cur.execute("""
        INSERT OR IGNORE INTO docs_fresh (ejs_id, reference, pdf_path, full_text)
        VALUES (?, ?, ?, ?)
    """, (
        doc.get("ejs_id"),
        doc.get("reference"),
        doc.get("pdf_path"),
        doc.get("full_text")
    ))
    cur.execute("""
        INSERT OR IGNORE INTO docs_meta (ejs_id, claimant, respondent, adjudicator,
                                         decision_date, application_no,
                                         payment_claim_amount, payment_schedule_amount,
                                         adjudicated_amount, jurisdiction_issue,
                                         frivolous_or_vexatious, act)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        doc.get("ejs_id"),
        doc.get("claimant"),
        doc.get("respondent"),
        doc.get("adjudicator"),
        doc.get("decision_date"),
        doc.get("application_no"),
        doc.get("payment_claim_amount"),
        doc.get("payment_schedule_amount"),
        doc.get("adjudicated_amount"),
        doc.get("jurisdiction_issue"),
        doc.get("frivolous_or_vexatious"),
        doc.get("act")
    ))

def main():
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()

    # Counts before
    cur.execute("SELECT COUNT(*) FROM docs_fresh")
    before = cur.fetchone()[0]
    print(f"SQLite docs before: {before}")

    # Compare
    meili_ids = get_meili_ids()
    sqlite_ids = get_sqlite_ids(con)
    missing = meili_ids - sqlite_ids
    print(f"Meili docs: {len(meili_ids)}")
    print(f"SQLite docs: {len(sqlite_ids)}")
    print(f"Missing docs to add: {len(missing)}")

    # Insert missing only
    for i, ejs_id in enumerate(missing, start=1):
        doc = fetch_doc(ejs_id)
        insert_doc(cur, doc)
        if i % 10 == 0:
            print(f"Inserted {i}/{len(missing)}...")
    con.commit()

    # Counts after
    cur.execute("SELECT COUNT(*) FROM docs_fresh")
    after = cur.fetchone()[0]
    print(f"SQLite docs after: {after}")

    con.close()
    print("✅ Sync complete.")

if __name__ == "__main__":
    main()
