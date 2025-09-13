import sqlite3
import requests
import os

# ---------------- CONFIG ----------------
MEILI_URL = "https://meilisearch-v1-9-3xaz.onrender.com"
MEILI_KEY = os.getenv("MEILI_MASTER_KEY")  # must be set in Render env vars
SQLITE_PATH = "/tmp/qbcc.db"
PAGE_SIZE = 200
# ----------------------------------------

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

    # Count before
    cur.execute("SELECT COUNT(*) FROM docs_fresh")
    before = cur.fetchone()[0]
    print(f"SQLite docs before: {before}")

    # Get SQLite IDs
    sqlite_ids = get_sqlite_ids(con)

    # Get total Meili count
    stats = requests.get(
        f"{MEILI_URL}/indexes/decisions/stats",
        headers={"Authorization": f"Bearer {MEILI_KEY}"}
    ).json()
    meili_total = stats["numberOfDocuments"]
    print(f"Meili docs: {meili_total}")

    # Find missing IDs
    missing = []
    offset = 0
    while True:
        resp = requests.get(
            f"{MEILI_URL}/indexes/decisions/documents?limit={PAGE_SIZE}&offset={offset}&fields=ejs_id",
            headers={"Authorization": f"Bearer {MEILI_KEY}"}
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        for d in page:
            # Handle both dict and string formats
            if isinstance(d, dict):
                ejs_id = d.get("ejs_id")
            else:
                ejs_id = d
            if ejs_id and ejs_id not in sqlite_ids:
                missing.append(ejs_id)
        offset += PAGE_SIZE
        if offset % 1000 == 0:
            print(f"Checked {offset} docs...")

    print(f"Missing docs to add: {len(missing)}")
    if missing:
        print("Missing IDs:", missing)

    # Insert missing only
    for i, ejs_id in enumerate(missing, start=1):
        doc = fetch_doc(ejs_id)
        insert_doc(cur, doc)
        if i % 5 == 0:
            print(f"Inserted {i}/{len(missing)}...")
    con.commit()

    # Count after
    cur.execute("SELECT COUNT(*) FROM docs_fresh")
    after = cur.fetchone()[0]
    print(f"SQLite docs after: {after}")

    con.close()
    print("✅ Sync complete.")

if __name__ == "__main__":
    main()
