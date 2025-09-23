import os
import sqlite3
import time
from meilisearch import Client
from meilisearch.errors import MeilisearchCommunicationError

# ---- CONFIG ----
DB_PATH = "/tmp/qbcc.db"
MEILI_HOST = "https://meilisearch-v1-9-3xaz.onrender.com"
MEILI_KEY = os.getenv("MEILI_MASTER_KEY")  # make sure this is set in qbcc-site-1 environment
EJS_IDS = ("EJS07378","EJS07379","EJS07380","EJS07381","EJS07382","EJS07383","EJS07384")

# ---- CONNECT TO SQLITE ----
con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row

placeholders = ",".join("?" for _ in EJS_IDS)

query = f"""
SELECT m.ejs_id AS id,
       m.claimant,
       m.respondent,
       m.adjudicator,
       m.decision_date_norm AS decision_date,
       f.full_text
FROM docs_meta m
JOIN docs_fresh f ON m.ejs_id = f.ejs_id
WHERE m.ejs_id IN ({placeholders});
"""

rows = con.execute(query, EJS_IDS).fetchall()
docs = [dict(r) for r in rows]
print(f"Fetched {len(docs)} docs from SQLite.")

if not docs:
    print("No documents found. Exiting.")
    exit(0)

# ---- CONNECT + PUSH TO MEILI ----
try:
    client = Client(MEILI_HOST, MEILI_KEY)
    index = client.index("decisions")

    task = index.add_documents(docs)
    task_uid = task.task_uid
    print(f"Pushed {len(docs)} docs. Task UID: {task_uid}")

    # ---- POLL STATUS ----
    while True:
        status = client.get_task(task_uid)
        if status.status in ("succeeded", "failed"):
            print("Final status:", status.status)
            if status.status == "failed":
                print(status)
            break
        print("Task still processing... waiting 2s")
        time.sleep(2)

except MeilisearchCommunicationError as e:
    print("❌ Could not connect to MeiliSearch. Check MEILI_HOST and MEILI_KEY.")
    print(str(e))
    exit(1)
