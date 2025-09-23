import os, sqlite3, math, time
from meilisearch import Client

DB_PATH = "/tmp/qbcc.db"
MEILI_HOST = os.environ["MEILI_URL"]
MEILI_KEY  = os.environ["MEILI_MASTER_KEY"]
INDEX_UID  = "decisions"
BATCH_SIZE = 1000

sql = """
SELECT
  id            AS ejs_id,      -- Meili primary key
  id            AS id,          -- frontend also expects 'id'
  reference,
  pdf_path,
  claimant,
  respondent,
  adjudicator,
  date,                         -- frontend expects 'date'
  act,
  content
FROM search_index
ORDER BY id
"""

con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row

# Count rows
cnt = con.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
print(f"Found {cnt} rows in search_index.")

client = Client(MEILI_HOST, MEILI_KEY)
index = client.index(INDEX_UID)

# Ensure primary key is correct (won't recreate the index; safe to call)
try:
    idx_info = client.get_index(INDEX_UID)
    if idx_info.get("primaryKey") != "ejs_id":
        client.index(INDEX_UID).update(primary_key="ejs_id")
except Exception:
    client.create_index(INDEX_UID, {"primaryKey": "ejs_id"})

# Stream in batches
cur = con.execute(sql)
pushed = 0
while True:
    rows = cur.fetchmany(BATCH_SIZE)
    if not rows:
        break
    docs = [dict(r) for r in rows]
    task = index.add_documents(docs)
    task_uid = getattr(task, "uid", getattr(task, "taskUid", None))
    # Simple poll
    while True:
        st = client.get_task(task_uid)
        if st["status"] in ("succeeded", "failed"):
            if st["status"] == "failed":
                print("Batch failed:", st)
                raise SystemExit(1)
            break
        time.sleep(0.5)
    pushed += len(docs)
    print(f"Pushed {pushed}/{cnt}")

print("Reindex done.")
