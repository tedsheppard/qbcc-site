import os, sqlite3, time
from meilisearch import Client

DB_PATH    = "/tmp/qbcc.db"
MEILI_HOST = os.environ["MEILI_URL"]
MEILI_KEY  = os.environ["MEILI_MASTER_KEY"]
INDEX_UID  = "decisions"
BATCH_SIZE = 1000

SQL = """
SELECT
  ejs_id,
  ejs_id                    AS id,        -- frontend expects 'id'
  reference,
  pdf_path,
  claimant,
  respondent,
  adjudicator,
  decision_date             AS date,      -- frontend expects 'date'
  application_no,
  payment_claim_amount,
  payment_schedule_amount,
  adjudicated_amount,
  jurisdiction_issue,
  frivolous_or_vexatious,
  act,
  full_text                 AS content    -- frontend expects 'content'
FROM search_index
ORDER BY ejs_id
"""

def get_task_uid(task):
    """Handle TaskInfo across client versions safely."""
    # Try common attribute names
    for attr in ("task_uid", "uid"):
        if hasattr(task, attr):
            return getattr(task, attr)
    # Try model_dump (Pydantic v2)
    try:
        d = task.model_dump()
        return d.get("task_uid") or d.get("uid") or d.get("taskUid")
    except Exception:
        pass
    # Last resort: try dict-style (older clients)
    try:
        return task["taskUid"]
    except Exception:
        raise RuntimeError(f"Could not read task uid from {type(task).__name__}: {task!r}")

def get_status(taskinfo):
    """Return status string safely."""
    if hasattr(taskinfo, "status"):
        return taskinfo.status
    try:
        return taskinfo.model_dump().get("status")
    except Exception:
        pass
    try:
        return taskinfo["status"]
    except Exception:
        raise RuntimeError(f"Could not read status from {type(taskinfo).__name__}: {taskinfo!r}")

# --- Connect SQLite ---
con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row

cnt = con.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
print(f"Found {cnt} rows in search_index.")

# --- Meili client/index ---
client = Client(MEILI_HOST, MEILI_KEY)
index = client.index(INDEX_UID)

# Ensure primary key is 'ejs_id' (safe if already set)
try:
    info = index.get_raw_info()  # returns dict with 'primaryKey'
    if info.get("primaryKey") != "ejs_id":
        # If the index exists with a different PK, this will fail;
        # but in your deployment it’s already 'ejs_id'.
        index.update(primary_key="ejs_id")
except Exception:
    client.create_index(INDEX_UID, {"primaryKey": "ejs_id"})

# --- Stream in batches ---
cur = con.execute(SQL)
pushed = 0
while True:
    rows = cur.fetchmany(BATCH_SIZE)
    if not rows:
        break
    docs = [dict(r) for r in rows]

    task = index.add_documents(docs)
    task_uid = get_task_uid(task)

    # Poll
    while True:
        st = client.get_task(task_uid)
        status = get_status(st)
        if status in ("succeeded", "failed"):
            if status == "failed":
                print("Batch failed:", st)
                raise SystemExit(1)
            break
        time.sleep(0.5)

    pushed += len(docs)
    print(f"Pushed {pushed}/{cnt}")

print("Reindex done.")
