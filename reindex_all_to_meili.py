import os, sqlite3, time, math
from meilisearch import Client

DB_PATH    = "/tmp/qbcc.db"
MEILI_HOST = os.environ["MEILI_URL"]
MEILI_KEY  = os.environ["MEILI_MASTER_KEY"]
INDEX_UID  = "decisions"

# TUNE THESE LOW to avoid memory spikes on the web dyno
BATCH_SIZE = 50            # keep small
SLEEP_BETWEEN_BATCHES = 0.5  # seconds

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

def task_uid_of(task):
    for a in ("task_uid", "uid"):
        if hasattr(task, a):
            return getattr(task, a)
    try:
        d = task.model_dump()
        return d.get("task_uid") or d.get("uid") or d.get("taskUid")
    except Exception:
        pass
    try:
        return task["taskUid"]
    except Exception:
        raise RuntimeError(f"Cannot read task uid from {type(task).__name__}: {task!r}")

def task_status_of(taskinfo):
    if hasattr(taskinfo, "status"):
        return taskinfo.status
    try:
        return taskinfo.model_dump().get("status")
    except Exception:
        pass
    try:
        return taskinfo["status"]
    except Exception:
        raise RuntimeError(f"Cannot read status from {type(taskinfo).__name__}: {taskinfo!r}")

con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row

cnt = con.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
print(f"Found {cnt} rows in search_index.")

client = Client(MEILI_HOST, MEILI_KEY)
index = client.index(INDEX_UID)

# ensure index exists with correct primary key
try:
    info = index.get_raw_info()
    if info.get("primaryKey") != "ejs_id":
        index.update(primary_key="ejs_id")
except Exception:
    client.create_index(INDEX_UID, {"primaryKey": "ejs_id"})

cur = con.execute(SQL)
pushed = 0
batch_no = 0

while True:
    rows = cur.fetchmany(BATCH_SIZE)
    if not rows:
        break
    docs = [dict(r) for r in rows]
    batch_no += 1

    task = index.add_documents(docs)
    tuid = task_uid_of(task)

    while True:
        st = client.get_task(tuid)
        status = task_status_of(st)
        if status in ("succeeded", "failed"):
            if status == "failed":
                print("Batch failed:", st)
                raise SystemExit(1)
            break
        time.sleep(0.5)

    pushed += len(docs)
    print(f"Batch {batch_no}: pushed {pushed}/{cnt}")
    time.sleep(SLEEP_BETWEEN_BATCHES)

print("Reindex done.")
