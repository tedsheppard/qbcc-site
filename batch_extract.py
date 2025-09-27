import os, json, sqlite3, argparse, time
from openai import OpenAI
from datetime import datetime
from zoneinfo import ZoneInfo

# Init OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = "/tmp/qbcc.db"
con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row

# Create NEW v4 table
con.execute("""
CREATE TABLE IF NOT EXISTS ai_adjudicator_extract_v4 (
    ejs_id TEXT PRIMARY KEY,
    adjudicator_name TEXT,
    claimant_name TEXT,
    respondent_name TEXT,
    claimed_amount REAL,
    payment_schedule_amount REAL,
    adjudicated_amount REAL,
    jurisdiction_upheld INTEGER,
    fee_claimant_proportion REAL,
    fee_respondent_proportion REAL,
    decision_date TEXT,
    keywords TEXT,
    outcome TEXT,
    sections_referenced TEXT,
    project_type TEXT,
    contract_type TEXT,
    doc_length_pages INTEGER,
    act_category TEXT,
    raw_json TEXT
)
""")

LOG_FILE = "/tmp/extract.log"
FAIL_FILE = "/tmp/failures.log"

def log(msg):
    """Log with AEST timestamp to console and file"""
    now = datetime.now(ZoneInfo("Australia/Brisbane"))
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

prompt_template = """
You are extracting structured data from Queensland adjudication decisions.

Rules:
- All monetary figures must be GST inclusive.
  - If only GST exclusive is shown, multiply by 1.1.
  - If unclear, assume GST inclusive.
  - If adjudicated > claimed by ~10%, adjust for GST discrepancy.
- Percentages must be numeric only (0–100). No "%" signs.
  - fee_respondent_proportion = 100 - fee_claimant_proportion.
- Claimant/respondent names → Title Case, not ALL CAPS.
- Outcome → classify as: "Claimant Fully Successful", "Partly Successful", or "Unsuccessful".
- Sections Referenced → list BIF/BCIPA Act sections (e.g. "s 75, s 69"), or blank if none.
- Keywords → 10 short legal/technical tags that help summarise the decision.
- Project Type → classify if obvious (e.g. civil, residential, mining, commercial, industrial).
- Contract Type → classify as "head contract (principal / main contractor)", 
  "subcontract (main contractor / subcontractor)", "residential (owner / builder)", or "other".
- Document length → estimate number of pages (assume ~500 words per page).
- Act Category → classify as either "BCIPA 2004 (Qld)" or "BIF Act 2017 (Qld)" depending on which Act governs the decision.
- jurisdiction_upheld → 1 if jurisdictional objection upheld, else 0.

Extract as JSON with fields:
- adjudicator_name
- claimant_name
- respondent_name
- claimed_amount
- payment_schedule_amount
- adjudicated_amount
- jurisdiction_upheld
- fee_claimant_proportion
- fee_respondent_proportion
- decision_date
- keywords (array of 10 strings)
- outcome
- sections_referenced
- project_type
- contract_type
- doc_length_pages
- act_category
- ejs_id (same ID provided)
"""

def extract_and_save(ejs_id, text):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a careful legal data extraction assistant."},
                    {"role": "user", "content": prompt_template + f"\n\nEJS ID: {ejs_id}\n\n---\n\n" + text}
                ]
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            if not data.get("ejs_id"):
                data["ejs_id"] = ejs_id

            con.execute("""
                INSERT OR REPLACE INTO ai_adjudicator_extract_v4
                (ejs_id, adjudicator_name, claimant_name, respondent_name,
                 claimed_amount, payment_schedule_amount, adjudicated_amount,
                 jurisdiction_upheld, fee_claimant_proportion, fee_respondent_proportion,
                 decision_date, keywords, outcome, sections_referenced,
                 project_type, contract_type, doc_length_pages, act_category, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("ejs_id"),
                data.get("adjudicator_name"),
                data.get("claimant_name"),
                data.get("respondent_name"),
                data.get("claimed_amount"),
                data.get("payment_schedule_amount"),
                data.get("adjudicated_amount"),
                data.get("jurisdiction_upheld"),
                data.get("fee_claimant_proportion"),
                data.get("fee_respondent_proportion"),
                data.get("decision_date"),
                ", ".join(data.get("keywords", [])) if isinstance(data.get("keywords"), list) else data.get("keywords"),
                data.get("outcome"),
                data.get("sections_referenced"),
                data.get("project_type"),
                data.get("contract_type"),
                data.get("doc_length_pages"),
                data.get("act_category"),
                json.dumps(data)
            ))
            con.commit()

            log(f"✅ {ejs_id} extracted")
            return True
        except Exception as e:
            time.sleep(2)
            if attempt == 2:
                log(f"❌ {ejs_id} failed: {e}")
                with open(FAIL_FILE, "a") as f:
                    f.write(f"{ejs_id}\n")
                return False

def main(offset=0, limit=100, start_id=None, end_id=None):
    if start_id and end_id:
        rows = con.execute("""
            SELECT m.ejs_id, f.full_text
            FROM docs_meta m
            JOIN docs_fresh f ON m.ejs_id = f.ejs_id
            WHERE m.ejs_id >= ? AND m.ejs_id <= ?
            ORDER BY m.ejs_id
        """, (start_id, end_id)).fetchall()
    elif start_id:
        rows = con.execute("""
            SELECT m.ejs_id, f.full_text
            FROM docs_meta m
            JOIN docs_fresh f ON m.ejs_id = f.ejs_id
            WHERE m.ejs_id >= ?
            ORDER BY m.ejs_id
            LIMIT ?
        """, (start_id, limit)).fetchall()
    else:
        rows = con.execute(f"""
            SELECT m.ejs_id, f.full_text
            FROM docs_meta m
            JOIN docs_fresh f ON m.ejs_id = f.ejs_id
            ORDER BY m.ejs_id
            LIMIT {limit} OFFSET {offset}
        """).fetchall()

    for row in rows:
        ejs_id = row["ejs_id"]
        text = row["full_text"][:50000]
        log(f"➡️ Extracting {ejs_id}...")
        extract_and_save(ejs_id, text)

    log(f"✅ Finished extracting {len(rows)} decisions → ai_adjudicator_extract_v4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0, help="Row offset for batching")
    parser.add_argument("--limit", type=int, default=100, help="Batch size")
    parser.add_argument("--start_id", type=str, help="Resume from specific EJS ID (e.g. EJS02669)")
    parser.add_argument("--end_id", type=str, help="Stop at specific EJS ID (e.g. EJS07300)")
    args = parser.parse_args()
    main(offset=args.offset, limit=args.limit, start_id=args.start_id, end_id=args.end_id)
