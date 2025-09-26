import os, json, sqlite3
from openai import OpenAI

# init OpenAI client (needs OPENAI_API_KEY in Render env vars)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = "/tmp/qbcc.db"
con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row

# create NEW clean table
con.execute("""
CREATE TABLE IF NOT EXISTS ai_adjudicator_extract_v2 (
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
    raw_json TEXT
)
""")

prompt_template = """
You are extracting structured data from Queensland adjudication decisions under the BIF Act.

Rules:
- All monetary figures must be GST inclusive. 
  - If only GST exclusive amounts are given, multiply by 1.1.
  - If unclear, assume GST inclusive.
  - If adjudicated amount is about 10% higher than the claimed amount, adjust for GST discrepancy.
- Percentages must be numeric only (no "%" sign). Always return 0–100.
- fee_respondent_proportion = 100 - fee_claimant_proportion.
- Claimant and respondent names must be in proper Title Case (not ALL CAPS).

Extract the following fields as JSON:
- adjudicator_name (string, full name)
- claimant_name (string, Title Case)
- respondent_name (string, Title Case)
- claimed_amount (AUD, GST inclusive)
- payment_schedule_amount (AUD, GST inclusive if given)
- adjudicated_amount (AUD, GST inclusive)
- jurisdiction_upheld (1 if jurisdictional objection upheld, else 0)
- fee_claimant_proportion (numeric, 0–100)
- fee_respondent_proportion (numeric, 0–100)
- decision_date (YYYY-MM-DD)
- ejs_id (same ID we provide)
"""

def main(offset=0, limit=100):
    rows = con.execute(f"""
        SELECT m.ejs_id, f.full_text
        FROM docs_meta m
        JOIN docs_fresh f ON m.ejs_id = f.ejs_id
        ORDER BY m.ejs_id
        LIMIT {limit} OFFSET {offset}
    """).fetchall()

    for row in rows:
        ejs_id = row["ejs_id"]
        text = row["full_text"][:50000]  # trim if huge

        print(f"➡️ Extracting {ejs_id}...")

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

        try:
            data = json.loads(content)
        except Exception as e:
            print(f"❌ JSON error for {ejs_id}: {e}")
            data = {"ejs_id": ejs_id, "error": str(e)}

        if not data.get("ejs_id"):
            data["ejs_id"] = ejs_id

        con.execute("""
            INSERT OR REPLACE INTO ai_adjudicator_extract_v2
            (ejs_id, adjudicator_name, claimant_name, respondent_name,
             claimed_amount, payment_schedule_amount, adjudicated_amount,
             jurisdiction_upheld, fee_claimant_proportion, fee_respondent_proportion,
             decision_date, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(data)
        ))

        con.commit()

    print(f"✅ Finished extracting {len(rows)} decisions → ai_adjudicator_extract_v2")

if __name__ == "__main__":
    main()
