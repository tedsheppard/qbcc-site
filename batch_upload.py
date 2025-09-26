import os, json, sqlite3
from openai import OpenAI

# init OpenAI client (make sure OPENAI_API_KEY is in Render env vars)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = "/tmp/qbcc.db"
con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row

# new safe table for results
con.execute("""
CREATE TABLE IF NOT EXISTS ai_adjudicator_extract (
    ejs_id TEXT PRIMARY KEY,
    adjudicator_name TEXT,
    claimed_amount REAL,
    adjudicated_amount REAL,
    no_jurisdiction_findings INTEGER,
    avg_fee_claimant_proportion TEXT,
    avg_fee_respondent_proportion TEXT,
    decision_date TEXT,
    raw_json TEXT
)
""")

prompt_template = """
You are extracting structured data from Queensland adjudication decisions under the BIF Act.

Decision text is below. Extract the following as JSON:

- adjudicator_name (string, full name)
- claimed_amount (AUD number, no $ sign)
- adjudicated_amount (AUD number, no $ sign)
- no_jurisdiction_findings (integer: 1 if a jurisdictional objection was upheld, else 0)
- avg_fee_claimant_proportion (percentage if given, else blank)
- avg_fee_respondent_proportion (percentage if given, else blank)
- decision_date (as YYYY-MM-DD if possible)
- ejs_id (must be the same ID we provide you)

If uncertain, leave the field blank. Return ONLY valid JSON.
"""

def main():
    # Grab first 100 decisions by EJS ID (sorted so predictable)
    rows = con.execute("""
        SELECT m.ejs_id, f.full_text
        FROM docs_meta m
        JOIN docs_fresh f ON m.ejs_id = f.ejs_id
        ORDER BY m.ejs_id
        LIMIT 100
    """).fetchall()

    for row in rows:
        ejs_id = row["ejs_id"]
        text = row["full_text"][:50000]  # trim if too long

        print(f"➡️ Extracting {ejs_id}...")

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},   # <-- forces JSON
            messages=[
                {"role": "system", "content": "You are a careful legal data extraction assistant."},
                {"role": "user", "content": prompt_template + f"\n\nEJS ID: {ejs_id}\n\n---\n\n" + text}
            ]
        )

        content = resp.choices[0].message.content.strip()

        try:
            data = json.loads(content)
        except Exception as e:
            print(f"❌ JSON error for {ejs_id}: {e}")
            data = {"ejs_id": ejs_id, "error": str(e)}

        # fallback if missing ID
        if not data.get("ejs_id"):
            data["ejs_id"] = ejs_id

        con.execute("""
            INSERT OR REPLACE INTO ai_adjudicator_extract
            (ejs_id, adjudicator_name, claimed_amount, adjudicated_amount,
             no_jurisdiction_findings, avg_fee_claimant_proportion, avg_fee_respondent_proportion,
             decision_date, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("ejs_id"),
            data.get("adjudicator_name"),
            data.get("claimed_amount"),
            data.get("adjudicated_amount"),
            data.get("no_jurisdiction_findings"),
            data.get("avg_fee_claimant_proportion"),
            data.get("avg_fee_respondent_proportion"),
            data.get("decision_date"),
            json.dumps(data)
        ))

        con.commit()

    print("✅ Finished extracting 100 decisions → ai_adjudicator_extract table")

if __name__ == "__main__":
    main()
