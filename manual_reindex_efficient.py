import os
import sqlite3
import requests
import time
from datetime import datetime
import json

# --- CONFIGURATION (Reads from your server's environment variables) ---
DB_PATH = "/tmp/qbcc.db"
MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:7700")
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "")
MEILI_INDEX = "decisions"

def run_efficient_reindex():
    """
    Connects to the local SQLite DB and pushes documents to Meilisearch
    in a memory-efficient way, processing them in a stream.
    """
    print("--- Starting MEMORY-EFFICIENT Meilisearch Re-index ---")

    # 1. Connect to the database
    if not os.path.exists(DB_PATH):
        print(f"FATAL: Database file not found at {DB_PATH}. Please restart the server first.")
        return

    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        print("✅ Successfully connected to the database.")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return

    # 2. Execute the query to get a cursor (an iterator, NOT all results)
    print("Fetching documents from the database in a stream...")
    try:
        # Query now includes all necessary fields from both tables
        cursor = con.execute("""
            SELECT
                d.ejs_id, d.reference, d.pdf_path, d.full_text,
                m.claimant, m.respondent, m.adjudicator, m.decision_date_norm,
                a.claimed_amount, a.adjudicated_amount,
                a.fee_claimant_proportion, a.fee_respondent_proportion
            FROM docs_fresh d
            LEFT JOIN docs_meta m ON d.ejs_id = m.ejs_id
            LEFT JOIN ai_adjudicator_extract_v4 a ON d.ejs_id = a.ejs_id
        """)
    except Exception as e:
        print(f"❌ Failed to query the database: {e}")
        con.close()
        return

    # 3. Process the stream and send to Meilisearch in batches
    batch_size = 200
    batch = []
    doc_count = 0
    headers = {"Authorization": f"Bearer {MEILI_KEY}"} if MEILI_KEY else {}

    print(f"🚀 Processing and sending documents in batches of {batch_size}...")

    # Helper function to safely convert values to float, defaulting to 0.0
    def to_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # Loop directly over the cursor to process one row at a time
    for row in cursor:
        doc_count += 1
        try:
            sortable_date = 0
            if row["decision_date_norm"]:
                sortable_date = int(time.mktime(datetime.strptime(row["decision_date_norm"], "%Y-%m-%d").timetuple()))

            # Add the processed document to the current batch (WITH ALL FIELDS)
            batch.append({
                "ejs_id": row["ejs_id"], # Correct primary key
                "reference": row["reference"],
                "pdf_path": row["pdf_path"],
                "claimant": row["claimant"],
                "respondent": row["respondent"],
                "adjudicator": row["adjudicator"],
                "date": row["decision_date_norm"],
                "sortable_date": sortable_date,
                "act": "BIF Act",
                
                # --- CORRECTED FINANCIAL FIELDS ---
                "claimed_amount": to_float(row["claimed_amount"]),
                "adjudicated_amount": to_float(row["adjudicated_amount"]),
                "fee_claimant_proportion": to_float(row["fee_claimant_proportion"]),
                "fee_respondent_proportion": to_float(row["fee_respondent_proportion"]),
                
                "content": row["full_text"] or ""
            })

            # If the batch is full, send it to Meilisearch
            if len(batch) >= batch_size:
                res = requests.post(f"{MEILI_URL}/indexes/{MEILI_INDEX}/documents", headers=headers, json=batch, timeout=30)
                res.raise_for_status()
                print(f"✅ Sent {doc_count} total documents so far...")
                batch = [] # Clear the batch to free memory

        except (ValueError, TypeError) as e:
            print(f"⚠️  Skipping document {row['ejs_id']} due to invalid date: '{row['decision_date_norm']}'")
        except requests.exceptions.RequestException as e:
            print(f"❌ FAILED to send batch after {doc_count} documents: {e}")
            print("   Aborting the process.")
            con.close()
            return

    # 4. Send the final, leftover batch if it's not empty
    if batch:
        print(f"Sending the final batch of {len(batch)} documents...")
        try:
            res = requests.post(f"{MEILI_URL}/indexes/{MEILI_INDEX}/documents", headers=headers, json=batch, timeout=30)
            res.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ FAILED to send the final batch: {e}")

    con.close()
    print(f"\n🎉 --- Re-index complete! {doc_count} documents were processed and sent. ---")

if __name__ == "__main__":
    run_efficient_reindex()