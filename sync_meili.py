import sqlite3
import requests
import time
import os
import sys
from datetime import datetime

# --- INSTRUCTIONS ---
# 1. Ensure this script is in your project on the Render server.
# 2. Restart your Render service to ensure it has the latest qbcc.db from GCS.
# 3. Open the shell for your service and run: python3 sync_meili.py
# ---

# --- Configuration ---
# The script will look for the database in the /tmp directory on the server.
DB_PATH = "/tmp/qbcc.db"
MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:7700")
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "")
MEILI_INDEX = "decisions"
BATCH_SIZE = 500 # How many documents to send in each batch

def sync_database_to_meili():
    """
    Reads all documents from the SQLite database and syncs them with MeiliSearch.
    This is a full sync: it adds new documents and updates existing ones.
    """
    # --- Step 1: Safety Checks ---
    print("--- Starting MeiliSearch Sync ---")
    if not os.path.exists(DB_PATH):
        print(f"FATAL: Database file not found at '{DB_PATH}'.")
        print("Please ensure your Render service has been restarted to download the database from GCS.")
        sys.exit(1)
    
    if not MEILI_URL or not MEILI_KEY:
        print("FATAL: MEILI_URL or MEILI_MASTER_KEY environment variables are not set on your Render service.")
        sys.exit(1)

    # --- Step 2: Fetch Data from SQLite ---
    print(f"-> Step 1: Connecting to database at '{DB_PATH}'...")
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        
        print("-> Step 2: Fetching all decision data from the database...")
        # This query joins the tables to get all necessary fields.
        docs_to_sync = con.execute("""
            SELECT 
                d.ejs_id, d.reference, d.pdf_path, d.full_text,
                m.claimant, m.respondent, m.adjudicator, 
                COALESCE(a.decision_date, m.decision_date_norm) as decision_date, -- Prioritize the clean date
                m.act
            FROM docs_fresh d
            LEFT JOIN docs_meta m ON d.ejs_id = m.ejs_id
            LEFT JOIN ai_adjudicator_extract_v4 a ON d.ejs_id = a.ejs_id
        """).fetchall()
        con.close()
        print(f"   ... Found {len(docs_to_sync)} documents to sync.")
    except Exception as e:
        print(f"FATAL: Failed to read from the database. Error: {e}")
        sys.exit(1)

    # --- Step 3: Prepare and Upload Data in Batches ---
    meili_docs = []
    for doc in docs_to_sync:
        try:
            sortable_date_unix = 0
            # Convert date to a UNIX timestamp for MeiliSearch sorting
            if doc["decision_date"]:
                dt_object = None
                try:
                    # Try parsing YYYY-MM-DD first
                    dt_object = datetime.strptime(doc["decision_date"], "%Y-%m-%d")
                except (ValueError, TypeError):
                    try:
                       # Fallback to DD/MM/YYYY
                       dt_object = datetime.strptime(doc["decision_date"], "%d/%m/%Y")
                    except (ValueError, TypeError):
                       pass # Date format is invalid, will default to 0
                if dt_object:
                    sortable_date_unix = int(time.mktime(dt_object.timetuple()))

            meili_docs.append({
                "id": doc["ejs_id"],
                "reference": doc["reference"],
                "pdf_path": doc["pdf_path"],
                "claimant": doc["claimant"],
                "respondent": doc["respondent"],
                "adjudicator": doc["adjudicator"],
                "date": doc["decision_date"],
                "sortable_date": sortable_date_unix,
                "act": doc["act"],
                "content": doc["full_text"] or ""
            })
        except Exception as e:
            print(f"Warning: Skipping document {doc['ejs_id']} due to a data error: {e}")

    headers = {"Authorization": f"Bearer {MEILI_KEY}"}
    
    print(f"-> Step 3: Uploading {len(meili_docs)} documents to MeiliSearch in batches...")
    for i in range(0, len(meili_docs), BATCH_SIZE):
        batch = meili_docs[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"   ... Sending batch {batch_num} of {((len(meili_docs) -1) // BATCH_SIZE) + 1} ({len(batch)} documents)...")
        try:
            res = requests.post(f"{MEILI_URL}/indexes/{MEILI_INDEX}/documents", headers=headers, json=batch, timeout=30)
            res.raise_for_status()
            print(f"     ... Batch {batch_num} accepted by MeiliSearch.")
            time.sleep(1) # Brief pause to not overwhelm the server
        except requests.exceptions.RequestException as e:
            print(f"FATAL: Error uploading batch {batch_num}. Aborting.")
            print(f"Error details: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response from server: {e.response.text}")
            sys.exit(1)
            
    print("\n--- Synchronization Complete ---")
    print("All new and updated decisions have been sent to MeiliSearch.")

if __name__ == "__main__":
    sync_database_to_meili()

