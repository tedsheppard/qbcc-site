#!/usr/bin/env python3
"""
Check which decisions are missing from the database.

Usage:
    python check_missing_decisions.py --latest EJS07500

This will show you which decisions between your most recent (EJS07405)
and the latest from QBCC (EJS07500) are missing.
"""

import sqlite3
import argparse
from google.cloud import storage
import os
import sys

DB_PATH = "/tmp/qbcc.db"

def get_gcs_client():
    """Create GCS client from environment variable credentials"""
    gcs_credentials_json = os.getenv("GCS_CREDENTIALS_JSON")

    if not gcs_credentials_json:
        print("FATAL: GCS_CREDENTIALS_JSON environment variable not found.")
        return None

    temp_credentials_path = "/tmp/gcs_credentials.json"

    try:
        with open(temp_credentials_path, "w") as f:
            f.write(gcs_credentials_json)

        client = storage.Client.from_service_account_json(temp_credentials_path)
        print("✅ GCS client created successfully")
        return client

    except Exception as e:
        print(f"❌ Failed to create GCS client. Error: {e}")
        return None

def download_database(gcs_client, bucket_name="sopal-bucket"):
    """Download database from GCS"""
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob("qbcc.db")
        blob.download_to_filename(DB_PATH)
        print(f"✅ Downloaded database from GCS")
        return True
    except Exception as e:
        print(f"❌ Failed to download database: {e}")
        return False

def get_most_recent_decision(con):
    """Get the most recent decision in the database"""
    row = con.execute("SELECT ejs_id, decision_date FROM decision_details ORDER BY decision_date DESC LIMIT 1").fetchone()
    if row:
        return row[0], row[1]
    return None, None

def extract_ejs_number(ejs_id):
    """Extract numeric part from EJS ID (e.g., 'EJS07405' -> 7405)"""
    if ejs_id and ejs_id.startswith("EJS"):
        return int(ejs_id[3:])
    return None

def check_missing_range(con, start_ejs, end_ejs):
    """Check which decisions are missing in a range"""
    start_num = extract_ejs_number(start_ejs)
    end_num = extract_ejs_number(end_ejs)

    if not start_num or not end_num:
        print("❌ Invalid EJS IDs provided")
        return []

    missing = []
    existing = []

    for num in range(start_num + 1, end_num + 1):
        ejs_id = f"EJS{num:05d}"
        row = con.execute("SELECT ejs_id FROM docs_fresh WHERE ejs_id = ?", (ejs_id,)).fetchone()
        if row:
            existing.append(ejs_id)
        else:
            missing.append(ejs_id)

    return missing, existing

def main():
    parser = argparse.ArgumentParser(description="Check for missing decisions")
    parser.add_argument("--latest", type=str, required=True, help="Latest EJS ID from QBCC (e.g., EJS07500)")
    parser.add_argument("--download_db", type=bool, default=True, help="Download database from GCS first")
    args = parser.parse_args()

    # Check if database exists locally
    if args.download_db or not os.path.exists(DB_PATH):
        if not os.getenv("GCS_CREDENTIALS_JSON"):
            print("❌ GCS_CREDENTIALS_JSON environment variable not set")
            sys.exit(1)

        gcs_client = get_gcs_client()
        if not gcs_client:
            sys.exit(1)

        if not download_database(gcs_client):
            sys.exit(1)

    # Connect to database
    con = sqlite3.connect(DB_PATH)

    # Get most recent decision
    most_recent_ejs, most_recent_date = get_most_recent_decision(con)
    print(f"\n📊 Current Database Status:")
    print(f"  Most recent decision: {most_recent_ejs} (dated {most_recent_date})")

    total_count = con.execute("SELECT COUNT(*) FROM docs_fresh").fetchone()[0]
    print(f"  Total decisions in database: {total_count}")

    # Check missing range
    print(f"\n🔍 Checking for missing decisions between {most_recent_ejs} and {args.latest}...")
    missing, existing = check_missing_range(con, most_recent_ejs, args.latest)

    print(f"\n📈 Results:")
    print(f"  Missing decisions: {len(missing)}")
    print(f"  Existing decisions in range: {len(existing)}")

    if missing:
        print(f"\n❌ Missing decisions:")
        for ejs_id in missing:
            print(f"  - {ejs_id}")
        print(f"\n💡 You need to download {len(missing)} PDF files from QBCC")
    else:
        print(f"\n✅ No missing decisions! Your database is up to date.")

    con.close()

if __name__ == "__main__":
    main()
