#!/usr/bin/env python3
"""
Batch upload new QBCC adjudication decisions.

Usage:
    1. Download PDF decisions from QBCC and place them in a folder (e.g., ./new_decisions/)
    2. Set environment variables: OPENAI_API_KEY, GCS_CREDENTIALS_JSON, GCS_BUCKET_NAME
    3. Run: python upload_new_decisions.py --pdf_dir ./new_decisions/

This script will:
    - Extract text and metadata from each PDF using OpenAI
    - Upload PDFs to GCS bucket (pdfs/ folder)
    - Insert records into docs_fresh table
    - Insert records into decision_details table
    - Update FTS index
    - Upload updated database to GCS
    - Optionally sync to Meilisearch
"""

import os
import sys
import json
import sqlite3
import argparse
import time
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import PyPDF2
from openai import OpenAI
from google.cloud import storage

# Configuration
DB_PATH = "/tmp/qbcc.db"
LOG_FILE = "/tmp/upload_decisions.log"
FAIL_FILE = "/tmp/upload_failures.log"

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def log(msg):
    """Log with AEST timestamp to console and file"""
    now = datetime.now(ZoneInfo("Australia/Brisbane"))
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def get_gcs_client():
    """Create GCS client from environment variable credentials"""
    gcs_credentials_json = os.getenv("GCS_CREDENTIALS_JSON")

    if not gcs_credentials_json:
        log("FATAL: GCS_CREDENTIALS_JSON environment variable not found.")
        return None

    temp_credentials_path = "/tmp/gcs_credentials.json"

    try:
        with open(temp_credentials_path, "w") as f:
            f.write(gcs_credentials_json)

        client = storage.Client.from_service_account_json(temp_credentials_path)
        log("GCS client created successfully")
        return client

    except Exception as e:
        log(f"FATAL: Failed to create GCS client. Error: {e}")
        return None

def extract_ejs_id_from_filename(filename):
    """
    Extract EJS ID from filename.
    Examples:
        - EJS07419_Decision.pdf -> EJS07419
        - EJS-07419.pdf -> EJS07419
        - EJS_07419_John_Smith.pdf -> EJS07419
    """
    match = re.search(r'EJS[-_]?0*(\d+)', filename, re.IGNORECASE)
    if match:
        number = match.group(1)
        return f"EJS{int(number):05d}"  # Format as EJS##### (5 digits)
    return None

def extract_text_from_pdf(pdf_path):
    """Extract full text from PDF file"""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            return full_text, len(reader.pages)
    except Exception as e:
        log(f"❌ Error extracting text from {pdf_path}: {e}")
        return None, 0

EXTRACTION_PROMPT = """
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
- Document length → number of pages provided separately.
- Act Category → classify as either "BCIPA 2004 (Qld)" or "BIF Act 2017 (Qld)" depending on which Act governs the decision.
- jurisdiction_upheld → 1 if jurisdictional objection upheld, else 0.
- reference → the application/decision reference number (e.g., "ADJ-2025-123" or similar format from the document).

Extract as JSON with fields:
- ejs_id
- reference
- adjudicator_name
- claimant_name
- respondent_name
- claimed_amount
- payment_schedule_amount
- adjudicated_amount
- jurisdiction_upheld
- fee_claimant_proportion
- fee_respondent_proportion
- decision_date (format: YYYY-MM-DD)
- keywords (array of 10 strings)
- outcome
- sections_referenced
- project_type
- contract_type
- act_category
"""

def extract_metadata_with_ai(ejs_id, full_text, doc_length_pages):
    """Use OpenAI to extract structured metadata from decision text"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a careful legal data extraction assistant."},
                    {"role": "user", "content": EXTRACTION_PROMPT + f"\n\nEJS ID: {ejs_id}\nDocument Pages: {doc_length_pages}\n\n---\n\n" + full_text[:50000]}
                ]
            )

            content = resp.choices[0].message.content
            data = json.loads(content)

            # Ensure required fields
            data["ejs_id"] = ejs_id
            data["doc_length_pages"] = doc_length_pages

            return data

        except Exception as e:
            if attempt == 2:
                log(f"❌ Failed to extract metadata for {ejs_id}: {e}")
                return None
            time.sleep(2)

    return None

def upload_pdf_to_gcs(pdf_path, ejs_id, gcs_client, bucket_name):
    """Upload PDF to GCS bucket"""
    try:
        pdf_filename = f"{ejs_id}.pdf"
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(f"pdfs/{pdf_filename}")

        with open(pdf_path, 'rb') as f:
            blob.upload_from_file(f, content_type='application/pdf')

        gcs_path = f"/gcs/{bucket_name}/pdfs/{pdf_filename}"
        log(f"✅ Uploaded {pdf_filename} to GCS")
        return gcs_path

    except Exception as e:
        log(f"❌ Failed to upload {pdf_path} to GCS: {e}")
        return None

def insert_into_database(con, ejs_id, metadata, gcs_path, full_text):
    """Insert decision into all database tables"""
    try:
        # Insert into docs_fresh
        con.execute(
            "INSERT OR REPLACE INTO docs_fresh (ejs_id, reference, pdf_path, full_text) VALUES (?, ?, ?, ?)",
            (ejs_id, metadata.get("reference"), gcs_path, full_text)
        )

        # Insert into decision_details
        con.execute(
            """
            INSERT OR REPLACE INTO decision_details (
                ejs_id, adjudicator_name, claimant_name, respondent_name, claimed_amount,
                payment_schedule_amount, adjudicated_amount, jurisdiction_upheld,
                fee_claimant_proportion, fee_respondent_proportion, decision_date,
                keywords, outcome, sections_referenced, project_type, contract_type,
                doc_length_pages, act_category, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ejs_id,
                metadata.get("adjudicator_name"),
                metadata.get("claimant_name"),
                metadata.get("respondent_name"),
                metadata.get("claimed_amount"),
                metadata.get("payment_schedule_amount"),
                metadata.get("adjudicated_amount"),
                metadata.get("jurisdiction_upheld", 0),
                metadata.get("fee_claimant_proportion"),
                metadata.get("fee_respondent_proportion"),
                metadata.get("decision_date"),
                ", ".join(metadata.get("keywords", [])) if isinstance(metadata.get("keywords"), list) else metadata.get("keywords"),
                metadata.get("outcome"),
                metadata.get("sections_referenced"),
                metadata.get("project_type"),
                metadata.get("contract_type"),
                metadata.get("doc_length_pages"),
                metadata.get("act_category"),
                json.dumps(metadata)
            )
        )

        # Update FTS index
        new_doc_row = con.execute("SELECT rowid FROM docs_fresh WHERE ejs_id = ?", (ejs_id,)).fetchone()
        if new_doc_row:
            new_rowid = new_doc_row[0]
            con.execute("INSERT OR REPLACE INTO fts (rowid, full_text) VALUES (?, ?)", (new_rowid, full_text))

        con.commit()
        log(f"✅ Inserted {ejs_id} into database")
        return True

    except Exception as e:
        log(f"❌ Failed to insert {ejs_id} into database: {e}")
        con.rollback()
        return False

def upload_database_to_gcs(gcs_client, bucket_name, db_path):
    """Upload updated database to GCS"""
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob("qbcc.db")
        blob.upload_from_filename(db_path)
        log(f"✅ Uploaded updated database to GCS")
        return True
    except Exception as e:
        log(f"❌ Failed to upload database to GCS: {e}")
        return False

def process_pdf(pdf_path, con, gcs_client, bucket_name, skip_existing=True):
    """Process a single PDF file"""
    filename = os.path.basename(pdf_path)

    # Extract EJS ID from filename
    ejs_id = extract_ejs_id_from_filename(filename)
    if not ejs_id:
        log(f"⚠️ Could not extract EJS ID from filename: {filename}")
        return False

    # Check if already exists
    if skip_existing:
        existing = con.execute("SELECT ejs_id FROM docs_fresh WHERE ejs_id = ?", (ejs_id,)).fetchone()
        if existing:
            log(f"⏭️ Skipping {ejs_id} - already exists in database")
            return False

    log(f"➡️ Processing {ejs_id} ({filename})")

    # Extract text
    full_text, doc_length_pages = extract_text_from_pdf(pdf_path)
    if not full_text:
        log(f"❌ Failed to extract text from {filename}")
        with open(FAIL_FILE, "a") as f:
            f.write(f"{ejs_id} - {filename} - Failed text extraction\n")
        return False

    # Extract metadata with AI
    log(f"  🤖 Extracting metadata for {ejs_id}...")
    metadata = extract_metadata_with_ai(ejs_id, full_text, doc_length_pages)
    if not metadata:
        log(f"❌ Failed to extract metadata for {ejs_id}")
        with open(FAIL_FILE, "a") as f:
            f.write(f"{ejs_id} - {filename} - Failed metadata extraction\n")
        return False

    # Upload PDF to GCS
    log(f"  ☁️ Uploading {ejs_id} to GCS...")
    gcs_path = upload_pdf_to_gcs(pdf_path, ejs_id, gcs_client, bucket_name)
    if not gcs_path:
        with open(FAIL_FILE, "a") as f:
            f.write(f"{ejs_id} - {filename} - Failed GCS upload\n")
        return False

    # Insert into database
    log(f"  💾 Inserting {ejs_id} into database...")
    success = insert_into_database(con, ejs_id, metadata, gcs_path, full_text)
    if not success:
        with open(FAIL_FILE, "a") as f:
            f.write(f"{ejs_id} - {filename} - Failed database insert\n")
        return False

    log(f"✅ Successfully processed {ejs_id}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Batch upload new QBCC adjudication decisions")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files to upload")
    parser.add_argument("--skip_existing", type=bool, default=True, help="Skip PDFs that already exist in database")
    parser.add_argument("--upload_db", type=bool, default=True, help="Upload updated database to GCS after processing")
    args = parser.parse_args()

    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        log("❌ OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    if not os.getenv("GCS_CREDENTIALS_JSON"):
        log("❌ GCS_CREDENTIALS_JSON environment variable not set")
        sys.exit(1)

    bucket_name = os.getenv("GCS_BUCKET_NAME", "sopal-bucket")

    # Initialize GCS client
    gcs_client = get_gcs_client()
    if not gcs_client:
        log("❌ Failed to initialize GCS client")
        sys.exit(1)

    # Download existing database from GCS
    log("📥 Downloading existing database from GCS...")
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob("qbcc.db")
        blob.download_to_filename(DB_PATH)
        log("✅ Downloaded existing database")
    except Exception as e:
        log(f"❌ Failed to download database: {e}")
        sys.exit(1)

    # Connect to database
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    # Get list of PDFs
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        log(f"❌ Directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    log(f"📁 Found {len(pdf_files)} PDF files in {pdf_dir}")

    if len(pdf_files) == 0:
        log("⚠️ No PDF files found. Exiting.")
        sys.exit(0)

    # Process each PDF
    success_count = 0
    skip_count = 0
    fail_count = 0

    for pdf_path in pdf_files:
        result = process_pdf(pdf_path, con, gcs_client, bucket_name, skip_existing=args.skip_existing)
        if result:
            success_count += 1
        elif result is False:
            skip_count += 1
        else:
            fail_count += 1

    # Summary
    log("\n" + "="*60)
    log(f"📊 SUMMARY:")
    log(f"  ✅ Successfully processed: {success_count}")
    log(f"  ⏭️ Skipped (already exist): {skip_count}")
    log(f"  ❌ Failed: {fail_count}")
    log("="*60)

    # Upload updated database to GCS
    if success_count > 0 and args.upload_db:
        log("\n☁️ Uploading updated database to GCS...")
        if upload_database_to_gcs(gcs_client, bucket_name, DB_PATH):
            log("✅ Database uploaded successfully!")
            log("🎉 All done! Your decisions are now live.")
        else:
            log("❌ Failed to upload database. Manual upload required.")

    # Close database
    con.close()

    if fail_count > 0:
        log(f"\n⚠️ {fail_count} files failed. Check {FAIL_FILE} for details.")

if __name__ == "__main__":
    main()
