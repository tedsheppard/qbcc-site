#!/usr/bin/env python3
"""
Upload a batch of decisions with application number filenames.
This script assigns sequential EJS IDs starting from EJS07419.
"""

import os
import sys
import json
import sqlite3
import time
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import PyPDF2
from openai import OpenAI
from google.cloud import storage

# Configuration
DB_PATH = "../qbcc-site/qbcc.db"  # Use the downloaded one
PDF_DIR = "../qbcc-site/new_decisions_upload"
LOG_FILE = "/tmp/batch_upload.log"
FAIL_FILE = "/tmp/batch_failures.log"
START_EJS_NUM = 7419  # Next EJS ID to assign

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def log(msg):
    """Log with AEST timestamp"""
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

def extract_app_number_from_filename(filename):
    """Extract application number from filename like 00000002884506_1.pdf"""
    match = re.search(r'(\d{10})', filename)
    if match:
        return match.group(1).lstrip('0')  # Remove leading zeros
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
- Act Category → classify as either "BCIPA 2004 (Qld)" or "BIF Act 2017 (Qld)".
- jurisdiction_upheld → 1 if jurisdictional objection upheld, else 0.
- decision_date must be in YYYY-MM-DD format.

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
- decision_date (YYYY-MM-DD format)
- keywords (array of 10 strings)
- outcome
- sections_referenced
- project_type
- contract_type
- act_category
"""

def extract_metadata_with_ai(ejs_id, full_text, doc_length_pages):
    """Use OpenAI to extract structured metadata"""
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
        log(f"  ☁️  Uploaded to GCS")
        return gcs_path
    except Exception as e:
        log(f"❌ Failed to upload to GCS: {e}")
        return None

def insert_into_database(con, ejs_id, app_number, metadata, gcs_path, full_text):
    """Insert decision into all database tables"""
    try:
        # Insert into docs_fresh
        con.execute(
            "INSERT OR REPLACE INTO docs_fresh (ejs_id, reference, pdf_path, full_text) VALUES (?, ?, ?, ?)",
            (ejs_id, app_number, gcs_path, full_text)
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
        log(f"  💾 Inserted into database")
        return True
    except Exception as e:
        log(f"❌ Failed to insert into database: {e}")
        import traceback
        traceback.print_exc()
        con.rollback()
        return False

def main():
    log("="*60)
    log("Starting batch upload of new decisions")
    log("="*60)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        log("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    if not os.getenv("GCS_CREDENTIALS_JSON"):
        log("❌ GCS_CREDENTIALS_JSON not set")
        sys.exit(1)

    bucket_name = os.getenv("GCS_BUCKET_NAME", "sopal-bucket")

    # Initialize GCS client
    gcs_client = get_gcs_client()
    if not gcs_client:
        sys.exit(1)

    # Connect to database
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    # Get PDF files sorted by application number
    pdf_dir = Path(PDF_DIR)
    pdf_files = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        app_num = extract_app_number_from_filename(pdf_path.name)
        if app_num:
            pdf_files.append((int(app_num), app_num, pdf_path))

    pdf_files.sort()  # Sort by application number
    log(f"📁 Found {len(pdf_files)} PDFs to process")

    # Process each PDF
    success_count = 0
    fail_count = 0
    ejs_counter = START_EJS_NUM

    for _, app_num, pdf_path in pdf_files:
        ejs_id = f"EJS{ejs_counter:05d}"
        log(f"\n➡️  Processing {pdf_path.name}")
        log(f"  📋 App Number: {app_num}")
        log(f"  🆔 EJS ID: {ejs_id}")

        # Extract text
        full_text, doc_length_pages = extract_text_from_pdf(pdf_path)
        if not full_text:
            log(f"❌ Failed to extract text")
            with open(FAIL_FILE, "a") as f:
                f.write(f"{ejs_id} - {app_num} - {pdf_path.name} - Failed text extraction\n")
            fail_count += 1
            ejs_counter += 1
            continue

        # Extract metadata with AI
        log(f"  🤖 Extracting metadata...")
        metadata = extract_metadata_with_ai(ejs_id, full_text, doc_length_pages)
        if not metadata:
            log(f"❌ Failed to extract metadata")
            with open(FAIL_FILE, "a") as f:
                f.write(f"{ejs_id} - {app_num} - {pdf_path.name} - Failed metadata extraction\n")
            fail_count += 1
            ejs_counter += 1
            continue

        # Upload PDF to GCS
        gcs_path = upload_pdf_to_gcs(pdf_path, ejs_id, gcs_client, bucket_name)
        if not gcs_path:
            with open(FAIL_FILE, "a") as f:
                f.write(f"{ejs_id} - {app_num} - {pdf_path.name} - Failed GCS upload\n")
            fail_count += 1
            ejs_counter += 1
            continue

        # Insert into database
        if insert_into_database(con, ejs_id, app_num, metadata, gcs_path, full_text):
            log(f"✅ Successfully processed {ejs_id} (App {app_num})")
            success_count += 1
        else:
            with open(FAIL_FILE, "a") as f:
                f.write(f"{ejs_id} - {app_num} - {pdf_path.name} - Failed database insert\n")
            fail_count += 1

        ejs_counter += 1

    # Summary
    log("\n" + "="*60)
    log("📊 SUMMARY:")
    log(f"  ✅ Successfully processed: {success_count}")
    log(f"  ❌ Failed: {fail_count}")
    log(f"  🆔 EJS IDs assigned: EJS{START_EJS_NUM:05d} - EJS{ejs_counter-1:05d}")
    log("="*60)

    # Close database
    con.close()

    # Upload database to GCS
    if success_count > 0:
        log("\n☁️  Uploading updated database to GCS...")
        try:
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob("qbcc.db")
            blob.upload_from_filename(DB_PATH)
            log("✅ Database uploaded successfully to GCS!")
            log("🎉 All done! New decisions are now live.")
        except Exception as e:
            log(f"❌ Failed to upload database to GCS: {e}")
            log("⚠️  Manual database upload required!")

    if fail_count > 0:
        log(f"\n⚠️  {fail_count} PDFs failed. Check {FAIL_FILE}")

if __name__ == "__main__":
    main()
