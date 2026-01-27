# Instructions for Uploading New QBCC Decisions

This guide explains how to add new adjudication decisions to your Sopal database.

## Prerequisites

1. **Environment Variables** - Set these in your shell:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export GCS_CREDENTIALS_JSON='{"type": "service_account", ...}'  # Your GCS service account JSON
   export GCS_BUCKET_NAME="sopal-bucket"  # Optional, defaults to sopal-bucket
   ```

2. **Python Dependencies** - Already in requirements.txt:
   - PyPDF2
   - openai
   - google-cloud-storage

## Step-by-Step Process

### Step 1: Check What's Missing

First, find out how many decisions you're missing:

```bash
python check_missing_decisions.py --latest EJS07500
```

Replace `EJS07500` with the latest EJS ID you see on the QBCC registry.

This will show you:
- Your current most recent decision
- How many decisions are missing
- A list of all missing EJS IDs

### Step 2: Download PDFs from QBCC

1. Visit https://my.qbcc.qld.gov.au/myQBCC/s/adjudication-registry
2. Search/filter for decisions from your most recent date onwards
3. Download each PDF decision
4. Name them with the EJS ID (e.g., `EJS07419.pdf`, `EJS07420.pdf`, etc.)
   - The script accepts various formats: `EJS07419.pdf`, `EJS-07419.pdf`, `EJS_07419_anything.pdf`
5. Place all PDFs in a folder (e.g., `./new_decisions/`)

### Step 3: Run the Upload Script

```bash
python upload_new_decisions.py --pdf_dir ./new_decisions/
```

**What this does:**
1. Downloads your current database from GCS
2. For each PDF:
   - Extracts the EJS ID from filename
   - Extracts text from the PDF
   - Uses OpenAI to extract structured metadata (adjudicator, parties, amounts, dates, etc.)
   - Uploads the PDF to GCS bucket (`pdfs/` folder)
   - Inserts records into `docs_fresh` and `decision_details` tables
   - Updates the FTS (full-text search) index
3. Uploads the updated database back to GCS
4. Logs everything to `/tmp/upload_decisions.log`

### Step 4: Verify & Deploy

After the script completes successfully:

1. **Verify the upload:**
   ```bash
   python check_missing_decisions.py --latest EJS07500
   ```
   Should now show 0 missing decisions!

2. **Commit your changes** (if you modified any code):
   ```bash
   git add .
   git commit -m "Update database with new decisions through EJS07500"
   git push
   ```

3. **Render will automatically redeploy** with the updated database from GCS

## Options & Flags

### upload_new_decisions.py

- `--pdf_dir` (required): Directory containing PDF files
- `--skip_existing` (default: True): Skip PDFs already in database
- `--upload_db` (default: True): Upload updated database to GCS after processing

Example with options:
```bash
python upload_new_decisions.py \
  --pdf_dir ./new_decisions/ \
  --skip_existing True \
  --upload_db True
```

### check_missing_decisions.py

- `--latest` (required): Latest EJS ID from QBCC registry
- `--download_db` (default: True): Download database from GCS first

## Troubleshooting

### Script Fails During Upload

Check the log file:
```bash
cat /tmp/upload_decisions.log
```

Check the failures file:
```bash
cat /tmp/upload_failures.log
```

### Common Issues

1. **"Could not extract EJS ID from filename"**
   - Ensure PDF filenames contain the EJS ID (e.g., `EJS07419.pdf`)

2. **"Failed to extract text from PDF"**
   - PDF may be corrupted or image-based (scanned)
   - Try downloading the PDF again from QBCC

3. **"Failed metadata extraction"**
   - OpenAI API may be down or rate-limited
   - Check your OPENAI_API_KEY is valid
   - Script will retry 3 times automatically

4. **"Failed GCS upload"**
   - Check your GCS_CREDENTIALS_JSON is valid
   - Verify you have write permissions to the bucket

### Rerunning Failed Uploads

The script skips existing decisions by default, so you can safely rerun it:

```bash
python upload_new_decisions.py --pdf_dir ./new_decisions/
```

It will only process the PDFs that haven't been uploaded yet.

## Manual Intervention

If you need to manually add a single decision, you can also use the web UI:
1. Go to your admin panel at sopal.com.au/admin.html
2. Use the "Upload Decision" form
3. Fill in all required fields manually

## Cost Estimates

**Per decision:**
- OpenAI API (gpt-4o-mini): ~$0.01-0.03 per decision
- GCS storage: ~$0.02/GB/month
- GCS bandwidth: Free (within same region)

**For 100 new decisions:**
- OpenAI: ~$1-3
- GCS storage: ~$0.50/month
- Total: ~$1.50-3.50 one-time + $0.50/month ongoing

## Notes

- The script uses `gpt-4o-mini` for cost-efficiency
- Full text is truncated to 50,000 characters for AI processing
- All dates are normalized to YYYY-MM-DD format
- Monetary amounts are stored as GST-inclusive
- FTS (full-text search) index is automatically updated
- MeiliSearch sync can be done separately using `sync_meili.py`

## Next Steps

After uploading decisions, you may want to:

1. **Sync to MeiliSearch** (if using):
   ```bash
   python sync_meili.py
   ```

2. **Generate AI summaries**:
   ```bash
   python batch_extract.py --start_id EJS07419 --end_id EJS07500
   ```

3. **Backup the database**:
   ```bash
   python backup_meili.py
   ```
