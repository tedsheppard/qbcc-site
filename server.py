import os, re, shutil, sqlite3, requests, unicodedata, pandas as pd, io
from urllib.parse import unquote_plus
from fastapi import FastAPI, Query, Form, Path, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from email.message import EmailMessage
import aiosmtplib
from openai import OpenAI
from google.cloud import storage
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

# ---------------- setup ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SITE_DIR = os.path.join(ROOT, "site")
DB_PATH = "/tmp/qbcc.db"

# Download DB from GCS on startup if it doesn't exist in /tmp
if not os.path.exists(DB_PATH):
    try:
        gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
        gcs_db_object_name = os.getenv("GCS_DB_OBJECT_NAME", "qbcc.db")
        
        if gcs_bucket_name:
            print(f"Database not found at {DB_PATH}. Downloading from GCS bucket '{gcs_bucket_name}'...")
            storage_client = storage.Client()
            bucket = storage_client.bucket(gcs_bucket_name)
            blob = bucket.blob(gcs_db_object_name)
            blob.download_to_filename(DB_PATH)
            print("Database downloaded successfully.")
        else:
            # Fallback for local development if GCS env var isn't set
            print("GCS_BUCKET_NAME not set. Assuming local DB exists or will be created.")
            # check if a local copy exists in project root
            if os.path.exists("qbcc.db"):
              shutil.copy("qbcc.db", DB_PATH)
              print("Copied local qbcc.db to /tmp/")

    except Exception as e:
        print(f"ERROR: Failed to download database from GCS: {e}")


app = FastAPI()
client = OpenAI()

# ---------------- helpers ----------------
def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db

def preprocess_sqlite_query(q_in):
    print(f"preprocess_sqlite_query input: {q_in}")

    q = q_in or ""

    def expand_wildcards(s):
        print(f"Before wildcard expansion: {s}")
        # Add asterisk to any word ending in !
        s = re.sub(r'(\w+)!', r'\1*', s)
        print(f"After wildcard expansion: {s}")
        return s

    def normalize_default(s):
        # Default to ANDing terms together
        parts = s.split()
        # Don't add AND if a boolean operator is already present
        if not any(op in parts for op in ["AND", "OR", "NOT"]):
            s = " AND ".join(parts)
        return s

    def normalize_operators(s):
        # Convert search operators to uppercase for SQLite FTS5
        s = re.sub(r'\b(and|or|not)\b', lambda m: m.group(1).upper(), s, flags=re.IGNORECASE)
        return s

    # Process phrases first
    phrases = re.findall(r'"[^"]+"', q)
    remaining_q = re.sub(r'"[^"]+"', '', q)

    # Apply processing to remaining query parts
    remaining_q = expand_wildcards(remaining_q)
    print(f"After expand_wildcards: {remaining_q}")
    
    remaining_q = normalize_operators(remaining_q)
    print(f"After operator normalization: {remaining_q}")
    
    # put it all back together
    final_q = ' '.join(phrases) + ' ' + remaining_q
    final_q = final_q.strip()

    # if there were no phrases, and no operators, do default normalization
    if not phrases and not any(op in final_q for op in ["AND", "OR", "NOT"]):
        final_q = normalize_default(final_q)
        print(f"normalize_default result: {final_q}")

    return final_q.strip()

def normalize_for_highlight(s):
  return (s or '')\
    .replace('"', '')\
    .replace('!', '*')

def get_highlight_terms(q):
    q = normalize_for_highlight(q)
    # find phrases first "a b c"
    phrase_terms = re.findall(r'"([^"]+)"', q)
    # remove phrases from q
    q = re.sub(r'"[^"]+"', '', q)
    # get individual words, removing wildcards
    word_terms = [re.sub(r'[*!]', '', w) for w in q.split() if w]
    print(f"get_highlight_terms - phrase_terms: {phrase_terms}, word_terms: {word_terms}")
    return (phrase_terms, word_terms)


# ---------------- API routes ----------------

@app.get("/search_fast")
async def search_fast(
    q: str = None,
    sort: str = 'relevance',
    limit: int = 20,
    offset: int = 0
):
    db = get_db()
    cursor = db.cursor()

    # Pre-process query for FTS5
    match_query = preprocess_sqlite_query(q)
    print(f"Executing MATCH with: {match_query}")

    # base query
    query = f"""
        SELECT id, title, decision_id, snippet(text, 2, '<span class="highlight">', '</span>', '...', 15), date, url, adjudicator
        FROM fts
        WHERE fts MATCH ?
    """

    # sorting
    if sort == 'newest':
        query += " ORDER BY date DESC NULLS LAST"
    elif sort == 'oldest':
        query += " ORDER BY date ASC NULLS LAST"
    # relevance is default FTS sorting, no ORDER BY needed

    query += " LIMIT ? OFFSET ?"

    try:
        cursor.execute(query, (match_query, limit, offset))
        rows = cursor.fetchall()
        
        # Get total count for pagination
        count_cursor = db.cursor()
        count_cursor.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH ?", (match_query,))
        total_count = count_cursor.fetchone()[0]

    except sqlite3.OperationalError as e:
        # This will catch FTS syntax errors
        raise HTTPException(status_code=400, detail=f"Search query syntax error: {e}")

    # get terms to highlight on frontend
    phrase_terms, word_terms = get_highlight_terms(q)

    results = []
    columns = [d[0] for d in cursor.description]
    MAX_SNIPPET_LEN = 300 # Character limit for snippet

    for row_tuple in rows:
        row = list(row_tuple) # convert tuple to list to modify
        snippet_text = row[3] # assuming snippet is at index 3

        if len(snippet_text) > MAX_SNIPPET_LEN:
            truncated_text = snippet_text[:MAX_SNIPPET_LEN]
            last_space = truncated_text.rfind(' ')
            if last_space != -1:
                truncated_text = truncated_text[:last_space]
            
            # Ensure we don't have open highlight tags
            open_tags = truncated_text.count('<span class="highlight">')
            closed_tags = truncated_text.count('</span>')
            
            if open_tags > closed_tags:
                truncated_text += '</span>'
                
            snippet_text = truncated_text + '...'

        row[3] = snippet_text
        results.append(dict(zip(columns, row)))

    db.close()

    return JSONResponse(content={
        "results": results,
        "total": total_count,
        "highlight_terms": {"phrases": phrase_terms, "words": word_terms}
    })


@app.get("/decision/{decision_id}")
async def get_decision(decision_id: str):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM decisions WHERE decision_id = ?", (decision_id,))
    decision = cursor.fetchone()
    db.close()
    if decision is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return JSONResponse(content=dict(decision))

# ---------- Feedback endpoint ----------
@app.post("/submit-feedback")
async def submit_feedback(
    name: str = Form(None),
    email: str = Form(None),
    feedback: str = Form(...)
):
    # Retrieve credentials from environment variables
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")

    if not all([sender_email, sender_password, recipient_email]):
        print("ERROR: Email credentials not configured.")
        raise HTTPException(status_code=500, detail="Email service is not configured.")

    message = EmailMessage()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = "Sopal Feedback Received"
    
    body = f"Feedback from: {name or 'Anonymous'}\n"
    body += f"Email: {email or 'Not provided'}\n\n"
    body += "Feedback:\n"
    body += feedback
    message.set_content(body)

    try:
        await aiosmtplib.send(
            message,
            hostname="smtp.gmail.com",
            port=465,
            use_tls=True,
            username=sender_email,
            password=sender_password,
        )
        return JSONResponse({"message": "Feedback submitted successfully"})
    except Exception as e:
        print(f"ERROR sending email: {e}")
        raise HTTPException(status_code=500, detail="Failed to send feedback.")

# ---------- Ask AI endpoint ----------
@app.post("/ask")
async def ask_question(
    decision_id: str = Form(...),
    question: str = Form(...)
):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT text FROM decisions WHERE decision_id = ?", (decision_id,))
        decision_row = cursor.fetchone()
        db.close()

        if not decision_row:
            raise HTTPException(status_code=404, detail="Decision not found")

        text = decision_row["text"]
        # truncate text to ~15k tokens if it's too long
        if len(text) > 60000:
          text = text[:60000]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SopalAI, assisting users with adjudication decisions under the BIF Act or BCIPA. "
                        "Respond directly to the user's question as if they asked you about the decision. "
                        "Do not say 'the adjudication decision you provided' or 'the text you gave me'. "
                        "Write in clear, plain English with headings and bullet points formatted in HTML."
                        "Structure the summary using only HTML tags (<strong>, <ul>, <li>, <p>), "
                        "not Markdown (#, ##, ###)."
                    )
                },
                {"role": "user", "content": f"Decision text:\\n{text}"},
                {"role": "user", "content": f"Question: {question}"}
            ],
            max_tokens=600
        )

        answer = resp.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        print("ERROR in /ask:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------- DB Download ----------
@app.get("/download-db")
async def download_db():
    return FileResponse("/tmp/qbcc.db", filename="qbcc.db")

# ---------- serve frontend ----------
app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")
