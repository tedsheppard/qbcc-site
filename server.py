import os, re, shutil, sqlite3, requests, unicodedata, pandas as pd, io, json
from urllib.parse import unquote_plus
from fastapi import FastAPI, Query, Form, Path, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from email.message import EmailMessage
import aiosmtplib
from openai import OpenAI
from google.cloud import storage
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import PyPDF2
import docx
import extract_msg # Added for .msg and .eml support

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
            print("GCS_BUCKET_NAME not set. Trying to copy local 'qbcc.db'.")
            if os.path.exists("qbcc.db"):
                shutil.copy("qbcc.db", DB_PATH)
            else:
                print("FATAL: No local 'qbcc.db' found and GCS bucket not configured.")
    except Exception as e:
        print(f"FATAL: Failed to download database from GCS. Error: {e}")

# sqlite connection
con = sqlite3.connect(DB_PATH, check_same_thread=False)
con.row_factory = sqlite3.Row
con.execute("PRAGMA cache_size = -20000")
con.execute("PRAGMA temp_store = MEMORY")
con.execute("PRAGMA mmap_size = 30000000000")

app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A simple in-memory store for projects for demonstration purposes
PROJECTS_DB = []

# --- RBA Interest Rate Setup ---
def setup_rba_table():
    con.execute("""
    CREATE TABLE IF NOT EXISTS rba_rates (
        rate_date DATE PRIMARY KEY,
        rate_value REAL NOT NULL
    )
    """)
    con.commit()

def fetch_and_update_rba_rates():
    print("Scheduler: Starting RBA rate update job...")
    try:
        url = "https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx"
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        excel_data = io.BytesIO(response.content)
        df = pd.read_excel(excel_data, sheet_name='Data', header=1)
        
        date_col = 'Title' 
        rate_col = 'EOD 3-month BABs/NCDs'

        if date_col not in df.columns or rate_col not in df.columns:
            print(f"Scheduler ERROR: Required columns ('{date_col}', '{rate_col}') not found in RBA file.")
            return

        rates_df = df[[date_col, rate_col]].copy()
        rates_df.columns = ['rate_date', 'rate_value']
        
        rates_df['rate_date'] = pd.to_datetime(rates_df['rate_date'], errors='coerce')
        rates_df.dropna(subset=['rate_date', 'rate_value'], inplace=True)
        
        rates_df['rate_date'] = rates_df['rate_date'].dt.date

        cursor = con.cursor()
        for index, row in rates_df.iterrows():
            cursor.execute("""
                INSERT OR IGNORE INTO rba_rates (rate_date, rate_value) VALUES (?, ?)
            """, (row['rate_date'], row['rate_value']))
        con.commit()
        print(f"Scheduler: RBA rate update complete. {len(rates_df)} rows processed.")

    except Exception as e:
        print(f"Scheduler ERROR: Failed to fetch or update RBA rates. Error: {e}")

# Setup and run the scheduler
setup_rba_table()
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_and_update_rba_rates, 'interval', days=1)
scheduler.start()
# Run once on startup as well
fetch_and_update_rba_rates()


# OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
con.execute("""
CREATE TABLE IF NOT EXISTS ai_summaries (
    decision_id TEXT PRIMARY KEY,
    summary TEXT
)
""")
con.commit()

# ---------------- ensure FTS ----------------
def ensure_fts():
    try:
        con.execute("DROP TABLE IF EXISTS fts;")
        con.execute("CREATE VIRTUAL TABLE fts USING fts4(full_text, content='docs_fresh', tokenize=unicode61);")
        con.execute("INSERT INTO fts(fts) VALUES('rebuild');")
        con.commit()
    except Exception as e:
        print("Error rebuilding FTS index:", e)

ensure_fts()

# ---------------- meilisearch config ----------------
MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:7700")
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "")
MEILI_INDEX = "decisions"


# ---------------- helpers ----------------
def normalize_query(q: str) -> str:
    s = q or ""
    for _ in range(2):
        s2 = unquote_plus(s)
        if s2 == s:
            break
        s = s2
    s = unicodedata.normalize("NFKC", s)
    
    replacements = {
        'â€œ': '"', 'â€': '"', 'â€˜': "'", 'â€™': "'", 
        'â€"': '-', 'â€"': '-', 'â€': '-'
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    
    s = re.sub(r"\s+", " ", s).strip()
    return s

def escape_fts_phrase(s: str) -> str:
    return s.replace('"', '""')

def normalize_default(q: str) -> str:
    if not re.search(r'"|\bNEAR/\d+\b|\bw/\d+\b|\bAND\b|\bOR\b|\bNOT\b|\(|\)', q or "", flags=re.I):
        toks = re.findall(r'\w+\*?', q or "")
        q = ' AND '.join(toks)
    return q

def _fix_unbalanced_quotes(s: str) -> str:
    if s.count('"') % 2 == 1:
        s = s + '"'
    return s

def expand_wildcards(q: str) -> str:
    print(f"Before wildcard expansion: {q}")
    expanded = re.sub(r'(\w+)!', r'\1*', q)
    print(f"After wildcard expansion: {expanded}")
    return expanded

def _parse_near_robust(q: str) -> str | None:
    s = _fix_unbalanced_quotes(q)
    
    if not re.search(r'\b(?:w|near)\s*/\s*\d+\b', s, flags=re.I):
        return None
    
    s = re.sub(r'\b(?:w|near)\s*/\s*(\d+)\b', r'NEAR/\1', s, flags=re.I)

    near_match = re.search(r'NEAR/(\d+)', s, flags=re.I)
    if not near_match:
        return None
    dist = int(near_match.group(1))
    
    parts = re.split(r'\s+NEAR/\d+\s+', s, flags=re.I)
    if len(parts) == 2:
        left_part = parts[0].strip()
        right_part = parts[1].strip()

        def clean_term(term):
            term = term.strip()
            while term.startswith('(') and term.endswith(')'):
                term = term[1:-1].strip()
            if term.startswith('"') and term.endswith('"'):
                term = term[1:-1].strip()
            return term

        left_clean = clean_term(left_part)
        right_clean = clean_term(right_part)
            
        print(f"_parse_near_robust - left: '{left_clean}', dist: {dist}, right: '{right_clean}'")
        return f'"{left_clean}" NEAR/{dist} "{right_clean}"'
    
    return None

def preprocess_sqlite_query(q: str) -> str:
    print(f"preprocess_sqlite_query input: {q}")
    
    q = expand_wildcards(q)
    print(f"After expand_wildcards: {q}")
    
    q = re.sub(r'\band\b', 'AND', q, flags=re.I)
    q = re.sub(r'\bor\b', 'OR', q, flags=re.I)  
    q = re.sub(r'\bnot\b', 'NOT', q, flags=re.I)
    q = re.sub(r'\b(?:w|near)\s*/\s*(\d+)\b', r'NEAR/\1', q, flags=re.I)
    print(f"After operator normalization: {q}")

    near_group_pattern = re.compile(r'(".*?"|\S+)\s+NEAR/(\d+)\s+\((.*?)\)', flags=re.I)
    m = near_group_pattern.search(q)
    
    if m:
        left_term = m.group(1).strip()
        dist = m.group(2)
        right_group = m.group(3).strip()
        
        operator = None
        terms = []
        if ' AND ' in right_group:
            operator = ' AND '
            terms = re.split(r'\s+AND\s+', right_group, flags=re.I)
        elif ' OR ' in right_group:
            operator = ' OR '
            terms = re.split(r'\s+OR\s+', right_group, flags=re.I)
        
        if operator and len(terms) > 1:
            expanded_clauses = []
            for term in terms:
                clean_left = left_term.strip().strip('"')
                clean_right = term.strip().strip('"')
                expanded_clauses.append(f'("{clean_left}" NEAR/{dist} "{clean_right}")')
            
            final_query = operator.join(expanded_clauses)
            print(f"Expanded NEAR/{operator.strip()} query to: {final_query}")
            return final_query

    near_expr = _parse_near_robust(q)
    if near_expr:
        print(f"Found NEAR expression: {near_expr}")
        return near_expr

    if re.search(r'\b(AND|OR|NOT)\b', q, flags=re.I):
        print(f"Contains boolean operators, returning: {q.strip()}")
        return q.strip()

    result = normalize_default(q)
    print(f"normalize_default result: {result}")
    return result

def get_highlight_terms(query: str) -> tuple[list[str], list[str]]:
    phrase_terms = []
    word_terms = []
    
    phrases = re.findall(r'"([^"]+)"', query)
    for phrase in phrases:
        phrase_terms.append(phrase)
    
    query_no_quotes = re.sub(r'"[^"]*"', '', query)
    words = re.findall(r'\b\w+\*?\b', query_no_quotes)
    
    operators = {'AND', 'OR', 'NOT', 'NEAR', 'W'}
    for word in words:
        if (word.upper() not in operators and 
            not word.isdigit() and 
            not re.match(r'\d+', word)):
            word_terms.append(word)
    
    print(f"get_highlight_terms - phrase_terms: {phrase_terms}, word_terms: {word_terms}")
    return phrase_terms, word_terms

_word_re = re.compile(r"\w+", flags=re.UNICODE)

def _tokenize(text: str):
    return _word_re.findall(text.lower())

def _phrase_positions(tokens: list[str], phrase: str) -> list[int]:
    words = _word_re.findall(phrase.lower())
    if not words:
        return []
    if len(words) == 1:
        w = words[0]
        return [i for i, t in enumerate(tokens) if t == w]
    L = len(words)
    out = []
    for i in range(0, len(tokens) - L + 1):
        if tokens[i:i+L] == words:
            out.append(i)
    return out

def _extract_near_components(nq2: str):
    m = re.search(r'"([^"]+)"\s+NEAR/(\d+)\s+"([^"]+)"', nq2, flags=re.I)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)

def _filter_rows_for_true_phrase_near(rows, nq2):
    comp = _extract_near_components(nq2)
    if not comp:
        return rows
    left, dist, right = comp
    left_is_phrase  = ' ' in left.strip()
    right_is_phrase = ' ' in right.strip()
    if not (left_is_phrase or right_is_phrase):
        return rows

    filtered = []
    rowids = [r["rowid"] for r in rows]
    if not rowids:
        return rows

    qmarks = ",".join(["?"]*len(rowids))
    text_rows = con.execute(f"SELECT rowid, full_text FROM docs_fresh WHERE rowid IN ({qmarks})", rowids).fetchall()
    text_map = {tr["rowid"]: tr["full_text"] or "" for tr in text_rows}

    for r in rows:
        ft = text_map.get(r["rowid"], "")
        if not ft:
            continue
        toks = _tokenize(ft)
        left_positions  = _phrase_positions(toks, left)
        right_positions = _phrase_positions(toks, right)
        if not left_positions or not right_positions:
            continue

        ok = False
        for li in left_positions:
            for ri in right_positions:
                if abs(ri - li) <= dist:
                    ok = True
                    break
            if ok:
                break
        if ok:
            filtered.append(r)
    return filtered

def highlight_wildcard_matches(text: str, stem: str) -> str:
    pattern = f'\\b({re.escape(stem)}\\w*)'
    return re.sub(pattern, r'<mark>\1</mark>', text, flags=re.I)

# ---------------- routes ----------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/get_interest_rate")
def get_interest_rate(start_date_str: str = Query(..., alias="startDate"), end_date_str: str = Query(..., alias="endDate")):
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        cursor = con.cursor()
        
        # Create a temporary table of all dates in the range
        cursor.execute("CREATE TEMP TABLE date_range (day DATE)")
        current_date = start_date
        while current_date <= end_date:
            cursor.execute("INSERT INTO date_range (day) VALUES (?)", (current_date,))
            current_date += timedelta(days=1)

        # For each date in our range, find the most recent RBA rate
        cursor.execute("""
            SELECT 
                dr.day,
                (SELECT rate_value FROM rba_rates WHERE rate_date <= dr.day ORDER BY rate_date DESC LIMIT 1) as rate
            FROM date_range dr
        """)
        
        rows = cursor.fetchall()
        
        if not rows:
             raise HTTPException(status_code=404, detail="No interest rates found for the specified date range.")

        # Convert rows to a list of dictionaries
        daily_rates = [{"date": row["day"], "rate": row["rate"]} for row in rows if row["rate"] is not None]

        if not daily_rates:
             raise HTTPException(status_code=404, detail="No valid interest rates could be mapped to the specified date range.")

        return {"dailyRates": daily_rates}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")
    except Exception as e:
        print(f"Error in /get_interest_rate: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/search_fast")
def search_fast(q: str = "", limit: int = 20, offset: int = 0, sort: str = "newest"):
    q_norm = normalize_query(q)
    nq = q_norm

    nq_expanded = expand_wildcards(nq)
    
    is_complex_query = (
        re.search(r'\b(AND|OR|NOT)\b', nq, flags=re.I) or
        re.search(r'\bw/\d+\b', nq, flags=re.I) or
        re.search(r'\bNEAR/\d+\b', nq, flags=re.I) or
        nq.startswith('"') or
        '!' in nq or
        '*' in nq_expanded or
        nq != nq_expanded
    )

    if is_complex_query:
        try:
            nq2 = preprocess_sqlite_query(nq)
            print("Executing MATCH with:", nq2)
            
            total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH ?", (nq2,)).fetchone()[0]

            order_clause = ""
            if sort == "newest":
                order_clause = "ORDER BY m.decision_date_norm DESC"
            elif sort == "oldest":
                order_clause = "ORDER BY m.decision_date_norm ASC"
            elif sort == "atoz":
                order_clause = "ORDER BY m.claimant ASC"
            elif sort == "ztoa":
                order_clause = "ORDER BY m.claimant DESC"

            if order_clause:
                sql = f"""
                  SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
                  FROM fts
                  JOIN docs_fresh d ON fts.rowid = d.rowid
                  LEFT JOIN docs_meta m ON d.ejs_id = m.ejs_id
                  WHERE fts MATCH ?
                  {order_clause}
                  LIMIT ? OFFSET ?
                """
            else:
                sql = """
                  SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
                  FROM fts
                  WHERE fts MATCH ?
                  LIMIT ? OFFSET ?
                """
            
            rows = con.execute(sql, (nq2, limit, offset)).fetchall()

        except sqlite3.OperationalError as e:
            print("FTS MATCH error for:", nq, "->", e)
            if re.search(r'\b(w|near)\s*/\s*\d+', nq, flags=re.I):
                repaired = _parse_near_robust(nq)
                if repaired:
                    print("Retrying with repaired proximity:", repaired)
                    try:
                        total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH ?", (repaired,)).fetchone()[0]
                        rows = con.execute("""
                          SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
                          FROM fts
                          WHERE fts MATCH ?
                          LIMIT ? OFFSET ?
                        """, (repaired, limit, offset)).fetchall()
                        nq2 = repaired
                    except sqlite3.OperationalError:
                        degraded = re.sub(r'\b(?:w|near)\s*/\s*\d+\b', 'AND', nq, flags=re.I)
                        degraded = re.sub(r'[!*]', '', degraded)
                        print("Degrading to:", degraded)
                        total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH ?", (degraded,)).fetchone()[0]
                        rows = con.execute("""
                          SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
                          FROM fts
                          WHERE fts MATCH ?
                          LIMIT ? OFFSET ?
                        """, (degraded, limit, offset)).fetchall()
                        nq2 = degraded
                else:
                    degraded = re.sub(r'\b(?:w|near)\s*/\s*\d+\b', 'AND', nq, flags=re.I)
                    degraded = re.sub(r'[!*]', '', degraded)
                    print("Degrading proximity to AND:", degraded)
                    total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH ?", (degraded,)).fetchone()[0]
                    rows = con.execute("""
                      SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
                      FROM fts
                      WHERE fts MATCH ?
                      LIMIT ? OFFSET ?
                    """, (degraded, limit, offset)).fetchall()
                    nq2 = degraded
            else:
                fallback = re.sub(r'[!*]', '', nq)
                total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH ?", (fallback,)).fetchone()[0]
                rows = con.execute("""
                  SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
                  FROM fts
                  WHERE fts MATCH ?
                  LIMIT ? OFFSET ?
                """, (fallback, limit, offset)).fetchall()
                nq2 = fallback

        is_simple_near_query = "NEAR/" in nq2 and " AND " not in nq2 and " OR " not in nq2

        if is_simple_near_query:
            comp = _extract_near_components(nq2)
            if comp:
                left, _, right = comp
                if (' ' in left.strip()) or (' ' in right.strip()):
                    print(f"Applying true-phrase proximity filter for simple NEAR query.")
                    before = len(rows)
                    rows = _filter_rows_for_true_phrase_near(rows, nq2)
                    print(f"Phrase-proximity filtered page results: {before} → {len(rows)}")

        items = []
        phrase_terms, word_terms = get_highlight_terms(nq2)
        
        for r in rows:
            meta = con.execute("""
              SELECT m.claimant, m.respondent, m.adjudicator, m.decision_date_norm,
                     m.act, d.reference, d.pdf_path, d.ejs_id
              FROM docs_fresh d
              LEFT JOIN docs_meta m ON d.ejs_id = m.ejs_id
              WHERE d.rowid = ?
            """, (r["rowid"],)).fetchone()

            d = dict(meta) if meta else {}
            d["id"] = d.get("ejs_id", r["rowid"])

            snippet_raw = r["snippet"]
            snippet_clean = re.sub(r'\b\d+([A-Za-z]+)\b', r'\1', snippet_raw)

            for phrase in sorted(set(phrase_terms), key=len, reverse=True):
                pattern = re.escape(phrase)
                snippet_clean = re.sub(fr'(?i)\b{pattern}\b', f"<mark>{phrase}</mark>", snippet_clean)

            for term in sorted(set(word_terms), key=len, reverse=True):
                if any(term.lower() in phrase.lower() for phrase in phrase_terms):
                    continue
                
                if term.endswith('*'):
                    stem = term[:-1]
                    pattern = f'\\b({re.escape(stem)}\\w*)'
                    snippet_clean = re.sub(pattern, r'<mark>\1</mark>', snippet_clean, flags=re.I)
                else:
                    pattern = re.escape(term)
                    snippet_clean = re.sub(fr'(?i)\b({pattern})\b', r'<mark>\1</mark>', snippet_clean)

            d["snippet"] = snippet_clean
            items.append(d)

        return {"total": total, "items": items}

    # --- Natural language via Meili ---
    payload = {
        "q": q_norm,
        "limit": limit,
        "offset": offset,
        "attributesToRetrieve": [
            "id", "reference", "pdf_path",
            "claimant", "respondent", "adjudicator",
            "date", "act", "content"
        ],
        "attributesToHighlight": ["content"],
        "highlightPreTag": "<mark>",
        "highlightPostTag": "</mark>",
        "attributesToCrop": ["content"],
        "cropLength": 100
    }
    
    if sort == "relevance" and not q_norm:
        sort = "newest"

    if sort == "newest":
        payload["sort"] = ["id:desc"]
    elif sort == "oldest":
        payload["sort"] = ["id:asc"]
    elif sort == "atoz":
        payload["sort"] = ["claimant:asc"]
    elif sort == "ztoa":
        payload["sort"] = ["claimant:desc"]
        
    headers = {"Authorization": f"Bearer {MEILI_KEY}"} if MEILI_KEY else {}
    res = requests.post(f"{MEILI_URL}/indexes/{MEILI_INDEX}/search", headers=headers, json=payload)
    data = res.json()
    items = []
    for hit in data.get("hits", []):
        snippet = hit.get("_formatted", {}).get("content", "")
        items.append({
            "id": hit.get("id"),
            "reference": hit.get("reference"),
            "pdf_path": hit.get("pdf_path"),
            "claimant": hit.get("claimant"),
            "respondent": hit.get("respondent"),
            "adjudicator": hit.get("adjudicator"),
            "decision_date_norm": hit.get("date"),
            "act": hit.get("act"),
            "snippet": snippet
        })
    return {"total": data.get("estimatedTotalHits", 0), "items": items}

# ---------- PDF links ----------
GCS_BUCKET = "sopal-bucket"
GCS_PREFIX = "pdfs"

def build_gcs_url(file_name: str) -> str:
    return f"https://storage.googleapis.com/{GCS_BUCKET}/{GCS_PREFIX}/{file_name}"

@app.get("/open")
def open_pdf(p: str, disposition: str = "inline"):
    try:
        file_name = os.path.basename(p)
        return {"url": build_gcs_url(file_name)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ---------- AI Summarise ----------
@app.get("/summarise/{decision_id}")
def summarise(decision_id: str = Path(...)):
    try:
        row = con.execute("SELECT summary FROM ai_summaries WHERE decision_id = ?", (decision_id,)).fetchone()
        if row:
            return {"id": decision_id, "summary": row["summary"]}

        r = con.execute("SELECT full_text FROM docs_fresh WHERE ejs_id = ?", (decision_id,)).fetchone()
        if not r or not r["full_text"]:
            raise HTTPException(status_code=404, detail="Decision not found")

        text = r["full_text"][:300000]

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SopalAI, a legal assistant specialising in construction law. "
                        "Summarise adjudication decisions under Queensland's Security of Payment legislation "
                        "in clear, plain English. "
                        "Respond as if the user is asking you about the decision directly. "
                        "Do not say 'the adjudication decision you provided' or similar. "
                        "Structure the summary into bullet points or short sections with HTML-friendly formatting "
                        "(<strong> for headings, <ul>/<li> for lists) with compact spacing (no double/triple lines)."
                        "Structure the summary using only HTML tags (<strong>, <ul>, <li>, <p>), "
                        "not Markdown (#, ##, ###)."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
Summarise this adjudication decision in 5–7 bullet points, covering:
- The parties and the works
- Payment claim amount and payment schedule response
- Any jurisdictional challenges
- The factual disputes and evidence (provide at least 5 sentences of detail on this point, be thorough)
- The adjudicator's reasoning
- The final outcome, amount awarded, and fee split
Don't actually title it "Summary" or the like please, go straight into bullet points

Decision text:
{text}
"""
                }
            ],
            max_tokens=1024,
        )

        summary = resp.choices[0].message.content.strip()
        con.execute("INSERT OR REPLACE INTO ai_summaries(decision_id, summary) VALUES (?, ?)", (decision_id, summary))
        con.commit()
        return {"id": decision_id, "summary": summary}
    except Exception as e:
        print("ERROR in /summarise:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------- FEEDBACK FORM ----------
@app.post("/send-feedback")
async def send_feedback(
    type: str = Form(...),
    name: str = Form(""),
    email: str = Form(""),
    subject: str = Form(...),
    priority: str = Form(...),
    details: str = Form(...),
    browser: str = Form(""),
    device: str = Form("")
):
    msg = EmailMessage()
    msg["From"] = "sopal.aus@gmail.com"
    msg["To"] = "sopal.aus@gmail.com"
    msg["Subject"] = f"[{type.upper()}] {subject}"
    body = f"""
Feedback Type: {type}
Name: {name}
Email: {email}
Priority: {priority}
Browser: {browser}
Device: {device}

Details:
{details}
"""
    msg.set_content(body)

    try:
        await aiosmtplib.send(
            msg,
            hostname="smtp.gmail.com",
            port=587,
            start_tls=True,
            username="sopal.aus@gmail.com",
            password=os.getenv("SMTP_PASSWORD"),
        )
        return {"ok": True, "message": "Feedback sent successfully"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- AI Ask ----------
@app.post("/ask/{decision_id}")
def ask_ai(decision_id: str = Path(...), question: str = Form(...), history: str = Form("[]")):
    try:
        row = con.execute("SELECT full_text FROM docs_fresh WHERE ejs_id = ?", (decision_id,)).fetchone()
        if not row or not row["full_text"]:
            raise HTTPException(status_code=404, detail="Decision not found")

        text = row["full_text"][:30000] # Increased context for chat

        try:
            chat_history = json.loads(history)
            if not isinstance(chat_history, list):
                chat_history = []
        except json.JSONDecodeError:
            chat_history = []

        system_prompt = {
            "role": "system",
            "content": (
                "You are SopalAI, assisting users with adjudication decisions under the BIF Act or BCIPA. "
                "Respond directly to the user's question as if they asked you about the decision. "
                "Do not say 'the adjudication decision you provided' or 'the text you gave me'. "
                "Write in clear, plain English with headings and bullet points formatted in HTML."
                "Structure the summary using only HTML tags (<strong>, <ul>, <li>, <p>), "
                "not Markdown (#, ##, ###). The user has already seen your initial summary, "
                "so your follow-up answers should be concise and directly address their question."
            )
        }
        
        context_prompt = {"role": "user", "content": f"Here is the context from the decision document. Answer questions based on this. CONTEXT:\n{text}"}

        messages = [system_prompt, context_prompt] + chat_history + [{"role": "user", "content": question}]

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1024
        )

        answer = resp.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        print("ERROR in /ask:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Compliance Checker AI Endpoint ----------

def extract_text_from_file(file: io.BytesIO, filename: str) -> str:
    """Extracts text from PDF, DOCX, or TXT files."""
    text = ""
    try:
        if filename.lower().endswith('.pdf'):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif filename.lower().endswith('.docx'):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif filename.lower().endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {filename}")
    return text

def get_system_prompt(doc_type: str) -> str:
    """Returns the expert system prompt for the AI based on document type."""
    if doc_type == 'pc':
        return """
You are an expert construction lawyer in Queensland, specialising in the Building Industry Fairness (Security of Payment) Act 2017 (BIF Act). Your tone must be cautious, thorough, and professional, always erring on the side of highlighting potential risks.

Your task is to analyse the provided text from a document purporting to be a payment claim. Critically assess it for compliance against the BIF Act and key principles from relevant case law. For each check, provide a status ('pass', 'warning', or 'fail'), a short title, and a detailed feedback paragraph explaining your reasoning, citing legislative sections and case law where appropriate.

Your entire response must be a single JSON object with a key "checks" which is an array of check objects. Do not include any text outside of this JSON object.

The check objects must cover the following points in detail:

1.  **Claimed Amount Stated (section 68(1)(b))**: Identify if a clear, specific monetary amount is claimed. A 'pass' requires a single, unambiguous figure. A 'fail' results from ambiguity, multiple conflicting amounts, or no stated amount.
2.  **Identification of Works (section 68(1)(a))**: Assess if the work, related goods, and services are identified with sufficient detail. The standard is not overly onerous, but it must be reasonably clear what the claim is for (see *Trask Development Pty Ltd v. Moreton Bay Regional Council*). A generic description like 'works on site' may be a 'fail'. A project name and general description is usually a 'pass'.
3.  **Request for Payment (section 68(1)(c))**: Check for an explicit request for payment. The Act states a document bearing the word ‘invoice’ is sufficient (section 68(3)). Phrases like 'This is a payment claim made under the BIF Act' are strong indicators. A document merely stating an amount owing without a clear demand for payment may 'fail' (e.g., *KDV Sport Pty Ltd v. Muggeridge Constructions Pty Ltd*).
4.  **Reference Date Validity (section 67)**: This is a critical jurisdictional requirement. 'Fail' the document if no reference date is stated or can be clearly identified. If a date is present, give it a 'pass' but include a comment explaining that the AI cannot verify if this date is valid under the contract or if a claim for this date has already been made. Explain that an invalid reference date is a fatal flaw for the claim.
5.  **Timeliness of Claim (section 75)**: This cannot be verified from the document alone, so always assign a 'warning'. The feedback must explain the critical timeframes: 6 months after the work was last carried out for a progress claim (section 75(2)), and the longer of the periods in section 75(3) for a final claim. Emphasise that failure to comply is a complete bar to proceeding.
6.  **Correct Service**: This also cannot be verified from the document text, so always assign a 'warning'. The feedback must stress the importance of serving the claim on the correct party at the address for notices stipulated in the contract, as improper service can invalidate the entire claim.
"""
    elif doc_type == 'ps':
        return """
You are an expert construction lawyer in Queensland, specialising in the Building Industry Fairness (Security of Payment) Act 2017 (BIF Act). Your tone must be cautious, thorough, and professional, always erring on the side of highlighting potential risks for the respondent.

Your task is to analyse the provided text from a document purporting to be a payment schedule. Critically assess it for compliance against the BIF Act and key principles from relevant case law. For each check, provide a status ('pass', 'warning', or 'fail'), a short title, and a detailed feedback paragraph explaining your reasoning, citing legislative sections and case law where appropriate.

Your entire response must be a single JSON object with a key "checks" which is an array of check objects. Do not include any text outside of this JSON object.

The check objects must cover the following points in detail:

1.  **Identifies Payment Claim (section 69(a))**: Check if the schedule clearly and unambiguously identifies the payment claim it is responding to (e.g., by date, claim number, or project reference). A 'fail' here could render the entire schedule invalid.
2.  **Scheduled Amount Stated (section 69(b))**: Check if it states the amount of the payment, if any, that the respondent proposes to make. This can be zero, but a figure must be stated. Failure to do so is a 'fail'.
3.  **Reasons for Withholding (section 69(c))**: This is the most critical compliance point. The reasons for withholding payment must be articulated with sufficient particularity. A 'fail' is warranted for vague, generic reasons like 'defective works' or 'incomplete works' without further detail. The reasons must be comprehensive enough for the claimant to understand the case they have to meet at adjudication (*John Holland Pty Ltd v. TAC Pacific Pty Ltd*). Emphasise that the respondent is bound by the reasons in their schedule and cannot introduce new reasons later.
4.  **Timeliness of Schedule (section 76)**: This cannot be verified from the document alone, so always assign a 'warning'. The feedback must explain the strict timeframe for service: 15 business days after receiving the payment claim or a shorter period if specified in the contract) and state that failure to provide a compliant schedule on time makes the respondent liable for the full claimed amount.
5.  **Correct Service**: This also cannot be verified from the document text, so always assign a 'warning'. The feedback must stress the importance of serving the schedule on the correct party at the address for notices stipulated in the contract, as improper service is equivalent to not serving one at all.
"""
    return ""

@app.post("/analyse-document")
async def analyse_document(doc_type: str = Form(...), file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        # Read file content into a BytesIO object for processing
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)
        
        extracted_text = extract_text_from_file(file_stream, file.filename)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

        system_prompt = get_system_prompt(doc_type)
        if not system_prompt:
            raise HTTPException(status_code=400, detail="Invalid document type specified.")
            
        user_prompt = f"Here is the text from the document to be analysed:\n\n---DOCUMENT START---\n{extracted_text}\n---DOCUMENT END---"

        # Make the call to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        ai_response_content = response.choices[0].message.content
        
        # The response should be a JSON string, which we parse and return
        return json.loads(ai_response_content)

    except HTTPException as e:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise e
    except Exception as e:
        print(f"ERROR in /analyse-document: {e}")
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)

# -------------------------------------------------------------------
# --- START: LEXIFILE FUNCTIONALITY (MERGED & UPDATED) ---
# -------------------------------------------------------------------

def get_renaming_system_prompt() -> str:
    """Returns the expert system prompt for the AI to rename documents."""
    return """
You are an expert legal assistant AI named LexiFile. Your task is to analyze text from a legal document and extract key information.

Your entire response must be a single, valid JSON object. Do not include any text, notes, or apologies outside of this JSON object.

The JSON object must have the following keys:
1.  "date": A string representing the primary date found in the document, formatted as YYYYMMDD. Find the execution date, letter date, or filing date. If no date can be found, use "00000000".
2.  "docType": A string for the "Strict Name". This must follow a strict, uniform format.
    - For letters/emails: "Letter from [Sender] to [Recipient]" or "Email from [Sender] to [Recipient]". Identify parties by name.
    - For contracts/agreements: "Contract between [Party 1] and [Party 2]".
    - For court documents: "Affidavit of [Name]", "Statement of Claim", "Notice of Motion".
    - For other common types: "Invoice", "Receipt", "Photograph", "Meeting Minutes".
    - Be specific. "Letter of Offer" is not specific enough. It should be "Letter from [Company] to [Applicant]".
3.  "description": A string for the "Looser Description". This should be a brief, 2-5 word summary of the document's subject matter.
4.  "summary": A concise, one-sentence summary of the document's main purpose.
5.  "keywords": An array of up to 10 relevant string keywords extracted from the document.
6.  "metadata": An object with the following keys. If the information is not present, use an empty string "" as the value.
    - "privileged": A string, either "Yes" or "No". Infer this if the document contains phrases like "privileged and confidential". Default to "No".
    - "author": A string for the document author, if identifiable.
    - "createdDate": A string for the creation date, if different from the primary date, formatted as YYYY-MM-DD.
    - "lastModified": A string for the last modified date, if available.
"""

def extract_lexifile_text(file: io.BytesIO, filename: str) -> str:
    """Extracts text for LexiFile, now supporting more types."""
    text = ""
    lower_filename = filename.lower()
    try:
        if lower_filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif lower_filename.endswith('.docx') or lower_filename.endswith('.doc'):
            # python-docx can handle both in many cases
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif lower_filename.endswith('.msg') or lower_filename.endswith('.eml'):
            msg = extract_msg.Message(file)
            text += f"From: {msg.sender}\n"
            text += f"To: {msg.to}\n"
            text += f"Subject: {msg.subject}\n\n"
            text += msg.body
        else:
            # For other types like images, we won't extract text here
            return ""
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        # Don't raise an exception, just return empty text so AI can handle it
        return ""
    return text

@app.post("/rename-document")
async def rename_document(file: UploadFile = File(...)):
    """Handles the document upload and renaming logic for LexiFile."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)
        extracted_text = extract_lexifile_text(file_stream, file.filename)
        
        if not extracted_text.strip():
             # For files with no text (like images), create a default response
            file_extension = file.filename.split('.')[-1]
            doc_type = "Image" if file_extension.lower() in ['jpg', 'jpeg', 'png', 'gif'] else "Unsupported File"
            return JSONResponse(content={
                "date": "00000000",
                "docType": doc_type,
                "description": "Media file",
                "summary": "This is a file with no extractable text.",
                "keywords": ["media", file.filename.split('.')[-1]],
                "metadata": { "privileged": "No", "author": "", "createdDate": "", "lastModified": "" }
            })

        system_prompt = get_renaming_system_prompt()
        user_prompt = f"Please analyze the following document text and return the structured JSON for renaming:\n\n---DOCUMENT TEXT---\n{extracted_text[:12000]}\n---END TEXT---"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        ai_response_content = response.choices[0].message.content
        return json.loads(ai_response_content)
    except json.JSONDecodeError:
         raise HTTPException(status_code=500, detail="AI returned an invalid JSON response.")
    except Exception as e:
        print(f"ERROR in /rename-document: {e}")
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)

@app.post("/preview-email")
async def preview_email(file: UploadFile = File(...)):
    """Parses an email file and returns its components for preview."""
    if not file or not (file.filename.lower().endswith('.msg') or file.filename.lower().endswith('.eml')):
        raise HTTPException(status_code=400, detail="Invalid file type for email preview.")
    try:
        file_content = await file.read()
        msg = extract_msg.Message(io.BytesIO(file_content))
        return {
            "from": msg.sender,
            "to": msg.to,
            "cc": msg.cc,
            "subject": msg.subject,
            "date": msg.date,
            "body": msg.body # extract-msg prefers HTML body if available
        }
    except Exception as e:
        print(f"Error parsing email file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse email file.")

# --- Project Management Endpoints ---
@app.get("/projects")
async def get_projects():
    return PROJECTS_DB

@app.post("/create-project")
async def create_project(projectName: str = Form(...), clientName: str = Form(...), matterNumber: str = Form(...)):
    project_id = len(PROJECTS_DB) + 1
    new_project = {
        "id": project_id,
        "name": projectName,
        "client": clientName,
        "matter": matterNumber,
        "dateCreated": datetime.now().strftime("%Y-%m-%d")
    }
    PROJECTS_DB.append(new_project)
    return new_project

# -------------------------------------------------------------------
# --- END: LEXIFILE FUNCTIONALITY ---
# -------------------------------------------------------------------


# ---------- DB Download ----------
@app.get("/download-db")
async def download_db():
    return FileResponse("/tmp/qbcc.db", filename="qbcc.db")

from fastapi.responses import RedirectResponse

# ---------- serve frontend with clean URLs ----------
@app.get("/{path_name}")
async def serve_html_page(path_name: str):
    file_path = os.path.join(SITE_DIR, f"{path_name}.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(SITE_DIR, "index.html"))

# keep the old mount for static assets like CSS/JS/images
app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")

