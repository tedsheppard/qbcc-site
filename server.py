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
from datetime import datetime, date

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
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        excel_data = io.BytesIO(response.content)
        # Skip the initial metadata rows to get to the actual headers
        df = pd.read_excel(excel_data, sheet_name='Data', header=10)
        
        date_col = 'Title' 
        rate_col = 'EOD 3-month BABs/NCDs'

        if date_col not in df.columns or rate_col not in df.columns:
            print(f"Scheduler ERROR: Required columns ('{date_col}', '{rate_col}') not found in RBA file.")
            return

        rates_df = df[[date_col, rate_col]].copy()
        rates_df.columns = ['rate_date', 'rate_value']
        
        # Convert date column, coercing errors to NaT (Not a Time)
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

app = FastAPI()

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
        '“': '"', '”': '"', '‘': "'", '’': "'", 
        '—': '-', '–': '-', '‐': '-'
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
            expanded_clauses = [f'("{left_term.strip().strip(" ")}" NEAR/{dist} "{term.strip().strip(" ")}")' for term in terms]
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
    phrase_terms = re.findall(r'"([^"]+)"', query)
    query_no_quotes = re.sub(r'"[^"]*"', '', query)
    words = re.findall(r'\b\w+\*?\b', query_no_quotes)
    
    operators = {'AND', 'OR', 'NOT', 'NEAR', 'W'}
    word_terms = [
        word for word in words 
        if word.upper() not in operators and not word.isdigit()
    ]
    
    print(f"get_highlight_terms - phrase_terms: {phrase_terms}, word_terms: {word_terms}")
    return phrase_terms, word_terms

_word_re = re.compile(r"\w+", flags=re.UNICODE)

def _tokenize(text: str):
    return _word_re.findall(text.lower())

def _phrase_positions(tokens: list[str], phrase: str) -> list[int]:
    words = _word_re.findall(phrase.lower())
    if not words: return []
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
    if not m: return None
    return m.group(1), int(m.group(2)), m.group(3)

def _filter_rows_for_true_phrase_near(rows, nq2):
    comp = _extract_near_components(nq2)
    if not comp: return rows
    left, dist, right = comp
    if ' ' not in left.strip() and ' ' not in right.strip():
        return rows

    rowids = [r["rowid"] for r in rows]
    if not rowids: return rows

    qmarks = ",".join(["?"]*len(rowids))
    text_rows = con.execute(f"SELECT rowid, full_text FROM docs_fresh WHERE rowid IN ({qmarks})", rowids).fetchall()
    text_map = {tr["rowid"]: tr["full_text"] or "" for tr in text_rows}
    
    filtered = []
    for r in rows:
        ft = text_map.get(r["rowid"], "")
        if not ft: continue
        toks = _tokenize(ft)
        left_positions  = _phrase_positions(toks, left)
        right_positions = _phrase_positions(toks, right)
        if not left_positions or not right_positions: continue

        ok = any(abs(ri - li) <= dist for li in left_positions for ri in right_positions)
        if ok:
            filtered.append(r)
    return filtered

# ---------------- routes ----------------
@app.get("/health")
def health(): return {"ok": True}

@app.get("/get_interest_rate")
def get_interest_rate(startDate: date, endDate: date):
    try:
        cursor = con.cursor()
        
        cursor.execute("""
            SELECT rate_date, rate_value FROM rba_rates
            WHERE rate_date >= ? AND rate_date <= ?
            ORDER BY rate_date ASC
        """, (startDate, endDate))
        rows = cursor.fetchall()

        daily_rates = {row['rate_date']: row['rate_value'] for row in rows}
        
        # Fill in missing dates (weekends/holidays) with the last known rate
        all_dates = pd.date_range(start=startDate, end=endDate)
        filled_rates = []
        last_rate = None

        # Get the last known rate before the start date for the initial fill
        cursor.execute("""
            SELECT rate_value FROM rba_rates WHERE rate_date < ? ORDER BY rate_date DESC LIMIT 1
        """, (startDate,))
        res = cursor.fetchone()
        if res: last_rate = res['rate_value']

        for dt in all_dates:
            current_date_str = dt.strftime('%Y-%m-%d')
            if current_date_str in daily_rates:
                last_rate = daily_rates[current_date_str]
            
            if last_rate is not None:
                 filled_rates.append({"date": current_date_str, "rate": last_rate})
            else:
                # This should ideally not happen if the DB is populated
                raise HTTPException(status_code=404, detail=f"No rate data available for {current_date_str} or earlier.")

        return {"dailyRates": filled_rates}
    except Exception as e:
        print(f"Error in /get_interest_rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search_fast")
def search_fast(q: str = "", limit: int = 20, offset: int = 0, sort: str = "newest"):
    q_norm = normalize_query(q)
    
    is_complex_query = (
        re.search(r'\b(AND|OR|NOT|w/|NEAR/)\b', q_norm, flags=re.I) or
        '"' in q_norm or '!' in q_norm
    )

    if is_complex_query:
        try:
            nq2 = preprocess_sqlite_query(q_norm)
            print("Executing MATCH with:", nq2)
            
            total_sql = "SELECT COUNT(*) FROM fts WHERE fts MATCH ?"
            total = con.execute(total_sql, (nq2,)).fetchone()[0]

            order_clause = ""
            if sort == "newest": order_clause = "ORDER BY m.decision_date_norm DESC NULLS LAST"
            elif sort == "oldest": order_clause = "ORDER BY m.decision_date_norm ASC NULLS LAST"
            elif sort == "atoz": order_clause = "ORDER BY m.claimant ASC NULLS LAST"
            elif sort == "ztoa": order_clause = "ORDER BY m.claimant DESC NULLS LAST"

            if order_clause:
                sql = f"""
                  SELECT fts.rowid FROM fts
                  JOIN docs_fresh d ON fts.rowid = d.rowid
                  LEFT JOIN docs_meta m ON d.ejs_id = m.ejs_id
                  WHERE fts MATCH ? {order_clause} LIMIT ? OFFSET ?
                """
            else:
                sql = "SELECT rowid FROM fts WHERE fts MATCH ? LIMIT ? OFFSET ?"
            
            rowid_rows = con.execute(sql, (nq2, limit, offset)).fetchall()
            rowids = [r['rowid'] for r in rowid_rows]
            
            if not rowids: return {"total": total, "items": []}

            qmarks = ",".join(["?"]*len(rowids))
            
            # Fetch metadata and snippets in separate queries
            meta_sql = f"""
                SELECT m.claimant, m.respondent, m.adjudicator, m.decision_date_norm,
                       m.act, d.reference, d.pdf_path, d.ejs_id, d.rowid
                FROM docs_fresh d
                LEFT JOIN docs_meta m ON d.ejs_id = m.ejs_id
                WHERE d.rowid IN ({qmarks})
            """
            meta_rows = con.execute(meta_sql, rowids).fetchall()
            meta_map = {r['rowid']: dict(r) for r in meta_rows}

            snippet_sql = f"""
                SELECT rowid, snippet(fts, '[', ']', '…', 35, 35) AS snippet FROM fts WHERE rowid IN ({qmarks})
            """
            snippet_rows = con.execute(snippet_sql, rowids).fetchall()
            snippet_map = {r['rowid']: r['snippet'] for r in snippet_rows}

            items = []
            phrase_terms, word_terms = get_highlight_terms(nq2)
            
            # Reorder items to match the sorted rowids
            for rowid in rowids:
                d = meta_map.get(rowid, {})
                d["id"] = d.get("ejs_id", rowid)
                
                snippet_raw = snippet_map.get(rowid, "")
                # Clean snippet: remove leading zeros from numbers
                snippet_clean = re.sub(r'\b0+(\d+)', r'\1', snippet_raw)

                # Highlight phrases
                for phrase in sorted(set(phrase_terms), key=len, reverse=True):
                    pattern = re.escape(phrase)
                    snippet_clean = re.sub(f'(?i)({pattern})', r'<mark>\1</mark>', snippet_clean)

                # Highlight individual and wildcard terms
                for term in sorted(set(word_terms), key=len, reverse=True):
                    # Skip if term is part of an already highlighted phrase
                    if any(term.lower() in phrase.lower() for phrase in phrase_terms):
                        continue
                    
                    if term.endswith('*'):
                        stem = re.escape(term[:-1])
                        pattern = f'\\b({stem}\\w*)'
                    else:
                        pattern = f'\\b({re.escape(term)})\\b'
                    # Use a function to avoid re-highlighting inside <mark> tags
                    snippet_clean = re.sub(pattern, lambda m: f'<mark>{m.group(0)}</mark>', snippet_clean, flags=re.I)


                d["snippet"] = snippet_clean.replace('[','').replace(']','')
                items.append(d)

            return {"total": total, "items": items}

        except sqlite3.OperationalError as e:
            print(f"FTS MATCH error for: {q_norm} -> {e}")
            return {"total": 0, "items": []}

    # --- Natural language via Meili ---
    payload = {"q": q_norm, "limit": limit, "offset": offset}
    if sort == "newest": payload["sort"] = ["id:desc"]
    elif sort == "oldest": payload["sort"] = ["id:asc"]
    elif sort == "atoz": payload["sort"] = ["claimant:asc"]
    elif sort == "ztoa": payload["sort"] = ["claimant:desc"]
        
    headers = {"Authorization": f"Bearer {MEILI_KEY}"} if MEILI_KEY else {}
    res = requests.post(f"{MEILI_URL}/indexes/{MEILI_INDEX}/search", headers=headers, json=payload)
    data = res.json()
    items = []
    for hit in data.get("hits", []):
        items.append({
            "id": hit.get("id"), "reference": hit.get("reference"), "pdf_path": hit.get("pdf_path"),
            "claimant": hit.get("claimant"), "respondent": hit.get("respondent"),
            "adjudicator": hit.get("adjudicator"), "decision_date_norm": hit.get("date"),
            "act": hit.get("act"), "snippet": hit.get("content","")[:300]
        })
    return {"total": data.get("estimatedTotalHits", 0), "items": items}

@app.post("/send-feedback")
async def send_feedback(
    type: str = Form(...), name: str = Form(""), email: str = Form(""),
    subject: str = Form(...), priority: str = Form(...), details: str = Form(...),
    browser: str = Form(""), device: str = Form("")
):
    msg = EmailMessage()
    msg["From"] = "sopal.aus@gmail.com"
    msg["To"] = "sopal.aus@gmail.com"
    msg["Subject"] = f"[{type.upper()}] {subject}"
    body = f"Feedback Type: {type}\nName: {name}\nEmail: {email}\nPriority: {priority}\nBrowser: {browser}\nDevice: {device}\n\nDetails:\n{details}"
    msg.set_content(body)
    try:
        await aiosmtplib.send(
            msg, hostname="smtp.gmail.com", port=587, start_tls=True,
            username="sopal.aus@gmail.com", password=os.getenv("SMTP_PASSWORD"),
        )
        return {"ok": True, "message": "Feedback sent successfully"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ask/{decision_id}")
def ask_ai(decision_id: str = Path(...), question: str = Form(...)):
    try:
        row = con.execute("SELECT full_text FROM docs_fresh WHERE ejs_id = ?", (decision_id,)).fetchone()
        if not row or not row["full_text"]:
            raise HTTPException(status_code=404, detail="Decision not found")
        text = row["full_text"][:15000]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are SopalAI..."},
                {"role": "user", "content": f"Decision text:\n{text}"},
                {"role": "user", "content": f"Question: {question}"}
            ], max_tokens=600
        )
        return {"answer": resp.choices[0].message.content.strip()}
    except Exception as e:
        print(f"ERROR in /ask: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")

