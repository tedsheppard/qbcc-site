import os, re, shutil, sqlite3, requests, unicodedata, pandas as pd, io, json
from urllib.parse import unquote_plus
from fastapi import FastAPI, Query, Form, Path, HTTPException, UploadFile, File, Body
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from email.message import EmailMessage
import aiosmtplib
from openai import OpenAI
from google.cloud import storage
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta, date
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import PyPDF2
import docx
import extract_msg
import pypandoc
from typing import List, Optional
import zipstream
import pytesseract
from PIL import Image

# ---------------- setup ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SITE_DIR = os.path.join(ROOT, "site")
DB_PATH = "/tmp/qbcc.db"
LEXIFILE_STORAGE = "/tmp/lexifile_storage"
os.makedirs(LEXIFILE_STORAGE, exist_ok=True)


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

# ---------------- LexiFile DB (separate from Sopal qbcc.db) ----------------
LEXIFILE_DB_PATH = "/tmp/lexifile.db"
lexi_con = sqlite3.connect(LEXIFILE_DB_PATH, check_same_thread=False)
lexi_con.row_factory = sqlite3.Row

# Create users table (LexiFile only)
lexi_con.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
lexi_con.commit()

# ---------------- Auth setup ----------------
SECRET_KEY = os.getenv("LEXIFILE_SECRET_KEY", "dev-secret-key")  # change in prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_email(email: str):
    cur = lexi_con.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    return dict(row) if row else None

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Databases for LexiFile Demonstration ---
PROJECTS_DB = []
DOCUMENTS_DB = {} 
ARTIFACT_ID_COUNTER = 1


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
    cursor = con.cursor()
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
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
    finally:
        # Ensure the temporary table is dropped
        cursor.execute("DROP TABLE IF EXISTS date_range")

from fastapi import Depends

@app.post("/register")
def register(email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email)
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = get_password_hash(password)
    lexi_con.execute("INSERT INTO users (email, hashed_password) VALUES (?, ?)", (email, hashed_pw))
    lexi_con.commit()
    return {"msg": "User registered successfully"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.get("/me")
def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"email": current_user["email"], "created_at": current_user["created_at"]}

@app.get("/search_fast")
def search_fast(q: str = "", limit: int = 20, offset: int = 0, sort: str = "newest"):
    q_norm = normalize_query(q)
    nq = q_norm

    # Check if query is an EJS ID (format: numbers followed by letters, e.g. "12345ABC")
    if re.match(r'^\d+[A-Z]+$', nq.upper().replace(' ', '')):
        ejs_id = nq.upper().replace(' ', '')
        try:
            # Direct lookup by EJS ID
            meta = con.execute("""
              SELECT m.claimant, m.respondent, m.adjudicator, m.decision_date_norm,
                     m.act, d.reference, d.pdf_path, d.ejs_id, d.full_text
              FROM docs_fresh d
              LEFT JOIN docs_meta m ON d.ejs_id = m.ejs_id
              WHERE d.ejs_id = ?
            """, (ejs_id,)).fetchone()
            
            if meta:
                result = dict(meta)
                result["id"] = result.get("ejs_id")
                result["snippet"] = (result.get("full_text", "") or "")[:500] + "..."
                return {"total": 1, "items": [result]}
            else:
                return {"total": 0, "items": []}
        except Exception as e:
            print(f"Error searching by EJS ID: {e}")
            # Fall through to normal search if error

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

@app.get("/api/adjudicators")
def get_adjudicators():
    """Get list of all adjudicators with their statistics from ai_adjudicator_extract_v4"""
    try:
        # First get all decisions per adjudicator to calculate median
        query_all = """
        SELECT 
            adjudicator_name,
            claimed_amount,
            adjudicated_amount,
            fee_claimant_proportion,
            fee_respondent_proportion
        FROM ai_adjudicator_extract_v4
        WHERE adjudicator_name IS NOT NULL 
        AND adjudicator_name != ''
        AND TRIM(adjudicator_name) != ''
        """
        
        all_rows = con.execute(query_all).fetchall()
        
        # Group by adjudicator and calculate individual award rates and fee proportions
        adjudicator_rates = {}
        adjudicator_fees = {}
        adjudicator_zero_awards = {}
        
        for row in all_rows:
            name = row["adjudicator_name"]
            
            # Safely convert to float, skipping 'N/A' and other invalid values
            try:
                claimed = float(row["claimed_amount"]) if row["claimed_amount"] not in ('N/A', '', None) else None
                adjudicated = float(row["adjudicated_amount"]) if row["adjudicated_amount"] not in ('N/A', '', None) else None
            except (ValueError, TypeError):
                continue
            
            # Count $0 awards
            if adjudicated is not None and adjudicated == 0 and claimed is not None and claimed > 0:
                if name not in adjudicator_zero_awards:
                    adjudicator_zero_awards[name] = 0
                adjudicator_zero_awards[name] += 1
            
            # Include ALL decisions where we have valid numbers, including $0 awards
            if claimed is not None and adjudicated is not None and claimed > 0:
                rate = min((adjudicated / claimed) * 100, 100.0)
                if name not in adjudicator_rates:
                    adjudicator_rates[name] = []
                adjudicator_rates[name].append(rate)
            
            # Handle fee proportions
            try:
                claimant_fee = float(row["fee_claimant_proportion"]) if row["fee_claimant_proportion"] not in ('N/A', '', None) else None
                respondent_fee = float(row["fee_respondent_proportion"]) if row["fee_respondent_proportion"] not in ('N/A', '', None) else None
                
                if claimant_fee is not None:
                    if name not in adjudicator_fees:
                        adjudicator_fees[name] = {'claimant': [], 'respondent': []}
                    adjudicator_fees[name]['claimant'].append(claimant_fee)
                    if respondent_fee is None or abs((claimant_fee + respondent_fee) - 100) > 0.1:
                        respondent_fee = 100 - claimant_fee
                    adjudicator_fees[name]['respondent'].append(respondent_fee)
            except (ValueError, TypeError):
                continue
        
        # Now get aggregated stats
        query = """
        SELECT 
            adjudicator_name,
            COUNT(*) as total_decisions,
            SUM(CASE 
                WHEN claimed_amount NOT IN ('N/A', '') AND claimed_amount IS NOT NULL 
                THEN CAST(claimed_amount AS REAL) 
                ELSE 0 
            END) as total_claimed,
            SUM(CASE 
                WHEN adjudicated_amount NOT IN ('N/A', '') AND adjudicated_amount IS NOT NULL 
                THEN CAST(adjudicated_amount AS REAL) 
                ELSE 0 
            END) as total_adjudicated
        FROM ai_adjudicator_extract_v4
        WHERE adjudicator_name IS NOT NULL 
        AND adjudicator_name != ''
        AND TRIM(adjudicator_name) != ''
        GROUP BY adjudicator_name
        HAVING COUNT(*) >= 0
        ORDER BY total_decisions DESC
        """
        
        rows = con.execute(query).fetchall()
        
        adjudicators = []
        for row in rows:
            name = row["adjudicator_name"]
            total_claimed = float(row["total_claimed"]) if row["total_claimed"] else 0
            total_adjudicated = float(row["total_adjudicated"]) if row["total_adjudicated"] else 0
            
            # Calculate average (mean) award rate from individual decisions
            avg_award_rate = 0
            if name in adjudicator_rates and adjudicator_rates[name]:
                avg_award_rate = sum(adjudicator_rates[name]) / len(adjudicator_rates[name])
            
            # Calculate average fee proportions
            avg_claimant_fee = 0
            avg_respondent_fee = 0
            if name in adjudicator_fees:
                if adjudicator_fees[name]['claimant']:
                    avg_claimant_fee = sum(adjudicator_fees[name]['claimant']) / len(adjudicator_fees[name]['claimant'])
                if adjudicator_fees[name]['respondent']:
                    avg_respondent_fee = sum(adjudicator_fees[name]['respondent']) / len(adjudicator_fees[name]['respondent'])
            
            adjudicator = {
                "id": name.replace(" ", "_").lower(),
                "name": name,
                "totalDecisions": row["total_decisions"],
                "totalClaimAmount": total_claimed,
                "totalAwardedAmount": total_adjudicated,
                "avgAwardRate": avg_award_rate,
                "avgClaimantFeeProportion": avg_claimant_fee,
                "avgRespondentFeeProportion": avg_respondent_fee,
                "zeroAwardCount": adjudicator_zero_awards.get(name, 0)
            }
            adjudicators.append(adjudicator)
        
        return adjudicators
        
    except Exception as e:
        print(f"Error in /api/adjudicators: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/adjudicator/{adjudicator_name}")
def get_adjudicator_decisions(adjudicator_name: str = Path(...)):
    """Get all decisions for a specific adjudicator from ai_adjudicator_extract_v4"""
    try:
        decoded_name = unquote_plus(adjudicator_name)
        
        query = """
        SELECT 
            a.ejs_id,
            a.adjudicator_name,
            a.claimant_name,
            a.respondent_name,
            a.claimed_amount,
            a.payment_schedule_amount,
            a.adjudicated_amount,
            a.decision_date,
            a.outcome,
            a.project_type,
            a.contract_type,
            a.fee_claimant_proportion,
            a.fee_respondent_proportion,
            d.reference,
            d.pdf_path
        FROM ai_adjudicator_extract_v4 a
        LEFT JOIN docs_fresh d ON a.ejs_id = d.ejs_id
        WHERE LOWER(TRIM(a.adjudicator_name)) = LOWER(TRIM(?))
        ORDER BY a.decision_date DESC
        """
        
        rows = con.execute(query, (decoded_name,)).fetchall()
        
        decisions = []
        for row in rows:
            claimed = 0
            try:
                claimed = float(row["claimed_amount"]) if row["claimed_amount"] not in ('N/A', '', None) else 0
            except (ValueError, TypeError):
                claimed = 0
                
            adjudicated = 0
            try:
                adjudicated = float(row["adjudicated_amount"]) if row["adjudicated_amount"] not in ('N/A', '', None) else 0
            except (ValueError, TypeError):
                adjudicated = 0
            
            # Parse fee proportions
            claimant_fee = 0
            respondent_fee = 0
            try:
                claimant_fee = float(row["fee_claimant_proportion"]) if row["fee_claimant_proportion"] not in ('N/A', '', None) else 0
                respondent_fee = float(row["fee_respondent_proportion"]) if row["fee_respondent_proportion"] not in ('N/A', '', None) else 0
            except (ValueError, TypeError):
                pass
            
            decision = {
                "id": row["ejs_id"],
                "title": f"{row['claimant_name'] or 'Unknown'} v {row['respondent_name'] or 'Unknown'}",
                "reference": row["reference"],
                "date": row["decision_date"],
                "claimant": row["claimant_name"],
                "respondent": row["respondent_name"],
                "claimAmount": claimed,
                "awardedAmount": adjudicated,
                "claimantFeeProportion": claimant_fee,
                "respondentFeeProportion": respondent_fee,
                "outcome": row["outcome"],
                "projectType": row["project_type"],
                "contractType": row["contract_type"],
                "pdfPath": row["pdf_path"]
            }
            decisions.append(decision)
        
        return decisions
        
    except Exception as e:
        print(f"Error in /api/adjudicator/{adjudicator_name}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Add this helper endpoint to debug the database schema
@app.get("/debug/schema")
def debug_schema():
    """Debug endpoint to check database schema"""
    try:
        # Get docs_meta table schema
        meta_columns = con.execute("PRAGMA table_info(docs_meta)").fetchall()
        
        # Get a sample row to see what data looks like
        sample_row = con.execute("SELECT * FROM docs_meta LIMIT 1").fetchone()
        
        return {
            "docs_meta_columns": [{"name": col[1], "type": col[2]} for col in meta_columns],
            "sample_data": dict(sample_row) if sample_row else None
        }
    except Exception as e:
        return {"error": str(e)}

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

def get_human_readable_type(content_type: str, filename: str) -> str:
    """Converts MIME type and filename to a user-friendly object type."""
    ext = filename.split('.')[-1].lower()
    if ext in ['msg', 'eml']:
        return "Email Message"
    if ext == 'pdf':
        return "PDF Document"
    if ext == 'docx':
        return "Word Document"
    if ext == 'doc':
        return "Legacy Word Document"
    if ext in ['png', 'jpg', 'jpeg', 'gif']:
        return "Image"
    return content_type or "Unknown File"

def get_renaming_system_prompt() -> str:
    """Returns the expert system prompt for the AI to rename documents."""
    return """
You are an expert legal discovery AI named LexiFile. Your task is to analyze text from a legal document and extract key information. Your entire response must be a single, valid JSON object.

The JSON object must have keys for:
1.  "date": The primary date (YYYY-MM-DD). Use today's date if none is found.
2.  "docType": The strict document type (e.g., "Email from [Sender's Email] to [Recipient's Email]", "Contract between [Party A] and [Party B]", "Affidavit of [Name]"). For emails, use the actual email addresses.
3.  "description": A brief 2-5 word summary of the subject matter.
4.  "keywords": An array of up to 10 relevant string keywords.
5.  "metadata": An object with the following keys. If info is not present, use an empty string "" or a sensible default.
    - "sender": The sender's email address ONLY if it's an email. Otherwise, this should be an empty string.
    - "recipient": The recipient's email address(es) ONLY if it's an email. Otherwise, this should be an empty string.
    - "author": The document author, extracted from document properties or text.
    - "lastModified": The last modified date (YYYY-MM-DD format).
"""

def extract_lexifile_text(file_path: str, filename: str) -> str:
    """Robustly extracts text from various file types from a file path."""
    text = ""
    lower_filename = filename.lower()
    try:
        if lower_filename.endswith('.pdf'):
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            # OCR Fallback for scanned PDFs
            if len(text.strip()) < 50: 
                print(f"Performing OCR on {filename}...")
                text = pytesseract.image_to_string(file_path)
        elif lower_filename.endswith('.docx'):
            with open(file_path, "rb") as f:
                doc = docx.Document(f)
                core_props = doc.core_properties
                author = core_props.author if core_props.author else ""
                last_modified_by = core_props.last_modified_by if core_props.last_modified_by else ""
                text += f"[Author: {author}]\n[Last Modified By: {last_modified_by}]\n\n"
                for para in doc.paragraphs:
                    text += para.text + "\n"
        elif lower_filename.endswith('.doc'):
            try:
                text = pypandoc.convert_file(file_path, 'plain', format='doc')
            except Exception as pandoc_error:
                print(f"Pandoc failed for {filename}: {pandoc_error}. Falling back.")
                text = "" 
        elif lower_filename.endswith('.msg') or lower_filename.endswith('.eml'):
            msg = extract_msg.Message(file_path)
            text += f"From: {msg.sender}\nTo: {msg.to}\nCC: {msg.cc}\nSubject: {msg.subject}\n\n{msg.body}"
        return text
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return ""

@app.post("/rename-document")
async def rename_document(file: UploadFile = File(...), project_id: str = Form(...)):
    global ARTIFACT_ID_COUNTER
    if not file or not project_id:
        raise HTTPException(status_code=400, detail="Missing file or project ID.")
    
    temp_path = os.path.join(LEXIFILE_STORAGE, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        file_stat = os.stat(temp_path)
        last_modified_date = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d')
        
        extracted_text = extract_lexifile_text(temp_path, file.filename)
        
        ai_response = {}
        if not extracted_text.strip():
            doc_type = get_human_readable_type("", file.filename)
            ai_response = {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "docType": doc_type, "description": "Media file",
                "keywords": ["media", file.filename.split('.')[-1]],
                "metadata": { "sender": "", "recipient": "", "author": "", "lastModified": last_modified_date }
            }
        else:
            system_prompt = get_renaming_system_prompt()
            user_prompt = f"Analyze the following text. Available metadata: Last Modified='{last_modified_date}'.\n\n---TEXT---\n{extracted_text[:8000]}\n---END---"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}
            )
            ai_response = json.loads(response.choices[0].message.content)
            if not ai_response.get("metadata", {}).get("lastModified"):
                if "metadata" not in ai_response: ai_response["metadata"] = {}
                ai_response["metadata"]["lastModified"] = last_modified_date

        artifact_id = f"LEX-{str(ARTIFACT_ID_COUNTER).zfill(8)}"
        
        permanent_path = os.path.join(LEXIFILE_STORAGE, f"{artifact_id}_{file.filename}")
        os.rename(temp_path, permanent_path)

        date_part = ai_response.get('date', '0000-00-00').replace('-', '')
        doc_type_part = ai_response.get('docType', 'Untitled')
        description_part = ai_response.get('description', 'No Description')
        ext = file.filename.split('.')[-1]
        improved_filename = f"{date_part} - {doc_type_part} - {description_part}.{ext}"

        doc_record = {
            "artifactID": artifact_id, "projectID": project_id,
            "objectType": get_human_readable_type(file.content_type, file.filename),
            "originalFilename": file.filename,
            "improvedFilename": improved_filename,
            "date": ai_response.get('date', ''),
            "description": ai_response.get('description', ''),
            "author": ai_response.get('metadata', {}).get('author', ''),
            "lastModifiedOn": ai_response.get('metadata', {}).get('lastModified', ''),
            "sender": ai_response.get('metadata', {}).get('sender', ''),
            "recipient": ai_response.get('metadata', {}).get('recipient', ''),
            "uploadedBy": "demo.user@example.com",
            "keywords": ai_response.get('keywords', []),
            "ai_full_response": ai_response
        }
        
        DOCUMENTS_DB[artifact_id] = doc_record
        ARTIFACT_ID_COUNTER += 1
        
        return doc_record

    except Exception as e:
        print(f"Error in rename-document: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get("/get-project-documents")
async def get_project_documents(project_id: str = Query(...)):
    if not project_id:
        raise HTTPException(status_code=400, detail="Project ID is required.")
    
    if project_id == 'all':
        return list(DOCUMENTS_DB.values())
        
    project_docs = [doc for doc in DOCUMENTS_DB.values() if doc.get("projectID") == project_id]
    return project_docs
    
@app.post("/fulltext-search")
async def fulltext_search(project_id: str = Form(...), query: str = Form(...)):
    if not query:
        return []

    results = []
    docs_to_search = [doc for doc in DOCUMENTS_DB.values() if project_id == 'all' or str(doc.get("projectID")) == project_id]
    
    for doc in docs_to_search:
        file_path = os.path.join(LEXIFILE_STORAGE, f"{doc['artifactID']}_{doc['originalFilename']}")
        if os.path.exists(file_path):
            content = extract_lexifile_text(file_path, doc['originalFilename'])
            if query.lower() in content.lower():
                match_pos = content.lower().find(query.lower())
                start = max(0, match_pos - 25)
                end = min(len(content), match_pos + len(query) + 25)
                snippet = "..." + content[start:end].strip().replace("\n", " ") + "..."
                doc_with_snippet = doc.copy()
                doc_with_snippet["snippet"] = snippet
                results.append(doc_with_snippet)
    return results

@app.get("/download-single-doc/{artifact_id}")
async def download_single_doc(artifact_id: str):
    doc_record = DOCUMENTS_DB.get(artifact_id)
    if not doc_record:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = os.path.join(LEXIFILE_STORAGE, f"{doc_record['artifactID']}_{doc_record['originalFilename']}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    return FileResponse(file_path, filename=doc_record['improvedFilename'])

@app.post("/bulk-download")
async def bulk_download(artifact_ids: List[str] = Body(...)):
    def files_generator():
        z = zipstream.ZipFile(mode='w', compression=zipstream.ZIP_DEFLATED)
        for artifact_id in artifact_ids:
            doc_record = DOCUMENTS_DB.get(artifact_id)
            if doc_record:
                file_path = os.path.join(LEXIFILE_STORAGE, f"{doc_record['artifactID']}_{doc_record['originalFilename']}")
                if os.path.exists(file_path):
                    z.write(file_path, arcname=doc_record['improvedFilename'])
        for chunk in z:
            yield chunk

    return StreamingResponse(files_generator(), media_type="application/zip", headers={
        'Content-Disposition': f'attachment; filename="LexiFile_Bulk_Export_{datetime.now().strftime("%Y%m%d")}.zip"'
    })
    
@app.delete("/delete-document/{artifact_id}")
async def delete_document(artifact_id: str):
    if artifact_id in DOCUMENTS_DB:
        doc_record = DOCUMENTS_DB[artifact_id]
        file_path = os.path.join(LEXIFILE_STORAGE, f"{doc_record['artifactID']}_{doc_record['originalFilename']}")
        if os.path.exists(file_path):
            os.remove(file_path)
        del DOCUMENTS_DB[artifact_id]
        return {"status": "success", "message": "Document deleted"}
    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/preview-email")
async def preview_email(file: UploadFile = File(...)):
    if not file or not (file.filename.lower().endswith('.msg') or file.filename.lower().endswith('.eml')):
        raise HTTPException(status_code=400, detail="Invalid file type for email preview.")
    try:
        file_content = await file.read()
        msg = extract_msg.Message(io.BytesIO(file_content))
        return {
            "from": msg.sender, "to": msg.to, "cc": msg.cc,
            "subject": msg.subject, "date": msg.date,
            "body": msg.body
        }
    except Exception as e:
        print(f"Error parsing email file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse email file.")

# --- Project Management Endpoints ---
@app.get("/projects")
async def get_projects():
    # Add document count to each project
    for project in PROJECTS_DB:
        count = sum(1 for doc in DOCUMENTS_DB.values() if str(doc.get("projectID")) == str(project["id"]))
        project["documentCount"] = count
    return PROJECTS_DB

@app.post("/create-project")
async def create_project(projectName: str = Form(...), clientName: str = Form(...), matterNumber: str = Form(...)):
    project_id = len(PROJECTS_DB) + 1
    new_project = {
        "id": project_id, "name": projectName, "client": clientName,
        "matter": matterNumber, "dateCreated": datetime.now().strftime("%Y-%m-%d")
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

# ---------- serve frontend with clean URLs ----------
@app.get("/{path_name:path}")
async def serve_html_page(path_name: str):
    if not path_name:
        path_name = "index"

    file_path = os.path.join(SITE_DIR, f"{path_name}.html")
    
    if os.path.abspath(file_path).startswith(os.path.abspath(SITE_DIR)) and os.path.exists(file_path) and not os.path.isdir(file_path):
        return FileResponse(file_path)
    
    static_file_path = os.path.join(SITE_DIR, path_name)
    if os.path.abspath(static_file_path).startswith(os.path.abspath(SITE_DIR)) and os.path.exists(static_file_path) and not os.path.isdir(static_file_path):
        return FileResponse(static_file_path)

    # Fallback for SPA-like behavior / non-html pages
    index_path = os.path.join(SITE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)

    raise HTTPException(status_code=404, detail="Not Found")

