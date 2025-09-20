import os, re, shutil, sqlite3, requests, unicodedata
from urllib.parse import unquote_plus
from fastapi import FastAPI, Query, Form, Path, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from email.message import EmailMessage
import aiosmtplib
from openai import OpenAI
from google.cloud import storage

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
    
    # Fix Unicode characters using replace instead of translate
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
        # Modified to preserve wildcards (*)
        toks = re.findall(r'\w+\*?', q or "")
        q = ' AND '.join(toks)
    return q

def _fix_unbalanced_quotes(s: str) -> str:
    if s.count('"') % 2 == 1:
        s = s + '"'
    return s

def expand_wildcards(q: str) -> str:
    """Expand ! wildcards to SQLite GLOB patterns"""
    # Replace ! with * for SQLite FTS GLOB matching
    # This handles cases like "exclusiv!" -> "exclusiv*"
    print(f"Before wildcard expansion: {q}")
    expanded = re.sub(r'(\w+)!', r'\1*', q)
    print(f"After wildcard expansion: {expanded}")
    return expanded

def _parse_near_robust(q: str) -> str | None:
    """
    Enhanced NEAR parsing that properly handles quoted phrases, wildcards, and mixed combinations
    """
    s = _fix_unbalanced_quotes(q)
    
    # Check for any w/N or NEAR/N pattern first
    if not re.search(r'\b(?:w|near)\s*/\s*\d+\b', s, flags=re.I):
        return None
    
    # Normalize w/N and NEAR/N to NEAR/N
    s = re.sub(r'\b(?:w|near)\s*/\s*(\d+)\b', r'NEAR/\1', s, flags=re.I)

    # Then find the NEAR pattern
    near_match = re.search(r'NEAR/(\d+)', s, flags=re.I)
    if not near_match:
        return None
    dist = int(near_match.group(1))
    
    # Fallback: try to parse around NEAR
    parts = re.split(r'\s+NEAR/\d+\s+', s, flags=re.I)
    if len(parts) == 2:
        left_part = parts[0].strip()
        right_part = parts[1].strip()

        # Clean up both parts by stripping outer parentheses and quotes
        def clean_term(term):
            term = term.strip()
            # Repeatedly strip parentheses and whitespace
            while term.startswith('(') and term.endswith(')'):
                term = term[1:-1].strip()
            # Finally strip quotes
            if term.startswith('"') and term.endswith('"'):
                term = term[1:-1].strip()
            return term

        left_clean = clean_term(left_part)
        right_clean = clean_term(right_part)
            
        print(f"_parse_near_robust - left: '{left_clean}', dist: {dist}, right: '{right_clean}'")
        return f'"{left_clean}" NEAR/{dist} "{right_clean}"'
    
    return None

def preprocess_sqlite_query(q: str) -> str:
    """Enhanced query preprocessing with proper phrase and operator handling"""
    print(f"preprocess_sqlite_query input: {q}")
    
    q = expand_wildcards(q)
    print(f"After expand_wildcards: {q}")
    
    # Normalize operators
    q = re.sub(r'\band\b', 'AND', q, flags=re.I)
    q = re.sub(r'\bor\b', 'OR', q, flags=re.I)  
    q = re.sub(r'\bnot\b', 'NOT', q, flags=re.I)
    q = re.sub(r'\b(?:w|near)\s*/\s*(\d+)\b', r'NEAR/\1', q, flags=re.I) # Normalize w/N to NEAR/N
    print(f"After operator normalization: {q}")

    # Handle <term> NEAR/d (<term1> OP <term2>) pattern
    near_group_pattern = re.compile(r'(".*?"|\S+)\s+NEAR/(\d+)\s+\((.*?)\)', flags=re.I)
    m = near_group_pattern.search(q)
    
    if m:
        left_term = m.group(1).strip()
        dist = m.group(2)
        right_group = m.group(3).strip()
        
        # Determine the operator inside the group and split
        operator = None
        terms = []
        # Check for AND first as it can be part of a phrase, then OR
        if ' AND ' in right_group:
            operator = ' AND '
            terms = re.split(r'\s+AND\s+', right_group, flags=re.I)
        elif ' OR ' in right_group:
            operator = ' OR '
            terms = re.split(r'\s+OR\s+', right_group, flags=re.I)
        
        if operator and len(terms) > 1:
            expanded_clauses = []
            for term in terms:
                # Ensure each part is a clean, singly-quoted phrase
                clean_left = left_term.strip().strip('"')
                clean_right = term.strip().strip('"')
                
                # FTS queries need phrases to be double-quoted
                expanded_clauses.append(f'("{clean_left}" NEAR/{dist} "{clean_right}")')
            
            final_query = operator.join(expanded_clauses)
            print(f"Expanded NEAR/{operator.strip()} query to: {final_query}")
            return final_query

    # Handle simple proximity operators if the complex pattern didn't match
    near_expr = _parse_near_robust(q)
    if near_expr:
        print(f"Found NEAR expression: {near_expr}")
        return near_expr

    # If it contains other Boolean operators, preserve the structure
    if re.search(r'\b(AND|OR|NOT)\b', q, flags=re.I):
        print(f"Contains boolean operators, returning: {q.strip()}")
        return q.strip()

    # Otherwise, default to ANDing all terms
    result = normalize_default(q)
    print(f"normalize_default result: {result}")
    return result

def get_highlight_terms(query: str) -> tuple[list[str], list[str]]:
    """
    Extract terms for highlighting, separating phrases from individual words
    Returns: (phrase_terms, word_terms)
    """
    phrase_terms = []
    word_terms = []
    
    # Extract quoted phrases first
    phrases = re.findall(r'"([^"]+)"', query)
    for phrase in phrases:
        phrase_terms.append(phrase)
    
    # Remove quoted content and operators to find individual words
    query_no_quotes = re.sub(r'"[^"]*"', '', query)
    words = re.findall(r'\b\w+\*?\b', query_no_quotes)  # Include wildcards
    
    # Filter out operators and numbers
    operators = {'AND', 'OR', 'NOT', 'NEAR', 'W'}
    for word in words:
        if (word.upper() not in operators and 
            not word.isdigit() and 
            not re.match(r'\d+', word)):
            word_terms.append(word)
    
    print(f"get_highlight_terms - phrase_terms: {phrase_terms}, word_terms: {word_terms}")
    return phrase_terms, word_terms

# -------- phrase-aware proximity filtering (true phrase NEAR) --------
_word_re = re.compile(r"\w+", flags=re.UNICODE)

def _tokenize(text: str):
    # returns list of lowercase tokens
    return _word_re.findall(text.lower())

def _phrase_positions(tokens: list[str], phrase: str) -> list[int]:
    """Return start indices where the exact multi-word phrase occurs (token-wise)."""
    words = _word_re.findall(phrase.lower())
    if not words:
        return []
    if len(words) == 1:
        # single-word 'phrase'
        w = words[0]
        return [i for i, t in enumerate(tokens) if t == w]
    L = len(words)
    out = []
    for i in range(0, len(tokens) - L + 1):
        if tokens[i:i+L] == words:
            out.append(i)
    return out

def _extract_near_components(nq2: str):
    """From canonical '"LEFT" NEAR/N "RIGHT"' return (left, n, right)."""
    m = re.search(r'"([^"]+)"\s+NEAR/(\d+)\s+"([^"]+)"', nq2, flags=re.I)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)

def _filter_rows_for_true_phrase_near(rows, nq2):
    """
    If NEAR and either side is a multi-word phrase, require the exact phrase
    to be within N tokens of the other side. Otherwise return rows unchanged.
    """
    comp = _extract_near_components(nq2)
    if not comp:
        return rows
    left, dist, right = comp
    left_is_phrase  = ' ' in left.strip()
    right_is_phrase = ' ' in right.strip()
    if not (left_is_phrase or right_is_phrase):
        return rows  # both single words → SQLite NEAR already correct

    filtered = []
    # collect rowids to fetch full_text in one go
    rowids = [r["rowid"] for r in rows]
    if not rowids:
        return rows

    # Batch pull full_texts
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
        # define the representative index for a phrase as its first token
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
    """Highlight all words that start with the given stem"""
    pattern = f'\\b({re.escape(stem)}\\w*)'
    return re.sub(pattern, r'<mark>\1</mark>', text, flags=re.I)

# ---------------- routes ----------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search_fast")
def search_fast(q: str = "", limit: int = 20, offset: int = 0, sort: str = "newest"):
    q_norm = normalize_query(q)
    nq = q_norm  # Use normalized query as-is

    # Check if this is a complex query requiring FTS processing
    # First expand wildcards to check for them properly
    nq_expanded = expand_wildcards(nq)
    
    is_complex_query = (
        re.search(r'\b(AND|OR|NOT)\b', nq, flags=re.I) or
        re.search(r'\bw/\d+\b', nq, flags=re.I) or
        re.search(r'\bNEAR/\d+\b', nq, flags=re.I) or
        nq.startswith('"') or
        '!' in nq or  # wildcard queries
        '*' in nq_expanded or  # expanded wildcards  
        nq != nq_expanded  # any transformation happened
    )

    if is_complex_query:
        try:
            nq2 = preprocess_sqlite_query(nq)
            print("Executing MATCH with:", nq2)
            
            total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH ?", (nq2,)).fetchone()[0]

            # Build the ORDER BY clause
            order_clause = ""
            if sort == "newest":
                order_clause = "ORDER BY m.decision_date_norm DESC"
            elif sort == "oldest":
                order_clause = "ORDER BY m.decision_date_norm ASC"
            elif sort == "atoz":
                order_clause = "ORDER BY m.claimant ASC"
            elif sort == "ztoa":
                order_clause = "ORDER BY m.claimant DESC"

            # The query needs a join for sorting, but not for relevance
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
            else: # Relevance search (default), no join needed for the main query
                sql = """
                  SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
                  FROM fts
                  WHERE fts MATCH ?
                  LIMIT ? OFFSET ?
                """
            
            rows = con.execute(sql, (nq2, limit, offset)).fetchall()

        except sqlite3.OperationalError as e:
            print("FTS MATCH error for:", nq, "->", e)
            # Fallback handling for malformed queries
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
                        # If repair fails, degrade further
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
                # Remove wildcards for fallback
                fallback = re.sub(r'[!*]', '', nq)
                total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH ?", (fallback,)).fetchone()[0]
                rows = con.execute("""
                  SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
                  FROM fts
                  WHERE fts MATCH ?
                  LIMIT ? OFFSET ?
                """, (fallback, limit, offset)).fetchall()
                nq2 = fallback

        # ---- TRUE PHRASE PROXIMITY FILTER (only when needed) ----
        # This custom filter ensures multi-word phrases in NEAR queries are exact.
        # It should only run on simple NEAR queries, not ones combined with AND/OR,
        # as the current filter logic doesn't support boolean combinations.
        is_simple_near_query = "NEAR/" in nq2 and " AND " not in nq2 and " OR " not in nq2

        if is_simple_near_query:
            comp = _extract_near_components(nq2)
            if comp:
                left, _, right = comp
                # Check if either side of the NEAR operator is a multi-word phrase
                if (' ' in left.strip()) or (' ' in right.strip()):
                    print(f"Applying true-phrase proximity filter for simple NEAR query.")
                    before = len(rows)
                    rows = _filter_rows_for_true_phrase_near(rows, nq2)
                    # Note: The 'total' count is not updated post-filtering to avoid
                    # performance issues, so it may be a slight overestimate.
                    print(f"Phrase-proximity filtered page results: {before} → {len(rows)}")

        # ---- build items & enhanced highlighting ----
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

            # Enhanced phrase-aware highlighting
            # Highlight phrases first (longer matches take precedence)
            for phrase in sorted(set(phrase_terms), key=len, reverse=True):
                pattern = re.escape(phrase)
                snippet_clean = re.sub(f'(?i)\\b{pattern}\\b', f"<mark>{phrase}</mark>", snippet_clean)

            # Then highlight individual words (excluding those already in phrases)
            for term in sorted(set(word_terms), key=len, reverse=True):
                # Skip if this term is part of an already highlighted phrase
                if any(term.lower() in phrase.lower() for phrase in phrase_terms):
                    continue
                
                # Handle wildcard terms - highlight any word starting with the stem
                if term.endswith('*'):
                    stem = term[:-1]
                    pattern = f'\\b({re.escape(stem)}\\w*)'
                    snippet_clean = re.sub(pattern, r'<mark>\1</mark>', snippet_clean, flags=re.I)
                else:
                    pattern = re.escape(term)
                    snippet_clean = re.sub(f'(?i)\\b({pattern})\\b', r'<mark>\1</mark>', snippet_clean)

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
    
    # If the search is for relevance but the query is empty (initial page load), 
    # default to newest instead.
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
            model="gpt-4o-mini",
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
- The factual disputes and evidence
- The adjudicator's reasoning
- The final outcome, amount awarded, and fee split
Don't actually title it "Summary" or the like please, go straight into bullet points

Decision text:
{text}
"""
                }
            ],
            max_tokens=800,
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
def ask_ai(decision_id: str = Path(...), question: str = Form(...)):
    try:
        row = con.execute("SELECT full_text FROM docs_fresh WHERE ejs_id = ?", (decision_id,)).fetchone()
        if not row or not row["full_text"]:
            raise HTTPException(status_code=404, detail="Decision not found")

        text = row["full_text"][:15000]

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
                {"role": "user", "content": f"Decision text:\n{text}"},
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

