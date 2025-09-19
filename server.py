import os, re, shutil, sqlite3, requests, unicodedata
from urllib.parse import unquote_plus
from fastapi import FastAPI, Query, Form, Path, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from email.message import EmailMessage
import aiosmtplib
from openai import OpenAI

# ---------------- setup ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SITE_DIR = os.path.join(ROOT, "site")

# ensure DB lives in /tmp for faster access
DB_PATH = "/tmp/qbcc.db"
if not os.path.exists(DB_PATH):
    shutil.copy("qbcc.db", DB_PATH)

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
    s = s.translate(str.maketrans({
        '“':'"', '”':'"', '‘':"'", '’':"'", '—':'-', '–':'-', '‐':'-'
    }))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def escape_fts_phrase(s: str) -> str:
    return s.replace('"', '""')

def normalize_default(q: str) -> str:
    if not re.search(r'"|\bNEAR/\d+\b|\bw/\d+\b|\bAND\b|\bOR\b|\bNOT\b|\(|\)', q or "", flags=re.I):
        toks = re.findall(r'\w+', q or "")
        q = ' AND '.join(toks)
    return q

def _fix_unbalanced_quotes(s: str) -> str:
    if s.count('"') % 2 == 1:
        s = s + '"'
    return s

def _parse_near_robust(q: str) -> str | None:
    s = _fix_unbalanced_quotes(q)
    s = re.sub(r'\b(?:w|near)\s*/\s*(\d+)\b', r'NEAR/\1', s, flags=re.I)

    m = re.search(r'("([^"]+)"|(\w+))\s+NEAR/(\d+)\s+("([^"]+)"|(\w+))', s, flags=re.I)
    if not m:
        s2 = s.replace('"', '')
        m = re.search(r'(\w+)\s+NEAR/(\d+)\s+(\w+)', s2, flags=re.I)
        if not m:
            return None
        left, dist, right = m.group(1), int(m.group(2)), m.group(3)
        return f'"{left}" NEAR/{dist} "{right}"'

    left  = m.group(2) or m.group(3)
    dist  = int(m.group(4))
    right = m.group(6) or m.group(7)
    return f'"{left}" NEAR/{dist} "{right}"'

def preprocess_sqlite_query(q: str) -> str:
    q = re.sub(r'\band\b', 'AND', q, flags=re.I)
    q = re.sub(r'\bor\b',  'OR',  q, flags=re.I)
    q = re.sub(r'\bnot\b', 'NOT', q, flags=re.I)

    near_expr = _parse_near_robust(q)
    if near_expr:
        return near_expr

    if re.search(r'\b(AND|OR|NOT)\b', q, flags=re.I):
        return q.strip()

    return normalize_default(q)

# ---------------- routes ----------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search_fast")
def search_fast(q: str = "", limit: int = 20, offset: int = 0, sort: str = "relevance"):
    q_norm = normalize_query(q)

    if len(q_norm) >= 2 and q_norm[0] == '"' and q_norm[-1] == '"':
        inner = q_norm[1:-1]
        nq = f"\"{escape_fts_phrase(inner)}\""
    else:
        nq = q_norm

    if (re.search(r'\b(AND|OR|NOT)\b', nq, flags=re.I)
        or re.search(r'\bw/\d+\b', nq, flags=re.I)
        or re.search(r'\bNEAR/\d+\b', nq, flags=re.I)
        or nq.startswith('"')):

        try:
            nq2 = preprocess_sqlite_query(nq)
            print("Executing MATCH with:", nq2)
            total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH :q", {"q": nq2}).fetchone()[0]
            sql = """
              SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
              FROM fts
              WHERE fts MATCH :q
              LIMIT :limit OFFSET :offset
            """
            rows = con.execute(sql, {"q": nq2, "limit": limit, "offset": offset}).fetchall()

        except sqlite3.OperationalError as e:
            print("FTS MATCH error for:", nq, "->", e)
            return {"total": 0, "items": []}

        items = []
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

            # --- snippet cleaning ---
            snippet_raw = r["snippet"]
            snippet_clean = re.sub(r'\b0+([A-Za-z])', r'\1', snippet_raw)  # remove bogus 0s

            # --- highlighting ---
            raw_terms = re.findall(r'"([^"]+)"|\w+', nq2)
            terms = []
            for t in raw_terms:
                if isinstance(t, tuple):
                    t = [x for x in t if x]
                    if t:
                        terms.append(t[0])
                elif t and t.upper() not in {"W","NEAR","AND","OR","NOT"}:
                    terms.append(t)

            terms = sorted(set(terms), key=lambda x: -len(x.split()))

            for term in terms:
                if " " in term:  # phrase
                    snippet_clean = re.sub(
                        fr'(?i){re.escape(term)}',
                        lambda m: f"<mark>{m.group(0)}</mark>",
                        snippet_clean,
                    )
                else:  # single word
                    snippet_clean = re.sub(
                        fr'(?i)\b({re.escape(term)})\b',
                        r'<mark>\1</mark>',
                        snippet_clean,
                    )

            d["snippet"] = snippet_clean
            items.append(d)

        return {"total": total, "items": items}

    # --- fallback to Meili natural language search ---
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
                        "Structure the summary into bullet points or short sections with HTML-friendly formatting."
                    )
                },
                {
                    "role": "user",
                    "content": f"Decision text:\n{text}"
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
                {"role": "system", "content": "You are SopalAI, assisting users with adjudication decisions."},
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

