import os, re, shutil, sqlite3, requests
from fastapi import FastAPI, Query, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from email.message import EmailMessage
import aiosmtplib
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
def normalize_default(q: str) -> str:
    if not re.search(r'"|\bNEAR/\d+\b|\bw/\d+\b|\bAND\b|\bOR\b|\bNOT\b|\(|\)', q or "", flags=re.I):
        toks = re.findall(r'\w+', q or "")
        q = ' AND '.join(toks)
    return q

def parse_near(q: str):
    m = re.search(r'(".*?"|\w+)\s+(?:w|NEAR)\s*/\s*(\d+)\s+(".*?"|\w+)', q or "", flags=re.I)
    if not m:
        return None
    left, dist, right = m.group(1), int(m.group(2)), m.group(3)
    if left.startswith('"') and left.endswith('"'):
        left = left[1:-1]
    if right.startswith('"') and right.endswith('"'):
        right = right[1:-1]
    return f'"{left}" NEAR/{dist} "{right}"'

def preprocess_sqlite_query(q: str) -> str:
    near_expr = parse_near(q)
    if near_expr:
        return near_expr
    q = re.sub(r'\band\b', 'AND', q, flags=re.I)
    q = re.sub(r'\bor\b', 'OR', q, flags=re.I)
    q = re.sub(r'\bnot\b', 'NOT', q, flags=re.I)
    return normalize_default(q)

# ---------------- routes ----------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search_fast")
def search_fast(q: str = "", limit: int = 20, offset: int = 0, sort: str = "relevance"):
    # --- Boolean / proximity handled by SQLite ---
    if re.search(r'\b(AND|OR|NOT)\b', q, flags=re.I) or re.search(r'\bw/\d+\b', q, flags=re.I) or re.search(r'\bNEAR/\d+\b', q, flags=re.I):
        nq = preprocess_sqlite_query(q)

        total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH :q", {"q": nq}).fetchone()[0]
        sql = """
          SELECT fts.rowid, snippet(fts, 0, '', '', ' … ', 100) AS snippet
          FROM fts
          WHERE fts MATCH :q
          LIMIT :limit OFFSET :offset
        """
        rows = con.execute(sql, {"q": nq, "limit": limit, "offset": offset}).fetchall()

        items = []
        for r in rows:
            # fetch metadata
            meta = con.execute("""
              SELECT m.claimant, m.respondent, m.adjudicator, m.decision_date_norm,
                     m.act, d.reference, d.pdf_path
              FROM docs_fresh d
              LEFT JOIN docs_meta m ON d.ejs_id = m.ejs_id
              WHERE d.rowid = ?
            """, (r["rowid"],)).fetchone()

            d = dict(meta) if meta else {}
            d["id"] = r["rowid"]

            # ---------------- clean snippet ----------------
            snippet_raw = r["snippet"]

            # remove weird "0word" artifacts from FTS output
            snippet_clean = re.sub(r'\b\d+([A-Za-z]+)\b', r'\1', snippet_raw)

            # highlight search terms manually
            raw_terms = re.findall(r'\w+', q)
            terms = [
                t for t in raw_terms
                if not re.fullmatch(r'\d+', t) and t.upper() not in {"W", "NEAR", "AND", "OR", "NOT"}
            ]
            for term in terms:
                snippet_clean = re.sub(
                    fr'(?i)\b({re.escape(term)})\b',
                    r'<mark>\1</mark>',
                    snippet_clean
                )

            d["snippet"] = snippet_clean
            items.append(d)

        return {"total": total, "items": items}

    # --- Natural language via Meili ---
    payload = {
        "q": q,
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

# ---------- PDF links via Google Cloud ----------
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

@app.get("/summarise/{decision_id}")
def summarise(decision_id: str):
    # check cache first
    row = con.execute("SELECT summary FROM ai_summaries WHERE decision_id = ?", (decision_id,)).fetchone()
    if row:
        return {"id": decision_id, "summary": row["summary"]}

    # fetch full text
    r = con.execute("SELECT full_text FROM docs_fresh WHERE ejs_id = ?", (decision_id,)).fetchone()
    if not r:
        return JSONResponse({"error": "Decision not found"}, status_code=404)
    full_text = r["full_text"]

    # send to OpenAI
    prompt = f"""Summarise this adjudication decision in 5 bullet points:
- Parties
- Payment claim amount and defence
- Main issues
- Adjudicator’s reasoning
- Outcome

Decision text:
{full_text[:300000]}"""  # safeguard: truncate to ~30k chars

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",   # cheapest 128k context
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        summary = resp.choices[0].message.content.strip()

        # save cache
        con.execute("INSERT OR REPLACE INTO ai_summaries(decision_id, summary) VALUES(?, ?)", (decision_id, summary))
        con.commit()

        return {"id": decision_id, "summary": summary}
    except Exception as e:
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

# ---------- serve frontend ----------
app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")
