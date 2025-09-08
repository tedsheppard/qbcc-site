import os, re, shutil, sqlite3, requests
from fastapi import FastAPI, Query, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from email.message import EmailMessage
import aiosmtplib

# ---------------- setup ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
PDF_ROOT = os.path.join(ROOT, 'pdf')
SITE_DIR = os.path.join(ROOT, "site")

# ensure DB lives in /tmp for faster access
DB_PATH = "/tmp/qbcc.db"
if not os.path.exists(DB_PATH):
    shutil.copy("qbcc.db", DB_PATH)

# create global sqlite connection
con = sqlite3.connect(DB_PATH, check_same_thread=False)
con.row_factory = sqlite3.Row
con.execute("PRAGMA cache_size = -20000")   # ~20MB page cache
con.execute("PRAGMA temp_store = MEMORY")
con.execute("PRAGMA mmap_size = 30000000000")  # 30GB if kernel allows

# meilisearch config
MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:7700")
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "")
MEILI_INDEX = "decisions"

# smtp config (use your App Password via Render env var)
SMTP_USER = "sopal.aus@gmail.com"
SMTP_PASS = os.getenv("SMTP_PASSWORD")

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
    return f'NEAR("{left}" "{right}", {dist})'

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
    # boolean / NEAR → SQLite
    if re.search(r'\b(AND|OR|NOT)\b', q, flags=re.I) or re.search(r'\bw/\d+\b', q, flags=re.I) or re.search(r'\bNEAR/\d+\b', q, flags=re.I):
        nq = preprocess_sqlite_query(q)
        total = con.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH :q", {"q": nq}).fetchone()[0]

        sql = """
          SELECT
            fts.rowid,
            snippet(fts, 0, '<mark>', '</mark>', ' … ', 80) AS snippet,
            bm25(fts) AS score
          FROM fts
          WHERE fts MATCH :q
          ORDER BY score
          LIMIT :limit OFFSET :offset
        """
        rows = con.execute(sql, {"q": nq, "limit": limit, "offset": offset}).fetchall()

        items = []
        for r in rows:
            meta = con.execute("""
              SELECT claimant, respondent, adjudicator, decision_date, decision_date_norm,
                     act, reference, pdf_path
              FROM docs_meta
              LEFT JOIN docs_fresh ON docs_meta.ejs_id = docs_fresh.ejs_id
              WHERE docs_fresh.id = ?
            """, (r["rowid"],)).fetchone()
            d = dict(meta) if meta else {}
            d["id"] = r["rowid"]
            d["snippet"] = r["snippet"]
            items.append(d)

        return {"total": total, "items": items}

    # otherwise → Meili
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
        "cropLength": 40
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

# ---------- feedback email ----------
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
    msg["From"] = f"SOPAL <{SMTP_USER}>"
    msg["To"] = SMTP_USER
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
            username=SMTP_USER,
            password=SMTP_PASS,
        )
        return {"ok": True, "message": "Feedback sent successfully"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- serve frontend ----------
app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")
