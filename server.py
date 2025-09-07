import os, re, shutil, sqlite3, requests
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 🔹 Add this import
from openai import OpenAI

# 🔹 Create global client here
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
            snippet(fts, 0, '<mark>', '</mark>', ' … ', 50) AS snippet,
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
        payload["sort"] = ["id:desc"]   # highest EJS first
    elif sort == "oldest":
        payload["sort"] = ["id:asc"]    # lowest EJS first
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

@app.post("/rewrite_query")
async def rewrite_query(req: Request):
    data = await req.json()
    query = data.get("query", "").strip()
    if not query:
        return JSONResponse({"rewritten": ""})

    # Skip rewriting if query looks boolean-style
    if any(op in query.upper() for op in [" AND ", " OR ", " W/"]) or '"' in query or "(" in query or ")" in query:
        return JSONResponse({"rewritten": query})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You rewrite verbose natural-language legal search questions into concise boolean/keyword queries."},
                {"role": "user", "content": query}
            ],
            max_tokens=50,
            temperature=0
        )
        rewritten = resp.choices[0].message.content.strip()
        print(f"[GPT Rewrite] Original: {query} → Rewritten: {rewritten}")
        return JSONResponse({"rewritten": rewritten})
    except Exception as e:
        print(f"[GPT Rewrite ERROR] {e}")
        return JSONResponse({"rewritten": query, "error": str(e)})

# ---------- PDF links via Google Cloud ----------
GCS_BUCKET = "sopal-bucket"
GCS_PREFIX = "pdfs"

def build_gcs_url(file_name: str) -> str:
    return f"https://storage.googleapis.com/{GCS_BUCKET}/{GCS_PREFIX}/{file_name}"

@app.get("/open")
def open_pdf(p: str, disposition: str = "inline"):
    try:
        # Just take the filename part of whatever is stored in pdf_path
        file_name = os.path.basename(p)
        return {"url": build_gcs_url(file_name)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

import openai
import os
from fastapi import Body

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/rewrite_query")
async def rewrite_query(data: dict = Body(...)):
    q = data.get("q", "").strip()
    if not q:
        return {"query": ""}

    # Simple heuristic: detect if it's already a boolean query
    if any(op in q.upper() for op in [" AND ", " OR ", " W/"]) or '"' in q or "(" in q or ")":
        return {"query": q}  # return unchanged

    # Otherwise, call OpenAI to rewrite into structured query
    prompt = f"""
You are a query rewriting assistant for a legal adjudication decision search engine.

TASK:
Rewrite the following natural-language query into a boolean-style search query with expansions for legal synonyms and variants.

RULES:
- Always output ONLY the query string, no commentary.
- Wrap ALL terms (even single words) in quotes unless they are operators (AND, OR, w/N).
- Expand legal/statutory terms into OR clusters of common variants:

  • Statutory sections:
    s 71 → ("s 71" OR "s71" OR "s.71" OR "s. 71" OR "section 71")

  • BIF Act:
    ("BIF Act" OR "Building Industry Fairness (Security of Payment) Act 2017" 
     OR "BIFA" OR "Security of Payment Act" OR "SOPA" OR "SOP Act" OR "BIFSOPA")

  • Nil:
    ("nil" OR "0" OR "zero")

  • Take out right:
    ("take out right" OR "take out")

  • Extension of time:
    ("EOT" OR "extension of time")

  • Time at large:
    ("time at large" OR "time became at large")

  • Reference date:
    ("reference date" OR "ref date" OR "claim reference date")

- Preserve statutory references exactly as they appear, but also generate alternatives in OR form.
- Use AND between essential concepts.
- Use OR only inside synonym/variant groups.
- Use w/N proximity if the query implies a relationship between terms 
  (e.g. "payment claim valued nil" → ("payment claim" w/5 ("nil" OR "0" OR "zero"))).
- Default N = 5 unless the query specifies a number (e.g. "within 10 days" → w/10).
- Drop filler words like: what, is, are, any, cases, about, because, of, etc.
- Do not invent synonyms — stick strictly to the expansions listed.
- Keep result concise, structured, and highly discriminating.

EXAMPLES:

Input: "are there any cases where a payment claim was valued nil under s 71 because of the exercise of a take out right?"
Output: ("payment claim" w/5 ("nil" OR "0" OR "zero")) AND ("s 71" OR "s71" OR "s.71" OR "s. 71" OR "section 71") AND ("take out right" OR "take out")

Input: "how is a reference date defined under the BIF Act?"
Output: ("reference date" OR "ref date" OR "claim reference date") AND defined AND ("BIF Act" OR "Building Industry Fairness (Security of Payment) Act 2017" OR "BIFA" OR "Security of Payment Act" OR "SOPA" OR "SOP Act" OR "BIFSOPA")

Input: "cases about time at large when an EOT is wrongly refused"
Output: ("time at large" OR "time became at large") AND ("EOT" OR "extension of time") AND (refused OR denial OR rejected)

Input: "was a payment schedule served within 5 business days"
Output: ("payment schedule" AND served) AND ("5 business days" OR ("business days" w/5 "5"))

Query: {q}
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal search query assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0
        )
        rewritten = resp["choices"][0]["message"]["content"].strip()
        print(f"[GPT Rewrite] Original: {q} → Rewritten: {rewritten}")
        return {"query": rewritten}
    except Exception as e:
        print(f"[GPT Rewrite ERROR] {e}")
        return {"query": q, "error": str(e)}
        

# ---------- serve frontend ----------
app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")
