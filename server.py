import os, sqlite3, re, requests
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, 'qbcc.db')
PDF_ROOT = os.path.join(ROOT, 'pdf')
SITE_DIR = os.path.join(ROOT, "site")

MEILI_URL = "http://127.0.0.1:7700"
MEILI_INDEX = "decisions"
MEILI_KEY = "sopal123"   # your master key

app = FastAPI()

# ---------- helpers ----------
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

# ---------- routes ----------
@app.get("/health")
def health():
    return {"ok": True}

# ---------- fast hybrid search ----------
@app.get("/search_fast")
def search_fast(q: str = "", limit: int = 20, offset: int = 0, sort: str = "relevance"):
    # --- Detect boolean/NEAR queries → SQLite ---
    # --- Detect boolean/NEAR queries → SQLite ---
if re.search(r'\b(AND|OR|NOT)\b', q, flags=re.I) or re.search(r'\bw/\d+\b', q, flags=re.I) or re.search(r'\bNEAR/\d+\b', q, flags=re.I):
    # force SQLite path
        con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
        try:
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
        finally:
            con.close()

    # --- Otherwise → Meilisearch ---
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
        payload["sort"] = ["date:desc"]
    elif sort == "oldest":
        payload["sort"] = ["date:asc"]
    elif sort == "atoz":
        payload["sort"] = ["claimant:asc"]
    elif sort == "ztoa":
        payload["sort"] = ["claimant:desc"]

    import os
    
    # at the top of server.py, after imports:
    MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:7700")
    MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "")
    MEILI_INDEX = "decisions"
    
    # inside your /search_fast route, replace with this:
    headers = {"Authorization": f"Bearer {MEILI_KEY}"} if MEILI_KEY else {}
    
    res = requests.post(
        f"{MEILI_URL}/indexes/{MEILI_INDEX}/search",
        headers=headers,
        json=payload
    )
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

# ---------- PDF serving ----------
def _resolve_pdf_from_any(path_str: str) -> str:
    if not path_str: raise ValueError("empty path")
    m = re.search(r'(pdf/.*?\.pdf)$', path_str, flags=re.I)
    rel = m.group(1) if m else path_str
    if rel.lower().startswith('pdf/'): rel = rel[4:]
    abs_path = os.path.normpath(os.path.join(PDF_ROOT, rel))
    if not abs_path.startswith(os.path.abspath(PDF_ROOT)):
        raise ValueError("invalid path traversal")
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(abs_path)
    return abs_path

@app.get("/open")
def open_pdf(p: str, disposition: str = "inline"):
    try:
        fpath = _resolve_pdf_from_any(p)
        filename = os.path.basename(fpath)
        headers = {"Content-Disposition": f'{disposition}; filename="{filename}"'}
        return FileResponse(fpath, media_type="application/pdf", headers=headers)
    except FileNotFoundError as e:
        return JSONResponse({"error": "not found", "resolved": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/pdf/{tail:path}")
def pdf_tail(tail: str, disposition: str = "inline"):
    try:
        fpath = _resolve_pdf_from_any("pdf/" + tail)
        filename = os.path.basename(fpath)
        headers = {"Content-Disposition": f'{disposition}; filename="{filename}"'}
        return FileResponse(fpath, media_type="application/pdf", headers=headers)
    except FileNotFoundError as e:
        return JSONResponse({"error": "not found", "resolved": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ---------- serve frontend ----------
app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")
