import os, re, shutil, sqlite3, requests
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
def search_fast(
    q: str = None,
    claimant: str = None,
    respondent: str = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "newest"  # default newest
):
    search_params = {
        "limit": limit,
        "offset": offset
    }

    # ✅ Sorting rules
    if sort == "newest":
        search_params["sort"] = ["id:desc"]   # highest EJS first
    elif sort == "oldest":
        search_params["sort"] = ["id:asc"]    # lowest EJS first
    elif sort == "atoz":
        search_params["sort"] = ["claimant:asc"]
    elif sort == "ztoa":
        search_params["sort"] = ["claimant:desc"]

    filter_clauses = []
    if claimant:
        filter_clauses.append(f'claimant = "{claimant}"')
    if respondent:
        filter_clauses.append(f'respondent = "{respondent}"')
    if filter_clauses:
        search_params["filter"] = " AND ".join(filter_clauses)

    # Build query
    query = q or ""

    resp = requests.post(
        f"{MEILI_URL}/indexes/decisions/search",
        headers={"Authorization": f"Bearer {MEILI_MASTER_KEY}"},
        json={"q": query, **search_params}
    )

    return resp.json()


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
