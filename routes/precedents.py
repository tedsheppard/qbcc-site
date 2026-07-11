"""Firm Precedent Vault — API routes (/api/precedents/*) + page (/precedents).

Isolated feature: reuses the Sopal v2 JWT (same accounts as app.sopal.com.au)
but stores everything in its own precedents.db + its own GCS prefix. Tenant
isolation rule: firm_id is ALWAYS resolved through a firm_members lookup on
the authenticated email — request bodies and query params never pick the
tenant directly.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from routes.sopal_v2 import _current_user_email
from services.precedents import core
from services.precedents.taxonomy import TAXONOMY, taxonomy_public

ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = ROOT / "site" / "precedents.html"

router = APIRouter(prefix="/api/precedents", tags=["precedents"])
page_router = APIRouter(tags=["precedents-page"])

ALLOWED_EXTS = {".pdf", ".docx"}
MAX_FILE_BYTES = 60 * 1024 * 1024
MAX_FILES_PER_REQUEST = 20
INVITE_DAYS = 14


def _ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "")[:64]


def _member(firm_id: str, email: str) -> sqlite3.Row:
    con = core.get_con()
    row = con.execute(
        "SELECT m.*, f.name AS firm_name, f.status AS firm_status FROM firm_members m "
        "JOIN firms f ON f.id = m.firm_id WHERE m.firm_id = ? AND m.email = ?",
        (firm_id, email),
    ).fetchone()
    if row is None or row["firm_status"] != "active":
        raise HTTPException(status_code=403, detail="You are not a member of this firm.")
    return row


def _admin(firm_id: str, email: str) -> sqlite3.Row:
    row = _member(firm_id, email)
    if row["role"] != "admin":
        raise HTTPException(status_code=403, detail="Firm admin access required.")
    return row


def _doc_or_404(firm_id: str, doc_id: str) -> sqlite3.Row:
    doc = core.get_con().execute(
        "SELECT * FROM documents WHERE id = ? AND firm_id = ?", (doc_id, firm_id)
    ).fetchone()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


def _doc_public(d: sqlite3.Row, section_count: int | None = None) -> dict[str, Any]:
    out = {
        "id": d["id"],
        "filename": d["filename"],
        "title": d["title"] or d["filename"],
        "docType": d["doc_type"],
        "partySide": d["party_side"],
        "act": d["act"],
        "dateHint": d["doc_date_hint"],
        "sizeBytes": d["size_bytes"],
        "pages": d["pages"],
        "status": d["status"],
        "error": d["error"],
        "uploadedBy": d["uploaded_by"],
        "createdAt": d["created_at"],
        "updatedAt": d["updated_at"],
    }
    if section_count is not None:
        out["sectionCount"] = section_count
    return out


# ---------------------------------------------------------------- firms

class FirmCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)


@router.post("/firms")
def create_firm(payload: FirmCreate, request: Request, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    con = core.get_con()
    firm_id = core.new_id()
    now = core.now_iso()
    with core._con_lock:
        con.execute(
            "INSERT INTO firms (id, name, status, created_by, created_at) VALUES (?,?,?,?,?)",
            (firm_id, payload.name.strip(), "active", email, now),
        )
        con.execute(
            "INSERT INTO firm_members (firm_id, email, role, invited_by, created_at) VALUES (?,?,?,?,?)",
            (firm_id, email, "admin", email, now),
        )
        con.commit()
    core.audit(firm_id, email, "firm.create", firm_id, payload.name.strip(), _ip(request))
    return {"id": firm_id, "name": payload.name.strip(), "role": "admin"}


@router.get("/firms")
def my_firms(email: str = Depends(_current_user_email)) -> dict[str, Any]:
    con = core.get_con()
    rows = con.execute(
        "SELECT f.id, f.name, m.role, f.created_at, "
        "(SELECT COUNT(*) FROM documents d WHERE d.firm_id = f.id) AS doc_count, "
        "(SELECT COUNT(*) FROM firm_members mm WHERE mm.firm_id = f.id) AS member_count "
        "FROM firm_members m JOIN firms f ON f.id = m.firm_id "
        "WHERE m.email = ? AND f.status = 'active' ORDER BY f.created_at",
        (email,),
    ).fetchall()
    return {
        "firms": [
            {"id": r["id"], "name": r["name"], "role": r["role"], "docCount": r["doc_count"], "memberCount": r["member_count"]}
            for r in rows
        ]
    }


@router.get("/firms/{firm_id}/members")
def firm_members(firm_id: str, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _member(firm_id, email)
    con = core.get_con()
    members = con.execute(
        "SELECT email, role, created_at FROM firm_members WHERE firm_id = ? ORDER BY created_at", (firm_id,)
    ).fetchall()
    invites = con.execute(
        "SELECT token, email, role, created_at, expires_at FROM firm_invites "
        "WHERE firm_id = ? AND accepted_at IS NULL AND expires_at > ? ORDER BY created_at DESC",
        (firm_id, core.now_iso()),
    ).fetchall()
    return {
        "members": [{"email": m["email"], "role": m["role"], "joinedAt": m["created_at"]} for m in members],
        "pendingInvites": [
            {"token": i["token"], "email": i["email"], "role": i["role"], "expiresAt": i["expires_at"]}
            for i in invites
        ],
    }


class InviteCreate(BaseModel):
    email: str = Field(default="", max_length=254)
    role: str = Field(default="member", pattern="^(admin|member)$")


@router.post("/firms/{firm_id}/invites")
def create_invite(firm_id: str, payload: InviteCreate, request: Request, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _admin(firm_id, email)
    con = core.get_con()
    token = secrets.token_urlsafe(24)
    expires = (datetime.utcnow() + timedelta(days=INVITE_DAYS)).isoformat(timespec="seconds") + "Z"
    with core._con_lock:
        con.execute(
            "INSERT INTO firm_invites (token, firm_id, email, role, invited_by, created_at, expires_at) VALUES (?,?,?,?,?,?,?)",
            (token, firm_id, (payload.email or "").strip().lower(), payload.role, email, core.now_iso(), expires),
        )
        con.commit()
    core.audit(firm_id, email, "invite.create", token, f"{payload.email} as {payload.role}", _ip(request))
    return {"token": token, "url": f"https://app.sopal.com.au/precedents?invite={token}", "expiresAt": expires}


class InviteAccept(BaseModel):
    token: str = Field(min_length=8, max_length=128)


@router.post("/invites/accept")
def accept_invite(payload: InviteAccept, request: Request, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    con = core.get_con()
    inv = con.execute("SELECT * FROM firm_invites WHERE token = ?", (payload.token,)).fetchone()
    if inv is None or inv["accepted_at"] is not None or inv["expires_at"] <= core.now_iso():
        raise HTTPException(status_code=400, detail="Invite link is invalid or has expired.")
    if inv["email"] and inv["email"] != email:
        raise HTTPException(status_code=403, detail="This invite was issued to a different email address.")
    firm = con.execute("SELECT * FROM firms WHERE id = ? AND status = 'active'", (inv["firm_id"],)).fetchone()
    if firm is None:
        raise HTTPException(status_code=400, detail="Firm no longer exists.")
    with core._con_lock:
        con.execute(
            "INSERT OR IGNORE INTO firm_members (firm_id, email, role, invited_by, created_at) VALUES (?,?,?,?,?)",
            (inv["firm_id"], email, inv["role"], inv["invited_by"], core.now_iso()),
        )
        con.execute(
            "UPDATE firm_invites SET accepted_at = ?, accepted_by = ? WHERE token = ?",
            (core.now_iso(), email, payload.token),
        )
        con.commit()
    core.audit(inv["firm_id"], email, "invite.accept", payload.token, "", _ip(request))
    return {"firmId": inv["firm_id"], "firmName": firm["name"], "role": inv["role"]}


@router.delete("/firms/{firm_id}/members/{member_email}")
def remove_member(firm_id: str, member_email: str, request: Request, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _admin(firm_id, email)
    member_email = member_email.strip().lower()
    con = core.get_con()
    target = con.execute(
        "SELECT * FROM firm_members WHERE firm_id = ? AND email = ?", (firm_id, member_email)
    ).fetchone()
    if target is None:
        raise HTTPException(status_code=404, detail="Member not found.")
    if target["role"] == "admin":
        admins = con.execute(
            "SELECT COUNT(*) AS n FROM firm_members WHERE firm_id = ? AND role = 'admin'", (firm_id,)
        ).fetchone()["n"]
        if admins <= 1:
            raise HTTPException(status_code=400, detail="A firm must keep at least one admin.")
    with core._con_lock:
        con.execute("DELETE FROM firm_members WHERE firm_id = ? AND email = ?", (firm_id, member_email))
        con.commit()
    core.audit(firm_id, email, "member.remove", member_email, "", _ip(request))
    return {"removed": member_email}


# ---------------------------------------------------------------- documents

@router.post("/firms/{firm_id}/documents")
async def upload_documents(
    firm_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
    email: str = Depends(_current_user_email),
) -> dict[str, Any]:
    _member(firm_id, email)
    if not files:
        raise HTTPException(status_code=400, detail="No files supplied.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_FILES_PER_REQUEST} files per request.")
    con = core.get_con()
    accepted, skipped = [], []
    for f in files:
        name = os.path.basename(f.filename or "document.pdf")
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTS:
            skipped.append({"filename": name, "reason": "Only PDF and DOCX files are accepted."})
            continue
        data = await f.read()
        if len(data) > MAX_FILE_BYTES:
            skipped.append({"filename": name, "reason": "File is over the 60 MB limit."})
            continue
        if len(data) < 100:
            skipped.append({"filename": name, "reason": "File is empty."})
            continue
        sha = hashlib.sha256(data).hexdigest()
        dupe = con.execute(
            "SELECT id, filename FROM documents WHERE firm_id = ? AND sha256 = ?", (firm_id, sha)
        ).fetchone()
        if dupe:
            skipped.append({"filename": name, "reason": f"Duplicate of {dupe['filename']} — already in the library."})
            continue
        doc_id = core.new_id()
        storage_kind, blob_path = core.save_blob(firm_id, doc_id, name, data)
        now = core.now_iso()
        with core._con_lock:
            con.execute(
                "INSERT INTO documents (id, firm_id, filename, storage, blob_path, size_bytes, sha256, status, uploaded_by, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, firm_id, name, storage_kind, blob_path, len(data), sha, "queued", email, now, now),
            )
            con.commit()
        core.audit(firm_id, email, "doc.upload", doc_id, f"{name} ({len(data)} bytes)", _ip(request))
        accepted.append({"id": doc_id, "filename": name, "status": "queued"})
    return {"accepted": accepted, "skipped": skipped}


@router.get("/firms/{firm_id}/documents")
def list_documents(
    firm_id: str,
    status: str = "",
    category: str = "",
    subcategory: str = "",
    limit: int = 500,
    email: str = Depends(_current_user_email),
) -> dict[str, Any]:
    _member(firm_id, email)
    con = core.get_con()
    limit = max(1, min(limit, 1000))
    params: list[Any] = [firm_id]
    where = "d.firm_id = ?"
    if status:
        where += " AND d.status = ?"
        params.append(status)
    if category:
        where += " AND EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.category = ?" + (
            " AND s.subcategory = ?)" if subcategory else ")"
        )
        params.append(category)
        if subcategory:
            params.append(subcategory)
    params.append(limit)
    rows = con.execute(
        f"SELECT d.*, (SELECT COUNT(*) FROM sections s WHERE s.doc_id = d.id) AS section_count "
        f"FROM documents d WHERE {where} ORDER BY d.created_at DESC LIMIT ?",
        params,
    ).fetchall()
    return {"documents": [_doc_public(r, r["section_count"]) for r in rows]}


@router.get("/firms/{firm_id}/documents/{doc_id}")
def document_detail(firm_id: str, doc_id: str, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _member(firm_id, email)
    doc = _doc_or_404(firm_id, doc_id)
    con = core.get_con()
    sections = con.execute(
        "SELECT * FROM sections WHERE doc_id = ? ORDER BY page_start, id", (doc_id,)
    ).fetchall()
    return {
        "document": _doc_public(doc, len(sections)),
        "sections": [
            {
                "id": s["id"],
                "category": s["category"],
                "categoryLabel": TAXONOMY.get(s["category"], {}).get("label", s["category"]),
                "subcategory": s["subcategory"],
                "subcategoryLabel": TAXONOMY.get(s["category"], {}).get("subs", {}).get(s["subcategory"], s["subcategory"]),
                "heading": s["heading"],
                "summary": s["summary"],
                "pageStart": s["page_start"],
                "pageEnd": s["page_end"],
                "confidence": s["confidence"],
            }
            for s in sections
        ],
    }


@router.get("/firms/{firm_id}/documents/{doc_id}/file")
def document_file(firm_id: str, doc_id: str, request: Request, email: str = Depends(_current_user_email)):
    _member(firm_id, email)
    doc = _doc_or_404(firm_id, doc_id)
    core.audit(firm_id, email, "doc.view", doc_id, doc["filename"], _ip(request))
    url = core.signed_url(doc["storage"], doc["blob_path"], doc["filename"])
    if url:
        return RedirectResponse(url, status_code=307)
    try:
        data = core.read_blob(doc["storage"], doc["blob_path"])
    except Exception:
        raise HTTPException(status_code=404, detail="Stored file is unavailable.")
    media = "application/pdf" if doc["filename"].lower().endswith(".pdf") else (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    import io

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media,
        headers={"Content-Disposition": f'inline; filename="{doc["filename"].replace(chr(34), "")}"'},
    )


@router.delete("/firms/{firm_id}/documents/{doc_id}")
def delete_document(firm_id: str, doc_id: str, request: Request, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _admin(firm_id, email)
    doc = _doc_or_404(firm_id, doc_id)
    core.purge_document(doc)
    core.audit(firm_id, email, "doc.delete", doc_id, doc["filename"], _ip(request))
    return {"deleted": doc_id}


@router.post("/firms/{firm_id}/documents/{doc_id}/reclassify")
def reclassify_document(firm_id: str, doc_id: str, request: Request, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _admin(firm_id, email)
    doc = _doc_or_404(firm_id, doc_id)
    if doc["status"] not in ("ready", "failed", "needs_review"):
        raise HTTPException(status_code=400, detail="Document is still being processed.")
    con = core.get_con()
    new_status = "extracted" if doc["pages"] else "queued"
    with core._con_lock:
        con.execute(
            "UPDATE documents SET status=?, attempts=0, error='', batch_id='', updated_at=? WHERE id=?",
            (new_status, core.now_iso(), doc_id),
        )
        con.commit()
    core.audit(firm_id, email, "doc.reclassify", doc_id, "", _ip(request))
    return {"id": doc_id, "status": new_status}


# ---------------------------------------------------------------- search & categories

def _fts_query(user_q: str) -> str:
    tokens = [t for t in user_q.replace('"', " ").split() if t.strip()][:12]
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)


@router.get("/firms/{firm_id}/search")
def search(
    firm_id: str,
    request: Request,
    q: str = "",
    category: str = "",
    subcategory: str = "",
    limit: int = 40,
    email: str = Depends(_current_user_email),
) -> dict[str, Any]:
    _member(firm_id, email)
    con = core.get_con()
    limit = max(1, min(limit, 100))
    match = _fts_query(q or "")
    if not match:
        raise HTTPException(status_code=400, detail="Enter a search query.")
    if core.FTS_FLAVOUR == "fts5":
        snippet_expr = "snippet(sections_fts, 0, '<mark>', '</mark>', ' … ', 28)"
    else:
        snippet_expr = "snippet(sections_fts, '<mark>', '</mark>', ' … ', 0, 28)"
    sql = (
        f"SELECT f.section_id, f.doc_id, {snippet_expr} AS snip "
        "FROM sections_fts f WHERE f.sections_fts MATCH ? AND f.firm_id = ? LIMIT ?"
    )
    try:
        hits = con.execute(sql, (match, firm_id, limit * 3)).fetchall()
    except sqlite3.OperationalError:
        raise HTTPException(status_code=400, detail="Search query could not be parsed — try plain keywords.")
    results = []
    for h in hits:
        s = con.execute("SELECT * FROM sections WHERE id = ?", (h["section_id"],)).fetchone()
        if s is None or s["firm_id"] != firm_id:
            continue
        if category and s["category"] != category:
            continue
        if subcategory and s["subcategory"] != subcategory:
            continue
        d = con.execute("SELECT id, filename, title, status FROM documents WHERE id = ?", (s["doc_id"],)).fetchone()
        if d is None:
            continue
        results.append(
            {
                "sectionId": s["id"],
                "docId": d["id"],
                "docTitle": d["title"] or d["filename"],
                "filename": d["filename"],
                "category": s["category"],
                "categoryLabel": TAXONOMY.get(s["category"], {}).get("label", s["category"]),
                "subcategory": s["subcategory"],
                "subcategoryLabel": TAXONOMY.get(s["category"], {}).get("subs", {}).get(s["subcategory"], s["subcategory"]),
                "heading": s["heading"],
                "summary": s["summary"],
                "pageStart": s["page_start"],
                "pageEnd": s["page_end"],
                "snippet": h["snip"],
            }
        )
        if len(results) >= limit:
            break
    core.audit(firm_id, email, "search", "", q[:200], _ip(request))
    return {"query": q, "results": results}


@router.get("/firms/{firm_id}/categories")
def categories(firm_id: str, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _member(firm_id, email)
    con = core.get_con()
    rows = con.execute(
        "SELECT category, subcategory, COUNT(*) AS n, COUNT(DISTINCT doc_id) AS docs "
        "FROM sections WHERE firm_id = ? GROUP BY category, subcategory",
        (firm_id,),
    ).fetchall()
    counts: dict[str, Any] = {}
    for r in rows:
        node = counts.setdefault(r["category"], {"sections": 0, "docs": 0, "subs": {}})
        node["sections"] += r["n"]
        node["subs"][r["subcategory"]] = {"sections": r["n"], "docs": r["docs"]}
    tree = []
    for slug, meta in TAXONOMY.items():
        c = counts.get(slug, {"sections": 0, "subs": {}})
        tree.append(
            {
                "slug": slug,
                "label": meta["label"],
                "sections": c["sections"],
                "subs": [
                    {
                        "slug": sub_slug,
                        "label": sub_label,
                        "sections": c["subs"].get(sub_slug, {}).get("sections", 0),
                        "docs": c["subs"].get(sub_slug, {}).get("docs", 0),
                    }
                    for sub_slug, sub_label in meta["subs"].items()
                ],
            }
        )
    return {"tree": tree}


@router.get("/taxonomy")
def get_taxonomy() -> dict[str, Any]:
    return taxonomy_public()


# ---------------------------------------------------------------- admin: audit, usage, offboard

@router.get("/firms/{firm_id}/audit")
def audit_log(firm_id: str, limit: int = 200, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _admin(firm_id, email)
    limit = max(1, min(limit, 1000))
    rows = core.get_con().execute(
        "SELECT email, action, target, detail, ip, created_at FROM audit_log WHERE firm_id = ? ORDER BY id DESC LIMIT ?",
        (firm_id, limit),
    ).fetchall()
    return {
        "events": [
            {"email": r["email"], "action": r["action"], "target": r["target"], "detail": r["detail"], "ip": r["ip"], "at": r["created_at"]}
            for r in rows
        ]
    }


@router.get("/firms/{firm_id}/usage")
def usage(firm_id: str, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _admin(firm_id, email)
    con = core.get_con()
    totals = con.execute(
        "SELECT COALESCE(SUM(input_tokens),0) AS tin, COALESCE(SUM(output_tokens),0) AS tout, "
        "COALESCE(SUM(cost_usd),0) AS cost, COUNT(*) AS runs FROM cost_ledger WHERE firm_id = ?",
        (firm_id,),
    ).fetchone()
    docs = con.execute(
        "SELECT status, COUNT(*) AS n FROM documents WHERE firm_id = ? GROUP BY status", (firm_id,)
    ).fetchall()
    storage = con.execute(
        "SELECT COALESCE(SUM(size_bytes),0) AS b FROM documents WHERE firm_id = ?", (firm_id,)
    ).fetchone()["b"]
    return {
        "inputTokens": totals["tin"],
        "outputTokens": totals["tout"],
        "aiCostUsd": round(totals["cost"], 4),
        "classificationRuns": totals["runs"],
        "storageBytes": storage,
        "documentsByStatus": {r["status"]: r["n"] for r in docs},
    }


class OffboardConfirm(BaseModel):
    confirmName: str = Field(min_length=1, max_length=200)


@router.post("/firms/{firm_id}/offboard")
def offboard(firm_id: str, payload: OffboardConfirm, request: Request, email: str = Depends(_current_user_email)) -> dict[str, Any]:
    _admin(firm_id, email)
    firm = core.get_con().execute("SELECT * FROM firms WHERE id = ?", (firm_id,)).fetchone()
    if firm is None or payload.confirmName.strip() != firm["name"]:
        raise HTTPException(status_code=400, detail="Type the firm name exactly to confirm the purge.")
    core.audit(firm_id, email, "firm.offboard.start", firm_id, firm["name"], _ip(request))
    n = core.purge_firm(firm_id)
    core.audit(firm_id, email, "firm.offboard.done", firm_id, f"purged {n} documents", _ip(request))
    return {"purgedDocuments": n}


# ---------------------------------------------------------------- page

@page_router.get("/precedents", include_in_schema=False)
async def precedents_page() -> FileResponse:
    if not PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Precedents page not found")
    return FileResponse(str(PAGE_PATH), media_type="text/html")
