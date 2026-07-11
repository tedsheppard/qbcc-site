"""Firm Precedent Vault — core: database, encryption, blob storage, audit.

Design constraints (deliberate):
- Everything lives in its OWN sqlite database (precedents.db) and its own
  GCS prefix. No existing table, file, or bucket object is touched.
- Tenant isolation is enforced at the query layer: every read/write is
  scoped by firm_id, and firm_id always comes from a membership lookup on
  the authenticated email — never from the client.
- Extracted document text is encrypted at rest (Fernet/AES-128-CBC+HMAC).
  The one necessary plaintext surface is the FTS index over section text,
  which is what makes search work; the original page text and raw model
  output are ciphertext in the DB file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent

_USE_PERSISTENT = os.path.isdir("/var/data") and os.access("/var/data", os.W_OK)
DB_PATH = (
    "/var/data/precedents.db" if _USE_PERSISTENT else str(ROOT / "_local_data" / "precedents.db")
)
LOCAL_BLOB_DIR = (
    "/var/data/precedents_blobs" if _USE_PERSISTENT else str(ROOT / "_local_data" / "precedents_blobs")
)

GCS_BUCKET = os.getenv("PRECEDENTS_GCS_BUCKET") or os.getenv("GCS_BUCKET_NAME", "sopal-bucket")
GCS_PREFIX = os.getenv("PRECEDENTS_GCS_PREFIX", "precedents/")

# ---------------------------------------------------------------- encryption

def _fernet():
    """Fernet keyed from PRECEDENTS_ENC_KEY, else derived from the server's
    existing JWT secret so the feature works on the current Render env with
    zero new env vars. Setting PRECEDENTS_ENC_KEY later is supported: newly
    written rows use it, and _fernet_all() lets reads try both keys."""
    from cryptography.fernet import Fernet

    explicit = (os.getenv("PRECEDENTS_ENC_KEY") or "").strip()
    if explicit:
        return Fernet(explicit.encode("utf-8"))
    secret = os.getenv("LEXIFILE_SECRET_KEY", "dev-secret-key")
    derived = base64.urlsafe_b64encode(hashlib.sha256(f"precedents-vault:{secret}".encode()).digest())
    return Fernet(derived)


def _fernet_all():
    from cryptography.fernet import Fernet, MultiFernet

    fernets = [_fernet()]
    secret = os.getenv("LEXIFILE_SECRET_KEY", "dev-secret-key")
    derived = base64.urlsafe_b64encode(hashlib.sha256(f"precedents-vault:{secret}".encode()).digest())
    if (os.getenv("PRECEDENTS_ENC_KEY") or "").strip():
        fernets.append(Fernet(derived))
    return MultiFernet(fernets)


def encrypt_text(plaintext: str) -> bytes:
    return _fernet().encrypt((plaintext or "").encode("utf-8"))


def decrypt_text(ciphertext: bytes | None) -> str:
    if not ciphertext:
        return ""
    return _fernet_all().decrypt(bytes(ciphertext)).decode("utf-8")


# ---------------------------------------------------------------- database

_con: sqlite3.Connection | None = None
_con_lock = threading.RLock()
FTS_FLAVOUR = "fts5"  # downgraded to fts4 at init if fts5 is unavailable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS firms (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS firm_members (
    firm_id TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin','member')),
    invited_by TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (firm_id, email)
);
CREATE TABLE IF NOT EXISTS firm_invites (
    token TEXT PRIMARY KEY,
    firm_id TEXT NOT NULL,
    email TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'member',
    invited_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    accepted_at TEXT,
    accepted_by TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    firm_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT DEFAULT '',
    doc_type TEXT DEFAULT '',
    party_side TEXT DEFAULT '',
    act TEXT DEFAULT '',
    doc_date_hint TEXT DEFAULT '',
    storage TEXT NOT NULL,
    blob_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL DEFAULT '',
    pages INTEGER NOT NULL DEFAULT 0,
    tokens_est INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT DEFAULT '',
    attempts INTEGER NOT NULL DEFAULT 0,
    batch_id TEXT DEFAULT '',
    raw_result_enc BLOB,
    taxonomy_version TEXT DEFAULT '',
    uploaded_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimant TEXT DEFAULT '',
    respondent TEXT DEFAULT '',
    claimant_lawyers TEXT DEFAULT '',
    respondent_lawyers TEXT DEFAULT '',
    claimed_amount REAL,
    scheduled_amount REAL,
    meta_edited INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_documents_firm ON documents(firm_id, status);
CREATE INDEX IF NOT EXISTS idx_documents_batch ON documents(batch_id);
CREATE TABLE IF NOT EXISTS doc_pages (
    doc_id TEXT NOT NULL,
    page_no INTEGER NOT NULL,
    text_enc BLOB NOT NULL,
    chars INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (doc_id, page_no)
);
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    firm_id TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    heading TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    page_start INTEGER NOT NULL DEFAULT 1,
    page_end INTEGER NOT NULL DEFAULT 1,
    confidence REAL DEFAULT 0,
    created_at TEXT NOT NULL,
    stance TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sections_firm ON sections(firm_id, category, subcategory);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'processing',
    doc_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firm_id TEXT NOT NULL,
    email TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT DEFAULT '',
    detail TEXT DEFAULT '',
    ip TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_firm ON audit_log(firm_id, id);
CREATE TABLE IF NOT EXISTS cost_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firm_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cost_firm ON cost_ledger(firm_id);
"""


def get_con() -> sqlite3.Connection:
    global _con, FTS_FLAVOUR
    with _con_lock:
        if _con is not None:
            return _con
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        con = sqlite3.connect(DB_PATH, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        con.executescript(_SCHEMA)
        # Column migrations for DBs created under an earlier schema (the
        # CREATE TABLE IF NOT EXISTS above only shapes fresh installs).
        _migrations = [
            ("documents", "claimant", "TEXT DEFAULT ''"),
            ("documents", "respondent", "TEXT DEFAULT ''"),
            ("documents", "claimant_lawyers", "TEXT DEFAULT ''"),
            ("documents", "respondent_lawyers", "TEXT DEFAULT ''"),
            ("documents", "claimed_amount", "REAL"),
            ("documents", "scheduled_amount", "REAL"),
            ("documents", "meta_edited", "INTEGER NOT NULL DEFAULT 0"),
            ("sections", "stance", "TEXT DEFAULT ''"),
        ]
        for _table, _col, _ddl in _migrations:
            try:
                con.execute(f"ALTER TABLE {_table} ADD COLUMN {_col} {_ddl}")
            except sqlite3.OperationalError:
                pass  # column already exists
        # FTS over section text: fts5 preferred, fts4 fallback. Columns:
        # section text + summary searchable; ids carried alongside.
        try:
            con.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5("
                "section_text, summary, heading, section_id UNINDEXED, doc_id UNINDEXED, firm_id UNINDEXED)"
            )
            FTS_FLAVOUR = "fts5"
        except sqlite3.OperationalError:
            con.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts4("
                "section_text, summary, heading, section_id, doc_id, firm_id, tokenize=unicode61)"
            )
            FTS_FLAVOUR = "fts4"
        con.commit()
        _con = con
        return con


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def new_id() -> str:
    return uuid.uuid4().hex


def audit(firm_id: str, email: str, action: str, target: str = "", detail: str = "", ip: str = "") -> None:
    try:
        con = get_con()
        with _con_lock:
            con.execute(
                "INSERT INTO audit_log (firm_id, email, action, target, detail, ip, created_at) VALUES (?,?,?,?,?,?,?)",
                (firm_id, email, action, target, (detail or "")[:2000], ip or "", now_iso()),
            )
            con.commit()
    except Exception as exc:  # audit failures must never break the request
        print(f"[precedents] audit write failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------- blob storage

_gcs_client = None
_gcs_failed = False


def _get_gcs_client():
    """Same credential mechanism as the main server (GCS_CREDENTIALS_JSON env
    -> temp file), duplicated here to avoid importing server.py (circular)."""
    global _gcs_client, _gcs_failed
    if _gcs_client is not None:
        return _gcs_client
    if _gcs_failed:
        return None
    creds = os.getenv("GCS_CREDENTIALS_JSON")
    if not creds:
        _gcs_failed = True
        return None
    try:
        from google.cloud import storage

        path = "/tmp/gcs_credentials_precedents.json"
        with open(path, "w") as f:
            f.write(creds)
        _gcs_client = storage.Client.from_service_account_json(path)
        return _gcs_client
    except Exception as exc:
        print(f"[precedents] GCS client unavailable, using local blob store: {exc}", file=sys.stderr)
        _gcs_failed = True
        return None


def _gcs_object_name(firm_id: str, doc_id: str, filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in (".pdf", ".docx"):
        ext = ".bin"
    return f"{GCS_PREFIX}firm_{firm_id}/{doc_id}{ext}"


def save_blob(firm_id: str, doc_id: str, filename: str, data: bytes) -> tuple[str, str]:
    """Store the original upload. Returns (storage_kind, blob_path)."""
    client = _get_gcs_client()
    if client is not None:
        object_name = _gcs_object_name(firm_id, doc_id, filename)
        blob = client.bucket(GCS_BUCKET).blob(object_name)
        blob.upload_from_string(data, content_type="application/octet-stream")
        return "gcs", object_name
    os.makedirs(os.path.join(LOCAL_BLOB_DIR, f"firm_{firm_id}"), exist_ok=True)
    local_path = os.path.join(LOCAL_BLOB_DIR, f"firm_{firm_id}", f"{doc_id}{os.path.splitext(filename)[1].lower()}")
    with open(local_path, "wb") as f:
        f.write(data)
    return "local", local_path


def read_blob(storage_kind: str, blob_path: str) -> bytes:
    if storage_kind == "gcs":
        client = _get_gcs_client()
        if client is None:
            raise RuntimeError("GCS client unavailable")
        return client.bucket(GCS_BUCKET).blob(blob_path).download_as_bytes()
    with open(blob_path, "rb") as f:
        return f.read()


def delete_blob(storage_kind: str, blob_path: str) -> None:
    try:
        if storage_kind == "gcs":
            client = _get_gcs_client()
            if client is not None:
                client.bucket(GCS_BUCKET).blob(blob_path).delete()
        elif blob_path and os.path.exists(blob_path):
            os.remove(blob_path)
    except Exception as exc:
        print(f"[precedents] blob delete failed for {blob_path}: {exc}", file=sys.stderr)


def signed_url(storage_kind: str, blob_path: str, filename: str, minutes: int = 10) -> str | None:
    """Short-lived signed URL for GCS blobs; None means caller should stream."""
    if storage_kind != "gcs":
        return None
    client = _get_gcs_client()
    if client is None:
        return None
    try:
        blob = client.bucket(GCS_BUCKET).blob(blob_path)
        disposition = f'inline; filename="{(filename or "document.pdf").replace(chr(34), "")}"'
        content_type = "application/pdf" if blob_path.lower().endswith(".pdf") else "application/octet-stream"
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=minutes),
            method="GET",
            response_disposition=disposition,
            response_type=content_type,
        )
    except Exception as exc:
        print(f"[precedents] signed URL failed, will stream instead: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------- purge helpers

def purge_document(doc_row: sqlite3.Row | dict[str, Any]) -> None:
    """Hard-delete one document: blob + pages + sections + FTS rows + row."""
    con = get_con()
    doc_id = doc_row["id"]
    delete_blob(doc_row["storage"], doc_row["blob_path"])
    with _con_lock:
        con.execute("DELETE FROM sections_fts WHERE doc_id = ?", (doc_id,))
        con.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
        con.execute("DELETE FROM doc_pages WHERE doc_id = ?", (doc_id,))
        con.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        con.commit()


def purge_firm(firm_id: str) -> int:
    """Offboarding: hard-delete every document, member and invite for a firm.
    The firm row is kept (status='purged') and the audit trail is retained so
    there is a durable record that the purge happened and who asked for it."""
    con = get_con()
    docs = con.execute("SELECT * FROM documents WHERE firm_id = ?", (firm_id,)).fetchall()
    for d in docs:
        purge_document(d)
    with _con_lock:
        con.execute("DELETE FROM firm_members WHERE firm_id = ?", (firm_id,))
        con.execute("DELETE FROM firm_invites WHERE firm_id = ?", (firm_id,))
        con.execute("DELETE FROM cost_ledger WHERE firm_id = ?", (firm_id,))
        con.execute("UPDATE firms SET status = 'purged' WHERE id = ?", (firm_id,))
        con.commit()
    return len(docs)
