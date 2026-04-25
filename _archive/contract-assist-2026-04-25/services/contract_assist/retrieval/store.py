"""Session-scoped, in-memory ChromaDB store for the contract_assist pipeline.

One process-wide ``EphemeralClient`` is created lazily on first use; each
session gets its own collection named ``contract_assist_{session_id}``.
Sessions vanish when the process restarts — that's intentional, since the
contract_assist UX is per-conversation and we never want a re-uploaded
contract to leak across processes.

ChromaDB handles concurrent reads internally. We protect collection
*creation* and *deletion* with a module-level ``threading.Lock`` so two
parallel ingest calls for the same session don't race.

All ``chromadb`` imports are deferred inside helper functions.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

log = logging.getLogger("contract_assist.retrieval.store")

_COLLECTION_PREFIX = "contract_assist_"
_DISTANCE_METRIC = "cosine"

# Process-singleton client.
_client = None  # type: ignore[var-annotated]
_client_lock = threading.Lock()
# Per-session creation/deletion lock.
_collection_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Client / collection access
# ---------------------------------------------------------------------------

def _get_client():
    """Return the process-wide EphemeralClient, creating it on first call."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            import chromadb  # deferred
            _client = chromadb.EphemeralClient()
    return _client


def _collection_name(session_id: str) -> str:
    if not session_id or not str(session_id).strip():
        raise ValueError("session_id is required")
    # Chroma names must be 3-63 chars and match a restricted regex; sessions
    # are usually UUIDs which are fine, but we sanitize defensively.
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(session_id))
    return f"{_COLLECTION_PREFIX}{safe}"


def _get_or_create_collection(session_id: str):
    name = _collection_name(session_id)
    client = _get_client()
    with _collection_lock:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": _DISTANCE_METRIC},
        )


def _maybe_get_collection(session_id: str):
    name = _collection_name(session_id)
    client = _get_client()
    try:
        return client.get_collection(name=name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public store API
# ---------------------------------------------------------------------------

def add_chunks(
    session_id: str,
    *,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """Add a batch of chunks to the session collection."""
    if not ids:
        return
    if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
        raise ValueError("add_chunks: ids/documents/embeddings/metadatas length mismatch")

    safe_metas = [_metadata_safe(m) for m in metadatas]
    coll = _get_or_create_collection(session_id)
    coll.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=safe_metas,
    )


def query_vector(
    session_id: str,
    embedding: list[float],
    *,
    n_results: int = 25,
) -> list[dict[str, Any]]:
    """Run a vector similarity query. Returns list of hit dicts."""
    coll = _maybe_get_collection(session_id)
    if coll is None:
        return []

    n_results = max(1, int(n_results))
    res = coll.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )
    return _flatten_query_result(res)


def get_all(session_id: str) -> list[dict[str, Any]]:
    """Return every chunk in the session collection (for keyword/clause filtering)."""
    coll = _maybe_get_collection(session_id)
    if coll is None:
        return []
    res = coll.get(include=["metadatas", "documents"])
    ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    docs = res.get("documents") or []
    out: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        meta = (metas[i] if i < len(metas) else {}) or {}
        out.append({
            "id": cid,
            "metadata": _metadata_unpack(meta),
            "document": docs[i] if i < len(docs) else "",
            "distance": None,
        })
    return out


def get_by_ids(session_id: str, ids: list[str]) -> list[dict[str, Any]]:
    """Fetch specific chunks by id."""
    if not ids:
        return []
    coll = _maybe_get_collection(session_id)
    if coll is None:
        return []
    res = coll.get(ids=ids, include=["metadatas", "documents"])
    found_ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    docs = res.get("documents") or []
    out: list[dict[str, Any]] = []
    for i, cid in enumerate(found_ids):
        meta = (metas[i] if i < len(metas) else {}) or {}
        out.append({
            "id": cid,
            "metadata": _metadata_unpack(meta),
            "document": docs[i] if i < len(docs) else "",
            "distance": None,
        })
    return out


def drop_session(session_id: str) -> None:
    """Delete the session collection. Idempotent."""
    name = _collection_name(session_id)
    client = _get_client()
    with _collection_lock:
        try:
            client.delete_collection(name=name)
        except Exception as e:
            # ChromaDB raises NotFoundError (or its subclasses) when the
            # collection isn't there — that's a no-op for us. We swallow
            # any "not found" style error silently and let real errors
            # bubble up via the log.
            try:
                from chromadb.errors import NotFoundError  # type: ignore
            except Exception:
                NotFoundError = ()  # type: ignore[assignment]
            if isinstance(e, NotFoundError) if NotFoundError else False:
                return
            msg = str(e).lower()
            if "does not exist" in msg or "not found" in msg or "no such" in msg:
                return
            log.warning("drop_session(%s) failed: %s", session_id, e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def distance_to_similarity(distance: float | None) -> float:
    """Convert chroma cosine distance (0=identical, 2=opposite) to a 0..1 score."""
    if distance is None:
        return 0.0
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    score = 1.0 - (d / 2.0)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _flatten_query_result(res: dict[str, Any]) -> list[dict[str, Any]]:
    ids = (res.get("ids") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        meta = (metas[i] if i < len(metas) else {}) or {}
        out.append({
            "id": cid,
            "metadata": _metadata_unpack(meta),
            "document": docs[i] if i < len(docs) else "",
            "distance": dists[i] if i < len(dists) else None,
        })
    return out


def _metadata_safe(meta: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB metadata values must be primitives. Stringify lists.

    We store ``clause_numbers`` as a comma-separated string and unpack it
    back into a list on read.
    """
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, list):
            out[k] = ",".join(str(x) for x in v)
        elif v is None:
            # Chroma rejects None — coerce to empty string and we'll undo on read.
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _metadata_unpack(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(meta)
    cn = out.get("clause_numbers")
    if isinstance(cn, str):
        out["clause_numbers"] = [t for t in cn.split(",") if t]
    elif cn is None:
        out["clause_numbers"] = []
    return out


# Test / process-recycle helper. NOT part of the public contract.
def _reset_for_tests() -> None:
    """Drop the singleton client. Used by the test suite to isolate runs."""
    global _client
    with _client_lock:
        _client = None
