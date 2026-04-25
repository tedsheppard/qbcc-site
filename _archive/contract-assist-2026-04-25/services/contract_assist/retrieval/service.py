"""Public ingest / retrieve / clear entry points for /contract-assist.

Wires the chunker, embedder, and store together. Reuses
``services.claim_check.extractor.extract_rich`` for PDF/DOCX text
extraction so we don't have two copies of that logic in the repo.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

log = logging.getLogger("contract_assist.retrieval.service")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

VECTOR_SIMILARITY_THRESHOLD = 0.30
VECTOR_TOP_N = 25
CLAUSE_HITS_PER_REF = 5
KEYWORD_HITS_TOTAL = 5
MERGE_TOP_N = 12
EXPAND_NEIGHBOURS_FOR_TOP = 5

# Score floors per source method (per Astruct Section 5 patterns doc, adapted).
SCORE_CLAUSE_MATCH = 0.95
SCORE_KEYWORD_MATCH = 0.55

# Identified-form patterns — substring (case-insensitive) match on the first
# 4000 characters of the document.
_IDENTIFIED_FORMS: list[tuple[str, str]] = [
    ("as 4000-1997", "AS 4000-1997"),
    ("as 4000",      "AS 4000-1997"),
    ("as 4902",      "AS 4902"),
    ("as 2124",      "AS 2124"),
    ("as 4300",      "AS 4300"),
    ("as 4904",      "AS 4904"),
]

_FORM_DETECTION_WINDOW = 4000

# Stoplist for keyword search. Small on purpose — this is a fallback that
# helps surface chunks containing distinctive contract terms.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "by", "from", "with", "as", "and",
    "or", "but", "if", "then", "else", "this", "that", "these", "those",
    "it", "its", "i", "you", "we", "they", "he", "she", "them", "us", "me",
    "do", "does", "did", "have", "has", "had", "will", "would", "should",
    "can", "could", "may", "might", "must", "what", "which", "who", "whom",
    "when", "where", "why", "how", "about", "into", "out", "up", "down",
    "over", "under", "than", "so", "not", "no", "yes", "any", "some", "all",
    "tell", "explain", "show", "give", "say", "ask", "please",
    "clause", "section",
}

# Detect clause references in the user's query.
_QUERY_CLAUSE_RE = re.compile(
    r"(?:clause|section)\s+(\d+(?:\.\d+){0,3})|\b(\d+\.\d+(?:\.\d+){0,2})\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(file_bytes: bytes, filename: str, session_id: str) -> dict[str, Any]:
    """Extract → chunk → embed → store.

    Returns a status dict (see module docstring).

    Raises:
      ValueError: empty extraction or unsupported file type.
      RuntimeError: missing OPENAI_API_KEY (raised from the embedder).
    """
    if not session_id or not str(session_id).strip():
        raise ValueError("session_id is required")
    if not filename:
        raise ValueError("filename is required")

    started = time.time()

    text, extras = _extract(filename, file_bytes)

    page_count = extras.get("pages") if isinstance(extras, dict) else None
    if not isinstance(page_count, int) or page_count <= 0:
        page_count = None

    identified_form = _identify_form(text)

    from .chunker import chunk_text
    from .embedder import embed_texts
    from . import store

    chunks = chunk_text(text, document_name=filename)
    if not chunks:
        raise ValueError("No chunks could be produced from this document.")

    docs = [c["content"] for c in chunks]
    embeddings = embed_texts(docs)
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"embedding count mismatch: got {len(embeddings)} vectors for {len(chunks)} chunks"
        )

    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for c in chunks:
        ids.append(_chunk_id(filename, c["chunk_index"]))
        metadatas.append({
            "document_name": c.get("document_name") or filename,
            "chunk_index": int(c["chunk_index"]),
            "section_heading": c.get("section_heading") or "",
            "clause_numbers": c.get("clause_numbers") or [],
            "page_number": c.get("page_number") if c.get("page_number") is not None else "",
            "identified_form": identified_form or "",
        })

    store.add_chunks(
        session_id,
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    elapsed_ms = int((time.time() - started) * 1000)
    return {
        "session_id": session_id,
        "filename": filename,
        "page_count": page_count,
        "chunk_count": len(chunks),
        "identified_form": identified_form,
        "elapsed_ms": elapsed_ms,
    }


def retrieve(query: str, session_id: str, top_k: int = 6) -> list[dict[str, Any]]:
    """Hybrid retrieval. See module docstring for the result schema."""
    if not query or not query.strip():
        return []
    if not session_id or not str(session_id).strip():
        return []
    top_k = max(1, int(top_k))

    from .embedder import embed_query
    from . import store

    # 1) Vector search.
    vector_hits: list[dict[str, Any]] = []
    try:
        q_emb = embed_query(query)
        raw = store.query_vector(session_id, q_emb, n_results=VECTOR_TOP_N)
        for hit in raw:
            sim = store.distance_to_similarity(hit.get("distance"))
            if sim < VECTOR_SIMILARITY_THRESHOLD:
                continue
            vector_hits.append({**hit, "_sim": sim})
    except RuntimeError:
        # Bubble OPENAI_API_KEY errors so callers can surface them clearly.
        raise
    except Exception as e:
        log.warning("vector search failed for session=%s: %s", session_id, e)

    # 2) Clause-specific search.
    clause_refs = _query_clause_refs(query)
    clause_hits: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] | None = None
    if clause_refs:
        all_chunks = store.get_all(session_id)
        for ref in clause_refs:
            matches = []
            for ch in all_chunks:
                cn = (ch.get("metadata") or {}).get("clause_numbers") or []
                if ref in cn or any(c == ref for c in cn):
                    matches.append(ch)
                if len(matches) >= CLAUSE_HITS_PER_REF:
                    break
            clause_hits.extend(matches)

    # 3) Keyword search.
    keyword_hits: list[dict[str, Any]] = []
    keywords = _query_keywords(query)
    if keywords:
        if all_chunks is None:
            all_chunks = store.get_all(session_id)
        for ch in all_chunks:
            doc = (ch.get("document") or "").lower()
            if any(kw in doc for kw in keywords):
                keyword_hits.append(ch)
            if len(keyword_hits) >= KEYWORD_HITS_TOTAL:
                break

    # 4) Merge & dedupe — clause first (highest priority), then vector, then keyword.
    merged: list[tuple[dict[str, Any], float, str]] = []
    seen: set[str] = set()

    def _push(hit: dict[str, Any], score: float, method: str) -> None:
        cid = hit.get("id")
        if not cid or cid in seen:
            return
        seen.add(cid)
        merged.append((hit, score, method))

    for h in clause_hits:
        _push(h, SCORE_CLAUSE_MATCH, "clause_match")
    # Vector hits must be sorted high-to-low before merging.
    vector_hits.sort(key=lambda h: h.get("_sim", 0.0), reverse=True)
    for h in vector_hits:
        _push(h, float(h.get("_sim", 0.0)), "vector")
    for h in keyword_hits:
        _push(h, SCORE_KEYWORD_MATCH, "keyword")

    if not merged:
        return []

    # Stable sort by score desc — preserves the clause-first bias on ties.
    merged.sort(key=lambda t: t[1], reverse=True)
    merged = merged[:MERGE_TOP_N]

    # 5) Adjacent expansion for the top N.
    expansions = _build_expansions(session_id, merged[:EXPAND_NEIGHBOURS_FOR_TOP])

    # 6) Build result dicts and trim to top_k.
    results: list[dict[str, Any]] = []
    for hit, score, method in merged[:top_k]:
        meta = hit.get("metadata") or {}
        full_text = hit.get("document") or ""
        cid = hit.get("id") or ""
        page = meta.get("page_number")
        if isinstance(page, str):
            page = int(page) if page.strip().isdigit() else None
        results.append({
            "chunk_id": cid,
            "document_name": meta.get("document_name") or "",
            "section_heading": (meta.get("section_heading") or None) or None,
            "clause_numbers": list(meta.get("clause_numbers") or []),
            "page_number": page,
            "excerpt": _excerpt(full_text, 200),
            "full_text": full_text,
            "expanded_text": expansions.get(cid, full_text),
            "score": float(score),
            "score_method": method,
        })
    return results


def clear(session_id: str) -> None:
    """Purge all chunks for a session. Idempotent."""
    if not session_id:
        return
    from . import store
    store.drop_session(session_id)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _extract(filename: str, file_bytes: bytes) -> tuple[str, dict[str, Any]]:
    name = (filename or "").lower()
    if not name.endswith((".pdf", ".docx", ".txt")):
        raise ValueError(
            f"Unsupported file type: {filename!r}. Use PDF, DOCX, or TXT."
        )
    # Reuse the claim_check extractor so we don't duplicate PDF/DOCX logic.
    from services.claim_check.extractor import extract_rich
    text, extras = extract_rich(filename, file_bytes)
    if not text or not text.strip():
        raise ValueError("No text could be extracted from this document.")
    return text, extras or {}


def _identify_form(text: str) -> str | None:
    if not text:
        return None
    window = text[:_FORM_DETECTION_WINDOW].lower()
    for needle, canonical in _IDENTIFIED_FORMS:
        if needle in window:
            return canonical
    return None


def _chunk_id(filename: str, chunk_index: int) -> str:
    # Stable, but include a uuid4 hex so re-ingesting the same file in the
    # same session won't collide on chroma's primary key.
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename or "doc")
    return f"{safe}::{chunk_index:04d}::{uuid.uuid4().hex[:8]}"


def _query_clause_refs(query: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for m in _QUERY_CLAUSE_RE.finditer(query or ""):
        ref = (m.group(1) or m.group(2) or "").strip()
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def _query_keywords(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (query or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if t in _STOPWORDS:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _excerpt(text: str, n: int) -> str:
    if not text:
        return ""
    s = text.strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def _build_expansions(
    session_id: str,
    top_hits: list[tuple[dict[str, Any], float, str]],
) -> dict[str, str]:
    """For each top hit, return concatenated text of (prev, hit, next) chunks
    within the same document. Falls back to the hit's own text if neighbours
    can't be located."""
    if not top_hits:
        return {}
    from . import store

    # Bucket the document → ordered list of (chunk_index, id, content).
    needed_docs: set[str] = set()
    for hit, _, _ in top_hits:
        meta = hit.get("metadata") or {}
        doc_name = meta.get("document_name") or ""
        if doc_name:
            needed_docs.add(doc_name)

    if not needed_docs:
        return {(h.get("id") or ""): (h.get("document") or "") for h, _, _ in top_hits}

    all_chunks = store.get_all(session_id)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for ch in all_chunks:
        meta = ch.get("metadata") or {}
        doc = meta.get("document_name") or ""
        if doc not in needed_docs:
            continue
        by_doc.setdefault(doc, []).append(ch)
    for doc, items in by_doc.items():
        items.sort(key=lambda c: int((c.get("metadata") or {}).get("chunk_index") or 0))

    out: dict[str, str] = {}
    for hit, _, _ in top_hits:
        meta = hit.get("metadata") or {}
        cid = hit.get("id") or ""
        doc_name = meta.get("document_name") or ""
        chunk_index = int(meta.get("chunk_index") or 0)
        items = by_doc.get(doc_name, [])
        # Locate the position of this chunk in the doc-ordered list.
        pos = -1
        for i, ch in enumerate(items):
            if ch.get("id") == cid:
                pos = i
                break
        if pos == -1:
            out[cid] = hit.get("document") or ""
            continue
        prev_text = items[pos - 1].get("document") if pos > 0 else ""
        next_text = items[pos + 1].get("document") if pos + 1 < len(items) else ""
        parts = [t for t in (prev_text, hit.get("document") or "", next_text) if t]
        out[cid] = "\n\n".join(parts)
        # chunk_index isn't read again, but we keep it for forward-compat.
        _ = chunk_index
    return out
