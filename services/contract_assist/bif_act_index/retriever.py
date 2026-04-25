"""Retrieval API for the BIF Act guide embedded index.

Single public function: :func:`retrieve_bif`. Embeds the query with
``text-embedding-3-small`` and runs a cosine similarity search against
the persistent ChromaDB collection built by :mod:`builder`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .builder import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    ANCHOR_URL_BASE,
)

log = logging.getLogger("contract_assist.bif_act_index.retriever")

# Snippet display cap per spec.
_MAX_SNIPPET_CHARS = 800


def _embed_query(text: str) -> list[float]:
    from openai import OpenAI  # deferred

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured; cannot run BIF Act retrieval."
        )
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def _open_collection():
    import chromadb  # deferred

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def _distance_to_score(distance: float | None) -> float:
    """Convert chroma cosine distance (0=identical, 2=opposite) to a
    0–1 similarity score (1=identical)."""
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


def retrieve_bif(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Retrieve up to ``top_k`` BIF Act guide chunks similar to ``query``.

    Returns a list of dicts each containing:
      - ``section_ref`` (e.g. "s 68"; may be empty if undetectable)
      - ``heading`` (the section's <h2> text)
      - ``snippet`` (chunk text, truncated to 800 chars for display)
      - ``score`` (cosine similarity, 0..1; higher = more similar)
      - ``anchor_url`` (deep-link of the form "/bif-act-guide#section-id")

    The list is ordered by similarity (highest first). This is a pure
    similarity search — off-topic queries still return ``top_k`` rows,
    but with low scores.
    """
    if not query or not query.strip():
        return []
    top_k = max(1, int(top_k))

    try:
        collection = _open_collection()
    except Exception as e:
        log.warning("BIF Act collection unavailable: %s", e)
        return []

    embedding = _embed_query(query.strip())

    res = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    metadatas = (res.get("metadatas") or [[]])[0]
    documents = (res.get("documents") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    out: list[dict[str, Any]] = []
    for meta, doc, dist in zip(metadatas, documents, distances):
        meta = meta or {}
        snippet = doc or ""
        if len(snippet) > _MAX_SNIPPET_CHARS:
            snippet = snippet[:_MAX_SNIPPET_CHARS].rstrip() + "…"
        anchor_url = meta.get("anchor_url") or ANCHOR_URL_BASE
        out.append({
            "section_ref": meta.get("section_ref", "") or "",
            "heading": meta.get("heading", "") or "",
            "snippet": snippet,
            "score": _distance_to_score(dist),
            "anchor_url": anchor_url,
        })
    return out
