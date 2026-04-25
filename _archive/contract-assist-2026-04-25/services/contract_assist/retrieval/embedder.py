"""OpenAI embeddings wrapper for the contract_assist retrieval pipeline.

Uses ``text-embedding-3-small`` (1536 dims). The smaller model is plenty
for clause-level retrieval and matches the BIF Act index agent's choice
for project consistency.

Imports of ``openai`` are deferred inside the call so importing this
module costs nothing.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

log = logging.getLogger("contract_assist.retrieval.embedder")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
EMBEDDING_BATCH_SIZE = 96


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of strings via OpenAI; returns one vector per input.

    Splits requests of more than ``EMBEDDING_BATCH_SIZE`` items into
    multiple OpenAI calls and concatenates the results in order.

    Raises:
      RuntimeError: if ``OPENAI_API_KEY`` is missing.
      ValueError: if ``texts`` is empty.
    """
    if texts is None or len(texts) == 0:
        raise ValueError("embed_texts requires at least one input string")

    client = _client()

    out: list[list[float]] = []
    for batch in _batched(texts, EMBEDDING_BATCH_SIZE):
        # The OpenAI SDK tolerates empty strings, but our chunker won't
        # emit them; defensive guard:
        cleaned = [t if (t and t.strip()) else " " for t in batch]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=cleaned)
        for item in resp.data:
            out.append(list(item.embedding))
    return out


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    if not text or not text.strip():
        raise ValueError("embed_query requires a non-empty string")
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _client():
    from openai import OpenAI  # deferred

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def _batched(items: Sequence[str], size: int):
    if size <= 0:
        size = 1
    for i in range(0, len(items), size):
        yield items[i : i + size]
