"""Tests for the BIF Act guide embedded index.

These tests stub out the OpenAI embedding call (so no network access or
API key is required) and let ChromaDB persist to a temp dir injected
via :data:`builder.CHROMA_DIR`. We use a small, deterministic
embedding function so that semantic-ish ordering still works for
keyword-overlap queries.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pytest

# Make the repo root importable when pytest is run from any directory.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.contract_assist.bif_act_index import (  # noqa: E402
    build_index,
    retrieve_bif,
)
from services.contract_assist.bif_act_index import builder as builder_mod  # noqa: E402
from services.contract_assist.bif_act_index import retriever as retriever_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fake embedding: deterministic, content-aware, so similar texts cluster.
# ---------------------------------------------------------------------------


_DIM = 256

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "will", "with", "which", "but", "not", "no",
    "if", "so", "any", "all", "may", "can", "do", "does", "did", "i", "you",
    "we", "they", "he", "she", "his", "her", "their", "our", "out", "one",
    "two", "three", "than", "then", "there", "these", "those", "such", "also",
    "into", "over", "under", "about", "after", "before", "where", "when",
    "what", "how", "why", "who", "whom", "whose", "would", "should", "could",
    "must", "might", "very", "much", "more", "less", "between",
}


def _tokenise(s: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in s)
    return [w for w in cleaned.split() if w and w not in _STOPWORDS and len(w) > 2]


def _fake_embed(text: str) -> list[float]:
    """Hash-bag-of-content-words embedding into a unit vector.

    Two texts that share content keywords will land closer in cosine
    space, which is enough to validate retrieval ordering for our
    tests. Stop-words are dropped so an off-topic English sentence
    doesn't share dimensions with the corpus simply by virtue of using
    the language.
    """
    vec = [0.0] * _DIM
    for tok in _tokenise(text):
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % _DIM] += 1.0
        vec[(h >> 16) % _DIM] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _fake_embed_batch(texts):
    return [_fake_embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_index(tmp_path, monkeypatch):
    """Point the builder + retriever at a temp chroma dir and stub embeddings."""
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(builder_mod, "CHROMA_DIR", chroma_dir, raising=True)
    monkeypatch.setattr(retriever_mod, "CHROMA_DIR", chroma_dir, raising=True)

    # Reset the per-process build flag so build_index actually runs.
    monkeypatch.setattr(builder_mod, "_built", False, raising=True)

    # Stub the OpenAI-backed embedding helpers.
    monkeypatch.setattr(builder_mod, "_embed_texts", _fake_embed_batch, raising=True)
    monkeypatch.setattr(retriever_mod, "_embed_query", _fake_embed, raising=True)

    return chroma_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_guide_produces_chunks():
    """The parser should yield multiple chunks across multiple guide sections."""
    chunks = builder_mod.parse_guide()
    assert len(chunks) >= 20, f"expected >=20 chunks, got {len(chunks)}"

    # Multiple distinct sections (anchor_ids) should be represented.
    anchor_ids = {c["anchor_id"] for c in chunks if c["anchor_id"]}
    assert len(anchor_ids) >= 10, f"expected >=10 distinct sections, got {len(anchor_ids)}"

    # Required keys on every chunk.
    required = {"id", "section_ref", "heading", "anchor_id", "anchor_url", "text"}
    for c in chunks:
        assert required.issubset(c.keys())
        assert c["text"].strip(), "chunk text must not be empty"
        assert c["anchor_url"].startswith("/bif-act-guide")

    # Several BIF Act sections should be inferred.
    refs = {c["section_ref"] for c in chunks if c["section_ref"]}
    assert "s 68" in refs, "expected commentary on s 68 to be present"


def test_build_index_creates_collection(patched_index):
    """build_index parses the guide, embeds the chunks, and persists them."""
    build_index()

    import chromadb

    client = chromadb.PersistentClient(path=str(patched_index))
    coll = client.get_collection(name=builder_mod.COLLECTION_NAME)

    parsed = builder_mod.parse_guide()
    assert coll.count() == len(parsed) > 0


def test_retrieve_returns_top_k_with_required_fields(patched_index):
    build_index()

    results = retrieve_bif("how is reference date defined", top_k=4)
    assert isinstance(results, list)
    assert len(results) == 4

    required = {"section_ref", "heading", "snippet", "score", "anchor_url"}
    for r in results:
        assert required.issubset(r.keys())
        assert isinstance(r["snippet"], str) and r["snippet"]
        assert len(r["snippet"]) <= 800 + 1  # +1 for trailing ellipsis
        assert isinstance(r["score"], float)
        assert 0.0 <= r["score"] <= 1.0
        assert r["anchor_url"].startswith("/bif-act-guide")

    # Reference-date queries should preferentially surface the
    # reference-date section.
    headings = " ".join(r["heading"].lower() for r in results)
    assert "reference date" in headings


def test_retrieve_offtopic_query_still_returns_top_k(patched_index):
    """Similarity search returns top_k rows even for unrelated queries; scores
    should be meaningfully lower than for an on-topic query."""
    build_index()

    on_topic = retrieve_bif("payment claim must identify construction work", top_k=4)
    off_topic = retrieve_bif(
        "octopus tentacles photosynthesise marshmallow saxophone glaciers",
        top_k=4,
    )

    assert len(off_topic) == 4
    for r in off_topic:
        required = {"section_ref", "heading", "snippet", "score", "anchor_url"}
        assert required.issubset(r.keys())

    assert on_topic[0]["score"] > off_topic[0]["score"], (
        f"expected on-topic top score ({on_topic[0]['score']:.3f}) to exceed "
        f"off-topic top score ({off_topic[0]['score']:.3f})"
    )


def test_retrieve_works_without_rebuilding(patched_index, monkeypatch):
    """Once the persistent collection exists, retrieve_bif must not need
    build_index to be called again."""
    # First, build the index in this process.
    build_index()

    # Simulate a fresh process: clear the in-process build flag and make
    # any further call to ``builder._embed_texts`` fail loudly so we can
    # be certain no re-embedding happens.
    monkeypatch.setattr(builder_mod, "_built", False, raising=True)

    def _no_more_embedding(_texts):
        raise AssertionError("embeddings should not be recomputed on retrieval")

    monkeypatch.setattr(builder_mod, "_embed_texts", _no_more_embedding, raising=True)

    # retrieve_bif on its own (no build_index call) should still work
    # because the collection is already on disk.
    results = retrieve_bif("reference date", top_k=4)
    assert len(results) == 4
    assert all("snippet" in r for r in results)
