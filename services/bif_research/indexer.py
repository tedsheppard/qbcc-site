"""Build the unified bif_research index.

Walks all four chunkers (statutes, judgments, annotated, decisions),
assigns globally-unique chunk_ids, persists chunks to SQLite, embeds via
the budget-tracked OpenAI client, and builds a BM25 index alongside the
Chroma vector store.

Outputs (all in services/bif_research/store/):
  - chunks.sqlite   — chunk_id (PK), source_type, header, text, metadata JSON
  - chroma/         — Chroma persistent vector store
  - bm25.pkl        — pickled rank_bm25.BM25Okapi + token corpus

Run:
    python -m services.bif_research.indexer
    python -m services.bif_research.indexer --dry-run    # just count + cost projection
    python -m services.bif_research.indexer --skip-decisions  # statutes/judgments/annotated only
    python -m services.bif_research.indexer --decisions-limit 2000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

import chromadb
from rank_bm25 import BM25Okapi

from .corpus.base import Chunk
from .corpus import statutes, judgments, annotated, decisions
from . import budget, llm_config

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "store"
CHROMA_PATH = STORE / "chroma"
CHUNKS_DB = STORE / "chunks.sqlite"
BM25_PATH = STORE / "bm25.pkl"

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
INTER_BATCH_SLEEP_S = 2.0  # throttle to stay under OpenAI's 1M TPM cap

log = logging.getLogger("bif_research.indexer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Simple BM25 tokeniser — lowercase, alphanumerics + section/case-aware splits
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenise(text: str) -> list[str]:
    """Tokeniser used for BM25 indexing and querying. Must be deterministic."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def init_chunks_db() -> sqlite3.Connection:
    STORE.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(CHUNKS_DB))
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            source_id TEXT,
            source_type TEXT,
            header TEXT,
            text TEXT,
            metadata_json TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(source_type)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id)")
    con.commit()
    return con


def gather_chunks(skip_decisions: bool = False, decisions_limit: int | None = None) -> Iterator[Chunk]:
    log.info("walking statutes ...")
    yield from statutes.chunk_all()
    log.info("walking annotated ...")
    yield from annotated.chunk_all()
    log.info("walking judgments ...")
    yield from judgments.chunk_all()
    if not skip_decisions:
        log.info(f"walking decisions (limit={decisions_limit}) ...")
        yield from decisions.chunk_all(limit=decisions_limit)


def estimate_cost(chunks: list[Chunk]) -> float:
    total_chars = sum(len(c.indexed_text()) for c in chunks)
    tokens = total_chars // 4
    return budget.estimate_cost(EMBED_MODEL, tokens, 0)


def persist_chunks(chunks: list[Chunk], con: sqlite3.Connection) -> dict[str, str]:
    """Persist chunks to SQLite and return source_id -> chunk_id map."""
    con.execute("DELETE FROM chunks")  # fresh build
    cur = con.cursor()
    sid_to_chunk: dict[str, str] = {}
    for i, c in enumerate(chunks):
        chunk_id = f"chunk_{i:06d}"
        sid_to_chunk[c.source_id] = chunk_id
        cur.execute(
            "INSERT INTO chunks (chunk_id, source_id, source_type, header, text, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chunk_id, c.source_id, c.source_type, c.header, c.text, json.dumps(c.metadata)),
        )
    con.commit()
    log.info(f"persisted {len(chunks)} chunks to {CHUNKS_DB.name}")
    return sid_to_chunk


def build_bm25(chunks: list[Chunk]) -> BM25Okapi:
    log.info("tokenising for BM25 ...")
    corpus_tokens: list[list[str]] = []
    for c in chunks:
        corpus_tokens.append(tokenise(c.indexed_text()))
    log.info("fitting BM25 ...")
    bm25 = BM25Okapi(corpus_tokens)
    log.info(f"saving BM25 to {BM25_PATH.name}")
    with open(BM25_PATH, "wb") as f:
        # Save the BM25 object plus the corpus token lists so we can rebuild
        pickle.dump({"bm25": bm25, "corpus_tokens": corpus_tokens}, f)
    return bm25


def build_chroma(chunks: list[Chunk], chunk_ids: list[str], dry_run: bool = False) -> None:
    if not chunks:
        return
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # Drop old collection if it exists
    try:
        chroma_client.delete_collection("bif_research")
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name="bif_research",
        metadata={"hnsw:space": "cosine"},
    )

    log.info(f"embedding {len(chunks)} chunks via {EMBED_MODEL} (batch {BATCH_SIZE}) ...")
    n_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    t_start = time.time()
    for batch_i in range(n_batches):
        budget.check_budget()
        lo = batch_i * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, len(chunks))
        batch = chunks[lo:hi]
        batch_ids = chunk_ids[lo:hi]
        texts = [c.indexed_text() for c in batch]
        if dry_run:
            log.info(f"  [dry-run] batch {batch_i+1}/{n_batches} "
                     f"({sum(len(t) for t in texts):,} chars)")
            continue
        try:
            embeddings = llm_config.embed(texts, model=EMBED_MODEL, operation="index-embed")
        except budget.BudgetExceeded as e:
            log.error(f"budget exceeded mid-index: {e}")
            log.error(f"stopping after {batch_i} of {n_batches} batches "
                      f"({lo} of {len(chunks)} chunks embedded)")
            raise
        # Build metadata for Chroma (must be flat, str/int/float/bool only)
        metas = []
        for c in batch:
            md = {"source_type": c.source_type, "header": c.header[:500]}
            for k, v in c.metadata.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    # Truncate long string values
                    md[k] = v if not isinstance(v, str) else v[:500]
            metas.append(md)
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=metas,
            documents=texts,
        )
        elapsed = time.time() - t_start
        per_batch = elapsed / (batch_i + 1)
        eta = per_batch * (n_batches - batch_i - 1)
        log.info(
            f"  batch {batch_i+1}/{n_batches}  "
            f"cum=${budget.cumulative_usd():.4f}  eta={eta/60:.1f}m"
        )
        # Throttle between batches to respect TPM rate limit
        if batch_i + 1 < n_batches:
            time.sleep(INTER_BATCH_SLEEP_S)
    log.info(f"chroma collection 'bif_research' has {collection.count()} items")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Project cost only, no embedding")
    ap.add_argument("--skip-decisions", action="store_true", help="Index statutes+judgments+annotated only")
    ap.add_argument("--decisions-limit", type=int, default=None, help="Cap number of decisions indexed")
    args = ap.parse_args()

    log.info("gathering chunks ...")
    chunks = list(gather_chunks(
        skip_decisions=args.skip_decisions,
        decisions_limit=args.decisions_limit,
    ))
    log.info(f"total chunks: {len(chunks):,}")

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.source_type] = by_type.get(c.source_type, 0) + 1
    for t, n in sorted(by_type.items()):
        log.info(f"  {t}: {n:,}")

    proj = estimate_cost(chunks)
    log.info(f"projected embed cost ({EMBED_MODEL}): ${proj:.4f}")
    log.info(f"current cumulative spend: ${budget.cumulative_usd():.4f}")
    log.info(f"would leave remaining: ${budget.HARD_CAP_USD - budget.cumulative_usd() - proj:.4f}")

    if args.dry_run:
        return 0

    if budget.cumulative_usd() + proj > budget.HARD_CAP_USD:
        log.error(f"projected cost would exceed hard cap. Aborting. "
                  f"Current=${budget.cumulative_usd():.4f}, projected=${proj:.4f}, "
                  f"cap=${budget.HARD_CAP_USD:.2f}")
        return 2

    # Stable IDs persisted to SQLite
    con = init_chunks_db()
    sid_to_chunk = persist_chunks(chunks, con)
    chunk_ids = [sid_to_chunk[c.source_id] for c in chunks]

    build_bm25(chunks)
    build_chroma(chunks, chunk_ids)

    log.info(f"indexing complete. cum spend ${budget.cumulative_usd():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
