"""Hybrid (BM25 + dense) retrieval over the bif_research index.

For Phase 4 (naive baseline): BM25-only retrieval. Dense retrieval and
RRF fusion are wired but disabled by default — Phase 5 enables them and
measures the delta.

Public API:
    retr = Retriever()
    hits = retr.retrieve(query, k=15, mode="bm25" | "dense" | "hybrid",
                         intent="statutory" | "case_law" | "procedural" |
                                "definitional" | "general")

Each Hit:
    {"chunk_id": "chunk_004471",
     "source_type": "statute",
     "header": "BIF Act s 68 — ...",
     "text": "...",
     "metadata": {...},
     "scores": {"bm25": 0.42, "dense": 0.81, "fused": 0.61},
     "rank": 1}
"""
from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chromadb
from rank_bm25 import BM25Okapi

from . import budget, llm_config, name_index
from .indexer import (
    BM25_PATH, CHROMA_PATH, CHUNKS_DB, EMBED_MODEL, tokenise,
)

log = logging.getLogger("bif_research.retriever")


# Source-type weights by intent (per spec section 5.2).
# Iter-3 of Phase 5 tried bumping the case_law judgment weight to 2.0 and
# suppressing statute/annotated weights for case_law intent; that combined
# with a wider candidate pool produced a -27 pt aggregate regression
# (iter-2 87.1% -> iter-3 59.7%) — the answerer started citing chunks
# outside the retrieved set, likely because the wider pool introduced
# noise that the model couldn't disambiguate. Per Phase 5 rule (delta
# < +2 -> revert), reverted to iter-2 weights.
WEIGHTS = {
    "statutory":    {"statute": 1.5, "regulation": 1.4, "annotated": 1.0, "judgment": 0.9, "decision": 0.7},
    "case_law":     {"statute": 1.2, "regulation": 1.0, "judgment": 1.5, "decision": 1.0, "annotated": 0.8},
    "procedural":   {"statute": 1.4, "regulation": 1.3, "judgment": 1.1, "decision": 1.0, "annotated": 1.0},
    "definitional": {"statute": 1.6, "regulation": 1.3, "annotated": 1.2, "judgment": 0.9, "decision": 0.7},
    "general":      {"statute": 1.2, "regulation": 1.1, "judgment": 1.1, "decision": 1.0, "annotated": 1.0},
}

# Refusal thresholds (configurable per spec section 4.5)
REFUSAL_BM25_MIN = 3.0   # top BM25 raw score
REFUSAL_DENSE_MIN = 0.30  # top cosine similarity (-3 large) — for small, slightly lower bar


@dataclass
class Hit:
    chunk_id: str
    source_type: str
    header: str
    text: str
    metadata: dict
    scores: dict
    rank: int = 0


class Retriever:
    def __init__(self):
        self._chunks_con: sqlite3.Connection | None = None
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] | None = None
        self._chunk_ids_ordered: list[str] | None = None  # parallel to bm25 corpus
        self._chroma_collection = None

    def _ensure_loaded(self):
        if self._chunks_con is None:
            # check_same_thread=False — Retriever instance is shared across
            # FastAPI worker threads. Read-only so it's safe.
            self._chunks_con = sqlite3.connect(str(CHUNKS_DB), check_same_thread=False)
            self._chunks_con.row_factory = sqlite3.Row
        if self._bm25 is None:
            with open(BM25_PATH, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._corpus_tokens = data["corpus_tokens"]
            # The corpus is in insertion order, which matches chunk_NNNNNN ids
            cur = self._chunks_con.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id")
            self._chunk_ids_ordered = [r[0] for r in cur.fetchall()]
            assert len(self._chunk_ids_ordered) == len(self._corpus_tokens), (
                f"BM25 corpus size {len(self._corpus_tokens)} != "
                f"chunks table size {len(self._chunk_ids_ordered)}"
            )
        if self._chroma_collection is None:
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            try:
                self._chroma_collection = client.get_collection("bif_research")
            except Exception as e:
                log.warning(f"chroma collection unavailable: {e}")
                self._chroma_collection = None

    def _fetch_chunks(self, chunk_ids: list[str]) -> dict[str, Hit]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._chunks_con.execute(
            f"SELECT chunk_id, source_id, source_type, header, text, metadata_json "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        out: dict[str, Hit] = {}
        for r in rows:
            out[r["chunk_id"]] = Hit(
                chunk_id=r["chunk_id"],
                source_type=r["source_type"],
                header=r["header"],
                text=r["text"],
                metadata=json.loads(r["metadata_json"] or "{}"),
                scores={},
            )
        return out

    def _bm25_top(self, query: str, k: int) -> list[tuple[str, float]]:
        self._ensure_loaded()
        q_tokens = tokenise(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # Top-k with index
        import heapq
        top = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])
        return [(self._chunk_ids_ordered[i], float(scores[i])) for i in top if scores[i] > 0]

    def _dense_top(self, query: str, k: int) -> list[tuple[str, float]]:
        self._ensure_loaded()
        if self._chroma_collection is None:
            return []
        try:
            embedding = llm_config.embed([query], model=EMBED_MODEL, operation="query-embed")[0]
        except budget.BudgetExceeded:
            log.warning("dense retrieval skipped — budget cap")
            return []
        result = self._chroma_collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        dists = result.get("distances", [[]])[0]
        # Chroma cosine returns distance = 1 - similarity. Convert.
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, dists)]

    def _rrf_fuse(self, *rankings: list[tuple[str, float]], k_rrf: int = 60) -> dict[str, float]:
        """Reciprocal Rank Fusion. Returns chunk_id -> fused score."""
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, (cid, _) in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
        return fused

    def _apply_weights(self, hits: dict[str, Hit], intent: str) -> None:
        """Multiply each hit's `fused` score by source-type weight for the intent.

        Hits flagged with `named_provision` or `named_authority` were
        explicitly chosen by the planner; do not let intent re-weighting
        push a planner-named judgment below a non-named statute. They
        keep their synthetic high score (10.0) untouched.
        """
        weights = WEIGHTS.get(intent, WEIGHTS["general"])
        for h in hits.values():
            if h.scores.get("named_provision") or h.scores.get("named_authority"):
                h.scores["weight"] = 1.0
                h.scores["fused_weighted"] = max(
                    h.scores.get("fused_weighted", 0.0), 10.0,
                )
                continue
            w = weights.get(h.source_type, 1.0)
            h.scores["weight"] = w
            if "fused" in h.scores:
                h.scores["fused_weighted"] = h.scores["fused"] * w
            elif "bm25" in h.scores:
                h.scores["fused_weighted"] = h.scores["bm25"] * w
            elif "dense" in h.scores:
                h.scores["fused_weighted"] = h.scores["dense"] * w

    def retrieve(
        self,
        query: str,
        k: int = 15,
        mode: Literal["bm25", "dense", "hybrid"] = "bm25",
        intent: str = "general",
        candidate_pool: int = 50,
    ) -> tuple[list[Hit], dict]:
        """Return top-k hits and a diagnostics dict."""
        self._ensure_loaded()
        diagnostics = {"mode": mode, "intent": intent, "query": query}

        bm25_results: list[tuple[str, float]] = []
        dense_results: list[tuple[str, float]] = []

        if mode in ("bm25", "hybrid"):
            bm25_results = self._bm25_top(query, candidate_pool)
            diagnostics["top_bm25_score"] = bm25_results[0][1] if bm25_results else 0.0
        if mode in ("dense", "hybrid"):
            dense_results = self._dense_top(query, candidate_pool)
            diagnostics["top_dense_score"] = dense_results[0][1] if dense_results else 0.0

        # Refusal check
        refused = False
        if mode == "bm25":
            refused = (not bm25_results) or bm25_results[0][1] < REFUSAL_BM25_MIN
        elif mode == "dense":
            refused = (not dense_results) or dense_results[0][1] < REFUSAL_DENSE_MIN
        else:  # hybrid
            top_bm = bm25_results[0][1] if bm25_results else 0.0
            top_d = dense_results[0][1] if dense_results else 0.0
            refused = (top_bm < REFUSAL_BM25_MIN) and (top_d < REFUSAL_DENSE_MIN)

        diagnostics["refused"] = refused

        # Build candidate hit objects with their per-method scores
        all_ids: set[str] = set()
        for cid, _ in bm25_results:
            all_ids.add(cid)
        for cid, _ in dense_results:
            all_ids.add(cid)
        chunk_objs = self._fetch_chunks(list(all_ids))

        for cid, score in bm25_results:
            if cid in chunk_objs:
                chunk_objs[cid].scores["bm25"] = score
        for cid, score in dense_results:
            if cid in chunk_objs:
                chunk_objs[cid].scores["dense"] = score

        # Fusion
        if mode == "hybrid":
            fused_scores = self._rrf_fuse(bm25_results, dense_results)
            for cid, s in fused_scores.items():
                if cid in chunk_objs:
                    chunk_objs[cid].scores["fused"] = s
        elif mode == "bm25":
            for cid, score in bm25_results:
                if cid in chunk_objs:
                    chunk_objs[cid].scores["fused"] = score
        else:  # dense
            for cid, score in dense_results:
                if cid in chunk_objs:
                    chunk_objs[cid].scores["fused"] = score

        # Apply weights
        self._apply_weights(chunk_objs, intent)

        # Final rank by fused_weighted (or fused if no weight)
        def sort_key(h: Hit) -> float:
            return h.scores.get("fused_weighted", h.scores.get("fused", 0.0))

        ranked = sorted(chunk_objs.values(), key=sort_key, reverse=True)[:k]
        for i, h in enumerate(ranked):
            h.rank = i + 1

        diagnostics["n_returned"] = len(ranked)
        return ranked, diagnostics

    # ------------------------------------------------------------------
    # Knowledge-augmented retrieval channels (Phase 4 of the upgrade)
    # ------------------------------------------------------------------

    def retrieve_named_provisions(
        self,
        provisions: list[str],
        *,
        max_chunks_per_provision: int = 4,
    ) -> tuple[list[Hit], dict]:
        """Resolve planner-named provisions deterministically against the
        name index, then materialise the underlying chunks. No scoring —
        the planner asked for these by name, so they go in.

        Returns (hits, diag). Each hit gets a synthetic high `fused_weighted`
        so when merged with the hybrid pool it sits near the top, but the
        downstream re-rank still respects source-type weights.
        """
        self._ensure_loaded()
        diag = {"requested": list(provisions), "resolved": [], "missed": []}
        if not provisions:
            return [], diag

        wanted_chunk_ids: list[str] = []
        seen: set[str] = set()
        for raw in provisions:
            match = name_index.lookup_provision(raw)
            if not match:
                diag["missed"].append(raw)
                continue
            diag["resolved"].append({
                "input": raw,
                "provision_key": match.provision_key,
                "n_chunks": len(match.chunk_ids),
            })
            for cid in match.chunk_ids[:max_chunks_per_provision]:
                if cid not in seen:
                    wanted_chunk_ids.append(cid)
                    seen.add(cid)

        chunk_objs = self._fetch_chunks(wanted_chunk_ids)
        hits: list[Hit] = []
        for cid in wanted_chunk_ids:
            h = chunk_objs.get(cid)
            if h is None:
                continue
            # Mark this hit as planner-injected. fused_weighted sentinel of
            # 10.0 is well above any organic BM25/RRF score so it survives
            # a top-k truncation in the merge step.
            h.scores = {"named_provision": 10.0, "fused_weighted": 10.0}
            hits.append(h)
        diag["n_returned"] = len(hits)
        return hits, diag

    def retrieve_named_authorities(
        self,
        authorities: list[str],
        *,
        max_chunks_per_case: int = 12,
        max_total_chunks: int = 36,
    ) -> tuple[list[Hit], dict]:
        """Resolve planner-named cases against the case_index. For each
        match, pull the first N chunks of that judgment (they include the
        headnote / orders / reasoning kernel).

        Returns (hits, diag). Like provisions, hits get a synthetic high
        score so they survive merge. The answerer prompt explicitly tells
        the model to lead with named authorities when applicable.
        """
        self._ensure_loaded()
        diag = {"requested": list(authorities), "resolved": [], "missed": []}
        if not authorities:
            return [], diag

        wanted_chunk_ids: list[str] = []
        seen: set[str] = set()
        for raw in authorities:
            match = name_index.lookup_case(raw)
            if not match:
                diag["missed"].append(raw)
                continue
            diag["resolved"].append({
                "input": raw,
                "case_id": match.case_id,
                "case_name": match.case_name,
                "confidence": match.confidence,
                "n_chunks_total": len(match.chunk_ids),
            })
            for cid in match.chunk_ids[:max_chunks_per_case]:
                if cid not in seen:
                    wanted_chunk_ids.append(cid)
                    seen.add(cid)
            if len(wanted_chunk_ids) >= max_total_chunks:
                break

        wanted_chunk_ids = wanted_chunk_ids[:max_total_chunks]
        chunk_objs = self._fetch_chunks(wanted_chunk_ids)
        hits: list[Hit] = []
        for cid in wanted_chunk_ids:
            h = chunk_objs.get(cid)
            if h is None:
                continue
            h.scores = {"named_authority": 10.0, "fused_weighted": 10.0}
            hits.append(h)
        diag["n_returned"] = len(hits)
        return hits, diag

    def retrieve_three_channel(
        self,
        queries: list[str],
        intent: str,
        named_provisions: list[str],
        named_authorities: list[str],
        *,
        k: int = 12,
        hybrid_per_query_k: int = 12,
        candidate_pool: int = 50,
    ) -> tuple[list[Hit], dict]:
        """Run all three retrieval channels and return a deduplicated, merged
        and re-ranked list of up to k hits.

        Channel 1: hybrid (BM25 + dense + RRF + source-type weights), per
                   reformulated query, merged.
        Channel 2: named provisions resolved via name_index.lookup_provision.
        Channel 3: named authorities resolved via name_index.lookup_case.

        Channels 2 and 3 take precedence at merge time (synthetic high
        score), but never displace hybrid hits beyond k — the final list
        is a re-rank by `fused_weighted` of the union, capped at k.
        """
        self._ensure_loaded()
        merged: dict[str, Hit] = {}
        any_refused = False
        last_diag: dict | None = None
        per_query_diag: list[dict] = []

        # ---- Channel 1: hybrid over each reformulated query ----
        for q in (queries or [])[:4]:
            hits, diag = self.retrieve(
                q, k=hybrid_per_query_k, mode="hybrid",
                intent=intent, candidate_pool=candidate_pool,
            )
            per_query_diag.append({"query": q, **{k_: v for k_, v in diag.items() if k_ != "query"}})
            last_diag = diag
            if diag.get("refused"):
                any_refused = True
                continue
            for h in hits:
                if h.chunk_id not in merged or (
                    h.scores.get("fused_weighted", 0)
                    > merged[h.chunk_id].scores.get("fused_weighted", 0)
                ):
                    merged[h.chunk_id] = h

        # ---- Channel 2: named provisions ----
        prov_hits, prov_diag = self.retrieve_named_provisions(named_provisions)
        for h in prov_hits:
            if h.chunk_id in merged:
                # Mark existing hit as also matched by name-index (helps debugging)
                merged[h.chunk_id].scores["named_provision"] = 10.0
                merged[h.chunk_id].scores["fused_weighted"] = max(
                    merged[h.chunk_id].scores.get("fused_weighted", 0.0), 10.0,
                )
            else:
                merged[h.chunk_id] = h

        # ---- Channel 3: named authorities ----
        case_hits, case_diag = self.retrieve_named_authorities(named_authorities)
        for h in case_hits:
            if h.chunk_id in merged:
                merged[h.chunk_id].scores["named_authority"] = 10.0
                merged[h.chunk_id].scores["fused_weighted"] = max(
                    merged[h.chunk_id].scores.get("fused_weighted", 0.0), 10.0,
                )
            else:
                merged[h.chunk_id] = h

        # Refusal: only refuse if hybrid refused AND no name-channel rescue
        refused = any_refused and not merged

        # Re-rank merged set by fused_weighted, top-k. Apply intent weights to
        # name-injected chunks too so e.g. a "case_law" intent re-rank can
        # promote case-law hits within the top-k.
        self._apply_weights(merged, intent)
        ranked = sorted(
            merged.values(),
            key=lambda h: h.scores.get("fused_weighted", 0.0),
            reverse=True,
        )[:k]
        for i, h in enumerate(ranked):
            h.rank = i + 1

        diagnostics = {
            "intent": intent,
            "n_queries": len(queries or []),
            "per_query": per_query_diag,
            "named_provisions": prov_diag,
            "named_authorities": case_diag,
            "n_merged_pre_rerank": len(merged),
            "n_returned": len(ranked),
            "refused": refused,
        }
        return ranked, diagnostics
