"""On-demand retrieval from legal_corpus/ for the chatbot.

The chatbot exposes `search_legislation(query, scope)` as an OpenAI tool.
This module owns the vector index under services/claim_check/legal_index/
and ranks BIF Act hits first per legal_corpus/README.md.

Stage 9 will implement indexing and query.
"""

from __future__ import annotations

SCOPES = ("bif_act", "annotated_bif_act", "bif_regs", "qbcc_act", "qbcc_regs", "aia_act", "all")


def search_legislation(query: str, scope: str = "all", limit: int = 6) -> list[dict]:
    raise NotImplementedError("retrieval.search_legislation — implemented in stage 9")
