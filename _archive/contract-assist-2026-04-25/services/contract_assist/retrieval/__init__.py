"""Session-scoped retrieval pipeline for /contract-assist.

Public surface (re-exported for convenience):

    from services.contract_assist.retrieval import ingest, retrieve, clear

The pipeline ingests an uploaded construction contract, chunks and embeds
it with OpenAI ``text-embedding-3-small``, and stores the result in a
session-scoped, in-memory ChromaDB collection. ``retrieve`` performs a
hybrid (vector + clause + keyword) search and returns the top-k chunks
ordered by combined score. ``clear`` purges a session.

All non-stdlib imports are deferred inside the implementation modules so
that ``import services.contract_assist.retrieval`` is cheap.
"""

from .service import ingest, retrieve, clear

__all__ = ["ingest", "retrieve", "clear"]
