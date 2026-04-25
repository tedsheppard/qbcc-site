"""BIF Act guide embedded index.

Public surface:

    from services.contract_assist.bif_act_index import (
        build_index,
        retrieve_bif,
        register_startup,
    )

``build_index`` is idempotent and intended to be called once at server
startup. ``retrieve_bif`` is a thin similarity-search API over the
persistent ChromaDB collection produced by ``build_index``.
``register_startup(app)`` wires the build into a FastAPI startup hook
and runs it in a background thread so the server boot isn't blocked.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from .builder import build_index, is_built
from .retriever import retrieve_bif

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI


log = logging.getLogger("contract_assist.bif_act_index")

__all__ = ["build_index", "retrieve_bif", "register_startup", "is_built"]


def register_startup(app: "FastAPI") -> None:
    """Register a FastAPI startup hook that builds the BIF Act index.

    Build runs in a background thread so a slow embedding round-trip
    cannot block the server from accepting requests. The build is
    idempotent — if the index is already up to date on disk, the hook
    is effectively a no-op.
    """

    @app.on_event("startup")
    def _start_bif_index_build() -> None:  # pragma: no cover - exercised at runtime
        def _runner() -> None:
            try:
                build_index()
            except Exception:
                log.exception("BIF Act index build failed at startup")

        t = threading.Thread(
            target=_runner, name="bif-act-index-build", daemon=True
        )
        t.start()
