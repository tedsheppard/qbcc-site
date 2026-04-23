"""Meilisearch-backed case citation surfacing for /claim-check.

Queries the existing Sopal adjudication-decisions Meilisearch index to find
on-point decisions for a given check query. Matches the HTTP pattern used
elsewhere in server.py (requests + Bearer auth on MEILI_URL / MEILI_KEY).

Read-only. Never writes.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("claim_check.case_law")


def relevant_decisions(query: str, limit: int = 5) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    meili_url = os.getenv("MEILI_URL")
    meili_key = os.getenv("MEILI_KEY")
    meili_index = os.getenv("MEILI_INDEX") or "decisions"
    if not meili_url:
        return []

    try:
        import requests  # already in requirements.txt
    except Exception:
        return []

    headers = {"Authorization": f"Bearer {meili_key}"} if meili_key else {}
    try:
        resp = requests.post(
            f"{meili_url.rstrip('/')}/indexes/{meili_index}/search",
            headers=headers,
            json={"q": query, "limit": max(1, min(int(limit), 10))},
            timeout=4,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("Meilisearch query failed for %r: %s", query, e)
        return []

    hits = (resp.json() or {}).get("hits") or []
    out: list[dict[str, Any]] = []
    for h in hits:
        out.append(
            {
                "id": h.get("ejs_id") or h.get("id") or "",
                "title": h.get("title") or h.get("case_name") or h.get("adjudicator_name") or "(untitled)",
                "snippet": (h.get("_formatted") or {}).get("content")
                or (h.get("content") or "")[:240],
                "url": f"/search?q={query}",
            }
        )
    return out
