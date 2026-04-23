"""QBCC Licensed Contractors Register lookup via the Queensland Government CKAN API.

Per spec Section 6: search is proxied through the backend so we can rate-limit
and cache, never expose CKAN directly to the browser. The register reflects
CURRENT status at dataset last update — not historical status at the time the
work was performed. Callers should display the STALE_NOTICE alongside results.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

log = logging.getLogger("claim_check.qbcc")

CKAN_BASE = "https://www.data.qld.gov.au/api/3/action"
PACKAGE_ID = "qbcc-licensed-contractors-register"

RESOURCE_CACHE_TTL = 24 * 60 * 60      # 24 hours
LOOKUP_CACHE_TTL = 6 * 60 * 60         # 6 hours
HTTP_TIMEOUT = 6                        # seconds

STALE_NOTICE = (
    "QBCC registry lookup verifies CURRENT licence status only, as at the dataset's "
    "last update on data.qld.gov.au. BIF Act s 75(2) is about licensing at the time "
    "the work was carried out; for disputes about historical licensing, independent "
    "verification is recommended."
)


class QBCCUnavailable(RuntimeError):
    pass


_resource_cache: dict[str, Any] = {"id": None, "fetched_at": 0.0}
_lookup_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _get_resource_id(force_refresh: bool = False) -> str:
    with _lock:
        if (
            not force_refresh
            and _resource_cache["id"]
            and (_now() - _resource_cache["fetched_at"]) < RESOURCE_CACHE_TTL
        ):
            return _resource_cache["id"]

    try:
        import requests
    except Exception as e:
        raise QBCCUnavailable(f"requests library unavailable: {e}")

    try:
        resp = requests.get(
            f"{CKAN_BASE}/package_show",
            params={"id": PACKAGE_ID},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise QBCCUnavailable(f"CKAN package_show failed: {e}")

    if not data.get("success"):
        raise QBCCUnavailable("CKAN package_show returned success=false")

    resources = (data.get("result") or {}).get("resources") or []
    # Prefer active + datastore_active, latest by last_modified.
    candidates = [
        r for r in resources
        if (r.get("state") or "active") == "active"
        and r.get("datastore_active")
    ]
    if not candidates:
        # Fallback: any active CSV resource.
        candidates = [
            r for r in resources
            if (r.get("state") or "active") == "active"
            and (r.get("format") or "").lower() in ("csv", "json")
        ]
    if not candidates:
        raise QBCCUnavailable("No active/datastore resource found in CKAN package.")

    def _mod_key(r: dict[str, Any]) -> str:
        return r.get("last_modified") or r.get("revision_timestamp") or r.get("created") or ""

    candidates.sort(key=_mod_key, reverse=True)
    rid = candidates[0].get("id")
    if not rid:
        raise QBCCUnavailable("Selected CKAN resource has no id.")

    with _lock:
        _resource_cache["id"] = rid
        _resource_cache["fetched_at"] = _now()

    return rid


def _normalise_key(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def search(q: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search the QBCC register by name / ABN / licence number."""
    q = (q or "").strip()
    if not q:
        return []

    cache_key = f"{_normalise_key(q)}::{limit}"
    with _lock:
        cached = _lookup_cache.get(cache_key)
        if cached and (_now() - cached[0]) < LOOKUP_CACHE_TTL:
            return cached[1]

    try:
        import requests
    except Exception as e:
        raise QBCCUnavailable(f"requests library unavailable: {e}")

    try:
        rid = _get_resource_id()
    except QBCCUnavailable:
        raise

    def _run(rid_: str) -> list[dict[str, Any]]:
        resp = requests.get(
            f"{CKAN_BASE}/datastore_search",
            params={"resource_id": rid_, "q": q, "limit": min(max(int(limit), 1), 50)},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success"):
            raise QBCCUnavailable("CKAN datastore_search returned success=false")
        records = (payload.get("result") or {}).get("records") or []
        return records

    try:
        records = _run(rid)
    except Exception as e:
        # Retry once with a fresh resource id (the dataset may have rotated).
        log.info("datastore_search failed (%s); refreshing resource id and retrying", e)
        try:
            rid = _get_resource_id(force_refresh=True)
            records = _run(rid)
        except Exception as e2:
            raise QBCCUnavailable(f"CKAN datastore_search failed: {e2}")

    results = [_shape_record(r) for r in records]

    with _lock:
        _lookup_cache[cache_key] = (_now(), results)

    return results


def _shape_record(r: dict[str, Any]) -> dict[str, Any]:
    """Normalise a CKAN record into a shape the frontend can render.

    Field names on the dataset vary between publisher revisions. We probe the
    likely candidates and fall back to the raw record as ``raw``.
    """
    def pick(*keys: str) -> str:
        for k in keys:
            for kk in r.keys():
                if kk.lower() == k.lower():
                    v = r.get(kk)
                    if v in (None, ""):
                        continue
                    return str(v).strip()
        return ""

    entity_name = pick("Licence Name", "Entity Name", "Licensee Name", "Name", "Licensee")
    trading_name = pick("Trading Name", "Business Name")
    licence_number = pick("Licence Number", "Licence No", "Licence #", "Licence")
    licence_status = pick("Licence Status", "Status")
    classes_raw = pick("Licence Classes", "Licence Class", "Classes")
    expiry = pick("Expiry Date", "Expiry", "Expires")
    abn = pick("ABN")
    postcode = pick("Postcode", "Post Code")

    classes = [c.strip() for c in (classes_raw or "").split(",") if c.strip()]

    return {
        "entity_name": entity_name,
        "trading_name": trading_name,
        "licence_number": licence_number,
        "licence_status": licence_status,
        "licence_classes": classes,
        "expiry": expiry,
        "abn": abn,
        "postcode": postcode,
        "display": entity_name + (f" (trading as {trading_name})" if trading_name and trading_name != entity_name else ""),
    }
