"""Bounded-concurrency fan-out for case-reader workers.

Runs `case_reader.read_case_for_question` across multiple judgments at
once, capped at MAX_CONCURRENT_READERS workers. Output preserves input
order. A failing worker returns a safe stub rather than crashing the
whole fan-out — the reasoner downstream sees the failure as
is_on_point=false with a note explaining what went wrong.

The OpenAI / Anthropic SDKs are thread-safe, so a ThreadPoolExecutor is
sufficient — no asyncio needed. Total wall-clock approximates the
latency of the slowest single read (modulo the concurrency cap), not
the sum.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from . import case_reader

log = logging.getLogger("bif_research.parallel_reader")

MAX_CONCURRENT_READERS = 4


def _safe_stub(case_id: str, err: BaseException) -> dict:
    return {
        "case_id": case_id,
        "case_name": "",
        "is_on_point": False,
        "extracts": [],
        "follows": [],
        "distinguishes": [],
        "notes": f"read failed: {err}",
    }


def _read_one(case_id: str, question: str, planner_notes: str) -> dict:
    t0 = time.monotonic()
    log.info("reader worker start case_id=%s", case_id)
    try:
        result = case_reader.read_case_for_question(case_id, question, planner_notes)
    except Exception as e:
        elapsed = time.monotonic() - t0
        log.warning(
            "reader worker end case_id=%s elapsed=%.2fs FAILED (%s)",
            case_id, elapsed, e,
        )
        return _safe_stub(case_id, e)
    elapsed = time.monotonic() - t0
    log.info("reader worker end case_id=%s elapsed=%.2fs", case_id, elapsed)
    return result


def parallel_read_cases(
    case_ids: list[str],
    question: str,
    planner_notes: str = "",
) -> list[dict]:
    """Run reader workers across `case_ids` with bounded concurrency.

    Returns reader outputs in the SAME ORDER as the input list. Duplicate
    case_ids in the input are de-duplicated (first occurrence wins) before
    fan-out so the same judgment isn't read twice. Worker failures are
    surfaced as safe stubs with is_on_point=false.
    """
    if not case_ids:
        return []

    # Preserve first-seen order while dropping duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for cid in case_ids:
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(cid)

    results: list[dict | None] = [None] * len(deduped)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_READERS) as executor:
        future_to_index = {
            executor.submit(_read_one, cid, question, planner_notes): i
            for i, cid in enumerate(deduped)
        }
        for future in future_to_index:
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception as e:
                # _read_one already swallows exceptions; this is belt-and-braces.
                results[i] = _safe_stub(deduped[i], e)

    return [r for r in results if r is not None]
