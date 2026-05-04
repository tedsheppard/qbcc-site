"""Lightweight v2 eval runner for the knowledge-augmented pipeline.

Reads eval_set_v2.jsonl, runs each question through the live FullPipeline,
then scores against the expected fields:
  - must_cite_case        — case_id that should appear in the cited sources
  - must_cite_provisions  — provision_keys (e.g. "bif_act_s76") whose chunk
                            ids should appear among the cited sources
  - expected_lead         — free-text indicator of what the answer should
                            lead with (recorded for human review only)

Writes results/v2-knowledge-augmented.json. Designed to be cheap (~$2-3
total) and fast (a couple of minutes).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SET_PATH = ROOT / "eval_set_v2.jsonl"
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "v2-knowledge-augmented.json"


def _case_id_from_chunk_id(chunk_id: str, source_id: str) -> str:
    """Strip _p..._c... suffix to get base case_id like QCA_2024_94."""
    parts = source_id.split("_")
    keep = []
    for p in parts:
        if re.match(r"^p\d", p) or re.match(r"^c\d", p):
            break
        keep.append(p)
    return "_".join(keep)


def score_case_hit(result: dict, want_case_id: str) -> bool:
    """Did `want_case_id` appear in the cited sources of any proposition?"""
    cited_ids: set[str] = set()
    for p in result.get("propositions", []):
        for cid in p.get("citations", []):
            cited_ids.add(cid)
    # Map cited chunk_ids back to case_ids via sources list
    for s in result.get("sources", []):
        if s.get("id") in cited_ids:
            sid = s.get("metadata", {}).get("case_id") or ""
            if not sid:
                # derive from chunk_id by looking at source_id in metadata
                source_id = s.get("metadata", {}).get("source_file", "")
                # Or pull from the source chunk's source_id field — which we
                # don't have here. Fall back to header parse: case names are
                # already in `header` so we just check whether the citation
                # text matches the case_id pattern via scan over result['sources'].
                pass
            # Rebuild case_id from the chunk_id-source_id mapping isn't trivial
            # without re-querying. Trick: we know sources have metadata.citation
            # with the year/court/number. Match against want_case_id.
            citation = s.get("metadata", {}).get("citation", "") or s.get("header", "")
            # Quick test: case_id "QCA_2024_94" -> want "[2024] QCA 94" in citation
            parts = want_case_id.split("_")
            if len(parts) >= 3:
                court, year, num = parts[0], parts[1], parts[2]
                if f"[{year}] {court} {num}" in citation or f"[{year}] {court} {num}." in citation:
                    return True
    return False


def score_provision_hit(result: dict, want_keys: list[str]) -> dict:
    """For each desired provision_key, did any cited source come from it?"""
    from services.bif_research import name_index
    cited_ids: set[str] = set()
    for p in result.get("propositions", []):
        for cid in p.get("citations", []):
            cited_ids.add(cid)
    out = {}
    db = name_index._db()
    for key in want_keys:
        rows = db.execute(
            "SELECT chunk_id FROM provision_chunks WHERE provision_key=?", (key,)
        ).fetchall()
        provision_chunk_ids = {r["chunk_id"] for r in rows}
        out[key] = bool(cited_ids & provision_chunk_ids)
    return out


def main():
    if not SET_PATH.exists():
        print(f"missing {SET_PATH}", file=sys.stderr)
        sys.exit(1)
    RESULTS_DIR.mkdir(exist_ok=True)
    questions = [json.loads(line) for line in SET_PATH.read_text().splitlines() if line.strip()]
    print(f"loaded {len(questions)} questions")

    from services.bif_research.pipeline import FullPipeline
    p = FullPipeline(k_chunks=16, real_planner=True)

    results = []
    t_total = time.time()
    for q in questions:
        print(f"\n=== {q['id']} [{q['category']}] ===")
        print(f"Q: {q['question'][:100]}{'...' if len(q['question'])>100 else ''}")
        t0 = time.time()
        try:
            r = p.answer(q["question"])
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": q["id"], "category": q["category"],
                            "question": q["question"], "error": str(e)})
            continue

        score = {}
        if q.get("must_cite_case"):
            score["case_hit"] = score_case_hit(r, q["must_cite_case"])
        if q.get("must_cite_provisions"):
            score["provision_hits"] = score_provision_hit(r, q["must_cite_provisions"])
        record = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_lead": q.get("expected_lead", ""),
            "elapsed_s": round(elapsed, 1),
            "answer_summary": r.get("answer_summary", ""),
            "confidence": r.get("confidence"),
            "refused": r.get("refused"),
            "n_propositions": len(r.get("propositions", [])),
            "n_sources_cited": len(r.get("sources", [])),
            "planner": r.get("_planner", {}),
            "score": score,
        }
        results.append(record)
        # Pretty print mid-run
        print(f"  intent={record['planner'].get('intent')} conf={record['confidence']} "
              f"refused={record['refused']} elapsed={elapsed:.1f}s")
        print(f"  summary: {record['answer_summary'][:200]}")
        if score:
            print(f"  score:   {score}")

    elapsed_total = time.time() - t_total
    out = {
        "n_questions": len(results),
        "elapsed_total_s": round(elapsed_total, 1),
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}  ({elapsed_total:.0f}s total)")

    # Summary
    case_targets = [r for r in results if "case_hit" in r.get("score", {})]
    case_pass = sum(1 for r in case_targets if r["score"]["case_hit"])
    prov_targets = [r for r in results if "provision_hits" in r.get("score", {})]
    prov_pass = sum(1 for r in prov_targets if all(r["score"]["provision_hits"].values()))
    print(f"\ncase-hit: {case_pass}/{len(case_targets)}")
    print(f"provision-hit (all required): {prov_pass}/{len(prov_targets)}")


if __name__ == "__main__":
    main()
