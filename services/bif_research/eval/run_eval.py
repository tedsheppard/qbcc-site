"""Eval harness for bif_research.

Loads eval_set.jsonl, runs each question through a Pipeline, scores the
output on five dimensions, writes per-run results and prints aggregates.

Pipeline contract:
    pipeline.answer(question: str, history: list = []) -> dict
        returns:
            {
              "propositions": [
                  {"text": "...", "citations": ["chunk_id", ...],
                   "quotes": [{"chunk_id": "...", "text": "..."}]},
                  ...
              ],
              "answer_summary": "...",
              "answer_html": "...",
              "confidence": "high" | "medium" | "low",
              "refused": false,
              "sources": [{"id": "chunk_id", "metadata": {...}, "text": "..."}, ...]
            }

Usage:
    python -m services.bif_research.eval.run_eval [--pipeline stub|naive|full]
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET = EVAL_DIR / "eval_set.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


def load_eval_set() -> list[dict]:
    return [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

class StubPipeline:
    """Returns an empty answer. Used to confirm the harness produces 0%."""
    name = "stub"

    def answer(self, question: str, history: list | None = None) -> dict:
        return {
            "propositions": [],
            "answer_summary": "stub",
            "answer_html": "stub",
            "confidence": "low",
            "refused": False,
            "sources": [],
        }


def load_pipeline(name: str):
    if name == "stub":
        return StubPipeline()
    if name == "naive":
        from services.bif_research.pipeline import NaivePipeline  # noqa: F401
        return NaivePipeline()
    if name == "full":
        from services.bif_research.pipeline import FullPipeline  # noqa: F401
        return FullPipeline()
    raise ValueError(f"Unknown pipeline: {name}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _match_expected_source(expected: dict, sources: list[dict]) -> bool:
    """Does `expected` match any of the `sources` (the chunks cited in the answer)?

    Match rules:
      - statute / regulation: act + section both match the source metadata
      - case: name_pattern (case-insensitive) appears in case_name
      - decision: decision_id matches
      - annotated: annotated_section matches
    """
    typ = expected.get("type", "")
    if typ in ("statute", "regulation"):
        exp_act = (expected.get("act") or "").lower().replace(" ", "")
        exp_sec = (expected.get("section") or "").lower().replace(" ", "")
        for src in sources:
            md = src.get("metadata", {}) or {}
            src_act = (md.get("act_short") or md.get("act_name") or "").lower().replace(" ", "")
            src_sec = (md.get("section_number") or "").lower().replace(" ", "")
            if exp_act and exp_sec and exp_act in src_act and exp_sec == src_sec:
                return True
        return False
    if typ == "case":
        pat = (expected.get("name_pattern") or "").lower()
        if not pat:
            return False
        for src in sources:
            md = src.get("metadata", {}) or {}
            case_name = (md.get("case_name") or "").lower()
            if pat in case_name:
                return True
        return False
    if typ == "decision":
        ref = (expected.get("decision_id") or "").lower()
        if not ref:
            return False
        for src in sources:
            md = src.get("metadata", {}) or {}
            if ref in (md.get("decision_id") or "").lower():
                return True
        return False
    if typ == "annotated":
        exp_sec = (expected.get("annotated_section") or "").lower().replace(" ", "")
        for src in sources:
            md = src.get("metadata", {}) or {}
            src_type = (md.get("source_type") or "").lower()
            src_sec = (md.get("section_number") or "").lower().replace(" ", "")
            if "annotated" in src_type and exp_sec == src_sec:
                return True
        return False
    return False


def score_question(q: dict, result: dict) -> dict:
    """Score one question. Returns a dict of scores."""
    propositions = result.get("propositions", []) or []
    sources = result.get("sources", []) or []
    refused = bool(result.get("refused", False))

    # Build the set of cited chunk_ids
    cited_ids: set[str] = set()
    for p in propositions:
        for c in p.get("citations", []) or []:
            cited_ids.add(c)

    source_ids = {s.get("id"): s for s in sources}

    # citation_precision: % of cited ids that resolve to a real source in `sources`
    if cited_ids:
        resolved = sum(1 for cid in cited_ids if cid in source_ids)
        citation_precision = 100.0 * resolved / len(cited_ids)
    else:
        # If no propositions cite anything but the system also refused, that's fine
        citation_precision = 100.0 if refused else 0.0

    # citation_recall: % of expected sources that appear in cited_ids' source metadata
    cited_sources = [source_ids[cid] for cid in cited_ids if cid in source_ids]
    expected_sources = q.get("expected_sources", []) or []
    if expected_sources and not refused:
        hits = sum(1 for exp in expected_sources if _match_expected_source(exp, cited_sources))
        citation_recall = 100.0 * hits / len(expected_sources)
    else:
        # If refused, recall is moot; mark as 0 unless refusal is itself appropriate
        citation_recall = 0.0

    # quote_fidelity: every quote.text must be a verbatim substring of its chunk.text
    total_quotes = 0
    fidelity_hits = 0
    for p in propositions:
        for q_obj in p.get("quotes", []) or []:
            total_quotes += 1
            chunk_id = q_obj.get("chunk_id", "")
            quote_text = q_obj.get("text", "") or ""
            src = source_ids.get(chunk_id)
            if not src:
                continue  # already caught by precision
            chunk_text = src.get("text", "") or ""
            # Tolerate whitespace differences
            normalised_chunk = " ".join(chunk_text.split())
            normalised_quote = " ".join(quote_text.split())
            if normalised_quote and normalised_quote in normalised_chunk:
                fidelity_hits += 1
    if total_quotes > 0:
        quote_fidelity = 100.0 * fidelity_hits / total_quotes
    else:
        quote_fidelity = 100.0  # no quotes -> nothing to fail

    # must_not_contain: scan answer_summary + propositions for forbidden strings
    must_not = q.get("must_not_contain", []) or []
    answer_text = " ".join(
        [result.get("answer_summary", "") or ""]
        + [(p.get("text", "") or "") for p in propositions]
    )
    forbidden_hits = [s for s in must_not if s.lower() in answer_text.lower()]

    # refusal_appropriate: did the system refuse and was that appropriate?
    # For these eval questions, refusal is NEVER appropriate (every question has expected sources)
    refusal_appropriate = (not refused)

    return {
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "quote_fidelity": quote_fidelity,
        "forbidden_hits": forbidden_hits,
        "refused": refused,
        "refusal_appropriate": refusal_appropriate,
        "n_propositions": len(propositions),
        "n_quotes": total_quotes,
        "n_cited_chunks": len(cited_ids),
    }


def judge_relevance(question: str, expected: str, actual_summary: str) -> float:
    """Use a small model to judge relevance 0-5. Returns 0.0 on stub answers."""
    if not actual_summary or actual_summary == "stub":
        return 0.0
    try:
        from services.bif_research import llm_config
    except Exception:
        return 0.0
    prompt = (
        "You are a strict legal-research evaluator. Given a question, an expected answer summary, "
        "and an actual answer, score the actual answer on a 0-5 scale for whether it captures the "
        "key propositions of the expected answer. 5 = fully captures all key propositions; "
        "3 = captures most; 1 = mentions the topic but misses the substance; 0 = wrong or missing.\n\n"
        f"QUESTION:\n{question}\n\nEXPECTED:\n{expected}\n\nACTUAL:\n{actual_summary}\n\n"
        "Respond with ONLY a single number 0-5 (decimals allowed)."
    )
    try:
        text, _ = llm_config.complete_chat(
            [{"role": "user", "content": prompt}],
            kind="default",
            operation="eval-judge",
            max_output_tokens=10,
        )
        return max(0.0, min(5.0, float(text.strip())))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(pipeline_name: str = "stub", judge: bool = True, label: str = "") -> Path:
    questions = load_eval_set()
    pipeline = load_pipeline(pipeline_name)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {
        "started_at": now_utc(),
        "pipeline": pipeline.name if hasattr(pipeline, "name") else pipeline_name,
        "label": label,
        "n_questions": len(questions),
        "per_question": [],
    }
    cat_totals: dict[str, dict[str, float]] = {}
    agg_keys = ("citation_precision", "citation_recall", "quote_fidelity")

    for q in questions:
        t0 = time.time()
        try:
            result = pipeline.answer(q["question"])
        except Exception as e:
            result = {"propositions": [], "answer_summary": f"ERROR: {e}", "sources": [], "refused": False}
        elapsed_ms = int((time.time() - t0) * 1000)

        scores = score_question(q, result)
        relevance = judge_relevance(
            q["question"], q.get("expected_answer_summary", ""), result.get("answer_summary", "")
        ) if judge else 0.0
        scores["answer_relevance"] = relevance
        scores["elapsed_ms"] = elapsed_ms

        out["per_question"].append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "scores": scores,
            "result_summary": (result.get("answer_summary") or "")[:300],
            "n_sources": len(result.get("sources", []) or []),
        })

        cat = q["category"]
        cat_totals.setdefault(cat, {k: 0.0 for k in (*agg_keys, "answer_relevance", "n")})
        for k in agg_keys:
            cat_totals[cat][k] += scores[k]
        cat_totals[cat]["answer_relevance"] += relevance
        cat_totals[cat]["n"] += 1

    # aggregate
    aggregate = {k: 0.0 for k in (*agg_keys, "answer_relevance")}
    for q in out["per_question"]:
        for k in agg_keys:
            aggregate[k] += q["scores"][k]
        aggregate["answer_relevance"] += q["scores"]["answer_relevance"]
    n = max(1, len(out["per_question"]))
    aggregate = {k: v / n for k, v in aggregate.items()}
    aggregate["answer_relevance_5"] = aggregate.pop("answer_relevance")
    aggregate["overall_pct"] = (
        aggregate["citation_precision"] * 0.30
        + aggregate["citation_recall"] * 0.30
        + aggregate["quote_fidelity"] * 0.20
        + (aggregate["answer_relevance_5"] / 5 * 100) * 0.20
    )
    out["aggregate"] = aggregate

    # category breakdown
    cat_summary = {}
    for cat, vals in cat_totals.items():
        n_c = max(1, vals["n"])
        cat_summary[cat] = {
            "n": int(vals["n"]),
            "citation_precision": vals["citation_precision"] / n_c,
            "citation_recall": vals["citation_recall"] / n_c,
            "quote_fidelity": vals["quote_fidelity"] / n_c,
            "answer_relevance_5": vals["answer_relevance"] / n_c,
        }
    out["category_summary"] = cat_summary

    out["finished_at"] = now_utc()

    # write
    label_suffix = f"-{label}" if label else ""
    out_path = RESULTS_DIR / f"{out['started_at']}-{pipeline_name}{label_suffix}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def print_aggregate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    a = data["aggregate"]
    print(f"\n=== Eval results: {path.name} ===")
    print(f"Pipeline: {data['pipeline']}  Questions: {data['n_questions']}")
    print(f"  citation_precision: {a['citation_precision']:.1f}%")
    print(f"  citation_recall:    {a['citation_recall']:.1f}%")
    print(f"  quote_fidelity:     {a['quote_fidelity']:.1f}%")
    print(f"  answer_relevance:   {a['answer_relevance_5']:.2f} / 5")
    print(f"  overall:            {a['overall_pct']:.1f}%")
    print("\nBy category:")
    for cat, s in sorted(data["category_summary"].items()):
        print(f"  {cat:24s}  n={s['n']:2d}  prec={s['citation_precision']:5.1f}%  "
              f"rec={s['citation_recall']:5.1f}%  fid={s['quote_fidelity']:5.1f}%  "
              f"rel={s['answer_relevance_5']:.2f}/5")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="stub", choices=["stub", "naive", "full"])
    ap.add_argument("--no-judge", action="store_true",
                    help="Skip the relevance-judge LLM call (free but no relevance score)")
    ap.add_argument("--label", default="", help="Optional label appended to results filename")
    args = ap.parse_args()
    out_path = run(args.pipeline, judge=not args.no_judge, label=args.label)
    print_aggregate(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
