"""End-to-end pipeline objects.

Two flavours:
  - NaivePipeline : BM25-only retrieval, no-op planner. Phase 4 baseline.
  - FullPipeline  : real planner, hybrid retrieval, source-type weighting,
                    conversation-aware query rewriting.

Both expose `answer(question, history=None)` where history is a list of
prior turns: [{"role": "user"|"assistant", "content": "..."}, ...]
"""
from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from . import answerer, planner, postprocessor, verifier, llm_config
from .retriever import Retriever, Hit

log = logging.getLogger("bif_research.pipeline")


def _format_hit_for_eval(h: Hit) -> dict:
    return {
        "id": h.chunk_id,
        "metadata": h.metadata | {"source_type": h.source_type, "header": h.header},
        "text": h.text,
        "header": h.header,
        "scores": h.scores,
    }


def _to_eval_format(answer: postprocessor.FinalAnswer) -> dict:
    """Serialise a FinalAnswer into the dict shape the eval harness expects."""
    propositions = []
    for p in answer.propositions:
        # Convert citation_indices (1-based) back to chunk_ids
        citation_chunk_ids = []
        for idx in p.citation_indices:
            if 1 <= idx <= len(answer.sources):
                citation_chunk_ids.append(answer.sources[idx - 1].chunk_id)
        quotes = [
            {"chunk_id": q.chunk_id, "text": q.text}
            for q in p.quotes if q.located and q.text
        ]
        propositions.append({
            "text": p.text,
            "citations": citation_chunk_ids,
            "quotes": quotes,
        })
    sources = [_format_hit_for_eval(h) for h in answer.sources]
    return {
        "propositions": propositions,
        "answer_summary": answer.answer_summary,
        "answer_html": answer.answer_html,
        "confidence": answer.confidence,
        "refused": answer.refused,
        "sources": sources,
        "flags": answer.flags,
    }


class NaivePipeline:
    """Phase 4 baseline: BM25-only retrieval, heuristic planner, real answerer."""
    name = "naive"

    def __init__(self, k_chunks: int = 15):
        self.retriever = Retriever()
        self.k = k_chunks

    def answer(self, question: str, history: list | None = None) -> dict:
        plan = planner.plan(question, real=False)
        # Use the (single) heuristic query
        query = plan.queries[0] if plan.queries else question
        hits, diag = self.retriever.retrieve(query, k=self.k, mode="bm25", intent=plan.intent)
        if diag.get("refused"):
            empty = postprocessor.FinalAnswer(
                propositions=[], sources=[],
                answer_summary="I could not find sufficient sources for this question.",
                confidence="low",
                answer_html=postprocessor._render_refusal(),
                refused=True, flags=["bm25 score below threshold"],
            )
            return _to_eval_format(empty)
        answer_json = answerer.compose(question, hits, history=history)
        final = postprocessor.process(answer_json, hits, refused=False)
        verified, _ = verifier.verify(final)
        return _to_eval_format(verified)


REWRITE_SYSTEM_PROMPT = """You are a conversation-aware query rewriter for a legal
research tool. Given the prior turns of a conversation and the user's
new question, produce a SELF-CONTAINED question that captures what the
user actually wants in the new turn.

Examples:

  Prior assistant turn discussed s 68 payment-claim wording requirements.
  New user turn: "so yes or no, is it sufficient?"
  Rewrite: "Under s 68 of the BIF Act, is it sufficient for a payment claim
            to state only 'this is a progress claim under the Building Industry
            Fairness (Security of Payment) Act 2017'?"

  Prior assistant turn discussed Brodyn / jurisdictional error.
  New user turn: "and how do Qld courts treat that?"
  Rewrite: "How have Queensland courts treated the Brodyn approach to
            jurisdictional error in adjudication decisions?"

If the new question is already self-contained (mentions sections, cases or
concepts on its own), return it unchanged.

Output ONLY the rewritten question text. No prose, no JSON, no quotes."""


def _rewrite_with_history(question: str, history: list[dict] | None) -> str:
    """If there's prior conversation, rewrite the question to be self-contained.
    Falls back to the original question on any error."""
    if not history:
        return question
    # Take the last 4 turns max (2 user + 2 assistant) — enough context, not too much
    recent = history[-4:]
    history_text_parts = []
    for h in recent:
        role = h.get("role", "")
        content = h.get("content", "")
        if isinstance(content, dict):
            # Assistant turns store the structured answer dict; pull summary
            content = content.get("answer_summary") or content.get("text") or ""
        if not content:
            continue
        history_text_parts.append(f"{role.upper()}: {str(content)[:600]}")
    if not history_text_parts:
        return question
    history_text = "\n\n".join(history_text_parts)
    user_msg = (
        f"PRIOR TURNS:\n{history_text}\n\n"
        f"NEW USER QUESTION:\n{question}\n\n"
        f"Rewrite the new question as self-contained:"
    )
    try:
        text, _ = llm_config.complete_chat(
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            kind="default",
            operation="query-rewrite",
            max_output_tokens=200,
        )
        rewritten = (text or "").strip().strip('"').strip("'")
        if rewritten and len(rewritten) < 1000:
            return rewritten
    except Exception:
        pass
    return question


class LegacyFullPipeline:
    """Pre-upgrade: heuristic-or-LLM planner, hybrid retrieval, source-type
    weighting, multi-query reformulation, conversation-aware query rewriting.

    Retained so we can A/B against the upgraded knowledge-augmented pipeline.
    """
    name = "legacy_full"

    def __init__(self, k_chunks: int = 15, mode: str = "hybrid", real_planner: bool = True):
        self.retriever = Retriever()
        self.k = k_chunks
        self.mode = mode
        self.real_planner = real_planner

    def answer(self, question: str, history: list | None = None) -> dict:
        rewritten = _rewrite_with_history(question, history)
        plan = planner.plan(rewritten, real=self.real_planner)
        all_hits: dict[str, Hit] = {}
        any_refused = False
        for q in plan.queries[:3]:
            hits, diag = self.retriever.retrieve(q, k=self.k, mode=self.mode, intent=plan.intent)
            if diag.get("refused"):
                any_refused = True
                continue
            for h in hits:
                if h.chunk_id not in all_hits or (
                    h.scores.get("fused_weighted", 0)
                    > all_hits[h.chunk_id].scores.get("fused_weighted", 0)
                ):
                    all_hits[h.chunk_id] = h

        if not all_hits and any_refused:
            empty = postprocessor.FinalAnswer(
                propositions=[], sources=[],
                answer_summary="I could not find sufficient sources for this question.",
                confidence="low",
                answer_html=postprocessor._render_refusal(),
                refused=True, flags=["all retrieval queries below threshold"],
            )
            return _to_eval_format(empty)

        merged = sorted(
            all_hits.values(),
            key=lambda h: h.scores.get("fused_weighted", h.scores.get("fused", 0.0)),
            reverse=True,
        )[: self.k]
        for i, h in enumerate(merged):
            h.rank = i + 1

        answerer_history = _format_history_for_answerer(history) if history else None
        answer_json = answerer.compose(rewritten, merged, history=answerer_history)
        final = postprocessor.process(answer_json, merged, refused=False)
        verified, _ = verifier.verify(final)
        result = _to_eval_format(verified)
        if rewritten != question:
            result["_rewritten_question"] = rewritten
        return result


class FullPipeline:
    """Knowledge-augmented pipeline (post-upgrade default).

    Flow:
      1. Conversation-aware query rewrite (if history present).
      2. Knowledge-augmented planner (Claude Opus 4.7 -> GPT fallbacks):
         names provisions, names authorities, sets fringe/confidence.
      3. Three-channel retrieval:
         - Hybrid (BM25 + dense + RRF) per reformulated query, intent-weighted.
         - Named provisions resolved via name_index -> chunk_ids.
         - Named authorities resolved via name_index -> chunk_ids.
         Channels merged, deduped, intent-weighted, top-k.
      4. Answerer composes structured propositions, given:
         - The (rewritten) question
         - The merged chunk set with [CITABLE]/[CONTEXT-ONLY] tags
         - The planner's findings as a steer (lead with named authorities,
           use structured uncertainty mode if is_fringe)
      5. Postprocessor mechanically extracts quotes via span markers,
         strips non-citable citations defensively, renders HTML.
      6. Verifier walks the final structure for span/citation integrity.
    """
    name = "full"

    def __init__(
        self,
        k_chunks: int = 12,
        real_planner: bool = True,
        hybrid_per_query_k: int = 12,
    ):
        self.retriever = Retriever()
        self.k = k_chunks
        self.real_planner = real_planner
        self.hybrid_per_query_k = hybrid_per_query_k

    def answer(self, question: str, history: list | None = None) -> dict:
        rewritten = _rewrite_with_history(question, history)
        plan = planner.plan(rewritten, real=self.real_planner)
        return self.answer_with_plan(
            rewritten, plan, history=history, original_question=question,
        )

    def answer_with_plan(
        self,
        question: str,
        plan: "planner.Plan",
        history: list | None = None,
        original_question: str | None = None,
    ) -> dict:
        """Compose an answer for `question` (already-rewritten) given a
        pre-computed plan. Skips re-running the planner so the API can
        route once and feed the same plan to either pipeline."""
        log.info(
            "fast pipeline: intent=%s conf=%s fringe=%s provisions=%s authorities=%s",
            plan.intent, plan.confidence, plan.is_fringe,
            plan.named_provisions, plan.named_authorities,
        )

        merged, retrieval_diag = self.retriever.retrieve_three_channel(
            queries=plan.queries,
            intent=plan.intent,
            named_provisions=plan.named_provisions,
            named_authorities=plan.named_authorities,
            k=self.k,
            hybrid_per_query_k=self.hybrid_per_query_k,
        )

        # Refuse only if every channel returned nothing usable
        if not merged and retrieval_diag.get("refused"):
            empty = postprocessor.FinalAnswer(
                propositions=[], sources=[],
                answer_summary="I could not find sufficient sources for this question.",
                confidence="low",
                answer_html=postprocessor._render_refusal(),
                refused=True,
                flags=["three-channel retrieval returned no chunks"],
            )
            result = _to_eval_format(empty)
            result["_pipeline"] = self.name
            result["_planner"] = _planner_summary(plan)
            result["_retrieval"] = retrieval_diag
            return result

        planner_findings = {
            "intent": plan.intent,
            "named_provisions": plan.named_provisions,
            "named_authorities": plan.named_authorities,
            "confidence": plan.confidence,
            "is_fringe": plan.is_fringe,
            "notes": plan.notes,
        }
        answerer_history = _format_history_for_answerer(history) if history else None

        answer_json = answerer.compose(
            question, merged,
            history=answerer_history,
            planner_findings=planner_findings,
        )
        final = postprocessor.process(answer_json, merged, refused=False)
        verified, _ = verifier.verify(final)
        result = _to_eval_format(verified)
        if original_question is not None and question != original_question:
            result["_rewritten_question"] = question
        result["_pipeline"] = self.name
        result["_planner"] = _planner_summary(plan)
        result["_retrieval"] = retrieval_diag
        return result


def _planner_summary(plan: "planner.Plan") -> dict:
    return {
        "intent": plan.intent,
        "confidence": plan.confidence,
        "is_fringe": plan.is_fringe,
        "named_provisions": plan.named_provisions,
        "named_authorities": plan.named_authorities,
        "queries": plan.queries,
        "notes": plan.notes,
    }


def _format_history_for_answerer(history: list[dict]) -> list[dict]:
    """Convert stored conversation messages into a slim chat-history format
    suitable for the answerer system prompt context. Strips structured
    answer payloads down to plain text summaries."""
    out: list[dict] = []
    for h in history[-6:]:  # last 6 turns
        role = h.get("role", "")
        content = h.get("content", "")
        if isinstance(content, dict):
            text = content.get("answer_summary") or ""
            if not text:
                # fall back to first proposition text
                props = content.get("propositions") or []
                if props:
                    text = props[0].get("text", "")
        else:
            text = str(content)
        if not text:
            continue
        out.append({"role": role, "content": text[:1500]})
    return out


def route_question(plan: "planner.Plan", question: str) -> str:
    """Decide whether to use the fast path (FullPipeline) or the hard path
    (HardQuestionPipeline) for this question.

    Hard path triggers on any of:
      - planner flagged the question as fringe
      - planner confidence is medium or low
      - intent is case_law (doctrinal questions need full-case reading)
      - planner named two or more authorities (worth reading them properly)
    """
    if plan.is_fringe:
        return "hard"
    if plan.confidence in {"medium", "low"}:
        return "hard"
    if plan.intent == "case_law":
        return "hard"
    if len(plan.named_authorities or []) >= 2:
        return "hard"
    return "fast"
