"""Contract Assist chat orchestrator.

For one user turn:
  1. Classify the question briefly so we can show useful thinking states.
  2. Retrieve top-6 contract chunks (subagent-2 retrieval) and top-4 BIF
     Act passages (subagent-3 index).
  3. Build the system prompt (services.contract_assist.prompts).
  4. Stream the LLM response via OpenAI's streaming chat completions.
  5. Parse the streamed content into SSE events: status / content / sources / draft / done.

The orchestrator yields SSE event tuples (event_name, data_dict) the caller
serialises onto the wire. Production caller is routes/contract_assist.py.

Model selection mirrors services.claim_check.llm_config — gpt-5.4-mini at
reasoning=medium by default, escalating to reasoning=high for multi-step
chat messages or drafting requests, with a fallback chain if the model ID
is rejected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, AsyncIterator

from . import prompts

log = logging.getLogger("contract_assist.chatbot")

DEFAULT_MODEL_CHAIN = ["gpt-5.4-mini", "gpt-5-mini", "gpt-4o-mini", "gpt-4o"]
HIGH_MODEL_CHAIN = ["gpt-5.4", "gpt-5", "gpt-4o", "gpt-4-turbo"]
VALID_REASONING = {"low", "medium", "high"}
MAX_OUTPUT_TOKENS = 1800

DRAFT_TRIGGER_KEYWORDS = (
    "draft",
    "prepare a notice",
    "write a notice",
    "compose",
    "give me a",
    "extension of time",
    "variation notice",
)


def _classify_intent(message: str, history: list[dict[str, str]] | None) -> dict[str, Any]:
    """Cheap, deterministic classifier used purely to pick thinking-state messages
    and reasoning tier — never as a hard filter."""
    m = (message or "").lower()
    drafting = any(k in m for k in DRAFT_TRIGGER_KEYWORDS)
    multistep = any(k in m for k in ("compare", "explain the difference", "step by step", "why")) or len(message or "") > 220
    clause_lookup = bool(re.search(r"\bclause\s+\d", m)) or bool(re.search(r"\bs\s+\d{1,3}\b", m))
    return {
        "drafting": drafting,
        "multistep": multistep,
        "clause_lookup": clause_lookup,
        "reasoning": "high" if (drafting or multistep) else "medium",
    }


def _thinking_messages(intent: dict[str, Any]) -> list[str]:
    if intent["drafting"]:
        return [
            "Reading your question…",
            "Locating relevant clauses and party details…",
            "Cross-referencing the BIF Act…",
            "Drafting the document…",
        ]
    if intent["clause_lookup"]:
        return [
            "Reading your question…",
            "Searching for the clause in your contract…",
            "Reading clause text…",
            "Drafting response…",
        ]
    if intent["multistep"]:
        return [
            "Reading your question…",
            "Reviewing relevant clauses…",
            "Cross-referencing the BIF Act…",
            "Drafting response…",
        ]
    return [
        "Reading your question…",
        "Searching your contract…",
        "Drafting response…",
    ]


def _format_attachments(attachments: list[dict[str, Any]] | None) -> str:
    """Return a short textual summary of any inline attachments. We do NOT
    decode them or embed binaries — the model gets a description only.
    Spec: 'processed inline, added to context for that turn only, never persisted'."""
    if not attachments:
        return ""
    lines = ["The user attached the following file(s) to THIS message only — refer to them as needed:"]
    for a in attachments[:5]:
        size_kb = (a.get("size") or 0) / 1024.0
        lines.append(f"  - {a.get('name') or '(unnamed)'} ({size_kb:.0f} KB, {a.get('type') or 'application/octet-stream'})")
    lines.append("(Attachment bytes are not included in this prompt; if you need to extract specific text from one, ask the user.)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streaming entry point
# ---------------------------------------------------------------------------

async def stream_chat(
    *,
    session_id: str,
    message: str,
    history: list[dict[str, str]],
    contract_meta: dict[str, Any] | None,
    attachments: list[dict[str, Any]] | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield (event_name, data_dict) tuples for the SSE stream.

    Events:
      thinking  {message}
      sources   {sources: [...]}
      content   {delta}
      draft     {kind, content}      — emitted once at end if a draft was produced
      done      {tokens_in?, tokens_out?, model?}
      error     {message}
    """
    intent = _classify_intent(message, history)

    # 1) Show first thinking state immediately.
    thinking = _thinking_messages(intent)
    yield ("thinking", {"message": thinking[0]})

    # 2) Retrieve from the contract (session-scoped).
    contract_chunks: list[dict[str, Any]] = []
    bif_chunks: list[dict[str, Any]] = []
    try:
        from .retrieval import retrieve as retrieve_contract
        contract_chunks = retrieve_contract(message, session_id=session_id, top_k=6) or []
    except Exception as e:
        log.warning("contract retrieval failed: %s", e)

    yield ("thinking", {"message": thinking[1]})

    # 3) Retrieve from the BIF Act guide.
    try:
        from .bif_act_index import retrieve_bif
        bif_chunks = retrieve_bif(message, top_k=4) or []
    except Exception as e:
        log.warning("BIF Act retrieval failed: %s", e)

    if len(thinking) > 2:
        yield ("thinking", {"message": thinking[2]})

    # 4) Emit consolidated sources up-front so the frontend can show source
    #    pills as soon as content starts arriving.
    sources = _shape_sources(contract_chunks, bif_chunks)
    yield ("sources", {"sources": sources})

    # 5) Build the system prompt and call the LLM.
    system = prompts.build_system_prompt(
        contract_meta=contract_meta,
        contract_chunks=contract_chunks,
        bif_chunks=bif_chunks,
        history=history,
    )
    user_content = message
    attach_block = _format_attachments(attachments)
    if attach_block:
        user_content = attach_block + "\n\n---\n\n" + user_content

    if len(thinking) > 3:
        yield ("thinking", {"message": thinking[-1]})

    try:
        accumulated = ""
        async for delta, meta in _stream_openai(
            system=system,
            history=history or [],
            user_content=user_content,
            reasoning=intent["reasoning"],
        ):
            if delta:
                accumulated += delta
                yield ("content", {"delta": delta})
        # 6) After streaming, scan for a draft block and emit `draft` if found.
        draft = prompts.detect_draft(accumulated)
        if draft:
            yield ("draft", draft)
        yield ("done", {
            "model": meta.get("model"),
            "reasoning": meta.get("reasoning"),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
        })
    except Exception as e:
        log.exception("contract chat stream failed")
        yield ("error", {"message": str(e)})


# ---------------------------------------------------------------------------
# OpenAI streaming wrapper with fallback chain
# ---------------------------------------------------------------------------

async def _stream_openai(
    *,
    system: str,
    history: list[dict[str, str]],
    user_content: str,
    reasoning: str,
):
    """Async generator yielding (delta_str, meta_dict_at_end). The final yield
    always carries the meta dict; intermediate yields carry meta={} (unused)."""
    if reasoning not in VALID_REASONING:
        reasoning = "medium"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured on the server.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Build messages (cap history to keep prompts small).
    msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    trimmed = [m for m in (history or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    for m in trimmed[-12:]:
        msgs.append({"role": m["role"], "content": str(m.get("content", ""))})
    msgs.append({"role": "user", "content": user_content})

    chain = HIGH_MODEL_CHAIN if reasoning == "high" else DEFAULT_MODEL_CHAIN

    last_err: Exception | None = None
    for model in chain:
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=msgs,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
                reasoning_effort=reasoning,
            )
        except Exception as e:
            msg = str(e)
            if include_param_failure(msg, "reasoning_effort"):
                # Older models reject reasoning_effort — retry without.
                try:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=msgs,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                except Exception as e2:
                    last_err = e2
                    if model_not_found(str(e2)):
                        continue
                    raise
            elif model_not_found(msg):
                last_err = e
                continue
            else:
                raise

        # Consume the stream.
        in_tokens = out_tokens = 0
        for evt in stream:
            choice = (evt.choices or [None])[0] if hasattr(evt, "choices") else None
            if choice and getattr(choice, "delta", None) and getattr(choice.delta, "content", None):
                yield (choice.delta.content, {})
            usage = getattr(evt, "usage", None)
            if usage:
                in_tokens = getattr(usage, "prompt_tokens", in_tokens)
                out_tokens = getattr(usage, "completion_tokens", out_tokens)

        # Final yield: empty delta + meta.
        yield ("", {
            "model": model,
            "reasoning": reasoning,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        })
        return

    raise RuntimeError(f"All models in chain failed. Last error: {last_err}")


def model_not_found(err_msg: str) -> bool:
    msg = err_msg.lower()
    return "model" in msg and ("not found" in msg or "does not exist" in msg or "404" in msg)


def include_param_failure(err_msg: str, param: str) -> bool:
    m = err_msg.lower()
    return param in m and ("unknown" in m or "unrecognized" in m or "unsupported" in m)


# ---------------------------------------------------------------------------
# Source shaping — what the frontend renders as pills + source list
# ---------------------------------------------------------------------------

def _shape_sources(contract_chunks: list[dict[str, Any]], bif_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in contract_chunks or []:
        # Pick a clause tag if present, otherwise use page or chunk id.
        nums = c.get("clause_numbers") or []
        if isinstance(nums, str):
            nums = [n.strip() for n in nums.split(",") if n.strip()]
        if nums:
            tag = "clause " + sorted(nums, key=lambda x: (-len(str(x)), x))[0]
        elif c.get("page_number"):
            tag = f"contract p{c['page_number']}"
        else:
            tag = "contract"
        out.append({
            "type": "contract",
            "tag": tag,
            "heading": c.get("section_heading") or "",
            "excerpt": (c.get("excerpt") or c.get("full_text") or "")[:240],
            "full_text": (c.get("expanded_text") or c.get("full_text") or "")[:3000],
            "page_number": c.get("page_number"),
            "score": float(c.get("score") or 0.0),
        })
    for c in bif_chunks or []:
        sec = (c.get("section_ref") or "").strip() or "BIF Act"
        out.append({
            "type": "bif_act",
            "tag": f"{sec} BIF Act" if not sec.lower().endswith("bif act") else sec,
            "heading": c.get("heading") or "",
            "excerpt": (c.get("snippet") or "")[:240],
            "full_text": (c.get("snippet") or "")[:3000],
            "anchor_url": c.get("anchor_url"),
            "score": float(c.get("score") or 0.0),
        })
    return out
