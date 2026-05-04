"""LLM client wrapper for bif_research.

Routes through `budget.py` for spend tracking. Uses model fallback chains:
- DEFAULT chain (small, planner, judge): gpt-5.4-mini -> gpt-5-mini -> gpt-4o-mini
- MAX chain (answerer): gpt-5.5 -> gpt-5.4 -> gpt-5 -> gpt-4o

Two entry points:
- `complete_chat(messages, *, kind, ...)` for chat completions
- `embed(texts, *, model)` for embeddings

The `kind` argument selects the chain; the wrapper walks the chain on
ModelNotFound / ModelDeprecated errors.
"""
from __future__ import annotations

import os
import json
import logging
import time
from typing import Any, Iterable, Optional

from openai import OpenAI, BadRequestError, NotFoundError, RateLimitError, APIConnectionError

from . import budget

log = logging.getLogger("bif_research.llm")

# Chain config
CHAIN_DEFAULT = ["gpt-5.4-mini", "gpt-5-mini", "gpt-4o-mini"]
CHAIN_MAX = ["gpt-5.5", "gpt-5.4", "gpt-5", "gpt-4o"]
# Knowledge-augmented planner — wants the strongest reasoning model. Tries
# Claude Opus 4.7 first (via Anthropic SDK), falls through to GPT chain if
# the Anthropic SDK isn't installed or no ANTHROPIC_API_KEY is set.
CHAIN_PLANNER_KA = [
    "claude-opus-4-7",
    "claude-opus-4",
    "claude-sonnet-4-5",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5",
]
# Reader workers — cheap comprehension model, runs per-case fan-out.
# Falls through to claude-haiku-4-5 if the GPT minis aren't available;
# haiku is skipped silently when no ANTHROPIC_API_KEY is set.
CHAIN_READER = ["gpt-5.4-mini", "gpt-5-mini", "claude-haiku-4-5", "gpt-4o-mini"]

# Reasoner — final answer composer for the hard pipeline. Wants the
# strongest model. Same Anthropic-then-GPT fallthrough as the planner.
CHAIN_REASONER = [
    "claude-opus-4-7",
    "claude-opus-4",
    "claude-sonnet-4-5",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5",
]

# Process cache so we don't keep retrying a model that doesn't exist
_working_model: dict[str, str] = {}

_client: OpenAI | None = None
_anthropic_client = None
_anthropic_disabled = False  # set True once we've confirmed it can't be used


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # picks up OPENAI_API_KEY env var
    return _client


def _anthropic():
    """Lazy-init Anthropic client. Returns None if SDK missing or key unset."""
    global _anthropic_client, _anthropic_disabled
    if _anthropic_disabled:
        return None
    if _anthropic_client is not None:
        return _anthropic_client
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _anthropic_disabled = True
        return None
    try:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
        return _anthropic_client
    except Exception as e:
        log.info(f"anthropic unavailable: {e}")
        _anthropic_disabled = True
        return None


def _approx_tokens(text: str) -> int:
    """Cheap heuristic — 4 chars per token. Good enough for budget tracking."""
    return max(1, len(text) // 4)


def _approx_messages_tokens(messages: list[dict]) -> int:
    return sum(_approx_tokens(m.get("content", "") or "") for m in messages)


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic wants `system` as a top-level param, not a message role."""
    sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
    other = [m for m in messages if m.get("role") != "system"]
    return ("\n\n".join(sys_parts) if sys_parts else ""), other


def _call_anthropic(
    model: str,
    messages: list[dict],
    response_format: Optional[dict],
    max_output_tokens: int,
    temperature: Optional[float],
) -> tuple[str, int, int]:
    """Single Anthropic call. Returns (text, input_tokens, output_tokens)."""
    a = _anthropic()
    if a is None:
        raise RuntimeError("anthropic-unavailable")
    system, conv = _split_system(messages)
    if response_format and response_format.get("type") == "json_object":
        # Nudge Claude to emit a JSON object — it doesn't have a hard
        # response_format param like OpenAI, but it is very reliable when
        # told to output ONLY JSON.
        if system:
            system = system + "\n\nRespond with ONLY a single valid JSON object. No prose, no code fences."
        else:
            system = "Respond with ONLY a single valid JSON object. No prose, no code fences."
    kw = dict(model=model, messages=conv, max_tokens=max_output_tokens or 1024)
    if system:
        kw["system"] = system
    if temperature is not None:
        kw["temperature"] = temperature
    resp = a.messages.create(**kw)
    text_parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    text = "".join(text_parts)
    in_tok = getattr(resp.usage, "input_tokens", 0) or 0
    out_tok = getattr(resp.usage, "output_tokens", 0) or 0
    return text, in_tok, out_tok


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude")


def complete_chat(
    messages: list[dict],
    *,
    kind: str = "default",
    operation: str = "chat",
    response_format: Optional[dict] = None,
    max_output_tokens: int = 4096,
    temperature: Optional[float] = None,
    tools: Optional[list] = None,
) -> tuple[str, dict]:
    """Run a chat completion through the appropriate model chain.

    Returns (text, usage) where usage has `model`, `input_tokens`, `output_tokens`.
    Raises BudgetExceeded if the call would breach the cap.
    """
    if kind == "max":
        chain = CHAIN_MAX
    elif kind == "planner_ka":
        chain = CHAIN_PLANNER_KA
    elif kind == "reader":
        chain = CHAIN_READER
    elif kind == "reasoner":
        chain = CHAIN_REASONER
    else:
        chain = CHAIN_DEFAULT
    cached = _working_model.get(kind)
    if cached and cached in chain:
        ordered = [cached] + [m for m in chain if m != cached]
    else:
        ordered = list(chain)

    # Pre-flight budget check (use first model's price as estimate)
    in_tokens_est = _approx_messages_tokens(messages)
    cost_est = budget.estimate_cost(ordered[0], in_tokens_est, max_output_tokens)
    budget.check_budget(headroom_usd=cost_est)

    last_err: Exception | None = None
    for model in ordered:
        # Skip Claude models silently if Anthropic isn't available
        if _is_anthropic_model(model) and _anthropic() is None:
            log.info(f"skipping {model}: anthropic SDK / API key unavailable")
            continue
        try:
            if _is_anthropic_model(model):
                text, in_tokens, out_tokens = _call_anthropic(
                    model, messages, response_format, max_output_tokens, temperature
                )
                if not in_tokens:
                    in_tokens = in_tokens_est
                if not out_tokens:
                    out_tokens = _approx_tokens(text)
            else:
                kwargs = {"model": model, "messages": messages}
                if response_format:
                    kwargs["response_format"] = response_format
                if max_output_tokens:
                    kwargs["max_completion_tokens"] = max_output_tokens
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if tools:
                    kwargs["tools"] = tools
                resp = client().chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                in_tokens = getattr(usage, "prompt_tokens", in_tokens_est)
                out_tokens = getattr(usage, "completion_tokens", _approx_tokens(text))
            budget.record_call(operation, model, in_tokens, out_tokens)
            _working_model[kind] = model
            return text, {"model": model, "input_tokens": in_tokens, "output_tokens": out_tokens}
        except (NotFoundError, BadRequestError) as e:
            msg = str(e).lower()
            if "model" in msg or "not found" in msg or "does not exist" in msg or "unsupported" in msg:
                log.info(f"model {model} unavailable ({e}); trying next in chain")
                last_err = e
                continue
            raise
        except Exception as e:
            # Anthropic errors land here — auth / model-not-found / etc.
            log.warning(f"model {model} call failed ({e}); trying next")
            last_err = e
            continue

    raise RuntimeError(f"All models in chain {ordered} failed. Last error: {last_err}")


def embed(texts: list[str], *, model: str = "text-embedding-3-small",
          operation: str = "embed", max_retries: int = 6) -> list[list[float]]:
    """Embed a batch of texts. Records spend. Raises if cap would be exceeded.

    Caller is responsible for batching to a sane batch size (recommend 100).
    Retries on RateLimitError / APIConnectionError with exponential backoff
    capped at 30s.
    """
    if not texts:
        return []
    in_tokens_est = sum(_approx_tokens(t) for t in texts)
    cost_est = budget.estimate_cost(model, in_tokens_est, 0)
    budget.check_budget(headroom_usd=cost_est)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client().embeddings.create(model=model, input=texts)
            in_tokens = getattr(resp.usage, "prompt_tokens", in_tokens_est)
            budget.record_call(operation, model, in_tokens, 0)
            return [d.embedding for d in resp.data]
        except RateLimitError as e:
            wait = min(30.0, 2 ** attempt + 1.0)
            log.warning(f"embed rate-limited (attempt {attempt+1}/{max_retries}); sleeping {wait}s")
            time.sleep(wait)
            last_err = e
        except APIConnectionError as e:
            wait = min(15.0, 2 ** attempt)
            log.warning(f"embed connection error (attempt {attempt+1}/{max_retries}); sleeping {wait}s")
            time.sleep(wait)
            last_err = e
    raise RuntimeError(f"embed failed after {max_retries} attempts: {last_err}")
