"""Central LLM configuration for /claim-check.

All claim-check LLM calls go through ``complete()`` below. This is where
model selection, reasoning effort, and fallback chains live.

Design per spec Section 5:
  - Default:   model=gpt-5.4-mini, reasoning=medium
  - Escalated: model=gpt-5.4-mini, reasoning=high      (for rules marked `Escalate: high-reasoning`)
  - Max:       model=gpt-5.4,      reasoning=high      (for explicit low-confidence escalation)

Fallback chain: if the preferred model ID or the reasoning_effort
parameter is not accepted by the account, fall back gracefully so the
feature does not hard-error on a model name mismatch. The fallback that
works is cached for the process lifetime.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("claim_check.llm")

# Ordered preference. First entry is tried first; on model-not-found, we walk down.
MODEL_CHAIN_DEFAULT = ["gpt-5.4-mini", "gpt-5-mini", "gpt-4o-mini", "gpt-4o"]
MODEL_CHAIN_MAX     = ["gpt-5.4",       "gpt-5",      "gpt-4o",       "gpt-4-turbo"]

# Reasoning effort applies to reasoning-capable models only. We'll try with it first;
# if the API rejects with `unknown parameter` or similar, retry without.
VALID_REASONING = {"low", "medium", "high"}

# Per-process cache of "this model worked last time, try it first".
_working_model_cache: dict[str, str] = {}
_reasoning_unsupported: set[str] = set()
_cache_lock = threading.Lock()

# Simple daily cost cap (USD). Crude USD estimate based on a rough blend.
DAILY_COST_CAP_USD = float(os.getenv("CLAIM_CHECK_DAILY_COST_CAP_USD", "50"))
USAGE_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "claim-check-usage.jsonl"

# Rough per-1k-token cost estimates (USD). Tuned conservatively; real cost depends
# on actual billing. This is only used for the daily cap; exact accounting isn't
# the point.
COST_PER_1K_INPUT_USD = {
    "gpt-5.4-mini": 0.0015,
    "gpt-5-mini":   0.0015,
    "gpt-4o-mini":  0.00015,
    "gpt-4o":       0.005,
    "gpt-5.4":      0.015,
    "gpt-5":        0.015,
    "gpt-4-turbo":  0.01,
}
COST_PER_1K_OUTPUT_USD = {
    "gpt-5.4-mini": 0.006,
    "gpt-5-mini":   0.006,
    "gpt-4o-mini":  0.0006,
    "gpt-4o":       0.015,
    "gpt-5.4":      0.06,
    "gpt-5":        0.06,
    "gpt-4-turbo":  0.03,
}

_cost_today: dict[str, float] = {"date": "", "total_usd": 0.0}
_cost_lock = threading.Lock()


class CostCapExceededError(RuntimeError):
    pass


def _today_key() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate = COST_PER_1K_INPUT_USD.get(model, 0.005)
    out_rate = COST_PER_1K_OUTPUT_USD.get(model, 0.015)
    return (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate


def _bump_cost(usd: float) -> None:
    with _cost_lock:
        today = _today_key()
        if _cost_today.get("date") != today:
            _cost_today["date"] = today
            _cost_today["total_usd"] = 0.0
        _cost_today["total_usd"] = _cost_today.get("total_usd", 0.0) + usd


def _cost_today_usd() -> float:
    with _cost_lock:
        if _cost_today.get("date") != _today_key():
            return 0.0
        return float(_cost_today.get("total_usd", 0.0))


def _check_cost_cap() -> None:
    current = _cost_today_usd()
    if current >= DAILY_COST_CAP_USD:
        raise CostCapExceededError(
            f"Daily cost cap reached (${current:.2f} / ${DAILY_COST_CAP_USD:.2f}). "
            "Try again tomorrow."
        )


def _log_usage(entry: dict[str, Any]) -> None:
    try:
        USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        log.warning("Failed to write usage log", exc_info=True)


def complete(
    *,
    messages: list[dict[str, str]],
    reasoning_effort: str = "medium",
    tier: str = "default",  # "default" | "max"
    response_format: dict[str, str] | None = None,
    max_output_tokens: int = 1500,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Call the appropriate OpenAI chat.completions model.

    Returns a dict: {"content": str, "model": str, "reasoning": str,
                     "input_tokens": int, "output_tokens": int,
                     "cost_usd": float, "low_confidence": bool}
    """
    if reasoning_effort not in VALID_REASONING:
        reasoning_effort = "medium"

    _check_cost_cap()

    from openai import OpenAI  # deferred
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured on the server.")
    client = OpenAI(api_key=api_key)

    chain = MODEL_CHAIN_MAX if tier == "max" else MODEL_CHAIN_DEFAULT
    # Put the last-known-working model first.
    with _cache_lock:
        preferred = _working_model_cache.get(tier)
    if preferred and preferred in chain:
        chain = [preferred] + [m for m in chain if m != preferred]

    last_err: Exception | None = None
    for model in chain:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_output_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if temperature is not None:
            kwargs["temperature"] = temperature
        include_reasoning = model not in _reasoning_unsupported
        if include_reasoning:
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            # Model not available -> try next.
            if "model" in msg.lower() and ("not found" in msg.lower() or "does not exist" in msg.lower() or "404" in msg):
                log.info("model %s not available, trying next", model)
                last_err = e
                continue
            # Unknown param -> retry without reasoning_effort.
            if include_reasoning and ("reasoning_effort" in msg or "unknown_parameter" in msg.lower() or "unrecognized request argument" in msg.lower()):
                with _cache_lock:
                    _reasoning_unsupported.add(model)
                log.info("model %s does not accept reasoning_effort, retrying without", model)
                try:
                    kwargs.pop("reasoning_effort", None)
                    # Some older models need max_tokens instead of max_completion_tokens.
                    if "max_completion_tokens" in kwargs:
                        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    resp = client.chat.completions.create(**kwargs)
                except Exception as e2:
                    last_err = e2
                    if "max_tokens" in str(e2) or "max_completion_tokens" in str(e2):
                        # Swap once more if param name mismatch bites.
                        continue
                    # Some other problem; try next model.
                    continue
            else:
                last_err = e
                continue

        # Success.
        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = _estimate_cost_usd(model, in_tokens, out_tokens)
        _bump_cost(cost)

        low_conf = _looks_low_confidence(content)

        with _cache_lock:
            _working_model_cache[tier] = model

        _log_usage({
            "ts": int(time.time()),
            "tier": tier,
            "model": model,
            "reasoning_effort": reasoning_effort if include_reasoning else None,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": round(cost, 6),
            "low_confidence": low_conf,
        })

        return {
            "content": content,
            "model": model,
            "reasoning": reasoning_effort if include_reasoning else None,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": cost,
            "low_confidence": low_conf,
        }

    raise RuntimeError(f"All models in chain failed. Last error: {last_err}")


_LOW_CONF_MARKERS: tuple[str, ...] = (
    "i'm not certain",
    "i am not certain",
    "this requires more analysis",
    "cannot determine",
    "insufficient information",
    "unclear from the text",
)


def _looks_low_confidence(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _LOW_CONF_MARKERS)


def reasoning_for_rule(rule: dict[str, Any], default: str = "medium") -> str:
    """Determine reasoning level for a rule based on its `Escalate` tag."""
    esc = (rule.get("escalate") or "").strip().lower()
    if "high-reasoning" in esc or esc == "high":
        return "high"
    return default


def reasoning_for_chat(message: str) -> str:
    """Heuristic escalation for chat messages per Section 5."""
    m = (message or "").lower()
    if len(message or "") > 200:
        return "high"
    multi_step = ("compare", "why", "explain the difference", "differ", "contrast", "step by step")
    if any(k in m for k in multi_step):
        return "high"
    return "medium"
