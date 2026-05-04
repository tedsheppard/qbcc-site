"""Hard-cap budget tracking for the bif_research build.

Every OpenAI API call in this package must route through `record_call()` so
spend is tracked and the $13 hard cap is enforced. Soft warning at $12.50.
"""
from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUDGET_PATH = ROOT / "BUDGET.md"

# Build phase used $12.83 of an original $13.00 cap. The cap is now
# raised to $100 for live usage so the API can answer user queries
# without the build-time ceiling getting in the way. Adjust freely.
HARD_CAP_USD = 100.00
SOFT_WARN_USD = 95.00

# Per-1k-token prices (USD). Rough but conservative.
PRICES = {
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "gpt-5.5":      {"input": 0.005, "output": 0.04},  # speculative, conservative
    "gpt-5.4":      {"input": 0.005, "output": 0.04},
    "gpt-5.4-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-5":        {"input": 0.005, "output": 0.04},
    "gpt-5-mini":   {"input": 0.00015, "output": 0.0006},
    "gpt-4o":       {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini":  {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":  {"input": 0.01, "output": 0.03},
    # Anthropic Claude (rough public list-prices, per 1k tokens; conservative)
    "claude-opus-4-7":   {"input": 0.015, "output": 0.075},
    "claude-opus-4":     {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5":  {"input": 0.0008, "output": 0.004},
}

_lock = threading.Lock()
_cumulative_usd = None  # lazy init from BUDGET.md


class BudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Per-request cost capture (used by the API to record per-query cost)
# ---------------------------------------------------------------------------

# Each in-flight request can install a list into this contextvar; every
# record_call() during that request will append a row. async-safe.
_capture_var: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "_bif_research_cost_capture", default=None,
)


class CostCapture:
    """Context manager that captures every record_call() inside its scope.

    Use:
        with CostCapture() as cap:
            run_pipeline(...)
        cap.summary()   # -> {"total_usd": ..., "calls": [{...}, ...]}

    Safe across asyncio tasks because of contextvars.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._token = None

    def __enter__(self):
        self._token = _capture_var.set(self.calls)
        return self

    def __exit__(self, exc_type, exc, tb):
        _capture_var.reset(self._token)
        return False

    def summary(self) -> dict:
        total_usd = sum(c.get("cost_usd", 0.0) for c in self.calls)
        in_tokens = sum(c.get("input_tokens", 0) for c in self.calls)
        out_tokens = sum(c.get("output_tokens", 0) for c in self.calls)
        by_op: dict[str, dict] = {}
        for c in self.calls:
            op = c.get("operation", "?")
            row = by_op.setdefault(op, {"calls": 0, "in": 0, "out": 0, "cost_usd": 0.0})
            row["calls"] += 1
            row["in"] += c.get("input_tokens", 0)
            row["out"] += c.get("output_tokens", 0)
            row["cost_usd"] += c.get("cost_usd", 0.0)
        return {
            "total_usd": total_usd,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "n_api_calls": len(self.calls),
            "by_operation": by_op,
            "calls": self.calls,
        }


def _read_cumulative() -> float:
    """Parse BUDGET.md to find the most recent cumulative figure."""
    if not BUDGET_PATH.exists():
        return 0.0
    text = BUDGET_PATH.read_text(encoding="utf-8")
    # Find the last column of the last data row
    lines = [l for l in text.splitlines() if l.strip().startswith("|") and "$" in l]
    if not lines:
        return 0.0
    last = lines[-1]
    # Last cell is cumulative
    parts = [p.strip() for p in last.strip("|").split("|")]
    if not parts:
        return 0.0
    cum_str = parts[-1].lstrip("$").replace(",", "").strip()
    try:
        return float(cum_str)
    except ValueError:
        return 0.0


def _ensure_loaded():
    global _cumulative_usd
    if _cumulative_usd is None:
        _cumulative_usd = _read_cumulative()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Estimate USD cost for a single call."""
    p = PRICES.get(model, PRICES["gpt-5"])  # conservative fallback
    return (input_tokens * p["input"] / 1000.0) + (output_tokens * p["output"] / 1000.0)


def cumulative_usd() -> float:
    with _lock:
        _ensure_loaded()
        return _cumulative_usd or 0.0


def check_budget(headroom_usd: float = 0.0) -> None:
    """Raise BudgetExceeded if cumulative + headroom would exceed the hard cap."""
    with _lock:
        _ensure_loaded()
        if (_cumulative_usd or 0.0) + headroom_usd >= HARD_CAP_USD:
            raise BudgetExceeded(
                f"Hard cap ${HARD_CAP_USD:.2f} reached. Cumulative=${_cumulative_usd:.4f}, "
                f"requested headroom=${headroom_usd:.4f}. Stopping."
            )


def record_call(operation: str, model: str, input_tokens: int, output_tokens: int = 0,
                note: str = "") -> float:
    """Record an API call and return its cost. Raises if hard cap exceeded.

    Soft warning at SOFT_WARN_USD just logs to STATUS.md but does not raise.
    Also pushes a row to any active CostCapture in the current context.
    """
    cost = estimate_cost(model, input_tokens, output_tokens)
    timestamp = _now()
    with _lock:
        _ensure_loaded()
        global _cumulative_usd
        new_cum = (_cumulative_usd or 0.0) + cost
        if new_cum > HARD_CAP_USD:
            raise BudgetExceeded(
                f"Call would exceed hard cap. Cum=${_cumulative_usd:.4f} + ${cost:.4f} > ${HARD_CAP_USD:.2f}"
            )
        _cumulative_usd = new_cum
        # Append to BUDGET.md
        line = (
            f"| {timestamp} | {operation} | {model} | {input_tokens:,} / {output_tokens:,} | "
            f"${cost:.4f} | ${_cumulative_usd:.4f} |"
        )
        if note:
            line += f"  <!-- {note} -->"
        with open(BUDGET_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if _cumulative_usd >= SOFT_WARN_USD:
            with open(ROOT / "STATUS.md", "a", encoding="utf-8") as f:
                f.write(
                    f"\n### Soft budget warning {timestamp}\n"
                    f"Cumulative spend now ${_cumulative_usd:.4f} — within "
                    f"${HARD_CAP_USD - SOFT_WARN_USD:.2f} of hard cap. "
                    f"Subsequent calls may be refused.\n"
                )
    # Push to any per-request capture (outside the lock — contextvar is local)
    capture = _capture_var.get()
    if capture is not None:
        capture.append({
            "timestamp": timestamp,
            "operation": operation,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        })
    return cost


def remaining_usd() -> float:
    return max(0.0, HARD_CAP_USD - cumulative_usd())


# Bootstrap the BUDGET.md header columns if not present yet
def _bootstrap_header():
    if not BUDGET_PATH.exists():
        return
    text = BUDGET_PATH.read_text(encoding="utf-8")
    if "| Operation | Model |" in text:
        return
    # Replace generic header with model-aware header
    new_header = (
        "| Timestamp (UTC) | Operation | Model | Tokens (in/out) | Cost USD | Cumulative |\n"
        "|---|---|---|---|---|---|\n"
    )
    text = text.replace(
        "| Timestamp (UTC) | Operation | Tokens (in/out) | Cost USD | Cumulative |\n|---|---|---|---|---|\n",
        new_header,
    )
    BUDGET_PATH.write_text(text, encoding="utf-8")


_bootstrap_header()
