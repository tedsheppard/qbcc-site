"""Parser for rules/bif_act_rules.md.

Single source of truth: the /claim-check engine never hardcodes rule
content. This module turns the markdown file into a list of Rule dicts
that the rule engine, the /checks endpoint, and the frontend consume.

The file is re-read on every call (no caching) so the user can edit
rules without a redeploy. The file is small and reads are cheap.

Contract with the authoring file is documented at the top of
rules/bif_act_rules.md.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("claim_check.rules_parser")

RULES_PATH = Path(__file__).resolve().parents[2] / "rules" / "bif_act_rules.md"

VALID_MODES = {
    "payment_claim_serving",
    "payment_claim_received",
    "payment_schedule_giving",
    "payment_schedule_received",
}

VALID_INPUT_TYPES = {"date", "radio", "text", "number", "licensee_lookup"}

VALID_IMPLEMENTATIONS = {"semantic", "user_input", "deterministic", "qbcc_lookup"}


class RulesParseError(ValueError):
    pass


def load_rules(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Parse the rules markdown file into a list of rule dicts.

    Each returned dict has:
      id, title, applies_to (list[str]), act_reference (str),
      search_query (str), implementation (list[str]),
      escalate (str | None), quote_requirement (str | None),
      pass_criteria (str | None), warning_criteria (str | None),
      fail_criteria (str | None), inputs (list[dict])
    """
    p = Path(path) if path else RULES_PATH
    if not p.exists():
        raise RulesParseError(f"Rules file not found: {p}")
    text = p.read_text(encoding="utf-8")
    rules: list[dict[str, Any]] = []

    # Split into rule blocks on level-2 headings that look like "## <ID>: <title>".
    # A heading without a colon (like "## Rule file schema") is treated as a
    # header section and skipped.
    heading_re = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    positions: list[tuple[int, str]] = [(m.start(), m.group(1).strip()) for m in heading_re.finditer(text)]
    positions.append((len(text), ""))  # sentinel

    for i in range(len(positions) - 1):
        start, heading = positions[i]
        end, _ = positions[i + 1]
        block_full = text[start:end]
        # Strip the heading line from the block.
        nl = block_full.find("\n")
        heading_line = block_full[:nl] if nl != -1 else block_full
        body = block_full[nl + 1 :] if nl != -1 else ""

        # Parse "## <ID>: <title>".
        m = re.match(r"^##\s+([A-Za-z0-9][A-Za-z0-9_\-]*)\s*:\s*(.+?)\s*$", heading_line)
        if not m:
            continue  # not a rule heading (e.g. the schema-explainer section)
        rule_id = m.group(1).strip()
        title = m.group(2).strip()

        rule = _parse_rule_block(rule_id, title, body)
        if rule:
            rules.append(rule)

    return rules


def rules_for_mode(mode: str, path: Path | str | None = None) -> list[dict[str, Any]]:
    if mode not in VALID_MODES:
        return []
    return [r for r in load_rules(path) if mode in r["applies_to"]]


def _parse_rule_block(rule_id: str, title: str, body: str) -> dict[str, Any] | None:
    # Extract bolded single-line fields (**Field:** value).
    def field(name: str) -> str | None:
        m = re.search(rf"^\*\*{re.escape(name)}:\*\*\s*(.+?)\s*$", body, re.MULTILINE)
        return m.group(1).strip() if m else None

    applies_to_raw = field("Applies to") or ""
    applies_to = [a.strip() for a in applies_to_raw.split(",") if a.strip()]
    # Filter to valid modes, quietly drop unknowns.
    applies_to = [a for a in applies_to if a in VALID_MODES]
    if not applies_to:
        return None  # a rule that applies to nothing is ignored

    act_reference = field("Act reference") or ""
    search_query = field("Search query") or title

    impl_raw = field("Implementation") or "semantic"
    impl = [p.strip() for p in impl_raw.split("+") if p.strip()]
    for p in impl:
        if p not in VALID_IMPLEMENTATIONS:
            raise RulesParseError(
                f"Rule {rule_id}: unknown Implementation component {p!r}. "
                f"Valid: {', '.join(sorted(VALID_IMPLEMENTATIONS))}"
            )

    escalate = field("Escalate") or None
    quote_requirement = field("Quote requirement") or None

    # Extract the multi-line criteria blocks.
    def criteria(label: str) -> str | None:
        # Match **Pass criteria:** up to the next **Label:** or EOF.
        pat = re.compile(
            rf"\*\*{re.escape(label)}:\*\*\s*\n(.+?)(?=\n\*\*[A-Z][A-Za-z ]+:\*\*|\n---|\Z)",
            re.DOTALL,
        )
        m = pat.search(body)
        if not m:
            return None
        return m.group(1).strip() or None

    pass_criteria = criteria("Pass criteria")
    warning_criteria = criteria("Warning criteria")
    fail_criteria = criteria("Fail criteria")

    inputs = _parse_inputs(rule_id, body)

    return {
        "id": rule_id,
        "title": title,
        "applies_to": applies_to,
        "act_reference": act_reference,
        "search_query": search_query,
        "implementation": impl,
        "escalate": escalate,
        "quote_requirement": quote_requirement,
        "pass_criteria": pass_criteria,
        "warning_criteria": warning_criteria,
        "fail_criteria": fail_criteria,
        "inputs": inputs,
    }


_SHOW_IF_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*==\s*\"(.*)\"\s*$")


def _parse_inputs(rule_id: str, body: str) -> list[dict[str, Any]]:
    """Parse the **Inputs required:** list block.

    Each question is a YAML-ish entry like:

        - id: last_work_date
          question: "When was construction work last carried out?"
          type: date
          required: true
          no_future: true
          show_if: foo == "Yes"
    """
    m = re.search(
        r"\*\*Inputs required:\*\*\s*\n(.+?)(?=\n\*\*[A-Z][A-Za-z ]+:\*\*|\n---|\Z)",
        body,
        re.DOTALL,
    )
    if not m:
        return []

    raw = m.group(1)

    # Split on "- id:" lines at indentation zero-or-two spaces.
    entry_starts = [e.start() for e in re.finditer(r"(?m)^-\s+id:\s*", raw)]
    if not entry_starts:
        return []

    entry_starts.append(len(raw))
    entries_raw = [raw[entry_starts[i]:entry_starts[i + 1]] for i in range(len(entry_starts) - 1)]

    inputs: list[dict[str, Any]] = []
    for entry in entries_raw:
        # Normalise: each subsequent key/value is on its own line at any indentation.
        # We accept key: value pairs (value possibly quoted).
        parsed: dict[str, Any] = {}
        for line in entry.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- id:"):
                parsed["id"] = line[len("- id:"):].strip()
                continue
            # `key: value`
            km = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
            if not km:
                continue
            k = km.group(1)
            v_raw = km.group(2).strip()
            parsed[k] = _parse_yaml_scalar(v_raw)

        if not parsed.get("id"):
            continue

        q_type = parsed.get("type", "text")
        if q_type not in VALID_INPUT_TYPES:
            raise RulesParseError(
                f"Rule {rule_id}, input {parsed.get('id')!r}: unknown input type {q_type!r}. "
                f"Valid: {', '.join(sorted(VALID_INPUT_TYPES))}"
            )

        # Validate show_if syntax.
        show_if = parsed.get("show_if")
        if show_if:
            sim = _SHOW_IF_RE.match(show_if if isinstance(show_if, str) else str(show_if))
            if not sim:
                raise RulesParseError(
                    f"Rule {rule_id}, input {parsed.get('id')!r}: show_if must be "
                    f"`<id> == \"<value>\"`, got: {show_if!r}"
                )

        inputs.append({
            "id": parsed["id"],
            "question": parsed.get("question") or "",
            "type": q_type,
            "options": parsed.get("options") or None,
            "required": bool(parsed.get("required", False)),
            "show_if": show_if,
            "no_future": bool(parsed.get("no_future", False)),
        })

    return inputs


def _parse_yaml_scalar(v: str) -> Any:
    v = v.strip()
    if not v:
        return ""
    # Quoted string
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    # Booleans
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    # Integer
    try:
        return int(v)
    except ValueError:
        pass
    # Array
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        # Split on commas that are not inside quotes.
        parts: list[str] = []
        buf = ""
        in_str = False
        quote_ch = ""
        for ch in inner:
            if in_str:
                if ch == quote_ch:
                    in_str = False
                buf += ch
                continue
            if ch in ('"', "'"):
                in_str = True
                quote_ch = ch
                buf += ch
                continue
            if ch == ",":
                parts.append(buf)
                buf = ""
                continue
            buf += ch
        if buf.strip():
            parts.append(buf)
        return [_parse_yaml_scalar(p.strip()) for p in parts]
    return v


def checks_summary_for_mode(mode: str, path: Path | str | None = None) -> list[dict[str, Any]]:
    """Shape returned by GET /api/claim-check/checks — what the frontend needs to render the pending checklist."""
    out: list[dict[str, Any]] = []
    for r in rules_for_mode(mode, path):
        inputs = r.get("inputs") or []
        input_required = any(q.get("required") for q in inputs)
        input_questions: list[dict[str, Any]] | None = None
        if inputs:
            input_questions = [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "type": q["type"],
                    "options": q.get("options"),
                    "required": q.get("required", False),
                    "show_if": q.get("show_if"),
                    "no_future": q.get("no_future", False),
                }
                for q in inputs
            ]
        out.append({
            "id": r["id"],
            "title": r["title"],
            "section_ref": r["act_reference"],
            "search_query": r["search_query"],
            "implementation": r["implementation"],
            "escalate": r.get("escalate"),
            "input_required": input_required,
            "input_questions": input_questions,
        })
    return out
