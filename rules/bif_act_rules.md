# BIF Act rule set for /claim-check

> **This file is a PLACEHOLDER.** The user authors the rules; Claude builds
> the engine that applies them. Do not fill this in speculatively — every
> unauthored rule is a potential source of incorrect legal output.

## How the engine reads this file

The rule engine (`services/claim_check/rule_engine.py`, stage 5) parses
each level-2 heading as a rule and reads the following structured fields.

Required fields per rule:

- **ID** — unique string, e.g. `PC-001`
- **Applies to** — any of: `payment_claim_outgoing`, `payment_claim_incoming`, `payment_schedule_outgoing`, `payment_schedule_incoming` (comma-separated)
- **Check title** — one line, plain English
- **Act reference** — e.g. `s 68(1)(a) BIF Act`
- **Implementation** — one of: `deterministic`, `semantic`, `user-input`
- **Test logic** — plain English description of what passes / fails / warns
- **Meilisearch query** — string used to surface relevant decisions
- **Pass message** / **Fail message** / **Warning message** — text shown in the UI

Additional fields when `Implementation: semantic`:

- **Prompt template** — the exact prompt to send to the LLM. May reference `{{document_text}}` and `{{user_answers}}` placeholders.

Additional fields when `Implementation: user-input`:

- **Questions** — one or more questions to ask the user inline, each with an ID and input type (`date`, `text`, `yes-no`, `number`).

## Example rule (for reference — replace when authoring real rules)

## PC-EXAMPLE-001

- **Applies to:** payment_claim_outgoing, payment_claim_incoming
- **Check title:** Document identifies the construction work claimed for
- **Act reference:** s 68(1)(a) BIF Act
- **Implementation:** semantic
- **Test logic:** A payment claim must identify the construction work (or related goods/services) to which the claim relates. Pass if the document clearly identifies the works. Warn if the identification is vague. Fail if no work is identified.
- **Prompt template:** You are checking whether the following document identifies the construction work the payment claim relates to. Document: {{document_text}}. Reply JSON: {"status": "pass|warning|fail", "quote": "...", "reason": "..."}
- **Meilisearch query:** identify construction work s 68
- **Pass message:** The document identifies the construction work, as required by s 68(1)(a).
- **Warning message:** Identification of the construction work is present but vague — consider tightening.
- **Fail message:** No clear identification of the construction work. This is an s 68(1)(a) issue that typically invalidates the payment claim.

---

**Begin authoring real rules below this line.**
