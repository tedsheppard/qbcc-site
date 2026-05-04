# Sopal BIF Research — AUDIT.md

Phase 1 output. What is in the repo and what we are reusing.

## Source corpora available

| Corpus | Path | Files | Size | Tokens (est.) |
|---|---|---|---|---|
| Queensland Acts (text) | `exports/qld_acts_txt/` | 5 | ~1.1 MB | ~270 K |
| Queensland judgments (security of payment) | `exports/qld_judgments_txt/` | 393 | ~150 MB | ~37 M |
| Annotated BIF Act v29 (per-section) | `bif_guide_build/v3/source/annotated/` | 246 | ~700 KB | ~175 K |
| QBCC adjudication decisions (full text) | `qbcc.db` (SQLite) | 7,483 rows | 292.7 M chars | ~73 M |

**Acts available** (`exports/qld_acts_txt/`):
- `act-1954-003.txt` — Acts Interpretation Act 1954 (Qld) — 118 KB
- `act-1991-098.txt` — Queensland Building and Construction Commission Act 1991 — 495 KB
- `act-2017-043.txt` — Building Industry Fairness (Security of Payment) Act 2017 — 312 KB
- `sl-2018-0016.txt` — BIF Regulation 2018 — 44 KB
- `sl-2018-0138.txt` — QBCC Regulation 2018 — 110 KB

**Total corpus tokens: ~110 M.**

## Judgments — paragraph marker convention

Each judgment file begins with a 3-line header:

```
CITATION: <case name> [YEAR] COURT NN
COURT: <COURT_CODE>
YEAR: <YYYY>
```

Followed by a structured court header (PARTIES, JUDGES, ORDERS, CATCHWORDS, etc.), then the body with paragraph markers `[1]`, `[2]`, etc. at the start of each numbered paragraph.

**Caveat for the judgment chunker:** the body also contains year references like `[2008]` from inline citations — the paragraph regex must require the bracketed number to be at the start of a line and not be a 4-digit year:

```python
PARA_RE = re.compile(r'^\s*\[(\d{1,4})\]\s', re.MULTILINE)
# accept matches where int(group(1)) < 1000 OR > 2100 (paragraph numbers, not years)
```

Filename convention is `COURT_YYYY_N.txt` (e.g. `QCA_2008_213.txt`, `QSC_2017_85.txt`). The metadata CSV at `exports/qld_judgments_security_of_payment_cases.csv` provides citation, court, year, number, url, source_pdfs.

## qbcc.db schema

11 tables. Two relevant for indexing:

- **`docs_fresh`** (7,483 rows): `ejs_id` PK, `reference`, `pdf_path`, `full_text`, `id`. Of these, 7,465 have `full_text` longer than 1,000 chars.
- **`decision_details`** (7,483 rows): `ejs_id` PK joined to `docs_fresh`. Useful columns: `adjudicator_name`, `claimant_name`, `respondent_name`, `claimed_amount`, `payment_schedule_amount`, `adjudicated_amount`, `decision_date`, `sections_referenced`, `outcome`, `act_category`, plus `raw_json` for the full extracted metadata blob.

Full-text already indexed via `fts` (FTS4 virtual table over `docs_fresh.full_text`). We will **not** reuse this — we are building a unified BM25 over all chunk text in `services/bif_research/store/bm25.pkl` so retrieval is consistent across source types.

The full_text content is OCR-derived and quality varies (sample preview included `"rsua"t to B, ,itdim grid Constr"ctio"` — clearly OCR errors). The chunker should be tolerant; the headers we add will make these chunks identifiable even with garbled body text.

## Annotated BIF Act per-section files

`bif_guide_build/v3/source/annotated/` contains 246 files, one per section (e.g. `section_068.txt`). Each file starts with metadata header lines (`# Annotated BIF Act source — Section 68`, etc.) and then the annotated content with style tags preserved (`[2 Com-BIFSOPA Heading 1]`, etc.).

Style tag legend (carried over from Phase 4 of the BIF Act guide build):
- `[2 Com-BIFSOPA Heading 1]` — section header
- `[2.1 Com-BIFSOPA Heading 2]` — A/B sub-headers (Legislation / Commentary)
- `[1 BIFSOPA Heading]` — legislation heading
- `[1.x BIFSOPA level X (CDI)]` — legislation text
- `[2.5 Com-BIFSOPA Normal Body Quote]` — block quotes from cases
- `[2.4 Com-BIFSOPA CDI Normal Body Text]` — author commentary

The annotated chunker will strip the style-tag prefixes from output chunk text (they are noise to the model) but keep them visible in the chunk metadata for debugging.

## Existing infrastructure — what is reusable

| Asset | Reuse? | Reason |
|---|---|---|
| `services/claim_check/llm_config.py` | **Yes** — extend, do not modify | Existing fallback chain `gpt-5.4-mini → gpt-5-mini → gpt-4o-mini → gpt-4o` and the `MODEL_CHAIN_MAX` `gpt-5.4 → gpt-5 → gpt-4o → gpt-4-turbo` cover what we need. We will add `gpt-5.5` to the front of `MODEL_CHAIN_MAX` for the answerer. Cost-cap mechanism is reusable. |
| `chroma_db/` (existing `adjudication_decisions` collection, 3.7 GB) | **No** | Built with a different chunking strategy (paragraph-aware + section context per `build_rag.py`) and unknown embedding model. Per spec section 3.2, we treat existing infra as raw material, not a foundation to extend. We build a fresh Chroma store at `services/bif_research/store/chroma/`. |
| `qbcc.db` `fts` virtual table | **No** | Indexed at decision granularity, not chunk granularity. Building unified BM25 over all chunks (statutes + regs + judgments + decisions + annotated) at `services/bif_research/store/bm25.pkl`. |
| `services/claim_check/retrieval.py` | **No** | Stub that raises NotImplementedError. |
| `test_rag_server.py` + `test_rag.html` | **No** | Useful as reference (agentic RAG demo, gpt-5-mini routing + Claude Sonnet 4.6 final answer) but architecture is wrong for our purpose: Claude as final answerer, no mechanical quote extraction, no eval harness. Do not extend. |

## Python environment

| Dep | Status |
|---|---|
| `chromadb` 1.5.5 | installed |
| `openai` 2.6.1 | installed |
| `fastapi` 0.120.1 | installed |
| `uvicorn` 0.38.0 | installed |
| `rank_bm25` | installed (this audit, via `--break-system-packages`) |
| `tiktoken` | installed (this audit) |

Python: macOS system Python 3.14.

## Embedding model decision

Cost projection at `text-embedding-3-large` ($0.13 / 1M tokens) for ~110 M corpus tokens: **~$14.30 — exceeds the $13 hard cap on its own.**

Cost at `text-embedding-3-small` ($0.02 / 1M tokens): **~$2.20 — well inside cap.**

**Decision (per spec Phase 3 instruction): use `text-embedding-3-small`.** Logged to STATUS.md and BUDGET.md. Quality trade-off accepted; chunking + hybrid retrieval should mitigate.

## Decision: scope of adjudication-decision indexing

7,465 decisions × ~9,800 chars average = ~73 M tokens, dominating the embedding cost. At `text-embedding-3-small` this is ~$1.45 of the embedding budget.

OCR quality on these decisions is mixed. Plan:
- Index all 7,465 decisions but with conservative chunk sizes (target 500 tokens per chunk) so retrieval surfaces the relevant passage rather than a giant noisy block.
- Filter out decisions where `full_text` is null or shorter than 1,000 chars.
- Header line includes `decision_id`, `parties`, `adjudicator`, `decision_date`, `sections_referenced` so even garbled OCR bodies can be located via the header.

If during Phase 3 the projected adjudication-decision embedding cost exceeds $3 alone, the chunker will sample down to most-recent 3,000 decisions and log to STATUS.md.
