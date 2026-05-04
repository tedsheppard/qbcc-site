# Sopal BIF Research

Westlaw-grade natural-language Q&A over Queensland construction law.

A research tool for BIF Act / BIF Regulation / QBCC Act / Acts Interpretation
Act, the Queensland security-of-payment line of judgments (QSC + QCA), and
QBCC adjudication decisions, with mechanically-verified verbatim quote
extraction.

## What makes this different from a generic RAG demo

The core architectural property: **the model never produces citation strings
or quote text.** It produces only chunk identifiers and span markers; the
post-processor mechanically extracts the verbatim quote from the indexed
chunk and renders the human-readable citation from chunk metadata.

This makes hallucinated quotes and invented citations architecturally
impossible. The verifier confirms it on every answer.

See `STATUS.md` for the build log and `AUDIT.md` for the corpus inventory.

## Architecture (post knowledge-augmented upgrade)

```
question + conversation history
  |
  v
[query rewrite]   : if history present, rewrite follow-ups to be self-contained
  |
  v
planner.py        : KNOWLEDGE-AUGMENTED — names provisions and authorities
                    using the model's prior legal knowledge, sets confidence
                    and is_fringe flag.  Chain: claude-opus-4-7 ->
                    claude-opus-4 -> claude-sonnet-4-5 -> gpt-5.5 -> ...
  |
  v
retriever.py      : THREE-CHANNEL RETRIEVAL
                    1. hybrid BM25 + dense (Chroma) + RRF + intent weights,
                       per reformulated query, merged
                    2. named provisions resolved deterministically via
                       name_index.lookup_provision() -> chunk_ids
                    3. named authorities resolved via
                       name_index.lookup_case() -> chunk_ids
                    Channels deduped, intent-weighted (named hits exempt
                    from re-weighting), top-k.
  |
  v
answerer.py       : structured-output JSON (gpt-5.5 chain), only chunk_ids
                    and span markers; never raw citation strings.
                    PLANNER FINDINGS block in user message tells the model
                    which authorities to lead with. Structured-uncertainty
                    mode replaces refusal for fringe-but-in-corpus questions.
  |
  v
postprocessor.py  : mechanical quote extraction (whitespace-tolerant span
                    location), citation index assignment, HTML render
  |
  v
verifier.py       : every quote verbatim in its chunk; drop unsupported
  |
  v
final answer  +  source panel
```

Why three channels: pure RAG can only surface what BM25/cosine ranks
highly, which buries leading authorities under generically similar
chunks. Letting the planner *name* the controlling provisions and cases
(channels 2 and 3) puts them on the answerer's desk reliably, while the
hybrid channel (1) provides whatever else the question touches. The
answerer's chunk-level no-hallucination guarantee is preserved because
it can only cite what was retrieved.

Name index build: `python3 -m services.bif_research.name_index build`

## Corpus indexed

| Source | Files | Chunks |
|---|---|---|
| Queensland Acts and Regulations | 5 | 1,158 |
| Annotated BIF Act (v29 March 2026) | 246 | 390 |
| Queensland security-of-payment judgments (QSC + QCA) | 393 | 12,732 |
| QBCC adjudication decisions | (deferred) | (0 — see STATUS.md) |
| **Total** | | **14,280** |

Embedding model: `text-embedding-3-small` (chosen over -large to stay under
the build budget — see STATUS Phase 3).

## Running locally

Set the API key (the project sources it from a sibling `.env.local`):

```sh
export OPENAI_API_KEY=$(grep '^OPENAI_API_KEY=' astruct/astruct-next/.env.local | cut -d= -f2-)
```

Start the API server:

```sh
uvicorn services.bif_research.api:app --port 8000 --reload
```

Open `http://localhost:8000` for the Claude.ai-style frontend, or hit the
API directly:

```sh
curl -X POST http://localhost:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the test for a valid payment claim under the BIF Act?"}'
```

## Re-building the index

```sh
# Walk all sources, chunk, embed, write store/
python -m services.bif_research.indexer

# Smaller / cheaper rebuild — statutes + judgments + annotated only
python -m services.bif_research.indexer --skip-decisions

# Cap the decision corpus
python -m services.bif_research.indexer --decisions-limit 3000

# Project cost only (no API calls)
python -m services.bif_research.indexer --dry-run
```

## Running the eval

```sh
# Full pipeline with the relevance judge
python -m services.bif_research.eval.run_eval --pipeline full --label run-1

# Naive baseline (BM25-only)
python -m services.bif_research.eval.run_eval --pipeline naive

# Skip the judge (free, no relevance score)
python -m services.bif_research.eval.run_eval --pipeline full --no-judge

# Stub pipeline (smoke test of the harness)
python -m services.bif_research.eval.run_eval --pipeline stub --no-judge
```

Results land in `eval/results/<timestamp>-<pipeline>-<label>.json`. Aggregate
metrics print to stdout.

## Final eval results

| Metric | Target | Final | Status |
|---|---|---|---|
| Citation precision | 100% | **100.0%** | ✓ met |
| Quote fidelity | 100% | **100.0%** | ✓ met |
| Answer relevance (judge, 0-5) | ≥4.0 | **4.00** | ✓ met |
| Aggregate score | ≥85% | **87.1%** | ✓ met |
| Citation recall | ≥80% | **70.3%** | ✗ missed by 9.7 pts |

The full per-question and per-category breakdown is in
`eval/results/final.json`. The build log including the
improvement-iteration history is in `STATUS.md`.

The recall miss is concentrated in the `case_law` category (15.3%).
That category has two known failure modes documented in STATUS.md:
JSON-output truncation on long answers (2/6 questions), and
right-answer / wrong-citation where the system gives a substantively
correct answer but cites a different Qld case that says the same thing
rather than the specific leading authority the eval expected (4/6
questions). Several of the expected leading cases (Brodyn = NSW;
Southern Han = HCA) sit outside the Qld-only judgment corpus.

Total build cost: **$12.83 USD** (under the $13 hard cap).

## Hard constraints (do not relax)

These are the architectural guarantees. They are tested by the eval harness
and they are the reason this tool can be trusted with real legal work.

1. **Model never emits citation strings.** Only chunk_ids. Postprocessor
   resolves to human-readable citations from metadata.
2. **Model never emits quote text.** Only chunk_id + span_start + span_end.
   Postprocessor mechanically extracts verbatim text. Quotes that fail to
   locate are dropped — the answerer is given one retry to fix or remove
   the proposition.
3. **Refusal when retrieval is weak.** If the top chunk score is below
   threshold, the system says "I can't answer this with confidence" rather
   than guess. Better than a confident wrong answer.
4. **Eval-driven iteration only.** Phase 5 changes must beat the previous
   eval by ≥2 aggregate points or be reverted.

## File structure

```
services/bif_research/
├── README.md                     this file
├── STATUS.md                     build log + decision log
├── AUDIT.md                      Phase 1 corpus + repo audit
├── BUDGET.md                     spend tracking (every API call)
├── budget.py                     hard-cap enforcement
├── llm_config.py                 OpenAI client wrapper, model fallback
├── indexer.py                    one-time index build
├── retriever.py                  BM25 + dense + fusion + weighting
├── planner.py                    query understanding (small model)
├── answerer.py                   structured-output composer (gpt-5.5)
├── postprocessor.py              mechanical quote extraction
├── verifier.py                   citation + quote validation
├── pipeline.py                   NaivePipeline + FullPipeline
├── api.py                        FastAPI app
├── corpus/
│   ├── base.py                   Chunk dataclass
│   ├── statutes.py               Acts + Regs chunker
│   ├── judgments.py              Qld judgment chunker
│   ├── decisions.py              Adjudication decision chunker (qbcc.db)
│   └── annotated.py              Annotated BIF Act chunker
├── eval/
│   ├── eval_set.jsonl            30 questions across 7 categories
│   ├── schema.py                 EvalQuestion + ExpectedSource
│   ├── build_eval_set.py         re-emits eval_set.jsonl
│   ├── run_eval.py               harness + scorer
│   └── results/                  per-run scores (gitignored)
├── store/
│   ├── chunks.sqlite             chunk_id -> source_id, text, metadata
│   ├── bm25.pkl                  pickled BM25Okapi + token corpus
│   └── chroma/                   Chroma persistent vector store
└── web/
    ├── index.html                Claude.ai-style shell
    ├── styles.css                light, sober, sidebar 260px
    └── app.js                    SSE client + source-panel logic
```

## Notes for next iteration (post-v1)

Things deliberately deferred:

- **Adjudication decisions** are not yet in the index. ~110K chunks would
  add ~$0.30 of embedding cost and ~70 minutes of indexing wall time.
  Source-type weight is already configured to deprioritise them when
  judgments and statutes can answer the question; they shine on
  "have other adjudicators approached this scope before" questions.
- **Drafting features** ("draft me a notice of charge") are out of scope
  for v1. The architecture supports adding them as a separate endpoint.
- **Citator features** ("is this case still good law") would need a
  subsequent-treatment graph; out of scope for v1.
- **Larger embedding model** (`text-embedding-3-large`) was ruled out by
  the $13 cost cap. Future runs at higher budget should use it — expected
  recall improvement on case-law questions specifically.
- **Re-ranker** (e.g. Cohere rerank or a cross-encoder) wasn't needed to
  hit the v1 thresholds. Keep in reserve for harder questions.
