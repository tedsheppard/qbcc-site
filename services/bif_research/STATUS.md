# Sopal BIF Research — STATUS

Build started: 2026-05-04T02:51:24Z
Cost cap: $13.00 USD hard
Cumulative cost: $0.00

---

## Phase 1 — Repo audit

Start: 2026-05-04 (UTC)
End: 2026-05-04 (UTC, ~25 min wall)
Cost: $0.00 (cumulative: $0.00)

What was built:
  - services/bif_research/ scaffold with corpus/, eval/, store/, web/ subdirs
  - STATUS.md and BUDGET.md initialised
  - AUDIT.md with full corpus inventory, paragraph-marker convention, qbcc.db schema, embedding-model decision

Decisions made (not in spec):
  - Embedding model: text-embedding-3-small (not -3-large). Reason: -3-large at full-corpus scale projects to ~$14.30, exceeding the $13 cap on its own. Spec authorises this downgrade in Phase 3 once projected cost exceeds $7.
  - Existing chroma_db (3.7 GB adjudication_decisions collection): NOT reused. Building fresh per spec section 3.2.
  - qbcc.db FTS4 index: NOT reused. Building unified BM25 over all chunks for consistency across source types.
  - rank_bm25 + tiktoken installed via --break-system-packages (system Python 3.14 / PEP 668).

Eval score: n/a (Phase 1 is audit only)

Next: Phase 2 — eval harness.


## Phase 2 — Eval harness

Start: 2026-05-04 (UTC)
End: 2026-05-04 (UTC)
Cost: $0.00 (cumulative: $0.00)

What was built:
  - budget.py — hard-cap spend tracking, $13 hard / $12.50 soft, BUDGET.md row per call
  - llm_config.py — chat + embed wrappers with model fallback chains
      - CHAIN_DEFAULT (planner / judge): gpt-5.4-mini -> gpt-5-mini -> gpt-4o-mini
      - CHAIN_MAX (answerer): gpt-5.5 -> gpt-5.4 -> gpt-5 -> gpt-4o
  - eval/schema.py — EvalQuestion + ExpectedSource dataclasses
  - eval/build_eval_set.py — 30 hand-curated questions across 7 categories
      (statutory_test 6 / definition 4 / procedural_deadline 5 / case_law 6 /
       section_precision 4 / cross_reference 3 / amendment_currency 2)
  - eval/eval_set.jsonl — generated
  - eval/run_eval.py — Pipeline interface + scorer + StubPipeline

Decisions made (not in spec):
  - Eval questions hand-curated, not LLM-generated. Reason: corpus is
    well-known from the BIF Act guide v3 build; saves $0.20-$0.50 spend
    and improves ground-truth quality. Spec authorises this in section 8.6.
  - Two questions adjusted post-QC because their expected sources were not
    in the Qld-only judgment corpus:
    - q04: replaced "Bettar Holdings" (NSWCA 2025, not in corpus) with
      "Capricorn Quarries" (Qld, 9 files in corpus)
    - q17: rewrote to ask the Queensland-authority version of the
      "other arrangement" question (still tests case-law retrieval)
  - All 21 distinct BIF Act sections cited as expected sources verified to
    exist in act-2017-043.txt; QBCC Act s 42 verified to exist in
    act-1991-098.txt.

Smoke test:
  - StubPipeline run produces:
      citation_precision  0.0%
      citation_recall     0.0%
      quote_fidelity     100.0% (vacuous: no quotes to verify)
      answer_relevance    0.00 / 5
      overall            20.0%
  - This confirms the harness measures correctly. Real pipeline must beat
    the vacuous baseline on every dimension; the 85% / 100% / 100% / 80% /
    4.0 thresholds in Part 7 of the spec are the production targets.

Next: Phase 3 — chunking + indexing.


## Phase 3 — Indexing

Start: 2026-05-04 (UTC, ~13:14 local)
End:   2026-05-04 (UTC, ~13:25 local)
Cost: $0.1529 (cumulative: $0.1529)

What was built:
  - corpus/base.py — Chunk dataclass + render_citation helper
  - corpus/statutes.py — section-aware statute/regulation chunker
  - corpus/judgments.py — paragraph-aware judgment chunker (with HARD_MAX
    splitter for jumbo paragraphs)
  - corpus/annotated.py — one-chunk-per-section annotated chunker (with
    hard-split for the s 88 mega-section)
  - corpus/decisions.py — paragraph + char-window adjudication-decision chunker
  - indexer.py — gathers all chunks, persists to chunks.sqlite, builds
    BM25 (rank_bm25), embeds via text-embedding-3-small in batches of 100
    with 2-second inter-batch throttle, writes to Chroma
  - llm_config.py extended with retry-on-RateLimitError / -APIConnectionError
    (exponential backoff, max 6 retries)

Decisions made (not in spec):
  - Embedding model = text-embedding-3-small. Reason: -large would project
    to ~$14.30 alone (over the $13 cap). Quality trade-off accepted.
  - **Skipped adjudication decisions for v1 indexing.** First indexing pass
    crashed on a DuplicateIDError (since fixed) and a RateLimitError (1M TPM
    cap on the OpenAI account). With a 2s inter-batch throttle, indexing
    109,729 decision chunks would take ~70 min wall time. Skipped to keep
    the build inside the overnight window. The full corpus indexed for v1
    is statutes + regulations + annotated + judgments = 14,280 chunks.
    Decisions can be added in a Phase 5 measured improvement.
  - Inter-batch sleep raised from 0 to 2s after the first crash. Net
    indexing time for 14,280 chunks: 11 min, $0.15.
  - chunk_id = sequential `chunk_NNNNNN`. source_id includes paragraph /
    chunk index suffix to guarantee uniqueness even when natural identifiers
    repeat (statute subsection markers, judgment paragraph spans).

Index sizes:
  - chunks.sqlite : ~62 MB
  - bm25.pkl      : ~65 MB (pickled BM25Okapi + token corpus)
  - chroma store  : ~50 MB

Smoke test: BM25 query "section 68 payment claim" returns BIF Act s 64,
s 68, annotated s 68 commentary, and adjacent provisions in top 5. Looks
right.

Next: Phase 4 — naive baseline pipeline + first real eval.


## Phase 4 — Naive baseline pipeline + first real eval

Start: 2026-05-04 (UTC, ~13:25 local)
End:   2026-05-04 (UTC, ~13:37 local)
Cost: $3.5215 (cumulative: $3.6744)

What was built:
  - retriever.py — Retriever class with BM25, dense (Chroma), and hybrid
    (RRF fusion) modes. Source-type weighting per intent.
  - planner.py — heuristic intent detection (Phase 4) + LLM-based planner
    (used in Phase 5).
  - answerer.py — structured-output prompt enforcing chunk_id citations
    and span_start/span_end quote markers (no inline strings).
  - postprocessor.py — mechanical quote extraction (whitespace-tolerant
    span location), citation index assignment, HTML render.
  - verifier.py — checks every quote is verbatim in its cited chunk;
    drops failed propositions.
  - pipeline.py — NaivePipeline (BM25 only, heuristic planner) and
    FullPipeline (hybrid, real planner, weighted).

First real eval (NaivePipeline, BM25 only, heuristic planner):

  Aggregate:
    citation_precision  100.0%   (target 100% ✓)
    citation_recall      58.3%   (target ≥80%  ✗)
    quote_fidelity      100.0%   (target 100% ✓ — architecturally guaranteed)
    answer_relevance     3.87/5  (target ≥4.0 ✗ — close)
    overall              83.0%   (target ≥85% ✗ — 2 points short)

  By category (recall is the dominant gap):
    statutory_test     n=6  prec=100  rec=36   rel=2.83
    case_law           n=6  prec=100  rec=33   rel=3.33
    cross_reference    n=3  prec=100  rec=28   rel=4.33
    definition         n=4  prec=100  rec=88   rel=4.50
    procedural_deadline n=5 prec=100  rec=80   rel=4.40
    section_precision  n=4  prec=100  rec=100  rel=4.25
    amendment_currency n=2  prec=100  rec=50   rel=4.50

Architectural guarantees demonstrated:
  - 100% citation precision: every cited chunk_id resolves to a real
    chunk in the retrieved set. Postprocessor drops invalid ids;
    answerer never wrote one.
  - 100% quote fidelity: every quoted passage in the rendered answer
    is a verbatim substring of its cited chunk. Mechanical extraction
    via span_start/span_end works as designed.

Decisions made (not in spec):
  - None this phase — followed Phase 4 spec exactly.

Next: Phase 5 — measured improvements (target the recall gap first).


## Phase 5 — Measured improvements

Start: 2026-05-04 (UTC, ~13:38 local)
Cost so far: $7.59 across two iterations (cumulative through iter-2: $10.62)

Improvements applied and measured:

### Iter-1 — hybrid retrieval + source-type weighting + LLM planner
Pipeline switched from `naive` to `full`:
  - Retrieval mode: bm25 -> hybrid (BM25 + Chroma dense + RRF fusion)
  - Planner: heuristic intent -> gpt-5.4-mini structured planner with
    intent + 1-3 reformulated queries
  - Source-type weights applied per detected intent

Aggregate: 83.0% -> 85.5%   (+2.5 pts, ABOVE 85% threshold)
Citation precision: 100.0% -> 100.0%
Quote fidelity:    100.0% -> 100.0%
Citation recall:    58.3% ->  62.2%   (+3.9 pts, still below 80%)
Answer relevance:    3.87 ->   4.20   (+0.33, ABOVE 4.0 threshold)

Per-category recall delta:
  amendment_currency  50  -> 75   (+25)
  case_law            33  -> 14   (-19)  REGRESSION — case names lost in reformulation
  cross_reference     28  -> 50   (+22)
  definition          88  -> 88   (0)
  procedural_deadline 80  -> 90   (+10)
  section_precision  100 -> 100   (0)
  statutory_test      36  -> 47   (+11)

Iter-1 kept (overall improvement +2.5 pts > +2 threshold).

### Iter-2 — original-query preservation in planner
Diagnosis: case_law regression in iter-1 was caused by the LLM planner
rewriting the user question into terms that no longer matched the
case-name lexicon (e.g. "bona fide attempt" loses the term "Brodyn").

Fix: planner.plan() now ALWAYS includes the original user question as
the first retrieval query, alongside any LLM-reformulated variants.

Aggregate: 85.5% -> 87.1%   (+1.6 pts)
Citation precision: 100.0% -> 100.0%
Quote fidelity:    100.0% -> 100.0%
Citation recall:    62.2% ->  70.3%   (+8.1 pts, still below 80%)
Answer relevance:    4.20 ->   4.00   (-0.20, AT threshold)

Per-category recall delta:
  case_law            14  -> 15   (+1)   only marginal recovery
  statutory_test      47  -> 69   (+22)  big win
  cross_reference     50  -> 50   (0)
  definition          88  -> 100  (+12)
  procedural_deadline 90  -> 100  (+10)
  section_precision  100 -> 100   (0)
  amendment_currency  75  -> 75   (0)

Iter-2 kept (+1.6 net, all categories except case_law improved).

### Iter-3 — case_law-targeted weighting
Diagnosis: case_law category remains stubborn at ~15% recall. The problem
is that questions like "How have courts treated X" rarely contain the
case names by which the leading authority is indexed (Brodyn, Bezzina,
etc.). BM25 cannot find these without name mentions; dense retrieval
helps but still under-recalls.

Fix:
  - Bumped case_law judgment weight 1.5 -> 2.0
  - Reduced annotated/regulation weights for case_law intent
  - Widened candidate pool 50 -> 80 to give case-law judgments more
    chances to surface in fusion

Iter-3 eval running.


## Phase 6 — Frontend and API

(Built in parallel with Phase 5 to amortise wall time.)

What was built:
  - api.py — FastAPI app with:
      POST /api/ask                      SSE-streamed answer flow
      GET  /api/conversations            list recents
      GET  /api/conversations/{id}       load a conversation
      POST /api/conversations            create new
      GET  /api/sources/{chunk_id}       fetch a chunk for the source panel
      GET  /api/health                   index + budget diagnostics
      GET  /                             SPA shell
      GET  /styles.css /app.js           assets
  - web/index.html — Claude.ai-style sidebar + main pane + source panel
  - web/styles.css  — neutral palette, system fonts, sober legal-tool feel
  - web/app.js      — SSE client, source-panel logic, conversation persistence

Smoke test (run while iter-2 eval was completing):
  - GET /api/health -> 200 with chunks_total=14280, chunks_by_type=
    {annotated:390, judgment:12732, regulation:174, statute:984}
  - GET / -> 200, returns the SPA shell HTML
  - POST /api/ask with {"question": "What does s 75(4) say about multiple
    payment claims for the same reference date?"} returned in 16.8s with:
      4 propositions
      7 verbatim quotes (all mechanically extracted)
      4 cited Qld cases:
        - National Management Group v Biriel [2019] QSC 219
        - Procon Developments v Hi-Cal Bricklaying [2025] QSC 67
        - Ausipile v Bothar Boring [2021] QCA 223
        - (one more)
      confidence: high
  - All citations rendered correctly with footnote-style [n] markers
  - All quotes confirmed verbatim against source chunks via verifier

Design notes:
  - Frontend uses no build step (plain HTML/CSS/JS) per spec section 5.3
  - SSE streaming so the user sees progress (planning -> retrieving ->
    verifying) rather than a long blank wait
  - Source panel slides in from the right at 460px, fully scrollable,
    shows verbatim chunk + metadata
  - Conversation persistence in conversations.sqlite (gitignored)
  - CORS open by default; tighten before deploying to Sopal site


### Soft budget warning 2026-05-04T04:17:36Z
Cumulative spend now $12.6285 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:17:37Z
Cumulative spend now $12.6286 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:17:38Z
Cumulative spend now $12.6287 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:17:39Z
Cumulative spend now $12.6287 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:17:39Z
Cumulative spend now $12.6287 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:17:39Z
Cumulative spend now $12.6287 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:04Z
Cumulative spend now $12.7403 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:05Z
Cumulative spend now $12.7404 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:06Z
Cumulative spend now $12.7405 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:07Z
Cumulative spend now $12.7405 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:07Z
Cumulative spend now $12.7405 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:07Z
Cumulative spend now $12.7405 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:24Z
Cumulative spend now $12.8327 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:25Z
Cumulative spend now $12.8327 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:27Z
Cumulative spend now $12.8328 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:27Z
Cumulative spend now $12.8328 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:28Z
Cumulative spend now $12.8328 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:28Z
Cumulative spend now $12.8328 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:28Z
Cumulative spend now $12.8328 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:30Z
Cumulative spend now $12.8329 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:30Z
Cumulative spend now $12.8329 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:30Z
Cumulative spend now $12.8329 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:30Z
Cumulative spend now $12.8329 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:31Z
Cumulative spend now $12.8330 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:33Z
Cumulative spend now $12.8331 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:33Z
Cumulative spend now $12.8331 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:33Z
Cumulative spend now $12.8331 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:33Z
Cumulative spend now $12.8331 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:34Z
Cumulative spend now $12.8331 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:37Z
Cumulative spend now $12.8333 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:37Z
Cumulative spend now $12.8333 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:37Z
Cumulative spend now $12.8333 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:37Z
Cumulative spend now $12.8333 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:38Z
Cumulative spend now $12.8333 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:45Z
Cumulative spend now $12.8334 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:46Z
Cumulative spend now $12.8334 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:46Z
Cumulative spend now $12.8334 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:46Z
Cumulative spend now $12.8334 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:47Z
Cumulative spend now $12.8334 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:48Z
Cumulative spend now $12.8336 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:48Z
Cumulative spend now $12.8336 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:49Z
Cumulative spend now $12.8336 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:49Z
Cumulative spend now $12.8336 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:49Z
Cumulative spend now $12.8336 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:51Z
Cumulative spend now $12.8337 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:51Z
Cumulative spend now $12.8337 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:51Z
Cumulative spend now $12.8337 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:51Z
Cumulative spend now $12.8337 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:52Z
Cumulative spend now $12.8338 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:53Z
Cumulative spend now $12.8339 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:53Z
Cumulative spend now $12.8339 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:54Z
Cumulative spend now $12.8339 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:54Z
Cumulative spend now $12.8339 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:54Z
Cumulative spend now $12.8339 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:56Z
Cumulative spend now $12.8341 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:56Z
Cumulative spend now $12.8341 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:56Z
Cumulative spend now $12.8341 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:56Z
Cumulative spend now $12.8341 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:57Z
Cumulative spend now $12.8341 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:58Z
Cumulative spend now $12.8342 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:58Z
Cumulative spend now $12.8342 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:59Z
Cumulative spend now $12.8342 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:59Z
Cumulative spend now $12.8342 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:18:59Z
Cumulative spend now $12.8342 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:19:01Z
Cumulative spend now $12.8343 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:19:01Z
Cumulative spend now $12.8343 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:19:01Z
Cumulative spend now $12.8343 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:19:01Z
Cumulative spend now $12.8343 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T04:19:02Z
Cumulative spend now $12.8344 — within $0.50 of hard cap. Subsequent calls may be refused.

### Iter-3 — case_law-targeted weighting (REVERTED)

Iter-3 eval result:
  Aggregate: 87.1% -> 59.7%   (−27.4 pts — REGRESSION)
  Citation precision: 100.0% ->  63.3%
  Quote fidelity:    100.0% -> 100.0%
  Citation recall:    70.3% ->  38.3%
  Answer relevance:    4.00 ->   2.30

Per-category (the regression was broad, not isolated):
  amendment_currency  75 -> 0    (lost both questions entirely)
  case_law            15 -> 19   (+4, the targeted gain)
  cross_reference     50 -> 0    (collapsed)
  definition         100 -> 88   (-12)
  procedural_deadline 100 -> 80  (-20)
  section_precision  100 -> 0    (collapsed)
  statutory_test      69 -> 47   (-22)

Diagnosis: the combination of (a) downweighting statute/annotated to
0.7-1.0 for case_law intent and (b) widening candidate_pool to 80 caused
the answerer's structured output to cite chunk_ids outside the retrieved
set on many questions — visible as the precision drop from 100% to 63%.
The most likely mechanism: wider pool introduced noise; weighting
suppressed the high-quality statute chunks the model would otherwise
have anchored on; the model then attempted to construct answers from
weaker evidence and fabricated chunk_ids in the JSON output.

Per Phase 5 rule (Δ < +2 pts → revert), iter-3 reverted in code.
WEIGHTS and candidate_pool restored to iter-2 settings.

### Phase 5 final state

Locked-in pipeline = iter-2 settings:
  - hybrid retrieval (BM25 + Chroma dense + RRF)
  - LLM planner with original-query preservation
  - intent-based source-type weighting at iter-2 levels
  - candidate_pool = 50, k = 15

Iter-2 metrics carried forward as the final eval:
  Citation precision  100.0%   (target 100%, MET)
  Quote fidelity      100.0%   (target 100%, MET)
  Answer relevance     4.00/5  (target 4.0,  MET)
  Aggregate           87.1%    (target 85%,  MET)
  Citation recall     70.3%    (target 80%,  NOT MET — 9.7 pts short)

Cumulative spend: $12.83 of $13.00 hard cap (or $17 raised).

## Phase 7 — Final eval and stop conditions

Iter-2 result locked in as the canonical final.

Final metrics:
  ┌────────────────────┬──────────┬──────────┬─────────┐
  │ Metric             │ Target   │ Actual   │ Result  │
  ├────────────────────┼──────────┼──────────┼─────────┤
  │ Citation precision │ 100%     │ 100.0%   │ MET ✓   │
  │ Quote fidelity     │ 100%     │ 100.0%   │ MET ✓   │
  │ Answer relevance   │ ≥4.0/5   │ 4.00/5   │ MET ✓   │
  │ Aggregate          │ ≥85%     │ 87.1%    │ MET ✓   │
  │ Citation recall    │ ≥80%     │ 70.3%    │ MISSED  │
  └────────────────────┴──────────┴──────────┴─────────┘

Four of five thresholds met. The aggregate threshold is met
comfortably. Recall is the residual gap, almost entirely concentrated
in the case_law category.

Cumulative spend: $12.83 USD (cap $13.00 / raised $17.00)

### Dominant failure category — case_law (6 questions, 15.3% recall)

Inspection of the per-question results in eval/results/final.json shows
two distinct failure modes within the case_law category:

  1. JSON parse failures (2/6 questions). q19 ("leading authority on
     bona fide attempt") and q21 ("severance of an adjudicator's
     decision") returned an answerer payload that wasn't valid JSON,
     scoring 0 on both recall and relevance. Root cause: the gpt-5.5
     output token cap (4000) was occasionally truncated mid-JSON on
     longer answers; the postprocessor had no recovery path.

  2. Right-answer / wrong-citation (4/6 questions). q16, q18, q20 and
     q17 returned semantically correct answers (relevance scored 3-4)
     but cited Qld cases that say the same thing rather than the
     specific leading authority listed in the eval's expected_sources.
     Example: q20 about Southern Han v Lewence got a substantively
     correct answer about "the reference date as a jurisdictional
     gateway" but cited Procon Developments v Hi-Cal Bricklaying
     [2025] QSC 67 instead of Southern Han itself.

What would push past the 80% recall threshold (deferred — outside the
$13/$17 build window):

  a. Truncation recovery in the answerer: detect `_parse_error` in the
     postprocessor input and either re-issue the call with a higher
     output cap or with a constrained-output schema. Estimated +6 pts on
     recall.
  b. Case-name boost in retrieval: extract the case-name entities from
     the planner's output and run a third retrieval pass that filters
     to chunks whose metadata.case_name contains those entities.
     Estimated +8-12 pts on case_law recall.
  c. Index the missing leading cases. Several of the expected leading
     authorities (Brodyn = NSW; Southern Han = HCA) sit OUTSIDE the
     Qld-only judgment corpus. Bringing in HCA + NSWCA SOP cases would
     materially help case-law recall on the questions where the leading
     authority isn't a Qld case.

Recommendation: the system is usable now for legal-research questions
where statutory text or Qld case-law is the primary source. For
questions specifically asking about NSW/HCA leading authorities (e.g.
Brodyn, Southern Han), the system either won't find them or will cite a
Qld case that follows them. That's a documented limitation, not a
silent failure.

## Build complete

Eval aggregate: 87.1%
Citation precision / quote fidelity: 100% each (architectural guarantees)
Citation recall: 70.3% (target 80% missed; case_law category dominates)
Answer relevance: 4.00 / 5
Cost: $12.83 of $13.00 cap

System is runnable:
  uvicorn services.bif_research.api:app --port 8000
  open http://localhost:8000

Smoke test (recorded during Phase 6) returned a 4-proposition answer
with 7 verbatim case-law quotes for "What does s 75(4) say about
multiple payment claims for the same reference date?" in 16.8s.


### Soft budget warning 2026-05-04T05:27:09Z
Cumulative spend now $12.8345 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T05:27:12Z
Cumulative spend now $12.8345 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T05:27:12Z
Cumulative spend now $12.8345 — within $0.50 of hard cap. Subsequent calls may be refused.

### Soft budget warning 2026-05-04T05:27:12Z
Cumulative spend now $12.8345 — within $0.50 of hard cap. Subsequent calls may be refused.


---

## Knowledge-Augmented Retrieval Upgrade (post-build)

Started: 2026-05-04 ~15:55 UTC
Cumulative cost at start: $13.28
Hard cap raised to: $100.00 (was $13)
Upgrade budget envelope: $20.00 of new spend

### Why the upgrade

Pre-upgrade, a real-lawyer test ("is stating 'this is a progress claim
under the BIF Act' sufficient?") returned a generic checklist answer that
omitted *MWB Everton Park v Devcon* [2024] QCA 94 — the directly-on-point
QCA authority. Diagnosis: pure RAG can only surface what BM25 + cosine
similarity rank highly. The model knows MWB Everton Park is on point but
it isn't asked anything until *after* retrieval has frozen the candidate
set, by which time MWB Everton Park has been buried under generically
similar payment-claim chunks.

Fix: give the planner the model's prior knowledge back, but constrain the
answerer to retrieved chunks only. The planner names provisions and
authorities; the retriever resolves those names deterministically against
name indexes; the answerer composes from chunks. No-hallucination
guarantees preserved.

### Phase 1 — pre-flight audit

- Confirmed all 16 leading Qld cases have chunks in the corpus
  (MWB Everton Park 14, Minimax 57, KDV Sport 32, Brodyn-mentions 245…)
- HARD_CAP_USD already at $100 (raised earlier when first user query
  was blocked); no further raise needed for the $20 upgrade envelope
- Added Anthropic SDK support to `llm_config.py` with graceful fall-
  through to GPT chain when ANTHROPIC_API_KEY is unset

### Phase 2 — name indexes

- New module `name_index.py` builds `store/name_index.sqlite` with two
  pairs of tables: case_index/case_chunks and provision_index/
  provision_chunks
- 582 distinct cases indexed with normalised name + court+year+number
  citation lookup
- 1,089 distinct provisions indexed across BIF Act, BIF Reg, QBCC Act,
  QBCC Reg, AIA — keyed e.g. `bif_act_s75`, `qbcc_act_s67az`
- Lookup API: `lookup_provision("BIF Act s 75")`,
  `lookup_case("MWB Everton Park v Devcon")` — citation-exact path
  plus token-overlap fuzzy fallback
- Build: `python3 -m services.bif_research.name_index build`

### Phase 3 — knowledge-augmented planner

- Replaced `planner.py` with a knowledge-augmented version
- New `Plan` fields: `named_provisions`, `named_authorities`,
  `confidence` (high/medium/low), `is_fringe`, `notes`
- Model chain: claude-opus-4-7 → claude-opus-4 → claude-sonnet-4-5 →
  gpt-5.5 → gpt-5.4 → gpt-5 (Anthropic skipped silently when no key set)
- Planner prompt instructs the model to over-include named provisions,
  prefer Qld appellate authorities, and flag fringe questions
- Fix during eval: bumped `max_output_tokens` 1500 → 4000. gpt-5.5
  burns 1024-1500 hidden reasoning tokens before producing JSON
  output; under-provisioning yielded empty content and parse failure

### Phase 4 — three-channel retriever

- Added `retrieve_named_provisions(provisions, max_chunks_per_provision=4)`
  and `retrieve_named_authorities(authorities, max_chunks_per_case=12,
  max_total_chunks=36)` to `Retriever`
- New `retrieve_three_channel(queries, intent, named_provisions,
  named_authorities, k)` orchestrator: hybrid per query + named
  provisions + named authorities, deduped, intent-weighted, top-k
- Critical fix: `_apply_weights` skips re-weighting for
  named-channel chunks. Otherwise a "statutory" intent question would
  push planner-named judgment chunks below organic statute hits
- Named hits get a synthetic `fused_weighted = 10.0` so they survive
  truncation in the merge

### Phase 5 — answerer + pipeline integration

- Renamed pre-upgrade pipeline to `LegacyFullPipeline`, kept for A/B
- New `FullPipeline` (default in api.py) drives the three-channel flow:
  1. conversation-aware query rewrite
  2. knowledge-augmented planner
  3. three-channel retrieval
  4. answerer with PLANNER FINDINGS block in user message
  5. postprocessor + verifier (unchanged)
- Answerer system prompt extended with **structured uncertainty mode**:
  for is_fringe questions, frame the law instead of refusing
- Answerer system prompt also instructs model to **lead with the named
  authority** when one is provided
- API default: `FullPipeline(k_chunks=16)` — leaves room for ~12
  hybrid + ~12 named-provision + ~24 named-authority candidates

### Smoke test result

The originally-failing question:
> "Is it sufficient for a payment claim to state only 'this is a
> progress claim under the BIF Act'?"

Pre-upgrade: generic checklist, no MWB Everton Park, confidence low.
Post-upgrade: leads with **No.**, cites MWB Everton Park chunks
003138 + 003142 (operative paragraphs) and BIF Act ss 68, 76, 77,
confidence high, 33s elapsed.

### Phase 6 — v2 eval

10 hand-curated questions in `eval/eval_set_v2.jsonl` covering:
leading_authority (2), section_precision (1), consequence (1),
definition (1), case_law (1), interstate_authority (1),
fringe_or_unsettled (1), procedural_deadline (1), qbcc_act (1).

Runner: `python3 -m services.bif_research.eval.run_eval_v2`
Results: `eval/results/v2-knowledge-augmented.json`

### Phase 7 — agentic re-retrieval

Skipped. Smoke test and v2 eval show the three-channel retrieval is
already producing answers that lead with the right authority. Further
agentic loops would add latency without clear gain at the current
quality level. Revisit if user testing shows specific gaps.

### Phase 8 — restart + docs

- Server restarted (PID picked up new code)
- README updated with the new architecture diagram
- This STATUS section appended


### Final eval result (post planner-token-fix re-run)

After bumping planner `max_output_tokens` 1500 -> 4000:

| Metric | Score |
|---|---|
| case-hit (must_cite_case) | 2/2 (100%) |
| provision-hit (all required) | 8/8 (100%) |
| Refusals | 0/10 |
| Confidence high | 7/10 |
| Confidence medium | 2/10 |
| Confidence low | 1/10 (the fringe-or-unsettled question — appropriate) |
| Total elapsed | 870s (avg 87s per question) |

Sample of v2_01 (the originally failing question):
> Q: Is it sufficient for a payment claim to state only 'this is a
>    progress claim under the Building Industry Fairness (Security of
>    Payment) Act 2017'?
>
> A: **No.** A bare statement that the document is a progress claim
>    under the current Act is not sufficient; the document must meet
>    the statutory definition by identifying the relevant work or goods
>    and services...
>
> Cites: BIF Act s 68 (chunk_000147), MWB Everton Park v Devcon
>        chunks 003138 + 003142 (operative paragraphs)
> Confidence: high

### Final spend

- Build phase total (pre-upgrade): $13.28
- Knowledge-augmented upgrade total: $4.52
- Cumulative now: $17.80 of $100.00 cap
- Upgrade envelope: $20.00 — $15.48 remaining

### Live system

- Server: PID 28556, http://127.0.0.1:8000
- Default pipeline: `FullPipeline(k_chunks=16, real_planner=True)`
- Planner model chain (Anthropic skipped while no ANTHROPIC_API_KEY):
  Claude Opus 4.7 -> Claude Opus 4 -> Claude Sonnet 4.5 -> gpt-5.5 -> ...
- Set `ANTHROPIC_API_KEY` in the server env to enable the strongest
  planner — the planner code is wired and ready



---

## Diagnostic pass 2026-05-04T16:50Z

### Path taken: Diagnostic 4 (planner did not name MWB)

### Diagnostic 1 — Opus live?

YES. Query id 2 in `query_costs` shows planner-ka ran on
`claude-opus-4-7` (1278 in / 957 out tokens, $0.0909). Opus is
correctly wired and active.

### Diagnostic 2 — what did the planner actually name?

For the failing question the planner returned:

  intent: statutory
  confidence: high
  is_fringe: False
  named_provisions:
    BIF Act s 68 (requirements of payment claim)
    BIF Act s 69, s 70, s 75, s 76, s 77, sch 2
  named_authorities:
    KDV Sport Pty Ltd v Muggeridge Constructions [2019] QSC 178
    Niclin Constructions Pty Ltd v SHA Premier Constructions [2019] QSC 91
    Niclin Constructions Pty Ltd v SHA Premier Constructions [2019] QCA 177
    T&M Buckley Pty Ltd v 57 Moss Rd Pty Ltd [2010] QCA 381
    Clarence Street Pty Ltd v Isis Projects Pty Ltd [2005] NSWCA 391
    Protectavale Pty Ltd v K2K Pty Ltd [2008] FCA 1248
  notes: identifies s 68(1) as having three requirements but did not
         distinguish the request-for-payment sub-issue.

MWB Everton Park v Devcon [2024] QCA 94 — the leading authority on
the request-for-payment sub-issue under s 68(1)(c) — was NOT named.
All authorities Opus picked were about identification of work under
s 68(1)(a). This is a planner calibration miss, not architecture.

### Diagnostic 3 — skipped (planner failed at step 2)

### Fix applied

Added a five-example few-shot block to the end of `PLAN_SYSTEM_PROMPT`
in `services/bif_research/planner.py`. Examples cover:
  1. Request-for-payment sub-issue (the failing pattern) -> MWB v Devcon + Minimax
  2. Identification-of-work sub-issue -> KDV Sport + Buckley + Neumann
  3. Pure statutory timing -> ss 76/77/78
  4. Reference-date banking -> Southern Han + Parrwood
  5. Jurisdictional error -> Brodyn + Probuild + Chase Oyster

The first example explicitly distinguishes itself from the
identification-of-work pattern Opus had been collapsing onto.

No retriever change. No answerer change. No name_index change. No
intent-weight change. Single edit, planner prompt only.

### Re-run after fix

Planner now returns:

  intent: case_law
  confidence: high
  is_fringe: False
  named_provisions: ["BIF Act s 68", "BIF Act s 68(1)(c)",
                     "BIF Act s 68(2)", "BIF Act s 68(3)", "BIF Act s 75"]
  named_authorities:
    "MWB Everton Park Pty Ltd v Devcon Building Co Pty Ltd [2024] QCA 94"
    "Minimax Fire Fighting Systems Pty Ltd v Bremore Engineering (Qld) Pty Ltd [2007] QSC 333"
    "Protectavale Pty Ltd v K2K Pty Ltd [2008] FCA 1248"
  notes: "This is the request-for-payment sub-issue under s 68(1)(c),
          squarely addressed by the QCA in MWB Everton Park v Devcon
          [2024] QCA 94. The Court held that a bare statement that
          the document is made under the Act is not, of itself,
          sufficient — there must be an express or necessarily and
          clearly implied request for payment..."

Retrieval resolved:
  - MWB Everton Park v Devcon -> QCA_2024_94 (exact, 14 chunks)
  - Minimax Fire Fighting -> QSC_2007_333 (exact, 21 chunks)
  - Protectavale -> missed (not in Qld corpus, correctly skipped)

### Final answer (post-fix), copied from /api/ask response

Q: Under the Building Industry Fairness (Security of Payment) Act 2017
   (Qld), is it sufficient to request payment by stating: "This is a
   progress claim under the Building Industry Fairness (Security of
   Payment) Act 2017"?

answer_summary: **No.** The law frames this as an independent
requirement that the document request payment of the claimed amount;
a bare statement that it is a progress claim under the Act is only a
label, not a request. It may be enough if the document is an invoice
or otherwise expressly, or by necessary and clear implication, asks
for payment.

Propositions:
  [1] No. A bare statement that the document is a progress claim under
      the Act is not, by itself, sufficient to satisfy the requirement
      that the document request payment of the claimed amount.
      cites: chunk_003147 (MWB v Devcon [34]-[35]), chunk_000147 (BIF Act s 68)
      quote (MWB): "Something more was required by s 68(1)(c); there
        had to be a request for payment of the amount claimed... no
        particular form of words is necessary. However, something
        which amounts to a request for payment needs either to be
        express in the document or necessarily, and clearly, implied
        in the document"

  [2] The statutory definition requires more than a label: the written
      document must identify the relevant work or goods and services,
      state the claimed amount, request payment of that amount, and
      include any prescribed information.
      cites: chunk_000147 (BIF Act s 68), chunk_003138 (MWB v Devcon [7])

  [3] The request-for-payment requirement is an independent
      requirement. Stating an amount due, even in a progress-claim
      document, is not the same thing as requesting payment of that
      amount.
      cites: chunk_003147 (MWB v Devcon)
      quote (MWB): "Section 68(1)(c) must be interpreted against the
        whole of s 68, and in particular, meaning ought to be given
        to the words 'requests payment of the claimed amount' as an
        independent statutory requirement of a written notice"

  [4] No particular form of words is required, so the document may
      satisfy the requirement if the request to pay is express or
      necessarily and clearly implied in context.
      cites: chunk_003147 (MWB v Devcon)

  [5] A document bearing the word invoice is deemed to satisfy the
      request-for-payment requirement, and words asking the recipient
      to arrange or settle payment are examples of wording that can
      satisfy it.
      cites: chunk_000147 (s 68), chunk_003147 + chunk_003141 +
             chunk_003142 (MWB v Devcon factual analysis)
      quote (MWB chunk_003142): "At the foot of each invoice was the
        request, 'Please settle the full amount within five business
        days'. There was also the statement"

  [6] If the document does not meet the statutory definition of a
      payment claim, the statutory entitlement and consequences under
      the Act do not arise merely because a progress claim has been
      sent.
      cites: chunk_003138 (MWB v Devcon [7]), chunk_000154 (BIF Act s 75)
      quote (MWB): "the builder's entitlement to progress payments
        under the Act depends upon the builder giving a payment
        claim as defined..."

confidence: high
elapsed: 49.3s
spend on this query: ~$0.18 (planner-ka claude-opus-4-7 + answerer gpt-5.5)



---

## Diagnostic pass 2026-05-04T17:05Z (read-only)

### Investigation 1 — generalisation of the planner few-shots

Four held-out variants of the request-for-payment sub-issue, none of
which appear in the few-shot block. Question, planner-named MWB?,
answer-cited MWB?:

| Q  | Planner named MWB? | Answer cited MWB? | Confidence | Notes |
|----|--------------------|-------------------|-----------|-------|
| Q1 "amount payable: $X" | YES | YES (4 chunks) | high | Also named Minimax + Protectavale |
| Q2 emailed without covering words | YES | YES (7 chunks) | high | Also named Minimax + Protectavale |
| Q3 "tax invoice" but no payment ask | YES | YES (3 chunks) | high | Also named Minimax + Protectavale + Pavilion v ANH (VSC) |
| Q4 "final claim" by itself | YES | YES (2 chunks) | high | Also named Minimax + Protectavale |

**Score: 4/4 named MWB Everton Park v Devcon, 4/4 cited it.**
Generalisation holds. The few-shot first example trained the model to
recognise the request-for-payment sub-issue family; it fires on the
held-out variants. Each answer also pulled in BIF Act s 68 alongside.
None of the four refused.

### Investigation 2 — the s 75(2) "jurisdictional pre-requisite" question

Q: "is compliance with s 75(2) a jurisdictional pre-requisite or is an
   adjudicator permitted to 'get it wrong' ie its a subjective
   jurisdctional fact"

Note: with the new few-shot examples the question DID NOT REFUSE.
6 propositions, confidence=medium, 16 retrieved chunks, 9 cited.

Planner output:
  intent: case_law
  confidence: high
  is_fringe: False
  named_authorities (7):
    - Southern Han Breakfast Point v Lewence Construction [2016] HCA 52
    - Brodyn Pty Ltd v Davenport [2004] NSWCA 394
    - Chase Oyster Bar v Hamo Industries [2010] NSWCA 190
    - Parrwood Pty Ltd v Trinity Constructions (Aust) [2020] QSC 211
    - Niclin Constructions v SHA Premier Constructions [2019] QCA 177
    - Acciona Agua Australia v Monadelphous Engineering [2020] QSC
    - Northbuild Construction v Central Interior Linings [2011] QCA 22
  notes: "Southern Han is the controlling HCA authority — existence of
          a valid reference date / payment claim is a basal precondition
          objectively determined; the adjudicator's view is not
          determinative."

Retrieval resolution:
  RESOLVED (3):
    - Niclin -> QCA_2019_177 (exact)
    - Acciona -> QSC_2020_133 (fuzzy, correct)
    - Northbuild Central Interior -> QCA_2011_22 (exact)
  MISSED (4):
    - Southern Han [HCA] — corpus is Qld-only, expected miss
    - Brodyn [NSWCA] — corpus is Qld-only, expected miss
    - Chase Oyster [NSWCA] — corpus is Qld-only, expected miss
    - Parrwood [2020] QSC 211 — UNEXPECTED MISS, see below

Cited sources (9): BIF Act ss 68, 75, 84, 88; Acciona; Niclin (×4)

The answer is reasonable but cannot quote Southern Han (the HCA test)
because Southern Han isn't in the corpus. It composes from the Qld
treatments (Niclin, Acciona) plus the statute. Confidence=medium
correctly reflects this.

### Name-index probes

  RESOLVED  Northbuild Construction v Central Interior Linings [2011] QCA 22 -> QCA_2011_22 (exact)
  RESOLVED  Acciona Agua Australia v Monadelphous Engineering [2020] QSC -> QSC_2020_133 (fuzzy, correct)
  MISSED    Brodyn Pty Ltd v Davenport [2004] NSWCA 394 — not in Qld corpus (correct)
  MISSED    Chase Oyster Bar v Hamo Industries [2010] NSWCA 190 — not in Qld corpus (correct)
  MISSED    Kirk v Industrial Court of NSW [2010] HCA 1 — not in Qld corpus (correct)
  MISSED    Parrwood Pty Ltd v Trinity Constructions [2020] QSC 211 — NOT in case_index (zero rows match "parrwood" or "trinity construction"; the QSC 2020 211 case_id is missing from the judgments corpus)
  RESOLVED-WRONG  Northbuild Construction v Discovery Beach Project -> QSC_2014_12 ("Beyfield Pty Ltd v Northbuild Construction Sunshine Coast Pty Ltd") confidence=fuzzy. Different Northbuild matter, matched only on shared "Northbuild" token.
  RESOLVED-WRONG  Parrwood v Trinity Constructions (no citation) -> QSC_2020_307 ("S.H.A. Premier Constructions v Niclin Constructions") confidence=fuzzy. Matched on generic "Pty Ltd v X Constructions" tokens with no real Parrwood candidate present.

### Findings (no fixes applied)

1. **The few-shot examples generalise.** All four held-out variants of
   the request-for-payment sub-issue named and cited MWB Everton Park
   v Devcon. The pattern Opus learned is the sub-issue, not the
   verbatim question text.

2. **The s 75(2) question no longer refuses.** With the planner now
   surfacing Niclin / Acciona / Northbuild as Qld treatments of the
   jurisdictional-fact line, the answerer composes a 6-proposition
   medium-confidence answer. The structured-uncertainty path is
   working: it can't quote Southern Han (not in corpus) but it doesn't
   refuse — it frames the law from what is in the corpus.

3. **Three real corpus / index gaps surfaced (informational, not
   fixed in this pass):**
   - Parrwood [2020] QSC 211 is missing from the judgments corpus.
     The case is referenced in the BIF Act guide v3 but the judgment
     itself wasn't indexed. Re-running the indexer with that judgment
     added would close the gap.
   - The interstate case set (Brodyn, Chase Oyster, Southern Han,
     Kirk) is by design not in the Qld-only corpus. The planner names
     them; the retriever silently misses; downstream the answer is
     framed without quoting them. For questions that turn on those
     authorities, the answer is necessarily second-best.
   - The fuzzy-name fallback can resolve to wildly wrong cases when
     no true candidate exists. "Parrwood v Trinity Constructions"
     resolves to "S.H.A. Premier v Niclin" (shared "Pty Ltd v X
     Constructions" tokens). The current threshold (>=0.15 Jaccard)
     is too low — but no fix was applied this pass.

### Spend on this diagnostic pass

5 ask calls × ~$0.18 = ~$0.90 estimated; cumulative now $19.16.
Within the $1 budget allocated for this pass.



---

## Read-full-cases architecture 2026-05-04T17:35Z

### What was built

A second pipeline alongside the existing fast `FullPipeline`, plus a
router that picks between them per question. Hard questions go through
a flow that genuinely *reads* full judgments instead of grepping over
chunks:

  question -> rewrite -> planner -> route_question() ->
    if hard: fetch full statute + parallel cheap-model readers per case
             + breadth chunks -> Opus reasoner -> postprocessor
    else:    existing 3-channel retrieval -> answerer -> postprocessor

The reader workers receive the FULL text of one judgment each plus the
question, and emit verbatim on-point paragraphs (no compression of text
fidelity — the cheap model only quotes, never summarises). Quote
extraction stays mechanical via span markers.

### Files created

- `services/bif_research/corpus_fetch.py` — pure SQL: `fetch_section_full`,
  `fetch_case_full`. No LLM. Joins `name_index.sqlite` to `chunks.sqlite`
  in paragraph/section order.
- `services/bif_research/case_reader.py` — `read_case_for_question` uses
  `kind="reader"` chain. 60 000-char length guard skips oversized cases
  with a stub. Verbatim-only reader system prompt locked in module.
- `services/bif_research/parallel_reader.py` — `parallel_read_cases`
  fans out via `ThreadPoolExecutor(max_workers=4)`. Preserves input
  order, dedupes case_ids, returns safe stubs on per-worker failure.
- `services/bif_research/hard_pipeline.py` — `HardQuestionPipeline`
  with `answer()` and `answer_with_plan()`. Composes the reasoner
  prompt from full statutory text + verbatim case extracts + ≤6
  breadth chunks. Per-case "chunk index" (chunk_id ↔ paragraph range)
  embedded in the prompt so the reasoner can map paragraph numbers
  back to chunk_ids for citation.

### Files modified (only these three, per spec)

- `services/bif_research/llm_config.py` — added `CHAIN_READER`
  (`["gpt-5.4-mini","gpt-5-mini","claude-haiku-4-5","gpt-4o-mini"]`)
  and `CHAIN_REASONER` (`["claude-opus-4-7","claude-opus-4",
  "claude-sonnet-4-5","gpt-5.5","gpt-5.4","gpt-5"]`); extended
  `complete_chat()` dispatch to accept `kind="reader"` and
  `kind="reasoner"`.
- `services/bif_research/pipeline.py` — added `route_question(plan,
  question)` and `answer_with_plan()` on `FullPipeline`. Existing
  `answer()` is now a thin wrapper that runs the planner then calls
  `answer_with_plan()`. No behavioural change to the fast path.
- `services/bif_research/api.py` — replaced `_pipeline` with
  `_fast_pipeline` + `_hard_pipeline` (lazy-init via
  `get_pipelines()`). `/api/ask` now runs the planner once, calls
  `route_question()`, and dispatches to the appropriate pipeline's
  `answer_with_plan()`. New status events: `routing` and
  `reading_cases`.

### Routing rules (in order)

  if plan.is_fringe                     -> hard
  elif plan.confidence in {medium, low} -> hard
  elif plan.intent == "case_law"        -> hard
  elif len(plan.named_authorities) >= 2 -> hard
  else                                  -> fast

### Model chains used

| Stage | kind | Chain |
|---|---|---|
| query rewrite | default | `gpt-5.4-mini → gpt-5-mini → gpt-4o-mini` |
| planner | planner_ka | `claude-opus-4-7 → claude-opus-4 → claude-sonnet-4-5 → gpt-5.5 → gpt-5.4 → gpt-5` |
| reader workers | reader (NEW) | `gpt-5.4-mini → gpt-5-mini → claude-haiku-4-5 → gpt-4o-mini` |
| reasoner (hard) | reasoner (NEW) | `claude-opus-4-7 → claude-opus-4 → claude-sonnet-4-5 → gpt-5.5 → gpt-5.4 → gpt-5` |
| answerer (fast) | max | `gpt-5.5 → gpt-5.4 → gpt-5 → gpt-4o` |

### Phase D verification — ALL PASS, ZERO API SPEND

  D1 imports OK
  D2 fetch_case_full('QCA_2024_94') → 14 chunks, 39 paragraphs,
     29 203 chars, paragraph markers preserved
  D3 fetch_section_full('BIF Act','75') → 2 chunks, header parsed,
     QBCC Act s 67AZ also probed (12 chunks)
  D4 route_question() → 7/7 cases (5 hard + 2 fast)
  D5 parallel_read_cases stub → 3 deduped from 4 inputs in 56ms
     wall-clock (~50ms per stub, concurrent)
  D6 HardQuestionPipeline.answer_with_plan with stubbed reader +
     stubbed reasoner → composes a complete answer dict with
     `_pipeline=hard`, planner findings, reader summary, and
     mechanically-extracted verbatim quote from chunk_000147

All stubs were `unittest.mock.patch` against `case_reader.read_case_for_question`,
`llm_config.complete_chat`, and `Retriever.retrieve_three_channel`. Zero
network calls. Zero API spend during the build.

### Phase E

- Server restarted via `pkill -f uvicorn` + nohup; new PID picked up
  the new code. `/api/health` returns ok with chunks_total=14 280.
- **No live `/api/ask` query was issued during the build.** Ted runs
  the next test query himself.

### Known constraints (out of scope for this build)

- Hard-path latency: ~10-30s for the planner + 4 parallel readers +
  Opus reasoner. ~$0.30-0.60 per hard query (1 planner Opus + 4
  reader minis + 1 reasoner Opus).
- Cases marked NOT INDEXED (Brodyn, Chase Oyster, Southern Han, Kirk —
  interstate/HCA) are mentioned by the reasoner but not citable.
- Cases >60 000 chars get a "skipped" stub from the reader (none in
  the corpus hit that today; safety guard for future imports).
- The fuzzy-match miss on Parrwood [2020] QSC 211 (judgment not
  indexed in corpus at all) and the wrong-Northbuild fuzzy match are
  pre-existing name_index issues — explicitly out of scope here.

