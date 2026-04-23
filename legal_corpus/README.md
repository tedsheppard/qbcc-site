# Legal corpus — source documents for /claim-check

This folder holds the raw legal source material that the /claim-check feature
indexes into a vector store for on-demand retrieval by the chatbot and rule
engine. Nothing here is served to the browser; documents are read at
index-build time only.

## Retrieval priority (most to least important)

1. **`bif_act/`** — primary source of truth, first port of call for any question
2. **`annotated_bif_act/`** — commentary + cases per section of the BIF Act
3. **`bif_regs/`** — subordinate legislation under the BIF Act
4. **`qbcc_act/`** — QBCC Act
5. **`qbcc_regs/`** — QBCC Regulation
6. **`aia_act/`** — Architects & Industry Act-equivalent reference material

When the chatbot calls `search_legislation(...)`, results are ranked so BIF
Act hits surface before annotated / regulatory / other-Act hits, unless an
explicit scope is passed.

## Current contents

| Folder | File |
|---|---|
| `bif_act/` | `act-2017-043 (2).pdf` |
| `annotated_bif_act/` | `v29 Annotated BIFSOPA (March 2026) (Clean)-1.docx` |
| `bif_regs/` | `sl-2018-0016.pdf` |
| `qbcc_act/` | `act-1991-098.pdf` |
| `qbcc_regs/` | `sl-2018-0138.pdf` |
| `aia_act/` | `act-1954-003.pdf` |

## Preferred file formats (in priority order)

1. `.md` — cleanest, fastest to index, no extraction step
2. `.txt` — fine
3. `.docx` — fine (extracted via python-docx)
4. `.pdf` — works, but extraction can be noisy on complex layouts

File names don't matter — the indexer reads everything in each folder.

## Rebuilding the index

After adding or replacing files, rebuild the index with:

```
python -m services.claim_check.build_index
```

(The index build script is added in stage 9 of the build.)

The built index lives in `services/claim_check/legal_index/` and is
feature-scoped — it does not interfere with the repo-level `chroma_db/`.
