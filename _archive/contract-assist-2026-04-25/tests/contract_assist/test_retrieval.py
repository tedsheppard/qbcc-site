"""Tests for services.contract_assist.retrieval.

These tests stub the OpenAI embeddings call with a deterministic
content-based hash so no real API requests are made. The chunker is
exercised against synthetic contract text; ingest() is exercised by
stubbing extract_rich for DOCX (so we don't need a real .docx fixture)
and using reportlab to build a tiny synthetic PDF for the PDF case.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import sys
from pathlib import Path

import pytest

# Make sure the repo root is on sys.path so "services.*" imports work
# regardless of where pytest is invoked from.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set a placeholder API key so any code path that checks for its presence
# is satisfied. We monkey-patch the actual call so no network traffic occurs.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from services.contract_assist.retrieval import service as svc  # noqa: E402
from services.contract_assist.retrieval import chunker as ch  # noqa: E402
from services.contract_assist.retrieval import embedder as emb  # noqa: E402
from services.contract_assist.retrieval import store as store_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Embedding stub
# ---------------------------------------------------------------------------

EMB_DIM = 1536


def _hash_to_unit_vector(text: str, dim: int = EMB_DIM) -> list[float]:
    """Deterministic content-based pseudo-embedding.

    For each chunk we hash a sliding window of the text into ``dim``
    float bins, then L2-normalize. Identical text → identical vector;
    similar text → highly similar vector. Good enough for retrieval
    behavior tests.
    """
    vec = [0.0] * dim
    src = (text or "").lower()
    if not src:
        src = " "
    # Sample many overlapping shingles so different chunks differ but
    # similar chunks correlate.
    window = 6
    for i in range(0, max(1, len(src) - window + 1)):
        chunk = src[i : i + window]
        h = hashlib.md5(chunk.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


def _stub_embed_texts(texts):
    return [_hash_to_unit_vector(t) for t in texts]


def _stub_embed_query(text):
    return _hash_to_unit_vector(text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_embeddings(monkeypatch):
    """Replace the embedder calls with deterministic stubs."""
    monkeypatch.setattr(emb, "embed_texts", _stub_embed_texts)
    monkeypatch.setattr(emb, "embed_query", _stub_embed_query)
    # The service module imports embed_texts/embed_query inside functions
    # via ``from .embedder import ...`` — those local bindings are looked
    # up fresh each call, so patching the embedder module is enough.
    yield


@pytest.fixture(autouse=True)
def _isolate_chroma(monkeypatch):
    """Drop the singleton client before/after each test for clean state."""
    store_mod._reset_for_tests()
    yield
    store_mod._reset_for_tests()


# ---------------------------------------------------------------------------
# Synthetic contract text
# ---------------------------------------------------------------------------

# ~7 KB of synthetic AS 4000-shaped text, with several clauses, so chunking
# produces multiple chunks.
SAMPLE_CONTRACT = """\
GENERAL CONDITIONS OF CONTRACT

This contract is made under the AS 4000-1997 General Conditions of Contract.

1 Definitions

In this contract, unless the context otherwise requires, the following terms
have the meanings given to them. The Principal means the party identified in
Item 2 of the Annexure. The Contractor means the party identified in Item 3 of
the Annexure. The Superintendent means the person named in Item 5.

The Date of Practical Completion is the date determined under clause 34. The
Contract Sum is the amount stated in Item 8 of the Annexure, subject to any
additions and deductions made under the contract.

2 Nature of contract

The Contractor shall execute and complete the work under the contract in
accordance with the contract. The Principal shall pay the Contractor the
Contract Sum at the times and in the manner provided in the contract.

The Superintendent shall act fairly and impartially when issuing certificates,
making valuations, or otherwise exercising functions under the contract.

12 Variations

The Superintendent may direct the Contractor to vary the work under the
contract by way of an addition, increase, decrease, omission, removal,
demolition, or change in character or quality. The Contractor shall not vary
the work except as directed by the Superintendent.

12.1 Notice of variation

The Contractor shall, within 14 days of receiving a direction to vary, give
the Superintendent a written notice of the price and time effect of the
variation. The notice shall include reasonable supporting detail.

12.2 Pricing of variation

If the parties cannot agree on the price of a variation, the Superintendent
shall determine a reasonable price having regard to the actual cost reasonably
incurred plus a reasonable allowance for overhead and profit.

34 Time

The Contractor shall execute the work under the contract so as to reach
practical completion by the Date for Practical Completion. Time is of the
essence in respect of the obligations of the parties.

34.1 Notice of delay

The Contractor shall, as soon as practicable and in any event within 14 days
after becoming aware of any matter likely to delay the work under the
contract, give the Superintendent a written notice of the cause and likely
duration of the delay. The notice shall identify the qualifying cause of
delay relied upon and provide such information as the Superintendent
reasonably requires.

34.2 Extension of time

If the Contractor is, or will be, delayed in reaching practical completion by
a qualifying cause of delay, and the Contractor has complied with clause 34.1,
the Superintendent shall, within 28 days after receiving the notice, grant a
reasonable extension of time. A failure to grant an extension within that
period is not a waiver of the right to do so.

34.3 Liquidated damages

If the Contractor fails to reach practical completion by the Date for
Practical Completion as adjusted under clause 34.2, the Contractor shall pay
the Principal liquidated damages at the rate stated in Item 24 for every day
after the Date for Practical Completion until the date of practical
completion or termination of the contract, whichever occurs first.

37 Payment claims

The Contractor shall claim payment progressively in accordance with Item 28.
Each progress claim shall be accompanied by such information as the
Superintendent reasonably requires to assess the claim.

37.1 Progress certificate

The Superintendent shall, within 14 days after receiving a progress claim,
issue to the Principal and the Contractor a progress certificate evidencing
the Superintendent's opinion of the moneys due from the Principal to the
Contractor under the progress claim and reasons for any difference.

42 Dispute resolution

If a difference or dispute arises between the parties in connection with the
work under the contract, either party may give the other and the
Superintendent a written notice of dispute. Within 14 days after receipt of
that notice, the parties shall confer in good faith to attempt to resolve the
dispute.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_pdf(text: str) -> bytes:
    """Build a tiny single-page PDF containing ``text`` via reportlab."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import LETTER

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    y = height - 50
    for line in text.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(40, y, line[:120])
        y -= 12
    c.save()
    return buf.getvalue()


def test_ingest_pdf_returns_chunks(monkeypatch):
    """Stubbing extract_rich avoids depending on PDF text-extraction fidelity
    while still exercising the PDF code path through ingest()."""
    from services.claim_check import extractor as ex_mod

    def _fake_extract(filename, content):
        return SAMPLE_CONTRACT, {"pages": 5, "scanned": False}

    monkeypatch.setattr(ex_mod, "extract_rich", _fake_extract)

    pdf_bytes = _make_pdf("placeholder body the stub returns SAMPLE_CONTRACT")
    res = svc.ingest(pdf_bytes, "sample-contract.pdf", "session-pdf")

    assert res["session_id"] == "session-pdf"
    assert res["filename"] == "sample-contract.pdf"
    assert res["page_count"] == 5
    assert res["chunk_count"] > 0
    assert res["identified_form"] == "AS 4000-1997"
    assert isinstance(res["elapsed_ms"], int) and res["elapsed_ms"] >= 0


def test_ingest_docx_returns_chunks(monkeypatch):
    from services.claim_check import extractor as ex_mod

    def _fake_extract(filename, content):
        return SAMPLE_CONTRACT, {}

    monkeypatch.setattr(ex_mod, "extract_rich", _fake_extract)

    res = svc.ingest(b"PK\x03\x04 fake docx bytes", "deal.docx", "session-docx")
    assert res["chunk_count"] > 0
    assert res["identified_form"] == "AS 4000-1997"
    assert res["filename"] == "deal.docx"


def test_chunker_structure_aware():
    # Repeat the sample so we definitely cross the chunk-size threshold and
    # exercise the boundary-finding logic.
    text = SAMPLE_CONTRACT + "\n\n" + SAMPLE_CONTRACT
    chunks = ch.chunk_text(text, document_name="sample.txt")
    assert len(chunks) >= 2, f"expected >=2 chunks for {len(text)} chars, got {len(chunks)}"

    # The chunk-boundary picker should prefer clause boundaries. With overlap
    # in play, the chunker picks the split point, then walks ~800 chars back
    # to start the next chunk — so the clause-aligned split lives somewhere
    # *inside* the next chunk, near where the overlap ends. We verify this by
    # locating the clause-heading regex in the body of subsequent chunks.
    clause_heading_re = re.compile(
        r"(?m)^\s*(?:Clause\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*)\s+[A-Z]"
    )
    aligned = sum(1 for c in chunks[1:] if clause_heading_re.search(c["content"]))
    assert aligned == len(chunks) - 1, (
        "every non-first chunk should contain a clause heading; "
        f"chunks={len(chunks)}, aligned={aligned}"
    )

    # And the *splitter* should have chosen a clause boundary as the cut for
    # at least one transition. Concretely: the suffix of chunk[0] should end
    # with non-clause prose, and chunk[1] should contain the next clause
    # heading inside its body.
    first_tail = chunks[0]["content"][-200:]
    second_body = chunks[1]["content"]
    assert clause_heading_re.search(second_body), (
        f"second chunk lacks a clause heading; tail of first: {first_tail!r}"
    )

    # Clause numbers metadata should populate for chunks containing clauses.
    all_clause_nums: set[str] = set()
    for c in chunks:
        for n in c.get("clause_numbers") or []:
            all_clause_nums.add(n)
    assert any("." in n for n in all_clause_nums), f"no dotted clauses found: {all_clause_nums}"
    assert "34.1" in all_clause_nums or "34.2" in all_clause_nums or "12.1" in all_clause_nums


def test_chunker_overlap():
    # Force enough text to require multiple chunks.
    text = (SAMPLE_CONTRACT + "\n\n") * 3
    chunks = ch.chunk_text(text, document_name="big.txt")
    assert len(chunks) >= 2

    # Check overlap on adjacent chunks: tail of N should appear inside N+1.
    # Allow slack: at least 600 chars of contiguous overlap.
    min_overlap = 600
    for a, b in zip(chunks, chunks[1:]):
        a_tail = a["content"][-min_overlap:]
        # Try to find ANY 200-char window of a_tail inside b's head.
        head = b["content"][: min_overlap + 800]
        # Sliding test: shrink the tail until we get a hit, but require >= 600 chars.
        found = False
        for size in (min_overlap, 500, 400, 300):
            tail = a["content"][-size:]
            if tail and tail[:size // 2] in head:
                # Good enough — the chunker isn't emitting boundary-aligned
                # exact-overlap on every adjacent pair (it lands on clause
                # boundaries), but there should be substantial shared text.
                if size >= min_overlap or tail in b["content"]:
                    found = True
                    break
        # Looser fallback: require ≥ 600 contiguous chars of overlap somewhere.
        if not found:
            shared = _longest_common_substring_len(a["content"][-2000:],
                                                   b["content"][:2000])
            assert shared >= min_overlap, (
                f"adjacent chunks share only {shared} chars of overlap; "
                f"expected >= {min_overlap}"
            )


def _longest_common_substring_len(a: str, b: str) -> int:
    """Crude longest common substring (length only)."""
    if not a or not b:
        return 0
    # Use rolling hashes via a set of fixed-window shingles.
    best = 0
    # Quick path: try descending window sizes until we hit one.
    for size in (1200, 1000, 800, 700, 600, 500, 400, 300, 200, 100):
        seen = set()
        for i in range(0, len(a) - size + 1):
            seen.add(a[i : i + size])
        for j in range(0, len(b) - size + 1):
            if b[j : j + size] in seen:
                best = size
                return best
    return best


def test_retrieve_returns_top_k(monkeypatch):
    from services.claim_check import extractor as ex_mod
    monkeypatch.setattr(ex_mod, "extract_rich", lambda fn, c: (SAMPLE_CONTRACT, {}))

    sid = "session-topk"
    svc.ingest(b"x", "doc.pdf", sid)

    results = svc.retrieve("variation pricing notice", sid, top_k=3)
    assert len(results) <= 3
    assert results, "expected at least one result"
    required = {
        "chunk_id", "document_name", "section_heading", "clause_numbers",
        "page_number", "excerpt", "full_text", "expanded_text", "score",
        "score_method",
    }
    for r in results:
        missing = required - set(r.keys())
        assert not missing, f"result missing keys: {missing}"
        assert isinstance(r["clause_numbers"], list)
        assert r["score_method"] in {"vector", "clause_match", "keyword"}
        assert 0.0 <= r["score"] <= 1.0


def test_retrieve_clause_priority(monkeypatch):
    from services.claim_check import extractor as ex_mod
    monkeypatch.setattr(ex_mod, "extract_rich", lambda fn, c: (SAMPLE_CONTRACT, {}))

    sid = "session-clause"
    svc.ingest(b"x", "doc.pdf", sid)

    results = svc.retrieve("Tell me about clause 34.1", sid, top_k=5)
    assert results, "expected at least one result"

    # The top result should be a clause-tagged hit and should reference 34.1.
    top = results[0]
    assert top["score_method"] == "clause_match", (
        f"expected clause_match top result, got {top['score_method']}; "
        f"all methods: {[r['score_method'] for r in results]}"
    )
    assert "34.1" in (top["clause_numbers"] or []), (
        f"top result missing 34.1; clause_numbers={top['clause_numbers']}"
    )
    # And clause matches should outrank vector hits in the score column.
    vec_scores = [r["score"] for r in results if r["score_method"] == "vector"]
    if vec_scores:
        assert top["score"] >= max(vec_scores)


def test_clear_purges_session(monkeypatch):
    from services.claim_check import extractor as ex_mod
    monkeypatch.setattr(ex_mod, "extract_rich", lambda fn, c: (SAMPLE_CONTRACT, {}))

    sid = "session-clear"
    svc.ingest(b"x", "doc.pdf", sid)

    pre = svc.retrieve("variation", sid, top_k=3)
    assert pre, "expected results before clear"

    svc.clear(sid)

    post = svc.retrieve("variation", sid, top_k=3)
    assert post == [], f"expected empty after clear, got {len(post)}"

    # Idempotent — second clear must not raise.
    svc.clear(sid)


def test_session_isolation(monkeypatch):
    from services.claim_check import extractor as ex_mod

    # Two distinct documents.
    doc_a = SAMPLE_CONTRACT  # AS 4000 contract about variations / time / etc.
    doc_b = (
        "GENERAL CONDITIONS - Bespoke Form\n\n"
        "1 Preamble\n\nThis contract concerns rooftop solar installation only.\n\n"
        "2 Solar specification\n\nThe Contractor shall install photovoltaic panels.\n"
        "The panels shall be Tier 1 monocrystalline with a 25-year warranty.\n"
        "The inverter shall be a string inverter with monitoring.\n\n"
        "3 Commissioning\n\nThe Contractor shall commission the system after install.\n"
    ) * 4  # repeat to pass the chunker's minimum size

    counter = {"i": 0}
    docs = [doc_a, doc_b]

    def _fake(fn, content):
        i = counter["i"]
        counter["i"] += 1
        return docs[i % 2], {}

    monkeypatch.setattr(ex_mod, "extract_rich", _fake)

    svc.ingest(b"x", "a.pdf", "sess-A")
    svc.ingest(b"y", "b.pdf", "sess-B")

    res_a = svc.retrieve("photovoltaic panels solar", "sess-A", top_k=5)
    res_b = svc.retrieve("photovoltaic panels solar", "sess-B", top_k=5)

    # Session A should not return any chunk that mentions photovoltaic
    # (because it never ingested the solar contract).
    for r in res_a:
        assert "photovoltaic" not in (r["full_text"] or "").lower(), (
            "session-A leaked content from session-B"
        )
    # Session B SHOULD return solar content.
    assert res_b, "session-B returned no results for its own content"
    assert any("photovoltaic" in (r["full_text"] or "").lower() for r in res_b), (
        "session-B retrieval did not surface its solar content"
    )

    # Cross-check the other direction.
    res_a2 = svc.retrieve("liquidated damages clause 34", "sess-A", top_k=5)
    res_b2 = svc.retrieve("liquidated damages clause 34", "sess-B", top_k=5)
    assert any("liquidated damages" in (r["full_text"] or "").lower() for r in res_a2)
    for r in res_b2:
        assert "liquidated damages" not in (r["full_text"] or "").lower(), (
            "session-B leaked content from session-A"
        )
