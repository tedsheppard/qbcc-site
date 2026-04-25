"""Builder for the BIF Act guide embedded index.

Parses ``site/bif-act-guide.html`` into chunks aligned to its top-level
``<section id=...>`` blocks (with ``<h2>`` headings), with size-based
subdivision when a single section is too long. Chunks are embedded with
OpenAI ``text-embedding-3-small`` and stored in a persistent ChromaDB
collection at ``services/contract_assist/bif_act_index/chroma``.

The builder is idempotent: if the persisted collection already has the
same number of chunks as this run produces, embedding is skipped.

Heavy imports (``chromadb``, ``openai``) are deferred into functions so
that simply importing this module is cheap.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

log = logging.getLogger("contract_assist.bif_act_index")

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
GUIDE_HTML_PATH = _REPO_ROOT / "site" / "bif-act-guide.html"

_INDEX_DIR = Path(__file__).resolve().parent
CHROMA_DIR = _INDEX_DIR / "chroma"
COLLECTION_NAME = "bif_act_guide_v1"

EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking targets (characters, not tokens — kept simple).
TARGET_CHUNK_CHARS = 1000
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 400
CHUNK_OVERLAP_CHARS = 150
# A "single section too long" threshold roughly equivalent to 1500 tokens
# at ~4 chars/token. Sections longer than this are split.
SOFT_SECTION_LIMIT_CHARS = 6000

# Frontend deep-link base. The guide page lives at /bif-act-guide and
# uses ``id="..."`` anchors on each <section>.
ANCHOR_URL_BASE = "/bif-act-guide"

# Idempotency / build-once locks.
_build_lock = threading.Lock()
_built = False


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


class _GuideSectionParser(HTMLParser):
    """Walks the guide HTML and yields one record per top-level <section>.

    For each ``<section id=...>`` it records:
      - ``anchor_id`` (the id attribute)
      - ``heading`` (text of the first <h2> inside)
      - ``text`` (concatenated paragraph/li/case-excerpt text inside the
        section, with HTML stripped and whitespace collapsed)

    Only top-level sections are recorded (nested <section> elements are
    rare in the guide but tracked via depth). The ``<script>``,
    ``<style>``, and ``<details class="statute-excerpt">`` blocks are
    skipped — statute excerpts are the verbatim Act text and would
    dominate embeddings; we want the commentary.
    """

    # Tags whose textual content we keep, when inside a section.
    _TEXT_BEARING_TAGS = {
        "p", "li", "h2", "h3", "h4", "strong", "em", "span", "ul", "ol",
        "div", "blockquote", "summary",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[dict[str, Any]] = []
        self._section_stack: list[dict[str, Any]] = []
        # ``_in_statute_excerpt`` is set to the depth (>=1) of nesting
        # inside a <details class="statute-excerpt"> block. We skip text
        # inside those blocks because the verbatim statute text would
        # dominate retrieval; we want the surrounding commentary.
        self._in_statute_excerpt = 0
        # ``_collect_h2_for`` is the section dict we're currently
        # accumulating heading text for (only the first <h2> per
        # section).
        self._collect_h2_for: dict[str, Any] | None = None
        # ``_in_breadcrumb`` is set while inside an <h4
        # class="breadcrumb-heading"> so we drop it from text body
        # (otherwise every section would start with "Requirements of a
        # payment claim").
        self._in_breadcrumb = 0

    # -- helpers -------------------------------------------------------

    def _current_section(self) -> dict[str, Any] | None:
        return self._section_stack[-1] if self._section_stack else None

    def _should_skip_text(self) -> bool:
        if self._current_section() is None:
            return True
        if self._in_statute_excerpt > 0:
            return True
        if self._in_breadcrumb > 0:
            return True
        return False

    def _add_text(self, text: str) -> None:
        sec = self._current_section()
        if sec is None:
            return
        if self._collect_h2_for is sec:
            sec["_heading_buf"].append(text)
        sec["_text_buf"].append(text)

    # -- HTMLParser hooks ---------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: (v or "") for k, v in attrs}

        # HTMLParser handles <script> and <style> bodies as CDATA, so we
        # don't need to track them here.

        if self._in_statute_excerpt > 0:
            # Track nested <details>/inner elements so we know when the
            # excerpt block actually closes.
            if tag == "details":
                self._in_statute_excerpt += 1
            return

        if tag == "details":
            cls = attr_dict.get("class", "")
            if "statute-excerpt" in cls:
                self._in_statute_excerpt = 1
                return

        if tag == "section":
            sec = {
                "anchor_id": attr_dict.get("id", "").strip(),
                "heading": "",
                "_heading_buf": [],
                "_text_buf": [],
                "_h2_seen": False,
                "depth": len(self._section_stack),
            }
            self._section_stack.append(sec)
            return

        # Drop the breadcrumb heading that prefixes every guide section.
        if tag == "h4" and "breadcrumb-heading" in attr_dict.get("class", ""):
            self._in_breadcrumb += 1
            return

        if tag == "h2" and self._current_section() is not None:
            sec = self._current_section()
            assert sec is not None
            if not sec["_h2_seen"]:
                self._collect_h2_for = sec
            return

        # All other tags inside a section: just keep collecting text.

    def handle_endtag(self, tag: str) -> None:
        if self._in_statute_excerpt > 0:
            if tag == "details":
                self._in_statute_excerpt -= 1
            return

        if tag == "h4" and self._in_breadcrumb > 0:
            self._in_breadcrumb -= 1
            return

        if tag == "h2" and self._collect_h2_for is not None:
            sec = self._collect_h2_for
            sec["heading"] = " ".join("".join(sec["_heading_buf"]).split()).strip()
            sec["_h2_seen"] = True
            self._collect_h2_for = None
            # Add a blank line after the heading in the text buffer so
            # it remains visually separate from body.
            sec["_text_buf"].append("\n")
            return

        if tag == "section" and self._section_stack:
            sec = self._section_stack.pop()
            text = "".join(sec["_text_buf"])
            text = _normalise_whitespace(text)
            if not text:
                return
            self.sections.append({
                "anchor_id": sec["anchor_id"],
                "heading": sec["heading"],
                "text": text,
                "depth": sec["depth"],
            })
            return

        # Insert spacing after block-level tags so words don't run
        # together when we drop the markup.
        if tag in ("p", "li", "h3", "h4", "div", "blockquote", "summary", "ul", "ol"):
            self._add_text("\n")

    def handle_data(self, data: str) -> None:
        if self._should_skip_text():
            return
        self._add_text(data)


def _normalise_whitespace(text: str) -> str:
    # Collapse internal runs of whitespace, but preserve paragraph breaks.
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = []
    for p in paragraphs:
        p = re.sub(r"[ \t\r\f\v]+", " ", p)
        p = re.sub(r" *\n *", " ", p).strip()
        if p:
            cleaned.append(p)
    return "\n\n".join(cleaned)


# ---------------------------------------------------------------------------
# Section number inference and chunking
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"\bsection\s+(\d+)(?:\([^)]+\))?", re.IGNORECASE)
# Matches "s 68", "s.68", "s 75(2)(b)" etc. — kept conservative.
_SHORT_SECTION_RE = re.compile(r"\bs\s*\.?\s*(\d{1,3})(?:\([^)]+\))?\b")
# "Section 68 of the BIF Act" / "section 68 BIF" — strongest signal.
_BIF_SECTION_RE = re.compile(
    r"\bsection\s+(\d+)(?:\([^)]+\))?\s+(?:of\s+the\s+)?BIF\b",
    re.IGNORECASE,
)

# BIF Act Chapter 3 (Security of Payment) sections start at s 60. Anything
# below that in this guide is almost always a citation to an interstate
# equivalent (e.g. s 8 / s 13 of the NSW or VIC Acts), so we weight those
# down to avoid spurious "s 8" labels on commentary that's actually about
# s 67–s 76 of the BIF Act.
_BIF_MIN_SECTION = 60
_BIF_MAX_SECTION = 200


def _infer_section_ref(heading: str, text: str) -> str:
    """Best-effort: identify the dominant BIF Act section discussed.

    Weighting:
      * "section N (of the) BIF Act" mentions count 4x.
      * "section N" anywhere in the heading or first 500 chars counts 3x.
      * Bare "s N" in the body counts 0.5x.
    Section numbers outside the BIF Chapter 3 range are deprioritised
    because they're almost always citations to interstate equivalents.
    """
    head = heading or ""
    body = text or ""
    early = head + "\n" + body[:500]
    blob = head + "\n" + body[:2500]

    scores: dict[int, float] = {}

    def add(n: int, w: float) -> None:
        scores[n] = scores.get(n, 0.0) + w

    for m in _BIF_SECTION_RE.finditer(blob):
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        add(n, 4.0)
    for m in _SECTION_RE.finditer(early):
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        add(n, 3.0)
    for m in _SECTION_RE.finditer(blob):
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        add(n, 1.0)
    for m in _SHORT_SECTION_RE.finditer(blob):
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        add(n, 0.5)

    if not scores:
        return ""

    # Penalise sections outside the BIF Act Chapter 3 range (interstate
    # equivalents quoted in case law).
    adjusted = {
        n: (s if _BIF_MIN_SECTION <= n <= _BIF_MAX_SECTION else s * 0.2)
        for n, s in scores.items()
    }
    top = sorted(adjusted.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return f"s {top}"


def _split_long_text(text: str) -> list[str]:
    """Split ``text`` into ~800–1200 char chunks with ~150-char overlap.

    Tries to break on paragraph boundaries; falls back to sentence and
    finally raw character cuts. Always preserves overlap between
    consecutive chunks for retrieval continuity.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    # Greedy paragraph packing.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0

    for p in paragraphs:
        # If a single paragraph exceeds MAX_CHUNK_CHARS, hard-split it.
        if len(p) > MAX_CHUNK_CHARS:
            flush()
            chunks.extend(_hard_split(p, MAX_CHUNK_CHARS))
            continue

        prospective = buf_len + (2 if buf else 0) + len(p)
        if prospective > TARGET_CHUNK_CHARS and buf_len >= MIN_CHUNK_CHARS:
            flush()
        buf.append(p)
        buf_len += len(p) + (2 if buf_len > 0 else 0)

    flush()

    # Add overlap.
    if CHUNK_OVERLAP_CHARS > 0 and len(chunks) > 1:
        with_overlap: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-CHUNK_OVERLAP_CHARS:]
            with_overlap.append((prev_tail + "\n\n" + chunks[i]).strip())
        chunks = with_overlap

    return chunks


def _hard_split(s: str, size: int) -> list[str]:
    out = []
    i = 0
    while i < len(s):
        out.append(s[i : i + size])
        i += size - CHUNK_OVERLAP_CHARS if size > CHUNK_OVERLAP_CHARS else size
    return out


# ---------------------------------------------------------------------------
# Public: parse + chunk
# ---------------------------------------------------------------------------


def parse_guide(html_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Parse the BIF Act guide HTML into chunk records.

    Returns a list of dicts with keys:
      ``id``, ``section_ref``, ``heading``, ``anchor_id``, ``anchor_url``,
      ``text``.

    Each record is a single chunk ready for embedding.
    """
    p = Path(html_path) if html_path else GUIDE_HTML_PATH
    if not p.exists():
        raise FileNotFoundError(f"BIF Act guide HTML not found at {p}")
    html = p.read_text(encoding="utf-8")

    parser = _GuideSectionParser()
    parser.feed(html)
    parser.close()

    raw_sections = parser.sections

    # Deduplicate near-identical sections that appear because the guide
    # contains a second copy of some <section id="..."> blocks after the
    # closing </html>. We keep the first occurrence per (anchor_id,
    # heading) pair and drop trivially-empty ones.
    seen_keys: set[tuple[str, str]] = set()
    unique_sections: list[dict[str, Any]] = []
    for sec in raw_sections:
        if not sec["text"] or len(sec["text"]) < 80:
            continue
        # Only keep top-level sections (depth == 0). Nested <section>
        # tags inside a parent would otherwise duplicate text upward.
        if sec["depth"] != 0:
            continue
        key = (sec["anchor_id"], sec["heading"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_sections.append(sec)

    chunks: list[dict[str, Any]] = []
    for sec in unique_sections:
        section_ref = _infer_section_ref(sec["heading"], sec["text"])
        anchor_id = sec["anchor_id"] or ""
        anchor_url = (
            f"{ANCHOR_URL_BASE}#{anchor_id}" if anchor_id else ANCHOR_URL_BASE
        )

        if len(sec["text"]) <= SOFT_SECTION_LIMIT_CHARS and len(sec["text"]) <= MAX_CHUNK_CHARS:
            pieces = [sec["text"]]
        else:
            pieces = _split_long_text(sec["text"])

        for i, piece in enumerate(pieces):
            chunk_id = f"{anchor_id or 'sec'}::{i}"
            chunks.append({
                "id": chunk_id,
                "section_ref": section_ref,
                "heading": sec["heading"],
                "anchor_id": anchor_id,
                "anchor_url": anchor_url,
                "text": piece,
            })

    return chunks


# ---------------------------------------------------------------------------
# Public: build the index
# ---------------------------------------------------------------------------


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via OpenAI ``text-embedding-3-small``.

    Batched in groups of 100 to stay well under request limits.
    """
    from openai import OpenAI  # deferred

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured; cannot build BIF Act index."
        )
    client = OpenAI(api_key=api_key)

    out: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


def _get_collection(create_if_missing: bool = True):
    """Return the chroma Collection. Creates the directory + collection if needed."""
    import chromadb  # deferred

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if create_if_missing:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=COLLECTION_NAME)


def build_index(*, force: bool = False) -> None:
    """Build (or refresh) the persistent BIF Act index.

    Idempotent: if the existing collection already has the same number
    of chunks as the parser produces this run, the build is skipped.

    Set ``force=True`` to wipe and rebuild regardless. Used by tests.
    """
    global _built
    with _build_lock:
        if _built and not force:
            return

        chunks = parse_guide()
        if not chunks:
            log.warning("BIF Act guide produced zero chunks; index not built.")
            _built = True
            return

        import chromadb  # deferred

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        if force:
            try:
                client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        existing = 0
        try:
            existing = collection.count()
        except Exception:
            existing = 0

        if existing == len(chunks) and not force:
            log.info(
                "BIF Act index already up-to-date (%d chunks); skipping embed.",
                existing,
            )
            _built = True
            return

        # Stale or empty -> wipe and rebuild.
        if existing > 0:
            try:
                client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                pass
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        embeddings = _embed_texts([c["text"] for c in chunks])

        ids = [c["id"] for c in chunks]
        metadatas = [
            {
                "section_ref": c["section_ref"],
                "heading": c["heading"],
                "anchor_id": c["anchor_id"],
                "anchor_url": c["anchor_url"],
            }
            for c in chunks
        ]
        documents = [c["text"] for c in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        log.info("Built BIF Act index with %d chunks.", len(chunks))
        _built = True


def is_built() -> bool:
    """Return True if the persisted collection exists and has rows."""
    try:
        coll = _get_collection(create_if_missing=False)
        return coll.count() > 0
    except Exception:
        return False
