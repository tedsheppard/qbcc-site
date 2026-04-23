"""Document text extraction for /claim-check.

Supported:
  - PDF  -> pdfplumber (primary), PyPDF2 fallback
  - DOCX -> python-docx
  - Plain text (pasted)

Heavy imports are deferred so importing this module at server startup is
cheap and can't crash on missing optional deps.
"""

from __future__ import annotations

import io
import logging

log = logging.getLogger("claim_check.extractor")

MAX_TEXT_CHARS = 200_000  # guardrail against pathological PDFs


def extract_text(filename: str, content: bytes) -> str:
    if not content:
        raise ValueError("Empty document")

    name = (filename or "").lower()
    if name.endswith(".pdf"):
        text = _extract_pdf(content)
    elif name.endswith(".docx"):
        text = _extract_docx(content)
    elif name.endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {filename!r}. Use PDF, DOCX, or paste the text.")

    text = (text or "").strip()
    if not text:
        raise ValueError("No text could be extracted from this document.")

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n\n[...document truncated at {:,} chars...]".format(MAX_TEXT_CHARS)

    return text


def _extract_pdf(content: bytes) -> str:
    # pdfplumber first — better at tables, layout, and forms.
    try:
        import pdfplumber  # deferred
    except Exception as e:
        log.warning("pdfplumber import failed, falling back to PyPDF2: %s", e)
        return _extract_pdf_pypdf2(content)

    parts: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
    except Exception as e:
        log.warning("pdfplumber extraction failed (%s), falling back to PyPDF2", e)
        return _extract_pdf_pypdf2(content)

    text = "\n\n".join(parts).strip()
    if not text:
        # pdfplumber may return nothing on image-only PDFs. Try PyPDF2 as a sanity fallback.
        return _extract_pdf_pypdf2(content)
    return text


def _extract_pdf_pypdf2(content: bytes) -> str:
    import PyPDF2  # deferred; already in requirements.txt
    reader = PyPDF2.PdfReader(io.BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(parts).strip()


def _extract_docx(content: bytes) -> str:
    from docx import Document  # deferred; python-docx is in requirements.txt
    doc = Document(io.BytesIO(content))

    parts: list[str] = []
    for p in doc.paragraphs:
        if p.text:
            parts.append(p.text)

    # Tables often carry the claim line-items and scheduled amounts — must capture.
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts).strip()
