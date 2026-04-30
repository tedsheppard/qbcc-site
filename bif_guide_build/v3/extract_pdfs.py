"""
Extract verbatim statute/regulation text from the BIF Act and BIF Regulation
PDFs, as well as from QBCC Act and AIA Act PDFs (specific sections only).

Outputs flat per-section text files into bif_guide_build/v3/source/statute/
and bif_guide_build/v3/source/regs/, plus targeted excerpts in
bif_guide_build/v3/source/other/.

For the BIF Act and BIF Regulation, sections are detected by a regex that
matches lines starting with "<number><optional-letter> <title>" on the
left margin. We strip running headers/footers ("Page N", "Authorised by...",
"Current as at...") so they don't pollute the section bodies.
"""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

REPO = Path(__file__).resolve().parent.parent.parent
V3 = Path(__file__).resolve().parent
OUT_STATUTE = V3 / "source" / "statute"
OUT_REGS = V3 / "source" / "regs"
OUT_OTHER = V3 / "source" / "other"

BIF_ACT_PDF = REPO / "legal_corpus" / "bif_act" / "act-2017-043 (2).pdf"
BIF_REGS_PDF = REPO / "legal_corpus" / "bif_regs" / "sl-2018-0016.pdf"
QBCC_ACT_PDF = REPO / "legal_corpus" / "qbcc_act" / "act-1991-098.pdf"
AIA_ACT_PDF = REPO / "legal_corpus" / "aia_act" / "act-1954-003.pdf"

# Pattern for a section header line in the gazetted text:
#   "8 Definitions for chapter"
#   "24A Contracting party to report related entities"
#   "75 Making payment claim"
SECTION_HEAD_RE = re.compile(r"^(\d{1,3}[A-Z]{0,2})\s+([A-Z][^\n]{2,200})$", re.MULTILINE)

# Patterns for header/footer noise we strip:
NOISE_RES = [
    re.compile(r"^Building Industry Fairness \(Security of Payment\) Act 2017\s*$", re.MULTILINE),
    re.compile(r"^Building Industry Fairness \(Security of Payment\) Regulation 2018\s*$", re.MULTILINE),
    re.compile(r"^Queensland Building and Construction Commission Act 1991\s*$", re.MULTILINE),
    re.compile(r"^Acts Interpretation Act 1954\s*$", re.MULTILINE),
    re.compile(r"^Chapter\s+\d+\s+[A-Z][^\n]{2,80}\s*$", re.MULTILINE),
    re.compile(r"^Part\s+\d+[A-Z]?\s+[A-Z][^\n]{2,80}\s*$", re.MULTILINE),
    re.compile(r"^Division\s+\d+[A-Z]?\s+[A-Z][^\n]{2,80}\s*$", re.MULTILINE),
    re.compile(r"^Subdivision\s+\d+[A-Z]?\s+[A-Z][^\n]{2,80}\s*$", re.MULTILINE),
    re.compile(r"^\[s\s+\d+[A-Z]?\]\s*$", re.MULTILINE),
    re.compile(r"^Page\s+\d+\s*$", re.MULTILINE),
    re.compile(r"^Authorised by the Parliamentary Counsel\s*$", re.MULTILINE),
    re.compile(r"^Current as at[^\n]{0,80}$", re.MULTILINE),
    re.compile(r"^=== PAGE \d+ ===\s*$", re.MULTILINE),
]


def read_pdf_text(path: Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            parts.append(t)
    return "\n".join(parts)


def strip_noise(text: str) -> str:
    for r in NOISE_RES:
        text = r.sub("", text)
    # collapse triple+ blank lines down to double
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def find_chapters(text: str) -> dict[str, tuple[int, int]]:
    """Return {chapter_num: (start_offset, end_offset)} for the BIF Act.

    The PDF repeats the chapter heading on every page as a running header,
    so we take the LAST occurrence of each chapter heading as the actual
    boundary (everything before that is page-header noise from earlier
    chapters or the table of contents).
    """
    pattern = re.compile(r"^Chapter\s+(\d+)\s+([A-Z][^\n]{2,80})$", re.MULTILINE)
    # collect first occurrence after the table-of-contents area;
    # ToC is approximately the first 5% of the text.
    skip_until = int(len(text) * 0.04)
    seen: dict[str, int] = {}
    for m in pattern.finditer(text):
        if m.start() < skip_until:
            continue
        if m.group(1) not in seen:
            seen[m.group(1)] = m.start()
    starts = sorted(seen.items(), key=lambda x: x[1])
    boundaries: dict[str, tuple[int, int]] = {}
    for i, (n, s) in enumerate(starts):
        e = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        boundaries[n] = (s, e)
    return boundaries


_TOC_TAIL_RE = re.compile(r"(?:\.{4,}\s*\d{1,4}|\s{2,}\d{1,4})\s*$")
# A real section's body must contain a numbered subsection marker "(1)"
# within the first 400 chars. This is a near-universal feature of Qld
# statute drafting and rules out:
# - ToC entries (which have other section headings, not body text, after them)
# - in-body notes / cross-references that happen to start with a digit
_BODY_HAS_SUBSECTION_RE = re.compile(r"^\s*\(1\)\s", re.MULTILINE)


def _looks_like_real_heading(title: str, following_text: str, sid: str) -> bool:
    """Return True if this section header looks like a real statutory heading
    rather than a table-of-contents entry or in-body false positive."""
    if _TOC_TAIL_RE.search(title):
        return False
    # Real BIF Act sections are numbered from 1; section 1 is the short title.
    # But our extraction skips Chapter 1 prelude on most chapters; in practice
    # section IDs below 7 are almost always false positives from in-body notes.
    m = re.match(r"^(\d+)([A-Z]{0,2})$", sid)
    if not m:
        return False
    head = following_text.lstrip()[:600]
    if not head:
        return False
    # Real sections almost always have "(1)" or a "Maximum penalty" pointer
    # within the first 400 chars of body. Single-subsection sections that
    # don't start with "(1)" are rare but exist (e.g. pure definitions).
    if _BODY_HAS_SUBSECTION_RE.search(head[:400]):
        return True
    # Allow short single-paragraph sections whose body is a definitions
    # signature: starts with "In this Act—" or "The purpose of..." etc.
    starters = [
        "In this Act",
        "In this section",
        "In this chapter",
        "In this part",
        "In this division",
        "The purpose of this",
        "The object of this",
        "The main purpose of this",
        "This Act",
        "This chapter",
        "This part",
        "This section",
        "This division",
        "This Act binds",
    ]
    for s in starters:
        if head.startswith(s):
            return True
    return False


def split_sections(text: str, *, strict: bool = True) -> list[tuple[str, str, str]]:
    """Split a body of statute text into (section_id, title, body) triples.

    When `strict` is True (default), ToC entries and in-body false positives
    are filtered out using `_looks_like_real_heading`. When `strict` is False
    (used for already-ToC-anchored text), every match is kept — useful when
    you've already trimmed the ToC and want to capture sections whose title
    wraps over multiple lines (which would otherwise fail the body filter).
    """
    matches = list(SECTION_HEAD_RE.finditer(text))
    keep: list[tuple[int, str, str]] = []
    for i, m in enumerate(matches):
        sid = m.group(1)
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        following = text[body_start:body_end]
        if strict and not _looks_like_real_heading(title, following, sid):
            continue
        # Drop obvious ToC entries even in non-strict mode
        if _TOC_TAIL_RE.search(title):
            continue
        keep.append((m.start(), sid, title))

    out: list[tuple[str, str, str]] = []
    for j, (start_off, sid, title) in enumerate(keep):
        body_end_off = keep[j + 1][0] if j + 1 < len(keep) else len(text)
        # Find where this header line ends so the body starts after it.
        line_end = text.find("\n", start_off)
        if line_end == -1:
            line_end = start_off + len(title) + len(sid) + 1
        body = text[line_end + 1 : body_end_off].strip()
        out.append((sid, title, body))
    return out


def write_sections(sections: list[tuple[str, str, str]], chapter_label: str, out_dir: Path, prefix: str = "section") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for sid, title, body in sections:
        m = re.match(r"^(\d+)([A-Z]{0,2})$", sid)
        if m:
            num, letter = m.group(1), m.group(2)
            slug = f"{int(num):03d}{letter}"
        else:
            slug = sid
        out = out_dir / f"{prefix}_{slug}.txt"
        header = [
            f"# Source: {chapter_label}",
            f"# Section {sid} — {title}",
            "",
        ]
        out.write_text("\n".join(header) + body.strip() + "\n", encoding="utf-8")
        n += 1
    return n


def extract_bif_act() -> None:
    raw = read_pdf_text(BIF_ACT_PDF)

    # Find where the actual Act body starts — the FIRST occurrence of "1 Short title"
    # that is NOT a ToC entry (i.e. doesn't end with leader dots / page number).
    body_start = None
    for m in re.finditer(r"^1\s+Short title\s*$", raw, re.MULTILINE):
        body_start = m.start()
        break
    if body_start is None:
        raise SystemExit("Could not find s 1 Short title in BIF Act PDF")
    body_text = raw[body_start:]

    # Strip noise from the body
    body_text = strip_noise(body_text)

    # Split into sections (ToC is now gone — accept all matches, including
    # those whose title wraps over multiple lines)
    sections = split_sections(body_text, strict=False)

    # Group sections into chapters using known section number ranges in the
    # current Act (April 2025): Ch 1 = 1-6, Ch 2 = 7-72ish, Ch 3 = 73-102,
    # Ch 4 = 103+. Use the next chapter's first section to set the upper bound.
    # Inspecting the PDF: s 73 starts Chapter 3 (Progress payments) and s 103
    # starts Chapter 4 (Subcontractors' charges).
    def chapter_for(sid: str) -> str:
        m = re.match(r"^(\d+)([A-Z]{0,2})$", sid)
        if not m:
            return "?"
        n = int(m.group(1))
        # Verified against the current Act body: Ch 1 = ss 1-6,
        # Ch 2 (Statutory trusts) = ss 7-60 plus letter suffixes,
        # Ch 3 (Progress payments) = ss 61-102 plus letter suffixes,
        # Ch 4 (Subcontractor's charges) = ss 103-148.
        if n <= 6:
            return "1"
        if n <= 60:
            return "2"
        if n <= 102:
            return "3"
        if n <= 148:
            return "4"
        return "?"

    by_chapter: dict[str, list[tuple[str, str, str]]] = {"1": [], "2": [], "3": [], "4": []}
    for sec in sections:
        ch = chapter_for(sec[0])
        if ch in by_chapter:
            by_chapter[ch].append(sec)

    total = 0
    for ch, secs in by_chapter.items():
        chapter_label = f"BIF Act 2017 — Chapter {ch}"
        n = write_sections(secs, chapter_label, OUT_STATUTE / f"chapter_{ch}", prefix="section")
        print(f"  Chapter {ch}: {n} sections written")
        total += n
    print(f"BIF Act total: {total} per-section files in {OUT_STATUTE.relative_to(REPO)}/")


def extract_bif_regs() -> None:
    raw = read_pdf_text(BIF_REGS_PDF)
    cleaned = strip_noise(raw)
    sections = split_sections(cleaned)
    n = write_sections(sections, "BIF Regulation 2018", OUT_REGS, prefix="reg")
    print(f"BIF Regs: {n} per-section files in {OUT_REGS.relative_to(REPO)}/")


def extract_qbcc_s42() -> None:
    """Save QBCC Act s 42 (Person carrying out building work without licence)
    plus the surrounding licensing offences (s 42A, s 42B etc) as one file."""
    raw = read_pdf_text(QBCC_ACT_PDF)
    cleaned = strip_noise(raw)
    # locate s 42 ... up to the next round number that's >= 50 to keep neighbouring sections.
    m = re.search(r"^42\s+[A-Z][^\n]{2,200}$", cleaned, re.MULTILINE)
    if not m:
        print("  WARN: could not find QBCC Act s 42 header")
        return
    start = m.start()
    # Stop at the first section whose number >= 50.
    end_match = re.search(r"^(5[0-9]|6\d|7\d|8\d|9\d|1\d{2})\s+[A-Z]", cleaned[start:], re.MULTILINE)
    end = start + end_match.start() if end_match else min(len(cleaned), start + 30000)
    body = cleaned[start:end].strip()
    OUT_OTHER.mkdir(parents=True, exist_ok=True)
    (OUT_OTHER / "qbcc_act_s42_unlicensed_work.txt").write_text(
        "# Source: Queensland Building and Construction Commission Act 1991\n"
        "# Sections 42 onwards (unlicensed building work — illegality affects PC validity under BIF Act)\n\n"
        + body + "\n",
        encoding="utf-8",
    )
    print(f"  wrote QBCC Act s 42 excerpt ({len(body):,d} chars)")


def extract_aia_s39() -> None:
    """Save AIA s 39 (Service of documents) and adjacent service-related sections."""
    raw = read_pdf_text(AIA_ACT_PDF)
    cleaned = strip_noise(raw)
    # Find the s 39 header that ISN'T a table-of-contents entry. ToC entries
    # are followed by ". . . . page-number"; the real heading is followed by
    # actual statutory text.
    real_match = None
    for m in re.finditer(r"^39\s+Service of documents$", cleaned, re.MULTILINE):
        # check what follows immediately — if it's text (not dots), this is the real one
        tail = cleaned[m.end():m.end() + 200]
        if "." * 5 not in tail:
            real_match = m
            break
    if not real_match:
        print("  WARN: could not find real AIA s 39 header (only ToC matches)")
        return
    start = real_match.start()
    # Stop at s 40 or later (real, not ToC)
    end_match = None
    for em in re.finditer(r"^(4[0-9]|50)\s+[A-Z][a-z]", cleaned[start + 100:], re.MULTILINE):
        end_match = em
        break
    end = start + 100 + end_match.start() if end_match else min(len(cleaned), start + 8000)
    body = cleaned[start:end].strip()
    OUT_OTHER.mkdir(parents=True, exist_ok=True)
    (OUT_OTHER / "aia_act_s39_service.txt").write_text(
        "# Source: Acts Interpretation Act 1954 (Qld)\n"
        "# Section 39 — Service of documents\n\n"
        + body + "\n",
        encoding="utf-8",
    )
    print(f"  wrote AIA s 39 excerpt ({len(body):,d} chars)")


def main() -> None:
    print("Extracting BIF Act…")
    extract_bif_act()
    print("Extracting BIF Regulation…")
    extract_bif_regs()
    print("Extracting QBCC Act s 42…")
    extract_qbcc_s42()
    print("Extracting AIA s 39…")
    extract_aia_s39()


if __name__ == "__main__":
    main()
