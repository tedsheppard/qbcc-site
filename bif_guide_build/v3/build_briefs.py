"""
For each element in elements.py, build a self-contained brief file the
subagent can read. The brief includes:

- The element's identity (slug, title, breadcrumb)
- A copy of the format model page
- A copy of the global format rules
- The full text of every relevant source file (statute, annotated, regs, other),
  bundled into the brief itself so the agent doesn't have to navigate paths

Output: bif_guide_build/v3/briefs/<slug>.md (one file per element)
"""
from __future__ import annotations

from pathlib import Path

from elements import ALL_CATEGORIES, Element

V3 = Path(__file__).resolve().parent
SOURCE = V3 / "source"
PAGES = V3 / "pages"
BRIEFS = V3 / "briefs"
MODEL_PATH = PAGES / "model_pc-construction-contract.html"

GLOBAL_RULES = """\
# Global rules (same for every element-page)

1. **Match the format model exactly.** Read the model below and replicate
   the structure: breadcrumb h4, h2 title, opening paragraph, statute
   excerpt, prose body with topical h3 sub-headings, optional case
   extract block, "What this means on a project" bullet list,
   footnotes ordered list at the bottom.

2. **No headings labelling cases.** Do NOT use sub-headings like
   "Key authority", "Leading authority", "Key case" or "Important
   decision". Discuss the case naturally in prose under a topical
   sub-heading instead.

3. **Section reference style.** Always use `s 80` with a space, not
   `s80`. Apply this to every section reference in your prose, your
   footnotes, and your statute-excerpt summary text. The ONE exception
   is the verbatim statute body itself, where the gazetted heading
   number stays as printed (e.g. `<strong>61 Application of chapter</strong>`).

4. **Plain prose default.** Do NOT use `<div class="analysis">`. The
   only special blocks you may use are:
   - `<details class="statute-excerpt" open>` for verbatim statute
   - `<div class="case-excerpt">` for a single judicial extract (use
     this at most ONCE per page, only where the extract earns its
     place — i.e. it's the leading proposition or contains words the
     courts repeatedly quote)
   - `<ul>` / `<ol>` for lists
   - `<ol class="footnotes">` for the footnotes block at the bottom

5. **Verbatim source.** Statutory text, case names, citations, judicial
   block quotes and adjudication reference numbers must be reproduced
   word-for-word from the source. Curly→straight quote conversion is
   fine; everything else is not. Do NOT silently "correct" what looks
   like a typo or wrong cross-reference in the source — reproduce as
   written.

6. **No hallucination.** If a case, section or proposition is not in
   the source bundled below, do not include it.

7. **Audience and register.** Construction project manager / project
   director. Plain English, active voice, "you" framing where natural.
   Avoid "pursuant to", "notwithstanding", "inter alia", "it is
   submitted", "the authorities establish".

8. **Length.** Aim for 400–800 words of prose plus a tight footnote
   block. Don't pad. Don't truncate either if the source has substance.

9. **Case selection.** Pick at most TWO cases to discuss in-text. The
   rest go in the footnotes with citation, judge in parentheses, and a
   one-line gloss of what the case is for. The two in-text cases should
   be either:
   - the leading appellate authority (HCA, QCA, recent NSWCA), or
   - a case whose facts directly illustrate the point.
   Use AT MOST ONE `<div class="case-excerpt">` block on the page,
   reserved for a passage the courts treat as the canonical formulation.

10. **HTML entities.** Use `&mdash;`, `&ndash;`, `&hellip;`,
    `&ldquo;`/`&rdquo;`, `&amp;` rather than literal characters where
    appropriate.

11. **Output ONE `<section id="SLUG">…</section>` block.** No `<html>`,
    `<head>`, `<body>`, `<style>`, `<script>` wrappers. No markdown
    fences. No preamble or explanation outside the HTML.
"""


def read(path: Path) -> str:
    if not path.exists():
        return f"[MISSING SOURCE FILE: {path.relative_to(V3)}]"
    return path.read_text(encoding="utf-8")


def build_brief(elem: Element) -> str:
    parts: list[str] = []
    parts.append(f"# Element-page brief — {elem.slug}\n")
    parts.append(f"**Title:** {elem.title}\n")
    parts.append(f"**Breadcrumb:** {elem.breadcrumb}\n")
    parts.append(f"**Anchor id:** `{elem.slug}`\n")
    parts.append(f"**Output file:** `bif_guide_build/v3/pages/page_{elem.slug}.html`\n")

    if elem.statute_scope:
        parts.append("\n## Statute scope note\n")
        parts.append(elem.statute_scope + "\n")

    if elem.extra:
        parts.append("\n## Extra guidance for this element\n")
        parts.append(elem.extra + "\n")

    parts.append("\n---\n")
    parts.append(GLOBAL_RULES)
    parts.append("\n---\n")
    parts.append("\n## Format model — replicate this structure exactly\n\n")
    parts.append("```html\n")
    parts.append(read(MODEL_PATH))
    parts.append("\n```\n")
    parts.append("\n---\n")

    # Sources
    if elem.statute:
        parts.append("\n## Verbatim BIF Act statute\n")
        for p in elem.statute:
            full = SOURCE / p
            parts.append(f"\n### `{p}`\n```\n")
            parts.append(read(full))
            parts.append("\n```\n")

    if elem.annotated:
        parts.append("\n## Annotated commentary (rewrite for PM audience; preserve case names / citations / block quotes verbatim)\n")
        for p in elem.annotated:
            full = SOURCE / p
            parts.append(f"\n### `{p}`\n```\n")
            parts.append(read(full))
            parts.append("\n```\n")

    if elem.regs:
        parts.append("\n## BIF Regulation\n")
        for p in elem.regs:
            full = SOURCE / p
            parts.append(f"\n### `{p}`\n```\n")
            parts.append(read(full))
            parts.append("\n```\n")

    if elem.other:
        parts.append("\n## Other authority\n")
        for p in elem.other:
            full = SOURCE / p
            parts.append(f"\n### `{p}`\n```\n")
            parts.append(read(full))
            parts.append("\n```\n")

    return "".join(parts)


def main() -> None:
    BRIEFS.mkdir(parents=True, exist_ok=True)
    n = 0
    total_kb = 0
    for cat in ALL_CATEGORIES:
        for elem in cat.elements:
            brief = build_brief(elem)
            path = BRIEFS / f"{elem.slug}.md"
            path.write_text(brief, encoding="utf-8")
            n += 1
            total_kb += path.stat().st_size / 1024
    print(f"wrote {n} briefs to {BRIEFS.relative_to(V3.parent.parent)}/  (avg {total_kb/n:.1f} KB)")


if __name__ == "__main__":
    main()
