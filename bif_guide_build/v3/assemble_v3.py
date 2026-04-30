"""
Assemble the final v3 BIF Act Guide.

Strategy:
- Start from the clean pre-v2 archive (bif_guide_build/archive/bif_act_guide_original_2026-04-25.html).
- Replace the sidebar block with a new 11-category nested structure.
- Replace the contents of <main> with the welcome-message + all 87 element-pages,
  ordered by category then by element order in elements.py.
- Inject the small CSS additions for .footnotes (carried over from v2 preview).
- Write to bif_guide_build/v3/output/bif_act_guide_v3.html.
"""
from __future__ import annotations

import re
from pathlib import Path

from elements import ALL_CATEGORIES

V3 = Path(__file__).resolve().parent
REPO = V3.parent.parent
ARCHIVE = REPO / "bif_guide_build" / "archive" / "bif_act_guide_original_2026-04-25.html"
PAGES_DIR = V3 / "pages"
OUT_PATH = V3 / "output" / "bif_act_guide_v3.html"


def build_sidebar() -> str:
    lines = [
        "      <aside class=\"knowledge-sidebar\">",
        "        <h4>Guide</h4>",
        "        <ul class=\"sidebar-nav\">",
    ]
    for cat in ALL_CATEGORIES:
        first = cat.elements[0]
        lines.append(f"          <li>")
        lines.append(f"            <a href=\"#{first.slug}\">{cat.title}</a>")
        lines.append(f"            <ul>")
        for elem in cat.elements:
            lines.append(f"              <li><a href=\"#{elem.slug}\">{elem.title}</a></li>")
        lines.append(f"            </ul>")
        lines.append(f"          </li>")
    lines.append("        </ul>")
    lines.append("      </aside>")
    return "\n".join(lines)


def build_main() -> str:
    parts = [
        "      <main class=\"knowledge-content\">",
        "",
        "        <div id=\"welcome-message\">",
        "          <h2>Welcome to the BIF Act Guide</h2>",
        "          <p>Select a topic from the guide on the left.</p>",
        "        </div>",
        "",
    ]
    for cat in ALL_CATEGORIES:
        parts.append(f"        <!-- ============================================================== -->")
        parts.append(f"        <!-- CATEGORY: {cat.title} -->")
        parts.append(f"        <!-- ============================================================== -->")
        parts.append("")
        for elem in cat.elements:
            page_path = PAGES_DIR / f"page_{elem.slug}.html"
            if not page_path.exists():
                raise SystemExit(f"Missing page file: {page_path}")
            html = page_path.read_text(encoding="utf-8").rstrip()
            # The page files already have leading 8-space indent; keep as-is
            parts.append(f"        <!-- {elem.title} ({elem.slug}) -->")
            parts.append(html)
            parts.append("")
    parts.append("      </main>")
    return "\n".join(parts)


CSS_INJECTION = """\

    /* Footnotes (v3) */
    .knowledge-content sup a { color: #008a5c; text-decoration: none; font-weight: 600; padding: 0 2px; }
    .knowledge-content sup a:hover { text-decoration: underline; }
    .knowledge-content ol.footnotes {
      margin-top: 36px;
      padding-top: 18px;
      padding-left: 24px;
      border-top: 1px solid var(--border-light);
      font-size: 12.5px;
      color: var(--text-secondary);
      line-height: 1.55;
    }
    .knowledge-content ol.footnotes li { margin-bottom: 8px; font-size: 12.5px; }
    .knowledge-content ol.footnotes li:last-child { margin-bottom: 0; }
"""


def main() -> None:
    if not ARCHIVE.exists():
        raise SystemExit(f"Archive missing: {ARCHIVE}")

    html = ARCHIVE.read_text(encoding="utf-8")

    # Trim everything after </html>
    end_html = re.search(r"(</html>\s*)", html, re.IGNORECASE)
    if not end_html:
        raise SystemExit("No </html> tag in archive")
    html = html[: end_html.end()].rstrip() + "\n"

    # Inject footnote CSS just before the closing </style> in <head>
    if "ol.footnotes" not in html:
        html = re.sub(
            r"(\.authority-list \{[^}]+\})",
            r"\1" + CSS_INJECTION,
            html,
            count=1,
        )

    # Replace sidebar block
    sidebar_re = re.compile(
        r"      <aside class=\"knowledge-sidebar\">.*?</aside>\n",
        re.DOTALL,
    )
    new_sidebar = build_sidebar() + "\n"
    html, n = sidebar_re.subn(new_sidebar, html, count=1)
    if n != 1:
        raise SystemExit("Failed to locate sidebar block")

    # Replace main block
    main_re = re.compile(
        r"      <main class=\"knowledge-content\">.*?</main>\n",
        re.DOTALL,
    )
    new_main = build_main() + "\n"
    html, n = main_re.subn(new_main, html, count=1)
    if n != 1:
        raise SystemExit("Failed to locate main block")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO)} ({OUT_PATH.stat().st_size:,d} bytes)")

    # Sanity checks
    expected_anchors = sum(len(c.elements) for c in ALL_CATEGORIES)
    actual_section_ids = len(re.findall(r'<section id="[^"]+"', html))
    actual_sidebar_links = len(re.findall(r'<li><a href="#', html))
    print(f"  expected element-pages: {expected_anchors}")
    print(f"  <section id> tags in output: {actual_section_ids}")
    print(f"  <li><a href=\"#> sidebar links: {actual_sidebar_links}")


if __name__ == "__main__":
    main()
