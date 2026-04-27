"""
Assemble the v2 BIF Act guide.

Strategy:
- Read site/bif-act-guide.html (the live source).
- Replace the <aside class="knowledge-sidebar"> ... </aside> block with a new
  flat 15-entry sidebar that mirrors the new page anchors.
- Replace the contents of <main class="knowledge-content"> ... </main> with
  the original welcome-message div followed by the 15 generated page sections
  (in order).
- Leave everything else (head, page nav, page header, footer, CSS, JS) untouched.

Output:
    bif_guide_build/output/bif_act_guide_v2.html
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_GUIDE = REPO / "site" / "bif-act-guide.html"
PAGES_DIR = Path(__file__).resolve().parent / "guide_pages"
OUT_PATH = Path(__file__).resolve().parent / "output" / "bif_act_guide_v2.html"

# Order matters — this is the page order in the guide.
PAGE_MAP: list[tuple[str, str, str]] = [
    # (anchor_id, sidebar_label, html_filename)
    ("application",                "Application of the Act",            "page_01_application.html"),
    ("construction-contract",      "Construction contract",             "page_02_construction_contract.html"),
    ("construction-work",          "Construction work",                 "page_03_construction_work.html"),
    ("reference-date",             "Reference date",                    "page_04_reference_date.html"),
    ("payment-claim",              "Payment claim requirements",        "page_05_payment_claim.html"),
    ("payment-schedule",           "Payment schedule requirements",     "page_06_payment_schedule.html"),
    ("no-payment-schedule",        "Failing to give a payment schedule","page_07_no_payment_schedule.html"),
    ("adjudication-application",   "Adjudication application",          "page_08_adjudication_application.html"),
    ("adjudicator-appointment",    "Adjudicator appointment",           "page_09_adjudicator_appointment.html"),
    ("adjudication-response",      "Adjudication response",             "page_10_adjudication_response.html"),
    ("adjudication-procedures",    "Adjudication procedures",           "page_11_adjudication_procedures.html"),
    ("adjudicators-decision",      "Adjudicator&rsquo;s decision",      "page_12_adjudicators_decision.html"),
    ("enforcement",                "Enforcement",                       "page_13_enforcement.html"),
    ("suspension",                 "Suspension of work",                "page_14_suspension.html"),
    ("civil-proceedings",          "Civil proceedings",                 "page_15_civil_proceedings.html"),
]


def build_sidebar() -> str:
    lines = [
        "      <aside class=\"knowledge-sidebar\">",
        "        <h4>Guide</h4>",
        "        <ul class=\"sidebar-nav\">",
    ]
    for anchor, label, _ in PAGE_MAP:
        lines.append(f"          <li><a href=\"#{anchor}\">{label}</a></li>")
    lines += [
        "        </ul>",
        "      </aside>",
    ]
    return "\n".join(lines)


def build_main() -> str:
    parts = [
        "      <main class=\"knowledge-content\">",
        "",
        "        <div id=\"welcome-message\">",
        "          <h2>Welcome to the BIF Act Guide</h2>",
        "          <p>Please select a topic from the guide on the left to get started.</p>",
        "        </div>",
        "",
    ]
    for anchor, label, filename in PAGE_MAP:
        page_html = (PAGES_DIR / filename).read_text(encoding="utf-8").rstrip()
        # Indent each page so it sits cleanly inside <main>. Pages already
        # start with "<section id=...>" so just indent uniformly with 8 spaces.
        indented = "\n".join("        " + line if line.strip() else "" for line in page_html.splitlines())
        parts.append(f"        <!-- === {label} ({anchor}) === -->")
        parts.append(indented)
        parts.append("")
    parts.append("      </main>")
    return "\n".join(parts)


def main() -> None:
    if not SOURCE_GUIDE.exists():
        raise SystemExit(f"Source guide missing: {SOURCE_GUIDE}")
    for _, _, f in PAGE_MAP:
        if not (PAGES_DIR / f).exists():
            raise SystemExit(f"Missing generated page: {f}")

    html = SOURCE_GUIDE.read_text(encoding="utf-8")

    # The live source has stray <section> stubs orphaned after </html>.
    # Trim everything after the closing </html> tag.
    end_html_re = re.compile(r"(</html>\s*)", re.IGNORECASE)
    m = end_html_re.search(html)
    if not m:
        raise SystemExit("No </html> tag found in source")
    html = html[: m.end()].rstrip() + "\n"

    # Replace the sidebar block.
    sidebar_re = re.compile(
        r"      <aside class=\"knowledge-sidebar\">.*?</aside>\n",
        re.DOTALL,
    )
    new_sidebar = build_sidebar() + "\n"
    html, n_sidebar = sidebar_re.subn(new_sidebar, html, count=1)
    if n_sidebar != 1:
        raise SystemExit("Failed to locate exactly one <aside class=\"knowledge-sidebar\"> block")

    # Replace the main block.
    main_re = re.compile(
        r"      <main class=\"knowledge-content\">.*?</main>\n",
        re.DOTALL,
    )
    new_main = build_main() + "\n"
    html, n_main = main_re.subn(new_main, html, count=1)
    if n_main != 1:
        raise SystemExit("Failed to locate exactly one <main class=\"knowledge-content\"> block")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO)}  ({OUT_PATH.stat().st_size:,d} bytes)")

    # Sanity checks on the assembled file.
    for anchor, _, _ in PAGE_MAP:
        if f'id="{anchor}"' not in html:
            print(f"  WARN: section id missing in output: {anchor}")
        if f'href="#{anchor}"' not in html:
            print(f"  WARN: sidebar href missing in output: {anchor}")


if __name__ == "__main__":
    main()
