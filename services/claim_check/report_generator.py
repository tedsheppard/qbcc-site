"""PDF analysis report generator for /claim-check.

Uses reportlab (already in requirements.txt). Output is streamed to the
browser as bytes; no server-side persistence.

Layout:
  Page 1 — header with Sopal logo, metadata (mode, filename, date),
           document summary, prominent disclaimer box
  Page 2+ — each check: status badge, title, section, status summary,
           explanation, labelled quote, case citations
  Appendix — user's answers to interactive inputs
  Every page footer — short disclaimer + page number
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("claim_check.report_generator")

LOGO_PATH = Path(__file__).resolve().parents[2] / "site" / "assets" / "sopal_logo_v1.png"

STATUS_COLORS = {
    "pass":     ("#166534", "#f0fdf4", "No issues detected"),
    "warning":  ("#92400e", "#fffbeb", "Potential issue — review"),
    "fail":     ("#b91c1c", "#fef2f2", "Likely non-compliant"),
    "input":    ("#374151", "#f3f4f6", "Additional information required"),
    "pending":  ("#6b6b6b", "#f9fafb", "Awaiting"),
    "running":  ("#4338ca", "#eef2ff", "Analysing"),
}

MODE_LABELS = {
    "payment_claim_serving":     "Payment claim — about to serve",
    "payment_claim_received":    "Payment claim — received",
    "payment_schedule_giving":   "Payment schedule — about to give",
    "payment_schedule_received": "Payment schedule — received",
}

DISCLAIMER_FULL = (
    "This is an automated compliance check based on publicly available law and the document "
    "provided. It is not a substitute for legal advice. Sopal is a research tool, not a legal "
    "adviser. For your specific situation, consult a qualified construction lawyer."
)
DISCLAIMER_SHORT = "General information only — not legal advice. Sopal is a research tool."


def build_report_pdf(
    *,
    mode: str,
    source_name: str | None,
    summary: str,
    checks: list[dict[str, Any]],
    user_answers: dict[str, Any] | None,
    check_input_labels: dict[str, dict[str, str]] | None = None,
) -> bytes:
    """Returns the PDF as a bytes object.

    ``check_input_labels`` optional: {check_id: {question_id: label}} so the
    appendix can show the question text rather than just IDs.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (
        BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
        KeepTogether, PageBreak, Image, HRFlowable,
    )
    from reportlab.lib.enums import TA_LEFT

    user_answers = user_answers or {}

    buf = io.BytesIO()

    page_w, page_h = A4
    margin_x = 18 * mm
    margin_top = 24 * mm
    margin_bottom = 22 * mm
    frame_width = page_w - margin_x * 2
    frame_height = page_h - margin_top - margin_bottom

    frame = Frame(
        margin_x, margin_bottom, frame_width, frame_height,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="body",
    )

    def _on_page(canvas, doc):
        canvas.saveState()
        # Footer disclaimer line.
        canvas.setFillColor(colors.HexColor("#6b6b6b"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(margin_x, margin_bottom - 12, DISCLAIMER_SHORT)
        canvas.drawRightString(
            page_w - margin_x, margin_bottom - 12,
            f"Page {canvas.getPageNumber()}"
        )
        # Top hairline (except first page which has the logo header).
        if canvas.getPageNumber() > 1:
            canvas.setStrokeColor(colors.HexColor("#e5e5e5"))
            canvas.setLineWidth(0.4)
            canvas.line(margin_x, page_h - margin_top + 14, page_w - margin_x, page_h - margin_top + 14)
            canvas.setFont("Helvetica", 8.5)
            canvas.setFillColor(colors.HexColor("#6b6b6b"))
            canvas.drawString(margin_x, page_h - margin_top + 18, "Sopal — Claim Check analysis report")
        canvas.restoreState()

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin_x, rightMargin=margin_x,
        topMargin=margin_top, bottomMargin=margin_bottom,
        title=f"Sopal Claim Check — {source_name or 'analysis'}",
        author="Sopal",
    )
    doc.addPageTemplates([PageTemplate(id="normal", frames=[frame], onPage=_on_page)])

    styles = getSampleStyleSheet()
    base = styles["Normal"]
    h1 = ParagraphStyle(
        "h1", parent=base, fontName="Helvetica-Bold", fontSize=20, leading=24,
        spaceAfter=4, textColor=colors.HexColor("#1a1a1a"),
    )
    h2 = ParagraphStyle(
        "h2", parent=base, fontName="Helvetica-Bold", fontSize=13, leading=16,
        spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"),
    )
    meta = ParagraphStyle(
        "meta", parent=base, fontName="Helvetica", fontSize=10, leading=14,
        textColor=colors.HexColor("#6b6b6b"),
    )
    body_style = ParagraphStyle(
        "body", parent=base, fontName="Helvetica", fontSize=10, leading=14,
        textColor=colors.HexColor("#1a1a1a"), spaceAfter=4,
    )
    strong_body = ParagraphStyle(
        "strongbody", parent=body_style, fontName="Helvetica-Bold",
    )
    quote_label = ParagraphStyle(
        "quotelabel", parent=base, fontName="Helvetica", fontSize=8.5, leading=11,
        textColor=colors.HexColor("#888"), spaceBefore=6, spaceAfter=0,
    )
    quote_style = ParagraphStyle(
        "quote", parent=base, fontName="Helvetica-Oblique", fontSize=9.5, leading=13,
        textColor=colors.HexColor("#4b4b4b"), leftIndent=10, rightIndent=4,
        borderColor=colors.HexColor("#e5e5e5"), borderWidth=0,
        spaceBefore=2, spaceAfter=4,
    )
    disclaimer_style = ParagraphStyle(
        "disc", parent=base, fontName="Helvetica", fontSize=9.5, leading=13,
        textColor=colors.HexColor("#7a4a00"), spaceBefore=2, spaceAfter=2,
    )
    section_badge_style = ParagraphStyle(
        "section", parent=base, fontName="Helvetica-Bold", fontSize=8.5, leading=11,
        textColor=colors.HexColor("#6b6b6b"),
    )

    story: list = []

    # --- Header (logo + title) ---
    try:
        if LOGO_PATH.exists():
            img = Image(str(LOGO_PATH), width=32 * mm, height=10 * mm, kind="proportional")
            story.append(img)
    except Exception as e:
        log.warning("logo load failed: %s", e)
    story.append(Paragraph("Claim Check — Analysis report", h1))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e5e5e5"), thickness=0.6, spaceBefore=6, spaceAfter=10))

    # --- Metadata ---
    mode_label = MODE_LABELS.get(mode, mode)
    date_str = datetime.now(timezone.utc).astimezone().strftime("%d %B %Y · %H:%M %Z")
    meta_rows = [
        ["Mode", mode_label],
        ["Document", source_name or "(pasted text)"],
        ["Generated", date_str],
    ]
    meta_table = Table(meta_rows, colWidths=[32 * mm, frame_width - 32 * mm])
    meta_table.setStyle(TableStyle([
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9.5),
        ("FONT", (1, 0), (1, -1), "Helvetica", 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6b6b6b")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#1a1a1a")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    if summary:
        story.append(Paragraph(f"<i>{_escape(summary)}</i>", meta))
        story.append(Spacer(1, 8))

    # --- Prominent disclaimer (first page) ---
    disc_table = Table(
        [[Paragraph(f"<b>This is not legal advice.</b> {DISCLAIMER_FULL}", disclaimer_style)]],
        colWidths=[frame_width],
    )
    disc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#fde68a")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disc_table)
    story.append(Spacer(1, 12))

    # --- Summary header (counts) ---
    counts = _counts(checks)
    summary_line = (
        f"<b>{len(checks)}</b> checks · "
        f"<font color='#b91c1c'><b>{counts['fail']}</b> issues</font> · "
        f"<font color='#92400e'><b>{counts['warning']}</b> warnings</font> · "
        f"<font color='#166534'><b>{counts['pass']}</b> passed</font> · "
        f"<font color='#374151'><b>{counts['input']}</b> need input</font>"
    )
    story.append(Paragraph(summary_line, body_style))
    story.append(Spacer(1, 10))

    # --- Check results ---
    story.append(Paragraph("Check results", h2))
    for c in checks:
        story.append(_check_block(c, strong_body, body_style, quote_label, quote_style, section_badge_style, frame_width, colors, Table, TableStyle, Paragraph, Spacer, KeepTogether))

    # --- Appendix: user answers ---
    if user_answers:
        story.append(PageBreak())
        story.append(Paragraph("Your answers to interactive checks", h2))
        rows = []
        for qid, val in user_answers.items():
            label = _resolve_label(qid, check_input_labels) or qid
            rows.append([_escape(label), _escape(_fmt_val(val))])
        if rows:
            tbl = Table(rows, colWidths=[frame_width * 0.55, frame_width * 0.45])
            tbl.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9.5),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6b6b6b")),
                ("TEXTCOLOR", (1, 0), (-1, -1), colors.HexColor("#1a1a1a")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#eee")),
            ]))
            story.append(tbl)

    # --- Final disclaimer page ---
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e5e5e5"), thickness=0.4))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Disclaimer.</b> {DISCLAIMER_FULL}", meta))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_block(c, strong_body, body_style, quote_label, quote_style, section_badge_style,
                 frame_width, colors, Table, TableStyle, Paragraph, Spacer, KeepTogether):
    status = c.get("status") or "pending"
    col_text, col_bg, default_summary = STATUS_COLORS.get(status, STATUS_COLORS["pending"])
    status_summary = c.get("status_summary") or default_summary
    title = c.get("title") or "(untitled)"
    section = c.get("section") or ""
    cid = c.get("id") or ""
    explanation = c.get("explanation") or ""
    quote = c.get("quote") or ""
    decisions = c.get("decisions") or []

    # Header row: status pill + check id + title + section badge
    header_paragraphs = Paragraph(
        f"<b><font color='{col_text}' size='9'>{_escape(status.upper())}</font></b> &nbsp; "
        f"<font color='#888' size='9'>{_escape(cid)}</font> &nbsp; "
        f"<b><font color='#1a1a1a' size='10.5'>{_escape(title)}</font></b>",
        body_style,
    )
    section_paragraph = Paragraph(
        f"<font color='#6b6b6b' size='9'>{_escape(section)}</font>", body_style
    )

    block = [
        Spacer(1, 8),
        Table([
            [header_paragraphs],
            [section_paragraph],
        ], colWidths=[frame_width], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(col_bg)),
            ("BACKGROUND", (0, 1), (-1, 1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e5e5")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])),
        Spacer(1, 5),
        Paragraph(f"<font color='{col_text}'><i>{_escape(status_summary)}</i></font>", body_style),
        Spacer(1, 3),
    ]
    if explanation:
        block.append(Paragraph(_escape(explanation), body_style))
    if quote:
        block.append(Paragraph("From your document:", quote_label))
        block.append(Paragraph(f"“{_escape(quote)}”", quote_style))
    if decisions:
        d_texts = []
        for d in decisions[:3]:
            t = d.get("title") or ""
            if t:
                d_texts.append(_escape(t))
        if d_texts:
            block.append(Paragraph(
                f"<font size='8.5' color='#6b6b6b'>Related: {' · '.join(d_texts)}</font>",
                body_style,
            ))
    return KeepTogether(block)


def _counts(checks: list[dict]) -> dict[str, int]:
    out = {"pass": 0, "warning": 0, "fail": 0, "input": 0}
    for c in checks or []:
        s = c.get("status")
        if s in out:
            out[s] += 1
    return out


def _escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # Strip any stray control characters and escape for reportlab Paragraph.
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    if isinstance(v, dict):
        # Licensee record — show the display line.
        return str(v.get("display") or v.get("entity_name") or v.get("licence_number") or v)
    return str(v)


def _resolve_label(qid: str, check_input_labels: dict | None) -> str | None:
    if not check_input_labels:
        return None
    for per_check in check_input_labels.values():
        if qid in per_check:
            return per_check[qid]
    return None
