"""DOCX -> PDF conversion via LibreOffice headless.

Per spec Section 1: if LibreOffice is not installed on the server, the
route degrades gracefully — returns kind="unavailable" so the frontend
can show "DOCX preview temporarily unavailable" while the compliance
engine continues to run on the extracted text.

No document is persisted: conversion happens in a per-request tmp dir
which is deleted before this function returns.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("claim_check.docx_to_pdf")

# Look for libreoffice in the usual places (`which libreoffice`) or fall back to
# the common binary names.
_SOFFICE_CANDIDATES = ("libreoffice", "soffice")


def _find_soffice() -> str | None:
    for name in _SOFFICE_CANDIDATES:
        p = shutil.which(name)
        if p:
            return p
    # Common macOS install path.
    macos_bin = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    if os.path.exists(macos_bin):
        return macos_bin
    return None


def convert(docx_bytes: bytes, *, timeout: int = 45) -> dict[str, Any]:
    """Convert DOCX bytes to a PDF and return it base64-encoded.

    Returns one of:
      {"kind": "pdf", "bytes_base64": "<b64>", "size": <int>}
      {"kind": "unavailable", "reason": "<human-readable reason>"}
    """
    soffice = _find_soffice()
    if not soffice:
        return {
            "kind": "unavailable",
            "reason": "LibreOffice is not installed on the server. The document will still be analysed but can't be rendered in-browser.",
        }

    base = tempfile.mkdtemp(prefix=f"claim-check-conv-{uuid.uuid4().hex[:8]}-")
    try:
        src_path = Path(base) / "input.docx"
        src_path.write_bytes(docx_bytes)
        out_dir = Path(base) / "out"
        out_dir.mkdir(exist_ok=True)

        cmd = [
            soffice,
            "--headless",
            "--nologo",
            "--nodefault",
            "--nolockcheck",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(src_path),
        ]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                env={**os.environ, "HOME": base},  # avoid polluting user HOME
            )
        except FileNotFoundError:
            return {"kind": "unavailable", "reason": "LibreOffice binary not executable."}
        except subprocess.TimeoutExpired:
            return {"kind": "unavailable", "reason": "LibreOffice conversion timed out."}

        if res.returncode != 0:
            log.warning(
                "libreoffice exit %s: stdout=%s stderr=%s",
                res.returncode,
                (res.stdout or b"")[:200],
                (res.stderr or b"")[:200],
            )
            return {"kind": "unavailable", "reason": "LibreOffice conversion failed."}

        # Find the produced PDF.
        pdfs = sorted(out_dir.glob("*.pdf"))
        if not pdfs:
            return {"kind": "unavailable", "reason": "Conversion produced no PDF."}

        pdf_bytes = pdfs[0].read_bytes()
        return {
            "kind": "pdf",
            "bytes_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            "size": len(pdf_bytes),
        }
    finally:
        shutil.rmtree(base, ignore_errors=True)
