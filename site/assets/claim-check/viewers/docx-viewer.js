/* DOCX viewer for /claim-check.
 *
 * Strategy: upload the DOCX to /api/claim-check/preview which converts it
 * to PDF server-side via LibreOffice. If LibreOffice isn't installed, the
 * endpoint returns {"kind": "unavailable"} and we fall back to a friendly
 * message — the compliance engine still runs on the extracted text.
 *
 * Exported: window.ClaimCheckViewers._docx.render(mountEl, file, helpers)
 *           helpers.renderPDFBytes(mountEl, uint8arr, label) renders the
 *           returned PDF using the pdf-viewer module.
 */

(function () {
  'use strict';

  async function render(mountEl, file, helpers) {
    mountEl.innerHTML = `<div class="docxv-status">Converting DOCX for preview…</div>`;

    const fd = new FormData();
    fd.append('file', file);
    let data;
    try {
      const resp = await fetch('/api/claim-check/preview', { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `Preview failed (${resp.status})`);
      }
      data = await resp.json();
    } catch (e) {
      mountEl.innerHTML = `<div class="viewer-unavailable">DOCX preview unavailable (${(e && e.message) || 'error'}). The document is still being analysed.</div>`;
      return;
    }

    if (data && data.kind === 'pdf' && data.bytes_base64) {
      const uint8arr = b64ToUint8Array(data.bytes_base64);
      mountEl.innerHTML = '';
      await helpers.renderPDFBytes(mountEl, uint8arr, file.name);
      return;
    }

    // Unavailable — show a clear message.
    const reason = (data && data.reason) || 'DOCX preview is temporarily unavailable on this server.';
    mountEl.innerHTML = `<div class="viewer-unavailable">${escapeHtml(reason)} Analysis will still run on the extracted text.</div>`;
  }

  function b64ToUint8Array(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  window.ClaimCheckViewers = window.ClaimCheckViewers || {};
  window.ClaimCheckViewers._docx = { render };
})();
