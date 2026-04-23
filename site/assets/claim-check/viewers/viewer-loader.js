/* Claim Check document viewers — loader/dispatcher.
 *
 * Public API:
 *   window.ClaimCheckViewers.render(mountEl, file)   -> Promise<void>
 *   window.ClaimCheckViewers.destroy(mountEl)        -> void
 *
 * `file` is a File object from the upload/drop input. The loader decides
 * which viewer to use based on the filename extension and initialises it.
 */

(function () {
  'use strict';

  const LOGO_CDN_PDFJS_VERSION = '3.11.174';

  async function render(mountEl, file) {
    destroy(mountEl);
    mountEl.classList.add('viewer-mount');
    const name = (file.name || '').toLowerCase();

    try {
      if (name.endsWith('.pdf')) {
        await renderPDF(mountEl, file);
      } else if (name.endsWith('.xlsx') || name.endsWith('.xlsm')) {
        await renderXLSX(mountEl, file);
      } else if (name.endsWith('.docx')) {
        await renderDOCX(mountEl, file);
      } else {
        mountEl.innerHTML = `<div class="viewer-unavailable">Preview not available for this file type. The document is still being analysed.</div>`;
      }
    } catch (e) {
      console.error('viewer error', e);
      mountEl.innerHTML = `<div class="viewer-unavailable">Could not render preview: ${(e && e.message) || 'unknown error'}. The document is still being analysed.</div>`;
    }
  }

  function destroy(mountEl) {
    if (!mountEl) return;
    mountEl.innerHTML = '';
    mountEl.classList.remove('viewer-mount');
  }

  // ---------- PDF ----------
  async function renderPDF(mountEl, fileOrBytes) {
    await ensurePdfJs();
    if (window.ClaimCheckViewers._pdf) {
      return window.ClaimCheckViewers._pdf.render(mountEl, fileOrBytes);
    }
    throw new Error('PDF viewer module not loaded.');
  }

  async function renderPDFBytes(mountEl, uint8arr, label) {
    await ensurePdfJs();
    if (window.ClaimCheckViewers._pdf) {
      return window.ClaimCheckViewers._pdf.render(mountEl, uint8arr, { label });
    }
    throw new Error('PDF viewer module not loaded.');
  }

  // ---------- XLSX ----------
  async function renderXLSX(mountEl, file) {
    await ensureSheetJS();
    if (window.ClaimCheckViewers._xlsx) {
      return window.ClaimCheckViewers._xlsx.render(mountEl, file);
    }
    throw new Error('XLSX viewer module not loaded.');
  }

  // ---------- DOCX ----------
  async function renderDOCX(mountEl, file) {
    if (window.ClaimCheckViewers._docx) {
      return window.ClaimCheckViewers._docx.render(mountEl, file, { renderPDFBytes });
    }
    throw new Error('DOCX viewer module not loaded.');
  }

  // ---------- lazy loading of external libraries ----------
  let pdfjsPromise = null;
  function ensurePdfJs() {
    if (window.pdfjsLib) return Promise.resolve();
    if (pdfjsPromise) return pdfjsPromise;
    pdfjsPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${LOGO_CDN_PDFJS_VERSION}/pdf.min.js`;
      s.onload = () => {
        try {
          window.pdfjsLib.GlobalWorkerOptions.workerSrc =
            `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${LOGO_CDN_PDFJS_VERSION}/pdf.worker.min.js`;
          resolve();
        } catch (e) { reject(e); }
      };
      s.onerror = () => reject(new Error('Failed to load pdf.js from CDN.'));
      document.head.appendChild(s);
    });
    return pdfjsPromise;
  }

  let sheetjsPromise = null;
  function ensureSheetJS() {
    if (window.XLSX) return Promise.resolve();
    if (sheetjsPromise) return sheetjsPromise;
    sheetjsPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.20.3/xlsx.full.min.js';
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('Failed to load SheetJS from CDN.'));
      document.head.appendChild(s);
    });
    return sheetjsPromise;
  }

  // ---------- export ----------
  window.ClaimCheckViewers = window.ClaimCheckViewers || {};
  window.ClaimCheckViewers.render = render;
  window.ClaimCheckViewers.destroy = destroy;
  window.ClaimCheckViewers._ensurePdfJs = ensurePdfJs;
  window.ClaimCheckViewers._ensureSheetJS = ensureSheetJS;
})();
