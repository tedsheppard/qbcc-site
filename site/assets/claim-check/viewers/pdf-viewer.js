/* PDF viewer for /claim-check.
 *
 * Renders a PDF with pdf.js. Page navigation (prev/next), zoom (fit-to-width,
 * 100%, in/out), lazy canvas rendering for pages near the viewport, and
 * selectable text via pdf.js's text layer.
 *
 * Exported: window.ClaimCheckViewers._pdf.render(mountEl, fileOrBytes, opts)
 */

(function () {
  'use strict';

  const LAZY_MARGIN_PX = 800; // render pages within this margin of viewport

  async function render(mountEl, fileOrBytes, opts) {
    opts = opts || {};
    mountEl.innerHTML = `
      <div class="pdfv">
        <div class="pdfv-toolbar">
          <div class="pdfv-nav">
            <button class="pdfv-btn" data-act="prev" title="Previous page">‹</button>
            <span class="pdfv-page"><input type="number" class="pdfv-page-input" min="1" value="1"> / <span class="pdfv-page-total">?</span></span>
            <button class="pdfv-btn" data-act="next" title="Next page">›</button>
          </div>
          <div class="pdfv-zoom">
            <button class="pdfv-btn" data-act="fit" title="Fit width">Fit</button>
            <button class="pdfv-btn" data-act="zout" title="Zoom out">−</button>
            <span class="pdfv-zoom-label">100%</span>
            <button class="pdfv-btn" data-act="zin" title="Zoom in">+</button>
            <button class="pdfv-btn" data-act="actual" title="100%">100%</button>
          </div>
        </div>
        <div class="pdfv-scroll">
          <div class="pdfv-pages"></div>
        </div>
      </div>`;

    const scrollEl = mountEl.querySelector('.pdfv-scroll');
    const pagesEl = mountEl.querySelector('.pdfv-pages');
    const pageInput = mountEl.querySelector('.pdfv-page-input');
    const pageTotal = mountEl.querySelector('.pdfv-page-total');
    const zoomLabel = mountEl.querySelector('.pdfv-zoom-label');

    // Load doc from File or Uint8Array.
    let data;
    if (fileOrBytes instanceof Uint8Array) data = fileOrBytes;
    else if (fileOrBytes instanceof ArrayBuffer) data = new Uint8Array(fileOrBytes);
    else if (fileOrBytes && fileOrBytes.arrayBuffer) data = new Uint8Array(await fileOrBytes.arrayBuffer());
    else throw new Error('Unsupported PDF input.');

    const loadingTask = window.pdfjsLib.getDocument({ data });
    const pdf = await loadingTask.promise;
    const numPages = pdf.numPages;
    pageTotal.textContent = String(numPages);
    pageInput.max = String(numPages);

    // State
    const state = {
      pdf,
      numPages,
      zoom: 1.0,
      fitMode: 'fit', // 'fit' | 'custom'
      pageWrappers: [],
      rendered: new Set(),
      currentPage: 1,
    };

    // Build page placeholders; lazy-render when scrolled in.
    for (let i = 1; i <= numPages; i++) {
      const wrap = document.createElement('div');
      wrap.className = 'pdfv-page-wrap';
      wrap.dataset.pageNum = String(i);
      wrap.innerHTML = `
        <div class="pdfv-page-num">Page ${i}</div>
        <div class="pdfv-canvas-wrap">
          <canvas class="pdfv-canvas"></canvas>
          <div class="pdfv-textlayer"></div>
        </div>`;
      pagesEl.appendChild(wrap);
      state.pageWrappers.push(wrap);
    }

    async function computeScaleForFit() {
      const firstPage = await pdf.getPage(1);
      const unscaled = firstPage.getViewport({ scale: 1.0 });
      const avail = scrollEl.clientWidth - 32;
      return Math.max(0.4, Math.min(3, avail / unscaled.width));
    }

    async function layoutPages() {
      if (state.fitMode === 'fit') {
        state.zoom = await computeScaleForFit();
      }
      zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
      // Size each page wrapper to the expected dimensions so the scroll container has the right height.
      for (let i = 1; i <= numPages; i++) {
        const wrap = state.pageWrappers[i - 1];
        const cacheKey = `size_${i}`;
        let vp = wrap[cacheKey];
        if (!vp) {
          const p = await pdf.getPage(i);
          vp = p.getViewport({ scale: 1.0 });
          wrap[cacheKey] = vp;
        }
        const canvasWrap = wrap.querySelector('.pdfv-canvas-wrap');
        canvasWrap.style.width = `${Math.floor(vp.width * state.zoom)}px`;
        canvasWrap.style.height = `${Math.floor(vp.height * state.zoom)}px`;
        // Any already-rendered canvas should be re-rendered at the new zoom.
        if (state.rendered.has(i)) {
          state.rendered.delete(i);
        }
      }
      // Trigger lazy-render pass for visible pages.
      lazyRenderVisible();
    }

    async function renderPage(i) {
      if (state.rendered.has(i)) return;
      const wrap = state.pageWrappers[i - 1];
      const canvas = wrap.querySelector('.pdfv-canvas');
      const textLayer = wrap.querySelector('.pdfv-textlayer');
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: state.zoom });
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * dpr);
      canvas.height = Math.floor(viewport.height * dpr);
      canvas.style.width = `${Math.floor(viewport.width)}px`;
      canvas.style.height = `${Math.floor(viewport.height)}px`;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      await page.render({ canvasContext: ctx, viewport }).promise;
      // Text layer for selection.
      try {
        const textContent = await page.getTextContent();
        textLayer.innerHTML = '';
        textLayer.style.width = `${Math.floor(viewport.width)}px`;
        textLayer.style.height = `${Math.floor(viewport.height)}px`;
        if (window.pdfjsLib.renderTextLayer) {
          window.pdfjsLib.renderTextLayer({
            textContentSource: textContent,
            container: textLayer,
            viewport,
            textDivs: [],
          });
        }
      } catch (e) {
        console.warn('text layer failed for page', i, e);
      }
      state.rendered.add(i);
    }

    function lazyRenderVisible() {
      const top = scrollEl.scrollTop - LAZY_MARGIN_PX;
      const bottom = scrollEl.scrollTop + scrollEl.clientHeight + LAZY_MARGIN_PX;
      state.pageWrappers.forEach((wrap, idx) => {
        const t = wrap.offsetTop;
        const b = t + wrap.offsetHeight;
        if (b >= top && t <= bottom) renderPage(idx + 1);
      });
    }

    scrollEl.addEventListener('scroll', () => {
      lazyRenderVisible();
      // Update current page indicator based on which wrapper's top is closest to scrollTop.
      let best = 1, bestDist = Infinity;
      const y = scrollEl.scrollTop + 32;
      state.pageWrappers.forEach((wrap, idx) => {
        const dist = Math.abs(wrap.offsetTop - y);
        if (dist < bestDist) { bestDist = dist; best = idx + 1; }
      });
      if (best !== state.currentPage) {
        state.currentPage = best;
        pageInput.value = String(best);
      }
    });

    // Toolbar actions.
    mountEl.querySelectorAll('.pdfv-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const act = btn.dataset.act;
        if (act === 'prev') goToPage(state.currentPage - 1);
        else if (act === 'next') goToPage(state.currentPage + 1);
        else if (act === 'fit') { state.fitMode = 'fit'; await layoutPages(); }
        else if (act === 'actual') { state.fitMode = 'custom'; state.zoom = 1.0; await layoutPages(); }
        else if (act === 'zin') { state.fitMode = 'custom'; state.zoom = Math.min(3, state.zoom + 0.15); await layoutPages(); }
        else if (act === 'zout') { state.fitMode = 'custom'; state.zoom = Math.max(0.4, state.zoom - 0.15); await layoutPages(); }
      });
    });
    pageInput.addEventListener('change', () => {
      const n = parseInt(pageInput.value, 10);
      if (!isNaN(n)) goToPage(n);
    });

    function goToPage(n) {
      n = Math.max(1, Math.min(numPages, n));
      state.currentPage = n;
      pageInput.value = String(n);
      const wrap = state.pageWrappers[n - 1];
      if (wrap) scrollEl.scrollTo({ top: wrap.offsetTop - 12, behavior: 'smooth' });
    }

    // Initial layout + render first page.
    await layoutPages();

    // Re-layout on container resize.
    let ro;
    if (window.ResizeObserver) {
      ro = new ResizeObserver(() => { if (state.fitMode === 'fit') layoutPages(); });
      ro.observe(scrollEl);
      mountEl._ro = ro;
    }

    if (opts.label) {
      const badge = document.createElement('div');
      badge.className = 'pdfv-label';
      badge.textContent = opts.label;
      mountEl.querySelector('.pdfv-toolbar').appendChild(badge);
    }
  }

  window.ClaimCheckViewers = window.ClaimCheckViewers || {};
  window.ClaimCheckViewers._pdf = { render };
})();
