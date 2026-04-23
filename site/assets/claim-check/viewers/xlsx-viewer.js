/* XLSX viewer for /claim-check.
 *
 * Parses the uploaded spreadsheet with SheetJS, renders a tab bar of sheet
 * names, and shows each sheet as a scrollable HTML table. Merged cells are
 * preserved via rowspan/colspan (SheetJS provides cell merges directly).
 *
 * Row cap: 5,000 per sheet with a "Show all N rows" expand button.
 *
 * Exported: window.ClaimCheckViewers._xlsx.render(mountEl, file)
 */

(function () {
  'use strict';

  const INITIAL_ROW_CAP = 5000;
  const MAX_MEMORY_BYTES = 100 * 1024 * 1024; // 100MB in-memory safety

  async function render(mountEl, file) {
    mountEl.innerHTML = `
      <div class="xlsxv">
        <div class="xlsxv-tabs"></div>
        <div class="xlsxv-scroll">
          <div class="xlsxv-status">Reading spreadsheet…</div>
          <div class="xlsxv-table-wrap"></div>
        </div>
      </div>`;

    const tabsEl = mountEl.querySelector('.xlsxv-tabs');
    const tableWrap = mountEl.querySelector('.xlsxv-table-wrap');
    const statusEl = mountEl.querySelector('.xlsxv-status');

    if (file.size > MAX_MEMORY_BYTES) {
      statusEl.textContent = 'File is too large to render in-browser (>100MB). The document is still being analysed.';
      return;
    }

    let buf;
    try {
      buf = await file.arrayBuffer();
    } catch (e) {
      statusEl.textContent = 'Could not read file.';
      return;
    }

    let wb;
    try {
      wb = window.XLSX.read(new Uint8Array(buf), { type: 'array', cellStyles: true, cellDates: true });
    } catch (e) {
      statusEl.textContent = `Could not parse spreadsheet: ${e.message || e}`;
      return;
    }

    if (!wb.SheetNames || !wb.SheetNames.length) {
      statusEl.textContent = 'Spreadsheet is empty.';
      return;
    }

    // Build tabs.
    const state = { activeSheet: wb.SheetNames[0], expanded: {} };

    function paintTabs() {
      tabsEl.innerHTML = wb.SheetNames.map((name) => {
        const active = name === state.activeSheet ? ' active' : '';
        return `<button class="xlsxv-tab${active}" data-sheet="${escapeAttr(name)}">${escapeHtml(name)}</button>`;
      }).join('');
      tabsEl.querySelectorAll('.xlsxv-tab').forEach((btn) => {
        btn.addEventListener('click', () => {
          state.activeSheet = btn.dataset.sheet;
          paintTabs();
          renderSheet();
        });
      });
    }

    function renderSheet() {
      const sheet = wb.Sheets[state.activeSheet];
      if (!sheet) {
        tableWrap.innerHTML = '<div class="xlsxv-empty">Sheet is empty.</div>';
        statusEl.hidden = true;
        return;
      }

      // Determine dimensions.
      const ref = sheet['!ref'];
      if (!ref) {
        tableWrap.innerHTML = '<div class="xlsxv-empty">Sheet is empty.</div>';
        statusEl.hidden = true;
        return;
      }
      const range = window.XLSX.utils.decode_range(ref);
      const totalRows = range.e.r - range.s.r + 1;
      const totalCols = range.e.c - range.s.c + 1;

      const expanded = state.expanded[state.activeSheet] === true;
      const rowCap = expanded ? totalRows : Math.min(totalRows, INITIAL_ROW_CAP);
      const capApplied = totalRows > INITIAL_ROW_CAP && !expanded;

      // Build HTML table. SheetJS sheet_to_html handles merges and values.
      // We'll build it manually so we can cap rows precisely.
      const html = buildSheetHTML(sheet, range, rowCap);
      const moreBtn = capApplied
        ? `<button class="xlsxv-expand">Show all ${totalRows.toLocaleString()} rows (currently showing first ${INITIAL_ROW_CAP.toLocaleString()})</button>`
        : '';
      tableWrap.innerHTML = `${moreBtn}${html}`;
      const btn = tableWrap.querySelector('.xlsxv-expand');
      if (btn) {
        btn.addEventListener('click', () => {
          state.expanded[state.activeSheet] = true;
          renderSheet();
        });
      }
      statusEl.hidden = true;
    }

    paintTabs();
    renderSheet();
  }

  function buildSheetHTML(sheet, range, rowCap) {
    // Column headers (A, B, C, ...).
    const startCol = range.s.c;
    const endCol = range.e.c;
    const startRow = range.s.r;
    const endRow = Math.min(range.e.r, range.s.r + rowCap - 1);

    // Merge map: key "r,c" -> {rowspan, colspan, isAnchor:true} for anchor cells,
    //            key "r,c" -> {skip: true} for cells hidden by a merge.
    const mergeMap = new Map();
    (sheet['!merges'] || []).forEach((m) => {
      const rs = m.e.r - m.s.r + 1;
      const cs = m.e.c - m.s.c + 1;
      mergeMap.set(`${m.s.r},${m.s.c}`, { isAnchor: true, rowspan: rs, colspan: cs });
      for (let r = m.s.r; r <= m.e.r; r++) {
        for (let c = m.s.c; c <= m.e.c; c++) {
          if (r === m.s.r && c === m.s.c) continue;
          mergeMap.set(`${r},${c}`, { skip: true });
        }
      }
    });

    const out = ['<table class="xlsxv-table"><thead><tr>'];
    out.push('<th class="xlsxv-th xlsxv-th-corner"></th>');
    for (let c = startCol; c <= endCol; c++) {
      out.push(`<th class="xlsxv-th">${window.XLSX.utils.encode_col(c)}</th>`);
    }
    out.push('</tr></thead><tbody>');

    for (let r = startRow; r <= endRow; r++) {
      out.push('<tr>');
      out.push(`<th class="xlsxv-row-num">${r + 1}</th>`);
      for (let c = startCol; c <= endCol; c++) {
        const k = `${r},${c}`;
        const meta = mergeMap.get(k);
        if (meta && meta.skip) continue;
        const addr = window.XLSX.utils.encode_cell({ r, c });
        const cell = sheet[addr];
        let display = '';
        let cls = 'xlsxv-td';
        if (cell) {
          if (cell.w != null) display = cell.w;
          else if (cell.v != null) display = String(cell.v);
          if (cell.t === 'n') cls += ' xlsxv-td-num';
          if (cell.s && cell.s.font && cell.s.font.bold) cls += ' xlsxv-td-bold';
          if (cell.s && cell.s.fill && cell.s.fill.fgColor && cell.s.fill.fgColor.rgb) {
            // inline bg color (xlsx uses ARGB hex)
            const rgb = cell.s.fill.fgColor.rgb;
            const hex = rgb.length === 8 ? rgb.slice(2) : rgb;
            cls += `" style="background:#${hex};`;
          }
        }
        let attrs = '';
        if (meta && meta.isAnchor) {
          if (meta.rowspan > 1) attrs += ` rowspan="${meta.rowspan}"`;
          if (meta.colspan > 1) attrs += ` colspan="${meta.colspan}"`;
        }
        out.push(`<td class="${cls}"${attrs}>${escapeHtml(display)}</td>`);
      }
      out.push('</tr>');
    }
    out.push('</tbody></table>');
    return out.join('');
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function escapeAttr(s) { return escapeHtml(s); }

  window.ClaimCheckViewers = window.ClaimCheckViewers || {};
  window.ClaimCheckViewers._xlsx = { render };
})();
