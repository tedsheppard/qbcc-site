/* Claim Check — frontend controller (v2).

   Fetches the checklist for the picked mode up-front, renders all checks in
   "pending" state with structured user-input forms, and streams analyse
   results back via SSE once a document is uploaded.

   State lives in sessionStorage (tab-scoped). No auth. No persistent server
   storage of document content.
*/

(function () {
  'use strict';

  // ---------- constants ----------
  const MAX_BYTES = 50 * 1024 * 1024; // 50MB per spec Section 1
  const SS_KEY = 'sopal.claimCheck.v2';
  const MIN_STATUS_DISPLAY_MS = 400; // Section 4: each status visible for >=400ms

  const MODE_LABELS = {
    payment_claim_serving:     "Payment claim I'm about to serve",
    payment_claim_received:    "Payment claim I've received",
    payment_schedule_giving:   "Payment schedule I'm about to give",
    payment_schedule_received: "Payment schedule I've received",
  };

  const STATUS_SUMMARY = {
    pass:    'No issues detected based on the information provided',
    warning: 'Potential issue identified — review recommended',
    fail:    'Likely non-compliant — this requires attention',
    input:   'Additional information required to complete this check',
    pending: 'Awaiting document',
    running: 'Analysing…',
  };

  // ---------- state ----------
  const state = {
    mode: null,
    doc: null,
    documentText: '',
    summary: '',
    checks: [],            // [{id, title, section_ref, search_query, input_required, input_questions}]
    results: {},           // { [check_id]: { status, status_summary, explanation, quote, decisions } }
    states: {},            // { [check_id]: 'pending' | 'running' | 'pass' | 'warning' | 'fail' | 'input' }
    userAnswers: {},       // { [input_id]: value }  for input questions
    licenseeRecords: {},   // { [input_id]: { ...record } } for licensee_lookup
    history: [],
  };

  // ---------- element refs ----------
  const el = (id) => document.getElementById(id);
  const tiles = document.querySelectorAll('.mode-tile');
  const workspace = el('workspace');
  const explainer = el('explainer');
  const modeValue = el('workspace-mode-value');

  const uploadZone = el('upload-zone');
  const fileInput = el('file-input');
  const pasteZone = el('paste-zone');
  const pasteInput = el('paste-input');
  const pasteMeta = el('paste-meta');
  const pasteSubmit = el('btn-paste-submit');
  const btnPasteToggle = el('btn-paste-toggle');

  const previewZone = el('preview-zone');
  const previewFilename = el('preview-filename');
  const previewSize = el('preview-size');
  const previewText = el('preview-text');
  const btnChangeDoc = el('btn-change-doc');

  const analysisPane = document.querySelector('.pane-analysis');
  const analysisEmpty = el('analysis-empty');
  const analysisList = el('analysis-list');
  const analysisCount = el('analysis-count');
  const analysisDisclaimer = el('analysis-disclaimer');
  const thinkingBar = el('thinking-bar');
  const thinkingText = el('thinking-text');

  const btnReport = el('btn-report');
  const chatbotMessages = el('chatbot-messages');
  const chatbotForm = el('chatbot-form');
  const chatbotInput = el('chatbot-input');
  const chatbotSend = el('chatbot-send');

  const mobileTabs = document.querySelectorAll('.mobile-tab');

  // ---------- session persistence ----------
  function saveSession() {
    try {
      sessionStorage.setItem(SS_KEY, JSON.stringify({
        mode: state.mode,
        documentText: state.documentText,
        summary: state.summary,
        checks: state.checks,
        results: state.results,
        states: state.states,
        userAnswers: state.userAnswers,
        licenseeRecords: state.licenseeRecords,
        history: state.history,
        docMeta: state.doc ? { filename: state.doc.filename, size: state.doc.size, kind: state.doc.kind } : null,
      }));
    } catch (_) {}
  }
  function loadSession() {
    try {
      const raw = sessionStorage.getItem(SS_KEY);
      if (!raw) return false;
      const s = JSON.parse(raw);
      Object.assign(state, {
        mode: s.mode || null,
        documentText: s.documentText || '',
        summary: s.summary || '',
        checks: s.checks || [],
        results: s.results || {},
        states: s.states || {},
        userAnswers: s.userAnswers || {},
        licenseeRecords: s.licenseeRecords || {},
        history: s.history || [],
        doc: s.docMeta ? { ...s.docMeta } : null,
      });
      return true;
    } catch (_) { return false; }
  }

  // ---------- mode picker ----------
  tiles.forEach((tile) => tile.addEventListener('click', () => setMode(tile.dataset.mode)));

  async function setMode(mode) {
    if (!MODE_LABELS[mode]) return;
    const changed = state.mode !== mode;
    state.mode = mode;
    tiles.forEach((t) => {
      const active = t.dataset.mode === mode;
      t.classList.toggle('active', active);
      t.setAttribute('aria-checked', active ? 'true' : 'false');
    });
    modeValue.textContent = MODE_LABELS[mode] || '—';
    workspace.hidden = false;
    explainer.hidden = true;
    analysisEmpty.hidden = true;
    analysisDisclaimer.hidden = false;

    if (changed) {
      // Clear results from a previous mode; keep the document if one is loaded.
      state.results = {};
      state.states = {};
      // Don't clear userAnswers — answers are often mode-agnostic (dates etc).
    }

    await loadChecksForMode(mode);

    if (state.doc && changed) runAnalysis();
    saveSession();
  }

  async function loadChecksForMode(mode) {
    try {
      const resp = await fetch(`/api/claim-check/checks?mode=${encodeURIComponent(mode)}`);
      if (!resp.ok) throw new Error(`checks endpoint ${resp.status}`);
      const data = await resp.json();
      state.checks = data.checks || [];
      state.checks.forEach((c) => {
        if (!state.states[c.id]) state.states[c.id] = 'pending';
      });
      renderChecklist();
    } catch (e) {
      analysisList.innerHTML = `<li class="analysis-item error">Could not load checks: ${escapeHtml(e.message)}</li>`;
      analysisList.hidden = false;
    }
  }

  // ---------- upload / paste ----------
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  ['dragenter', 'dragover'].forEach((evt) =>
    uploadZone.addEventListener(evt, (e) => { e.preventDefault(); uploadZone.classList.add('dragging'); })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    uploadZone.addEventListener(evt, (e) => { e.preventDefault(); uploadZone.classList.remove('dragging'); })
  );
  uploadZone.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  fileInput.addEventListener('change', () => {
    const f = fileInput.files && fileInput.files[0];
    if (f) handleFile(f);
  });

  function handleFile(file) {
    if (!state.mode) { alert('Pick a mode first.'); return; }
    if (file.size > MAX_BYTES) { alert('File is too large. Max 50 MB.'); return; }
    const name = (file.name || '').toLowerCase();
    if (!/\.(pdf|docx|xlsx|xlsm)$/i.test(name)) {
      alert('Please upload a PDF, DOCX, or XLSX file.');
      return;
    }
    state.doc = { kind: 'file', filename: file.name, size: file.size, _file: file };
    showPreviewShell();
    runAnalysis();
  }

  btnPasteToggle.addEventListener('click', () => {
    const showPaste = pasteZone.hidden;
    pasteZone.hidden = !showPaste;
    uploadZone.hidden = showPaste;
    document.querySelectorAll('[data-paste-label]').forEach((s) => {
      const show = s.dataset.pasteLabel === (showPaste ? 'toggle-to-upload' : 'toggle-to-paste');
      s.hidden = !show;
    });
  });
  pasteInput.addEventListener('input', () => {
    const text = pasteInput.value;
    pasteMeta.textContent = `${text.length.toLocaleString()} characters`;
    pasteSubmit.disabled = text.trim().length < 40;
  });
  pasteSubmit.addEventListener('click', () => {
    if (!state.mode) { alert('Pick a mode first.'); return; }
    const text = pasteInput.value;
    state.doc = { kind: 'paste', filename: 'Pasted text', size: new Blob([text]).size, text };
    state.documentText = text;
    showPreviewShell();
    runAnalysis();
  });

  function showPreviewShell() {
    previewFilename.textContent = state.doc.filename;
    previewSize.textContent = state.doc.size ? `— ${formatBytes(state.doc.size)}` : '';
    previewText.textContent = state.doc.kind === 'paste' ? (state.doc.text || '') : 'Extracting text…';
    uploadZone.hidden = true;
    pasteZone.hidden = true;
    previewZone.hidden = false;
    analysisPane.hidden = false;
    chatbotInput.placeholder = 'Ask a question about this document…';
  }

  btnChangeDoc.addEventListener('click', () => resetDocState());

  function resetDocState() {
    state.doc = null;
    state.documentText = '';
    state.summary = '';
    state.results = {};
    state.states = {};
    state.history = [];
    previewZone.hidden = true;
    uploadZone.hidden = false;
    pasteInput.value = '';
    pasteMeta.textContent = '0 characters';
    pasteSubmit.disabled = true;
    chatbotInput.placeholder = 'Upload a document to start the conversation…';
    chatbotInput.disabled = true;
    chatbotSend.disabled = true;
    btnReport.disabled = true;
    chatbotMessages.innerHTML = '<div class="chatbot-empty"><p>Upload a document above, then ask questions like <em>"why did you flag the reference date?"</em> or <em>"what defences could I raise under s 69?"</em></p></div>';
    if (state.mode) {
      state.checks.forEach((c) => { state.states[c.id] = 'pending'; });
      renderChecklist();
    }
    saveSession();
  }

  // ---------- checklist rendering ----------
  function renderChecklist() {
    if (!state.checks.length) {
      analysisEmpty.hidden = false;
      analysisList.hidden = true;
      analysisDisclaimer.hidden = true;
      return;
    }
    analysisEmpty.hidden = true;
    analysisDisclaimer.hidden = false;
    analysisList.hidden = false;
    analysisCount.hidden = false;
    analysisCount.textContent = `${state.checks.length} checks`;

    analysisList.innerHTML = state.checks.map((c) => renderCheckRow(c)).join('');
    wireCheckInputs();
  }

  function renderCheckRow(check) {
    const st = state.states[check.id] || 'pending';
    const result = state.results[check.id];
    const iconHtml = statusIconSvg(st, check);
    const searchHref = `/search?q=${encodeURIComponent(check.search_query || check.title)}`;
    const summaryLine = (result && result.status_summary) || STATUS_SUMMARY[st] || '';
    const explanationHtml = (result && result.explanation)
      ? `<div class="check-explanation">${escapeHtml(result.explanation)}</div>`
      : '';
    const quoteHtml = (result && result.quote)
      ? `<div class="check-quote">“${escapeHtml(result.quote)}”</div>`
      : '';
    const inputsHtml = (check.input_questions && check.input_questions.length)
      ? renderInputsBlock(check)
      : '';
    const decisionsHtml = (result && result.decisions && result.decisions.length)
      ? `<div class="check-decisions">Related decisions: ${result.decisions.map(d => `<a href="${escapeAttr(d.url || searchHref)}" target="_blank" rel="noopener">${escapeHtml(d.title || 'decision')}</a>`).join(' · ')}</div>`
      : '';

    return `<li class="analysis-item" data-check-id="${escapeAttr(check.id)}" data-state="${escapeAttr(st)}">
      <div class="check-row">
        <span class="check-icon" data-status="${escapeAttr(st)}">${iconHtml}</span>
        <div class="check-body">
          <div class="check-title">${escapeHtml(check.title)}</div>
          <div class="check-meta">
            ${check.section_ref ? `<span class="check-section">${escapeHtml(check.section_ref)}</span>` : ''}
            <a class="check-link" href="${searchHref}" target="_blank" rel="noopener">See relevant decisions →</a>
          </div>
          <div class="check-summary">${escapeHtml(summaryLine)}</div>
          ${explanationHtml}
          ${quoteHtml}
          ${decisionsHtml}
          ${inputsHtml}
        </div>
      </div>
    </li>`;
  }

  // ---------- structured inputs ----------
  function renderInputsBlock(check) {
    const qs = check.input_questions || [];
    const rows = qs.map((q) => renderInputRow(check.id, q)).join('');
    return `<div class="check-inputs" data-for-check="${escapeAttr(check.id)}">${rows}</div>`;
  }

  function renderInputRow(checkId, q) {
    const visible = evalShowIf(q.show_if);
    const stored = state.userAnswers[q.id];
    const storedStr = stored == null ? '' : String(stored);
    const safeStored = escapeAttr(storedStr);
    const reqMark = q.required ? ' <span class="req">*</span>' : '';
    const inputId = `in-${checkId}-${q.id}`;
    const today = new Date().toISOString().slice(0, 10);
    let controlHtml = '';

    if (q.type === 'date') {
      const maxAttr = q.no_future ? ` max="${today}"` : '';
      controlHtml = `<input type="date" id="${inputId}" data-qid="${escapeAttr(q.id)}" data-qtype="date" value="${safeStored}"${maxAttr}>`;
    } else if (q.type === 'radio') {
      const opts = (q.options || []).map((o) => {
        const checked = storedStr === o ? ' checked' : '';
        return `<label class="radio-pill"><input type="radio" name="${inputId}" data-qid="${escapeAttr(q.id)}" data-qtype="radio" value="${escapeAttr(o)}"${checked}><span>${escapeHtml(o)}</span></label>`;
      }).join('');
      controlHtml = `<div class="radio-row" id="${inputId}">${opts}</div>`;
    } else if (q.type === 'number') {
      controlHtml = `<input type="number" id="${inputId}" data-qid="${escapeAttr(q.id)}" data-qtype="number" value="${safeStored}">`;
    } else if (q.type === 'licensee_lookup') {
      const record = state.licenseeRecords[q.id];
      const selectedHtml = record
        ? `<div class="licensee-selected" data-qid="${escapeAttr(q.id)}">
            <div class="licensee-selected-name">${escapeHtml(record.display || record.entity_name || '')}</div>
            <div class="licensee-selected-meta">
              ${record.licence_number ? `Licence ${escapeHtml(record.licence_number)}` : ''}
              ${record.licence_status ? ` · ${escapeHtml(record.licence_status)}` : ''}
              ${record.licence_classes && record.licence_classes.length ? ` · ${escapeHtml(record.licence_classes.join(', '))}` : ''}
            </div>
            <button type="button" class="licensee-clear" data-qid="${escapeAttr(q.id)}">Change</button>
          </div>`
        : '';
      controlHtml = `
        <div class="licensee-lookup" data-qid="${escapeAttr(q.id)}">
          ${selectedHtml}
          <input type="text" id="${inputId}" class="licensee-input" data-qid="${escapeAttr(q.id)}" placeholder="Start typing the claimant's company name…" autocomplete="off"${record ? ' hidden' : ''}>
          <div class="licensee-results" id="${inputId}-results" hidden></div>
          <div class="licensee-caveat">The register reflects CURRENT licence status only — not historical status at the time the work was performed.</div>
        </div>`;
    } else {
      // text fallback
      controlHtml = `<input type="text" id="${inputId}" data-qid="${escapeAttr(q.id)}" data-qtype="text" value="${safeStored}" placeholder="Your answer">`;
    }

    return `<div class="check-input-row${visible ? '' : ' hidden'}" data-qid="${escapeAttr(q.id)}" data-show-if="${escapeAttr(q.show_if || '')}">
      <label class="check-input-label" for="${inputId}">${escapeHtml(q.question || q.id)}${reqMark}</label>
      ${controlHtml}
    </div>`;
  }

  function evalShowIf(expr) {
    if (!expr) return true;
    const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*==\s*"(.*)"\s*$/.exec(expr);
    if (!m) return true;
    return String(state.userAnswers[m[1]] || '') === m[2];
  }

  function wireCheckInputs() {
    // Text / date / number
    analysisList.querySelectorAll('input[type="date"], input[type="number"], input[type="text"]:not(.licensee-input)')
      .forEach((inp) => {
        inp.addEventListener('change', () => recordAnswer(inp.dataset.qid, inp.value));
        inp.addEventListener('blur', () => recordAnswer(inp.dataset.qid, inp.value));
      });
    // Radios
    analysisList.querySelectorAll('input[type="radio"]').forEach((inp) => {
      inp.addEventListener('change', () => {
        if (inp.checked) recordAnswer(inp.dataset.qid, inp.value);
      });
    });
    // Licensee lookup
    analysisList.querySelectorAll('.licensee-input').forEach((inp) => setupLicenseeInput(inp));
    analysisList.querySelectorAll('.licensee-clear').forEach((btn) => {
      btn.addEventListener('click', () => {
        const qid = btn.dataset.qid;
        delete state.licenseeRecords[qid];
        delete state.userAnswers[qid];
        saveSession();
        renderChecklist();
      });
    });
  }

  function recordAnswer(qid, value) {
    if (!qid) return;
    state.userAnswers[qid] = value;
    // Conditional visibility may now change — re-evaluate.
    updateConditionalVisibility();
    saveSession();
  }

  function updateConditionalVisibility() {
    analysisList.querySelectorAll('.check-input-row[data-show-if]').forEach((row) => {
      const expr = row.dataset.showIf;
      if (!expr) return;
      row.classList.toggle('hidden', !evalShowIf(expr));
    });
  }

  // ---------- licensee lookup ----------
  function setupLicenseeInput(inp) {
    let debounceTimer = null;
    const qid = inp.dataset.qid;
    const resultsEl = document.getElementById(inp.id + '-results');
    if (!resultsEl) return;

    inp.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      const q = inp.value.trim();
      if (q.length < 2) {
        resultsEl.hidden = true;
        resultsEl.innerHTML = '';
        return;
      }
      debounceTimer = setTimeout(() => runLicenseeSearch(q, resultsEl, qid), 300);
    });
    inp.addEventListener('blur', () => {
      // Let click-on-result fire first.
      setTimeout(() => { resultsEl.hidden = true; }, 150);
    });
    inp.addEventListener('focus', () => {
      if (resultsEl.innerHTML) resultsEl.hidden = false;
    });
  }

  async function runLicenseeSearch(q, resultsEl, qid) {
    resultsEl.innerHTML = '<div class="licensee-row muted">Searching QBCC register…</div>';
    resultsEl.hidden = false;
    try {
      const resp = await fetch(`/api/claim-check/qbcc-search?q=${encodeURIComponent(q)}`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Lookup failed (${resp.status})`);
      }
      const data = await resp.json();
      const results = data.results || [];
      if (!results.length) {
        resultsEl.innerHTML = '<div class="licensee-row muted">No matches. The claimant may not hold a QBCC licence.</div>';
        return;
      }
      resultsEl.innerHTML = results.slice(0, 10).map((r, i) => {
        const classes = (r.licence_classes || []).join(', ');
        const statusCls = (r.licence_status || '').toLowerCase().includes('active') ? 'ok' : 'warn';
        return `<button class="licensee-row" data-idx="${i}" data-qid="${escapeAttr(qid)}">
          <div class="licensee-name">${escapeHtml(r.display || r.entity_name || '')}</div>
          <div class="licensee-meta">
            <span class="licensee-status ${statusCls}">${escapeHtml(r.licence_status || '—')}</span>
            ${r.licence_number ? `<span>· Licence ${escapeHtml(r.licence_number)}</span>` : ''}
            ${classes ? `<span>· ${escapeHtml(classes)}</span>` : ''}
          </div>
        </button>`;
      }).join('');
      resultsEl.querySelectorAll('.licensee-row[data-idx]').forEach((btn) => {
        btn.addEventListener('mousedown', (e) => {
          e.preventDefault();
          const idx = parseInt(btn.dataset.idx, 10);
          const record = results[idx];
          state.licenseeRecords[qid] = record;
          state.userAnswers[qid] = record.display || record.entity_name || '';
          saveSession();
          renderChecklist();
        });
      });
    } catch (e) {
      resultsEl.innerHTML = `<div class="licensee-row error">QBCC lookup failed — ${escapeHtml(e.message)}. You can still continue and answer manually.</div>`;
    }
  }

  // ---------- analysis (SSE) ----------
  let statusQueue = [];
  let statusDraining = false;

  async function runAnalysis() {
    if (!state.mode || !state.doc) return;
    // Mark all checks as running
    state.checks.forEach((c) => { state.states[c.id] = 'running'; });
    state.results = {};
    renderChecklist();
    thinkingBar.hidden = false;
    setThinking('Starting analysis…');
    btnReport.disabled = true;
    chatbotInput.disabled = true;
    chatbotSend.disabled = true;

    const form = new FormData();
    form.append('mode', state.mode);
    if (state.doc.kind === 'file' && state.doc._file) {
      form.append('file', state.doc._file);
    } else {
      form.append('pasted_text', state.doc.text || state.documentText || '');
    }
    if (Object.keys(state.userAnswers).length) {
      form.append('user_answers', JSON.stringify(state.userAnswers));
    }

    try {
      const resp = await fetch('/api/claim-check/analyse-stream', { method: 'POST', body: form });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `Analysis failed (${resp.status})`);
      }
      if (!resp.body) throw new Error('Streaming response body unavailable.');
      await consumeSSE(resp.body, handleSseEvent);
    } catch (e) {
      thinkingBar.hidden = true;
      const msg = e.message || 'Analysis failed.';
      state.checks.forEach((c) => {
        if (state.states[c.id] === 'running') state.states[c.id] = 'warning';
      });
      state.results['_error'] = { explanation: msg };
      renderChecklist();
      appendErrorRow(msg);
    } finally {
      await flushStatusQueue();
      thinkingBar.hidden = true;
      btnReport.disabled = false;
      chatbotInput.disabled = false;
      chatbotSend.disabled = false;
      saveSession();
    }
  }

  function handleSseEvent(event, data) {
    if (event === 'status') {
      enqueueStatus(data.message || '');
    } else if (event === 'meta') {
      state.summary = data.summary || '';
      state.documentText = data.document_text || state.documentText;
      if (state.doc && state.doc.kind === 'file' && data.chars) {
        // Preview text will be replaced by viewers in Section 1; for now show a brief notice.
        previewText.textContent = `${data.source_name || 'Document'} — ${data.chars.toLocaleString()} characters extracted`;
      }
    } else if (event === 'check_result') {
      const id = data.id;
      state.states[id] = data.status || 'warning';
      state.results[id] = data;
      renderChecklist();
    } else if (event === 'complete') {
      if (data && data.document_text) state.documentText = data.document_text;
      state.checks.forEach((c) => {
        if (state.states[c.id] === 'running') state.states[c.id] = 'warning';
      });
    } else if (event === 'error') {
      appendErrorRow(data.message || 'Unknown error.');
    }
  }

  function enqueueStatus(msg) {
    statusQueue.push(msg);
    if (!statusDraining) drainStatusQueue();
  }

  async function drainStatusQueue() {
    statusDraining = true;
    while (statusQueue.length) {
      const msg = statusQueue.shift();
      setThinking(msg);
      await sleep(MIN_STATUS_DISPLAY_MS);
    }
    statusDraining = false;
  }

  async function flushStatusQueue() {
    while (statusDraining || statusQueue.length) {
      await sleep(50);
    }
  }

  function setThinking(msg) {
    thinkingText.textContent = msg || '';
  }

  async function consumeSSE(stream, onEvent) {
    const reader = stream.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const parsed = parseSseChunk(chunk);
        if (parsed) {
          try { onEvent(parsed.event, parsed.data); } catch (e) { console.error('SSE handler error', e); }
        }
      }
    }
  }

  function parseSseChunk(chunk) {
    const lines = chunk.split('\n');
    let event = 'message';
    const dataParts = [];
    for (const ln of lines) {
      if (ln.startsWith('event:')) event = ln.slice(6).trim();
      else if (ln.startsWith('data:')) dataParts.push(ln.slice(5).trim());
    }
    if (!dataParts.length) return null;
    try {
      const data = JSON.parse(dataParts.join('\n'));
      return { event, data };
    } catch (_) {
      return { event, data: {} };
    }
  }

  function appendErrorRow(msg) {
    // Push a synthetic row at the top of the list.
    const existing = analysisList.querySelector('.analysis-item.error');
    if (existing) existing.remove();
    const node = document.createElement('li');
    node.className = 'analysis-item error';
    node.innerHTML = `<div class="check-row"><div class="check-body"><div class="check-title">Analysis error</div><div class="check-explanation">${escapeHtml(msg)}</div></div></div>`;
    analysisList.insertBefore(node, analysisList.firstChild);
  }

  // ---------- chatbot ----------
  chatbotForm.addEventListener('submit', (e) => { e.preventDefault(); sendChat(); });

  async function sendChat() {
    const msg = (chatbotInput.value || '').trim();
    if (!msg || chatbotInput.disabled) return;
    appendMsg('user', msg);
    state.history.push({ role: 'user', content: msg });
    chatbotInput.value = '';
    chatbotInput.disabled = true;
    chatbotSend.disabled = true;
    const thinkingNode = appendMsg('assistant', '…');
    thinkingNode.classList.add('thinking');
    try {
      const resp = await fetch('/api/claim-check/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: state.mode,
          message: msg,
          history: state.history.slice(0, -1),
          document_text: state.documentText,
          checks: state.checks.map((c) => {
            const r = state.results[c.id] || {};
            return {
              id: c.id,
              title: c.title,
              section: c.section_ref,
              status: state.states[c.id] || 'pending',
              explanation: r.explanation || '',
            };
          }),
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `Chat failed (${resp.status})`);
      }
      const data = await resp.json();
      const reply = (data.reply || '').trim() || '(no reply)';
      thinkingNode.textContent = reply;
      thinkingNode.classList.remove('thinking');
      state.history.push({ role: 'assistant', content: reply });
      saveSession();
    } catch (e) {
      thinkingNode.textContent = 'Error: ' + (e.message || 'Chat failed.');
      thinkingNode.classList.remove('thinking');
      thinkingNode.classList.add('error');
    } finally {
      chatbotInput.disabled = false;
      chatbotSend.disabled = false;
      chatbotInput.focus();
      chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }
  }

  function appendMsg(role, content) {
    const empty = chatbotMessages.querySelector('.chatbot-empty');
    if (empty) empty.remove();
    const node = document.createElement('div');
    node.className = `chatbot-msg ${role}`;
    node.textContent = content;
    chatbotMessages.appendChild(node);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    return node;
  }

  // ---------- mobile tabs ----------
  mobileTabs.forEach((tab) => tab.addEventListener('click', () => {
    const pane = tab.dataset.pane;
    mobileTabs.forEach((t) => t.classList.toggle('active', t === tab));
    document.querySelectorAll('.pane').forEach((p) => p.classList.toggle('hidden-mobile', p.dataset.pane !== pane));
  }));

  // ---------- report ----------
  btnReport.addEventListener('click', () => {
    alert("PDF report will be wired up in a follow-up. Your analysis results and answers are preserved in this tab's session.");
  });

  // ---------- utilities ----------
  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function escapeAttr(s) { return escapeHtml(s); }
  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  function statusIconSvg(st, check) {
    switch (st) {
      case 'pass':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
      case 'warning':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>';
      case 'fail':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
      case 'running':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="spin-svg"><path d="M12 2a10 10 0 0 1 10 10"/></svg>';
      case 'input':
      case 'pending':
      default:
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/></svg>';
    }
  }

  // ---------- boot ----------
  if (loadSession()) {
    if (state.mode) {
      setMode(state.mode);
      // setMode loads checklist; results from session will repaint once it returns.
    }
    if (state.doc) {
      workspace.hidden = false;
      explainer.hidden = true;
      previewFilename.textContent = state.doc.filename || '';
      previewSize.textContent = state.doc.size ? `— ${formatBytes(state.doc.size)}` : '';
      previewText.textContent = state.documentText || '(session restored — re-upload to see text)';
      uploadZone.hidden = true;
      pasteZone.hidden = true;
      previewZone.hidden = false;
      analysisPane.hidden = false;
      analysisDisclaimer.hidden = false;
      if (Object.keys(state.results).length) {
        btnReport.disabled = false;
        chatbotInput.disabled = false;
        chatbotSend.disabled = false;
      }
      if (state.history.length) {
        const empty = chatbotMessages.querySelector('.chatbot-empty');
        if (empty) empty.remove();
        state.history.forEach((m) => {
          const node = document.createElement('div');
          node.className = `chatbot-msg ${m.role}`;
          node.textContent = m.content;
          chatbotMessages.appendChild(node);
        });
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
      }
    }
  }
})();
