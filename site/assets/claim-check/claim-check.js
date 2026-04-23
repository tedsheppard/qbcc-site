/* Claim Check — frontend controller.

   Calls the real /api/claim-check endpoints. No mocks.
   - POST /api/claim-check/analyse   (multipart: mode, file OR pasted_text, user_answers)
   - POST /api/claim-check/chat      (JSON: mode, message, history, document_text, checks)

   All state is held in the browser only. Documents never hit storage on the
   server beyond the request lifecycle; chat history lives in sessionStorage
   so closing the tab erases everything. */

(function () {
  'use strict';

  const MAX_BYTES = 10 * 1024 * 1024;

  const MODE_LABELS = {
    payment_claim_outgoing: "Payment claim I'm about to serve",
    payment_claim_incoming: "Payment claim I've received",
    payment_schedule_outgoing: "Payment schedule I'm about to give",
    payment_schedule_incoming: "Payment schedule I've received",
  };

  const SS_KEY = 'sopal.claimCheck.v1';

  const state = {
    mode: null,
    doc: null,            // { kind: 'file'|'paste', filename, size, _file? | text }
    documentText: '',     // server-extracted text, used by chatbot context
    summary: '',
    checks: [],
    history: [],          // [{role, content}, ...]
    userAnswers: {},      // keyed by check id
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
  const analysisLoading = el('analysis-loading');
  const analysisList = el('analysis-list');
  const analysisCount = el('analysis-count');

  const btnReport = el('btn-report');

  const chatbotMessages = el('chatbot-messages');
  const chatbotForm = el('chatbot-form');
  const chatbotInput = el('chatbot-input');
  const chatbotSend = el('chatbot-send');

  const mobileTabs = document.querySelectorAll('.mobile-tab');

  // ---------- session persistence (tab-scoped per spec) ----------
  function saveSession() {
    try {
      const snapshot = {
        mode: state.mode,
        documentText: state.documentText,
        summary: state.summary,
        checks: state.checks,
        history: state.history,
        userAnswers: state.userAnswers,
        docMeta: state.doc ? { filename: state.doc.filename, size: state.doc.size, kind: state.doc.kind } : null,
      };
      sessionStorage.setItem(SS_KEY, JSON.stringify(snapshot));
    } catch (_) { /* quota — ignore */ }
  }
  function loadSession() {
    try {
      const raw = sessionStorage.getItem(SS_KEY);
      if (!raw) return false;
      const s = JSON.parse(raw);
      if (!s || !s.mode || !s.checks) return false;
      state.mode = s.mode;
      state.documentText = s.documentText || '';
      state.summary = s.summary || '';
      state.checks = s.checks || [];
      state.history = s.history || [];
      state.userAnswers = s.userAnswers || {};
      if (s.docMeta) state.doc = { ...s.docMeta };
      return true;
    } catch (_) { return false; }
  }

  // ---------- mode picker ----------
  tiles.forEach((tile) => {
    tile.addEventListener('click', () => setMode(tile.dataset.mode));
  });

  function setMode(mode) {
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
    if (changed && state.doc) runAnalysis();
    saveSession();
  }

  // ---------- upload / drag-drop ----------
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
    if (file.size > MAX_BYTES) { alert('File is too large. Max 10 MB.'); return; }
    const name = (file.name || '').toLowerCase();
    if (!name.endsWith('.pdf') && !name.endsWith('.docx')) {
      alert('Please upload a PDF or DOCX file.');
      return;
    }
    state.doc = { kind: 'file', filename: file.name, size: file.size, _file: file };
    state.userAnswers = {};
    showPreviewShell();
    runAnalysis();
  }

  // ---------- paste ----------
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
    state.doc = {
      kind: 'paste',
      filename: 'Pasted text',
      size: new Blob([text]).size,
      text,
    };
    state.documentText = text; // pasted text is already extracted
    state.userAnswers = {};
    showPreviewShell();
    runAnalysis();
  });

  // ---------- preview ----------
  function showPreviewShell() {
    previewFilename.textContent = state.doc.filename;
    previewSize.textContent = state.doc.size ? `— ${formatBytes(state.doc.size)}` : '';
    if (state.doc.kind === 'paste' && state.doc.text) {
      previewText.textContent = state.doc.text;
    } else {
      previewText.textContent = 'Extracting text…';
    }
    uploadZone.hidden = true;
    pasteZone.hidden = true;
    previewZone.hidden = false;
    analysisPane.hidden = false;
    chatbotInput.placeholder = 'Ask a question about this document…';
  }

  btnChangeDoc.addEventListener('click', () => {
    resetDocState();
    sessionStorage.removeItem(SS_KEY);
  });

  function resetDocState() {
    state.doc = null;
    state.documentText = '';
    state.summary = '';
    state.checks = [];
    state.history = [];
    state.userAnswers = {};
    previewZone.hidden = true;
    analysisPane.hidden = true;
    uploadZone.hidden = false;
    pasteInput.value = '';
    pasteMeta.textContent = '0 characters';
    pasteSubmit.disabled = true;
    chatbotInput.placeholder = 'Upload a document to start the conversation…';
    chatbotInput.disabled = true;
    chatbotSend.disabled = true;
    btnReport.disabled = true;
    analysisList.innerHTML = '';
    analysisList.hidden = true;
    analysisEmpty.hidden = false;
    analysisLoading.hidden = true;
    analysisCount.hidden = true;
    chatbotMessages.innerHTML = '<div class="chatbot-empty"><p>Upload a document above, then ask questions like <em>"why did you flag the reference date?"</em> or <em>"what defences could I raise under s 69?"</em></p></div>';
  }

  // ---------- analysis ----------
  async function runAnalysis() {
    if (!state.mode || !state.doc) return;
    analysisEmpty.hidden = true;
    analysisList.hidden = true;
    analysisLoading.hidden = false;
    analysisCount.hidden = true;
    btnReport.disabled = true;
    chatbotInput.disabled = true;
    chatbotSend.disabled = true;

    try {
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

      const resp = await fetch('/api/claim-check/analyse', { method: 'POST', body: form });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `Analysis failed (${resp.status})`);
      }
      const data = await resp.json();

      state.documentText = data.document_text || state.documentText || '';
      state.summary = data.summary || '';
      state.checks = Array.isArray(data.checks) ? data.checks : [];

      // Update preview for file uploads now that we have extracted text.
      if (state.doc.kind === 'file') {
        previewText.textContent = state.documentText || '(no text extracted)';
      }

      renderChecks();
      btnReport.disabled = false;
      chatbotInput.disabled = false;
      chatbotSend.disabled = false;
      saveSession();
    } catch (e) {
      analysisLoading.hidden = true;
      analysisList.hidden = false;
      analysisList.innerHTML = `<li class="analysis-item" style="border-color:var(--red-mid);background:var(--red-soft);color:var(--red);font-size:13px;">${escapeHtml(e.message || 'Analysis failed.')}</li>`;
    }
  }

  function renderChecks() {
    analysisLoading.hidden = true;
    if (!state.checks.length) {
      analysisList.innerHTML = '<li class="analysis-item" style="color:var(--text-muted);font-size:13px;">No checks returned.</li>';
      analysisList.hidden = false;
      return;
    }
    const summaryHtml = state.summary
      ? `<div class="analysis-summary">${escapeHtml(state.summary)}</div>`
      : '';
    analysisList.innerHTML = summaryHtml + state.checks.map((c, i) => renderCheckHtml(c, i)).join('');
    analysisList.hidden = false;
    analysisCount.hidden = false;
    analysisCount.textContent = `${state.checks.length} checks`;
    wireCheckInputs();
  }

  function renderCheckHtml(check, index) {
    const icon = statusIconSvg(check.status);
    const q = check.query || check.title || '';
    const searchHref = `/search?q=${encodeURIComponent(q)}`;
    const explanationHtml = check.explanation
      ? `<div class="check-explanation">${escapeHtml(check.explanation)}</div>`
      : '';
    const quoteHtml = check.quote
      ? `<div class="check-quote">“${escapeHtml(check.quote)}”</div>`
      : '';
    const inputHtml = check.status === 'input' && check.prompt
      ? renderInlineInputHtml(check, index)
      : '';
    return `<li class="analysis-item" data-check-id="${escapeAttr(check.id || ('c' + index))}">
      <div class="check-row">
        <span class="check-icon" data-status="${escapeAttr(check.status)}">${icon}</span>
        <div class="check-body">
          <div class="check-title">${escapeHtml(check.title || 'Untitled check')}</div>
          <div class="check-meta">
            ${check.section ? `<span class="check-section">${escapeHtml(check.section)}</span>` : ''}
            <a class="check-link" href="${searchHref}" target="_blank" rel="noopener">See relevant decisions →</a>
          </div>
          ${explanationHtml}
          ${quoteHtml}
          ${inputHtml}
        </div>
      </div>
    </li>`;
  }

  function renderInlineInputHtml(check, index) {
    const inputId = `chk-in-${index}`;
    let control = '';
    const stored = state.userAnswers[check.id || ''] || '';
    const safeStored = escapeAttr(stored);
    if (check.input_type === 'date') {
      control = `<input type="date" id="${inputId}" value="${safeStored}">`;
    } else if (check.input_type === 'yes-no') {
      control = `<select id="${inputId}">
        <option value="">Select…</option>
        <option ${stored === 'Yes' ? 'selected' : ''}>Yes</option>
        <option ${stored === 'No' ? 'selected' : ''}>No</option>
      </select>`;
    } else {
      control = `<input type="text" id="${inputId}" value="${safeStored}" placeholder="Your answer">`;
    }
    return `<div class="check-input" data-check-id="${escapeAttr(check.id || ('c' + index))}">
      <label for="${inputId}">${escapeHtml(check.prompt)}</label>
      ${control}
      <button type="button" class="btn-apply-answer" data-for="${inputId}">Apply</button>
    </div>`;
  }

  function wireCheckInputs() {
    document.querySelectorAll('.btn-apply-answer').forEach((btn) => {
      btn.addEventListener('click', () => {
        const forId = btn.dataset.for;
        const input = document.getElementById(forId);
        if (!input) return;
        const li = btn.closest('.analysis-item');
        const checkId = li ? li.dataset.checkId : null;
        if (!checkId) return;
        state.userAnswers[checkId] = input.value;
        saveSession();
        runAnalysis();
      });
    });
  }

  // ---------- chatbot ----------
  chatbotForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendChat();
  });

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
          checks: state.checks,
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
    // Remove empty-state on first real message.
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
  mobileTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const pane = tab.dataset.pane;
      mobileTabs.forEach((t) => t.classList.toggle('active', t === tab));
      document.querySelectorAll('.pane').forEach((p) => {
        p.classList.toggle('hidden-mobile', p.dataset.pane !== pane);
      });
    });
  });

  // ---------- report (placeholder) ----------
  btnReport.addEventListener('click', async () => {
    alert("PDF report will be wired up in a follow-up. Analysis results and your answers are already preserved in this tab's session — you won't lose them.");
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
  function statusIconSvg(status) {
    switch (status) {
      case 'pass':    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
      case 'warning': return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>';
      case 'fail':    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
      case 'input':
      default:        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/></svg>';
    }
  }

  // ---------- boot ----------
  if (loadSession()) {
    // Restore UI from previous session.
    if (state.mode) setMode(state.mode);
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
      if (state.checks.length) {
        renderChecks();
        btnReport.disabled = false;
        chatbotInput.disabled = false;
        chatbotSend.disabled = false;
      }
      // Restore chat history.
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
