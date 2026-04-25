/* Contract Assist — frontend controller.
 *
 * Wires the contract upload, viewer, chat composer, SSE consumer, and
 * source-pill rendering. Backend endpoints used:
 *   POST /api/contract-assist/ingest         (multipart: file)
 *   POST /api/contract-assist/chat           (JSON; SSE response)
 *   POST /api/contract-assist/draft-export   (JSON; DOCX bytes)
 *
 * Session state lives in localStorage (mirrors the Claim Assist pattern):
 *   key prefix: sopal-contract-assist-session-<id>
 *   stores: filename, page count, identified form, chat history, last activity.
 *   does NOT store: the contract file bytes or retrieved chunks.
 */

(function () {
  'use strict';

  // ---------- constants ----------
  const MAX_CONTRACT_BYTES = 25 * 1024 * 1024; // 25 MB per spec
  const MAX_INPUT_CHARS = 4000;
  const MIN_THINKING_DISPLAY_MS = 400;
  const LS_PREFIX = 'sopal-contract-assist-session-';
  const SS_KEY = 'sopal.contractAssist.v1';
  const HANDOFF_KEY = 'sopal.assistHandoff';

  // ---------- state ----------
  const state = {
    sessionId: null,
    contract: null,            // { filename, size, page_count, identified_form, type }
    contractBytesArmed: false, // true once a viewer has been mounted with bytes this session
    history: [],               // [{role, content, sources?, draft?}]
    streaming: false,
    pendingAttachments: [],    // [{ name, size, type, b64 }]
  };

  // ---------- element refs ----------
  const el = (id) => document.getElementById(id);
  const workspace = el('ca-workspace');
  const mobileTabs = document.querySelectorAll('.ca-mobile-tab');
  const docPane = document.querySelector('.ca-pane-document');
  const chatPane = document.querySelector('.ca-pane-chat');
  const contractTitle = el('contract-title');
  const contractMeta = el('contract-meta');
  const contractEmpty = el('contract-empty');
  const contractViewer = el('contract-viewer');
  const btnChangeContract = el('btn-change-contract');

  const contractCard = el('contract-card');
  const contractCardName = el('contract-card-name');
  const contractCardMeta = el('contract-card-meta');
  const contractCardForm = el('contract-card-form');

  const chatEmpty = el('ca-chat-empty');
  const uploadZone = el('ca-upload-zone');
  const uploadInput = el('ca-upload-input');
  const thread = el('ca-thread');
  const composer = el('ca-composer');
  const inputEl = el('ca-input');
  const countEl = el('ca-count');
  const sendBtn = el('ca-send');
  const attachBtn = el('ca-attach-btn');
  const attachInput = el('ca-attach-input');
  const attachTray = el('ca-attach-tray');
  const btnSwitchProduct = el('btn-switch-product');

  // ---------- session helpers ----------
  function newSessionId() {
    return 'cs_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
  }

  function ensureSessionId() {
    if (!state.sessionId) state.sessionId = newSessionId();
    return state.sessionId;
  }

  function lsAvailable() {
    try {
      const k = '__sopal_ca_probe';
      localStorage.setItem(k, '1');
      localStorage.removeItem(k);
      return true;
    } catch (_) { return false; }
  }

  function saveSession() {
    if (!state.sessionId || !state.contract) return;
    try {
      sessionStorage.setItem(SS_KEY, JSON.stringify({
        sessionId: state.sessionId,
        contract: state.contract,
        history: state.history,
      }));
    } catch (_) {}
    if (!lsAvailable()) return;
    try {
      localStorage.setItem(LS_PREFIX + state.sessionId, JSON.stringify({
        id: state.sessionId,
        savedAt: Date.now(),
        contract: state.contract,
        contractFilename: state.contract.filename,
        history: state.history.map(m => ({
          role: m.role,
          content: m.content,
          // Persist source metadata but not full chunk text.
          sources: (m.sources || []).map(s => ({
            type: s.type,
            tag: s.tag,
            heading: s.heading,
            excerpt: s.excerpt,
            anchor_url: s.anchor_url,
          })),
          ts: m.ts,
        })),
      }));
      enforceSessionCap();
    } catch (_) {}
  }

  function enforceSessionCap() {
    // Combined cap of 20 across both products per spec.
    const all = [];
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (!k) continue;
        if (k.startsWith(LS_PREFIX) || k.startsWith('sopal-claim-check-session-')) {
          try {
            const s = JSON.parse(localStorage.getItem(k));
            all.push({ key: k, ts: s.savedAt || 0 });
          } catch (_) {}
        }
      }
    } catch (_) { return; }
    all.sort((a, b) => b.ts - a.ts);
    for (let i = 20; i < all.length; i++) {
      try { localStorage.removeItem(all[i].key); } catch (_) {}
    }
  }

  function loadSession() {
    // 1) Resume key set by /assist landing? — restore that one.
    let resumeKey = null;
    try { resumeKey = sessionStorage.getItem('sopal.resumeKey'); } catch (_) {}
    if (resumeKey && resumeKey.startsWith(LS_PREFIX)) {
      try {
        const raw = localStorage.getItem(resumeKey);
        if (raw) {
          const s = JSON.parse(raw);
          state.sessionId = s.id || newSessionId();
          state.contract = s.contract || null;
          state.history = s.history || [];
          try { sessionStorage.removeItem('sopal.resumeKey'); } catch (_) {}
          return true;
        }
      } catch (_) {}
    }
    // 2) Tab-scoped session (this is the same tab returning).
    try {
      const raw = sessionStorage.getItem(SS_KEY);
      if (raw) {
        const s = JSON.parse(raw);
        state.sessionId = s.sessionId || newSessionId();
        state.contract = s.contract || null;
        state.history = s.history || [];
        return true;
      }
    } catch (_) {}
    return false;
  }

  function deleteSessionStorage() {
    try { sessionStorage.removeItem(SS_KEY); } catch (_) {}
  }

  // ---------- upload ----------
  uploadZone.addEventListener('click', () => uploadInput.click());
  uploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); uploadInput.click(); }
  });
  ['dragenter', 'dragover'].forEach((evt) =>
    uploadZone.addEventListener(evt, (e) => { e.preventDefault(); uploadZone.classList.add('dragging'); })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    uploadZone.addEventListener(evt, (e) => { e.preventDefault(); uploadZone.classList.remove('dragging'); })
  );
  uploadZone.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleContractFile(f);
  });
  uploadInput.addEventListener('change', () => {
    const f = uploadInput.files && uploadInput.files[0];
    if (f) handleContractFile(f);
  });

  btnChangeContract.addEventListener('click', async () => {
    if (window.ClaimCheckModal) {
      const ok = await window.ClaimCheckModal.confirm(
        'Change contract?',
        'Replacing the contract will clear the chat history for this session. Your saved sessions remain available on the Sopal Assist landing page.',
        { confirmLabel: 'Change contract', confirmVariant: 'primary' }
      );
      if (!ok) return;
    }
    resetContractState();
  });

  function handleContractFile(file) {
    if (!file) return;
    if (file.size > MAX_CONTRACT_BYTES) {
      modalError('Contract too large', 'The maximum contract size is 25 MB.');
      return;
    }
    const lower = (file.name || '').toLowerCase();
    if (!lower.endsWith('.pdf') && !lower.endsWith('.docx')) {
      modalError('Unsupported file type', 'Please upload a PDF or DOCX contract.');
      return;
    }
    ingestContract(file);
  }

  async function ingestContract(file) {
    ensureSessionId();
    const progress = window.ClaimCheckModal
      ? window.ClaimCheckModal.progress('Ingesting your contract…', 'Extracting text, identifying clause structure, and building a private session-scoped index.')
      : null;
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('session_id', state.sessionId);
      const resp = await fetch('/api/contract-assist/ingest', { method: 'POST', body: fd });
      if (!resp.ok) {
        const detail = await readErrorDetail(resp);
        throw new Error(detail || `Ingestion failed (${resp.status})`);
      }
      const data = await resp.json();
      state.contract = {
        filename: data.filename || file.name,
        size: data.source_size || file.size,
        page_count: data.page_count || null,
        identified_form: data.identified_form || null,
        chunk_count: data.chunk_count || 0,
        type: lowerExt(file.name),
      };
      state.contractBytesArmed = false; // viewer still needs the bytes — provide via mountViewer below.
      paintContractLoaded();
      await mountViewer(file);
      if (progress) progress.close();
      saveSession();
    } catch (e) {
      if (progress) progress.close();
      modalError('Ingestion failed', e.message || 'Please try again. The contract was not stored on our servers.');
    }
  }

  async function readErrorDetail(resp) {
    try { const j = await resp.clone().json(); return j && j.detail ? j.detail : ''; }
    catch (_) { try { return await resp.text(); } catch (_) { return ''; } }
  }

  function lowerExt(name) {
    const n = (name || '').toLowerCase();
    if (n.endsWith('.pdf')) return 'pdf';
    if (n.endsWith('.docx')) return 'docx';
    return 'unknown';
  }

  async function mountViewer(file) {
    if (!window.ClaimCheckViewers) return;
    contractEmpty.hidden = true;
    contractViewer.hidden = false;
    try {
      await window.ClaimCheckViewers.render(contractViewer, file);
      state.contractBytesArmed = true;
    } catch (e) {
      console.error('viewer mount failed', e);
    }
  }

  function paintContractLoaded() {
    chatEmpty.hidden = true;
    thread.hidden = false;
    contractCard.hidden = false;
    contractCardName.textContent = state.contract.filename;
    const pages = state.contract.page_count;
    contractCardMeta.textContent = pages ? `${pages} ${pages === 1 ? 'page' : 'pages'}` : 'Loaded';
    if (state.contract.identified_form) {
      contractCardForm.hidden = false;
      contractCardForm.textContent = `Appears to be ${state.contract.identified_form}`;
    } else {
      contractCardForm.hidden = true;
    }
    contractTitle.textContent = state.contract.filename;
    contractMeta.textContent = pages ? `${pages} ${pages === 1 ? 'page' : 'pages'}` : '';
    btnChangeContract.hidden = false;
    inputEl.disabled = false;
    inputEl.placeholder = 'Ask anything about your contract — clauses, timing, payment obligations, notice requirements…';
    sendBtn.disabled = false;
    attachBtn.disabled = false;
    // If we have prior history (e.g., resumed session), repaint it.
    if (state.history.length) repaintThread();
  }

  function resetContractState() {
    state.contract = null;
    state.history = [];
    state.contractBytesArmed = false;
    state.pendingAttachments = [];
    contractCard.hidden = true;
    contractEmpty.hidden = false;
    contractViewer.hidden = true;
    if (window.ClaimCheckViewers && contractViewer) {
      try { window.ClaimCheckViewers.destroy(contractViewer); } catch (_) {}
    }
    btnChangeContract.hidden = true;
    contractTitle.textContent = 'Contract';
    contractMeta.textContent = '';
    inputEl.value = '';
    inputEl.disabled = true;
    inputEl.placeholder = 'Upload a contract above to start asking questions…';
    sendBtn.disabled = true;
    attachBtn.disabled = true;
    thread.innerHTML = '';
    thread.hidden = true;
    chatEmpty.hidden = false;
    uploadInput.value = '';
    paintAttachTray();
    deleteSessionStorage();
  }

  // ---------- chat composer ----------
  inputEl.addEventListener('input', () => {
    const len = inputEl.value.length;
    countEl.textContent = `${len.toLocaleString()} / ${MAX_INPUT_CHARS}`;
    countEl.classList.toggle('over', len > MAX_INPUT_CHARS);
    // Auto-grow.
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
  });

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      composer.requestSubmit();
    }
  });

  composer.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (state.streaming) return;
    const text = inputEl.value.trim();
    if (!text) return;
    if (text.length > MAX_INPUT_CHARS) {
      modalError('Message too long', `Please trim your message to ${MAX_INPUT_CHARS} characters or fewer.`);
      return;
    }
    if (!state.contract) {
      modalError('No contract loaded', 'Upload a contract first so I can ground my answer in your specific document.');
      return;
    }
    sendChat(text);
  });

  attachBtn.addEventListener('click', () => attachInput.click());
  attachInput.addEventListener('change', async () => {
    const f = attachInput.files && attachInput.files[0];
    attachInput.value = '';
    if (!f) return;
    if (f.size > 5 * 1024 * 1024) {
      modalError('Attachment too large', 'Inline attachments must be 5 MB or smaller. They are sent for this turn only and not stored.');
      return;
    }
    try {
      const b64 = await fileToBase64(f);
      state.pendingAttachments.push({ name: f.name, size: f.size, type: f.type || 'application/octet-stream', b64 });
      paintAttachTray();
    } catch (e) {
      modalError('Could not attach file', e.message || 'Please try a different file.');
    }
  });

  function paintAttachTray() {
    attachTray.innerHTML = '';
    if (!state.pendingAttachments.length) {
      attachTray.hidden = true;
      return;
    }
    attachTray.hidden = false;
    state.pendingAttachments.forEach((a, idx) => {
      const chip = document.createElement('span');
      chip.className = 'ca-attach-chip';
      chip.innerHTML = `${escapeHtml(a.name)} <button type="button" class="ca-attach-chip-remove" data-idx="${idx}" aria-label="Remove attachment">×</button>`;
      attachTray.appendChild(chip);
    });
    attachTray.querySelectorAll('.ca-attach-chip-remove').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.idx, 10);
        state.pendingAttachments.splice(idx, 1);
        paintAttachTray();
      });
    });
  }

  // ---------- send + SSE ----------
  async function sendChat(text) {
    state.streaming = true;
    sendBtn.disabled = true;
    attachBtn.disabled = true;

    const attachments = state.pendingAttachments.slice();
    state.pendingAttachments = [];
    paintAttachTray();

    const userMsg = { role: 'user', content: text, ts: Date.now(), attachments: attachments.map(a => ({ name: a.name, size: a.size })) };
    state.history.push(userMsg);
    appendMessage(userMsg);

    inputEl.value = '';
    inputEl.style.height = 'auto';
    countEl.textContent = `0 / ${MAX_INPUT_CHARS}`;

    // Bot placeholder bubble — starts as a thinking shimmer.
    const botMsg = { role: 'assistant', content: '', sources: [], ts: Date.now() };
    state.history.push(botMsg);
    const botNode = appendMessage(botMsg);
    setThinkingState(botNode, 'Reading your question…');

    try {
      const resp = await fetch('/api/contract-assist/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: ensureSessionId(),
          message: text,
          history: state.history.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
          attachments,
          contract_meta: state.contract,
        }),
      });
      if (!resp.ok) {
        const detail = await readErrorDetail(resp);
        const err = new Error(detail || `Chat failed (${resp.status})`);
        err.status = resp.status;
        throw err;
      }
      if (!resp.body) throw new Error('Streaming response body unavailable.');
      await consumeSSE(resp.body, (event, data) => handleSse(event, data, botMsg, botNode));
    } catch (e) {
      // Remove the bot placeholder; show modal with retry.
      removeNode(botNode);
      state.history.pop();
      const detail = e && e.message;
      let title = "I couldn't generate a response just then.";
      let body = 'Please try again.';
      if (e && e.status === 429) { title = 'Rate limit reached'; body = detail || 'Try again later.'; }
      else if (e && e.status >= 500) { title = "Couldn't reach the assistant"; body = 'The server had trouble responding. Please try again.'; }
      else if (detail) { body = detail; }
      modalRetry(title, body, () => { state.history.pop(); inputEl.value = text; sendChat(text); });
    } finally {
      state.streaming = false;
      sendBtn.disabled = !state.contract;
      attachBtn.disabled = !state.contract;
      inputEl.focus();
      saveSession();
    }
  }

  function handleSse(event, data, botMsg, botNode) {
    if (event === 'thinking') {
      const msg = (data && data.message) || '';
      if (msg) setThinkingState(botNode, msg);
    } else if (event === 'content') {
      const piece = (data && data.delta) || '';
      botMsg.content += piece;
      paintBotContent(botNode, botMsg);
    } else if (event === 'sources') {
      const arr = (data && data.sources) || [];
      botMsg.sources = arr;
      paintBotContent(botNode, botMsg); // re-render to apply pills
      paintSourcesList(botNode, botMsg);
    } else if (event === 'draft') {
      botMsg.draft = data || null;
      // Render a download-DOCX chip below the message.
      paintDraftChip(botNode, botMsg);
    } else if (event === 'done') {
      // Final pass.
      paintBotContent(botNode, botMsg);
      paintSourcesList(botNode, botMsg);
    } else if (event === 'error') {
      const m = (data && data.message) || 'Stream error.';
      botMsg.content = (botMsg.content ? botMsg.content + '\n\n' : '') + '_Error: ' + m + '_';
      paintBotContent(botNode, botMsg);
    }
  }

  // ---------- message rendering ----------
  function appendMessage(msg) {
    if (chatEmpty && !chatEmpty.hidden) chatEmpty.hidden = true;
    if (thread.hidden) thread.hidden = false;
    const node = document.createElement('div');
    node.className = `ca-msg ${msg.role}`;
    const bubble = document.createElement('div');
    bubble.className = 'ca-msg-bubble';
    if (msg.role === 'user') {
      bubble.textContent = msg.content;
    } else {
      bubble.innerHTML = renderMarkdownToBubbleHtml(msg.content || '', msg.sources || []);
    }
    node.appendChild(bubble);
    if (msg.role === 'assistant' && msg.sources && msg.sources.length) {
      const srcWrap = document.createElement('div');
      srcWrap.className = 'ca-msg-sources';
      node._srcWrap = srcWrap;
      node.appendChild(srcWrap);
    }
    thread.appendChild(node);
    scrollThread();
    return node;
  }

  function paintBotContent(node, msg) {
    if (node.classList.contains('thinking')) node.classList.remove('thinking');
    const bubble = node.querySelector('.ca-msg-bubble');
    if (!bubble) return;
    bubble.innerHTML = renderMarkdownToBubbleHtml(msg.content || '', msg.sources || []);
    bubble.querySelectorAll('.ca-source-pill').forEach((pill) => {
      pill.addEventListener('click', () => onSourcePillClick(pill, msg.sources || []));
    });
    scrollThread();
  }

  function paintSourcesList(node, msg) {
    let wrap = node._srcWrap;
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.className = 'ca-msg-sources';
      node._srcWrap = wrap;
      node.appendChild(wrap);
    }
    if (!msg.sources || !msg.sources.length) {
      wrap.innerHTML = '';
      return;
    }
    wrap.innerHTML = '<div class="ca-msg-sources-label">Sources</div>' +
      msg.sources.slice(0, 6).map((s, i) => sourceRowHtml(s, i)).join('');
    wrap.querySelectorAll('.ca-msg-source-row').forEach((row, i) => {
      row.addEventListener('click', () => navigateToSource(msg.sources[i]));
    });
  }

  function paintDraftChip(node, msg) {
    if (!msg.draft) return;
    let chip = node.querySelector('.ca-draft-chip');
    if (chip) chip.remove();
    chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'ca-draft-chip';
    chip.textContent = `Download draft (${msg.draft.kind || 'document'}) as DOCX`;
    chip.style.cssText = 'margin-top:8px;background:var(--green);color:#000;font-weight:600;font-size:12px;padding:6px 12px;border-radius:8px;border:none;cursor:pointer;font-family:inherit;';
    chip.addEventListener('click', () => downloadDraft(msg.draft));
    node.appendChild(chip);
  }

  async function downloadDraft(draft) {
    const progress = window.ClaimCheckModal && window.ClaimCheckModal.progress('Building your DOCX…', 'Watermarking the draft and rendering for download.');
    try {
      const resp = await fetch('/api/contract-assist/draft-export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId, draft }),
      });
      if (!resp.ok) {
        const detail = await readErrorDetail(resp);
        throw new Error(detail || `Export failed (${resp.status})`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (draft.kind || 'draft') + '.docx';
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
      if (progress) progress.close();
    } catch (e) {
      if (progress) progress.close();
      modalError('Could not export draft', e.message || 'Please try again.');
    }
  }

  function setThinkingState(node, msg) {
    node.classList.add('thinking');
    let bubble = node.querySelector('.ca-msg-bubble');
    if (!bubble) {
      bubble = document.createElement('div');
      bubble.className = 'ca-msg-bubble';
      node.appendChild(bubble);
    }
    let span = bubble.querySelector('.ca-thinking-text');
    if (!span) {
      bubble.innerHTML = '';
      span = document.createElement('span');
      span.className = 'ca-thinking-text';
      span.textContent = msg;
      bubble.appendChild(span);
    } else {
      // Fade swap.
      span.classList.add('fading');
      setTimeout(() => {
        span.textContent = msg;
        span.classList.remove('fading');
      }, MIN_THINKING_DISPLAY_MS / 3);
    }
    scrollThread();
  }

  function repaintThread() {
    thread.innerHTML = '';
    state.history.forEach((m) => {
      const node = appendMessage(m);
      if (m.role === 'assistant' && m.sources && m.sources.length) paintSourcesList(node, m);
      if (m.role === 'assistant' && m.draft) paintDraftChip(node, m);
    });
  }

  function removeNode(n) { if (n && n.parentNode) n.parentNode.removeChild(n); }
  function scrollThread() { thread.scrollTop = thread.scrollHeight; }

  // ---------- markdown + source pills ----------
  function renderMarkdownToBubbleHtml(text, sources) {
    let raw = text || '';
    if (!raw) return '';
    let html;
    if (window.marked && window.DOMPurify) {
      window.marked.setOptions({ breaks: true, gfm: true });
      const dirty = window.marked.parse(raw);
      html = window.DOMPurify.sanitize(dirty, {
        ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'h3', 'h4'],
        ALLOWED_ATTR: ['href', 'target', 'rel'],
      });
    } else {
      html = escapeHtml(raw).replace(/\n/g, '<br>');
    }
    return injectSourcePills(html, sources || []);
  }

  // Replace [clause X.Y] / [s N BIF Act] markers in the rendered HTML with
  // clickable source pills. Walks text nodes to avoid touching attributes/code.
  function injectSourcePills(html, sources) {
    // Build a quick lookup by tag.
    const tagToSource = new Map();
    sources.forEach((s) => {
      if (s && s.tag) tagToSource.set(String(s.tag).toLowerCase(), s);
    });

    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    const re = /\[(?:(?:clause|cl\.?)\s+(\d+(?:\.\d+)*[a-z]?)|s\s+(\d+(?:\(\d+\))*(?:\([a-z]\))?(?:\([ivx]+\))?)\s+(?:BIF|QBCC)(?:\s+Act)?)\]/gi;

    walkText(tmp, (textNode) => {
      const t = textNode.nodeValue || '';
      if (!re.test(t)) return;
      re.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0;
      let m;
      while ((m = re.exec(t)) !== null) {
        if (m.index > last) frag.appendChild(document.createTextNode(t.slice(last, m.index)));
        const isContract = !!m[1];
        const label = isContract ? `clause ${m[1]}` : m[0].slice(1, -1).trim();
        const tag = label.toLowerCase();
        const src = tagToSource.get(tag) || null;
        const pill = document.createElement('button');
        pill.type = 'button';
        pill.className = isContract ? 'ca-source-pill' : 'ca-source-pill bif';
        pill.textContent = label;
        if (src) {
          pill.title = src.heading || src.excerpt || label;
          pill.dataset.tag = tag;
        }
        frag.appendChild(pill);
        last = re.lastIndex;
      }
      if (last < t.length) frag.appendChild(document.createTextNode(t.slice(last)));
      textNode.parentNode.replaceChild(frag, textNode);
    });

    return tmp.innerHTML;
  }

  function walkText(node, fn) {
    if (!node) return;
    if (node.nodeType === 3) { fn(node); return; }
    if (node.nodeType === 1) {
      const tag = node.nodeName;
      if (tag === 'CODE' || tag === 'PRE' || tag === 'A') return; // don't pill inside code/links
      const kids = Array.from(node.childNodes);
      kids.forEach((k) => walkText(k, fn));
    }
  }

  function sourceRowHtml(s, i) {
    const isBif = s.type === 'bif_act' || (s.tag || '').startsWith('s ');
    return `<div class="ca-msg-source-row" data-idx="${i}">
      <span class="ca-msg-source-tag ${isBif ? 'bif' : ''}">${escapeHtml(s.tag || (isBif ? 'BIF Act' : 'Contract'))}</span>
      <div class="ca-msg-source-text">
        ${s.heading ? `<div class="ca-msg-source-heading">${escapeHtml(s.heading)}</div>` : ''}
        <div>${escapeHtml((s.excerpt || '').slice(0, 240))}</div>
      </div>
    </div>`;
  }

  function onSourcePillClick(pill, sources) {
    const tag = pill.dataset.tag;
    const src = sources.find(s => (s.tag || '').toLowerCase() === tag);
    if (!src) return;
    navigateToSource(src);
  }

  function navigateToSource(src) {
    if (!src) return;
    if (src.type === 'bif_act' && src.anchor_url) {
      window.open(src.anchor_url, '_blank', 'noopener');
      return;
    }
    if (src.page_number) {
      // Tell the PDF viewer to go to that page if it's loaded.
      const pageInput = contractViewer.querySelector('.pdfv-page-input');
      if (pageInput) {
        pageInput.value = String(src.page_number);
        pageInput.dispatchEvent(new Event('change'));
      }
      return;
    }
    // Fall back: show the full text in a modal.
    if (window.ClaimCheckModal) {
      window.ClaimCheckModal.open({
        title: src.heading || src.tag || 'Source',
        bodyHtml: `<div style="white-space:pre-wrap;font-size:13px;line-height:1.55;">${escapeHtml(src.full_text || src.excerpt || '')}</div>`,
        kind: 'info',
        actions: [{ label: 'Close', variant: 'primary' }],
      });
    }
  }

  // ---------- mobile tabs ----------
  mobileTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.pane;
      mobileTabs.forEach((t) => t.classList.toggle('active', t === tab));
      [docPane, chatPane].forEach((p) => {
        if (!p) return;
        p.classList.toggle('hidden-mobile', p.dataset.pane !== target);
      });
    });
  });

  // ---------- cross-product switcher ----------
  if (btnSwitchProduct) {
    btnSwitchProduct.addEventListener('click', async () => {
      const target = btnSwitchProduct.dataset.target || '/assist/claim';
      const targetName = btnSwitchProduct.dataset.targetName || 'Claim Assist';
      const hasWork = !!(state.contract || state.history.length || (inputEl.value || '').trim());
      if (hasWork && window.ClaimCheckModal) {
        const ok = await window.ClaimCheckModal.confirm(
          'Switch products?',
          `You have an active session in Contract Assist. Continue to ${targetName}? Your work will be saved and you can return any time.`,
          { confirmLabel: `Open ${targetName}`, confirmVariant: 'primary' }
        );
        if (!ok) return;
      }
      saveSession();
      window.location.href = target;
    });
  }

  // ---------- handoff from /assist landing ----------
  function consumeHandoff() {
    let raw = null;
    try { raw = sessionStorage.getItem(HANDOFF_KEY); } catch (_) { return null; }
    if (!raw) return null;
    try {
      const obj = JSON.parse(raw);
      try { sessionStorage.removeItem(HANDOFF_KEY); } catch (_) {}
      if (!obj || obj.target !== '/assist/contract' || !obj.b64 || !obj.filename) return null;
      // Reconstruct File from base64.
      const bin = atob(obj.b64);
      const len = bin.length;
      const arr = new Uint8Array(len);
      for (let i = 0; i < len; i++) arr[i] = bin.charCodeAt(i);
      return new File([arr], obj.filename, { type: obj.type || 'application/octet-stream' });
    } catch (_) { return null; }
  }

  // ---------- utilities ----------
  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => {
        try {
          const arr = new Uint8Array(r.result);
          const chunk = 0x8000;
          let bin = '';
          for (let i = 0; i < arr.length; i += chunk) bin += String.fromCharCode.apply(null, arr.subarray(i, i + chunk));
          resolve(btoa(bin));
        } catch (e) { reject(e); }
      };
      r.onerror = () => reject(new Error('Could not read file'));
      r.readAsArrayBuffer(file);
    });
  }
  function modalError(title, body) {
    if (window.ClaimCheckModal) window.ClaimCheckModal.error(title, body);
    else console.error(title, body);
  }
  function modalRetry(title, body, retryFn) {
    if (!window.ClaimCheckModal) { console.error(title, body); return; }
    window.ClaimCheckModal.open({
      title, body, kind: 'error',
      actions: [
        { label: 'Close', variant: 'default' },
        { label: 'Try again', variant: 'primary', onClick: () => { try { retryFn(); } catch (_) {} } },
      ],
    });
  }

  // ---------- SSE ----------
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
    } catch (_) { return { event, data: {} }; }
  }

  // ---------- boot ----------
  ensureSessionId();

  // Restore an existing session (resume from landing OR same-tab continuation).
  if (loadSession() && state.contract) {
    paintContractLoaded();
    // Viewer requires the contract bytes which are not stored. The user will
    // see a banner indicating they should re-upload to enable viewer + RAG.
    contractEmpty.hidden = false;
    contractViewer.hidden = true;
    if (window.ClaimCheckModal) {
      window.ClaimCheckModal.open({
        title: 'Session resumed',
        body: 'Your chat history is restored. The contract file is not stored in your browser — re-upload the same contract to re-enable retrieval and the in-page viewer. You can still scroll the previous conversation.',
        kind: 'info',
        actions: [{ label: 'Got it', variant: 'primary' }],
      });
    }
  }

  // Handoff from /assist landing dropzone.
  const handoffFile = consumeHandoff();
  if (handoffFile) handleContractFile(handoffFile);

  // Initial composer disabled state.
  if (!state.contract) {
    inputEl.disabled = true;
    sendBtn.disabled = true;
    attachBtn.disabled = true;
  }
})();
