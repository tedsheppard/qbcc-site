/* SopalAI — Construction Law Research frontend
 * API mounted at /ai/api/*. Auth: JWT (purchase_token in localStorage)
 * for signed users; X-Anon-ID header (UUID in localStorage) for anonymous.
 */
(() => {
  // The API is mounted under /ai when running inside the main sopal.com.au
  // server. For local dev (uvicorn services.bif_research.api:app on port 8000)
  // we fall back to same-origin.
  // When served from sopal.com.au, the bif_research app is mounted at
  // /ai so all API paths get a /ai prefix. For local dev
  // (uvicorn services.bif_research.api:app) the app sits at the root.
  const MOUNTED = location.pathname.startsWith("/ai");
  const API = MOUNTED ? "/ai" : "";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const els = {
    welcome: $("#welcome"),
    convStream: $("#conv-stream"),
    convContainer: $("#conv-container"),
    input: $("#question-input"),
    askBtn: $("#ask-btn"),
    newChatBtn: $("#new-chat-btn"),
    recents: $("#recents-list"),
    statusPill: $("#status-pill"),
    usagePill: $("#usage-pill"),
    examples: $$(".example"),
    sourcePanel: $("#source-panel"),
    sourcePanelTitle: $("#source-panel-title"),
    sourcePanelBody: $("#source-panel-body"),
    sourcePanelClose: $("#source-panel-close"),
    navSignin: $("#nav-signin"),
    gateModal: $("#gate-modal"),
    gateModalClose: $("#gate-modal-close"),
    gateModalTitle: $("#gate-modal-title"),
    gateModalBody: $("#gate-modal-body"),
    gateModalPrimary: $("#gate-modal-primary"),
    gateModalSecondary: $("#gate-modal-secondary"),
    uploadBtn: $("#upload-btn"),
    uploadInput: $("#upload-input"),
    attachments: $("#attachments"),
  };

  // -------- state --------
  const state = {
    conversationId: null,
    sourcesByIdx: {},
    asking: false,
    user: null,           // {email} when signed in, else null
    usage: null,          // {kind, used, limit, remaining}
    documents: [],        // attached docs for the active conversation
  };

  const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
  const ALLOWED_UPLOAD_EXT = [".pdf", ".docx", ".txt", ".md"];
  const fmtBytes = (n) => {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(0) + " KB";
    return (n / (1024 * 1024)).toFixed(1) + " MB";
  };

  // -------- helpers --------
  const escape = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  const setStatus = (msg, kind = "") => {
    els.statusPill.textContent = msg;
    els.statusPill.className = `status-pill ${kind}`;
  };

  const showWelcome = (show) => {
    els.welcome.style.display = show ? "" : "none";
    // Toggle the centred / pulsing welcome layout on <main>. When the
    // user starts a conversation we drop the input bar back to the
    // bottom and stop the pulse.
    const main = document.getElementById("main-pane");
    if (main) main.classList.toggle("is-welcome", !!show);
  };

  // -------- auth + anon identity --------
  function getJWT() {
    return localStorage.getItem("purchase_token") || null;
  }
  function getAnonId() {
    let aid = localStorage.getItem("sopalai_anon_id");
    if (!aid) {
      // crypto.randomUUID is widely available; fallback to Math.random hex
      aid = (crypto && crypto.randomUUID) ? crypto.randomUUID()
            : Array.from({length: 32}, () => Math.floor(Math.random()*16).toString(16)).join("");
      localStorage.setItem("sopalai_anon_id", aid);
    }
    return aid;
  }
  function authHeaders() {
    const h = { "X-Anon-ID": getAnonId() };
    const t = getJWT();
    if (t) h["Authorization"] = "Bearer " + t;
    return h;
  }

  function updateUsageDisplay() {
    if (!state.usage) { els.usagePill.textContent = ""; return; }
    const u = state.usage;
    if (u.kind === "anon") {
      els.usagePill.textContent = `${u.remaining} free question${u.remaining === 1 ? "" : "s"} left`;
    } else {
      els.usagePill.textContent = `${u.remaining}/${u.limit} today`;
    }
  }

  async function refreshUsage() {
    try {
      const r = await fetch(`${API}/api/usage`, { headers: authHeaders() });
      if (!r.ok) return;
      const d = await r.json();
      state.usage = { kind: d.kind, used: d.used, limit: d.limit, remaining: d.remaining };
      state.user = d.is_signed ? { email: d.email } : null;
      updateUsageDisplay();
      updateSigninLink();
    } catch (e) { /* ignore */ }
  }

  function updateSigninLink() {
    if (!els.navSignin) return;
    if (state.user) {
      els.navSignin.textContent = "Account";
      els.navSignin.setAttribute("href", "/account");
    } else {
      els.navSignin.textContent = "Sign In";
      els.navSignin.setAttribute("href", "/login?redirect=/ai");
    }
  }

  function openGateModal(payload) {
    const isAnon = (payload && payload.kind === "anon") || !state.user;
    if (isAnon) {
      els.gateModalTitle.textContent = "You've used your free questions";
      els.gateModalBody.textContent = "Sign in (free) to get 30 questions per day.";
      els.gateModalPrimary.textContent = "Sign in";
      els.gateModalPrimary.setAttribute("href", "/login?redirect=/ai");
      els.gateModalSecondary.style.display = "";
      els.gateModalSecondary.setAttribute("href", "/register?redirect=/ai");
    } else {
      els.gateModalTitle.textContent = "Daily limit reached";
      els.gateModalBody.textContent =
        "You've reached today's 30-question limit. For unlimited access, please contact info@sopal.com.au to arrange enterprise access.";
      els.gateModalPrimary.textContent = "Email info@sopal.com.au";
      els.gateModalPrimary.setAttribute("href",
        "mailto:info@sopal.com.au?subject=SopalAI%20enterprise%20access");
      els.gateModalSecondary.style.display = "none";
    }
    els.gateModal.classList.add("open");
    els.gateModal.setAttribute("aria-hidden", "false");
  }
  function closeGateModal() {
    els.gateModal.classList.remove("open");
    els.gateModal.setAttribute("aria-hidden", "true");
  }
  if (els.gateModalClose) els.gateModalClose.addEventListener("click", closeGateModal);
  if (els.gateModal) els.gateModal.addEventListener("click", (e) => {
    if (e.target === els.gateModal) closeGateModal();
  });

  // -------- attachments / upload --------
  function renderAttachments() {
    if (!els.attachments) return;
    if (!state.documents.length) {
      els.attachments.hidden = true;
      els.attachments.innerHTML = "";
      return;
    }
    els.attachments.hidden = false;
    els.attachments.innerHTML = state.documents.map(d => `
      <span class="chip${d.error ? ' error' : ''}" data-id="${escape(d.id || '')}">
        <span class="chip-name">${escape(d.filename)}</span>
        <span class="chip-meta">${escape(fmtBytes(d.size_bytes || 0))}${d.truncated ? ' · trimmed' : ''}${d.error ? ' · ' + escape(d.error) : ''}</span>
        <button class="chip-remove" type="button" aria-label="Remove">×</button>
      </span>`).join("");
    els.attachments.querySelectorAll(".chip-remove").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const chip = e.target.closest(".chip");
        const id = chip && chip.dataset.id;
        if (id) removeAttachment(id);
      });
    });
  }

  async function refreshAttachmentsForConversation() {
    if (!state.conversationId) {
      state.documents = [];
      renderAttachments();
      return;
    }
    try {
      const r = await fetch(`${API}/api/conversations/${state.conversationId}/documents`);
      if (!r.ok) return;
      const data = await r.json();
      state.documents = (data.documents || []).map(d => ({...d}));
      renderAttachments();
    } catch (e) { /* ignore */ }
  }

  async function removeAttachment(docId) {
    // Optimistic UI
    state.documents = state.documents.filter(d => d.id !== docId);
    renderAttachments();
    try {
      await fetch(`${API}/api/upload/${encodeURIComponent(docId)}`, {
        method: "DELETE", headers: authHeaders(),
      });
    } catch (e) { /* ignore — local state already updated */ }
  }

  function validateUpload(file) {
    if (!file) return "No file";
    if (file.size > MAX_UPLOAD_BYTES) return "File too large (20 MB max)";
    const lower = (file.name || "").toLowerCase();
    if (!ALLOWED_UPLOAD_EXT.some(ext => lower.endsWith(ext))) {
      return "Unsupported type — use PDF, DOCX, TXT or MD";
    }
    return null;
  }

  async function handleUpload(file) {
    const err = validateUpload(file);
    if (err) {
      // Show transient error chip
      state.documents = state.documents.concat([{
        id: "err-" + Date.now(),
        filename: file.name || "upload",
        size_bytes: file.size || 0,
        error: err,
      }]);
      renderAttachments();
      setTimeout(() => {
        state.documents = state.documents.filter(d => !d.error);
        renderAttachments();
      }, 4000);
      return;
    }
    await ensureConversation();
    // Show optimistic uploading chip
    const tempId = "tmp-" + Date.now();
    state.documents = state.documents.concat([{
      id: tempId, filename: file.name, size_bytes: file.size, uploading: true,
    }]);
    renderAttachments();
    els.uploadBtn.classList.add("uploading");

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("conversation_id", state.conversationId);
      const r = await fetch(`${API}/api/upload`, {
        method: "POST",
        headers: authHeaders(),
        body: fd,
      });
      if (!r.ok) {
        let msg = `Upload failed (${r.status})`;
        try { const e = await r.json(); if (e.detail) msg = e.detail; } catch {}
        throw new Error(msg);
      }
      const data = await r.json();
      // Replace temp chip with real one
      state.documents = state.documents.map(d => d.id === tempId
        ? { id: data.id, filename: data.filename, size_bytes: data.size_bytes,
            n_chars: data.n_chars, truncated: data.truncated }
        : d);
    } catch (e) {
      state.documents = state.documents.map(d => d.id === tempId
        ? { ...d, uploading: false, error: e.message || "upload failed" } : d);
      setTimeout(() => {
        state.documents = state.documents.filter(d => d.id !== tempId);
        renderAttachments();
      }, 4500);
    } finally {
      els.uploadBtn.classList.remove("uploading");
      renderAttachments();
    }
  }

  if (els.uploadBtn) {
    els.uploadBtn.addEventListener("click", () => els.uploadInput && els.uploadInput.click());
  }
  if (els.uploadInput) {
    els.uploadInput.addEventListener("change", async (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) await handleUpload(f);
      e.target.value = ""; // allow re-upload of the same file
    });
  }

  // -------- tool view (claim-check etc) inside the SopalAI shell --------
  const TOOL_LABELS = {
    "claim-check": { claim: "Payment Claim Checker", schedule: "Payment Schedule Checker" },
  };
  function openToolView(tool, mode) {
    const view = document.getElementById("tool-view");
    const frame = document.getElementById("tool-view-frame");
    const title = document.getElementById("tool-view-title");
    const main = document.getElementById("main-pane");
    if (!view || !frame || !main) return;
    // The sitewide page-transition guard sets sessionStorage.nt-transitioning
    // to overlay a black cover on the next page until JS clears it. The
    // claim-check iframe inherits sessionStorage from this tab and would
    // pick up the flag, drawing a solid black rectangle. Clear it before
    // the iframe loads so the embedded page renders normally.
    try {
      sessionStorage.removeItem("nt-transitioning");
      if (frame.contentWindow && frame.contentWindow.sessionStorage) {
        frame.contentWindow.sessionStorage.removeItem("nt-transitioning");
      }
    } catch (e) { /* cross-origin or storage disabled — ignore */ }
    let src = "/claim-check?embed=1";
    if (mode === "claim" || mode === "schedule") src += "#" + mode;
    if (frame.dataset.currentSrc !== src) {
      frame.src = src;
      frame.dataset.currentSrc = src;
    }
    if (title) {
      const labels = TOOL_LABELS[tool] || {};
      title.textContent = labels[mode] || labels.claim || "SopalAI tool";
    }
    view.hidden = false;
    main.classList.add("in-tool-view");
  }
  function closeToolView(pushState = true) {
    const view = document.getElementById("tool-view");
    const main = document.getElementById("main-pane");
    if (view) view.hidden = true;
    if (main) main.classList.remove("in-tool-view");
    if (pushState) {
      const u = new URL(location.href);
      u.searchParams.delete("view");
      u.searchParams.delete("mode");
      history.pushState({}, "", u.toString());
    }
  }
  function applyUrlView() {
    const qs = new URLSearchParams(location.search);
    const view = qs.get("view");
    const mode = qs.get("mode");
    if (view === "claim-check") openToolView("claim-check", mode);
    else closeToolView(false);
  }
  document.querySelectorAll('.tool-item[data-tool="claim-check"]').forEach(a => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const mode = a.dataset.mode || "claim";
      const u = new URL(location.href);
      u.searchParams.set("view", "claim-check");
      u.searchParams.set("mode", mode);
      history.pushState({}, "", u.toString());
      openToolView("claim-check", mode);
    });
  });
  const toolViewClose = document.getElementById("tool-view-close");
  if (toolViewClose) toolViewClose.addEventListener("click", () => closeToolView(true));
  window.addEventListener("popstate", applyUrlView);
  applyUrlView();

  // -------- jurisdiction picker (QLD only for now) --------
  const jurisdictionBtn = document.getElementById("jurisdiction-btn");
  const jurisdictionMenu = document.getElementById("jurisdiction-menu");
  if (jurisdictionBtn && jurisdictionMenu) {
    const close = () => {
      jurisdictionMenu.hidden = true;
      jurisdictionBtn.setAttribute("aria-expanded", "false");
    };
    const open = () => {
      jurisdictionMenu.hidden = false;
      jurisdictionBtn.setAttribute("aria-expanded", "true");
    };
    jurisdictionBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      jurisdictionMenu.hidden ? open() : close();
    });
    document.addEventListener("click", (e) => {
      if (!jurisdictionMenu.hidden && !jurisdictionMenu.contains(e.target) && e.target !== jurisdictionBtn) close();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !jurisdictionMenu.hidden) close();
    });
  }

  // -------- recents --------
  async function loadRecents() {
    try {
      const r = await fetch(`${API}/api/conversations`);
      const items = await r.json();
      els.recents.innerHTML = "";
      items.forEach((c) => {
        const li = document.createElement("li");
        const span = document.createElement("span");
        span.className = "chat-title-text";
        span.textContent = c.title || "New chat";
        li.appendChild(span);
        li.dataset.id = c.id;
        li.addEventListener("click", () => loadConversation(c.id));
        els.recents.appendChild(li);
      });
      // Highlight current
      $$(".recents-list li").forEach((li) =>
        li.classList.toggle("active", li.dataset.id === state.conversationId)
      );
    } catch (e) {
      console.warn("recents load failed", e);
    }
  }

  // Sidebar: animate the chat title into place when the LLM-generated
  // title arrives over SSE. If the row isn't there yet (first turn of a
  // brand-new conversation), insert it at the top with the slide-in.
  function applyTitleToSidebar(conversationId, title) {
    if (!conversationId || !title) return;
    let li = els.recents.querySelector(`li[data-id="${conversationId}"]`);
    if (!li) {
      li = document.createElement("li");
      li.dataset.id = conversationId;
      const span = document.createElement("span");
      span.className = "chat-title-text";
      li.appendChild(span);
      li.addEventListener("click", () => loadConversation(conversationId));
      els.recents.prepend(li);
    }
    // Highlight as active
    $$(".recents-list li").forEach(x =>
      x.classList.toggle("active", x.dataset.id === conversationId)
    );
    let span = li.querySelector(".chat-title-text");
    if (!span) {
      span = document.createElement("span");
      span.className = "chat-title-text";
      li.appendChild(span);
    }
    // Trigger reflow so the animation re-runs even if the same node is reused
    span.classList.remove("title-slide-in");
    void span.offsetWidth;
    span.textContent = title;
    span.classList.add("title-slide-in");
  }

  async function ensureConversation() {
    if (state.conversationId) return state.conversationId;
    const r = await fetch(`${API}/api/conversations`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await r.json();
    state.conversationId = data.id;
    return state.conversationId;
  }

  async function loadConversation(id) {
    try {
      const r = await fetch(`${API}/api/conversations/${id}`);
      if (!r.ok) return;
      const conv = await r.json();
      // If the user is currently inside a tool view, close it so the
      // chat pane is visible again.
      if (typeof closeToolView === "function") closeToolView(true);
      state.conversationId = id;
      els.convStream.innerHTML = "";
      showWelcome(false);
      conv.messages.forEach((m) => {
        if (m.role === "user") {
          renderUserTurn(m.content.text || "");
        } else if (m.role === "assistant") {
          // Don't re-type historical answers when re-opening a conversation.
          renderAssistantTurn(m.content, { animate: false });
        }
      });
      loadRecents();
      refreshAttachmentsForConversation();
      scrollToBottom();
    } catch (e) {
      console.warn("conv load failed", e);
    }
  }

  // -------- new chat --------
  function newChat() {
    // If the tool view is open, close it so the chat pane is visible
    if (typeof closeToolView === "function") closeToolView(true);
    state.conversationId = null;
    els.convStream.innerHTML = "";
    showWelcome(true);
    els.input.value = "";
    state.documents = [];
    renderAttachments();
    els.input.focus();
    loadRecents();
  }

  // -------- rendering --------
  function renderUserTurn(text) {
    const div = document.createElement("div");
    div.className = "turn turn-user";
    div.innerHTML = `<div class="role">You</div><div class="user-text">${escape(text)}</div>`;
    els.convStream.appendChild(div);
  }

  function renderStatus(parent, msg, payload) {
    // Mark all previous status rows as done (green dot, no pulse)
    const prior = parent.querySelectorAll(".turn-status.in-progress");
    prior.forEach(p => {
      p.classList.remove("in-progress");
      p.classList.add("done");
    });
    const div = document.createElement("div");
    div.className = "turn-status in-progress";
    let inner = `<span class="dot"></span><span class="msg">${escape(msg)}</span>`;
    // For reading_cases, render the case-citation list under the status row
    if (payload && payload.phase === "reading_cases" && Array.isArray(payload.cases)) {
      const items = payload.cases.map(c => `<li>${escape(c)}</li>`).join("");
      inner += `<ul class="status-cases">${items}</ul>`;
      if (Array.isArray(payload.missed) && payload.missed.length) {
        const missed = payload.missed.map(c => `<li>${escape(c)} <span class="missed-tag">(not in corpus)</span></li>`).join("");
        inner += `<div class="status-cases-label">Mentioned but not indexed:</div><ul class="status-cases status-cases-missed">${missed}</ul>`;
      }
    }
    div.innerHTML = inner;
    parent.appendChild(div);
    return div;
  }

  function markAllStatusesDone(parent) {
    parent.querySelectorAll(".turn-status.in-progress").forEach(p => {
      p.classList.remove("in-progress");
      p.classList.add("done");
    });
  }

  // Wrap each visible word inside `root` in a <span class="type-word"> with
  // a staggered animation-delay so the answer fades in left-to-right at a
  // typing-style cadence. CSS does the actual animation.
  //
  // Citation markers ([1] superscripts) ARE animated in flow so they don't
  // pop in before the surrounding paragraph text. Block-level containers
  // that have visual chrome (blockquotes / answer-quote) are made
  // initially-hidden and fade in synchronously with their first word, so
  // the green box doesn't appear empty before its text.
  function applyTypingAnimation(root, msPerWord = 16) {
    if (!root) return;
    // Walk every text node; only skip nodes inside elements that already
    // have their own initial hiding (none, currently) or that we don't
    // want animated like <code>/<pre>.
    const skipNode = (node) => {
      let n = node;
      while (n && n !== root) {
        if (n.nodeType === 1 && n.tagName) {
          const tag = n.tagName.toLowerCase();
          if (tag === "code" || tag === "pre") return true;
          if (n.classList && n.classList.contains("confidence-indicator")) return true;
        }
        n = n.parentNode;
      }
      return false;
    };
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (skipNode(n.parentNode) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    let wordIdx = 0;
    for (const node of nodes) {
      if (!node.nodeValue || !node.nodeValue.trim()) continue;
      const parts = node.nodeValue.split(/(\s+)/);
      const frag = document.createDocumentFragment();
      for (const p of parts) {
        if (!p) continue;
        if (/^\s+$/.test(p)) {
          frag.appendChild(document.createTextNode(p));
        } else {
          const span = document.createElement("span");
          span.className = "type-word";
          span.style.animationDelay = (wordIdx * msPerWord) + "ms";
          span.textContent = p;
          frag.appendChild(span);
          wordIdx++;
        }
      }
      node.parentNode.replaceChild(frag, node);
    }

    // Block-level containers with visual chrome must wait for their first
    // word to start before becoming visible — otherwise the green
    // blockquote box appears empty before its quoted text fades in.
    root.querySelectorAll("blockquote, .answer-quote").forEach(bq => {
      const firstWord = bq.querySelector(".type-word");
      if (!firstWord) return;
      bq.classList.add("type-block");
      bq.style.animationDelay = firstWord.style.animationDelay;
    });
  }

  function renderAssistantTurn(answer, opts = {}) {
    const animate = opts.animate !== false;
    const div = document.createElement("div");
    div.className = "turn turn-assistant";
    div.innerHTML = `
      <div class="role">Sopal Research</div>
      ${answer.answer_html || "<p>(no answer)</p>"}
    `;

    // Type-out animation on the answer body (not the role label, not sources).
    if (animate) {
      const summary = div.querySelector(".answer-summary");
      const body = div.querySelector(".answer-body");
      if (summary) applyTypingAnimation(summary, 14);
      if (body) {
        // Continue the cadence after the summary so the body picks up where
        // summary left off rather than overlapping. Approximate offset:
        // count summary words and shift body delays.
        const summaryWords = summary ? summary.querySelectorAll(".type-word").length : 0;
        applyTypingAnimation(body, 14);
        if (summaryWords) {
          body.querySelectorAll(".type-word").forEach((s, i) => {
            s.style.animationDelay = ((summaryWords + i) * 14) + "ms";
          });
        }
      }
    }

    // Sources block
    if (answer.sources && answer.sources.length) {
      const wrap = document.createElement("div");
      wrap.className = "sources-list";
      wrap.innerHTML = "<h4>Sources</h4>";
      answer.sources.forEach((s, i) => {
        const idx = i + 1;
        const card = document.createElement("a");
        card.href = `#src-${idx}`;
        card.id = `src-${idx}`;
        card.className = "source-card";
        const title = renderCitationLine(s.metadata, s.header);
        const meta = renderMetaSubline(s.metadata);
        card.innerHTML = `
          <div><span class="src-num">${idx}</span><span class="src-title">${escape(title)}</span></div>
          ${meta ? `<div class="src-meta">${escape(meta)}</div>` : ""}
        `;
        card.addEventListener("click", (e) => {
          e.preventDefault();
          openSourcePanel(s.id, idx);
        });
        wrap.appendChild(card);
        state.sourcesByIdx[idx] = s;
      });
      div.appendChild(wrap);
    }
    els.convStream.appendChild(div);

    // Wire up inline cite markers
    $$(".cite-marker", div).forEach((a) => {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        const idx = parseInt(a.dataset.cite || a.textContent.replace(/[\[\]]/g, ""), 10);
        const src = state.sourcesByIdx[idx];
        if (src) openSourcePanel(src.id, idx);
      });
    });
  }

  function renderCitationLine(meta, fallback) {
    if (!meta) return fallback || "Source";
    const t = meta.source_type || "";
    if (t === "statute" || t === "regulation") {
      const act = meta.act_short || meta.act_name || "";
      const sec = meta.section_number || "";
      const sub = meta.subsection_path || "";
      const title = meta.section_title || "";
      let line = `${act} s ${sec}${sub}`.trim();
      if (title) line += ` — ${title}`;
      return line;
    }
    if (t === "judgment") {
      const cite = meta.citation || meta.case_name || "";
      const ps = meta.paragraph_start, pe = meta.paragraph_end;
      if (ps && pe && ps !== pe) return `${cite} at [${ps}]–[${pe}]`;
      if (ps) return `${cite} at [${ps}]`;
      return cite;
    }
    if (t === "decision") {
      const ref = meta.decision_id || "";
      const parties = meta.parties || "";
      return `Adjudication Decision ${ref}${parties ? " — " + parties : ""}`;
    }
    if (t === "annotated") {
      const sec = meta.section_number || "";
      return `Annotated BIF Act — commentary on s ${sec}`;
    }
    return fallback || "Source";
  }

  function renderMetaSubline(meta) {
    if (!meta) return "";
    if (meta.source_type === "judgment") return [meta.court, meta.year].filter(Boolean).join(" · ");
    if (meta.source_type === "decision") return [meta.adjudicator, meta.decision_date].filter(Boolean).join(" · ");
    return "";
  }

  // Source text comes from PDF/document extraction and has hard line wraps
  // every ~80 chars within each logical paragraph. Collapse those soft wraps
  // back into prose; treat blank lines as paragraph breaks. Preserve [N]
  // judgment paragraph markers and (a)(b)(c) statute markers.
  function normalizeSourceText(text) {
    if (!text) return "";
    // Split on blank-line paragraph breaks (also tolerates form-feed \f from PDF)
    const paras = text.split(/(?:\r?\n\s*){2,}|\f/);
    const blocks = paras
      .map(p => p
        // collapse soft line wraps (newline + leading whitespace) into one space
        .replace(/\s*\n\s*/g, " ")
        // collapse runs of horizontal whitespace
        .replace(/[ \t]+/g, " ")
        .trim()
      )
      .filter(Boolean);
    return blocks.map(p => `<p class="src-para">${escape(p)}</p>`).join("");
  }

  // -------- source panel --------
  async function openSourcePanel(chunkId, idx) {
    els.sourcePanel.classList.add("open");
    els.sourcePanel.setAttribute("aria-hidden", "false");
    els.sourcePanelTitle.textContent = `Source [${idx}]`;
    els.sourcePanelBody.innerHTML = "<p>Loading…</p>";
    try {
      const r = await fetch(`${API}/api/sources/${encodeURIComponent(chunkId)}`);
      if (!r.ok) throw new Error("not found");
      const data = await r.json();
      const meta = data.metadata || {};
      const metaRows = Object.entries(meta)
        .filter(([_, v]) => v !== "" && v !== null && v !== undefined)
        .map(([k, v]) => `<div class="meta-row"><dt>${escape(k)}:</dt> <dd>${escape(String(v))}</dd></div>`)
        .join("");
      els.sourcePanelBody.innerHTML = `
        <div class="source-header-line">${escape(data.header || "")}</div>
        <dl class="source-meta">${metaRows}</dl>
        <div class="source-text">${normalizeSourceText(data.text || "")}</div>
      `;
    } catch (e) {
      els.sourcePanelBody.innerHTML = `<p>Could not load source: ${escape(String(e))}</p>`;
    }
  }

  els.sourcePanelClose.addEventListener("click", () => {
    els.sourcePanel.classList.remove("open");
    els.sourcePanel.setAttribute("aria-hidden", "true");
  });

  // -------- scroll --------
  function scrollToBottom() {
    requestAnimationFrame(() => {
      els.convContainer.scrollTop = els.convContainer.scrollHeight;
    });
  }

  // -------- ask flow --------
  async function ask() {
    if (state.asking) return;
    const q = els.input.value.trim();
    if (!q) return;
    state.asking = true;
    els.askBtn.disabled = true;
    setStatus("Working…");
    showWelcome(false);
    state.sourcesByIdx = {};

    await ensureConversation();

    renderUserTurn(q);
    els.input.value = "";
    autosize();
    scrollToBottom();

    // assistant turn placeholder with status messages
    const turn = document.createElement("div");
    turn.className = "turn turn-assistant";
    turn.innerHTML = `<div class="role">Sopal Research</div>`;
    const statusBlock = document.createElement("div");
    statusBlock.className = "status-block";
    turn.appendChild(statusBlock);
    els.convStream.appendChild(turn);
    scrollToBottom();

    try {
      const res = await fetch(`${API}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ question: q, conversation_id: state.conversationId }),
      });
      if (res.status === 429) {
        // Quota gate: show the upgrade modal and back out
        let payload = null;
        try { payload = await res.json(); } catch { /* ignore */ }
        statusBlock.remove();
        // Remove the placeholder assistant turn since we won't fill it in
        if (turn.parentNode) turn.parentNode.removeChild(turn);
        openGateModal(payload);
        await refreshUsage();
        setStatus("Ready");
        return;
      }
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let answerData = null;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const events = buf.split("\n\n");
        buf = events.pop() || "";
        for (const ev of events) {
          const lines = ev.split("\n");
          let event = "message", data = "";
          for (const ln of lines) {
            if (ln.startsWith("event: ")) event = ln.slice(7).trim();
            else if (ln.startsWith("data: ")) data += ln.slice(6);
          }
          if (!data) continue;
          let parsed;
          try { parsed = JSON.parse(data); } catch { continue; }

          if (event === "status") {
            renderStatus(statusBlock, parsed.msg || parsed.phase || "Working…", parsed);
            scrollToBottom();
          } else if (event === "title") {
            // Slide the AI-generated chat title into the sidebar
            applyTitleToSidebar(parsed.conversation_id, parsed.title);
          } else if (event === "answer") {
            answerData = parsed;
            markAllStatusesDone(statusBlock);
          } else if (event === "error") {
            markAllStatusesDone(statusBlock);
            const errDiv = document.createElement("div");
            errDiv.className = "answer-refusal";
            errDiv.innerHTML = `<p>${escape(parsed.error || "Unknown error")}</p>`;
            turn.appendChild(errDiv);
          } else if (event === "done") {
            statusBlock.remove();
            if (answerData) {
              // Replace the whole turn with the rendered version
              els.convStream.removeChild(turn);
              renderAssistantTurn(answerData);
              scrollToBottom();
            }
          }
        }
      }
      setStatus("Ready");
    } catch (e) {
      console.error(e);
      setStatus(`Error: ${e.message}`);
      const err = document.createElement("div");
      err.className = "answer-refusal";
      err.innerHTML = `<p>Could not get an answer: ${escape(String(e))}</p>`;
      turn.appendChild(err);
    } finally {
      state.asking = false;
      els.askBtn.disabled = !els.input.value.trim();
      loadRecents();
      refreshUsage();
    }
  }

  // -------- input behaviour --------
  function autosize() {
    els.input.style.height = "auto";
    // Grow with content up to 240px, but never collapse below the CSS
    // min-height (64px) — that's enforced by CSS even if we set lower.
    els.input.style.height = Math.min(240, Math.max(64, els.input.scrollHeight)) + "px";
  }
  els.input.addEventListener("input", () => {
    autosize();
    els.askBtn.disabled = !els.input.value.trim() || state.asking;
  });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!els.askBtn.disabled) ask();
    }
  });
  els.askBtn.addEventListener("click", ask);
  els.newChatBtn.addEventListener("click", newChat);
  els.examples.forEach((b) => {
    b.addEventListener("click", () => {
      els.input.value = b.dataset.q;
      autosize();
      els.askBtn.disabled = false;
      els.input.focus();
    });
  });

  // -------- init --------
  loadRecents();
  refreshUsage();
  els.input.focus();
})();
