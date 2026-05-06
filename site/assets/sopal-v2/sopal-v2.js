/* Sopal v2 workspace — isolated client. Talks only to existing Sopal endpoints
   and /api/sopal-v2/* helpers. Does not modify any live Sopal page. */
(function () {
  "use strict";

  const root = document.getElementById("sopal-v2-root");
  const STORAGE_KEY = "sopal-v2-local-workspace-v1";
  const CHAT_HISTORY_KEY = "sopal-v2-chat-history-v1";
  const TOKEN_KEY = "purchase_token";

  /* ---------- Static config ---------- */

  const sections = [
    { title: "Research", items: [
      ["Adjudication Decisions", "/sopal-v2/research/adjudication-decisions"],
      ["Adjudicator Statistics", "/sopal-v2/research/adjudicator-statistics"],
    ] },
    { title: "Tools", items: [
      ["Due Date Calculator", "/sopal-v2/tools/due-date-calculator"],
      ["Interest Calculator", "/sopal-v2/tools/interest-calculator"],
    ] },
    { title: "Projects", items: [
      ["Contracts", "/sopal-v2/projects/contracts"],
      ["Project Library", "/sopal-v2/projects/library"],
      ["Assistant", "/sopal-v2/projects/assistant"],
    ] },
    { title: "Agents", items: [
      ["Payment Claims", "/sopal-v2/agents/payment-claims"],
      ["Payment Schedules", "/sopal-v2/agents/payment-schedules"],
      ["EOTs", "/sopal-v2/agents/eots"],
      ["Variations", "/sopal-v2/agents/variations"],
      ["Delay Costs", "/sopal-v2/agents/delay-costs"],
      ["Adjudication Application", "/sopal-v2/agents/adjudication-application"],
      ["Adjudication Response", "/sopal-v2/agents/adjudication-response"],
    ] },
  ];

  const agentLabels = Object.fromEntries(
    sections[3].items.map(([label, href]) => [href.split("/").pop(), label])
  );

  const agentDescriptions = {
    "payment-claims": "Review or draft payment claim material with SOPA compliance, work identification, dates, service, and evidence focus.",
    "payment-schedules": "Review or draft payment schedules with scheduled amount, withholding reasons, timing, and adjudication-risk focus.",
    eots: "Review or draft extension of time notices and claims against contract requirements, causation, and critical delay evidence.",
    variations: "Review or draft variation notices and claims with direction, scope, valuation, notice, and evidence focus.",
    "delay-costs": "Review or draft delay cost, prolongation, or disruption claims with entitlement, causation, quantum, and duplication focus.",
    "adjudication-application": "Review or draft adjudication application submissions, jurisdiction, chronology, entitlement, quantum, and annexures.",
    "adjudication-response": "Review or draft adjudication response submissions, jurisdictional objections, payment schedule alignment, and evidence.",
  };

  const includeLists = {
    "payment-claims": ["Payment claim text", "Contract clauses", "Date served / received", "Reference date or claim date", "Prior claims if relevant", "Invoices and supporting schedules"],
    "payment-schedules": ["Payment claim being answered", "Draft/current schedule", "Scheduled amount", "Reasons for withholding", "Date claim received", "Contract payment clauses"],
    eots: ["Contract EOT clause", "Delay event", "Notice date", "Delay period", "Programme/critical path facts", "Supporting correspondence/photos"],
    variations: ["Contract variation clause", "Instruction or direction", "Changed scope", "Notice date", "Valuation material", "Time impact facts"],
    "delay-costs": ["Entitlement clause", "Delay event and period", "Causation facts", "Quantum calculation", "Notice correspondence", "Overlap/duplication checks"],
    "adjudication-application": ["Payment claim", "Payment schedule", "Contract", "Chronology", "Evidence bundle", "Quantum/supporting calculations"],
    "adjudication-response": ["Application", "Payment schedule", "Contract", "Jurisdictional objections", "Evidence responding to each item", "Reasons already raised"],
  };

  // Scenario starters tailored to each agent and mode.
  const scenarioStarters = {
    "payment-claims:review": [
      "Review my payment claim served on [date] for $[amount]. Identify any BIF Act compliance issues and missing items.",
      "I served a payment claim that the respondent says is invalid. Help me check whether it meets the BIF Act requirements.",
      "Check whether this is a repeat claim and whether the reference date is valid.",
    ],
    "payment-claims:draft": [
      "Draft a payment claim for [scope of work] for $[amount] under contract [name]. Include the BIF Act endorsement.",
      "Draft a cover email serving a payment claim on the respondent.",
      "Prepare a payment claim itemised by trade with placeholders for invoices and dates.",
    ],
    "payment-schedules:review": [
      "Review this payment schedule for adequacy, withholding reasons, and adjudication risk.",
      "I'm the claimant. Find weaknesses or vague reasons in the respondent's payment schedule.",
      "Check whether reasons in this schedule will support a later adjudication response under s 82.",
    ],
    "payment-schedules:draft": [
      "Draft a payment schedule scheduling $[amount] with itemised reasons for withholding.",
      "Draft a $0 payment schedule with full statutory reasons and a reservation of rights.",
      "Prepare a payment schedule responding to the attached claim, item by item.",
    ],
    "eots:review": [
      "Review my draft EOT notice for [event] under clause [#]. Check causation and timing.",
      "I missed the contractual notice deadline. Tell me my exposure and any options.",
      "Check whether this delay is on the critical path and whether the programme evidence is enough.",
    ],
    "eots:draft": [
      "Draft an EOT notice for [event] of [days] days delay under clause [#].",
      "Draft a detailed EOT claim for prolonged wet weather affecting concrete pours.",
      "Prepare a covering letter serving an EOT notice with a programme analysis annexure.",
    ],
    "variations:review": [
      "Review this variation claim for entitlement, contractual basis, and valuation.",
      "Check whether the principal's instruction is a variation or a clarification.",
      "Identify time-bar risks on this variation notice.",
    ],
    "variations:draft": [
      "Draft a variation notice for [scope] valued at $[amount] under clause [#].",
      "Draft a variation claim with cost breakdown, time impact, and reservation of rights.",
      "Prepare a notice converting a verbal direction into a written variation request.",
    ],
    "delay-costs:review": [
      "Review my prolongation claim. Check causation, overlap with EOTs, and quantum support.",
      "Are there duplication risks between my EOT, variation, and delay cost claims?",
      "Test whether this disruption claim has enough contemporaneous records.",
    ],
    "delay-costs:draft": [
      "Draft a prolongation claim for [days] days at preliminaries of $[rate]/week.",
      "Draft a disruption claim using a measured-mile approach.",
      "Prepare a delay cost claim with entitlement, causation, quantum, and evidence schedule.",
    ],
    "adjudication-application:review": [
      "Review my adjudication application for jurisdiction, timing, and structure under the BIF Act.",
      "Check whether my chronology and evidence schedule support each claim item.",
      "Find weak points the respondent will attack in the response.",
    ],
    "adjudication-application:draft": [
      "Draft the structure of an adjudication application for $[amount].",
      "Prepare the issues, entitlement, and quantum sections of an adjudication application.",
      "Draft submissions answering anticipated jurisdictional objections.",
    ],
    "adjudication-response:review": [
      "Review my draft adjudication response. Check alignment with the payment schedule.",
      "Identify any new reasons that I cannot raise under s 82(4).",
      "Check whether my jurisdictional objections are properly framed.",
    ],
    "adjudication-response:draft": [
      "Draft an adjudication response structure with jurisdictional objections.",
      "Prepare a response to each item in the application with evidence references.",
      "Draft submissions on quantum and offset reductions.",
    ],
  };

  const headers = {
    home: ["Sopal", "Research decisions, calculate dates and interest, and run SOPA document workflows."],
    "research/adjudication-decisions": ["Adjudication Decisions", "Search the full Sopal adjudication decision database."],
    "research/adjudicator-statistics": ["Adjudicator Statistics", "Real outcomes by adjudicator across the Sopal database."],
    "tools/due-date-calculator": ["Due Date Calculator", "BIF Act business-day calculator with Queensland holiday handling."],
    "tools/interest-calculator": ["Interest Calculator", "Statutory and contractual interest with daily RBA rates."],
    "projects/contracts": ["Contracts", "Build local contract context for the Assistant and agents."],
    "projects/library": ["Project Library", "Build local project context: claims, notices, correspondence."],
    "projects/assistant": ["Assistant", "Ask AI questions against typed instructions and your local context."],
  };

  // QLD public holidays + regional show holidays (matches the live calculator).
  const holidays = {
    qld: [
      ["2025-01-01", "New Year's Day"], ["2025-01-27", "Australia Day"], ["2025-04-18", "Good Friday"], ["2025-04-19", "Day after Good Friday"], ["2025-04-21", "Easter Monday"], ["2025-04-25", "Anzac Day"], ["2025-05-05", "Labour Day"], ["2025-10-06", "King's Birthday"], ["2025-12-25", "Christmas Day"], ["2025-12-26", "Boxing Day"],
      ["2026-01-01", "New Year's Day"], ["2026-01-26", "Australia Day"], ["2026-04-03", "Good Friday"], ["2026-04-04", "Day after Good Friday"], ["2026-04-06", "Easter Monday"], ["2026-04-25", "Anzac Day"], ["2026-05-04", "Labour Day"], ["2026-10-05", "King's Birthday"], ["2026-12-25", "Christmas Day"], ["2026-12-28", "Boxing Day Holiday"],
      ["2027-01-01", "New Year's Day"], ["2027-01-26", "Australia Day"], ["2027-03-26", "Good Friday"], ["2027-03-27", "Day after Good Friday"], ["2027-03-29", "Easter Monday"], ["2027-04-26", "Anzac Day Holiday"], ["2027-05-03", "Labour Day"], ["2027-10-04", "King's Birthday"], ["2027-12-27", "Christmas Day Holiday"], ["2027-12-28", "Boxing Day Holiday"],
      ["2028-01-03", "New Year's Day Holiday"], ["2028-01-26", "Australia Day"], ["2028-04-14", "Good Friday"], ["2028-04-15", "Day after Good Friday"], ["2028-04-17", "Easter Monday"], ["2028-04-25", "Anzac Day"], ["2028-05-01", "Labour Day"], ["2028-10-02", "King's Birthday"], ["2028-12-25", "Christmas Day"], ["2028-12-26", "Boxing Day"],
    ].map(([date, name]) => ({ date, name })),
    bne: [["2025-08-13", "Brisbane EKKA"], ["2026-08-12", "Brisbane EKKA"], ["2027-08-11", "Brisbane EKKA"], ["2028-08-16", "Brisbane EKKA"]].map(([date, name]) => ({ date, name })),
    gld: [["2025-08-29", "Gold Coast Show"]].map(([date, name]) => ({ date, name })),
    cns: [["2025-07-18", "Cairns Show"]].map(([date, name]) => ({ date, name })),
    tsw: [["2025-07-07", "Townsville Show"]].map(([date, name]) => ({ date, name })),
    ipswich: [["2025-05-16", "Ipswich Show"]].map(([date, name]) => ({ date, name })),
    toowoomba: [["2025-03-28", "Toowoomba Show"]].map(([date, name]) => ({ date, name })),
    sunshine_coast: [["2025-05-30", "Maleny Show"], ["2025-06-13", "Sunshine Coast Show"]].map(([date, name]) => ({ date, name })),
    rockhampton: [["2025-06-12", "Rockhampton Show"]].map(([date, name]) => ({ date, name })),
    mackay: [["2025-06-19", "Mackay Show"]].map(([date, name]) => ({ date, name })),
  };

  /* ---------- State ---------- */

  let sidebarOpen = false;
  let workspace = loadWorkspace();
  let chatHistory = loadChatHistory();
  let authUser = null; // populated by refreshAuth()
  let modal = null;    // {render: () => string, bind: (root) => void}

  function loadWorkspace() {
    try {
      return Object.assign({ contracts: [], library: [] }, JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"));
    } catch { return { contracts: [], library: [] }; }
  }
  function saveWorkspace() { localStorage.setItem(STORAGE_KEY, JSON.stringify(workspace)); }

  function loadChatHistory() {
    try { return JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || "{}"); }
    catch { return {}; }
  }
  function saveChatHistory() { localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatHistory)); }

  function chatKey(options) {
    if (options.assistant) return "assistant";
    return `${options.agentType || "unknown"}:${options.mode || "review"}`;
  }

  function authToken() { return localStorage.getItem(TOKEN_KEY) || ""; }

  function authHeaders(extra) {
    const h = Object.assign({}, extra || {});
    const t = authToken();
    if (t) h.Authorization = `Bearer ${t}`;
    return h;
  }

  async function refreshAuth() {
    const t = authToken();
    if (!t) { authUser = null; return; }
    try {
      const r = await fetch("/purchase-me", { headers: authHeaders(), credentials: "include" });
      if (r.ok) authUser = await r.json();
      else { authUser = null; if (r.status === 401) localStorage.removeItem(TOKEN_KEY); }
    } catch { authUser = null; }
  }

  function signOut() {
    localStorage.removeItem(TOKEN_KEY);
    authUser = null;
    render();
  }

  /* ---------- Helpers ---------- */

  function escapeHtml(value) {
    return String(value || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }

  function cleanPath() {
    const path = window.location.pathname.replace(/\/+$/, "");
    return path.replace(/^\/sopal-v2\/?/, "") || "home";
  }

  function navigate(href) { window.history.pushState({}, "", href); render(); }

  function isActive(href) {
    const current = window.location.pathname.replace(/\/+$/, "") || "/sopal-v2";
    return current === href || (href !== "/sopal-v2" && current.startsWith(href + "/"));
  }

  function routeTitle(route) {
    if (route.startsWith("agents/")) {
      const key = route.split("/")[1];
      return [agentLabels[key] || "Agent", agentDescriptions[key] || "Review or draft SOPA material."];
    }
    return headers[route] || headers.home;
  }

  function parseDate(value) {
    if (!value) return null;
    const date = new Date(`${value}T00:00:00`);
    return Number.isNaN(date.getTime()) ? null : date;
  }
  function formatDate(date) {
    return date ? date.toLocaleDateString("en-AU", { weekday: "long", year: "numeric", month: "long", day: "numeric" }) : "";
  }
  function formatCurrency(value) {
    const num = Number(value || 0);
    return num.toLocaleString("en-AU", { style: "currency", currency: "AUD", maximumFractionDigits: 0 });
  }
  function formatPercent(value) { return `${Number(value || 0).toFixed(1)}%`; }
  function money(value) {
    if (value === null || value === undefined || value === "" || value === "N/A") return "";
    const n = Number(value);
    return Number.isFinite(n) && n > 0 ? formatCurrency(n) : "";
  }
  function formatSnippet(text) {
    return escapeHtml(text || "").replace(/&lt;mark&gt;/g, "<mark>").replace(/&lt;\/mark&gt;/g, "</mark>");
  }

  /* ---------- Markdown rendering (line-based, safe) ---------- */

  function renderMarkdown(text) {
    const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let mode = null;       // null | "ul" | "ol"
    let para = [];

    function flushPara() {
      if (!para.length) return;
      out.push(`<p>${inline(para.join(" "))}</p>`);
      para = [];
    }
    function flushList() {
      if (mode === "ul") out.push("</ul>");
      if (mode === "ol") out.push("</ol>");
      mode = null;
    }

    for (let raw of lines) {
      const line = raw.replace(/\s+$/, "");
      if (!line.trim()) { flushPara(); flushList(); continue; }

      const h = line.match(/^(#{1,6})\s+(.+)$/);
      if (h) {
        flushPara(); flushList();
        const level = Math.min(h[1].length + 2, 6); // # -> h3 (avoid duplicating page h1)
        out.push(`<h${level}>${inline(h[2])}</h${level}>`);
        continue;
      }

      const bullet = line.match(/^\s*[-*]\s+(.+)$/);
      if (bullet) {
        flushPara();
        if (mode !== "ul") { flushList(); out.push("<ul>"); mode = "ul"; }
        out.push(`<li>${inline(bullet[1])}</li>`);
        continue;
      }

      const num = line.match(/^\s*(\d+)[.)]\s+(.+)$/);
      if (num) {
        flushPara();
        if (mode !== "ol") { flushList(); out.push("<ol>"); mode = "ol"; }
        out.push(`<li>${inline(num[2])}</li>`);
        continue;
      }

      flushList();
      para.push(line);
    }
    flushPara();
    flushList();
    return out.join("");

    function inline(s) {
      let safe = escapeHtml(s);
      safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
      safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      safe = safe.replace(/(^|[\s(>])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
      safe = safe.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
      return safe;
    }
  }

  /* ---------- Sidebar / header ---------- */

  function authButton() {
    if (authUser) {
      const initial = (authUser.first_name || authUser.email || "?").trim().charAt(0).toUpperCase();
      const label = authUser.first_name ? authUser.first_name : authUser.email;
      return `<div class="auth-pill" data-user-menu>
        <span class="avatar avatar-sm">${escapeHtml(initial)}</span>
        <span class="auth-pill-label">${escapeHtml(label)}</span>
        <button class="ghost-button compact" type="button" data-sign-out>Sign out</button>
      </div>`;
    }
    return `<button class="dark-button compact" type="button" data-sign-in>Sign in</button>`;
  }

  function Sidebar() {
    return `
      <aside class="sopal-sidebar ${sidebarOpen ? "open" : ""}">
        <div class="sidebar-top">
          <div class="wordmark">
            <a href="/sopal-v2" data-nav>Sopal</a>
            <span class="prototype-pill">v2</span>
          </div>
          <button class="new-project" type="button" data-go="/sopal-v2/projects/contracts">+ New context</button>
        </div>
        <div class="nav-scroll">
          ${sections.map((section) => `
            <div class="nav-group">
              <div class="nav-group-title">${escapeHtml(section.title)}</div>
              ${section.items.map(([label, href]) => `<a class="nav-item ${isActive(href) ? "active" : ""}" href="${href}" data-nav>${escapeHtml(label)}</a>`).join("")}
            </div>
          `).join("")}
        </div>
        <div class="sidebar-bottom">
          <div class="sidebar-auth">${authButton()}</div>
          <div class="sidebar-context-count">${workspace.contracts.length + workspace.library.length} local context item${workspace.contracts.length + workspace.library.length === 1 ? "" : "s"}</div>
        </div>
      </aside>
    `;
  }

  function MainHeader(route) {
    const [title, description] = routeTitle(route);
    return `
      <header class="main-header">
        <div class="main-title">
          <h1>${escapeHtml(title)}</h1>
          <p>${escapeHtml(description)}</p>
        </div>
        <div class="header-actions">
          <button class="ghost-button mobile-toggle" type="button" data-toggle-sidebar aria-label="Open menu">Menu</button>
          <a class="link-button" href="/ai" target="_blank" rel="noopener">SopalAI</a>
          <a class="link-button" href="https://sopal.com.au" target="_blank" rel="noopener">sopal.com.au</a>
        </div>
      </header>
    `;
  }

  function EmptyState(title, body, actionHtml) {
    return `<div class="empty-state"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(body)}</p>${actionHtml || ""}</div>`;
  }

  /* ---------- Home ---------- */

  function HomePage() {
    const tiles = [
      { title: "Search adjudication decisions", body: "Full-text search the BIF Act / BCIPA decision database with filters and detail views.", href: "/sopal-v2/research/adjudication-decisions", group: "Research" },
      { title: "Adjudicator statistics", body: "Browse award rates, total claimed/awarded, and decision history by adjudicator.", href: "/sopal-v2/research/adjudicator-statistics", group: "Research" },
      { title: "Due date calculator", body: "BIF Act business-day deadlines for claims, schedules, applications, responses and decisions.", href: "/sopal-v2/tools/due-date-calculator", group: "Tools" },
      { title: "Interest calculator", body: "Statutory (QBCC s 67P, daily RBA) or contractual interest with full daily breakdown.", href: "/sopal-v2/tools/interest-calculator", group: "Tools" },
      { title: "Review a payment claim", body: "AI review against BIF Act compliance, evidence and amendment recommendations.", href: "/sopal-v2/agents/payment-claims?mode=review", group: "Agents" },
      { title: "Draft an EOT claim", body: "AI-drafted notice or claim with assumptions, placeholders and evidence schedule.", href: "/sopal-v2/agents/eots?mode=draft", group: "Agents" },
      { title: "Adjudication application", body: "Build a structured application: chronology, jurisdiction, entitlement, quantum.", href: "/sopal-v2/agents/adjudication-application?mode=draft", group: "Agents" },
      { title: "Project assistant", body: "Chat across pasted/extracted contract and project context.", href: "/sopal-v2/projects/assistant", group: "Projects" },
    ];

    const ctxCount = workspace.contracts.length + workspace.library.length;
    const recentChats = Object.entries(chatHistory)
      .filter(([, h]) => Array.isArray(h.messages) && h.messages.length > 0)
      .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
      .slice(0, 4);

    const greeting = authUser
      ? `Welcome back, ${escapeHtml(authUser.first_name || authUser.email)}.`
      : `Sign in to use the decision search. The rest of the workspace works without an account.`;

    return `
      <section class="home-hero">
        <div class="home-hero-text">
          <h2>What are you working on?</h2>
          <p>${greeting}</p>
          ${authUser ? "" : `<button class="dark-button" type="button" data-sign-in>Sign in</button>`}
        </div>
      </section>
      <section class="home-tiles">
        ${tiles.map((t) => `
          <a class="tile" href="${t.href}" data-nav>
            <span class="tile-tag">${escapeHtml(t.group)}</span>
            <strong>${escapeHtml(t.title)}</strong>
            <span class="tile-body">${escapeHtml(t.body)}</span>
          </a>
        `).join("")}
      </section>
      <section class="home-bottom">
        <div class="panel">
          <div class="panel-header"><div><h2>Local workspace</h2><p>Saved in this browser. Clear cache and it goes away.</p></div></div>
          <div class="panel-body metric-grid">
            <div class="metric"><span>${workspace.contracts.length}</span><label>contract items</label></div>
            <div class="metric"><span>${workspace.library.length}</span><label>library items</label></div>
            <div class="metric"><span>${ctxCount ? "Ready" : "Empty"}</span><label>agent context</label></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header"><div><h2>Recent conversations</h2><p>Pick up where you left off.</p></div></div>
          <div class="panel-body recent-list">
            ${recentChats.length
              ? recentChats.map(([key, h]) => {
                const [agent, mode] = key.split(":");
                const label = key === "assistant" ? "Project Assistant" : `${agentLabels[agent] || agent} · ${mode || "review"}`;
                const href = key === "assistant" ? "/sopal-v2/projects/assistant" : `/sopal-v2/agents/${agent}?mode=${mode || "review"}`;
                const last = h.messages[h.messages.length - 1] || {};
                return `<a class="recent-item" href="${href}" data-nav>
                  <strong>${escapeHtml(label)}</strong>
                  <span>${escapeHtml((last.content || "").slice(0, 110))}${(last.content || "").length > 110 ? "…" : ""}</span>
                </a>`;
              }).join("")
              : EmptyState("No conversations yet.", "Start an agent or open the project assistant.")}
          </div>
        </div>
      </section>
    `;
  }

  /* ---------- Research: decisions ---------- */

  function ResearchPage(kind) {
    if (kind === "adjudication-decisions") return AdjudicationDecisionsPage();
    if (kind === "adjudicator-statistics") return AdjudicatorStatisticsPage();
    return HomePage();
  }

  function labelSort(value) {
    return ({ relevance: "Relevance", newest: "Newest", oldest: "Oldest", claim_high: "Claimed: high to low", claim_low: "Claimed: low to high", adj_high: "Awarded: high to low", adj_low: "Awarded: low to high" })[value] || value;
  }

  function AdjudicationDecisionsPage() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q") || "";
    const sort = params.get("sort") || "relevance";

    setTimeout(() => {
      const form = document.querySelector("[data-decision-search]");
      if (form) form.addEventListener("submit", (event) => {
        event.preventDefault();
        const data = new FormData(form);
        const next = new URLSearchParams();
        if (data.get("q")) next.set("q", data.get("q"));
        next.set("sort", data.get("sort") || "relevance");
        ["startDate", "endDate", "minClaim", "maxClaim"].forEach((key) => { if (data.get(key)) next.set(key, data.get(key)); });
        navigate(`/sopal-v2/research/adjudication-decisions?${next.toString()}`);
      });
      if (params.toString() && q) fetchDecisionResults(params, 0);
    }, 0);

    const filtersOn = ["startDate", "endDate", "minClaim", "maxClaim"].some((k) => params.get(k));

    return `
      <div class="research-layout">
        <section class="panel">
          <div class="panel-header"><div><h2>Decision search</h2><p>Searches the live Sopal decision database. Sign in to run searches.</p></div></div>
          <form class="panel-body search-form" data-decision-search>
            <input class="text-input span-2" name="q" type="search" value="${escapeHtml(q)}" placeholder="Search terms, party names, adjudicator, section references" autofocus>
            <select class="select-input" name="sort">
              ${["relevance", "newest", "oldest", "claim_high", "claim_low", "adj_high", "adj_low"].map((s) => `<option value="${s}" ${sort === s ? "selected" : ""}>${labelSort(s)}</option>`).join("")}
            </select>
            <button class="dark-button" type="submit">Search</button>
            <details class="filters-toggle span-2" ${filtersOn ? "open" : ""}>
              <summary>Filters${filtersOn ? " (active)" : ""}</summary>
              <div class="filters-grid">
                <label>From<input class="text-input" name="startDate" type="date" value="${escapeHtml(params.get("startDate") || "")}"></label>
                <label>To<input class="text-input" name="endDate" type="date" value="${escapeHtml(params.get("endDate") || "")}"></label>
                <label>Min claimed<input class="text-input" name="minClaim" type="number" step="1000" value="${escapeHtml(params.get("minClaim") || "")}"></label>
                <label>Max claimed<input class="text-input" name="maxClaim" type="number" step="1000" value="${escapeHtml(params.get("maxClaim") || "")}"></label>
              </div>
            </details>
          </form>
        </section>
        <section id="decision-results">${q ? EmptyState("Searching decisions…", "Querying the real Sopal decision database.") : EmptyState("Enter a query.", "Try an adjudicator name, a party, a section reference, or keywords from a decision.")}</section>
        <aside id="decision-detail" class="panel detail-panel">${EmptyState("Select a decision.", "Click any result to view metadata and load the full decision text.")}</aside>
      </div>
    `;
  }

  async function fetchDecisionResults(params, offset) {
    const mount = document.getElementById("decision-results");
    if (!mount) return;
    mount.innerHTML = `<div class="panel"><div class="panel-body skeleton-results"><div class="skeleton-row"></div><div class="skeleton-row"></div><div class="skeleton-row"></div></div></div>`;
    const qs = new URLSearchParams(params);
    qs.set("limit", "20");
    qs.set("offset", String(offset || 0));
    try {
      const response = await fetch(`/search_fast?${qs.toString()}`, { headers: authHeaders(), credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (response.status === 401) {
        mount.innerHTML = `
          <div class="panel">
            <div class="panel-body">
              ${EmptyState("Sign in to search.", "Searching the decision database requires an account.", `<button class="dark-button" type="button" data-sign-in>Sign in</button> <a class="link-button" href="https://sopal.com.au/login" target="_blank" rel="noopener">Create account</a>`)}
            </div>
          </div>`;
        return;
      }
      if (response.status === 429) {
        mount.innerHTML = `<div class="error-banner">Search limit reached for this month. ${escapeHtml(data.message || "")}</div>`;
        return;
      }
      if (!response.ok) throw new Error(data.message || data.detail || data.error || "Search failed");
      const items = Array.isArray(data.items) ? data.items : [];
      const total = Number(data.total || items.length);
      if (!items.length) {
        mount.innerHTML = EmptyState("No decisions match.", "Adjust your query or filters.");
        return;
      }
      mount.innerHTML = `
        <div class="panel">
          <div class="panel-header">
            <div><h2>${total.toLocaleString()} result${total === 1 ? "" : "s"}</h2><p>Live records from the Sopal decision database.</p></div>
          </div>
          <div class="panel-body results-list">${items.map(renderDecisionItem).join("")}</div>
          ${items.length < total ? `<div class="panel-footer"><button class="ghost-button" type="button" data-load-more="${(offset || 0) + items.length}">Load more (${(total - ((offset || 0) + items.length)).toLocaleString()} remaining)</button></div>` : ""}
        </div>`;
      mount.querySelectorAll("[data-decision-id]").forEach((el) => el.addEventListener("click", () => loadDecisionDetail(el.dataset.decisionId, el.dataset.title)));
      const more = mount.querySelector("[data-load-more]");
      if (more) more.addEventListener("click", () => appendDecisionResults(params, Number(more.dataset.loadMore)));
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Search failed")}</div>`;
    }
  }

  async function appendDecisionResults(params, offset) {
    const mount = document.getElementById("decision-results");
    if (!mount) return;
    const list = mount.querySelector(".results-list");
    const footer = mount.querySelector(".panel-footer");
    if (!list) return;
    const qs = new URLSearchParams(params);
    qs.set("limit", "20");
    qs.set("offset", String(offset));
    try {
      const response = await fetch(`/search_fast?${qs.toString()}`, { headers: authHeaders(), credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Search failed");
      const items = Array.isArray(data.items) ? data.items : [];
      list.insertAdjacentHTML("beforeend", items.map(renderDecisionItem).join(""));
      list.querySelectorAll("[data-decision-id]:not([data-bound])").forEach((el) => {
        el.dataset.bound = "1";
        el.addEventListener("click", () => loadDecisionDetail(el.dataset.decisionId, el.dataset.title));
      });
      const total = Number(data.total || 0);
      const newOffset = offset + items.length;
      if (newOffset >= total || !items.length) { if (footer) footer.remove(); }
      else if (footer) footer.querySelector("[data-load-more]").dataset.loadMore = String(newOffset);
    } catch (error) {
      if (footer) footer.innerHTML = `<div class="error-banner">${escapeHtml(error.message)}</div>`;
    }
  }

  function renderDecisionItem(item) {
    const claimant = item.claimant_name || item.claimant || "";
    const respondent = item.respondent_name || item.respondent || "";
    const title = [claimant, respondent].filter(Boolean).join(" v ") || item.reference || item.ejs_id || "Decision";
    const id = item.ejs_id || item.id || "";
    const meta = [
      item.decision_date || item.decision_date_norm,
      item.adjudicator_name || item.adjudicator,
      item.act_category || item.act,
      money(item.claimed_amount) ? `${money(item.claimed_amount)} claimed` : "",
      money(item.adjudicated_amount) ? `${money(item.adjudicated_amount)} awarded` : "",
    ].filter(Boolean);
    return `
      <article class="result-item clickable" data-decision-id="${escapeHtml(id)}" data-title="${escapeHtml(title)}" tabindex="0">
        <h3>${escapeHtml(title)}</h3>
        <div class="result-meta">${meta.map((m) => `<span>${escapeHtml(m)}</span>`).join("")}</div>
        <p>${formatSnippet(item.snippet)}</p>
      </article>`;
  }

  async function loadDecisionDetail(id, title) {
    const mount = document.getElementById("decision-detail");
    if (!mount || !id) return;
    mount.innerHTML = `<div class="panel-header"><div><h2>${escapeHtml(title || "Decision")}</h2><p>Loading decision text…</p></div></div><div class="panel-body"><div class="skeleton-row"></div><div class="skeleton-row"></div><div class="skeleton-row"></div></div>`;
    try {
      const response = await fetch(`/api/decision-text/${encodeURIComponent(id)}`, { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Decision text failed");
      const text = (data.fullText || "").trim();
      mount.innerHTML = `
        <div class="panel-header">
          <div><h2>${escapeHtml(title || id)}</h2><p>${escapeHtml(id)}</p></div>
          <div class="panel-actions">
            <button class="ghost-button compact" type="button" data-copy-text="${escapeHtml(text.slice(0, 8000))}">Copy text</button>
            <a class="link-button" href="/open?id=${encodeURIComponent(id)}" target="_blank" rel="noopener">Open original</a>
          </div>
        </div>
        <div class="panel-body">
          ${text ? `<div class="decision-text">${escapeHtml(text.slice(0, 12000))}${text.length > 12000 ? "\n\n[Text truncated. Open original for the full record.]" : ""}</div>` : EmptyState("No text on file.", "This decision has no extracted text. Open the original PDF.")}
        </div>`;
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load decision text")}</div>`;
    }
  }

  /* ---------- Research: adjudicator stats ---------- */

  function AdjudicatorStatisticsPage() {
    setTimeout(fetchAdjudicators, 0);
    return `
      <div class="section-stack">
        <div class="panel">
          <div class="panel-header"><div><h2>Adjudicator statistics</h2><p>Live data from the Sopal decision database.</p></div></div>
          <div class="panel-body toolbar">
            <input class="text-input" data-adj-filter placeholder="Filter adjudicators by name">
            <select class="select-input" data-adj-sort>
              <option value="decisions">Most decisions</option>
              <option value="award">Highest award rate</option>
              <option value="award_low">Lowest award rate</option>
              <option value="claimed">Total claimed</option>
              <option value="awarded">Total awarded</option>
              <option value="zeroes">Most $0 awards</option>
            </select>
          </div>
        </div>
        <div class="stats-grid-shell">
          <section id="adjudicator-results">${EmptyState("Loading statistics…", "Querying /api/adjudicators.")}</section>
          <aside id="adjudicator-detail" class="panel detail-panel">${EmptyState("Select an adjudicator.", "Click a card to view their decision history and totals.")}</aside>
        </div>
      </div>`;
  }

  async function fetchAdjudicators() {
    const mount = document.getElementById("adjudicator-results");
    if (!mount) return;
    try {
      const response = await fetch("/api/adjudicators", { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.error || "Adjudicator endpoint failed");
      window.__sopalAdjudicators = Array.isArray(data) ? data : [];
      renderAdjudicators();
      document.querySelector("[data-adj-filter]")?.addEventListener("input", renderAdjudicators);
      document.querySelector("[data-adj-sort]")?.addEventListener("change", renderAdjudicators);
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load adjudicators")}</div>`;
    }
  }

  function renderAdjudicators() {
    const mount = document.getElementById("adjudicator-results");
    if (!mount) return;
    const filter = (document.querySelector("[data-adj-filter]")?.value || "").toLowerCase();
    const sort = document.querySelector("[data-adj-sort]")?.value || "decisions";
    const all = window.__sopalAdjudicators || [];
    let items = all.filter((a) => a.name && a.name.toLowerCase().includes(filter));
    items = items.sort((a, b) => ({
      decisions: b.totalDecisions - a.totalDecisions,
      award: b.avgAwardRate - a.avgAwardRate,
      award_low: a.avgAwardRate - b.avgAwardRate,
      claimed: b.totalClaimAmount - a.totalClaimAmount,
      awarded: b.totalAwardedAmount - a.totalAwardedAmount,
      zeroes: (b.zeroAwardCount || 0) - (a.zeroAwardCount || 0),
    }[sort]));
    mount.innerHTML = items.length
      ? `<div class="adjudicator-grid">${items.slice(0, 80).map((item) => `
          <button class="stat-card-btn" type="button" data-adjudicator="${escapeHtml(item.name)}">
            <strong>${escapeHtml(item.name)}</strong>
            <span>${item.totalDecisions} decisions</span>
            <span>${formatCurrency(item.totalClaimAmount)} claimed</span>
            <span>${formatCurrency(item.totalAwardedAmount)} awarded</span>
            <span class="rate-pill">${formatPercent(item.avgAwardRate)} avg award</span>
          </button>`).join("")}</div>`
      : EmptyState("No adjudicators match.", "Clear or change the filter.");
    mount.querySelectorAll("[data-adjudicator]").forEach((b) => b.addEventListener("click", () => loadAdjudicatorDetail(b.dataset.adjudicator)));
  }

  async function loadAdjudicatorDetail(name) {
    const mount = document.getElementById("adjudicator-detail");
    if (!mount) return;
    mount.innerHTML = `<div class="panel-header"><div><h2>${escapeHtml(name)}</h2><p>Loading decisions…</p></div></div><div class="panel-body"><div class="skeleton-row"></div><div class="skeleton-row"></div><div class="skeleton-row"></div></div>`;
    try {
      const response = await fetch(`/api/adjudicator/${encodeURIComponent(name)}`, { headers: authHeaders(), credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.error || "Adjudicator detail failed");
      const decisions = Array.isArray(data) ? data : [];
      const summary = (window.__sopalAdjudicators || []).find((x) => x.name === name) || {};
      const claimedSum = decisions.reduce((s, d) => s + (Number(d.claimAmount) || 0), 0);
      const awardedSum = decisions.reduce((s, d) => s + (Number(d.awardedAmount) || 0), 0);
      const zeroes = decisions.filter((d) => Number(d.awardedAmount) === 0).length;
      mount.innerHTML = `
        <div class="panel-header">
          <div><h2>${escapeHtml(name)}</h2><p>${decisions.length} decision${decisions.length === 1 ? "" : "s"}</p></div>
          <a class="link-button" href="/adjudicators" target="_blank" rel="noopener">Open full page</a>
        </div>
        <div class="panel-body">
          <div class="metric-grid compact">
            <div class="metric"><span>${decisions.length}</span><label>decisions</label></div>
            <div class="metric"><span>${formatCurrency(claimedSum)}</span><label>total claimed</label></div>
            <div class="metric"><span>${formatCurrency(awardedSum)}</span><label>total awarded</label></div>
            <div class="metric"><span>${formatPercent(summary.avgAwardRate || 0)}</span><label>avg award rate</label></div>
            <div class="metric"><span>${zeroes}</span><label>$0 awards</label></div>
            <div class="metric"><span>${formatPercent(summary.avgClaimantFeeProportion || 0)}</span><label>claimant fee share</label></div>
          </div>
          <div class="mini-list">
            ${decisions.slice(0, 25).map((d) => `
              <article ${d.id ? `class="clickable" data-decision-id="${escapeHtml(d.id)}" data-title="${escapeHtml(d.title || "")}" tabindex="0"` : ""}>
                <strong>${escapeHtml(d.title || "Decision")}</strong>
                <span>${escapeHtml(d.date || "")}${d.outcome ? ` · ${escapeHtml(d.outcome)}` : ""}${d.projectType ? ` · ${escapeHtml(d.projectType)}` : ""}</span>
                <span>claimed ${formatCurrency(d.claimAmount)} · awarded ${formatCurrency(d.awardedAmount)}</span>
              </article>`).join("")}
            ${decisions.length > 25 ? `<div class="muted">${decisions.length - 25} more not shown — open the full page.</div>` : ""}
          </div>
        </div>`;
      mount.querySelectorAll("[data-decision-id]").forEach((el) => el.addEventListener("click", () => {
        navigate(`/sopal-v2/research/adjudication-decisions?q=${encodeURIComponent(el.dataset.title || "")}`);
      }));
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load adjudicator")}</div>`;
    }
  }

  /* ---------- Tools: due date + interest calculators ---------- */

  function ToolPage(kind) {
    return kind === "interest-calculator" ? InterestCalculator() : DueDateCalculator();
  }

  function DueDateCalculator() {
    setTimeout(bindDueDateCalculator, 0);
    return `
      <div class="tool-grid">
        <section class="panel">
          <div class="panel-header"><div><h2>BIF Act due date</h2><p>Native business-day calculator with Queensland holiday handling.</p></div></div>
          <form class="panel-body calc-form" data-due-form>
            <label>Scenario
              <select class="select-input" name="scenario">
                <option value="paymentSchedule">Payment schedule due date (s 76)</option>
                <option value="adjudicationAppLess">Adjudication application — schedule less than claimed (s 79(2)(b)(iii))</option>
                <option value="adjudicationAppNoPayAmount">Adjudication application — scheduled amount unpaid (s 79(2)(b)(ii))</option>
                <option value="adjudicationAppNoSchedule">Adjudication application — no schedule and no payment (s 79(2)(b)(i))</option>
                <option value="adjudicationResponseStandard">Adjudication response — standard claim (s 83)</option>
                <option value="adjudicationResponseComplex">Adjudication response — complex claim (s 83)</option>
                <option value="adjudicatorDecisionStandard">Adjudicator decision — standard claim (s 85)</option>
                <option value="adjudicatorDecisionComplex">Adjudicator decision — complex claim (s 85)</option>
              </select>
            </label>
            <label>Location
              <select class="select-input" name="location">
                <option value="qld">Queensland statewide</option>
                <option value="bne">Brisbane</option>
                <option value="gld">Gold Coast</option>
                <option value="cns">Cairns</option>
                <option value="tsw">Townsville</option>
                <option value="ipswich">Ipswich</option>
                <option value="toowoomba">Toowoomba</option>
                <option value="sunshine_coast">Sunshine Coast</option>
                <option value="rockhampton">Rockhampton</option>
                <option value="mackay">Mackay</option>
              </select>
            </label>
            <label>Primary start date<input class="text-input" name="startDate" type="date"></label>
            <label>Second date (where required)<input class="text-input" name="secondDate" type="date"></label>
            <label>Extension days (where applicable)<input class="text-input" name="eotDays" type="number" min="0" max="30" value="0"></label>
            <button class="dark-button span-2" type="submit">Calculate</button>
          </form>
        </section>
        <section class="panel">
          <div class="panel-header"><div><h2>Result</h2><p>Excludes weekends, QLD/local public holidays and the s 87 Christmas shutdown.</p></div></div>
          <div class="panel-body" id="due-result">${EmptyState("No calculation yet.", "Choose a scenario and enter the relevant dates.")}</div>
        </section>
      </div>`;
  }

  function bindDueDateCalculator() {
    const form = document.querySelector("[data-due-form]");
    if (!form) return;
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const data = Object.fromEntries(new FormData(form).entries());
      const result = calculateDueDate(data);
      const mount = document.getElementById("due-result");
      mount.innerHTML = result.error ? `<div class="error-banner">${escapeHtml(result.error)}</div>` : renderDateResult(result);
    });
  }

  function calculateDueDate(data) {
    const scenario = data.scenario;
    const location = data.location || "qld";
    const start = parseDate(data.startDate);
    const second = parseDate(data.secondDate);
    const eotDays = Math.max(0, parseInt(data.eotDays || "0", 10) || 0);
    if (!start) return { error: "Primary start date is required." };
    let basis = "", days = 0, startDate = start, eot = 0;
    if (scenario === "paymentSchedule") { days = 15; basis = "15 business days after the payment claim is given (s 76 BIF Act)."; }
    if (scenario === "adjudicationAppLess") { days = 30; basis = "30 business days after the payment schedule is received (s 79(2)(b)(iii))."; }
    if (scenario === "adjudicationAppNoPayAmount") { days = 20; basis = "20 business days after the due date for payment (s 79(2)(b)(ii))."; }
    if (scenario === "adjudicationAppNoSchedule") {
      if (!second) return { error: "Second date is required for the later of payment due date and schedule due date." };
      startDate = new Date(Math.max(start.getTime(), second.getTime()));
      days = 30; basis = "30 business days after the later of the payment due date or payment schedule due date (s 79(2)(b)(i)).";
    }
    if (scenario === "adjudicationResponseStandard" || scenario === "adjudicationResponseComplex") {
      if (!second) return { error: "Second date is required: application documents received date and acceptance received date." };
      const appDays = scenario === "adjudicationResponseStandard" ? 10 : 15;
      const acceptanceDays = scenario === "adjudicationResponseStandard" ? 7 : 12;
      const appResult = addBusinessDays(start, appDays, location);
      const acceptanceResult = addBusinessDays(second, acceptanceDays, location);
      const laterIsApp = appResult.finalDate.getTime() >= acceptanceResult.finalDate.getTime();
      eot = scenario === "adjudicationResponseComplex" ? Math.min(eotDays, 15) : 0;
      const picked = laterIsApp ? appResult : acceptanceResult;
      const extension = eot ? addBusinessDays(picked.finalDate, eot, location) : picked;
      return {
        title: "Adjudication response due date",
        startDate: laterIsApp ? start : second,
        days: laterIsApp ? appDays : acceptanceDays,
        eot,
        finalDate: extension.finalDate,
        skipped: picked.skipped.concat(eot ? extension.skipped : []),
        basis: `Later of ${appDays} business days after receiving application documents, or ${acceptanceDays} business days after receiving notice of acceptance (s 83 BIF Act).${eot ? " Complex-claim extension applied." : ""}`,
      };
    }
    if (scenario === "adjudicatorDecisionStandard") { days = 10; eot = eotDays; basis = "10 business days after adjudication response is given, plus any agreed extension (s 85)."; }
    if (scenario === "adjudicatorDecisionComplex") { days = 15; eot = eotDays; basis = "15 business days after adjudication response is given, plus any agreed extension (s 85)."; }
    const base = addBusinessDays(startDate, days, location);
    const final = eot ? addBusinessDays(base.finalDate, eot, location) : base;
    return { title: labelScenario(scenario), startDate, days, eot, finalDate: final.finalDate, skipped: base.skipped.concat(eot ? final.skipped : []), basis };
  }

  function labelScenario(value) {
    return ({
      paymentSchedule: "Payment schedule due date",
      adjudicationAppLess: "Adjudication application due date",
      adjudicationAppNoPayAmount: "Adjudication application due date",
      adjudicationAppNoSchedule: "Adjudication application due date",
      adjudicationResponseStandard: "Adjudication response due date",
      adjudicationResponseComplex: "Adjudication response due date",
      adjudicatorDecisionStandard: "Adjudicator decision due date",
      adjudicatorDecisionComplex: "Adjudicator decision due date",
    })[value] || "Due date";
  }

  function isBusinessDay(date, location) {
    const day = date.getDay();
    if (day === 0 || day === 6) return { isBiz: false, reason: "Weekend" };
    const month = date.getMonth(), dayOfMonth = date.getDate();
    if ((month === 11 && dayOfMonth >= 22 && dayOfMonth <= 24) || (month === 11 && dayOfMonth >= 27 && dayOfMonth <= 31) || (month === 0 && dayOfMonth >= 2 && dayOfMonth <= 10)) return { isBiz: false, reason: "Christmas shutdown (s 87)" };
    const dateString = date.toISOString().slice(0, 10);
    const publicHoliday = (holidays.qld || []).concat(holidays[location] || []).find((h) => h.date === dateString);
    return publicHoliday ? { isBiz: false, reason: publicHoliday.name } : { isBiz: true, reason: "" };
  }

  function addBusinessDays(startDate, days, location) {
    const currentDate = new Date(startDate.getTime());
    let daysAdded = 0; const skipped = [];
    currentDate.setDate(currentDate.getDate() + 1);
    while (daysAdded < days) {
      const check = isBusinessDay(currentDate, location);
      if (check.isBiz) daysAdded++;
      else skipped.push({ date: new Date(currentDate.getTime()), reason: check.reason });
      if (daysAdded < days) currentDate.setDate(currentDate.getDate() + 1);
    }
    return { finalDate: currentDate, skipped };
  }

  function renderDateResult(result) {
    const skippedSummary = summarizeSkipped(result.skipped);
    const copyText = `${result.title}: ${formatDate(result.finalDate)}\n${result.basis}`;
    return `<div class="calc-result">
      <span class="calc-result-tag">${escapeHtml(result.title)}</span>
      <strong>${formatDate(result.finalDate)}</strong>
      <p>${escapeHtml(result.basis)}</p>
      <dl>
        <dt>Start date</dt><dd>${formatDate(result.startDate)}</dd>
        <dt>Business-day period</dt><dd>${result.days}${result.eot ? ` + ${result.eot} extension days` : ""}</dd>
        <dt>Non-business days skipped</dt><dd>${escapeHtml(skippedSummary)}</dd>
      </dl>
      <button class="ghost-button compact" data-copy-text="${escapeHtml(copyText)}">Copy</button>
    </div>`;
  }

  function summarizeSkipped(skipped) {
    if (!skipped.length) return "None";
    const grouped = {};
    skipped.forEach((d) => { grouped[d.reason] = (grouped[d.reason] || 0) + 1; });
    return Object.entries(grouped).map(([reason, count]) => `${count} ${reason}`).join(", ");
  }

  function InterestCalculator() {
    setTimeout(bindInterestCalculator, 0);
    const today = new Date().toISOString().slice(0, 10);
    return `
      <div class="tool-grid">
        <section class="panel">
          <div class="panel-header"><div><h2>Interest</h2><p>QBCC s 67P uses live RBA cash rate. Contractual mode uses your annual rate.</p></div></div>
          <form class="panel-body calc-form" data-interest-form>
            <label>Rate type
              <select class="select-input" name="type">
                <option value="qbcc">QBCC Act s 67P (10% + RBA cash rate)</option>
                <option value="contractual">Contractual rate</option>
              </select>
            </label>
            <label>Principal amount<input class="text-input" name="principal" type="number" min="0" step="0.01" placeholder="Amount unpaid"></label>
            <label>Due date<input class="text-input" name="startDate" type="date"></label>
            <label>Calculation date<input class="text-input" name="endDate" type="date" value="${today}"></label>
            <label>Contractual annual rate %<input class="text-input" name="annualRate" type="number" min="0" step="0.01" value="10"></label>
            <button class="dark-button span-2" type="submit">Calculate</button>
          </form>
        </section>
        <section class="panel">
          <div class="panel-header"><div><h2>Result</h2><p>Daily compounded calculation including both endpoints.</p></div></div>
          <div class="panel-body" id="interest-result">${EmptyState("No calculation yet.", "Enter the unpaid amount and dates.")}</div>
        </section>
      </div>`;
  }

  function bindInterestCalculator() {
    const form = document.querySelector("[data-interest-form]");
    if (!form) return;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const mount = document.getElementById("interest-result");
      mount.innerHTML = `<div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Calculating…</span></div>`;
      const data = Object.fromEntries(new FormData(form).entries());
      try {
        const result = await calculateInterest(data);
        mount.innerHTML = renderInterestResult(result);
      } catch (error) {
        mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Interest calculation failed")}</div>`;
      }
    });
  }

  async function calculateInterest(data) {
    const principal = Number(data.principal);
    const startDate = parseDate(data.startDate);
    const endDate = parseDate(data.endDate);
    if (!principal || !startDate || !endDate) throw new Error("Principal and valid dates are required.");
    if (endDate < startDate) throw new Error("Calculation date must be after the due date.");
    const days = Math.ceil((endDate.getTime() - startDate.getTime()) / 86400000) + 1;
    if (data.type === "contractual") {
      const annualRate = Number(data.annualRate);
      if (Number.isNaN(annualRate)) throw new Error("Contractual rate is required.");
      return { type: "Contractual", principal, days, interest: principal * ((annualRate / 365) / 100) * days, annualRate, startDate, endDate };
    }
    const response = await fetch(`/get_interest_rate?startDate=${startDate.toISOString().slice(0, 10)}&endDate=${endDate.toISOString().slice(0, 10)}`, { credentials: "include" });
    const ratesData = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(ratesData.detail || "Could not fetch RBA rates.");
    let interest = 0;
    (ratesData.dailyRates || []).forEach((row) => { interest += (principal / 365) * ((10 + Number(row.rate)) / 100); });
    const rates = (ratesData.dailyRates || []).map((row) => Number(row.rate));
    return { type: "QBCC Act s 67P", principal, days, interest, startDate, endDate, minRate: rates.length ? Math.min(...rates) : 0, maxRate: rates.length ? Math.max(...rates) : 0, dailyRates: ratesData.dailyRates || [] };
  }

  function renderInterestResult(result) {
    const total = result.principal + result.interest;
    const rateLine = result.type === "QBCC Act s 67P" ? `10% + RBA rate (${result.minRate.toFixed(2)}% to ${result.maxRate.toFixed(2)}%)` : `${result.annualRate.toFixed(2)}% contractual`;
    const copyText = `Interest: ${formatCurrency(result.interest)}\nTotal: ${formatCurrency(total)}\nRate: ${rateLine}`;
    return `<div class="calc-result">
      <span class="calc-result-tag">${escapeHtml(result.type)}</span>
      <strong>${formatCurrency(result.interest)}</strong>
      <p>Interest on ${formatCurrency(result.principal)} over ${result.days} days. Total: ${formatCurrency(total)}.</p>
      <dl>
        <dt>Rate</dt><dd>${escapeHtml(rateLine)}</dd>
        <dt>Due date</dt><dd>${formatDate(result.startDate)}</dd>
        <dt>Calculation date</dt><dd>${formatDate(result.endDate)}</dd>
      </dl>
      <button class="ghost-button compact" data-copy-text="${escapeHtml(copyText)}">Copy</button>
      ${result.dailyRates && result.dailyRates.length ? `<details class="breakdown"><summary>Daily breakdown (${result.dailyRates.length} rows)</summary>
        <table>
          <thead><tr><th>Date</th><th>RBA</th><th>Effective</th><th>Daily interest</th></tr></thead>
          <tbody>${result.dailyRates.slice(0, 200).map((row) => `<tr><td>${escapeHtml(row.date)}</td><td>${Number(row.rate).toFixed(2)}%</td><td>${(10 + Number(row.rate)).toFixed(2)}%</td><td>${formatCurrency((result.principal / 365) * ((10 + Number(row.rate)) / 100))}</td></tr>`).join("")}</tbody>
        </table>
      </details>` : ""}
    </div>`;
  }

  /* ---------- Projects ---------- */

  function ProjectPage(kind) {
    if (kind === "assistant") {
      return `<div class="section-stack">${ContextSummary()}${ChatPanel({
        endpoint: "/api/sopal-v2/chat",
        assistant: true,
        placeholder: "Ask about project facts, contract wording, notices, claims, schedules or correspondence.",
        emptyTitle: "Project assistant",
        emptyBody: "Ask a question. Toggle project context on to include local extracted/pasted context.",
      })}</div>`;
    }
    return ContextManager(kind === "contracts" ? "contracts" : "library");
  }

  function ContextSummary() {
    const total = workspace.contracts.length + workspace.library.length;
    return `<section class="panel compact-panel">
      <div class="panel-header">
        <div><h2>Available local context</h2><p>${workspace.contracts.length} contract item${workspace.contracts.length === 1 ? "" : "s"} · ${workspace.library.length} project library item${workspace.library.length === 1 ? "" : "s"}</p></div>
        <div class="panel-actions">
          ${total ? `<button class="ghost-button compact" type="button" data-toggle-context-preview>Preview context</button>` : ""}
          <a class="link-button" href="/sopal-v2/projects/contracts" data-nav>Manage</a>
        </div>
      </div>
      <div class="panel-body context-preview" data-context-preview hidden>
        ${total
          ? [...workspace.contracts.map((d) => ({ ...d, kind: "contract" })), ...workspace.library.map((d) => ({ ...d, kind: "library" }))]
              .map((d) => `<details><summary><strong>${escapeHtml(d.name)}</strong> <span class="muted">${d.kind} · ${d.text.length.toLocaleString()} chars</span></summary><pre>${escapeHtml(d.text.slice(0, 4000))}${d.text.length > 4000 ? "\n…" : ""}</pre></details>`).join("")
          : EmptyState("No context yet.", "Add contracts or project library items to make the assistant context-aware.")}
      </div>
    </section>`;
  }

  function ContextManager(bucket) {
    const label = bucket === "contracts" ? "Contracts" : "Project Library";
    const helper = bucket === "contracts"
      ? "Paste contract clauses or extract text from PDF/DOCX/TXT. Becomes browser-local context for the assistant and agents."
      : "Paste RFIs, correspondence, notices, claims, schedules, programme notes, or extract text from PDF/DOCX/TXT.";
    setTimeout(() => bindContextManager(bucket), 0);
    return `
      <div class="context-layout">
        <section class="panel">
          <div class="panel-header"><div><h2>${escapeHtml(label)}</h2><p>${escapeHtml(helper)}</p></div></div>
          <form class="panel-body context-form" data-context-form="${bucket}">
            <label>Label<input class="text-input" name="name" placeholder="Document or context name"></label>
            <label class="span-2">Paste text<textarea class="text-area" name="text" placeholder="Paste clauses, correspondence, claim text, schedule text, or facts."></textarea></label>
            <div class="file-dropzone span-2">
              <label>Or extract from PDF/DOCX/TXT<input type="file" data-context-file accept=".pdf,.docx,.txt"></label>
              <div class="file-list" data-context-file-status>No file selected.</div>
            </div>
            <button class="dark-button span-2" type="submit">Save to local context</button>
          </form>
        </section>
        <section class="panel">
          <div class="panel-header">
            <div><h2>Saved context</h2><p>Stored in this browser's localStorage.</p></div>
            ${(workspace[bucket] || []).length ? `<button class="ghost-button compact" type="button" data-clear-context="${bucket}">Clear all</button>` : ""}
          </div>
          <div class="panel-body context-list">${renderContextList(bucket)}</div>
        </section>
      </div>`;
  }

  function renderContextList(bucket) {
    const items = workspace[bucket] || [];
    if (!items.length) return EmptyState(`No ${bucket === "contracts" ? "contracts" : "project documents"} saved.`, "Add pasted or extracted text to make agents context-aware.");
    return items.map((item, index) => `
      <article class="context-item">
        <div class="context-item-head">
          <strong>${escapeHtml(item.name)}</strong>
          <span class="muted">${item.text.length.toLocaleString()} chars · ${escapeHtml(item.source || "pasted")}</span>
        </div>
        <details><summary>Preview</summary><pre>${escapeHtml(item.text.slice(0, 2000))}${item.text.length > 2000 ? "\n…" : ""}</pre></details>
        <div class="context-item-actions">
          <button class="ghost-button compact" data-copy-text="${escapeHtml(item.text.slice(0, 8000))}" type="button">Copy</button>
          <button class="ghost-button compact danger" data-remove-context="${bucket}:${index}" type="button">Remove</button>
        </div>
      </article>`).join("");
  }

  function bindContextManager(bucket) {
    const form = document.querySelector(`[data-context-form="${bucket}"]`);
    if (!form) return;
    const fileInput = form.querySelector("[data-context-file]");
    const status = form.querySelector("[data-context-file-status]");
    let extracted = null;
    fileInput?.addEventListener("change", async () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      status.textContent = "Extracting text…";
      const fd = new FormData(); fd.append("file", file);
      try {
        const response = await fetch("/api/sopal-v2/extract", { method: "POST", body: fd, credentials: "include" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "Extraction failed");
        extracted = data;
        if (!form.elements.name.value) form.elements.name.value = data.filename;
        form.elements.text.value = [form.elements.text.value, data.text].filter(Boolean).join("\n\n");
        status.textContent = `${data.filename}: ${data.characters.toLocaleString()} chars extracted${data.truncated ? " (truncated)" : ""}.`;
      } catch (error) {
        status.textContent = error.message || "Extraction failed";
      }
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const data = Object.fromEntries(new FormData(form).entries());
      if (!data.text || !String(data.text).trim()) return;
      workspace[bucket].push({
        name: String(data.name || extracted?.filename || "Untitled context"),
        text: String(data.text).trim(),
        source: extracted ? "extracted file + paste" : "pasted",
        createdAt: new Date().toISOString(),
      });
      saveWorkspace();
      render();
    });
  }

  function allContextText() {
    const contractText = workspace.contracts.map((d) => `Contract: ${d.name}\n${d.text}`).join("\n\n---\n\n");
    const libraryText = workspace.library.map((d) => `Project document: ${d.name}\n${d.text}`).join("\n\n---\n\n");
    return [contractText, libraryText].filter(Boolean).join("\n\n===\n\n").slice(0, 40000);
  }

  /* ---------- Agents ---------- */

  function AgentPage(agentKey) {
    const params = new URLSearchParams(window.location.search);
    const mode = params.get("mode") === "draft" ? "draft" : "review";
    const label = agentLabels[agentKey] || "Agent";
    const starters = scenarioStarters[`${agentKey}:${mode}`] || [];
    return `
      <div class="agent-layout">
        <section class="panel agent-brief">
          <div class="panel-header">
            <div><h2>${escapeHtml(label)}</h2><p>${escapeHtml(agentDescriptions[agentKey] || "")}</p></div>
            <div class="mode-tabs" role="tablist" aria-label="Agent mode">
              <button class="mode-tab ${mode === "review" ? "active" : ""}" data-go="/sopal-v2/agents/${agentKey}?mode=review" type="button">Review</button>
              <button class="mode-tab ${mode === "draft" ? "active" : ""}" data-go="/sopal-v2/agents/${agentKey}?mode=draft" type="button">Draft</button>
            </div>
          </div>
          <div class="panel-body helper-grid">
            <div>
              <strong>What to include</strong>
              <ul>${(includeLists[agentKey] || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
            </div>
            <div>
              <strong>Quick starters</strong>
              <div class="chip-row">
                ${starters.map((s) => `<button class="chip" type="button" data-starter="${escapeHtml(s)}">${escapeHtml(s.length > 80 ? s.slice(0, 78) + "…" : s)}</button>`).join("")}
              </div>
            </div>
          </div>
        </section>
        ${ChatPanel({
          endpoint: "/api/sopal-v2/agent",
          agentType: agentKey,
          mode,
          placeholder: mode === "review"
            ? "Paste the document, key dates, contract clauses, and facts to review."
            : "Describe what needs drafting and paste the relevant project/contract facts.",
          emptyTitle: `${label} — ${mode === "review" ? "Review" : "Draft"}`,
          emptyBody: "Submit pasted/extracted material for an AI response. Toggle local project context if relevant.",
        })}
      </div>`;
  }

  /* ---------- Chat panel ---------- */

  function ChatPanel(options) {
    const id = `chat-${Math.random().toString(36).slice(2)}`;
    setTimeout(() => bindChatPanel(id, options), 0);
    const contextAvailable = workspace.contracts.length + workspace.library.length > 0;
    const key = chatKey(options);
    const history = (chatHistory[key] && chatHistory[key].messages) || [];
    return `
      <section class="chat-panel" id="${id}" data-chat-key="${escapeHtml(key)}">
        <div class="message-area" data-message-area>
          <div class="message-stack" data-messages>
            ${history.length ? history.map((m) => renderMessage(m.role === "user" ? "You" : "Sopal", m.content, m.role, m.role === "assistant")).join("") : EmptyState(options.emptyTitle || "No conversation yet.", options.emptyBody || "Start with typed instructions or pasted text.")}
          </div>
        </div>
        <form class="composer" data-chat-form>
          <div class="composer-inner">
            <div class="composer-options">
              <label class="check"><input type="checkbox" name="useContext" ${contextAvailable ? "checked" : ""} ${contextAvailable ? "" : "disabled"}> Project context (${workspace.contracts.length + workspace.library.length})</label>
              <label class="file-inline"><span class="ghost-button compact">Attach file</span><input type="file" data-chat-file accept=".pdf,.docx,.txt"></label>
              <span class="muted" data-chat-file-status></span>
            </div>
            <textarea class="text-area auto-grow" name="message" placeholder="${escapeHtml(options.placeholder || "Type your message")}" rows="3"></textarea>
            <div class="composer-footer">
              <span class="kbd-hint muted">⌘/Ctrl + Enter to send</span>
              <div>
                <button class="ghost-button compact" type="button" data-clear-chat>Clear</button>
                <button class="send-button" type="submit">Send</button>
              </div>
            </div>
            <div class="status-line" data-status></div>
          </div>
        </form>
      </section>`;
  }

  function bindChatPanel(id, options) {
    const panel = document.getElementById(id);
    if (!panel) return;
    const form = panel.querySelector("[data-chat-form]");
    const messageArea = panel.querySelector("[data-message-area]");
    const messages = panel.querySelector("[data-messages]");
    const status = panel.querySelector("[data-status]");
    const textarea = form.elements.message;
    const fileInput = panel.querySelector("[data-chat-file]");
    const fileStatus = panel.querySelector("[data-chat-file-status]");
    const key = panel.dataset.chatKey;
    let extractedFile = null;

    autoGrow(textarea);
    textarea.addEventListener("input", () => autoGrow(textarea));
    textarea.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    fileInput?.addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      fileStatus.textContent = "Extracting…";
      const fd = new FormData(); fd.append("file", file);
      try {
        const response = await fetch("/api/sopal-v2/extract", { method: "POST", body: fd, credentials: "include" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "Extraction failed");
        extractedFile = data;
        textarea.value = [textarea.value, `\n\nExtracted text from ${data.filename}:\n${data.text}`].filter(Boolean).join("\n");
        autoGrow(textarea);
        fileStatus.textContent = `${data.filename}: ${data.characters.toLocaleString()} chars`;
      } catch (error) {
        fileStatus.textContent = error.message || "Extraction failed";
      }
    });

    panel.querySelector("[data-clear-chat]")?.addEventListener("click", () => {
      delete chatHistory[key];
      saveChatHistory();
      messages.innerHTML = EmptyState(options.emptyTitle || "No conversation yet.", options.emptyBody || "Start with typed instructions or pasted text.");
      status.textContent = "";
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = textarea.value.trim();
      if (!message) { status.textContent = "Enter text before sending."; return; }

      // Append to UI + history
      if (messages.querySelector(".empty-state")) messages.innerHTML = "";
      messages.insertAdjacentHTML("beforeend", renderMessage("You", message, "user"));
      const placeholderId = `msg-${Math.random().toString(36).slice(2)}`;
      messages.insertAdjacentHTML("beforeend", `
        <div class="message assistant" id="${placeholderId}">
          <div class="avatar">S</div>
          <div class="message-body">
            <div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is thinking…</span></div>
          </div>
        </div>`);
      scrollMessagesToBottom(messageArea);

      const userEntry = { role: "user", content: message, at: Date.now() };
      chatHistory[key] = chatHistory[key] || { messages: [] };
      chatHistory[key].messages.push(userEntry);
      chatHistory[key].updatedAt = Date.now();
      saveChatHistory();

      textarea.value = "";
      autoGrow(textarea);
      status.textContent = "";

      const useContext = form.elements.useContext && form.elements.useContext.checked;
      const projectContext = useContext ? allContextText() : "";

      try {
        const response = await fetch(options.endpoint, {
          method: "POST",
          headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
          credentials: "include",
          body: JSON.stringify({
            agentType: options.agentType || null,
            mode: options.mode || null,
            message,
            projectContext,
            files: extractedFile ? [{ name: extractedFile.filename, characters: extractedFile.characters }] : [],
          }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || data.error || "AI request failed");

        const assistantEntry = { role: "assistant", content: data.answer || "", at: Date.now() };
        chatHistory[key].messages.push(assistantEntry);
        chatHistory[key].updatedAt = Date.now();
        saveChatHistory();

        const placeholder = document.getElementById(placeholderId);
        if (placeholder) placeholder.outerHTML = renderMessage("Sopal", data.answer || "", "assistant", true);
        status.textContent = "";
        scrollMessagesToBottom(messageArea);
      } catch (error) {
        const placeholder = document.getElementById(placeholderId);
        const errMsg = error.message || "AI request failed";
        if (placeholder) placeholder.outerHTML = `<div class="message assistant"><div class="avatar">S</div><div class="message-body"><div class="error-banner">${escapeHtml(errMsg)}</div></div></div>`;
        scrollMessagesToBottom(messageArea);
      }
    });
  }

  function autoGrow(textarea) {
    textarea.style.height = "auto";
    const max = 280;
    textarea.style.height = `${Math.min(max, textarea.scrollHeight)}px`;
  }

  function scrollMessagesToBottom(area) {
    if (!area) return;
    requestAnimationFrame(() => { area.scrollTop = area.scrollHeight; });
  }

  function renderMessage(name, content, role, withActions) {
    const body = role === "assistant" ? renderMarkdown(content) : `<p>${escapeHtml(content).replace(/\n/g, "<br>")}</p>`;
    return `<div class="message ${role}">
      <div class="avatar">${role === "assistant" ? "S" : "You"}</div>
      <div class="message-body">
        <div class="message-content" aria-label="${escapeHtml(name)} message">${body}</div>
        ${withActions ? `<div class="message-actions">
          <button class="ghost-button compact" data-copy-text="${escapeHtml(content)}" type="button">Copy</button>
        </div>` : ""}
      </div>
    </div>`;
  }

  /* ---------- Sign in modal ---------- */

  function openSignInModal() {
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
            <div class="modal-header">
              <h2 id="modal-title">Sign in to Sopal</h2>
              <button class="ghost-button compact" type="button" data-modal-close aria-label="Close">×</button>
            </div>
            <form class="modal-body" data-sign-in-form>
              <label>Email<input class="text-input" type="email" name="username" autocomplete="email" required autofocus></label>
              <label>Password<input class="text-input" type="password" name="password" autocomplete="current-password" required></label>
              <div class="modal-error" data-modal-error hidden></div>
              <div class="modal-actions">
                <a class="link-button" href="https://sopal.com.au/login" target="_blank" rel="noopener">Create account</a>
                <button class="dark-button" type="submit">Sign in</button>
              </div>
            </form>
          </div>
        </div>`,
      bind: (rootEl) => {
        const closeFn = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) closeFn(); });
        rootEl.querySelector("[data-modal-close]")?.addEventListener("click", closeFn);
        rootEl.querySelector("[data-sign-in-form]")?.addEventListener("submit", async (event) => {
          event.preventDefault();
          const form = event.currentTarget;
          const error = form.querySelector("[data-modal-error]");
          error.hidden = true;
          const fd = new FormData(form);
          const body = new URLSearchParams();
          body.set("username", fd.get("username") || "");
          body.set("password", fd.get("password") || "");
          form.querySelector("button[type=submit]").disabled = true;
          try {
            const r = await fetch("/purchase-login", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body, credentials: "include" });
            const data = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(data.detail || "Sign in failed");
            if (!data.access_token) throw new Error("No access token returned");
            localStorage.setItem(TOKEN_KEY, data.access_token);
            await refreshAuth();
            modal = null;
            render();
          } catch (e) {
            error.textContent = e.message || "Sign in failed";
            error.hidden = false;
            form.querySelector("button[type=submit]").disabled = false;
          }
        });
        document.addEventListener("keydown", function escListener(ev) {
          if (ev.key === "Escape") { document.removeEventListener("keydown", escListener); closeFn(); }
        });
      },
    };
    render();
  }

  /* ---------- Shell + render ---------- */

  function pageForRoute(route) {
    if (route === "home") return HomePage();
    const parts = route.split("/");
    if (parts[0] === "research") return ResearchPage(parts[1]);
    if (parts[0] === "tools") return ToolPage(parts[1]);
    if (parts[0] === "projects") return ProjectPage(parts[1]);
    if (parts[0] === "agents") return AgentPage(parts[1]);
    return HomePage();
  }

  function SopalV2Shell(route) {
    return `
      <div class="sopal-shell">
        ${Sidebar()}
        <main class="main">
          ${MainHeader(route)}
          <div class="content">${pageForRoute(route)}</div>
          <footer class="footer-disclaimer">Sopal assists with legal and contract analysis but does not replace professional legal advice.</footer>
        </main>
      </div>
      ${modal ? modal.render() : ""}
    `;
  }

  function bindShellEvents() {
    document.querySelectorAll("[data-nav]").forEach((link) => link.addEventListener("click", (event) => {
      const href = link.getAttribute("href");
      if (!href || !href.startsWith("/sopal-v2")) return;
      event.preventDefault();
      sidebarOpen = false;
      navigate(href);
    }));
    document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => navigate(b.getAttribute("data-go"))));
    document.querySelector("[data-toggle-sidebar]")?.addEventListener("click", () => { sidebarOpen = !sidebarOpen; render(); });
    document.querySelectorAll("[data-sign-in]").forEach((b) => b.addEventListener("click", openSignInModal));
    document.querySelectorAll("[data-sign-out]").forEach((b) => b.addEventListener("click", signOut));
    document.querySelectorAll("[data-starter]").forEach((b) => b.addEventListener("click", () => {
      const ta = document.querySelector(".chat-panel textarea");
      if (!ta) return;
      ta.value = b.dataset.starter;
      autoGrow(ta);
      ta.focus();
    }));
    document.querySelectorAll("[data-remove-context]").forEach((b) => b.addEventListener("click", () => {
      const [bucket, index] = b.dataset.removeContext.split(":");
      workspace[bucket].splice(Number(index), 1);
      saveWorkspace();
      render();
    }));
    document.querySelectorAll("[data-clear-context]").forEach((b) => b.addEventListener("click", () => {
      if (!confirm(`Clear all ${b.dataset.clearContext}?`)) return;
      workspace[b.dataset.clearContext] = [];
      saveWorkspace();
      render();
    }));
    document.querySelectorAll("[data-toggle-context-preview]").forEach((b) => b.addEventListener("click", () => {
      const target = document.querySelector("[data-context-preview]");
      if (!target) return;
      target.hidden = !target.hidden;
    }));
    if (modal) modal.bind(document);
  }

  function render() {
    root.innerHTML = SopalV2Shell(cleanPath());
    bindShellEvents();
  }

  function copyText(text) {
    if (navigator.clipboard) navigator.clipboard.writeText(text || "").catch(() => {});
  }

  /* ---------- Boot ---------- */

  window.addEventListener("popstate", render);
  document.addEventListener("click", (event) => {
    const copyButton = event.target.closest("[data-copy-text]");
    if (copyButton) {
      copyText(copyButton.dataset.copyText || "");
      const original = copyButton.textContent;
      copyButton.textContent = "Copied";
      setTimeout(() => { copyButton.textContent = original; }, 1200);
    }
    const clickable = event.target.closest("[data-decision-id].clickable");
    if (clickable && !event.target.closest("button,a,input")) {
      const id = clickable.dataset.decisionId;
      const title = clickable.dataset.title;
      if (id) loadDecisionDetail(id, title);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      const clickable = event.target.closest && event.target.closest("[data-decision-id].clickable");
      if (clickable && document.activeElement === clickable) {
        event.preventDefault();
        loadDecisionDetail(clickable.dataset.decisionId, clickable.dataset.title);
      }
    }
  });

  // Initial render, then refresh auth in the background and re-render header.
  render();
  refreshAuth().then(() => render());
})();
