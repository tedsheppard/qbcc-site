(function () {
  "use strict";

  const root = document.getElementById("sopal-v2-root");
  const STORAGE_KEY = "sopal-v2-local-workspace-v1";

  const sections = [
    { title: "Research", items: [["Adjudication Decisions", "/sopal-v2/research/adjudication-decisions"], ["Adjudicator Statistics", "/sopal-v2/research/adjudicator-statistics"], ["Caselaw", "/sopal-v2/research/caselaw"]] },
    { title: "Tools", items: [["Due Date Calculator", "/sopal-v2/tools/due-date-calculator"], ["Interest Calculator", "/sopal-v2/tools/interest-calculator"]] },
    { title: "Projects", items: [["Contracts", "/sopal-v2/projects/contracts"], ["Project Library", "/sopal-v2/projects/library"], ["Assistant", "/sopal-v2/projects/assistant"]] },
    { title: "Agents", items: [["Payment Claims", "/sopal-v2/agents/payment-claims"], ["Payment Schedules", "/sopal-v2/agents/payment-schedules"], ["EOTs", "/sopal-v2/agents/eots"], ["Variations", "/sopal-v2/agents/variations"], ["Delay Costs", "/sopal-v2/agents/delay-costs"], ["Adjudication Application", "/sopal-v2/agents/adjudication-application"], ["Adjudication Response", "/sopal-v2/agents/adjudication-response"]] },
  ];

  const agentLabels = Object.fromEntries(sections[3].items.map(([label, href]) => [href.split("/").pop(), label]));
  const agentDescriptions = {
    "payment-claims": "Review or draft payment claim material with SOPA compliance, amount, work identification, dates, service, and evidence focus.",
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

  const promptChips = {
    review: ["Identify fatal issues", "List missing evidence", "Check timing risks", "Suggest amendments"],
    draft: ["Draft formal wording", "Create evidence schedule", "Add placeholders", "Prepare cover email"],
  };

  const headers = {
    home: ["Sopal v2", "Research decisions, calculate SOPA dates and interest, and run document workflows."],
    "research/adjudication-decisions": ["Adjudication Decisions", "Search existing adjudication decision records using Sopal's real search API."],
    "research/adjudicator-statistics": ["Adjudicator Statistics", "Explore real adjudicator statistics from Sopal's decision database."],
    "research/caselaw": ["Caselaw", "Caselaw is not yet connected to a real search source in this repo."],
    "tools/due-date-calculator": ["Due Date Calculator", "Native BIF Act due date workflow using the existing Sopal calculation logic."],
    "tools/interest-calculator": ["Interest Calculator", "Native interest workflow using the existing Sopal RBA-rate endpoint."],
    "projects/contracts": ["Contracts", "Build local contract context from extracted files and pasted clauses."],
    "projects/library": ["Project Library", "Build local project context from correspondence, notices, claims, schedules, and other material."],
    "projects/assistant": ["Assistant", "Ask real AI questions against typed instructions and local project context."],
  };

  const holidays = {
    qld: [
      ["2025-01-01", "New Year's Day"], ["2025-01-27", "Australia Day"], ["2025-04-18", "Good Friday"], ["2025-04-19", "Day after Good Friday"], ["2025-04-21", "Easter Monday"], ["2025-04-25", "Anzac Day"], ["2025-05-05", "Labour Day"], ["2025-10-06", "King's Birthday"], ["2025-12-25", "Christmas Day"], ["2025-12-26", "Boxing Day"],
      ["2026-01-01", "New Year's Day"], ["2026-01-26", "Australia Day"], ["2026-04-03", "Good Friday"], ["2026-04-04", "Day after Good Friday"], ["2026-04-06", "Easter Monday"], ["2026-04-25", "Anzac Day"], ["2026-05-04", "Labour Day"], ["2026-10-05", "King's Birthday"], ["2026-12-25", "Christmas Day"], ["2026-12-28", "Boxing Day Holiday"],
      ["2027-01-01", "New Year's Day"], ["2027-01-26", "Australia Day"], ["2027-03-26", "Good Friday"], ["2027-03-27", "Day after Good Friday"], ["2027-03-29", "Easter Monday"], ["2027-04-26", "Anzac Day Holiday"], ["2027-05-03", "Labour Day"], ["2027-10-04", "King's Birthday"], ["2027-12-27", "Christmas Day Holiday"], ["2027-12-28", "Boxing Day Holiday"],
      ["2028-01-03", "New Year's Day Holiday"], ["2028-01-26", "Australia Day"], ["2028-04-14", "Good Friday"], ["2028-04-15", "Day after Good Friday"], ["2028-04-17", "Easter Monday"], ["2028-04-25", "Anzac Day"], ["2028-05-01", "Labour Day"], ["2028-10-02", "King's Birthday"], ["2028-12-25", "Christmas Day"], ["2028-12-26", "Boxing Day"],
    ].map(([date, name]) => ({ date, name })),
    bne: [["2025-08-13", "Brisbane EKKA Show Holiday"], ["2026-08-12", "Brisbane EKKA Show Holiday"], ["2027-08-11", "Brisbane EKKA Show Holiday"], ["2028-08-16", "Brisbane EKKA Show Holiday"]].map(([date, name]) => ({ date, name })),
    gld: [["2025-08-29", "Gold Coast Show Holiday"]].map(([date, name]) => ({ date, name })),
    cns: [["2025-07-18", "Cairns Show Holiday"]].map(([date, name]) => ({ date, name })),
    tsw: [["2025-07-07", "Townsville Show Holiday"]].map(([date, name]) => ({ date, name })),
    ipswich: [["2025-05-16", "Ipswich Show Holiday"]].map(([date, name]) => ({ date, name })),
    toowoomba: [["2025-03-28", "Toowoomba Show Holiday"]].map(([date, name]) => ({ date, name })),
    sunshine_coast: [["2025-05-30", "Maleny Show Holiday"], ["2025-06-13", "Sunshine Coast Show Holiday"]].map(([date, name]) => ({ date, name })),
    rockhampton: [["2025-06-12", "Rockhampton Show Holiday"]].map(([date, name]) => ({ date, name })),
    mackay: [["2025-06-19", "Mackay Show Holiday"]].map(([date, name]) => ({ date, name })),
  };

  let sidebarOpen = false;
  let workspace = loadWorkspace();

  function loadWorkspace() {
    try {
      return Object.assign({ contracts: [], library: [] }, JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"));
    } catch {
      return { contracts: [], library: [] };
    }
  }

  function saveWorkspace() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(workspace));
  }

  function allContextText() {
    const contractText = workspace.contracts.map((d) => `Contract: ${d.name}\n${d.text}`).join("\n\n---\n\n");
    const libraryText = workspace.library.map((d) => `Project document: ${d.name}\n${d.text}`).join("\n\n---\n\n");
    return [contractText, libraryText].filter(Boolean).join("\n\n===\n\n").slice(0, 40000);
  }

  function escapeHtml(value) {
    return String(value || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }

  function cleanPath() {
    const path = window.location.pathname.replace(/\/+$/, "");
    return path.replace(/^\/sopal-v2\/?/, "") || "home";
  }

  function navigate(href) {
    window.history.pushState({}, "", href);
    render();
  }

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

  function authHeaders() {
    const token = localStorage.getItem("purchase_token") || localStorage.getItem("token") || "";
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  function Sidebar() {
    return `
      <aside class="sopal-sidebar ${sidebarOpen ? "open" : ""}">
        <div class="sidebar-top">
          <div class="wordmark"><a href="/sopal-v2" data-nav>Sopal</a><span class="prototype-pill">v2 workspace</span></div>
          <button class="new-project" type="button" data-go="/sopal-v2/projects/contracts">New project context</button>
        </div>
        <div class="nav-scroll">
          ${sections.map((section) => `
            <div class="nav-group">
              <div class="nav-group-title">${escapeHtml(section.title)}</div>
              ${section.items.map(([label, href]) => `<a class="nav-item ${isActive(href) ? "active" : ""}" href="${href}" data-nav>${escapeHtml(label)}</a>`).join("")}
            </div>
          `).join("")}
        </div>
        <div class="sidebar-bottom">${workspace.contracts.length + workspace.library.length} local context item${workspace.contracts.length + workspace.library.length === 1 ? "" : "s"}</div>
      </aside>
    `;
  }

  function MainHeader(route) {
    const [title, description] = routeTitle(route);
    return `
      <header class="main-header">
        <div class="main-title"><h1>${escapeHtml(title)}</h1><p>${escapeHtml(description)}</p></div>
        <div class="header-actions">
          <button class="ghost-button mobile-toggle" type="button" data-toggle-sidebar>Menu</button>
          <a class="link-button" href="/ai">SopalAI</a>
          <a class="link-button" href="/">Current Sopal</a>
        </div>
      </header>
    `;
  }

  function EmptyState(title, body, actionHtml) {
    return `<div class="empty-state"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(body)}</p>${actionHtml || ""}</div>`;
  }

  function HomePage() {
    const contextCount = workspace.contracts.length + workspace.library.length;
    return `
      <div class="home-grid">
        <section class="home-intro">
          <h2>What are you working on?</h2>
          <p>Search adjudication decisions, inspect adjudicator outcomes, calculate dates and interest, or run SOPA document workflows using your local project context.</p>
          <div class="quick-actions">
            ${[
              ["Search decisions", "/sopal-v2/research/adjudication-decisions"],
              ["Review payment claim", "/sopal-v2/agents/payment-claims?mode=review"],
              ["Draft EOT claim", "/sopal-v2/agents/eots?mode=draft"],
              ["Ask assistant", "/sopal-v2/projects/assistant"],
            ].map(([label, href]) => `<a class="dark-button" href="${href}" data-nav>${escapeHtml(label)}</a>`).join("")}
          </div>
        </section>
        <section class="panel">
          <div class="panel-header"><div><h2>Workspace context</h2><p>Stored locally in this browser. Nothing is persisted to a Sopal database yet.</p></div></div>
          <div class="panel-body metric-grid">
            <div class="metric"><span>${workspace.contracts.length}</span><label>contract context items</label></div>
            <div class="metric"><span>${workspace.library.length}</span><label>project library items</label></div>
            <div class="metric"><span>${contextCount ? "Ready" : "Empty"}</span><label>agent context</label></div>
          </div>
        </section>
      </div>
    `;
  }

  function ResearchPage(kind) {
    if (kind === "adjudication-decisions") return AdjudicationDecisionsPage();
    if (kind === "adjudicator-statistics") return AdjudicatorStatisticsPage();
    return `<div class="section-stack"><div class="panel"><div class="panel-header"><div><h2>Caselaw</h2><p>No real caselaw database/API was found in this repo.</p></div></div><div class="panel-body">${EmptyState("Caselaw search is not configured yet.", "Adjudication decisions and adjudicator statistics are wired to real Sopal data. Caselaw needs a real data source before results can be shown.")}</div></div></div>`;
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
      if (q || params.toString()) fetchDecisionResults(params);
    }, 0);

    return `
      <div class="research-layout">
        <section class="panel">
          <div class="panel-header"><div><h2>Search decisions</h2><p>Uses Sopal's real <code>/search_fast</code> endpoint with native v2 results.</p></div></div>
          <form class="panel-body search-form" data-decision-search>
            <input class="text-input span-2" name="q" type="search" value="${escapeHtml(q)}" placeholder="Search terms, party names, adjudicator, section references">
            <select class="select-input" name="sort">
              ${["relevance", "newest", "oldest", "claim_high", "claim_low", "adj_high", "adj_low"].map((s) => `<option value="${s}" ${sort === s ? "selected" : ""}>${labelSort(s)}</option>`).join("")}
            </select>
            <input class="text-input" name="startDate" type="date" value="${escapeHtml(params.get("startDate") || "")}">
            <input class="text-input" name="endDate" type="date" value="${escapeHtml(params.get("endDate") || "")}">
            <input class="text-input" name="minClaim" type="number" step="1000" placeholder="Min claimed" value="${escapeHtml(params.get("minClaim") || "")}">
            <input class="text-input" name="maxClaim" type="number" step="1000" placeholder="Max claimed" value="${escapeHtml(params.get("maxClaim") || "")}">
            <button class="dark-button" type="submit">Search</button>
          </form>
        </section>
        <section id="decision-results">${EmptyState("Run a search.", "Results from the real adjudication decision database will appear here.")}</section>
        <aside id="decision-detail" class="panel detail-panel">${EmptyState("Select a decision.", "Open a result to inspect metadata and load full decision text.")}</aside>
      </div>
    `;
  }

  function labelSort(value) {
    return ({ relevance: "Relevance", newest: "Newest", oldest: "Oldest", claim_high: "Claimed high-low", claim_low: "Claimed low-high", adj_high: "Awarded high-low", adj_low: "Awarded low-high" })[value] || value;
  }

  async function fetchDecisionResults(params) {
    const mount = document.getElementById("decision-results");
    if (!mount) return;
    mount.innerHTML = EmptyState("Searching decisions.", "Querying Sopal's real decision database.");
    const qs = new URLSearchParams(params);
    qs.set("limit", "20");
    try {
      const response = await fetch(`/search_fast?${qs.toString()}`, { headers: authHeaders() });
      const data = await response.json();
      if (!response.ok) throw new Error(data.message || data.detail || data.error || "Search failed");
      const items = Array.isArray(data.items) ? data.items : [];
      if (!items.length) {
        mount.innerHTML = EmptyState("No decisions returned.", "The real search endpoint did not return records for this query.");
        return;
      }
      mount.innerHTML = `
        <div class="panel"><div class="panel-header"><div><h2>${escapeHtml(data.total || items.length)} result${Number(data.total || items.length) === 1 ? "" : "s"}</h2><p>Showing real Sopal decision records.</p></div></div>
        <div class="panel-body results-list">${items.map(renderDecisionItem).join("")}</div></div>
      `;
      mount.querySelectorAll("[data-decision-id]").forEach((button) => button.addEventListener("click", () => loadDecisionDetail(button.dataset.decisionId, button.dataset.title)));
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Search failed")}</div>`;
    }
  }

  function renderDecisionItem(item) {
    const claimant = item.claimant_name || item.claimant || "";
    const respondent = item.respondent_name || item.respondent || "";
    const title = [claimant, respondent].filter(Boolean).join(" v ") || item.reference || item.ejs_id || "Decision";
    const id = item.ejs_id || item.id || "";
    return `
      <article class="result-item">
        <div class="result-title-row"><h3>${escapeHtml(title)}</h3><button class="ghost-button compact" data-decision-id="${escapeHtml(id)}" data-title="${escapeHtml(title)}" type="button">Inspect</button></div>
        <div class="result-meta">${[item.decision_date || item.decision_date_norm, item.adjudicator_name || item.adjudicator, item.act_category || item.act, money(item.claimed_amount), money(item.adjudicated_amount)].filter(Boolean).map((m) => `<span>${escapeHtml(m)}</span>`).join("")}</div>
        <p>${formatSnippet(item.snippet)}</p>
      </article>`;
  }

  async function loadDecisionDetail(id, title) {
    const mount = document.getElementById("decision-detail");
    if (!mount || !id) return;
    mount.innerHTML = `<div class="panel-header"><div><h2>${escapeHtml(title || "Decision")}</h2><p>Loading real decision text.</p></div></div><div class="panel-body">${EmptyState("Loading decision text.", "Fetching from /api/decision-text.")}</div>`;
    try {
      const response = await fetch(`/api/decision-text/${encodeURIComponent(id)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Decision text failed");
      const text = (data.fullText || "").trim();
      mount.innerHTML = `<div class="panel-header"><div><h2>${escapeHtml(title || id)}</h2><p>${escapeHtml(id)}</p></div><a class="link-button" href="/open?id=${encodeURIComponent(id)}">Open original</a></div><div class="panel-body"><div class="decision-text">${escapeHtml(text.slice(0, 9000))}${text.length > 9000 ? "\n\n[Text truncated in preview. Open original for full record.]" : ""}</div></div>`;
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load decision text")}</div>`;
    }
  }

  function AdjudicatorStatisticsPage() {
    setTimeout(fetchAdjudicators, 0);
    return `
      <div class="section-stack">
        <div class="panel">
          <div class="panel-header"><div><h2>Real adjudicator statistics</h2><p>Uses Sopal's existing <code>/api/adjudicators</code> and detail endpoints.</p></div></div>
          <div class="panel-body toolbar"><input class="text-input" data-adj-filter placeholder="Filter adjudicators"><select class="select-input" data-adj-sort><option value="decisions">Most decisions</option><option value="award">Highest award rate</option><option value="claimed">Total claimed</option><option value="awarded">Total awarded</option></select></div>
        </div>
        <div class="stats-grid-shell"><section id="adjudicator-results">${EmptyState("Loading real statistics.", "Querying /api/adjudicators.")}</section><aside id="adjudicator-detail" class="panel detail-panel">${EmptyState("Select an adjudicator.", "View real decision rows for one adjudicator.")}</aside></div>
      </div>`;
  }

  async function fetchAdjudicators() {
    const mount = document.getElementById("adjudicator-results");
    if (!mount) return;
    try {
      const response = await fetch("/api/adjudicators");
      const data = await response.json();
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
    const filter = (document.querySelector("[data-adj-filter]")?.value || "").toLowerCase();
    const sort = document.querySelector("[data-adj-sort]")?.value || "decisions";
    let items = (window.__sopalAdjudicators || []).filter((a) => a.name.toLowerCase().includes(filter));
    items = items.sort((a, b) => ({ decisions: b.totalDecisions - a.totalDecisions, award: b.avgAwardRate - a.avgAwardRate, claimed: b.totalClaimAmount - a.totalClaimAmount, awarded: b.totalAwardedAmount - a.totalAwardedAmount }[sort]));
    mount.innerHTML = `<div class="adjudicator-grid">${items.slice(0, 60).map((item) => `
      <button class="stat-card-btn" type="button" data-adjudicator="${escapeHtml(item.name)}">
        <strong>${escapeHtml(item.name)}</strong>
        <span>${item.totalDecisions} decisions</span>
        <span>${formatCurrency(item.totalClaimAmount)} claimed</span>
        <span>${formatPercent(item.avgAwardRate)} avg award rate</span>
      </button>`).join("")}</div>`;
    mount.querySelectorAll("[data-adjudicator]").forEach((button) => button.addEventListener("click", () => loadAdjudicatorDetail(button.dataset.adjudicator)));
  }

  async function loadAdjudicatorDetail(name) {
    const mount = document.getElementById("adjudicator-detail");
    mount.innerHTML = `<div class="panel-header"><div><h2>${escapeHtml(name)}</h2><p>Loading real decision history.</p></div></div>`;
    try {
      const response = await fetch(`/api/adjudicator/${encodeURIComponent(name)}`, { headers: authHeaders() });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Adjudicator detail failed");
      const decisions = Array.isArray(data) ? data : [];
      mount.innerHTML = `<div class="panel-header"><div><h2>${escapeHtml(name)}</h2><p>${decisions.length} real decision${decisions.length === 1 ? "" : "s"}</p></div><a class="link-button" href="/adjudicators">Full page</a></div><div class="panel-body mini-list">${decisions.slice(0, 20).map((d) => `<article><strong>${escapeHtml(d.title)}</strong><span>${escapeHtml(d.date || "")} · claimed ${formatCurrency(d.claimAmount)} · awarded ${formatCurrency(d.awardedAmount)}</span></article>`).join("")}</div>`;
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load adjudicator")}</div>`;
    }
  }

  function ToolPage(kind) {
    return kind === "interest-calculator" ? InterestCalculator() : DueDateCalculator();
  }

  function DueDateCalculator() {
    setTimeout(bindDueDateCalculator, 0);
    return `
      <div class="tool-grid">
        <section class="panel"><div class="panel-header"><div><h2>BIF Act date calculator</h2><p>Native v2 UI using the same business-day logic as the existing Sopal calculator.</p></div></div>
          <form class="panel-body calc-form" data-due-form>
            <label>Scenario<select class="select-input" name="scenario"><option value="paymentSchedule">Payment schedule due date</option><option value="adjudicationAppLess">Adjudication application: scheduled amount less than claimed</option><option value="adjudicationAppNoPayAmount">Adjudication application: scheduled amount unpaid</option><option value="adjudicationAppNoSchedule">Adjudication application: no schedule and no payment</option><option value="adjudicationResponseStandard">Adjudication response: standard claim</option><option value="adjudicationResponseComplex">Adjudication response: complex claim</option><option value="adjudicatorDecisionStandard">Adjudicator decision: standard claim</option><option value="adjudicatorDecisionComplex">Adjudicator decision: complex claim</option></select></label>
            <label>Location<select class="select-input" name="location"><option value="qld">Queensland statewide</option><option value="bne">Brisbane</option><option value="gld">Gold Coast</option><option value="cns">Cairns</option><option value="tsw">Townsville</option><option value="ipswich">Ipswich</option><option value="toowoomba">Toowoomba</option><option value="sunshine_coast">Sunshine Coast</option><option value="rockhampton">Rockhampton</option><option value="mackay">Mackay</option></select></label>
            <label>Primary start date<input class="text-input" name="startDate" type="date"></label>
            <label>Second date, where required<input class="text-input" name="secondDate" type="date"></label>
            <label>Extension days, if applicable<input class="text-input" name="eotDays" type="number" min="0" max="30" value="0"></label>
            <button class="dark-button" type="submit">Calculate</button>
          </form>
        </section>
        <section class="panel"><div class="panel-header"><div><h2>Result</h2><p>Business days exclude weekends, listed Queensland/local public holidays, and the Christmas shutdown in the existing calculator.</p></div></div><div class="panel-body" id="due-result">${EmptyState("No calculation yet.", "Choose a scenario and enter the relevant dates.")}</div></section>
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
      if (result.error) {
        mount.innerHTML = `<div class="error-banner">${escapeHtml(result.error)}</div>`;
      } else {
        mount.innerHTML = renderDateResult(result);
      }
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
      if (!second) return { error: "Second date is required: application documents received date and adjudicator acceptance received date are both needed." };
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
    if (scenario === "adjudicatorDecisionStandard") { days = 10; eot = eotDays; basis = "10 business days after adjudication response is given, plus any agreed extension (s 85/s 86)."; }
    if (scenario === "adjudicatorDecisionComplex") { days = 15; eot = eotDays; basis = "15 business days after adjudication response is given, plus any agreed extension (s 85/s 86)."; }
    const base = addBusinessDays(startDate, days, location);
    const final = eot ? addBusinessDays(base.finalDate, eot, location) : base;
    return { title: labelScenario(scenario), startDate, days, eot, finalDate: final.finalDate, skipped: base.skipped.concat(eot ? final.skipped : []), basis };
  }

  function labelScenario(value) {
    return ({ paymentSchedule: "Payment schedule due date", adjudicationAppLess: "Adjudication application due date", adjudicationAppNoPayAmount: "Adjudication application due date", adjudicationAppNoSchedule: "Adjudication application due date", adjudicationResponseStandard: "Adjudication response due date", adjudicationResponseComplex: "Adjudication response due date", adjudicatorDecisionStandard: "Adjudicator decision due date", adjudicatorDecisionComplex: "Adjudicator decision due date" })[value] || "Due date";
  }

  function isBusinessDay(date, location) {
    const day = date.getDay();
    if (day === 0 || day === 6) return { isBiz: false, reason: "Weekend" };
    const month = date.getMonth(), dayOfMonth = date.getDate();
    if ((month === 11 && dayOfMonth >= 22 && dayOfMonth <= 24) || (month === 11 && dayOfMonth >= 27 && dayOfMonth <= 31) || (month === 0 && dayOfMonth >= 2 && dayOfMonth <= 10)) return { isBiz: false, reason: "Christmas Shutdown" };
    const dateString = date.toISOString().slice(0, 10);
    const publicHoliday = (holidays.qld || []).concat(holidays[location] || []).find((h) => h.date === dateString);
    return publicHoliday ? { isBiz: false, reason: publicHoliday.name } : { isBiz: true, reason: "" };
  }

  function addBusinessDays(startDate, days, location) {
    const currentDate = new Date(startDate.getTime());
    let daysAdded = 0, skipped = [];
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
    return `<div class="calc-result"><span>${escapeHtml(result.title)}</span><strong>${formatDate(result.finalDate)}</strong><p>${escapeHtml(result.basis)}</p><dl><dt>Start date</dt><dd>${formatDate(result.startDate)}</dd><dt>Business-day period</dt><dd>${result.days}${result.eot ? ` + ${result.eot} extension days` : ""}</dd><dt>Non-business days skipped</dt><dd>${escapeHtml(skippedSummary)}</dd></dl><button class="ghost-button compact" data-copy-text="${escapeHtml(`${result.title}: ${formatDate(result.finalDate)}\\n${result.basis}`)}">Copy</button></div>`;
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
    return `<div class="tool-grid"><section class="panel"><div class="panel-header"><div><h2>Interest calculator</h2><p>QBCC mode uses Sopal's real RBA rate endpoint. Contractual mode uses the existing frontend formula.</p></div></div><form class="panel-body calc-form" data-interest-form><label>Rate type<select class="select-input" name="type"><option value="qbcc">QBCC Act s 67P</option><option value="contractual">Contractual rate</option></select></label><label>Principal amount<input class="text-input" name="principal" type="number" min="0" step="0.01" placeholder="Amount unpaid"></label><label>Due date<input class="text-input" name="startDate" type="date"></label><label>Calculation date<input class="text-input" name="endDate" type="date" value="${today}"></label><label>Contractual annual rate %<input class="text-input" name="annualRate" type="number" min="0" step="0.01" value="10"></label><button class="dark-button" type="submit">Calculate</button></form></section><section class="panel"><div class="panel-header"><div><h2>Result</h2><p>Interest is calculated daily and includes the start and end dates, matching the existing Sopal calculator.</p></div></div><div class="panel-body" id="interest-result">${EmptyState("No calculation yet.", "Enter the unpaid amount and dates.")}</div></section></div>`;
  }

  function bindInterestCalculator() {
    const form = document.querySelector("[data-interest-form]");
    if (!form) return;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const mount = document.getElementById("interest-result");
      mount.innerHTML = EmptyState("Calculating.", "Fetching rates if required.");
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
    const principal = Number(data.principal), startDate = parseDate(data.startDate), endDate = parseDate(data.endDate);
    if (!principal || !startDate || !endDate) throw new Error("Principal and valid dates are required.");
    if (endDate < startDate) throw new Error("Calculation date must be after the due date.");
    const days = Math.ceil((endDate.getTime() - startDate.getTime()) / 86400000) + 1;
    if (data.type === "contractual") {
      const annualRate = Number(data.annualRate);
      if (Number.isNaN(annualRate)) throw new Error("Contractual rate is required.");
      return { type: "Contractual", principal, days, interest: principal * ((annualRate / 365) / 100) * days, annualRate, startDate, endDate };
    }
    const response = await fetch(`/get_interest_rate?startDate=${startDate.toISOString().slice(0, 10)}&endDate=${endDate.toISOString().slice(0, 10)}`);
    const ratesData = await response.json();
    if (!response.ok) throw new Error(ratesData.detail || "Could not fetch RBA rates.");
    let interest = 0;
    ratesData.dailyRates.forEach((row) => { interest += (principal / 365) * ((10 + Number(row.rate)) / 100); });
    const rates = ratesData.dailyRates.map((row) => Number(row.rate));
    return { type: "QBCC Act s 67P", principal, days, interest, startDate, endDate, minRate: Math.min(...rates), maxRate: Math.max(...rates), dailyRates: ratesData.dailyRates };
  }

  function renderInterestResult(result) {
    const total = result.principal + result.interest;
    const rateLine = result.type === "QBCC Act s 67P" ? `10% + RBA rate (${result.minRate.toFixed(2)}% to ${result.maxRate.toFixed(2)}%)` : `${result.annualRate.toFixed(2)}% contractual`;
    return `<div class="calc-result"><span>${escapeHtml(result.type)}</span><strong>${formatCurrency(result.interest)}</strong><p>Interest on ${formatCurrency(result.principal)} over ${result.days} days. Total: ${formatCurrency(total)}.</p><dl><dt>Rate</dt><dd>${escapeHtml(rateLine)}</dd><dt>Due date</dt><dd>${formatDate(result.startDate)}</dd><dt>Calculation date</dt><dd>${formatDate(result.endDate)}</dd></dl><button class="ghost-button compact" data-copy-text="${escapeHtml(`Interest: ${formatCurrency(result.interest)}\\nTotal: ${formatCurrency(total)}\\nRate: ${rateLine}`)}">Copy</button>${result.dailyRates ? `<details class="breakdown"><summary>Daily breakdown (${result.dailyRates.length} rows)</summary><table><thead><tr><th>Date</th><th>RBA</th><th>Rate</th><th>Daily interest</th></tr></thead><tbody>${result.dailyRates.slice(0, 120).map((row) => `<tr><td>${escapeHtml(row.date)}</td><td>${Number(row.rate).toFixed(2)}%</td><td>${(10 + Number(row.rate)).toFixed(2)}%</td><td>${formatCurrency((result.principal / 365) * ((10 + Number(row.rate)) / 100))}</td></tr>`).join("")}</tbody></table></details>` : ""}</div>`;
  }

  function ProjectPage(kind) {
    if (kind === "assistant") return `<div class="section-stack"><ContextSummary />${ChatPanel({ endpoint: "/api/sopal-v2/chat", assistant: true, placeholder: "Ask about project facts, contract wording, notices, claims, schedules, or correspondence.", emptyTitle: "Project assistant", emptyBody: "Ask a question. Toggle project context on to include local extracted/pasted context." })}</div>`.replace("<ContextSummary />", ContextSummary());
    return ContextManager(kind === "contracts" ? "contracts" : "library");
  }

  function ContextSummary() {
    return `<section class="panel compact-panel"><div class="panel-header"><div><h2>Available local context</h2><p>${workspace.contracts.length} contract item${workspace.contracts.length === 1 ? "" : "s"} · ${workspace.library.length} project library item${workspace.library.length === 1 ? "" : "s"}</p></div><a class="link-button" href="/sopal-v2/projects/contracts" data-nav>Manage context</a></div></section>`;
  }

  function ContextManager(bucket) {
    const label = bucket === "contracts" ? "Contracts" : "Project Library";
    const helper = bucket === "contracts" ? "Paste contract clauses or extract text from PDF/DOCX/TXT. This creates browser-local context for the Assistant and agents." : "Paste RFIs, correspondence, notices, claims, schedules, programme notes, or extract text from PDF/DOCX/TXT.";
    setTimeout(() => bindContextManager(bucket), 0);
    return `<div class="context-layout"><section class="panel"><div class="panel-header"><div><h2>${label}</h2><p>${helper}</p></div></div><form class="panel-body context-form" data-context-form="${bucket}"><label>Label<input class="text-input" name="name" placeholder="Document/context name"></label><label>Paste text<textarea class="text-area" name="text" placeholder="Paste relevant clauses, correspondence, claim text, schedule text, or facts."></textarea></label><div class="file-dropzone"><label>Select PDF/DOCX/TXT to extract text<input type="file" data-context-file accept=".pdf,.docx,.txt"></label><div class="file-list" data-context-file-status>No file selected.</div></div><button class="dark-button" type="submit">Save to local context</button></form></section><section class="panel"><div class="panel-header"><div><h2>Saved local context</h2><p>Stored in browser localStorage. There is no server persistence yet.</p></div><button class="ghost-button compact" data-clear-context="${bucket}" type="button">Clear</button></div><div class="panel-body context-list">${renderContextList(bucket)}</div></section></div>`;
  }

  function renderContextList(bucket) {
    const items = workspace[bucket] || [];
    if (!items.length) return EmptyState(`No ${bucket === "contracts" ? "contracts" : "project documents"} saved.`, "Add pasted or extracted text to make the assistant and agents context-aware.");
    return items.map((item, index) => `<article class="context-item"><strong>${escapeHtml(item.name)}</strong><span>${item.text.length.toLocaleString()} characters · ${escapeHtml(item.source || "pasted")}</span><p>${escapeHtml(item.text.slice(0, 280))}${item.text.length > 280 ? "..." : ""}</p><button class="ghost-button compact" data-remove-context="${bucket}:${index}" type="button">Remove</button></article>`).join("");
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
      status.textContent = "Extracting text...";
      const fd = new FormData();
      fd.append("file", file);
      try {
        const response = await fetch("/api/sopal-v2/extract", { method: "POST", body: fd });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Extraction failed");
        extracted = data;
        form.elements.name.value = form.elements.name.value || data.filename;
        form.elements.text.value = [form.elements.text.value, data.text].filter(Boolean).join("\n\n");
        status.textContent = `${data.filename}: ${data.characters.toLocaleString()} characters extracted${data.truncated ? " (truncated)" : ""}.`;
      } catch (error) {
        status.textContent = error.message || "Extraction failed";
      }
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const data = Object.fromEntries(new FormData(form).entries());
      if (!data.text || !String(data.text).trim()) return;
      workspace[bucket].push({ name: String(data.name || extracted?.filename || "Untitled context"), text: String(data.text).trim(), source: extracted ? "extracted file + paste" : "pasted", createdAt: new Date().toISOString() });
      saveWorkspace();
      render();
    });
  }

  function AgentPage(agentKey) {
    const params = new URLSearchParams(window.location.search);
    const mode = params.get("mode") === "draft" ? "draft" : "review";
    const label = agentLabels[agentKey] || "Agent";
    return `<div class="agent-layout"><section class="panel agent-brief"><div class="panel-header"><div><h2>${escapeHtml(label)}</h2><p>${escapeHtml(agentDescriptions[agentKey] || "")}</p></div><div class="mode-tabs"><button class="mode-tab ${mode === "review" ? "active" : ""}" data-go="/sopal-v2/agents/${agentKey}?mode=review" type="button">Review</button><button class="mode-tab ${mode === "draft" ? "active" : ""}" data-go="/sopal-v2/agents/${agentKey}?mode=draft" type="button">Draft</button></div></div><div class="panel-body helper-grid"><div><strong>What to include</strong><ul>${(includeLists[agentKey] || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div><div><strong>Useful prompts</strong><div class="chip-row">${promptChips[mode].map((chip) => `<button class="chip" type="button" data-chip="${escapeHtml(chip)}">${escapeHtml(chip)}</button>`).join("")}</div></div></div></section>${ChatPanel({ endpoint: "/api/sopal-v2/agent", agentType: agentKey, mode, placeholder: mode === "review" ? "Paste the document, key dates, contract clauses, and facts to review." : "Describe what needs drafting and paste the relevant project/contract facts.", emptyTitle: `${label} ${mode === "review" ? "Review" : "Draft"}`, emptyBody: "Submit pasted/extracted material for a real AI response. Use project context if relevant." })}</div>`;
  }

  function ChatPanel(options) {
    const id = `chat-${Math.random().toString(36).slice(2)}`;
    setTimeout(() => bindChatPanel(id, options), 0);
    const contextAvailable = workspace.contracts.length + workspace.library.length > 0;
    return `<section class="chat-panel" id="${id}"><div class="message-area"><div class="message-stack" data-messages>${EmptyState(options.emptyTitle || "No conversation yet.", options.emptyBody || "Start with typed instructions or pasted text.")}</div></div><form class="composer" data-chat-form><div class="composer-inner"><div class="composer-options"><label><input type="checkbox" name="useContext" ${contextAvailable ? "checked" : ""}> Use local project context (${workspace.contracts.length + workspace.library.length})</label><label class="file-inline">Extract file<input type="file" data-chat-file accept=".pdf,.docx,.txt"></label><span data-chat-file-status></span></div><textarea class="text-area" name="message" placeholder="${escapeHtml(options.placeholder || "Type your message")}"></textarea><div class="composer-footer"><button class="ghost-button" type="button" data-clear-chat>Clear</button><button class="send-button" type="submit">Send</button></div><div class="status-line" data-status></div></div></form></section>`;
  }

  function bindChatPanel(id, options) {
    const panel = document.getElementById(id);
    if (!panel) return;
    const form = panel.querySelector("[data-chat-form]");
    const messages = panel.querySelector("[data-messages]");
    const status = panel.querySelector("[data-status]");
    const textarea = form.elements.message;
    let extractedFile = null;
    panel.querySelector("[data-chat-file]")?.addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      const fileStatus = panel.querySelector("[data-chat-file-status]");
      fileStatus.textContent = "Extracting...";
      const fd = new FormData();
      fd.append("file", file);
      try {
        const response = await fetch("/api/sopal-v2/extract", { method: "POST", body: fd });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Extraction failed");
        extractedFile = data;
        textarea.value = [textarea.value, `\n\nExtracted text from ${data.filename}:\n${data.text}`].filter(Boolean).join("\n");
        fileStatus.textContent = `${data.characters.toLocaleString()} chars extracted`;
      } catch (error) {
        fileStatus.textContent = error.message || "Extraction failed";
      }
    });
    panel.querySelector("[data-clear-chat]")?.addEventListener("click", () => {
      messages.innerHTML = EmptyState(options.emptyTitle || "No conversation yet.", options.emptyBody || "Start with typed instructions or pasted text.");
      status.textContent = "";
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = textarea.value.trim();
      if (!message) { status.textContent = "Enter text before sending."; return; }
      if (messages.querySelector(".empty-state")) messages.innerHTML = "";
      messages.insertAdjacentHTML("beforeend", renderMessage("You", message, "user"));
      textarea.value = "";
      status.textContent = "Requesting AI response...";
      const projectContext = form.elements.useContext.checked ? allContextText() : "";
      try {
        const response = await fetch(options.endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agentType: options.agentType || null, mode: options.mode || null, message, projectContext, files: extractedFile ? [{ name: extractedFile.filename, characters: extractedFile.characters }] : [] }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.error || "AI request failed");
        messages.insertAdjacentHTML("beforeend", renderMessage("Sopal", data.answer || "", "assistant", true));
        status.textContent = "Response generated.";
      } catch (error) {
        status.innerHTML = `<span class="error-banner">${escapeHtml(error.message || "AI request failed")}</span>`;
      }
      panel.querySelector(".message-area").scrollTop = panel.querySelector(".message-area").scrollHeight;
    });
  }

  function renderMessage(name, content, role, withActions) {
    return `<div class="message ${role}"><div class="avatar">${role === "assistant" ? "S" : "You"}</div><div class="message-body"><div class="message-content" aria-label="${escapeHtml(name)} message">${renderMarkdown(content)}</div>${withActions ? `<div class="message-actions"><button class="ghost-button compact" data-copy-text="${escapeHtml(content)}" type="button">Copy output</button></div>` : ""}</div></div>`;
  }

  function renderMarkdown(text) {
    let safe = escapeHtml(text || "");
    safe = safe.replace(/^### (.*)$/gm, "<h4>$1</h4>").replace(/^## (.*)$/gm, "<h3>$1</h3>").replace(/^# (.*)$/gm, "<h3>$1</h3>").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    safe = safe.replace(/^\s*[-*] (.*)$/gm, "<li>$1</li>").replace(/(<li>.*<\/li>)/gs, (m) => `<ul>${m}</ul>`);
    return safe.replace(/\n{2,}/g, "<br><br>").replace(/\n/g, "<br>");
  }

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
    return `<div class="sopal-shell">${Sidebar()}<main class="main">${MainHeader(route)}<div class="content">${pageForRoute(route)}</div><footer class="footer-disclaimer">Sopal assists with legal and contract analysis but does not replace professional legal advice.</footer></main></div>`;
  }

  function bindShellEvents() {
    document.querySelectorAll("[data-nav]").forEach((link) => link.addEventListener("click", (event) => {
      const href = link.getAttribute("href");
      if (!href || !href.startsWith("/sopal-v2")) return;
      event.preventDefault();
      sidebarOpen = false;
      navigate(href);
    }));
    document.querySelectorAll("[data-go]").forEach((button) => button.addEventListener("click", () => navigate(button.getAttribute("data-go"))));
    document.querySelector("[data-toggle-sidebar]")?.addEventListener("click", () => { sidebarOpen = !sidebarOpen; render(); });
    document.querySelectorAll("[data-chip]").forEach((button) => button.addEventListener("click", () => {
      const textarea = document.querySelector(".chat-panel textarea");
      if (textarea) textarea.value = [textarea.value, button.dataset.chip].filter(Boolean).join(textarea.value ? "\n" : "");
    }));
    document.querySelectorAll("[data-remove-context]").forEach((button) => button.addEventListener("click", () => {
      const [bucket, index] = button.dataset.removeContext.split(":");
      workspace[bucket].splice(Number(index), 1);
      saveWorkspace();
      render();
    }));
    document.querySelectorAll("[data-clear-context]").forEach((button) => button.addEventListener("click", () => {
      workspace[button.dataset.clearContext] = [];
      saveWorkspace();
      render();
    }));
  }

  function render() {
    root.innerHTML = SopalV2Shell(cleanPath());
    bindShellEvents();
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

  function formatPercent(value) {
    return `${Number(value || 0).toFixed(1)}%`;
  }

  function money(value) {
    if (value === null || value === undefined || value === "" || value === "N/A") return "";
    return formatCurrency(value);
  }

  function formatSnippet(text) {
    return escapeHtml(text || "").replace(/&lt;mark&gt;/g, "<mark>").replace(/&lt;\/mark&gt;/g, "</mark>");
  }

  function copyText(text) {
    if (navigator.clipboard) navigator.clipboard.writeText(text);
  }

  window.addEventListener("popstate", render);
  document.addEventListener("click", (event) => {
    const copyButton = event.target.closest("[data-copy-text]");
    if (copyButton) copyText(copyButton.dataset.copyText || "");
  });
  render();
})();
