(function () {
  "use strict";

  const root = document.getElementById("sopal-v2-root");

  const sections = [
    {
      title: "Research",
      items: [
        ["Adjudication Decisions", "/sopal-v2/research/adjudication-decisions"],
        ["Adjudicator Statistics", "/sopal-v2/research/adjudicator-statistics"],
        ["Caselaw", "/sopal-v2/research/caselaw"],
      ],
    },
    {
      title: "Tools",
      items: [
        ["Due Date Calculator", "/sopal-v2/tools/due-date-calculator"],
        ["Interest Calculator", "/sopal-v2/tools/interest-calculator"],
      ],
    },
    {
      title: "Projects",
      items: [
        ["Contracts", "/sopal-v2/projects/contracts"],
        ["Project Library", "/sopal-v2/projects/library"],
        ["Assistant", "/sopal-v2/projects/assistant"],
      ],
    },
    {
      title: "Agents",
      items: [
        ["Payment Claims", "/sopal-v2/agents/payment-claims"],
        ["Payment Schedules", "/sopal-v2/agents/payment-schedules"],
        ["EOTs", "/sopal-v2/agents/eots"],
        ["Variations", "/sopal-v2/agents/variations"],
        ["Delay Costs", "/sopal-v2/agents/delay-costs"],
        ["Adjudication Application", "/sopal-v2/agents/adjudication-application"],
        ["Adjudication Response", "/sopal-v2/agents/adjudication-response"],
      ],
    },
  ];

  const agentDescriptions = {
    "payment-claims": "Review or draft payment claim material with careful SOPA and contract analysis.",
    "payment-schedules": "Review or draft payment schedules, withholding reasons, and response structure.",
    eots: "Review or draft extension of time notices and claims against contract requirements.",
    variations: "Review or draft variation notices and claims with entitlement, valuation, and evidence focus.",
    "delay-costs": "Review or draft delay cost, prolongation, or disruption claim material.",
    "adjudication-application": "Review or draft adjudication application submissions and supporting structure.",
    "adjudication-response": "Review or draft adjudication response submissions and jurisdictional objections.",
  };

  const agentLabels = Object.fromEntries(sections[3].items.map(([label, href]) => [href.split("/").pop(), label]));

  const headers = {
    home: ["What are you working on?", "Search decisions, use real tools, or work with a SOPA agent."],
    "research/adjudication-decisions": ["Adjudication Decisions", "Search existing adjudication decision records."],
    "research/adjudicator-statistics": ["Adjudicator Statistics", "Explore existing adjudicator decision statistics."],
    "research/caselaw": ["Caselaw", "Caselaw search shell for a future real database connection."],
    "tools/due-date-calculator": ["Due Date Calculator", "Existing Sopal calculator embedded in the v2 workspace."],
    "tools/interest-calculator": ["Interest Calculator", "Existing Sopal calculator embedded in the v2 workspace."],
    "projects/contracts": ["Contracts", "Contract document workspace."],
    "projects/library": ["Project Library", "Project document workspace."],
    "projects/assistant": ["Assistant", "Project-specific chat using typed instructions and available context."],
  };

  let sidebarOpen = false;

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function cleanPath() {
    const path = window.location.pathname.replace(/\/+$/, "");
    const rest = path.replace(/^\/sopal-v2\/?/, "");
    return rest || "home";
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

  function Sidebar() {
    const nav = sections.map((section) => `
      <div class="nav-group">
        <div class="nav-group-title">${escapeHtml(section.title)}</div>
        ${section.items.map(([label, href]) => `
          <a class="nav-item ${isActive(href) ? "active" : ""}" href="${href}" data-nav>
            ${escapeHtml(label)}
          </a>
        `).join("")}
      </div>
    `).join("");

    return `
      <aside class="sopal-sidebar ${sidebarOpen ? "open" : ""}">
        <div class="sidebar-top">
          <div class="wordmark">
            <a href="/sopal-v2" data-nav>Sopal</a>
            <span class="prototype-pill">Local prototype</span>
          </div>
          <button class="new-project" type="button" data-go="/sopal-v2/projects/contracts">New project</button>
        </div>
        <div class="nav-scroll">${nav}</div>
        <div class="sidebar-bottom">Sopal assists with legal and contract analysis but does not replace professional legal advice.</div>
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
          <button class="ghost-button mobile-toggle" type="button" data-toggle-sidebar>Menu</button>
          <a class="link-button" href="/" target="_self">Current Sopal</a>
        </div>
      </header>
    `;
  }

  function EmptyState(title, body, actionHtml) {
    return `
      <div class="empty-state">
        <strong>${escapeHtml(title)}</strong>
        <p>${escapeHtml(body)}</p>
        ${actionHtml || ""}
      </div>
    `;
  }

  function HomePage() {
    const actions = [
      ["Search adjudication decisions", "/sopal-v2/research/adjudication-decisions", "Open the decision search workspace."],
      ["Review a payment claim", "/sopal-v2/agents/payment-claims?mode=review", "Check requirements, timing, and evidence gaps."],
      ["Draft an EOT claim", "/sopal-v2/agents/eots?mode=draft", "Prepare careful wording from project facts."],
      ["Open project assistant", "/sopal-v2/projects/assistant", "Work from typed instructions and pasted text."],
    ];
    return `
      <div class="home-center">
        <h2 class="home-title">What are you working on?</h2>
        <p class="home-subtitle">Search adjudication decisions, review SOPA documents, draft claims, or work inside a project.</p>
        <div class="action-grid">
          ${actions.map(([title, href, body]) => `
            <a class="action-card" href="${href}" data-nav>
              <strong>${escapeHtml(title)}</strong>
              <span>${escapeHtml(body)}</span>
            </a>
          `).join("")}
        </div>
      </div>
    `;
  }

  function ResearchPage(kind) {
    if (kind === "adjudication-decisions") return AdjudicationDecisionsPage();
    if (kind === "adjudicator-statistics") return AdjudicatorStatisticsPage();
    return `
      <div class="section-stack">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2>Caselaw</h2>
              <p>No real caselaw database endpoint was found for this prototype.</p>
            </div>
          </div>
          <div class="panel-body">
            ${EmptyState("Caselaw search is not configured yet.", "Connect this page to a real caselaw database or API before showing results.")}
          </div>
        </div>
      </div>
    `;
  }

  function AdjudicationDecisionsPage() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q") || "";
    setTimeout(() => {
      const form = document.querySelector("[data-decision-search]");
      if (!form) return;
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        const query = form.querySelector("input").value.trim();
        const href = query ? `/sopal-v2/research/adjudication-decisions?q=${encodeURIComponent(query)}` : "/sopal-v2/research/adjudication-decisions";
        navigate(href);
      });
      if (q) fetchDecisionResults(q);
    }, 0);

    return `
      <div class="section-stack">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2>Decision Search</h2>
              <p>Search existing adjudication decision records.</p>
            </div>
            <a class="link-button" href="/search">Open existing search</a>
          </div>
          <div class="panel-body">
            <form class="search-row" data-decision-search>
              <input class="text-input" type="search" value="${escapeHtml(q)}" placeholder="Search adjudication decisions">
              <button class="dark-button" type="submit">Search</button>
            </form>
          </div>
        </div>
        <div id="decision-results">
          ${q ? EmptyState("Searching real decisions.", "Results will appear here when the existing endpoint responds.") : EmptyState("No search entered.", "Enter a term to query the existing adjudication decision database.")}
        </div>
      </div>
    `;
  }

  async function fetchDecisionResults(query) {
    const mount = document.getElementById("decision-results");
    if (!mount) return;
    mount.innerHTML = EmptyState("Searching real decisions.", "Waiting for the existing adjudication decision endpoint.");
    try {
      const response = await fetch(`/search_fast?q=${encodeURIComponent(query)}&limit=12&sort=relevance`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.message || data.detail || data.error || "Search failed");
      const items = Array.isArray(data.items) ? data.items : [];
      if (!items.length) {
        mount.innerHTML = EmptyState("No decisions returned.", "The existing search endpoint did not return results for that query.");
        return;
      }
      mount.innerHTML = `
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2>${escapeHtml(String(data.total || items.length))} result${Number(data.total || items.length) === 1 ? "" : "s"}</h2>
              <p>Showing real records returned by the current Sopal search API.</p>
            </div>
          </div>
          <div class="panel-body results-list">
            ${items.map(renderDecisionItem).join("")}
          </div>
        </div>
      `;
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Search failed")}</div>`;
    }
  }

  function renderDecisionItem(item) {
    const claimant = item.claimant_name || item.claimant || "";
    const respondent = item.respondent_name || item.respondent || "";
    const title = [claimant, respondent].filter(Boolean).join(" v ") || item.reference || item.ejs_id || "Decision";
    const meta = [
      item.decision_date || item.decision_date_norm,
      item.adjudicator_name || item.adjudicator,
      item.act_category || item.act,
    ].filter(Boolean);
    const link = item.ejs_id ? `/open?id=${encodeURIComponent(item.ejs_id)}` : "/search";
    return `
      <article class="result-item">
        <h3>${escapeHtml(title)}</h3>
        <div class="result-meta">${meta.map((m) => `<span>${escapeHtml(m)}</span>`).join("")}</div>
        <p>${formatSnippet(item.snippet)}</p>
        <div class="result-meta" style="margin-top:10px"><a href="${link}">Open decision</a></div>
      </article>
    `;
  }

  function AdjudicatorStatisticsPage() {
    setTimeout(fetchAdjudicators, 0);
    return `
      <div class="section-stack">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2>Adjudicator Statistics</h2>
              <p>Review existing adjudicator decision statistics.</p>
            </div>
            <a class="link-button" href="/adjudicators">Open existing statistics</a>
          </div>
          <div class="panel-body" id="adjudicator-results">
            ${EmptyState("Loading real adjudicator statistics.", "The prototype is requesting the existing Sopal statistics endpoint.")}
          </div>
        </div>
      </div>
    `;
  }

  async function fetchAdjudicators() {
    const mount = document.getElementById("adjudicator-results");
    if (!mount) return;
    try {
      const response = await fetch("/api/adjudicators");
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Adjudicator endpoint failed");
      const items = Array.isArray(data) ? data : [];
      if (!items.length) {
        mount.innerHTML = EmptyState("No adjudicator statistics returned.", "The existing endpoint responded without records.");
        return;
      }
      mount.innerHTML = `
        <div class="results-list">
          ${items.slice(0, 20).map((item) => `
            <article class="result-item">
              <h3>${escapeHtml(item.name)}</h3>
              <div class="result-meta">
                <span>${escapeHtml(item.totalDecisions)} decisions</span>
                <span>${formatCurrency(item.totalClaimAmount)} claimed</span>
                <span>${formatCurrency(item.totalAwardedAmount)} awarded</span>
                <span>${formatPercent(item.avgAwardRate)} average award rate</span>
              </div>
            </article>
          `).join("")}
        </div>
      `;
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load adjudicators")}</div>`;
    }
  }

  function ToolPage(kind) {
    const src = kind === "interest-calculator" ? "/interest-calculator" : "/due-date-calculator";
    const title = kind === "interest-calculator" ? "Interest Calculator" : "Due Date Calculator";
    return `
      <div class="section-stack">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2>${escapeHtml(title)}</h2>
              <p>Use the existing Sopal calculator inside the v2 workspace.</p>
            </div>
            <a class="link-button" href="${src}">Open full page</a>
          </div>
          <div class="iframe-shell">
            <iframe title="${escapeHtml(title)}" src="${src}"></iframe>
          </div>
        </div>
      </div>
    `;
  }

  function ProjectPage(kind) {
    if (kind === "assistant") {
      return `
        <div class="section-stack">
          <div class="notice">No project document store is connected to this prototype. Answers are based only on typed instructions and pasted text.</div>
          ${ChatPanel({ endpoint: "/api/sopal-v2/chat", placeholder: "Ask about project facts, contract wording, notices, claims, or correspondence you paste here.", emptyTitle: "No project conversation yet.", emptyBody: "Start by typing instructions or pasting relevant project text." })}
        </div>
      `;
    }
    if (kind === "library") {
      return `
        <div class="section-stack">
          <div class="panel">
            <div class="panel-header">
              <div>
                <h2>Project Library</h2>
                <p>For RFIs, correspondence, notices, claims, schedules, photos, programmes, and general documents.</p>
              </div>
            </div>
            <div class="panel-body">
              ${FileDropzone("library-files")}
              ${EmptyState("No project documents uploaded yet.", "Document storage is not configured yet. Selected files are not uploaded or parsed by this prototype.")}
            </div>
          </div>
        </div>
      `;
    }
    return `
      <div class="section-stack">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2>Contracts</h2>
              <p>Contract documents can be selected locally, but no storage or parsing endpoint is configured here.</p>
            </div>
          </div>
          <div class="panel-body">
            ${FileDropzone("contract-files")}
            ${EmptyState("No contracts uploaded yet.", "Document storage is not configured yet. Selected files are not uploaded or parsed by this prototype.")}
          </div>
        </div>
      </div>
    `;
  }

  function AgentPage(agentKey) {
    const label = agentLabels[agentKey] || "Agent";
    const params = new URLSearchParams(window.location.search);
    const mode = params.get("mode") === "draft" ? "draft" : "review";
    const modeHref = (nextMode) => `/sopal-v2/agents/${agentKey}?mode=${nextMode}`;
    return `
      <div class="section-stack">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2>${escapeHtml(label)}</h2>
              <p>${escapeHtml(agentDescriptions[agentKey] || "Review or draft SOPA material.")}</p>
            </div>
            <div class="mode-tabs" role="tablist" aria-label="Agent mode">
              <button class="mode-tab ${mode === "review" ? "active" : ""}" data-go="${modeHref("review")}" type="button">Review</button>
              <button class="mode-tab ${mode === "draft" ? "active" : ""}" data-go="${modeHref("draft")}" type="button">Draft</button>
            </div>
          </div>
        </div>
        ${ChatPanel({
          endpoint: "/api/sopal-v2/agent",
          agentType: agentKey,
          mode,
          placeholder: mode === "review" ? "Paste the document text, dates, contract clauses, and facts to review." : "Describe what needs drafting and paste the relevant contract/project facts.",
          emptyTitle: `${label} ${mode === "review" ? "Review" : "Draft"}`,
          emptyBody: "No conversation yet. The first response will be generated by the real AI endpoint if configured.",
        })}
      </div>
    `;
  }

  function FileDropzone(id) {
    return `
      <div class="file-dropzone" data-file-zone>
        <label for="${escapeHtml(id)}">
          <span>Select files locally</span>
          <span>Not uploaded</span>
        </label>
        <input id="${escapeHtml(id)}" type="file" multiple data-file-input>
        <div class="file-list" data-file-list>No files selected.</div>
      </div>
    `;
  }

  function ChatPanel(options) {
    const panelId = `chat-${Math.random().toString(36).slice(2)}`;
    setTimeout(() => bindChatPanel(panelId, options), 0);
    return `
      <section class="chat-panel" id="${panelId}">
        <div class="message-area">
          <div class="message-stack" data-messages>
            ${EmptyState(options.emptyTitle || "No conversation yet.", options.emptyBody || "Start with typed instructions or pasted document text.")}
          </div>
        </div>
        <form class="composer" data-chat-form>
          <div class="composer-inner">
            ${FileDropzone(`${panelId}-files`)}
            <div class="composer-actions">
              <textarea class="text-area" name="message" placeholder="${escapeHtml(options.placeholder || "Type your message")}"></textarea>
              <button class="send-button" type="submit">Send</button>
            </div>
            <div class="status-line" data-status></div>
          </div>
        </form>
      </section>
    `;
  }

  function bindChatPanel(panelId, options) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    bindFileZones(panel);
    const form = panel.querySelector("[data-chat-form]");
    const messages = panel.querySelector("[data-messages]");
    const status = panel.querySelector("[data-status]");
    const textarea = form.querySelector("textarea");
    const selectedFiles = [];

    panel.querySelectorAll("[data-file-input]").forEach((input) => {
      input.addEventListener("change", () => {
        selectedFiles.splice(0, selectedFiles.length, ...Array.from(input.files || []).map((file) => ({ name: file.name, size: file.size, type: file.type })));
      });
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = textarea.value.trim();
      if (!message) {
        status.textContent = "Enter text before sending.";
        return;
      }

      if (messages.querySelector(".empty-state")) messages.innerHTML = "";
      messages.insertAdjacentHTML("beforeend", renderMessage("You", message, "user"));
      textarea.value = "";
      status.textContent = "Requesting AI response...";

      try {
        const response = await fetch(options.endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            agentType: options.agentType || null,
            mode: options.mode || null,
            message,
            files: selectedFiles,
          }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.error || "AI request failed");
        messages.insertAdjacentHTML("beforeend", renderMessage("Sopal", data.answer || "", "assistant"));
        status.textContent = "Response generated.";
      } catch (error) {
        status.innerHTML = `<span class="error-banner">${escapeHtml(error.message || "AI request failed")}</span>`;
      }
      panel.querySelector(".message-area").scrollTop = panel.querySelector(".message-area").scrollHeight;
    });
  }

  function bindFileZones(scope) {
    scope.querySelectorAll("[data-file-zone]").forEach((zone) => {
      const input = zone.querySelector("[data-file-input]");
      const list = zone.querySelector("[data-file-list]");
      if (!input || !list || input.dataset.bound) return;
      input.dataset.bound = "1";
      input.addEventListener("change", () => {
        const files = Array.from(input.files || []);
        list.textContent = files.length ? files.map((file) => file.name).join(", ") : "No files selected.";
      });
    });
  }

  function renderMessage(name, content, role) {
    return `
      <div class="message ${role}">
        <div class="avatar">${role === "assistant" ? "S" : "You"}</div>
        <div class="message-content" aria-label="${escapeHtml(name)} message">${formatResponse(content)}</div>
      </div>
    `;
  }

  function formatResponse(text) {
    const safe = escapeHtml(text || "");
    return safe
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n{3,}/g, "\n\n")
      .replace(/\n/g, "<br>");
  }

  function formatSnippet(text) {
    return escapeHtml(text || "")
      .replace(/&lt;mark&gt;/g, "<mark>")
      .replace(/&lt;\/mark&gt;/g, "</mark>");
  }

  function formatCurrency(value) {
    const num = Number(value || 0);
    return num.toLocaleString("en-AU", { style: "currency", currency: "AUD", maximumFractionDigits: 0 });
  }

  function formatPercent(value) {
    const num = Number(value || 0);
    return `${num.toFixed(1)}%`;
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
    return `
      <div class="sopal-shell">
        ${Sidebar()}
        <main class="main">
          ${MainHeader(route)}
          <div class="content">${pageForRoute(route)}</div>
          <footer class="footer-disclaimer">Sopal assists with legal and contract analysis but does not replace professional legal advice.</footer>
        </main>
      </div>
    `;
  }

  function bindShellEvents() {
    document.querySelectorAll("[data-nav]").forEach((link) => {
      link.addEventListener("click", (event) => {
        const href = link.getAttribute("href");
        if (!href || !href.startsWith("/sopal-v2")) return;
        event.preventDefault();
        sidebarOpen = false;
        navigate(href);
      });
    });

    document.querySelectorAll("[data-go]").forEach((button) => {
      button.addEventListener("click", () => {
        sidebarOpen = false;
        navigate(button.getAttribute("data-go"));
      });
    });

    const toggle = document.querySelector("[data-toggle-sidebar]");
    if (toggle) {
      toggle.addEventListener("click", () => {
        sidebarOpen = !sidebarOpen;
        render();
      });
    }

    bindFileZones(document);
  }

  function render() {
    const route = cleanPath();
    root.innerHTML = SopalV2Shell(route);
    bindShellEvents();
  }

  window.addEventListener("popstate", render);
  render();

  window.SopalV2Shell = SopalV2Shell;
  window.Sidebar = Sidebar;
  window.MainHeader = MainHeader;
  window.ChatPanel = ChatPanel;
  window.AgentPage = AgentPage;
  window.ResearchPage = ResearchPage;
  window.ToolPage = ToolPage;
  window.ProjectPage = ProjectPage;
  window.EmptyState = EmptyState;
  window.FileDropzone = FileDropzone;
})();
