/* Sopal v2 workspace — isolated client. Only touches /api/sopal-v2/* + the
   existing read-only Sopal endpoints (/api/adjudicators, /api/adjudicator/{name},
   /api/decision-text, /get_interest_rate, /open). No live Sopal pages or JS
   are modified by this file. */
(function () {
  "use strict";

  const root = document.getElementById("sopal-v2-root");
  const STORE_KEY = "sopal-v2-workspace-v2";

  /* ---------- Static config ---------- */

  const AGENT_KEYS = [
    "payment-claims",
    "payment-schedules",
    "eots",
    "variations",
    "delay-costs",
    "adjudication-application",
    "adjudication-response",
  ];
  const AGENT_LABELS = {
    "payment-claims": "Payment Claims",
    "payment-schedules": "Payment Schedules",
    eots: "EOTs",
    variations: "Variations",
    "delay-costs": "Delay Costs",
    "adjudication-application": "Adjudication Application",
    "adjudication-response": "Adjudication Response",
  };
  const AGENT_DESCRIPTIONS = {
    "payment-claims": "Review or draft payment claim material — BIF Act compliance, work identification, dates, service, evidence.",
    "payment-schedules": "Review or draft payment schedules — scheduled amount, withholding reasons, timing, adjudication risk.",
    eots: "Review or draft extension of time notices and claims — contract notice, causation, critical delay, evidence.",
    variations: "Review or draft variation notices and claims — direction, scope, valuation, time/cost impact, evidence.",
    "delay-costs": "Review or draft delay cost / prolongation / disruption claims — entitlement, causation, quantum.",
    "adjudication-application": "Review or draft adjudication application material — chronology, jurisdiction, entitlement, quantum, annexures.",
    "adjudication-response": "Review or draft adjudication response material — jurisdictional objections, payment schedule alignment, evidence.",
  };
  // Agents in this set drop the Review tab entirely — you draft submissions,
  // you don't structurally "review" your own draft adjudication piece the same
  // way you review someone's payment claim. They go straight into Draft mode.
  const DRAFT_ONLY_AGENTS = new Set(["adjudication-application", "adjudication-response"]);

  // Per-agent review modes. Mirrors the live /claim-check checker — each mode
  // is a distinct perspective ("about to serve" / "received") with its own
  // checklist of structured items.
  const AGENT_REVIEW_MODES = {
    "payment-claims": [
      { id: "serving", label: "About to serve", sub: "Check a payment claim before you serve it." },
      { id: "received", label: "Received", sub: "Audit a payment claim served on you." },
    ],
    "payment-schedules": [
      { id: "giving", label: "About to give", sub: "Check a payment schedule before issuing it." },
      { id: "received", label: "Received", sub: "Audit a payment schedule served on you." },
    ],
    eots: [
      { id: "serving", label: "About to serve", sub: "Stress-test an EOT notice or claim before you send it." },
      { id: "received", label: "Received", sub: "Audit an EOT notice or claim served on you." },
    ],
    variations: [
      { id: "serving", label: "About to serve", sub: "Check a variation notice or claim before you send it." },
      { id: "received", label: "Received", sub: "Audit a variation notice or claim served on you." },
    ],
    "delay-costs": [
      { id: "serving", label: "About to serve", sub: "Stress-test a delay cost / prolongation claim before you send it." },
      { id: "received", label: "Received", sub: "Audit a delay cost claim served on you." },
    ],
  };

  // Each review mode has a list of structured check categories. The AI must
  // return its analysis broken across these so the right pane can render the
  // claim-check style accordion. Keep titles tight so they fit one line.
  const REVIEW_CHECKS = {
    "payment-claims": [
      "Is the document a valid payment claim?",
      "Identification of the construction work or related goods/services",
      "Amount claimed and basis for the amount",
      "Reference date validity and repeat-claim risk",
      "Service: who, when, how, evidence of service",
      "Statutory endorsement / required content (s 68)",
      "Supporting documents and evidence schedule",
      "Practical amendments and next steps",
    ],
    "payment-schedules": [
      "Was the schedule given within time (s 76 — 15 BD or contract)?",
      "Scheduled amount and clarity",
      "Reasons for withholding — adequacy and itemisation",
      "Reasons not properly raised vs s 82(4) risk",
      "Identification of the payment claim being responded to",
      "Reservation of rights and standard endorsements",
      "Adjudication-risk view (claimant vs respondent)",
      "Practical amendments and next steps",
    ],
    eots: [
      "Trigger event — qualifying delay event under the contract",
      "Notice timing — did it meet the contractual deadline?",
      "Causation — link between event and critical-path delay",
      "Delay period — how it has been measured",
      "Programme and float analysis evidence",
      "Supporting documentation (correspondence, photos, RFIs)",
      "Time-bar risk and waiver / estoppel arguments",
      "Whether the claim should also raise variation or delay costs",
    ],
    variations: [
      "Direction or instruction — is there a variation in fact?",
      "Contractual basis (variation clause, scope, gateway)",
      "Notice compliance — content and timing",
      "Valuation method and rates / day-work substantiation",
      "Time impact — separate EOT needed?",
      "Supporting evidence (cost build-up, quotes, dockets)",
      "Time-bar / waiver risk",
      "Reservation of rights and next steps",
    ],
    "delay-costs": [
      "Entitlement basis (contract clause, breach, prevention)",
      "Compensable vs non-compensable delay periods",
      "Causation and concurrent-delay analysis",
      "Quantum methodology (preliminaries, Hudson, measured-mile)",
      "Quantum substantiation (records, payroll, plant)",
      "Overlap / duplication with EOT / variation claims",
      "Notice compliance and time bars",
      "Reservation of rights and next steps",
    ],
  };

  const REVIEW_PROMPT_HINTS = {
    "payment-claims:serving": "Paste the draft payment claim text. Include claimed amount, claim date, intended service date, contract reference, and any prior claim dates if relevant.",
    "payment-claims:received": "Paste the payment claim text you received. Include the date and method of service, the contract reference, and any prior claims you've received from this claimant.",
    "payment-schedules:giving": "Paste the draft schedule, the payment claim it responds to, the date the claim was received, and the relevant contract clauses.",
    "payment-schedules:received": "Paste the schedule received, the original payment claim, and the contract clauses governing the response.",
    "eots:serving": "Paste the draft EOT notice / claim, the contract EOT clause (e.g. cl 34), the trigger-event details, the notice date, and any programme analysis.",
    "eots:received": "Paste the EOT notice / claim received, the contract EOT clause, your view on causation, and the programme analysis.",
    "variations:serving": "Paste the draft variation notice / claim, the contract variation clause, the instruction or direction, and the cost / time impact build-up.",
    "variations:received": "Paste the variation notice / claim received, the contract clause, the instruction or direction, and your view on entitlement.",
    "delay-costs:serving": "Paste the draft delay cost / prolongation claim, the contract entitlement clause, causation facts, and the quantum build-up.",
    "delay-costs:received": "Paste the delay cost claim received, the contract entitlement clause, your view on causation, and any concurrent delay arguments.",
  };

  const INCLUDE_LISTS = {
    "payment-claims": ["Payment claim text", "Contract clauses", "Date served / received", "Reference date", "Prior claims if relevant", "Invoices and supporting schedules"],
    "payment-schedules": ["Payment claim being answered", "Draft / current schedule", "Scheduled amount", "Reasons for withholding", "Date claim received", "Contract payment clauses"],
    eots: ["Contract EOT clause", "Delay event", "Notice date", "Delay period", "Programme / critical path facts", "Supporting correspondence / photos"],
    variations: ["Contract variation clause", "Instruction or direction", "Changed scope", "Notice date", "Valuation material", "Time impact facts"],
    "delay-costs": ["Entitlement clause", "Delay event and period", "Causation facts", "Quantum calculation", "Notice correspondence", "Overlap / duplication checks"],
    "adjudication-application": ["Payment claim", "Payment schedule", "Contract", "Chronology", "Evidence bundle", "Quantum / supporting calculations"],
    "adjudication-response": ["Application", "Payment schedule", "Contract", "Jurisdictional objections", "Evidence responding to each item", "Reasons already raised"],
  };

  const SCENARIO_STARTERS = {
    review: [
      "Review this document for BIF Act compliance and identify any fatal issues.",
      "Identify the strongest jurisdictional objections and weakest claim items.",
      "List the missing evidence I should request before progressing.",
      "Check timing — has the document been served and dated correctly under the BIF Act?",
      "Stress-test the document the way a respondent's lawyer would.",
    ],
    draft: [
      "Draft a payment claim for [scope of work] for $[amount] under contract [name].",
      "Draft an extension of time notice for [event] under clause [#].",
      "Draft a variation notice for [scope change] valued at $[amount].",
      "Draft an adjudication application structure responding to a $0 schedule.",
      "Draft a covering email serving the attached document on the respondent.",
    ],
    assistant: [
      "Summarise the key risks in my contract.",
      "What are my next steps after receiving this payment schedule?",
      "Walk me through the timeline for an adjudication application under the BIF Act.",
      "What's the difference between a variation and a delay cost claim?",
      "Help me draft a short response to a Show Cause notice.",
    ],
  };

  // QLD public holidays + regional show holidays — copied from the live Sopal due-date calculator.
  const HOLIDAYS = {
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

  const LOCATION_OPTIONS = [
    ["bne", "Brisbane"], ["gld", "Gold Coast"], ["sunshine_coast", "Sunshine Coast"], ["tsw", "Townsville"],
    ["cns", "Cairns"], ["toowoomba", "Toowoomba"], ["mackay", "Mackay"], ["rockhampton", "Rockhampton"],
    ["ipswich", "Ipswich"], ["qld", "Queensland (general)"],
  ];

  const CONTRACT_FORMS = ["AS 4000", "AS 4902", "AS 2124", "AS 4300", "AS 4905", "GC21", "MW21", "Bespoke", "Other"];

  /* ---------- State ---------- */

  let store = loadStore();
  let modal = null; // { render(): string, bind(root): void, close(): void }
  let projectMenuOpen = false;
  let sidebarOpen = false;

  function emptyStore() { return { projects: {}, currentProjectId: null }; }
  function loadStore() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
      if (parsed && parsed.projects) return parsed;
    } catch {}
    return emptyStore();
  }
  function saveStore() { localStorage.setItem(STORE_KEY, JSON.stringify(store)); }

  function getProject(id) { return id ? store.projects[id] || null : null; }
  function projectList() { return Object.values(store.projects).sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)); }
  function currentProject() { return getProject(store.currentProjectId); }
  function selectProject(id) {
    if (!store.projects[id]) return;
    store.currentProjectId = id;
    saveStore();
  }
  function newProjectId() { return `p_${Math.random().toString(36).slice(2, 10)}`; }
  function createProject(input) {
    const id = newProjectId();
    const now = Date.now();
    const project = {
      id,
      name: (input.name || "Untitled project").trim(),
      claimant: (input.claimant || "").trim(),
      respondent: (input.respondent || "").trim(),
      contractForm: input.contractForm || "Bespoke",
      reference: (input.reference || "").trim(),
      userIsParty: input.userIsParty || "claimant",
      contracts: [],
      library: [],
      chats: {},
      createdAt: now,
      updatedAt: now,
    };
    store.projects[id] = project;
    store.currentProjectId = id;
    saveStore();
    return project;
  }
  function saveProject(p) {
    p.updatedAt = Date.now();
    store.projects[p.id] = p;
    saveStore();
  }
  function deleteProject(id) {
    delete store.projects[id];
    if (store.currentProjectId === id) store.currentProjectId = projectList()[0]?.id || null;
    saveStore();
  }
  function seedSampleProject() {
    // Onboarding shortcut for the empty-home state. Creates a fictional but
    // realistic project so a first-time user can immediately try every agent
    // without hand-typing setup. Drops the user straight into the assistant
    // page so they can see project context working.
    const project = createProject({
      name: "Queen Street Tower — Stage 2 (sample)",
      claimant: "Acme Builders Pty Ltd",
      respondent: "QH Group Pty Ltd",
      contractForm: "AS 4902",
      reference: "PO-2024-014",
      userIsParty: "claimant",
    });
    project.contracts = [{
      name: "Head contract — extension of time clause",
      text: "Clause 34.3 — Extension of time. The Contractor must, within 10 business days of becoming aware of a delay event qualifying for an extension of time under clause 34.2, give the Superintendent written notice setting out the cause and likely effect of the delay. Failure to give notice within that period is a bar to any extension of time claim.",
      source: "sample",
      addedAt: new Date().toISOString(),
    }];
    project.library = [{
      name: "RFI 014 — Latent ground condition",
      text: "RFI 014 (4 May 2026): During pile installation for Building B on 28 April 2026 the Contractor encountered fill below the predicted soft-clay layer that was not characterised in the geotechnical report. Pile design assumptions no longer valid. Awaiting structural engineer's redesign.",
      source: "sample",
      addedAt: new Date().toISOString(),
    }];
    saveProject(project);
    navigate(`/sopal-v2/projects/${project.id}/overview`);
  }
  function projectChat(p, key) {
    if (!p.chats[key]) p.chats[key] = { messages: [] };
    return p.chats[key];
  }

  /* ---------- Tiny helpers ---------- */

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }
  function attr(value) { return escapeHtml(value); }
  function pct(n) { return `${Number(n || 0).toFixed(1)}%`; }
  function formatCurrencyCompact(n) {
    const num = Number(n || 0);
    return num.toLocaleString("en-AU", { style: "currency", currency: "AUD", maximumFractionDigits: 0 });
  }
  function formatCurrencyFull(n) {
    const num = Number(n || 0);
    return num.toLocaleString("en-AU", { style: "currency", currency: "AUD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function formatCurrencyMicro(n) {
    const num = Number(n || 0);
    return num.toLocaleString("en-AU", { style: "currency", currency: "AUD", minimumFractionDigits: 4, maximumFractionDigits: 4 });
  }
  function money(value) {
    if (value === null || value === undefined || value === "" || value === "N/A") return "";
    const n = Number(value);
    return Number.isFinite(n) && n > 0 ? formatCurrencyCompact(n) : "";
  }
  function parseDate(value) {
    if (!value) return null;
    const date = new Date(`${value}T00:00:00`);
    return Number.isNaN(date.getTime()) ? null : date;
  }
  function formatDate(d) {
    if (!d) return "";
    const months = ["January","February","March","April","May","June","July","August","September","October","November","December"];
    const days = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
    return `${days[d.getDay()]} ${d.getDate()} ${months[d.getMonth()]} ${d.getFullYear()}`;
  }
  function shortDate(value) {
    const d = parseDate(value) || new Date(value);
    if (!d || Number.isNaN(d.getTime())) return value || "";
    return d.toLocaleDateString("en-AU");
  }
  function formatSnippet(text) {
    return escapeHtml(text || "").replace(/&lt;mark&gt;/g, "<mark>").replace(/&lt;\/mark&gt;/g, "</mark>");
  }

  /* ---------- Markdown (line-based) ---------- */

  function renderMarkdown(text) {
    const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let mode = null;
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
    for (const raw of lines) {
      const line = raw.replace(/\s+$/, "");
      if (!line.trim()) { flushPara(); flushList(); continue; }
      const h = line.match(/^(#{1,6})\s+(.+)$/);
      if (h) {
        flushPara(); flushList();
        const level = Math.min(h[1].length + 2, 6);
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
    flushPara(); flushList();
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

  /* ---------- Routing ---------- */

  function cleanPath() {
    const path = window.location.pathname.replace(/\/+$/, "");
    return path.replace(/^\/sopal-v2\/?/, "") || "home";
  }
  function navigate(href) {
    window.history.pushState({}, "", href);
    render();
  }
  function isActiveExact(href) {
    const current = window.location.pathname.replace(/\/+$/, "") || "/sopal-v2";
    return current === href;
  }
  function isActivePrefix(href) {
    const current = window.location.pathname.replace(/\/+$/, "") || "/sopal-v2";
    return current === href || current.startsWith(href + "/");
  }

  /* ---------- Sidebar ---------- */

  const ICON = {
    search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    users: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    calendar: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>',
    coins: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="8" r="6"/><path d="M18.09 10.37A6 6 0 1 1 10.34 18M7 6h1v4M16.71 13.88l.7.71-2.82 2.82"/></svg>',
    home: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/></svg>',
    file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>',
    chat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    sparkles: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5z"/><path d="M19 15l.5 1.5L21 17l-1.5.5L19 19l-.5-1.5L17 17l1.5-.5z"/></svg>',
    plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>',
    chevDown: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>',
    chevRight: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>',
    settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M10 11v6M14 11v6"/></svg>',
    upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8 12 3 7 8"/><path d="M12 3v12"/></svg>',
    paperclip: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m21 12-9.5 9.5a5 5 0 0 1-7.1-7.1L14 4.7a3.5 3.5 0 0 1 4.95 4.95L9.5 19.07a2 2 0 0 1-2.83-2.83L16 6.9"/></svg>',
    send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m22 2-7 20-4-9-9-4z"/><path d="m22 2-11 11"/></svg>',
    book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    grid: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>',
    arrowUpRight: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17 17 7"/><path d="M7 7h10v10"/></svg>',
    close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
    copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
    clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>',
    layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 10 6-10 6L2 8z"/><path d="m2 14 10 6 10-6"/></svg>',
  };

  function workspaceNav() {
    return [
      { label: "Decision Search", href: "/sopal-v2/research/decisions", icon: ICON.search },
      { label: "Adjudicator Statistics", href: "/sopal-v2/research/adjudicators", icon: ICON.users },
      { label: "Due Date Calculator", href: "/sopal-v2/tools/due-date-calculator", icon: ICON.calendar },
      { label: "Interest Calculator", href: "/sopal-v2/tools/interest-calculator", icon: ICON.coins },
    ];
  }

  function projectSubNav(projectId) {
    const base = `/sopal-v2/projects/${projectId}`;
    return [
      { label: "Overview", href: `${base}/overview`, icon: ICON.home },
      { label: "Contract", href: `${base}/contract`, icon: ICON.file },
      { label: "Project Library", href: `${base}/library`, icon: ICON.folder },
      { label: "Assistant", href: `${base}/assistant`, icon: ICON.chat },
    ];
  }

  function projectAgentNav(projectId) {
    const base = `/sopal-v2/projects/${projectId}/agents`;
    return AGENT_KEYS.map((key) => ({ label: AGENT_LABELS[key], href: `${base}/${key}`, icon: ICON.sparkles }));
  }

  function Sidebar() {
    const project = currentProject();
    const projects = projectList();
    return `
      <aside class="sopal-sidebar ${sidebarOpen ? "open" : ""}">
        <div class="sidebar-brand"><a href="/sopal-v2" data-nav>Sopal</a><span class="brand-pill">v2</span></div>

        <div class="sidebar-scroll">
          <div class="nav-group-title">Workspace</div>
          ${workspaceNav().map((item) => `
            <a class="nav-item ${isActivePrefix(item.href) ? "active" : ""}" href="${item.href}" data-nav>
              <span class="nav-icon">${item.icon}</span>
              <span class="nav-label">${escapeHtml(item.label)}</span>
            </a>`).join("")}

          <div class="nav-divider"></div>

          <div class="nav-group-title row">
            <span>Projects</span>
            <button class="icon-button" type="button" data-new-project title="New project">${ICON.plus}</button>
          </div>

          ${projects.length === 0 ? `
            <div class="sidebar-empty">
              <p>No projects yet.</p>
              <button class="ghost-button compact" type="button" data-new-project>+ New project</button>
            </div>
          ` : `
            <div class="project-switcher">
              <button class="project-switcher-trigger" type="button" data-project-menu-toggle>
                <span class="truncate">${project ? escapeHtml(project.name) : "Select a project"}</span>
                <span class="chev">${ICON.chevDown}</span>
              </button>
              ${projectMenuOpen ? `
                <div class="project-menu" role="menu">
                  ${projects.map((p) => `
                    <button class="project-menu-row ${p.id === store.currentProjectId ? "active" : ""}" type="button" data-select-project="${attr(p.id)}">
                      <span class="truncate">${escapeHtml(p.name)}</span>
                    </button>`).join("")}
                  <div class="project-menu-divider"></div>
                  <button class="project-menu-row create" type="button" data-new-project>${ICON.plus}<span>New project</span></button>
                </div>` : ""}
            </div>

            ${project ? `
              ${projectSubNav(project.id).map((item) => `
                <a class="nav-item ${isActivePrefix(item.href) ? "active" : ""}" href="${item.href}" data-nav>
                  <span class="nav-icon">${item.icon}</span>
                  <span class="nav-label">${escapeHtml(item.label)}</span>
                </a>`).join("")}
              <div class="nav-subgroup-title">Agents</div>
              ${projectAgentNav(project.id).map((item) => `
                <a class="nav-item nav-item-sub ${isActivePrefix(item.href) ? "active" : ""}" href="${item.href}" data-nav>
                  <span class="nav-icon">${item.icon}</span>
                  <span class="nav-label">${escapeHtml(item.label)}</span>
                </a>`).join("")}
            ` : ""}
          `}
        </div>

        <div class="sidebar-foot">
          <a class="nav-item small" href="/sopal-v2/projects" data-nav>
            <span class="nav-icon">${ICON.grid}</span>
            <span class="nav-label">Your projects</span>
          </a>
        </div>
      </aside>
    `;
  }

  /* ---------- Top header ---------- */

  function MainHeader(crumbs) {
    return `
      <header class="main-header">
        <div class="header-left">
          <button class="ghost-button mobile-toggle" type="button" data-toggle-sidebar aria-label="Open menu">Menu</button>
          <nav class="breadcrumb">
            ${crumbs.map((c, i) => {
              const last = i === crumbs.length - 1;
              if (last) return `<span class="crumb current">${escapeHtml(c.label)}</span>`;
              const sep = `<span class="crumb-sep">${ICON.chevRight}</span>`;
              return c.href
                ? `<a class="crumb" href="${attr(c.href)}" data-nav>${escapeHtml(c.label)}</a>${sep}`
                : `<span class="crumb">${escapeHtml(c.label)}</span>${sep}`;
            }).join("")}
          </nav>
        </div>
        <div class="header-right">
          <a class="link-button small" href="https://sopal.com.au" target="_blank" rel="noopener">sopal.com.au</a>
        </div>
      </header>
    `;
  }

  function PageBody(content) {
    return `<div class="page-body">${content}</div>`;
  }

  /* ---------- Empty state ---------- */

  function EmptyState(title, body, actions) {
    return `<div class="empty-state">
      <strong>${escapeHtml(title)}</strong>
      <p>${escapeHtml(body)}</p>
      ${actions || ""}
    </div>`;
  }

  /* ---------- Home ---------- */

  function HomePage() {
    const tools = workspaceNav();
    const projects = projectList();
    return PageBody(`
      <div class="home-shell">
        <section class="home-hero">
          <h2>Welcome to Sopal v2</h2>
          <p>Search adjudication decisions, run BIF Act calculators, and manage SOPA workflows project by project.</p>
        </section>

        <section class="home-section">
          <div class="section-head"><h3>Workspace tools</h3><p>Available everywhere — no project required.</p></div>
          <div class="tile-grid">
            ${tools.map((t) => `
              <a class="tile" href="${t.href}" data-nav>
                <span class="tile-icon">${t.icon}</span>
                <strong>${escapeHtml(t.label)}</strong>
              </a>`).join("")}
          </div>
        </section>

        <section class="home-section">
          <div class="section-head row">
            <div><h3>Your projects</h3><p>Each project is one construction contract — head contract or subcontract.</p></div>
            <button class="dark-button" type="button" data-new-project>${ICON.plus}<span>New project</span></button>
          </div>
          ${projects.length === 0 ? `
            <div class="card-empty">
              <div class="card-empty-icon">${ICON.file}</div>
              <h4>Create your first project</h4>
              <p>Add the contract details, paste in clauses or upload your contract — Sopal then runs every agent (Payment Claims, EOTs, Adjudication etc.) inside that project's context.</p>
              <div class="card-empty-actions">
                <button class="dark-button" type="button" data-new-project>Create project</button>
                <button class="ghost-button" type="button" data-seed-sample>Try a sample project</button>
              </div>
              <p class="muted card-empty-hint">The sample is a fictional Queen Street Tower head contract under AS 4902 — pre-loaded with a contract clause and an RFI so you can immediately try the agents.</p>
            </div>
          ` : `
            <div class="project-list">
              ${projects.map((p) => projectRow(p)).join("")}
            </div>
          `}
        </section>
      </div>
    `);
  }

  function projectRow(p) {
    const meta = [p.reference, p.contractForm, p.claimant ? `${p.claimant} v ${p.respondent || "?"}` : ""].filter(Boolean).join(" · ");
    return `
      <a class="project-row" href="/sopal-v2/projects/${attr(p.id)}/overview" data-nav>
        <div class="project-row-icon">${ICON.file}</div>
        <div class="project-row-text">
          <strong>${escapeHtml(p.name)}</strong>
          <span>${escapeHtml(meta || "Bespoke contract")}</span>
        </div>
        <span class="status-pill">${p.contracts.length || 0} contract docs · ${p.library.length || 0} library</span>
        <span class="row-chev">${ICON.chevRight}</span>
      </a>
    `;
  }

  /* ---------- Research: Decision search ---------- */

  function DecisionsPage() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q") || "";
    const sort = params.get("sort") || "relevance";
    const filtersOn = ["startDate", "endDate", "minClaim", "maxClaim"].some((k) => params.get(k));
    setTimeout(() => {
      const form = document.querySelector("[data-decision-search]");
      if (form) form.addEventListener("submit", (event) => {
        event.preventDefault();
        const data = new FormData(form);
        const next = new URLSearchParams();
        if (data.get("q")) next.set("q", data.get("q"));
        next.set("sort", data.get("sort") || "relevance");
        ["startDate", "endDate", "minClaim", "maxClaim"].forEach((k) => { if (data.get(k)) next.set(k, data.get(k)); });
        navigate(`/sopal-v2/research/decisions?${next.toString()}`);
      });
      if (q || params.toString()) fetchDecisionResults(params, 0);
    }, 0);

    return PageBody(`
      <div class="page-shell">
        <h1 class="page-title">Decision search</h1>
        <p class="page-sub">Searches Sopal's adjudication decision database. Results render here — no jumps to the live site.</p>

        <div class="card">
          <form class="search-form" data-decision-search>
            <input class="text-input span-all" name="q" type="search" value="${attr(q)}" placeholder="Adjudicator, party, section, keywords…" autofocus>
            <select class="select-input" name="sort">
              ${["relevance","newest","oldest","claim_high","claim_low","adj_high","adj_low"].map((s) => `<option value="${s}" ${sort===s?"selected":""}>${labelSort(s)}</option>`).join("")}
            </select>
            <button class="dark-button" type="submit">Search</button>
            <details class="filters" ${filtersOn ? "open" : ""}>
              <summary>Filters${filtersOn ? " · active" : ""}</summary>
              <div class="filters-grid">
                <label>From<input class="text-input" name="startDate" type="date" value="${attr(params.get("startDate") || "")}"></label>
                <label>To<input class="text-input" name="endDate" type="date" value="${attr(params.get("endDate") || "")}"></label>
                <label>Min claimed<input class="text-input" name="minClaim" type="number" step="1000" value="${attr(params.get("minClaim") || "")}"></label>
                <label>Max claimed<input class="text-input" name="maxClaim" type="number" step="1000" value="${attr(params.get("maxClaim") || "")}"></label>
              </div>
            </details>
          </form>
        </div>

        <div class="research-grid">
          <section id="decision-results">${q ? skeletonRows() : EmptyState("Enter a search.", "Try an adjudicator name, a party, a section reference, or keywords from a decision.")}</section>
          <aside id="decision-detail" class="card detail-panel">${EmptyState("Select a decision.", "Click any result to view the full text inline.")}</aside>
        </div>
      </div>
    `);
  }

  function labelSort(value) {
    return ({ relevance: "Relevance", newest: "Newest", oldest: "Oldest", claim_high: "Claimed: high → low", claim_low: "Claimed: low → high", adj_high: "Awarded: high → low", adj_low: "Awarded: low → high" })[value] || value;
  }

  function skeletonRows() {
    return `<div class="card"><div class="card-body"><div class="skeleton-row"></div><div class="skeleton-row"></div><div class="skeleton-row"></div></div></div>`;
  }

  async function fetchDecisionResults(params, offset) {
    const mount = document.getElementById("decision-results");
    if (!mount) return;
    mount.innerHTML = skeletonRows();
    const qs = new URLSearchParams(params);
    qs.set("limit", "20");
    qs.set("offset", String(offset || 0));
    try {
      const response = await fetch(`/api/sopal-v2/search?${qs.toString()}`, { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.error || "Search failed");
      const items = Array.isArray(data.items) ? data.items : [];
      const total = Number(data.total || items.length);
      if (!items.length) {
        mount.innerHTML = EmptyState("No decisions match.", "Adjust your query or filters.");
        return;
      }
      mount.innerHTML = `
        <div class="card">
          <div class="card-head"><h3>${total.toLocaleString()} result${total === 1 ? "" : "s"}</h3></div>
          <div class="card-body results-list">${items.map(renderDecisionItem).join("")}</div>
          ${items.length < total ? `<div class="card-foot"><button class="ghost-button" type="button" data-load-more="${(offset || 0) + items.length}">Load more (${(total - ((offset || 0) + items.length)).toLocaleString()} remaining)</button></div>` : ""}
        </div>`;
      mount.querySelectorAll("[data-decision-id]").forEach((el) => el.addEventListener("click", () => {
        const meta = el.dataset.meta ? safeParseJson(el.dataset.meta) : null;
        loadDecisionDetail(el.dataset.decisionId, el.dataset.title, meta);
      }));
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
    const foot = mount.querySelector(".card-foot");
    const qs = new URLSearchParams(params);
    qs.set("limit", "20");
    qs.set("offset", String(offset));
    try {
      const response = await fetch(`/api/sopal-v2/search?${qs.toString()}`, { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Search failed");
      const items = Array.isArray(data.items) ? data.items : [];
      list.insertAdjacentHTML("beforeend", items.map(renderDecisionItem).join(""));
      list.querySelectorAll("[data-decision-id]:not([data-bound])").forEach((el) => {
        el.dataset.bound = "1";
        el.addEventListener("click", () => {
          const meta = el.dataset.meta ? safeParseJson(el.dataset.meta) : null;
          loadDecisionDetail(el.dataset.decisionId, el.dataset.title, meta);
        });
      });
      const total = Number(data.total || 0);
      const newOffset = offset + items.length;
      if (newOffset >= total || !items.length) { if (foot) foot.remove(); }
      else if (foot) foot.querySelector("[data-load-more]").dataset.loadMore = String(newOffset);
    } catch (error) {
      if (foot) foot.innerHTML = `<div class="error-banner">${escapeHtml(error.message)}</div>`;
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
    // Stash a tiny meta blob on the row so the detail panel can render a rich
    // header without a second round-trip.
    const detailMeta = JSON.stringify({
      title, id,
      claimant, respondent,
      decisionDate: item.decision_date || item.decision_date_norm || "",
      adjudicator: item.adjudicator_name || item.adjudicator || "",
      act: item.act_category || item.act || "",
      claimed: money(item.claimed_amount) || "",
      awarded: money(item.adjudicated_amount) || "",
    });
    return `
      <article class="result-row" data-decision-id="${attr(id)}" data-title="${attr(title)}" data-meta="${attr(detailMeta)}" tabindex="0">
        <h4>${escapeHtml(title)}</h4>
        <div class="result-meta">${meta.map((m) => `<span>${escapeHtml(m)}</span>`).join("")}</div>
        <p>${formatSnippet(item.snippet)}</p>
      </article>`;
  }

  function safeParseJson(s) {
    try { return JSON.parse(s); } catch { return null; }
  }

  // Standalone deep-link page so users can share /sopal-v2/research/decisions/{ejs_id}
  function DecisionDetailPage(decisionId) {
    setTimeout(() => loadDeepDecision(decisionId), 0);
    return PageBody(`
      <div class="page-shell">
        <a class="link-button small" href="/sopal-v2/research/decisions" data-nav>← Back to decision search</a>
        <section id="decision-detail" class="card detail-panel" style="position:static;max-height:none;">${skeletonRows()}</section>
      </div>
    `);
  }

  async function loadDeepDecision(decisionId) {
    // First try to enrich with metadata via a quick search by decision id
    // (the decision-text endpoint returns text only).
    try {
      const r = await fetch(`/api/sopal-v2/search?q=${encodeURIComponent(decisionId)}&limit=1`, { credentials: "include" });
      const data = await r.json().catch(() => ({}));
      const hit = (data.items || []).find((it) => (it.ejs_id || it.id) === decisionId) || (data.items || [])[0];
      let meta = null;
      let title = decisionId;
      if (hit) {
        const claimant = hit.claimant_name || hit.claimant || "";
        const respondent = hit.respondent_name || hit.respondent || "";
        title = [claimant, respondent].filter(Boolean).join(" v ") || decisionId;
        meta = {
          title, id: decisionId,
          claimant, respondent,
          decisionDate: hit.decision_date || hit.decision_date_norm || "",
          adjudicator: hit.adjudicator_name || hit.adjudicator || "",
          act: hit.act_category || hit.act || "",
          claimed: money(hit.claimed_amount) || "",
          awarded: money(hit.adjudicated_amount) || "",
        };
      }
      await loadDecisionDetail(decisionId, title, meta);
    } catch (error) {
      const mount = document.getElementById("decision-detail");
      if (mount) mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load decision")}</div>`;
    }
  }

  function AdjudicatorDetailPage(name) {
    setTimeout(() => loadDeepAdjudicator(name), 0);
    return PageBody(`
      <div class="page-shell">
        <a class="link-button small" href="/sopal-v2/research/adjudicators" data-nav>← Back to adjudicator statistics</a>
        <section id="adj-detail" class="card detail-panel" style="position:static;max-height:none;">${skeletonRows()}</section>
      </div>
    `);
  }

  async function loadDeepAdjudicator(name) {
    // Need the adjudicators list for the summary (avg award rate, fee shares).
    if (!window.__sopalAdjudicators) {
      try {
        const r = await fetch("/api/adjudicators", { credentials: "include" });
        window.__sopalAdjudicators = await r.json().catch(() => []);
      } catch { window.__sopalAdjudicators = []; }
    }
    await loadAdjudicatorDetail(name);
  }

  async function loadDecisionDetail(id, title, meta) {
    const mount = document.getElementById("decision-detail");
    if (!mount || !id) return;
    mount.innerHTML = `<div class="card-head"><h3>${escapeHtml(title || "Decision")}</h3></div><div class="card-body">${skeletonRows()}</div>`;
    try {
      const response = await fetch(`/api/decision-text/${encodeURIComponent(id)}`, { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Decision text failed");
      const text = (data.fullText || "").trim();
      const metaHeader = meta ? renderDecisionMetaHeader(meta) : "";
      mount.innerHTML = `
        <div class="card-head">
          <div><h3>${escapeHtml(title || id)}</h3><p class="muted">${escapeHtml(id)}</p></div>
          <div class="panel-actions">
            <a class="link-button small" href="/sopal-v2/research/decisions/${encodeURIComponent(id)}" data-nav title="Shareable link to this decision">Open page</a>
            <button class="ghost-button compact" type="button" data-copy-text="${attr(text.slice(0, 8000))}" title="Copy decision text">${ICON.copy}<span>Copy</span></button>
          </div>
        </div>
        ${metaHeader}
        <div class="card-body">
          ${text ? `<div class="decision-text">${formatDecisionText(text)}</div>${text.length > 12000 ? `<p class="muted decision-text-trunc">Text truncated to first 12,000 characters. ${text.length.toLocaleString()} chars total.</p>` : ""}` : EmptyState("No text on file.", "This record has no extracted text.")}
        </div>`;
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load decision text")}</div>`;
    }
  }

  function renderDecisionMetaHeader(meta) {
    const rows = [
      meta.decisionDate ? ["Decision date", meta.decisionDate] : null,
      (meta.claimant || meta.respondent) ? ["Parties", `${escapeHtml(meta.claimant || "?")} v ${escapeHtml(meta.respondent || "?")}`] : null,
      meta.adjudicator ? ["Adjudicator", meta.adjudicator] : null,
      meta.act ? ["Act", meta.act] : null,
      meta.claimed ? ["Claimed", meta.claimed] : null,
      meta.awarded ? ["Awarded", meta.awarded] : null,
    ].filter(Boolean);
    if (!rows.length) return "";
    return `<div class="decision-meta-block">
      ${rows.map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${k === "Parties" ? v : escapeHtml(v)}</dd></div>`).join("")}
    </div>`;
  }

  function formatDecisionText(text) {
    // Decision text is already paragraph-broken by the OCR; preserve newlines
    // and split into <p> blocks so it reads as prose, not a wall of pre.
    const trimmed = (text || "").slice(0, 12000);
    const paras = trimmed.split(/\n{2,}/).filter((p) => p.trim());
    return paras.map((p) => `<p>${escapeHtml(p).replace(/\n/g, "<br>")}</p>`).join("");
  }

  /* ---------- Research: adjudicators ---------- */

  function AdjudicatorsPage() {
    setTimeout(fetchAdjudicators, 0);
    return PageBody(`
      <div class="page-shell">
        <h1 class="page-title">Adjudicator statistics</h1>
        <p class="page-sub">Live data from the Sopal decision database. Click an adjudicator to view their decision history.</p>
        <div class="card">
          <div class="card-body toolbar">
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
        <div class="research-grid">
          <section id="adj-results">${skeletonRows()}</section>
          <aside id="adj-detail" class="card detail-panel">${EmptyState("Select an adjudicator.", "Open any card to see their decisions, totals and award rates rendered here.")}</aside>
        </div>
      </div>
    `);
  }

  async function fetchAdjudicators() {
    const mount = document.getElementById("adj-results");
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
    const mount = document.getElementById("adj-results");
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
      ? `<div class="adj-grid">${items.slice(0, 80).map((item) => `
          <button class="adj-card" type="button" data-adjudicator="${attr(item.name)}">
            <strong>${escapeHtml(item.name)}</strong>
            <span class="muted">${item.totalDecisions} decisions</span>
            <div class="adj-card-row">${formatCurrencyCompact(item.totalClaimAmount)} claimed</div>
            <div class="adj-card-row">${formatCurrencyCompact(item.totalAwardedAmount)} awarded</div>
            <span class="rate-pill">${pct(item.avgAwardRate)} avg award</span>
          </button>`).join("")}</div>`
      : EmptyState("No adjudicators match.", "Clear or change the filter.");
    mount.querySelectorAll("[data-adjudicator]").forEach((b) => b.addEventListener("click", () => loadAdjudicatorDetail(b.dataset.adjudicator)));
  }

  async function loadAdjudicatorDetail(name) {
    const mount = document.getElementById("adj-detail");
    if (!mount) return;
    mount.innerHTML = `<div class="card-head"><h3>${escapeHtml(name)}</h3><p class="muted">Loading decisions…</p></div><div class="card-body">${skeletonRows()}</div>`;
    try {
      const response = await fetch(`/api/adjudicator/${encodeURIComponent(name)}`, { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.error || "Adjudicator detail failed");
      const decisions = Array.isArray(data) ? data : [];
      const summary = (window.__sopalAdjudicators || []).find((x) => x.name === name) || {};
      const claimedSum = decisions.reduce((s, d) => s + (Number(d.claimAmount) || 0), 0);
      const awardedSum = decisions.reduce((s, d) => s + (Number(d.awardedAmount) || 0), 0);
      const zeroes = decisions.filter((d) => Number(d.awardedAmount) === 0).length;
      mount.innerHTML = `
        <div class="card-head"><div><h3>${escapeHtml(name)}</h3><p class="muted">${decisions.length} decision${decisions.length === 1 ? "" : "s"}</p></div></div>
        <div class="card-body">
          <div class="metric-grid compact">
            <div class="metric"><strong>${decisions.length}</strong><span>decisions</span></div>
            <div class="metric"><strong>${formatCurrencyCompact(claimedSum)}</strong><span>total claimed</span></div>
            <div class="metric"><strong>${formatCurrencyCompact(awardedSum)}</strong><span>total awarded</span></div>
            <div class="metric"><strong>${pct(summary.avgAwardRate || 0)}</strong><span>avg award rate</span></div>
            <div class="metric"><strong>${zeroes}</strong><span>$0 awards</span></div>
            <div class="metric"><strong>${pct(summary.avgClaimantFeeProportion || 0)}</strong><span>claimant fee share</span></div>
          </div>
          <div class="mini-list">
            ${decisions.slice(0, 30).map((d) => `
              <article class="mini-item" ${d.id ? `data-decision-id="${attr(d.id)}" data-title="${attr(d.title || "")}" tabindex="0"` : ""}>
                <strong>${escapeHtml(d.title || "Decision")}</strong>
                <span class="muted">${escapeHtml(shortDate(d.date) || "")}${d.outcome ? ` · ${escapeHtml(d.outcome)}` : ""}${d.projectType ? ` · ${escapeHtml(d.projectType)}` : ""}</span>
                <span class="muted">claimed ${formatCurrencyCompact(d.claimAmount)} · awarded ${formatCurrencyCompact(d.awardedAmount)}</span>
              </article>`).join("")}
            ${decisions.length > 30 ? `<div class="muted">${decisions.length - 30} more not shown.</div>` : ""}
          </div>
        </div>`;
      mount.querySelectorAll("[data-decision-id]").forEach((el) => el.addEventListener("click", () => {
        navigate(`/sopal-v2/research/decisions?q=${encodeURIComponent(el.dataset.title || "")}`);
      }));
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load adjudicator")}</div>`;
    }
  }

  /* ---------- Tools: due date calculator (mirrors live Sopal layout) ---------- */

  const DUE_DATE_SCENARIOS = [
    { id: "paymentSchedule", title: "Payment Schedule", subtitle: "When is the schedule due?", section: "s 76 BIF Act" },
    { id: "adjudicationApp", title: "Adjudication Application", subtitle: "Deadline to lodge", section: "s 79 BIF Act" },
    { id: "adjudicationResp", title: "Adjudication Response", subtitle: "Deadline for the response", section: "s 83 BIF Act" },
    { id: "adjudicatorDecision", title: "Adjudicator's Decision", subtitle: "Deadline to deliver", section: "s 85 BIF Act" },
  ];

  function DueDatePage() {
    const params = new URLSearchParams(window.location.search);
    const active = params.get("scenario") && DUE_DATE_SCENARIOS.some((s) => s.id === params.get("scenario")) ? params.get("scenario") : "paymentSchedule";
    setTimeout(() => bindDueDate(active), 0);
    return PageBody(`
      <div class="page-shell">
        <h1 class="page-title">Due date calculator</h1>
        <p class="page-sub">BIF Act business-day deadlines. Excludes weekends, QLD/local public holidays and the s 87 Christmas shutdown.</p>
        <div class="due-grid">
          <div class="scenario-list">
            ${DUE_DATE_SCENARIOS.map((s) => `
              <button class="scenario-card ${s.id === active ? "active" : ""}" type="button" data-scenario="${attr(s.id)}">
                <strong>${escapeHtml(s.title)}</strong>
                <span class="muted">${escapeHtml(s.subtitle)}</span>
                <span class="section-tag">${escapeHtml(s.section)}</span>
              </button>`).join("")}
          </div>
          <div class="card calc-card">
            <div class="card-head">
              <div>
                <h3 id="due-card-title"></h3>
                <p class="muted"><label class="loc-row">Location <select class="select-input compact" id="due-location">${LOCATION_OPTIONS.map(([v,l]) => `<option value="${v}">${escapeHtml(l)}</option>`).join("")}</select></label></p>
              </div>
              <span class="section-badge" id="due-card-badge"></span>
            </div>
            <div class="card-body" id="due-form-mount"></div>
            <div class="card-body" id="due-result"></div>
          </div>
        </div>
      </div>
    `);
  }

  function bindDueDate(initial) {
    let scenario = initial;
    const titleEl = document.getElementById("due-card-title");
    const badgeEl = document.getElementById("due-card-badge");
    const formMount = document.getElementById("due-form-mount");
    const resultMount = document.getElementById("due-result");
    if (!titleEl || !formMount) return;

    function applyScenario(id) {
      scenario = id;
      const meta = DUE_DATE_SCENARIOS.find((s) => s.id === id);
      titleEl.textContent = `${meta.title} — due date`;
      badgeEl.textContent = meta.section;
      document.querySelectorAll("[data-scenario]").forEach((el) => el.classList.toggle("active", el.dataset.scenario === id));
      const url = new URL(window.location.href);
      url.searchParams.set("scenario", id);
      window.history.replaceState({}, "", url);
      resultMount.innerHTML = "";
      formMount.innerHTML = renderDueForm(id);
      bindDueForm(id);
    }

    document.querySelectorAll("[data-scenario]").forEach((el) => el.addEventListener("click", () => applyScenario(el.dataset.scenario)));
    applyScenario(scenario);

    function renderDueForm(id) {
      if (id === "paymentSchedule") {
        return `
          <details class="provision">
            <summary>Relevant provision · s 76 BIF Act</summary>
            <div class="provision-body">
              <p><strong>76 Responding to payment claim.</strong> A respondent must respond by giving the claimant a payment schedule within whichever ends first — (a) the period under the contract; or (b) 15 business days after the payment claim is given.</p>
            </div>
          </details>
          <form class="calc-form" data-due-form>
            <label class="span-2">Date payment claim given<input class="text-input" type="date" name="given"></label>
            <button class="dark-button span-2" type="submit">Calculate due date</button>
          </form>`;
      }
      if (id === "adjudicationApp") {
        return `
          <details class="provision">
            <summary>Relevant provision · s 79 BIF Act</summary>
            <div class="provision-body">
              <p><strong>79 Application for adjudication</strong> — must be made within: (i) for failure to give a schedule and pay the claim — 30 business days after the later of the payment due date or last day a schedule could have been given; (ii) for failure to pay the scheduled amount — 20 business days after the payment due date; (iii) where scheduled amount is less than claimed — 30 business days after receipt of the schedule.</p>
            </div>
          </details>
          <form class="calc-form" data-due-form>
            <label class="span-2">Reason for application
              <select class="select-input" name="aaScenario" data-due-aa-scenario>
                <option value="less" selected>Scheduled amount was less than claimed amount</option>
                <option value="no-pay-schedule">Respondent failed to provide a payment schedule</option>
                <option value="no-pay-amount">Respondent failed to pay the scheduled amount</option>
              </select>
            </label>
            <label class="span-2" data-aa-field="schedule-received">Date payment schedule received<input class="text-input" type="date" name="scheduleReceived"></label>
            <label data-aa-field="payment-due" hidden>Due date for progress payment<input class="text-input" type="date" name="paymentDue"></label>
            <label data-aa-field="schedule-due" hidden>Last day to provide schedule<input class="text-input" type="date" name="scheduleDue"></label>
            <button class="dark-button span-2" type="submit">Calculate due date</button>
          </form>`;
      }
      if (id === "adjudicationResp") {
        return `
          <details class="provision">
            <summary>Relevant provision · s 83 BIF Act</summary>
            <div class="provision-body">
              <p><strong>83 Time for making adjudication response</strong> — Standard claim: later of 10 BD after receiving s 79(4) documents OR 7 BD after acceptance. Complex claim: later of 15 BD OR 12 BD, with up to 15 additional BD extension under s 83(3).</p>
            </div>
          </details>
          <form class="calc-form" data-due-form>
            <label>Date application received<input class="text-input" type="date" name="appReceived"></label>
            <label>Date acceptance received<input class="text-input" type="date" name="acceptanceReceived"></label>
            <div class="span-2">
              <span class="form-label">Claim type</span>
              <div class="radio-group">
                <label class="radio-option"><input type="radio" name="claimType" value="standard" checked>Standard (≤ $750k)</label>
                <label class="radio-option"><input type="radio" name="claimType" value="complex">Complex (> $750k)</label>
              </div>
            </div>
            <div class="span-2 eot-block" data-eot-block hidden>
              <label class="check-line"><input type="checkbox" name="eotEnabled" data-eot-enabled> Extension of time granted by adjudicator (s 83(3))</label>
              <div class="eot-days" data-eot-days hidden>
                <label>Additional business days (1-15)<input class="text-input compact" type="number" name="eotDays" min="1" max="15" value="1"></label>
              </div>
            </div>
            <button class="dark-button span-2" type="submit">Calculate due date</button>
          </form>`;
      }
      if (id === "adjudicatorDecision") {
        return `
          <details class="provision">
            <summary>Relevant provision · s 85 BIF Act</summary>
            <div class="provision-body">
              <p><strong>85 Time for deciding adjudication application</strong> — 10 BD after the response date for a standard claim, 15 BD for a complex claim. Parties may agree to extend under s 86 (any agreed days).</p>
            </div>
          </details>
          <form class="calc-form" data-due-form>
            <label class="span-2">Date adjudication response given<input class="text-input" type="date" name="responseGiven"></label>
            <div class="span-2">
              <span class="form-label">Claim type</span>
              <div class="radio-group">
                <label class="radio-option"><input type="radio" name="claimType" value="standard" checked>Standard (≤ $750k)</label>
                <label class="radio-option"><input type="radio" name="claimType" value="complex">Complex (> $750k)</label>
              </div>
            </div>
            <div class="span-2 eot-block-2">
              <label class="check-line"><input type="checkbox" name="eotEnabled" data-eot-enabled> Extension of time agreed by parties (s 86)</label>
              <div class="eot-days" data-eot-days hidden>
                <label>Extension days<input class="text-input compact" type="number" name="eotDays" min="1" value="1"></label>
                <label>Day type
                  <select class="select-input compact" name="eotDayType">
                    <option value="business">Business days</option>
                    <option value="calendar">Calendar days</option>
                  </select>
                </label>
              </div>
            </div>
            <button class="dark-button span-2" type="submit">Calculate due date</button>
          </form>`;
      }
      return "";
    }

    function bindDueForm(id) {
      const form = formMount.querySelector("[data-due-form]");
      if (!form) return;

      if (id === "adjudicationApp") {
        const sel = form.querySelector("[data-due-aa-scenario]");
        const fields = {
          "schedule-received": form.querySelector('[data-aa-field="schedule-received"]'),
          "payment-due": form.querySelector('[data-aa-field="payment-due"]'),
          "schedule-due": form.querySelector('[data-aa-field="schedule-due"]'),
        };
        const apply = () => {
          Object.values(fields).forEach((f) => { if (f) f.hidden = true; });
          if (sel.value === "less") fields["schedule-received"].hidden = false;
          else if (sel.value === "no-pay-amount") fields["payment-due"].hidden = false;
          else if (sel.value === "no-pay-schedule") { fields["payment-due"].hidden = false; fields["schedule-due"].hidden = false; }
        };
        sel.addEventListener("change", apply);
        apply();
      }

      if (id === "adjudicationResp") {
        const radios = form.querySelectorAll('input[name="claimType"]');
        const eotBlock = form.querySelector("[data-eot-block]");
        const eotEnabled = form.querySelector("[data-eot-enabled]");
        const eotDays = form.querySelector("[data-eot-days]");
        const apply = () => {
          const isComplex = form.querySelector('input[name="claimType"]:checked').value === "complex";
          eotBlock.hidden = !isComplex;
          if (!isComplex) { eotEnabled.checked = false; eotDays.hidden = true; }
        };
        radios.forEach((r) => r.addEventListener("change", apply));
        eotEnabled.addEventListener("change", () => { eotDays.hidden = !eotEnabled.checked; });
        apply();
      }

      if (id === "adjudicatorDecision") {
        const eotEnabled = form.querySelector("[data-eot-enabled]");
        const eotDays = form.querySelector("[data-eot-days]");
        eotEnabled.addEventListener("change", () => { eotDays.hidden = !eotEnabled.checked; });
      }

      form.addEventListener("submit", (event) => {
        event.preventDefault();
        const data = Object.fromEntries(new FormData(form).entries());
        const location = document.getElementById("due-location").value;
        try {
          const result = computeDueDate(id, data, location);
          resultMount.innerHTML = renderDateResult(result);
        } catch (error) {
          resultMount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Calculation failed")}</div>`;
        }
      });
    }
  }

  function computeDueDate(scenario, data, location) {
    if (scenario === "paymentSchedule") {
      const start = parseDate(data.given);
      if (!start) throw new Error("Enter the date the payment claim was given.");
      const r = addBusinessDays(start, 15, location);
      return { title: "Payment Schedule due date", finalDate: r.finalDate, startDate: start, days: 15, eotNote: "", basis: "15 business days after the payment claim is given (s 76 BIF Act).", skipped: r.skipped };
    }
    if (scenario === "adjudicationApp") {
      if (data.aaScenario === "less") {
        const start = parseDate(data.scheduleReceived);
        if (!start) throw new Error("Enter the date the payment schedule was received.");
        const r = addBusinessDays(start, 30, location);
        return { title: "Adjudication application due date", finalDate: r.finalDate, startDate: start, days: 30, basis: "30 business days after the payment schedule was received (s 79(2)(b)(iii) BIF Act).", skipped: r.skipped };
      }
      if (data.aaScenario === "no-pay-amount") {
        const start = parseDate(data.paymentDue);
        if (!start) throw new Error("Enter the due date for the progress payment.");
        const r = addBusinessDays(start, 20, location);
        return { title: "Adjudication application due date", finalDate: r.finalDate, startDate: start, days: 20, basis: "20 business days after the due date for the progress payment (s 79(2)(b)(ii) BIF Act).", skipped: r.skipped };
      }
      const a = parseDate(data.paymentDue);
      const b = parseDate(data.scheduleDue);
      if (!a || !b) throw new Error("Enter both the payment due date and the last day to provide a schedule.");
      const start = new Date(Math.max(a.getTime(), b.getTime()));
      const r = addBusinessDays(start, 30, location);
      return { title: "Adjudication application due date", finalDate: r.finalDate, startDate: start, days: 30, basis: "30 business days after the LATER of the payment due date or schedule due date (s 79(2)(b)(i) BIF Act).", skipped: r.skipped };
    }
    if (scenario === "adjudicationResp") {
      const appReceived = parseDate(data.appReceived);
      const acceptanceReceived = parseDate(data.acceptanceReceived);
      if (!appReceived || !acceptanceReceived) throw new Error("Enter both the application-received and acceptance-received dates.");
      const claimType = data.claimType || "standard";
      const days1 = claimType === "standard" ? 10 : 15;
      const days2 = claimType === "standard" ? 7 : 12;
      const r1 = addBusinessDays(appReceived, days1, location);
      const r2 = addBusinessDays(acceptanceReceived, days2, location);
      const r1IsLater = r1.finalDate.getTime() >= r2.finalDate.getTime();
      const picked = r1IsLater ? r1 : r2;
      const startDate = r1IsLater ? appReceived : acceptanceReceived;
      const startDays = r1IsLater ? days1 : days2;
      let finalDate = picked.finalDate;
      let skipped = picked.skipped.slice();
      let eotNote = "";
      if (claimType === "complex" && data.eotEnabled === "on") {
        const eotDays = Math.min(15, Math.max(1, parseInt(data.eotDays || "1", 10) || 1));
        const ext = addBusinessDays(finalDate, eotDays, location);
        finalDate = ext.finalDate;
        skipped = skipped.concat(ext.skipped);
        eotNote = ` Plus ${eotDays} additional business day${eotDays > 1 ? "s" : ""} extension under s 83(3).`;
      }
      return { title: "Adjudication response due date", finalDate, startDate, days: startDays, basis: `Later of ${days1} business days after receiving the application, or ${days2} business days after receiving notice of acceptance (s 83 BIF Act).${eotNote}`, skipped };
    }
    if (scenario === "adjudicatorDecision") {
      const start = parseDate(data.responseGiven);
      if (!start) throw new Error("Enter the date the adjudication response was given.");
      const claimType = data.claimType || "standard";
      const days = claimType === "standard" ? 10 : 15;
      const base = addBusinessDays(start, days, location);
      let finalDate = base.finalDate;
      let skipped = base.skipped.slice();
      let eotNote = "";
      if (data.eotEnabled === "on") {
        const eotDays = Math.max(1, parseInt(data.eotDays || "1", 10) || 1);
        const dayType = data.eotDayType || "business";
        if (dayType === "business") {
          const ext = addBusinessDays(finalDate, eotDays, location);
          finalDate = ext.finalDate;
          skipped = skipped.concat(ext.skipped);
          eotNote = ` Plus ${eotDays} business day${eotDays > 1 ? "s" : ""} extension agreed under s 86.`;
        } else {
          const next = new Date(finalDate.getTime());
          next.setDate(next.getDate() + eotDays);
          finalDate = next;
          eotNote = ` Plus ${eotDays} calendar day${eotDays > 1 ? "s" : ""} extension agreed under s 86.`;
        }
      }
      return { title: "Adjudicator's decision due date", finalDate, startDate: start, days, basis: `${days} business days after the adjudication response was given (s 85 BIF Act).${eotNote}`, skipped };
    }
    throw new Error("Unknown scenario.");
  }

  function isBusinessDay(date, location) {
    const d = date.getDay();
    if (d === 0 || d === 6) return { isBiz: false, reason: "Weekend" };
    const m = date.getMonth(), day = date.getDate();
    if ((m === 11 && day >= 22 && day <= 24) || (m === 11 && day >= 27 && day <= 31) || (m === 0 && day >= 2 && day <= 10)) return { isBiz: false, reason: "Christmas shutdown (s 87)" };
    const dateString = date.toISOString().slice(0, 10);
    const list = (HOLIDAYS.qld || []).concat(HOLIDAYS[location] || []);
    const hit = list.find((h) => h.date === dateString);
    return hit ? { isBiz: false, reason: hit.name } : { isBiz: true, reason: "" };
  }

  function addBusinessDays(startDate, days, location) {
    const current = new Date(startDate.getTime());
    let added = 0;
    const skipped = [];
    current.setDate(current.getDate() + 1);
    while (added < days) {
      const check = isBusinessDay(current, location);
      if (check.isBiz) added++;
      else skipped.push({ date: new Date(current.getTime()), reason: check.reason });
      if (added < days) current.setDate(current.getDate() + 1);
    }
    return { finalDate: current, skipped };
  }

  function summariseSkipped(skipped) {
    if (!skipped.length) return "None";
    const groups = {};
    skipped.forEach((s) => { groups[s.reason] = (groups[s.reason] || 0) + 1; });
    return Object.entries(groups).map(([reason, count]) => `${count} × ${reason}`).join(", ");
  }

  function renderDateResult(result) {
    const copyText = `${result.title}: ${formatDate(result.finalDate)}\n${result.basis}`;
    return `<div class="calc-result">
      <span class="calc-result-tag">${escapeHtml(result.title)}</span>
      <strong>${escapeHtml(formatDate(result.finalDate))}</strong>
      <p>${escapeHtml(result.basis)}</p>
      <dl>
        <dt>Start date</dt><dd>${escapeHtml(formatDate(result.startDate))}</dd>
        <dt>Business-day period</dt><dd>${result.days}</dd>
        <dt>Non-business days skipped</dt><dd>${escapeHtml(summariseSkipped(result.skipped))}</dd>
      </dl>
      <button class="ghost-button compact" type="button" data-copy-text="${attr(copyText)}">${ICON.copy}<span>Copy</span></button>
    </div>`;
  }

  /* ---------- Tools: Interest calculator (mirrors live Sopal layout) ---------- */

  function InterestPage() {
    setTimeout(bindInterestCalculator, 0);
    const today = new Date().toISOString().slice(0, 10);
    return PageBody(`
      <div class="page-shell">
        <h1 class="page-title">Interest calculator</h1>
        <p class="page-sub">Calculate interest on overdue progress payments under BIF Act s 73.</p>
        <div class="interest-grid">
          <div class="card">
            <div class="card-head"><h3>Calculate interest</h3><p class="muted">Choose the rate type below.</p></div>
            <div class="card-body">
              <div class="tab-strip">
                <button class="tab-strip-btn active" type="button" data-rate-type="qbcc">QBCC s 67P rate</button>
                <button class="tab-strip-btn" type="button" data-rate-type="contractual">Contractual rate</button>
              </div>
              <form class="calc-form" data-interest-form>
                <input type="hidden" name="type" value="qbcc">
                <label class="span-2">Progress payment amount<div class="input-group"><span class="prefix">$</span><input class="text-input" type="number" name="principal" min="0" step="0.01" placeholder="0.00"></div></label>
                <label class="span-2 contractual-only" data-contractual hidden>Annual interest rate<div class="input-group"><input class="text-input" type="number" name="annualRate" min="0" step="0.01" value="10"><span class="suffix">%</span></div></label>
                <label>Due date<input class="text-input" type="date" name="startDate"></label>
                <label>Calculation date<input class="text-input" type="date" name="endDate" value="${today}"></label>
                <button class="dark-button span-2" type="submit">Calculate interest</button>
              </form>
            </div>
            <details class="provision">
              <summary>BIF Act s 73 — interest on overdue progress payments</summary>
              <div class="provision-body">
                <p>(1) A progress payment becomes payable — (a) on the day it becomes payable under the contract; or (b) 10 business days after the payment claim is made if the contract is silent.</p>
                <p>(2) Interest is payable at the greater of the contract rate or the rate prescribed under the <em>Civil Proceedings Act 2011</em> s 59(3) for a money order debt.</p>
                <p>(3) For a building contract under the <em>QBCC Act 1991</em> s 67P, interest is payable at the penalty rate under that section (10% + RBA cash rate).</p>
              </div>
            </details>
          </div>
          <div id="interest-result-card" class="card">${interestPlaceholder()}</div>
        </div>
      </div>
    `);
  }

  function interestPlaceholder() {
    return `<div class="card-body interest-placeholder">${EmptyState("No calculation yet.", "Enter the unpaid amount and dates, then calculate.")}</div>`;
  }

  function bindInterestCalculator() {
    const form = document.querySelector("[data-interest-form]");
    if (!form) return;
    const tabs = document.querySelectorAll("[data-rate-type]");
    const contractualField = document.querySelector("[data-contractual]");
    const typeInput = form.elements.type;
    tabs.forEach((tab) => tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      const t = tab.dataset.rateType;
      typeInput.value = t;
      contractualField.hidden = t !== "contractual";
      const card = document.getElementById("interest-result-card");
      if (card) card.innerHTML = interestPlaceholder();
    }));
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const card = document.getElementById("interest-result-card");
      card.innerHTML = `<div class="card-body"><div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Calculating…</span></div></div>`;
      try {
        const data = Object.fromEntries(new FormData(form).entries());
        const result = await computeInterest(data);
        card.innerHTML = renderInterestResult(result);
        card.querySelectorAll("[data-toggle-breakdown]").forEach((b) => b.addEventListener("click", () => {
          const body = document.getElementById("interest-breakdown-body");
          if (!body) return;
          body.hidden = !body.hidden;
          b.classList.toggle("open", !body.hidden);
        }));
      } catch (error) {
        card.innerHTML = `<div class="card-body"><div class="error-banner">${escapeHtml(error.message || "Interest calculation failed")}</div></div>`;
      }
    });
  }

  async function computeInterest(data) {
    const principal = Number(data.principal);
    const startDate = parseDate(data.startDate);
    const endDate = parseDate(data.endDate);
    if (!principal || principal <= 0) throw new Error("Enter the unpaid principal amount.");
    if (!startDate || !endDate) throw new Error("Enter both the due date and the calculation date.");
    if (endDate < startDate) throw new Error("The calculation date must be on or after the due date.");
    const days = Math.ceil((endDate.getTime() - startDate.getTime()) / 86400000) + 1;
    if (data.type === "contractual") {
      const annualRate = Number(data.annualRate);
      if (Number.isNaN(annualRate)) throw new Error("Enter the contractual annual rate.");
      const interest = principal * (annualRate / 100 / 365) * days;
      return { type: "Contractual", principal, days, interest, startDate, endDate, annualRate };
    }
    const sd = startDate.toISOString().slice(0, 10);
    const ed = endDate.toISOString().slice(0, 10);
    const response = await fetch(`/get_interest_rate?startDate=${sd}&endDate=${ed}`, { credentials: "include" });
    const ratesData = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(ratesData.detail || "Could not fetch the live RBA cash-rate series.");
    const dailyRates = (ratesData.dailyRates || []).map((r) => ({ date: r.date, rate: Number(r.rate) }));
    if (!dailyRates.length) throw new Error("No daily rates returned for that period.");
    let interest = 0;
    dailyRates.forEach((row) => { interest += (principal / 365) * ((10 + row.rate) / 100); });
    const rates = dailyRates.map((r) => r.rate);
    return { type: "QBCC s 67P", principal, days, interest, startDate, endDate, minRate: Math.min(...rates), maxRate: Math.max(...rates), dailyRates };
  }

  function renderInterestResult(result) {
    const total = result.principal + result.interest;
    const isQbcc = result.type === "QBCC s 67P";
    const rateRange = isQbcc
      ? (result.minRate.toFixed(2) === result.maxRate.toFixed(2)
        ? `${(10 + result.minRate).toFixed(2)}% (10% + ${result.minRate.toFixed(2)}% RBA)`
        : `${(10 + result.minRate).toFixed(2)}% – ${(10 + result.maxRate).toFixed(2)}%`)
      : `${result.annualRate.toFixed(2)}%`;
    const rateDetail = isQbcc
      ? (result.minRate.toFixed(2) === result.maxRate.toFixed(2) ? "Single RBA cash rate over the period." : "Variable daily rate (10% + RBA cash rate).")
      : "Contractual rate as entered.";

    const breakdown = (result.dailyRates && result.dailyRates.length)
      ? `<div class="breakdown">
          <button class="breakdown-toggle" type="button" data-toggle-breakdown>
            <span>Daily breakdown · ${result.dailyRates.length} day${result.dailyRates.length === 1 ? "" : "s"}</span>
            <span class="chev">${ICON.chevDown}</span>
          </button>
          <div class="breakdown-body" id="interest-breakdown-body" hidden>
            <table class="breakdown-table">
              <thead><tr><th>Date</th><th>RBA</th><th>Effective</th><th class="num">Daily interest</th></tr></thead>
              <tbody>
                ${result.dailyRates.slice(0, 365).map((row) => {
                  const eff = (10 + row.rate) / 100;
                  const day = (result.principal / 365) * eff;
                  return `<tr><td>${escapeHtml(row.date)}</td><td>${row.rate.toFixed(2)}%</td><td>${(eff * 100).toFixed(2)}%</td><td class="num">${formatCurrencyMicro(day)}</td></tr>`;
                }).join("")}
              </tbody>
            </table>
          </div>
        </div>`
      : "";

    const copy = `Calculation summary
Principal\t${formatCurrencyFull(result.principal)}
Rate type\t${result.type}
Rate\t${rateRange}
Due date\t${formatDate(result.startDate)}
Calculation date\t${formatDate(result.endDate)}
Days\t${result.days}
Interest\t${formatCurrencyFull(result.interest)}
Total\t${formatCurrencyFull(total)}`;

    return `<div class="card-head"><h3>Result</h3></div>
      <div class="card-body interest-result">
        <div class="result-headline">
          <span class="muted">Interest payable</span>
          <strong>${formatCurrencyFull(result.interest)}</strong>
          <span class="muted">on ${formatCurrencyFull(result.principal)} over ${result.days} day${result.days === 1 ? "" : "s"}</span>
        </div>
        <table class="result-table">
          <tbody>
            <tr><td>Principal</td><td>${formatCurrencyFull(result.principal)}</td></tr>
            <tr><td>Rate type</td><td>${escapeHtml(result.type)}</td></tr>
            <tr><td>Annual rate</td><td>${escapeHtml(rateRange)}</td></tr>
            <tr><td>Rate detail</td><td>${escapeHtml(rateDetail)}</td></tr>
            <tr><td>Due date</td><td>${escapeHtml(formatDate(result.startDate))}</td></tr>
            <tr><td>Calculation date</td><td>${escapeHtml(formatDate(result.endDate))}</td></tr>
            <tr><td>Days</td><td>${result.days}</td></tr>
            <tr><td>Interest</td><td>${formatCurrencyFull(result.interest)}</td></tr>
          </tbody>
        </table>
        <div class="result-total"><span>Total payable</span><strong>${formatCurrencyFull(total)}</strong></div>
        <button class="ghost-button compact" type="button" data-copy-text="${attr(copy)}">${ICON.copy}<span>Copy results</span></button>
        ${breakdown}
      </div>`;
  }

  /* ---------- Project workspace ---------- */

  function ProjectsListPage() {
    const projects = projectList();
    return PageBody(`
      <div class="page-shell">
        <div class="page-head">
          <div><h1 class="page-title">Your projects</h1><p class="page-sub">Each project is one construction contract — head contract or subcontract.</p></div>
          <button class="dark-button" type="button" data-new-project>${ICON.plus}<span>New project</span></button>
        </div>
        ${projects.length === 0 ? `
          <div class="card-empty">
            <div class="card-empty-icon">${ICON.file}</div>
            <h4>Create your first project</h4>
            <p>Give it a name, the parties, the contract form. Then upload or paste the contract — the assistant and every agent will work in that project's context.</p>
            <button class="dark-button" type="button" data-new-project>Create project</button>
          </div>
        ` : `<div class="project-list">${projects.map((p) => projectRow(p)).join("")}</div>`}
      </div>
    `);
  }

  function ProjectOverviewPage(projectId) {
    const project = getProject(projectId);
    if (!project) return notFoundPage();
    const recentChats = Object.entries(project.chats || {})
      .filter(([, c]) => Array.isArray(c.messages) && c.messages.length > 0)
      .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
      .slice(0, 5);
    setTimeout(() => bindProjectActions(projectId), 0);
    return PageBody(`
      <div class="page-shell">
        <div class="page-head">
          <div><h1 class="page-title">${escapeHtml(project.name)}</h1><p class="page-sub">${escapeHtml([project.reference, project.contractForm, [project.claimant, project.respondent].filter(Boolean).join(" v ")].filter(Boolean).join(" · ") || "Bespoke contract")}</p></div>
          <div class="page-actions">
            <button class="ghost-button compact" type="button" data-edit-project>Edit details</button>
            <button class="ghost-button compact danger" type="button" data-delete-project="${attr(project.id)}">${ICON.trash}<span>Delete</span></button>
          </div>
        </div>
        <div class="project-grid">
          <section class="card">
            <div class="card-head"><h3>Project details</h3></div>
            <div class="card-body">
              <dl class="kv">
                <dt>Project name</dt><dd>${escapeHtml(project.name)}</dd>
                <dt>Contract form</dt><dd>${escapeHtml(project.contractForm)}</dd>
                <dt>Reference</dt><dd>${escapeHtml(project.reference || "—")}</dd>
                <dt>Claimant</dt><dd>${escapeHtml(project.claimant || "—")}</dd>
                <dt>Respondent</dt><dd>${escapeHtml(project.respondent || "—")}</dd>
                <dt>You are</dt><dd>${escapeHtml(project.userIsParty === "respondent" ? "Respondent" : "Claimant")}</dd>
              </dl>
            </div>
          </section>
          <section class="card">
            <div class="card-head"><h3>Documents</h3></div>
            <div class="card-body">
              <div class="metric-grid compact">
                <div class="metric"><strong>${project.contracts.length}</strong><span>contract docs</span></div>
                <div class="metric"><strong>${project.library.length}</strong><span>library items</span></div>
                <div class="metric"><strong>${(project.contracts.reduce((s, d) => s + (d.text || "").length, 0) + project.library.reduce((s, d) => s + (d.text || "").length, 0)).toLocaleString()}</strong><span>chars indexed</span></div>
              </div>
              <div class="quick-link-row">
                <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/contract" data-nav>Open contract</a>
                <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/library" data-nav>Open library</a>
                <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/assistant" data-nav>Open assistant</a>
              </div>
            </div>
          </section>
          <section class="card span-all">
            <div class="card-head"><h3>Recent conversations</h3></div>
            <div class="card-body">
              ${recentChats.length === 0
                ? EmptyState("No conversations yet.", "Open the assistant or any agent to start a conversation in this project.")
                : `<div class="recent-list">${recentChats.map(([key, h]) => recentChatRow(project, key, h)).join("")}</div>`}
            </div>
          </section>
        </div>
      </div>
    `);
  }

  function plainPreview(text) {
    // Strip markdown for inline previews so users see prose, not "## Heading".
    return String(text || "")
      .replace(/^#{1,6}\s+/gm, "")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/^\s*[-*]\s+/gm, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function recentChatRow(project, key, h) {
    let label, href;
    if (key === "assistant") { label = "Assistant"; href = `/sopal-v2/projects/${project.id}/assistant`; }
    else if (key.startsWith("agent:")) {
      const [, agentKey, mode] = key.split(":");
      label = `${AGENT_LABELS[agentKey] || agentKey} · ${mode}`;
      href = `/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=${mode}`;
    } else { label = key; href = `/sopal-v2/projects/${project.id}/assistant`; }
    const last = h.messages[h.messages.length - 1] || {};
    const preview = plainPreview(last.content || "");
    return `<a class="recent-item" href="${href}" data-nav>
      <strong>${escapeHtml(label)}</strong>
      <span class="muted">${escapeHtml(preview.slice(0, 130))}${preview.length > 130 ? "…" : ""}</span>
    </a>`;
  }

  function bindProjectActions(projectId) {
    document.querySelector("[data-edit-project]")?.addEventListener("click", () => openProjectModal(projectId));
    document.querySelectorAll("[data-delete-project]").forEach((b) => b.addEventListener("click", () => {
      if (!confirm("Delete this project and all its conversations and context?")) return;
      deleteProject(b.dataset.deleteProject);
      navigate("/sopal-v2/projects");
    }));
  }

  function notFoundPage() {
    return PageBody(`<div class="page-shell">${EmptyState("Project not found.", "It may have been deleted. Return to the project list.", `<a class="ghost-button compact" href="/sopal-v2/projects" data-nav>Open projects</a>`)}</div>`);
  }

  function ContextPage(projectId, bucket) {
    const project = getProject(projectId);
    if (!project) return notFoundPage();
    const items = project[bucket] || [];
    const labels = bucket === "contracts" ? { single: "Contract", title: "Contract documents", helper: "Paste contract clauses or extract text from PDF/DOCX/TXT. The assistant and every agent in this project will see this content." } : { single: "Project document", title: "Project library", helper: "Paste correspondence, RFIs, claims, schedules, programme notes — or extract from PDF/DOCX/TXT." };
    setTimeout(() => bindContextManager(projectId, bucket), 0);
    return PageBody(`
      <div class="page-shell">
        <h1 class="page-title">${escapeHtml(labels.title)}</h1>
        <p class="page-sub">${escapeHtml(labels.helper)} Stored in this browser only.</p>
        <div class="context-grid">
          <div class="card">
            <div class="card-head"><h3>Add ${escapeHtml(labels.single.toLowerCase())}</h3></div>
            <form class="card-body context-form" data-context-form="${bucket}">
              <label class="span-2">Label<input class="text-input" name="name" placeholder="e.g. Head contract — clauses 1-12"></label>
              <label class="span-2">Paste text<textarea class="text-area" name="text" rows="8" placeholder="Paste clauses, correspondence, claim text, schedule text, or facts."></textarea></label>
              <div class="file-zone span-2">
                <label class="file-zone-label">${ICON.upload}<span>Click or drop to extract from PDF / DOCX / TXT</span><input type="file" data-context-file accept=".pdf,.docx,.txt"></label>
                <div class="muted file-status" data-context-file-status>No file selected.</div>
              </div>
              <button class="dark-button span-2" type="submit">${ICON.plus}<span>Add to project</span></button>
            </form>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Saved (${items.length})</h3></div>${items.length ? `<button class="ghost-button compact danger" type="button" data-clear-context="${bucket}">Clear all</button>` : ""}</div>
            <div class="card-body context-list">
              ${items.length === 0 ? EmptyState(`No ${labels.single.toLowerCase()} yet.`, "Add pasted or extracted text to make agents context-aware.") : items.map((item, i) => `
                <article class="context-item">
                  <div class="context-item-head">
                    <strong>${escapeHtml(item.name)}</strong>
                    <span class="muted">${item.text.length.toLocaleString()} chars · ${escapeHtml(item.source || "pasted")}</span>
                  </div>
                  <details><summary>Preview</summary><pre>${escapeHtml(item.text.slice(0, 2000))}${item.text.length > 2000 ? "\n…" : ""}</pre></details>
                  <div class="context-item-actions">
                    <button class="ghost-button compact" type="button" data-copy-text="${attr(item.text.slice(0, 8000))}">${ICON.copy}<span>Copy</span></button>
                    <button class="ghost-button compact danger" type="button" data-remove-context="${attr(bucket)}:${i}">Remove</button>
                  </div>
                </article>`).join("")}
            </div>
          </div>
        </div>
      </div>
    `);
  }

  function bindContextManager(projectId, bucket) {
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
      const project = getProject(projectId);
      if (!project) return;
      project[bucket].push({
        name: String(data.name || extracted?.filename || "Untitled"),
        text: String(data.text).trim(),
        source: extracted ? "extracted file + paste" : "pasted",
        addedAt: new Date().toISOString(),
      });
      saveProject(project);
      render();
    });
  }

  /* ---------- Project Assistant + Agents (Astruct-inspired chat) ---------- */

  function projectContextString(project) {
    const contractText = project.contracts.map((d) => `Contract: ${d.name}\n${d.text}`).join("\n\n---\n\n");
    const libraryText = project.library.map((d) => `Project document: ${d.name}\n${d.text}`).join("\n\n---\n\n");
    return [contractText, libraryText].filter(Boolean).join("\n\n===\n\n").slice(0, 40000);
  }

  function AssistantPage(projectId) {
    const project = getProject(projectId);
    if (!project) return notFoundPage();
    const opts = {
      project,
      chatKey: "assistant",
      endpoint: "/api/sopal-v2/chat",
      title: project.name,
      titleSub: "Project assistant",
      placeholder: `Ask anything about ${project.name}…`,
      starters: SCENARIO_STARTERS.assistant,
      contextDefaultOn: project.contracts.length + project.library.length > 0,
    };
    setTimeout(() => bindChatPanel(opts), 0);
    return PageBody(ChatPage(opts));
  }

  function AgentPage(projectId, agentKey) {
    const project = getProject(projectId);
    if (!project) return notFoundPage();
    if (!AGENT_KEYS.includes(agentKey)) return notFoundPage();

    const draftOnly = DRAFT_ONLY_AGENTS.has(agentKey);
    const params = new URLSearchParams(window.location.search);
    const requestedMode = params.get("mode");
    const mode = draftOnly ? "draft" : (requestedMode === "draft" ? "draft" : "review");

    if (mode === "draft") {
      const opts = {
        project,
        chatKey: `agent:${agentKey}:draft`,
        endpoint: "/api/sopal-v2/agent",
        agentKey,
        mode: "draft",
        title: AGENT_LABELS[agentKey],
        titleSub: AGENT_DESCRIPTIONS[agentKey],
        placeholder: "Describe what needs drafting and paste the relevant project / contract facts.",
        starters: SCENARIO_STARTERS.draft || [],
        contextDefaultOn: project.contracts.length + project.library.length > 0,
        includeList: INCLUDE_LISTS[agentKey] || [],
        draftOnly,
      };
      setTimeout(() => bindChatPanel(opts), 0);
      return PageBody(ChatPage(opts));
    }

    // Review = claim-check style workspace.
    const submodes = AGENT_REVIEW_MODES[agentKey] || [];
    const submodeId = params.get("submode");
    const activeSubmode = submodes.find((m) => m.id === submodeId) || null;
    setTimeout(() => bindReviewWorkspace(project, agentKey, activeSubmode), 0);
    return PageBody(ReviewWorkspace(project, agentKey, activeSubmode));
  }

  function ReviewWorkspace(project, agentKey, submode) {
    const submodes = AGENT_REVIEW_MODES[agentKey] || [];
    const reviewKey = submode ? `review:${agentKey}:${submode.id}` : null;
    const review = reviewKey && project.reviews ? project.reviews[reviewKey] : null;
    const document = review?.document || null;
    const analysis = review?.analysis || null;
    const chat = reviewKey ? projectChat(project, `chat:${reviewKey}`) : null;

    const draftHref = `/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft`;
    const modeBaseHref = `/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=review`;

    const head = `
      <div class="chat-page-head">
        <div>
          <h1 class="page-title">${escapeHtml(AGENT_LABELS[agentKey])}</h1>
          <p class="page-sub">${escapeHtml(AGENT_DESCRIPTIONS[agentKey] || "")}</p>
        </div>
        <div class="mode-tabs" role="tablist">
          <button class="mode-tab active" type="button">Review</button>
          <button class="mode-tab" type="button" data-go="${draftHref}">Draft</button>
        </div>
      </div>`;

    if (!submode) {
      return `
        <div class="page-shell review-shell">
          ${head}
          <section class="review-mode-picker">
            <h3>What are you reviewing?</h3>
            <p class="muted">Pick the perspective. The checks are tailored to it.</p>
            <div class="mode-tile-grid">
              ${submodes.map((m) => `
                <a class="mode-tile" href="${modeBaseHref}&submode=${m.id}" data-nav>
                  <span class="mode-tile-icon">${m.id === "received" || m.id === "received" ? ICON.upload : ICON.file}</span>
                  <span class="mode-tile-body">
                    <strong>${escapeHtml(AGENT_LABELS[agentKey])} I'm ${m.label.toLowerCase()}</strong>
                    <span class="muted">${escapeHtml(m.sub)}</span>
                  </span>
                </a>
              `).join("")}
            </div>
          </section>
        </div>`;
    }

    return `
      <div class="page-shell review-shell">
        ${head}
        <div class="review-meta-bar">
          <div class="review-meta-left">
            <span class="muted">Mode:</span>
            <strong>${escapeHtml(AGENT_LABELS[agentKey])} — ${escapeHtml(submode.label.toLowerCase())}</strong>
            <a class="link-button small" href="${modeBaseHref}" data-nav>Change</a>
          </div>
        </div>
        <div class="review-grid">
          <div class="review-left">
            <section class="card review-doc-card">
              <div class="card-head">
                <h3>Document</h3>
                <button class="link-button small" type="button" data-toggle-paste>${document ? "Replace" : "Paste text instead"}</button>
              </div>
              <div class="card-body" data-doc-body>
                ${renderDocumentInput(document)}
              </div>
            </section>
            <section class="card review-chat-card">
              <div class="card-head">
                <h3>Ask about this document</h3>
                <span class="muted">${reviewKey && analysis ? "Use the analysis on the right as you ask" : "Run an analysis first to ground the chat"}</span>
              </div>
              <div class="review-chat" data-chat-pane>
                ${ChatPane(project, agentKey, submode, chat, !!document)}
              </div>
            </section>
          </div>
          <aside class="review-right">
            <section class="card review-analysis-card">
              <div class="card-head">
                <h3>Analysis</h3>
                ${analysis ? `<button class="ghost-button compact" type="button" data-rerun-analysis>Re-run</button>` : ""}
              </div>
              <div class="card-body" data-analysis-body>
                ${renderAnalysis(agentKey, document, analysis, review?.status)}
              </div>
            </section>
          </aside>
        </div>
      </div>
    `;
  }

  function renderDocumentInput(document) {
    if (document) {
      const preview = (document.text || "").slice(0, 1200);
      return `
        <div class="doc-loaded">
          <div class="doc-loaded-head">
            <strong>${escapeHtml(document.name || "Document")}</strong>
            <span class="muted">${(document.text || "").length.toLocaleString()} characters · ${escapeHtml(document.source || "pasted")}</span>
          </div>
          <pre class="doc-preview">${escapeHtml(preview)}${(document.text || "").length > preview.length ? "\n…" : ""}</pre>
        </div>`;
    }
    return `
      <div class="doc-input">
        <label class="upload-zone" data-upload-zone>
          <input type="file" accept=".pdf,.docx,.txt" data-doc-file hidden>
          <span class="upload-icon">${ICON.upload}</span>
          <span class="upload-primary">Drop a PDF / DOCX / TXT here, or <span class="upload-browse">browse</span></span>
          <span class="upload-sub muted">Held in memory only. Nothing is stored on the server.</span>
        </label>
        <details class="paste-fallback" data-paste-fallback>
          <summary>Paste text instead</summary>
          <textarea class="text-area" data-paste-input placeholder="Paste the document text here…" rows="6"></textarea>
          <div class="paste-actions">
            <span class="muted" data-paste-meta>0 characters</span>
            <button class="dark-button" type="button" data-paste-submit disabled>Use this text</button>
          </div>
        </details>
      </div>`;
  }

  function renderAnalysis(agentKey, document, analysis, status) {
    if (!document) {
      return EmptyState("Add a document to start.", "Upload or paste the document, then run the structured BIF Act / SOPA review.");
    }
    if (status === "running") {
      const items = REVIEW_CHECKS[agentKey] || [];
      return `
        <div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Running ${items.length} structured checks…</span></div>
        <ol class="check-list pending">${items.map((t) => `<li class="check-item pending"><span class="check-status">…</span><div class="check-body"><strong>${escapeHtml(t)}</strong></div></li>`).join("")}</ol>`;
    }
    if (!analysis) {
      const items = REVIEW_CHECKS[agentKey] || [];
      return `
        <div class="analysis-action">
          <button class="dark-button" type="button" data-run-analysis>${ICON.sparkles}<span>Run analysis</span></button>
          <span class="muted">Will check ${items.length} items.</span>
        </div>
        <ol class="check-list pending">${items.map((t) => `<li class="check-item idle"><span class="check-status">○</span><div class="check-body"><strong>${escapeHtml(t)}</strong></div></li>`).join("")}</ol>`;
    }
    if (analysis.error) {
      return `<div class="error-banner">${escapeHtml(analysis.error)}</div>
        <button class="ghost-button compact" type="button" data-run-analysis>Try again</button>`;
    }
    const counts = analysis.counts || { fail: 0, warn: 0, info: 0, pass: 0 };
    return `
      <div class="analysis-summary">
        <div class="summary-counts">
          <span class="sc-pill sc-fail"><strong>${counts.fail}</strong> issues</span>
          <span class="sc-pill sc-warn"><strong>${counts.warn}</strong> warnings</span>
          <span class="sc-pill sc-info"><strong>${counts.info}</strong> need input</span>
          <span class="sc-pill sc-pass"><strong>${counts.pass}</strong> passed</span>
        </div>
        ${analysis.summary ? `<div class="analysis-overview">${renderMarkdown(analysis.summary)}</div>` : ""}
      </div>
      <ol class="check-list">
        ${(analysis.checks || []).map((c, i) => `
          <li class="check-item ${c.status || "info"}">
            <span class="check-status">${checkIcon(c.status)}</span>
            <div class="check-body">
              <strong>${escapeHtml(c.title || `Check ${i + 1}`)}</strong>
              <div class="check-detail">${renderMarkdown(c.detail || "")}</div>
            </div>
          </li>
        `).join("")}
      </ol>
      ${(analysis.recommendations || []).length ? `
        <section class="analysis-block">
          <h4>Recommendations</h4>
          <ul>${analysis.recommendations.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
        </section>` : ""}
      ${(analysis.missing || []).length ? `
        <section class="analysis-block">
          <h4>Missing information to confirm</h4>
          <ul>${analysis.missing.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
        </section>` : ""}
    `;
  }

  function checkIcon(status) {
    if (status === "pass") return "✓";
    if (status === "fail") return "✕";
    if (status === "warn") return "!";
    if (status === "info") return "?";
    return "•";
  }

  function ChatPane(project, agentKey, submode, chat, hasDocument) {
    const reviewKey = `review:${agentKey}:${submode.id}`;
    const ctxCount = project.contracts.length + project.library.length;
    const messagesHtml = (chat.messages || []).length
      ? (chat.messages || []).map((m) => renderMessage(m.role, m.content, m.role === "assistant")).join("")
      : EmptyState("No questions yet.", hasDocument
          ? "Run the analysis or ask a question about the document."
          : "Add a document to give the chat something to anchor to.");
    return `
      <div class="message-area review-message-area" data-message-area>
        <div class="message-stack" data-messages>${messagesHtml}</div>
      </div>
      <form class="composer-active review-composer" data-chat-form data-review-key="${attr(reviewKey)}">
        <div class="composer-row">
          <textarea class="text-area auto-grow" name="message" rows="1" placeholder="${hasDocument ? "Ask about this document…" : "Add a document above to start the chat."}" ${hasDocument ? "" : "disabled"}></textarea>
          <button class="send-button" type="submit" aria-label="Send" ${hasDocument ? "" : "disabled"}>${ICON.send}</button>
        </div>
        <div class="composer-meta">
          <label class="check"><input type="checkbox" name="useContext" ${ctxCount ? "checked" : "disabled"}><span>Project context (${ctxCount})</span></label>
          <span class="muted kbd-hint">⌘ / Ctrl + Enter to send</span>
        </div>
      </form>`;
  }

  function bindReviewWorkspace(project, agentKey, submode) {
    if (!submode) return;
    const docBody = document.querySelector("[data-doc-body]");
    const analysisBody = document.querySelector("[data-analysis-body]");
    const reviewKey = `review:${agentKey}:${submode.id}`;
    const ensureReview = () => {
      if (!project.reviews) project.reviews = {};
      if (!project.reviews[reviewKey]) project.reviews[reviewKey] = { document: null, analysis: null, status: "idle" };
      return project.reviews[reviewKey];
    };

    function refreshDoc() {
      const r = ensureReview();
      docBody.innerHTML = renderDocumentInput(r.document);
      bindDocInput();
    }

    function refreshAnalysis() {
      const r = ensureReview();
      analysisBody.innerHTML = renderAnalysis(agentKey, r.document, r.analysis, r.status);
      bindAnalysisActions();
    }

    function refreshChat() {
      const pane = document.querySelector("[data-chat-pane]");
      if (!pane) return;
      const r = ensureReview();
      const chat = projectChat(project, `chat:${reviewKey}`);
      pane.innerHTML = ChatPane(project, agentKey, submode, chat, !!r.document);
      bindReviewChatForm(pane);
    }

    function bindDocInput() {
      const fileInput = docBody.querySelector("[data-doc-file]");
      const dropzone = docBody.querySelector("[data-upload-zone]");
      const pasteSection = docBody.querySelector("[data-paste-fallback]");
      const pasteText = docBody.querySelector("[data-paste-input]");
      const pasteSubmit = docBody.querySelector("[data-paste-submit]");
      const pasteMeta = docBody.querySelector("[data-paste-meta]");

      fileInput?.addEventListener("change", async () => {
        const file = fileInput.files && fileInput.files[0];
        if (!file) return;
        await ingestFile(file);
      });
      dropzone?.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
      dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
      dropzone?.addEventListener("drop", async (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        const file = e.dataTransfer?.files?.[0];
        if (file) await ingestFile(file);
      });

      pasteText?.addEventListener("input", () => {
        const len = (pasteText.value || "").length;
        pasteMeta.textContent = `${len.toLocaleString()} characters`;
        pasteSubmit.disabled = len < 30;
      });
      pasteSubmit?.addEventListener("click", () => {
        const text = (pasteText.value || "").trim();
        if (!text) return;
        const r = ensureReview();
        r.document = { name: "Pasted text", text, source: "pasted", addedAt: new Date().toISOString() };
        r.analysis = null;
        r.status = "idle";
        saveProject(project);
        refreshDoc();
        refreshAnalysis();
        refreshChat();
      });

      const replace = document.querySelector("[data-toggle-paste]");
      if (replace) {
        replace.onclick = () => {
          const r = ensureReview();
          if (r.document) {
            r.document = null;
            r.analysis = null;
            r.status = "idle";
            saveProject(project);
            refreshDoc();
            refreshAnalysis();
            refreshChat();
          } else {
            pasteSection?.setAttribute("open", "");
            pasteText?.focus();
          }
        };
      }
    }

    async function ingestFile(file) {
      const dropzone = docBody.querySelector("[data-upload-zone]");
      if (dropzone) dropzone.querySelector(".upload-primary").textContent = `Extracting ${file.name}…`;
      const fd = new FormData(); fd.append("file", file);
      try {
        const response = await fetch("/api/sopal-v2/extract", { method: "POST", body: fd, credentials: "include" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "Extraction failed");
        const r = ensureReview();
        r.document = { name: data.filename, text: data.text, source: "extracted", addedAt: new Date().toISOString() };
        r.analysis = null;
        r.status = "idle";
        saveProject(project);
        refreshDoc();
        refreshAnalysis();
        refreshChat();
      } catch (error) {
        const dz = docBody.querySelector("[data-upload-zone] .upload-primary");
        if (dz) dz.textContent = error.message || "Extraction failed";
      }
    }

    function bindAnalysisActions() {
      const runBtn = analysisBody.querySelector("[data-run-analysis]");
      if (runBtn) runBtn.addEventListener("click", runAnalysis);
      const rerun = document.querySelector("[data-rerun-analysis]");
      if (rerun) rerun.addEventListener("click", () => {
        const r = ensureReview(); r.analysis = null; r.status = "idle"; saveProject(project); refreshAnalysis();
      });
    }

    async function runAnalysis() {
      const r = ensureReview();
      if (!r.document) return;
      r.status = "running";
      saveProject(project);
      refreshAnalysis();

      const checks = REVIEW_CHECKS[agentKey] || [];
      const projectMeta = `Project: ${project.name}\nContract form: ${project.contractForm}${project.reference ? `\nReference: ${project.reference}` : ""}\nUser is: ${project.userIsParty || "claimant"}`;
      const ctxText = projectContextString(project);

      try {
        const response = await fetch("/api/sopal-v2/agent", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            agentType: agentKey,
            mode: "review",
            reviewSubmode: submode.id,
            checks,
            structured: true,
            message: `Review the document below.\n\nDocument:\n${r.document.text.slice(0, 60000)}`,
            projectContext: [projectMeta, ctxText].filter(Boolean).join("\n\n---\n\n"),
            files: [{ name: r.document.name, characters: r.document.text.length }],
          }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || data.error || "Analysis failed");
        const parsed = parseStructuredAnalysis(data.answer || "", checks);
        r.analysis = parsed;
        r.status = "done";
        saveProject(project);
        refreshAnalysis();
      } catch (error) {
        r.status = "error";
        r.analysis = { error: error.message || "Analysis failed" };
        saveProject(project);
        refreshAnalysis();
      }
    }

    function bindReviewChatForm(pane) {
      const form = pane.querySelector("[data-chat-form]");
      if (!form) return;
      const textarea = form.elements.message;
      const messages = pane.querySelector("[data-messages]");
      const messageArea = pane.querySelector("[data-message-area]");
      autoGrow(textarea);
      textarea.addEventListener("input", () => autoGrow(textarea));
      textarea.addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
          event.preventDefault();
          form.requestSubmit();
        }
      });
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const message = textarea.value.trim();
        if (!message) return;
        const r = ensureReview();
        if (!r.document) return;
        const chat = projectChat(project, `chat:${reviewKey}`);
        if (messages.querySelector(".empty-state")) messages.innerHTML = "";
        messages.insertAdjacentHTML("beforeend", renderMessage("user", message));
        const placeholderId = `msg-${Math.random().toString(36).slice(2)}`;
        messages.insertAdjacentHTML("beforeend", `
          <div class="message msg-assistant" id="${placeholderId}">
            <div class="message-body"><div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is working…</span></div></div>
          </div>`);
        chat.messages.push({ role: "user", content: message, at: Date.now() });
        chat.updatedAt = Date.now();
        textarea.value = "";
        autoGrow(textarea);
        scrollToBottom(messageArea);

        try {
          const useContext = form.elements.useContext?.checked;
          const projectMeta = `Project: ${project.name}\nContract form: ${project.contractForm}${project.reference ? `\nReference: ${project.reference}` : ""}`;
          const ctxText = useContext ? projectContextString(project) : "";
          const docBlock = `Document under review (${AGENT_LABELS[agentKey]} — ${submode.label.toLowerCase()}):\n${r.document.text.slice(0, 60000)}`;
          const response = await fetch("/api/sopal-v2/agent", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: "include",
            body: JSON.stringify({
              agentType: agentKey,
              mode: "review",
              reviewSubmode: submode.id,
              chatFollowup: true,
              message,
              projectContext: [projectMeta, ctxText, docBlock].filter(Boolean).join("\n\n---\n\n"),
              files: [{ name: r.document.name, characters: r.document.text.length }],
            }),
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || data.error || "Reply failed");
          chat.messages.push({ role: "assistant", content: data.answer || "", at: Date.now() });
          chat.updatedAt = Date.now();
          saveProject(project);
          const placeholder = document.getElementById(placeholderId);
          if (placeholder) placeholder.outerHTML = renderMessage("assistant", data.answer || "", true);
          scrollToBottom(messageArea);
        } catch (error) {
          const placeholder = document.getElementById(placeholderId);
          if (placeholder) placeholder.outerHTML = `<div class="message msg-assistant"><div class="message-body"><div class="error-banner">${escapeHtml(error.message || "Reply failed")}</div></div></div>`;
          scrollToBottom(messageArea);
        }
      });
    }

    bindDocInput();
    bindAnalysisActions();
    const initialPane = document.querySelector("[data-chat-pane]");
    if (initialPane) bindReviewChatForm(initialPane);
  }

  function parseStructuredAnalysis(text, checkTitles) {
    // Try JSON first.
    const trimmed = (text || "").trim();
    try {
      const start = trimmed.indexOf("{");
      const end = trimmed.lastIndexOf("}");
      if (start !== -1 && end > start) {
        const json = JSON.parse(trimmed.slice(start, end + 1));
        if (Array.isArray(json.checks)) {
          const counts = { pass: 0, fail: 0, warn: 0, info: 0 };
          json.checks.forEach((c) => {
            const s = (c.status || "info").toLowerCase();
            counts[s] = (counts[s] || 0) + 1;
          });
          return {
            summary: json.summary || "",
            checks: json.checks,
            counts,
            recommendations: Array.isArray(json.recommendations) ? json.recommendations : [],
            missing: Array.isArray(json.missing) ? json.missing : [],
          };
        }
      }
    } catch {
      // fall through to markdown parse
    }
    // Fallback: AI didn't return strict JSON. Surface that explicitly so the
    // user understands the structured pills aren't real, and show the raw
    // analysis the model produced so it isn't lost.
    const checks = checkTitles.map((title) => ({
      title,
      status: "info",
      detail: "Couldn't parse a per-check status from the model output. See the raw analysis above.",
    }));
    return {
      summary: "_The model didn't return strict structured JSON — showing its raw response below. Try Re-run if you want a structured analysis._\n\n" + (text || ""),
      checks,
      counts: { pass: 0, fail: 0, warn: 0, info: checks.length },
      recommendations: [],
      missing: [],
      _fallback: true,
    };
  }

  function ChatPage(opts) {
    const { project, agentKey, mode, includeList, starters, draftOnly } = opts;
    const chat = projectChat(project, opts.chatKey);
    const isEmpty = !chat.messages.length;
    const modeTabs = (agentKey && !draftOnly) ? `
      <div class="mode-tabs" role="tablist" aria-label="Agent mode">
        <button class="mode-tab ${mode === "review" ? "active" : ""}" type="button" data-go="/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=review">Review</button>
        <button class="mode-tab ${mode === "draft" ? "active" : ""}" type="button" data-go="/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft">Draft</button>
      </div>` : "";
    const helperPanel = agentKey ? `
      <aside class="helper-panel">
        <details class="helper-card" ${isEmpty ? "open" : ""}>
          <summary>What to include</summary>
          <ul>${(includeList || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
        </details>
        <details class="helper-card" ${isEmpty ? "open" : ""}>
          <summary>Project context · ${project.contracts.length}/${project.library.length}</summary>
          <p class="muted">${project.contracts.length} contract doc${project.contracts.length === 1 ? "" : "s"} · ${project.library.length} library item${project.library.length === 1 ? "" : "s"}</p>
          <div class="helper-actions">
            <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/contract" data-nav>Manage contract</a>
            <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/library" data-nav>Manage library</a>
          </div>
        </details>
      </aside>` : "";

    const header = `
      <div class="chat-page-head">
        <div>
          <h1 class="page-title">${escapeHtml(opts.title)}</h1>
          <p class="page-sub">${escapeHtml(opts.titleSub)}</p>
        </div>
        ${modeTabs}
      </div>`;

    return `
      <div class="page-shell chat-shell">
        ${header}
        <div class="chat-layout ${agentKey ? "with-helper" : ""}">
          <section class="chat-pane" data-chat-pane>
            ${isEmpty ? renderEmptyComposer(opts) : renderActiveChat(opts, chat)}
          </section>
          ${helperPanel}
        </div>
      </div>
    `;
  }

  function renderEmptyComposer(opts) {
    const { project, starters, placeholder, contextDefaultOn } = opts;
    return `
      <div class="chat-empty">
        <h2 class="chat-empty-title">${escapeHtml(opts.agentKey ? AGENT_LABELS[opts.agentKey] : "Sopal")}</h2>
        <p class="chat-empty-sub">${escapeHtml(opts.titleSub)}</p>
        ${composerCard(opts, /* compact */ false, contextDefaultOn)}
        <div class="starter-chips">
          ${(starters || []).map((s) => `<button class="starter-chip" type="button" data-starter="${attr(s)}">${escapeHtml(s)}</button>`).join("")}
        </div>
      </div>
    `;
  }

  function renderActiveChat(opts, chat) {
    return `
      <div class="chat-stream-wrap">
        <div class="chat-stream" data-message-area>
          <div class="message-stack" data-messages>
            ${chat.messages.map((m) => renderMessage(m.role, m.content, m.role === "assistant")).join("")}
          </div>
        </div>
      </div>
      ${composerCard(opts, /* compact */ true, opts.contextDefaultOn)}
    `;
  }

  function composerCard(opts, compact, contextOn) {
    const { project } = opts;
    const ctxCount = project.contracts.length + project.library.length;
    const baseCls = compact ? "composer-active" : "composer-card";
    // Compact mode (mid-conversation) gets a short "Reply…" placeholder so it
    // doesn't echo the verbose empty-state copy.
    const placeholder = compact ? "Reply…" : (opts.placeholder || "Type a message…");
    return `
      <form class="${baseCls}" data-chat-form>
        <div class="composer-row">
          <button class="icon-button" type="button" data-attach-trigger title="Attach file" aria-label="Attach file">${ICON.paperclip}</button>
          <textarea class="text-area auto-grow" name="message" rows="${compact ? 1 : 3}" placeholder="${attr(placeholder)}" aria-label="Message"></textarea>
          <button class="send-button" type="submit" aria-label="Send message">${ICON.send}</button>
          <input type="file" hidden data-chat-file accept=".pdf,.docx,.txt">
        </div>
        <div class="composer-meta">
          <label class="check"><input type="checkbox" name="useContext" ${ctxCount ? (contextOn ? "checked" : "") : "disabled"}><span>Project context (${ctxCount})</span></label>
          <span class="muted" data-chat-file-status></span>
          <span class="muted kbd-hint">⌘ / Ctrl + Enter to send</span>
        </div>
      </form>`;
  }

  function bindChatPanel(opts) {
    const pane = document.querySelector("[data-chat-pane]");
    if (!pane) return;
    bindChatForm(pane, opts);
  }

  function bindChatForm(pane, opts) {
    const form = pane.querySelector("[data-chat-form]");
    if (!form) return;
    const textarea = form.elements.message;
    const fileInput = form.querySelector("[data-chat-file]");
    const fileStatus = form.querySelector("[data-chat-file-status]");
    const messages = pane.querySelector("[data-messages]");
    const messageArea = pane.querySelector("[data-message-area]");

    let extractedFile = null;

    autoGrow(textarea);
    textarea.addEventListener("input", () => autoGrow(textarea));
    textarea.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    pane.querySelectorAll("[data-starter]").forEach((b) => b.addEventListener("click", () => {
      textarea.value = b.dataset.starter;
      autoGrow(textarea);
      textarea.focus();
    }));

    pane.querySelector("[data-attach-trigger]")?.addEventListener("click", () => fileInput?.click());
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

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = textarea.value.trim();
      if (!message) return;

      const project = getProject(opts.project.id);
      if (!project) return;
      const chat = projectChat(project, opts.chatKey);
      const wasEmpty = chat.messages.length === 0;

      chat.messages.push({ role: "user", content: message, at: Date.now() });
      chat.updatedAt = Date.now();
      saveProject(project);

      // If the chat was empty we need to morph the empty composer into the active layout.
      if (wasEmpty) {
        pane.innerHTML = renderActiveChat({ ...opts, contextDefaultOn: form.elements.useContext.checked }, chat);
        bindChatForm(pane, opts);
        return continueGeneration(pane, opts, message, form.elements.useContext.checked, extractedFile);
      }

      // Active layout already in DOM — append messages directly.
      const placeholderId = `msg-${Math.random().toString(36).slice(2)}`;
      messages.insertAdjacentHTML("beforeend", renderMessage("user", message));
      messages.insertAdjacentHTML("beforeend", `
        <div class="message msg-assistant" id="${placeholderId}">
          <div class="message-body"><div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is working…</span></div></div>
        </div>`);
      textarea.value = "";
      autoGrow(textarea);
      scrollToBottom(messageArea);

      try {
        const data = await callAi(opts, message, form.elements.useContext.checked, extractedFile, project);
        chat.messages.push({ role: "assistant", content: data.answer || "", at: Date.now() });
        chat.updatedAt = Date.now();
        saveProject(project);
        const placeholder = document.getElementById(placeholderId);
        if (placeholder) placeholder.outerHTML = renderMessage("assistant", data.answer || "", true);
        scrollToBottom(messageArea);
      } catch (error) {
        const placeholder = document.getElementById(placeholderId);
        const msg = error.message || "AI request failed";
        if (placeholder) placeholder.outerHTML = `<div class="message msg-assistant"><div class="message-body"><div class="error-banner">${escapeHtml(msg)}</div></div></div>`;
        scrollToBottom(messageArea);
      }
    });
  }

  async function continueGeneration(pane, opts, message, useContext, extractedFile) {
    const messages = pane.querySelector("[data-messages]");
    const messageArea = pane.querySelector("[data-message-area]");
    const placeholderId = `msg-${Math.random().toString(36).slice(2)}`;
    if (messages) {
      messages.insertAdjacentHTML("beforeend", `
        <div class="message msg-assistant" id="${placeholderId}">
          <div class="message-body"><div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is working…</span></div></div>
        </div>`);
      scrollToBottom(messageArea);
    }
    try {
      const project = getProject(opts.project.id);
      const data = await callAi(opts, message, useContext, extractedFile, project);
      const chat = projectChat(project, opts.chatKey);
      chat.messages.push({ role: "assistant", content: data.answer || "", at: Date.now() });
      chat.updatedAt = Date.now();
      saveProject(project);
      const placeholder = document.getElementById(placeholderId);
      if (placeholder) placeholder.outerHTML = renderMessage("assistant", data.answer || "", true);
      scrollToBottom(messageArea);
    } catch (error) {
      const placeholder = document.getElementById(placeholderId);
      const msg = error.message || "AI request failed";
      if (placeholder) placeholder.outerHTML = `<div class="message msg-assistant"><div class="message-body"><div class="error-banner">${escapeHtml(msg)}</div></div></div>`;
      scrollToBottom(messageArea);
    }
  }

  async function callAi(opts, message, useContext, extractedFile, project) {
    const projectContext = useContext && project ? projectContextString(project) : "";
    const projectMeta = project ? `Project: ${project.name}\nContract form: ${project.contractForm}${project.reference ? `\nReference: ${project.reference}` : ""}${project.claimant || project.respondent ? `\nParties: ${project.claimant || "(claimant)"} v ${project.respondent || "(respondent)"}` : ""}\nUser is: ${project.userIsParty || "claimant"}` : "";
    const fullContext = [projectMeta, projectContext].filter(Boolean).join("\n\n---\n\n");
    const response = await fetch(opts.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        agentType: opts.agentKey || null,
        mode: opts.mode || null,
        message,
        projectContext: fullContext,
        files: extractedFile ? [{ name: extractedFile.filename, characters: extractedFile.characters }] : [],
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || data.error || "AI request failed");
    return data;
  }

  function renderMessage(role, content, withActions) {
    if (role === "user") {
      const safe = escapeHtml(content || "").replace(/\n/g, "<br>");
      return `<div class="message msg-user"><div class="bubble">${safe}</div></div>`;
    }
    return `<div class="message msg-assistant">
      <div class="message-body">
        <div class="md">${renderMarkdown(content || "")}</div>
        ${withActions ? `<div class="message-actions"><button class="ghost-button compact" type="button" data-copy-text="${attr(content || "")}">${ICON.copy}<span>Copy</span></button></div>` : ""}
      </div>
    </div>`;
  }

  function autoGrow(textarea) {
    if (!textarea) return;
    textarea.style.height = "auto";
    const max = 220;
    textarea.style.height = `${Math.min(max, textarea.scrollHeight)}px`;
  }
  function scrollToBottom(area) {
    if (!area) return;
    requestAnimationFrame(() => { area.scrollTop = area.scrollHeight; });
  }

  /* ---------- New / Edit project modal ---------- */

  function openProjectModal(editId) {
    const editing = editId ? getProject(editId) : null;
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>${editing ? "Edit project" : "New project"}</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <form class="modal-body" data-project-form>
              <label>Project name<input class="text-input" name="name" required value="${attr(editing?.name || "")}" placeholder="e.g. Queen Street Tower"></label>
              <div class="row-2">
                <label>Claimant<input class="text-input" name="claimant" value="${attr(editing?.claimant || "")}" placeholder="e.g. Acme Builders Pty Ltd"></label>
                <label>Respondent<input class="text-input" name="respondent" value="${attr(editing?.respondent || "")}" placeholder="e.g. ABC Developments Pty Ltd"></label>
              </div>
              <div class="row-2">
                <label>Contract form
                  <select class="select-input" name="contractForm">
                    ${CONTRACT_FORMS.map((f) => `<option value="${attr(f)}" ${editing && editing.contractForm === f ? "selected" : ""}>${escapeHtml(f)}</option>`).join("")}
                  </select>
                </label>
                <label>Reference / contract no.<input class="text-input" name="reference" value="${attr(editing?.reference || "")}" placeholder="e.g. PO-2024-014"></label>
              </div>
              <div class="row-2">
                <label class="span-all">You act for
                  <div class="radio-group">
                    <label class="radio-option"><input type="radio" name="userIsParty" value="claimant" ${(!editing || editing.userIsParty === "claimant") ? "checked" : ""}>The claimant</label>
                    <label class="radio-option"><input type="radio" name="userIsParty" value="respondent" ${editing && editing.userIsParty === "respondent" ? "checked" : ""}>The respondent</label>
                  </div>
                </label>
              </div>
              <div class="modal-actions">
                <button class="ghost-button" type="button" data-modal-close>Cancel</button>
                <button class="dark-button" type="submit">${editing ? "Save changes" : "Create project"}</button>
              </div>
            </form>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        rootEl.querySelector("[data-project-form]")?.addEventListener("submit", (event) => {
          event.preventDefault();
          const data = Object.fromEntries(new FormData(event.currentTarget).entries());
          if (editing) {
            const project = getProject(editing.id);
            Object.assign(project, {
              name: (data.name || "").trim() || project.name,
              claimant: (data.claimant || "").trim(),
              respondent: (data.respondent || "").trim(),
              contractForm: data.contractForm,
              reference: (data.reference || "").trim(),
              userIsParty: data.userIsParty || "claimant",
            });
            saveProject(project);
            modal = null;
            render();
          } else {
            const project = createProject(data);
            modal = null;
            navigate(`/sopal-v2/projects/${project.id}/overview`);
          }
        });
        document.addEventListener("keydown", function handler(ev) {
          if (ev.key === "Escape") { document.removeEventListener("keydown", handler); close(); }
        });
      },
    };
    render();
  }

  /* ---------- Page resolver ---------- */

  function pageForRoute(route) {
    if (route === "home") return { crumbs: [], body: HomePage() };
    const parts = route.split("/").filter(Boolean);

    if (parts[0] === "research") {
      if (parts[1] === "decisions") {
        if (parts[2]) {
          const decisionId = parts[2];
          return {
            crumbs: [{ label: "Decision search", href: "/sopal-v2/research/decisions" }, { label: decisionId }],
            body: DecisionDetailPage(decisionId),
          };
        }
        return { crumbs: [{ label: "Decision search" }], body: DecisionsPage() };
      }
      if (parts[1] === "adjudicators") {
        if (parts[2]) {
          const name = decodeURIComponent(parts[2]);
          return {
            crumbs: [{ label: "Adjudicator statistics", href: "/sopal-v2/research/adjudicators" }, { label: name }],
            body: AdjudicatorDetailPage(name),
          };
        }
        return { crumbs: [{ label: "Adjudicator statistics" }], body: AdjudicatorsPage() };
      }
      return { crumbs: [{ label: "Research" }], body: notFoundPage() };
    }
    if (parts[0] === "tools") {
      if (parts[1] === "due-date-calculator") return { crumbs: [{ label: "Due date calculator" }], body: DueDatePage() };
      if (parts[1] === "interest-calculator") return { crumbs: [{ label: "Interest calculator" }], body: InterestPage() };
      return { crumbs: [{ label: "Tools" }], body: notFoundPage() };
    }
    if (parts[0] === "projects") {
      if (!parts[1]) return { crumbs: [{ label: "Your projects" }], body: ProjectsListPage() };
      const projectId = parts[1];
      const project = getProject(projectId);
      const projectCrumb = project ? { label: project.name, href: `/sopal-v2/projects/${projectId}/overview` } : { label: "Project" };
      const head = [{ label: "Projects", href: "/sopal-v2/projects" }, projectCrumb];
      if (project) selectProject(projectId);
      const sub = parts[2] || "overview";
      if (sub === "overview") return { crumbs: head.concat([{ label: "Overview" }]), body: ProjectOverviewPage(projectId) };
      if (sub === "contract") return { crumbs: head.concat([{ label: "Contract" }]), body: ContextPage(projectId, "contracts") };
      if (sub === "library") return { crumbs: head.concat([{ label: "Project library" }]), body: ContextPage(projectId, "library") };
      if (sub === "assistant") return { crumbs: head.concat([{ label: "Assistant" }]), body: AssistantPage(projectId) };
      if (sub === "agents") {
        const agentKey = parts[3];
        if (!agentKey || !AGENT_KEYS.includes(agentKey)) return { crumbs: head.concat([{ label: "Agents" }]), body: notFoundPage() };
        return { crumbs: head.concat([{ label: AGENT_LABELS[agentKey] }]), body: AgentPage(projectId, agentKey) };
      }
      return { crumbs: head, body: notFoundPage() };
    }
    return { crumbs: [], body: HomePage() };
  }

  /* ---------- Shell + render ---------- */

  function Shell() {
    const route = cleanPath();
    const { crumbs, body } = pageForRoute(route);
    return `
      <div class="sopal-shell ${sidebarOpen ? "drawer-open" : ""}">
        ${Sidebar()}
        <div class="main">
          ${MainHeader(crumbs)}
          ${body}
        </div>
      </div>
      ${modal ? modal.render() : ""}
      ${sidebarOpen ? `<div class="mobile-backdrop" data-toggle-sidebar></div>` : ""}
    `;
  }

  function bindShellEvents() {
    document.querySelectorAll("[data-nav]").forEach((el) => el.addEventListener("click", (event) => {
      const href = el.getAttribute("href");
      if (!href || !href.startsWith("/sopal-v2")) return;
      event.preventDefault();
      sidebarOpen = false;
      projectMenuOpen = false;
      navigate(href);
    }));
    document.querySelectorAll("[data-go]").forEach((el) => el.addEventListener("click", () => {
      sidebarOpen = false;
      navigate(el.getAttribute("data-go"));
    }));
    document.querySelectorAll("[data-toggle-sidebar]").forEach((el) => el.addEventListener("click", () => { sidebarOpen = !sidebarOpen; render(); }));
    document.querySelectorAll("[data-new-project]").forEach((el) => el.addEventListener("click", () => openProjectModal(null)));
    document.querySelectorAll("[data-seed-sample]").forEach((el) => el.addEventListener("click", () => seedSampleProject()));
    document.querySelector("[data-project-menu-toggle]")?.addEventListener("click", (event) => {
      event.stopPropagation();
      projectMenuOpen = !projectMenuOpen;
      render();
    });
    document.querySelectorAll("[data-select-project]").forEach((el) => el.addEventListener("click", () => {
      selectProject(el.dataset.selectProject);
      projectMenuOpen = false;
      navigate(`/sopal-v2/projects/${el.dataset.selectProject}/overview`);
    }));
    // cleanPath() strips "/sopal-v2/" so segment 0 is "projects" and segment 1
    // is the project id. Earlier code mistakenly read route[1]==="projects"
    // which silently no-op'd Remove and Clear-all on Library + Contracts.
    document.querySelectorAll("[data-remove-context]").forEach((b) => b.addEventListener("click", () => {
      const [bucket, indexStr] = b.dataset.removeContext.split(":");
      const route = cleanPath().split("/");
      const projectId = route[0] === "projects" ? route[1] : null;
      const project = getProject(projectId);
      if (!project) return;
      project[bucket].splice(Number(indexStr), 1);
      saveProject(project);
      render();
    }));
    document.querySelectorAll("[data-clear-context]").forEach((b) => b.addEventListener("click", () => {
      if (!confirm(`Clear all ${b.dataset.clearContext}?`)) return;
      const route = cleanPath().split("/");
      const projectId = route[0] === "projects" ? route[1] : null;
      const project = getProject(projectId);
      if (!project) return;
      project[b.dataset.clearContext] = [];
      saveProject(project);
      render();
    }));
    document.addEventListener("click", closeProjectMenuOnOutside, { once: true });
    if (modal) modal.bind(document);
  }

  function closeProjectMenuOnOutside(event) {
    if (!projectMenuOpen) return;
    const menu = document.querySelector(".project-menu");
    const trigger = document.querySelector("[data-project-menu-toggle]");
    if (menu && !menu.contains(event.target) && !trigger?.contains(event.target)) {
      projectMenuOpen = false;
      render();
    }
  }

  function render() {
    root.innerHTML = Shell();
    bindShellEvents();
  }

  function copyText(text) {
    if (navigator.clipboard) navigator.clipboard.writeText(text || "").catch(() => {});
  }

  /* ---------- Boot ---------- */

  window.addEventListener("popstate", render);
  document.addEventListener("click", (event) => {
    const copyBtn = event.target.closest("[data-copy-text]");
    if (copyBtn) {
      copyText(copyBtn.dataset.copyText || "");
      const original = copyBtn.innerHTML;
      copyBtn.innerHTML = `${ICON.copy}<span>Copied</span>`;
      setTimeout(() => { copyBtn.innerHTML = original; }, 1100);
    }
  });

  render();
})();
