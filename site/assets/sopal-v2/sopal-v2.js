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
    "general-correspondence",
    "adjudication-application",
    "adjudication-response",
  ];
  // Order in which drafting agents appear inside a project's sidebar. Keep
  // adjudication-application / response off this list — they're still routable
  // for any old saved chats but no longer surfaced in the v2 sidebar nav.
  const DRAFTING_AGENT_KEYS = [
    "payment-claims",
    "payment-schedules",
    "eots",
    "variations",
    "delay-costs",
    "general-correspondence",
  ];

  // Complex agents are multi-stage workflows with their own page logic —
  // not simple chat / drafting agents. v1 has just the Adjudication
  // Application drafter (see docs/complex-adjudication-application-plan.md).
  const COMPLEX_AGENT_KEYS = [
    "adjudication-application",
  ];
  const COMPLEX_AGENT_LABELS = {
    "adjudication-application": "Adjudication Application",
  };
  const COMPLEX_AGENT_DESCRIPTIONS = {
    "adjudication-application": "Guided multi-stage adjudication application drafter — intake, dispute table, RFIs, parallel item drafting, live master document.",
  };
  const AGENT_LABELS = {
    "payment-claims": "Payment Claims",
    "payment-schedules": "Payment Schedules",
    eots: "EOTs",
    variations: "Variations",
    "delay-costs": "Delay Costs",
    "general-correspondence": "General Correspondence",
    "adjudication-application": "Adjudication Application",
    "adjudication-response": "Adjudication Response",
  };
  const AGENT_DESCRIPTIONS = {
    "payment-claims": "Review or draft payment claim material — BIF Act compliance, work identification, dates, service, evidence.",
    "payment-schedules": "Review or draft payment schedules — scheduled amount, withholding reasons, timing, adjudication risk.",
    eots: "Review or draft extension of time notices and claims — contract notice, causation, critical delay, evidence.",
    variations: "Review or draft variation notices and claims — direction, scope, valuation, time/cost impact, evidence.",
    "delay-costs": "Review or draft delay cost / prolongation / disruption claims — entitlement, causation, quantum.",
    "general-correspondence": "Draft general project correspondence — letters, emails, notices, RFIs, show-cause, suspension, default.",
    "adjudication-application": "Review or draft adjudication application material — chronology, jurisdiction, entitlement, quantum, annexures.",
    "adjudication-response": "Review or draft adjudication response material — jurisdictional objections, payment schedule alignment, evidence.",
  };
  // Every agent in the Drafting Agents sidebar group is drafting-only — the
  // Review/Draft mode picker only made sense when these doubled as document
  // reviewers, but reviewing received documents now lives in the standalone
  // Tools (Payment Claim Reviewer / Payment Schedule Reviewer). Drafting
  // agents go straight into the Word-style doc editor + AI chat workspace.
  const DRAFT_ONLY_AGENTS = new Set([
    "payment-claims",
    "payment-schedules",
    "eots",
    "variations",
    "delay-costs",
    "general-correspondence",
    "adjudication-application",
    "adjudication-response",
  ]);

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
    "general-correspondence": ["Type of letter or email", "Recipient and sender", "Key facts to convey", "Contract clause references", "Supporting documents", "Tone (firm, conciliatory, formal)"],
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

  const AGENT_QUICK_ACTIONS = {
    "payment-claims": [
      "Draft a payment claim for [scope] valued at $[amount] for [period].",
      "Audit a payment claim served on me — what are the strongest jurisdictional objections?",
      "Suggest s 68 BIF Act endorsement wording for this claim.",
      "Compare this claim against the prior claim — what's new or repeated?",
    ],
    "payment-schedules": [
      "Draft a payment schedule responding to a $[amount] claim, scheduled at $[amount].",
      "Itemise the reasons for withholding in a 4-column table.",
      "Stress-test the reasons against s 82(4) BIF Act risk.",
      "Check timing — was the schedule given within s 76 (15 BD or contract)?",
    ],
    eots: [
      "Draft an EOT notice citing clause [#] for [delay event].",
      "Stress-test this EOT for time-bar and causation risks.",
      "List the contemporaneous records I need to support the claim.",
      "Should this also raise a variation or delay cost claim?",
    ],
    variations: [
      "Draft a variation notice for [scope change] valued at $[amount].",
      "Was there a clear instruction or only clarification?",
      "Build a valuation paragraph using Schedule of Rates as a fallback.",
      "Identify the strongest time-bar / waiver risk on this variation.",
    ],
    "delay-costs": [
      "Draft a delay cost / prolongation claim for [period] of [N] working days.",
      "Build a prolongation cost table by week with preliminaries.",
      "Identify overlap with EOT or variation claims.",
      "What's the cleanest causation argument here?",
    ],
    "adjudication-application": [
      "Draft an adjudication application structure responding to a $0 schedule.",
      "Build a chronology of the dispute from claim service to schedule.",
      "List the evidence bundle I need to attach.",
      "Frame the strongest jurisdictional argument upfront.",
    ],
    "adjudication-response": [
      "Draft an adjudication response structure with jurisdictional objections first.",
      "Itemise the reasons already raised vs new reasons (s 82(4)).",
      "Build a quantum response table item-by-item.",
      "Identify points of merit weakness in the application.",
    ],
    "general-correspondence": [
      "Draft a show-cause notice for failure to proceed under clause [#].",
      "Draft a reservation-of-rights letter responding to [the other party]'s [letter / claim].",
      "Draft an email serving the attached document on the [respondent / superintendent].",
      "Draft a polite chase-up letter for an outstanding payment under the contract.",
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
  let projectSelection = new Set();
  let sidebarCollapsed = (() => {
    try { return localStorage.getItem("sopal-v2-sidebar-collapsed") === "1"; } catch (_) { return false; }
  })();
  function setSidebarCollapsed(value) {
    sidebarCollapsed = !!value;
    try { localStorage.setItem("sopal-v2-sidebar-collapsed", sidebarCollapsed ? "1" : "0"); } catch (_) {}
    render();
  }

  let theme = (() => {
    try { return localStorage.getItem("sopal-v2-theme") === "dark" ? "dark" : "light"; } catch (_) { return "light"; }
  })();
  function applyTheme() {
    if (theme === "dark") document.documentElement.setAttribute("data-theme", "dark");
    else document.documentElement.removeAttribute("data-theme");
  }
  function setTheme(value) {
    theme = value === "dark" ? "dark" : "light";
    try { localStorage.setItem("sopal-v2-theme", theme); } catch (_) {}
    applyTheme();
    render();
  }
  applyTheme();

  const PROJECT_CATEGORIES = ["Head contract", "Subcontract", "Pre-dispute", "Active dispute", "Advice", "Other"];

  function localStorageBytesUsed() {
    let total = 0;
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        const value = localStorage.getItem(key) || "";
        total += (key || "").length + value.length;
      }
    } catch (_) {}
    return total * 2; // chars are UTF-16
  }
  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function emptyStore() { return { projects: {}, currentProjectId: null, recentDecisions: [] }; }
  function loadStore() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
      if (parsed && parsed.projects) {
        if (!Array.isArray(parsed.recentDecisions)) parsed.recentDecisions = [];
        return parsed;
      }
    } catch {}
    return emptyStore();
  }
  function saveStore() { localStorage.setItem(STORE_KEY, JSON.stringify(store)); }

  // Track decisions the user has opened so the home page can surface them.
  function trackRecentDecision(meta) {
    if (!meta || !meta.id) return;
    if (!Array.isArray(store.recentDecisions)) store.recentDecisions = [];
    const existing = store.recentDecisions.findIndex((d) => d.id === meta.id);
    if (existing !== -1) store.recentDecisions.splice(existing, 1);
    store.recentDecisions.unshift({
      id: meta.id,
      title: meta.title || meta.id,
      decisionDate: meta.decisionDate || "",
      adjudicator: meta.adjudicator || "",
      claimed: meta.claimed || "",
      awarded: meta.awarded || "",
      visitedAt: Date.now(),
    });
    store.recentDecisions = store.recentDecisions.slice(0, 8);
    saveStore();
  }

  function getProject(id) { return id ? store.projects[id] || null : null; }
  function projectList(opts) {
    const includeArchived = !!(opts && opts.includeArchived);
    const sort = (opts && opts.sort) || "favourites";
    const list = Object.values(store.projects).filter((p) => includeArchived || !p.archived);
    const compareUpdated = (a, b) => (b.updatedAt || 0) - (a.updatedAt || 0);
    if (sort === "name") return list.sort((a, b) => (a.name || "").localeCompare(b.name || ""));
    if (sort === "category") return list.sort((a, b) => ((a.category || "Other").localeCompare(b.category || "Other")) || compareUpdated(a, b));
    if (sort === "recent") return list.sort(compareUpdated);
    // default 'favourites': favourited first, then most-recent
    return list.sort((a, b) => {
      const fa = a.favourite ? 1 : 0;
      const fb = b.favourite ? 1 : 0;
      if (fa !== fb) return fb - fa;
      return compareUpdated(a, b);
    });
  }
  function toggleProjectFavourite(id) {
    const p = getProject(id);
    if (!p) return;
    p.favourite = !p.favourite;
    saveStore();
  }
  function archivedProjectList() {
    return Object.values(store.projects).filter((p) => p.archived).sort((a, b) => (b.archivedAt || 0) - (a.archivedAt || 0));
  }
  function archiveProject(id) {
    const p = getProject(id);
    if (!p) return;
    p.archived = true;
    p.archivedAt = Date.now();
    if (store.currentProjectId === id) {
      const next = projectList()[0];
      store.currentProjectId = next ? next.id : null;
    }
    saveStore();
  }
  function restoreProject(id) {
    const p = getProject(id);
    if (!p) return;
    p.archived = false;
    delete p.archivedAt;
    p.updatedAt = Date.now();
    saveStore();
  }
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

  function sanitiseImportedDoc(d) {
    if (!d || typeof d !== "object") return null;
    return {
      name: String(d.name || "Untitled"),
      text: String(d.text || ""),
      source: String(d.source || "imported"),
      addedAt: typeof d.addedAt === "string" ? d.addedAt : new Date().toISOString(),
    };
  }

  function importProjectFromJson(text) {
    let payload;
    try { payload = JSON.parse(text); } catch (_) { throw new Error("That file isn't valid JSON."); }
    const incoming = payload && typeof payload === "object" && payload.project ? payload.project : payload;
    if (!incoming || typeof incoming !== "object" || typeof incoming.name !== "string") {
      throw new Error("File doesn't look like a Sopal project export.");
    }
    const id = newProjectId();
    const now = Date.now();
    const project = {
      id,
      name: incoming.name.trim().slice(0, 200) || "Imported project",
      claimant: String(incoming.claimant || "").trim(),
      respondent: String(incoming.respondent || "").trim(),
      contractForm: String(incoming.contractForm || "Bespoke"),
      reference: String(incoming.reference || "").trim(),
      userIsParty: incoming.userIsParty === "respondent" ? "respondent" : "claimant",
      contracts: Array.isArray(incoming.contracts) ? incoming.contracts.map(sanitiseImportedDoc).filter(Boolean) : [],
      library: Array.isArray(incoming.library) ? incoming.library.map(sanitiseImportedDoc).filter(Boolean) : [],
      chats: incoming.chats && typeof incoming.chats === "object" ? incoming.chats : {},
      reviews: incoming.reviews && typeof incoming.reviews === "object" ? incoming.reviews : {},
      // Drafting agents' Word-style draft state (per agent: html + chat).
      drafts: incoming.drafts && typeof incoming.drafts === "object" ? incoming.drafts : {},
      // Complex agent state (Adjudication Application workflow: stage,
      // s79Scenario, parsed PC/PS, disputes, RFIs, definitions, etc.).
      complexApps: incoming.complexApps && typeof incoming.complexApps === "object" ? incoming.complexApps : {},
      // Project notes — narrative running log surfaced by the drafting/
      // assistant chats' 'Save as note' button.
      notes: typeof incoming.notes === "string" ? incoming.notes : "",
      // Pinned chat threads in the sidebar's recents.
      pinnedThreads: Array.isArray(incoming.pinnedThreads) ? incoming.pinnedThreads : [],
      // Per-agent saved review state (when the project's review-mode
      // workspace was used). Kept for back-compat.
      category: typeof incoming.category === "string" ? incoming.category : "",
      favourite: !!incoming.favourite,
      createdAt: typeof incoming.createdAt === "number" ? incoming.createdAt : now,
      updatedAt: now,
    };
    store.projects[id] = project;
    store.currentProjectId = id;
    saveStore();
    return project;
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

  // FastAPI returns three error shapes: {detail:"string"}, {detail:[Pydantic]},
  // and {error:"string"}. Format all three as a single readable line so the
  // chat error banner doesn't render "[object Object]".
  function describeApiError(data, fallback) {
    if (!data) return fallback || "Request failed";
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((d) => {
        const where = (d.loc || []).filter((part) => part !== "body").join(" / ") || "request";
        return `${where}: ${d.msg}`;
      }).join("; ");
    }
    if (typeof data.error === "string") return data.error;
    return fallback || "Request failed";
  }
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
    chevLeft: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m15 6-6 6 6 6"/></svg>',
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
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>',
  };

  function researchNav() {
    return [
      { label: "Decision Search", href: "/sopal-v2/research/decisions", icon: ICON.search },
      { label: "Adjudicator Statistics", href: "/sopal-v2/research/adjudicators", icon: ICON.users },
      { label: "Research Agent", href: "/sopal-v2/research/agent", icon: ICON.sparkles },
    ];
  }
  function toolsNav() {
    return [
      { label: "Payment Claim Reviewer", href: "/sopal-v2/tools/payment-claim-reviewer", icon: ICON.file },
      { label: "Payment Schedule Reviewer", href: "/sopal-v2/tools/payment-schedule-reviewer", icon: ICON.file },
      { label: "Due Date Calculator", href: "/sopal-v2/tools/due-date-calculator", icon: ICON.calendar },
      { label: "Interest Calculator", href: "/sopal-v2/tools/interest-calculator", icon: ICON.coins },
    ];
  }
  // Kept for back-compat with Cmd+K palette items / home tiles that still call
  // it. Returns the union of research + tools navs.
  function workspaceNav() {
    return researchNav().concat(toolsNav());
  }

  function projectSubNav(projectId) {
    const base = `/sopal-v2/projects/${projectId}`;
    return [
      { label: "Contract", href: `${base}/contract`, icon: ICON.file },
      { label: "Project Library", href: `${base}/library`, icon: ICON.folder },
      { label: "Assistant", href: `${base}/assistant`, icon: ICON.chat },
    ];
  }

  function projectAgentNav(projectId) {
    const base = `/sopal-v2/projects/${projectId}/agents`;
    return DRAFTING_AGENT_KEYS.map((key) => ({ label: AGENT_LABELS[key], href: `${base}/${key}`, icon: ICON.sparkles }));
  }
  function projectComplexAgentNav(projectId) {
    const base = `/sopal-v2/projects/${projectId}/complex`;
    return COMPLEX_AGENT_KEYS.map((key) => ({ label: COMPLEX_AGENT_LABELS[key], href: `${base}/${key}`, icon: ICON.layers }));
  }

  function Sidebar() {
    const project = currentProject();
    const projects = projectList();
    return `
      <aside class="sopal-sidebar ${sidebarOpen ? "open" : ""} ${sidebarCollapsed ? "collapsed" : ""}">
        <div class="sidebar-brand">
          <a href="/sopal-v2" data-nav>Sopal</a>
          <span class="brand-pill">v2</span>
          <button class="sidebar-collapse-btn" type="button" data-toggle-collapse aria-label="${sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}" title="${sidebarCollapsed ? "Expand sidebar (⌘\\)" : "Collapse sidebar (⌘\\)"}">${sidebarCollapsed ? ICON.chevRight : ICON.chevLeft || ICON.close}</button>
        </div>

        <div class="sidebar-scroll">
          <div class="nav-group-title">Research</div>
          ${researchNav().map((item) => `
            <a class="nav-item ${isActivePrefix(item.href) ? "active" : ""}" href="${item.href}" data-nav>
              <span class="nav-icon">${item.icon}</span>
              <span class="nav-label">${escapeHtml(item.label)}</span>
            </a>`).join("")}

          <div class="nav-divider"></div>

          <div class="nav-group-title">Tools</div>
          ${toolsNav().map((item) => `
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
                    <div class="project-menu-row-wrap">
                      <button class="project-menu-row-fav" type="button" data-toggle-favourite="${attr(p.id)}" title="${p.favourite ? "Unfavourite" : "Favourite"}">${p.favourite ? "★" : "☆"}</button>
                    <button class="project-menu-row ${p.id === store.currentProjectId ? "active" : ""}" type="button" data-select-project="${attr(p.id)}">
                      <span class="truncate">${escapeHtml(p.name)}</span>
                    </button>
                    </div>`).join("")}
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
              <div class="nav-subgroup-title">Drafting agents</div>
              ${projectAgentNav(project.id).map((item) => `
                <a class="nav-item nav-item-sub ${isActivePrefix(item.href) ? "active" : ""}" href="${item.href}" data-nav>
                  <span class="nav-icon">${item.icon}</span>
                  <span class="nav-label">${escapeHtml(item.label)}</span>
                </a>`).join("")}
              <div class="nav-subgroup-title">Complex agents</div>
              ${projectComplexAgentNav(project.id).map((item) => `
                <a class="nav-item nav-item-sub ${isActivePrefix(item.href) ? "active" : ""}" href="${item.href}" data-nav>
                  <span class="nav-icon">${item.icon}</span>
                  <span class="nav-label">${escapeHtml(item.label)}</span>
                </a>`).join("")}
              ${sidebarRecentThreads(project)}
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
          <button class="palette-hint-btn theme-toggle-btn" type="button" data-toggle-theme title="${theme === "dark" ? "Switch to light mode" : "Switch to dark mode"} (⌘/Ctrl + Shift + D)" aria-label="Toggle theme">
            ${theme === "dark"
              ? '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>'
              : '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'}
            <span class="theme-toggle-label">${theme === "dark" ? "Light" : "Dark"}</span>
          </button>
          <button class="palette-hint-btn" type="button" data-open-palette title="Open command palette">
            <span class="kbd-key">${navigator.platform && navigator.platform.toLowerCase().includes("mac") ? "⌘" : "Ctrl"}</span>
            <span class="kbd-key">K</span>
          </button>
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

  function findLatestReview() {
    let best = null;
    for (const project of Object.values(store.projects || {})) {
      for (const [key, review] of Object.entries(project.reviews || {})) {
        if (!review || !review.analysis || review.analysis.error) continue;
        const ts = review.updatedAt || review.analysis.updatedAt || project.updatedAt || 0;
        if (!best || ts > best.ts) {
          const parts = key.split(":"); // review:agentKey:submodeId
          best = { ts, project, agentKey: parts[1], submodeId: parts[2], review };
        }
      }
    }
    return best;
  }

  function resumePreviewFor(latest) {
    if (!latest) return "";
    const project = latest.project;
    // Prefer the most recent assistant message from the review chat thread.
    const chatKey = `chat:review:${latest.agentKey}:${latest.submodeId}`;
    const chat = project.chats && project.chats[chatKey];
    if (chat && Array.isArray(chat.messages)) {
      for (let i = chat.messages.length - 1; i >= 0; i--) {
        const m = chat.messages[i];
        if (m && m.role === "assistant" && m.content) {
          return plainPreview(m.content).slice(0, 130) + (plainPreview(m.content).length > 130 ? "…" : "");
        }
      }
    }
    // Fallback: the analysis summary itself.
    const summary = latest.review && latest.review.analysis && latest.review.analysis.summary;
    if (summary) {
      const p = plainPreview(summary);
      return p.slice(0, 130) + (p.length > 130 ? "…" : "");
    }
    return "";
  }

  function HomePage() {
    const research = researchNav();
    const tools = toolsNav();
    const projects = projectList();
    const recent = (store.recentDecisions || []).slice(0, 6);
    return PageBody(`
      <div class="home-shell">
        <section class="home-hero">
          <div class="home-hero-row">
            <div>
              <h2>Welcome to Sopal v2</h2>
              <p>Search adjudication decisions, run BIF Act calculators, and manage SOPA workflows project by project.</p>
            </div>
            <button class="ghost-button compact whatsnew-btn" type="button" data-open-whatsnew title="See recent feature releases">${ICON.sparkles}<span>What's new</span></button>
          </div>
        </section>

        ${recent.length ? `
          <section class="home-section">
            <div class="section-head"><h3>Recently viewed decisions</h3><p>Pick up where you left off.</p></div>
            <div class="recent-decisions-grid">
              ${recent.map((d) => `
                <a class="recent-decision-card" href="/sopal-v2/research/decisions/${encodeURIComponent(d.id)}" data-nav>
                  <strong>${escapeHtml(d.title || d.id)}</strong>
                  <span class="muted">${[d.decisionDate, d.adjudicator].filter(Boolean).map((s) => escapeHtml(s)).join(" · ")}</span>
                  ${d.claimed || d.awarded ? `<span class="muted">${[d.claimed && `${escapeHtml(d.claimed)} claimed`, d.awarded && `${escapeHtml(d.awarded)} awarded`].filter(Boolean).join(" · ")}</span>` : ""}
                </a>
              `).join("")}
            </div>
          </section>
        ` : ""}

        <section class="home-section">
          <div class="section-head"><h3>Research</h3><p>Search decisions, look up adjudicators, ask the research agent.</p></div>
          <div class="tile-grid">
            ${research.map((t) => `
              <a class="tile" href="${t.href}" data-nav>
                <span class="tile-icon">${t.icon}</span>
                <strong>${escapeHtml(t.label)}</strong>
              </a>`).join("")}
          </div>
        </section>

        <section class="home-section">
          <div class="section-head"><h3>Tools</h3><p>Standalone utilities — no project required.</p></div>
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
              <button class="dark-button" type="button" data-new-project>Create project</button>
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

  function projectRow(p, opts) {
    const meta = [p.reference, p.contractForm, p.claimant ? `${p.claimant} v ${p.respondent || "?"}` : ""].filter(Boolean).join(" · ");
    const isSelected = projectSelection.has(p.id);
    const checkboxHtml = `<label class="project-row-check" onclick="event.stopPropagation()"><input type="checkbox" data-project-checkbox="${attr(p.id)}" ${isSelected ? "checked" : ""}></label>`;
    if (opts && opts.archived) {
      return `
        <div class="project-row archived-row ${isSelected ? "is-selected" : ""}">
          ${checkboxHtml}
          <div class="project-row-icon">${ICON.folder}</div>
          <div class="project-row-text">
            <strong>${escapeHtml(p.name)}</strong>
            <span>${escapeHtml(meta || "Bespoke contract")}</span>
          </div>
          <span class="status-pill">${p.contracts.length || 0} contract docs · ${p.library.length || 0} library</span>
          <button class="ghost-button compact" type="button" data-restore-project="${attr(p.id)}">${ICON.arrowUpRight}<span>Restore</span></button>
        </div>`;
    }
    return `
      <a class="project-row ${isSelected ? "is-selected" : ""} ${p.favourite ? "is-favourite" : ""}" href="/sopal-v2/projects/${attr(p.id)}/overview" data-nav>
        ${checkboxHtml}
        <div class="project-row-icon">${ICON.file}</div>
        <div class="project-row-text">
          <strong>${escapeHtml(p.name)}${p.favourite ? ' <span class="fav-star">★</span>' : ""}</strong>
          <span>${escapeHtml(meta || "Bespoke contract")}</span>
        </div>
        <button class="project-row-fav" type="button" data-toggle-favourite="${attr(p.id)}" onclick="event.stopPropagation(); event.preventDefault();" title="${p.favourite ? "Unfavourite" : "Favourite"}">${p.favourite ? "★" : "☆"}</button>
        ${p.category ? `<span class="category-pill">${escapeHtml(p.category)}</span>` : ""}
        <span class="status-pill">${p.contracts.length || 0} contract docs · ${p.library.length || 0} library</span>
        <span class="row-chev">${ICON.chevRight}</span>
      </a>
    `;
  }

  /* ---------- Research: Decision search ---------- */

  function decisionSearchLabel(params) {
    const q = params.get("q") || "any decision";
    const filters = ["startDate", "endDate", "minClaim", "maxClaim"].filter((k) => params.get(k));
    return filters.length ? `${q} · ${filters.length} filter${filters.length === 1 ? "" : "s"}` : q;
  }
  function paramsKeyDecisions(params) {
    return ["q", "sort", "startDate", "endDate", "minClaim", "maxClaim"].map((k) => `${k}=${params.get(k) || ""}`).join("&");
  }
  function getSavedDecisionSearches() {
    return Array.isArray(store.savedSearches) ? store.savedSearches : [];
  }
  function saveCurrentDecisionSearch(params) {
    if (!Array.isArray(store.savedSearches)) store.savedSearches = [];
    const key = paramsKeyDecisions(params);
    if (!params.get("q") && !["startDate", "endDate", "minClaim", "maxClaim"].some((k) => params.get(k))) return null;
    if (store.savedSearches.find((s) => s.key === key)) return null;
    const entry = { id: `s_${Math.random().toString(36).slice(2, 8)}`, key, qs: params.toString(), label: decisionSearchLabel(params), savedAt: Date.now() };
    store.savedSearches.unshift(entry);
    store.savedSearches = store.savedSearches.slice(0, 12);
    saveStore();
    return entry;
  }
  function deleteSavedDecisionSearch(id) {
    if (!Array.isArray(store.savedSearches)) return;
    store.savedSearches = store.savedSearches.filter((s) => s.id !== id);
    saveStore();
  }
  function renameSavedDecisionSearch(id, label) {
    if (!Array.isArray(store.savedSearches)) return;
    const entry = store.savedSearches.find((s) => s.id === id);
    if (!entry) return;
    entry.label = (label || "").trim() || entry.label;
    saveStore();
  }

  function DecisionsPage() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q") || "";
    const sort = params.get("sort") || "relevance";
    const filtersOn = ["startDate", "endDate", "minClaim", "maxClaim"].some((k) => params.get(k));
    const saved = getSavedDecisionSearches();
    const canSave = !!q || filtersOn;
    const currentKey = paramsKeyDecisions(params);
    const alreadySaved = saved.some((s) => s.key === currentKey);
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
      document.querySelector("[data-save-search]")?.addEventListener("click", () => {
        const entry = saveCurrentDecisionSearch(params);
        if (entry) render();
      });
      document.querySelectorAll("[data-saved-search]").forEach((el) => el.addEventListener("click", (e) => {
        if (e.target.closest("[data-delete-saved], [data-rename-saved]")) return;
        navigate(`/sopal-v2/research/decisions?${el.dataset.savedSearch}`);
      }));
      document.querySelectorAll("[data-delete-saved]").forEach((el) => el.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        deleteSavedDecisionSearch(el.dataset.deleteSaved);
        render();
      }));
      document.querySelectorAll("[data-rename-saved]").forEach((el) => el.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const id = el.dataset.renameSaved;
        const entry = (store.savedSearches || []).find((s) => s.id === id);
        if (!entry) return;
        const next = prompt("Rename saved search", entry.label);
        if (next === null) return;
        renameSavedDecisionSearch(id, next);
        render();
      }));
      const initialPage = Math.max(1, parseInt(params.get("page") || "1", 10) || 1);
      const initialOffset = (initialPage - 1) * 10;
      if (q || params.toString()) fetchDecisionResults(params, initialOffset);
    }, 0);

    return PageBody(`
      <div class="page-shell">
        <h1 class="page-title">Decision search</h1>
        <p class="page-sub">Searches Sopal's adjudication decision database. Results render here — no jumps to the live site.</p>

        <form class="search-form-v2" data-decision-search>
          <div class="search-row-main">
            <input class="text-input" name="q" type="search" value="${attr(q)}" placeholder="Adjudicator, party, section, keywords…" autofocus>
            <button class="dark-button" type="submit">Search</button>
          </div>
          <div class="search-row-meta">
            <label class="search-meta-label">Sort
              <select class="select-input compact" name="sort">
                ${["relevance","newest","oldest","claim_high","claim_low","adj_high","adj_low"].map((s) => `<option value="${s}" ${sort===s?"selected":""}>${labelSort(s)}</option>`).join("")}
              </select>
            </label>
            <details class="filters-v2" ${filtersOn ? "open" : ""}>
              <summary><span class="filters-toggle-label">Filters${filtersOn ? " · active" : ""}</span></summary>
              <div class="filters-grid">
                <label>From<input class="text-input" name="startDate" type="date" value="${attr(params.get("startDate") || "")}"></label>
                <label>To<input class="text-input" name="endDate" type="date" value="${attr(params.get("endDate") || "")}"></label>
                <label>Min claimed<input class="text-input" name="minClaim" type="number" step="1000" value="${attr(params.get("minClaim") || "")}"></label>
                <label>Max claimed<input class="text-input" name="maxClaim" type="number" step="1000" value="${attr(params.get("maxClaim") || "")}"></label>
              </div>
            </details>
            ${canSave ? `<button class="ghost-button compact" type="button" data-save-search ${alreadySaved ? "disabled" : ""}>${alreadySaved ? "Saved" : "Save search"}</button>` : ""}
          </div>
        </form>

        ${saved.length ? `
          <div class="saved-search-row">
            <span class="saved-search-eyebrow">Saved searches</span>
            ${saved.map((s) => `
              <button class="saved-search-chip" type="button" data-saved-search="${attr(s.qs)}">
                <span class="saved-search-label">${escapeHtml(s.label)}</span>
                <span class="saved-search-edit" data-rename-saved="${attr(s.id)}" title="Rename">✎</span>
                <span class="saved-search-x" data-delete-saved="${attr(s.id)}" title="Remove">×</span>
              </button>`).join("")}
          </div>
        ` : ""}

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

  // Build the page-selector control rendered at the top and the bottom of
  // the results list. Returns "" when there's only a single page worth of
  // results so we don't show a pointless lone "1" pill.
  function buildPagination(currentPage, total, pageSize) {
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    if (totalPages <= 1) return "";
    const items = [];
    function add(p) { items.push(p); }
    function addEllipsis() { items.push("…"); }
    // Compact 1 … (cur-1) (cur) (cur+1) … N pattern, with the cur window
    // expanding to the edges so we never render two adjacent ellipses.
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) add(i);
    } else {
      add(1);
      if (currentPage > 4) addEllipsis();
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);
      for (let i = start; i <= end; i++) add(i);
      if (currentPage < totalPages - 3) addEllipsis();
      add(totalPages);
    }
    const prevDisabled = currentPage === 1;
    const nextDisabled = currentPage === totalPages;
    return `
      <nav class="pager" aria-label="Pagination">
        <button class="pager-btn pager-step" type="button" data-page="${currentPage - 1}" ${prevDisabled ? "disabled aria-disabled='true'" : ""}>← Prev</button>
        ${items.map((it) => it === "…"
          ? `<span class="pager-ellipsis" aria-hidden="true">…</span>`
          : `<button class="pager-btn ${it === currentPage ? "active" : ""}" type="button" data-page="${it}" ${it === currentPage ? "aria-current='page'" : ""}>${it}</button>`
        ).join("")}
        <button class="pager-btn pager-step" type="button" data-page="${currentPage + 1}" ${nextDisabled ? "disabled aria-disabled='true'" : ""}>Next →</button>
        <span class="pager-meta muted">Page ${currentPage.toLocaleString()} of ${totalPages.toLocaleString()} · ${total.toLocaleString()} result${total === 1 ? "" : "s"}</span>
      </nav>`;
  }

  function bindPagination(mount, params) {
    mount.querySelectorAll(".pager [data-page]").forEach((btn) => {
      if (btn.disabled) return;
      btn.addEventListener("click", () => {
        const targetPage = Number(btn.dataset.page);
        if (!targetPage || targetPage < 1) return;
        const next = new URLSearchParams(params);
        if (targetPage === 1) next.delete("page");
        else next.set("page", String(targetPage));
        navigate(`/sopal-v2/research/decisions?${next.toString()}`);
      });
    });
  }

  async function fetchDecisionResults(params, offset) {
    const mount = document.getElementById("decision-results");
    if (!mount) return;
    mount.innerHTML = skeletonRows();
    const pageSize = 10;
    const qs = new URLSearchParams(params);
    qs.set("limit", String(pageSize));
    qs.set("offset", String(offset || 0));
    try {
      const response = await fetch(`/api/sopal-v2/search?${qs.toString()}`, { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(describeApiError(data, "Search failed"));
      const items = Array.isArray(data.items) ? data.items : [];
      const total = Number(data.total || items.length);
      if (!items.length) {
        mount.innerHTML = EmptyState("No decisions match.", "Adjust your query or filters.");
        return;
      }
      const currentPage = Math.floor((offset || 0) / pageSize) + 1;
      const pagerHtml = buildPagination(currentPage, total, pageSize);
      mount.innerHTML = `
        <div class="results-shell">
          <div class="results-head">
            <h3>${total.toLocaleString()} result${total === 1 ? "" : "s"}</h3>
            ${pagerHtml ? `<div class="pager-wrap pager-top">${pagerHtml}</div>` : ""}
          </div>
          <div class="results-list">${items.map(renderDecisionItem).join("")}</div>
          ${pagerHtml ? `<div class="pager-wrap pager-bottom">${pagerHtml}</div>` : ""}
        </div>`;
      mount.querySelectorAll("[data-decision-id]").forEach((el) => el.addEventListener("click", () => {
        const meta = el.dataset.meta ? safeParseJson(el.dataset.meta) : null;
        loadDecisionDetail(el.dataset.decisionId, el.dataset.title, meta);
      }));
      bindPagination(mount, params);
      // Scroll the results back to the top when the page changes so the user
      // doesn't land deep in the previous page's tail.
      mount.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Search failed")}</div>`;
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
    if (meta) trackRecentDecision({ ...meta, id, title: title || meta.title || id });
    mount.innerHTML = `<div class="card-head"><h3>${escapeHtml(title || "Decision")}</h3></div><div class="card-body">${skeletonRows()}</div>`;
    try {
      const response = await fetch(`/api/decision-text/${encodeURIComponent(id)}`, { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Decision text failed");
      const text = (data.fullText || "").trim();
      const metaHeader = meta ? renderDecisionMetaHeader(meta) : "";
      const citation = formatDecisionCitation(title, id, meta);
      mount.innerHTML = `
        <div class="card-head">
          <div><h3>${escapeHtml(title || id)}</h3><p class="muted">${escapeHtml(id)}</p></div>
          <div class="panel-actions">
            <a class="link-button small" href="/sopal-v2/research/decisions/${encodeURIComponent(id)}" data-nav title="Shareable link to this decision">Open page</a>
            <button class="ghost-button compact" type="button" data-copy-text="${attr(citation)}" title="Copy a short citation for this decision">${ICON.copy}<span>Copy citation</span></button>
            <button class="ghost-button compact" type="button" data-copy-text="${attr(text.slice(0, 8000))}" title="Copy decision text">${ICON.copy}<span>Copy text</span></button>
            ${projectList().length ? `<button class="ghost-button compact" type="button" data-save-decision="${attr(id)}" data-decision-title="${attr(title || id)}" title="Save this decision to a project's library">${ICON.layers}<span>Save to project</span></button>` : ""}
          </div>
        </div>
        ${metaHeader}
        <div class="card-body">
          ${text ? `<div class="decision-text">${formatDecisionText(text)}</div>${text.length > 12000 ? `<p class="muted decision-text-trunc">Text truncated to first 12,000 characters. ${text.length.toLocaleString()} chars total.</p>` : ""}` : EmptyState("No text on file.", "This record has no extracted text.")}
        </div>`;
      mount.querySelector("[data-save-decision]")?.addEventListener("click", () => {
        openSaveDecisionModal({ id, title, meta, text });
      });
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load decision text")}</div>`;
    }
  }

  function openSaveDecisionModal({ id, title, meta, text }) {
    const projects = projectList();
    if (!projects.length) return;
    const defaultName = `Decision: ${title || id}`;
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>Save to project library</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <form class="modal-body" data-save-decision-form>
              <label class="span-2">Project<select class="select-input" name="projectId" required>
                ${projects.map((p) => `<option value="${attr(p.id)}" ${p.id === store.currentProjectId ? "selected" : ""}>${escapeHtml(p.name)}</option>`).join("")}
              </select></label>
              <label class="span-2">Library item name<input class="text-input" name="name" value="${attr(defaultName)}" required></label>
              <p class="muted">The decision text and metadata will be saved as a library item in the selected project.</p>
              <div class="modal-actions">
                <button class="ghost-button" type="button" data-modal-close>Cancel</button>
                <button class="dark-button" type="submit">Save to project</button>
              </div>
            </form>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        rootEl.querySelector("[data-save-decision-form]")?.addEventListener("submit", (event) => {
          event.preventDefault();
          const data = Object.fromEntries(new FormData(event.currentTarget).entries());
          const project = getProject(data.projectId);
          if (!project) { close(); return; }
          const metaLines = [
            meta?.decisionDate ? `Date: ${meta.decisionDate}` : "",
            meta?.adjudicator ? `Adjudicator: ${meta.adjudicator}` : "",
            (meta?.claimant || meta?.respondent) ? `Parties: ${meta?.claimant || "?"} v ${meta?.respondent || "?"}` : "",
            meta?.claimed ? `Claimed: ${meta.claimed}` : "",
            meta?.awarded ? `Awarded: ${meta.awarded}` : "",
            `Decision id: ${id}`,
          ].filter(Boolean).join("\n");
          const body = `${metaLines}\n\n---\n\n${text || ""}`;
          project.library.push({
            name: data.name || defaultName,
            text: body,
            source: `decision:${id}`,
            addedAt: new Date().toISOString(),
          });
          saveProject(project);
          modal = null;
          render();
        });
      },
    };
    render();
  }

  function formatDecisionCitation(title, id, meta) {
    // Sopal corpus is QLD adjudication decisions and historical BCIPA
    // matters. Citations are typically informal — adjudicator name and the
    // decision date are the useful identifiers — but we preserve the EJS id
    // so the reader can pull it back up in Sopal.
    const t = (title || "").trim();
    const dt = meta?.decisionDate ? ` (${meta.decisionDate})` : "";
    const adj = meta?.adjudicator ? ` — Adjudicator ${meta.adjudicator}` : "";
    const act = meta?.act ? ` [${meta.act}]` : "";
    const ref = id ? ` [${id}]` : "";
    return `${t}${dt}${adj}${act}${ref}`.trim();
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

  // Adjudicator list paginates client-side over the cached /api/adjudicators
  // result. State is module-local so filter/sort changes can reset back to the
  // first page without a URL round-trip.
  let adjListPage = 1;
  const ADJ_PAGE_SIZE = 10;

  async function fetchAdjudicators() {
    const mount = document.getElementById("adj-results");
    if (!mount) return;
    try {
      const response = await fetch("/api/adjudicators", { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(describeApiError(data, "Adjudicator endpoint failed"));
      window.__sopalAdjudicators = Array.isArray(data) ? data : [];
      adjListPage = 1;
      renderAdjudicators();
      document.querySelector("[data-adj-filter]")?.addEventListener("input", () => { adjListPage = 1; renderAdjudicators(); });
      document.querySelector("[data-adj-sort]")?.addEventListener("change", () => { adjListPage = 1; renderAdjudicators(); });
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load adjudicators")}</div>`;
    }
  }

  function bindInPagePager(scope, onPage) {
    scope.querySelectorAll(".pager [data-page]").forEach((btn) => {
      if (btn.disabled) return;
      btn.addEventListener("click", () => {
        const target = Number(btn.dataset.page);
        if (!target || target < 1) return;
        onPage(target);
      });
    });
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
    if (!items.length) {
      mount.innerHTML = EmptyState("No adjudicators match.", "Clear or change the filter.");
      return;
    }
    const total = items.length;
    const totalPages = Math.max(1, Math.ceil(total / ADJ_PAGE_SIZE));
    if (adjListPage > totalPages) adjListPage = totalPages;
    if (adjListPage < 1) adjListPage = 1;
    const start = (adjListPage - 1) * ADJ_PAGE_SIZE;
    const pageItems = items.slice(start, start + ADJ_PAGE_SIZE);
    const pagerHtml = buildPagination(adjListPage, total, ADJ_PAGE_SIZE);
    mount.innerHTML = `
      <div class="results-shell">
        <div class="results-head">
          <h3>${total.toLocaleString()} adjudicator${total === 1 ? "" : "s"}</h3>
          ${pagerHtml ? `<div class="pager-wrap pager-top">${pagerHtml}</div>` : ""}
        </div>
        <div class="adj-grid">${pageItems.map((item) => `
          <button class="adj-card" type="button" data-adjudicator="${attr(item.name)}">
            <strong>${escapeHtml(item.name)}</strong>
            <span class="muted">${item.totalDecisions} decisions</span>
            <div class="adj-card-row">${formatCurrencyCompact(item.totalClaimAmount)} claimed</div>
            <div class="adj-card-row">${formatCurrencyCompact(item.totalAwardedAmount)} awarded</div>
            <span class="rate-pill">${pct(item.avgAwardRate)} avg award</span>
          </button>`).join("")}</div>
        ${pagerHtml ? `<div class="pager-wrap pager-bottom">${pagerHtml}</div>` : ""}
      </div>`;
    mount.querySelectorAll("[data-adjudicator]").forEach((b) => b.addEventListener("click", () => loadAdjudicatorDetail(b.dataset.adjudicator)));
    bindInPagePager(mount, (target) => { adjListPage = target; renderAdjudicators(); });
  }

  // Module-local state so the detail panel can paginate without leaking
  // through the URL.
  let adjDetailPage = 1;
  let adjDetailDecisions = [];
  let adjDetailName = "";

  async function loadAdjudicatorDetail(name) {
    const mount = document.getElementById("adj-detail");
    if (!mount) return;
    mount.innerHTML = `<div class="card-head"><h3>${escapeHtml(name)}</h3><p class="muted">Loading decisions…</p></div><div class="card-body">${skeletonRows()}</div>`;
    try {
      const response = await fetch(`/api/adjudicator/${encodeURIComponent(name)}`, { credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(describeApiError(data, "Adjudicator detail failed"));
      adjDetailDecisions = Array.isArray(data) ? data : [];
      adjDetailName = name;
      adjDetailPage = 1;
      renderAdjudicatorDetail();
    } catch (error) {
      mount.innerHTML = `<div class="error-banner">${escapeHtml(error.message || "Could not load adjudicator")}</div>`;
    }
  }

  function renderAdjudicatorDetail() {
    const mount = document.getElementById("adj-detail");
    if (!mount) return;
    const decisions = adjDetailDecisions;
    const name = adjDetailName;
    const summary = (window.__sopalAdjudicators || []).find((x) => x.name === name) || {};
    const claimedSum = decisions.reduce((s, d) => s + (Number(d.claimAmount) || 0), 0);
    const awardedSum = decisions.reduce((s, d) => s + (Number(d.awardedAmount) || 0), 0);
    const zeroes = decisions.filter((d) => Number(d.awardedAmount) === 0).length;
    const total = decisions.length;
    const totalPages = Math.max(1, Math.ceil(total / ADJ_PAGE_SIZE));
    if (adjDetailPage > totalPages) adjDetailPage = totalPages;
    if (adjDetailPage < 1) adjDetailPage = 1;
    const start = (adjDetailPage - 1) * ADJ_PAGE_SIZE;
    const pageItems = decisions.slice(start, start + ADJ_PAGE_SIZE);
    const pagerHtml = buildPagination(adjDetailPage, total, ADJ_PAGE_SIZE);
    mount.innerHTML = `
      <div class="card-head"><div><h3>${escapeHtml(name)}</h3><p class="muted">${total} decision${total === 1 ? "" : "s"}</p></div></div>
      <div class="card-body">
        <div class="metric-grid compact">
          <div class="metric"><strong>${total}</strong><span>decisions</span></div>
          <div class="metric"><strong>${formatCurrencyCompact(claimedSum)}</strong><span>total claimed</span></div>
          <div class="metric"><strong>${formatCurrencyCompact(awardedSum)}</strong><span>total awarded</span></div>
          <div class="metric"><strong>${pct(summary.avgAwardRate || 0)}</strong><span>avg award rate</span></div>
          <div class="metric"><strong>${zeroes}</strong><span>$0 awards</span></div>
          <div class="metric"><strong>${pct(summary.avgClaimantFeeProportion || 0)}</strong><span>claimant fee share</span></div>
        </div>
        ${pagerHtml ? `<div class="pager-wrap pager-top">${pagerHtml}</div>` : ""}
        <div class="mini-list">
          ${pageItems.map((d) => `
            <article class="mini-item" ${d.id ? `data-decision-id="${attr(d.id)}" data-title="${attr(d.title || "")}" tabindex="0"` : ""}>
              <strong>${escapeHtml(d.title || "Decision")}</strong>
              <span class="muted">${escapeHtml(shortDate(d.date) || "")}${d.outcome ? ` · ${escapeHtml(d.outcome)}` : ""}${d.projectType ? ` · ${escapeHtml(d.projectType)}` : ""}</span>
              <span class="muted">claimed ${formatCurrencyCompact(d.claimAmount)} · awarded ${formatCurrencyCompact(d.awardedAmount)}</span>
            </article>`).join("")}
        </div>
        ${pagerHtml ? `<div class="pager-wrap pager-bottom">${pagerHtml}</div>` : ""}
      </div>`;
    mount.querySelectorAll("[data-decision-id]").forEach((el) => el.addEventListener("click", () => {
      navigate(`/sopal-v2/research/decisions?q=${encodeURIComponent(el.dataset.title || "")}`);
    }));
    bindInPagePager(mount, (target) => { adjDetailPage = target; renderAdjudicatorDetail(); });
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
    const copyText = `${result.title}: ${formatDate(result.finalDate)}\n${result.basis}\nStart date: ${formatDate(result.startDate)}\nBusiness-day period: ${result.days}\nNon-business days skipped: ${summariseSkipped(result.skipped)}`;
    return `<div class="calc-result">
      <span class="calc-result-tag">${escapeHtml(result.title)}</span>
      <strong>${escapeHtml(formatDate(result.finalDate))}</strong>
      <p>${escapeHtml(result.basis)}</p>
      <dl>
        <dt>Start date</dt><dd>${escapeHtml(formatDate(result.startDate))}</dd>
        <dt>Business-day period</dt><dd>${result.days}</dd>
        <dt>Non-business days skipped</dt><dd>${escapeHtml(summariseSkipped(result.skipped))}</dd>
      </dl>
      <div class="result-actions">
        <button class="ghost-button compact" type="button" data-copy-text="${attr(copyText)}">${ICON.copy}<span>Copy</span></button>
        ${projectList().length ? `<button class="ghost-button compact" type="button" data-save-calc-to-project="due-date" data-calc-payload="${attr(JSON.stringify({ kind: "due-date", title: `${result.title}: ${formatDate(result.finalDate)}`, body: copyText }))}">${ICON.layers}<span>Save to project</span></button>` : ""}
      </div>
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
        <div class="result-actions">
          <button class="ghost-button compact" type="button" data-copy-text="${attr(copy)}">${ICON.copy}<span>Copy results</span></button>
          ${projectList().length ? `<button class="ghost-button compact" type="button" data-save-calc-to-project="interest" data-calc-payload="${attr(JSON.stringify({ kind: "interest", title: `Interest on ${formatCurrencyFull(result.principal)} (${result.days} days)`, body: copy }))}">${ICON.layers}<span>Save to project</span></button>` : ""}
        </div>
        ${breakdown}
      </div>`;
  }

  /* ---------- Project workspace ---------- */

  function ProjectsListPage() {
    const params = new URLSearchParams(window.location.search);
    const showArchived = params.get("archived") === "1";
    const categoryFilter = params.get("category") || "";
    const sort = params.get("sort") || "favourites";
    const active = projectList({ sort });
    const archived = archivedProjectList();
    const baseList = showArchived ? archived : active;
    const projects = categoryFilter ? baseList.filter((p) => (p.category || "Other") === categoryFilter) : baseList;
    const categoriesPresent = Array.from(new Set(baseList.map((p) => p.category || "Other"))).filter(Boolean);
    const bytes = localStorageBytesUsed();
    const quotaApprox = 5 * 1024 * 1024; // browsers typically grant ~5MB to a single origin
    const pct = Math.min(100, Math.round((bytes / quotaApprox) * 100));
    // Drop selections that are no longer on this view (e.g. tab switch).
    const visibleIds = new Set(projects.map((p) => p.id));
    for (const id of projectSelection) if (!visibleIds.has(id)) projectSelection.delete(id);
    const selectedCount = projectSelection.size;
    const allSelected = projects.length > 0 && projects.every((p) => projectSelection.has(p.id));
    setTimeout(() => {
      document.querySelectorAll("[data-restore-project]").forEach((b) => b.addEventListener("click", () => {
        restoreProject(b.dataset.restoreProject);
        render();
      }));
      document.querySelectorAll("[data-project-checkbox]").forEach((cb) => cb.addEventListener("change", () => {
        const id = cb.dataset.projectCheckbox;
        if (cb.checked) projectSelection.add(id); else projectSelection.delete(id);
        render();
      }));
      document.querySelector("[data-select-all-projects]")?.addEventListener("change", (event) => {
        if (event.target.checked) projects.forEach((p) => projectSelection.add(p.id));
        else projects.forEach((p) => projectSelection.delete(p.id));
        render();
      });
      document.querySelector("[data-bulk-archive]")?.addEventListener("click", () => {
        Array.from(projectSelection).forEach((id) => archiveProject(id));
        projectSelection = new Set();
        render();
      });
      document.querySelector("[data-bulk-restore]")?.addEventListener("click", () => {
        Array.from(projectSelection).forEach((id) => restoreProject(id));
        projectSelection = new Set();
        render();
      });
      document.querySelector("[data-bulk-delete]")?.addEventListener("click", () => {
        if (!confirm(`Permanently delete ${projectSelection.size} project${projectSelection.size === 1 ? "" : "s"}? This cannot be undone.`)) return;
        Array.from(projectSelection).forEach((id) => deleteProject(id));
        projectSelection = new Set();
        render();
      });
      document.querySelector("[data-bulk-clear]")?.addEventListener("click", () => {
        projectSelection = new Set();
        render();
      });
      document.querySelector("[data-project-sort]")?.addEventListener("change", (event) => {
        const next = new URLSearchParams(window.location.search);
        next.set("sort", event.target.value);
        navigate(`/sopal-v2/projects?${next.toString()}`);
      });
    }, 0);
    return PageBody(`
      <div class="page-shell">
        <div class="page-head">
          <div><h1 class="page-title">${showArchived ? "Archived projects" : "Your projects"}</h1><p class="page-sub">Each project is one construction contract — head contract or subcontract.</p></div>
          <div class="page-actions">
            <select class="select-input compact" data-project-sort title="Sort projects">
              <option value="favourites" ${sort === "favourites" ? "selected" : ""}>Favourites first</option>
              <option value="recent" ${sort === "recent" ? "selected" : ""}>Most recent</option>
              <option value="name" ${sort === "name" ? "selected" : ""}>Name A–Z</option>
              <option value="category" ${sort === "category" ? "selected" : ""}>Category</option>
            </select>
            <label class="ghost-button compact" title="Import a sopal-*.json export">${ICON.upload}<span>Import</span><input type="file" data-import-project accept="application/json,.json" hidden></label>
            <button class="dark-button" type="button" data-new-project>${ICON.plus}<span>New project</span></button>
          </div>
        </div>
        ${(active.length || archived.length) ? `
          <div class="projects-tabs">
            <a class="projects-tab ${!showArchived ? "active" : ""}" href="/sopal-v2/projects" data-nav>Active${active.length ? ` · ${active.length}` : ""}</a>
            <a class="projects-tab ${showArchived ? "active" : ""}" href="/sopal-v2/projects?archived=1" data-nav>Archived${archived.length ? ` · ${archived.length}` : ""}</a>
          </div>` : ""}
        ${categoriesPresent.length > 1 ? `
          <div class="tag-filter-row">
            <a class="tag-filter ${!categoryFilter ? "active" : ""}" href="/sopal-v2/projects${showArchived ? "?archived=1" : ""}" data-nav>All categories</a>
            ${categoriesPresent.map((c) => `<a class="tag-filter ${categoryFilter === c ? "active" : ""}" href="/sopal-v2/projects?${showArchived ? "archived=1&" : ""}category=${encodeURIComponent(c)}" data-nav>${escapeHtml(c)}</a>`).join("")}
          </div>` : ""}
        ${projects.length > 0 ? `
          <div class="bulk-toolbar ${selectedCount ? "active" : ""}">
            <label class="bulk-select-all"><input type="checkbox" data-select-all-projects ${allSelected ? "checked" : ""}><span>${selectedCount ? `${selectedCount} selected` : `Select all`}</span></label>
            ${selectedCount ? `
              ${showArchived
                ? `<button class="ghost-button compact" type="button" data-bulk-restore>${ICON.arrowUpRight}<span>Restore</span></button>`
                : `<button class="ghost-button compact" type="button" data-bulk-archive>${ICON.folder}<span>Archive</span></button>`}
              <button class="ghost-button compact danger" type="button" data-bulk-delete>${ICON.trash}<span>Delete</span></button>
              <button class="ghost-button compact" type="button" data-bulk-clear>Clear</button>
            ` : ""}
          </div>` : ""}
        ${projects.length === 0 ? (showArchived ? `
          <div class="card-empty"><div class="card-empty-icon">${ICON.folder}</div><h4>No archived projects.</h4><p>Archive a project from its overview page to tuck it out of sight without deleting.</p></div>
        ` : `
          <div class="card-empty">
            <div class="card-empty-icon">${ICON.file}</div>
            <h4>${categoryFilter ? `No ${categoryFilter} projects` : "Create your first project"}</h4>
            <p>${categoryFilter ? "Try a different category." : "Give it a name, the parties, the contract form. Then upload or paste the contract — the assistant and every agent will work in that project's context."}</p>
            <div class="card-empty-actions">
              <button class="dark-button" type="button" data-new-project>Create project</button>
              <label class="ghost-button" title="Import a sopal-*.json export">${ICON.upload}<span>Import from JSON</span><input type="file" data-import-project accept="application/json,.json" hidden></label>
            </div>
          </div>`) : `<div class="project-list">${projects.map((p) => projectRow(p, { archived: showArchived })).join("")}</div>`}
        <footer class="storage-footer">
          <div class="storage-bar"><div class="storage-bar-fill ${pct >= 80 ? "high" : ""}" style="width:${pct}%"></div></div>
          <p class="muted">${formatBytes(bytes)} of ~${formatBytes(quotaApprox)} local browser storage used (${pct}%). Sopal v2 stores all project data in this browser only — no server account.</p>
        </footer>
      </div>
    `);
  }

  function ProjectOverviewPage(projectId) {
    const project = getProject(projectId);
    if (!project) return notFoundPage();
    const allChats = Object.entries(project.chats || {})
      .filter(([, c]) => Array.isArray(c.messages) && c.messages.length > 0);
    const recentChats = [
      ...allChats.filter(([, c]) => c.pinned).sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0)),
      ...allChats.filter(([, c]) => !c.pinned).sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0)),
    ].slice(0, 5);
    setTimeout(() => bindProjectActions(projectId), 0);
    return PageBody(`
      <div class="page-shell">
        <div class="page-head">
          <div><h1 class="page-title">${escapeHtml(project.name)}</h1><p class="page-sub">${escapeHtml([project.reference, project.contractForm, [project.claimant, project.respondent].filter(Boolean).join(" v ")].filter(Boolean).join(" · ") || "Bespoke contract")}</p></div>
          <div class="page-actions">
            <button class="ghost-button compact" type="button" data-edit-project>Edit details</button>
            <button class="ghost-button compact" type="button" data-duplicate-project="${attr(project.id)}" title="Clone this project's contract + library into a new project (no chats / reviews)">${ICON.copy}<span>Duplicate</span></button>
            <button class="ghost-button compact" type="button" data-export-project="${attr(project.id)}" title="Download a JSON snapshot of this project">${ICON.download || ICON.file}<span>Export</span></button>
            ${project.archived
              ? `<button class="ghost-button compact" type="button" data-restore-project="${attr(project.id)}">${ICON.arrowUpRight}<span>Restore</span></button>`
              : `<button class="ghost-button compact" type="button" data-archive-project="${attr(project.id)}" title="Hide this project from your active list">${ICON.folder}<span>Archive</span></button>`}
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
              ${(() => {
                const pinned = [
                  ...(project.contracts || []).map((d, i) => ({ ...d, bucket: "contracts", _i: i })).filter((d) => d.pinned),
                  ...(project.library || []).map((d, i) => ({ ...d, bucket: "library", _i: i })).filter((d) => d.pinned),
                ];
                return pinned.length ? `
                  <div class="pinned-row">
                    <div class="pinned-row-title">${ICON.layers}<span>Pinned (always in chat context)</span></div>
                    <ul class="doc-list">${pinned.map((d) => `<li><a href="/sopal-v2/projects/${attr(project.id)}/${d.bucket === "contracts" ? "contract" : "library"}" data-doc-preview="${attr(project.id)}:${d.bucket}:${d._i}"><span class="doc-list-name">${escapeHtml(d.name || "Untitled")}</span><span class="doc-list-meta">${d.bucket === "contracts" ? "Contract" : "Library"} · ${escapeHtml(formatDocMeta(d))}</span></a></li>`).join("")}</ul>
                  </div>` : "";
              })()}
              ${project.contracts.length || project.library.length ? `
                <div class="doc-list-grid">
                  ${project.contracts.length ? `
                    <div class="doc-list-col">
                      <div class="doc-list-title">Contract</div>
                      <ul class="doc-list">
                        ${docListEntries(project.contracts).map((d) => `<li><a href="/sopal-v2/projects/${attr(project.id)}/contract" data-doc-preview="${attr(project.id)}:contracts:${d._i}"><span class="doc-list-name">${escapeHtml(d.name || "Untitled")}${d.pinned ? ` <span class="pin-badge">PINNED</span>` : ""}</span><span class="doc-list-meta">${escapeHtml(formatDocMeta(d))}</span></a></li>`).join("")}
                      </ul>
                    </div>` : ""}
                  ${project.library.length ? `
                    <div class="doc-list-col">
                      <div class="doc-list-title">Library</div>
                      <ul class="doc-list">
                        ${docListEntries(project.library).map((d) => `<li><a href="/sopal-v2/projects/${attr(project.id)}/library" data-doc-preview="${attr(project.id)}:library:${d._i}"><span class="doc-list-name">${escapeHtml(d.name || "Untitled")}${d.pinned ? ` <span class="pin-badge">PINNED</span>` : ""}</span><span class="doc-list-meta">${escapeHtml(formatDocMeta(d))}</span></a></li>`).join("")}
                      </ul>
                    </div>` : ""}
                </div>
              ` : ""}
              <div class="quick-link-row">
                <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/contract" data-nav>Open contract</a>
                <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/library" data-nav>Open library</a>
                <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/assistant" data-nav>Open assistant</a>
              </div>
            </div>
          </section>
          <section class="card span-all project-notes-card">
            <div class="card-head">
              <h3>Notes</h3>
              <span class="muted notes-status" data-notes-status></span>
            </div>
            <div class="card-body">
              <textarea class="text-area notes-textarea" data-notes-input rows="5" placeholder="Free-form scratchpad — chronology, key dates, open questions. Saved automatically.">${escapeHtml(project.notes || "")}</textarea>
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

  function docListEntries(docs) {
    return [...(docs || [])]
      .map((d, i) => ({ ...d, _i: i }))
      .sort((a, b) => {
        const ta = Date.parse(a.addedAt || "") || a._i;
        const tb = Date.parse(b.addedAt || "") || b._i;
        return tb - ta;
      })
      .slice(0, 3);
  }

  function formatDocMeta(d) {
    const len = (d.text || "").length;
    const date = d.addedAt ? new Date(d.addedAt) : null;
    const parts = [];
    if (date && !isNaN(date)) parts.push(date.toLocaleDateString(undefined, { month: "short", day: "numeric" }));
    if (len) parts.push(`${len.toLocaleString()} chars`);
    return parts.join(" · ");
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

  function togglePinThread(projectId, chatKey) {
    const project = getProject(projectId);
    if (!project || !project.chats || !project.chats[chatKey]) return;
    const chat = project.chats[chatKey];
    chat.pinned = !chat.pinned;
    saveProject(project);
    render();
  }

  function describeChatKey(project, key) {
    if (key === "assistant") return { label: "Assistant", href: `/sopal-v2/projects/${project.id}/assistant` };
    if (key.startsWith("chat:review:")) {
      const [, , agentKey, submodeId] = key.split(":");
      const submode = (AGENT_REVIEW_MODES[agentKey] || []).find((m) => m.id === submodeId);
      const submodeLabel = submode ? submode.label.toLowerCase() : submodeId;
      return {
        label: `${AGENT_LABELS[agentKey] || agentKey} · ${submodeLabel}`,
        href: `/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=review&submode=${submodeId}`,
      };
    }
    if (key.startsWith("agent:")) {
      const [, agentKey, mode] = key.split(":");
      return {
        label: `${AGENT_LABELS[agentKey] || agentKey} · ${mode}`,
        href: `/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=${mode}`,
      };
    }
    return { label: key, href: `/sopal-v2/projects/${project.id}/assistant` };
  }

  function sidebarRecentThreads(project) {
    const all = Object.entries(project.chats || {})
      .filter(([, c]) => Array.isArray(c.messages) && c.messages.length > 0);
    if (!all.length) return "";
    const pinned = all.filter(([, c]) => c.pinned).sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0));
    const recent = all.filter(([, c]) => !c.pinned).sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0)).slice(0, 3);
    const renderRow = ([key, h], isPinned) => {
      const { label, href } = describeChatKey(project, key);
      const last = h.messages[h.messages.length - 1] || {};
      const preview = plainPreview(last.content || "");
      return `<a class="nav-thread ${isPinned ? "is-pinned" : ""}" href="${href}" data-nav title="${attr(preview)}">
        <span class="nav-thread-row">
          <span class="nav-thread-label">${escapeHtml(label)}</span>
          <button class="nav-thread-pin" type="button" data-toggle-pin-thread="${attr(key)}" data-project-id="${attr(project.id)}" title="${isPinned ? "Unpin thread" : "Pin thread"}">${isPinned ? "★" : "☆"}</button>
        </span>
        <span class="nav-thread-preview">${escapeHtml(preview.slice(0, 64))}${preview.length > 64 ? "…" : ""}</span>
      </a>`;
    };
    return `
      ${pinned.length ? `<div class="nav-subgroup-title">Pinned threads</div>${pinned.map((t) => renderRow(t, true)).join("")}` : ""}
      ${recent.length ? `<div class="nav-subgroup-title">Recent threads</div>${recent.map((t) => renderRow(t, false)).join("")}` : ""}
    `;
  }

  function recentChatRow(project, key, h) {
    const { label, href } = describeChatKey(project, key);
    const last = h.messages[h.messages.length - 1] || {};
    const preview = plainPreview(last.content || "");
    return `<a class="recent-item ${h.pinned ? "is-pinned" : ""}" href="${href}" data-nav>
      <div class="recent-item-row">
        <strong>${escapeHtml(label)}</strong>
        <button class="recent-pin-btn" type="button" data-toggle-pin-thread="${attr(key)}" data-project-id="${attr(project.id)}" title="${h.pinned ? "Unpin thread" : "Pin thread"}">${h.pinned ? "★" : "☆"}</button>
      </div>
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
    document.querySelectorAll("[data-export-project]").forEach((b) => b.addEventListener("click", () => {
      exportProject(b.dataset.exportProject);
    }));
    document.querySelectorAll("[data-duplicate-project]").forEach((b) => b.addEventListener("click", () => {
      const cloned = duplicateProject(b.dataset.duplicateProject);
      if (cloned) navigate(`/sopal-v2/projects/${cloned.id}/overview`);
    }));
    document.querySelectorAll("[data-archive-project]").forEach((b) => b.addEventListener("click", () => {
      archiveProject(b.dataset.archiveProject);
      navigate("/sopal-v2/projects");
    }));
    document.querySelectorAll("[data-restore-project]").forEach((b) => b.addEventListener("click", () => {
      restoreProject(b.dataset.restoreProject);
      render();
    }));
    const notesInput = document.querySelector("[data-notes-input]");
    if (notesInput) {
      const status = document.querySelector("[data-notes-status]");
      let notesTimer = null;
      notesInput.addEventListener("input", () => {
        if (status) status.textContent = "Saving…";
        if (notesTimer) clearTimeout(notesTimer);
        notesTimer = setTimeout(() => {
          const proj = getProject(projectId);
          if (!proj) return;
          proj.notes = notesInput.value;
          saveProject(proj);
          if (status) {
            status.textContent = "Saved";
            setTimeout(() => { if (status) status.textContent = ""; }, 1500);
          }
        }, 400);
      });
    }
  }

  function duplicateProject(sourceId) {
    const source = getProject(sourceId);
    if (!source) return null;
    const id = newProjectId();
    const now = Date.now();
    const cloned = {
      id,
      name: `${source.name} (copy)`,
      claimant: source.claimant || "",
      respondent: source.respondent || "",
      contractForm: source.contractForm || "Bespoke",
      reference: source.reference || "",
      userIsParty: source.userIsParty || "claimant",
      contracts: (source.contracts || []).map((d) => ({ ...d, addedAt: new Date().toISOString() })),
      library: (source.library || []).map((d) => ({ ...d, addedAt: new Date().toISOString() })),
      chats: {},
      reviews: {},
      createdAt: now,
      updatedAt: now,
    };
    store.projects[id] = cloned;
    store.currentProjectId = id;
    saveStore();
    return cloned;
  }

  function exportProject(projectId) {
    const project = getProject(projectId);
    if (!project) return;
    const payload = {
      sopalVersion: "v2",
      exportedAt: new Date().toISOString(),
      project,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const safeName = (project.name || "project").replace(/[^a-z0-9-]+/gi, "_").slice(0, 60);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sopal-${safeName}-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }

  function notFoundPage() {
    return PageBody(`<div class="page-shell">${EmptyState("Project not found.", "It may have been deleted. Return to the project list.", `<a class="ghost-button compact" href="/sopal-v2/projects" data-nav>Open projects</a>`)}</div>`);
  }

  function ContextPage(projectId, bucket) {
    const project = getProject(projectId);
    if (!project) return notFoundPage();
    const allItems = project[bucket] || [];
    const params = new URLSearchParams(window.location.search);
    const tagFilter = params.get("tag") || "";
    const items = tagFilter ? allItems.filter((it) => (it.tags || []).includes(tagFilter)) : allItems;
    const allTags = Array.from(new Set(allItems.flatMap((it) => it.tags || []))).sort();
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
              <label class="span-2">Label<input class="text-input" name="name" placeholder="Document label"></label>
              <label class="span-2">Paste text<textarea class="text-area" name="text" rows="8" placeholder="Paste clauses, correspondence, claim text, schedule text, or facts."></textarea></label>
              <div class="file-zone span-2" data-bulk-drop>
                <label class="file-zone-label">${ICON.upload}<span>Click or drop one or more PDF / DOCX / TXT files — each becomes a separate entry</span><input type="file" data-context-file accept=".pdf,.docx,.txt" multiple></label>
                <div class="muted file-status" data-context-file-status>No files selected.</div>
              </div>
              <div class="span-2 split-action" data-split-action hidden>
                <button class="ghost-button compact" type="button" data-split-detect>${ICON.layers}<span>Detect clauses</span></button>
                <span class="muted split-status" data-split-status></span>
              </div>
              <button class="dark-button span-2" type="submit">${ICON.plus}<span>Add to project</span></button>
            </form>
          </div>
          <div class="card">
            <div class="card-head"><div><h3>Saved (${items.length}${tagFilter ? ` of ${allItems.length}` : ""})</h3></div>${allItems.length ? `<button class="ghost-button compact danger" type="button" data-clear-context="${bucket}">Clear all</button>` : ""}</div>
            ${allTags.length ? `
              <div class="tag-filter-row">
                <a class="tag-filter ${!tagFilter ? "active" : ""}" href="/sopal-v2/projects/${attr(projectId)}/${bucket === "contracts" ? "contract" : "library"}" data-nav>All</a>
                ${allTags.map((t) => `<a class="tag-filter ${tagFilter === t ? "active" : ""}" href="/sopal-v2/projects/${attr(projectId)}/${bucket === "contracts" ? "contract" : "library"}?tag=${encodeURIComponent(t)}" data-nav>${escapeHtml(t)}</a>`).join("")}
              </div>` : ""}
            <div class="card-body context-list">
              ${items.length === 0 ? EmptyState(tagFilter ? `No ${labels.single.toLowerCase()} with tag '${tagFilter}'.` : `No ${labels.single.toLowerCase()} yet.`, tagFilter ? "Try a different tag or clear the filter." : "Add pasted or extracted text to make agents context-aware.") : items.map((item) => {
                const i = allItems.indexOf(item);
                return `
                <article class="context-item ${item.pinned ? "is-pinned" : ""}">
                  <div class="context-item-head">
                    <button class="context-item-title-btn" type="button" data-doc-preview="${attr(projectId)}:${bucket}:${i}" title="Open document preview">
                      <strong>${escapeHtml(item.name)}</strong>
                    </button>
                    <span class="muted">${item.text.length.toLocaleString()} chars · ${escapeHtml(item.source || "pasted")}${item.pinned ? ` · <span class="pin-badge">PINNED</span>` : ""}</span>
                  </div>
                  ${(item.tags || []).length ? `<div class="doc-tag-row">${item.tags.map((t) => `<span class="doc-tag">${escapeHtml(t)}</span>`).join("")}</div>` : ""}
                  <details><summary>Preview</summary><pre>${escapeHtml(item.text.slice(0, 2000))}${item.text.length > 2000 ? "\n…" : ""}</pre></details>
                  <div class="context-item-actions">
                    <button class="ghost-button compact" type="button" data-doc-preview="${attr(projectId)}:${bucket}:${i}">${ICON.arrowUpRight}<span>Open / edit</span></button>
                    <button class="ghost-button compact" type="button" data-copy-text="${attr(item.text.slice(0, 8000))}">${ICON.copy}<span>Copy</span></button>
                    <button class="ghost-button compact danger" type="button" data-remove-context="${attr(bucket)}:${i}">Remove</button>
                  </div>
                </article>`;
              }).join("")}
            </div>
          </div>
        </div>
      </div>
    `);
  }

  function detectClauseSections(text) {
    if (!text) return [];
    // Match common contract clause / section / part headers at the start of a line:
    //   "Clause 41.2 Heading", "Section 12 Heading", "12. Heading"
    const re = /(?:^|\n)\s*((?:Clause|Cl\.?|Section|Sec\.?|Part)\s+\d+(?:\.\d+)*[A-Za-z]?\b[^\n]{0,120})/gi;
    const points = [];
    let m;
    while ((m = re.exec(text)) !== null) {
      points.push({ index: m.index + m[0].indexOf(m[1]), header: m[1].trim() });
    }
    if (points.length < 2) return [];
    const sections = [];
    for (let i = 0; i < points.length; i++) {
      const start = points[i].index;
      const end = i + 1 < points.length ? points[i + 1].index : text.length;
      const body = text.slice(start, end).trim();
      const header = points[i].header.replace(/\s+/g, " ").slice(0, 100);
      sections.push({ name: header, text: body });
    }
    return sections;
  }

  function bindContextManager(projectId, bucket) {
    const form = document.querySelector(`[data-context-form="${bucket}"]`);
    if (!form) return;
    const fileInput = form.querySelector("[data-context-file]");
    const status = form.querySelector("[data-context-file-status]");
    const textarea = form.querySelector("textarea[name=text]");
    const splitAction = form.querySelector("[data-split-action]");
    const splitStatus = form.querySelector("[data-split-status]");
    let extracted = null;
    let detected = null;

    const updateSplitVisibility = () => {
      if (!splitAction) return;
      const text = textarea?.value || "";
      const candidates = detectClauseSections(text);
      const enable = candidates.length >= 2;
      splitAction.toggleAttribute("hidden", !enable);
      detected = enable ? candidates : null;
      if (splitStatus) splitStatus.textContent = enable ? `${candidates.length} clause sections detected` : "";
      const splitBtn = form.querySelector("[data-split-detect]");
      if (splitBtn) {
        splitBtn.innerHTML = `${ICON.layers}<span>Split into ${candidates.length} items</span>`;
      }
    };

    textarea?.addEventListener("input", updateSplitVisibility);

    form.querySelector("[data-split-detect]")?.addEventListener("click", () => {
      if (!detected || !detected.length) return;
      const project = getProject(projectId);
      if (!project) return;
      const stamp = new Date().toISOString();
      detected.forEach((s) => {
        project[bucket].push({
          name: s.name,
          text: s.text,
          source: extracted ? "extracted file + split" : "split",
          addedAt: stamp,
        });
      });
      saveProject(project);
      render();
    });

    async function processFiles(files) {
      const list = Array.from(files || []);
      if (!list.length) return;
      // Single file with no pasted text → fall through into the existing single-file
      // flow so the user can review/edit before saving (preserves the existing UX).
      if (list.length === 1 && !textarea.value.trim()) {
        const file = list[0];
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
          updateSplitVisibility();
        } catch (error) {
          status.textContent = error.message || "Extraction failed";
        }
        return;
      }
      // Multi-file: extract each in parallel, then save all as separate entries
      // bypassing the form's text/name fields entirely.
      status.textContent = `Extracting ${list.length} files…`;
      const results = await Promise.all(list.map(async (file) => {
        const fd = new FormData(); fd.append("file", file);
        try {
          const response = await fetch("/api/sopal-v2/extract", { method: "POST", body: fd, credentials: "include" });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.detail || "Extraction failed");
          return { ok: true, data, file };
        } catch (error) {
          return { ok: false, file, message: error.message || "Extraction failed" };
        }
      }));
      const project = getProject(projectId);
      if (!project) return;
      const stamp = new Date().toISOString();
      const succeeded = results.filter((r) => r.ok);
      const failed = results.filter((r) => !r.ok);
      succeeded.forEach((r) => {
        project[bucket].push({
          name: r.data.filename || r.file.name || "Untitled",
          text: r.data.text || "",
          source: "extracted file (bulk)",
          addedAt: stamp,
        });
      });
      if (succeeded.length) saveProject(project);
      const summary = `${succeeded.length} added${failed.length ? ` · ${failed.length} failed (${failed.map((f) => f.file.name).join(", ")})` : ""}`;
      if (succeeded.length) {
        status.textContent = summary;
        render();
      } else {
        status.textContent = `All ${failed.length} extractions failed: ${failed.map((f) => `${f.file.name} (${f.message})`).join("; ")}`;
      }
    }

    fileInput?.addEventListener("change", () => processFiles(fileInput.files));

    const dropZone = form.querySelector("[data-bulk-drop]");
    if (dropZone) {
      ["dragenter", "dragover"].forEach((evt) => dropZone.addEventListener(evt, (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-over");
      }));
      ["dragleave", "drop"].forEach((evt) => dropZone.addEventListener(evt, (event) => {
        event.preventDefault();
        dropZone.classList.remove("drag-over");
      }));
      dropZone.addEventListener("drop", (event) => {
        const files = event.dataTransfer && event.dataTransfer.files;
        if (files && files.length) processFiles(files);
      });
    }
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
    updateSplitVisibility();
  }

  /* ---------- Project Assistant + Agents (Astruct-inspired chat) ---------- */

  function projectContextString(project, opts) {
    const pinnedOnly = !!(opts && opts.pinnedOnly);
    const contracts = pinnedOnly ? (project.contracts || []).filter((d) => d.pinned) : (project.contracts || []);
    const library = pinnedOnly ? (project.library || []).filter((d) => d.pinned) : (project.library || []);
    const contractText = contracts.map((d) => `Contract: ${d.name}${d.pinned ? " (pinned)" : ""}\n${d.text}`).join("\n\n---\n\n");
    const libraryText = library.map((d) => `Project document: ${d.name}${d.pinned ? " (pinned)" : ""}\n${d.text}`).join("\n\n---\n\n");
    return [contractText, libraryText].filter(Boolean).join("\n\n===\n\n").slice(0, 40000);
  }
  function projectHasPinnedDocs(project) {
    return (project.contracts || []).some((d) => d.pinned) || (project.library || []).some((d) => d.pinned);
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

  /* ---------- Complex Agent: Adjudication Application ----------
     Multi-stage workflow:
       intake → dispute-table → rfi → draft → review
     See docs/complex-adjudication-application-plan.md for the full plan.
     v1 ships the end-to-end skeleton: paste PC+PS, parse, lock the
     dispute table, work each dispute via AI-driven RFIs, watch the
     master document assemble live, export to .doc. Phase B layers in
     numbering, ToC, exec summary, definitions panel, deadline timer. */

  const AA_STAGES = [
    { id: "intake",        label: "Intake" },
    { id: "dispute-table", label: "Dispute Table" },
    { id: "rfi",           label: "RFI" },
    { id: "draft",         label: "Draft" },
    { id: "review",        label: "Review" },
  ];

  // Section 79 BIF Act sets out three timing scenarios for an adjudication
  // application. The picker drives downstream behaviour (PS optional,
  // jurisdictional submissions adjusted, deadline arithmetic, master doc
  // copy). Definitions taken straight from the Act.
  const AA_S79_SCENARIOS = [
    {
      id: "no-schedule",
      label: "No payment schedule received and no payment made",
      sub: "s 79(2)(a) — 30 BD after the LATER of (i) day amount became payable; or (ii) last day a schedule could have been given (15 BD after PC).",
      psOptional: true,
    },
    {
      id: "less-than-claimed",
      label: "Schedule received — scheduled amount LESS than claimed",
      sub: "s 79(2)(b) — 30 BD after receipt of the payment schedule.",
      psOptional: false,
    },
    {
      id: "scheduled-but-unpaid",
      label: "Schedule received — scheduled amount EQUAL to claim, but not paid",
      sub: "s 79(2)(c) — 20 BD after the day on which payment is due under the contract.",
      psOptional: false,
    },
  ];

  const AA_ISSUE_TYPES = [
    "variation", "eot", "delay-costs", "defects",
    "set-off", "retention", "prevention", "scope",
    "valuation", "other",
  ];
  const AA_ISSUE_TYPE_LABELS = {
    variation: "Variation",
    eot: "EOT",
    "delay-costs": "Delay costs",
    defects: "Defects / set-off",
    "set-off": "Set-off",
    retention: "Retention",
    prevention: "Prevention principle",
    scope: "Scope",
    valuation: "Valuation",
    other: "Other",
  };
  const AA_DISPUTE_STATUSES = ["disputed", "admitted", "partial", "jurisdictional"];

  function getComplexAA(project) {
    if (!project.complexApps) project.complexApps = {};
    if (!project.complexApps["adjudication-application"]) {
      project.complexApps["adjudication-application"] = {
        stage: "intake",
        deadline: null,
        s79Scenario: "less-than-claimed",
        documents: { paymentClaim: null, paymentSchedule: null },
        psReasonsUniverse: "",
        parties: { claimant: "", respondent: "" },
        contractReference: "",
        referenceDate: "",
        claimedAmount: 0,
        scheduledAmount: 0,
        disputes: [],
        jurisdictionalRfis: { rounds: [], submissions: "", evidenceIndex: [], statDecContent: "" },
        generalRfis: { rounds: [], submissions: "", evidenceIndex: [], statDecContent: "" },
        definitions: {},
        activeKey: null,
        updatedAt: Date.now(),
      };
      saveProject(project);
    }
    // Backfill: older state may not have s79Scenario.
    const aa = project.complexApps["adjudication-application"];
    if (!aa.s79Scenario) {
      aa.s79Scenario = "less-than-claimed";
    }
    // Migrate any per-dispute artefact fields that were stored at the top
    // level of the dispute (older shape) into the canonical d.rfis location
    // (where the engine writes and where the items nav reads). This makes
    // master / stat-dec / evidence-index reads consistent.
    let migrated = false;
    (aa.disputes || []).forEach((d) => {
      if (!d.rfis || !Array.isArray(d.rfis.rounds)) {
        d.rfis = { rounds: [], submissions: "", evidenceIndex: [], statDecContent: "", isReady: false };
        migrated = true;
      }
      ["submissions", "evidenceIndex", "statDecContent", "isReady"].forEach((k) => {
        if (d[k] !== undefined && (d.rfis[k] === undefined || d.rfis[k] === "" || (Array.isArray(d.rfis[k]) && d.rfis[k].length === 0))) {
          d.rfis[k] = d[k];
          migrated = true;
        }
        if (d[k] !== undefined) {
          delete d[k];
          migrated = true;
        }
      });
    });
    if (migrated) saveProject(project);
    return aa;
  }

  function newDisputeId() { return `d_${Math.random().toString(36).slice(2, 8)}`; }

  function ComplexAdjudicationPage(projectId) {
    const project = getProject(projectId);
    if (!project) return notFoundPage();
    const aa = getComplexAA(project);
    setTimeout(() => bindComplexAA(project), 0);

    const stageIdx = AA_STAGES.findIndex((s) => s.id === aa.stage);
    // Stages are click-to-jump. The user can preview any stage at any time;
    // earlier "lock" semantics removed. Stages are visually marked as done
    // (passed) / active / available so the path is still suggested.
    const stageBar = `
      <div class="aa-stage-bar" role="tablist">
        ${AA_STAGES.map((s, i) => `
          <button class="aa-stage ${i < stageIdx ? "done" : ""} ${i === stageIdx ? "active" : ""}" type="button" data-aa-jump="${attr(s.id)}" aria-selected="${i === stageIdx ? "true" : "false"}">
            <span class="aa-stage-num">${i + 1}</span>
            <span class="aa-stage-label">${escapeHtml(s.label)}</span>
          </button>
        `).join("")}
      </div>`;

    let body = "";
    if (aa.stage === "intake") body = renderAAIntake(aa);
    else if (aa.stage === "dispute-table") body = renderAADisputeTable(aa);
    else if (aa.stage === "review") body = renderAAReview(project, aa);
    else body = renderAAWorkspace(project, aa);

    const deadlineMeta = aaDeadlineMeta(aa.deadline);
    const warnings = (aa.parseWarnings || []);
    // Progress = drafted threads / total threads. Threads are the two shared
    // (jurisdictional + general) plus one per dispute. Surfaced in the header
    // so the user can see at a glance how close the application is to ready.
    const allThreads = [aa.jurisdictionalRfis, aa.generalRfis].concat((aa.disputes || []).map((d) => d.rfis || { submissions: "" }));
    const draftedThreads = allThreads.filter((t) => (t.submissions || "").length > 60).length;
    const totalThreads = allThreads.length;
    const progressPct = totalThreads ? Math.round((draftedThreads / totalThreads) * 100) : 0;
    return PageBody(`
      <div class="page-shell aa-shell">
        <div class="chat-page-head">
          <div>
            <h1 class="page-title">Adjudication Application</h1>
            <p class="page-sub">Guided drafter — intake, dispute mapping, RFIs per item, live master document.</p>
          </div>
          <div class="aa-header-actions">
            ${aa.stage !== "intake" ? `<span class="aa-progress-pill" title="Drafted threads">${draftedThreads}/${totalThreads} drafted · ${progressPct}%</span>` : ""}
            ${aa.stage !== "intake" ? `<button class="ghost-button compact" type="button" data-aa-back-stage>← Back a stage</button>` : ""}
            ${deadlineMeta ? `<span class="aa-deadline-pill ${deadlineMeta.cls}" title="Lodgement deadline">${deadlineMeta.label}</span>` : ""}
            <button class="ghost-button compact danger" type="button" data-aa-reset>Reset</button>
          </div>
        </div>
        ${stageBar}
        ${warnings.length ? `<div class="aa-warnings">${warnings.map((w) => `<div class="aa-warning"><strong>${escapeHtml(w.code || "warning")}</strong> ${escapeHtml(w.message || "")}</div>`).join("")}</div>` : ""}
        ${body}
      </div>
    `);
  }

  function aaDeadlineMeta(iso) {
    if (!iso) return null;
    const d = new Date(`${iso}T00:00:00`);
    if (isNaN(d.getTime())) return null;
    const now = new Date();
    now.setHours(0, 0, 0, 0);
    const diffDays = Math.round((d - now) / 86400000);
    let cls = "ok";
    let label = "";
    if (diffDays < 0) { cls = "overdue"; label = `Overdue by ${Math.abs(diffDays)} day${Math.abs(diffDays) === 1 ? "" : "s"}`; }
    else if (diffDays === 0) { cls = "urgent"; label = "Lodge TODAY"; }
    else if (diffDays <= 5) { cls = "urgent"; label = `Lodge in ${diffDays} day${diffDays === 1 ? "" : "s"}`; }
    else if (diffDays <= 14) { cls = "soon"; label = `Lodge in ${diffDays} days`; }
    else { cls = "ok"; label = `Lodge in ${diffDays} days`; }
    const formatted = d.toLocaleDateString("en-AU", { weekday: "short", day: "numeric", month: "short", year: "numeric" });
    return { cls, label: `${label} · ${formatted}` };
  }

  function renderAAIntake(aa) {
    const pc = aa.documents.paymentClaim;
    const ps = aa.documents.paymentSchedule;
    const scenarioId = aa.s79Scenario || "less-than-claimed";
    const scenario = AA_S79_SCENARIOS.find((s) => s.id === scenarioId) || AA_S79_SCENARIOS[0];
    return `
      <section class="aa-intake card">
        <div class="card-head">
          <div>
            <h3>Stage 1 — Document intake</h3>
            <p class="muted">Paste or upload the Payment Claim (and the Payment Schedule, where one was given). Sopal extracts the parties, amounts, line items, and the respondent's reasons.</p>
          </div>
        </div>
        <div class="card-body">
          <div class="aa-s79-picker">
            <label class="aa-doc-label">s 79 BIF Act scenario</label>
            <p class="muted aa-s79-help">Pick the scenario that applies. This drives whether a payment schedule is required, the s 79 deadline calculation, and the jurisdictional submissions.</p>
            <div class="aa-s79-options">
              ${AA_S79_SCENARIOS.map((s) => `
                <label class="aa-s79-option ${scenarioId === s.id ? "active" : ""}">
                  <input type="radio" name="aa-s79" value="${attr(s.id)}" ${scenarioId === s.id ? "checked" : ""}>
                  <span class="aa-s79-body">
                    <strong>${escapeHtml(s.label)}</strong>
                    <span class="muted">${escapeHtml(s.sub)}</span>
                  </span>
                </label>
              `).join("")}
            </div>
          </div>

          <div class="aa-intake-grid">
            <div class="aa-doc-slot">
              <label class="aa-doc-label">Payment Claim</label>
              <div class="file-zone">
                <label class="file-zone-label">${ICON.upload}<span>Click or drop a PDF / DOCX / TXT</span><input type="file" data-aa-file="paymentClaim" accept=".pdf,.docx,.txt"></label>
                <div class="muted file-status" data-aa-file-status-paymentClaim>${pc ? `${escapeHtml(pc.name)} · ${pc.text.length.toLocaleString()} chars` : "No file selected."}</div>
              </div>
              <textarea class="text-area" data-aa-text="paymentClaim" rows="8" placeholder="Or paste the payment claim text here…">${escapeHtml(pc ? pc.text : "")}</textarea>
            </div>
            <div class="aa-doc-slot ${scenario.psOptional ? "aa-doc-slot-optional" : ""}">
              <label class="aa-doc-label">Payment Schedule${scenario.psOptional ? " (none received — optional)" : ""}</label>
              ${scenario.psOptional ? `<p class="muted aa-doc-help">No PS was given by the respondent in the s 76 window. Leave this blank — Sopal will frame the application accordingly.</p>` : ""}
              <div class="file-zone">
                <label class="file-zone-label">${ICON.upload}<span>Click or drop a PDF / DOCX / TXT</span><input type="file" data-aa-file="paymentSchedule" accept=".pdf,.docx,.txt"></label>
                <div class="muted file-status" data-aa-file-status-paymentSchedule>${ps ? `${escapeHtml(ps.name)} · ${ps.text.length.toLocaleString()} chars` : "No file selected."}</div>
              </div>
              <textarea class="text-area" data-aa-text="paymentSchedule" rows="8" placeholder="${scenario.psOptional ? "(Leave blank if no PS was received.)" : "Or paste the payment schedule text here…"}">${escapeHtml(ps ? ps.text : "")}</textarea>
            </div>
            <div class="aa-intake-meta">
              <label>Lodgement deadline (optional)
                <input class="text-input" type="date" data-aa-deadline value="${attr(aa.deadline || "")}">
              </label>
            </div>
            <div class="aa-intake-actions">
              <button class="dark-button" type="button" data-aa-parse>${ICON.sparkles}<span>Parse documents</span></button>
              <span class="muted aa-intake-help">Parsing extracts the parties, amounts, claim line items, and (if a PS was given) the respondent's reasons. You'll review and edit the result on the next stage.</span>
            </div>
          </div>
        </div>
      </section>
    `;
  }

  function renderAADisputeTable(aa) {
    const noParse = !aa.parties.claimant && !aa.parties.respondent && aa.claimedAmount === 0 && aa.disputes.length === 0;
    if (noParse) {
      return `
        <section class="aa-dispute-table card">
          <div class="card-head">
            <div>
              <h3>Stage 2 — Dispute table</h3>
              <p class="muted">Once you parse the PC and PS in Stage 1, the line items appear here as an editable dispute table. You can also build it manually.</p>
            </div>
            <div class="aa-table-actions">
              <button class="ghost-button compact" type="button" data-aa-jump="intake">← Go to Intake</button>
              <button class="ghost-button compact" type="button" data-aa-add-row>${ICON.plus}<span>Add row manually</span></button>
            </div>
          </div>
          <div class="card-body">
            ${EmptyState("No items yet.", "Either parse the PC + PS in Stage 1, or click 'Add row manually' to build the dispute table by hand.")}
          </div>
        </section>
      `;
    }
    return `
      <section class="aa-dispute-table card">
        <div class="card-head">
          <div>
            <h3>Stage 2 — Dispute table</h3>
            <p class="muted">Edit the rows. Merge or split where the PC artificially divides one dispute. Set the issue type so the RFIs are tailored. Lock to advance.</p>
          </div>
          <div class="aa-table-actions">
            <button class="ghost-button compact" type="button" data-aa-add-row>${ICON.plus}<span>Add row</span></button>
            <button class="dark-button" type="button" data-aa-lock-table>Lock dispute table →</button>
          </div>
        </div>
        <div class="card-body">
          <div class="aa-extract-summary">
            <span><strong>Claimant:</strong> ${escapeHtml(aa.parties.claimant || "—")}</span>
            <span><strong>Respondent:</strong> ${escapeHtml(aa.parties.respondent || "—")}</span>
            <span><strong>Reference date:</strong> ${escapeHtml(aa.referenceDate || "—")}</span>
            <span><strong>Claimed:</strong> ${formatCurrencyCompact(aa.claimedAmount)}</span>
            <span><strong>Scheduled:</strong> ${formatCurrencyCompact(aa.scheduledAmount)}</span>
            <span><strong>s 79 scenario:</strong> ${escapeHtml((AA_S79_SCENARIOS.find((s) => s.id === (aa.s79Scenario || "less-than-claimed")) || AA_S79_SCENARIOS[0]).label)}</span>
          </div>
          <table class="aa-table">
            <thead>
              <tr>
                <th class="aa-col-item">Item</th>
                <th>Description</th>
                <th class="aa-col-money">Claimed</th>
                <th class="aa-col-money">Scheduled</th>
                <th class="aa-col-status">Status</th>
                <th class="aa-col-issue">Issue type</th>
                <th>Respondent's reasons (PS)</th>
                <th class="aa-col-actions"></th>
              </tr>
            </thead>
            <tbody data-aa-table-body>
              ${aa.disputes.map((d) => renderAARow(d)).join("")}
            </tbody>
          </table>
          ${aa.disputes.length === 0 ? EmptyState("No items extracted.", "Click 'Add row' to add a dispute manually.") : ""}
        </div>
      </section>
    `;
  }

  function renderAAReview(project, aa) {
    const navItems = [
      { key: "jurisdictional", label: "Jurisdictional", thread: aa.jurisdictionalRfis },
      { key: "general", label: "Background / General", thread: aa.generalRfis },
      ...aa.disputes.map((d) => ({ key: `dispute:${d.id}`, label: d.item || d.id, thread: d.rfis })),
    ];
    const drafted = navItems.filter((n) => n.thread && n.thread.submissions && n.thread.submissions.length > 60).length;
    const total = navItems.length;
    const evidence = [];
    aa.disputes.forEach((d) => ((d.rfis && d.rfis.evidenceIndex) || []).forEach((e) => evidence.push(e)));
    const scenarioId = aa.s79Scenario || "less-than-claimed";
    const scenario = AA_S79_SCENARIOS.find((s) => s.id === scenarioId) || AA_S79_SCENARIOS[0];
    return `
      <section class="aa-review">
        <div class="card aa-review-summary">
          <div class="card-head">
            <h3>Stage 5 — Final review &amp; lodgement</h3>
            <div class="aa-review-actions">
              <button class="dark-button" type="button" data-aa-export>${ICON.download}<span>Export master .doc</span></button>
              <button class="ghost-button compact" type="button" data-aa-export-statdecs>${ICON.download}<span>Export combined stat dec</span></button>
              <button class="ghost-button compact" type="button" data-aa-export-soe>${ICON.download}<span>Export evidence index</span></button>
              <button class="ghost-button compact" type="button" data-aa-print-master>${ICON.file}<span>Print master</span></button>
              <button class="ghost-button compact" type="button" data-aa-copy-master>${ICON.copy}<span>Copy as Markdown</span></button>
              <button class="ghost-button compact" type="button" data-aa-draft-all title="Run a draft pass for every thread that has answered RFIs but isn't drafted yet">${ICON.sparkles}<span>Draft all</span></button>
              <span class="aa-draft-all-status-inline" data-aa-draft-all-status hidden></span>
            </div>
          </div>
          <div class="card-body">
            <div class="aa-review-checklist">
              <h4>Lodgement checklist</h4>
              <ul>
                <li>${aa.documents.paymentClaim ? "✓" : "○"} Payment Claim ingested</li>
                <li>${scenario.psOptional ? "—" : (aa.documents.paymentSchedule ? "✓" : "○")} Payment Schedule ${scenario.psOptional ? "(not required for this scenario)" : "ingested"}</li>
                <li>${aa.disputes.length ? "✓" : "○"} Dispute table populated (${aa.disputes.length} item${aa.disputes.length === 1 ? "" : "s"})</li>
                <li>${(aa.jurisdictionalRfis.submissions || "").length > 60 ? "✓" : "○"} Jurisdictional submissions drafted</li>
                <li>${(aa.generalRfis.submissions || "").length > 60 ? "✓" : "○"} Background drafted</li>
                <li>${drafted}/${total} threads drafted overall</li>
                <li>${evidence.length ? "✓" : "○"} Index of supporting evidence (${evidence.length} item${evidence.length === 1 ? "" : "s"})</li>
                <li>${aa.deadline ? `✓ Lodgement deadline set (${escapeHtml(aa.deadline)})` : "○ Lodgement deadline not set"}</li>
              </ul>
            </div>
            <div class="aa-review-scenario">
              <h4>s 79 scenario</h4>
              <p><strong>${escapeHtml(scenario.label)}</strong><br>
              <span class="muted">${escapeHtml(scenario.sub)}</span></p>
            </div>
          </div>
        </div>
        <div class="card aa-review-master">
          <div class="card-head">
            <h3>Master document preview</h3>
            <button class="ghost-button compact" type="button" data-aa-rebuild>Rebuild</button>
          </div>
          <div class="aa-master-body" data-aa-master>${renderAAMaster(project, aa)}</div>
        </div>
      </section>
    `;
  }

  function renderAARow(d) {
    return `
      <tr data-aa-row="${attr(d.id)}">
        <td><input class="aa-cell" data-aa-cell="item" value="${attr(d.item || "")}"></td>
        <td><textarea class="aa-cell aa-cell-multi" data-aa-cell="description" rows="2">${escapeHtml(d.description || "")}</textarea></td>
        <td><input class="aa-cell aa-cell-num" type="number" data-aa-cell="claimed" value="${attr(d.claimed || 0)}"></td>
        <td><input class="aa-cell aa-cell-num" type="number" data-aa-cell="scheduled" value="${attr(d.scheduled || 0)}"></td>
        <td>
          <select class="aa-cell" data-aa-cell="status">
            ${AA_DISPUTE_STATUSES.map((s) => `<option value="${s}" ${d.status === s ? "selected" : ""}>${s}</option>`).join("")}
          </select>
        </td>
        <td>
          <select class="aa-cell" data-aa-cell="issueType">
            ${AA_ISSUE_TYPES.map((t) => `<option value="${t}" ${d.issueType === t ? "selected" : ""}>${escapeHtml(AA_ISSUE_TYPE_LABELS[t])}</option>`).join("")}
          </select>
        </td>
        <td><textarea class="aa-cell aa-cell-multi" data-aa-cell="psReasons" rows="2">${escapeHtml(d.psReasons || "")}</textarea></td>
        <td><button class="ghost-button compact danger" type="button" data-aa-delete-row title="Delete row">${ICON.trash}</button></td>
      </tr>
    `;
  }

  function renderAAWorkspace(project, aa) {
    const navItems = [
      { key: "jurisdictional", label: "Jurisdictional", thread: aa.jurisdictionalRfis, kind: "shared" },
      { key: "general", label: "Background / General", thread: aa.generalRfis, kind: "shared" },
      ...aa.disputes.map((d) => ({ key: `dispute:${d.id}`, label: d.item || d.id, thread: d.rfis, kind: "dispute", dispute: d })),
    ];
    const activeKey = aa.activeKey || navItems[0].key;
    const active = navItems.find((n) => n.key === activeKey) || navItems[0];
    const definitionsCount = Object.keys(aa.definitions || {}).length;

    return `
      <div class="aa-workspace-toolbar">
        <button class="dark-button compact" type="button" data-aa-open-master>${ICON.file}<span>View master document</span></button>
        <button class="ghost-button compact" type="button" data-aa-draft-all title="Run a draft pass for every thread that has answered RFIs but isn't drafted yet">${ICON.sparkles}<span>Draft all</span></button>
        <span class="aa-draft-all-status-inline" data-aa-draft-all-status hidden></span>
      </div>
      <div class="aa-two-pane">
        <aside class="aa-disputes-nav card">
          <div class="card-head"><h3>Items</h3></div>
          <div class="card-body aa-disputes-list">
            ${navItems.map((n) => {
              const itemAA = n.thread && Array.isArray(n.thread.rounds) ? n.thread : { rounds: [] };
              const rounds = itemAA.rounds.length;
              const answered = itemAA.rounds.filter((r) => r.answer).length;
              const ready = itemAA.submissions && itemAA.submissions.length > 60;
              const engineReady = !!itemAA.isReady;
              const status = ready ? "drafted" : (engineReady ? "ready-to-draft" : (rounds > 0 && answered === rounds ? "answered" : (rounds > 0 ? "in-progress" : "idle")));
              return `
                <button class="aa-nav-item aa-nav-${status} ${n.key === activeKey ? "active" : ""}" type="button" data-aa-select="${attr(n.key)}">
                  <span class="aa-nav-row">
                    <span class="aa-nav-label">${escapeHtml(n.label)}</span>
                    ${ready ? '<span class="aa-nav-dot ok" title="Drafted">✓</span>'
                      : engineReady ? '<span class="aa-nav-dot ready" title="Sopal has enough info — click Draft this item">⚡</span>'
                      : (rounds > 0 ? '<span class="aa-nav-dot in-progress" title="In progress">●</span>' : '<span class="aa-nav-dot idle" title="Not started">○</span>')}
                  </span>
                  <span class="aa-nav-meta muted">${rounds === 0 ? "Not started" : `${answered}/${rounds} answered`}${ready ? " · drafted" : (engineReady ? " · ready to draft" : "")}</span>
                </button>`;
            }).join("")}
          </div>
          <footer class="aa-disputes-foot">
            <button class="ghost-button compact" type="button" data-aa-toggle-definitions>${ICON.book}<span>Definitions${definitionsCount ? ` (${definitionsCount})` : ""}</span></button>
          </footer>
        </aside>
        <section class="aa-rfi-pane card">
          <div class="card-head">
            <h3>${escapeHtml(active.label)}</h3>
            <span class="muted">${active.kind === "dispute" ? "Per-item RFI thread" : "Shared RFI thread"}</span>
          </div>
          <div class="aa-rfi-stream" data-aa-rfi-stream>
            ${(active.thread.rounds || []).length === 0
              ? `<div class="empty-state"><strong>Ready when you are.</strong><p>Click <em>Ask first RFI</em> below to have Sopal generate the first targeted question for this ${active.kind === "dispute" ? "dispute" : "thread"}.</p></div>`
              : (active.thread.rounds || []).map((r, i) => `
                  <div class="aa-rfi-round">
                    <div class="aa-rfi-q"><strong>RFI ${i + 1}.</strong> ${renderMarkdown(r.question || "")}</div>
                    ${r.answer
                      ? `<div class="aa-rfi-a"><span class="muted">Your answer:</span><div>${renderMarkdown(r.answer)}</div></div>`
                      : `<form class="aa-rfi-answer-form" data-aa-rfi-answer="${i}">
                           <textarea class="text-area auto-grow" name="answer" rows="3" placeholder="Type your answer to RFI ${i + 1}…"></textarea>
                           <button class="dark-button compact" type="submit">Submit answer</button>
                         </form>`}
                  </div>
                `).join("")}
          </div>
          <footer class="aa-rfi-footer">
            <button class="ghost-button compact" type="button" data-aa-next-rfi>${ICON.sparkles}<span>${(active.thread.rounds || []).length === 0 ? "Ask first RFI" : "Ask another RFI"}</span></button>
            <button class="ghost-button compact" type="button" data-aa-draft-thread>Draft this thread now</button>
            ${(active.thread.evidenceIndex && active.thread.evidenceIndex.length) || (active.thread.statDecContent || "").length > 30
              ? `<button class="ghost-button compact" type="button" data-aa-view-artifacts>View evidence + stat dec</button>` : ""}
            ${(active.thread.rounds || []).length > 0 || (active.thread.submissions || "").length > 0
              ? `<button class="ghost-button compact danger aa-rfi-reset" type="button" data-aa-reset-thread>Reset this thread</button>` : ""}
          </footer>
        </section>
      </div>
    `;
  }

  // Fullscreen master document modal — opened from the workspace toolbar.
  // Reuses renderAAMaster and bindAATocLinks. Includes all the export
  // actions in one place so the user can review + download in flow.
  function openAAMasterModal(project, aa) {
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal aa-master-modal" role="dialog" aria-modal="true">
            <div class="modal-head aa-master-modal-head">
              <h2>Master document</h2>
              <div class="aa-master-modal-actions">
                <button class="dark-button compact" type="button" data-aa-export>${ICON.download}<span>Export .doc</span></button>
                <button class="ghost-button compact" type="button" data-aa-export-statdecs>${ICON.download}<span>Stat dec</span></button>
                <button class="ghost-button compact" type="button" data-aa-export-soe>${ICON.download}<span>Evidence index</span></button>
                <button class="ghost-button compact" type="button" data-aa-print-master>${ICON.file}<span>Print</span></button>
                <button class="ghost-button compact" type="button" data-aa-copy-master>${ICON.copy}<span>Copy as Markdown</span></button>
                <button class="ghost-button compact" type="button" data-aa-rebuild>Rebuild</button>
                <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
              </div>
            </div>
            <div class="modal-body aa-master-modal-body">
              <div class="aa-master-body" data-aa-master>${renderAAMaster(project, aa)}</div>
            </div>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        rootEl.querySelector("[data-aa-rebuild]")?.addEventListener("click", () => {
          const mount = rootEl.querySelector("[data-aa-master]");
          if (mount) { mount.innerHTML = renderAAMaster(project, aa); bindAATocLinks(); }
        });
        rootEl.querySelector("[data-aa-export]")?.addEventListener("click", () => {
          aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-adjudication-application.doc`,
            `${escapeHtml(project.name)} — Adjudication Application`,
            renderAAMaster(project, aa));
        });
        rootEl.querySelector("[data-aa-export-statdecs]")?.addEventListener("click", () => {
          aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-statutory-declaration.doc`,
            `${escapeHtml(project.name)} — Statutory Declaration`,
            renderAAStatDecCompilation(project, aa));
        });
        rootEl.querySelector("[data-aa-export-soe]")?.addEventListener("click", () => {
          aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-evidence-index.doc`,
            `${escapeHtml(project.name)} — Index of Supporting Evidence`,
            renderAAEvidenceIndex(project, aa));
        });
        rootEl.querySelector("[data-aa-print-master]")?.addEventListener("click", () => {
          const win = window.open("", "_blank", "noopener");
          if (!win) { alert("Could not open the print preview. Please allow popups for this site and try again."); return; }
          win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>${escapeHtml(project.name)} — Adjudication Application</title>
            <style>body{font-family:"Source Serif Pro",Georgia,"Times New Roman",serif;font-size:12.5pt;line-height:1.55;color:#1a1a1a;max-width:760px;margin:28px auto;padding:0 24px}h1{font-size:20pt;text-align:center;margin:0 0 14px}h2{font-size:14pt;margin:22px 0 8px;padding-bottom:4px;border-bottom:1px solid #ccc}h3{font-size:12pt;margin:14px 0 6px}p{margin:0 0 10px}table{width:100%;border-collapse:collapse;margin:8px 0 14px;font-size:11pt}th,td{border:1px solid #999;padding:4px 6px;text-align:left;vertical-align:top}th{background:#f0ece4}.aa-toc{background:#f5f2ed;border:1px solid #ddd;border-radius:6px;padding:12px 16px;margin:0 0 22px;font-family:-apple-system,"Segoe UI",sans-serif;font-size:11pt}.aa-toc-link{display:flex;gap:8px;padding:2px 0;color:#1a1a1a;text-decoration:none}.aa-toc-num{flex:0 0 36px;font-weight:600}.aa-toc-indent-1{padding-left:22px}.aa-issue-tag{display:inline-block;font-size:9pt;padding:1px 6px;margin-left:6px;background:#e0e7ff;border-radius:999px}.print-actions{display:flex;gap:8px;margin:0 0 18px}.print-actions button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid #aaa;background:#fff;cursor:pointer}@media print{.print-actions{display:none}body{margin:0;padding:0 18px;max-width:none}}</style>
            </head><body>
              <div class="print-actions"><button onclick="window.print()">Print</button><button onclick="window.close()">Close</button></div>
              ${renderAAMaster(project, aa)}
            </body></html>`);
          win.document.close();
        });
        rootEl.querySelector("[data-aa-copy-master]")?.addEventListener("click", () => {
          copyText(aaMasterToMarkdown(project, aa));
          const btn = rootEl.querySelector("[data-aa-copy-master]");
          if (btn) {
            const original = btn.innerHTML;
            btn.innerHTML = `${ICON.copy}<span>Copied</span>`;
            setTimeout(() => { btn.innerHTML = original; }, 1100);
          }
        });
        bindAATocLinks();
        // Close on Escape.
        document.addEventListener("keydown", function handler(ev) {
          if (ev.key === "Escape") { document.removeEventListener("keydown", handler); close(); }
        });
      },
    };
    render();
  }

  function renderAAMaster(project, aa) {
    // Build the section list first with stable IDs so we can render a live
    // table-of-contents that scrolls the master pane to each section.
    const toc = [];
    function id(slug) { return `aa-sec-${slug}`; }
    const sections = [];
    sections.push(`<h1>Adjudication Application</h1>`);
    toc.push({ id: id("parties"), num: "1", label: "Parties", indent: 0 });
    const scenarioId = aa.s79Scenario || "less-than-claimed";
    const scenario = AA_S79_SCENARIOS.find((s) => s.id === scenarioId) || AA_S79_SCENARIOS[0];
    sections.push(`<h2 id="${id("parties")}">1. Parties</h2>
      <p><strong>Claimant:</strong> ${escapeHtml(aa.parties.claimant || project.claimant || "[Claimant]")}<br>
      <strong>Respondent:</strong> ${escapeHtml(aa.parties.respondent || project.respondent || "[Respondent]")}<br>
      <strong>Contract reference:</strong> ${escapeHtml(aa.contractReference || project.reference || "[Contract reference]")}<br>
      <strong>Reference date:</strong> ${escapeHtml(aa.referenceDate || "[Reference date]")}<br>
      <strong>Claimed amount:</strong> ${formatCurrencyFull(aa.claimedAmount || 0)}<br>
      <strong>Scheduled amount:</strong> ${formatCurrencyFull(aa.scheduledAmount || 0)}<br>
      <strong>s 79 BIF Act scenario:</strong> ${escapeHtml(scenario.label)}</p>`);
    toc.push({ id: id("jurisdiction"), num: "2", label: "Jurisdiction", indent: 0 });
    sections.push(`<h2 id="${id("jurisdiction")}">2. Jurisdiction</h2>${aa.jurisdictionalRfis.submissions || "<p><em>(Jurisdictional submissions will appear here once the jurisdictional RFI thread is drafted.)</em></p>"}`);
    toc.push({ id: id("background"), num: "3", label: "Background", indent: 0 });
    sections.push(`<h2 id="${id("background")}">3. Background</h2>${aa.generalRfis.submissions || "<p><em>(Background will appear here once the general RFI thread is drafted.)</em></p>"}`);
    if (aa.disputes.length) {
      toc.push({ id: id("disputes"), num: "4", label: "Submissions on disputed items", indent: 0 });
      sections.push(`<h2 id="${id("disputes")}">4. Submissions on disputed items</h2>`);
      aa.disputes.forEach((d, i) => {
        const slug = `dispute-${d.id}`;
        toc.push({ id: id(slug), num: `4.${i + 1}`, label: d.item || "Item", indent: 1 });
        sections.push(`<h3 id="${id(slug)}">4.${i + 1} ${escapeHtml(d.item || "Item")}${d.issueType ? ` <span class="aa-issue-tag">${escapeHtml(AA_ISSUE_TYPE_LABELS[d.issueType] || d.issueType)}</span>` : ""}</h3>`);
        const subs = (d.rfis && d.rfis.submissions) || "";
        sections.push(subs || "<p><em>(Drafted once enough RFIs are answered.)</em></p>");
      });
    }
    if (aa.disputes.length) {
      toc.push({ id: id("quantum"), num: "5", label: "Quantum summary", indent: 0 });
      const totalClaimed = aa.disputes.reduce((s, d) => s + Number(d.claimed || 0), 0);
      const totalScheduled = aa.disputes.reduce((s, d) => s + Number(d.scheduled || 0), 0);
      sections.push(`<h2 id="${id("quantum")}">5. Quantum summary</h2>
        <table>
          <thead><tr><th>Item</th><th>Claimed</th><th>Scheduled</th><th>In dispute</th></tr></thead>
          <tbody>
            ${aa.disputes.map((d) => {
              const claimed = Number(d.claimed || 0);
              const scheduled = Number(d.scheduled || 0);
              const inDispute = Math.max(0, claimed - scheduled);
              return `<tr><td>${escapeHtml(d.item || "Item")}</td><td>${formatCurrencyFull(claimed)}</td><td>${formatCurrencyFull(scheduled)}</td><td>${formatCurrencyFull(inDispute)}</td></tr>`;
            }).join("")}
            <tr><td><strong>Totals</strong></td><td><strong>${formatCurrencyFull(totalClaimed)}</strong></td><td><strong>${formatCurrencyFull(totalScheduled)}</strong></td><td><strong>${formatCurrencyFull(Math.max(0, totalClaimed - totalScheduled))}</strong></td></tr>
          </tbody>
        </table>`);
    }
    toc.push({ id: id("conclusion"), num: aa.disputes.length ? "6" : "5", label: "Conclusion and amount sought", indent: 0 });
    sections.push(`<h2 id="${id("conclusion")}">${aa.disputes.length ? "6" : "5"}. Conclusion and amount sought</h2><p>For the reasons set out above, the Claimant respectfully seeks an adjudicated amount of ${formatCurrencyFull(aa.claimedAmount || 0)}.</p>`);
    const evidence = [];
    aa.disputes.forEach((d) => ((d.rfis && d.rfis.evidenceIndex) || []).forEach((e) => evidence.push(e)));
    const evidenceNum = aa.disputes.length ? "7" : "6";
    toc.push({ id: id("evidence"), num: evidenceNum, label: "Index of supporting evidence", indent: 0 });
    sections.push(`<h2 id="${id("evidence")}">${evidenceNum}. Index of supporting evidence</h2>${evidence.length ? `<ol>${evidence.map((e) => `<li><strong>${escapeHtml(e.ref || "")}</strong> — ${escapeHtml(e.desc || "")}${e.location ? ` (${escapeHtml(e.location)})` : ""}</li>`).join("")}</ol>` : "<p><em>(No exhibits indexed yet.)</em></p>"}`);

    const tocHtml = `
      <nav class="aa-toc" aria-label="Master document contents">
        <span class="aa-toc-label muted">Contents</span>
        ${toc.map((t) => `<a class="aa-toc-link aa-toc-indent-${t.indent}" href="#${t.id}" data-aa-toc-target="${t.id}"><span class="aa-toc-num">${escapeHtml(t.num)}</span><span class="aa-toc-text">${escapeHtml(t.label)}</span></a>`).join("")}
      </nav>`;

    return tocHtml + sections.join("\n");
  }

  function openAADefinitionsModal(project, aa) {
    if (!aa.definitions) aa.definitions = {};
    const entries = () => Object.entries(aa.definitions || {});
    function listHtml() {
      if (!entries().length) {
        return `<p class="muted">No defined terms yet. As Sopal drafts each thread it'll add the capitalised terms it introduces (e.g. "Contract", "Project", "Variation Notice"). You can also add or edit terms here directly.</p>`;
      }
      return `<div class="aa-defs-list">${entries().map(([term, def]) => `
        <div class="aa-def-row" data-aa-def-row="${attr(term)}">
          <input class="text-input aa-def-term" data-aa-def-term value="${attr(term)}">
          <textarea class="text-area aa-def-meaning" data-aa-def-meaning rows="2">${escapeHtml(def)}</textarea>
          <button class="ghost-button compact danger" type="button" data-aa-def-delete title="Remove">${ICON.trash}</button>
        </div>`).join("")}</div>`;
    }
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal aa-defs-modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>Definitions</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <div class="modal-body">
              <p class="muted">Defined terms are passed to every per-thread engine call so Sopal uses them consistently across the master document.</p>
              <div data-aa-defs-list-host>${listHtml()}</div>
              <form class="aa-def-add-row" data-aa-def-add-form>
                <input class="text-input" name="term" placeholder="New term (e.g. Contract)">
                <input class="text-input" name="def" placeholder="Definition (e.g. the AS 4902 head contract dated 12 March 2025)">
                <button class="dark-button" type="submit">${ICON.plus}<span>Add</span></button>
              </form>
            </div>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));

        function rerenderList() {
          const host = rootEl.querySelector("[data-aa-defs-list-host]");
          if (host) host.innerHTML = listHtml();
          bindRowEvents();
        }
        function bindRowEvents() {
          rootEl.querySelectorAll("[data-aa-def-row]").forEach((row) => {
            const oldTerm = row.dataset.aaDefRow;
            const termInput = row.querySelector("[data-aa-def-term]");
            const meaningInput = row.querySelector("[data-aa-def-meaning]");
            const onChange = () => {
              const newTerm = termInput.value.trim();
              const newDef = meaningInput.value;
              if (!newTerm) return;
              if (newTerm !== oldTerm) {
                delete aa.definitions[oldTerm];
                row.dataset.aaDefRow = newTerm;
              }
              aa.definitions[newTerm] = newDef;
              saveProject(project);
            };
            termInput.addEventListener("change", onChange);
            meaningInput.addEventListener("input", onChange);
            row.querySelector("[data-aa-def-delete]")?.addEventListener("click", () => {
              delete aa.definitions[oldTerm];
              saveProject(project);
              rerenderList();
            });
          });
        }
        rootEl.querySelector("[data-aa-def-add-form]")?.addEventListener("submit", (e) => {
          e.preventDefault();
          const f = e.currentTarget;
          const t = f.elements.term.value.trim();
          const d = f.elements.def.value.trim();
          if (!t) return;
          aa.definitions[t] = d;
          saveProject(project);
          f.reset();
          rerenderList();
        });
        document.addEventListener("keydown", function handler(ev) {
          if (ev.key === "Escape") { document.removeEventListener("keydown", handler); close(); }
        });
        bindRowEvents();
      },
    };
    render();
  }

  function openAAArtifactsModal(project, aa) {
    const ctx = aaActiveThread(aa);
    const evidence = ctx.thread.evidenceIndex || [];
    const statDec = ctx.thread.statDecContent || "";
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal aa-artifacts-modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>${escapeHtml(ctx.label)} — supporting artefacts</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <div class="modal-body aa-artifacts-body">
              <section>
                <h4>Evidence index for this thread</h4>
                ${evidence.length
                  ? `<ul class="aa-artifact-list">${evidence.map((e) => `<li><strong>${escapeHtml(e.ref || "")}</strong> — ${escapeHtml(e.desc || "")}${e.location ? ` <span class="muted">(${escapeHtml(e.location)})</span>` : ""}</li>`).join("")}</ul>`
                  : `<p class="muted">No exhibits indexed for this thread yet. Sopal adds them as you draft.</p>`}
              </section>
              <section>
                <h4>Statutory declaration content for this thread</h4>
                ${statDec
                  ? `<div class="aa-statdec">${renderMarkdown(statDec)}</div><div class="aa-artifact-actions"><button class="ghost-button compact" type="button" data-aa-statdec-export>${ICON.download}<span>Export this stat dec section as .doc</span></button></div>`
                  : `<p class="muted">No stat-dec content drafted for this thread yet. Sopal adds first-person factual statements as you draft each item.</p>`}
              </section>
            </div>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        rootEl.querySelector("[data-aa-statdec-export]")?.addEventListener("click", () => {
          const filename = `${project.name.replace(/[^a-z0-9]+/gi, "-")}-statdec-${ctx.id || ctx.kind}.doc`;
          const html = `<h1>Statutory declaration — ${escapeHtml(ctx.label)}</h1>${renderMarkdown(statDec)}`;
          const blob = new Blob([
            '<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">',
            '<head><meta charset="UTF-8"></head><body>', html, '</body></html>',
          ], { type: "application/msword" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          a.click();
          URL.revokeObjectURL(url);
        });
        document.addEventListener("keydown", function handler(ev) {
          if (ev.key === "Escape") { document.removeEventListener("keydown", handler); close(); }
        });
      },
    };
    render();
  }

  function bindAATocLinks() {
    // ToC <a href="#aa-sec-..."> would scroll the WHOLE page (and slam the
    // master pane out of frame). Intercept and scroll only inside the
    // master pane's scroll container.
    const master = document.querySelector("[data-aa-master]");
    if (!master) return;
    master.querySelectorAll("[data-aa-toc-target]").forEach((a) => {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        const id = a.dataset.aaTocTarget;
        const target = document.getElementById(id);
        if (target && master.contains(target)) {
          master.scrollTop = target.offsetTop - 8;
        }
      });
    });
  }

  /* ---------- Complex AA — wiring ---------- */

  function bindComplexAA(project) {
    const aa = getComplexAA(project);

    document.querySelectorAll("[data-aa-back-stage]").forEach((b) => b.addEventListener("click", () => {
      const idx = AA_STAGES.findIndex((s) => s.id === aa.stage);
      if (idx > 0) {
        aa.stage = AA_STAGES[idx - 1].id;
        saveProject(project);
        render();
      }
    }));
    document.querySelectorAll("[data-aa-jump]").forEach((b) => b.addEventListener("click", () => {
      const id = b.dataset.aaJump;
      if (!id || id === aa.stage) return;
      aa.stage = id;
      saveProject(project);
      render();
    }));
    document.querySelectorAll("[data-aa-reset]").forEach((b) => b.addEventListener("click", () => {
      if (!confirm("Reset the entire adjudication application workflow? All extracted items, RFIs, and drafts will be cleared.")) return;
      delete project.complexApps["adjudication-application"];
      saveProject(project);
      render();
    }));

    if (aa.stage === "intake") return bindAAIntake(project, aa);
    if (aa.stage === "dispute-table") return bindAADisputeTable(project, aa);
    if (aa.stage === "review") return bindAAReview(project, aa);
    return bindAAWorkspace(project, aa);
  }

  function bindAAReview(project, aa) {
    document.querySelector("[data-aa-rebuild]")?.addEventListener("click", () => {
      const mount = document.querySelector("[data-aa-master]");
      if (mount) {
        mount.innerHTML = renderAAMaster(project, aa);
        bindAATocLinks();
      }
    });
    document.querySelector("[data-aa-export]")?.addEventListener("click", () => {
      aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-adjudication-application.doc`,
        `${escapeHtml(project.name)} — Adjudication Application`,
        renderAAMaster(project, aa));
    });
    document.querySelector("[data-aa-export-statdecs]")?.addEventListener("click", () => {
      aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-statutory-declaration.doc`,
        `${escapeHtml(project.name)} — Statutory Declaration`,
        renderAAStatDecCompilation(project, aa));
    });
    document.querySelector("[data-aa-export-soe]")?.addEventListener("click", () => {
      aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-evidence-index.doc`,
        `${escapeHtml(project.name)} — Index of Supporting Evidence`,
        renderAAEvidenceIndex(project, aa));
    });
    document.querySelector("[data-aa-print-master]")?.addEventListener("click", () => {
      const title = `${project.name} — Adjudication Application`;
      const win = window.open("", "_blank", "noopener");
      if (!win) {
        alert("Could not open the print preview. Please allow popups for this site and try again.");
        return;
      }
      // Reuse the main app's typography for the print preview so the user
      // sees what they're printing in the same style as the live workspace.
      win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>${escapeHtml(title)}</title>
        <style>
          :root { color-scheme: light; }
          body { font-family: "Source Serif Pro", Georgia, "Times New Roman", serif; font-size: 12.5pt; line-height: 1.55; color: #1a1a1a; max-width: 760px; margin: 28px auto; padding: 0 24px; }
          h1 { font-size: 20pt; text-align: center; margin: 0 0 14px; }
          h2 { font-size: 14pt; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #ccc; }
          h3 { font-size: 12pt; margin: 14px 0 6px; }
          p { margin: 0 0 10px; }
          table { width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 11pt; }
          th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; vertical-align: top; }
          th { background: #f0ece4; }
          .aa-toc { background: #f5f2ed; border: 1px solid #ddd; border-radius: 6px; padding: 12px 16px; margin: 0 0 22px; font-family: -apple-system, "Segoe UI", sans-serif; font-size: 11pt; }
          .aa-toc-link { display: flex; gap: 8px; padding: 2px 0; color: #1a1a1a; text-decoration: none; }
          .aa-toc-num { flex: 0 0 36px; font-weight: 600; }
          .aa-toc-indent-1 { padding-left: 22px; }
          .aa-issue-tag { display: inline-block; font-size: 9pt; padding: 1px 6px; margin-left: 6px; background: #e0e7ff; border-radius: 999px; }
          .print-actions { display: flex; gap: 8px; margin: 0 0 18px; }
          .print-actions button { font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid #aaa; background: #fff; cursor: pointer; }
          @media print { .print-actions { display: none; } body { margin: 0; padding: 0 18px; max-width: none; } }
        </style></head><body>
          <div class="print-actions">
            <button onclick="window.print()">Print</button>
            <button onclick="window.close()">Close</button>
          </div>
          ${renderAAMaster(project, aa)}
        </body></html>`);
      win.document.close();
    });
    document.querySelector("[data-aa-copy-master]")?.addEventListener("click", () => {
      const md = aaMasterToMarkdown(project, aa);
      copyText(md);
      const btn = document.querySelector("[data-aa-copy-master]");
      if (btn) {
        const original = btn.innerHTML;
        btn.innerHTML = `${ICON.copy}<span>Copied</span>`;
        setTimeout(() => { btn.innerHTML = original; }, 1100);
      }
    });
    document.querySelector("[data-aa-draft-all]")?.addEventListener("click", () => runDraftAll(project, aa));
    bindAATocLinks();
  }

  function aaMasterToMarkdown(project, aa) {
    // Convert the master HTML to a reasonable Markdown rendering. Quick + fit
    // for purpose: handles h1/h2/h3, p, ul/ol/li, table, strong/em, br.
    const html = renderAAMaster(project, aa);
    const tmpl = document.createElement("div");
    tmpl.innerHTML = html;
    const out = [];
    function walk(node, depth) {
      for (const c of Array.from(node.childNodes)) {
        if (c.nodeType === 3) {
          const t = c.textContent.replace(/\s+/g, " ");
          out.push(t);
          continue;
        }
        if (c.nodeType !== 1) continue;
        const tag = c.tagName;
        if (tag === "NAV" && c.classList.contains("aa-toc")) continue; // skip ToC in Markdown
        if (tag === "H1") { out.push(`\n\n# ${c.textContent.trim()}\n\n`); continue; }
        if (tag === "H2") { out.push(`\n\n## ${c.textContent.trim()}\n\n`); continue; }
        if (tag === "H3") { out.push(`\n\n### ${c.textContent.trim()}\n\n`); continue; }
        if (tag === "H4") { out.push(`\n\n#### ${c.textContent.trim()}\n\n`); continue; }
        if (tag === "P") { out.push("\n"); walk(c, depth); out.push("\n"); continue; }
        if (tag === "BR") { out.push("\n"); continue; }
        if (tag === "STRONG" || tag === "B") { out.push(`**${c.textContent.trim()}**`); continue; }
        if (tag === "EM" || tag === "I") { out.push(`*${c.textContent.trim()}*`); continue; }
        if (tag === "UL" || tag === "OL") {
          let i = 1;
          for (const li of Array.from(c.children)) {
            if (li.tagName !== "LI") continue;
            const marker = tag === "UL" ? "-" : `${i++}.`;
            out.push(`\n${marker} ${li.textContent.trim()}`);
          }
          out.push("\n");
          continue;
        }
        if (tag === "TABLE") {
          const rows = Array.from(c.querySelectorAll("tr"));
          if (!rows.length) continue;
          const headerCells = Array.from(rows[0].querySelectorAll("th,td")).map((x) => x.textContent.trim());
          out.push("\n\n| " + headerCells.join(" | ") + " |\n");
          out.push("|" + headerCells.map(() => " --- ").join("|") + "|\n");
          for (let i = 1; i < rows.length; i++) {
            const tds = Array.from(rows[i].querySelectorAll("td,th")).map((x) => x.textContent.trim());
            out.push("| " + tds.join(" | ") + " |\n");
          }
          out.push("\n");
          continue;
        }
        walk(c, depth);
      }
    }
    walk(tmpl, 0);
    return out.join("").replace(/\n{3,}/g, "\n\n").trim() + "\n";
  }

  function aaDownloadDoc(filename, title, body) {
    const blob = new Blob([
      '<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">',
      '<head><meta charset="UTF-8"><title>', title, '</title></head><body>',
      body,
      '</body></html>',
    ], { type: "application/msword" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function renderAAStatDecCompilation(project, aa) {
    const dep = aa.parties.claimant || project.claimant || "[Deponent]";
    const out = [];
    out.push(`<h1>Statutory Declaration</h1>`);
    out.push(`<p>I, [name], of [address], [occupation], do solemnly and sincerely declare that:</p>`);
    out.push(`<p>1. I am authorised to make this declaration on behalf of ${escapeHtml(dep)} (the <strong>Claimant</strong>) in support of the Adjudication Application made under the Building Industry Fairness (Security of Payment) Act 2017 (Qld) in respect of the matter known as ${escapeHtml(aa.contractReference || project.reference || "[Contract reference]")}.</p>`);
    out.push(`<p>2. The matters declared below are within my own knowledge except where otherwise stated, and where stated to be on information and belief, I believe them to be true.</p>`);
    let para = 3;
    function addThread(label, content) {
      const text = (content || "").trim();
      if (!text) return;
      out.push(`<h3>${escapeHtml(label)}</h3>`);
      // Renumber inline first-person paragraphs into the master numbering.
      const html = renderMarkdown(text);
      out.push(html);
      para += 1;
    }
    addThread("Jurisdictional facts", aa.jurisdictionalRfis.statDecContent);
    addThread("Background facts", aa.generalRfis.statDecContent);
    (aa.disputes || []).forEach((d) => addThread(`Item — ${d.item || "Item"}`, (d.rfis && d.rfis.statDecContent) || ""));
    out.push(`<h3>Declaration</h3>`);
    out.push(`<p>And I make this solemn declaration conscientiously believing the same to be true and by virtue of the provisions of the <em>Oaths Act 1867</em> (Qld).</p>`);
    out.push(`<p>Declared at [place] in the State of Queensland on [date].</p>`);
    out.push(`<p>........................................<br>[Deponent name]<br>Before me:<br>........................................<br>[JP / Solicitor / Commissioner for Declarations]</p>`);
    return out.join("\n");
  }

  function renderAAEvidenceIndex(project, aa) {
    const all = [];
    function addRows(label, items) {
      (items || []).forEach((e) => all.push({ label, e }));
    }
    addRows("Jurisdictional", aa.jurisdictionalRfis.evidenceIndex);
    addRows("Background", aa.generalRfis.evidenceIndex);
    (aa.disputes || []).forEach((d) => addRows(d.item || "Item", (d.rfis && d.rfis.evidenceIndex) || []));
    if (!all.length) return `<h1>Index of Supporting Evidence</h1><p><em>(No exhibits indexed yet.)</em></p>`;
    return `<h1>Index of Supporting Evidence</h1>
      <table>
        <thead><tr><th>Ref</th><th>Description</th><th>Cross-reference</th><th>Thread</th></tr></thead>
        <tbody>
          ${all.map(({ label, e }) => `<tr><td><strong>${escapeHtml(e.ref || "")}</strong></td><td>${escapeHtml(e.desc || "")}</td><td>${escapeHtml(e.location || "")}</td><td>${escapeHtml(label)}</td></tr>`).join("")}
        </tbody>
      </table>`;
  }

  function bindAAIntake(project, aa) {
    function setText(slot, text, name) {
      aa.documents[slot] = text ? { name: name || aa.documents[slot]?.name || "Pasted text", text } : null;
      aa.updatedAt = Date.now();
      saveProject(project);
    }
    document.querySelectorAll("[data-aa-text]").forEach((ta) => {
      ta.addEventListener("input", () => {
        const slot = ta.dataset.aaText;
        setText(slot, ta.value, aa.documents[slot]?.name || "Pasted text");
      });
    });
    document.querySelectorAll("[data-aa-file]").forEach((input) => {
      input.addEventListener("change", async () => {
        const slot = input.dataset.aaFile;
        const status = document.querySelector(`[data-aa-file-status-${slot}]`);
        const file = input.files && input.files[0];
        if (!file) return;
        if (status) status.textContent = `Extracting ${file.name}…`;
        const fd = new FormData();
        fd.append("file", file);
        try {
          const response = await fetch("/api/sopal-v2/extract", { method: "POST", body: fd, credentials: "include" });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(describeApiError(data, "Extraction failed"));
          setText(slot, data.text, data.filename);
          if (status) status.textContent = `${data.filename} · ${(data.characters || 0).toLocaleString()} chars`;
          const ta = document.querySelector(`[data-aa-text="${slot}"]`);
          if (ta) ta.value = data.text;
        } catch (error) {
          if (status) status.textContent = error.message || "Extraction failed";
        }
      });
    });
    const deadlineInput = document.querySelector("[data-aa-deadline]");
    if (deadlineInput) deadlineInput.addEventListener("change", () => {
      aa.deadline = deadlineInput.value || null;
      saveProject(project);
    });
    document.querySelectorAll('input[name="aa-s79"]').forEach((r) => r.addEventListener("change", () => {
      aa.s79Scenario = r.value;
      saveProject(project);
      render();
    }));
    document.querySelector("[data-aa-parse]")?.addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      const pc = aa.documents.paymentClaim;
      const ps = aa.documents.paymentSchedule;
      const scenario = AA_S79_SCENARIOS.find((s) => s.id === (aa.s79Scenario || "less-than-claimed")) || AA_S79_SCENARIOS[0];
      if (!pc || !pc.text) {
        alert("Add the Payment Claim before parsing.");
        return;
      }
      if (!scenario.psOptional && (!ps || !ps.text)) {
        alert("This s 79 scenario requires a Payment Schedule. Either paste it in or switch the scenario to 'No payment schedule received'.");
        return;
      }
      btn.disabled = true;
      btn.innerHTML = `<span class="thinking-dots"><i></i><i></i><i></i></span><span>Parsing…</span>`;
      try {
        const ctrl = new AbortController();
        const timeoutId = setTimeout(() => ctrl.abort(), 120_000);
        let response, data;
        try {
          response = await fetch("/api/sopal-v2/complex/aa/parse-documents", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: "include",
            body: JSON.stringify({
              paymentClaimText: pc.text,
              paymentScheduleText: ps && ps.text ? ps.text : "",
              s79Scenario: aa.s79Scenario || "less-than-claimed",
              projectMeta: { name: project.name, claimant: project.claimant, respondent: project.respondent, contractForm: project.contractForm, reference: project.reference },
            }),
            signal: ctrl.signal,
          });
          data = await response.json().catch(() => ({}));
        } catch (err) {
          if (err.name === "AbortError") throw new Error("Parsing took too long (over 120s). Try smaller documents or paste the key sections instead.");
          throw err;
        } finally {
          clearTimeout(timeoutId);
        }
        if (!response.ok) throw new Error(describeApiError(data, "Parse failed"));
        // Seed AA from the extract.
        aa.parties.claimant = data.parties?.claimant || project.claimant || "";
        aa.parties.respondent = data.parties?.respondent || project.respondent || "";
        aa.contractReference = data.contractReference || project.reference || "";
        aa.referenceDate = data.referenceDate || "";
        aa.claimedAmount = Number(data.claimedAmount || 0);
        aa.scheduledAmount = Number(data.scheduledAmount || 0);
        aa.psReasonsUniverse = data.psReasonsUniverse || "";
        aa.parseWarnings = Array.isArray(data.warnings) ? data.warnings : [];
        aa.disputes = (Array.isArray(data.lineItems) ? data.lineItems : []).map((li) => ({
          id: newDisputeId(),
          item: li.label || "",
          description: li.description || "",
          claimed: Number(li.claimed || li.amount || 0),
          scheduled: Number(li.scheduled || 0),
          psReasons: li.psReasons || "",
          status: li.status || "disputed",
          issueType: li.issueType || "other",
          rfis: { rounds: [], submissions: "", evidenceIndex: [], statDecContent: "", isReady: false },
          updatedAt: Date.now(),
        }));
        aa.stage = "dispute-table";
        saveProject(project);
        render();
      } catch (error) {
        alert(error.message || "Parse failed");
        btn.disabled = false;
        btn.innerHTML = `${ICON.sparkles}<span>Parse documents</span>`;
      }
    });
  }

  function bindAADisputeTable(project, aa) {
    document.querySelectorAll("[data-aa-row]").forEach((row) => {
      const id = row.dataset.aaRow;
      const dispute = aa.disputes.find((d) => d.id === id);
      if (!dispute) return;
      row.querySelectorAll("[data-aa-cell]").forEach((el) => {
        el.addEventListener("input", () => {
          const field = el.dataset.aaCell;
          const value = el.tagName === "SELECT" ? el.value : (field === "claimed" || field === "scheduled" ? Number(el.value || 0) : el.value);
          dispute[field] = value;
          dispute.updatedAt = Date.now();
          saveProject(project);
        });
      });
      row.querySelector("[data-aa-delete-row]")?.addEventListener("click", () => {
        if (!confirm("Delete this dispute row?")) return;
        aa.disputes = aa.disputes.filter((d) => d.id !== id);
        saveProject(project);
        render();
      });
    });
    document.querySelector("[data-aa-add-row]")?.addEventListener("click", () => {
      aa.disputes.push({
        id: newDisputeId(),
        item: "", description: "", claimed: 0, scheduled: 0,
        psReasons: "", status: "disputed", issueType: "other",
        rfis: { rounds: [], submissions: "", evidenceIndex: [], statDecContent: "", isReady: false },
        updatedAt: Date.now(),
      });
      saveProject(project);
      render();
    });
    document.querySelector("[data-aa-lock-table]")?.addEventListener("click", () => {
      if (!aa.disputes.length) {
        if (!confirm("No disputed items in the table — lock anyway? You can come back and add items later.")) return;
      }
      aa.stage = "rfi";
      saveProject(project);
      render();
    });
  }

  function aaActiveThread(aa) {
    const key = aa.activeKey || "jurisdictional";
    if (key === "jurisdictional") return { kind: "shared", thread: aa.jurisdictionalRfis, label: "Jurisdictional", id: null };
    if (key === "general") return { kind: "shared", thread: aa.generalRfis, label: "Background / General", id: null };
    if (key.startsWith("dispute:")) {
      const id = key.split(":", 2)[1];
      const d = aa.disputes.find((x) => x.id === id);
      if (d) return { kind: "dispute", thread: d.rfis, dispute: d, label: d.item || id, id };
    }
    return { kind: "shared", thread: aa.jurisdictionalRfis, label: "Jurisdictional", id: null };
  }

  function bindAAWorkspace(project, aa) {
    document.querySelectorAll("[data-aa-select]").forEach((b) => b.addEventListener("click", () => {
      aa.activeKey = b.dataset.aaSelect;
      saveProject(project);
      render();
    }));
    document.querySelectorAll("[data-aa-rfi-answer]").forEach((form) => {
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const idx = Number(form.dataset.aaRfiAnswer);
        const ans = form.elements.answer.value.trim();
        if (!ans) return;
        const ctx = aaActiveThread(aa);
        const round = ctx.thread.rounds[idx];
        if (!round) return;
        round.answer = ans;
        round.answeredAt = Date.now();
        saveProject(project);
        // After the user answers, ask the server to either generate the next
        // RFI for this thread or to draft this item. We default to "next RFI"
        // unless this was the per-item thread's answer to "ready to draft?".
        await aaCallEngine(project, aa, "rfi-followup");
        render();
      });
    });
    document.querySelector("[data-aa-next-rfi]")?.addEventListener("click", async () => {
      await aaCallEngine(project, aa, "rfi-next");
      render();
    });
    document.querySelector("[data-aa-draft-item]")?.addEventListener("click", async () => {
      await aaCallEngine(project, aa, "draft");
      render();
    });
    document.querySelector("[data-aa-draft-thread]")?.addEventListener("click", async () => {
      await aaCallEngine(project, aa, "draft");
      render();
    });
    document.querySelector("[data-aa-reset-thread]")?.addEventListener("click", () => {
      const ctx = aaActiveThread(aa);
      if (!ctx || !ctx.thread) return;
      const label = ctx.label || "this thread";
      if (!confirm(`Reset ${label}? This clears the RFIs, draft submissions, evidence index and stat-dec content for this thread. Other threads are not affected.`)) return;
      ctx.thread.rounds = [];
      ctx.thread.submissions = "";
      ctx.thread.evidenceIndex = [];
      ctx.thread.statDecContent = "";
      ctx.thread.isReady = false;
      saveProject(project);
      render();
    });
    document.querySelector("[data-aa-rebuild]")?.addEventListener("click", () => {
      const mount = document.querySelector("[data-aa-master]");
      if (mount) {
        mount.innerHTML = renderAAMaster(project, aa);
        bindAATocLinks();
      }
    });
    document.querySelector("[data-aa-toggle-definitions]")?.addEventListener("click", () => openAADefinitionsModal(project, aa));
    document.querySelector("[data-aa-view-artifacts]")?.addEventListener("click", () => openAAArtifactsModal(project, aa));
    document.querySelector("[data-aa-draft-all]")?.addEventListener("click", () => runDraftAll(project, aa));
    document.querySelector("[data-aa-open-master]")?.addEventListener("click", () => openAAMasterModal(project, aa));
    // Auto-fire the first RFI on a freshly-opened thread so the user
    // doesn't have to click 'Ask first RFI'. Gated by autoAskedAt to avoid
    // spamming if the user navigates away and back, or if the first RFI
    // call is in flight.
    const ctx = aaActiveThread(aa);
    if (ctx && ctx.thread && (ctx.thread.rounds || []).length === 0 && !ctx.thread.autoAskedAt && !ctx.thread.submissions) {
      ctx.thread.autoAskedAt = Date.now();
      saveProject(project);
      // Fire and forget — render() inside aaCallEngine's caller will pick up
      // the new RFI when it lands.
      (async () => {
        await aaCallEngine(project, aa, "rfi-next");
        render();
      })();
    }
    bindAATocLinks();
    document.querySelector("[data-aa-export]")?.addEventListener("click", () => {
      aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-adjudication-application.doc`,
        `${escapeHtml(project.name)} — Adjudication Application`,
        renderAAMaster(project, aa));
    });
  }

  // Run engine.draft for every thread that has at least one answered RFI
  // and isn't already drafted. Threads run in parallel — each landing
  // patches its own thread state. The master pane refreshes once at the
  // end. Status line in the master footer reports running / N drafted /
  // M failed live.
  async function runDraftAll(project, aa) {
    const candidates = [];
    function addCandidate(kind, key, thread, dispute) {
      const answered = (thread.rounds || []).filter((r) => r.answer).length;
      const drafted = (thread.submissions || "").length > 60;
      if (answered === 0) return;          // no facts yet — skip
      if (drafted) return;                  // already drafted — skip
      candidates.push({ kind, key, thread, dispute });
    }
    addCandidate("shared", "jurisdictional", aa.jurisdictionalRfis, null);
    addCandidate("shared", "general", aa.generalRfis, null);
    (aa.disputes || []).forEach((d) => addCandidate("dispute", `dispute:${d.id}`, d.rfis, d));

    const status = document.querySelector("[data-aa-draft-all-status]");
    function setStatus(html) { if (status) { status.hidden = false; status.innerHTML = html; } }
    function clearStatus() { if (status) { status.hidden = true; status.innerHTML = ""; } }

    if (!candidates.length) {
      setStatus(`<span class="muted">Nothing to draft yet — answer at least one RFI on a thread first.</span>`);
      setTimeout(clearStatus, 4000);
      return;
    }
    setStatus(`<span class="thinking-dots"><i></i><i></i><i></i></span><span>Drafting ${candidates.length} thread${candidates.length === 1 ? "" : "s"} in parallel…</span>`);

    let done = 0;
    let failed = 0;
    const results = await Promise.all(candidates.map(async (c) => {
      try {
        const payload = {
          mode: "draft",
          threadKind: c.kind,
          threadLabel: c.kind === "dispute" ? (c.dispute && c.dispute.item) || "" : (c.key === "jurisdictional" ? "Jurisdictional" : "Background / General"),
          disputeId: c.kind === "dispute" ? (c.dispute && c.dispute.id) : null,
          dispute: c.dispute || null,
          rounds: c.thread.rounds || [],
          existingSubmissions: c.thread.submissions || "",
          parties: aa.parties,
          contractReference: aa.contractReference,
          referenceDate: aa.referenceDate,
          claimedAmount: aa.claimedAmount,
          scheduledAmount: aa.scheduledAmount,
          psReasonsUniverse: aa.psReasonsUniverse,
          s79Scenario: aa.s79Scenario || "less-than-claimed",
          definitions: aa.definitions,
          contractDocs: (project.contracts || []).slice(0, 6).map((d) => ({ name: d.name || "Contract", text: (d.text || "").slice(0, 25_000) })),
          libraryDocs: (project.library || []).slice(0, 8).map((d) => ({ name: d.name || "Library", text: (d.text || "").slice(0, 18_000) })),
          projectMeta: { name: project.name, contractForm: project.contractForm },
        };
        const response = await fetch("/api/sopal-v2/complex/aa/engine", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify(payload),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(describeApiError(data, "Engine call failed"));
        if (data.submissionsHtml) c.thread.submissions = data.submissionsHtml;
        if (Array.isArray(data.evidenceIndex)) c.thread.evidenceIndex = data.evidenceIndex;
        if (typeof data.statDecContent === "string") c.thread.statDecContent = data.statDecContent;
        if (data.definitions) Object.assign(aa.definitions, data.definitions);
        done += 1;
      } catch (_err) {
        failed += 1;
      }
      // Re-render the master pane after each thread lands so progress is
      // visible without waiting for the whole batch.
      const mount = document.querySelector("[data-aa-master]");
      if (mount) { mount.innerHTML = renderAAMaster(project, aa); bindAATocLinks(); }
      setStatus(`<span class="thinking-dots"><i></i><i></i><i></i></span><span>${done + failed}/${candidates.length} done · ${done} drafted${failed ? ` · ${failed} failed` : ""}</span>`);
      saveProject(project);
    }));
    if (failed) {
      setStatus(`<span class="muted">Drafted ${done} of ${candidates.length}. ${failed} failed — try Draft again on the affected items.</span>`);
    } else {
      setStatus(`<span class="muted">Drafted ${done} of ${candidates.length}.</span>`);
    }
    // Re-render the whole shell so the items-nav status dots update.
    setTimeout(() => render(), 600);
    return results;
  }

  async function aaCallEngine(project, aa, mode) {
    const ctx = aaActiveThread(aa);
    // Show a transient thinking indicator on the active stream.
    const stream = document.querySelector("[data-aa-rfi-stream]");
    if (stream) stream.insertAdjacentHTML("beforeend", `<div class="aa-rfi-thinking" id="aa-thinking"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is working…</span></div>`);
    try {
      const payload = {
        mode,
        threadKind: ctx.kind,
        threadLabel: ctx.label,
        disputeId: ctx.id,
        dispute: ctx.dispute || null,
        rounds: ctx.thread.rounds || [],
        existingSubmissions: ctx.thread.submissions || "",
        parties: aa.parties,
        contractReference: aa.contractReference,
        referenceDate: aa.referenceDate,
        claimedAmount: aa.claimedAmount,
        scheduledAmount: aa.scheduledAmount,
        psReasonsUniverse: aa.psReasonsUniverse,
        s79Scenario: aa.s79Scenario || "less-than-claimed",
        definitions: aa.definitions,
        // Surface the project's uploaded Contract + Project Library docs so
        // the AI can quote real contract clauses in submissions instead of
        // emitting [bracketed placeholders]. Capped per-doc + total to keep
        // the payload sane.
        contractDocs: (project.contracts || []).slice(0, 6).map((d) => ({ name: d.name || "Contract", text: (d.text || "").slice(0, 25_000) })),
        libraryDocs: (project.library || []).slice(0, 8).map((d) => ({ name: d.name || "Library", text: (d.text || "").slice(0, 18_000) })),
        projectMeta: { name: project.name, contractForm: project.contractForm },
      };
      const ctrl = new AbortController();
      const timeoutId = setTimeout(() => ctrl.abort(), 90_000);
      let response, data;
      try {
        response = await fetch("/api/sopal-v2/complex/aa/engine", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify(payload),
          signal: ctrl.signal,
        });
        data = await response.json().catch(() => ({}));
      } catch (err) {
        if (err.name === "AbortError") throw new Error("Sopal took too long to respond (over 90s). Try again.");
        throw err;
      } finally {
        clearTimeout(timeoutId);
      }
      if (!response.ok) throw new Error(describeApiError(data, "Engine call failed"));
      // Apply patches.
      if (data.appendRfi) {
        ctx.thread.rounds.push({ question: data.appendRfi, answer: null, askedAt: Date.now(), answeredAt: null });
      }
      if (data.submissionsHtml) ctx.thread.submissions = data.submissionsHtml;
      if (Array.isArray(data.evidenceIndex)) ctx.thread.evidenceIndex = data.evidenceIndex;
      if (typeof data.statDecContent === "string") ctx.thread.statDecContent = data.statDecContent;
      if (data.definitions) Object.assign(aa.definitions, data.definitions);
      // The engine signals when it has enough to draft this thread without
      // more RFIs. Persist so the items nav can surface it.
      ctx.thread.isReady = !!data.isReady;
      saveProject(project);
    } catch (error) {
      // Surface the engine error inline next to the thinking row instead of
      // bouncing an alert — keeps the workflow flowing and lets the user
      // retry with one click.
      if (stream) {
        const t = document.getElementById("aa-thinking");
        if (t) t.remove();
        stream.insertAdjacentHTML("beforeend", `<div class="error-banner aa-rfi-error">${escapeHtml(error.message || "Engine call failed")}<button class="ghost-button compact aa-rfi-retry" type="button">Retry</button></div>`);
        const retry = stream.querySelector(".aa-rfi-error:last-child .aa-rfi-retry");
        if (retry) retry.addEventListener("click", () => {
          retry.closest(".aa-rfi-error").remove();
          aaCallEngine(project, aa, mode).then(() => render());
        });
      } else {
        alert(error.message || "Engine call failed");
      }
    } finally {
      const t = document.getElementById("aa-thinking");
      if (t) t.remove();
    }
  }

  /* ---------- Drafting workspace (Word-style editor + AI chat) ---------- */

  // Starter templates per drafting agent. The editor is a contenteditable div
  // that takes raw HTML, so each template is HTML with bracketed placeholders
  // the user (or the AI) fills in. New project drafts initialise from these
  // templates the first time the user opens that drafting agent.
  const AGENT_TEMPLATES = {
    "payment-claims": `
      <h1>Payment Claim</h1>
      <p><strong>Claimant:</strong> [Claimant name]<br>
      <strong>Respondent:</strong> [Respondent name]<br>
      <strong>Project:</strong> [Project name]<br>
      <strong>Contract reference:</strong> [Contract / PO no.]<br>
      <strong>Reference date:</strong> [DD Month YYYY]<br>
      <strong>Date served:</strong> [DD Month YYYY]</p>
      <h2>Amount claimed</h2>
      <p><strong>$[Amount] (incl. GST)</strong></p>
      <h2>Identification of the construction work / related goods &amp; services</h2>
      <p>[Describe the construction work or related goods and services to which this claim relates, by reference to the contract scope and the period worked.]</p>
      <h2>Breakdown</h2>
      <table>
        <thead><tr><th>Item</th><th>Description</th><th>Amount (excl. GST)</th></tr></thead>
        <tbody>
          <tr><td>1</td><td>[Item description]</td><td>$[Amount]</td></tr>
          <tr><td>2</td><td>[Item description]</td><td>$[Amount]</td></tr>
        </tbody>
      </table>
      <h2>Statutory endorsement</h2>
      <p>This is a payment claim made under section 75 of the Building Industry Fairness (Security of Payment) Act 2017 (Qld).</p>
      <h2>Service</h2>
      <p>Served on [Respondent name] on [DD Month YYYY] by [email / post / hand].</p>
      <p>Signed,<br>
      [Signatory name]<br>
      [Position]<br>
      [Claimant name]</p>
    `,
    "payment-schedules": `
      <h1>Payment Schedule</h1>
      <p><strong>Respondent (issuer):</strong> [Respondent name]<br>
      <strong>Claimant:</strong> [Claimant name]<br>
      <strong>Project:</strong> [Project name]<br>
      <strong>Payment claim being responded to:</strong> Claim dated [DD Month YYYY], received on [DD Month YYYY]</p>
      <h2>Scheduled amount</h2>
      <p><strong>$[Amount] (incl. GST)</strong></p>
      <h2>Reasons for any difference</h2>
      <p>[Itemise the reasons for any difference between the claimed amount and the scheduled amount. Each reason should identify the disputed item, the basis under the contract or BIF Act, and the supporting facts / documents.]</p>
      <table>
        <thead><tr><th>Claim item</th><th>Claimed</th><th>Scheduled</th><th>Reason for withholding</th></tr></thead>
        <tbody>
          <tr><td>1</td><td>$[Amount]</td><td>$[Amount]</td><td>[Reason]</td></tr>
        </tbody>
      </table>
      <h2>Reservation of rights</h2>
      <p>Nothing in this payment schedule is to be taken as an admission of liability or a waiver of any defence or right under the contract or at law.</p>
      <p>Signed,<br>
      [Signatory name]<br>
      [Position]<br>
      [Respondent name]</p>
    `,
    "eots": `
      <h1>Notice of Extension of Time</h1>
      <p><strong>From:</strong> [Contractor name]<br>
      <strong>To:</strong> [Superintendent / Principal name]<br>
      <strong>Project:</strong> [Project name]<br>
      <strong>Contract:</strong> [Contract reference]<br>
      <strong>Notice date:</strong> [DD Month YYYY]</p>
      <h2>1. Qualifying cause of delay</h2>
      <p>[Describe the qualifying cause of delay relied on, identifying the contractual provision and the underlying facts.]</p>
      <h2>2. When the Contractor became aware</h2>
      <p>The Contractor became aware of the cause of delay on [DD Month YYYY].</p>
      <h2>3. Likely effect on progress</h2>
      <p>[Describe the likely effect on the critical path and the date for practical completion. Reference programme analysis where available.]</p>
      <h2>4. Estimated extension claimed</h2>
      <p>[N] working days, extending the date for practical completion from [date] to [date].</p>
      <h2>5. Supporting documents</h2>
      <ul>
        <li>[Programme analysis]</li>
        <li>[Contemporaneous correspondence]</li>
        <li>[Photographs / site records]</li>
      </ul>
      <h2>6. Reservation of rights</h2>
      <p>This notice is given under clause [#] of the Contract. The Contractor reserves all other rights under the Contract and at law including, without limitation, claims for delay costs and variations.</p>
      <p>Signed,<br>
      [Signatory name]<br>
      [Position]<br>
      [Contractor name]</p>
    `,
    "variations": `
      <h1>Notice / Claim of Variation</h1>
      <p><strong>From:</strong> [Contractor name]<br>
      <strong>To:</strong> [Superintendent / Principal name]<br>
      <strong>Project:</strong> [Project name]<br>
      <strong>Contract:</strong> [Contract reference]<br>
      <strong>Notice date:</strong> [DD Month YYYY]</p>
      <h2>1. Direction or change relied on</h2>
      <p>[Describe the direction, instruction or change in scope relied on, with date and the person who gave it.]</p>
      <h2>2. Contractual basis</h2>
      <p>This claim is made under clause [#] of the Contract.</p>
      <h2>3. Description of the varied work</h2>
      <p>[Describe the varied work and how it differs from the original contract scope.]</p>
      <h2>4. Valuation</h2>
      <p>[Set out the proposed valuation method and amount. Reference the Schedule of Rates / quotes / day-work records as applicable.]</p>
      <p><strong>Variation value:</strong> $[Amount] (excl. GST)</p>
      <h2>5. Time impact</h2>
      <p>[State whether the variation is expected to affect the date for practical completion. If yes, a separate EOT notice is given / will follow.]</p>
      <h2>6. Supporting documents</h2>
      <ul>
        <li>[Cost build-up]</li>
        <li>[Quotes / invoices]</li>
        <li>[Site records / dockets]</li>
      </ul>
      <p>Signed,<br>
      [Signatory name]<br>
      [Position]<br>
      [Contractor name]</p>
    `,
    "delay-costs": `
      <h1>Claim for Delay Costs / Prolongation</h1>
      <p><strong>From:</strong> [Contractor name]<br>
      <strong>To:</strong> [Superintendent / Principal name]<br>
      <strong>Project:</strong> [Project name]<br>
      <strong>Contract:</strong> [Contract reference]<br>
      <strong>Claim date:</strong> [DD Month YYYY]</p>
      <h2>1. Entitlement basis</h2>
      <p>[Cite the contract clause or common-law basis on which the delay costs are claimed.]</p>
      <h2>2. Compensable delay period</h2>
      <p>The compensable delay period is [N] working days, from [start date] to [end date].</p>
      <h2>3. Causation</h2>
      <p>[Identify the qualifying cause(s) of delay, the link to the critical path, and any concurrent-delay analysis.]</p>
      <h2>4. Quantum</h2>
      <table>
        <thead><tr><th>Cost head</th><th>Rate</th><th>Period</th><th>Amount</th></tr></thead>
        <tbody>
          <tr><td>Site preliminaries</td><td>$[Rate/day]</td><td>[N] days</td><td>$[Amount]</td></tr>
          <tr><td>Off-site overheads</td><td>[Hudson / Emden / measured]</td><td>[N] days</td><td>$[Amount]</td></tr>
        </tbody>
      </table>
      <p><strong>Total claimed:</strong> $[Amount] (excl. GST)</p>
      <h2>5. Supporting records</h2>
      <ul>
        <li>[Programme / float analysis]</li>
        <li>[Payroll, plant and subcontractor records]</li>
        <li>[Contemporaneous correspondence]</li>
      </ul>
      <p>Signed,<br>
      [Signatory name]<br>
      [Position]<br>
      [Contractor name]</p>
    `,
    "general-correspondence": `
      <h1>[Letter / Email title]</h1>
      <p><strong>From:</strong> [Sender name, position, company]<br>
      <strong>To:</strong> [Recipient name, position, company]<br>
      <strong>Project:</strong> [Project name]<br>
      <strong>Contract:</strong> [Contract reference]<br>
      <strong>Date:</strong> [DD Month YYYY]</p>
      <p>Dear [Recipient name],</p>
      <p>[Opening paragraph identifying the contract and the matter being addressed.]</p>
      <p>[Substantive body — facts, contractual references, and any action requested.]</p>
      <p>[Closing paragraph — reservation of rights as appropriate.]</p>
      <p>Yours sincerely,<br>
      [Signatory name]<br>
      [Position]<br>
      [Company]</p>
    `,
    "adjudication-application": `
      <h1>Adjudication Application</h1>
      <p>[Use the structure: jurisdiction, statutory framework, contract background, payment claim, payment schedule, issues, entitlement, quantum, anticipated objections, evidence schedule.]</p>
    `,
    "adjudication-response": `
      <h1>Adjudication Response</h1>
      <p>[Use the structure: jurisdictional objections, response to each item of the application, evidence already raised in the schedule, evidence schedule.]</p>
    `,
  };

  function getProjectDraft(project, agentKey) {
    if (!project.drafts) project.drafts = {};
    if (!project.drafts[agentKey]) {
      project.drafts[agentKey] = {
        html: (AGENT_TEMPLATES[agentKey] || "<p>[Start drafting…]</p>").trim(),
        chat: { messages: [] },
        updatedAt: Date.now(),
      };
      saveProject(project);
    }
    if (!project.drafts[agentKey].chat || !Array.isArray(project.drafts[agentKey].chat.messages)) {
      project.drafts[agentKey].chat = { messages: [] };
    }
    return project.drafts[agentKey];
  }

  function DraftingWorkspace(project, agentKey, draftOnly) {
    const draft = getProjectDraft(project, agentKey);
    const reviewHref = `/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=review`;
    const tabs = draftOnly ? "" : `
      <div class="mode-tabs" role="tablist">
        <button class="mode-tab" type="button" data-go="${reviewHref}">Review</button>
        <button class="mode-tab active" type="button">Draft</button>
      </div>`;
    const head = `
      <div class="chat-page-head">
        <div>
          <h1 class="page-title">${escapeHtml(AGENT_LABELS[agentKey])}</h1>
          <p class="page-sub">${escapeHtml(AGENT_DESCRIPTIONS[agentKey] || "")}</p>
        </div>
        ${tabs}
      </div>`;

    const ctxCount = project.contracts.length + project.library.length;
    return `
      <div class="page-shell drafting-shell">
        ${head}
        <div class="drafting-toolbar">
          <div class="drafting-toolbar-left">
            <button class="ghost-button compact" type="button" data-doc-cmd="bold" title="Bold (⌘B)"><strong>B</strong></button>
            <button class="ghost-button compact" type="button" data-doc-cmd="italic" title="Italic (⌘I)"><em>I</em></button>
            <button class="ghost-button compact" type="button" data-doc-cmd="underline" title="Underline (⌘U)"><u>U</u></button>
            <span class="toolbar-sep"></span>
            <button class="ghost-button compact" type="button" data-doc-block="h1" title="Heading 1">H1</button>
            <button class="ghost-button compact" type="button" data-doc-block="h2" title="Heading 2">H2</button>
            <button class="ghost-button compact" type="button" data-doc-block="p" title="Paragraph">¶</button>
            <button class="ghost-button compact" type="button" data-doc-cmd="insertUnorderedList" title="Bulleted list">•</button>
            <button class="ghost-button compact" type="button" data-doc-cmd="insertOrderedList" title="Numbered list">1.</button>
          </div>
          <div class="drafting-toolbar-right">
            <span class="muted drafting-savestate" data-drafting-savestate>Saved</span>
            <button class="ghost-button compact" type="button" data-doc-copy>${ICON.copy}<span>Copy HTML</span></button>
            <button class="ghost-button compact" type="button" data-doc-download>${ICON.download}<span>Download .doc</span></button>
            <button class="ghost-button compact danger" type="button" data-doc-reset title="Reset to blank template">Reset</button>
          </div>
        </div>
        <div class="drafting-grid">
          <section class="drafting-doc-card card">
            <div class="drafting-editor" contenteditable="true" data-drafting-editor spellcheck="true">${draft.html}</div>
          </section>
          <aside class="drafting-chat-card card">
            <header class="drafting-chat-head">
              <h3>Ask the agent to edit the draft</h3>
              <span class="muted">Sopal will rewrite the document on the left.</span>
            </header>
            <div class="drafting-chat-stream" data-drafting-chat-stream>
              ${draft.chat.messages.length === 0
                ? `<div class="empty-state"><strong>Tell Sopal what to change.</strong><p>Try "Set the claimed amount to $487,250", "Add a section reserving rights to delay costs", or "Tighten the language on the statutory endorsement".</p></div>`
                : draft.chat.messages.map((m) => renderMessage(m.role, m.content, m.role === "assistant")).join("")}
            </div>
            <form class="drafting-chat-form" data-drafting-chat-form>
              <div class="composer-row">
                <textarea class="text-area auto-grow" name="message" rows="2" placeholder="Tell Sopal what to change in the draft…"></textarea>
                <button class="send-button" type="submit" aria-label="Send">${ICON.send}</button>
              </div>
              <div class="composer-meta">
                <label class="check"><input type="checkbox" name="useContext" ${ctxCount ? "checked" : "disabled"}><span>Project context (${ctxCount})</span></label>
                <span class="muted kbd-hint">⌘ / Ctrl + Enter to apply</span>
              </div>
            </form>
          </aside>
        </div>
      </div>
    `;
  }

  function bindDraftingWorkspace(project, agentKey, draftOnly) {
    const editor = document.querySelector("[data-drafting-editor]");
    const saveState = document.querySelector("[data-drafting-savestate]");
    const stream = document.querySelector("[data-drafting-chat-stream]");
    const form = document.querySelector("[data-drafting-chat-form]");
    if (!editor || !form) return;

    const draft = getProjectDraft(project, agentKey);
    let saveTimer = null;
    function persist() {
      draft.html = editor.innerHTML;
      draft.updatedAt = Date.now();
      saveProject(project);
      if (saveState) saveState.textContent = "Saved";
    }
    function scheduleSave() {
      if (saveState) saveState.textContent = "Saving…";
      clearTimeout(saveTimer);
      saveTimer = setTimeout(persist, 600);
    }

    editor.addEventListener("input", scheduleSave);
    editor.addEventListener("blur", () => { clearTimeout(saveTimer); persist(); });

    // Clean-paste — strip the inline style soup that browsers pull from Word /
    // Google Docs / web pages. We keep the structural HTML (headings, lists,
    // tables, bold / italic / underline) and drop everything else.
    editor.addEventListener("paste", (event) => {
      event.preventDefault();
      const cd = event.clipboardData || window.clipboardData;
      if (!cd) return;
      const html = cd.getData("text/html");
      const text = cd.getData("text/plain");
      const sanitised = html ? sanitisePastedHtml(html) : escapeHtml(text || "").replace(/\n/g, "<br>");
      document.execCommand("insertHTML", false, sanitised);
      scheduleSave();
    });

    // Toolbar — uses the legacy execCommand API, which is sufficient for
    // basic Bold/Italic/Underline/lists in a contenteditable div. Block-type
    // changes (H1/H2/P) use formatBlock.
    document.querySelectorAll("[data-doc-cmd]").forEach((btn) => btn.addEventListener("click", (e) => {
      e.preventDefault();
      editor.focus();
      document.execCommand(btn.dataset.docCmd, false, null);
      scheduleSave();
    }));
    document.querySelectorAll("[data-doc-block]").forEach((btn) => btn.addEventListener("click", (e) => {
      e.preventDefault();
      editor.focus();
      document.execCommand("formatBlock", false, btn.dataset.docBlock);
      scheduleSave();
    }));

    document.querySelector("[data-doc-copy]")?.addEventListener("click", () => {
      copyText(editor.innerHTML);
    });
    document.querySelector("[data-doc-download]")?.addEventListener("click", () => {
      // Word opens .doc files containing HTML directly. Wrap the editor HTML in
      // a minimal HTML envelope and trigger a download.
      const filename = `${project.name.replace(/[^a-z0-9]+/gi, "-")}-${agentKey}.doc`;
      const blob = new Blob([
        '<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">',
        '<head><meta charset="UTF-8"><title>',
        escapeHtml(project.name),
        '</title></head><body>',
        editor.innerHTML,
        '</body></html>',
      ], { type: "application/msword" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    });
    document.querySelector("[data-doc-reset]")?.addEventListener("click", () => {
      if (!confirm("Reset this draft back to the blank template? The current content will be lost.")) return;
      const tpl = (AGENT_TEMPLATES[agentKey] || "<p>[Start drafting…]</p>").trim();
      editor.innerHTML = tpl;
      draft.html = tpl;
      draft.chat = { messages: [] };
      saveProject(project);
      render();
    });

    const textarea = form.elements.message;
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

      // Persist current editor HTML before sending so anything the user typed
      // since the last autosave is included in the request.
      persist();

      if (stream.querySelector(".empty-state")) stream.innerHTML = "";
      stream.insertAdjacentHTML("beforeend", renderMessage("user", message));
      const placeholderId = `msg-${Math.random().toString(36).slice(2)}`;
      stream.insertAdjacentHTML("beforeend", `
        <div class="message msg-assistant" id="${placeholderId}">
          <div class="message-body"><div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is rewriting the draft…</span></div></div>
        </div>`);
      draft.chat.messages.push({ role: "user", content: message, at: Date.now() });
      textarea.value = "";
      autoGrow(textarea);
      stream.scrollTop = stream.scrollHeight;

      try {
        const useContext = form.elements.useContext?.checked;
        const projectContext = useContext ? projectContextString(project) : "";
        const projectMeta = `Project: ${project.name}\nContract form: ${project.contractForm}${project.reference ? `\nReference: ${project.reference}` : ""}${project.claimant || project.respondent ? `\nParties: ${project.claimant || "(claimant)"} v ${project.respondent || "(respondent)"}` : ""}\nUser is: ${project.userIsParty || "claimant"}`;
        const response = await fetch("/api/sopal-v2/agent/edit-draft", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            agentType: agentKey,
            currentDocumentHtml: draft.html,
            message,
            projectContext: [projectMeta, projectContext].filter(Boolean).join("\n\n---\n\n"),
          }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(describeApiError(data, "Edit failed"));
        const updatedHtml = (data.documentHtml || "").trim();
        if (updatedHtml) {
          editor.innerHTML = updatedHtml;
          draft.html = updatedHtml;
        }
        const summary = (data.summary || "Updated the draft.").trim();
        draft.chat.messages.push({ role: "assistant", content: summary, at: Date.now() });
        draft.updatedAt = Date.now();
        saveProject(project);
        const placeholder = document.getElementById(placeholderId);
        if (placeholder) placeholder.outerHTML = renderMessage("assistant", summary, true);
        stream.scrollTop = stream.scrollHeight;
      } catch (error) {
        const placeholder = document.getElementById(placeholderId);
        if (placeholder) placeholder.outerHTML = `<div class="message msg-assistant"><div class="message-body"><div class="error-banner">${escapeHtml(error.message || "Edit failed")}</div></div></div>`;
        stream.scrollTop = stream.scrollHeight;
      }
    });
  }

  function sanitisePastedHtml(html) {
    // Allowlist of tags we trust inside the drafting editor. Everything else
    // becomes plain text. Strips inline styles, classes, MS Word namespacing,
    // HTML comments. Keeps headings, lists, tables, and basic emphasis.
    const allowedTags = new Set(["P","BR","STRONG","B","EM","I","U","H1","H2","H3","H4","UL","OL","LI","TABLE","THEAD","TBODY","TR","TH","TD","BLOCKQUOTE","CODE"]);
    const tmpl = document.createElement("div");
    tmpl.innerHTML = html;
    function clean(node) {
      const kids = Array.from(node.childNodes);
      for (const c of kids) {
        if (c.nodeType === 1) {
          if (!allowedTags.has(c.tagName)) {
            const replacement = document.createDocumentFragment();
            const inner = (c.textContent || "").trim();
            if (inner) replacement.appendChild(document.createTextNode(inner));
            c.replaceWith(replacement);
            continue;
          }
          for (const attr of Array.from(c.attributes)) c.removeAttribute(attr.name);
          clean(c);
        } else if (c.nodeType === 8) {
          c.remove();
        }
      }
    }
    clean(tmpl);
    return tmpl.innerHTML;
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
      // Drafting workspace: a Word-style editor on the left with a starter
      // template, an AI chat on the right that can rewrite the document.
      setTimeout(() => bindDraftingWorkspace(project, agentKey, draftOnly), 0);
      return PageBody(DraftingWorkspace(project, agentKey, draftOnly));
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

    // Persistent mode tab-strip — visible whether or not a submode is picked,
    // so the user always sees both perspectives and can switch with one click.
    const modeStrip = `
      <div class="mode-strip">
        <span class="mode-strip-label muted">What are you reviewing?</span>
        ${submodes.map((m) => `
          <a class="mode-strip-tab ${submode && m.id === submode.id ? "active" : ""}" href="${modeBaseHref}&submode=${m.id}" data-nav>
            <span class="mode-strip-icon">${m.id === "received" ? ICON.upload : ICON.file}</span>
            <span class="mode-strip-body">
              <strong>I'm ${escapeHtml(m.label.toLowerCase())}</strong>
              <span class="muted">${escapeHtml(m.sub)}</span>
            </span>
          </a>
        `).join("")}
      </div>`;

    if (!submode) {
      return `
        <div class="page-shell review-shell">
          ${head}
          ${modeStrip}
          <p class="muted review-pick-hint">Pick a perspective above. The checks below are tailored to it.</p>
        </div>`;
    }

    return `
      <div class="page-shell review-shell">
        ${head}
        ${modeStrip}
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
                ${renderAnalysis(agentKey, document, analysis, review?.status, review)}
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

  function openPrintAnalysis(agentKey, document, analysis, project) {
    const counts = analysis.counts || { fail: 0, warn: 0, info: 0, pass: 0 };
    const symbol = (s) => s === "pass" ? "✓" : s === "fail" ? "✕" : s === "warn" ? "!" : s === "info" ? "?" : "•";
    const win = window.open("", "_blank");
    if (!win) {
      alert("Could not open print view. Please allow popups for this site and try again.");
      return;
    }
    const safe = (s) => String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const heading = `${AGENT_LABELS[agentKey] || agentKey} review${project ? ` — ${project.name}` : ""}`;
    const today = new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
    const checksHtml = (analysis.checks || []).map((c, i) => `
      <article class="print-check ${c.status || "info"}">
        <h3><span class="print-check-status">${symbol(c.status)}</span> ${safe(c.title || `Check ${i + 1}`)}</h3>
        <div class="print-check-detail">${safe(c.detail || "").split(/\n+/).map((p) => `<p>${p}</p>`).join("")}</div>
      </article>`).join("");
    win.document.write(`
      <!DOCTYPE html><html><head><meta charset="utf-8"><title>${safe(heading)}</title>
      <style>
        :root { color-scheme: light; }
        body { font-family: Inter, -apple-system, "Segoe UI", sans-serif; color: #1a1a1a; max-width: 760px; margin: 32px auto; padding: 0 24px; line-height: 1.55; font-size: 13.5px; }
        h1 { font-size: 22px; margin: 0 0 4px; }
        h2 { font-size: 16px; margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #e8e5e0; }
        h3 { font-size: 14px; margin: 14px 0 6px; }
        .print-meta { color: #6b6b6b; font-size: 12.5px; margin-bottom: 18px; }
        .print-summary { background: #f7f5f1; border-left: 3px solid #1a1a1a; padding: 12px 14px; border-radius: 4px; }
        .print-counts { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 0; font-size: 12px; font-weight: 600; }
        .print-count-pill { padding: 3px 9px; border-radius: 999px; border: 1px solid; }
        .print-count-pill.fail { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
        .print-count-pill.warn { background: #fffbeb; color: #92400e; border-color: #fde68a; }
        .print-count-pill.info { background: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
        .print-count-pill.pass { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
        .print-check { margin: 12px 0 16px; page-break-inside: avoid; }
        .print-check.fail h3 { color: #991b1b; }
        .print-check.warn h3 { color: #92400e; }
        .print-check.info h3 { color: #1e40af; }
        .print-check.pass h3 { color: #047857; }
        .print-check-status { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
        .print-check-detail p { margin: 4px 0; }
        .print-list { padding-left: 20px; margin: 8px 0; }
        .print-list li { margin: 4px 0; }
        .print-foot { margin-top: 36px; padding-top: 12px; border-top: 1px solid #e8e5e0; color: #9b9994; font-size: 11.5px; }
        .print-actions { display: flex; gap: 8px; margin-bottom: 16px; padding: 8px 12px; background: #f7f5f1; border-radius: 6px; }
        .print-actions button { font: inherit; padding: 6px 12px; border-radius: 5px; border: 1px solid #d1d5db; background: #fff; cursor: pointer; }
        @media print { .print-actions { display: none; } body { margin: 0 auto; max-width: none; padding: 0 16px; } }
      </style></head><body>
        <div class="print-actions">
          <button onclick="window.print()">Print</button>
          <button onclick="window.close()">Close</button>
        </div>
        <h1>${safe(heading)}</h1>
        <p class="print-meta">${document?.name ? `<strong>Document:</strong> ${safe(document.name)}<br>` : ""}<strong>Generated:</strong> ${safe(today)}${project?.reference ? ` &middot; <strong>Ref:</strong> ${safe(project.reference)}` : ""}${project?.contractForm ? ` &middot; <strong>Contract:</strong> ${safe(project.contractForm)}` : ""}</p>
        ${analysis.summary ? `<h2>Summary</h2><div class="print-summary">${safe(analysis.summary).split(/\n+/).map((p) => `<p>${p}</p>`).join("")}</div>` : ""}
        <div class="print-counts">
          <span class="print-count-pill fail">${counts.fail || 0} issues</span>
          <span class="print-count-pill warn">${counts.warn || 0} warnings</span>
          <span class="print-count-pill info">${counts.info || 0} need input</span>
          <span class="print-count-pill pass">${counts.pass || 0} passed</span>
        </div>
        ${checksHtml ? `<h2>Checks</h2>${checksHtml}` : ""}
        ${(analysis.recommendations || []).length ? `<h2>Recommended next steps</h2><ol class="print-list">${analysis.recommendations.map((r) => `<li>${safe(r)}</li>`).join("")}</ol>` : ""}
        ${(analysis.missing || []).length ? `<h2>Information to gather</h2><ul class="print-list">${analysis.missing.map((m) => `<li>${safe(m)}</li>`).join("")}</ul>` : ""}
        <p class="print-foot">Generated by Sopal v2 — sopal.com.au</p>
      </body></html>`);
    win.document.close();
  }

  function analysisToMarkdown(agentKey, document, analysis) {
    if (!analysis) return "";
    const project = currentProject();
    const date = new Date().toISOString().slice(0, 10);
    const heading = `${AGENT_LABELS[agentKey] || agentKey} review${project ? ` — ${project.name}` : ""}`;
    const docLine = document?.name ? `**Document:** ${document.name}` : "";
    const counts = analysis.counts || { fail: 0, warn: 0, info: 0, pass: 0 };
    const countLine = `**${counts.fail} issues · ${counts.warn} warnings · ${counts.info} need input · ${counts.pass} passed**`;
    const symbol = (s) => s === "pass" ? "✓" : s === "fail" ? "✕" : s === "warn" ? "!" : s === "info" ? "?" : "•";
    const checks = (analysis.checks || []).map((c, i) => {
      const title = (c.title || `Check ${i + 1}`).trim();
      const detail = (c.detail || "").trim();
      return `### ${symbol(c.status)} ${title}\n\n${detail}`;
    }).join("\n\n");
    const recs = (analysis.recommendations || []).length
      ? `## Recommended next steps\n\n${analysis.recommendations.map((r, i) => `${i + 1}. ${r}`).join("\n")}`
      : "";
    const missing = (analysis.missing || []).length
      ? `## Information to gather\n\n${analysis.missing.map((m) => `- ${m}`).join("\n")}`
      : "";
    return [
      `# ${heading}`,
      [docLine, `Generated ${date}`].filter(Boolean).join("  \n"),
      analysis.summary ? `## Summary\n\n${analysis.summary.trim()}` : "",
      countLine,
      checks ? `## Checks\n\n${checks}` : "",
      recs,
      missing,
      "---\n_Generated by Sopal v2_",
    ].filter(Boolean).join("\n\n");
  }

  function renderAnalysis(agentKey, document, analysis, status, review) {
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
    const md = analysisToMarkdown(agentKey, document, analysis);
    const history = (review && Array.isArray(review.history)) ? review.history : [];
    return `
      <div class="analysis-summary">
        <div class="analysis-summary-row">
          <div class="summary-counts">
            <span class="sc-pill sc-fail"><strong>${counts.fail}</strong> issues</span>
            <span class="sc-pill sc-warn"><strong>${counts.warn}</strong> warnings</span>
            <span class="sc-pill sc-info"><strong>${counts.info}</strong> need input</span>
            <span class="sc-pill sc-pass"><strong>${counts.pass}</strong> passed</span>
          </div>
          <button class="ghost-button compact analysis-copy-btn" type="button" data-copy-text="${attr(md)}" title="Copy this review as markdown">${ICON.copy}<span>Copy as markdown</span></button>
          <button class="ghost-button compact analysis-print-btn" type="button" data-print-analysis title="Open a clean print-friendly view">${ICON.file}<span>Print</span></button>
        </div>
        ${history.length ? `
          <div class="analysis-history-row">
            <span class="muted">Previous runs:</span>
            <select class="select-input compact" data-restore-history>
              <option value="-1" selected>Current run</option>
              ${history.map((h, i) => {
                const c = (h.analysis && h.analysis.counts) || { fail: 0, warn: 0, info: 0, pass: 0 };
                const date = h.savedAt ? new Date(h.savedAt) : null;
                const stamp = date && !isNaN(date) ? date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : `Run ${i + 1}`;
                return `<option value="${i}">${escapeHtml(stamp)} — ${c.fail || 0} issues · ${c.warn || 0} warnings · ${c.pass || 0} passed</option>`;
              }).join("")}
            </select>
          </div>
        ` : ""}
        ${analysis.summary ? `<blockquote class="analysis-overview">${renderMarkdown(analysis.summary)}</blockquote>` : ""}
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
        <section class="analysis-block analysis-block-recs">
          <header><span class="analysis-block-icon">${ICON.sparkles}</span><h4>Recommended next steps</h4></header>
          <ol>${analysis.recommendations.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ol>
        </section>` : ""}
      ${(analysis.missing || []).length ? `
        <section class="analysis-block analysis-block-missing">
          <header><span class="analysis-block-icon">${ICON.file}</span><h4>Information to gather</h4></header>
          <ul>${analysis.missing.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
        </section>` : ""}
    `;
  }

  // Per-agent quick-question chips that appear in the review chat empty state
  // once an analysis has run. Tailored to each agent so the suggestions feel
  // sharp instead of generic.
  const REVIEW_CHAT_SUGGESTIONS = {
    "payment-claims": [
      "Why is the reference date warning a risk?",
      "What evidence am I missing for the claim amount?",
      "Draft a corrective endorsement under s 68 BIF Act.",
      "Re-run the review pretending the claim is served via email only.",
    ],
    "payment-schedules": [
      "Are any reasons too vague to survive adjudication?",
      "What withholding reasons would I need before s 82(4)?",
      "Draft an itemised reasons table from this schedule.",
      "Compare these reasons against the claim items.",
    ],
    eots: [
      "Is the notice within the contractual deadline?",
      "What programme evidence do I need next?",
      "Draft a reservation of rights for the EOT notice.",
      "Should this also raise variations or delay costs?",
    ],
    variations: [
      "Is there a clear instruction or only a clarification?",
      "What's the time-bar risk on this variation?",
      "Draft the valuation paragraph with a Schedule of Rates fallback.",
      "What evidence supports the cost build-up?",
    ],
    "delay-costs": [
      "Is there overlap with EOT or variation claims?",
      "What's the cleanest causation argument here?",
      "Draft a prolongation cost table by week.",
      "What contemporaneous records do I need?",
    ],
  };

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
    const review = project.reviews && project.reviews[reviewKey];
    const hasAnalysis = !!(review && review.analysis && !review.analysis.error);
    const suggestions = (REVIEW_CHAT_SUGGESTIONS[agentKey] || []).slice(0, 4);

    let emptyHtml;
    if (!hasDocument) {
      emptyHtml = EmptyState("No questions yet.", "Add a document above to give the chat something to anchor to.");
    } else if (!hasAnalysis) {
      emptyHtml = EmptyState("No questions yet.", "Run the analysis on the right, or ask anything about the document below.");
    } else {
      // Doc + analysis present → offer per-agent suggestion chips.
      emptyHtml = `
        <div class="empty-state review-empty-with-suggestions">
          <strong>Ask a follow-up</strong>
          <p>The analysis is grounded in this document — chase a specific item or draft an amendment.</p>
          <div class="chip-row">
            ${suggestions.map((s) => `<button class="chip" type="button" data-chip="${attr(s)}">${escapeHtml(s)}</button>`).join("")}
          </div>
        </div>`;
    }

    const messagesHtml = (chat.messages || []).length
      ? (chat.messages || []).map((m) => renderMessage(m.role, m.content, m.role === "assistant")).join("")
      : emptyHtml;

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
      analysisBody.innerHTML = renderAnalysis(agentKey, r.document, r.analysis, r.status, r);
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
        const r = ensureReview();
        // Preserve the current run as a history entry so the user can compare.
        if (r.analysis && !r.analysis.error) {
          if (!Array.isArray(r.history)) r.history = [];
          r.history.unshift({ analysis: r.analysis, savedAt: Date.now() });
          r.history = r.history.slice(0, 5);
        }
        r.analysis = null;
        r.status = "idle";
        saveProject(project);
        refreshAnalysis();
      });
      document.querySelector("[data-print-analysis]")?.addEventListener("click", () => {
        const r = ensureReview();
        if (!r.analysis || r.analysis.error) return;
        openPrintAnalysis(agentKey, r.document, r.analysis, project);
      });
      document.querySelector("[data-restore-history]")?.addEventListener("change", (event) => {
        const idx = Number(event.target.value);
        if (Number.isNaN(idx) || idx < 0) return;
        const r = ensureReview();
        if (!Array.isArray(r.history) || !r.history[idx]) return;
        // Swap current with the selected history entry so the user can flip back.
        const swap = r.history[idx];
        const prev = r.analysis;
        r.history[idx] = { analysis: prev, savedAt: r.history[idx].savedAt || Date.now() };
        r.analysis = swap.analysis;
        saveProject(project);
        refreshAnalysis();
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
        if (!response.ok) throw new Error(describeApiError(data, "Analysis failed"));
        const parsed = parseStructuredAnalysis(data.answer || "", checks);
        r.analysis = parsed;
        r.status = "done";
        saveProject(project);
        refreshAnalysis();
        // Re-render the chat pane too — the empty-state chips depend on
        // analysis being present.
        refreshChat();
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
      // Suggestion chips: clicking one fills the composer and focuses it.
      pane.querySelectorAll("[data-chip]").forEach((b) => b.addEventListener("click", () => {
        textarea.value = b.dataset.chip;
        autoGrow(textarea);
        textarea.focus();
      }));
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
          if (!response.ok) throw new Error(describeApiError(data, "Reply failed"));
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
          <div class="composer-textarea-wrap">
            <textarea class="text-area auto-grow" name="message" rows="${compact ? 1 : 3}" placeholder="${attr(placeholder)}" aria-label="Message"></textarea>
            <div class="doc-mention-pop" data-mention-pop hidden></div>
          </div>
          <button class="send-button" type="submit" aria-label="Send message">${ICON.send}</button>
          <input type="file" hidden data-chat-file accept=".pdf,.docx,.txt">
        </div>
        <div class="composer-meta">
          <label class="check"><input type="checkbox" name="useContext" ${ctxCount ? (contextOn ? "checked" : "") : "disabled"}><span>Project context (${ctxCount})</span></label>
          <span class="muted" data-chat-file-status></span>
          <span class="muted ref-tags" data-ref-tags></span>
          <span class="muted kbd-hint">@ to reference a doc · ⌘/Ctrl + Enter to send</span>
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
    const mentionPop = form.querySelector("[data-mention-pop]");
    const refTagsEl = form.querySelector("[data-ref-tags]");

    let extractedFile = null;
    const referencedDocs = []; // { name, text, bucket }
    let mention = null; // { atIndex, query, items, sel }

    function refreshRefTags() {
      if (!refTagsEl) return;
      if (!referencedDocs.length) { refTagsEl.innerHTML = ""; return; }
      refTagsEl.innerHTML = `Refs: ${referencedDocs.map((d) => `<span class="ref-tag">@${escapeHtml(d.name)}</span>`).join(" ")}`;
    }

    function projectDocs() {
      const project = getProject(opts.project.id);
      if (!project) return [];
      const out = [];
      (project.contracts || []).forEach((d, i) => out.push({ ...d, bucket: "contracts", _i: i }));
      (project.library || []).forEach((d, i) => out.push({ ...d, bucket: "library", _i: i }));
      return out;
    }

    function closeMention() {
      mention = null;
      mentionPop.hidden = true;
      mentionPop.innerHTML = "";
    }

    function renderMention() {
      if (!mention) return;
      const all = projectDocs();
      const q = mention.query.toLowerCase();
      const items = q
        ? all.filter((d) => (d.name || "").toLowerCase().includes(q)).slice(0, 8)
        : all.slice(0, 8);
      mention.items = items;
      if (mention.sel >= items.length) mention.sel = Math.max(0, items.length - 1);
      mentionPop.hidden = false;
      mentionPop.innerHTML = items.length === 0
        ? `<div class="doc-mention-empty">No matching documents.</div>`
        : items.map((d, i) => `
            <button class="doc-mention-row ${i === mention.sel ? "active" : ""}" type="button" data-mention-pick="${i}">
              <span class="doc-mention-bucket">${d.bucket === "contracts" ? "Contract" : "Library"}</span>
              <span class="doc-mention-name">${escapeHtml(d.name || "Untitled")}</span>
              <span class="doc-mention-meta">${(d.text || "").length.toLocaleString()} chars</span>
            </button>`).join("");
      mentionPop.querySelectorAll("[data-mention-pick]").forEach((el) => el.addEventListener("click", () => {
        pickMention(Number(el.dataset.mentionPick));
      }));
    }

    function pickMention(idx) {
      if (!mention || !mention.items || !mention.items[idx]) return;
      const doc = mention.items[idx];
      const before = textarea.value.slice(0, mention.atIndex);
      const after = textarea.value.slice(textarea.selectionStart);
      const token = `[@${doc.name}] `;
      textarea.value = before + token + after;
      const caret = (before + token).length;
      textarea.setSelectionRange(caret, caret);
      if (!referencedDocs.some((d) => d.name === doc.name && d.bucket === doc.bucket)) {
        referencedDocs.push({ name: doc.name, text: doc.text || "", bucket: doc.bucket });
      }
      refreshRefTags();
      autoGrow(textarea);
      closeMention();
      textarea.focus();
    }

    function maybeStartMention() {
      const caret = textarea.selectionStart;
      const before = textarea.value.slice(0, caret);
      const at = before.lastIndexOf("@");
      if (at < 0) { closeMention(); return; }
      // Only treat as mention if @ is at start or preceded by whitespace
      const prev = at === 0 ? " " : before[at - 1];
      if (!/\s/.test(prev) && at !== 0) { closeMention(); return; }
      const after = before.slice(at + 1);
      if (/\s/.test(after)) { closeMention(); return; }
      mention = mention || { atIndex: at, query: "", items: [], sel: 0 };
      mention.atIndex = at;
      mention.query = after;
      renderMention();
    }

    autoGrow(textarea);
    textarea.addEventListener("input", () => { autoGrow(textarea); maybeStartMention(); });
    textarea.addEventListener("blur", () => { setTimeout(closeMention, 150); });
    textarea.addEventListener("keydown", (event) => {
      if (mention && !mentionPop.hidden) {
        if (event.key === "ArrowDown") { event.preventDefault(); mention.sel = Math.min(mention.sel + 1, mention.items.length - 1); renderMention(); return; }
        if (event.key === "ArrowUp") { event.preventDefault(); mention.sel = Math.max(mention.sel - 1, 0); renderMention(); return; }
        if (event.key === "Enter" && !(event.metaKey || event.ctrlKey)) {
          event.preventDefault();
          pickMention(mention.sel);
          return;
        }
        if (event.key === "Escape") { event.preventDefault(); closeMention(); return; }
      }
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

      const refsSnapshot = referencedDocs.slice();

      // If the chat was empty we need to morph the empty composer into the active layout.
      if (wasEmpty) {
        pane.innerHTML = renderActiveChat({ ...opts, contextDefaultOn: form.elements.useContext.checked }, chat);
        bindChatForm(pane, opts);
        return continueGeneration(pane, opts, message, form.elements.useContext.checked, extractedFile, refsSnapshot);
      }

      // Active layout already in DOM — append messages directly.
      const placeholderId = `msg-${Math.random().toString(36).slice(2)}`;
      messages.insertAdjacentHTML("beforeend", renderMessage("user", message));
      messages.insertAdjacentHTML("beforeend", `
        <div class="message msg-assistant" id="${placeholderId}">
          <div class="message-body"><div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is working…</span></div></div>
        </div>`);
      textarea.value = "";
      referencedDocs.length = 0;
      refreshRefTags();
      autoGrow(textarea);
      scrollToBottom(messageArea);

      try {
        const data = await callAi(opts, message, form.elements.useContext.checked, extractedFile, project, refsSnapshot);
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

  async function continueGeneration(pane, opts, message, useContext, extractedFile, refs) {
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
      const data = await callAi(opts, message, useContext, extractedFile, project, refs);
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

  async function callAi(opts, message, useContext, extractedFile, project, refs) {
    const projectContext = project
      ? (useContext ? projectContextString(project) : projectContextString(project, { pinnedOnly: true }))
      : "";
    const projectMeta = project ? `Project: ${project.name}\nContract form: ${project.contractForm}${project.reference ? `\nReference: ${project.reference}` : ""}${project.claimant || project.respondent ? `\nParties: ${project.claimant || "(claimant)"} v ${project.respondent || "(respondent)"}` : ""}\nUser is: ${project.userIsParty || "claimant"}` : "";
    const refsBlock = (refs || []).length
      ? "Referenced documents (the user @-mentioned these):\n\n" + refs.map((d) => `[@${d.name}]\n${(d.text || "").slice(0, 18000)}`).join("\n\n---\n\n")
      : "";
    const fullContext = [projectMeta, projectContext, refsBlock].filter(Boolean).join("\n\n---\n\n");
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
    if (!response.ok) throw new Error(describeApiError(data, "AI request failed"));
    return data;
  }

  function enrichCitations(html) {
    const project = currentProject();
    if (!project) return html;
    // Build a name → {bucket, index} lookup once per call.
    const lookup = new Map();
    (project.contracts || []).forEach((d, i) => { if (d.name) lookup.set(d.name.toLowerCase(), { bucket: "contracts", index: i, name: d.name }); });
    (project.library || []).forEach((d, i) => { if (d.name) lookup.set(d.name.toLowerCase(), { bucket: "library", index: i, name: d.name }); });
    if (lookup.size === 0) return html;
    return html.replace(/\[@([^\]]{1,120})\]/g, (full, raw) => {
      const key = raw.trim().toLowerCase();
      const hit = lookup.get(key);
      if (!hit) return `<span class="citation-chip citation-unknown">@${escapeHtml(raw)}</span>`;
      return `<button type="button" class="citation-chip" data-doc-preview="${attr(project.id)}:${attr(hit.bucket)}:${hit.index}" title="Open ${escapeHtml(hit.name)}">@${escapeHtml(hit.name)}</button>`;
    });
  }

  function renderMessage(role, content, withActions) {
    if (role === "user") {
      const safe = escapeHtml(content || "").replace(/\n/g, "<br>");
      return `<div class="message msg-user"><div class="bubble">${enrichCitations(safe)}</div></div>`;
    }
    const project = currentProject();
    return `<div class="message msg-assistant">
      <div class="message-body">
        <div class="md">${enrichCitations(renderMarkdown(content || ""))}</div>
        ${withActions ? `<div class="message-actions">
          <button class="ghost-button compact" type="button" data-copy-text="${attr(content || "")}">${ICON.copy}<span>Copy</span></button>
          ${project ? `<button class="ghost-button compact" type="button" data-save-msg-as-note="${attr(project.id)}" data-msg-text="${attr(content || "")}" title="Append this message to the project notes">${ICON.layers}<span>Save as note</span></button>` : ""}
        </div>` : ""}
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

  function openDocPreview(projectId, bucket, index) {
    const project = getProject(projectId);
    if (!project) return;
    const doc = (project[bucket] || [])[index];
    if (!doc) return;
    const dest = bucket === "contracts" ? `/sopal-v2/projects/${project.id}/contract` : `/sopal-v2/projects/${project.id}/library`;
    const destLabel = bucket === "contracts" ? "Open in Contract" : "Open in Project Library";
    let editing = false;
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <aside class="doc-drawer" role="dialog" aria-modal="true" aria-labelledby="docDrawerTitle">
            <header class="doc-drawer-head">
              <div class="doc-drawer-head-text">
                <p class="doc-drawer-eyebrow">${bucket === "contracts" ? "Contract" : "Library"}${doc.pinned ? ` · <span class="pin-badge">PINNED</span>` : ""}</p>
                ${editing
                  ? `<input class="text-input doc-drawer-name-input" type="text" data-doc-edit-name value="${attr(doc.name || "")}" placeholder="Document name">`
                  : `<h2 id="docDrawerTitle">${escapeHtml(doc.name || "Untitled")}</h2>`}
                <p class="doc-drawer-meta muted">${escapeHtml(formatDocMeta(doc) || "—")}${doc.source ? ` · ${escapeHtml(doc.source)}` : ""}</p>
                <div class="doc-tag-row" data-tag-row>
                  ${(doc.tags || []).map((t) => `<span class="doc-tag">${escapeHtml(t)}<button class="doc-tag-x" type="button" data-doc-tag-remove="${attr(t)}" aria-label="Remove tag">×</button></span>`).join("")}
                  <select class="select-input compact doc-tag-add" data-doc-tag-add>
                    <option value="">+ Add tag</option>
                    ${["RFI", "Variation", "Notice", "Programme", "Schedule", "Correspondence", "Other"].filter((t) => !(doc.tags || []).includes(t)).map((t) => `<option value="${attr(t)}">${escapeHtml(t)}</option>`).join("")}
                  </select>
                </div>
              </div>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </header>
            <div class="doc-drawer-body">
              ${editing
                ? `<textarea class="text-area doc-drawer-text-input" data-doc-edit-text rows="20" placeholder="Paste or edit the document text">${escapeHtml(doc.text || "")}</textarea>`
                : doc.text
                  ? `<pre class="doc-drawer-text">${escapeHtml(doc.text)}</pre>`
                  : `<p class="muted">This document has no text content.</p>`}
            </div>
            <footer class="doc-drawer-foot">
              ${editing ? `
                <button class="ghost-button compact" type="button" data-doc-edit-cancel>Cancel</button>
                <button class="dark-button compact" type="button" data-doc-edit-save>Save changes</button>
              ` : `
                <button class="ghost-button compact ${doc.pinned ? "doc-pin-on" : ""}" type="button" data-doc-pin title="${doc.pinned ? "Unpin from chat context" : "Pin so this doc is always sent to chat as context"}">${ICON.layers}<span>${doc.pinned ? "Unpin" : "Pin"}</span></button>
                <button class="ghost-button compact" type="button" data-copy-text="${attr(doc.text || "")}">${ICON.copy}<span>Copy text</span></button>
                <button class="ghost-button compact" type="button" data-doc-edit-start>${ICON.settings}<span>Edit</span></button>
                <a class="ghost-button compact" href="${dest}" data-doc-drawer-open data-nav>${ICON.arrowUpRight}<span>${destLabel}</span></a>
              `}
            </footer>
          </aside>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        rootEl.querySelector("[data-doc-drawer-open]")?.addEventListener("click", () => { modal = null; });
        rootEl.querySelector("[data-doc-edit-start]")?.addEventListener("click", () => { editing = true; render(); });
        rootEl.querySelector("[data-doc-edit-cancel]")?.addEventListener("click", () => { editing = false; render(); });
        rootEl.querySelector("[data-doc-pin]")?.addEventListener("click", () => {
          const proj = getProject(projectId);
          const target = proj && (proj[bucket] || [])[index];
          if (!proj || !target) return;
          target.pinned = !target.pinned;
          target.updatedAt = new Date().toISOString();
          saveProject(proj);
          Object.assign(doc, target);
          render();
        });
        rootEl.querySelectorAll("[data-doc-tag-remove]").forEach((b) => b.addEventListener("click", () => {
          const proj = getProject(projectId);
          const target = proj && (proj[bucket] || [])[index];
          if (!proj || !target) return;
          target.tags = (target.tags || []).filter((t) => t !== b.dataset.docTagRemove);
          target.updatedAt = new Date().toISOString();
          saveProject(proj);
          Object.assign(doc, target);
          render();
        }));
        rootEl.querySelector("[data-doc-tag-add]")?.addEventListener("change", (event) => {
          const tag = event.target.value;
          if (!tag) return;
          const proj = getProject(projectId);
          const target = proj && (proj[bucket] || [])[index];
          if (!proj || !target) return;
          target.tags = Array.from(new Set([...(target.tags || []), tag]));
          target.updatedAt = new Date().toISOString();
          saveProject(proj);
          Object.assign(doc, target);
          render();
        });
        rootEl.querySelector("[data-doc-edit-save]")?.addEventListener("click", () => {
          const proj = getProject(projectId);
          const target = proj && (proj[bucket] || [])[index];
          if (!proj || !target) { modal = null; render(); return; }
          const newName = (rootEl.querySelector("[data-doc-edit-name]")?.value || "").trim();
          const newText = rootEl.querySelector("[data-doc-edit-text]")?.value || "";
          target.name = newName || target.name;
          target.text = newText;
          target.updatedAt = new Date().toISOString();
          saveProject(proj);
          editing = false;
          // Refresh the closed-over `doc` reference for the next render so
          // the view mode shows the new content.
          Object.assign(doc, target);
          render();
        });
        document.addEventListener("keydown", function handler(ev) {
          if (ev.key === "Escape") { document.removeEventListener("keydown", handler); close(); }
        });
      },
    };
    render();
  }

  function openSaveCalcModal(payload) {
    const projects = projectList();
    if (!projects.length) return;
    const defaultName = payload.title || "Calculation";
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>Save calculation to project</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <form class="modal-body" data-save-calc-form>
              <label class="span-2">Project<select class="select-input" name="projectId" required>
                ${projects.map((p) => `<option value="${attr(p.id)}" ${p.id === store.currentProjectId ? "selected" : ""}>${escapeHtml(p.name)}</option>`).join("")}
              </select></label>
              <label class="span-2">Library item name<input class="text-input" name="name" value="${attr(defaultName)}" required></label>
              <p class="muted">The calculation summary will be saved as a library item in the selected project.</p>
              <div class="modal-actions">
                <button class="ghost-button" type="button" data-modal-close>Cancel</button>
                <button class="dark-button" type="submit">Save</button>
              </div>
            </form>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        rootEl.querySelector("[data-save-calc-form]")?.addEventListener("submit", (event) => {
          event.preventDefault();
          const data = Object.fromEntries(new FormData(event.currentTarget).entries());
          const project = getProject(data.projectId);
          if (!project) { close(); return; }
          project.library.push({
            name: data.name || defaultName,
            text: payload.body || "",
            source: payload.kind === "interest" ? "interest-calculator" : "due-date-calculator",
            addedAt: new Date().toISOString(),
            tags: ["Calculation"],
          });
          saveProject(project);
          modal = null;
          render();
        });
      },
    };
    render();
  }

  function saveMessageAsNote(projectId, text, btn) {
    const project = getProject(projectId);
    if (!project || !text) return;
    const stamp = new Date().toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    const block = `\n\n--- Note saved ${stamp} ---\n${text}`;
    project.notes = (project.notes || "") + block;
    saveProject(project);
    if (btn) {
      const original = btn.innerHTML;
      btn.innerHTML = `${ICON.layers}<span>Saved</span>`;
      setTimeout(() => { btn.innerHTML = original; }, 1100);
    }
  }

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
              <label>Project name<input class="text-input" name="name" required value="${attr(editing?.name || "")}" placeholder="Project name"></label>
              <div class="row-2">
                <label>Claimant<input class="text-input" name="claimant" value="${attr(editing?.claimant || "")}" placeholder="Claimant name"></label>
                <label>Respondent<input class="text-input" name="respondent" value="${attr(editing?.respondent || "")}" placeholder="Respondent name"></label>
              </div>
              <div class="row-2">
                <label>Contract form
                  <select class="select-input" name="contractForm">
                    ${CONTRACT_FORMS.map((f) => `<option value="${attr(f)}" ${editing && editing.contractForm === f ? "selected" : ""}>${escapeHtml(f)}</option>`).join("")}
                  </select>
                </label>
                <label>Reference / contract no.<input class="text-input" name="reference" value="${attr(editing?.reference || "")}" placeholder="Reference or contract no."></label>
              </div>
              <div class="row-2">
                <label class="span-all">Category
                  <select class="select-input" name="category">
                    ${PROJECT_CATEGORIES.map((c) => `<option value="${attr(c)}" ${(editing?.category || "Head contract") === c ? "selected" : ""}>${escapeHtml(c)}</option>`).join("")}
                  </select>
                </label>
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
              category: data.category || project.category || "Other",
            });
            saveProject(project);
            modal = null;
            render();
          } else {
            const project = createProject(data);
            project.category = data.category || "Head contract";
            saveProject(project);
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

  /* ---------- Research Agent (project-less chat) ---------- */

  // Project-less, non-persistent chat backed by /api/sopal-v2/research. Stores
  // its own message stack in localStorage under store.researchChat so the
  // conversation survives reloads but doesn't pollute the project list.

  // Jurisdictions Sopal can scope the research agent to. Only QLD has full
  // legislation + decision-corpus support today; the others surface a banner
  // that explains they're degraded ("general knowledge only").
  const JURISDICTIONS = [
    { id: "qld", label: "QLD", full: "Queensland", supported: true },
    { id: "nsw", label: "NSW", full: "New South Wales", supported: false },
    { id: "vic", label: "VIC", full: "Victoria", supported: false },
    { id: "wa", label: "WA", full: "Western Australia", supported: false },
    { id: "sa", label: "SA", full: "South Australia", supported: false },
  ];

  function getResearchJurisdiction() {
    const id = (store.researchJurisdiction || "qld").toLowerCase();
    return JURISDICTIONS.find((j) => j.id === id) || JURISDICTIONS[0];
  }
  function setResearchJurisdiction(id) {
    store.researchJurisdiction = id;
    saveStore();
  }

  function getResearchChat() {
    if (!store.researchChat || !Array.isArray(store.researchChat.messages)) {
      store.researchChat = { messages: [] };
    }
    return store.researchChat;
  }

  function ResearchAgentPage() {
    const chat = getResearchChat();
    const jur = getResearchJurisdiction();
    setTimeout(bindResearchAgent, 0);
    const isEmpty = chat.messages.length === 0;
    const jurisdictionPicker = `
      <div class="jurisdiction-bar">
        <span class="muted">Jurisdiction</span>
        <div class="jurisdiction-tabs" role="tablist">
          ${JURISDICTIONS.map((j) => `
            <button class="jurisdiction-tab ${j.id === jur.id ? "active" : ""} ${j.supported ? "" : "limited"}" type="button" data-jurisdiction="${attr(j.id)}" title="${j.full}${j.supported ? "" : " — limited support (general knowledge only)"}">
              ${escapeHtml(j.label)}${j.supported ? "" : '<span class="jurisdiction-tag" aria-label="Limited support">·</span>'}
            </button>`).join("")}
        </div>
        ${jur.supported ? "" : `<span class="jurisdiction-warn muted">${escapeHtml(jur.full)} sources aren't yet integrated. Answers rely on general knowledge only — verify against the local act before relying.</span>`}
      </div>`;
    return PageBody(`
      <div class="page-shell chat-shell">
        <div class="chat-page-head">
          <div>
            <h1 class="page-title">Research agent</h1>
            <p class="page-sub">Ask construction-law / SOPA research questions. No project required.</p>
          </div>
          ${chat.messages.length ? `<button class="ghost-button compact" type="button" data-clear-research>${ICON.trash}<span>Clear conversation</span></button>` : ""}
        </div>
        ${jurisdictionPicker}
        <div class="chat-layout">
          <section class="chat-pane" data-chat-pane>
            ${isEmpty ? `
              <div class="chat-empty">
                <h2 class="chat-empty-title">Sopal research agent</h2>
                <p class="chat-empty-sub">Ask anything about ${escapeHtml(jur.full)} construction law, SOPA, or specific decisions.</p>
                ${standaloneComposer({ placeholder: "Ask a research question…", compact: false })}
              </div>
            ` : `
              <div class="chat-stream-wrap">
                <div class="chat-stream" data-message-area>
                  <div class="message-stack" data-messages>
                    ${chat.messages.map((m) => renderMessage(m.role, m.content, m.role === "assistant")).join("")}
                  </div>
                </div>
              </div>
              ${standaloneComposer({ placeholder: "Reply…", compact: true })}
            `}
          </section>
        </div>
      </div>
    `);
  }

  // Composer used by research agent + standalone reviewers (no project context
  // checkbox, since these tools never have a project).
  function standaloneComposer(opts) {
    const cls = opts.compact ? "composer-active" : "composer-card";
    return `
      <form class="${cls}" data-research-form>
        <div class="composer-row">
          <textarea class="text-area auto-grow" name="message" rows="${opts.compact ? 1 : 3}" placeholder="${attr(opts.placeholder || "Type a message…")}" aria-label="Message"></textarea>
          <button class="send-button" type="submit" aria-label="Send">${ICON.send}</button>
        </div>
        <div class="composer-meta">
          <span class="muted kbd-hint">⌘ / Ctrl + Enter to send</span>
        </div>
      </form>`;
  }

  function bindResearchAgent() {
    const pane = document.querySelector("[data-chat-pane]");
    if (!pane) return;
    const form = pane.querySelector("[data-research-form]");
    if (!form) return;
    const textarea = form.elements.message;
    autoGrow(textarea);
    textarea.addEventListener("input", () => autoGrow(textarea));
    textarea.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    document.querySelectorAll("[data-jurisdiction]").forEach((b) => b.addEventListener("click", () => {
      const id = b.dataset.jurisdiction;
      if (!id || id === getResearchJurisdiction().id) return;
      setResearchJurisdiction(id);
      render();
    }));
    document.querySelector("[data-clear-research]")?.addEventListener("click", () => {
      if (!confirm("Clear the research conversation?")) return;
      store.researchChat = { messages: [] };
      saveStore();
      render();
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = textarea.value.trim();
      if (!message) return;
      const chat = getResearchChat();
      const wasEmpty = chat.messages.length === 0;
      chat.messages.push({ role: "user", content: message, at: Date.now() });
      saveStore();

      if (wasEmpty) {
        // Re-render so the empty composer is replaced by the active stream.
        render();
        // Continue generation against the just-rendered stream.
        return continueResearchGeneration(message);
      }

      const messages = pane.querySelector("[data-messages]");
      const messageArea = pane.querySelector("[data-message-area]");
      const placeholderId = `msg-${Math.random().toString(36).slice(2)}`;
      messages.insertAdjacentHTML("beforeend", renderMessage("user", message));
      messages.insertAdjacentHTML("beforeend", `
        <div class="message msg-assistant" id="${placeholderId}">
          <div class="message-body"><div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is researching…</span></div></div>
        </div>`);
      textarea.value = "";
      autoGrow(textarea);
      scrollToBottom(messageArea);

      try {
        const data = await callResearch(message);
        chat.messages.push({ role: "assistant", content: data.answer || "", at: Date.now() });
        saveStore();
        const placeholder = document.getElementById(placeholderId);
        if (placeholder) placeholder.outerHTML = renderMessage("assistant", data.answer || "", true);
        scrollToBottom(messageArea);
      } catch (error) {
        const placeholder = document.getElementById(placeholderId);
        if (placeholder) placeholder.outerHTML = `<div class="message msg-assistant"><div class="message-body"><div class="error-banner">${escapeHtml(error.message || "Request failed")}</div></div></div>`;
        scrollToBottom(messageArea);
      }
    });
  }

  async function continueResearchGeneration(message) {
    const pane = document.querySelector("[data-chat-pane]");
    if (!pane) return;
    const messages = pane.querySelector("[data-messages]");
    const messageArea = pane.querySelector("[data-message-area]");
    const placeholderId = `msg-${Math.random().toString(36).slice(2)}`;
    if (messages) {
      messages.insertAdjacentHTML("beforeend", `
        <div class="message msg-assistant" id="${placeholderId}">
          <div class="message-body"><div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is researching…</span></div></div>
        </div>`);
      scrollToBottom(messageArea);
    }
    try {
      const data = await callResearch(message);
      const chat = getResearchChat();
      chat.messages.push({ role: "assistant", content: data.answer || "", at: Date.now() });
      saveStore();
      const placeholder = document.getElementById(placeholderId);
      if (placeholder) placeholder.outerHTML = renderMessage("assistant", data.answer || "", true);
      scrollToBottom(messageArea);
    } catch (error) {
      const placeholder = document.getElementById(placeholderId);
      if (placeholder) placeholder.outerHTML = `<div class="message msg-assistant"><div class="message-body"><div class="error-banner">${escapeHtml(error.message || "Request failed")}</div></div></div>`;
      scrollToBottom(messageArea);
    }
  }

  async function callResearch(message) {
    // Send the prior turns so the model can follow up. Capped to keep payloads
    // sane; we trim back to the most-recent ~20 messages. Jurisdiction is
    // included so the server prompt can scope the answer (and warn loudly when
    // the chosen jurisdiction isn't fully integrated yet).
    const chat = getResearchChat();
    const jur = getResearchJurisdiction();
    const history = (chat.messages || []).slice(-20).map((m) => ({ role: m.role, content: m.content }));
    const response = await fetch("/api/sopal-v2/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ message, history, jurisdiction: jur.id }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(describeApiError(data, "Research request failed"));
    return data;
  }

  /* ---------- Standalone reviewers (Tools → Payment Claim/Schedule Reviewer) ---------- */

  // Project-less variant of the project review workflow. State is stored in
  // store.standaloneReviews[agentKey] so each reviewer remembers the last
  // uploaded document + analysis between page navigations.
  function getStandaloneReview(agentKey) {
    if (!store.standaloneReviews || typeof store.standaloneReviews !== "object") {
      store.standaloneReviews = {};
    }
    if (!store.standaloneReviews[agentKey]) {
      store.standaloneReviews[agentKey] = { document: null, analysis: null, status: "idle", submode: null };
    }
    return store.standaloneReviews[agentKey];
  }

  function StandaloneReviewerPage(agentKey) {
    if (!REVIEW_CHECKS[agentKey]) return notFoundPage();
    const review = getStandaloneReview(agentKey);
    const submodes = AGENT_REVIEW_MODES[agentKey] || [];
    const params = new URLSearchParams(window.location.search);
    const submodeId = params.get("submode") || review.submode || null;
    const activeSubmode = submodes.find((m) => m.id === submodeId) || null;
    setTimeout(() => bindStandaloneReviewer(agentKey, activeSubmode), 0);

    const baseHref = agentKey === "payment-claims"
      ? "/sopal-v2/tools/payment-claim-reviewer"
      : "/sopal-v2/tools/payment-schedule-reviewer";
    const title = agentKey === "payment-claims" ? "Payment claim reviewer" : "Payment schedule reviewer";
    const sub = agentKey === "payment-claims"
      ? "Upload or paste a payment claim. Sopal runs a structured BIF Act / SOPA review. No project required."
      : "Upload or paste a payment schedule. Sopal runs a structured BIF Act / SOPA review. No project required.";

    const head = `
      <div class="chat-page-head">
        <div>
          <h1 class="page-title">${escapeHtml(title)}</h1>
          <p class="page-sub">${escapeHtml(sub)}</p>
        </div>
      </div>`;

    const modeStrip = `
      <div class="mode-strip">
        <span class="mode-strip-label muted">What are you reviewing?</span>
        ${submodes.map((m) => `
          <a class="mode-strip-tab ${activeSubmode && m.id === activeSubmode.id ? "active" : ""}" href="${baseHref}?submode=${m.id}" data-nav>
            <span class="mode-strip-icon">${m.id === "received" ? ICON.upload : ICON.file}</span>
            <span class="mode-strip-body">
              <strong>I'm ${escapeHtml(m.label.toLowerCase())}</strong>
              <span class="muted">${escapeHtml(m.sub)}</span>
            </span>
          </a>
        `).join("")}
      </div>`;

    if (!activeSubmode) {
      return `
        <div class="page-shell review-shell">
          ${head}
          ${modeStrip}
          <p class="muted review-pick-hint">Pick a perspective above. The checks are tailored to it.</p>
        </div>`;
    }

    return `
      <div class="page-shell review-shell">
        ${head}
        ${modeStrip}
        <div class="review-grid">
          <div class="review-left">
            <section class="card review-doc-card">
              <div class="card-head">
                <h3>Document</h3>
                <button class="link-button small" type="button" data-toggle-paste>${review.document ? "Replace" : "Paste text instead"}</button>
              </div>
              <div class="card-body" data-doc-body>
                ${renderDocumentInput(review.document)}
              </div>
            </section>
            <section class="card review-chat-card">
              <div class="card-head">
                <h3>Ask about this document</h3>
                <span class="muted">${review.analysis ? "Use the analysis on the right as you ask" : "Run an analysis first to ground the chat"}</span>
              </div>
              <div class="review-chat" data-chat-pane>
                ${standaloneReviewChatPane(agentKey, activeSubmode, !!review.document)}
              </div>
            </section>
          </div>
          <aside class="review-right">
            <section class="card review-analysis-card">
              <div class="card-head">
                <h3>Analysis</h3>
                ${review.analysis ? `<button class="ghost-button compact" type="button" data-rerun-analysis>Re-run</button>` : ""}
              </div>
              <div class="card-body" data-analysis-body>
                ${renderAnalysis(agentKey, review.document, review.analysis, review.status, review)}
              </div>
            </section>
          </aside>
        </div>
      </div>
    `;
  }

  function standaloneReviewChatPane(agentKey, submode, hasDocument) {
    const review = getStandaloneReview(agentKey);
    if (!review.chat || !Array.isArray(review.chat.messages)) review.chat = { messages: [] };
    const suggestions = (REVIEW_CHAT_SUGGESTIONS[agentKey] || []).slice(0, 4);
    let emptyHtml;
    if (!hasDocument) {
      emptyHtml = EmptyState("No questions yet.", "Add a document above to give the chat something to anchor to.");
    } else if (!review.analysis) {
      emptyHtml = EmptyState("No questions yet.", "Run the analysis on the right, or ask anything about the document below.");
    } else {
      emptyHtml = `
        <div class="empty-state review-empty-with-suggestions">
          <strong>Ask a follow-up</strong>
          <p>The analysis is grounded in this document — chase a specific item or draft an amendment.</p>
          <div class="chip-row">
            ${suggestions.map((s) => `<button class="chip" type="button" data-chip="${attr(s)}">${escapeHtml(s)}</button>`).join("")}
          </div>
        </div>`;
    }
    const messagesHtml = (review.chat.messages || []).length
      ? review.chat.messages.map((m) => renderMessage(m.role, m.content, m.role === "assistant")).join("")
      : emptyHtml;
    return `
      <div class="message-area review-message-area" data-message-area>
        <div class="message-stack" data-messages>${messagesHtml}</div>
      </div>
      <form class="composer-active review-composer" data-standalone-chat-form>
        <div class="composer-row">
          <textarea class="text-area auto-grow" name="message" rows="1" placeholder="${hasDocument ? "Ask about this document…" : "Add a document above to start the chat."}" ${hasDocument ? "" : "disabled"}></textarea>
          <button class="send-button" type="submit" aria-label="Send" ${hasDocument ? "" : "disabled"}>${ICON.send}</button>
        </div>
        <div class="composer-meta">
          <span class="muted kbd-hint">⌘ / Ctrl + Enter to send</span>
        </div>
      </form>`;
  }

  function bindStandaloneReviewer(agentKey, submode) {
    if (!submode) return;
    const review = getStandaloneReview(agentKey);
    review.submode = submode.id;
    saveStore();
    const docBody = document.querySelector("[data-doc-body]");
    const analysisBody = document.querySelector("[data-analysis-body]");

    function refreshDoc() {
      docBody.innerHTML = renderDocumentInput(review.document);
      bindDocInput();
    }
    function refreshAnalysis() {
      analysisBody.innerHTML = renderAnalysis(agentKey, review.document, review.analysis, review.status, review);
      bindAnalysisActions();
    }
    function refreshChat() {
      const pane = document.querySelector("[data-chat-pane]");
      if (!pane) return;
      pane.innerHTML = standaloneReviewChatPane(agentKey, submode, !!review.document);
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
        if (file) await ingestFile(file);
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
        review.document = { name: "Pasted text", text, source: "pasted", addedAt: new Date().toISOString() };
        review.analysis = null;
        review.status = "idle";
        review.chat = { messages: [] };
        saveStore();
        refreshDoc(); refreshAnalysis(); refreshChat();
      });

      const replace = document.querySelector("[data-toggle-paste]");
      if (replace) {
        replace.onclick = () => {
          if (review.document) {
            review.document = null;
            review.analysis = null;
            review.status = "idle";
            review.chat = { messages: [] };
            saveStore();
            refreshDoc(); refreshAnalysis(); refreshChat();
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
        if (!response.ok) throw new Error(describeApiError(data, "Extraction failed"));
        review.document = { name: data.filename, text: data.text, source: "extracted", addedAt: new Date().toISOString() };
        review.analysis = null;
        review.status = "idle";
        review.chat = { messages: [] };
        saveStore();
        refreshDoc(); refreshAnalysis(); refreshChat();
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
        review.analysis = null; review.status = "idle"; saveStore(); refreshAnalysis();
      });
      const printBtn = analysisBody.querySelector("[data-print-analysis]");
      if (printBtn) printBtn.addEventListener("click", () => {
        if (review.analysis) openPrintAnalysis(agentKey, review.document, review.analysis, null);
      });
    }

    async function runAnalysis() {
      if (!review.document) return;
      review.status = "running";
      saveStore();
      refreshAnalysis();
      const checks = REVIEW_CHECKS[agentKey] || [];
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
            message: `Review the document below.\n\nDocument:\n${review.document.text.slice(0, 60000)}`,
            files: [{ name: review.document.name, characters: review.document.text.length }],
          }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(describeApiError(data, "Analysis failed"));
        review.analysis = parseStructuredAnalysis(data.answer || "", checks);
        review.status = "done";
        saveStore();
        refreshAnalysis();
      } catch (error) {
        review.status = "error";
        review.analysis = { error: error.message || "Analysis failed" };
        saveStore();
        refreshAnalysis();
      }
    }

    function bindReviewChatForm(pane) {
      const form = pane.querySelector("[data-standalone-chat-form]");
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
      pane.querySelectorAll("[data-chip]").forEach((b) => b.addEventListener("click", () => {
        textarea.value = b.dataset.chip;
        autoGrow(textarea);
        textarea.focus();
      }));
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const message = textarea.value.trim();
        if (!message || !review.document) return;
        if (!review.chat || !Array.isArray(review.chat.messages)) review.chat = { messages: [] };
        if (messages.querySelector(".empty-state")) messages.innerHTML = "";
        messages.insertAdjacentHTML("beforeend", renderMessage("user", message));
        const placeholderId = `msg-${Math.random().toString(36).slice(2)}`;
        messages.insertAdjacentHTML("beforeend", `
          <div class="message msg-assistant" id="${placeholderId}">
            <div class="message-body"><div class="thinking-row"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Sopal is working…</span></div></div>
          </div>`);
        review.chat.messages.push({ role: "user", content: message, at: Date.now() });
        textarea.value = "";
        autoGrow(textarea);
        scrollToBottom(messageArea);

        try {
          const docBlock = `Document under review (${AGENT_LABELS[agentKey]} — ${submode.label.toLowerCase()}):\n${review.document.text.slice(0, 60000)}`;
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
              projectContext: docBlock,
              files: [{ name: review.document.name, characters: review.document.text.length }],
            }),
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(describeApiError(data, "Reply failed"));
          review.chat.messages.push({ role: "assistant", content: data.answer || "", at: Date.now() });
          saveStore();
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
      if (parts[1] === "agent") return { crumbs: [{ label: "Research agent" }], body: ResearchAgentPage() };
      return { crumbs: [{ label: "Research" }], body: notFoundPage() };
    }
    if (parts[0] === "tools") {
      if (parts[1] === "due-date-calculator") return { crumbs: [{ label: "Due date calculator" }], body: DueDatePage() };
      if (parts[1] === "interest-calculator") return { crumbs: [{ label: "Interest calculator" }], body: InterestPage() };
      if (parts[1] === "payment-claim-reviewer") return { crumbs: [{ label: "Payment claim reviewer" }], body: StandaloneReviewerPage("payment-claims") };
      if (parts[1] === "payment-schedule-reviewer") return { crumbs: [{ label: "Payment schedule reviewer" }], body: StandaloneReviewerPage("payment-schedules") };
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
      if (sub === "complex") {
        const complexKey = parts[3];
        if (!complexKey || !COMPLEX_AGENT_KEYS.includes(complexKey)) return { crumbs: head.concat([{ label: "Complex agents" }]), body: notFoundPage() };
        if (complexKey === "adjudication-application") {
          return { crumbs: head.concat([{ label: COMPLEX_AGENT_LABELS[complexKey] }]), body: ComplexAdjudicationPage(projectId) };
        }
        return { crumbs: head.concat([{ label: COMPLEX_AGENT_LABELS[complexKey] }]), body: notFoundPage() };
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
    document.querySelectorAll("[data-toggle-collapse]").forEach((el) => el.addEventListener("click", () => setSidebarCollapsed(!sidebarCollapsed)));
    document.querySelectorAll("[data-open-palette]").forEach((el) => el.addEventListener("click", () => openCommandPalette()));
    document.querySelectorAll("[data-toggle-theme]").forEach((el) => el.addEventListener("click", () => setTheme(theme === "dark" ? "light" : "dark")));
    document.querySelectorAll("[data-open-whatsnew]").forEach((el) => el.addEventListener("click", () => openWhatsNew()));
    document.querySelectorAll("[data-toggle-pin-thread]").forEach((el) => el.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      togglePinThread(el.dataset.projectId, el.dataset.togglePinThread);
    }));
    document.querySelectorAll("[data-new-project]").forEach((el) => el.addEventListener("click", () => openProjectModal(null)));
    document.querySelectorAll("[data-import-project]").forEach((input) => input.addEventListener("change", async (event) => {
      const file = event.currentTarget.files && event.currentTarget.files[0];
      event.currentTarget.value = "";
      if (!file) return;
      try {
        const text = await file.text();
        const project = importProjectFromJson(text);
        navigate(`/sopal-v2/projects/${project.id}/overview`);
      } catch (err) {
        alert(`Import failed: ${err.message}`);
      }
    }));
    document.querySelectorAll("[data-doc-preview]").forEach((el) => el.addEventListener("click", (event) => {
      event.preventDefault();
      const [projectId, bucket, indexStr] = el.dataset.docPreview.split(":");
      openDocPreview(projectId, bucket, Number(indexStr));
    }));
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
    document.querySelectorAll("[data-toggle-favourite]").forEach((el) => el.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      toggleProjectFavourite(el.dataset.toggleFavourite);
      render();
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
    // Keep the currently-active sidebar item visible. With Drafting Agents
    // (6) + Complex Agents (1+) the sidebar can overflow on common viewport
    // heights — without this the user lands on a route whose nav item is
    // below the fold.
    requestAnimationFrame(() => {
      const active = document.querySelector(".sidebar-scroll .nav-item.active");
      if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
  }

  function copyText(text) {
    if (navigator.clipboard) navigator.clipboard.writeText(text || "").catch(() => {});
  }

  /* ---------- Cmd+K command palette ---------- */

  function buildPaletteItems() {
    const items = [];
    const project = currentProject();
    const projects = projectList();
    // Actions
    items.push({ section: "Action", label: "New project", hint: "Create a fresh project", run: () => openProjectModal(null) });
    items.push({ section: "Action", label: "Import project (JSON)", hint: "Round-trip a sopal-*.json export", run: () => {
      const inp = document.querySelector("[data-import-project]");
      if (inp) inp.click(); else navigate("/sopal-v2/projects");
    }});

    // Workspace tools
    workspaceNav().forEach((t) => items.push({ section: "Tool", label: t.label, hint: "Workspace tool", run: () => navigate(t.href) }));
    items.push({ section: "Tool", label: "Home", hint: "Sopal v2 home", run: () => navigate("/sopal-v2") });
    items.push({ section: "Tool", label: "Your projects", hint: "All projects", run: () => navigate("/sopal-v2/projects") });

    // Projects
    projects.slice(0, 12).forEach((p) => {
      items.push({ section: "Project", label: `Open ${p.name}`, hint: [p.reference, p.contractForm].filter(Boolean).join(" · ") || "Project overview", run: () => navigate(`/sopal-v2/projects/${p.id}/overview`) });
    });

    // Current project's drafting agents (all draft-only now — go straight
    // into the Word-style editor) and complex agents.
    if (project) {
      DRAFTING_AGENT_KEYS.forEach((key) => {
        items.push({ section: "Drafting agent", label: AGENT_LABELS[key], hint: project.name, run: () => navigate(`/sopal-v2/projects/${project.id}/agents/${key}`) });
      });
      COMPLEX_AGENT_KEYS.forEach((key) => {
        items.push({ section: "Complex agent", label: COMPLEX_AGENT_LABELS[key], hint: project.name, run: () => navigate(`/sopal-v2/projects/${project.id}/complex/${key}`) });
      });
    }

    // Recent decisions
    (store.recentDecisions || []).slice(0, 6).forEach((d) => {
      items.push({ section: "Recent decision", label: d.title || d.id, hint: [d.decisionDate, d.adjudicator].filter(Boolean).join(" · "), run: () => navigate(`/sopal-v2/research/decisions/${encodeURIComponent(d.id)}`) });
    });
    return items;
  }

  function filterPaletteItems(items, query) {
    const q = query.trim().toLowerCase();
    if (!q) return items.slice(0, 30);
    const tokens = q.split(/\s+/);
    return items
      .map((item) => {
        const hay = `${item.section} ${item.label} ${item.hint || ""}`.toLowerCase();
        let score = 0;
        for (const tok of tokens) {
          const idx = hay.indexOf(tok);
          if (idx === -1) return null;
          score += 100 - idx; // earlier match scores higher
        }
        return { ...item, _score: score };
      })
      .filter(Boolean)
      .sort((a, b) => b._score - a._score)
      .slice(0, 30);
  }

  let paletteState = null;

  function openCommandPalette() {
    if (modal) return; // don't stack
    paletteState = { query: "", index: 0, allItems: buildPaletteItems() };
    modal = {
      render: () => {
        const visible = filterPaletteItems(paletteState.allItems, paletteState.query);
        paletteState.visible = visible;
        if (paletteState.index >= visible.length) paletteState.index = Math.max(0, visible.length - 1);
        return `
          <div class="modal-backdrop palette-backdrop" data-modal-backdrop>
            <div class="palette" role="dialog" aria-modal="true">
              <div class="palette-input-row">
                <span class="palette-icon">${ICON.search}</span>
                <input class="palette-input" type="text" data-palette-input placeholder="Search projects, agents, tools, decisions…" value="${attr(paletteState.query)}" autocomplete="off" spellcheck="false">
                <span class="palette-kbd">esc</span>
              </div>
              <ol class="palette-list">
                ${visible.length === 0 ? `<li class="palette-empty">No matches.</li>` : visible.map((it, i) => `
                  <li class="palette-item ${i === paletteState.index ? "active" : ""}" data-palette-index="${i}">
                    <span class="palette-section">${escapeHtml(it.section)}</span>
                    <span class="palette-label">${escapeHtml(it.label)}</span>
                    ${it.hint ? `<span class="palette-hint">${escapeHtml(it.hint)}</span>` : ""}
                  </li>`).join("")}
              </ol>
            </div>
          </div>`;
      },
      bind: (rootEl) => {
        const close = () => { modal = null; paletteState = null; render(); };
        const fire = (item) => {
          if (!item) return;
          modal = null;
          paletteState = null;
          item.run();
        };
        const input = rootEl.querySelector("[data-palette-input]");
        if (input) {
          input.focus();
          input.setSelectionRange(input.value.length, input.value.length);
          input.addEventListener("input", () => {
            paletteState.query = input.value;
            paletteState.index = 0;
            // Re-render only the inner list to keep input focus.
            const drawer = rootEl.querySelector(".palette");
            if (drawer) {
              const tmp = document.createElement("div");
              tmp.innerHTML = modal.render();
              const newPalette = tmp.querySelector(".palette");
              drawer.innerHTML = newPalette.innerHTML;
              const re = drawer.querySelector("[data-palette-input]");
              if (re) {
                re.value = paletteState.query;
                re.focus();
                re.setSelectionRange(re.value.length, re.value.length);
                bindList();
              }
            }
          });
        }
        const bindList = () => {
          rootEl.querySelectorAll("[data-palette-index]").forEach((el) => {
            el.addEventListener("mouseenter", () => {
              paletteState.index = Number(el.dataset.paletteIndex);
              rootEl.querySelectorAll(".palette-item").forEach((n) => n.classList.toggle("active", Number(n.dataset.paletteIndex) === paletteState.index));
            });
            el.addEventListener("click", () => fire(paletteState.visible[Number(el.dataset.paletteIndex)]));
          });
        };
        bindList();
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        document.addEventListener("keydown", function handler(ev) {
          if (!paletteState) { document.removeEventListener("keydown", handler); return; }
          if (ev.key === "Escape") { document.removeEventListener("keydown", handler); close(); return; }
          if (ev.key === "ArrowDown") { ev.preventDefault(); paletteState.index = Math.min(paletteState.index + 1, paletteState.visible.length - 1); rootEl.querySelectorAll(".palette-item").forEach((n) => n.classList.toggle("active", Number(n.dataset.paletteIndex) === paletteState.index)); }
          if (ev.key === "ArrowUp") { ev.preventDefault(); paletteState.index = Math.max(paletteState.index - 1, 0); rootEl.querySelectorAll(".palette-item").forEach((n) => n.classList.toggle("active", Number(n.dataset.paletteIndex) === paletteState.index)); }
          if (ev.key === "Enter") { ev.preventDefault(); fire(paletteState.visible[paletteState.index]); document.removeEventListener("keydown", handler); }
        });
      },
    };
    render();
  }

  /* ---------- In-project search (Cmd+F) ---------- */

  function buildProjectSearchIndex(project) {
    const items = [];
    (project.contracts || []).forEach((d, i) => items.push({
      kind: "contract", bucket: "contracts", index: i, name: d.name || "Untitled",
      hay: `${d.name || ""}\n${d.text || ""}`.toLowerCase(),
      preview: (d.text || "").slice(0, 200),
    }));
    (project.library || []).forEach((d, i) => items.push({
      kind: "library", bucket: "library", index: i, name: d.name || "Untitled",
      hay: `${d.name || ""}\n${d.text || ""}`.toLowerCase(),
      preview: (d.text || "").slice(0, 200),
    }));
    Object.entries(project.chats || {}).forEach(([key, chat]) => {
      if (!chat || !Array.isArray(chat.messages)) return;
      const { label, href } = describeChatKey(project, key);
      chat.messages.forEach((m, i) => {
        if (!m || !m.content) return;
        items.push({
          kind: "chat", chatKey: key, msgIndex: i,
          name: `${label} · ${m.role === "user" ? "you" : "Sopal"}`,
          hay: (m.content || "").toLowerCase(),
          preview: plainPreview(m.content).slice(0, 200),
          href,
        });
      });
    });
    return items;
  }

  function searchProjectIndex(items, query) {
    const q = query.trim().toLowerCase();
    if (!q) return items.slice(0, 30);
    const tokens = q.split(/\s+/);
    return items
      .map((it) => {
        let score = 0;
        for (const tok of tokens) {
          const idx = it.hay.indexOf(tok);
          if (idx === -1) return null;
          score += 100 - Math.min(idx, 99);
        }
        return { ...it, _score: score };
      })
      .filter(Boolean)
      .sort((a, b) => b._score - a._score)
      .slice(0, 30);
  }

  let projectSearchState = null;

  function openProjectSearch() {
    if (modal) return;
    const project = currentProject();
    if (!project) return;
    const items = buildProjectSearchIndex(project);
    projectSearchState = { query: "", index: 0, items, project };
    modal = {
      render: () => {
        const visible = searchProjectIndex(projectSearchState.items, projectSearchState.query);
        projectSearchState.visible = visible;
        if (projectSearchState.index >= visible.length) projectSearchState.index = Math.max(0, visible.length - 1);
        return `
          <div class="modal-backdrop palette-backdrop" data-modal-backdrop>
            <div class="palette" role="dialog" aria-modal="true">
              <div class="palette-input-row">
                <span class="palette-icon">${ICON.search}</span>
                <input class="palette-input" type="text" data-project-search-input placeholder="Search ${escapeHtml(project.name)} — contracts, library, chats…" value="${attr(projectSearchState.query)}" autocomplete="off" spellcheck="false">
                <span class="palette-kbd">esc</span>
              </div>
              <ol class="palette-list">
                ${visible.length === 0 ? `<li class="palette-empty">No matches in this project.</li>` : visible.map((it, i) => `
                  <li class="palette-item ${i === projectSearchState.index ? "active" : ""}" data-project-search-index="${i}">
                    <span class="palette-section">${escapeHtml(it.kind)}</span>
                    <span class="palette-label">${escapeHtml(it.name)}</span>
                    <span class="palette-hint">${escapeHtml((it.preview || "").slice(0, 90))}${(it.preview || "").length > 90 ? "…" : ""}</span>
                  </li>`).join("")}
              </ol>
            </div>
          </div>`;
      },
      bind: (rootEl) => {
        const close = () => { modal = null; projectSearchState = null; render(); };
        const fire = (item) => {
          if (!item) return;
          modal = null;
          projectSearchState = null;
          if (item.kind === "contract" || item.kind === "library") {
            navigate(item.kind === "contract"
              ? `/sopal-v2/projects/${project.id}/contract`
              : `/sopal-v2/projects/${project.id}/library`);
            // Open the doc drawer for that item after navigation
            setTimeout(() => openDocPreview(project.id, item.bucket, item.index), 250);
          } else if (item.kind === "chat" && item.href) {
            navigate(item.href);
          }
        };
        const input = rootEl.querySelector("[data-project-search-input]");
        if (input) {
          input.focus();
          input.setSelectionRange(input.value.length, input.value.length);
          input.addEventListener("input", () => {
            projectSearchState.query = input.value;
            projectSearchState.index = 0;
            const drawer = rootEl.querySelector(".palette");
            if (drawer) {
              const tmp = document.createElement("div");
              tmp.innerHTML = modal.render();
              drawer.innerHTML = tmp.querySelector(".palette").innerHTML;
              const re = drawer.querySelector("[data-project-search-input]");
              if (re) {
                re.value = projectSearchState.query;
                re.focus();
                re.setSelectionRange(re.value.length, re.value.length);
                bindList();
              }
            }
          });
        }
        const bindList = () => {
          rootEl.querySelectorAll("[data-project-search-index]").forEach((el) => {
            el.addEventListener("mouseenter", () => {
              projectSearchState.index = Number(el.dataset.projectSearchIndex);
              rootEl.querySelectorAll(".palette-item").forEach((n) => n.classList.toggle("active", Number(n.dataset.projectSearchIndex) === projectSearchState.index));
            });
            el.addEventListener("click", () => fire(projectSearchState.visible[Number(el.dataset.projectSearchIndex)]));
          });
        };
        bindList();
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        document.addEventListener("keydown", function handler(ev) {
          if (!projectSearchState) { document.removeEventListener("keydown", handler); return; }
          if (ev.key === "Escape") { document.removeEventListener("keydown", handler); close(); return; }
          if (ev.key === "ArrowDown") { ev.preventDefault(); projectSearchState.index = Math.min(projectSearchState.index + 1, projectSearchState.visible.length - 1); rootEl.querySelectorAll(".palette-item").forEach((n) => n.classList.toggle("active", Number(n.dataset.projectSearchIndex) === projectSearchState.index)); }
          if (ev.key === "ArrowUp") { ev.preventDefault(); projectSearchState.index = Math.max(projectSearchState.index - 1, 0); rootEl.querySelectorAll(".palette-item").forEach((n) => n.classList.toggle("active", Number(n.dataset.projectSearchIndex) === projectSearchState.index)); }
          if (ev.key === "Enter") { ev.preventDefault(); fire(projectSearchState.visible[projectSearchState.index]); document.removeEventListener("keydown", handler); }
        });
      },
    };
    render();
  }

  /* ---------- Keyboard shortcut overlay ---------- */

  const SHORTCUTS = [
    { keys: ["⌘/Ctrl", "K"], label: "Open command palette", group: "Navigation" },
    { keys: ["⌘/Ctrl", "F"], label: "Search within current project", group: "Navigation" },
    { keys: ["⌘/Ctrl", "\\"], label: "Collapse / expand sidebar", group: "Navigation" },
    { keys: ["?"], label: "Show this cheat sheet", group: "Navigation" },
    { keys: ["Esc"], label: "Close any modal / drawer / palette", group: "Navigation" },
    { keys: ["⌘/Ctrl", "Enter"], label: "Send chat message from any composer", group: "Chat" },
    { keys: ["@"], label: "Reference a project doc inline (in chat)", group: "Chat" },
    { keys: ["↑", "↓"], label: "Navigate items in palette / search", group: "Lists" },
    { keys: ["Enter"], label: "Open / fire selected item in palette / search", group: "Lists" },
  ];

  function openShortcutOverlay() {
    if (modal) return;
    const grouped = {};
    SHORTCUTS.forEach((s) => { if (!grouped[s.group]) grouped[s.group] = []; grouped[s.group].push(s); });
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal shortcut-modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>Keyboard shortcuts</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <div class="modal-body shortcut-body">
              ${Object.entries(grouped).map(([group, items]) => `
                <section class="shortcut-group">
                  <div class="shortcut-group-title">${escapeHtml(group)}</div>
                  <ul class="shortcut-list">
                    ${items.map((s) => `<li class="shortcut-row"><span class="shortcut-keys">${s.keys.map((k) => `<kbd>${escapeHtml(k)}</kbd>`).join('<span class="shortcut-plus">+</span>')}</span><span class="shortcut-label">${escapeHtml(s.label)}</span></li>`).join("")}
                  </ul>
                </section>`).join("")}
            </div>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        document.addEventListener("keydown", function handler(ev) {
          if (ev.key === "Escape") { document.removeEventListener("keydown", handler); close(); }
        });
      },
    };
    render();
  }

  /* ---------- What's new modal ---------- */

  const WHATS_NEW = [
    { date: "May 2026", title: "Complex Agents — Adjudication Application drafter (NEW)", body: "Multi-stage guided drafter: paste the PC + PS, lock the dispute table, work each item via per-issue-type RFIs, watch the master document assemble live. Three s 79 BIF Act scenarios supported (no schedule / less than claimed / scheduled-but-unpaid). 'Draft all' fires parallel passes for every answered thread. Exports the master, the combined statutory declaration, and the index of supporting evidence — each as its own .doc." },
    { date: "May 2026", title: "AA: definitions panel, ToC, deadline countdown, progress pill", body: "Defined terms picked up across threads propagate to every engine call. Live ToC at the top of the master scrolls only the master pane. Lodgement deadline shows a colour-coded countdown (ok / soon / urgent / overdue). At-a-glance 'X/Y drafted · NN%' progress pill in the header." },
    { date: "May 2026", title: "AA: print preview, copy as Markdown, per-thread artefact drawer", body: "View / export the per-thread evidence index + statutory declaration content. Copy the master as Markdown for pasting into other tools. Print preview opens a clean, styled new window." },
    { date: "May 2026", title: "Drafting agents are draft-only with a Word-style editor", body: "Payment Claims, Payment Schedules, EOTs, Variations, Delay Costs, General Correspondence go straight to a Word-style contenteditable doc with toolbar (B/I/U/H1/H2/¶/lists), preloaded template, autosave, .doc download, and clean-paste from Word." },
    { date: "May 2026", title: "Research Agent: jurisdiction selector (QLD / NSW / VIC / WA / SA)", body: "Pick the active jurisdiction. QLD has full BIF Act framing + decision corpus. Other states show a 'Limited support — general knowledge only' banner and the system prompt warns the model accordingly until per-jurisdiction corpora ship." },
    { date: "May 2026", title: "Pinned context, citations, doc tags", body: "Pin a contract or library doc and it's always sent to chat as context — even with the project-context box unchecked. Assistant replies that mention [@DocName] now render as clickable chips. Tag any doc as RFI / Variation / Notice / Programme and filter the lists by tag." },
    { date: "May 2026", title: "Bulk upload + in-project search", body: "Drag many files into the contract or library list at once — each becomes its own entry. ⌘/Ctrl+F opens a fast project-wide search across contracts, library and chat threads." },
    { date: "May 2026", title: "Project archive + duplicate + JSON export/import", body: "Tuck old projects out of sight without losing them. Clone a project's contract + library into a fresh project. Round-trip the whole project to JSON for backup." },
    { date: "May 2026", title: "Command palette (⌘K) + saved searches + clause splitter", body: "Jump to any project, agent, decision or tool from a single fuzzy palette. Save Decision Search queries as one-click chips. Paste a contract and split it into one entry per Clause / Section automatically." },
    { date: "May 2026", title: "Doc preview drawer + edit-in-drawer + Resume CTA", body: "Open any doc in a side drawer (with Edit / Pin / Copy). The home page surfaces a Resume chip with a one-line preview of where you left off." },
    { date: "May 2026", title: "Copy analysis as markdown + per-message copy + analysis history", body: "One click copies a structured review (summary + checks + recs + gaps) as clean markdown. Every assistant message has a Copy button. Re-run preserves prior runs in a Previous-runs dropdown." },
  ];

  function openWhatsNew() {
    if (modal) return;
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal whatsnew-modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>What's new in Sopal v2</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <div class="modal-body whatsnew-body">
              ${WHATS_NEW.map((entry) => `
                <article class="whatsnew-entry">
                  <header><span class="whatsnew-date">${escapeHtml(entry.date)}</span><h3>${escapeHtml(entry.title)}</h3></header>
                  <p>${escapeHtml(entry.body)}</p>
                </article>`).join("")}
            </div>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); try { localStorage.setItem("sopal-v2-whatsnew-seen", String(Date.now())); } catch (_) {} };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        document.addEventListener("keydown", function handler(ev) {
          if (ev.key === "Escape") { document.removeEventListener("keydown", handler); close(); }
        });
      },
    };
    render();
  }

  /* ---------- Boot ---------- */

  window.addEventListener("popstate", render);
  window.addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && (ev.key === "k" || ev.key === "K")) {
      ev.preventDefault();
      openCommandPalette();
    }
    if ((ev.metaKey || ev.ctrlKey) && (ev.key === "f" || ev.key === "F")) {
      // Only intercept when there's a current project to search.
      if (currentProject()) {
        ev.preventDefault();
        openProjectSearch();
      }
    }
    if ((ev.metaKey || ev.ctrlKey) && ev.key === "\\") {
      ev.preventDefault();
      setSidebarCollapsed(!sidebarCollapsed);
    }
    if ((ev.metaKey || ev.ctrlKey) && ev.shiftKey && (ev.key === "d" || ev.key === "D")) {
      ev.preventDefault();
      setTheme(theme === "dark" ? "light" : "dark");
    }
    if (ev.key === "?" && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
      const tag = (ev.target && ev.target.tagName) || "";
      const isEditable = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (ev.target && ev.target.isContentEditable);
      if (!isEditable) {
        ev.preventDefault();
        openShortcutOverlay();
      }
    }
  });
  document.addEventListener("click", (event) => {
    const copyBtn = event.target.closest("[data-copy-text]");
    if (copyBtn) {
      copyText(copyBtn.dataset.copyText || "");
      const original = copyBtn.innerHTML;
      copyBtn.innerHTML = `${ICON.copy}<span>Copied</span>`;
      setTimeout(() => { copyBtn.innerHTML = original; }, 1100);
    }
    const saveCalc = event.target.closest("[data-save-calc-to-project]");
    if (saveCalc) {
      try {
        const payload = JSON.parse(saveCalc.dataset.calcPayload || "{}");
        openSaveCalcModal(payload);
      } catch (_) {}
    }
    const noteBtn = event.target.closest("[data-save-msg-as-note]");
    if (noteBtn) {
      const projectId = noteBtn.dataset.saveMsgAsNote;
      const text = noteBtn.dataset.msgText || "";
      saveMessageAsNote(projectId, text, noteBtn);
    }
  });

  render();
})();
