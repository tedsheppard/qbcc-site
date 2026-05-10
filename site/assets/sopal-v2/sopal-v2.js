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
  // Which agents are expanded in the sidebar (shows their drafting instances).
  // Keyed by `${projectId}:${agentKey}`. Not persisted: opens fresh per load.
  const sidebarAgentOpen = new Set();
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

  // Firm branding defaults — used wherever the user hasn't set anything
  // themselves. Kept conservative so a brand-new user gets a clean,
  // unbranded look that still passes for legal output rather than a
  // half-applied template.
  function defaultFirmSettings() {
    return {
      name: "",
      letterheadAddress: "",
      footerText: "",
      logoDataUrl: "",
      bodyFont: "serif",          // 'serif' | 'sans' | 'inter'
      pageSize: "a4",             // 'a4' | 'letter'
      accentColour: "#243043",
      headingNumbering: "decimal", // 'decimal' | 'decimal-nested' | 'alpha' | 'roman' | 'none'
    };
  }
  function emptyStore() {
    return { projects: {}, currentProjectId: null, recentDecisions: [], firm: defaultFirmSettings() };
  }
  function loadStore() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
      if (parsed && parsed.projects) {
        if (!Array.isArray(parsed.recentDecisions)) parsed.recentDecisions = [];
        // Backfill firm slot for stores that pre-date the Firm Settings card.
        parsed.firm = { ...defaultFirmSettings(), ...(parsed.firm || {}) };
        return parsed;
      }
    } catch {}
    return emptyStore();
  }
  function saveStore() { localStorage.setItem(STORE_KEY, JSON.stringify(store)); }
  function getFirm() {
    if (!store.firm) store.firm = defaultFirmSettings();
    return store.firm;
  }
  function saveFirm(patch) {
    store.firm = { ...getFirm(), ...(patch || {}) };
    saveStore();
    firmCloudSync.enqueue();
  }

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
    cloudSync.enqueue(p.id);
  }
  function deleteProject(id) {
    delete store.projects[id];
    if (store.currentProjectId === id) store.currentProjectId = projectList()[0]?.id || null;
    saveStore();
    cloudSync.deleteRemote(id);
  }

  // Cloud sync: a thin debounced wrapper over /api/sopal-v2/projects. Only
  // fires when the user is signed in. Last-write-wins on conflict (the SPA
  // is single-user single-device for now). Failures are logged to a queue so
  // the next successful sync covers everything that was pending.
  const cloudSync = (() => {
    const SYNC_DEBOUNCE_MS = 1500;
    const pending = new Set();
    let timer = null;
    function scheduleFlush() {
      if (timer) clearTimeout(timer);
      timer = setTimeout(flush, SYNC_DEBOUNCE_MS);
    }
    async function flush() {
      timer = null;
      if (!window.SopalAuth || window.SopalAuth.state !== "authed") return;
      const ids = Array.from(pending);
      pending.clear();
      for (const id of ids) {
        const project = store.projects[id];
        if (!project) continue;
        try {
          await fetch(`/api/sopal-v2/projects/${encodeURIComponent(id)}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json", ...window.SopalAuth.headers() },
            body: JSON.stringify({ data: project }),
          });
        } catch (_) {
          // Network blip: re-enqueue so the next save picks it back up.
          pending.add(id);
        }
      }
    }
    return {
      enqueue(id) {
        if (!window.SopalAuth || window.SopalAuth.state !== "authed") return;
        pending.add(id);
        scheduleFlush();
      },
      async deleteRemote(id) {
        if (!window.SopalAuth || window.SopalAuth.state !== "authed") return;
        try {
          await fetch(`/api/sopal-v2/projects/${encodeURIComponent(id)}`, {
            method: "DELETE",
            headers: window.SopalAuth.headers(),
          });
        } catch (_) {}
      },
      async pullMissing() {
        // Called once after auth succeeds. Fetches the lightweight project
        // list, and for any project the cloud has but local does not, pulls
        // the full blob and merges it into the local store.
        if (!window.SopalAuth || window.SopalAuth.state !== "authed") return;
        try {
          const r = await fetch("/api/sopal-v2/projects", { headers: window.SopalAuth.headers() });
          if (!r.ok) return;
          const data = await r.json();
          const list = (data && data.projects) || [];
          let pulled = 0;
          for (const meta of list) {
            if (store.projects[meta.id]) continue;
            try {
              const fr = await fetch(`/api/sopal-v2/projects/${encodeURIComponent(meta.id)}`, { headers: window.SopalAuth.headers() });
              if (!fr.ok) continue;
              const full = await fr.json();
              if (full && full.data && typeof full.data === "object") {
                store.projects[meta.id] = full.data;
                pulled += 1;
              }
            } catch (_) {}
          }
          if (pulled) {
            saveStore();
            render();
          }
        } catch (_) {}
      },
      async pushAll() {
        // Manual full-push action. Used when the user wants to seed the
        // cloud copy from the current browser state (after enabling sync
        // on a machine that already has projects).
        if (!window.SopalAuth || window.SopalAuth.state !== "authed") return { pushed: 0 };
        let pushed = 0;
        for (const id of Object.keys(store.projects)) {
          const project = store.projects[id];
          if (!project) continue;
          try {
            const r = await fetch(`/api/sopal-v2/projects/${encodeURIComponent(id)}`, {
              method: "PUT",
              headers: { "Content-Type": "application/json", ...window.SopalAuth.headers() },
              body: JSON.stringify({ data: project }),
            });
            if (r.ok) pushed += 1;
          } catch (_) {}
        }
        return { pushed };
      },
    };
  })();
  window.SopalCloudSync = cloudSync;

  // Firm-wide settings sync. Mirrors the project sync pattern but operates
  // on a single per-user blob, not a list. Debounced so editing the logo
  // / fonts / colours doesn't fire one PUT per keystroke.
  const firmCloudSync = (() => {
    const SYNC_DEBOUNCE_MS = 1500;
    let timer = null;
    let pending = false;
    function scheduleFlush() {
      if (timer) clearTimeout(timer);
      timer = setTimeout(flush, SYNC_DEBOUNCE_MS);
    }
    async function flush() {
      timer = null;
      if (!pending) return;
      pending = false;
      if (!window.SopalAuth || window.SopalAuth.state !== "authed") return;
      try {
        await fetch("/api/sopal-v2/firm", {
          method: "PUT",
          headers: { "Content-Type": "application/json", ...window.SopalAuth.headers() },
          body: JSON.stringify({ data: getFirm() }),
        });
      } catch (_) {
        // Re-arm so the next save still picks the change up.
        pending = true;
      }
    }
    return {
      enqueue() {
        if (!window.SopalAuth || window.SopalAuth.state !== "authed") return;
        pending = true;
        scheduleFlush();
      },
      async pull() {
        if (!window.SopalAuth || window.SopalAuth.state !== "authed") return;
        try {
          const r = await fetch("/api/sopal-v2/firm", { headers: window.SopalAuth.headers() });
          if (!r.ok) return;
          const data = await r.json();
          if (data && data.data && typeof data.data === "object") {
            store.firm = { ...defaultFirmSettings(), ...data.data };
            saveStore();
            render();
          }
        } catch (_) {}
      },
      async push() {
        if (!window.SopalAuth || window.SopalAuth.state !== "authed") return false;
        try {
          const r = await fetch("/api/sopal-v2/firm", {
            method: "PUT",
            headers: { "Content-Type": "application/json", ...window.SopalAuth.headers() },
            body: JSON.stringify({ data: getFirm() }),
          });
          return r.ok;
        } catch (_) { return false; }
      },
    };
  })();
  window.SopalFirmSync = firmCloudSync;

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

  function sidebarDraftingAgentRow(project, agentKey) {
    const base = `/sopal-v2/projects/${project.id}/agents/${agentKey}`;
    const draftBase = `${base}?mode=draft`;
    const instances = getDraftInstances(project, agentKey);
    const expandKey = `${project.id}:${agentKey}`;
    const params = new URLSearchParams(window.location.search);
    const activeIid = params.get("iid");
    const isAgentActive = isActivePrefix(base);
    const expanded = sidebarAgentOpen.has(expandKey) || isAgentActive;
    const draftOnly = DRAFT_ONLY_AGENTS.has(agentKey);
    const landingHref = draftOnly ? draftBase : base;
    return `
      <div class="nav-agent-row">
        <div class="nav-agent-head">
          <a class="nav-item nav-item-sub nav-item-agent ${isAgentActive ? "active" : ""}" href="${landingHref}" data-nav>
            <span class="nav-icon">${ICON.sparkles}</span>
            <span class="nav-label">${escapeHtml(AGENT_LABELS[agentKey])}</span>
            ${instances.length ? `<span class="nav-instance-count">${instances.length}</span>` : ""}
          </a>
          <button class="nav-agent-toggle ${expanded ? "open" : ""}" type="button" data-toggle-agent="${attr(expandKey)}" aria-label="Toggle drafts" title="Show drafts">${ICON.chevRight}</button>
        </div>
        <div class="nav-agent-instances ${expanded ? "open" : ""}">
          ${instances.map((inst) => `
            <a class="nav-instance-item ${activeIid === inst.id ? "active" : ""}" href="${base}?mode=draft&iid=${attr(inst.id)}" data-nav>
              <span class="nav-instance-bullet"></span>
              <span class="nav-instance-text">${escapeHtml(inst.label || defaultDraftLabel(agentKey, 0))}</span>
            </a>`).join("")}
          <button class="nav-instance-add" type="button" data-new-agent-draft="${attr(agentKey)}" title="New draft">
            <span class="nav-instance-bullet new">+</span>
            <span class="nav-instance-text">New draft</span>
          </button>
        </div>
      </div>
    `;
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
              ${DRAFTING_AGENT_KEYS.map((key) => sidebarDraftingAgentRow(project, key)).join("")}
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
          <a class="nav-item small ${isActivePrefix("/sopal-v2/help") ? "active" : ""}" href="/sopal-v2/help" data-nav>
            <span class="nav-icon">${ICON.book}</span>
            <span class="nav-label">Help and support</span>
          </a>
          <a class="nav-item small ${isActivePrefix("/sopal-v2/settings") ? "active" : ""}" href="/sopal-v2/settings" data-nav>
            <span class="nav-icon">${ICON.settings}</span>
            <span class="nav-label">Settings</span>
          </a>
          ${SidebarAuth()}
        </div>
      </aside>
    `;
  }

  // Renders the auth row at the bottom of the sidebar foot. Three states:
  // unknown (initial paint, before /purchase-me lands), guest (no token or
  // 401), authed (purchase-me returned a user). We keep the markup small so
  // the sidebar does not feel front-loaded with auth chrome.
  function SidebarAuth() {
    const a = sopalAuth;
    if (a.state === "authed" && a.user) {
      const display = (a.user.first_name || a.user.last_name)
        ? [a.user.first_name, a.user.last_name].filter(Boolean).join(" ")
        : (a.user.email || "Account");
      return `
        <div class="sidebar-auth signed-in">
          <div class="sidebar-auth-row">
            <span class="sidebar-auth-name" title="${attr(a.user.email || "")}">${escapeHtml(display)}</span>
          </div>
          <div class="sidebar-auth-row">
            <a class="link-button small" href="/account.html" target="_blank" rel="noopener">Account</a>
            <button class="link-button small" type="button" data-sopal-signout>Sign out</button>
          </div>
        </div>`;
    }
    if (a.state === "guest") {
      return `
        <div class="sidebar-auth guest">
          <p>You are using Sopal as a guest. Sign in to keep your work tied to your account.</p>
          <a class="dark-button compact" href="/login?redirect=${encodeURIComponent("/sopal-v2")}">Sign in</a>
        </div>`;
    }
    // Unknown: keep the row mute so the sidebar does not flash a "guest"
    // banner before /purchase-me has had a chance to respond.
    return `<div class="sidebar-auth checking"><span class="muted">Checking sign-in...</span></div>`;
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
    // Contextual greeting based on auth state. Falls back to the generic
    // "Welcome to Sopal v2" until the auth check has resolved.
    const auth = window.SopalAuth;
    const greeting = (() => {
      if (auth && auth.state === "authed" && auth.user) {
        const first = (auth.user.first_name || "").trim();
        const display = first || (auth.user.email ? auth.user.email.split("@")[0] : "");
        if (display) return `Welcome back, ${escapeHtml(display)}`;
      }
      return "Welcome to Sopal v2";
    })();
    return PageBody(`
      <div class="home-shell">
        <section class="home-hero">
          <div class="home-hero-row">
            <div>
              <h2>${greeting}</h2>
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
          <div class="section-head"><h3>Tools</h3><p>Standalone utilities. No project required.</p></div>
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
            <div><h3>Your projects</h3><p>Each project is one construction contract: head contract or subcontract.</p></div>
            <button class="dark-button" type="button" data-new-project>${ICON.plus}<span>New project</span></button>
          </div>
          ${projects.length === 0 ? `
            <div class="card-empty">
              <div class="card-empty-icon">${ICON.file}</div>
              <h4>Create your first project</h4>
              <p>Add the contract details, paste in clauses or upload your contract. Sopal then runs every agent (Payment Claims, EOTs, Adjudication etc.) inside that project's context.</p>
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
        <p class="page-sub">Searches Sopal's adjudication decision database. Results render here, with no jumps to the live site.</p>

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
          <div><h1 class="page-title">${showArchived ? "Archived projects" : "Your projects"}</h1><p class="page-sub">Each project is one construction contract: head contract or subcontract.</p></div>
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
            <p>${categoryFilter ? "Try a different category." : "Give it a name, the parties, the contract form. Then upload or paste the contract; the assistant and every agent will work in that project's context."}</p>
            <div class="card-empty-actions">
              <button class="dark-button" type="button" data-new-project>Create project</button>
              <label class="ghost-button" title="Import a sopal-*.json export">${ICON.upload}<span>Import from JSON</span><input type="file" data-import-project accept="application/json,.json" hidden></label>
            </div>
          </div>`) : `<div class="project-list">${projects.map((p) => projectRow(p, { archived: showArchived })).join("")}</div>`}
        <footer class="storage-footer">
          <div class="storage-bar"><div class="storage-bar-fill ${pct >= 80 ? "high" : ""}" style="width:${pct}%"></div></div>
          <p class="muted">${formatBytes(bytes)} of ~${formatBytes(quotaApprox)} local browser storage used (${pct}%). Sopal v2 stores all project data in this browser only when you are not signed in. Sign in to enable cloud sync to your account.</p>
        </footer>
      </div>
    `);
  }

  function ProjectOverviewPage(projectId) {
    const project = getProject(projectId);
    if (!project) return notFoundPage();
    const allChats = Object.entries(project.chats || {})
      .filter(([, c]) => Array.isArray(c.messages) && c.messages.length > 0);
    // Empty-state quick-start panel. Renders only when the project is fresh
    // (no contracts, no library, no chats). Disappears as soon as the user
    // adds their first piece of content. Helps a first-time user understand
    // the right order of operations without a full tour.
    const isEmptyProject = project.contracts.length === 0
      && project.library.length === 0
      && allChats.length === 0;
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
        ${isEmptyProject ? `
          <section class="card project-quickstart">
            <div class="card-head">
              <h3>Quick start for a fresh project</h3>
              <span class="muted">This panel disappears once you add your first piece of content.</span>
            </div>
            <div class="card-body">
              <ol class="project-quickstart-list">
                <li>
                  <strong>Add the contract.</strong>
                  <span>Paste the relevant clauses or drop the executed PDF. Sopal's agents quote real contract clauses by number when you give them the source.</span>
                  <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/contract" data-nav>Open contract</a>
                </li>
                <li>
                  <strong>Add project library items.</strong>
                  <span>Correspondence, RFIs, programme notes, prior payment claims and schedules. Tag each item so it stays scannable.</span>
                  <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/library" data-nav>Open library</a>
                </li>
                <li>
                  <strong>Try a drafting agent or the assistant.</strong>
                  <span>Open Variations, EOTs, Payment Claims or the project Assistant to sense-check the project context. Your contract and library are sent automatically.</span>
                  <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/assistant" data-nav>Open assistant</a>
                </li>
                <li>
                  <strong>When the dispute is ready, run the Adjudication Application.</strong>
                  <span>Paste the payment claim and the payment schedule into Stage 1 and walk through the five-stage workflow. The end result is a Word-ready master document.</span>
                  <a class="ghost-button compact" href="/sopal-v2/projects/${attr(project.id)}/complex/adjudication-application" data-nav>Open Adjudication Application</a>
                </li>
              </ol>
              <p class="muted project-quickstart-foot">Need more detail? <a href="/sopal-v2/help/getting-started" data-nav>Read the Getting started guide</a> or <a href="/sopal-v2/help" data-nav>browse all help articles</a>.</p>
            </div>
          </section>
        ` : ""}
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
              <textarea class="text-area notes-textarea" data-notes-input rows="5" placeholder="Free-form scratchpad. Chronology, key dates, open questions. Saved automatically.">${escapeHtml(project.notes || "")}</textarea>
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

  // Generic not-found page used across many routes (project, help article,
  // tool, agent). Falls back to the project-list CTA but accepts overrides
  // for callers that have a more specific recovery action.
  function notFoundPage(opts) {
    const o = opts || {};
    const title = o.title || "We could not find that page.";
    const body = o.body || "The link may be stale, or the item may have been deleted.";
    const cta = o.cta || `<a class="ghost-button compact" href="/sopal-v2" data-nav>Back to home</a>`;
    return PageBody(`<div class="page-shell">${EmptyState(title, body, cta)}</div>`);
  }

  function ContextPage(projectId, bucket) {
    const project = getProject(projectId);
    if (!project) return notFoundPage();
    const allItems = project[bucket] || [];
    const params = new URLSearchParams(window.location.search);
    const tagFilter = params.get("tag") || "";
    const items = tagFilter ? allItems.filter((it) => (it.tags || []).includes(tagFilter)) : allItems;
    const allTags = Array.from(new Set(allItems.flatMap((it) => it.tags || []))).sort();
    const labels = bucket === "contracts"
      ? {
          single: "Contract",
          title: "Contract documents",
          helper: "Paste contract clauses or extract text from PDF/DOCX/TXT. The assistant and every agent in this project will see this content.",
          textareaPlaceholder: "Paste contract clauses or terms here. For example: cl 36 (Variations), cl 41 (Default), cl 42 (Payment).",
        }
      : {
          single: "Project document",
          title: "Project library",
          helper: "Paste correspondence, RFIs, claims, schedules, programme notes, or extract them from PDF/DOCX/TXT.",
          textareaPlaceholder: "Paste correspondence, RFIs, claims, schedules, programme notes, or facts.",
        };
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
              <label class="span-2">Paste text<textarea class="text-area" name="text" rows="8" placeholder="${attr(labels.textareaPlaceholder)}"></textarea></label>
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
    // Idempotency guard. bindContextManager fires from a setTimeout in
    // ContextPage; if the page renders twice in quick succession (e.g. a
    // side-effect re-render) the old form was double-bound, so a single
    // click would fire the submit handler twice and add two duplicate
    // entries to project.contracts / project.library. Mark the element on
    // first bind and bail on subsequent calls for the same DOM node.
    if (form._sopalContextBound) return;
    form._sopalContextBound = true;
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
    // Backfill: master-document optional sections + cover-page meta.
    // All of these are FLUID — they only render in the master if they have
    // content. Storage is here so the user can edit them at any time without
    // re-running an engine pass.
    if (!aa.coverMeta || typeof aa.coverMeta !== "object") {
      aa.coverMeta = {
        applicationDate: "",
        ana: "",
        anaReference: "",
        contractDate: "",
        siteAddress: "",
        pcDate: "",
        psDate: "",
        claimantAbn: "",
        claimantAddress: "",
        claimantContact: "",
        claimantEmail: "",
        claimantPhone: "",
        respondentAbn: "",
        respondentAddress: "",
        respondentContact: "",
        respondentEmail: "",
        respondentPhone: "",
      };
    }
    // Backfill optional fields on existing projects so the cover modal and
    // master table render the rows when the user fills them in.
    ["claimantAbn", "respondentAbn", "contractDate", "siteAddress"].forEach((k) => {
      if (aa.coverMeta && typeof aa.coverMeta[k] !== "string") aa.coverMeta[k] = "";
    });
    if (typeof aa.introductionHtml !== "string") aa.introductionHtml = "";
    if (typeof aa.execSummaryHtml !== "string") aa.execSummaryHtml = "";
    if (typeof aa.overarchingHtml !== "string") aa.overarchingHtml = "";
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

  /* ---------- AA snapshots (multi-AA per project, thin archive wrapper) ----------
     The AA workflow internally reads project.complexApps["adjudication-application"]
     directly in many places (engine calls, dispute table, RFI rounds, master doc,
     evidence index). Parameterising every internal function would be a multi-day
     refactor.

     Instead: keep the live AA where it is. Snapshots are NAMED checkpoints kept
     in project.complexApps.snapshots = [{ id, name, savedAt, blob }]. The blob
     is a deep-copy of the live AA at save time. Loading a snapshot deep-copies
     the blob back into the active slot. Existing readers see no shape change.
  */
  function newAASnapshotId() { return `aa_${Math.random().toString(36).slice(2, 10)}`; }
  function getAASnapshots(project) {
    if (!project.complexApps) project.complexApps = {};
    if (!Array.isArray(project.complexApps.snapshots)) project.complexApps.snapshots = [];
    return project.complexApps.snapshots;
  }
  function saveAASnapshot(project, name) {
    const aa = project.complexApps && project.complexApps["adjudication-application"];
    if (!aa) return null;
    const list = getAASnapshots(project);
    const cleanName = String(name || "").trim() || `Snapshot ${list.length + 1}`;
    // JSON deep-copy. AA blobs are pure data (strings, numbers, arrays, plain
    // objects); no functions, no Dates, so JSON round-trip is safe.
    const blob = JSON.parse(JSON.stringify(aa));
    const snapshot = {
      id: newAASnapshotId(),
      name: cleanName,
      savedAt: Date.now(),
      blob,
    };
    list.push(snapshot);
    saveProject(project);
    return snapshot;
  }
  function loadAASnapshot(project, snapshotId) {
    const list = getAASnapshots(project);
    const snap = list.find((s) => s.id === snapshotId);
    if (!snap) return false;
    if (!project.complexApps) project.complexApps = {};
    project.complexApps["adjudication-application"] = JSON.parse(JSON.stringify(snap.blob));
    saveProject(project);
    return true;
  }
  function deleteAASnapshot(project, snapshotId) {
    const list = getAASnapshots(project);
    const idx = list.findIndex((s) => s.id === snapshotId);
    if (idx >= 0) {
      list.splice(idx, 1);
      saveProject(project);
    }
  }
  function renameAASnapshot(project, snapshotId, name) {
    const list = getAASnapshots(project);
    const snap = list.find((s) => s.id === snapshotId);
    if (!snap) return;
    snap.name = String(name || "").trim() || snap.name;
    saveProject(project);
  }

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
    const snapshots = getAASnapshots(project);
    const snapshotBar = `
      <div class="aa-snapshot-bar">
        <div class="aa-snapshot-bar-left">
          <span class="aa-snapshot-label">Working AA</span>
          <span class="muted">${snapshots.length} saved snapshot${snapshots.length === 1 ? "" : "s"}</span>
        </div>
        <div class="aa-snapshot-bar-right">
          ${snapshots.length > 0 ? `
            <select class="ribbon-select" data-aa-snapshot-load aria-label="Load snapshot">
              <option value="">Load snapshot…</option>
              ${snapshots.slice().sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0)).map((s) => `<option value="${attr(s.id)}">${escapeHtml(s.name)} · ${escapeHtml(new Date(s.savedAt || Date.now()).toLocaleString("en-AU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }))}</option>`).join("")}
            </select>
            <button class="ghost-button compact" type="button" data-aa-snapshot-manage>Manage…</button>
          ` : ""}
          <button class="ghost-button compact" type="button" data-aa-snapshot-save>${ICON.plus}<span>Save snapshot</span></button>
        </div>
      </div>`;
    return PageBody(`
      <div class="page-shell aa-shell">
        <div class="chat-page-head">
          <div>
            <h1 class="page-title">Adjudication Application</h1>
            <p class="page-sub">Guided drafter for intake, dispute mapping, RFIs per item, and the live master document.</p>
          </div>
          <div class="aa-header-actions">
            ${aa.stage !== "intake" ? `<span class="aa-progress-pill" title="Drafted threads">${draftedThreads}/${totalThreads} drafted · ${progressPct}%</span>` : ""}
            ${aa.stage !== "intake" ? `<button class="ghost-button compact" type="button" data-aa-back-stage>← Back a stage</button>` : ""}
            ${deadlineMeta ? `<span class="aa-deadline-pill ${deadlineMeta.cls}" title="Lodgement deadline">${deadlineMeta.label}</span>` : ""}
            <button class="ghost-button compact danger" type="button" data-aa-reset>Reset</button>
          </div>
        </div>
        ${snapshotBar}
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
            <h3>Stage 1: Document intake</h3>
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
              <label class="aa-doc-label">Payment Schedule${scenario.psOptional ? " (none received, optional)" : ""}</label>
              ${scenario.psOptional ? `<p class="muted aa-doc-help">No PS was given by the respondent in the s 76 window. Leave this blank; Sopal will frame the application accordingly.</p>` : ""}
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
              <a class="link-button small aa-intake-deadline-link" href="/sopal-v2/tools/due-date-calculator?scenario=adjudicationApp" data-nav title="Open the Due Date Calculator to compute the s 79 BIF Act deadline from the scenario + key dates">Calculate from dates →</a>
            </div>
            <div class="aa-intake-actions">
              <button class="dark-button" type="button" data-aa-parse>${ICON.sparkles}<span>Parse documents</span></button>
              <span class="muted aa-intake-help">Parsing extracts the parties, amounts, claim line items, and (if a PS was given) the respondent's reasons. You'll review and edit the result on the next stage.</span>
            </div>
            <div class="aa-intake-error" data-aa-parse-error role="alert" hidden></div>
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
              <h3>Stage 2: Dispute table</h3>
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
            <h3>Stage 2: Dispute table</h3>
            <p class="muted">Edit the rows. Merge or split where the PC artificially divides one dispute. Set the issue type so the RFIs are tailored. Lock to advance.</p>
          </div>
          <div class="aa-table-actions">
            <button class="ghost-button compact" type="button" data-aa-edit-matter title="Edit the parties, contract reference and reference date that the engine and master document use">${ICON.settings}<span>Matter details</span></button>
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
            <h3>Stage 5: Final review and lodgement</h3>
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
                <li>${(aa.jurisdictionalRfis.submissions || "").length > 60 ? "✓" : "○"} Jurisdictional submissions drafted <span class="muted">(optional, only render if there's a jurisdictional issue)</span></li>
                <li>${((aa.introductionHtml || "").length > 60 || (aa.generalRfis.submissions || "").length > 60) ? "✓" : "○"} Introduction / background drafted <span class="muted">(optional)</span></li>
                <li>${(aa.execSummaryHtml || "").length > 60 ? "✓" : "○"} Executive summary drafted <span class="muted">(optional; generate from the master modal)</span></li>
                <li>${(aa.overarchingHtml || "").length > 60 ? "✓" : "○"} Overarching arguments drafted <span class="muted">(optional)</span></li>
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
        <td><span class="aa-currency-cell"><span class="aa-currency-prefix">$</span><input class="aa-cell aa-cell-num" type="number" min="0" step="1" data-aa-cell="claimed" value="${attr(d.claimed || 0)}" aria-label="Claimed amount in dollars"></span></td>
        <td><span class="aa-currency-cell"><span class="aa-currency-prefix">$</span><input class="aa-cell aa-cell-num" type="number" min="0" step="1" data-aa-cell="scheduled" value="${attr(d.scheduled || 0)}" aria-label="Scheduled amount in dollars"></span></td>
        <td>
          <select class="aa-cell" data-aa-cell="status" data-aa-status="${attr(d.status || "disputed")}">
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
                      : engineReady ? '<span class="aa-nav-dot ready" title="Sopal has enough info, click Draft this item">⚡</span>'
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
          <div class="card-head aa-rfi-head">
            <div>
              <h3>${escapeHtml(active.label)}</h3>
              <span class="muted">${active.kind === "dispute" ? "Per-item RFIs" : "Shared RFIs"}${active.kind === "dispute" && active.dispute ? ` · ${escapeHtml(AA_ISSUE_TYPE_LABELS[active.dispute.issueType] || active.dispute.issueType || "Item")}` : ""}</span>
            </div>
            ${(active.thread.rounds || []).length > 0
              ? `<span class="aa-rfi-round-count" title="Total RFIs raised in this thread">${(active.thread.rounds || []).length} RFI${(active.thread.rounds || []).length === 1 ? "" : "s"}</span>`
              : ""}
          </div>
          <div class="aa-rfi-stream" data-aa-rfi-stream>
            ${(active.thread.rounds || []).length === 0
              ? `<div class="empty-state aa-rfi-empty"><strong>Ready when you are.</strong><p>Click <em>Ask first RFI</em> below. Sopal will generate the first targeted question for this ${active.kind === "dispute" ? "dispute" : "thread"} and add it as a row in the table.</p></div>`
              : `<table class="aa-rfi-table">
                   <thead>
                     <tr>
                       <th class="aa-rfi-col-num" scope="col">#</th>
                       <th class="aa-rfi-col-q" scope="col">Sopal's question</th>
                       <th class="aa-rfi-col-a" scope="col">Your response</th>
                       <th class="aa-rfi-col-act" scope="col"><span class="visually-hidden">Action</span></th>
                     </tr>
                   </thead>
                   <tbody>
                     ${(active.thread.rounds || []).map((r, i) => `
                       <tr class="aa-rfi-row ${r.answer ? "answered" : "pending"}" data-aa-rfi-row="${i}">
                         <td class="aa-rfi-col-num"><span class="aa-rfi-num">RFI ${i + 1}</span></td>
                         <td class="aa-rfi-col-q">${renderMarkdown(r.question || "")}</td>
                         <td class="aa-rfi-col-a">
                           ${r.answer
                             ? `<div class="aa-rfi-answer-view" data-aa-rfi-answer-view="${i}"><div class="aa-rfi-answer-text">${renderMarkdown(r.answer)}</div></div>`
                             : `<form class="aa-rfi-answer-form" data-aa-rfi-answer="${i}">
                                  <textarea class="text-area auto-grow aa-rfi-answer-input" name="answer" rows="2" placeholder="Type your answer to RFI ${i + 1}…"></textarea>
                                </form>`}
                         </td>
                         <td class="aa-rfi-col-act">
                           ${r.answer
                             ? `<button class="ghost-button compact" type="button" data-aa-rfi-edit="${i}" title="Edit your answer">Edit</button>`
                             : `<button class="dark-button compact" type="button" data-aa-rfi-submit="${i}">Submit</button>`}
                         </td>
                       </tr>`).join("")}
                   </tbody>
                 </table>`}
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
      render: () => {
        const firm = getFirm();
        const dims = firmPageDimensions(firm);
        const margins = firmPageMargins(firm);
        const accent = firm.accentColour || "#243043";
        const fontStack = firmFontFamily(firm);
        const branded = firmHasBranding(firm) ? "is-branded" : "is-unbranded";
        return `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal aa-master-modal firm-paper-modal ${branded}" role="dialog" aria-modal="true" style="--firm-accent:${attr(accent)};--firm-page-width:${dims.width}px;--firm-page-height:${dims.height}px;--firm-font:${attr(fontStack)};--firm-page-label:'${dims.label}';--firm-margin-top:${margins.top}px;--firm-margin-right:${margins.right}px;--firm-margin-bottom:${margins.bottom}px;--firm-margin-left:${margins.left}px;">
            <div class="modal-head aa-master-modal-head">
              <h2>Master document <span class="muted firm-page-label-tag">${dims.label}${firmHasBranding(firm) ? "" : " · unbranded"}</span></h2>
              <div class="aa-master-modal-actions">
                <button class="ghost-button compact" type="button" data-aa-edit-cover>Cover page</button>
                <button class="ghost-button compact" type="button" data-aa-edit-intro>Introduction</button>
                <button class="ghost-button compact" type="button" data-aa-gen-summary>${ICON.sparkles}<span data-aa-gen-summary-label>${aa.execSummaryHtml ? "Re-generate summary" : "Generate summary"}</span></button>
                <button class="ghost-button compact" type="button" data-aa-edit-summary>Edit summary</button>
                <button class="ghost-button compact" type="button" data-aa-edit-overarching>Overarching</button>
                <span class="aa-master-actions-sep"></span>
                <button class="dark-button compact" type="button" data-aa-export>${ICON.download}<span>Export .doc</span></button>
                <button class="ghost-button compact" type="button" data-aa-export-statdecs>${ICON.download}<span>Stat dec</span></button>
                <button class="ghost-button compact" type="button" data-aa-export-soe>${ICON.download}<span>Evidence index</span></button>
                <button class="ghost-button compact" type="button" data-aa-print-master>${ICON.file}<span>Print</span></button>
                <button class="ghost-button compact" type="button" data-aa-copy-master>${ICON.copy}<span>Copy as Markdown</span></button>
                <button class="ghost-button compact" type="button" data-aa-rebuild>Rebuild</button>
                <a class="ghost-button compact" href="/sopal-v2/settings" data-nav title="Edit firm branding">Firm</a>
                <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
              </div>
            </div>
            <div class="modal-body aa-master-modal-body firm-paper-stage">
              <div class="firm-paper-stack" data-firm-paper-stack data-aa-master></div>
            </div>
          </div>
        </div>`;
      },
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        function refreshMasterPane() {
          const mount = rootEl.querySelector("[data-aa-master]");
          if (mount) {
            mount.innerHTML = "";
            paintFirmPaperStack(mount, renderAAMaster(project, aa), getFirm());
            bindAATocLinks();
          }
          const lbl = rootEl.querySelector("[data-aa-gen-summary-label]");
          if (lbl) lbl.textContent = aa.execSummaryHtml ? "Re-generate summary" : "Generate summary";
        }
        // Initial paint — the modal HTML leaves the stack empty so we can
        // measure the freshly-mounted container before we slice content
        // into pages.
        const initialMount = rootEl.querySelector("[data-aa-master]");
        if (initialMount) {
          paintFirmPaperStack(initialMount, renderAAMaster(project, aa), getFirm());
          bindAATocLinks();
        }
        rootEl.querySelector("[data-aa-rebuild]")?.addEventListener("click", refreshMasterPane);
        // Section editors. Each one closes the master modal, opens the editor
        // sub-modal, and the sub-modal saves back to aa state. The master
        // modal is re-opened by the user when they're done editing.
        rootEl.querySelector("[data-aa-edit-cover]")?.addEventListener("click", () => {
          modal = null; render();
          openAACoverMetaModal(project, aa, true);
        });
        rootEl.querySelector("[data-aa-edit-intro]")?.addEventListener("click", () => {
          modal = null; render();
          openAAEditModal({
            project, aa, mode: "html", returnToMaster: true,
            title: "Introduction",
            hint: "Short overview of project, parties, contract execution, and what brought the matter to adjudication. Renders as the second section of the master if populated. If left blank the general / background RFI thread submissions will be used in its place.",
            getValue: () => aa.introductionHtml || "",
            setValue: (v) => { aa.introductionHtml = v; },
          });
        });
        rootEl.querySelector("[data-aa-edit-summary]")?.addEventListener("click", () => {
          modal = null; render();
          openAAEditModal({
            project, aa, mode: "html", returnToMaster: true,
            title: "Executive summary",
            hint: "Sits near the top of the master document. Use 'Generate summary' to draft this from the per-item threads, or hand-edit here.",
            getValue: () => aa.execSummaryHtml || "",
            setValue: (v) => { aa.execSummaryHtml = v; },
          });
        });
        rootEl.querySelector("[data-aa-edit-overarching]")?.addEventListener("click", () => {
          modal = null; render();
          openAAEditModal({
            project, aa, mode: "html", returnToMaster: true,
            title: "Overarching arguments",
            hint: "Cross-cutting arguments that don't fit a single per-item section — prevention principle, estoppel, waiver, contract construction, repudiation. Optional: leave blank to omit this section from the master.",
            getValue: () => aa.overarchingHtml || "",
            setValue: (v) => { aa.overarchingHtml = v; },
          });
        });
        rootEl.querySelector("[data-aa-gen-summary]")?.addEventListener("click", async () => {
          const btn = rootEl.querySelector("[data-aa-gen-summary]");
          const originalLabel = btn ? btn.innerHTML : "";
          if (btn) {
            btn.disabled = true;
            btn.innerHTML = `<span class="thinking-dots"><i></i><i></i><i></i></span><span>Generating…</span>`;
          }
          try {
            await aaCallExecSummary(project, aa);
            refreshMasterPane();
          } catch (err) {
            alert(err.message || "Could not generate the executive summary. Please try again.");
          } finally {
            if (btn) {
              btn.disabled = false;
              btn.innerHTML = originalLabel;
              const lbl = btn.querySelector("[data-aa-gen-summary-label]");
              if (lbl) lbl.textContent = aa.execSummaryHtml ? "Re-generate summary" : "Generate summary";
            }
          }
        });
        rootEl.querySelector("[data-aa-export]")?.addEventListener("click", () => {
          aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-adjudication-application.doc`,
            `${escapeHtml(project.name)} — Adjudication Application`,
            renderAAMaster(project, aa), getFirm());
        });
        rootEl.querySelector("[data-aa-export-statdecs]")?.addEventListener("click", () => {
          aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-statutory-declaration.doc`,
            `${escapeHtml(project.name)} — Statutory Declaration`,
            renderAAStatDecCompilation(project, aa), getFirm());
        });
        rootEl.querySelector("[data-aa-export-soe]")?.addEventListener("click", () => {
          aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-evidence-index.doc`,
            `${escapeHtml(project.name)} — Index of Supporting Evidence`,
            renderAAEvidenceIndex(project, aa), getFirm());
        });
        rootEl.querySelector("[data-aa-print-master]")?.addEventListener("click", () => {
          openFirmPrintPreview({
            title: `${project.name} — Adjudication Application`,
            bodyHtml: renderAAMaster(project, aa),
            firm: getFirm(),
          });
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
    // Master document is FLUID: only render sections that have content.
    // Required: cover page (always), conclusion (always).
    // Optional (only render when populated): introduction, exec summary,
    // jurisdiction, overarching arguments, per-item submissions, evidence index.
    //
    // Section numbering is dynamic — we count up only for sections we actually
    // render. The ToC + h2/h3 IDs use a stable slug, but the "1.", "2.", etc.
    // is computed at render time so adding/removing an optional section keeps
    // numbering tight.
    const toc = [];
    function id(slug) { return `aa-sec-${slug}`; }
    const sections = [];
    let sectionCounter = 0;
    function nextNum() { sectionCounter += 1; return String(sectionCounter); }
    function hasHtmlContent(html) {
      if (!html) return false;
      const stripped = String(html).replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ").trim();
      return stripped.length > 0;
    }

    const scenarioId = aa.s79Scenario || "less-than-claimed";
    const scenario = AA_S79_SCENARIOS.find((s) => s.id === scenarioId) || AA_S79_SCENARIOS[0];
    const cover = aa.coverMeta || {};
    const claimantName = aa.parties.claimant || project.claimant || "[Claimant]";
    const respondentName = aa.parties.respondent || project.respondent || "[Respondent]";
    const contractRef = aa.contractReference || project.reference || "[Contract reference]";
    const refDate = aa.referenceDate || "[Reference date]";
    const claimed = Number(aa.claimedAmount || 0);
    const scheduled = Number(aa.scheduledAmount || 0);
    const inDispute = Math.max(0, claimed - scheduled);

    // ---- Cover page (always) ----
    // Modelled on the layout commonly used for QLD adjudication applications:
    // formal title, opening line citing s 21 BIF Act, then bordered tables for
    // CLAIMANT and RESPONDENT details and a third table for application meta.
    const opener = scenarioId === "no-schedule"
      ? "This Adjudication Application is made by the Claimant under section 79(2)(a) of the Building Industry Fairness (Security of Payment) Act 2017 (Qld) (the <strong>BIF Act</strong>)."
      : scenarioId === "scheduled-but-unpaid"
      ? "This Adjudication Application is made by the Claimant under section 79(2)(c) of the Building Industry Fairness (Security of Payment) Act 2017 (Qld) (the <strong>BIF Act</strong>)."
      : "This Adjudication Application is made by the Claimant under section 79(2)(b) of the Building Industry Fairness (Security of Payment) Act 2017 (Qld) (the <strong>BIF Act</strong>).";
    const psBlockRows = scenarioId === "no-schedule"
      ? `<tr><th>Payment schedule</th><td>No payment schedule received (s 79(2)(a))</td></tr>`
      : `<tr><th>Payment schedule</th><td>${cover.psDate ? `Served ${escapeHtml(cover.psDate)}. ` : ""}Scheduled amount: ${formatCurrencyFull(scheduled)}</td></tr>`;
    sections.push(`<div class="aa-cover">
      <h1 class="aa-cover-title">ADJUDICATION APPLICATION</h1>
      <p class="aa-cover-opener">${opener}</p>
      <h3 class="aa-cover-section">Claimant details</h3>
      <table class="aa-cover-table">
        <tr><th>Name</th><td>${escapeHtml(claimantName)}</td></tr>
        ${cover.claimantAbn ? `<tr><th>ABN</th><td>${escapeHtml(cover.claimantAbn)}</td></tr>` : ""}
        ${cover.claimantContact ? `<tr><th>Contact</th><td>${escapeHtml(cover.claimantContact)}</td></tr>` : ""}
        ${cover.claimantAddress ? `<tr><th>Address</th><td>${escapeHtml(cover.claimantAddress)}</td></tr>` : ""}
        ${cover.claimantPhone ? `<tr><th>Phone</th><td>${escapeHtml(cover.claimantPhone)}</td></tr>` : ""}
        ${cover.claimantEmail ? `<tr><th>Email</th><td>${escapeHtml(cover.claimantEmail)}</td></tr>` : ""}
      </table>
      <h3 class="aa-cover-section">Respondent details</h3>
      <table class="aa-cover-table">
        <tr><th>Name</th><td>${escapeHtml(respondentName)}</td></tr>
        ${cover.respondentAbn ? `<tr><th>ABN</th><td>${escapeHtml(cover.respondentAbn)}</td></tr>` : ""}
        ${cover.respondentContact ? `<tr><th>Contact</th><td>${escapeHtml(cover.respondentContact)}</td></tr>` : ""}
        ${cover.respondentAddress ? `<tr><th>Address</th><td>${escapeHtml(cover.respondentAddress)}</td></tr>` : ""}
        ${cover.respondentPhone ? `<tr><th>Phone</th><td>${escapeHtml(cover.respondentPhone)}</td></tr>` : ""}
        ${cover.respondentEmail ? `<tr><th>Email</th><td>${escapeHtml(cover.respondentEmail)}</td></tr>` : ""}
      </table>
      <h3 class="aa-cover-section">Application details</h3>
      <table class="aa-cover-table">
        <tr><th>Contract reference</th><td>${escapeHtml(contractRef)}</td></tr>
        ${cover.contractDate ? `<tr><th>Contract executed</th><td>${escapeHtml(cover.contractDate)}</td></tr>` : ""}
        ${cover.siteAddress ? `<tr><th>Project / site</th><td>${escapeHtml(cover.siteAddress)}</td></tr>` : ""}
        <tr><th>Reference date</th><td>${escapeHtml(refDate)}</td></tr>
        ${cover.pcDate ? `<tr><th>Payment claim served</th><td>${escapeHtml(cover.pcDate)}</td></tr>` : ""}
        <tr><th>Payment claim amount</th><td>${formatCurrencyFull(claimed)}</td></tr>
        ${psBlockRows}
        <tr><th>Amount in dispute</th><td>${formatCurrencyFull(inDispute || claimed)}</td></tr>
        <tr><th>s 79 BIF Act basis</th><td>${escapeHtml(scenario.label)}</td></tr>
        ${cover.ana ? `<tr><th>Authorised Nominating Authority</th><td>${escapeHtml(cover.ana)}${cover.anaReference ? ` (ref: ${escapeHtml(cover.anaReference)})` : ""}</td></tr>` : ""}
        ${cover.applicationDate ? `<tr><th>Application date</th><td>${escapeHtml(cover.applicationDate)}</td></tr>` : ""}
      </table>
    </div>`);

    // ---- Introduction (optional) ----
    // Hierarchy: prefer the explicit aa.introductionHtml if the user has typed
    // one in the master modal; otherwise fall back to the general / background
    // RFI thread submissions (which is the natural slot for project + parties +
    // contract execution background).
    const introHtml = hasHtmlContent(aa.introductionHtml)
      ? aa.introductionHtml
      : (hasHtmlContent(aa.generalRfis && aa.generalRfis.submissions) ? aa.generalRfis.submissions : "");
    if (hasHtmlContent(introHtml)) {
      const num = nextNum();
      toc.push({ id: id("introduction"), num, label: "Introduction", indent: 0 });
      sections.push(`<h2 id="${id("introduction")}">${num}. Introduction</h2>${introHtml}`);
    }

    // ---- Executive summary (optional) ----
    if (hasHtmlContent(aa.execSummaryHtml)) {
      const num = nextNum();
      toc.push({ id: id("exec-summary"), num, label: "Executive summary", indent: 0 });
      sections.push(`<h2 id="${id("exec-summary")}">${num}. Executive summary</h2>${aa.execSummaryHtml}`);
    }

    // ---- Jurisdiction (optional) ----
    if (hasHtmlContent(aa.jurisdictionalRfis && aa.jurisdictionalRfis.submissions)) {
      const num = nextNum();
      toc.push({ id: id("jurisdiction"), num, label: "Jurisdiction", indent: 0 });
      sections.push(`<h2 id="${id("jurisdiction")}">${num}. Jurisdiction</h2>${aa.jurisdictionalRfis.submissions}`);
    }

    // ---- Overarching arguments (optional) ----
    if (hasHtmlContent(aa.overarchingHtml)) {
      const num = nextNum();
      toc.push({ id: id("overarching"), num, label: "Overarching arguments", indent: 0 });
      sections.push(`<h2 id="${id("overarching")}">${num}. Overarching arguments</h2>${aa.overarchingHtml}`);
    }

    // ---- Per-item submissions (optional, but the heart of most AAs) ----
    if ((aa.disputes || []).length) {
      const itemsNum = nextNum();
      toc.push({ id: id("disputes"), num: itemsNum, label: "Submissions on disputed items", indent: 0 });
      sections.push(`<h2 id="${id("disputes")}">${itemsNum}. Submissions on disputed items</h2>`);
      aa.disputes.forEach((d, i) => {
        const slug = `dispute-${d.id}`;
        const subNum = `${itemsNum}.${i + 1}`;
        toc.push({ id: id(slug), num: subNum, label: d.item || "Item", indent: 1 });
        sections.push(`<h3 id="${id(slug)}">${subNum} ${escapeHtml(d.item || "Item")}${d.issueType ? ` <span class="aa-issue-tag">${escapeHtml(AA_ISSUE_TYPE_LABELS[d.issueType] || d.issueType)}</span>` : ""}</h3>`);
        // Per-item summary mini-table — claimed / scheduled / in dispute /
        // PS reasons. Renders even when the thread hasn't been drafted yet so
        // the master always shows "what's at issue here" at a glance.
        const itemClaimed = Number(d.claimed || 0);
        const itemScheduled = Number(d.scheduled || 0);
        const itemInDispute = Math.max(0, itemClaimed - itemScheduled);
        const psReasonsHtml = d.psReasons ? `<tr><th>Respondent's reasons (PS)</th><td>${escapeHtml(d.psReasons)}</td></tr>` : "";
        sections.push(`<table class="aa-item-meta">
          <tr><th>Claimed</th><td>${formatCurrencyFull(itemClaimed)}</td></tr>
          <tr><th>Scheduled</th><td>${formatCurrencyFull(itemScheduled)}</td></tr>
          <tr><th>In dispute</th><td>${formatCurrencyFull(itemInDispute)}</td></tr>
          ${psReasonsHtml}
        </table>`);
        const subs = (d.rfis && d.rfis.submissions) || "";
        sections.push(subs || "<p><em>(The Claimant's submissions on this item will appear here once the per-item RFI thread is drafted.)</em></p>");
      });

      // ---- Quantum summary (only when there are items with non-zero claimed/scheduled) ----
      const totalClaimed = aa.disputes.reduce((s, d) => s + Number(d.claimed || 0), 0);
      const totalScheduled = aa.disputes.reduce((s, d) => s + Number(d.scheduled || 0), 0);
      if (totalClaimed > 0 || totalScheduled > 0) {
        const qNum = nextNum();
        toc.push({ id: id("quantum"), num: qNum, label: "Quantum summary", indent: 0 });
        sections.push(`<h2 id="${id("quantum")}">${qNum}. Quantum summary</h2>
          <table>
            <thead><tr><th>Item</th><th>Claimed</th><th>Scheduled</th><th>In dispute</th></tr></thead>
            <tbody>
              ${aa.disputes.map((d) => {
                const c = Number(d.claimed || 0);
                const s = Number(d.scheduled || 0);
                const inD = Math.max(0, c - s);
                return `<tr><td>${escapeHtml(d.item || "Item")}</td><td>${formatCurrencyFull(c)}</td><td>${formatCurrencyFull(s)}</td><td>${formatCurrencyFull(inD)}</td></tr>`;
              }).join("")}
              <tr><td><strong>Totals</strong></td><td><strong>${formatCurrencyFull(totalClaimed)}</strong></td><td><strong>${formatCurrencyFull(totalScheduled)}</strong></td><td><strong>${formatCurrencyFull(Math.max(0, totalClaimed - totalScheduled))}</strong></td></tr>
            </tbody>
          </table>`);
      }
    }

    // ---- Conclusion (always) ----
    // Single, direct paragraph in the NZ-doc style: "In the premises of the
    // above, the Claimant respectfully submits that the Adjudicator should
    // determine that the Respondent is liable to pay the Claimant the sum of
    // $X (excluding GST)." Adapts to a less-than-claimed scenario by saying
    // "the adjudicated amount" rather than "the full Payment Claim amount" —
    // the user can edit this freely in the master modal.
    const conclusionNum = nextNum();
    toc.push({ id: id("conclusion"), num: conclusionNum, label: "Conclusion", indent: 0 });
    sections.push(`<h2 id="${id("conclusion")}">${conclusionNum}. Conclusion</h2>
      <p>In the premises of the above, the Claimant respectfully submits that the Adjudicator should determine that the Respondent is liable to pay the Claimant the sum of <strong>${formatCurrencyFull(claimed)}</strong>${scheduled > 0 ? `, less the amount of ${formatCurrencyFull(scheduled)} already scheduled by the Respondent (resulting in an amount in dispute of ${formatCurrencyFull(inDispute)})` : ""}, together with interest under section 90 of the BIF Act and the Adjudicator's fees as the Adjudicator sees fit.</p>`);

    // ---- Evidence index (optional) ----
    const evidence = [];
    (aa.disputes || []).forEach((d) => ((d.rfis && d.rfis.evidenceIndex) || []).forEach((e) => evidence.push(e)));
    if (aa.jurisdictionalRfis && Array.isArray(aa.jurisdictionalRfis.evidenceIndex)) aa.jurisdictionalRfis.evidenceIndex.forEach((e) => evidence.push(e));
    if (aa.generalRfis && Array.isArray(aa.generalRfis.evidenceIndex)) aa.generalRfis.evidenceIndex.forEach((e) => evidence.push(e));
    if (evidence.length) {
      const eNum = nextNum();
      toc.push({ id: id("evidence"), num: eNum, label: "Index of supporting evidence", indent: 0 });
      sections.push(`<h2 id="${id("evidence")}">${eNum}. Index of supporting evidence</h2><ol>${evidence.map((e) => `<li><strong>${escapeHtml(e.ref || "")}</strong>: ${escapeHtml(e.desc || "")}${e.location ? ` (${escapeHtml(e.location)})` : ""}</li>`).join("")}</ol>`);
    }

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
              <h2>${escapeHtml(ctx.label)}: supporting artefacts</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <div class="modal-body aa-artifacts-body">
              <section>
                <h4>Evidence index for this thread</h4>
                ${evidence.length
                  ? `<ul class="aa-artifact-list">${evidence.map((e) => `<li><strong>${escapeHtml(e.ref || "")}</strong>: ${escapeHtml(e.desc || "")}${e.location ? ` <span class="muted">(${escapeHtml(e.location)})</span>` : ""}</li>`).join("")}</ul>`
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
          const html = `<h1>Statutory declaration: ${escapeHtml(ctx.label)}</h1>${renderMarkdown(statDec)}`;
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

  function openAASnapshotManager(project) {
    const renderList = () => {
      const list = getAASnapshots(project).slice().sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0));
      if (!list.length) return `<div class="empty-state"><strong>No snapshots yet.</strong><p>Click "Save snapshot" to capture the current working AA.</p></div>`;
      return `
        <div class="aa-snapshot-list">
          ${list.map((s) => `
            <div class="aa-snapshot-row" data-aa-snap-row="${attr(s.id)}">
              <div class="aa-snapshot-meta">
                <strong>${escapeHtml(s.name)}</strong>
                <span class="muted">Saved ${escapeHtml(new Date(s.savedAt || Date.now()).toLocaleString("en-AU", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }))}</span>
              </div>
              <div class="aa-snapshot-actions">
                <button class="ghost-button compact" type="button" data-aa-snap-load="${attr(s.id)}">Load</button>
                <button class="ghost-button compact" type="button" data-aa-snap-rename="${attr(s.id)}">Rename</button>
                <button class="ghost-button compact danger" type="button" data-aa-snap-delete="${attr(s.id)}">Delete</button>
              </div>
            </div>
          `).join("")}
        </div>
      `;
    };
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal" role="dialog" aria-modal="true" style="max-width:560px;">
            <div class="modal-head">
              <h2>Manage AA snapshots</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <div class="modal-body" data-aa-snapshot-list>
              ${renderList()}
            </div>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        const refresh = () => {
          const mount = document.querySelector("[data-aa-snapshot-list]");
          if (mount) mount.innerHTML = renderList();
          // Re-bind the inner action buttons after refresh.
          bindRows();
        };
        const bindRows = () => {
          document.querySelectorAll("[data-aa-snap-load]").forEach((b) => b.addEventListener("click", () => {
            const id = b.dataset.aaSnapLoad;
            const snap = getAASnapshots(project).find((s) => s.id === id);
            if (!snap) return;
            const hasLive = !!(project.complexApps && project.complexApps["adjudication-application"]);
            const msg = hasLive
              ? `Load snapshot "${snap.name}"? Your current working AA will be replaced.\n\nClick Cancel and save the working AA first if you want to keep it.`
              : `Load snapshot "${snap.name}" as the working AA?`;
            if (!confirm(msg)) return;
            loadAASnapshot(project, id);
            close();
          }));
          document.querySelectorAll("[data-aa-snap-rename]").forEach((b) => b.addEventListener("click", () => {
            const id = b.dataset.aaSnapRename;
            const snap = getAASnapshots(project).find((s) => s.id === id);
            if (!snap) return;
            const next = prompt("Rename snapshot", snap.name);
            if (next === null) return;
            renameAASnapshot(project, id, next);
            refresh();
          }));
          document.querySelectorAll("[data-aa-snap-delete]").forEach((b) => b.addEventListener("click", () => {
            const id = b.dataset.aaSnapDelete;
            const snap = getAASnapshots(project).find((s) => s.id === id);
            if (!snap) return;
            if (!confirm(`Delete snapshot "${snap.name}"? This cannot be undone.`)) return;
            deleteAASnapshot(project, id);
            refresh();
          }));
        };
        bindRows();
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
      if (!confirm("Reset the entire adjudication application workflow? All extracted items, RFIs, and drafts will be cleared.\n\nTip: if you want to start a second AA but keep this one, use \"Save snapshot\" first.")) return;
      delete project.complexApps["adjudication-application"];
      saveProject(project);
      render();
    }));

    // ----- Snapshot bar handlers -----
    document.querySelectorAll("[data-aa-snapshot-save]").forEach((b) => b.addEventListener("click", () => {
      const aaLive = project.complexApps && project.complexApps["adjudication-application"];
      if (!aaLive) { alert("No AA to snapshot yet — work on Stage 1 first."); return; }
      const name = prompt("Name this snapshot (e.g. \"Reference date 25 March\")", `Snapshot ${getAASnapshots(project).length + 1}`);
      if (name === null) return;
      saveAASnapshot(project, name);
      render();
    }));
    const loadSelect = document.querySelector("[data-aa-snapshot-load]");
    if (loadSelect) {
      loadSelect.addEventListener("change", () => {
        const id = loadSelect.value;
        if (!id) return;
        const snap = getAASnapshots(project).find((s) => s.id === id);
        if (!snap) { loadSelect.value = ""; return; }
        const hasLive = !!(project.complexApps && project.complexApps["adjudication-application"]);
        const msg = hasLive
          ? `Load snapshot "${snap.name}"? Your current working AA will be replaced.\n\nIf you want to keep the current working AA, click Cancel and use "Save snapshot" first.`
          : `Load snapshot "${snap.name}" as the working AA?`;
        if (!confirm(msg)) { loadSelect.value = ""; return; }
        loadAASnapshot(project, id);
        render();
      });
    }
    document.querySelectorAll("[data-aa-snapshot-manage]").forEach((b) => b.addEventListener("click", () => openAASnapshotManager(project)));

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
        renderAAMaster(project, aa), getFirm());
    });
    document.querySelector("[data-aa-export-statdecs]")?.addEventListener("click", () => {
      aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-statutory-declaration.doc`,
        `${escapeHtml(project.name)} — Statutory Declaration`,
        renderAAStatDecCompilation(project, aa), getFirm());
    });
    document.querySelector("[data-aa-export-soe]")?.addEventListener("click", () => {
      aaDownloadDoc(`${project.name.replace(/[^a-z0-9]+/gi, "-")}-evidence-index.doc`,
        `${escapeHtml(project.name)} — Index of Supporting Evidence`,
        renderAAEvidenceIndex(project, aa), getFirm());
    });
    document.querySelector("[data-aa-print-master]")?.addEventListener("click", () => {
      openFirmPrintPreview({
        title: `${project.name} — Adjudication Application`,
        bodyHtml: renderAAMaster(project, aa),
        firm: getFirm(),
      });
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

  function aaDownloadDoc(filename, title, body, firm) {
    // Word opens HTML with the right office namespaces directly. We embed a
    // small inline stylesheet so the cover page, per-item meta tables and
    // ToC render with the same formal look the user sees in the app — Word
    // respects `<style>` blocks for `body`, headings, tables and class-based
    // selectors when the doc is opened in compatibility mode. The ToC nav is
    // hidden in the .doc export because Word doesn't follow in-doc anchors
    // the same way the browser does.
    //
    // When firm branding is configured we wrap the body in a cover-page
    // letterhead block + add a Word-style page footer using the
    // <div style="mso-element:footer"> trick, which Word picks up as a
    // running footer when the document is opened.
    const f = firm || {};
    const accent = f.accentColour || "#000000";
    const familySerif = f.bodyFont === "sans"
      ? '"Helvetica", "Arial", sans-serif'
      : f.bodyFont === "inter"
      ? '"Inter", "Calibri", sans-serif'
      : '"Times New Roman", "Source Serif Pro", serif';
    const pageSize = f.pageSize === "letter" ? "8.5in 11in" : "210mm 297mm";
    const wordStyles = `
      <style>
        @page { size: ${pageSize}; margin: 25mm 25mm 25mm 25mm; mso-header-margin: 12mm; mso-footer-margin: 12mm; }
        body { font-family: ${familySerif}; font-size: 12pt; line-height: 1.5; color: #1a1a1a; }
        h1 { font-size: 22pt; font-weight: bold; text-align: center; margin: 0 0 14px; }
        h2 { font-size: 14pt; font-weight: bold; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1pt solid ${accent}; color: ${accent}; }
        h3 { font-size: 12pt; font-weight: bold; margin: 14px 0 6px; }
        h4 { font-size: 11pt; font-weight: bold; margin: 10px 0 4px; }
        p { margin: 0 0 10px; text-align: justify; }
        table { width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 11pt; }
        th, td { border: 1pt solid #888; padding: 5px 8px; text-align: left; vertical-align: top; }
        th { background: #f0ece4; font-weight: bold; }
        nav.aa-toc { display: none; }
        .aa-cover { padding: 0 0 18pt; border-bottom: 2pt solid #000; margin: 0 0 28pt; }
        .aa-cover-title { font-size: 24pt; letter-spacing: 0.04em; text-transform: uppercase; text-align: center; margin: 0 0 14pt; }
        .aa-cover-opener { font-size: 12pt; margin: 0 0 18pt; }
        .aa-cover-section { font-size: 11pt; font-weight: bold; letter-spacing: 0.06em; text-transform: uppercase; margin: 14pt 0 6pt; border: 0; padding: 0; }
        table.aa-cover-table { width: 100%; }
        table.aa-cover-table th { width: 38%; }
        table.aa-item-meta { width: auto; min-width: 60%; margin: 4px 0 12px; font-size: 10pt; }
        table.aa-item-meta th { width: 180px; background: #faf7f1; }
        .aa-issue-tag { display: inline; font-size: 9pt; padding: 1px 6px; margin-left: 6px; background: #e0e7ff; }
        .firm-cover-letterhead-block { width: 100%; border-bottom: 2pt solid ${accent}; padding-bottom: 14pt; margin: 0 0 22pt; overflow: hidden; }
        .firm-cover-letterhead-logo { float: left; max-height: 90pt; max-width: 240pt; }
        .firm-cover-letterhead-text { float: right; text-align: right; font-size: 10pt; line-height: 1.4; color: #2a2a2a; white-space: pre-line; }
        .firm-doc-footer { font-size: 9pt; color: #555; border-top: 1pt solid #aaa; padding-top: 4pt; margin-top: 18pt; }
        .firm-doc-footer span.right { float: right; }
      </style>`;

    const showHeader = firmHasBranding(f);
    const headerHtml = showHeader
      ? `<div class="firm-cover-letterhead-block">
          ${f.logoDataUrl ? `<img class="firm-cover-letterhead-logo" src="${f.logoDataUrl}" alt="">` : ""}
          ${f.letterheadAddress ? `<div class="firm-cover-letterhead-text">${escapeHtml(f.letterheadAddress)}</div>` : (f.name ? `<div class="firm-cover-letterhead-text"><strong>${escapeHtml(f.name)}</strong></div>` : "")}
          <div style="clear:both"></div>
        </div>`
      : "";
    // Word reads the special <div style="mso-element:footer"> pattern as a
    // running page footer. PAGE / NUMPAGES are Word fields that resolve at
    // open time so the user gets correct page numbers.
    const footerLeft = f.footerText ? escapeHtml(f.footerText) : (f.name ? escapeHtml(f.name) : "");
    const wordFooter = showHeader
      ? `<div style="mso-element:footer" id="f1">
           <p class="firm-doc-footer">${footerLeft}<span class="right">Page <span style="mso-field-code:&quot;PAGE&quot;">1</span> of <span style="mso-field-code:&quot;NUMPAGES&quot;">1</span></span></p>
         </div>`
      : "";

    const blob = new Blob([
      '<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">',
      '<head><meta charset="UTF-8"><title>', title, '</title>',
      wordStyles,
      '<!--[if gte mso 9]><xml><w:WordDocument><w:View>Print</w:View><w:Zoom>100</w:Zoom></w:WordDocument></xml><![endif]-->',
      '</head><body>',
      headerHtml,
      body,
      wordFooter,
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
      // Inline error helper that surfaces validation and runtime errors next
      // to the Parse button instead of an alert() that loses context the
      // moment it closes. Auto-clears on the next click.
      const errEl = document.querySelector("[data-aa-parse-error]");
      const showErr = (msg) => {
        if (!errEl) { alert(msg); return; }
        errEl.textContent = msg;
        errEl.hidden = false;
      };
      const clearErr = () => {
        if (!errEl) return;
        errEl.textContent = "";
        errEl.hidden = true;
      };
      clearErr();
      if (!pc || !pc.text) {
        showErr("Add the Payment Claim before parsing. Drop a PDF / DOCX / TXT into the Payment Claim slot, or paste the text directly.");
        return;
      }
      if (!scenario.psOptional && (!ps || !ps.text)) {
        showErr("This s 79 scenario requires a Payment Schedule. Either paste it in or switch the scenario to 'No payment schedule received and no payment made'.");
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
        showErr(error.message || "Parse failed. Please try again or paste the document text directly into the slots.");
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
          // Mirror the status value into a data attribute so CSS can wash
          // the cell with the right colour (CSS can't target a select's
          // dynamic .value via attribute selectors).
          if (field === "status" && el.tagName === "SELECT") {
            el.setAttribute("data-aa-status", value);
          }
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
    document.querySelector("[data-aa-edit-matter]")?.addEventListener("click", () => openAAMatterDetailsModal(project, aa));
  }

  // Edit core matter fields the parser populated. These drive the cover
  // page, the engine prompt, and the dispute-table summary, so they need
  // to be editable independently of the cover-page extras (ABN, contact,
  // etc.) which live in openAACoverMetaModal.
  function openAAMatterDetailsModal(project, aa) {
    const fields = [
      { key: "claimant", label: "Claimant (full legal name)", get: () => aa.parties.claimant || "", set: (v) => { aa.parties.claimant = v; } },
      { key: "respondent", label: "Respondent (full legal name)", get: () => aa.parties.respondent || "", set: (v) => { aa.parties.respondent = v; } },
      { key: "contractReference", label: "Contract reference", get: () => aa.contractReference || "", set: (v) => { aa.contractReference = v; }, placeholder: "e.g. HC-2025-12 / PO-9988 / contract dated 12 March 2025" },
      { key: "referenceDate", label: "Reference date", get: () => aa.referenceDate || "", set: (v) => { aa.referenceDate = v; }, placeholder: "YYYY-MM-DD or 31 March 2026" },
    ];
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal aa-cover-modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>Matter details</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <div class="modal-body">
              <p class="muted aa-edit-hint">These are the core fields the engine and master document use. Edit if the parser got something wrong, or to clean up names (e.g. "Acme" → "Acme Builders Pty Ltd"). Leave blank to fall back to the project-level value.</p>
              <div class="aa-cover-grid">
                ${fields.map((f) => `
                  <label class="aa-cover-field">
                    <span>${escapeHtml(f.label)}</span>
                    <input class="text-input" data-aa-matter-key="${attr(f.key)}" type="text" value="${attr(f.get())}" placeholder="${attr(f.placeholder || "")}">
                  </label>
                `).join("")}
              </div>
            </div>
            <div class="modal-foot aa-edit-foot">
              <span class="spacer"></span>
              <button class="ghost-button compact" type="button" data-modal-close>Cancel</button>
              <button class="dark-button compact" type="button" data-aa-matter-save>Save</button>
            </div>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => { modal = null; render(); };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        rootEl.querySelector("[data-aa-matter-save]")?.addEventListener("click", () => {
          rootEl.querySelectorAll("[data-aa-matter-key]").forEach((inp) => {
            const f = fields.find((x) => x.key === inp.dataset.aaMatterKey);
            if (f) f.set((inp.value || "").trim());
          });
          saveProject(project);
          close();
        });
      },
    };
    render();
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
    // Submit an RFI answer. The submit button now lives in a separate <td>
    // from the form (table layout), so we wire submit-by-button explicitly
    // in addition to native form submit (which still fires if the user hits
    // Cmd/Ctrl+Enter inside the textarea).
    const submitRfiAnswer = async (idx) => {
      const form = document.querySelector(`[data-aa-rfi-answer="${idx}"]`);
      if (!form) return;
      const ans = (form.elements.answer && form.elements.answer.value || "").trim();
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
    };
    document.querySelectorAll("[data-aa-rfi-answer]").forEach((form) => {
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        await submitRfiAnswer(Number(form.dataset.aaRfiAnswer));
      });
      // Allow Cmd/Ctrl+Enter inside the textarea to submit, since the actual
      // Submit button is in another <td> outside the form.
      const ta = form.querySelector("textarea[name='answer']");
      if (ta) {
        ta.addEventListener("keydown", (event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            submitRfiAnswer(Number(form.dataset.aaRfiAnswer));
          }
        });
      }
    });
    document.querySelectorAll("[data-aa-rfi-submit]").forEach((b) => {
      b.addEventListener("click", () => submitRfiAnswer(Number(b.dataset.aaRfiSubmit)));
    });
    // Edit a previously-answered RFI: clear the saved answer so the next
    // render shows the textarea + submit button again, with the prior text
    // pre-filled so the user can refine rather than retype.
    document.querySelectorAll("[data-aa-rfi-edit]").forEach((b) => {
      b.addEventListener("click", () => {
        const idx = Number(b.dataset.aaRfiEdit);
        const ctx = aaActiveThread(aa);
        const round = ctx.thread.rounds[idx];
        if (!round) return;
        const previous = round.answer || "";
        round.answer = "";
        round.answeredAt = null;
        round._editPrev = previous;
        saveProject(project);
        render();
        // After re-render, find the textarea for this row and prefill it.
        const form = document.querySelector(`[data-aa-rfi-answer="${idx}"]`);
        const ta = form && form.querySelector("textarea[name='answer']");
        if (ta) { ta.value = previous; ta.focus(); }
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
        renderAAMaster(project, aa), getFirm());
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
      setStatus(`<span class="muted">Nothing to draft yet. Answer at least one RFI on a thread first.</span>`);
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
      setStatus(`<span class="muted">Drafted ${done} of ${candidates.length}. ${failed} failed; try Draft again on the affected items.</span>`);
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
        // Surface the cover-page extras (ABN, addresses, contract date,
        // site address, ANA) so the engine can include them in introductions /
        // background threads where useful, instead of leaving [bracketed
        // placeholders]. Stripped of empty fields.
        coverMeta: Object.fromEntries(Object.entries(aa.coverMeta || {}).filter(([, v]) => typeof v === "string" && v.trim().length > 0)),
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

  // Drives the master document's "Generate executive summary" action.
  // Builds a digest of every drafted thread and sends it to the backend,
  // which produces a 4-6 paragraph HTML summary suitable for the top of the
  // master. Cheap to re-run when an item is re-drafted.
  async function aaCallExecSummary(project, aa) {
    const digest = [];
    function headlineFromHtml(html) {
      // Strip tags and grab the first ~600 chars — enough for the model to
      // understand the thread's punchline without bloating the prompt.
      const txt = String(html || "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
      return txt.slice(0, 600);
    }
    if (aa.jurisdictionalRfis && (aa.jurisdictionalRfis.submissions || "").length) {
      digest.push({
        kind: "jurisdictional",
        label: "Jurisdiction",
        issueType: "jurisdiction",
        status: "in-issue",
        claimed: 0,
        scheduled: 0,
        headline: headlineFromHtml(aa.jurisdictionalRfis.submissions),
      });
    }
    (aa.disputes || []).forEach((d) => {
      const subs = (d.rfis && d.rfis.submissions) || "";
      if (!subs && !d.psReasons) return;
      digest.push({
        kind: "dispute",
        label: d.item || "Item",
        issueType: d.issueType || "other",
        status: d.status || "disputed",
        claimed: Number(d.claimed || 0),
        scheduled: Number(d.scheduled || 0),
        headline: headlineFromHtml(subs) || (d.psReasons ? `Respondent's reasons: ${d.psReasons}` : ""),
      });
    });

    const payload = {
      parties: aa.parties || {},
      contractReference: aa.contractReference || "",
      referenceDate: aa.referenceDate || "",
      claimedAmount: Number(aa.claimedAmount || 0),
      scheduledAmount: Number(aa.scheduledAmount || 0),
      s79Scenario: aa.s79Scenario || "less-than-claimed",
      threadDigest: digest,
      projectMeta: { name: project.name, contractForm: project.contractForm },
    };
    const ctrl = new AbortController();
    const timeoutId = setTimeout(() => ctrl.abort(), 90_000);
    try {
      const response = await fetch("/api/sopal-v2/complex/aa/exec-summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(payload),
        signal: ctrl.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(describeApiError(data, "Could not generate the executive summary"));
      const html = (data && data.summaryHtml) || "";
      if (!html) throw new Error("The executive summary came back empty. Try again.");
      aa.execSummaryHtml = html;
      saveProject(project);
      return html;
    } catch (err) {
      if (err.name === "AbortError") throw new Error("Sopal took too long to generate the executive summary (over 90s). Try again.");
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // Generic "edit any HTML field on the AA matter" modal. Used for the cover
  // meta, introduction, exec summary edits and overarching arguments. Keeps
  // the AA matter fluid — the user can hand-edit any optional section without
  // running the engine.
  // After save / cancel the master modal is re-opened so the user doesn't lose
  // their place in the master review flow.
  function openAAEditModal({ project, aa, title, hint, getValue, setValue, mode, returnToMaster }) {
    const initial = getValue() || "";
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal aa-edit-modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>${escapeHtml(title)}</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <div class="modal-body">
              ${hint ? `<p class="muted aa-edit-hint">${escapeHtml(hint)}</p>` : ""}
              ${mode === "html"
                ? `<div class="aa-edit-html" contenteditable="true" data-aa-edit-html>${initial || "<p></p>"}</div>`
                : `<textarea class="text-area auto-grow" data-aa-edit-textarea rows="12" placeholder="Type or paste content here…">${escapeHtml(initial)}</textarea>`}
            </div>
            <div class="modal-foot aa-edit-foot">
              <button class="ghost-button compact" type="button" data-aa-edit-clear>Clear</button>
              <span class="spacer"></span>
              <button class="ghost-button compact" type="button" data-modal-close>Cancel</button>
              <button class="dark-button compact" type="button" data-aa-edit-save>Save</button>
            </div>
          </div>
        </div>`,
      bind: (rootEl) => {
        // Closing without saving still returns the user to the master modal —
        // they meant to be reviewing the master, the edit sub-modal is just a
        // detour. The reopen happens after the close so the modal slot is
        // free.
        const close = () => {
          modal = null;
          render();
          if (returnToMaster) setTimeout(() => openAAMasterModal(project, aa), 0);
        };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        rootEl.querySelector("[data-aa-edit-clear]")?.addEventListener("click", () => {
          if (mode === "html") {
            const ed = rootEl.querySelector("[data-aa-edit-html]");
            if (ed) ed.innerHTML = "<p></p>";
          } else {
            const t = rootEl.querySelector("[data-aa-edit-textarea]");
            if (t) t.value = "";
          }
        });
        rootEl.querySelector("[data-aa-edit-save]")?.addEventListener("click", () => {
          if (mode === "html") {
            const ed = rootEl.querySelector("[data-aa-edit-html]");
            const html = ed ? ed.innerHTML.trim() : "";
            // Treat an empty <p></p> shell as cleared content
            const stripped = html.replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ").trim();
            setValue(stripped ? html : "");
          } else {
            const t = rootEl.querySelector("[data-aa-edit-textarea]");
            setValue((t && t.value) || "");
          }
          saveProject(project);
          close();
        });
      },
    };
    render();
  }

  // Cover-meta editor — small form for the bordered cover-page tables.
  function openAACoverMetaModal(project, aa, returnToMaster) {
    const c = aa.coverMeta || {};
    // Grouped so the modal reads as "application context → claimant →
    // respondent" rather than 17 mixed fields.
    const groups = [
      {
        title: "Application context",
        fields: [
          { key: "applicationDate", label: "Application date", placeholder: "e.g. 12 May 2026" },
          { key: "ana", label: "Authorised Nominating Authority", placeholder: "e.g. Adjudicate Today" },
          { key: "anaReference", label: "ANA reference", placeholder: "e.g. ATO-12345" },
          { key: "contractDate", label: "Contract executed on", placeholder: "e.g. 12 March 2025" },
          { key: "siteAddress", label: "Project / site address", placeholder: "e.g. 123 Sample St, Brisbane QLD 4000" },
          { key: "pcDate", label: "Payment claim served on", placeholder: "e.g. 5 March 2026" },
          { key: "psDate", label: "Payment schedule served on", placeholder: "e.g. 19 March 2026 (or N/A)" },
        ],
      },
      {
        title: "Claimant",
        fields: [
          { key: "claimantAbn", label: "ABN", placeholder: "e.g. 12 345 678 901" },
          { key: "claimantContact", label: "Contact (name, role)", placeholder: "" },
          { key: "claimantAddress", label: "Address", placeholder: "" },
          { key: "claimantPhone", label: "Phone", placeholder: "" },
          { key: "claimantEmail", label: "Email", placeholder: "" },
        ],
      },
      {
        title: "Respondent",
        fields: [
          { key: "respondentAbn", label: "ABN", placeholder: "e.g. 12 345 678 901" },
          { key: "respondentContact", label: "Contact (name, role)", placeholder: "" },
          { key: "respondentAddress", label: "Address", placeholder: "" },
          { key: "respondentPhone", label: "Phone", placeholder: "" },
          { key: "respondentEmail", label: "Email", placeholder: "" },
        ],
      },
    ];
    // Flat list still used by the save handler. The order of the groups
    // above must reach every key the legacy field array used to.
    const fields = groups.flatMap((g) => g.fields);
    modal = {
      render: () => `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal aa-cover-modal" role="dialog" aria-modal="true">
            <div class="modal-head">
              <h2>Cover page details</h2>
              <button class="icon-button" type="button" data-modal-close aria-label="Close">${ICON.close}</button>
            </div>
            <div class="modal-body">
              <p class="muted aa-edit-hint">These fields populate the bordered tables on the master document's cover page. Leave blank to omit the row.</p>
              ${groups.map((g) => `
                <h4 class="aa-cover-group-title">${escapeHtml(g.title)}</h4>
                <div class="aa-cover-grid">
                  ${g.fields.map((f) => `
                    <label class="aa-cover-field">
                      <span>${escapeHtml(f.label)}</span>
                      <input class="text-input" data-aa-cover-key="${attr(f.key)}" type="text" value="${attr(c[f.key] || "")}" placeholder="${attr(f.placeholder || "")}">
                    </label>
                  `).join("")}
                </div>
              `).join("")}
            </div>
            <div class="modal-foot aa-edit-foot">
              <span class="spacer"></span>
              <button class="ghost-button compact" type="button" data-modal-close>Cancel</button>
              <button class="dark-button compact" type="button" data-aa-cover-save>Save</button>
            </div>
          </div>
        </div>`,
      bind: (rootEl) => {
        const close = () => {
          modal = null;
          render();
          if (returnToMaster) setTimeout(() => openAAMasterModal(project, aa), 0);
        };
        rootEl.querySelector("[data-modal-backdrop]")?.addEventListener("click", (e) => { if (e.target.matches("[data-modal-backdrop]")) close(); });
        rootEl.querySelectorAll("[data-modal-close]").forEach((b) => b.addEventListener("click", close));
        rootEl.querySelector("[data-aa-cover-save]")?.addEventListener("click", () => {
          if (!aa.coverMeta) aa.coverMeta = {};
          rootEl.querySelectorAll("[data-aa-cover-key]").forEach((inp) => {
            aa.coverMeta[inp.dataset.aaCoverKey] = (inp.value || "").trim();
          });
          saveProject(project);
          close();
        });
      },
    };
    render();
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
      <p>[Substantive body. Facts, contractual references, and any action requested.]</p>
      <p>[Closing paragraph. Reservation of rights as appropriate.]</p>
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

  // Pre-fill the user's firm name into "[Claimant name]" / "[Contractor name]"
  // / "[Sender name, position, company]" placeholders in the standard
  // drafting templates so a firm with branding configured doesn't have to
  // hand-fill the obvious slot every time. Only the agent-side "self"
  // placeholders are touched — counterparty placeholders ([Respondent name],
  // [Recipient name]) are left alone for the user to fill.
  function applyFirmToDraftTemplate(html, firm, agentKey) {
    if (!firm || !firm.name) return html;
    const name = firm.name;
    const firmRoles = ["[Claimant name]", "[Contractor name]"];
    let out = html;
    for (const placeholder of firmRoles) {
      // Replace every occurrence so all the bullet-point and header
      // references get filled at once.
      out = out.split(placeholder).join(escapeHtml(name));
    }
    // The general-correspondence template uses a richer "[Sender name,
    // position, company]" line — only fill the company slot, leave name +
    // position for the user.
    out = out.replace(/\[Sender name, position, company\]/g, `[Sender name], [Position], ${escapeHtml(name)}`);
    // Payment Schedule template: the issuer is the Respondent role; do
    // not touch Respondent placeholders. The Variations / EOTs / Delay
    // Costs templates use [Contractor name] which we just covered.
    return out;
  }

  // Draft storage. Each agentKey holds an ARRAY of drafting instances so a
  // single project can carry multiple Payment Claims (e.g. March + April
  // progress claims), multiple EOT notices, etc. The legacy single-draft
  // shape is migrated transparently on first read: any pre-existing object
  // becomes the first instance with id "legacy" so users don't lose work.
  function newDraftInstanceId() { return `dr_${Math.random().toString(36).slice(2, 10)}`; }
  function defaultDraftLabel(agentKey, index) {
    const base = AGENT_LABELS[agentKey] || "Draft";
    return index === 0 ? base : `${base} ${index + 1}`;
  }
  function migrateDraftsForAgent(project, agentKey) {
    if (!project.drafts) project.drafts = {};
    const existing = project.drafts[agentKey];
    if (Array.isArray(existing)) return existing;
    if (existing && typeof existing === "object" && (existing.html !== undefined || existing.chat !== undefined)) {
      // Legacy single-draft shape — wrap as the first instance.
      const inst = {
        id: existing.id || "legacy",
        label: existing.label || defaultDraftLabel(agentKey, 0),
        html: typeof existing.html === "string" ? existing.html : "",
        chat: existing.chat && Array.isArray(existing.chat.messages) ? existing.chat : { messages: [] },
        createdAt: existing.createdAt || existing.updatedAt || Date.now(),
        updatedAt: existing.updatedAt || Date.now(),
      };
      project.drafts[agentKey] = [inst];
      saveProject(project);
      return project.drafts[agentKey];
    }
    project.drafts[agentKey] = [];
    return project.drafts[agentKey];
  }
  function getDraftInstances(project, agentKey) {
    return migrateDraftsForAgent(project, agentKey);
  }
  function findDraftInstance(project, agentKey, instanceId) {
    if (!instanceId) return null;
    const list = getDraftInstances(project, agentKey);
    return list.find((d) => d.id === instanceId) || null;
  }
  function createDraftInstance(project, agentKey, label) {
    const list = getDraftInstances(project, agentKey);
    const tpl = (AGENT_TEMPLATES[agentKey] || "<p>[Start drafting…]</p>").trim();
    const inst = {
      id: newDraftInstanceId(),
      label: (label || "").trim() || defaultDraftLabel(agentKey, list.length),
      html: applyFirmToDraftTemplate(tpl, getFirm(), agentKey),
      chat: { messages: [] },
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    list.push(inst);
    saveProject(project);
    return inst;
  }
  function renameDraftInstance(project, agentKey, instanceId, label) {
    const inst = findDraftInstance(project, agentKey, instanceId);
    if (!inst) return;
    inst.label = String(label || "").trim() || inst.label;
    inst.updatedAt = Date.now();
    saveProject(project);
  }
  function deleteDraftInstance(project, agentKey, instanceId) {
    const list = getDraftInstances(project, agentKey);
    const idx = list.findIndex((d) => d.id === instanceId);
    if (idx >= 0) {
      list.splice(idx, 1);
      saveProject(project);
    }
  }

  // The original accessor. Used by code paths that don't know an instance id
  // (e.g. saveProject sync). Returns the first instance, creating it if the
  // project has none. New callers should prefer findDraftInstance + an iid.
  function getProjectDraft(project, agentKey, instanceId) {
    const list = getDraftInstances(project, agentKey);
    if (instanceId) {
      const found = list.find((d) => d.id === instanceId);
      if (found) {
        if (!found.chat || !Array.isArray(found.chat.messages)) found.chat = { messages: [] };
        return found;
      }
    }
    if (list.length === 0) createDraftInstance(project, agentKey, "");
    const head = list[0];
    if (!head.chat || !Array.isArray(head.chat.messages)) head.chat = { messages: [] };
    return head;
  }

  function bindDraftInstanceBar(project, agentKey, draft, draftOnly) {
    const renameEl = document.querySelector("[data-draft-rename]");
    renameEl?.addEventListener("change", () => {
      renameDraftInstance(project, agentKey, draft.id, renameEl.value);
    });
    renameEl?.addEventListener("blur", () => {
      renameDraftInstance(project, agentKey, draft.id, renameEl.value);
    });
    const switchEl = document.querySelector("[data-draft-switch]");
    switchEl?.addEventListener("change", () => {
      navigate(`/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft&iid=${switchEl.value}`);
    });
    document.querySelector("[data-draft-new]")?.addEventListener("click", () => {
      const inst = createDraftInstance(project, agentKey, "");
      navigate(`/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft&iid=${inst.id}`);
    });
    document.querySelector("[data-draft-duplicate]")?.addEventListener("click", () => {
      const list = getDraftInstances(project, agentKey);
      const inst = {
        id: newDraftInstanceId(),
        label: `${draft.label || defaultDraftLabel(agentKey, 0)} (copy)`,
        html: draft.html,
        chat: { messages: [] },
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      list.push(inst);
      saveProject(project);
      navigate(`/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft&iid=${inst.id}`);
    });
    document.querySelector("[data-draft-delete]")?.addEventListener("click", () => {
      const label = draft.label || defaultDraftLabel(agentKey, 0);
      if (!confirm(`Delete the draft "${label}"? This cannot be undone.`)) return;
      deleteDraftInstance(project, agentKey, draft.id);
      navigate(`/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft`);
    });
  }

  function DraftInstanceListPage(project, agentKey, instances, draftOnly) {
    const reviewHref = `/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=review`;
    const tabs = draftOnly ? "" : `
      <div class="mode-tabs" role="tablist">
        <button class="mode-tab" type="button" data-go="${reviewHref}">Review</button>
        <button class="mode-tab active" type="button">Draft</button>
      </div>`;
    const sorted = instances.slice().sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
    return `
      <div class="page-shell">
        <div class="chat-page-head">
          <div>
            <h1 class="page-title">${escapeHtml(AGENT_LABELS[agentKey])}</h1>
            <p class="page-sub">${escapeHtml(AGENT_DESCRIPTIONS[agentKey] || "")}</p>
          </div>
          ${tabs}
        </div>
        <div class="drafts-listing-toolbar">
          <div class="muted">${sorted.length} draft${sorted.length === 1 ? "" : "s"} in this project</div>
          <button class="dark-button" type="button" data-draft-list-new>${ICON.plus}<span>New draft</span></button>
        </div>
        ${sorted.length === 0 ? `
          <div class="card-empty" style="margin-top:14px;">
            <div class="card-empty-icon">${ICON.file}</div>
            <h4>No drafts yet</h4>
            <p>Create a new draft and Sopal opens the Word-style editor with the firm template loaded.</p>
            <button class="dark-button" type="button" data-draft-list-new>${ICON.plus}<span>Start a draft</span></button>
          </div>
        ` : `
          <div class="drafts-listing-list">
            ${sorted.map((inst) => {
              const wordCount = approxDraftWordCount(inst.html);
              const updated = formatRelativeTimeShort(inst.updatedAt || inst.createdAt);
              return `
                <div class="draft-listing-row" data-draft-row="${attr(inst.id)}">
                  <button class="draft-listing-open" type="button" data-open-draft="${attr(inst.id)}">
                    <span class="draft-listing-icon">${ICON.file}</span>
                    <span class="draft-listing-text">
                      <strong>${escapeHtml(inst.label || defaultDraftLabel(agentKey, 0))}</strong>
                      <span class="muted">Updated ${escapeHtml(updated)} · ${wordCount} word${wordCount === 1 ? "" : "s"}</span>
                    </span>
                  </button>
                  <div class="draft-listing-actions">
                    <button class="ghost-button compact" type="button" data-rename-draft="${attr(inst.id)}">Rename</button>
                    <button class="ghost-button compact" type="button" data-duplicate-draft="${attr(inst.id)}">Duplicate</button>
                    <button class="ghost-button compact danger" type="button" data-delete-draft="${attr(inst.id)}">Delete</button>
                  </div>
                </div>`;
            }).join("")}
          </div>
        `}
      </div>
    `;
  }

  function approxDraftWordCount(html) {
    if (!html) return 0;
    const text = String(html).replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
    if (!text) return 0;
    return text.split(" ").length;
  }
  function formatRelativeTimeShort(ts) {
    if (!ts) return "never";
    const diff = Date.now() - Number(ts);
    if (diff < 60_000) return "just now";
    if (diff < 3600_000) return `${Math.floor(diff / 60_000)} min ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3600_000)} hr ago`;
    if (diff < 7 * 86_400_000) return `${Math.floor(diff / 86_400_000)} d ago`;
    return new Date(ts).toLocaleDateString("en-AU");
  }

  function bindDraftInstanceList(project, agentKey, draftOnly) {
    document.querySelectorAll("[data-draft-list-new]").forEach((b) => b.addEventListener("click", () => {
      const inst = createDraftInstance(project, agentKey, "");
      navigate(`/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft&iid=${inst.id}`);
    }));
    document.querySelectorAll("[data-open-draft]").forEach((b) => b.addEventListener("click", () => {
      navigate(`/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft&iid=${b.dataset.openDraft}`);
    }));
    document.querySelectorAll("[data-rename-draft]").forEach((b) => b.addEventListener("click", () => {
      const iid = b.dataset.renameDraft;
      const inst = findDraftInstance(project, agentKey, iid);
      if (!inst) return;
      const next = prompt("Rename draft", inst.label || "");
      if (next === null) return;
      renameDraftInstance(project, agentKey, iid, next);
      render();
    }));
    document.querySelectorAll("[data-duplicate-draft]").forEach((b) => b.addEventListener("click", () => {
      const iid = b.dataset.duplicateDraft;
      const inst = findDraftInstance(project, agentKey, iid);
      if (!inst) return;
      const list = getDraftInstances(project, agentKey);
      list.push({
        id: newDraftInstanceId(),
        label: `${inst.label || defaultDraftLabel(agentKey, 0)} (copy)`,
        html: inst.html,
        chat: { messages: [] },
        createdAt: Date.now(),
        updatedAt: Date.now(),
      });
      saveProject(project);
      render();
    }));
    document.querySelectorAll("[data-delete-draft]").forEach((b) => b.addEventListener("click", () => {
      const iid = b.dataset.deleteDraft;
      const inst = findDraftInstance(project, agentKey, iid);
      if (!inst) return;
      if (!confirm(`Delete the draft "${inst.label || "Untitled"}"? This cannot be undone.`)) return;
      deleteDraftInstance(project, agentKey, iid);
      render();
    }));
  }

  function DraftingWorkspace(project, agentKey, draftOnly, instanceId) {
    const draft = getProjectDraft(project, agentKey, instanceId);
    const firm = getFirm();
    const dims = firmPageDimensions(firm);
    const margins = firmPageMargins(firm);
    const fontStack = firmFontFamily(firm);
    const accent = firm.accentColour || "#243043";

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

    const instances = getDraftInstances(project, agentKey);
    const listHref = `/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft`;
    const newHref = `/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft&iid=new`;
    const instanceBar = `
      <div class="draft-instance-bar">
        <div class="draft-instance-bar-left">
          <input class="draft-instance-title" type="text" value="${attr(draft.label || defaultDraftLabel(agentKey, 0))}" data-draft-rename placeholder="Untitled draft" aria-label="Draft name">
          ${instances.length > 1 ? `
            <select class="draft-instance-switch" data-draft-switch aria-label="Switch draft">
              ${instances.map((d) => `<option value="${attr(d.id)}" ${d.id === draft.id ? "selected" : ""}>${escapeHtml(d.label || defaultDraftLabel(agentKey, 0))}</option>`).join("")}
            </select>
          ` : ""}
        </div>
        <div class="draft-instance-bar-right">
          <a class="link-button small" href="${listHref}" data-nav>All drafts (${instances.length})</a>
          <button class="ghost-button compact" type="button" data-draft-duplicate>Duplicate</button>
          <button class="ghost-button compact" type="button" data-draft-new>${ICON.plus}<span>New</span></button>
          ${instances.length > 1 ? `<button class="ghost-button compact danger" type="button" data-draft-delete>Delete this draft</button>` : ""}
        </div>
      </div>`;

    const ctxCount = project.contracts.length + project.library.length;

    // Word-style font choices — names match what the user expects to see in
    // Word's font picker, with a CSS fallback chain so the editor still renders
    // sensibly if a particular font is not installed.
    const FONT_CHOICES = [
      { name: "Calibri",          stack: 'Calibri, "Carlito", "Helvetica Neue", Arial, sans-serif' },
      { name: "Times New Roman",  stack: '"Times New Roman", Times, "Liberation Serif", serif' },
      { name: "Arial",            stack: 'Arial, "Helvetica Neue", Helvetica, sans-serif' },
      { name: "Georgia",          stack: 'Georgia, "Source Serif Pro", serif' },
      { name: "Garamond",         stack: '"EB Garamond", Garamond, "Times New Roman", serif' },
      { name: "Verdana",          stack: 'Verdana, Geneva, sans-serif' },
      { name: "Cambria",          stack: 'Cambria, Georgia, serif' },
      { name: "Source Serif Pro", stack: '"Source Serif Pro", "Times New Roman", Georgia, serif' },
      { name: "Inter",            stack: 'Inter, "Segoe UI", Roboto, system-ui, sans-serif' },
    ];
    const SIZE_CHOICES = [9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 48, 72];
    const PALETTE = [
      "#000000", "#1a1a1a", "#3a3a3a", "#666666", "#a0a0a0", "#d4d4d4", "#ffffff",
      "#c00000", "#ff0000", "#ffc000", "#ffff00", "#92d050", "#00b050", "#00b0f0",
      "#0070c0", "#002060", "#7030a0", accent,
    ];
    const HIGHLIGHT_PALETTE = ["transparent", "#fff59d", "#a5d6a7", "#90caf9", "#f48fb1", "#ce93d8", "#ffab91", "#bcaaa4", "#ffd54f"];

    const fontOptions = FONT_CHOICES.map((f) => `<option value="${attr(f.stack)}" data-font-name="${attr(f.name)}" style="font-family:${attr(f.stack)}">${escapeHtml(f.name)}</option>`).join("");
    const sizeOptions = SIZE_CHOICES.map((s) => `<option value="${s}">${s}</option>`).join("");
    const colorSwatches = PALETTE.map((c) => `<button type="button" class="word-swatch" data-color="${attr(c)}" style="background:${attr(c)}" aria-label="${attr(c)}"></button>`).join("");
    const highlightSwatches = HIGHLIGHT_PALETTE.map((c) => `<button type="button" class="word-swatch ${c === "transparent" ? "is-clear" : ""}" data-highlight="${attr(c)}" style="background:${attr(c === "transparent" ? "#fff" : c)}" aria-label="${attr(c)}">${c === "transparent" ? "×" : ""}</button>`).join("");

    const firmRunningName = firm.name ? escapeHtml(firm.name) : "";
    const firmFooterLeft = firm.footerText ? escapeHtml(firm.footerText) : "";

    return `
      <div class="page-shell drafting-shell word-shell"
           style="--firm-accent:${attr(accent)};--firm-font:${attr(fontStack)};--firm-page-width:${dims.width}px;--firm-page-height:${dims.height}px;--firm-margin-top:${margins.top}px;--firm-margin-right:${margins.right}px;--firm-margin-bottom:${margins.bottom}px;--firm-margin-left:${margins.left}px;">
        ${head}
        ${instanceBar}

        <div class="word-ribbon" data-word-ribbon>
          <div class="ribbon-quick-row">
            <div class="ribbon-quick-left">
              <button class="ribbon-flat-btn" type="button" data-doc-undo title="Undo (⌘Z)"><span class="rb-glyph">↶</span><span>Undo</span></button>
              <button class="ribbon-flat-btn" type="button" data-doc-redo title="Redo (⌘⇧Z)"><span class="rb-glyph">↷</span><span>Redo</span></button>
              <span class="ribbon-quick-sep"></span>
              <button class="ribbon-flat-btn" type="button" data-doc-find title="Find &amp; replace (⌘F)"><span class="rb-glyph">⌕</span><span>Find</span></button>
              <span class="ribbon-quick-sep"></span>
              <span class="ribbon-savestate muted" data-drafting-savestate>Saved</span>
            </div>
            <div class="ribbon-quick-right">
              <span class="ribbon-page-readout" data-page-readout>Page <span data-page-current>1</span> of <span data-page-total>1</span></span>
              <span class="ribbon-quick-sep"></span>
              <button class="ribbon-flat-btn" type="button" data-doc-copy title="Copy document HTML">${ICON.copy}<span>Copy</span></button>
              <button class="ribbon-flat-btn" type="button" data-doc-print title="Print preview">${ICON.file}<span>Print</span></button>
              <button class="ribbon-flat-btn primary" type="button" data-doc-download title="Download .doc">${ICON.download}<span>.doc</span></button>
              <button class="ribbon-flat-btn danger" type="button" data-doc-reset title="Reset to blank template">Reset</button>
            </div>
          </div>

          <div class="ribbon-main-row">
            <div class="ribbon-group">
              <div class="ribbon-group-stack">
                <div class="ribbon-line ribbon-line-1">
                  <select class="ribbon-select ribbon-font-select" data-doc-font title="Font family">
                    ${fontOptions}
                  </select>
                  <select class="ribbon-select ribbon-size-select" data-doc-size title="Font size">
                    ${sizeOptions}
                  </select>
                  <button class="ribbon-icon-btn" type="button" data-doc-grow title="Grow font">A<span class="rb-grow">▲</span></button>
                  <button class="ribbon-icon-btn" type="button" data-doc-shrink title="Shrink font">A<span class="rb-grow">▼</span></button>
                </div>
                <div class="ribbon-line ribbon-line-2">
                  <button class="ribbon-icon-btn rb-bold" type="button" data-doc-cmd="bold" title="Bold (⌘B)"><strong>B</strong></button>
                  <button class="ribbon-icon-btn rb-italic" type="button" data-doc-cmd="italic" title="Italic (⌘I)"><em>I</em></button>
                  <button class="ribbon-icon-btn rb-underline" type="button" data-doc-cmd="underline" title="Underline (⌘U)"><u>U</u></button>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="strikeThrough" title="Strikethrough"><s>S</s></button>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="subscript" title="Subscript">x<sub>2</sub></button>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="superscript" title="Superscript">x<sup>2</sup></button>
                  <span class="ribbon-color-wrap">
                    <button class="ribbon-icon-btn rb-color" type="button" data-doc-color-toggle title="Font colour">A<span class="rb-color-bar" data-rb-color-bar style="background:#c00000"></span></button>
                    <div class="ribbon-color-pop" data-rb-color-pop hidden>${colorSwatches}<button type="button" class="ribbon-color-auto" data-color="auto">Automatic</button></div>
                  </span>
                  <span class="ribbon-color-wrap">
                    <button class="ribbon-icon-btn rb-highlight" type="button" data-doc-highlight-toggle title="Text highlight"><span class="rb-hl-icon">A</span><span class="rb-hl-bar" data-rb-hl-bar style="background:#fff59d"></span></button>
                    <div class="ribbon-color-pop" data-rb-hl-pop hidden>${highlightSwatches}</div>
                  </span>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="removeFormat" title="Clear formatting">A<span class="rb-clear">↺</span></button>
                </div>
              </div>
              <span class="ribbon-group-label">Font</span>
            </div>

            <div class="ribbon-group">
              <div class="ribbon-group-stack">
                <div class="ribbon-line ribbon-line-1">
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="insertUnorderedList" title="Bulleted list">•≡</button>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="insertOrderedList" title="Numbered list">1.≡</button>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="outdent" title="Decrease indent">←≡</button>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="indent" title="Increase indent">→≡</button>
                  <span class="ribbon-divider-vert"></span>
                  <select class="ribbon-select ribbon-linespace-select" data-doc-linespace title="Line spacing">
                    <option value="1.0">1.0</option>
                    <option value="1.15" selected>1.15</option>
                    <option value="1.5">1.5</option>
                    <option value="2.0">2.0</option>
                    <option value="2.5">2.5</option>
                    <option value="3.0">3.0</option>
                  </select>
                </div>
                <div class="ribbon-line ribbon-line-2">
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="justifyLeft" title="Align left"><span class="rb-align rb-align-l"></span></button>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="justifyCenter" title="Align centre"><span class="rb-align rb-align-c"></span></button>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="justifyRight" title="Align right"><span class="rb-align rb-align-r"></span></button>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="justifyFull" title="Justify"><span class="rb-align rb-align-j"></span></button>
                  <span class="ribbon-divider-vert"></span>
                  <button class="ribbon-icon-btn" type="button" data-doc-cmd="formatBlock" data-doc-block-arg="blockquote" title="Quote">"</button>
                  <button class="ribbon-icon-btn" type="button" data-doc-hr title="Horizontal line">─</button>
                </div>
              </div>
              <span class="ribbon-group-label">Paragraph</span>
            </div>

            <div class="ribbon-group">
              <div class="ribbon-group-stack">
                <div class="ribbon-line ribbon-line-1">
                  <select class="ribbon-select ribbon-style-select" data-doc-block-select title="Paragraph style">
                    <option value="">Style</option>
                    <option value="h1">Title</option>
                    <option value="h2">Heading 1</option>
                    <option value="h3">Heading 2</option>
                    <option value="h4">Heading 3</option>
                    <option value="p">Normal</option>
                  </select>
                </div>
                <div class="ribbon-line ribbon-line-2">
                  <span class="ribbon-style-preview rb-sp-h1">Title</span>
                  <span class="ribbon-style-preview rb-sp-h2">Heading 1</span>
                  <span class="ribbon-style-preview rb-sp-p">Normal</span>
                </div>
              </div>
              <span class="ribbon-group-label">Styles</span>
            </div>

            <div class="ribbon-group">
              <div class="ribbon-group-stack">
                <div class="ribbon-line ribbon-line-1">
                  <button class="ribbon-icon-btn" type="button" data-doc-table title="Insert table">⊞ Table</button>
                  <button class="ribbon-icon-btn" type="button" data-doc-image title="Insert image">🖼 Image</button>
                </div>
                <div class="ribbon-line ribbon-line-2">
                  <button class="ribbon-icon-btn" type="button" data-doc-page-break title="Insert page break">¶¶ Page break</button>
                  <button class="ribbon-icon-btn" type="button" data-doc-special title="Special character">Ω</button>
                </div>
              </div>
              <span class="ribbon-group-label">Insert</span>
            </div>
          </div>
        </div>

        <div class="word-find-bar" data-word-find-bar hidden>
          <div class="wfb-row">
            <label>Find <input class="text-input wfb-input" type="text" data-wfb-find></label>
            <label>Replace <input class="text-input wfb-input" type="text" data-wfb-replace></label>
            <button class="ghost-button compact" type="button" data-wfb-prev title="Previous">↑</button>
            <button class="ghost-button compact" type="button" data-wfb-next title="Next">↓</button>
            <button class="ghost-button compact" type="button" data-wfb-replace-one>Replace</button>
            <button class="ghost-button compact" type="button" data-wfb-replace-all>Replace all</button>
            <span class="wfb-status muted" data-wfb-status></span>
            <button class="icon-button" type="button" data-wfb-close aria-label="Close find">${ICON.close}</button>
          </div>
        </div>

        <div class="drafting-grid word-grid">
          <section class="drafting-doc-card word-doc-card card">
            <div class="word-canvas" data-word-canvas>
              <div class="word-page-stack" data-word-page-stack>
                <article class="word-page" data-word-page>
                  <div class="word-page-running-header" data-word-running-header>
                    <span class="wph-name">${firmRunningName}</span>
                  </div>
                  <div class="word-page-body">
                    <div class="drafting-editor word-editor" contenteditable="true" data-drafting-editor spellcheck="true">${draft.html}</div>
                    <div class="word-page-overlays" data-word-page-overlays aria-hidden="true"></div>
                  </div>
                  <div class="word-page-running-footer" data-word-running-footer>
                    <span class="wpf-left">${firmFooterLeft}</span>
                    <span class="wpf-right">Page <span data-page-current-foot>1</span> of <span data-page-total-foot>1</span></span>
                  </div>
                </article>
              </div>
            </div>
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

  function bindDraftingWorkspace(project, agentKey, draftOnly, instanceId) {
    const editor = document.querySelector("[data-drafting-editor]");
    const saveState = document.querySelector("[data-drafting-savestate]");
    const stream = document.querySelector("[data-drafting-chat-stream]");
    const form = document.querySelector("[data-drafting-chat-form]");
    if (!editor || !form) return;

    const draft = getProjectDraft(project, agentKey, instanceId);
    bindDraftInstanceBar(project, agentKey, draft, draftOnly);
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

    // ---- Word-style toolbar bindings ----------------------------------------
    // The browser's contenteditable + execCommand surface is enough for the
    // bulk of Word-style commands (B/I/U/S, sub/sup, lists, alignment, indent,
    // formatBlock, fontName, fontSize, foreColor, hiliteColor, undo/redo,
    // removeFormat, insertHTML). For richer features (insertable table, image
    // upload, find/replace, line spacing, page breaks) we add custom handlers
    // that operate on the current Selection.

    // Generic exec wrapper — focuses the editor first so the command lands on
    // the live Selection rather than the ribbon button itself.
    function exec(cmd, value) {
      editor.focus();
      try { document.execCommand(cmd, false, value === undefined ? null : value); } catch (_) {}
      scheduleSave();
      requestAnimationFrame(() => { recomputePages(); reflectSelectionState(); });
    }

    // Restore selection helper — when the user clicks a ribbon control the
    // editor loses focus, so we cache the selection range on every selection
    // change inside the editor and restore it before applying the command.
    let savedRange = null;
    function captureSelection() {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      const r = sel.getRangeAt(0);
      if (editor.contains(r.commonAncestorContainer)) savedRange = r.cloneRange();
    }
    function restoreSelection() {
      if (!savedRange) return;
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(savedRange);
    }
    document.addEventListener("selectionchange", () => {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      const r = sel.getRangeAt(0);
      if (editor.contains(r.commonAncestorContainer)) savedRange = r.cloneRange();
    });

    document.querySelectorAll("[data-doc-cmd]").forEach((btn) => btn.addEventListener("mousedown", (e) => {
      // mousedown so the editor's selection isn't lost when the button is
      // clicked. We restore + run inside a microtask.
      e.preventDefault();
      restoreSelection();
      const cmd = btn.dataset.docCmd;
      const arg = btn.dataset.docBlockArg || null;
      exec(cmd, arg);
    }));

    // Paragraph-style dropdown (Title / Heading 1 / Heading 2 / Heading 3 / Normal).
    const blockSelect = document.querySelector("[data-doc-block-select]");
    blockSelect?.addEventListener("change", () => {
      const v = blockSelect.value;
      if (!v) return;
      restoreSelection();
      exec("formatBlock", v === "p" ? "p" : v);
      blockSelect.value = "";
    });

    // Font family — execCommand("fontName", value) wraps the selection in a
    // <font face=...> tag, which contenteditable supports universally even
    // though it's deprecated. We accept the full CSS stack as the value so
    // that fallbacks render when the named font isn't installed.
    const fontSelect = document.querySelector("[data-doc-font]");
    if (fontSelect) {
      // Default to firm body font so new docs pick up the user's choice.
      const firmStack = firmFontFamily(getFirm());
      const matchOpt = Array.from(fontSelect.options).find((o) => o.value.toLowerCase() === firmStack.toLowerCase());
      fontSelect.value = matchOpt ? matchOpt.value : fontSelect.options[0].value;
      // Apply default to the editor so the cursor inherits it before any
      // text is typed.
      editor.style.fontFamily = fontSelect.value;
      fontSelect.addEventListener("change", () => {
        restoreSelection();
        exec("fontName", fontSelect.value);
        editor.style.fontFamily = fontSelect.value;
      });
    }

    // Font size — execCommand("fontSize") only accepts 1..7 buckets, which
    // is too coarse. We instead wrap the selection in a <span style="font-size:Npt">
    // by inserting HTML, preserving any existing inline formatting on the
    // selected text where possible.
    const sizeSelect = document.querySelector("[data-doc-size]");
    if (sizeSelect) {
      sizeSelect.value = "12";
      sizeSelect.addEventListener("change", () => {
        restoreSelection();
        applyFontSize(sizeSelect.value + "pt");
      });
    }
    function applyFontSize(size) {
      editor.focus();
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      // Use the legacy fontSize bucket then rewrite font tags to spans with
      // the precise pt size. Browser-native enough to keep undo working.
      document.execCommand("fontSize", false, "7");
      editor.querySelectorAll('font[size="7"]').forEach((node) => {
        const span = document.createElement("span");
        span.style.fontSize = size;
        while (node.firstChild) span.appendChild(node.firstChild);
        node.replaceWith(span);
      });
      scheduleSave();
      requestAnimationFrame(recomputePages);
    }

    document.querySelector("[data-doc-grow]")?.addEventListener("click", () => {
      restoreSelection();
      const cur = parseInt(sizeSelect?.value || "12", 10);
      const next = Math.min(72, cur + 2);
      if (sizeSelect) sizeSelect.value = String(next);
      applyFontSize(next + "pt");
    });
    document.querySelector("[data-doc-shrink]")?.addEventListener("click", () => {
      restoreSelection();
      const cur = parseInt(sizeSelect?.value || "12", 10);
      const next = Math.max(8, cur - 2);
      if (sizeSelect) sizeSelect.value = String(next);
      applyFontSize(next + "pt");
    });

    // Font color and highlight — popovers with palette swatches. Clicking a
    // swatch applies it via execCommand and updates the indicator bar on
    // the toolbar button so the user can re-apply with one click.
    const colorPop = document.querySelector("[data-rb-color-pop]");
    const colorBar = document.querySelector("[data-rb-color-bar]");
    document.querySelector("[data-doc-color-toggle]")?.addEventListener("click", (e) => {
      e.preventDefault();
      // Bare click (not on the chevron) just re-applies the current colour.
      if (colorBar && colorBar.dataset.lastColor) {
        restoreSelection();
        exec("foreColor", colorBar.dataset.lastColor);
        return;
      }
      togglePop(colorPop);
    });
    colorPop?.addEventListener("click", (e) => {
      const target = e.target.closest("[data-color]");
      if (!target) return;
      const c = target.dataset.color;
      restoreSelection();
      if (c === "auto") {
        exec("foreColor", "#1a1a1a");
        if (colorBar) { colorBar.style.background = "#1a1a1a"; colorBar.dataset.lastColor = "#1a1a1a"; }
      } else {
        exec("foreColor", c);
        if (colorBar) { colorBar.style.background = c; colorBar.dataset.lastColor = c; }
      }
      hidePop(colorPop);
    });

    const hlPop = document.querySelector("[data-rb-hl-pop]");
    const hlBar = document.querySelector("[data-rb-hl-bar]");
    document.querySelector("[data-doc-highlight-toggle]")?.addEventListener("click", (e) => {
      e.preventDefault();
      if (hlBar && hlBar.dataset.lastColor) {
        restoreSelection();
        applyHighlight(hlBar.dataset.lastColor);
        return;
      }
      togglePop(hlPop);
    });
    hlPop?.addEventListener("click", (e) => {
      const target = e.target.closest("[data-highlight]");
      if (!target) return;
      const c = target.dataset.highlight;
      restoreSelection();
      applyHighlight(c);
      if (hlBar) {
        hlBar.style.background = c === "transparent" ? "#fff" : c;
        hlBar.dataset.lastColor = c;
      }
      hidePop(hlPop);
    });

    function applyHighlight(color) {
      editor.focus();
      // hiliteColor works in Firefox; backColor is the cross-browser fallback.
      // Try hiliteColor first via execCommand("styleWithCSS"); on transparent
      // we need to remove the wrapping <span style="background:...">.
      try { document.execCommand("styleWithCSS", false, true); } catch (_) {}
      if (color === "transparent") {
        document.execCommand("hiliteColor", false, "transparent");
      } else {
        document.execCommand("hiliteColor", false, color);
      }
      scheduleSave();
      requestAnimationFrame(recomputePages);
    }

    function togglePop(el) {
      if (!el) return;
      document.querySelectorAll(".ribbon-color-pop").forEach((p) => { if (p !== el) p.hidden = true; });
      el.hidden = !el.hidden;
    }
    function hidePop(el) { if (el) el.hidden = true; }
    document.addEventListener("click", (e) => {
      if (e.target.closest(".ribbon-color-wrap")) return;
      document.querySelectorAll(".ribbon-color-pop").forEach((p) => p.hidden = true);
    });

    // Line spacing — applies to the block containing the current selection.
    const lineSelect = document.querySelector("[data-doc-linespace]");
    lineSelect?.addEventListener("change", () => {
      restoreSelection();
      editor.focus();
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      const range = sel.getRangeAt(0);
      // Walk up to the nearest block element from the selection's start; if
      // a range spans multiple blocks, apply to all blocks the range touches.
      const blocks = blocksInRange(range, editor);
      blocks.forEach((b) => { b.style.lineHeight = lineSelect.value; });
      scheduleSave();
      requestAnimationFrame(recomputePages);
    });

    function blocksInRange(range, root) {
      const startBlock = nearestBlock(range.startContainer, root);
      const endBlock = nearestBlock(range.endContainer, root);
      if (startBlock === endBlock) return [startBlock].filter(Boolean);
      const all = Array.from(root.querySelectorAll("p, h1, h2, h3, h4, h5, h6, li, blockquote, pre"));
      const startIdx = all.indexOf(startBlock);
      const endIdx = all.indexOf(endBlock);
      if (startIdx === -1 || endIdx === -1) return [startBlock, endBlock].filter(Boolean);
      return all.slice(startIdx, endIdx + 1);
    }
    function nearestBlock(node, root) {
      let n = node;
      while (n && n !== root) {
        if (n.nodeType === 1) {
          const tag = n.tagName.toLowerCase();
          if (["p","h1","h2","h3","h4","h5","h6","li","blockquote","pre","div"].includes(tag)) return n;
        }
        n = n.parentNode;
      }
      return null;
    }

    // Insert table — small dialog asks for rows and cols (defaults 3x3),
    // inserts a clean <table> with empty cells at the cursor position.
    document.querySelector("[data-doc-table]")?.addEventListener("click", (e) => {
      e.preventDefault();
      restoreSelection();
      const cols = Math.max(1, Math.min(20, parseInt(prompt("Columns?", "3") || "0", 10)));
      if (!cols) return;
      const rows = Math.max(1, Math.min(50, parseInt(prompt("Rows?", "3") || "0", 10)));
      if (!rows) return;
      let html = '<table style="width:100%; border-collapse:collapse; margin:8px 0;">';
      for (let r = 0; r < rows; r++) {
        html += "<tr>";
        for (let c = 0; c < cols; c++) {
          const cell = r === 0 ? "th" : "td";
          html += `<${cell} style="border:1pt solid #888; padding:5px 8px;">&nbsp;</${cell}>`;
        }
        html += "</tr>";
      }
      html += "</table><p>&nbsp;</p>";
      restoreSelection();
      editor.focus();
      document.execCommand("insertHTML", false, html);
      scheduleSave();
      requestAnimationFrame(recomputePages);
    });

    // Insert image — prompts for a file, embeds as data URL. Capped at ~2MB
    // so the autosave payload stays manageable.
    document.querySelector("[data-doc-image]")?.addEventListener("click", (e) => {
      e.preventDefault();
      restoreSelection();
      const inp = document.createElement("input");
      inp.type = "file";
      inp.accept = "image/png,image/jpeg,image/gif,image/webp";
      inp.addEventListener("change", () => {
        const file = inp.files && inp.files[0];
        if (!file) return;
        if (file.size > 2 * 1024 * 1024) {
          alert("Image is over 2 MB. Please use a smaller version (Sopal embeds images directly into the document, so very large files bloat the .doc export).");
          return;
        }
        const reader = new FileReader();
        reader.onload = () => {
          restoreSelection();
          editor.focus();
          const html = `<img src="${reader.result}" alt="" style="max-width:100%; height:auto; margin:6px 0;">`;
          document.execCommand("insertHTML", false, html);
          scheduleSave();
          requestAnimationFrame(recomputePages);
        };
        reader.readAsDataURL(file);
      });
      inp.click();
    });

    document.querySelector("[data-doc-hr]")?.addEventListener("click", (e) => {
      e.preventDefault();
      restoreSelection();
      exec("insertHorizontalRule");
    });

    document.querySelector("[data-doc-page-break]")?.addEventListener("click", (e) => {
      e.preventDefault();
      restoreSelection();
      editor.focus();
      // Visible "manual page break" — a div with a custom class that the
      // pagination layout treats as a hard break. The .doc export passes it
      // through as a Word page break via mso-page-break.
      const html = '<div class="manual-page-break" style="page-break-before:always; mso-special-character:line-break;">&nbsp;</div><p>&nbsp;</p>';
      document.execCommand("insertHTML", false, html);
      scheduleSave();
      requestAnimationFrame(recomputePages);
    });

    document.querySelector("[data-doc-special]")?.addEventListener("click", (e) => {
      e.preventDefault();
      const ch = prompt("Insert special character (e.g. § © ¶ ™ ↗ ✓ Ω π ½ ¼ £ €)", "§");
      if (!ch) return;
      restoreSelection();
      editor.focus();
      document.execCommand("insertText", false, ch);
      scheduleSave();
    });

    // Undo / Redo — use the browser's native contenteditable history.
    document.querySelector("[data-doc-undo]")?.addEventListener("click", () => exec("undo"));
    document.querySelector("[data-doc-redo]")?.addEventListener("click", () => exec("redo"));

    // ---- Find & Replace ----------------------------------------------------
    const findBar = document.querySelector("[data-word-find-bar]");
    const findInput = document.querySelector("[data-wfb-find]");
    const replaceInput = document.querySelector("[data-wfb-replace]");
    const findStatus = document.querySelector("[data-wfb-status]");
    let findMatches = [];
    let findIdx = -1;
    function runFind() {
      const q = (findInput?.value || "").trim();
      findMatches = [];
      findIdx = -1;
      clearFindHighlights();
      if (!q) { if (findStatus) findStatus.textContent = ""; return; }
      const lc = q.toLowerCase();
      // Walk text nodes and collect ranges. We mark them with <mark class=wfb-match>
      // for visibility; this is reverted on close.
      const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT);
      const nodes = [];
      let n;
      while ((n = walker.nextNode())) nodes.push(n);
      nodes.forEach((tn) => {
        const text = tn.nodeValue;
        const tl = text.toLowerCase();
        let from = 0;
        let idx;
        while ((idx = tl.indexOf(lc, from)) !== -1) {
          const range = document.createRange();
          range.setStart(tn, idx);
          range.setEnd(tn, idx + lc.length);
          findMatches.push(range);
          from = idx + lc.length;
        }
      });
      // Visually mark by wrapping; iterate in reverse so ranges remain valid.
      findMatches.slice().reverse().forEach((r) => {
        const mark = document.createElement("mark");
        mark.className = "wfb-match";
        try { r.surroundContents(mark); } catch (_) {}
      });
      if (findStatus) findStatus.textContent = findMatches.length ? `${findMatches.length} match${findMatches.length === 1 ? "" : "es"}` : "No matches";
      // Refresh ranges from the live <mark> elements (the surroundContents
      // above invalidates the original Range objects).
      findMatches = Array.from(editor.querySelectorAll("mark.wfb-match"));
      if (findMatches.length) gotoFindMatch(0);
    }
    function clearFindHighlights() {
      editor.querySelectorAll("mark.wfb-match").forEach((m) => {
        const parent = m.parentNode;
        while (m.firstChild) parent.insertBefore(m.firstChild, m);
        parent.removeChild(m);
        parent.normalize();
      });
    }
    function gotoFindMatch(i) {
      if (!findMatches.length) return;
      findIdx = ((i % findMatches.length) + findMatches.length) % findMatches.length;
      findMatches.forEach((m, k) => m.classList.toggle("is-active", k === findIdx));
      const target = findMatches[findIdx];
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    document.querySelector("[data-doc-find]")?.addEventListener("click", () => {
      if (!findBar) return;
      findBar.hidden = false;
      findInput?.focus();
    });
    findInput?.addEventListener("input", runFind);
    document.querySelector("[data-wfb-next]")?.addEventListener("click", () => gotoFindMatch(findIdx + 1));
    document.querySelector("[data-wfb-prev]")?.addEventListener("click", () => gotoFindMatch(findIdx - 1));
    document.querySelector("[data-wfb-replace-one]")?.addEventListener("click", () => {
      if (findIdx < 0 || findIdx >= findMatches.length) return;
      const node = findMatches[findIdx];
      const replacement = replaceInput?.value || "";
      const text = document.createTextNode(replacement);
      node.replaceWith(text);
      scheduleSave();
      runFind();
    });
    document.querySelector("[data-wfb-replace-all]")?.addEventListener("click", () => {
      if (!findMatches.length) return;
      const replacement = replaceInput?.value || "";
      findMatches.forEach((m) => m.replaceWith(document.createTextNode(replacement)));
      scheduleSave();
      runFind();
      requestAnimationFrame(recomputePages);
    });
    document.querySelector("[data-wfb-close]")?.addEventListener("click", () => {
      clearFindHighlights();
      findMatches = []; findIdx = -1;
      if (findStatus) findStatus.textContent = "";
      if (findBar) findBar.hidden = true;
    });
    // ⌘F in the editor opens find.
    editor.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "f") {
        e.preventDefault();
        document.querySelector("[data-doc-find]")?.click();
      }
    });

    // ---- Pagination overlays + page count ---------------------------------
    // We don't actually split the contenteditable into separate page DOM
    // nodes (that breaks selection across pages and is fragile to keep in
    // sync). Instead the editor is one tall white sheet, and we draw
    // horizontal "page break" rules + per-page "Page N of M" mini-footers
    // as absolutely-positioned overlays at every page-content-height
    // interval. Visually it reads as a stack of pages with content flowing
    // between them; structurally it's one contenteditable.
    const overlays = document.querySelector("[data-word-page-overlays]");
    const pageCurrentEls = document.querySelectorAll("[data-page-current], [data-page-current-foot]");
    const pageTotalEls = document.querySelectorAll("[data-page-total], [data-page-total-foot]");
    let recomputeTimer = null;
    function recomputePages() {
      if (!overlays) return;
      clearTimeout(recomputeTimer);
      recomputeTimer = setTimeout(_recomputePages, 80);
    }
    function _recomputePages() {
      const firmNow = getFirm();
      const dimsNow = firmPageDimensions(firmNow);
      const marginsNow = firmPageMargins(firmNow);
      // Each page's content area = page height − running header/footer strips
      // − top/bottom margins. This is the visual height between page-break
      // dividers in the editor.
      const contentH = Math.max(200, dimsNow.height - 56 - marginsNow.top - marginsNow.bottom);
      const editorH = editor.scrollHeight || editor.offsetHeight;
      // Inner content height (between the editor's own top + bottom padding
      // which mirror the page margins).
      const innerH = Math.max(0, editorH - marginsNow.top - marginsNow.bottom);
      const total = Math.max(1, Math.ceil(innerH / contentH));
      pageTotalEls.forEach((el) => { el.textContent = String(total); });
      // Draw the dividers — at y = top-margin + i × contentH — measured in
      // editor coordinates (overlays are positioned relative to the editor's
      // .word-page-body parent which starts at the editor's top).
      let html = "";
      for (let i = 1; i < total; i++) {
        const y = marginsNow.top + i * contentH;
        html += `<div class="word-page-divider" style="top:${y}px"><span class="wpd-label">End of page ${i} · Page ${i + 1}</span></div>`;
      }
      overlays.innerHTML = html;
      overlays.style.height = `${editorH}px`;
      updateCurrentPage(contentH, marginsNow.top);
    }
    function updateCurrentPage(contentH, marginTop) {
      let y = 0;
      const sel = window.getSelection();
      if (sel && sel.rangeCount > 0) {
        const r = sel.getRangeAt(0);
        if (editor.contains(r.commonAncestorContainer)) {
          const caretRect = r.getBoundingClientRect();
          const editorRect = editor.getBoundingClientRect();
          if (caretRect && caretRect.top) {
            // Caret y in editor coordinates (i.e. inside the editor element,
            // which already includes padding-top = margin-top).
            y = (caretRect.top - editorRect.top);
          }
        }
      }
      // Subtract the top margin so y=0 means "first line of page 1".
      const innerY = Math.max(0, y - (marginTop || 0));
      const cur = Math.max(1, Math.floor(innerY / contentH) + 1);
      pageCurrentEls.forEach((el) => { el.textContent = String(cur); });
    }
    // Debounced recompute on input, immediate on caret moves and resize.
    editor.addEventListener("input", () => recomputePages());
    function caretPagePing() {
      const m = firmPageMargins(getFirm());
      const d = firmPageDimensions(getFirm());
      const contentH = Math.max(200, d.height - 56 - m.top - m.bottom);
      updateCurrentPage(contentH, m.top);
    }
    editor.addEventListener("keyup", caretPagePing);
    editor.addEventListener("click", caretPagePing);
    window.addEventListener("resize", recomputePages);
    // Initial paint after layout.
    requestAnimationFrame(recomputePages);
    // Recompute again once fonts have loaded — letters of different fonts
    // change line wrapping which changes the page boundaries.
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => recomputePages()).catch(() => {});
    }

    // ---- Reflect selection state into ribbon controls ----------------------
    function reflectSelectionState() {
      const cmds = ["bold", "italic", "underline", "strikeThrough"];
      cmds.forEach((c) => {
        const btn = document.querySelector(`[data-doc-cmd="${c}"]`);
        if (!btn) return;
        try {
          btn.classList.toggle("is-active", document.queryCommandState(c));
        } catch (_) {}
      });
    }
    editor.addEventListener("keyup", reflectSelectionState);
    editor.addEventListener("mouseup", reflectSelectionState);

    // ---- File actions ------------------------------------------------------
    document.querySelector("[data-doc-copy]")?.addEventListener("click", () => {
      copyText(editor.innerHTML);
    });
    document.querySelector("[data-doc-print]")?.addEventListener("click", () => {
      // Persist any unsaved keystrokes first so the preview reflects what
      // the editor actually shows.
      persist();
      openFirmPrintPreview({
        title: `${project.name} — ${AGENT_LABELS[agentKey] || agentKey}`,
        bodyHtml: editor.innerHTML,
        firm: getFirm(),
      });
    });
    document.querySelector("[data-doc-download]")?.addEventListener("click", () => {
      // Persist before exporting so the .doc matches the editor.
      persist();
      const filename = `${project.name.replace(/[^a-z0-9]+/gi, "-")}-${agentKey}.doc`;
      // Reuse the AA .doc envelope so the firm header / footer / fonts
      // come along for the ride. Suppresses the cover-page-specific
      // overrides since drafting agents render top-down without a
      // separate cover.
      aaDownloadDoc(filename, `${project.name}: ${AGENT_LABELS[agentKey] || agentKey}`, editor.innerHTML, getFirm());
    });
    document.querySelector("[data-doc-reset]")?.addEventListener("click", () => {
      if (!confirm("Reset this draft back to the blank template? The current content will be lost.")) return;
      const tpl = (AGENT_TEMPLATES[agentKey] || "<p>[Start drafting…]</p>").trim();
      const filled = applyFirmToDraftTemplate(tpl, getFirm(), agentKey);
      editor.innerHTML = filled;
      draft.html = filled;
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
      const iid = params.get("iid");
      // ?iid=new creates a fresh instance and redirects to its editor. Used
      // by the "+ New" buttons in the sidebar and the drafts-list page.
      if (iid === "new") {
        const inst = createDraftInstance(project, agentKey, "");
        navigate(`/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft&iid=${inst.id}`);
        return PageBody("");
      }
      const instances = getDraftInstances(project, agentKey);
      // Bare ?mode=draft with multiple instances → show the picker so the
      // user can choose which draft to open. With zero or one instance we
      // fall through to the legacy single-doc editor (auto-creating the
      // first instance) so the experience is unchanged for new projects.
      if (!iid && instances.length > 1) {
        setTimeout(() => bindDraftInstanceList(project, agentKey, draftOnly), 0);
        return PageBody(DraftInstanceListPage(project, agentKey, instances, draftOnly));
      }
      const targetInstance = iid ? findDraftInstance(project, agentKey, iid) : null;
      // Unknown iid → bounce to the drafts list rather than silently swapping
      // them onto another draft.
      if (iid && !targetInstance) {
        setTimeout(() => bindDraftInstanceList(project, agentKey, draftOnly), 0);
        return PageBody(DraftInstanceListPage(project, agentKey, instances, draftOnly));
      }
      const activeIid = targetInstance ? targetInstance.id : (instances[0] && instances[0].id) || null;
      setTimeout(() => bindDraftingWorkspace(project, agentKey, draftOnly, activeIid), 0);
      return PageBody(DraftingWorkspace(project, agentKey, draftOnly, activeIid));
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
        <p class="print-foot">Generated by Sopal v2 (sopal.com.au)</p>
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
                return `<option value="${i}">${escapeHtml(stamp)} · ${c.fail || 0} issues · ${c.warn || 0} warnings · ${c.pass || 0} passed</option>`;
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
          <p>The analysis is grounded in this document. Chase a specific item or draft an amendment.</p>
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
                <label>Claimant<input class="text-input" name="claimant" value="${attr(editing?.claimant || (editing ? "" : (getFirm().name || "")))}" placeholder="${attr(getFirm().name ? `Defaults to ${getFirm().name}` : "Claimant name")}"></label>
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
            <button class="jurisdiction-tab ${j.id === jur.id ? "active" : ""} ${j.supported ? "" : "limited"}" type="button" data-jurisdiction="${attr(j.id)}" title="${j.full}${j.supported ? "" : ": limited support (general knowledge only)"}">
              ${escapeHtml(j.label)}${j.supported ? "" : '<span class="jurisdiction-tag" aria-label="Limited support">·</span>'}
            </button>`).join("")}
        </div>
        ${jur.supported ? "" : `<span class="jurisdiction-warn muted">${escapeHtml(jur.full)} sources aren't yet integrated. Answers rely on general knowledge only; verify against the local act before relying.</span>`}
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
          <p>The analysis is grounded in this document. Chase a specific item or draft an amendment.</p>
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

  /* ---------- Settings page ---------- */

  // Single Settings page that surfaces every cross-cutting preference and
  // account thing in one place. Lives at /sopal-v2/settings. Relies on the
  // window.SopalAuth and window.SopalCloudSync objects exposed from the boot
  // module; both are tolerant of being absent.
  function SettingsPage() {
    const auth = window.SopalAuth || { state: "guest", user: null };
    const projects = projectList();
    const archived = archivedProjectList();
    const bytes = localStorageBytesUsed();
    const quota = 5 * 1024 * 1024;
    const pct = Math.min(100, Math.round((bytes / quota) * 100));
    const themeLabel = theme === "dark" ? "Dark" : "Light";

    setTimeout(() => bindSettingsActions(), 0);

    const accountCard = (() => {
      if (auth.state === "authed" && auth.user) {
        const display = (auth.user.first_name || auth.user.last_name)
          ? [auth.user.first_name, auth.user.last_name].filter(Boolean).join(" ")
          : auth.user.email;
        return `
          <div class="settings-card">
            <div class="settings-card-head">
              <h3>Account</h3>
              <span class="settings-pill settings-pill-on">Signed in</span>
            </div>
            <dl class="settings-dl">
              <dt>Name</dt><dd>${escapeHtml(display || "")}</dd>
              <dt>Email</dt><dd>${escapeHtml(auth.user.email || "")}</dd>
              ${auth.user.firm_name ? `<dt>Firm</dt><dd>${escapeHtml(auth.user.firm_name)}</dd>` : ""}
            </dl>
            <div class="settings-actions">
              <a class="ghost-button compact" href="/account.html" target="_blank" rel="noopener">Open account on sopal.com.au ${ICON.arrowUpRight}</a>
              <button class="ghost-button compact" type="button" data-sopal-signout>Sign out</button>
            </div>
          </div>`;
      }
      return `
        <div class="settings-card">
          <div class="settings-card-head">
            <h3>Account</h3>
            <span class="settings-pill settings-pill-off">Guest</span>
          </div>
          <p>You are using Sopal as a guest. Project content is stored in this browser only and will not follow you across devices. Sign in with your Sopal account to enable cloud sync.</p>
          <div class="settings-actions">
            <a class="dark-button compact" href="/login?redirect=${encodeURIComponent("/sopal-v2/settings")}">Sign in</a>
            <a class="ghost-button compact" href="/register?redirect=${encodeURIComponent("/sopal-v2/settings")}">Create an account</a>
          </div>
        </div>`;
    })();

    const cloudCard = (() => {
      if (auth.state === "authed") {
        return `
          <div class="settings-card">
            <div class="settings-card-head">
              <h3>Cloud sync</h3>
              <span class="settings-pill settings-pill-on">On</span>
            </div>
            <p>Your projects sync to your account automatically. New projects on this browser are pushed; projects on your account that this browser does not have are pulled the next time you sign in.</p>
            <div class="settings-actions">
              <button class="ghost-button compact" type="button" data-sopal-push-all>Push all local projects to cloud now</button>
              <button class="ghost-button compact" type="button" data-sopal-pull-missing>Pull missing projects from cloud</button>
            </div>
            <p class="muted settings-status" data-sopal-sync-status></p>
          </div>`;
      }
      return `
        <div class="settings-card settings-card-muted">
          <div class="settings-card-head">
            <h3>Cloud sync</h3>
            <span class="settings-pill settings-pill-off">Off (sign in)</span>
          </div>
          <p>Cloud sync becomes available once you sign in. Until then, every project lives in this browser only.</p>
        </div>`;
    })();

    const firm = getFirm();
    const firmCard = `
      <div class="settings-card firm-card">
        <div class="settings-card-head">
          <h3>Firm</h3>
          <span class="settings-pill ${firm.name ? "settings-pill-on" : "settings-pill-off"}">${firm.name ? "Configured" : "Not set"}</span>
        </div>
        <p>Firm-wide branding for documents Sopal drafts on your behalf. The logo, letterhead, footer, font and page size are applied to the AA master document and the six standalone drafting agents (Payment Claims, Payment Schedules, EOTs, Variations, Delay Costs, General Correspondence).</p>
        <form class="firm-form" data-firm-form autocomplete="off">
          <div class="firm-form-grid">
            <label class="firm-field firm-field-wide">
              <span>Firm name</span>
              <input class="text-input" name="name" type="text" maxlength="160" placeholder="e.g. Acme Builders Pty Ltd" value="${attr(firm.name || "")}">
              <small class="muted">Used as the default Claimant on new projects and on the cover page if no project Claimant is set.</small>
            </label>
            <label class="firm-field firm-field-wide">
              <span>Letterhead address</span>
              <textarea class="text-area" name="letterheadAddress" rows="3" maxlength="600" placeholder="Suite 12, Level 3&#10;123 Example Street&#10;Brisbane QLD 4000&#10;ABN 12 345 678 901">${escapeHtml(firm.letterheadAddress || "")}</textarea>
              <small class="muted">Renders top-right of the cover page on the master document.</small>
            </label>
            <label class="firm-field firm-field-wide">
              <span>Footer text (left side)</span>
              <input class="text-input" name="footerText" type="text" maxlength="200" placeholder="e.g. Acme Builders Pty Ltd · Adjudication Application · Confidential" value="${attr(firm.footerText || "")}">
              <small class="muted">Page number "Page X of Y" sits on the right; this text on the left.</small>
            </label>
            <label class="firm-field">
              <span>Body font</span>
              <select class="select-input" name="bodyFont">
                <option value="serif" ${firm.bodyFont === "serif" ? "selected" : ""}>Serif (Source Serif Pro / Times)</option>
                <option value="sans" ${firm.bodyFont === "sans" ? "selected" : ""}>Sans (Helvetica / Arial)</option>
                <option value="inter" ${firm.bodyFont === "inter" ? "selected" : ""}>Inter (modern sans)</option>
              </select>
            </label>
            <label class="firm-field">
              <span>Page size</span>
              <select class="select-input" name="pageSize">
                <option value="a4" ${firm.pageSize === "a4" ? "selected" : ""}>A4 (default for AU)</option>
                <option value="letter" ${firm.pageSize === "letter" ? "selected" : ""}>US Letter</option>
              </select>
            </label>
            <label class="firm-field">
              <span>Page margins</span>
              <select class="select-input" name="pageMargin">
                <option value="narrow" ${firm.pageMargin === "narrow" ? "selected" : ""}>Narrow (1.27 cm)</option>
                <option value="moderate" ${firm.pageMargin === "moderate" ? "selected" : ""}>Moderate (1.91 cm)</option>
                <option value="normal" ${(firm.pageMargin === "normal" || !firm.pageMargin) ? "selected" : ""}>Normal (2.54 cm)</option>
                <option value="wide" ${firm.pageMargin === "wide" ? "selected" : ""}>Wide (5.08 cm)</option>
              </select>
            </label>
            <label class="firm-field">
              <span>Heading numbering</span>
              <select class="select-input" name="headingNumbering">
                <option value="decimal" ${firm.headingNumbering === "decimal" ? "selected" : ""}>1, 2, 3</option>
                <option value="decimal-nested" ${firm.headingNumbering === "decimal-nested" ? "selected" : ""}>1.1, 1.1.1</option>
                <option value="alpha" ${firm.headingNumbering === "alpha" ? "selected" : ""}>(a), (b), (c)</option>
                <option value="roman" ${firm.headingNumbering === "roman" ? "selected" : ""}>I, II, III</option>
                <option value="none" ${firm.headingNumbering === "none" ? "selected" : ""}>None</option>
              </select>
            </label>
            <fieldset class="firm-field firm-field-wide firm-swatches">
              <legend>Accent colour</legend>
              <div class="swatch-row">
                ${["#243043","#1f4e3d","#7a2236","#5c4a8c","#0e6b8a","#8a5a2b"].map((hex) => `
                  <label class="swatch ${firm.accentColour === hex ? "selected" : ""}" style="--swatch:${hex}">
                    <input type="radio" name="accentColour" value="${hex}" ${firm.accentColour === hex ? "checked" : ""}>
                    <span class="swatch-chip" aria-hidden="true"></span>
                    <span class="swatch-label">${hex.toUpperCase()}</span>
                  </label>`).join("")}
              </div>
            </fieldset>
            <div class="firm-field firm-field-wide firm-logo-field">
              <span>Firm logo</span>
              <div class="firm-logo-row">
                <div class="firm-logo-preview" data-firm-logo-preview>
                  ${firm.logoDataUrl ? `<img src="${attr(firm.logoDataUrl)}" alt="Firm logo">` : `<span class="muted">No logo uploaded</span>`}
                </div>
                <div class="firm-logo-actions">
                  <label class="ghost-button compact">
                    ${ICON.upload}<span>Upload PNG / JPG</span>
                    <input type="file" hidden accept="image/png,image/jpeg" data-firm-logo-input>
                  </label>
                  ${firm.logoDataUrl ? `<button class="ghost-button compact danger" type="button" data-firm-logo-clear>Remove logo</button>` : ""}
                  <small class="muted" data-firm-logo-status>PNG or JPG. Sopal downscales anything over 250 KB so the .doc export stays portable.</small>
                </div>
              </div>
            </div>
          </div>
          <div class="firm-form-actions">
            <button class="dark-button compact" type="submit">Save firm settings</button>
            <span class="muted settings-status" data-firm-save-status></span>
          </div>
        </form>
        <details class="firm-preview-wrap">
          <summary>Preview cover page</summary>
          <div class="firm-preview-pane" data-firm-preview>${renderFirmPreviewPage(firm)}</div>
        </details>
      </div>`;

    const dataCard = `
      <div class="settings-card">
        <div class="settings-card-head"><h3>Data and storage</h3></div>
        <dl class="settings-dl">
          <dt>Active projects</dt><dd>${projects.length}</dd>
          <dt>Archived projects</dt><dd>${archived.length}</dd>
          <dt>Browser storage used</dt><dd>${formatBytes(bytes)} of about ${formatBytes(quota)} (${pct}%)</dd>
        </dl>
        <div class="settings-actions">
          <a class="ghost-button compact" href="/sopal-v2/projects" data-nav>Open project list</a>
          <button class="ghost-button compact danger" type="button" data-sopal-clear-local>Clear all local data</button>
        </div>
      </div>`;

    const appearanceCard = `
      <div class="settings-card">
        <div class="settings-card-head"><h3>Appearance</h3></div>
        <div class="settings-row">
          <div>
            <strong>Theme</strong>
            <p class="muted">Currently ${escapeHtml(themeLabel)}. Toggle from the header or with Cmd+Shift+D.</p>
          </div>
          <button class="ghost-button compact" type="button" data-toggle-theme>Switch to ${theme === "dark" ? "Light" : "Dark"}</button>
        </div>
      </div>`;

    return PageBody(`
      <div class="page-shell settings-shell">
        <h1 class="page-title">Settings</h1>
        <p class="page-sub">Account, cloud sync, firm branding, data and appearance for Sopal v2.</p>
        ${accountCard}
        ${cloudCard}
        ${firmCard}
        ${dataCard}
        ${appearanceCard}
      </div>
    `);
  }

  function bindSettingsActions() {
    document.querySelector("[data-sopal-push-all]")?.addEventListener("click", async (event) => {
      const btn = event.currentTarget;
      const status = document.querySelector("[data-sopal-sync-status]");
      btn.disabled = true;
      btn.textContent = "Pushing...";
      try {
        const r = await window.SopalCloudSync.pushAll();
        if (status) status.textContent = `Pushed ${r.pushed} project${r.pushed === 1 ? "" : "s"} to your account.`;
      } catch (err) {
        if (status) status.textContent = `Push failed: ${err.message || err}`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Push all local projects to cloud now";
      }
    });
    document.querySelector("[data-sopal-pull-missing]")?.addEventListener("click", async (event) => {
      const btn = event.currentTarget;
      const status = document.querySelector("[data-sopal-sync-status]");
      btn.disabled = true;
      btn.textContent = "Pulling...";
      try {
        await window.SopalCloudSync.pullMissing();
        if (status) status.textContent = "Pull complete. Any missing projects have been added to this browser.";
      } catch (err) {
        if (status) status.textContent = `Pull failed: ${err.message || err}`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Pull missing projects from cloud";
      }
    });
    document.querySelector("[data-sopal-clear-local]")?.addEventListener("click", () => {
      if (!confirm("Clear every project, recent decision and saved search from this browser? This does not affect your cloud-synced copies.")) return;
      try {
        localStorage.removeItem(STORE_KEY);
      } catch (_) {}
      window.location.replace("/sopal-v2");
    });
    bindFirmForm();
  }

  // The firm form lives inside the Settings page. Save commits the patch
  // and re-renders only the preview tile so the user doesn't lose focus.
  // The logo input runs an in-browser canvas downscale when the file is
  // bigger than 250 KB before saving — that keeps the .doc export portable
  // even if the user uploads a 4 MB PNG straight off their drive.
  function bindFirmForm() {
    const form = document.querySelector("[data-firm-form]");
    if (!form) return;
    const status = document.querySelector("[data-firm-save-status]");

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const fd = new FormData(form);
      const patch = {
        name: String(fd.get("name") || "").trim(),
        letterheadAddress: String(fd.get("letterheadAddress") || "").trim(),
        footerText: String(fd.get("footerText") || "").trim(),
        bodyFont: String(fd.get("bodyFont") || "serif"),
        pageSize: String(fd.get("pageSize") || "a4"),
        pageMargin: String(fd.get("pageMargin") || "normal"),
        accentColour: String(fd.get("accentColour") || "#243043"),
        headingNumbering: String(fd.get("headingNumbering") || "decimal"),
      };
      saveFirm(patch);
      if (status) {
        status.textContent = "Saved." + (window.SopalAuth && window.SopalAuth.state === "authed" ? " Syncing to your account." : "");
        setTimeout(() => { if (status.textContent.startsWith("Saved")) status.textContent = ""; }, 2400);
      }
      const previewMount = document.querySelector("[data-firm-preview]");
      if (previewMount) previewMount.innerHTML = renderFirmPreviewPage(getFirm());
    });

    // Live-update the preview without blowing away the form's focus.
    form.addEventListener("change", (event) => {
      if (event.target && event.target.matches('input[name="accentColour"], select[name="bodyFont"], select[name="pageSize"], select[name="pageMargin"], select[name="headingNumbering"]')) {
        const fd = new FormData(form);
        const previewFirm = {
          ...getFirm(),
          name: String(fd.get("name") || ""),
          letterheadAddress: String(fd.get("letterheadAddress") || ""),
          footerText: String(fd.get("footerText") || ""),
          bodyFont: String(fd.get("bodyFont") || "serif"),
          pageSize: String(fd.get("pageSize") || "a4"),
          pageMargin: String(fd.get("pageMargin") || "normal"),
          accentColour: String(fd.get("accentColour") || "#243043"),
          headingNumbering: String(fd.get("headingNumbering") || "decimal"),
        };
        const previewMount = document.querySelector("[data-firm-preview]");
        if (previewMount) previewMount.innerHTML = renderFirmPreviewPage(previewFirm);
        // Visually update swatch selection without re-rendering the field set.
        form.querySelectorAll(".swatch").forEach((el) => el.classList.toggle("selected", el.querySelector("input").checked));
      }
    });

    const logoInput = form.querySelector("[data-firm-logo-input]");
    const logoStatus = form.querySelector("[data-firm-logo-status]");
    logoInput?.addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      if (logoStatus) logoStatus.textContent = "Reading…";
      try {
        const dataUrl = await downscaleLogoIfNeeded(file, 250 * 1024);
        saveFirm({ logoDataUrl: dataUrl });
        render();
      } catch (err) {
        if (logoStatus) logoStatus.textContent = err.message || "Could not read that image.";
      }
    });

    document.querySelector("[data-firm-logo-clear]")?.addEventListener("click", () => {
      saveFirm({ logoDataUrl: "" });
      render();
    });
  }

  // Read an uploaded image into a data URL. If the file is larger than the
  // soft cap, downscale it through a canvas so the .doc export and the
  // localStorage blob stay manageable. Returns a base64 PNG data URL.
  function downscaleLogoIfNeeded(file, capBytes) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Could not read the file."));
      reader.onload = () => {
        const original = String(reader.result || "");
        // Below the cap: trust the original bytes and return them verbatim.
        if (original.length < capBytes) return resolve(original);
        const img = new Image();
        img.onerror = () => reject(new Error("That doesn't look like a valid PNG or JPG."));
        img.onload = () => {
          const maxDim = 800;
          const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
          const w = Math.round(img.width * scale);
          const h = Math.round(img.height * scale);
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0, w, h);
          // PNG is the safe bet for logos with transparency. If the file was
          // a JPG and is large because of photographic content, fall back to
          // a quality-tuned JPG.
          let out = canvas.toDataURL("image/png");
          if (out.length > capBytes) out = canvas.toDataURL("image/jpeg", 0.85);
          resolve(out);
        };
        img.src = original;
      };
      reader.readAsDataURL(file);
    });
  }

  // ---------- Firm-page rendering helpers ----------
  //
  // Used by the Settings preview, the AA master modal, and the drafting
  // agents' print preview. A single source of truth for letterhead +
  // footer means a Firm change anywhere in the app lights up everywhere.

  function firmHasBranding(firm) {
    if (!firm) return false;
    return !!(firm.name || firm.letterheadAddress || firm.footerText || firm.logoDataUrl);
  }
  function firmFontFamily(firm) {
    const f = (firm && firm.bodyFont) || "serif";
    if (f === "sans") return '"Helvetica Neue", Helvetica, Arial, sans-serif';
    if (f === "inter") return 'Inter, "Segoe UI", Roboto, system-ui, sans-serif';
    return '"Source Serif Pro", "Times New Roman", Georgia, serif';
  }
  function firmPageDimensions(firm) {
    // Visual page sizes in CSS pixels at 96dpi. A4 = 794×1123, Letter = 816×1056.
    const size = (firm && firm.pageSize) || "a4";
    if (size === "letter") return { width: 816, height: 1056, label: "Letter" };
    return { width: 794, height: 1123, label: "A4" };
  }

  // Word-style margin presets converted to CSS pixels at 96dpi (1 inch = 96px,
  // 1 cm ~= 37.795px). Returns { topPx, rightPx, bottomPx, leftPx, label }.
  function firmPageMargins(firm) {
    const key = (firm && firm.pageMargin) || "normal";
    const cmToPx = (cm) => Math.round(cm * 37.7952755906);
    if (key === "narrow")   return { top: cmToPx(1.27), right: cmToPx(1.27), bottom: cmToPx(1.27), left: cmToPx(1.27), label: "Narrow", cm: "1.27 cm" };
    if (key === "moderate") return { top: cmToPx(2.54), right: cmToPx(1.91), bottom: cmToPx(2.54), left: cmToPx(1.91), label: "Moderate", cm: "1.91 cm" };
    if (key === "wide")     return { top: cmToPx(2.54), right: cmToPx(5.08), bottom: cmToPx(2.54), left: cmToPx(5.08), label: "Wide", cm: "5.08 cm" };
    return { top: cmToPx(2.54), right: cmToPx(2.54), bottom: cmToPx(2.54), left: cmToPx(2.54), label: "Normal", cm: "2.54 cm" };
  }

  function renderFirmHeader(firm, isCover) {
    if (!firmHasBranding(firm)) return "";
    if (isCover) {
      const logo = firm.logoDataUrl ? `<div class="firm-cover-logo"><img src="${attr(firm.logoDataUrl)}" alt="${attr(firm.name || "Firm logo")}"></div>` : "";
      const address = firm.letterheadAddress
        ? `<div class="firm-cover-letterhead">${escapeHtml(firm.letterheadAddress).replace(/\n/g, "<br>")}</div>`
        : "";
      return `<header class="firm-page-header firm-cover-header">
        ${logo}
        ${address}
      </header>`;
    }
    // Subsequent pages: condensed firm-name strip top right.
    return `<header class="firm-page-header firm-running-header">
      <span class="firm-running-name">${escapeHtml(firm.name || "")}</span>
    </header>`;
  }

  function renderFirmFooter(firm, pageNum, pageCount, isCover) {
    if (!firmHasBranding(firm) && !pageCount) return "";
    if (isCover) {
      // Convention from formal QLD adjudication apps and legal letters: the
      // cover page has the letterhead at top, no page number at bottom.
      return "";
    }
    const left = firm && firm.footerText ? escapeHtml(firm.footerText) : "";
    return `<footer class="firm-page-footer">
      <span class="firm-footer-left">${left}</span>
      <span class="firm-footer-right">Page ${pageNum} of ${pageCount}</span>
    </footer>`;
  }

  // Settings preview tile: a miniature A4/Letter page that shows how the
  // user's branding choices will look.
  function renderFirmPreviewPage(firm) {
    const dims = firmPageDimensions(firm);
    const ratio = 320 / dims.width;
    const previewW = Math.round(dims.width * ratio);
    const previewH = Math.round(dims.height * ratio);
    const accent = (firm && firm.accentColour) || "#243043";
    const fontStack = firmFontFamily(firm);
    return `
      <div class="firm-preview-stage" style="--firm-accent:${attr(accent)};font-family:${attr(fontStack)}">
        <div class="firm-preview-sheet" style="width:${previewW}px;height:${previewH}px">
          <div class="firm-preview-inner">
            ${renderFirmHeader(firm, true)}
            <div class="firm-preview-body">
              <h1 class="firm-preview-h1">ADJUDICATION APPLICATION</h1>
              <p class="firm-preview-meta"><strong>Claimant:</strong> ${escapeHtml(firm && firm.name ? firm.name : "[Claimant name]")}</p>
              <p class="firm-preview-meta"><strong>Respondent:</strong> [Respondent name]</p>
              <p class="firm-preview-meta"><strong>Reference:</strong> [Contract reference]</p>
              <h2 class="firm-preview-h2">${firmNumberHeading(firm, 1)} Introduction</h2>
              <p class="firm-preview-p">This Adjudication Application is made by the Claimant under section 79 of the Building Industry Fairness (Security of Payment) Act 2017 (Qld).</p>
              <h2 class="firm-preview-h2">${firmNumberHeading(firm, 2)} Background</h2>
              <p class="firm-preview-p">The parties entered into a contract for the carrying out of construction work at the project site.</p>
            </div>
          </div>
        </div>
      </div>`;
  }

  // Paginate a long HTML string into discrete A4 / Letter sheets, each with a
  // page-1 letterhead treatment + page-N condensed header + page-N/M footer.
  //
  // Strategy: render the whole content into an off-screen probe so we can
  // measure heights, then walk the children and assign each one to the
  // current page until the cumulative height crosses the per-page content
  // budget. We bias on the conservative side (16pt tolerance) to leave room
  // for the footer and avoid clipping a final line.
  //
  // The mount node is replaced with N <article class="firm-paper"> sheets.
  // Used by the AA master modal and the drafting-editor print preview.
  function paintFirmPaperStack(mountEl, contentHtml, firm) {
    if (!mountEl) return;
    const dims = firmPageDimensions(firm);
    const margins = firmPageMargins(firm);
    const accent = (firm && firm.accentColour) || "#243043";
    const fontStack = firmFontFamily(firm);

    // Pre-render content into a hidden probe so we can measure children.
    // The probe has the same width as the page content area so layout
    // matches when we slice. Padding mirrors `.firm-paper-content`.
    const probe = document.createElement("div");
    probe.className = "firm-paper-probe";
    // Content width = page width minus left + right margins from Firm Settings.
    probe.style.width = `${dims.width - margins.left - margins.right}px`;
    // Match the paginated layout's font + size so measurements line up;
    // without this, switching the firm font from serif to sans would
    // silently drift the page boundaries.
    probe.style.fontFamily = fontStack;
    probe.style.fontSize = "11.5pt";
    probe.style.lineHeight = "1.55";
    probe.innerHTML = contentHtml;
    // The screen ToC nav is a navigation aid for the inline Stage 5 view,
    // not a document element. Strip it from the paginated layout so the
    // cover page lands on page 1 and so the printed/PDF output reads
    // like a real document.
    probe.querySelectorAll("nav.aa-toc").forEach((n) => n.remove());
    document.body.appendChild(probe);

    const children = Array.from(probe.children);
    // Page content height budget. Cover page eats more vertical space for
    // the letterhead block, so we drop its budget further. Subsequent
    // pages reserve room for the running header (44px) and footer (44px).
    const totalH = dims.height;
    const coverContentH = totalH - 320;     // letterhead block + bottom margin
    const runningContentH = totalH - 200;   // header + footer + top/bottom margin

    const pages = [[]];
    let curHeight = 0;
    let onCoverPage = true;
    let budget = coverContentH;

    function newPage() {
      pages.push([]);
      curHeight = 0;
      onCoverPage = false;
      budget = runningContentH;
    }

    children.forEach((child) => {
      const h = child.getBoundingClientRect().height;
      // Force the cover-page block to live alone on page 1 — it carries the
      // formal title + meta tables and the rest should reflow underneath
      // starting on page 2. This matches the way the user thinks about the
      // document and avoids the cover bleeding into a half-orphaned table.
      if (child.classList.contains("aa-cover")) {
        if (pages[pages.length - 1].length) newPage();
        pages[pages.length - 1].push(child);
        newPage();
        return;
      }
      // Single child taller than a page (eg. a long quantum table): give it
      // its own page rather than splitting it — Word handles intra-table
      // page breaks better than the browser does at this fidelity.
      if (h > budget && pages[pages.length - 1].length === 0) {
        pages[pages.length - 1].push(child);
        newPage();
        return;
      }
      if (curHeight + h > budget && pages[pages.length - 1].length) {
        newPage();
      }
      pages[pages.length - 1].push(child);
      curHeight += h;
    });

    // Drop trailing empty page that the cover-forces-newPage call may
    // create when there is no body content yet.
    if (pages.length > 1 && pages[pages.length - 1].length === 0) pages.pop();

    document.body.removeChild(probe);

    const pageCount = pages.length;
    mountEl.style.setProperty("--firm-accent", accent);
    mountEl.style.setProperty("--firm-page-width", `${dims.width}px`);
    mountEl.style.setProperty("--firm-page-height", `${dims.height}px`);
    mountEl.style.setProperty("--firm-font", fontStack);
    mountEl.style.setProperty("--firm-margin-top", `${margins.top}px`);
    mountEl.style.setProperty("--firm-margin-right", `${margins.right}px`);
    mountEl.style.setProperty("--firm-margin-bottom", `${margins.bottom}px`);
    mountEl.style.setProperty("--firm-margin-left", `${margins.left}px`);
    mountEl.classList.add("firm-paper-stack");

    const html = pages.map((nodes, i) => {
      const pageNum = i + 1;
      const isCover = i === 0;
      // Move the actual DOM nodes (not innerHTML) so contenteditable cover
      // tables in the DOM keep their state if anything ever needs it. We
      // simply collect their outerHTML for the page render.
      const inner = nodes.map((n) => n.outerHTML).join("");
      return `<article class="firm-paper ${isCover ? "firm-paper-cover" : "firm-paper-running"}" data-page="${pageNum}">
        ${renderFirmHeader(firm, isCover)}
        <div class="firm-paper-content">${inner}</div>
        ${renderFirmFooter(firm, pageNum, pageCount, isCover)}
      </article>`;
    }).join("");

    mountEl.innerHTML = html;
  }

  // Open a new browser tab containing a print-ready, paginated view of the
  // given content. Reused by the AA master modal and the drafting editor.
  function openFirmPrintPreview({ title, bodyHtml, firm }) {
    const win = window.open("", "_blank", "noopener");
    if (!win) {
      alert("Could not open the print preview. Please allow popups for this site and try again.");
      return;
    }
    const dims = firmPageDimensions(firm);
    const accent = (firm && firm.accentColour) || "#243043";
    const fontStack = firmFontFamily(firm);
    const pageSizeCss = dims.label === "Letter" ? "letter" : "a4";
    // Print stylesheet: @page rules size the printed page, the visual
    // .firm-paper sheets carry header/footer for screen viewing, and we
    // use page-break-before: always to enforce the paginated boundaries
    // when sent to the printer / "Save as PDF".
    const css = `
      :root { --firm-accent: ${accent}; --firm-font: ${fontStack}; }
      *, *::before, *::after { box-sizing: border-box; }
      html, body { margin: 0; padding: 0; background: #ece9e2; color: #1a1a1a; font-family: ${fontStack}; }
      body { font-size: 11.5pt; line-height: 1.55; }
      .firm-print-toolbar { position: sticky; top: 0; display: flex; gap: 8px; padding: 12px 16px; background: #fff; border-bottom: 1px solid #d6d2c6; z-index: 10; }
      .firm-print-toolbar button { font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid #aaa; background: #fff; cursor: pointer; }
      .firm-print-stage { padding: 24px 0; display: flex; flex-direction: column; align-items: center; gap: 18px; }
      .firm-paper { width: ${dims.width}px; min-height: ${dims.height}px; background: #fff; box-shadow: 0 4px 18px rgba(0,0,0,0.12); display: flex; flex-direction: column; position: relative; padding: 0; }
      .firm-paper-content { padding: 56px 96px 56px; flex: 1; }
      .firm-paper-cover .firm-paper-content { padding-top: 12px; }
      .firm-page-header { padding: 36px 56px 0; display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
      .firm-cover-header { padding-bottom: 14px; border-bottom: 2pt solid var(--firm-accent); }
      .firm-cover-logo img { max-height: 96px; max-width: 240px; object-fit: contain; }
      .firm-cover-letterhead { font-size: 10pt; line-height: 1.45; text-align: right; color: #2a2a2a; }
      .firm-running-header { justify-content: flex-end; padding-top: 18px; padding-bottom: 6px; border-bottom: 1pt solid #d6d2c6; }
      .firm-running-name { font-size: 9.5pt; letter-spacing: 0.04em; text-transform: uppercase; color: var(--firm-accent); font-weight: 600; }
      .firm-page-footer { display: flex; justify-content: space-between; align-items: center; padding: 8px 56px 24px; font-size: 9pt; color: #555; border-top: 1pt solid #d6d2c6; margin-top: auto; }
      .firm-footer-right { font-variant-numeric: tabular-nums; }
      h1 { font-size: 22pt; font-weight: 700; text-align: center; margin: 0 0 16px; color: #111; }
      h2 { font-size: 14pt; font-weight: 700; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1pt solid var(--firm-accent); color: var(--firm-accent); }
      h3 { font-size: 12pt; font-weight: 700; margin: 14px 0 6px; }
      p { margin: 0 0 10px; text-align: justify; hyphens: auto; }
      table { width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 10.5pt; }
      th, td { border: 1pt solid #888; padding: 5px 8px; text-align: left; vertical-align: top; }
      th { background: #f0ece4; font-weight: 600; }
      .aa-toc { display: none; }
      .aa-cover { padding: 0 0 18pt; margin: 0 0 24pt; border-bottom: 0; }
      .aa-cover-title { font-size: 24pt; letter-spacing: 0.04em; text-transform: uppercase; text-align: center; margin: 0 0 16pt; color: #111; }
      .aa-cover-opener { font-size: 11.5pt; margin: 0 0 18pt; text-align: justify; }
      .aa-cover-section { font-size: 10.5pt; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin: 16pt 0 6pt; border: 0; padding: 0; color: #111; }
      table.aa-cover-table { width: 100%; }
      table.aa-cover-table th { width: 38%; }
      table.aa-item-meta { width: auto; min-width: 60%; margin: 4px 0 12px; font-size: 10pt; }
      table.aa-item-meta th { width: 180px; background: #faf7f1; }
      .aa-issue-tag { display: inline-block; font-size: 9pt; padding: 1px 6px; margin-left: 6px; background: #e0e7ff; border-radius: 999px; }
      @page { size: ${pageSizeCss}; margin: 0; }
      @media print {
        body { background: #fff; }
        .firm-print-toolbar { display: none; }
        .firm-print-stage { padding: 0; gap: 0; }
        .firm-paper { box-shadow: none; page-break-after: always; min-height: ${dims.height}px; }
        .firm-paper:last-child { page-break-after: auto; }
      }
    `;
    win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>${escapeHtml(title)}</title><style>${css}</style></head><body>
      <div class="firm-print-toolbar"><button onclick="window.print()">Print / Save as PDF</button><button onclick="window.close()">Close</button></div>
      <div class="firm-print-stage" id="firm-print-stage"></div>
      <script>
        // Re-paginate inside the new window so we measure heights against
        // the print stylesheet, not the SPA's. Same algorithm as the SPA.
        (function () {
          var contentHtml = ${JSON.stringify(bodyHtml)};
          var firm = ${JSON.stringify(firm || {})};
          var dims = ${JSON.stringify(dims)};
          var stage = document.getElementById("firm-print-stage");
          var probe = document.createElement("div");
          probe.style.cssText = "position:absolute;left:-99999px;top:0;visibility:hidden;width:" + (dims.width - 192) + "px;font-family:" + ${JSON.stringify(fontStack)} + ";font-size:11.5pt;line-height:1.55;";
          probe.innerHTML = contentHtml;
          document.body.appendChild(probe);
          var kids = Array.prototype.slice.call(probe.children);
          var pages = [[]];
          var cur = 0;
          var onCover = true;
          var budget = dims.height - 320;
          function flush() { pages.push([]); cur = 0; onCover = false; budget = dims.height - 200; }
          kids.forEach(function (k) {
            var h = k.getBoundingClientRect().height;
            if (k.classList.contains("aa-cover")) {
              if (pages[pages.length - 1].length) flush();
              pages[pages.length - 1].push(k);
              flush();
              return;
            }
            if (h > budget && pages[pages.length - 1].length === 0) { pages[pages.length - 1].push(k); flush(); return; }
            if (cur + h > budget && pages[pages.length - 1].length) flush();
            pages[pages.length - 1].push(k);
            cur += h;
          });
          if (pages.length > 1 && pages[pages.length - 1].length === 0) pages.pop();
          document.body.removeChild(probe);
          function header(isCover) {
            if (!firm || (!firm.name && !firm.letterheadAddress && !firm.logoDataUrl && !firm.footerText)) return "";
            if (isCover) {
              var logo = firm.logoDataUrl ? '<div class="firm-cover-logo"><img src="' + firm.logoDataUrl + '" alt=""></div>' : "";
              var addr = firm.letterheadAddress ? '<div class="firm-cover-letterhead">' + firm.letterheadAddress.replace(/[<>&]/g, function (c) { return c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&amp;"; }).replace(/\\n/g, "<br>") + '</div>' : "";
              return '<header class="firm-page-header firm-cover-header">' + logo + addr + '</header>';
            }
            return '<header class="firm-page-header firm-running-header"><span class="firm-running-name">' + (firm.name || "").replace(/[<>&]/g, function (c) { return c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&amp;"; }) + '</span></header>';
          }
          function footer(n, total, isCover) {
            if (isCover) return "";
            return '<footer class="firm-page-footer"><span class="firm-footer-left">' + ((firm && firm.footerText || "").replace(/[<>&]/g, function (c) { return c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&amp;"; })) + '</span><span class="firm-footer-right">Page ' + n + ' of ' + total + '</span></footer>';
          }
          stage.innerHTML = pages.map(function (nodes, i) {
            var inner = nodes.map(function (n) { return n.outerHTML; }).join("");
            var isCover = i === 0;
            return '<article class="firm-paper ' + (isCover ? "firm-paper-cover" : "firm-paper-running") + '">' + header(isCover) + '<div class="firm-paper-content">' + inner + '</div>' + footer(i + 1, pages.length, isCover) + '</article>';
          }).join("");
        })();
      <\/script>
    </body></html>`);
    win.document.close();
  }

  // Converts a 1-based heading index into the user's chosen numbering style.
  // Used for the preview tile and (later) for the rendered master sections.
  function firmNumberHeading(firm, idx) {
    const style = (firm && firm.headingNumbering) || "decimal";
    if (style === "none") return "";
    if (style === "alpha") {
      // (a), (b), … (z), (aa), (ab), …
      let n = idx;
      let out = "";
      while (n > 0) { n -= 1; out = String.fromCharCode(97 + (n % 26)) + out; n = Math.floor(n / 26); }
      return `(${out})`;
    }
    if (style === "roman") {
      const romans = [["M",1000],["CM",900],["D",500],["CD",400],["C",100],["XC",90],["L",50],["XL",40],["X",10],["IX",9],["V",5],["IV",4],["I",1]];
      let n = idx; let out = "";
      for (const [sym, val] of romans) { while (n >= val) { out += sym; n -= val; } }
      return `${out}.`;
    }
    if (style === "decimal-nested") return `${idx}.`;
    return `${idx}.`;
  }

  /* ---------- Help & support system ---------- */

  // The help system is a self-contained set of articles routed under
  // /sopal-v2/help. Each article is plain HTML inside a `body` string. Articles
  // are categorised so the index page can present them in groups. We keep the
  // copy in one place (this constant) so it is easy to audit and update.
  const HELP_ARTICLES = [
    {
      slug: "getting-started",
      title: "Getting started with Sopal",
      category: "Start here",
      summary: "A quick tour of what Sopal does and how to set up your first project.",
      body: `
        <p class="lead">Sopal is a Queensland security-of-payment workspace. It helps you draft and review the documents that flow under the Building Industry Fairness (Security of Payment) Act 2017 (Qld) (the BIF Act): payment claims, payment schedules, adjudication applications, adjudication responses, extension of time notices, variation claims, delay-cost claims, and the day-to-day correspondence around a project.</p>

        <h2>What Sopal is, and what it is not</h2>
        <p>Sopal sits between a blank document and the engagement of a lawyer. It is not legal advice and it does not replace independent professional judgement on a difficult or high-stakes matter. What it does is help an experienced project manager, contract administrator or claimant build a solid first draft, sense-check timing, search the public adjudication record, and keep the paperwork organised in one place.</p>

        <h2>Pick a starting point</h2>
        <p>You can use Sopal in two distinct ways. Most people end up using both.</p>

        <h3>Standalone tools</h3>
        <p>The Tools group in the sidebar contains four utilities that work without needing a project. The Decision Search lets you search every public adjudication decision in the corpus by adjudicator, party, section reference or keyword. The Adjudicator Statistics page lets you size up a particular adjudicator before nominating one. The Payment Claim Reviewer and Payment Schedule Reviewer both accept a pasted document and return a structured BIF Act check. The Due Date Calculator and Interest Calculator handle the most common timing and money calculations.</p>

        <h3>Project workspaces</h3>
        <p>For anything that runs longer than a single document, create a project. A project is one construction contract. It carries the contract documents, a project library (correspondence, programme notes, RFIs, claims, schedules), a free-form chat assistant, the suite of drafting agents, and the Adjudication Application complex agent. Everything inside a project is stored in your browser and tied to that project, so the AI agents have proper context when they help you draft.</p>

        <h2>Create your first project</h2>
        <p>From the home page or the sidebar, click the plus icon next to Projects. Pick a short, identifiable project name (the actual contract reference works well: for example, "HC-2025-12 Riverside Apartments"). Choose the contract form (AS 4000, AS 4902, an in-house template, or other). Identify yourself as Claimant or Respondent for this matter. The project is saved as soon as you create it.</p>

        <h2>Add your contract</h2>
        <p>Open Contract from the project sub-navigation. Either paste the relevant clauses directly or drop the executed contract PDF into the upload zone. The agents in this project use this content to anchor their drafting in real clause numbers rather than bracketed placeholders. You can split a long contract into separate clause entries using the Detect clauses action so each clause is independently retrievable.</p>

        <h2>Add the project library</h2>
        <p>Open Project Library and add the surrounding paper trail: payment claims, payment schedules served, RFIs raised, variation notices, EOT notices, programme updates, key correspondence, latent condition notices. The more context, the better the drafting agents perform. Tag each item (RFI, Variation, Notice, Programme, Schedule, Correspondence, Other) so the lists stay scannable.</p>

        <h2>Choose the right tool for the job you have right now</h2>
        <p>If you are about to serve a payment claim, open the Payment Claims drafting agent inside your project. If you have just received one, open the Payment Schedule drafting agent. If a payment schedule has come back at zero or short, open the Adjudication Application complex agent and start with Stage 1 Intake. If you need a variation notice or an EOT notice, the matching drafting agent has a starter template ready to go. The Adjudicator Statistics page is a good cross-check before you nominate.</p>

        <h2>Where your data lives</h2>
        <p>Project content is stored in your browser using local storage. Nothing is uploaded to a server unless you explicitly run an action that needs the AI engine, in which case the relevant text is sent to Sopal's processing endpoint and the response is stored back into your local project. There is no shared workspace yet, and exporting your data (Project menu, Export) gives you a JSON snapshot you can keep elsewhere for backup.</p>
      `,
    },

    {
      slug: "adjudication-application",
      title: "Adjudication Application: end-to-end guide",
      category: "Workflows",
      summary: "How the five-stage Adjudication Application complex agent works, what each stage produces, and how to get the most useful output.",
      body: `
        <p class="lead">The Adjudication Application complex agent is the most involved workflow in Sopal. It is designed for a claimant preparing a section 79 application under the BIF Act, where the payment schedule has come back at less than the claimed amount, or no payment schedule has been provided, or a schedule was provided but the scheduled amount has not been paid by the due date.</p>

        <h2>Before you start</h2>
        <p>Have these documents ready in plain text or a clean PDF or DOCX: the payment claim you served, the payment schedule the respondent gave you (if any), the contract or its key clauses, and any correspondence that turned the dispute on a particular factual point (variation directions, RFI exchanges, latent condition notices, programme updates, dispute notices). You will get materially better drafting output if these documents are uploaded into the project's Contract and Project Library before you start the workflow, because the engine can then quote real clauses and real correspondence by name.</p>

        <h2>Stage 1: Document intake</h2>
        <p>This stage extracts the structured spine of the matter from the payment claim and payment schedule. Pick the right section 79 scenario first, because that drives every downstream calculation, framing and deadline.</p>
        <ul>
          <li><strong>No payment schedule received and no payment made:</strong> section 79(2)(a). The respondent did not give a payment schedule within the section 76 window (15 business days after the payment claim was given, or the period the contract specifies if shorter). The application is due 30 business days after the later of (i) the day the amount became payable under the contract, or (ii) the last day a schedule could have been given.</li>
          <li><strong>Schedule received and scheduled amount is less than claimed:</strong> section 79(2)(b). This is the most common scenario. The application is due 30 business days after the day you received the payment schedule.</li>
          <li><strong>Schedule received, scheduled amount equals the claim, but it has not been paid:</strong> section 79(2)(c). The application is due 20 business days after the day on which payment is due under the contract.</li>
        </ul>
        <p>Drop the payment claim into the left slot and the payment schedule into the right slot, or paste the text into the textareas if PDF extraction is messy. Optionally set the lodgement deadline; the Calculate from dates link opens the Due Date Calculator preset to the Adjudication Application scenario so you can compute it from the relevant key dates.</p>
        <p>When you click Parse documents, Sopal extracts the parties, the contract reference, the reference date, the claimed and scheduled totals, every line item with its claimed and scheduled amounts and the respondent's reasons for any difference, and the universe of reasons the respondent has put on the table (the section 82(4) ceiling for what the respondent can later argue at adjudication). Any extraction warnings appear inline next to the Parse button.</p>

        <h2>Stage 2: Dispute table</h2>
        <p>Stage 2 is where you sanity-check the extracted dispute table. The respondent's reasons that came out of the schedule are pre-filled per row. You can edit the item label, description, claimed and scheduled amounts, status, issue type, and the verbatim reasons.</p>
        <p>Pay particular attention to the issue type column, because it drives the per-item RFI templates the engine uses in Stage 3. The available types are variation, EOT, delay costs, defects, set-off, retention, prevention, scope, valuation, and other. The status column also matters: disputed (red wash) is the default for an item the respondent rejected; jurisdictional (purple) is for an item that turns on a jurisdictional argument rather than the merits; admitted (green) and partial (amber) are useful when an item is partly conceded.</p>
        <p>If the parser missed a line item, click Add row to append a blank one. If two PC line items are really one dispute, edit one row to combine them and delete the other. If the parser got the parties or contract reference wrong, click Matter details in the card head and fix them; those fields drive both the engine prompt and the master document cover page.</p>
        <p>When the table accurately reflects the dispute as you see it, click Lock dispute table to advance.</p>

        <h2>Stages 3 and 4: RFI and Draft</h2>
        <p>Stages 3 and 4 share a workspace. The left side is your items navigation: a Jurisdictional thread, a Background and General thread, and one thread per dispute item from Stage 2. The right side is the active thread.</p>
        <p>Each thread runs as a small AI conversation, scoped strictly to the topic of that thread. When you open a freshly-created thread, Sopal automatically generates the first targeted RFI for it. You answer in the table row, click Submit (or press Cmd or Ctrl plus Enter), and Sopal either generates the next RFI or, if it has enough information, marks the thread ready to draft.</p>
        <p>Click Draft this thread now whenever you want Sopal to take what it has and write the per-thread submissions. The Draft all action at the top runs the draft pass for every thread that has answered RFIs but is not yet drafted. The items navigation shows a status dot per thread (idle, in-progress, ready-to-draft, drafted) so you can see at a glance where everything stands.</p>
        <p>Definitions you and Sopal introduce in any one thread (for example, defining "the Contract" or "the Variation Notice") are shared across every other thread, so the language stays consistent across the whole application.</p>

        <h2>Stage 5: Review and lodge</h2>
        <p>Stage 5 has two parts: a lodgement checklist on the left and a master document preview on the right.</p>
        <p>The lodgement checklist tells you what is in place and what is still missing: payment claim ingested, payment schedule ingested, dispute table populated, jurisdictional submissions drafted, introduction drafted, executive summary drafted, overarching arguments drafted, threads drafted, evidence index populated, deadline set.</p>
        <p>Click View master document to open the master document in a fullscreen modal. The master is fluid: only sections you have populated will render. The seven possible sections, in order, are the cover page, the introduction, the executive summary, jurisdictional submissions, overarching arguments, per-item submissions with a quantum summary table, and the conclusion. The cover page is the only section that always renders.</p>
        <p>The master modal's action row gives you Cover page (opens the cover-page editor with grouped fields for application context, claimant details and respondent details), Introduction, Generate summary (which runs an engine call to produce a 4-6 paragraph executive summary from the per-thread headlines), Edit summary, and Overarching. The export buttons produce a Word-friendly .doc of the master, the combined statutory declaration, the index of supporting evidence, and a clean Markdown copy.</p>

        <h2>Tips that materially improve the output</h2>
        <ul>
          <li>Upload the contract before you parse. The engine quotes real clauses when it can see them, and falls back to bracketed placeholders when it cannot.</li>
          <li>Be precise with party names. "Acme Builders Pty Ltd" is materially different to "Acme Builders" in submissions; if you let the parser get this wrong, the engine repeats the error throughout. Use the Matter details editor to fix it.</li>
          <li>Answer the first RFI on a thread carefully. The engine bases the next questions on the first answer; if your answer is vague, every subsequent question is vague too.</li>
          <li>Use the Definitions panel proactively. The shared definitions are how the engine keeps language consistent across threads and sections.</li>
          <li>For each item that was rejected on multiple grounds, mention every ground in your RFI answers, even briefly. The engine will not invent reasons or address grounds it has not been told about.</li>
          <li>The exec summary is best generated last, after every per-item thread has been drafted. Re-generating is cheap; the new summary inherits any updates.</li>
        </ul>
      `,
    },

    {
      slug: "drafting-agents",
      title: "Drafting agents: when to use which",
      category: "Workflows",
      summary: "A practical guide to the six drafting agents and how the Word-style editor works.",
      body: `
        <p class="lead">The drafting agents in the Projects group give you a Word-style editor with a starter template for each common BIF Act and project document. The right-hand pane is an AI chat that rewrites the document on instruction. Use them for one-off documents that do not need the multi-stage workflow of the Adjudication Application complex agent.</p>

        <h2>The six agents</h2>
        <p><strong>Payment Claims</strong>: prepare a payment claim under section 75 of the BIF Act. The template covers parties, claimed amount, identification of the construction work or related goods and services, item-by-item breakdown, the statutory endorsement, and service detail.</p>
        <p><strong>Payment Schedules</strong>: respond to a payment claim under section 76. The template covers parties, the payment claim being responded to, scheduled amount, itemised reasons for any difference, and a reservation-of-rights paragraph.</p>
        <p><strong>EOTs</strong>: notice of extension of time. The template covers the qualifying cause of delay, when the contractor became aware, the likely effect on progress, supporting analysis, and the contractual basis for the claim.</p>
        <p><strong>Variations</strong>: a variation notice or claim. The template covers the direction or change relied on, the contractual basis, a description of the varied work, valuation, time impact, and supporting documents.</p>
        <p><strong>Delay Costs</strong>: a prolongation or disruption claim. The template covers entitlement, causation, period, and quantum.</p>
        <p><strong>General Correspondence</strong>: a free-form letter, email or notice template. Useful for show-cause letters, suspension notices, default notices, or anything that does not fit one of the structured agents.</p>

        <h2>The editor</h2>
        <p>The left pane is a contenteditable document. Type or paste content directly. The toolbar gives you bold, italic, underline, headings (H1, H2), paragraph formatting, bulleted lists and numbered lists. The keyboard shortcuts you would expect (Cmd or Ctrl plus B, I, U) work as well.</p>
        <p>Pasting from Word or Google Docs is supported and the inline style soup is stripped automatically; the structural HTML (headings, lists, tables, basic emphasis) is preserved. The document is auto-saved every time you stop typing, with the save indicator in the toolbar.</p>
        <p>Three buttons in the top right of the toolbar give you Copy HTML (copies the rich content to your clipboard), Download .doc (downloads a Word-compatible .doc file), and Reset (rolls the document back to the blank template; this asks for confirmation because it is destructive).</p>

        <h2>The chat pane</h2>
        <p>Type any instruction into the chat composer. Sopal rewrites the document according to your instruction and gives you a one-line summary of what it changed. Examples that work well:</p>
        <ul>
          <li>"Set the claimed amount to $487,250 and recalculate the breakdown so the items add up to that figure."</li>
          <li>"Add a section reserving the contractor's rights to delay costs and disruption costs in respect of the latent condition described in the variation notice."</li>
          <li>"Tighten the language on the statutory endorsement so it tracks section 75 of the BIF Act."</li>
          <li>"Change every reference to BCIPA to BIF Act."</li>
          <li>"Add a row to the breakdown table for the Variation V14 amount of $214,500 (excl. GST)."</li>
        </ul>
        <p>The chat keeps a running history per agent per project so you can see the lineage of edits. The Project context checkbox at the bottom controls whether your contract and library content is also sent with the instruction; turn it on when you want Sopal to quote real clauses, turn it off if you want a faster turn-around with no project grounding.</p>

        <h2>Where to draw the line</h2>
        <p>If you find yourself making the same kind of document many times, with multiple disputes and a complex narrative, switch to the Adjudication Application complex agent instead. It runs a structured RFI workflow per dispute and assembles the master document for you. The drafting agents are best for one document at a time.</p>
      `,
    },

    {
      slug: "decision-search",
      title: "Decision search and adjudicator statistics",
      category: "Research",
      summary: "How to search the public adjudication-decision corpus and how to size up an adjudicator before you nominate.",
      body: `
        <p class="lead">Sopal carries a searchable corpus of public Queensland adjudication decisions under both the current BIF Act 2017 and the older BCIPA 2004. It is the only place inside the app where you do not need a project to do useful research.</p>

        <h2>Decision search</h2>
        <p>The search box accepts a free-text query: an adjudicator name, a party name, a section reference, a contract clause, or any keyword from the decision text. Results are ranked by relevance by default; you can sort by date instead from the Sort dropdown.</p>
        <p>The Filters control lets you narrow by date range, decision year, and amount claimed. Saved searches let you keep a frequent query as a one-click chip. Pagination at the top and bottom of the results lets you walk through long result sets ten at a time.</p>
        <p>Click any result to open the decision detail panel on the right. The detail header shows the date, parties, adjudicator, the Act under which the decision was made, and the claimed and awarded amounts. The body shows the formatted decision text. The action row gives you Open page (a shareable link to the same decision), Copy citation (a short citation suitable for pasting into a brief or chat), Copy text (the full text), and Save to project (saves the decision into a chosen project's library so it is available as context for the agents in that project).</p>

        <h2>Adjudicator statistics</h2>
        <p>Adjudicator Statistics groups the corpus by adjudicator and shows you, for each one, the count of decisions, total claimed across those decisions, total awarded, and the average award rate (awarded as a percentage of claimed). This is the right place to look before you nominate an adjudicator: the average award rate gives you a coarse sense of how generous they tend to be on the merits, and the sample size tells you how confident you can be in that average.</p>
        <p>Click any card to open the adjudicator's detail page. That page shows their full decision list, the section references most commonly cited in their reasons, and an option to filter to just BIF Act decisions or just BCIPA decisions. From there you can save particular decisions to a project library or copy citations.</p>

        <h2>Research agent</h2>
        <p>The Research Agent is a free-form chat targeted at construction-law and SOPA questions. It uses the same decision corpus as background. Pick a jurisdiction at the top: Queensland is the default (with full BIF Act framing), and New South Wales, Victoria, Western Australia and South Australia are also available with limited support (the agent flags when it is operating outside its primary jurisdiction).</p>
      `,
    },

    {
      slug: "calculators",
      title: "BIF Act calculators",
      category: "Tools",
      summary: "Due dates and statutory interest calculations explained.",
      body: `
        <p class="lead">Two calculators take care of the most common money and time arithmetic. Both work without needing a project.</p>

        <h2>Due Date Calculator</h2>
        <p>The Due Date Calculator handles four common BIF Act deadlines. Each one is presented with the relevant statutory section reference so you can cross-check.</p>
        <ul>
          <li><strong>Payment Schedule</strong> (section 76): when the schedule must be given in response to a payment claim. Default 15 business days after the payment claim was given, or the period the contract specifies if shorter.</li>
          <li><strong>Adjudication Application</strong> (section 79): when the application must be lodged. Three sub-scenarios reflect the three section 79(2) limbs.</li>
          <li><strong>Adjudication Response</strong> (section 83): when the respondent's response must be given. The window depends on whether the respondent is in the business of construction work and whether a payment schedule was previously given.</li>
          <li><strong>Adjudicator's Decision</strong> (section 85): when the adjudicator must give their decision. Default 10 business days from the date the adjudicator notified acceptance of the application, extendable by agreement.</li>
        </ul>
        <p>Pick the location (Brisbane, Cairns, Toowoomba, Townsville, Mackay or other QLD region) so the right show-day public holidays apply. Public holidays and weekends are skipped automatically. The calculator reports the final due date, the count of business days added, and a list of any holidays that were skipped along the way.</p>

        <h2>Interest Calculator</h2>
        <p>The Interest Calculator works out the interest payable on an overdue progress payment under section 73 of the BIF Act. Two rate types are supported.</p>
        <ul>
          <li><strong>QBCC section 67P rate:</strong> the prescribed rate from section 67P of the Queensland Building and Construction Commission Act 1991, which is the default unless the contract specifies a higher rate. The rate updates periodically; the calculator carries the current published rate.</li>
          <li><strong>Contractual rate:</strong> the rate specified in the contract, where it is higher than the section 67P rate. Enter the rate as an annual percentage.</li>
        </ul>
        <p>Enter the unpaid amount, the date payment was due, and the date you want interest calculated to (defaults to today). The calculator returns the interest payable, the day count, and the daily rate used.</p>

        <h2>Standalone reviewers</h2>
        <p>Two reviewers also live in Tools: the Payment Claim Reviewer and the Payment Schedule Reviewer. Each takes a pasted document or a PDF or DOCX upload and runs a structured BIF Act review. You can pick the review mode at the top: "I'm about to serve" runs a pre-service check (statutory compliance, identification of work, dates, evidence); "I'm received" runs an audit (jurisdictional objections, prior-claim comparison, withholding adequacy). Held in memory only; nothing is stored on the server.</p>
      `,
    },

    {
      slug: "faq",
      title: "Frequently asked questions",
      category: "Reference",
      summary: "Direct answers to the questions most users send us.",
      body: `
        <h2>About Sopal</h2>

        <h3>Is Sopal a law firm?</h3>
        <p>No. Sopal is software. It does not provide legal advice and using it does not create a solicitor-client relationship. Where the consequences of a decision are material, run the output past a construction lawyer who acts for you.</p>

        <h3>Who is Sopal for?</h3>
        <p>Project managers, contract administrators, claims managers, in-house counsel and small to mid-sized construction businesses who want to draft and review BIF Act documents quickly without paying for every cycle to be done by an external firm.</p>

        <h3>What jurisdictions does Sopal cover?</h3>
        <p>Primary jurisdiction is Queensland (BIF Act 2017). The Research Agent will run queries across New South Wales, Victoria, Western Australia and South Australia with a "limited support" banner; the dedicated workflows (Adjudication Application, drafting agents) are tuned to the BIF Act. Multi-jurisdiction support for the dedicated workflows is on the roadmap.</p>

        <h2>Account and billing</h2>

        <h3>How do I create an account?</h3>
        <p>Click Login at the top right of the marketing site. From the login page, click Create account and enter your details. Your subscription gives you access to every page on app.sopal.com.au.</p>

        <h3>What does Sopal cost?</h3>
        <p>Pricing is on the public pricing page. Plans range from a per-seat monthly subscription for individuals through to firm-wide plans with shared projects.</p>

        <h3>How do I cancel?</h3>
        <p>From your account page on the marketing site, open the Subscription section and click Manage subscription. That opens the Stripe customer portal where you can cancel, change plan, update payment method, or download invoices.</p>

        <h3>Is there a free trial?</h3>
        <p>Yes. New accounts can run the Standalone tools (Decision Search, Adjudicator Statistics, the calculators) without paying. The drafting agents and the Adjudication Application complex agent require a paid plan because they incur AI processing costs per use.</p>

        <h2>Data and privacy</h2>

        <h3>Where is my project data stored?</h3>
        <p>Your project content (contracts, library, drafts, definitions, RFIs, dispute tables, master documents) is stored in your browser using local storage. It does not leave your machine unless you explicitly run an action that needs the AI engine.</p>

        <h3>What gets sent to the AI engine when I use it?</h3>
        <p>When you ask the engine to draft, parse or rewrite, the relevant text is sent to Sopal's processing endpoint. That includes the documents you have uploaded for context (capped per request to keep latency sane), the RFI history for the relevant thread, and your most recent instruction. Sopal does not retain that content beyond the duration of the request.</p>

        <h3>Can I export everything?</h3>
        <p>Yes. From any project, click Export in the project header. You get a JSON snapshot containing the contracts, library, drafts, definitions and chat history for that project. Keep it as a backup or move the project between machines.</p>

        <h3>What happens if I clear my browser data?</h3>
        <p>Without an export, your project content is lost. Server-side persistent storage is on the roadmap; until then, exporting before any browser-data wipe is the safest practice.</p>

        <h2>Workflow questions</h2>

        <h3>The parser missed a line item in my payment claim. What do I do?</h3>
        <p>Open Stage 2 (Dispute Table) and click Add row. Enter the missed item's label, description, claimed amount, scheduled amount, status and issue type. Save. Continue to Stage 3 normally.</p>

        <h3>The parser got my client's name wrong. What do I do?</h3>
        <p>On Stage 2, click Matter details in the card head and edit the Claimant or Respondent name. The corrected names flow into the engine prompt and the master document cover page from the next save. You do not need to re-parse.</p>

        <h3>The engine's draft looks generic. How do I make it better?</h3>
        <p>Three things make the biggest difference. First, upload the contract clauses (or paste them) into the project's Contract page so the engine can quote them by clause number. Second, answer the early RFIs precisely; the engine builds on what you give it. Third, populate the Definitions panel proactively so the engine uses your defined terms consistently.</p>

        <h3>Can Sopal lodge the application for me?</h3>
        <p>No. Sopal produces the documents; you (or your nominated authorised nominating authority) lodge them. The export buttons on Stage 5 give you the master document, the combined statutory declaration and the index of supporting evidence as Word files you can attach to your lodgement email.</p>

        <h3>Why does the engine sometimes leave [bracketed placeholders] in the draft?</h3>
        <p>The engine never invents facts. If a fact is not in the documents you have uploaded or in your RFI answers, it leaves a placeholder for you to fill in. If you would prefer the engine to ask you for the fact instead of leaving a placeholder, click Ask another RFI on the relevant thread.</p>

        <h2>Technical questions</h2>

        <h3>Which browsers are supported?</h3>
        <p>Sopal is tested on the current versions of Chrome, Safari, Edge and Firefox. The Word-style editor uses contenteditable, which is well-supported across all four. Mobile Safari and mobile Chrome work for read and review; for heavy drafting we recommend a desktop browser.</p>

        <h3>Is there a mobile app?</h3>
        <p>Not yet. The web app is responsive enough to use on a phone for review and quick edits, but the dispute table, master document and Word-style editor are best on a screen wider than 1024 pixels.</p>

        <h3>Can I run Sopal offline?</h3>
        <p>The pages that have already loaded keep working offline (because everything is in local storage), but the AI engine, the Decision Search and the Adjudicator Statistics page need a network connection to function.</p>

        <h3>I think I found a bug. Where do I report it?</h3>
        <p>Send us a note via the Feedback link in the footer. Include a short description, the page URL, and a screenshot if possible. We triage every report.</p>
      `,
    },

    {
      slug: "glossary",
      title: "Glossary",
      category: "Reference",
      summary: "Defined terms used in Sopal and in BIF Act work.",
      body: `
        <p class="lead">A short reference of terms you will encounter throughout Sopal. The definitions below are written for working understanding, not as legal definitions; the BIF Act and the contract govern actual scope.</p>

        <h2>Statutory terms</h2>
        <dl>
          <dt>BIF Act</dt>
          <dd>The Building Industry Fairness (Security of Payment) Act 2017 (Qld). The current Queensland security-of-payment legislation, in force from 17 December 2018.</dd>

          <dt>BCIPA</dt>
          <dd>The Building and Construction Industry Payments Act 2004 (Qld). Repealed and replaced by the BIF Act. Decisions decided under BCIPA are still useful research material because the substantive principles often carry across.</dd>

          <dt>Section 75</dt>
          <dd>The provision under which a payment claim is made. A valid section 75 claim must identify the construction work or related goods and services to which it relates, state the claimed amount, and carry the statutory endorsement.</dd>

          <dt>Section 76</dt>
          <dd>The provision governing payment schedules. A respondent must give a payment schedule within 15 business days after the payment claim is given (or the period the contract specifies if shorter). The schedule must identify the payment claim, state the scheduled amount, and (where the scheduled amount is less than the claimed amount) give reasons for the difference.</dd>

          <dt>Section 79</dt>
          <dd>The provision under which an adjudication application is made. Section 79(2) sets out the three timing scenarios that drive the lodgement deadline.</dd>

          <dt>Section 82(4)</dt>
          <dd>The provision that limits the respondent in adjudication to the reasons it gave in its payment schedule (or to reasons that could not have been included). New reasons cannot be raised at adjudication; this is the "section 82(4) ceiling".</dd>

          <dt>Section 83</dt>
          <dd>The provision governing the adjudication response. Timing depends on whether the respondent is in the business of construction work and whether a payment schedule was previously given.</dd>

          <dt>Section 85</dt>
          <dd>The provision governing the adjudicator's decision. The adjudicator must give the decision within 10 business days of accepting the application (extendable by agreement).</dd>
        </dl>

        <h2>Procedural terms</h2>
        <dl>
          <dt>Reference date</dt>
          <dd>The date on or after which a payment claim may be made under the contract. Often the last day of a calendar month, or the date a particular milestone is reached. Each reference date supports one payment claim.</dd>

          <dt>Authorised Nominating Authority (ANA)</dt>
          <dd>An organisation authorised under the BIF Act to receive adjudication applications and refer them to an adjudicator. Adjudicate Today is the most active in Queensland.</dd>

          <dt>Statutory declaration</dt>
          <dd>A formal declaration signed before an authorised witness, used to verify the facts in an adjudication application. Sopal generates a combined statutory declaration covering the matter, parties, application materials and supporting evidence.</dd>

          <dt>Index of supporting evidence</dt>
          <dd>A schedule of every document attached to the adjudication application, cross-referenced from the per-item submissions. Sopal generates this automatically from the evidence references its engine produces while drafting.</dd>
        </dl>

        <h2>Sopal-specific terms</h2>
        <dl>
          <dt>Drafting agent</dt>
          <dd>A single-document workspace with a Word-style editor on the left and an AI chat on the right. There is one per common BIF Act document type.</dd>

          <dt>Complex agent</dt>
          <dd>A multi-stage workflow that produces a structured output. The Adjudication Application is the only complex agent in the current build.</dd>

          <dt>Item, dispute, thread</dt>
          <dd>In the Adjudication Application complex agent, an item (or dispute) is one row in the Stage 2 dispute table. A thread is the per-item RFI conversation that runs in Stage 3 and 4.</dd>

          <dt>RFI</dt>
          <dd>Request for Information. In the Adjudication Application complex agent, the engine raises one RFI at a time per thread to gather what it needs to draft that thread's submissions.</dd>

          <dt>Master document</dt>
          <dd>The assembled adjudication application document. Renders fluidly: only sections you have populated will appear.</dd>

          <dt>Cover meta</dt>
          <dd>The optional cover-page details (ABN, contact information, contract date, project address, ANA reference) that render in the bordered tables on the master document cover page.</dd>
        </dl>
      `,
    },

    {
      slug: "privacy",
      title: "Privacy and data",
      category: "Reference",
      summary: "How Sopal handles your data, what is sent where, and what control you have.",
      body: `
        <p class="lead">Sopal is built on a "your project, your machine" model. Project content lives in the browser; the AI engine only sees what is needed to handle the immediate request; nothing is shared between users.</p>

        <h2>Local-first storage</h2>
        <p>Project content (contracts, library, drafts, definitions, RFI history, dispute tables, master documents, chat history) is stored in your browser using local storage. The same storage holds the Recently viewed decisions list and your saved searches.</p>
        <p>Local storage is per browser per device. It is not synced across devices or browsers automatically, and it is cleared when you clear browsing data for the site. Use the Export action on a project to take a JSON snapshot you can keep elsewhere.</p>

        <h2>What goes to the server</h2>
        <p>When you click an action that uses the AI engine (Parse documents, Ask first RFI, Draft this thread now, Generate executive summary, the chat composer in any drafting agent or assistant), Sopal sends a request to its processing endpoint. The request contains:</p>
        <ul>
          <li>The user instruction (your prompt or the system action you triggered).</li>
          <li>The relevant context (the document being edited, the thread's RFI history, the matter context, the project's contracts and library to a reasonable cap).</li>
          <li>The shared definitions for the active project.</li>
        </ul>
        <p>The engine's response is stored back into the project's local storage. Sopal does not retain the request body or the response beyond the duration of that request, and does not use your project content to train any model.</p>

        <h2>Account data</h2>
        <p>Account data (email, name, firm details, billing address, subscription state) is stored on Sopal's servers and is required to authenticate you and to bill the subscription. You can update or remove account data from the account page on the marketing site.</p>

        <h2>Cookies</h2>
        <p>Sopal uses essential cookies only: a session cookie to keep you logged in, and a preference cookie for theme (light or dark). No advertising or tracking cookies.</p>

        <h2>Removing your data</h2>
        <p>To remove the project content from your browser: open Settings (the cog icon), choose Clear all local data, confirm. To close your account: from the account page on the marketing site, choose Close account; this removes account data after the subscription is cancelled and any open invoices are settled.</p>

        <h2>Data we will share, and when</h2>
        <p>We share data with third parties only in narrow, specified circumstances: with our payment processor (Stripe) to take subscription payments; with our infrastructure providers (cloud hosting, email delivery) under standard data-processing agreements; and where compelled by a court or by law. We will not sell your data and we will not share it with construction industry firms or other third parties.</p>
      `,
    },

    {
      slug: "account-and-cloud-sync",
      title: "Account and cloud sync",
      category: "Reference",
      summary: "How accounts work, how cloud sync of your projects works, and what happens when you sign out.",
      body: `
        <p class="lead">Sopal uses a single account across the marketing site and the app. Sign in once and you have everything: pricing-page features, project workspace, cloud sync of your projects, and the account page on sopal.com.au.</p>

        <h2>Creating an account</h2>
        <p>Go to <a href="/register">/register</a> on the marketing site or use the Sign in button in Sopal v2's sidebar foot, which takes you to <a href="/login">/login</a> with a Create account link. The fields are email, password, name, firm name (optional), ABN (optional), billing details, and a phone number.</p>

        <h2>Signing in</h2>
        <p>The login flow stores a JWT in your browser's local storage under the key "purchase_token". Sopal v2's sidebar foot picks this up automatically. If the token is missing, malformed or expired, the foot row shows a guest banner with a Sign in button.</p>

        <h2>Cloud sync</h2>
        <p>When you are signed in, every project edit auto-syncs to your account. The sync is a debounced PUT to /api/sopal-v2/projects/{id} that fires roughly 1.5 seconds after your last save. Failures are queued and retried on the next save.</p>
        <p>On boot, when the SPA detects you are signed in, it fetches the lightweight project index from your account. For any project the cloud has but this browser does not, the SPA pulls the full blob and merges it in. This is how a fresh browser, a different machine, or a freshly cleared local-storage rehydrates everything.</p>
        <p>Conflict resolution is last-write-wins. If you edit the same project from two browsers in parallel, the most recent save wins. Sopal does not currently merge concurrent edits at the field level.</p>

        <h2>Manual sync actions</h2>
        <p>Settings → Cloud sync exposes two manual actions:</p>
        <ul>
          <li><strong>Push all local projects to cloud now</strong>: posts every project in this browser to your account. Useful after enabling sync on a machine that already had projects, or after a long offline session.</li>
          <li><strong>Pull missing projects from cloud</strong>: re-runs the boot pull. Useful if you cleared local storage and want everything back without restarting the app.</li>
        </ul>

        <h2>Signing out</h2>
        <p>Sidebar foot → Sign out clears the JWT from local storage and redirects to /login?redirect=/sopal-v2. Your project content stays in this browser; it is also still in your account, untouched. Signing back in restores access to the cloud copies.</p>

        <h2>Closing your account</h2>
        <p>Open <a href="/account.html" target="_blank" rel="noopener">/account.html</a> on the marketing site and choose Close account. This cancels any active subscription and removes account data after the closing balance is settled. Your cloud-stored projects are deleted as part of this process.</p>
      `,
    },

    {
      slug: "legal-disclaimer",
      title: "Legal disclaimer",
      category: "Reference",
      summary: "What Sopal is and is not, and the limits of the AI output.",
      body: `
        <p class="lead">Sopal is software. The output is a working draft, not legal advice.</p>

        <h2>Not legal advice</h2>
        <p>Nothing in Sopal constitutes legal advice. Using Sopal does not create a solicitor-client relationship between you and any law firm. Where the matter is high value, time-critical, or turns on a difficult question of law, run the output past a qualified construction lawyer who acts for you. Sopal is best understood as a faster way to produce a first draft and to do background research, not a replacement for legal judgement.</p>

        <h2>AI output limits</h2>
        <p>The drafting agents and the Adjudication Application complex agent run on large language models. The output is generally accurate but is sometimes wrong: it can mis-cite a section number, mis-state a date, or mischaracterise a fact. Treat every draft as a draft. Read it carefully. Cross-check section references against the BIF Act. Cross-check dates against your records. Cross-check quoted contract clauses against the contract text. Sopal flags placeholders explicitly so you can see where it has not invented facts.</p>

        <h2>Decision corpus accuracy</h2>
        <p>The decision corpus is built from publicly-available adjudication decisions. Reasonable care is taken in extracting party names, adjudicator names, dates, and amounts, but extraction errors occur from time to time. Always verify against the official copy of the decision before relying on it in submissions or correspondence.</p>

        <h2>No guarantee of outcome</h2>
        <p>Adjudication outcomes turn on a wide range of factors: the strength of the underlying contract, the documentary record, the conduct of the parties, the timing, and the adjudicator's view. Sopal does not, and cannot, guarantee a particular outcome. The Adjudicator Statistics page shows historical award rates, which is a useful data point but not a forecast.</p>

        <h2>Service availability</h2>
        <p>Sopal aims for high availability but does not guarantee uninterrupted service. The processing endpoint depends on third-party language model providers; if those providers are unavailable, the dependent features will be unavailable too. Standalone tools that do not need the engine continue to work in those windows.</p>

        <h2>Updates to Sopal</h2>
        <p>Sopal is updated regularly. Some updates change the way the engine prompts itself, or the way drafts are structured, in light of feedback or new case law. Old drafts are not retroactively rewritten. Re-run a draft action if you want the latest engine to revisit an old draft.</p>
      `,
    },
  ];

  function HelpIndexPage() {
    const groups = {};
    HELP_ARTICLES.forEach((a) => {
      groups[a.category] = groups[a.category] || [];
      groups[a.category].push(a);
    });
    const groupOrder = ["Start here", "Workflows", "Research", "Tools", "Reference"];
    const groupHtml = groupOrder
      .filter((g) => groups[g])
      .map((g) => `
        <section class="help-group">
          <h2 class="help-group-title">${escapeHtml(g)}</h2>
          <div class="help-card-grid">
            ${groups[g].map((a) => `
              <a class="help-card" href="/sopal-v2/help/${attr(a.slug)}" data-nav>
                <h3>${escapeHtml(a.title)}</h3>
                <p>${escapeHtml(a.summary)}</p>
                <span class="help-card-cta">Read ${ICON.arrowUpRight}</span>
              </a>
            `).join("")}
          </div>
        </section>
      `).join("");
    return PageBody(`
      <div class="page-shell help-shell">
        <h1 class="page-title">Help and support</h1>
        <p class="page-sub">Guides to every part of Sopal, plus FAQs and reference material. Search the page in the browser (Cmd or Ctrl plus F) for a quick lookup.</p>
        ${groupHtml}
        <section class="help-contact">
          <h2 class="help-group-title">Still stuck?</h2>
          <p>Send us a note via the <a href="/feedback" target="_blank" rel="noopener">Feedback page</a>. Include the page URL, a short description of what you were doing, and a screenshot if possible. We triage every report.</p>
        </section>
      </div>
    `);
  }

  function HelpArticlePage(slug) {
    const article = HELP_ARTICLES.find((a) => a.slug === slug);
    if (!article) return notFoundPage();
    const others = HELP_ARTICLES.filter((a) => a.slug !== slug && a.category === article.category).slice(0, 3);
    return PageBody(`
      <div class="page-shell help-article-shell">
        <p class="muted help-article-cat">${escapeHtml(article.category)}</p>
        <h1 class="page-title">${escapeHtml(article.title)}</h1>
        <article class="help-article-body">${article.body}</article>
        ${others.length ? `
          <section class="help-related">
            <h2 class="help-group-title">Related</h2>
            <div class="help-card-grid">
              ${others.map((a) => `
                <a class="help-card" href="/sopal-v2/help/${attr(a.slug)}" data-nav>
                  <h3>${escapeHtml(a.title)}</h3>
                  <p>${escapeHtml(a.summary)}</p>
                </a>
              `).join("")}
            </div>
          </section>
        ` : ""}
        <p class="help-back"><a href="/sopal-v2/help" data-nav>${ICON.chevLeft || "<"} Back to all help</a></p>
      </div>
    `);
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
    if (parts[0] === "settings") {
      return { crumbs: [{ label: "Settings" }], body: SettingsPage() };
    }
    if (parts[0] === "help") {
      if (!parts[1]) return { crumbs: [{ label: "Help and support" }], body: HelpIndexPage() };
      const article = HELP_ARTICLES.find((a) => a.slug === parts[1]);
      if (article) return { crumbs: [{ label: "Help and support", href: "/sopal-v2/help" }, { label: article.title }], body: HelpArticlePage(parts[1]) };
      return {
        crumbs: [{ label: "Help and support", href: "/sopal-v2/help" }],
        body: notFoundPage({
          title: "We could not find that help article.",
          body: "It may have been renamed or removed. Browse the index for the current article list.",
          cta: `<a class="ghost-button compact" href="/sopal-v2/help" data-nav>Open the help index</a>`,
        }),
      };
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
    document.querySelectorAll("[data-sopal-signout]").forEach((el) => el.addEventListener("click", () => sopalAuth.signOut()));
    document.querySelectorAll("[data-open-palette]").forEach((el) => el.addEventListener("click", () => openCommandPalette()));
    document.querySelectorAll("[data-toggle-theme]").forEach((el) => el.addEventListener("click", () => setTheme(theme === "dark" ? "light" : "dark")));
    document.querySelectorAll("[data-open-whatsnew]").forEach((el) => el.addEventListener("click", () => openWhatsNew()));
    document.querySelectorAll("[data-toggle-pin-thread]").forEach((el) => el.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      togglePinThread(el.dataset.projectId, el.dataset.togglePinThread);
    }));
    document.querySelectorAll("[data-new-project]").forEach((el) => el.addEventListener("click", () => openProjectModal(null)));
    document.querySelectorAll("[data-toggle-agent]").forEach((btn) => btn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const key = btn.dataset.toggleAgent;
      if (sidebarAgentOpen.has(key)) sidebarAgentOpen.delete(key);
      else sidebarAgentOpen.add(key);
      render();
    }));
    document.querySelectorAll("[data-new-agent-draft]").forEach((btn) => btn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const project = currentProject();
      if (!project) return;
      const agentKey = btn.dataset.newAgentDraft;
      const inst = createDraftInstance(project, agentKey, "");
      sidebarAgentOpen.add(`${project.id}:${agentKey}`);
      navigate(`/sopal-v2/projects/${project.id}/agents/${agentKey}?mode=draft&iid=${inst.id}`);
    }));
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

    // Help articles. Surfacing these in the palette is what makes the help
    // system actually findable; without them the user has to remember the
    // sidebar foot link.
    items.push({ section: "Help", label: "Help and support index", hint: "All articles", run: () => navigate("/sopal-v2/help") });
    HELP_ARTICLES.forEach((a) => {
      items.push({ section: "Help", label: a.title, hint: a.summary, run: () => navigate(`/sopal-v2/help/${a.slug}`) });
    });

    // Settings.
    items.push({ section: "Settings", label: "Settings", hint: "Account, cloud sync, data, appearance", run: () => navigate("/sopal-v2/settings") });

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
                <input class="palette-input" type="text" data-project-search-input placeholder="Search ${escapeHtml(project.name)}: contracts, library, chats…" value="${attr(projectSearchState.query)}" autocomplete="off" spellcheck="false">
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
    { keys: ["⌘/Ctrl", "Shift", "D"], label: "Toggle light / dark theme", group: "Navigation" },
    { keys: ["?"], label: "Show this cheat sheet", group: "Navigation" },
    { keys: ["Esc"], label: "Close any modal / drawer / palette", group: "Navigation" },
    { keys: ["⌘/Ctrl", "Enter"], label: "Send chat message from any composer", group: "Chat" },
    { keys: ["⌘/Ctrl", "Enter"], label: "Submit answer in an RFI table row", group: "Chat" },
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
    { date: "May 2026", title: "Help and Support, Settings, and cloud sync", body: "Nine long-form help articles routed under /sopal-v2/help and indexed in the Cmd+K palette. A new Settings page (Account, Cloud sync, Data and storage, Appearance) at /sopal-v2/settings. Sign in with your Sopal account and your projects auto-sync to the cloud (debounced PUTs to /api/sopal-v2/projects), with manual push and pull controls in Settings." },
    { date: "May 2026", title: "AA workflow polish: RFI table, Matter details, ABN and contract date on cover", body: "Stage 3/4 RFI panel rebuilt as a table with Edit on answered rows and Cmd+Enter to submit. Stage 2 dispute table now colour-codes status, prefixes Claimed/Scheduled with $, and exposes a Matter details editor for Claimant, Respondent, Contract reference and Reference date. Cover page editor adds ABN, Contract executed date and Project / site address fields, grouped under Application context, Claimant and Respondent." },
    { date: "May 2026", title: "Project Quick start, contextual greeting, inline parse errors", body: "New empty projects show a four-step Quick start panel that disappears when content arrives. The home page greets signed-in users by first name. Stage 1 Intake replaces alert() validation and parse failures with an inline error pane so the upload context is preserved." },
    { date: "May 2026", title: "Complex Agents: Adjudication Application drafter (NEW)", body: "Multi-stage guided drafter: paste the PC + PS, lock the dispute table, work each item via per-issue-type RFIs, watch the master document assemble live. Three s 79 BIF Act scenarios supported (no schedule, less than claimed, scheduled-but-unpaid). 'Draft all' fires parallel passes for every answered thread. Exports the master, the combined statutory declaration, and the index of supporting evidence as separate .doc files." },
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

  /* ---------- Auth (purchase user via shared /purchase-login JWT) ---------- */

  // The marketing site stores its JWT in localStorage under "purchase_token"
  // and authenticates against /purchase-me with a Bearer header. We reuse that
  // same token here so the user only has to sign in once across both sites.
  // Hard-gating (redirect to /login when no token) is opt-in via the flag
  // below; until it is flipped on the SPA still loads for guests, but the
  // sidebar surfaces a Sign in prompt and keeps the user's identity visible
  // when they are signed in.
  const AUTH_HARD_GATE = false;
  const AUTH_TOKEN_KEY = "purchase_token";
  const sopalAuth = {
    user: null,
    state: "unknown", // "unknown" | "guest" | "authed"
    token() {
      try { return localStorage.getItem(AUTH_TOKEN_KEY) || ""; } catch (_) { return ""; }
    },
    headers() {
      const t = this.token();
      return t ? { "Authorization": "Bearer " + t } : {};
    },
    async refresh() {
      const t = this.token();
      if (!t) {
        this.user = null;
        this.state = "guest";
        return;
      }
      try {
        const r = await fetch("/purchase-me", { headers: { "Authorization": "Bearer " + t } });
        if (r.ok) {
          this.user = await r.json();
          this.state = "authed";
        } else if (r.status === 401) {
          try { localStorage.removeItem(AUTH_TOKEN_KEY); } catch (_) {}
          this.user = null;
          this.state = "guest";
        } else {
          // Server error: treat as transient, keep token, mark unknown so we
          // do not punt the user out of a working session.
          this.state = "unknown";
        }
      } catch (_) {
        this.state = "unknown";
      }
    },
    requireOrRedirect() {
      if (!AUTH_HARD_GATE) return;
      if (this.state !== "authed") {
        const here = window.location.pathname + window.location.search + window.location.hash;
        window.location.replace("/login?redirect=" + encodeURIComponent(here));
      }
    },
    signOut() {
      try { localStorage.removeItem(AUTH_TOKEN_KEY); } catch (_) {}
      this.user = null;
      this.state = "guest";
      window.location.replace("/login?redirect=" + encodeURIComponent("/sopal-v2"));
    },
  };
  // Expose so other modules and the inspector can read auth state.
  window.SopalAuth = sopalAuth;

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

  // Auth check runs after the first paint so the user does not stare at a
  // blank screen waiting for the network round-trip. When the call lands the
  // sidebar foot re-renders with the user's identity (or a Sign in prompt)
  // and the hard gate fires only if the AUTH_HARD_GATE flag is on. Cloud
  // sync's pull-missing pass runs once the auth state is known, so a fresh
  // browser on a signed-in account auto-rehydrates that user's projects.
  (async () => {
    await sopalAuth.refresh();
    sopalAuth.requireOrRedirect();
    render();
    if (sopalAuth.state === "authed") {
      cloudSync.pullMissing();
      // Pull firm settings after auth so the user's branding follows them
      // across browsers. Local-first wins until the network responds.
      firmCloudSync.pull();
    }
  })();
})();
