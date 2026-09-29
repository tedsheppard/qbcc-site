/* Sopal Workspace landing page: a demo of Boolean search over a made-up
   library. Everything here is invented: the firms, parties, people, dates and
   passages. Company names were checked against the ABN register on
   29 September 2026 and none of the distinctive words is in use. Runs entirely
   in the browser; nothing is sent anywhere. */
(function () {
  var DOCS = [
    { type: "Adjudication application", parties: "Tamsley Glass Pty Ltd v Orrindale Built Pty Ltd", state: "QLD", date: "August 2024", passages: [
      [9, "The payment claim was given on 9 July 2024. The payment schedule was given late, by email at 6.12 pm on 31 July 2024, the sixteenth business day after the payment claim was given, and so outside the time allowed by s 76 of the Act."],
      [14, "Variations 7 to 11 were directed orally by the respondent's site manager at the weekly site meetings. The minutes record each direction, and the respondent accepted the work without objection."],
      [17, "The claimant claims interest on the unpaid amount from the due date at the rate prescribed under the Act, as set out in Annexure C."] ] },
    { type: "Adjudication response", parties: "Brenlock Civil Pty Ltd v Ostrova Projects Pty Ltd", state: "QLD", date: "June 2025", passages: [
      [6, "The payment claim does not identify the construction work to which it relates. It refers only to 'works to date' and a lump sum, without any breakdown by trade, location or contract item, and so is not a payment claim under s 68 of the Act."],
      [11, "The respondent withheld $184,300 for defective work to the stormwater pits. The defects are shown in the independent engineer's report, and the cost to rectify was quoted by two contractors."],
      [15, "The respondent is entitled to liquidated damages for 23 days of delay at the rate in item 31 of the Contract Particulars, being $2,400 per day."] ] },
    { type: "Further submissions", parties: "Keddlestone Electrical Pty Ltd v Varnaby Constructions Pty Ltd", state: "NSW", date: "October 2023", passages: [
      [4, "The respondent's payment schedule, served late on 2 October, cannot be relied on. Under s 14 of the NSW Act the schedule had to be provided within 10 business days after the payment claim was served."],
      [7, "The reference date for the claim arose on 28 August 2023 under clause 37.1 of the subcontract, and the claim was served after that date."] ] },
    { type: "Adjudication application", parties: "Pemmerly Tiling Pty Ltd v Aldmere Residential Pty Ltd", state: "QLD", date: "February 2025", passages: [
      [5, "The contract provides for monthly claims on the 25th day of each month. The payment claim was made on 25 November 2024 and relates to work carried out to that date."],
      [12, "Variation 4 is for additional waterproofing to the level 2 balconies, which the architect instructed in writing on 3 October 2024 after the membrane specification changed."] ] },
    { type: "Adjudication response", parties: "Ellismere Plumbing Pty Ltd v Corradyne Group Pty Ltd", state: "VIC", date: "November 2024", passages: [
      [8, "The amounts claimed for delay costs are excluded amounts under the Victorian Act and cannot be taken into account by the adjudicator."],
      [13, "The latent conditions claim is not a claimable variation. The contract allocates the risk of ground conditions to the claimant, and no written direction was given."] ] },
    { type: "Adjudication application", parties: "Maudrey Steel Pty Ltd v Belvane Developments Pty Ltd", state: "QLD", date: "April 2025", passages: [
      [12, "The payment schedule was served out of time. It went to an address the contract did not nominate, and did not come to the claimant's attention until 21 March 2025."],
      [16, "The claimant gave notice of its intention to suspend work on 2 April 2025 after the respondent failed to pay the claimed amount by the due date."] ] },
    { type: "Adjudication response", parties: "Covenhurst Formwork Pty Ltd v Kilbarra Projects Pty Ltd", state: "NSW", date: "July 2024", passages: [
      [9, "Clause 34.2 of the subcontract is a time bar. It requires notice of a delay within 7 days of the claimant becoming aware of it. No notice was given for the delays now claimed."],
      [13, "The extension of time claims were each assessed and rejected by the superintendent's representative, with reasons, in the letters at Tab 14."] ] },
    { type: "Adjudication response", parties: "Kestrovan Roofing Pty Ltd v Tresswick Homes Pty Ltd", state: "QLD", date: "March 2025", passages: [
      [7, "Liquidated damages of $1,850 per day are deducted for the period from the date for practical completion to the date practical completion was reached, being 41 days."],
      [10, "Practical completion was not reached until the roof sheeting defects were rectified on 14 February 2025, as the certificate at Tab 6 records."] ] },
    { type: "Adjudication application", parties: "Quillaby Concrete Pty Ltd v Yarrowin Group Pty Ltd", state: "QLD", date: "September 2025", passages: [
      [8, "Variation 3 is valued at the daywork rates in Schedule 4 of the subcontract, as clause 36.4 requires where the parties have not agreed a price."],
      [11, "The payment schedule gives no reasons for withholding payment for variations 1 and 2 beyond the word 'disputed', and does not explain why."] ] },
    { type: "Adjudication response", parties: "Denthorne Electrical Pty Ltd v Harlowvale Projects Pty Ltd", state: "QLD", date: "January 2026", passages: [
      [5, "The payment claim was made after the contract was terminated on 12 October 2025. The claimant had no entitlement to a further progress payment after termination."],
      [9, "If the claim is treated as a final payment claim, it was made outside the time allowed by s 75 of the Act."] ] },
    { type: "Adjudication application", parties: "Tavistrel Scaffolding Pty Ltd v Halvenbrook Built Pty Ltd", state: "NSW", date: "May 2025", passages: [
      [6, "The respondent had accepted service of claims by email throughout the project, and service of this claim by email was effective."],
      [10, "Hire charges continued after practical completion because the respondent did not give the direction to dismantle the scaffold until 30 April 2025."] ] },
    { type: "Further submissions", parties: "Merrowgate Painting Pty Ltd v Ostrevan Homes Pty Ltd", state: "QLD", date: "December 2025", passages: [
      [3, "The respondent's submissions raise new reasons for withholding payment that were not included in the payment schedule. They cannot be considered under s 82 of the Act."],
      [6, "The alleged defective painting was rectified in the week after the inspection, as the photographs at Tab 9 show, and no amount should be withheld for it."] ] }
  ];

  var WORD = /[A-Za-z0-9$][A-Za-z0-9$,.'’-]*[A-Za-z0-9]|[A-Za-z0-9]/g;
  function words(text) {
    var out = [], m;
    WORD.lastIndex = 0;
    while ((m = WORD.exec(text))) out.push({ w: m[0].toLowerCase().replace(/[’']s$/, "").replace(/[.,]+$/, ""), i: m.index, n: m[0].length });
    return out;
  }
  DOCS.forEach(function (d) { d.passages = d.passages.map(function (p) { return { page: p[0], text: p[1], words: words(p[1]) }; }); });

  // Tokens: ( ) "phrase" AND OR NOT w/N and words (with * or ! for any ending).
  function tokenize(q) {
    var t = [], re = /\s*("([^"]*)"|\(|\)|w\/\d+|\/\d+|[^\s()"]+)/gi, m;
    while ((m = re.exec(q))) {
      var s = m[1];
      if (m[2] !== undefined) t.push({ k: "phrase", v: words(m[2]).map(function (x) { return x.w; }) });
      else if (s === "(" || s === ")") t.push({ k: s });
      else if (/^(and|&)$/i.test(s)) t.push({ k: "and" });
      else if (/^(or|\|)$/i.test(s)) t.push({ k: "or" });
      else if (/^(not|%)$/i.test(s)) t.push({ k: "not" });
      else if (/^w?\/\d+$/i.test(s)) t.push({ k: "near", n: +s.replace(/\D/g, "") });
      else { var ws = words(s.replace(/[*!]$/, "")); if (ws.length) t.push({ k: "word", v: ws[0].w, pre: /[*!]$/.test(s) }); }
    }
    return t;
  }
  // Precedence, tightest first: OR, w/N, AND (or a space), NOT.
  function parse(tokens) {
    var i = 0;
    function peek() { return tokens[i]; }
    function atom() {
      var t = tokens[i++];
      if (!t) throw 0;
      if (t.k === "(") { var e = not(); if (!tokens[i] || tokens[i].k !== ")") throw 0; i++; return e; }
      if (t.k === "word" || t.k === "phrase") return t;
      throw 0;
    }
    function or() { var l = atom(); while (peek() && peek().k === "or") { i++; l = { k: "OR", l: l, r: atom() }; } return l; }
    function near() { var l = or(); while (peek() && peek().k === "near") { var n = tokens[i++].n; l = { k: "NEAR", n: n, l: l, r: or() }; } return l; }
    function and() {
      var l = near();
      while (peek() && (peek().k === "and" || peek().k === "word" || peek().k === "phrase" || peek().k === "(")) { if (peek().k === "and") i++; l = { k: "AND", l: l, r: near() }; }
      return l;
    }
    function not() { var l = and(); while (peek() && peek().k === "not") { i++; l = { k: "NOT", l: l, r: and() }; } return l; }
    var e = not();
    if (i < tokens.length) throw 0;
    return e;
  }
  // Evaluate on one passage: the set of matching word positions, or null.
  function evalNode(n, ws) {
    var hit = [], j, k;
    if (n.k === "word") {
      for (j = 0; j < ws.length; j++) if (n.pre ? ws[j].w.indexOf(n.v) === 0 : ws[j].w === n.v) hit.push(j);
      return hit.length ? hit : null;
    }
    if (n.k === "phrase") {
      if (!n.v.length) return null;
      for (j = 0; j + n.v.length <= ws.length; j++) {
        for (k = 0; k < n.v.length && ws[j + k].w === n.v[k]; k++);
        if (k === n.v.length) for (k = 0; k < n.v.length; k++) hit.push(j + k);
      }
      return hit.length ? hit : null;
    }
    var a = evalNode(n.l, ws), b = evalNode(n.r, ws);
    if (n.k === "OR") return a || b ? (a || []).concat(b || []) : null;
    if (n.k === "AND") return a && b ? a.concat(b) : null;
    if (n.k === "NOT") return a && !b ? a : null;
    if (n.k === "NEAR") {
      if (!a || !b) return null;
      a.forEach(function (x) { b.forEach(function (y) { if (Math.abs(x - y) <= n.n) { hit.push(x); hit.push(y); } }); });
      return hit.length ? hit : null;
    }
    return null;
  }
  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function highlight(p, hits) {
    var set = {}, out = "", at = 0;
    hits.forEach(function (h) { set[h] = 1; });
    // Consecutive matching words share one highlight.
    for (var j = 0; j < p.words.length; j++) {
      if (!set[j]) continue;
      var k = j;
      while (set[k + 1]) k++;
      var a = p.words[j], b = p.words[k];
      out += esc(p.text.slice(at, a.i)) + "<mark>" + esc(p.text.slice(a.i, b.i + b.n)) + "</mark>";
      at = b.i + b.n;
      j = k;
    }
    return out + esc(p.text.slice(at));
  }

  function run(q, out, count) {
    q = q.trim();
    if (!q) { out.innerHTML = ""; count.textContent = ""; return; }
    var tree;
    try { tree = parse(tokenize(q)); } catch (e) { tree = null; }
    if (!tree) { count.textContent = "That search couldn’t be read. Check the brackets and quotation marks."; out.innerHTML = ""; return; }
    var docs = 0, passages = 0, html = "";
    DOCS.forEach(function (d) {
      var found = [];
      d.passages.forEach(function (p) { var h = evalNode(tree, p.words); if (h) found.push(highlight(p, h) + '<span class="demo-page">p ' + p.page + "</span>"); });
      if (!found.length) return;
      docs++; passages += found.length;
      html += '<article class="demo-doc"><header><b>' + d.type + '</b><span>' + d.parties + " &middot; " + d.state + " &middot; " + d.date + "</span></header>" +
        found.map(function (f) { return "<p>" + f + "</p>"; }).join("") + "</article>";
    });
    count.textContent = docs ? docs + (docs === 1 ? " document, " : " documents, ") + passages + (passages === 1 ? " passage" : " passages") : "No passages match. Try fewer terms, or OR between alternatives.";
    out.innerHTML = html;
  }

  var form = document.getElementById("demo-form");
  if (!form) return;
  var input = document.getElementById("demo-q"), out = document.getElementById("demo-results"), count = document.getElementById("demo-count");
  form.addEventListener("submit", function (e) { e.preventDefault(); run(input.value, out, count); });
  var timer;
  input.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(function () { run(input.value, out, count); }, 180); });
  document.querySelectorAll("[data-demo-q]").forEach(function (b) {
    b.addEventListener("click", function () { input.value = b.getAttribute("data-demo-q"); run(input.value, out, count); });
  });
  document.getElementById("demo-total").textContent = DOCS.length;
  input.value = '"payment schedule" w/15 (late OR "out of time")';
  run(input.value, out, count);
})();
